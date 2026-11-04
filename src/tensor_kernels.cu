// Tensor Core matmuls for long prefills (`mma.sync` bfloat16, f32
// accumulation). Compiled separately, for compute capability 8.0 and
// later, and only loaded on such a Device: the main Kernel source keeps
// building for every Device.
//
// `b` is a compact weight (its bfloat16 bits, `[k, n]` row-major), so it
// enters the Tensor Cores exactly. Each `a` value (f32) is split into two
// bfloat16 terms, hi = bf16(a) and lo = bf16(a - hi), and both are
// multiplied by `b`: the product keeps about 16 significant bits of `a`,
// close to f32, at twice the Tensor Core work. Accumulation is f32.
//
// Block tile: TC_BLOCK_M x TC_BLOCK_N outputs, 4 warps of 32 x 32, each
// warp 2 x 4 `m16n8k16` tiles. `a` and `b` are staged through shared
// memory TC_BLOCK_K inners at a time, `a` as hi/lo bfloat16 pairs and `b`
// transposed (`[n][k]`), so every fragment register is one 32-bit load.

#define TC_BLOCK_M 64
#define TC_BLOCK_N 64
#define TC_BLOCK_K 16
// Row pitch of the staged tiles, in bfloat16 elements (padded against bank
// conflicts).
#define TC_PITCH (TC_BLOCK_K + 8)
#define TC_THREADS 128

__device__ __forceinline__ unsigned short tc_float_to_bf16(float value) {
    unsigned int bits = (unsigned int)__float_as_int(value);
    if (((bits >> 23) & 0xFF) == 0xFF && (bits & 0x007FFFFF) != 0) {
        return (unsigned short)((bits >> 16) | 0x40);
    }
    unsigned int rounding_bias = 0x00007FFFu + ((bits >> 16) & 1u);
    return (unsigned short)((bits + rounding_bias) >> 16);
}

__device__ __forceinline__ float tc_bf16_to_float(unsigned short bits) {
    return __int_as_float((int)(((unsigned int)bits) << 16));
}

__device__ __forceinline__ void tc_mma(float c[4], const unsigned int a[4], unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Stages `a`'s tile (rows m0.., inners k0..) as hi/lo bfloat16.
__device__ __forceinline__ void tc_stage_a(
    const float* a, unsigned long long m, unsigned long long k, unsigned long long lda,
    unsigned long long m0, unsigned long long k0,
    unsigned short (*a_hi)[TC_PITCH], unsigned short (*a_lo)[TC_PITCH]
) {
    for (unsigned int e = threadIdx.x; e < TC_BLOCK_M * TC_BLOCK_K; e += TC_THREADS) {
        unsigned int r = e / TC_BLOCK_K;
        unsigned int c = e % TC_BLOCK_K;
        float value = (m0 + r < m && k0 + c < k) ? a[(m0 + r) * lda + k0 + c] : 0.0f;
        unsigned short hi = tc_float_to_bf16(value);
        a_hi[r][c] = hi;
        a_lo[r][c] = tc_float_to_bf16(value - tc_bf16_to_float(hi));
    }
}

// Stages a compact weight's tile (inners k0.., columns n0..), transposed.
__device__ __forceinline__ void tc_stage_b(
    const unsigned short* b16, unsigned long long k, unsigned long long n,
    unsigned long long k0, unsigned long long n0, unsigned short (*b_t)[TC_PITCH]
) {
    for (unsigned int e = threadIdx.x; e < TC_BLOCK_K * TC_BLOCK_N; e += TC_THREADS) {
        unsigned int kk = e / TC_BLOCK_N;
        unsigned int nn = e % TC_BLOCK_N;
        b_t[nn][kk] = (k0 + kk < k && n0 + nn < n) ? b16[(k0 + kk) * n + n0 + nn] : (unsigned short)0;
    }
}

// One warp's 2 x 4 tiles for the staged inners: hi and lo terms of `a`.
__device__ __forceinline__ void tc_warp_tiles(
    unsigned short (*a_hi)[TC_PITCH], unsigned short (*a_lo)[TC_PITCH],
    unsigned short (*b_t)[TC_PITCH], unsigned int warp_m, unsigned int warp_n,
    unsigned int group, unsigned int pair, float acc[2][4][4]
) {
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
        unsigned int r = warp_m + mt * 16 + group;
        unsigned int hi[4];
        unsigned int lo[4];
        hi[0] = *(const unsigned int*)&a_hi[r][2 * pair];
        hi[1] = *(const unsigned int*)&a_hi[r + 8][2 * pair];
        hi[2] = *(const unsigned int*)&a_hi[r][2 * pair + 8];
        hi[3] = *(const unsigned int*)&a_hi[r + 8][2 * pair + 8];
        lo[0] = *(const unsigned int*)&a_lo[r][2 * pair];
        lo[1] = *(const unsigned int*)&a_lo[r + 8][2 * pair];
        lo[2] = *(const unsigned int*)&a_lo[r][2 * pair + 8];
        lo[3] = *(const unsigned int*)&a_lo[r + 8][2 * pair + 8];
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            unsigned int col = warp_n + nt * 8 + group;
            unsigned int b0 = *(const unsigned int*)&b_t[col][2 * pair];
            unsigned int b1 = *(const unsigned int*)&b_t[col][2 * pair + 8];
            tc_mma(acc[mt][nt], lo, b0, b1);
            tc_mma(acc[mt][nt], hi, b0, b1);
        }
    }
}

// `a [m, k] x b16 [k, n]`, plus `addend[col]` (mode 1) or
// `addend[row * n + col]` (mode 2).
extern "C" __global__ void matmul_tc_kernel(
    const float* a,
    const unsigned short* b16,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    unsigned long long lda,
    const float* addend,
    int addend_mode
) {
    __shared__ __align__(16) unsigned short a_hi[TC_BLOCK_M][TC_PITCH];
    __shared__ __align__(16) unsigned short a_lo[TC_BLOCK_M][TC_PITCH];
    __shared__ __align__(16) unsigned short b_t[TC_BLOCK_N][TC_PITCH];
    unsigned long long m0 = (unsigned long long)blockIdx.y * TC_BLOCK_M;
    unsigned long long n0 = (unsigned long long)blockIdx.x * TC_BLOCK_N;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int lane = threadIdx.x & 31;
    unsigned int group = lane >> 2;
    unsigned int pair = lane & 3;
    unsigned int warp_m = (warp >> 1) * 32;
    unsigned int warp_n = (warp & 1) * 32;
    float acc[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[mt][nt][i] = 0.0f;

    for (unsigned long long k0 = 0; k0 < k; k0 += TC_BLOCK_K) {
        __syncthreads();
        tc_stage_a(a, m, k, lda, m0, k0, a_hi, a_lo);
        tc_stage_b(b16, k, n, k0, n0, b_t);
        __syncthreads();
        tc_warp_tiles(a_hi, a_lo, b_t, warp_m, warp_n, group, pair, acc);
    }

#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                unsigned long long row = m0 + warp_m + mt * 16 + group + (i >= 2 ? 8 : 0);
                unsigned long long col = n0 + warp_n + nt * 8 + 2 * pair + (i & 1);
                if (row < m && col < n) {
                    float value = acc[mt][nt][i];
                    if (addend_mode == 1) {
                        value += addend[col];
                    } else if (addend_mode == 2) {
                        value += addend[row * n + col];
                    }
                    out[row * n + col] = value;
                }
            }
        }
    }
}

// `silu(a x gate16) * (a x up16)`.
extern "C" __global__ void gated_matmul_tc_kernel(
    const float* a,
    const unsigned short* gate16,
    const unsigned short* up16,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    unsigned long long lda
) {
    __shared__ __align__(16) unsigned short a_hi[TC_BLOCK_M][TC_PITCH];
    __shared__ __align__(16) unsigned short a_lo[TC_BLOCK_M][TC_PITCH];
    __shared__ __align__(16) unsigned short gate_t[TC_BLOCK_N][TC_PITCH];
    __shared__ __align__(16) unsigned short up_t[TC_BLOCK_N][TC_PITCH];
    unsigned long long m0 = (unsigned long long)blockIdx.y * TC_BLOCK_M;
    unsigned long long n0 = (unsigned long long)blockIdx.x * TC_BLOCK_N;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int lane = threadIdx.x & 31;
    unsigned int group = lane >> 2;
    unsigned int pair = lane & 3;
    unsigned int warp_m = (warp >> 1) * 32;
    unsigned int warp_n = (warp & 1) * 32;
    float gate_acc[2][4][4];
    float up_acc[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                gate_acc[mt][nt][i] = 0.0f;
                up_acc[mt][nt][i] = 0.0f;
            }

    for (unsigned long long k0 = 0; k0 < k; k0 += TC_BLOCK_K) {
        __syncthreads();
        tc_stage_a(a, m, k, lda, m0, k0, a_hi, a_lo);
        tc_stage_b(gate16, k, n, k0, n0, gate_t);
        tc_stage_b(up16, k, n, k0, n0, up_t);
        __syncthreads();
        tc_warp_tiles(a_hi, a_lo, gate_t, warp_m, warp_n, group, pair, gate_acc);
        tc_warp_tiles(a_hi, a_lo, up_t, warp_m, warp_n, group, pair, up_acc);
    }

#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                unsigned long long row = m0 + warp_m + mt * 16 + group + (i >= 2 ? 8 : 0);
                unsigned long long col = n0 + warp_n + nt * 8 + 2 * pair + (i & 1);
                if (row < m && col < n) {
                    float gate = gate_acc[mt][nt][i];
                    float activated = gate * (1.0f / (1.0f + expf(-gate)));
                    out[row * n + col] = activated * up_acc[mt][nt][i];
                }
            }
        }
    }
}
