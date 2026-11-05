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
        if ((hi & 0x7FFF) == 0x7F80 && isfinite(value)) {
            // A finite value that rounds past bfloat16's largest finite
            // value: truncate instead (still finite), and let the low
            // term carry the remainder.
            hi = (unsigned short)(((unsigned int)__float_as_int(value)) >> 16);
        }
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

// ---------------------------------------------------------------------------
// Prefill attention on the Tensor Cores (head dimension 64)
// ---------------------------------------------------------------------------
//
// Block (query tile, head) serves TCA_QUERIES queries with 4 warps of 16.
// Keys are taken a tile at a time: S = Q K^T and the block's share of P V
// run as `m16n8k16` bfloat16 `mma` with f32 accumulation. Every f32 operand
// enters as two bfloat16 terms (hi, lo) and each product as hi*hi + hi*lo +
// lo*hi, which keeps about 16 significant bits. Softmax is online per query
// row; each key tile's probabilities are divided by the new running sum
// before they weight V, so the output stays a weighted mean of V rows.
// The scale 1/8 (exact for 64 dimensions) is folded into Q.

__device__ __forceinline__ float neg_inf() {
    return __int_as_float(0xff800000);
}

#define TCA_QUERIES 64
#define TCA_D 64
#define TCA_PITCH (TCA_D + 8)

__device__ __forceinline__ void tca_split(float value, unsigned short* hi, unsigned short* lo) {
    unsigned short h = tc_float_to_bf16(value);
    if ((h & 0x7FFF) == 0x7F80 && isfinite(value)) {
        h = (unsigned short)(((unsigned int)__float_as_int(value)) >> 16);
    }
    *hi = h;
    *lo = tc_float_to_bf16(value - tc_bf16_to_float(h));
}

__device__ __forceinline__ void tca_split_pair(float first, float second, unsigned int* hi, unsigned int* lo) {
    unsigned short h0, l0, h1, l1;
    tca_split(first, &h0, &l0);
    tca_split(second, &h1, &l1);
    *hi = (unsigned int)h0 | ((unsigned int)h1 << 16);
    *lo = (unsigned int)l0 | ((unsigned int)l1 << 16);
}

__device__ __forceinline__ unsigned int tca_u32(const unsigned short* p) {
    return *(const unsigned int*)p;
}

__device__ __forceinline__ void tca_keys_of(
    unsigned long long query_position, int causal, long long window_size,
    unsigned long long kv_seq_len, unsigned long long* lower, unsigned long long* upper
) {
    unsigned long long up = causal ? (query_position + 1) : kv_seq_len;
    unsigned long long low = 0;
    if (window_size >= 0) {
        unsigned long long window = (unsigned long long)window_size;
        unsigned long long back = window > 0 ? window - 1 : 0;
        low = query_position > back ? query_position - back : 0;
        if (low > up) {
            low = up;
        }
    }
    *lower = low;
    *upper = up;
}

// ---------------------------------------------------------------------------
// Pipelined Tensor Core matmuls
// ---------------------------------------------------------------------------
//
// `a` is first split, once per matmul, into its bfloat16 hi and lo terms
// (`tc_split_kernel`), so no block converts it again. Tiles of TCP_BLOCK_M x
// TCP_BLOCK_N outputs (8 warps of 32 x 32) then stream TCP_BLOCK_K inners at
// a time through two shared-memory stages filled by `cp.async`, so the next
// stage loads while this one computes. Fragments are read with `ldmatrix`
// (the weight with `.trans`, from its natural [k][n] layout). The products
// are those of `matmul_tc_kernel`: hi*b and lo*b per inner, f32 accumulation.
// Requires k and n to be multiples of 8 (16-byte rows) and a contiguous `a`.

#define TCP_BLOCK_M 128
#define TCP_BLOCK_N 64
#define TCP_BLOCK_K 16
#define TCP_A_PITCH (TCP_BLOCK_K + 8)
#define TCP_B_PITCH (TCP_BLOCK_N + 8)
#define TCP_THREADS 256

extern "C" __global__ void tc_split_kernel(
    const float* a,
    unsigned short* hi,
    unsigned short* lo,
    unsigned long long count
) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        float value = a[i];
        unsigned short h = tc_float_to_bf16(value);
        if ((h & 0x7FFF) == 0x7F80 && isfinite(value)) {
            h = (unsigned short)(((unsigned int)__float_as_int(value)) >> 16);
        }
        hi[i] = h;
        lo[i] = tc_float_to_bf16(value - tc_bf16_to_float(h));
    }
}

__device__ __forceinline__ unsigned int tcp_smem(const void* pointer) {
    unsigned int address;
    asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }\n"
                 : "=r"(address) : "l"(pointer));
    return address;
}

// 16 bytes from `source` into shared `destination`, or zeros when `valid` is
// false.
__device__ __forceinline__ void tcp_copy16(void* destination, const void* source, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(tcp_smem(destination)), "l"(source), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void tcp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void tcp_wait_all_but_one() {
    asm volatile("cp.async.wait_group 1;\n" ::);
}

__device__ __forceinline__ void tcp_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void tcp_ldmatrix_x4(unsigned int r[4], const void* pointer) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(tcp_smem(pointer)));
}

__device__ __forceinline__ void tcp_ldmatrix_x4_trans(unsigned int r[4], const void* pointer) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(tcp_smem(pointer)));
}

template <int NB>
__device__ __forceinline__ void tcp_load_stage(
    const unsigned short* a_hi, const unsigned short* a_lo, const unsigned short* const* b16,
    unsigned long long m, unsigned long long k, unsigned long long n,
    unsigned long long m0, unsigned long long n0, unsigned long long k0,
    unsigned short (*s_hi)[TCP_A_PITCH], unsigned short (*s_lo)[TCP_A_PITCH],
    unsigned short (*s_b)[TCP_BLOCK_K][TCP_B_PITCH]
) {
    unsigned int tid = threadIdx.x;
    // `a`: TCP_BLOCK_M rows of TCP_BLOCK_K inners, two 16-byte chunks per row.
    for (unsigned int c = tid; c < TCP_BLOCK_M * 2; c += TCP_THREADS) {
        unsigned int r = c >> 1;
        unsigned int col = (c & 1) * 8;
        bool valid = m0 + r < m && k0 + col < k;
        unsigned long long offset = valid ? (m0 + r) * k + k0 + col : 0;
        tcp_copy16(&s_hi[r][col], a_hi + offset, valid);
        tcp_copy16(&s_lo[r][col], a_lo + offset, valid);
    }
    // Each weight: TCP_BLOCK_K inners of TCP_BLOCK_N columns, 8 chunks per row.
#pragma unroll
    for (int w = 0; w < NB; w++) {
        for (unsigned int c = tid; c < TCP_BLOCK_K * (TCP_BLOCK_N / 8); c += TCP_THREADS) {
            unsigned int r = c / (TCP_BLOCK_N / 8);
            unsigned int col = (c % (TCP_BLOCK_N / 8)) * 8;
            bool valid = k0 + r < k && n0 + col < n;
            unsigned long long offset = valid ? (k0 + r) * n + n0 + col : 0;
            tcp_copy16(&s_b[w][r][col], b16[w] + offset, valid);
        }
    }
}

template <int NB>
__device__ __forceinline__ void tcp_matmul(
    const unsigned short* a_hi, const unsigned short* a_lo, const unsigned short* const* b16,
    unsigned long long m, unsigned long long k, unsigned long long n,
    float acc[NB][2][4][4]
) {
    __shared__ __align__(16) unsigned short s_hi[2][TCP_BLOCK_M][TCP_A_PITCH];
    __shared__ __align__(16) unsigned short s_lo[2][TCP_BLOCK_M][TCP_A_PITCH];
    __shared__ __align__(16) unsigned short s_b[2][NB][TCP_BLOCK_K][TCP_B_PITCH];
    unsigned long long m0 = (unsigned long long)blockIdx.y * TCP_BLOCK_M;
    unsigned long long n0 = (unsigned long long)blockIdx.x * TCP_BLOCK_N;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int lane = threadIdx.x & 31;
    unsigned int warp_m = (warp >> 1) * 32;
    unsigned int warp_n = (warp & 1) * 32;
#pragma unroll
    for (int w = 0; w < NB; w++)
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++) acc[w][mt][nt][i] = 0.0f;

    unsigned long long tiles = (k + TCP_BLOCK_K - 1) / TCP_BLOCK_K;
    tcp_load_stage<NB>(a_hi, a_lo, b16, m, k, n, m0, n0, 0, s_hi[0], s_lo[0], s_b[0]);
    tcp_commit();
    for (unsigned long long tile = 0; tile < tiles; tile++) {
        unsigned int stage = tile & 1;
        if (tile + 1 < tiles) {
            tcp_load_stage<NB>(a_hi, a_lo, b16, m, k, n, m0, n0, (tile + 1) * TCP_BLOCK_K,
                               s_hi[stage ^ 1], s_lo[stage ^ 1], s_b[stage ^ 1]);
            tcp_commit();
            tcp_wait_all_but_one();
        } else {
            tcp_wait_all();
        }
        __syncthreads();
        unsigned int hi[2][4];
        unsigned int lo[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            unsigned int row = warp_m + mt * 16 + (lane & 15);
            unsigned int col = (lane >> 4) * 8;
            tcp_ldmatrix_x4(hi[mt], &s_hi[stage][row][col]);
            tcp_ldmatrix_x4(lo[mt], &s_lo[stage][row][col]);
        }
#pragma unroll
        for (int w = 0; w < NB; w++) {
#pragma unroll
            for (int np = 0; np < 2; np++) {
                unsigned int b[4];
                unsigned int krow = (lane & 7) + ((lane >> 3) & 1) * 8;
                unsigned int ncol = warp_n + np * 16 + (lane >> 4) * 8;
                tcp_ldmatrix_x4_trans(b, &s_b[stage][w][krow][ncol]);
#pragma unroll
                for (int mt = 0; mt < 2; mt++) {
                    tc_mma(acc[w][mt][2 * np], lo[mt], b[0], b[1]);
                    tc_mma(acc[w][mt][2 * np], hi[mt], b[0], b[1]);
                    tc_mma(acc[w][mt][2 * np + 1], lo[mt], b[2], b[3]);
                    tc_mma(acc[w][mt][2 * np + 1], hi[mt], b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }
}

// `a x b16` (a given as its hi/lo bfloat16 terms), plus `addend[col]` (mode
// 1) or `addend[row * n + col]` (mode 2).
extern "C" __global__ void matmul_tcp_kernel(
    const unsigned short* a_hi,
    const unsigned short* a_lo,
    const unsigned short* b16,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    const float* addend,
    int addend_mode
) {
    const unsigned short* weights[1] = {b16};
    float acc[1][2][4][4];
    tcp_matmul<1>(a_hi, a_lo, weights, m, k, n, acc);
    unsigned long long m0 = (unsigned long long)blockIdx.y * TCP_BLOCK_M;
    unsigned long long n0 = (unsigned long long)blockIdx.x * TCP_BLOCK_N;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int lane = threadIdx.x & 31;
    unsigned int group = lane >> 2;
    unsigned int pair = lane & 3;
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                unsigned long long row = m0 + (warp >> 1) * 32 + mt * 16 + group + (i >= 2 ? 8 : 0);
                unsigned long long col = n0 + (warp & 1) * 32 + nt * 8 + 2 * pair + (i & 1);
                if (row < m && col < n) {
                    float value = acc[0][mt][nt][i];
                    if (addend_mode == 1) {
                        value += addend[col];
                    } else if (addend_mode == 2) {
                        value += addend[row * n + col];
                    }
                    out[row * n + col] = value;
                }
            }
}

// `silu(a x gate16) * (a x up16)`, `a` given as its hi/lo bfloat16 terms.
extern "C" __global__ void gated_matmul_tcp_kernel(
    const unsigned short* a_hi,
    const unsigned short* a_lo,
    const unsigned short* gate16,
    const unsigned short* up16,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n
) {
    const unsigned short* weights[2] = {gate16, up16};
    float acc[2][2][4][4];
    tcp_matmul<2>(a_hi, a_lo, weights, m, k, n, acc);
    unsigned long long m0 = (unsigned long long)blockIdx.y * TCP_BLOCK_M;
    unsigned long long n0 = (unsigned long long)blockIdx.x * TCP_BLOCK_N;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int lane = threadIdx.x & 31;
    unsigned int group = lane >> 2;
    unsigned int pair = lane & 3;
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                unsigned long long row = m0 + (warp >> 1) * 32 + mt * 16 + group + (i >= 2 ? 8 : 0);
                unsigned long long col = n0 + (warp & 1) * 32 + nt * 8 + 2 * pair + (i & 1);
                if (row < m && col < n) {
                    float gate = acc[0][mt][nt][i];
                    float activated = gate * (1.0f / (1.0f + expf(-gate)));
                    out[row * n + col] = activated * acc[1][mt][nt][i];
                }
            }
}

// ---------------------------------------------------------------------------
// Pipelined Tensor Core prefill attention (head dimension 64)
// ---------------------------------------------------------------------------
//
// K and V are given as their bfloat16 hi and lo terms, split once per
// attention by `tc_split_kernel` rather than once per query block. Key tiles of TCAP_KEYS stream through two
// shared-memory stages filled by `cp.async`; fragments are read with
// `ldmatrix` (V with `.trans`, from its natural [key][d] layout). Requires
// `kv_model_dim` to be a multiple of 8.

#define TCAP_KEYS 32

extern "C" __global__ void attention_tcp_kernel(
    const float* q,
    const unsigned short* k_hi,
    const unsigned short* k_lo,
    const unsigned short* v_hi,
    const unsigned short* v_lo,
    float* out,
    unsigned long long seq_len,
    unsigned long long kv_seq_len,
    unsigned long long head_count,
    unsigned long long kv_head_count,
    unsigned long long q_model_dim,
    unsigned long long kv_model_dim,
    int causal,
    long long window_size,
    unsigned long long query_position_offset
) {
    // Q is staged in the two K stages first (64 rows), then each key tile.
    __shared__ __align__(16) unsigned short s_khi[2 * TCAP_KEYS][TCA_PITCH];
    __shared__ __align__(16) unsigned short s_klo[2 * TCAP_KEYS][TCA_PITCH];
    __shared__ __align__(16) unsigned short s_vhi[2 * TCAP_KEYS][TCA_PITCH];
    __shared__ __align__(16) unsigned short s_vlo[2 * TCAP_KEYS][TCA_PITCH];
    unsigned int tid = threadIdx.x;
    unsigned int warp = tid >> 5;
    unsigned int lane = tid & 31;
    unsigned int g = lane >> 2;
    unsigned int t = lane & 3;
    unsigned long long head = blockIdx.y;
    unsigned long long tile_first = (unsigned long long)blockIdx.x * TCA_QUERIES;
    unsigned long long group_size = head_count / kv_head_count;
    unsigned long long q_offset = head * TCA_D;
    unsigned long long kv_offset = (head / group_size) * TCA_D;
    const float scale = 0.125f;

    for (unsigned int e = tid; e < TCA_QUERIES * TCA_D; e += 128) {
        unsigned int r = e / TCA_D;
        unsigned int c = e % TCA_D;
        unsigned long long qi = tile_first + r;
        float value = qi < seq_len ? q[qi * q_model_dim + q_offset + c] * scale : 0.0f;
        tca_split(value, &s_khi[r][c], &s_klo[r][c]);
    }
    __syncthreads();
    unsigned int qh[4][4];
    unsigned int ql[4][4];
    unsigned int wr = warp * 16 + g;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        unsigned int c0 = ks * 16 + 2 * t;
        qh[ks][0] = tca_u32(&s_khi[wr][c0]);
        qh[ks][1] = tca_u32(&s_khi[wr + 8][c0]);
        qh[ks][2] = tca_u32(&s_khi[wr][c0 + 8]);
        qh[ks][3] = tca_u32(&s_khi[wr + 8][c0 + 8]);
        ql[ks][0] = tca_u32(&s_klo[wr][c0]);
        ql[ks][1] = tca_u32(&s_klo[wr + 8][c0]);
        ql[ks][2] = tca_u32(&s_klo[wr][c0 + 8]);
        ql[ks][3] = tca_u32(&s_klo[wr + 8][c0 + 8]);
    }
    __syncthreads();

    unsigned long long last_query = tile_first + TCA_QUERIES - 1;
    if (last_query >= seq_len) {
        last_query = seq_len - 1;
    }
    unsigned long long row[2] = {tile_first + wr, tile_first + wr + 8};
    unsigned long long lower[2];
    unsigned long long upper[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
        unsigned long long position = query_position_offset + (row[h] < seq_len ? row[h] : last_query);
        tca_keys_of(position, causal, window_size, kv_seq_len, &lower[h], &upper[h]);
    }
    unsigned long long block_lower;
    unsigned long long block_upper;
    {
        unsigned long long unused;
        tca_keys_of(query_position_offset + tile_first, causal, window_size, kv_seq_len, &block_lower, &unused);
        tca_keys_of(query_position_offset + last_query, causal, window_size, kv_seq_len, &unused, &block_upper);
    }

    float max_value[2] = {neg_inf(), neg_inf()};
    float sum[2] = {0.0f, 0.0f};
    float o[8][4];
#pragma unroll
    for (int dt = 0; dt < 8; dt++)
#pragma unroll
        for (int i = 0; i < 4; i++) o[dt][i] = 0.0f;

    // One stage: TCAP_KEYS keys of 64 dimensions, 8 chunks of 16 bytes per
    // row, for each of K hi, K lo, V hi and V lo.
    auto load_stage = [&](unsigned long long tile, unsigned int stage) {
        for (unsigned int c = tid; c < TCAP_KEYS * 8; c += 128) {
            unsigned int r = c >> 3;
            unsigned int col = (c & 7) * 8;
            unsigned long long key = tile + r;
            bool valid = key < kv_seq_len;
            unsigned long long offset = valid ? key * kv_model_dim + kv_offset + col : 0;
            unsigned int sr = stage * TCAP_KEYS + r;
            tcp_copy16(&s_khi[sr][col], k_hi + offset, valid);
            tcp_copy16(&s_klo[sr][col], k_lo + offset, valid);
            tcp_copy16(&s_vhi[sr][col], v_hi + offset, valid);
            tcp_copy16(&s_vlo[sr][col], v_lo + offset, valid);
        }
    };

    if (block_lower < block_upper) {
        load_stage(block_lower, 0);
        tcp_commit();
    }
    unsigned int stage = 0;
    for (unsigned long long tile = block_lower; tile < block_upper; tile += TCAP_KEYS) {
        if (tile + TCAP_KEYS < block_upper) {
            load_stage(tile + TCAP_KEYS, stage ^ 1);
            tcp_commit();
            tcp_wait_all_but_one();
        } else {
            tcp_wait_all();
        }
        __syncthreads();
        unsigned int base = stage * TCAP_KEYS;

        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) s[nt][i] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
#pragma unroll
            for (int np = 0; np < 2; np++) {
                // Keys np*16 .. np*16+15: b0/b1 of n-tile 2np, then 2np+1.
                unsigned int kr = base + np * 16 + (lane & 7) + (lane >> 4) * 8;
                unsigned int kc = ks * 16 + ((lane >> 3) & 1) * 8;
                unsigned int bh[4];
                unsigned int bl[4];
                tcp_ldmatrix_x4(bh, &s_khi[kr][kc]);
                tcp_ldmatrix_x4(bl, &s_klo[kr][kc]);
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    int nt = 2 * np + half;
                    tc_mma(s[nt], ql[ks], bh[2 * half], bh[2 * half + 1]);
                    tc_mma(s[nt], qh[ks], bl[2 * half], bl[2 * half + 1]);
                    tc_mma(s[nt], qh[ks], bh[2 * half], bh[2 * half + 1]);
                }
            }
        }

        float tile_max[2] = {neg_inf(), neg_inf()};
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                int h = i >> 1;
                unsigned long long key = tile + nt * 8 + 2 * t + (i & 1);
                bool admitted = key >= lower[h] && key < upper[h] && key < kv_seq_len;
                s[nt][i] = admitted ? s[nt][i] : neg_inf();
                tile_max[h] = fmaxf(tile_max[h], s[nt][i]);
            }
        float keep[2];
        float take[2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            tile_max[h] = fmaxf(tile_max[h], __shfl_xor_sync(0xffffffffu, tile_max[h], 1));
            tile_max[h] = fmaxf(tile_max[h], __shfl_xor_sync(0xffffffffu, tile_max[h], 2));
        }
        float new_max[2];
        float tile_sum[2] = {0.0f, 0.0f};
#pragma unroll
        for (int h = 0; h < 2; h++) {
            new_max[h] = fmaxf(max_value[h], tile_max[h]);
        }
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                int h = i >> 1;
                float p = new_max[h] == neg_inf() ? 0.0f : expf(s[nt][i] - new_max[h]);
                s[nt][i] = p;
                tile_sum[h] += p;
            }
#pragma unroll
        for (int h = 0; h < 2; h++) {
            tile_sum[h] += __shfl_xor_sync(0xffffffffu, tile_sum[h], 1);
            tile_sum[h] += __shfl_xor_sync(0xffffffffu, tile_sum[h], 2);
            float kept = sum[h] > 0.0f ? sum[h] * expf(max_value[h] - new_max[h]) : 0.0f;
            float total = kept + tile_sum[h];
            keep[h] = total > 0.0f ? kept / total : 0.0f;
            take[h] = total > 0.0f ? 1.0f / total : 0.0f;
            sum[h] = total;
            max_value[h] = new_max[h];
        }
#pragma unroll
        for (int dt = 0; dt < 8; dt++)
#pragma unroll
            for (int i = 0; i < 4; i++) o[dt][i] *= keep[i >> 1];

#pragma unroll
        for (int j = 0; j < 2; j++) {
            unsigned int ph[4];
            unsigned int pl[4];
            tca_split_pair(s[2 * j][0] * take[0], s[2 * j][1] * take[0], &ph[0], &pl[0]);
            tca_split_pair(s[2 * j][2] * take[1], s[2 * j][3] * take[1], &ph[1], &pl[1]);
            tca_split_pair(s[2 * j + 1][0] * take[0], s[2 * j + 1][1] * take[0], &ph[2], &pl[2]);
            tca_split_pair(s[2 * j + 1][2] * take[1], s[2 * j + 1][3] * take[1], &ph[3], &pl[3]);
            unsigned int vr = base + j * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
            for (int dp = 0; dp < 4; dp++) {
                unsigned int vc = dp * 16 + (lane >> 4) * 8;
                unsigned int bh[4];
                unsigned int bl[4];
                tcp_ldmatrix_x4_trans(bh, &s_vhi[vr][vc]);
                tcp_ldmatrix_x4_trans(bl, &s_vlo[vr][vc]);
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    int dt = 2 * dp + half;
                    tc_mma(o[dt], pl, bh[2 * half], bh[2 * half + 1]);
                    tc_mma(o[dt], ph, bl[2 * half], bl[2 * half + 1]);
                    tc_mma(o[dt], ph, bh[2 * half], bh[2 * half + 1]);
                }
            }
        }
        __syncthreads();
        stage ^= 1;
    }

#pragma unroll
    for (int dt = 0; dt < 8; dt++)
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int h = i >> 1;
            if (row[h] < seq_len) {
                out[row[h] * q_model_dim + q_offset + dt * 8 + 2 * t + (i & 1)] =
                    sum[h] > 0.0f ? o[dt][i] : 0.0f;
            }
        }
}
