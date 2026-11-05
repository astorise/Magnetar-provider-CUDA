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
// Keys are taken TCA_KEYS at a time: S = Q K^T and the block's share of P V
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
#define TCA_KEYS 64
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

extern "C" __global__ void attention_tc_kernel(
    const float* q,
    const float* k,
    const float* v,
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
    // Q is staged in the key buffers first, then each key tile.
    __shared__ __align__(16) unsigned short s_hi[TCA_KEYS][TCA_PITCH];
    __shared__ __align__(16) unsigned short s_lo[TCA_KEYS][TCA_PITCH];
    __shared__ __align__(16) unsigned short vt_hi[TCA_D][TCA_PITCH];
    __shared__ __align__(16) unsigned short vt_lo[TCA_D][TCA_PITCH];
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
        tca_split(value, &s_hi[r][c], &s_lo[r][c]);
    }
    __syncthreads();
    unsigned int qh[4][4];
    unsigned int ql[4][4];
    unsigned int wr = warp * 16 + g;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        unsigned int c0 = ks * 16 + 2 * t;
        qh[ks][0] = tca_u32(&s_hi[wr][c0]);
        qh[ks][1] = tca_u32(&s_hi[wr + 8][c0]);
        qh[ks][2] = tca_u32(&s_hi[wr][c0 + 8]);
        qh[ks][3] = tca_u32(&s_hi[wr + 8][c0 + 8]);
        ql[ks][0] = tca_u32(&s_lo[wr][c0]);
        ql[ks][1] = tca_u32(&s_lo[wr + 8][c0]);
        ql[ks][2] = tca_u32(&s_lo[wr][c0 + 8]);
        ql[ks][3] = tca_u32(&s_lo[wr + 8][c0 + 8]);
    }

    // This thread's two query rows (g and g + 8 of its warp).
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

    for (unsigned long long tile = block_lower; tile < block_upper; tile += TCA_KEYS) {
        __syncthreads();
        for (unsigned int e = tid; e < TCA_KEYS * TCA_D; e += 128) {
            unsigned int r = e / TCA_D;
            unsigned int c = e % TCA_D;
            unsigned long long key = tile + r;
            bool present = key < kv_seq_len;
            float kv = present ? k[key * kv_model_dim + kv_offset + c] : 0.0f;
            float vv = present ? v[key * kv_model_dim + kv_offset + c] : 0.0f;
            tca_split(kv, &s_hi[r][c], &s_lo[r][c]);
            tca_split(vv, &vt_hi[c][r], &vt_lo[c][r]);
        }
        __syncthreads();

        float s[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) s[nt][i] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            unsigned int c0 = ks * 16 + 2 * t;
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                unsigned int kr = nt * 8 + g;
                unsigned int bh0 = tca_u32(&s_hi[kr][c0]);
                unsigned int bh1 = tca_u32(&s_hi[kr][c0 + 8]);
                unsigned int bl0 = tca_u32(&s_lo[kr][c0]);
                unsigned int bl1 = tca_u32(&s_lo[kr][c0 + 8]);
                tc_mma(s[nt], ql[ks], bh0, bh1);
                tc_mma(s[nt], qh[ks], bl0, bl1);
                tc_mma(s[nt], qh[ks], bh0, bh1);
            }
        }

        // Masks, then each row's online softmax update.
        float tile_max[2] = {neg_inf(), neg_inf()};
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
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
        for (int nt = 0; nt < 8; nt++)
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

        // P V, P normalized by the running sum: two key n-tiles form one
        // m16k16 A fragment.
#pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned int ph[4];
            unsigned int pl[4];
            tca_split_pair(s[2 * j][0] * take[0], s[2 * j][1] * take[0], &ph[0], &pl[0]);
            tca_split_pair(s[2 * j][2] * take[1], s[2 * j][3] * take[1], &ph[1], &pl[1]);
            tca_split_pair(s[2 * j + 1][0] * take[0], s[2 * j + 1][1] * take[0], &ph[2], &pl[2]);
            tca_split_pair(s[2 * j + 1][2] * take[1], s[2 * j + 1][3] * take[1], &ph[3], &pl[3]);
            unsigned int c0 = j * 16 + 2 * t;
#pragma unroll
            for (int dt = 0; dt < 8; dt++) {
                unsigned int dr = dt * 8 + g;
                unsigned int bh0 = tca_u32(&vt_hi[dr][c0]);
                unsigned int bh1 = tca_u32(&vt_hi[dr][c0 + 8]);
                unsigned int bl0 = tca_u32(&vt_lo[dr][c0]);
                unsigned int bl1 = tca_u32(&vt_lo[dr][c0 + 8]);
                tc_mma(o[dt], pl, bh0, bh1);
                tc_mma(o[dt], ph, bl0, bl1);
                tc_mma(o[dt], ph, bh0, bh1);
            }
        }
    }

    // No key admitted (an empty window) leaves the output zero.
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
