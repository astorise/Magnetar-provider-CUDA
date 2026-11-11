// CUDA Provider baseline kernels: correctness-first, unfused, f32-only,
// contiguous-only ports of providers/cpu's reference numeric kernels
// (matmul, embedding_lookup, rmsnorm, rope, attention, softmax_rows, silu,
// add, mul). One thread per output row/element/query throughout -- no block
// tiling, no shared memory, no fusion. See design.md's "NVRTC-compiled
// kernels are unfused, non-tuned CUDA C++" risk entry: this baseline
// optimizes for matching providers/cpu's semantics, not for speed.
//
// `extern "C"` on every kernel keeps its symbol name unmangled so the Rust
// side can load it by the plain name via CudaModule::load_function.
//
// NVRTC's minimal preprocessor environment does not predefine the
// `INFINITY` macro (unlike nvcc with a full host toolchain), even though it
// does accept `isfinite`/`fmaxf`/etc. as device builtins. `neg_inf()`
// reconstructs real IEEE-754 negative infinity from its bit pattern via the
// `__int_as_float` intrinsic -- needed for genuine equality with `isfinite`,
// not just a very-negative sentinel: a fully-masked row's real `-inf`
// entries must still reduce to a non-finite max for the "no finite entry"
// check below to match providers/cpu::softmax_rows exactly.
__device__ __forceinline__ float neg_inf() {
    return __int_as_float(0xff800000);
}

// A compact weight's value (`execute-native-half-precision`): bfloat16
// bits are the high half of the f32 they encode, so this is exact.
__device__ __forceinline__ float bf16_bits_to_float(unsigned short bits) {
    return __int_as_float((int)(((unsigned int)bits) << 16));
}

extern "C" __global__ void add_kernel(const float* a, const float* b, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

// a[i] += b[i]: a tensor-parallel reduction's in-place sum, which
// allocates nothing and so can be captured.
extern "C" __global__ void accumulate_kernel(float* a, const float* b, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] += b[i];
    }
}

// dst[i] = src[i], `src` possibly another GPU's memory (peer access
// enabled): a copy between GPUs that a step capture records as a Kernel.
extern "C" __global__ void copy_kernel(float* dst, const float* src, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}

extern "C" __global__ void mul_kernel(const float* a, const float* b, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i];
    }
}

// `silu_kernel` followed by `mul_kernel`, in one pass: the Runtime's fused
// `silu-mul` (SwiGLU). Same expression, same order, so bit-identical.
extern "C" __global__ void silu_mul_kernel(const float* gate, const float* up, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        float activated = v * (1.0f / (1.0f + expf(-v)));
        out[i] = activated * up[i];
    }
}

extern "C" __global__ void silu_kernel(const float* x, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i] = v * (1.0f / (1.0f + expf(-v)));
    }
}

// One thread per output row. `ids` are validated device-side (in-range,
// non-negative integers) rather than pre-checked on the host: `table` and
// `ids` may both already be device-resident outputs of a prior Kernel
// (`enable-device-resident-kernel-chaining`'s explicit-data-movement
// decision), so no host-visible copy of `ids` is assumed to exist here.
// An out-of-range or non-integer id sets `invalid_id_flag` and the row is
// left unwritten, matching `softmax_rows_kernel`'s device-computed-flag
// pattern below for a data-dependent failure the host must still observe.
extern "C" __global__ void embedding_lookup_kernel(
    const float* table,
    const float* ids,
    float* out,
    unsigned long long dim,
    unsigned long long num_ids,
    unsigned long long vocab,
    int* invalid_id_flag,
    // The table's bfloat16 bits when it is compact; `table` is then unused.
    const unsigned short* table16,
    // A packed Q8_0 table (`execute-direct-quantized-weights`), stored
    // untransposed: its quants and per-32-value scales; `table` is then
    // unused.
    const signed char* table_q8,
    const float* table_q8_scales
) {
    unsigned long long row = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_ids) {
        return;
    }
    float raw_id = ids[row];
    if (raw_id < 0.0f || raw_id != floorf(raw_id)) {
        atomicExch(invalid_id_flag, 1);
        return;
    }
    unsigned long long id = (unsigned long long)raw_id;
    if (id >= vocab) {
        atomicExch(invalid_id_flag, 1);
        return;
    }
    float* dst = out + row * dim;
    if (table_q8) {
        unsigned long long base = id * dim;
        for (unsigned long long i = 0; i < dim; i++) {
            dst[i] = (float)table_q8[base + i] * table_q8_scales[(base + i) >> 5];
        }
        return;
    }
    if (table16) {
        const unsigned short* src16 = table16 + id * dim;
        for (unsigned long long i = 0; i < dim; i++) {
            dst[i] = bf16_bits_to_float(src16[i]);
        }
        return;
    }
    const float* src = table + id * dim;
    for (unsigned long long i = 0; i < dim; i++) {
        dst[i] = src[i];
    }
}

// Packed Q8_0 weights (`execute-direct-quantized-weights`): signed 8-bit
// quants in stored row-major order, one f32 scale per 32 consecutive
// quants (a row is a whole number of blocks).

// One thread per stored value: the logical matrix as f32, the stored one
// or (`transposed`) its transpose.
extern "C" __global__ void q8_dequantize_kernel(
    const signed char* quants,
    const float* scales,
    float* out,
    unsigned long long stored_rows,
    unsigned long long stored_cols,
    unsigned int transposed
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= stored_rows * stored_cols) {
        return;
    }
    float value = (float)quants[idx] * scales[idx >> 5];
    if (transposed) {
        unsigned long long r = idx / stored_cols;
        unsigned long long c = idx % stored_cols;
        out[c * stored_rows + r] = value;
    } else {
        out[idx] = value;
    }
}

// The dot product of 16 consecutive quants (`q`, one 16-byte load: half a
// `Q8_0` block, so one scale applies) with 16 consecutive floats of `x`
// (16-byte aligned), summed in f32 in index order. Every decode-step
// `Q8_0` Kernel reads its weights through it: 16 bytes per lane per step
// keeps four times more weight bytes in flight than 4-byte loads, which
// left the RTX 3060 at about half its bandwidth.
__device__ __forceinline__ float q8_dot16(int4 q, const float* x) {
    const signed char* b = reinterpret_cast<const signed char*>(&q);
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; j += 4) {
        float4 v = *reinterpret_cast<const float4*>(x + j);
        s += v.x * (float)b[j] + v.y * (float)b[j + 1] + v.z * (float)b[j + 2]
            + v.w * (float)b[j + 3];
    }
    return s;
}

// Quants per lane per step of the decode-step `Q8_0` Kernels, and per warp.
#define Q8_LANE_QUANTS 16
#define Q8_WARP_QUANTS 512

// Rows of `a` one block of `q8_matmul_rows_kernel` serves.
#define Q8_MATMUL_ROWS 8
// Output columns (one warp each) per block of `q8_matmul_rows_kernel`.
#define Q8_MATMUL_WARPS 4

// out[r, c] = sum_k a[r, k] * W[k, c] (+ addend) for a contiguous `a` of
// `k` columns and a packed weight W stored transposed: stored row c holds
// column c's `k` quants. One warp per output column reads that row once,
// sixteen consecutive quants per lane per step (contiguous across the warp),
// for up to Q8_MATMUL_ROWS rows of `a`; each lane's four products are
// summed in f32 and scaled by their block's scale. The decode-step matmul:
// the weight is read packed, at about half the bytes of bfloat16.
extern "C" __global__ void q8_matmul_rows_kernel(
    const float* a,
    const signed char* quants,
    const float* scales,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    const float* addend,
    int addend_mode
) {
    unsigned long long col = (unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y;
    unsigned long long first_row = (unsigned long long)blockIdx.y * Q8_MATMUL_ROWS;
    if (col >= n || first_row >= m) {
        return;
    }
    unsigned int rows = (unsigned int)min((unsigned long long)Q8_MATMUL_ROWS, m - first_row);
    unsigned int lane = threadIdx.x;
    const signed char* w = quants + col * k;
    const float* w_scales = scales + col * (k >> 5);
    float acc[Q8_MATMUL_ROWS];
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        acc[r] = 0.0f;
    }
    // Each step covers 512 quants: sixteen blocks, two lanes per block.
    for (unsigned long long i = (unsigned long long)lane * Q8_LANE_QUANTS; i < k;
         i += Q8_WARP_QUANTS) {
        int4 q = *reinterpret_cast<const int4*>(w + i);
        float d = w_scales[i >> 5];
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                acc[r] += d * q8_dot16(q, a + (first_row + r) * k + i);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        float sum = acc[r];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0 && r < (int)rows) {
            unsigned long long row = first_row + r;
            if (addend_mode == 1) {
                sum += addend[col];
            } else if (addend_mode == 2) {
                sum += addend[row * n + col];
            }
            out[row * n + col] = sum;
        }
    }
}

// `silu(gate) * up` for a contiguous `a` and two packed weights stored
// transposed, as `q8_matmul_rows_kernel` computes each projection: one warp
// per output column reads both weights' rows, so the gated MLP's decode
// step writes one output and allocates no intermediate. The arithmetic is
// `q8_matmul_rows_kernel`'s per projection followed by `silu_mul_kernel`'s.
extern "C" __global__ void q8_gated_matmul_rows_kernel(
    const float* a,
    const signed char* gate_quants,
    const float* gate_scales,
    const signed char* up_quants,
    const float* up_scales,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n
) {
    unsigned long long col = (unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y;
    unsigned long long first_row = (unsigned long long)blockIdx.y * Q8_MATMUL_ROWS;
    if (col >= n || first_row >= m) {
        return;
    }
    unsigned int rows = (unsigned int)min((unsigned long long)Q8_MATMUL_ROWS, m - first_row);
    unsigned int lane = threadIdx.x;
    const signed char* wg = gate_quants + col * k;
    const signed char* wu = up_quants + col * k;
    const float* g_scales = gate_scales + col * (k >> 5);
    const float* u_scales = up_scales + col * (k >> 5);
    float gate_acc[Q8_MATMUL_ROWS];
    float up_acc[Q8_MATMUL_ROWS];
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        gate_acc[r] = 0.0f;
        up_acc[r] = 0.0f;
    }
    for (unsigned long long i = (unsigned long long)lane * Q8_LANE_QUANTS; i < k;
         i += Q8_WARP_QUANTS) {
        int4 qg = *reinterpret_cast<const int4*>(wg + i);
        int4 qu = *reinterpret_cast<const int4*>(wu + i);
        float dg = g_scales[i >> 5];
        float du = u_scales[i >> 5];
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                const float* x = a + (first_row + r) * k + i;
                gate_acc[r] += dg * q8_dot16(qg, x);
                up_acc[r] += du * q8_dot16(qu, x);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        float g = gate_acc[r];
        float u = up_acc[r];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            g += __shfl_xor_sync(0xffffffffu, g, offset);
            u += __shfl_xor_sync(0xffffffffu, u, offset);
        }
        if (lane == 0 && r < (int)rows) {
            float activated = g * (1.0f / (1.0f + expf(-g)));
            out[(first_row + r) * n + col] = activated * u;
        }
    }
}

// The most routed experts `moe_route_kernel` handles: 32 lanes of 8.
#define MOE_ROUTE_EXPERTS_PER_LANE 8

// `execute-and-cache-moe-experts` F02. One warp per token row: softmax
// of its `experts` router logits in f32, the `top_k` largest probabilities
// (ties to the lower expert), divided by their sum when `normalize` --
// `magnetar_runtime::reference_cpu::moe_route`'s order of operations.
// Writes `indices`/`weights[t * top_k + k]`; the routing never leaves the
// Device, so a decode step that routes stays capturable. Only experts
// `[expert_first, expert_first + expert_count)` are held here (a Device's
// share of an expert-split layer): `indices` holds each chosen expert's
// index among those, or -1 for an expert held elsewhere, whose pair the
// two kernels below skip.
extern "C" __global__ void moe_route_kernel(
    const float* logits,
    int* indices,
    float* weights,
    unsigned long long tokens,
    unsigned long long experts,
    unsigned long long top_k,
    int normalize,
    unsigned long long expert_first,
    unsigned long long expert_count
) {
    // One warp per token row: lane l owns experts l, l + 32, ... (at most
    // MOE_ROUTE_EXPERTS_PER_LANE of them).
    unsigned long long t = (unsigned long long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    unsigned int lane = threadIdx.x % 32;
    if (t >= tokens) {
        return;
    }
    const float* row = logits + t * experts;
    float max_logit = -__int_as_float(0x7f800000);
    for (unsigned long long e = lane; e < experts; e += 32) {
        max_logit = fmaxf(max_logit, row[e]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_logit = fmaxf(max_logit, __shfl_xor_sync(0xffffffffu, max_logit, offset));
    }
    float p[MOE_ROUTE_EXPERTS_PER_LANE];
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < MOE_ROUTE_EXPERTS_PER_LANE; j++) {
        unsigned long long e = lane + 32ull * j;
        p[j] = e < experts ? expf(row[e] - max_logit) : -1.0f;
        if (e < experts) {
            sum += p[j];
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    float chosen_total = 0.0f;
    for (unsigned long long k = 0; k < top_k; k++) {
        // This lane's best remaining expert (ties to the lower index), then
        // the warp's.
        float best_p = -1.0f;
        long long best = -1;
#pragma unroll
        for (int j = 0; j < MOE_ROUTE_EXPERTS_PER_LANE; j++) {
            if (p[j] > best_p) {
                best_p = p[j];
                best = (long long)(lane + 32ull * j);
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_p = __shfl_xor_sync(0xffffffffu, best_p, offset);
            long long other = __shfl_xor_sync(0xffffffffu, best, offset);
            if (other_p > best_p || (other_p == best_p && other >= 0 && (best < 0 || other < best))) {
                best_p = other_p;
                best = other;
            }
        }
        // The owning lane retires the chosen expert.
        if (best >= 0 && (unsigned long long)best % 32 == lane) {
            p[(unsigned long long)best / 32] = -1.0f;
        }
        if (lane == 0) {
            unsigned long long chosen = (unsigned long long)best;
            indices[t * top_k + k] = chosen >= expert_first && chosen - expert_first < expert_count
                ? (int)(chosen - expert_first)
                : -1;
            weights[t * top_k + k] = best_p / sum;
        }
        chosen_total += best_p / sum;
    }
    if (normalize && lane == 0) {
        for (unsigned long long k = 0; k < top_k; k++) {
            weights[t * top_k + k] /= chosen_total;
        }
    }
}

// act[tk, j] = silu(h[t] . gate_e[:, j]) * (h[t] . up_e[:, j]) for the
// routed pair tk = t * top_k + k and its expert e = indices[tk], with the
// stacked experts packed `Q8_0` in transposed storage: expert e's column j
// is stored row e * inter + j, `hidden` quants long. One warp per (tk, j),
// `q8_matmul_rows_kernel`'s reading pattern.
extern "C" __global__ void moe_q8_gated_kernel(
    const float* h,
    const int* indices,
    const signed char* gate_quants,
    const float* gate_scales,
    const signed char* up_quants,
    const float* up_scales,
    float* act,
    unsigned long long top_k,
    unsigned long long hidden,
    unsigned long long inter
) {
    unsigned long long j = (unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y;
    unsigned long long tk = blockIdx.y;
    if (j >= inter) {
        return;
    }
    if (indices[tk] < 0) {
        return;
    }
    unsigned long long t = tk / top_k;
    unsigned long long row = (unsigned long long)indices[tk] * inter + j;
    const signed char* wg = gate_quants + row * hidden;
    const signed char* wu = up_quants + row * hidden;
    const float* sg = gate_scales + row * (hidden >> 5);
    const float* su = up_scales + row * (hidden >> 5);
    const float* x = h + t * hidden;
    float g = 0.0f;
    float u = 0.0f;
    for (unsigned long long i = (unsigned long long)threadIdx.x * Q8_LANE_QUANTS; i < hidden;
         i += Q8_WARP_QUANTS) {
        int4 qg = *reinterpret_cast<const int4*>(wg + i);
        int4 qu = *reinterpret_cast<const int4*>(wu + i);
        g += sg[i >> 5] * q8_dot16(qg, x + i);
        u += su[i >> 5] * q8_dot16(qu, x + i);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        g += __shfl_xor_sync(0xffffffffu, g, offset);
        u += __shfl_xor_sync(0xffffffffu, u, offset);
    }
    if (threadIdx.x == 0) {
        act[tk * inter + j] = g * (1.0f / (1.0f + expf(-g))) * u;
    }
}

// Chosen experts `moe_q8_down_kernel` reads at once.
#define MOE_DOWN_EXPERTS 4

// out[t, c] = sum_k weights[tk] * (act[tk] . down_e[:, c]) with the stacked
// down projections packed `Q8_0` in transposed storage: expert e's column c
// is stored row e * hidden + c, `inter` quants long. One warp per (t, c);
// the experts add in routing order.
extern "C" __global__ void moe_q8_down_kernel(
    const float* act,
    const int* indices,
    const float* weights,
    const signed char* down_quants,
    const float* down_scales,
    float* out,
    unsigned long long top_k,
    unsigned long long hidden,
    unsigned long long inter
) {
    unsigned long long c = (unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y;
    unsigned long long t = blockIdx.y;
    if (c >= hidden) {
        return;
    }
    // Up to MOE_DOWN_EXPERTS chosen experts read at once (their loads in
    // flight together); each expert's sum, and the experts' total, in the
    // same order as one at a time.
    float total = 0.0f;
    for (unsigned long long k0 = 0; k0 < top_k; k0 += MOE_DOWN_EXPERTS) {
        const signed char* w[MOE_DOWN_EXPERTS];
        const float* scales[MOE_DOWN_EXPERTS];
        const float* a[MOE_DOWN_EXPERTS];
        float sum[MOE_DOWN_EXPERTS];
        bool held[MOE_DOWN_EXPERTS];
#pragma unroll
        for (int k = 0; k < MOE_DOWN_EXPERTS; k++) {
            unsigned long long tk = t * top_k + k0 + k;
            held[k] = k0 + k < top_k && indices[tk] >= 0;
            unsigned long long row = held[k] ? (unsigned long long)indices[tk] * hidden + c : 0;
            w[k] = down_quants + row * inter;
            scales[k] = down_scales + row * (inter >> 5);
            a[k] = act + (held[k] ? tk : 0) * inter;
            sum[k] = 0.0f;
        }
        for (unsigned long long i = (unsigned long long)threadIdx.x * Q8_LANE_QUANTS; i < inter;
             i += Q8_WARP_QUANTS) {
#pragma unroll
            for (int k = 0; k < MOE_DOWN_EXPERTS; k++) {
                if (held[k]) {
                    int4 q = *reinterpret_cast<const int4*>(w[k] + i);
                    sum[k] += scales[k][i >> 5] * q8_dot16(q, a[k] + i);
                }
            }
        }
#pragma unroll
        for (int k = 0; k < MOE_DOWN_EXPERTS; k++) {
            float value = sum[k];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_xor_sync(0xffffffffu, value, offset);
            }
            if (held[k]) {
                total += weights[t * top_k + k0 + k] * value;
            }
        }
    }
    if (threadIdx.x == 0) {
        out[t * hidden + c] = total;
    }
}

// The f32 forms of the two kernels above, for experts stored as plain f32
// logical matrices: `gate`/`up` `[hidden, experts * inter]` and `down`
// `[inter, experts * hidden]`, row-major. One thread per output element.
extern "C" __global__ void moe_f32_gated_kernel(
    const float* h,
    const int* indices,
    const float* gate,
    const float* up,
    float* act,
    unsigned long long pairs,
    unsigned long long top_k,
    unsigned long long hidden,
    unsigned long long inter,
    unsigned long long experts
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pairs * inter) {
        return;
    }
    unsigned long long tk = index / inter;
    unsigned long long j = index % inter;
    if (indices[tk] < 0) {
        return;
    }
    unsigned long long column = (unsigned long long)indices[tk] * inter + j;
    unsigned long long stride = experts * inter;
    const float* x = h + (tk / top_k) * hidden;
    float g = 0.0f;
    float u = 0.0f;
    for (unsigned long long i = 0; i < hidden; i++) {
        g += x[i] * gate[i * stride + column];
        u += x[i] * up[i * stride + column];
    }
    act[index] = g * (1.0f / (1.0f + expf(-g))) * u;
}

extern "C" __global__ void moe_f32_down_kernel(
    const float* act,
    const int* indices,
    const float* weights,
    const float* down,
    float* out,
    unsigned long long tokens,
    unsigned long long top_k,
    unsigned long long hidden,
    unsigned long long inter,
    unsigned long long experts
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tokens * hidden) {
        return;
    }
    unsigned long long t = index / hidden;
    unsigned long long c = index % hidden;
    unsigned long long stride = experts * hidden;
    float total = 0.0f;
    for (unsigned long long k = 0; k < top_k; k++) {
        unsigned long long tk = t * top_k + k;
        if (indices[tk] < 0) {
            continue;
        }
        unsigned long long column = (unsigned long long)indices[tk] * hidden + c;
        const float* a = act + tk * inter;
        float sum = 0.0f;
        for (unsigned long long j = 0; j < inter; j++) {
            sum += a[j] * down[j * stride + column];
        }
        total += weights[tk] * sum;
    }
    out[index] = total;
}

// out[i] = x[i] * scale[i / cols]: each row scaled by its own scalar.
extern "C" __global__ void row_scale_kernel(
    const float* x,
    const float* scale,
    float* out,
    unsigned long long cols,
    unsigned long long n
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] * scale[i / cols];
    }
}

extern "C" __global__ void sigmoid_kernel(const float* x, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

// One thread per output element. `bias` is a single row of `cols` values,
// broadcast across every row of `a` (real Qwen2/2.5 QKV projection bias:
// `[rows, cols] + [cols] -> [rows, cols]`) -- mirrors `rmsnorm_kernel`'s
// broadcast-row convention below, and `providers/cpu::add`'s own broadcast
// case.
extern "C" __global__ void bias_add_kernel(const float* a, const float* bias, float* out, unsigned long long cols, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + bias[i % cols];
    }
}

// One block of RMSNORM_THREADS threads per row. `weight_row_stride` is 0
// for a single broadcast weight row, or `cols` for a per-row weight --
// mirrors providers/cpu::rmsnorm's `row_weight_stride` exactly. Each thread
// sums a strided slice of the row's squares, then a fixed-order tree sum
// combines the slices, so the result is deterministic.
#define RMSNORM_THREADS 256
extern "C" __global__ void rmsnorm_kernel(
    const float* input,
    const float* weight,
    float* out,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long weight_row_stride,
    float epsilon
) {
    __shared__ float partial[RMSNORM_THREADS];
    unsigned long long row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float* in_row = input + row * cols;
    const float* w_row = weight + row * weight_row_stride;
    float sum = 0.0f;
    for (unsigned long long i = threadIdx.x; i < cols; i += RMSNORM_THREADS) {
        sum += in_row[i] * in_row[i];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = RMSNORM_THREADS / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float scale = 1.0f / sqrtf(partial[0] / (float)cols + epsilon);
    float* out_row = out + row * cols;
    for (unsigned long long i = threadIdx.x; i < cols; i += RMSNORM_THREADS) {
        out_row[i] = in_row[i] * scale * w_row[i];
    }
}

// One thread per (row, head, pair). `half = dimension / 2`, `head_width =
// cols / head_count`. `out` is pre-seeded (device-to-device copy of
// `input`, done Rust-side) so columns this kernel never touches -- a
// partial-RoPE block's untouched tail -- keep their original values.
// `half_split` selects the element pairing (`fix-rope-rotation-pairing`):
// 0 rotates (2i, 2i+1), 1 rotates (i, i + half) as Hugging Face
// `rotate_half` does.
extern "C" __global__ void rope_kernel(
    const float* input,
    float* out,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long half,
    float base,
    float scale,
    unsigned long long dimension,
    unsigned long long position_offset,
    unsigned long long head_count,
    unsigned int half_split,
    // A captured step's replay offset (`capture-and-replay-cuda-graphs`),
    // added to `position_offset`; null outside a capture.
    const unsigned long long* step_offset,
    // Banded per-frequency scaling (`frequency_scaling = "banded"`);
    // `banded_factor == 0` leaves the frequencies unscaled. Same f32
    // operations, in the same order, as `RopeBandedFrequencyScaling::scale_frequency`.
    float banded_factor,
    float low_freq_factor,
    float high_freq_factor,
    float original_context_length
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long per_row = head_count * half;
    unsigned long long total = rows * per_row;
    if (idx >= total) {
        return;
    }
    unsigned long long head_width = cols / head_count;
    unsigned long long row = idx / per_row;
    unsigned long long rem = idx % per_row;
    unsigned long long head = rem / half;
    unsigned long long pair = rem % half;
    if (step_offset) {
        position_offset += *step_offset;
    }
    float position = (float)(position_offset + row) * scale;
    float frequency = powf(base, -2.0f * (float)pair / (float)dimension);
    if (banded_factor > 0.0f) {
        float wavelength = 6.283185307179586f / frequency;
        if (wavelength < original_context_length / high_freq_factor) {
            // High frequencies are kept.
        } else if (wavelength > original_context_length / low_freq_factor) {
            frequency = frequency / banded_factor;
        } else {
            float smooth = (original_context_length / wavelength - low_freq_factor)
                / (high_freq_factor - low_freq_factor);
            frequency = (1.0f - smooth) * frequency / banded_factor + smooth * frequency;
        }
    }
    float angle = position * frequency;
    float s = sinf(angle);
    float c = cosf(angle);
    unsigned long long col_base = row * cols + head * head_width;
    unsigned long long first = half_split ? col_base + pair : col_base + 2 * pair;
    unsigned long long second = half_split ? col_base + pair + half : col_base + 2 * pair + 1;
    float a = input[first];
    float b = input[second];
    out[first] = a * c - b * s;
    out[second] = a * s + b * c;
}

// One thread per row.
extern "C" __global__ void softmax_rows_kernel(
    const float* input,
    float* out,
    unsigned long long rows,
    unsigned long long cols,
    int* had_non_finite_max
) {
    unsigned long long row = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float* in_row = input + row * cols;
    float max_value = neg_inf();
    for (unsigned long long i = 0; i < cols; i++) {
        max_value = fmaxf(max_value, in_row[i]);
    }
    if (!isfinite(max_value)) {
        atomicExch(had_non_finite_max, 1);
        return;
    }
    float sum = 0.0f;
    for (unsigned long long i = 0; i < cols; i++) {
        sum += expf(in_row[i] - max_value);
    }
    float* out_row = out + row * cols;
    for (unsigned long long i = 0; i < cols; i++) {
        out_row[i] = expf(in_row[i] - max_value) / sum;
    }
}

// One thread per output element (row, col). Accumulates over `k` in the same
// ascending order providers/cpu::matmul uses, so floating-point rounding
// matches as closely as two independent implementations can.
extern "C" __global__ void matmul_kernel(
    const float* a,
    const float* b,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    unsigned long long a_row_stride,
    unsigned long long a_inner_stride,
    unsigned long long b_inner_stride,
    unsigned long long b_col_stride,
    // `b`'s bfloat16 bits when it is compact; `b` is then unused.
    const unsigned short* b16
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = m * n;
    if (idx >= total) {
        return;
    }
    unsigned long long row = idx / n;
    unsigned long long col = idx % n;
    float acc = 0.0f;
    for (unsigned long long inner = 0; inner < k; inner++) {
        float a_value = a[row * a_row_stride + inner * a_inner_stride];
        unsigned long long b_index = inner * b_inner_stride + col * b_col_stride;
        float b_value = b16 ? bf16_bits_to_float(b16[b_index]) : b[b_index];
        acc += a_value * b_value;
    }
    out[row * n + col] = acc;
}

// Split-K matmul for a `b` whose columns are contiguous (not transposed):
// one block per (row, 32-column tile), MATMUL_SPLIT_K warps each summing an
// interleaved slice of `k`, then a fixed-order sum of the slices, so the
// result is deterministic. Every warp reads 32 consecutive columns of `b`
// at a time (coalesced), and a decode step's one-row projections get
// MATMUL_SPLIT_K times more threads than `matmul_kernel` gives them.
// `addend_mode` 1 adds `addend[col]` (a bias row) and 2 adds
// `addend[row * n + col]` (a residual) to the finished sum: the same
// operations as this kernel followed by `bias_add_kernel` or `add_kernel`
// (float addition of two operands is commutative), so a fused
// `matmul-bias` or `matmul-residual` is bit-identical to them.
#define MATMUL_SPLIT_K 16
// Loads of one slice issued before their products are added.
#define MATMUL_UNROLL 8
__device__ __forceinline__ float matmul_weight(const float* b, unsigned long long i) {
    return b[i];
}
__device__ __forceinline__ float matmul_weight(const unsigned short* b16, unsigned long long i) {
    return bf16_bits_to_float(b16[i]);
}
// One split-K slice's sum: `a[inner] * b[inner]` for inner = slice,
// slice + MATMUL_SPLIT_K, ... < k, added in that order. The loads of
// MATMUL_UNROLL consecutive terms are all issued before the first add, so
// each warp keeps that many weight loads in flight instead of one; the
// adds, and so the result, are those of the one-term-at-a-time loop.
template <typename W>
__device__ __forceinline__ float matmul_slice_sum(
    const float* a_row,
    unsigned long long a_inner_stride,
    const W* b_col,
    unsigned long long b_inner_stride,
    unsigned int slice,
    unsigned long long k
) {
    float acc = 0.0f;
    unsigned long long inner = slice;
    const unsigned long long step = MATMUL_SPLIT_K;
    for (; inner + (MATMUL_UNROLL - 1) * step < k; inner += MATMUL_UNROLL * step) {
        float av[MATMUL_UNROLL];
        float bv[MATMUL_UNROLL];
#pragma unroll
        for (int u = 0; u < MATMUL_UNROLL; u++) {
            av[u] = a_row[(inner + u * step) * a_inner_stride];
            bv[u] = matmul_weight(b_col, (inner + u * step) * b_inner_stride);
        }
#pragma unroll
        for (int u = 0; u < MATMUL_UNROLL; u++) {
            acc += av[u] * bv[u];
        }
    }
    for (; inner < k; inner += step) {
        acc += a_row[inner * a_inner_stride] * matmul_weight(b_col, inner * b_inner_stride);
    }
    return acc;
}
extern "C" __global__ void matmul_split_k_kernel(
    const float* a,
    const float* b,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    unsigned long long a_row_stride,
    unsigned long long a_inner_stride,
    unsigned long long b_inner_stride,
    const float* addend,
    int addend_mode,
    // `b`'s bfloat16 bits when it is compact; `b` is then unused.
    const unsigned short* b16
) {
    __shared__ float partial[MATMUL_SPLIT_K][32];
    unsigned long long row = blockIdx.y;
    unsigned long long col = (unsigned long long)blockIdx.x * 32 + threadIdx.x;
    unsigned int slice = threadIdx.y;
    float acc = 0.0f;
    if (row < m && col < n) {
        const float* a_row = a + row * a_row_stride;
        acc = b16
            ? matmul_slice_sum(a_row, a_inner_stride, b16 + col, b_inner_stride, slice, k)
            : matmul_slice_sum(a_row, a_inner_stride, b + col, b_inner_stride, slice, k);
    }
    partial[slice][threadIdx.x] = acc;
    __syncthreads();
    if (slice == 0 && row < m && col < n) {
        float sum = 0.0f;
        for (unsigned int s = 0; s < MATMUL_SPLIT_K; s++) {
            sum += partial[s][threadIdx.x];
        }
        if (addend_mode == 1) {
            sum += addend[col];
        } else if (addend_mode == 2) {
            sum += addend[row * n + col];
        }
        out[row * n + col] = sum;
    }
}

// Rows of `a` one block of the multi-row split-K kernels serves: each
// weight they load is used for every one of these rows (prefill), instead
// of being loaded once per row.
#define MATMUL_ROWS 8
// Weight loads a multi-row slice issues before adding, fewer than
// MATMUL_UNROLL to keep MATMUL_ROWS rows of `a` values in registers.
#define MATMUL_ROWS_UNROLL 4
// `matmul_slice_sum` for the `rows` (at most MATMUL_ROWS) consecutive rows
// of `a` from `a_rows`: each weight is loaded once for all of them, and
// each row's products are added in exactly `matmul_slice_sum`'s order
// (inner ascending), so each row's sum is bit-identical to it.
template <typename W>
__device__ __forceinline__ void matmul_slice_sums(
    const float* a_rows,
    unsigned long long a_row_stride,
    unsigned long long a_inner_stride,
    unsigned int rows,
    const W* b_col,
    unsigned long long b_inner_stride,
    unsigned int slice,
    unsigned long long k,
    float acc[MATMUL_ROWS]
) {
#pragma unroll
    for (int r = 0; r < MATMUL_ROWS; r++) {
        acc[r] = 0.0f;
    }
    unsigned long long inner = slice;
    const unsigned long long step = MATMUL_SPLIT_K;
    for (; inner + (MATMUL_ROWS_UNROLL - 1) * step < k; inner += MATMUL_ROWS_UNROLL * step) {
        float bv[MATMUL_ROWS_UNROLL];
        float av[MATMUL_ROWS][MATMUL_ROWS_UNROLL];
#pragma unroll
        for (int u = 0; u < MATMUL_ROWS_UNROLL; u++) {
            bv[u] = matmul_weight(b_col, (inner + u * step) * b_inner_stride);
        }
#pragma unroll
        for (int r = 0; r < MATMUL_ROWS; r++) {
#pragma unroll
            for (int u = 0; u < MATMUL_ROWS_UNROLL; u++) {
                av[r][u] = r < rows
                    ? a_rows[r * a_row_stride + (inner + u * step) * a_inner_stride]
                    : 0.0f;
            }
        }
#pragma unroll
        for (int r = 0; r < MATMUL_ROWS; r++) {
#pragma unroll
            for (int u = 0; u < MATMUL_ROWS_UNROLL; u++) {
                acc[r] += av[r][u] * bv[u];
            }
        }
    }
    for (; inner < k; inner += step) {
        float w = matmul_weight(b_col, inner * b_inner_stride);
#pragma unroll
        for (int r = 0; r < MATMUL_ROWS; r++) {
            float a_value = r < rows ? a_rows[r * a_row_stride + inner * a_inner_stride] : 0.0f;
            acc[r] += a_value * w;
        }
    }
}

// `matmul_split_k_kernel` for up to MATMUL_ROWS rows per block (blockIdx.y
// selects the row group): the same per-row sums, slice order and addend,
// so each output is bit-identical to it, with each weight read once per
// row group instead of once per row.
extern "C" __global__ void matmul_split_k_rows_kernel(
    const float* a,
    const float* b,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    unsigned long long a_row_stride,
    unsigned long long a_inner_stride,
    unsigned long long b_inner_stride,
    const float* addend,
    int addend_mode,
    const unsigned short* b16
) {
    __shared__ float partial[MATMUL_ROWS][MATMUL_SPLIT_K][32];
    unsigned long long first_row = (unsigned long long)blockIdx.y * MATMUL_ROWS;
    unsigned int rows = (unsigned int)min((unsigned long long)MATMUL_ROWS, m - first_row);
    unsigned long long col = (unsigned long long)blockIdx.x * 32 + threadIdx.x;
    unsigned int slice = threadIdx.y;
    float acc[MATMUL_ROWS];
    const float* a_rows = a + first_row * a_row_stride;
    if (col < n) {
        if (b16) {
            matmul_slice_sums(a_rows, a_row_stride, a_inner_stride, rows, b16 + col, b_inner_stride, slice, k, acc);
        } else {
            matmul_slice_sums(a_rows, a_row_stride, a_inner_stride, rows, b + col, b_inner_stride, slice, k, acc);
        }
    } else {
#pragma unroll
        for (int r = 0; r < MATMUL_ROWS; r++) {
            acc[r] = 0.0f;
        }
    }
#pragma unroll
    for (int r = 0; r < MATMUL_ROWS; r++) {
        partial[r][slice][threadIdx.x] = acc[r];
    }
    __syncthreads();
    // Slice-0 threads finish rows 0, 8, ...; every slice finishes the row
    // it is numbered after, so the MATMUL_ROWS rows finish in parallel.
    if (slice < rows && col < n) {
        unsigned long long row = first_row + slice;
        float sum = 0.0f;
        for (unsigned int s = 0; s < MATMUL_SPLIT_K; s++) {
            sum += partial[slice][s][threadIdx.x];
        }
        if (addend_mode == 1) {
            sum += addend[col];
        } else if (addend_mode == 2) {
            sum += addend[row * n + col];
        }
        out[row * n + col] = sum;
    }
}

// `gated_matmul_split_k_kernel` for up to MATMUL_ROWS rows per block,
// bit-identical to it row by row.
extern "C" __global__ void gated_matmul_split_k_rows_kernel(
    const float* a,
    const float* gate,
    const unsigned short* gate16,
    const float* up,
    const unsigned short* up16,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    unsigned long long a_row_stride,
    unsigned long long a_inner_stride,
    unsigned long long b_inner_stride
) {
    __shared__ float gate_partial[MATMUL_ROWS][MATMUL_SPLIT_K][32];
    __shared__ float up_partial[MATMUL_ROWS][MATMUL_SPLIT_K][32];
    unsigned long long first_row = (unsigned long long)blockIdx.y * MATMUL_ROWS;
    unsigned int rows = (unsigned int)min((unsigned long long)MATMUL_ROWS, m - first_row);
    unsigned long long col = (unsigned long long)blockIdx.x * 32 + threadIdx.x;
    unsigned int slice = threadIdx.y;
    float gate_acc[MATMUL_ROWS];
    float up_acc[MATMUL_ROWS];
    const float* a_rows = a + first_row * a_row_stride;
    if (col < n) {
        if (gate16) {
            matmul_slice_sums(a_rows, a_row_stride, a_inner_stride, rows, gate16 + col, b_inner_stride, slice, k, gate_acc);
        } else {
            matmul_slice_sums(a_rows, a_row_stride, a_inner_stride, rows, gate + col, b_inner_stride, slice, k, gate_acc);
        }
        if (up16) {
            matmul_slice_sums(a_rows, a_row_stride, a_inner_stride, rows, up16 + col, b_inner_stride, slice, k, up_acc);
        } else {
            matmul_slice_sums(a_rows, a_row_stride, a_inner_stride, rows, up + col, b_inner_stride, slice, k, up_acc);
        }
    } else {
#pragma unroll
        for (int r = 0; r < MATMUL_ROWS; r++) {
            gate_acc[r] = 0.0f;
            up_acc[r] = 0.0f;
        }
    }
#pragma unroll
    for (int r = 0; r < MATMUL_ROWS; r++) {
        gate_partial[r][slice][threadIdx.x] = gate_acc[r];
        up_partial[r][slice][threadIdx.x] = up_acc[r];
    }
    __syncthreads();
    if (slice < rows && col < n) {
        unsigned long long row = first_row + slice;
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (unsigned int s = 0; s < MATMUL_SPLIT_K; s++) {
            gate_sum += gate_partial[slice][s][threadIdx.x];
        }
        for (unsigned int s = 0; s < MATMUL_SPLIT_K; s++) {
            up_sum += up_partial[slice][s][threadIdx.x];
        }
        float activated = gate_sum * (1.0f / (1.0f + expf(-gate_sum)));
        out[row * n + col] = activated * up_sum;
    }
}

// Prefill matmul tiles: each thread computes MATMUL_ROWS rows times
// MATMUL_TILE_COLUMNS columns (lane, lane + 32, ...) of one split-K slice,
// so each weight it loads serves MATMUL_ROWS rows and each `a` value
// MATMUL_TILE_COLUMNS columns. The block's `a` rows are staged through
// shared memory MATMUL_TILE_INNER inners at a time (a multiple of
// MATMUL_SPLIT_K, so every slice keeps its own inners). Each output's slice
// sums still add inner ascending, and the slices are added in slice order,
// so every output is bit-identical to `matmul_split_k_kernel`'s.
#define MATMUL_TILE_COLUMNS 4
#define MATMUL_TILE_INNER 256
template <typename W>
__device__ __forceinline__ void matmul_tile_sums(
    const float* a_rows,
    unsigned long long a_row_stride,
    unsigned long long a_inner_stride,
    unsigned int rows,
    const W* b,
    unsigned long long b_inner_stride,
    unsigned long long first_col,
    unsigned long long n,
    unsigned int slice,
    unsigned long long k,
    float (*a_shared)[MATMUL_TILE_INNER],
    float acc[MATMUL_ROWS][MATMUL_TILE_COLUMNS]
) {
    unsigned int tid = threadIdx.y * 32 + threadIdx.x;
    unsigned int threads = 32 * MATMUL_SPLIT_K;
#pragma unroll
    for (int r = 0; r < MATMUL_ROWS; r++) {
#pragma unroll
        for (int c = 0; c < MATMUL_TILE_COLUMNS; c++) {
            acc[r][c] = 0.0f;
        }
    }
    for (unsigned long long base = 0; base < k; base += MATMUL_TILE_INNER) {
        __syncthreads();
        for (unsigned int e = tid; e < MATMUL_ROWS * MATMUL_TILE_INNER; e += threads) {
            unsigned int r = e / MATMUL_TILE_INNER;
            unsigned int i = e % MATMUL_TILE_INNER;
            a_shared[r][i] = r < rows && base + i < k
                ? a_rows[r * a_row_stride + (base + i) * a_inner_stride]
                : 0.0f;
        }
        __syncthreads();
        unsigned long long end = base + MATMUL_TILE_INNER < k ? base + MATMUL_TILE_INNER : k;
        for (unsigned long long inner = base + slice; inner < end; inner += MATMUL_SPLIT_K) {
            float w[MATMUL_TILE_COLUMNS];
#pragma unroll
            for (int c = 0; c < MATMUL_TILE_COLUMNS; c++) {
                unsigned long long col = first_col + c * 32;
                w[c] = col < n ? matmul_weight(b, inner * b_inner_stride + col) : 0.0f;
            }
#pragma unroll
            for (int r = 0; r < MATMUL_ROWS; r++) {
                float a_value = a_shared[r][inner - base];
#pragma unroll
                for (int c = 0; c < MATMUL_TILE_COLUMNS; c++) {
                    acc[r][c] += a_value * w[c];
                }
            }
        }
    }
}

extern "C" __global__ void matmul_split_k_tile_kernel(
    const float* a,
    const float* b,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    unsigned long long a_row_stride,
    unsigned long long a_inner_stride,
    unsigned long long b_inner_stride,
    const float* addend,
    int addend_mode,
    const unsigned short* b16
) {
    __shared__ float a_shared[MATMUL_ROWS][MATMUL_TILE_INNER];
    __shared__ float partial[MATMUL_ROWS][MATMUL_SPLIT_K][32];
    unsigned long long first_row = (unsigned long long)blockIdx.y * MATMUL_ROWS;
    unsigned int rows = (unsigned int)min((unsigned long long)MATMUL_ROWS, m - first_row);
    unsigned long long first_col = (unsigned long long)blockIdx.x * 32 * MATMUL_TILE_COLUMNS + threadIdx.x;
    unsigned int slice = threadIdx.y;
    float acc[MATMUL_ROWS][MATMUL_TILE_COLUMNS];
    const float* a_rows = a + first_row * a_row_stride;
    if (b16) {
        matmul_tile_sums(a_rows, a_row_stride, a_inner_stride, rows, b16, b_inner_stride, first_col, n, slice, k, a_shared, acc);
    } else {
        matmul_tile_sums(a_rows, a_row_stride, a_inner_stride, rows, b, b_inner_stride, first_col, n, slice, k, a_shared, acc);
    }
#pragma unroll
    for (int c = 0; c < MATMUL_TILE_COLUMNS; c++) {
        __syncthreads();
#pragma unroll
        for (int r = 0; r < MATMUL_ROWS; r++) {
            partial[r][slice][threadIdx.x] = acc[r][c];
        }
        __syncthreads();
        unsigned long long col = first_col + c * 32;
        if (slice < rows && col < n) {
            unsigned long long row = first_row + slice;
            float sum = 0.0f;
            for (unsigned int s = 0; s < MATMUL_SPLIT_K; s++) {
                sum += partial[slice][s][threadIdx.x];
            }
            if (addend_mode == 1) {
                sum += addend[col];
            } else if (addend_mode == 2) {
                sum += addend[row * n + col];
            }
            out[row * n + col] = sum;
        }
    }
}

// `gated_matmul_split_k_rows_kernel` in tiles, bit-identical to it.
extern "C" __global__ void gated_matmul_split_k_tile_kernel(
    const float* a,
    const float* gate,
    const unsigned short* gate16,
    const float* up,
    const unsigned short* up16,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    unsigned long long a_row_stride,
    unsigned long long a_inner_stride,
    unsigned long long b_inner_stride
) {
    __shared__ float a_shared[MATMUL_ROWS][MATMUL_TILE_INNER];
    __shared__ float gate_partial[MATMUL_ROWS][MATMUL_SPLIT_K][32];
    __shared__ float up_partial[MATMUL_ROWS][MATMUL_SPLIT_K][32];
    unsigned long long first_row = (unsigned long long)blockIdx.y * MATMUL_ROWS;
    unsigned int rows = (unsigned int)min((unsigned long long)MATMUL_ROWS, m - first_row);
    unsigned long long first_col = (unsigned long long)blockIdx.x * 32 * MATMUL_TILE_COLUMNS + threadIdx.x;
    unsigned int slice = threadIdx.y;
    float gate_acc[MATMUL_ROWS][MATMUL_TILE_COLUMNS];
    float up_acc[MATMUL_ROWS][MATMUL_TILE_COLUMNS];
    const float* a_rows = a + first_row * a_row_stride;
    if (gate16) {
        matmul_tile_sums(a_rows, a_row_stride, a_inner_stride, rows, gate16, b_inner_stride, first_col, n, slice, k, a_shared, gate_acc);
    } else {
        matmul_tile_sums(a_rows, a_row_stride, a_inner_stride, rows, gate, b_inner_stride, first_col, n, slice, k, a_shared, gate_acc);
    }
    if (up16) {
        matmul_tile_sums(a_rows, a_row_stride, a_inner_stride, rows, up16, b_inner_stride, first_col, n, slice, k, a_shared, up_acc);
    } else {
        matmul_tile_sums(a_rows, a_row_stride, a_inner_stride, rows, up, b_inner_stride, first_col, n, slice, k, a_shared, up_acc);
    }
#pragma unroll
    for (int c = 0; c < MATMUL_TILE_COLUMNS; c++) {
        __syncthreads();
#pragma unroll
        for (int r = 0; r < MATMUL_ROWS; r++) {
            gate_partial[r][slice][threadIdx.x] = gate_acc[r][c];
            up_partial[r][slice][threadIdx.x] = up_acc[r][c];
        }
        __syncthreads();
        unsigned long long col = first_col + c * 32;
        if (slice < rows && col < n) {
            unsigned long long row = first_row + slice;
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            for (unsigned int s = 0; s < MATMUL_SPLIT_K; s++) {
                gate_sum += gate_partial[slice][s][threadIdx.x];
            }
            for (unsigned int s = 0; s < MATMUL_SPLIT_K; s++) {
                up_sum += up_partial[slice][s][threadIdx.x];
            }
            float activated = gate_sum * (1.0f / (1.0f + expf(-gate_sum)));
            out[row * n + col] = activated * up_sum;
        }
    }
}

// `silu(a x gate) * (a x up)` for the Runtime's fused `gated-matmul`: the
// split-K matmul's sums for both weights, in its order, then
// `silu_mul_kernel`'s arithmetic, so the result is bit-identical to two
// `matmul_split_k_kernel` launches and a `silu_mul_kernel` one. Reading
// both weights in one launch gives each block twice the work.
extern "C" __global__ void gated_matmul_split_k_kernel(
    const float* a,
    const float* gate,
    const unsigned short* gate16,
    const float* up,
    const unsigned short* up16,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    unsigned long long a_row_stride,
    unsigned long long a_inner_stride,
    unsigned long long b_inner_stride
) {
    __shared__ float gate_partial[MATMUL_SPLIT_K][32];
    __shared__ float up_partial[MATMUL_SPLIT_K][32];
    unsigned long long row = blockIdx.y;
    unsigned long long col = (unsigned long long)blockIdx.x * 32 + threadIdx.x;
    unsigned int slice = threadIdx.y;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    if (row < m && col < n) {
        const float* a_row = a + row * a_row_stride;
        gate_acc = gate16
            ? matmul_slice_sum(a_row, a_inner_stride, gate16 + col, b_inner_stride, slice, k)
            : matmul_slice_sum(a_row, a_inner_stride, gate + col, b_inner_stride, slice, k);
        up_acc = up16
            ? matmul_slice_sum(a_row, a_inner_stride, up16 + col, b_inner_stride, slice, k)
            : matmul_slice_sum(a_row, a_inner_stride, up + col, b_inner_stride, slice, k);
    }
    gate_partial[slice][threadIdx.x] = gate_acc;
    up_partial[slice][threadIdx.x] = up_acc;
    __syncthreads();
    if (slice == 0 && row < m && col < n) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (unsigned int s = 0; s < MATMUL_SPLIT_K; s++) {
            gate_sum += gate_partial[s][threadIdx.x];
        }
        for (unsigned int s = 0; s < MATMUL_SPLIT_K; s++) {
            up_sum += up_partial[s][threadIdx.x];
        }
        float activated = gate_sum * (1.0f / (1.0f + expf(-gate_sum)));
        out[row * n + col] = activated * up_sum;
    }
}

// Greedy next-token selection (`keep-sampling-on-device`): the column of
// one logits row with the highest score divided by `temperature` (as is
// when 0), among columns outside the `range_count` [start, end) pairs in
// `ranges` whose raw score is finite, ties going to the lowest column, as
// the Runtime's host greedy selection picks it. Each eligible column is
// one 64-bit key: the score's bits, mapped so unsigned order is float
// order (with -0 as +0, which compare equal), above the column's
// complement; the largest key is the answer whatever order threads and
// blocks run in. `best` must start at 0, which no eligible column yields.
extern "C" __global__ void greedy_select_kernel(
    const float* logits,
    unsigned long long cols,
    const unsigned long long* ranges,
    unsigned int range_count,
    float temperature,
    unsigned long long* best
) {
    unsigned long long key = 0;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < cols;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        float value = logits[i];
        if (!isfinite(value)) {
            continue;
        }
        bool excluded = false;
        for (unsigned int r = 0; r < range_count; r++) {
            if (i >= ranges[2 * r] && i < ranges[2 * r + 1]) {
                excluded = true;
                break;
            }
        }
        if (excluded) {
            continue;
        }
        if (temperature != 0.0f) {
            value = value / temperature;
        }
        if (value == 0.0f) {
            value = 0.0f;
        }
        unsigned int bits = __float_as_uint(value);
        unsigned int ordered = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        unsigned long long candidate =
            ((unsigned long long)ordered << 32) | (0xFFFFFFFFull - (i & 0xFFFFFFFFull));
        if (candidate > key) {
            key = candidate;
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        unsigned long long other = __shfl_down_sync(0xFFFFFFFFu, key, offset);
        if (other > key) {
            key = other;
        }
    }
    if ((threadIdx.x & 31) == 0 && key != 0) {
        atomicMax(best, key);
    }
}

// One thread per (head, query_index). Recomputes each query/key dot product
// up to three times (max pass, sum-of-exp pass, output pass) instead of
// materializing a per-thread scores buffer -- correctness-first, matching
// providers/cpu::attention's masking/GQA/sliding-window semantics exactly,
// not an optimized flash-attention-style single pass.
extern "C" __global__ void attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    unsigned long long seq_len,
    unsigned long long kv_seq_len,
    unsigned long long head_count,
    unsigned long long kv_head_count,
    unsigned long long head_dimension,
    unsigned long long q_model_dim,
    unsigned long long kv_model_dim,
    int causal,
    long long window_size, // -1 means "no window"
    unsigned long long query_position_offset
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = head_count * seq_len;
    if (idx >= total) {
        return;
    }
    unsigned long long head = idx / seq_len;
    unsigned long long query_index = idx % seq_len;
    unsigned long long group_size = head_count / kv_head_count;
    unsigned long long kv_head = head / group_size;
    unsigned long long q_offset = head * head_dimension;
    unsigned long long kv_offset = kv_head * head_dimension;
    float scale = 1.0f / sqrtf((float)head_dimension);

    unsigned long long query_position = query_position_offset + query_index;
    unsigned long long key_upper = causal ? (query_position + 1) : kv_seq_len;
    unsigned long long key_lower = 0;
    if (window_size >= 0) {
        unsigned long long window = (unsigned long long)window_size;
        unsigned long long back = window > 0 ? window - 1 : 0;
        key_lower = query_position > back ? query_position - back : 0;
        if (key_lower > key_upper) {
            key_lower = key_upper;
        }
    }

    const float* q_row = q + query_index * q_model_dim + q_offset;

    float max_value = neg_inf();
    for (unsigned long long key_index = key_lower; key_index < key_upper; key_index++) {
        const float* k_row = k + key_index * kv_model_dim + kv_offset;
        float dot = 0.0f;
        for (unsigned long long d = 0; d < head_dimension; d++) {
            dot += q_row[d] * k_row[d];
        }
        max_value = fmaxf(max_value, dot * scale);
    }

    float sum = 0.0f;
    for (unsigned long long key_index = key_lower; key_index < key_upper; key_index++) {
        const float* k_row = k + key_index * kv_model_dim + kv_offset;
        float dot = 0.0f;
        for (unsigned long long d = 0; d < head_dimension; d++) {
            dot += q_row[d] * k_row[d];
        }
        sum += expf(dot * scale - max_value);
    }

    float* out_row = out + query_index * q_model_dim + q_offset;
    for (unsigned long long d = 0; d < head_dimension; d++) {
        out_row[d] = 0.0f;
    }
    for (unsigned long long key_index = key_lower; key_index < key_upper; key_index++) {
        const float* k_row = k + key_index * kv_model_dim + kv_offset;
        float dot = 0.0f;
        for (unsigned long long d = 0; d < head_dimension; d++) {
            dot += q_row[d] * k_row[d];
        }
        float weight = expf(dot * scale - max_value) / sum;
        const float* v_row = v + key_index * kv_model_dim + kv_offset;
        for (unsigned long long d = 0; d < head_dimension; d++) {
            out_row[d] += weight * v_row[d];
        }
    }
}

// One block of ATTENTION_THREADS threads per (head, query): the same
// masking, scaling and softmax as `attention_kernel`, with each q.k score
// computed once (keys spread over the threads) and kept in dynamic shared
// memory, the softmax max and sum combined by fixed-order tree reductions
// (deterministic), and each output dimension accumulated by its own thread
// over coalesced V reads. Shared memory holds `head_dimension` floats of
// q, ATTENTION_THREADS reduction slots and `kv_seq_len` scores; callers
// fall back to `attention_kernel` when that does not fit.
#define ATTENTION_THREADS 128
extern "C" __global__ void attention_block_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    unsigned long long seq_len,
    unsigned long long kv_seq_len,
    unsigned long long head_count,
    unsigned long long kv_head_count,
    unsigned long long head_dimension,
    unsigned long long q_model_dim,
    unsigned long long kv_model_dim,
    int causal,
    long long window_size, // -1 means "no window"
    unsigned long long query_position_offset,
    // A captured step's replay offset, added to the key count and the
    // query position; null outside a capture.
    const unsigned long long* step_offset
) {
    extern __shared__ float shared[];
    if (step_offset) {
        kv_seq_len += *step_offset;
        query_position_offset += *step_offset;
    }
    float* q_shared = shared;
    float* reduce = shared + head_dimension;
    float* scores = reduce + ATTENTION_THREADS;
    unsigned long long idx = blockIdx.x;
    if (idx >= head_count * seq_len) {
        return;
    }
    unsigned int tid = threadIdx.x;
    unsigned long long head = idx / seq_len;
    unsigned long long query_index = idx % seq_len;
    unsigned long long group_size = head_count / kv_head_count;
    unsigned long long kv_head = head / group_size;
    unsigned long long q_offset = head * head_dimension;
    unsigned long long kv_offset = kv_head * head_dimension;
    float scale = 1.0f / sqrtf((float)head_dimension);

    unsigned long long query_position = query_position_offset + query_index;
    unsigned long long key_upper = causal ? (query_position + 1) : kv_seq_len;
    unsigned long long key_lower = 0;
    if (window_size >= 0) {
        unsigned long long window = (unsigned long long)window_size;
        unsigned long long back = window > 0 ? window - 1 : 0;
        key_lower = query_position > back ? query_position - back : 0;
        if (key_lower > key_upper) {
            key_lower = key_upper;
        }
    }
    unsigned long long key_count = key_upper - key_lower;

    const float* q_row = q + query_index * q_model_dim + q_offset;
    for (unsigned long long d = tid; d < head_dimension; d += ATTENTION_THREADS) {
        q_shared[d] = q_row[d];
    }
    __syncthreads();

    float local_max = neg_inf();
    for (unsigned long long j = tid; j < key_count; j += ATTENTION_THREADS) {
        const float* k_row = k + (key_lower + j) * kv_model_dim + kv_offset;
        float dot = 0.0f;
        for (unsigned long long d = 0; d < head_dimension; d++) {
            dot += q_shared[d] * k_row[d];
        }
        float score = dot * scale;
        scores[j] = score;
        local_max = fmaxf(local_max, score);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (unsigned int stride = ATTENTION_THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
        }
        __syncthreads();
    }
    float max_value = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (unsigned long long j = tid; j < key_count; j += ATTENTION_THREADS) {
        float weight = expf(scores[j] - max_value);
        scores[j] = weight;
        local_sum += weight;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (unsigned int stride = ATTENTION_THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            reduce[tid] += reduce[tid + stride];
        }
        __syncthreads();
    }
    float sum = reduce[0];
    // Normalized before weighting V, as `attention_kernel` does: summing
    // unnormalized weights times large finite V values could overflow
    // where the normalized result is representable.
    for (unsigned long long j = tid; j < key_count; j += ATTENTION_THREADS) {
        scores[j] = scores[j] / sum;
    }
    __syncthreads();

    // No key admitted (an empty window) leaves every output zero, as
    // `attention_kernel`.
    float* out_row = out + query_index * q_model_dim + q_offset;
    for (unsigned long long d = tid; d < head_dimension; d += ATTENTION_THREADS) {
        float acc = 0.0f;
        for (unsigned long long j = 0; j < key_count; j++) {
            acc += scores[j] * v[(key_lower + j) * kv_model_dim + kv_offset + d];
        }
        out_row[d] = acc;
    }
}

// Prefill attention, tiled: block (query tile, head) serves
// ATTENTION_TILE_QUERIES queries of one head with 4 threads each (thread
// `part` of a query owns head dimensions part, part + 4, ...: PART of them,
// a compile-time count so a query's q and output stay in registers). K and
// V rows are staged through shared memory one tile of `key_tile` keys at a
// time, so every tile is read once per query tile. Keys are taken
// ATTENTION_SUB_KEYS at a time: their scores, their max and their
// normalized weighted mean of V are computed together, then merged into the
// query's running max, sum and mean of V, each weighted by its share of the
// softmax mass. The mean never leaves the V rows' range (large finite values
// stay finite). A query's 4 partial dot products are added with a fixed
// butterfly and its keys are visited in order: deterministic.
#define ATTENTION_TILE_QUERIES 64
#define ATTENTION_SUB_KEYS 16
template <int PART>
__device__ __forceinline__ void attention_tiled(
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
    unsigned long long query_position_offset,
    unsigned int key_tile
) {
    extern __shared__ float tiles[];
    const unsigned int head_dimension = 4 * PART;
    float* k_tile = tiles;
    float* v_tile = tiles + key_tile * head_dimension;
    unsigned int tid = threadIdx.x;
    unsigned int part = tid & 3;
    unsigned long long head = blockIdx.y;
    unsigned long long tile_first = (unsigned long long)blockIdx.x * ATTENTION_TILE_QUERIES;
    unsigned long long query_index = tile_first + (tid >> 2);
    bool active = query_index < seq_len;
    unsigned long long group_size = head_count / kv_head_count;
    unsigned long long q_offset = head * head_dimension;
    unsigned long long kv_offset = (head / group_size) * head_dimension;
    float scale = 1.0f / sqrtf((float)head_dimension);

    unsigned long long last_query = tile_first + ATTENTION_TILE_QUERIES - 1;
    if (last_query >= seq_len) {
        last_query = seq_len - 1;
    }
    unsigned long long tile_upper = causal ? (query_position_offset + last_query + 1) : kv_seq_len;
    unsigned long long tile_lower = 0;
    unsigned long long key_lower = 0;
    unsigned long long query_position = query_position_offset + (active ? query_index : last_query);
    unsigned long long key_upper = causal ? (query_position + 1) : kv_seq_len;
    if (window_size >= 0) {
        unsigned long long window = (unsigned long long)window_size;
        unsigned long long back = window > 0 ? window - 1 : 0;
        key_lower = query_position > back ? query_position - back : 0;
        if (key_lower > key_upper) {
            key_lower = key_upper;
        }
        unsigned long long first_position = query_position_offset + tile_first;
        tile_lower = first_position > back ? first_position - back : 0;
    }

    float q_part[PART];
    float acc[PART];
#pragma unroll
    for (int i = 0; i < PART; i++) {
        q_part[i] = active ? q[query_index * q_model_dim + q_offset + part + 4 * i] * scale : 0.0f;
        acc[i] = 0.0f;
    }
    float max_value = neg_inf();
    float sum = 0.0f;

    for (unsigned long long tile = tile_lower; tile < tile_upper; tile += key_tile) {
        unsigned int tile_keys = (unsigned int)(tile_upper - tile < key_tile ? tile_upper - tile : key_tile);
        __syncthreads();
        for (unsigned int row = tid / head_dimension; row < tile_keys; row += blockDim.x / head_dimension) {
            unsigned int d = tid % head_dimension;
            const unsigned long long source = (tile + row) * kv_model_dim + kv_offset + d;
            k_tile[row * head_dimension + d] = k[source];
            v_tile[row * head_dimension + d] = v[source];
        }
        __syncthreads();
        for (unsigned int sub = 0; sub < tile_keys; sub += ATTENTION_SUB_KEYS) {
            float scores[ATTENTION_SUB_KEYS];
            float sub_max = neg_inf();
#pragma unroll
            for (int j = 0; j < ATTENTION_SUB_KEYS; j++) {
                unsigned int row = sub + j;
                float dot = 0.0f;
                if (row < tile_keys) {
                    const float* k_row = k_tile + row * head_dimension + part;
#pragma unroll
                    for (int i = 0; i < PART; i++) {
                        dot += q_part[i] * k_row[4 * i];
                    }
                }
                dot += __shfl_xor_sync(0xffffffffu, dot, 1);
                dot += __shfl_xor_sync(0xffffffffu, dot, 2);
                unsigned long long key = tile + row;
                bool admitted = active && row < tile_keys && key >= key_lower && key < key_upper;
                scores[j] = admitted ? dot : neg_inf();
                sub_max = fmaxf(sub_max, scores[j]);
            }
            if (sub_max == neg_inf()) {
                continue;
            }
            float sub_sum = 0.0f;
#pragma unroll
            for (int j = 0; j < ATTENTION_SUB_KEYS; j++) {
                scores[j] = expf(scores[j] - sub_max);
                sub_sum += scores[j];
            }
            float sub_mean[PART];
#pragma unroll
            for (int i = 0; i < PART; i++) {
                sub_mean[i] = 0.0f;
            }
#pragma unroll
            for (int j = 0; j < ATTENTION_SUB_KEYS; j++) {
                float weight = scores[j] / sub_sum;
                if (weight != 0.0f) {
                    const float* v_row = v_tile + (sub + j) * head_dimension + part;
#pragma unroll
                    for (int i = 0; i < PART; i++) {
                        sub_mean[i] += weight * v_row[4 * i];
                    }
                }
            }
            float new_max = fmaxf(max_value, sub_max);
            float kept = sum > 0.0f ? sum * expf(max_value - new_max) : 0.0f;
            float added = sub_sum * expf(sub_max - new_max);
            sum = kept + added;
            float keep = kept / sum;
            float take = added / sum;
#pragma unroll
            for (int i = 0; i < PART; i++) {
                acc[i] = acc[i] * keep + sub_mean[i] * take;
            }
            max_value = new_max;
        }
    }
    if (active) {
        // No key admitted (an empty window) leaves the output zero.
#pragma unroll
        for (int i = 0; i < PART; i++) {
            out[query_index * q_model_dim + q_offset + part + 4 * i] = sum > 0.0f ? acc[i] : 0.0f;
        }
    }
}

#define ATTENTION_TILED_KERNEL(PART)                                              \
extern "C" __global__ void attention_tiled_##PART##_kernel(                       \
    const float* q, const float* k, const float* v, float* out,                   \
    unsigned long long seq_len, unsigned long long kv_seq_len,                    \
    unsigned long long head_count, unsigned long long kv_head_count,              \
    unsigned long long q_model_dim, unsigned long long kv_model_dim,              \
    int causal, long long window_size,                                            \
    unsigned long long query_position_offset, unsigned int key_tile) {            \
    attention_tiled<PART>(q, k, v, out, seq_len, kv_seq_len, head_count,          \
        kv_head_count, q_model_dim, kv_model_dim, causal, window_size,            \
        query_position_offset, key_tile);                                         \
}
ATTENTION_TILED_KERNEL(1)
ATTENTION_TILED_KERNEL(2)
ATTENTION_TILED_KERNEL(4)
ATTENTION_TILED_KERNEL(8)
ATTENTION_TILED_KERNEL(16)
ATTENTION_TILED_KERNEL(24)
ATTENTION_TILED_KERNEL(32)

// Split-key attention for one query row (a decode step). The keys a query
// admits are cut into chunks of ATTENTION_CHUNK_KEYS from its first admitted
// key; block (chunk, head) computes that chunk's softmax max, its sum of
// exponentials and its normalized weighted sum of V into `partial`
// (head_dimension + 2 floats per (head, chunk)). A chunk past the admitted
// keys writes an empty partial. `attention_combine_kernel` then merges the
// chunks of each head in chunk order, so the result does not depend on
// block scheduling. Each warp scores one key at a time, its lanes reading
// consecutive elements of the K row and adding their products with a
// fixed butterfly.
#define ATTENTION_CHUNK_KEYS 128
#define ATTENTION_SPLIT_THREADS 128
extern "C" __global__ void attention_split_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* partial,
    unsigned long long kv_seq_len,
    unsigned long long head_count,
    unsigned long long kv_head_count,
    unsigned long long head_dimension,
    unsigned long long q_model_dim,
    unsigned long long kv_model_dim,
    int causal,
    long long window_size, // -1 means "no window"
    unsigned long long query_position_offset,
    unsigned long long chunks,
    const unsigned long long* step_offset
) {
    __shared__ float scores[ATTENTION_CHUNK_KEYS];
    __shared__ float reduce[ATTENTION_SPLIT_THREADS];
    extern __shared__ float q_shared[];
    if (step_offset) {
        kv_seq_len += *step_offset;
        query_position_offset += *step_offset;
    }
    unsigned long long chunk = blockIdx.x;
    unsigned long long head = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 31;
    unsigned int warp = tid >> 5;
    unsigned long long group_size = head_count / kv_head_count;
    unsigned long long kv_offset = (head / group_size) * head_dimension;
    float scale = 1.0f / sqrtf((float)head_dimension);

    unsigned long long query_position = query_position_offset;
    unsigned long long key_upper = causal ? (query_position + 1) : kv_seq_len;
    unsigned long long key_lower = 0;
    if (window_size >= 0) {
        unsigned long long window = (unsigned long long)window_size;
        unsigned long long back = window > 0 ? window - 1 : 0;
        key_lower = query_position > back ? query_position - back : 0;
        if (key_lower > key_upper) {
            key_lower = key_upper;
        }
    }
    unsigned long long first = key_lower + chunk * ATTENTION_CHUNK_KEYS;
    unsigned long long last = first + ATTENTION_CHUNK_KEYS;
    if (last > key_upper) {
        last = key_upper;
    }
    float* out = partial + (head * chunks + chunk) * (head_dimension + 2);
    if (first >= last) {
        for (unsigned long long d = tid; d < head_dimension; d += ATTENTION_SPLIT_THREADS) {
            out[d] = 0.0f;
        }
        if (tid == 0) {
            out[head_dimension] = neg_inf();
            out[head_dimension + 1] = 0.0f;
        }
        return;
    }
    unsigned long long count = last - first;

    const float* q_row = q + head * head_dimension;
    for (unsigned long long d = tid; d < head_dimension; d += ATTENTION_SPLIT_THREADS) {
        q_shared[d] = q_row[d];
    }
    __syncthreads();

    for (unsigned long long j = warp; j < count; j += ATTENTION_SPLIT_THREADS / 32) {
        const float* k_row = k + (first + j) * kv_model_dim + kv_offset;
        float dot = 0.0f;
        for (unsigned long long d = lane; d < head_dimension; d += 32) {
            dot += q_shared[d] * k_row[d];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0) {
            scores[j] = dot * scale;
        }
    }
    __syncthreads();

    float local_max = neg_inf();
    for (unsigned long long j = tid; j < count; j += ATTENTION_SPLIT_THREADS) {
        local_max = fmaxf(local_max, scores[j]);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (unsigned int stride = ATTENTION_SPLIT_THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
        }
        __syncthreads();
    }
    float max_value = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (unsigned long long j = tid; j < count; j += ATTENTION_SPLIT_THREADS) {
        float weight = expf(scores[j] - max_value);
        scores[j] = weight;
        local_sum += weight;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (unsigned int stride = ATTENTION_SPLIT_THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            reduce[tid] += reduce[tid + stride];
        }
        __syncthreads();
    }
    float sum = reduce[0];
    // Weights normalized before they weight V, as `attention_block_kernel`:
    // a chunk's output is then a weighted mean of V rows, which cannot
    // overflow where the result is representable.
    for (unsigned long long j = tid; j < count; j += ATTENTION_SPLIT_THREADS) {
        scores[j] = scores[j] / sum;
    }
    __syncthreads();

    for (unsigned long long d = tid; d < head_dimension; d += ATTENTION_SPLIT_THREADS) {
        float acc = 0.0f;
        for (unsigned long long j = 0; j < count; j++) {
            acc += scores[j] * v[(first + j) * kv_model_dim + kv_offset + d];
        }
        out[d] = acc;
    }
    if (tid == 0) {
        out[head_dimension] = max_value;
        out[head_dimension + 1] = sum;
    }
}

// Split-key attention for one query row, all query heads of one KV head
// together (grouped-query attention, 64- or 128-dimension heads, at most
// ATTENTION_GQA_MAX_GROUP query heads per KV head). Block (chunk, kv_head)
// stages one chunk of K and V rows once, with coalesced loads,
// scores them for every query head of the group, and writes each head's
// partial in `attention_split_kernel`'s layout, so
// `attention_combine_kernel` merges them unchanged. Weights are normalized
// within the chunk before they weight V, as there.
#define ATTENTION_GQA_MAX_GROUP 8
#define ATTENTION_GQA_THREADS 128
template <int D, int KEYS>
__device__ __forceinline__ void attention_split_gqa_body(
    const float* q,
    const float* k,
    const float* v,
    float* partial,
    unsigned long long kv_seq_len,
    unsigned long long head_count,
    unsigned long long kv_head_count,
    unsigned long long kv_model_dim,
    int causal,
    long long window_size, // -1 means "no window"
    unsigned long long query_position_offset,
    unsigned long long chunks,
    const unsigned long long* step_offset
) {
    // K rows padded to D + 1 floats: a warp scoring 32 consecutive keys
    // reads 32 distinct banks.
    __shared__ float k_s[KEYS][D + 1];
    __shared__ float v_s[KEYS][D];
    __shared__ float q_s[ATTENTION_GQA_MAX_GROUP][D];
    __shared__ float p_s[ATTENTION_GQA_MAX_GROUP][KEYS];
    __shared__ float stats[ATTENTION_GQA_MAX_GROUP][2];
    if (step_offset) {
        kv_seq_len += *step_offset;
        query_position_offset += *step_offset;
    }
    unsigned long long chunk = blockIdx.x;
    unsigned long long kv_head = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 31;
    unsigned int warp = tid >> 5;
    unsigned int group = (unsigned int)(head_count / kv_head_count);
    unsigned long long first_head = kv_head * group;
    unsigned long long kv_offset = kv_head * D;
    // 1/sqrt(D), correctly rounded (exact for 64 dimensions).
    const float scale = 1.0f / sqrtf((float)D);

    unsigned long long query_position = query_position_offset;
    unsigned long long key_upper = causal ? (query_position + 1) : kv_seq_len;
    unsigned long long key_lower = 0;
    if (window_size >= 0) {
        unsigned long long window = (unsigned long long)window_size;
        unsigned long long back = window > 0 ? window - 1 : 0;
        key_lower = query_position > back ? query_position - back : 0;
        if (key_lower > key_upper) {
            key_lower = key_upper;
        }
    }
    unsigned long long first = key_lower + chunk * KEYS;
    unsigned long long last = first + KEYS;
    if (last > key_upper) {
        last = key_upper;
    }
    if (first >= last) {
        for (unsigned int e = tid; e < group * (D + 2); e += ATTENTION_GQA_THREADS) {
            unsigned int g = e / (D + 2);
            unsigned int d = e % (D + 2);
            float* out = partial + ((first_head + g) * chunks + chunk) * (D + 2);
            out[d] = d == D ? neg_inf() : 0.0f;
        }
        return;
    }
    unsigned int count = (unsigned int)(last - first);

    // Stage K, V (rows past `count` as zeros) and the group's queries. The
    // loop has a fixed trip count and is unrolled in batches, so each
    // thread has several loads in flight instead of one at a time.
    const float* k_base = k + first * kv_model_dim + kv_offset;
    const float* v_base = v + first * kv_model_dim + kv_offset;
#pragma unroll 8
    for (unsigned int it = 0; it < KEYS * D / ATTENTION_GQA_THREADS; it++) {
        unsigned int e = tid + it * ATTENTION_GQA_THREADS;
        unsigned int r = e / D;
        unsigned int d = e % D;
        bool present = r < count;
        unsigned long long index = (unsigned long long)r * kv_model_dim + d;
        k_s[r][d] = present ? k_base[index] : 0.0f;
        v_s[r][d] = present ? v_base[index] : 0.0f;
    }
    for (unsigned int e = tid; e < group * D; e += ATTENTION_GQA_THREADS) {
        q_s[e / D][e % D] = q[first_head * D + e];
    }
    __syncthreads();

    // Scores: pair (g, j) for key j of head g; a warp takes 32 keys of one
    // head.
    for (unsigned int pair = tid; pair < group * KEYS; pair += ATTENTION_GQA_THREADS) {
        unsigned int g = pair / KEYS;
        unsigned int j = pair % KEYS;
        float dot = 0.0f;
#pragma unroll 16
        for (unsigned int d = 0; d < D; d++) {
            dot += q_s[g][d] * k_s[j][d];
        }
        p_s[g][j] = j < count ? dot * scale : neg_inf();
    }
    __syncthreads();

    // Softmax per head: a warp per head, KEYS / 32 keys per lane.
    for (unsigned int g = warp; g < group; g += ATTENTION_GQA_THREADS / 32) {
        float score[KEYS / 32];
        float max_value = neg_inf();
#pragma unroll
        for (int i = 0; i < KEYS / 32; i++) {
            score[i] = p_s[g][lane + 32 * i];
            max_value = fmaxf(max_value, score[i]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffffu, max_value, offset));
        }
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < KEYS / 32; i++) {
            score[i] = expf(score[i] - max_value);
            sum += score[i];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        }
#pragma unroll
        for (int i = 0; i < KEYS / 32; i++) {
            p_s[g][lane + 32 * i] = score[i] / sum;
        }
        if (lane == 0) {
            stats[g][0] = max_value;
            stats[g][1] = sum;
        }
    }
    __syncthreads();

    // Weighted sums of V: output (g, d), a warp covering 32 dimensions of
    // one head.
    for (unsigned int e = tid; e < group * D; e += ATTENTION_GQA_THREADS) {
        unsigned int g = e / D;
        unsigned int d = e % D;
        float acc = 0.0f;
#pragma unroll 16
        for (unsigned int j = 0; j < KEYS; j++) {
            acc += p_s[g][j] * v_s[j][d];
        }
        float* out = partial + ((first_head + g) * chunks + chunk) * (D + 2);
        out[d] = acc;
        if (d == 0) {
            out[D] = stats[g][0];
            out[D + 1] = stats[g][1];
        }
    }
}

#define ATTENTION_GQA_ARGS                                                        \
    const float* q, const float* k, const float* v, float* partial,              \
    unsigned long long kv_seq_len, unsigned long long head_count,                \
    unsigned long long kv_head_count, unsigned long long kv_model_dim,           \
    int causal, long long window_size, unsigned long long query_position_offset, \
    unsigned long long chunks, const unsigned long long* step_offset
#define ATTENTION_GQA_PASS                                                        \
    q, k, v, partial, kv_seq_len, head_count, kv_head_count, kv_model_dim,       \
    causal, window_size, query_position_offset, chunks, step_offset

// 64-dimension heads, 64-key chunks (about 37 KB of shared memory).
extern "C" __global__ void attention_split_gqa_kernel(ATTENTION_GQA_ARGS) {
    attention_split_gqa_body<64, 64>(ATTENTION_GQA_PASS);
}

// 128-dimension heads, 32-key chunks (about 37 KB of shared memory).
extern "C" __global__ void attention_split_gqa128_kernel(ATTENTION_GQA_ARGS) {
    attention_split_gqa_body<128, 32>(ATTENTION_GQA_PASS);
}

// Merges each head's chunks from `attention_split_kernel` (or the grouped
// `attention_split_gqa_kernel`), in chunk order:
// out = sum_c o_c * (s_c * exp(m_c - M) / sum_c' s_c' * exp(m_c' - M)), with M the
// largest chunk max. Empty chunks are skipped; a head with no admitted key
// gets zeros. The block shares the per-chunk work: the maximum and the
// chunk weights are computed once per chunk (into dynamic shared memory,
// `chunks` floats, when they fit) instead of once per output dimension,
// and the weighted sum of each dimension is split across lanes of chunks.
extern "C" __global__ void attention_combine_kernel(
    const float* partial,
    float* out,
    unsigned long long head_count,
    unsigned long long head_dimension,
    unsigned long long chunks,
    int cache_weights
) {
    // `chunks` floats when `cache_weights`; otherwise (too many chunks for
    // shared memory) each weight is recomputed from its partial.
    extern __shared__ float chunk_weight[];
    __shared__ float reduce[1024];
    unsigned long long head = blockIdx.x;
    if (head >= head_count) {
        return;
    }
    unsigned int tid = threadIdx.x;
    unsigned int threads = blockDim.x;
    const float* head_partial = partial + head * chunks * (head_dimension + 2);
    float local_max = neg_inf();
    for (unsigned long long c = tid; c < chunks; c += threads) {
        const float* p = head_partial + c * (head_dimension + 2);
        if (p[head_dimension + 1] > 0.0f) {
            local_max = fmaxf(local_max, p[head_dimension]);
        }
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (unsigned int stride = 1; stride < threads; stride <<= 1) {
        if (tid % (2 * stride) == 0 && tid + stride < threads) {
            reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
        }
        __syncthreads();
    }
    float max_value = reduce[0];
    __syncthreads();
    auto weight_of = [&](unsigned long long c) -> float {
        if (cache_weights) {
            return chunk_weight[c];
        }
        const float* p = head_partial + c * (head_dimension + 2);
        return p[head_dimension + 1] > 0.0f
            ? p[head_dimension + 1] * expf(p[head_dimension] - max_value)
            : 0.0f;
    };
    if (cache_weights) {
        for (unsigned long long c = tid; c < chunks; c += threads) {
            const float* p = head_partial + c * (head_dimension + 2);
            chunk_weight[c] = p[head_dimension + 1] > 0.0f
                ? p[head_dimension + 1] * expf(p[head_dimension] - max_value)
                : 0.0f;
        }
        __syncthreads();
    }
    // The total in chunk order (one thread: `chunks` is small next to the
    // dimensions' work).
    if (tid == 0) {
        float total = 0.0f;
        for (unsigned long long c = 0; c < chunks; c++) {
            total += weight_of(c);
        }
        reduce[0] = total;
    }
    __syncthreads();
    float total = reduce[0];
    // Thread t sums dimension t % width over every `lanes`-th chunk from
    // t / width, several loads in flight; the lanes' sums are then added in
    // lane order, so the result does not depend on scheduling.
    unsigned int lanes = threads / (unsigned int)head_dimension;
    if (lanes == 0) {
        lanes = 1;
    }
    unsigned int width = threads / lanes;
    __syncthreads();
    for (unsigned long long d0 = 0; d0 < head_dimension; d0 += width) {
        unsigned long long d = d0 + tid % width;
        unsigned int lane = tid / width;
        float acc = 0.0f;
        if (d < head_dimension && lane < lanes) {
#pragma unroll 4
            for (unsigned long long c = lane; c < chunks; c += lanes) {
                float w = weight_of(c);
                if (w > 0.0f) {
                    // Each chunk's share of the softmax mass, at most 1:
                    // the sum stays a weighted mean of V rows.
                    acc += head_partial[c * (head_dimension + 2) + d] * (w / total);
                }
            }
        }
        reduce[tid] = acc;
        __syncthreads();
        if (lane == 0 && d < head_dimension) {
            float sum = 0.0f;
            for (unsigned int l = 0; l < lanes; l++) {
                sum += reduce[l * width + tid % width];
            }
            out[head * head_dimension + d] = sum;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Native half-precision (F16/BF16) elementwise compute
// (`add-native-cuda-half-precision-compute` Phase 2)
// ---------------------------------------------------------------------------
//
// Device-resident buffers hold real 2-byte `unsigned short` bit patterns,
// not `f32` promoted to 4 bytes -- genuine memory savings and genuine
// half-precision rounding behavior on-device, not merely a storage-format
// label. Conversion to/from `float` happens per-element inside the kernel
// via manual IEEE 754 bit manipulation (the same algorithm as this crate's
// own `half_precision.rs`, itself ported from `magnetar-runtime`'s
// exhaustively round-trip-tested original) rather than `cuda_fp16.h`/
// `cuda_bf16.h`'s `__half`/`nv_bfloat16` types and intrinsics: NVRTC's
// minimal preprocessor environment (see this file's own header comment)
// does not reliably expose those headers, and `__int_as_float`/
// `__float_as_int` are already proven to work here (`neg_inf()` above).

__device__ __forceinline__ float half_bits_to_float(unsigned short bits) {
    unsigned int sign = ((unsigned int)(bits >> 15)) << 31;
    unsigned int exponent = (unsigned int)((bits >> 10) & 0x1F);
    unsigned int mantissa = (unsigned int)(bits & 0x3FF);
    unsigned int magnitude_bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            magnitude_bits = 0;
        } else {
            unsigned int m = mantissa;
            unsigned int shift = 0;
            while ((m & 0x400) == 0) {
                m <<= 1;
                shift += 1;
            }
            m &= 0x3FF;
            unsigned int f32_exponent = 127 - 15 - shift + 1;
            magnitude_bits = (f32_exponent << 23) | (m << 13);
        }
    } else if (exponent == 0x1F) {
        magnitude_bits = (0xFFu << 23) | (mantissa << 13);
    } else {
        int f32_exponent = (int)exponent - 15 + 127;
        magnitude_bits = ((unsigned int)f32_exponent << 23) | (mantissa << 13);
    }
    return __int_as_float((int)(sign | magnitude_bits));
}

__device__ __forceinline__ unsigned short float_to_half_bits(float value) {
    unsigned int bits = (unsigned int)__float_as_int(value);
    unsigned short sign = (unsigned short)((bits >> 16) & 0x8000);
    unsigned int mantissa_f32 = bits & 0x007FFFFF;
    int exp_f32 = (int)((bits >> 23) & 0xFF);

    if (exp_f32 == 0xFF) {
        if (mantissa_f32 == 0) {
            return sign | 0x7C00;
        }
        unsigned short payload = (unsigned short)(mantissa_f32 >> 13);
        if (payload == 0) {
            payload = 1;
        }
        return sign | 0x7C00 | payload;
    }

    int unbiased = exp_f32 - 127;

    if (unbiased > 15) {
        return sign | 0x7C00;
    }

    if (unbiased < -14) {
        unsigned int significand = (exp_f32 == 0) ? mantissa_f32 : (mantissa_f32 | 0x00800000u);
        if (significand == 0) {
            return sign;
        }
        unsigned int shift = (unsigned int)(-(unbiased + 1));
        if (shift >= 32) {
            return sign;
        }
        unsigned int half = 1u << (shift - 1);
        unsigned int mask = (1u << shift) - 1;
        unsigned int truncated = significand >> shift;
        unsigned int remainder = significand & mask;
        unsigned int result = truncated;
        if (remainder > half || (remainder == half && (result & 1u) == 1u)) {
            result += 1;
        }
        return sign | (unsigned short)result;
    }

    unsigned int half = 1u << 12;
    unsigned int mask = (1u << 13) - 1;
    unsigned int truncated = mantissa_f32 >> 13;
    unsigned int remainder = mantissa_f32 & mask;
    unsigned int mantissa10 = truncated;
    if (remainder > half || (remainder == half && (mantissa10 & 1u) == 1u)) {
        mantissa10 += 1;
    }
    unsigned int exp16 = (unsigned int)(unbiased + 15);
    if (mantissa10 == 0x400) {
        unsigned int new_exp = exp16 + 1;
        if (new_exp >= 0x1F) {
            return sign | 0x7C00;
        }
        return sign | (unsigned short)(new_exp << 10);
    }
    return sign | (unsigned short)((exp16 << 10) | mantissa10);
}

__device__ __forceinline__ unsigned short float_to_bf16_bits(float value) {
    unsigned int bits = (unsigned int)__float_as_int(value);
    unsigned int exp_f32 = (bits >> 23) & 0xFF;
    unsigned int mantissa_f32 = bits & 0x007FFFFF;
    if (exp_f32 == 0xFF && mantissa_f32 != 0) {
        unsigned short sign = (unsigned short)((bits >> 16) & 0x8000);
        unsigned short mantissa7 = (unsigned short)((bits >> 16) & 0x7F);
        if (mantissa7 == 0) {
            mantissa7 = 0x40;
        }
        return sign | 0x7F80 | mantissa7;
    }
    unsigned int rounding_bias = 0x00007FFFu + ((bits >> 16) & 1u);
    unsigned int rounded = bits + rounding_bias;
    return (unsigned short)(rounded >> 16);
}

extern "C" __global__ void add_f16_kernel(const unsigned short* a, const unsigned short* b, unsigned short* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = float_to_half_bits(half_bits_to_float(a[i]) + half_bits_to_float(b[i]));
    }
}

extern "C" __global__ void mul_f16_kernel(const unsigned short* a, const unsigned short* b, unsigned short* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = float_to_half_bits(half_bits_to_float(a[i]) * half_bits_to_float(b[i]));
    }
}

extern "C" __global__ void add_bf16_kernel(const unsigned short* a, const unsigned short* b, unsigned short* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = float_to_bf16_bits(bf16_bits_to_float(a[i]) + bf16_bits_to_float(b[i]));
    }
}

extern "C" __global__ void mul_bf16_kernel(const unsigned short* a, const unsigned short* b, unsigned short* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = float_to_bf16_bits(bf16_bits_to_float(a[i]) * bf16_bits_to_float(b[i]));
    }
}

// `kv-write` (`append-kv-history-in-place`): copies `n` floats of new rows
// into the store at `row_offset` rows of `width`, plus a captured step's
// replay offset when `step_offset` is not null.
extern "C" __global__ void kv_write_kernel(
    const float* rows,
    float* store,
    unsigned long long n,
    unsigned long long width,
    unsigned long long row_offset,
    const unsigned long long* step_offset
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    unsigned long long offset = row_offset + (step_offset ? *step_offset : 0);
    store[offset * width + i] = rows[i];
}

// ---------------------------------------------------------------------------
// Packed `Q4_K` weights (`execute-direct-quantized-weights`): GGUF
// `block_q4_K` super-blocks kept as stored, 144 bytes for 256 values --
// `f16` `d` and `dmin`, twelve bytes packing eight 6-bit sub-block scales
// and mins, then 128 bytes of 4-bit quants. Within a super-block, quant
// byte `32 * c + i` (c = 0..3, i = 0..31) holds value `64 * c + i` in its
// low nibble (sub-block `2c`) and value `64 * c + 32 + i` in its high
// nibble (sub-block `2c + 1`); `value = d * scale * q - dmin * min`
// (`ggml-quants.c`'s `dequantize_row_q4_K`).

#define Q4K_BLOCK_VALUES 256
#define Q4K_BLOCK_BYTES 144

// An `f16`'s value, converted by the hardware (`cvt.f32.f16`).
__device__ __forceinline__ float q4k_half(const unsigned char* bytes) {
    unsigned short bits = (unsigned short)(bytes[0] | (bytes[1] << 8));
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    return value;
}

// A 4-bit quant as f32, exactly: placed in the low mantissa bits of 2^23,
// then 2^23 subtracted -- an integer OR and a float add instead of the
// slower integer-to-float conversion.
__device__ __forceinline__ float q4k_nibble(unsigned int nibble) {
    return __int_as_float(0x4B000000 | nibble) - 8388608.0f;
}

// Sub-block `j`'s 6-bit scale and min (`ggml-quants.c`'s
// `get_scale_min_k4`).
__device__ __forceinline__ void q4k_scale_min(
    int j,
    const unsigned char* q,
    float* scale,
    float* min
) {
    if (j < 4) {
        *scale = (float)(q[j] & 63);
        *min = (float)(q[j + 4] & 63);
    } else {
        *scale = (float)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *min = (float)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// The logical matrix as f32, one thread per stored element: stored row r,
// stored column c, written to `out[c * stored_rows + r]` when transposed.
extern "C" __global__ void q4k_dequantize_kernel(
    const unsigned char* blocks,
    float* out,
    unsigned long long stored_rows,
    unsigned long long stored_cols,
    unsigned int transposed
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= stored_rows * stored_cols) {
        return;
    }
    unsigned long long r = idx / stored_cols;
    unsigned long long c = idx % stored_cols;
    const unsigned char* block = blocks
        + (r * (stored_cols / Q4K_BLOCK_VALUES) + c / Q4K_BLOCK_VALUES) * Q4K_BLOCK_BYTES;
    unsigned int w = (unsigned int)(c % Q4K_BLOCK_VALUES);
    unsigned int chunk = w / 64;
    unsigned int within = w % 64;
    unsigned char byte = block[16 + 32 * chunk + (within % 32)];
    int sub = 2 * chunk + (within >= 32 ? 1 : 0);
    unsigned int q = within >= 32 ? (byte >> 4) : (byte & 0xF);
    float d = half_bits_to_float((unsigned short)(block[0] | (block[1] << 8)));
    float dmin = half_bits_to_float((unsigned short)(block[2] | (block[3] << 8)));
    float scale, min;
    q4k_scale_min(sub, block + 4, &scale, &min);
    float value = (d * scale) * (float)q - dmin * min;
    if (transposed) {
        out[c * stored_rows + r] = value;
    } else {
        out[idx] = value;
    }
}

// Output columns one warp of `q4k_matmul_rows_kernel` computes: each
// activation load serves all of them.
#define Q4K_COLS 2

// out[r, c] = sum_k a[r, k] * W[k, c] (+ addend) for a packed `Q4_K` weight
// stored transposed (stored row c holds column c's `k` values, `k / 256`
// super-blocks). One warp per Q4K_COLS output columns, up to
// Q8_MATMUL_ROWS rows of `a`: each step covers four super-blocks, eight
// lanes each; a lane loads 16 quant bytes per column (32 values: 16 of
// sub-block 2c in the low nibbles, 16 of sub-block 2c + 1 in the high
// ones), the matching 32 activations once for all its columns, and
// applies each column's sub-block scales and mins.
extern "C" __global__ void q4k_matmul_rows_kernel(
    const float* a,
    const unsigned char* blocks,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    const float* addend,
    int addend_mode
) {
    unsigned long long col0 =
        ((unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y) * Q4K_COLS;
    unsigned long long first_row = (unsigned long long)blockIdx.y * Q8_MATMUL_ROWS;
    if (col0 >= n || first_row >= m) {
        return;
    }
    unsigned int rows = (unsigned int)min((unsigned long long)Q8_MATMUL_ROWS, m - first_row);
    unsigned int cols = (unsigned int)min((unsigned long long)Q4K_COLS, n - col0);
    unsigned int lane = threadIdx.x;
    unsigned long long super_blocks = k / Q4K_BLOCK_VALUES;
    unsigned int part = lane % 8;
    unsigned int chunk = part / 2;
    unsigned int offset = 16 * (part % 2);
    float acc[Q8_MATMUL_ROWS][Q4K_COLS];
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
#pragma unroll
        for (int c = 0; c < Q4K_COLS; c++) {
            acc[r][c] = 0.0f;
        }
    }
    for (unsigned long long sb = lane / 8; sb < super_blocks; sb += 4) {
        int4 packed[Q4K_COLS];
        float d_lo[Q4K_COLS], m_lo[Q4K_COLS], d_hi[Q4K_COLS], m_hi[Q4K_COLS];
#pragma unroll
        for (int c = 0; c < Q4K_COLS; c++) {
            unsigned int column = (unsigned int)c < cols ? c : 0;
            const unsigned char* block =
                blocks + ((col0 + column) * super_blocks + sb) * Q4K_BLOCK_BYTES;
            packed[c] = *reinterpret_cast<const int4*>(block + 16 + 32 * chunk + offset);
            float d = q4k_half(block);
            float dmin = q4k_half(block + 2);
            float scale_lo, min_lo, scale_hi, min_hi;
            q4k_scale_min(2 * chunk, block + 4, &scale_lo, &min_lo);
            q4k_scale_min(2 * chunk + 1, block + 4, &scale_hi, &min_hi);
            d_lo[c] = d * scale_lo;
            m_lo[c] = dmin * min_lo;
            d_hi[c] = d * scale_hi;
            m_hi[c] = dmin * min_hi;
        }
        unsigned long long base = sb * Q4K_BLOCK_VALUES + 64 * chunk + offset;
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                const float* x = a + (first_row + r) * k + base;
                float dot_lo[Q4K_COLS], dot_hi[Q4K_COLS];
#pragma unroll
                for (int c = 0; c < Q4K_COLS; c++) {
                    dot_lo[c] = 0.0f;
                    dot_hi[c] = 0.0f;
                }
                float sum_lo = 0.0f, sum_hi = 0.0f;
#pragma unroll
                for (int j = 0; j < 16; j += 4) {
                    float4 lo = *reinterpret_cast<const float4*>(x + j);
                    float4 hi = *reinterpret_cast<const float4*>(x + 32 + j);
                    sum_lo += lo.x + lo.y + lo.z + lo.w;
                    sum_hi += hi.x + hi.y + hi.z + hi.w;
#pragma unroll
                    for (int c = 0; c < Q4K_COLS; c++) {
                        const unsigned char* q = reinterpret_cast<const unsigned char*>(&packed[c]);
                        dot_lo[c] += lo.x * q4k_nibble(q[j] & 0xF)
                            + lo.y * q4k_nibble(q[j + 1] & 0xF)
                            + lo.z * q4k_nibble(q[j + 2] & 0xF)
                            + lo.w * q4k_nibble(q[j + 3] & 0xF);
                        dot_hi[c] += hi.x * q4k_nibble(q[j] >> 4)
                            + hi.y * q4k_nibble(q[j + 1] >> 4)
                            + hi.z * q4k_nibble(q[j + 2] >> 4)
                            + hi.w * q4k_nibble(q[j + 3] >> 4);
                    }
                }
#pragma unroll
                for (int c = 0; c < Q4K_COLS; c++) {
                    acc[r][c] += d_lo[c] * dot_lo[c] - m_lo[c] * sum_lo
                        + d_hi[c] * dot_hi[c] - m_hi[c] * sum_hi;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
#pragma unroll
        for (int c = 0; c < Q4K_COLS; c++) {
            float sum = acc[r][c];
#pragma unroll
            for (int offset_ = 16; offset_ > 0; offset_ >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset_);
            }
            if (lane == 0 && r < (int)rows && (unsigned int)c < cols) {
                unsigned long long row = first_row + r;
                unsigned long long col = col0 + c;
                if (addend_mode == 1) {
                    sum += addend[col];
                } else if (addend_mode == 2) {
                    sum += addend[row * n + col];
                }
                out[row * n + col] = sum;
            }
        }
    }
}

// One packed `Q4_K` column's partial dot products for this lane's 32
// values of super-block `block` against `x` (the row of `a` at the lane's
// first value): its low-nibble half scaled by sub-block `2c`, its high one
// by `2c + 1`.
__device__ __forceinline__ float q4k_lane_dot(
    const unsigned char* block,
    unsigned int chunk,
    unsigned int offset,
    const float* x
) {
    int4 packed = *reinterpret_cast<const int4*>(block + 16 + 32 * chunk + offset);
    const unsigned char* q = reinterpret_cast<const unsigned char*>(&packed);
    float d = q4k_half(block);
    float dmin = q4k_half(block + 2);
    float scale_lo, min_lo, scale_hi, min_hi;
    q4k_scale_min(2 * chunk, block + 4, &scale_lo, &min_lo);
    q4k_scale_min(2 * chunk + 1, block + 4, &scale_hi, &min_hi);
    float dot_lo = 0.0f, sum_lo = 0.0f, dot_hi = 0.0f, sum_hi = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; j += 4) {
        float4 lo = *reinterpret_cast<const float4*>(x + j);
        float4 hi = *reinterpret_cast<const float4*>(x + 32 + j);
        dot_lo += lo.x * q4k_nibble(q[j] & 0xF) + lo.y * q4k_nibble(q[j + 1] & 0xF)
            + lo.z * q4k_nibble(q[j + 2] & 0xF) + lo.w * q4k_nibble(q[j + 3] & 0xF);
        dot_hi += hi.x * q4k_nibble(q[j] >> 4) + hi.y * q4k_nibble(q[j + 1] >> 4)
            + hi.z * q4k_nibble(q[j + 2] >> 4) + hi.w * q4k_nibble(q[j + 3] >> 4);
        sum_lo += lo.x + lo.y + lo.z + lo.w;
        sum_hi += hi.x + hi.y + hi.z + hi.w;
    }
    return (d * scale_lo) * dot_lo - (dmin * min_lo) * sum_lo
        + (d * scale_hi) * dot_hi - (dmin * min_hi) * sum_hi;
}

// `silu(a x gate) * (a x up)` for two packed `Q4_K` weights stored
// transposed: `q4k_matmul_rows_kernel`'s reading pattern over both, so a
// gated MLP's decode step writes one output and allocates nothing.
extern "C" __global__ void q4k_gated_matmul_rows_kernel(
    const float* a,
    const unsigned char* gate_blocks,
    const unsigned char* up_blocks,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n
) {
    unsigned long long col = (unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y;
    unsigned long long first_row = (unsigned long long)blockIdx.y * Q8_MATMUL_ROWS;
    if (col >= n || first_row >= m) {
        return;
    }
    unsigned int rows = (unsigned int)min((unsigned long long)Q8_MATMUL_ROWS, m - first_row);
    unsigned int lane = threadIdx.x;
    unsigned long long super_blocks = k / Q4K_BLOCK_VALUES;
    const unsigned char* wg = gate_blocks + col * super_blocks * Q4K_BLOCK_BYTES;
    const unsigned char* wu = up_blocks + col * super_blocks * Q4K_BLOCK_BYTES;
    unsigned int part = lane % 8;
    unsigned int chunk = part / 2;
    unsigned int offset = 16 * (part % 2);
    float gate_acc[Q8_MATMUL_ROWS];
    float up_acc[Q8_MATMUL_ROWS];
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        gate_acc[r] = 0.0f;
        up_acc[r] = 0.0f;
    }
    for (unsigned long long sb = lane / 8; sb < super_blocks; sb += 4) {
        unsigned long long base = sb * Q4K_BLOCK_VALUES + 64 * chunk + offset;
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                const float* x = a + (first_row + r) * k + base;
                gate_acc[r] += q4k_lane_dot(wg + sb * Q4K_BLOCK_BYTES, chunk, offset, x);
                up_acc[r] += q4k_lane_dot(wu + sb * Q4K_BLOCK_BYTES, chunk, offset, x);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        float g = gate_acc[r];
        float u = up_acc[r];
#pragma unroll
        for (int offset_ = 16; offset_ > 0; offset_ >>= 1) {
            g += __shfl_xor_sync(0xffffffffu, g, offset_);
            u += __shfl_xor_sync(0xffffffffu, u, offset_);
        }
        if (lane == 0 && r < (int)rows) {
            out[(first_row + r) * n + col] = g * (1.0f / (1.0f + expf(-g))) * u;
        }
    }
}

// ---------------------------------------------------------------------------
// Packed `Q6_K` weights (`execute-direct-quantized-weights`): GGUF
// `block_q6_K` super-blocks without their trailing `f16` `d`, which is
// kept as an f32 per super-block in a separate array -- 208 bytes per 256
// values, so every super-block stays 16-byte aligned: 128 bytes of low
// nibbles (`ql`), 64 bytes of high bit pairs (`qh`), sixteen signed 8-bit
// sub-block scales. In half `h` (values `128 h ..`), for `l` = 0..31,
// `ql[64 h + l]` holds values `l` (low nibble) and `l + 64` (high),
// `ql[64 h + 32 + l]` values `l + 32` and `l + 96`, and `qh[32 h + l]`
// their high bits in pairs 0-1, 2-3, 4-5, 6-7 for `l`, `l + 32`, `l + 64`,
// `l + 96`; value `l + 32 g` (g = 0..3) uses scale `8 h + l / 16 + 2 g`;
// `value = d * scale * (q - 32)` (`ggml-quants.c`'s `dequantize_row_q6_K`).

#define Q6K_BLOCK_VALUES 256
#define Q6K_BLOCK_BYTES 208

// The logical matrix as f32, one thread per stored element: stored row r,
// stored column c, written to `out[c * stored_rows + r]` when transposed.
extern "C" __global__ void q6k_dequantize_kernel(
    const unsigned char* blocks,
    const float* d,
    float* out,
    unsigned long long stored_rows,
    unsigned long long stored_cols,
    unsigned int transposed
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= stored_rows * stored_cols) {
        return;
    }
    unsigned long long r = idx / stored_cols;
    unsigned long long c = idx % stored_cols;
    unsigned long long super_block = r * (stored_cols / Q6K_BLOCK_VALUES) + c / Q6K_BLOCK_VALUES;
    const unsigned char* block = blocks + super_block * Q6K_BLOCK_BYTES;
    unsigned int w = (unsigned int)(c % Q6K_BLOCK_VALUES);
    unsigned int h = w / 128;
    unsigned int g = (w % 128) / 32;
    unsigned int l = w % 32;
    unsigned char low = block[64 * h + 32 * (g % 2) + l];
    unsigned int nibble = g < 2 ? (low & 0xF) : (low >> 4);
    unsigned int high = (block[128 + 32 * h + l] >> (2 * g)) & 3;
    int q = (int)(nibble | (high << 4)) - 32;
    float scale = (float)(signed char)block[192 + 8 * h + l / 16 + 2 * g];
    float value = (d[super_block] * scale) * (float)q;
    if (transposed) {
        out[c * stored_rows + r] = value;
    } else {
        out[idx] = value;
    }
}

// Output columns one warp of `q6k_matmul_rows_kernel` computes: each
// activation load serves all of them.
#define Q6K_COLS 2

// out[r, c] = sum_k a[r, k] * W[k, c] (+ addend) for a packed `Q6_K` weight
// stored transposed (stored row c holds column c's `k` values, `k / 256`
// super-blocks). One warp per Q6K_COLS output columns, up to
// Q8_MATMUL_ROWS rows of `a`: each step covers four super-blocks, eight
// lanes each; a lane takes half `h` = part / 4 and `l` = 8 (part % 4) ..
// + 8, loading eight bytes each of `ql` (twice) and `qh` per column --
// 32 values in four runs of eight, each run under one sub-block scale.
extern "C" __global__ void q6k_matmul_rows_kernel(
    const float* a,
    const unsigned char* blocks,
    const float* d,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    const float* addend,
    int addend_mode
) {
    unsigned long long col0 =
        ((unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y) * Q6K_COLS;
    unsigned long long first_row = (unsigned long long)blockIdx.y * Q8_MATMUL_ROWS;
    if (col0 >= n || first_row >= m) {
        return;
    }
    unsigned int rows = (unsigned int)min((unsigned long long)Q8_MATMUL_ROWS, m - first_row);
    unsigned int cols = (unsigned int)min((unsigned long long)Q6K_COLS, n - col0);
    unsigned int lane = threadIdx.x;
    unsigned long long super_blocks = k / Q6K_BLOCK_VALUES;
    unsigned int part = lane % 8;
    unsigned int h = part / 4;
    unsigned int l0 = 8 * (part % 4);
    float acc[Q8_MATMUL_ROWS][Q6K_COLS];
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
#pragma unroll
        for (int c = 0; c < Q6K_COLS; c++) {
            acc[r][c] = 0.0f;
        }
    }
    for (unsigned long long sb = lane / 8; sb < super_blocks; sb += 4) {
        uint2 low_a[Q6K_COLS], low_b[Q6K_COLS], high[Q6K_COLS];
        float scale[Q6K_COLS][4];
#pragma unroll
        for (int c = 0; c < Q6K_COLS; c++) {
            unsigned int column = (unsigned int)c < cols ? c : 0;
            unsigned long long index = (col0 + column) * super_blocks + sb;
            const unsigned char* block = blocks + index * Q6K_BLOCK_BYTES;
            low_a[c] = *reinterpret_cast<const uint2*>(block + 64 * h + l0);
            low_b[c] = *reinterpret_cast<const uint2*>(block + 64 * h + 32 + l0);
            high[c] = *reinterpret_cast<const uint2*>(block + 128 + 32 * h + l0);
            float block_d = d[index];
            const signed char* sc =
                reinterpret_cast<const signed char*>(block + 192 + 8 * h + l0 / 16);
#pragma unroll
            for (int g = 0; g < 4; g++) {
                scale[c][g] = block_d * (float)sc[2 * g];
            }
        }
        unsigned long long base = sb * Q6K_BLOCK_VALUES + 128 * h + l0;
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                const float* x = a + (first_row + r) * k + base;
                float xs[4][8];
                float sums[4];
#pragma unroll
                for (int g = 0; g < 4; g++) {
                    float4 v0 = *reinterpret_cast<const float4*>(x + 32 * g);
                    float4 v1 = *reinterpret_cast<const float4*>(x + 32 * g + 4);
                    xs[g][0] = v0.x; xs[g][1] = v0.y; xs[g][2] = v0.z; xs[g][3] = v0.w;
                    xs[g][4] = v1.x; xs[g][5] = v1.y; xs[g][6] = v1.z; xs[g][7] = v1.w;
                    sums[g] = (v0.x + v0.y + v0.z + v0.w) + (v1.x + v1.y + v1.z + v1.w);
                }
#pragma unroll
                for (int c = 0; c < Q6K_COLS; c++) {
                    const unsigned char* qa = reinterpret_cast<const unsigned char*>(&low_a[c]);
                    const unsigned char* qb = reinterpret_cast<const unsigned char*>(&low_b[c]);
                    const unsigned char* qh = reinterpret_cast<const unsigned char*>(&high[c]);
                    float dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        unsigned int hb = qh[i];
                        dot[0] += xs[0][i] * q4k_nibble((qa[i] & 0xF) | ((hb & 3) << 4));
                        dot[1] += xs[1][i] * q4k_nibble((qb[i] & 0xF) | (((hb >> 2) & 3) << 4));
                        dot[2] += xs[2][i] * q4k_nibble((qa[i] >> 4) | (((hb >> 4) & 3) << 4));
                        dot[3] += xs[3][i] * q4k_nibble((qb[i] >> 4) | (((hb >> 6) & 3) << 4));
                    }
                    float total = 0.0f;
#pragma unroll
                    for (int g = 0; g < 4; g++) {
                        total += scale[c][g] * (dot[g] - 32.0f * sums[g]);
                    }
                    acc[r][c] += total;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
#pragma unroll
        for (int c = 0; c < Q6K_COLS; c++) {
            float sum = acc[r][c];
#pragma unroll
            for (int offset_ = 16; offset_ > 0; offset_ >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset_);
            }
            if (lane == 0 && r < (int)rows && (unsigned int)c < cols) {
                unsigned long long row = first_row + r;
                unsigned long long col = col0 + c;
                if (addend_mode == 1) {
                    sum += addend[col];
                } else if (addend_mode == 2) {
                    sum += addend[row * n + col];
                }
                out[row * n + col] = sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Int8 activations (on unless `MAGNETAR_CUDA_INT8_ACTIVATIONS=0`): the
// decode-step packed matmuls read `a` quantized per 32-value block to
// signed 8-bit values and one f32 scale (`ggml`'s `block_q8_1` without its
// sum), and multiply with `__dp4a` (`i8_dot4`), four byte products per
// instruction, instead of converting every weight value to f32. This
// rounds the activations; the f32-activation Kernels above stay the
// disable control.

// Quantizes `blocks` consecutive 32-value blocks of `a`: one warp per
// block, `aq = round(a / d)` with `d = max |a| / 127` (0 for an all-zero
// block).
extern "C" __global__ void quantize_activations_q8_kernel(
    const float* a,
    signed char* aq,
    float* ad,
    unsigned long long blocks
) {
    unsigned long long block = (unsigned long long)blockIdx.x * blockDim.y + threadIdx.y;
    if (block >= blocks) {
        return;
    }
    unsigned int lane = threadIdx.x;
    float x = a[block * 32 + lane];
    float amax = fabsf(x);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    float d = amax / 127.0f;
    aq[block * 32 + lane] = (signed char)(amax == 0.0f ? 0 : __float2int_rn(x / d));
    if (lane == 0) {
        ad[block] = d;
    }
}

// Four signed byte products of `a` and `b` added to `c`: `__dp4a` from
// compute capability 6.1 (the module is compiled for the NVRTC default
// architecture, which may be older), else the same sum byte by byte.
__device__ __forceinline__ int i8_dot4(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        c += (int)(signed char)(a >> i) * (int)(signed char)(b >> i);
    }
    return c;
#endif
}

// The sum of `x`'s four signed bytes.
__device__ __forceinline__ int i8_sum4(int x) {
    return i8_dot4(x, 0x01010101, 0);
}

// out[r, c] = sum_k a[r, k] * W[k, c] (+ addend) as `q8_matmul_rows_kernel`
// lays it out, with `a` quantized (`aq`, one scale per 32 values in `ad`):
// each lane's sixteen weight quants meet sixteen activation bytes in four
// `__dp4a`.
extern "C" __global__ void q8_matmul_rows_int8_kernel(
    const signed char* aq,
    const float* ad,
    const signed char* quants,
    const float* scales,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    const float* addend,
    int addend_mode
) {
    unsigned long long col = (unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y;
    unsigned long long first_row = (unsigned long long)blockIdx.y * Q8_MATMUL_ROWS;
    if (col >= n || first_row >= m) {
        return;
    }
    unsigned int rows = (unsigned int)min((unsigned long long)Q8_MATMUL_ROWS, m - first_row);
    unsigned int lane = threadIdx.x;
    const signed char* w = quants + col * k;
    const float* w_scales = scales + col * (k >> 5);
    float acc[Q8_MATMUL_ROWS];
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        acc[r] = 0.0f;
    }
    for (unsigned long long i = (unsigned long long)lane * Q8_LANE_QUANTS; i < k;
         i += Q8_WARP_QUANTS) {
        int4 q = *reinterpret_cast<const int4*>(w + i);
        float d = w_scales[i >> 5];
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                unsigned long long at = (first_row + r) * k + i;
                int4 x = *reinterpret_cast<const int4*>(aq + at);
                int dot = i8_dot4(q.x, x.x, i8_dot4(q.y, x.y, i8_dot4(q.z, x.z, i8_dot4(q.w, x.w, 0))));
                acc[r] += d * ad[at >> 5] * (float)dot;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        float sum = acc[r];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0 && r < (int)rows) {
            unsigned long long row = first_row + r;
            if (addend_mode == 1) {
                sum += addend[col];
            } else if (addend_mode == 2) {
                sum += addend[row * n + col];
            }
            out[row * n + col] = sum;
        }
    }
}

// `q8_gated_matmul_rows_kernel` over int8 activations.
extern "C" __global__ void q8_gated_matmul_rows_int8_kernel(
    const signed char* aq,
    const float* ad,
    const signed char* gate_quants,
    const float* gate_scales,
    const signed char* up_quants,
    const float* up_scales,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n
) {
    unsigned long long col = (unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y;
    unsigned long long first_row = (unsigned long long)blockIdx.y * Q8_MATMUL_ROWS;
    if (col >= n || first_row >= m) {
        return;
    }
    unsigned int rows = (unsigned int)min((unsigned long long)Q8_MATMUL_ROWS, m - first_row);
    unsigned int lane = threadIdx.x;
    const signed char* wg = gate_quants + col * k;
    const signed char* wu = up_quants + col * k;
    const float* g_scales = gate_scales + col * (k >> 5);
    const float* u_scales = up_scales + col * (k >> 5);
    float gate_acc[Q8_MATMUL_ROWS];
    float up_acc[Q8_MATMUL_ROWS];
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        gate_acc[r] = 0.0f;
        up_acc[r] = 0.0f;
    }
    for (unsigned long long i = (unsigned long long)lane * Q8_LANE_QUANTS; i < k;
         i += Q8_WARP_QUANTS) {
        int4 qg = *reinterpret_cast<const int4*>(wg + i);
        int4 qu = *reinterpret_cast<const int4*>(wu + i);
        float dg = g_scales[i >> 5];
        float du = u_scales[i >> 5];
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                unsigned long long at = (first_row + r) * k + i;
                int4 x = *reinterpret_cast<const int4*>(aq + at);
                float da = ad[at >> 5];
                int g = i8_dot4(qg.x, x.x, i8_dot4(qg.y, x.y, i8_dot4(qg.z, x.z, i8_dot4(qg.w, x.w, 0))));
                int u = i8_dot4(qu.x, x.x, i8_dot4(qu.y, x.y, i8_dot4(qu.z, x.z, i8_dot4(qu.w, x.w, 0))));
                gate_acc[r] += dg * da * (float)g;
                up_acc[r] += du * da * (float)u;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        float g = gate_acc[r];
        float u = up_acc[r];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            g += __shfl_xor_sync(0xffffffffu, g, offset);
            u += __shfl_xor_sync(0xffffffffu, u, offset);
        }
        if (lane == 0 && r < (int)rows) {
            out[(first_row + r) * n + col] = g * (1.0f / (1.0f + expf(-g))) * u;
        }
    }
}

// One packed `Q4_K` column's contribution for this lane's 32 values of a
// super-block (`q4k_matmul_rows_kernel`'s lane mapping) against int8
// activations: `xlo`/`xhi` the sixteen activation bytes of the low- and
// high-nibble values, `dlo`/`dhi` their block scales.
__device__ __forceinline__ float q4k_lane_dot_int8(
    const unsigned char* block,
    unsigned int chunk,
    int4 packed,
    int4 xlo,
    int4 xhi,
    float dlo,
    float dhi
) {
    float d = q4k_half(block);
    float dmin = q4k_half(block + 2);
    float scale_lo, min_lo, scale_hi, min_hi;
    q4k_scale_min(2 * chunk, block + 4, &scale_lo, &min_lo);
    q4k_scale_min(2 * chunk + 1, block + 4, &scale_hi, &min_hi);
    const int* q = reinterpret_cast<const int*>(&packed);
    const int* lo = reinterpret_cast<const int*>(&xlo);
    const int* hi = reinterpret_cast<const int*>(&xhi);
    int dot_lo = 0, dot_hi = 0, sum_lo = 0, sum_hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        dot_lo = i8_dot4(q[j] & 0x0F0F0F0F, lo[j], dot_lo);
        dot_hi = i8_dot4((q[j] >> 4) & 0x0F0F0F0F, hi[j], dot_hi);
        sum_lo += i8_sum4(lo[j]);
        sum_hi += i8_sum4(hi[j]);
    }
    return dlo * ((d * scale_lo) * (float)dot_lo - (dmin * min_lo) * (float)sum_lo)
        + dhi * ((d * scale_hi) * (float)dot_hi - (dmin * min_hi) * (float)sum_hi);
}

// `q4k_matmul_rows_kernel` over int8 activations.
extern "C" __global__ void q4k_matmul_rows_int8_kernel(
    const signed char* aq,
    const float* ad,
    const unsigned char* blocks,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    const float* addend,
    int addend_mode
) {
    unsigned long long col0 =
        ((unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y) * Q4K_COLS;
    unsigned long long first_row = (unsigned long long)blockIdx.y * Q8_MATMUL_ROWS;
    if (col0 >= n || first_row >= m) {
        return;
    }
    unsigned int rows = (unsigned int)min((unsigned long long)Q8_MATMUL_ROWS, m - first_row);
    unsigned int cols = (unsigned int)min((unsigned long long)Q4K_COLS, n - col0);
    unsigned int lane = threadIdx.x;
    unsigned long long super_blocks = k / Q4K_BLOCK_VALUES;
    unsigned int part = lane % 8;
    unsigned int chunk = part / 2;
    unsigned int offset = 16 * (part % 2);
    float acc[Q8_MATMUL_ROWS][Q4K_COLS];
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
#pragma unroll
        for (int c = 0; c < Q4K_COLS; c++) {
            acc[r][c] = 0.0f;
        }
    }
    for (unsigned long long sb = lane / 8; sb < super_blocks; sb += 4) {
        const unsigned char* block[Q4K_COLS];
        int4 packed[Q4K_COLS];
#pragma unroll
        for (int c = 0; c < Q4K_COLS; c++) {
            unsigned int column = (unsigned int)c < cols ? c : 0;
            block[c] = blocks + ((col0 + column) * super_blocks + sb) * Q4K_BLOCK_BYTES;
            packed[c] = *reinterpret_cast<const int4*>(block[c] + 16 + 32 * chunk + offset);
        }
        unsigned long long base = sb * Q4K_BLOCK_VALUES + 64 * chunk + offset;
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                unsigned long long at = (first_row + r) * k + base;
                int4 xlo = *reinterpret_cast<const int4*>(aq + at);
                int4 xhi = *reinterpret_cast<const int4*>(aq + at + 32);
                float dlo = ad[at >> 5];
                float dhi = ad[(at + 32) >> 5];
#pragma unroll
                for (int c = 0; c < Q4K_COLS; c++) {
                    acc[r][c] +=
                        q4k_lane_dot_int8(block[c], chunk, packed[c], xlo, xhi, dlo, dhi);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
#pragma unroll
        for (int c = 0; c < Q4K_COLS; c++) {
            float sum = acc[r][c];
#pragma unroll
            for (int offset_ = 16; offset_ > 0; offset_ >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset_);
            }
            if (lane == 0 && r < (int)rows && (unsigned int)c < cols) {
                unsigned long long row = first_row + r;
                unsigned long long col = col0 + c;
                if (addend_mode == 1) {
                    sum += addend[col];
                } else if (addend_mode == 2) {
                    sum += addend[row * n + col];
                }
                out[row * n + col] = sum;
            }
        }
    }
}

// `q4k_gated_matmul_rows_kernel` over int8 activations.
extern "C" __global__ void q4k_gated_matmul_rows_int8_kernel(
    const signed char* aq,
    const float* ad,
    const unsigned char* gate_blocks,
    const unsigned char* up_blocks,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n
) {
    unsigned long long col = (unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y;
    unsigned long long first_row = (unsigned long long)blockIdx.y * Q8_MATMUL_ROWS;
    if (col >= n || first_row >= m) {
        return;
    }
    unsigned int rows = (unsigned int)min((unsigned long long)Q8_MATMUL_ROWS, m - first_row);
    unsigned int lane = threadIdx.x;
    unsigned long long super_blocks = k / Q4K_BLOCK_VALUES;
    const unsigned char* wg = gate_blocks + col * super_blocks * Q4K_BLOCK_BYTES;
    const unsigned char* wu = up_blocks + col * super_blocks * Q4K_BLOCK_BYTES;
    unsigned int part = lane % 8;
    unsigned int chunk = part / 2;
    unsigned int offset = 16 * (part % 2);
    float gate_acc[Q8_MATMUL_ROWS];
    float up_acc[Q8_MATMUL_ROWS];
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        gate_acc[r] = 0.0f;
        up_acc[r] = 0.0f;
    }
    for (unsigned long long sb = lane / 8; sb < super_blocks; sb += 4) {
        const unsigned char* bg = wg + sb * Q4K_BLOCK_BYTES;
        const unsigned char* bu = wu + sb * Q4K_BLOCK_BYTES;
        int4 pg = *reinterpret_cast<const int4*>(bg + 16 + 32 * chunk + offset);
        int4 pu = *reinterpret_cast<const int4*>(bu + 16 + 32 * chunk + offset);
        unsigned long long base = sb * Q4K_BLOCK_VALUES + 64 * chunk + offset;
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                unsigned long long at = (first_row + r) * k + base;
                int4 xlo = *reinterpret_cast<const int4*>(aq + at);
                int4 xhi = *reinterpret_cast<const int4*>(aq + at + 32);
                float dlo = ad[at >> 5];
                float dhi = ad[(at + 32) >> 5];
                gate_acc[r] += q4k_lane_dot_int8(bg, chunk, pg, xlo, xhi, dlo, dhi);
                up_acc[r] += q4k_lane_dot_int8(bu, chunk, pu, xlo, xhi, dlo, dhi);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        float g = gate_acc[r];
        float u = up_acc[r];
#pragma unroll
        for (int offset_ = 16; offset_ > 0; offset_ >>= 1) {
            g += __shfl_xor_sync(0xffffffffu, g, offset_);
            u += __shfl_xor_sync(0xffffffffu, u, offset_);
        }
        if (lane == 0 && r < (int)rows) {
            out[(first_row + r) * n + col] = g * (1.0f / (1.0f + expf(-g))) * u;
        }
    }
}

// `q6k_matmul_rows_kernel` over int8 activations: each of the lane's four
// runs of eight values is two `__dp4a` against its eight activation bytes,
// the -32 offset applied through their sum.
extern "C" __global__ void q6k_matmul_rows_int8_kernel(
    const signed char* aq,
    const float* ad,
    const unsigned char* blocks,
    const float* d,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    const float* addend,
    int addend_mode
) {
    unsigned long long col0 =
        ((unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y) * Q6K_COLS;
    unsigned long long first_row = (unsigned long long)blockIdx.y * Q8_MATMUL_ROWS;
    if (col0 >= n || first_row >= m) {
        return;
    }
    unsigned int rows = (unsigned int)min((unsigned long long)Q8_MATMUL_ROWS, m - first_row);
    unsigned int cols = (unsigned int)min((unsigned long long)Q6K_COLS, n - col0);
    unsigned int lane = threadIdx.x;
    unsigned long long super_blocks = k / Q6K_BLOCK_VALUES;
    unsigned int part = lane % 8;
    unsigned int h = part / 4;
    unsigned int l0 = 8 * (part % 4);
    float acc[Q8_MATMUL_ROWS][Q6K_COLS];
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
#pragma unroll
        for (int c = 0; c < Q6K_COLS; c++) {
            acc[r][c] = 0.0f;
        }
    }
    for (unsigned long long sb = lane / 8; sb < super_blocks; sb += 4) {
        // Each column's 32 six-bit quants as eight words of four bytes:
        // run g in words 2g and 2g + 1.
        int q[Q6K_COLS][8];
        float scale[Q6K_COLS][4];
#pragma unroll
        for (int c = 0; c < Q6K_COLS; c++) {
            unsigned int column = (unsigned int)c < cols ? c : 0;
            unsigned long long index = (col0 + column) * super_blocks + sb;
            const unsigned char* block = blocks + index * Q6K_BLOCK_BYTES;
            uint2 la = *reinterpret_cast<const uint2*>(block + 64 * h + l0);
            uint2 lb = *reinterpret_cast<const uint2*>(block + 64 * h + 32 + l0);
            uint2 hb = *reinterpret_cast<const uint2*>(block + 128 + 32 * h + l0);
            unsigned int a_w[2] = {la.x, la.y};
            unsigned int b_w[2] = {lb.x, lb.y};
            unsigned int h_w[2] = {hb.x, hb.y};
#pragma unroll
            for (int j = 0; j < 2; j++) {
                q[c][j] = (int)((a_w[j] & 0x0F0F0F0Fu) | ((h_w[j] & 0x03030303u) << 4));
                q[c][2 + j] = (int)((b_w[j] & 0x0F0F0F0Fu) | (((h_w[j] >> 2) & 0x03030303u) << 4));
                q[c][4 + j] = (int)(((a_w[j] >> 4) & 0x0F0F0F0Fu) | (((h_w[j] >> 4) & 0x03030303u) << 4));
                q[c][6 + j] = (int)(((b_w[j] >> 4) & 0x0F0F0F0Fu) | (((h_w[j] >> 6) & 0x03030303u) << 4));
            }
            float block_d = d[index];
            const signed char* sc =
                reinterpret_cast<const signed char*>(block + 192 + 8 * h + l0 / 16);
#pragma unroll
            for (int g = 0; g < 4; g++) {
                scale[c][g] = block_d * (float)sc[2 * g];
            }
        }
        unsigned long long base = sb * Q6K_BLOCK_VALUES + 128 * h + l0;
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                unsigned long long at = (first_row + r) * k + base;
                int x[8];
                float da[4];
                int sums[4];
#pragma unroll
                for (int g = 0; g < 4; g++) {
                    int2 v = *reinterpret_cast<const int2*>(aq + at + 32 * g);
                    x[2 * g] = v.x;
                    x[2 * g + 1] = v.y;
                    da[g] = ad[(at + 32 * g) >> 5];
                    sums[g] = i8_sum4(v.x) + i8_sum4(v.y);
                }
#pragma unroll
                for (int c = 0; c < Q6K_COLS; c++) {
                    float total = 0.0f;
#pragma unroll
                    for (int g = 0; g < 4; g++) {
                        int dot = i8_dot4(q[c][2 * g], x[2 * g], i8_dot4(q[c][2 * g + 1], x[2 * g + 1], 0));
                        total += scale[c][g] * da[g] * (float)(dot - 32 * sums[g]);
                    }
                    acc[r][c] += total;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
#pragma unroll
        for (int c = 0; c < Q6K_COLS; c++) {
            float sum = acc[r][c];
#pragma unroll
            for (int offset_ = 16; offset_ > 0; offset_ >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset_);
            }
            if (lane == 0 && r < (int)rows && (unsigned int)c < cols) {
                unsigned long long row = first_row + r;
                unsigned long long col = col0 + c;
                if (addend_mode == 1) {
                    sum += addend[col];
                } else if (addend_mode == 2) {
                    sum += addend[row * n + col];
                }
                out[row * n + col] = sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Packed `Q2_K` and `Q3_K` weights (`execute-direct-quantized-weights`),
// read with int8 activations only (a matmul that cannot quantize its
// activations dequantizes the weight for the call). Within a super-block,
// value `128 h + 32 j + 16 p + l` (h = 0..1, j = 0..3, p = 0..1,
// l = 0..15) is bits `2 j` of quant byte `32 h + 16 p + l`, under
// sub-block `8 h + 2 j + p`.
//
// `Q2_K` (format 0) on the Device: 80 bytes per super-block -- sixteen
// scale bytes (4-bit scale low, 4-bit min high) then 64 quant bytes -- and
// its `d` and `dmin` as two f32 apart; `value = d * scale * q - dmin * min`.
// `Q3_K` (format 1): 112 bytes -- 32 high-bit mask bytes, 64 quant bytes,
// then the sixteen 6-bit scales unpacked to signed bytes with their -32
// applied -- and `d` as an f32 apart; `value = d * scale * (q - 4)` with
// `q` taking bit `4 h + j` of mask byte `16 p + l` as its third bit.

#define LOWK_Q2K 0
#define LOWK_Q3K 1

__device__ __forceinline__ unsigned int lowk_block_bytes(int format) {
    return format == LOWK_Q2K ? 80u : 112u;
}

__device__ __forceinline__ unsigned int lowk_scale_floats(int format) {
    return format == LOWK_Q2K ? 2u : 1u;
}

// The logical matrix as f32, one thread per stored element.
extern "C" __global__ void lowk_dequantize_kernel(
    int format,
    const unsigned char* blocks,
    const float* dscales,
    float* out,
    unsigned long long stored_rows,
    unsigned long long stored_cols,
    unsigned int transposed
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= stored_rows * stored_cols) {
        return;
    }
    unsigned long long r = idx / stored_cols;
    unsigned long long c = idx % stored_cols;
    unsigned long long super_block = r * (stored_cols / 256) + c / 256;
    const unsigned char* block = blocks + super_block * lowk_block_bytes(format);
    const float* ds = dscales + super_block * lowk_scale_floats(format);
    unsigned int w = (unsigned int)(c % 256);
    unsigned int h = w / 128, j = (w % 128) / 32, p = (w % 32) / 16, l = w % 16;
    float value;
    if (format == LOWK_Q2K) {
        unsigned int q = (block[16 + 32 * h + 16 * p + l] >> (2 * j)) & 3;
        unsigned char sc = block[8 * h + 2 * j + p];
        value = ds[0] * (float)(sc & 0xF) * (float)q - ds[1] * (float)(sc >> 4);
    } else {
        unsigned int q = (block[32 + 32 * h + 16 * p + l] >> (2 * j)) & 3;
        unsigned int high = (block[16 * p + l] >> (4 * h + j)) & 1;
        int scale = (int)(signed char)block[96 + 8 * h + 2 * j + p];
        value = ds[0] * (float)scale * (float)((int)(q | (high << 2)) - 4);
    }
    if (transposed) {
        out[c * stored_rows + r] = value;
    } else {
        out[idx] = value;
    }
}

// One packed column's contribution for this lane's 32 values (`h`, `j`:
// values `128 h + 32 j ..`, one activation block) against their int8
// activations `x0` (first 16) and `x1`, scale `da`.
__device__ __forceinline__ float lowk_lane_dot_int8(
    int format,
    const unsigned char* block,
    const float* ds,
    unsigned int h,
    unsigned int j,
    int4 x0,
    int4 x1,
    float da
) {
    const int* xa = reinterpret_cast<const int*>(&x0);
    const int* xb = reinterpret_cast<const int*>(&x1);
    int sum0 = 0, sum1 = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        sum0 += i8_sum4(xa[i]);
        sum1 += i8_sum4(xb[i]);
    }
    if (format == LOWK_Q2K) {
        int4 qa = *reinterpret_cast<const int4*>(block + 16 + 32 * h);
        int4 qb = *reinterpret_cast<const int4*>(block + 16 + 32 * h + 16);
        const int* wa = reinterpret_cast<const int*>(&qa);
        const int* wb = reinterpret_cast<const int*>(&qb);
        int dot0 = 0, dot1 = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            dot0 = i8_dot4((wa[i] >> (2 * j)) & 0x03030303, xa[i], dot0);
            dot1 = i8_dot4((wb[i] >> (2 * j)) & 0x03030303, xb[i], dot1);
        }
        unsigned char sc0 = block[8 * h + 2 * j];
        unsigned char sc1 = block[8 * h + 2 * j + 1];
        return da * (ds[0] * ((float)(sc0 & 0xF) * (float)dot0 + (float)(sc1 & 0xF) * (float)dot1)
            - ds[1] * ((float)(sc0 >> 4) * (float)sum0 + (float)(sc1 >> 4) * (float)sum1));
    }
    int4 qa = *reinterpret_cast<const int4*>(block + 32 + 32 * h);
    int4 qb = *reinterpret_cast<const int4*>(block + 32 + 32 * h + 16);
    int4 ma = *reinterpret_cast<const int4*>(block);
    int4 mb = *reinterpret_cast<const int4*>(block + 16);
    const int* wa = reinterpret_cast<const int*>(&qa);
    const int* wb = reinterpret_cast<const int*>(&qb);
    const int* ha = reinterpret_cast<const int*>(&ma);
    const int* hb = reinterpret_cast<const int*>(&mb);
    unsigned int bit = 4 * h + j;
    int dot0 = 0, dot1 = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int va = ((wa[i] >> (2 * j)) & 0x03030303) | ((((unsigned int)ha[i] >> bit) & 0x01010101) << 2);
        int vb = ((wb[i] >> (2 * j)) & 0x03030303) | ((((unsigned int)hb[i] >> bit) & 0x01010101) << 2);
        dot0 = i8_dot4(va, xa[i], dot0);
        dot1 = i8_dot4(vb, xb[i], dot1);
    }
    float sc0 = (float)(signed char)block[96 + 8 * h + 2 * j];
    float sc1 = (float)(signed char)block[96 + 8 * h + 2 * j + 1];
    return da * ds[0] * (sc0 * (float)(dot0 - 4 * sum0) + sc1 * (float)(dot1 - 4 * sum1));
}

// Output columns one warp of `lowk_matmul_rows_int8_kernel` computes.
#define LOWK_COLS 2

// out[r, c] = sum_k a[r, k] * W[k, c] (+ addend) for a packed `Q2_K` or
// `Q3_K` weight stored transposed, over int8 activations: one warp per
// LOWK_COLS columns, eight lanes per super-block, a lane's 32 values one
// activation block.
extern "C" __global__ void lowk_matmul_rows_int8_kernel(
    int format,
    const signed char* aq,
    const float* ad,
    const unsigned char* blocks,
    const float* dscales,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    const float* addend,
    int addend_mode
) {
    unsigned long long col0 =
        ((unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y) * LOWK_COLS;
    unsigned long long first_row = (unsigned long long)blockIdx.y * Q8_MATMUL_ROWS;
    if (col0 >= n || first_row >= m) {
        return;
    }
    unsigned int rows = (unsigned int)min((unsigned long long)Q8_MATMUL_ROWS, m - first_row);
    unsigned int cols = (unsigned int)min((unsigned long long)LOWK_COLS, n - col0);
    unsigned int lane = threadIdx.x;
    unsigned long long super_blocks = k / 256;
    unsigned int part = lane % 8;
    unsigned int h = part / 4;
    unsigned int j = part % 4;
    unsigned int bytes = lowk_block_bytes(format);
    unsigned int floats = lowk_scale_floats(format);
    float acc[Q8_MATMUL_ROWS][LOWK_COLS];
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
#pragma unroll
        for (int c = 0; c < LOWK_COLS; c++) {
            acc[r][c] = 0.0f;
        }
    }
    for (unsigned long long sb = lane / 8; sb < super_blocks; sb += 4) {
        unsigned long long base = sb * 256 + 128 * h + 32 * j;
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                unsigned long long at = (first_row + r) * k + base;
                int4 x0 = *reinterpret_cast<const int4*>(aq + at);
                int4 x1 = *reinterpret_cast<const int4*>(aq + at + 16);
                float da = ad[at >> 5];
#pragma unroll
                for (int c = 0; c < LOWK_COLS; c++) {
                    unsigned int column = (unsigned int)c < cols ? c : 0;
                    unsigned long long index = (col0 + column) * super_blocks + sb;
                    acc[r][c] += lowk_lane_dot_int8(
                        format, blocks + index * bytes, dscales + index * floats, h, j, x0, x1, da);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
#pragma unroll
        for (int c = 0; c < LOWK_COLS; c++) {
            float sum = acc[r][c];
#pragma unroll
            for (int offset_ = 16; offset_ > 0; offset_ >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset_);
            }
            if (lane == 0 && r < (int)rows && (unsigned int)c < cols) {
                unsigned long long row = first_row + r;
                unsigned long long col = col0 + c;
                if (addend_mode == 1) {
                    sum += addend[col];
                } else if (addend_mode == 2) {
                    sum += addend[row * n + col];
                }
                out[row * n + col] = sum;
            }
        }
    }
}

// `silu(a x gate) * (a x up)` for two packed `Q2_K`/`Q3_K` weights of one
// format, stored transposed, over int8 activations.
extern "C" __global__ void lowk_gated_matmul_rows_int8_kernel(
    int format,
    const signed char* aq,
    const float* ad,
    const unsigned char* gate_blocks,
    const float* gate_dscales,
    const unsigned char* up_blocks,
    const float* up_dscales,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n
) {
    unsigned long long col = (unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y;
    unsigned long long first_row = (unsigned long long)blockIdx.y * Q8_MATMUL_ROWS;
    if (col >= n || first_row >= m) {
        return;
    }
    unsigned int rows = (unsigned int)min((unsigned long long)Q8_MATMUL_ROWS, m - first_row);
    unsigned int lane = threadIdx.x;
    unsigned long long super_blocks = k / 256;
    unsigned int part = lane % 8;
    unsigned int h = part / 4;
    unsigned int j = part % 4;
    unsigned int bytes = lowk_block_bytes(format);
    unsigned int floats = lowk_scale_floats(format);
    float gate_acc[Q8_MATMUL_ROWS];
    float up_acc[Q8_MATMUL_ROWS];
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        gate_acc[r] = 0.0f;
        up_acc[r] = 0.0f;
    }
    for (unsigned long long sb = lane / 8; sb < super_blocks; sb += 4) {
        unsigned long long index = col * super_blocks + sb;
        unsigned long long base = sb * 256 + 128 * h + 32 * j;
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                unsigned long long at = (first_row + r) * k + base;
                int4 x0 = *reinterpret_cast<const int4*>(aq + at);
                int4 x1 = *reinterpret_cast<const int4*>(aq + at + 16);
                float da = ad[at >> 5];
                gate_acc[r] += lowk_lane_dot_int8(format, gate_blocks + index * bytes,
                    gate_dscales + index * floats, h, j, x0, x1, da);
                up_acc[r] += lowk_lane_dot_int8(format, up_blocks + index * bytes,
                    up_dscales + index * floats, h, j, x0, x1, da);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        float g = gate_acc[r];
        float u = up_acc[r];
#pragma unroll
        for (int offset_ = 16; offset_ > 0; offset_ >>= 1) {
            g += __shfl_xor_sync(0xffffffffu, g, offset_);
            u += __shfl_xor_sync(0xffffffffu, u, offset_);
        }
        if (lane == 0 && r < (int)rows) {
            out[(first_row + r) * n + col] = g * (1.0f / (1.0f + expf(-g))) * u;
        }
    }
}
