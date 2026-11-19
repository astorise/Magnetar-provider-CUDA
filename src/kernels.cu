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

// The tanh approximation of GELU, in the Reference CPU's order of
// operations (`gelu`, `operator_semantics::gelu_tanh`).
__device__ __forceinline__ float gelu_tanh(float x) {
    const float sqrt_2_over_pi = 0.7978846f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

extern "C" __global__ void gelu_tanh_kernel(const float* x, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = gelu_tanh(x[i]);
    }
}

// `activation` with `kind = "scale"` (and a scaled `embedding`): x * factor.
extern "C" __global__ void scale_kernel(const float* x, float* out, float factor, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] * factor;
    }
}

// `activation` with `kind = "softcap"`: cap * tanh(x / cap).
extern "C" __global__ void softcap_kernel(const float* x, float* out, float cap, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = cap * tanhf(x[i] / cap);
    }
}

// `mul` by a `[1]` operand, read on the Device: out[i] = a[i] * b[0].
extern "C" __global__ void mul_scalar_kernel(const float* a, const float* b, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[0];
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

// The stored element a dequantization thread reads. Without `row_ids`,
// thread `idx` reads stored element `idx`. With them (an embedding lookup
// of a table stored untransposed), thread `idx` reads column
// `idx % stored_cols` of the stored row id `idx / stored_cols` names, so
// only the looked-up rows are dequantized; an id that is negative, not an
// integer or not below `stored_rows` sets `invalid_id_flag` and its output
// row is left unwritten (as `embedding_lookup_kernel`).
__device__ __forceinline__ bool packed_element(
    unsigned long long idx,
    unsigned long long stored_rows,
    unsigned long long stored_cols,
    const float* row_ids,
    unsigned long long id_count,
    int* invalid_id_flag,
    unsigned long long* r,
    unsigned long long* c
) {
    if (!row_ids) {
        if (idx >= stored_rows * stored_cols) {
            return false;
        }
        *r = idx / stored_cols;
        *c = idx % stored_cols;
        return true;
    }
    if (idx >= id_count * stored_cols) {
        return false;
    }
    float raw_id = row_ids[idx / stored_cols];
    if (raw_id < 0.0f || raw_id != floorf(raw_id)
        || (unsigned long long)raw_id >= stored_rows) {
        atomicExch(invalid_id_flag, 1);
        return false;
    }
    *r = (unsigned long long)raw_id;
    *c = idx % stored_cols;
    return true;
}

// One thread per stored value: the logical matrix as f32, the stored one
// or (`transposed`) its transpose.
extern "C" __global__ void q8_dequantize_kernel(
    const signed char* quants,
    const float* scales,
    float* out,
    unsigned long long stored_rows,
    unsigned long long stored_cols,
    unsigned int transposed,
    // An embedding lookup's ids, or null (`packed_element`).
    const float* row_ids,
    unsigned long long id_count,
    int* invalid_id_flag
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long r, c;
    if (!packed_element(idx, stored_rows, stored_cols, row_ids, id_count, invalid_id_flag, &r, &c)) {
        return;
    }
    unsigned long long stored = r * stored_cols + c;
    float value = (float)quants[stored] * scales[stored >> 5];
    if (transposed) {
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
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
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
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
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
    unsigned long long expert_count,
    // `[experts]`: each chosen expert's weight is multiplied by its entry,
    // after the renormalization (`moe-experts`' sixth input); null for none.
    const float* per_expert_scale
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
    if (per_expert_scale && lane == 0) {
        for (unsigned long long k = 0; k < top_k; k++) {
            int local = indices[t * top_k + k];
            if (local >= 0) {
                weights[t * top_k + k] *= per_expert_scale[expert_first + (unsigned long long)local];
            }
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
    unsigned long long inter,
    // 1: the gate takes GELU (tanh) instead of SiLU (`moe-experts.activation`).
    int gelu
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
        act[tk * inter + j] = (gelu ? gelu_tanh(g) : g * (1.0f / (1.0f + expf(-g)))) * u;
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

// The grouped form of the two kernels above, for several tokens (a
// prefill): the chosen pairs `[pair_first, pair_first + pairs)` are grouped
// by expert (`order`: each expert's pairs in a run, `offsets[e]` its first
// slot, `tiles[e]` its first tile of MOE_GROUP_ROWS pairs, `tiles[experts]`
// their count), and a block serves one tile: it reads its expert's weights
// once for every pair of the tile instead of once per pair. Each pair's
// value takes the same lane partition, order and reduction as in the
// per-pair kernels, so the results are theirs; the order of the pairs
// within an expert's run does not matter.
#define MOE_GROUP_ROWS 8

// counts[e] = the pairs choosing local expert e (zeroed beforehand).
extern "C" __global__ void moe_group_count_kernel(
    const int* indices,
    unsigned int* counts,
    unsigned long long pair_first,
    unsigned long long pairs
) {
    unsigned long long p = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) {
        return;
    }
    int e = indices[pair_first + p];
    if (e >= 0) {
        atomicAdd(&counts[e], 1u);
    }
}

// One thread: each expert's first slot and first tile (and a cursor at its
// first slot), then the slots' and tiles' totals at index `experts`.
extern "C" __global__ void moe_group_offsets_kernel(
    const unsigned int* counts,
    unsigned int* offsets,
    unsigned int* tiles,
    unsigned int* cursor,
    unsigned long long experts
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    unsigned int offset = 0;
    unsigned int tile = 0;
    for (unsigned long long e = 0; e < experts; e++) {
        offsets[e] = offset;
        cursor[e] = offset;
        tiles[e] = tile;
        offset += counts[e];
        tile += (counts[e] + MOE_GROUP_ROWS - 1) / MOE_GROUP_ROWS;
    }
    offsets[experts] = offset;
    tiles[experts] = tile;
}

// order[slot] = a pair's index (among every token's pairs), each pair in
// its expert's run.
extern "C" __global__ void moe_group_scatter_kernel(
    const int* indices,
    unsigned int* cursor,
    unsigned int* order,
    unsigned long long pair_first,
    unsigned long long pairs
) {
    unsigned long long p = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) {
        return;
    }
    int e = indices[pair_first + p];
    if (e >= 0) {
        unsigned int slot = atomicAdd(&cursor[e], 1u);
        order[slot] = (unsigned int)(pair_first + p);
    }
}

// Tile `tile`'s expert, first slot and pair count; false past the last
// tile. The expert is the last whose first tile is at most `tile` (it has
// pairs: the next expert's first tile is past `tile`).
__device__ __forceinline__ bool moe_group_tile(
    const unsigned int* offsets,
    const unsigned int* tiles,
    unsigned long long experts,
    unsigned int tile,
    unsigned int* expert,
    unsigned int* first,
    unsigned int* rows
) {
    if (tile >= tiles[experts]) {
        return false;
    }
    unsigned int low = 0;
    unsigned int high = (unsigned int)experts;
    while (high - low > 1) {
        unsigned int middle = (low + high) / 2;
        if (tiles[middle] <= tile) {
            low = middle;
        } else {
            high = middle;
        }
    }
    *expert = low;
    *first = offsets[low] + (tile - tiles[low]) * MOE_GROUP_ROWS;
    unsigned int end = offsets[low + 1];
    *rows = end - *first < MOE_GROUP_ROWS ? end - *first : MOE_GROUP_ROWS;
    return true;
}

// Output columns one warp of the grouped kernels below serves: each lane
// converts its quants of every column once per step and reads its x values
// of a pair once for every column.
#define MOE_GROUP_COLS 2

// `q8_dot16` over quants already converted to floats, and x values already
// loaded: the same products added in the same order (each quant converts
// exactly).
__device__ __forceinline__ float moe_dot16(const float* w, const float* x) {
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; j += 4) {
        s += x[j] * w[j] + x[j + 1] * w[j + 1] + x[j + 2] * w[j + 2] + x[j + 3] * w[j + 3];
    }
    return s;
}

__device__ __forceinline__ void moe_quants16(const signed char* quants, float* w) {
    int4 q = *reinterpret_cast<const int4*>(quants);
    const signed char* b = reinterpret_cast<const signed char*>(&q);
#pragma unroll
    for (int j = 0; j < 16; j++) {
        w[j] = (float)b[j];
    }
}

__device__ __forceinline__ void moe_values16(const float* source, float* x) {
#pragma unroll
    for (int j = 0; j < 16; j += 4) {
        float4 v = *reinterpret_cast<const float4*>(source + j);
        x[j] = v.x;
        x[j + 1] = v.y;
        x[j + 2] = v.z;
        x[j + 3] = v.w;
    }
}

// `moe_q8_gated_kernel` over a tile, MOE_GROUP_COLS columns per warp:
// act[(tk - pair_first) * inter + j].
extern "C" __global__ void moe_q8_gated_grouped_kernel(
    const float* h,
    const unsigned int* order,
    const unsigned int* offsets,
    const unsigned int* tiles,
    unsigned long long experts,
    const signed char* gate_quants,
    const float* gate_scales,
    const signed char* up_quants,
    const float* up_scales,
    float* act,
    unsigned long long pair_first,
    unsigned long long top_k,
    unsigned long long hidden,
    unsigned long long inter,
    int gelu
) {
    unsigned long long j0 =
        ((unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y) * MOE_GROUP_COLS;
    unsigned int expert, first, rows;
    if (j0 >= inter || !moe_group_tile(offsets, tiles, experts, blockIdx.y, &expert, &first, &rows)) {
        return;
    }
    const signed char* wg[MOE_GROUP_COLS];
    const signed char* wu[MOE_GROUP_COLS];
    const float* sg[MOE_GROUP_COLS];
    const float* su[MOE_GROUP_COLS];
#pragma unroll
    for (int c = 0; c < MOE_GROUP_COLS; c++) {
        // A column past the last reads the first's weights, unwritten.
        unsigned long long j = j0 + c < inter ? j0 + c : j0;
        unsigned long long row = (unsigned long long)expert * inter + j;
        wg[c] = gate_quants + row * hidden;
        wu[c] = up_quants + row * hidden;
        sg[c] = gate_scales + row * (hidden >> 5);
        su[c] = up_scales + row * (hidden >> 5);
    }
    unsigned int tk[MOE_GROUP_ROWS];
    float g[MOE_GROUP_COLS][MOE_GROUP_ROWS];
    float u[MOE_GROUP_COLS][MOE_GROUP_ROWS];
#pragma unroll
    for (int r = 0; r < MOE_GROUP_ROWS; r++) {
        tk[r] = order[first + (r < rows ? r : 0)];
#pragma unroll
        for (int c = 0; c < MOE_GROUP_COLS; c++) {
            g[c][r] = 0.0f;
            u[c][r] = 0.0f;
        }
    }
    for (unsigned long long i = (unsigned long long)threadIdx.x * Q8_LANE_QUANTS; i < hidden;
         i += Q8_WARP_QUANTS) {
        float gate_w[MOE_GROUP_COLS][16];
        float up_w[MOE_GROUP_COLS][16];
        float gate_scale[MOE_GROUP_COLS];
        float up_scale[MOE_GROUP_COLS];
#pragma unroll
        for (int c = 0; c < MOE_GROUP_COLS; c++) {
            moe_quants16(wg[c] + i, gate_w[c]);
            moe_quants16(wu[c] + i, up_w[c]);
            gate_scale[c] = sg[c][i >> 5];
            up_scale[c] = su[c][i >> 5];
        }
#pragma unroll
        for (int r = 0; r < MOE_GROUP_ROWS; r++) {
            if (r < rows) {
                float x[16];
                moe_values16(h + (unsigned long long)(tk[r] / top_k) * hidden + i, x);
#pragma unroll
                for (int c = 0; c < MOE_GROUP_COLS; c++) {
                    g[c][r] += gate_scale[c] * moe_dot16(gate_w[c], x);
                    u[c][r] += up_scale[c] * moe_dot16(up_w[c], x);
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < MOE_GROUP_COLS; c++) {
#pragma unroll
        for (int r = 0; r < MOE_GROUP_ROWS; r++) {
            float gr = g[c][r];
            float ur = u[c][r];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                gr += __shfl_xor_sync(0xffffffffu, gr, offset);
                ur += __shfl_xor_sync(0xffffffffu, ur, offset);
            }
            if (threadIdx.x == 0 && r < rows && j0 + c < inter) {
                act[((unsigned long long)tk[r] - pair_first) * inter + j0 + c] =
                    (gelu ? gelu_tanh(gr) : gr * (1.0f / (1.0f + expf(-gr)))) * ur;
            }
        }
    }
}

// `moe_q8_down_kernel`'s per-pair sums over a tile, MOE_GROUP_COLS columns
// per warp, before their weighted total:
// partial[(tk - pair_first) * hidden + c] = act[tk] . down_e[:, c].
extern "C" __global__ void moe_q8_down_grouped_kernel(
    const float* act,
    const unsigned int* order,
    const unsigned int* offsets,
    const unsigned int* tiles,
    unsigned long long experts,
    const signed char* down_quants,
    const float* down_scales,
    float* partial,
    unsigned long long pair_first,
    unsigned long long hidden,
    unsigned long long inter
) {
    unsigned long long c0 =
        ((unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y) * MOE_GROUP_COLS;
    unsigned int expert, first, rows;
    if (c0 >= hidden || !moe_group_tile(offsets, tiles, experts, blockIdx.y, &expert, &first, &rows)) {
        return;
    }
    const signed char* w[MOE_GROUP_COLS];
    const float* scales[MOE_GROUP_COLS];
#pragma unroll
    for (int c = 0; c < MOE_GROUP_COLS; c++) {
        unsigned long long column = c0 + c < hidden ? c0 + c : c0;
        unsigned long long row = (unsigned long long)expert * hidden + column;
        w[c] = down_quants + row * inter;
        scales[c] = down_scales + row * (inter >> 5);
    }
    unsigned int tk[MOE_GROUP_ROWS];
    float sum[MOE_GROUP_COLS][MOE_GROUP_ROWS];
#pragma unroll
    for (int r = 0; r < MOE_GROUP_ROWS; r++) {
        tk[r] = order[first + (r < rows ? r : 0)];
#pragma unroll
        for (int c = 0; c < MOE_GROUP_COLS; c++) {
            sum[c][r] = 0.0f;
        }
    }
    for (unsigned long long i = (unsigned long long)threadIdx.x * Q8_LANE_QUANTS; i < inter;
         i += Q8_WARP_QUANTS) {
        float weights[MOE_GROUP_COLS][16];
        float scale[MOE_GROUP_COLS];
#pragma unroll
        for (int c = 0; c < MOE_GROUP_COLS; c++) {
            moe_quants16(w[c] + i, weights[c]);
            scale[c] = scales[c][i >> 5];
        }
#pragma unroll
        for (int r = 0; r < MOE_GROUP_ROWS; r++) {
            if (r < rows) {
                float a[16];
                moe_values16(act + ((unsigned long long)tk[r] - pair_first) * inter + i, a);
#pragma unroll
                for (int c = 0; c < MOE_GROUP_COLS; c++) {
                    sum[c][r] += scale[c] * moe_dot16(weights[c], a);
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < MOE_GROUP_COLS; c++) {
#pragma unroll
        for (int r = 0; r < MOE_GROUP_ROWS; r++) {
            float value = sum[c][r];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_xor_sync(0xffffffffu, value, offset);
            }
            if (threadIdx.x == 0 && r < rows && c0 + c < hidden) {
                partial[((unsigned long long)tk[r] - pair_first) * hidden + c0 + c] = value;
            }
        }
    }
}

// out[t, c] for the tokens `[token_first, token_first + tokens)`: the
// chosen experts' partial sums weighted and added in routing order, as
// `moe_q8_down_kernel` adds them.
extern "C" __global__ void moe_down_combine_kernel(
    const float* partial,
    const int* indices,
    const float* weights,
    float* out,
    unsigned long long token_first,
    unsigned long long tokens,
    unsigned long long top_k,
    unsigned long long hidden
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tokens * hidden) {
        return;
    }
    unsigned long long local = index / hidden;
    unsigned long long c = index % hidden;
    unsigned long long t = token_first + local;
    float total = 0.0f;
    for (unsigned long long k = 0; k < top_k; k++) {
        unsigned long long tk = t * top_k + k;
        if (indices[tk] >= 0) {
            total += weights[tk] * partial[(local * top_k + k) * hidden + c];
        }
    }
    out[t * hidden + c] = total;
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
    unsigned long long experts,
    int gelu
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
    act[index] = (gelu ? gelu_tanh(g) : g * (1.0f / (1.0f + expf(-g)))) * u;
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
    // A null `weight` is a scale-less RMSNorm (scale 1).
    const float* w_row = weight ? weight + row * weight_row_stride : nullptr;
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
        out_row[i] = w_row ? in_row[i] * scale * w_row[i] : in_row[i] * scale;
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
    float original_context_length,
    // Partial rotation (`partial_rotary_factor`): pairs from this one on
    // keep frequency zero, so they pass through unchanged.
    unsigned long long rotated_pairs,
    // YaRN (`frequency_scaling = "yarn"`); `yarn_factor == 0` leaves the
    // frequencies unscaled. `yarn_low` and `yarn_high` are the correction
    // range `RopeYarnFrequencyScaling::correction_range` computes, and the
    // blend is the f32 operations of its `scale_frequency`. `amplitude`
    // multiplies the rotated pairs' cos and sin (`attention_factor`, 1
    // otherwise).
    float yarn_factor,
    float yarn_low,
    float yarn_high,
    float amplitude
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
    if (yarn_factor > 0.0f) {
        float high = yarn_low == yarn_high ? yarn_high + 0.001f : yarn_high;
        float ramp = ((float)pair - yarn_low) / (high - yarn_low);
        ramp = fminf(fmaxf(ramp, 0.0f), 1.0f);
        float interpolated = frequency / yarn_factor;
        frequency = interpolated * ramp + frequency * (1.0f - ramp);
    }
    float angle = pair < rotated_pairs ? position * frequency : 0.0f;
    float pair_amplitude = pair < rotated_pairs ? amplitude : 1.0f;
    float s = sinf(angle) * pair_amplitude;
    float c = cosf(angle) * pair_amplitude;
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
    unsigned long long query_position_offset,
    float score_scale,
    // Each head's V and output width (`attention.value_head_dimension`,
    // `head_dimension` otherwise) and the V row width.
    unsigned long long value_dimension,
    unsigned long long v_model_dim
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
    // The factor on QK^T: `score_scale` when the node sets one
    // (`attention.scale`), else 1/sqrt(head_dimension).
    float scale = score_scale > 0.0f ? score_scale : 1.0f / sqrtf((float)head_dimension);

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

    float* out_row = out + (query_index * head_count + head) * value_dimension;
    for (unsigned long long d = 0; d < value_dimension; d++) {
        out_row[d] = 0.0f;
    }
    for (unsigned long long key_index = key_lower; key_index < key_upper; key_index++) {
        const float* k_row = k + key_index * kv_model_dim + kv_offset;
        float dot = 0.0f;
        for (unsigned long long d = 0; d < head_dimension; d++) {
            dot += q_row[d] * k_row[d];
        }
        float weight = expf(dot * scale - max_value) / sum;
        const float* v_row = v + key_index * v_model_dim + kv_head * value_dimension;
        for (unsigned long long d = 0; d < value_dimension; d++) {
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
    const unsigned long long* step_offset,
    float score_scale,
    // As `attention_kernel`.
    unsigned long long value_dimension,
    unsigned long long v_model_dim
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
    // The factor on QK^T: `score_scale` when the node sets one
    // (`attention.scale`), else 1/sqrt(head_dimension).
    float scale = score_scale > 0.0f ? score_scale : 1.0f / sqrtf((float)head_dimension);

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
    float* out_row = out + (query_index * head_count + head) * value_dimension;
    unsigned long long v_offset = kv_head * value_dimension;
    for (unsigned long long d = tid; d < value_dimension; d += ATTENTION_THREADS) {
        float acc = 0.0f;
        for (unsigned long long j = 0; j < key_count; j++) {
            acc += scores[j] * v[(key_lower + j) * v_model_dim + v_offset + d];
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
// V rows may be narrower than K's (`value_head_dimension`,
// `express-latent-attention-forms`): a thread then owns VPART value
// dimensions, and V rows and the output are `4 * VPART` wide per head.
#define ATTENTION_TILE_QUERIES 64
#define ATTENTION_SUB_KEYS 16
template <int PART, int VPART>
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
    unsigned int key_tile,
    float score_scale
) {
    extern __shared__ float tiles[];
    const unsigned int head_dimension = 4 * PART;
    const unsigned int value_dimension = 4 * VPART;
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
    // V rows hold kv_head_count heads of value_dimension, output rows
    // head_count of them (q_model_dim and kv_model_dim when they match).
    unsigned long long v_model_dim = kv_head_count * value_dimension;
    unsigned long long v_offset = (head / group_size) * value_dimension;
    unsigned long long out_model_dim = head_count * value_dimension;
    unsigned long long out_offset = head * value_dimension;
    // The factor on QK^T: `score_scale` when the node sets one
    // (`attention.scale`), else 1/sqrt(head_dimension).
    float scale = score_scale > 0.0f ? score_scale : 1.0f / sqrtf((float)head_dimension);

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
    float acc[VPART];
#pragma unroll
    for (int i = 0; i < PART; i++) {
        q_part[i] = active ? q[query_index * q_model_dim + q_offset + part + 4 * i] * scale : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < VPART; i++) {
        acc[i] = 0.0f;
    }
    float max_value = neg_inf();
    float sum = 0.0f;

    for (unsigned long long tile = tile_lower; tile < tile_upper; tile += key_tile) {
        unsigned int tile_keys = (unsigned int)(tile_upper - tile < key_tile ? tile_upper - tile : key_tile);
        __syncthreads();
        for (unsigned int index = tid; index < tile_keys * head_dimension; index += blockDim.x) {
            unsigned int row = index / head_dimension;
            unsigned int d = index % head_dimension;
            k_tile[index] = k[(tile + row) * kv_model_dim + kv_offset + d];
        }
        for (unsigned int index = tid; index < tile_keys * value_dimension; index += blockDim.x) {
            unsigned int row = index / value_dimension;
            unsigned int d = index % value_dimension;
            v_tile[index] = v[(tile + row) * v_model_dim + v_offset + d];
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
            float sub_mean[VPART];
#pragma unroll
            for (int i = 0; i < VPART; i++) {
                sub_mean[i] = 0.0f;
            }
#pragma unroll
            for (int j = 0; j < ATTENTION_SUB_KEYS; j++) {
                float weight = scores[j] / sub_sum;
                if (weight != 0.0f) {
                    const float* v_row = v_tile + (sub + j) * value_dimension + part;
#pragma unroll
                    for (int i = 0; i < VPART; i++) {
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
            for (int i = 0; i < VPART; i++) {
                acc[i] = acc[i] * keep + sub_mean[i] * take;
            }
            max_value = new_max;
        }
    }
    if (active) {
        // No key admitted (an empty window) leaves the output zero.
#pragma unroll
        for (int i = 0; i < VPART; i++) {
            out[query_index * out_model_dim + out_offset + part + 4 * i] = sum > 0.0f ? acc[i] : 0.0f;
        }
    }
}

#define ATTENTION_TILED_KERNEL(PART) ATTENTION_TILED_VALUED_KERNEL(PART, PART, PART)
#define ATTENTION_TILED_VALUED_KERNEL(NAME, PART, VPART)                          \
extern "C" __global__ void attention_tiled_##NAME##_kernel(                       \
    const float* q, const float* k, const float* v, float* out,                   \
    unsigned long long seq_len, unsigned long long kv_seq_len,                    \
    unsigned long long head_count, unsigned long long kv_head_count,              \
    unsigned long long q_model_dim, unsigned long long kv_model_dim,              \
    int causal, long long window_size,                                            \
    unsigned long long query_position_offset, unsigned int key_tile,             \
    float score_scale) {                                                          \
    attention_tiled<PART, VPART>(q, k, v, out, seq_len, kv_seq_len, head_count,   \
        kv_head_count, q_model_dim, kv_model_dim, causal, window_size,            \
        query_position_offset, key_tile, score_scale);                            \
}
ATTENTION_TILED_KERNEL(1)
ATTENTION_TILED_KERNEL(2)
ATTENTION_TILED_KERNEL(4)
ATTENTION_TILED_KERNEL(8)
ATTENTION_TILED_KERNEL(16)
ATTENTION_TILED_KERNEL(24)
ATTENTION_TILED_KERNEL(32)
// 192-dimension keys, 128-dimension values (an expanded latent attention).
ATTENTION_TILED_VALUED_KERNEL(48_32, 48, 32)

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
    const unsigned long long* step_offset,
    float score_scale,
    // Each head's V width (its partial holds `value_dimension + 2` floats)
    // and the V row width, as `attention_kernel`.
    unsigned long long value_dimension,
    unsigned long long v_model_dim
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
    // The factor on QK^T: `score_scale` when the node sets one
    // (`attention.scale`), else 1/sqrt(head_dimension).
    float scale = score_scale > 0.0f ? score_scale : 1.0f / sqrtf((float)head_dimension);

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
    float* out = partial + (head * chunks + chunk) * (value_dimension + 2);
    if (first >= last) {
        for (unsigned long long d = tid; d < value_dimension; d += ATTENTION_SPLIT_THREADS) {
            out[d] = 0.0f;
        }
        if (tid == 0) {
            out[value_dimension] = neg_inf();
            out[value_dimension + 1] = 0.0f;
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

    // Each KV head's V starts `v_model_dim / kv_head_count` values after the
    // last: its value width, or its stride when the values are each head's
    // leading `value_dimension` of a wider V head (`value_head_stride`).
    unsigned long long v_offset = (head / group_size) * (v_model_dim / kv_head_count);
    for (unsigned long long d = tid; d < value_dimension; d += ATTENTION_SPLIT_THREADS) {
        float acc = 0.0f;
        for (unsigned long long j = 0; j < count; j++) {
            acc += scores[j] * v[(first + j) * v_model_dim + v_offset + d];
        }
        out[d] = acc;
    }
    if (tid == 0) {
        out[value_dimension] = max_value;
        out[value_dimension + 1] = sum;
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
    const unsigned long long* step_offset,
    float score_scale
) {
    // K rows padded to D + 4 floats: whole float4s, and the eight threads
    // of each quarter-warp phase of a float4 load read distinct banks.
    // The weights are key-major, the group's heads contiguous, so a thread
    // loads four heads' weights of a key at once
    // (`speed-up-decode-attention-on-codes`, decision 5).
    __shared__ __align__(16) float k_s[KEYS][D + 4];
    __shared__ __align__(16) float v_s[KEYS][D];
    __shared__ __align__(16) float q_s[ATTENTION_GQA_MAX_GROUP][D];
    __shared__ __align__(16) float p_s[KEYS][ATTENTION_GQA_MAX_GROUP];
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
    // The factor on QK^T: `score_scale` when the node sets one
    // (`attention.scale`), else 1/sqrt(D).
    // 1/sqrt(D) is correctly rounded (exact for 64 dimensions).
    const float scale = score_scale > 0.0f ? score_scale : 1.0f / sqrtf((float)D);

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

    // Scores: thread (j, h) takes key j against SCORE_HEADS heads from
    // h * SCORE_HEADS on, each float4 of the key's row feeding every head
    // (a warp's threads share their heads, so the queries' loads are
    // broadcasts). Each dot product is summed in the order of the
    // dimensions, as one thread per (head, key) did.
    constexpr int SCORE_HEADS = ATTENTION_GQA_MAX_GROUP * KEYS / ATTENTION_GQA_THREADS;
    {
        unsigned int j = tid % KEYS;
        unsigned int first_score_head = (tid / KEYS) * SCORE_HEADS;
        float dot[SCORE_HEADS];
#pragma unroll
        for (int h = 0; h < SCORE_HEADS; h++) {
            dot[h] = 0.0f;
        }
        const float4* k_row = (const float4*)k_s[j];
#pragma unroll 4
        for (int c = 0; c < D / 4; c++) {
            float4 key = k_row[c];
#pragma unroll
            for (int h = 0; h < SCORE_HEADS; h++) {
                float4 query = ((const float4*)q_s[first_score_head + h])[c];
                dot[h] = fmaf(query.x, key.x, dot[h]);
                dot[h] = fmaf(query.y, key.y, dot[h]);
                dot[h] = fmaf(query.z, key.z, dot[h]);
                dot[h] = fmaf(query.w, key.w, dot[h]);
            }
        }
#pragma unroll
        for (int h = 0; h < SCORE_HEADS; h++) {
            unsigned int g = first_score_head + h;
            if (g < group) {
                p_s[j][g] = j < count ? dot[h] * scale : neg_inf();
            }
        }
    }
    __syncthreads();

    // Softmax per head: a warp per head, KEYS / 32 keys per lane.
    for (unsigned int g = warp; g < group; g += ATTENTION_GQA_THREADS / 32) {
        float score[KEYS / 32];
        float max_value = neg_inf();
#pragma unroll
        for (int i = 0; i < KEYS / 32; i++) {
            score[i] = p_s[lane + 32 * i][g];
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
            p_s[lane + 32 * i][g] = score[i] / sum;
        }
        if (lane == 0) {
            stats[g][0] = max_value;
            stats[g][1] = sum;
        }
    }
    __syncthreads();

    // Weighted sums of V: thread (d, h) takes coordinate d of VALUE_HEADS
    // heads from h * VALUE_HEADS on, each V value feeding every head and
    // each float4 of weights four heads. Summed in key order, as one thread
    // per (head, coordinate) did.
    constexpr int VALUE_HEADS = ATTENTION_GQA_MAX_GROUP * D / ATTENTION_GQA_THREADS;
    {
        unsigned int d = tid % D;
        unsigned int first_value_head = (tid / D) * VALUE_HEADS;
        float acc[VALUE_HEADS];
#pragma unroll
        for (int h = 0; h < VALUE_HEADS; h++) {
            acc[h] = 0.0f;
        }
#pragma unroll 8
        for (int j = 0; j < KEYS; j++) {
            float value = v_s[j][d];
#pragma unroll
            for (int h4 = 0; h4 < VALUE_HEADS / 4; h4++) {
                float4 weight = ((const float4*)p_s[j])[first_value_head / 4 + h4];
                acc[4 * h4] = fmaf(weight.x, value, acc[4 * h4]);
                acc[4 * h4 + 1] = fmaf(weight.y, value, acc[4 * h4 + 1]);
                acc[4 * h4 + 2] = fmaf(weight.z, value, acc[4 * h4 + 2]);
                acc[4 * h4 + 3] = fmaf(weight.w, value, acc[4 * h4 + 3]);
            }
        }
#pragma unroll
        for (int h = 0; h < VALUE_HEADS; h++) {
            unsigned int g = first_value_head + h;
            if (g < group) {
                float* out = partial + ((first_head + g) * chunks + chunk) * (D + 2);
                out[d] = acc[h];
                if (d == 0) {
                    out[D] = stats[g][0];
                    out[D + 1] = stats[g][1];
                }
            }
        }
    }
}

#define ATTENTION_GQA_ARGS                                                        \
    const float* q, const float* k, const float* v, float* partial,              \
    unsigned long long kv_seq_len, unsigned long long head_count,                \
    unsigned long long kv_head_count, unsigned long long kv_model_dim,           \
    int causal, long long window_size, unsigned long long query_position_offset, \
    unsigned long long chunks, const unsigned long long* step_offset,            \
    float score_scale
#define ATTENTION_GQA_PASS                                                        \
    q, k, v, partial, kv_seq_len, head_count, kv_head_count, kv_model_dim,       \
    causal, window_size, query_position_offset, chunks, step_offset, score_scale

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
    unsigned int transposed,
    // An embedding lookup's ids, or null (`packed_element`).
    const float* row_ids,
    unsigned long long id_count,
    int* invalid_id_flag
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long r, c;
    if (!packed_element(idx, stored_rows, stored_cols, row_ids, id_count, invalid_id_flag, &r, &c)) {
        return;
    }
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
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
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
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
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
    unsigned int transposed,
    // An embedding lookup's ids, or null (`packed_element`).
    const float* row_ids,
    unsigned long long id_count,
    int* invalid_id_flag
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long r, c;
    if (!packed_element(idx, stored_rows, stored_cols, row_ids, id_count, invalid_id_flag, &r, &c)) {
        return;
    }
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
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
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
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
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
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
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
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
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
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
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
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
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
    unsigned int transposed,
    // An embedding lookup's ids, or null (`packed_element`).
    const float* row_ids,
    unsigned long long id_count,
    int* invalid_id_flag
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long r, c;
    if (!packed_element(idx, stored_rows, stored_cols, row_ids, id_count, invalid_id_flag, &r, &c)) {
        return;
    }
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
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
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
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
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

// ---------------------------------------------------------------------------
// Packed `IQ4_NL` and `IQ4_XS` weights (`execute-direct-quantized-weights`),
// read with int8 activations only, like `Q2_K`/`Q3_K`. Both are runs of 32
// values: 16 bytes whose low nibbles index values 0..15 and high nibbles
// values 16..31 in the non-linear table below. On the Device every run's
// 16 bytes are contiguous in stored order; `IQ4_NL` (format 0) keeps one
// f32 scale per run, `IQ4_XS` (format 1) one f32 `d` per super-block of
// eight runs and, after all runs' bytes, one signed byte per run: its
// 6-bit sub-block scale minus 32. `Q4_0` (format 2) is laid out like
// `IQ4_NL`, its 4-bit quants standing for `q - 8` instead of table values.

#define IQ4_NL_FORMAT 0
#define IQ4_XS_FORMAT 1
#define Q4_0_FORMAT 2

__constant__ signed char IQ4_VALUES[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// Four table values for the four nibbles in the low halves of `nibbles`'
// bytes (each byte 0..15), from registers: `__byte_perm` picks each byte
// from the first eight table entries and from the last eight by the
// index's low three bits, and its bit 3 chooses between the two. A
// lookup in `IQ4_VALUES` instead serializes the warp whenever its lanes
// read different entries.
__device__ __forceinline__ int iq4_lookup4(unsigned int nibbles) {
    unsigned int selector = (nibbles & 0x7u) | ((nibbles >> 4) & 0x70u)
        | ((nibbles >> 8) & 0x700u) | ((nibbles >> 12) & 0x7000u);
    unsigned int low = __byte_perm(0xBFAD9881u, 0xF6EADDCFu, selector);
    unsigned int high = __byte_perm(0x26190D01u, 0x71594535u, selector);
    unsigned int upper = ((nibbles >> 3) & 0x01010101u) * 0xFFu;
    return (int)((low & ~upper) | (high & upper));
}

// Run `run`'s scale (`run` counted over the whole stored matrix).
__device__ __forceinline__ float iq4_run_scale(
    int format,
    const float* dscales,
    const signed char* subscales,
    unsigned long long run
) {
    return format == IQ4_XS_FORMAT ? dscales[run / 8] * (float)subscales[run] : dscales[run];
}

// Four values for the four nibbles in the low halves of `nibbles`' bytes:
// `q - 8` per byte for `Q4_0`, else the IQ4 table's.
__device__ __forceinline__ int iq4_values4(int format, unsigned int nibbles) {
    return format == Q4_0_FORMAT ? (int)__vsub4(nibbles, 0x08080808u) : iq4_lookup4(nibbles);
}

// The logical matrix as f32, one thread per stored element.
extern "C" __global__ void iq4_dequantize_kernel(
    int format,
    const unsigned char* qs,
    const signed char* subscales,
    const float* dscales,
    float* out,
    unsigned long long stored_rows,
    unsigned long long stored_cols,
    unsigned int transposed,
    // An embedding lookup's ids, or null (`packed_element`).
    const float* row_ids,
    unsigned long long id_count,
    int* invalid_id_flag
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long r, c;
    if (!packed_element(idx, stored_rows, stored_cols, row_ids, id_count, invalid_id_flag, &r, &c)) {
        return;
    }
    unsigned long long run = (r * stored_cols + c) / 32;
    unsigned int w = (unsigned int)(c % 32);
    unsigned char byte = qs[run * 16 + (w % 16)];
    unsigned int index = w < 16 ? (byte & 0xF) : (byte >> 4);
    float quant = format == Q4_0_FORMAT ? (float)((int)index - 8) : (float)IQ4_VALUES[index];
    float value = iq4_run_scale(format, dscales, subscales, run) * quant;
    if (transposed) {
        out[c * stored_rows + r] = value;
    } else {
        out[idx] = value;
    }
}

// One run's dot product with its 32 int8 activations (`x0` the first 16).
__device__ __forceinline__ int iq4_run_dot(int format, int4 q, int4 x0, int4 x1) {
    const unsigned int* w = reinterpret_cast<const unsigned int*>(&q);
    const int* xa = reinterpret_cast<const int*>(&x0);
    const int* xb = reinterpret_cast<const int*>(&x1);
    int dot = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        dot = i8_dot4(iq4_values4(format, w[i] & 0x0F0F0F0Fu), xa[i], dot);
        dot = i8_dot4(iq4_values4(format, (w[i] >> 4) & 0x0F0F0F0Fu), xb[i], dot);
    }
    return dot;
}

// out[r, c] = sum_k a[r, k] * W[k, c] (+ addend) for a packed `IQ4_NL`,
// `IQ4_XS` or `Q4_0` weight stored transposed, over int8 activations: one warp per
// column, one 32-value run (one activation block) per lane per step.
extern "C" __global__ void iq4_matmul_rows_int8_kernel(
    int format,
    const signed char* aq,
    const float* ad,
    const unsigned char* qs,
    const signed char* subscales,
    const float* dscales,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    const float* addend,
    int addend_mode
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
    unsigned long long col = (unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y;
    unsigned long long first_row = (unsigned long long)blockIdx.y * Q8_MATMUL_ROWS;
    if (col >= n || first_row >= m) {
        return;
    }
    unsigned int rows = (unsigned int)min((unsigned long long)Q8_MATMUL_ROWS, m - first_row);
    unsigned int lane = threadIdx.x;
    unsigned long long runs = k / 32;
    float acc[Q8_MATMUL_ROWS];
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        acc[r] = 0.0f;
    }
    for (unsigned long long u = lane; u < runs; u += 32) {
        unsigned long long run = col * runs + u;
        int4 q = *reinterpret_cast<const int4*>(qs + run * 16);
        float scale = iq4_run_scale(format, dscales, subscales, run);
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                unsigned long long at = (first_row + r) * k + u * 32;
                int4 x0 = *reinterpret_cast<const int4*>(aq + at);
                int4 x1 = *reinterpret_cast<const int4*>(aq + at + 16);
                acc[r] += scale * ad[at >> 5] * (float)iq4_run_dot(format, q, x0, x1);
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

// `silu(a x gate) * (a x up)` for two packed `IQ4_NL`/`IQ4_XS`/`Q4_0` weights of
// one format, stored transposed, over int8 activations.
extern "C" __global__ void iq4_gated_matmul_rows_int8_kernel(
    int format,
    const signed char* aq,
    const float* ad,
    const unsigned char* gate_qs,
    const signed char* gate_subscales,
    const float* gate_dscales,
    const unsigned char* up_qs,
    const signed char* up_subscales,
    const float* up_dscales,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
    unsigned long long col = (unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y;
    unsigned long long first_row = (unsigned long long)blockIdx.y * Q8_MATMUL_ROWS;
    if (col >= n || first_row >= m) {
        return;
    }
    unsigned int rows = (unsigned int)min((unsigned long long)Q8_MATMUL_ROWS, m - first_row);
    unsigned int lane = threadIdx.x;
    unsigned long long runs = k / 32;
    float gate_acc[Q8_MATMUL_ROWS];
    float up_acc[Q8_MATMUL_ROWS];
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        gate_acc[r] = 0.0f;
        up_acc[r] = 0.0f;
    }
    for (unsigned long long u = lane; u < runs; u += 32) {
        unsigned long long run = col * runs + u;
        int4 qg = *reinterpret_cast<const int4*>(gate_qs + run * 16);
        int4 qu = *reinterpret_cast<const int4*>(up_qs + run * 16);
        float sg = iq4_run_scale(format, gate_dscales, gate_subscales, run);
        float su = iq4_run_scale(format, up_dscales, up_subscales, run);
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                unsigned long long at = (first_row + r) * k + u * 32;
                int4 x0 = *reinterpret_cast<const int4*>(aq + at);
                int4 x1 = *reinterpret_cast<const int4*>(aq + at + 16);
                float da = ad[at >> 5];
                gate_acc[r] += sg * da * (float)iq4_run_dot(format, qg, x0, x1);
                up_acc[r] += su * da * (float)iq4_run_dot(format, qu, x0, x1);
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

// ---------------------------------------------------------------------------
// Packed IQ2 and IQ3 weights (`execute-direct-quantized-weights`): GGUF
// `IQ2_XXS` (format 0), `IQ2_XS` (1), `IQ2_S` (2), `IQ3_XXS` (3) and
// `IQ3_S` (4), read with int8 activations only. Values come in groups of
// eight from a fixed grid (one 8-value entry for IQ2, two 4-value entries
// for IQ3, values 1..62 one per byte), each group with a sign byte and a
// scale. On the Device each super-block keeps its bytes after the leading
// `f16` `d` (64, 72, 80, 96 and 108 bytes), `d` an f32 apart. The grids,
// generated from gguf-py's tables:
__device__ const unsigned long long IQ2XXS_GRID[256] = {
    0x0808080808080808ull, 0x080808080808082bull, 0x0808080808081919ull, 0x0808080808082b08ull,
    0x0808080808082b2bull, 0x0808080808190819ull, 0x0808080808191908ull, 0x08080808082b0808ull,
    0x08080808082b082bull, 0x08080808082b2b08ull, 0x08080808082b2b2bull, 0x0808080819080819ull,
    0x0808080819081908ull, 0x0808080819190808ull, 0x0808080819192b08ull, 0x08080808192b0819ull,
    0x08080808192b1908ull, 0x080808082b080808ull, 0x080808082b08082bull, 0x080808082b082b2bull,
    0x080808082b2b082bull, 0x0808081908080819ull, 0x0808081908081908ull, 0x0808081908190808ull,
    0x0808081908191919ull, 0x0808081919080808ull, 0x080808192b081908ull, 0x080808192b192b08ull,
    0x0808082b08080808ull, 0x0808082b0808082bull, 0x0808082b082b082bull, 0x0808082b2b08082bull,
    0x0808190808080819ull, 0x0808190808081908ull, 0x0808190808190808ull, 0x08081908082b0819ull,
    0x08081908082b1908ull, 0x0808190819080808ull, 0x080819081908082bull, 0x0808190819082b08ull,
    0x08081908192b0808ull, 0x080819082b080819ull, 0x080819082b081908ull, 0x080819082b190808ull,
    0x080819082b2b1908ull, 0x0808191908080808ull, 0x080819190808082bull, 0x0808191908082b08ull,
    0x08081919082b0808ull, 0x080819191908192bull, 0x08081919192b2b19ull, 0x080819192b080808ull,
    0x080819192b190819ull, 0x0808192b08082b19ull, 0x0808192b08190808ull, 0x0808192b19080808ull,
    0x0808192b2b081908ull, 0x0808192b2b2b1908ull, 0x08082b0808080808ull, 0x08082b0808081919ull,
    0x08082b0808082b08ull, 0x08082b0808191908ull, 0x08082b08082b2b08ull, 0x08082b0819080819ull,
    0x08082b0819081908ull, 0x08082b0819190808ull, 0x08082b081919082bull, 0x08082b082b082b08ull,
    0x08082b1908081908ull, 0x08082b1919080808ull, 0x08082b2b0808082bull, 0x08082b2b08191908ull,
    0x0819080808080819ull, 0x0819080808081908ull, 0x0819080808190808ull, 0x08190808082b0819ull,
    0x0819080819080808ull, 0x08190808192b0808ull, 0x081908082b081908ull, 0x081908082b190808ull,
    0x081908082b191919ull, 0x0819081908080808ull, 0x0819081908082b08ull, 0x08190819082b0808ull,
    0x0819081919190808ull, 0x0819081919192b2bull, 0x081908192b080808ull, 0x0819082b082b1908ull,
    0x0819082b19081919ull, 0x0819190808080808ull, 0x0819190808082b08ull, 0x08191908082b0808ull,
    0x08191908082b1919ull, 0x0819190819082b19ull, 0x081919082b080808ull, 0x0819191908192b08ull,
    0x08191919192b082bull, 0x0819192b08080808ull, 0x0819192b0819192bull, 0x08192b0808080819ull,
    0x08192b0808081908ull, 0x08192b0808190808ull, 0x08192b0819080808ull, 0x08192b082b080819ull,
    0x08192b1908080808ull, 0x08192b1908081919ull, 0x08192b192b2b0808ull, 0x08192b2b19190819ull,
    0x082b080808080808ull, 0x082b08080808082bull, 0x082b080808082b2bull, 0x082b080819081908ull,
    0x082b0808192b0819ull, 0x082b08082b080808ull, 0x082b08082b08082bull, 0x082b0819082b2b19ull,
    0x082b081919082b08ull, 0x082b082b08080808ull, 0x082b082b0808082bull, 0x082b190808080819ull,
    0x082b190808081908ull, 0x082b190808190808ull, 0x082b190819080808ull, 0x082b19081919192bull,
    0x082b191908080808ull, 0x082b191919080819ull, 0x082b1919192b1908ull, 0x082b192b2b190808ull,
    0x082b2b0808082b08ull, 0x082b2b08082b0808ull, 0x082b2b082b191908ull, 0x082b2b2b19081908ull,
    0x1908080808080819ull, 0x1908080808081908ull, 0x1908080808190808ull, 0x1908080808192b08ull,
    0x19080808082b0819ull, 0x19080808082b1908ull, 0x1908080819080808ull, 0x1908080819082b08ull,
    0x190808081919192bull, 0x19080808192b0808ull, 0x190808082b080819ull, 0x190808082b081908ull,
    0x190808082b190808ull, 0x1908081908080808ull, 0x19080819082b0808ull, 0x19080819192b0819ull,
    0x190808192b080808ull, 0x190808192b081919ull, 0x1908082b08080819ull, 0x1908082b08190808ull,
    0x1908082b19082b08ull, 0x1908082b1919192bull, 0x1908082b192b2b08ull, 0x1908190808080808ull,
    0x1908190808082b08ull, 0x19081908082b0808ull, 0x190819082b080808ull, 0x190819082b192b19ull,
    0x190819190819082bull, 0x19081919082b1908ull, 0x1908192b08080808ull, 0x19082b0808080819ull,
    0x19082b0808081908ull, 0x19082b0808190808ull, 0x19082b0819080808ull, 0x19082b0819081919ull,
    0x19082b1908080808ull, 0x19082b1919192b08ull, 0x19082b19192b0819ull, 0x19082b192b08082bull,
    0x19082b2b19081919ull, 0x19082b2b2b190808ull, 0x1919080808080808ull, 0x1919080808082b08ull,
    0x1919080808190819ull, 0x1919080808192b19ull, 0x19190808082b0808ull, 0x191908082b080808ull,
    0x191908082b082b08ull, 0x1919081908081908ull, 0x191908191908082bull, 0x191908192b2b1908ull,
    0x1919082b2b190819ull, 0x191919082b190808ull, 0x191919082b19082bull, 0x1919191908082b2bull,
    0x1919192b08080819ull, 0x1919192b19191908ull, 0x19192b0808080808ull, 0x19192b0808190819ull,
    0x19192b0808192b19ull, 0x19192b08192b1908ull, 0x19192b1919080808ull, 0x19192b2b08082b08ull,
    0x192b080808081908ull, 0x192b080808190808ull, 0x192b080819080808ull, 0x192b0808192b2b08ull,
    0x192b081908080808ull, 0x192b081919191919ull, 0x192b082b08192b08ull, 0x192b082b192b0808ull,
    0x192b190808080808ull, 0x192b190808081919ull, 0x192b191908190808ull, 0x192b19190819082bull,
    0x192b19192b081908ull, 0x192b2b081908082bull, 0x2b08080808080808ull, 0x2b0808080808082bull,
    0x2b08080808082b2bull, 0x2b08080819080819ull, 0x2b0808082b08082bull, 0x2b08081908081908ull,
    0x2b08081908192b08ull, 0x2b08081919080808ull, 0x2b08082b08190819ull, 0x2b08190808080819ull,
    0x2b08190808081908ull, 0x2b08190808190808ull, 0x2b08190808191919ull, 0x2b08190819080808ull,
    0x2b081908192b0808ull, 0x2b08191908080808ull, 0x2b0819191908192bull, 0x2b0819192b191908ull,
    0x2b08192b08082b19ull, 0x2b08192b19080808ull, 0x2b08192b192b0808ull, 0x2b082b080808082bull,
    0x2b082b1908081908ull, 0x2b082b2b08190819ull, 0x2b19080808081908ull, 0x2b19080808190808ull,
    0x2b190808082b1908ull, 0x2b19080819080808ull, 0x2b1908082b2b0819ull, 0x2b1908190819192bull,
    0x2b1908192b080808ull, 0x2b19082b19081919ull, 0x2b19190808080808ull, 0x2b191908082b082bull,
    0x2b19190819081908ull, 0x2b19191919190819ull, 0x2b192b082b080819ull, 0x2b192b19082b0808ull,
    0x2b2b08080808082bull, 0x2b2b080819190808ull, 0x2b2b08082b081919ull, 0x2b2b081908082b19ull,
    0x2b2b082b08080808ull, 0x2b2b190808192b08ull, 0x2b2b2b0819190808ull, 0x2b2b2b1908081908ull,
};

__device__ const unsigned long long IQ2XS_GRID[512] = {
    0x0808080808080808ull, 0x080808080808082bull, 0x0808080808081919ull, 0x0808080808082b08ull,
    0x0808080808082b2bull, 0x0808080808190819ull, 0x0808080808191908ull, 0x080808080819192bull,
    0x0808080808192b19ull, 0x08080808082b0808ull, 0x08080808082b082bull, 0x08080808082b1919ull,
    0x08080808082b2b08ull, 0x0808080819080819ull, 0x0808080819081908ull, 0x080808081908192bull,
    0x0808080819082b19ull, 0x0808080819190808ull, 0x080808081919082bull, 0x0808080819191919ull,
    0x0808080819192b08ull, 0x08080808192b0819ull, 0x08080808192b1908ull, 0x080808082b080808ull,
    0x080808082b08082bull, 0x080808082b081919ull, 0x080808082b082b08ull, 0x080808082b190819ull,
    0x080808082b191908ull, 0x080808082b192b19ull, 0x080808082b2b0808ull, 0x0808081908080819ull,
    0x0808081908081908ull, 0x080808190808192bull, 0x0808081908082b19ull, 0x0808081908190808ull,
    0x080808190819082bull, 0x0808081908191919ull, 0x0808081908192b08ull, 0x0808081908192b2bull,
    0x08080819082b0819ull, 0x08080819082b1908ull, 0x0808081919080808ull, 0x080808191908082bull,
    0x0808081919081919ull, 0x0808081919082b08ull, 0x0808081919190819ull, 0x0808081919191908ull,
    0x08080819192b0808ull, 0x08080819192b2b08ull, 0x080808192b080819ull, 0x080808192b081908ull,
    0x080808192b190808ull, 0x0808082b08080808ull, 0x0808082b0808082bull, 0x0808082b08081919ull,
    0x0808082b08082b08ull, 0x0808082b08190819ull, 0x0808082b08191908ull, 0x0808082b082b0808ull,
    0x0808082b19080819ull, 0x0808082b19081908ull, 0x0808082b19190808ull, 0x0808082b19191919ull,
    0x0808082b2b080808ull, 0x0808082b2b082b2bull, 0x0808190808080819ull, 0x0808190808081908ull,
    0x080819080808192bull, 0x0808190808082b19ull, 0x0808190808190808ull, 0x080819080819082bull,
    0x0808190808191919ull, 0x0808190808192b08ull, 0x08081908082b0819ull, 0x08081908082b1908ull,
    0x0808190819080808ull, 0x080819081908082bull, 0x0808190819081919ull, 0x0808190819082b08ull,
    0x0808190819190819ull, 0x0808190819191908ull, 0x080819081919192bull, 0x08081908192b0808ull,
    0x080819082b080819ull, 0x080819082b081908ull, 0x080819082b190808ull, 0x0808191908080808ull,
    0x080819190808082bull, 0x0808191908081919ull, 0x0808191908082b08ull, 0x0808191908190819ull,
    0x0808191908191908ull, 0x08081919082b0808ull, 0x0808191919080819ull, 0x0808191919081908ull,
    0x0808191919190808ull, 0x08081919192b0819ull, 0x080819192b080808ull, 0x0808192b08080819ull,
    0x0808192b08081908ull, 0x0808192b08190808ull, 0x0808192b082b192bull, 0x0808192b19080808ull,
    0x0808192b1908082bull, 0x0808192b2b081908ull, 0x08082b0808080808ull, 0x08082b080808082bull,
    0x08082b0808081919ull, 0x08082b0808082b08ull, 0x08082b0808082b2bull, 0x08082b0808190819ull,
    0x08082b0808191908ull, 0x08082b08082b0808ull, 0x08082b08082b1919ull, 0x08082b0819080819ull,
    0x08082b0819081908ull, 0x08082b0819190808ull, 0x08082b0819192b08ull, 0x08082b082b080808ull,
    0x08082b082b2b0808ull, 0x08082b082b2b2b2bull, 0x08082b1908080819ull, 0x08082b1908081908ull,
    0x08082b1908190808ull, 0x08082b1919080808ull, 0x08082b192b080819ull, 0x08082b192b082b19ull,
    0x08082b2b08080808ull, 0x08082b2b082b0808ull, 0x08082b2b082b2b08ull, 0x08082b2b2b19192bull,
    0x08082b2b2b2b0808ull, 0x0819080808080819ull, 0x0819080808081908ull, 0x081908080808192bull,
    0x0819080808082b19ull, 0x0819080808190808ull, 0x081908080819082bull, 0x0819080808191919ull,
    0x0819080808192b08ull, 0x08190808082b0819ull, 0x08190808082b1908ull, 0x0819080819080808ull,
    0x081908081908082bull, 0x0819080819081919ull, 0x0819080819082b08ull, 0x0819080819190819ull,
    0x0819080819191908ull, 0x08190808192b0808ull, 0x08190808192b2b2bull, 0x081908082b080819ull,
    0x081908082b081908ull, 0x081908082b190808ull, 0x0819081908080808ull, 0x081908190808082bull,
    0x0819081908081919ull, 0x0819081908082b08ull, 0x0819081908190819ull, 0x0819081908191908ull,
    0x08190819082b0808ull, 0x0819081919080819ull, 0x0819081919081908ull, 0x0819081919190808ull,
    0x081908192b080808ull, 0x081908192b191908ull, 0x081908192b19192bull, 0x0819082b08080819ull,
    0x0819082b08081908ull, 0x0819082b0808192bull, 0x0819082b08190808ull, 0x0819082b19080808ull,
    0x0819082b192b0808ull, 0x0819190808080808ull, 0x081919080808082bull, 0x0819190808081919ull,
    0x0819190808082b08ull, 0x0819190808190819ull, 0x0819190808191908ull, 0x08191908082b0808ull,
    0x0819190819080819ull, 0x0819190819081908ull, 0x0819190819082b19ull, 0x0819190819190808ull,
    0x08191908192b1908ull, 0x081919082b080808ull, 0x0819191908080819ull, 0x0819191908081908ull,
    0x0819191908190808ull, 0x0819191919080808ull, 0x0819192b08080808ull, 0x0819192b08191908ull,
    0x0819192b19082b19ull, 0x08192b0808080819ull, 0x08192b0808081908ull, 0x08192b0808190808ull,
    0x08192b080819082bull, 0x08192b0819080808ull, 0x08192b0819191908ull, 0x08192b082b08192bull,
    0x08192b1908080808ull, 0x08192b1908081919ull, 0x08192b19192b192bull, 0x08192b2b19190819ull,
    0x08192b2b2b2b2b19ull, 0x082b080808080808ull, 0x082b08080808082bull, 0x082b080808081919ull,
    0x082b080808082b08ull, 0x082b080808082b2bull, 0x082b080808190819ull, 0x082b080808191908ull,
    0x082b0808082b0808ull, 0x082b080819080819ull, 0x082b080819081908ull, 0x082b080819190808ull,
    0x082b08082b080808ull, 0x082b08082b2b0808ull, 0x082b081908080819ull, 0x082b081908081908ull,
    0x082b081908190808ull, 0x082b081919080808ull, 0x082b081919082b08ull, 0x082b0819192b1919ull,
    0x082b082b08080808ull, 0x082b082b082b082bull, 0x082b082b2b080808ull, 0x082b082b2b2b2b08ull,
    0x082b190808080819ull, 0x082b190808081908ull, 0x082b190808190808ull, 0x082b1908082b2b19ull,
    0x082b190819080808ull, 0x082b191908080808ull, 0x082b191919080819ull, 0x082b19191919082bull,
    0x082b19192b192b19ull, 0x082b192b08080819ull, 0x082b192b08192b2bull, 0x082b192b2b2b192bull,
    0x082b2b0808080808ull, 0x082b2b0808082b08ull, 0x082b2b0808082b2bull, 0x082b2b08082b0808ull,
    0x082b2b0819191919ull, 0x082b2b082b082b08ull, 0x082b2b082b2b082bull, 0x082b2b19192b2b08ull,
    0x082b2b192b190808ull, 0x082b2b2b08082b08ull, 0x082b2b2b082b0808ull, 0x082b2b2b2b08082bull,
    0x082b2b2b2b082b08ull, 0x082b2b2b2b082b2bull, 0x1908080808080819ull, 0x1908080808081908ull,
    0x190808080808192bull, 0x1908080808082b19ull, 0x1908080808190808ull, 0x190808080819082bull,
    0x1908080808191919ull, 0x1908080808192b08ull, 0x19080808082b0819ull, 0x19080808082b1908ull,
    0x1908080819080808ull, 0x190808081908082bull, 0x1908080819081919ull, 0x1908080819082b08ull,
    0x1908080819082b2bull, 0x1908080819190819ull, 0x1908080819191908ull, 0x19080808192b0808ull,
    0x19080808192b1919ull, 0x190808082b080819ull, 0x190808082b081908ull, 0x190808082b190808ull,
    0x1908081908080808ull, 0x190808190808082bull, 0x1908081908081919ull, 0x1908081908082b08ull,
    0x1908081908190819ull, 0x1908081908191908ull, 0x19080819082b0808ull, 0x1908081919080819ull,
    0x1908081919081908ull, 0x1908081919190808ull, 0x190808192b080808ull, 0x190808192b081919ull,
    0x190808192b2b082bull, 0x1908082b08080819ull, 0x1908082b08081908ull, 0x1908082b08190808ull,
    0x1908082b0819082bull, 0x1908082b082b2b19ull, 0x1908082b19080808ull, 0x1908190808080808ull,
    0x190819080808082bull, 0x1908190808081919ull, 0x1908190808082b08ull, 0x1908190808190819ull,
    0x1908190808191908ull, 0x1908190808192b19ull, 0x19081908082b0808ull, 0x1908190819080819ull,
    0x1908190819081908ull, 0x1908190819190808ull, 0x190819082b080808ull, 0x190819082b191908ull,
    0x1908191908080819ull, 0x1908191908081908ull, 0x1908191908190808ull, 0x19081919082b1908ull,
    0x1908191919080808ull, 0x190819192b192b2bull, 0x1908192b08080808ull, 0x1908192b08082b2bull,
    0x1908192b19081908ull, 0x1908192b19190808ull, 0x19082b0808080819ull, 0x19082b0808081908ull,
    0x19082b0808190808ull, 0x19082b0819080808ull, 0x19082b0819081919ull, 0x19082b0819191908ull,
    0x19082b08192b082bull, 0x19082b1908080808ull, 0x19082b1908190819ull, 0x19082b1919081908ull,
    0x19082b1919190808ull, 0x19082b19192b2b19ull, 0x19082b2b08081908ull, 0x1919080808080808ull,
    0x191908080808082bull, 0x1919080808081919ull, 0x1919080808082b08ull, 0x1919080808190819ull,
    0x1919080808191908ull, 0x19190808082b0808ull, 0x19190808082b2b08ull, 0x1919080819080819ull,
    0x1919080819081908ull, 0x1919080819190808ull, 0x191908082b080808ull, 0x1919081908080819ull,
    0x1919081908081908ull, 0x1919081908190808ull, 0x1919081908191919ull, 0x1919081919080808ull,
    0x191908191908082bull, 0x1919082b08080808ull, 0x1919082b19081908ull, 0x1919082b2b2b2b2bull,
    0x1919190808080819ull, 0x1919190808081908ull, 0x1919190808190808ull, 0x19191908082b0819ull,
    0x1919190819080808ull, 0x19191908192b0808ull, 0x191919082b080819ull, 0x191919082b2b0819ull,
    0x1919191908080808ull, 0x1919191908082b08ull, 0x191919192b080808ull, 0x191919192b082b08ull,
    0x1919192b082b0819ull, 0x1919192b192b2b08ull, 0x1919192b2b2b0819ull, 0x19192b0808080808ull,
    0x19192b0808191908ull, 0x19192b0819080819ull, 0x19192b0819190808ull, 0x19192b082b192b19ull,
    0x19192b1908192b2bull, 0x19192b1919080808ull, 0x19192b191908082bull, 0x19192b2b2b081919ull,
    0x192b080808080819ull, 0x192b080808081908ull, 0x192b080808190808ull, 0x192b080819080808ull,
    0x192b080819191908ull, 0x192b0808192b082bull, 0x192b08082b08192bull, 0x192b08082b2b2b19ull,
    0x192b081908080808ull, 0x192b082b082b1908ull, 0x192b082b19082b2bull, 0x192b082b2b19082bull,
    0x192b190808080808ull, 0x192b19080819192bull, 0x192b191908190808ull, 0x192b191919080808ull,
    0x192b191919081919ull, 0x192b19192b2b1908ull, 0x192b2b0808080819ull, 0x192b2b08192b2b2bull,
    0x192b2b19082b1919ull, 0x192b2b2b0808192bull, 0x192b2b2b19191908ull, 0x192b2b2b192b082bull,
    0x2b08080808080808ull, 0x2b0808080808082bull, 0x2b08080808081919ull, 0x2b08080808082b08ull,
    0x2b08080808190819ull, 0x2b08080808191908ull, 0x2b080808082b0808ull, 0x2b080808082b2b2bull,
    0x2b08080819080819ull, 0x2b08080819081908ull, 0x2b08080819190808ull, 0x2b0808082b080808ull,
    0x2b0808082b08082bull, 0x2b0808082b2b2b08ull, 0x2b0808082b2b2b2bull, 0x2b08081908080819ull,
    0x2b08081908081908ull, 0x2b0808190808192bull, 0x2b08081908190808ull, 0x2b08081919080808ull,
    0x2b08081919190819ull, 0x2b08081919192b19ull, 0x2b08082b08080808ull, 0x2b08082b082b0808ull,
    0x2b08082b2b080808ull, 0x2b08082b2b08082bull, 0x2b08082b2b2b0808ull, 0x2b08082b2b2b2b08ull,
    0x2b08190808080819ull, 0x2b08190808081908ull, 0x2b08190808190808ull, 0x2b0819080819082bull,
    0x2b08190808191919ull, 0x2b08190819080808ull, 0x2b081908192b0808ull, 0x2b0819082b082b19ull,
    0x2b08191908080808ull, 0x2b08191919081908ull, 0x2b0819192b2b1919ull, 0x2b08192b08192b08ull,
    0x2b08192b192b2b2bull, 0x2b082b0808080808ull, 0x2b082b0808082b08ull, 0x2b082b08082b1919ull,
    0x2b082b0819192b2bull, 0x2b082b082b080808ull, 0x2b082b082b08082bull, 0x2b082b082b2b2b08ull,
    0x2b082b190808192bull, 0x2b082b2b082b082bull, 0x2b082b2b2b080808ull, 0x2b082b2b2b082b08ull,
    0x2b082b2b2b19192bull, 0x2b082b2b2b2b2b08ull, 0x2b19080808080819ull, 0x2b19080808081908ull,
    0x2b19080808190808ull, 0x2b19080819080808ull, 0x2b1908081919192bull, 0x2b1908082b081908ull,
    0x2b19081908080808ull, 0x2b190819082b082bull, 0x2b190819192b1908ull, 0x2b19082b1919192bull,
    0x2b19082b2b082b19ull, 0x2b19190808080808ull, 0x2b19190808081919ull, 0x2b19190819081908ull,
    0x2b19190819190808ull, 0x2b19190819192b08ull, 0x2b191919082b2b19ull, 0x2b1919192b190808ull,
    0x2b1919192b19082bull, 0x2b19192b19080819ull, 0x2b192b0819190819ull, 0x2b192b082b2b192bull,
    0x2b192b1919082b19ull, 0x2b192b2b08191919ull, 0x2b192b2b192b0808ull, 0x2b2b080808080808ull,
    0x2b2b08080808082bull, 0x2b2b080808082b08ull, 0x2b2b080808082b2bull, 0x2b2b0808082b0808ull,
    0x2b2b0808082b2b2bull, 0x2b2b08082b2b0808ull, 0x2b2b081919190819ull, 0x2b2b081919192b19ull,
    0x2b2b08192b2b192bull, 0x2b2b082b08080808ull, 0x2b2b082b0808082bull, 0x2b2b082b08082b08ull,
    0x2b2b082b082b2b2bull, 0x2b2b082b2b080808ull, 0x2b2b082b2b2b0808ull, 0x2b2b190819080808ull,
    0x2b2b19082b191919ull, 0x2b2b192b192b1919ull, 0x2b2b192b2b192b08ull, 0x2b2b2b0808082b2bull,
    0x2b2b2b08082b0808ull, 0x2b2b2b08082b082bull, 0x2b2b2b08082b2b08ull, 0x2b2b2b082b2b0808ull,
    0x2b2b2b082b2b2b08ull, 0x2b2b2b1908081908ull, 0x2b2b2b192b081908ull, 0x2b2b2b192b08192bull,
    0x2b2b2b2b082b2b08ull, 0x2b2b2b2b082b2b2bull, 0x2b2b2b2b2b190819ull, 0x2b2b2b2b2b2b2b2bull,
};

__device__ const unsigned long long IQ2S_GRID[1024] = {
    0x0808080808080808ull, 0x080808080808082bull, 0x0808080808081919ull, 0x0808080808082b08ull,
    0x0808080808082b2bull, 0x0808080808190819ull, 0x0808080808191908ull, 0x080808080819192bull,
    0x0808080808192b19ull, 0x08080808082b0808ull, 0x08080808082b082bull, 0x08080808082b1919ull,
    0x08080808082b2b08ull, 0x0808080819080819ull, 0x0808080819081908ull, 0x080808081908192bull,
    0x0808080819082b19ull, 0x0808080819190808ull, 0x080808081919082bull, 0x0808080819191919ull,
    0x0808080819192b08ull, 0x08080808192b0819ull, 0x08080808192b1908ull, 0x08080808192b192bull,
    0x08080808192b2b19ull, 0x080808082b080808ull, 0x080808082b08082bull, 0x080808082b081919ull,
    0x080808082b082b08ull, 0x080808082b190819ull, 0x080808082b191908ull, 0x080808082b2b0808ull,
    0x080808082b2b1919ull, 0x080808082b2b2b2bull, 0x0808081908080819ull, 0x0808081908081908ull,
    0x080808190808192bull, 0x0808081908082b19ull, 0x0808081908190808ull, 0x080808190819082bull,
    0x0808081908191919ull, 0x0808081908192b08ull, 0x08080819082b0819ull, 0x08080819082b1908ull,
    0x0808081919080808ull, 0x080808191908082bull, 0x0808081919081919ull, 0x0808081919082b08ull,
    0x0808081919190819ull, 0x0808081919191908ull, 0x080808191919192bull, 0x0808081919192b19ull,
    0x08080819192b0808ull, 0x08080819192b1919ull, 0x08080819192b2b08ull, 0x080808192b080819ull,
    0x080808192b081908ull, 0x080808192b190808ull, 0x080808192b19082bull, 0x080808192b191919ull,
    0x080808192b2b0819ull, 0x080808192b2b1908ull, 0x0808082b08080808ull, 0x0808082b0808082bull,
    0x0808082b08081919ull, 0x0808082b08082b08ull, 0x0808082b08190819ull, 0x0808082b08191908ull,
    0x0808082b082b0808ull, 0x0808082b082b2b2bull, 0x0808082b19080819ull, 0x0808082b19081908ull,
    0x0808082b1908192bull, 0x0808082b19082b19ull, 0x0808082b19190808ull, 0x0808082b19191919ull,
    0x0808082b2b080808ull, 0x0808082b2b081919ull, 0x0808082b2b082b2bull, 0x0808082b2b191908ull,
    0x0808082b2b2b082bull, 0x0808190808080819ull, 0x0808190808081908ull, 0x080819080808192bull,
    0x0808190808082b19ull, 0x0808190808190808ull, 0x080819080819082bull, 0x0808190808191919ull,
    0x0808190808192b08ull, 0x08081908082b0819ull, 0x08081908082b1908ull, 0x08081908082b192bull,
    0x08081908082b2b19ull, 0x0808190819080808ull, 0x080819081908082bull, 0x0808190819081919ull,
    0x0808190819082b08ull, 0x0808190819082b2bull, 0x0808190819190819ull, 0x0808190819191908ull,
    0x080819081919192bull, 0x0808190819192b19ull, 0x08081908192b0808ull, 0x08081908192b082bull,
    0x08081908192b1919ull, 0x080819082b080819ull, 0x080819082b081908ull, 0x080819082b08192bull,
    0x080819082b082b19ull, 0x080819082b190808ull, 0x080819082b191919ull, 0x080819082b192b08ull,
    0x080819082b2b0819ull, 0x080819082b2b1908ull, 0x0808191908080808ull, 0x080819190808082bull,
    0x0808191908081919ull, 0x0808191908082b08ull, 0x0808191908082b2bull, 0x0808191908190819ull,
    0x0808191908191908ull, 0x080819190819192bull, 0x0808191908192b19ull, 0x08081919082b0808ull,
    0x08081919082b1919ull, 0x08081919082b2b08ull, 0x0808191919080819ull, 0x0808191919081908ull,
    0x080819191908192bull, 0x0808191919082b19ull, 0x0808191919190808ull, 0x080819191919082bull,
    0x0808191919191919ull, 0x0808191919192b08ull, 0x08081919192b0819ull, 0x08081919192b1908ull,
    0x080819192b080808ull, 0x080819192b08082bull, 0x080819192b081919ull, 0x080819192b082b08ull,
    0x080819192b190819ull, 0x080819192b191908ull, 0x080819192b2b0808ull, 0x0808192b08080819ull,
    0x0808192b08081908ull, 0x0808192b0808192bull, 0x0808192b08082b19ull, 0x0808192b08190808ull,
    0x0808192b08191919ull, 0x0808192b19080808ull, 0x0808192b19081919ull, 0x0808192b19082b08ull,
    0x0808192b19190819ull, 0x0808192b19191908ull, 0x0808192b192b0808ull, 0x0808192b2b080819ull,
    0x0808192b2b081908ull, 0x0808192b2b190808ull, 0x08082b0808080808ull, 0x08082b080808082bull,
    0x08082b0808081919ull, 0x08082b0808082b08ull, 0x08082b0808190819ull, 0x08082b0808191908ull,
    0x08082b080819192bull, 0x08082b0808192b19ull, 0x08082b08082b0808ull, 0x08082b08082b1919ull,
    0x08082b08082b2b2bull, 0x08082b0819080819ull, 0x08082b0819081908ull, 0x08082b081908192bull,
    0x08082b0819082b19ull, 0x08082b0819190808ull, 0x08082b081919082bull, 0x08082b0819191919ull,
    0x08082b0819192b08ull, 0x08082b08192b0819ull, 0x08082b08192b1908ull, 0x08082b082b080808ull,
    0x08082b082b081919ull, 0x08082b082b191908ull, 0x08082b082b2b2b2bull, 0x08082b1908080819ull,
    0x08082b1908081908ull, 0x08082b1908190808ull, 0x08082b190819082bull, 0x08082b1908191919ull,
    0x08082b1908192b08ull, 0x08082b19082b0819ull, 0x08082b1919080808ull, 0x08082b1919081919ull,
    0x08082b1919082b08ull, 0x08082b1919190819ull, 0x08082b1919191908ull, 0x08082b19192b0808ull,
    0x08082b192b080819ull, 0x08082b192b190808ull, 0x08082b2b08080808ull, 0x08082b2b08190819ull,
    0x08082b2b08191908ull, 0x08082b2b082b082bull, 0x08082b2b082b2b08ull, 0x08082b2b082b2b2bull,
    0x08082b2b19190808ull, 0x08082b2b2b192b19ull, 0x0819080808080819ull, 0x0819080808081908ull,
    0x081908080808192bull, 0x0819080808082b19ull, 0x0819080808190808ull, 0x081908080819082bull,
    0x0819080808191919ull, 0x0819080808192b08ull, 0x08190808082b0819ull, 0x08190808082b1908ull,
    0x08190808082b192bull, 0x0819080819080808ull, 0x081908081908082bull, 0x0819080819081919ull,
    0x0819080819082b08ull, 0x0819080819190819ull, 0x0819080819191908ull, 0x081908081919192bull,
    0x0819080819192b19ull, 0x08190808192b0808ull, 0x08190808192b082bull, 0x08190808192b1919ull,
    0x08190808192b2b08ull, 0x081908082b080819ull, 0x081908082b081908ull, 0x081908082b08192bull,
    0x081908082b190808ull, 0x081908082b191919ull, 0x081908082b192b08ull, 0x081908082b2b0819ull,
    0x081908082b2b1908ull, 0x0819081908080808ull, 0x081908190808082bull, 0x0819081908081919ull,
    0x0819081908082b08ull, 0x0819081908082b2bull, 0x0819081908190819ull, 0x0819081908191908ull,
    0x081908190819192bull, 0x0819081908192b19ull, 0x08190819082b0808ull, 0x08190819082b082bull,
    0x08190819082b1919ull, 0x08190819082b2b08ull, 0x0819081919080819ull, 0x0819081919081908ull,
    0x081908191908192bull, 0x0819081919082b19ull, 0x0819081919190808ull, 0x081908191919082bull,
    0x0819081919191919ull, 0x0819081919192b08ull, 0x08190819192b0819ull, 0x08190819192b1908ull,
    0x081908192b080808ull, 0x081908192b08082bull, 0x081908192b081919ull, 0x081908192b082b08ull,
    0x081908192b190819ull, 0x081908192b191908ull, 0x0819082b08080819ull, 0x0819082b08081908ull,
    0x0819082b08082b19ull, 0x0819082b08190808ull, 0x0819082b08191919ull, 0x0819082b082b0819ull,
    0x0819082b082b1908ull, 0x0819082b19080808ull, 0x0819082b19081919ull, 0x0819082b19190819ull,
    0x0819082b19191908ull, 0x0819082b2b080819ull, 0x0819082b2b081908ull, 0x0819082b2b190808ull,
    0x0819190808080808ull, 0x081919080808082bull, 0x0819190808081919ull, 0x0819190808082b08ull,
    0x0819190808190819ull, 0x0819190808191908ull, 0x081919080819192bull, 0x0819190808192b19ull,
    0x08191908082b0808ull, 0x08191908082b1919ull, 0x08191908082b2b08ull, 0x0819190819080819ull,
    0x0819190819081908ull, 0x081919081908192bull, 0x0819190819082b19ull, 0x0819190819190808ull,
    0x081919081919082bull, 0x0819190819191919ull, 0x0819190819192b08ull, 0x08191908192b0819ull,
    0x08191908192b1908ull, 0x081919082b080808ull, 0x081919082b08082bull, 0x081919082b081919ull,
    0x081919082b082b08ull, 0x081919082b190819ull, 0x081919082b191908ull, 0x081919082b2b0808ull,
    0x0819191908080819ull, 0x0819191908081908ull, 0x081919190808192bull, 0x0819191908082b19ull,
    0x0819191908190808ull, 0x081919190819082bull, 0x0819191908191919ull, 0x0819191908192b08ull,
    0x08191919082b0819ull, 0x08191919082b1908ull, 0x0819191919080808ull, 0x081919191908082bull,
    0x0819191919081919ull, 0x0819191919082b08ull, 0x0819191919190819ull, 0x0819191919191908ull,
    0x08191919192b0808ull, 0x081919192b080819ull, 0x081919192b081908ull, 0x081919192b190808ull,
    0x0819192b08080808ull, 0x0819192b08081919ull, 0x0819192b08082b08ull, 0x0819192b08190819ull,
    0x0819192b08191908ull, 0x0819192b082b0808ull, 0x0819192b19080819ull, 0x0819192b19081908ull,
    0x0819192b19190808ull, 0x0819192b2b080808ull, 0x0819192b2b2b2b2bull, 0x08192b0808080819ull,
    0x08192b0808081908ull, 0x08192b080808192bull, 0x08192b0808082b19ull, 0x08192b0808190808ull,
    0x08192b0808191919ull, 0x08192b0808192b08ull, 0x08192b08082b0819ull, 0x08192b0819080808ull,
    0x08192b081908082bull, 0x08192b0819081919ull, 0x08192b0819082b08ull, 0x08192b0819190819ull,
    0x08192b0819191908ull, 0x08192b08192b0808ull, 0x08192b082b080819ull, 0x08192b082b081908ull,
    0x08192b1908080808ull, 0x08192b190808082bull, 0x08192b1908081919ull, 0x08192b1908082b08ull,
    0x08192b1908190819ull, 0x08192b1908191908ull, 0x08192b19082b0808ull, 0x08192b1919080819ull,
    0x08192b1919081908ull, 0x08192b1919190808ull, 0x08192b19192b2b19ull, 0x08192b192b2b082bull,
    0x08192b2b08081908ull, 0x08192b2b08190808ull, 0x08192b2b19080808ull, 0x08192b2b1919192bull,
    0x082b080808080808ull, 0x082b08080808082bull, 0x082b080808081919ull, 0x082b080808082b08ull,
    0x082b080808190819ull, 0x082b080808191908ull, 0x082b08080819192bull, 0x082b080808192b19ull,
    0x082b0808082b0808ull, 0x082b0808082b1919ull, 0x082b0808082b2b2bull, 0x082b080819080819ull,
    0x082b080819081908ull, 0x082b080819190808ull, 0x082b08081919082bull, 0x082b080819191919ull,
    0x082b0808192b1908ull, 0x082b08082b080808ull, 0x082b08082b082b2bull, 0x082b08082b191908ull,
    0x082b08082b2b2b2bull, 0x082b081908080819ull, 0x082b081908081908ull, 0x082b081908190808ull,
    0x082b08190819082bull, 0x082b081908191919ull, 0x082b0819082b0819ull, 0x082b081919080808ull,
    0x082b08191908082bull, 0x082b081919081919ull, 0x082b081919190819ull, 0x082b081919191908ull,
    0x082b0819192b0808ull, 0x082b08192b080819ull, 0x082b08192b081908ull, 0x082b08192b190808ull,
    0x082b082b08080808ull, 0x082b082b08082b2bull, 0x082b082b082b082bull, 0x082b082b082b2b08ull,
    0x082b082b082b2b2bull, 0x082b082b19081908ull, 0x082b082b19190808ull, 0x082b082b2b082b08ull,
    0x082b082b2b082b2bull, 0x082b082b2b2b2b08ull, 0x082b190808080819ull, 0x082b190808081908ull,
    0x082b19080808192bull, 0x082b190808082b19ull, 0x082b190808190808ull, 0x082b190808191919ull,
    0x082b190808192b08ull, 0x082b1908082b0819ull, 0x082b1908082b1908ull, 0x082b190819080808ull,
    0x082b19081908082bull, 0x082b190819081919ull, 0x082b190819082b08ull, 0x082b190819190819ull,
    0x082b190819191908ull, 0x082b1908192b0808ull, 0x082b19082b080819ull, 0x082b19082b081908ull,
    0x082b19082b190808ull, 0x082b191908080808ull, 0x082b191908081919ull, 0x082b191908082b08ull,
    0x082b191908190819ull, 0x082b191908191908ull, 0x082b1919082b0808ull, 0x082b191919080819ull,
    0x082b191919081908ull, 0x082b191919190808ull, 0x082b1919192b192bull, 0x082b19192b080808ull,
    0x082b192b08080819ull, 0x082b192b08081908ull, 0x082b192b08190808ull, 0x082b192b19080808ull,
    0x082b192b19192b19ull, 0x082b2b0808080808ull, 0x082b2b0808081919ull, 0x082b2b0808190819ull,
    0x082b2b0808191908ull, 0x082b2b0819080819ull, 0x082b2b0819081908ull, 0x082b2b0819190808ull,
    0x082b2b082b082b2bull, 0x082b2b082b2b2b2bull, 0x082b2b1908080819ull, 0x082b2b1908081908ull,
    0x082b2b1908190808ull, 0x082b2b192b191919ull, 0x082b2b2b08082b2bull, 0x082b2b2b082b082bull,
    0x082b2b2b192b1908ull, 0x082b2b2b2b082b08ull, 0x082b2b2b2b082b2bull, 0x1908080808080819ull,
    0x1908080808081908ull, 0x190808080808192bull, 0x1908080808082b19ull, 0x1908080808190808ull,
    0x190808080819082bull, 0x1908080808191919ull, 0x1908080808192b08ull, 0x1908080808192b2bull,
    0x19080808082b0819ull, 0x19080808082b1908ull, 0x19080808082b192bull, 0x1908080819080808ull,
    0x190808081908082bull, 0x1908080819081919ull, 0x1908080819082b08ull, 0x1908080819082b2bull,
    0x1908080819190819ull, 0x1908080819191908ull, 0x190808081919192bull, 0x1908080819192b19ull,
    0x19080808192b0808ull, 0x19080808192b082bull, 0x19080808192b1919ull, 0x190808082b080819ull,
    0x190808082b081908ull, 0x190808082b190808ull, 0x190808082b191919ull, 0x190808082b192b08ull,
    0x190808082b2b0819ull, 0x190808082b2b1908ull, 0x1908081908080808ull, 0x190808190808082bull,
    0x1908081908081919ull, 0x1908081908082b08ull, 0x1908081908190819ull, 0x1908081908191908ull,
    0x190808190819192bull, 0x1908081908192b19ull, 0x19080819082b0808ull, 0x19080819082b082bull,
    0x19080819082b1919ull, 0x1908081919080819ull, 0x1908081919081908ull, 0x190808191908192bull,
    0x1908081919082b19ull, 0x1908081919190808ull, 0x190808191919082bull, 0x1908081919191919ull,
    0x1908081919192b08ull, 0x19080819192b0819ull, 0x19080819192b1908ull, 0x190808192b080808ull,
    0x190808192b08082bull, 0x190808192b081919ull, 0x190808192b082b08ull, 0x190808192b190819ull,
    0x190808192b191908ull, 0x190808192b2b0808ull, 0x1908082b08080819ull, 0x1908082b08081908ull,
    0x1908082b08190808ull, 0x1908082b0819082bull, 0x1908082b08191919ull, 0x1908082b08192b08ull,
    0x1908082b082b1908ull, 0x1908082b19080808ull, 0x1908082b19081919ull, 0x1908082b19082b08ull,
    0x1908082b19190819ull, 0x1908082b19191908ull, 0x1908082b192b0808ull, 0x1908082b2b080819ull,
    0x1908082b2b081908ull, 0x1908190808080808ull, 0x190819080808082bull, 0x1908190808081919ull,
    0x1908190808082b08ull, 0x1908190808082b2bull, 0x1908190808190819ull, 0x1908190808191908ull,
    0x190819080819192bull, 0x1908190808192b19ull, 0x19081908082b0808ull, 0x19081908082b082bull,
    0x19081908082b1919ull, 0x19081908082b2b08ull, 0x1908190819080819ull, 0x1908190819081908ull,
    0x190819081908192bull, 0x1908190819082b19ull, 0x1908190819190808ull, 0x190819081919082bull,
    0x1908190819191919ull, 0x1908190819192b08ull, 0x19081908192b0819ull, 0x19081908192b1908ull,
    0x190819082b080808ull, 0x190819082b08082bull, 0x190819082b081919ull, 0x190819082b082b08ull,
    0x190819082b190819ull, 0x190819082b191908ull, 0x190819082b2b0808ull, 0x1908191908080819ull,
    0x1908191908081908ull, 0x190819190808192bull, 0x1908191908082b19ull, 0x1908191908190808ull,
    0x190819190819082bull, 0x1908191908191919ull, 0x1908191908192b08ull, 0x19081919082b0819ull,
    0x19081919082b1908ull, 0x1908191919080808ull, 0x190819191908082bull, 0x1908191919081919ull,
    0x1908191919082b08ull, 0x1908191919190819ull, 0x1908191919191908ull, 0x19081919192b0808ull,
    0x19081919192b2b2bull, 0x190819192b080819ull, 0x190819192b081908ull, 0x190819192b190808ull,
    0x1908192b08080808ull, 0x1908192b0808082bull, 0x1908192b08081919ull, 0x1908192b08082b08ull,
    0x1908192b08190819ull, 0x1908192b08191908ull, 0x1908192b082b0808ull, 0x1908192b19080819ull,
    0x1908192b19081908ull, 0x1908192b19190808ull, 0x1908192b2b080808ull, 0x1908192b2b2b1919ull,
    0x19082b0808080819ull, 0x19082b0808081908ull, 0x19082b0808082b19ull, 0x19082b0808190808ull,
    0x19082b080819082bull, 0x19082b0808191919ull, 0x19082b0808192b08ull, 0x19082b08082b0819ull,
    0x19082b08082b1908ull, 0x19082b0819080808ull, 0x19082b081908082bull, 0x19082b0819081919ull,
    0x19082b0819082b08ull, 0x19082b0819190819ull, 0x19082b0819191908ull, 0x19082b08192b0808ull,
    0x19082b082b081908ull, 0x19082b082b190808ull, 0x19082b1908080808ull, 0x19082b190808082bull,
    0x19082b1908081919ull, 0x19082b1908082b08ull, 0x19082b1908190819ull, 0x19082b1908191908ull,
    0x19082b19082b0808ull, 0x19082b1919080819ull, 0x19082b1919081908ull, 0x19082b1919190808ull,
    0x19082b192b080808ull, 0x19082b192b19192bull, 0x19082b2b08080819ull, 0x19082b2b08081908ull,
    0x19082b2b08190808ull, 0x19082b2b19080808ull, 0x1919080808080808ull, 0x191908080808082bull,
    0x1919080808081919ull, 0x1919080808082b08ull, 0x1919080808190819ull, 0x1919080808191908ull,
    0x191908080819192bull, 0x1919080808192b19ull, 0x19190808082b0808ull, 0x19190808082b082bull,
    0x19190808082b1919ull, 0x19190808082b2b08ull, 0x1919080819080819ull, 0x1919080819081908ull,
    0x191908081908192bull, 0x1919080819082b19ull, 0x1919080819190808ull, 0x191908081919082bull,
    0x1919080819191919ull, 0x1919080819192b08ull, 0x19190808192b0819ull, 0x19190808192b1908ull,
    0x191908082b080808ull, 0x191908082b08082bull, 0x191908082b081919ull, 0x191908082b082b08ull,
    0x191908082b190819ull, 0x191908082b191908ull, 0x1919081908080819ull, 0x1919081908081908ull,
    0x191908190808192bull, 0x1919081908082b19ull, 0x1919081908190808ull, 0x191908190819082bull,
    0x1919081908191919ull, 0x1919081908192b08ull, 0x19190819082b0819ull, 0x19190819082b1908ull,
    0x1919081919080808ull, 0x191908191908082bull, 0x1919081919081919ull, 0x1919081919082b08ull,
    0x1919081919190819ull, 0x1919081919191908ull, 0x19190819192b0808ull, 0x191908192b080819ull,
    0x191908192b081908ull, 0x191908192b190808ull, 0x1919082b08080808ull, 0x1919082b08081919ull,
    0x1919082b08082b08ull, 0x1919082b08190819ull, 0x1919082b08191908ull, 0x1919082b082b0808ull,
    0x1919082b19080819ull, 0x1919082b19081908ull, 0x1919082b19190808ull, 0x1919082b192b2b19ull,
    0x1919082b2b080808ull, 0x1919190808080819ull, 0x1919190808081908ull, 0x191919080808192bull,
    0x1919190808082b19ull, 0x1919190808190808ull, 0x191919080819082bull, 0x1919190808191919ull,
    0x1919190808192b08ull, 0x19191908082b0819ull, 0x19191908082b1908ull, 0x1919190819080808ull,
    0x191919081908082bull, 0x1919190819081919ull, 0x1919190819082b08ull, 0x1919190819190819ull,
    0x1919190819191908ull, 0x19191908192b0808ull, 0x191919082b080819ull, 0x191919082b081908ull,
    0x191919082b190808ull, 0x1919191908080808ull, 0x191919190808082bull, 0x1919191908081919ull,
    0x1919191908082b08ull, 0x1919191908190819ull, 0x1919191908191908ull, 0x19191919082b0808ull,
    0x1919191919080819ull, 0x1919191919081908ull, 0x1919191919190808ull, 0x191919192b080808ull,
    0x1919192b08080819ull, 0x1919192b08081908ull, 0x1919192b08190808ull, 0x1919192b082b192bull,
    0x1919192b19080808ull, 0x19192b0808080808ull, 0x19192b080808082bull, 0x19192b0808081919ull,
    0x19192b0808082b08ull, 0x19192b0808190819ull, 0x19192b0808191908ull, 0x19192b08082b0808ull,
    0x19192b0819080819ull, 0x19192b0819081908ull, 0x19192b0819190808ull, 0x19192b0819192b2bull,
    0x19192b082b080808ull, 0x19192b1908080819ull, 0x19192b1908081908ull, 0x19192b1908190808ull,
    0x19192b1919080808ull, 0x19192b2b08080808ull, 0x19192b2b08192b19ull, 0x19192b2b2b081919ull,
    0x19192b2b2b2b2b08ull, 0x192b080808080819ull, 0x192b080808081908ull, 0x192b08080808192bull,
    0x192b080808190808ull, 0x192b08080819082bull, 0x192b080808191919ull, 0x192b080808192b08ull,
    0x192b0808082b0819ull, 0x192b0808082b1908ull, 0x192b080819080808ull, 0x192b080819081919ull,
    0x192b080819082b08ull, 0x192b080819190819ull, 0x192b080819191908ull, 0x192b0808192b0808ull,
    0x192b08082b081908ull, 0x192b08082b190808ull, 0x192b081908080808ull, 0x192b08190808082bull,
    0x192b081908081919ull, 0x192b081908082b08ull, 0x192b081908190819ull, 0x192b081908191908ull,
    0x192b0819082b0808ull, 0x192b081919080819ull, 0x192b081919081908ull, 0x192b081919190808ull,
    0x192b08192b080808ull, 0x192b08192b192b19ull, 0x192b082b08081908ull, 0x192b082b08190808ull,
    0x192b082b19080808ull, 0x192b082b1919192bull, 0x192b082b2b2b0819ull, 0x192b190808080808ull,
    0x192b190808081919ull, 0x192b190808082b08ull, 0x192b190808190819ull, 0x192b190808191908ull,
    0x192b1908082b0808ull, 0x192b190819080819ull, 0x192b190819081908ull, 0x192b190819190808ull,
    0x192b19082b080808ull, 0x192b191908080819ull, 0x192b191908081908ull, 0x192b191908190808ull,
    0x192b191919080808ull, 0x192b191919082b2bull, 0x192b1919192b2b08ull, 0x192b19192b19082bull,
    0x192b192b08080808ull, 0x192b192b2b191908ull, 0x192b2b0808080819ull, 0x192b2b0808081908ull,
    0x192b2b0808190808ull, 0x192b2b08192b1919ull, 0x192b2b082b192b08ull, 0x192b2b1908080808ull,
    0x192b2b19082b2b2bull, 0x192b2b2b1908082bull, 0x192b2b2b2b2b0819ull, 0x2b08080808080808ull,
    0x2b0808080808082bull, 0x2b08080808081919ull, 0x2b08080808082b08ull, 0x2b08080808190819ull,
    0x2b08080808191908ull, 0x2b08080808192b19ull, 0x2b080808082b0808ull, 0x2b080808082b1919ull,
    0x2b08080819080819ull, 0x2b08080819081908ull, 0x2b08080819190808ull, 0x2b0808081919082bull,
    0x2b08080819191919ull, 0x2b08080819192b08ull, 0x2b080808192b0819ull, 0x2b0808082b080808ull,
    0x2b0808082b081919ull, 0x2b0808082b190819ull, 0x2b0808082b191908ull, 0x2b08081908080819ull,
    0x2b08081908081908ull, 0x2b08081908082b19ull, 0x2b08081908190808ull, 0x2b0808190819082bull,
    0x2b08081908191919ull, 0x2b08081908192b08ull, 0x2b080819082b0819ull, 0x2b080819082b1908ull,
    0x2b08081919080808ull, 0x2b0808191908082bull, 0x2b08081919081919ull, 0x2b08081919082b08ull,
    0x2b08081919190819ull, 0x2b08081919191908ull, 0x2b0808192b080819ull, 0x2b0808192b081908ull,
    0x2b0808192b190808ull, 0x2b0808192b2b2b19ull, 0x2b08082b08080808ull, 0x2b08082b08081919ull,
    0x2b08082b08082b2bull, 0x2b08082b08190819ull, 0x2b08082b08191908ull, 0x2b08082b19080819ull,
    0x2b08082b19081908ull, 0x2b08082b19190808ull, 0x2b08190808080819ull, 0x2b08190808081908ull,
    0x2b0819080808192bull, 0x2b08190808082b19ull, 0x2b08190808190808ull, 0x2b0819080819082bull,
    0x2b08190808191919ull, 0x2b08190808192b08ull, 0x2b081908082b0819ull, 0x2b08190819080808ull,
    0x2b0819081908082bull, 0x2b08190819081919ull, 0x2b08190819082b08ull, 0x2b08190819190819ull,
    0x2b08190819191908ull, 0x2b081908192b0808ull, 0x2b0819082b080819ull, 0x2b0819082b081908ull,
    0x2b0819082b190808ull, 0x2b08191908080808ull, 0x2b0819190808082bull, 0x2b08191908081919ull,
    0x2b08191908082b08ull, 0x2b08191908190819ull, 0x2b08191908191908ull, 0x2b081919082b0808ull,
    0x2b08191919080819ull, 0x2b08191919081908ull, 0x2b08191919190808ull, 0x2b0819192b080808ull,
    0x2b0819192b082b2bull, 0x2b08192b08080819ull, 0x2b08192b08081908ull, 0x2b08192b08190808ull,
    0x2b08192b082b2b19ull, 0x2b08192b19080808ull, 0x2b082b0808080808ull, 0x2b082b0808081919ull,
    0x2b082b0808190819ull, 0x2b082b0808191908ull, 0x2b082b0819080819ull, 0x2b082b0819081908ull,
    0x2b082b0819190808ull, 0x2b082b082b2b082bull, 0x2b082b1908080819ull, 0x2b082b1908081908ull,
    0x2b082b1919080808ull, 0x2b082b19192b1919ull, 0x2b082b2b082b082bull, 0x2b082b2b19192b08ull,
    0x2b082b2b19192b2bull, 0x2b082b2b2b08082bull, 0x2b082b2b2b2b082bull, 0x2b19080808080819ull,
    0x2b19080808081908ull, 0x2b19080808082b19ull, 0x2b19080808190808ull, 0x2b1908080819082bull,
    0x2b19080808191919ull, 0x2b19080808192b08ull, 0x2b190808082b1908ull, 0x2b19080819080808ull,
    0x2b1908081908082bull, 0x2b19080819081919ull, 0x2b19080819082b08ull, 0x2b19080819190819ull,
    0x2b19080819191908ull, 0x2b190808192b0808ull, 0x2b1908082b080819ull, 0x2b1908082b081908ull,
    0x2b1908082b190808ull, 0x2b19081908080808ull, 0x2b19081908081919ull, 0x2b19081908190819ull,
    0x2b19081908191908ull, 0x2b19081919080819ull, 0x2b19081919081908ull, 0x2b19081919190808ull,
    0x2b19081919192b2bull, 0x2b19082b08080819ull, 0x2b19082b08081908ull, 0x2b19082b08190808ull,
    0x2b19082b19080808ull, 0x2b19082b2b2b192bull, 0x2b19190808080808ull, 0x2b1919080808082bull,
    0x2b19190808081919ull, 0x2b19190808082b08ull, 0x2b19190808190819ull, 0x2b19190808191908ull,
    0x2b191908082b0808ull, 0x2b19190819080819ull, 0x2b19190819081908ull, 0x2b19190819190808ull,
    0x2b1919082b080808ull, 0x2b1919082b19192bull, 0x2b19191908080819ull, 0x2b19191908081908ull,
    0x2b19191908190808ull, 0x2b19191919080808ull, 0x2b1919192b192b08ull, 0x2b1919192b2b0819ull,
    0x2b19192b08080808ull, 0x2b19192b1908192bull, 0x2b19192b192b1908ull, 0x2b192b0808080819ull,
    0x2b192b0808081908ull, 0x2b192b0808190808ull, 0x2b192b08082b192bull, 0x2b192b0819080808ull,
    0x2b192b082b2b2b19ull, 0x2b192b1908080808ull, 0x2b192b1919082b19ull, 0x2b192b191919082bull,
    0x2b192b2b2b190808ull, 0x2b2b080808080808ull, 0x2b2b080808081919ull, 0x2b2b080808082b2bull,
    0x2b2b080808191908ull, 0x2b2b0808082b082bull, 0x2b2b0808082b2b2bull, 0x2b2b080819080819ull,
    0x2b2b080819081908ull, 0x2b2b080819190808ull, 0x2b2b08082b2b082bull, 0x2b2b08082b2b2b2bull,
    0x2b2b081919080808ull, 0x2b2b0819192b1919ull, 0x2b2b082b0808082bull, 0x2b2b082b08082b2bull,
    0x2b2b082b082b082bull, 0x2b2b082b082b2b08ull, 0x2b2b082b082b2b2bull, 0x2b2b082b2b08082bull,
    0x2b2b082b2b082b08ull, 0x2b2b082b2b082b2bull, 0x2b2b082b2b2b2b08ull, 0x2b2b190808080819ull,
    0x2b2b190808081908ull, 0x2b2b190808190808ull, 0x2b2b190819080808ull, 0x2b2b19082b082b19ull,
    0x2b2b19082b2b1908ull, 0x2b2b191908080808ull, 0x2b2b191908192b19ull, 0x2b2b192b19190819ull,
    0x2b2b2b0808082b2bull, 0x2b2b2b08082b2b08ull, 0x2b2b2b082b2b082bull, 0x2b2b2b1919191908ull,
    0x2b2b2b192b08192bull, 0x2b2b2b2b08082b08ull, 0x2b2b2b2b08082b2bull, 0x2b2b2b2b082b0808ull,
    0x2b2b2b2b082b082bull, 0x2b2b2b2b082b2b08ull, 0x2b2b2b2b2b082b08ull, 0x2b2b2b2b2b2b2b2bull,
};

__device__ const unsigned int IQ3XXS_GRID[256] = {
    0x04040404u, 0x04040414u, 0x04040424u, 0x04040c0cu,
    0x04040c1cu, 0x04040c3eu, 0x04041404u, 0x04041414u,
    0x04041c0cu, 0x04042414u, 0x04043e1cu, 0x04043e2cu,
    0x040c040cu, 0x040c041cu, 0x040c0c04u, 0x040c0c14u,
    0x040c140cu, 0x040c142cu, 0x040c1c04u, 0x040c1c14u,
    0x040c240cu, 0x040c2c24u, 0x040c3e04u, 0x04140404u,
    0x04140414u, 0x04140424u, 0x04140c0cu, 0x04141404u,
    0x04141414u, 0x04141c0cu, 0x04141c1cu, 0x04141c3eu,
    0x04142c0cu, 0x04142c3eu, 0x04143e2cu, 0x041c040cu,
    0x041c043eu, 0x041c0c04u, 0x041c0c14u, 0x041c142cu,
    0x041c3e04u, 0x04240c1cu, 0x04241c3eu, 0x04242424u,
    0x04242c3eu, 0x04243e1cu, 0x04243e2cu, 0x042c040cu,
    0x042c043eu, 0x042c1c14u, 0x042c2c14u, 0x04341c2cu,
    0x04343424u, 0x043e0c04u, 0x043e0c24u, 0x043e0c34u,
    0x043e241cu, 0x043e340cu, 0x0c04040cu, 0x0c04041cu,
    0x0c040c04u, 0x0c040c14u, 0x0c04140cu, 0x0c04141cu,
    0x0c041c04u, 0x0c041c14u, 0x0c041c24u, 0x0c04243eu,
    0x0c042c04u, 0x0c0c0404u, 0x0c0c0414u, 0x0c0c0c0cu,
    0x0c0c1404u, 0x0c0c1414u, 0x0c14040cu, 0x0c14041cu,
    0x0c140c04u, 0x0c140c14u, 0x0c14140cu, 0x0c141c04u,
    0x0c143e14u, 0x0c1c0404u, 0x0c1c0414u, 0x0c1c1404u,
    0x0c1c1c0cu, 0x0c1c2434u, 0x0c1c3434u, 0x0c24040cu,
    0x0c24042cu, 0x0c242c04u, 0x0c2c1404u, 0x0c2c1424u,
    0x0c2c2434u, 0x0c2c3e0cu, 0x0c34042cu, 0x0c3e1414u,
    0x0c3e2404u, 0x14040404u, 0x14040414u, 0x14040c0cu,
    0x14040c1cu, 0x14041404u, 0x14041414u, 0x14041434u,
    0x14041c0cu, 0x14042414u, 0x140c040cu, 0x140c041cu,
    0x140c042cu, 0x140c0c04u, 0x140c0c14u, 0x140c140cu,
    0x140c1c04u, 0x140c341cu, 0x140c343eu, 0x140c3e04u,
    0x14140404u, 0x14140414u, 0x14140c0cu, 0x14140c3eu,
    0x14141404u, 0x14141414u, 0x14141c3eu, 0x14142404u,
    0x14142c2cu, 0x141c040cu, 0x141c0c04u, 0x141c0c24u,
    0x141c3e04u, 0x141c3e24u, 0x14241c2cu, 0x14242c1cu,
    0x142c041cu, 0x142c143eu, 0x142c240cu, 0x142c3e24u,
    0x143e040cu, 0x143e041cu, 0x143e0c34u, 0x143e242cu,
    0x1c04040cu, 0x1c040c04u, 0x1c040c14u, 0x1c04140cu,
    0x1c04141cu, 0x1c042c04u, 0x1c04342cu, 0x1c043e14u,
    0x1c0c0404u, 0x1c0c0414u, 0x1c0c1404u, 0x1c0c1c0cu,
    0x1c0c2424u, 0x1c0c2434u, 0x1c14040cu, 0x1c14041cu,
    0x1c140c04u, 0x1c14142cu, 0x1c142c14u, 0x1c143e14u,
    0x1c1c0c0cu, 0x1c1c1c1cu, 0x1c241c04u, 0x1c24243eu,
    0x1c243e14u, 0x1c2c0404u, 0x1c2c0434u, 0x1c2c1414u,
    0x1c2c2c2cu, 0x1c340c24u, 0x1c341c34u, 0x1c34341cu,
    0x1c3e1c1cu, 0x1c3e3404u, 0x24040424u, 0x24040c3eu,
    0x24041c2cu, 0x24041c3eu, 0x24042c1cu, 0x24042c3eu,
    0x240c3e24u, 0x24141404u, 0x24141c3eu, 0x24142404u,
    0x24143404u, 0x24143434u, 0x241c043eu, 0x241c242cu,
    0x24240424u, 0x24242c0cu, 0x24243424u, 0x242c142cu,
    0x242c241cu, 0x242c3e04u, 0x243e042cu, 0x243e0c04u,
    0x243e0c14u, 0x243e1c04u, 0x2c040c14u, 0x2c04240cu,
    0x2c043e04u, 0x2c0c0404u, 0x2c0c0434u, 0x2c0c1434u,
    0x2c0c2c2cu, 0x2c140c24u, 0x2c141c14u, 0x2c143e14u,
    0x2c1c0414u, 0x2c1c2c1cu, 0x2c240c04u, 0x2c24141cu,
    0x2c24143eu, 0x2c243e14u, 0x2c2c0414u, 0x2c2c1c0cu,
    0x2c342c04u, 0x2c3e1424u, 0x2c3e2414u, 0x34041424u,
    0x34042424u, 0x34042434u, 0x34043424u, 0x340c140cu,
    0x340c340cu, 0x34140c3eu, 0x34143424u, 0x341c1c04u,
    0x341c1c34u, 0x34242424u, 0x342c042cu, 0x342c2c14u,
    0x34341c1cu, 0x343e041cu, 0x343e140cu, 0x3e04041cu,
    0x3e04042cu, 0x3e04043eu, 0x3e040c04u, 0x3e041c14u,
    0x3e042c14u, 0x3e0c1434u, 0x3e0c2404u, 0x3e140c14u,
    0x3e14242cu, 0x3e142c14u, 0x3e1c0404u, 0x3e1c0c2cu,
    0x3e1c1c1cu, 0x3e1c3404u, 0x3e24140cu, 0x3e24240cu,
    0x3e2c0404u, 0x3e2c0414u, 0x3e2c1424u, 0x3e341c04u,
};

__device__ const unsigned int IQ3S_GRID[512] = {
    0x01010101u, 0x01010103u, 0x01010105u, 0x0101010bu,
    0x0101010fu, 0x01010301u, 0x01010303u, 0x01010305u,
    0x01010309u, 0x0101030du, 0x01010501u, 0x01010503u,
    0x0101050bu, 0x01010707u, 0x01010901u, 0x01010905u,
    0x0101090bu, 0x0101090fu, 0x01010b03u, 0x01010b07u,
    0x01010d01u, 0x01010d05u, 0x01010f03u, 0x01010f09u,
    0x01010f0fu, 0x01030101u, 0x01030103u, 0x01030105u,
    0x01030109u, 0x01030301u, 0x01030303u, 0x0103030bu,
    0x01030501u, 0x01030507u, 0x0103050fu, 0x01030703u,
    0x0103070bu, 0x01030909u, 0x01030d03u, 0x01030d0bu,
    0x01030f05u, 0x01050101u, 0x01050103u, 0x0105010bu,
    0x0105010fu, 0x01050301u, 0x01050307u, 0x0105030du,
    0x01050503u, 0x0105050bu, 0x01050701u, 0x01050709u,
    0x01050905u, 0x0105090bu, 0x0105090fu, 0x01050b03u,
    0x01050b07u, 0x01050f01u, 0x01050f07u, 0x01070107u,
    0x01070303u, 0x0107030bu, 0x01070501u, 0x01070505u,
    0x01070703u, 0x01070707u, 0x0107070du, 0x01070909u,
    0x01070b01u, 0x01070b05u, 0x01070d0fu, 0x01070f03u,
    0x01070f0bu, 0x01090101u, 0x01090307u, 0x0109030fu,
    0x01090503u, 0x01090509u, 0x01090705u, 0x01090901u,
    0x01090907u, 0x01090b03u, 0x01090f01u, 0x010b0105u,
    0x010b0109u, 0x010b0501u, 0x010b0505u, 0x010b050du,
    0x010b0707u, 0x010b0903u, 0x010b090bu, 0x010b090fu,
    0x010b0d0du, 0x010b0f07u, 0x010d010du, 0x010d0303u,
    0x010d0307u, 0x010d0703u, 0x010d0b05u, 0x010d0f03u,
    0x010f0101u, 0x010f0105u, 0x010f0109u, 0x010f0501u,
    0x010f0505u, 0x010f050du, 0x010f0707u, 0x010f0b01u,
    0x010f0b09u, 0x03010101u, 0x03010103u, 0x03010105u,
    0x03010109u, 0x03010301u, 0x03010303u, 0x03010307u,
    0x0301030bu, 0x0301030fu, 0x03010501u, 0x03010505u,
    0x03010703u, 0x03010709u, 0x0301070du, 0x03010b09u,
    0x03010b0du, 0x03010d03u, 0x03010f05u, 0x03030101u,
    0x03030103u, 0x03030107u, 0x0303010du, 0x03030301u,
    0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u,
    0x03030903u, 0x03030b01u, 0x03030b05u, 0x03030f01u,
    0x03030f0du, 0x03050101u, 0x03050305u, 0x0305030bu,
    0x0305030fu, 0x03050501u, 0x03050509u, 0x03050705u,
    0x03050901u, 0x03050907u, 0x03050b0bu, 0x03050d01u,
    0x03050f05u, 0x03070103u, 0x03070109u, 0x0307010fu,
    0x03070301u, 0x03070307u, 0x03070503u, 0x0307050fu,
    0x03070701u, 0x03070709u, 0x03070903u, 0x03070d05u,
    0x03070f01u, 0x03090107u, 0x0309010bu, 0x03090305u,
    0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u,
    0x0309090du, 0x03090b01u, 0x03090b09u, 0x030b0103u,
    0x030b0301u, 0x030b0307u, 0x030b0503u, 0x030b0701u,
    0x030b0705u, 0x030b0b03u, 0x030d0501u, 0x030d0509u,
    0x030d050fu, 0x030d0909u, 0x030d090du, 0x030f0103u,
    0x030f0107u, 0x030f0301u, 0x030f0305u, 0x030f0503u,
    0x030f070bu, 0x030f0903u, 0x030f0d05u, 0x030f0f01u,
    0x05010101u, 0x05010103u, 0x05010107u, 0x0501010bu,
    0x0501010fu, 0x05010301u, 0x05010305u, 0x05010309u,
    0x0501030du, 0x05010503u, 0x05010507u, 0x0501050fu,
    0x05010701u, 0x05010705u, 0x05010903u, 0x05010907u,
    0x0501090bu, 0x05010b01u, 0x05010b05u, 0x05010d0fu,
    0x05010f01u, 0x05010f07u, 0x05010f0bu, 0x05030101u,
    0x05030105u, 0x05030301u, 0x05030307u, 0x0503030fu,
    0x05030505u, 0x0503050bu, 0x05030703u, 0x05030709u,
    0x05030905u, 0x05030b03u, 0x05050103u, 0x05050109u,
    0x0505010fu, 0x05050503u, 0x05050507u, 0x05050701u,
    0x0505070fu, 0x05050903u, 0x05050b07u, 0x05050b0fu,
    0x05050f03u, 0x05050f09u, 0x05070101u, 0x05070105u,
    0x0507010bu, 0x05070303u, 0x05070505u, 0x05070509u,
    0x05070703u, 0x05070707u, 0x05070905u, 0x05070b01u,
    0x05070d0du, 0x05090103u, 0x0509010fu, 0x05090501u,
    0x05090507u, 0x05090705u, 0x0509070bu, 0x05090903u,
    0x05090f05u, 0x05090f0bu, 0x050b0109u, 0x050b0303u,
    0x050b0505u, 0x050b070fu, 0x050b0901u, 0x050b0b07u,
    0x050b0f01u, 0x050d0101u, 0x050d0105u, 0x050d010fu,
    0x050d0503u, 0x050d0b0bu, 0x050d0d03u, 0x050f010bu,
    0x050f0303u, 0x050f050du, 0x050f0701u, 0x050f0907u,
    0x050f0b01u, 0x07010105u, 0x07010303u, 0x07010307u,
    0x0701030bu, 0x0701030fu, 0x07010505u, 0x07010703u,
    0x07010707u, 0x0701070bu, 0x07010905u, 0x07010909u,
    0x0701090fu, 0x07010b03u, 0x07010d07u, 0x07010f03u,
    0x07030103u, 0x07030107u, 0x0703010bu, 0x07030309u,
    0x07030503u, 0x07030507u, 0x07030901u, 0x07030d01u,
    0x07030f05u, 0x07030f0du, 0x07050101u, 0x07050305u,
    0x07050501u, 0x07050705u, 0x07050709u, 0x07050b01u,
    0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u,
    0x07070507u, 0x0707050fu, 0x07070701u, 0x07070903u,
    0x07070907u, 0x0707090fu, 0x07070b0bu, 0x07070f07u,
    0x07090107u, 0x07090303u, 0x0709030du, 0x07090505u,
    0x07090703u, 0x07090b05u, 0x07090d01u, 0x07090d09u,
    0x070b0103u, 0x070b0301u, 0x070b0305u, 0x070b050bu,
    0x070b0705u, 0x070b0909u, 0x070b0b0du, 0x070b0f07u,
    0x070d030du, 0x070d0903u, 0x070f0103u, 0x070f0107u,
    0x070f0501u, 0x070f0505u, 0x070f070bu, 0x09010101u,
    0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u,
    0x0901050fu, 0x09010705u, 0x09010903u, 0x09010b01u,
    0x09010f01u, 0x09030105u, 0x0903010fu, 0x09030303u,
    0x09030307u, 0x09030505u, 0x09030701u, 0x0903070bu,
    0x09030907u, 0x09030b03u, 0x09030b0bu, 0x09050103u,
    0x09050107u, 0x09050301u, 0x0905030bu, 0x09050503u,
    0x09050707u, 0x09050901u, 0x09050b0fu, 0x09050d05u,
    0x09050f01u, 0x09070109u, 0x09070303u, 0x09070307u,
    0x09070501u, 0x09070505u, 0x09070703u, 0x0907070bu,
    0x09090101u, 0x09090105u, 0x09090509u, 0x0909070fu,
    0x09090901u, 0x09090f03u, 0x090b010bu, 0x090b010fu,
    0x090b0503u, 0x090b0d05u, 0x090d0307u, 0x090d0709u,
    0x090d0d01u, 0x090f0301u, 0x090f030bu, 0x090f0701u,
    0x090f0907u, 0x090f0b03u, 0x0b010105u, 0x0b010301u,
    0x0b010309u, 0x0b010505u, 0x0b010901u, 0x0b010909u,
    0x0b01090fu, 0x0b010b05u, 0x0b010d0du, 0x0b010f09u,
    0x0b030103u, 0x0b030107u, 0x0b03010bu, 0x0b030305u,
    0x0b030503u, 0x0b030705u, 0x0b030f05u, 0x0b050101u,
    0x0b050303u, 0x0b050507u, 0x0b050701u, 0x0b05070du,
    0x0b050b07u, 0x0b070105u, 0x0b07010fu, 0x0b070301u,
    0x0b07050fu, 0x0b070909u, 0x0b070b03u, 0x0b070d0bu,
    0x0b070f07u, 0x0b090103u, 0x0b090109u, 0x0b090501u,
    0x0b090705u, 0x0b09090du, 0x0b0b0305u, 0x0b0b050du,
    0x0b0b0b03u, 0x0b0b0b07u, 0x0b0d0905u, 0x0b0f0105u,
    0x0b0f0109u, 0x0b0f0505u, 0x0d010303u, 0x0d010307u,
    0x0d01030bu, 0x0d010703u, 0x0d010707u, 0x0d010d01u,
    0x0d030101u, 0x0d030501u, 0x0d03050fu, 0x0d030d09u,
    0x0d050305u, 0x0d050709u, 0x0d050905u, 0x0d050b0bu,
    0x0d050d05u, 0x0d050f01u, 0x0d070101u, 0x0d070309u,
    0x0d070503u, 0x0d070901u, 0x0d09050bu, 0x0d090907u,
    0x0d090d05u, 0x0d0b0101u, 0x0d0b0107u, 0x0d0b0709u,
    0x0d0b0d01u, 0x0d0d010bu, 0x0d0d0901u, 0x0d0f0303u,
    0x0d0f0307u, 0x0f010101u, 0x0f010109u, 0x0f01010fu,
    0x0f010501u, 0x0f010505u, 0x0f01070du, 0x0f010901u,
    0x0f010b09u, 0x0f010d05u, 0x0f030105u, 0x0f030303u,
    0x0f030509u, 0x0f030907u, 0x0f03090bu, 0x0f050103u,
    0x0f050109u, 0x0f050301u, 0x0f05030du, 0x0f050503u,
    0x0f050701u, 0x0f050b03u, 0x0f070105u, 0x0f070705u,
    0x0f07070bu, 0x0f070b07u, 0x0f090103u, 0x0f09010bu,
    0x0f090307u, 0x0f090501u, 0x0f090b01u, 0x0f0b0505u,
    0x0f0b0905u, 0x0f0d0105u, 0x0f0d0703u, 0x0f0f0101u,
};

#define IQ_IQ2_XXS 0
#define IQ_IQ2_XS 1
#define IQ_IQ2_S 2
#define IQ_IQ3_XXS 3
#define IQ_IQ3_S 4

__device__ __forceinline__ unsigned int iq_block_bytes(int format) {
    switch (format) {
        case IQ_IQ2_XXS: return 64u;
        case IQ_IQ2_XS: return 72u;
        case IQ_IQ2_S: return 80u;
        case IQ_IQ3_XXS: return 96u;
        default: return 108u;
    }
}

// `ksigns_iq2xs`: a 7-bit sign pattern plus the bit making the count of
// negatives even.
__device__ __forceinline__ unsigned int iq_signs(unsigned int index) {
    index &= 0x7Fu;
    return index | ((__popc(index) & 1u) << 7);
}

__device__ __forceinline__ unsigned int iq_word(const unsigned char* block, unsigned int offset) {
    return *reinterpret_cast<const unsigned int*>(block + offset);
}

// Group `l` (values 8 l .. 8 l + 7) of run `ib` of a super-block: its grid
// values as two words, its sign byte and its scale factor (before `d`).
__device__ __forceinline__ void iq_group(
    int format,
    const unsigned int* grid,
    const unsigned char* block,
    unsigned int ib,
    unsigned int l,
    unsigned int* w0,
    unsigned int* w1,
    unsigned int* signs,
    float* scale
) {
    switch (format) {
        case IQ_IQ2_XXS: {
            unsigned int index = (iq_word(block, 8 * ib) >> (8 * l)) & 0xFFu;
            unsigned int aux = iq_word(block, 8 * ib + 4);
            *w0 = grid[2 * index];
            *w1 = grid[2 * index + 1];
            *signs = iq_signs(aux >> (7 * l));
            *scale = (0.5f + (float)(aux >> 28)) * 0.25f;
            break;
        }
        case IQ_IQ2_XS: {
            unsigned int code = (iq_word(block, 8 * ib + 4 * (l / 2)) >> (16 * (l % 2))) & 0xFFFFu;
            unsigned int index = code & 511u;
            *w0 = grid[2 * index];
            *w1 = grid[2 * index + 1];
            *signs = iq_signs(code >> 9);
            *scale = (0.5f + (float)((block[64 + ib] >> (4 * (l / 2))) & 0xF)) * 0.25f;
            break;
        }
        case IQ_IQ2_S: {
            unsigned int index = block[4 * ib + l] | (((block[64 + ib] >> (2 * l)) & 3u) << 8);
            *w0 = grid[2 * index];
            *w1 = grid[2 * index + 1];
            *signs = block[32 + 4 * ib + l];
            *scale = (0.5f + (float)((block[72 + ib] >> (4 * (l / 2))) & 0xF)) * 0.25f;
            break;
        }
        case IQ_IQ3_XXS: {
            unsigned int aux = iq_word(block, 64 + 4 * ib);
            *w0 = grid[block[8 * ib + 2 * l]];
            *w1 = grid[block[8 * ib + 2 * l + 1]];
            *signs = iq_signs(aux >> (7 * l));
            *scale = (0.5f + (float)(aux >> 28)) * 0.5f;
            break;
        }
        default: {
            unsigned int high = block[64 + ib];
            unsigned int i0 = block[8 * ib + 2 * l] | (((high >> (2 * l)) & 1u) << 8);
            unsigned int i1 = block[8 * ib + 2 * l + 1] | (((high >> (2 * l + 1)) & 1u) << 8);
            *w0 = grid[i0];
            *w1 = grid[i1];
            *signs = block[72 + 4 * ib + l];
            *scale = (float)(1 + 2 * ((block[104 + ib / 2] >> (4 * (ib % 2))) & 0xF));
            break;
        }
    }
}

// The grid of `format` as words: two per 8-value IQ2 entry, one per
// 4-value IQ3 entry.
__device__ __forceinline__ const unsigned int* iq_global_grid(int format, unsigned int* words) {
    switch (format) {
        case IQ_IQ2_XXS: *words = 512; return reinterpret_cast<const unsigned int*>(IQ2XXS_GRID);
        case IQ_IQ2_XS: *words = 1024; return reinterpret_cast<const unsigned int*>(IQ2XS_GRID);
        case IQ_IQ2_S: *words = 2048; return reinterpret_cast<const unsigned int*>(IQ2S_GRID);
        case IQ_IQ3_XXS: *words = 256; return IQ3XXS_GRID;
        default: *words = 512; return IQ3S_GRID;
    }
}

// Bytes 0xFF where the four low bits of `bits` are set.
__device__ __forceinline__ unsigned int iq_sign_mask(unsigned int bits) {
    return (((bits & 0xFu) * 0x00204081u) & 0x01010101u) * 0xFFu;
}

// Group values with their signs applied, as signed bytes.
__device__ __forceinline__ int iq_signed(unsigned int values, unsigned int mask) {
    return (int)__vsub4(values ^ mask, mask);
}

// The logical matrix as f32, one thread per stored element.
extern "C" __global__ void iq_dequantize_kernel(
    int format,
    const unsigned char* blocks,
    const float* d,
    float* out,
    unsigned long long stored_rows,
    unsigned long long stored_cols,
    unsigned int transposed,
    // An embedding lookup's ids, or null (`packed_element`).
    const float* row_ids,
    unsigned long long id_count,
    int* invalid_id_flag
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long r, c;
    if (!packed_element(idx, stored_rows, stored_cols, row_ids, id_count, invalid_id_flag, &r, &c)) {
        return;
    }
    unsigned long long super_block = (r * stored_cols + c) / 256;
    unsigned int w = (unsigned int)(c % 256);
    unsigned int words;
    const unsigned int* grid = iq_global_grid(format, &words);
    unsigned int w0, w1, signs;
    float scale;
    iq_group(format, grid, blocks + super_block * iq_block_bytes(format), w / 32, (w % 32) / 8,
        &w0, &w1, &signs, &scale);
    unsigned int j = w % 8;
    unsigned int byte = ((j < 4 ? w0 : w1) >> (8 * (j % 4))) & 0xFFu;
    float value = d[super_block] * scale * (float)byte;
    if ((signs >> j) & 1u) {
        value = -value;
    }
    if (transposed) {
        out[c * stored_rows + r] = value;
    } else {
        out[idx] = value;
    }
}

// One run's (32 values) contribution against its int8 activations `x`
// (eight words), before `d` and the activation scale.
__device__ __forceinline__ float iq_run_dot(
    int format,
    const unsigned int* grid,
    const unsigned char* block,
    unsigned int ib,
    const int* x
) {
    float total = 0.0f;
#pragma unroll
    for (unsigned int l = 0; l < 4; l++) {
        unsigned int w0, w1, signs;
        float scale;
        iq_group(format, grid, block, ib, l, &w0, &w1, &signs, &scale);
        int dot = i8_dot4(iq_signed(w0, iq_sign_mask(signs)), x[2 * l],
            i8_dot4(iq_signed(w1, iq_sign_mask(signs >> 4)), x[2 * l + 1], 0));
        total += scale * (float)dot;
    }
    return total;
}

// Loads `format`'s grid into `shared` (every thread of the block takes
// part, so call it before any early return).
__device__ __forceinline__ void iq_load_grid(int format, unsigned int* shared) {
    unsigned int words;
    const unsigned int* grid = iq_global_grid(format, &words);
    unsigned int threads = blockDim.x * blockDim.y;
    for (unsigned int i = threadIdx.y * blockDim.x + threadIdx.x; i < words; i += threads) {
        shared[i] = grid[i];
    }
    __syncthreads();
}

#define IQ_COLS 2

// out[r, c] = sum_k a[r, k] * W[k, c] (+ addend) for a packed IQ2/IQ3
// weight stored transposed, over int8 activations: one warp per IQ_COLS
// columns, eight lanes per super-block, a lane's 32 values one activation
// block; the grid read from shared memory.
extern "C" __global__ void iq_matmul_rows_int8_kernel(
    int format,
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
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
    __shared__ unsigned int grid[2048];
    iq_load_grid(format, grid);
    unsigned long long col0 =
        ((unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y) * IQ_COLS;
    unsigned long long first_row = (unsigned long long)blockIdx.y * Q8_MATMUL_ROWS;
    if (col0 >= n || first_row >= m) {
        return;
    }
    unsigned int rows = (unsigned int)min((unsigned long long)Q8_MATMUL_ROWS, m - first_row);
    unsigned int cols = (unsigned int)min((unsigned long long)IQ_COLS, n - col0);
    unsigned int lane = threadIdx.x;
    unsigned long long super_blocks = k / 256;
    unsigned int ib = lane % 8;
    unsigned int bytes = iq_block_bytes(format);
    float acc[Q8_MATMUL_ROWS][IQ_COLS];
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
#pragma unroll
        for (int c = 0; c < IQ_COLS; c++) {
            acc[r][c] = 0.0f;
        }
    }
    for (unsigned long long sb = lane / 8; sb < super_blocks; sb += 4) {
        unsigned long long base = sb * 256 + 32 * ib;
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                unsigned long long at = (first_row + r) * k + base;
                int4 x0 = *reinterpret_cast<const int4*>(aq + at);
                int4 x1 = *reinterpret_cast<const int4*>(aq + at + 16);
                int x[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                float da = ad[at >> 5];
#pragma unroll
                for (int c = 0; c < IQ_COLS; c++) {
                    unsigned int column = (unsigned int)c < cols ? c : 0;
                    unsigned long long index = (col0 + column) * super_blocks + sb;
                    acc[r][c] += da * d[index]
                        * iq_run_dot(format, grid, blocks + index * bytes, ib, x);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
#pragma unroll
        for (int c = 0; c < IQ_COLS; c++) {
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

// `silu(a x gate) * (a x up)` for two packed IQ2/IQ3 weights of one
// format, stored transposed, over int8 activations.
extern "C" __global__ void iq_gated_matmul_rows_int8_kernel(
    int format,
    const signed char* aq,
    const float* ad,
    const unsigned char* gate_blocks,
    const float* gate_d,
    const unsigned char* up_blocks,
    const float* up_d,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
    __shared__ unsigned int grid[2048];
    iq_load_grid(format, grid);
    unsigned long long col = (unsigned long long)blockIdx.x * Q8_MATMUL_WARPS + threadIdx.y;
    unsigned long long first_row = (unsigned long long)blockIdx.y * Q8_MATMUL_ROWS;
    if (col >= n || first_row >= m) {
        return;
    }
    unsigned int rows = (unsigned int)min((unsigned long long)Q8_MATMUL_ROWS, m - first_row);
    unsigned int lane = threadIdx.x;
    unsigned long long super_blocks = k / 256;
    unsigned int ib = lane % 8;
    unsigned int bytes = iq_block_bytes(format);
    float gate_acc[Q8_MATMUL_ROWS];
    float up_acc[Q8_MATMUL_ROWS];
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
        gate_acc[r] = 0.0f;
        up_acc[r] = 0.0f;
    }
    for (unsigned long long sb = lane / 8; sb < super_blocks; sb += 4) {
        unsigned long long index = col * super_blocks + sb;
        unsigned long long base = sb * 256 + 32 * ib;
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                unsigned long long at = (first_row + r) * k + base;
                int4 x0 = *reinterpret_cast<const int4*>(aq + at);
                int4 x1 = *reinterpret_cast<const int4*>(aq + at + 16);
                int x[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                float da = ad[at >> 5];
                gate_acc[r] += da * gate_d[index]
                    * iq_run_dot(format, grid, gate_blocks + index * bytes, ib, x);
                up_acc[r] += da * up_d[index]
                    * iq_run_dot(format, grid, up_blocks + index * bytes, ib, x);
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
// Packed `Q5_K` weights (`execute-direct-quantized-weights`): GGUF
// `block_q5_K` super-blocks kept as stored, 176 bytes each -- `f16` `d`
// and `dmin`, the twelve `Q4_K` scale/min bytes, a 32-byte high-bit plane
// `qh`, then the 128 bytes of low nibbles laid out as `Q4_K`'s. Value `64 c
// + i` (c = 0..3, i = 0..31) is `(qs[32 c + i] & 0xF) | (bit 2c of qh[i]) <<
// 4`, value `64 c + 32 + i` is `(qs[32 c + i] >> 4) | (bit 2c + 1 of qh[i])
// << 4`; `value = d * scale * q - dmin * min` (`ggml-quants.c`'s
// `dequantize_row_q5_K`). Every super-block stays 16-byte aligned.

#define Q5K_BLOCK_BYTES 176

// The logical matrix as f32, one thread per stored element (as
// `q4k_dequantize_kernel`).
extern "C" __global__ void q5k_dequantize_kernel(
    const unsigned char* blocks,
    float* out,
    unsigned long long stored_rows,
    unsigned long long stored_cols,
    unsigned int transposed,
    // An embedding lookup's ids, or null (`packed_element`).
    const float* row_ids,
    unsigned long long id_count,
    int* invalid_id_flag
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long r, c;
    if (!packed_element(idx, stored_rows, stored_cols, row_ids, id_count, invalid_id_flag, &r, &c)) {
        return;
    }
    const unsigned char* block = blocks
        + (r * (stored_cols / Q4K_BLOCK_VALUES) + c / Q4K_BLOCK_VALUES) * Q5K_BLOCK_BYTES;
    unsigned int w = (unsigned int)(c % Q4K_BLOCK_VALUES);
    unsigned int chunk = w / 64;
    unsigned int within = w % 64;
    unsigned char byte = block[48 + 32 * chunk + (within % 32)];
    unsigned char high = block[16 + (within % 32)];
    int sub = 2 * chunk + (within >= 32 ? 1 : 0);
    unsigned int q = (within >= 32 ? (byte >> 4) : (byte & 0xF)) | (((high >> sub) & 1) << 4);
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

// One packed `Q5_K` column's partial dot products for this lane's 32 values
// (16 of sub-block 2c, 16 of 2c + 1) against f32 activations `x`.
__device__ __forceinline__ float q5k_lane_dot(
    const unsigned char* block,
    unsigned int chunk,
    unsigned int offset,
    const float* x
) {
    int4 packed = *reinterpret_cast<const int4*>(block + 48 + 32 * chunk + offset);
    int4 packed_high = *reinterpret_cast<const int4*>(block + 16 + offset);
    const unsigned char* q = reinterpret_cast<const unsigned char*>(&packed);
    const unsigned char* h = reinterpret_cast<const unsigned char*>(&packed_high);
    unsigned int bit_lo = 2 * chunk, bit_hi = 2 * chunk + 1;
    float d = q4k_half(block);
    float dmin = q4k_half(block + 2);
    float scale_lo, min_lo, scale_hi, min_hi;
    q4k_scale_min(2 * chunk, block + 4, &scale_lo, &min_lo);
    q4k_scale_min(2 * chunk + 1, block + 4, &scale_hi, &min_hi);
    float dot_lo = 0.0f, sum_lo = 0.0f, dot_hi = 0.0f, sum_hi = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        float lo = x[j];
        float hi = x[32 + j];
        dot_lo += lo * q4k_nibble((q[j] & 0xF) | (((h[j] >> bit_lo) & 1) << 4));
        dot_hi += hi * q4k_nibble((q[j] >> 4) | (((h[j] >> bit_hi) & 1) << 4));
        sum_lo += lo;
        sum_hi += hi;
    }
    return (d * scale_lo) * dot_lo - (dmin * min_lo) * sum_lo
        + (d * scale_hi) * dot_hi - (dmin * min_hi) * sum_hi;
}

// [`q5k_lane_dot`] over int8 activations (`xlo`, `xhi` with block scales
// `dlo`, `dhi`): the 5-bit values built four at a time and multiplied with
// `__dp4a`.
__device__ __forceinline__ float q5k_lane_dot_int8(
    const unsigned char* block,
    unsigned int chunk,
    unsigned int offset,
    int4 xlo,
    int4 xhi,
    float dlo,
    float dhi
) {
    int4 packed = *reinterpret_cast<const int4*>(block + 48 + 32 * chunk + offset);
    int4 packed_high = *reinterpret_cast<const int4*>(block + 16 + offset);
    float d = q4k_half(block);
    float dmin = q4k_half(block + 2);
    float scale_lo, min_lo, scale_hi, min_hi;
    q4k_scale_min(2 * chunk, block + 4, &scale_lo, &min_lo);
    q4k_scale_min(2 * chunk + 1, block + 4, &scale_hi, &min_hi);
    const int* q = reinterpret_cast<const int*>(&packed);
    const int* h = reinterpret_cast<const int*>(&packed_high);
    const int* lo = reinterpret_cast<const int*>(&xlo);
    const int* hi = reinterpret_cast<const int*>(&xhi);
    unsigned int bit_lo = 2 * chunk, bit_hi = 2 * chunk + 1;
    int dot_lo = 0, dot_hi = 0, sum_lo = 0, sum_hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned int high = (unsigned int)h[j];
        int qlo = (q[j] & 0x0F0F0F0F) | (int)(((high >> bit_lo) & 0x01010101u) << 4);
        int qhi = ((q[j] >> 4) & 0x0F0F0F0F) | (int)(((high >> bit_hi) & 0x01010101u) << 4);
        dot_lo = i8_dot4(qlo, lo[j], dot_lo);
        dot_hi = i8_dot4(qhi, hi[j], dot_hi);
        sum_lo += i8_sum4(lo[j]);
        sum_hi += i8_sum4(hi[j]);
    }
    return dlo * ((d * scale_lo) * (float)dot_lo - (dmin * min_lo) * (float)sum_lo)
        + dhi * ((d * scale_hi) * (float)dot_hi - (dmin * min_hi) * (float)sum_hi);
}

// The final warp reduction and store shared by the `Q5_K` matmuls.
__device__ __forceinline__ void q5k_store(
    float (*acc)[Q4K_COLS],
    float* out,
    unsigned long long first_row,
    unsigned long long col0,
    unsigned int rows,
    unsigned int cols,
    unsigned long long n,
    const float* addend,
    int addend_mode
) {
#pragma unroll
    for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
#pragma unroll
        for (int c = 0; c < Q4K_COLS; c++) {
            float sum = acc[r][c];
#pragma unroll
            for (int offset_ = 16; offset_ > 0; offset_ >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset_);
            }
            if (threadIdx.x == 0 && r < (int)rows && (unsigned int)c < cols) {
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

// out[r, c] = sum_k a[r, k] * W[k, c] (+ addend) for a packed `Q5_K`
// weight stored transposed: `q4k_matmul_rows_kernel`'s layout of warps,
// lanes and columns.
extern "C" __global__ void q5k_matmul_rows_kernel(
    const float* a,
    const unsigned char* blocks,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    const float* addend,
    int addend_mode
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
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
        unsigned long long base = sb * Q4K_BLOCK_VALUES + 64 * chunk + offset;
#pragma unroll
        for (int r = 0; r < Q8_MATMUL_ROWS; r++) {
            if (r < (int)rows) {
                const float* x = a + (first_row + r) * k + base;
#pragma unroll
                for (int c = 0; c < Q4K_COLS; c++) {
                    unsigned int column = (unsigned int)c < cols ? c : 0;
                    const unsigned char* block =
                        blocks + ((col0 + column) * super_blocks + sb) * Q5K_BLOCK_BYTES;
                    acc[r][c] += q5k_lane_dot(block, chunk, offset, x);
                }
            }
        }
    }
    q5k_store(acc, out, first_row, col0, rows, cols, n, addend, addend_mode);
}

// `q5k_matmul_rows_kernel` over int8 activations.
extern "C" __global__ void q5k_matmul_rows_int8_kernel(
    const signed char* aq,
    const float* ad,
    const unsigned char* blocks,
    float* out,
    unsigned long long m,
    unsigned long long k,
    unsigned long long n,
    const float* addend,
    int addend_mode
,
    // replay-bucketed-prefill: trailing padding rows a replayed prefill
    // skips (null outside such a capture).
    const unsigned long long* skipped_rows) {
    if (skipped_rows) {
        m -= *skipped_rows;
    }
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
#pragma unroll
        for (int c = 0; c < Q4K_COLS; c++) {
            unsigned int column = (unsigned int)c < cols ? c : 0;
            block[c] = blocks + ((col0 + column) * super_blocks + sb) * Q5K_BLOCK_BYTES;
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
                    acc[r][c] += q5k_lane_dot_int8(block[c], chunk, offset, xlo, xhi, dlo, dhi);
                }
            }
        }
    }
    q5k_store(acc, out, first_row, col0, rows, cols, n, addend, addend_mode);
}

// Head-wise concatenation (`concat` with `axis = "heads"`,
// `express-latent-attention-forms`): each output row holds `heads` heads,
// head `h` being the first input's head `h` (`a_width` values) followed by
// the second input's head `h`, or its only head when `b_heads == 1`
// (`b_width` values). One thread per output element.
extern "C" __global__ void concat_heads_kernel(
    const float* a,
    const float* b,
    float* out,
    unsigned long long rows,
    unsigned long long heads,
    unsigned long long a_width,
    unsigned long long b_width,
    unsigned long long b_heads
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long width = a_width + b_width;
    if (idx >= rows * heads * width) {
        return;
    }
    unsigned long long row = idx / (heads * width);
    unsigned long long rem = idx % (heads * width);
    unsigned long long head = rem / width;
    unsigned long long j = rem % width;
    out[idx] = j < a_width
        ? a[(row * heads + head) * a_width + j]
        : b[(row * b_heads + (b_heads == 1 ? 0 : head)) * b_width + (j - a_width)];
}

// ---------------------------------------------------------------------------
// TurboQuant KV stores (`turboquant-kv-history`): each head's row is stored
// as its f32 norm and, per coordinate of the rotated unit vector
// `y = Pi x / |x|` (`Pi = H D / sqrt(d)`: the Walsh-Hadamard transform `H`
// after the fixed signs `D`), the index of its nearest Lloyd-Max centroid
// (`magnetar_provider_api::turboquant`). A (row, head)'s codes are
// contiguous: `bits` (3 or 4) per coordinate, little-endian bit order (two
// 4-bit codes per byte, low nibble first). The store also keeps one f32
// offset per column of its rows: the mean of the rows of the write that
// starts its history (`tq_offsets_kernel`). A row is quantized as its
// difference from the offsets, and the store reads back as
// `offsets + |x - offsets| * Pi^T c[idx]`.

#define TQ_MAX_DIM 512
#define TQ_MAX_THREADS 256

// Code `i` of one (row, head)'s packed codes.
__device__ __forceinline__ unsigned int tq_code(
    const unsigned char* codes,
    unsigned int bits,
    unsigned int i
) {
    unsigned int bit = i * bits;
    unsigned int low = codes[bit >> 3];
    unsigned int high = ((bit & 7u) + bits > 8u) ? codes[(bit >> 3) + 1] : 0u;
    return ((low | (high << 8)) >> (bit & 7u)) & ((1u << bits) - 1u);
}

// The unnormalized Walsh-Hadamard transform of `d` (a power of two) values
// in shared memory, in place, by the whole block: the butterflies of
// `walsh_hadamard_in_place`, stride 1 first, so the sums are the same.
__device__ __forceinline__ void tq_hadamard(float* values, unsigned int d) {
    for (unsigned int half = 1; half < d; half <<= 1) {
        for (unsigned int t = threadIdx.x; t < d / 2; t += blockDim.x) {
            unsigned int i = (t / half) * 2 * half + (t % half);
            float a = values[i];
            float b = values[i + half];
            values[i] = a + b;
            values[i + half] = a - b;
        }
        __syncthreads();
    }
}

// A TurboQuant store's offsets, set by the write at row 0: per column of
// the `width`-wide rows, the mean of the first `count - *padding` of its
// `count` rows (a replayed bucketed prefill does not compute its trailing
// padding rows), summed in row order as `turboquant_store_offsets` sums
// them. One thread per column.
extern "C" __global__ void tq_offsets_kernel(
    const float* rows,
    float* offsets,
    unsigned long long width,
    unsigned long long count,
    const unsigned long long* padding
) {
    unsigned long long column = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= width) {
        return;
    }
    unsigned long long skipped = padding ? *padding : 0ull;
    unsigned long long read = count > skipped ? count - skipped : 0ull;
    float sum = 0.0f;
    for (unsigned long long r = 0; r < read; r++) {
        sum += rows[r * width + column];
    }
    offsets[column] = read > 0 ? sum / (float)read : 0.0f;
}

// `kv-write` into a TurboQuant store: block (r, h) quantizes head `h` of
// written row `r`, less the store's offsets, into store row
// `row_offset + step_offset + r`; a store row below `exact_rows` is copied
// as it is into the exact head (`keep-turboquant-kv-head-exact`) instead,
// its codes left unwritten (nothing reads them). The block has a
// power-of-two thread count of at most TQ_MAX_THREADS.
extern "C" __global__ void tq_kv_write_kernel(
    const float* rows,
    const float* offsets,
    unsigned char* codes,
    float* norms,
    const float* signs,
    const float* boundaries,
    unsigned long long heads,
    unsigned long long d,
    unsigned int bits,
    unsigned long long row_offset,
    const unsigned long long* step_offset,
    float* exact,
    unsigned long long exact_rows
) {
    __shared__ float values[TQ_MAX_DIM];
    __shared__ unsigned char indices[TQ_MAX_DIM];
    __shared__ float reduce[TQ_MAX_THREADS];
    unsigned long long r = blockIdx.x;
    unsigned long long h = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int dim = (unsigned int)d;
    const float* x = rows + (r * heads + h) * d;
    unsigned long long row = row_offset + (step_offset ? *step_offset : 0ull) + r;
    if (row < exact_rows) {
        float* out = exact + (row * heads + h) * d;
        for (unsigned int i = tid; i < dim; i += blockDim.x) {
            out[i] = x[i];
        }
        return;
    }
    const float* offset = offsets + h * d;
    float local = 0.0f;
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        float value = x[i] - offset[i];
        values[i] = value;
        local += value * value;
    }
    reduce[tid] = local;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce[tid] += reduce[tid + stride];
        }
        __syncthreads();
    }
    float norm = sqrtf(reduce[0]);
    // A zero row stores zero codes and norm; a non-finite one keeps its
    // norm, so it reads back non-finite.
    bool live = norm > 0.0f && isfinite(norm);
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        values[i] = live ? values[i] / norm * signs[i] : 0.0f;
    }
    __syncthreads();
    tq_hadamard(values, dim);
    float scale = 1.0f / sqrtf((float)dim);
    unsigned int boundary_count = (1u << bits) - 1u;
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        float y = values[i] * scale;
        unsigned int index = 0;
        for (unsigned int b = 0; b < boundary_count; b++) {
            index += boundaries[b] < y ? 1u : 0u;
        }
        indices[i] = live ? (unsigned char)index : (unsigned char)0;
    }
    __syncthreads();
    unsigned int code_bytes = dim * bits / 8u;
    unsigned char* out = codes + (row * heads + h) * code_bytes;
    for (unsigned int byte = tid; byte < code_bytes; byte += blockDim.x) {
        unsigned int first_bit = byte * 8u;
        unsigned int first = first_bit / bits;
        unsigned int last = (first_bit + 7u) / bits;
        unsigned int packed = 0;
        for (unsigned int i = first; i <= last && i < dim; i++) {
            int shift = (int)(i * bits) - (int)first_bit;
            unsigned int index = indices[i];
            packed |= shift >= 0 ? (index << shift) : (index >> (-shift));
        }
        out[byte] = (unsigned char)(packed & 0xFFu);
    }
    if (tid == 0) {
        norms[row * heads + h] = norm;
    }
}

// The f32 values of a TurboQuant store's first `rows + step_offset` rows,
// as `[rows, heads * d]`: block (r, h) reads back head `h` of row `r`, from
// the exact head below `exact_rows` (`keep-turboquant-kv-head-exact`).
// Blocks past those rows return at once, so one launch over the store's
// capacity serves every replay.
extern "C" __global__ void tq_dequantize_kernel(
    const unsigned char* codes,
    const float* norms,
    const float* signs,
    const float* centroids,
    const float* offsets,
    float* out,
    unsigned long long heads,
    unsigned long long d,
    unsigned int bits,
    unsigned long long rows,
    const unsigned long long* step_offset,
    const float* exact,
    unsigned long long exact_rows
) {
    __shared__ float values[TQ_MAX_DIM];
    unsigned long long r = blockIdx.x;
    unsigned long long h = blockIdx.y;
    if (r >= rows + (step_offset ? *step_offset : 0ull)) {
        return;
    }
    unsigned int tid = threadIdx.x;
    unsigned int dim = (unsigned int)d;
    if (r < exact_rows) {
        const float* row = exact + (r * heads + h) * d;
        float* out_row = out + (r * heads + h) * d;
        for (unsigned int i = tid; i < dim; i += blockDim.x) {
            out_row[i] = row[i];
        }
        return;
    }
    unsigned int code_bytes = dim * bits / 8u;
    const unsigned char* row_codes = codes + (r * heads + h) * code_bytes;
    float norm = norms[r * heads + h];
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        values[i] = centroids[tq_code(row_codes, bits, i)];
    }
    __syncthreads();
    tq_hadamard(values, dim);
    float scale = 1.0f / sqrtf((float)dim);
    float* out_row = out + (r * heads + h) * d;
    const float* offset = offsets + h * d;
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        out_row[i] = (norm == 0.0f ? 0.0f : values[i] * (signs[i] * scale) * norm) + offset[i];
    }
}

// `out[h] <- Pi^T out[h] + offsets[h / group_size]` for each of `heads`
// rows of `d` values: the decode attention's rotated value sums back to
// values, plus the value offsets of the query head's KV head (the weights
// sum to one). Block `h`.
extern "C" __global__ void tq_rotate_back_kernel(
    float* out,
    const float* signs,
    const float* offsets,
    unsigned long long d,
    unsigned long long group_size
) {
    __shared__ float values[TQ_MAX_DIM];
    unsigned int dim = (unsigned int)d;
    float* row = out + blockIdx.x * d;
    const float* offset = offsets + (blockIdx.x / group_size) * d;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        values[i] = row[i];
    }
    __syncthreads();
    tq_hadamard(values, dim);
    float scale = 1.0f / sqrtf((float)dim);
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        row[i] = values[i] * (signs[i] * scale) + offset[i];
    }
}

// Split-key decode attention over TurboQuant K and V stores, one query row:
// `attention_split_kernel`'s blocks, chunks and partial layout, computed on
// the codes. The block rotates its query head (`q' = Pi q`), scores a key
// as `q . k_offsets + |k - k_offsets| * sum_i q'_i c_k[idx_i]`, and
// accumulates `p * |v - v_offsets| * c_v[idx_i]`: values in the rotated
// domain, which `attention_combine_kernel` merges and
// `tq_rotate_back_kernel` rotates back and adds the value offsets to.
extern "C" __global__ void tq_attention_split_kernel(
    const float* q,
    const unsigned char* k_codes,
    const float* k_norms,
    const float* k_centroids,
    const float* k_signs,
    const float* k_offsets,
    unsigned int k_bits,
    const unsigned char* v_codes,
    const float* v_norms,
    const float* v_centroids,
    unsigned int v_bits,
    float* partial,
    unsigned long long kv_seq_len,
    unsigned long long head_count,
    unsigned long long kv_head_count,
    unsigned long long head_dimension,
    int causal,
    long long window_size,
    unsigned long long query_position_offset,
    unsigned long long chunks,
    const unsigned long long* step_offset,
    float score_scale,
    unsigned long long value_dimension,
    const float* k_exact,
    const float* v_exact,
    unsigned long long exact_rows,
    const float* v_signs,
    const float* v_offsets
) {
    __shared__ float scores[ATTENTION_CHUNK_KEYS];
    __shared__ float reduce[ATTENTION_SPLIT_THREADS];
    __shared__ float q_shared[TQ_MAX_DIM];
    __shared__ float k_book[16];
    __shared__ float v_book[16];
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
    unsigned long long kv_head = head / group_size;
    unsigned int dim = (unsigned int)head_dimension;
    unsigned int v_dim = (unsigned int)value_dimension;
    float scale = score_scale > 0.0f ? score_scale : 1.0f / sqrtf((float)head_dimension);

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
    float* out = partial + (head * chunks + chunk) * (value_dimension + 2);
    if (first >= last) {
        for (unsigned int d = tid; d < v_dim; d += ATTENTION_SPLIT_THREADS) {
            out[d] = 0.0f;
        }
        if (tid == 0) {
            out[value_dimension] = neg_inf();
            out[value_dimension + 1] = 0.0f;
        }
        return;
    }
    unsigned long long count = last - first;
    // The chunk's keys below `exact_rows` (a prefix) are read from the
    // exact head (`keep-turboquant-kv-head-exact`).
    unsigned long long exact_end = exact_rows < last ? exact_rows : last;
    unsigned long long exact_count = first < exact_end ? exact_end - first : 0ull;

    if (tid < (1u << k_bits)) {
        k_book[tid] = k_centroids[tid];
    }
    if (tid < (1u << v_bits)) {
        v_book[tid] = v_centroids[tid];
    }
    const float* q_row = q + head * head_dimension;
    // Every key's score has `q . k_offsets` of its KV head.
    const float* k_offset = k_offsets + kv_head * head_dimension;
    float offset_part = 0.0f;
    for (unsigned int d = tid; d < dim; d += ATTENTION_SPLIT_THREADS) {
        q_shared[d] = q_row[d] * k_signs[d];
        offset_part += q_row[d] * k_offset[d];
    }
    reduce[tid] = offset_part;
    __syncthreads();
    for (unsigned int stride = ATTENTION_SPLIT_THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            reduce[tid] += reduce[tid + stride];
        }
        __syncthreads();
    }
    float q_offset = reduce[0];
    __syncthreads();
    tq_hadamard(q_shared, dim);
    float rotate_scale = 1.0f / sqrtf((float)dim);
    for (unsigned int d = tid; d < dim; d += ATTENTION_SPLIT_THREADS) {
        q_shared[d] *= rotate_scale;
    }
    __syncthreads();

    unsigned int k_code_bytes = dim * k_bits / 8u;
    for (unsigned long long j = warp; j < count; j += ATTENTION_SPLIT_THREADS / 32) {
        unsigned long long cell = (first + j) * kv_head_count + kv_head;
        float dot = 0.0f;
        if (j < exact_count) {
            // An exact key against the query as it is.
            const float* k_row = k_exact + cell * head_dimension;
            for (unsigned int d = lane; d < dim; d += 32) {
                dot += q_row[d] * k_row[d];
            }
        } else {
            const unsigned char* k_row = k_codes + cell * k_code_bytes;
            for (unsigned int d = lane; d < dim; d += 32) {
                dot += q_shared[d] * k_book[tq_code(k_row, k_bits, d)];
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0) {
            scores[j] = j < exact_count ? dot * scale : (q_offset + dot * k_norms[cell]) * scale;
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
    __syncthreads();
    // Each weight normalized and, for a key on codes, times its value's
    // norm, so the sum below is a weighted mean of rotated values; the
    // exact keys' weights are summed.
    float local_exact = 0.0f;
    for (unsigned long long j = tid; j < count; j += ATTENTION_SPLIT_THREADS) {
        float weight = scores[j] / sum;
        if (j < exact_count) {
            local_exact += weight;
            scores[j] = weight;
        } else {
            scores[j] = weight * v_norms[(first + j) * kv_head_count + kv_head];
        }
    }
    float exact_weight = 0.0f;
    if (exact_count > 0) {
        reduce[tid] = local_exact;
        __syncthreads();
        for (unsigned int stride = ATTENTION_SPLIT_THREADS / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        exact_weight = reduce[0];
    }
    __syncthreads();

    unsigned int v_code_bytes = v_dim * v_bits / 8u;
    const float* v_offset = v_offsets + kv_head * value_dimension;
    for (unsigned int d = tid; d < v_dim; d += ATTENTION_SPLIT_THREADS) {
        float acc = 0.0f;
        for (unsigned long long j = exact_count; j < count; j++) {
            const unsigned char* v_row =
                v_codes + ((first + j) * kv_head_count + kv_head) * v_code_bytes;
            acc += scores[j] * v_book[tq_code(v_row, v_bits, d)];
        }
        out[d] = acc;
        if (exact_count > 0) {
            // The exact values' weighted sum, less the value offsets they
            // carry, rotated below into the codes' domain.
            float exact_acc = 0.0f;
            for (unsigned long long j = 0; j < exact_count; j++) {
                exact_acc += scores[j]
                    * v_exact[((first + j) * kv_head_count + kv_head) * value_dimension + d];
            }
            q_shared[d] = (exact_acc - exact_weight * v_offset[d]) * v_signs[d];
        }
    }
    if (exact_count > 0) {
        __syncthreads();
        tq_hadamard(q_shared, v_dim);
        float v_scale = 1.0f / sqrtf((float)v_dim);
        for (unsigned int d = tid; d < v_dim; d += ATTENTION_SPLIT_THREADS) {
            out[d] += q_shared[d] * v_scale;
        }
    }
    if (tid == 0) {
        out[value_dimension] = max_value;
        out[value_dimension + 1] = sum;
    }
}

// ---------------------------------------------------------------------------
// `cache-latent-attention-history`.

// `batched-matmul`: `out [rows, groups * n]`, group `g` of row `r` being
// `a[r, g*k..(g+1)*k]` times `b[g*k..(g+1)*k, :]` or, with `transpose_b`,
// times `b[g*n..(g+1)*n, :]` transposed. One thread per output value,
// summing in index order as `magnetar_provider_api::batched_matmul`.
extern "C" __global__ void batched_matmul_kernel(
    const float* a,
    const float* b,
    float* out,
    unsigned long long rows,
    unsigned long long groups,
    unsigned long long k,
    unsigned long long n,
    int transpose_b
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long width = groups * n;
    if (index >= rows * width) {
        return;
    }
    unsigned long long r = index / width;
    unsigned long long column = index % width;
    unsigned long long g = column / n;
    unsigned long long j = column % n;
    const float* input = a + r * groups * k + g * k;
    float sum = 0.0f;
    if (transpose_b) {
        const float* weight = b + (g * n + j) * k;
        for (unsigned long long i = 0; i < k; i++) {
            sum += input[i] * weight[i];
        }
    } else {
        const float* weight = b + g * k * n + j;
        for (unsigned long long i = 0; i < k; i++) {
            sum += input[i] * weight[i * n];
        }
    }
    out[index] = sum;
}

// `batched-matmul` with `transpose_b`, one warp per output value: its lanes
// read the input row and the weight row (group `g`'s row `j`, `k` values)
// together, so every load is coalesced. Blocks of `blockDim.x / 32` outputs.
extern "C" __global__ void batched_matmul_t_kernel(
    const float* a,
    const float* b,
    float* out,
    unsigned long long rows,
    unsigned long long groups,
    unsigned long long k,
    unsigned long long n
) {
    unsigned long long index =
        (unsigned long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    unsigned int lane = threadIdx.x & 31;
    unsigned long long width = groups * n;
    if (index >= rows * width) {
        return;
    }
    unsigned long long r = index / width;
    unsigned long long column = index % width;
    unsigned long long g = column / n;
    unsigned long long j = column % n;
    const float* input = a + r * groups * k + g * k;
    const float* weight = b + (g * n + j) * k;
    float sum = 0.0f;
    for (unsigned long long i = lane; i < k; i += 32) {
        sum += input[i] * weight[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        out[index] = sum;
    }
}

// The values `attention` reads under `value_head_stride`: of `rows` rows of
// `heads` heads of `stride` values, each head's leading `width`, as
// `out [rows, heads * width]`. One thread per output value.
extern "C" __global__ void value_heads_kernel(
    const float* v,
    float* out,
    unsigned long long rows,
    unsigned long long heads,
    unsigned long long stride,
    unsigned long long width
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long row_width = heads * width;
    if (index >= rows * row_width) {
        return;
    }
    unsigned long long r = index / row_width;
    unsigned long long h = (index % row_width) / width;
    unsigned long long j = index % width;
    out[index] = v[r * heads * stride + h * stride + j];
}

// ---------------------------------------------------------------------------
// TurboQuant decode attention, grouped (`turboquant-kv-history` task 5.4).

#define TQ_MAX_GROUP 8

// `dot[g] += q[g][d + e] * centroid[e]` for e = 0..7 and every head g of
// the group, in that order, the queries read as float4s
// (`speed-up-decode-attention-on-codes`, decision 5). `q` is 16-byte
// aligned, `dim` and `d` multiples of 8.
__device__ __forceinline__ void tq_dot8(
    float* dot,
    const float* q,
    unsigned int group,
    unsigned int dim,
    unsigned int d,
    const float* centroid
) {
#pragma unroll
    for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
        if (g < group) {
            const float4* row = (const float4*)(q + g * dim + d);
            float4 a = row[0];
            float4 b = row[1];
            dot[g] = fmaf(a.x, centroid[0], dot[g]);
            dot[g] = fmaf(a.y, centroid[1], dot[g]);
            dot[g] = fmaf(a.z, centroid[2], dot[g]);
            dot[g] = fmaf(a.w, centroid[3], dot[g]);
            dot[g] = fmaf(b.x, centroid[4], dot[g]);
            dot[g] = fmaf(b.y, centroid[5], dot[g]);
            dot[g] = fmaf(b.z, centroid[6], dot[g]);
            dot[g] = fmaf(b.w, centroid[7], dot[g]);
        }
    }
}

// `acc[i] += weights[h0 + i] * value` for the `heads` (4 or 8) heads from
// `h0` on, the weights read as float4s; `weights` is a key's row of the
// group's weights, 16-byte aligned.
__device__ __forceinline__ void tq_weigh8(
    float* acc,
    const float* weights,
    unsigned int h0,
    unsigned int heads,
    float value
) {
#pragma unroll
    for (unsigned int i4 = 0; i4 < TQ_MAX_GROUP / 4; i4++) {
        if (i4 * 4 < heads) {
            float4 w = ((const float4*)weights)[h0 / 4 + i4];
            acc[4 * i4] = fmaf(w.x, value, acc[4 * i4]);
            acc[4 * i4 + 1] = fmaf(w.y, value, acc[4 * i4 + 1]);
            acc[4 * i4 + 2] = fmaf(w.z, value, acc[4 * i4 + 2]);
            acc[4 * i4 + 3] = fmaf(w.w, value, acc[4 * i4 + 3]);
        }
    }
}

// `tq_attention_split_kernel` with block (chunk, kv_head) serving the
// `head_count / kv_head_count` (at most TQ_MAX_GROUP) query heads of its KV
// head, so every key and value code is decoded once for them all:
// - the prologue rotates every query head and computes its `q . k_offsets`;
// - one thread per key scores it against every query head (4-bit codes
//   read eight at a time);
// - one warp per query head takes the chunk's softmax, each weight times
//   its value's norm;
// - one thread per value coordinate accumulates every query head.
// The partials are laid out as `tq_attention_split_kernel`'s, per query
// head, for `attention_combine_kernel` and `tq_rotate_back_kernel`.
extern "C" __global__ void tq_attention_split_grouped_kernel(
    const float* q,
    const unsigned char* k_codes,
    const float* k_norms,
    const float* k_centroids,
    const float* k_signs,
    const float* k_offsets,
    unsigned int k_bits,
    const unsigned char* v_codes,
    const float* v_norms,
    const float* v_centroids,
    unsigned int v_bits,
    float* partial,
    unsigned long long kv_seq_len,
    unsigned long long head_count,
    unsigned long long kv_head_count,
    unsigned long long head_dimension,
    int causal,
    long long window_size,
    unsigned long long query_position_offset,
    unsigned long long chunks,
    const unsigned long long* step_offset,
    float score_scale,
    unsigned long long value_dimension,
    const float* k_exact,
    const float* v_exact,
    unsigned long long exact_rows,
    const float* v_signs,
    const float* v_offsets
) {
    // Key-major weights, the group's heads contiguous: a thread loads four
    // heads' weights of a key at once (`speed-up-decode-attention-on-codes`,
    // decision 5).
    __shared__ __align__(16) float scores[ATTENTION_CHUNK_KEYS][TQ_MAX_GROUP];
    __shared__ __align__(16) float q_shared[TQ_MAX_GROUP * TQ_MAX_DIM];
    __shared__ float q_offsets[TQ_MAX_GROUP];
    __shared__ float row_max[TQ_MAX_GROUP];
    __shared__ float row_sum[TQ_MAX_GROUP];
    __shared__ float exact_sum[TQ_MAX_GROUP];
    __shared__ float k_book[16];
    __shared__ float v_book[16];
    if (step_offset) {
        kv_seq_len += *step_offset;
        query_position_offset += *step_offset;
    }
    unsigned long long chunk = blockIdx.x;
    unsigned long long kv_head = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 31;
    unsigned int warp = tid >> 5;
    unsigned int warps = blockDim.x >> 5;
    unsigned int group = (unsigned int)(head_count / kv_head_count);
    unsigned int dim = (unsigned int)head_dimension;
    unsigned int v_dim = (unsigned int)value_dimension;
    float scale = score_scale > 0.0f ? score_scale : 1.0f / sqrtf((float)head_dimension);

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
    unsigned long long head0 = kv_head * group;
    if (first >= last) {
        for (unsigned int g = 0; g < group; g++) {
            float* out = partial + ((head0 + g) * chunks + chunk) * (value_dimension + 2);
            for (unsigned int d = tid; d < v_dim; d += blockDim.x) {
                out[d] = 0.0f;
            }
            if (tid == 0) {
                out[value_dimension] = neg_inf();
                out[value_dimension + 1] = 0.0f;
            }
        }
        return;
    }
    unsigned int count = (unsigned int)(last - first);
    // The chunk's keys below `exact_rows` (a prefix) are read from the
    // exact head (`keep-turboquant-kv-head-exact`).
    unsigned long long exact_end = exact_rows < last ? exact_rows : last;
    unsigned int exact_count = first < exact_end ? (unsigned int)(exact_end - first) : 0u;

    if (tid < (1u << k_bits)) {
        k_book[tid] = k_centroids[tid];
    }
    if (tid < (1u << v_bits)) {
        v_book[tid] = v_centroids[tid];
    }
    // Every query head after the signs, and its `q . k_offsets`.
    const float* k_offset = k_offsets + kv_head * head_dimension;
    for (unsigned int i = tid; i < group * dim; i += blockDim.x) {
        unsigned int d = i % dim;
        q_shared[i] = q[(head0 + i / dim) * head_dimension + d] * k_signs[d];
    }
    for (unsigned int g = warp; g < group; g += warps) {
        const float* q_row = q + (head0 + g) * head_dimension;
        float part = 0.0f;
        for (unsigned int d = lane; d < dim; d += 32) {
            part += q_row[d] * k_offset[d];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            part += __shfl_xor_sync(0xffffffffu, part, offset);
        }
        if (lane == 0) {
            q_offsets[g] = part;
        }
    }
    __syncthreads();
    // The Walsh-Hadamard transform of every head at once (the butterflies
    // of `tq_hadamard`, stride 1 first), then the scale.
    for (unsigned int half = 1; half < dim; half <<= 1) {
        for (unsigned int t = tid; t < group * (dim / 2); t += blockDim.x) {
            unsigned int base = (t / (dim / 2)) * dim;
            unsigned int u = t % (dim / 2);
            unsigned int i = base + (u / half) * 2 * half + (u % half);
            float a = q_shared[i];
            float b = q_shared[i + half];
            q_shared[i] = a + b;
            q_shared[i + half] = a - b;
        }
        __syncthreads();
    }
    float rotate_scale = 1.0f / sqrtf((float)dim);
    for (unsigned int i = tid; i < group * dim; i += blockDim.x) {
        q_shared[i] *= rotate_scale;
    }
    __syncthreads();

    // Scores: one thread per key, every query head.
    unsigned int k_code_bytes = dim * k_bits / 8u;
    for (unsigned int j = tid; j < count; j += blockDim.x) {
        unsigned long long cell = (first + j) * kv_head_count + kv_head;
        const unsigned char* k_row = k_codes + cell * k_code_bytes;
        float dot[TQ_MAX_GROUP];
#pragma unroll
        for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
            dot[g] = 0.0f;
        }
        if (j < exact_count) {
            // An exact key against every query head as it is.
            const float* exact_row = k_exact + cell * head_dimension;
            for (unsigned int d = 0; d < dim; d++) {
                float key = exact_row[d];
#pragma unroll
                for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
                    if (g < group) {
                        dot[g] += q[(head0 + g) * head_dimension + d] * key;
                    }
                }
            }
#pragma unroll
            for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
                if (g < group) {
                    scores[j][g] = dot[g] * scale;
                }
            }
            continue;
        }
        if (k_bits == 4u) {
            const unsigned int* words = (const unsigned int*)k_row;
            for (unsigned int d = 0; d < dim; d += 8) {
                unsigned int word = words[d >> 3];
                float centroid[8];
#pragma unroll
                for (unsigned int e = 0; e < 8; e++) {
                    centroid[e] = k_book[(word >> (4u * e)) & 15u];
                }
                tq_dot8(dot, q_shared, group, dim, d, centroid);
            }
        } else if (k_bits == 3u) {
            // Eight 3-bit codes in every three bytes, little-endian.
            for (unsigned int d = 0; d < dim; d += 8) {
                const unsigned char* bytes = k_row + (d >> 3) * 3u;
                unsigned int word = (unsigned int)bytes[0] | ((unsigned int)bytes[1] << 8)
                    | ((unsigned int)bytes[2] << 16);
                float centroid[8];
#pragma unroll
                for (unsigned int e = 0; e < 8; e++) {
                    centroid[e] = k_book[(word >> (3u * e)) & 7u];
                }
                tq_dot8(dot, q_shared, group, dim, d, centroid);
            }
        } else {
            for (unsigned int d = 0; d < dim; d++) {
                float centroid = k_book[tq_code(k_row, k_bits, d)];
#pragma unroll
                for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
                    if (g < group) {
                        dot[g] += q_shared[g * dim + d] * centroid;
                    }
                }
            }
        }
        float norm = k_norms[cell];
#pragma unroll
        for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
            if (g < group) {
                scores[j][g] = (q_offsets[g] + dot[g] * norm) * scale;
            }
        }
    }
    __syncthreads();

    // Softmax: one warp per query head; each weight normalized and times
    // its value's norm, so the sums below are weighted means of rotated
    // values.
    for (unsigned int g = warp; g < group; g += warps) {
        float local_max = neg_inf();
        for (unsigned int j = lane; j < count; j += 32) {
            local_max = fmaxf(local_max, scores[j][g]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, offset));
        }
        float local_sum = 0.0f;
        for (unsigned int j = lane; j < count; j += 32) {
            float weight = expf(scores[j][g] - local_max);
            scores[j][g] = weight;
            local_sum += weight;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, offset);
        }
        float exact_part = 0.0f;
        for (unsigned int j = lane; j < count; j += 32) {
            float weight = scores[j][g] / local_sum;
            if (j < exact_count) {
                exact_part += weight;
                scores[j][g] = weight;
            } else {
                scores[j][g] = weight * v_norms[(first + j) * kv_head_count + kv_head];
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            exact_part += __shfl_xor_sync(0xffffffffu, exact_part, offset);
        }
        if (lane == 0) {
            row_max[g] = local_max;
            row_sum[g] = local_sum;
            exact_sum[g] = exact_part;
        }
    }
    __syncthreads();

    // Values: thread (d, h) sums coordinate d of `value_heads` query heads
    // from h * value_heads on (four for heads up to 64 wide, so every
    // thread works; all eight otherwise), each code's centroid feeding
    // every head and each float4 of weights four heads. Summed in key
    // order, as one thread per coordinate did.
    unsigned int v_code_bytes = v_dim * v_bits / 8u;
    unsigned int value_heads = v_dim * 2u <= blockDim.x ? 4u : TQ_MAX_GROUP;
    unsigned int head_slices = TQ_MAX_GROUP / value_heads;
    for (unsigned int e = tid; e < v_dim * head_slices; e += blockDim.x) {
        unsigned int d = e % v_dim;
        unsigned int h0 = (e / v_dim) * value_heads;
        if (h0 >= group) {
            continue;
        }
        // The group's heads from h0 on, in whole float4s.
        unsigned int remaining = (group - h0 + 3u) & ~3u;
        unsigned int heads = remaining < value_heads ? remaining : value_heads;
        float acc[TQ_MAX_GROUP];
#pragma unroll
        for (unsigned int i = 0; i < TQ_MAX_GROUP; i++) {
            acc[i] = 0.0f;
        }
        for (unsigned int j = exact_count; j < count; j++) {
            const unsigned char* v_row =
                v_codes + ((first + j) * kv_head_count + kv_head) * v_code_bytes;
            unsigned int code = v_bits == 4u
                ? (v_row[d >> 1] >> ((d & 1u) << 2)) & 15u
                : tq_code(v_row, v_bits, d);
            tq_weigh8(acc, scores[j], h0, heads, v_book[code]);
        }
#pragma unroll
        for (unsigned int i = 0; i < TQ_MAX_GROUP; i++) {
            unsigned int g = h0 + i;
            if (i < value_heads && g < group) {
                partial[((head0 + g) * chunks + chunk) * (value_dimension + 2) + d] = acc[i];
            }
        }
        if (exact_count > 0) {
            // The exact values' weighted sums, less the value offsets they
            // carry, rotated below into the codes' domain.
#pragma unroll
            for (unsigned int i = 0; i < TQ_MAX_GROUP; i++) {
                acc[i] = 0.0f;
            }
            for (unsigned int j = 0; j < exact_count; j++) {
                float value =
                    v_exact[((first + j) * kv_head_count + kv_head) * value_dimension + d];
                tq_weigh8(acc, scores[j], h0, heads, value);
            }
            float offset = v_offsets[kv_head * value_dimension + d];
            // Constant indices into `acc`: indexed at run time, the array
            // would leave the registers for local memory, in every loop.
#pragma unroll
            for (unsigned int i = 0; i < TQ_MAX_GROUP; i++) {
                unsigned int g = h0 + i;
                if (i < value_heads && g < group) {
                    q_shared[g * v_dim + d] = (acc[i] - exact_sum[g] * offset) * v_signs[d];
                }
            }
        }
    }
    if (exact_count > 0) {
        __syncthreads();
        // The Walsh-Hadamard transform of every head's exact sum at once,
        // as the queries' above, then the scale, added to the partials.
        for (unsigned int half = 1; half < v_dim; half <<= 1) {
            for (unsigned int t = tid; t < group * (v_dim / 2); t += blockDim.x) {
                unsigned int base = (t / (v_dim / 2)) * v_dim;
                unsigned int u = t % (v_dim / 2);
                unsigned int i = base + (u / half) * 2 * half + (u % half);
                float a = q_shared[i];
                float b = q_shared[i + half];
                q_shared[i] = a + b;
                q_shared[i + half] = a - b;
            }
            __syncthreads();
        }
        float v_scale = 1.0f / sqrtf((float)v_dim);
        for (unsigned int d = tid; d < v_dim; d += blockDim.x) {
            for (unsigned int g = 0; g < group; g++) {
                partial[((head0 + g) * chunks + chunk) * (value_dimension + 2) + d] +=
                    q_shared[g * v_dim + d] * v_scale;
            }
        }
    }
    if (tid < group) {
        float* out = partial + ((head0 + tid) * chunks + chunk) * (value_dimension + 2);
        out[value_dimension] = row_max[tid];
        out[value_dimension + 1] = row_sum[tid];
    }
}

// ---------------------------------------------------------------------------
// TurboQuant decode attention, staged and split
// (`speed-up-decode-attention-on-codes`).

// `tq_attention_split_grouped_kernel`'s attention for 64- to 128-wide (at
// most) heads, scheduled differently. Block (split, kv_head) serves the
// group's query heads over its share of the admitted keys' chunks of
// ATTENTION_CHUNK_KEYS: `splits` blocks per KV head share the `n` chunks
// computed here from the key count, so a captured step's replays spread
// their longer histories over the same blocks. The block
// - rotates its query heads once (dynamic shared memory sized to the
//   group);
// - per chunk, stages the chunk's K and V codes and norms in shared memory
//   with coalesced loads (rows padded by a word, so a thread per key reads
//   distinct banks), scores the keys one thread per key, and takes an
//   online softmax per query head (running maximum and sum; the
//   accumulators rescaled when the maximum grows);
// - sums the values with every thread: thread `t` owns value coordinate
//   `t % v_dim` of the query heads `t / v_dim + i * (blockDim.x / v_dim)`;
// - keeps an exact head's (`keep-turboquant-kv-head-exact`) keys scored
//   against the query as it is and their values summed unrotated, rotated
//   once at the end into the codes' domain;
// - writes one partial per query head in `attention_split_kernel`'s layout
//   (`splits` partials per head), for `attention_combine_kernel` and
//   `tq_rotate_back_kernel`.
// Dynamic shared memory: `group * head_dimension + group * value_dimension
// + 2 * ATTENTION_CHUNK_KEYS` floats, then ATTENTION_CHUNK_KEYS rows of K
// codes and of V codes, each `code bytes` rounded up to a word plus a word.
extern "C" __global__ void tq_attention_split_staged_kernel(
    const float* q,
    const unsigned char* k_codes,
    const float* k_norms,
    const float* k_centroids,
    const float* k_signs,
    const float* k_offsets,
    unsigned int k_bits,
    const unsigned char* v_codes,
    const float* v_norms,
    const float* v_centroids,
    unsigned int v_bits,
    float* partial,
    unsigned long long kv_seq_len,
    unsigned long long head_count,
    unsigned long long kv_head_count,
    unsigned long long head_dimension,
    int causal,
    long long window_size,
    unsigned long long query_position_offset,
    unsigned long long splits,
    const unsigned long long* step_offset,
    float score_scale,
    unsigned long long value_dimension,
    const float* k_exact,
    const float* v_exact,
    unsigned long long exact_rows,
    const float* v_signs,
    const float* v_offsets
) {
    extern __shared__ float tq_staged[];
    __shared__ float scores[TQ_MAX_GROUP][ATTENTION_CHUNK_KEYS];
    __shared__ float q_offsets[TQ_MAX_GROUP];
    __shared__ float run_max[TQ_MAX_GROUP];
    __shared__ float run_sum[TQ_MAX_GROUP];
    __shared__ float exact_weight[TQ_MAX_GROUP];
    __shared__ float chunk_scale[TQ_MAX_GROUP];
    __shared__ float k_book[16];
    __shared__ float v_book[16];
    if (step_offset) {
        kv_seq_len += *step_offset;
        query_position_offset += *step_offset;
    }
    unsigned long long split = blockIdx.x;
    unsigned long long kv_head = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 31;
    unsigned int warp = tid >> 5;
    unsigned int warps = blockDim.x >> 5;
    unsigned int group = (unsigned int)(head_count / kv_head_count);
    unsigned int dim = (unsigned int)head_dimension;
    unsigned int v_dim = (unsigned int)value_dimension;
    unsigned long long head0 = kv_head * group;
    float scale = score_scale > 0.0f ? score_scale : 1.0f / sqrtf((float)head_dimension);

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
    unsigned long long chunk_count =
        (key_upper - key_lower + ATTENTION_CHUNK_KEYS - 1) / ATTENTION_CHUNK_KEYS;
    unsigned long long per_split = (chunk_count + splits - 1) / splits;
    unsigned long long chunk_begin = split * per_split;
    unsigned long long chunk_end = chunk_begin + per_split;
    if (chunk_end > chunk_count) {
        chunk_end = chunk_count;
    }
    if (chunk_begin >= chunk_end) {
        for (unsigned int g = 0; g < group; g++) {
            float* out = partial + ((head0 + g) * splits + split) * (value_dimension + 2);
            for (unsigned int d = tid; d < v_dim; d += blockDim.x) {
                out[d] = 0.0f;
            }
            if (tid == 0) {
                out[value_dimension] = neg_inf();
                out[value_dimension + 1] = 0.0f;
            }
        }
        return;
    }

    unsigned int k_code_bytes = dim * k_bits / 8u;
    unsigned int v_code_bytes = v_dim * v_bits / 8u;
    unsigned int k_row = ((k_code_bytes + 3u) & ~3u) + 4u;
    unsigned int v_row = ((v_code_bytes + 3u) & ~3u) + 4u;
    float* q_rot = tq_staged;
    float* exact_rotated = q_rot + group * dim;
    float* k_norm_s = exact_rotated + group * v_dim;
    float* v_norm_s = k_norm_s + ATTENTION_CHUNK_KEYS;
    unsigned char* k_stage = (unsigned char*)(v_norm_s + ATTENTION_CHUNK_KEYS);
    unsigned char* v_stage = k_stage + ATTENTION_CHUNK_KEYS * k_row;

    // The value coordinate and query heads this thread owns.
    unsigned int slices = v_dim < blockDim.x ? blockDim.x / v_dim : 1u;
    unsigned int own_d = tid % v_dim;
    unsigned int slice = tid / v_dim;
    bool owns = tid < slices * v_dim;

    if (tid < (1u << k_bits)) {
        k_book[tid] = k_centroids[tid];
    }
    if (tid < (1u << v_bits)) {
        v_book[tid] = v_centroids[tid];
    }
    // Every query head after the signs, and its `q . k_offsets`.
    const float* k_offset = k_offsets + kv_head * head_dimension;
    for (unsigned int i = tid; i < group * dim; i += blockDim.x) {
        unsigned int d = i % dim;
        q_rot[i] = q[(head0 + i / dim) * head_dimension + d] * k_signs[d];
    }
    for (unsigned int g = warp; g < group; g += warps) {
        const float* q_row = q + (head0 + g) * head_dimension;
        float part = 0.0f;
        for (unsigned int d = lane; d < dim; d += 32) {
            part += q_row[d] * k_offset[d];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            part += __shfl_xor_sync(0xffffffffu, part, offset);
        }
        if (lane == 0) {
            q_offsets[g] = part;
            run_max[g] = neg_inf();
            run_sum[g] = 0.0f;
            exact_weight[g] = 0.0f;
        }
    }
    __syncthreads();
    for (unsigned int half = 1; half < dim; half <<= 1) {
        for (unsigned int t = tid; t < group * (dim / 2); t += blockDim.x) {
            unsigned int base = (t / (dim / 2)) * dim;
            unsigned int u = t % (dim / 2);
            unsigned int i = base + (u / half) * 2 * half + (u % half);
            float a = q_rot[i];
            float b = q_rot[i + half];
            q_rot[i] = a + b;
            q_rot[i + half] = a - b;
        }
        __syncthreads();
    }
    float rotate_scale = 1.0f / sqrtf((float)dim);
    for (unsigned int i = tid; i < group * dim; i += blockDim.x) {
        q_rot[i] *= rotate_scale;
    }
    __syncthreads();

    // The running sums of the owned coordinate: of rotated values (codes)
    // and of unrotated ones (the exact head), relative to `run_max`.
    float acc[TQ_MAX_GROUP];
    float acc_exact[TQ_MAX_GROUP];
#pragma unroll
    for (unsigned int i = 0; i < TQ_MAX_GROUP; i++) {
        acc[i] = 0.0f;
        acc_exact[i] = 0.0f;
    }
    bool any_exact = false;
    bool word_rows = (k_code_bytes & 3u) == 0 && (v_code_bytes & 3u) == 0;

    for (unsigned long long c = chunk_begin; c < chunk_end; c++) {
        unsigned long long first = key_lower + c * ATTENTION_CHUNK_KEYS;
        unsigned long long last = first + ATTENTION_CHUNK_KEYS;
        if (last > key_upper) {
            last = key_upper;
        }
        unsigned int count = (unsigned int)(last - first);
        unsigned long long exact_end = exact_rows < last ? exact_rows : last;
        unsigned int exact_count = first < exact_end ? (unsigned int)(exact_end - first) : 0u;
        any_exact = any_exact || exact_count > 0;

        // Stage the coded keys' codes and norms, consecutive threads
        // reading consecutive words (or bytes) of a row.
        unsigned int coded = count - exact_count;
        if (word_rows) {
            unsigned int k_words = k_code_bytes / 4u;
            unsigned int v_words = v_code_bytes / 4u;
            const unsigned int* k_src = (const unsigned int*)k_codes;
            const unsigned int* v_src = (const unsigned int*)v_codes;
            for (unsigned int e = tid; e < coded * k_words; e += blockDim.x) {
                unsigned int j = exact_count + e / k_words;
                unsigned long long cell = (first + j) * kv_head_count + kv_head;
                ((unsigned int*)(k_stage + j * k_row))[e % k_words] =
                    k_src[cell * k_words + e % k_words];
            }
            for (unsigned int e = tid; e < coded * v_words; e += blockDim.x) {
                unsigned int j = exact_count + e / v_words;
                unsigned long long cell = (first + j) * kv_head_count + kv_head;
                ((unsigned int*)(v_stage + j * v_row))[e % v_words] =
                    v_src[cell * v_words + e % v_words];
            }
        } else {
            for (unsigned int e = tid; e < coded * k_code_bytes; e += blockDim.x) {
                unsigned int j = exact_count + e / k_code_bytes;
                unsigned long long cell = (first + j) * kv_head_count + kv_head;
                k_stage[j * k_row + e % k_code_bytes] =
                    k_codes[cell * k_code_bytes + e % k_code_bytes];
            }
            for (unsigned int e = tid; e < coded * v_code_bytes; e += blockDim.x) {
                unsigned int j = exact_count + e / v_code_bytes;
                unsigned long long cell = (first + j) * kv_head_count + kv_head;
                v_stage[j * v_row + e % v_code_bytes] =
                    v_codes[cell * v_code_bytes + e % v_code_bytes];
            }
        }
        for (unsigned int j = exact_count + tid; j < count; j += blockDim.x) {
            unsigned long long cell = (first + j) * kv_head_count + kv_head;
            k_norm_s[j] = k_norms[cell];
            v_norm_s[j] = v_norms[cell];
        }
        __syncthreads();

        // Scores: one thread per key, every query head.
        for (unsigned int j = tid; j < count; j += blockDim.x) {
            float dot[TQ_MAX_GROUP];
#pragma unroll
            for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
                dot[g] = 0.0f;
            }
            if (j < exact_count) {
                // An exact key against every query head as it is.
                const float* exact_row =
                    k_exact + ((first + j) * kv_head_count + kv_head) * head_dimension;
                for (unsigned int d = 0; d < dim; d++) {
                    float key = exact_row[d];
#pragma unroll
                    for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
                        if (g < group) {
                            dot[g] += q[(head0 + g) * head_dimension + d] * key;
                        }
                    }
                }
#pragma unroll
                for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
                    if (g < group) {
                        scores[g][j] = dot[g] * scale;
                    }
                }
                continue;
            }
            const unsigned char* k_row_codes = k_stage + j * k_row;
            if (k_bits == 4u && word_rows) {
                const unsigned int* words = (const unsigned int*)k_row_codes;
                for (unsigned int d = 0; d < dim; d += 8) {
                    unsigned int word = words[d >> 3];
#pragma unroll
                    for (unsigned int e = 0; e < 8; e++) {
                        float centroid = k_book[(word >> (4u * e)) & 15u];
#pragma unroll
                        for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
                            if (g < group) {
                                dot[g] += q_rot[g * dim + d + e] * centroid;
                            }
                        }
                    }
                }
            } else if (k_bits == 3u && (dim & 7u) == 0) {
                // Eight 3-bit codes in every three bytes, little-endian.
                for (unsigned int d = 0; d < dim; d += 8) {
                    const unsigned char* bytes = k_row_codes + (d >> 3) * 3u;
                    unsigned int word = (unsigned int)bytes[0] | ((unsigned int)bytes[1] << 8)
                        | ((unsigned int)bytes[2] << 16);
#pragma unroll
                    for (unsigned int e = 0; e < 8; e++) {
                        float centroid = k_book[(word >> (3u * e)) & 7u];
#pragma unroll
                        for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
                            if (g < group) {
                                dot[g] += q_rot[g * dim + d + e] * centroid;
                            }
                        }
                    }
                }
            } else {
                for (unsigned int d = 0; d < dim; d++) {
                    float centroid = k_book[tq_code(k_row_codes, k_bits, d)];
#pragma unroll
                    for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
                        if (g < group) {
                            dot[g] += q_rot[g * dim + d] * centroid;
                        }
                    }
                }
            }
            float norm = k_norm_s[j];
#pragma unroll
            for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
                if (g < group) {
                    scores[g][j] = (q_offsets[g] + dot[g] * norm) * scale;
                }
            }
        }
        __syncthreads();

        // Online softmax: a warp per query head. The chunk's weights are
        // taken against the new running maximum; the running sums and the
        // owned accumulators are rescaled by `chunk_scale`. A key on codes
        // weighs with its value's norm.
        for (unsigned int g = warp; g < group; g += warps) {
            float chunk_max = neg_inf();
            for (unsigned int j = lane; j < count; j += 32) {
                chunk_max = fmaxf(chunk_max, scores[g][j]);
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                chunk_max = fmaxf(chunk_max, __shfl_xor_sync(0xffffffffu, chunk_max, offset));
            }
            float old_max = run_max[g];
            float new_max = fmaxf(old_max, chunk_max);
            float rescale = old_max == neg_inf() ? 0.0f : expf(old_max - new_max);
            float sum = 0.0f;
            float exact_part = 0.0f;
            for (unsigned int j = lane; j < count; j += 32) {
                float weight = expf(scores[g][j] - new_max);
                sum += weight;
                if (j < exact_count) {
                    exact_part += weight;
                    scores[g][j] = weight;
                } else {
                    scores[g][j] = weight * v_norm_s[j];
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset);
                exact_part += __shfl_xor_sync(0xffffffffu, exact_part, offset);
            }
            if (lane == 0) {
                run_max[g] = new_max;
                run_sum[g] = run_sum[g] * rescale + sum;
                exact_weight[g] = exact_weight[g] * rescale + exact_part;
                chunk_scale[g] = rescale;
            }
        }
        __syncthreads();

        // Values: every thread its coordinate of its query heads.
        if (owns) {
#pragma unroll
            for (unsigned int i = 0; i < TQ_MAX_GROUP; i++) {
                unsigned int g = slice + i * slices;
                if (g < group) {
                    acc[i] *= chunk_scale[g];
                    acc_exact[i] *= chunk_scale[g];
                }
            }
            for (unsigned int j = exact_count; j < count; j++) {
                const unsigned char* row = v_stage + j * v_row;
                unsigned int code = v_bits == 4u
                    ? (row[own_d >> 1] >> ((own_d & 1u) << 2)) & 15u
                    : tq_code(row, v_bits, own_d);
                float centroid = v_book[code];
#pragma unroll
                for (unsigned int i = 0; i < TQ_MAX_GROUP; i++) {
                    unsigned int g = slice + i * slices;
                    if (g < group) {
                        acc[i] += scores[g][j] * centroid;
                    }
                }
            }
            for (unsigned int j = 0; j < exact_count; j++) {
                float value =
                    v_exact[((first + j) * kv_head_count + kv_head) * value_dimension + own_d];
#pragma unroll
                for (unsigned int i = 0; i < TQ_MAX_GROUP; i++) {
                    unsigned int g = slice + i * slices;
                    if (g < group) {
                        acc_exact[i] += scores[g][j] * value;
                    }
                }
            }
        }
        // The next chunk restages the codes and rewrites the scores.
        __syncthreads();
    }

    if (any_exact) {
        // The exact sums, less the value offsets they carry, rotated into
        // the codes' domain, every query head at once.
        if (owns) {
            float offset = v_offsets[kv_head * value_dimension + own_d];
            float sign = v_signs[own_d];
#pragma unroll
            for (unsigned int i = 0; i < TQ_MAX_GROUP; i++) {
                unsigned int g = slice + i * slices;
                if (g < group) {
                    exact_rotated[g * v_dim + own_d] =
                        (acc_exact[i] - exact_weight[g] * offset) * sign;
                }
            }
        }
        __syncthreads();
        for (unsigned int half = 1; half < v_dim; half <<= 1) {
            for (unsigned int t = tid; t < group * (v_dim / 2); t += blockDim.x) {
                unsigned int base = (t / (v_dim / 2)) * v_dim;
                unsigned int u = t % (v_dim / 2);
                unsigned int i = base + (u / half) * 2 * half + (u % half);
                float a = exact_rotated[i];
                float b = exact_rotated[i + half];
                exact_rotated[i] = a + b;
                exact_rotated[i + half] = a - b;
            }
            __syncthreads();
        }
        if (owns) {
            float v_scale = 1.0f / sqrtf((float)v_dim);
#pragma unroll
            for (unsigned int i = 0; i < TQ_MAX_GROUP; i++) {
                unsigned int g = slice + i * slices;
                if (g < group) {
                    acc[i] += exact_rotated[g * v_dim + own_d] * v_scale;
                }
            }
        }
    }

    // The block's partials: weighted means and their statistics.
    if (owns) {
#pragma unroll
        for (unsigned int i = 0; i < TQ_MAX_GROUP; i++) {
            unsigned int g = slice + i * slices;
            if (g < group) {
                partial[((head0 + g) * splits + split) * (value_dimension + 2) + own_d] =
                    acc[i] / run_sum[g];
            }
        }
    }
    if (tid < group) {
        float* out = partial + ((head0 + tid) * splits + split) * (value_dimension + 2);
        out[value_dimension] = run_max[tid];
        out[value_dimension + 1] = run_sum[tid];
    }
}

// ---------------------------------------------------------------------------
// TurboQuant decode attention over a latent history
// (`cache-latent-attention-history`, `speed-up-decode-attention-on-codes`).

// One query row against a latent TurboQuant store read as both K and V: a
// row of `k_groups` groups of `d` values (each group quantized on its own:
// codes, a norm, the store's offsets, the signs of `d`), each query head
// `k_groups * d` wide, the values each row's leading `v_groups` groups.
// Block (chunk, head block) serves up to TQ_MAX_GROUP query heads over a
// chunk of ATTENTION_CHUNK_KEYS keys, so every code is decoded once for
// them all:
// - the prologue rotates every group of every query head (`q' = Pi q`, a
//   transform per group) and computes each head's `q . offsets`;
// - one thread per key scores it against every query head, a group at a
//   time (`q . k = q . offsets + sum_g |k_g - o_g| q'_g . c_g`);
// - one warp per query head takes the chunk's softmax;
// - thread (d, h) sums value coordinate d of `value_heads` heads, each
//   value `|v_g - o_g| c_g[d]` in its group's rotated domain;
// - an exact head's (`keep-turboquant-kv-head-exact`) keys are scored as
//   they are and their values summed unrotated, less the offsets, then
//   rotated group by group into the codes' domain.
// The partials are laid out as `attention_split_kernel`'s
// (`v_groups * d + 2` floats per head and chunk), for
// `attention_combine_kernel` and `tq_latent_rotate_back_kernel`.
extern "C" __global__ void tq_latent_attention_split_kernel(
    const float* q,
    const unsigned char* codes,
    const float* norms,
    const float* centroids,
    const float* signs,
    const float* offsets,
    unsigned int bits,
    float* partial,
    unsigned long long kv_seq_len,
    unsigned long long head_count,
    unsigned long long k_groups,
    unsigned long long v_groups,
    unsigned long long d,
    int causal,
    long long window_size,
    unsigned long long query_position_offset,
    unsigned long long chunks,
    const unsigned long long* step_offset,
    float score_scale,
    const float* exact,
    unsigned long long exact_rows
) {
    extern __shared__ __align__(16) float latent_shared[];
    __shared__ __align__(16) float scores[ATTENTION_CHUNK_KEYS][TQ_MAX_GROUP];
    __shared__ float q_offsets[TQ_MAX_GROUP];
    __shared__ float row_max[TQ_MAX_GROUP];
    __shared__ float row_sum[TQ_MAX_GROUP];
    __shared__ float exact_sum[TQ_MAX_GROUP];
    __shared__ float book[16];
    if (step_offset) {
        kv_seq_len += *step_offset;
        query_position_offset += *step_offset;
    }
    unsigned long long chunk = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 31;
    unsigned int warp = tid >> 5;
    unsigned int warps = blockDim.x >> 5;
    unsigned int dim = (unsigned int)d;
    unsigned int k_width = (unsigned int)(k_groups * d);
    unsigned int v_width = (unsigned int)(v_groups * d);
    unsigned long long head0 = (unsigned long long)blockIdx.y * TQ_MAX_GROUP;
    unsigned int group = (unsigned int)(head_count - head0 < TQ_MAX_GROUP
        ? head_count - head0 : TQ_MAX_GROUP);
    float scale = score_scale > 0.0f ? score_scale : 1.0f / sqrtf((float)k_width);
    // The rotated queries, `group` rows of `k_width`; later the exact
    // values' sums, `group` rows of `v_width`.
    float* q_rot = latent_shared;

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
    if (first >= last) {
        for (unsigned int g = 0; g < group; g++) {
            float* out = partial + ((head0 + g) * chunks + chunk) * (v_width + 2);
            for (unsigned int i = tid; i < v_width; i += blockDim.x) {
                out[i] = 0.0f;
            }
            if (tid == 0) {
                out[v_width] = neg_inf();
                out[v_width + 1] = 0.0f;
            }
        }
        return;
    }
    unsigned int count = (unsigned int)(last - first);
    unsigned long long exact_end = exact_rows < last ? exact_rows : last;
    unsigned int exact_count = first < exact_end ? (unsigned int)(exact_end - first) : 0u;

    if (tid < (1u << bits)) {
        book[tid] = centroids[tid];
    }
    // Every query head after the signs (group by group), and its
    // `q . offsets`.
    for (unsigned int i = tid; i < group * k_width; i += blockDim.x) {
        q_rot[i] = q[(head0 + i / k_width) * k_width + i % k_width] * signs[i % dim];
    }
    for (unsigned int g = warp; g < group; g += warps) {
        const float* q_row = q + (head0 + g) * k_width;
        float part = 0.0f;
        for (unsigned int i = lane; i < k_width; i += 32) {
            part += q_row[i] * offsets[i];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            part += __shfl_xor_sync(0xffffffffu, part, offset);
        }
        if (lane == 0) {
            q_offsets[g] = part;
        }
    }
    __syncthreads();
    // The Walsh-Hadamard transform of every group of every head at once.
    unsigned int transforms = group * (unsigned int)k_groups;
    for (unsigned int half = 1; half < dim; half <<= 1) {
        for (unsigned int t = tid; t < transforms * (dim / 2); t += blockDim.x) {
            unsigned int base = (t / (dim / 2)) * dim;
            unsigned int u = t % (dim / 2);
            unsigned int i = base + (u / half) * 2 * half + (u % half);
            float a = q_rot[i];
            float b = q_rot[i + half];
            q_rot[i] = a + b;
            q_rot[i + half] = a - b;
        }
        __syncthreads();
    }
    float rotate_scale = 1.0f / sqrtf((float)dim);
    for (unsigned int i = tid; i < group * k_width; i += blockDim.x) {
        q_rot[i] *= rotate_scale;
    }
    __syncthreads();

    // Scores: four threads per key (LATENT_THREADS = 4 *
    // ATTENTION_CHUNK_KEYS), thread `part` of a key taking the eight-value
    // blocks part, part + 4, ... of every group (an exact key: its float4s
    // part, part + 4, ...) against every query head, so the four read
    // rotated queries in different shared-memory banks; the four sums are
    // added by a fixed butterfly.
    unsigned int code_bytes = dim * bits / 8u;
    {
        unsigned int j = tid >> 2;
        unsigned int part = tid & 3u;
        unsigned long long row = first + j;
        float dot[TQ_MAX_GROUP];
#pragma unroll
        for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
            dot[g] = 0.0f;
        }
        if (j < exact_count) {
            // An exact key against every query head as it is, from float4s
            // of its row and of the queries (every thread reads the same
            // query values, which the cache serves).
            const float4* exact_row = (const float4*)(exact + row * k_width);
            for (unsigned int c = part; c < k_width / 4; c += 4) {
                float4 key = exact_row[c];
#pragma unroll
                for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
                    if (g < group) {
                        float4 query = ((const float4*)(q + (head0 + g) * k_width))[c];
                        dot[g] = fmaf(query.x, key.x, dot[g]);
                        dot[g] = fmaf(query.y, key.y, dot[g]);
                        dot[g] = fmaf(query.z, key.z, dot[g]);
                        dot[g] = fmaf(query.w, key.w, dot[g]);
                    }
                }
            }
        } else if (j < count) {
            for (unsigned int kg = 0; kg < (unsigned int)k_groups; kg++) {
                unsigned long long cell = row * k_groups + kg;
                const unsigned char* row_codes = codes + cell * code_bytes;
                float code_part[TQ_MAX_GROUP];
#pragma unroll
                for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
                    code_part[g] = 0.0f;
                }
                for (unsigned int e8 = 8u * part; e8 < dim; e8 += 32u) {
                    float centroid[8];
                    if (bits == 4u) {
                        unsigned int word = ((const unsigned int*)row_codes)[e8 >> 3];
#pragma unroll
                        for (unsigned int e = 0; e < 8; e++) {
                            centroid[e] = book[(word >> (4u * e)) & 15u];
                        }
                    } else {
#pragma unroll
                        for (unsigned int e = 0; e < 8; e++) {
                            centroid[e] = book[tq_code(row_codes, bits, e8 + e)];
                        }
                    }
                    tq_dot8(code_part, q_rot, group, k_width, kg * dim + e8, centroid);
                }
                float norm = norms[cell];
#pragma unroll
                for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
                    dot[g] = fmaf(norm, code_part[g], dot[g]);
                }
            }
        }
#pragma unroll
        for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
            dot[g] += __shfl_xor_sync(0xffffffffu, dot[g], 1);
            dot[g] += __shfl_xor_sync(0xffffffffu, dot[g], 2);
        }
        if (j < count && part == 0) {
            // An exact key's score has no offsets term.
            bool exact_key = j < exact_count;
#pragma unroll
            for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
                if (g < group) {
                    scores[j][g] = exact_key ? dot[g] * scale : (q_offsets[g] + dot[g]) * scale;
                }
            }
        }
    }
    __syncthreads();

    // Softmax: one warp per query head, the weights normalized; the exact
    // keys' weights summed.
    for (unsigned int g = warp; g < group; g += warps) {
        float local_max = neg_inf();
        for (unsigned int j = lane; j < count; j += 32) {
            local_max = fmaxf(local_max, scores[j][g]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, offset));
        }
        float local_sum = 0.0f;
        for (unsigned int j = lane; j < count; j += 32) {
            float weight = expf(scores[j][g] - local_max);
            scores[j][g] = weight;
            local_sum += weight;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, offset);
        }
        float exact_part = 0.0f;
        for (unsigned int j = lane; j < count; j += 32) {
            float weight = scores[j][g] / local_sum;
            scores[j][g] = weight;
            if (j < exact_count) {
                exact_part += weight;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            exact_part += __shfl_xor_sync(0xffffffffu, exact_part, offset);
        }
        if (lane == 0) {
            row_max[g] = local_max;
            row_sum[g] = local_sum;
            exact_sum[g] = exact_part;
        }
    }
    __syncthreads();

    // Values: thread (i, h) sums coordinate i of the heads from h *
    // value_heads on, each code's value (its group's norm times the
    // centroid) feeding every head and each float4 of weights four heads.
    unsigned int value_heads = v_width * 2u <= blockDim.x ? 4u : TQ_MAX_GROUP;
    unsigned int head_slices = TQ_MAX_GROUP / value_heads;
    float* exact_rotated = q_rot;
    for (unsigned int e = tid; e < v_width * head_slices; e += blockDim.x) {
        unsigned int i = e % v_width;
        unsigned int h0 = (e / v_width) * value_heads;
        if (h0 >= group) {
            continue;
        }
        unsigned int remaining = (group - h0 + 3u) & ~3u;
        unsigned int heads = remaining < value_heads ? remaining : value_heads;
        unsigned int vg = i / dim;
        unsigned int di = i % dim;
        float acc[TQ_MAX_GROUP];
#pragma unroll
        for (unsigned int a = 0; a < TQ_MAX_GROUP; a++) {
            acc[a] = 0.0f;
        }
#pragma unroll 4
        for (unsigned int j = exact_count; j < count; j++) {
            unsigned long long cell = (first + j) * k_groups + vg;
            const unsigned char* row_codes = codes + cell * code_bytes;
            unsigned int code = bits == 4u
                ? (row_codes[di >> 1] >> ((di & 1u) << 2)) & 15u
                : tq_code(row_codes, bits, di);
            tq_weigh8(acc, scores[j], h0, heads, norms[cell] * book[code]);
        }
#pragma unroll
        for (unsigned int a = 0; a < TQ_MAX_GROUP; a++) {
            unsigned int g = h0 + a;
            if (a < value_heads && g < group) {
                partial[((head0 + g) * chunks + chunk) * (v_width + 2) + i] = acc[a];
            }
        }
        if (exact_count > 0) {
#pragma unroll
            for (unsigned int a = 0; a < TQ_MAX_GROUP; a++) {
                acc[a] = 0.0f;
            }
            for (unsigned int j = 0; j < exact_count; j++) {
                tq_weigh8(acc, scores[j], h0, heads, exact[(first + j) * k_width + i]);
            }
            float offset = offsets[i];
            float sign = signs[di];
#pragma unroll
            for (unsigned int a = 0; a < TQ_MAX_GROUP; a++) {
                unsigned int g = h0 + a;
                if (a < value_heads && g < group) {
                    exact_rotated[g * v_width + i] = (acc[a] - exact_sum[g] * offset) * sign;
                }
            }
        }
    }
    if (exact_count > 0) {
        __syncthreads();
        // The exact sums' transform, every group of every head at once,
        // then the scale, added to the partials.
        unsigned int value_transforms = group * (unsigned int)v_groups;
        for (unsigned int half = 1; half < dim; half <<= 1) {
            for (unsigned int t = tid; t < value_transforms * (dim / 2); t += blockDim.x) {
                unsigned int base = (t / (dim / 2)) * dim;
                unsigned int u = t % (dim / 2);
                unsigned int i = base + (u / half) * 2 * half + (u % half);
                float a = exact_rotated[i];
                float b = exact_rotated[i + half];
                exact_rotated[i] = a + b;
                exact_rotated[i + half] = a - b;
            }
            __syncthreads();
        }
        float v_scale = 1.0f / sqrtf((float)dim);
        for (unsigned int i = tid; i < v_width; i += blockDim.x) {
            for (unsigned int g = 0; g < group; g++) {
                partial[((head0 + g) * chunks + chunk) * (v_width + 2) + i] +=
                    exact_rotated[g * v_width + i] * v_scale;
            }
        }
    }
    if (tid < group) {
        float* out = partial + ((head0 + tid) * chunks + chunk) * (v_width + 2);
        out[v_width] = row_max[tid];
        out[v_width + 1] = row_sum[tid];
    }
}

// `out[h][g] <- Pi^T out[h][g] + offsets[g]` for every `d`-wide group `g`
// of every head's `groups * d` values: the latent decode's rotated value
// sums back to values (the weights sum to one). Block (h * groups + g).
extern "C" __global__ void tq_latent_rotate_back_kernel(
    float* out,
    const float* signs,
    const float* offsets,
    unsigned long long d,
    unsigned long long groups
) {
    __shared__ float values[TQ_MAX_DIM];
    unsigned int dim = (unsigned int)d;
    float* row = out + blockIdx.x * d;
    const float* offset = offsets + (blockIdx.x % groups) * d;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        values[i] = row[i];
    }
    __syncthreads();
    tq_hadamard(values, dim);
    float scale = 1.0f / sqrtf((float)dim);
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        row[i] = values[i] * (signs[i] * scale) + offset[i];
    }
}

// `attention`'s latent K and V (`chunk-latent-prefills`): the first `rows`
// rows of `latent` (rows of `width` values) split into their leading
// `latent_width` values `c` and the rest, the rope key every head shares.
extern "C" __global__ void latent_split_kernel(
    const float* latent,
    float* c,
    float* rope,
    unsigned long long rows,
    unsigned long long width,
    unsigned long long latent_width
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * width) {
        return;
    }
    unsigned long long row = index / width;
    unsigned long long col = index % width;
    float value = latent[index];
    if (col < latent_width) {
        c[row * latent_width + col] = value;
    } else {
        rope[row * (width - latent_width) + (col - latent_width)] = value;
    }
}

// One query row against an f32 latent history read as both K and V
// (`cache-latent-attention-history`): rows of `k_width` values, each query
// head `k_width` wide, the values each row's leading `v_width`. Block
// (chunk, head block) serves up to TQ_MAX_GROUP query heads over a chunk
// of ATTENTION_CHUNK_KEYS keys, so each row is read once for them all
// (`speed-up-decode-attention-on-codes`): one thread per key scores it
// against every head from float4s of the row and of the queries (in
// dynamic shared memory), one warp per head takes the chunk's softmax, and
// thread i sums value coordinate i of every head from float4s of
// key-major weights. Partials as `attention_split_kernel`'s
// (`v_width + 2` floats per head and chunk), for `attention_combine_kernel`.
// `k_width` is a multiple of 4.
extern "C" __global__ void latent_attention_split_kernel(
    const float* q,
    const float* kv,
    float* partial,
    unsigned long long kv_seq_len,
    unsigned long long head_count,
    unsigned long long k_width,
    unsigned long long v_width,
    int causal,
    long long window_size,
    unsigned long long query_position_offset,
    unsigned long long chunks,
    const unsigned long long* step_offset,
    float score_scale
) {
    extern __shared__ __align__(16) float latent_queries[];
    __shared__ __align__(16) float scores[ATTENTION_CHUNK_KEYS][TQ_MAX_GROUP];
    __shared__ float row_max[TQ_MAX_GROUP];
    __shared__ float row_sum[TQ_MAX_GROUP];
    if (step_offset) {
        kv_seq_len += *step_offset;
        query_position_offset += *step_offset;
    }
    unsigned long long chunk = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 31;
    unsigned int warp = tid >> 5;
    unsigned int warps = blockDim.x >> 5;
    unsigned int width = (unsigned int)k_width;
    unsigned int values = (unsigned int)v_width;
    unsigned long long head0 = (unsigned long long)blockIdx.y * TQ_MAX_GROUP;
    unsigned int group = (unsigned int)(head_count - head0 < TQ_MAX_GROUP
        ? head_count - head0 : TQ_MAX_GROUP);
    float scale = score_scale > 0.0f ? score_scale : 1.0f / sqrtf((float)k_width);

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
    if (first >= last) {
        for (unsigned int g = 0; g < group; g++) {
            float* out = partial + ((head0 + g) * chunks + chunk) * (v_width + 2);
            for (unsigned int i = tid; i < values; i += blockDim.x) {
                out[i] = 0.0f;
            }
            if (tid == 0) {
                out[v_width] = neg_inf();
                out[v_width + 1] = 0.0f;
            }
        }
        return;
    }
    unsigned int count = (unsigned int)(last - first);

    for (unsigned int i = tid; i < group * width; i += blockDim.x) {
        latent_queries[i] = q[head0 * k_width + i];
    }
    __syncthreads();

    // Scores: four threads per key (LATENT_THREADS = 4 *
    // ATTENTION_CHUNK_KEYS), thread `part` of a key taking its float4s part,
    // part + 4, ..., against every query head; the four sums are added by
    // a fixed butterfly.
    {
        unsigned int j = tid >> 2;
        unsigned int part = tid & 3u;
        float dot[TQ_MAX_GROUP];
#pragma unroll
        for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
            dot[g] = 0.0f;
        }
        if (j < count) {
            const float4* row = (const float4*)(kv + (first + j) * k_width);
            for (unsigned int c = part; c < width / 4; c += 4) {
                float4 key = row[c];
#pragma unroll
                for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
                    if (g < group) {
                        float4 query = ((const float4*)(latent_queries + g * width))[c];
                        dot[g] = fmaf(query.x, key.x, dot[g]);
                        dot[g] = fmaf(query.y, key.y, dot[g]);
                        dot[g] = fmaf(query.z, key.z, dot[g]);
                        dot[g] = fmaf(query.w, key.w, dot[g]);
                    }
                }
            }
        }
#pragma unroll
        for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
            dot[g] += __shfl_xor_sync(0xffffffffu, dot[g], 1);
            dot[g] += __shfl_xor_sync(0xffffffffu, dot[g], 2);
        }
        if (j < count && part == 0) {
#pragma unroll
            for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
                if (g < group) {
                    scores[j][g] = dot[g] * scale;
                }
            }
        }
    }
    __syncthreads();

    // Softmax: one warp per query head, the weights normalized.
    for (unsigned int g = warp; g < group; g += warps) {
        float local_max = neg_inf();
        for (unsigned int j = lane; j < count; j += 32) {
            local_max = fmaxf(local_max, scores[j][g]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, offset));
        }
        float local_sum = 0.0f;
        for (unsigned int j = lane; j < count; j += 32) {
            float weight = expf(scores[j][g] - local_max);
            scores[j][g] = weight;
            local_sum += weight;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, offset);
        }
        for (unsigned int j = lane; j < count; j += 32) {
            scores[j][g] = scores[j][g] / local_sum;
        }
        if (lane == 0) {
            row_max[g] = local_max;
            row_sum[g] = local_sum;
        }
    }
    __syncthreads();

    // Values: thread i sums coordinate i of every query head, in key order.
    unsigned int heads = (group + 3u) & ~3u;
    for (unsigned int i = tid; i < values; i += blockDim.x) {
        float acc[TQ_MAX_GROUP];
#pragma unroll
        for (unsigned int a = 0; a < TQ_MAX_GROUP; a++) {
            acc[a] = 0.0f;
        }
#pragma unroll 4
        for (unsigned int j = 0; j < count; j++) {
            tq_weigh8(acc, scores[j], 0u, heads, kv[(first + j) * k_width + i]);
        }
#pragma unroll
        for (unsigned int g = 0; g < TQ_MAX_GROUP; g++) {
            if (g < group) {
                partial[((head0 + g) * chunks + chunk) * (v_width + 2) + i] = acc[g];
            }
        }
    }
    if (tid < group) {
        float* out = partial + ((head0 + tid) * chunks + chunk) * (v_width + 2);
        out[v_width] = row_max[tid];
        out[v_width + 1] = row_sum[tid];
    }
}
