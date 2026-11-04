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
    const unsigned short* table16
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
    const unsigned long long* step_offset
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
// `rmsnorm_kernel`'s normalization of one row, computed by a whole block
// (at least RMSNORM_THREADS threads) into `out` (shared memory): the same
// RMSNORM_THREADS-strided sums, the same tree, the same scale and the same
// `x * scale * w` products, so a matmul reading `out` reads exactly the
// values `rmsnorm_kernel` would have written. Every thread of the block
// must call it.
__device__ void rmsnorm_row_into(
    const float* in_row,
    const float* w,
    unsigned long long cols,
    float epsilon,
    float* out,
    float* partial
) {
    unsigned int t = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int threads = blockDim.x * blockDim.y;
    if (t < RMSNORM_THREADS) {
        float sum = 0.0f;
        for (unsigned long long i = t; i < cols; i += RMSNORM_THREADS) {
            sum += in_row[i] * in_row[i];
        }
        partial[t] = sum;
    }
    __syncthreads();
    for (unsigned int stride = RMSNORM_THREADS / 2; stride > 0; stride /= 2) {
        if (t < stride) {
            partial[t] += partial[t + stride];
        }
        __syncthreads();
    }
    float scale = 1.0f / sqrtf(partial[0] / (float)cols + epsilon);
    for (unsigned long long i = t; i < cols; i += threads) {
        out[i] = in_row[i] * scale * w[i];
    }
    __syncthreads();
}

// With `norm_weight` set, `a` is first normalized row by row as the
// Runtime's `rmsnorm-` fused Operators define (`rmsnorm_row_into`, into
// `k` floats of dynamic shared memory); `a` must then be row-contiguous.
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
    const unsigned short* b16,
    const float* norm_weight,
    float norm_epsilon
) {
    extern __shared__ float normalized_row[];
    __shared__ float partial[MATMUL_SPLIT_K][32];
    __shared__ float norm_partial[RMSNORM_THREADS];
    unsigned long long row = blockIdx.y;
    unsigned long long col = (unsigned long long)blockIdx.x * 32 + threadIdx.x;
    unsigned int slice = threadIdx.y;
    const float* a_row = a + row * a_row_stride;
    if (norm_weight) {
        rmsnorm_row_into(a_row, norm_weight, k, norm_epsilon, normalized_row, norm_partial);
        a_row = normalized_row;
    }
    float acc = 0.0f;
    if (row < m && col < n) {
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
    unsigned long long b_inner_stride,
    const float* norm_weight,
    float norm_epsilon
) {
    extern __shared__ float normalized_row[];
    __shared__ float gate_partial[MATMUL_SPLIT_K][32];
    __shared__ float up_partial[MATMUL_SPLIT_K][32];
    __shared__ float norm_partial[RMSNORM_THREADS];
    unsigned long long row = blockIdx.y;
    unsigned long long col = (unsigned long long)blockIdx.x * 32 + threadIdx.x;
    unsigned int slice = threadIdx.y;
    const float* a_row = a + row * a_row_stride;
    if (norm_weight) {
        rmsnorm_row_into(a_row, norm_weight, k, norm_epsilon, normalized_row, norm_partial);
        a_row = normalized_row;
    }
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    if (row < m && col < n) {
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
