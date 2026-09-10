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
    int* invalid_id_flag
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
    const float* src = table + id * dim;
    float* dst = out + row * dim;
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

// One thread per row. `weight_row_stride` is 0 for a single broadcast weight
// row, or `cols` for a per-row weight -- mirrors providers/cpu::rmsnorm's
// `row_weight_stride` exactly.
extern "C" __global__ void rmsnorm_kernel(
    const float* input,
    const float* weight,
    float* out,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long weight_row_stride,
    float epsilon
) {
    unsigned long long row = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float* in_row = input + row * cols;
    const float* w_row = weight + row * weight_row_stride;
    float mean_square = 0.0f;
    for (unsigned long long i = 0; i < cols; i++) {
        mean_square += in_row[i] * in_row[i];
    }
    mean_square /= (float)cols;
    float scale = 1.0f / sqrtf(mean_square + epsilon);
    float* out_row = out + row * cols;
    for (unsigned long long i = 0; i < cols; i++) {
        out_row[i] = in_row[i] * scale * w_row[i];
    }
}

// One thread per (row, head, pair). `half = dimension / 2`, `head_width =
// cols / head_count`. `out` is pre-seeded (device-to-device copy of
// `input`, done Rust-side) so columns this kernel never touches -- a
// partial-RoPE block's untouched tail -- keep their original values.
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
    unsigned long long head_count
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
    float position = (float)(position_offset + row) * scale;
    float frequency = powf(base, -2.0f * (float)pair / (float)dimension);
    float angle = position * frequency;
    float s = sinf(angle);
    float c = cosf(angle);
    unsigned long long col_base = row * cols + head * head_width;
    float even = input[col_base + 2 * pair];
    float odd = input[col_base + 2 * pair + 1];
    out[col_base + 2 * pair] = even * c - odd * s;
    out[col_base + 2 * pair + 1] = even * s + odd * c;
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
    unsigned long long b_col_stride
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
        float b_value = b[inner * b_inner_stride + col * b_col_stride];
        acc += a_value * b_value;
    }
    out[row * n + col] = acc;
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
