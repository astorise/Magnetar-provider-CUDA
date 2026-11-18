//! GPU launch wrappers for the `operator-scope` required-now kernel set.
//! Mirrors `providers/cpu`'s free-function signatures and validation
//! (`matmul`, `embedding_lookup`, `rmsnorm`, `rope`, `attention`,
//! `softmax_rows`, `silu`, `add`, `mul`, `residual_add`) so the two are
//! directly comparable in conformance tests, but every method here can fail
//! (allocation, launch, driver errors) in ways a pure host loop cannot --
//! see `CudaError`.
//!
//! # Explicit data movement, device-resident chaining
//!
//! Every method here takes and returns [`CudaDeviceBuffer`], not
//! [`HostTensor`]: inputs already on the device stay on the device, and a
//! kernel's output is left in a freshly allocated device buffer rather than
//! downloaded before returning (`enable-device-resident-kernel-chaining`).
//! [`CudaKernels::upload`]/[`CudaKernels::download`] are the only two points
//! that cross the host/device boundary, and only [`crate::CudaExecutor`]
//! (`executor.rs`) calls them -- when a resource is already device-resident
//! under its existing `TensorResourceId`, the executor reuses the stored
//! [`CudaDeviceBuffer`] directly instead of downloading and re-uploading it.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaModule, CudaSlice, CudaStream, DevicePtr,
    DevicePtrMut, LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::compile_ptx;
use magnetar_core::operator::{RopeBandedFrequencyScaling, RopeYarnFrequencyScaling};
use magnetar_provider_api::HostTensor;
use magnetar_provider_api::{KvStorePrecision, turboquant_codebook, turboquant_signs};
use magnetar_provider_api::{packed_weight::PackedWeight, packed_weight::PackedWeightFormat};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{CudaError, CudaErrorCode};
use crate::half_precision;

const KERNEL_SOURCE: &str = include_str!("kernels.cu");
/// Tensor Core matmuls (`mma.sync` bfloat16), built for compute capability
/// 8.0 and later only.
const TENSOR_KERNEL_SOURCE: &str = include_str!("tensor_kernels.cu");
/// `TC_BLOCK_M` / `TC_BLOCK_N` / `TC_THREADS` in `tensor_kernels.cu`.
const TC_BLOCK_M: u64 = 64;
const TC_BLOCK_N: u64 = 64;
const TC_THREADS: u32 = 128;
/// `TCP_BLOCK_M` / `TCP_BLOCK_N` / `TCP_THREADS` in `tensor_kernels.cu`:
/// the pipelined Tensor Core matmuls, which need `k` and `n` to be
/// multiples of `TCP_ALIGN` (16-byte rows of bfloat16).
const TCP_BLOCK_M: u64 = 128;
const TCP_BLOCK_N: u64 = 64;
const TCP_THREADS: u32 = 256;
const TCP_ALIGN: u64 = 8;
/// Dynamic shared memory of `matmul_tcp_split3_kernel`: two pipeline stages
/// of three `a` tiles (`TCP_BLOCK_M` rows of 16 + 8 halves) and three `b`
/// tiles (16 rows of `TCP_BLOCK_N` + 8 halves).
const TCP3_SHARED_BYTES: u32 = 2 * (3 * 128 * 24 + 3 * 16 * 72) * 2;
/// `TCA_QUERIES` in `tensor_kernels.cu`: queries per Tensor Core attention
/// block.
const TCA_QUERIES: u64 = 64;
/// `TCAP_KEYS` in `tensor_kernels.cu`: keys per Tensor Core attention stage.
const TCAP_KEYS: u64 = 32;
/// Set to `0` to keep long prefill matmuls off the Tensor Cores (the
/// bit-identical CUDA-core tile Kernels then serve them).
pub const TENSOR_CORES_ENV: &str = "MAGNETAR_CUDA_TENSOR_CORES";
/// Set to `0` to decline packed weights (`execute-direct-quantized-weights`):
/// Model Loading then hands this Provider their dequantized f32 values, the
/// disable control for direct quantized execution.
pub const PACKED_WEIGHTS_ENV: &str = "MAGNETAR_CUDA_PACKED_WEIGHTS";
/// Set to `0` to keep f32 activations in the decode-step packed matmuls.
/// By default they read their activations quantized to 8 bits per
/// 32-value block and multiply with `__dp4a` (`q8_matmul_rows_int8_kernel`
/// and its `Q4_K`/`Q6_K` forms): this rounds the activations, so outputs
/// can differ slightly from the f32-activation Kernels, which this
/// disable control restores.
pub const INT8_ACTIVATIONS_ENV: &str = "MAGNETAR_CUDA_INT8_ACTIVATIONS";
/// Activation values the int8 scratch holds: `Q8_MATMUL_MAX_ROWS` rows of
/// up to 32768 columns. A larger matmul keeps f32 activations.
const INT8_SCRATCH_VALUES: u64 = Q8_MATMUL_MAX_ROWS * 32_768;

/// What a fused matmul adds to its product.
#[derive(Clone, Copy)]
enum MatmulAddend<'a> {
    /// A `[n]` (or `[1, n]`) bias row added to every output row.
    Row(&'a CudaDeviceBuffer),
    /// A full `[m, n]` residual.
    Full(&'a CudaDeviceBuffer),
}

/// Warps per block of `matmul_split_k_kernel`; must equal `MATMUL_SPLIT_K`
/// in `kernels.cu`.
const MATMUL_SPLIT_K: u32 = 16;

/// Rows a split-K launch covers in its grid's y dimension (CUDA's limit).
const MATMUL_SPLIT_K_MAX_ROWS: u64 = 65_535;

/// Rows of `a` one block of the multi-row split-K Kernels serves; must
/// equal `MATMUL_ROWS` in `kernels.cu`.
const MATMUL_ROWS: u64 = 8;

/// Rows from which a multi-row matmul uses the tile Kernels: enough row
/// groups to occupy the Device with 128-column blocks.
const MATMUL_TILE_MIN_ROWS: u64 = 64;

/// Columns of one tile-Kernel block: `32 * MATMUL_TILE_COLUMNS` in
/// `kernels.cu`.
const MATMUL_TILE_BLOCK_COLUMNS: u64 = 128;

/// The split-K function and grid for `m` rows of `n` columns: one row per
/// block for a decode step, MATMUL_ROWS per block otherwise (each weight
/// is then read once per row group), and MATMUL_ROWS by 128 columns per
/// block for a long prefill (each weight load serves 8 rows and each `a`
/// value 4 columns). All are bit-identical row by row.
fn split_k_rows(
    m: u64,
    n: u64,
    [single, multi, tile]: [&'static str; 3],
) -> (&'static str, (u32, u32, u32)) {
    if m >= MATMUL_TILE_MIN_ROWS {
        (
            tile,
            (
                n.div_ceil(MATMUL_TILE_BLOCK_COLUMNS) as u32,
                m.div_ceil(MATMUL_ROWS) as u32,
                1,
            ),
        )
    } else if m > 1 {
        (
            multi,
            (n.div_ceil(32) as u32, m.div_ceil(MATMUL_ROWS) as u32, 1),
        )
    } else {
        (single, (n.div_ceil(32) as u32, m as u32, 1))
    }
}

/// Threads per row of `rmsnorm_kernel`; must equal `RMSNORM_THREADS` in
/// `kernels.cu`.
const RMSNORM_THREADS: u32 = 256;

/// Threads per (head, query) of `attention_block_kernel`; must equal
/// `ATTENTION_THREADS` in `kernels.cu`.
const ATTENTION_THREADS: u32 = 128;
/// `ATTENTION_CHUNK_KEYS` in `kernels.cu`: keys per split-attention block.
const ATTENTION_CHUNK_KEYS: u64 = 128;
/// `ATTENTION_SPLIT_THREADS` in `kernels.cu`.
const ATTENTION_SPLIT_THREADS: u32 = 128;
/// `ATTENTION_GQA_MAX_GROUP` / `ATTENTION_GQA_THREADS` in `kernels.cu`:
/// the grouped-query split decode attention, which serves 64- and
/// 128-dimension heads with at most 8 query heads per KV head.
const ATTENTION_GQA_MAX_GROUP: u64 = 8;
const ATTENTION_GQA_THREADS: u32 = 128;

/// The grouped split decode attention for a head dimension: its kernel and
/// keys per chunk (`attention_split_gqa_body`'s template arguments).
fn grouped_decode_kernel(head_dimension: u64) -> Option<(&'static str, u64)> {
    match head_dimension {
        64 => Some(("attention_split_gqa_kernel", 64)),
        128 => Some(("attention_split_gqa128_kernel", 32)),
        _ => None,
    }
}

/// The Tensor Core prefill attention for a head dimension: its kernel and
/// dynamic shared memory, four stages (K hi/lo, V hi/lo) of 2 x 32 keys of
/// `head_dimension + 8` halves (`attention_tcp_body`).
fn tensor_core_prefill_attention(head_dimension: u64) -> Option<(&'static str, u32)> {
    let name = match head_dimension {
        64 => "attention_tcp_kernel",
        128 => "attention_tcp128_kernel",
        _ => return None,
    };
    let bytes = 4 * 2 * TCAP_KEYS * (head_dimension + 8) * 2;
    Some((name, u32::try_from(bytes).ok()?))
}
/// Dynamic shared memory `attention_combine_kernel` may take for its
/// per-chunk weights: 48 KB less its 4 KB static reduction scratch.
const ATTENTION_COMBINE_MAX_DYNAMIC_SHARED: u32 = 44 * 1024;
/// `ATTENTION_TILE_QUERIES` in `kernels.cu`: queries per tiled-attention
/// block, 4 threads each.
const ATTENTION_TILE_QUERIES: u64 = 64;

/// Dynamic shared memory `attention_block_kernel` may use without opting a
/// function into more (the CUDA default limit per block).
const ATTENTION_MAX_SHARED_BYTES: u64 = 48 * 1024;

/// Which half-precision format a [`CudaHalfBuffer`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaHalfDType {
    Float16,
    BrainFloat16,
}

/// A tensor resident in this Provider's device memory as real 2-byte
/// half-precision bit patterns (`add-native-cuda-half-precision-compute`
/// Phase 2) -- genuine on-device storage savings and genuine half-precision
/// rounding behavior for [`CudaKernels::add_half`]/[`CudaKernels::mul_half`],
/// not `f32` promoted and merely labeled. Deliberately a distinct type from
/// [`CudaDeviceBuffer`] rather than a variant folded into it: every existing
/// method on [`CudaDeviceBuffer`] (`matmul`, `rmsnorm`, `rope`,
/// `attention`, ...) is untouched by this addition, and only the two
/// methods that genuinely have a half-precision implementation gain one.
/// Not yet reachable through [`crate::advertisements`] or the generic
/// Kernel dispatch contract -- wiring an actual compute-dtype selection
/// into the Runtime's planner is Phase 3, a separate, not-yet-designed
/// chantier (see `add-native-cuda-half-precision-compute`'s `design.md`).
pub struct CudaHalfBuffer {
    slice: CudaSlice<u16>,
    pub shape: Vec<u64>,
    pub dtype: CudaHalfDType,
}

impl CudaHalfBuffer {
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }
}

fn half_kernel_name(op: &str, dtype: CudaHalfDType) -> &'static str {
    match (op, dtype) {
        ("add", CudaHalfDType::Float16) => "add_f16_kernel",
        ("add", CudaHalfDType::BrainFloat16) => "add_bf16_kernel",
        ("mul", CudaHalfDType::Float16) => "mul_f16_kernel",
        ("mul", CudaHalfDType::BrainFloat16) => "mul_bf16_kernel",
        _ => unreachable!("half_kernel_name called with an unknown operator {op:?}"),
    }
}

fn same_shape_half(a: &CudaHalfBuffer, b: &CudaHalfBuffer) -> Result<(), CudaError> {
    if a.shape != b.shape {
        return Err(CudaError::new(
            CudaErrorCode::ShapeUnsupported,
            format!("shape mismatch: {:?} vs {:?}", a.shape, b.shape),
        ));
    }
    if a.dtype != b.dtype {
        return Err(CudaError::new(
            CudaErrorCode::ShapeUnsupported,
            format!("dtype mismatch: {:?} vs {:?}", a.dtype, b.dtype),
        ));
    }
    Ok(())
}

/// A tensor resident entirely in this Provider's device memory, persisting
/// across separate Kernel invocations (design.md's "device allocation table
/// keyed by `TensorResourceId`" decision) instead of only for the duration
/// of one upload-compute-download call.
pub struct CudaDeviceBuffer {
    /// The values, unless the buffer is compact (`bf16`), in which case
    /// this is a one-element placeholder no Kernel reads.
    pub(crate) slice: CudaSlice<f32>,
    pub shape: Vec<u64>,
    /// A compact weight (`execute-native-half-precision`): every value is
    /// exactly representable in bfloat16 and is stored as its bfloat16
    /// bits, half the bytes of f32. Only Kernels that read bfloat16
    /// operands accept a compact buffer: the matmul family's `b` and the
    /// embedding lookup's table. They convert each value back to the same
    /// f32 exactly, so their results are unchanged.
    pub(crate) bf16: Option<CudaSlice<u16>>,
    /// A packed `Q8_0` weight (`execute-direct-quantized-weights`), stored
    /// as the checkpoint stored it: no Kernel reads `slice` then either.
    pub(crate) q8: Option<CudaQ8Weight>,
    /// A TurboQuant KV store (`turboquant-kv-history`): no Kernel reads
    /// `slice`; `kv-write` and attention read the codes, every other read
    /// dequantizes them.
    pub(crate) tq: Option<CudaTqStore>,
}

/// A TurboQuant KV store's device form: per row and head, the packed
/// centroid indices and the f32 norm of `magnetar_provider_api::turboquant`,
/// with the signs and the codebook of its head width and precision.
pub struct CudaTqStore {
    /// `[capacity][heads][head_dimension * bits / 8]` bytes.
    pub(crate) codes: CudaSlice<u8>,
    /// `[capacity][heads]` norms.
    pub(crate) norms: CudaSlice<f32>,
    /// `[heads][head_dimension]`: the offsets the write at row 0 sets
    /// (`tq_offsets_kernel`), zeros before it.
    pub(crate) offsets: CudaSlice<f32>,
    pub(crate) precision: KvStorePrecision,
    pub(crate) bits: u32,
    pub(crate) head_dimension: u64,
    pub(crate) heads: u64,
    pub(crate) signs: CudaSlice<f32>,
    pub(crate) centroids: CudaSlice<f32>,
    pub(crate) boundaries: CudaSlice<f32>,
    /// The store's first rows, kept exact as an f32 `[exact_rows, width]`
    /// buffer (`keep-turboquant-kv-head-exact`); `None` without a head.
    pub(crate) exact: Option<Box<CudaDeviceBuffer>>,
    /// The rows `exact` holds: the requested head, at most the capacity.
    pub(crate) exact_rows: u64,
}

impl CudaTqStore {
    /// The exact head, when it holds the first `rows` rows.
    fn head_covering(&self, rows: u64) -> Option<&CudaDeviceBuffer> {
        self.exact.as_deref().filter(|_| rows <= self.exact_rows)
    }
}

/// A `Q8_0` matrix on the device: its signed 8-bit quants in stored
/// row-major order and one `f32` scale per 32-value block (the block's
/// `f16` scale, converted exactly). The logical matrix (the buffer's
/// `shape`) is the stored one, or its transpose when `transposed`.
pub struct CudaQ8Weight {
    /// `IQ4_NL`/`IQ4_XS`: see `upload_iq4_weight`; IQ2/IQ3: see
    /// `upload_iq_weight`.
    /// `Q8_0`: each block's 32 quants, contiguous. `Q4_K`: the super-blocks'
    /// own 144-byte encoding, as stored. `Q6_K`: each super-block's first
    /// 208 bytes (its quants and sub-block scales).
    pub(crate) quants: CudaSlice<i8>,
    /// `Q8_0`: each block's scale. `Q4_K`: unused (the scales live in the
    /// super-blocks). `Q6_K`: each super-block's `d`.
    pub(crate) scales: CudaSlice<f32>,
    /// Which block encoding `quants` holds.
    pub(crate) format: PackedWeightFormat,
    pub(crate) stored_rows: u64,
    pub(crate) stored_cols: u64,
    pub(crate) transposed: bool,
}

/// The ids of an embedding lookup in a packed table, as device addresses
/// (`dequantize_packed`): `count` f32 ids and the lookup's invalid-id flag.
/// Without them, the kernels read every stored row (`ids` and the flag 0,
/// `count` the stored rows).
struct PackedRows {
    ids: u64,
    count: u64,
    invalid_id_flag: u64,
}

impl CudaDeviceBuffer {
    /// Elements, compact or not.
    pub fn len(&self) -> usize {
        if self.tq.is_some() {
            return self.shape.iter().product::<u64>() as usize;
        }
        match (&self.bf16, &self.q8) {
            (Some(bf16), _) => bf16.len(),
            (None, Some(q8)) => (q8.stored_rows * q8.stored_cols) as usize,
            (None, None) => self.slice.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the values are stored compact: as bfloat16 bits or as a
    /// packed `Q8_0` weight. Only Kernels that read a compact operand
    /// accept one.
    pub fn is_compact(&self) -> bool {
        self.bf16.is_some() || self.q8.is_some() || self.tq.is_some()
    }

    /// Whether this is a TurboQuant KV store (`turboquant-kv-history`).
    pub fn is_tq(&self) -> bool {
        self.tq.is_some()
    }

    /// The precision of a TurboQuant KV store.
    pub fn kv_store_precision(&self) -> Option<KvStorePrecision> {
        self.tq.as_ref().map(|tq| tq.precision)
    }

    /// Whether the values are a packed `Q8_0` weight.
    pub fn is_q8(&self) -> bool {
        self.q8.is_some()
    }
}

/// Rows of a matmul against a packed `Q8_0` weight served by
/// `q8_matmul_rows_kernel` (decode steps and short prompts); a longer
/// matmul dequantizes the weight to f32 for the duration of the call.
const Q8_MATMUL_MAX_ROWS: u64 = 64;

/// The capture attribute whose replay offset is how many trailing padding
/// rows a replayed bucketed prefill skips in the packed row matmuls
/// (`replay-bucketed-prefill`): their outputs are never read.
pub const PADDING_ROWS_ATTRIBUTE: &str = "padding_rows";

/// Dequantized TurboQuant K and V scratch, by role (0 K, 1 V) and shape.
type TqScratch = std::collections::BTreeMap<(u8, Vec<u64>), CudaSlice<f32>>;

/// `TQ_MAX_GROUP` in `kernels.cu`: the most query heads one block of the
/// grouped TurboQuant decode attention serves.
const TQ_MAX_GROUP: u64 = 8;

/// `TQ_MAX_DIM` in `kernels.cu`: the widest TurboQuant head.
const TQ_MAX_DIMENSION: u64 = 512;

/// The widest K and V heads `tq_attention_split_staged_kernel` serves
/// (`speed-up-decode-attention-on-codes`): a value coordinate per thread.
const TQ_STAGED_MAX_DIMENSION: u64 = 128;

/// The dynamic shared memory the latent decode Kernels may take for their
/// query heads (`latent_attention_split_kernel`,
/// `tq_latent_attention_split_kernel`): within the 48 KB a block may take
/// without opting in, beside their static arrays.
const LATENT_QUERY_SHARED_BYTES: u64 = 40 * 1024;

/// The most blocks per KV head a TurboQuant decode step splits its history
/// over (`speed-up-decode-attention-on-codes`, decision 4).
const TQ_MAX_SPLITS: u64 = 32;

/// How decode attention over TurboQuant stores is scheduled
/// (`speed-up-decode-attention-on-codes`). [`TqDecodeSchedule::Unstaged`]
/// unless a conformance or timing test picks another: staging the codes
/// and splitting the history did not make the Kernel faster (run
/// 44299614540).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TqDecodeSchedule {
    /// The staged Kernel, its history split over a bounded number of
    /// blocks per KV head.
    Split = 0,
    /// The staged Kernel, one chunk per block.
    ChunkPerBlock = 1,
    /// The Kernels before staging: a chunk per block, codes read from
    /// global memory.
    Unstaged = 2,
}

/// Decision 4 of `speed-up-decode-attention-on-codes`: the blocks per KV
/// head for `target` blocks resident at once over `kv_heads` KV heads and
/// `chunks` chunks of keys. Enough blocks to fill the target, at least two
/// chunks per block and at most [`TQ_MAX_SPLITS`]; fewer when the last wave
/// of `target` blocks would be less than three quarters full.
pub(crate) fn tq_split_count(target: u64, kv_heads: u64, chunks: u64) -> u64 {
    let mut splits = target
        .div_ceil(kv_heads)
        .min((chunks / 2).max(1))
        .clamp(1, TQ_MAX_SPLITS);
    let blocks = splits * kv_heads;
    let waves = blocks.div_ceil(target);
    if waves > 1 && blocks * 4 < waves * target * 3 {
        splits = ((waves - 1) * target / kv_heads).max(1);
    }
    splits
}

/// Threads of a TurboQuant block over a `head_dimension`-wide head: a power
/// of two, at most `TQ_MAX_THREADS` (256) in `kernels.cu`.
fn tq_threads(head_dimension: u64) -> u32 {
    head_dimension.clamp(32, 256) as u32
}
/// Output columns per warp of `q4k_matmul_rows_kernel` (`Q4K_COLS`) and
/// `q6k_matmul_rows_kernel` (`Q6K_COLS`).
const Q4K_MATMUL_COLS: u64 = 2;
const Q6K_MATMUL_COLS: u64 = 2;
/// `LOWK_COLS` in `kernels.cu`.
const LOW_BIT_MATMUL_COLS: u64 = 2;

/// `IQ_COLS` in `kernels.cu`.
const IQ_MATMUL_COLS: u64 = 2;

/// `IQ_IQ2_XXS` .. `IQ_IQ3_S` in `kernels.cu`, for the IQ2/IQ3 formats.
fn iq_format(format: PackedWeightFormat) -> Option<i32> {
    match format {
        PackedWeightFormat::Iq2Xxs => Some(0),
        PackedWeightFormat::Iq2Xs => Some(1),
        PackedWeightFormat::Iq2S => Some(2),
        PackedWeightFormat::Iq3Xxs => Some(3),
        PackedWeightFormat::Iq3S => Some(4),
        _ => None,
    }
}

/// `IQ4_NL_FORMAT` / `IQ4_XS_FORMAT` / `Q4_0_FORMAT` in `kernels.cu`.
fn iq4_format(format: PackedWeightFormat) -> i32 {
    match format {
        PackedWeightFormat::Iq4Nl => 0,
        PackedWeightFormat::Q4_0 => 2,
        _ => 1,
    }
}

/// Where an `IQ4_XS` weight's per-run sub-block scales start in its
/// `quants`, after every run's 16 index bytes.
fn iq4_subscale_offset(weight: &CudaQ8Weight) -> usize {
    match weight.format {
        // No sub-block scales: the kernels never read them.
        PackedWeightFormat::Iq4Nl | PackedWeightFormat::Q4_0 => 0,
        _ => (weight.stored_rows * weight.stored_cols / 32 * 16) as usize,
    }
}

/// `LOWK_Q2K` / `LOWK_Q3K` in `kernels.cu`.
fn low_bit_format(format: PackedWeightFormat) -> i32 {
    match format {
        PackedWeightFormat::Q2K => 0,
        _ => 1,
    }
}
/// `Q8_MATMUL_ROWS` / `Q8_MATMUL_WARPS` in `kernels.cu`.
const Q8_MATMUL_ROWS: u64 = 8;
const Q8_MATMUL_WARPS: u32 = 4;

/// Weights at least this large are stored compact when every value is
/// exactly a bfloat16 value; smaller tensors (biases, norm weights,
/// activations) always stay f32.
pub const COMPACT_WEIGHT_MIN_ELEMENTS: usize = 1 << 16;

/// The bfloat16 bits of `value` when converting them back gives exactly
/// `value` (the low 16 bits of its f32 encoding are zero).
fn exact_bf16_bits(value: f32) -> Option<u16> {
    let bits = value.to_bits();
    (bits & 0xFFFF == 0).then_some((bits >> 16) as u16)
}

fn same_shape_buf(a: &CudaDeviceBuffer, b: &CudaDeviceBuffer) -> Result<(), CudaError> {
    if a.shape != b.shape {
        return Err(CudaError::new(
            CudaErrorCode::ShapeUnsupported,
            format!("shape mismatch: {:?} vs {:?}", a.shape, b.shape),
        ));
    }
    Ok(())
}

fn host_error(error: magnetar_provider_api::ReferenceCpuError) -> CudaError {
    CudaError::new(CudaErrorCode::ShapeUnsupported, error.to_string())
}

/// Compiled, loaded CUDA kernels for one [`CudaContext`]. Compilation
/// happens once, at construction (NVRTC, at first `CudaProvider` use on a
/// machine with a usable driver -- see design.md's "kernels are CUDA C++
/// source compiled to PTX via NVRTC at first Provider use" decision), not
/// ahead of time and not per call.
pub struct CudaKernels {
    stream: Arc<CudaStream>,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    /// The Tensor Core Kernels, when the Device has them (compute
    /// capability 8.0 or later), they compiled, and
    /// [`TENSOR_CORES_ENV`] does not turn them off.
    tensor_module: Option<Arc<CudaModule>>,
    /// Counts real host<->device crossings (`upload`/`download` calls
    /// only, never a Kernel launch itself) -- used by tests to prove two
    /// chained Kernel invocations do not round-trip through the host
    /// (`enable-device-resident-kernel-chaining` task 3.2), not for any
    /// production decision.
    upload_count: AtomicU64,
    download_count: AtomicU64,
    /// Kernel functions already looked up in `module`, by name: every
    /// launch needs one, and `CudaModule::load_function` allocates a C
    /// string and calls `cuModuleGetFunction` each time.
    functions: std::sync::Mutex<std::collections::HashMap<&'static str, CudaFunction>>,
    /// An output buffer the executor no longer needs (the previous value
    /// of the resource the current invocation is about to overwrite),
    /// taken by the next [`Self::alloc_output`] of the same length instead
    /// of a fresh device allocation. See [`Self::offer_output`].
    offered_output: std::sync::Mutex<Option<CudaSlice<f32>>>,
    /// The step capture in progress on this stream, if any
    /// (`capture-and-replay-cuda-graphs`).
    capture: std::sync::Mutex<Option<KernelCapture>>,
    /// The replay offset of the next capture, allocated ahead by
    /// [`Self::prepare_capture`].
    prepared_offset: std::sync::Mutex<Option<CudaSlice<u64>>>,
    /// Whether captures leave the thread unrestricted
    /// (`CU_STREAM_CAPTURE_MODE_RELAXED`): set when several GPUs' streams
    /// are captured from one thread at once, where a thread-local capture
    /// on one would forbid the others' setup, instantiation and launch.
    relaxed_capture: std::sync::atomic::AtomicBool,
    /// See [`CudaKernels::set_peer_access`].
    peer_access: std::sync::atomic::AtomicBool,
    /// The embedding lookup's invalid-id flag, allocated once.
    embedding_flag: std::sync::Mutex<Option<CudaSlice<i32>>>,
    /// The split-key attention's per-chunk partials: the current buffer,
    /// then every buffer it outgrew. A captured step may still read an
    /// outgrown one, so none is freed before this Provider.
    attention_scratch: std::sync::Mutex<AttentionScratch>,
    /// Dequantized TurboQuant K and V (`turboquant-kv-history`) for
    /// attention that does not read codes, by role (0 K, 1 V) and store
    /// shape: allocated outside capture, then reused.
    tq_scratch: std::sync::Mutex<TqScratch>,
    /// The bfloat16 hi and lo terms of a pipelined Tensor Core matmul's
    /// `a`: the current buffer, then every buffer it outgrew (kept until
    /// this Provider drops, like the attention scratch).
    split_scratch: std::sync::Mutex<SplitScratch>,
    /// `moe-experts`' routing and expert activations for a step of up to
    /// [`MOE_PERSISTENT_TOKENS`] tokens: the current buffers, then every
    /// set they outgrew (a captured step may still read one, so none is
    /// freed before this Provider).
    moe_scratch: std::sync::Mutex<(Option<MoeScratch>, Vec<MoeScratch>)>,
    /// Whether int8 activations are on (unless [`INT8_ACTIVATIONS_ENV`] is
    /// `0`).
    int8_activations: std::sync::atomic::AtomicBool,
    /// The quantized activations of the current packed matmul and their
    /// block scales, allocated once at [`INT8_SCRATCH_VALUES`] (never
    /// reallocated, so a captured step's addresses stay valid).
    int8_scratch: std::sync::Mutex<Option<Int8Scratch>>,
    /// The Device's multiprocessors.
    multiprocessors: u64,
    /// See [`TqDecodeSchedule`], as its discriminant.
    tq_decode_schedule: std::sync::atomic::AtomicU8,
    /// Blocks of the staged TurboQuant decode Kernel a multiprocessor
    /// holds, by dynamic shared memory bytes.
    tq_staged_occupancy: std::sync::Mutex<std::collections::BTreeMap<usize, u64>>,
    /// Whether a decode step over a latent TurboQuant store runs on its
    /// codes (`tq_latent_attention_split_kernel`) rather than through
    /// dequantized scratch; only the conformance and timing tests clear it.
    tq_latent_on_codes: std::sync::atomic::AtomicBool,
    /// Whether a `moe-experts` prefill over packed `Q8_0` experts groups its
    /// pairs by expert (`moe_q8_gated_grouped_kernel`); only the
    /// conformance and timing tests clear it.
    moe_grouped: std::sync::atomic::AtomicBool,
}

/// Int8 activations: [`INT8_SCRATCH_VALUES`] signed bytes and one f32 scale
/// per 32 of them.
struct Int8Scratch {
    values: CudaSlice<i8>,
    scales: CudaSlice<f32>,
}

/// A `moe-experts` weight's packed `Q8_0` form the expert Kernels read
/// (transposed storage, rows of `stored_cols` quants), if it has one.
fn packed_moe_q8(buffer: &CudaDeviceBuffer, stored_cols: u64) -> Option<&CudaQ8Weight> {
    buffer.q8.as_ref().filter(|q8| {
        q8.format == PackedWeightFormat::Q8_0 && q8.transposed && q8.stored_cols == stored_cols
    })
}

/// `moe-experts` scratch: the routed experts and weights of each
/// `(token, k)` pair, and each pair's expert activation.
struct MoeScratch {
    indices: CudaSlice<i32>,
    weights: CudaSlice<f32>,
    act: CudaSlice<f32>,
}

/// Steps of at most this many tokens (decode) route through the persistent
/// scratch, so they capture; longer ones (prefill) use temporaries.
const MOE_PERSISTENT_TOKENS: u64 = 64;

/// Rows of A one launch of `matmul_tcp_split3_kernel` serves: the split
/// buffer holds their three bfloat16 terms, never every row's, and the
/// gated form holds their products.
const SPLIT_TCP_BLOCK_ROWS: u64 = 2048;

/// `MOE_GROUP_ROWS` in `kernels.cu`: the pairs of one expert a block of the
/// grouped `moe-experts` Kernels serves.
const MOE_GROUP_ROWS: u64 = 8;

/// Tokens of a grouped `moe-experts` prefill served at once: its
/// activations and per-pair sums are this many tokens' at most.
const MOE_GROUP_CHUNK_TOKENS: u64 = 1024;

/// The current hi/lo split buffer and the ones it outgrew.
type SplitScratch = (Option<CudaSlice<u16>>, Vec<CudaSlice<u16>>);

/// A split's scratch lock and the device addresses of its hi and lo terms.
type SplitTerms<'a> = (std::sync::MutexGuard<'a, SplitScratch>, u64, u64);

/// A multi-input split's scratch lock and each input's (hi, lo) addresses.
type SplitTermsAll<'a> = (std::sync::MutexGuard<'a, SplitScratch>, Vec<(u64, u64)>);
/// [`SplitTermsAll`] with three terms per input (hi, mid, lo).
type SplitTerms3All<'a> = (std::sync::MutexGuard<'a, SplitScratch>, Vec<[u64; 3]>);

/// The current split-attention scratch buffer and the ones it outgrew.
type AttentionScratch = (Option<CudaSlice<f32>>, Vec<CudaSlice<f32>>);

/// A step capture in progress: what the captured launches need, and what
/// a replay must check.
struct KernelCapture {
    /// Integer attributes that advance by the replay offset.
    advancing: BTreeSet<String>,
    /// The replay offset, read by every captured launch whose attribute
    /// advances.
    offset: CudaSlice<u64>,
    /// The largest offset every captured launch stays within bounds for.
    max_offset: u64,
    /// Why the capture cannot be used, once something unsupported happened.
    failure: Option<String>,
    /// Whether the capture contains an embedding lookup, whose invalid-id
    /// flag the capture resets once, at its start, so the flag accumulates
    /// every captured lookup's failures.
    has_embedding: bool,
}

/// What a GPU that joined another GPU's step capture
/// ([`CudaKernels::join_capture`]) needs to replay its part: the captured
/// launches live in the other GPU's graph, but read this GPU's own replay
/// offset.
pub struct JoinedCapture {
    offset: CudaSlice<u64>,
    /// See [`CapturedStep::max_offset`].
    pub max_offset: u64,
    /// See [`CapturedStep::has_embedding`].
    pub has_embedding: bool,
}

/// A page-locked host buffer every GPU context can copy to and from
/// (`CU_MEMHOSTALLOC_PORTABLE`): the stage a tensor-parallel reduction
/// crosses between GPUs without peer access. Copies through it are
/// ordinary asynchronous copies, which a step capture records.
pub struct PinnedStage {
    ptr: *mut f32,
    len: usize,
}

// SAFETY: the buffer is only ever read and written by copies queued on
// GPU streams, ordered by events.
unsafe impl Send for PinnedStage {}
unsafe impl Sync for PinnedStage {}

impl PinnedStage {
    pub fn new(kernels: &CudaKernels, len: usize) -> Result<Self, CudaError> {
        kernels.stream.context().bind_to_thread()?;
        // SAFETY: a fresh allocation of `len` floats, freed in `drop`.
        let ptr = unsafe {
            cudarc::driver::result::malloc_host(
                len * std::mem::size_of::<f32>(),
                sys::CU_MEMHOSTALLOC_PORTABLE,
            )
        }? as *mut f32;
        Ok(Self { ptr, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[allow(clippy::mut_from_ref)]
    fn host(&self) -> &mut [f32] {
        // SAFETY: see the `Send`/`Sync` impls: only queued copies touch it.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for PinnedStage {
    fn drop(&mut self) {
        // SAFETY: allocated by `malloc_host` in `new`.
        let _ = unsafe { cudarc::driver::result::free_host(self.ptr.cast()) };
    }
}

/// A captured step's CUDA graph and what replaying it needs.
pub struct CapturedStep {
    graph: CapturedGraph,
    offset: CudaSlice<u64>,
    /// The largest replay offset every captured launch stays within bounds
    /// for.
    pub max_offset: u64,
    /// Whether the captured step contains an embedding lookup.
    pub has_embedding: bool,
}

/// A `CudaGraph`, which holds raw driver handles.
struct CapturedGraph(CudaGraph);

// SAFETY: a captured graph is only ever launched or dropped by the
// executor that owns it, under that executor's capture lock, so no two
// threads touch its handles at once (CUDA graphs require external
// serialization, which the lock provides).
unsafe impl Send for CapturedGraph {}

/// Drops an [`CudaKernels::offer_output`] buffer that the invocation did
/// not take.
pub struct OfferedOutput<'k> {
    kernels: &'k CudaKernels,
}

impl OfferedOutput<'_> {
    /// The offered buffer, if no Kernel took it.
    pub fn reclaim(self) -> Option<CudaSlice<f32>> {
        self.kernels.offered_output.lock().unwrap().take()
    }
}

impl Drop for OfferedOutput<'_> {
    fn drop(&mut self) {
        self.kernels.offered_output.lock().unwrap().take();
    }
}

impl CudaKernels {
    pub fn compile_and_load(context: &Arc<CudaContext>) -> Result<Self, CudaError> {
        // cudarc synchronizes buffers across streams through per-buffer
        // events. This Provider uses exactly one stream (created below):
        // every buffer is allocated, used and freed on it. Event tracking
        // would still create and destroy two events with every allocation
        // (one per Kernel output, hundreds per decode step) without the
        // ordering ever needing them.
        //
        // SAFETY: every buffer of this Provider lives on its one stream, so
        // no buffer is used across streams; single-stream ordering is all
        // the synchronization its work needs.
        unsafe { context.disable_event_tracking() };
        let ptx = compile_ptx(KERNEL_SOURCE)?;
        let module = context.load_module(ptx)?;
        let tensor_cores_allowed = std::env::var(TENSOR_CORES_ENV).as_deref() != Ok("0");
        let tensor_module = match context.compute_capability() {
            Ok((major, _)) if major >= 8 && tensor_cores_allowed => {
                cudarc::nvrtc::compile_ptx_with_opts(
                    TENSOR_KERNEL_SOURCE,
                    cudarc::nvrtc::CompileOptions {
                        arch: Some("compute_80"),
                        ..Default::default()
                    },
                )
                .ok()
                .and_then(|ptx| context.load_module(ptx).ok())
            }
            _ => None,
        };
        // A stream of its own: CUDA cannot capture the legacy default
        // stream (`capture-and-replay-cuda-graphs`). Every buffer and
        // launch of this Provider still uses this one stream.
        let stream = context.new_stream()?;
        Ok(Self {
            stream,
            module,
            tensor_module,
            upload_count: AtomicU64::new(0),
            download_count: AtomicU64::new(0),
            functions: std::sync::Mutex::new(std::collections::HashMap::new()),
            offered_output: std::sync::Mutex::new(None),
            capture: std::sync::Mutex::new(None),
            prepared_offset: std::sync::Mutex::new(None),
            relaxed_capture: std::sync::atomic::AtomicBool::new(false),
            peer_access: std::sync::atomic::AtomicBool::new(false),
            embedding_flag: std::sync::Mutex::new(None),
            attention_scratch: std::sync::Mutex::new((None, Vec::new())),
            tq_scratch: std::sync::Mutex::new(std::collections::BTreeMap::new()),
            split_scratch: std::sync::Mutex::new((None, Vec::new())),
            moe_scratch: std::sync::Mutex::new((None, Vec::new())),
            int8_activations: std::sync::atomic::AtomicBool::new(
                std::env::var(INT8_ACTIVATIONS_ENV).as_deref() != Ok("0"),
            ),
            int8_scratch: std::sync::Mutex::new(None),
            multiprocessors: context
                .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                .map_or(1, |count| count.max(1) as u64),
            tq_decode_schedule: std::sync::atomic::AtomicU8::new(TqDecodeSchedule::Unstaged as u8),
            tq_staged_occupancy: std::sync::Mutex::new(std::collections::BTreeMap::new()),
            tq_latent_on_codes: std::sync::atomic::AtomicBool::new(true),
            moe_grouped: std::sync::atomic::AtomicBool::new(true),
        })
    }

    /// See `moe_grouped` (conformance and timing tests).
    #[cfg(test)]
    pub(crate) fn set_moe_grouped(&self, grouped: bool) {
        self.moe_grouped
            .store(grouped, std::sync::atomic::Ordering::Relaxed);
    }

    /// See `tq_latent_on_codes` (conformance and timing tests).
    #[cfg(test)]
    pub(crate) fn set_tq_latent_on_codes(&self, on_codes: bool) {
        self.tq_latent_on_codes
            .store(on_codes, std::sync::atomic::Ordering::Relaxed);
    }

    /// Schedules decode attention over TurboQuant stores
    /// (conformance and timing tests).
    #[cfg(test)]
    pub(crate) fn set_tq_decode_schedule(&self, schedule: TqDecodeSchedule) {
        self.tq_decode_schedule
            .store(schedule as u8, std::sync::atomic::Ordering::Relaxed);
    }

    fn tq_decode_schedule(&self) -> TqDecodeSchedule {
        match self
            .tq_decode_schedule
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            0 => TqDecodeSchedule::Split,
            1 => TqDecodeSchedule::ChunkPerBlock,
            _ => TqDecodeSchedule::Unstaged,
        }
    }

    /// The blocks per KV head a staged TurboQuant decode step splits `keys`
    /// admitted keys over (`speed-up-decode-attention-on-codes`, decision
    /// 4): enough to give every multiprocessor as many blocks as it holds
    /// (at most four), at least two chunks per block, at most
    /// [`TQ_MAX_SPLITS`]; fewer when the last wave of blocks would be less
    /// than three quarters full.
    fn tq_decode_splits(
        &self,
        function: &CudaFunction,
        shared: usize,
        kv_heads: u64,
        keys: u64,
    ) -> u64 {
        let per_multiprocessor = *self
            .tq_staged_occupancy
            .lock()
            .unwrap()
            .entry(shared)
            .or_insert_with(|| {
                function
                    .occupancy_max_active_blocks_per_multiprocessor(
                        ATTENTION_SPLIT_THREADS,
                        shared,
                        None,
                    )
                    .map_or(1, |blocks| u64::from(blocks).clamp(1, 4))
            });
        tq_split_count(
            self.multiprocessors * per_multiprocessor,
            kv_heads.max(1),
            keys.div_ceil(ATTENTION_CHUNK_KEYS).max(1),
        )
    }

    /// See `relaxed_capture`.
    pub fn set_relaxed_capture(&self, relaxed: bool) {
        self.relaxed_capture
            .store(relaxed, std::sync::atomic::Ordering::Relaxed);
    }

    /// Allocates what the next [`Self::begin_capture`] needs, so that it
    /// allocates nothing: with several GPUs' streams captured from one
    /// thread, nothing may allocate once the first capture has begun.
    pub fn prepare_capture(&self) -> Result<(), CudaError> {
        let mut prepared = self.prepared_offset.lock().unwrap();
        if prepared.is_none() {
            *prepared = Some(self.stream.alloc_zeros::<u64>(1)?);
        }
        let mut flag_slot = self.embedding_flag.lock().unwrap();
        if flag_slot.is_none() {
            *flag_slot = Some(self.stream.alloc_zeros::<i32>(1)?);
        }
        drop(flag_slot);
        if self.int8_activations() {
            self.int8_scratch_slot()?;
        }
        Ok(())
    }

    /// Whether the decode-step packed matmuls read int8 activations
    /// ([`INT8_ACTIVATIONS_ENV`]).
    pub fn int8_activations(&self) -> bool {
        self.int8_activations
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Turns int8 activations on or off for later launches (the
    /// environment variable sets the initial value).
    pub fn set_int8_activations(&self, on: bool) {
        self.int8_activations
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// The int8 scratch, allocated on first use (never during a capture:
    /// then `Ok(None)` and the caller keeps f32 activations).
    fn int8_scratch_slot(
        &self,
    ) -> Result<Option<std::sync::MutexGuard<'_, Option<Int8Scratch>>>, CudaError> {
        let mut slot = self.int8_scratch.lock().unwrap();
        if slot.is_none() {
            if self.capture.lock().unwrap().is_some() {
                return Ok(None);
            }
            *slot = Some(Int8Scratch {
                values: self
                    .stream
                    .alloc_zeros::<i8>(INT8_SCRATCH_VALUES as usize)?,
                scales: self
                    .stream
                    .alloc_zeros::<f32>((INT8_SCRATCH_VALUES / 32) as usize)?,
            });
        }
        Ok(Some(slot))
    }

    /// Quantizes `a` (`m` contiguous rows of `k`) into the int8 scratch
    /// when int8 activations are on and it fits; `None` keeps the f32
    /// activations.
    fn quantize_activations(
        &self,
        a: &CudaDeviceBuffer,
        m: u64,
        k: u64,
    ) -> Result<Option<std::sync::MutexGuard<'_, Option<Int8Scratch>>>, CudaError> {
        if !self.int8_activations() || !k.is_multiple_of(32) || m * k > INT8_SCRATCH_VALUES {
            return Ok(None);
        }
        let Some(mut guard) = self.int8_scratch_slot()? else {
            return Ok(None);
        };
        let scratch = guard.as_mut().expect("allocated above");
        let blocks = m * k / 32;
        let func = self.function("quantize_activations_q8_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&a.slice)
            .arg(&mut scratch.values)
            .arg(&mut scratch.scales)
            .arg(&blocks);
        let config = LaunchConfig {
            grid_dim: (blocks.div_ceil(4) as u32, 1, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        Ok(Some(guard))
    }

    /// Starts capturing this stream's work into a CUDA graph. Every later
    /// launch whose attribute is named in `advancing` reads the replay
    /// offset from the Device.
    pub fn begin_capture(&self, advancing: BTreeSet<String>) -> Result<(), CudaError> {
        let mut capture = self.capture.lock().unwrap();
        if capture.is_some() {
            return Err(CudaError::new(
                CudaErrorCode::CaptureUnsupported,
                "a step capture is already in progress",
            ));
        }
        // Allocated before the capture starts: nothing allocates while
        // capturing.
        let prepared = self.prepared_offset.lock().unwrap().take();
        // Zeroed when allocated; each launch writes it anyway.
        let offset = match prepared {
            Some(offset) => offset,
            None => self.stream.alloc_zeros::<u64>(1)?,
        };
        let mut flag_slot = self.embedding_flag.lock().unwrap();
        let flag = match flag_slot.as_mut() {
            Some(flag) => flag,
            None => flag_slot.insert(self.stream.alloc_zeros::<i32>(1)?),
        };
        let mode = if self
            .relaxed_capture
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED
        } else {
            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        };
        self.stream.begin_capture(mode)?;
        // The captured step's first work: one reset of the invalid-id flag,
        // which its embedding lookups then only ever set.
        if let Err(error) = self.stream.memset_zeros(flag) {
            let _ = self.stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            return Err(error.into());
        }
        drop(flag_slot);
        *capture = Some(KernelCapture {
            advancing,
            offset,
            max_offset: u64::MAX,
            failure: None,
            has_embedding: false,
        });
        Ok(())
    }

    /// Joins `host`'s step capture, in progress: this stream waits on
    /// `host`'s, which makes every later launch on it part of `host`'s
    /// graph, until [`Self::leave_capture`]. One graph then holds every
    /// GPU's work of a tensor-parallel step and the reductions between
    /// them. Needs [`Self::prepare_capture`] first: nothing may allocate
    /// while capturing.
    pub fn join_capture(
        &self,
        host: &CudaKernels,
        advancing: BTreeSet<String>,
    ) -> Result<(), CudaError> {
        let mut capture = self.capture.lock().unwrap();
        let prepared = self.prepared_offset.lock().unwrap().take();
        let (Some(offset), true) = (prepared, capture.is_none()) else {
            return Err(CudaError::new(
                CudaErrorCode::CaptureUnsupported,
                "joining a step capture needs a prepared, idle stream",
            ));
        };
        self.stream.wait(&host.stream.record_event(None)?)?;
        let mut flag_slot = self.embedding_flag.lock().unwrap();
        if let Some(flag) = flag_slot.as_mut() {
            self.stream.memset_zeros(flag)?;
        }
        drop(flag_slot);
        *capture = Some(KernelCapture {
            advancing,
            offset,
            max_offset: u64::MAX,
            failure: None,
            has_embedding: false,
        });
        Ok(())
    }

    /// Leaves `host`'s step capture: `host`'s stream waits on this one, so
    /// `host`'s graph ends after this GPU's captured work. Fails when
    /// anything the capture cannot replay happened here.
    /// Leaves `host`'s capture whatever its state, discarding this GPU's
    /// part and any error recorded meanwhile, so `host` can end or abort
    /// its capture and this stream works again afterwards.
    pub fn abandon_join(&self, host: &CudaKernels) {
        if self.capture.lock().unwrap().take().is_some() {
            let _ = self.take_deferred_error();
            if let Ok(event) = self.stream.record_event(None) {
                let _ = host.stream.wait(&event);
            }
            let _ = self.take_deferred_error();
        }
    }

    pub fn leave_capture(&self, host: &CudaKernels) -> Result<JoinedCapture, CudaError> {
        let capture = self.capture.lock().unwrap().take().ok_or_else(|| {
            CudaError::new(
                CudaErrorCode::CaptureUnsupported,
                "no joined step capture is in progress",
            )
        })?;
        let deferred = self.take_deferred_error();
        host.stream.wait(&self.stream.record_event(None)?)?;
        if let Some(failure) = capture.failure {
            return Err(CudaError::new(CudaErrorCode::CaptureUnsupported, failure));
        }
        if let Err(error) = deferred {
            let failure = format!("the step capture met a driver error: {error:?}");
            return Err(CudaError::new(CudaErrorCode::CaptureUnsupported, failure));
        }
        Ok(JoinedCapture {
            offset: capture.offset,
            max_offset: capture.max_offset,
            has_embedding: capture.has_embedding,
        })
    }

    /// Writes a joined capture's replay offset, before `host` launches the
    /// graph: `host`'s stream then waits on this one.
    pub fn write_joined_offset(
        &self,
        joined: &mut JoinedCapture,
        offset: u64,
        host: &CudaKernels,
    ) -> Result<(), CudaError> {
        self.stream.memcpy_htod(&[offset], &mut joined.offset)?;
        host.stream.join(&self.stream)?;
        Ok(())
    }

    /// Makes this stream wait for everything queued on `host`'s so far.
    pub fn follow(&self, host: &CudaKernels) -> Result<(), CudaError> {
        self.stream.join(&host.stream)?;
        Ok(())
    }

    /// Whether this GPU may read other GPUs' memory directly (peer access
    /// enabled towards every other GPU of its Provider).
    pub fn set_peer_access(&self, enabled: bool) {
        self.peer_access
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn has_peer_access(&self) -> bool {
        self.peer_access.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Copies `src`, held by `peer`'s GPU, into this GPU's `dst`, after
    /// `peer`'s stream has written it. With peer access, a Kernel reading
    /// `peer`'s memory directly, which a step capture records; else the
    /// driver's peer copy, which cannot be captured.
    pub fn copy_from_peer(
        &self,
        peer: &CudaKernels,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        if src.len() != dst.len() {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("peer copy of {} floats into {}", src.len(), dst.len()),
            ));
        }
        if !self.has_peer_access() {
            self.stream.memcpy_dtod(src, dst)?;
            return Ok(());
        }
        self.stream.wait(&peer.stream.record_event(None)?)?;
        let source = src.device_ptr(&peer.stream).0;
        let n = src.len() as u64;
        let func = self.function("copy_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(dst).arg(&source).arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(())
    }

    /// Copies `src` into `stage` on this stream.
    pub fn stage_out(&self, src: &CudaSlice<f32>, stage: &PinnedStage) -> Result<(), CudaError> {
        if src.len() != stage.len() {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("staging {} floats through {}", src.len(), stage.len()),
            ));
        }
        self.stream.context().bind_to_thread()?;
        let source = src.device_ptr(&self.stream).0;
        // SAFETY: `stage` holds `len` floats and outlives the copy.
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(stage.host(), source, self.stream.cu_stream())
        }?;
        Ok(())
    }

    /// Copies `stage` into `dst` on this stream.
    pub fn stage_in(&self, stage: &PinnedStage, dst: &mut CudaSlice<f32>) -> Result<(), CudaError> {
        if dst.len() != stage.len() {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("staging {} floats into {}", stage.len(), dst.len()),
            ));
        }
        self.stream.context().bind_to_thread()?;
        let destination = dst.device_ptr(&self.stream).0;
        // SAFETY: `stage` holds `len` floats and outlives the copy.
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                destination,
                stage.host(),
                self.stream.cu_stream(),
            )
        }?;
        Ok(())
    }

    /// `a += b` with `b` held by `peer`'s GPU, read directly (peer access
    /// enabled) after `peer`'s stream has written it: a tensor-parallel
    /// reduction that allocates nothing and can be captured.
    pub fn accumulate_from_peer(
        &self,
        a: &mut CudaDeviceBuffer,
        peer: &CudaKernels,
        b: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        Self::require_f32(&[a, b], "accumulate")?;
        if a.shape != b.shape {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("accumulate: shape mismatch {:?} vs {:?}", a.shape, b.shape),
            ));
        }
        self.stream.wait(&peer.stream.record_event(None)?)?;
        let addend = b.slice.device_ptr(&peer.stream).0;
        let n = a.slice.len() as u64;
        let func = self.function("accumulate_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&mut a.slice).arg(&addend).arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(())
    }

    /// `a += b`, element-wise, in place: plain f32 buffers of the same
    /// shape. Allocates nothing, so it can be captured.
    pub fn accumulate(
        &self,
        a: &mut CudaDeviceBuffer,
        b: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        Self::require_f32(&[a, b], "accumulate")?;
        if a.shape != b.shape {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("accumulate: shape mismatch {:?} vs {:?}", a.shape, b.shape),
            ));
        }
        let n = a.slice.len() as u64;
        let func = self.function("accumulate_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&mut a.slice).arg(&b.slice).arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(())
    }

    /// Ends the capture and instantiates its graph. Fails when anything
    /// the capture cannot replay happened while capturing.
    pub fn end_capture(&self) -> Result<CapturedStep, CudaError> {
        let capture = self.capture.lock().unwrap().take().ok_or_else(|| {
            CudaError::new(
                CudaErrorCode::CaptureUnsupported,
                "no step capture is in progress",
            )
        })?;
        let deferred = self.take_deferred_error();
        let graph = self.stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if let Some(failure) = capture.failure {
            return Err(CudaError::new(CudaErrorCode::CaptureUnsupported, failure));
        }
        if let Err(error) = deferred {
            let failure = format!("the step capture met a driver error: {error:?}");
            return Err(CudaError::new(CudaErrorCode::CaptureUnsupported, failure));
        }
        let graph = graph?.ok_or_else(|| {
            CudaError::new(
                CudaErrorCode::CaptureUnsupported,
                "the step capture recorded no work",
            )
        })?;
        Ok(CapturedStep {
            graph: CapturedGraph(graph),
            offset: capture.offset,
            max_offset: capture.max_offset,
            has_embedding: capture.has_embedding,
        })
    }

    /// Takes the error cudarc recorded from work it could not report
    /// (freeing a buffer dropped while capturing). Left in place, it would
    /// fail the next driver call -- ending the capture included, leaving
    /// the stream capturing.
    pub(crate) fn take_deferred_error(&self) -> Result<(), cudarc::driver::DriverError> {
        self.stream.context().check_err()
    }

    /// Ends the capture in progress, if any, and discards it.
    pub fn abort_capture(&self) {
        if self.capture.lock().unwrap().take().is_some() {
            let _ = self.take_deferred_error();
            let _ = self.stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
        }
    }

    /// Launches a captured step with every advancing attribute increased by
    /// `offset`. The caller checks `offset` against `step.max_offset`.
    pub fn launch_captured(&self, step: &mut CapturedStep, offset: u64) -> Result<(), CudaError> {
        self.stream.memcpy_htod(&[offset], &mut step.offset)?;
        step.graph.0.launch()?;
        Ok(())
    }

    /// The stream every launch of these Kernels uses.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The device address of `buffer`'s allocation, as a capture records it.
    pub fn address(&self, buffer: &CudaDeviceBuffer) -> u64 {
        match (&buffer.bf16, &buffer.q8) {
            (Some(bf16), _) => bf16.device_ptr(&self.stream).0,
            (None, Some(q8)) => q8.quants.device_ptr(&self.stream).0,
            (None, None) => buffer.slice.device_ptr(&self.stream).0,
        }
    }

    /// The device address of a compact buffer's bfloat16 bits, or zero (no
    /// compact operand) for an f32 buffer.
    fn bf16_address(&self, buffer: &CudaDeviceBuffer) -> u64 {
        buffer
            .bf16
            .as_ref()
            .map_or(0, |bf16| bf16.device_ptr(&self.stream).0)
    }

    /// Fails for a compact buffer: `what` reads its operands as f32.
    fn require_f32(buffers: &[&CudaDeviceBuffer], what: &str) -> Result<(), CudaError> {
        if buffers.iter().any(|buffer| buffer.is_compact()) {
            return Err(CudaError::new(
                CudaErrorCode::DTypeUnsupported,
                format!("{what} does not read compact (bfloat16) operands"),
            ));
        }
        Ok(())
    }

    /// Writes `values` into `buffer`'s existing allocation.
    pub fn write_into(
        &self,
        buffer: &mut CudaDeviceBuffer,
        values: &[f32],
    ) -> Result<(), CudaError> {
        Self::require_f32(&[&*buffer], "a buffer write")?;
        if values.len() != buffer.slice.len() {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "{} values cannot overwrite a buffer of {}",
                    values.len(),
                    buffer.slice.len()
                ),
            ));
        }
        self.stream.memcpy_htod(values, &mut buffer.slice)?;
        self.upload_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Fails the capture in progress, if any, because of `what`: nothing
    /// that allocates, transfers or synchronizes can be captured.
    fn forbid_in_capture(&self, what: &str) -> Result<(), CudaError> {
        let mut capture = self.capture.lock().unwrap();
        match capture.as_mut() {
            Some(capture) => {
                let reason = format!("{what} cannot be captured");
                capture.failure.get_or_insert(reason.clone());
                Err(CudaError::new(CudaErrorCode::CaptureUnsupported, reason))
            }
            None => Ok(()),
        }
    }

    /// While capturing, the device address of the replay offset for a
    /// launch whose `attribute` advances, after recording that the launch
    /// stays within bounds for offsets up to `max_offset`. Zero (no offset)
    /// outside a capture or for an attribute that does not advance.
    fn advancing_offset(&self, attribute: &str, max_offset: u64) -> u64 {
        let mut capture = self.capture.lock().unwrap();
        match capture.as_mut() {
            Some(capture) if capture.advancing.contains(attribute) => {
                capture.max_offset = capture.max_offset.min(max_offset);
                let (address, _sync) = capture.offset.device_ptr(&self.stream);
                address
            }
            _ => 0,
        }
    }

    /// While capturing a bucketed prefill (`replay-bucketed-prefill`), the
    /// device address of how many of an `m`-row launch's trailing rows a
    /// replay skips, after recording that at most `m - 1` may be. Zero (no
    /// row skipped) otherwise, and for a one-row launch: a prefill's LM
    /// head reads the one gathered logits row, which no replay skips.
    fn padding_rows(&self, m: u64) -> u64 {
        if m <= 1 {
            return 0;
        }
        self.advancing_offset(PADDING_ROWS_ATTRIBUTE, m - 1)
    }

    fn is_capturing(&self) -> bool {
        self.capture.lock().unwrap().is_some()
    }

    /// Test/diagnostic-only: number of real host-to-device uploads
    /// performed so far.
    pub fn upload_count(&self) -> u64 {
        self.upload_count.load(Ordering::Relaxed)
    }

    /// Test/diagnostic-only: number of real device-to-host downloads
    /// performed so far. Does not count the small data-dependent-flag
    /// downloads `embedding_lookup`/`softmax_rows` perform internally --
    /// those are a single `i32`, not the tensor payload itself, and are
    /// not the host round-trip this counter exists to catch.
    pub fn download_count(&self) -> u64 {
        self.download_count.load(Ordering::Relaxed)
    }

    /// Whether long prefill matmuls run on the Tensor Cores.
    pub fn has_tensor_cores(&self) -> bool {
        self.tensor_module.is_some()
    }

    /// A Tensor Core Kernel function, when the Tensor Core module loaded.
    fn tensor_function(&self, name: &'static str) -> Option<CudaFunction> {
        let module = self.tensor_module.as_ref()?;
        let mut functions = self.functions.lock().unwrap();
        if let Some(function) = functions.get(name) {
            return Some(function.clone());
        }
        let function = module.load_function(name).ok()?;
        functions.insert(name, function.clone());
        Some(function)
    }

    /// Splits the contiguous f32 `a` of `len` values into its bfloat16 hi
    /// and lo terms (`tc_split_kernel`), returning their device addresses
    /// with the scratch lock. The caller holds that lock until it has
    /// queued the Kernel reading them, so another thread's split cannot
    /// land in between on the stream (Codex review on
    /// astorise/Magnetar-provider-CUDA#27).
    fn split_bf16(
        &self,
        a: &CudaDeviceBuffer,
        len: u64,
    ) -> Result<Option<SplitTerms<'_>>, CudaError> {
        Ok(self
            .split_bf16_all(&[(a, len)])?
            .map(|(lock, terms)| (lock, terms[0].0, terms[0].1)))
    }

    /// [`Self::split_bf16`] for several inputs into one scratch buffer:
    /// each input's (hi, lo) addresses, in order, with the scratch lock.
    fn split_bf16_all(
        &self,
        inputs: &[(&CudaDeviceBuffer, u64)],
    ) -> Result<Option<SplitTermsAll<'_>>, CudaError> {
        let Some(func) = self.tensor_function("tc_split_kernel") else {
            return Ok(None);
        };
        let values: u64 = inputs.iter().map(|(_, len)| len).sum();
        let Ok(total) = usize::try_from(values * 2) else {
            return Ok(None);
        };
        let mut scratch = self.split_scratch.lock().unwrap();
        if scratch.0.as_ref().is_none_or(|slice| slice.len() < total) {
            self.forbid_in_capture("growing the Tensor Core split buffer")?;
            // SAFETY: `tc_split_kernel` writes both terms of every value
            // before a Kernel on this same stream reads them.
            let grown = unsafe { self.stream.alloc::<u16>(total.max(1 << 20)) }?;
            if let Some(outgrown) = scratch.0.replace(grown) {
                scratch.1.push(outgrown);
            }
        }
        let buffer = scratch.0.as_mut().expect("allocated above");
        let mut next = buffer.device_ptr(&self.stream).0;
        let mut terms = Vec::with_capacity(inputs.len());
        for &(input, len) in inputs {
            let hi = next;
            let lo = hi + len * 2;
            next = lo + len * 2;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&input.slice).arg(&hi).arg(&lo).arg(&len);
            let config = LaunchConfig {
                grid_dim: (len.div_ceil(256).clamp(1, 4096) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { args.launch(config) }?;
            terms.push((hi, lo));
        }
        Ok(Some((scratch, terms)))
    }

    /// Room in the split scratch buffer for three bfloat16 terms of each
    /// of `regions` values (hi + mid + lo, `tc_split3_kernel`): each
    /// region's (hi, mid, lo) addresses, in order, with the scratch lock.
    /// `None` without Tensor Cores.
    fn split_bf16_3_regions(
        &self,
        regions: &[u64],
    ) -> Result<Option<SplitTerms3All<'_>>, CudaError> {
        if self.tensor_function("tc_split3_kernel").is_none() {
            return Ok(None);
        }
        let values: u64 = regions.iter().sum();
        let Ok(total) = usize::try_from(values * 3) else {
            return Ok(None);
        };
        let mut scratch = self.split_scratch.lock().unwrap();
        if scratch.0.as_ref().is_none_or(|slice| slice.len() < total) {
            self.forbid_in_capture("growing the Tensor Core split buffer")?;
            // SAFETY: `tc_split3_kernel` writes every term of every value
            // before a Kernel on this same stream reads them.
            let grown = unsafe { self.stream.alloc::<u16>(total.max(1 << 20)) }?;
            if let Some(outgrown) = scratch.0.replace(grown) {
                scratch.1.push(outgrown);
            }
        }
        let buffer = scratch.0.as_mut().expect("allocated above");
        let mut next = buffer.device_ptr(&self.stream).0;
        let mut terms = Vec::with_capacity(regions.len());
        for &len in regions {
            let hi = next;
            let mid = hi + len * 2;
            let lo = mid + len * 2;
            next = lo + len * 2;
            terms.push([hi, mid, lo]);
        }
        Ok(Some((scratch, terms)))
    }

    /// `tc_split3_kernel`: the `len` f32 values at `input` into the three
    /// bfloat16 terms at `terms` (a region of [`Self::split_bf16_3_regions`]).
    fn split_bf16_3_into(&self, input: u64, len: u64, terms: [u64; 3]) -> Result<(), CudaError> {
        let Some(func) = self.tensor_function("tc_split3_kernel") else {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "tc_split3_kernel needs Tensor Cores",
            ));
        };
        let [hi, mid, lo] = terms;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&input).arg(&hi).arg(&mid).arg(&lo).arg(&len);
        let config = LaunchConfig {
            grid_dim: (len.div_ceil(256).clamp(1, 4096) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        Ok(())
    }

    /// A Kernel's registers per thread, local memory per thread and static
    /// shared memory per block, in bytes, as compiled for this Device
    /// (diagnostics).
    #[cfg(test)]
    pub(crate) fn kernel_resources(&self, name: &'static str) -> Result<[i32; 3], CudaError> {
        let function = self.function(name)?;
        Ok([
            function.num_regs()?,
            function.local_size_bytes()?,
            function.shared_size_bytes()?,
        ])
    }

    fn function(&self, name: &'static str) -> Result<CudaFunction, CudaError> {
        let mut functions = self.functions.lock().unwrap();
        if let Some(function) = functions.get(name) {
            return Ok(function.clone());
        }
        let function = self.module.load_function(name)?;
        functions.insert(name, function.clone());
        Ok(function)
    }

    /// Uploads a host tensor into a fresh device allocation. The only
    /// host-to-device crossing point in this module -- callers (the
    /// executor's device allocation table) invoke this only when a
    /// resource does not already have a live device buffer.
    pub fn upload(&self, tensor: &HostTensor) -> Result<CudaDeviceBuffer, CudaError> {
        self.forbid_in_capture("a host-to-device upload")?;
        let slice = self.stream.clone_htod(&tensor.data)?;
        self.upload_count.fetch_add(1, Ordering::Relaxed);
        Ok(CudaDeviceBuffer {
            slice,
            shape: tensor.shape.clone(),
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// A fresh f32 buffer of `shape` whose values are all zero, set on the
    /// Device: nothing is built on the host or transferred.
    pub fn zeros(&self, shape: &[u64]) -> Result<CudaDeviceBuffer, CudaError> {
        self.forbid_in_capture("a device allocation")?;
        let len = shape.iter().try_fold(1usize, |len, &dimension| {
            usize::try_from(dimension)
                .ok()
                .and_then(|dimension| len.checked_mul(dimension))
        });
        let len = len.ok_or_else(|| {
            CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("a zeroed tensor of shape {shape:?} does not fit in memory"),
            )
        })?;
        Ok(CudaDeviceBuffer {
            slice: self.stream.alloc_zeros::<f32>(len)?,
            shape: shape.to_vec(),
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// Uploads a Model Instance weight (`execute-native-half-precision`):
    /// compact, as bfloat16 bits, when it is a matrix of at least
    /// [`COMPACT_WEIGHT_MIN_ELEMENTS`] values that are all exactly bfloat16
    /// values (as every weight of a bfloat16 checkpoint is), otherwise as
    /// [`Self::upload`] does. Lossless either way.
    pub fn upload_weight(&self, tensor: &HostTensor) -> Result<CudaDeviceBuffer, CudaError> {
        if tensor.shape.len() != 2 || tensor.data.len() < COMPACT_WEIGHT_MIN_ELEMENTS {
            return self.upload(tensor);
        }
        // Checked before anything is allocated: a weight that is not exact
        // is uploaded as f32 with no compact allocation alive alongside.
        if !tensor
            .data
            .iter()
            .all(|&value| exact_bf16_bits(value).is_some())
        {
            return self.upload(tensor);
        }
        self.forbid_in_capture("a host-to-device upload")?;
        // Converted and uploaded a chunk at a time: a whole host copy of the
        // bits would add half of the largest weight to the peak host memory.
        const CHUNK: usize = 1 << 20;
        // SAFETY: every element is written by the chunk copies below before
        // anything reads the buffer.
        let mut bf16 = unsafe { self.stream.alloc::<u16>(tensor.data.len()) }?;
        let mut bits = Vec::with_capacity(CHUNK.min(tensor.data.len()));
        for (index, chunk) in tensor.data.chunks(CHUNK).enumerate() {
            bits.clear();
            bits.extend(chunk.iter().map(|&value| (value.to_bits() >> 16) as u16));
            let start = index * CHUNK;
            let mut destination = bf16.slice_mut(start..start + chunk.len());
            self.stream.memcpy_htod(&bits, &mut destination)?;
        }
        // The f32 placeholder no Kernel reads.
        let slice = self.stream.alloc_zeros::<f32>(1)?;
        self.upload_count.fetch_add(1, Ordering::Relaxed);
        Ok(CudaDeviceBuffer {
            slice,
            shape: tensor.shape.clone(),
            bf16: Some(bf16),
            q8: None,
            tq: None,
        })
    }

    /// The same values as a compact `buffer`, stored as f32: for a Kernel
    /// that only reads f32 operands.
    pub fn expand(&self, buffer: &CudaDeviceBuffer) -> Result<CudaDeviceBuffer, CudaError> {
        if buffer.is_tq() {
            return self.tq_expand(buffer);
        }
        if buffer.is_q8() {
            return self.dequantize_q8(buffer);
        }
        let tensor = self.download(buffer)?;
        self.upload(&tensor)
    }

    /// Stores a packed weight as the checkpoint stored it
    /// (`execute-direct-quantized-weights`): its `Q8_0` quants and one f32
    /// scale per block, with the buffer shaped as the logical matrix.
    pub fn upload_packed_weight(
        &self,
        weight: &PackedWeight,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.forbid_in_capture("a host-to-device upload")?;
        if matches!(
            weight.format(),
            PackedWeightFormat::Q2K | PackedWeightFormat::Q3K
        ) {
            return self.upload_low_bit_weight(weight);
        }
        if matches!(
            weight.format(),
            PackedWeightFormat::Iq4Nl | PackedWeightFormat::Iq4Xs | PackedWeightFormat::Q4_0
        ) {
            return self.upload_iq4_weight(weight);
        }
        if iq_format(weight.format()).is_some() {
            return self.upload_iq_weight(weight);
        }
        if weight.format() == PackedWeightFormat::Q6K {
            // Each super-block without its trailing f16 `d` (208 bytes,
            // so every one stays 16-byte aligned), `d` as an f32 apart.
            let blocks = weight.bytes().as_chunks::<210>().0;
            let mut quants = Vec::with_capacity(blocks.len() * 208);
            let mut scales = Vec::with_capacity(blocks.len());
            for block in blocks {
                quants.extend(block[..208].iter().map(|&byte| byte as i8));
                scales.push(half_precision::f16_to_f32(u16::from_le_bytes([
                    block[208], block[209],
                ])));
            }
            let quants = self.stream.clone_htod(&quants)?;
            let scales = self.stream.clone_htod(&scales)?;
            let slice = self.stream.alloc_zeros::<f32>(1)?;
            self.upload_count.fetch_add(1, Ordering::Relaxed);
            return Ok(CudaDeviceBuffer {
                slice,
                shape: weight.logical_shape().to_vec(),
                bf16: None,
                q8: Some(CudaQ8Weight {
                    quants,
                    scales,
                    format: PackedWeightFormat::Q6K,
                    stored_rows: weight.stored_rows(),
                    stored_cols: weight.stored_cols(),
                    transposed: weight.transposed(),
                }),
                tq: None,
            });
        }
        if matches!(
            weight.format(),
            PackedWeightFormat::Q4K | PackedWeightFormat::Q5K
        ) {
            // Kept as stored: the Kernels decode the super-blocks (144 and
            // 176 bytes, both multiples of 16).
            let bytes: Vec<i8> = weight.bytes().iter().map(|&byte| byte as i8).collect();
            let quants = self.stream.clone_htod(&bytes)?;
            let scales = self.stream.alloc_zeros::<f32>(1)?;
            let slice = self.stream.alloc_zeros::<f32>(1)?;
            self.upload_count.fetch_add(1, Ordering::Relaxed);
            return Ok(CudaDeviceBuffer {
                slice,
                shape: weight.logical_shape().to_vec(),
                bf16: None,
                q8: Some(CudaQ8Weight {
                    quants,
                    scales,
                    format: weight.format(),
                    stored_rows: weight.stored_rows(),
                    stored_cols: weight.stored_cols(),
                    transposed: weight.transposed(),
                }),
                tq: None,
            });
        }
        let blocks = weight.bytes().as_chunks::<34>().0;
        let mut quants = Vec::with_capacity(blocks.len() * 32);
        let mut scales = Vec::with_capacity(blocks.len());
        for block in blocks {
            scales.push(half_precision::f16_to_f32(u16::from_le_bytes([
                block[0], block[1],
            ])));
            quants.extend(block[2..].iter().map(|&q| q as i8));
        }
        let quants = self.stream.clone_htod(&quants)?;
        let scales = self.stream.clone_htod(&scales)?;
        // The f32 placeholder no Kernel reads.
        let slice = self.stream.alloc_zeros::<f32>(1)?;
        self.upload_count.fetch_add(1, Ordering::Relaxed);
        Ok(CudaDeviceBuffer {
            slice,
            shape: weight.logical_shape().to_vec(),
            bf16: None,
            q8: Some(CudaQ8Weight {
                quants,
                scales,
                format: PackedWeightFormat::Q8_0,
                stored_rows: weight.stored_rows(),
                stored_cols: weight.stored_cols(),
                transposed: weight.transposed(),
            }),
            tq: None,
        })
    }

    /// Stores a `Q2_K` or `Q3_K` weight in the layout
    /// `lowk_matmul_rows_int8_kernel` reads: `Q2_K` super-blocks without
    /// their `d` and `dmin` (80 bytes), those two as f32 apart; `Q3_K`
    /// super-blocks as their mask and quants with the sixteen scales
    /// unpacked to signed bytes (112 bytes), `d` as an f32 apart. Every
    /// super-block stays 16-byte aligned.
    fn upload_low_bit_weight(&self, weight: &PackedWeight) -> Result<CudaDeviceBuffer, CudaError> {
        let format = weight.format();
        let block_bytes = format.block_bytes() as usize;
        let blocks = weight.bytes().chunks_exact(block_bytes);
        let mut quants = Vec::with_capacity(blocks.len() * 112);
        let mut scales = Vec::with_capacity(blocks.len() * 2);
        let half = |lo: u8, hi: u8| half_precision::f16_to_f32(u16::from_le_bytes([lo, hi]));
        for block in blocks {
            if format == PackedWeightFormat::Q2K {
                quants.extend(block[..80].iter().map(|&byte| byte as i8));
                scales.push(half(block[80], block[81]));
                scales.push(half(block[82], block[83]));
            } else {
                quants.extend(block[..96].iter().map(|&byte| byte as i8));
                quants.extend(
                    magnetar_provider_api::q3_k_scales(&block[96..108])
                        .iter()
                        .map(|&scale| scale - 32),
                );
                scales.push(half(block[108], block[109]));
            }
        }
        let quants = self.stream.clone_htod(&quants)?;
        let scales = self.stream.clone_htod(&scales)?;
        let slice = self.stream.alloc_zeros::<f32>(1)?;
        self.upload_count.fetch_add(1, Ordering::Relaxed);
        Ok(CudaDeviceBuffer {
            slice,
            shape: weight.logical_shape().to_vec(),
            bf16: None,
            q8: Some(CudaQ8Weight {
                quants,
                scales,
                format,
                stored_rows: weight.stored_rows(),
                stored_cols: weight.stored_cols(),
                transposed: weight.transposed(),
            }),
            tq: None,
        })
    }

    /// Stores an `IQ4_NL`, `IQ4_XS` or `Q4_0` weight in the layout
    /// `iq4_matmul_rows_int8_kernel` reads: every 32-value run's 16 index
    /// bytes in stored order, then (`IQ4_XS`) one signed byte per run, its
    /// sub-block scale minus 32; `scales` holds each `IQ4_NL` or `Q4_0`
    /// block's `d`, or each `IQ4_XS` super-block's.
    fn upload_iq4_weight(&self, weight: &PackedWeight) -> Result<CudaDeviceBuffer, CudaError> {
        let format = weight.format();
        let block_bytes = format.block_bytes() as usize;
        let blocks = weight.bytes().chunks_exact(block_bytes);
        let runs = weight.bytes().len() / block_bytes * (format.block_elements() as usize / 32);
        let mut quants: Vec<i8> = Vec::with_capacity(runs * 17);
        let mut subscales = Vec::new();
        let mut scales = Vec::with_capacity(blocks.len());
        for block in blocks {
            scales.push(half_precision::f16_to_f32(u16::from_le_bytes([
                block[0], block[1],
            ])));
            if format != PackedWeightFormat::Iq4Xs {
                quants.extend(block[2..18].iter().map(|&byte| byte as i8));
            } else {
                subscales.extend(magnetar_provider_api::iq4_xs_scales(&block[2..8]));
                quants.extend(block[8..136].iter().map(|&byte| byte as i8));
            }
        }
        quants.extend(subscales);
        let quants = self.stream.clone_htod(&quants)?;
        let scales = self.stream.clone_htod(&scales)?;
        let slice = self.stream.alloc_zeros::<f32>(1)?;
        self.upload_count.fetch_add(1, Ordering::Relaxed);
        Ok(CudaDeviceBuffer {
            slice,
            shape: weight.logical_shape().to_vec(),
            bf16: None,
            q8: Some(CudaQ8Weight {
                quants,
                scales,
                format,
                stored_rows: weight.stored_rows(),
                stored_cols: weight.stored_cols(),
                transposed: weight.transposed(),
            }),
            tq: None,
        })
    }

    /// Stores an IQ2/IQ3 weight in the layout `iq_matmul_rows_int8_kernel`
    /// reads: each super-block's bytes after its leading `f16` `d` (a
    /// multiple of four, so word loads stay aligned), `d` as an f32 apart.
    fn upload_iq_weight(&self, weight: &PackedWeight) -> Result<CudaDeviceBuffer, CudaError> {
        let format = weight.format();
        let block_bytes = format.block_bytes() as usize;
        let blocks = weight.bytes().chunks_exact(block_bytes);
        let mut quants = Vec::with_capacity(blocks.len() * (block_bytes - 2));
        let mut scales = Vec::with_capacity(blocks.len());
        for block in blocks {
            scales.push(half_precision::f16_to_f32(u16::from_le_bytes([
                block[0], block[1],
            ])));
            quants.extend(block[2..].iter().map(|&byte| byte as i8));
        }
        let quants = self.stream.clone_htod(&quants)?;
        let scales = self.stream.clone_htod(&scales)?;
        let slice = self.stream.alloc_zeros::<f32>(1)?;
        self.upload_count.fetch_add(1, Ordering::Relaxed);
        Ok(CudaDeviceBuffer {
            slice,
            shape: weight.logical_shape().to_vec(),
            bf16: None,
            q8: Some(CudaQ8Weight {
                quants,
                scales,
                format,
                stored_rows: weight.stored_rows(),
                stored_cols: weight.stored_cols(),
                transposed: weight.transposed(),
            }),
            tq: None,
        })
    }

    /// A packed buffer's logical matrix as f32, dequantized on the Device.
    pub fn dequantize_q8(&self, buffer: &CudaDeviceBuffer) -> Result<CudaDeviceBuffer, CudaError> {
        self.dequantize_packed(buffer, None)
    }

    /// A packed buffer's logical matrix as f32, or with `rows` (a table
    /// stored untransposed) only the stored rows its ids name, shaped
    /// `[ids, columns]`: an embedding lookup then reads those rows' blocks,
    /// not the whole table. An invalid id sets the rows' flag.
    fn dequantize_packed(
        &self,
        buffer: &CudaDeviceBuffer,
        rows: Option<PackedRows>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        let q8 = buffer.q8.as_ref().ok_or_else(|| {
            CudaError::new(CudaErrorCode::DTypeUnsupported, "not a packed weight")
        })?;
        if rows.is_some() && q8.transposed {
            return Err(CudaError::new(
                CudaErrorCode::DTypeUnsupported,
                "rows of a packed weight stored transposed",
            ));
        }
        let PackedRows {
            ids: row_ids,
            count: id_count,
            invalid_id_flag,
        } = rows.unwrap_or(PackedRows {
            ids: 0,
            count: q8.stored_rows,
            invalid_id_flag: 0,
        });
        let elements = id_count * q8.stored_cols;
        let mut out = self.alloc_output(elements as usize)?;
        let transposed = u32::from(q8.transposed);
        match q8.format {
            PackedWeightFormat::Q8_0 => {
                let func = self.function("q8_dequantize_kernel")?;
                let mut args = self.stream.launch_builder(&func);
                args.arg(&q8.quants)
                    .arg(&q8.scales)
                    .arg(&mut out)
                    .arg(&q8.stored_rows)
                    .arg(&q8.stored_cols)
                    .arg(&transposed)
                    .arg(&row_ids)
                    .arg(&id_count)
                    .arg(&invalid_id_flag);
                unsafe { args.launch(LaunchConfig::for_num_elems(elements as u32)) }?;
            }
            PackedWeightFormat::Iq2Xxs
            | PackedWeightFormat::Iq2Xs
            | PackedWeightFormat::Iq2S
            | PackedWeightFormat::Iq3Xxs
            | PackedWeightFormat::Iq3S => {
                let func = self.function("iq_dequantize_kernel")?;
                let format = iq_format(q8.format).expect("an IQ2/IQ3 format");
                let mut args = self.stream.launch_builder(&func);
                args.arg(&format)
                    .arg(&q8.quants)
                    .arg(&q8.scales)
                    .arg(&mut out)
                    .arg(&q8.stored_rows)
                    .arg(&q8.stored_cols)
                    .arg(&transposed)
                    .arg(&row_ids)
                    .arg(&id_count)
                    .arg(&invalid_id_flag);
                unsafe { args.launch(LaunchConfig::for_num_elems(elements as u32)) }?;
            }
            PackedWeightFormat::Iq4Nl | PackedWeightFormat::Iq4Xs | PackedWeightFormat::Q4_0 => {
                let func = self.function("iq4_dequantize_kernel")?;
                let format = iq4_format(q8.format);
                let subscales = q8.quants.slice(iq4_subscale_offset(q8)..);
                let mut args = self.stream.launch_builder(&func);
                args.arg(&format)
                    .arg(&q8.quants)
                    .arg(&subscales)
                    .arg(&q8.scales)
                    .arg(&mut out)
                    .arg(&q8.stored_rows)
                    .arg(&q8.stored_cols)
                    .arg(&transposed)
                    .arg(&row_ids)
                    .arg(&id_count)
                    .arg(&invalid_id_flag);
                unsafe { args.launch(LaunchConfig::for_num_elems(elements as u32)) }?;
            }
            PackedWeightFormat::Q2K | PackedWeightFormat::Q3K => {
                let func = self.function("lowk_dequantize_kernel")?;
                let format = low_bit_format(q8.format);
                let mut args = self.stream.launch_builder(&func);
                args.arg(&format)
                    .arg(&q8.quants)
                    .arg(&q8.scales)
                    .arg(&mut out)
                    .arg(&q8.stored_rows)
                    .arg(&q8.stored_cols)
                    .arg(&transposed)
                    .arg(&row_ids)
                    .arg(&id_count)
                    .arg(&invalid_id_flag);
                unsafe { args.launch(LaunchConfig::for_num_elems(elements as u32)) }?;
            }
            PackedWeightFormat::Q6K => {
                let func = self.function("q6k_dequantize_kernel")?;
                let mut args = self.stream.launch_builder(&func);
                args.arg(&q8.quants)
                    .arg(&q8.scales)
                    .arg(&mut out)
                    .arg(&q8.stored_rows)
                    .arg(&q8.stored_cols)
                    .arg(&transposed)
                    .arg(&row_ids)
                    .arg(&id_count)
                    .arg(&invalid_id_flag);
                unsafe { args.launch(LaunchConfig::for_num_elems(elements as u32)) }?;
            }
            PackedWeightFormat::Q4K | PackedWeightFormat::Q5K => {
                let func = self.function(if q8.format == PackedWeightFormat::Q5K {
                    "q5k_dequantize_kernel"
                } else {
                    "q4k_dequantize_kernel"
                })?;
                let mut args = self.stream.launch_builder(&func);
                args.arg(&q8.quants)
                    .arg(&mut out)
                    .arg(&q8.stored_rows)
                    .arg(&q8.stored_cols)
                    .arg(&transposed)
                    .arg(&row_ids)
                    .arg(&id_count)
                    .arg(&invalid_id_flag);
                unsafe { args.launch(LaunchConfig::for_num_elems(elements as u32)) }?;
            }
        }
        let shape = if row_ids == 0 {
            buffer.shape.clone()
        } else {
            vec![id_count, q8.stored_cols]
        };
        Ok(CudaDeviceBuffer {
            slice: out,
            shape,
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// Downloads a device buffer to host-visible bytes. The only
    /// device-to-host crossing point in this module -- callers invoke this
    /// only when host-visible bytes are genuinely requested (`read_tensor`,
    /// or `TensorValue::into_host` at a real materialization boundary), not
    /// as an automatic step of every Kernel invocation.
    pub fn download(&self, buffer: &CudaDeviceBuffer) -> Result<HostTensor, CudaError> {
        self.forbid_in_capture("a device-to-host download")?;
        if buffer.is_tq() {
            return self.download(&self.tq_expand(buffer)?);
        }
        if buffer.is_q8() {
            return self.download(&self.dequantize_q8(buffer)?);
        }
        let data = match &buffer.bf16 {
            Some(bf16) => {
                let bits = self.stream.clone_dtoh(bf16)?;
                bits.iter()
                    .map(|&bits| f32::from_bits(u32::from(bits) << 16))
                    .collect()
            }
            None => self.stream.clone_dtoh(&buffer.slice)?,
        };
        self.stream.synchronize()?;
        self.download_count.fetch_add(1, Ordering::Relaxed);
        HostTensor::new(buffer.shape.clone(), data).map_err(host_error)
    }

    /// Rows `first_row..first_row + rows` of a rank-2 f32 buffer, shaped
    /// `[rows, columns]`: only those rows cross to the host. `None` for a
    /// compact buffer, another rank or a range past its end.
    pub fn download_rows(
        &self,
        buffer: &CudaDeviceBuffer,
        first_row: u64,
        rows: u64,
    ) -> Result<Option<HostTensor>, CudaError> {
        self.forbid_in_capture("a device-to-host download")?;
        let [total_rows, columns] = buffer.shape[..] else {
            return Ok(None);
        };
        if buffer.is_compact() || first_row.saturating_add(rows) > total_rows {
            return Ok(None);
        }
        let start = (first_row * columns) as usize;
        let end = ((first_row + rows) * columns) as usize;
        let data = self.stream.clone_dtoh(&buffer.slice.slice(start..end))?;
        self.stream.synchronize()?;
        self.download_count.fetch_add(1, Ordering::Relaxed);
        HostTensor::new([rows, columns], data)
            .map(Some)
            .map_err(host_error)
    }

    /// Uploads a host tensor into a fresh device allocation as real
    /// half-precision bytes, converting each `f32` value on the host via
    /// this crate's `half_precision::f32_to_f16`/`f32_to_bf16` before the
    /// upload -- `HostTensor` itself never carries a non-`f32` byte
    /// (`add-native-cuda-half-precision-compute` design.md's decision not
    /// to retype the shared Runtime<->Provider transport type).
    pub fn upload_half(
        &self,
        tensor: &HostTensor,
        dtype: CudaHalfDType,
    ) -> Result<CudaHalfBuffer, CudaError> {
        self.forbid_in_capture("a host-to-device upload")?;
        let bits: Vec<u16> = match dtype {
            CudaHalfDType::Float16 => tensor
                .data
                .iter()
                .map(|&value| half_precision::f32_to_f16(value))
                .collect(),
            CudaHalfDType::BrainFloat16 => tensor
                .data
                .iter()
                .map(|&value| half_precision::f32_to_bf16(value))
                .collect(),
        };
        let slice = self.stream.clone_htod(&bits)?;
        self.upload_count.fetch_add(1, Ordering::Relaxed);
        Ok(CudaHalfBuffer {
            slice,
            shape: tensor.shape.clone(),
            dtype,
        })
    }

    /// Downloads a half-precision device buffer to host-visible `f32`
    /// bytes, converting each value via this crate's
    /// `half_precision::f16_to_f32`/`bf16_to_f32` -- the inverse of
    /// [`Self::upload_half`].
    pub fn download_half(&self, buffer: &CudaHalfBuffer) -> Result<HostTensor, CudaError> {
        self.forbid_in_capture("a device-to-host download")?;
        let bits = self.stream.clone_dtoh(&buffer.slice)?;
        self.stream.synchronize()?;
        self.download_count.fetch_add(1, Ordering::Relaxed);
        let data: Vec<f32> = match buffer.dtype {
            CudaHalfDType::Float16 => bits
                .iter()
                .map(|&b| half_precision::f16_to_f32(b))
                .collect(),
            CudaHalfDType::BrainFloat16 => bits
                .iter()
                .map(|&b| half_precision::bf16_to_f32(b))
                .collect(),
        };
        HostTensor::new(buffer.shape.clone(), data).map_err(host_error)
    }

    /// `a + b`, element-wise, computed and stored as real on-device
    /// half-precision bit patterns (`add-native-cuda-half-precision-
    /// compute` Phase 2) -- `a` and `b` must share the same shape and
    /// [`CudaHalfDType`]. Unlike [`Self::add`], no row-broadcast variant
    /// exists yet; this is a narrowly-scoped first increment, not full
    /// parity with the `f32` path.
    pub fn add_half(
        &self,
        a: &CudaHalfBuffer,
        b: &CudaHalfBuffer,
    ) -> Result<CudaHalfBuffer, CudaError> {
        same_shape_half(a, b)?;
        self.forbid_in_capture("a half-precision Kernel")?;
        let n = a.slice.len() as u64;
        let mut out_dev = self.stream.alloc_zeros::<u16>(a.slice.len())?;
        let func = self.function(half_kernel_name("add", a.dtype))?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&a.slice).arg(&b.slice).arg(&mut out_dev).arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(CudaHalfBuffer {
            slice: out_dev,
            shape: a.shape.clone(),
            dtype: a.dtype,
        })
    }

    /// `a * b`, element-wise. See [`Self::add_half`].
    pub fn mul_half(
        &self,
        a: &CudaHalfBuffer,
        b: &CudaHalfBuffer,
    ) -> Result<CudaHalfBuffer, CudaError> {
        same_shape_half(a, b)?;
        self.forbid_in_capture("a half-precision Kernel")?;
        let n = a.slice.len() as u64;
        let mut out_dev = self.stream.alloc_zeros::<u16>(a.slice.len())?;
        let func = self.function(half_kernel_name("mul", a.dtype))?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&a.slice).arg(&b.slice).arg(&mut out_dev).arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(CudaHalfBuffer {
            slice: out_dev,
            shape: a.shape.clone(),
            dtype: a.dtype,
        })
    }

    /// Duplicates `buffer`'s current bytes into a fresh device allocation
    /// via a real device-to-device copy -- never touching host memory.
    /// The executor-level `copy_tensor_admitted` primitive
    /// (`implement-device-resident-multi-step-cuda-decode`) uses this to
    /// give a Tensor Resource's bytes a second, independent identity
    /// (e.g. promoting a decode step's pending KV write to its committed
    /// identity) without a D2H+H2D round trip.
    /// Writes `rows` over `store`'s rows `row_offset..row_offset + rows` in
    /// place, with one device-to-device copy (`kv-write`,
    /// `append-kv-history-in-place`). Rows outside that range are unchanged.
    pub fn kv_write(
        &self,
        store: &mut CudaDeviceBuffer,
        rows: &CudaDeviceBuffer,
        row_offset: u64,
    ) -> Result<(), CudaError> {
        if store.is_tq() {
            return self.tq_kv_write(store, rows, row_offset);
        }
        Self::require_f32(&[&*store, rows], "kv-write")?;
        let (capacity, width) = rows_cols_buf(store)?;
        let (count, row_width) = rows_cols_buf(rows)?;
        if row_width != width {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("kv-write rows have width {row_width}, the store {width}"),
            ));
        }
        if row_offset
            .checked_add(count)
            .is_none_or(|end| end > capacity)
        {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "kv-write of {count} rows at {row_offset} exceeds the store's {capacity} rows"
                ),
            ));
        }
        // A Kernel rather than a copy: a captured copy's destination is
        // fixed, while a captured `kv-write` writes at its row offset plus
        // the replay offset.
        let step_offset = self.advancing_offset("row_offset", capacity - (row_offset + count));
        let n = rows.slice.len() as u64;
        if n == 0 {
            return Ok(());
        }
        let func = self.function("kv_write_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&rows.slice)
            .arg(&mut store.slice)
            .arg(&n)
            .arg(&width)
            .arg(&row_offset)
            .arg(&step_offset);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(())
    }

    /// Offers `buffer` to the next [`Self::alloc_output`] of its length
    /// until the returned guard drops. The caller must own `buffer`
    /// exclusively (removed from its storage) and must not read it again:
    /// a Kernel writing its output into it is ordered after every earlier
    /// use on this Provider's single stream.
    pub fn offer_output(&self, buffer: Option<CudaSlice<f32>>) -> OfferedOutput<'_> {
        *self.offered_output.lock().unwrap() = buffer;
        OfferedOutput { kernels: self }
    }

    /// An output buffer of `len` floats left uninitialized, for a Kernel
    /// (or copy) that writes every one of its elements before anything
    /// reads it. Skips the memset `alloc_zeros` would queue for each of the
    /// hundreds of outputs a decode step produces. Callers whose Kernel may
    /// leave elements unwritten keep `alloc_zeros`.
    fn alloc_output(&self, len: usize) -> Result<CudaSlice<f32>, CudaError> {
        {
            let mut offered = self.offered_output.lock().unwrap();
            if offered.as_ref().is_some_and(|slice| slice.len() == len) {
                // Every queued use of the offered buffer precedes this
                // Kernel on the one stream this Provider uses.
                return Ok(offered.take().expect("checked above"));
            }
        }
        self.forbid_in_capture("a device allocation")?;
        // SAFETY: every caller launches a Kernel or copy that writes all
        // `len` elements on this same stream before the buffer is read.
        Ok(unsafe { self.stream.alloc::<f32>(len) }?)
    }

    pub fn clone_buffer(&self, buffer: &CudaDeviceBuffer) -> Result<CudaDeviceBuffer, CudaError> {
        self.forbid_in_capture("a buffer copy")?;
        let slice = self.stream.clone_dtod(&buffer.slice)?;
        if let Some(tq) = &buffer.tq {
            return Ok(CudaDeviceBuffer {
                slice,
                shape: buffer.shape.clone(),
                bf16: None,
                q8: None,
                tq: Some(CudaTqStore {
                    codes: self.stream.clone_dtod(&tq.codes)?,
                    norms: self.stream.clone_dtod(&tq.norms)?,
                    offsets: self.stream.clone_dtod(&tq.offsets)?,
                    precision: tq.precision,
                    bits: tq.bits,
                    head_dimension: tq.head_dimension,
                    heads: tq.heads,
                    signs: self.stream.clone_dtod(&tq.signs)?,
                    centroids: self.stream.clone_dtod(&tq.centroids)?,
                    boundaries: self.stream.clone_dtod(&tq.boundaries)?,
                    exact: match &tq.exact {
                        Some(exact) => Some(Box::new(self.clone_buffer(exact)?)),
                        None => None,
                    },
                    exact_rows: tq.exact_rows,
                }),
            });
        }
        if buffer.is_q8() {
            return Err(CudaError::new(
                CudaErrorCode::DTypeUnsupported,
                "a packed weight is never copied",
            ));
        }
        let bf16 = match &buffer.bf16 {
            Some(bf16) => Some(self.stream.clone_dtod(bf16)?),
            None => None,
        };
        Ok(CudaDeviceBuffer {
            slice,
            shape: buffer.shape.clone(),
            bf16,
            q8: None,
            tq: None,
        })
    }

    /// A zeroed TurboQuant KV store of `shape` (`[capacity, heads *
    /// head_dimension]`) in `precision` (`turboquant-kv-history`): zero
    /// norms, so every row reads back as zeros until `kv-write` writes it.
    pub fn tq_zeros(
        &self,
        shape: &[u64],
        precision: KvStorePrecision,
        head_dimension: u64,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.tq_zeros_with_exact_head(shape, precision, head_dimension, 0)
    }

    /// [`Self::tq_zeros`] whose first `exact_rows` rows (at most the
    /// capacity) are kept exact in an f32 head
    /// (`keep-turboquant-kv-head-exact`): `kv-write` copies them there,
    /// and every read takes them from there.
    pub fn tq_zeros_with_exact_head(
        &self,
        shape: &[u64],
        precision: KvStorePrecision,
        head_dimension: u64,
        exact_rows: u64,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.forbid_in_capture("a device allocation")?;
        let unsupported = |reason: String| CudaError::new(CudaErrorCode::ShapeUnsupported, reason);
        let bits = precision.bits().ok_or_else(|| {
            unsupported(format!(
                "{} is not a TurboQuant precision",
                precision.as_str()
            ))
        })?;
        let &[capacity, width] = shape else {
            return Err(unsupported(format!(
                "a KV store is [capacity, width], not {shape:?}"
            )));
        };
        if !precision.supports_head_dimension(head_dimension)
            || head_dimension > TQ_MAX_DIMENSION
            || !width.is_multiple_of(head_dimension)
        {
            return Err(unsupported(format!(
                "a {} store of {width}-wide rows cannot hold {head_dimension}-wide heads",
                precision.as_str()
            )));
        }
        let heads = width / head_dimension;
        let cells = usize::try_from(capacity * heads)
            .map_err(|_| unsupported("the KV store does not fit in memory".into()))?;
        let code_bytes = (head_dimension * u64::from(bits) / 8) as usize;
        let codebook = turboquant_codebook(head_dimension, bits);
        let exact_rows = exact_rows.min(capacity);
        let exact = match exact_rows {
            0 => None,
            rows => Some(Box::new(self.zeros(&[rows, width])?)),
        };
        Ok(CudaDeviceBuffer {
            slice: self.stream.alloc_zeros::<f32>(1)?,
            shape: shape.to_vec(),
            bf16: None,
            q8: None,
            tq: Some(CudaTqStore {
                codes: self.stream.alloc_zeros::<u8>(cells * code_bytes)?,
                norms: self.stream.alloc_zeros::<f32>(cells)?,
                offsets: self.stream.alloc_zeros::<f32>(width as usize)?,
                precision,
                bits,
                head_dimension,
                heads,
                signs: self.stream.clone_htod(&turboquant_signs(head_dimension))?,
                centroids: self.stream.clone_htod(&codebook.centroids)?,
                boundaries: self.stream.clone_htod(&codebook.boundaries)?,
                exact,
                exact_rows,
            }),
        })
    }

    /// `kv-write` into a TurboQuant store: each head of each row quantized
    /// in the kernel that writes it, at the captured step's advancing
    /// offset (`turboquant-kv-history`).
    fn tq_kv_write(
        &self,
        store: &mut CudaDeviceBuffer,
        rows: &CudaDeviceBuffer,
        row_offset: u64,
    ) -> Result<(), CudaError> {
        Self::require_f32(&[rows], "kv-write rows")?;
        let (count, row_width) = rows_cols_buf(rows)?;
        let &[capacity, width] = store.shape.as_slice() else {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("a KV store is [capacity, width], not {:?}", store.shape),
            ));
        };
        if row_width != width {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("kv-write rows have width {row_width}, the store {width}"),
            ));
        }
        if row_offset
            .checked_add(count)
            .is_none_or(|end| end > capacity)
        {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "kv-write of {count} rows at {row_offset} exceeds the store's {capacity} rows"
                ),
            ));
        }
        let step_offset = self.advancing_offset("row_offset", capacity - (row_offset + count));
        if count == 0 {
            return Ok(());
        }
        let tq = store.tq.as_mut().expect("a TurboQuant store");
        // The write at row 0 starts the history: it sets the offsets from
        // the rows it computes (`tq_offsets_kernel`).
        if row_offset == 0 && step_offset == 0 {
            let padding = self.padding_rows(count);
            let func = self.function("tq_offsets_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&rows.slice)
                .arg(&mut tq.offsets)
                .arg(&width)
                .arg(&count)
                .arg(&padding);
            unsafe { args.launch(LaunchConfig::for_num_elems(width as u32)) }?;
        }
        let func = self.function("tq_kv_write_kernel")?;
        let no_head = 0u64;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&rows.slice)
            .arg(&tq.offsets)
            .arg(&mut tq.codes)
            .arg(&mut tq.norms)
            .arg(&tq.signs)
            .arg(&tq.boundaries)
            .arg(&tq.heads)
            .arg(&tq.head_dimension)
            .arg(&tq.bits)
            .arg(&row_offset)
            .arg(&step_offset);
        match tq.exact.as_mut() {
            Some(exact) => args.arg(&mut exact.slice),
            None => args.arg(&no_head),
        };
        args.arg(&tq.exact_rows);
        let config = LaunchConfig {
            grid_dim: (count as u32, tq.heads as u32, 1),
            block_dim: (tq_threads(tq.head_dimension), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        Ok(())
    }

    /// Writes the values of `store`'s first `rows` rows, plus the captured
    /// step's `step_offset` (a device address, or 0), into `out` as
    /// `[capacity, width]` f32.
    fn tq_dequantize_into(
        &self,
        store: &CudaDeviceBuffer,
        out: &mut CudaSlice<f32>,
        rows: u64,
        step_offset: u64,
    ) -> Result<(), CudaError> {
        let tq = store.tq.as_ref().expect("a TurboQuant store");
        let capacity = store.shape[0];
        if capacity == 0 {
            return Ok(());
        }
        let func = self.function("tq_dequantize_kernel")?;
        let no_head = 0u64;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&tq.codes)
            .arg(&tq.norms)
            .arg(&tq.signs)
            .arg(&tq.centroids)
            .arg(&tq.offsets)
            .arg(out)
            .arg(&tq.heads)
            .arg(&tq.head_dimension)
            .arg(&tq.bits)
            .arg(&rows)
            .arg(&step_offset);
        match tq.exact.as_ref() {
            Some(exact) => args.arg(&exact.slice),
            None => args.arg(&no_head),
        };
        args.arg(&tq.exact_rows);
        let config = LaunchConfig {
            grid_dim: (capacity as u32, tq.heads as u32, 1),
            block_dim: (tq_threads(tq.head_dimension), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        Ok(())
    }

    /// A TurboQuant store's values as an f32 buffer of its shape.
    pub fn tq_expand(&self, store: &CudaDeviceBuffer) -> Result<CudaDeviceBuffer, CudaError> {
        self.forbid_in_capture("a device allocation")?;
        let len = store.shape.iter().product::<u64>() as usize;
        let mut out = self.stream.alloc_zeros::<f32>(len.max(1))?;
        self.tq_dequantize_into(store, &mut out, store.shape[0], 0)?;
        Ok(CudaDeviceBuffer {
            slice: out,
            shape: store.shape.clone(),
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// Attention reading TurboQuant K and/or V (`turboquant-kv-history`): a
    /// decode step over two TurboQuant stores computes on the codes; any
    /// other call reads the stores' first `key_count` rows dequantized into
    /// scratch buffers kept per role and shape, so a captured step
    /// allocates nothing, and runs the f32 attention.
    #[allow(clippy::too_many_arguments)]
    fn tq_attention(
        &self,
        q: &CudaDeviceBuffer,
        k: &CudaDeviceBuffer,
        v: &CudaDeviceBuffer,
        [head_count, head_dimension]: [u64; 2],
        value_dimension: Option<u64>,
        kv_head_count: Option<u64>,
        window_size: Option<u64>,
        causal: bool,
        key_count: Option<u64>,
        scale: Option<f32>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[q], "attention query")?;
        let value_width = value_dimension.unwrap_or(head_dimension);
        let kv_heads = kv_head_count.unwrap_or(head_count);
        let (seq_len, q_width) = rows_cols_buf(q)?;
        let kv_rows = k.shape.first().copied().unwrap_or(0);
        if let (1, Some(kt), Some(vt)) = (seq_len, &k.tq, &v.tq)
            && kt.head_dimension == head_dimension
            && vt.head_dimension == value_width
            && kt.heads == kv_heads
            && vt.heads == kv_heads
            && v.shape.first() == Some(&kv_rows)
            && kt.exact_rows == vt.exact_rows
            && kv_heads > 0
            && head_count.is_multiple_of(kv_heads)
            && head_count * head_dimension == q_width
            && window_size != Some(0)
            && (window_size.is_none() || causal)
            && scale.is_none_or(|scale| scale.is_finite() && scale > 0.0)
        {
            let kv_seq_len = key_count.unwrap_or(kv_rows);
            if kv_seq_len == 0 || kv_seq_len > kv_rows {
                return Err(CudaError::new(
                    CudaErrorCode::ShapeUnsupported,
                    format!("key_count {kv_seq_len} must lie in 1..={kv_rows}"),
                ));
            }
            // A history the exact heads hold decodes as f32 stores do, on
            // the heads (`keep-turboquant-kv-head-exact`): a captured step
            // replays only while its history stays in them.
            if let (Some(k_head), Some(v_head)) =
                (kt.head_covering(kv_seq_len), vt.head_covering(kv_seq_len))
            {
                return self.attention_valued(
                    q,
                    k_head,
                    v_head,
                    head_count,
                    head_dimension,
                    value_dimension,
                    kv_head_count,
                    window_size,
                    causal,
                    key_count,
                    scale,
                );
            }
            let step_offset = match key_count {
                Some(_) => self.advancing_offset("key_count", kv_rows - kv_seq_len),
                None => 0,
            };
            return self.tq_attention_decode(
                q,
                [kt, vt],
                [
                    kv_seq_len,
                    head_count,
                    kv_heads,
                    head_dimension,
                    value_width,
                    kv_rows,
                ],
                causal,
                window_size,
                step_offset,
                scale.unwrap_or(0.0),
            );
        }
        // Read through dequantized scratch: the rows attention may read; on
        // the exact heads directly while they hold those rows.
        let rows = key_count.unwrap_or(kv_rows);
        if let (Some(kt), Some(vt)) = (&k.tq, &v.tq)
            && let (Some(k_head), Some(v_head)) = (kt.head_covering(rows), vt.head_covering(rows))
        {
            return self.attention_valued(
                q,
                k_head,
                v_head,
                head_count,
                head_dimension,
                value_dimension,
                kv_head_count,
                window_size,
                causal,
                key_count,
                scale,
            );
        }
        let step_offset = match key_count {
            Some(_) => self.advancing_offset("key_count", kv_rows.saturating_sub(rows)),
            None => 0,
        };
        let mut scratch = self.tq_scratch.lock().unwrap();
        let mut take =
            |role: u8, store: &CudaDeviceBuffer| -> Result<Option<CudaDeviceBuffer>, CudaError> {
                if !store.is_tq() {
                    return Ok(None);
                }
                let key = (role, store.shape.clone());
                let mut slice = match scratch.remove(&key) {
                    Some(slice) => slice,
                    None => {
                        self.forbid_in_capture("allocating a TurboQuant attention scratch buffer")?;
                        let len = store.shape.iter().product::<u64>() as usize;
                        self.stream.alloc_zeros::<f32>(len.max(1))?
                    }
                };
                self.tq_dequantize_into(store, &mut slice, rows, step_offset)?;
                Ok(Some(CudaDeviceBuffer {
                    slice,
                    shape: store.shape.clone(),
                    bf16: None,
                    q8: None,
                    tq: None,
                }))
            };
        let k_values = take(0, k)?;
        let v_values = take(1, v)?;
        drop(take);
        let result = self.attention_valued(
            q,
            k_values.as_ref().unwrap_or(k),
            v_values.as_ref().unwrap_or(v),
            head_count,
            head_dimension,
            value_dimension,
            kv_head_count,
            window_size,
            causal,
            key_count,
            scale,
        );
        for (role, values) in [(0u8, k_values), (1, v_values)] {
            if let Some(values) = values {
                scratch.insert((role, values.shape), values.slice);
            }
        }
        result
    }

    /// Decode attention on the codes of TurboQuant K and V stores, one query
    /// row: `tq_attention_split_kernel`, `attention_combine_kernel`, then
    /// `tq_rotate_back_kernel`.
    #[allow(clippy::too_many_arguments)]
    fn tq_attention_decode(
        &self,
        q: &CudaDeviceBuffer,
        [k, v]: [&CudaTqStore; 2],
        [
            kv_seq_len,
            head_count,
            kv_head_count,
            head_dimension,
            value_dimension,
            kv_rows,
        ]: [u64; 6],
        causal: bool,
        window_size: Option<u64>,
        step_offset: u64,
        score_scale: f32,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        let overflow = || {
            CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "attention launch size overflows a grid dimension",
            )
        };
        let chunks = kv_rows.div_ceil(ATTENTION_CHUNK_KEYS).max(1);
        // A block per KV head serves its grouped query heads when they are
        // few enough (`TQ_MAX_GROUP`): every code is decoded once for them
        // all. Otherwise a block per query head. Heads up to 128 wide take
        // the staged Kernel, its history split over a bounded number of
        // blocks (`speed-up-decode-attention-on-codes`).
        let group = head_count / kv_head_count;
        let grouped = group <= TQ_MAX_GROUP;
        let schedule = self.tq_decode_schedule();
        let staged_shared = (grouped
            && schedule != TqDecodeSchedule::Unstaged
            && head_dimension <= TQ_STAGED_MAX_DIMENSION
            && value_dimension <= TQ_STAGED_MAX_DIMENSION)
            .then(|| {
                // A row of codes: whole words, plus one against bank
                // conflicts.
                let row = |width: u64, bits: u32| (width * u64::from(bits) / 8).div_ceil(4) * 4 + 4;
                (4 * (group * head_dimension + group * value_dimension + 2 * ATTENTION_CHUNK_KEYS)
                    + ATTENTION_CHUNK_KEYS
                        * (row(head_dimension, k.bits) + row(value_dimension, v.bits)))
                    as usize
            });
        let split = self.function(match (staged_shared, grouped) {
            (Some(_), _) => "tq_attention_split_staged_kernel",
            (None, true) => "tq_attention_split_grouped_kernel",
            (None, false) => "tq_attention_split_kernel",
        })?;
        // Partials per query head: the staged Kernel's blocks, or a chunk's.
        let parts = match (staged_shared, schedule) {
            (Some(shared), TqDecodeSchedule::Split) => {
                let keys = match window_size {
                    Some(window) if causal => kv_seq_len.min(window),
                    _ => kv_seq_len,
                };
                self.tq_decode_splits(&split, shared, kv_head_count, keys)
            }
            _ => chunks,
        };
        // The scratch holds as many partials as any step over these stores
        // may write: the split count grows with the history, and a captured
        // step cannot grow the scratch.
        let most_parts = match (staged_shared, schedule) {
            (Some(_), TqDecodeSchedule::Split) => chunks.min(TQ_MAX_SPLITS).max(parts),
            _ => parts,
        };
        let partial_len = usize::try_from(head_count * most_parts * (value_dimension + 2))
            .map_err(|_| overflow())?;
        let mut scratch = self.attention_scratch.lock().unwrap();
        if scratch
            .0
            .as_ref()
            .is_none_or(|slice| slice.len() < partial_len)
        {
            self.forbid_in_capture("growing the attention scratch buffer")?;
            // SAFETY: the split Kernel writes every partial of this launch
            // before the combine Kernel reads it, on this same stream.
            let grown = unsafe { self.stream.alloc::<f32>(partial_len.max(1 << 16)) }?;
            if let Some(outgrown) = scratch.0.replace(grown) {
                scratch.1.push(outgrown);
            }
        }
        let partial = scratch.0.as_mut().expect("allocated above");
        let mut out_dev = self
            .alloc_output(usize::try_from(head_count * value_dimension).map_err(|_| overflow())?)?;
        let causal_flag: i32 = if causal { 1 } else { 0 };
        let window_arg: i64 = window_size.map(|w| w as i64).unwrap_or(-1);
        let query_position_offset = kv_seq_len - 1;
        let mut args = self.stream.launch_builder(&split);
        args.arg(&q.slice)
            .arg(&k.codes)
            .arg(&k.norms)
            .arg(&k.centroids)
            .arg(&k.signs)
            .arg(&k.offsets)
            .arg(&k.bits)
            .arg(&v.codes)
            .arg(&v.norms)
            .arg(&v.centroids)
            .arg(&v.bits)
            .arg(&mut *partial)
            .arg(&kv_seq_len)
            .arg(&head_count)
            .arg(&kv_head_count)
            .arg(&head_dimension)
            .arg(&causal_flag)
            .arg(&window_arg)
            .arg(&query_position_offset)
            .arg(&parts)
            .arg(&step_offset)
            .arg(&score_scale)
            .arg(&value_dimension);
        // Keys below the exact heads' rows are read from them
        // (`keep-turboquant-kv-head-exact`).
        let no_head = 0u64;
        let exact_rows = match (k.exact.as_ref(), v.exact.as_ref()) {
            (Some(k_exact), Some(v_exact)) => {
                args.arg(&k_exact.slice).arg(&v_exact.slice);
                k.exact_rows.min(v.exact_rows)
            }
            _ => {
                args.arg(&no_head).arg(&no_head);
                0
            }
        };
        args.arg(&exact_rows).arg(&v.signs).arg(&v.offsets);
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(parts).map_err(|_| overflow())?,
                u32::try_from(if grouped { kv_head_count } else { head_count })
                    .map_err(|_| overflow())?,
                1,
            ),
            block_dim: (ATTENTION_SPLIT_THREADS, 1, 1),
            shared_mem_bytes: u32::try_from(staged_shared.unwrap_or(0)).map_err(|_| overflow())?,
        };
        unsafe { args.launch(config) }?;
        self.combine_attention_partials(
            &*partial,
            &mut out_dev,
            head_count,
            value_dimension,
            parts,
        )?;
        let rotate = self.function("tq_rotate_back_kernel")?;
        let mut args = self.stream.launch_builder(&rotate);
        let group_size = head_count / kv_head_count;
        args.arg(&mut out_dev)
            .arg(&v.signs)
            .arg(&v.offsets)
            .arg(&value_dimension)
            .arg(&group_size);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(head_count).map_err(|_| overflow())?, 1, 1),
            block_dim: (tq_threads(value_dimension), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![1, head_count * value_dimension],
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// `a + b`, element-wise. `b` may either exactly match `a`'s shape, or
    /// -- mirroring `rmsnorm`'s row-broadcast convention -- be a single row
    /// (`b.shape == [cols]` or `[1, cols]` where `cols` is `a`'s last
    /// dimension) broadcast across every row of `a` (real Qwen2/2.5 QKV
    /// projection bias). Purely additive over the prior same-shape-only
    /// behavior: matches `providers/cpu::add`'s own broadcast extension.
    pub fn add(
        &self,
        a: &CudaDeviceBuffer,
        b: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[a, b], "add")?;
        if a.shape == b.shape {
            let n = a.slice.len() as u64;
            let mut out_dev = self.alloc_output(a.slice.len())?;
            let func = self.function("add_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&a.slice).arg(&b.slice).arg(&mut out_dev).arg(&n);
            unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
            return Ok(CudaDeviceBuffer {
                slice: out_dev,
                shape: a.shape.clone(),
                bf16: None,
                q8: None,
                tq: None,
            });
        }
        let cols = *a.shape.last().ok_or_else(|| {
            CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "add expects at least one dimension",
            )
        })?;
        if b.shape != [cols] && b.shape != [1, cols] {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "add expects b to match a's shape {:?} or broadcast as [{cols}]/[1, {cols}], got {:?}",
                    a.shape, b.shape
                ),
            ));
        }
        let n = a.slice.len() as u64;
        if !n.is_multiple_of(cols) {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("add data length {n} is not divisible by broadcast dimension {cols}"),
            ));
        }
        let mut out_dev = self.alloc_output(a.slice.len())?;
        let func = self.function("bias_add_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&a.slice)
            .arg(&b.slice)
            .arg(&mut out_dev)
            .arg(&cols)
            .arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: a.shape.clone(),
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    pub fn mul(
        &self,
        a: &CudaDeviceBuffer,
        b: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[a, b], "mul")?;
        same_shape_buf(a, b)?;
        let n = a.slice.len() as u64;
        let mut out_dev = self.alloc_output(a.slice.len())?;
        let func = self.function("mul_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&a.slice).arg(&b.slice).arg(&mut out_dev).arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: a.shape.clone(),
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    pub fn residual_add(
        &self,
        input: &CudaDeviceBuffer,
        residual: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[input, residual], "residual-add")?;
        self.add(input, residual)
    }

    /// Row-wise concatenation: `a` and `b` must share the same trailing
    /// dimensions; the result stacks `a`'s rows above `b`'s rows.
    /// KV-history concatenation's exact contract
    /// (`implement-device-resident-multi-step-cuda-decode`): `a` is the
    /// previous steps' historical K/V, `b` is this step's newly computed
    /// K/V, both already Device-resident -- the whole point of this Kernel
    /// is that neither ever needs to leave the device to be combined. No
    /// `.cu` kernel is needed: `out_dev.split_at_mut` produces two
    /// non-overlapping mutable views into one fresh allocation, and
    /// `memcpy_dtod` copies each input directly into its own view.
    pub fn concat(
        &self,
        a: &CudaDeviceBuffer,
        b: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[a, b], "concat")?;
        let Some((&a_rows, a_rest)) = a.shape.split_first() else {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "concat expects a to have at least one dimension",
            ));
        };
        let Some((&b_rows, b_rest)) = b.shape.split_first() else {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "concat expects b to have at least one dimension",
            ));
        };
        if a_rest != b_rest {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "concat expects a and b to share trailing dimensions, got {:?} and {:?}",
                    a.shape, b.shape
                ),
            ));
        }
        let mut out_dev = self.alloc_output(a.slice.len() + b.slice.len())?;
        {
            let (mut first, mut second) = out_dev.split_at_mut(a.slice.len());
            self.stream.memcpy_dtod(&a.slice, &mut first)?;
            self.stream.memcpy_dtod(&b.slice, &mut second)?;
        }
        let mut shape = Vec::with_capacity(a.shape.len());
        shape.push(a_rows + b_rows);
        shape.extend_from_slice(a_rest);
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape,
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    pub fn silu(&self, input: &CudaDeviceBuffer) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[input], "silu")?;
        let n = input.slice.len() as u64;
        let mut out_dev = self.alloc_output(input.slice.len())?;
        let func = self.function("silu_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&input.slice).arg(&mut out_dev).arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: input.shape.clone(),
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// `sigmoid(x)`, element-wise (`activation` with `kind = sigmoid`).
    pub fn sigmoid(&self, input: &CudaDeviceBuffer) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[input], "sigmoid")?;
        let n = input.slice.len() as u64;
        let mut out_dev = self.alloc_output(input.slice.len())?;
        let func = self.function("sigmoid_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&input.slice).arg(&mut out_dev).arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: input.shape.clone(),
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// One f32 element-wise Kernel over `input`, with `extra` scalar
    /// arguments before the element count.
    fn elementwise(
        &self,
        input: &CudaDeviceBuffer,
        kernel: &'static str,
        extra: Option<f32>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[input], kernel)?;
        let n = input.slice.len() as u64;
        let mut out_dev = self.alloc_output(input.slice.len())?;
        let func = self.function(kernel)?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&input.slice).arg(&mut out_dev);
        if let Some(extra) = &extra {
            args.arg(extra);
        }
        args.arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: input.shape.clone(),
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// GELU, tanh approximation (`gelu`, `approximate = "tanh"`).
    pub fn gelu_tanh(&self, input: &CudaDeviceBuffer) -> Result<CudaDeviceBuffer, CudaError> {
        self.elementwise(input, "gelu_tanh_kernel", None)
    }

    /// `x * factor` (`activation` `kind = "scale"`, a scaled `embedding`).
    pub fn scale(
        &self,
        input: &CudaDeviceBuffer,
        factor: f32,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        if !factor.is_finite() {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("scale factor {factor} must be finite"),
            ));
        }
        self.elementwise(input, "scale_kernel", Some(factor))
    }

    /// [`Self::scale`] in place: `buffer` keeps its allocation, so a step
    /// capture that scales a Kernel's own output (a scaled `embedding`)
    /// references no intermediate it then frees.
    pub fn scale_in_place(
        &self,
        buffer: &mut CudaDeviceBuffer,
        factor: f32,
    ) -> Result<(), CudaError> {
        Self::require_f32(&[buffer], "scale")?;
        if !factor.is_finite() {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("scale factor {factor} must be finite"),
            ));
        }
        let n = buffer.slice.len() as u64;
        let input = self.address(buffer);
        let func = self.function("scale_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&input).arg(&mut buffer.slice).arg(&factor).arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(())
    }

    /// `cap * tanh(x / cap)` (`activation` `kind = "softcap"`).
    pub fn softcap(
        &self,
        input: &CudaDeviceBuffer,
        cap: f32,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        if !cap.is_finite() || cap <= 0.0 {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("softcap cap {cap} must be finite and positive"),
            ));
        }
        self.elementwise(input, "softcap_kernel", Some(cap))
    }

    /// `a * b[0]`, `b` a `[1]` operand read on the Device (`mul`'s scalar
    /// broadcast).
    pub fn mul_scalar(
        &self,
        a: &CudaDeviceBuffer,
        b: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[a, b], "mul")?;
        if b.shape.as_slice() != [1] {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("a scalar mul operand has shape [1], got {:?}", b.shape),
            ));
        }
        let n = a.slice.len() as u64;
        let mut out_dev = self.alloc_output(a.slice.len())?;
        let func = self.function("mul_scalar_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&a.slice).arg(&b.slice).arg(&mut out_dev).arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: a.shape.clone(),
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// `out[t, :] = x[t, :] * scale[t]`, `scale` of `[T, 1]` or `[T]`.
    pub fn row_scale(
        &self,
        x: &CudaDeviceBuffer,
        scale: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[x, scale], "row-scale")?;
        let (rows, cols) = rows_cols_buf(x)?;
        if scale.slice.len() as u64 != rows {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "row-scale expects one scale per row of {:?}, got {:?}",
                    x.shape, scale.shape
                ),
            ));
        }
        let n = x.slice.len() as u64;
        let mut out_dev = self.alloc_output(x.slice.len())?;
        let func = self.function("row_scale_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&x.slice)
            .arg(&scale.slice)
            .arg(&mut out_dev)
            .arg(&cols)
            .arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: x.shape.clone(),
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// The most routed experts `moe_route_kernel` handles (`kernels.cu`'s
    /// `MOE_ROUTE_EXPERTS_PER_LANE` lanes of 32).
    const MOE_ROUTE_MAX_EXPERTS: u64 = 8 * 32;

    /// The `moe-experts` Operator (`execute-and-cache-moe-experts` F02):
    /// `magnetar_provider_api::moe_experts`' semantics. Routing
    /// stays on the Device. Packed `Q8_0` experts (transposed storage,
    /// each expert a run of stored rows) are read packed; f32 experts
    /// directly; any other compact storage is expanded for this call.
    pub fn moe_experts(
        &self,
        h: &CudaDeviceBuffer,
        logits: &CudaDeviceBuffer,
        weights: [&CudaDeviceBuffer; 3],
        routing: magnetar_provider_api::MoeRouting,
        local: (u64, u64),
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.moe_experts_with(
            h,
            logits,
            weights,
            routing,
            local,
            CudaMoeOptions::default(),
        )
    }

    /// [`Self::moe_experts`] with its optional forms (`moe-experts`'
    /// `activation` and sixth input, `per_expert_scale`).
    pub fn moe_experts_with(
        &self,
        h: &CudaDeviceBuffer,
        logits: &CudaDeviceBuffer,
        [gate, up, down]: [&CudaDeviceBuffer; 3],
        routing: magnetar_provider_api::MoeRouting,
        (expert_first, local_experts): (u64, u64),
        options: CudaMoeOptions<'_>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[h, logits], "moe-experts' activations")?;
        let shape_error = |reason: String| CudaError::new(CudaErrorCode::ShapeUnsupported, reason);
        let (tokens, hidden) = rows_cols_buf(h)?;
        let (logit_rows, experts) = rows_cols_buf(logits)?;
        let top_k = routing.top_k;
        if experts != routing.experts
            || logit_rows != tokens
            || top_k == 0
            || top_k > experts
            || local_experts == 0
            || expert_first
                .checked_add(local_experts)
                .is_none_or(|end| end > experts)
            || experts > Self::MOE_ROUTE_MAX_EXPERTS
        {
            return Err(shape_error(format!(
                "moe-experts routes top {top_k} of {} experts over [{logit_rows}, {experts}] \
                 logits for {tokens} tokens",
                routing.experts
            )));
        }
        let (gate_rows, gate_cols) = rows_cols_buf(gate)?;
        let (down_rows, down_cols) = rows_cols_buf(down)?;
        if gate_rows != hidden
            || !gate_cols.is_multiple_of(local_experts)
            || up.shape != gate.shape
            || down_cols != local_experts * hidden
            || down_rows != gate_cols / local_experts
        {
            return Err(shape_error(format!(
                "moe-experts expects gate/up [{hidden}, {local_experts} * I] and down [I, \
                 {local_experts} * {hidden}], got {:?}, {:?} and {:?}",
                gate.shape, up.shape, down.shape
            )));
        }
        let inter = gate_cols / local_experts;
        let pairs = tokens * top_k;
        let per_expert_scale = match options.per_expert_scale {
            None => 0u64,
            Some(scale) => {
                Self::require_f32(&[scale], "moe-experts' per_expert_scale")?;
                if scale.slice.len() as u64 != experts {
                    return Err(shape_error(format!(
                        "moe-experts' per_expert_scale has {} values for {experts} experts",
                        scale.slice.len()
                    )));
                }
                self.address(scale)
            }
        };
        let gelu = i32::from(options.gelu_tanh);
        let normalize = i32::from(routing.normalize);
        let route = [tokens, experts, top_k, expert_first, local_experts];

        if tokens > MOE_PERSISTENT_TOKENS
            && self.moe_grouped.load(std::sync::atomic::Ordering::Relaxed)
            && let (Some(gate_q8), Some(up_q8), Some(down_q8)) = (
                packed_moe_q8(gate, hidden),
                packed_moe_q8(up, hidden),
                packed_moe_q8(down, inter),
            )
        {
            return self.moe_q8_grouped(
                h,
                logits,
                [gate_q8, up_q8, down_q8],
                route,
                [hidden, inter],
                normalize,
                per_expert_scale,
                gelu,
            );
        }

        let mut scratch_slot = self.moe_scratch.lock().unwrap();
        let mut temporary = None;
        let scratch: &mut MoeScratch = if tokens <= MOE_PERSISTENT_TOKENS {
            let fits = scratch_slot.0.as_ref().is_some_and(|scratch| {
                scratch.indices.len() as u64 >= pairs && scratch.act.len() as u64 >= pairs * inter
            });
            if !fits {
                self.forbid_in_capture("growing the moe-experts scratch")?;
                let capacity = MOE_PERSISTENT_TOKENS * top_k;
                let grown = MoeScratch {
                    indices: self.stream.alloc_zeros::<i32>(capacity as usize)?,
                    weights: self.stream.alloc_zeros::<f32>(capacity as usize)?,
                    act: self
                        .stream
                        .alloc_zeros::<f32>((capacity * inter) as usize)?,
                };
                if let Some(outgrown) = scratch_slot.0.replace(grown) {
                    scratch_slot.1.push(outgrown);
                }
            }
            scratch_slot.0.as_mut().expect("ensured above")
        } else {
            self.forbid_in_capture("a moe-experts prefill scratch")?;
            temporary.insert(MoeScratch {
                indices: self.stream.alloc_zeros::<i32>(pairs as usize)?,
                weights: self.stream.alloc_zeros::<f32>(pairs as usize)?,
                act: self.stream.alloc_zeros::<f32>((pairs * inter) as usize)?,
            })
        };

        self.moe_route(
            logits,
            &mut scratch.indices,
            &mut scratch.weights,
            route,
            normalize,
            per_expert_scale,
        )?;
        let packed = packed_moe_q8;
        let warps_grid = |cols: u64, rows: u64| LaunchConfig {
            grid_dim: (
                cols.div_ceil(u64::from(Q8_MATMUL_WARPS)) as u32,
                rows as u32,
                1,
            ),
            block_dim: (32, Q8_MATMUL_WARPS, 1),
            shared_mem_bytes: 0,
        };
        if let (Some(gate_q8), Some(up_q8)) = (packed(gate, hidden), packed(up, hidden)) {
            let func = self.function("moe_q8_gated_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&h.slice)
                .arg(&scratch.indices)
                .arg(&gate_q8.quants)
                .arg(&gate_q8.scales)
                .arg(&up_q8.quants)
                .arg(&up_q8.scales)
                .arg(&mut scratch.act)
                .arg(&top_k)
                .arg(&hidden)
                .arg(&inter)
                .arg(&gelu);
            unsafe { args.launch(warps_grid(inter, pairs)) }?;
        } else {
            let (mut gate_f32, mut up_f32) = (None, None);
            let gate = self.as_f32(gate, &mut gate_f32)?;
            let up = self.as_f32(up, &mut up_f32)?;
            let func = self.function("moe_f32_gated_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&h.slice)
                .arg(&scratch.indices)
                .arg(&gate.slice)
                .arg(&up.slice)
                .arg(&mut scratch.act)
                .arg(&pairs)
                .arg(&top_k)
                .arg(&hidden)
                .arg(&inter)
                .arg(&local_experts)
                .arg(&gelu);
            unsafe { args.launch(LaunchConfig::for_num_elems((pairs * inter) as u32)) }?;
        }

        let mut out_dev = self.alloc_output((tokens * hidden) as usize)?;
        if let Some(down_q8) = packed(down, inter) {
            let func = self.function("moe_q8_down_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&scratch.act)
                .arg(&scratch.indices)
                .arg(&scratch.weights)
                .arg(&down_q8.quants)
                .arg(&down_q8.scales)
                .arg(&mut out_dev)
                .arg(&top_k)
                .arg(&hidden)
                .arg(&inter);
            unsafe { args.launch(warps_grid(hidden, tokens)) }?;
        } else {
            let mut down_f32 = None;
            let down = self.as_f32(down, &mut down_f32)?;
            let func = self.function("moe_f32_down_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&scratch.act)
                .arg(&scratch.indices)
                .arg(&scratch.weights)
                .arg(&down.slice)
                .arg(&mut out_dev)
                .arg(&tokens)
                .arg(&top_k)
                .arg(&hidden)
                .arg(&inter)
                .arg(&local_experts);
            unsafe { args.launch(LaunchConfig::for_num_elems((tokens * hidden) as u32)) }?;
        }
        drop(temporary);
        drop(scratch_slot);
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![tokens, hidden],
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// `moe_route_kernel`: each token's chosen local experts and weights
    /// (`route` = tokens, experts, top_k, first local expert, local
    /// experts).
    fn moe_route(
        &self,
        logits: &CudaDeviceBuffer,
        indices: &mut CudaSlice<i32>,
        weights: &mut CudaSlice<f32>,
        [tokens, experts, top_k, expert_first, local_experts]: [u64; 5],
        normalize: i32,
        per_expert_scale: u64,
    ) -> Result<(), CudaError> {
        let func = self.function("moe_route_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&logits.slice)
            .arg(indices)
            .arg(weights)
            .arg(&tokens)
            .arg(&experts)
            .arg(&top_k)
            .arg(&normalize)
            .arg(&expert_first)
            .arg(&local_experts)
            .arg(&per_expert_scale);
        // One warp per token, four tokens per block.
        let route_config = LaunchConfig {
            grid_dim: (tokens.div_ceil(4) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(route_config) }?;
        Ok(())
    }

    /// A `moe-experts` prefill over packed `Q8_0` experts, its pairs grouped
    /// by expert: each block of `moe_q8_gated_grouped_kernel` and
    /// `moe_q8_down_grouped_kernel` reads one expert's weights once for up
    /// to [`MOE_GROUP_ROWS`] of its pairs. Tokens are served
    /// [`MOE_GROUP_CHUNK_TOKENS`] at a time. Every value is the per-pair
    /// Kernels' own (same lanes, order and reductions).
    #[allow(clippy::too_many_arguments)]
    fn moe_q8_grouped(
        &self,
        h: &CudaDeviceBuffer,
        logits: &CudaDeviceBuffer,
        [gate, up, down]: [&CudaQ8Weight; 3],
        route: [u64; 5],
        [hidden, inter]: [u64; 2],
        normalize: i32,
        per_expert_scale: u64,
        gelu: i32,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.forbid_in_capture("a moe-experts prefill scratch")?;
        let [tokens, _, top_k, _, local_experts] = route;
        let pairs = tokens * top_k;
        let mut indices = self.stream.alloc_zeros::<i32>(pairs as usize)?;
        let mut weights = self.stream.alloc_zeros::<f32>(pairs as usize)?;
        self.moe_route(
            logits,
            &mut indices,
            &mut weights,
            route,
            normalize,
            per_expert_scale,
        )?;
        let chunk_tokens = MOE_GROUP_CHUNK_TOKENS.min(tokens);
        let chunk_pairs = chunk_tokens * top_k;
        let slots = (local_experts + 1) as usize;
        let mut counts = self.stream.alloc_zeros::<u32>(slots)?;
        let mut offsets = self.stream.alloc_zeros::<u32>(slots)?;
        let mut tiles = self.stream.alloc_zeros::<u32>(slots)?;
        let mut cursor = self.stream.alloc_zeros::<u32>(slots)?;
        let mut order = self.stream.alloc_zeros::<u32>(chunk_pairs as usize)?;
        let mut act = self
            .stream
            .alloc_zeros::<f32>((chunk_pairs * inter) as usize)?;
        let mut partial = self
            .stream
            .alloc_zeros::<f32>((chunk_pairs * hidden) as usize)?;
        let mut out_dev = self.alloc_output((tokens * hidden) as usize)?;
        let per_pair = |count: u64| LaunchConfig::for_num_elems(count as u32);
        // Every expert of the chunk may leave a partial tile.
        let tiles_bound = |pairs: u64| pairs.div_ceil(MOE_GROUP_ROWS) + local_experts;
        let warps_grid = |cols: u64, blocks: u64| LaunchConfig {
            grid_dim: (
                cols.div_ceil(u64::from(Q8_MATMUL_WARPS)) as u32,
                blocks as u32,
                1,
            ),
            block_dim: (32, Q8_MATMUL_WARPS, 1),
            shared_mem_bytes: 0,
        };
        let mut token_first = 0;
        while token_first < tokens {
            let chunk = chunk_tokens.min(tokens - token_first);
            let (pair_first, chunk_pairs) = (token_first * top_k, chunk * top_k);
            self.stream.memset_zeros(&mut counts)?;
            let func = self.function("moe_group_count_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&indices)
                .arg(&mut counts)
                .arg(&pair_first)
                .arg(&chunk_pairs);
            unsafe { args.launch(per_pair(chunk_pairs)) }?;
            let func = self.function("moe_group_offsets_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&counts)
                .arg(&mut offsets)
                .arg(&mut tiles)
                .arg(&mut cursor)
                .arg(&local_experts);
            unsafe { args.launch(LaunchConfig::for_num_elems(1)) }?;
            let func = self.function("moe_group_scatter_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&indices)
                .arg(&mut cursor)
                .arg(&mut order)
                .arg(&pair_first)
                .arg(&chunk_pairs);
            unsafe { args.launch(per_pair(chunk_pairs)) }?;

            let func = self.function("moe_q8_gated_grouped_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&h.slice)
                .arg(&order)
                .arg(&offsets)
                .arg(&tiles)
                .arg(&local_experts)
                .arg(&gate.quants)
                .arg(&gate.scales)
                .arg(&up.quants)
                .arg(&up.scales)
                .arg(&mut act)
                .arg(&pair_first)
                .arg(&top_k)
                .arg(&hidden)
                .arg(&inter)
                .arg(&gelu);
            unsafe { args.launch(warps_grid(inter, tiles_bound(chunk_pairs))) }?;
            let func = self.function("moe_q8_down_grouped_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&act)
                .arg(&order)
                .arg(&offsets)
                .arg(&tiles)
                .arg(&local_experts)
                .arg(&down.quants)
                .arg(&down.scales)
                .arg(&mut partial)
                .arg(&pair_first)
                .arg(&hidden)
                .arg(&inter);
            unsafe { args.launch(warps_grid(hidden, tiles_bound(chunk_pairs))) }?;
            let func = self.function("moe_down_combine_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&partial)
                .arg(&indices)
                .arg(&weights)
                .arg(&mut out_dev)
                .arg(&token_first)
                .arg(&chunk)
                .arg(&top_k)
                .arg(&hidden);
            unsafe { args.launch(per_pair(chunk * hidden)) }?;
            token_first += chunk;
        }
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![tokens, hidden],
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// `buffer` as plain f32: itself, or its expansion for this call,
    /// kept in `expanded`.
    fn as_f32<'b>(
        &self,
        buffer: &'b CudaDeviceBuffer,
        expanded: &'b mut Option<CudaDeviceBuffer>,
    ) -> Result<&'b CudaDeviceBuffer, CudaError> {
        if !buffer.is_compact() {
            return Ok(buffer);
        }
        Ok(expanded.insert(self.expand(buffer)?))
    }

    pub fn embedding_lookup(
        &self,
        table: &CudaDeviceBuffer,
        ids: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        let (vocab, dim) = rows_cols_buf(table)?;
        Self::require_f32(&[ids], "the embedding lookup's ids")?;
        // A packed table read row by row must be stored untransposed.
        let context = |step: &'static str| {
            let packed = table.is_q8();
            move |error: CudaError| {
                CudaError::new(
                    CudaErrorCode::ExecutionFailed,
                    format!(
                        "embedding lookup {step} ({} ids, table {vocab}x{dim}, packed {packed}): \
                         {error}",
                        ids.slice.len()
                    ),
                )
            }
        };
        if table.q8.as_ref().is_some_and(|q8| q8.transposed) {
            let dense = self
                .dequantize_q8(table)
                .map_err(context("dequantizing a transposed table"))?;
            return self.embedding_lookup(&dense, ids);
        }
        // A packed table in another format than `Q8_0`: its kernel
        // dequantizes the looked-up rows only.
        let packed_rows = table
            .q8
            .as_ref()
            .is_some_and(|q8| q8.format != PackedWeightFormat::Q8_0);
        let table16 = self.bf16_address(table);
        let (table_q8, table_q8_scales) = match &table.q8 {
            Some(q8) => (
                q8.quants.device_ptr(&self.stream).0,
                q8.scales.device_ptr(&self.stream).0,
            ),
            None => (0, 0),
        };
        let num_ids = ids.slice.len() as u64;
        // Rows of invalid ids stay unwritten, but those fail the lookup.
        let out_dev = if packed_rows {
            None
        } else {
            Some(
                self.alloc_output(ids.slice.len() * dim as usize)
                    .map_err(context("output allocation"))?,
            )
        };
        let mut flag_slot = self.embedding_flag.lock().unwrap();
        let mut invalid_id_flag = match flag_slot.take() {
            Some(flag) => flag,
            None => {
                self.forbid_in_capture("a device allocation")?;
                self.stream
                    .alloc_zeros::<i32>(1)
                    .map_err(CudaError::from)
                    .map_err(context("flag allocation"))?
            }
        };
        // A captured step resets the flag once, at its start.
        if !self.is_capturing() {
            self.stream
                .memset_zeros(&mut invalid_id_flag)
                .map_err(CudaError::from)
                .map_err(context("flag reset"))?;
        }
        let launched = match out_dev {
            None => {
                let rows = PackedRows {
                    ids: ids.slice.device_ptr(&self.stream).0,
                    count: num_ids,
                    invalid_id_flag: invalid_id_flag.device_ptr_mut(&self.stream).0,
                };
                self.dequantize_packed(table, Some(rows))
                    .map(|rows| rows.slice)
                    .map_err(context("of packed rows"))
            }
            Some(mut out_dev) => {
                self.function("embedding_lookup_kernel")
                    .map_err(context("function load"))
                    .and_then(|func| {
                        let mut args = self.stream.launch_builder(&func);
                        args.arg(&table.slice)
                            .arg(&ids.slice)
                            .arg(&mut out_dev)
                            .arg(&dim)
                            .arg(&num_ids)
                            .arg(&vocab)
                            .arg(&mut invalid_id_flag)
                            .arg(&table16)
                            .arg(&table_q8)
                            .arg(&table_q8_scales);
                        unsafe { args.launch(LaunchConfig::for_num_elems(num_ids as u32)) }
                            .map_err(|error| {
                                CudaError::new(
                                    CudaErrorCode::ExecutionFailed,
                                    format!(
                                        "embedding_lookup_kernel launch ({num_ids} ids, table \
                                     {vocab}x{dim}, packed {}): {error}",
                                        table.is_q8()
                                    ),
                                )
                            })?;
                        Ok(out_dev)
                    })
            }
        };
        *flag_slot = Some(invalid_id_flag);
        drop(flag_slot);
        let out_dev = launched?;
        // A captured lookup's flag is checked once the captured step has
        // run (`Self::check_embedding_flag`), and every replay checks its
        // ids on the host.
        if let Some(capture) = self.capture.lock().unwrap().as_mut() {
            capture.has_embedding = true;
            return Ok(CudaDeviceBuffer {
                slice: out_dev,
                shape: vec![num_ids, dim],
                bf16: None,
                q8: None,
                tq: None,
            });
        }
        // Data-dependent failure (an out-of-range or non-integer token id)
        // can only be observed by downloading this one-element flag -- the
        // same device-computed-flag pattern `softmax_rows` already uses
        // below, not a blanket per-kernel host round-trip.
        self.check_embedding_flag().map_err(|error| {
            CudaError::new(
                CudaErrorCode::ExecutionFailed,
                format!(
                    "embedding lookup ({num_ids} ids, table {vocab}x{dim}, packed {}) \
                     completion: {error}",
                    table.is_q8()
                ),
            )
        })?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![num_ids, dim],
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// Fails when the last embedding lookup met an invalid id. Waits for
    /// the Device.
    pub fn check_embedding_flag(&self) -> Result<(), CudaError> {
        let flag = match self.embedding_flag.lock().unwrap().as_ref() {
            Some(flag) => self.stream.clone_dtoh(flag)?,
            None => return Ok(()),
        };
        self.stream.synchronize()?;
        if flag[0] != 0 {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "embedding lookup id is not a valid in-range non-negative integer",
            ));
        }
        Ok(())
    }

    pub fn rmsnorm(
        &self,
        input: &CudaDeviceBuffer,
        weight: &CudaDeviceBuffer,
        epsilon: f32,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.rmsnorm_with(input, Some(weight), epsilon, None)
    }

    /// [`Self::rmsnorm`] with its optional forms: no weight (scale 1), and
    /// per-head normalization (`heads = (head_count, head_dimension)`):
    /// each row's `head_count` blocks normalized alone, with one
    /// `[head_dimension]` weight. Rows are contiguous, so the heads are the
    /// rows of a `[rows * head_count, head_dimension]` view.
    pub fn rmsnorm_with(
        &self,
        input: &CudaDeviceBuffer,
        weight: Option<&CudaDeviceBuffer>,
        epsilon: f32,
        heads: Option<(u64, u64)>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[input], "rmsnorm")?;
        if let Some(weight) = weight {
            Self::require_f32(&[weight], "rmsnorm")?;
        }
        let row_width = *input.shape.last().ok_or_else(|| {
            CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "RMSNorm expects at least one dimension",
            )
        })?;
        let cols = match heads {
            None => row_width,
            Some((count, dimension)) => {
                if count == 0 || dimension == 0 || count.checked_mul(dimension) != Some(row_width) {
                    return Err(CudaError::new(
                        CudaErrorCode::ShapeUnsupported,
                        format!(
                            "RMSNorm's {count} heads of {dimension} do not tile rows of {row_width}"
                        ),
                    ));
                }
                dimension
            }
        };
        if cols == 0 || !input.slice.len().is_multiple_of(cols as usize) {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "RMSNorm data length {} is not divisible by hidden dimension {cols}",
                    input.slice.len()
                ),
            ));
        }
        let rows = (input.slice.len() / cols as usize) as u64;
        let weight_row_stride = match weight {
            None => 0u64,
            Some(weight) if weight.shape == [cols] || weight.shape == [1, cols] => 0u64,
            Some(weight)
                if heads.is_none()
                    && (weight.shape == input.shape || weight.shape == [rows, cols]) =>
            {
                cols
            }
            Some(weight) => {
                return Err(CudaError::new(
                    CudaErrorCode::ShapeUnsupported,
                    format!(
                        "RMSNorm weight shape must be [{cols}], [1, {cols}], input shape {:?}, or [{rows}, {cols}], got {:?}",
                        input.shape, weight.shape
                    ),
                ));
            }
        };
        if epsilon <= 0.0 {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "RMSNorm epsilon must be positive",
            ));
        }
        let mut out_dev = self.alloc_output(input.slice.len())?;
        // A null weight is a scale-less RMSNorm.
        let weight_address = weight.map_or(0u64, |weight| self.address(weight));
        let func = self.function("rmsnorm_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&input.slice)
            .arg(&weight_address)
            .arg(&mut out_dev)
            .arg(&rows)
            .arg(&cols)
            .arg(&weight_row_stride)
            .arg(&epsilon);
        let config = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (RMSNORM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: input.shape.clone(),
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// Each row divides into `head_count` equal-width blocks of
    /// `head_width = cols / head_count` columns; within each block
    /// independently, the first `dimension` columns (`dimension <=
    /// head_width`, partial RoPE is legal) are rotated in consecutive
    /// pairs. Columns outside a rotated range are left unchanged --
    /// `out_dev` is seeded as a device-to-device copy of `input`, not a
    /// zero-allocation, so the kernel only needs to write the columns it
    /// actually rotates. `head_count = 1` reproduces this Kernel's
    /// pre-`make-first-native-cuda-hot-path-device-resident` single-block
    /// behavior exactly (`head_width` becomes `cols`).
    pub fn rope(
        &self,
        input: &CudaDeviceBuffer,
        base: f32,
        scale: f32,
        dimension: u64,
        position_offset: u64,
        head_count: u64,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.rope_with_pairing(
            input,
            base,
            scale,
            dimension,
            position_offset,
            head_count,
            CudaRopePairing::Interleaved,
        )
    }

    /// [`Self::rope`] with an explicit element pairing
    /// (`fix-rope-rotation-pairing`).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_with_pairing(
        &self,
        input: &CudaDeviceBuffer,
        base: f32,
        scale: f32,
        dimension: u64,
        position_offset: u64,
        head_count: u64,
        pairing: CudaRopePairing,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.rope_with_frequency_scaling(
            input,
            base,
            scale,
            dimension,
            position_offset,
            head_count,
            pairing,
            None,
        )
    }

    /// [`Self::rope_with_pairing`] with optional banded per-frequency
    /// scaling (`frequency_scaling = "banded"`).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_with_frequency_scaling(
        &self,
        input: &CudaDeviceBuffer,
        base: f32,
        scale: f32,
        dimension: u64,
        position_offset: u64,
        head_count: u64,
        pairing: CudaRopePairing,
        frequency_scaling: Option<&RopeBandedFrequencyScaling>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.rope_partial(
            input,
            base,
            scale,
            dimension,
            position_offset,
            head_count,
            pairing,
            frequency_scaling,
            None,
        )
    }

    /// [`Self::rope_with_frequency_scaling`] rotating only the first
    /// `rotated_pairs` pairs of each head's `dimension / 2`
    /// (`rope.partial_rotary_factor`): the others keep frequency zero and
    /// pass through. The frequencies still divide by the whole `dimension`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_partial(
        &self,
        input: &CudaDeviceBuffer,
        base: f32,
        scale: f32,
        dimension: u64,
        position_offset: u64,
        head_count: u64,
        pairing: CudaRopePairing,
        frequency_scaling: Option<&RopeBandedFrequencyScaling>,
        rotated_pairs: Option<u64>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.rope_scaled(
            input,
            base,
            scale,
            dimension,
            position_offset,
            head_count,
            pairing,
            frequency_scaling,
            None,
            rotated_pairs,
        )
    }

    /// [`Self::rope_partial`] with YaRN instead of the banded scaling
    /// (`frequency_scaling = "yarn"`, `express-latent-attention-forms`):
    /// each rotated pair's frequency blended along the correction range,
    /// and its `cos` and `sin` multiplied by `attention_factor`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_scaled(
        &self,
        input: &CudaDeviceBuffer,
        base: f32,
        scale: f32,
        dimension: u64,
        position_offset: u64,
        head_count: u64,
        pairing: CudaRopePairing,
        frequency_scaling: Option<&RopeBandedFrequencyScaling>,
        yarn: Option<&RopeYarnFrequencyScaling>,
        rotated_pairs: Option<u64>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[input], "rope")?;
        if frequency_scaling.is_some() && yarn.is_some() {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "RoPE takes one frequency scaling, not banded and YaRN together",
            ));
        }
        if let Some(scaling) = yarn {
            scaling
                .validate()
                .map_err(|reason| CudaError::new(CudaErrorCode::ShapeUnsupported, reason))?;
        }
        if let Some(scaling) = frequency_scaling {
            scaling
                .validate()
                .map_err(|reason| CudaError::new(CudaErrorCode::ShapeUnsupported, reason))?;
        }
        // `0` disables the scaling in the Kernel.
        let (banded_factor, low_freq_factor, high_freq_factor, original_context_length) =
            frequency_scaling.map_or((0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32), |scaling| {
                (
                    scaling.factor as f32,
                    scaling.low_freq_factor as f32,
                    scaling.high_freq_factor as f32,
                    scaling.original_context_length as f32,
                )
            });
        let (rows, cols) = rows_cols_buf(input)?;
        if head_count == 0 || !cols.is_multiple_of(head_count) {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "RoPE head_count {head_count} must be positive and evenly divide the row width {cols}"
                ),
            ));
        }
        let head_width = cols / head_count;
        if dimension == 0 || !dimension.is_multiple_of(2) || dimension > head_width {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "RoPE dimension {dimension} must be positive, even, and at most the head width {head_width}"
                ),
            ));
        }
        if !base.is_finite() || base <= 0.0 {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "RoPE base must be finite and positive",
            ));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "RoPE scale must be finite and positive",
            ));
        }
        let half = dimension / 2;
        let rotated_pairs = rotated_pairs.unwrap_or(half).min(half);
        // `0` disables YaRN in the Kernel; the range is computed once here,
        // in `f64`, as every Kernel takes it.
        let (yarn_factor, yarn_low, yarn_high, amplitude) =
            yarn.map_or((0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32), |scaling| {
                let (low, high) = scaling.correction_range(dimension, base);
                (
                    scaling.factor as f32,
                    low,
                    high,
                    scaling.attention_factor as f32,
                )
            });
        let half_split: u32 = match pairing {
            CudaRopePairing::Interleaved => 0,
            CudaRopePairing::HalfSplit => 1,
        };
        let work_items = rope_launch_work_items(rows, head_count, half)?;
        // Each head's `dimension` leading columns are rotated pair by pair,
        // the rest pass through unchanged. When the rotation covers the
        // whole head (Qwen2, Llama) the Kernel writes every element, so the
        // output needs no copy of the input first.
        let mut out_dev = if dimension == head_width {
            self.alloc_output(input.slice.len())?
        } else {
            self.forbid_in_capture("a buffer copy")?;
            self.stream.clone_dtod(&input.slice)?
        };
        let step_offset = self.advancing_offset("position_offset", u64::MAX);
        let func = self.function("rope_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&input.slice)
            .arg(&mut out_dev)
            .arg(&rows)
            .arg(&cols)
            .arg(&half)
            .arg(&base)
            .arg(&scale)
            .arg(&dimension)
            .arg(&position_offset)
            .arg(&head_count)
            .arg(&half_split)
            .arg(&step_offset)
            .arg(&banded_factor)
            .arg(&low_freq_factor)
            .arg(&high_freq_factor)
            .arg(&original_context_length)
            .arg(&rotated_pairs)
            .arg(&yarn_factor)
            .arg(&yarn_low)
            .arg(&yarn_high)
            .arg(&amplitude);
        unsafe { args.launch(LaunchConfig::for_num_elems(work_items)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: input.shape.clone(),
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    pub fn softmax_rows(&self, input: &CudaDeviceBuffer) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[input], "softmax")?;
        let (rows, cols) = rows_cols_buf(input)?;
        self.forbid_in_capture("softmax")?;
        let mut out_dev = self.stream.alloc_zeros::<f32>(input.slice.len())?;
        let mut flag_dev = self.stream.alloc_zeros::<i32>(1)?;
        let func = self.function("softmax_rows_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&input.slice)
            .arg(&mut out_dev)
            .arg(&rows)
            .arg(&cols)
            .arg(&mut flag_dev);
        unsafe { args.launch(LaunchConfig::for_num_elems(rows as u32)) }?;
        let flag = self.stream.clone_dtoh(&flag_dev)?;
        self.stream.synchronize()?;
        if flag[0] != 0 {
            return Err(CudaError::new(
                CudaErrorCode::ExecutionFailed,
                "softmax has a row with no finite entry to normalize",
            ));
        }
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: input.shape.clone(),
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    pub fn matmul(
        &self,
        a: &CudaDeviceBuffer,
        b: &CudaDeviceBuffer,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.matmul_with_addend(a, b, None, transpose_a, transpose_b)
    }

    /// `matmul(a, b) + bias`, `bias` a `[n]` (or `[1, n]`) row added to
    /// every output row: the Runtime's fused `matmul-bias` Operator,
    /// bit-identical to `matmul` followed by `add`.
    pub fn matmul_bias(
        &self,
        a: &CudaDeviceBuffer,
        b: &CudaDeviceBuffer,
        bias: &CudaDeviceBuffer,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.matmul_with_addend(
            a,
            b,
            Some(MatmulAddend::Row(bias)),
            transpose_a,
            transpose_b,
        )
    }

    /// `residual + matmul(a, b)`, `residual` the `[m, n]` output's shape: the
    /// Runtime's fused `matmul-residual` Operator, bit-identical to `matmul`
    /// followed by `residual_add`.
    pub fn matmul_residual(
        &self,
        a: &CudaDeviceBuffer,
        b: &CudaDeviceBuffer,
        residual: &CudaDeviceBuffer,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.matmul_with_addend(
            a,
            b,
            Some(MatmulAddend::Full(residual)),
            transpose_a,
            transpose_b,
        )
    }

    /// The greedy next-token column of `logits`' row `row`
    /// (`ProviderExecutionApi::select_greedy_column`): `None` when no
    /// column is eligible. Only the 8-byte answer is downloaded.
    pub fn greedy_select(
        &self,
        logits: &CudaDeviceBuffer,
        row: u64,
        excluded: &[std::ops::Range<u64>],
        temperature: f32,
    ) -> Result<Option<u64>, CudaError> {
        Self::require_f32(&[logits], "greedy selection's logits")?;
        let cols = *logits.shape.last().ok_or_else(|| {
            CudaError::new(CudaErrorCode::ShapeUnsupported, "logits have no columns")
        })?;
        let rows = logits.slice.len() as u64 / cols.max(1);
        if cols == 0 || row >= rows || cols > u64::from(u32::MAX) {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("no logits row {row} of {rows} rows of {cols} columns"),
            ));
        }
        let ranges: Vec<u64> = excluded
            .iter()
            .flat_map(|range| [range.start, range.end])
            .collect();
        // A device buffer cannot be empty; one unused pair stands in.
        let ranges: &[u64] = if ranges.is_empty() { &[0, 0] } else { &ranges };
        let ranges_dev = self.stream.clone_htod(ranges)?;
        let range_count = excluded.len() as u32;
        let best = self.stream.alloc_zeros::<u64>(1)?;
        let row_start = logits
            .slice
            .slice((row * cols) as usize..((row + 1) * cols) as usize);
        let func = self.function("greedy_select_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&row_start)
            .arg(&cols)
            .arg(&ranges_dev)
            .arg(&range_count)
            .arg(&temperature)
            .arg(&best);
        let config = LaunchConfig {
            grid_dim: (cols.div_ceil(256).min(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        let key = self.stream.clone_dtoh(&best)?[0];
        self.download_count.fetch_add(1, Ordering::Relaxed);
        Ok((key != 0).then(|| 0xFFFF_FFFF - (key & 0xFFFF_FFFF)))
    }

    /// `silu(gate) * up`, elementwise: the Runtime's fused `silu-mul`
    /// Operator, bit-identical to `silu` followed by `mul`.
    pub fn silu_mul(
        &self,
        gate: &CudaDeviceBuffer,
        up: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[gate, up], "silu-mul")?;
        if gate.shape != up.shape {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "silu-mul expects gate and up to share a shape, got {:?} and {:?}",
                    gate.shape, up.shape
                ),
            ));
        }
        let n = gate.slice.len() as u64;
        let mut out_dev = self.alloc_output(gate.slice.len())?;
        let func = self.function("silu_mul_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&gate.slice)
            .arg(&up.slice)
            .arg(&mut out_dev)
            .arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: gate.shape.clone(),
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// `silu(a x gate) * (a x up)`: the Runtime's fused `gated-matmul`,
    /// bit-identical to both matmuls followed by [`Self::silu_mul`]. One
    /// launch reads both weights when they are not transposed; otherwise
    /// it runs exactly those three operations.
    pub fn gated_matmul(
        &self,
        a: &CudaDeviceBuffer,
        gate: &CudaDeviceBuffer,
        up: &CudaDeviceBuffer,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        if gate.shape != up.shape {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "gated-matmul expects gate and up weights of one shape, got {:?} and {:?}",
                    gate.shape, up.shape
                ),
            ));
        }
        let (a_rows, a_cols) = rows_cols_buf(a)?;
        let (b_rows, b_cols) = rows_cols_buf(gate)?;
        let (m, k) = if transpose_a {
            (a_cols, a_rows)
        } else {
            (a_rows, a_cols)
        };
        // Packed weights read packed against a few contiguous rows: both
        // projections in one launch, with no intermediate to allocate (a
        // captured decode step allocates nothing).
        if let (Some(gate_q8), Some(up_q8)) = (&gate.q8, &up.q8)
            && gate_q8.format == PackedWeightFormat::Q8_0
            && up_q8.format == PackedWeightFormat::Q8_0
            && gate_q8.transposed
            && up_q8.transposed
            && !transpose_a
            && !transpose_b
            && m <= Q8_MATMUL_MAX_ROWS
            && k == b_rows
        {
            Self::require_f32(&[a], "gated-matmul's input")?;
            return self.q8_gated_matmul_rows(a, gate_q8, up_q8, m, k, b_cols);
        }
        if let (Some(gate_iq), Some(up_iq)) = (&gate.q8, &up.q8)
            && let Some(format) = iq_format(gate_iq.format)
            && up_iq.format == gate_iq.format
            && gate_iq.transposed
            && up_iq.transposed
            && !transpose_a
            && !transpose_b
            && m <= Q8_MATMUL_MAX_ROWS
            && k == b_rows
        {
            Self::require_f32(&[a], "gated-matmul's input")?;
            let int8 = self.quantize_activations(a, m, k)?;
            if let Some(scratch) = int8.as_ref().and_then(|guard| guard.as_ref()) {
                let mut out_dev = self.alloc_output((m * b_cols) as usize)?;
                let func = self.function("iq_gated_matmul_rows_int8_kernel")?;
                let skipped_rows = self.padding_rows(m);
                let mut args = self.stream.launch_builder(&func);
                args.arg(&format)
                    .arg(&scratch.values)
                    .arg(&scratch.scales)
                    .arg(&gate_iq.quants)
                    .arg(&gate_iq.scales)
                    .arg(&up_iq.quants)
                    .arg(&up_iq.scales)
                    .arg(&mut out_dev)
                    .arg(&m)
                    .arg(&k)
                    .arg(&b_cols);
                args.arg(&skipped_rows);
                let config = LaunchConfig {
                    grid_dim: (
                        b_cols.div_ceil(u64::from(Q8_MATMUL_WARPS)) as u32,
                        m.div_ceil(Q8_MATMUL_ROWS) as u32,
                        1,
                    ),
                    block_dim: (32, Q8_MATMUL_WARPS, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { args.launch(config) }?;
                return Ok(CudaDeviceBuffer {
                    slice: out_dev,
                    shape: vec![m, b_cols],
                    bf16: None,
                    q8: None,
                    tq: None,
                });
            }
        }
        if let (Some(gate_iq), Some(up_iq)) = (&gate.q8, &up.q8)
            && matches!(
                gate_iq.format,
                PackedWeightFormat::Iq4Nl | PackedWeightFormat::Iq4Xs | PackedWeightFormat::Q4_0
            )
            && up_iq.format == gate_iq.format
            && gate_iq.transposed
            && up_iq.transposed
            && !transpose_a
            && !transpose_b
            && m <= Q8_MATMUL_MAX_ROWS
            && k == b_rows
        {
            Self::require_f32(&[a], "gated-matmul's input")?;
            let int8 = self.quantize_activations(a, m, k)?;
            if let Some(scratch) = int8.as_ref().and_then(|guard| guard.as_ref()) {
                let mut out_dev = self.alloc_output((m * b_cols) as usize)?;
                let format = iq4_format(gate_iq.format);
                let gate_sub = gate_iq.quants.slice(iq4_subscale_offset(gate_iq)..);
                let up_sub = up_iq.quants.slice(iq4_subscale_offset(up_iq)..);
                let func = self.function("iq4_gated_matmul_rows_int8_kernel")?;
                let skipped_rows = self.padding_rows(m);
                let mut args = self.stream.launch_builder(&func);
                args.arg(&format)
                    .arg(&scratch.values)
                    .arg(&scratch.scales)
                    .arg(&gate_iq.quants)
                    .arg(&gate_sub)
                    .arg(&gate_iq.scales)
                    .arg(&up_iq.quants)
                    .arg(&up_sub)
                    .arg(&up_iq.scales)
                    .arg(&mut out_dev)
                    .arg(&m)
                    .arg(&k)
                    .arg(&b_cols);
                args.arg(&skipped_rows);
                let config = LaunchConfig {
                    grid_dim: (
                        b_cols.div_ceil(u64::from(Q8_MATMUL_WARPS)) as u32,
                        m.div_ceil(Q8_MATMUL_ROWS) as u32,
                        1,
                    ),
                    block_dim: (32, Q8_MATMUL_WARPS, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { args.launch(config) }?;
                return Ok(CudaDeviceBuffer {
                    slice: out_dev,
                    shape: vec![m, b_cols],
                    bf16: None,
                    q8: None,
                    tq: None,
                });
            }
        }
        if let (Some(gate_low), Some(up_low)) = (&gate.q8, &up.q8)
            && matches!(
                gate_low.format,
                PackedWeightFormat::Q2K | PackedWeightFormat::Q3K
            )
            && up_low.format == gate_low.format
            && gate_low.transposed
            && up_low.transposed
            && !transpose_a
            && !transpose_b
            && m <= Q8_MATMUL_MAX_ROWS
            && k == b_rows
        {
            Self::require_f32(&[a], "gated-matmul's input")?;
            let int8 = self.quantize_activations(a, m, k)?;
            if let Some(scratch) = int8.as_ref().and_then(|guard| guard.as_ref()) {
                let mut out_dev = self.alloc_output((m * b_cols) as usize)?;
                let format = low_bit_format(gate_low.format);
                let func = self.function("lowk_gated_matmul_rows_int8_kernel")?;
                let skipped_rows = self.padding_rows(m);
                let mut args = self.stream.launch_builder(&func);
                args.arg(&format)
                    .arg(&scratch.values)
                    .arg(&scratch.scales)
                    .arg(&gate_low.quants)
                    .arg(&gate_low.scales)
                    .arg(&up_low.quants)
                    .arg(&up_low.scales)
                    .arg(&mut out_dev)
                    .arg(&m)
                    .arg(&k)
                    .arg(&b_cols);
                args.arg(&skipped_rows);
                let config = LaunchConfig {
                    grid_dim: (
                        b_cols.div_ceil(u64::from(Q8_MATMUL_WARPS)) as u32,
                        m.div_ceil(Q8_MATMUL_ROWS) as u32,
                        1,
                    ),
                    block_dim: (32, Q8_MATMUL_WARPS, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { args.launch(config) }?;
                return Ok(CudaDeviceBuffer {
                    slice: out_dev,
                    shape: vec![m, b_cols],
                    bf16: None,
                    q8: None,
                    tq: None,
                });
            }
        }
        if let (Some(gate_q4k), Some(up_q4k)) = (&gate.q8, &up.q8)
            && gate_q4k.format == PackedWeightFormat::Q4K
            && up_q4k.format == PackedWeightFormat::Q4K
            && gate_q4k.transposed
            && up_q4k.transposed
            && !transpose_a
            && !transpose_b
            && m <= Q8_MATMUL_MAX_ROWS
            && k == b_rows
        {
            Self::require_f32(&[a], "gated-matmul's input")?;
            let mut out_dev = self.alloc_output((m * b_cols) as usize)?;
            let int8 = self.quantize_activations(a, m, k)?;
            let int8 = int8.as_ref().and_then(|guard| guard.as_ref());
            let func = self.function(if int8.is_some() {
                "q4k_gated_matmul_rows_int8_kernel"
            } else {
                "q4k_gated_matmul_rows_kernel"
            })?;
            let skipped_rows = self.padding_rows(m);
            let mut args = self.stream.launch_builder(&func);
            match int8 {
                Some(scratch) => args.arg(&scratch.values).arg(&scratch.scales),
                None => args.arg(&a.slice),
            };
            args.arg(&gate_q4k.quants)
                .arg(&up_q4k.quants)
                .arg(&mut out_dev)
                .arg(&m)
                .arg(&k)
                .arg(&b_cols);
            args.arg(&skipped_rows);
            let config = LaunchConfig {
                grid_dim: (
                    b_cols.div_ceil(u64::from(Q8_MATMUL_WARPS)) as u32,
                    m.div_ceil(Q8_MATMUL_ROWS) as u32,
                    1,
                ),
                block_dim: (32, Q8_MATMUL_WARPS, 1),
                shared_mem_bytes: 0,
            };
            unsafe { args.launch(config) }?;
            return Ok(CudaDeviceBuffer {
                slice: out_dev,
                shape: vec![m, b_cols],
                bf16: None,
                q8: None,
                tq: None,
            });
        }
        // A long prefill against packed weights: both expanded to f32 for
        // this call, the products held a block of rows at a time (what
        // each projection's packed matmul and `silu_mul` give).
        if gate.is_q8()
            && up.is_q8()
            && !transpose_a
            && !transpose_b
            && m > Q8_MATMUL_MAX_ROWS
            && m >= MATMUL_TILE_MIN_ROWS
            && k == b_rows
        {
            Self::require_f32(&[a], "gated-matmul's input")?;
            let (dense_gate, dense_up) = (self.dequantize_q8(gate)?, self.dequantize_q8(up)?);
            if let Some(out) =
                self.gated_matmul_split_tcp(a, &dense_gate, &dense_up, [m, k, b_cols])?
            {
                return Ok(out);
            }
        }
        // Other packed weights: each projection through the packed matmul.
        if transpose_b || m > MATMUL_SPLIT_K_MAX_ROWS || k != b_rows || gate.is_q8() || up.is_q8() {
            let gate_out = self.matmul(a, gate, transpose_a, transpose_b)?;
            let up_out = self.matmul(a, up, transpose_a, transpose_b)?;
            return self.silu_mul(&gate_out, &up_out);
        }
        Self::require_f32(&[a], "gated-matmul's input")?;
        let n = b_cols;
        let (a_row_stride, a_inner_stride) = if transpose_a {
            (1, a_cols)
        } else {
            (a_cols, 1)
        };
        let b_inner_stride = n;
        let (gate16, up16) = (self.bf16_address(gate), self.bf16_address(up));
        let mut out_dev = self.alloc_output((m * n) as usize)?;
        // A long prefill against compact weights: the pipelined Tensor
        // Core Kernel when the rows align, else the plain one.
        if m >= MATMUL_TILE_MIN_ROWS
            && gate16 != 0
            && up16 != 0
            && !transpose_a
            && k % TCP_ALIGN == 0
            && n % TCP_ALIGN == 0
            && let Some(func) = self.tensor_function("gated_matmul_tcp_kernel")
            && let Some((_split_lock, a_hi, a_lo)) = self.split_bf16(a, m * k)?
        {
            let mut args = self.stream.launch_builder(&func);
            args.arg(&a_hi)
                .arg(&a_lo)
                .arg(&gate16)
                .arg(&up16)
                .arg(&mut out_dev)
                .arg(&m)
                .arg(&k)
                .arg(&n);
            let config = LaunchConfig {
                grid_dim: (
                    n.div_ceil(TCP_BLOCK_N) as u32,
                    m.div_ceil(TCP_BLOCK_M) as u32,
                    1,
                ),
                block_dim: (TCP_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { args.launch(config) }?;
            return Ok(CudaDeviceBuffer {
                slice: out_dev,
                shape: vec![m, n],
                bf16: None,
                q8: None,
                tq: None,
            });
        }
        if m >= MATMUL_TILE_MIN_ROWS
            && gate16 != 0
            && up16 != 0
            && a_inner_stride == 1
            && let Some(func) = self.tensor_function("gated_matmul_tc_kernel")
        {
            let mut args = self.stream.launch_builder(&func);
            args.arg(&a.slice)
                .arg(&gate16)
                .arg(&up16)
                .arg(&mut out_dev)
                .arg(&m)
                .arg(&k)
                .arg(&n)
                .arg(&a_row_stride);
            let config = LaunchConfig {
                grid_dim: (
                    n.div_ceil(TC_BLOCK_N) as u32,
                    m.div_ceil(TC_BLOCK_M) as u32,
                    1,
                ),
                block_dim: (TC_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { args.launch(config) }?;
            return Ok(CudaDeviceBuffer {
                slice: out_dev,
                shape: vec![m, n],
                bf16: None,
                q8: None,
                tq: None,
            });
        }
        let (name, grid_dim) = split_k_rows(
            m,
            n,
            [
                "gated_matmul_split_k_kernel",
                "gated_matmul_split_k_rows_kernel",
                "gated_matmul_split_k_tile_kernel",
            ],
        );
        let func = self.function(name)?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&a.slice)
            .arg(&gate.slice)
            .arg(&gate16)
            .arg(&up.slice)
            .arg(&up16)
            .arg(&mut out_dev)
            .arg(&m)
            .arg(&k)
            .arg(&n)
            .arg(&a_row_stride)
            .arg(&a_inner_stride)
            .arg(&b_inner_stride);
        let config = LaunchConfig {
            grid_dim,
            block_dim: (32, MATMUL_SPLIT_K, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![m, n],
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// `q8_matmul_rows_kernel`: `a` (`m` contiguous rows of `k`) times the
    /// packed, transposed-storage weight `q8` (`[k, n]` logically).
    fn q8_matmul_rows(
        &self,
        a: &CudaDeviceBuffer,
        q8: &CudaQ8Weight,
        m: u64,
        k: u64,
        n: u64,
        addend: Option<MatmulAddend<'_>>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        let mut out_dev = self.alloc_output((m * n) as usize)?;
        let (addend_buffer, addend_mode) = match addend {
            None => (a, 0i32),
            Some(MatmulAddend::Row(bias)) => (bias, 1),
            Some(MatmulAddend::Full(residual)) => (residual, 2),
        };
        let int8 = self.quantize_activations(a, m, k)?;
        let int8 = int8.as_ref().and_then(|guard| guard.as_ref());
        let func = self.function(if int8.is_some() {
            "q8_matmul_rows_int8_kernel"
        } else {
            "q8_matmul_rows_kernel"
        })?;
        let skipped_rows = self.padding_rows(m);
        let mut args = self.stream.launch_builder(&func);
        match int8 {
            Some(scratch) => args.arg(&scratch.values).arg(&scratch.scales),
            None => args.arg(&a.slice),
        };
        args.arg(&q8.quants)
            .arg(&q8.scales)
            .arg(&mut out_dev)
            .arg(&m)
            .arg(&k)
            .arg(&n)
            .arg(&addend_buffer.slice)
            .arg(&addend_mode);
        args.arg(&skipped_rows);
        let config = LaunchConfig {
            grid_dim: (
                n.div_ceil(u64::from(Q8_MATMUL_WARPS)) as u32,
                m.div_ceil(Q8_MATMUL_ROWS) as u32,
                1,
            ),
            block_dim: (32, Q8_MATMUL_WARPS, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![m, n],
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// `q4k_matmul_rows_kernel` (or its `Q5_K` twin): `a` (`m` contiguous
    /// rows of `k`) times the packed `Q4_K` or `Q5_K`, transposed-storage
    /// weight `q4k` (`[k, n]` logically).
    fn q4k_matmul_rows(
        &self,
        a: &CudaDeviceBuffer,
        q4k: &CudaQ8Weight,
        m: u64,
        k: u64,
        n: u64,
        addend: Option<MatmulAddend<'_>>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        let mut out_dev = self.alloc_output((m * n) as usize)?;
        let (addend_buffer, addend_mode) = match addend {
            None => (a, 0i32),
            Some(MatmulAddend::Row(bias)) => (bias, 1),
            Some(MatmulAddend::Full(residual)) => (residual, 2),
        };
        let int8 = self.quantize_activations(a, m, k)?;
        let int8 = int8.as_ref().and_then(|guard| guard.as_ref());
        let q5k = q4k.format == PackedWeightFormat::Q5K;
        let func = self.function(match (q5k, int8.is_some()) {
            (false, true) => "q4k_matmul_rows_int8_kernel",
            (false, false) => "q4k_matmul_rows_kernel",
            (true, true) => "q5k_matmul_rows_int8_kernel",
            (true, false) => "q5k_matmul_rows_kernel",
        })?;
        let skipped_rows = self.padding_rows(m);
        let mut args = self.stream.launch_builder(&func);
        match int8 {
            Some(scratch) => args.arg(&scratch.values).arg(&scratch.scales),
            None => args.arg(&a.slice),
        };
        args.arg(&q4k.quants)
            .arg(&mut out_dev)
            .arg(&m)
            .arg(&k)
            .arg(&n)
            .arg(&addend_buffer.slice)
            .arg(&addend_mode);
        args.arg(&skipped_rows);
        let config = LaunchConfig {
            grid_dim: (
                n.div_ceil(u64::from(Q8_MATMUL_WARPS) * Q4K_MATMUL_COLS) as u32,
                m.div_ceil(Q8_MATMUL_ROWS) as u32,
                1,
            ),
            block_dim: (32, Q8_MATMUL_WARPS, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![m, n],
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// `iq_matmul_rows_int8_kernel` for an IQ2/IQ3 weight, when the
    /// activations can be quantized to int8; `None` otherwise.
    fn iq_matmul_rows(
        &self,
        a: &CudaDeviceBuffer,
        weight: &CudaQ8Weight,
        m: u64,
        k: u64,
        n: u64,
        addend: Option<MatmulAddend<'_>>,
    ) -> Result<Option<CudaDeviceBuffer>, CudaError> {
        let int8 = self.quantize_activations(a, m, k)?;
        let Some(scratch) = int8.as_ref().and_then(|guard| guard.as_ref()) else {
            return Ok(None);
        };
        let mut out_dev = self.alloc_output((m * n) as usize)?;
        let (addend_buffer, addend_mode) = match addend {
            None => (a, 0i32),
            Some(MatmulAddend::Row(bias)) => (bias, 1),
            Some(MatmulAddend::Full(residual)) => (residual, 2),
        };
        let format = iq_format(weight.format).expect("an IQ2/IQ3 format");
        let func = self.function("iq_matmul_rows_int8_kernel")?;
        let skipped_rows = self.padding_rows(m);
        let mut args = self.stream.launch_builder(&func);
        args.arg(&format)
            .arg(&scratch.values)
            .arg(&scratch.scales)
            .arg(&weight.quants)
            .arg(&weight.scales)
            .arg(&mut out_dev)
            .arg(&m)
            .arg(&k)
            .arg(&n)
            .arg(&addend_buffer.slice)
            .arg(&addend_mode);
        args.arg(&skipped_rows);
        let config = LaunchConfig {
            grid_dim: (
                n.div_ceil(u64::from(Q8_MATMUL_WARPS) * IQ_MATMUL_COLS) as u32,
                m.div_ceil(Q8_MATMUL_ROWS) as u32,
                1,
            ),
            block_dim: (32, Q8_MATMUL_WARPS, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        Ok(Some(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![m, n],
            bf16: None,
            q8: None,
            tq: None,
        }))
    }

    /// `iq4_matmul_rows_int8_kernel` for an `IQ4_NL`/`IQ4_XS` weight, when
    /// the activations can be quantized to int8; `None` otherwise.
    fn iq4_matmul_rows(
        &self,
        a: &CudaDeviceBuffer,
        weight: &CudaQ8Weight,
        m: u64,
        k: u64,
        n: u64,
        addend: Option<MatmulAddend<'_>>,
    ) -> Result<Option<CudaDeviceBuffer>, CudaError> {
        let int8 = self.quantize_activations(a, m, k)?;
        let Some(scratch) = int8.as_ref().and_then(|guard| guard.as_ref()) else {
            return Ok(None);
        };
        let mut out_dev = self.alloc_output((m * n) as usize)?;
        let (addend_buffer, addend_mode) = match addend {
            None => (a, 0i32),
            Some(MatmulAddend::Row(bias)) => (bias, 1),
            Some(MatmulAddend::Full(residual)) => (residual, 2),
        };
        let format = iq4_format(weight.format);
        let subscales = weight.quants.slice(iq4_subscale_offset(weight)..);
        let func = self.function("iq4_matmul_rows_int8_kernel")?;
        let skipped_rows = self.padding_rows(m);
        let mut args = self.stream.launch_builder(&func);
        args.arg(&format)
            .arg(&scratch.values)
            .arg(&scratch.scales)
            .arg(&weight.quants)
            .arg(&subscales)
            .arg(&weight.scales)
            .arg(&mut out_dev)
            .arg(&m)
            .arg(&k)
            .arg(&n)
            .arg(&addend_buffer.slice)
            .arg(&addend_mode);
        args.arg(&skipped_rows);
        let config = LaunchConfig {
            grid_dim: (
                n.div_ceil(u64::from(Q8_MATMUL_WARPS)) as u32,
                m.div_ceil(Q8_MATMUL_ROWS) as u32,
                1,
            ),
            block_dim: (32, Q8_MATMUL_WARPS, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        Ok(Some(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![m, n],
            bf16: None,
            q8: None,
            tq: None,
        }))
    }

    /// `lowk_matmul_rows_int8_kernel` for a `Q2_K`/`Q3_K` weight, when the
    /// activations can be quantized to int8; `None` otherwise.
    fn low_bit_matmul_rows(
        &self,
        a: &CudaDeviceBuffer,
        weight: &CudaQ8Weight,
        m: u64,
        k: u64,
        n: u64,
        addend: Option<MatmulAddend<'_>>,
    ) -> Result<Option<CudaDeviceBuffer>, CudaError> {
        let int8 = self.quantize_activations(a, m, k)?;
        let Some(scratch) = int8.as_ref().and_then(|guard| guard.as_ref()) else {
            return Ok(None);
        };
        let mut out_dev = self.alloc_output((m * n) as usize)?;
        let (addend_buffer, addend_mode) = match addend {
            None => (a, 0i32),
            Some(MatmulAddend::Row(bias)) => (bias, 1),
            Some(MatmulAddend::Full(residual)) => (residual, 2),
        };
        let format = low_bit_format(weight.format);
        let func = self.function("lowk_matmul_rows_int8_kernel")?;
        let skipped_rows = self.padding_rows(m);
        let mut args = self.stream.launch_builder(&func);
        args.arg(&format)
            .arg(&scratch.values)
            .arg(&scratch.scales)
            .arg(&weight.quants)
            .arg(&weight.scales)
            .arg(&mut out_dev)
            .arg(&m)
            .arg(&k)
            .arg(&n)
            .arg(&addend_buffer.slice)
            .arg(&addend_mode);
        args.arg(&skipped_rows);
        let config = LaunchConfig {
            grid_dim: (
                n.div_ceil(u64::from(Q8_MATMUL_WARPS) * LOW_BIT_MATMUL_COLS) as u32,
                m.div_ceil(Q8_MATMUL_ROWS) as u32,
                1,
            ),
            block_dim: (32, Q8_MATMUL_WARPS, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        Ok(Some(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![m, n],
            bf16: None,
            q8: None,
            tq: None,
        }))
    }

    /// `q6k_matmul_rows_kernel`: `a` (`m` contiguous rows of `k`) times the
    /// packed `Q6_K`, transposed-storage weight `q6k` (`[k, n]` logically).
    fn q6k_matmul_rows(
        &self,
        a: &CudaDeviceBuffer,
        q6k: &CudaQ8Weight,
        m: u64,
        k: u64,
        n: u64,
        addend: Option<MatmulAddend<'_>>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        let mut out_dev = self.alloc_output((m * n) as usize)?;
        let (addend_buffer, addend_mode) = match addend {
            None => (a, 0i32),
            Some(MatmulAddend::Row(bias)) => (bias, 1),
            Some(MatmulAddend::Full(residual)) => (residual, 2),
        };
        let int8 = self.quantize_activations(a, m, k)?;
        let int8 = int8.as_ref().and_then(|guard| guard.as_ref());
        let func = self.function(if int8.is_some() {
            "q6k_matmul_rows_int8_kernel"
        } else {
            "q6k_matmul_rows_kernel"
        })?;
        let skipped_rows = self.padding_rows(m);
        let mut args = self.stream.launch_builder(&func);
        match int8 {
            Some(scratch) => args.arg(&scratch.values).arg(&scratch.scales),
            None => args.arg(&a.slice),
        };
        args.arg(&q6k.quants)
            .arg(&q6k.scales)
            .arg(&mut out_dev)
            .arg(&m)
            .arg(&k)
            .arg(&n)
            .arg(&addend_buffer.slice)
            .arg(&addend_mode);
        args.arg(&skipped_rows);
        let config = LaunchConfig {
            grid_dim: (
                n.div_ceil(u64::from(Q8_MATMUL_WARPS) * Q6K_MATMUL_COLS) as u32,
                m.div_ceil(Q8_MATMUL_ROWS) as u32,
                1,
            ),
            block_dim: (32, Q8_MATMUL_WARPS, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![m, n],
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// `silu(a x gate) * (a x up)` for two transposed-storage packed
    /// weights; see `q8_gated_matmul_rows_kernel`.
    fn q8_gated_matmul_rows(
        &self,
        a: &CudaDeviceBuffer,
        gate: &CudaQ8Weight,
        up: &CudaQ8Weight,
        m: u64,
        k: u64,
        n: u64,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        let mut out_dev = self.alloc_output((m * n) as usize)?;
        let int8 = self.quantize_activations(a, m, k)?;
        let int8 = int8.as_ref().and_then(|guard| guard.as_ref());
        let func = self.function(if int8.is_some() {
            "q8_gated_matmul_rows_int8_kernel"
        } else {
            "q8_gated_matmul_rows_kernel"
        })?;
        let skipped_rows = self.padding_rows(m);
        let mut args = self.stream.launch_builder(&func);
        match int8 {
            Some(scratch) => args.arg(&scratch.values).arg(&scratch.scales),
            None => args.arg(&a.slice),
        };
        args.arg(&gate.quants)
            .arg(&gate.scales)
            .arg(&up.quants)
            .arg(&up.scales)
            .arg(&mut out_dev)
            .arg(&m)
            .arg(&k)
            .arg(&n);
        args.arg(&skipped_rows);
        let config = LaunchConfig {
            grid_dim: (
                n.div_ceil(u64::from(Q8_MATMUL_WARPS)) as u32,
                m.div_ceil(Q8_MATMUL_ROWS) as u32,
                1,
            ),
            block_dim: (32, Q8_MATMUL_WARPS, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![m, n],
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// `a x b` (plus `addend`) on Tensor Cores for a contiguous f32 `b` that
    /// is not exactly bfloat16, a dequantized packed weight: `a` and `b`
    /// are both split into three bfloat16 terms (about 24 significant
    /// bits) and every product of terms down to 2^-24 of the whole is
    /// accumulated in f32 (`matmul_tcp_split3_kernel`), as precise as the
    /// f32 Kernels. `None` without Tensor Cores or with rows that do not
    /// align.
    fn matmul_split_tcp(
        &self,
        a: &CudaDeviceBuffer,
        b: &CudaDeviceBuffer,
        [m, k, n]: [u64; 3],
        addend: Option<MatmulAddend<'_>>,
    ) -> Result<Option<CudaDeviceBuffer>, CudaError> {
        let Some((func, block_rows)) = self.split_tcp_function([m, k, n])? else {
            return Ok(None);
        };
        // B's terms, then room for one block of A rows' terms.
        let Some((_split_lock, terms)) = self.split_bf16_3_regions(&[k * n, block_rows * k])?
        else {
            return Ok(None);
        };
        let [b_terms, a_terms] = [terms[0], terms[1]];
        self.split_bf16_3_into(b.slice.device_ptr(&self.stream).0, k * n, b_terms)?;
        let out_dev = self.alloc_output((m * n) as usize)?;
        let out = out_dev.device_ptr(&self.stream).0;
        let (addend_address, addend_mode) = match addend {
            None => (0u64, 0i32),
            Some(MatmulAddend::Row(bias)) => (bias.slice.device_ptr(&self.stream).0, 1),
            Some(MatmulAddend::Full(residual)) => (residual.slice.device_ptr(&self.stream).0, 2),
        };
        let a_address = a.slice.device_ptr(&self.stream).0;
        let mut first = 0;
        while first < m {
            let rows = block_rows.min(m - first);
            self.split_bf16_3_into(a_address + first * k * 4, rows * k, a_terms)?;
            // A full addend's rows follow the block's; a bias is every row's.
            let addend_rows = if addend_mode == 2 {
                addend_address + first * n * 4
            } else {
                addend_address
            };
            self.launch_split_tcp(
                &func,
                a_terms,
                b_terms,
                out + first * n * 4,
                [rows, k, n],
                (addend_rows, addend_mode),
            )?;
            first += rows;
        }
        Ok(Some(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![m, n],
            bf16: None,
            q8: None,
            tq: None,
        }))
    }

    /// `matmul_tcp_split3_kernel`, ready to launch for an `[m, k] x [k, n]`
    /// product, and the rows of A one launch serves
    /// ([`SPLIT_TCP_BLOCK_ROWS`] at most); `None` without Tensor Cores or
    /// when K or N do not align.
    fn split_tcp_function(
        &self,
        [m, k, n]: [u64; 3],
    ) -> Result<Option<(CudaFunction, u64)>, CudaError> {
        if m == 0 || k % TCP_ALIGN != 0 || n % TCP_ALIGN != 0 {
            return Ok(None);
        }
        let Some(func) = self.tensor_function("matmul_tcp_split3_kernel") else {
            return Ok(None);
        };
        if func
            .set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                TCP3_SHARED_BYTES as i32,
            )
            .is_err()
        {
            return Ok(None);
        }
        Ok(Some((func, m.min(SPLIT_TCP_BLOCK_ROWS))))
    }

    /// One launch of `matmul_tcp_split3_kernel` over `m` rows of A (their
    /// terms at `a_terms`) into the `m` output rows at `out`, plus the
    /// addend at `addend` (`addend_mode` 1: one row; 2: `m` rows).
    fn launch_split_tcp(
        &self,
        func: &CudaFunction,
        [a_hi, a_mid, a_lo]: [u64; 3],
        [b_hi, b_mid, b_lo]: [u64; 3],
        out: u64,
        [m, k, n]: [u64; 3],
        (addend, addend_mode): (u64, i32),
    ) -> Result<(), CudaError> {
        let overflow = || {
            CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "matmul launch size overflows a grid dimension",
            )
        };
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(n.div_ceil(TCP_BLOCK_N)).map_err(|_| overflow())?,
                u32::try_from(m.div_ceil(TCP_BLOCK_M)).map_err(|_| overflow())?,
                1,
            ),
            block_dim: (TCP_THREADS, 1, 1),
            shared_mem_bytes: TCP3_SHARED_BYTES,
        };
        let mut args = self.stream.launch_builder(func);
        args.arg(&a_hi)
            .arg(&a_mid)
            .arg(&a_lo)
            .arg(&b_hi)
            .arg(&b_mid)
            .arg(&b_lo)
            .arg(&out)
            .arg(&m)
            .arg(&k)
            .arg(&n)
            .arg(&addend)
            .arg(&addend_mode);
        unsafe { args.launch(config) }?;
        Ok(())
    }

    /// `silu(a x gate) * (a x up)` for a long prefill against f32 weights,
    /// on Tensor Cores ([`Self::matmul_split_tcp`]'s products and
    /// `silu_mul_kernel`, so its values are theirs), [`SPLIT_TCP_BLOCK_ROWS`]
    /// rows of A at a time: the gate's and up's products are held for one
    /// block, never for every row. `None` without Tensor Cores or when K or
    /// N do not align.
    fn gated_matmul_split_tcp(
        &self,
        a: &CudaDeviceBuffer,
        gate: &CudaDeviceBuffer,
        up: &CudaDeviceBuffer,
        [m, k, n]: [u64; 3],
    ) -> Result<Option<CudaDeviceBuffer>, CudaError> {
        let Some((func, block_rows)) = self.split_tcp_function([m, k, n])? else {
            return Ok(None);
        };
        let Some((_split_lock, terms)) =
            self.split_bf16_3_regions(&[k * n, k * n, block_rows * k])?
        else {
            return Ok(None);
        };
        let [gate_terms, up_terms, a_terms] = [terms[0], terms[1], terms[2]];
        self.split_bf16_3_into(gate.slice.device_ptr(&self.stream).0, k * n, gate_terms)?;
        self.split_bf16_3_into(up.slice.device_ptr(&self.stream).0, k * n, up_terms)?;
        let silu_mul = self.function("silu_mul_kernel")?;
        // SAFETY: every element is written by `matmul_tcp_split3_kernel`
        // before `silu_mul_kernel` reads it, on this one stream.
        let gate_rows = unsafe { self.stream.alloc::<f32>((block_rows * n) as usize) }?;
        let up_rows = unsafe { self.stream.alloc::<f32>((block_rows * n) as usize) }?;
        let (gate_address, up_address) = (
            gate_rows.device_ptr(&self.stream).0,
            up_rows.device_ptr(&self.stream).0,
        );
        let out_dev = self.alloc_output((m * n) as usize)?;
        let out = out_dev.device_ptr(&self.stream).0;
        let a_address = a.slice.device_ptr(&self.stream).0;
        let mut first = 0;
        while first < m {
            let rows = block_rows.min(m - first);
            self.split_bf16_3_into(a_address + first * k * 4, rows * k, a_terms)?;
            for (weight_terms, products) in [(gate_terms, gate_address), (up_terms, up_address)] {
                self.launch_split_tcp(
                    &func,
                    a_terms,
                    weight_terms,
                    products,
                    [rows, k, n],
                    (0, 0),
                )?;
            }
            let count = rows * n;
            let out_rows = out + first * n * 4;
            let mut args = self.stream.launch_builder(&silu_mul);
            args.arg(&gate_address)
                .arg(&up_address)
                .arg(&out_rows)
                .arg(&count);
            unsafe { args.launch(LaunchConfig::for_num_elems(count as u32)) }?;
            first += rows;
        }
        Ok(Some(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![m, n],
            bf16: None,
            q8: None,
            tq: None,
        }))
    }

    fn matmul_with_addend(
        &self,
        a: &CudaDeviceBuffer,
        b: &CudaDeviceBuffer,
        addend: Option<MatmulAddend<'_>>,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        let (a_rows, a_cols) = rows_cols_buf(a)?;
        let (b_rows, b_cols) = rows_cols_buf(b)?;
        let (m, k) = if transpose_a {
            (a_cols, a_rows)
        } else {
            (a_rows, a_cols)
        };
        let (k2, n) = if transpose_b {
            (b_cols, b_rows)
        } else {
            (b_rows, b_cols)
        };
        if k != k2 {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("matmul inner dimension mismatch: {k} vs {k2}"),
            ));
        }
        let (a_row_stride, a_inner_stride) = if transpose_a {
            (1u64, a_cols)
        } else {
            (a_cols, 1u64)
        };
        let (b_inner_stride, b_col_stride) = if transpose_b {
            (1u64, b_cols)
        } else {
            (b_cols, 1u64)
        };
        match addend {
            Some(MatmulAddend::Row(bias)) if bias.shape != [n] && bias.shape != [1, n] => {
                return Err(CudaError::new(
                    CudaErrorCode::ShapeUnsupported,
                    format!(
                        "matmul-bias bias shape must be [{n}] or [1, {n}], got {:?}",
                        bias.shape
                    ),
                ));
            }
            Some(MatmulAddend::Full(residual)) if residual.shape != [m, n] => {
                return Err(CudaError::new(
                    CudaErrorCode::ShapeUnsupported,
                    format!(
                        "matmul-residual residual shape must be [{m}, {n}], got {:?}",
                        residual.shape
                    ),
                ));
            }
            _ => {}
        }
        Self::require_f32(&[a], "matmul's a")?;
        if let Some(MatmulAddend::Row(addend) | MatmulAddend::Full(addend)) = addend {
            Self::require_f32(&[addend], "a fused matmul's addend")?;
        }
        if let Some(q8) = &b.q8 {
            // Read packed: a transposed-storage weight (each output
            // column's quants contiguous) against a few contiguous rows.
            if q8.transposed && !transpose_a && !transpose_b && m <= Q8_MATMUL_MAX_ROWS {
                return match q8.format {
                    PackedWeightFormat::Q8_0 => self.q8_matmul_rows(a, q8, m, k, n, addend),
                    PackedWeightFormat::Q4K | PackedWeightFormat::Q5K => {
                        self.q4k_matmul_rows(a, q8, m, k, n, addend)
                    }
                    PackedWeightFormat::Q6K => self.q6k_matmul_rows(a, q8, m, k, n, addend),
                    PackedWeightFormat::Iq2Xxs
                    | PackedWeightFormat::Iq2Xs
                    | PackedWeightFormat::Iq2S
                    | PackedWeightFormat::Iq3Xxs
                    | PackedWeightFormat::Iq3S => {
                        match self.iq_matmul_rows(a, q8, m, k, n, addend)? {
                            Some(out) => Ok(out),
                            None => {
                                let dense = self.dequantize_q8(b)?;
                                self.matmul_with_addend(a, &dense, addend, transpose_a, transpose_b)
                            }
                        }
                    }
                    PackedWeightFormat::Iq4Nl
                    | PackedWeightFormat::Iq4Xs
                    | PackedWeightFormat::Q4_0 => {
                        match self.iq4_matmul_rows(a, q8, m, k, n, addend)? {
                            Some(out) => Ok(out),
                            None => {
                                let dense = self.dequantize_q8(b)?;
                                self.matmul_with_addend(a, &dense, addend, transpose_a, transpose_b)
                            }
                        }
                    }
                    PackedWeightFormat::Q2K | PackedWeightFormat::Q3K => {
                        match self.low_bit_matmul_rows(a, q8, m, k, n, addend)? {
                            Some(out) => Ok(out),
                            // No int8 activations: against the f32 weight.
                            None => {
                                let dense = self.dequantize_q8(b)?;
                                self.matmul_with_addend(a, &dense, addend, transpose_a, transpose_b)
                            }
                        }
                    }
                };
            }
            // Otherwise against the f32 weight, for this call only: a long
            // prefill on Tensor Cores, the weight split into its bfloat16
            // hi and lo terms like the activations.
            let dense = self.dequantize_q8(b)?;
            if m >= MATMUL_TILE_MIN_ROWS
                && !transpose_a
                && !transpose_b
                && let Some(out) = self.matmul_split_tcp(a, &dense, [m, k, n], addend)?
            {
                return Ok(out);
            }
            return self.matmul_with_addend(a, &dense, addend, transpose_a, transpose_b);
        }
        let b16 = self.bf16_address(b);
        let mut out_dev = self.alloc_output((m * n) as usize)?;
        let (addend_buffer, addend_mode) = match addend {
            None => (a, 0i32),
            Some(MatmulAddend::Row(bias)) => (bias, 1),
            Some(MatmulAddend::Full(residual)) => (residual, 2),
        };
        // A long prefill against a compact weight: the pipelined Tensor
        // Core Kernel when the rows align, else the plain one.
        if m >= MATMUL_TILE_MIN_ROWS
            && b16 != 0
            && !transpose_a
            && !transpose_b
            && k % TCP_ALIGN == 0
            && n % TCP_ALIGN == 0
            && let Some(func) = self.tensor_function("matmul_tcp_kernel")
            && let Some((_split_lock, a_hi, a_lo)) = self.split_bf16(a, m * k)?
        {
            let mut args = self.stream.launch_builder(&func);
            args.arg(&a_hi)
                .arg(&a_lo)
                .arg(&b16)
                .arg(&mut out_dev)
                .arg(&m)
                .arg(&k)
                .arg(&n)
                .arg(&addend_buffer.slice)
                .arg(&addend_mode);
            let config = LaunchConfig {
                grid_dim: (
                    n.div_ceil(TCP_BLOCK_N) as u32,
                    m.div_ceil(TCP_BLOCK_M) as u32,
                    1,
                ),
                block_dim: (TCP_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { args.launch(config) }?;
            return Ok(CudaDeviceBuffer {
                slice: out_dev,
                shape: vec![m, n],
                bf16: None,
                q8: None,
                tq: None,
            });
        }
        if m >= MATMUL_TILE_MIN_ROWS
            && b16 != 0
            && a_inner_stride == 1
            && b_col_stride == 1
            && b_inner_stride == n
            && let Some(func) = self.tensor_function("matmul_tc_kernel")
        {
            let mut args = self.stream.launch_builder(&func);
            args.arg(&a.slice)
                .arg(&b16)
                .arg(&mut out_dev)
                .arg(&m)
                .arg(&k)
                .arg(&n)
                .arg(&a_row_stride)
                .arg(&addend_buffer.slice)
                .arg(&addend_mode);
            let config = LaunchConfig {
                grid_dim: (
                    n.div_ceil(TC_BLOCK_N) as u32,
                    m.div_ceil(TC_BLOCK_M) as u32,
                    1,
                ),
                block_dim: (TC_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { args.launch(config) }?;
            return Ok(CudaDeviceBuffer {
                slice: out_dev,
                shape: vec![m, n],
                bf16: None,
                q8: None,
                tq: None,
            });
        }
        if b_col_stride == 1 && m <= MATMUL_SPLIT_K_MAX_ROWS {
            let (name, grid_dim) = split_k_rows(
                m,
                n,
                [
                    "matmul_split_k_kernel",
                    "matmul_split_k_rows_kernel",
                    "matmul_split_k_tile_kernel",
                ],
            );
            let func = self.function(name)?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&a.slice)
                .arg(&b.slice)
                .arg(&mut out_dev)
                .arg(&m)
                .arg(&k)
                .arg(&n)
                .arg(&a_row_stride)
                .arg(&a_inner_stride)
                .arg(&b_inner_stride);
            // Without an addend the kernel never reads it; any valid device
            // pointer will do.
            let (addend_buffer, addend_mode) = match addend {
                None => (b, 0i32),
                Some(MatmulAddend::Row(bias)) => (bias, 1),
                Some(MatmulAddend::Full(residual)) => (residual, 2),
            };
            args.arg(&addend_buffer.slice).arg(&addend_mode).arg(&b16);
            let config = LaunchConfig {
                grid_dim,
                block_dim: (32, MATMUL_SPLIT_K, 1),
                shared_mem_bytes: 0,
            };
            unsafe { args.launch(config) }?;
            return Ok(CudaDeviceBuffer {
                slice: out_dev,
                shape: vec![m, n],
                bf16: None,
                q8: None,
                tq: None,
            });
        }
        let func = self.function("matmul_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&a.slice)
            .arg(&b.slice)
            .arg(&mut out_dev)
            .arg(&m)
            .arg(&k)
            .arg(&n)
            .arg(&a_row_stride)
            .arg(&a_inner_stride)
            .arg(&b_inner_stride)
            .arg(&b_col_stride)
            .arg(&b16);
        unsafe { args.launch(LaunchConfig::for_num_elems((m * n) as u32)) }?;
        let product = CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![m, n],
            bf16: None,
            q8: None,
            tq: None,
        };
        match addend {
            Some(MatmulAddend::Row(addend) | MatmulAddend::Full(addend)) => {
                self.add(&product, addend)
            }
            None => Ok(product),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: &CudaDeviceBuffer,
        k: &CudaDeviceBuffer,
        v: &CudaDeviceBuffer,
        head_count: u64,
        head_dimension: u64,
        kv_head_count: Option<u64>,
        window_size: Option<u64>,
        causal: bool,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.attention_bounded(
            q,
            k,
            v,
            head_count,
            head_dimension,
            kv_head_count,
            window_size,
            causal,
            None,
        )
    }

    /// [`Self::attention`] over only the first `key_count` rows of K and V
    /// (`append-kv-history-in-place`): an in-place KV store holds more rows
    /// than the history, and the rows past `key_count` are never read.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_bounded(
        &self,
        q: &CudaDeviceBuffer,
        k: &CudaDeviceBuffer,
        v: &CudaDeviceBuffer,
        head_count: u64,
        head_dimension: u64,
        kv_head_count: Option<u64>,
        window_size: Option<u64>,
        causal: bool,
        key_count: Option<u64>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.attention_scaled(
            q,
            k,
            v,
            head_count,
            head_dimension,
            kv_head_count,
            window_size,
            causal,
            key_count,
            None,
        )
    }

    /// [`Self::attention_bounded`] with the factor on `QK^T` the node sets
    /// (`attention.scale`); `None` keeps `1/sqrt(head_dimension)`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_scaled(
        &self,
        q: &CudaDeviceBuffer,
        k: &CudaDeviceBuffer,
        v: &CudaDeviceBuffer,
        head_count: u64,
        head_dimension: u64,
        kv_head_count: Option<u64>,
        window_size: Option<u64>,
        causal: bool,
        key_count: Option<u64>,
        scale: Option<f32>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.attention_valued(
            q,
            k,
            v,
            head_count,
            head_dimension,
            None,
            kv_head_count,
            window_size,
            causal,
            key_count,
            scale,
        )
    }

    /// `batched-matmul` (`cache-latent-attention-history`): `a [rows,
    /// groups * k]` times `b` group by group, each group's block `[k, n]`
    /// (`b [groups * k, n]`) or, with `transpose_b`, `[n, k]` used
    /// transposed (`b [groups * n, k]`), giving `[rows, groups * n]`. f32
    /// operands.
    pub fn batched_matmul(
        &self,
        a: &CudaDeviceBuffer,
        b: &CudaDeviceBuffer,
        groups: u64,
        transpose_b: bool,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[a, b], "batched-matmul")?;
        if a.is_q8() || b.is_q8() || a.is_tq() || b.is_tq() {
            return Err(CudaError::new(
                CudaErrorCode::DTypeUnsupported,
                "batched-matmul reads f32 operands",
            ));
        }
        let (rows, a_cols) = rows_cols_buf(a)?;
        let (b_rows, b_cols) = rows_cols_buf(b)?;
        let fits = groups > 0
            && a_cols.is_multiple_of(groups)
            && b_rows.is_multiple_of(groups)
            && if transpose_b {
                b_cols == a_cols / groups
            } else {
                b_rows == a_cols
            };
        if !fits {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "batched-matmul of {groups} groups (transpose_b {transpose_b}) cannot take {:?} and {:?}",
                    a.shape, b.shape
                ),
            ));
        }
        let k = a_cols / groups;
        let n = if transpose_b { b_rows / groups } else { b_cols };
        let len = rows * groups * n;

        let mut out = self.alloc_output(len as usize)?;
        if len > 0 && transpose_b {
            // One warp per output value: each reads a weight row.
            let func = self.function("batched_matmul_t_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&a.slice)
                .arg(&b.slice)
                .arg(&mut out)
                .arg(&rows)
                .arg(&groups)
                .arg(&k)
                .arg(&n);
            let warps_per_block = 4u64;
            let config = LaunchConfig {
                grid_dim: (
                    u32::try_from(len.div_ceil(warps_per_block)).map_err(|_| {
                        CudaError::new(
                            CudaErrorCode::ShapeUnsupported,
                            "batched-matmul launch size overflows a grid dimension",
                        )
                    })?,
                    1,
                    1,
                ),
                block_dim: (32 * warps_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { args.launch(config) }?;
        } else if len > 0 {
            // One thread per output value: a warp reads a weight row's
            // consecutive values.
            let func = self.function("batched_matmul_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            let transposed = 0i32;
            args.arg(&a.slice)
                .arg(&b.slice)
                .arg(&mut out)
                .arg(&rows)
                .arg(&groups)
                .arg(&k)
                .arg(&n)
                .arg(&transposed);
            unsafe { args.launch(LaunchConfig::for_num_elems(len as u32)) }?;
        }
        Ok(CudaDeviceBuffer {
            slice: out,
            shape: vec![rows, groups * n],
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// Attention whose V heads are `value_stride` wide, the value being
    /// each head's leading `value_dimension` (`attention.value_head_stride`,
    /// `cache-latent-attention-history`): V may be the key history. The
    /// values are gathered into a scratch buffer kept per shape, so a
    /// captured step allocates nothing, and the attention reads them.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_strided_values(
        &self,
        q: &CudaDeviceBuffer,
        k: &CudaDeviceBuffer,
        v: &CudaDeviceBuffer,
        head_count: u64,
        head_dimension: u64,
        value_dimension: Option<u64>,
        value_stride: u64,
        kv_head_count: Option<u64>,
        window_size: Option<u64>,
        causal: bool,
        key_count: Option<u64>,
        scale: Option<f32>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        // TurboQuant stores (`turboquant-kv-history`): the rows attention
        // may read, dequantized into scratch buffers kept per role and
        // shape; a store read as both K and V once.
        if k.is_tq() || v.is_tq() {
            let kv_rows = k.shape.first().copied().unwrap_or(0);
            let rows = key_count.unwrap_or(kv_rows);
            // On the exact heads directly while they hold the rows read
            // (`keep-turboquant-kv-head-exact`).
            if let (Some(kt), Some(vt)) = (&k.tq, &v.tq)
                && let (Some(k_head), Some(v_head)) =
                    (kt.head_covering(rows), vt.head_covering(rows))
            {
                return self.attention_strided_values(
                    q,
                    k_head,
                    v_head,
                    head_count,
                    head_dimension,
                    value_dimension,
                    value_stride,
                    kv_head_count,
                    window_size,
                    causal,
                    key_count,
                    scale,
                );
            }
            // A decode step over a latent store read as both K and V (each
            // row's leading values) runs on its codes.
            if let Some(tq) = &k.tq
                && std::ptr::eq(k, v)
                && rows_cols_buf(q).is_ok_and(|(seq_len, _)| seq_len == 1)
                && kv_head_count.unwrap_or(head_count) == 1
                && value_stride == head_dimension
                && tq.heads * tq.head_dimension == head_dimension
                && value_dimension.unwrap_or(head_dimension) <= head_dimension
                && value_dimension
                    .unwrap_or(head_dimension)
                    .is_multiple_of(tq.head_dimension)
                && head_count > 0
                && head_count.min(TQ_MAX_GROUP) * head_dimension * 4 <= LATENT_QUERY_SHARED_BYTES
                && self
                    .tq_latent_on_codes
                    .load(std::sync::atomic::Ordering::Relaxed)
            {
                return self.tq_latent_attention_decode(
                    q,
                    k,
                    head_count,
                    value_dimension.unwrap_or(head_dimension),
                    window_size,
                    causal,
                    key_count,
                    scale,
                );
            }
            let step_offset = match key_count {
                Some(_) => self.advancing_offset("key_count", kv_rows.saturating_sub(rows)),
                None => 0,
            };
            let take = |role: u8,
                        store: &CudaDeviceBuffer|
             -> Result<Option<CudaDeviceBuffer>, CudaError> {
                if !store.is_tq() {
                    return Ok(None);
                }
                let key = (role, store.shape.clone());
                let reused = self.tq_scratch.lock().unwrap().remove(&key);
                let mut slice = match reused {
                    Some(slice) => slice,
                    None => {
                        self.forbid_in_capture("allocating a TurboQuant attention scratch buffer")?;
                        let len = store.shape.iter().product::<u64>() as usize;
                        self.stream.alloc_zeros::<f32>(len.max(1))?
                    }
                };
                self.tq_dequantize_into(store, &mut slice, rows, step_offset)?;
                Ok(Some(CudaDeviceBuffer {
                    slice,
                    shape: store.shape.clone(),
                    bf16: None,
                    q8: None,
                    tq: None,
                }))
            };
            let same = std::ptr::eq(k, v);
            let k_values = take(3, k)?;
            let v_values = if same { None } else { take(4, v)? };
            let k_read = k_values.as_ref().unwrap_or(k);
            let v_read = if same {
                k_read
            } else {
                v_values.as_ref().unwrap_or(v)
            };
            let result = self.attention_strided_values(
                q,
                k_read,
                v_read,
                head_count,
                head_dimension,
                value_dimension,
                value_stride,
                kv_head_count,
                window_size,
                causal,
                key_count,
                scale,
            );
            let mut scratch = self.tq_scratch.lock().unwrap();
            for (role, values) in [(3u8, k_values), (4, v_values)] {
                if let Some(values) = values {
                    scratch.insert((role, values.shape), values.slice);
                }
            }
            return result;
        }
        Self::require_f32(&[v], "attention's strided values")?;
        if v.is_q8() {
            return Err(CudaError::new(
                CudaErrorCode::DTypeUnsupported,
                "attention reads strided values from f32 rows only",
            ));
        }
        let heads = kv_head_count.unwrap_or(head_count);
        let width = value_dimension.unwrap_or(head_dimension);
        let (rows, cols) = rows_cols_buf(v)?;
        if heads == 0 || width > value_stride || cols != heads * value_stride {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "V rows of {cols} values cannot hold {heads} heads of {value_stride} values \
                     with {width}-value values"
                ),
            ));
        }
        // A decode step reads the values in place: the split Kernel takes
        // the V row width apart from each head's value width
        // (`speed-up-decode-attention-on-codes`), where gathering them
        // would copy every row of the history at every step.
        if let Some(result) = self.attention_strided_decode(
            q,
            k,
            v,
            [head_count, head_dimension, width, heads],
            window_size,
            causal,
            key_count,
            scale,
        ) {
            return result;
        }
        let shape = vec![rows, heads * width];
        let len = rows * heads * width;
        let mut scratch = self.tq_scratch.lock().unwrap();
        let key = (2u8, shape.clone());
        let mut slice = match scratch.remove(&key) {
            Some(slice) => slice,
            None => {
                self.forbid_in_capture("allocating a strided-values scratch buffer")?;
                self.stream.alloc_zeros::<f32>((len as usize).max(1))?
            }
        };
        if len > 0 {
            let func = self.function("value_heads_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&v.slice)
                .arg(&mut slice)
                .arg(&rows)
                .arg(&heads)
                .arg(&value_stride)
                .arg(&width);
            unsafe { args.launch(LaunchConfig::for_num_elems(len as u32)) }?;
        }
        let values = CudaDeviceBuffer {
            slice,
            shape,
            bf16: None,
            q8: None,
            tq: None,
        };
        drop(scratch);
        let result = self.attention_valued(
            q,
            k,
            &values,
            head_count,
            head_dimension,
            Some(width),
            kv_head_count,
            window_size,
            causal,
            key_count,
            scale,
        );
        self.tq_scratch.lock().unwrap().insert(key, values.slice);
        result
    }

    /// Decode attention over a latent TurboQuant store read as both K and
    /// V (`tq_latent_attention_split_kernel`, then the combine and a
    /// rotation back per group): one query row of `head_count` heads as wide
    /// as the store's rows, the values each row's leading `width`.
    #[allow(clippy::too_many_arguments)]
    fn tq_latent_attention_decode(
        &self,
        q: &CudaDeviceBuffer,
        store: &CudaDeviceBuffer,
        head_count: u64,
        width: u64,
        window_size: Option<u64>,
        causal: bool,
        key_count: Option<u64>,
        scale: Option<f32>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        let overflow = || {
            CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "attention launch size overflows a grid dimension",
            )
        };
        let tq = store.tq.as_ref().expect("a TurboQuant store");
        let (kv_rows, k_width) = rows_cols_buf(store)?;
        let kv_seq_len = key_count.unwrap_or(kv_rows);
        if kv_seq_len == 0 || kv_seq_len > kv_rows || window_size == Some(0) {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("key_count {kv_seq_len} must lie in 1..={kv_rows}"),
            ));
        }
        let score_scale = match scale {
            None => 0.0_f32,
            Some(scale) if scale.is_finite() && scale > 0.0 => scale,
            Some(scale) => {
                return Err(CudaError::new(
                    CudaErrorCode::ShapeUnsupported,
                    format!("attention scale {scale} must be finite and positive"),
                ));
            }
        };
        let step_offset = match key_count {
            Some(_) => self.advancing_offset("key_count", kv_rows - kv_seq_len),
            None => 0,
        };
        let chunks = kv_rows.div_ceil(ATTENTION_CHUNK_KEYS).max(1);
        let partial_len =
            usize::try_from(head_count * chunks * (width + 2)).map_err(|_| overflow())?;
        let mut scratch = self.attention_scratch.lock().unwrap();
        if scratch
            .0
            .as_ref()
            .is_none_or(|slice| slice.len() < partial_len)
        {
            self.forbid_in_capture("growing the attention scratch buffer")?;
            // SAFETY: the split Kernel writes every partial of this launch
            // before the combine Kernel reads it, on this same stream.
            let grown = unsafe { self.stream.alloc::<f32>(partial_len.max(1 << 16)) }?;
            if let Some(outgrown) = scratch.0.replace(grown) {
                scratch.1.push(outgrown);
            }
        }
        let partial = scratch.0.as_mut().expect("allocated above");
        let mut out_dev =
            self.alloc_output(usize::try_from(head_count * width).map_err(|_| overflow())?)?;
        let d = tq.head_dimension;
        let (k_groups, v_groups) = (tq.heads, width / d);
        let causal_flag: i32 = if causal { 1 } else { 0 };
        let window_arg: i64 = window_size.map(|w| w as i64).unwrap_or(-1);
        let query_position_offset = kv_seq_len - 1;
        let no_head = 0u64;
        let split = self.function("tq_latent_attention_split_kernel")?;
        let mut args = self.stream.launch_builder(&split);
        args.arg(&q.slice)
            .arg(&tq.codes)
            .arg(&tq.norms)
            .arg(&tq.centroids)
            .arg(&tq.signs)
            .arg(&tq.offsets)
            .arg(&tq.bits)
            .arg(&mut *partial)
            .arg(&kv_seq_len)
            .arg(&head_count)
            .arg(&k_groups)
            .arg(&v_groups)
            .arg(&d)
            .arg(&causal_flag)
            .arg(&window_arg)
            .arg(&query_position_offset)
            .arg(&chunks)
            .arg(&step_offset)
            .arg(&score_scale);
        match tq.exact.as_ref() {
            Some(exact) => args.arg(&exact.slice),
            None => args.arg(&no_head),
        };
        args.arg(&tq.exact_rows);
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(chunks).map_err(|_| overflow())?,
                u32::try_from(head_count.div_ceil(TQ_MAX_GROUP)).map_err(|_| overflow())?,
                1,
            ),
            block_dim: (ATTENTION_SPLIT_THREADS, 1, 1),
            // The rotated queries of up to `TQ_MAX_GROUP` heads.
            shared_mem_bytes: u32::try_from(4 * head_count.min(TQ_MAX_GROUP) * k_width)
                .map_err(|_| overflow())?,
        };
        unsafe { args.launch(config) }?;
        self.combine_attention_partials(&*partial, &mut out_dev, head_count, width, chunks)?;
        let rotate = self.function("tq_latent_rotate_back_kernel")?;
        let mut args = self.stream.launch_builder(&rotate);
        args.arg(&mut out_dev)
            .arg(&tq.signs)
            .arg(&tq.offsets)
            .arg(&d)
            .arg(&v_groups);
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(head_count * v_groups).map_err(|_| overflow())?,
                1,
                1,
            ),
            block_dim: (tq_threads(d), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![1, head_count * width],
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// Decode attention over an f32 latent history read as both K and V
    /// (`latent_attention_split_kernel`, then the combine): one query row of
    /// `head_count` heads as wide as the rows, the values each row's
    /// leading `width`.
    #[allow(clippy::too_many_arguments)]
    fn latent_attention_decode(
        &self,
        q: &CudaDeviceBuffer,
        kv: &CudaDeviceBuffer,
        [head_count, k_width, width, kv_seq_len, kv_rows]: [u64; 5],
        causal_flag: i32,
        window_arg: i64,
        step_offset: u64,
        score_scale: f32,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        let overflow = || {
            CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "attention launch size overflows a grid dimension",
            )
        };
        let chunks = kv_rows.div_ceil(ATTENTION_CHUNK_KEYS).max(1);
        let partial_len =
            usize::try_from(head_count * chunks * (width + 2)).map_err(|_| overflow())?;
        let mut scratch = self.attention_scratch.lock().unwrap();
        if scratch
            .0
            .as_ref()
            .is_none_or(|slice| slice.len() < partial_len)
        {
            self.forbid_in_capture("growing the attention scratch buffer")?;
            // SAFETY: the split Kernel writes every partial of this launch
            // before the combine Kernel reads it, on this same stream.
            let grown = unsafe { self.stream.alloc::<f32>(partial_len.max(1 << 16)) }?;
            if let Some(outgrown) = scratch.0.replace(grown) {
                scratch.1.push(outgrown);
            }
        }
        let partial = scratch.0.as_mut().expect("allocated above");
        let mut out_dev =
            self.alloc_output(usize::try_from(head_count * width).map_err(|_| overflow())?)?;
        let query_position_offset = kv_seq_len - 1;
        let split = self.function("latent_attention_split_kernel")?;
        let mut args = self.stream.launch_builder(&split);
        args.arg(&q.slice)
            .arg(&kv.slice)
            .arg(&mut *partial)
            .arg(&kv_seq_len)
            .arg(&head_count)
            .arg(&k_width)
            .arg(&width)
            .arg(&causal_flag)
            .arg(&window_arg)
            .arg(&query_position_offset)
            .arg(&chunks)
            .arg(&step_offset)
            .arg(&score_scale);
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(chunks).map_err(|_| overflow())?,
                u32::try_from(head_count.div_ceil(TQ_MAX_GROUP)).map_err(|_| overflow())?,
                1,
            ),
            block_dim: (ATTENTION_SPLIT_THREADS, 1, 1),
            // The queries of up to `TQ_MAX_GROUP` heads.
            shared_mem_bytes: u32::try_from(4 * head_count.min(TQ_MAX_GROUP) * k_width)
                .map_err(|_| overflow())?,
        };
        unsafe { args.launch(config) }?;
        self.combine_attention_partials(&*partial, &mut out_dev, head_count, width, chunks)?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![1, head_count * width],
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// [`Self::attention_strided_values`] for one query row: the split
    /// Kernel reads each head's leading `width` values from V rows of
    /// `heads * value_stride` values in place. `None` for several query
    /// rows or shapes it does not serve, which gather the values instead.
    #[allow(clippy::too_many_arguments)]
    fn attention_strided_decode(
        &self,
        q: &CudaDeviceBuffer,
        k: &CudaDeviceBuffer,
        v: &CudaDeviceBuffer,
        [head_count, head_dimension, width, heads]: [u64; 4],
        window_size: Option<u64>,
        causal: bool,
        key_count: Option<u64>,
        scale: Option<f32>,
    ) -> Option<Result<CudaDeviceBuffer, CudaError>> {
        let (seq_len, q_model_dim) = rows_cols_buf(q).ok()?;
        let (kv_rows, kv_model_dim) = rows_cols_buf(k).ok()?;
        let (v_rows, v_model_dim) = rows_cols_buf(v).ok()?;
        let kv_seq_len = key_count.unwrap_or(kv_rows);
        let score_scale = match scale {
            None => 0.0_f32,
            Some(scale) if scale.is_finite() && scale > 0.0 => scale,
            Some(_) => return None,
        };
        if seq_len != 1
            || k.is_compact()
            || v.is_compact()
            || q.is_compact()
            || heads == 0
            || !head_count.is_multiple_of(heads)
            || head_count * head_dimension != q_model_dim
            || heads * head_dimension != kv_model_dim
            || v_rows != kv_rows
            || kv_seq_len == 0
            || kv_seq_len > kv_rows
            || window_size == Some(0)
            || (window_size.is_some() && !causal)
            // The grouped Kernels, for equal widths, read V at K's stride.
            || (width == head_dimension && v_model_dim != kv_model_dim)
        {
            return None;
        }
        let step_offset = match key_count {
            Some(_) => self.advancing_offset("key_count", kv_rows - kv_seq_len),
            None => 0,
        };
        // A latent history read as both K and V (one KV head, the values
        // each row's leading `width`): a block serves several query heads,
        // reading each row once for them all.
        if std::ptr::eq(k, v)
            && heads == 1
            && head_count > 1
            && v_model_dim == kv_model_dim
            && head_dimension.is_multiple_of(4)
            && width <= head_dimension
            && head_count.min(TQ_MAX_GROUP) * head_dimension * 4 <= LATENT_QUERY_SHARED_BYTES
        {
            return Some(self.latent_attention_decode(
                q,
                k,
                [head_count, head_dimension, width, kv_seq_len, kv_rows],
                if causal { 1 } else { 0 },
                window_size.map(|w| w as i64).unwrap_or(-1),
                step_offset,
                score_scale,
            ));
        }
        Some(self.attention_split(
            q,
            k,
            v,
            [
                kv_seq_len,
                head_count,
                heads,
                head_dimension,
                q_model_dim,
                kv_model_dim,
                kv_seq_len - 1,
                kv_rows,
                width,
                v_model_dim,
            ],
            if causal { 1 } else { 0 },
            window_size.map(|w| w as i64).unwrap_or(-1),
            step_offset,
            score_scale,
        ))
    }

    /// [`Self::attention_scaled`] with V rows `value_dimension` wide per
    /// head (`attention.value_head_dimension`,
    /// `express-latent-attention-forms`): Q and K rows stay `head_dimension`
    /// wide, and the output is `[T, head_count * value_dimension]`. A
    /// distinct value width runs the general Kernels (block, one thread per
    /// head and query, split-key decode); the Tensor Core, tiled and grouped
    /// decode Kernels serve equal widths only.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_valued(
        &self,
        q: &CudaDeviceBuffer,
        k: &CudaDeviceBuffer,
        v: &CudaDeviceBuffer,
        head_count: u64,
        head_dimension: u64,
        value_dimension: Option<u64>,
        kv_head_count: Option<u64>,
        window_size: Option<u64>,
        causal: bool,
        key_count: Option<u64>,
        scale: Option<f32>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        if k.is_tq() || v.is_tq() {
            return self.tq_attention(
                q,
                k,
                v,
                [head_count, head_dimension],
                value_dimension,
                kv_head_count,
                window_size,
                causal,
                key_count,
                scale,
            );
        }
        Self::require_f32(&[q, k, v], "attention")?;
        let value_dimension = value_dimension.unwrap_or(head_dimension);
        let same_width = value_dimension == head_dimension;
        if same_width {
            same_shape_buf(k, v)?;
        }
        // `0` makes every Kernel use 1/sqrt(head_dimension).
        let score_scale = match scale {
            None => 0.0_f32,
            Some(scale) if scale.is_finite() && scale > 0.0 => scale,
            Some(scale) => {
                return Err(CudaError::new(
                    CudaErrorCode::ShapeUnsupported,
                    format!("attention scale {scale} must be finite and positive"),
                ));
            }
        };
        let kv_head_count = kv_head_count.unwrap_or(head_count);
        if head_count == 0 || head_dimension == 0 || kv_head_count == 0 || value_dimension == 0 {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "head_count, kv_head_count, and head_dimension must all be positive",
            ));
        }
        if !head_count.is_multiple_of(kv_head_count) {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "head_count {head_count} must be an exact multiple of kv_head_count {kv_head_count}"
                ),
            ));
        }
        if window_size == Some(0) {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "window_size must be positive; a zero window admits no keys",
            ));
        }
        if window_size.is_some() && !causal {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "window_size is only defined for causal attention",
            ));
        }
        let (seq_len, q_model_dim) = rows_cols_buf(q)?;
        let (kv_rows, kv_model_dim) = rows_cols_buf(k)?;
        let kv_seq_len = key_count.unwrap_or(kv_rows);
        if kv_seq_len > kv_rows {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!("key_count {kv_seq_len} exceeds the {kv_rows} rows of K/V"),
            ));
        }
        if seq_len > kv_seq_len {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "q sequence length {seq_len} cannot exceed k/v sequence length {kv_seq_len}"
                ),
            ));
        }
        if head_count * head_dimension != q_model_dim {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "head_count * head_dimension must equal q row width {q_model_dim}, got {head_count} * {head_dimension}"
                ),
            ));
        }
        if kv_head_count * head_dimension != kv_model_dim {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "kv_head_count * head_dimension must equal k/v row width {kv_model_dim}, got {kv_head_count} * {head_dimension}"
                ),
            ));
        }
        let (v_rows, v_model_dim) = rows_cols_buf(v)?;
        if v_rows != kv_rows || kv_head_count * value_dimension != v_model_dim {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "v must hold {kv_rows} rows of kv_head_count * value_dimension = \
                     {kv_head_count} * {value_dimension}, got {:?}",
                    v.shape
                ),
            ));
        }
        let out_len = usize::try_from(seq_len * head_count * value_dimension).map_err(|_| {
            CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "attention output size overflows",
            )
        })?;
        let out_shape = vec![seq_len, head_count * value_dimension];
        let query_position_offset = kv_seq_len - seq_len;
        let causal_flag: i32 = if causal { 1 } else { 0 };
        let window_arg: i64 = window_size.map(|w| w as i64).unwrap_or(-1);

        if seq_len == 1 {
            let step_offset = match key_count {
                Some(_) => self.advancing_offset("key_count", kv_rows - kv_seq_len),
                None => 0,
            };
            return self.attention_split(
                q,
                k,
                v,
                [
                    kv_seq_len,
                    head_count,
                    kv_head_count,
                    head_dimension,
                    q_model_dim,
                    kv_model_dim,
                    query_position_offset,
                    // Every key a replay may admit has a chunk, so a
                    // captured launch covers every later step.
                    kv_rows,
                    value_dimension,
                    v_model_dim,
                ],
                causal_flag,
                window_arg,
                step_offset,
                score_scale,
            );
        }

        // Several query rows have no replay offset: a captured one (a
        // bucketed prefill, `replay-bucketed-prefill`) replays only at the
        // key count it was captured with.
        if key_count.is_some() {
            self.advancing_offset("key_count", 0);
        }
        // Several query rows, 64- or 128-dimension heads, on a Device with
        // Tensor Cores: `attention_tcp_kernel` / `attention_tcp128_kernel`
        // on K and V split once, when their rows align. Otherwise the tiled
        // Kernel compiled for the head dimension.
        let kv_len = kv_seq_len * kv_model_dim;
        if same_width
            && seq_len >= 16
            && kv_model_dim % TCP_ALIGN == 0
            && kv_len > 0
            && let Some((name, shared_bytes)) = tensor_core_prefill_attention(head_dimension)
            && let Some(func) = self.tensor_function(name)
            && func
                .set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    i32::try_from(shared_bytes).unwrap_or(i32::MAX),
                )
                .is_ok()
            && let Some((_split_lock, terms)) = self.split_bf16_all(&[(k, kv_len), (v, kv_len)])?
        {
            let overflow = || {
                CudaError::new(
                    CudaErrorCode::ShapeUnsupported,
                    "attention launch size overflows a grid dimension",
                )
            };
            let [(k_hi, k_lo), (v_hi, v_lo)] = [terms[0], terms[1]];
            let mut out_dev = self.alloc_output(q.slice.len())?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&q.slice)
                .arg(&k_hi)
                .arg(&k_lo)
                .arg(&v_hi)
                .arg(&v_lo)
                .arg(&mut out_dev)
                .arg(&seq_len)
                .arg(&kv_seq_len)
                .arg(&head_count)
                .arg(&kv_head_count)
                .arg(&q_model_dim)
                .arg(&kv_model_dim)
                .arg(&causal_flag)
                .arg(&window_arg)
                .arg(&query_position_offset)
                .arg(&score_scale);
            let config = LaunchConfig {
                grid_dim: (
                    u32::try_from(seq_len.div_ceil(TCA_QUERIES)).map_err(|_| overflow())?,
                    u32::try_from(head_count).map_err(|_| overflow())?,
                    1,
                ),
                block_dim: (TC_THREADS, 1, 1),
                shared_mem_bytes: shared_bytes,
            };
            unsafe { args.launch(config) }?;
            return Ok(CudaDeviceBuffer {
                slice: out_dev,
                shape: q.shape.clone(),
                bf16: None,
                q8: None,
                tq: None,
            });
        }
        // The tiled Kernel compiled for the head (and value) dimension:
        // equal widths, or an expanded latent attention's 192-dimension
        // keys and 128-dimension values.
        let tiled = match (head_dimension, value_dimension) {
            (4, 4) => Some("attention_tiled_1_kernel"),
            (8, 8) => Some("attention_tiled_2_kernel"),
            (16, 16) => Some("attention_tiled_4_kernel"),
            (32, 32) => Some("attention_tiled_8_kernel"),
            (64, 64) => Some("attention_tiled_16_kernel"),
            (96, 96) => Some("attention_tiled_24_kernel"),
            (128, 128) => Some("attention_tiled_32_kernel"),
            (192, 128) => Some("attention_tiled_48_32_kernel"),
            _ => None,
        };
        if let Some(tiled) = tiled.filter(|_| seq_len > 1) {
            let overflow = || {
                CudaError::new(
                    CudaErrorCode::ShapeUnsupported,
                    "attention launch size overflows a grid dimension",
                )
            };
            let key_tile: u32 = if head_dimension <= 64 { 64 } else { 32 };
            let mut out_dev = self.alloc_output(out_len)?;
            let func = self.function(tiled)?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&q.slice)
                .arg(&k.slice)
                .arg(&v.slice)
                .arg(&mut out_dev)
                .arg(&seq_len)
                .arg(&kv_seq_len)
                .arg(&head_count)
                .arg(&kv_head_count)
                .arg(&q_model_dim)
                .arg(&kv_model_dim)
                .arg(&causal_flag)
                .arg(&window_arg)
                .arg(&query_position_offset)
                .arg(&key_tile)
                .arg(&score_scale);
            let config = LaunchConfig {
                grid_dim: (
                    u32::try_from(seq_len.div_ceil(ATTENTION_TILE_QUERIES))
                        .map_err(|_| overflow())?,
                    u32::try_from(head_count).map_err(|_| overflow())?,
                    1,
                ),
                block_dim: (4 * ATTENTION_TILE_QUERIES as u32, 1, 1),
                shared_mem_bytes: u32::try_from(
                    u64::from(key_tile) * (head_dimension + value_dimension) * 4,
                )
                .map_err(|_| overflow())?,
            };
            unsafe { args.launch(config) }?;
            return Ok(CudaDeviceBuffer {
                slice: out_dev,
                shape: out_shape,
                bf16: None,
                q8: None,
                tq: None,
            });
        }

        // `attention_block_kernel` keeps q, its reduction slots and every
        // score of one (head, query) in shared memory; longer histories
        // than that fits use the one-thread-per-(head, query) kernel.
        // A captured, bounded attention reads its key count from the
        // replay offset, so its shared memory must hold every row of K.
        let step_offset = match key_count {
            Some(_) => self.advancing_offset("key_count", kv_rows - kv_seq_len),
            None => 0,
        };
        let score_rows = if step_offset != 0 {
            kv_rows
        } else {
            kv_seq_len
        };
        let block_shared_bytes = (head_dimension + ATTENTION_THREADS as u64 + score_rows)
            * std::mem::size_of::<f32>() as u64;
        if block_shared_bytes <= ATTENTION_MAX_SHARED_BYTES {
            let blocks = u32::try_from(head_count * seq_len).map_err(|_| {
                CudaError::new(
                    CudaErrorCode::ShapeUnsupported,
                    "attention launch size overflows a grid dimension",
                )
            })?;
            // Writes every element (zero for an empty key window).
            let mut out_dev = self.alloc_output(out_len)?;
            let func = self.function("attention_block_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&q.slice)
                .arg(&k.slice)
                .arg(&v.slice)
                .arg(&mut out_dev)
                .arg(&seq_len)
                .arg(&kv_seq_len)
                .arg(&head_count)
                .arg(&kv_head_count)
                .arg(&head_dimension)
                .arg(&q_model_dim)
                .arg(&kv_model_dim)
                .arg(&causal_flag)
                .arg(&window_arg)
                .arg(&query_position_offset)
                .arg(&step_offset)
                .arg(&score_scale)
                .arg(&value_dimension)
                .arg(&v_model_dim);
            let config = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (ATTENTION_THREADS, 1, 1),
                shared_mem_bytes: block_shared_bytes as u32,
            };
            unsafe { args.launch(config) }?;
            return Ok(CudaDeviceBuffer {
                slice: out_dev,
                shape: out_shape,
                bf16: None,
                q8: None,
                tq: None,
            });
        }
        if self.is_capturing() {
            // The one-thread-per-(head, query) Kernel has no replay offset.
            self.forbid_in_capture("attention over a history this long")?;
        }
        let mut out_dev = self.stream.alloc_zeros::<f32>(out_len)?;
        let func = self.function("attention_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&q.slice)
            .arg(&k.slice)
            .arg(&v.slice)
            .arg(&mut out_dev)
            .arg(&seq_len)
            .arg(&kv_seq_len)
            .arg(&head_count)
            .arg(&kv_head_count)
            .arg(&head_dimension)
            .arg(&q_model_dim)
            .arg(&kv_model_dim)
            .arg(&causal_flag)
            .arg(&window_arg)
            .arg(&query_position_offset)
            .arg(&score_scale)
            .arg(&value_dimension)
            .arg(&v_model_dim);
        unsafe { args.launch(LaunchConfig::for_num_elems((head_count * seq_len) as u32)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: out_shape,
            bf16: None,
            q8: None,
            tq: None,
        })
    }
}

impl CudaKernels {
    /// Merges each head's per-chunk partials (`value_dimension + 2` floats
    /// per chunk) into `out_dev` with `attention_combine_kernel`.
    fn combine_attention_partials(
        &self,
        partial: &CudaSlice<f32>,
        out_dev: &mut CudaSlice<f32>,
        head_count: u64,
        value_dimension: u64,
        chunks: u64,
    ) -> Result<(), CudaError> {
        let overflow = || {
            CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "attention launch size overflows a grid dimension",
            )
        };
        let combine = self.function("attention_combine_kernel")?;
        let weight_bytes = u32::try_from(chunks * 4).unwrap_or(u32::MAX);
        let cache_weights = i32::from(weight_bytes <= ATTENTION_COMBINE_MAX_DYNAMIC_SHARED);
        let mut args = self.stream.launch_builder(&combine);
        // The partials and the output are V-wide: the combine Kernel's
        // width is the value width.
        args.arg(partial)
            .arg(out_dev)
            .arg(&head_count)
            .arg(&value_dimension)
            .arg(&chunks)
            .arg(&cache_weights);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(head_count).map_err(|_| overflow())?, 1, 1),
            // Lanes of chunks per output dimension: up to 8, within 256
            // threads for small heads, one lane past 1024 dimensions.
            block_dim: (
                u32::try_from(if value_dimension > 1024 {
                    1024
                } else {
                    value_dimension * (256 / value_dimension).clamp(1, 8)
                })
                .map_err(|_| overflow())?,
                1,
                1,
            ),
            // One weight per chunk while they fit in shared memory (48 KB
            // less the 4 KB static reduction scratch); past that the
            // Kernel recomputes each weight from its partial.
            shared_mem_bytes: if cache_weights == 1 { weight_bytes } else { 0 },
        };
        unsafe { args.launch(config) }?;
        Ok(())
    }

    /// One query row's attention (a decode step), its keys split into
    /// chunks of [`ATTENTION_CHUNK_KEYS`] scored by separate blocks, then
    /// merged in chunk order (`attention_split_kernel`,
    /// `attention_combine_kernel` in `kernels.cu`). `dims` is the key
    /// count, head count, KV head count, head dimension, q and k row
    /// widths, query position, the K/V row count, and each head's V width
    /// and the V row width.
    #[allow(clippy::too_many_arguments)]
    fn attention_split(
        &self,
        q: &CudaDeviceBuffer,
        k: &CudaDeviceBuffer,
        v: &CudaDeviceBuffer,
        dims: [u64; 10],
        causal_flag: i32,
        window_arg: i64,
        step_offset: u64,
        score_scale: f32,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        let [
            kv_seq_len,
            head_count,
            kv_head_count,
            head_dimension,
            q_model_dim,
            kv_model_dim,
            query_position_offset,
            kv_rows,
            value_dimension,
            v_model_dim,
        ] = dims;
        let overflow = || {
            CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "attention launch size overflows a grid dimension",
            )
        };
        // All query heads of a KV head in one block, reading its K and V
        // once, where the shapes allow it.
        let grouped = grouped_decode_kernel(head_dimension).filter(|_| {
            value_dimension == head_dimension
                && kv_head_count > 0
                && head_count % kv_head_count == 0
                && head_count / kv_head_count <= ATTENTION_GQA_MAX_GROUP
        });
        let chunk_keys = grouped.map_or(ATTENTION_CHUNK_KEYS, |(_, keys)| keys);
        let chunks = kv_rows.div_ceil(chunk_keys).max(1);
        let partial_len =
            usize::try_from(head_count * chunks * (value_dimension + 2)).map_err(|_| overflow())?;
        let mut scratch = self.attention_scratch.lock().unwrap();
        if scratch
            .0
            .as_ref()
            .is_none_or(|slice| slice.len() < partial_len)
        {
            self.forbid_in_capture("growing the attention scratch buffer")?;
            // SAFETY: the split Kernel writes every partial of this launch
            // before the combine Kernel reads it, on this same stream.
            let grown = unsafe { self.stream.alloc::<f32>(partial_len.max(1 << 16)) }?;
            if let Some(outgrown) = scratch.0.replace(grown) {
                scratch.1.push(outgrown);
            }
        }
        let partial = scratch.0.as_mut().expect("allocated above");
        let mut out_dev = self
            .alloc_output(usize::try_from(head_count * value_dimension).map_err(|_| overflow())?)?;
        if let Some((grouped_name, _)) = grouped {
            let split = self.function(grouped_name)?;
            let mut args = self.stream.launch_builder(&split);
            args.arg(&q.slice)
                .arg(&k.slice)
                .arg(&v.slice)
                .arg(&mut *partial)
                .arg(&kv_seq_len)
                .arg(&head_count)
                .arg(&kv_head_count)
                .arg(&kv_model_dim)
                .arg(&causal_flag)
                .arg(&window_arg)
                .arg(&query_position_offset)
                .arg(&chunks)
                .arg(&step_offset)
                .arg(&score_scale);
            let config = LaunchConfig {
                grid_dim: (
                    u32::try_from(chunks).map_err(|_| overflow())?,
                    u32::try_from(kv_head_count).map_err(|_| overflow())?,
                    1,
                ),
                block_dim: (ATTENTION_GQA_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { args.launch(config) }?;
        } else {
            let split = self.function("attention_split_kernel")?;
            let mut args = self.stream.launch_builder(&split);
            args.arg(&q.slice)
                .arg(&k.slice)
                .arg(&v.slice)
                .arg(&mut *partial)
                .arg(&kv_seq_len)
                .arg(&head_count)
                .arg(&kv_head_count)
                .arg(&head_dimension)
                .arg(&q_model_dim)
                .arg(&kv_model_dim)
                .arg(&causal_flag)
                .arg(&window_arg)
                .arg(&query_position_offset)
                .arg(&chunks)
                .arg(&step_offset)
                .arg(&score_scale)
                .arg(&value_dimension)
                .arg(&v_model_dim);
            let config = LaunchConfig {
                grid_dim: (
                    u32::try_from(chunks).map_err(|_| overflow())?,
                    u32::try_from(head_count).map_err(|_| overflow())?,
                    1,
                ),
                block_dim: (ATTENTION_SPLIT_THREADS, 1, 1),
                shared_mem_bytes: u32::try_from(head_dimension * 4).map_err(|_| overflow())?,
            };
            unsafe { args.launch(config) }?;
        }
        self.combine_attention_partials(
            &*partial,
            &mut out_dev,
            head_count,
            value_dimension,
            chunks,
        )?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![1, head_count * value_dimension],
            bf16: None,
            q8: None,
            tq: None,
        })
    }

    /// Head-wise `concat` (`axis = "heads"`, `express-latent-attention-forms`):
    /// `[T, heads * a]` and `[T, second_head_count * b]` into
    /// `[T, heads * (a + b)]`, each output head the first input's head then
    /// the second's (its only head when `second_head_count` is 1).
    pub fn concat_heads(
        &self,
        a: &CudaDeviceBuffer,
        b: &CudaDeviceBuffer,
        head_count: u64,
        second_head_count: u64,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        Self::require_f32(&[a, b], "concat")?;
        let (rows, a_cols) = rows_cols_buf(a)?;
        let (b_rows, b_cols) = rows_cols_buf(b)?;
        let invalid = |reason: String| CudaError::new(CudaErrorCode::ShapeUnsupported, reason);
        if rows != b_rows {
            return Err(invalid(format!(
                "concat across heads needs inputs with the same rows, got {:?} and {:?}",
                a.shape, b.shape
            )));
        }
        let (a_width, b_width) = magnetar_core::operator::concat_heads_widths(
            a_cols,
            b_cols,
            head_count,
            second_head_count,
        )
        .map_err(invalid)?;
        let total = rows
            .checked_mul(head_count * (a_width + b_width))
            .filter(|total| u32::try_from(*total).is_ok())
            .ok_or_else(|| invalid("concat across heads overflows a launch".into()))?;
        let mut out_dev = self.alloc_output(total as usize)?;
        let func = self.function("concat_heads_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&a.slice)
            .arg(&b.slice)
            .arg(&mut out_dev)
            .arg(&rows)
            .arg(&head_count)
            .arg(&a_width)
            .arg(&b_width)
            .arg(&second_head_count);
        unsafe { args.launch(LaunchConfig::for_num_elems(total as u32)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![rows, head_count * (a_width + b_width)],
            bf16: None,
            q8: None,
            tq: None,
        })
    }
}

fn rows_cols_buf(buffer: &CudaDeviceBuffer) -> Result<(u64, u64), CudaError> {
    match buffer.shape.as_slice() {
        [rows, cols] => Ok((*rows, *cols)),
        other => Err(CudaError::new(
            CudaErrorCode::ShapeUnsupported,
            format!("expected rank-2 tensor, got shape {other:?}"),
        )),
    }
}

/// `rope`'s launch size, `rows * head_count * half`, checked rather than a
/// bare `as u32` cast (audit-complet-cuda-hot-path-2026-09-08 P2-2): that
/// product can overflow `u64` before the cast, or be silently truncated by
/// it, for large enough shapes -- today's fixtures are far too small to hit
/// this in practice, which is exactly why it needs an explicit check rather
/// than relying on a test against real hardware to ever notice. A free
/// function (not inlined in `CudaKernels::rope`) so it can be unit-tested
/// directly, without a real CUDA context/device.
/// `moe-experts`' optional forms.
#[derive(Clone, Copy, Default)]
pub struct CudaMoeOptions<'a> {
    /// The experts' gate takes GELU (tanh) instead of SiLU.
    pub gelu_tanh: bool,
    /// `[experts]`: each chosen expert's routing weight is multiplied by
    /// its entry, after the renormalization.
    pub per_expert_scale: Option<&'a CudaDeviceBuffer>,
}

/// Which elements of a RoPE head block form each rotated pair, from the
/// portable `rope` Operator's `rotation_pairing` attribute
/// (`fix-rope-rotation-pairing`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaRopePairing {
    /// Pairs `(2i, 2i + 1)`; the default when the attribute is absent.
    Interleaved,
    /// Pairs `(i, i + dimension / 2)`: Hugging Face `rotate_half`.
    HalfSplit,
}

impl CudaRopePairing {
    /// `None` for an unknown value, which callers reject.
    pub fn from_attribute(value: &str) -> Option<Self> {
        match value {
            "interleaved" => Some(Self::Interleaved),
            "half_split" => Some(Self::HalfSplit),
            _ => None,
        }
    }
}

fn rope_launch_work_items(rows: u64, head_count: u64, half: u64) -> Result<u32, CudaError> {
    let work_items = rows
        .checked_mul(head_count)
        .and_then(|value| value.checked_mul(half))
        .ok_or_else(|| {
            CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "RoPE launch size overflows: rows {rows} * head_count {head_count} * half {half}"
                ),
            )
        })?;
    u32::try_from(work_items).map_err(|_| {
        CudaError::new(
            CudaErrorCode::ShapeUnsupported,
            format!("RoPE launch size {work_items} exceeds u32::MAX work-items"),
        )
    })
}

#[cfg(test)]
mod rope_launch_work_items_tests {
    use super::rope_launch_work_items;
    use crate::error::CudaErrorCode;

    #[test]
    fn accepts_a_realistic_small_shape() {
        let work_items = rope_launch_work_items(2, 4, 8).expect("small shapes never overflow");
        assert_eq!(work_items, 2 * 4 * 8);
    }

    #[test]
    fn rejects_a_u64_multiplication_overflow() {
        let error = rope_launch_work_items(u64::MAX, 2, 2).unwrap_err();
        assert_eq!(error.code, CudaErrorCode::ShapeUnsupported);
    }

    #[test]
    fn rejects_a_product_that_does_not_fit_in_u32() {
        let error = rope_launch_work_items(1 << 20, 1 << 20, 2).unwrap_err();
        assert_eq!(error.code, CudaErrorCode::ShapeUnsupported);
    }
}

#[cfg(test)]
mod rope_pairing_tests {
    use super::CudaRopePairing;

    #[test]
    fn rotation_pairing_attribute_values_parse_and_unknown_values_are_rejected() {
        assert_eq!(
            CudaRopePairing::from_attribute("interleaved"),
            Some(CudaRopePairing::Interleaved)
        );
        assert_eq!(
            CudaRopePairing::from_attribute("half_split"),
            Some(CudaRopePairing::HalfSplit)
        );
        assert_eq!(CudaRopePairing::from_attribute("neox"), None);
    }
}
