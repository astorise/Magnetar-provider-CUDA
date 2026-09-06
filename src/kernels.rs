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
//! that cross the host/device boundary, and only [`CudaExecutor`]
//! (`executor.rs`) calls them -- when a resource is already device-resident
//! under its existing `TensorResourceId`, the executor reuses the stored
//! [`CudaDeviceBuffer`] directly instead of downloading and re-uploading it.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use magnetar_runtime::HostTensor;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{CudaError, CudaErrorCode};

const KERNEL_SOURCE: &str = include_str!("kernels.cu");

/// A tensor resident entirely in this Provider's device memory, persisting
/// across separate Kernel invocations (design.md's "device allocation table
/// keyed by `TensorResourceId`" decision) instead of only for the duration
/// of one upload-compute-download call.
pub struct CudaDeviceBuffer {
    pub(crate) slice: CudaSlice<f32>,
    pub shape: Vec<u64>,
}

impl CudaDeviceBuffer {
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }
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

fn host_error(error: magnetar_runtime::ReferenceCpuError) -> CudaError {
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
    /// Counts real host<->device crossings (`upload`/`download` calls
    /// only, never a Kernel launch itself) -- used by tests to prove two
    /// chained Kernel invocations do not round-trip through the host
    /// (`enable-device-resident-kernel-chaining` task 3.2), not for any
    /// production decision.
    upload_count: AtomicU64,
    download_count: AtomicU64,
}

impl CudaKernels {
    pub fn compile_and_load(context: &Arc<CudaContext>) -> Result<Self, CudaError> {
        let ptx = compile_ptx(KERNEL_SOURCE)?;
        let module = context.load_module(ptx)?;
        let stream = context.default_stream();
        Ok(Self {
            stream,
            module,
            upload_count: AtomicU64::new(0),
            download_count: AtomicU64::new(0),
        })
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

    fn function(&self, name: &'static str) -> Result<CudaFunction, CudaError> {
        Ok(self.module.load_function(name)?)
    }

    /// Uploads a host tensor into a fresh device allocation. The only
    /// host-to-device crossing point in this module -- callers (the
    /// executor's device allocation table) invoke this only when a
    /// resource does not already have a live device buffer.
    pub fn upload(&self, tensor: &HostTensor) -> Result<CudaDeviceBuffer, CudaError> {
        let slice = self.stream.clone_htod(&tensor.data)?;
        self.upload_count.fetch_add(1, Ordering::Relaxed);
        Ok(CudaDeviceBuffer {
            slice,
            shape: tensor.shape.clone(),
        })
    }

    /// Downloads a device buffer to host-visible bytes. The only
    /// device-to-host crossing point in this module -- callers invoke this
    /// only when host-visible bytes are genuinely requested (`read_tensor`,
    /// or `TensorValue::into_host` at a real materialization boundary), not
    /// as an automatic step of every Kernel invocation.
    pub fn download(&self, buffer: &CudaDeviceBuffer) -> Result<HostTensor, CudaError> {
        let data = self.stream.clone_dtoh(&buffer.slice)?;
        self.stream.synchronize()?;
        self.download_count.fetch_add(1, Ordering::Relaxed);
        HostTensor::new(buffer.shape.clone(), data).map_err(host_error)
    }

    pub fn add(
        &self,
        a: &CudaDeviceBuffer,
        b: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        same_shape_buf(a, b)?;
        let n = a.slice.len() as u64;
        let mut out_dev = self.stream.alloc_zeros::<f32>(a.slice.len())?;
        let func = self.function("add_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&a.slice).arg(&b.slice).arg(&mut out_dev).arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: a.shape.clone(),
        })
    }

    pub fn mul(
        &self,
        a: &CudaDeviceBuffer,
        b: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        same_shape_buf(a, b)?;
        let n = a.slice.len() as u64;
        let mut out_dev = self.stream.alloc_zeros::<f32>(a.slice.len())?;
        let func = self.function("mul_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&a.slice).arg(&b.slice).arg(&mut out_dev).arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: a.shape.clone(),
        })
    }

    pub fn residual_add(
        &self,
        input: &CudaDeviceBuffer,
        residual: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.add(input, residual)
    }

    pub fn silu(&self, input: &CudaDeviceBuffer) -> Result<CudaDeviceBuffer, CudaError> {
        let n = input.slice.len() as u64;
        let mut out_dev = self.stream.alloc_zeros::<f32>(input.slice.len())?;
        let func = self.function("silu_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&input.slice).arg(&mut out_dev).arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: input.shape.clone(),
        })
    }

    pub fn embedding_lookup(
        &self,
        table: &CudaDeviceBuffer,
        ids: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        let (vocab, dim) = rows_cols_buf(table)?;
        let num_ids = ids.slice.len() as u64;
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(ids.slice.len() * dim as usize)?;
        let mut invalid_id_flag = self.stream.alloc_zeros::<i32>(1)?;
        let func = self.function("embedding_lookup_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&table.slice)
            .arg(&ids.slice)
            .arg(&mut out_dev)
            .arg(&dim)
            .arg(&num_ids)
            .arg(&vocab)
            .arg(&mut invalid_id_flag);
        unsafe { args.launch(LaunchConfig::for_num_elems(num_ids as u32)) }?;
        // Data-dependent failure (an out-of-range or non-integer token id)
        // can only be observed by downloading this one-element flag -- the
        // same device-computed-flag pattern `softmax_rows` already uses
        // below, not a blanket per-kernel host round-trip.
        let flag = self.stream.clone_dtoh(&invalid_id_flag)?;
        self.stream.synchronize()?;
        if flag[0] != 0 {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "embedding lookup id is not a valid in-range non-negative integer",
            ));
        }
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![num_ids, dim],
        })
    }

    pub fn rmsnorm(
        &self,
        input: &CudaDeviceBuffer,
        weight: &CudaDeviceBuffer,
        epsilon: f32,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        let cols = *input.shape.last().ok_or_else(|| {
            CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "RMSNorm expects at least one dimension",
            )
        })?;
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
        let weight_row_stride = if weight.shape == [cols] || weight.shape == [1, cols] {
            0u64
        } else if weight.shape == input.shape || weight.shape == [rows, cols] {
            cols
        } else {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "RMSNorm weight shape must be [{cols}], [1, {cols}], input shape {:?}, or [{rows}, {cols}], got {:?}",
                    input.shape, weight.shape
                ),
            ));
        };
        if epsilon <= 0.0 {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "RMSNorm epsilon must be positive",
            ));
        }
        let mut out_dev = self.stream.alloc_zeros::<f32>(input.slice.len())?;
        let func = self.function("rmsnorm_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&input.slice)
            .arg(&weight.slice)
            .arg(&mut out_dev)
            .arg(&rows)
            .arg(&cols)
            .arg(&weight_row_stride)
            .arg(&epsilon);
        unsafe { args.launch(LaunchConfig::for_num_elems(rows as u32)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: input.shape.clone(),
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
        let mut out_dev = self.stream.clone_dtod(&input.slice)?;
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
            .arg(&head_count);
        unsafe {
            args.launch(LaunchConfig::for_num_elems(
                (rows * head_count * half) as u32,
            ))
        }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: input.shape.clone(),
        })
    }

    pub fn softmax_rows(&self, input: &CudaDeviceBuffer) -> Result<CudaDeviceBuffer, CudaError> {
        let (rows, cols) = rows_cols_buf(input)?;
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
        })
    }

    pub fn matmul(
        &self,
        a: &CudaDeviceBuffer,
        b: &CudaDeviceBuffer,
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
        let mut out_dev = self.stream.alloc_zeros::<f32>((m * n) as usize)?;
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
            .arg(&b_col_stride);
        unsafe { args.launch(LaunchConfig::for_num_elems((m * n) as u32)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: vec![m, n],
        })
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
        same_shape_buf(k, v)?;
        let kv_head_count = kv_head_count.unwrap_or(head_count);
        if head_count == 0 || head_dimension == 0 || kv_head_count == 0 {
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
        let (kv_seq_len, kv_model_dim) = rows_cols_buf(k)?;
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
        let query_position_offset = kv_seq_len - seq_len;
        let causal_flag: i32 = if causal { 1 } else { 0 };
        let window_arg: i64 = window_size.map(|w| w as i64).unwrap_or(-1);

        let mut out_dev = self.stream.alloc_zeros::<f32>(q.slice.len())?;
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
            .arg(&query_position_offset);
        unsafe { args.launch(LaunchConfig::for_num_elems((head_count * seq_len) as u32)) }?;
        Ok(CudaDeviceBuffer {
            slice: out_dev,
            shape: q.shape.clone(),
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
