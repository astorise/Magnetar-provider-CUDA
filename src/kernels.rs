//! GPU launch wrappers for the `operator-scope` required-now kernel set.
//! Mirrors `providers/cpu`'s free-function signatures and validation
//! (`matmul`, `embedding_lookup`, `rmsnorm`, `rope`, `attention`,
//! `softmax_rows`, `silu`, `add`, `mul`, `residual_add`) so the two are
//! directly comparable in conformance tests, but every method here can fail
//! (allocation, launch, driver errors) in ways a pure host loop cannot --
//! see `CudaError`.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use magnetar_runtime::HostTensor;
use std::sync::Arc;

use crate::error::{CudaError, CudaErrorCode};

const KERNEL_SOURCE: &str = include_str!("kernels.cu");

fn same_shape(a: &HostTensor, b: &HostTensor) -> Result<(), CudaError> {
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
}

impl CudaKernels {
    pub fn compile_and_load(context: &Arc<CudaContext>) -> Result<Self, CudaError> {
        let ptx = compile_ptx(KERNEL_SOURCE)?;
        let module = context.load_module(ptx)?;
        let stream = context.default_stream();
        Ok(Self { stream, module })
    }

    fn function(&self, name: &'static str) -> Result<CudaFunction, CudaError> {
        Ok(self.module.load_function(name)?)
    }

    pub fn add(&self, a: &HostTensor, b: &HostTensor) -> Result<HostTensor, CudaError> {
        same_shape(a, b)?;
        let n = a.data.len() as u64;
        let a_dev = self.stream.clone_htod(&a.data)?;
        let b_dev = self.stream.clone_htod(&b.data)?;
        let mut out_dev = self.stream.alloc_zeros::<f32>(a.data.len())?;
        let func = self.function("add_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&a_dev).arg(&b_dev).arg(&mut out_dev).arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        let out = self.stream.clone_dtoh(&out_dev)?;
        self.stream.synchronize()?;
        HostTensor::new(a.shape.clone(), out).map_err(host_error)
    }

    pub fn mul(&self, a: &HostTensor, b: &HostTensor) -> Result<HostTensor, CudaError> {
        same_shape(a, b)?;
        let n = a.data.len() as u64;
        let a_dev = self.stream.clone_htod(&a.data)?;
        let b_dev = self.stream.clone_htod(&b.data)?;
        let mut out_dev = self.stream.alloc_zeros::<f32>(a.data.len())?;
        let func = self.function("mul_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&a_dev).arg(&b_dev).arg(&mut out_dev).arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        let out = self.stream.clone_dtoh(&out_dev)?;
        self.stream.synchronize()?;
        HostTensor::new(a.shape.clone(), out).map_err(host_error)
    }

    pub fn residual_add(
        &self,
        input: &HostTensor,
        residual: &HostTensor,
    ) -> Result<HostTensor, CudaError> {
        self.add(input, residual)
    }

    pub fn silu(&self, input: &HostTensor) -> Result<HostTensor, CudaError> {
        let n = input.data.len() as u64;
        let in_dev = self.stream.clone_htod(&input.data)?;
        let mut out_dev = self.stream.alloc_zeros::<f32>(input.data.len())?;
        let func = self.function("silu_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&in_dev).arg(&mut out_dev).arg(&n);
        unsafe { args.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        let out = self.stream.clone_dtoh(&out_dev)?;
        self.stream.synchronize()?;
        HostTensor::new(input.shape.clone(), out).map_err(host_error)
    }

    pub fn embedding_lookup(
        &self,
        table: &HostTensor,
        ids: &HostTensor,
    ) -> Result<HostTensor, CudaError> {
        let (vocab, dim) = rows_cols(table)?;
        // Validated on the host, exactly like `providers/cpu::embedding_lookup`,
        // so the error messages/behavior match: the kernel trusts every id it
        // receives is already a valid in-range non-negative integer.
        for &raw_id in &ids.data {
            if raw_id < 0.0 || raw_id.fract() != 0.0 {
                return Err(CudaError::new(
                    CudaErrorCode::ShapeUnsupported,
                    format!("token id {raw_id} is not a non-negative integer"),
                ));
            }
            let id = raw_id as u64;
            if id >= vocab {
                return Err(CudaError::new(
                    CudaErrorCode::ShapeUnsupported,
                    format!("token id {id} exceeds vocabulary size {vocab}"),
                ));
            }
        }
        let num_ids = ids.data.len() as u64;
        let table_dev = self.stream.clone_htod(&table.data)?;
        let ids_dev = self.stream.clone_htod(&ids.data)?;
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(ids.data.len() * dim as usize)?;
        let func = self.function("embedding_lookup_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&table_dev)
            .arg(&ids_dev)
            .arg(&mut out_dev)
            .arg(&dim)
            .arg(&num_ids);
        unsafe { args.launch(LaunchConfig::for_num_elems(num_ids as u32)) }?;
        let out = self.stream.clone_dtoh(&out_dev)?;
        self.stream.synchronize()?;
        HostTensor::new([num_ids, dim], out).map_err(host_error)
    }

    pub fn rmsnorm(
        &self,
        input: &HostTensor,
        weight: &HostTensor,
        epsilon: f32,
    ) -> Result<HostTensor, CudaError> {
        let cols = *input.shape.last().ok_or_else(|| {
            CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                "RMSNorm expects at least one dimension",
            )
        })?;
        if cols == 0 || !input.data.len().is_multiple_of(cols as usize) {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "RMSNorm data length {} is not divisible by hidden dimension {cols}",
                    input.data.len()
                ),
            ));
        }
        let rows = (input.data.len() / cols as usize) as u64;
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
        let in_dev = self.stream.clone_htod(&input.data)?;
        let weight_dev = self.stream.clone_htod(&weight.data)?;
        let mut out_dev = self.stream.alloc_zeros::<f32>(input.data.len())?;
        let func = self.function("rmsnorm_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&in_dev)
            .arg(&weight_dev)
            .arg(&mut out_dev)
            .arg(&rows)
            .arg(&cols)
            .arg(&weight_row_stride)
            .arg(&epsilon);
        unsafe { args.launch(LaunchConfig::for_num_elems(rows as u32)) }?;
        let out = self.stream.clone_dtoh(&out_dev)?;
        self.stream.synchronize()?;
        HostTensor::new(input.shape.clone(), out).map_err(host_error)
    }

    pub fn rope(
        &self,
        input: &HostTensor,
        base: f32,
        scale: f32,
        dimension: u64,
        position_offset: u64,
    ) -> Result<HostTensor, CudaError> {
        let (rows, cols) = rows_cols(input)?;
        if dimension == 0 || !dimension.is_multiple_of(2) || dimension > cols {
            return Err(CudaError::new(
                CudaErrorCode::ShapeUnsupported,
                format!(
                    "RoPE dimension {dimension} must be positive, even, and at most the row width {cols}"
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
        let in_dev = self.stream.clone_htod(&input.data)?;
        let mut out_dev = self.stream.alloc_zeros::<f32>(input.data.len())?;
        let func = self.function("rope_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&in_dev)
            .arg(&mut out_dev)
            .arg(&rows)
            .arg(&cols)
            .arg(&half)
            .arg(&base)
            .arg(&scale)
            .arg(&dimension)
            .arg(&position_offset);
        unsafe { args.launch(LaunchConfig::for_num_elems((rows * half) as u32)) }?;
        let out = self.stream.clone_dtoh(&out_dev)?;
        self.stream.synchronize()?;
        HostTensor::new(input.shape.clone(), out).map_err(host_error)
    }

    pub fn softmax_rows(&self, input: &HostTensor) -> Result<HostTensor, CudaError> {
        let (rows, cols) = rows_cols(input)?;
        let in_dev = self.stream.clone_htod(&input.data)?;
        let mut out_dev = self.stream.alloc_zeros::<f32>(input.data.len())?;
        let mut flag_dev = self.stream.alloc_zeros::<i32>(1)?;
        let func = self.function("softmax_rows_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&in_dev)
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
        let out = self.stream.clone_dtoh(&out_dev)?;
        self.stream.synchronize()?;
        HostTensor::new(input.shape.clone(), out).map_err(host_error)
    }

    pub fn matmul(
        &self,
        a: &HostTensor,
        b: &HostTensor,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<HostTensor, CudaError> {
        let (a_rows, a_cols) = rows_cols(a)?;
        let (b_rows, b_cols) = rows_cols(b)?;
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
        let a_dev = self.stream.clone_htod(&a.data)?;
        let b_dev = self.stream.clone_htod(&b.data)?;
        let mut out_dev = self.stream.alloc_zeros::<f32>((m * n) as usize)?;
        let func = self.function("matmul_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&a_dev)
            .arg(&b_dev)
            .arg(&mut out_dev)
            .arg(&m)
            .arg(&k)
            .arg(&n)
            .arg(&a_row_stride)
            .arg(&a_inner_stride)
            .arg(&b_inner_stride)
            .arg(&b_col_stride);
        unsafe { args.launch(LaunchConfig::for_num_elems((m * n) as u32)) }?;
        let out = self.stream.clone_dtoh(&out_dev)?;
        self.stream.synchronize()?;
        HostTensor::new([m, n], out).map_err(host_error)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: &HostTensor,
        k: &HostTensor,
        v: &HostTensor,
        head_count: u64,
        head_dimension: u64,
        kv_head_count: Option<u64>,
        window_size: Option<u64>,
        causal: bool,
    ) -> Result<HostTensor, CudaError> {
        same_shape(k, v)?;
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
        let (seq_len, q_model_dim) = rows_cols(q)?;
        let (kv_seq_len, kv_model_dim) = rows_cols(k)?;
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

        let q_dev = self.stream.clone_htod(&q.data)?;
        let k_dev = self.stream.clone_htod(&k.data)?;
        let v_dev = self.stream.clone_htod(&v.data)?;
        let mut out_dev = self.stream.alloc_zeros::<f32>(q.data.len())?;
        let func = self.function("attention_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&q_dev)
            .arg(&k_dev)
            .arg(&v_dev)
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
        let out = self.stream.clone_dtoh(&out_dev)?;
        self.stream.synchronize()?;
        HostTensor::new(q.shape.clone(), out).map_err(host_error)
    }
}

fn rows_cols(tensor: &HostTensor) -> Result<(u64, u64), CudaError> {
    match tensor.shape.as_slice() {
        [rows, cols] => Ok((*rows, *cols)),
        other => Err(CudaError::new(
            CudaErrorCode::ShapeUnsupported,
            format!("expected rank-2 tensor, got shape {other:?}"),
        )),
    }
}
