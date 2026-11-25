//! Qwen3.5's forms (`support-qwen38-flash-next`): the per-head `split` of
//! `q_proj` and the gated-delta linear-attention mixer. The mixer mirrors
//! `magnetar_provider_api::operator_semantics::gated_delta_mixer` step by
//! step: projections against the checkpoint-order `[out, in]` weights
//! ([`CudaKernels::project`]), then one Kernel per stage.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::{CudaDeviceBuffer, CudaKernels};
use crate::error::{CudaError, CudaErrorCode};

/// The longest convolution history `gdn_conv_kernel` keeps
/// (`GDN_MAX_KERNEL` in `kernels.cu`).
pub const GATED_DELTA_MAX_CONV_KERNEL: u64 = 8;
/// The largest key head dimension `gdn_delta_kernel` keeps in registers
/// (`GDN_MAX_KEY_DIM` in `kernels.cu`).
pub const GATED_DELTA_MAX_KEY_DIM: u64 = 128;

/// A `gated-delta-mixer` node's geometry, from its attributes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedDeltaGeometry {
    pub key_heads: u64,
    pub value_heads: u64,
    pub key_dim: u64,
    pub value_dim: u64,
    pub conv_kernel: u64,
    pub epsilon: f32,
    /// The output gate is `sigmoid(z)` rather than `silu(z)`
    /// (`output_gate_type`).
    pub sigmoid_gate: bool,
}

fn shape_error(detail: impl Into<String>) -> CudaError {
    CudaError::new(CudaErrorCode::ShapeUnsupported, detail)
}

fn buffer(slice: cudarc::driver::CudaSlice<f32>, shape: Vec<u64>) -> CudaDeviceBuffer {
    CudaDeviceBuffer {
        slice,
        shape,
        bf16: None,
        q8: None,
        tq: None,
    }
}

/// Output columns (warps) and rows of `a` one `gdn_project_kernel` block
/// serves (`GDN_PROJECT_WARPS`, `GDN_PROJECT_ROWS` in `kernels.cu`).
const PROJECT_WARPS: u64 = 8;
const PROJECT_ROWS: u64 = 8;
/// Threads per `gdn_delta_kernel` block (`GDN_DELTA_THREADS`): its
/// register-held state column rules out larger blocks.
const DELTA_THREADS: u64 = 64;

impl CudaKernels {
    /// `x · weightᵀ` for a `[n, k]` weight in checkpoint `[out, in]` order,
    /// f32 or compact: one warp per output column reads its contiguous
    /// weight row (the generic matmul would read it with a stride of `k`).
    /// A packed weight takes the matmul's own path.
    fn project(
        &self,
        x: &CudaDeviceBuffer,
        weight: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        let ([m, k], [n, k2]) = (&x.shape[..], &weight.shape[..]) else {
            return Err(shape_error("a projection needs rank-two operands"));
        };
        let (m, k, n) = (*m, *k, *n);
        if weight.is_q8() || weight.is_tq() || m.div_ceil(PROJECT_ROWS) > 65_535 {
            return self.matmul(x, weight, false, true);
        }
        if k != *k2 {
            return Err(shape_error(format!(
                "projection inner dimension mismatch: {k} vs {k2}"
            )));
        }
        Self::require_f32(&[x], "a projection")?;
        let weight16 = self.bf16_address(weight);
        let mut out = self.alloc_output((m * n) as usize)?;
        let func = self.function("gdn_project_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&x.slice)
            .arg(&weight.slice)
            .arg(&weight16)
            .arg(&mut out)
            .arg(&m)
            .arg(&k)
            .arg(&n);
        let config = LaunchConfig {
            grid_dim: (
                n.div_ceil(PROJECT_WARPS) as u32,
                m.div_ceil(PROJECT_ROWS) as u32,
                1,
            ),
            block_dim: (32, PROJECT_WARPS as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(config) }?;
        Ok(buffer(out, vec![m, n]))
    }

    /// `split` (`split-per-head`): the last dimension divides into
    /// `head_count` equal blocks; the first half of every block goes to the
    /// first output and the second half to the second, block after block.
    /// `head_count = 1` halves the whole last dimension.
    pub fn split_halves(
        &self,
        input: &CudaDeviceBuffer,
        head_count: u64,
    ) -> Result<[CudaDeviceBuffer; 2], CudaError> {
        Self::require_f32(&[input], "split")?;
        let Some((&width, outer)) = input.shape.split_last() else {
            return Err(shape_error("split needs at least one dimension"));
        };
        if head_count == 0 || width == 0 || !width.is_multiple_of(2 * head_count) {
            return Err(shape_error(format!(
                "split of width {width} across {head_count} head(s) needs a multiple of {}",
                2 * head_count
            )));
        }
        let half = width / head_count / 2;
        let blocks = input.slice.len() as u64 / (2 * half);
        let len = (blocks * half) as usize;
        let mut left = self.alloc_output(len)?;
        let mut right = self.alloc_output(len)?;
        let func = self.function("split_halves_kernel")?;
        let mut args = self.stream.launch_builder(&func);
        args.arg(&input.slice)
            .arg(&mut left)
            .arg(&mut right)
            .arg(&blocks)
            .arg(&half);
        unsafe { args.launch(LaunchConfig::for_num_elems(len as u32)) }?;
        let mut shape = outer.to_vec();
        shape.push(width / 2);
        Ok([buffer(left, shape.clone()), buffer(right, shape)])
    }

    /// `gated-delta-mixer`. `inputs` in the Operator's order: hidden
    /// `[T, H]`, convolution state `[C, K]`, recurrent state `[Vh, kd,
    /// vd]`, `in_proj_qkv` `[C, H]`, `in_proj_z` `[V, H]`, `in_proj_a` and
    /// `in_proj_b` `[Vh, H]`, the depthwise convolution `[C, K]`, `A_log`
    /// and `dt_bias` `[Vh]`, the norm `[vd]`, `out_proj` `[H, V]`, and an
    /// optional 0/1 token mask `[T]`. Returns the hidden output `[T, H]`
    /// and the next convolution and recurrent states.
    pub fn gated_delta_mixer(
        &self,
        inputs: &[&CudaDeviceBuffer],
        geometry: GatedDeltaGeometry,
    ) -> Result<[CudaDeviceBuffer; 3], CudaError> {
        let GatedDeltaGeometry {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            conv_kernel,
            epsilon,
            sigmoid_gate,
        } = geometry;
        if !(12..=13).contains(&inputs.len()) {
            return Err(shape_error(
                "gated-delta mixer requires twelve or thirteen inputs",
            ));
        }
        if key_heads == 0
            || value_heads == 0
            || !value_heads.is_multiple_of(key_heads)
            || key_dim == 0
            || value_dim == 0
            || conv_kernel == 0
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(shape_error("invalid gated-delta geometry"));
        }
        if conv_kernel > GATED_DELTA_MAX_CONV_KERNEL || key_dim > GATED_DELTA_MAX_KEY_DIM {
            return Err(shape_error(format!(
                "the CUDA gated-delta mixer keeps at most {GATED_DELTA_MAX_CONV_KERNEL} \
                 convolution taps and {GATED_DELTA_MAX_KEY_DIM} key dimensions, got \
                 {conv_kernel} and {key_dim}"
            )));
        }
        let [tokens, hidden] = inputs[0].shape[..] else {
            return Err(shape_error("gated-delta hidden input must be rank two"));
        };
        let key_width = key_heads * key_dim;
        let value_width = value_heads * value_dim;
        let channels = 2 * key_width + value_width;
        let expected: [&[u64]; 13] = [
            &[tokens, hidden],
            &[channels, conv_kernel],
            &[value_heads, key_dim, value_dim],
            &[channels, hidden],
            &[value_width, hidden],
            &[value_heads, hidden],
            &[value_heads, hidden],
            &[channels, conv_kernel],
            &[value_heads],
            &[value_heads],
            &[value_dim],
            &[hidden, value_width],
            &[tokens],
        ];
        for (index, (input, shape)) in inputs.iter().zip(expected).enumerate() {
            if input.shape != shape {
                return Err(shape_error(format!(
                    "gated-delta input {index} has shape {:?}, expected {shape:?}",
                    input.shape
                )));
            }
        }
        // The projections read the weights (compact or packed) directly;
        // every other operand is read as f32.
        Self::require_f32(
            &[
                inputs[0], inputs[1], inputs[2], inputs[7], inputs[8], inputs[9], inputs[10],
            ],
            "gated-delta-mixer",
        )?;
        if let Some(mask) = inputs.get(12) {
            Self::require_f32(&[mask], "gated-delta-mixer")?;
        }
        let masked;
        let x = match inputs.get(12) {
            Some(mask) => {
                let mut out = self.alloc_output(inputs[0].slice.len())?;
                let func = self.function("mask_rows_kernel")?;
                let mut args = self.stream.launch_builder(&func);
                args.arg(&inputs[0].slice)
                    .arg(&mask.slice)
                    .arg(&mut out)
                    .arg(&tokens)
                    .arg(&hidden);
                unsafe { args.launch(LaunchConfig::for_num_elems((tokens * hidden) as u32)) }?;
                masked = buffer(out, vec![tokens, hidden]);
                &masked
            }
            None => inputs[0],
        };
        let projected = self.project(x, inputs[3])?;
        let z = self.project(x, inputs[4])?;
        let a = self.project(x, inputs[5])?;
        let b = self.project(x, inputs[6])?;

        // The causal depthwise convolution and the next convolution state.
        let mut convolution_state = self.alloc_output((channels * conv_kernel) as usize)?;
        let mut convolved = self.alloc_output((tokens * channels) as usize)?;
        {
            let func = self.function("gdn_conv_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&projected.slice)
                .arg(&inputs[1].slice)
                .arg(&inputs[7].slice)
                .arg(&mut convolution_state)
                .arg(&mut convolved)
                .arg(&tokens)
                .arg(&channels)
                .arg(&conv_kernel);
            unsafe { args.launch(LaunchConfig::for_num_elems(channels as u32)) }?;
        }

        // Per-token, per-value-head decay and beta.
        let gate_len = tokens * value_heads;
        let mut decay = self.alloc_output(gate_len as usize)?;
        let mut beta = self.alloc_output(gate_len as usize)?;
        {
            let func = self.function("gdn_gates_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&a.slice)
                .arg(&b.slice)
                .arg(&inputs[8].slice)
                .arg(&inputs[9].slice)
                .arg(&mut decay)
                .arg(&mut beta)
                .arg(&tokens)
                .arg(&value_heads);
            unsafe { args.launch(LaunchConfig::for_num_elems(gate_len as u32)) }?;
        }

        // L2-normalized queries (scaled by 1/sqrt(key_dim)) and keys.
        let mut query = self.alloc_output((tokens * key_width) as usize)?;
        let mut key = self.alloc_output((tokens * key_width) as usize)?;
        {
            let func = self.function("gdn_qk_norm_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&convolved)
                .arg(&mut query)
                .arg(&mut key)
                .arg(&tokens)
                .arg(&channels)
                .arg(&key_heads)
                .arg(&key_dim);
            unsafe { args.launch(LaunchConfig::for_num_elems((tokens * key_heads) as u32)) }?;
        }

        // The gated delta rule and the next recurrent state.
        let mut recurrent_state =
            self.alloc_output((value_heads * key_dim * value_dim) as usize)?;
        let mut values = self.alloc_output((tokens * value_width) as usize)?;
        {
            let func = self.function("gdn_delta_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&query)
                .arg(&key)
                .arg(&convolved)
                .arg(&decay)
                .arg(&beta)
                .arg(&inputs[2].slice)
                .arg(&mut recurrent_state)
                .arg(&mut values)
                .arg(&tokens)
                .arg(&channels)
                .arg(&key_heads)
                .arg(&value_heads)
                .arg(&key_dim)
                .arg(&value_dim);
            let config = LaunchConfig {
                grid_dim: (value_width.div_ceil(DELTA_THREADS) as u32, 1, 1),
                block_dim: (DELTA_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { args.launch(config) }?;
        }

        // The gated RMSNorm, then the output projection.
        let rows = tokens * value_heads;
        let sigmoid_flag = i32::from(sigmoid_gate);
        let mut normalized = self.alloc_output((tokens * value_width) as usize)?;
        {
            let func = self.function("gdn_gated_norm_kernel")?;
            let mut args = self.stream.launch_builder(&func);
            args.arg(&values)
                .arg(&z.slice)
                .arg(&inputs[10].slice)
                .arg(&mut normalized)
                .arg(&rows)
                .arg(&value_dim)
                .arg(&epsilon)
                .arg(&sigmoid_flag);
            unsafe { args.launch(LaunchConfig::for_num_elems(rows as u32)) }?;
        }
        let normalized = buffer(normalized, vec![tokens, value_width]);
        let output = self.project(&normalized, inputs[11])?;
        Ok([
            output,
            buffer(convolution_state, vec![channels, conv_kernel]),
            buffer(recurrent_state, vec![value_heads, key_dim, value_dim]),
        ])
    }
}
