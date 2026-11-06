//! Numerical conformance: `CudaKernels` output checked against
//! `providers/cpu`'s reference functions on the same small fixtures
//! (`provider-roadmap`'s "Reference CPU Remains Correctness Baseline" /
//! `cuda-provider`'s "CUDA Kernels Match Reference CPU Semantics").
//!
//! Every test here skips cleanly (returns without assertions) when no
//! compatible CUDA driver/device is present, rather than failing --
//! `cuda-provider`'s conformance scope is explicitly hardware-gated (see
//! `design.md`'s "graceful unavailability" decision and tasks.md 9.3).
//!
//! Each test uploads its `HostTensor` fixtures once via
//! [`CudaKernels::upload`] and downloads the kernel's output once via
//! [`CudaKernels::download`] -- the same two host/device crossing points
//! `CudaExecutor` itself uses, not an implicit per-kernel round trip.

use magnetar_runtime::HostTensor;

use crate::half_precision;
use crate::kernels::{CudaDeviceBuffer, CudaHalfDType, CudaKernels};
use crate::provider::CudaProvider;

const TOLERANCE: f32 = 1e-3;

fn kernels_or_skip() -> Option<CudaKernels> {
    let provider = CudaProvider::new();
    let context = provider.context()?;
    Some(CudaKernels::compile_and_load(&context).expect(
        "kernel compilation must succeed on a machine that already passed device discovery",
    ))
}

fn upload(kernels: &CudaKernels, tensor: &HostTensor) -> CudaDeviceBuffer {
    kernels.upload(tensor).expect("upload must succeed")
}

fn download(kernels: &CudaKernels, buffer: &CudaDeviceBuffer) -> HostTensor {
    kernels.download(buffer).expect("download must succeed")
}

fn assert_close(actual: &HostTensor, expected: &HostTensor) {
    assert_eq!(
        actual.shape, expected.shape,
        "shape mismatch: {:?} vs {:?}",
        actual.shape, expected.shape
    );
    for (index, (a, e)) in actual.data.iter().zip(&expected.data).enumerate() {
        assert!(
            (a - e).abs() <= TOLERANCE,
            "element {index}: cuda={a} reference-cpu={e} exceeds tolerance {TOLERANCE}"
        );
    }
}

#[test]
fn add_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a = HostTensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = HostTensor::new([2, 3], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
    let expected = magnetar_provider_cpu::add(&a, &b).unwrap();
    let actual_dev = kernels
        .add(&upload(&kernels, &a), &upload(&kernels, &b))
        .unwrap();
    assert_close(&download(&kernels, &actual_dev), &expected);
}

/// Real Qwen2/2.5 QKV projection bias: `[rows, cols] + [cols]`, the bias
/// row broadcast across every row -- `providers/cpu::add`'s own broadcast
/// extension, dispatched here to CUDA's separate `bias_add_kernel`.
#[test]
fn add_broadcasts_a_bias_row_matching_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a = HostTensor::new([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let bias = HostTensor::new([2], [100.0, 1000.0]).unwrap();
    let expected = magnetar_provider_cpu::add(&a, &bias).unwrap();
    let actual_dev = kernels
        .add(&upload(&kernels, &a), &upload(&kernels, &bias))
        .unwrap();
    assert_close(&download(&kernels, &actual_dev), &expected);
}

/// KV-history concatenation's exact contract
/// (`implement-device-resident-multi-step-cuda-decode`): stacking a
/// growing historical K/V (`a`) above this step's newly computed K/V
/// (`b`), both real Device-resident CUDA buffers, matching
/// `magnetar_provider_cpu::concat`'s output exactly.
#[test]
fn concat_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a = HostTensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = HostTensor::new([1, 3], [7.0, 8.0, 9.0]).unwrap();
    let expected = magnetar_provider_cpu::concat(&a, &b).unwrap();
    let actual_dev = kernels
        .concat(&upload(&kernels, &a), &upload(&kernels, &b))
        .unwrap();
    assert_close(&download(&kernels, &actual_dev), &expected);
}

#[test]
fn mul_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a = HostTensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = HostTensor::new([2, 3], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
    let expected = magnetar_provider_cpu::mul(&a, &b).unwrap();
    let actual_dev = kernels
        .mul(&upload(&kernels, &a), &upload(&kernels, &b))
        .unwrap();
    assert_close(&download(&kernels, &actual_dev), &expected);
}

#[test]
fn silu_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let input = HostTensor::new([2, 3], [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0]).unwrap();
    let expected = magnetar_provider_cpu::silu(&input);
    let actual_dev = kernels.silu(&upload(&kernels, &input)).unwrap();
    assert_close(&download(&kernels, &actual_dev), &expected);
}

#[test]
fn softmax_rows_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let input = HostTensor::new([2, 3], [1.0, 2.0, 3.0, -1.0, 0.0, 1.0]).unwrap();
    let expected = magnetar_provider_cpu::softmax_rows(&input).unwrap();
    let actual_dev = kernels.softmax_rows(&upload(&kernels, &input)).unwrap();
    assert_close(&download(&kernels, &actual_dev), &expected);
}

#[test]
fn embedding_lookup_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let table = HostTensor::new(
        [4, 3],
        [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2],
    )
    .unwrap();
    let ids = HostTensor::new([3], [0.0, 2.0, 1.0]).unwrap();
    let expected = magnetar_provider_cpu::embedding_lookup(&table, &ids).unwrap();
    let actual_dev = kernels
        .embedding_lookup(&upload(&kernels, &table), &upload(&kernels, &ids))
        .unwrap();
    assert_close(&download(&kernels, &actual_dev), &expected);
}

#[test]
fn embedding_lookup_rejects_out_of_range_id_before_dispatch() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let table = HostTensor::new([2, 2], [0.0, 0.0, 0.0, 0.0]).unwrap();
    let ids = HostTensor::new([1], [5.0]).unwrap();
    assert!(
        kernels
            .embedding_lookup(&upload(&kernels, &table), &upload(&kernels, &ids))
            .is_err()
    );
}

#[test]
fn rmsnorm_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let input = HostTensor::new([2, 3], [1.0, 2.0, 3.0, -1.0, 0.5, 2.0]).unwrap();
    let weight = HostTensor::new([3], [1.0, 0.5, 2.0]).unwrap();
    let expected = magnetar_provider_cpu::rmsnorm(&input, &weight, 1e-5).unwrap();
    let actual_dev = kernels
        .rmsnorm(&upload(&kernels, &input), &upload(&kernels, &weight), 1e-5)
        .unwrap();
    assert_close(&download(&kernels, &actual_dev), &expected);
}

#[test]
fn rope_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let input = HostTensor::new([2, 4], [1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, -0.5]).unwrap();
    let expected = magnetar_provider_cpu::rope(&input, 10000.0, 1.0, 4, 0, 1).unwrap();
    let actual_dev = kernels
        .rope(&upload(&kernels, &input), 10000.0, 1.0, 4, 0, 1)
        .unwrap();
    assert_close(&download(&kernels, &actual_dev), &expected);
}

/// `make-first-native-cuda-hot-path-device-resident` task 3.8: a genuine
/// multi-head call (`head_count > 1`, `dimension == head_width`) must
/// match `providers/cpu`'s corrected implementation on real hardware, not
/// just the single-block case above.
#[test]
fn rope_multi_head_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    // 2 rows, head_count = 4, head_width = 2 -> cols = 8.
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.25 - 2.0).collect();
    let input = HostTensor::new([2, 8], data).unwrap();
    let expected = magnetar_provider_cpu::rope(&input, 10000.0, 1.0, 2, 3, 4).unwrap();
    let actual_dev = kernels
        .rope(&upload(&kernels, &input), 10000.0, 1.0, 2, 3, 4)
        .unwrap();
    assert_close(&download(&kernels, &actual_dev), &expected);
}

/// Partial RoPE on real hardware: `dimension < head_width` must rotate
/// only the first `dimension` columns of each head's block, matching
/// `providers/cpu`'s corrected implementation -- including the untouched
/// tail, which the kernel now preserves via a device-to-device copy seed
/// rather than a zero-allocation.
#[test]
fn rope_partial_rotation_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    // 2 rows, head_count = 2, head_width = 4, dimension = 2 (partial).
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 - 0.8).collect();
    let input = HostTensor::new([2, 8], data).unwrap();
    let expected = magnetar_provider_cpu::rope(&input, 10000.0, 1.0, 2, 0, 2).unwrap();
    let actual_dev = kernels
        .rope(&upload(&kernels, &input), 10000.0, 1.0, 2, 0, 2)
        .unwrap();
    assert_close(&download(&kernels, &actual_dev), &expected);
}

/// GQA-shaped case on real hardware: a smaller `head_count` (as K would
/// have relative to Q under grouped-query attention) must also match
/// `providers/cpu`'s corrected implementation, independently of any other
/// `head_count`.
#[test]
fn rope_gqa_shaped_head_count_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    // 2 rows, head_count = 2 (e.g. kv_head_count), head_width = 4.
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.05).collect();
    let input = HostTensor::new([2, 8], data).unwrap();
    let expected = magnetar_provider_cpu::rope(&input, 10000.0, 1.0, 4, 1, 2).unwrap();
    let actual_dev = kernels
        .rope(&upload(&kernels, &input), 10000.0, 1.0, 4, 1, 2)
        .unwrap();
    assert_close(&download(&kernels, &actual_dev), &expected);
}

#[test]
fn matmul_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a = HostTensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = HostTensor::new([3, 2], [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let expected = magnetar_provider_cpu::matmul(&a, &b, false, false).unwrap();
    let actual_dev = kernels
        .matmul(&upload(&kernels, &a), &upload(&kernels, &b), false, false)
        .unwrap();
    assert_close(&download(&kernels, &actual_dev), &expected);
}

#[test]
fn matmul_transposed_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a = HostTensor::new([3, 2], [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    let b = HostTensor::new([3, 2], [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let expected = magnetar_provider_cpu::matmul(&a, &b, true, false).unwrap();
    let actual_dev = kernels
        .matmul(&upload(&kernels, &a), &upload(&kernels, &b), true, false)
        .unwrap();
    assert_close(&download(&kernels, &actual_dev), &expected);
}

/// `matmul_bias` (the Runtime's fused `matmul-bias`) against providers/cpu,
/// and bit-identical to this Provider's own `matmul` followed by `add`, on
/// the split-K path and the transposed fallback path.
#[test]
fn matmul_bias_matches_reference_cpu_and_unfused_device_result() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let values = |count: u64, seed: u64| -> Vec<f32> {
        (0..count)
            .map(|i| (((i * 7919 + seed * 104_729) % 2001) as f32 - 1000.0) * 1e-4)
            .collect()
    };
    for (m, k, n, transpose_a) in [(1, 1000, 70, false), (3, 129, 65, false), (4, 50, 40, true)] {
        let a = if transpose_a {
            HostTensor::new([k, m], values(m * k, 1)).unwrap()
        } else {
            HostTensor::new([m, k], values(m * k, 1)).unwrap()
        };
        let b = HostTensor::new([k, n], values(k * n, 2)).unwrap();
        let bias = HostTensor::new([n], values(n, 3)).unwrap();
        let expected = magnetar_provider_cpu::add(
            &magnetar_provider_cpu::matmul(&a, &b, transpose_a, false).unwrap(),
            &bias,
        )
        .unwrap();
        let (a_dev, b_dev, bias_dev) = (
            upload(&kernels, &a),
            upload(&kernels, &b),
            upload(&kernels, &bias),
        );
        let fused = download(
            &kernels,
            &kernels
                .matmul_bias(&a_dev, &b_dev, &bias_dev, transpose_a, false)
                .unwrap(),
        );
        assert_close(&fused, &expected);
        let unfused = download(
            &kernels,
            &kernels
                .add(
                    &kernels.matmul(&a_dev, &b_dev, transpose_a, false).unwrap(),
                    &bias_dev,
                )
                .unwrap(),
        );
        assert_eq!(
            fused.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            unfused.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}

/// `matmul_residual` and `silu_mul` (the Runtime's fused `matmul-residual`
/// and `silu-mul`) against providers/cpu, and bit-identical to this
/// Provider's own unfused pairs.
#[test]
fn matmul_residual_and_silu_mul_match_reference_cpu_and_unfused_device_results() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let values = |count: u64, seed: u64| -> Vec<f32> {
        (0..count)
            .map(|i| (((i * 7919 + seed * 104_729) % 2001) as f32 - 1000.0) * 1e-4)
            .collect()
    };
    let bits = |tensor: &HostTensor| tensor.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for (m, k, n, transpose_a) in [(1, 1000, 70, false), (3, 129, 65, false), (4, 50, 40, true)] {
        let a = if transpose_a {
            HostTensor::new([k, m], values(m * k, 1)).unwrap()
        } else {
            HostTensor::new([m, k], values(m * k, 1)).unwrap()
        };
        let b = HostTensor::new([k, n], values(k * n, 2)).unwrap();
        let residual = HostTensor::new([m, n], values(m * n, 3)).unwrap();
        let expected = magnetar_provider_cpu::residual_add(
            &residual,
            &magnetar_provider_cpu::matmul(&a, &b, transpose_a, false).unwrap(),
        )
        .unwrap();
        let (a_dev, b_dev, residual_dev) = (
            upload(&kernels, &a),
            upload(&kernels, &b),
            upload(&kernels, &residual),
        );
        let fused = download(
            &kernels,
            &kernels
                .matmul_residual(&a_dev, &b_dev, &residual_dev, transpose_a, false)
                .unwrap(),
        );
        assert_close(&fused, &expected);
        let unfused = download(
            &kernels,
            &kernels
                .residual_add(
                    &residual_dev,
                    &kernels.matmul(&a_dev, &b_dev, transpose_a, false).unwrap(),
                )
                .unwrap(),
        );
        assert_eq!(bits(&fused), bits(&unfused));
    }

    let gate = HostTensor::new([3, 70], values(210, 4)).unwrap();
    let up = HostTensor::new([3, 70], values(210, 5)).unwrap();
    let expected = magnetar_provider_cpu::mul(&magnetar_provider_cpu::silu(&gate), &up).unwrap();
    let (gate_dev, up_dev) = (upload(&kernels, &gate), upload(&kernels, &up));
    let fused = download(&kernels, &kernels.silu_mul(&gate_dev, &up_dev).unwrap());
    assert_close(&fused, &expected);
    let unfused = download(
        &kernels,
        &kernels
            .mul(&kernels.silu(&gate_dev).unwrap(), &up_dev)
            .unwrap(),
    );
    assert_eq!(bits(&fused), bits(&unfused));
}

/// The block-per-row `rmsnorm_kernel` on a width that is not a multiple of
/// its thread count, with a broadcast and a per-row weight.
#[test]
fn block_rmsnorm_matches_reference_cpu_on_wide_rows() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (rows, cols) = (3u64, 1000u64);
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01)
        .collect();
    let input = HostTensor::new([rows, cols], data).unwrap();
    for weight in [
        HostTensor::new(
            [cols],
            (0..cols).map(|i| 0.5 + i as f32 * 1e-3).collect::<Vec<_>>(),
        )
        .unwrap(),
        HostTensor::new(
            [rows, cols],
            (0..rows * cols)
                .map(|i| 1.0 - (i % 17) as f32 * 0.01)
                .collect::<Vec<_>>(),
        )
        .unwrap(),
    ] {
        let expected = magnetar_provider_cpu::rmsnorm(&input, &weight, 1e-6).unwrap();
        let actual_dev = kernels
            .rmsnorm(&upload(&kernels, &input), &upload(&kernels, &weight), 1e-6)
            .unwrap();
        assert_close(&download(&kernels, &actual_dev), &expected);
    }
}

/// `matmul_split_k_kernel`: column counts that are not a multiple of 32,
/// an inner dimension that is not a multiple of the split, one-row (decode)
/// and multi-row (prefill) shapes, and a transposed `a`.
#[test]
fn split_k_matmul_matches_reference_cpu_on_uneven_shapes() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let values = |count: u64, seed: u64| -> Vec<f32> {
        (0..count)
            .map(|i| (((i * 7919 + seed * 104_729) % 2001) as f32 - 1000.0) * 1e-4)
            .collect()
    };
    for (m, k, n, transpose_a) in [
        (1, 1000, 70, false),
        (1, 37, 33, false),
        (3, 129, 65, false),
        (5, 64, 1, false),
        (4, 50, 40, true),
    ] {
        let a = if transpose_a {
            HostTensor::new([k, m], values(m * k, 1)).unwrap()
        } else {
            HostTensor::new([m, k], values(m * k, 1)).unwrap()
        };
        let b = HostTensor::new([k, n], values(k * n, 2)).unwrap();
        let expected = magnetar_provider_cpu::matmul(&a, &b, transpose_a, false).unwrap();
        let actual_dev = kernels
            .matmul(
                &upload(&kernels, &a),
                &upload(&kernels, &b),
                transpose_a,
                false,
            )
            .unwrap();
        assert_close(&download(&kernels, &actual_dev), &expected);
    }
}

/// `gated-matmul` (the Runtime's fused SwiGLU projection) is bit-identical
/// to its two matmuls followed by `silu-mul`, with full and compact
/// weights, one- and multi-row inputs, and on its transposed fallback.
#[test]
fn gated_matmul_is_bit_identical_to_two_matmuls_and_silu_mul() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let bits = |tensor: &HostTensor| tensor.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for (m, k, n) in [(1u64, 896u64, 4864u64), (3, 96, 1000), (2, 37, 33)] {
        let a = upload(
            &kernels,
            &HostTensor::new([m, k], bf16_exact_values((m * k) as usize, 1)).unwrap(),
        );
        let gate = HostTensor::new([k, n], bf16_exact_values((k * n) as usize, 2)).unwrap();
        let up = HostTensor::new([k, n], bf16_exact_values((k * n) as usize, 3)).unwrap();
        let unfused = |gate: &CudaDeviceBuffer, up: &CudaDeviceBuffer| {
            let gate = kernels.matmul(&a, gate, false, false).unwrap();
            let up = kernels.matmul(&a, up, false, false).unwrap();
            download(&kernels, &kernels.silu_mul(&gate, &up).unwrap())
        };
        let (full_gate, full_up) = (upload(&kernels, &gate), upload(&kernels, &up));
        let fused = kernels
            .gated_matmul(&a, &full_gate, &full_up, false, false)
            .unwrap();
        assert_eq!(fused.shape, vec![m, n]);
        assert_eq!(
            bits(&download(&kernels, &fused)),
            bits(&unfused(&full_gate, &full_up)),
            "[{m}, {k}] x [{k}, {n}], f32 weights"
        );
        let (compact_gate, compact_up) = (
            kernels.upload_weight(&gate).unwrap(),
            kernels.upload_weight(&up).unwrap(),
        );
        let fused = kernels
            .gated_matmul(&a, &compact_gate, &compact_up, false, false)
            .unwrap();
        assert_eq!(
            bits(&download(&kernels, &fused)),
            bits(&unfused(&compact_gate, &compact_up)),
            "[{m}, {k}] x [{k}, {n}], compact weights"
        );
    }
    // Transposed weights take the three-launch fallback.
    let a = upload(
        &kernels,
        &HostTensor::new([2, 8], bf16_exact_values(16, 4)).unwrap(),
    );
    let gate = upload(
        &kernels,
        &HostTensor::new([5, 8], bf16_exact_values(40, 5)).unwrap(),
    );
    let up = upload(
        &kernels,
        &HostTensor::new([5, 8], bf16_exact_values(40, 6)).unwrap(),
    );
    let fused = kernels.gated_matmul(&a, &gate, &up, false, true).unwrap();
    let expected = kernels
        .silu_mul(
            &kernels.matmul(&a, &gate, false, true).unwrap(),
            &kernels.matmul(&a, &up, false, true).unwrap(),
        )
        .unwrap();
    assert_eq!(
        bits(&download(&kernels, &fused)),
        bits(&download(&kernels, &expected))
    );
    // Weights of different shapes are refused.
    let other = upload(
        &kernels,
        &HostTensor::new([8, 4], bf16_exact_values(32, 7)).unwrap(),
    );
    assert!(
        kernels
            .gated_matmul(&a, &gate, &other, false, true)
            .is_err()
    );
}

/// Greedy selection on the Device picks the column host greedy selection
/// picks: the highest finite, non-excluded score after the temperature
/// division, ties (including -0 against +0) going to the lowest column.
#[test]
fn greedy_select_matches_host_greedy_selection() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let host = |scores: &[f32], excluded: &[std::ops::Range<u64>], temperature: f32| {
        let mut best: Option<(f32, u64)> = None;
        for (column, &score) in scores.iter().enumerate() {
            let column = column as u64;
            if !score.is_finite() || excluded.iter().any(|range| range.contains(&column)) {
                continue;
            }
            let score = if temperature == 0.0 {
                score
            } else {
                score / temperature
            };
            if best.is_none_or(|(value, _)| score > value) {
                best = Some((score, column));
            }
        }
        best.map(|(_, column)| column)
    };
    let mut state = 0x9e37_79b9_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for round in 0..40 {
        let cols = [151_936u64, 1000, 33, 7][round % 4];
        let rows = 1 + (round as u64 % 3);
        // Coarse values so ties are common.
        let mut data: Vec<f32> = (0..rows * cols)
            .map(|_| ((next() % 64) as f32 - 32.0) * 0.25)
            .collect();
        for _ in 0..4 {
            let index = (next() % (rows * cols)) as usize;
            data[index] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0][(next() % 4) as usize];
        }
        let excluded: Vec<std::ops::Range<u64>> = (0..(next() % 4))
            .map(|_| {
                let start = next() % cols;
                start..(start + 1 + next() % 5).min(cols)
            })
            .collect();
        let temperature = [0.0f32, 1.0, 0.7, 1.3][round % 4];
        let row = next() % rows;
        let logits = upload(
            &kernels,
            &HostTensor::new([rows, cols], data.clone()).unwrap(),
        );
        let row_scores = &data[(row * cols) as usize..((row + 1) * cols) as usize];
        assert_eq!(
            kernels
                .greedy_select(&logits, row, &excluded, temperature)
                .unwrap(),
            host(row_scores, &excluded, temperature),
            "round {round}: [{rows}, {cols}] row {row}, temperature {temperature}"
        );
    }
    // -0 and +0 compare equal: the lower column wins.
    let logits = upload(
        &kernels,
        &HostTensor::new([1, 3], vec![-1.0, 0.0, -0.0]).unwrap(),
    );
    assert_eq!(
        kernels.greedy_select(&logits, 0, &[], 0.0).unwrap(),
        Some(1)
    );
    let logits = upload(
        &kernels,
        &HostTensor::new([1, 3], vec![-1.0, -0.0, 0.0]).unwrap(),
    );
    assert_eq!(
        kernels.greedy_select(&logits, 0, &[], 0.0).unwrap(),
        Some(1)
    );
    // Nothing eligible.
    let logits = upload(
        &kernels,
        &HostTensor::new([1, 3], vec![1.0, f32::NAN, 2.0]).unwrap(),
    );
    assert_eq!(
        kernels
            .greedy_select(&logits, 0, &[0..1, 2..3], 0.0)
            .unwrap(),
        None
    );
    // A row past the logits is refused.
    assert!(kernels.greedy_select(&logits, 1, &[], 0.0).is_err());
}

/// The multi-row split-K Kernels (prefill: several rows of `a`) give each
/// row exactly the bits the one-row Kernel gives it alone, for matmul,
/// matmul-bias, matmul-residual and gated-matmul, with f32 and compact
/// weights, row counts that are and are not multiples of the row group,
/// and inner lengths that leave a tail.
#[test]
fn multi_row_split_k_rows_are_bit_identical_to_single_row_matmuls() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let bits = |tensor: &HostTensor| tensor.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for (m, k, n) in [
        (16u64, 896u64, 128u64),
        (3, 37, 33),
        (9, 130, 70),
        (8, 64, 32),
        // The tile Kernels: several 256-inner stages, a partial last stage,
        // a partial last column tile and a partial last row group.
        (64, 896, 256),
        (130, 600, 200),
    ] {
        let a = HostTensor::new([m, k], bf16_exact_values((m * k) as usize, 1)).unwrap();
        let weight = HostTensor::new([k, n], bf16_exact_values((k * n) as usize, 2)).unwrap();
        let up = HostTensor::new([k, n], bf16_exact_values((k * n) as usize, 3)).unwrap();
        let bias = upload(
            &kernels,
            &HostTensor::new([n], bf16_exact_values(n as usize, 4)).unwrap(),
        );
        let residual_host =
            HostTensor::new([m, n], bf16_exact_values((m * n) as usize, 5)).unwrap();
        let residual = upload(&kernels, &residual_host);
        let a_dev = upload(&kernels, &a);
        for compact in [false, true] {
            let place = |tensor: &HostTensor| {
                if compact {
                    kernels.upload_weight(tensor).unwrap()
                } else {
                    upload(&kernels, tensor)
                }
            };
            let (b, up_dev) = (place(&weight), place(&up));
            let outputs = [
                download(&kernels, &kernels.matmul(&a_dev, &b, false, false).unwrap()),
                download(
                    &kernels,
                    &kernels
                        .matmul_bias(&a_dev, &b, &bias, false, false)
                        .unwrap(),
                ),
                download(
                    &kernels,
                    &kernels
                        .matmul_residual(&a_dev, &b, &residual, false, false)
                        .unwrap(),
                ),
                download(
                    &kernels,
                    &kernels
                        .gated_matmul(&a_dev, &b, &up_dev, false, false)
                        .unwrap(),
                ),
            ];
            for row in 0..m as usize {
                let a_row = upload(
                    &kernels,
                    &HostTensor::new(
                        [1, k],
                        a.data[row * k as usize..(row + 1) * k as usize].to_vec(),
                    )
                    .unwrap(),
                );
                let residual_row = upload(
                    &kernels,
                    &HostTensor::new(
                        [1, n],
                        residual_host.data[row * n as usize..(row + 1) * n as usize].to_vec(),
                    )
                    .unwrap(),
                );
                let singles = [
                    download(&kernels, &kernels.matmul(&a_row, &b, false, false).unwrap()),
                    download(
                        &kernels,
                        &kernels
                            .matmul_bias(&a_row, &b, &bias, false, false)
                            .unwrap(),
                    ),
                    download(
                        &kernels,
                        &kernels
                            .matmul_residual(&a_row, &b, &residual_row, false, false)
                            .unwrap(),
                    ),
                    download(
                        &kernels,
                        &kernels
                            .gated_matmul(&a_row, &b, &up_dev, false, false)
                            .unwrap(),
                    ),
                ];
                // A long prefill against compact weights runs on the
                // Tensor Cores where the Device has them: same values to
                // within the f32 accumulation order, not the same bits.
                let tensor_cores = compact && m >= 64 && kernels.has_tensor_cores();
                for (operator, (multi, single)) in outputs.iter().zip(&singles).enumerate() {
                    let range = row * n as usize..(row + 1) * n as usize;
                    if tensor_cores {
                        for (actual, expected) in multi.data[range].iter().zip(&single.data) {
                            assert!(
                                (actual - expected).abs()
                                    <= 1e-5 * k as f32 * expected.abs().max(1.0),
                                "operator {operator}, [{m}, {k}] x [{k}, {n}] row {row}: \
                                 tensor cores {actual} vs one row {expected}"
                            );
                        }
                        continue;
                    }
                    let row_bits = &bits(multi)[range];
                    assert_eq!(
                        row_bits,
                        bits(single).as_slice(),
                        "operator {operator}, [{m}, {k}] x [{k}, {n}] row {row}, compact {compact}"
                    );
                }
            }
        }
    }
}

/// A Device of compute capability 8.0 or later loads the Tensor Core
/// Kernels (unless they are turned off): a compilation failure must not
/// silently leave long prefills on the CUDA cores.
#[test]
fn tensor_cores_load_on_ampere_and_later() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (major, _) = kernels.stream().context().compute_capability().unwrap();
    let allowed = std::env::var(crate::kernels::TENSOR_CORES_ENV).as_deref() != Ok("0");
    assert_eq!(kernels.has_tensor_cores(), major >= 8 && allowed);
}

/// The Tensor Core matmul keeps f32 activations: each `a` value enters as
/// two bfloat16 terms. With every `a` equal to 1 + 2^-12 (bfloat16 rounds
/// it to 1), dropping the low term would be off by 2^-12 of every sum.
/// [70, 900] x [900, 130] takes the plain Tensor Core Kernel (900 is not a
/// multiple of 8); [200, 904] x [904, 136] the pipelined one.
#[test]
fn tensor_core_matmul_keeps_f32_activation_precision() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    if !kernels.has_tensor_cores() {
        return;
    }
    for (m, k, n) in [(70u64, 900u64, 130u64), (200, 904, 136)] {
        check_activation_precision(&kernels, m, k, n);
    }
}

fn check_activation_precision(kernels: &CudaKernels, m: u64, k: u64, n: u64) {
    let a_value = 1.0f32 + 2f32.powi(-12);
    let a = upload(
        kernels,
        &HostTensor::new([m, k], vec![a_value; (m * k) as usize]).unwrap(),
    );
    let weight: Vec<f32> = bf16_exact_values((k * n) as usize, 21)
        .into_iter()
        .map(f32::abs)
        .collect();
    let b = kernels
        .upload_weight(&HostTensor::new([k, n], weight.clone()).unwrap())
        .unwrap();
    assert!(b.is_compact());
    let actual = download(kernels, &kernels.matmul(&a, &b, false, false).unwrap());
    for row in 0..m as usize {
        for col in 0..n as usize {
            let column_sum: f64 = (0..k as usize)
                .map(|inner| f64::from(weight[inner * n as usize + col]))
                .sum();
            let expected = f64::from(a_value) * column_sum;
            let value = f64::from(actual.data[row * n as usize + col]);
            assert!(
                (value - expected).abs() <= 5e-5 * expected,
                "row {row} col {col}: {value} vs {expected}"
            );
        }
    }
}

/// A finite activation that bfloat16 rounding would turn into infinity
/// stays finite on the Tensor Core path: `f32::MAX` times 2^-120, summed
/// over 256 inners, is about 65536 (Codex review on
/// astorise/Magnetar-provider-CUDA#25).
#[test]
fn tensor_core_matmul_keeps_activations_near_f32_max_finite() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    if !kernels.has_tensor_cores() {
        return;
    }
    let (m, k, n) = (64u64, 256u64, 256u64);
    let a = upload(
        &kernels,
        &HostTensor::new([m, k], vec![f32::MAX; (m * k) as usize]).unwrap(),
    );
    let small = 2f32.powi(-120);
    let b = kernels
        .upload_weight(&HostTensor::new([k, n], vec![small; (k * n) as usize]).unwrap())
        .unwrap();
    assert!(b.is_compact());
    let expected = f64::from(f32::MAX) * f64::from(small) * k as f64;
    let actual = download(&kernels, &kernels.matmul(&a, &b, false, false).unwrap());
    for value in &actual.data {
        assert!(
            value.is_finite() && ((f64::from(*value) - expected) / expected).abs() < 1e-4,
            "{value} vs {expected}"
        );
    }
}

/// The Tensor Core prefill attention (64-dimension heads) keeps large
/// finite V values finite: its output is a weighted mean of V rows.
#[test]
fn tensor_core_attention_keeps_large_finite_values_finite() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let rows = 70u64;
    let q = HostTensor::new([rows, 64], vec![0.0; 64 * rows as usize]).unwrap();
    let k = HostTensor::new([rows, 64], vec![0.0; 64 * rows as usize]).unwrap();
    let v = HostTensor::new([rows, 64], vec![1e38; 64 * rows as usize]).unwrap();
    let actual = download(
        &kernels,
        &kernels
            .attention(
                &upload(&kernels, &q),
                &upload(&kernels, &k),
                &upload(&kernels, &v),
                1,
                64,
                None,
                None,
                true,
            )
            .unwrap(),
    );
    for value in &actual.data {
        assert!(
            value.is_finite() && (value / 1e38 - 1.0).abs() < 1e-4,
            "{value}"
        );
    }
}

/// `download_rows` returns exactly the requested rows of a rank-2 buffer.
#[test]
fn download_rows_returns_only_the_requested_rows() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let tensor = HostTensor::new([4, 3], (0..12).map(|v| v as f32).collect::<Vec<_>>()).unwrap();
    let buffer = upload(&kernels, &tensor);
    let last = kernels.download_rows(&buffer, 3, 1).unwrap().unwrap();
    assert_eq!(last.shape, vec![1, 3]);
    assert_eq!(last.data, vec![9.0, 10.0, 11.0]);
    let middle = kernels.download_rows(&buffer, 1, 2).unwrap().unwrap();
    assert_eq!(middle.data, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert!(kernels.download_rows(&buffer, 3, 2).unwrap().is_none());
}

#[test]
fn zeros_are_set_on_the_device_without_an_upload() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let uploads = kernels.upload_count();
    let buffer = kernels.zeros(&[5, 3]).unwrap();
    assert_eq!(kernels.upload_count(), uploads);
    let tensor = kernels.download(&buffer).unwrap();
    assert_eq!(tensor.shape, vec![5, 3]);
    assert_eq!(tensor.data, vec![0.0; 15]);
    assert!(kernels.zeros(&[u64::MAX, 2]).is_err());
}

#[test]
fn causal_attention_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let q = HostTensor::new([3, 2], [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let k = HostTensor::new([3, 2], [1.0, 0.0, 0.0, 1.0, 0.5, 0.5]).unwrap();
    let v = HostTensor::new([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let expected = magnetar_provider_cpu::attention(&q, &k, &v, 1, 2, None, None, true).unwrap();
    let actual_dev = kernels
        .attention(
            &upload(&kernels, &q),
            &upload(&kernels, &k),
            &upload(&kernels, &v),
            1,
            2,
            None,
            None,
            true,
        )
        .unwrap();
    assert_close(&download(&kernels, &actual_dev), &expected);
}

#[test]
fn back_to_back_kernels_do_not_round_trip_through_host() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a = HostTensor::new([2, 2], [1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = HostTensor::new([2, 2], [5.0, 6.0, 7.0, 8.0]).unwrap();
    let a_dev = upload(&kernels, &a);
    let b_dev = upload(&kernels, &b);
    let uploads_before = kernels.upload_count();
    let downloads_before = kernels.download_count();

    // `sum`'s output feeds directly into `mul` as an input, exactly like
    // two consecutive Kernels sharing a Provider/Device in the first-native
    // dispatch loop -- neither call should touch the host.
    let sum_dev = kernels.add(&a_dev, &b_dev).unwrap();
    let product_dev = kernels.mul(&sum_dev, &b_dev).unwrap();
    assert_eq!(
        kernels.upload_count(),
        uploads_before,
        "chaining two device-resident kernels must not trigger an upload"
    );
    assert_eq!(
        kernels.download_count(),
        downloads_before,
        "chaining two device-resident kernels must not trigger a download"
    );

    // Only the final, genuine host read crosses back.
    let product = download(&kernels, &product_dev);
    assert_eq!(kernels.download_count(), downloads_before + 1);
    let expected_sum = magnetar_provider_cpu::add(&a, &b).unwrap();
    let expected_product = magnetar_provider_cpu::mul(&expected_sum, &b).unwrap();
    assert_close(&product, &expected_product);
}

#[test]
fn grouped_query_sliding_window_attention_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    // head_count=2, kv_head_count=1 (both query heads share one kv head),
    // head_dimension=2 -> q model dim 4, kv model dim 2.
    let q = HostTensor::new(
        [4, 4],
        [
            1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.5, -0.5, 1.0, 1.0, -0.5, 0.5, -1.0, 0.0, 0.5, 0.5,
        ],
    )
    .unwrap();
    let k = HostTensor::new([4, 2], [1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, 0.5]).unwrap();
    let v = HostTensor::new([4, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let expected =
        magnetar_provider_cpu::attention(&q, &k, &v, 2, 2, Some(1), Some(2), true).unwrap();
    let actual_dev = kernels
        .attention(
            &upload(&kernels, &q),
            &upload(&kernels, &k),
            &upload(&kernels, &v),
            2,
            2,
            Some(1),
            Some(2),
            true,
        )
        .unwrap();
    assert_close(&download(&kernels, &actual_dev), &expected);
}

/// An offered output buffer (the previous value of the resource being
/// overwritten) is reused by a Kernel of the same output length and fully
/// rewritten; one of another length is left unused.
#[test]
fn offered_output_buffer_is_reused_and_fully_rewritten() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a = HostTensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = HostTensor::new([2, 3], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
    let expected = magnetar_provider_cpu::add(&a, &b).unwrap();
    let garbage = HostTensor::new([2, 3], [f32::NAN; 6]).unwrap();
    for (offered_shape, offered_len) in [([2u64, 3u64], 6usize), ([1, 4], 4)] {
        let offered = upload(
            &kernels,
            &HostTensor::new(offered_shape, garbage.data[..offered_len].to_vec()).unwrap(),
        );
        let _offer = kernels.offer_output(Some(offered.slice));
        let actual = kernels
            .add(&upload(&kernels, &a), &upload(&kernels, &b))
            .unwrap();
        assert_close(&download(&kernels, &actual), &expected);
    }
}

/// `append-kv-history-in-place`: `kv-write` writes rows into the store in
/// place and rejects overruns; attention bounded by `key_count` over a store
/// whose tail is NaN matches the CPU reference over exactly the history, on
/// both the block kernel and its long-history fallback.
#[test]
fn kv_write_and_key_count_attention_ignore_the_store_tail() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let mut store = upload(&kernels, &HostTensor::new([4, 2], vec![0.0; 8]).unwrap());
    let rows = upload(
        &kernels,
        &HostTensor::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    );
    kernels.kv_write(&mut store, &rows, 1).unwrap();
    assert_eq!(
        download(&kernels, &store).data,
        vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0]
    );
    assert!(kernels.kv_write(&mut store, &rows, 3).is_err());

    // (history rows, heads, kv heads, head dimension): the block kernel, and
    // a history too long for its shared memory.
    for (case, (history, heads, kv_heads, dim)) in [(300u64, 14u64, 2u64, 64u64), (12_500, 2, 1, 4)]
        .into_iter()
        .enumerate()
    {
        let capacity = history + 7;
        let q = HostTensor::new(
            [1, heads * dim],
            attention_values((heads * dim) as usize, 10 * case + 1),
        )
        .unwrap();
        let k = HostTensor::new(
            [history, kv_heads * dim],
            attention_values((history * kv_heads * dim) as usize, 10 * case + 2),
        )
        .unwrap();
        let v = HostTensor::new(
            [history, kv_heads * dim],
            attention_values((history * kv_heads * dim) as usize, 10 * case + 3),
        )
        .unwrap();
        let expected =
            magnetar_provider_cpu::attention(&q, &k, &v, heads, dim, Some(kv_heads), None, true)
                .unwrap();
        let with_tail = |tensor: &HostTensor| {
            let mut data = vec![f32::NAN; (capacity * kv_heads * dim) as usize];
            data[..tensor.data.len()].copy_from_slice(&tensor.data);
            HostTensor::new([capacity, kv_heads * dim], data).unwrap()
        };
        let actual = kernels
            .attention_bounded(
                &upload(&kernels, &q),
                &upload(&kernels, &with_tail(&k)),
                &upload(&kernels, &with_tail(&v)),
                heads,
                dim,
                Some(kv_heads),
                None,
                true,
                Some(history),
            )
            .unwrap();
        assert_close(&download(&kernels, &actual), &expected);
    }
}

/// Deterministic pseudo-random values in [-0.5, 0.5).
fn attention_values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 7919 + seed * 104_729) % 2003) as f32) / 2003.0 - 0.5)
        .collect()
}

/// `attention_block_kernel` (one block per head and query) on Qwen2.5-like
/// shapes: a decode step over a long history, a causal prefill, a sliding
/// window over a GQA prefill, and a history too long for its shared memory,
/// which uses `attention_kernel` instead.
#[test]
fn block_attention_matches_reference_cpu_on_decode_prefill_and_long_histories() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    // (seq_len, kv_seq_len, head_count, kv_head_count, head_dimension, window)
    let cases: [(u64, u64, u64, u64, u64, Option<u64>); 20] = [
        // 128-dimension heads (Qwen2.5-1.5B's 12 query / 2 KV heads): the
        // grouped decode Kernel's 32-key chunks, with and without a
        // window, and the Tensor Core prefill on its larger stages.
        (1, 300, 12, 2, 128, None),
        (1, 1000, 12, 2, 128, Some(300)),
        (130, 130, 12, 2, 128, None),
        (200, 1000, 12, 2, 128, Some(150)),
        (37, 37, 12, 2, 128, None),
        (1, 300, 14, 2, 64, None),
        // The grouped decode Kernel: one exact 64-key chunk, a long history
        // ending in a partial chunk, and one query head per KV head.
        (1, 64, 14, 2, 64, None),
        (1, 4100, 14, 2, 64, None),
        (1, 129, 2, 2, 64, Some(64)),
        (1, 1000, 14, 2, 64, Some(300)),
        (130, 130, 14, 2, 64, None),
        (200, 1000, 14, 2, 64, Some(150)),
        (16, 16, 2, 2, 64, None),
        (300, 300, 14, 2, 64, Some(100)),
        (40, 200, 4, 2, 16, None),
        (70, 70, 2, 1, 128, Some(9)),
        (37, 37, 14, 2, 64, None),
        (23, 23, 4, 2, 8, Some(5)),
        (1, 12_500, 2, 1, 4, None),
        // More chunks than the combine's shared memory holds weights for
        // (11,329 x 4 bytes > 44 KB): it recomputes each weight instead
        // (Codex review on astorise/Magnetar-provider-CUDA#29).
        (1, 1_450_000, 2, 1, 4, None),
    ];
    for (case, (seq_len, kv_seq_len, heads, kv_heads, dim, window)) in cases.into_iter().enumerate()
    {
        let q = HostTensor::new(
            [seq_len, heads * dim],
            attention_values((seq_len * heads * dim) as usize, 3 * case + 1),
        )
        .unwrap();
        let k = HostTensor::new(
            [kv_seq_len, kv_heads * dim],
            attention_values((kv_seq_len * kv_heads * dim) as usize, 3 * case + 2),
        )
        .unwrap();
        let v = HostTensor::new(
            [kv_seq_len, kv_heads * dim],
            attention_values((kv_seq_len * kv_heads * dim) as usize, 3 * case + 3),
        )
        .unwrap();
        let expected =
            magnetar_provider_cpu::attention(&q, &k, &v, heads, dim, Some(kv_heads), window, true)
                .unwrap();
        let actual_dev = kernels
            .attention(
                &upload(&kernels, &q),
                &upload(&kernels, &k),
                &upload(&kernels, &v),
                heads,
                dim,
                Some(kv_heads),
                window,
                true,
            )
            .unwrap();
        assert_close(&download(&kernels, &actual_dev), &expected);
    }
}

/// Large finite V values stay finite: weights are normalized before they
/// weight V (Codex review on astorise/Magnetar-provider-CUDA#13).
#[test]
fn block_attention_keeps_large_finite_values_finite() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let q = HostTensor::new([1, 2], vec![0.0; 2]).unwrap();
    let k = HostTensor::new([4, 2], vec![0.0; 8]).unwrap();
    let v = HostTensor::new([4, 2], vec![1e38; 8]).unwrap();
    let actual = download(
        &kernels,
        &kernels
            .attention(
                &upload(&kernels, &q),
                &upload(&kernels, &k),
                &upload(&kernels, &v),
                1,
                2,
                None,
                None,
                false,
            )
            .unwrap(),
    );
    for value in &actual.data {
        assert!(
            value.is_finite() && (value / 1e38 - 1.0).abs() < 1e-5,
            "{value}"
        );
    }
}

/// The same for the grouped decode Kernel (64-dimension heads sharing a KV
/// head): each chunk's output is a weighted mean of V rows.
#[test]
fn grouped_decode_attention_keeps_large_finite_values_finite() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let keys = 130u64;
    let q = HostTensor::new([1, 4 * 64], vec![0.0; 4 * 64]).unwrap();
    let k = HostTensor::new([keys, 64], vec![0.0; 64 * keys as usize]).unwrap();
    let v = HostTensor::new([keys, 64], vec![1e38; 64 * keys as usize]).unwrap();
    let actual = download(
        &kernels,
        &kernels
            .attention(
                &upload(&kernels, &q),
                &upload(&kernels, &k),
                &upload(&kernels, &v),
                4,
                64,
                Some(1),
                None,
                true,
            )
            .unwrap(),
    );
    for value in &actual.data {
        assert!(
            value.is_finite() && (value / 1e38 - 1.0).abs() < 1e-5,
            "{value}"
        );
    }
}

/// The same for a prefill (the tiled Kernel): each query's running
/// output is a weighted mean of V rows, never their weighted sum.
#[test]
fn tiled_prefill_attention_keeps_large_finite_values_finite() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let rows = 70u64;
    let q = HostTensor::new([rows, 8], vec![0.0; 8 * rows as usize]).unwrap();
    let k = HostTensor::new([rows, 8], vec![0.0; 8 * rows as usize]).unwrap();
    let v = HostTensor::new([rows, 8], vec![1e38; 8 * rows as usize]).unwrap();
    let actual = download(
        &kernels,
        &kernels
            .attention(
                &upload(&kernels, &q),
                &upload(&kernels, &k),
                &upload(&kernels, &v),
                2,
                4,
                None,
                None,
                true,
            )
            .unwrap(),
    );
    for value in &actual.data {
        assert!(
            value.is_finite() && (value / 1e38 - 1.0).abs() < 1e-5,
            "{value}"
        );
    }
}

// ---------------------------------------------------------------------------
// Native half-precision compute (`add-native-cuda-half-precision-compute`
// Phase 2)
// ---------------------------------------------------------------------------
//
// Rather than a fudge-factor tolerance against the f32 reference, the
// expected value is computed by faithfully modeling exactly what the real
// on-device kernel does: decode each already-half-precision-rounded input,
// perform the arithmetic in `f32`, then re-round the result to the same
// half-precision format -- using this crate's own `half_precision` module
// (the same functions `CudaKernels::upload_half`/`download_half`/the
// dispatched kernel use). This proves the actual GPU arithmetic instruction
// matches bit-for-bit, not merely "close enough" -- a genuinely strong
// correctness proof precisely because half-precision rounding is real,
// lossy, and would show up as a mismatch if the device kernel's bit
// manipulation diverged from this crate's Rust implementation in any way.

type Encoder = fn(f32) -> u16;
type Decoder = fn(u16) -> f32;

fn half_expected(a: &[f32], b: &[f32], dtype: CudaHalfDType, op: fn(f32, f32) -> f32) -> Vec<f32> {
    let (encode, decode): (Encoder, Decoder) = match dtype {
        CudaHalfDType::Float16 => (half_precision::f32_to_f16, half_precision::f16_to_f32),
        CudaHalfDType::BrainFloat16 => (half_precision::f32_to_bf16, half_precision::bf16_to_f32),
    };
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let x = decode(encode(x));
            let y = decode(encode(y));
            decode(encode(op(x, y)))
        })
        .collect()
}

#[test]
fn add_half_f16_matches_the_real_device_bit_manipulation_exactly() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a_data = vec![1.0, 2.5, 3.0, -4.25, 0.1, 100.0];
    let b_data = vec![6.0, 0.2, 4.0, 3.0, 0.2, -99.5];
    let a = HostTensor::new([2, 3], a_data.clone()).unwrap();
    let b = HostTensor::new([2, 3], b_data.clone()).unwrap();
    let expected = half_expected(&a_data, &b_data, CudaHalfDType::Float16, |x, y| x + y);
    let a_dev = kernels.upload_half(&a, CudaHalfDType::Float16).unwrap();
    let b_dev = kernels.upload_half(&b, CudaHalfDType::Float16).unwrap();
    let actual_dev = kernels.add_half(&a_dev, &b_dev).unwrap();
    let actual = kernels.download_half(&actual_dev).unwrap();
    assert_eq!(actual.data, expected);
}

#[test]
fn mul_half_f16_matches_the_real_device_bit_manipulation_exactly() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a_data = vec![1.0, 2.5, 3.0, -4.25, 0.1, 100.0];
    let b_data = vec![6.0, 0.2, 4.0, 3.0, 0.2, -99.5];
    let a = HostTensor::new([2, 3], a_data.clone()).unwrap();
    let b = HostTensor::new([2, 3], b_data.clone()).unwrap();
    let expected = half_expected(&a_data, &b_data, CudaHalfDType::Float16, |x, y| x * y);
    let a_dev = kernels.upload_half(&a, CudaHalfDType::Float16).unwrap();
    let b_dev = kernels.upload_half(&b, CudaHalfDType::Float16).unwrap();
    let actual_dev = kernels.mul_half(&a_dev, &b_dev).unwrap();
    let actual = kernels.download_half(&actual_dev).unwrap();
    assert_eq!(actual.data, expected);
}

#[test]
fn add_half_bf16_matches_the_real_device_bit_manipulation_exactly() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a_data = vec![1.0, 2.5, 3.0, -4.25, 0.1, 100.0];
    let b_data = vec![6.0, 0.2, 4.0, 3.0, 0.2, -99.5];
    let a = HostTensor::new([2, 3], a_data.clone()).unwrap();
    let b = HostTensor::new([2, 3], b_data.clone()).unwrap();
    let expected = half_expected(&a_data, &b_data, CudaHalfDType::BrainFloat16, |x, y| x + y);
    let a_dev = kernels
        .upload_half(&a, CudaHalfDType::BrainFloat16)
        .unwrap();
    let b_dev = kernels
        .upload_half(&b, CudaHalfDType::BrainFloat16)
        .unwrap();
    let actual_dev = kernels.add_half(&a_dev, &b_dev).unwrap();
    let actual = kernels.download_half(&actual_dev).unwrap();
    assert_eq!(actual.data, expected);
}

#[test]
fn mul_half_bf16_matches_the_real_device_bit_manipulation_exactly() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a_data = vec![1.0, 2.5, 3.0, -4.25, 0.1, 100.0];
    let b_data = vec![6.0, 0.2, 4.0, 3.0, 0.2, -99.5];
    let a = HostTensor::new([2, 3], a_data.clone()).unwrap();
    let b = HostTensor::new([2, 3], b_data.clone()).unwrap();
    let expected = half_expected(&a_data, &b_data, CudaHalfDType::BrainFloat16, |x, y| x * y);
    let a_dev = kernels
        .upload_half(&a, CudaHalfDType::BrainFloat16)
        .unwrap();
    let b_dev = kernels
        .upload_half(&b, CudaHalfDType::BrainFloat16)
        .unwrap();
    let actual_dev = kernels.mul_half(&a_dev, &b_dev).unwrap();
    let actual = kernels.download_half(&actual_dev).unwrap();
    assert_eq!(actual.data, expected);
}

#[test]
fn add_half_rejects_shape_mismatch() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a = HostTensor::new([2], [1.0, 2.0]).unwrap();
    let b = HostTensor::new([3], [1.0, 2.0, 3.0]).unwrap();
    let a_dev = kernels.upload_half(&a, CudaHalfDType::Float16).unwrap();
    let b_dev = kernels.upload_half(&b, CudaHalfDType::Float16).unwrap();
    assert!(kernels.add_half(&a_dev, &b_dev).is_err());
}

#[test]
fn add_half_rejects_dtype_mismatch() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a = HostTensor::new([2], [1.0, 2.0]).unwrap();
    let b = HostTensor::new([2], [1.0, 2.0]).unwrap();
    let a_dev = kernels.upload_half(&a, CudaHalfDType::Float16).unwrap();
    let b_dev = kernels
        .upload_half(&b, CudaHalfDType::BrainFloat16)
        .unwrap();
    assert!(kernels.add_half(&a_dev, &b_dev).is_err());
}

/// `fix-rope-rotation-pairing`: the half-split pairing (Hugging Face
/// `rotate_half`) on real hardware matches `providers/cpu`, multi-head and
/// partial RoPE included.
#[test]
fn rope_half_split_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    // 3 rows, head_count = 2, head_width = 6, dimension = 4 (partial).
    let data: Vec<f32> = (0..36).map(|i| i as f32 * 0.07 - 1.2).collect();
    let input = HostTensor::new([3, 12], data).unwrap();
    for (dimension, offset) in [(4, 5), (6, 0)] {
        let expected = magnetar_provider_cpu::rope_with_pairing(
            &input,
            10000.0,
            1.0,
            dimension,
            offset,
            2,
            magnetar_provider_cpu::RopePairing::HalfSplit,
        )
        .unwrap();
        let actual_dev = kernels
            .rope_with_pairing(
                &upload(&kernels, &input),
                10000.0,
                1.0,
                dimension,
                offset,
                2,
                crate::kernels::CudaRopePairing::HalfSplit,
            )
            .unwrap();
        assert_close(&download(&kernels, &actual_dev), &expected);
    }
}

/// Llama 3 per-frequency scaling matches the CPU Provider, across the
/// kept, smoothed and divided frequency bands, interleaved and half-split.
#[test]
fn rope_llama3_frequency_scaling_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let scaling = magnetar_runtime::operator::RopeLlama3Scaling {
        factor: 8.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_context_length: 64,
    };
    // 2 rows, head_count = 2, head_width = 16: pairs span all three bands.
    let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.031 - 1.0).collect();
    let input = HostTensor::new([2, 32], data).unwrap();
    for (pairing, cuda_pairing) in [
        (
            magnetar_provider_cpu::RopePairing::HalfSplit,
            crate::kernels::CudaRopePairing::HalfSplit,
        ),
        (
            magnetar_provider_cpu::RopePairing::Interleaved,
            crate::kernels::CudaRopePairing::Interleaved,
        ),
    ] {
        let expected = magnetar_provider_cpu::rope_with_frequency_scaling(
            &input,
            500_000.0,
            1.0,
            16,
            37,
            2,
            pairing,
            Some(&scaling),
        )
        .unwrap();
        let unscaled =
            magnetar_provider_cpu::rope_with_pairing(&input, 500_000.0, 1.0, 16, 37, 2, pairing)
                .unwrap();
        assert_ne!(
            expected.data, unscaled.data,
            "the scaling changes the rotation"
        );
        let actual_dev = kernels
            .rope_with_frequency_scaling(
                &upload(&kernels, &input),
                500_000.0,
                1.0,
                16,
                37,
                2,
                cuda_pairing,
                Some(&scaling),
            )
            .unwrap();
        assert_close(&download(&kernels, &actual_dev), &expected);
    }
}

/// Exactly bfloat16 values (`execute-native-half-precision`): f32 values
/// whose low 16 bits are zero.
fn bf16_exact_values(len: usize, seed: u32) -> Vec<f32> {
    (0..len as u32)
        .map(|index| {
            let value = ((index.wrapping_mul(2_654_435_761).wrapping_add(seed)) % 2001) as f32
                / 997.0
                - 1.0;
            f32::from_bits(value.to_bits() & 0xFFFF_0000)
        })
        .collect()
}

fn bits(tensor: &HostTensor) -> Vec<u32> {
    tensor.data.iter().map(|value| value.to_bits()).collect()
}

/// A compact weight gives bit-identical matmul, fused matmul and embedding
/// results to the same weight stored as f32, on the split-K, transposed
/// and row-tail paths; non-exact or small weights stay f32.
#[test]
fn compact_bf16_weights_give_bit_identical_results() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    // [k, n] = [96, 1000]: n is not a multiple of the 32-column tiles.
    let (m, k, n) = (3u64, 96u64, 1000u64);
    let weight = HostTensor::new([k, n], bf16_exact_values((k * n) as usize, 7)).unwrap();
    let compact = kernels.upload_weight(&weight).unwrap();
    assert!(compact.is_compact());
    let full = upload(&kernels, &weight);
    assert_eq!(bits(&download(&kernels, &compact)), bits(&weight));

    let a_host = HostTensor::new(
        [m, k],
        (0..m * k)
            .map(|index| (index as f32 * 0.37).sin())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let a = upload(&kernels, &a_host);
    let bias = upload(
        &kernels,
        &HostTensor::new(
            [n],
            (0..n).map(|index| index as f32 * 1e-3).collect::<Vec<_>>(),
        )
        .unwrap(),
    );
    let residual = upload(
        &kernels,
        &HostTensor::new([m, n], vec![0.5; (m * n) as usize]).unwrap(),
    );
    let same = |left: &CudaDeviceBuffer, right: &CudaDeviceBuffer| {
        assert_eq!(
            bits(&download(&kernels, left)),
            bits(&download(&kernels, right))
        );
    };
    same(
        &kernels.matmul(&a, &compact, false, false).unwrap(),
        &kernels.matmul(&a, &full, false, false).unwrap(),
    );
    same(
        &kernels
            .matmul_bias(&a, &compact, &bias, false, false)
            .unwrap(),
        &kernels.matmul_bias(&a, &full, &bias, false, false).unwrap(),
    );
    same(
        &kernels
            .matmul_residual(&a, &compact, &residual, false, false)
            .unwrap(),
        &kernels
            .matmul_residual(&a, &full, &residual, false, false)
            .unwrap(),
    );
    // Transposed `b` takes the one-thread-per-output Kernel.
    let transposed = HostTensor::new([n, k], bf16_exact_values((k * n) as usize, 11)).unwrap();
    let compact_t = kernels.upload_weight(&transposed).unwrap();
    let full_t = upload(&kernels, &transposed);
    same(
        &kernels.matmul(&a, &compact_t, false, true).unwrap(),
        &kernels.matmul(&a, &full_t, false, true).unwrap(),
    );
    // An embedding table: rows of `n` values.
    let ids = upload(
        &kernels,
        &HostTensor::new([3], vec![0.0, 95.0, 41.0]).unwrap(),
    );
    same(
        &kernels.embedding_lookup(&compact, &ids).unwrap(),
        &kernels.embedding_lookup(&full, &ids).unwrap(),
    );
    // A compact buffer is never an f32 operand.
    assert!(kernels.matmul(&compact, &full_t, false, false).is_err());
    assert!(kernels.add(&full, &compact).is_err());
    assert!(kernels.silu(&compact).is_err());
    let expanded = kernels.expand(&compact).unwrap();
    assert!(!expanded.is_compact());
    assert_eq!(bits(&download(&kernels, &expanded)), bits(&weight));

    // Values with low bits set, or a small tensor, stay f32.
    let mut inexact = weight.clone();
    inexact.data[17] = 0.1;
    assert!(!kernels.upload_weight(&inexact).unwrap().is_compact());
    let small = HostTensor::new([4, 4], bf16_exact_values(16, 3)).unwrap();
    assert!(!kernels.upload_weight(&small).unwrap().is_compact());
}

/// Device time of a Qwen2.5-0.5B layer's long-context work: each prefill
/// matmul shape and the attention, at 512 and 4096 rows, plus a decode
/// step's attention over that many keys. Not a conformance check: run it
/// with `--ignored --nocapture` on a GPU.
#[test]
#[ignore = "GPU timing diagnostic"]
fn long_context_kernel_times() {
    // Minutes of Device work in a debug build, which runs ignored tests too.
    if cfg!(debug_assertions) {
        return;
    }
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let context = kernels.stream().context().clone();
    let timing = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let time = |repeats: usize, run: &dyn Fn()| -> f32 {
        run();
        let start = context.new_event(timing).unwrap();
        let end = context.new_event(timing).unwrap();
        start.record(kernels.stream()).unwrap();
        for _ in 0..repeats {
            run();
        }
        end.record(kernels.stream()).unwrap();
        start.elapsed_ms(&end).unwrap() / repeats as f32
    };
    let filled = |rows: u64, cols: u64, seed: u32| {
        upload(
            &kernels,
            &HostTensor::new(
                [rows, cols],
                bf16_exact_values((rows * cols) as usize, seed),
            )
            .unwrap(),
        )
    };
    for rows in [512u64, 4096] {
        let repeats = if rows > 1024 { 2 } else { 5 };
        for (k, n) in [
            (896u64, 896u64),
            (896, 128),
            (896, 4864),
            (4864, 896),
            (896, 151_936),
        ] {
            let a = filled(rows, k, 3);
            let weight = HostTensor::new([k, n], bf16_exact_values((k * n) as usize, 13)).unwrap();
            let b = kernels.upload_weight(&weight).unwrap();
            let ms = time(repeats, &|| {
                kernels.matmul(&a, &b, false, false).unwrap();
            });
            let tflops = 2.0 * (rows * k * n) as f32 / ms / 1e9;
            eprintln!(
                "prefill matmul [{rows}, {k}] x [{k}, {n}]: {ms:.2} ms ({tflops:.2} TFLOP/s)"
            );
        }
        let q = filled(rows, 896, 5);
        let kv_k = filled(rows, 128, 7);
        let kv_v = filled(rows, 128, 9);
        let ms = time(repeats, &|| {
            kernels
                .attention(&q, &kv_k, &kv_v, 14, 64, Some(2), None, true)
                .unwrap();
        });
        eprintln!("prefill attention, {rows} rows, 14 heads / 2 KV heads: {ms:.2} ms");
        let q1 = filled(1, 896, 11);
        let ms = time(20, &|| {
            kernels
                .attention(&q1, &kv_k, &kv_v, 14, 64, Some(2), None, false)
                .unwrap();
        });
        eprintln!("decode attention over {rows} keys: {:.1} us", ms * 1e3);
    }
}

/// Diagnostic, not a gate: the Device time of one decode-step matmul of
/// each Qwen2.5-0.5B projection shape, f32 and compact, and the bandwidth
/// it reaches. 200 back-to-back launches are timed, so a shape whose
/// Kernel is shorter than its host launch cost reads as that cost. Run
/// with `--ignored --nocapture` on a GPU.
#[test]
#[ignore = "GPU timing diagnostic"]
fn decode_matmul_bandwidth() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let context = kernels.stream().context().clone();
    let timing = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let repeats = 200;
    for (k, n) in [
        (896u64, 896u64),
        (896, 128),
        (896, 4864),
        (4864, 896),
        (896, 151_936),
    ] {
        let weight = HostTensor::new([k, n], bf16_exact_values((k * n) as usize, 13)).unwrap();
        let a = upload(
            &kernels,
            &HostTensor::new([1, k], vec![0.25; k as usize]).unwrap(),
        );
        let full = upload(&kernels, &weight);
        let compact = kernels.upload_weight(&weight).unwrap();
        let time = |b: &CudaDeviceBuffer| -> f32 {
            // Warm the function cache and the allocation.
            kernels.matmul(&a, b, false, false).unwrap();
            let start = context.new_event(timing).unwrap();
            let end = context.new_event(timing).unwrap();
            start.record(kernels.stream()).unwrap();
            for _ in 0..repeats {
                kernels.matmul(&a, b, false, false).unwrap();
            }
            end.record(kernels.stream()).unwrap();
            start.elapsed_ms(&end).unwrap() / repeats as f32
        };
        let f32_ms = time(&full);
        let bf16_ms = time(&compact);
        let gb = |bytes: u64, ms: f32| bytes as f32 / ms / 1e6;
        eprintln!(
            "matmul [1, {k}] x [{k}, {n}]: f32 {:.1} us ({:.0} GB/s), bf16 {:.1} us ({:.0} GB/s)",
            f32_ms * 1e3,
            gb(k * n * 4, f32_ms),
            bf16_ms * 1e3,
            gb(k * n * 2, bf16_ms),
        );
    }
}
