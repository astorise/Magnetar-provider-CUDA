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

use magnetar_provider_api::HostTensor;
use magnetar_provider_api::{packed_weight::PackedWeight, packed_weight::PackedWeightFormat};

use crate::half_precision;
use crate::kernels::{CudaDeviceBuffer, CudaHalfDType, CudaKernels, TqDecodeSchedule};
use crate::provider::CudaProvider;

const TOLERANCE: f32 = 1e-3;

fn kernels_or_skip() -> Option<CudaKernels> {
    let provider = CudaProvider::new();
    let context = provider.context()?;
    let kernels = CudaKernels::compile_and_load(&context).expect(
        "kernel compilation must succeed on a machine that already passed device discovery",
    );
    // The f32-activation Kernels, which these tests compare with the
    // Reference CPU tightly; the int8-activation test turns them on.
    kernels.set_int8_activations(false);
    Some(kernels)
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

/// `turboquant-kv-history`: `kv-write` into a TurboQuant store keeps each
/// row as the CPU's store does (`TurboQuantStoreState`: offsets set by the
/// write at row 0, each row's difference from them quantized), and
/// attention over the store -- a decode step on the codes, a prefill
/// through dequantized scratch -- matches the CPU reference over those
/// values, for 4- and 3-bit codes, grouped-query heads 64 to 512 wide (one
/// block per KV head for up to 8 query heads, one per query head beyond)
/// and a sliding window.
#[test]
fn turboquant_kv_stores_round_trip_and_attend_like_the_cpu() {
    use magnetar_provider_api::{KvStorePrecision, TurboQuantStoreState};
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    for precision in [KvStorePrecision::Tq4, KvStorePrecision::Tq3] {
        for (case, (history, heads, kv_heads, dim, window)) in [
            (37u64, 8u64, 2u64, 64u64, None),
            (300, 4, 1, 128, None),
            (129, 8, 1, 256, Some(64u64)),
            (5, 2, 2, 512, None),
            // More query heads per KV head than one grouped block serves:
            // a block per query head.
            (70, 16, 1, 64, None),
            // Many chunks: several per block of the split schedule, some
            // blocks past the history.
            (4100, 8, 2, 128, None),
        ]
        .into_iter()
        .enumerate()
        {
            let label = format!("{} case {case}", precision.as_str());
            let width = kv_heads * dim;
            let capacity = history + 3;
            // Rows sharing a large per-column component, as keys with a
            // bias do: the store's offsets hold it. Hashed values: the
            // arithmetic sequence of `attention_values` puts rotated
            // coordinates exactly on codebook boundaries once the offsets
            // are subtracted, where any rounding difference picks the other
            // centroid.
            let values = |seed: usize| {
                let common = hashed_values(width as usize, 10 * case + seed + 5);
                let data: Vec<f32> = hashed_values((history * width) as usize, 10 * case + seed)
                    .into_iter()
                    .enumerate()
                    .map(|(i, value)| value + 8.0 * common[i % width as usize])
                    .collect();
                HostTensor::new([history, width], data).unwrap()
            };
            let (k, v) = (values(2), values(3));
            // Written as a prefill and then one appended row.
            let store_of = |tensor: &HostTensor| {
                let mut store = kernels
                    .tq_zeros(&[capacity, width], precision, dim)
                    .unwrap();
                let split = ((history - 1) * width) as usize;
                let prefix =
                    HostTensor::new([history - 1, width], tensor.data[..split].to_vec()).unwrap();
                let last = HostTensor::new([1, width], tensor.data[split..].to_vec()).unwrap();
                kernels
                    .kv_write(&mut store, &upload(&kernels, &prefix), 0)
                    .unwrap();
                kernels
                    .kv_write(&mut store, &upload(&kernels, &last), history - 1)
                    .unwrap();
                store
            };
            let (k_store, v_store) = (store_of(&k), store_of(&v));
            // The same writes into the CPU's store: offsets from the
            // prefill's rows.
            let round_trip = |tensor: &HostTensor| {
                let mut store = TurboQuantStoreState::new(precision, dim, width);
                let split = ((history - 1) * width) as usize;
                let mut data = store.write(&tensor.data[..split], 0).unwrap();
                data.extend(store.write(&tensor.data[split..], history - 1).unwrap());
                HostTensor::new([history, width], data).unwrap()
            };
            let (k_values, v_values) = (round_trip(&k), round_trip(&v));

            // Each head row reads back as its round trip: the offsets are
            // summed in the same order, but the norms in another, so a
            // coordinate on a centroid boundary may rarely take the other
            // centroid.
            let stored = download(&kernels, &k_store);
            assert_eq!(stored.shape, vec![capacity, width], "{label}");
            let head_rows = (history * kv_heads) as usize;
            let mut exact = 0;
            for (read, expected) in stored.data[..(history * width) as usize]
                .chunks(dim as usize)
                .zip(k_values.data.chunks(dim as usize))
            {
                let difference = read
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                assert!(difference < 0.05, "{label}: {difference}");
                if difference < 1e-4 {
                    exact += 1;
                }
            }
            assert!(
                exact * 100 >= head_rows * 99,
                "{label}: {exact}/{head_rows}"
            );
            // Unwritten rows read back as the offsets.
            let offsets = magnetar_provider_api::turboquant_store_offsets(
                &k.data[..((history - 1) * width) as usize],
                width as usize,
            );
            assert!(
                stored.data[(history * width) as usize..]
                    .chunks(width as usize)
                    .all(|row| row == offsets.as_slice()),
                "{label}: unwritten rows read back as the offsets"
            );

            // A decode step on the codes, and a four-query prefill.
            for queries in [1u64, 4] {
                let q = HostTensor::new(
                    [queries, heads * dim],
                    attention_values((queries * heads * dim) as usize, 10 * case + 1),
                )
                .unwrap();
                let expected = magnetar_provider_cpu::attention(
                    &q,
                    &k_values,
                    &v_values,
                    heads,
                    dim,
                    Some(kv_heads),
                    window,
                    true,
                )
                .unwrap();
                // A decode step under every schedule of the code Kernels
                // (`speed-up-decode-attention-on-codes`).
                for schedule in [
                    TqDecodeSchedule::Split,
                    TqDecodeSchedule::ChunkPerBlock,
                    TqDecodeSchedule::Unstaged,
                ] {
                    kernels.set_tq_decode_schedule(schedule);
                    let actual = kernels
                        .attention_valued(
                            &upload(&kernels, &q),
                            &k_store,
                            &v_store,
                            heads,
                            dim,
                            None,
                            Some(kv_heads),
                            window,
                            true,
                            Some(history),
                            None,
                        )
                        .unwrap();
                    assert_close(&download(&kernels, &actual), &expected);
                }
                kernels.set_tq_decode_schedule(TqDecodeSchedule::Unstaged);
            }
        }
    }
}

/// `keep-turboquant-kv-head-exact`: a TurboQuant store whose first rows
/// are kept exact holds them as written, and attention over it -- a
/// decode step and a four-query prefill -- matches the CPU reference over
/// the CPU store's values (`TurboQuantStoreState::with_exact_rows`): for a
/// history the head holds (attention on the head), and for longer ones
/// whose chunks start inside the head (100 and 40 rows: a chunk of exact
/// keys and codes), at a chunk boundary (128 rows), under a sliding window
/// starting inside the head, with grouped blocks and a block per query
/// head.
#[test]
fn turboquant_exact_heads_attend_like_the_cpu() {
    use magnetar_provider_api::{KvStorePrecision, TurboQuantStoreState};
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    for precision in [KvStorePrecision::Tq4, KvStorePrecision::Tq3] {
        for (case, (history, heads, kv_heads, dim, window, exact)) in [
            (37u64, 8u64, 2u64, 64u64, None, 64u64),
            (300, 4, 1, 128, None, 100),
            (300, 8, 2, 128, None, 128),
            (129, 8, 1, 256, Some(64u64), 100),
            (70, 16, 1, 64, None, 40),
            // Many chunks: the head inside the first block of the split
            // schedule, several chunks per block.
            (4100, 8, 2, 128, None, 1024),
            (2000, 7, 1, 64, Some(1500u64), 1024),
        ]
        .into_iter()
        .enumerate()
        {
            let label = format!("{} case {case}", precision.as_str());
            let width = kv_heads * dim;
            let capacity = history + 3;
            // Rows sharing a large per-column component, as keys with a
            // bias do: the store's offsets hold it.
            let values = |seed: usize| {
                let common = hashed_values(width as usize, 10 * case + seed + 5);
                let data: Vec<f32> = hashed_values((history * width) as usize, 10 * case + seed)
                    .into_iter()
                    .enumerate()
                    .map(|(i, value)| value + 8.0 * common[i % width as usize])
                    .collect();
                HostTensor::new([history, width], data).unwrap()
            };
            let (k, v) = (values(2), values(3));
            let split = ((history - 1) * width) as usize;
            // Written as a prefill and then one appended row.
            let store_of = |tensor: &HostTensor| {
                let mut store = kernels
                    .tq_zeros_with_exact_head(&[capacity, width], precision, dim, exact)
                    .unwrap();
                let prefix =
                    HostTensor::new([history - 1, width], tensor.data[..split].to_vec()).unwrap();
                let last = HostTensor::new([1, width], tensor.data[split..].to_vec()).unwrap();
                kernels
                    .kv_write(&mut store, &upload(&kernels, &prefix), 0)
                    .unwrap();
                kernels
                    .kv_write(&mut store, &upload(&kernels, &last), history - 1)
                    .unwrap();
                store
            };
            let (k_store, v_store) = (store_of(&k), store_of(&v));
            let cpu_values = |tensor: &HostTensor| {
                let mut store =
                    TurboQuantStoreState::new(precision, dim, width).with_exact_rows(exact);
                let mut data = store.write(&tensor.data[..split], 0).unwrap();
                data.extend(store.write(&tensor.data[split..], history - 1).unwrap());
                HostTensor::new([history, width], data).unwrap()
            };
            let (k_values, v_values) = (cpu_values(&k), cpu_values(&v));

            // The head's rows read back as written.
            let held = (exact.min(history) * width) as usize;
            for (store, written) in [(&k_store, &k), (&v_store, &v)] {
                let stored = download(&kernels, store);
                assert_eq!(stored.shape, vec![capacity, width], "{label}");
                assert_eq!(stored.data[..held], written.data[..held], "{label}");
            }

            for queries in [1u64, 4] {
                let q = HostTensor::new(
                    [queries, heads * dim],
                    attention_values((queries * heads * dim) as usize, 10 * case + 1),
                )
                .unwrap();
                let expected = magnetar_provider_cpu::attention(
                    &q,
                    &k_values,
                    &v_values,
                    heads,
                    dim,
                    Some(kv_heads),
                    window,
                    true,
                )
                .unwrap();
                // A decode step under every schedule of the code Kernels
                // (`speed-up-decode-attention-on-codes`).
                for schedule in [
                    TqDecodeSchedule::Split,
                    TqDecodeSchedule::ChunkPerBlock,
                    TqDecodeSchedule::Unstaged,
                ] {
                    kernels.set_tq_decode_schedule(schedule);
                    let actual = kernels
                        .attention_valued(
                            &upload(&kernels, &q),
                            &k_store,
                            &v_store,
                            heads,
                            dim,
                            None,
                            Some(kv_heads),
                            window,
                            true,
                            Some(history),
                            None,
                        )
                        .unwrap();
                    assert_close(&download(&kernels, &actual), &expected);
                }
                kernels.set_tq_decode_schedule(TqDecodeSchedule::Unstaged);
            }
        }
    }
}

/// `keep-turboquant-kv-head-exact`: a captured decode step (`kv-write` of
/// its row, then attention) over TurboQuant stores with exact heads
/// replays only within the regime it was captured in -- while the history
/// stays in the heads, its bound ends where they do; past them, it
/// replays on the codes up to the stores' capacity -- and every replay
/// matches the same step run eagerly on a copy of the history.
#[test]
fn captured_decode_steps_keep_their_exact_head_regime() {
    use magnetar_provider_api::KvStorePrecision;
    use std::collections::BTreeSet;
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (heads, kv_heads, dim, capacity, exact, prefix) = (4u64, 1u64, 64u64, 24u64, 8u64, 6u64);
    let width = kv_heads * dim;
    let rows = |seed: u64, count: u64| {
        HostTensor::new(
            [count, width],
            hashed_values((count * width) as usize, seed as usize),
        )
        .unwrap()
    };
    let query = |position: u64| {
        HostTensor::new(
            [1, heads * dim],
            hashed_values((heads * dim) as usize, 300 + position as usize),
        )
        .unwrap()
    };
    let store = || {
        kernels
            .tq_zeros_with_exact_head(&[capacity, width], KvStorePrecision::Tq4, dim, exact)
            .unwrap()
    };
    // The same history twice: stepped by captured steps, and eagerly.
    let [mut k_captured, mut v_captured, mut k_eager, mut v_eager] =
        [store(), store(), store(), store()];
    for (store, seed) in [
        (&mut k_captured, 1),
        (&mut v_captured, 2),
        (&mut k_eager, 1),
        (&mut v_eager, 2),
    ] {
        kernels
            .kv_write(store, &upload(&kernels, &rows(seed, prefix)), 0)
            .unwrap();
    }
    let decode = |q: &CudaDeviceBuffer,
                  k: &mut CudaDeviceBuffer,
                  v: &mut CudaDeviceBuffer,
                  k_row: &CudaDeviceBuffer,
                  v_row: &CudaDeviceBuffer,
                  position: u64| {
        kernels.kv_write(k, k_row, position)?;
        kernels.kv_write(v, v_row, position)?;
        kernels.attention_valued(
            q,
            k,
            v,
            heads,
            dim,
            None,
            Some(kv_heads),
            None,
            true,
            Some(position + 1),
            None,
        )
    };
    let mut eager = |position: u64| {
        let out = decode(
            &upload(&kernels, &query(position)),
            &mut k_eager,
            &mut v_eager,
            &upload(&kernels, &rows(100 + position, 1)),
            &upload(&kernels, &rows(200 + position, 1)),
            position,
        )
        .unwrap();
        download(&kernels, &out)
    };
    // Captured at a history the heads hold (7 rows), then past them (9).
    for (captured_at, bound, replays) in [
        (prefix, exact - (prefix + 1), 0..=1u64),
        (exact, capacity - (exact + 1), 0..=3),
    ] {
        // Run eagerly first: Kernels compiled, scratch buffers grown.
        let mut expected = vec![eager(captured_at)];
        let mut q = upload(&kernels, &query(captured_at));
        let mut k_row = upload(&kernels, &rows(100 + captured_at, 1));
        let mut v_row = upload(&kernels, &rows(200 + captured_at, 1));
        let out = kernels
            .stream()
            .alloc_zeros::<f32>((heads * dim) as usize)
            .unwrap();
        kernels
            .begin_capture(BTreeSet::from(["key_count".into(), "row_offset".into()]))
            .unwrap();
        let offered = kernels.offer_output(Some(out));
        let captured = decode(
            &q,
            &mut k_captured,
            &mut v_captured,
            &k_row,
            &v_row,
            captured_at,
        )
        .unwrap();
        drop(offered);
        let mut step = kernels.end_capture().unwrap();
        assert_eq!(step.max_offset, bound, "captured at {captured_at}");
        for offset in replays {
            let position = captured_at + offset;
            if offset > 0 {
                expected.push(eager(position));
                kernels.write_into(&mut q, &query(position).data).unwrap();
                kernels
                    .write_into(&mut k_row, &rows(100 + position, 1).data)
                    .unwrap();
                kernels
                    .write_into(&mut v_row, &rows(200 + position, 1).data)
                    .unwrap();
            }
            kernels.launch_captured(&mut step, offset).unwrap();
            assert_eq!(
                download(&kernels, &captured).data,
                expected[offset as usize].data,
                "replayed at {position}"
            );
        }
    }
}

/// `speed-up-decode-attention-on-codes`, decision 4: the split count
/// fills the target with blocks, keeps at least two chunks per block and
/// at most 32 blocks per KV head, and fills whole waves.
#[test]
fn turboquant_decode_splits_follow_the_heuristic() {
    use crate::kernels::tq_split_count;
    // Qwen2.5-0.5B's 2 KV heads on 28 multiprocessors holding 4 blocks.
    assert_eq!(tq_split_count(112, 2, 256), 32);
    assert_eq!(tq_split_count(112, 2, 32), 16);
    assert_eq!(tq_split_count(112, 2, 5), 2);
    assert_eq!(tq_split_count(112, 2, 1), 1);
    // Mistral 7B's 8 KV heads: 14 blocks each fill the 112 exactly.
    assert_eq!(tq_split_count(112, 8, 256), 14);
    // 84 resident blocks: 11 per KV head would leave a second wave of 4
    // blocks; 10 fill one wave.
    assert_eq!(tq_split_count(84, 8, 256), 10);
    // As many KV heads as resident blocks: one block each.
    assert_eq!(tq_split_count(28, 32, 256), 1);
}

/// `speed-up-decode-attention-on-codes`: a decode step over TurboQuant
/// stores, captured with its history split over two blocks per KV head,
/// replays over histories up to 1,001 keys with the same blocks (four
/// chunks each), every replay matching the CPU reference over the stores'
/// values.
#[test]
fn captured_split_decode_steps_cover_longer_histories() {
    use magnetar_provider_api::{KvStorePrecision, TurboQuantStoreState};
    use std::collections::BTreeSet;
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    kernels.set_tq_decode_schedule(TqDecodeSchedule::Split);
    let (heads, kv_heads, dim, capacity, prefix) = (8u64, 2u64, 64u64, 1200u64, 600u64);
    let width = kv_heads * dim;
    let all_k = hashed_values((capacity * width) as usize, 1);
    let all_v = hashed_values((capacity * width) as usize, 2);
    let rows = |all: &[f32], from: u64, count: u64| {
        HostTensor::new(
            [count, width],
            all[(from * width) as usize..((from + count) * width) as usize].to_vec(),
        )
        .unwrap()
    };
    let query = |position: u64| {
        HostTensor::new(
            [1, heads * dim],
            hashed_values((heads * dim) as usize, 300 + position as usize),
        )
        .unwrap()
    };
    let mut stores = [&all_k, &all_v].map(|all| {
        let mut store = kernels
            .tq_zeros(&[capacity, width], KvStorePrecision::Tq4, dim)
            .unwrap();
        kernels
            .kv_write(&mut store, &upload(&kernels, &rows(all, 0, prefix)), 0)
            .unwrap();
        store
    });
    let mut cpu = [&all_k, &all_v].map(|all| {
        let mut store = TurboQuantStoreState::new(KvStorePrecision::Tq4, dim, width);
        let values = store.write(&rows(all, 0, prefix).data, 0).unwrap();
        (store, values)
    });
    let mut q = upload(&kernels, &query(prefix));
    let mut k_row = upload(&kernels, &rows(&all_k, prefix, 1));
    let mut v_row = upload(&kernels, &rows(&all_v, prefix, 1));
    // An eager step first: Kernels compiled, scratch grown.
    {
        let [k_store, v_store] = &stores;
        kernels
            .attention_valued(
                &q,
                k_store,
                v_store,
                heads,
                dim,
                None,
                Some(kv_heads),
                None,
                true,
                Some(prefix),
                None,
            )
            .unwrap();
    }
    let out = kernels
        .stream()
        .alloc_zeros::<f32>((heads * dim) as usize)
        .unwrap();
    kernels
        .begin_capture(BTreeSet::from(["key_count".into(), "row_offset".into()]))
        .unwrap();
    let offered = kernels.offer_output(Some(out));
    let captured = {
        let [k_store, v_store] = &mut stores;
        kernels.kv_write(k_store, &k_row, prefix).unwrap();
        kernels.kv_write(v_store, &v_row, prefix).unwrap();
        kernels
            .attention_valued(
                &q,
                k_store,
                v_store,
                heads,
                dim,
                None,
                Some(kv_heads),
                None,
                true,
                Some(prefix + 1),
                None,
            )
            .unwrap()
    };
    drop(offered);
    let mut step = kernels.end_capture().unwrap();
    assert_eq!(step.max_offset, capacity - (prefix + 1));
    for offset in 0..=400u64 {
        let position = prefix + offset;
        kernels.write_into(&mut q, &query(position).data).unwrap();
        kernels
            .write_into(&mut k_row, &rows(&all_k, position, 1).data)
            .unwrap();
        kernels
            .write_into(&mut v_row, &rows(&all_v, position, 1).data)
            .unwrap();
        for ((store, values), all) in cpu.iter_mut().zip([&all_k, &all_v]) {
            values.extend(store.write(&rows(all, position, 1).data, position).unwrap());
        }
        kernels.launch_captured(&mut step, offset).unwrap();
        // Histories of 5 to 8 chunks: two blocks of three or four chunks.
        if offset % 100 == 0 || offset == 167 {
            let keys = position + 1;
            let history = |values: &Vec<f32>| {
                HostTensor::new([keys, width], values[..(keys * width) as usize].to_vec()).unwrap()
            };
            let expected = magnetar_provider_cpu::attention(
                &query(position),
                &history(&cpu[0].1),
                &history(&cpu[1].1),
                heads,
                dim,
                Some(kv_heads),
                None,
                true,
            )
            .unwrap();
            assert_close(&download(&kernels, &captured), &expected);
        }
    }
}

/// `cache-latent-attention-history`: a decode step over an f32 latent
/// history read as both K and V (16 query heads over 576-value rows, the
/// values the leading 512; a block serving eight heads) matches the CPU's
/// attention, for short and longer histories and a sliding window.
#[test]
fn latent_decode_matches_the_cpu() {
    use magnetar_provider_api::value_heads;
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (heads, width, latent) = (16u64, 576u64, 512u64);
    for (keys, window) in [
        (33u64, None),
        (300, None),
        (1100, None),
        (700, Some(300u64)),
    ] {
        let capacity = keys + 5;
        let data = hashed_values((capacity * width) as usize, 21 + keys as usize);
        let store = upload(
            &kernels,
            &HostTensor::new([capacity, width], data.clone()).unwrap(),
        );
        let history =
            HostTensor::new([keys, width], data[..(keys * width) as usize].to_vec()).unwrap();
        let history_values = value_heads(&history, 1, width, latent).unwrap();
        let q = HostTensor::new(
            [1, heads * width],
            attention_values((heads * width) as usize, 13),
        )
        .unwrap();
        let expected = magnetar_provider_cpu::attention_with_value_width(
            &q,
            &history,
            &history_values,
            heads,
            width,
            latent,
            Some(1),
            window,
            true,
            Some(0.1),
        )
        .unwrap();
        let actual = kernels
            .attention_strided_values(
                &upload(&kernels, &q),
                &store,
                &store,
                heads,
                width,
                Some(latent),
                width,
                Some(1),
                window,
                true,
                Some(keys),
                Some(0.1),
            )
            .unwrap();
        assert_close(&download(&kernels, &actual), &expected);
    }
}

/// `cache-latent-attention-history`, `speed-up-decode-attention-on-codes`:
/// a decode step over a latent TurboQuant store read as both K and V (16
/// query heads over 576-value rows in groups of 64, the values the leading
/// 512) runs on its codes, and matches the CPU's attention over the CPU
/// store's values, as the path through dequantized scratch does: short and
/// longer histories, an exact head ending inside a chunk and one past
/// 1,024 rows, a sliding window starting inside the head, `tq4` and `tq3`.
#[test]
fn turboquant_latent_decode_on_codes_matches_the_cpu() {
    use magnetar_provider_api::{KvStorePrecision, TurboQuantStoreState, value_heads};
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (heads, width, latent, group) = (16u64, 576u64, 512u64, 64u64);
    for precision in [KvStorePrecision::Tq4, KvStorePrecision::Tq3] {
        for (keys, exact, window) in [
            (33u64, 0u64, None),
            (300, 0, None),
            (300, 100, None),
            (1100, 1024, None),
            (700, 600, Some(300u64)),
        ] {
            let label = format!("{} {keys} keys, head {exact}", precision.as_str());
            let capacity = keys + 7;
            let data = hashed_values((keys * width) as usize, 11 + keys as usize);
            let split = ((keys - 1) * width) as usize;
            let mut store = kernels
                .tq_zeros_with_exact_head(&[capacity, width], precision, group, exact)
                .unwrap();
            for (from, rows, offset) in [(0, keys - 1, 0), (split, 1, keys - 1)] {
                let tensor = HostTensor::new(
                    [rows, width],
                    data[from..from + (rows * width) as usize].to_vec(),
                )
                .unwrap();
                kernels
                    .kv_write(&mut store, &upload(&kernels, &tensor), offset)
                    .unwrap();
            }
            let mut cpu = TurboQuantStoreState::new(precision, group, width).with_exact_rows(exact);
            let mut values = cpu.write(&data[..split], 0).unwrap();
            values.extend(cpu.write(&data[split..], keys - 1).unwrap());
            let rounded = HostTensor::new([keys, width], values).unwrap();
            let rounded_values = value_heads(&rounded, 1, width, latent).unwrap();
            let q = HostTensor::new(
                [1, heads * width],
                attention_values((heads * width) as usize, 9),
            )
            .unwrap();
            let expected = magnetar_provider_cpu::attention_with_value_width(
                &q,
                &rounded,
                &rounded_values,
                heads,
                width,
                latent,
                Some(1),
                window,
                true,
                Some(0.1),
            )
            .unwrap();
            for on_codes in [true, false] {
                kernels.set_tq_latent_on_codes(on_codes);
                let actual = kernels
                    .attention_strided_values(
                        &upload(&kernels, &q),
                        &store,
                        &store,
                        heads,
                        width,
                        Some(latent),
                        width,
                        Some(1),
                        window,
                        true,
                        Some(keys),
                        Some(0.1),
                    )
                    .unwrap();
                let actual = download(&kernels, &actual);
                assert_eq!(actual.shape, expected.shape, "{label}");
                let difference = actual
                    .data
                    .iter()
                    .zip(&expected.data)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    difference <= TOLERANCE,
                    "{label}, on codes {on_codes}: {difference}"
                );
            }
            kernels.set_tq_latent_on_codes(true);
        }
    }
}

/// `cache-latent-attention-history`: `batched-matmul` matches the CPU's,
/// and attention over a latent history read as its own V
/// (`value_head_stride`) matches the CPU's attention over the extracted
/// values, for a decode step over a store and a prefill.
#[test]
fn batched_matmul_and_strided_values_match_the_cpu() {
    use magnetar_provider_api::{batched_matmul, value_heads};
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (heads, nope, latent, rope) = (16u64, 128u64, 512u64, 64u64);
    for rows in [1u64, 5] {
        let a = HostTensor::new(
            [rows, heads * nope],
            attention_values((rows * heads * nope) as usize, 3),
        )
        .unwrap();
        let b = HostTensor::new(
            [heads * nope, latent],
            attention_values((heads * nope * latent) as usize, 4),
        )
        .unwrap();
        let actual = kernels
            .batched_matmul(&upload(&kernels, &a), &upload(&kernels, &b), heads, false)
            .unwrap();
        assert_close(
            &download(&kernels, &actual),
            &batched_matmul(&a, &b, heads, false).unwrap(),
        );
        // `transpose_b`: per-head `[n, k]` blocks, as the value rows of the
        // latent up-projection are stored.
        let latent_rows = HostTensor::new(
            [rows, heads * latent],
            attention_values((rows * heads * latent) as usize, 7),
        )
        .unwrap();
        let values = HostTensor::new(
            [heads * nope, latent],
            attention_values((heads * nope * latent) as usize, 8),
        )
        .unwrap();
        let actual = kernels
            .batched_matmul(
                &upload(&kernels, &latent_rows),
                &upload(&kernels, &values),
                heads,
                true,
            )
            .unwrap();
        assert_close(
            &download(&kernels, &actual),
            &batched_matmul(&latent_rows, &values, heads, true).unwrap(),
        );
    }
    let width = latent + rope;
    let (capacity, keys) = (40u64, 33u64);
    let history = HostTensor::new(
        [capacity, width],
        attention_values((capacity * width) as usize, 5),
    )
    .unwrap();
    let values = value_heads(&history, 1, width, latent).unwrap();
    let leading = |tensor: &HostTensor, rows: u64| {
        HostTensor::new(
            [rows, tensor.shape[1]],
            tensor.data[..(rows * tensor.shape[1]) as usize].to_vec(),
        )
        .unwrap()
    };
    for queries in [1u64, 4] {
        let q = HostTensor::new(
            [queries, heads * width],
            attention_values((queries * heads * width) as usize, 6),
        )
        .unwrap();
        let expected = magnetar_provider_cpu::attention_with_value_width(
            &q,
            &leading(&history, keys),
            &leading(&values, keys),
            heads,
            width,
            latent,
            Some(1),
            None,
            true,
            Some(0.1),
        )
        .unwrap();
        let store = upload(&kernels, &history);
        let actual = kernels
            .attention_strided_values(
                &upload(&kernels, &q),
                &store,
                &store,
                heads,
                width,
                Some(latent),
                width,
                Some(1),
                None,
                true,
                Some(keys),
                Some(0.1),
            )
            .unwrap();
        assert_close(&download(&kernels, &actual), &expected);
    }

    // `turboquant-kv-history` on the latent history: a tq4 store of
    // 576-value rows in groups of 64, read as both K and V, against the
    // CPU's attention over the same rounded rows.
    use magnetar_provider_api::{KvStorePrecision, TurboQuantStoreState};
    let group = 64u64;
    // Hashed rows: no rotated coordinate on a codebook boundary.
    let latent_rows = hashed_values((keys * width) as usize, 11);
    let mut store = kernels
        .tq_zeros(&[capacity, width], KvStorePrecision::Tq4, group)
        .unwrap();
    kernels
        .kv_write(
            &mut store,
            &upload(
                &kernels,
                &HostTensor::new([keys, width], latent_rows.clone()).unwrap(),
            ),
            0,
        )
        .unwrap();
    let mut rounded = TurboQuantStoreState::new(KvStorePrecision::Tq4, group, width);
    let rounded = HostTensor::new([keys, width], rounded.write(&latent_rows, 0).unwrap()).unwrap();
    let rounded_values = value_heads(&rounded, 1, width, latent).unwrap();
    for queries in [1u64, 4] {
        let q = HostTensor::new(
            [queries, heads * width],
            attention_values((queries * heads * width) as usize, 9),
        )
        .unwrap();
        let expected = magnetar_provider_cpu::attention_with_value_width(
            &q,
            &rounded,
            &rounded_values,
            heads,
            width,
            latent,
            Some(1),
            None,
            true,
            Some(0.1),
        )
        .unwrap();
        let actual = kernels
            .attention_strided_values(
                &upload(&kernels, &q),
                &store,
                &store,
                heads,
                width,
                Some(latent),
                width,
                Some(1),
                None,
                true,
                Some(keys),
                Some(0.1),
            )
            .unwrap();
        assert_close(&download(&kernels, &actual), &expected);
    }
}

/// Pseudo-random values in [-0.5, 0.5) from splitmix64: no arithmetic
/// structure for a rotation to line up.
fn hashed_values(len: usize, seed: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ (seed as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            (z >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
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

/// Banded per-frequency scaling matches the CPU Provider, across the
/// kept, smoothed and divided frequency bands, interleaved and half-split.
#[test]
fn rope_banded_frequency_scaling_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let scaling = magnetar_core::operator::RopeBandedFrequencyScaling {
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

/// Device time of one decode step's attention -- a query row over a
/// history of each length -- in f32 stores and in `tq4` and `tq3` stores
/// with every row on codes and with an exact head of 1,024 rows
/// (`keep-turboquant-kv-head-exact`), for Qwen2.5-0.5B's heads (14 query,
/// 2 KV, 64 wide) and Mistral 7B's (32, 8, 128), with the bandwidth each
/// reaches on the bytes it must read. Not a conformance check: run it with
/// `--ignored --nocapture` on a GPU.
#[test]
#[ignore = "GPU timing diagnostic"]
fn decode_attention_times() {
    use magnetar_provider_api::KvStorePrecision;
    if cfg!(debug_assertions) {
        return;
    }
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let context = kernels.stream().context().clone();
    let timing = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let time_us = |repeats: usize, run: &dyn Fn()| -> f32 {
        run();
        let start = context.new_event(timing).unwrap();
        let end = context.new_event(timing).unwrap();
        start.record(kernels.stream()).unwrap();
        for _ in 0..repeats {
            run();
        }
        end.record(kernels.stream()).unwrap();
        start.elapsed_ms(&end).unwrap() * 1e3 / repeats as f32
    };
    // Local memory per thread means an array left the registers.
    for name in [
        "attention_split_gqa_kernel",
        "attention_split_gqa128_kernel",
        "attention_combine_kernel",
        "tq_attention_split_grouped_kernel",
        "tq_attention_split_staged_kernel",
        "tq_attention_split_kernel",
        "tq_rotate_back_kernel",
    ] {
        let [registers, local, shared] = kernels.kernel_resources(name).unwrap();
        eprintln!(
            "{name}: {registers} registers, {local} bytes of local memory per thread, \
             {shared} bytes of static shared memory"
        );
    }
    for (label, heads, kv_heads, dim) in [
        ("Qwen2.5-0.5B", 14u64, 2u64, 64u64),
        ("Mistral 7B", 32, 8, 128),
    ] {
        let width = kv_heads * dim;
        let q = upload(
            &kernels,
            &HostTensor::new([1, heads * dim], hashed_values((heads * dim) as usize, 11)).unwrap(),
        );
        for keys in [512u64, 1024, 4096, 16384, 32768] {
            let rows = |seed| {
                HostTensor::new([keys, width], hashed_values((keys * width) as usize, seed))
                    .unwrap()
            };
            let (k_rows, v_rows) = (rows(7), rows(9));
            let attend = |k: &CudaDeviceBuffer, v: &CudaDeviceBuffer| {
                time_us(50, &|| {
                    kernels
                        .attention_valued(
                            &q,
                            k,
                            v,
                            heads,
                            dim,
                            None,
                            Some(kv_heads),
                            None,
                            true,
                            Some(keys),
                            None,
                        )
                        .unwrap();
                })
            };
            let f32_us = attend(&upload(&kernels, &k_rows), &upload(&kernels, &v_rows));
            let tq = |precision: KvStorePrecision, exact: u64| {
                let store = |tensor: &HostTensor| {
                    let mut store = kernels
                        .tq_zeros_with_exact_head(&[keys, width], precision, dim, exact)
                        .unwrap();
                    kernels
                        .kv_write(&mut store, &upload(&kernels, tensor), 0)
                        .unwrap();
                    store
                };
                attend(&store(&k_rows), &store(&v_rows))
            };
            // The bytes read: K and V rows in f32; codes and a norm per
            // head row in TurboQuant, with the head's rows in f32.
            let f32_bytes = (2 * keys * width * 4) as f32;
            let tq_bytes = |bits: u64, exact: u64| {
                let exact = exact.min(keys);
                let codes = (keys - exact) * kv_heads * (dim * bits / 8 + 4);
                (2 * (codes + exact * width * 4)) as f32
            };
            let mut line = format!(
                "decode attention, {label}, {keys} keys: f32 {f32_us:.1} us ({:.0} GB/s)",
                f32_bytes / f32_us / 1e3
            );
            // Every schedule of the code Kernels on codes alone
            // (`speed-up-decode-attention-on-codes`), and the default one
            // with an exact head.
            for (precision, bits) in [(KvStorePrecision::Tq4, 4), (KvStorePrecision::Tq3, 3)] {
                for (schedule, exact, name) in [
                    (TqDecodeSchedule::Unstaged, 0u64, "unstaged"),
                    (
                        TqDecodeSchedule::ChunkPerBlock,
                        0,
                        "staged, a chunk per block",
                    ),
                    (TqDecodeSchedule::Split, 0, "staged and split"),
                    (
                        TqDecodeSchedule::Unstaged,
                        1024,
                        "unstaged, a 1,024-row head",
                    ),
                ] {
                    kernels.set_tq_decode_schedule(schedule);
                    let us = tq(precision, exact);
                    line.push_str(&format!(
                        ", {} {name} {us:.1} us ({:.0} GB/s)",
                        precision.as_str(),
                        tq_bytes(bits, exact) / us / 1e3
                    ));
                }
            }
            kernels.set_tq_decode_schedule(TqDecodeSchedule::Unstaged);
            eprintln!("{line}");
        }
    }
}

/// Device time of DeepSeek-V2-Lite's absorbed decode attention
/// (`cache-latent-attention-history`): 16 query heads over one latent
/// history of 576-value rows (512 latent values and 64 rope ones), the
/// values each row's leading 512, scale 0.1; in an f32 store and in `tq4`
/// stores (groups of 64): through dequantized scratch, on the codes, and on
/// the codes with a 1,024-row exact head, over
/// 1,024 to 32,768 keys. Not a conformance check: run it with `--ignored
/// --nocapture` on a GPU.
#[test]
#[ignore = "GPU timing diagnostic"]
fn latent_decode_attention_times() {
    use magnetar_provider_api::KvStorePrecision;
    if cfg!(debug_assertions) {
        return;
    }
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let context = kernels.stream().context().clone();
    let timing = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let time_us = |repeats: usize, run: &dyn Fn()| -> f32 {
        run();
        let start = context.new_event(timing).unwrap();
        let end = context.new_event(timing).unwrap();
        start.record(kernels.stream()).unwrap();
        for _ in 0..repeats {
            run();
        }
        end.record(kernels.stream()).unwrap();
        start.elapsed_ms(&end).unwrap() * 1e3 / repeats as f32
    };
    for name in [
        "latent_attention_split_kernel",
        "tq_latent_attention_split_kernel",
    ] {
        let [registers, local, shared] = kernels.kernel_resources(name).unwrap();
        eprintln!(
            "{name}: {registers} registers, {local} bytes of local memory per thread, \
             {shared} bytes of static shared memory"
        );
    }
    let (heads, width, latent, group) = (16u64, 576u64, 512u64, 64u64);
    let q = upload(
        &kernels,
        &HostTensor::new(
            [1, heads * width],
            hashed_values((heads * width) as usize, 11),
        )
        .unwrap(),
    );
    for keys in [1024u64, 4096, 16384, 32768] {
        let rows =
            HostTensor::new([keys, width], hashed_values((keys * width) as usize, 7)).unwrap();
        let attend = |store: &CudaDeviceBuffer| {
            time_us(20, &|| {
                kernels
                    .attention_strided_values(
                        &q,
                        store,
                        store,
                        heads,
                        width,
                        Some(latent),
                        width,
                        Some(1),
                        None,
                        true,
                        Some(keys),
                        Some(0.1),
                    )
                    .unwrap();
            })
        };
        let f32_us = attend(&upload(&kernels, &rows));
        let tq = |exact: u64, on_codes: bool| {
            let mut store = kernels
                .tq_zeros_with_exact_head(&[keys, width], KvStorePrecision::Tq4, group, exact)
                .unwrap();
            kernels
                .kv_write(&mut store, &upload(&kernels, &rows), 0)
                .unwrap();
            kernels.set_tq_latent_on_codes(on_codes);
            let us = attend(&store);
            kernels.set_tq_latent_on_codes(true);
            us
        };
        let (scratch_us, codes_us, head_us) = (tq(0, false), tq(0, true), tq(1024, true));
        let f32_bytes = (keys * width * 4) as f32;
        let tq_bytes = (keys * (width / group) * (group / 2 + 4)) as f32;
        eprintln!(
            "latent decode attention, DeepSeek-V2-Lite, {keys} keys: f32 {f32_us:.1} us \
             ({:.0} GB/s), tq4 through dequantized scratch {scratch_us:.1} us, tq4 on codes \
             {codes_us:.1} us ({:.0} GB/s), tq4 on codes with a 1,024-row head {head_us:.1} us",
            f32_bytes / f32_us / 1e3,
            tq_bytes / codes_us / 1e3,
        );
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

/// The Device time and transient bytes of an embedding lookup from a
/// packed table, one id (a decode step) and 512 (a prefill), at the shapes
/// of `gemma-4-E2B-it-qat-q4_0-gguf` (each of its 35 per-layer tables,
/// `[262144, 256]`, and its token table, `[262144, 1536]`, both `Q6_K`)
/// and of a `Q4_K` `[32000, 4096]` token table: the lookup reading the
/// looked-up rows against dequantizing the whole table first, as it did
/// before. Not a conformance check: run it with `--ignored --nocapture`
/// on a GPU.
#[test]
#[ignore = "GPU timing diagnostic"]
fn packed_embedding_lookup_times() {
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
    let gb = |bytes: u64| bytes as f64 / 1e9;
    for (name, table) in [
        (
            "Q6_K per-layer [262144, 256]",
            packed_q6k(262_144, 256, false, 81),
        ),
        (
            "Q6_K token [262144, 1536]",
            packed_q6k(262_144, 1536, false, 82),
        ),
        (
            "Q4_K token [32000, 4096]",
            packed_q4k(32_000, 4096, false, 83),
        ),
    ] {
        let [rows, cols] = table.logical_shape();
        let table_dev = kernels.upload_packed_weight(&table).unwrap();
        drop(table);
        for count in [1_u64, 512] {
            let ids = upload(
                &kernels,
                &HostTensor::new(
                    [count],
                    (0..count)
                        .map(|i| ((i * 7919) % rows) as f32)
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
            );
            let rows_ms = time(20, &|| {
                kernels.embedding_lookup(&table_dev, &ids).unwrap();
            });
            let whole_ms = time(3, &|| {
                let dense = kernels.dequantize_q8(&table_dev).unwrap();
                kernels.embedding_lookup(&dense, &ids).unwrap();
            });
            eprintln!(
                "embedding lookup, {name}, {count} ids: looked-up rows {:.1} us \
                 ({:.6} GB written), whole table first {:.1} us ({:.3} GB written)",
                rows_ms * 1e3,
                gb(count * cols * 4),
                whole_ms * 1e3,
                gb((rows + count) * cols * 4),
            );
        }
    }
}

/// The Device time of a mixture-of-experts decode step's Kernels at
/// Qwen1.5-MoE-A2.7B's shapes (hidden 2048, 60 experts of 1408, top 4, a
/// 5632 shared expert, 2048-wide attention projections, a 151936-token
/// output head), packed `Q8_0`, one token: each Kernel's time and the
/// bandwidth of the weight bytes it must read. Also the routed experts
/// split in two (30 held), as each GPU runs them under tensor parallelism.
#[test]
#[ignore = "GPU timing diagnostic"]
fn moe_decode_kernel_times() {
    use magnetar_provider_api::MoeRouting;
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let context = kernels.stream().context().clone();
    let timing = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let repeats = 100;
    let time = |run: &dyn Fn()| -> f32 {
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
    let q8_bytes = |elements: u64| elements / 32 * 34;
    let report = |what: &str, ms: f32, bytes: u64| {
        eprintln!(
            "{what}: {:.1} us ({:.0} GB/s over {:.1} MB)",
            ms * 1e3,
            bytes as f32 / ms / 1e6,
            bytes as f32 / 1e6
        );
    };
    let (hidden, inter, experts, top_k) = (2048_u64, 1408_u64, 60_u64, 4_u64);
    let h = upload(&kernels, &activations(1, hidden, 61));
    let logits = upload(&kernels, &activations(1, experts, 62));
    let routing = MoeRouting {
        experts,
        top_k,
        normalize: false,
    };
    for held in [experts, experts / 2] {
        let gate = kernels
            .upload_packed_weight(&packed_q8(held * inter, hidden, true, 63))
            .unwrap();
        let up = kernels
            .upload_packed_weight(&packed_q8(held * inter, hidden, true, 64))
            .unwrap();
        let down = kernels
            .upload_packed_weight(&packed_q8(held * hidden, inter, true, 65))
            .unwrap();
        let ms = time(&|| {
            kernels
                .moe_experts(&h, &logits, [&gate, &up, &down], routing, (0, held))
                .unwrap();
        });
        // At most top_k experts' three matrices are read.
        let routed = top_k.min(held);
        report(
            &format!("moe-experts, {held} of {experts} experts held"),
            ms,
            routed * 3 * q8_bytes(hidden * inter),
        );
    }
    for (what, k, n) in [
        (
            "attention projection [1, 2048] x [2048, 2048]",
            hidden,
            hidden,
        ),
        ("shared expert up [1, 2048] x [2048, 5632]", hidden, 5632),
        ("shared expert down [1, 5632] x [5632, 2048]", 5632, hidden),
        ("router [1, 2048] x [2048, 60]", hidden, experts),
        ("output head [1, 2048] x [2048, 151936]", hidden, 151_936),
        ("Mistral 7B q/o [1, 4096] x [4096, 4096]", 4096, 4096),
        ("Mistral 7B k/v [1, 4096] x [4096, 1024]", 4096, 1024),
        ("Mistral 7B gate/up [1, 4096] x [4096, 14336]", 4096, 14_336),
        ("Mistral 7B down [1, 14336] x [14336, 4096]", 14_336, 4096),
    ] {
        let a = upload(&kernels, &activations(1, k, 66));
        let b = kernels
            .upload_packed_weight(&packed_q8(n, k, true, 67))
            .unwrap();
        let ms = time(&|| {
            kernels.matmul(&a, &b, false, false).unwrap();
        });
        report(what, ms, q8_bytes(k * n));
    }
    // The same projections packed `Q4_K` (144 bytes per 256 values).
    for (what, k, n) in [
        (
            "Q4_K attention projection [1, 2048] x [2048, 2048]",
            hidden,
            hidden,
        ),
        ("Q4_K Mistral 7B q/o [1, 4096] x [4096, 4096]", 4096, 4096),
        (
            "Q4_K Mistral 7B gate/up [1, 4096] x [4096, 14336]",
            4096,
            14_336,
        ),
        (
            "Q4_K Mistral 7B down [1, 14336] x [14336, 4096]",
            14_336,
            4096,
        ),
        (
            "Q4_K output head [1, 2048] x [2048, 151936]",
            hidden,
            151_936,
        ),
    ] {
        let a = upload(&kernels, &activations(1, k, 68));
        let b = kernels
            .upload_packed_weight(&packed_q4k(n, k, true, 69))
            .unwrap();
        let ms = time(&|| {
            kernels.matmul(&a, &b, false, false).unwrap();
        });
        report(what, ms, k * n / 256 * 144);
    }
    // `Q6_K` (210 bytes per 256 values): `Q4_K_M`'s output head, `attn_v`
    // and half of its `ffn_down`.
    for (what, k, n) in [
        ("Q6_K Mistral 7B v [1, 4096] x [4096, 1024]", 4096, 1024),
        (
            "Q6_K Mistral 7B down [1, 14336] x [14336, 4096]",
            14_336,
            4096,
        ),
        (
            "Q6_K Mistral 7B output head [1, 4096] x [4096, 32768]",
            4096,
            32_768,
        ),
    ] {
        let a = upload(&kernels, &activations(1, k, 70));
        let b = kernels
            .upload_packed_weight(&packed_q6k(n, k, true, 71))
            .unwrap();
        let ms = time(&|| {
            kernels.matmul(&a, &b, false, false).unwrap();
        });
        report(what, ms, k * n / 256 * 210);
    }
    // The Mistral shapes again with int8 activations (`__dp4a`).
    kernels.set_int8_activations(true);
    for (format, k, n) in [
        ("Q8_0", 4096, 4096),
        ("Q8_0", 4096, 14_336),
        ("Q8_0", 14_336, 4096),
        ("Q4_K", 4096, 4096),
        ("Q4_K", 4096, 14_336),
        ("Q4_K", 14_336, 4096),
        ("Q6_K", 4096, 1024),
        ("Q6_K", 14_336, 4096),
        ("Q6_K", 4096, 32_768),
        ("Q3_K", 4096, 4096),
        ("Q3_K", 4096, 14_336),
        ("Q3_K", 14_336, 4096),
        ("Q5_K", 4096, 14_336),
        ("Q5_K", 14_336, 4096),
        ("Q2_K", 4096, 4096),
        ("Q2_K", 4096, 14_336),
        ("Q2_K", 14_336, 4096),
        ("IQ4_XS", 4096, 4096),
        ("IQ4_XS", 4096, 14_336),
        ("IQ4_XS", 14_336, 4096),
        ("IQ4_NL", 4096, 14_336),
        ("IQ3_XXS", 4096, 14_336),
        ("IQ3_S", 4096, 14_336),
        ("IQ2_XXS", 4096, 14_336),
        ("IQ2_XS", 4096, 14_336),
        ("IQ2_S", 4096, 14_336),
    ] {
        let a = upload(&kernels, &activations(1, k, 72));
        let (packed, bytes) = match format {
            "Q8_0" => (packed_q8(n, k, true, 73), q8_bytes(k * n)),
            "Q4_K" => (packed_q4k(n, k, true, 73), k * n / 256 * 144),
            "Q6_K" => (packed_q6k(n, k, true, 73), k * n / 256 * 210),
            "Q5_K" => (packed_q5k(n, k, 73), k * n / 256 * 176),
            "Q3_K" => (
                packed_low_bit(PackedWeightFormat::Q3K, n, k, 73),
                k * n / 256 * 110,
            ),
            "Q2_K" => (
                packed_low_bit(PackedWeightFormat::Q2K, n, k, 73),
                k * n / 256 * 84,
            ),
            "IQ4_XS" => (
                packed_iq4(PackedWeightFormat::Iq4Xs, n, k, 73),
                k * n / 256 * 136,
            ),
            "IQ3_XXS" | "IQ3_S" | "IQ2_XXS" | "IQ2_XS" | "IQ2_S" => {
                let format = match format {
                    "IQ3_XXS" => PackedWeightFormat::Iq3Xxs,
                    "IQ3_S" => PackedWeightFormat::Iq3S,
                    "IQ2_XXS" => PackedWeightFormat::Iq2Xxs,
                    "IQ2_XS" => PackedWeightFormat::Iq2Xs,
                    _ => PackedWeightFormat::Iq2S,
                };
                (
                    packed_iq4(format, n, k, 73),
                    k * n / 256 * format.block_bytes(),
                )
            }
            _ => (
                packed_iq4(PackedWeightFormat::Iq4Nl, n, k, 73),
                k * n / 32 * 18,
            ),
        };
        let b = kernels.upload_packed_weight(&packed).unwrap();
        let ms = time(&|| {
            kernels.matmul(&a, &b, false, false).unwrap();
        });
        report(&format!("int8 {format} [1, {k}] x [{k}, {n}]"), ms, bytes);
    }
    kernels.set_int8_activations(false);
}

/// A deterministic packed `Q8_0` matrix of `stored_rows` x `stored_cols`
/// (`execute-direct-quantized-weights`), with varied scales and quants
/// covering the full signed 8-bit range.
fn packed_q8(stored_rows: u64, stored_cols: u64, transposed: bool, seed: u32) -> PackedWeight {
    let blocks = stored_rows * stored_cols / 32;
    let mut bytes = Vec::with_capacity(blocks as usize * 34);
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for _ in 0..blocks {
        // f16 scales between about 2^-9 and 2^-5.
        let exponent = 6 + (next() % 5) as u16;
        let d_bits = (exponent << 10) | (next() % 1024) as u16;
        bytes.extend(d_bits.to_le_bytes());
        bytes.extend(
            (0..32)
                .map(|_| (next() % 255) as i32 - 127)
                .map(|q| q as i8 as u8),
        );
    }
    PackedWeight::new(
        PackedWeightFormat::Q8_0,
        stored_rows,
        stored_cols,
        transposed,
        bytes,
    )
    .unwrap()
}

fn activations(rows: u64, cols: u64, seed: u32) -> HostTensor {
    let data = (0..rows * cols)
        .map(|i| {
            (((i as u32).wrapping_mul(2_246_822_519).wrapping_add(seed) % 2001) as f32 - 1000.0)
                / 1000.0
        })
        .collect::<Vec<_>>();
    HostTensor::new([rows, cols], data).unwrap()
}

/// `actual` against `expected` with a tolerance scaled to the products'
/// magnitude: the packed kernels sum in another order than the Reference.
fn assert_close_scaled(actual: &HostTensor, expected: &HostTensor, scale: f32) {
    assert_eq!(actual.shape, expected.shape);
    for (index, (a, e)) in actual.data.iter().zip(&expected.data).enumerate() {
        assert!(
            (a - e).abs() <= 1e-5 + 1e-5 * scale,
            "index {index}: {a} vs {e}"
        );
    }
}

/// The packed `Q8_0` matmul (decode rows read packed, longer ones
/// dequantized for the call) matches the Reference CPU matmul of the
/// dequantized weight, with and without a bias or residual addend, and
/// download/expand give back exactly the dequantized weight.
#[test]
fn q8_packed_matmul_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (k, n) = (256_u64, 96_u64);
    // Stored as n rows of k quants: the graph sees [k, n].
    let packed = packed_q8(n, k, true, 7);
    let dense = packed.dequantize();
    let weight = kernels.upload_packed_weight(&packed).unwrap();
    assert!(weight.is_q8() && weight.is_compact());
    assert_eq!(weight.shape, vec![k, n]);
    assert_eq!(download(&kernels, &weight).data, dense.data);
    assert_eq!(
        download(&kernels, &kernels.expand(&weight).unwrap()).data,
        dense.data
    );

    let bias = HostTensor::new([n], (0..n).map(|i| i as f32 * 0.01).collect::<Vec<_>>()).unwrap();
    for m in [1_u64, 3, 8, 64, 65] {
        let a = activations(m, k, m as u32);
        let expected = magnetar_provider_cpu::matmul(&a, &dense, false, false).unwrap();
        let a_dev = upload(&kernels, &a);
        let actual = download(
            &kernels,
            &kernels.matmul(&a_dev, &weight, false, false).unwrap(),
        );
        assert_close_scaled(&actual, &expected, k as f32);

        let with_bias = kernels
            .matmul_bias(&a_dev, &weight, &upload(&kernels, &bias), false, false)
            .unwrap();
        let expected_bias = magnetar_provider_cpu::add(&expected, &bias).unwrap();
        assert_close_scaled(&download(&kernels, &with_bias), &expected_bias, k as f32);

        let residual = activations(m, n, 99);
        let with_residual = kernels
            .matmul_residual(&a_dev, &weight, &upload(&kernels, &residual), false, false)
            .unwrap();
        let expected_residual = magnetar_provider_cpu::add(&expected, &residual).unwrap();
        assert_close_scaled(
            &download(&kernels, &with_residual),
            &expected_residual,
            k as f32,
        );
    }
}

/// A deterministic packed `Q4_K` matrix of `stored_rows` x `stored_cols`
/// super-blocks: small positive `d`/`dmin`, every 6-bit scale and min, and
/// every nibble, exercised.
fn packed_q4k(stored_rows: u64, stored_cols: u64, transposed: bool, seed: u32) -> PackedWeight {
    let blocks = stored_rows * stored_cols / 256;
    let mut bytes = Vec::with_capacity(blocks as usize * 144);
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for _ in 0..blocks {
        for _ in 0..2 {
            // f16 between about 2^-10 and 2^-6.
            let exponent = 5 + (next() % 5) as u16;
            let bits = (exponent << 10) | (next() % 1024) as u16;
            bytes.extend(bits.to_le_bytes());
        }
        bytes.extend((0..12).map(|_| (next() % 256) as u8));
        bytes.extend((0..128).map(|_| (next() % 256) as u8));
    }
    PackedWeight::new(
        PackedWeightFormat::Q4K,
        stored_rows,
        stored_cols,
        transposed,
        bytes,
    )
    .unwrap()
}

/// The packed `Q4_K` weight: downloading or expanding it gives back exactly
/// the Runtime's dequantization, and its decode matmul (rows read packed,
/// longer ones dequantized for the call), with and without a bias or
/// residual addend, and a gated matmul over two of them, match the
/// Reference CPU on the dequantized weights.
#[test]
fn q4k_packed_matmul_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (k, n) = (1024_u64, 97_u64); // odd: a warp with one column left
    let packed = packed_q4k(n, k, true, 21);
    let dense = packed.dequantize();
    let weight = kernels.upload_packed_weight(&packed).unwrap();
    assert!(weight.is_q8() && weight.is_compact());
    assert_eq!(weight.shape, vec![k, n]);
    assert_eq!(weight.len(), (k * n) as usize);
    assert_eq!(download(&kernels, &weight).data, dense.data);
    assert_eq!(
        download(&kernels, &kernels.expand(&weight).unwrap()).data,
        dense.data
    );
    let bias = HostTensor::new([n], (0..n).map(|i| i as f32 * 0.01).collect::<Vec<_>>()).unwrap();
    for m in [1_u64, 3, 8, 9, 65] {
        let a = activations(m, k, m as u32 + 40);
        let expected = magnetar_provider_cpu::matmul(&a, &dense, false, false).unwrap();
        let a_dev = upload(&kernels, &a);
        let actual = download(
            &kernels,
            &kernels.matmul(&a_dev, &weight, false, false).unwrap(),
        );
        assert_close_relative(&actual, &expected);
        let with_bias = kernels
            .matmul_bias(&a_dev, &weight, &upload(&kernels, &bias), false, false)
            .unwrap();
        let expected_bias = magnetar_provider_cpu::add(&expected, &bias).unwrap();
        assert_close_relative(&download(&kernels, &with_bias), &expected_bias);
        let residual = activations(m, n, 98);
        let with_residual = kernels
            .matmul_residual(&a_dev, &weight, &upload(&kernels, &residual), false, false)
            .unwrap();
        let expected_residual = magnetar_provider_cpu::add(&expected, &residual).unwrap();
        assert_close_relative(&download(&kernels, &with_residual), &expected_residual);
    }
    let up = packed_q4k(n, k, true, 22);
    let up_dense = up.dequantize();
    let up_weight = kernels.upload_packed_weight(&up).unwrap();
    for m in [1_u64, 5] {
        let a = activations(m, k, 7);
        let expected = magnetar_provider_cpu::mul(
            &magnetar_provider_cpu::silu(
                &magnetar_provider_cpu::matmul(&a, &dense, false, false).unwrap(),
            ),
            &magnetar_provider_cpu::matmul(&a, &up_dense, false, false).unwrap(),
        )
        .unwrap();
        let actual = kernels
            .gated_matmul(&upload(&kernels, &a), &weight, &up_weight, false, false)
            .unwrap();
        assert_close_relative(&download(&kernels, &actual), &expected);
    }
}

/// A deterministic packed `Q5_K` matrix, stored transposed: `Q4_K`'s
/// header and scales, then random high-bit planes and nibbles.
fn packed_q5k(stored_rows: u64, stored_cols: u64, seed: u32) -> PackedWeight {
    let blocks = stored_rows * stored_cols / 256;
    let mut bytes = Vec::with_capacity(blocks as usize * 176);
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for _ in 0..blocks {
        for _ in 0..2 {
            // f16 between about 2^-11 and 2^-7.
            let exponent = 4 + (next() % 5) as u16;
            let bits = (exponent << 10) | (next() % 1024) as u16;
            bytes.extend(bits.to_le_bytes());
        }
        bytes.extend((0..172).map(|_| (next() % 256) as u8));
    }
    PackedWeight::new(
        PackedWeightFormat::Q5K,
        stored_rows,
        stored_cols,
        true,
        bytes,
    )
    .unwrap()
}

/// The packed `Q5_K` weight: downloading or expanding it gives back exactly
/// the Runtime's dequantization, and its decode matmul with a residual
/// addend matches the Reference CPU on the rounded activations with int8
/// activations, and on the f32 ones without (and for longer matmuls).
#[test]
fn q5k_packed_matmul_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (k, n) = (1024_u64, 97_u64); // odd: a warp with one column left
    let packed = packed_q5k(n, k, 51);
    let dense = packed.dequantize();
    let weight = kernels.upload_packed_weight(&packed).unwrap();
    assert!(weight.is_q8() && weight.is_compact());
    assert_eq!(weight.shape, vec![k, n]);
    assert_eq!(download(&kernels, &weight).data, dense.data);
    assert_eq!(
        download(&kernels, &kernels.expand(&weight).unwrap()).data,
        dense.data
    );
    for int8 in [true, false] {
        kernels.set_int8_activations(int8);
        for m in [1_u64, 3, 9, 65] {
            let a = activations(m, k, m as u32 + 50);
            let seen = if int8 && m <= 64 {
                int8_rounded(&a)
            } else {
                a.clone()
            };
            let residual = activations(m, n, 52);
            let expected = magnetar_provider_cpu::add(
                &magnetar_provider_cpu::matmul(&seen, &dense, false, false).unwrap(),
                &residual,
            )
            .unwrap();
            let actual = kernels
                .matmul_residual(
                    &upload(&kernels, &a),
                    &weight,
                    &upload(&kernels, &residual),
                    false,
                    false,
                )
                .unwrap();
            assert_close_relative(&download(&kernels, &actual), &expected);
        }
    }
    kernels.set_int8_activations(false);
}

/// A deterministic packed `Q6_K` matrix of `stored_rows` x `stored_cols`
/// super-blocks: small positive `d`, signed sub-block scales, and every
/// 6-bit quant exercised.
fn packed_q6k(stored_rows: u64, stored_cols: u64, transposed: bool, seed: u32) -> PackedWeight {
    let blocks = stored_rows * stored_cols / 256;
    let mut bytes = Vec::with_capacity(blocks as usize * 210);
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for _ in 0..blocks {
        bytes.extend((0..192).map(|_| (next() % 256) as u8));
        bytes.extend((0..16).map(|_| (next() % 256) as u8));
        // f16 between about 2^-12 and 2^-8.
        let exponent = 3 + (next() % 5) as u16;
        let bits = (exponent << 10) | (next() % 1024) as u16;
        bytes.extend(bits.to_le_bytes());
    }
    PackedWeight::new(
        PackedWeightFormat::Q6K,
        stored_rows,
        stored_cols,
        transposed,
        bytes,
    )
    .unwrap()
}

/// The packed `Q6_K` weight: downloading or expanding it gives back exactly
/// the Runtime's dequantization, and its decode matmul (rows read packed,
/// longer ones dequantized for the call), with and without a bias or
/// residual addend, matches the Reference CPU on the dequantized weight.
#[test]
fn q6k_packed_matmul_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (k, n) = (1024_u64, 97_u64); // odd: a warp with one column left
    let packed = packed_q6k(n, k, true, 31);
    let dense = packed.dequantize();
    let weight = kernels.upload_packed_weight(&packed).unwrap();
    assert!(weight.is_q8() && weight.is_compact());
    assert_eq!(weight.shape, vec![k, n]);
    assert_eq!(download(&kernels, &weight).data, dense.data);
    assert_eq!(
        download(&kernels, &kernels.expand(&weight).unwrap()).data,
        dense.data
    );
    let bias = HostTensor::new([n], (0..n).map(|i| i as f32 * 0.01).collect::<Vec<_>>()).unwrap();
    for m in [1_u64, 3, 8, 9, 65] {
        let a = activations(m, k, m as u32 + 50);
        let expected = magnetar_provider_cpu::matmul(&a, &dense, false, false).unwrap();
        let a_dev = upload(&kernels, &a);
        let actual = download(
            &kernels,
            &kernels.matmul(&a_dev, &weight, false, false).unwrap(),
        );
        assert_close_relative(&actual, &expected);
        let with_bias = kernels
            .matmul_bias(&a_dev, &weight, &upload(&kernels, &bias), false, false)
            .unwrap();
        let expected_bias = magnetar_provider_cpu::add(&expected, &bias).unwrap();
        assert_close_relative(&download(&kernels, &with_bias), &expected_bias);
        let residual = activations(m, n, 97);
        let with_residual = kernels
            .matmul_residual(&a_dev, &weight, &upload(&kernels, &residual), false, false)
            .unwrap();
        let expected_residual = magnetar_provider_cpu::add(&expected, &residual).unwrap();
        assert_close_relative(&download(&kernels, &with_residual), &expected_residual);
    }
}

/// Long prefills against packed weights: more rows than the packed row
/// Kernels take, so the weight is dequantized for the call and, on a Device
/// with Tensor Cores, multiplied with both operands split into three
/// bfloat16 terms (`matmul_tcp_split3_kernel`); the f32 Kernels otherwise.
/// With and without a bias or residual, against the Reference CPU on the
/// dequantized weights, within 1e-5 of the largest output: a weight rounded
/// to bfloat16 alone would be off by about 5e-4.
#[test]
fn long_prefills_against_packed_weights_match_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    // 136 columns and 300 rows leave partial Tensor Core tiles.
    let (k, n) = (512_u64, 136_u64);
    let weights = [
        ("Q8_0", packed_q8(n, k, true, 61)),
        ("Q4_K", packed_q4k(n, k, true, 62)),
        ("Q6_K", packed_q6k(n, k, true, 63)),
    ];
    let bias = HostTensor::new([n], (0..n).map(|i| i as f32 * 0.01).collect::<Vec<_>>()).unwrap();
    for (label, packed) in weights {
        let dense = packed.dequantize();
        let weight = kernels.upload_packed_weight(&packed).unwrap();
        for m in [64_u64, 200, 300] {
            let a = activations(m, k, m as u32 + 70);
            let expected = magnetar_provider_cpu::matmul(&a, &dense, false, false).unwrap();
            let a_dev = upload(&kernels, &a);
            let actual = kernels.matmul(&a_dev, &weight, false, false).unwrap();
            assert_eq!(actual.shape, vec![m, n], "{label} m={m}");
            assert_close_relative(&download(&kernels, &actual), &expected);
            let with_bias = kernels
                .matmul_bias(&a_dev, &weight, &upload(&kernels, &bias), false, false)
                .unwrap();
            let expected_bias = magnetar_provider_cpu::add(&expected, &bias).unwrap();
            assert_close_relative(&download(&kernels, &with_bias), &expected_bias);
            let residual = activations(m, n, 96);
            let with_residual = kernels
                .matmul_residual(&a_dev, &weight, &upload(&kernels, &residual), false, false)
                .unwrap();
            let expected_residual = magnetar_provider_cpu::add(&expected, &residual).unwrap();
            assert_close_relative(&download(&kernels, &with_residual), &expected_residual);
        }
    }
}

/// A deterministic packed `Q2_K` or `Q3_K` matrix of `stored_rows` x
/// `stored_cols` super-blocks: random quants, masks and scales, small
/// positive f16 `d` (and `dmin`).
fn packed_low_bit(
    format: PackedWeightFormat,
    stored_rows: u64,
    stored_cols: u64,
    seed: u32,
) -> PackedWeight {
    let blocks = stored_rows * stored_cols / 256;
    let block_bytes = format.block_bytes() as usize;
    let halves = if format == PackedWeightFormat::Q2K {
        2
    } else {
        1
    };
    let mut bytes = Vec::with_capacity(blocks as usize * block_bytes);
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for _ in 0..blocks {
        bytes.extend((0..block_bytes - 2 * halves).map(|_| (next() % 256) as u8));
        for _ in 0..halves {
            // f16 between about 2^-10 and 2^-6.
            let exponent = 5 + (next() % 5) as u16;
            let bits = (exponent << 10) | (next() % 1024) as u16;
            bytes.extend(bits.to_le_bytes());
        }
    }
    PackedWeight::new(format, stored_rows, stored_cols, true, bytes).unwrap()
}

/// A deterministic packed i-quant matrix (`IQ4_NL`, `IQ4_XS` or an
/// IQ2/IQ3 format): a small positive f16 `d` first in every block, random
/// scales, signs and indices after it.
fn packed_iq4(
    format: PackedWeightFormat,
    stored_rows: u64,
    stored_cols: u64,
    seed: u32,
) -> PackedWeight {
    let block_elements = format.block_elements();
    let blocks = stored_rows * stored_cols / block_elements;
    let block_bytes = format.block_bytes() as usize;
    let mut bytes = Vec::with_capacity(blocks as usize * block_bytes);
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for _ in 0..blocks {
        // f16 between about 2^-12 and 2^-8.
        let exponent = 3 + (next() % 5) as u16;
        let bits = (exponent << 10) | (next() % 1024) as u16;
        bytes.extend(bits.to_le_bytes());
        bytes.extend((0..block_bytes - 2).map(|_| (next() % 256) as u8));
    }
    PackedWeight::new(format, stored_rows, stored_cols, true, bytes).unwrap()
}

/// Packed i-quant weights (`IQ4_NL`, `IQ4_XS` and the IQ2/IQ3 formats),
/// and `Q4_0`, laid out like `IQ4_NL`:
/// downloading or expanding gives
/// back exactly the Runtime's dequantization; with int8 activations their
/// decode matmul (with a residual addend) and gated matmul match the
/// Reference CPU on the rounded activations; without, and for longer
/// matmuls, they match it on the f32 activations.
#[test]
fn iq_packed_matmuls_match_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (k, n) = (1024_u64, 97_u64);
    for format in [
        PackedWeightFormat::Iq4Nl,
        PackedWeightFormat::Q4_0,
        PackedWeightFormat::Iq4Xs,
        PackedWeightFormat::Iq2Xxs,
        PackedWeightFormat::Iq2Xs,
        PackedWeightFormat::Iq2S,
        PackedWeightFormat::Iq3Xxs,
        PackedWeightFormat::Iq3S,
    ] {
        let packed = packed_iq4(format, n, k, 61);
        let up = packed_iq4(format, n, k, 62);
        let dense = packed.dequantize();
        let weight = kernels.upload_packed_weight(&packed).unwrap();
        let up_weight = kernels.upload_packed_weight(&up).unwrap();
        assert_eq!(weight.shape, vec![k, n]);
        assert_eq!(download(&kernels, &weight).data, dense.data, "{format:?}");
        assert_eq!(
            download(&kernels, &kernels.expand(&weight).unwrap()).data,
            dense.data
        );
        for int8 in [true, false] {
            kernels.set_int8_activations(int8);
            for m in [1_u64, 3, 9, 65] {
                let a = activations(m, k, m as u32 + 90);
                let seen = if int8 && m <= 64 {
                    int8_rounded(&a)
                } else {
                    a.clone()
                };
                let a_dev = upload(&kernels, &a);
                let residual = activations(m, n, 91);
                let expected = magnetar_provider_cpu::add(
                    &magnetar_provider_cpu::matmul(&seen, &dense, false, false).unwrap(),
                    &residual,
                )
                .unwrap();
                let actual = kernels
                    .matmul_residual(&a_dev, &weight, &upload(&kernels, &residual), false, false)
                    .unwrap();
                assert_close_relative(&download(&kernels, &actual), &expected);
                let expected = magnetar_provider_cpu::mul(
                    &magnetar_provider_cpu::silu(
                        &magnetar_provider_cpu::matmul(&seen, &dense, false, false).unwrap(),
                    ),
                    &magnetar_provider_cpu::matmul(&seen, &up.dequantize(), false, false).unwrap(),
                )
                .unwrap();
                let actual = kernels
                    .gated_matmul(&a_dev, &weight, &up_weight, false, false)
                    .unwrap();
                assert_close_relative(&download(&kernels, &actual), &expected);
            }
        }
    }
    kernels.set_int8_activations(false);
}

/// Packed `Q2_K` and `Q3_K` weights: downloading or expanding gives back
/// exactly the Runtime's dequantization; with int8 activations their
/// decode matmul (with a residual addend) and gated matmul match the
/// Reference CPU on the rounded activations; without, and for longer
/// matmuls, they match it on the f32 activations.
#[test]
fn low_bit_packed_matmuls_match_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (k, n) = (1024_u64, 97_u64);
    for format in [PackedWeightFormat::Q2K, PackedWeightFormat::Q3K] {
        let packed = packed_low_bit(format, n, k, 51);
        let up = packed_low_bit(format, n, k, 52);
        let dense = packed.dequantize();
        let weight = kernels.upload_packed_weight(&packed).unwrap();
        let up_weight = kernels.upload_packed_weight(&up).unwrap();
        assert_eq!(weight.shape, vec![k, n]);
        assert_eq!(download(&kernels, &weight).data, dense.data, "{format:?}");
        assert_eq!(
            download(&kernels, &kernels.expand(&weight).unwrap()).data,
            dense.data
        );
        for int8 in [true, false] {
            kernels.set_int8_activations(int8);
            for m in [1_u64, 3, 9, 65] {
                let a = activations(m, k, m as u32 + 80);
                let seen = if int8 && m <= 64 {
                    int8_rounded(&a)
                } else {
                    a.clone()
                };
                let a_dev = upload(&kernels, &a);
                let residual = activations(m, n, 81);
                let expected = magnetar_provider_cpu::add(
                    &magnetar_provider_cpu::matmul(&seen, &dense, false, false).unwrap(),
                    &residual,
                )
                .unwrap();
                let actual = kernels
                    .matmul_residual(&a_dev, &weight, &upload(&kernels, &residual), false, false)
                    .unwrap();
                assert_close_relative(&download(&kernels, &actual), &expected);
                let expected = magnetar_provider_cpu::mul(
                    &magnetar_provider_cpu::silu(
                        &magnetar_provider_cpu::matmul(&seen, &dense, false, false).unwrap(),
                    ),
                    &magnetar_provider_cpu::matmul(&seen, &up.dequantize(), false, false).unwrap(),
                )
                .unwrap();
                let actual = kernels
                    .gated_matmul(&a_dev, &weight, &up_weight, false, false)
                    .unwrap();
                assert_close_relative(&download(&kernels, &actual), &expected);
            }
        }
    }
    kernels.set_int8_activations(false);
}

/// `a` as the int8-activation Kernels see it: each 32-value block rounded
/// to `d * round(x / d)`, `d = max |x| / 127`.
fn int8_rounded(a: &HostTensor) -> HostTensor {
    let mut data = a.data.clone();
    for block in data.chunks_mut(32) {
        let amax = block.iter().fold(0.0_f32, |max, x| max.max(x.abs()));
        let d = amax / 127.0;
        for x in block {
            *x = if amax == 0.0 {
                0.0
            } else {
                (*x / d).round_ties_even() * d
            };
        }
    }
    HostTensor::new(a.shape.clone(), data).unwrap()
}

/// With int8 activations on, the packed `Q8_0`, `Q4_K` and `Q6_K` decode
/// matmuls (with a residual addend) and gated matmuls compute the Reference
/// CPU's result on the rounded activations, and longer matmuls are
/// unaffected.
#[test]
fn int8_activation_matmuls_match_reference_cpu_on_rounded_activations() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    kernels.set_int8_activations(true);
    let (k, n) = (1024_u64, 97_u64);
    let weights = [
        packed_q8(n, k, true, 41),
        packed_q4k(n, k, true, 42),
        packed_q6k(n, k, true, 43),
    ];
    let ups = [packed_q8(n, k, true, 44), packed_q4k(n, k, true, 45)];
    for (packed, up) in weights.iter().zip(ups.iter().map(Some).chain([None])) {
        let dense = packed.dequantize();
        let weight = kernels.upload_packed_weight(packed).unwrap();
        for m in [1_u64, 3, 9, 65] {
            let a = activations(m, k, m as u32 + 60);
            let rounded = if m <= 64 { int8_rounded(&a) } else { a.clone() };
            let a_dev = upload(&kernels, &a);
            let residual = activations(m, n, 61);
            let expected = magnetar_provider_cpu::add(
                &magnetar_provider_cpu::matmul(&rounded, &dense, false, false).unwrap(),
                &residual,
            )
            .unwrap();
            let actual = kernels
                .matmul_residual(&a_dev, &weight, &upload(&kernels, &residual), false, false)
                .unwrap();
            assert_close_relative(&download(&kernels, &actual), &expected);
            if let Some(up) = up
                && m <= 64
            {
                let up_weight = kernels.upload_packed_weight(up).unwrap();
                let expected = magnetar_provider_cpu::mul(
                    &magnetar_provider_cpu::silu(
                        &magnetar_provider_cpu::matmul(&rounded, &dense, false, false).unwrap(),
                    ),
                    &magnetar_provider_cpu::matmul(&rounded, &up.dequantize(), false, false)
                        .unwrap(),
                )
                .unwrap();
                let actual = kernels
                    .gated_matmul(&a_dev, &weight, &up_weight, false, false)
                    .unwrap();
                assert_close_relative(&download(&kernels, &actual), &expected);
            }
        }
    }
}

/// `gated-matmul` against packed gate and up weights, and an embedding
/// lookup from a packed (untransposed) table, match the Reference CPU on
/// the dequantized weights.
#[test]
fn q8_packed_gated_matmul_and_embedding_match_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (k, n) = (128_u64, 64_u64);
    let gate = packed_q8(n, k, true, 11);
    let up = packed_q8(n, k, true, 12);
    for m in [1_u64, 70] {
        let a = activations(m, k, 5);
        let expected = magnetar_provider_cpu::mul(
            &magnetar_provider_cpu::silu(
                &magnetar_provider_cpu::matmul(&a, &gate.dequantize(), false, false).unwrap(),
            ),
            &magnetar_provider_cpu::matmul(&a, &up.dequantize(), false, false).unwrap(),
        )
        .unwrap();
        let actual = kernels
            .gated_matmul(
                &upload(&kernels, &a),
                &kernels.upload_packed_weight(&gate).unwrap(),
                &kernels.upload_packed_weight(&up).unwrap(),
                false,
                false,
            )
            .unwrap();
        assert_close_scaled(&download(&kernels, &actual), &expected, k as f32);
    }

    // A [vocab = 6, hidden = 64] table, rows read packed.
    let table = packed_q8(6, 64, false, 13);
    let ids = HostTensor::new([4], vec![5.0, 0.0, 3.0, 5.0]).unwrap();
    let expected = magnetar_provider_cpu::embedding_lookup(&table.dequantize(), &ids).unwrap();
    let actual = kernels
        .embedding_lookup(
            &kernels.upload_packed_weight(&table).unwrap(),
            &upload(&kernels, &ids),
        )
        .unwrap();
    assert_eq!(download(&kernels, &actual).data, expected.data);
}

/// An embedding lookup from a packed table stored untransposed, in every
/// packed format: the looked-up rows equal the same rows of the
/// dequantized table (other formats than `Q8_0` dequantize the looked-up
/// rows only), and an id that is not a row of the table fails the lookup.
#[test]
fn packed_embedding_tables_read_only_the_looked_up_rows() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (vocab, dim) = (7_u64, 512_u64);
    let untransposed = |weight: PackedWeight| {
        PackedWeight::new(
            weight.format(),
            weight.stored_rows(),
            weight.stored_cols(),
            false,
            weight.bytes().to_vec(),
        )
        .unwrap()
    };
    let mut tables = vec![
        packed_q8(vocab, dim, false, 71),
        packed_q4k(vocab, dim, false, 72),
        untransposed(packed_q5k(vocab, dim, 73)),
        packed_q6k(vocab, dim, false, 74),
    ];
    for format in [PackedWeightFormat::Q2K, PackedWeightFormat::Q3K] {
        tables.push(untransposed(packed_low_bit(format, vocab, dim, 75)));
    }
    for format in [
        PackedWeightFormat::Q4_0,
        PackedWeightFormat::Iq4Nl,
        PackedWeightFormat::Iq4Xs,
        PackedWeightFormat::Iq2Xxs,
        PackedWeightFormat::Iq2Xs,
        PackedWeightFormat::Iq2S,
        PackedWeightFormat::Iq3Xxs,
        PackedWeightFormat::Iq3S,
    ] {
        tables.push(untransposed(packed_iq4(format, vocab, dim, 76)));
    }
    let ids = HostTensor::new([5], vec![6.0, 0.0, 3.0, 6.0, 1.0]).unwrap();
    for table in &tables {
        let format = table.format();
        assert!(!table.transposed(), "{format:?}");
        let expected = magnetar_provider_cpu::embedding_lookup(&table.dequantize(), &ids).unwrap();
        let table_dev = kernels.upload_packed_weight(table).unwrap();
        let actual = kernels
            .embedding_lookup(&table_dev, &upload(&kernels, &ids))
            .unwrap();
        assert_eq!(actual.shape, vec![5, dim], "{format:?}");
        assert_eq!(
            download(&kernels, &actual).data,
            expected.data,
            "{format:?}"
        );
        for invalid in [vocab as f32, -1.0, 2.5] {
            let ids = HostTensor::new([2], vec![1.0, invalid]).unwrap();
            assert!(
                kernels
                    .embedding_lookup(&table_dev, &upload(&kernels, &ids))
                    .is_err(),
                "{format:?} id {invalid}"
            );
        }
        // The flag is reset for the next lookup.
        let actual = kernels
            .embedding_lookup(&table_dev, &upload(&kernels, &ids))
            .unwrap();
        assert_eq!(
            download(&kernels, &actual).data,
            expected.data,
            "{format:?}"
        );
    }
}

/// The real Qwen2.5-0.5B shapes (`execute-direct-quantized-weights`): a
/// packed `[151936, 896]` embedding table read row by row, and the tied
/// lm_head as the same bytes in transposed storage (`[896, 151936]`
/// logically) through the decode matmul, checked against the dequantized
/// values on sampled rows and columns.
#[test]
fn q8_packed_weights_at_real_checkpoint_sizes() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (vocab, hidden) = (151_936_u64, 896_u64);
    let blocks = (vocab * hidden / 32) as usize;
    let mut bytes = Vec::with_capacity(blocks * 34);
    for block in 0..blocks {
        // d = 2^-7 * (1 + (block % 7) / 8), exactly an f16.
        let d_bits: u16 = (8 << 10) | (((block % 7) as u16) << 7);
        bytes.extend(d_bits.to_le_bytes());
        bytes.extend((0..32).map(|i| (((block * 31 + i * 7) % 255) as i32 - 127) as i8 as u8));
    }
    let value = |row: u64, col: u64| {
        let index = (row * hidden + col) as usize;
        let block = index / 32;
        let d = (1.0 + (block % 7) as f32 / 8.0) / 128.0;
        let q = ((block * 31 + (index % 32) * 7) % 255) as i32 - 127;
        d * q as f32
    };
    let table = PackedWeight::new(
        PackedWeightFormat::Q8_0,
        vocab,
        hidden,
        false,
        bytes.clone(),
    )
    .unwrap();
    let lm_head = PackedWeight::new(PackedWeightFormat::Q8_0, vocab, hidden, true, bytes).unwrap();

    let table_dev = kernels.upload_packed_weight(&table).unwrap();
    let ids = [0_u64, 1, 77_777, vocab - 1, 4242];
    let ids_tensor = HostTensor::new(
        [ids.len() as u64],
        ids.iter().map(|&id| id as f32).collect::<Vec<_>>(),
    )
    .unwrap();
    let rows = download(
        &kernels,
        &kernels
            .embedding_lookup(&table_dev, &upload(&kernels, &ids_tensor))
            .unwrap(),
    );
    for (r, &id) in ids.iter().enumerate() {
        for col in 0..hidden {
            assert_eq!(
                rows.data[r * hidden as usize + col as usize],
                value(id, col)
            );
        }
    }

    let lm_head_dev = kernels.upload_packed_weight(&lm_head).unwrap();
    let a = activations(5, hidden, 3);
    let logits = download(
        &kernels,
        &kernels
            .matmul(&upload(&kernels, &a), &lm_head_dev, false, false)
            .unwrap(),
    );
    assert_eq!(logits.shape, vec![5, vocab]);
    for r in 0..5_u64 {
        for col in [0_u64, 1, 31, 32, 99_999, vocab - 1] {
            let expected: f32 = (0..hidden)
                .map(|k| a.data[(r * hidden + k) as usize] * value(col, k))
                .sum();
            let actual = logits.data[(r * vocab + col) as usize];
            assert!(
                (actual - expected).abs() <= 1e-3,
                "[{r}, {col}]: {actual} vs {expected}"
            );
        }
    }
}

/// A decode step over packed weights (`execute-direct-quantized-weights`)
/// captures into a CUDA graph -- the packed gated matmul allocates no
/// intermediate -- and its replay matches the uncaptured step. A capture
/// failed after a device buffer was dropped while capturing leaves the
/// stream usable: the driver error that free recorded does not outlive the
/// aborted capture.
#[test]
fn q8_packed_decode_step_captures_and_a_failed_capture_leaves_the_stream_usable() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (k, n) = (128_u64, 64_u64);
    let gate = kernels
        .upload_packed_weight(&packed_q8(n, k, true, 21))
        .unwrap();
    let up = kernels
        .upload_packed_weight(&packed_q8(n, k, true, 22))
        .unwrap();
    let table = kernels
        .upload_packed_weight(&packed_q8(6, k, false, 23))
        .unwrap();
    let ids = upload(&kernels, &HostTensor::new([1], vec![4.0]).unwrap());
    // Outputs are allocated before and offered, as the executor does:
    // nothing allocates or frees while capturing.
    let outputs = || {
        let alloc = |len| kernels.stream().alloc_zeros::<f32>(len).unwrap();
        (alloc(k as usize), alloc(n as usize))
    };
    let step = |(hidden_out, gated_out)| {
        let offered = kernels.offer_output(Some(hidden_out));
        let hidden = kernels.embedding_lookup(&table, &ids)?;
        drop(offered);
        let offered = kernels.offer_output(Some(gated_out));
        let out = kernels.gated_matmul(&hidden, &gate, &up, false, false)?;
        drop(offered);
        Ok::<_, crate::error::CudaError>((hidden, out))
    };
    let (_, eager) = step(outputs()).unwrap();
    let expected = download(&kernels, &eager);

    let buffers = outputs();
    kernels.begin_capture(Default::default()).unwrap();
    let (_hidden, captured) = step(buffers).unwrap();
    let mut graph = kernels.end_capture().unwrap();
    kernels.launch_captured(&mut graph, 0).unwrap();
    assert_eq!(download(&kernels, &captured).data, expected.data);

    let dropped = upload(&kernels, &activations(1, k, 7));
    kernels.begin_capture(Default::default()).unwrap();
    drop(dropped);
    assert!(
        kernels.upload(&activations(1, k, 8)).is_err(),
        "a host upload cannot be captured"
    );
    kernels.abort_capture();
    let (_, again) = step(outputs()).unwrap();
    assert_eq!(download(&kernels, &again).data, expected.data);
}

/// `execute-and-cache-moe-experts` F02: the CUDA `moe-experts` against the
/// Reference CPU for f32 and packed `Q8_0` experts, for a decode-sized
/// step (persistent scratch) and a prefill-sized one (temporaries); a
/// decode-sized step captures and replays exactly.
#[test]
fn moe_experts_match_reference_cpu_for_f32_and_packed_experts() {
    use magnetar_provider_api::{MoeRouting, moe_experts};
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (hidden, inter, experts, top_k) = (64_u64, 32_u64, 5_u64, 2_u64);
    let routing = MoeRouting {
        experts,
        top_k,
        normalize: false,
    };
    // Stored as each expert's rows: gate/up [E*I, H], down [E*H, I].
    let gate = packed_q8(experts * inter, hidden, true, 31);
    let up = packed_q8(experts * inter, hidden, true, 32);
    let down = packed_q8(experts * hidden, inter, true, 33);
    let (gate_f32, up_f32, down_f32) = (gate.dequantize(), up.dequantize(), down.dequantize());
    for tokens in [1_u64, 70] {
        let h = activations(tokens, hidden, 34);
        let logits = activations(tokens, experts, 35);
        for normalize in [false, true] {
            let routing = MoeRouting {
                normalize,
                ..routing
            };
            let expected =
                moe_experts(&h, &logits, [&gate_f32, &up_f32, &down_f32], routing).unwrap();
            let f32_experts = [&gate_f32, &up_f32, &down_f32].map(|w| upload(&kernels, w));
            let actual = kernels
                .moe_experts(
                    &upload(&kernels, &h),
                    &upload(&kernels, &logits),
                    [&f32_experts[0], &f32_experts[1], &f32_experts[2]],
                    routing,
                    (0, experts),
                )
                .unwrap();
            assert_close_relative(&download(&kernels, &actual), &expected);
            let packed_experts =
                [&gate, &up, &down].map(|w| kernels.upload_packed_weight(w).unwrap());
            let actual = kernels
                .moe_experts(
                    &upload(&kernels, &h),
                    &upload(&kernels, &logits),
                    [&packed_experts[0], &packed_experts[1], &packed_experts[2]],
                    routing,
                    (0, experts),
                )
                .unwrap();
            assert_close_relative(&download(&kernels, &actual), &expected);
        }
    }

    // A decode step captures (its output offered ahead) and replays.
    let packed_experts = [&gate, &up, &down].map(|w| kernels.upload_packed_weight(w).unwrap());
    let h = upload(&kernels, &activations(1, hidden, 36));
    let logits = upload(&kernels, &activations(1, experts, 37));
    let step = |out| {
        let offered = kernels.offer_output(Some(out));
        let result = kernels.moe_experts(
            &h,
            &logits,
            [&packed_experts[0], &packed_experts[1], &packed_experts[2]],
            routing,
            (0, experts),
        );
        drop(offered);
        result
    };
    let alloc = || {
        kernels
            .stream()
            .alloc_zeros::<f32>(hidden as usize)
            .unwrap()
    };
    let eager = download(&kernels, &step(alloc()).unwrap());
    let out = alloc();
    kernels.begin_capture(Default::default()).unwrap();
    let captured = step(out).unwrap();
    let mut graph = kernels.end_capture().unwrap();
    kernels.launch_captured(&mut graph, 0).unwrap();
    assert_eq!(download(&kernels, &captured).data, eager.data);
}

/// A long prefill's matmul against a packed weight runs on Tensor Cores
/// 2,048 rows of A at a time (`SPLIT_TCP_BLOCK_ROWS`): each output row is
/// the one a matmul over fewer rows gives, bit for bit, in the first
/// block, across a block boundary and in the last; with a full addend
/// (`matmul-residual`); and `gated-matmul` holds the gate's and up's
/// products a block at a time, its values `silu_mul` of both matmuls'.
#[test]
fn long_prefill_matmuls_run_in_row_blocks() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    if !kernels.has_tensor_cores() {
        eprintln!("skipped: no Tensor Cores");
        return;
    }
    let (m, k, n) = (4500_u64, 256_u64, 512_u64);
    let a = activations(m, k, 81);
    let residual = activations(m, n, 82);
    let weight = kernels
        .upload_packed_weight(&packed_q8(n, k, true, 83))
        .unwrap();
    let gate = kernels
        .upload_packed_weight(&packed_q8(n, k, true, 84))
        .unwrap();
    let up = kernels
        .upload_packed_weight(&packed_q8(n, k, true, 85))
        .unwrap();
    let rows_of = |tensor: &HostTensor, width: u64, first: u64, count: u64| {
        HostTensor::new(
            [count, width],
            tensor.data[(first * width) as usize..((first + count) * width) as usize].to_vec(),
        )
        .unwrap()
    };
    let (da, dresidual) = (upload(&kernels, &a), upload(&kernels, &residual));
    let full = download(
        &kernels,
        &kernels.matmul(&da, &weight, false, false).unwrap(),
    );
    let full_residual = download(
        &kernels,
        &kernels
            .matmul_residual(&da, &weight, &dresidual, false, false)
            .unwrap(),
    );
    let gated = download(
        &kernels,
        &kernels.gated_matmul(&da, &gate, &up, false, false).unwrap(),
    );
    let separate = download(
        &kernels,
        &kernels
            .silu_mul(
                &kernels.matmul(&da, &gate, false, false).unwrap(),
                &kernels.matmul(&da, &up, false, false).unwrap(),
            )
            .unwrap(),
    );
    assert_eq!(gated.shape, vec![m, n]);
    for (index, (g, s)) in gated.data.iter().zip(&separate.data).enumerate() {
        assert_eq!(
            g.to_bits(),
            s.to_bits(),
            "gated element {index}: {g} against {s}"
        );
    }
    for (first, count) in [(0_u64, 100_u64), (2000, 100), (4400, 100)] {
        let sub = upload(&kernels, &rows_of(&a, k, first, count));
        let sub_residual = upload(&kernels, &rows_of(&residual, n, first, count));
        let checks = [
            (
                "matmul",
                &full,
                download(
                    &kernels,
                    &kernels.matmul(&sub, &weight, false, false).unwrap(),
                ),
            ),
            (
                "matmul-residual",
                &full_residual,
                download(
                    &kernels,
                    &kernels
                        .matmul_residual(&sub, &weight, &sub_residual, false, false)
                        .unwrap(),
                ),
            ),
            (
                "gated-matmul",
                &gated,
                download(
                    &kernels,
                    &kernels
                        .gated_matmul(&sub, &gate, &up, false, false)
                        .unwrap(),
                ),
            ),
        ];
        for (what, whole, part) in checks {
            let expected = &whole.data[(first * n) as usize..((first + count) * n) as usize];
            for (index, (p, w)) in part.data.iter().zip(expected).enumerate() {
                assert_eq!(
                    p.to_bits(),
                    w.to_bits(),
                    "{what}, rows {first}..{}, element {index}: {p} against {w}",
                    first + count
                );
            }
        }
    }
    // Against the Reference CPU's product of the expanded weight.
    let expected = magnetar_runtime::reference_cpu::matmul(
        &a,
        &packed_q8(n, k, true, 83).dequantize(),
        false,
        false,
    )
    .unwrap();
    assert_close_relative(&full, &expected);
}

/// A `moe-experts` prefill over packed `Q8_0` experts, its pairs grouped by
/// expert (`moe_q8_gated_grouped_kernel`), gives the per-pair Kernels' own
/// values: for prompts served in one and in several token chunks, a share
/// of the experts (pairs held elsewhere skipped), SiLU and GELU gates with
/// a per-expert scale; and matches the Reference CPU.
#[test]
fn grouped_moe_prefill_matches_the_per_pair_kernels() {
    use crate::kernels::CudaMoeOptions;
    use magnetar_provider_api::{MoeRouting, moe_experts};
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (hidden, inter, experts, top_k) = (64_u64, 32_u64, 16_u64, 4_u64);
    for (tokens, held) in [(65_u64, (0, experts)), (300, (5, 6)), (2500, (0, experts))] {
        let count = held.1;
        let gate = packed_q8(count * inter, hidden, true, 71);
        let up = packed_q8(count * inter, hidden, true, 72);
        let down = packed_q8(count * hidden, inter, true, 73);
        let packed = [&gate, &up, &down].map(|w| kernels.upload_packed_weight(w).unwrap());
        let h = activations(tokens, hidden, 74);
        let logits = activations(tokens, experts, 75);
        let (dh, dlogits) = (upload(&kernels, &h), upload(&kernels, &logits));
        let scale = upload(
            &kernels,
            &HostTensor::new(
                [experts],
                (0..experts)
                    .map(|e| 0.5 + e as f32 / 16.0)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        for (normalize, gelu_tanh) in [(true, false), (false, true)] {
            let routing = MoeRouting {
                experts,
                top_k,
                normalize,
            };
            let options = CudaMoeOptions {
                gelu_tanh,
                per_expert_scale: gelu_tanh.then_some(&scale),
            };
            let run = |grouped: bool| {
                kernels.set_moe_grouped(grouped);
                let out = kernels
                    .moe_experts_with(
                        &dh,
                        &dlogits,
                        [&packed[0], &packed[1], &packed[2]],
                        routing,
                        held,
                        options,
                    )
                    .unwrap();
                download(&kernels, &out)
            };
            let per_pair = run(false);
            let grouped = run(true);
            assert_eq!(grouped.shape, per_pair.shape);
            for (index, (g, p)) in grouped.data.iter().zip(&per_pair.data).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    p.to_bits(),
                    "{tokens} tokens, experts {held:?}, normalize {normalize}, gelu \
                     {gelu_tanh}, element {index}: grouped {g} per-pair {p}"
                );
            }
            if held == (0, experts) && !gelu_tanh {
                let expected = moe_experts(
                    &h,
                    &logits,
                    [&gate.dequantize(), &up.dequantize(), &down.dequantize()],
                    routing,
                )
                .unwrap();
                assert_close_relative(&grouped, &expected);
            }
        }
    }
    kernels.set_moe_grouped(true);
}

/// The Device time of one `moe-experts` prefill at DeepSeek-V2-Lite's
/// routed experts (2048 hidden, 1408 intermediate, 64 experts, top 6),
/// packed `Q8_0`, over 1,024 and 4,096 tokens: the pairs grouped by expert
/// against one pair at a time; diagnostic only.
#[test]
#[ignore = "GPU timing diagnostic"]
fn moe_prefill_kernel_times() {
    use magnetar_provider_api::MoeRouting;
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let context = kernels.stream().context().clone();
    let timing = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let time = |run: &dyn Fn(), repeats: u32| -> f32 {
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
    let (hidden, inter, experts, top_k) = (2048_u64, 1408_u64, 64_u64, 6_u64);
    let routing = MoeRouting {
        experts,
        top_k,
        normalize: false,
    };
    let gate = kernels
        .upload_packed_weight(&packed_q8(experts * inter, hidden, true, 76))
        .unwrap();
    let up = kernels
        .upload_packed_weight(&packed_q8(experts * inter, hidden, true, 77))
        .unwrap();
    let down = kernels
        .upload_packed_weight(&packed_q8(experts * hidden, inter, true, 78))
        .unwrap();
    for tokens in [1024_u64, 4096] {
        let h = upload(&kernels, &activations(tokens, hidden, 79));
        let logits = upload(&kernels, &activations(tokens, experts, 80));
        for grouped in [false, true] {
            kernels.set_moe_grouped(grouped);
            let ms = time(
                &|| {
                    kernels
                        .moe_experts(&h, &logits, [&gate, &up, &down], routing, (0, experts))
                        .unwrap();
                },
                3,
            );
            eprintln!(
                "moe-experts prefill, {tokens} tokens, {}: {:.2} ms",
                if grouped {
                    "pairs grouped by expert"
                } else {
                    "one pair at a time"
                },
                ms
            );
        }
    }
    kernels.set_moe_grouped(true);
}

/// Tensor parallelism splits a `moe-experts` layer's experts whole across
/// GPUs: each share, f32 or packed `Q8_0`, matches the Reference CPU's
/// share (routing over every expert, summing only those held), and the
/// shares add up to the whole layer.
#[test]
fn moe_expert_shares_match_reference_cpu_and_add_up() {
    use magnetar_provider_api::{MoeRouting, moe_experts, moe_experts_local};
    use magnetar_provider_api::{packed_weight::PackedWeight, packed_weight::PackedWeightFormat};
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (hidden, inter, experts, top_k, tokens) = (64_u64, 32_u64, 5_u64, 3_u64, 4_u64);
    let routing = MoeRouting {
        experts,
        top_k,
        normalize: true,
    };
    let gate = packed_q8(experts * inter, hidden, true, 51);
    let up = packed_q8(experts * inter, hidden, true, 52);
    let down = packed_q8(experts * hidden, inter, true, 53);
    // Stored rows of experts [first, first + count).
    let rows = |weight: &PackedWeight, per_expert: u64, first: u64, count: u64| {
        let row_bytes = (weight.stored_cols() / 32 * 34) as usize;
        let start = (first * per_expert) as usize * row_bytes;
        let end = ((first + count) * per_expert) as usize * row_bytes;
        PackedWeight::new(
            PackedWeightFormat::Q8_0,
            count * per_expert,
            weight.stored_cols(),
            true,
            weight.bytes()[start..end].to_vec(),
        )
        .unwrap()
    };
    let h = activations(tokens, hidden, 54);
    let logits = activations(tokens, experts, 55);
    let whole = moe_experts(
        &h,
        &logits,
        [&gate.dequantize(), &up.dequantize(), &down.dequantize()],
        routing,
    )
    .unwrap();
    let mut summed = vec![0.0_f32; whole.data.len()];
    for (first, count) in [(0_u64, 2_u64), (2, 3)] {
        let share = [
            rows(&gate, inter, first, count),
            rows(&up, inter, first, count),
            rows(&down, hidden, first, count),
        ];
        let dense = share.each_ref().map(PackedWeight::dequantize);
        let expected = moe_experts_local(
            &h,
            &logits,
            [&dense[0], &dense[1], &dense[2]],
            routing,
            first,
            count,
        )
        .unwrap();
        let f32_share = dense.each_ref().map(|w| upload(&kernels, w));
        let actual = kernels
            .moe_experts(
                &upload(&kernels, &h),
                &upload(&kernels, &logits),
                [&f32_share[0], &f32_share[1], &f32_share[2]],
                routing,
                (first, count),
            )
            .unwrap();
        assert_close_relative(&download(&kernels, &actual), &expected);
        let packed_share = share
            .each_ref()
            .map(|w| kernels.upload_packed_weight(w).unwrap());
        let actual = download(
            &kernels,
            &kernels
                .moe_experts(
                    &upload(&kernels, &h),
                    &upload(&kernels, &logits),
                    [&packed_share[0], &packed_share[1], &packed_share[2]],
                    routing,
                    (first, count),
                )
                .unwrap(),
        );
        assert_close_relative(&actual, &expected);
        for (slot, value) in summed.iter_mut().zip(&actual.data) {
            *slot += value;
        }
    }
    assert_close_relative(
        &HostTensor::new(whole.shape.clone(), summed).unwrap(),
        &whole,
    );
}

/// `row-scale` and the `sigmoid` activation against the Reference CPU.
#[test]
fn row_scale_and_sigmoid_match_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let x = activations(3, 40, 41);
    let scale = activations(3, 1, 42);
    let expected = magnetar_provider_api::row_scale(&x, &scale).unwrap();
    let actual = kernels
        .row_scale(&upload(&kernels, &x), &upload(&kernels, &scale))
        .unwrap();
    assert_close_scaled(&download(&kernels, &actual), &expected, 1.0);
    let expected = magnetar_provider_api::sigmoid(&x);
    let actual = kernels.sigmoid(&upload(&kernels, &x)).unwrap();
    assert_close_scaled(&download(&kernels, &actual), &expected, 1.0);
}

/// `actual` within f32 summation error of `expected`, measured against the
/// largest expected magnitude: an output is a difference of terms that
/// large, so cancellation leaves an error of their size, not its own (a
/// wrong expert or weight changes outputs by that whole magnitude).
fn assert_close_relative(actual: &HostTensor, expected: &HostTensor) {
    assert_eq!(actual.shape, expected.shape);
    let scale = expected
        .data
        .iter()
        .fold(1.0_f32, |max, e| max.max(e.abs()));
    for (index, (a, e)) in actual.data.iter().zip(&expected.data).enumerate() {
        assert!(
            (a - e).abs() <= 1e-5 * scale,
            "index {index}: {a} vs {e} (scale {scale})"
        );
    }
}

/// A tensor-parallel step captured as one graph across two GPUs: the
/// second GPU joins the first's capture, runs a Kernel, and the first sums
/// the second's buffer into its own by reading it directly; the second
/// then copies the sum back. The graph, launched twice, gives the same
/// result as the same work run eagerly. Skipped without two GPUs with
/// peer access.
#[test]
fn a_step_captured_across_two_gpus_replays_its_reduction() {
    let first = CudaProvider::for_device(0, "first");
    let second = CudaProvider::for_device(1, "second");
    let (Some(context0), Some(context1)) = (first.context(), second.context()) else {
        eprintln!("skipping: fewer than two CUDA devices");
        return;
    };
    let reachable = crate::peer::device_can_access_peer(&context0, &context1).unwrap_or(false)
        && crate::peer::device_can_access_peer(&context1, &context0).unwrap_or(false);
    if !reachable {
        eprintln!("skipping: the two GPUs have no peer access");
        return;
    }
    crate::peer::enable_peer_access(&context0, &context1).unwrap();
    crate::peer::enable_peer_access(&context1, &context0).unwrap();
    let host = CudaKernels::compile_and_load(&context0).unwrap();
    let joined = CudaKernels::compile_and_load(&context1).unwrap();
    for kernels in [&host, &joined] {
        kernels.set_peer_access(true);
        kernels.set_relaxed_capture(true);
    }
    let values = HostTensor::new([1, 64], (0..64).map(|v| v as f32).collect::<Vec<_>>()).unwrap();
    let mut a = upload(&host, &values);
    let mut b = upload(&joined, &values);
    let c = upload(&joined, &values);
    // b += c on the second GPU, a += b read from it by the first, then b = a
    // copied back. Eagerly once, which also loads every Kernel: b = 2v,
    // a = 3v, b = 3v.
    let step = |a: &mut CudaDeviceBuffer, b: &mut CudaDeviceBuffer| {
        joined.accumulate(b, &c).unwrap();
        host.accumulate_from_peer(a, &joined, b).unwrap();
        joined
            .copy_from_peer(&host, &a.slice, &mut b.slice)
            .unwrap();
        host.follow(&joined).unwrap();
    };
    step(&mut a, &mut b);
    assert_eq!(
        download(&host, &a).data,
        (0..64).map(|v| 3.0 * v as f32).collect::<Vec<_>>()
    );
    for kernels in [&host, &joined] {
        kernels.prepare_capture().unwrap();
    }
    host.begin_capture(Default::default()).unwrap();
    joined.join_capture(&host, Default::default()).unwrap();
    assert!(
        format!("{:?}", joined.stream().capture_status().unwrap()).contains("ACTIVE"),
        "the second GPU's stream joined the capture"
    );
    step(&mut a, &mut b);
    let mut joined_part = joined.leave_capture(&host).unwrap();
    let mut graph = host.end_capture().unwrap();
    joined
        .write_joined_offset(&mut joined_part, 0, &host)
        .unwrap();
    host.launch_captured(&mut graph, 0).unwrap();
    joined.follow(&host).unwrap();
    // end_capture only instantiates: two launches after the eager run take
    // a from 3v to 7v, then 15v.
    joined
        .write_joined_offset(&mut joined_part, 0, &host)
        .unwrap();
    host.launch_captured(&mut graph, 0).unwrap();
    joined.follow(&host).unwrap();
    assert_eq!(
        download(&host, &a).data,
        (0..64).map(|v| 15.0 * v as f32).collect::<Vec<_>>()
    );
    assert_eq!(
        download(&joined, &b).data,
        (0..64).map(|v| 15.0 * v as f32).collect::<Vec<_>>()
    );
}

/// `replay-bucketed-prefill`: a packed row matmul captured with the
/// `padding_rows` attribute skips a replay's trailing padding rows. Its
/// real rows match an uncaptured matmul exactly, its skipped rows keep
/// what they held, and the capture allows at most `m - 1` skipped rows.
/// With f32 and with int8 activations.
#[test]
fn a_captured_packed_matmul_skips_a_replays_padding_rows() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (m, k, n) = (8_u64, 512_u64, 256_u64);
    let weight = kernels
        .upload_packed_weight(&packed_q4k(n, k, true, 81))
        .unwrap();
    for int8 in [false, true] {
        kernels.set_int8_activations(int8);
        let first = activations(m, k, 82);
        let second = activations(m, k, 83);
        let expected = |values: &HostTensor| {
            let a = upload(&kernels, values);
            download(
                &kernels,
                &kernels.matmul(&a, &weight, false, false).unwrap(),
            )
        };
        let (expected_first, expected_second) = (expected(&first), expected(&second));
        let mut a = upload(&kernels, &first);
        let out = kernels
            .stream()
            .alloc_zeros::<f32>((m * n) as usize)
            .unwrap();
        let offered = kernels.offer_output(Some(out));
        kernels
            .begin_capture(std::collections::BTreeSet::from([
                crate::kernels::PADDING_ROWS_ATTRIBUTE.to_string(),
            ]))
            .unwrap();
        let captured = kernels.matmul(&a, &weight, false, false).unwrap();
        drop(offered);
        let mut graph = kernels.end_capture().unwrap();
        assert_eq!(graph.max_offset, m - 1, "int8 {int8}");
        kernels.launch_captured(&mut graph, 0).unwrap();
        assert_eq!(
            download(&kernels, &captured).data,
            expected_first.data,
            "int8 {int8}"
        );

        // Three padding rows: the five real rows are the second input's,
        // the last three still the first's.
        let skipped = 3;
        kernels
            .stream()
            .memcpy_htod(&second.data, &mut a.slice)
            .unwrap();
        kernels.launch_captured(&mut graph, skipped).unwrap();
        let replayed = download(&kernels, &captured).data;
        let real = ((m - skipped) * n) as usize;
        assert_eq!(
            replayed[..real],
            expected_second.data[..real],
            "int8 {int8}"
        );
        assert_eq!(replayed[real..], expected_first.data[real..], "int8 {int8}");

        // A one-row launch in the same capture skips nothing and leaves
        // the bound to the many-row ones.
        let row = upload(&kernels, &activations(1, k, 84));
        let expected_row = download(
            &kernels,
            &kernels.matmul(&row, &weight, false, false).unwrap(),
        );
        let rows_out = kernels
            .stream()
            .alloc_zeros::<f32>((m * n) as usize)
            .unwrap();
        let row_out = kernels.stream().alloc_zeros::<f32>(n as usize).unwrap();
        kernels
            .begin_capture(std::collections::BTreeSet::from([
                crate::kernels::PADDING_ROWS_ATTRIBUTE.to_string(),
            ]))
            .unwrap();
        let offered = kernels.offer_output(Some(rows_out));
        let captured_rows = kernels.matmul(&a, &weight, false, false).unwrap();
        drop(offered);
        let offered = kernels.offer_output(Some(row_out));
        let captured_row = kernels.matmul(&row, &weight, false, false).unwrap();
        drop(offered);
        let mut graph = kernels.end_capture().unwrap();
        assert_eq!(graph.max_offset, m - 1, "int8 {int8}");
        kernels.launch_captured(&mut graph, skipped).unwrap();
        assert_eq!(
            download(&kernels, &captured_row).data,
            expected_row.data,
            "int8 {int8}"
        );
        let replayed = download(&kernels, &captured_rows).data;
        assert_eq!(
            replayed[..real],
            expected_second.data[..real],
            "int8 {int8}"
        );
    }
}

// ---------------------------------------------------------------------------
// Layered models' optional operator forms (`dispatch-layered-model-forms`),
// against the Reference CPU and the shared `operator_semantics`.
// ---------------------------------------------------------------------------

/// GELU (tanh), `scale`, `softcap` and the `[1]`-broadcast `mul`.
#[test]
fn layered_elementwise_forms_match_reference_cpu() {
    use magnetar_provider_api::operator_semantics;
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let x = activations(3, 70, 81);
    let mut scaled = x.clone();
    for value in &mut scaled.data {
        *value *= 4.0;
    }
    let device = upload(&kernels, &scaled);
    let gelu = HostTensor::new(
        scaled.shape.clone(),
        scaled
            .data
            .iter()
            .map(|value| operator_semantics::gelu_tanh(*value))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_close(
        &download(&kernels, &kernels.gelu_tanh(&device).unwrap()),
        &gelu,
    );
    assert_close(
        &download(&kernels, &kernels.scale(&device, 0.25).unwrap()),
        &operator_semantics::scale(&scaled, 0.25).unwrap(),
    );
    assert_close(
        &download(&kernels, &kernels.softcap(&device, 1.5).unwrap()),
        &operator_semantics::softcap(&scaled, 1.5).unwrap(),
    );
    let scalar = HostTensor::new([1], vec![-0.75]).unwrap();
    assert_close(
        &download(
            &kernels,
            &kernels
                .mul_scalar(&device, &upload(&kernels, &scalar))
                .unwrap(),
        ),
        &operator_semantics::mul_broadcast(&scaled, &scalar).unwrap(),
    );
    assert!(kernels.softcap(&device, 0.0).is_err());
    let mut in_place = upload(&kernels, &scaled);
    kernels.scale_in_place(&mut in_place, 0.25).unwrap();
    assert_close(
        &download(&kernels, &in_place),
        &operator_semantics::scale(&scaled, 0.25).unwrap(),
    );
}

/// Per-head RMSNorm with a `[head_dimension]` weight, and without a weight
/// (per row and per head).
#[test]
fn per_head_and_weightless_rmsnorm_match_reference_cpu() {
    use magnetar_provider_api::operator_semantics::rmsnorm_blocks;
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let input = activations(5, 4 * 16, 82);
    let weight = activations(1, 16, 83);
    let weight = HostTensor::new([16], weight.data).unwrap();
    let device = upload(&kernels, &input);
    let device_weight = upload(&kernels, &weight);
    for (weight_host, weight_device, heads) in [
        (Some(&weight), Some(&device_weight), Some((4, 16))),
        (None, None, Some((4, 16))),
        (None, None, None),
    ] {
        let expected = rmsnorm_blocks(&input, weight_host, 1e-6, heads).unwrap();
        let actual = kernels
            .rmsnorm_with(&device, weight_device, 1e-6, heads)
            .unwrap();
        assert_close(&download(&kernels, &actual), &expected);
    }
    // Heads that do not tile the row are refused.
    assert!(
        kernels
            .rmsnorm_with(&device, None, 1e-6, Some((3, 16)))
            .is_err()
    );
}

/// Partial rotation: only the first `floor(factor * dimension / 2)` pairs
/// rotate, the frequencies still dividing by the whole dimension.
#[test]
fn partial_rope_matches_reference_cpu() {
    use crate::kernels::CudaRopePairing;
    use magnetar_runtime::RopePairing;
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let input = activations(6, 2 * 32, 84);
    let device = upload(&kernels, &input);
    for (pairing, cuda_pairing) in [
        (RopePairing::HalfSplit, CudaRopePairing::HalfSplit),
        (RopePairing::Interleaved, CudaRopePairing::Interleaved),
    ] {
        for factor in [0.25_f32, 0.5, 1.0] {
            let expected = magnetar_runtime::reference_cpu::rope_partial(
                &input,
                4.0,
                1.0,
                32,
                3,
                2,
                pairing,
                None,
                Some(factor),
            )
            .unwrap();
            let rotated = (factor as f64 * 32.0 / 2.0).floor() as u64;
            let actual = kernels
                .rope_partial(
                    &device,
                    4.0,
                    1.0,
                    32,
                    3,
                    2,
                    cuda_pairing,
                    None,
                    Some(rotated),
                )
                .unwrap();
            assert_close(&download(&kernels, &actual), &expected);
        }
    }
}

/// `attention.scale` replaces 1/sqrt(head_dimension) in every attention
/// Kernel: decode (split and grouped), the tiled and Tensor Core prefills,
/// and the block Kernel for heads wider than those handle; `None` keeps
/// today's scale exactly.
#[test]
fn scaled_attention_matches_reference_cpu_in_every_kernel() {
    use magnetar_runtime::reference_cpu::attention_scaled;
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    // (query rows, key rows, heads, kv heads, head dimension, window).
    let shapes = [
        (1_u64, 40_u64, 4_u64, 1_u64, 64_u64, None),
        (1, 40, 4, 1, 256, Some(16_u64)),
        (24, 24, 2, 1, 64, None),
        (24, 24, 2, 2, 128, Some(8)),
        (6, 9, 2, 1, 256, None),
        (5, 5, 2, 1, 512, Some(3)),
    ];
    for (seq, kv_seq, heads, kv_heads, dim, window) in shapes {
        let q = activations(seq, heads * dim, 85);
        let k = activations(kv_seq, kv_heads * dim, 86);
        let v = activations(kv_seq, kv_heads * dim, 87);
        let (dq, dk, dv) = (
            upload(&kernels, &q),
            upload(&kernels, &k),
            upload(&kernels, &v),
        );
        for scale in [None, Some(1.0_f32), Some(0.03)] {
            let expected =
                attention_scaled(&q, &k, &v, heads, dim, Some(kv_heads), window, true, scale)
                    .unwrap();
            let actual = kernels
                .attention_scaled(
                    &dq,
                    &dk,
                    &dv,
                    heads,
                    dim,
                    Some(kv_heads),
                    window,
                    true,
                    None,
                    scale,
                )
                .unwrap();
            let actual = download(&kernels, &actual);
            assert_eq!(actual.shape, expected.shape);
            for (index, (a, e)) in actual.data.iter().zip(&expected.data).enumerate() {
                assert!(
                    (a - e).abs() <= 2e-3,
                    "shape {:?}, scale {scale:?}, element {index}: cuda={a} reference={e}",
                    (seq, kv_seq, heads, kv_heads, dim, window)
                );
            }
        }
    }
    let q = upload(&kernels, &activations(1, 64, 88));
    assert!(
        kernels
            .attention_scaled(&q, &q, &q, 1, 64, None, None, true, None, Some(f32::NAN))
            .is_err()
    );
}

/// Routed experts with a GELU (tanh) gate and a per-expert output scale,
/// f32 and packed `Q8_0`, prefill and decode.
#[test]
fn moe_experts_with_gelu_and_per_expert_scale_match_reference_cpu() {
    use crate::kernels::CudaMoeOptions;
    use magnetar_provider_api::MoeRouting;
    use magnetar_provider_api::operator_semantics::{
        MoeExpertActivation, MoeExpertOptions, moe_experts_local_with,
    };
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let (experts, top_k, hidden, inter) = (6_u64, 2_u64, 64_u64, 32_u64);
    let routing = MoeRouting {
        experts,
        top_k,
        normalize: true,
    };
    let gate = packed_q8(experts * inter, hidden, true, 91);
    let up = packed_q8(experts * inter, hidden, true, 92);
    let down = packed_q8(experts * hidden, inter, true, 93);
    let (gate_f32, up_f32, down_f32) = (gate.dequantize(), up.dequantize(), down.dequantize());
    let scale = HostTensor::new(
        [experts],
        (0..experts)
            .map(|e| 0.5 + e as f32 * 0.25)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let device_scale = upload(&kernels, &scale);
    for tokens in [1_u64, 9] {
        let h = activations(tokens, hidden, 94);
        let logits = activations(tokens, experts, 95);
        let expected = moe_experts_local_with(
            &h,
            &logits,
            [&gate_f32, &up_f32, &down_f32],
            routing,
            0,
            experts,
            MoeExpertOptions {
                activation: MoeExpertActivation::GeluTanh,
                per_expert_scale: Some(&scale),
            },
        )
        .unwrap();
        let options = CudaMoeOptions {
            gelu_tanh: true,
            per_expert_scale: Some(&device_scale),
        };
        let f32_experts = [&gate_f32, &up_f32, &down_f32].map(|w| upload(&kernels, w));
        let packed_experts = [&gate, &up, &down].map(|w| kernels.upload_packed_weight(w).unwrap());
        for weights in [&f32_experts, &packed_experts] {
            let actual = kernels
                .moe_experts_with(
                    &upload(&kernels, &h),
                    &upload(&kernels, &logits),
                    [&weights[0], &weights[1], &weights[2]],
                    routing,
                    (0, experts),
                    options,
                )
                .unwrap();
            assert_close_relative(&download(&kernels, &actual), &expected);
        }
    }
}

/// YaRN (`express-latent-attention-forms`): DeepSeek-V2-Lite's rotated
/// width and scaling, with an `attention_factor` on `cos` and `sin`, whole
/// and partial rotation, both pairings.
#[test]
fn yarn_rope_matches_reference_cpu() {
    use crate::kernels::CudaRopePairing;
    use magnetar_core::operator::RopeYarnFrequencyScaling;
    use magnetar_runtime::RopePairing;
    use magnetar_runtime::reference_cpu::{RopeFrequencies, rope_scaled};
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let scaling = RopeYarnFrequencyScaling {
        factor: 40.0,
        original_context_length: 4096,
        beta_fast: 32.0,
        beta_slow: 1.0,
        attention_factor: 1.25,
        truncate: true,
    };
    let input = activations(5, 2 * 64, 91);
    let device = upload(&kernels, &input);
    for (pairing, cuda_pairing) in [
        (RopePairing::HalfSplit, CudaRopePairing::HalfSplit),
        (RopePairing::Interleaved, CudaRopePairing::Interleaved),
    ] {
        for position_offset in [0_u64, 300] {
            for factor in [None, Some(0.5_f32)] {
                let expected = rope_scaled(
                    &input,
                    10_000.0,
                    1.0,
                    64,
                    position_offset,
                    2,
                    pairing,
                    RopeFrequencies::Yarn(&scaling),
                    factor,
                )
                .unwrap();
                let rotated = factor.map(|factor| (factor as f64 * 64.0 / 2.0).floor() as u64);
                let actual = kernels
                    .rope_scaled(
                        &device,
                        10_000.0,
                        1.0,
                        64,
                        position_offset,
                        2,
                        cuda_pairing,
                        None,
                        Some(&scaling),
                        rotated,
                    )
                    .unwrap();
                assert_close(&download(&kernels, &actual), &expected);
            }
        }
    }
    // Banded and YaRN together are refused.
    let banded = magnetar_core::operator::RopeBandedFrequencyScaling {
        factor: 8.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_context_length: 8192,
    };
    assert!(
        kernels
            .rope_scaled(
                &device,
                10_000.0,
                1.0,
                64,
                0,
                2,
                CudaRopePairing::HalfSplit,
                Some(&banded),
                Some(&scaling),
                None,
            )
            .is_err()
    );
}

/// Attention over latent K and V (`latent_kv_width`,
/// `chunk-latent-prefills`): the key rows a pass reads (`key_count`, the
/// queries the last of them) expanded per head by packed projections, from
/// an f32 history and from `tq4` histories with and without an exact head,
/// against the Reference CPU's expansion and attention.
#[test]
fn attention_over_latent_keys_and_values_matches_the_cpu() {
    use magnetar_provider_api::operator_semantics::{latent_keys, latent_values};
    use magnetar_provider_api::{KvStorePrecision, TurboQuantStoreState};
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    // Four heads of 16 + 8 key values and 16 values, 32 latent values.
    let (heads, nope, rope, latent, value) = (4u64, 16u64, 8u64, 32u64, 16u64);
    let (width, head_dimension) = (latent + rope, nope + rope);
    let w_k = packed_q8(heads * nope, latent, true, 86);
    let w_v = packed_q8(heads * value, latent, true, 87);
    let (dw_k, dw_v) = (
        kernels.upload_packed_weight(&w_k).unwrap(),
        kernels.upload_packed_weight(&w_v).unwrap(),
    );
    let (w_k, w_v) = (w_k.dequantize(), w_v.dequantize());
    for (queries, keys, exact, precision) in [
        (70_u64, 260_u64, 0_u64, None),
        (40, 40, 0, None),
        (70, 260, 0, Some(KvStorePrecision::Tq4)),
        (70, 260, 100, Some(KvStorePrecision::Tq4)),
    ] {
        let label = format!("{queries} queries over {keys} keys, {precision:?}, head {exact}");
        let capacity = keys + 9;
        let data = hashed_values((keys * width) as usize, 31 + keys as usize);
        let (store, rows) = match precision {
            None => {
                let mut padded = data.clone();
                padded.resize((capacity * width) as usize, 0.5);
                (
                    upload(
                        &kernels,
                        &HostTensor::new([capacity, width], padded).unwrap(),
                    ),
                    HostTensor::new([keys, width], data.clone()).unwrap(),
                )
            }
            Some(precision) => {
                let mut store = kernels
                    .tq_zeros_with_exact_head(&[capacity, width], precision, 8, exact)
                    .unwrap();
                kernels
                    .kv_write(
                        &mut store,
                        &upload(
                            &kernels,
                            &HostTensor::new([keys, width], data.clone()).unwrap(),
                        ),
                        0,
                    )
                    .unwrap();
                let mut cpu = TurboQuantStoreState::new(precision, 8, width).with_exact_rows(exact);
                let values = cpu.write(&data, 0).unwrap();
                (store, HostTensor::new([keys, width], values).unwrap())
            }
        };
        let q = activations(queries, heads * head_dimension, 88);
        let expected = magnetar_provider_cpu::attention_with_value_width(
            &q,
            &latent_keys(&rows, &w_k, latent, heads).unwrap(),
            &latent_values(&rows, &w_v, latent).unwrap(),
            heads,
            head_dimension,
            value,
            Some(heads),
            None,
            true,
            Some(0.2),
        )
        .unwrap();
        let actual = kernels
            .attention_latent_kv(
                [&upload(&kernels, &q), &store, &store],
                [&dw_k, &dw_v],
                latent,
                [heads, head_dimension],
                Some(value),
                Some(heads),
                None,
                true,
                Some(keys),
                Some(0.2),
            )
            .unwrap();
        let actual = download(&kernels, &actual);
        assert_eq!(actual.shape, expected.shape, "{label}");
        for (index, (a, e)) in actual.data.iter().zip(&expected.data).enumerate() {
            assert!(
                (a - e).abs() <= 2e-3,
                "{label}, element {index}: cuda={a} reference={e}"
            );
        }
    }
}

/// The Device time of one latent prefill attention pass
/// (`attention_latent_kv`) at DeepSeek-V2-Lite's shapes (16 heads, 512
/// latent and 64 rope values, 128-wide keys and values, packed `Q8_0`
/// projections): 2,048 queries after 2,048 to 16,384 keys of an f32 and a
/// `tq4` history (with and without a 1,024-row exact head), and of its
/// first step alone (the rows split, a `tq4` history dequantized);
/// diagnostic only.
#[test]
#[ignore = "GPU timing diagnostic"]
fn latent_prefill_attention_times() {
    use magnetar_provider_api::KvStorePrecision;
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let context = kernels.stream().context().clone();
    let timing = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let time_ms = |repeats: usize, run: &dyn Fn()| -> f32 {
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
    let (heads, nope, rope, latent, value) = (16u64, 128u64, 64u64, 512u64, 128u64);
    let width = latent + rope;
    let w_k = kernels
        .upload_packed_weight(&packed_q8(heads * nope, latent, true, 89))
        .unwrap();
    let w_v = kernels
        .upload_packed_weight(&packed_q8(heads * value, latent, true, 90))
        .unwrap();
    let queries = 2048u64;
    let q = upload(&kernels, &activations(queries, heads * (nope + rope), 91));
    for keys in [2048u64, 8192, 16384] {
        let capacity = keys + 64;
        let data = activations(capacity, width, 92);
        let f32_store = upload(&kernels, &data);
        let mut stores = vec![("f32".to_string(), f32_store)];
        for exact in [0u64, 1024] {
            let mut store = kernels
                .tq_zeros_with_exact_head(&[capacity, width], KvStorePrecision::Tq4, 64, exact)
                .unwrap();
            kernels
                .kv_write(&mut store, &upload(&kernels, &data), 0)
                .unwrap();
            stores.push((format!("tq4, head {exact}"), store));
        }
        for (label, store) in &stores {
            let pass = time_ms(3, &|| {
                kernels
                    .attention_latent_kv(
                        [&q, store, store],
                        [&w_k, &w_v],
                        latent,
                        [heads, nope + rope],
                        Some(value),
                        Some(heads),
                        None,
                        true,
                        Some(keys),
                        None,
                    )
                    .unwrap();
            });
            let split = time_ms(3, &|| {
                kernels.latent_split(store, Some(keys), latent).unwrap();
            });
            eprintln!(
                "latent prefill attention, {queries} queries after {keys} keys, {label}: \
                 {pass:.2} ms, of which the split {split:.2} ms"
            );
        }
    }
}

/// Attention with V rows narrower than Q and K (`value_head_dimension`,
/// `express-latent-attention-forms`) in the Kernels that serve it: split-key
/// decode (one chunk and several), the tiled prefill Kernel (192-dimension
/// keys, 128-dimension values: several query tiles, a window, grouped
/// heads, queries after a longer history), the block Kernel and the
/// one-thread Kernel (a history too long for the block Kernel's shared
/// memory); equal widths unchanged.
#[test]
fn attention_with_a_value_width_matches_reference_cpu() {
    use magnetar_runtime::reference_cpu::attention_with_value_width;
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    // (query rows, key rows, heads, kv heads, head dimension, value
    // dimension, window).
    let shapes = [
        (1_u64, 40_u64, 4_u64, 4_u64, 192_u64, 128_u64, None),
        (1, 300, 2, 2, 192, 128, None),
        (1, 300, 4, 2, 24, 16, Some(64_u64)),
        (7, 7, 2, 2, 192, 128, None),
        (130, 130, 3, 3, 192, 128, None),
        (70, 300, 4, 2, 192, 128, None),
        (100, 100, 2, 1, 192, 128, Some(40)),
        (24, 24, 4, 1, 12, 20, Some(8)),
        (3, 13_000, 1, 1, 16, 8, None),
        (6, 9, 2, 2, 64, 64, None),
    ];
    for (seq, kv_seq, heads, kv_heads, dim, value_dim, window) in shapes {
        let q = activations(seq, heads * dim, 92);
        let k = activations(kv_seq, kv_heads * dim, 93);
        let v = activations(kv_seq, kv_heads * value_dim, 94);
        let (dq, dk, dv) = (
            upload(&kernels, &q),
            upload(&kernels, &k),
            upload(&kernels, &v),
        );
        for scale in [None, Some(0.05_f32)] {
            let expected = attention_with_value_width(
                &q,
                &k,
                &v,
                heads,
                dim,
                value_dim,
                Some(kv_heads),
                window,
                true,
                scale,
            )
            .unwrap();
            let actual = kernels
                .attention_valued(
                    &dq,
                    &dk,
                    &dv,
                    heads,
                    dim,
                    Some(value_dim),
                    Some(kv_heads),
                    window,
                    true,
                    None,
                    scale,
                )
                .unwrap();
            let actual = download(&kernels, &actual);
            assert_eq!(actual.shape, expected.shape);
            for (index, (a, e)) in actual.data.iter().zip(&expected.data).enumerate() {
                assert!(
                    (a - e).abs() <= 2e-3,
                    "shape {:?}, scale {scale:?}, element {index}: cuda={a} reference={e}",
                    (seq, kv_seq, heads, kv_heads, dim, value_dim, window)
                );
            }
        }
    }
    // A decode step over the first rows of a larger store (`key_count`).
    let (heads, dim, value_dim, rows, keys) = (2_u64, 192_u64, 128_u64, 256_u64, 150_u64);
    let q = activations(1, heads * dim, 95);
    let k = activations(rows, heads * dim, 96);
    let v = activations(rows, heads * value_dim, 97);
    let first = |tensor: &HostTensor, width: u64| {
        HostTensor::new(
            [keys, width],
            tensor.data[..(keys * width) as usize].to_vec(),
        )
        .unwrap()
    };
    let expected = attention_with_value_width(
        &q,
        &first(&k, heads * dim),
        &first(&v, heads * value_dim),
        heads,
        dim,
        value_dim,
        None,
        None,
        true,
        None,
    )
    .unwrap();
    let actual = kernels
        .attention_valued(
            &upload(&kernels, &q),
            &upload(&kernels, &k),
            &upload(&kernels, &v),
            heads,
            dim,
            Some(value_dim),
            None,
            None,
            true,
            Some(keys),
            None,
        )
        .unwrap();
    let actual = download(&kernels, &actual);
    assert_eq!(actual.shape, expected.shape);
    for (a, e) in actual.data.iter().zip(&expected.data) {
        assert!((a - e).abs() <= 2e-3, "key_count: cuda={a} reference={e}");
    }
    // V rows of the wrong width are refused.
    let q = upload(&kernels, &activations(1, 2 * 16, 98));
    let k = upload(&kernels, &activations(4, 2 * 16, 99));
    let v = upload(&kernels, &activations(4, 2 * 12, 100));
    assert!(
        kernels
            .attention_valued(&q, &k, &v, 2, 16, Some(8), None, None, true, None, None)
            .is_err()
    );
}

/// Head-wise `concat` (`axis = "heads"`), per head and with the second
/// input's one head given to every head; mismatched shapes refused.
#[test]
fn concat_heads_matches_reference_cpu() {
    use magnetar_provider_api::operator_semantics::concat_heads;
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    for (heads, a_width, second_heads, b_width) in [(4_u64, 6_u64, 4_u64, 2_u64), (16, 128, 1, 64)]
    {
        let a = activations(5, heads * a_width, 101);
        let b = activations(5, second_heads * b_width, 102);
        let expected = concat_heads(&a, &b, heads, second_heads).unwrap();
        let actual = kernels
            .concat_heads(
                &upload(&kernels, &a),
                &upload(&kernels, &b),
                heads,
                second_heads,
            )
            .unwrap();
        let actual = download(&kernels, &actual);
        assert_eq!(actual.shape, expected.shape);
        assert_eq!(actual.data, expected.data, "{heads} heads");
    }
    let a = upload(&kernels, &activations(5, 12, 103));
    assert!(
        kernels
            .concat_heads(&a, &upload(&kernels, &activations(4, 3, 104)), 4, 1)
            .is_err()
    );
    assert!(
        kernels
            .concat_heads(&a, &upload(&kernels, &activations(5, 6, 105)), 4, 2)
            .is_err()
    );
}
