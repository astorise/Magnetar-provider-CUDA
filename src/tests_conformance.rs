//! Numerical conformance: `CudaKernels` output checked against
//! `providers/cpu`'s reference functions on the same small fixtures
//! (`provider-roadmap`'s "Reference CPU Remains Correctness Baseline" /
//! `cuda-provider`'s "CUDA Kernels Match Reference CPU Semantics").
//!
//! Every test here skips cleanly (returns without assertions) when no
//! compatible CUDA driver/device is present, rather than failing --
//! `cuda-provider`'s conformance scope is explicitly hardware-gated (see
//! `design.md`'s "graceful unavailability" decision and tasks.md 9.3).

use magnetar_runtime::HostTensor;

use crate::kernels::CudaKernels;
use crate::provider::CudaProvider;

const TOLERANCE: f32 = 1e-3;

fn kernels_or_skip() -> Option<CudaKernels> {
    let provider = CudaProvider::new();
    let context = provider.context()?;
    Some(CudaKernels::compile_and_load(&context).expect(
        "kernel compilation must succeed on a machine that already passed device discovery",
    ))
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
    let actual = kernels.add(&a, &b).unwrap();
    assert_close(&actual, &expected);
}

#[test]
fn mul_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a = HostTensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = HostTensor::new([2, 3], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
    let expected = magnetar_provider_cpu::mul(&a, &b).unwrap();
    let actual = kernels.mul(&a, &b).unwrap();
    assert_close(&actual, &expected);
}

#[test]
fn silu_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let input = HostTensor::new([2, 3], [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0]).unwrap();
    let expected = magnetar_provider_cpu::silu(&input);
    let actual = kernels.silu(&input).unwrap();
    assert_close(&actual, &expected);
}

#[test]
fn softmax_rows_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let input = HostTensor::new([2, 3], [1.0, 2.0, 3.0, -1.0, 0.0, 1.0]).unwrap();
    let expected = magnetar_provider_cpu::softmax_rows(&input).unwrap();
    let actual = kernels.softmax_rows(&input).unwrap();
    assert_close(&actual, &expected);
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
    let actual = kernels.embedding_lookup(&table, &ids).unwrap();
    assert_close(&actual, &expected);
}

#[test]
fn embedding_lookup_rejects_out_of_range_id_before_dispatch() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let table = HostTensor::new([2, 2], [0.0, 0.0, 0.0, 0.0]).unwrap();
    let ids = HostTensor::new([1], [5.0]).unwrap();
    assert!(kernels.embedding_lookup(&table, &ids).is_err());
}

#[test]
fn rmsnorm_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let input = HostTensor::new([2, 3], [1.0, 2.0, 3.0, -1.0, 0.5, 2.0]).unwrap();
    let weight = HostTensor::new([3], [1.0, 0.5, 2.0]).unwrap();
    let expected = magnetar_provider_cpu::rmsnorm(&input, &weight, 1e-5).unwrap();
    let actual = kernels.rmsnorm(&input, &weight, 1e-5).unwrap();
    assert_close(&actual, &expected);
}

#[test]
fn rope_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let input = HostTensor::new([2, 4], [1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, -0.5]).unwrap();
    let expected = magnetar_provider_cpu::rope(&input, 10000.0, 1.0, 4, 0).unwrap();
    let actual = kernels.rope(&input, 10000.0, 1.0, 4, 0).unwrap();
    assert_close(&actual, &expected);
}

#[test]
fn matmul_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a = HostTensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = HostTensor::new([3, 2], [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let expected = magnetar_provider_cpu::matmul(&a, &b, false, false).unwrap();
    let actual = kernels.matmul(&a, &b, false, false).unwrap();
    assert_close(&actual, &expected);
}

#[test]
fn matmul_transposed_matches_reference_cpu() {
    let Some(kernels) = kernels_or_skip() else {
        return;
    };
    let a = HostTensor::new([3, 2], [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    let b = HostTensor::new([3, 2], [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let expected = magnetar_provider_cpu::matmul(&a, &b, true, false).unwrap();
    let actual = kernels.matmul(&a, &b, true, false).unwrap();
    assert_close(&actual, &expected);
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
    let actual = kernels
        .attention(&q, &k, &v, 1, 2, None, None, true)
        .unwrap();
    assert_close(&actual, &expected);
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
    let actual = kernels
        .attention(&q, &k, &v, 2, 2, Some(1), Some(2), true)
        .unwrap();
    assert_close(&actual, &expected);
}
