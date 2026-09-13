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
