//! CUDA Kernel advertisements for the `operator-scope` required-now tier
//! this baseline implements. Mirrors `providers/cpu`'s
//! `reference_cpu_kernel_advertisements()` pattern -- same portable Operator
//! identities (`matmul`, `embedding`, `rmsnorm`, `rope`, `attention`,
//! `softmax`, `silu`, `add`, `mul`, `residual-add`), so Runtime Kernel
//! Registry validation treats CUDA and Reference CPU kernels as
//! alternative implementations of the *same* Operator -- but only ever
//! advertising what [`crate::kernels::CudaKernels`] actually implements
//! (`cuda-provider`'s "CUDA Provider Kernel Advertisements" requirement:
//! gelu/activation/dtype-conversion/layout-conversion are not implemented
//! here and so are never advertised).

/// The `attention` Kernel feature advertised when it honours the `key_count`
/// attribute (`append-kv-history-in-place`).
pub const KEY_COUNT_FEATURE: &str = "key-count";

use magnetar_core::ComputeDType;
use magnetar_core::capability::CapabilityVersion;
use magnetar_core::device::DeviceId;
use magnetar_core::{DeviceBinding, ProviderBinding};
use magnetar_core::{
    operator::ATTENTION_SCALE_FEATURE, operator::EMBEDDING_SCALE_FEATURE,
    operator::MOE_EXPERTS_ACTIVATION_FEATURE, operator::MOE_PER_EXPERT_SCALE_FEATURE,
    operator::OperatorFamily, operator::OperatorId, operator::RMSNORM_PER_HEAD_FEATURE,
    operator::RMSNORM_WITHOUT_WEIGHT_FEATURE, operator::ROPE_BANDED_FREQUENCY_SCALING_FEATURE,
    operator::ROPE_PARTIAL_ROTARY_FEATURE, operator::TensorLayoutKind, operator::TensorRole,
};
use magnetar_kernel::{
    kernel::KernelAdvertisement, kernel::KernelCancellationSupport, kernel::KernelId,
    kernel::KernelImplementationFamily, kernel::KernelMemoryClass,
    kernel::KernelOperatorVersionRange,
};

pub const CUDA_KERNEL_FAMILY: KernelImplementationFamily = KernelImplementationFamily::Cuda;
pub const CUDA_CONFORMANCE_PROFILE: &str = "cuda-conformance-v1";

fn cuda_kernel_id(operator: OperatorId, name: &str, provider_name: &str) -> KernelId {
    KernelId::new(
        ProviderBinding::new(provider_name),
        name,
        CapabilityVersion::new(1, 0, 0),
        operator,
        KernelOperatorVersionRange::exact(1),
        CUDA_KERNEL_FAMILY,
    )
    .with_conformance_profile(CUDA_CONFORMANCE_PROFILE)
}

fn baseline_advertisement(
    name: &str,
    family: OperatorFamily,
    device_id: &DeviceId,
    provider_name: &str,
) -> KernelAdvertisement {
    let operator = OperatorId::magnetar(name, 1, family);
    let id = cuda_kernel_id(operator, name, provider_name);
    let mut advertisement = KernelAdvertisement::new(id)
        .with_dtypes(TensorRole::Input, [ComputeDType::Float32])
        .with_dtypes(TensorRole::Output, [ComputeDType::Float32])
        .with_layouts([TensorLayoutKind::Contiguous])
        .with_memory_classes([KernelMemoryClass::Device])
        .with_devices([DeviceBinding::new(device_id.clone())]);
    // This baseline executes synchronously per call (no Execution Stream
    // extension yet -- design.md's "no async streams yet" decision) and
    // cannot cooperatively cancel mid-kernel.
    advertisement.cancellation = KernelCancellationSupport::TimeoutOnly;
    if matches!(name, "matmul" | "rope" | "attention" | "softmax") {
        advertisement.shape.rank = Some(2);
    }
    advertisement
}

/// An alternative implementation of the same `add`/`mul` Operator as
/// [`baseline_advertisement`], advertising `Float16`/`BrainFloat16` instead
/// of `Float32` -- a distinct `KernelId.name` (`"add-half"`/`"mul-half"`)
/// under the identical `OperatorId`, exactly the same "same Operator,
/// multiple candidate Kernels" shape this Provider and Reference CPU
/// already form for every other Operator (`enable-native-cuda-half-
/// precision-elementwise-compute`'s Phase 3: making
/// [`crate::kernels::CudaKernels::add_half`]/`mul_half` -- real and
/// hardware-verified since Phase 2, but previously reachable only by
/// calling those methods directly -- selectable through the Runtime's
/// actual Kernel Registry/dispatch contract for the first time).
/// [`baseline_advertisement`] declaring `features`: the optional operator
/// forms its Kernel honours (`required_kernel_features`).
fn advertisement_with_features(
    name: &str,
    family: OperatorFamily,
    device_id: &DeviceId,
    provider_name: &str,
    features: &[&str],
) -> KernelAdvertisement {
    let mut advertisement = baseline_advertisement(name, family, device_id, provider_name);
    advertisement
        .id
        .features
        .extend(features.iter().map(|feature| feature.to_string()));
    advertisement
}

fn half_precision_advertisement(
    base_name: &str,
    family: OperatorFamily,
    device_id: &DeviceId,
    provider_name: &str,
) -> KernelAdvertisement {
    let operator = OperatorId::magnetar(base_name, 1, family);
    let name = format!("{base_name}-half");
    let id = cuda_kernel_id(operator, &name, provider_name);
    let mut advertisement = KernelAdvertisement::new(id)
        .with_dtypes(
            TensorRole::Input,
            [ComputeDType::Float16, ComputeDType::BrainFloat16],
        )
        .with_dtypes(
            TensorRole::Output,
            [ComputeDType::Float16, ComputeDType::BrainFloat16],
        )
        .with_layouts([TensorLayoutKind::Contiguous])
        .with_memory_classes([KernelMemoryClass::Device])
        .with_devices([DeviceBinding::new(device_id.clone())]);
    advertisement.cancellation = KernelCancellationSupport::TimeoutOnly;
    advertisement
}

/// The CUDA Provider's implemented kernel set for one discovered Device.
///
/// `provider_name` is a real parameter (see `device::cuda_device_descriptor`'s
/// own doc comment): a second `CudaProvider` bound to a different real GPU
/// ordinal advertises its Kernels under its own distinct Provider name, not
/// the default `provider::CUDA_PROVIDER_NAME`.
pub fn cuda_kernel_advertisements(
    device_id: &DeviceId,
    provider_name: &str,
) -> Vec<KernelAdvertisement> {
    vec![
        baseline_advertisement(
            "matmul",
            OperatorFamily::LinearAlgebra,
            device_id,
            provider_name,
        ),
        baseline_advertisement(
            "matmul-bias",
            OperatorFamily::LinearAlgebra,
            device_id,
            provider_name,
        ),
        baseline_advertisement(
            "matmul-residual",
            OperatorFamily::LinearAlgebra,
            device_id,
            provider_name,
        ),
        baseline_advertisement(
            "silu-mul",
            OperatorFamily::Activation,
            device_id,
            provider_name,
        ),
        baseline_advertisement(
            "gated-matmul",
            OperatorFamily::LinearAlgebra,
            device_id,
            provider_name,
        ),
        // Honours `activation` and the sixth input, `per_expert_scale`.
        advertisement_with_features(
            "moe-experts",
            OperatorFamily::LinearAlgebra,
            device_id,
            provider_name,
            &[MOE_EXPERTS_ACTIVATION_FEATURE, MOE_PER_EXPERT_SCALE_FEATURE],
        ),
        baseline_advertisement(
            "row-scale",
            OperatorFamily::Tensor,
            device_id,
            provider_name,
        ),
        baseline_advertisement(
            "activation",
            OperatorFamily::Activation,
            device_id,
            provider_name,
        ),
        // Honours `scale`.
        advertisement_with_features(
            "embedding",
            OperatorFamily::Tensor,
            device_id,
            provider_name,
            &[EMBEDDING_SCALE_FEATURE],
        ),
        // Honours per-head normalization and a missing weight.
        advertisement_with_features(
            "rmsnorm",
            OperatorFamily::Normalization,
            device_id,
            provider_name,
            &[RMSNORM_PER_HEAD_FEATURE, RMSNORM_WITHOUT_WEIGHT_FEATURE],
        ),
        // Honours the banded frequency-scaling attributes and partial
        // rotation.
        advertisement_with_features(
            "rope",
            OperatorFamily::PositionEncoding,
            device_id,
            provider_name,
            &[
                ROPE_BANDED_FREQUENCY_SCALING_FEATURE,
                ROPE_PARTIAL_ROTARY_FEATURE,
            ],
        ),
        // Honours `key_count` (`append-kv-history-in-place`) and `scale`.
        advertisement_with_features(
            "attention",
            OperatorFamily::Attention,
            device_id,
            provider_name,
            &[KEY_COUNT_FEATURE, ATTENTION_SCALE_FEATURE],
        ),
        baseline_advertisement(
            "softmax",
            OperatorFamily::Activation,
            device_id,
            provider_name,
        ),
        baseline_advertisement("silu", OperatorFamily::Activation, device_id, provider_name),
        // The tanh approximation.
        baseline_advertisement("gelu", OperatorFamily::Activation, device_id, provider_name),
        baseline_advertisement("add", OperatorFamily::Tensor, device_id, provider_name),
        baseline_advertisement("mul", OperatorFamily::Tensor, device_id, provider_name),
        baseline_advertisement("concat", OperatorFamily::Tensor, device_id, provider_name),
        baseline_advertisement("kv-write", OperatorFamily::Tensor, device_id, provider_name),
        baseline_advertisement(
            "residual-add",
            OperatorFamily::Tensor,
            device_id,
            provider_name,
        ),
        half_precision_advertisement("add", OperatorFamily::Tensor, device_id, provider_name),
        half_precision_advertisement("mul", OperatorFamily::Tensor, device_id, provider_name),
    ]
}
