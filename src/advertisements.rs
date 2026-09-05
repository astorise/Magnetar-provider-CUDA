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

use magnetar_runtime::affinity::{DeviceBinding, ProviderBinding};
use magnetar_runtime::capability::CapabilityVersion;
use magnetar_runtime::compute::ComputeDType;
use magnetar_runtime::device::DeviceId;
use magnetar_runtime::kernel::{
    KernelAdvertisement, KernelCancellationSupport, KernelId, KernelImplementationFamily,
    KernelMemoryClass, KernelOperatorVersionRange,
};
use magnetar_runtime::operator::{OperatorFamily, OperatorId, TensorLayoutKind, TensorRole};

use crate::provider::CUDA_PROVIDER_NAME;

pub const CUDA_KERNEL_FAMILY: KernelImplementationFamily = KernelImplementationFamily::Cuda;
pub const CUDA_CONFORMANCE_PROFILE: &str = "cuda-conformance-v1";

fn cuda_kernel_id(operator: OperatorId, name: &str) -> KernelId {
    KernelId::new(
        ProviderBinding::new(CUDA_PROVIDER_NAME),
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
) -> KernelAdvertisement {
    let operator = OperatorId::magnetar(name, 1, family);
    let id = cuda_kernel_id(operator, name);
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

/// The CUDA Provider's implemented kernel set for one discovered Device.
pub fn cuda_kernel_advertisements(device_id: &DeviceId) -> Vec<KernelAdvertisement> {
    vec![
        baseline_advertisement("matmul", OperatorFamily::LinearAlgebra, device_id),
        baseline_advertisement("embedding", OperatorFamily::Tensor, device_id),
        baseline_advertisement("rmsnorm", OperatorFamily::Normalization, device_id),
        baseline_advertisement("rope", OperatorFamily::PositionEncoding, device_id),
        baseline_advertisement("attention", OperatorFamily::Attention, device_id),
        baseline_advertisement("softmax", OperatorFamily::Activation, device_id),
        baseline_advertisement("silu", OperatorFamily::Activation, device_id),
        baseline_advertisement("add", OperatorFamily::Tensor, device_id),
        baseline_advertisement("mul", OperatorFamily::Tensor, device_id),
        baseline_advertisement("residual-add", OperatorFamily::Tensor, device_id),
    ]
}
