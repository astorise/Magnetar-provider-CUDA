//! Runtime-owned Device metadata for discovered CUDA devices.
//!
//! [`cuda_device_descriptor`] never exposes the native `CudaContext`,
//! `CUdevice`, or `CUcontext` handle it reads from -- only stable portable
//! metadata (name, compute capability folded into `architecture`, memory
//! capacity, a pressure estimate), matching `device`'s "Device Does Not
//! Expose Native Pointer" and `provider-roadmap`'s "Native Handles Remain
//! Hidden" requirements.

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, DriverError};
use magnetar_runtime::affinity::ProviderPressureLevel;
use magnetar_runtime::compute::ComputeDType;
use magnetar_runtime::device::{
    DeviceDescriptor, DeviceExecutionLimits, DeviceId, DeviceMetadata, DeviceType,
};
use magnetar_runtime::kernel::KernelMemoryClass;
use magnetar_runtime::operator::TensorLayoutKind;
use std::sync::Arc;

use crate::provider::CUDA_PROVIDER_NAME;

/// Builds this Runtime's stable identifier for the CUDA device at `ordinal`.
pub fn cuda_device_id(ordinal: usize) -> DeviceId {
    DeviceId::new(format!("cuda:{ordinal}"))
}

/// Builds Runtime-owned Device metadata for one discovered, context-bound
/// CUDA device. Contiguous-f32-only for this baseline (`operator-scope`'s
/// Initial DType/Layout Scope), matching what `cuda-provider`'s spec commits
/// to.
pub fn cuda_device_descriptor(ctx: &Arc<CudaContext>) -> Result<DeviceDescriptor, DriverError> {
    let name = ctx.name()?;
    let (major, minor) = ctx.compute_capability()?;
    let total_mem = ctx.total_mem()? as u64;
    let multiprocessor_count = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .unwrap_or(0)
        .max(0) as u32;
    let pressure = match ctx.mem_get_info() {
        Ok((free, total)) if total > 0 => pressure_from_free_ratio(free, total),
        _ => ProviderPressureLevel::Unknown,
    };

    let mut metadata = DeviceMetadata::new(
        cuda_device_id(ctx.ordinal()),
        name,
        DeviceType::Gpu,
        CUDA_PROVIDER_NAME,
    );
    metadata.vendor = "NVIDIA".into();
    metadata.architecture = format!("sm_{major}{minor}");
    metadata.memory_capacity = total_mem;
    metadata.compute_units = multiprocessor_count;
    metadata.dtype_support = [ComputeDType::Float32].into_iter().collect();
    metadata.layout_support = [TensorLayoutKind::Contiguous].into_iter().collect();
    metadata.memory_class_support = [KernelMemoryClass::Device].into_iter().collect();
    metadata.execution_limits = DeviceExecutionLimits::default();
    metadata.pressure = pressure;

    Ok(DeviceDescriptor::new(metadata))
}

/// Coarse, deliberately bucketed pressure estimate derived from
/// `cuMemGetInfo`'s free/total bytes -- `device`'s "Device SHALL Expose
/// Pressure Estimate" requires an estimate, not exact accounting, so this
/// intentionally does not expose raw byte counts as the pressure signal
/// itself.
fn pressure_from_free_ratio(free: usize, total: usize) -> ProviderPressureLevel {
    let free_ratio = free as f64 / total as f64;
    if free_ratio > 0.5 {
        ProviderPressureLevel::Low
    } else if free_ratio > 0.2 {
        ProviderPressureLevel::Moderate
    } else if free_ratio > 0.05 {
        ProviderPressureLevel::High
    } else {
        ProviderPressureLevel::Saturated
    }
}
