//! `CudaProvider`: an optimized, GPU-executing Provider built on top of
//! `cudarc`'s dynamic-loading driver bindings.
//!
//! # Graceful unavailability
//!
//! `CudaProvider::new()` always constructs successfully, even on a host with
//! no CUDA driver, no compatible GPU, or a driver/runtime version mismatch --
//! it never fails Runtime initialization for the absence of hardware. When
//! driver discovery fails, this Provider reports zero Devices and
//! [`ProviderHealth::Unavailable`] instead of [`ProviderHealth::Unhealthy`]/
//! failing outright: this is expected, policy-relevant absence of hardware,
//! not an internal fault (`cuda-provider`'s "Graceful Unavailability Without
//! Compatible Hardware" requirement). This is also what keeps this crate's
//! `cargo test` green on CI's GPU-less, CUDA-Toolkit-less
//! `submodule-integration` runner: `cudarc`'s `dynamic-loading` feature never
//! links against the CUDA driver/NVRTC at build time (see `Cargo.toml`), so
//! the only thing that can fail there is the runtime `dlopen` attempt this
//! module already treats as a normal, expected outcome.
//!
//! One real-hardware-vs-CI discrepancy this had to learn the hard way
//! (confirmed by CI's `submodule-integration`/`provider integration` jobs,
//! which run on a genuinely driver-less `ubuntu-latest`, unlike every
//! machine this crate had been manually verified on so far): `cudarc`'s
//! dynamic-loading mode does **not** turn "the shared library file does not
//! exist anywhere on this system" into a catchable [`DriverError`] --
//! `cudarc::panic_no_lib_found` unconditionally `panic!`s in that specific
//! case (a version-mismatch-but-library-present case, the only one this
//! module originally accounted for, *does* return a normal `Err`). Every
//! function that can trigger a fresh dlopen attempt is therefore wrapped in
//! [`std::panic::catch_unwind`] below, converting that panic into the same
//! graceful-unavailable outcome a `DriverError` would have produced. The
//! panic message still prints to stderr (Rust's default hook runs before
//! unwinding, and swapping the process-global hook here would risk
//! swallowing a genuinely different panic on another thread in the same
//! `cargo test` run) -- noisy but harmless, and still a normal test pass.

use cudarc::driver::sys::CUresult;
use cudarc::driver::{CudaContext, DriverError};
use magnetar_runtime::affinity::ProviderHealth;
use magnetar_runtime::device::{Device, DeviceDescriptor};
use magnetar_runtime::kernel::KernelAdvertisement;
use magnetar_runtime::provider::{
    Provider, ProviderError, ProviderExecutionApi, ProviderMetadata, ProviderRegistry,
};
use std::sync::Arc;

use crate::advertisements::cuda_kernel_advertisements;
use crate::device::cuda_device_descriptor;
use crate::executor::CudaExecutor;
use crate::kernels::CudaKernels;

/// Stable, package-qualified CUDA Provider identity.
pub const CUDA_PROVIDER_NAME: &str = "magnetar:provider/cuda";
pub const CUDA_PROVIDER_VERSION: &str = "0.1.0";
pub const CUDA_PROVIDER_VENDOR: &str = "magnetar";

pub fn cuda_provider_metadata() -> ProviderMetadata {
    ProviderMetadata::new(
        CUDA_PROVIDER_NAME,
        CUDA_PROVIDER_VERSION,
        CUDA_PROVIDER_VENDOR,
        "Optimized, GPU-executing Provider built on the CUDA driver API",
    )
}

/// The CUDA Provider itself. Registers as a built-in Provider (no dynamic-
/// library Provider ABI in this baseline -- see design.md's "Dynamic-library
/// Provider ABI loading" non-goal).
pub struct CudaProvider {
    metadata: ProviderMetadata,
    /// `Some` only when a compatible CUDA driver and device were found at
    /// construction time. Kept alive for the Provider's lifetime: dropping
    /// it would release the underlying primary context out from under any
    /// live device-resident resource.
    context: Option<Arc<CudaContext>>,
    device: Option<Arc<DeviceDescriptor>>,
    /// `Some` only when a device was found *and* this baseline's kernels
    /// compiled successfully through NVRTC (design.md: compiled once, here
    /// at construction rather than lazily on first use -- functionally
    /// equivalent for "not ahead of time, not per call", just simpler to
    /// reason about). A compile failure on a machine that does have a GPU
    /// is not the "no hardware" case [`ProviderHealth::Unavailable`]
    /// describes; it is treated narrowly here by leaving `devices()`/
    /// `health()` reporting the real hardware as found while
    /// `execution_api()` alone reports `None`, since no broader Provider
    /// Health state in this baseline's contract cleanly names "hardware
    /// present, own kernels broken".
    executor: Option<Arc<CudaExecutor>>,
}

impl CudaProvider {
    pub fn new() -> Self {
        match Self::discover_primary_device_catching_missing_library_panic() {
            Ok((context, device)) => {
                let device = Arc::new(device);
                let executor =
                    Self::compile_kernels_catching_missing_library_panic(&context).map(|kernels| {
                        Arc::new(CudaExecutor::new(kernels, device.metadata.id.clone()))
                    });
                Self {
                    metadata: cuda_provider_metadata(),
                    context: Some(context),
                    device: Some(device),
                    executor,
                }
            }
            Err(_reason) => Self {
                metadata: cuda_provider_metadata(),
                context: None,
                device: None,
                executor: None,
            },
        }
    }

    /// [`Self::discover_primary_device`], but also converts a
    /// `cudarc::panic_no_lib_found` panic (the shared library is completely
    /// absent, not merely an incompatible version -- see this module's doc
    /// comment) into the same `Err` outcome the caller already handles.
    fn discover_primary_device_catching_missing_library_panic()
    -> Result<(Arc<CudaContext>, DeviceDescriptor), DriverError> {
        match std::panic::catch_unwind(Self::discover_primary_device) {
            Ok(result) => result,
            Err(_panic) => Err(DriverError(CUresult::CUDA_ERROR_NO_DEVICE)),
        }
    }

    /// Attempts to load the CUDA driver and bind device ordinal 0. May
    /// panic via `cudarc` if the driver shared library is completely
    /// absent -- callers must go through
    /// [`Self::discover_primary_device_catching_missing_library_panic`],
    /// never this directly.
    fn discover_primary_device() -> Result<(Arc<CudaContext>, DeviceDescriptor), DriverError> {
        let device_count = CudaContext::device_count()?;
        if device_count <= 0 {
            return Err(DriverError(CUresult::CUDA_ERROR_NO_DEVICE));
        }
        let context = CudaContext::new(0)?;
        let device = cuda_device_descriptor(&context)?;
        Ok((context, device))
    }

    /// [`CudaKernels::compile_and_load`], but also converts an NVRTC
    /// `cudarc::panic_no_lib_found` panic (the NVRTC shared library is
    /// completely absent -- same class of issue as the driver, in principle
    /// reachable even when the driver itself was found) into `None`, the
    /// same outcome a compile/load `Err` already produces here.
    fn compile_kernels_catching_missing_library_panic(
        context: &Arc<CudaContext>,
    ) -> Option<CudaKernels> {
        std::panic::catch_unwind(|| CudaKernels::compile_and_load(context))
            .ok()
            .and_then(Result::ok)
    }

    /// Whether this Provider found a usable CUDA driver and device.
    pub fn is_available(&self) -> bool {
        self.context.is_some()
    }

    /// This Provider's CUDA context, if a compatible driver/device was
    /// found. Runtime-internal: never exposed through the portable
    /// `Provider`/`Device` contract, only used by this crate's own
    /// [`crate::kernels::CudaKernels`] and tests.
    pub fn context(&self) -> Option<Arc<CudaContext>> {
        self.context.clone()
    }
}

impl Default for CudaProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for CudaProvider {
    fn metadata(&self) -> ProviderMetadata {
        self.metadata.clone()
    }

    fn register(&self, _registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
        // Device registration is performed by the Runtime/ProviderLoader via
        // `devices()`, matching `ReferenceCpuProvider`'s pattern -- registering
        // here too would double-register them.
        Ok(())
    }

    fn health(&self) -> ProviderHealth {
        if self.is_available() {
            ProviderHealth::Available
        } else {
            ProviderHealth::Unavailable
        }
    }

    fn devices(&self) -> Vec<Arc<dyn Device>> {
        match &self.device {
            Some(device) => vec![device.clone()],
            None => Vec::new(),
        }
    }

    fn kernel_advertisements(&self) -> Vec<KernelAdvertisement> {
        match &self.device {
            Some(device) => cuda_kernel_advertisements(&device.metadata.id),
            // No Device discovered means no Kernel can execute anywhere;
            // advertising kernels bound to no real Device would be a false
            // claim of availability.
            None => Vec::new(),
        }
    }

    fn initialize(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn shutdown(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn execution_api(&self) -> Option<Arc<dyn ProviderExecutionApi>> {
        self.executor
            .clone()
            .map(|executor| executor as Arc<dyn ProviderExecutionApi>)
    }
}
