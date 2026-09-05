//! `CudaProvider` unit tests. These are written to pass identically whether
//! or not a compatible CUDA driver/GPU is present, because `CudaProvider`'s
//! own contract branches on that (see `provider.rs`'s module doc): CI's
//! GPU-less `submodule-integration` runner exercises the "unavailable"
//! branch, this workstation and `arc-gpu-magnetar` exercise the "available"
//! branch, and both are asserted here rather than skipped.

use crate::provider::{CUDA_PROVIDER_NAME, CudaProvider};
use magnetar_runtime::affinity::ProviderHealth;
use magnetar_runtime::provider::Provider;

#[test]
fn construction_never_panics_and_reports_stable_identity() {
    let provider = CudaProvider::new();
    let metadata = provider.metadata();
    assert_eq!(metadata.name, CUDA_PROVIDER_NAME);
}

#[test]
fn health_and_devices_agree_with_availability() {
    let provider = CudaProvider::new();
    let devices = provider.devices();

    if provider.is_available() {
        assert_eq!(provider.health(), ProviderHealth::Available);
        assert_eq!(
            devices.len(),
            1,
            "an available CudaProvider must expose exactly the one device it discovered"
        );
        let metadata = devices[0].metadata();
        assert!(!metadata.name.is_empty());
        assert!(
            metadata.memory_capacity > 0,
            "a real CUDA device always reports non-zero total memory"
        );
    } else {
        assert_eq!(
            provider.health(),
            ProviderHealth::Unavailable,
            "no compatible driver/device found must report Unavailable, not Failed/Unhealthy"
        );
        assert!(
            devices.is_empty(),
            "an unavailable CudaProvider must expose zero devices"
        );
    }
}

#[test]
fn kernel_advertisements_agree_with_availability() {
    let provider = CudaProvider::new();
    let advertisements = provider.kernel_advertisements();
    if provider.is_available() {
        assert_eq!(
            advertisements.len(),
            10,
            "expected exactly the required-now kernel set this baseline implements"
        );
        let names: std::collections::BTreeSet<_> =
            advertisements.iter().map(|a| a.id.name.as_str()).collect();
        for expected in [
            "matmul",
            "embedding",
            "rmsnorm",
            "rope",
            "attention",
            "softmax",
            "silu",
            "add",
            "mul",
            "residual-add",
        ] {
            assert!(
                names.contains(expected),
                "missing advertisement: {expected}"
            );
        }
    } else {
        assert!(
            advertisements.is_empty(),
            "an unavailable CudaProvider must not advertise kernels bound to no real Device"
        );
    }
}

#[test]
fn initialize_and_shutdown_are_infallible_regardless_of_hardware() {
    let provider = CudaProvider::new();
    provider.initialize().expect("initialize must not fail");
    provider.shutdown().expect("shutdown must not fail");
}
