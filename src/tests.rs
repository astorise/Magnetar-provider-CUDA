//! `CudaProvider` unit tests. These are written to pass identically whether
//! or not a compatible CUDA driver/GPU is present, because `CudaProvider`'s
//! own contract branches on that (see `provider.rs`'s module doc): CI's
//! GPU-less `submodule-integration` runner exercises the "unavailable"
//! branch, this workstation and `arc-gpu-magnetar` exercise the "available"
//! branch, and both are asserted here rather than skipped.

use crate::error::{CudaError, CudaErrorCode};
use crate::provider::{CUDA_PROVIDER_NAME, CudaProvider};
use magnetar_runtime::affinity::ProviderHealth;
use magnetar_runtime::kernel::{KernelError, KernelErrorCode};
use magnetar_runtime::provider::Provider;

#[test]
fn out_of_device_memory_maps_to_dedicated_kernel_error_category() {
    let error = CudaError::new(
        CudaErrorCode::OutOfDeviceMemory,
        "requested 8GiB, 2GiB free",
    );
    let kernel_error: KernelError = error.into();
    assert_eq!(
        kernel_error.code(),
        KernelErrorCode::KernelOutOfDeviceMemory
    );
    match kernel_error {
        KernelError::KernelOutOfDeviceMemory { reason } => {
            assert!(reason.contains("8GiB"));
        }
        other => panic!("expected KernelOutOfDeviceMemory, got {other:?}"),
    }
}

/// `implement-cuda-provider-baseline` task 7.3: the test above only proves
/// the *mapping* once a `CudaErrorCode::OutOfDeviceMemory` already exists;
/// this one proves a genuine `cuMemAlloc` call actually fails and maps
/// through that same path end to end, deterministically -- without flaking
/// across GPUs with wildly different amounts of *free* VRAM at test time.
///
/// A request just modestly over the device's own *total* memory is not
/// enough: on Windows/WDDM (confirmed empirically against this
/// workstation's driver), `cuMemAlloc` can transparently page a small
/// excess out to system memory instead of failing -- over-subscription,
/// not a bug this test should punish. Requesting 64x the device's total
/// memory instead exceeds any realistic combination of VRAM, system RAM,
/// and page/swap file on real hardware, forcing a genuine allocation
/// failure regardless of the platform's over-subscription behavior.
#[test]
fn out_of_device_memory_is_triggered_by_a_real_over_capacity_allocation() {
    let provider = CudaProvider::new();
    let Some(context) = provider.context() else {
        // No compatible driver/device on this host at all -- nothing to
        // force an allocation failure against (GPU-less CI).
        return;
    };
    let total_bytes = context
        .total_mem()
        .expect("a discovered context must report its own total memory");
    let element_bytes = std::mem::size_of::<f32>();
    let impossible_len = total_bytes.saturating_mul(64) / element_bytes;
    let stream = context.default_stream();
    let driver_error = stream
        .alloc_zeros::<f32>(impossible_len)
        .expect_err("allocating 64x the device's total memory must fail on any real machine");
    let cuda_error: CudaError = driver_error.into();
    assert_eq!(cuda_error.code, CudaErrorCode::OutOfDeviceMemory);
    let kernel_error: KernelError = cuda_error.into();
    assert_eq!(
        kernel_error.code(),
        KernelErrorCode::KernelOutOfDeviceMemory
    );
}

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
fn health_is_degraded_when_device_found_but_executor_missing() {
    let provider = CudaProvider::new();
    let Some(context) = provider.context() else {
        // No compatible driver/device on this host at all -- nothing to
        // simulate a partial failure against (GPU-less CI).
        return;
    };
    let device = crate::device::cuda_device_descriptor(&context)
        .expect("device discovery must succeed given a context already exists");
    let degraded = CudaProvider::with_device_but_no_executor_for_test(context, device);
    assert_eq!(
        degraded.health(),
        ProviderHealth::Degraded,
        "a Device found but no working executor must report Degraded, not Available"
    );
    assert!(degraded.execution_api().is_none());
    assert_eq!(
        degraded.devices().len(),
        1,
        "the Device itself is still real and should still be reported"
    );
}

#[test]
fn initialize_and_shutdown_are_infallible_regardless_of_hardware() {
    let provider = CudaProvider::new();
    provider.initialize().expect("initialize must not fail");
    provider.shutdown().expect("shutdown must not fail");
}

/// `docs/audits/cuda-provider-full-audit-2026-09-05.md`'s P1 finding: every
/// other hardware-gated test/conformance profile in this suite quietly
/// early-returns (`if !provider.is_available() { return; }`) when no GPU
/// is present, so a green report alone cannot distinguish "genuinely
/// exercised real hardware" from "silently skipped everywhere" -- the
/// audit's literal ask was an explicit `provider-hardware-cuda == Passed`
/// signal, not just "the report contains no Failed".
///
/// `#[ignore]`d by default for exactly that reason: it must never quietly
/// pass on a GPU-less machine the way the others correctly do. It exists
/// to be run explicitly, with `--include-ignored`, only by
/// `gpu-runner-smoke.yml` on `arc-gpu-magnetar`, which guarantees a real
/// GPU -- there, this test's failure means the self-hosted runner itself
/// lost GPU access (turning every other hardware-gated test in this suite
/// into a silent no-op pass), not "no GPU present" (never a valid state on
/// that specific job).
#[test]
#[ignore = "run explicitly via `cargo test -- --include-ignored` on a host guaranteed to have a GPU (arc-gpu-magnetar); every other test in this suite already covers the GPU-less path"]
fn hardware_conformance_actually_ran_not_silently_skipped() {
    let provider = CudaProvider::new();
    assert!(
        provider.is_available(),
        "this test only runs where a compatible CUDA driver/device is \
         guaranteed present -- if it fails, the runner lost GPU access"
    );
}
