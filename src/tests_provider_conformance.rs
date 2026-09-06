//! Runs `CudaProvider` through `magnetar_runtime`'s real
//! `ProviderConformanceSuite` (`provider-core`/`provider-compute`/
//! `provider-data-movement`) -- the authoritative mechanism `provider`'s
//! "Provider Conformance Suite" requirement defines, not just this crate's
//! own ad hoc assertions.
//!
//! `provider-core` is checked regardless of hardware (it must hold even when
//! `CudaProvider` is gracefully unavailable). `provider-compute`/
//! `provider-data-movement` additionally require an `execution_api()` (task
//! group 8's `CudaExecutor`), so they only run -- and only can pass -- when a
//! real device was found and its kernels compiled.
//!
//! `provider-data-movement`'s own conformance check
//! (`magnetar_runtime::conformance::validate_data_movement`) validates
//! whatever `ComputeDataMovementKind`s a Provider's own metadata advertises
//! via `compute_advertisement.data_movement` -- no real Provider (Reference
//! CPU included) populates that map yet, so this profile currently passes
//! vacuously for every real Provider, CUDA included, until a future change
//! wires real data-movement advertisements up. Asserted here anyway
//! (`enable-device-resident-kernel-chaining`'s "no longer deferred" spec
//! change) so a regression that starts advertising movement kinds without
//! actually supporting them is caught by this same test, not silently.

use std::sync::Arc;

use magnetar_runtime::conformance::{
    ProviderConformanceConfig, ProviderConformanceProfile, ProviderConformanceSuite,
    ProviderConformanceTarget,
};

use crate::provider::CudaProvider;

#[test]
fn passes_provider_core_conformance_regardless_of_hardware() {
    let provider = CudaProvider::new();
    let suite = ProviderConformanceSuite::new(
        ProviderConformanceConfig::default()
            .with_profiles([ProviderConformanceProfile::ProviderCore]),
    );
    let report = suite.run(ProviderConformanceTarget::built_in(Arc::new(provider)));
    assert!(
        report.is_conformant(),
        "CudaProvider must pass provider-core whether or not a GPU is present: {report:#?}"
    );
}

#[test]
fn passes_provider_compute_conformance_when_available() {
    let provider = CudaProvider::new();
    if !provider.is_available() {
        return;
    }
    let suite = ProviderConformanceSuite::new(
        ProviderConformanceConfig::default()
            .with_profiles([ProviderConformanceProfile::ProviderCompute]),
    );
    let report = suite.run(ProviderConformanceTarget::built_in(Arc::new(provider)));
    assert!(report.is_conformant(), "{report:#?}");
}

#[test]
fn passes_provider_data_movement_conformance_when_available() {
    let provider = CudaProvider::new();
    if !provider.is_available() {
        return;
    }
    let suite = ProviderConformanceSuite::new(
        ProviderConformanceConfig::default()
            .with_profiles([ProviderConformanceProfile::ProviderDataMovement]),
    );
    let report = suite.run(ProviderConformanceTarget::built_in(Arc::new(provider)));
    assert!(report.is_conformant(), "{report:#?}");
}
