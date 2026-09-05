//! Runs `CudaProvider` through `magnetar_runtime`'s real
//! `ProviderConformanceSuite` (`provider-core`/`provider-compute`) -- the
//! authoritative mechanism `provider`'s "Provider Conformance Suite"
//! requirement defines, not just this crate's own ad hoc assertions.
//!
//! `provider-core` is checked regardless of hardware (it must hold even when
//! `CudaProvider` is gracefully unavailable). `provider-compute` additionally
//! requires an `execution_api()` (task group 8's `CudaExecutor`), so it only
//! runs -- and only can pass -- when a real device was found and its kernels
//! compiled.

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
