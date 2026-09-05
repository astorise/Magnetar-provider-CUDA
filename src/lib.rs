//! CUDA Provider: an optimized, GPU-executing Provider implementing
//! `magnetar:compute/run`, built on top of `cudarc`'s dynamic-loading CUDA
//! driver bindings.
//!
//! See `openspec/changes/implement-cuda-provider-baseline/design.md` for the
//! full rationale, in particular why this crate has no build-time dependency
//! on the CUDA Toolkit or driver (`cudarc`'s `dynamic-loading` feature) and
//! why it constructs successfully even with no compatible GPU present
//! (graceful unavailability, covered by this crate's own tests).

pub mod advertisements;
pub mod device;
pub mod error;
pub mod executor;
pub mod kernels;
pub mod provider;

pub use error::{CudaError, CudaErrorCode};
pub use executor::CudaExecutor;
pub use kernels::CudaKernels;
pub use provider::CudaProvider;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_conformance;
#[cfg(test)]
mod tests_provider_conformance;
