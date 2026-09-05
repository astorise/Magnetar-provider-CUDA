//! Structured CUDA Provider errors. Mirrors `providers/cpu`'s
//! `ReferenceCpuError`/`ReferenceCpuErrorCode` shape so the two Providers'
//! failure reporting is easy to compare, but stays a distinct type: CUDA
//! failure categories (device memory, NVRTC compilation, kernel launch)
//! don't all have a CPU equivalent.
//!
//! `provider`'s "CUDA Provider Error Categories" requirement: native CUDA
//! driver/NVRTC errors are mapped to these stable categories, with the
//! native error attached only as a redacted diagnostic string -- never a raw
//! pointer, context, or module handle.

use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaErrorCode {
    /// Requested tensor shape/rank/dimension combination is not valid for
    /// the operator (mirrors `ReferenceCpuErrorCode::ShapeUnsupported`).
    ShapeUnsupported,
    /// Requested dtype is not `f32` (this baseline's only supported dtype).
    DTypeUnsupported,
    /// Requested layout is not contiguous (this baseline's only supported
    /// layout).
    LayoutUnsupported,
    /// The kernel ran but could not produce a defined result (e.g. softmax
    /// over an all-`-inf` row).
    ExecutionFailed,
    /// Device memory allocation failed.
    OutOfDeviceMemory,
    /// NVRTC failed to compile this Provider's own kernel source. Not
    /// expected in normal operation; surfaced rather than panicking.
    CompilationFailed,
    /// A CUDA driver call unrelated to allocation failed (module load,
    /// function lookup, kernel launch, memcpy, synchronize).
    DriverFailed,
    /// No compatible CUDA driver/device was available when this call was
    /// attempted (mirrors [`crate::provider::CudaProvider::is_available`]
    /// being `false`).
    DeviceUnavailable,
}

/// A structured CUDA Provider error. `detail` carries a redacted diagnostic
/// message (native error text) -- never a native pointer, handle, or context
/// value (`provider`'s "Native Detail Privacy" / "CUDA Provider Does Not
/// Expose Native Handles").
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaError {
    pub code: CudaErrorCode,
    pub detail: String,
}

impl CudaError {
    pub fn new(code: CudaErrorCode, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.detail)
    }
}

impl std::error::Error for CudaError {}

impl From<cudarc::driver::DriverError> for CudaError {
    fn from(error: cudarc::driver::DriverError) -> Self {
        let code = match error.0 {
            cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY => {
                CudaErrorCode::OutOfDeviceMemory
            }
            cudarc::driver::sys::CUresult::CUDA_ERROR_NO_DEVICE
            | cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_DEVICE => {
                CudaErrorCode::DeviceUnavailable
            }
            _ => CudaErrorCode::DriverFailed,
        };
        Self::new(code, error.to_string())
    }
}

impl From<cudarc::nvrtc::CompileError> for CudaError {
    fn from(error: cudarc::nvrtc::CompileError) -> Self {
        Self::new(CudaErrorCode::CompilationFailed, error.to_string())
    }
}

/// Mirrors `magnetar_runtime::reference_cpu`'s
/// `From<ReferenceCpuError> for KernelError` mapping, so a `CudaError`
/// surfaces through `run_invocation` the same way a `ReferenceCpuError`
/// does for `providers/cpu`.
impl From<CudaError> for magnetar_runtime::kernel::KernelError {
    fn from(error: CudaError) -> Self {
        use magnetar_runtime::kernel::KernelError;
        match error.code {
            CudaErrorCode::DTypeUnsupported => KernelError::KernelDTypeUnsupported {
                dtype: error.detail,
            },
            CudaErrorCode::LayoutUnsupported => KernelError::KernelLayoutUnsupported {
                layout: error.detail,
            },
            CudaErrorCode::ShapeUnsupported => KernelError::KernelShapeUnsupported {
                reason: error.detail,
            },
            CudaErrorCode::DeviceUnavailable => KernelError::KernelDeviceUnsupported {
                device: error.detail,
            },
            // No dedicated KernelError variant for allocation/compilation/
            // driver failures; KernelExecutionFailed is the closest honest
            // fit -- these are real per-invocation execution failures, not
            // a request the Runtime could have avoided by asking for
            // something else (unlike the categories above).
            CudaErrorCode::ExecutionFailed
            | CudaErrorCode::OutOfDeviceMemory
            | CudaErrorCode::CompilationFailed
            | CudaErrorCode::DriverFailed => KernelError::KernelExecutionFailed {
                reason: error.detail,
            },
        }
    }
}
