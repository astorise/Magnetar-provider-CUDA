# magnetar-provider-cuda

## Purpose

A CUDA Provider for the [Magnetar](https://github.com/astorise/Magnetar)
local AI Runtime: an optimized, GPU-executing Kernel execution backend
implementing `ProviderExecutionApi` and Magnetar's Provider contract for
NVIDIA Devices. Implements the `operator-scope` required-now Kernel set
(matmul, embedding, rmsnorm, rope, attention, softmax, silu, add, mul,
residual-add) as real CUDA C++ compiled via NVRTC, checked for numerical
conformance against [`providers/cpu`](https://github.com/astorise/Magnetar-provider-CPU)'s
reference implementation on real hardware.

## Status

**Real baseline implementation**, not a template: `CudaProvider` discovers a
CUDA Device through `cudarc`'s dynamic-loading driver bindings, compiles and
runs all ten required-now kernels through NVRTC, and implements
`ProviderExecutionApi` (`CudaExecutor`) end-to-end through
`magnetar-runtime`'s real `Runtime`/`ProviderConformanceSuite` -- both the
`provider-core` and `provider-compute` conformance profiles pass on real
hardware (an RTX 3070 Ti Laptop GPU during development).

**Explicit non-goals for this baseline** (see the governing OpenSpec change
below for the reasoning):

- **No dynamic-library Provider ABI.** `CudaProvider` is built-in only.
- **No async Execution Stream extension.** Every Kernel call is synchronous:
  it launches on one CUDA stream and the call does not return until the
  stream has been synchronized.
- **No persistent cross-call device residency.** Tensors live in this
  Provider's own host-side storage (`CudaExecutor`, mirroring
  `ReferenceCpuExecutor`'s pattern) between separate Kernel invocations; each
  Kernel call uploads its inputs and downloads its output internally. Real
  `cuMemAlloc`/`cuMemFree` happen and Memory Manager residency is reported as
  genuine `MemoryPlacement::Device`, but two back-to-back kernels round-trip
  through host memory rather than chaining device-resident results, and
  `TensorValue::Opaque` (declining host materialization) is never returned.
- **No Device Memory Pool.** Direct per-buffer allocate/free, no pooling.
- **No multi-GPU placement, quantization, or flash/paged attention.**
- **f32, contiguous layout only.**

**Graceful unavailability**: `CudaProvider::new()` always constructs
successfully, even with no CUDA driver, no compatible GPU, or a
driver/runtime version mismatch. It reports zero Devices and
`ProviderHealth::Unavailable` rather than failing -- this is what keeps
`cargo test` passing on CI's GPU-less, CUDA-Toolkit-less
`submodule-integration` runner: `cudarc`'s `dynamic-loading` feature has no
link-time or build-time dependency on the CUDA driver/Toolkit at all (see
`Cargo.toml`'s comments for exactly which `cudarc` features and why), so
nothing can fail there except the runtime `dlopen` attempt this crate already
treats as a normal, expected outcome. The same test suite exercises the
opposite, hardware-present branch on any machine that does have a compatible
GPU, including the self-hosted `arc-gpu-magnetar` CI runner.

## Governing contract

[`cuda-provider`](https://github.com/astorise/Magnetar/blob/main/openspec/specs/cuda-provider/spec.md)
in the main Magnetar repository's OpenSpec capability set defines this
Provider's baseline requirements, introduced and implemented by the
`implement-cuda-provider-baseline` change (see that change's `design.md` for
the full rationale behind every decision above, including the two hardware
bugs found only by actually running on real GPUs: a `cudarc` CUDA-version
feature that produced a nonexistent NVRTC library filename, and NVRTC's
minimal preprocessor not predefining the `INFINITY` macro). The broader
`provider` and `provider-roadmap` capability specs govern the general
`ProviderExecutionApi`/`Provider` contract and this Provider's place in
Magnetar's post-baseline optimized-Provider roadmap, respectively.

## Relationship to magnetar-runtime

This crate depends only on `magnetar-runtime`'s public Provider/Device/
Kernel/Tensor/Memory contracts (`providers/cuda -> magnetar-runtime`, never
the reverse); `magnetar-runtime` compiles and tests cleanly without this
crate present. It is pinned into the main
[Magnetar](https://github.com/astorise/Magnetar) repository as a git
submodule at `providers/cuda`.

Like [`providers/cpu`](https://github.com/astorise/Magnetar-provider-CPU),
`HostTensor` is imported from `magnetar_runtime` rather than redefined here,
for the same reason documented in that crate's README: `ProviderExecutionApi`
is still typed directly against it as a provisional transport ahead of a
fully Resource-based rewrite.
