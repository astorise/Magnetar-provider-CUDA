## ADDED Requirements

### Requirement: CUDA Preparation Of Precompiled Images

The CUDA Provider SHALL expose kernel preparation for `nvidia:cubin` images whose target architecture matches the Device's compute capability. It SHALL load the image from memory through the driver, resolve the descriptor's entry symbol, and launch the prepared kernel on its stream with device-resident tensors. It SHALL NOT JIT-compile, substitute another architecture or read a file for an external artifact.

#### Scenario: A matching CUBIN

- **WHEN** an eligible `sm_86` CUBIN with a valid launch descriptor is prepared on an `sm_86` Device
- **THEN** the Provider returns a prepared kernel and an advertisement, and an invocation of that kernel runs on the Device

#### Scenario: A mismatched architecture

- **WHEN** an `sm_90` CUBIN is prepared on an `sm_86` Device
- **THEN** preparation fails with an incompatible-target error, and nothing is loaded

### Requirement: Prepared Kernels In Step Captures

A CUDA step capture that records a launch of a prepared kernel SHALL keep a reference to it for as long as the capture lives, and replay SHALL launch the same prepared kernel. Releasing the capture SHALL drop the reference.

#### Scenario: Replay after the pin is removed

- **WHEN** a decode capture recorded the prepared kernel and the pin is later removed
- **THEN** the existing capture replays with the prepared kernel until it is invalidated, and new captures use the newly selected kernel

### Requirement: Built-In Kernels Remain The Fallback

The CUDA Provider SHALL keep advertising its built-in kernels whatever external kernels are prepared, so that removing, failing or revoking a prepared kernel leaves every operator served.

#### Scenario: A prepared kernel is revoked

- **WHEN** the prepared `silu-mul` kernel's bundle is revoked
- **THEN** new work selects the built-in `silu-mul`, and generation continues
