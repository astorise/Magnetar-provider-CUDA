# magnetar-provider-cuda

## Purpose

A CUDA Provider for the [Magnetar](https://github.com/astorise/Magnetar)
local AI Runtime: a GPU-accelerated Kernel execution backend implementing
`ProviderExecutionApi` and Magnetar's Provider contract for NVIDIA Devices.
Intended to prove Magnetar's Provider abstraction genuinely supports a
second, real, non-host-visible execution backend -- exercising the
device-resident Tensor Resource requirements
(`device-resident-resource` in the main repository's OpenSpec capability
set) that [`providers/cpu`](https://github.com/astorise/Magnetar-provider-CPU),
being host-visible, cannot exercise on its own.

## Status

**Empty template.** This crate is currently a bare `cargo new --lib`
scaffold: no CUDA bindings, no Kernel implementations, no Device
enumeration exist here yet. Unlike [`providers/cpu`](https://github.com/astorise/Magnetar-provider-CPU),
there is no interim, working implementation anywhere in the Magnetar
workspace to extract from -- this is genuinely unstarted work, not yet
scheduled against any OpenSpec change. `magnetar-runtime`'s
`define-provider-prepared-kernel-execution-contract` change added the
Provider-agnostic `TensorValue` contract (`Host`/`Opaque`) this Provider
would need to implement to answer `Opaque` for device-resident tensors,
and proved the contract works against a synthetic, non-CUDA test double
(`DeviceResidentOnlyExecutor`) -- but a synthetic double is not a
substitute for this crate actually existing.

## Governing contract

No dedicated `cuda-provider` capability spec exists yet in the main
[Magnetar](https://github.com/astorise/Magnetar) repository's
`openspec/specs/`. The general `provider` and `device-resident-resource`
capabilities govern the contract this crate would need to implement; a
CUDA-specific spec (Device enumeration, driver API surface, memory pool
behavior, peer access) would be scoped when real work here begins.

## Relationship to magnetar-runtime

Once implemented, this crate would depend only on `magnetar-runtime`'s
public Provider/Device/Kernel/Tensor contracts
(`providers/cuda -> magnetar-runtime`, never the reverse), registering
itself as an optional, hardware-gated Provider -- Magnetar's
`submodule-integration` CI job already anticipates this: CPU is mandatory
there, CUDA is expected to be optional and hardware-gated. It is pinned
into the main [Magnetar](https://github.com/astorise/Magnetar) repository
as a git submodule at `providers/cuda`.
