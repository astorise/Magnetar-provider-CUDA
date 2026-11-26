# cuda-provider Specification

## Purpose
Defines the CUDA Provider's baseline contract: identity, graceful unavailability without compatible hardware, device discovery, Kernel advertisements and their correctness against Reference CPU, explicit data movement and Memory Manager integration (including caller-owned output pre-admission and Device-resident chaining between Kernels), synchronous execution, error categories, and conformance scope -- including native multi-head RoPE rotation, binding to a specific real GPU ordinal under a distinct Provider name so more than one instance can coexist in one Runtime (`add-real-second-gpu-cuda-provider`), and real peer-to-peer device-to-device movement between two such instances, gated by an explicit, never-assumed peer-capability query (`add-real-peer-to-peer-gpu-movement`).
## Requirements
### Requirement: CUDA RoPE Kernel Supports Native Multi-Head Rotation

The CUDA Provider's `rope` Kernel SHALL accept an optional `head_count`
parameter. When `head_count` is present and greater than `1`, the Kernel
SHALL divide each row into `head_count` equal-width blocks of
`head_width = cols / head_count` columns and, within each block
independently, rotate exactly the first `dimension` columns of that block
(`dimension` MAY be less than `head_width`, partial rotation) using the
same per-pair rotation formula already defined for the single-block case;
`position` is per-row, not per-head. Columns outside a rotated range SHALL
be copied through unchanged, never zeroed. `head_count` absent (or `1`)
SHALL reproduce the existing single-block rotation behavior exactly, with
`dimension` unconstrained by `head_count`'s divisibility rule.

Multi-head rotation SHALL execute as a single Kernel invocation regardless
of `head_count`; the Provider SHALL NOT require one invocation per head.

#### Scenario: Single-block call is unaffected

- **GIVEN** a `rope` Kernel invocation with no `head_count` attribute (or
  `head_count = 1`)
- **WHEN** the CUDA Provider dispatches it
- **THEN** the rotated output is bit-for-bit identical to this Provider's
  pre-existing single-block `rope` behavior

#### Scenario: Multi-head call with different Q and K head counts

- **GIVEN** two `rope` Kernel invocations over the same row count, one with
  `head_count` equal to a model's attention head count and one with
  `head_count` equal to a smaller grouped-query key/value head count
- **WHEN** the CUDA Provider dispatches each
- **THEN** each rotates its own row width using its own `head_width = cols
  / head_count`, independently of the other invocation's head count

#### Scenario: Partial RoPE within a head is supported

- **GIVEN** a `rope` Kernel invocation with `head_count > 1` and a rotation
  `dimension` strictly less than `head_width = cols / head_count`
- **WHEN** the CUDA Provider dispatches it
- **THEN** only the first `dimension` columns of each head's block are
  rotated
- **AND** the remaining columns of that block are copied from input to
  output unchanged, not zeroed

#### Scenario: Multi-head output matches Reference CPU

- **GIVEN** the same input tensor, `head_count`, rotation `dimension`, base,
  scale, and position offset
- **WHEN** CUDA's multi-head `rope` and Reference CPU's multi-head `rope`
  are each run once
- **THEN** their outputs agree within the declared numeric tolerance

### Requirement: CUDA Provider Baseline

Magnetar SHALL define a CUDA Provider as an optimized, GPU-executing Provider
implementing `magnetar:compute/run`.

#### Scenario: Register CUDA Provider

- **WHEN** Runtime starts with the CUDA Provider enabled
- **THEN** Runtime registers it as a built-in Provider if a compatible CUDA
  driver is discoverable, and registers it with zero Devices otherwise

### Requirement: CUDA Provider Is Not Correctness Baseline

CUDA Provider SHALL preserve Reference CPU's portable Operator semantics and
SHALL NOT be treated as the correctness oracle.

#### Scenario: Divergent output

- **GIVEN** CUDA Provider output differs from Reference CPU beyond declared
  tolerance
- **WHEN** conformance runs
- **THEN** CUDA Provider conformance fails and Reference CPU output remains
  authoritative

### Requirement: CUDA Provider Identity

CUDA Provider SHALL expose stable Provider identity through the same
`ProviderMetadata` contract other Providers use.

#### Scenario: Inspect provider

- **WHEN** Runtime lists Providers with CUDA Provider registered
- **THEN** stable redacted provider metadata is returned, with no native CUDA
  context, driver, or handle value included

### Requirement: Graceful Unavailability Without Compatible Hardware

CUDA Provider SHALL construct successfully and report Provider Health as
`unavailable` (not `failed`) when no compatible CUDA driver or Device is
discoverable, rather than failing Runtime initialization or the build.

#### Scenario: No driver present

- **GIVEN** the host has no CUDA driver installed
- **WHEN** CUDA Provider initializes
- **THEN** it registers with zero Devices and health `unavailable`
- **AND** Runtime initialization continues for other Providers

#### Scenario: CI runner without a GPU

- **GIVEN** `providers/cuda` is built and tested on a runner with no GPU and
  no CUDA Toolkit
- **WHEN** its test suite runs
- **THEN** the crate builds and its unit tests pass by asserting the
  unavailable-health path, without requiring real hardware

### Requirement: CUDA Device Discovery

CUDA Provider SHALL expose its primary compatible CUDA Device (device
ordinal 0) through Runtime-owned Device metadata, including name, compute
capability, and a memory pressure estimate. This baseline targets exactly
one `ModelInstance`/execution bound to exactly one selected GPU Device;
discovering or exposing more than one Device on a multi-GPU host is
explicitly out of scope until a real multi-GPU deployment need drives a
dedicated change (product decision, 2026-09-06). The already-archived
`multi-device-placement` capability (Runtime-owned placement, explicit
`MultiDevicePlacementPlan`) is the target contract for that future work --
this baseline does not implement it, and the two-phase rollout the product
decision named (independent Devices with per-request Runtime placement
first; model/tensor sharding and replication across Devices only if a real
need later emerges) should build on that existing contract rather than
inventing a new one.

#### Scenario: Single GPU available

- **GIVEN** one compatible CUDA-capable GPU is present
- **WHEN** Runtime lists Devices
- **THEN** the primary CUDA Device is visible through Runtime-owned metadata
- **AND** no raw CUDA context, stream, or device pointer is exposed

#### Scenario: Multiple GPUs available

- **GIVEN** more than one compatible CUDA-capable GPU is present on the host
- **WHEN** Runtime lists Devices
- **THEN** only the primary Device (ordinal 0) is visible through
  Runtime-owned metadata
- **AND** no requirement in this baseline depends on any other Device being
  discovered or selectable

### Requirement: CUDA Provider Layout and DType Support

CUDA Provider SHALL declare supported layouts and dtypes explicitly for this
baseline: contiguous layout and f32 only.

No silent dtype or layout conversion SHALL occur.

#### Scenario: Non-contiguous layout requested

- **GIVEN** an invocation requires a non-contiguous layout
- **WHEN** Kernel selection runs
- **THEN** the CUDA Kernel is not selected and Runtime reports a structured
  unsupported-layout error unless explicit conversion is planned

#### Scenario: Non-f32 dtype requested

- **GIVEN** a compute request uses a dtype other than f32
- **WHEN** dispatch is planned
- **THEN** Runtime rejects the CUDA candidate or inserts explicit conversion
  according to policy, without silent coercion

### Requirement: CUDA Provider Kernel Advertisements

CUDA Provider SHALL advertise only implemented Kernels, limited for this
baseline to the `operator-scope` required-now tier needed for the first
decoder path: embedding lookup, RMSNorm, matmul, RoPE (baseline mode only),
causal attention, softmax, SiLU, add, mul, and residual-add.

#### Scenario: Unimplemented kernel

- **GIVEN** flash attention is not implemented by CUDA Provider
- **WHEN** Runtime queries the Kernel Registry
- **THEN** no flash-attention CUDA Kernel is assumed unless explicitly
  advertised

### Requirement: CUDA Kernels Match Reference CPU Semantics

Each advertised CUDA kernel SHALL produce output matching Reference CPU's
output for the same portable Operator within declared numerical tolerance.

#### Scenario: Matmul fixture

- **GIVEN** a small matmul fixture used by Reference CPU conformance
- **WHEN** the same fixture runs through CUDA matmul
- **THEN** output matches Reference CPU's output within tolerance

### Requirement: CUDA Provider Explicit Data Movement

CUDA Provider SHALL require explicit host-to-device and device-to-host data
movement. Host memory inputs SHALL NOT be silently uploaded, and Device-
resident outputs SHALL NOT be silently downloaded. A Kernel invocation whose
input is already device-resident under the same `TensorResourceId` SHALL
NOT trigger a redundant device-to-host-then-host-to-device round trip; the
Provider SHALL reuse the existing device allocation directly.

#### Scenario: Host tensor passed to CUDA kernel

- **GIVEN** an input tensor resides in host memory
- **WHEN** a CUDA Kernel is considered for dispatch
- **THEN** Runtime requires an explicit upload step or rejects dispatch
  according to policy

#### Scenario: Device-resident tensor consumed by a second CUDA kernel

- **GIVEN** a CUDA Kernel's output tensor remains in this Provider's device
  allocation table
- **WHEN** a second CUDA Kernel is dispatched with that same
  `TensorResourceId` as an input
- **THEN** the Provider computes directly against the existing device
  allocation without downloading to host or re-uploading

### Requirement: CUDA Provider Memory Manager Integration

CUDA Provider SHALL allocate real device memory per Tensor Resource directly
(without implementing the Device Memory Pool's soft/hard reservation
contract in this baseline). Admission (byte size, placement, and
`MemoryAllocationOwner`) for an output resource SHALL be performed by
Runtime/caller before submission whenever the caller has already
pre-admitted that resource's final identity; the Provider SHALL then write
its output directly into that identity rather than admitting a second,
Provider-owned allocation for the same resource. When no pre-admission is
found for an output's resource id (e.g. a caller with no independent
resource-identity concept of its own, such as a conformance harness), the
Provider SHALL fall back to admitting and reporting the residency itself,
exactly as before. A Tensor Resource's device allocation SHALL persist
across separate Kernel invocations until this Provider's
`release`/`release_tensor` (or equivalent lifecycle) is called for it; it
SHALL NOT be represented only as host-resident bytes while the Memory
Manager records Device residency for it.

#### Scenario: Caller pre-admits the output resource

- **GIVEN** Runtime has already admitted a `MemoryAllocationRequest` for a
  Kernel's output resource id, under a caller-chosen `MemoryAllocationOwner`
- **WHEN** the CUDA Kernel invocation dispatches
- **THEN** the Provider writes its output directly under that same resource
  id
- **AND** the Provider does not admit a second, Provider-owned allocation
  for it

#### Scenario: No pre-admission found (fallback)

- **GIVEN** a Kernel invocation's output resource id has no existing
  `TensorResidency` record when dispatch begins
- **WHEN** the CUDA Kernel writes that output
- **THEN** the Provider admits it itself, Provider-owned, exactly as this
  baseline's original behavior

#### Scenario: Output consumed by a later invocation

- **GIVEN** a Kernel output tensor was written in a prior invocation
- **WHEN** a later Kernel invocation reads that same `TensorResourceId`
- **THEN** the Provider's device allocation for it is still present and
  valid, without having been silently dropped between calls

### Requirement: CUDA Provider Synchronous Execution

CUDA Provider SHALL execute each submitted operation synchronously to
completion for this baseline, without implementing the Execution Stream
asynchronous completion-token contract.

#### Scenario: Submit compute operation

- **GIVEN** a validated Compute Execution Plan is submitted to CUDA Provider
- **WHEN** the Provider executes it
- **THEN** the call returns only after the kernel has completed or failed,
  with no pending asynchronous completion token

### Requirement: CUDA Provider Error Categories

CUDA Provider failures SHALL use structured error categories mapped from
native CUDA driver/NVRTC errors, and SHALL NOT expose native error codes as
the stable contract.

#### Scenario: Out of device memory

- **GIVEN** a CUDA allocation fails due to insufficient device memory
- **WHEN** Runtime reports the failure
- **THEN** Runtime returns a stable out-of-memory or allocation-failure
  category with the native CUDA error attached only as diagnostics

### Requirement: CUDA Provider Does Not Expose Native Handles

CUDA Provider SHALL NOT expose CUDA contexts, streams, modules, device
pointers, or driver/NVRTC handles through any public Runtime API or
diagnostic surface.

#### Scenario: Diagnostic request

- **GIVEN** CUDA Kernel dispatch fails
- **WHEN** observability records it
- **THEN** Runtime emits redacted structured error metadata with no native
  CUDA pointer or handle value

### Requirement: CUDA Provider Conformance Scope

CUDA Provider SHALL pass the `provider-core`, `provider-compute`, and
`provider-data-movement` conformance profiles for this baseline.
`provider-dynamic-abi` SHALL NOT apply while CUDA Provider remains built-in.
`provider-cancellation` remains deferred to a future change introducing
genuinely asynchronous execution.

#### Scenario: Conformance run

- **GIVEN** CUDA Provider is registered with at least one Device
- **WHEN** the Provider Conformance Suite runs the `provider-core`,
  `provider-compute`, and `provider-data-movement` profiles
- **THEN** the suite reports pass or fail for those profiles specifically,
  without asserting `provider-dynamic-abi` applicability

### Requirement: CUDA Provider Health Reflects Kernel Compilation Failure

CUDA Provider SHALL report Provider Health as `degraded`, not `available`,
when a compatible Device was discovered but this Provider's own Kernels
failed to compile or load.

#### Scenario: Device found, kernel compilation fails

- **GIVEN** a compatible CUDA Device is discovered
- **AND** this Provider's NVRTC kernel compilation fails
- **WHEN** Runtime queries Provider Health
- **THEN** health is reported as `degraded`
- **AND** `execution_api()` reports unavailable, consistent with `degraded`
  rather than contradicting an `available` health report

### Requirement: CUDA Provider Out-Of-Device-Memory Kernel Errors

A CUDA Kernel failure caused by insufficient device memory SHALL be reported
through the Kernel-level out-of-device-memory error category, not a generic
Kernel execution failure.

#### Scenario: Allocation exceeds available device memory

- **GIVEN** a CUDA allocation request exceeds available device memory
- **WHEN** the failure is mapped to a Kernel error
- **THEN** Runtime reports the out-of-device-memory Kernel error category
  with the native CUDA error retained only as diagnostics

### Requirement: CUDA Provider Supports Multi-Step Decode

CUDA Provider SHALL support a generation request requiring more than one decode step. Historical KV concatenation across decode steps SHALL complete without downloading the historical KV tensor to host memory, and the resulting concatenated KV state SHALL remain Device-resident. Device memory held for a decode session's KV state SHALL NOT grow without bound across steps: at most one live allocation per layer, per K/V role, per KV identity (graph edge, pending, committed) SHALL exist at a time, with the previous step's allocation for that same identity released when superseded.

#### Scenario: Multi-token decode succeeds on real hardware

- **GIVEN** a generation request whose `max_tokens` requires more than one decode step, bound to CUDA Provider
- **WHEN** generation runs
- **THEN** it completes successfully and produces the requested number of real, decoded tokens, not an `Unsupported` rejection or an internal residency error

#### Scenario: Historical KV concatenation stays Device-resident

- **GIVEN** a decode step needs to concatenate this step's new K/V with the previous step's historical K/V, both held Device-resident by CUDA Provider
- **WHEN** the concatenation executes
- **THEN** it completes without CUDA Provider producing host-visible bytes for the historical KV tensor, and the concatenated result is itself Device-resident

#### Scenario: Device memory for KV state does not grow unboundedly across steps

- **GIVEN** a decode step's KV pending-write or commit targets a stable resource identity that already holds a previous step's allocation
- **WHEN** the new value is admitted under that same identity
- **THEN** the previous allocation is released and replaced, not left active alongside the new one, so a real multi-token decode (verified: 8 tokens on a synthetic bundle, 16 on the real public checkpoint) does not accumulate KV allocations proportional to step count

#### Scenario: Multi-token CUDA output matches Reference CPU

- **GIVEN** the same real ingested model, tokenizer, and prompt, decoded for the same number of tokens under greedy sampling
- **WHEN** compared between Reference CPU and CUDA Provider
- **THEN** the generated token sequence and decoded text are identical

### Requirement: CUDA Provider Offers Native Half-Precision Elementwise Compute

CUDA Provider SHALL offer real, on-device `F16`/`bfloat16` storage and elementwise `add`/`mul` compute, reachable both as a directly-callable primitive and through the Kernel Registry's standard selection/dispatch contract (`KernelSelectionRequest` -> `KernelRegistry::select` -> `KernelDispatchPlan::from_selection` -> `KernelDispatcher::revalidate` -> `ProviderExecutionApi::submit_kernel`/`complete_kernel`), via a distinct advertised Kernel identity per `add`/`mul` (`CUDA Provider Layout and DType Support`/`CUDA Provider Kernel Advertisements`'s "f32 only" declaration remains accurate for every other advertised Kernel). A half-precision buffer SHALL hold real 2-byte bit patterns of the declared format, not `f32` promoted and labeled, and host `f32` values SHALL convert to/from those bytes only at the explicit upload/download boundary, exactly as CUDA Provider's existing `f32` path already requires explicit data movement. A half-precision resource is NOT required to remain device-resident as half-precision bytes between separate Kernel invocations in this baseline -- unlike this Provider's `f32` device allocation table, which SHALL remain untouched by this requirement.

#### Scenario: F16 elementwise add matches the reference conversion model

- **GIVEN** two `f32` host tensors of equal shape
- **WHEN** both are uploaded as `F16`, added via the native half-precision `add` kernel, and downloaded
- **THEN** the result matches, element-wise, decoding-then-re-encoding each already-half-precision-rounded input through the same round-to-nearest-even algorithm and re-rounding the sum

#### Scenario: BF16 elementwise mul matches the reference conversion model

- **GIVEN** two `f32` host tensors of equal shape
- **WHEN** both are uploaded as `bfloat16`, multiplied via the native half-precision `mul` kernel, and downloaded
- **THEN** the result matches, element-wise, the same reference conversion model as the F16 scenario, using `bfloat16`'s own round-to-nearest-even algorithm

#### Scenario: Half-precision buffers of different dtypes cannot be combined

- **GIVEN** one `F16` half-precision buffer and one `bfloat16` half-precision buffer
- **WHEN** the native half-precision `add` or `mul` kernel is invoked with both
- **THEN** the call is rejected before any kernel launch, without silent coercion

#### Scenario: A Float16 resource selects the native half-precision Kernel through the standard dispatch contract

- **GIVEN** a `KernelSelectionRequest` for the `add` (or `mul`) Operator whose input/output resources declare `Float16` (or `BrainFloat16`) as their `ComputeDType`
- **WHEN** the Kernel Registry selects a candidate and the resulting invocation is dispatched through `ProviderExecutionApi::submit_kernel`/`complete_kernel`
- **THEN** the native half-precision Kernel is selected in preference to the `f32`-only one, and the dispatched result matches the same reference conversion model the direct-call scenarios above already establish

### Requirement: CUDA Provider Can Bind to a Specific Real Device Ordinal Under a Distinct Name

The CUDA Provider SHALL support construction bound to a caller-specified real device ordinal, registering under a caller-specified Provider name distinct from the default. The default constructor SHALL be behaviorally identical to binding ordinal 0 under the default name.

#### Scenario: Default construction is unchanged

- **GIVEN** the CUDA Provider's default constructor and its ordinal-0/default-name constructor
- **WHEN** both are constructed on the same host
- **THEN** they report identical availability, health, and Device identity

#### Scenario: A second real GPU ordinal is requested under its own name

- **GIVEN** a host with two or more real, compatible CUDA devices
- **WHEN** the CUDA Provider is constructed bound to ordinal 1 under a distinct Provider name
- **THEN** it reports `ProviderHealth::Available`
- **AND** its one reported Device has an identity distinct from ordinal 0's

#### Scenario: An out-of-range ordinal is requested

- **GIVEN** a host with fewer real CUDA devices than the requested ordinal plus one
- **WHEN** the CUDA Provider is constructed bound to that ordinal
- **THEN** construction succeeds
- **AND** it reports `ProviderHealth::Unavailable`, never a construction error or panic

### Requirement: Two Distinctly-Named CUDA Providers Register Into One Runtime Together

Two CUDA Provider instances bound to two different real device ordinals, each registered under its own distinct Provider name, SHALL both register successfully into the same Runtime.

#### Scenario: Two real GPUs registered together

- **GIVEN** two CUDA Provider instances bound to two different real device ordinals under two distinct names
- **WHEN** both are registered into the same Runtime
- **THEN** Runtime construction succeeds
- **AND** both Providers' Devices and Kernels are present in that Runtime

### Requirement: CUDA Provider Exposes a Real, Explicit Peer-Capability Query

The CUDA Provider SHALL expose a real, explicit query for whether one real GPU Device can directly access another real GPU Device's memory. This query SHALL NOT infer or assume the result from Device similarity (shared vendor, architecture, or memory capacity).

#### Scenario: Two real GPUs are queried for peer capability

- **GIVEN** two real, available CUDA Devices
- **WHEN** the peer-capability query is called for that pair
- **THEN** it returns the real, driver-reported capability, not an assumption derived from the two Devices' own metadata

### Requirement: CUDA Provider Supports Enabling Real Peer Access

The CUDA Provider SHALL support enabling one real GPU Device's context to directly access another real GPU Device's memory, and SHALL NOT perform this without a caller having first obtained a positive result from the peer-capability query for that same pair.

#### Scenario: Peer access is enabled after a positive capability query

- **GIVEN** a peer-capability query that returned true for a real Device pair
- **WHEN** peer access is enabled for that pair
- **THEN** the operation succeeds, including when called more than once for the same pair

### Requirement: CUDA Provider Supports a Real Cross-Device Copy That Never Touches Host Memory

The CUDA Provider SHALL support copying a Tensor Resource's current device allocation directly from one real GPU Device's own storage into another's, via a real device-to-device transfer, without host materialization at any point in the call path.

#### Scenario: A tensor is moved between two real GPUs

- **GIVEN** a tensor resource resident on one real GPU Device, and peer access already enabled between it and a second real GPU Device
- **WHEN** the cross-Device copy is invoked with the second Device's own executor as the destination
- **THEN** the tensor's bytes are readable back from the second Device
- **AND** they are bit-identical to the source

### Requirement: CUDA Kernels honour the layered models' optional forms

The CUDA Provider SHALL implement these optional forms, and SHALL declare each one's Kernel feature on the Kernels that honour it:
- the attention score scale;
- the scaled embedding;
- per-head and weightless RMSNorm;
- partial RoPE;
- GELU-gated routed experts;
- per-expert output scales.

An attention Kernel given no scale SHALL compute with `1/sqrt(head_dimension)`, as before.

#### Scenario: Scaled attention on CUDA
- **WHEN** an `attention` node sets `scale = 1.0` and runs on the CUDA Provider
- **THEN** the scores use that factor, and the output matches the Reference CPU's

### Requirement: CUDA Provider Runs Packed Q4_0 Weights

The CUDA Provider SHALL accept packed `Q4_0` weights. Downloading or expanding one SHALL give back the Runtime's dequantization exactly, and its matmuls SHALL match the Reference CPU's.

#### Scenario: Q4_0 decode matmul
- **WHEN** a decode matmul reads a packed `Q4_0` weight with int8 activations
- **THEN** its result matches the Reference CPU's on the rounded activations and the dequantized weight

### Requirement: CUDA Provider Reads Only The Looked-Up Rows Of A Packed Table

An embedding lookup from a packed table stored untransposed SHALL dequantize only the rows its ids name, in every packed format the CUDA Provider stores, and SHALL give the same values as those rows of the dequantized table. An id that is negative, not an integer or not below the table's row count SHALL fail the lookup.

#### Scenario: A Q6_K per-layer table
- **WHEN** a decode step looks up one token in a packed `Q6_K` `[262144, 256]` table
- **THEN** the lookup writes one 256-value row, not the dequantized table

#### Scenario: Every packed format
- **WHEN** a lookup reads rows of an untransposed packed table in any stored format
- **THEN** each row equals the same row of the table's dequantization, and an invalid id fails the lookup

### Requirement: CUDA TurboQuant KV Stores

The CUDA Provider SHALL store a TurboQuant KV store as codes, norms and offsets, SHALL set the offsets and quantize rows in `kv-write` within step capture, and SHALL compute decode attention on the codes with a rotated query and an inverse rotation of the accumulated values, matching the Reference CPU on the same history within the f32 conformance tolerance.

#### Scenario: Decode on codes
- **WHEN** a decode step attends over a `tq4` history
- **THEN** its output matches the Reference CPU's over the dequantized history within tolerance, with no stored row dequantized
