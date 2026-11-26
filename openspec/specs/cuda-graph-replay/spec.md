# cuda-graph-replay Specification

## Purpose
This specification defines bounded CUDA graph replay: Provider-owned capture and replay of admissible work, the memory a capture reserves, its invalidation when a resource, stream, version or Device changes, and the promotion gate of the profile.
## Requirements
### Requirement: C04 Bounded CUDA graph replay

CUDA capture/replay SHALL remain Provider-owned, require admissible shapes and resource lifetimes, reserve capture memory and invalidate replay on resource, stream, version or Device changes.

#### Scenario: C04 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a supported graph is captured and replayed
- **THEN** launch/synchronization reduction is measured with preparation and retained memory costs

#### Scenario: C04 Ineligible or failed execution

- **WHEN** a captured address lease expires or replay preparation fails
- **THEN** replay is invalidated and execution uses a qualified uncaptured path without stale-address access

### Requirement: cuda-graph-replay promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
