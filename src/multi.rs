//! A CUDA Provider spanning several local GPUs
//! (`enforce-local-multigpu-and-expert-placement` G05, staged layer
//! segments first).
//!
//! The Runtime sees one Provider exposing every member GPU as a Device. It
//! places the Model Instance's decoder layers across them
//! ([`magnetar_runtime::layer_placement::LayerPlacement`]) and hands the plan
//! over through [`ProviderExecutionApi::apply_layer_placement`] before any
//! weight is written. This Provider then:
//!
//! - writes each weight on its layer's GPU (weights never move);
//! - runs each Kernel on the GPU holding its weights, or, for a Kernel with
//!   no weight input, on the GPU the step is currently running on;
//! - copies every input another GPU produced into the executing GPU first,
//!   device to device (`cuMemcpyPeerAsync`), after waiting for the producing
//!   GPU's stream -- the only movement, and an explicit one;
//! - tracks a version per resource, so a copy is made again only after its
//!   source changed, and a resource's stale copies are dropped when it is
//!   produced anew.
//!
//! Each member is an ordinary [`CudaExecutor`] bound to its own GPU, so every
//! Kernel, packed weight and admission rule is the single-GPU one. A step
//! is captured as one CUDA graph per contiguous run of a member's work,
//! with the boundary crossings between them recorded and repeated in place
//! at replay, so each graph's captured addresses stay valid.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::CudaContext;
use magnetar_runtime::HostTensor;
use magnetar_runtime::affinity::{ProviderBinding, ProviderHealth};
use magnetar_runtime::compute::TensorResourceId;
use magnetar_runtime::device::{Device, DeviceDescriptor, DeviceId};
use magnetar_runtime::kernel::{
    KernelAdvertisement, KernelInvocation, KernelObservation, KernelResult,
};
use magnetar_runtime::layer_placement::LayerPlacement;
use magnetar_runtime::memory::{
    MemoryAllocationClass, MemoryAllocationId, MemoryAllocationOwner, MemoryError, MemoryManager,
};
use magnetar_runtime::operator::OperatorSpec;
use magnetar_runtime::packed_weight::{PackedWeight, PackedWeightWrite};
use magnetar_runtime::provider::{
    Provider, ProviderError, ProviderExecutionApi, ProviderMetadata, ProviderRegistry,
    StepCaptureId, StepCaptureRequest, StepReplayError, TensorValue, TensorValueAdmissionError,
};
use magnetar_runtime::scheduler::{
    ProviderCancellationOutcome, ProviderExecutionError, ProviderExecutionErrorCode,
    ProviderExecutionHandle, ProviderExecutionId, ProviderExecutionPhase, ProviderExecutionRequest,
    ProviderExecutionResult, ProviderExecutionStatus,
};

use crate::advertisements::cuda_kernel_advertisements;
use crate::executor::CudaExecutor;
use crate::provider::CudaProvider;

/// Where each resource lives across the members.
#[derive(Default)]
struct Residency {
    /// The member each weight was written to.
    weights: BTreeMap<TensorResourceId, usize>,
    /// Every other resource: its current version, the member holding that
    /// version (its owner) and the version each member's copy holds.
    resources: BTreeMap<TensorResourceId, ResourceCopies>,
    /// The member the latest Kernel ran on.
    current: usize,
}

struct ResourceCopies {
    version: u64,
    owner: usize,
    copies: BTreeMap<usize, u64>,
}

impl Residency {
    /// Records that `member` produced a new version of `id`; returns the
    /// members whose copies just became stale.
    fn produced(&mut self, id: &TensorResourceId, member: usize) -> Vec<usize> {
        let entry = self
            .resources
            .entry(id.clone())
            .or_insert_with(|| ResourceCopies {
                version: 0,
                owner: member,
                copies: BTreeMap::new(),
            });
        entry.version += 1;
        entry.owner = member;
        let stale = entry
            .copies
            .keys()
            .copied()
            .filter(|holder| *holder != member)
            .collect();
        entry.copies = BTreeMap::from([(member, entry.version)]);
        stale
    }

    fn owner(&self, id: &TensorResourceId) -> Option<usize> {
        self.weights
            .get(id)
            .copied()
            .or_else(|| self.resources.get(id).map(|copies| copies.owner))
    }

    /// Whether `member` holds `id`'s current version.
    fn current_on(&self, id: &TensorResourceId, member: usize) -> bool {
        if let Some(weight_member) = self.weights.get(id) {
            return *weight_member == member;
        }
        self.resources
            .get(id)
            .is_some_and(|copies| copies.copies.get(&member) == Some(&copies.version))
    }

    fn holders(&self, id: &TensorResourceId) -> Vec<usize> {
        if let Some(member) = self.weights.get(id) {
            return vec![*member];
        }
        self.resources
            .get(id)
            .map(|copies| copies.copies.keys().copied().collect())
            .unwrap_or_default()
    }
}

/// One step of a multi-GPU capture, replayed in order.
enum CapturedOp {
    /// A member's own captured graph.
    Graph { member: usize, id: StepCaptureId },
    /// `id` copied in place from `from` to `to`, after `from`'s stream.
    Copy {
        id: TensorResourceId,
        from: usize,
        to: usize,
    },
}

/// The step capture in progress: the operations so far and the member
/// whose own capture is open.
struct Recording {
    request: StepCaptureRequest,
    ops: Vec<CapturedOp>,
    open: Option<usize>,
}

#[derive(Default)]
struct CaptureState {
    recording: Option<Recording>,
    captures: BTreeMap<StepCaptureId, Vec<CapturedOp>>,
    next: u64,
}

/// The execution side of [`CudaMultiProvider`].
pub struct CudaMultiExecutor {
    provider_name: String,
    members: Vec<Arc<CudaExecutor>>,
    placement: Mutex<Option<LayerPlacement>>,
    residency: Mutex<Residency>,
    completed: Mutex<BTreeMap<ProviderExecutionId, KernelResult>>,
    next_execution: std::sync::atomic::AtomicU64,
    capture: Mutex<CaptureState>,
}

impl CudaMultiExecutor {
    fn error(
        &self,
        phase: ProviderExecutionPhase,
        detail: impl Into<String>,
    ) -> ProviderExecutionError {
        ProviderExecutionError::new(
            ProviderExecutionErrorCode::ExecutionFailed,
            phase,
            ProviderBinding::new(self.provider_name.clone()),
            None,
            detail,
        )
    }

    fn member_for_device(&self, device: &DeviceId) -> Option<usize> {
        self.members
            .iter()
            .position(|member| member.device_id() == device)
    }

    /// The member a weight goes to under the current placement (the first
    /// member without one).
    fn weight_member(&self, id: &TensorResourceId) -> usize {
        self.placement
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|placement| self.member_for_device(placement.device_for_weight(id.as_str())))
            .unwrap_or(0)
    }

    /// The member a write of the non-weight `id` goes to: wherever it
    /// lives, else the member the step is running on.
    fn write_member(&self, id: &TensorResourceId) -> usize {
        let residency = self.residency.lock().unwrap();
        residency.owner(id).unwrap_or(residency.current)
    }

    /// The member `invocation` runs on: the one holding its weights, else
    /// the current one if it holds an input, else the owner of its first
    /// input.
    fn execution_member(&self, invocation: &KernelInvocation) -> usize {
        let residency = self.residency.lock().unwrap();
        let ids = || invocation.inputs.iter().map(|input| &input.resource.id);
        if let Some(member) = ids().find_map(|id| residency.weights.get(id)) {
            return *member;
        }
        if ids().any(|id| residency.current_on(id, residency.current)) {
            return residency.current;
        }
        ids()
            .find_map(|id| residency.owner(id))
            .unwrap_or(residency.current)
    }

    /// The inputs of `invocation` that `member` lacks, with their owner and
    /// current version.
    fn missing_inputs(
        &self,
        invocation: &KernelInvocation,
        member: usize,
    ) -> Vec<(TensorResourceId, usize, u64)> {
        let residency = self.residency.lock().unwrap();
        invocation
            .inputs
            .iter()
            .map(|input| &input.resource.id)
            .filter(|id| !residency.current_on(id, member))
            .filter_map(|id| {
                residency
                    .resources
                    .get(id)
                    .map(|copies| (id.clone(), copies.owner, copies.version))
            })
            .collect()
    }

    /// Copies `missing` into `member` from each owner, after the owner's
    /// stream has written it: in place when `member` already has a buffer
    /// of the same size, else into a new one.
    fn bring_inputs(
        &self,
        missing: &[(TensorResourceId, usize, u64)],
        member: usize,
        memory: &mut MemoryManager,
    ) -> Result<(), ProviderExecutionError> {
        for (id, owner, version) in missing {
            let owner = *owner;
            let moving = |error: String| {
                let streams: Vec<String> = self
                    .members
                    .iter()
                    .enumerate()
                    .map(|(index, member)| format!("GPU {index} {}", member.capture_status()))
                    .collect();
                self.error(
                    ProviderExecutionPhase::Submit,
                    format!(
                        "moving '{id}' from GPU {owner} to GPU {member}: {error} (streams: {})",
                        streams.join(", ")
                    ),
                )
            };
            self.members[owner]
                .synchronize()
                .map_err(|error| moving(error.to_string()))?;
            let in_place = self.members[member]
                .copy_from_peer_in_place(&self.members[owner], id)
                .map_err(|error| moving(error.to_string()))?;
            if !in_place {
                self.members[member]
                    .copy_tensor_from_peer_admitted(
                        memory,
                        &self.members[owner],
                        id,
                        id.clone(),
                        MemoryAllocationClass::Tensor,
                        MemoryAllocationOwner::Provider(ProviderBinding::new(
                            self.provider_name.clone(),
                        )),
                    )
                    .map_err(|error| moving(format!("{error:?}")))?;
            }
            if let Some(copies) = self.residency.lock().unwrap().resources.get_mut(id) {
                copies.copies.insert(member, *version);
            }
        }
        Ok(())
    }

    /// While capturing, prepares `member` to run an invocation needing
    /// `missing`: the open member capture is closed (its graph recorded)
    /// whenever the member changes or an input must cross first, the
    /// crossings are made and recorded, and `member`'s own capture opened.
    /// `false` when no capture is in progress.
    fn prepare_captured_invocation(
        &self,
        member: usize,
        missing: &[(TensorResourceId, usize, u64)],
        memory: &mut MemoryManager,
    ) -> Result<bool, ProviderExecutionError> {
        let mut capture = self.capture.lock().unwrap();
        let Some(recording) = capture.recording.as_mut() else {
            return Ok(false);
        };
        if recording.open != Some(member) || !missing.is_empty() {
            if let Some(open) = recording.open.take() {
                let id = self.members[open].end_step_capture()?;
                recording.ops.push(CapturedOp::Graph { member: open, id });
            }
            self.bring_inputs(missing, member, memory)?;
            for (id, from, _) in missing {
                recording.ops.push(CapturedOp::Copy {
                    id: id.clone(),
                    from: *from,
                    to: member,
                });
            }
            self.members[member].begin_step_capture(&recording.request)?;
            recording.open = Some(member);
        }
        Ok(true)
    }

    /// Releases every member capture `ops` holds.
    fn release_ops(&self, ops: &[CapturedOp]) {
        for op in ops {
            if let CapturedOp::Graph { member, id } = op {
                self.members[*member].release_step_capture(id);
            }
        }
    }

    /// Records `id` as produced by `member`, freeing the stale copies other
    /// members held.
    fn record_produced(
        &self,
        id: &TensorResourceId,
        member: usize,
        memory: Option<&mut MemoryManager>,
    ) {
        let stale = self.residency.lock().unwrap().produced(id, member);
        match memory {
            Some(memory) => {
                for holder in stale {
                    let _ = ProviderExecutionApi::release_admitted_tensor(
                        &*self.members[holder],
                        memory,
                        id,
                    );
                }
            }
            None => {
                for holder in stale {
                    let _ = ProviderExecutionApi::release_tensor(&*self.members[holder], id);
                }
            }
        }
    }

    fn owner_member(&self, id: &TensorResourceId) -> Option<&Arc<CudaExecutor>> {
        let owner = self.residency.lock().unwrap().owner(id)?;
        Some(&self.members[owner])
    }
}

impl ProviderExecutionApi for CudaMultiExecutor {
    fn submit(
        &self,
        request: ProviderExecutionRequest,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        self.members[0].submit(request)
    }

    fn status(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionStatus, ProviderExecutionError> {
        self.members[0].status(handle)
    }

    fn cancel(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderCancellationOutcome, ProviderExecutionError> {
        self.members[0].cancel(handle)
    }

    fn complete(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionResult, ProviderExecutionError> {
        self.members[0].complete(handle)
    }

    fn release(&self, handle: ProviderExecutionHandle) -> Result<(), ProviderExecutionError> {
        self.members[0].release(handle)
    }

    fn submit_kernel(
        &self,
        advertisement: &KernelAdvertisement,
        operator: &OperatorSpec,
        invocation: &KernelInvocation,
        memory: &mut MemoryManager,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        let member = self.execution_member(invocation);
        let missing = self.missing_inputs(invocation, member);
        if !self.prepare_captured_invocation(member, &missing, memory)? {
            self.bring_inputs(&missing, member, memory)?;
        }
        let member_handle =
            self.members[member].submit_kernel(advertisement, operator, invocation, memory)?;
        let result = self.members[member].complete_kernel(&member_handle)?;
        if result.status == magnetar_runtime::kernel::KernelResultStatus::Succeeded {
            for resource in &result.updated_resources {
                self.record_produced(&resource.id, member, Some(memory));
            }
            for output in &invocation.outputs {
                self.record_produced(&output.resource.id, member, Some(memory));
            }
        }
        self.residency.lock().unwrap().current = member;
        let ordinal = self
            .next_execution
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let id = ProviderExecutionId::new(format!(
            "{}:multi:{ordinal}:{}",
            self.provider_name,
            invocation.id.as_str()
        ));
        self.completed.lock().unwrap().insert(id.clone(), result);
        Ok(ProviderExecutionHandle {
            id,
            ..member_handle
        })
    }

    fn complete_kernel(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<KernelResult, ProviderExecutionError> {
        self.completed
            .lock()
            .unwrap()
            .remove(&handle.id)
            .ok_or_else(|| {
                self.error(
                    ProviderExecutionPhase::Complete,
                    "no Kernel execution is associated with this handle",
                )
            })
    }

    fn apply_layer_placement(
        &self,
        placement: &LayerPlacement,
    ) -> Result<(), ProviderExecutionError> {
        for segment in placement.segments() {
            if self.member_for_device(&segment.device).is_none() {
                return Err(self.error(
                    ProviderExecutionPhase::Submit,
                    format!(
                        "layer placement names Device '{}', not one of this Provider's",
                        segment.device
                    ),
                ));
            }
        }
        *self.placement.lock().unwrap() = Some(placement.clone());
        Ok(())
    }

    fn write_tensor(
        &self,
        id: TensorResourceId,
        tensor: HostTensor,
    ) -> Result<(), ProviderExecutionError> {
        let member = self.write_member(&id);
        ProviderExecutionApi::write_tensor(&*self.members[member], id.clone(), tensor)?;
        self.record_produced(&id, member, None);
        Ok(())
    }

    fn write_weight_tensor(
        &self,
        id: TensorResourceId,
        tensor: HostTensor,
    ) -> Result<(), ProviderExecutionError> {
        let member = self.weight_member(&id);
        ProviderExecutionApi::write_weight_tensor(&*self.members[member], id.clone(), tensor)?;
        self.residency.lock().unwrap().weights.insert(id, member);
        Ok(())
    }

    fn write_packed_weight_tensor(
        &self,
        id: TensorResourceId,
        weight: PackedWeight,
    ) -> Result<PackedWeightWrite, ProviderExecutionError> {
        let member = self.weight_member(&id);
        let written = ProviderExecutionApi::write_packed_weight_tensor(
            &*self.members[member],
            id.clone(),
            weight,
        )?;
        if written == PackedWeightWrite::Stored {
            self.residency.lock().unwrap().weights.insert(id, member);
        }
        Ok(written)
    }

    fn read_tensor(&self, id: &TensorResourceId) -> Option<HostTensor> {
        ProviderExecutionApi::read_tensor(&**self.owner_member(id)?, id)
    }

    fn read_tensor_rows(
        &self,
        id: &TensorResourceId,
        first_row: u64,
        rows: u64,
    ) -> Option<HostTensor> {
        self.owner_member(id)?.read_tensor_rows(id, first_row, rows)
    }

    fn release_tensor(&self, id: &TensorResourceId) -> Result<bool, ProviderExecutionError> {
        let holders = {
            let mut residency = self.residency.lock().unwrap();
            let holders = residency.holders(id);
            residency.resources.remove(id);
            residency.weights.remove(id);
            holders
        };
        let mut released = false;
        for holder in holders {
            released |= ProviderExecutionApi::release_tensor(&*self.members[holder], id)?;
        }
        Ok(released)
    }

    fn release_admitted_tensor(
        &self,
        memory: &mut MemoryManager,
        id: &TensorResourceId,
    ) -> Result<bool, ProviderExecutionError> {
        let holders = {
            let mut residency = self.residency.lock().unwrap();
            let holders = residency.holders(id);
            residency.resources.remove(id);
            residency.weights.remove(id);
            holders
        };
        let mut released = false;
        for holder in holders {
            released |=
                ProviderExecutionApi::release_admitted_tensor(&*self.members[holder], memory, id)?;
        }
        Ok(released)
    }

    fn write_tensor_admitted(
        &self,
        memory: &mut MemoryManager,
        resource_id: TensorResourceId,
        tensor: HostTensor,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), MemoryError> {
        let member = self.write_member(&resource_id);
        self.members[member].write_tensor_admitted(
            memory,
            resource_id.clone(),
            tensor,
            class,
            owner,
        )?;
        self.record_produced(&resource_id, member, Some(memory));
        Ok(())
    }

    fn read_tensor_value(&self, id: &TensorResourceId) -> Option<TensorValue> {
        self.owner_member(id)?.read_tensor_value(id)
    }

    fn resident_shape(&self, id: &TensorResourceId) -> Option<Vec<u64>> {
        self.owner_member(id)?.resident_shape(id)
    }

    fn write_tensor_value(
        &self,
        id: TensorResourceId,
        value: TensorValue,
    ) -> Result<(), ProviderExecutionError> {
        let member = self.write_member(&id);
        ProviderExecutionApi::write_tensor_value(&*self.members[member], id.clone(), value)?;
        self.record_produced(&id, member, None);
        Ok(())
    }

    fn write_zeroed_tensor(
        &self,
        id: &TensorResourceId,
        shape: &[u64],
    ) -> Result<bool, ProviderExecutionError> {
        let member = self.write_member(id);
        let written = ProviderExecutionApi::write_zeroed_tensor(&*self.members[member], id, shape)?;
        if written {
            self.record_produced(id, member, None);
        }
        Ok(written)
    }

    fn write_tensor_value_admitted(
        &self,
        memory: &mut MemoryManager,
        resource_id: TensorResourceId,
        value: TensorValue,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), TensorValueAdmissionError> {
        let member = self.write_member(&resource_id);
        self.members[member].write_tensor_value_admitted(
            memory,
            resource_id.clone(),
            value,
            class,
            owner,
        )?;
        self.record_produced(&resource_id, member, Some(memory));
        Ok(())
    }

    fn copy_tensor_admitted(
        &self,
        memory: &mut MemoryManager,
        from: &TensorResourceId,
        to: TensorResourceId,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<MemoryAllocationId, TensorValueAdmissionError> {
        let member = self.residency.lock().unwrap().owner(from).unwrap_or(0);
        let allocation =
            self.members[member].copy_tensor_admitted(memory, from, to.clone(), class, owner)?;
        self.record_produced(&to, member, Some(memory));
        Ok(allocation)
    }

    fn allocate_workspace(
        &self,
        memory: &mut MemoryManager,
        size_bytes: u64,
    ) -> Result<MemoryAllocationId, MemoryError> {
        let current = self.residency.lock().unwrap().current;
        self.members[current].allocate_workspace(memory, size_bytes)
    }

    fn observations(&self) -> Vec<KernelObservation> {
        self.members
            .iter()
            .flat_map(|member| ProviderExecutionApi::observations(&**member))
            .collect()
    }

    /// Opt-in while it is qualified: `MAGNETAR_CUDA_MULTI_CAPTURE=1`.
    fn supports_step_capture(&self) -> bool {
        std::env::var("MAGNETAR_CUDA_MULTI_CAPTURE").as_deref() == Ok("1")
    }

    /// Opens a multi-GPU capture: each member's own capture opens with the
    /// first invocation it runs, and closes when the step moves to another
    /// member or an input must cross GPUs first (the crossing is recorded
    /// between the two graphs).
    fn begin_step_capture(
        &self,
        request: &StepCaptureRequest,
    ) -> Result<(), ProviderExecutionError> {
        let mut capture = self.capture.lock().unwrap();
        if capture.recording.is_some() {
            return Err(self.error(
                ProviderExecutionPhase::Submit,
                "a step capture is already in progress",
            ));
        }
        capture.recording = Some(Recording {
            request: request.clone(),
            ops: Vec::new(),
            open: None,
        });
        Ok(())
    }

    fn end_step_capture(&self) -> Result<StepCaptureId, ProviderExecutionError> {
        let mut capture = self.capture.lock().unwrap();
        let Some(mut recording) = capture.recording.take() else {
            return Err(self.error(
                ProviderExecutionPhase::Complete,
                "no step capture is in progress",
            ));
        };
        if let Some(open) = recording.open.take() {
            match self.members[open].end_step_capture() {
                Ok(id) => recording.ops.push(CapturedOp::Graph { member: open, id }),
                Err(error) => {
                    self.release_ops(&recording.ops);
                    return Err(error);
                }
            }
        }
        let id = StepCaptureId::new(format!(
            "{}:multi-capture:{}",
            self.provider_name, capture.next
        ));
        capture.next += 1;
        capture.captures.insert(id.clone(), recording.ops);
        Ok(id)
    }

    fn abort_step_capture(&self) {
        let mut capture = self.capture.lock().unwrap();
        if let Some(recording) = capture.recording.take() {
            if let Some(open) = recording.open {
                self.members[open].abort_step_capture();
            }
            self.release_ops(&recording.ops);
        }
    }

    /// Replays each member graph in order with the same `offset`, each
    /// input going to the first graph of the member holding it, and
    /// repeats each recorded crossing in place after its source's stream.
    fn replay_step(
        &self,
        id: &StepCaptureId,
        offset: u64,
        inputs: Vec<(TensorResourceId, HostTensor)>,
    ) -> Result<(), StepReplayError> {
        let capture = self.capture.lock().unwrap();
        let ops = capture
            .captures
            .get(id)
            .ok_or_else(|| StepReplayError::Invalidated {
                reason: format!("no step capture '{id}'"),
            })?;
        let mut pending: Vec<(Option<usize>, TensorResourceId, HostTensor)> = {
            let residency = self.residency.lock().unwrap();
            inputs
                .into_iter()
                .map(|(resource, tensor)| (residency.owner(&resource), resource, tensor))
                .collect()
        };
        let failed = |detail: String| {
            StepReplayError::Failed(self.error(ProviderExecutionPhase::Submit, detail))
        };
        for op in ops {
            match op {
                CapturedOp::Graph { member, id } => {
                    let (mine, rest): (Vec<_>, Vec<_>) = pending
                        .into_iter()
                        .partition(|(owner, _, _)| owner.is_none_or(|owner| owner == *member));
                    pending = rest;
                    let mine = mine
                        .into_iter()
                        .map(|(_, resource, tensor)| (resource, tensor))
                        .collect();
                    self.members[*member].replay_step(id, offset, mine)?;
                }
                CapturedOp::Copy { id, from, to } => {
                    self.members[*from]
                        .synchronize()
                        .map_err(|error| failed(error.to_string()))?;
                    let copied = self.members[*to]
                        .copy_from_peer_in_place(&self.members[*from], id)
                        .map_err(|error| failed(error.to_string()))?;
                    if !copied {
                        return Err(StepReplayError::Invalidated {
                            reason: format!(
                                "'{id}' no longer has its captured storage on GPU {to}"
                            ),
                        });
                    }
                }
            }
        }
        if let Some((_, resource, _)) = pending.first() {
            return Err(StepReplayError::Invalidated {
                reason: format!("input '{resource}' was not read by the captured step"),
            });
        }
        Ok(())
    }

    fn release_step_capture(&self, id: &StepCaptureId) {
        if let Some(ops) = self.capture.lock().unwrap().captures.remove(id) {
            self.release_ops(&ops);
        }
    }

    fn select_greedy_column(
        &self,
        logits: &TensorResourceId,
        row: u64,
        excluded: &[std::ops::Range<u64>],
        temperature: f32,
    ) -> Option<u64> {
        self.owner_member(logits)?
            .select_greedy_column(logits, row, excluded, temperature)
    }
}

/// A Provider spanning several local GPUs; see the module documentation.
pub struct CudaMultiProvider {
    metadata: ProviderMetadata,
    devices: Vec<Arc<DeviceDescriptor>>,
    executor: Option<Arc<CudaMultiExecutor>>,
}

impl CudaMultiProvider {
    /// Binds the GPUs of `ordinals`, enabling peer access between every pair
    /// that supports it. Unavailable unless every ordinal is.
    pub fn for_devices(ordinals: &[usize], provider_name: impl Into<String>) -> Self {
        let provider_name = provider_name.into();
        let members: Vec<CudaProvider> = ordinals
            .iter()
            .map(|ordinal| CudaProvider::for_device(*ordinal, provider_name.clone()))
            .collect();
        let devices = members
            .iter()
            .filter_map(|member| member.device_descriptor())
            .collect::<Vec<_>>();
        let executors = members
            .iter()
            .filter_map(CudaProvider::executor)
            .collect::<Vec<_>>();
        let contexts = members
            .iter()
            .filter_map(CudaProvider::context)
            .collect::<Vec<Arc<CudaContext>>>();
        let available = !ordinals.is_empty() && executors.len() == ordinals.len();
        if available {
            for from in &contexts {
                for to in &contexts {
                    if Arc::ptr_eq(from, to) {
                        continue;
                    }
                    if crate::peer::device_can_access_peer(from, to).unwrap_or(false) {
                        let _ = crate::peer::enable_peer_access(from, to);
                    }
                }
            }
        }
        Self {
            metadata: crate::provider::cuda_provider_metadata_for(&provider_name),
            devices: if available { devices } else { Vec::new() },
            executor: available.then(|| {
                Arc::new(CudaMultiExecutor {
                    provider_name,
                    members: executors,
                    placement: Mutex::new(None),
                    residency: Mutex::new(Residency::default()),
                    completed: Mutex::new(BTreeMap::new()),
                    next_execution: std::sync::atomic::AtomicU64::new(0),
                    capture: Mutex::new(CaptureState::default()),
                })
            }),
        }
    }

    pub fn is_available(&self) -> bool {
        self.executor.is_some()
    }

    pub fn executor(&self) -> Option<Arc<CudaMultiExecutor>> {
        self.executor.clone()
    }
}

impl Provider for CudaMultiProvider {
    fn metadata(&self) -> ProviderMetadata {
        self.metadata.clone()
    }

    fn register(&self, _registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
        Ok(())
    }

    fn health(&self) -> ProviderHealth {
        if self.executor.is_some() {
            ProviderHealth::Available
        } else {
            ProviderHealth::Unavailable
        }
    }

    fn devices(&self) -> Vec<Arc<dyn Device>> {
        self.devices
            .iter()
            .map(|device| device.clone() as Arc<dyn Device>)
            .collect()
    }

    /// Kernels are advertised on the first GPU: the Runtime selects them
    /// there, and this Provider runs each invocation where its weights are.
    fn kernel_advertisements(&self) -> Vec<KernelAdvertisement> {
        match self.devices.first() {
            Some(device) => cuda_kernel_advertisements(&device.metadata.id, &self.metadata.name),
            None => Vec::new(),
        }
    }

    fn initialize(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn shutdown(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn execution_api(&self) -> Option<Arc<dyn ProviderExecutionApi>> {
        self.executor
            .clone()
            .map(|executor| executor as Arc<dyn ProviderExecutionApi>)
    }
}
