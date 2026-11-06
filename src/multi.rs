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
//! Kernel, packed weight and admission rule is the single-GPU one. Step
//! capture is not offered yet: a step spanning GPUs runs uncaptured.

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
    Provider, ProviderError, ProviderExecutionApi, ProviderMetadata, ProviderRegistry, TensorValue,
    TensorValueAdmissionError,
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

/// The execution side of [`CudaMultiProvider`].
pub struct CudaMultiExecutor {
    provider_name: String,
    members: Vec<Arc<CudaExecutor>>,
    placement: Mutex<Option<LayerPlacement>>,
    residency: Mutex<Residency>,
    completed: Mutex<BTreeMap<ProviderExecutionId, KernelResult>>,
    next_execution: std::sync::atomic::AtomicU64,
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

    /// Copies every input of `invocation` that `member` lacks from its
    /// owner, after the owner's stream has written it.
    fn bring_inputs(
        &self,
        invocation: &KernelInvocation,
        member: usize,
        memory: &mut MemoryManager,
    ) -> Result<(), ProviderExecutionError> {
        for input in &invocation.inputs {
            let id = &input.resource.id;
            let (owner, version) = {
                let residency = self.residency.lock().unwrap();
                if residency.current_on(id, member) {
                    continue;
                }
                match residency.resources.get(id) {
                    Some(copies) => (copies.owner, copies.version),
                    // Unknown here: let the member report it.
                    None => continue,
                }
            };
            self.members[owner].synchronize().map_err(|error| {
                self.error(
                    ProviderExecutionPhase::Submit,
                    format!("waiting for GPU {owner} before moving '{id}': {error}"),
                )
            })?;
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
                .map_err(|error| {
                    self.error(
                        ProviderExecutionPhase::Submit,
                        format!("moving '{id}' from GPU {owner} to GPU {member}: {error:?}"),
                    )
                })?;
            if let Some(copies) = self.residency.lock().unwrap().resources.get_mut(id) {
                copies.copies.insert(member, version);
            }
        }
        Ok(())
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
        self.bring_inputs(invocation, member, memory)?;
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
