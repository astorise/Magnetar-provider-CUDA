//! `CudaExecutor`: this Provider's [`ProviderExecutionApi`] implementation.
//! Mirrors `providers/cpu`'s `ReferenceCpuExecutor` structure (same opaque
//! `TensorResourceId`-keyed storage, same submit/complete bookkeeping, same
//! Kernel-level `submit_kernel`/`complete_kernel` dispatch), but
//! `run_invocation` calls [`CudaKernels`] methods instead of pure CPU
//! functions, and Memory Manager admission reports genuine
//! [`MemoryPlacement::Device`] residency instead of `ProviderOwnedOpaque`.
//!
//! # Storage is device-resident between calls
//!
//! Tensors live in an in-process `Mutex<BTreeMap<TensorResourceId,
//! CudaDeviceBuffer>>` between Kernel invocations: a real device
//! allocation, not host bytes (`enable-device-resident-kernel-chaining`'s
//! device allocation table decision). [`CudaKernels`]'s own methods take
//! and return [`CudaDeviceBuffer`] directly and perform no implicit
//! upload/download; this executor uploads only in `write_tensor`/
//! `write_tensor_admitted` (a genuine host-to-device crossing, e.g. weight
//! materialization) and downloads only in `read_tensor` (a genuine
//! device-to-host crossing, e.g. final output extraction). Two
//! back-to-back Kernel invocations that both read/write through this same
//! table never round-trip through host memory.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use magnetar_runtime::affinity::{DeviceBinding, ProviderBinding};
use magnetar_runtime::compute::TensorResourceId;
use magnetar_runtime::device::DeviceId;
use magnetar_runtime::kernel::{
    KernelAdvertisement, KernelError, KernelInvocation, KernelObservation, KernelObservationKind,
    KernelResult, KernelResultStatus,
};
use magnetar_runtime::memory::{
    MemoryAllocationClass, MemoryAllocationId, MemoryAllocationOwner, MemoryAllocationRequest,
    MemoryError, MemoryManager, MemoryPlacement, TensorResidency,
};
use magnetar_runtime::operator::{OperatorAttributeValue, OperatorSpec};
use magnetar_runtime::provider::{ProviderExecutionApi, TensorValue, TensorValueAdmissionError};
use magnetar_runtime::scheduler::{
    ProviderCancellationOutcome, ProviderExecutionError, ProviderExecutionErrorCode,
    ProviderExecutionHandle, ProviderExecutionId, ProviderExecutionPhase, ProviderExecutionRequest,
    ProviderExecutionResult, ProviderExecutionStatus, ScheduledOperationId, SchedulingState,
};
use magnetar_runtime::{ExecutionPlanId, HostTensor};

use crate::error::CudaError;
use crate::kernels::{CudaDeviceBuffer, CudaKernels};
use crate::provider::CUDA_PROVIDER_NAME;

pub struct CudaExecutor {
    kernels: CudaKernels,
    device_id: DeviceId,
    storage: Mutex<BTreeMap<TensorResourceId, CudaDeviceBuffer>>,
    observations: Mutex<Vec<KernelObservation>>,
    submitted: Mutex<BTreeMap<ProviderExecutionId, ProviderExecutionRequest>>,
    kernel_executions: Mutex<BTreeMap<ProviderExecutionId, KernelResult>>,
    resource_allocations: Mutex<BTreeMap<TensorResourceId, MemoryAllocationId>>,
    next_execution_ordinal: AtomicU64,
}

impl CudaExecutor {
    pub fn new(kernels: CudaKernels, device_id: DeviceId) -> Self {
        Self {
            kernels,
            device_id,
            storage: Mutex::new(BTreeMap::new()),
            observations: Mutex::new(Vec::new()),
            submitted: Mutex::new(BTreeMap::new()),
            kernel_executions: Mutex::new(BTreeMap::new()),
            resource_allocations: Mutex::new(BTreeMap::new()),
            next_execution_ordinal: AtomicU64::new(0),
        }
    }

    fn provider_binding(&self) -> ProviderBinding {
        ProviderBinding::new(CUDA_PROVIDER_NAME)
    }

    fn device_binding(&self) -> DeviceBinding {
        DeviceBinding::new(self.device_id.clone())
    }

    /// Uploads `tensor` into a fresh device allocation and stores it under
    /// `id`, replacing whatever this Provider previously held for `id` (its
    /// prior device allocation, if any, is freed when the replaced
    /// [`CudaDeviceBuffer`] drops). A real host-to-device crossing --
    /// callers that already have a same-Provider device-resident value
    /// should never reach this; see [`Self::write_tensor_value`].
    pub fn write_tensor(&self, id: TensorResourceId, tensor: HostTensor) -> Result<(), CudaError> {
        let buffer = self.kernels.upload(&tensor)?;
        self.storage.lock().unwrap().insert(id, buffer);
        Ok(())
    }

    /// Downloads `id`'s current device allocation to host-visible bytes. A
    /// real device-to-host crossing -- only reached when host bytes are
    /// genuinely requested (`TensorValue::into_host` at an actual
    /// materialization boundary), not on every Kernel invocation.
    pub fn read_tensor(&self, id: &TensorResourceId) -> Option<HostTensor> {
        let storage = self.storage.lock().unwrap();
        let buffer = storage.get(id)?;
        self.kernels.download(buffer).ok()
    }

    pub fn release_tensor(&self, id: &TensorResourceId) -> bool {
        self.storage.lock().unwrap().remove(id).is_some()
    }

    pub fn release_admitted_tensor(
        &self,
        memory: &mut MemoryManager,
        id: &TensorResourceId,
    ) -> bool {
        if let Some(allocation) = self.resource_allocations.lock().unwrap().remove(id) {
            let _ = memory.release(allocation);
        }
        self.storage.lock().unwrap().remove(id).is_some()
    }

    pub fn write_tensor_admitted(
        &self,
        memory: &mut MemoryManager,
        id: TensorResourceId,
        tensor: HostTensor,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), MemoryError> {
        let byte_size = tensor.data.len() as u64 * std::mem::size_of::<f32>() as u64;
        let allocation = memory.allocate(MemoryAllocationRequest::new(
            class,
            byte_size,
            MemoryPlacement::Device(self.device_binding()),
            owner,
        ))?;
        // Logical admission succeeded; now physically realize it. A real
        // device allocation can fail here (device OOM, driver error) in a
        // way host-resident storage never could -- roll the logical
        // admission back rather than leave the Memory Manager's ledger
        // claiming residency this Provider does not actually have.
        let buffer = match self.kernels.upload(&tensor) {
            Ok(buffer) => buffer,
            Err(error) => {
                let _ = memory.release(allocation.id);
                return Err(MemoryError::AllocationDenied {
                    reason: format!("CUDA device upload failed: {error}"),
                });
            }
        };
        let previous = self
            .resource_allocations
            .lock()
            .unwrap()
            .insert(id.clone(), allocation.id);
        if let Some(previous) = previous {
            let _ = memory.release(previous);
        }
        self.storage.lock().unwrap().insert(id, buffer);
        Ok(())
    }

    /// `Some(TensorValue::Opaque)` when `id` already has a live device
    /// allocation -- never downloads. Callers that need host bytes call
    /// [`Self::read_tensor`] (via `TensorValue::into_host`) explicitly.
    pub fn read_tensor_value(&self, id: &TensorResourceId) -> Option<TensorValue> {
        if self.storage.lock().unwrap().contains_key(id) {
            Some(TensorValue::Opaque)
        } else {
            None
        }
    }

    fn opaque_passthrough_error(&self, id: &TensorResourceId) -> ProviderExecutionError {
        ProviderExecutionError::new(
            ProviderExecutionErrorCode::MaterializationFailed,
            ProviderExecutionPhase::Submit,
            self.provider_binding(),
            Some(self.device_binding()),
            format!("no existing device allocation for opaque resource '{id}'"),
        )
    }

    pub fn write_tensor_value(
        &self,
        id: TensorResourceId,
        value: TensorValue,
    ) -> Result<(), ProviderExecutionError> {
        match value {
            TensorValue::Host(tensor) => self.write_tensor(id, tensor).map_err(|error| {
                ProviderExecutionError::new(
                    ProviderExecutionErrorCode::MaterializationFailed,
                    ProviderExecutionPhase::Submit,
                    self.provider_binding(),
                    Some(self.device_binding()),
                    format!("CUDA device upload failed: {error}"),
                )
            }),
            // This Provider already owns `id`'s data under its existing
            // device allocation (the same-Provider/Device passthrough
            // decision in `enable-device-resident-kernel-chaining`) -- an
            // `Opaque` write for a resource this table does not hold is
            // not a portable cross-Provider handle, it is a genuine
            // integrity failure.
            TensorValue::Opaque => {
                if self.storage.lock().unwrap().contains_key(&id) {
                    Ok(())
                } else {
                    Err(self.opaque_passthrough_error(&id))
                }
            }
        }
    }

    pub fn write_tensor_value_admitted(
        &self,
        memory: &mut MemoryManager,
        id: TensorResourceId,
        value: TensorValue,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), TensorValueAdmissionError> {
        match value {
            TensorValue::Host(tensor) => self
                .write_tensor_admitted(memory, id, tensor, class, owner)
                .map_err(TensorValueAdmissionError::Memory),
            TensorValue::Opaque => {
                if self.storage.lock().unwrap().contains_key(&id) {
                    Ok(())
                } else {
                    Err(TensorValueAdmissionError::Provider(
                        self.opaque_passthrough_error(&id),
                    ))
                }
            }
        }
    }

    pub fn observations(&self) -> Vec<KernelObservation> {
        self.observations.lock().unwrap().clone()
    }

    fn observe(&self, observation: KernelObservation) {
        self.observations.lock().unwrap().push(observation);
    }

    fn next_provider_execution_id(&self, label: &str) -> ProviderExecutionId {
        let ordinal = self.next_execution_ordinal.fetch_add(1, Ordering::Relaxed);
        ProviderExecutionId::new(format!("{CUDA_PROVIDER_NAME}:{label}:{ordinal}"))
    }

    fn input_resource_id(
        invocation: &KernelInvocation,
        index: usize,
    ) -> Result<&TensorResourceId, KernelError> {
        let resource =
            invocation
                .inputs
                .get(index)
                .ok_or_else(|| KernelError::KernelExecutionFailed {
                    reason: format!("missing input at index {index}"),
                })?;
        Ok(&resource.resource.id)
    }

    fn store_output(
        &self,
        invocation: &KernelInvocation,
        index: usize,
        buffer: CudaDeviceBuffer,
    ) -> Result<magnetar_runtime::compute::TensorResourceDescriptor, KernelError> {
        let resource =
            invocation
                .outputs
                .get(index)
                .ok_or_else(|| KernelError::KernelExecutionFailed {
                    reason: format!("missing output at index {index}"),
                })?;
        self.storage
            .lock()
            .unwrap()
            .insert(resource.resource.id.clone(), buffer);
        Ok(resource.resource.clone())
    }

    fn attribute_float(
        attributes: &BTreeMap<String, OperatorAttributeValue>,
        key: &str,
        default: f32,
    ) -> f32 {
        match attributes.get(key) {
            Some(OperatorAttributeValue::Float(value)) => *value as f32,
            _ => default,
        }
    }

    fn attribute_integer(
        attributes: &BTreeMap<String, OperatorAttributeValue>,
        key: &str,
    ) -> Option<u64> {
        match attributes.get(key) {
            Some(OperatorAttributeValue::Integer(value)) if *value >= 0 => Some(*value as u64),
            _ => None,
        }
    }

    fn attribute_bool(
        attributes: &BTreeMap<String, OperatorAttributeValue>,
        key: &str,
        default: bool,
    ) -> bool {
        match attributes.get(key) {
            Some(OperatorAttributeValue::Boolean(value)) => *value,
            _ => default,
        }
    }

    /// Executes one Runtime-created [`KernelInvocation`] against this
    /// Provider's advertised Kernel, dispatching to the matching
    /// [`CudaKernels`] method and recording its output in opaque host
    /// storage. Mirrors `providers/cpu::ReferenceCpuExecutor::execute_invocation`.
    pub fn execute_invocation(
        &self,
        advertisement: &KernelAdvertisement,
        operator: &OperatorSpec,
        invocation: &KernelInvocation,
    ) -> KernelResult {
        self.observe(
            KernelObservation::new(KernelObservationKind::KernelDispatchStarted)
                .with_kernel(&invocation.kernel)
                .with_invocation(invocation.id.clone()),
        );
        if invocation.deadline_millis == Some(0) {
            let error = KernelError::KernelTimeout;
            self.observe(
                KernelObservation::new(KernelObservationKind::KernelTimeout)
                    .with_kernel(&invocation.kernel)
                    .with_invocation(invocation.id.clone()),
            );
            return KernelResult::failure(invocation.id.clone(), error);
        }
        match advertisement
            .validate_invocation(operator, invocation)
            .and_then(|()| self.run_invocation(invocation))
        {
            Ok(result) => {
                self.observe(
                    KernelObservation::new(KernelObservationKind::KernelDispatchCompleted)
                        .with_kernel(&invocation.kernel)
                        .with_invocation(invocation.id.clone()),
                );
                result
            }
            Err(error) => {
                self.observe(
                    KernelObservation::new(KernelObservationKind::KernelDispatchFailed)
                        .with_kernel(&invocation.kernel)
                        .with_invocation(invocation.id.clone())
                        .with_redacted_metadata("error", error.id()),
                );
                KernelResult::failure(invocation.id.clone(), error)
            }
        }
    }

    fn run_invocation(&self, invocation: &KernelInvocation) -> Result<KernelResult, KernelError> {
        let name = invocation.kernel.name.as_str();
        let mut result = KernelResult::success(invocation.id.clone());
        // One storage lock for every input this invocation reads: each
        // `get(index)` borrows directly from the device allocation table
        // instead of downloading to host and re-uploading a fresh
        // per-invocation copy. The kernel call below returns a freshly
        // allocated, fully owned `CudaDeviceBuffer` that does not borrow
        // from `storage`, so the lock can drop as soon as the match arm
        // finishes.
        let output = {
            let storage = self.storage.lock().unwrap();
            let get = |index: usize| -> Result<&CudaDeviceBuffer, KernelError> {
                let id = Self::input_resource_id(invocation, index)?;
                storage
                    .get(id)
                    .ok_or_else(|| KernelError::KernelExecutionFailed {
                        reason: format!("no materialized data for input resource {id}"),
                    })
            };
            match name {
                "matmul" => {
                    let a = get(0)?;
                    let b = get(1)?;
                    let transpose_a =
                        Self::attribute_bool(&invocation.attributes, "transpose_a", false);
                    let transpose_b =
                        Self::attribute_bool(&invocation.attributes, "transpose_b", false);
                    self.kernels
                        .matmul(a, b, transpose_a, transpose_b)
                        .map_err(KernelError::from)?
                }
                "embedding" => {
                    let table = get(0)?;
                    let ids = get(1)?;
                    self.kernels
                        .embedding_lookup(table, ids)
                        .map_err(KernelError::from)?
                }
                "rmsnorm" => {
                    let input = get(0)?;
                    let weight = get(1)?;
                    let epsilon = Self::attribute_float(&invocation.attributes, "epsilon", 1e-6);
                    self.kernels
                        .rmsnorm(input, weight, epsilon)
                        .map_err(KernelError::from)?
                }
                "rope" => {
                    let input = get(0)?;
                    let base = Self::attribute_float(&invocation.attributes, "base", 10000.0);
                    let scale = Self::attribute_float(&invocation.attributes, "scale", 1.0);
                    let dimension = Self::attribute_integer(&invocation.attributes, "dimension")
                        .ok_or_else(|| KernelError::KernelAttributeUnsupported {
                            attribute: "dimension".into(),
                        })?;
                    if let Some(OperatorAttributeValue::String(mode)) =
                        invocation.attributes.get("position_mode")
                        && mode != "sequential"
                    {
                        return Err(KernelError::KernelAttributeUnsupported {
                            attribute: format!("position_mode '{mode}' is not implemented"),
                        });
                    }
                    let position_offset = match invocation.attributes.get("position_offset") {
                        None => 0,
                        Some(OperatorAttributeValue::Integer(offset)) if *offset >= 0 => {
                            *offset as u64
                        }
                        Some(OperatorAttributeValue::Integer(offset)) => {
                            return Err(KernelError::KernelAttributeUnsupported {
                                attribute: format!("position_offset {offset} must not be negative"),
                            });
                        }
                        Some(_) => {
                            return Err(KernelError::KernelAttributeUnsupported {
                                attribute: "position_offset must be an integer".into(),
                            });
                        }
                    };
                    self.kernels
                        .rope(input, base, scale, dimension, position_offset)
                        .map_err(KernelError::from)?
                }
                "attention" => {
                    let q = get(0)?;
                    let k = get(1)?;
                    let v = get(2)?;
                    let head_count = Self::attribute_integer(&invocation.attributes, "head_count")
                        .ok_or_else(|| KernelError::KernelAttributeUnsupported {
                            attribute: "head_count".into(),
                        })?;
                    let head_dimension =
                        Self::attribute_integer(&invocation.attributes, "head_dimension")
                            .ok_or_else(|| KernelError::KernelAttributeUnsupported {
                                attribute: "head_dimension".into(),
                            })?;
                    let kv_head_count =
                        Self::attribute_integer(&invocation.attributes, "kv_head_count");
                    let window_size =
                        Self::attribute_integer(&invocation.attributes, "window_size");
                    let causal = Self::attribute_bool(&invocation.attributes, "causal", false);
                    if let Some(OperatorAttributeValue::String(mask_kind)) =
                        invocation.attributes.get("attention_mask_kind")
                    {
                        let expected_causal = match mask_kind.as_str() {
                            "causal" => true,
                            "bidirectional" => false,
                            other => {
                                return Err(KernelError::KernelAttributeUnsupported {
                                    attribute: format!(
                                        "attention_mask_kind '{other}' is not implemented"
                                    ),
                                });
                            }
                        };
                        if expected_causal != causal {
                            return Err(KernelError::KernelAttributeUnsupported {
                                attribute: format!(
                                    "attention_mask_kind '{mask_kind}' is inconsistent with causal={causal}"
                                ),
                            });
                        }
                    }
                    self.kernels
                        .attention(
                            q,
                            k,
                            v,
                            head_count,
                            head_dimension,
                            kv_head_count,
                            window_size,
                            causal,
                        )
                        .map_err(KernelError::from)?
                }
                "softmax" => {
                    let input = get(0)?;
                    self.kernels
                        .softmax_rows(input)
                        .map_err(KernelError::from)?
                }
                "silu" => self.kernels.silu(get(0)?).map_err(KernelError::from)?,
                "add" => {
                    let a = get(0)?;
                    let b = get(1)?;
                    self.kernels.add(a, b).map_err(KernelError::from)?
                }
                "mul" => {
                    let a = get(0)?;
                    let b = get(1)?;
                    self.kernels.mul(a, b).map_err(KernelError::from)?
                }
                "residual-add" => {
                    let input = get(0)?;
                    let residual = get(1)?;
                    self.kernels
                        .residual_add(input, residual)
                        .map_err(KernelError::from)?
                }
                other => {
                    return Err(KernelError::KernelNotFound {
                        kernel: other.into(),
                    });
                }
            }
        };
        let descriptor = self.store_output(invocation, 0, output)?;
        result
            .output_readiness
            .insert(descriptor.id.to_string(), true);
        result.updated_resources.push(descriptor);
        Ok(result)
    }

    /// Admits every output's byte size through `memory` (with genuine
    /// [`MemoryPlacement::Device`] residency, unlike `ReferenceCpuProvider`'s
    /// `ProviderOwnedOpaque`) before executing, then records
    /// [`TensorResidency`] for whatever the invocation actually produced.
    /// Mirrors `providers/cpu::ReferenceCpuExecutor::execute_invocation_with_memory_manager`.
    pub fn execute_invocation_with_memory_manager(
        &self,
        advertisement: &KernelAdvertisement,
        operator: &OperatorSpec,
        invocation: &KernelInvocation,
        memory: &mut MemoryManager,
    ) -> KernelResult {
        let provider = self.provider_binding();
        let mut admitted: Vec<(TensorResourceId, MemoryAllocationId)> =
            Vec::with_capacity(invocation.outputs.len());
        // Resource ids a caller already admitted before submission
        // (`MemoryManager::admit_kernel_output`,
        // `unify-provider-output-admission-and-residency`) -- this
        // Provider never self-admitted them (`resource_allocations` has no
        // entry), yet `memory` already has a residency record for them.
        // Honored as-is: no second, Provider-owned allocation, no
        // overwriting the caller's own residency/placement/ownership
        // choice.
        let mut preadmitted: std::collections::BTreeSet<TensorResourceId> =
            std::collections::BTreeSet::new();
        for output in &invocation.outputs {
            let resource = &output.resource;
            let self_admitted_before = self
                .resource_allocations
                .lock()
                .unwrap()
                .contains_key(&resource.id);
            if !self_admitted_before && memory.tensor_residency(&resource.id).is_some() {
                preadmitted.insert(resource.id.clone());
                continue;
            }
            let byte_size = match resource.descriptor.byte_size() {
                Ok(byte_size) => byte_size,
                Err(error) => {
                    for (_, allocation_id) in &admitted {
                        let _ = memory.release(*allocation_id);
                    }
                    return KernelResult::failure(
                        invocation.id.clone(),
                        KernelError::KernelExecutionFailed {
                            reason: format!(
                                "cannot admit output {}: invalid tensor descriptor ({error:?})",
                                resource.id
                            ),
                        },
                    );
                }
            };
            let request = MemoryAllocationRequest::new(
                MemoryAllocationClass::Tensor,
                byte_size,
                MemoryPlacement::Device(self.device_binding()),
                MemoryAllocationOwner::Provider(provider.clone()),
            )
            .with_affinity(resource.affinity.clone());
            match memory.allocate(request) {
                Ok(allocation) => admitted.push((resource.id.clone(), allocation.id)),
                Err(error) => {
                    self.observe(
                        KernelObservation::new(
                            KernelObservationKind::KernelMemoryFeasibilityFailed,
                        )
                        .with_kernel(&invocation.kernel)
                        .with_invocation(invocation.id.clone()),
                    );
                    for (_, allocation_id) in &admitted {
                        let _ = memory.release(*allocation_id);
                    }
                    return KernelResult::failure(
                        invocation.id.clone(),
                        KernelError::KernelExecutionFailed {
                            reason: format!(
                                "memory admission denied for output {}: {error:?}",
                                resource.id
                            ),
                        },
                    );
                }
            }
        }

        let result = self.execute_invocation(advertisement, operator, invocation);
        if result.status != KernelResultStatus::Succeeded {
            for (_, allocation_id) in &admitted {
                let _ = memory.release(*allocation_id);
            }
            return result;
        }
        for resource in &result.updated_resources {
            if preadmitted.contains(&resource.id) {
                // The caller's own residency record is already
                // authoritative -- do not overwrite it with a
                // Provider-chosen placement/affinity, and this Provider
                // does not own its lifecycle (no entry to add to
                // `resource_allocations`).
                continue;
            }
            let Some((_, allocation_id)) = admitted
                .iter()
                .find(|(resource_id, _)| *resource_id == resource.id)
                .map(|(resource_id, allocation_id)| (resource_id.clone(), *allocation_id))
            else {
                continue;
            };
            let _ = memory.record_tensor_residency(
                TensorResidency::new(
                    resource.id.clone(),
                    MemoryPlacement::Device(self.device_binding()),
                    resource.affinity.clone(),
                )
                .with_allocation(allocation_id),
            );
            // Replace-and-release whatever this same resource id's previous
            // Kernel invocation admitted, mirroring `write_tensor_admitted`'s
            // own pattern (`enable-device-resident-kernel-chaining`'s
            // discovered leak fix): a Kernel-internal output id (e.g.
            // `{operation_id}.out`) is stable across every generation step
            // that dispatches the same graph node, so without this, every
            // single node dispatch would admit a fresh `MemoryAllocationId`
            // that nothing ever releases -- an unbounded Memory Manager
            // ledger leak over a long-running session, even though the
            // underlying `CudaDeviceBuffer` itself does not physically leak
            // (`storage`'s `BTreeMap::insert` already drops the prior entry).
            let previous = self
                .resource_allocations
                .lock()
                .unwrap()
                .insert(resource.id.clone(), allocation_id);
            if let Some(previous) = previous {
                let _ = memory.release(previous);
            }
        }
        result
    }

    pub fn submit_kernel_invocation(
        &self,
        advertisement: &KernelAdvertisement,
        operator: &OperatorSpec,
        invocation: &KernelInvocation,
        memory: &mut MemoryManager,
    ) -> ProviderExecutionHandle {
        let provider = self.provider_binding();
        let execution_id = self.next_provider_execution_id(invocation.id.as_str());
        let handle = ProviderExecutionHandle {
            id: execution_id.clone(),
            operation: ScheduledOperationId::new(
                self.next_execution_ordinal.load(Ordering::Relaxed),
            ),
            plan: ExecutionPlanId::new(invocation.id.as_str().to_string()),
            provider,
            device: Some(self.device_binding()),
        };
        let result = self.execute_invocation_with_memory_manager(
            advertisement,
            operator,
            invocation,
            memory,
        );
        self.kernel_executions
            .lock()
            .unwrap()
            .insert(execution_id, result);
        handle
    }

    pub fn complete_kernel_invocation(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<KernelResult, ProviderExecutionError> {
        self.kernel_executions
            .lock()
            .unwrap()
            .remove(&handle.id)
            .ok_or_else(|| {
                ProviderExecutionError::new(
                    ProviderExecutionErrorCode::ExecutionFailed,
                    ProviderExecutionPhase::Complete,
                    handle.provider.clone(),
                    handle.device.clone(),
                    "no Kernel execution is associated with this handle: it was never \
                     submitted through submit_kernel_invocation, or has already been \
                     completed once",
                )
            })
    }
}

impl ProviderExecutionApi for CudaExecutor {
    fn submit(
        &self,
        request: ProviderExecutionRequest,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        let handle = ProviderExecutionHandle::new(
            request.operation,
            request.plan.id.clone(),
            request.provider.clone(),
            request.device.clone(),
        );
        self.submitted
            .lock()
            .unwrap()
            .insert(handle.id.clone(), request);
        Ok(handle)
    }

    fn status(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionStatus, ProviderExecutionError> {
        if !self.submitted.lock().unwrap().contains_key(&handle.id) {
            return Err(ProviderExecutionError::new(
                ProviderExecutionErrorCode::ExecutionFailed,
                ProviderExecutionPhase::Observe,
                handle.provider.clone(),
                handle.device.clone(),
                "no submission is associated with this handle: it was never submitted, \
                 or has already been completed and released",
            ));
        }
        Ok(ProviderExecutionStatus::new(
            handle.clone(),
            SchedulingState::Completed,
        ))
    }

    fn cancel(
        &self,
        _handle: &ProviderExecutionHandle,
    ) -> Result<ProviderCancellationOutcome, ProviderExecutionError> {
        // Synchronous per-call execution (design.md's "no async streams
        // yet" decision): by the time a caller could ask to cancel, the
        // kernel launch this baseline issued has already been submitted to
        // the CUDA stream and this call has already returned its result.
        Ok(ProviderCancellationOutcome::Unsupported)
    }

    fn complete(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<ProviderExecutionResult, ProviderExecutionError> {
        self.submitted
            .lock()
            .unwrap()
            .remove(&handle.id)
            .ok_or_else(|| {
                ProviderExecutionError::new(
                    ProviderExecutionErrorCode::ExecutionFailed,
                    ProviderExecutionPhase::Complete,
                    handle.provider.clone(),
                    handle.device.clone(),
                    "no submission is associated with this handle: it was never \
                     submitted, or has already been completed once",
                )
            })?;
        Ok(ProviderExecutionResult::completed(
            handle.clone(),
            Vec::new(),
        ))
    }

    fn release(&self, handle: ProviderExecutionHandle) -> Result<(), ProviderExecutionError> {
        self.submitted.lock().unwrap().remove(&handle.id);
        Ok(())
    }

    fn submit_kernel(
        &self,
        advertisement: &KernelAdvertisement,
        operator: &OperatorSpec,
        invocation: &KernelInvocation,
        memory: &mut MemoryManager,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        Ok(self.submit_kernel_invocation(advertisement, operator, invocation, memory))
    }

    fn complete_kernel(
        &self,
        handle: &ProviderExecutionHandle,
    ) -> Result<KernelResult, ProviderExecutionError> {
        self.complete_kernel_invocation(handle)
    }

    fn write_tensor(
        &self,
        id: TensorResourceId,
        tensor: HostTensor,
    ) -> Result<(), ProviderExecutionError> {
        CudaExecutor::write_tensor(self, id, tensor).map_err(|error| {
            ProviderExecutionError::new(
                ProviderExecutionErrorCode::MaterializationFailed,
                ProviderExecutionPhase::Submit,
                self.provider_binding(),
                Some(self.device_binding()),
                format!("CUDA device upload failed: {error}"),
            )
        })
    }

    fn read_tensor(&self, id: &TensorResourceId) -> Option<HostTensor> {
        CudaExecutor::read_tensor(self, id)
    }

    fn release_tensor(&self, id: &TensorResourceId) -> Result<bool, ProviderExecutionError> {
        Ok(CudaExecutor::release_tensor(self, id))
    }

    fn release_admitted_tensor(
        &self,
        memory: &mut MemoryManager,
        id: &TensorResourceId,
    ) -> Result<bool, ProviderExecutionError> {
        Ok(CudaExecutor::release_admitted_tensor(self, memory, id))
    }

    fn write_tensor_admitted(
        &self,
        memory: &mut MemoryManager,
        resource_id: TensorResourceId,
        tensor: HostTensor,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), MemoryError> {
        CudaExecutor::write_tensor_admitted(self, memory, resource_id, tensor, class, owner)
    }

    fn read_tensor_value(&self, id: &TensorResourceId) -> Option<TensorValue> {
        CudaExecutor::read_tensor_value(self, id)
    }

    fn write_tensor_value(
        &self,
        id: TensorResourceId,
        value: TensorValue,
    ) -> Result<(), ProviderExecutionError> {
        CudaExecutor::write_tensor_value(self, id, value)
    }

    fn write_tensor_value_admitted(
        &self,
        memory: &mut MemoryManager,
        resource_id: TensorResourceId,
        value: TensorValue,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<(), TensorValueAdmissionError> {
        CudaExecutor::write_tensor_value_admitted(self, memory, resource_id, value, class, owner)
    }

    fn observations(&self) -> Vec<KernelObservation> {
        CudaExecutor::observations(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::CudaProvider;

    fn executor_or_skip() -> Option<CudaExecutor> {
        let provider = CudaProvider::new();
        let context = provider.context()?;
        let kernels = CudaKernels::compile_and_load(&context).expect(
            "kernel compilation must succeed on a machine that already passed device discovery",
        );
        Some(CudaExecutor::new(kernels, DeviceId::new("test-device")))
    }

    #[test]
    fn repeated_write_release_cycles_do_not_grow_storage_unboundedly() {
        let Some(executor) = executor_or_skip() else {
            return;
        };
        // Simulates a multi-step generation run: each step writes a fresh
        // device-resident resource and releases the previous step's, the
        // same pattern `first_native_runtime.rs` follows for per-step
        // intermediate resources (`enable-device-resident-kernel-chaining`
        // task 2.9).
        for step in 0..50u32 {
            let id = TensorResourceId::new(format!("step-{step}"));
            let tensor = HostTensor::new([2, 2], [1.0, 2.0, 3.0, 4.0]).unwrap();
            executor
                .write_tensor(id, tensor)
                .expect("upload must succeed");
            if step >= 1 {
                let previous = TensorResourceId::new(format!("step-{}", step - 1));
                assert!(
                    executor.release_tensor(&previous),
                    "previous step's resource must still be present to release"
                );
            }
            assert!(
                executor.storage.lock().unwrap().len() <= 2,
                "storage must not grow past the live window at step {step}"
            );
        }
    }
}
