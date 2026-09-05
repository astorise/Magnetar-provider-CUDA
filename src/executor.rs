//! `CudaExecutor`: this Provider's [`ProviderExecutionApi`] implementation.
//! Mirrors `providers/cpu`'s `ReferenceCpuExecutor` structure closely (same
//! opaque `TensorResourceId -> HostTensor` storage, same submit/complete
//! bookkeeping, same Kernel-level `submit_kernel`/`complete_kernel`
//! dispatch), but `run_invocation` calls [`CudaKernels`] methods instead of
//! pure CPU functions, and Memory Manager admission reports genuine
//! [`MemoryPlacement::Device`] residency instead of `ProviderOwnedOpaque`.
//!
//! # Storage is host-resident between calls
//!
//! Like `ReferenceCpuExecutor`, tensors live in an in-process
//! `Mutex<BTreeMap<TensorResourceId, HostTensor>>` between Kernel
//! invocations. Each [`CudaKernels`] method already uploads its inputs and
//! downloads its output internally per call (design.md's "explicit data
//! movement" decision), so this executor does not additionally keep a
//! *persistent* device-side buffer across separate Kernel invocations: two
//! back-to-back kernels round-trip through host memory rather than chaining
//! device-resident results. This is the same simplification `design.md`
//! names under "direct `cuMemAlloc`/`cuMemFree` per buffer, not the Device
//! Memory Pool contract" -- true cross-call device residency is future work,
//! not part of this baseline.

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

use crate::kernels::CudaKernels;
use crate::provider::CUDA_PROVIDER_NAME;

pub struct CudaExecutor {
    kernels: CudaKernels,
    device_id: DeviceId,
    storage: Mutex<BTreeMap<TensorResourceId, HostTensor>>,
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

    pub fn write_tensor(&self, id: TensorResourceId, tensor: HostTensor) {
        self.storage.lock().unwrap().insert(id, tensor);
    }

    pub fn read_tensor(&self, id: &TensorResourceId) -> Option<HostTensor> {
        self.storage.lock().unwrap().get(id).cloned()
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
        let previous = self
            .resource_allocations
            .lock()
            .unwrap()
            .insert(id.clone(), allocation.id);
        if let Some(previous) = previous {
            let _ = memory.release(previous);
        }
        self.storage.lock().unwrap().insert(id, tensor);
        Ok(())
    }

    pub fn read_tensor_value(&self, id: &TensorResourceId) -> Option<TensorValue> {
        self.read_tensor(id).map(TensorValue::Host)
    }

    pub fn write_tensor_value(
        &self,
        id: TensorResourceId,
        value: TensorValue,
    ) -> Result<(), ProviderExecutionError> {
        if let TensorValue::Host(tensor) = value {
            self.write_tensor(id, tensor);
        }
        Ok(())
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
            TensorValue::Opaque => Ok(()),
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

    fn input_tensor(
        &self,
        invocation: &KernelInvocation,
        index: usize,
    ) -> Result<HostTensor, KernelError> {
        let resource =
            invocation
                .inputs
                .get(index)
                .ok_or_else(|| KernelError::KernelExecutionFailed {
                    reason: format!("missing input at index {index}"),
                })?;
        self.read_tensor(&resource.resource.id)
            .ok_or_else(|| KernelError::KernelExecutionFailed {
                reason: format!(
                    "no materialized data for input resource {}",
                    resource.resource.id
                ),
            })
    }

    fn store_output(
        &self,
        invocation: &KernelInvocation,
        index: usize,
        tensor: HostTensor,
    ) -> Result<magnetar_runtime::compute::TensorResourceDescriptor, KernelError> {
        let resource =
            invocation
                .outputs
                .get(index)
                .ok_or_else(|| KernelError::KernelExecutionFailed {
                    reason: format!("missing output at index {index}"),
                })?;
        self.write_tensor(resource.resource.id.clone(), tensor);
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
        let output = match name {
            "matmul" => {
                let a = self.input_tensor(invocation, 0)?;
                let b = self.input_tensor(invocation, 1)?;
                let transpose_a =
                    Self::attribute_bool(&invocation.attributes, "transpose_a", false);
                let transpose_b =
                    Self::attribute_bool(&invocation.attributes, "transpose_b", false);
                self.kernels
                    .matmul(&a, &b, transpose_a, transpose_b)
                    .map_err(KernelError::from)?
            }
            "embedding" => {
                let table = self.input_tensor(invocation, 0)?;
                let ids = self.input_tensor(invocation, 1)?;
                self.kernels
                    .embedding_lookup(&table, &ids)
                    .map_err(KernelError::from)?
            }
            "rmsnorm" => {
                let input = self.input_tensor(invocation, 0)?;
                let weight = self.input_tensor(invocation, 1)?;
                let epsilon = Self::attribute_float(&invocation.attributes, "epsilon", 1e-6);
                self.kernels
                    .rmsnorm(&input, &weight, epsilon)
                    .map_err(KernelError::from)?
            }
            "rope" => {
                let input = self.input_tensor(invocation, 0)?;
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
                    Some(OperatorAttributeValue::Integer(offset)) if *offset >= 0 => *offset as u64,
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
                    .rope(&input, base, scale, dimension, position_offset)
                    .map_err(KernelError::from)?
            }
            "attention" => {
                let q = self.input_tensor(invocation, 0)?;
                let k = self.input_tensor(invocation, 1)?;
                let v = self.input_tensor(invocation, 2)?;
                let head_count = Self::attribute_integer(&invocation.attributes, "head_count")
                    .ok_or_else(|| KernelError::KernelAttributeUnsupported {
                        attribute: "head_count".into(),
                    })?;
                let head_dimension =
                    Self::attribute_integer(&invocation.attributes, "head_dimension").ok_or_else(
                        || KernelError::KernelAttributeUnsupported {
                            attribute: "head_dimension".into(),
                        },
                    )?;
                let kv_head_count =
                    Self::attribute_integer(&invocation.attributes, "kv_head_count");
                let window_size = Self::attribute_integer(&invocation.attributes, "window_size");
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
                        &q,
                        &k,
                        &v,
                        head_count,
                        head_dimension,
                        kv_head_count,
                        window_size,
                        causal,
                    )
                    .map_err(KernelError::from)?
            }
            "softmax" => {
                let input = self.input_tensor(invocation, 0)?;
                self.kernels
                    .softmax_rows(&input)
                    .map_err(KernelError::from)?
            }
            "silu" => self
                .kernels
                .silu(&self.input_tensor(invocation, 0)?)
                .map_err(KernelError::from)?,
            "add" => {
                let a = self.input_tensor(invocation, 0)?;
                let b = self.input_tensor(invocation, 1)?;
                self.kernels.add(&a, &b).map_err(KernelError::from)?
            }
            "mul" => {
                let a = self.input_tensor(invocation, 0)?;
                let b = self.input_tensor(invocation, 1)?;
                self.kernels.mul(&a, &b).map_err(KernelError::from)?
            }
            "residual-add" => {
                let input = self.input_tensor(invocation, 0)?;
                let residual = self.input_tensor(invocation, 1)?;
                self.kernels
                    .residual_add(&input, &residual)
                    .map_err(KernelError::from)?
            }
            other => {
                return Err(KernelError::KernelNotFound {
                    kernel: other.into(),
                });
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
        for output in &invocation.outputs {
            let resource = &output.resource;
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
        CudaExecutor::write_tensor(self, id, tensor);
        Ok(())
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
