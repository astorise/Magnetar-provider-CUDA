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
use magnetar_runtime::compute::{
    ComputeDType, DTypeDescriptor, TensorDescriptor, TensorResourceId,
};
use magnetar_runtime::device::DeviceId;
use magnetar_runtime::kernel::{
    KernelAdvertisement, KernelError, KernelInvocation, KernelObservation, KernelObservationKind,
    KernelResult, KernelResultStatus,
};
use magnetar_runtime::memory::{
    MemoryAllocationClass, MemoryAllocationId, MemoryAllocationOwner, MemoryAllocationRequest,
    MemoryError, MemoryManager, MemoryPlacement, TensorResidency,
};
use magnetar_runtime::operator::{
    OperatorAttributeValue, OperatorSpec, RopeBandedFrequencyScaling,
};
use magnetar_runtime::packed_weight::{PackedWeight, PackedWeightWrite};
use magnetar_runtime::provider::{
    ProviderExecutionApi, StepCaptureId, StepCaptureRequest, StepReplayError, TensorValue,
    TensorValueAdmissionError,
};
use magnetar_runtime::scheduler::{
    ProviderCancellationOutcome, ProviderExecutionError, ProviderExecutionErrorCode,
    ProviderExecutionHandle, ProviderExecutionId, ProviderExecutionPhase, ProviderExecutionRequest,
    ProviderExecutionResult, ProviderExecutionStatus, ScheduledOperationId, SchedulingState,
};
use magnetar_runtime::{ExecutionPlanId, HostTensor};

use crate::error::CudaError;
use crate::kernels::{CapturedStep, CudaDeviceBuffer, CudaHalfDType, CudaKernels, CudaRopePairing};

/// Reads which half-precision format an `add-half`/`mul-half` invocation
/// was selected for from its own declared input dtype -- the Kernel
/// Registry already only selects this advertisement for a request whose
/// `dtype_requirements` matched `Float16`/`BrainFloat16`
/// (`advertisements::half_precision_advertisement`), so this cannot
/// disagree with what selection already decided; it just recovers which of
/// the two the caller asked for.
fn half_dtype_from_descriptor(descriptor: &TensorDescriptor) -> Result<CudaHalfDType, KernelError> {
    match &descriptor.dtype {
        DTypeDescriptor::Portable(ComputeDType::Float16) => Ok(CudaHalfDType::Float16),
        DTypeDescriptor::Portable(ComputeDType::BrainFloat16) => Ok(CudaHalfDType::BrainFloat16),
        other => Err(KernelError::KernelDTypeUnsupported {
            dtype: format!("{other:?}"),
        }),
    }
}

/// How many recent Kernel observations a [`CudaExecutor`] keeps.
pub const KERNEL_OBSERVATION_LOG_CAPACITY: usize = 4096;

pub struct CudaExecutor {
    kernels: CudaKernels,
    device_id: DeviceId,
    /// This executor's own Provider name -- not always
    /// `provider::CUDA_PROVIDER_NAME`: a second `CudaProvider` instance
    /// bound to a different real GPU ordinal registers under its own
    /// distinct name (`add-real-second-gpu-cuda-provider`), and every
    /// binding/execution-id this executor produces must carry exactly that
    /// name, matching what actually registered its Device.
    provider_name: String,
    storage: Mutex<BTreeMap<TensorResourceId, CudaDeviceBuffer>>,
    observations: Mutex<Vec<KernelObservation>>,
    submitted: Mutex<BTreeMap<ProviderExecutionId, ProviderExecutionRequest>>,
    kernel_executions: Mutex<BTreeMap<ProviderExecutionId, KernelResult>>,
    resource_allocations: Mutex<BTreeMap<TensorResourceId, MemoryAllocationId>>,
    next_execution_ordinal: AtomicU64,
    /// Decode-step capture and replay (`capture-and-replay-cuda-graphs`).
    step_capture: Mutex<StepCaptureState>,
}

/// The device address and length (in floats) of a resource's buffer, as a
/// step capture recorded it.
type CapturedResources = BTreeMap<TensorResourceId, (u64, usize)>;

#[derive(Default)]
struct StepCaptureState {
    /// What the capture in progress touched so far.
    recording: Option<CaptureRecording>,
    captures: BTreeMap<StepCaptureId, CapturedExecutorStep>,
    next_capture: u64,
}

/// What a step capture records besides its graph.
#[derive(Default)]
struct CaptureRecording {
    /// Every resource the captured work reads or writes.
    resources: CapturedResources,
    /// The token-id inputs of captured embedding lookups, with their table's
    /// vocabulary size: replayed ids are checked on the host.
    embedding_ids: BTreeMap<TensorResourceId, u64>,
}

/// One captured step: its graph, and the exact buffers it reads and writes.
struct CapturedExecutorStep {
    step: CapturedStep,
    recording: CaptureRecording,
}

impl CudaExecutor {
    pub fn new(
        kernels: CudaKernels,
        device_id: DeviceId,
        provider_name: impl Into<String>,
    ) -> Self {
        Self {
            kernels,
            device_id,
            provider_name: provider_name.into(),
            storage: Mutex::new(BTreeMap::new()),
            observations: Mutex::new(Vec::new()),
            submitted: Mutex::new(BTreeMap::new()),
            kernel_executions: Mutex::new(BTreeMap::new()),
            resource_allocations: Mutex::new(BTreeMap::new()),
            next_execution_ordinal: AtomicU64::new(0),
            step_capture: Mutex::new(StepCaptureState::default()),
        }
    }

    fn step_capture_error(
        &self,
        phase: ProviderExecutionPhase,
        detail: String,
    ) -> ProviderExecutionError {
        ProviderExecutionError::new(
            ProviderExecutionErrorCode::ExecutionFailed,
            phase,
            self.provider_binding(),
            Some(self.device_binding()),
            detail,
        )
    }

    /// See [`ProviderExecutionApi::begin_step_capture`]: the invocations up
    /// to [`Self::end_step_capture`] are captured into a CUDA graph instead
    /// of running.
    pub fn begin_step_capture(
        &self,
        request: &StepCaptureRequest,
    ) -> Result<(), ProviderExecutionError> {
        let mut state = self.step_capture.lock().unwrap();
        if state.recording.is_some() {
            return Err(self.step_capture_error(
                ProviderExecutionPhase::Submit,
                "a step capture is already in progress".into(),
            ));
        }
        self.kernels
            .begin_capture(request.advancing_attributes.iter().cloned().collect())
            .map_err(|error| {
                self.step_capture_error(ProviderExecutionPhase::Submit, error.to_string())
            })?;
        state.recording = Some(CaptureRecording::default());
        Ok(())
    }

    /// See [`ProviderExecutionApi::end_step_capture`]: instantiates the
    /// captured graph and runs it once, so the captured step's results are
    /// available.
    pub fn end_step_capture(&self) -> Result<StepCaptureId, ProviderExecutionError> {
        let mut state = self.step_capture.lock().unwrap();
        let Some(recording) = state.recording.take() else {
            return Err(self.step_capture_error(
                ProviderExecutionPhase::Complete,
                "no step capture is in progress".into(),
            ));
        };
        let complete = |error: CudaError| {
            self.step_capture_error(ProviderExecutionPhase::Complete, error.to_string())
        };
        let mut step = self.kernels.end_capture().map_err(complete)?;
        self.kernels
            .launch_captured(&mut step, 0)
            .map_err(complete)?;
        if step.has_embedding {
            self.kernels.check_embedding_flag().map_err(complete)?;
        }
        let id = StepCaptureId::new(format!(
            "{}:step-capture:{}",
            self.provider_name, state.next_capture
        ));
        state.next_capture += 1;
        state
            .captures
            .insert(id.clone(), CapturedExecutorStep { step, recording });
        Ok(id)
    }

    /// See [`ProviderExecutionApi::abort_step_capture`].
    pub fn abort_step_capture(&self) {
        self.step_capture.lock().unwrap().recording = None;
        self.kernels.abort_capture();
    }

    /// See [`ProviderExecutionApi::replay_step`].
    pub fn replay_step(
        &self,
        id: &StepCaptureId,
        offset: u64,
        inputs: Vec<(TensorResourceId, HostTensor)>,
    ) -> Result<(), StepReplayError> {
        let invalidated = |reason: String| StepReplayError::Invalidated { reason };
        let mut state = self.step_capture.lock().unwrap();
        let captured = state
            .captures
            .get_mut(id)
            .ok_or_else(|| invalidated(format!("no step capture '{id}'")))?;
        if offset > captured.step.max_offset {
            return Err(invalidated(format!(
                "offset {offset} leaves the captured bounds (at most {})",
                captured.step.max_offset
            )));
        }
        let mut storage = self.storage.lock().unwrap();
        for (resource, (address, len)) in &captured.recording.resources {
            let current = storage
                .get(resource)
                .map(|buffer| (self.kernels.address(buffer), buffer.len()));
            if current != Some((*address, *len)) {
                return Err(invalidated(format!(
                    "resource '{resource}' no longer has its captured storage"
                )));
            }
        }
        for (resource, tensor) in &inputs {
            if !captured.recording.resources.contains_key(resource) {
                return Err(invalidated(format!(
                    "input '{resource}' was not read by the captured step"
                )));
            }
            if let Some(&vocab) = captured.recording.embedding_ids.get(resource)
                && tensor.data.iter().any(|&value| {
                    !(value >= 0.0 && value == value.floor() && (value as u64) < vocab)
                })
            {
                return Err(StepReplayError::Failed(self.step_capture_error(
                    ProviderExecutionPhase::Submit,
                    "embedding lookup id is not a valid in-range non-negative integer".into(),
                )));
            }
        }
        let failed = |error: CudaError| {
            StepReplayError::Failed(
                self.step_capture_error(ProviderExecutionPhase::Submit, error.to_string()),
            )
        };
        for (resource, tensor) in inputs {
            let buffer = storage
                .get_mut(&resource)
                .expect("every input was checked above");
            self.kernels
                .write_into(buffer, &tensor.data)
                .map_err(failed)?;
        }
        drop(storage);
        self.kernels
            .launch_captured(&mut captured.step, offset)
            .map_err(failed)
    }

    /// See [`ProviderExecutionApi::release_step_capture`].
    pub fn release_step_capture(&self, id: &StepCaptureId) {
        self.step_capture.lock().unwrap().captures.remove(id);
    }

    /// Records the buffers `invocation` read and wrote, while capturing.
    fn record_captured_resources(&self, invocation: &KernelInvocation) {
        if self.step_capture.lock().unwrap().recording.is_none() {
            return;
        }
        let (touched, embedding_ids) = {
            let storage = self.storage.lock().unwrap();
            // An embedding lookup reads its table (input 0) and its ids
            // (input 1).
            let embedding_ids = match (
                invocation.kernel.name.as_str(),
                invocation.inputs.first(),
                invocation.inputs.get(1),
            ) {
                ("embedding", Some(table), Some(ids)) => storage
                    .get(&table.resource.id)
                    .and_then(|table| table.shape.first().copied())
                    .map(|vocab| (ids.resource.id.clone(), vocab)),
                _ => None,
            };
            let touched: Vec<(TensorResourceId, (u64, usize))> = invocation
                .inputs
                .iter()
                .chain(&invocation.outputs)
                .filter_map(|resource| {
                    let id = &resource.resource.id;
                    storage
                        .get(id)
                        .map(|buffer| (id.clone(), (self.kernels.address(buffer), buffer.len())))
                })
                .collect();
            (touched, embedding_ids)
        };
        if let Some(recording) = self.step_capture.lock().unwrap().recording.as_mut() {
            recording.resources.extend(touched);
            recording.embedding_ids.extend(embedding_ids);
        }
    }

    fn provider_binding(&self) -> ProviderBinding {
        ProviderBinding::new(self.provider_name.as_str())
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
    /// See [`ProviderExecutionApi::write_weight_tensor`]: a weight whose
    /// values are all bfloat16 values is stored compact
    /// ([`CudaKernels::upload_weight`]).
    pub fn write_weight_tensor(
        &self,
        id: TensorResourceId,
        tensor: HostTensor,
    ) -> Result<(), CudaError> {
        let buffer = self.kernels.upload_weight(&tensor)?;
        self.storage.lock().unwrap().insert(id, buffer);
        Ok(())
    }

    /// See [`ProviderExecutionApi::write_packed_weight_tensor`]: a `Q8_0`
    /// weight is stored packed ([`CudaKernels::upload_packed_weight`])
    /// unless [`crate::kernels::PACKED_WEIGHTS_ENV`] is `0`.
    pub fn write_packed_weight_tensor(
        &self,
        id: TensorResourceId,
        weight: &PackedWeight,
    ) -> Result<PackedWeightWrite, CudaError> {
        if std::env::var(crate::kernels::PACKED_WEIGHTS_ENV).as_deref() == Ok("0") {
            return Ok(PackedWeightWrite::Declined);
        }
        let buffer = self.kernels.upload_packed_weight(weight)?;
        self.storage.lock().unwrap().insert(id, buffer);
        Ok(PackedWeightWrite::Stored)
    }

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

    /// See [`ProviderExecutionApi::resident_shape`]: the device allocation
    /// table already records each buffer's shape (`CudaDeviceBuffer.shape`),
    /// so this never touches the device itself, let alone downloads.
    pub fn resident_shape(&self, id: &TensorResourceId) -> Option<Vec<u64>> {
        self.storage
            .lock()
            .unwrap()
            .get(id)
            .map(|buffer| buffer.shape.clone())
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

    /// See [`ProviderExecutionApi::write_zeroed_tensor`]: the zeros are set
    /// on the Device, replacing whatever this Provider held for `id`.
    pub fn write_zeroed_tensor(
        &self,
        id: &TensorResourceId,
        shape: &[u64],
    ) -> Result<(), ProviderExecutionError> {
        let buffer = self.kernels.zeros(shape).map_err(|error| {
            ProviderExecutionError::new(
                ProviderExecutionErrorCode::MaterializationFailed,
                ProviderExecutionPhase::Submit,
                self.provider_binding(),
                Some(self.device_binding()),
                format!("CUDA zeroed allocation failed: {error}"),
            )
        })?;
        self.storage.lock().unwrap().insert(id.clone(), buffer);
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

    /// See [`ProviderExecutionApi::copy_tensor_admitted`]: duplicates
    /// `from`'s current device-resident bytes to a fresh identity `to`, via
    /// a real device-to-device copy -- never downloading to host. Mirrors
    /// [`Self::write_tensor_admitted`]'s admit-then-write-with-rollback
    /// sequence, replacing (and releasing) whatever `to` previously held.
    pub fn copy_tensor_admitted(
        &self,
        memory: &mut MemoryManager,
        from: &TensorResourceId,
        to: TensorResourceId,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<MemoryAllocationId, TensorValueAdmissionError> {
        let cloned_buffer = {
            let storage = self.storage.lock().unwrap();
            let buffer = storage.get(from).ok_or_else(|| {
                TensorValueAdmissionError::Provider(ProviderExecutionError::new(
                    ProviderExecutionErrorCode::MaterializationFailed,
                    ProviderExecutionPhase::Submit,
                    self.provider_binding(),
                    Some(self.device_binding()),
                    format!("copy_tensor_admitted: no existing device allocation for '{from}'"),
                ))
            })?;
            self.kernels.clone_buffer(buffer).map_err(|error| {
                TensorValueAdmissionError::Provider(ProviderExecutionError::new(
                    ProviderExecutionErrorCode::MaterializationFailed,
                    ProviderExecutionPhase::Submit,
                    self.provider_binding(),
                    Some(self.device_binding()),
                    format!("copy_tensor_admitted: device-to-device copy failed: {error}"),
                ))
            })?
        };
        let byte_size = cloned_buffer.len() as u64 * std::mem::size_of::<f32>() as u64;
        let allocation = memory
            .allocate(MemoryAllocationRequest::new(
                class,
                byte_size,
                MemoryPlacement::Device(self.device_binding()),
                owner,
            ))
            .map_err(TensorValueAdmissionError::Memory)?;
        let previous = self
            .resource_allocations
            .lock()
            .unwrap()
            .insert(to.clone(), allocation.id);
        if let Some(previous) = previous {
            let _ = memory.release(previous);
        }
        self.storage.lock().unwrap().insert(to, cloned_buffer);
        Ok(allocation.id)
    }

    /// Copies `peer_id`'s current device allocation from `peer`'s own
    /// storage directly into this executor's own storage under `id`, via
    /// a real cross-context CUDA device-to-device copy
    /// (`CudaKernels::clone_buffer`'s own `CudaStream::memcpy_dtod`, which
    /// automatically takes the real `cuMemcpyPeerAsync` path whenever
    /// source and destination belong to different real CUDA contexts --
    /// see `crate::peer`'s own doc comment) -- never touching host
    /// memory, genuinely distinct from `write_tensor_value_admitted`'s
    /// explicit host-staged crossing every other cross-Device movement in
    /// this repository uses. Callers MUST have already confirmed real
    /// peer capability via `crate::peer::device_can_access_peer` and
    /// called `crate::peer::enable_peer_access` between the two Devices
    /// -- this method does not check or enable peer access itself
    /// (`multi-device-placement`'s "Peer Capability Is Explicit").
    pub fn copy_tensor_from_peer_admitted(
        &self,
        memory: &mut MemoryManager,
        peer: &CudaExecutor,
        peer_id: &TensorResourceId,
        id: TensorResourceId,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<MemoryAllocationId, TensorValueAdmissionError> {
        let cloned_buffer = {
            let peer_storage = peer.storage.lock().unwrap();
            let buffer = peer_storage.get(peer_id).ok_or_else(|| {
                TensorValueAdmissionError::Provider(ProviderExecutionError::new(
                    ProviderExecutionErrorCode::MaterializationFailed,
                    ProviderExecutionPhase::Submit,
                    self.provider_binding(),
                    Some(self.device_binding()),
                    format!(
                        "copy_tensor_from_peer_admitted: no existing device allocation for \
                         '{peer_id}' on the peer Device"
                    ),
                ))
            })?;
            self.kernels.clone_buffer(buffer).map_err(|error| {
                TensorValueAdmissionError::Provider(ProviderExecutionError::new(
                    ProviderExecutionErrorCode::MaterializationFailed,
                    ProviderExecutionPhase::Submit,
                    self.provider_binding(),
                    Some(self.device_binding()),
                    format!(
                        "copy_tensor_from_peer_admitted: real peer device-to-device copy \
                         failed: {error}"
                    ),
                ))
            })?
        };
        let byte_size = cloned_buffer.len() as u64 * std::mem::size_of::<f32>() as u64;
        let allocation = memory
            .allocate(MemoryAllocationRequest::new(
                class,
                byte_size,
                MemoryPlacement::Device(self.device_binding()),
                owner,
            ))
            .map_err(TensorValueAdmissionError::Memory)?;
        let previous = self
            .resource_allocations
            .lock()
            .unwrap()
            .insert(id.clone(), allocation.id);
        if let Some(previous) = previous {
            let _ = memory.release(previous);
        }
        self.storage.lock().unwrap().insert(id, cloned_buffer);
        Ok(allocation.id)
    }

    /// The most recent Kernel observations, oldest first, at most
    /// `KERNEL_OBSERVATION_LOG_CAPACITY` of them.
    pub fn observations(&self) -> Vec<KernelObservation> {
        let observations = self.observations.lock().unwrap();
        let start = observations
            .len()
            .saturating_sub(KERNEL_OBSERVATION_LOG_CAPACITY);
        observations[start..].to_vec()
    }

    fn observe(&self, observation: KernelObservation) {
        let mut observations = self.observations.lock().unwrap();
        // Bounded like the Runtime's own observation logs: a resident model
        // records two observations per dispatched Kernel for as long as it
        // serves. Dropping the oldest half at twice the capacity keeps the
        // cost per observation amortized O(1).
        if observations.len() >= 2 * KERNEL_OBSERVATION_LOG_CAPACITY {
            observations.drain(..KERNEL_OBSERVATION_LOG_CAPACITY);
        }
        observations.push(observation);
    }

    fn next_provider_execution_id(&self, label: &str) -> ProviderExecutionId {
        let ordinal = self.next_execution_ordinal.fetch_add(1, Ordering::Relaxed);
        ProviderExecutionId::new(format!("{}:{label}:{ordinal}", self.provider_name))
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

    /// `matmul` and `matmul-bias` accumulate in f32 only, like the CPU
    /// Providers; any other requested `accumulation_dtype` is rejected
    /// rather than silently computed in f32.
    fn require_f32_accumulation(
        attributes: &BTreeMap<String, OperatorAttributeValue>,
    ) -> Result<(), KernelError> {
        if let Some(OperatorAttributeValue::DType(dtype)) = attributes.get("accumulation_dtype")
            && *dtype != ComputeDType::Float32
        {
            return Err(KernelError::KernelDTypeUnsupported {
                dtype: format!("the CUDA Provider only accumulates in f32; requested {dtype:?}"),
            });
        }
        Ok(())
    }

    fn run_invocation(&self, invocation: &KernelInvocation) -> Result<KernelResult, KernelError> {
        let result = self.run_invocation_uncaptured(invocation);
        if result.is_ok() {
            self.record_captured_resources(invocation);
        }
        result
    }

    fn run_invocation_uncaptured(
        &self,
        invocation: &KernelInvocation,
    ) -> Result<KernelResult, KernelError> {
        self.expand_unsupported_compact_inputs(invocation)?;
        if invocation.kernel.name == "kv-write" {
            return self.run_kv_write(invocation);
        }
        // The resource this invocation overwrites: its previous buffer is
        // dropped once the new output is stored anyway, so it is offered
        // to the Kernel as its output allocation (a fresh allocation and a
        // free per dispatched node otherwise). Never when the invocation
        // also reads that resource. A dispatch that fails before a Kernel
        // takes the buffer puts it back, so the resource keeps its value.
        let previous_output = invocation.outputs.first().and_then(|output| {
            let id = &output.resource.id;
            if invocation
                .inputs
                .iter()
                .any(|input| &input.resource.id == id)
            {
                return None;
            }
            let mut storage = self.storage.lock().unwrap();
            // A compact buffer has no f32 allocation to offer.
            if storage.get(id).is_none_or(CudaDeviceBuffer::is_compact) {
                return None;
            }
            let buffer = storage.remove(id)?;
            Some((id.clone(), buffer))
        });
        let (restore, offered_slice) = match previous_output {
            Some((id, buffer)) => (Some((id, buffer.shape)), Some(buffer.slice)),
            None => (None, None),
        };
        let offered = self.kernels.offer_output(offered_slice);
        let result = self.run_invocation_with_offer(invocation);
        if result.is_err()
            && let Some((id, shape)) = restore
            && let Some(slice) = offered.reclaim()
        {
            self.storage.lock().unwrap().insert(
                id,
                CudaDeviceBuffer {
                    slice,
                    shape,
                    bf16: None,
                    q8: None,
                },
            );
        }
        result
    }

    /// Stores as f32, for good, every compact input `invocation` reads
    /// through a Kernel argument that only reads f32: only the matmul
    /// family's `b` (input 1) and the embedding lookup's table (input 0) read
    /// compact weights (`execute-native-half-precision`).
    fn expand_unsupported_compact_inputs(
        &self,
        invocation: &KernelInvocation,
    ) -> Result<(), KernelError> {
        let reads_compact = |index: usize| match invocation.kernel.name.as_str() {
            "matmul" | "matmul-bias" | "matmul-residual" => index == 1,
            "gated-matmul" => index == 1 || index == 2,
            "embedding" => index == 0,
            _ => false,
        };
        let mut storage = self.storage.lock().unwrap();
        for (index, input) in invocation.inputs.iter().enumerate() {
            let id = &input.resource.id;
            if reads_compact(index) || !storage.get(id).is_some_and(CudaDeviceBuffer::is_compact) {
                continue;
            }
            let expanded = self
                .kernels
                .expand(&storage[id])
                .map_err(KernelError::from)?;
            storage.insert(id.clone(), expanded);
        }
        Ok(())
    }

    /// `kv-write` (`append-kv-history-in-place`): writes the rows into the
    /// store's own device buffer in place when the output is the store, as
    /// it is for a KV history; otherwise into a copy stored under the output.
    fn run_kv_write(&self, invocation: &KernelInvocation) -> Result<KernelResult, KernelError> {
        let row_offset =
            Self::attribute_integer(&invocation.attributes, "row_offset").ok_or_else(|| {
                KernelError::KernelAttributeUnsupported {
                    attribute: "row_offset".into(),
                }
            })?;
        let store_id = Self::input_resource_id(invocation, 0)?;
        let rows_id = Self::input_resource_id(invocation, 1)?;
        let output =
            invocation
                .outputs
                .first()
                .ok_or_else(|| KernelError::KernelExecutionFailed {
                    reason: "missing output at index 0".into(),
                })?;
        let mut storage = self.storage.lock().unwrap();
        let mut store = if output.resource.id == *store_id {
            storage.remove(store_id)
        } else {
            match storage.get(store_id) {
                Some(store) => Some(
                    self.kernels
                        .clone_buffer(store)
                        .map_err(KernelError::from)?,
                ),
                None => None,
            }
        }
        .ok_or_else(|| KernelError::KernelExecutionFailed {
            reason: format!("no materialized data for input resource {store_id}"),
        })?;
        let written = match storage.get(rows_id) {
            Some(rows) => self
                .kernels
                .kv_write(&mut store, rows, row_offset)
                .map_err(KernelError::from),
            None => Err(KernelError::KernelExecutionFailed {
                reason: format!("no materialized data for input resource {rows_id}"),
            }),
        };
        // The store goes back under its id either way: a failed write
        // leaves its previous rows.
        storage.insert(output.resource.id.clone(), store);
        written?;
        let mut result = KernelResult::success(invocation.id.clone());
        result
            .output_readiness
            .insert(output.resource.id.to_string(), true);
        result.updated_resources.push(output.resource.clone());
        Ok(result)
    }

    fn run_invocation_with_offer(
        &self,
        invocation: &KernelInvocation,
    ) -> Result<KernelResult, KernelError> {
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
                    Self::require_f32_accumulation(&invocation.attributes)?;
                    self.kernels
                        .matmul(a, b, transpose_a, transpose_b)
                        .map_err(KernelError::from)?
                }
                // The Runtime's fused `matmul(a, b) + bias` (`graph_fusion`).
                "matmul-bias" => {
                    let a = get(0)?;
                    let b = get(1)?;
                    let bias = get(2)?;
                    let transpose_a =
                        Self::attribute_bool(&invocation.attributes, "transpose_a", false);
                    let transpose_b =
                        Self::attribute_bool(&invocation.attributes, "transpose_b", false);
                    Self::require_f32_accumulation(&invocation.attributes)?;
                    self.kernels
                        .matmul_bias(a, b, bias, transpose_a, transpose_b)
                        .map_err(KernelError::from)?
                }
                // The Runtime's fused `residual + matmul(a, b)`.
                "matmul-residual" => {
                    let a = get(0)?;
                    let b = get(1)?;
                    let residual = get(2)?;
                    let transpose_a =
                        Self::attribute_bool(&invocation.attributes, "transpose_a", false);
                    let transpose_b =
                        Self::attribute_bool(&invocation.attributes, "transpose_b", false);
                    Self::require_f32_accumulation(&invocation.attributes)?;
                    self.kernels
                        .matmul_residual(a, b, residual, transpose_a, transpose_b)
                        .map_err(KernelError::from)?
                }
                // The Runtime's fused `silu(a x gate) * (a x up)`.
                "gated-matmul" => {
                    let transpose_a =
                        Self::attribute_bool(&invocation.attributes, "transpose_a", false);
                    let transpose_b =
                        Self::attribute_bool(&invocation.attributes, "transpose_b", false);
                    Self::require_f32_accumulation(&invocation.attributes)?;
                    self.kernels
                        .gated_matmul(get(0)?, get(1)?, get(2)?, transpose_a, transpose_b)
                        .map_err(KernelError::from)?
                }
                // The Runtime's fused `silu(gate) * up`.
                "silu-mul" => self
                    .kernels
                    .silu_mul(get(0)?, get(1)?)
                    .map_err(KernelError::from)?,
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
                    // `make-first-native-cuda-hot-path-device-resident`:
                    // absent means single-block, identical to this
                    // Kernel's behavior before `head_count` existed.
                    let head_count =
                        Self::attribute_integer(&invocation.attributes, "head_count").unwrap_or(1);
                    // `fix-rope-rotation-pairing`: absent means
                    // interleaved, this Kernel's behavior before the
                    // attribute existed.
                    let pairing = match invocation.attributes.get("rotation_pairing") {
                        None => CudaRopePairing::Interleaved,
                        Some(OperatorAttributeValue::String(value)) => {
                            CudaRopePairing::from_attribute(value).ok_or_else(|| {
                                KernelError::KernelAttributeUnsupported {
                                    attribute: format!(
                                        "rotation_pairing '{value}' is not implemented"
                                    ),
                                }
                            })?
                        }
                        Some(_) => {
                            return Err(KernelError::KernelAttributeUnsupported {
                                attribute: "rotation_pairing must be a string".into(),
                            });
                        }
                    };
                    let frequency_scaling =
                        RopeBandedFrequencyScaling::from_attributes(&invocation.attributes)
                            .map_err(|attribute| KernelError::KernelAttributeUnsupported {
                                attribute,
                            })?;
                    self.kernels
                        .rope_with_frequency_scaling(
                            input,
                            base,
                            scale,
                            dimension,
                            position_offset,
                            head_count,
                            pairing,
                            frequency_scaling.as_ref(),
                        )
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
                    let key_count = Self::attribute_integer(&invocation.attributes, "key_count");
                    self.kernels
                        .attention_bounded(
                            q,
                            k,
                            v,
                            head_count,
                            head_dimension,
                            kv_head_count,
                            window_size,
                            causal,
                            key_count,
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
                "concat" => {
                    let a = get(0)?;
                    let b = get(1)?;
                    self.kernels.concat(a, b).map_err(KernelError::from)?
                }
                "residual-add" => {
                    let input = get(0)?;
                    let residual = get(1)?;
                    self.kernels
                        .residual_add(input, residual)
                        .map_err(KernelError::from)?
                }
                "add-half" | "mul-half" => {
                    // Genuine on-device half-precision compute -- real
                    // rounding, real `CudaHalfBuffer` bytes -- but the
                    // *resource* stays stored as `f32` in this table
                    // between invocations (Phase 3's deliberately narrower
                    // scope: dispatch/selection through the standard
                    // contract, not persistent half-precision residency;
                    // see `enable-native-cuda-half-precision-elementwise-
                    // compute`'s design.md Risks). A host round trip
                    // converts each side: this is real, verified compute,
                    // not a `f32`-labeled no-op, but it is not yet
                    // device-resident chaining like every other Kernel
                    // here.
                    let a = get(0)?;
                    let b = get(1)?;
                    let dtype =
                        half_dtype_from_descriptor(&invocation.inputs[0].resource.descriptor)?;
                    let a_host = self.kernels.download(a).map_err(KernelError::from)?;
                    let b_host = self.kernels.download(b).map_err(KernelError::from)?;
                    let a_half = self
                        .kernels
                        .upload_half(&a_host, dtype)
                        .map_err(KernelError::from)?;
                    let b_half = self
                        .kernels
                        .upload_half(&b_host, dtype)
                        .map_err(KernelError::from)?;
                    let result_half = if name == "add-half" {
                        self.kernels
                            .add_half(&a_half, &b_half)
                            .map_err(KernelError::from)?
                    } else {
                        self.kernels
                            .mul_half(&a_half, &b_half)
                            .map_err(KernelError::from)?
                    };
                    let result_host = self
                        .kernels
                        .download_half(&result_half)
                        .map_err(KernelError::from)?;
                    self.kernels
                        .upload(&result_host)
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
            // Pre-admitted whenever the residency record names an
            // allocation other than the one this Provider tracks for the
            // id: the caller admitted it, or re-admitted it after this
            // Provider's own last write of the same id (an edge both
            // pre-admitted per step and explicitly written). Self-admitting
            // over it would overwrite the caller's residency record and
            // orphan the caller's allocation once per step
            // (`bound-memory-manager-bookkeeping`); instead the caller's
            // record stays authoritative and this Provider's stale
            // tracking entry for the id is dropped.
            let tracked = self
                .resource_allocations
                .lock()
                .unwrap()
                .get(&resource.id)
                .copied();
            let resident = memory
                .tensor_residency(&resource.id)
                .and_then(|residency| residency.allocation);
            if memory.tensor_residency(&resource.id).is_some() && resident != tracked {
                if let Some(stale) = self
                    .resource_allocations
                    .lock()
                    .unwrap()
                    .remove(&resource.id)
                {
                    let _ = memory.release(stale);
                }
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

    fn write_packed_weight_tensor(
        &self,
        id: TensorResourceId,
        weight: PackedWeight,
    ) -> Result<PackedWeightWrite, ProviderExecutionError> {
        CudaExecutor::write_packed_weight_tensor(self, id, &weight).map_err(|error| {
            ProviderExecutionError::new(
                ProviderExecutionErrorCode::MaterializationFailed,
                ProviderExecutionPhase::Submit,
                self.provider_binding(),
                Some(self.device_binding()),
                format!("CUDA packed weight upload failed: {error}"),
            )
        })
    }

    fn write_weight_tensor(
        &self,
        id: TensorResourceId,
        tensor: HostTensor,
    ) -> Result<(), ProviderExecutionError> {
        CudaExecutor::write_weight_tensor(self, id, tensor).map_err(|error| {
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

    /// Downloads only the requested rows (a prefill's last logits row).
    fn read_tensor_rows(
        &self,
        id: &TensorResourceId,
        first_row: u64,
        rows: u64,
    ) -> Option<HostTensor> {
        let storage = self.storage.lock().unwrap();
        let buffer = storage.get(id)?;
        match self.kernels.download_rows(buffer, first_row, rows) {
            Ok(Some(tensor)) => Some(tensor),
            // A compact or other-rank buffer: the whole tensor, sliced.
            Ok(None) => {
                let whole = self.kernels.download(buffer).ok()?;
                let [_, columns] = whole.shape[..] else {
                    return None;
                };
                let start = (first_row * columns) as usize;
                let end = ((first_row + rows) * columns) as usize;
                HostTensor::new([rows, columns], whole.data.get(start..end)?.to_vec()).ok()
            }
            Err(_) => None,
        }
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

    fn resident_shape(&self, id: &TensorResourceId) -> Option<Vec<u64>> {
        CudaExecutor::resident_shape(self, id)
    }

    fn write_tensor_value(
        &self,
        id: TensorResourceId,
        value: TensorValue,
    ) -> Result<(), ProviderExecutionError> {
        CudaExecutor::write_tensor_value(self, id, value)
    }

    fn write_zeroed_tensor(
        &self,
        id: &TensorResourceId,
        shape: &[u64],
    ) -> Result<bool, ProviderExecutionError> {
        CudaExecutor::write_zeroed_tensor(self, id, shape).map(|()| true)
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

    fn copy_tensor_admitted(
        &self,
        memory: &mut MemoryManager,
        from: &TensorResourceId,
        to: TensorResourceId,
        class: MemoryAllocationClass,
        owner: MemoryAllocationOwner,
    ) -> Result<MemoryAllocationId, TensorValueAdmissionError> {
        CudaExecutor::copy_tensor_admitted(self, memory, from, to, class, owner)
    }

    fn observations(&self) -> Vec<KernelObservation> {
        CudaExecutor::observations(self)
    }

    fn supports_step_capture(&self) -> bool {
        true
    }

    fn begin_step_capture(
        &self,
        request: &StepCaptureRequest,
    ) -> Result<(), ProviderExecutionError> {
        CudaExecutor::begin_step_capture(self, request)
    }

    fn end_step_capture(&self) -> Result<StepCaptureId, ProviderExecutionError> {
        CudaExecutor::end_step_capture(self)
    }

    fn abort_step_capture(&self) {
        CudaExecutor::abort_step_capture(self)
    }

    fn replay_step(
        &self,
        id: &StepCaptureId,
        offset: u64,
        inputs: Vec<(TensorResourceId, HostTensor)>,
    ) -> Result<(), StepReplayError> {
        CudaExecutor::replay_step(self, id, offset, inputs)
    }

    fn release_step_capture(&self, id: &StepCaptureId) {
        CudaExecutor::release_step_capture(self, id)
    }

    /// Selects a plain greedy step's token on the Device: one reduction
    /// over the logits row and an 8-byte download instead of the row.
    fn select_greedy_column(
        &self,
        logits: &TensorResourceId,
        row: u64,
        excluded: &[std::ops::Range<u64>],
        temperature: f32,
    ) -> Option<u64> {
        let storage = self.storage.lock().unwrap();
        let buffer = storage.get(logits)?;
        self.kernels
            .greedy_select(buffer, row, excluded, temperature)
            .ok()
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::CudaProvider;
    use magnetar_runtime::memory::MemoryManagerConfig;

    #[test]
    fn matmul_kernels_reject_any_accumulation_dtype_but_f32() {
        let with = |dtype| {
            BTreeMap::from([(
                "accumulation_dtype".to_string(),
                OperatorAttributeValue::DType(dtype),
            )])
        };
        assert!(CudaExecutor::require_f32_accumulation(&BTreeMap::new()).is_ok());
        assert!(CudaExecutor::require_f32_accumulation(&with(ComputeDType::Float32)).is_ok());
        for dtype in [
            ComputeDType::Float16,
            ComputeDType::BrainFloat16,
            ComputeDType::Float64,
        ] {
            assert!(matches!(
                CudaExecutor::require_f32_accumulation(&with(dtype)),
                Err(KernelError::KernelDTypeUnsupported { .. })
            ));
        }
    }

    fn executor_or_skip() -> Option<CudaExecutor> {
        let provider = CudaProvider::new();
        let context = provider.context()?;
        let kernels = CudaKernels::compile_and_load(&context).expect(
            "kernel compilation must succeed on a machine that already passed device discovery",
        );
        Some(CudaExecutor::new(
            kernels,
            DeviceId::new("test-device"),
            crate::provider::CUDA_PROVIDER_NAME,
        ))
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

    fn active_allocation_count(memory: &MemoryManager) -> usize {
        memory
            .allocations()
            .filter(|allocation| {
                allocation.state == magnetar_runtime::MemoryAllocationState::Active
            })
            .count()
    }

    /// `implement-device-resident-multi-step-cuda-decode` task 4.4:
    /// `copy_tensor_admitted` duplicates a Device-resident resource's real
    /// bytes to a fresh identity without downloading to host -- verified
    /// here by downloading only the *destination* (never the source) and
    /// comparing against the source's own real content.
    #[test]
    fn copy_tensor_admitted_duplicates_bytes_without_host_materialization() {
        let Some(executor) = executor_or_skip() else {
            return;
        };
        let mut memory = MemoryManager::new(MemoryManagerConfig::default());
        let source_id = TensorResourceId::new("copy-source");
        let tensor = HostTensor::new([2, 2], [1.0, 2.0, 3.0, 4.0]).unwrap();
        executor
            .write_tensor_admitted(
                &mut memory,
                source_id.clone(),
                tensor.clone(),
                MemoryAllocationClass::Tensor,
                MemoryAllocationOwner::Session("test".into()),
            )
            .expect("source write must succeed");

        let dest_id = TensorResourceId::new("copy-destination");
        executor
            .copy_tensor_admitted(
                &mut memory,
                &source_id,
                dest_id.clone(),
                MemoryAllocationClass::Tensor,
                MemoryAllocationOwner::Session("test".into()),
            )
            .expect("copy must succeed");

        let copied = executor
            .read_tensor(&dest_id)
            .expect("destination resource must be present after copy");
        assert_eq!(copied.shape, tensor.shape);
        assert_eq!(copied.data, tensor.data);
    }

    /// A second copy to the same destination identity replaces (and
    /// releases) the first, matching `write_tensor_value_admitted`'s own
    /// replacement discipline -- the exact bounded-growth requirement the
    /// KV pending/commit paths depend on.
    #[test]
    fn copy_tensor_admitted_replaces_a_previous_allocation_at_the_same_destination() {
        let Some(executor) = executor_or_skip() else {
            return;
        };
        let mut memory = MemoryManager::new(MemoryManagerConfig::default());
        let source_id = TensorResourceId::new("copy-source-2");
        executor
            .write_tensor_admitted(
                &mut memory,
                source_id.clone(),
                HostTensor::new([2], [1.0, 2.0]).unwrap(),
                MemoryAllocationClass::Tensor,
                MemoryAllocationOwner::Session("test".into()),
            )
            .expect("source write must succeed");

        let dest_id = TensorResourceId::new("copy-destination-stable");
        executor
            .copy_tensor_admitted(
                &mut memory,
                &source_id,
                dest_id.clone(),
                MemoryAllocationClass::Tensor,
                MemoryAllocationOwner::Session("test".into()),
            )
            .expect("first copy must succeed");
        let count_after_first_copy = active_allocation_count(&memory);

        for _ in 0..4 {
            executor
                .copy_tensor_admitted(
                    &mut memory,
                    &source_id,
                    dest_id.clone(),
                    MemoryAllocationClass::Tensor,
                    MemoryAllocationOwner::Session("test".into()),
                )
                .expect("repeated copy to the same destination must succeed");
        }
        assert_eq!(
            active_allocation_count(&memory),
            count_after_first_copy,
            "repeated copies to the same destination id must not accumulate allocations \
             beyond the source's own allocation plus one stable destination allocation"
        );
    }

    #[test]
    fn copy_tensor_admitted_rejects_a_missing_source() {
        let Some(executor) = executor_or_skip() else {
            return;
        };
        let mut memory = MemoryManager::new(MemoryManagerConfig::default());
        let error = executor
            .copy_tensor_admitted(
                &mut memory,
                &TensorResourceId::new("does-not-exist"),
                TensorResourceId::new("copy-destination-3"),
                MemoryAllocationClass::Tensor,
                MemoryAllocationOwner::Session("test".into()),
            )
            .expect_err("copying a nonexistent source must fail structurally");
        assert!(matches!(error, TensorValueAdmissionError::Provider(_)));
    }
}
