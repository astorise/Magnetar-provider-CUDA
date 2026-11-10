//! Tensor parallelism across the members (`enforce-local-multigpu-and-expert-placement`,
//! phase 1: split MLP).
//!
//! The Runtime hands over a [`TensorParallelPlan`] naming which weights are
//! split by columns or by rows. From then on:
//!
//! - a split weight is stored as each member's contiguous share (member `i`
//!   holds share `i` of the plan's Devices); every other weight, and every
//!   non-weight write, is replicated on every member;
//! - every invocation runs on every member, over that member's shares;
//! - an invocation reading a column-split weight, or an activation produced
//!   from one, produces a column share of its output on each member;
//! - an invocation reading a row-split weight produces a partial sum on
//!   each member, summed across members right after (reduced on the first
//!   member, then copied back to the others), so every later invocation
//!   sees the same full tensor everywhere.
//!
//! Shares along a dimension are the plan's equal contiguous parts, aligned
//! to a `Q8_0` block whenever the dimension allows it, so a column share of
//! an activation always matches the row share of the weight reading it.

use std::collections::BTreeMap;

use magnetar_runtime::HostTensor;
use magnetar_runtime::compute::TensorResourceId;
use magnetar_runtime::kernel::{
    KernelAdvertisement, KernelInvocation, KernelResource, KernelResultStatus,
};
use magnetar_runtime::memory::MemoryManager;
use magnetar_runtime::operator::OperatorSpec;
use magnetar_runtime::packed_weight::{PackedWeight, PackedWeightFormat, PackedWeightWrite};
use magnetar_runtime::provider::ProviderExecutionApi;
use magnetar_runtime::scheduler::{
    ProviderExecutionError, ProviderExecutionHandle, ProviderExecutionId, ProviderExecutionPhase,
};
use magnetar_runtime::tensor_parallel::{TensorParallelPlan, WeightShard};

use magnetar_runtime::provider::StepReplayError;

use super::{CapturedOp, CudaMultiExecutor, Recording};

/// The plan in force, and what it split so far.
pub(super) struct TensorParallel {
    plan: TensorParallelPlan,
    /// Each split weight's shard kind and logical (whole) shape.
    weights: BTreeMap<TensorResourceId, (WeightShard, Vec<u64>)>,
    /// Activations each member holds only its column share of, with their
    /// logical shape.
    sharded: BTreeMap<TensorResourceId, Vec<u64>>,
}

/// What an invocation's outputs are on each member.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputRole {
    /// The same whole tensor everywhere.
    Replicated,
    /// Each member's column share.
    Sharded,
    /// Each member's partial sum, reduced right after.
    Partial,
}

impl TensorParallel {
    pub(super) fn new(plan: TensorParallelPlan) -> Self {
        Self {
            plan,
            weights: BTreeMap::new(),
            sharded: BTreeMap::new(),
        }
    }

    fn parts(&self) -> usize {
        self.plan.devices().len()
    }

    /// The `[start, end)` share of `extent` member `member` holds.
    fn share(&self, member: usize, extent: u64) -> (u64, u64) {
        self.plan
            .share(member, extent, share_granule(extent, self.parts()))
    }

    fn local_extent(&self, member: usize, extent: u64) -> u64 {
        let (start, end) = self.share(member, extent);
        end - start
    }

    /// The logical shape of a split weight or sharded activation.
    pub(super) fn logical_shape(&self, id: &TensorResourceId) -> Option<Vec<u64>> {
        self.weights
            .get(id)
            .map(|(_, shape)| shape.clone())
            .or_else(|| self.sharded.get(id).cloned())
    }

    /// `to` is now a copy of `from`: sharded alike.
    pub(super) fn copied(&mut self, from: &TensorResourceId, to: &TensorResourceId) {
        match self.sharded.get(from).cloned() {
            Some(shape) => {
                self.sharded.insert(to.clone(), shape);
            }
            None => {
                self.sharded.remove(to);
            }
        }
    }

    pub(super) fn forget(&mut self, id: &TensorResourceId) {
        self.weights.remove(id);
        self.sharded.remove(id);
    }

    fn role(&self, invocation: &KernelInvocation) -> OutputRole {
        let mut role = OutputRole::Replicated;
        for input in &invocation.inputs {
            let id = &input.resource.id;
            match self.weights.get(id) {
                Some((WeightShard::Rows, _)) => return OutputRole::Partial,
                Some((WeightShard::Columns, _)) => role = OutputRole::Sharded,
                None if self.sharded.contains_key(id) => role = OutputRole::Sharded,
                None => {}
            }
        }
        role
    }

    /// `invocation` as `member` runs it: every split input and sharded
    /// output described by its local shape.
    fn local_invocation(
        &self,
        invocation: &KernelInvocation,
        member: usize,
        role: OutputRole,
    ) -> KernelInvocation {
        let mut local = invocation.clone();
        for input in &mut local.inputs {
            let id = input.resource.id.clone();
            let dimensions = &mut input.resource.descriptor.shape.dimensions;
            match self.weights.get(&id) {
                Some((WeightShard::Rows, _)) => {
                    if let Some(first) = dimensions.first_mut() {
                        *first = self.local_extent(member, *first);
                    }
                }
                Some((WeightShard::Columns, _)) => shrink_last(self, dimensions, member),
                None if self.sharded.contains_key(&id) => shrink_last(self, dimensions, member),
                None => {}
            }
        }
        if role == OutputRole::Sharded {
            for output in &mut local.outputs {
                shrink_last(
                    self,
                    &mut output.resource.descriptor.shape.dimensions,
                    member,
                );
            }
        }
        local
    }

    /// `tensor`'s share for `member` along `shard`.
    fn split_host(&self, tensor: &HostTensor, shard: WeightShard, member: usize) -> HostTensor {
        let shape = &tensor.shape;
        match shard {
            WeightShard::Columns => {
                let columns = *shape.last().unwrap_or(&1);
                let (start, end) = self.share(member, columns);
                let rows = (tensor.data.len() as u64 / columns.max(1)) as usize;
                let (columns, start, end) = (columns as usize, start as usize, end as usize);
                let data = (0..rows)
                    .flat_map(|row| tensor.data[row * columns + start..row * columns + end].iter())
                    .copied()
                    .collect();
                let mut local = shape.clone();
                if let Some(last) = local.last_mut() {
                    *last = (end - start) as u64;
                }
                HostTensor { shape: local, data }
            }
            WeightShard::Rows => {
                let rows = *shape.first().unwrap_or(&1);
                let (start, end) = self.share(member, rows);
                let per_row = tensor.data.len() / rows.max(1) as usize;
                let data = tensor.data[start as usize * per_row..end as usize * per_row].to_vec();
                let mut local = shape.clone();
                if let Some(first) = local.first_mut() {
                    *first = end - start;
                }
                HostTensor { shape: local, data }
            }
        }
    }

    /// `weight`'s share for `member` along `shard`, `None` when a share
    /// would cut through a quantization block.
    fn split_packed(
        &self,
        weight: &PackedWeight,
        shard: WeightShard,
        member: usize,
    ) -> Option<PackedWeight> {
        let format = weight.format();
        let [logical_rows, logical_columns] = weight.logical_shape();
        let (extent, along_stored_rows) = match (shard, weight.transposed()) {
            // Logical `[in, out]` stored as `[out, in]`.
            (WeightShard::Columns, true) => (logical_columns, true),
            (WeightShard::Rows, true) => (logical_rows, false),
            (WeightShard::Columns, false) => (logical_columns, false),
            (WeightShard::Rows, false) => (logical_rows, true),
        };
        let (start, end) = self.share(member, extent);
        let block = format.block_elements();
        let row_bytes = (weight.stored_cols() / block * format.block_bytes()) as usize;
        let bytes = weight.bytes();
        if along_stored_rows {
            let local = bytes[start as usize * row_bytes..end as usize * row_bytes].to_vec();
            return PackedWeight::new(
                format,
                end - start,
                weight.stored_cols(),
                weight.transposed(),
                local,
            )
            .ok();
        }
        if !start.is_multiple_of(block) || !end.is_multiple_of(block) {
            return None;
        }
        let (first, last) = (
            (start / block * format.block_bytes()) as usize,
            (end / block * format.block_bytes()) as usize,
        );
        let local = (0..weight.stored_rows() as usize)
            .flat_map(|row| bytes[row * row_bytes + first..row * row_bytes + last].iter())
            .copied()
            .collect();
        PackedWeight::new(
            format,
            weight.stored_rows(),
            end - start,
            weight.transposed(),
            local,
        )
        .ok()
    }
}

/// The alignment of shares along a dimension of `extent`: a `Q8_0` block
/// when every part can hold at least one, else single elements. A function
/// of the extent alone, so the same dimension splits the same way in every
/// tensor that has it.
fn share_granule(extent: u64, parts: usize) -> u64 {
    let block = PackedWeightFormat::Q8_0.block_elements();
    if extent.is_multiple_of(block) && extent >= block * parts as u64 {
        block
    } else {
        1
    }
}

fn shrink_last(tp: &TensorParallel, dimensions: &mut [u64], member: usize) {
    if let Some(last) = dimensions.last_mut() {
        *last = tp.local_extent(member, *last);
    }
}

impl CudaMultiExecutor {
    pub(super) fn tensor_parallel_active(&self) -> bool {
        self.tensor_parallel.lock().unwrap().is_some()
    }

    /// Validates `plan` against the members and puts it in force.
    pub(super) fn take_tensor_parallel_plan(
        &self,
        plan: &TensorParallelPlan,
    ) -> Result<(), ProviderExecutionError> {
        let members: Vec<_> = self
            .members
            .iter()
            .map(|member| member.device_id().clone())
            .collect();
        if plan.devices() != members.as_slice() {
            return Err(self.error(
                ProviderExecutionPhase::Submit,
                format!(
                    "a tensor-parallel plan must span this Provider's Devices in order ({members:?}), \
                     not {:?}",
                    plan.devices()
                ),
            ));
        }
        *self.tensor_parallel.lock().unwrap() = Some(TensorParallel::new(plan.clone()));
        Ok(())
    }

    /// Writes `tensor` as each member's share when `id` is split, else on
    /// every member.
    pub(super) fn write_weight_tensor_parallel(
        &self,
        id: TensorResourceId,
        tensor: HostTensor,
    ) -> Result<(), ProviderExecutionError> {
        let shard = {
            let tp = self.tensor_parallel.lock().unwrap();
            let tp = tp.as_ref().expect("tensor parallelism is active");
            tp.plan.shard_for_weight(id.as_str()).map(|shard| {
                let shares: Vec<_> = (0..self.members.len())
                    .map(|member| tp.split_host(&tensor, shard, member))
                    .collect();
                (shard, shares)
            })
        };
        match shard {
            Some((shard, shares)) => {
                for (member, share) in shares.into_iter().enumerate() {
                    ProviderExecutionApi::write_weight_tensor(
                        &*self.members[member],
                        id.clone(),
                        share,
                    )?;
                }
                self.record_split_weight(id, shard, tensor.shape);
            }
            None => {
                for member in &self.members {
                    ProviderExecutionApi::write_weight_tensor(
                        &**member,
                        id.clone(),
                        tensor.clone(),
                    )?;
                }
            }
        }
        Ok(())
    }

    /// [`Self::write_weight_tensor_parallel`] for a packed weight; declined
    /// (written nowhere) when a share would cut through a block or any
    /// member declines it.
    pub(super) fn write_packed_weight_tensor_parallel(
        &self,
        id: TensorResourceId,
        weight: PackedWeight,
    ) -> Result<PackedWeightWrite, ProviderExecutionError> {
        let (shard, shares) = {
            let tp = self.tensor_parallel.lock().unwrap();
            let tp = tp.as_ref().expect("tensor parallelism is active");
            match tp.plan.shard_for_weight(id.as_str()) {
                Some(shard) => {
                    let shares: Option<Vec<_>> = (0..self.members.len())
                        .map(|member| tp.split_packed(&weight, shard, member))
                        .collect();
                    let Some(shares) = shares else {
                        return Ok(PackedWeightWrite::Declined);
                    };
                    (Some(shard), shares)
                }
                None => (None, vec![weight.clone(); self.members.len()]),
            }
        };
        let mut stored = 0;
        for (member, share) in shares.into_iter().enumerate() {
            if ProviderExecutionApi::write_packed_weight_tensor(
                &*self.members[member],
                id.clone(),
                share,
            )? == PackedWeightWrite::Stored
            {
                stored += 1;
            }
        }
        if stored != self.members.len() {
            for member in &self.members {
                let _ = ProviderExecutionApi::release_tensor(&**member, &id);
            }
            return Ok(PackedWeightWrite::Declined);
        }
        if let Some(shard) = shard {
            self.record_split_weight(id, shard, weight.logical_shape().to_vec());
        }
        Ok(PackedWeightWrite::Stored)
    }

    fn record_split_weight(&self, id: TensorResourceId, shard: WeightShard, shape: Vec<u64>) {
        if let Some(tp) = self.tensor_parallel.lock().unwrap().as_mut() {
            tp.weights.insert(id, (shard, shape));
        }
    }

    /// Runs `invocation` on every member over its shares, then reduces a
    /// partial output. Returns the first member's result.
    pub(super) fn submit_kernel_tensor_parallel(
        &self,
        advertisement: &KernelAdvertisement,
        operator: &OperatorSpec,
        invocation: &KernelInvocation,
        memory: &mut MemoryManager,
    ) -> Result<ProviderExecutionHandle, ProviderExecutionError> {
        let (role, locals) = {
            let tp = self.tensor_parallel.lock().unwrap();
            let tp = tp.as_ref().expect("tensor parallelism is active");
            let role = tp.role(invocation);
            let locals: Vec<_> = (0..self.members.len())
                .map(|member| tp.local_invocation(invocation, member, role))
                .collect();
            (role, locals)
        };
        self.open_member_captures()?;
        let mut first = None;
        let mut failed = None;
        for (member, local) in locals.iter().enumerate() {
            let handle =
                self.members[member].submit_kernel(advertisement, operator, local, memory)?;
            let result = self.members[member].complete_kernel(&handle)?;
            let succeeded = result.status == KernelResultStatus::Succeeded;
            if member == 0 {
                first = Some((handle, result));
            } else if !succeeded {
                failed = Some(result);
            }
            if !succeeded {
                break;
            }
        }
        let (handle, mut result) = first.expect("a Provider has at least one member");
        if let Some(failure) = failed {
            result = failure;
        }
        if result.status == KernelResultStatus::Succeeded {
            if role == OutputRole::Partial {
                // A captured step ends each member's graph here and records
                // the reduction between them.
                self.close_open_member_captures()?;
                for output in &invocation.outputs {
                    self.all_reduce(&output.resource.id)?;
                    if let Some(recording) = self.capture.lock().unwrap().recording.as_mut() {
                        recording.ops.push(CapturedOp::Reduce {
                            id: output.resource.id.clone(),
                        });
                    }
                }
            }
            let mut tp = self.tensor_parallel.lock().unwrap();
            let tp = tp.as_mut().expect("tensor parallelism is active");
            for output in &invocation.outputs {
                let KernelResource { resource, .. } = output;
                if role == OutputRole::Sharded {
                    tp.sharded.insert(
                        resource.id.clone(),
                        resource.descriptor.shape.dimensions.clone(),
                    );
                } else {
                    tp.sharded.remove(&resource.id);
                }
            }
        }
        let ordinal = self
            .next_execution
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let id = ProviderExecutionId::new(format!(
            "{}:tensor-parallel:{ordinal}:{}",
            self.provider_name,
            invocation.id.as_str()
        ));
        self.completed.lock().unwrap().insert(id.clone(), result);
        Ok(ProviderExecutionHandle { id, ..handle })
    }

    /// While capturing, opens every member's own capture if they are closed.
    fn open_member_captures(&self) -> Result<(), ProviderExecutionError> {
        let mut capture = self.capture.lock().unwrap();
        let Some(recording) = capture.recording.as_mut() else {
            return Ok(());
        };
        if recording.all_open {
            return Ok(());
        }
        for (index, member) in self.members.iter().enumerate() {
            if let Err(error) = member.begin_step_capture(&recording.request) {
                for opened in &self.members[..index] {
                    opened.abort_step_capture();
                }
                return Err(error);
            }
        }
        recording.all_open = true;
        Ok(())
    }

    fn close_open_member_captures(&self) -> Result<(), ProviderExecutionError> {
        let mut capture = self.capture.lock().unwrap();
        match capture.recording.as_mut() {
            Some(recording) => self.close_member_captures(recording),
            None => Ok(()),
        }
    }

    /// Ends every member's open capture, in member order, recording each
    /// graph (each runs once as it ends).
    pub(super) fn close_member_captures(
        &self,
        recording: &mut Recording,
    ) -> Result<(), ProviderExecutionError> {
        if !std::mem::take(&mut recording.all_open) {
            return Ok(());
        }
        let mut first_error = None;
        for (member, executor) in self.members.iter().enumerate() {
            if first_error.is_some() {
                executor.abort_step_capture();
                continue;
            }
            match executor.end_step_capture() {
                Ok(id) => recording.ops.push(CapturedOp::Graph { member, id }),
                Err(error) => first_error = Some(error),
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Replays a tensor-parallel capture: each member's graphs and the
    /// reductions between them, in order. Every input is replicated, so
    /// each member's first graph receives all of them.
    pub(super) fn replay_tensor_parallel(
        &self,
        ops: &[CapturedOp],
        offset: u64,
        inputs: Vec<(TensorResourceId, HostTensor)>,
    ) -> Result<(), StepReplayError> {
        let mut pending: Vec<Option<Vec<(TensorResourceId, HostTensor)>>> =
            vec![Some(inputs); self.members.len()];
        for op in ops {
            match op {
                CapturedOp::Graph { member, id } => {
                    let mine = pending[*member].take().unwrap_or_default();
                    self.members[*member].replay_step(id, offset, mine)?;
                }
                CapturedOp::Reduce { id } => {
                    self.all_reduce(id).map_err(StepReplayError::Failed)?;
                }
                CapturedOp::Copy { .. } => {
                    return Err(StepReplayError::Invalidated {
                        reason: "a layer-segment capture outlived its placement".into(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Sums every member's `id` into the first member's, then copies the
    /// sum back to the others, in place. Ordered on the devices only: a
    /// copy between GPUs waits on its source's stream, and the first
    /// member waits for the copies back before it may overwrite `id`; the
    /// host never blocks.
    pub(super) fn all_reduce(&self, id: &TensorResourceId) -> Result<(), ProviderExecutionError> {
        let failed = |detail: String| {
            self.error(
                ProviderExecutionPhase::Submit,
                format!("reducing '{id}' across GPUs: {detail}"),
            )
        };
        let (root, others) = self
            .members
            .split_first()
            .expect("a Provider has at least one member");
        for member in others {
            root.accumulate_from_peer(member, id)
                .map_err(|error| failed(error.to_string()))?;
        }
        for member in others {
            if !member
                .copy_from_peer_in_place(root, id)
                .map_err(|error| failed(error.to_string()))?
            {
                return Err(failed("a member lost its buffer for it".into()));
            }
        }
        for member in others {
            root.wait_for_peer(member)
                .map_err(|error| failed(error.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_runtime::device::DeviceId;

    fn tp(parts: usize) -> TensorParallel {
        let devices = (0..parts)
            .map(|index| DeviceId::new(format!("gpu-{index}")))
            .collect();
        TensorParallel::new(TensorParallelPlan::new(
            devices,
            [
                ("up".to_string(), WeightShard::Columns),
                ("down".to_string(), WeightShard::Rows),
            ],
        ))
    }

    #[test]
    fn dense_shares_of_both_kinds_cover_the_whole_weight() {
        let tp = tp(2);
        let tensor = HostTensor {
            shape: vec![2, 6],
            data: (0..12).map(|value| value as f32).collect(),
        };
        let left = tp.split_host(&tensor, WeightShard::Columns, 0);
        let right = tp.split_host(&tensor, WeightShard::Columns, 1);
        assert_eq!(left.shape, vec![2, 3]);
        assert_eq!(left.data, vec![0.0, 1.0, 2.0, 6.0, 7.0, 8.0]);
        assert_eq!(right.data, vec![3.0, 4.0, 5.0, 9.0, 10.0, 11.0]);
        let top = tp.split_host(&tensor, WeightShard::Rows, 0);
        assert_eq!(top.shape, vec![1, 6]);
        assert_eq!(
            top.data,
            (0..6).map(|value| value as f32).collect::<Vec<_>>()
        );
        let bias = HostTensor {
            shape: vec![6],
            data: (0..6).map(|value| value as f32).collect(),
        };
        assert_eq!(
            tp.split_host(&bias, WeightShard::Columns, 1).data,
            vec![3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn packed_shares_follow_storage_and_respect_blocks() {
        let tp = tp(2);
        let block = PackedWeightFormat::Q8_0.block_bytes() as usize;
        // Stored [4 rows, 64 cols] = 2 blocks per row, transposed: logical
        // [in = 64, out = 4].
        let bytes: Vec<u8> = (0..4 * 2 * block).map(|byte| byte as u8).collect();
        let weight = PackedWeight::new(PackedWeightFormat::Q8_0, 4, 64, true, bytes.clone())
            .expect("valid weight");
        let columns = tp
            .split_packed(&weight, WeightShard::Columns, 1)
            .expect("splits by stored rows");
        assert_eq!(columns.logical_shape(), [64, 2]);
        assert_eq!(columns.bytes(), &bytes[2 * 2 * block..]);
        let rows = tp
            .split_packed(&weight, WeightShard::Rows, 0)
            .expect("splits by whole blocks");
        assert_eq!(rows.logical_shape(), [32, 4]);
        let expected: Vec<u8> = (0..4)
            .flat_map(|row| bytes[row * 2 * block..row * 2 * block + block].iter())
            .copied()
            .collect();
        assert_eq!(rows.bytes(), expected.as_slice());
        // Three parts of 64 values cannot each hold a block.
        assert!(tp_three_parts_decline(&weight));
    }

    fn tp_three_parts_decline(weight: &PackedWeight) -> bool {
        tp(3).split_packed(weight, WeightShard::Rows, 0).is_none()
    }

    #[test]
    fn shares_align_to_blocks_when_the_dimension_allows() {
        let tp = tp(2);
        assert_eq!(share_granule(64, 2), 32);
        assert_eq!(share_granule(48, 2), 1);
        assert_eq!(tp.local_extent(1, 64), 32);
        assert_eq!(tp.local_extent(1, 7), 4);
    }
}
