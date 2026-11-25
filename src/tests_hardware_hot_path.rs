//! Task group 7 (`make-first-native-cuda-hot-path-device-resident`'s
//! Decision 8): a required exit criterion, unlike `tests_conformance.rs`'s
//! per-kernel checks, which bypass the Runtime/Memory Manager/Kernel
//! Registry machinery entirely and call `CudaKernels` directly. This test
//! drives one representative forward-pass-shaped chain -- weight -> MatMul
//! -> RMSNorm -> RoPE -> a projection (MatMul again -- the same Kernel this
//! Runtime uses for every linear projection, `lm_head` included) -- through
//! the exact generic, Provider-agnostic Kernel dispatch contract Qwen's own
//! first-native dispatch uses: `KernelSelectionRequest` ->
//! `KernelRegistry::select` -> `KernelDispatchPlan::from_selection` ->
//! `KernelDispatcher::revalidate` -> `ProviderExecutionApi::submit_kernel`/
//! `complete_kernel`. Reimplemented here at the minimum necessary level
//! because Qwen's own dispatch functions
//! (`magnetar_runtime::first_native_runtime`'s `dispatch_qwen_*`) are
//! private to that crate -- only the generic primitives they are built from
//! are public, and this test uses nothing else.
//!
//! Every intermediate output stays Device-resident (`TensorValue::Opaque`)
//! between steps -- read back and reused by resource id directly, never
//! downloaded -- and only the chain's final output is materialized
//! host-side, to check against an independently-computed Reference CPU
//! expectation.
//!
//! Skips cleanly (returns without assertions) when no compatible CUDA
//! driver/device is present, matching every other hardware-gated test in
//! this crate.

use std::collections::BTreeMap;
use std::sync::Arc;

use magnetar_core::{
    ComputeDType, DTypeDescriptor, DeviceBinding, FallbackClass, KernelMemoryClass,
    LayoutDescriptor, OperatorAttributeValue, OperatorFamily, OperatorId, ProviderBinding,
    ResourceAffinity, ShapeDescriptor, TensorDescriptor, TensorResourceDescriptor,
    TensorResourceId, initial_operator_catalog,
};
use magnetar_graph::{MemoryAllocationOwner, MemoryAllocationState, MemoryPlacement};
use magnetar_kernel::{
    KernelDispatchPlan, KernelDispatchPlanId, KernelInvocationId, KernelResource,
    KernelResultStatus, KernelSelectionRequest,
};
use magnetar_provider_api::{HostTensor, ProviderExecutionApi, TensorValue};
use magnetar_provider_api::{Provider, StepCaptureRequest, StepReplayError};
use magnetar_runtime::Runtime;

use crate::provider::{CUDA_PROVIDER_NAME, CudaProvider};

const TOLERANCE: f32 = 1e-3;

struct ChainFixture {
    runtime: Runtime,
    executor: Arc<dyn ProviderExecutionApi>,
    affinity: ResourceAffinity,
}

fn chain_fixture_or_skip() -> Option<ChainFixture> {
    let provider = CudaProvider::new();
    if !provider.is_available() {
        return None;
    }
    let device_binding = DeviceBinding::new(provider.devices()[0].id().clone());
    let affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new(CUDA_PROVIDER_NAME))
        .with_device(device_binding);
    let executor = provider
        .execution_api()
        .expect("an available CudaProvider always exposes its execution API");
    let runtime = Runtime::builder()
        .register_provider(Arc::new(provider))
        .build()
        .expect("Runtime construction never fails");
    Some(ChainFixture {
        runtime,
        executor,
        affinity,
    })
}

fn descriptor(rows: u64, cols: u64) -> TensorDescriptor {
    TensorDescriptor::new(
        ShapeDescriptor::new([rows, cols]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Contiguous,
    )
}

fn descriptor_1d(len: u64) -> TensorDescriptor {
    TensorDescriptor::new(
        ShapeDescriptor::new([len]),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Contiguous,
    )
}

/// A 1D descriptor declaring a non-`Float32` `ComputeDType` -- used only to
/// select the `add-half`/`mul-half` Kernel Registry candidate
/// (`enable-native-cuda-half-precision-elementwise-compute` Phase 3):
/// `dispatch`'s Kernel Selection Request carries this dtype, independent of
/// what `stage_weight` actually uploaded (always real `f32` host bytes --
/// `run_invocation`'s `"add-half"`/`"mul-half"` arm converts internally).
fn descriptor_half(len: u64, dtype: ComputeDType) -> TensorDescriptor {
    TensorDescriptor::new(
        ShapeDescriptor::new([len]),
        DTypeDescriptor::portable(dtype),
        LayoutDescriptor::Contiguous,
    )
}

/// Writes `tensor` Device-resident under `id`, exactly as
/// `WeightMaterializationTransaction::stage_weight` stages a real Model
/// Instance's weights -- the "weight ->" half of this chain's name.
fn stage_weight(fixture: &mut ChainFixture, id: &TensorResourceId, tensor: HostTensor) {
    fixture
        .executor
        .write_tensor_value_admitted(
            fixture.runtime.memory_mut(),
            id.clone(),
            TensorValue::Host(tensor),
            magnetar_graph::MemoryAllocationClass::Tensor,
            MemoryAllocationOwner::Runtime,
        )
        .expect("staging a small weight Device-resident must succeed");
}

/// Dispatches one Operator through the full generic Kernel dispatch
/// contract: pre-admits `output_id` (mirroring
/// `first_native_runtime::resolve_output_target`'s pre-admission of a
/// caller-named output, the mechanism task group 6 added rollback for),
/// selects and revalidates a real Kernel candidate from the Runtime's
/// Kernel Registry, submits and completes it against the real
/// `CudaExecutor`, and returns the resulting `TensorValue` -- expected to be
/// `Opaque` for every step in this chain (never `ResidencyUnavailable`).
#[allow(clippy::too_many_arguments)]
fn dispatch(
    fixture: &mut ChainFixture,
    step_name: &str,
    operator_name: &str,
    inputs: Vec<(TensorResourceId, TensorDescriptor)>,
    output_id: TensorResourceId,
    output_descriptor: TensorDescriptor,
    attributes: BTreeMap<String, OperatorAttributeValue>,
) -> TensorValue {
    fixture
        .runtime
        .memory_mut()
        .admit_kernel_output(
            output_id.clone(),
            &output_descriptor,
            MemoryPlacement::ProviderOwnedOpaque(ProviderBinding::new(CUDA_PROVIDER_NAME)),
            MemoryAllocationOwner::Runtime,
            fixture.affinity.clone(),
        )
        .unwrap_or_else(|error| panic!("{step_name}: failed to pre-admit its output: {error:?}"));

    let family = match operator_name {
        "matmul" => OperatorFamily::LinearAlgebra,
        "rmsnorm" => OperatorFamily::Normalization,
        "rope" => OperatorFamily::PositionEncoding,
        "add" | "mul" | "embedding" => OperatorFamily::Tensor,
        "attention" => OperatorFamily::Attention,
        "activation" | "gelu" => OperatorFamily::Activation,
        "moe-experts" => OperatorFamily::LinearAlgebra,
        other => panic!("{step_name}: no OperatorFamily mapping for '{other}'"),
    };
    let operator = OperatorId::magnetar(operator_name, 1, family);
    let mut request = KernelSelectionRequest::new(
        format!("hot-path-{step_name}"),
        operator,
        fixture.affinity.clone(),
    );
    for (id, tensor_descriptor) in &inputs {
        // Selection matches each input's dtype, as the engine's
        // `merge_graph_edge_requirements` does: without it an `f32` `mul`
        // ties with `mul-half`, whose stable key sorts first.
        if let DTypeDescriptor::Portable(dtype) = &tensor_descriptor.dtype {
            request.dtype_requirements.insert(*dtype);
        }
        request = request.with_input(KernelResource::new(
            TensorResourceDescriptor::new(
                id.clone(),
                tensor_descriptor.clone(),
                fixture.affinity.clone(),
            ),
            KernelMemoryClass::Device,
        ));
    }
    request = request.with_output(KernelResource::new(
        TensorResourceDescriptor::new(
            output_id.clone(),
            output_descriptor,
            fixture.affinity.clone(),
        ),
        KernelMemoryClass::Device,
    ));

    let selection = fixture
        .runtime
        .kernel_registry()
        .select(&request)
        .unwrap_or_else(|error| panic!("{step_name}: Kernel Registry selection failed: {error}"));
    let candidate = selection
        .selected
        .unwrap_or_else(|| panic!("{step_name}: Kernel Registry selected no candidate"));
    let advertisement = fixture
        .runtime
        .kernel_registry()
        .active_advertisement(&candidate.kernel)
        .unwrap_or_else(|| panic!("{step_name}: selected advertisement is no longer active"))
        .clone();

    let mut plan = KernelDispatchPlan::from_selection(
        KernelDispatchPlanId::new(format!("hot-path-{step_name}-dispatch")),
        &request,
        &candidate,
        &advertisement,
        KernelInvocationId::new(format!("hot-path-{step_name}-invocation")),
    )
    .unwrap_or_else(|error| panic!("{step_name}: dispatch plan construction failed: {error:?}"));
    plan.invocation.attributes = attributes;

    let mut dispatcher = magnetar_kernel::KernelDispatcher::new();
    dispatcher
        .revalidate(fixture.runtime.kernel_registry(), &mut plan)
        .unwrap_or_else(|error| panic!("{step_name}: revalidation failed: {error:?}"));

    let operator_catalog = initial_operator_catalog();
    let operator_spec = operator_catalog
        .get(&advertisement.implemented_operator)
        .unwrap_or_else(|error| panic!("{step_name}: unknown Operator: {error}"));

    let handle = fixture
        .executor
        .submit_kernel(
            &advertisement,
            operator_spec,
            &plan.invocation,
            fixture.runtime.memory_mut(),
        )
        .unwrap_or_else(|error| panic!("{step_name}: submit_kernel failed: {error}"));
    let kernel_result = fixture
        .executor
        .complete_kernel(&handle)
        .unwrap_or_else(|error| panic!("{step_name}: complete_kernel failed: {error}"));
    assert_eq!(
        kernel_result.status,
        KernelResultStatus::Succeeded,
        "{step_name}: Kernel execution did not succeed: {:?}",
        kernel_result.error
    );

    fixture
        .executor
        .read_tensor_value(&output_id)
        .unwrap_or_else(|| panic!("{step_name}: produced no readable output for '{output_id}'"))
}

/// A dispatch that fails before its Kernel runs (here an input that was
/// never materialized) leaves the resource it would have overwritten with
/// its previous value: the executor offers that resource's buffer to the
/// Kernel as its output allocation, and puts it back when no Kernel took it.
#[test]
fn failed_dispatch_keeps_the_overwritten_resource_value_on_real_hardware() {
    let Some(mut fixture) = chain_fixture_or_skip() else {
        return;
    };
    let a = TensorResourceId::new("offer.a");
    let missing = TensorResourceId::new("offer.missing");
    let out = TensorResourceId::new("offer.out");
    let previous = HostTensor::new([1, 3], [7.0, 8.0, 9.0]).unwrap();
    stage_weight(
        &mut fixture,
        &a,
        HostTensor::new([1, 3], [1.0, 2.0, 3.0]).unwrap(),
    );
    stage_weight(&mut fixture, &out, previous.clone());

    let mut request = KernelSelectionRequest::new(
        "offer-add",
        OperatorId::magnetar("add", 1, OperatorFamily::Tensor),
        fixture.affinity.clone(),
    );
    for id in [&a, &missing] {
        request = request.with_input(KernelResource::new(
            TensorResourceDescriptor::new(id.clone(), descriptor(1, 3), fixture.affinity.clone()),
            KernelMemoryClass::Device,
        ));
    }
    request = request.with_output(KernelResource::new(
        TensorResourceDescriptor::new(out.clone(), descriptor(1, 3), fixture.affinity.clone()),
        KernelMemoryClass::Device,
    ));
    let candidate = fixture
        .runtime
        .kernel_registry()
        .select(&request)
        .unwrap()
        .selected
        .unwrap();
    let advertisement = fixture
        .runtime
        .kernel_registry()
        .active_advertisement(&candidate.kernel)
        .unwrap()
        .clone();
    let plan = KernelDispatchPlan::from_selection(
        KernelDispatchPlanId::new("offer-add-dispatch"),
        &request,
        &candidate,
        &advertisement,
        KernelInvocationId::new("offer-add-invocation"),
    )
    .unwrap();
    let catalog = initial_operator_catalog();
    let handle = fixture
        .executor
        .submit_kernel(
            &advertisement,
            catalog.get(&advertisement.implemented_operator).unwrap(),
            &plan.invocation,
            fixture.runtime.memory_mut(),
        )
        .unwrap();
    let result = fixture.executor.complete_kernel(&handle).unwrap();
    assert_ne!(result.status, KernelResultStatus::Succeeded);
    let kept = fixture
        .executor
        .read_tensor(&out)
        .expect("the previous value is still stored");
    assert_eq!(kept.data, previous.data);
}

/// The full chain, run entirely on real CUDA hardware: `weight -> MatMul ->
/// RMSNorm -> RoPE -> a projection`. Every step's `ResourceAffinity` is the
/// same CUDA/GPU0 affinity throughout (`fixture.affinity`, reused
/// unmodified for every `admit_kernel_output`/`KernelSelectionRequest`
/// call) -- task group 1's production `validate_invocation_provider_
/// matches_affinity` check (exercised on every dispatch here through the
/// real `KernelDispatchPlan`) never has anything to reject.
#[test]
fn weight_matmul_rmsnorm_rope_projection_chain_runs_device_resident_on_real_hardware() {
    let Some(mut fixture) = chain_fixture_or_skip() else {
        return;
    };

    // Small, hand-computable shapes: 2 "tokens", hidden size 4.
    let activation = HostTensor::new([2, 4], [1.0, 0.5, -0.5, 2.0, 0.25, -1.0, 1.5, 0.75]).unwrap();
    let projection_weight = HostTensor::new(
        [4, 4],
        (0..16).map(|i| (i as f32) * 0.05 - 0.4).collect::<Vec<_>>(),
    )
    .unwrap();
    let rmsnorm_weight = HostTensor::new([4], [1.0, 1.0, 1.0, 1.0]).unwrap();
    let output_projection_weight = HostTensor::new(
        [4, 3],
        (0..12).map(|i| (i as f32) * 0.1 - 0.5).collect::<Vec<_>>(),
    )
    .unwrap();

    let activation_id = TensorResourceId::new("hot-path.activation");
    let projection_weight_id = TensorResourceId::new("hot-path.weight.matmul");
    let rmsnorm_weight_id = TensorResourceId::new("hot-path.weight.rmsnorm");
    let output_projection_weight_id = TensorResourceId::new("hot-path.weight.output-projection");

    stage_weight(&mut fixture, &activation_id, activation.clone());
    stage_weight(
        &mut fixture,
        &projection_weight_id,
        projection_weight.clone(),
    );
    stage_weight(&mut fixture, &rmsnorm_weight_id, rmsnorm_weight.clone());
    stage_weight(
        &mut fixture,
        &output_projection_weight_id,
        output_projection_weight.clone(),
    );

    // Step 1: weight -> MatMul. [2,4] x [4,4] -> [2,4].
    let matmul_output_id = TensorResourceId::new("hot-path.matmul.out");
    let matmul_output = dispatch(
        &mut fixture,
        "matmul",
        "matmul",
        vec![
            (activation_id.clone(), descriptor(2, 4)),
            (projection_weight_id.clone(), descriptor(4, 4)),
        ],
        matmul_output_id.clone(),
        descriptor(2, 4),
        BTreeMap::new(),
    );
    assert!(
        matches!(matmul_output, TensorValue::Opaque),
        "matmul: output must stay Device-resident (Opaque), not be forced host-visible"
    );

    // Step 2: RMSNorm. [2,4] with a [4] weight -> [2,4], read back directly
    // by resource id -- no download/reupload between MatMul and RMSNorm.
    let rmsnorm_output_id = TensorResourceId::new("hot-path.rmsnorm.out");
    let rmsnorm_output = dispatch(
        &mut fixture,
        "rmsnorm",
        "rmsnorm",
        vec![
            (matmul_output_id.clone(), descriptor(2, 4)),
            (rmsnorm_weight_id.clone(), descriptor_1d(4)),
        ],
        rmsnorm_output_id.clone(),
        descriptor(2, 4),
        BTreeMap::from([("epsilon".to_string(), OperatorAttributeValue::Float(1e-6))]),
    );
    assert!(
        matches!(rmsnorm_output, TensorValue::Opaque),
        "rmsnorm: output must stay Device-resident (Opaque)"
    );

    // Step 3: RoPE. Single-block (no `head_count` attribute, defaulting to
    // 1 -- `make-first-native-cuda-hot-path-device-resident` task group 3's
    // GQA-correct semantics), full rotation across all 4 columns.
    let rope_output_id = TensorResourceId::new("hot-path.rope.out");
    let rope_output = dispatch(
        &mut fixture,
        "rope",
        "rope",
        vec![(rmsnorm_output_id.clone(), descriptor(2, 4))],
        rope_output_id.clone(),
        descriptor(2, 4),
        BTreeMap::from([
            ("base".to_string(), OperatorAttributeValue::Float(10000.0)),
            ("dimension".to_string(), OperatorAttributeValue::Integer(4)),
        ]),
    );
    assert!(
        matches!(rope_output, TensorValue::Opaque),
        "rope: output must stay Device-resident (Opaque)"
    );

    // Step 4: a projection -- MatMul again, [2,4] x [4,3] -> [2,3], the
    // chain's final output. Downloaded once here, its only host crossing.
    let final_output_id = TensorResourceId::new("hot-path.projection.out");
    let final_output = dispatch(
        &mut fixture,
        "projection",
        "matmul",
        vec![
            (rope_output_id.clone(), descriptor(2, 4)),
            (output_projection_weight_id.clone(), descriptor(4, 3)),
        ],
        final_output_id.clone(),
        descriptor(2, 3),
        BTreeMap::new(),
    );
    assert!(
        matches!(final_output, TensorValue::Opaque),
        "projection: output stays Device-resident like every other step; the explicit \
         `read_tensor` call below (not `read_tensor_value`/`into_host`) is this chain's \
         one deliberate, final host crossing -- `CudaExecutor::read_tensor_value` always \
         answers Opaque by design (`define-provider-prepared-kernel-execution-contract`), \
         only the HostTensor-typed `read_tensor` actually downloads"
    );
    let actual = fixture
        .executor
        .read_tensor(&final_output_id)
        .expect("the chain's final output must be downloadable on request");

    // Independent Reference CPU oracle for the exact same four steps.
    let expected_matmul =
        magnetar_provider_cpu::matmul(&activation, &projection_weight, false, false).unwrap();
    let expected_rmsnorm =
        magnetar_provider_cpu::rmsnorm(&expected_matmul, &rmsnorm_weight, 1e-6).unwrap();
    let expected_rope =
        magnetar_provider_cpu::rope(&expected_rmsnorm, 10000.0, 1.0, 4, 0, 1).unwrap();
    let expected =
        magnetar_provider_cpu::matmul(&expected_rope, &output_projection_weight, false, false)
            .unwrap();
    assert_eq!(actual.shape, expected.shape);
    for (index, (a, e)) in actual.data.iter().zip(&expected.data).enumerate() {
        assert!(
            (a - e).abs() <= TOLERANCE,
            "element {index}: cuda={a} reference-cpu={e} exceeds tolerance {TOLERANCE}"
        );
    }

    // task 7.4: ResourceAffinity/MemoryPlacement correctly reflect
    // CUDA/GPU0 throughout -- checked directly against the Memory
    // Manager's own residency ledger for every resource this chain
    // produced, not just assumed from the dispatch succeeding.
    for id in [
        &matmul_output_id,
        &rmsnorm_output_id,
        &rope_output_id,
        &final_output_id,
    ] {
        let residency = fixture
            .runtime
            .memory()
            .tensor_residency(id)
            .unwrap_or_else(|| panic!("'{id}' must have a recorded residency"));
        assert_eq!(
            residency.affinity.provider(),
            Some(&ProviderBinding::new(CUDA_PROVIDER_NAME)),
            "'{id}': ResourceAffinity must name the CUDA Provider"
        );
        assert!(
            matches!(
                residency.placement,
                MemoryPlacement::ProviderOwnedOpaque(ref binding) if binding.as_str() == CUDA_PROVIDER_NAME
            ),
            "'{id}': MemoryPlacement must be CUDA-Provider-owned, got {:?}",
            residency.placement
        );
    }

    // No leak after a successful run: every allocation this chain admitted
    // is present and Active (nothing was silently dropped or double-freed).
    let active_count = fixture
        .runtime
        .memory()
        .allocations()
        .filter(|allocation| allocation.state == MemoryAllocationState::Active)
        .count();
    assert_eq!(
        active_count, 8,
        "expected exactly the 4 staged weights/activation plus the 4 chain outputs to be \
         Active after a successful run"
    );
}

/// `enable-native-cuda-half-precision-elementwise-compute` Phase 3: proves
/// `add_half`/`mul_half` (real and hardware-verified against an exact
/// reference model since Phase 2, but previously reachable only by calling
/// `CudaKernels` methods directly) are now selectable and dispatchable
/// through the *exact same* generic Kernel Registry/dispatch contract this
/// file's other tests already prove for the `f32`-only path: a
/// `KernelSelectionRequest` declaring `Float16`/`BrainFloat16` resources
/// must cause the Kernel Registry to select the new `add-half`/`mul-half`
/// advertisement over the existing `f32`-only `add`/`mul` one, and the
/// resulting dispatch must produce the same bit-exact result Phase 2's
/// direct-call tests already established.
#[test]
fn half_precision_add_and_mul_dispatch_through_the_real_kernel_registry_on_real_hardware() {
    let Some(mut fixture) = chain_fixture_or_skip() else {
        return;
    };

    let a_data = vec![1.0, 2.5, 3.0, -4.25];
    let b_data = vec![6.0, 0.2, 4.0, 3.0];
    let a = HostTensor::new([4], a_data.clone()).unwrap();
    let b = HostTensor::new([4], b_data.clone()).unwrap();

    for (dtype, op_name, op) in [
        (
            ComputeDType::Float16,
            "add",
            (|x: f32, y: f32| x + y) as fn(f32, f32) -> f32,
        ),
        (ComputeDType::Float16, "mul", |x, y| x * y),
        (ComputeDType::BrainFloat16, "add", |x, y| x + y),
        (ComputeDType::BrainFloat16, "mul", |x, y| x * y),
    ] {
        let a_id = TensorResourceId::new(format!("half-dispatch.{op_name}.{dtype:?}.a"));
        let b_id = TensorResourceId::new(format!("half-dispatch.{op_name}.{dtype:?}.b"));
        stage_weight(&mut fixture, &a_id, a.clone());
        stage_weight(&mut fixture, &b_id, b.clone());

        let output_id = TensorResourceId::new(format!("half-dispatch.{op_name}.{dtype:?}.out"));
        let output = dispatch(
            &mut fixture,
            &format!("{op_name}-half-{dtype:?}"),
            op_name,
            vec![
                (a_id.clone(), descriptor_half(4, dtype)),
                (b_id.clone(), descriptor_half(4, dtype)),
            ],
            output_id.clone(),
            descriptor_half(4, dtype),
            BTreeMap::new(),
        );
        assert!(
            matches!(output, TensorValue::Opaque),
            "{op_name}-half ({dtype:?}): output must stay Device-resident (Opaque)"
        );

        let actual = fixture
            .executor
            .read_tensor(&output_id)
            .expect("the half-precision result must be downloadable on request");

        let expected = half_reference(&a_data, &b_data, dtype, op);
        assert_eq!(
            actual.data, expected,
            "{op_name}-half ({dtype:?}): dispatched-through-the-registry result must match \
             the exact reference conversion model, same as Phase 2's direct-call tests"
        );
    }
}

/// Same exact-reference-model technique as `tests_conformance.rs`'s
/// `half_expected`: the mathematically correct answer for an `op` on
/// already-half-precision-rounded inputs is
/// `decode(encode(decode(encode(a)) op decode(encode(b))))`, computed here
/// via this crate's own `half_precision` module (the same one
/// `CudaKernels::upload_half`/`download_half` use).
type HalfEncoder = fn(f32) -> u16;
type HalfDecoder = fn(u16) -> f32;

fn half_reference(a: &[f32], b: &[f32], dtype: ComputeDType, op: fn(f32, f32) -> f32) -> Vec<f32> {
    let (encode, decode): (HalfEncoder, HalfDecoder) = match dtype {
        ComputeDType::Float16 => (
            crate::half_precision::f32_to_f16,
            crate::half_precision::f16_to_f32,
        ),
        ComputeDType::BrainFloat16 => (
            crate::half_precision::f32_to_bf16,
            crate::half_precision::bf16_to_f32,
        ),
        other => panic!("half_reference: unsupported dtype {other:?}"),
    };
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let x = decode(encode(x));
            let y = decode(encode(y));
            decode(encode(op(x, y)))
        })
        .collect()
}

/// `capture-and-replay-cuda-graphs`: a captured `rope -> rmsnorm` step replays
/// with its RoPE position advanced by the replay offset and its input
/// rewritten in place, matching uncaptured dispatches exactly; a replaced
/// input invalidates the capture, and a host upload while capturing fails
/// the capture without leaving the stream capturing.
#[test]
fn a_captured_step_replays_with_an_advanced_position_on_real_hardware() {
    let Some(mut fixture) = chain_fixture_or_skip() else {
        return;
    };
    let x = TensorResourceId::new("replay.x");
    let bias = TensorResourceId::new("replay.norm.weight");
    let rotated = TensorResourceId::new("replay.rope.out");
    let sum = TensorResourceId::new("replay.sum.out");
    let first = HostTensor::new([2, 4], vec![0.5, -1.0, 2.0, 0.25, 1.5, 3.0, -0.75, 1.0]).unwrap();
    let second = HostTensor::new([2, 4], vec![-2.0, 0.5, 1.0, 4.0, 0.0, -1.5, 2.5, 0.75]).unwrap();
    stage_weight(
        &mut fixture,
        &bias,
        HostTensor::new([4], vec![0.5, 1.0, 1.5, 2.0]).unwrap(),
    );
    let step = |fixture: &mut ChainFixture, position: i64| {
        dispatch(
            fixture,
            "replay-rope",
            "rope",
            vec![(x.clone(), descriptor(2, 4))],
            rotated.clone(),
            descriptor(2, 4),
            BTreeMap::from([
                ("base".to_string(), OperatorAttributeValue::Float(10000.0)),
                ("dimension".to_string(), OperatorAttributeValue::Integer(4)),
                (
                    "position_offset".to_string(),
                    OperatorAttributeValue::Integer(position),
                ),
            ]),
        );
        dispatch(
            fixture,
            "replay-rmsnorm",
            "rmsnorm",
            vec![
                (rotated.clone(), descriptor(2, 4)),
                (bias.clone(), descriptor_1d(4)),
            ],
            sum.clone(),
            descriptor(2, 4),
            BTreeMap::from([("epsilon".to_string(), OperatorAttributeValue::Float(1e-6))]),
        );
        fixture.executor.read_tensor(&sum).unwrap().data
    };
    stage_weight(&mut fixture, &x, second.clone());
    let second_at_7 = step(&mut fixture, 7);
    stage_weight(&mut fixture, &x, first.clone());
    let first_at_3 = step(&mut fixture, 3);
    assert_ne!(first_at_3, second_at_7);

    let request = StepCaptureRequest {
        advancing_attributes: vec!["position_offset".into()],
    };
    fixture.executor.begin_step_capture(&request).unwrap();
    step_without_reading(&mut fixture, &x, &rotated, &bias, &sum, 3);
    let capture = fixture.executor.end_step_capture().unwrap();
    assert_eq!(fixture.executor.read_tensor(&sum).unwrap().data, first_at_3);

    fixture
        .executor
        .replay_step(&capture, 4, vec![(x.clone(), second.clone())])
        .unwrap();
    assert_eq!(
        fixture.executor.read_tensor(&sum).unwrap().data,
        second_at_7
    );
    fixture
        .executor
        .replay_step(&capture, 0, vec![(x.clone(), first.clone())])
        .unwrap();
    assert_eq!(fixture.executor.read_tensor(&sum).unwrap().data, first_at_3);

    // A fresh upload gives the input another buffer.
    stage_weight(&mut fixture, &x, first.clone());
    assert!(matches!(
        fixture
            .executor
            .replay_step(&capture, 0, vec![(x.clone(), first.clone())]),
        Err(StepReplayError::Invalidated { .. })
    ));
    fixture.executor.release_step_capture(&capture);

    fixture.executor.begin_step_capture(&request).unwrap();
    assert!(
        fixture
            .executor
            .write_tensor_value(x.clone(), TensorValue::Host(first.clone()))
            .is_err(),
        "a host upload cannot be captured"
    );
    assert!(fixture.executor.end_step_capture().is_err());
    // The stream is usable again.
    assert_eq!(step(&mut fixture, 3), first_at_3);
}

/// `replay-bucketed-prefill`: a captured multi-row (prefill) attention
/// over the first `key_count` rows of a larger store replays with new
/// queries exactly like an uncaptured dispatch, for each head width with a
/// Tensor Core path (and its tiled fallback below 16 rows). It has no
/// replay offset, so a replay at any other offset is refused.
#[test]
fn a_captured_prefill_attention_replays_at_its_captured_key_count_on_real_hardware() {
    let Some(mut fixture) = chain_fixture_or_skip() else {
        return;
    };
    for (rows, head_dimension) in [(16_u64, 64_u64), (32, 128), (8, 64)] {
        let width = 2 * head_dimension;
        let capacity = 4 * rows;
        let prefix = format!("prefill-attention.{rows}.{head_dimension}");
        let q = TensorResourceId::new(format!("{prefix}.q"));
        let k = TensorResourceId::new(format!("{prefix}.k"));
        let v = TensorResourceId::new(format!("{prefix}.v"));
        let out = TensorResourceId::new(format!("{prefix}.out"));
        let values = |count: u64, seed: u64| {
            (0..count)
                .map(|index| (((index * 37 + seed * 11) % 97) as f32 - 48.0) / 64.0)
                .collect::<Vec<f32>>()
        };
        let first = HostTensor::new([rows, width], values(rows * width, 1)).unwrap();
        let second = HostTensor::new([rows, width], values(rows * width, 2)).unwrap();
        stage_weight(
            &mut fixture,
            &k,
            HostTensor::new([capacity, width], values(capacity * width, 3)).unwrap(),
        );
        stage_weight(
            &mut fixture,
            &v,
            HostTensor::new([capacity, width], values(capacity * width, 4)).unwrap(),
        );
        let attention = |fixture: &mut ChainFixture| {
            dispatch(
                fixture,
                "prefill-attention",
                "attention",
                vec![
                    (q.clone(), descriptor(rows, width)),
                    (k.clone(), descriptor(capacity, width)),
                    (v.clone(), descriptor(capacity, width)),
                ],
                out.clone(),
                descriptor(rows, width),
                BTreeMap::from([
                    ("head_count".to_string(), OperatorAttributeValue::Integer(2)),
                    (
                        "head_dimension".to_string(),
                        OperatorAttributeValue::Integer(head_dimension as i64),
                    ),
                    ("causal".to_string(), OperatorAttributeValue::Boolean(true)),
                    (
                        "key_count".to_string(),
                        OperatorAttributeValue::Integer(rows as i64),
                    ),
                ]),
            );
        };
        stage_weight(&mut fixture, &q, second.clone());
        attention(&mut fixture);
        let expected_second = fixture.executor.read_tensor(&out).unwrap().data;
        stage_weight(&mut fixture, &q, first.clone());
        attention(&mut fixture);
        let expected_first = fixture.executor.read_tensor(&out).unwrap().data;
        assert_ne!(expected_first, expected_second);

        let request = StepCaptureRequest {
            advancing_attributes: vec!["key_count".into()],
        };
        fixture.executor.begin_step_capture(&request).unwrap();
        attention(&mut fixture);
        let capture = fixture
            .executor
            .end_step_capture()
            .unwrap_or_else(|error| panic!("{rows} rows of {head_dimension}: {error:?}"));
        assert_eq!(
            fixture.executor.read_tensor(&out).unwrap().data,
            expected_first
        );
        fixture
            .executor
            .replay_step(&capture, 0, vec![(q.clone(), second.clone())])
            .unwrap();
        assert_eq!(
            fixture.executor.read_tensor(&out).unwrap().data,
            expected_second,
            "{rows} rows of {head_dimension}"
        );
        assert!(matches!(
            fixture
                .executor
                .replay_step(&capture, 1, vec![(q.clone(), first.clone())]),
            Err(StepReplayError::Invalidated { .. })
        ));
        fixture.executor.release_step_capture(&capture);
    }
}

/// The captured dispatches of [`a_captured_step_replays_with_an_advanced_position_on_real_hardware`]:
/// nothing may be downloaded while capturing.
fn step_without_reading(
    fixture: &mut ChainFixture,
    x: &TensorResourceId,
    rotated: &TensorResourceId,
    bias: &TensorResourceId,
    sum: &TensorResourceId,
    position: i64,
) {
    dispatch(
        fixture,
        "replay-rope",
        "rope",
        vec![(x.clone(), descriptor(2, 4))],
        rotated.clone(),
        descriptor(2, 4),
        BTreeMap::from([
            ("base".to_string(), OperatorAttributeValue::Float(10000.0)),
            ("dimension".to_string(), OperatorAttributeValue::Integer(4)),
            (
                "position_offset".to_string(),
                OperatorAttributeValue::Integer(position),
            ),
        ]),
    );
    dispatch(
        fixture,
        "replay-rmsnorm",
        "rmsnorm",
        vec![
            (rotated.clone(), descriptor(2, 4)),
            (bias.clone(), descriptor_1d(4)),
        ],
        sum.clone(),
        descriptor(2, 4),
        BTreeMap::from([("epsilon".to_string(), OperatorAttributeValue::Float(1e-6))]),
    );
}

/// A captured `rope -> rmsnorm` step over `rows x width` values, as the
/// retirement and memory tests of a capture run it.
fn large_step(
    fixture: &mut ChainFixture,
    x: &TensorResourceId,
    rotated: &TensorResourceId,
    weight: &TensorResourceId,
    sum: &TensorResourceId,
    (rows, width): (u64, u64),
) {
    dispatch(
        fixture,
        "large-rope",
        "rope",
        vec![(x.clone(), descriptor(rows, width))],
        rotated.clone(),
        descriptor(rows, width),
        BTreeMap::from([
            ("base".to_string(), OperatorAttributeValue::Float(10000.0)),
            (
                "dimension".to_string(),
                OperatorAttributeValue::Integer(width as i64),
            ),
            (
                "position_offset".to_string(),
                OperatorAttributeValue::Integer(3),
            ),
        ]),
    );
    dispatch(
        fixture,
        "large-rmsnorm",
        "rmsnorm",
        vec![
            (rotated.clone(), descriptor(rows, width)),
            (weight.clone(), descriptor_1d(width)),
        ],
        sum.clone(),
        descriptor(rows, width),
        BTreeMap::from([("epsilon".to_string(), OperatorAttributeValue::Float(1e-6))]),
    );
}

fn large_values(count: u64, seed: u64) -> Vec<f32> {
    (0..count)
        .map(|index| (((index * 37 + seed * 11) % 97) as f32 - 48.0) / 64.0)
        .collect()
}

/// `capture-and-replay-cuda-graphs` 3.3, lease retirement: the buffers a
/// replay reads can be retired, and their memory reused, the moment the
/// replay is launched, before it completes. The Provider frees in stream
/// order, so the replay still computes what it was captured to compute, and
/// a capture whose resources are gone refuses to replay.
#[test]
fn a_replay_in_flight_survives_the_retirement_of_its_resources_on_real_hardware() {
    let Some(mut fixture) = chain_fixture_or_skip() else {
        return;
    };
    let shape = (1024_u64, 2048_u64);
    let (rows, width) = shape;
    let input = HostTensor::new([rows, width], large_values(rows * width, 1)).unwrap();
    let garbage = HostTensor::new([rows, width], large_values(rows * width, 2)).unwrap();
    let weight = TensorResourceId::new("retire.norm.weight");
    stage_weight(
        &mut fixture,
        &weight,
        HostTensor::new([width], large_values(width, 3)).unwrap(),
    );
    for iteration in 0..8 {
        let id = |name: &str| TensorResourceId::new(format!("retire.{iteration}.{name}"));
        let (x, rotated, sum) = (id("x"), id("rotated"), id("sum"));
        stage_weight(&mut fixture, &x, input.clone());
        large_step(&mut fixture, &x, &rotated, &weight, &sum, shape);
        let expected = fixture.executor.read_tensor(&sum).unwrap().data;

        let request = StepCaptureRequest {
            advancing_attributes: vec!["position_offset".into()],
        };
        fixture.executor.begin_step_capture(&request).unwrap();
        large_step(&mut fixture, &x, &rotated, &weight, &sum, shape);
        let capture = fixture.executor.end_step_capture().unwrap();

        // Launch, then retire what the replay reads at once, and reuse its
        // memory before anything waits for the replay.
        fixture
            .executor
            .replay_step(&capture, 0, vec![(x.clone(), input.clone())])
            .unwrap();
        let memory = fixture.runtime.memory_mut();
        assert!(
            fixture
                .executor
                .release_admitted_tensor(memory, &x)
                .unwrap()
        );
        assert!(
            fixture
                .executor
                .release_admitted_tensor(memory, &rotated)
                .unwrap()
        );
        let reused: Vec<TensorResourceId> = (0..4).map(|k| id(&format!("garbage.{k}"))).collect();
        for resource in &reused {
            stage_weight(&mut fixture, resource, garbage.clone());
        }
        assert_eq!(
            fixture.executor.read_tensor(&sum).unwrap().data,
            expected,
            "iteration {iteration}: the replay's result changed after its inputs were retired"
        );
        assert!(
            matches!(
                fixture.executor.replay_step(&capture, 0, vec![]),
                Err(StepReplayError::Invalidated { .. })
            ),
            "iteration {iteration}: a capture replayed against resources that are gone"
        );
        fixture.executor.release_step_capture(&capture);
        let memory = fixture.runtime.memory_mut();
        for resource in reused.iter().chain([&sum]) {
            fixture
                .executor
                .release_admitted_tensor(memory, resource)
                .unwrap();
        }
    }
}

/// `capture-and-replay-cuda-graphs` 1.2: the Device memory a captured step
/// retains (its graph executable), and what releasing the capture gives
/// back. One `rope -> rmsnorm` step is two Kernels, below the granularity
/// the driver allocates in, so the same step is also captured 128 and 1,024
/// times in one graph (a whole-model decode step is about 315 launches).
/// Diagnostic: the free memory of a Device other processes also use is not
/// exact, so it reports and only bounds the retained bytes.
#[test]
#[ignore = "diagnostic: reports the Device memory a captured step holds"]
fn captured_step_device_memory() {
    let Some(mut fixture) = chain_fixture_or_skip() else {
        return;
    };
    let context = cudarc::driver::CudaContext::new(0).expect("the first CUDA Device");
    let free = || context.mem_get_info().expect("Device memory info").0 as i64;
    let shape = (1024_u64, 2048_u64);
    let (rows, width) = shape;
    let (x, rotated, sum, weight) = (
        TensorResourceId::new("memory.x"),
        TensorResourceId::new("memory.rotated"),
        TensorResourceId::new("memory.sum"),
        TensorResourceId::new("memory.norm.weight"),
    );
    stage_weight(
        &mut fixture,
        &x,
        HostTensor::new([rows, width], large_values(rows * width, 1)).unwrap(),
    );
    stage_weight(
        &mut fixture,
        &weight,
        HostTensor::new([width], large_values(width, 3)).unwrap(),
    );
    // One uncaptured step first: its kernels are loaded and its buffers
    // allocated, so the capture's own cost is what the difference shows.
    large_step(&mut fixture, &x, &rotated, &weight, &sum, shape);
    let kib = |bytes: i64| bytes as f64 / 1024.0;
    for repeats in [1_u32, 128, 1024] {
        let before = free();
        let request = StepCaptureRequest {
            advancing_attributes: vec!["position_offset".into()],
        };
        fixture.executor.begin_step_capture(&request).unwrap();
        for _ in 0..repeats {
            large_step(&mut fixture, &x, &rotated, &weight, &sum, shape);
        }
        let capture = fixture.executor.end_step_capture().unwrap();
        let held = free();
        fixture.executor.release_step_capture(&capture);
        let released = free();
        eprintln!(
            "captured rope -> rmsnorm x {repeats} ({} Kernels, {rows} x {width}): {:.0} KiB of Device \
             memory retained by the capture (free {:.1} -> {:.1} MiB), {:.0} KiB given back by its release",
            repeats * 2,
            kib(before - held),
            before as f64 / 1048576.0,
            held as f64 / 1048576.0,
            kib(released - held),
        );
        assert!(
            before - held < 64 * 1024 * 1024,
            "a {repeats}-step capture retains {} bytes of Device memory",
            before - held
        );
    }
}

/// `execute-native-half-precision`: a weight written as one is stored
/// compact, and a Kernel argument that only reads f32 gets it expanded,
/// with the same result as the weight written as an ordinary tensor.
#[test]
fn a_compact_weight_read_as_f32_is_expanded_on_real_hardware() {
    let Some(mut fixture) = chain_fixture_or_skip() else {
        return;
    };
    let (rows, cols) = (256u64, 256u64);
    let values: Vec<f32> = (0..rows * cols)
        .map(|index| f32::from_bits(((index as f32 * 0.01).sin()).to_bits() & 0xFFFF_0000))
        .collect();
    let weight = HostTensor::new([rows, cols], values).unwrap();
    let compact = TensorResourceId::new("compact.weight");
    let plain = TensorResourceId::new("plain.weight");
    let x = TensorResourceId::new("compact.x");
    fixture
        .executor
        .write_weight_tensor(compact.clone(), weight.clone())
        .unwrap();
    stage_weight(&mut fixture, &plain, weight);
    stage_weight(
        &mut fixture,
        &x,
        HostTensor::new(
            [cols, 2],
            (0..cols * 2).map(|i| i as f32 * 1e-3).collect::<Vec<_>>(),
        )
        .unwrap(),
    );
    let mut product = |a: &TensorResourceId, out: &str| {
        let out = TensorResourceId::new(out);
        dispatch(
            &mut fixture,
            out.as_str(),
            "matmul",
            vec![
                (a.clone(), descriptor(rows, cols)),
                (x.clone(), descriptor(cols, 2)),
            ],
            out.clone(),
            descriptor(rows, 2),
            BTreeMap::new(),
        );
        fixture.executor.read_tensor(&out).unwrap().data
    };
    // The weight is matmul's `a`, which only reads f32.
    assert_eq!(
        product(&compact, "compact.out"),
        product(&plain, "plain.out")
    );
}

/// `dispatch-layered-model-forms`: every optional form of the layered
/// models' Operators, dispatched through the Kernel Registry and the real
/// `CudaExecutor` with its node attributes, matches the Reference CPU.
#[test]
fn layered_operator_forms_dispatch_through_the_executor_on_real_hardware() {
    use magnetar_provider_api::operator_semantics;
    let Some(mut fixture) = chain_fixture_or_skip() else {
        return;
    };
    let tensor = |rows: u64, cols: u64, seed: u32| {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| {
                (((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) % 1000) as f32) / 500.0
                    - 1.0
            })
            .collect();
        HostTensor::new([rows, cols], data).unwrap()
    };
    let float = |value: f64| OperatorAttributeValue::Float(value);
    let int = |value: i64| OperatorAttributeValue::Integer(value);
    let text = |value: &str| OperatorAttributeValue::String(value.into());
    let check = |fixture: &mut ChainFixture,
                 name: &str,
                 operator: &str,
                 inputs: Vec<(&str, HostTensor)>,
                 attributes: Vec<(&str, OperatorAttributeValue)>,
                 expected: HostTensor| {
        let mut resources = Vec::new();
        for (input, value) in inputs {
            let id = TensorResourceId::new(format!("layered-{name}-{input}"));
            let descriptor = match value.shape.as_slice() {
                [len] => descriptor_1d(*len),
                [rows, cols] => descriptor(*rows, *cols),
                other => panic!("{name}: unexpected shape {other:?}"),
            };
            stage_weight(fixture, &id, value);
            resources.push((id, descriptor));
        }
        let output = TensorResourceId::new(format!("layered-{name}-out"));
        let [rows, cols] = expected.shape[..] else {
            panic!("{name}: rank-2 output expected");
        };
        dispatch(
            fixture,
            name,
            operator,
            resources,
            output.clone(),
            descriptor(rows, cols),
            attributes
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
        );
        let actual = fixture.executor.read_tensor(&output).unwrap();
        assert_eq!(actual.shape, expected.shape, "{name}");
        for (index, (a, e)) in actual.data.iter().zip(&expected.data).enumerate() {
            assert!(
                (a - e).abs() <= TOLERANCE,
                "{name}, element {index}: cuda={a} reference={e}"
            );
        }
    };

    // embedding, scaled.
    let table = tensor(6, 8, 1);
    let ids = HostTensor::new([3], vec![4.0, 0.0, 5.0]).unwrap();
    let gathered = magnetar_provider_cpu::embedding_lookup(&table, &ids).unwrap();
    check(
        &mut fixture,
        "embedding",
        "embedding",
        vec![("table", table), ("ids", ids)],
        vec![("scale", float(2.0))],
        operator_semantics::scale(&gathered, 2.0).unwrap(),
    );
    // rmsnorm per head, without a weight.
    let x = tensor(3, 8, 2);
    check(
        &mut fixture,
        "rmsnorm",
        "rmsnorm",
        vec![("input", x.clone())],
        vec![
            ("epsilon", float(1e-6)),
            ("head_count", int(2)),
            ("head_dimension", int(4)),
            ("with_scale", OperatorAttributeValue::Boolean(false)),
        ],
        operator_semantics::rmsnorm_blocks(&x, None, 1e-6, Some((2, 4))).unwrap(),
    );
    // rope, half of each head rotated.
    check(
        &mut fixture,
        "rope",
        "rope",
        vec![("input", x.clone())],
        vec![
            ("base", float(4.0)),
            ("dimension", int(4)),
            ("head_count", int(2)),
            ("position_offset", int(1)),
            ("rotation_pairing", text("half_split")),
            ("partial_rotary_factor", float(0.5)),
        ],
        magnetar_runtime::reference_cpu::rope_partial(
            &x,
            4.0,
            1.0,
            4,
            1,
            2,
            magnetar_runtime::RopePairing::HalfSplit,
            None,
            Some(0.5),
        )
        .unwrap(),
    );
    // rope with YaRN (`express-latent-attention-forms`), its attributes
    // as the Runtime forwards them.
    let yarn = magnetar_core::operator::RopeYarnFrequencyScaling {
        factor: 4.0,
        original_context_length: 16,
        beta_fast: 32.0,
        beta_slow: 1.0,
        attention_factor: 1.25,
        truncate: true,
    };
    let mut yarn_attributes = BTreeMap::new();
    yarn.insert_attributes(&mut yarn_attributes);
    let mut attributes = vec![
        ("base", float(100.0)),
        ("dimension", int(4)),
        ("head_count", int(2)),
        ("position_offset", int(30)),
        ("rotation_pairing", text("interleaved")),
    ];
    attributes.extend(
        yarn_attributes
            .iter()
            .map(|(key, value)| (key.as_str(), value.clone())),
    );
    check(
        &mut fixture,
        "rope-yarn",
        "rope",
        vec![("input", x.clone())],
        attributes,
        magnetar_runtime::reference_cpu::rope_scaled(
            &x,
            100.0,
            1.0,
            4,
            30,
            2,
            magnetar_runtime::RopePairing::Interleaved,
            magnetar_runtime::reference_cpu::RopeFrequencies::Yarn(&yarn),
            None,
        )
        .unwrap(),
    );
    // attention with its own score scale.
    let (q, k, v) = (tensor(3, 8, 3), tensor(3, 4, 4), tensor(3, 4, 5));
    check(
        &mut fixture,
        "attention",
        "attention",
        vec![("q", q.clone()), ("k", k.clone()), ("v", v.clone())],
        vec![
            ("causal", OperatorAttributeValue::Boolean(true)),
            ("head_count", int(2)),
            ("kv_head_count", int(1)),
            ("head_dimension", int(4)),
            ("attention_mask_kind", text("causal")),
            ("scale", float(1.0)),
        ],
        magnetar_runtime::reference_cpu::attention_scaled(
            &q,
            &k,
            &v,
            2,
            4,
            Some(1),
            None,
            true,
            Some(1.0),
        )
        .unwrap(),
    );
    // activation softcap and scale, gelu (tanh), mul by a [1] operand.
    check(
        &mut fixture,
        "softcap",
        "activation",
        vec![("input", x.clone())],
        vec![("kind", text("softcap")), ("cap", float(0.5))],
        operator_semantics::softcap(&x, 0.5).unwrap(),
    );
    check(
        &mut fixture,
        "scale",
        "activation",
        vec![("input", x.clone())],
        vec![("kind", text("scale")), ("factor", float(-1.5))],
        operator_semantics::scale(&x, -1.5).unwrap(),
    );
    check(
        &mut fixture,
        "gelu",
        "gelu",
        vec![("input", x.clone())],
        vec![("approximate", text("tanh"))],
        HostTensor::new(
            x.shape.clone(),
            x.data
                .iter()
                .map(|value| operator_semantics::gelu_tanh(*value))
                .collect::<Vec<_>>(),
        )
        .unwrap(),
    );
    let scalar = HostTensor::new([1], vec![0.375]).unwrap();
    check(
        &mut fixture,
        "mul",
        "mul",
        vec![("a", x.clone()), ("b", scalar.clone())],
        vec![],
        operator_semantics::mul_broadcast(&x, &scalar).unwrap(),
    );
    // moe-experts with a GELU gate and per-expert scales.
    let (h, logits) = (tensor(2, 8, 6), tensor(2, 3, 7));
    let (gate, up, down) = (tensor(8, 12, 8), tensor(8, 12, 9), tensor(4, 24, 10));
    let per_expert = HostTensor::new([3], vec![0.5, 1.25, 2.0]).unwrap();
    let routing = magnetar_provider_api::MoeRouting {
        experts: 3,
        top_k: 2,
        normalize: true,
    };
    let expected = operator_semantics::moe_experts_local_with(
        &h,
        &logits,
        [&gate, &up, &down],
        routing,
        0,
        3,
        operator_semantics::MoeExpertOptions {
            activation: operator_semantics::MoeExpertActivation::GeluTanh,
            per_expert_scale: Some(&per_expert),
        },
    )
    .unwrap();
    check(
        &mut fixture,
        "moe",
        "moe-experts",
        vec![
            ("h", h),
            ("logits", logits),
            ("gate", gate),
            ("up", up),
            ("down", down),
            ("per_expert_scale", per_expert),
        ],
        vec![
            ("experts", int(3)),
            ("top_k", int(2)),
            ("normalize_top_k", OperatorAttributeValue::Boolean(true)),
            ("activation", text("gelu_tanh")),
        ],
        expected,
    );
}
