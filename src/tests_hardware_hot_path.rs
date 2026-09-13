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

use magnetar_runtime::provider::Provider;
use magnetar_runtime::{
    ComputeDType, DTypeDescriptor, DeviceBinding, FallbackClass, HostTensor, KernelDispatchPlan,
    KernelDispatchPlanId, KernelInvocationId, KernelMemoryClass, KernelResource,
    KernelResultStatus, KernelSelectionRequest, LayoutDescriptor, MemoryAllocationOwner,
    MemoryAllocationState, MemoryPlacement, OperatorAttributeValue, OperatorFamily, OperatorId,
    ProviderBinding, ProviderExecutionApi, ResourceAffinity, Runtime, ShapeDescriptor,
    TensorDescriptor, TensorResourceDescriptor, TensorResourceId, TensorValue,
    initial_operator_catalog,
};

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
            magnetar_runtime::MemoryAllocationClass::Tensor,
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
        "add" | "mul" => OperatorFamily::Tensor,
        other => panic!("{step_name}: no OperatorFamily mapping for '{other}'"),
    };
    let operator = OperatorId::magnetar(operator_name, 1, family);
    let mut request = KernelSelectionRequest::new(
        format!("hot-path-{step_name}"),
        operator,
        fixture.affinity.clone(),
    );
    for (id, tensor_descriptor) in &inputs {
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

    let mut dispatcher = magnetar_runtime::KernelDispatcher::new();
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
