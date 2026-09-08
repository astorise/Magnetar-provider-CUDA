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
