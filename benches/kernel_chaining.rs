//! `unify-provider-output-admission-and-residency`'s Decision 4: quantifies,
//! on real hardware, the H2D/D2H cost a forced host round-trip between two
//! device-resident Kernels pays versus chaining them directly through
//! existing device buffers -- the same physical mechanism
//! `enable-device-resident-kernel-chaining` eliminated inside `CudaExecutor`
//! and this change eliminated one layer up, in `first_native_runtime.rs`'s
//! own orchestration (a Rust-level id-rename/copy avoidance, not itself a
//! PCIe transfer, but gated on `CudaKernels`/`CudaExecutor` never forcing
//! one in the first place -- this benchmark measures that underlying cost).
//!
//! Measurement-only: never invoked by `cargo test`/CI, opt-in via
//! `cargo bench --bench kernel_chaining`. Gracefully skips (prints and
//! returns) on a host with no compatible CUDA driver/device, matching every
//! other hardware-gated check in this crate.
//!
//! # What this does and does not measure
//!
//! This measures the H2D/D2H transfer cost and the kernel-execution-vs-
//! transfer time split for a chained sequence of `CudaKernels` calls
//! directly -- real, on real hardware. It does **not** measure end-to-end
//! decode latency/tokens-per-second for a full first-native generation
//! step: that would require a CUDA-bound `Runtime`/`ModelInstance` actually
//! running a full Qwen decode step end to end, which does not exist yet
//! (first-native's production dispatch loop has only ever been exercised
//! against Reference CPU in this repository's own test suite so far -- see
//! `enable-device-resident-kernel-chaining`'s own audit trail). Building
//! that is separate, larger work, not part of this change. What is
//! reported here is the real, physical cost this change and its
//! predecessor exist to avoid, measured directly at its source.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use magnetar_provider_cuda::CudaProvider;
use magnetar_provider_cuda::kernels::{CudaDeviceBuffer, CudaKernels};
use magnetar_runtime::HostTensor;

/// A representative intermediate-activation size (`[sequence_length,
/// hidden_size]`-shaped, matching the E2E fixture's own dimensions) --
/// large enough that the PCIe transfer cost is measurable above launch
/// overhead noise, small enough to keep the benchmark fast to iterate.
const ROWS: u64 = 32;
const COLS: u64 = 256;

fn representative_tensor(seed: f32) -> HostTensor {
    let data: Vec<f32> = (0..(ROWS * COLS))
        .map(|i| seed + (i as f32) * 0.001)
        .collect();
    HostTensor::new([ROWS, COLS], data).expect("representative tensor shape is valid")
}

fn square_weight(seed: f32) -> HostTensor {
    let data: Vec<f32> = (0..(COLS * COLS))
        .map(|i| seed + (i as f32) * 0.0001)
        .collect();
    HostTensor::new([COLS, COLS], data).expect("representative weight shape is valid")
}

/// Chains three matmuls the way `first_native_runtime.rs` did *before*
/// `unify-provider-output-admission-and-residency`/`enable-device-resident-
/// kernel-chaining`: every intermediate result is downloaded to host and
/// immediately re-uploaded before the next Kernel consumes it, even though
/// both kernels run on the same device.
fn naive_round_trip_chain(
    kernels: &CudaKernels,
    input: &CudaDeviceBuffer,
    weight: &CudaDeviceBuffer,
) {
    let out1 = kernels.matmul(input, weight, false, false).unwrap();
    let host1 = kernels.download(&out1).unwrap();
    let reuploaded1 = kernels.upload(&host1).unwrap();

    let out2 = kernels.matmul(&reuploaded1, weight, false, false).unwrap();
    let host2 = kernels.download(&out2).unwrap();
    let reuploaded2 = kernels.upload(&host2).unwrap();

    let out3 = kernels.matmul(&reuploaded2, weight, false, false).unwrap();
    black_box(kernels.download(&out3).unwrap());
}

/// The same three-matmul chain, resident: each output feeds the next
/// Kernel directly by device buffer reference, no host round-trip until
/// the final, genuine result extraction.
fn resident_chain(kernels: &CudaKernels, input: &CudaDeviceBuffer, weight: &CudaDeviceBuffer) {
    let out1 = kernels.matmul(input, weight, false, false).unwrap();
    let out2 = kernels.matmul(&out1, weight, false, false).unwrap();
    let out3 = kernels.matmul(&out2, weight, false, false).unwrap();
    black_box(kernels.download(&out3).unwrap());
}

fn bench_kernel_chaining(c: &mut Criterion) {
    let provider = CudaProvider::new();
    let Some(context) = provider.context() else {
        eprintln!(
            "kernel_chaining benchmark skipped: no compatible CUDA driver/device on this host"
        );
        return;
    };
    let kernels = CudaKernels::compile_and_load(&context).expect(
        "kernel compilation must succeed on a machine that already passed device discovery",
    );

    let input_host = representative_tensor(1.0);
    let weight_host = square_weight(0.5);
    let input = kernels.upload(&input_host).unwrap();
    let weight = kernels.upload(&weight_host).unwrap();

    // Real, on-hardware H2D/D2H accounting: run each chain once outside the
    // timed loop and report how many real host<->device crossings it
    // performed and roughly how many bytes each crossing moved.
    let element_bytes = std::mem::size_of::<f32>() as u64;
    let tensor_bytes = ROWS * COLS * element_bytes;
    let uploads_before = kernels.upload_count();
    let downloads_before = kernels.download_count();
    naive_round_trip_chain(&kernels, &input, &weight);
    let naive_uploads = kernels.upload_count() - uploads_before;
    let naive_downloads = kernels.download_count() - downloads_before;

    let uploads_before = kernels.upload_count();
    let downloads_before = kernels.download_count();
    resident_chain(&kernels, &input, &weight);
    let resident_uploads = kernels.upload_count() - uploads_before;
    let resident_downloads = kernels.download_count() - downloads_before;

    println!("\n=== H2D/D2H crossing count, one 3-matmul chain ({ROWS}x{COLS} activations) ===");
    println!(
        "naive (host round-trip between kernels):    {naive_uploads} uploads, {naive_downloads} downloads (~{} KiB moved)",
        (naive_uploads + naive_downloads) * tensor_bytes / 1024
    );
    println!(
        "resident (chained via device buffers):      {resident_uploads} uploads, {resident_downloads} downloads (~{} KiB moved)",
        (resident_uploads + resident_downloads) * tensor_bytes / 1024
    );

    let mut group = c.benchmark_group("cuda_kernel_chaining");
    group.bench_function("naive_round_trip_chain", |b| {
        b.iter(|| naive_round_trip_chain(&kernels, black_box(&input), black_box(&weight)));
    });
    group.bench_function("resident_chain", |b| {
        b.iter(|| resident_chain(&kernels, black_box(&input), black_box(&weight)));
    });
    group.finish();

    // Kernel-execution-vs-transfer split: isolate one matmul's own compute
    // time (inputs already resident, output left resident) from one
    // round-trip transfer's own time (upload the same size, then download
    // it back), so the two can be read side by side in the report above.
    let mut split = c.benchmark_group("cuda_kernel_vs_transfer_split");
    split.bench_function("matmul_only_resident", |b| {
        b.iter(|| black_box(kernels.matmul(&input, &weight, false, false).unwrap()));
    });
    split.bench_function("upload_then_download_round_trip", |b| {
        b.iter(|| {
            let buffer = kernels.upload(black_box(&input_host)).unwrap();
            black_box(kernels.download(&buffer).unwrap());
        });
    });
    split.finish();
}

criterion_group!(benches, bench_kernel_chaining);
criterion_main!(benches);
