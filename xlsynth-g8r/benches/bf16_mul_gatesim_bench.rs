// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use xlsynth::IrBits;
use xlsynth_g8r::gate_sim::{self, Collect};
use xlsynth_g8r::test_utils::{load_bf16_mul_sample, BF16_TOTAL_BITS};

/// Benchmarks the gate simulation of bf16 multiplication using fixed zero
/// inputs.
fn bf16_mul_gatesim_benchmark(c: &mut Criterion) {
    let loaded_sample = load_bf16_mul_sample();
    let gate_fn = loaded_sample.gate_fn;

    // Using fixed inputs helps reduce benchmark variance.
    let num_samples: u64 = 100; // Number of samples per benchmark iteration batch (use u64 for Throughput)

    // Prepare *one* owned input slice for eval *before* the benchmark loop
    // since the input is constant (zeros).
    let zero_arg_bits = IrBits::make_ubits(BF16_TOTAL_BITS, 0).unwrap();
    let prepared_eval_input: [IrBits; 2] = [zero_arg_bits.clone(), zero_arg_bits.clone()];

    let mut group = c.benchmark_group("bf16_mul_gatesim_zero_input");

    // Configure throughput measurement: specify the number of samples processed per
    // iteration Even though b.iter runs once, it represents processing
    // `num_samples` items conceptually.
    group.throughput(Throughput::Elements(num_samples));

    // Define the benchmark using the group
    // Pass the single prepared slice to the benchmark
    group.bench_function(BenchmarkId::from_parameter(num_samples), |b| {
        b.iter(|| {
            // Call eval once per iteration with the prepared constant input
            black_box(gate_sim::eval(
                &gate_fn,
                &prepared_eval_input,
                Collect::None,
            ));
        });
    });

    group.finish();
}

// Register the benchmark function with criterion
criterion_group!(benches, bf16_mul_gatesim_benchmark);
criterion_main!(benches);
