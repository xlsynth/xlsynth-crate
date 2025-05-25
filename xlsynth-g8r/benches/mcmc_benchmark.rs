// SPDX-License-Identifier: Apache-2.0
#![feature(portable_simd)]

use core::simd::u64x4;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::SeedableRng;
use rand_pcg::Pcg64Mcg;
use std::time::Duration;
use xlsynth_g8r::gate_simd::{self, Vec256};
use xlsynth_g8r::mcmc_logic::{
    build_transform_weights, cost, load_start, mcmc_iteration, McmcContext, McmcIterationOutput,
    Objective,
};
use xlsynth_g8r::transforms::get_all_transforms;

fn benchmark_mcmc_iteration(c: &mut Criterion) {
    let start_gfn = match load_start("sample://bf16_add") {
        Ok(gfn) => gfn,
        Err(e) => {
            eprintln!("Failed to load bf16_add sample for benchmark: {:?}", e);
            panic!("Benchmark setup failed");
        }
    };

    let mut group = c.benchmark_group("MCMC Iteration Logic");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("mcmc_iteration_bf16_add", |b| {
        let initial_temp = 20.0;

        b.iter_batched(
            || {
                // Fresh RNG for each batched iteration so that prepare and
                // execution phases can both borrow it mutably without
                // overlap.
                let mut rng = Pcg64Mcg::seed_from_u64(12345);

                let current_gfn_cloned = start_gfn.clone();
                let current_cost_cloned = cost(&current_gfn_cloned);
                let best_gfn_cloned = start_gfn.clone();
                let best_cost_cloned = current_cost_cloned;
                let all_transforms = get_all_transforms();

                // Prepare SIMD batch and baseline outputs once for this iteration.
                let simd_inputs = {
                    let total_bits: usize =
                        start_gfn.inputs.iter().map(|i| i.get_bit_count()).sum();
                    let mut words_per_bit = vec![[0u64; 4]; total_bits];
                    for lane in 0..256 {
                        let mut bit_cursor = 0;
                        for input in &start_gfn.inputs {
                            let rand_bits = xlsynth_g8r::fuzz_utils::arbitrary_irbits(
                                &mut rng,
                                input.bit_vector.get_bit_count(),
                            );
                            for bit_idx in 0..input.bit_vector.get_bit_count() {
                                if rand_bits.get_bit(bit_idx).unwrap() {
                                    let limb = lane / 64;
                                    let offset = lane % 64;
                                    words_per_bit[bit_cursor + bit_idx][limb] |= 1u64 << offset;
                                }
                            }
                            bit_cursor += input.bit_vector.get_bit_count();
                        }
                    }
                    words_per_bit
                        .into_iter()
                        .map(|w| Vec256(u64x4::from_array(w)))
                        .collect::<Vec<Vec256>>()
                };

                let baseline_outputs = gate_simd::eval(&start_gfn, &simd_inputs).outputs;

                (
                    current_gfn_cloned,
                    current_cost_cloned,
                    best_gfn_cloned,
                    best_cost_cloned,
                    all_transforms,
                    rng,
                    simd_inputs,
                    baseline_outputs,
                )
            },
            |(
                current_gfn,
                current_cost,
                mut best_gfn,
                mut best_cost,
                all_transforms,
                mut rng,
                simd_inputs,
                baseline_outputs,
            )| {
                let objective = Objective::Product;
                let weights = build_transform_weights(&all_transforms, objective);
                let mut context = McmcContext {
                    rng: &mut rng,
                    all_transforms,
                    weights,
                };
                let _result: McmcIterationOutput = mcmc_iteration(
                    current_gfn,
                    current_cost,
                    &mut best_gfn,
                    &mut best_cost,
                    &mut context,
                    initial_temp,
                    objective,
                    false,
                    &simd_inputs,
                    &baseline_outputs,
                );
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_millis(500)).sample_size(10);
    targets = benchmark_mcmc_iteration
);
criterion_main!(benches);
