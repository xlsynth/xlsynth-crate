// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::SeedableRng;
use rand_pcg::Pcg64Mcg;
use std::time::Duration;
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
        let mut rng = Pcg64Mcg::seed_from_u64(12345);
        let initial_temp = 20.0;

        b.iter_batched(
            || {
                let current_gfn_cloned = start_gfn.clone();
                let current_cost_cloned = cost(&current_gfn_cloned);
                let best_gfn_cloned = start_gfn.clone();
                let best_cost_cloned = current_cost_cloned;
                let all_transforms = get_all_transforms();
                (
                    current_gfn_cloned,
                    current_cost_cloned,
                    best_gfn_cloned,
                    best_cost_cloned,
                    all_transforms,
                )
            },
            |(current_gfn, current_cost, mut best_gfn, mut best_cost, all_transforms)| {
                let objective = Objective::Product;
                let weights = build_transform_weights(&all_transforms, objective);
                let mut context = McmcContext {
                    rng: &mut rng,
                    all_transforms,
                    weights,
                    sat_ctx: xlsynth_g8r::validate_equiv::Ctx::new(),
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
