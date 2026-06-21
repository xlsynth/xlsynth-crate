// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use xlsynth_pir_compiler_aot_ir_test_crate::native_aot_tests_aot;
use xlsynth_pir_compiler_aot_ir_test_crate::native_aot_tests_aot::native_aot_tests;
use xlsynth_pir_compiler_runtime::ExecutionOptions;

fn bench_generated_scalar_runner(c: &mut Criterion) {
    let lhs = native_aot_tests_aot::U8::new(10);
    let rhs = native_aot_tests_aot::U8::new(20);

    c.bench_function("pir_aot_generated_runner_run", |b| {
        let mut runner =
            native_aot_tests::aot_add_inputs::new_runner().expect("runner should initialize");
        let mut output = native_aot_tests_aot::U8::all_zeros();
        b.iter(|| {
            runner
                .run(black_box(&lhs), black_box(&rhs), black_box(&mut output))
                .expect("benchmark execution should succeed");
            black_box(output);
        });
    });
}

fn bench_generated_event_runner(c: &mut Criterion) {
    let x = native_aot_tests_aot::U8::new(0xa5);
    let y = native_aot_tests_aot::U8::new(0x3c);
    let passed = native_aot_tests_aot::U1::new(1);
    let emit = native_aot_tests_aot::U1::new(1);

    c.bench_function("pir_aot_generated_runner_event_sites_run", |b| {
        let mut runner =
            native_aot_tests::aot_events::new_runner().expect("runner should initialize");
        let mut output = native_aot_tests_aot::U8::all_zeros();
        b.iter(|| {
            runner
                .run(
                    black_box(&x),
                    black_box(&y),
                    black_box(&passed),
                    black_box(&emit),
                    black_box(&mut output),
                )
                .expect("benchmark execution should succeed");
            black_box(output);
        });
    });

    c.bench_function("pir_aot_generated_runner_event_sites_collect_all", |b| {
        let mut runner =
            native_aot_tests::aot_events::new_runner().expect("runner should initialize");
        let mut output = native_aot_tests_aot::U8::all_zeros();
        b.iter(|| {
            let events = runner
                .run_with_events(
                    black_box(&x),
                    black_box(&y),
                    black_box(&passed),
                    black_box(&emit),
                    black_box(&mut output),
                    ExecutionOptions::collect_all(),
                )
                .expect("benchmark execution should succeed");
            black_box(output);
            black_box(events);
        });
    });
}

criterion_group! {
    name = generated_runner_performance;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .sample_size(20);
    targets = bench_generated_scalar_runner, bench_generated_event_runner
}

criterion_main!(generated_runner_performance);
