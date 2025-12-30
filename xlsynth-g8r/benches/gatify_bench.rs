// SPDX-License-Identifier: Apache-2.0

//! Benchmarks for gatifying the bfloat16 multiply function with various options
//! enabled.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use xlsynth_g8r::{
    aig_serdes::ir2gate,
    gate_builder::GateBuilderOptions,
    ir2gate_utils::AdderMapping,
    test_utils::{Opt, load_bf16_add_sample, load_bf16_mul_sample},
};

fn gatify_bf16_mul_benchmark(c: &mut Criterion) {
    let sample = load_bf16_mul_sample(Opt::Yes);
    let ir_fn = sample.g8r_pkg.get_fn(&sample.mangled_fn_name).unwrap();

    let mut group = c.benchmark_group("gatify_bf16_mul");

    group.bench_function("gatify_bf16_mul_no_opts", |b| {
        b.iter(|| {
            let builder_options = GateBuilderOptions {
                fold: false,
                hash: false,
            };
            let gatify_options = ir2gate::GatifyOptions {
                fold: builder_options.fold,
                hash: builder_options.hash,
                check_equivalence: false, // Not needed for benchmark
                adder_mapping: AdderMapping::default(),
                mul_adder_mapping: None,
            };
            ir2gate::gatify(black_box(&ir_fn), gatify_options).unwrap();
        })
    });

    group.bench_function("gatify_bf16_mul_fold_only", |b| {
        b.iter(|| {
            let builder_options = GateBuilderOptions {
                fold: true,
                hash: false,
            };
            let gatify_options = ir2gate::GatifyOptions {
                fold: builder_options.fold,
                hash: builder_options.hash,
                check_equivalence: false, // Not needed for benchmark
                adder_mapping: AdderMapping::default(),
                mul_adder_mapping: None,
            };
            ir2gate::gatify(black_box(&ir_fn), gatify_options).unwrap();
        })
    });

    group.bench_function("gatify_bf16_mul_fold_hash", |b| {
        b.iter(|| {
            let builder_options = GateBuilderOptions {
                fold: true,
                hash: true,
            }; // Equivalent to GateBuilderOptions::opt()
            let gatify_options = ir2gate::GatifyOptions {
                fold: builder_options.fold,
                hash: builder_options.hash,
                check_equivalence: false, // Not needed for benchmark
                adder_mapping: AdderMapping::default(),
                mul_adder_mapping: None,
            };
            ir2gate::gatify(black_box(&ir_fn), gatify_options).unwrap();
        })
    });

    group.finish();
}

fn gatify_bf16_add_benchmark(c: &mut Criterion) {
    let sample = load_bf16_add_sample(Opt::Yes);
    let ir_fn = sample.g8r_pkg.get_fn(&sample.mangled_fn_name).unwrap();

    let mut group = c.benchmark_group("gatify_bf16_add");

    group.bench_function("gatify_bf16_add_no_opts", |b| {
        b.iter(|| {
            let builder_options = GateBuilderOptions {
                fold: false,
                hash: false,
            };
            let gatify_options = ir2gate::GatifyOptions {
                fold: builder_options.fold,
                hash: builder_options.hash,
                check_equivalence: false,
                adder_mapping: AdderMapping::default(),
                mul_adder_mapping: None,
            };
            ir2gate::gatify(black_box(&ir_fn), gatify_options).unwrap();
        })
    });

    group.bench_function("gatify_bf16_add_fold_only", |b| {
        b.iter(|| {
            let builder_options = GateBuilderOptions {
                fold: true,
                hash: false,
            };
            let gatify_options = ir2gate::GatifyOptions {
                fold: builder_options.fold,
                hash: builder_options.hash,
                check_equivalence: false,
                adder_mapping: AdderMapping::default(),
                mul_adder_mapping: None,
            };
            ir2gate::gatify(black_box(&ir_fn), gatify_options).unwrap();
        })
    });

    group.bench_function("gatify_bf16_add_fold_hash", |b| {
        b.iter(|| {
            let builder_options = GateBuilderOptions {
                fold: true,
                hash: true,
            };
            let gatify_options = ir2gate::GatifyOptions {
                fold: builder_options.fold,
                hash: builder_options.hash,
                check_equivalence: false,
                adder_mapping: AdderMapping::default(),
                mul_adder_mapping: None,
            };
            ir2gate::gatify(black_box(&ir_fn), gatify_options).unwrap();
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    gatify_bf16_mul_benchmark,
    gatify_bf16_add_benchmark
);
criterion_main!(benches);
