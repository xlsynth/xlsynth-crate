use criterion::{black_box, criterion_group, criterion_main, Criterion};
use xlsynth_g8r::{gate_builder::GateBuilderOptions, ir2gate, test_utils::load_bf16_mul_sample};

fn gatify_bf16_mul_benchmark(c: &mut Criterion) {
    let sample = load_bf16_mul_sample();
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
            };
            ir2gate::gatify(black_box(&ir_fn), gatify_options).unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, gatify_bf16_mul_benchmark);
criterion_main!(benches);
