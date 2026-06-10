// SPDX-License-Identifier: Apache-2.0

use std::ptr;
use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir_compiler::{ExecutionContext, ExecutionOptions, PirFunctionCompiler};

fn compile_top(ir_text: &str) -> PirFunctionCompiler {
    let package = Parser::new(ir_text)
        .parse_and_validate_package()
        .expect("benchmark IR should parse and validate");
    PirFunctionCompiler::compile_package(&package).expect("benchmark IR should compile")
}

fn scratch_storage(compiler: &PirFunctionCompiler) -> Vec<u64> {
    vec![
        0;
        compiler
            .scratch_byte_count()
            .div_ceil(std::mem::size_of::<u64>())
    ]
}

fn scratch_pointer(scratch: &mut [u64]) -> *mut u8 {
    if scratch.is_empty() {
        ptr::null_mut()
    } else {
        scratch.as_mut_ptr().cast::<u8>()
    }
}

fn bench_no_event_run(c: &mut Criterion) {
    let compiler = compile_top(
        r#"package bench

top fn f(x: bits[32] id=1, y: bits[32] id=2) -> bits[32] {
  sum: bits[32] = add(x, y, id=3)
  product: bits[32] = smul(sum, x, id=4)
  shifted: bits[32] = shrl(product, y, id=5)
  ret result: bits[32] = xor(shifted, sum, id=6)
}
"#,
    );
    let x = 0x1234_5678u32;
    let y = 13u32;
    let inputs = [
        ptr::from_ref(&x).cast::<u8>(),
        ptr::from_ref(&y).cast::<u8>(),
    ];
    let mut output = 0u32;
    let mut scratch = scratch_storage(&compiler);
    let scratch_bytes = scratch.len() * std::mem::size_of::<u64>();
    let mut context =
        ExecutionContext::new_with_options(compiler.metadata(), ExecutionOptions::NO_EVENTS);

    c.bench_function("pir_aot_run_no_event_sites", |b| {
        b.iter(|| {
            context.clear_with_options(ExecutionOptions::NO_EVENTS);
            // SAFETY: inputs/output are native `bits[32]` carriers matching the
            // compiled signature, and scratch/context are owned by this bench.
            unsafe {
                compiler
                    .run_native_with_scratch_and_context(
                        black_box(inputs.as_slice()),
                        ptr::from_mut(&mut output).cast(),
                        scratch_pointer(&mut scratch),
                        scratch_bytes,
                        &mut context,
                    )
                    .expect("benchmark execution should succeed");
            }
            black_box(output);
        });
    });
}

fn bench_event_runs(c: &mut Criterion) {
    let compiler = compile_top(
        r#"package bench

top fn f(x: bits[32] id=1, y: bits[32] id=2, emit: bits[1] id=3) -> bits[32] {
  t: token = after_all(id=4)
  covered: () = cover(emit, label="event_hit", id=5)
  traced: token = trace(t, emit, format="x={} y={}", data_operands=[x, y], verbosity=1, id=6)
  sum: bits[32] = add(x, y, id=7)
  product: bits[32] = smul(sum, x, id=8)
  ret result: bits[32] = xor(product, y, id=9)
}
"#,
    );
    let x = 0x1234_5678u32;
    let y = 13u32;
    let emit = 1u8;
    let inputs = [
        ptr::from_ref(&x).cast::<u8>(),
        ptr::from_ref(&y).cast::<u8>(),
        ptr::from_ref(&emit).cast::<u8>(),
    ];
    let mut output = 0u32;
    let mut scratch = scratch_storage(&compiler);
    let scratch_bytes = scratch.len() * std::mem::size_of::<u64>();
    let mut context =
        ExecutionContext::new_with_options(compiler.metadata(), ExecutionOptions::NO_EVENTS);

    c.bench_function("pir_aot_run_event_sites_disabled", |b| {
        b.iter(|| {
            context.clear_with_options(ExecutionOptions::NO_EVENTS);
            // SAFETY: inputs/output are native carriers matching the compiled
            // signature, and scratch/context are owned by this bench.
            unsafe {
                compiler
                    .run_native_with_scratch_and_context(
                        black_box(inputs.as_slice()),
                        ptr::from_mut(&mut output).cast(),
                        scratch_pointer(&mut scratch),
                        scratch_bytes,
                        &mut context,
                    )
                    .expect("benchmark execution should succeed");
            }
            black_box(output);
        });
    });

    c.bench_function("pir_aot_run_event_sites_collect_all", |b| {
        b.iter(|| {
            context.clear_with_options(ExecutionOptions::collect_all());
            // SAFETY: inputs/output are native carriers matching the compiled
            // signature, and scratch/context are owned by this bench.
            unsafe {
                compiler
                    .run_native_with_scratch_and_context(
                        black_box(inputs.as_slice()),
                        ptr::from_mut(&mut output).cast(),
                        scratch_pointer(&mut scratch),
                        scratch_bytes,
                        &mut context,
                    )
                    .expect("benchmark execution should succeed");
            }
            black_box((&output, context.result()));
        });
    });
}

criterion_group! {
    name = aot_performance;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .sample_size(20);
    targets = bench_no_event_run, bench_event_runs
}

criterion_main!(aot_performance);
