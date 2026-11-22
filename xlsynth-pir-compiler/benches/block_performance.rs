// SPDX-License-Identifier: Apache-2.0

//! Timing and allocation measurements for deterministic block fixtures.
//!
//! The timed step and tick both return one owned, typed output. Preparation
//! starts from a parsed package. Compilation starts from a prepared block.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt::Write;
use std::hint::black_box;
use std::process::Command;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use xlsynth_pir::IrValue;
use xlsynth_pir::block_eval::{BlockEvaluator, RegisterInit};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir_compiler::ExecutionOptions;
use xlsynth_pir_compiler::block::{BlockCompilation, CompiledBlock};

/// Counts Rust allocation requests on the benchmark thread during a sample.
struct CountingAllocator;

thread_local! {
    static ALLOCATIONS: Cell<Option<u64>> = const { Cell::new(None) };
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn record_allocation() {
    ALLOCATIONS.with(|count| {
        if let Some(value) = count.get() {
            count.set(Some(value + 1));
        }
    });
}

// SAFETY: each operation forwards the identical arguments to the system
// allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(p, layout, size) }
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) }
    }
}

/// Warm the execution buffers before counting a fixed sequence of clocks.
fn allocations_per_clock(mut run: impl FnMut()) -> f64 {
    for _ in 0..8 {
        run();
    }
    ALLOCATIONS.with(|count| count.set(Some(0)));
    for _ in 0..128 {
        run();
    }
    ALLOCATIONS.with(|count| count.replace(None).unwrap() as f64 / 128.0)
}

/// A parsed design's source and a short repeating series of prebuilt inputs.
struct Fixture {
    name: String,
    ir: String,
    clocks: Vec<Vec<IrValue>>,
}

fn counter(width: usize) -> Fixture {
    Fixture {
        name: format!("counter-{width}b"),
        ir: format!(
            r#"package counter
top block dut(clk: clock, enable: bits[1], q: bits[{width}]) {{
  reg count(bits[{width}])
  enable: bits[1] = input_port(name=enable, id=1)
  old: bits[{width}] = register_read(register=count, id=2)
  one: bits[{width}] = literal(value=1, id=3)
  next: bits[{width}] = add(old, one, id=4)
  write: () = register_write(next, register=count, load_enable=enable, id=5)
  q: () = output_port(old, name=q, id=6)
}}"#
        ),
        clocks: [true, false, true, true]
            .map(|en| vec![IrValue::bool(en)])
            .into(),
    }
}

fn pipeline(width: usize, words: usize, stages: usize) -> Fixture {
    let ty = format!("bits[{width}][{words}]");
    let mut ir = format!("package pipeline\ntop block dut(clk: clock, d: {ty}, q: {ty}) {{\n");
    for stage in 0..stages {
        writeln!(ir, "  reg stage_{stage}({ty})").unwrap();
    }
    writeln!(ir, "  d: {ty} = input_port(name=d, id=1)").unwrap();
    let mut data = String::from("d");
    for stage in 0..stages {
        let old = format!("q_{stage}");
        writeln!(
            ir,
            "  {old}: {ty} = register_read(register=stage_{stage}, id={})",
            2 + stage * 2
        )
        .unwrap();
        writeln!(
            ir,
            "  w_{stage}: () = register_write({data}, register=stage_{stage}, id={})",
            3 + stage * 2
        )
        .unwrap();
        data = old;
    }
    writeln!(
        ir,
        "  q: () = output_port({data}, name=q, id={})\n}}",
        2 + stages * 2
    )
    .unwrap();
    let clocks = (0..4)
        .map(|sample| {
            let elements: Vec<_> = (0..words)
                .map(|i| IrValue::make_ubits(width, (i * 17 + sample) as u64).unwrap())
                .collect();
            vec![IrValue::make_array(&elements).unwrap()]
        })
        .collect();
    Fixture {
        name: format!("pipeline-{stages}stages-{words}x{width}b"),
        ir,
        clocks,
    }
}

fn wide_datapath(width: usize, passthrough: bool) -> Fixture {
    // Keep the native call, input, register and output shapes identical in the
    // allocation control. Only the body of the called function changes.
    let body = if passthrough {
        format!(
            "  shifted: bits[{width}] = identity(x, id=5)\n  parity: bits[1] = literal(value=0, id=6)"
        )
    } else {
        format!(
            "  masked: bits[{width}] = and(x, mask, id=4)\n  shifted: bits[{width}] = shrl(masked, amount, id=5)\n  parity: bits[1] = xor_reduce(shifted, id=6)"
        )
    };
    let ir = format!(
        r#"package wide
fn reduce_shift(x: bits[{width}] id=1, mask: bits[{width}] id=2, amount: bits[8] id=3) -> (bits[{width}], bits[1]) {{
{body}
  ret result: (bits[{width}], bits[1]) = tuple(shifted, parity, id=7)
}}
top block dut(clk: clock, x: bits[{width}], mask: bits[{width}], amount: bits[8], q: (bits[{width}], bits[1])) {{
  reg saved((bits[{width}], bits[1]))
  x: bits[{width}] = input_port(name=x, id=11)
  mask: bits[{width}] = input_port(name=mask, id=12)
  amount: bits[8] = input_port(name=amount, id=13)
  value: (bits[{width}], bits[1]) = invoke(x, mask, amount, to_apply=reduce_shift, id=14)
  old: (bits[{width}], bits[1]) = register_read(register=saved, id=15)
  write: () = register_write(value, register=saved, id=16)
  q: () = output_port(old, name=q, id=17)
}}"#
    );
    let clocks = (0..4)
        .map(|sample| {
            let v = format!(
                "bits[{width}]:0x1{}{:x}",
                "b".repeat((width - 1) / 4 - 1),
                7 + sample
            );
            let mask = format!("bits[{width}]:0x1{}", "d".repeat((width - 1) / 4));
            vec![
                IrValue::parse_typed(&v).unwrap(),
                IrValue::parse_typed(&mask).unwrap(),
                IrValue::make_ubits(8, sample * 17).unwrap(),
            ]
        })
        .collect();
    Fixture {
        name: format!("wide-mask-shift-reduce-{width}b"),
        ir,
        clocks,
    }
}

/// Separate the cost of enabled events and wide operations using matched
/// inputs.
fn allocation_controls() {
    let mut scalar = counter(32);
    let closing = scalar.ir.rfind('}').unwrap();
    scalar.ir.insert_str(
        closing,
        r#"  t: token = after_all(id=7)
  hit: () = cover(enable, label="enabled", id=8)
  tr: token = trace(t, enable, format="count={}", data_operands=[old], verbosity=0, id=9)
"#,
    );
    let package = Parser::new(&scalar.ir)
        .parse_and_validate_package()
        .unwrap();
    let compiled = CompiledBlock::compile(&package, "dut").unwrap();
    for (policy, options) in [
        ("disabled", ExecutionOptions::NO_EVENTS),
        ("collect_all", ExecutionOptions::collect_all()),
    ] {
        let mut instance = compiled
            .prepare_instance_with_options(RegisterInit::AllZeros, options)
            .unwrap();
        let mut inputs = scalar.clocks.iter().cycle();
        let allocations = allocations_per_clock(|| {
            black_box(instance.tick(inputs.next().unwrap()).unwrap());
        });
        eprintln!(
            "eventful counter-32b, events={policy}: {allocations:.2} Rust allocation requests/prepared tick"
        );
    }
    for (body, passthrough) in [("identity control", true), ("mask/shift/reduction", false)] {
        let fixture = wide_datapath(129, passthrough);
        let package = Parser::new(&fixture.ir)
            .parse_and_validate_package()
            .unwrap();
        let compiled = CompiledBlock::compile(&package, "dut").unwrap();
        let mut instance = compiled.prepare_instance(RegisterInit::AllZeros).unwrap();
        let mut inputs = fixture.clocks.iter().cycle();
        let allocations = allocations_per_clock(|| {
            black_box(instance.tick(inputs.next().unwrap()).unwrap());
        });
        eprintln!(
            "wide-129b, helper={body}, events=collect_all: {allocations:.2} Rust allocation requests/prepared tick"
        );
    }
}

/// Log hardware and code revision alongside Criterion's measurement report.
fn report_environment() {
    let revision = Command::new("git")
        .args(["describe", "--always", "--dirty"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned());
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix("model name")
                    .and_then(|v| v.split_once(':'))
                    .map(|(_, cpu)| cpu.trim().to_owned())
            })
        });
    eprintln!(
        "block benchmark: revision={}, CPU={}, debug_assertions={}, initial state=zero, events=collect_all, clocks repeat deterministically",
        revision.as_deref().unwrap_or("unknown"),
        cpu.as_deref().unwrap_or(std::env::consts::ARCH),
        cfg!(debug_assertions)
    );
}

fn bench_blocks(c: &mut Criterion) {
    report_environment();
    allocation_controls();
    let mut group = c.benchmark_group("compiled_block");
    for fixture in [
        counter(8),
        counter(32),
        pipeline(32, 4, 2),
        pipeline(64, 16, 4),
        wide_datapath(65, false),
        wide_datapath(129, false),
        wide_datapath(257, false),
    ] {
        let package = Parser::new(&fixture.ir)
            .parse_and_validate_package()
            .unwrap();
        let compiled = CompiledBlock::compile(&package, "dut").unwrap();
        let eval = BlockEvaluator::prepare(&package, "dut").unwrap();
        let mut reference = eval.initial_state(RegisterInit::AllZeros).unwrap();
        let mut state = compiled.initial_state(RegisterInit::AllZeros).unwrap();
        let mut instance = compiled.prepare_instance(RegisterInit::AllZeros).unwrap();
        for inputs in fixture.clocks.iter().cycle().take(16) {
            let expected = eval.step(&reference, inputs).unwrap();
            let actual = compiled.step(&state, inputs).unwrap();
            instance.tick(inputs).unwrap();
            assert_eq!(actual.outputs, expected.outputs);
            assert_eq!(instance.output(0).unwrap(), expected.outputs[0]);
            assert_eq!(actual.next_state.values(), expected.next_state.values());
            assert_eq!(
                instance.snapshot().unwrap().values(),
                expected.next_state.values()
            );
            state = actual.next_state;
            reference = expected.next_state;
        }

        group.bench_function(BenchmarkId::new("prepare", &fixture.name), |b| {
            b.iter_batched(
                || (),
                |()| BlockCompilation::prepare(black_box(&package), "dut").unwrap(),
                BatchSize::PerIteration,
            )
        });
        group.bench_function(BenchmarkId::new("compile", &fixture.name), |b| {
            b.iter_batched(
                || BlockCompilation::prepare(&package, "dut").unwrap(),
                |plan| plan.compile().unwrap(),
                BatchSize::PerIteration,
            )
        });

        let mut stimulus = fixture.clocks.iter().cycle();
        let mut typed_clock = || {
            let result = compiled
                .step(&state, black_box(stimulus.next().unwrap()))
                .unwrap();
            black_box(result.outputs);
            state = result.next_state;
        };
        let typed_allocations = allocations_per_clock(&mut typed_clock);
        group.bench_function(BenchmarkId::new("typed_step", &fixture.name), |b| {
            b.iter(&mut typed_clock)
        });

        let mut stimulus = fixture.clocks.iter().cycle();
        let tick_allocations = allocations_per_clock(|| {
            black_box(instance.tick(stimulus.next().unwrap()).unwrap());
        });
        let output_allocations = allocations_per_clock(|| {
            black_box(instance.output(0).unwrap());
        });
        let mut stimulus = fixture.clocks.iter().cycle();
        group.bench_function(
            BenchmarkId::new("prepared_tick_with_output", &fixture.name),
            |b| {
                b.iter(|| {
                    black_box(instance.tick(black_box(stimulus.next().unwrap())).unwrap());
                    black_box(instance.output(0).unwrap());
                })
            },
        );
        eprintln!(
            "{}: Rust allocation requests/clock, typed step incl. output={typed_allocations:.2}, prepared tick={tick_allocations:.2}, prepared output read={output_allocations:.2}",
            fixture.name
        );
    }
    group.finish();
}

criterion_group! {
    name = block_performance;
    config = Criterion::default().warm_up_time(Duration::from_secs(1)).measurement_time(Duration::from_secs(3)).sample_size(15);
    targets = bench_blocks
}
criterion_main!(block_performance);
