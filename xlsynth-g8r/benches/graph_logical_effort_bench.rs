// SPDX-License-Identifier: Apache-2.0
//! Exact graph-LE scaling: `cargo bench -p xlsynth-g8r --bench
//! graph_logical_effort_bench`. For process-isolated timing/RSS measurements,
//! run the built executable with `--measure --case fibonacci --size 1024
//! --repeat 5` (add `--score-only` to omit traceback), or `--case aig --aig
//! /path/to/graph.aig`. Bound each invocation externally, e.g. `(ulimit -v
//! 4194304; timeout 60s /usr/bin/time -v ...)` on Linux (4 GiB address space,
//! 60 seconds wall time). Peak RSS includes graph construction/loading, while
//! samples time only analysis. `umul` uses the same full-width IR workload as
//! ir2gates_mul_width_bench, with prep rewrites off.

use clap::{Parser, ValueEnum};
use criterion::{BenchmarkId, Criterion};
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use xlsynth_g8r::aig::graph_logical_effort::{
    GraphLogicalEffortOptions, LogicalEffortAnalysis, LogicalEffortStats,
    analyze_graph_logical_effort, score_graph_logical_effort,
};
use xlsynth_g8r::aig::{AigBitVector, AigNode, GateFn};
use xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary;
use xlsynth_g8r::aig_toggle_hotspots::load_aig_artifact;
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::ir2gates::{Ir2GatesOptions, ir2gates_from_ir_text};

/// Synthetic depth stress, arithmetic lowering, or an externally supplied
/// graph.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Case {
    Fibonacci,
    Umul,
    Aig,
}

/// One process per bounded measurement; Criterion is the default entry point.
#[derive(Parser)]
struct Args {
    #[arg(long)]
    measure: bool,
    #[arg(long)]
    score_only: bool,
    #[arg(long, value_enum, default_value = "fibonacci")]
    case: Case,
    #[arg(long, default_value_t = 256)]
    size: usize,
    #[arg(long)]
    aig: Option<PathBuf>,
    #[arg(long, default_value_t = 5)]
    repeat: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 1.0)]
    beta1: f64,
    #[arg(long, default_value_t = 0.0)]
    beta2: f64,
    #[arg(long, hide = true)]
    bench: bool,
}

/// Builds the review's single-sink DAG without folding or structural hashing.
fn fibonacci_graph(size: usize) -> GateFn {
    assert!(size > 0);
    let mut builder = GateBuilder::new("fibonacci".into(), GateBuilderOptions::no_opt());
    let input = builder.add_input("x".into(), 2);
    let mut a = *input.get_lsb(0);
    let mut b = *input.get_lsb(1);
    for _ in 0..size {
        let next = builder.add_and_binary(a, b);
        a = b;
        b = next;
    }
    builder.add_output("result".into(), AigBitVector::from_lsb_is_index_0(&[b]));
    let graph = builder.build();
    assert_eq!(and_count(&graph), size);
    graph
}

/// Lowers a full-width unsigned multiply outside the timed analysis region.
fn multiply_graph(width: usize) -> GateFn {
    let ir = format!(
        "package multiply\ntop fn main(a: bits[{width}] id=1, b: bits[{width}] id=2) -> bits[{}] {{\n  ret product: bits[{}] = umul(a, b, id=3)\n}}\n",
        2 * width,
        2 * width
    );
    ir2gates_from_ir_text(&ir, None, Ir2GatesOptions::all_opts_disabled())
        .unwrap()
        .gatify_output
        .gate_fn
}

fn and_count(graph: &GateFn) -> usize {
    graph
        .gates
        .iter()
        .filter(|node| matches!(node, AigNode::And2 { .. }))
        .count()
}

/// Independently checks the returned witness's edges, endpoints, and score.
fn check_path(graph: &GateFn, options: &GraphLogicalEffortOptions, result: &LogicalEffortAnalysis) {
    if and_count(graph) == 0 {
        assert_eq!(result.delay, -1.0);
        assert!(result.path.is_empty());
        return;
    }
    let mut fanouts = vec![0usize; graph.gates.len()];
    for node in &graph.gates {
        for arg in node.args() {
            fanouts[arg.id] += 1;
        }
    }
    assert!(!matches!(
        graph.gates[result.path[0].id],
        AigNode::And2 { .. }
    ));
    assert_eq!(fanouts[result.path.last().unwrap().id], 0);
    let mut log_effort = 0.0;
    for edge in result.path.windows(2) {
        assert!(graph.gates[edge[1].id].args().any(|arg| arg == edge[0]));
        let fanout = fanouts[edge[0].id] as f64;
        log_effort +=
            ((4.0 / 3.0) * (options.beta1 * fanout + options.beta2 * fanout * fanout)).ln();
    }
    let n = (result.path.len() - 1) as f64;
    let delay = n * (log_effort / n).exp() + 2.0 * n;
    assert!((delay - result.delay).abs() <= 1e-10 * delay.abs().max(1.0));
}

/// Common result for comparing exact score-only and path-producing calls.
struct Measurement {
    delay: f64,
    path: Option<Vec<usize>>,
    stats: LogicalEffortStats,
}

/// Reports individual repeated-cost samples and the full witness for
/// comparison.
fn measure(args: Args) {
    assert!(args.repeat > 0);
    let graph = match args.case {
        Case::Fibonacci => fibonacci_graph(args.size),
        Case::Umul => multiply_graph(args.size),
        Case::Aig => {
            load_aig_artifact(args.aig.as_ref().expect("--case aig requires --aig")).unwrap()
        }
    };
    let options = GraphLogicalEffortOptions {
        beta1: args.beta1,
        beta2: args.beta2,
    };
    // Match optimizer use: consume delay and destroy the full result within
    // the timed region. Collect counters and validate the witness separately.
    let score = || {
        if args.score_only {
            score_graph_logical_effort(black_box(&graph), black_box(&options)).delay
        } else {
            analyze_graph_logical_effort(black_box(&graph), black_box(&options)).delay
        }
    };
    for _ in 0..args.warmup {
        black_box(score());
    }
    let mut samples_ns = Vec::new();
    let mut previous = None;
    for _ in 0..args.repeat {
        let start = Instant::now();
        let delay = black_box(score());
        samples_ns.push(start.elapsed().as_nanos());
        if let Some(previous) = previous {
            assert_eq!(delay, previous);
        }
        previous = Some(delay);
    }
    let result = if args.score_only {
        let score = score_graph_logical_effort(&graph, &options);
        assert_eq!(score.stats.traceback_records, 0);
        Measurement {
            delay: score.delay,
            path: None,
            stats: score.stats,
        }
    } else {
        let analysis = analyze_graph_logical_effort(&graph, &options);
        check_path(&graph, &options, &analysis);
        Measurement {
            delay: analysis.delay,
            path: Some(analysis.path.iter().map(|node| node.id).collect()),
            stats: analysis.stats,
        }
    };
    assert_eq!(Some(result.delay), previous);
    let graph_hash = blake3::hash(&emit_aiger_binary(&graph, false).unwrap())
        .to_hex()
        .to_string();
    println!(
        "{}",
        serde_json::json!({
            "case": format!("{:?}", args.case), "size": args.size,
            "aig": args.aig, "and_count": and_count(&graph),
            "beta1": args.beta1, "beta2": args.beta2, "samples_ns": samples_ns,
            "delay": result.delay, "path": result.path, "score_only": args.score_only,
            "stats": result.stats, "warmup": args.warmup,
            "aiger_blake3": graph_hash,
        })
    );
}

fn main() {
    if std::env::args().any(|arg| arg == "--measure") {
        measure(Args::parse());
        return;
    }
    let mut criterion = Criterion::default().configure_from_args();
    let mut group = criterion.benchmark_group("graph_logical_effort");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2));
    let options = GraphLogicalEffortOptions {
        beta1: 1.0,
        beta2: 0.0,
    };
    for size in [128, 512, 2048, 8192, 32768] {
        let graph = fibonacci_graph(size);
        group.bench_with_input(BenchmarkId::new("fibonacci", size), &graph, |b, graph| {
            b.iter(|| analyze_graph_logical_effort(black_box(graph), black_box(&options)));
        });
        group.bench_with_input(
            BenchmarkId::new("fibonacci_score", size),
            &graph,
            |b, graph| {
                b.iter(|| score_graph_logical_effort(black_box(graph), black_box(&options)));
            },
        );
    }
    for width in [16, 32, 64] {
        let graph = multiply_graph(width);
        group.bench_with_input(BenchmarkId::new("umul", width), &graph, |b, graph| {
            b.iter(|| analyze_graph_logical_effort(black_box(graph), black_box(&options)));
        });
        group.bench_with_input(BenchmarkId::new("umul_score", width), &graph, |b, graph| {
            b.iter(|| score_graph_logical_effort(black_box(graph), black_box(&options)));
        });
    }
    group.finish();
    criterion.final_summary();
}
