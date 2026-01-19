// SPDX-License-Identifier: Apache-2.0
//! Benchmark: end-to-end `ir2gates`-like flow on `umul` for increasing bit
//! widths.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::sync::Arc;
use std::sync::OnceLock;
use xlsynth_g8r::aig::cut_db_rewrite::{RewriteOptions, rewrite_gatefn_with_cut_db};
use xlsynth_g8r::cut_db::loader::CutDb;
use xlsynth_g8r::cut_db_cli_defaults::{
    CUT_DB_REWRITE_MAX_CUTS_PER_NODE_CLI, CUT_DB_REWRITE_MAX_ITERATIONS_CLI,
};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::ir2gates;

static CUT_DB: OnceLock<Arc<CutDb>> = OnceLock::new();

fn mul_ir_text(width: usize) -> String {
    let out_width = width * 2;
    // Keep formatting stable; embed the width and output width.
    format!(
        "package sample\n\n\
top fn main(a: bits[{w}] id=1, b: bits[{w}] id=2) -> bits[{ow}] {{\n\
  ret umul.3: bits[{ow}] = umul(a, b, id=3)\n\
}}\n",
        w = width,
        ow = out_width
    )
}

fn run_ir2gates_like_flow_for_umul(width: usize) {
    let ir_text = mul_ir_text(width);

    let ir2gates_output = ir2gates::ir2gates_from_ir_text(
        &ir_text,
        None,
        ir2gates::Ir2GatesOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            enable_rewrite_carry_out: false,
            adder_mapping: AdderMapping::default(),
            mul_adder_mapping: None,
            aug_opt: Default::default(),
        },
    )
    .unwrap();
    let gate_fn = ir2gates_output.gatify_output.gate_fn;

    // Use the same bounded cut-db rewrite settings as the CLI path.
    let db: &Arc<CutDb> = CUT_DB.get_or_init(CutDb::load_default);
    let _rewritten = rewrite_gatefn_with_cut_db(
        black_box(&gate_fn),
        db.as_ref(),
        RewriteOptions {
            max_cuts_per_node: CUT_DB_REWRITE_MAX_CUTS_PER_NODE_CLI,
            max_iterations: CUT_DB_REWRITE_MAX_ITERATIONS_CLI,
        },
    );
}

fn ir2gates_mul_width_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ir2gates_umul_width");

    // Keep this modest; 64-bit multiplies can be quite expensive in debug builds.
    for width in [1usize, 2, 4, 8, 16, 32] {
        group.bench_with_input(BenchmarkId::new("umul", width), &width, |b, &w| {
            b.iter(|| run_ir2gates_like_flow_for_umul(w))
        });
    }

    group.finish();
}

criterion_group!(benches, ir2gates_mul_width_benchmark);
criterion_main!(benches);
