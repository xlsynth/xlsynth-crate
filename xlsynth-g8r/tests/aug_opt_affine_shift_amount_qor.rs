// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};
use xlsynth_pir::aug_opt::{AugOptMode, AugOptOptions, run_aug_opt_over_ir_text_with_stats};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;

#[derive(Debug, Clone, PartialEq, Eq)]
struct AffineShiftQorRow {
    op: &'static str,
    source_width: usize,
    amount_width: usize,
    k: usize,
    consumer: &'static str,
    old_and_nodes: usize,
    old_depth: usize,
    new_and_nodes: usize,
    new_depth: usize,
}

#[derive(Debug, Clone, Copy)]
struct AffineShiftQorCase {
    op: &'static str,
    source_width: usize,
    amount_width: usize,
    k: usize,
    start: usize,
    width: usize,
    consumer: &'static str,
}

fn parse_top_fn(ir_text: &str) -> ir::Fn {
    let mut parser = ir_parser::Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().expect("parse/validate");
    pkg.get_top_fn().expect("top fn").clone()
}

fn aug_opt_for_test(ir_text: &str) -> xlsynth_pir::aug_opt::AugOptRunResult {
    run_aug_opt_over_ir_text_with_stats(
        ir_text,
        Some("affine_shift_qor"),
        AugOptOptions {
            enable: true,
            rounds: 1,
            mode: AugOptMode::PirOnly,
        },
    )
    .expect("run aug-opt")
}

fn gatify_gate_fn(pir_fn: &ir::Fn) -> xlsynth_g8r::aig::GateFn {
    gatify_prepared_fn(
        pir_fn,
        GatifyOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            enable_rewrite_nary_add: false,
            enable_rewrite_mask_low: false,
            array_index_lowering_strategy: Default::default(),
        },
    )
    .expect("gatify_prepared_fn")
    .gate_fn
}

fn build_affine_shift_ir(case: AffineShiftQorCase) -> String {
    let high_width = case.amount_width - 1;
    format!(
        r#"package sample

top fn affine_shift_qor(x: bits[{source_width}] id=1, flag: bits[1] id=2) -> bits[{width}] {{
  zero_hi: bits[{high_width}] = literal(value=0, id=3)
  flag_ext: bits[{amount_width}] = concat(zero_hi, flag, id=4)
  k: bits[{amount_width}] = literal(value={k}, id=5)
  amount: bits[{amount_width}] = add(flag_ext, k, id=6)
  shifted: bits[{source_width}] = {op}(x, amount, id=7)
  ret out: bits[{width}] = bit_slice(shifted, start={start}, width={width}, id=8)
}}
"#,
        op = case.op,
        source_width = case.source_width,
        amount_width = case.amount_width,
        high_width = high_width,
        k = case.k,
        start = case.start,
        width = case.width,
    )
}

fn row_for_case(case: AffineShiftQorCase) -> AffineShiftQorRow {
    let ir_text = build_affine_shift_ir(case);
    let pir_fn = parse_top_fn(&ir_text);
    let result = aug_opt_for_test(&ir_text);
    let prepared = parse_top_fn(&result.output_text);
    let prepared_text = prepared.to_string();
    assert!(
        prepared_text.contains("sel(flag"),
        "expected affine shift rewrite for {case:?}; got:\n{prepared_text}"
    );

    let gate_old = gatify_gate_fn(&pir_fn);
    let gate_new = gatify_gate_fn(&prepared);
    check_equivalence::prove_same_gate_fn_via_ir(&gate_old, &gate_new)
        .unwrap_or_else(|e| panic!("gate equivalence failed for {case:?}: {e}"));
    check_equivalence::validate_same_fn(&pir_fn, &gate_new)
        .unwrap_or_else(|e| panic!("rewritten gate validation failed for {case:?}: {e}"));

    let old = get_aig_stats(&gate_old);
    let new = get_aig_stats(&gate_new);
    AffineShiftQorRow {
        op: case.op,
        source_width: case.source_width,
        amount_width: case.amount_width,
        k: case.k,
        consumer: case.consumer,
        old_and_nodes: old.and_nodes,
        old_depth: old.max_depth,
        new_and_nodes: new.and_nodes,
        new_depth: new.max_depth,
    }
}

fn gather_qor_rows() -> Vec<AffineShiftQorRow> {
    [
        AffineShiftQorCase {
            op: "shrl",
            source_width: 12,
            amount_width: 4,
            k: 3,
            start: 0,
            width: 7,
            consumer: "low_bitslice",
        },
        AffineShiftQorCase {
            op: "shll",
            source_width: 12,
            amount_width: 4,
            k: 3,
            start: 2,
            width: 7,
            consumer: "low_mid_bitslice",
        },
        AffineShiftQorCase {
            op: "shra",
            source_width: 12,
            amount_width: 4,
            k: 3,
            start: 5,
            width: 7,
            consumer: "sign_fill_bitslice",
        },
    ]
    .into_iter()
    .map(row_for_case)
    .collect()
}

#[test]
fn affine_shift_qor_characterization_covers_all_shift_ops() {
    let got = gather_qor_rows();

    for row in &got {
        assert!(
            row.new_and_nodes < row.old_and_nodes,
            "expected affine shift rewrite to reduce AND nodes: {row:?}"
        );
        assert!(
            row.new_depth <= row.old_depth,
            "expected affine shift rewrite not to increase depth: {row:?}"
        );
    }

    #[rustfmt::skip]
    let want: &[AffineShiftQorRow] = &[
        AffineShiftQorRow { op: "shrl", source_width: 12, amount_width: 4, k: 3, consumer: "low_bitslice", old_and_nodes: 86, old_depth: 6, new_and_nodes: 21, new_depth: 2 },
        AffineShiftQorRow { op: "shll", source_width: 12, amount_width: 4, k: 3, consumer: "low_mid_bitslice", old_and_nodes: 65, old_depth: 6, new_and_nodes: 16, new_depth: 2 },
        AffineShiftQorRow { op: "shra", source_width: 12, amount_width: 4, k: 3, consumer: "sign_fill_bitslice", old_and_nodes: 56, old_depth: 6, new_and_nodes: 11, new_depth: 2 },
    ];
    assert_eq!(got, want);
}
