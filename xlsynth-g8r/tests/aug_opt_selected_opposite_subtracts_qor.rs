// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::prove_gate_fn_equiv_common::EquivResult;
use xlsynth_g8r::prove_gate_fn_equiv_sat::{
    Ctx as VarisatCtx, prove_gate_fn_equiv as prove_gate_fn_equiv_sat,
};
use xlsynth_pir::aug_opt::{AugOptMode, AugOptOptions, run_aug_opt_over_ir_text_with_stats};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;

#[derive(Debug, Clone, PartialEq, Eq)]
struct SelectedOppositeSubtractsQorRow {
    width: usize,
    mapping: &'static str,
    original_and_nodes: usize,
    original_depth: usize,
    hoisted_and_nodes: usize,
    hoisted_depth: usize,
}

fn parse_top_fn(ir_text: &str) -> ir::Fn {
    let mut parser = ir_parser::Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().expect("parse/validate");
    pkg.get_top_fn().expect("top fn").clone()
}

fn aug_opt_for_test(ir_text: &str) -> xlsynth_pir::aug_opt::AugOptRunResult {
    run_aug_opt_over_ir_text_with_stats(
        ir_text,
        Some("selected_sub_qor"),
        AugOptOptions {
            enable: true,
            rounds: 1,
            mode: AugOptMode::PirOnly,
        },
    )
    .expect("run aug-opt")
}

fn gatify_gate_fn(pir_fn: &ir::Fn, adder_mapping: AdderMapping) -> xlsynth_g8r::aig::GateFn {
    gatify_prepared_fn(
        pir_fn,
        GatifyOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            adder_mapping,
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            enable_rewrite_nary_add: false,
            enable_rewrite_mask_low: false,
            array_index_lowering_strategy: Default::default(),
            unsafe_gatify_gate_operation: false,
        },
    )
    .expect("gatify_prepared_fn")
    .gate_fn
}

fn build_selected_sub_ir(width: usize) -> String {
    format!(
        r#"package sample

top fn selected_sub_qor(p: bits[1] id=1, a: bits[{width}] id=2, b: bits[{width}] id=3) -> bits[{width}] {{
  ba: bits[{width}] = sub(b, a, id=4)
  ab: bits[{width}] = sub(a, b, id=5)
  ret out: bits[{width}] = sel(p, cases=[ba, ab], id=6)
}}
"#
    )
}

fn build_hoisted_selected_sub_ir(width: usize) -> String {
    format!(
        r#"package sample

top fn selected_sub_qor(p: bits[1] id=1, a: bits[{width}] id=2, b: bits[{width}] id=3) -> bits[{width}] {{
  lhs: bits[{width}] = sel(p, cases=[b, a], id=4)
  rhs: bits[{width}] = sel(p, cases=[a, b], id=5)
  ret out: bits[{width}] = sub(lhs, rhs, id=6)
}}
"#
    )
}

fn adder_mapping_from_name(name: &str) -> AdderMapping {
    match name {
        "brent_kung" => AdderMapping::BrentKung,
        "kogge_stone" => AdderMapping::KoggeStone,
        "ripple" => AdderMapping::RippleCarry,
        _ => panic!("unknown adder mapping: {name}"),
    }
}

fn assert_gate_equiv_via_sat(
    lhs: &xlsynth_g8r::aig::GateFn,
    rhs: &xlsynth_g8r::aig::GateFn,
    context: &str,
) {
    let mut ctx = VarisatCtx::new();
    match prove_gate_fn_equiv_sat(lhs, rhs, &mut ctx) {
        EquivResult::Proved => {}
        EquivResult::Disproved(cex) => {
            panic!("gate equivalence failed for {context}; counterexample inputs: {cex:?}");
        }
    }
}

fn run_sat_proof_for_width(width: usize) -> bool {
    matches!(width, 1 | 2 | 3 | 4 | 8 | 16)
}

fn row_for_case(width: usize, mapping: &'static str) -> SelectedOppositeSubtractsQorRow {
    let adder_mapping = adder_mapping_from_name(mapping);
    let ir_text = build_selected_sub_ir(width);
    let hoisted_ir_text = build_hoisted_selected_sub_ir(width);
    let pir_fn = parse_top_fn(&ir_text);
    let hoisted_fn = parse_top_fn(&hoisted_ir_text);

    let result = aug_opt_for_test(&ir_text);
    let expected_rewrites = usize::from(width >= 3);
    assert_eq!(
        result.rewrite_stats.selected_opposite_subtracts, expected_rewrites,
        "unexpected selected-opposite-subtract rewrite count for width={width}; output:\n{}",
        result.output_text
    );
    let actual_rewritten_fn = parse_top_fn(&result.output_text);

    let gate_original = gatify_gate_fn(&pir_fn, adder_mapping_from_name(mapping));
    let gate_hoisted = gatify_gate_fn(&hoisted_fn, adder_mapping);
    let gate_actual = gatify_gate_fn(&actual_rewritten_fn, adder_mapping_from_name(mapping));
    if run_sat_proof_for_width(width) {
        assert_gate_equiv_via_sat(
            &gate_original,
            &gate_hoisted,
            &format!("manual hoist width={width} mapping={mapping}"),
        );
    }

    let original = get_aig_stats(&gate_original);
    let hoisted = get_aig_stats(&gate_hoisted);
    let actual = get_aig_stats(&gate_actual);
    if width >= 3 {
        if run_sat_proof_for_width(width) {
            assert_gate_equiv_via_sat(
                &gate_hoisted,
                &gate_actual,
                &format!("manual hoist vs aug-opt width={width} mapping={mapping}"),
            );
        }
        assert_eq!(
            (actual.and_nodes, actual.max_depth),
            (hoisted.and_nodes, hoisted.max_depth),
            "aug-opt rewritten QoR should match manual hoist for width={width} mapping={mapping}"
        );
        assert!(
            hoisted.and_nodes < original.and_nodes,
            "expected selected subtract hoist to reduce AND nodes for width={width} mapping={mapping}: original={original:?} hoisted={hoisted:?}"
        );
        assert!(
            hoisted.max_depth <= original.max_depth,
            "expected selected subtract hoist not to increase depth for width={width} mapping={mapping}: original={original:?} hoisted={hoisted:?}"
        );
    } else {
        assert_eq!(
            (actual.and_nodes, actual.max_depth),
            (original.and_nodes, original.max_depth),
            "width={width} should be skipped by aug-opt for mapping={mapping}"
        );
        assert!(
            hoisted.and_nodes >= original.and_nodes,
            "manual hoist should not be profitable at guarded width={width} mapping={mapping}: original={original:?} hoisted={hoisted:?}"
        );
    }

    SelectedOppositeSubtractsQorRow {
        width,
        mapping,
        original_and_nodes: original.and_nodes,
        original_depth: original.max_depth,
        hoisted_and_nodes: hoisted.and_nodes,
        hoisted_depth: hoisted.max_depth,
    }
}

fn gather_qor_rows() -> Vec<SelectedOppositeSubtractsQorRow> {
    let mut rows = Vec::new();
    for width in [1usize, 2, 3, 4, 8, 16, 32, 64, 128, 256] {
        for mapping in ["brent_kung", "kogge_stone", "ripple"] {
            rows.push(row_for_case(width, mapping));
        }
    }
    rows
}

#[test]
fn selected_opposite_subtracts_qor_and_equivalence_sweep() {
    let got = gather_qor_rows();

    #[rustfmt::skip]
    let want: &[SelectedOppositeSubtractsQorRow] = &[
        SelectedOppositeSubtractsQorRow { width: 1, mapping: "brent_kung", original_and_nodes: 6, original_depth: 4, hoisted_and_nodes: 9, hoisted_depth: 4 },
        SelectedOppositeSubtractsQorRow { width: 1, mapping: "kogge_stone", original_and_nodes: 6, original_depth: 4, hoisted_and_nodes: 9, hoisted_depth: 4 },
        SelectedOppositeSubtractsQorRow { width: 1, mapping: "ripple", original_and_nodes: 6, original_depth: 4, hoisted_and_nodes: 9, hoisted_depth: 4 },
        SelectedOppositeSubtractsQorRow { width: 2, mapping: "brent_kung", original_and_nodes: 22, original_depth: 7, hoisted_and_nodes: 23, hoisted_depth: 7 },
        SelectedOppositeSubtractsQorRow { width: 2, mapping: "kogge_stone", original_and_nodes: 22, original_depth: 7, hoisted_and_nodes: 23, hoisted_depth: 7 },
        SelectedOppositeSubtractsQorRow { width: 2, mapping: "ripple", original_and_nodes: 20, original_depth: 6, hoisted_and_nodes: 22, hoisted_depth: 6 },
        SelectedOppositeSubtractsQorRow { width: 3, mapping: "brent_kung", original_and_nodes: 43, original_depth: 9, hoisted_and_nodes: 40, hoisted_depth: 9 },
        SelectedOppositeSubtractsQorRow { width: 3, mapping: "kogge_stone", original_and_nodes: 43, original_depth: 9, hoisted_and_nodes: 40, hoisted_depth: 9 },
        SelectedOppositeSubtractsQorRow { width: 3, mapping: "ripple", original_and_nodes: 38, original_depth: 8, hoisted_and_nodes: 37, hoisted_depth: 8 },
        SelectedOppositeSubtractsQorRow { width: 4, mapping: "brent_kung", original_and_nodes: 64, original_depth: 11, hoisted_and_nodes: 57, hoisted_depth: 11 },
        SelectedOppositeSubtractsQorRow { width: 4, mapping: "kogge_stone", original_and_nodes: 69, original_depth: 10, hoisted_and_nodes: 60, hoisted_depth: 10 },
        SelectedOppositeSubtractsQorRow { width: 4, mapping: "ripple", original_and_nodes: 56, original_depth: 10, hoisted_and_nodes: 52, hoisted_depth: 10 },
        SelectedOppositeSubtractsQorRow { width: 8, mapping: "brent_kung", original_and_nodes: 158, original_depth: 15, hoisted_and_nodes: 131, hoisted_depth: 15 },
        SelectedOppositeSubtractsQorRow { width: 8, mapping: "kogge_stone", original_and_nodes: 188, original_depth: 12, hoisted_and_nodes: 149, hoisted_depth: 12 },
        SelectedOppositeSubtractsQorRow { width: 8, mapping: "ripple", original_and_nodes: 128, original_depth: 18, hoisted_and_nodes: 112, hoisted_depth: 18 },
        SelectedOppositeSubtractsQorRow { width: 16, mapping: "brent_kung", original_and_nodes: 356, original_depth: 19, hoisted_and_nodes: 285, hoisted_depth: 19 },
        SelectedOppositeSubtractsQorRow { width: 16, mapping: "kogge_stone", original_and_nodes: 471, original_depth: 14, hoisted_and_nodes: 354, hoisted_depth: 14 },
        SelectedOppositeSubtractsQorRow { width: 16, mapping: "ripple", original_and_nodes: 272, original_depth: 34, hoisted_and_nodes: 232, hoisted_depth: 34 },
        SelectedOppositeSubtractsQorRow { width: 32, mapping: "brent_kung", original_and_nodes: 762, original_depth: 23, hoisted_and_nodes: 599, hoisted_depth: 23 },
        SelectedOppositeSubtractsQorRow { width: 32, mapping: "kogge_stone", original_and_nodes: 1122, original_depth: 16, hoisted_and_nodes: 815, hoisted_depth: 16 },
        SelectedOppositeSubtractsQorRow { width: 32, mapping: "ripple", original_and_nodes: 560, original_depth: 66, hoisted_and_nodes: 472, hoisted_depth: 66 },
        SelectedOppositeSubtractsQorRow { width: 64, mapping: "brent_kung", original_and_nodes: 1584, original_depth: 27, hoisted_and_nodes: 1233, hoisted_depth: 27 },
        SelectedOppositeSubtractsQorRow { width: 64, mapping: "kogge_stone", original_and_nodes: 2589, original_depth: 18, hoisted_and_nodes: 1836, hoisted_depth: 18 },
        SelectedOppositeSubtractsQorRow { width: 64, mapping: "ripple", original_and_nodes: 1136, original_depth: 130, hoisted_and_nodes: 952, hoisted_depth: 130 },
        SelectedOppositeSubtractsQorRow { width: 128, mapping: "brent_kung", original_and_nodes: 3238, original_depth: 31, hoisted_and_nodes: 2507, hoisted_depth: 31 },
        SelectedOppositeSubtractsQorRow { width: 128, mapping: "kogge_stone", original_and_nodes: 5848, original_depth: 20, hoisted_and_nodes: 4073, hoisted_depth: 20 },
        SelectedOppositeSubtractsQorRow { width: 128, mapping: "ripple", original_and_nodes: 2288, original_depth: 258, hoisted_and_nodes: 1912, hoisted_depth: 258 },
        SelectedOppositeSubtractsQorRow { width: 256, mapping: "brent_kung", original_and_nodes: 6556, original_depth: 35, hoisted_and_nodes: 5061, hoisted_depth: 35 },
        SelectedOppositeSubtractsQorRow { width: 256, mapping: "kogge_stone", original_and_nodes: 13011, original_depth: 22, hoisted_and_nodes: 8934, hoisted_depth: 22 },
        SelectedOppositeSubtractsQorRow { width: 256, mapping: "ripple", original_and_nodes: 4592, original_depth: 514, hoisted_and_nodes: 3832, hoisted_depth: 514 },
    ];
    assert_eq!(got, want);
}
