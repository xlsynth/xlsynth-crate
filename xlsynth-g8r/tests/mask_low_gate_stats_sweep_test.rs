// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::{SummaryStats, get_summary_stats};
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::ir2gates;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;
use xlsynth_pir::math::ceil_log2;

#[derive(Clone, Debug, PartialEq, Eq)]
struct MaskLowRow {
    output_width: usize,
    live_nodes: usize,
    deepest_path: usize,
}

fn parse_top_fn(ir_text: &str) -> ir::Fn {
    let mut parser = ir_parser::Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().expect("parse/validate");
    pkg.get_top_fn().expect("top fn").clone()
}

fn build_ext_mask_low_ir_text(output_width: usize) -> String {
    let count_width = ceil_log2(output_width.saturating_add(1)).saturating_add(2);
    format!(
        "package sample\n\
top fn ext_mask_low_{output_width}b(count: bits[{count_width}] id=1) -> bits[{output_width}] {{\n\
  ret ext_mask_low.2: bits[{output_width}] = ext_mask_low(count, id=2)\n\
}}\n"
    )
}

fn build_shift_sub_mask_low_ir_text(output_width: usize) -> String {
    let count_width = ceil_log2(output_width.saturating_add(1)).saturating_add(2);
    format!(
        "package sample\n\
top fn shift_sub_mask_low_{output_width}b(count: bits[{count_width}] id=1) -> bits[{output_width}] {{\n\
  one: bits[{output_width}] = literal(value=1, id=2)\n\
  sh: bits[{output_width}] = shll(one, count, id=3)\n\
  ret mask: bits[{output_width}] = sub(sh, one, id=4)\n\
}}\n"
    )
}

fn stats_for_ext_mask_low(output_width: usize) -> SummaryStats {
    let pir_fn = parse_top_fn(&build_ext_mask_low_ir_text(output_width));
    let gate_fn = gatify(
        &pir_fn,
        GatifyOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            adder_mapping: AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            enable_rewrite_nary_add: false,
            enable_rewrite_mask_low: false,
            array_index_lowering_strategy: Default::default(),
        },
    )
    .expect("gatify")
    .gate_fn;
    get_summary_stats(&gate_fn)
}

fn stats_for_ir_text_via_ir2gates(ir_text: &str, enable_rewrite_mask_low: bool) -> (usize, usize) {
    let out = ir2gates::ir2gates_from_ir_text(
        ir_text,
        None,
        ir2gates::Ir2GatesOptions {
            fold: true,
            hash: true,
            check_equivalence: true,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            enable_rewrite_nary_add: false,
            enable_rewrite_mask_low,
            adder_mapping: AdderMapping::default(),
            mul_adder_mapping: None,
            aug_opt: Default::default(),
        },
    )
    .expect("ir2gates");
    let s = get_summary_stats(&out.gatify_output.gate_fn);
    (s.live_nodes, s.deepest_path)
}

fn gather_ext_mask_low_rows() -> Vec<MaskLowRow> {
    let mut got = Vec::new();
    for output_width in 1usize..=16 {
        let stats = stats_for_ext_mask_low(output_width);
        got.push(MaskLowRow {
            output_width,
            live_nodes: stats.live_nodes,
            deepest_path: stats.deepest_path,
        });
    }
    got
}

#[test]
fn test_ext_mask_low_gate_stats_sweep_1_to_16() {
    let got = gather_ext_mask_low_rows();

    #[rustfmt::skip]
    let want: &[MaskLowRow] = &[
        MaskLowRow { output_width: 1, live_nodes: 5, deepest_path: 3 },
        MaskLowRow { output_width: 2, live_nodes: 7, deepest_path: 4 },
        MaskLowRow { output_width: 3, live_nodes: 10, deepest_path: 3 },
        MaskLowRow { output_width: 4, live_nodes: 12, deepest_path: 4 },
        MaskLowRow { output_width: 5, live_nodes: 17, deepest_path: 4 },
        MaskLowRow { output_width: 6, live_nodes: 19, deepest_path: 4 },
        MaskLowRow { output_width: 7, live_nodes: 21, deepest_path: 4 },
        MaskLowRow { output_width: 8, live_nodes: 23, deepest_path: 4 },
        MaskLowRow { output_width: 9, live_nodes: 32, deepest_path: 5 },
        MaskLowRow { output_width: 10, live_nodes: 34, deepest_path: 5 },
        MaskLowRow { output_width: 11, live_nodes: 36, deepest_path: 5 },
        MaskLowRow { output_width: 12, live_nodes: 38, deepest_path: 5 },
        MaskLowRow { output_width: 13, live_nodes: 40, deepest_path: 5 },
        MaskLowRow { output_width: 14, live_nodes: 42, deepest_path: 5 },
        MaskLowRow { output_width: 15, live_nodes: 44, deepest_path: 5 },
        MaskLowRow { output_width: 16, live_nodes: 46, deepest_path: 5 },
    ];

    assert_eq!(got.as_slice(), want);
}

#[test]
fn ir2gates_shift_sub_mask_low_qor_improves_with_mask_low_rewrite() {
    for output_width in [8usize, 16, 32] {
        let ir_text = build_shift_sub_mask_low_ir_text(output_width);
        let (nodes_off, depth_off) = stats_for_ir_text_via_ir2gates(&ir_text, false);
        let (nodes_on, depth_on) = stats_for_ir_text_via_ir2gates(&ir_text, true);

        assert!(
            nodes_on < nodes_off,
            "expected enable_rewrite_mask_low=true to reduce live_nodes for output_width={output_width}; off={nodes_off} on={nodes_on} (depth off={depth_off} on={depth_on})"
        );
    }
}
