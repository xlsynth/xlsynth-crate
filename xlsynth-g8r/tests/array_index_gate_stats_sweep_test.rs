// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::{SummaryStats, get_summary_stats};
use xlsynth_g8r::gatify::ir2gate::{ArrayIndexLoweringStrategy, GatifyOptions, gatify};
use xlsynth_g8r::test_utils::Opt;
use xlsynth_pir::ir_parser;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ArrayIndexRow {
    element_width: u32,
    live_nodes_ohs: usize,
    deepest_path_ohs: usize,
    live_nodes_mux: usize,
    deepest_path_mux: usize,
    live_nodes_auto: usize,
    deepest_path_auto: usize,
}

fn build_array_index_ir_text(array_len: u32, element_width: u32, index_width: u32) -> String {
    format!(
        r#"package sample

top fn main(arr: bits[{element_width}][{array_len}], idx: bits[{index_width}]) -> bits[{element_width}] {{
  ret r: bits[{element_width}] = array_index(arr, indices=[idx], id=3)
}}
"#
    )
}

fn stats_for_ir_text_with_strategy(
    ir_text: &str,
    opt: Opt,
    strategy: ArrayIndexLoweringStrategy,
) -> SummaryStats {
    let mut parser = ir_parser::Parser::new(ir_text);
    let ir_package = parser.parse_and_validate_package().expect("parse package");
    let ir_fn = ir_package.get_top_fn().expect("top fn");
    let out = gatify(
        &ir_fn,
        GatifyOptions {
            fold: opt == Opt::Yes,
            check_equivalence: false,
            hash: opt == Opt::Yes,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::RippleCarry,
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            enable_rewrite_nary_add: false,
            array_index_lowering_strategy: strategy,
        },
    )
    .expect("gatify");
    get_summary_stats(&out.gate_fn)
}

fn gather_array_index_rows() -> Vec<ArrayIndexRow> {
    let widths = [1u32, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32];
    let mut got = Vec::new();
    for element_width in widths {
        let ir_text = build_array_index_ir_text(
            /* array_len= */ 27,
            element_width,
            /* index_width= */ 5,
        );
        let ohs = stats_for_ir_text_with_strategy(
            &ir_text,
            Opt::Yes,
            ArrayIndexLoweringStrategy::ForceOobOneHot,
        );
        let mux = stats_for_ir_text_with_strategy(
            &ir_text,
            Opt::Yes,
            ArrayIndexLoweringStrategy::ForceNearPow2MuxTree,
        );
        let auto =
            stats_for_ir_text_with_strategy(&ir_text, Opt::Yes, ArrayIndexLoweringStrategy::Auto);
        got.push(ArrayIndexRow {
            element_width,
            live_nodes_ohs: ohs.live_nodes,
            deepest_path_ohs: ohs.deepest_path,
            live_nodes_mux: mux.live_nodes,
            deepest_path_mux: mux.deepest_path,
            live_nodes_auto: auto.live_nodes,
            deepest_path_auto: auto.deepest_path,
        });
    }
    got
}

#[test]
fn test_array_index_gate_stats_sweep_27elem_width1_to_32() {
    let _ = env_logger::builder().is_test(true).try_init();

    let got = gather_array_index_rows();

    for row in &got {
        assert!(
            row.deepest_path_mux < row.deepest_path_ohs,
            "expected mux-tree to reduce depth for {}-bit elements: {:?}",
            row.element_width,
            row
        );
        let ohs_product = row.live_nodes_ohs * row.deepest_path_ohs;
        let mux_product = row.live_nodes_mux * row.deepest_path_mux;
        assert!(
            mux_product < ohs_product,
            "expected mux-tree to reduce nodes*depth for {}-bit elements: ohs={} mux={} row={:?}",
            row.element_width,
            ohs_product,
            mux_product,
            row
        );
    }

    #[rustfmt::skip]
    let want: &[ArrayIndexRow] = &[
        ArrayIndexRow { element_width: 1, live_nodes_ohs: 155, deepest_path_ohs: 15, live_nodes_mux: 118, deepest_path_mux: 11, live_nodes_auto: 118, deepest_path_auto: 11 },
        ArrayIndexRow { element_width: 2, live_nodes_ohs: 237, deepest_path_ohs: 15, live_nodes_mux: 231, deepest_path_mux: 11, live_nodes_auto: 231, deepest_path_auto: 11 },
        ArrayIndexRow { element_width: 3, live_nodes_ohs: 319, deepest_path_ohs: 15, live_nodes_mux: 344, deepest_path_mux: 11, live_nodes_auto: 319, deepest_path_auto: 15 },
        ArrayIndexRow { element_width: 4, live_nodes_ohs: 401, deepest_path_ohs: 15, live_nodes_mux: 457, deepest_path_mux: 11, live_nodes_auto: 401, deepest_path_auto: 15 },
        ArrayIndexRow { element_width: 5, live_nodes_ohs: 483, deepest_path_ohs: 15, live_nodes_mux: 570, deepest_path_mux: 11, live_nodes_auto: 483, deepest_path_auto: 15 },
        ArrayIndexRow { element_width: 6, live_nodes_ohs: 565, deepest_path_ohs: 15, live_nodes_mux: 683, deepest_path_mux: 11, live_nodes_auto: 565, deepest_path_auto: 15 },
        ArrayIndexRow { element_width: 8, live_nodes_ohs: 729, deepest_path_ohs: 15, live_nodes_mux: 909, deepest_path_mux: 11, live_nodes_auto: 729, deepest_path_auto: 15 },
        ArrayIndexRow { element_width: 12, live_nodes_ohs: 1057, deepest_path_ohs: 15, live_nodes_mux: 1361, deepest_path_mux: 11, live_nodes_auto: 1057, deepest_path_auto: 15 },
        ArrayIndexRow { element_width: 16, live_nodes_ohs: 1385, deepest_path_ohs: 15, live_nodes_mux: 1813, deepest_path_mux: 11, live_nodes_auto: 1385, deepest_path_auto: 15 },
        ArrayIndexRow { element_width: 24, live_nodes_ohs: 2041, deepest_path_ohs: 15, live_nodes_mux: 2717, deepest_path_mux: 11, live_nodes_auto: 2041, deepest_path_auto: 15 },
        ArrayIndexRow { element_width: 32, live_nodes_ohs: 2697, deepest_path_ohs: 15, live_nodes_mux: 3621, deepest_path_mux: 11, live_nodes_auto: 2697, deepest_path_auto: 15 },
    ];

    assert_eq!(got.as_slice(), want);
}
