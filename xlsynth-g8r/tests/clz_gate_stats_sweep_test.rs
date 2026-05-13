// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::{SummaryStats, get_summary_stats};
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;
use xlsynth_pir::math::ceil_log2;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ClzRow {
    bit_count: u32,
    live_nodes: usize,
    deepest_path: usize,
}

fn parse_top_fn(ir_text: &str) -> ir::Fn {
    let mut parser = ir_parser::Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().expect("parse/validate");
    pkg.get_top_fn().expect("top fn").clone()
}

fn build_ext_clz_ir_text(bit_count: u32) -> String {
    let out_w = ceil_log2((bit_count as usize).saturating_add(1));
    format!(
        "package sample\n\
top fn ext_clz_{bit_count}b(input: bits[{bit_count}] id=1) -> bits[{out_w}] {{\n\
  ret ext_clz.2: bits[{out_w}] = ext_clz(input, id=2)\n\
}}\n"
    )
}

fn stats_for_ext_clz(bit_count: u32) -> SummaryStats {
    let pir_fn = parse_top_fn(&build_ext_clz_ir_text(bit_count));
    let gate_fn = gatify(
        &pir_fn,
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
            unsafe_gatify_gate_operation: false,
        },
    )
    .expect("gatify")
    .gate_fn;
    get_summary_stats(&gate_fn)
}

fn gather_ext_clz_rows() -> Vec<ClzRow> {
    let mut got: Vec<ClzRow> = Vec::new();
    for bit_count in 1u32..=16 {
        let stats = stats_for_ext_clz(bit_count);
        got.push(ClzRow {
            bit_count,
            live_nodes: stats.live_nodes,
            deepest_path: stats.deepest_path,
        });
    }
    got
}

#[test]
fn test_ext_clz_gate_stats_sweep_1_to_16() {
    let _ = env_logger::builder().is_test(true).try_init();

    let got = gather_ext_clz_rows();

    #[rustfmt::skip]
    let want: &[ClzRow] = &[
        ClzRow { bit_count: 1, live_nodes: 1, deepest_path: 1 },
        ClzRow { bit_count: 2, live_nodes: 5, deepest_path: 4 },
        ClzRow { bit_count: 3, live_nodes: 13, deepest_path: 7 },
        ClzRow { bit_count: 4, live_nodes: 13, deepest_path: 6 },
        ClzRow { bit_count: 5, live_nodes: 25, deepest_path: 9 },
        ClzRow { bit_count: 6, live_nodes: 26, deepest_path: 8 },
        ClzRow { bit_count: 7, live_nodes: 31, deepest_path: 9 },
        ClzRow { bit_count: 8, live_nodes: 30, deepest_path: 8 },
        ClzRow { bit_count: 9, live_nodes: 46, deepest_path: 11 },
        ClzRow { bit_count: 10, live_nodes: 47, deepest_path: 10 },
        ClzRow { bit_count: 11, live_nodes: 52, deepest_path: 11 },
        ClzRow { bit_count: 12, live_nodes: 53, deepest_path: 10 },
        ClzRow { bit_count: 13, live_nodes: 61, deepest_path: 11 },
        ClzRow { bit_count: 14, live_nodes: 62, deepest_path: 10 },
        ClzRow { bit_count: 15, live_nodes: 67, deepest_path: 11 },
        ClzRow { bit_count: 16, live_nodes: 65, deepest_path: 10 },
    ];

    assert_eq!(got.as_slice(), want);
}
