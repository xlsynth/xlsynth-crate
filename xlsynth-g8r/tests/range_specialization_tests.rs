// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use xlsynth_g8r::aig::get_summary_stats::get_summary_stats;
use xlsynth_g8r::aig_serdes::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::ir_range_info::IrRangeInfo;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;

fn build_pir_fn_and_range_info(ir_text: &str, top: &str) -> (ir::Fn, Arc<IrRangeInfo>) {
    let mut pir_parser = ir_parser::Parser::new(ir_text);
    let pir_package = pir_parser.parse_and_validate_package().unwrap();
    let pir_fn = pir_package.get_fn(top).unwrap().clone();

    let mut xlsynth_package = xlsynth::IrPackage::parse_ir(ir_text, None).unwrap();
    xlsynth_package.set_top_by_name(top).unwrap();
    let analysis = xlsynth_package.create_ir_analysis().unwrap();
    let range_info = IrRangeInfo::build_from_analysis(&analysis, &pir_fn).unwrap();

    (pir_fn, range_info)
}

#[test]
fn test_range_specializes_array_index_gate_count() {
    let ir_text = "package sample

top fn main(a0: bits[8] id=1, a1: bits[8] id=2, a2: bits[8] id=3, a3: bits[8] id=4, a4: bits[8] id=5, idx: bits[8] id=6) -> bits[8] {
  arr: bits[8][5] = array(a0, a1, a2, a3, a4, id=7)
  m: bits[8] = literal(value=5, id=8)
  bounded: bits[8] = umod(idx, m, id=9)
  ret r: bits[8] = array_index(arr, indices=[bounded], id=10)
}
";

    let (pir_fn, range_info) = build_pir_fn_and_range_info(ir_text, "main");

    let base = gatify(
        &pir_fn,
        GatifyOptions {
            fold: false,
            hash: false,
            check_equivalence: false,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: None,
        },
    )
    .unwrap();
    let base_stats = get_summary_stats(&base.gate_fn);

    let spec = gatify(
        &pir_fn,
        GatifyOptions {
            fold: false,
            hash: false,
            check_equivalence: false,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: Some(range_info),
        },
    )
    .unwrap();
    check_equivalence::validate_same_fn(&pir_fn, &spec.gate_fn).unwrap();
    let spec_stats = get_summary_stats(&spec.gate_fn);

    assert_eq!(
        base_stats.live_nodes, 1573,
        "baseline live_nodes={}",
        base_stats.live_nodes
    );
    assert_eq!(
        spec_stats.live_nodes, 1553,
        "specialized live_nodes={}",
        spec_stats.live_nodes
    );
}
