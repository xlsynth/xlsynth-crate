// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::get_summary_stats;
use xlsynth_g8r::aig_serdes::ir2gate;
use xlsynth_g8r::ir2gates;
use xlsynth_pir::ir_parser;

#[test]
fn test_ir2gates_from_ir_text_enables_analysis_specialization() {
    let ir_text = "package sample

top fn main(a0: bits[8] id=1, a1: bits[8] id=2, a2: bits[8] id=3, a3: bits[8] id=4, a4: bits[8] id=5, idx: bits[8] id=6) -> bits[8] {
  arr: bits[8][5] = array(a0, a1, a2, a3, a4, id=7)
  m: bits[8] = literal(value=5, id=8)
  bounded: bits[8] = umod(idx, m, id=9)
  ret r: bits[8] = array_index(arr, indices=[bounded], id=10)
}
";

    let out = ir2gates::ir2gates_from_ir_text(
        ir_text,
        None,
        ir2gates::Ir2GatesOptions {
            fold: false,
            hash: false,
            check_equivalence: false,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
        },
    )
    .unwrap();
    let got = get_summary_stats(&out.gatify_output.gate_fn);
    assert_eq!(got.live_nodes, 1553);

    // Compare against a baseline that explicitly disables range_info.
    let mut pir_parser = ir_parser::Parser::new(ir_text);
    let pir_pkg = pir_parser.parse_and_validate_package().unwrap();
    let pir_fn = pir_pkg.get_top_fn().unwrap();
    let base = ir2gate::gatify(
        pir_fn,
        ir2gate::GatifyOptions {
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
    assert_eq!(base_stats.live_nodes, 1573);
}
