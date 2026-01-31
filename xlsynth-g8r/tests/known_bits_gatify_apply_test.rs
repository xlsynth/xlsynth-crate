// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_range_info::IrRangeInfo;

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

fn is_literal_false(op: &xlsynth_g8r::aig::gate::AigOperand) -> bool {
    op.node.id == 0 && !op.negated
}

#[test]
fn test_gatify_applies_known_bits_from_range_info() {
    let ir_text = "package sample

top fn main(x: bits[8] id=1) -> bits[8] {
  m: bits[8] = literal(value=4, id=2)
  r: bits[8] = umod(x, m, id=3)
  ret out: bits[8] = identity(r, id=4)
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
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
        },
    )
    .unwrap();

    let spec = gatify(
        &pir_fn,
        GatifyOptions {
            fold: false,
            hash: false,
            check_equivalence: false,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: Some(range_info),
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
        },
    )
    .unwrap();

    let base_out = &base.gate_fn.outputs[0].bit_vector;
    let spec_out = &spec.gate_fn.outputs[0].bit_vector;

    let base_high_all_zero = (2..8).all(|i| is_literal_false(base_out.get_lsb(i)));
    let spec_high_all_zero = (2..8).all(|i| is_literal_false(spec_out.get_lsb(i)));

    assert!(
        !base_high_all_zero,
        "baseline unexpectedly folded all known-zero bits"
    );
    assert!(
        spec_high_all_zero,
        "range info should force known-zero bits to literals"
    );
}
