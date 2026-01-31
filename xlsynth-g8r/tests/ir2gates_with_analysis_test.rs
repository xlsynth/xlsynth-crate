// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::{get_aig_stats, get_summary_stats};
use xlsynth_g8r::gatify::ir2gate;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
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
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            aug_opt: Default::default(),
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
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
        },
    )
    .unwrap();
    let base_stats = get_summary_stats(&base.gate_fn);
    assert_eq!(base_stats.live_nodes, 1573);
}

#[test]
fn test_ir2gates_umul_ugt_levels_and_nodes() {
    let _ = env_logger::builder().is_test(true).try_init();
    let ir_text = "package sample

top fn f(x: bits[32] id=1, y: bits[7] id=2, z: bits[32] id=3) -> bits[1] {
  a: bits[32] = umul(y, x, id=4)
  ret result: bits[1] = ugt(a, z, id=5)
}
";

    let out = ir2gates::ir2gates_from_ir_text(
        ir_text,
        None,
        ir2gates::Ir2GatesOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            adder_mapping: AdderMapping::default(),
            mul_adder_mapping: None,
            aug_opt: Default::default(),
        },
    )
    .unwrap();

    let got = get_summary_stats(&out.gatify_output.gate_fn);
    // This is a "microbenchmark sweep" style test: lock in the expected gate
    // count + depth so we can notice regressions and improvements.
    assert_eq!(got.live_nodes, 2397);
    assert_eq!(got.deepest_path, 50);
}

#[test]
fn test_ir2gates_umul_commutes_for_stats() {
    let _ = env_logger::builder().is_test(true).try_init();

    let ir_text_xy = "package sample

top fn f(x: bits[32] id=1, y: bits[7] id=2, z: bits[32] id=3) -> bits[1] {
  a: bits[32] = umul(x, y, id=4)
  ret result: bits[1] = ugt(a, z, id=5)
}
";
    let ir_text_yx = "package sample

top fn f(x: bits[32] id=1, y: bits[7] id=2, z: bits[32] id=3) -> bits[1] {
  a: bits[32] = umul(y, x, id=4)
  ret result: bits[1] = ugt(a, z, id=5)
}
";

    let out_xy = ir2gates::ir2gates_from_ir_text(
        ir_text_xy,
        None,
        ir2gates::Ir2GatesOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            adder_mapping: AdderMapping::default(),
            mul_adder_mapping: None,
            aug_opt: Default::default(),
        },
    )
    .unwrap();

    let out_yx = ir2gates::ir2gates_from_ir_text(
        ir_text_yx,
        None,
        ir2gates::Ir2GatesOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            adder_mapping: AdderMapping::default(),
            mul_adder_mapping: None,
            aug_opt: Default::default(),
        },
    )
    .unwrap();

    let stats_xy = get_summary_stats(&out_xy.gatify_output.gate_fn);
    let stats_yx = get_summary_stats(&out_yx.gatify_output.gate_fn);
    assert_eq!(stats_xy, stats_yx);
}

#[test]
fn test_ir2gates_ugt_against_ir_literal_threshold_shape() {
    // Historical reference baseline (tool-independent):
    // - For this cone we expect and=8 lev=6.
    let _ = env_logger::builder().is_test(true).try_init();

    let ir_text = "package bool_cone

top fn cone(leaf_94: bits[9] id=1) -> bits[1] {
  literal.2: bits[9] = literal(value=72, id=2)
  ret ugt.3: bits[1] = ugt(leaf_94, literal.2, id=3)
}
";

    let out = ir2gates::ir2gates_from_ir_text(
        ir_text,
        None,
        ir2gates::Ir2GatesOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            adder_mapping: AdderMapping::default(),
            mul_adder_mapping: None,
            aug_opt: Default::default(),
        },
    )
    .unwrap();

    let stats = get_aig_stats(&out.gatify_output.gate_fn);
    assert_eq!(stats.and_nodes, 8);
    assert_eq!(stats.max_depth, 6);
}
