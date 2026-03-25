// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::ir2gates;

fn build_ext_nary_add_ir_text(arch: &str) -> String {
    format!(
        r#"package test

fn f(a: bits[32] id=1, b: bits[32] id=2) -> bits[32] {{
  ret r: bits[32] = ext_nary_add(a, b, arch={arch}, id=3)
}}
"#
    )
}

fn get_ext_nary_add_gate_count(mapping: AdderMapping) -> usize {
    let arch = match mapping {
        AdderMapping::RippleCarry => "ripple_carry",
        AdderMapping::KoggeStone => "kogge_stone",
        AdderMapping::BrentKung => "brent_kung",
    };
    let out = ir2gates::ir2gates_from_ir_text(
        &build_ext_nary_add_ir_text(arch),
        None,
        ir2gates::Ir2GatesOptions {
            fold: false,
            hash: false,
            check_equivalence: false,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            adder_mapping: AdderMapping::RippleCarry,
            mul_adder_mapping: None,
            aug_opt: Default::default(),
        },
    )
    .expect("ir2gates");
    get_aig_stats(&out.gatify_output.gate_fn).and_nodes
}

#[test]
fn ext_nary_add_adder_architecture_controls_lowering() {
    let ripple_carry_gates = get_ext_nary_add_gate_count(AdderMapping::RippleCarry);
    let brent_kung_gates = get_ext_nary_add_gate_count(AdderMapping::BrentKung);
    let kogge_stone_gates = get_ext_nary_add_gate_count(AdderMapping::KoggeStone);

    assert!(
        ripple_carry_gates < brent_kung_gates,
        "expected ripple_carry to use fewer live And2 nodes than brent_kung; got ripple_carry={} brent_kung={}",
        ripple_carry_gates,
        brent_kung_gates
    );
    assert!(
        brent_kung_gates < kogge_stone_gates,
        "expected brent_kung to use fewer live And2 nodes than kogge_stone; got brent_kung={} kogge_stone={}",
        brent_kung_gates,
        kogge_stone_gates
    );
}
