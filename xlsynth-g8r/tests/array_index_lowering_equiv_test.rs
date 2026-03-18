// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{ArrayIndexLoweringStrategy, GatifyOptions, gatify};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::prove_gate_fn_equiv_varisat::{Ctx, EquivResult, prove_gate_fn_equiv};
use xlsynth_pir::ir_parser;

fn build_direct_array_index_ir_text(
    array_len: u32,
    element_width: u32,
    index_width: u32,
) -> String {
    format!(
        r#"package sample

top fn main(arr: bits[{element_width}][{array_len}], idx: bits[{index_width}]) -> bits[{element_width}] {{
  ret r: bits[{element_width}] = array_index(arr, indices=[idx], id=3)
}}
"#
    )
}

fn build_tuple_field0_array_index_ir_text(
    array_len: u32,
    payload_width: u32,
    index_width: u32,
) -> String {
    format!(
        r#"package sample

top fn main(arr: (bits[1], bits[{payload_width}])[{array_len}], idx: bits[{index_width}]) -> bits[1] {{
  ai: (bits[1], bits[{payload_width}]) = array_index(arr, indices=[idx], id=3)
  ret r: bits[1] = tuple_index(ai, index=0, id=4)
}}
"#
    )
}

fn gatify_ir_text_with_strategy(
    ir_text: &str,
    strategy: ArrayIndexLoweringStrategy,
) -> xlsynth_g8r::aig::GateFn {
    let mut parser = ir_parser::Parser::new(ir_text);
    let ir_package = parser.parse_and_validate_package().expect("parse package");
    let ir_fn = ir_package.get_top_fn().expect("top fn");
    let gatify_output = gatify(
        &ir_fn,
        GatifyOptions {
            fold: true,
            check_equivalence: false,
            hash: true,
            adder_mapping: AdderMapping::RippleCarry,
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            array_index_lowering_strategy: strategy,
        },
    )
    .expect("gatify");
    gatify_output.gate_fn
}

fn assert_strategy_matches_ir(ir_text: &str, strategy: ArrayIndexLoweringStrategy) {
    let mut parser = ir_parser::Parser::new(ir_text);
    let ir_package = parser.parse_and_validate_package().expect("parse package");
    let ir_fn = ir_package.get_top_fn().expect("top fn");
    let gate_fn = gatify_ir_text_with_strategy(ir_text, strategy);
    check_equivalence::validate_same_fn(&ir_fn, &gate_fn)
        .expect("strategy should preserve IR semantics");
}

fn assert_strategies_are_equivalent(
    ir_text: &str,
    lhs: ArrayIndexLoweringStrategy,
    rhs: ArrayIndexLoweringStrategy,
) {
    let lhs_gate = gatify_ir_text_with_strategy(ir_text, lhs);
    let rhs_gate = gatify_ir_text_with_strategy(ir_text, rhs);
    let mut ctx = Ctx::new();
    assert_eq!(
        prove_gate_fn_equiv(&lhs_gate, &rhs_gate, &mut ctx),
        EquivResult::Proved,
        "lowerings should be equivalent for IR:\n{}",
        ir_text
    );
}

#[test]
fn test_near_pow2_mux_tree_matches_ir_and_oob_one_hot_for_direct_array_index() {
    let _ = env_logger::builder().is_test(true).try_init();

    for element_width in [1u32, 2] {
        let ir_text = build_direct_array_index_ir_text(
            /* array_len= */ 27,
            element_width,
            /* index_width= */ 5,
        );
        assert_strategy_matches_ir(&ir_text, ArrayIndexLoweringStrategy::ForceNearPow2MuxTree);
        assert_strategies_are_equivalent(
            &ir_text,
            ArrayIndexLoweringStrategy::ForceNearPow2MuxTree,
            ArrayIndexLoweringStrategy::ForceOobOneHot,
        );
    }
}

#[test]
fn test_near_pow2_mux_tree_matches_ir_and_oob_one_hot_for_tuple_field_array_index() {
    let _ = env_logger::builder().is_test(true).try_init();

    let ir_text = build_tuple_field0_array_index_ir_text(
        /* array_len= */ 27, /* payload_width= */ 1, /* index_width= */ 5,
    );
    assert_strategy_matches_ir(&ir_text, ArrayIndexLoweringStrategy::ForceNearPow2MuxTree);
    assert_strategies_are_equivalent(
        &ir_text,
        ArrayIndexLoweringStrategy::ForceNearPow2MuxTree,
        ArrayIndexLoweringStrategy::ForceOobOneHot,
    );
}
