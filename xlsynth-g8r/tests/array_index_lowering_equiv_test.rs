// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::AigNode;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{ArrayIndexLoweringStrategy, GatifyOptions, gatify};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::prove_gate_fn_equiv_sat::{EquivResult, VarisatCtx, prove_gate_fn_equiv_varisat};
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

fn build_literal_array_index_ir_text(
    array_len: u32,
    element_width: u32,
    index_width: u32,
    index_value: u32,
    assumed_in_bounds: bool,
) -> String {
    format!(
        r#"package sample

top fn main(arr: bits[{element_width}][{array_len}]) -> bits[{element_width}] {{
  idx: bits[{index_width}] = literal(value={index_value}, id=2)
  ret r: bits[{element_width}] = array_index(arr, indices=[idx], assumed_in_bounds={assumed_in_bounds}, id=3)
}}
"#
    )
}

fn build_multidimensional_static_suffix_array_index_ir_text() -> String {
    r#"package sample

top fn main(arr: bits[4][8][2], outer: bits[1]) -> bits[4] {
  inner: bits[3] = literal(value=5, id=3)
  ret r: bits[4] = array_index(arr, indices=[outer, inner], assumed_in_bounds=true, id=4)
}
"#
    .to_string()
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
            adder_mapping: AdderMapping::RippleCarry,
            array_index_lowering_strategy: strategy,
            ..GatifyOptions::all_opts_disabled()
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
    check_equivalence::validate_same_fn_via_toolchain(&ir_fn, &gate_fn)
        .expect("strategy should preserve IR semantics");
}

fn assert_strategies_are_equivalent(
    ir_text: &str,
    lhs: ArrayIndexLoweringStrategy,
    rhs: ArrayIndexLoweringStrategy,
) {
    let lhs_gate = gatify_ir_text_with_strategy(ir_text, lhs);
    let rhs_gate = gatify_ir_text_with_strategy(ir_text, rhs);
    let mut ctx = VarisatCtx::new();
    assert_eq!(
        prove_gate_fn_equiv_varisat(&lhs_gate, &rhs_gate, &mut ctx),
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

#[test]
fn test_literal_array_index_uses_direct_slice_and_preserves_oob_semantics() {
    for (index_width, index_value, assumed_in_bounds) in
        [(3, 2, true), (96, 2, true), (3, 7, false)]
    {
        let ir_text = build_literal_array_index_ir_text(
            /* array_len= */ 4,
            /* element_width= */ 5,
            index_width,
            index_value,
            assumed_in_bounds,
        );
        assert_strategy_matches_ir(&ir_text, ArrayIndexLoweringStrategy::Auto);
        let gate_fn = gatify_ir_text_with_strategy(&ir_text, ArrayIndexLoweringStrategy::Auto);
        assert!(
            gate_fn
                .gates
                .iter()
                .all(|node| !matches!(node, AigNode::And2 { .. })),
            "literal array index should be a direct bit-vector slice"
        );
    }
}

#[test]
fn test_multidimensional_literal_suffix_is_sliced_after_dynamic_outer_index() {
    let ir_text = build_multidimensional_static_suffix_array_index_ir_text();
    assert_strategy_matches_ir(&ir_text, ArrayIndexLoweringStrategy::Auto);
    let gate_fn = gatify_ir_text_with_strategy(&ir_text, ArrayIndexLoweringStrategy::Auto);
    let and_count = gate_fn
        .gates
        .iter()
        .filter(|node| matches!(node, AigNode::And2 { .. }))
        .count();
    assert_eq!(
        and_count, 96,
        "the full outer dimension should be selected before the literal slice"
    );
}
