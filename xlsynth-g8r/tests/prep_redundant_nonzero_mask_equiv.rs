// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_g8r::test_utils::prove_prepared_ir_equiv;
use xlsynth_pir::ir::{self, Binop, NodePayload, Unop};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::types::EquivResult;

#[test]
fn zero_preserving_selections_discard_redundant_masks() {
    for width in [3, 9, 65, 129] {
        for predicate in 0..3 {
            for association in 0..4 {
                let source = parse(&selection_case(width, predicate, association));
                let prepared =
                    prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
                assert!(
                    prepared.nodes.iter().all(|node| !matches!(
                        node.payload,
                        NodePayload::Unop(Unop::OrReduce, _) | NodePayload::Binop(Binop::Ne, _, _)
                    )),
                    "redundant predicate survived: width={width}, predicate={predicate}, association={association}"
                );
                assert_equivalent(&source, &prepared);
            }
        }
    }
}

#[test]
fn removing_redundant_masks_reduces_native_and_gates() {
    let source = parse(&selection_case(9, 0, 0));
    let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
    let original = gatify_prepared_fn(&source, GatifyOptions::all_opts_enabled())
        .expect("original gates")
        .gate_fn;
    let improved = gatify_prepared_fn(&prepared, GatifyOptions::all_opts_enabled())
        .expect("prepared gates")
        .gate_fn;
    let before = get_aig_stats(&original);
    let after = get_aig_stats(&improved);
    assert!(
        after.and_nodes < before.and_nodes,
        "{before:?} -> {after:?}"
    );
    assert_equivalent(&source, &prepared);
}

#[test]
fn values_that_need_the_mask_retain_it() {
    for value in [
        "not(x, id=8)",
        "or(x, one, id=8)",
        "add(x, one, id=8)",
        "priority_sel(s, cases=[x, x], default=one, id=8)",
        "one_hot_sel(s, cases=[x, one], id=8)",
        "identity(y, id=8)",
    ] {
        let text = format!(
            r#"package needed
top fn f(x: bits[8] id=1, y: bits[8] id=2, s: bits[2] id=3) -> bits[8] {{
 one: bits[8] = literal(value=1, id=4)
 nonzero: bits[1] = or_reduce(x, id=5)
 mask: bits[8] = sign_ext(nonzero, new_bit_count=8, id=6)
 value: bits[8] = {value}
 ret result: bits[8] = and(mask, value, id=9)
}}
"#
        );
        let source = parse(&text);
        let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
        assert!(
            prepared
                .nodes
                .iter()
                .any(|node| matches!(node.payload, NodePayload::Unop(Unop::OrReduce, _))),
            "necessary predicate removed for {value}"
        );
        assert_equivalent(&source, &prepared);
    }
}

#[test]
fn one_hot_sentinel_decode_and_zero_extensions_are_not_zero_masks() {
    for (source_width, width, value, extension) in [
        (3, 4, "one_hot(x, lsb_prio=true, id=5)", "sign_ext"),
        (3, 8, "decode(x, width=8, id=5)", "sign_ext"),
        (8, 8, "identity(x, id=5)", "zero_ext"),
    ] {
        let text = format!(
            r#"package sentinel
top fn f(x: bits[{source_width}] id=1) -> bits[{width}] {{
 nonzero: bits[1] = or_reduce(x, id=2)
 mask: bits[{width}] = {extension}(nonzero, new_bit_count={width}, id=3)
 value: bits[{width}] = {value}
 ret result: bits[{width}] = and(mask, value, id=6)
}}
"#
        );
        let source = parse(&text);
        let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
        assert!(
            prepared
                .nodes
                .iter()
                .any(|node| matches!(node.payload, NodePayload::Unop(Unop::OrReduce, _)))
        );
        assert_equivalent(&source, &prepared);
    }
}

#[test]
fn shared_masked_values_preserve_every_use() {
    let text = r#"package shared
top fn f(x: bits[9] id=1, y: bits[9] id=2) -> (bits[9], bits[9]) {
 nonzero: bits[1] = or_reduce(x, id=3)
 mask: bits[9] = sign_ext(nonzero, new_bit_count=9, id=4)
 shared: bits[9] = and(mask, y, id=5)
 result: bits[9] = and(shared, x, id=6)
 ret pair: (bits[9], bits[9]) = tuple(result, shared, id=7)
}
"#;
    let source = parse(text);
    let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
    assert_equivalent(&source, &prepared);
}

/// Exercises unconstrained shift/select controls and commuted/reassociated
/// masks.
fn selection_case(width: usize, predicate: usize, association: usize) -> String {
    let narrow = width - 1;
    let predicate = match predicate {
        0 => "or_reduce(x, id=11)",
        1 => "ne(x, zero, id=11)",
        _ => "ne(zero, x, id=11)",
    };
    let output = match association {
        0 => format!("ret result: bits[{width}] = and(value, mask, id=14)"),
        1 => format!("ret result: bits[{width}] = and(mask, value, id=14)"),
        2 => format!("ret result: bits[{width}] = and(y, mask, value, id=14)"),
        _ => format!(
            "inner: bits[{width}] = and(mask, y, id=13)\n ret result: bits[{width}] = and(value, inner, id=14)"
        ),
    };
    format!(
        r#"package zero_preserving
top fn f(x: bits[{width}] id=1, y: bits[{width}] id=2, s: bits[2] id=3, amount: bits[2] id=4) -> bits[{width}] {{
 zero: bits[{width}] = literal(value=0, id=5)
 pad: bits[1] = literal(value=0, id=6)
 slice: bits[{narrow}] = bit_slice(x, start=1, width={narrow}, id=7)
 projected: bits[{width}] = concat(slice, pad, id=8)
 shifted: bits[{width}] = shra(x, amount, id=9)
 choices: bits[{width}] = one_hot_sel(s, cases=[projected, shifted], id=10)
 nonzero: bits[1] = {predicate}
 mask: bits[{width}] = sign_ext(nonzero, new_bit_count={width}, id=12)
 value: bits[{width}] = priority_sel(s, cases=[choices, projected], default=zero, id=15)
 {output}
}}
"#
    )
}

fn parse(text: &str) -> ir::Fn {
    Parser::new(text)
        .parse_and_validate_package()
        .expect("valid synthetic IR")
        .get_top_fn()
        .expect("top function")
        .clone()
}

fn assert_equivalent(source: &ir::Fn, prepared: &ir::Fn) {
    let result = prove_prepared_ir_equiv(source, prepared);
    assert!(matches!(result, EquivResult::Proved), "{result:?}");
}
