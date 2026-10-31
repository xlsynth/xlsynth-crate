// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_g8r::test_utils::prove_prepared_ir_equiv;
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, Type};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::types::EquivResult;

#[test]
fn unsigned_lexicographic_chains_become_one_carry() {
    for widths in [[1, 2, 3], [7, 4, 11], [65, 3, 129]] {
        for strict in [false, true] {
            for reversed in [false, true] {
                for association in 0..3 {
                    let source = parse(&chain(widths, strict, reversed, association));
                    let prepared =
                        prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
                    let result = prepared.get_node(prepared.ret_node_ref.expect("return node"));
                    let NodePayload::Unop(ir::Unop::Not, carry) = result.payload else {
                        panic!("lexicographic chain was not coalesced: {prepared}");
                    };
                    let NodePayload::ExtCarryOut { lhs, rhs, .. } =
                        prepared.get_node(carry).payload
                    else {
                        panic!("coalesced relation does not use a carry: {prepared}");
                    };
                    for word in [lhs, rhs] {
                        assert_eq!(prepared.get_node(word).ty, Type::Bits(widths.iter().sum()));
                    }
                    assert_equivalent(&source, &prepared);
                }
            }
        }
    }
}

#[test]
fn strict_comparison_or_equality_uses_non_strict_native_comparison() {
    for width in [1, 9, 129] {
        let source = parse(&format!(
            r#"package closing
 top fn f(a: bits[{width}] id=1, b: bits[{width}] id=2) -> bits[1] {{
  gt: bits[1] = ult(b, a, id=3)
  same: bits[1] = eq(b, a, id=4)
  ret result: bits[1] = or(same, gt, id=5)
 }}"#
        ));
        let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
        assert!(matches!(
            prepared.get_node(prepared.ret_node_ref.unwrap()).payload,
            NodePayload::Binop(Binop::Uge, _, _)
        ));
        assert_equivalent(&source, &prepared);
    }
}

#[test]
fn shared_boolean_consumers_retain_the_observed_comparison_region() {
    for observed in ["gt", "same", "tail"] {
        let source = parse(&format!(
            r#"package shared
 top fn f(a: bits[7] id=1, b: bits[7] id=2, c: bits[3] id=3, d: bits[3] id=4) -> (bits[1], bits[1]) {{
  gt: bits[1] = ugt(a, b, id=5)
  same: bits[1] = eq(b, a, id=6)
  low: bits[1] = uge(c, d, id=7)
  tail: bits[1] = and(same, low, id=8)
  result: bits[1] = or(gt, tail, id=9)
  ret pair: (bits[1], bits[1]) = tuple(result, {observed}, id=10)
 }}"#
        ));
        let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
        assert!(
            prepared.nodes.iter().any(|node| node.text_id == 9
                && matches!(node.payload, NodePayload::Nary(NaryOp::Or, _)))
        );
        assert_equivalent(&source, &prepared);
    }
}

#[test]
fn signed_or_incomplete_equality_prefixes_are_not_unsigned_lexicographic_chains() {
    let original = chain([5, 7, 9], false, false, 0);
    for text in [
        original.replace("ugt(a0, b0", "sgt(a0, b0"),
        original.replace("eq(a0, b0", "eq(a0, a0"),
    ] {
        let source = parse(&text);
        let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
        assert!(matches!(
            prepared.get_node(prepared.ret_node_ref.unwrap()).payload,
            NodePayload::Nary(NaryOp::Or, _)
        ));
        assert_equivalent(&source, &prepared);
    }
}

#[test]
fn zero_literal_limbs_preserve_native_equivalence() {
    let text = chain([3, 65, 7], false, true, 2)
        .replace("b1: bits[65] id=4, ", "")
        .replace("  hg:", "  b1: bits[65] = literal(value=0, id=4)\n  hg:");
    let source = parse(&text);
    let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
    assert_equivalent(&source, &prepared);
}

#[test]
fn many_independent_guarded_comparisons_preserve_their_disjunction() {
    for count in [16, 128] {
        let mut params = Vec::new();
        let mut body = Vec::new();
        let mut guards = Vec::new();
        let mut comparisons = Vec::new();
        let mut id = 1;
        for i in 0..count {
            for word in ["a", "b", "c", "d"] {
                params.push(format!("{word}{i}: bits[3] id={id}"));
                id += 1;
            }
        }
        for i in 0..count {
            for (name, op) in [
                (format!("gt{i}"), format!("ugt(a{i}, b{i}")),
                (format!("eq{i}"), format!("eq(a{i}, b{i}")),
                (format!("lo{i}"), format!("uge(c{i}, d{i}")),
                (format!("tail{i}"), format!("and(eq{i}, lo{i}")),
            ] {
                body.push(format!(" {name}: bits[1] = {op}, id={id})"));
                id += 1;
            }
            guards.push(format!("tail{i}"));
            comparisons.push(format!("gt{i}"));
        }
        // All guards precede all strict comparisons, exercising bounded failed
        // scans and repeated merges of a long, unordered operand list.
        guards.extend(comparisons);
        let text = format!(
            "package independent\ntop fn f({}) -> bits[1] {{\n{}\n ret result: bits[1] = or({}, id={id})\n}}\n",
            params.join(", "),
            body.join("\n"),
            guards.join(", ")
        );
        let source = parse(&text);
        let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
        assert!(matches!(
            prepared.get_node(prepared.ret_node_ref.unwrap()).payload,
            NodePayload::Nary(NaryOp::Or, _)
        ));
        assert_equivalent(&source, &prepared);
    }
}

fn chain(widths: [usize; 3], strict: bool, reversed: bool, association: usize) -> String {
    let [h, m, l] = widths;
    let (gt, tail, a, b) = if reversed {
        ("ult", if strict { "ult" } else { "ule" }, "b", "a")
    } else {
        ("ugt", if strict { "ugt" } else { "uge" }, "a", "b")
    };
    let body = match association {
        0 => {
            "  mt: bits[1] = and(me, low, id=12)\n  middle: bits[1] = or(mg, mt, id=13)\n  ht: bits[1] = and(he, middle, id=14)\n  ret result: bits[1] = or(hg, ht, id=15)"
        }
        1 => {
            "  ht: bits[1] = and(mg, he, id=12)\n  lt: bits[1] = and(low, me, he, id=13)\n  ret result: bits[1] = or(lt, ht, hg, id=14)"
        }
        _ => {
            "  prefix: bits[1] = and(me, he, id=12)\n  ht: bits[1] = and(he, mg, id=13)\n  lt: bits[1] = and(prefix, low, id=14)\n  tails: bits[1] = or(lt, ht, id=15)\n  ret result: bits[1] = or(tails, hg, id=16)"
        }
    };
    format!(
        r#"package chain
 top fn f(a0: bits[{h}] id=1, b0: bits[{h}] id=2, a1: bits[{m}] id=3, b1: bits[{m}] id=4, a2: bits[{l}] id=5, b2: bits[{l}] id=6) -> bits[1] {{
  hg: bits[1] = {gt}({a}0, {b}0, id=7)
  he: bits[1] = eq({a}0, {b}0, id=8)
  mg: bits[1] = {gt}({a}1, {b}1, id=9)
  me: bits[1] = eq({b}1, {a}1, id=10)
  low: bits[1] = {tail}({a}2, {b}2, id=11)
{body}
 }}"#
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
