// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::gate::AigNode;
use xlsynth_g8r::aig::get_summary_stats::get_aig_cone_stats;
use xlsynth_g8r::aig::table::build_area_table_report;
use xlsynth_g8r::aig::topo::extract_cone;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify, gatify_node_as_fn};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_g8r::ir2gates::{Ir2GatesOptions, Ir2GatesOutput, ir2gates_from_ir_text};
use xlsynth_g8r::test_utils::{
    equivalence_test_solver, prove_prepared_ir_equiv, structurally_equivalent,
};
use xlsynth_pir::IrBits;
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, Type};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::types::EquivResult;

const TWO_LIMB_CHAIN: &str = r#"package comparison
top fn f(a: bits[8], b: bits[8], x: bits[4], y: bits[4]) -> bits[1] {
  high: bits[1] = ugt(a, b, id=5)
  same: bits[1] = eq(a, b, id=6)
  low: bits[1] = ugt(x, y, id=7)
  tail: bits[1] = and(same, low, id=8)
  ret result: bits[1] = or(high, tail, id=9)
}"#;

#[test]
fn carry_gates_belong_to_the_original_comparison() {
    let source = parse(TWO_LIMB_CHAIN);
    let gates = gatify(
        &source,
        GatifyOptions {
            track_pir_node_ids: true,
            fold: false,
            hash: false,
            ..GatifyOptions::all_opts_enabled()
        },
    )
    .unwrap();
    let output = &gates.gate_fn.outputs[0].bit_vector;
    assert_eq!(
        gates.lowering_map[&source.ret_node_ref.unwrap()].get_lsb(0),
        output.get_lsb(0)
    );
    let root = output.get_lsb(0).node;
    assert!(matches!(gates.gate_fn.gates[root.id], AigNode::And2 { .. }));
    let (cone, _) = extract_cone(&[root], &gates.gate_fn.gates);
    for node in cone {
        let gate = &gates.gate_fn.gates[node.id];
        if matches!(gate, AigNode::And2 { .. }) {
            assert_eq!(gate.get_pir_node_ids(), &[9]);
        }
    }
}

#[test]
fn standalone_lexicographic_root_lowers_the_original_or() {
    let source = parse(TWO_LIMB_CHAIN);
    let root = source.ret_node_ref.unwrap();
    let gates = gatify_node_as_fn(&source, root, &GatifyOptions::all_opts_enabled()).unwrap();
    assert_eq!(gates.inputs.len(), 2);
    let output = *gates.outputs[0].bit_vector.get_lsb(0);
    assert_eq!(get_aig_cone_stats(&gates.gates, &[output]).root_depths, [1]);
    let mut sim = PreparedGateSim::new(&gates);
    for a in [false, true] {
        for b in [false, true] {
            assert_eq!(
                sim.eval_outputs(&[IrBits::bool(a), IrBits::bool(b)]),
                [IrBits::bool(a | b)]
            );
        }
    }
    let NodePayload::Nary(NaryOp::Or, operands) = &source.get_node(root).payload else {
        panic!("expected an OR");
    };
    let high = gatify_node_as_fn(&source, operands[0], &GatifyOptions::all_opts_enabled()).unwrap();
    assert_eq!(
        high.inputs
            .iter()
            .map(|i| i.bit_vector.get_bit_count())
            .collect::<Vec<_>>(),
        [8, 8]
    );
}

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
fn flat_four_limb_chain_matches_in_any_term_order() {
    for order in ["g0, t1, t2, t3", "t3, t2, t1, g0", "t2, g0, t3, t1"] {
        let source = parse(&format!(
            r#"package four_limbs
top fn f(a0: bits[4], b0: bits[4], a1: bits[4], b1: bits[4], a2: bits[4], b2: bits[4], a3: bits[4], b3: bits[4]) -> bits[1] {{
  g0: bits[1] = ugt(a0, b0, id=9)
  e0: bits[1] = eq(a0, b0, id=10)
  g1: bits[1] = ugt(a1, b1, id=11)
  e1: bits[1] = eq(a1, b1, id=12)
  g2: bits[1] = ugt(a2, b2, id=13)
  e2: bits[1] = eq(a2, b2, id=14)
  g3: bits[1] = ugt(a3, b3, id=15)
  t1: bits[1] = and(e0, g1, id=16)
  t2: bits[1] = and(e0, e1, g2, id=17)
  t3: bits[1] = and(e0, e1, e2, g3, id=18)
  ret result: bits[1] = or({order}, id=19)
}}"#
        ));
        let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
        let result = prepared.get_node(prepared.ret_node_ref.unwrap());
        let NodePayload::Unop(ir::Unop::Not, carry) = result.payload else {
            panic!("missing carry for OR order {order}");
        };
        let NodePayload::ExtCarryOut { lhs, .. } = prepared.get_node(carry).payload else {
            panic!("expected carry for OR order {order}");
        };
        assert_eq!(prepared.get_node(lhs).ty, Type::Bits(16));
        assert_equivalent(&source, &prepared);
    }
}

#[test]
fn repeated_limb_in_a_flat_chain_can_share_its_equality_prefix() {
    let text = chain([4, 4, 7], true, false, 1)
        .replace("ugt(a1, b1", "ugt(a0, b0")
        .replace("eq(b1, a1", "eq(b0, a0");
    let source = parse(&text);
    let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
    assert!(
        prepared
            .nodes
            .iter()
            .any(|n| matches!(n.payload, NodePayload::ExtCarryOut { .. }))
    );
    assert_equivalent(&source, &prepared);
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
fn equivalent_live_predicates_bound_the_comparison_region() {
    for (low_rhs, extra, observed) in [
        ("y", "ult(b, a, id=10)", true),
        ("high", "ult(b, a, id=10)", true),
        ("y", "ult(b, alias, id=10)", true),
        ("y", "ule(a, b, id=10)", true),
        ("y", "eq(b, a, id=10)", true),
        ("y", "ult(b, a, id=10)", false),
    ] {
        let observed_value = if observed { "extra" } else { "x" };
        let source = parse(&format!(
            r#"package shared
top fn f(a: bits[64], b: bits[64], x: bits[1], y: bits[1]) -> (bits[1], bits[1]) {{
  high: bits[1] = ugt(a, b, id=5)
  same: bits[1] = eq(a, b, id=6)
  low: bits[1] = ugt(x, {low_rhs}, id=7)
  tail: bits[1] = and(same, low, id=8)
  result: bits[1] = or(high, tail, id=9)
  a_id: bits[64] = identity(a, id=12)
  alias: bits[64] = identity(a_id, id=13)
  extra: bits[1] = {extra}
  ret pair: (bits[1], bits[1]) = tuple(result, {observed_value}, id=11)
}}"#
        ));
        let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
        let NodePayload::Tuple(outputs) =
            &prepared.get_node(prepared.ret_node_ref.unwrap()).payload
        else {
            panic!("expected a pair");
        };
        assert_eq!(
            matches!(
                prepared.get_node(outputs[0]).payload,
                NodePayload::Nary(NaryOp::Or, _)
            ),
            observed
        );
        assert_equivalent(&source, &prepared);
    }
}

fn whole_word_comparison(low_op: &str, whole_op: &str, output: &str) -> String {
    format!(
        r#"package whole_word
top fn f(a: bits[32], b: bits[32], x: bits[32], y: bits[32]) -> (bits[1], bits[1]) {{
  high: bits[1] = ugt(a, b, id=5)
  same: bits[1] = eq(a, b, id=6)
  low: bits[1] = {low_op}(x, y, id=7)
  tail: bits[1] = and(same, low, id=8)
  result: bits[1] = or(high, tail, id=9)
  left: bits[64] = concat(a, x, id=10)
  right: bits[64] = concat(b, y, id=11)
  whole: bits[1] = {whole_op}
  {output}
}}"#
    )
}

#[test]
fn comparison_shares_the_returned_whole_word_predicate() {
    for (low_op, whole_op, complemented) in [
        ("ugt", "ugt(left, right, id=12)", false),
        ("ugt", "uge(right, left, id=12)", true),
        ("uge", "ule(right, left, id=12)", false),
    ] {
        let text = whole_word_comparison(
            low_op,
            whole_op,
            "ret pair: (bits[1], bits[1]) = tuple(result, whole, id=13)",
        );
        let source = parse(&text);
        let prepared = prep_for_gatify(&source, None, PrepForGatifyOptions::all_opts_disabled());
        assert_equivalent(&source, &prepared);
        let mapped = map_comparison(&text).unwrap();
        let gates = &mapped.gatify_output.gate_fn;
        let pair = &gates.outputs[0].bit_vector;
        let (a, b) = (pair.get_lsb(0), pair.get_lsb(1));
        assert_eq!(a.node, b.node);
        assert_eq!(a.negated != b.negated, complemented);
        let area = build_area_table_report(gates, mapped.pir_top_fn()).unwrap();
        assert_eq!(gates.gates[a.node.id].get_pir_node_ids(), &[9, 12]);
        assert!(
            area.rows
                .iter()
                .any(|row| row.pir_node_id == 9 && row.weighted_aig_node_count > 0.0)
        );
    }
}

#[test]
fn comparison_retains_native_sharing_with_an_indirect_whole_word_consumer() {
    let text = whole_word_comparison(
        "ugt",
        "ugt(left, right, id=12)",
        r#"mask: bits[1] = bit_slice(a, start=0, width=1, id=13)
  observed: bits[1] = and(mask, whole, id=14)
  ret pair: (bits[1], bits[1]) = tuple(result, observed, id=15)"#,
    );
    let mapped = map_comparison(&text).unwrap();
    // The aligned 32-bit native comparisons hash to the two halves of the
    // 64-bit comparator, even when the whole predicate feeds another gate.
    let gates = &mapped.gatify_output.gate_fn;
    let chain = gates.outputs[0].bit_vector.get_msb(0);
    assert_eq!(gates.gates[chain.node.id].get_pir_node_ids(), &[9, 12]);
}

#[test]
fn comparison_ignores_the_unselected_whole_word_predicate() {
    let [direct, projected] = ["k", "selected"].map(|second_output| {
        let text = format!(r#"package projected
top fn f(p: bits[2], z: bits[8], b: bits[3], x: bits[4], y: bits[4], k: bits[1]) -> (bits[1], bits[1]) {{
  reduce: bits[1] = and_reduce(z, id=7)
  a: bits[3] = concat(p, reduce, id=8)
  high: bits[1] = ugt(a, b, id=9)
  same: bits[1] = eq(a, b, id=10)
  low: bits[1] = ugt(x, y, id=11)
  tail: bits[1] = and(same, low, id=12)
  chain: bits[1] = or(high, tail, id=13)
  left: bits[7] = concat(a, x, id=14)
  right: bits[7] = concat(b, y, id=15)
  whole: bits[1] = ugt(left, right, id=16)
  both: (bits[1], bits[1]) = tuple(k, whole, id=17)
  selected: bits[1] = tuple_index(both, index=0, id=18)
  ret pair: (bits[1], bits[1]) = tuple(chain, {second_output}, id=19)
}}"#);
        map_comparison(&text).unwrap().gatify_output.gate_fn
    });
    assert!(structurally_equivalent(&projected, &direct));
}

/// Runs the public mapper with provenance and an IR-to-AIG equivalence check.
fn map_comparison(text: &str) -> Result<Ir2GatesOutput, String> {
    ir2gates_from_ir_text(
        text,
        None,
        Ir2GatesOptions {
            track_pir_node_ids: true,
            check_equivalence: true,
            equivalence_solver: equivalence_test_solver(),
            enable_formal_array_alias_analysis: cfg!(feature = "has-bitwuzla"),
            ..Ir2GatesOptions::default()
        },
    )
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
fn literal_threshold_limb_only_reaches_its_deciding_bit() {
    for comparison in ["ugt(x, threshold, id=7)", "ult(threshold, x, id=7)"] {
        let source = parse(&format!(
            r#"package threshold
top fn f(a: bits[1], b: bits[1], x: bits[64]) -> bits[1] {{
  threshold: bits[64] = literal(value=0x7fff_ffff_ffff_ffff, id=4)
  high: bits[1] = ugt(a, b, id=5)
  same: bits[1] = eq(a, b, id=6)
  low: bits[1] = {comparison}
  tail: bits[1] = and(same, low, id=8)
  ret result: bits[1] = or(high, tail, id=9)
}}"#
        ));
        let gates = gatify(&source, GatifyOptions::all_opts_enabled())
            .unwrap()
            .gate_fn;
        let root = gates.outputs[0].bit_vector.get_lsb(0);
        let (_, inputs) = extract_cone(&[root.node], &gates.gates);
        let expected = gates
            .inputs
            .iter()
            .map(|i| i.bit_vector.get_msb(0).node)
            .collect();
        assert_eq!(inputs, expected);
    }
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
        // Multiple unguarded comparisons cannot form one ordered word,
        // even though every individual comparison has a matching guard.
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
