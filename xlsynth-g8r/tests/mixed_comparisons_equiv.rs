// SPDX-License-Identifier: Apache-2.0

use test_case::test_case;
use xlsynth_g8r::aig::dce::dce;
use xlsynth_g8r::aig::gate::AigNode;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig::table::build_area_table_report;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify, gatify_node_as_fn};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_g8r::test_utils::prove_prepared_ir_equiv;
use xlsynth_pir::IrBits;
use xlsynth_pir::ir::{Binop, NodePayload, Type};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::types::EquivResult;

#[test_case(Binop::Ult)]
#[test_case(Binop::Sle)]
fn standalone_comparison_uses_both_word_operands(op: Binop) {
    let text = r#"fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  e: bits[1] = eq(a, b, id=3)
  n: bits[1] = ne(a, b, id=4)
  u: bits[1] = ult(a, b, id=5)
  ul: bits[1] = ule(a, b, id=6)
  ug: bits[1] = ugt(a, b, id=7)
  ue: bits[1] = uge(a, b, id=8)
  s: bits[1] = slt(a, b, id=9)
  sl: bits[1] = sle(a, b, id=10)
  ret result: bits[8] = concat(e, n, u, ul, ug, ue, s, sl, id=11)
}"#;
    let f = Parser::new(text).parse_fn().unwrap();
    let target = f
        .node_refs()
        .into_iter()
        .find(|&nr| matches!(f.get_node(nr).payload, NodePayload::Binop(kind, _, _) if kind == op))
        .unwrap();
    let gates = gatify_node_as_fn(&f, target, &GatifyOptions::all_opts_enabled()).unwrap();
    assert_eq!(
        gates
            .inputs
            .iter()
            .map(|i| i.get_bit_count())
            .collect::<Vec<_>>(),
        [8, 8]
    );
    assert!(get_aig_stats(&gates).max_depth > 0);
    let mut sim = PreparedGateSim::new(&gates);
    for (a, b) in [(0u8, 0u8), (1, 2), (2, 1), (127, 128), (128, 127)] {
        let expected = match op {
            Binop::Ult => a < b,
            Binop::Sle => (a as i8) <= (b as i8),
            _ => unreachable!(),
        };
        assert_eq!(
            sim.eval_outputs(&[
                IrBits::make_ubits(8, a.into()).unwrap(),
                IrBits::make_ubits(8, b.into()).unwrap()
            ]),
            [IrBits::bool(expected)]
        );
    }
}

#[test_case("ugt", "sgt", "a, b")]
#[test_case("ult", "sge", "b, a")]
#[test_case("uge", "slt", "a, b")]
fn mixed_order_direction_follows_live_observers(unsigned: &str, signed: &str, operands: &str) {
    let text = format!(
        r#"fn f(a: bits[2] id=1, b: bits[2] id=2) -> bits[3] {{
  u: bits[1] = {unsigned}({operands}, id=3)
  s: bits[1] = {signed}({operands}, id=4)
  e: bits[1] = eq(a, b, id=5)
  unused_u: bits[1] = ult(a, b, id=6)
  unused_s: bits[1] = slt(a, b, id=7)
  ret result: bits[3] = concat(u, s, e, id=8)
}}"#
    );
    let f = Parser::new(&text).parse_fn().unwrap();
    let options = GatifyOptions {
        check_equivalence: true,
        equivalence_solver: xlsynth_g8r::test_utils::equivalence_test_solver(),
        ..GatifyOptions::all_opts_enabled()
    };
    let gates = gatify(&f, options).unwrap().gate_fn;
    let stats = get_aig_stats(&gates);
    assert!(stats.and_nodes <= 10 && stats.max_depth <= 4, "{stats:?}");
}

const SHARED_COMPARISONS: &str = r#"fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[3] {
  e: bits[1] = eq(a, b, id=3)
  u: bits[1] = ult(a, b, id=4)
  s: bits[1] = slt(a, b, id=5)
  ret result: bits[3] = concat(e, u, s, id=6)
}"#;

#[test]
fn comparisons_preserve_mapped_mask_predicates() {
    let text = r#"fn f(x: bits[3] id=1, y: bits[2] id=2) -> bits[3] {
  zero: bits[3] = literal(value=0, id=3)
  p: bits[1] = sle(x, zero, id=4)
  pos: bits[1] = not(p, id=5)
  low: bits[2] = bit_slice(x, start=0, width=2, id=6)
  mask: bits[2] = sign_ext(pos, new_bit_count=2, id=7)
  w: bits[2] = and(low, mask, id=8)
  e: bits[1] = eq(w, y, id=9)
  u: bits[1] = ult(w, y, id=10)
  s: bits[1] = slt(w, y, id=11)
  ret result: bits[3] = concat(e, u, s, id=12)
}"#;
    let f = Parser::new(text).parse_fn().unwrap();
    let mut lowered = gatify(&f, GatifyOptions::all_opts_enabled()).unwrap();
    // At zero the masked word stays zero under either predicate. Inspect the
    // gates advertised for each intermediate value, not just the return.
    let inputs = [
        IrBits::make_ubits(3, 0).unwrap(),
        IrBits::make_ubits(2, 0).unwrap(),
    ];
    for (name, expected) in [("p", 1), ("pos", 0), ("mask", 0)] {
        let nr = f
            .node_refs()
            .into_iter()
            .find(|&nr| f.get_node(nr).name.as_deref() == Some(name))
            .unwrap();
        if let Some(bits) = lowered.lowering_map.get(&nr) {
            lowered.gate_fn.outputs[0].bit_vector = bits.clone();
            let mut sim = PreparedGateSim::new(&lowered.gate_fn);
            assert_eq!(
                sim.eval_outputs(&inputs),
                [IrBits::make_ubits(f.get_node(nr).ty.bit_count(), expected).unwrap()],
                "{name} at x=0"
            );
        }
    }
}

#[test]
fn repeated_mixed_comparisons_share_the_order_merge_without_hashing() {
    let live_ands = |copies: usize| {
        let mut text = format!(
            "fn f(a: bits[2] id=1, b: bits[2] id=2) -> bits[{}] {{\n",
            copies + 3
        );
        let mut names = Vec::new();
        for (i, op) in std::iter::repeat_n("ult", copies)
            .chain(["ugt", "sgt", "eq"])
            .enumerate()
        {
            names.push(format!("r{i}"));
            text.push_str(&format!("  r{i}: bits[1] = {op}(a, b, id={})\n", i + 3));
        }
        text.push_str(&format!(
            "  ret results: bits[{}] = concat({}, id={})\n}}",
            names.len(),
            names.join(", "),
            names.len() + 3
        ));
        let f = Parser::new(&text).parse_fn().unwrap();
        let options = GatifyOptions {
            fold: false,
            hash: false,
            ..GatifyOptions::all_opts_disabled()
        };
        get_aig_stats(&gatify(&f, options).unwrap().gate_fn).and_nodes
    };
    assert_eq!(live_ands(64), live_ands(1));
}

#[test_case(0, false)]
#[test_case(16, false)]
#[test_case(17, true)]
fn mixed_comparisons_and_shifted_mask_at_id_limit(remaining: usize, can_share: bool) {
    let text = r#"fn f(a: bits[8] id=1, b: bits[8] id=2, k: bits[3] id=3) -> bits[11] {
  ones: bits[8] = literal(value=255, id=4)
  bits: bits[8] = shll(ones, k, id=5)
  e: bits[1] = eq(a, b, id=6)
  u: bits[1] = ult(a, b, id=7)
  s: bits[1] = slt(a, b, id=8)
  ret result: bits[11] = concat(e, u, s, bits, id=9)
}"#;
    let mut f = Parser::new(text).parse_fn().unwrap();
    let ret = f.ret_node_ref.unwrap();
    f.get_node_mut(ret).text_id = usize::MAX - remaining;
    let prepared = prep_for_gatify(&f, None, PrepForGatifyOptions::all_opts_enabled());
    assert_eq!(
        prepared
            .nodes
            .iter()
            .any(|n| matches!(n.payload, NodePayload::ExtMaskLow { .. })),
        remaining > 0
    );
    assert_eq!(
        prepared
            .nodes
            .iter()
            .any(|n| matches!(n.payload, NodePayload::Binop(Binop::Slt, _, _))),
        !can_share
    );
    gatify(&f, GatifyOptions::all_opts_enabled()).unwrap();
}

#[test]
fn mixed_comparisons_with_explicit_operator_names() {
    let mut f = Parser::new(SHARED_COMPARISONS).parse_fn().unwrap();
    let eq = f.nodes.iter_mut().find(|n| n.text_id == 3).unwrap();
    eq.name = Some("eq.3".into());
    let prepared = prep_for_gatify(&f, None, PrepForGatifyOptions::all_opts_enabled());
    let ult = prepared.nodes.iter().find(|n| n.text_id == 4).unwrap();
    assert_eq!(ult.name.as_deref(), Some("u"));
    gatify(&f, GatifyOptions::all_opts_enabled()).unwrap();
}

#[test_case(true)]
#[test_case(false)]
fn shared_comparison_area_belongs_to_its_original_observers(fold_and_hash: bool) {
    let f = Parser::new(SHARED_COMPARISONS).parse_fn().unwrap();
    let options = GatifyOptions {
        track_pir_node_ids: true,
        fold: fold_and_hash,
        hash: fold_and_hash,
        ..GatifyOptions::all_opts_enabled()
    };
    let gates = dce(&gatify(&f, options).unwrap().gate_fn);
    let area = build_area_table_report(&gates, &f).unwrap();
    assert_eq!(
        area.rows
            .iter()
            .map(|row| row.pir_node_id)
            .collect::<Vec<_>>(),
        [3, 4, 5]
    );
    assert_eq!(area.unattributed, None);
    // Both order observers use the same low-bit comparison.
    assert!(
        gates
            .gates
            .iter()
            .any(|node| matches!(node, AigNode::And2 { .. }) && node.get_pir_node_ids() == [4, 5])
    );
    for (i, id) in [3, 4, 5].into_iter().enumerate() {
        let root = gates.outputs[0].bit_vector.get_msb(i);
        assert_eq!(gates.gates[root.node.id].get_pir_node_ids(), &[id]);
    }
}

#[test_case(0, false)]
#[test_case(15, false)]
#[test_case(16, true)]
fn mixed_comparisons_reserve_a_full_group_of_text_ids(remaining: usize, can_share: bool) {
    let mut f = Parser::new(SHARED_COMPARISONS).parse_fn().unwrap();
    let ret = f.ret_node_ref.unwrap();
    f.get_node_mut(ret).text_id = usize::MAX - remaining;
    let prepared = prep_for_gatify(&f, None, PrepForGatifyOptions::all_opts_enabled());
    assert_eq!(
        prepared.nodes.iter().any(|n| matches!(n.payload,
            NodePayload::Binop(Binop::Ult, a, _) if prepared.get_node(a).ty == Type::Bits(7)
        )),
        can_share
    );
    gatify(&f, GatifyOptions::all_opts_enabled()).unwrap();
}

#[test]
fn mixed_relations_share_one_low_ordering_with_all_input_values_and_operand_orders() {
    for (width, late) in [(2, false), (3, true), (32, false), (65, true)] {
        for swapped in [false, true] {
            let ops = [
                "eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge",
            ];
            check(width, &ops, &ops, swapped, late, false, true);
        }
    }
}

#[test]
fn unrelated_or_unobserved_comparisons_do_not_trigger_sharing() {
    for used in [
        &["ult"][..],
        &["ult", "eq"],
        &["ugt", "slt"],
        &["slt", "ne"],
    ] {
        check(
            8,
            &["ult", "ugt", "slt", "eq", "ne"],
            used,
            false,
            false,
            false,
            false,
        );
    }
    let all = ["ult", "slt", "eq"];
    check(1, &all, &all, false, false, false, false);
    check(8, &all, &all, false, false, true, false);
}

fn check(
    width: usize,
    defined: &[&str],
    used: &[&str],
    reverse: bool,
    late: bool,
    literal: bool,
    sharing: bool,
) {
    let mut ir = format!(
        "package order\ntop fn f(x: bits[{width}] id=1, y: bits[{width}] id=2, z: bits[{width}] id=3) -> bits[{}] {{\n",
        used.len() + 1
    );
    let a = if late {
        ir.push_str(&format!("  late: bits[{width}] = add(x, z, id=4)\n"));
        "late"
    } else {
        "x"
    };
    let b = if literal {
        ir.push_str(&format!("  k: bits[{width}] = literal(value=1, id=5)\n"));
        "k"
    } else {
        "y"
    };
    for (i, op) in defined.iter().enumerate() {
        let (a, b) = if reverse ^ (i % 2 != 0) {
            (b, a)
        } else {
            (a, b)
        };
        ir.push_str(&format!(
            "  r_{op}: bits[1] = {op}({a}, {b}, id={})\n",
            i + 10
        ));
    }
    let mut results = used.iter().map(|op| format!("r_{op}")).collect::<Vec<_>>();
    // Repeated output references exercise comparison fanout.
    results.push(results[0].clone());
    ir.push_str(&format!(
        "  ret result: bits[{}] = concat({}, id=100)\n}}",
        results.len(),
        results.join(", ")
    ));
    let pkg = Parser::new(&ir).parse_and_validate_package().unwrap();
    let f = pkg.get_top_fn().unwrap();
    let prepared = prep_for_gatify(f, None, PrepForGatifyOptions::all_opts_disabled());
    let low_orderings = prepared.nodes.iter().filter(|n| matches!(n.payload,
        NodePayload::Binop(Binop::Ult, a, _) if width >= 2 && prepared.get_node(a).ty == Type::Bits(width-1))).count();
    assert_eq!(low_orderings, usize::from(sharing), "{prepared}");
    if sharing {
        assert!(!prepared.nodes.iter().any(|n| matches!(
            n.payload,
            NodePayload::Binop(Binop::Slt | Binop::Sle | Binop::Sgt | Binop::Sge, _, _)
        )));
    }
    let result = prove_prepared_ir_equiv(f, &prepared);
    assert!(matches!(result, EquivResult::Proved), "{result:?}");
}
