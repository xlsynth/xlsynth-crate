// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_g8r::test_utils::prove_prepared_ir_equiv;
use xlsynth_pir::ir::{Binop, NodePayload, Type};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::types::EquivResult;

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
