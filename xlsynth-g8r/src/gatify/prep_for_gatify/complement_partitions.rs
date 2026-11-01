// SPDX-License-Identifier: Apache-2.0

//! Removes a reconstructed Boolean complement from an exclusively observed AND.

use std::collections::HashSet;
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::{self, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_utils;

/// In `(guard & s) | (guard & q & factors)`, drop factors which together
/// reconstruct `!s`. A separate observer of the second AND needs its old value.
pub(super) fn rewrite(f: &mut ir::Fn) -> usize {
    let uses = super::get_use_counts(f);
    let mut work = f.nodes.len().saturating_add(uses.iter().sum());
    let mut replacements = Vec::new();
    for node in &f.nodes {
        if work == 0 {
            break;
        }
        let NodePayload::Nary(NaryOp::Or, terms) = &node.payload else {
            continue;
        };
        if node.ty != Type::Bits(1) {
            continue;
        }
        for &term in terms {
            if uses[term.index] != 1 {
                continue;
            }
            let NodePayload::Nary(NaryOp::And, operands) = &f.nodes[term.index].payload else {
                continue;
            };
            if let Some(keep) = retained_operands(f, term, operands, terms, &mut work) {
                replacements.push((term, keep));
            }
        }
    }
    let count = replacements.len();
    for (term, keep) in replacements {
        let payload = match keep.as_slice() {
            [] => NodePayload::Literal(IrValue::make_ubits(1, 1).unwrap()),
            &[value] => NodePayload::Unop(Unop::Identity, value),
            _ => NodePayload::Nary(NaryOp::And, keep),
        };
        ir_utils::replace_node_payload(f, term, payload, None)
            .expect("removing scalar AND operands preserves its type");
    }
    count
}

fn retained_operands(
    f: &ir::Fn,
    term: NodeRef,
    operands: &[NodeRef],
    siblings: &[NodeRef],
    work: &mut usize,
) -> Option<Vec<NodeRef>> {
    if !charge(work, operands.len()) {
        return None;
    }
    let present: HashSet<_> = operands.iter().copied().collect();
    let mut best: Option<Vec<NodeRef>> = None;
    for &sibling in siblings {
        if sibling == term {
            continue;
        }
        let factors = match &f.nodes[sibling.index].payload {
            NodePayload::Nary(NaryOp::And, factors) => factors.as_slice(),
            _ => std::slice::from_ref(&sibling),
        };
        if !charge(work, factors.len()) {
            break;
        }
        for &signal in factors {
            let NodePayload::Nary(NaryOp::Or, inputs) = &f.nodes[signal.index].payload else {
                continue;
            };
            if !charge(work, factors.len().saturating_sub(1)) {
                return best;
            }
            let guard: HashSet<_> = factors.iter().copied().filter(|&n| n != signal).collect();
            if !guard.is_subset(&present) {
                continue;
            }
            let Some(known) = conjuncts(f, signal, true, work) else {
                return best;
            };
            if known.len() < 2 {
                continue;
            }
            let immediate_inputs = inputs.iter().copied().collect();
            let mut covered = HashSet::new();
            let mut keep = Vec::new();
            let mut direct = false;
            for &operand in operands {
                if guard.contains(&operand) {
                    keep.push(operand);
                    continue;
                }
                let Some(required) = conjuncts(f, operand, false, work) else {
                    return best;
                };
                if !required.is_subset(&known) {
                    keep.push(operand);
                    continue;
                }
                // A literal inversion or a direct NOR already names the
                // complement; no separate implementation needs to disappear.
                direct |= explicit_inverse(f, operand, signal, &immediate_inputs);
                covered.extend(required);
            }
            if !direct && covered == known && best.as_ref().is_none_or(|old| keep.len() < old.len())
            {
                best = Some(keep);
            }
        }
    }
    best
}

fn explicit_inverse(
    f: &ir::Fn,
    operand: NodeRef,
    signal: NodeRef,
    inputs: &HashSet<NodeRef>,
) -> bool {
    let other = match &f.nodes[operand.index].payload {
        NodePayload::Unop(Unop::Not, value) if *value == signal => return true,
        NodePayload::Unop(Unop::Not, value) => match &f.nodes[value.index].payload {
            NodePayload::Nary(NaryOp::Or, other) => other,
            _ => return false,
        },
        NodePayload::Nary(NaryOp::Nor, other) => other,
        _ => return false,
    };
    inputs.len() <= other.len() && *inputs == other.iter().copied().collect()
}

/// Conjunctive signed SSA leaves after applying De Morgan's laws. Every other
/// expression is an opaque atom. Iterative traversal handles deeply shared
/// DAGs.
fn conjuncts(
    f: &ir::Fn,
    root: NodeRef,
    inverted: bool,
    work: &mut usize,
) -> Option<HashSet<(NodeRef, bool)>> {
    if !charge(work, 1) {
        return None;
    }
    let mut pending = vec![(root, inverted)];
    let mut seen = HashSet::new();
    let mut leaves = HashSet::new();
    while let Some((node, neg)) = pending.pop() {
        if !seen.insert((node, neg)) {
            continue;
        }
        let (children, negated) = match (&f.nodes[node.index].payload, neg) {
            (NodePayload::Unop(Unop::Not, arg), _) => (std::slice::from_ref(arg), !neg),
            (NodePayload::Nary(NaryOp::And, args), false)
            | (NodePayload::Nary(NaryOp::Nand, args), true) => (args.as_slice(), false),
            (NodePayload::Nary(NaryOp::Or, args), true)
            | (NodePayload::Nary(NaryOp::Nor, args), false) => (args.as_slice(), true),
            _ => {
                leaves.insert((node, neg));
                continue;
            }
        };
        if !charge(work, children.len()) {
            return None;
        }
        pending.extend(children.iter().map(|&arg| (arg, negated)));
    }
    Some(leaves)
}

fn charge(work: &mut usize, amount: usize) -> bool {
    if amount > *work {
        *work = 0;
        return false;
    }
    *work -= amount;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
    use xlsynth_pir::ir_parser::Parser;

    fn check(factors: &str, covering: &str, escaping: bool, expected: usize) -> ir::Fn {
        let rty = if escaping {
            "(bits[1], bits[1])"
        } else {
            "bits[1]"
        };
        let observer = if escaping {
            "o: bits[1] = or(left, right, id=18)\n ret both: (bits[1], bits[1]) = tuple(o, right, id=19)"
        } else {
            "ret o: bits[1] = or(left, right, id=18)"
        };
        let text = format!(
            "package partition\n\ntop fn f(a: bits[1] id=1, b: bits[1] id=2, c: bits[1] id=3, d: bits[1] id=4, g: bits[1] id=5, q: bits[1] id=6) -> {rty} {{\n\
             s: bits[1] = or(a, b, c, d, id=7)\n\
             na: bits[1] = not(a, id=8)\n\
             nrest: bits[1] = nor(b, c, d, id=9)\n\
             ns: bits[1] = not(s, id=10)\n\
             direct_nor: bits[1] = nor(d, c, b, a, id=11)\n\
             ab: bits[1] = or(a, b, id=12)\n\
             cd: bits[1] = or(c, d, id=13)\n\
             tree_nor: bits[1] = nor(ab, cd, id=14)\n\
             left: bits[1] = and({covering}, id=15)\n\
             right: bits[1] = and({factors}, id=16)\n {observer}\n}}"
        );
        let package = Parser::new(&text).parse_and_validate_package().unwrap();
        let original = package.get_top_fn().unwrap();
        let mut edited = original.clone();
        assert_eq!(rewrite(&mut edited), expected);
        assert_eq!(rewrite(&mut edited), 0);
        for assignment in 0u64..64 {
            let values: Vec<_> = (0..6)
                .map(|bit| IrValue::make_ubits(1, (assignment >> bit) & 1).unwrap())
                .collect();
            let run = |f| match eval_fn(f, &values) {
                FnEvalResult::Success(s) => s.value,
                FnEvalResult::Failure(e) => panic!("evaluation failed: {e:?}"),
            };
            assert_eq!(run(original), run(&edited));
        }
        edited
    }

    #[test]
    fn distributed_complement_drops_only_reconstructed_factors() {
        let edited = check("g, na, nrest, q", "g, s", false, 1);
        let right = edited.nodes.iter().find(|n| n.text_id == 16).unwrap();
        let NodePayload::Nary(NaryOp::And, keep) = &right.payload else {
            panic!("expected the remaining AND");
        };
        assert_eq!(
            keep.iter()
                .map(|r| edited.nodes[r.index].text_id)
                .collect::<Vec<_>>(),
            [5, 6]
        );
    }

    #[test]
    fn different_association_can_reconstruct_a_whole_complement() {
        check("g, tree_nor, q", "g, s", false, 1);
    }

    #[test]
    fn direct_complements_and_partial_cubes_keep_their_original_terms() {
        check("g, ns, q", "g, s", false, 0);
        check("g, direct_nor, q", "g, s", false, 0);
        check("g, na, q", "g, s", false, 0);
    }

    #[test]
    fn an_escaping_term_or_uncovered_guard_prevents_absorption() {
        check("g, na, nrest, q", "g, s", true, 0);
        check("g, na, nrest", "g, q, s", false, 0);
    }

    #[test]
    fn a_partition_covering_all_inputs_becomes_true() {
        let edited = check("na, nrest", "s", false, 1);
        let node = edited.nodes.iter().find(|n| n.text_id == 16).unwrap();
        assert!(matches!(node.payload, NodePayload::Literal(_)));
    }

    #[test]
    fn a_wide_disjunction_cannot_exceed_the_remaining_analysis_budget() {
        let edited = check("g, ns, q", "g, s", false, 0);
        let index = edited.nodes.iter().position(|n| n.text_id == 7).unwrap();
        let mut budget = 3;
        assert!(conjuncts(&edited, NodeRef { index }, true, &mut budget).is_none());
        assert_eq!(budget, 0);
    }

    #[test]
    fn complete_complement_with_one_retained_guard_becomes_an_identity() {
        let edited = check("g, na, nrest", "g, s", false, 1);
        let node = edited.nodes.iter().find(|n| n.text_id == 16).unwrap();
        let NodePayload::Unop(Unop::Identity, keep) = node.payload else {
            panic!("expected the remaining guard");
        };
        assert_eq!(edited.nodes[keep.index].text_id, 5);
    }
}
