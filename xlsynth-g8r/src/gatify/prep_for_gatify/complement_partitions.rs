// SPDX-License-Identifier: Apache-2.0

//! Removes a reconstructed Boolean complement from an exclusively observed AND.

use super::RewriteIds;
use std::collections::{HashMap, HashSet, hash_map::Entry};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::{self, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_utils;

/// In `(guard & s) | (guard & q & factors)`, drop factors which together
/// reconstruct `!s`. A separate observer of the second AND needs its old value.
pub(super) fn rewrite(f: &mut ir::Fn, origins: Option<&mut HashMap<usize, usize>>) -> usize {
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
                replacements.push((term, keep, node.text_id));
            }
        }
    }
    let count = replacements.len();
    if count == 0 {
        return 0;
    }
    let mut ids = RewriteIds::new(f, origins);
    if !ids.has_trackable_room(count) {
        return 0;
    }
    for (term, keep, or_id) in replacements {
        let payload = match keep.as_slice() {
            [] => NodePayload::Literal(IrValue::make_ubits(1, 1).unwrap()),
            &[value] => NodePayload::Unop(Unop::Identity, value),
            _ => NodePayload::Nary(NaryOp::And, keep),
        };
        ir_utils::replace_node_payload(f, term, payload, None)
            .expect("removing scalar AND operands preserves its type");
        // Retire the term's original value ID. Its gates and those of any
        // subgroups introduced by later rewrites belong to the enclosing OR.
        let node = f.get_node_mut(term);
        super::clear_opcode_name(node);
        node.text_id = ids.fresh(or_id);
    }
    count
}

/// Finds the smallest retained AND whose added assignments a sibling covers.
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
    let mut owners = HashMap::new();
    let mut nested = Vec::new();
    for &operand in operands {
        if owners.insert(operand, operand).is_none()
            && matches!(
                f.get_node(operand).payload,
                NodePayload::Nary(NaryOp::And, _)
            )
        {
            nested.push(operand);
        }
    }
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
            let guards = factors.iter().copied().filter(|&n| n != signal);
            if !nested.is_empty()
                && guards.clone().any(|n| !owners.contains_key(&n))
                && !index_nested_guards(f, &mut nested, &mut owners, work)
            {
                return best;
            }
            let Some(keep_for_guard): Option<HashSet<_>> =
                guards.map(|n| owners.get(&n).copied()).collect()
            else {
                continue;
            };
            let Some(known) = conjuncts(f, signal, true, work) else {
                return best;
            };
            if known.len() < 2 {
                continue;
            }
            let Some(immediate_inputs) = inputs.iter().map(|&n| source_bit(f, n, work)).collect()
            else {
                return best;
            };
            let mut covered = HashSet::new();
            let mut keep = Vec::new();
            let mut direct = false;
            for &operand in operands {
                if keep_for_guard.contains(&operand) {
                    // Other operands pay for this visit at the root of
                    // conjuncts.
                    if !charge(work, 1) {
                        return best;
                    }
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
                if required.len() == known.len() {
                    let Some(inverse) =
                        explicit_inverse(f, operand, signal, &immediate_inputs, work)
                    else {
                        return best;
                    };
                    direct |= inverse;
                }
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

/// Index each positive AND factor under an immediate operand that implies it.
/// Keeping that entire operand preserves the guard, even if the group is
/// shared.
fn index_nested_guards(
    f: &ir::Fn,
    pending: &mut Vec<NodeRef>,
    owners: &mut HashMap<NodeRef, NodeRef>,
    work: &mut usize,
) -> bool {
    while let Some(parent) = pending.pop() {
        let NodePayload::Nary(NaryOp::And, children) = &f.get_node(parent).payload else {
            continue;
        };
        if !charge(work, children.len()) {
            return false;
        }
        let owner = owners[&parent];
        for &child in children {
            if let Entry::Vacant(entry) = owners.entry(child) {
                entry.insert(owner);
                pending.push(child);
            }
        }
    }
    true
}

/// Recognizes a directly implemented complement modulo scalar bit wiring.
fn explicit_inverse(
    f: &ir::Fn,
    operand: NodeRef,
    signal: NodeRef,
    inputs: &HashSet<SourceBit>,
    work: &mut usize,
) -> Option<bool> {
    let operand = source_bit(f, operand, work)?.node;
    if f.get_node(operand).ty != Type::Bits(1) {
        return Some(false);
    }
    let other = match &f.nodes[operand.index].payload {
        NodePayload::Unop(Unop::Not, value) => {
            let value = source_bit(f, *value, work)?.node;
            if value == signal {
                return Some(true);
            }
            match &f.get_node(value).payload {
                NodePayload::Nary(NaryOp::Or, other) if f.get_node(value).ty == Type::Bits(1) => {
                    other
                }
                _ => return Some(false),
            }
        }
        NodePayload::Nary(NaryOp::Nor, other) => other,
        _ => return Some(false),
    };
    if inputs.len() > other.len() {
        return Some(false);
    }
    let other: HashSet<_> = other
        .iter()
        .map(|&n| source_bit(f, n, work))
        .collect::<Option<_>>()?;
    Some(*inputs == other)
}

/// One bit of an SSA value, with bit zero at the least significant end.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct SourceBit {
    node: NodeRef,
    bit: usize,
}

/// Follows the selected bit through static wiring without expanding a whole
/// word.
fn source_bit(f: &ir::Fn, mut node: NodeRef, work: &mut usize) -> Option<SourceBit> {
    let mut bit = 0;
    'trace: loop {
        match &f.get_node(node).payload {
            NodePayload::Unop(Unop::Identity, arg) => node = *arg,
            NodePayload::BitSlice { arg, start, .. } => {
                bit += start;
                node = *arg;
            }
            NodePayload::Nary(NaryOp::Concat, parts) => {
                for &part in parts.iter().rev() {
                    if !charge(work, 1) {
                        return None;
                    }
                    let width = f.get_node_ty(part).bit_count();
                    if bit < width {
                        node = part;
                        continue 'trace;
                    }
                    bit -= width;
                }
                return None;
            }
            _ => return Some(SourceBit { node, bit }),
        }
        if !charge(work, 1) {
            return None;
        }
    }
}

/// Collects signed source bits, expanding scalar conjunctions and their De
/// Morgan equivalents. Other expressions contribute a single opaque bit.
fn conjuncts(
    f: &ir::Fn,
    root: NodeRef,
    inverted: bool,
    work: &mut usize,
) -> Option<HashSet<(SourceBit, bool)>> {
    if !charge(work, 1) {
        return None;
    }
    let mut pending = vec![(root, inverted)];
    let mut seen = HashSet::new();
    let mut leaves = HashSet::new();
    while let Some((node, neg)) = pending.pop() {
        let source = source_bit(f, node, work)?;
        if !seen.insert((source, neg)) {
            continue;
        }
        let node = f.get_node(source.node);
        if node.ty != Type::Bits(1) {
            leaves.insert((source, neg));
            continue;
        }
        let (children, negated) = match (&node.payload, neg) {
            (NodePayload::Unop(Unop::Not, arg), _) => (std::slice::from_ref(arg), !neg),
            (NodePayload::Nary(NaryOp::And, args), false)
            | (NodePayload::Nary(NaryOp::Nand, args), true) => (args.as_slice(), false),
            (NodePayload::Nary(NaryOp::Or, args), true)
            | (NodePayload::Nary(NaryOp::Nor, args), false) => (args.as_slice(), true),
            _ => {
                leaves.insert((source, neg));
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
        check_rewrite(&text, expected)
    }

    /// Checks the match, idempotence and the truth table of a small test graph.
    fn check_rewrite(text: &str, expected: usize) -> ir::Fn {
        let package = Parser::new(text).parse_and_validate_package().unwrap();
        let original = package.get_top_fn().unwrap();
        let mut edited = original.clone();
        assert_eq!(rewrite(&mut edited, None), expected);
        assert_eq!(rewrite(&mut edited, None), 0);
        let widths: Vec<_> = original
            .params
            .iter()
            .map(|&r| original.get_node_ty(r).bit_count())
            .collect();
        for mut assignment in 0u64..1 << widths.iter().sum::<usize>() {
            let values: Vec<_> = widths
                .iter()
                .map(|&width| {
                    let value = assignment & ((1 << width) - 1);
                    assignment >>= width;
                    IrValue::make_ubits(width, value).unwrap()
                })
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
        let right = edited.get_node(ir_utils::find_node_by_name(&edited, "right").unwrap());
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
        let node = edited.get_node(ir_utils::find_node_by_name(&edited, "right").unwrap());
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
    fn repeated_guard_operands_consume_the_search_budget() {
        // Indexing the target and visiting its guard occurrences already
        // exceed the graph-sized budget, even with just one matching sibling.
        let guards = ["g"; 32].join(", ");
        let text = format!(
            r#"package budget

top fn f(g: bits[1] id=1, a: bits[1] id=2, b: bits[1] id=3) -> bits[1] {{
  s: bits[1] = or(a, b, id=4)
  na: bits[1] = not(a, id=5)
  nb: bits[1] = not(b, id=6)
  target: bits[1] = and({guards}, na, nb, id=7)
  sibling: bits[1] = and(g, s, id=8)
  ret result: bits[1] = or(target, sibling, id=9)
}}"#
        );
        check_rewrite(&text, 0);
    }

    #[test]
    fn wired_leaves_compare_by_source_and_bit_offset() {
        // a and b are different bits of the same word. Selecting b twice
        // leaves a uncovered, whereas selecting a through the concat matches.
        for (start, factors, expected) in [(0, "inv, nb", 1), (1, "inv, nb", 0), (0, "direct", 0)] {
            let text = format!(
                r#"package wiring

top fn f(word: bits[2] id=1) -> bits[1] {{
  a: bits[1] = bit_slice(word, start=0, width=1, id=2)
  b: bits[1] = bit_slice(word, start=1, width=1, id=3)
  s: bits[1] = or(a, b, id=4)
  pair: bits[2] = concat(b, a, id=5)
  bit: bits[1] = bit_slice(pair, start={start}, width=1, id=6)
  inv: bits[1] = not(bit, id=7)
  nb: bits[1] = not(b, id=8)
  direct: bits[1] = nor(bit, b, id=11)
  r: bits[1] = and({factors}, id=9)
  ret result: bits[1] = or(s, r, id=10)
}}"#
            );
            check_rewrite(&text, expected);
        }
    }

    #[test]
    fn retain_the_whole_conjunction_that_supplies_the_guard() {
        for (group_op, expected) in [("and", 1), ("or", 0)] {
            let text = format!(
                r#"package guard

top fn f(a: bits[1] id=1, b: bits[1] id=2, c: bits[1] id=3, g: bits[1] id=4, q: bits[1] id=5) -> bits[1] {{
  s: bits[1] = or(a, b, c, id=6)
  left: bits[1] = and(g, s, id=7)
  group: bits[1] = {group_op}(g, q, id=8)
  na: bits[1] = not(a, id=9)
  nrest: bits[1] = nor(b, c, id=10)
  right: bits[1] = and(group, na, nrest, id=11)
  ret result: bits[1] = or(left, right, id=12)
}}"#
            );
            let edited = check_rewrite(&text, expected);
            if expected == 1 {
                let right = edited.get_node(ir_utils::find_node_by_name(&edited, "right").unwrap());
                let NodePayload::Unop(Unop::Identity, group) = right.payload else {
                    panic!("expected the retained group");
                };
                assert_eq!(edited.get_node(group).text_id, 8);
            }
        }
    }

    #[test]
    fn complete_complement_with_one_retained_guard_becomes_an_identity() {
        let edited = check("g, na, nrest", "g, s", false, 1);
        let node = edited.get_node(ir_utils::find_node_by_name(&edited, "right").unwrap());
        let NodePayload::Unop(Unop::Identity, keep) = node.payload else {
            panic!("expected the remaining guard");
        };
        assert_eq!(edited.nodes[keep.index].text_id, 5);
    }
}
