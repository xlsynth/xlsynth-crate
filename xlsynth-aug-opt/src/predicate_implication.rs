// SPDX-License-Identifier: Apache-2.0

//! Simplify a negative-and-nonzero predicate when its zero test becomes dead.
//!
//! For `x: s32`, `(x < s32:0) && (x != s32:0)` equals `x < s32:0`.
//! The same identity covers inverted AND/OR spellings. Restricting the witness
//! to the source's sign bit and declining retained comparison cones is an
//! inexpensive structural policy, not a guarantee about heuristic gate mapping.

use std::collections::BTreeSet;

use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Unop};
use xlsynth_pir::ir_match::{MatchCtx, any, exact};
use xlsynth_pir::ir_utils::{UserList, compute_users};

// Bound local graph traversal independently of arithmetic width and shared DAG
// size.
const MAX_CLAUSE_NODES: usize = 256;
const MAX_CLAUSE_TERMS: usize = 32;
const MAX_WRAPPERS: usize = 64;

/// A zero or nonzero predicate over one interval of an unchanged source node.
#[derive(Clone, Copy)]
struct Predicate {
    source: NodeRef,
    start: usize,
    end: usize,
    zero: bool,
}

/// Canonicalizes nested slices without crossing arithmetic or extension nodes.
fn interval(f: &ir::Fn, mut node: NodeRef, zero: bool) -> Option<Predicate> {
    let width = MatchCtx::new(f).bits_width(node)?;
    if width == 0 {
        return None;
    }
    let mut start = 0usize;
    for _ in 0..MAX_WRAPPERS {
        if let NodePayload::BitSlice {
            arg, start: offset, ..
        } = f.get_node(node).payload
        {
            start = start.checked_add(offset)?;
            node = arg;
        } else {
            return Some(Predicate {
                source: node,
                start,
                end: start.checked_add(width)?,
                zero,
            });
        }
    }
    None
}

/// Recognizes either polarity of reduction, zero comparison, or scalar bit.
fn predicate(
    f: &ir::Fn,
    mut node: NodeRef,
    zero_literals: &mut [Option<bool>],
) -> Option<Predicate> {
    let ctx = MatchCtx::new(f);
    if ctx.bits_width(node) != Some(1) {
        return None;
    }
    let mut zero = false;
    for _ in 0..MAX_WRAPPERS {
        match f.get_node(node).payload {
            NodePayload::Unop(Unop::Not, arg) => {
                zero = !zero;
                node = arg;
            }
            NodePayload::Unop(Unop::OrReduce, arg) => return interval(f, arg, zero),
            NodePayload::Binop(op @ (Binop::Eq | Binop::Ne), lhs, rhs) => {
                for literal in [lhs, rhs] {
                    if let NodePayload::Literal(value) = &f.get_node(literal).payload
                        && *zero_literals[literal.index]
                            .get_or_insert_with(|| value.bits_equals_u64_value(0))
                    {
                        let bindings =
                            ctx.commutative_binop_pair(node, op, exact(literal), any("value"))?;
                        return interval(f, bindings.get_node("value")?, zero ^ (op == Binop::Eq));
                    }
                }
                return None;
            }
            _ => return interval(f, node, zero),
        }
    }
    None
}

/// A bounded Boolean clause and the internal nodes bypassed by flattening it.
struct Clause {
    terms: Vec<NodeRef>,
    interior: BTreeSet<usize>,
}

/// Flattens a bounded idempotent clause, visiting shared children only once.
fn clause(f: &ir::Fn, operands: &[NodeRef], plain_op: NaryOp) -> Option<Clause> {
    if operands.len() > MAX_CLAUSE_NODES {
        return None;
    }
    let mut pending: Vec<_> = operands.iter().rev().copied().collect();
    let mut visited = BTreeSet::new();
    let mut terms = Vec::new();
    let mut interior = BTreeSet::new();
    while let Some(node) = pending.pop() {
        if !visited.insert(node.index) {
            continue;
        }
        if visited.len() > MAX_CLAUSE_NODES {
            return None;
        }
        if let NodePayload::Nary(op, children) = &f.get_node(node).payload
            && *op == plain_op
        {
            if pending.len().checked_add(children.len())? > MAX_CLAUSE_NODES {
                return None;
            }
            interior.insert(node.index);
            pending.extend(children.iter().rev().copied());
        } else {
            terms.push(node);
            if terms.len() > MAX_CLAUSE_TERMS {
                return None;
            }
        }
    }
    Some(Clause { terms, interior })
}

/// Adds the discarded predicate and its inversion wrappers to a dead region.
fn discarded_predicate(f: &ir::Fn, mut node: NodeRef, region: &mut BTreeSet<usize>) -> Option<()> {
    for _ in 0..MAX_WRAPPERS {
        region.insert(node.index);
        match f.get_node(node).payload {
            NodePayload::Unop(Unop::Not, arg) => node = arg,
            NodePayload::Unop(Unop::OrReduce, _)
            | NodePayload::Binop(Binop::Eq | Binop::Ne, _, _) => return Some(()),
            _ => return None,
        }
    }
    None
}

/// Removes a full-source predicate implied by the sign bit of that source.
fn replacement(
    f: &ir::Fn,
    root: NodeRef,
    users: &[UserList],
    cache: &mut [Option<Option<Predicate>>],
    zero_literals: &mut [Option<bool>],
) -> Option<NodePayload> {
    let ctx = MatchCtx::new(f);
    if ctx.bits_width(root) != Some(1) {
        return None;
    }
    let NodePayload::Nary(op, operands) = &f.get_node(root).payload else {
        return None;
    };
    let conjunction = match op {
        NaryOp::And | NaryOp::Nand => true,
        NaryOp::Or | NaryOp::Nor => false,
        _ => return None,
    };
    let plain_op = if conjunction { NaryOp::And } else { NaryOp::Or };
    let Clause {
        terms,
        mut interior,
    } = clause(f, operands, plain_op)?;
    let predicates: Vec<_> = terms
        .iter()
        .map(|&node| *cache[node.index].get_or_insert_with(|| predicate(f, node, zero_literals)))
        .collect();
    let mut keep = vec![true; terms.len()];
    for i in 0..terms.len() {
        let Some(a) = predicates[i] else {
            continue;
        };
        for j in 0..terms.len() {
            let Some(b) = predicates[j] else {
                continue;
            };
            if i == j || a.source != b.source || a.zero != b.zero {
                continue;
            }
            let width = ctx.bits_width(a.source)?;
            // The sign being set implies nonzero; equivalently, zero implies
            // the sign is clear. Other ranges and opposite polarities stay put.
            if width > 1
                && conjunction != a.zero
                && a.start == 0
                && a.end == width
                && b.start == width - 1
                && b.end == width
            {
                keep[i] = false;
            }
        }
    }
    if keep.iter().all(|&value| value) {
        return None;
    }
    for (&term, &keep) in terms.iter().zip(&keep) {
        if !keep {
            discarded_predicate(f, term, &mut interior)?;
        }
    }
    // All bypassed clause nodes and discarded predicate wrappers must become
    // dead. In particular, a shared equality feeding another selector declines
    // the rewrite even when its inversion wrapper has only this root as a user.
    if interior.iter().any(|&index| {
        let node = NodeRef { index };
        f.ret_node_ref == Some(node)
            || users[index]
                .iter()
                .any(|user| *user != root && !interior.contains(&user.index))
    }) {
        return None;
    }
    let remaining: Vec<_> = terms
        .into_iter()
        .zip(keep)
        .filter_map(|(node, keep)| keep.then_some(node))
        .collect();
    match remaining.as_slice() {
        [] => None,
        [node] if matches!(op, NaryOp::Nand | NaryOp::Nor) => {
            Some(NodePayload::Unop(Unop::Not, *node))
        }
        [node] => Some(NodePayload::Unop(Unop::Identity, *node)),
        _ => Some(NodePayload::Nary(*op, remaining)),
    }
}

/// Applies sign/nonzero absorption only when the discarded predicate dies.
///
/// Liveness and users are computed once for an immutable graph snapshot.
/// Candidate clause work is bounded, and wide literal checks are cached.
/// Edits preserve indices and equivalent values, keeping range facts valid.
/// No gate graph is built or cost evaluator invoked by this pass. The consumer
/// guard avoids one known native mapping regression family, but the gate
/// optimizer can still trade area against logical effort on unshared inputs.
pub fn rewrite(f: &mut ir::Fn) -> usize {
    let mut live = vec![true; f.nodes.len()];
    for node in xlsynth_pir::dce::get_dead_nodes(f) {
        live[node.index] = false;
    }
    // Filter dead users once: repeatedly skipping a shared predicate's dead
    // users at every root would be quadratic. With live-only lists, a candidate
    // examines at most its bounded discarded region plus the first outside
    // user.
    let users: Vec<UserList> = compute_users(f)
        .iter()
        .map(|(_, users)| {
            users
                .iter()
                .copied()
                .filter(|user| live[user.index])
                .collect()
        })
        .collect();
    let mut cache = vec![None; f.nodes.len()];
    let mut zero_literals = vec![None; f.nodes.len()];
    let edits: Vec<_> = live
        .iter()
        .enumerate()
        .filter_map(|(index, &is_live)| {
            is_live
                .then(|| replacement(f, NodeRef { index }, &users, &mut cache, &mut zero_literals))
                .flatten()
                .map(|payload| (index, payload))
        })
        .collect();
    for (index, payload) in &edits {
        f.nodes[*index].payload = payload.clone();
    }
    edits.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser::Parser;

    fn sample(op: &str, zero: bool, start: usize, width: usize, other: bool) -> ir::Fn {
        let compare = if zero { "eq" } else { "ne" };
        let source = if other { "y" } else { "x" };
        let text = format!(
            r#"package implication
top fn main(x: bits[129] id=1, y: bits[129] id=2) -> bits[1] {{
  a: bits[{width}] = bit_slice(x, start={start}, width={width}, id=3)
  az: bits[{width}] = literal(value=0, id=4)
  bz: bits[129] = literal(value=0, id=5)
  p: bits[1] = {compare}(a, az, id=6)
  q: bits[1] = {compare}(bz, {source}, id=7)
  ret result: bits[1] = {op}(q, p, id=8)
}}
"#
        );
        Parser::new(&text)
            .parse_and_validate_package()
            .unwrap()
            .get_top_fn()
            .unwrap()
            .clone()
    }

    #[test]
    fn sign_polarity_and_operand_order_are_explicit() {
        for op in ["and", "nand", "or", "nor"] {
            for zero in [false, true] {
                for reverse in [false, true] {
                    let mut f = sample(op, zero, 128, 1, false);
                    if reverse {
                        let NodePayload::Nary(_, operands) =
                            &mut f.nodes.last_mut().unwrap().payload
                        else {
                            unreachable!()
                        };
                        operands.reverse();
                    }
                    let count = f.nodes.len();
                    let expected = usize::from(matches!(op, "and" | "nand") != zero);
                    assert_eq!(rewrite(&mut f), expected, "{op}, zero={zero}");
                    assert_eq!(f.nodes.len(), count);
                    Parser::new(&format!("package checked\n\ntop {f}"))
                        .parse_and_validate_package()
                        .unwrap();
                    if expected != 0 {
                        let NodePayload::Unop(_, arg) = f.nodes.last().unwrap().payload else {
                            panic!("expected sign predicate");
                        };
                        assert_eq!(f.get_node(arg).text_id, 6);
                    }
                }
            }
        }
    }

    #[test]
    fn other_sources_ranges_and_empty_witnesses_are_declined() {
        for (start, width, other) in [
            (128, 1, true),
            (127, 1, false),
            (127, 2, false),
            (128, 0, false),
        ] {
            let mut f = sample("and", false, start, width, other);
            let original = f.to_string();
            assert_eq!(rewrite(&mut f), 0);
            assert_eq!(f.to_string(), original);
        }
    }

    #[test]
    fn retained_comparison_and_inversion_wrappers_are_declined() {
        for retained in ["eq", "inv", "clause"] {
            let text = format!(
                r#"package retained
top fn main(x: bits[65] id=1) -> (bits[1], bits[1]) {{
  bit: bits[1] = bit_slice(x, start=64, width=1, id=2)
  z: bits[65] = literal(value=0, id=3)
  eq: bits[1] = eq(x, z, id=4)
  inv: bits[1] = not(eq, id=5)
  clause: bits[1] = and(inv, inv, id=6)
  result: bits[1] = and(clause, bit, id=7)
  ret out: (bits[1], bits[1]) = tuple(result, {retained}, id=8)
}}
"#
            );
            let mut f = Parser::new(&text)
                .parse_and_validate_package()
                .unwrap()
                .get_top_fn()
                .unwrap()
                .clone();
            let original = f.to_string();
            assert_eq!(rewrite(&mut f), 0, "{retained}");
            assert_eq!(f.to_string(), original);
        }
    }

    #[test]
    fn dead_users_before_many_live_roots_do_not_hide_shared_comparison() {
        // User order puts every dead identity before the retained consumers.
        // Liveness must be filtered once, not rescanned at every matching root.
        let count = 1024;
        let mut text = String::from(
            "package many_users\ntop fn main(x: bits[65] id=1) -> RET {\n  bit: bits[1] = bit_slice(x, start=64, width=1, id=2)\n  zero: bits[65] = literal(value=0, id=3)\n  nz: bits[1] = ne(x, zero, id=4)\n",
        );
        for index in 0..count {
            text += &format!("  dead_{index}: bits[1] = identity(nz, id={})\n", 5 + index);
        }
        for index in 0..count {
            text += &format!(
                "  live_{index}: bits[1] = and(bit, nz, id={})\n",
                5 + count + index
            );
        }
        let ty = format!("({})", vec!["bits[1]"; count].join(", "));
        let values = (0..count)
            .map(|index| format!("live_{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        text += &format!(
            "  ret out: {ty} = tuple({values}, id={})\n}}\n",
            5 + count * 2
        );
        let text = text.replace("RET", &ty);
        let mut f = Parser::new(&text)
            .parse_and_validate_package()
            .unwrap()
            .get_top_fn()
            .unwrap()
            .clone();
        let original = f.to_string();
        assert_eq!(rewrite(&mut f), 0);
        assert_eq!(f.to_string(), original);
    }

    #[test]
    fn deep_shared_clause_is_bounded_and_numeric_consumers_are_allowed() {
        for depth in [64, 260] {
            let mut text = String::from(
                "package shared\ntop fn main(x: bits[65] id=1) -> (bits[1], bits[65]) {\n  bit: bits[1] = bit_slice(x, start=64, width=1, id=2)\n  z: bits[65] = literal(value=0, id=3)\n  nz: bits[1] = ne(x, z, id=4)\n",
            );
            let mut previous = "bit".to_string();
            for i in 0..depth {
                let name = format!("shared_{i}");
                text += &format!(
                    "  {name}: bits[1] = and({previous}, {previous}, id={})\n",
                    5 + i
                );
                previous = name;
            }
            text += &format!(
                "  result: bits[1] = and({previous}, nz, id=1000)\n  ret out: (bits[1], bits[65]) = tuple(result, x, id=1001)\n}}\n"
            );
            let mut f = Parser::new(&text)
                .parse_and_validate_package()
                .unwrap()
                .get_top_fn()
                .unwrap()
                .clone();
            assert_eq!(rewrite(&mut f), usize::from(depth == 64));
            Parser::new(&format!("package checked\n\ntop {f}"))
                .parse_and_validate_package()
                .unwrap();
        }
    }
}
