// SPDX-License-Identifier: Apache-2.0

//! Remove redundant zero/nonzero tests of contained bit ranges.
//!
//! For `x: u32`, `x[7+:u1] & (x != u32:0)` equals `x[7+:u1]`.
//! More generally, a nonzero subrange implies that every containing range is
//! nonzero. The dual zero tests and inverted AND/OR results obey the same
//! absorption law. Whole-function costing retains all other consumers.

use std::collections::BTreeSet;

use crate::ir_cost::IrCost;
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Unop};
use xlsynth_pir::ir_match::{MatchCtx, any, exact};

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

/// Flattens a bounded idempotent clause, visiting shared children only once.
fn clause(f: &ir::Fn, operands: &[NodeRef], plain_op: NaryOp) -> Option<Vec<NodeRef>> {
    if operands.len() > MAX_CLAUSE_NODES {
        return None;
    }
    let mut pending: Vec<_> = operands.iter().rev().copied().collect();
    let mut visited = BTreeSet::new();
    let mut terms = Vec::new();
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
            pending.extend(children.iter().rev().copied());
        } else {
            terms.push(node);
            if terms.len() > MAX_CLAUSE_TERMS {
                return None;
            }
        }
    }
    Some(terms)
}

/// Removes an implied term, keeping a stable representative of equal ranges.
fn replacement(
    f: &ir::Fn,
    root: NodeRef,
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
    let terms = clause(f, operands, plain_op)?;
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
            let equal = a.start == b.start && a.end == b.end;
            if equal {
                if j < i {
                    keep[i] = false;
                }
                continue;
            }
            // AND(nonzero A, nonzero B), with A contained in B, drops B.
            // Negating the predicates or changing AND to OR reverses this.
            let drop_superset = conjunction != a.zero;
            let redundant = if drop_superset {
                a.start <= b.start && b.end <= a.end
            } else {
                b.start <= a.start && a.end <= b.end
            };
            if redundant {
                keep[i] = false;
            }
        }
    }
    if keep.iter().all(|&value| value) {
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

/// Installs only a strict whole-function raw area/delay Pareto improvement.
///
/// Analysis uses one immutable graph and one liveness scan. Local clause work
/// is bounded; no general implication solver or whole-function scan runs per
/// root. Accepted payload edits preserve node indices and equivalent values,
/// so the current round's range facts remain valid. Cost errors and ties leave
/// the input unchanged. Other live consumers of every predicate remain intact.
pub fn rewrite_with_evaluator(
    f: &mut ir::Fn,
    evaluator: &mut impl FnMut(&ir::Fn) -> Result<IrCost, String>,
) -> usize {
    let mut live = vec![true; f.nodes.len()];
    for node in xlsynth_pir::dce::get_dead_nodes(f) {
        live[node.index] = false;
    }
    let mut cache = vec![None; f.nodes.len()];
    // Inspect each literal at most once, including shared arbitrary-width
    // zeros.
    let mut zero_literals = vec![None; f.nodes.len()];
    let edits: Vec<_> = live
        .into_iter()
        .enumerate()
        .filter_map(|(index, live)| {
            live.then(|| replacement(f, NodeRef { index }, &mut cache, &mut zero_literals))
                .flatten()
                .map(|payload| (index, payload))
        })
        .collect();
    if edits.is_empty() {
        return 0;
    }
    let mut rewritten = f.clone();
    for (index, payload) in &edits {
        rewritten.nodes[*index].payload = payload.clone();
    }
    let (Ok(old), Ok(new)) = (evaluator(f), evaluator(&rewritten)) else {
        return 0;
    };
    if !new.is_pareto_improvement_on(old) {
        return 0;
    }
    *f = rewritten;
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
  b: bits[65] = bit_slice({source}, start=32, width=65, id=4)
  az: bits[{width}] = literal(value=0, id=5)
  bz: bits[65] = literal(value=0, id=6)
  p: bits[1] = {compare}(a, az, id=7)
  q: bits[1] = {compare}(bz, b, id=8)
  ret result: bits[1] = {op}(q, p, id=9)
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

    fn force(f: &mut ir::Fn) -> usize {
        let mut calls = 0;
        rewrite_with_evaluator(f, &mut |_| {
            calls += 1;
            Ok(IrCost {
                area: 3 - calls,
                delay: 1.0,
            })
        })
    }

    #[test]
    fn interval_polarity_and_operand_order_are_explicit() {
        for op in ["and", "nand", "or", "nor"] {
            for zero in [false, true] {
                let mut f = sample(op, zero, 40, 9, false);
                let count = f.nodes.len();
                assert_eq!(force(&mut f), 1, "{op}, zero={zero}");
                assert_eq!(f.nodes.len(), count);
                let text = format!("package checked\n\ntop {f}");
                Parser::new(&text).parse_and_validate_package().unwrap();
                let conjunction = matches!(op, "and" | "nand");
                let survivor_id = if conjunction != zero { 7 } else { 8 };
                let NodePayload::Unop(_, arg) = f.nodes.last().unwrap().payload else {
                    panic!("expected one surviving term");
                };
                assert_eq!(f.get_node(arg).text_id, survivor_id);
            }
        }
    }

    #[test]
    fn unrelated_overlapping_empty_and_wrong_polarity_ranges_are_declined() {
        for (start, width, other) in [(40, 9, true), (31, 2, false), (0, 1, false), (40, 0, false)]
        {
            let mut f = sample("and", false, start, width, other);
            let original = f.clone();
            assert_eq!(force(&mut f), 0);
            assert_eq!(f.to_string(), original.to_string());
        }
        let mut f = sample("and", false, 40, 9, false);
        let NodePayload::Binop(_, lhs, rhs) =
            f.nodes.iter().find(|n| n.text_id == 7).unwrap().payload
        else {
            unreachable!()
        };
        f.nodes.iter_mut().find(|n| n.text_id == 7).unwrap().payload =
            NodePayload::Binop(Binop::Eq, lhs, rhs);
        assert_eq!(force(&mut f), 0);
    }

    #[test]
    fn ties_invalid_costs_and_errors_preserve_the_input() {
        for delay in [1.0, f64::NAN, -1.0, f64::INFINITY] {
            let mut f = sample("and", false, 40, 9, false);
            let original = f.clone();
            assert_eq!(
                rewrite_with_evaluator(&mut f, &mut |_| Ok(IrCost { area: 1, delay })),
                0
            );
            assert_eq!(f.to_string(), original.to_string());
        }
        let mut f = sample("and", false, 40, 9, false);
        let original = f.clone();
        assert_eq!(
            rewrite_with_evaluator(&mut f, &mut |_| Err("unavailable".to_string())),
            0
        );
        assert_eq!(f.to_string(), original.to_string());
    }

    #[test]
    fn scalar_parameter_replacement_keeps_parameter_layout_valid() {
        let text = r#"package scalar
top fn main(x: bits[1] id=1) -> bits[1] {
  z: bits[1] = literal(value=0, id=2)
  nz: bits[1] = ne(x, z, id=3)
  ret result: bits[1] = and(x, nz, id=4)
}
"#;
        let mut f = Parser::new(text)
            .parse_and_validate_package()
            .unwrap()
            .get_top_fn()
            .unwrap()
            .clone();
        assert_eq!(force(&mut f), 1);
        Parser::new(&format!("package checked\n\ntop {f}"))
            .parse_and_validate_package()
            .unwrap();
    }
    #[test]
    fn deep_shared_clause_matching_preserves_retained_consumers() {
        let mut text = String::from(
            "package shared\ntop fn main(x: bits[65] id=1) -> (bits[1], bits[1], bits[65]) {\n  bit: bits[1] = bit_slice(x, start=64, width=1, id=2)\n  z: bits[65] = literal(value=0, id=3)\n  nz: bits[1] = ne(x, z, id=4)\n",
        );
        let mut previous = "bit".to_string();
        for depth in 0..64 {
            let name = format!("shared_{depth}");
            text += &format!(
                "  {name}: bits[1] = and({previous}, {previous}, id={})\n",
                5 + depth
            );
            previous = name;
        }
        text += &format!(
            "  result: bits[1] = and({previous}, nz, id=100)\n  ret out: (bits[1], bits[1], bits[65]) = tuple(result, nz, x, id=101)\n}}\n"
        );
        let mut f = Parser::new(&text)
            .parse_and_validate_package()
            .unwrap()
            .get_top_fn()
            .unwrap()
            .clone();
        let original = f.clone();
        assert_eq!(force(&mut f), 1);
        assert_eq!(f.nodes.len(), original.nodes.len());
        for id in [1, 4, 101] {
            assert_eq!(
                f.nodes.iter().find(|n| n.text_id == id).unwrap().payload,
                original
                    .nodes
                    .iter()
                    .find(|n| n.text_id == id)
                    .unwrap()
                    .payload,
            );
        }
        Parser::new(&format!("package checked\n\ntop {f}"))
            .parse_and_validate_package()
            .unwrap();
    }
}
