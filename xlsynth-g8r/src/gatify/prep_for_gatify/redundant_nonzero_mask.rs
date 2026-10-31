// SPDX-License-Identifier: Apache-2.0

//! Removes nonzero masks whose data cone already vanishes with their source.

use std::collections::HashMap;
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Unop};
use xlsynth_pir::ir_match::{self, MatchCtx};
use xlsynth_pir::ir_utils;

/// Removes `value & sign_ext(or_reduce(source))` when `source == 0`
/// implies `value == 0`, without duplicating any shared AND interior.
pub(super) fn rewrite(f: &mut ir::Fn) -> usize {
    let uses = super::get_use_counts(f);
    let mask_sources: Vec<_> = (0..f.nodes.len())
        .map(|index| nonzero_mask_source(f, NodeRef { index }))
        .collect();
    let mut and_parents = vec![Vec::new(); f.nodes.len()];
    for (index, node) in f.nodes.iter().enumerate() {
        if let NodePayload::Nary(NaryOp::And, operands) = &node.payload {
            for operand in operands {
                and_parents[operand.index].push(NodeRef { index });
            }
        }
    }
    let mut candidates = vec![false; f.nodes.len()];
    let mut pending: Vec<_> = mask_sources
        .iter()
        .enumerate()
        .filter(|(_, source)| source.is_some())
        .map(|(index, _)| NodeRef { index })
        .collect();
    while let Some(node) = pending.pop() {
        for &parent in &and_parents[node.index] {
            if !candidates[parent.index] {
                candidates[parent.index] = true;
                if uses[parent.index] == 1 {
                    pending.push(parent);
                }
            }
        }
    }
    let graph_work = f.nodes.len().saturating_add(
        f.nodes
            .iter()
            .map(|node| ir_utils::operands(&node.payload).len())
            .sum::<usize>(),
    );
    let mut analysis = ZeroImplication {
        f,
        memo: HashMap::new(),
        remaining_work: graph_work,
    };
    let mut replacements = Vec::new();
    for index in 0..f.nodes.len() {
        if analysis.remaining_work == 0 {
            break;
        }
        let root = NodeRef { index };
        // A maximal unshared AND tree contains every mask/value combination of
        // its interiors. Analyze it once instead of repeatedly flattening
        // prefixes.
        if !candidates[index] || (uses[index] == 1 && !and_parents[index].is_empty()) {
            continue;
        }
        let Some(mut operands) = analysis.flatten_and(root, &uses) else {
            break;
        };
        let original_count = operands.len();
        let mut i = 0;
        while i < operands.len() && analysis.remaining_work > 0 {
            let Some(source) = mask_sources[operands[i].index] else {
                i += 1;
                continue;
            };
            let mut redundant = false;
            for (j, &value) in operands.iter().enumerate() {
                if analysis.remaining_work == 0 {
                    break;
                }
                if i != j && analysis.proves_zero(value, source) {
                    redundant = true;
                    break;
                }
            }
            if redundant {
                operands.remove(i);
            } else {
                i += 1;
            }
        }
        if operands.len() < original_count {
            let payload = match operands.as_slice() {
                [value] => NodePayload::Unop(Unop::Identity, *value),
                _ => NodePayload::Nary(NaryOp::And, operands),
            };
            replacements.push((root, payload));
        }
    }
    // All queries share one immutable snapshot; substitutions preserve the
    // facts.
    let count = replacements.len();
    for (root, payload) in replacements {
        ir_utils::replace_node_payload(f, root, payload, None)
            .expect("redundant nonzero mask replacement preserves its type");
    }
    count
}

/// Recognizes a width-independent all-zero/all-one nonzero mask.
fn nonzero_mask_source(f: &ir::Fn, mask: NodeRef) -> Option<NodeRef> {
    let ctx = MatchCtx::new(f);
    let predicate = match f.get_node(mask).payload {
        NodePayload::SignExt { arg, .. } if ctx.bits_width(arg) == Some(1) => arg,
        _ if ctx.bits_width(mask) == Some(1) => mask,
        _ => return None,
    };
    match f.get_node(predicate).payload {
        NodePayload::Unop(Unop::OrReduce, source) => Some(source),
        NodePayload::Binop(Binop::Ne, lhs, rhs) => {
            let zero = [lhs, rhs].into_iter().find(|&n| {
                matches!(&f.get_node(n).payload, NodePayload::Literal(v)
                    if v.bits_equals_u64_value(0))
            })?;
            ctx.commutative_binop_pair(
                predicate,
                Binop::Ne,
                ir_match::any("source"),
                ir_match::exact(zero),
            )?
            .get_node("source")
        }
        _ => None,
    }
}

/// Conservative conditional facts, memoized across masks sharing a source.
///
/// The shared work budget is the graph's node-plus-edge count. Exhaustion only
/// declines further rewrites, bounding total analysis and matching work
/// linearly. Explicit traversal stacks avoid depending on the IR's
/// combinational depth.
struct ZeroImplication<'a> {
    f: &'a ir::Fn,
    memo: HashMap<(NodeRef, NodeRef), bool>,
    remaining_work: usize,
}

/// A pending conjunction/disjunction of sufficient zero conditions.
struct Frame {
    node: NodeRef,
    operands: Vec<NodeRef>,
    next: usize,
    any: bool,
}

impl ZeroImplication<'_> {
    /// Flattens only unshared AND trees, with matching charged to the same
    /// budget.
    fn flatten_and(&mut self, root: NodeRef, uses: &[usize]) -> Option<Vec<NodeRef>> {
        let mut pending = vec![root];
        let mut result = Vec::new();
        while let Some(node) = pending.pop() {
            self.remaining_work = self.remaining_work.checked_sub(1)?;
            match &self.f.get_node(node).payload {
                NodePayload::Nary(NaryOp::And, args) if node == root || uses[node.index] == 1 => {
                    pending.extend(args.iter().rev().copied());
                }
                _ => result.push(node),
            }
        }
        Some(result)
    }

    /// Proves a sufficient implication; false means unknown, not a
    /// counterexample.
    fn proves_zero(&mut self, value: NodeRef, source: NodeRef) -> bool {
        // Charge even cache hits, so many masks in one n-ary AND cannot trigger
        // an unbounded quadratic number of cheap implication queries.
        let Some(remaining) = self.remaining_work.checked_sub(1) else {
            return false;
        };
        self.remaining_work = remaining;
        let mut current = value;
        let mut pending: Vec<Frame> = Vec::new();
        let mut result = None;
        loop {
            if result.is_none() {
                result = if current == source {
                    Some(true)
                } else {
                    self.memo.get(&(source, current)).copied()
                };
            }
            if result.is_none() {
                if self.remaining_work == 0 {
                    return false;
                }
                self.remaining_work -= 1;
                let payload = &self.f.get_node(current).payload;
                let (operands, any) = match payload {
                    NodePayload::Literal(v) => {
                        result = Some(v.bits_equals_u64_value(0));
                        (Vec::new(), false)
                    }
                    NodePayload::BitSlice { arg, .. }
                    | NodePayload::DynamicBitSlice { arg, .. }
                    | NodePayload::SignExt { arg, .. }
                    | NodePayload::ZeroExt { arg, .. }
                    | NodePayload::Unop(
                        Unop::Identity
                        | Unop::Reverse
                        | Unop::Neg
                        | Unop::OrReduce
                        | Unop::XorReduce,
                        arg,
                    ) => (vec![*arg], false),
                    NodePayload::Binop(Binop::Shll | Binop::Shrl | Binop::Shra, data, _) => {
                        (vec![*data], false)
                    }
                    NodePayload::Binop(Binop::Add | Binop::Sub, lhs, rhs) => {
                        (vec![*lhs, *rhs], false)
                    }
                    NodePayload::Binop(Binop::Umul | Binop::Smul, lhs, rhs) => {
                        (vec![*lhs, *rhs], true)
                    }
                    NodePayload::Nary(NaryOp::And, args) => (args.clone(), true),
                    NodePayload::Nary(NaryOp::Or | NaryOp::Xor | NaryOp::Concat, args) => {
                        (args.clone(), false)
                    }
                    NodePayload::Sel { cases, default, .. }
                    | NodePayload::PrioritySel { cases, default, .. } => {
                        let mut args = cases.clone();
                        args.extend(default);
                        (args, false)
                    }
                    NodePayload::OneHotSel { cases, .. } => (cases.clone(), false),
                    _ => {
                        // Unknown operations, including one_hot/decode, may
                        // emit ones at zero.
                        result = Some(false);
                        (Vec::new(), false)
                    }
                };
                if result.is_none() {
                    let Some(remaining) = self.remaining_work.checked_sub(operands.len()) else {
                        self.remaining_work = 0;
                        return false;
                    };
                    self.remaining_work = remaining;
                    if let Some(&first) = operands.first() {
                        pending.push(Frame {
                            node: current,
                            operands,
                            next: 1,
                            any,
                        });
                        current = first;
                        continue;
                    }
                    result = Some(!any);
                }
            }
            let proven = result.expect("node has a sufficient zero condition");
            self.memo.insert((source, current), proven);
            let Some(frame) = pending.last_mut() else {
                return proven;
            };
            if proven == frame.any || frame.next == frame.operands.len() {
                current = frame.node;
                pending.pop();
            } else {
                current = frame.operands[frame.next];
                frame.next += 1;
                result = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser::Parser;

    #[test]
    fn exhausted_work_budget_stops_wide_nodes_and_cached_queries() {
        let operands = vec!["y"; 1024].join(", ");
        let text = format!(
            r#"package bounded
top fn f(x: bits[1] id=1, y: bits[1] id=2) -> bits[1] {{
 zero: bits[1] = literal(value=0, id=4)
 ret value: bits[1] = or({operands}, id=3)
}}"#
        );
        let package = Parser::new(&text)
            .parse_and_validate_package()
            .expect("valid wide operation");
        let f = package.get_top_fn().expect("top function");
        let source = NodeRef {
            index: f
                .nodes
                .iter()
                .position(|node| node.name.as_deref() == Some("x"))
                .expect("source parameter"),
        };
        let zero = NodeRef {
            index: f
                .nodes
                .iter()
                .position(|node| node.name.as_deref() == Some("zero"))
                .expect("zero literal"),
        };
        let mut analysis = ZeroImplication {
            f,
            memo: HashMap::new(),
            remaining_work: 8,
        };
        // An oversized operation must exhaust the shared budget, so repeated
        // queries cannot repeatedly copy its operands without paying for them.
        assert!(!analysis.proves_zero(f.ret_node_ref.unwrap(), source));
        assert_eq!(analysis.remaining_work, 0);
        assert!(analysis.memo.is_empty());
        for _ in 0..1024 {
            assert!(!analysis.proves_zero(f.ret_node_ref.unwrap(), source));
        }
        assert!(analysis.memo.is_empty());

        analysis.remaining_work = 8;
        assert!(analysis.proves_zero(zero, source));
        assert!(analysis.memo.contains_key(&(source, zero)));
        analysis.remaining_work = 8;
        for remaining in (0..8).rev() {
            assert!(analysis.proves_zero(zero, source));
            assert_eq!(analysis.remaining_work, remaining);
        }
        assert!(!analysis.proves_zero(zero, source));
    }
}
