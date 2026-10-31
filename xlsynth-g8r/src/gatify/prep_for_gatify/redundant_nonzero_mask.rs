// SPDX-License-Identifier: Apache-2.0

//! Removes nonzero masks whose data cone already vanishes with their source.

use super::RewriteIds;
use std::collections::HashMap;
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Unop};
use xlsynth_pir::ir_match::{self, MatchCtx};
use xlsynth_pir::ir_utils;

/// Removes `value & sign_ext(or_reduce(source))` when `source == 0`
/// implies `value == 0`, without duplicating any shared AND interior.
pub(super) fn rewrite(f: &mut ir::Fn, origins: Option<&mut HashMap<usize, usize>>) -> usize {
    let uses = super::get_use_counts(f);
    let mut zero_literals = HashMap::new();
    let mask_sources = nonzero_mask_sources(f, &mut zero_literals);
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
        zero_literals,
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
        let Some(mut tree) = analysis.flatten_and(root, &uses) else {
            break;
        };
        if analysis.mark_redundant_masks(&mut tree, &mask_sources) {
            replacements.push(tree);
        }
    }
    // All queries share one immutable snapshot. Only maximal tree roots
    // retain their original meaning when rebuilt from pruned children.
    let mut ids = RewriteIds::new(f, origins);
    let mut count = 0;
    for tree in replacements {
        if rebuild_pruned_tree(f, tree, &mut ids) {
            count += 1;
        }
    }
    count
}

/// A preorder entry in a maximal unshared AND tree. Each operand occurrence
/// has its own leaf, even when multiple edges refer to the same node.
enum AndTreeEntry {
    Group(NodeRef),
    Leaf(Option<NodeRef>),
}

/// Orders the surviving operands of one PIR AND for `GateBuilder::tree_reduce`.
/// Empty halves disappear and commutative children can swap places. Two
/// nonempty halves whose sizes differ by more than one would need another PIR
/// AND, hiding direct operands from matchers on the original group.
fn collect_pruned_operands(
    span: std::ops::Range<usize>,
    kept: &[(usize, NodeRef)],
    out: &mut Vec<NodeRef>,
) -> bool {
    if kept.len() <= 1 || kept.len() == span.len() {
        out.extend(kept.iter().map(|&(_, n)| n));
        return true;
    }
    let mid = span.start + span.len() / 2;
    let (left, right) = kept.split_at(kept.partition_point(|&(i, _)| i < mid));
    if !left.is_empty() && !right.is_empty() && left.len().abs_diff(right.len()) > 1 {
        return false;
    }
    let mut halves = [(span.start..mid, left), (mid..span.end, right)];
    if left.len() > right.len() {
        halves.swap(0, 1);
    }
    halves
        .into_iter()
        .all(|(span, half)| collect_pruned_operands(span, half, out))
}

/// Rebuilds changed PIR groups bottom-up. Only the maximal root is known
/// to retain its original value when a mask is zero.
fn rebuild_pruned_tree(f: &mut ir::Fn, tree: Vec<AndTreeEntry>, ids: &mut RewriteIds) -> bool {
    let mut helpers = Vec::new();
    let mut values = Vec::new();
    for (index, entry) in tree.into_iter().enumerate().rev() {
        let group = match entry {
            AndTreeEntry::Leaf(operand) => {
                values.push(operand);
                continue;
            }
            AndTreeEntry::Group(group) => group,
        };
        let NodePayload::Nary(NaryOp::And, args) = &f.get_node(group).payload else {
            unreachable!("recorded an AND group");
        };
        // Reverse preorder leaves the first operand at the top of the stack.
        let arity = args.len();
        let children = values.drain(values.len() - arity..).rev();
        let kept: Vec<_> = children
            .enumerate()
            .filter_map(|(i, n)| n.map(|operand| (i, operand)))
            .collect();
        if kept.iter().map(|(_, n)| n).eq(args) {
            values.push(Some(group));
            continue;
        }
        if kept.is_empty() {
            // A root always has a surviving witness. An interior can disappear.
            values.push(None);
            continue;
        }
        let payload = if let [(_, one)] = kept.as_slice() {
            if index != 0 {
                values.push(Some(*one));
                continue;
            }
            NodePayload::Unop(Unop::Identity, *one)
        } else {
            let mut operands = Vec::with_capacity(kept.len());
            if !collect_pruned_operands(0..arity, &kept, &mut operands) {
                return false;
            }
            NodePayload::Nary(NaryOp::And, operands)
        };
        if index != 0 {
            values.push(Some(NodeRef {
                index: f.nodes.len() + helpers.len(),
            }));
            helpers.push((group, payload));
            continue;
        }
        if !ids.has_room(helpers.len()) {
            return false;
        }
        for (source, payload) in helpers {
            let source = f.get_node(source);
            let node = ir::Node {
                text_id: ids.fresh(source.text_id),
                name: None,
                ty: source.ty.clone(),
                payload,
                pos: None,
            };
            f.nodes.push(node);
        }
        let root = f.get_node_mut(group);
        if matches!(payload, NodePayload::Unop(Unop::Identity, _)) {
            super::clear_opcode_name(root);
        }
        root.payload = payload;
        return true;
    }
    unreachable!("a pruned root changes and retains a surviving witness")
}

/// Scans each queried literal's limbs once on this graph snapshot.
fn is_zero_literal(f: &ir::Fn, node: NodeRef, cache: &mut HashMap<NodeRef, bool>) -> bool {
    let NodePayload::Literal(value) = &f.get_node(node).payload else {
        return false;
    };
    *cache
        .entry(node)
        .or_insert_with(|| value.bits_equals_u64_value(0))
}

/// Recognizes predicates once, then propagates their sources to sign
/// extensions. Two passes allow producers to follow their consumers in node
/// storage.
fn nonzero_mask_sources(
    f: &ir::Fn,
    zero_literals: &mut HashMap<NodeRef, bool>,
) -> Vec<Option<NodeRef>> {
    let ctx = MatchCtx::new(f);
    let predicates: Vec<_> = f
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| match node.payload {
            NodePayload::Unop(Unop::OrReduce, source) => Some(source),
            NodePayload::Binop(Binop::Ne, lhs, rhs) => {
                let zero = [lhs, rhs]
                    .into_iter()
                    .find(|&n| is_zero_literal(f, n, zero_literals))?;
                ctx.commutative_binop_pair(
                    NodeRef { index },
                    Binop::Ne,
                    ir_match::any("source"),
                    ir_match::exact(zero),
                )?
                .get_node("source")
            }
            _ => None,
        })
        .collect();
    f.nodes
        .iter()
        .enumerate()
        .map(|(index, node)| match node.payload {
            NodePayload::SignExt { arg, .. } => predicates[arg.index],
            _ => predicates[index],
        })
        .collect()
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
    zero_literals: HashMap<NodeRef, bool>,
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
    fn flatten_and(&mut self, root: NodeRef, uses: &[usize]) -> Option<Vec<AndTreeEntry>> {
        let mut pending = vec![root];
        let mut result = Vec::new();
        while let Some(node) = pending.pop() {
            self.remaining_work = self.remaining_work.checked_sub(1)?;
            match &self.f.get_node(node).payload {
                NodePayload::Nary(NaryOp::And, args) if node == root || uses[node.index] == 1 => {
                    result.push(AndTreeEntry::Group(node));
                    pending.extend(args.iter().rev().copied());
                }
                _ => result.push(AndTreeEntry::Leaf(Some(node))),
            }
        }
        Some(result)
    }

    /// Searches previously kept leaves followed by the untouched suffix.
    /// Dropping a leaf takes constant work and it cannot witness later
    /// removals.
    fn mark_redundant_masks(
        &mut self,
        tree: &mut [AndTreeEntry],
        mask_sources: &[Option<NodeRef>],
    ) -> bool {
        let leaves: Vec<_> = tree
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| match entry {
                AndTreeEntry::Leaf(Some(value)) => Some((index, *value)),
                _ => None,
            })
            .collect();
        let mut kept = Vec::new();
        let mut changed = false;
        for (i, &(index, value)) in leaves.iter().enumerate() {
            if self.remaining_work == 0 {
                break;
            }
            if let Some(source) = mask_sources[value.index] {
                let suffix = leaves[i + 1..].iter().map(|&(_, operand)| operand);
                let mut redundant = false;
                for witness in kept.iter().copied().chain(suffix) {
                    if self.remaining_work == 0 {
                        break;
                    }
                    if self.proves_zero(witness, source) {
                        redundant = true;
                        break;
                    }
                }
                if redundant {
                    tree[index] = AndTreeEntry::Leaf(None);
                    changed = true;
                    continue;
                }
            }
            kept.push(value);
        }
        changed
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
                    NodePayload::Literal(_) => {
                        result = Some(is_zero_literal(self.f, current, &mut self.zero_literals));
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
    use xlsynth_pir::{FnBuilder, IrValue};

    #[test]
    fn prunes_repeated_mask_edges_along_a_deep_and_chain() {
        let mut builder = FnBuilder::new("mask_chain");
        let x = builder.param("x", ir::Type::Bits(8)).unwrap();
        let nonzero = builder.or_reduce(x).unwrap();
        let mask = builder.sign_extend(nonzero, 8).unwrap();
        let mut value = x;
        for _ in 0..16_384 {
            value = builder.and(value, mask).unwrap();
        }
        let mut f = builder.build(value).unwrap();
        assert_eq!(rewrite(&mut f, None), 1);
        assert_eq!(
            f.get_node(f.ret_node_ref.unwrap()).payload,
            NodePayload::Unop(Unop::Identity, f.params[0])
        );
    }

    #[test]
    fn shared_wide_predicate_is_discovered_for_every_mask() {
        let mut builder = FnBuilder::new("shared_predicate");
        let width = 1 << 20;
        let x = builder.param("x", ir::Type::Bits(width)).unwrap();
        let zero = builder
            .literal(IrValue::make_ubits(width, 0).unwrap())
            .unwrap();
        let p = builder.ne(x, zero).unwrap();
        let masks = (0..8192)
            .map(|_| builder.sign_extend(p, 2).unwrap())
            .collect::<Vec<_>>();
        let tuple = builder.tuple(&masks).unwrap();
        let f = builder.build(tuple).unwrap();
        let mut zero_literals = HashMap::new();
        let sources = nonzero_mask_sources(&f, &mut zero_literals);
        assert_eq!(zero_literals.len(), 1);
        let NodePayload::Tuple(masks) = &f.get_node(f.ret_node_ref.unwrap()).payload else {
            panic!("expected tuple of masks");
        };
        for mask in masks {
            assert_eq!(sources[mask.index], Some(f.params[0]));
        }
    }

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
            zero_literals: HashMap::new(),
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
