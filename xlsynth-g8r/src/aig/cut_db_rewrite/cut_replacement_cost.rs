// SPDX-License-Identifier: Apache-2.0

//! Exact live-gate costing for cut replacement candidates.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use smallvec::SmallVec;

use crate::aig::dynamic_structural_hash::DynamicStructuralHash;
use crate::aig::gate::{AigNode, AigOperand, AigRef};
use crate::cut_db::fragment::{FragmentNode, GateFnFragment, Lit};

use super::large_cone::{FactoredExprNode, SopReplacement};
use super::{
    Replacement, ReplacementImpl, apply_replacement_to_dynamic_hash,
    cleanup_dangling_new_dynamic_hash_nodes,
};

mod effective_pricing;
pub(super) use effective_pricing::{AreaCost, EffectiveCutPricing, price_area_candidate};

/// Summary of a replacement that was materialized into the dynamic AIG.
#[derive(Debug, Clone)]
pub(super) struct MaterializedReplacement {
    /// First gate id that did not exist before materializing the replacement.
    pub(super) first_new_id: usize,
    /// Root operand produced by the replacement fragment after local strashing.
    pub(super) replacement_op: AigOperand,
    /// Exact local nodes whose depth state may have changed while rewiring.
    pub(super) depth_seed_nodes: Vec<AigRef>,
}

/// Exact live-AND delta for applying one replacement to a dynamic AIG state.
#[derive(Debug, Clone)]
pub(super) struct ReplacementGateCountDiff {
    /// Live AND count before the replacement.
    pub(super) before_live_ands: usize,
    /// Live AND count after the replacement.
    pub(super) after_live_ands: usize,
    /// `before_live_ands - after_live_ands`; positive means area decreased.
    pub(super) live_and_delta: isize,
    /// For positive gain, whether all requested live gates survive actual
    /// materialization. None means the caller needs to materialize a trial to
    /// determine this.
    pub(super) protected_live: Option<bool>,
}

/// Returns the exact live-AND count delta for replacing `replacement.root`.
///
/// The calculation intentionally uses the same dynamic structural hash
/// operation as the committed rewrite path, including cascading strash merges
/// and recursive deletion of newly dangling MFFCs.
pub(super) fn gate_count_diff_for_replacement(
    state: &DynamicStructuralHash,
    replacement: &Replacement,
) -> Result<Option<ReplacementGateCountDiff>, String> {
    gate_count_diff_with_protected(state, replacement, &[])
}

/// Prices a replacement and, where possible, certifies whether protected gates
/// survive. The caller supplies sorted physical gate IDs known to be live.
pub(super) fn gate_count_diff_with_protected(
    state: &DynamicStructuralHash,
    replacement: &Replacement,
    protected: &[AigRef],
) -> Result<Option<ReplacementGateCountDiff>, String> {
    debug_assert!(protected.is_sorted());
    debug_assert!(protected.iter().all(|&node| state.is_live(node)));
    if !state.is_live(replacement.root) {
        return Ok(None);
    }
    let tracking_root = (!protected.is_empty()).then_some(replacement.root);
    let mut overlay = VirtualCostOverlay::new(state, tracking_root);
    let replacement_op = overlay.instantiate_replacement(replacement)?;
    overlay
        .finish_cost(replacement, replacement_op, protected)
        .map(Some)
}

/// Applies a replacement to the real dynamic hash state.
pub(super) fn materialize_replacement(
    state: &mut DynamicStructuralHash,
    replacement: &Replacement,
) -> Result<Option<MaterializedReplacement>, String> {
    let first_new_id = state.gate_fn().gates.len();
    let Some((replacement_op, depth_seed_nodes)) =
        apply_replacement_to_dynamic_hash(state, replacement)?
    else {
        return Ok(None);
    };
    cleanup_dangling_new_dynamic_hash_nodes(state, first_new_id)?;
    Ok(Some(MaterializedReplacement {
        first_new_id,
        replacement_op,
        depth_seed_nodes,
    }))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct AndKey {
    lhs: AigOperand,
    rhs: AigOperand,
}

impl AndKey {
    fn new(lhs: AigOperand, rhs: AigOperand) -> Self {
        if lhs <= rhs {
            Self { lhs, rhs }
        } else {
            Self { lhs: rhs, rhs: lhs }
        }
    }
}

/// An AND gate that a proposed cut replacement would create.
///
/// To estimate the area saved, the rewriter simulates reconnecting the cut's
/// users and graph outputs. If this gate loses its last use, the simulation
/// follows its two inputs to find any further gates that would become unused.
struct VirtualAnd {
    children: (AigOperand, AigOperand),
    /// Total uses by AND input edges and graph output bits, in either polarity.
    uses: usize,
    /// The subset of `uses` from graph output bits. The simulation redirects
    /// these separately from AND inputs. A gate feeding one AND input and
    /// two output bits has three uses, two of them output uses.
    output_uses: usize,
}

struct VirtualCostOverlay<'a> {
    // Cleared when a physical rewrite could delete different gate IDs.
    tracking_root: Option<AigRef>,
    state: &'a DynamicStructuralHash,
    base_len: usize,
    use_count_deltas: BTreeMap<AigRef, isize>,
    output_count_deltas: BTreeMap<AigRef, isize>,
    // Virtual IDs are appended consecutively starting at base_len.
    virtual_ands: SmallVec<[VirtualAnd; 8]>,
    virtual_by_key: BTreeMap<AndKey, AigOperand>,
    rewritten_children: BTreeMap<AigRef, (AigOperand, AigOperand)>,
    rewritten_by_key: BTreeMap<AndKey, AigOperand>,
    output_replacements: BTreeMap<AigRef, AigOperand>,
    deleted: BTreeSet<AigRef>,
}

impl<'a> VirtualCostOverlay<'a> {
    fn new(state: &'a DynamicStructuralHash, tracking_root: Option<AigRef>) -> Self {
        let base_len = state.gate_fn().gates.len();
        Self {
            tracking_root,
            state,
            base_len,
            use_count_deltas: BTreeMap::new(),
            output_count_deltas: BTreeMap::new(),
            virtual_ands: SmallVec::new(),
            virtual_by_key: BTreeMap::new(),
            rewritten_children: BTreeMap::new(),
            rewritten_by_key: BTreeMap::new(),
            output_replacements: BTreeMap::new(),
            deleted: BTreeSet::new(),
        }
    }

    fn finish_cost(
        &mut self,
        replacement: &Replacement,
        replacement_op: AigOperand,
        protected: &[AigRef],
    ) -> Result<ReplacementGateCountDiff, String> {
        self.replace_node_with_operand(replacement.root, replacement_op)?;
        self.cleanup_dangling_virtual_nodes()?;

        let before_live_ands = self.state.live_and_count();
        let after_live_ands = self.after_live_and_count();
        // In the merge-free case a valid cut and its virtual implementation
        // remove the same original gate IDs as a committed replacement.
        let protected_live = (after_live_ands < before_live_ands
            && self.tracking_root.is_some()
            && replacement
                .leaf_ops
                .iter()
                .all(|op| op.node != replacement.root && self.state.is_live(op.node)))
        .then(|| {
            !self
                .deleted
                .iter()
                .any(|node| protected.binary_search(node).is_ok())
        });
        Ok(ReplacementGateCountDiff {
            protected_live,
            before_live_ands,
            after_live_ands,
            live_and_delta: before_live_ands as isize - after_live_ands as isize,
        })
    }

    fn instantiate_replacement(&mut self, replacement: &Replacement) -> Result<AigOperand, String> {
        match &replacement.implementation {
            ReplacementImpl::Fragment { frag, .. } => {
                self.instantiate_fragment(frag, &replacement.leaf_ops)
            }
            ReplacementImpl::Sop(sop) => self.instantiate_sop(sop, &replacement.leaf_ops),
        }
    }

    fn instantiate_fragment(
        &mut self,
        frag: &GateFnFragment,
        leaf_ops: &[AigOperand],
    ) -> Result<AigOperand, String> {
        let mut ops = Vec::with_capacity(5 + frag.nodes.len());
        for i in 0..4usize {
            if i < leaf_ops.len() {
                ops.push(leaf_ops[i]);
            } else {
                ops.push(Self::false_op());
            }
        }
        ops.push(Self::false_op());

        for node in &frag.nodes {
            let op = match *node {
                FragmentNode::And2 { a, b } => {
                    let a_op = Self::op_from_lit(a, &ops);
                    let b_op = Self::op_from_lit(b, &ops);
                    self.add_and(a_op, b_op)?
                }
            };
            ops.push(op);
        }

        Ok(Self::op_from_lit(frag.output, &ops))
    }

    fn instantiate_sop(
        &mut self,
        sop: &SopReplacement,
        leaf_ops: &[AigOperand],
    ) -> Result<AigOperand, String> {
        Ok(self
            .instantiate_sop_until(sop, leaf_ops, |_| false)?
            .expect("unrestricted SOP construction always finishes"))
    }

    fn instantiate_sop_until(
        &mut self,
        sop: &SopReplacement,
        leaf_ops: &[AigOperand],
        stop: impl Fn(usize) -> bool,
    ) -> Result<Option<AigOperand>, String> {
        let expr = &sop.factored;
        let mut ops = Vec::with_capacity(expr.nodes().len());
        for node in expr.nodes().iter().copied() {
            let op = match node {
                FactoredExprNode::Const(false) => Self::false_op(),
                FactoredExprNode::Const(true) => Self::true_op(),
                FactoredExprNode::Lit { var, negated } => {
                    let mut op = leaf_ops[var];
                    if negated {
                        op = op.negate();
                    }
                    op
                }
                FactoredExprNode::And { lhs, rhs } => self.add_and(ops[lhs.0], ops[rhs.0])?,
                FactoredExprNode::Or { lhs, rhs } => self.add_or_binary(ops[lhs.0], ops[rhs.0])?,
            };
            if stop(self.virtual_ands.len()) {
                return Ok(None);
            }
            ops.push(op);
        }
        let output = ops[expr.root().0];
        Ok(Some(if sop.output_negated {
            output.negate()
        } else {
            output
        }))
    }

    fn replace_node_with_operand(
        &mut self,
        old: AigRef,
        replacement: AigOperand,
    ) -> Result<(), String> {
        let mut queue = VecDeque::from([(old, replacement)]);
        while let Some((old, replacement)) = queue.pop_front() {
            if !self.is_live_node(old) {
                continue;
            }
            if replacement.node == old {
                if replacement.negated {
                    return Err(format!("cannot replace {:?} with its own negation", old));
                }
                continue;
            }
            self.validate_operand(replacement)?;
            if !self.is_and_node(old) {
                return Err(format!("only And2 nodes can be replaced; got {:?}", old));
            }

            self.remove_rewritten_index_node(old);
            self.rewrite_outputs(old, replacement);

            let direct_fanouts = self.state.fanout_nodes(old);
            for fanout in direct_fanouts {
                if !self.is_live_node(fanout) {
                    continue;
                }
                let (old_a, old_b) = self.current_and_operands(fanout)?;
                let moved_edges = usize::from(old_a.node == old) + usize::from(old_b.node == old);
                if moved_edges == 0 {
                    continue;
                }
                self.remove_rewritten_index_node(fanout);
                self.decrement_use_by(old, moved_edges)?;
                self.increment_use_by(replacement.node, moved_edges)?;
                let new_a = Self::replace_operand_node(old_a, old, replacement);
                let new_b = Self::replace_operand_node(old_b, old, replacement);

                let key = AndKey::new(new_a, new_b);
                if let Some(existing) = self.lookup_existing_and(key, Some(fanout)) {
                    self.rewritten_children.insert(fanout, (new_a, new_b));
                    // Physical rewrites consult updated fanouts and recheck
                    // queued representatives.
                    self.tracking_root = None;
                    queue.push_back((fanout, existing));
                } else {
                    self.rewritten_children.insert(fanout, (new_a, new_b));
                    self.rewritten_by_key.insert(key, fanout.into());
                }
            }

            self.delete_dangling_mffc_from(old)?;
        }
        Ok(())
    }

    fn cleanup_dangling_virtual_nodes(&mut self) -> Result<(), String> {
        for id in (self.base_len..self.base_len + self.virtual_ands.len()).rev() {
            let node = AigRef { id };
            if self.is_live_node(node) && self.use_count(node)? == 0 {
                self.delete_dangling_mffc_from(node)?;
            }
        }
        Ok(())
    }

    fn after_live_and_count(&self) -> usize {
        // Count the ANDs left after the trial without scanning the graph.
        // Each AND the trial creates gets one slot in virtual_ands. Cleanup
        // records each unused AND in deleted by gate ID, whether it came from
        // the original graph or the trial. A gate created and then discarded
        // therefore contributes zero.
        self.state.live_and_count() + self.virtual_ands.len() - self.deleted.len()
    }

    fn add_and(&mut self, lhs: AigOperand, rhs: AigOperand) -> Result<AigOperand, String> {
        self.validate_operand(lhs)?;
        self.validate_operand(rhs)?;

        let key = AndKey::new(lhs, rhs);
        // Construction precedes all edits and deletions in this overlay.
        // The physical graph cannot contain a gate with a virtual operand.
        let existing = self.virtual_by_key.get(&key).copied().or_else(|| {
            if lhs.node.id < self.base_len && rhs.node.id < self.base_len {
                self.state.lookup_and(lhs, rhs)
            } else {
                None
            }
        });
        if let Some(existing) = existing {
            // Physical construction excludes the original root from its
            // lookups.
            if self.tracking_root == Some(existing.node) {
                self.tracking_root = None;
            }
            return Ok(existing);
        }

        let node = AigRef {
            id: self.base_len + self.virtual_ands.len(),
        };
        self.virtual_ands.push(VirtualAnd {
            children: (lhs, rhs),
            uses: 0,
            output_uses: 0,
        });
        let op = AigOperand {
            node,
            negated: false,
        };
        self.virtual_by_key.insert(key, op);
        self.increment_use(lhs.node)?;
        self.increment_use(rhs.node)?;
        Ok(op)
    }

    fn add_or_binary(&mut self, lhs: AigOperand, rhs: AigOperand) -> Result<AigOperand, String> {
        if Self::is_true(lhs) || Self::is_true(rhs) {
            return Ok(Self::true_op());
        }
        if Self::is_false(lhs) && Self::is_false(rhs) {
            return Ok(Self::false_op());
        }
        if Self::is_false(lhs) {
            return Ok(rhs);
        }
        if Self::is_false(rhs) {
            return Ok(lhs);
        }
        Ok(self.add_and(lhs.negate(), rhs.negate())?.negate())
    }

    fn lookup_existing_and(&self, key: AndKey, exclude: Option<AigRef>) -> Option<AigOperand> {
        if let Some(op) = self.rewritten_by_key.get(&key).copied() {
            if Some(op.node) != exclude && self.is_live_node(op.node) {
                return Some(op);
            }
        }
        if let Some(op) = self.virtual_by_key.get(&key).copied() {
            if Some(op.node) != exclude && self.is_live_node(op.node) {
                return Some(op);
            }
        }

        self.state
            .lookup_and_excluding_predicate(key.lhs, key.rhs, |node| {
                Some(node) == exclude
                    || self.deleted.contains(&node)
                    || self.rewritten_children.contains_key(&node)
            })
            .filter(|op| self.is_live_node(op.node))
    }

    fn delete_dangling_mffc_from(&mut self, root: AigRef) -> Result<(), String> {
        if !self.is_and_node(root) {
            return Err(format!("only And2 nodes can be deleted; got {:?}", root));
        }
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if !self.is_live_node(node) || !self.is_and_node(node) {
                continue;
            }
            let use_count = self.use_count(node)?;
            if use_count != 0 {
                if node == root {
                    return Err(format!("cannot delete {:?}; use_count={}", node, use_count));
                }
                continue;
            }

            self.remove_virtual_index_node(node);
            self.remove_rewritten_index_node(node);
            self.deleted.insert(node);
            let (a, b) = self.current_and_operands(node)?;
            self.decrement_use(a.node)?;
            self.decrement_use(b.node)?;
            for child in [a.node, b.node] {
                if self.is_live_node(child)
                    && self.is_and_node(child)
                    && self.use_count(child)? == 0
                {
                    stack.push(child);
                }
            }
        }
        Ok(())
    }

    fn current_and_operands(&self, node: AigRef) -> Result<(AigOperand, AigOperand), String> {
        if let Some(children) = self.rewritten_children.get(&node).copied() {
            return Ok(children);
        }
        if let Some(and) = self.virtual_and(node) {
            return Ok(and.children);
        }
        match self.state.gate_fn().gates.get(node.id) {
            Some(AigNode::And2 { a, b, .. }) => Ok((*a, *b)),
            Some(other) => Err(format!("expected And2 at {:?}; got {:?}", node, other)),
            None => Err(format!("node {:?} out of bounds", node)),
        }
    }

    fn remove_virtual_index_node(&mut self, node: AigRef) {
        if let Some(and) = self.virtual_and(node) {
            let key = AndKey::new(and.children.0, and.children.1);
            if self
                .virtual_by_key
                .get(&key)
                .is_some_and(|op| op.node == node)
            {
                self.virtual_by_key.remove(&key);
            }
        }
    }

    fn remove_rewritten_index_node(&mut self, node: AigRef) {
        if let Some((a, b)) = self.rewritten_children.get(&node).copied() {
            let key = AndKey::new(a, b);
            if self
                .rewritten_by_key
                .get(&key)
                .is_some_and(|op| op.node == node)
            {
                self.rewritten_by_key.remove(&key);
            }
        }
    }

    fn rewrite_outputs(&mut self, old: AigRef, replacement: AigOperand) {
        let old_count = self
            .output_count(old)
            .expect("output count node should exist");
        if old_count == 0 {
            return;
        }
        self.output_replacements.insert(old, replacement);
        self.decrement_output_count_by(old, old_count)
            .expect("old output count should be positive");
        self.decrement_use_by(old, old_count)
            .expect("old output use count should be positive");
        self.increment_output_count_by(replacement.node, old_count)
            .expect("replacement output count node should exist");
        self.increment_use_by(replacement.node, old_count)
            .expect("replacement use count node should exist");
    }

    fn increment_use(&mut self, node: AigRef) -> Result<(), String> {
        self.increment_use_by(node, 1)
    }

    fn decrement_use(&mut self, node: AigRef) -> Result<(), String> {
        self.decrement_use_by(node, 1)
    }

    fn increment_use_by(&mut self, node: AigRef, amount: usize) -> Result<(), String> {
        self.adjust_use_count(node, amount as isize)
    }

    fn decrement_use_by(&mut self, node: AigRef, amount: usize) -> Result<(), String> {
        self.adjust_use_count(node, -(amount as isize))
    }

    fn increment_output_count_by(&mut self, node: AigRef, amount: usize) -> Result<(), String> {
        self.adjust_output_count(node, amount as isize)
    }

    fn decrement_output_count_by(&mut self, node: AigRef, amount: usize) -> Result<(), String> {
        self.adjust_output_count(node, -(amount as isize))
    }

    fn adjust_use_count(&mut self, node: AigRef, delta: isize) -> Result<(), String> {
        self.validate_node_index(node)?;
        if node.id < self.base_len {
            Self::adjust_real_count(
                &mut self.use_count_deltas,
                node,
                self.state.use_count(node),
                delta,
                "use count",
            )
        } else {
            let index = node.id - self.base_len;
            Self::adjust_virtual_count(
                &mut self.virtual_ands[index].uses,
                index,
                delta,
                "use count",
            )
        }
    }

    fn adjust_output_count(&mut self, node: AigRef, delta: isize) -> Result<(), String> {
        self.validate_node_index(node)?;
        if node.id < self.base_len {
            Self::adjust_real_count(
                &mut self.output_count_deltas,
                node,
                self.state.output_use_count(node),
                delta,
                "output count",
            )
        } else {
            let index = node.id - self.base_len;
            Self::adjust_virtual_count(
                &mut self.virtual_ands[index].output_uses,
                index,
                delta,
                "output count",
            )
        }
    }

    fn adjust_real_count(
        deltas: &mut BTreeMap<AigRef, isize>,
        node: AigRef,
        base: usize,
        delta: isize,
        label: &str,
    ) -> Result<(), String> {
        let current_delta = deltas.get(&node).copied().unwrap_or(0);
        let new_delta = current_delta + delta;
        if (base as isize) + new_delta < 0 {
            return Err(format!("{} underflow for {:?}", label, node));
        }
        if new_delta == 0 {
            deltas.remove(&node);
        } else {
            deltas.insert(node, new_delta);
        }
        Ok(())
    }

    fn adjust_virtual_count(
        count: &mut usize,
        index: usize,
        delta: isize,
        label: &str,
    ) -> Result<(), String> {
        if delta >= 0 {
            *count += delta as usize;
            return Ok(());
        }
        *count = count
            .checked_sub((-delta) as usize)
            .ok_or_else(|| format!("{} underflow for virtual node {}", label, index))?;
        Ok(())
    }

    fn use_count(&self, node: AigRef) -> Result<usize, String> {
        self.validate_node_index(node)?;
        if node.id < self.base_len {
            Self::count_with_delta(
                self.state.use_count(node),
                self.use_count_deltas.get(&node).copied().unwrap_or(0),
                "use count",
                node,
            )
        } else {
            Ok(self.virtual_ands[node.id - self.base_len].uses)
        }
    }

    fn output_count(&self, node: AigRef) -> Result<usize, String> {
        self.validate_node_index(node)?;
        if node.id < self.base_len {
            Self::count_with_delta(
                self.state.output_use_count(node),
                self.output_count_deltas.get(&node).copied().unwrap_or(0),
                "output count",
                node,
            )
        } else {
            Ok(self.virtual_ands[node.id - self.base_len].output_uses)
        }
    }

    fn count_with_delta(
        base: usize,
        delta: isize,
        label: &str,
        node: AigRef,
    ) -> Result<usize, String> {
        let value = (base as isize) + delta;
        if value < 0 {
            return Err(format!("{} underflow for {:?}", label, node));
        }
        Ok(value as usize)
    }

    fn validate_operand(&self, op: AigOperand) -> Result<(), String> {
        self.validate_node_index(op.node)?;
        if !self.is_live_node(op.node) {
            return Err(format!("operand {:?} references inactive node", op));
        }
        Ok(())
    }

    fn validate_node_index(&self, node: AigRef) -> Result<(), String> {
        if node.id >= self.base_len + self.virtual_ands.len() {
            return Err(format!("node {:?} out of bounds", node));
        }
        Ok(())
    }

    fn is_live_node(&self, node: AigRef) -> bool {
        if self.deleted.contains(&node) {
            return false;
        }
        if node.id < self.base_len {
            self.state.is_live(node)
        } else {
            self.virtual_and(node).is_some()
        }
    }

    fn is_and_node(&self, node: AigRef) -> bool {
        if node.id < self.base_len {
            matches!(self.state.gate_fn().gates[node.id], AigNode::And2 { .. })
        } else {
            self.virtual_and(node).is_some()
        }
    }

    fn virtual_and(&self, node: AigRef) -> Option<&VirtualAnd> {
        self.virtual_ands.get(node.id.checked_sub(self.base_len)?)
    }

    fn false_op() -> AigOperand {
        AigOperand {
            node: AigRef { id: 0 },
            negated: false,
        }
    }

    fn true_op() -> AigOperand {
        Self::false_op().negate()
    }

    fn is_false(op: AigOperand) -> bool {
        op.node.id == 0 && !op.negated
    }

    fn is_true(op: AigOperand) -> bool {
        op.node.id == 0 && op.negated
    }

    fn op_from_lit(lit: Lit, ops: &[AigOperand]) -> AigOperand {
        let mut op = ops[lit.id as usize];
        if lit.negated {
            op = op.negate();
        }
        op
    }

    fn replace_operand_node(op: AigOperand, old: AigRef, replacement: AigOperand) -> AigOperand {
        if op.node == old {
            AigOperand {
                node: replacement.node,
                negated: op.negated ^ replacement.negated,
            }
        } else {
            op
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cut_db::fragment::FIRST_NODE_ID;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    #[test]
    fn virtual_cost_matches_real_rewire_and_dangling_cleanup() {
        let mut gb = GateBuilder::new("cost".into(), GateBuilderOptions::no_opt());
        let a = *gb.add_input("a".into(), 1).get_lsb(0);
        let b = *gb.add_input("b".into(), 1).get_lsb(0);
        let c = *gb.add_input("c".into(), 1).get_lsb(0);
        let ab = gb.add_and_binary(a, b);
        let old = gb.add_and_binary(ab, c);
        let doubled = gb.add_and_binary(old, old);
        let single = gb.add_and_binary(old, a);
        for (name, op) in [("root", old), ("double", doubled), ("single", single)] {
            gb.add_output(name.into(), crate::aig::AigBitVector::from_bit(op));
        }
        let mut state = DynamicStructuralHash::new(gb.build()).unwrap();
        let and = |a, b| FragmentNode::And2 {
            a: Lit::new(a, false),
            b: Lit::new(b, false),
        };
        // One unused virtual gate followed by (b & c) & a in the live fragment.
        let frag = GateFnFragment {
            nodes: vec![and(0, 2), and(1, 2), and(FIRST_NODE_ID + 1, 0)],
            output: Lit::new(FIRST_NODE_ID + 2, false),
        };
        let repl = Replacement {
            root: old.node,
            leaf_ops: vec![a, b, c],
            implementation: ReplacementImpl::Fragment {
                frag,
                input_depths: [0; 4],
            },
            score_depth: 0,
            score_ands: 0,
            raw_score_ands: 0,
            structural_hash_only_area_win: false,
        };
        let diff = gate_count_diff_for_replacement(&state, &repl)
            .unwrap()
            .unwrap();
        let before = state.live_and_count();
        let materialized = materialize_replacement(&mut state, &repl).unwrap().unwrap();
        assert!(state.is_live(materialized.replacement_op.node));
        assert!(!state.is_live(old.node) && !state.is_live(ab.node));
        assert_eq!(diff.before_live_ands, before);
        assert_eq!(diff.after_live_ands, state.live_and_count());
        assert_eq!(diff.live_and_delta, 0);
    }
}

#[cfg(test)]
mod protected_tests;
