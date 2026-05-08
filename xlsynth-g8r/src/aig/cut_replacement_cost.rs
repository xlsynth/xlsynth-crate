// SPDX-License-Identifier: Apache-2.0

//! Exact live-gate costing for cut replacement candidates.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::aig::cut_db_rewrite::{
    Replacement, ReplacementImpl, apply_replacement_to_dynamic_hash,
    cleanup_dangling_new_dynamic_hash_nodes,
};
use crate::aig::dynamic_depth::DynamicDepthState;
use crate::aig::dynamic_structural_hash::DynamicStructuralHash;
use crate::aig::gate::{AigNode, AigOperand, AigRef};
use crate::cut_db::fragment::{FragmentNode, GateFnFragment, Lit};

/// Summary of a replacement that was materialized into the dynamic AIG.
#[derive(Debug, Clone, Copy)]
pub(super) struct MaterializedReplacement {
    /// First gate id that did not exist before materializing the replacement.
    pub(super) first_new_id: usize,
    /// Root operand produced by the replacement fragment after local strashing.
    pub(super) replacement_op: AigOperand,
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
    if !state.is_live(replacement.root) {
        return Ok(None);
    }
    let mut overlay = VirtualCostOverlay::new(state);
    let replacement_op = overlay.instantiate_replacement(replacement)?;
    overlay.replace_node_with_operand(replacement.root, replacement_op)?;
    overlay.cleanup_dangling_virtual_nodes()?;

    let before_live_ands = state.live_and_count();
    let after_live_ands = overlay.after_live_and_count();
    Ok(Some(ReplacementGateCountDiff {
        before_live_ands,
        after_live_ands,
        live_and_delta: before_live_ands as isize - after_live_ands as isize,
    }))
}

/// Returns the exact global output depth after applying one replacement.
///
/// This uses the same virtual replacement model as area costing, so depth-mode
/// candidate checks do not need to clone and trial-mutate the whole dynamic
/// AIG.
pub(super) fn output_depth_after_replacement(
    state: &DynamicStructuralHash,
    depth_state: &DynamicDepthState,
    replacement: &Replacement,
) -> Result<Option<usize>, String> {
    if !state.is_live(replacement.root) {
        return Ok(None);
    }
    let mut overlay = VirtualCostOverlay::new(state);
    let replacement_op = overlay.instantiate_replacement(replacement)?;
    overlay.replace_node_with_operand(replacement.root, replacement_op)?;
    overlay.cleanup_dangling_virtual_nodes()?;
    overlay
        .max_output_depth(depth_state.forward_depths())
        .map(Some)
}

/// Applies a replacement to the real dynamic hash state.
pub(super) fn materialize_replacement(
    state: &mut DynamicStructuralHash,
    replacement: &Replacement,
) -> Result<Option<MaterializedReplacement>, String> {
    let first_new_id = state.gate_fn().gates.len();
    let Some(replacement_op) = apply_replacement_to_dynamic_hash(state, replacement)? else {
        return Ok(None);
    };
    cleanup_dangling_new_dynamic_hash_nodes(state, first_new_id)?;
    Ok(Some(MaterializedReplacement {
        first_new_id,
        replacement_op,
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

struct VirtualCostOverlay<'a> {
    state: &'a DynamicStructuralHash,
    base_len: usize,
    use_count_deltas: BTreeMap<AigRef, isize>,
    output_count_deltas: BTreeMap<AigRef, isize>,
    virtual_use_counts: Vec<usize>,
    virtual_output_counts: Vec<usize>,
    virtual_children: BTreeMap<AigRef, (AigOperand, AigOperand)>,
    virtual_by_key: BTreeMap<AndKey, AigOperand>,
    rewritten_children: BTreeMap<AigRef, (AigOperand, AigOperand)>,
    rewritten_by_key: BTreeMap<AndKey, AigOperand>,
    output_replacements: BTreeMap<AigRef, AigOperand>,
    deleted: BTreeSet<AigRef>,
}

impl<'a> VirtualCostOverlay<'a> {
    fn new(state: &'a DynamicStructuralHash) -> Self {
        let base_len = state.gate_fn().gates.len();
        Self {
            state,
            base_len,
            use_count_deltas: BTreeMap::new(),
            output_count_deltas: BTreeMap::new(),
            virtual_use_counts: Vec::new(),
            virtual_output_counts: Vec::new(),
            virtual_children: BTreeMap::new(),
            virtual_by_key: BTreeMap::new(),
            rewritten_children: BTreeMap::new(),
            rewritten_by_key: BTreeMap::new(),
            output_replacements: BTreeMap::new(),
            deleted: BTreeSet::new(),
        }
    }

    fn instantiate_replacement(&mut self, replacement: &Replacement) -> Result<AigOperand, String> {
        match &replacement.implementation {
            ReplacementImpl::Fragment { frag, .. } => {
                self.instantiate_fragment(frag, &replacement.leaf_ops)
            }
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
                if old_a.node != old && old_b.node != old {
                    continue;
                }
                self.remove_rewritten_index_node(fanout);
                self.decrement_use(old_a.node)?;
                self.decrement_use(old_b.node)?;
                let new_a = Self::replace_operand_node(old_a, old, replacement);
                let new_b = Self::replace_operand_node(old_b, old, replacement);
                self.increment_use(new_a.node)?;
                self.increment_use(new_b.node)?;

                let key = AndKey::new(new_a, new_b);
                if let Some(existing) = self.lookup_existing_and(key, Some(fanout)) {
                    self.rewritten_children.insert(fanout, (new_a, new_b));
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
        for id in (self.base_len..self.base_len + self.virtual_use_counts.len()).rev() {
            let node = AigRef { id };
            if self.is_live_node(node) && self.use_count(node)? == 0 {
                self.delete_dangling_mffc_from(node)?;
            }
        }
        Ok(())
    }

    fn after_live_and_count(&self) -> usize {
        let deleted_real_ands = self
            .deleted
            .iter()
            .filter(|node| {
                node.id < self.base_len
                    && matches!(self.state.gate_fn().gates[node.id], AigNode::And2 { .. })
            })
            .count();
        let live_virtual_ands = self
            .virtual_children
            .keys()
            .filter(|node| !self.deleted.contains(node))
            .count();
        self.state.live_and_count() + live_virtual_ands - deleted_real_ands
    }

    fn max_output_depth(&self, base_forward_depths: &[usize]) -> Result<usize, String> {
        let affected = self.affected_forward_cone()?;
        let mut memo = BTreeMap::new();
        let mut max_depth = 0usize;
        for output in &self.state.gate_fn().outputs {
            for op in output.bit_vector.iter_lsb_to_msb() {
                let op = self.current_output_operand(*op);
                self.validate_operand(op)?;
                let depth =
                    self.forward_depth(op.node, base_forward_depths, &affected, &mut memo)?;
                max_depth = max_depth.max(depth);
            }
        }
        Ok(max_depth)
    }

    fn affected_forward_cone(&self) -> Result<BTreeSet<AigRef>, String> {
        let mut affected = BTreeSet::new();
        let mut queue = VecDeque::new();
        for node in self
            .virtual_children
            .keys()
            .chain(self.rewritten_children.keys())
            .chain(self.deleted.iter())
            .copied()
        {
            if affected.insert(node) {
                queue.push_back(node);
            }
        }
        for (old, replacement) in &self.output_replacements {
            if affected.insert(*old) {
                queue.push_back(*old);
            }
            if affected.insert(replacement.node) {
                queue.push_back(replacement.node);
            }
        }

        while let Some(node) = queue.pop_front() {
            for fanout in self.current_fanout_nodes(node)? {
                if affected.insert(fanout) {
                    queue.push_back(fanout);
                }
            }
        }
        Ok(affected)
    }

    fn current_fanout_nodes(&self, node: AigRef) -> Result<Vec<AigRef>, String> {
        let mut fanouts = BTreeSet::new();
        if node.id < self.base_len {
            for fanout in self.state.fanout_nodes(node) {
                if !self.is_live_node(fanout) {
                    continue;
                }
                let (a, b) = self.current_and_operands(fanout)?;
                if a.node == node || b.node == node {
                    fanouts.insert(fanout);
                }
            }
        }
        for (fanout, (a, b)) in self
            .rewritten_children
            .iter()
            .chain(self.virtual_children.iter())
        {
            if self.is_live_node(*fanout) && (a.node == node || b.node == node) {
                fanouts.insert(*fanout);
            }
        }
        Ok(fanouts.into_iter().collect())
    }

    fn forward_depth(
        &self,
        node: AigRef,
        base_forward_depths: &[usize],
        affected: &BTreeSet<AigRef>,
        memo: &mut BTreeMap<AigRef, usize>,
    ) -> Result<usize, String> {
        let mut marks = BTreeMap::new();
        let mut stack = vec![(node, false)];

        while let Some((current, exit)) = stack.pop() {
            if memo.contains_key(&current) {
                continue;
            }
            self.validate_node_index(current)?;
            if !self.is_live_node(current) {
                return Err(format!(
                    "cannot compute depth for inactive node {:?}",
                    current
                ));
            }
            if self.can_use_base_forward_depth(current, affected) {
                let depth = base_forward_depths
                    .get(current.id)
                    .copied()
                    .ok_or_else(|| format!("node {:?} has no base forward depth", current))?;
                memo.insert(current, depth);
                continue;
            }

            if exit {
                let depth = if current.id < self.base_len
                    && !matches!(self.state.gate_fn().gates[current.id], AigNode::And2 { .. })
                {
                    0
                } else {
                    let (a, b) = self.current_and_operands(current)?;
                    let a_depth = memo
                        .get(&a.node)
                        .copied()
                        .ok_or_else(|| format!("missing forward depth for {:?}", a.node))?;
                    let b_depth = memo
                        .get(&b.node)
                        .copied()
                        .ok_or_else(|| format!("missing forward depth for {:?}", b.node))?;
                    a_depth
                        .max(b_depth)
                        .checked_add(1)
                        .ok_or_else(|| format!("forward depth overflow at {:?}", current))?
                };
                marks.insert(current, 2u8);
                memo.insert(current, depth);
                continue;
            }

            match marks.get(&current).copied() {
                Some(1) => return Err(format!("cycle detected at {:?}", current)),
                Some(2) => continue,
                _ => {}
            }
            marks.insert(current, 1u8);
            stack.push((current, true));

            if current.id < self.base_len
                && !matches!(self.state.gate_fn().gates[current.id], AigNode::And2 { .. })
            {
                continue;
            }
            let (a, b) = self.current_and_operands(current)?;
            if !memo.contains_key(&b.node) {
                stack.push((b.node, false));
            }
            if !memo.contains_key(&a.node) {
                stack.push((a.node, false));
            }
        }

        memo.get(&node)
            .copied()
            .ok_or_else(|| format!("missing forward depth for {:?}", node))
    }

    fn can_use_base_forward_depth(&self, node: AigRef, affected: &BTreeSet<AigRef>) -> bool {
        node.id < self.base_len
            && !affected.contains(&node)
            && !self.rewritten_children.contains_key(&node)
    }

    fn add_and(&mut self, lhs: AigOperand, rhs: AigOperand) -> Result<AigOperand, String> {
        self.validate_operand(lhs)?;
        self.validate_operand(rhs)?;

        let key = AndKey::new(lhs, rhs);
        if let Some(existing) = self.lookup_existing_and(key, None) {
            return Ok(existing);
        }

        let node = AigRef {
            id: self.base_len + self.virtual_use_counts.len(),
        };
        self.virtual_use_counts.push(0);
        self.virtual_output_counts.push(0);
        self.virtual_children.insert(node, (lhs, rhs));
        let op = AigOperand {
            node,
            negated: false,
        };
        self.virtual_by_key.insert(key, op);
        self.increment_use(lhs.node)?;
        self.increment_use(rhs.node)?;
        Ok(op)
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
        if let Some(children) = self.virtual_children.get(&node).copied() {
            return Ok(children);
        }
        match self.state.gate_fn().gates.get(node.id) {
            Some(AigNode::And2 { a, b, .. }) => Ok((*a, *b)),
            Some(other) => Err(format!("expected And2 at {:?}; got {:?}", node, other)),
            None => Err(format!("node {:?} out of bounds", node)),
        }
    }

    fn remove_virtual_index_node(&mut self, node: AigRef) {
        if let Some((a, b)) = self.virtual_children.get(&node).copied() {
            let key = AndKey::new(a, b);
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
            Self::adjust_virtual_count(
                &mut self.virtual_use_counts,
                node.id - self.base_len,
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
            Self::adjust_virtual_count(
                &mut self.virtual_output_counts,
                node.id - self.base_len,
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
        counts: &mut [usize],
        index: usize,
        delta: isize,
        label: &str,
    ) -> Result<(), String> {
        let count = counts
            .get_mut(index)
            .ok_or_else(|| format!("virtual node index {} out of bounds", index))?;
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
            Ok(self.virtual_use_counts[node.id - self.base_len])
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
            Ok(self.virtual_output_counts[node.id - self.base_len])
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
        if node.id >= self.base_len + self.virtual_use_counts.len() {
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
            self.virtual_children.contains_key(&node)
        }
    }

    fn is_and_node(&self, node: AigRef) -> bool {
        if node.id < self.base_len {
            matches!(self.state.gate_fn().gates[node.id], AigNode::And2 { .. })
        } else {
            self.virtual_children.contains_key(&node)
        }
    }

    fn false_op() -> AigOperand {
        AigOperand {
            node: AigRef { id: 0 },
            negated: false,
        }
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

    fn current_output_operand(&self, op: AigOperand) -> AigOperand {
        if let Some(replacement) = self.output_replacements.get(&op.node).copied() {
            AigOperand {
                node: replacement.node,
                negated: op.negated ^ replacement.negated,
            }
        } else {
            op
        }
    }
}
