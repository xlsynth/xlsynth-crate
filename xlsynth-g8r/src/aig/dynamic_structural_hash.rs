// SPDX-License-Identifier: Apache-2.0

//! Mutable AIG edit state with an incrementally updated local strash table.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use smallvec::SmallVec;

use crate::aig::gate::{AigNode, AigOperand, AigRef, GateFn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LocalAndKey {
    lhs: AigOperand,
    rhs: AigOperand,
}

impl LocalAndKey {
    fn new(lhs: AigOperand, rhs: AigOperand) -> Self {
        if lhs <= rhs {
            Self { lhs, rhs }
        } else {
            Self { lhs: rhs, rhs: lhs }
        }
    }
}

type LocalAndBucket = SmallVec<[AigRef; 2]>;
const FANOUT_INLINE_CAPACITY: usize = 4;

type FanoutSmallVec = SmallVec<[AigRef; FANOUT_INLINE_CAPACITY]>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum FanoutBucket {
    Small(FanoutSmallVec),
    Large(BTreeSet<AigRef>),
}

impl Default for FanoutBucket {
    fn default() -> Self {
        Self::Small(SmallVec::new())
    }
}

impl FanoutBucket {
    fn len(&self) -> usize {
        match self {
            Self::Small(nodes) => nodes.len(),
            Self::Large(nodes) => nodes.len(),
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn insert(&mut self, node: AigRef) -> bool {
        match self {
            Self::Small(nodes) => {
                if nodes.contains(&node) {
                    return false;
                }
                if nodes.len() < FANOUT_INLINE_CAPACITY {
                    nodes.push(node);
                    return true;
                }
                let mut large = nodes.iter().copied().collect::<BTreeSet<_>>();
                let inserted = large.insert(node);
                *self = Self::Large(large);
                inserted
            }
            Self::Large(nodes) => nodes.insert(node),
        }
    }

    fn remove(&mut self, node: AigRef) -> bool {
        match self {
            Self::Small(nodes) => {
                let Some(index) = nodes.iter().position(|candidate| *candidate == node) else {
                    return false;
                };
                nodes.swap_remove(index);
                true
            }
            Self::Large(nodes) => nodes.remove(&node),
        }
    }

    fn to_vec(&self) -> Vec<AigRef> {
        match self {
            Self::Small(nodes) => nodes.iter().copied().collect(),
            Self::Large(nodes) => nodes.iter().copied().collect(),
        }
    }

    fn contents_equal(&self, other: &Self) -> bool {
        self.len() == other.len() && self.to_vec().into_iter().all(|node| other.contains(node))
    }

    fn contains(&self, node: AigRef) -> bool {
        match self {
            Self::Small(nodes) => nodes.contains(&node),
            Self::Large(nodes) => nodes.contains(&node),
        }
    }
}

/// A mutable AIG plus active-node, fanout, and local-strash side state.
///
/// `GateFn` remains append-only while edits are applied. Deleting a node marks
/// it inactive, removes it from fanout/hash side structures, and leaves the
/// physical `gates` slot in place so existing `AigRef` indices remain stable.
#[derive(Debug, Clone)]
pub struct DynamicStructuralHash {
    g: GateFn,
    live: Vec<bool>,
    fanouts: Vec<FanoutBucket>,
    output_uses: Vec<usize>,
    use_counts: Vec<usize>,
    by_key: BTreeMap<LocalAndKey, LocalAndBucket>,
    live_and_count: usize,
}

/// Identifies a mutable edge in a live AIG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeRef {
    /// An input of an `And2` node. `input_index` must be 0 or 1.
    AndFanin { node: AigRef, input_index: usize },
    /// One bit of one output vector.
    OutputBit {
        output_index: usize,
        bit_index: usize,
    },
}

#[derive(Debug, Clone, Copy)]
struct ReplacementQueueItem {
    old: AigRef,
    replacement: AigOperand,
    is_duplicate_merge: bool,
}

impl DynamicStructuralHash {
    /// Builds a dynamic local-strash state over an existing `GateFn`.
    pub fn new(g: GateFn) -> Result<Self, String> {
        let live = vec![true; g.gates.len()];
        build_dynamic_structural_hash(g, live)
    }

    /// Returns the underlying append-only graph.
    pub fn gate_fn(&self) -> &GateFn {
        &self.g
    }

    /// Returns the active-node bitset indexed by `AigRef::id`.
    pub fn live_mask(&self) -> &[bool] {
        &self.live
    }

    /// Returns true if `node` is currently active in this edit state.
    pub fn is_live(&self, node: AigRef) -> bool {
        self.live.get(node.id).copied().unwrap_or(false)
    }

    /// Returns all active node refs in deterministic order.
    pub fn live_nodes(&self) -> Vec<AigRef> {
        self.live
            .iter()
            .enumerate()
            .filter_map(|(id, live)| live.then_some(AigRef { id }))
            .collect()
    }

    /// Returns all active AND node refs in deterministic order.
    pub fn live_and_nodes(&self) -> Vec<AigRef> {
        self.live_nodes()
            .into_iter()
            .filter(|node| matches!(self.g.gates[node.id], AigNode::And2 { .. }))
            .collect()
    }

    /// Returns the number of active AND nodes.
    pub fn live_and_count(&self) -> usize {
        self.live_and_count
    }

    /// Returns the active use count of `node`, including output bits and fanin
    /// edges. Repeated fanin edges such as `x & x` count twice.
    pub fn use_count(&self, node: AigRef) -> usize {
        self.use_counts.get(node.id).copied().unwrap_or(0)
    }

    /// Returns the active fanout count of `node`.
    pub fn fanout_count(&self, node: AigRef) -> usize {
        self.fanouts
            .get(node.id)
            .map(|fanouts| fanouts.len())
            .unwrap_or(0)
    }

    /// Returns the active direct fanouts of `node` in deterministic order.
    pub fn fanout_nodes(&self, node: AigRef) -> Vec<AigRef> {
        self.fanouts
            .get(node.id)
            .map(FanoutBucket::to_vec)
            .unwrap_or_default()
            .into_iter()
            .filter(|fanout| self.is_live(*fanout))
            .collect()
    }

    /// Returns the number of output bits that currently reference `node`.
    pub fn output_use_count(&self, node: AigRef) -> usize {
        self.output_uses.get(node.id).copied().unwrap_or(0)
    }

    /// Looks up a live AND with exactly these immediate operands.
    pub fn lookup_and(&self, lhs: AigOperand, rhs: AigOperand) -> Option<AigOperand> {
        self.lookup_and_node(lhs, rhs).map(|node| AigOperand {
            node,
            negated: false,
        })
    }

    /// Looks up a live AND with these operands, ignoring nodes in `excluded`.
    pub fn lookup_and_excluding(
        &self,
        lhs: AigOperand,
        rhs: AigOperand,
        excluded: &BTreeSet<AigRef>,
    ) -> Option<AigOperand> {
        if !self.operand_is_live(lhs) || !self.operand_is_live(rhs) {
            return None;
        }
        self.lookup_key_excluding_nodes(LocalAndKey::new(lhs, rhs), excluded)
            .map(|node| AigOperand {
                node,
                negated: false,
            })
    }

    /// Looks up a live AND with these operands, ignoring nodes matched by
    /// `excluded`.
    pub fn lookup_and_excluding_predicate<F>(
        &self,
        lhs: AigOperand,
        rhs: AigOperand,
        excluded: F,
    ) -> Option<AigOperand>
    where
        F: FnMut(AigRef) -> bool,
    {
        if !self.operand_is_live(lhs) || !self.operand_is_live(rhs) {
            return None;
        }
        self.lookup_key_excluding_predicate(LocalAndKey::new(lhs, rhs), excluded)
            .map(|node| AigOperand {
                node,
                negated: false,
            })
    }

    /// Adds `lhs & rhs`, returning an existing local-strash representative when
    /// possible.
    pub fn add_and(&mut self, lhs: AigOperand, rhs: AigOperand) -> Result<AigOperand, String> {
        self.add_and_with_pir_node_ids(lhs, rhs, &[])
    }

    /// Adds `lhs & rhs` and attaches `pir_node_ids` to the returned node.
    ///
    /// This is a pure local-strash operation: it reuses exact existing ANDs but
    /// does not perform Boolean simplifications such as constant folding.
    pub fn add_and_with_pir_node_ids(
        &mut self,
        lhs: AigOperand,
        rhs: AigOperand,
        pir_node_ids: &[u32],
    ) -> Result<AigOperand, String> {
        self.validate_operand(lhs)?;
        self.validate_operand(rhs)?;
        if let Some(existing) = self.lookup_and(lhs, rhs) {
            return self.union_pir_node_ids_into_operand(existing, pir_node_ids);
        }

        let node = AigRef {
            id: self.g.gates.len(),
        };
        self.g.gates.push(AigNode::And2 {
            a: lhs,
            b: rhs,
            tags: None,
            pir_node_ids: Default::default(),
        });
        self.live.push(true);
        self.fanouts.push(FanoutBucket::default());
        self.output_uses.push(0);
        self.use_counts.push(0);
        self.live_and_count += 1;
        self.add_fanin_links(node, lhs, rhs);
        self.insert_index_node(node)?;
        self.add_pir_node_ids(node, pir_node_ids)?;
        Ok(AigOperand {
            node,
            negated: false,
        })
    }

    /// Unions provenance IDs into an active node.
    pub fn add_pir_node_ids(&mut self, node: AigRef, pir_node_ids: &[u32]) -> Result<(), String> {
        self.validate_live_node(node)?;
        self.g.gates[node.id].try_add_pir_node_ids(pir_node_ids);
        Ok(())
    }

    /// Moves one mutable graph edge to `new_operand`.
    pub fn move_edge(&mut self, edge: EdgeRef, new_operand: AigOperand) -> Result<(), String> {
        match edge {
            EdgeRef::AndFanin { node, input_index } => {
                self.move_fanin_edge(node, input_index, new_operand)
            }
            EdgeRef::OutputBit {
                output_index,
                bit_index,
            } => self.move_output_edge(output_index, bit_index, new_operand),
        }
    }

    /// Moves fanin `input_index` of an active AND node to `new_operand`.
    ///
    /// If the edit exposes a local structural duplicate, the edited node is
    /// replaced with the existing representative and the replacement cascades
    /// through direct fanouts, matching ABC-style strash maintenance.
    pub fn move_fanin_edge(
        &mut self,
        node: AigRef,
        input_index: usize,
        new_operand: AigOperand,
    ) -> Result<(), String> {
        self.validate_live_node(node)?;
        self.validate_operand(new_operand)?;
        if input_index > 1 {
            return Err(format!("input_index must be 0 or 1; got {}", input_index));
        }

        let (old_a, old_b) = self.and_operands(node)?;
        let old_operand = if input_index == 0 { old_a } else { old_b };
        if old_operand == new_operand {
            return Ok(());
        }

        self.remove_index_node(node)?;
        self.remove_fanin_links(node, old_a, old_b);
        match &mut self.g.gates[node.id] {
            AigNode::And2 { a, b, .. } => {
                if input_index == 0 {
                    *a = new_operand;
                } else {
                    *b = new_operand;
                }
            }
            _ => unreachable!("node type checked above"),
        }
        let (new_a, new_b) = self.and_operands(node)?;
        self.add_fanin_links(node, new_a, new_b);

        let key = LocalAndKey::new(new_a, new_b);
        if let Some(existing) = self.lookup_key_excluding(key, Some(node)) {
            self.replace_node_with_operand(node, existing.into())?;
        } else {
            self.insert_index_node(node)?;
        }
        Ok(())
    }

    /// Moves an output bit to `new_operand`.
    pub fn move_output_edge(
        &mut self,
        output_index: usize,
        bit_index: usize,
        new_operand: AigOperand,
    ) -> Result<(), String> {
        self.validate_operand(new_operand)?;
        if output_index >= self.g.outputs.len() {
            return Err(format!(
                "output_index {} out of bounds for {} outputs",
                output_index,
                self.g.outputs.len()
            ));
        }
        if bit_index >= self.g.outputs[output_index].get_bit_count() {
            return Err(format!(
                "bit_index {} out of bounds for output {} width {}",
                bit_index,
                output_index,
                self.g.outputs[output_index].get_bit_count()
            ));
        }

        let old_operand = *self.g.outputs[output_index].bit_vector.get_lsb(bit_index);
        if old_operand == new_operand {
            return Ok(());
        }
        self.output_uses[old_operand.node.id] = self.output_uses[old_operand.node.id]
            .checked_sub(1)
            .expect("old output operand should have a positive output use count");
        self.use_counts[old_operand.node.id] = self.use_counts[old_operand.node.id]
            .checked_sub(1)
            .expect("old output operand should have a positive use count");
        self.output_uses[new_operand.node.id] += 1;
        self.use_counts[new_operand.node.id] += 1;
        self.g.outputs[output_index]
            .bit_vector
            .set_lsb(bit_index, new_operand);
        Ok(())
    }

    /// Marks an active dangling AND node as inactive and removes it from the
    /// local strash table.
    pub fn delete_node(&mut self, node: AigRef) -> Result<(), String> {
        self.validate_live_node(node)?;
        if !matches!(self.g.gates[node.id], AigNode::And2 { .. }) {
            return Err(format!("only And2 nodes can be deleted; got {:?}", node));
        }
        if !self.fanouts[node.id].is_empty() || self.output_uses[node.id] != 0 {
            return Err(format!(
                "cannot delete {:?}; fanouts={} output_uses={}",
                node,
                self.fanouts[node.id].len(),
                self.output_uses[node.id]
            ));
        }

        self.delete_dangling_mffc_from(node)
    }

    /// Rebuilds a reference dynamic state from scratch and compares side state.
    pub fn check_invariants(&self) -> Result<(), String> {
        let rebuilt = build_dynamic_structural_hash(self.g.clone(), self.live.clone())?;
        if !fanout_vectors_equivalent(&self.fanouts, &rebuilt.fanouts) {
            return Err(format!(
                "fanout mismatch: current={:?} reference={:?}",
                self.fanouts, rebuilt.fanouts
            ));
        }
        if self.output_uses != rebuilt.output_uses {
            return Err(format!(
                "output use mismatch: current={:?} reference={:?}",
                self.output_uses, rebuilt.output_uses
            ));
        }
        if self.use_counts != rebuilt.use_counts {
            return Err(format!(
                "use count mismatch: current={:?} reference={:?}",
                self.use_counts, rebuilt.use_counts
            ));
        }
        if !by_key_maps_equivalent(&self.by_key, &rebuilt.by_key) {
            return Err(format!(
                "local strash mismatch: current={:?} reference={:?}",
                self.by_key, rebuilt.by_key
            ));
        }
        if self.live_and_count != rebuilt.live_and_count {
            return Err(format!(
                "live AND count mismatch: current={} reference={}",
                self.live_and_count, rebuilt.live_and_count
            ));
        }
        Ok(())
    }

    fn validate_live_node(&self, node: AigRef) -> Result<(), String> {
        if node.id >= self.g.gates.len() {
            return Err(format!("node {:?} out of bounds", node));
        }
        if !self.live[node.id] {
            return Err(format!("node {:?} is inactive", node));
        }
        Ok(())
    }

    fn validate_operand(&self, op: AigOperand) -> Result<(), String> {
        self.validate_live_node(op.node)
    }

    fn operand_is_live(&self, op: AigOperand) -> bool {
        self.is_live(op.node)
    }

    fn lookup_and_node(&self, lhs: AigOperand, rhs: AigOperand) -> Option<AigRef> {
        if !self.operand_is_live(lhs) || !self.operand_is_live(rhs) {
            return None;
        }
        self.lookup_key_excluding(LocalAndKey::new(lhs, rhs), None)
    }

    fn lookup_key_excluding(&self, key: LocalAndKey, excluded: Option<AigRef>) -> Option<AigRef> {
        self.lookup_key_excluding_predicate(key, |node| Some(node) == excluded)
    }

    fn lookup_key_excluding_nodes(
        &self,
        key: LocalAndKey,
        excluded: &BTreeSet<AigRef>,
    ) -> Option<AigRef> {
        self.lookup_key_excluding_predicate(key, |node| excluded.contains(&node))
    }

    fn lookup_key_excluding_predicate<F>(&self, key: LocalAndKey, mut excluded: F) -> Option<AigRef>
    where
        F: FnMut(AigRef) -> bool,
    {
        self.by_key.get(&key).and_then(|nodes| {
            nodes
                .iter()
                .copied()
                .find(|node| !excluded(*node) && self.is_live(*node))
        })
    }

    fn and_operands(&self, node: AigRef) -> Result<(AigOperand, AigOperand), String> {
        self.validate_live_node(node)?;
        match self.g.gates[node.id] {
            AigNode::And2 { a, b, .. } => Ok((a, b)),
            _ => Err(format!("node {:?} is not an And2", node)),
        }
    }

    fn add_fanin_links(&mut self, node: AigRef, a: AigOperand, b: AigOperand) {
        self.fanouts[a.node.id].insert(node);
        self.fanouts[b.node.id].insert(node);
        self.use_counts[a.node.id] += 1;
        self.use_counts[b.node.id] += 1;
    }

    fn remove_fanin_links(&mut self, node: AigRef, a: AigOperand, b: AigOperand) {
        self.fanouts[a.node.id].remove(node);
        self.fanouts[b.node.id].remove(node);
        self.use_counts[a.node.id] = self.use_counts[a.node.id]
            .checked_sub(1)
            .expect("removed fanin should have a positive use count");
        self.use_counts[b.node.id] = self.use_counts[b.node.id]
            .checked_sub(1)
            .expect("removed fanin should have a positive use count");
    }

    fn insert_index_node(&mut self, node: AigRef) -> Result<(), String> {
        if !self.is_live(node) || !matches!(self.g.gates[node.id], AigNode::And2 { .. }) {
            return Ok(());
        }
        let (a, b) = self.and_operands(node)?;
        let inserted =
            insert_bucket_node(self.by_key.entry(LocalAndKey::new(a, b)).or_default(), node);
        if !inserted {
            return Err(format!("node {:?} is already in local strash table", node));
        }
        Ok(())
    }

    fn remove_index_node(&mut self, node: AigRef) -> Result<(), String> {
        if !self.is_live(node) || !matches!(self.g.gates[node.id], AigNode::And2 { .. }) {
            return Ok(());
        }
        let (a, b) = self.and_operands(node)?;
        let key = LocalAndKey::new(a, b);
        let Some(bucket) = self.by_key.get_mut(&key) else {
            return Err(format!("local strash bucket missing for {:?}", node));
        };
        if !remove_bucket_node(bucket, node) {
            return Err(format!("local strash entry missing for {:?}", node));
        }
        if bucket.is_empty() {
            self.by_key.remove(&key);
        }
        Ok(())
    }

    fn remove_index_node_if_present(&mut self, node: AigRef) -> Result<(), String> {
        if !self.is_live(node) || !matches!(self.g.gates[node.id], AigNode::And2 { .. }) {
            return Ok(());
        }
        let (a, b) = self.and_operands(node)?;
        let key = LocalAndKey::new(a, b);
        let Some(bucket) = self.by_key.get_mut(&key) else {
            return Ok(());
        };
        remove_bucket_node(bucket, node);
        if bucket.is_empty() {
            self.by_key.remove(&key);
        }
        Ok(())
    }

    /// Replaces a live AND node with `replacement`, rewiring outputs and
    /// fanouts. Local strash duplicates exposed by the replacement are merged
    /// recursively, and newly dangling MFFCs are marked inactive.
    pub fn replace_node_with_operand(
        &mut self,
        old: AigRef,
        replacement: AigOperand,
    ) -> Result<(), String> {
        let mut queue = VecDeque::from([ReplacementQueueItem {
            old,
            replacement,
            is_duplicate_merge: false,
        }]);
        while let Some(item) = queue.pop_front() {
            let old = item.old;
            let mut replacement = item.replacement;
            if !self.is_live(old) {
                continue;
            }
            if item.is_duplicate_merge {
                replacement = match self.resolve_duplicate_merge_replacement(old, replacement)? {
                    Some(replacement) => replacement,
                    None => continue,
                };
            }
            if replacement.node == old {
                if replacement.negated {
                    return Err(format!("cannot replace {:?} with its own negation", old));
                }
                continue;
            }
            self.validate_operand(replacement)?;
            if !matches!(self.g.gates[old.id], AigNode::And2 { .. }) {
                return Err(format!("only And2 nodes can be replaced; got {:?}", old));
            }

            self.remove_index_node_if_present(old)?;
            self.rewrite_outputs(old, replacement);

            let direct_fanouts = self.fanouts[old.id].to_vec();
            for fanout in direct_fanouts {
                if !self.is_live(fanout) {
                    continue;
                }
                let (old_a, old_b) = self.and_operands(fanout)?;
                // A previous queued duplicate merge may have left this fanout
                // live but temporarily unindexed until its queue item runs.
                self.remove_index_node_if_present(fanout)?;
                self.remove_fanin_links(fanout, old_a, old_b);
                let new_a = replace_operand_node(old_a, old, replacement);
                let new_b = replace_operand_node(old_b, old, replacement);
                match &mut self.g.gates[fanout.id] {
                    AigNode::And2 { a, b, .. } => {
                        *a = new_a;
                        *b = new_b;
                    }
                    _ => unreachable!("fanout should be an And2"),
                }
                self.add_fanin_links(fanout, new_a, new_b);

                let key = LocalAndKey::new(new_a, new_b);
                if let Some(existing) = self.lookup_key_excluding(key, Some(fanout)) {
                    queue.push_back(ReplacementQueueItem {
                        old: fanout,
                        replacement: existing.into(),
                        is_duplicate_merge: true,
                    });
                } else {
                    self.insert_index_node(fanout)?;
                }
            }

            self.delete_dangling_mffc_from(old)?;
        }
        Ok(())
    }

    fn resolve_duplicate_merge_replacement(
        &mut self,
        old: AigRef,
        replacement: AigOperand,
    ) -> Result<Option<AigOperand>, String> {
        let (a, b) = self.and_operands(old)?;
        let key = LocalAndKey::new(a, b);
        if self.operand_matches_local_and_key(replacement, key) {
            return Ok(Some(replacement));
        }
        if let Some(existing) = self.lookup_key_excluding(key, Some(old)) {
            return Ok(Some(existing.into()));
        }
        self.remove_index_node_if_present(old)?;
        self.insert_index_node(old)?;
        Ok(None)
    }

    fn operand_matches_local_and_key(&self, operand: AigOperand, key: LocalAndKey) -> bool {
        if operand.negated || !self.is_live(operand.node) {
            return false;
        }
        let AigNode::And2 { a, b, .. } = self.g.gates[operand.node.id] else {
            return false;
        };
        LocalAndKey::new(a, b) == key
    }

    fn union_pir_node_ids_into_operand(
        &mut self,
        operand: AigOperand,
        pir_node_ids: &[u32],
    ) -> Result<AigOperand, String> {
        self.add_pir_node_ids(operand.node, pir_node_ids)?;
        Ok(operand)
    }

    fn rewrite_outputs(&mut self, old: AigRef, replacement: AigOperand) {
        for output_index in 0..self.g.outputs.len() {
            let width = self.g.outputs[output_index].get_bit_count();
            for bit_index in 0..width {
                let old_operand = *self.g.outputs[output_index].bit_vector.get_lsb(bit_index);
                if old_operand.node != old {
                    continue;
                }
                let new_operand = replace_operand_node(old_operand, old, replacement);
                self.output_uses[old.id] = self.output_uses[old.id]
                    .checked_sub(1)
                    .expect("old output operand should have a positive output use count");
                self.use_counts[old.id] = self.use_counts[old.id]
                    .checked_sub(1)
                    .expect("old output operand should have a positive use count");
                self.output_uses[new_operand.node.id] += 1;
                self.use_counts[new_operand.node.id] += 1;
                self.g.outputs[output_index]
                    .bit_vector
                    .set_lsb(bit_index, new_operand);
            }
        }
    }

    fn delete_dangling_mffc_from(&mut self, root: AigRef) -> Result<(), String> {
        self.validate_live_node(root)?;
        if !matches!(self.g.gates[root.id], AigNode::And2 { .. }) {
            return Err(format!("only And2 nodes can be deleted; got {:?}", root));
        }
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if !self.is_live(node) {
                continue;
            }
            if !matches!(self.g.gates[node.id], AigNode::And2 { .. }) {
                continue;
            }
            if self.use_counts[node.id] != 0 {
                if node == root {
                    return Err(format!(
                        "cannot delete {:?}; use_count={}",
                        node, self.use_counts[node.id]
                    ));
                }
                continue;
            }

            self.remove_index_node_if_present(node)?;
            let (a, b) = self.and_operands(node)?;
            self.remove_fanin_links(node, a, b);
            self.live[node.id] = false;
            self.live_and_count = self
                .live_and_count
                .checked_sub(1)
                .expect("deleted live AND should have contributed to live count");

            for child in [a.node, b.node] {
                if self.is_live(child)
                    && matches!(self.g.gates[child.id], AigNode::And2 { .. })
                    && self.use_counts[child.id] == 0
                {
                    stack.push(child);
                }
            }
        }
        Ok(())
    }
}

fn build_dynamic_structural_hash(
    g: GateFn,
    live: Vec<bool>,
) -> Result<DynamicStructuralHash, String> {
    if live.len() != g.gates.len() {
        return Err(format!(
            "live bitset length {} does not match gate count {}",
            live.len(),
            g.gates.len()
        ));
    }

    let mut fanouts = vec![FanoutBucket::default(); g.gates.len()];
    let mut output_uses = vec![0usize; g.gates.len()];
    let mut use_counts = vec![0usize; g.gates.len()];
    let mut by_key: BTreeMap<LocalAndKey, LocalAndBucket> = BTreeMap::new();
    let mut live_and_count = 0usize;
    for (id, node) in g.gates.iter().enumerate() {
        if !live[id] {
            continue;
        }
        if let AigNode::And2 { a, b, .. } = node {
            validate_live_operand(&g, &live, *a)?;
            validate_live_operand(&g, &live, *b)?;
            let node_ref = AigRef { id };
            fanouts[a.node.id].insert(node_ref);
            fanouts[b.node.id].insert(node_ref);
            use_counts[a.node.id] += 1;
            use_counts[b.node.id] += 1;
            live_and_count += 1;
            insert_bucket_node(
                by_key.entry(LocalAndKey::new(*a, *b)).or_default(),
                node_ref,
            );
        }
    }
    for output in &g.outputs {
        for op in output.bit_vector.iter_lsb_to_msb() {
            validate_live_operand(&g, &live, *op)?;
            output_uses[op.node.id] += 1;
            use_counts[op.node.id] += 1;
        }
    }
    validate_live_acyclic(&g, &live)?;

    Ok(DynamicStructuralHash {
        g,
        live,
        fanouts,
        output_uses,
        use_counts,
        by_key,
        live_and_count,
    })
}

fn insert_bucket_node(bucket: &mut LocalAndBucket, node: AigRef) -> bool {
    if bucket.contains(&node) {
        return false;
    }
    bucket.push(node);
    true
}

fn remove_bucket_node(bucket: &mut LocalAndBucket, node: AigRef) -> bool {
    let Some(index) = bucket.iter().position(|candidate| *candidate == node) else {
        return false;
    };
    bucket.swap_remove(index);
    true
}

fn fanout_vectors_equivalent(lhs: &[FanoutBucket], rhs: &[FanoutBucket]) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(lhs_bucket, rhs_bucket)| lhs_bucket.contents_equal(rhs_bucket))
}

fn by_key_maps_equivalent(
    lhs: &BTreeMap<LocalAndKey, LocalAndBucket>,
    rhs: &BTreeMap<LocalAndKey, LocalAndBucket>,
) -> bool {
    if lhs.len() != rhs.len() {
        return false;
    }
    lhs.iter().all(|(key, lhs_bucket)| {
        rhs.get(key)
            .is_some_and(|rhs_bucket| bucket_contents_equal(lhs_bucket, rhs_bucket))
    })
}

fn bucket_contents_equal(lhs: &LocalAndBucket, rhs: &LocalAndBucket) -> bool {
    lhs.len() == rhs.len() && lhs.iter().all(|node| rhs.contains(node))
}

fn validate_live_operand(g: &GateFn, live: &[bool], op: AigOperand) -> Result<(), String> {
    if op.node.id >= g.gates.len() {
        return Err(format!("operand {:?} out of bounds", op));
    }
    if !live[op.node.id] {
        return Err(format!("operand {:?} references inactive node", op));
    }
    Ok(())
}

fn validate_live_acyclic(g: &GateFn, live: &[bool]) -> Result<(), String> {
    let mut pending_fanins = vec![0usize; g.gates.len()];
    let mut fanouts = vec![Vec::<AigRef>::new(); g.gates.len()];
    let mut worklist = VecDeque::new();
    let mut processed = vec![false; g.gates.len()];
    let mut live_count = 0usize;

    for id in 0..g.gates.len() {
        if !live[id] {
            continue;
        }
        live_count += 1;
        match g.gates[id] {
            AigNode::Input { .. } | AigNode::Literal { .. } => {
                worklist.push_back(AigRef { id });
            }
            AigNode::And2 { a, b, .. } => {
                validate_live_operand(g, live, a)?;
                validate_live_operand(g, live, b)?;
                let node = AigRef { id };
                fanouts[a.node.id].push(node);
                if a.node == b.node {
                    pending_fanins[id] = 1;
                } else {
                    fanouts[b.node.id].push(node);
                    pending_fanins[id] = 2;
                }
            }
        }
    }

    let mut processed_count = 0usize;
    while let Some(node) = worklist.pop_front() {
        if processed[node.id] {
            continue;
        }
        processed[node.id] = true;
        processed_count += 1;
        for fanout in &fanouts[node.id] {
            pending_fanins[fanout.id] = pending_fanins[fanout.id]
                .checked_sub(1)
                .ok_or_else(|| format!("fanin dependency underflow at {:?}", fanout))?;
            if pending_fanins[fanout.id] == 0 {
                worklist.push_back(*fanout);
            }
        }
    }

    if processed_count != live_count {
        let node = live
            .iter()
            .enumerate()
            .find_map(|(id, is_live)| (*is_live && !processed[id]).then_some(AigRef { id }))
            .expect("processed count mismatch should have an unprocessed live node");
        return Err(format!("cycle detected at {:?}", node));
    }

    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::AigBitVector;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    fn simple_graph() -> (GateFn, AigOperand, AigOperand, AigOperand, AigOperand) {
        let mut gb = GateBuilder::new("dyn_hash".to_string(), GateBuilderOptions::no_opt());
        let a = *gb.add_input("a".to_string(), 1).get_lsb(0);
        let b = *gb.add_input("b".to_string(), 1).get_lsb(0);
        let c = *gb.add_input("c".to_string(), 1).get_lsb(0);
        let ab = gb.add_and_binary(a, b);
        gb.add_output("o".to_string(), AigBitVector::from_bit(ab));
        (gb.build(), a, b, c, ab)
    }

    fn deep_chain_graph(len: usize) -> GateFn {
        assert!(len != 0);
        let mut gb = GateBuilder::new("dyn_hash_deep".to_string(), GateBuilderOptions::no_opt());
        let a = *gb.add_input("a".to_string(), 1).get_lsb(0);
        let b = *gb.add_input("b".to_string(), 1).get_lsb(0);
        let mut acc = gb.add_and_binary(a, b);
        for _ in 1..len {
            acc = gb.add_and_binary(acc, b);
        }
        gb.add_output("o".to_string(), AigBitVector::from_bit(acc));
        gb.build()
    }

    #[test]
    fn add_and_reuses_commuted_existing_node() {
        let (g, a, b, _c, ab) = simple_graph();
        let mut state = DynamicStructuralHash::new(g).unwrap();
        state.check_invariants().unwrap();

        assert_eq!(state.add_and(b, a).unwrap(), ab);
        assert_eq!(state.lookup_and(a, b).unwrap(), ab);
        state.check_invariants().unwrap();
    }

    #[test]
    fn initial_state_handles_deep_chain_without_recursion() {
        const CHAIN_LEN: usize = 20_000;
        let state = DynamicStructuralHash::new(deep_chain_graph(CHAIN_LEN)).unwrap();

        assert_eq!(state.live_and_count(), CHAIN_LEN);
        state.check_invariants().unwrap();
    }

    #[test]
    fn lookup_and_excluding_ignores_excluded_node() {
        let (g, a, b, _c, ab) = simple_graph();
        let state = DynamicStructuralHash::new(g).unwrap();
        let mut excluded = BTreeSet::new();

        assert_eq!(state.lookup_and_excluding(a, b, &excluded).unwrap(), ab);
        excluded.insert(ab.node);
        assert!(state.lookup_and_excluding(a, b, &excluded).is_none());
        state.check_invariants().unwrap();
    }

    #[test]
    fn add_and_does_not_apply_or_absorption() {
        let mut gb = GateBuilder::new("or_no_absorb".to_string(), GateBuilderOptions::no_opt());
        let a = *gb.add_input("a".to_string(), 1).get_lsb(0);
        let b = *gb.add_input("b".to_string(), 1).get_lsb(0);
        let a_or_b = gb.add_or_binary(a, b);
        gb.add_output("o".to_string(), AigBitVector::from_bit(a_or_b));

        let mut state = DynamicStructuralHash::new(gb.build()).unwrap();
        let live_ands_before = state.live_and_count();
        let a_or_b_and_b = state.add_and(a_or_b, b).unwrap();
        assert_ne!(a_or_b_and_b, b);
        assert_eq!(state.live_and_count(), live_ands_before + 1);
        assert_eq!(state.lookup_and(a_or_b, b).unwrap(), a_or_b_and_b);
        state.check_invariants().unwrap();
    }

    #[test]
    fn add_and_does_not_fold_constants() {
        let (g, a, _b, _c, _ab) = simple_graph();
        let mut state = DynamicStructuralHash::new(g).unwrap();
        let false_op = AigOperand {
            node: AigRef { id: 0 },
            negated: false,
        };
        let true_op = false_op.negate();
        let live_ands_before = state.live_and_count();

        let false_and_a = state.add_and(false_op, a).unwrap();
        assert_ne!(false_and_a, false_op);
        assert_eq!(state.live_and_count(), live_ands_before + 1);
        assert_eq!(state.lookup_and(false_op, a).unwrap(), false_and_a);

        let true_and_a = state.add_and(true_op, a).unwrap();
        assert_ne!(true_and_a, a);
        assert_eq!(state.live_and_count(), live_ands_before + 2);
        assert_eq!(state.lookup_and(true_op, a).unwrap(), true_and_a);
        state.check_invariants().unwrap();
    }

    #[test]
    fn use_count_tracks_duplicate_fanin_edges() {
        let mut gb = GateBuilder::new(
            "dup_fanin_use_count".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let a = *gb.add_input("a".to_string(), 1).get_lsb(0);
        let aa = gb.add_and_binary(a, a);
        gb.add_output("o".to_string(), AigBitVector::from_bit(aa));

        let mut state = DynamicStructuralHash::new(gb.build()).unwrap();
        assert_eq!(state.fanout_count(a.node), 1);
        assert_eq!(state.output_use_count(a.node), 0);
        assert_eq!(state.use_count(a.node), 2);
        assert_eq!(state.output_use_count(aa.node), 1);
        assert_eq!(state.use_count(aa.node), 1);

        state
            .move_edge(
                EdgeRef::OutputBit {
                    output_index: 0,
                    bit_index: 0,
                },
                a,
            )
            .unwrap();
        assert_eq!(state.use_count(a.node), 3);
        assert_eq!(state.use_count(aa.node), 0);
        state.delete_node(aa.node).unwrap();
        assert_eq!(state.use_count(a.node), 1);
        state.check_invariants().unwrap();
    }

    #[test]
    fn move_output_then_delete_dangling_node() {
        let (g, a, b, c, ab) = simple_graph();
        let mut state = DynamicStructuralHash::new(g).unwrap();

        state.move_fanin_edge(ab.node, 1, c).unwrap();
        assert!(state.lookup_and(a, b).is_none());
        assert_eq!(state.lookup_and(a, c).unwrap().node, ab.node);
        state.check_invariants().unwrap();

        state
            .move_edge(
                EdgeRef::OutputBit {
                    output_index: 0,
                    bit_index: 0,
                },
                a,
            )
            .unwrap();
        state.delete_node(ab.node).unwrap();
        assert!(!state.is_live(ab.node));
        assert!(state.lookup_and(a, c).is_none());
        state.check_invariants().unwrap();
    }

    #[test]
    fn delete_node_recursively_deletes_dangling_mffc() {
        let mut gb = GateBuilder::new("delete_mffc".to_string(), GateBuilderOptions::no_opt());
        let a = *gb.add_input("a".to_string(), 1).get_lsb(0);
        let b = *gb.add_input("b".to_string(), 1).get_lsb(0);
        let c = *gb.add_input("c".to_string(), 1).get_lsb(0);
        let ab = gb.add_and_binary(a, b);
        let abc = gb.add_and_binary(ab, c);
        gb.add_output("o".to_string(), AigBitVector::from_bit(abc));

        let mut state = DynamicStructuralHash::new(gb.build()).unwrap();
        state
            .move_edge(
                EdgeRef::OutputBit {
                    output_index: 0,
                    bit_index: 0,
                },
                a,
            )
            .unwrap();
        state.delete_node(abc.node).unwrap();

        assert!(!state.is_live(abc.node));
        assert!(!state.is_live(ab.node));
        assert!(state.lookup_and(a, b).is_none());
        state.check_invariants().unwrap();
    }

    #[test]
    fn delete_node_keeps_shared_fanin_mffc_live() {
        let mut gb = GateBuilder::new("delete_shared".to_string(), GateBuilderOptions::no_opt());
        let a = *gb.add_input("a".to_string(), 1).get_lsb(0);
        let b = *gb.add_input("b".to_string(), 1).get_lsb(0);
        let c = *gb.add_input("c".to_string(), 1).get_lsb(0);
        let d = *gb.add_input("d".to_string(), 1).get_lsb(0);
        let ab = gb.add_and_binary(a, b);
        let abc = gb.add_and_binary(ab, c);
        let abd = gb.add_and_binary(ab, d);
        gb.add_output("o0".to_string(), AigBitVector::from_bit(abc));
        gb.add_output("o1".to_string(), AigBitVector::from_bit(abd));

        let mut state = DynamicStructuralHash::new(gb.build()).unwrap();
        state
            .move_edge(
                EdgeRef::OutputBit {
                    output_index: 0,
                    bit_index: 0,
                },
                a,
            )
            .unwrap();
        state.delete_node(abc.node).unwrap();

        assert!(!state.is_live(abc.node));
        assert!(state.is_live(ab.node));
        assert!(state.is_live(abd.node));
        assert_eq!(state.lookup_and(a, b).unwrap(), ab);
        state.check_invariants().unwrap();
    }

    #[test]
    fn move_fanin_merges_local_duplicate_and_cascades() {
        let mut gb = GateBuilder::new("cascade".to_string(), GateBuilderOptions::no_opt());
        let a = *gb.add_input("a".to_string(), 1).get_lsb(0);
        let b = *gb.add_input("b".to_string(), 1).get_lsb(0);
        let c = *gb.add_input("c".to_string(), 1).get_lsb(0);
        let ab = gb.add_and_binary(a, b);
        let ac = gb.add_and_binary(a, c);
        let ac_b = gb.add_and_binary(ac, b);
        let ab_b = gb.add_and_binary(ab, b);
        gb.add_output("o".to_string(), AigBitVector::from_bit(ac_b));

        let mut state = DynamicStructuralHash::new(gb.build()).unwrap();
        state.move_fanin_edge(ac.node, 1, b).unwrap();

        assert!(state.is_live(ab.node));
        assert!(!state.is_live(ac.node));
        assert!(!state.is_live(ac_b.node));
        assert!(state.is_live(ab_b.node));
        assert_eq!(state.lookup_and(a, b).unwrap(), ab);
        assert_eq!(state.lookup_and(ab, b).unwrap(), ab_b);
        assert_eq!(
            *state.gate_fn().outputs[0].bit_vector.get_lsb(0),
            ab_b,
            "output should be redirected through the cascade"
        );
        state.check_invariants().unwrap();
    }

    #[test]
    fn replace_repairs_duplicate_merge_when_representative_becomes_dangling() {
        let mut gb = GateBuilder::new(
            "stale_duplicate_representative".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let a = *gb.add_input("a".to_string(), 1).get_lsb(0);
        let b = *gb.add_input("b".to_string(), 1).get_lsb(0);
        let c = *gb.add_input("c".to_string(), 1).get_lsb(0);
        let ab = gb.add_and_binary(a, b);
        let old = gb.add_and_binary(ab, c);
        let fanout = gb.add_and_binary(old, b);
        gb.add_output("o".to_string(), AigBitVector::from_bit(fanout));

        let mut state = DynamicStructuralHash::new(gb.build()).unwrap();
        state.replace_node_with_operand(old.node, a).unwrap();

        assert!(!state.is_live(old.node));
        assert!(!state.is_live(ab.node));
        assert!(state.is_live(fanout.node));
        assert_eq!(state.lookup_and(a, b).unwrap().node, fanout.node);
        assert_eq!(
            *state.gate_fn().outputs[0].bit_vector.get_lsb(0),
            fanout,
            "fanout should remain the output after its duplicate representative dies"
        );
        state.check_invariants().unwrap();
    }
}
