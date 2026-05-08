// SPDX-License-Identifier: Apache-2.0

//! Exact forward and backward depth side state for a dynamic AIG.

use std::collections::VecDeque;

use crate::aig::dynamic_structural_hash::DynamicStructuralHash;
use crate::aig::gate::{AigNode, AigOperand, AigRef, GateFn};

/// Depth side state indexed over a `DynamicStructuralHash` graph.
///
/// This type deliberately does not own or edit the graph. Callers mutate the
/// `DynamicStructuralHash`, then pass a reference here to rebuild or validate
/// depth state. Backward depths are output-distance values; callers own any
/// timing cap or slack policy layered on top.
#[derive(Debug, Clone)]
pub struct DynamicDepthState {
    forward_depths: Vec<usize>,
    backward_depths: Vec<usize>,
    forward_queue_marks: Vec<usize>,
    backward_queue_marks: Vec<usize>,
    queue_epoch: usize,
}

impl DynamicDepthState {
    /// Builds depth state over the live graph in `state`.
    pub fn new(state: &DynamicStructuralHash) -> Result<Self, String> {
        let (forward_depths, backward_depths) = compute_depths(state)?;
        Ok(Self {
            forward_depths,
            backward_depths,
            forward_queue_marks: vec![0; state.gate_fn().gates.len()],
            backward_queue_marks: vec![0; state.gate_fn().gates.len()],
            queue_epoch: 0,
        })
    }

    /// Rebuilds all depth side state from the live graph in `state`.
    pub fn refresh_depths(&mut self, state: &DynamicStructuralHash) -> Result<(), String> {
        let (forward_depths, backward_depths) = compute_depths(state)?;
        self.forward_depths = forward_depths;
        self.backward_depths = backward_depths;
        self.resize_queue_marks(state.gate_fn().gates.len());
        Ok(())
    }

    /// Incrementally refreshes depths reachable from a set of changed nodes.
    ///
    /// `changed_nodes` should include the full edited cut interface: cut
    /// leaves, replacement fragment nodes including reused strash
    /// representatives, and the replacement output representative. If old
    /// nodes were deleted, include them too so backward-depth decreases can
    /// propagate through their old fanins.
    pub fn refresh_from_changed_nodes(
        &mut self,
        state: &DynamicStructuralHash,
        changed_nodes: &[AigRef],
    ) -> Result<(), String> {
        let node_count = state.gate_fn().gates.len();
        self.forward_depths.resize(node_count, 0);
        self.backward_depths.resize(node_count, usize::MAX);
        self.resize_queue_marks(node_count);

        let mut forward_queue = VecDeque::new();
        let mut backward_queue = VecDeque::new();
        let queue_epoch = self.next_queue_epoch();

        for node in changed_nodes.iter().copied() {
            if node.id >= node_count {
                return Err(format!(
                    "changed node {:?} out of bounds for gate count {}",
                    node, node_count
                ));
            }
            enqueue_node(
                node,
                &mut forward_queue,
                &mut self.forward_queue_marks,
                queue_epoch,
            );
            enqueue_node(
                node,
                &mut backward_queue,
                &mut self.backward_queue_marks,
                queue_epoch,
            );
        }

        while let Some(node) = forward_queue.pop_front() {
            self.forward_queue_marks[node.id] = 0;
            let old_depth = self.forward_depths[node.id];
            let new_depth = self.compute_local_forward_depth(state, node)?;
            if old_depth == new_depth {
                continue;
            }
            self.forward_depths[node.id] = new_depth;
            for fanout in state.fanout_nodes(node) {
                enqueue_node(
                    fanout,
                    &mut forward_queue,
                    &mut self.forward_queue_marks,
                    queue_epoch,
                );
            }
            if !state.is_live(node) {
                for fanin in node_fanins(state.gate_fn(), node) {
                    enqueue_node(
                        fanin,
                        &mut forward_queue,
                        &mut self.forward_queue_marks,
                        queue_epoch,
                    );
                }
            }
        }

        while let Some(node) = backward_queue.pop_front() {
            self.backward_queue_marks[node.id] = 0;
            let old_depth = self.backward_depths[node.id];
            let new_depth = self.compute_local_backward_depth(state, node)?;
            if old_depth == new_depth {
                continue;
            }
            self.backward_depths[node.id] = new_depth;
            for fanin in node_fanins(state.gate_fn(), node) {
                enqueue_node(
                    fanin,
                    &mut backward_queue,
                    &mut self.backward_queue_marks,
                    queue_epoch,
                );
            }
        }

        Ok(())
    }

    /// Returns the current maximum forward depth among output operands.
    pub fn max_output_node_depth(&self, state: &DynamicStructuralHash) -> Result<usize, String> {
        let g = state.gate_fn();
        let live = state.live_mask();
        let mut max_depth = 0usize;
        for output in &g.outputs {
            for op in output.bit_vector.iter_lsb_to_msb() {
                validate_live_operand(g, live, *op)?;
                let depth = self
                    .forward_depths
                    .get(op.node.id)
                    .copied()
                    .ok_or_else(|| {
                        format!(
                            "output operand {:?} has no forward depth entry; depths len={}",
                            op,
                            self.forward_depths.len()
                        )
                    })?;
                max_depth = max_depth.max(depth);
            }
        }
        Ok(max_depth)
    }

    /// Returns the current forward depth of a live node.
    pub fn forward_depth(&self, state: &DynamicStructuralHash, node: AigRef) -> Option<usize> {
        if !state.is_live(node) {
            return None;
        }
        self.forward_depths.get(node.id).copied()
    }

    /// Returns dense forward depths indexed by `AigRef::id`.
    pub fn forward_depths(&self) -> &[usize] {
        &self.forward_depths
    }

    /// Returns the maximum distance from this live node to any output bit.
    pub fn backward_depth(&self, state: &DynamicStructuralHash, node: AigRef) -> Option<usize> {
        if !state.is_live(node) {
            return None;
        }
        let backward = self.backward_depths.get(node.id).copied()?;
        (backward != usize::MAX).then_some(backward)
    }

    /// Returns dense backward depths indexed by `AigRef::id`.
    ///
    /// Unconstrained nodes have `usize::MAX`.
    pub fn backward_depths(&self) -> &[usize] {
        &self.backward_depths
    }

    /// Returns true if this live node currently has no output/fanout timing
    /// constraint.
    pub fn is_timing_unconstrained(&self, state: &DynamicStructuralHash, node: AigRef) -> bool {
        state.is_live(node) && self.backward_depths.get(node.id).copied() == Some(usize::MAX)
    }

    /// Rebuilds a reference index from scratch and compares it with this state.
    pub fn check_invariants(&self, state: &DynamicStructuralHash) -> Result<(), String> {
        let (forward_depths, backward_depths) = compute_depths(state)?;
        if self.forward_depths != forward_depths {
            return Err(format!(
                "forward depth mismatch: current={:?} reference={:?}",
                self.forward_depths, forward_depths
            ));
        }
        if self.backward_depths != backward_depths {
            return Err(format!(
                "backward depth mismatch: current={:?} reference={:?}",
                self.backward_depths, backward_depths
            ));
        }
        Ok(())
    }

    fn compute_local_forward_depth(
        &self,
        state: &DynamicStructuralHash,
        node: AigRef,
    ) -> Result<usize, String> {
        let g = state.gate_fn();
        if node.id >= g.gates.len() {
            return Err(format!("node {:?} out of bounds", node));
        }
        if !state.is_live(node) {
            return Ok(0);
        }
        match g.gates[node.id] {
            AigNode::Input { .. } | AigNode::Literal { .. } => Ok(0),
            AigNode::And2 { a, b, .. } => {
                validate_live_operand(g, state.live_mask(), a)?;
                validate_live_operand(g, state.live_mask(), b)?;
                let a_depth = self
                    .forward_depths
                    .get(a.node.id)
                    .copied()
                    .ok_or_else(|| format!("fanin {:?} has no forward depth entry", a.node))?;
                let b_depth = self
                    .forward_depths
                    .get(b.node.id)
                    .copied()
                    .ok_or_else(|| format!("fanin {:?} has no forward depth entry", b.node))?;
                Ok(1 + a_depth.max(b_depth))
            }
        }
    }

    fn compute_local_backward_depth(
        &self,
        state: &DynamicStructuralHash,
        node: AigRef,
    ) -> Result<usize, String> {
        if node.id >= state.gate_fn().gates.len() {
            return Err(format!("node {:?} out of bounds", node));
        }
        if !state.is_live(node) {
            return Ok(usize::MAX);
        }

        let mut depth = if state.output_use_count(node) != 0 {
            0
        } else {
            usize::MAX
        };
        for fanout in state.fanout_nodes(node) {
            let fanout_depth = self
                .backward_depths
                .get(fanout.id)
                .copied()
                .ok_or_else(|| format!("fanout {:?} has no backward depth entry", fanout))?;
            if fanout_depth != usize::MAX {
                let candidate = fanout_depth.saturating_add(1);
                if depth == usize::MAX || candidate > depth {
                    depth = candidate;
                }
            }
        }
        Ok(depth)
    }

    fn resize_queue_marks(&mut self, node_count: usize) {
        self.forward_queue_marks.resize(node_count, 0);
        self.backward_queue_marks.resize(node_count, 0);
    }

    fn next_queue_epoch(&mut self) -> usize {
        self.queue_epoch = self.queue_epoch.wrapping_add(1);
        if self.queue_epoch == 0 {
            for mark in &mut self.forward_queue_marks {
                *mark = 0;
            }
            for mark in &mut self.backward_queue_marks {
                *mark = 0;
            }
            self.queue_epoch = 1;
        }
        self.queue_epoch
    }
}

fn enqueue_node(
    node: AigRef,
    queue: &mut VecDeque<AigRef>,
    queued: &mut [usize],
    queue_epoch: usize,
) {
    if queued[node.id] != queue_epoch {
        queued[node.id] = queue_epoch;
        queue.push_back(node);
    }
}

fn node_fanins(g: &GateFn, node: AigRef) -> Vec<AigRef> {
    if node.id >= g.gates.len() {
        return Vec::new();
    }
    match g.gates[node.id] {
        AigNode::And2 { a, b, .. } => vec![a.node, b.node],
        AigNode::Input { .. } | AigNode::Literal { .. } => Vec::new(),
    }
}

fn compute_depths(state: &DynamicStructuralHash) -> Result<(Vec<usize>, Vec<usize>), String> {
    let g = state.gate_fn();
    let live = state.live_mask();
    if live.len() != g.gates.len() {
        return Err(format!(
            "live bitset length {} does not match gate count {}",
            live.len(),
            g.gates.len()
        ));
    }

    let forward_depths = compute_forward_depths(g, live)?;

    let mut backward_depths = vec![usize::MAX; g.gates.len()];
    let mut worklist = VecDeque::new();
    for output in &g.outputs {
        for op in output.bit_vector.iter_lsb_to_msb() {
            validate_live_operand(g, live, *op)?;
            if relax_backward_depth(&mut backward_depths[op.node.id], 0) {
                worklist.push_back(op.node);
            }
        }
    }

    while let Some(node) = worklist.pop_front() {
        let backward = backward_depths[node.id];
        if backward == usize::MAX {
            continue;
        }
        let AigNode::And2 { a, b, .. } = g.gates[node.id] else {
            continue;
        };
        validate_live_operand(g, live, a)?;
        validate_live_operand(g, live, b)?;
        let child_backward = backward.saturating_add(1);
        for child in [a.node, b.node] {
            if relax_backward_depth(&mut backward_depths[child.id], child_backward) {
                worklist.push_back(child);
            }
        }
    }

    Ok((forward_depths, backward_depths))
}

fn compute_forward_depths(g: &GateFn, live: &[bool]) -> Result<Vec<usize>, String> {
    let mut forward_depths = vec![0usize; g.gates.len()];
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

        let fanout_depth = forward_depths[node.id]
            .checked_add(1)
            .ok_or_else(|| format!("forward depth overflow at {:?}", node))?;
        for fanout in &fanouts[node.id] {
            if fanout_depth > forward_depths[fanout.id] {
                forward_depths[fanout.id] = fanout_depth;
            }
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

    Ok(forward_depths)
}

fn relax_backward_depth(current: &mut usize, candidate: usize) -> bool {
    if *current == usize::MAX || candidate > *current {
        *current = candidate;
        true
    } else {
        false
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::AigBitVector;
    use crate::aig::dynamic_structural_hash::DynamicStructuralHash;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    fn simple_graph() -> (GateFn, AigOperand, AigOperand, AigOperand, AigOperand) {
        let mut gb = GateBuilder::new("dyn_depth".to_string(), GateBuilderOptions::no_opt());
        let a = *gb.add_input("a".to_string(), 1).get_lsb(0);
        let b = *gb.add_input("b".to_string(), 1).get_lsb(0);
        let c = *gb.add_input("c".to_string(), 1).get_lsb(0);
        let ab = gb.add_and_binary(a, b);
        gb.add_output("o".to_string(), AigBitVector::from_bit(ab));
        (gb.build(), a, b, c, ab)
    }

    fn deep_chain_graph(len: usize) -> (GateFn, AigOperand) {
        assert!(len != 0);
        let mut gb = GateBuilder::new("dyn_depth_deep".to_string(), GateBuilderOptions::no_opt());
        let a = *gb.add_input("a".to_string(), 1).get_lsb(0);
        let b = *gb.add_input("b".to_string(), 1).get_lsb(0);
        let mut acc = gb.add_and_binary(a, b);
        for _ in 1..len {
            acc = gb.add_and_binary(acc, b);
        }
        gb.add_output("o".to_string(), AigBitVector::from_bit(acc));
        (gb.build(), acc)
    }

    #[test]
    fn initial_state_tracks_forward_and_backward_depths() {
        let (g, a, b, _c, ab) = simple_graph();
        let hash = DynamicStructuralHash::new(g).unwrap();
        let state = DynamicDepthState::new(&hash).unwrap();

        assert_eq!(state.forward_depth(&hash, a.node), Some(0));
        assert_eq!(state.forward_depth(&hash, b.node), Some(0));
        assert_eq!(state.forward_depth(&hash, ab.node), Some(1));
        assert_eq!(state.max_output_node_depth(&hash).unwrap(), 1);
        assert_eq!(state.backward_depth(&hash, ab.node), Some(0));
        assert_eq!(state.backward_depth(&hash, a.node), Some(1));
        assert_eq!(state.backward_depth(&hash, b.node), Some(1));
        state.check_invariants(&hash).unwrap();
    }

    #[test]
    fn initial_state_handles_deep_chain_without_recursion() {
        const CHAIN_LEN: usize = 20_000;
        let (g, root) = deep_chain_graph(CHAIN_LEN);
        let hash = DynamicStructuralHash::new(g).unwrap();
        let state = DynamicDepthState::new(&hash).unwrap();

        assert_eq!(state.forward_depth(&hash, root.node), Some(CHAIN_LEN));
        assert_eq!(state.max_output_node_depth(&hash).unwrap(), CHAIN_LEN);
        state.check_invariants(&hash).unwrap();
    }

    #[test]
    fn move_fanin_updates_forward_and_backward_depths() {
        let (g, a, b, c, ab) = simple_graph();
        let mut hash = DynamicStructuralHash::new(g).unwrap();
        let mut state = DynamicDepthState::new(&hash).unwrap();

        hash.move_fanin_edge(ab.node, 1, c).unwrap();
        state
            .refresh_from_changed_nodes(&hash, &[ab.node, b.node, c.node])
            .unwrap();

        assert_eq!(state.forward_depth(&hash, ab.node), Some(1));
        assert_eq!(state.backward_depth(&hash, ab.node), Some(0));
        assert_eq!(state.backward_depth(&hash, a.node), Some(1));
        assert_eq!(state.backward_depth(&hash, c.node), Some(1));
        state.check_invariants(&hash).unwrap();
    }

    #[test]
    fn output_move_relaxes_old_cone_backward_depths() {
        let (g, _a, _b, c, ab) = simple_graph();
        let mut hash = DynamicStructuralHash::new(g).unwrap();
        let mut state = DynamicDepthState::new(&hash).unwrap();

        hash.move_output_edge(0, 0, c).unwrap();
        state
            .refresh_from_changed_nodes(&hash, &[ab.node, c.node])
            .unwrap();

        assert_eq!(state.max_output_node_depth(&hash).unwrap(), 0);
        assert_eq!(state.backward_depth(&hash, c.node), Some(0));
        assert!(state.is_timing_unconstrained(&hash, ab.node));
        state.check_invariants(&hash).unwrap();
    }

    #[test]
    fn replace_node_deletes_dangling_mffc_and_updates_depths() {
        let (g, a, b, c, ab) = simple_graph();
        let mut hash = DynamicStructuralHash::new(g).unwrap();
        let mut state = DynamicDepthState::new(&hash).unwrap();

        hash.replace_node_with_operand(ab.node, c).unwrap();
        state
            .refresh_from_changed_nodes(&hash, &[ab.node, a.node, b.node, c.node])
            .unwrap();

        assert!(!hash.is_live(ab.node));
        assert_eq!(state.max_output_node_depth(&hash).unwrap(), 0);
        assert_eq!(state.backward_depth(&hash, c.node), Some(0));
        state.check_invariants(&hash).unwrap();
    }

    #[test]
    fn add_and_tracks_unconstrained_new_node() {
        let (g, a, _b, c, _ab) = simple_graph();
        let mut hash = DynamicStructuralHash::new(g).unwrap();
        let mut state = DynamicDepthState::new(&hash).unwrap();

        let ac = hash.add_and(a, c).unwrap();
        state.refresh_from_changed_nodes(&hash, &[ac.node]).unwrap();

        assert_eq!(state.forward_depth(&hash, ac.node), Some(1));
        assert_eq!(state.backward_depth(&hash, ac.node), None);
        assert!(state.is_timing_unconstrained(&hash, ac.node));
        state.check_invariants(&hash).unwrap();
    }

    #[test]
    fn replacement_cut_leaves_update_shared_interior_backward_depths() {
        let mut gb = GateBuilder::new("shared_cut".to_string(), GateBuilderOptions::no_opt());
        let a = *gb.add_input("a".to_string(), 1).get_lsb(0);
        let b = *gb.add_input("b".to_string(), 1).get_lsb(0);
        let x = *gb.add_input("x".to_string(), 1).get_lsb(0);
        let y = *gb.add_input("y".to_string(), 1).get_lsb(0);
        let q = *gb.add_input("q".to_string(), 1).get_lsb(0);
        let u = gb.add_and_binary(a, b);
        let root = gb.add_and_binary(u, x);
        let root_user = gb.add_and_binary(root, q);
        let z = gb.add_and_binary(u, y);
        gb.add_output("root_user".to_string(), AigBitVector::from_bit(root_user));
        gb.add_output("z".to_string(), AigBitVector::from_bit(z));

        let mut hash = DynamicStructuralHash::new(gb.build()).unwrap();
        let mut state = DynamicDepthState::new(&hash).unwrap();
        assert_eq!(state.backward_depth(&hash, u.node), Some(2));

        hash.replace_node_with_operand(root.node, x).unwrap();
        state
            .refresh_from_changed_nodes(&hash, &[root.node, u.node, x.node, root_user.node])
            .unwrap();

        assert!(!hash.is_live(root.node));
        assert_eq!(state.backward_depth(&hash, u.node), Some(1));
        state.check_invariants(&hash).unwrap();
    }
}
