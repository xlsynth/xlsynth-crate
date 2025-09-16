// SPDX-License-Identifier: Apache-2.0

//! Greedy edit-distance computation between two XLS IR functions.
//! Contains the matching machinery and conversion of matches into concrete
//! edits.

use crate::graph_edit::{IrEdit, IrEditSet};
use crate::ir::{Fn, Node, NodeRef};
use crate::ir_utils::{operands, remap_payload_with};
use crate::node_hashing::FwdHash;
use crate::node_hashing::compute_node_local_structural_hash;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

/// Represents an edit to transform one IR function into another.
///
/// This is a placeholder skeleton; variants will be expanded in future work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchAction {
    /// Delete a node present in the old function by index in `old_fn.nodes`.
    DeleteNode { old_index: usize },
    /// Add a node present in the new function by index in `new_fn.nodes`.
    AddNode {
        new_index: usize,
        is_new_return: bool,
    },
    /// Match an old node to a new node, with optional operand substitutions.
    ///
    /// Operand substitutions specify how operands of the old node map to
    /// operands of the new node. Each pair is (old_operand_index,
    /// new_operand_index), both indices refer to node indices in their
    /// respective functions.
    MatchNodes {
        old_index: usize,
        new_index: usize,
        operand_substitutions: Vec<(usize, usize)>,
        is_new_return: bool,
    },
}

/// A collection of match decisions produced by the matcher.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IrMatchSet {
    pub matches: Vec<MatchAction>,
}

/// Dependency information for a single node.
#[derive(Debug, Clone)]
pub struct DepNode {
    pub operands: Vec<usize>,       // deps (by node index)
    pub users: Vec<(usize, usize)>, // (user index, operand slot)
}

/// Dependency graph for a function: collection of per-node dependency info.
#[derive(Debug, Clone)]
pub struct DepGraph {
    pub nodes: Vec<DepNode>,
}

/// Builds a dependency graph for the given function.
pub fn build_dependency_graph(f: &Fn) -> DepGraph {
    let n = f.nodes.len();
    let mut operands_list: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, node) in f.nodes.iter().enumerate() {
        operands_list[i] = operands(&node.payload)
            .into_iter()
            .map(|r| r.index)
            .collect();
    }
    let mut users_list: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for (idx, ops) in operands_list.iter().enumerate() {
        for (slot, &d) in ops.iter().enumerate() {
            users_list[d].push((idx, slot));
        }
    }
    let mut nodes: Vec<DepNode> = Vec::with_capacity(n);
    for i in 0..n {
        nodes.push(DepNode {
            operands: operands_list[i].clone(),
            users: users_list[i].clone(),
        });
    }
    DepGraph { nodes }
}

/// Identifies which function a ready node belongs to when planning edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeSide {
    Old,
    New,
}

/// Describes a node that is ready (all dependencies satisfied) in either the
/// old or new function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadyNode {
    pub side: NodeSide,
    pub index: usize,
}

// compute_parameter_matches moved to graph_edit.rs

/// Reverse-direction match score between an old and new node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReverseMatch {
    pub old_index: usize,
    pub new_index: usize,
    pub score: f64,
}

/// Abstraction for prioritizing and selecting edits.
///
/// Implementations maintain internal state (e.g., priority queues) and decide
/// which `MatchAction` should be applied next from the set of nodes reported
/// ready via `add_ready_node`.
pub trait MatchSelector {
    /// Adds a node that has become ready to be handled.
    fn add_ready_node(&mut self, node: ReadyNode);

    /// Selects the next match action to apply, or `None` if none remain.
    fn select_next_match(&mut self) -> Option<MatchAction>;

    /// Notifies the selector that a match action has been observed/applied so
    /// it can update any internal state (e.g., statistics, heuristics).
    fn update_after_match(&mut self, _edit: &MatchAction) {}
}

// NaiveMatchSelector moved to graph_edit.rs

/// Greedy selector that pre-scores candidate matches using a reverse traversal
/// and then performs forward-direction matching guided by those scores.
pub struct GreedyMatchSelector<'a> {
    old: &'a Fn,
    new: &'a Fn,
    /// Per-node hash-correspondence indices.
    old_to_new_by_hash: HashMap<usize, Vec<usize>>,
    new_to_old_by_hash: HashMap<usize, Vec<usize>>,
    /// Reverse-direction similarity scores for candidate node pairs.
    reverse_scores: HashMap<(usize, usize), f64>,
    /// Ready sets for quick membership and stable iteration by node index.
    ready_old: BTreeSet<usize>,
    ready_new: BTreeSet<usize>,
    /// Mapping from matched old node index to new node index.
    matched_old_to_new: HashMap<usize, usize>,
    /// Nodes that have been handled (matched or added/removed) in old/new.
    handled_old: BTreeSet<usize>,
    handled_new: BTreeSet<usize>,
    /// Pending actions to emit (e.g., seeded parameter matches) in FIFO order.
    pending_actions: VecDeque<MatchAction>,
}

impl<'a> GreedyMatchSelector<'a> {
    pub fn new(old: &'a Fn, new: &'a Fn) -> Self {
        // Precompute local structural hashes and build per-node correspondence maps.
        let mut old_hashes: Vec<FwdHash> = Vec::with_capacity(old.nodes.len());
        for i in 0..old.nodes.len() {
            old_hashes.push(compute_node_local_structural_hash(
                old,
                NodeRef { index: i },
            ));
        }
        let mut new_hashes: Vec<FwdHash> = Vec::with_capacity(new.nodes.len());
        for i in 0..new.nodes.len() {
            new_hashes.push(compute_node_local_structural_hash(
                new,
                NodeRef { index: i },
            ));
        }
        let mut hash_to_old: HashMap<FwdHash, Vec<usize>> = HashMap::new();
        for (idx, h) in old_hashes.iter().enumerate() {
            hash_to_old.entry(*h).or_default().push(idx);
        }
        let mut hash_to_new: HashMap<FwdHash, Vec<usize>> = HashMap::new();
        for (idx, h) in new_hashes.iter().enumerate() {
            hash_to_new.entry(*h).or_default().push(idx);
        }
        let mut old_to_new_by_hash: HashMap<usize, Vec<usize>> = HashMap::new();
        for (oi, h) in old_hashes.iter().enumerate() {
            if let Some(list) = hash_to_new.get(h) {
                old_to_new_by_hash.insert(oi, list.clone());
            } else {
                old_to_new_by_hash.insert(oi, Vec::new());
            }
        }
        let mut new_to_old_by_hash: HashMap<usize, Vec<usize>> = HashMap::new();
        for (ni, h) in new_hashes.iter().enumerate() {
            if let Some(list) = hash_to_old.get(h) {
                new_to_old_by_hash.insert(ni, list.clone());
            } else {
                new_to_old_by_hash.insert(ni, Vec::new());
            }
        }
        let reverse_scores = compute_reverse_matches(old, new);
        let mut sel = Self {
            old,
            new,
            old_to_new_by_hash,
            new_to_old_by_hash,
            reverse_scores,
            ready_old: BTreeSet::new(),
            ready_new: BTreeSet::new(),
            matched_old_to_new: HashMap::new(),
            handled_old: BTreeSet::new(),
            handled_new: BTreeSet::new(),
            pending_actions: VecDeque::new(),
        };
        // Seed parameter matches into the pending queue.
        for m in crate::graph_edit::compute_parameter_matches(old, new).into_iter() {
            sel.pending_actions.push_back(m);
        }
        sel
    }

    /// Computes forward-direction similarity score based on operand matches.
    /// Returns 0.0 unless the local shapes of (old_index, new_index) are equal.
    /// When equal, returns the fraction of operand positions that are matched
    /// according to `matched_old_to_new` (1.0 for a matched operand, 0.0
    /// otherwise), normalized by the number of operands (must be equal
    /// between nodes).
    pub fn forward_match_score(&self, old_index: usize, new_index: usize) -> f64 {
        let old_ops: Vec<usize> = operands(&self.old.nodes[old_index].payload)
            .into_iter()
            .map(|r| r.index)
            .collect();
        let new_ops: Vec<usize> = operands(&self.new.nodes[new_index].payload)
            .into_iter()
            .map(|r| r.index)
            .collect();

        assert!(
            old_ops.len() == new_ops.len(),
            "forward_match_score expects equal operand counts for shape-equal nodes"
        );
        if old_ops.is_empty() {
            return 1.0;
        }

        let mut matches = 0usize;
        for (op_old, op_new) in old_ops.iter().zip(new_ops.iter()) {
            if self
                .matched_old_to_new
                .get(op_old)
                .map_or(false, |&mapped| mapped == *op_new)
            {
                matches += 1;
            }
        }
        (matches as f64) / (old_ops.len() as f64)
    }

    /// Opportunity cost helper (generic over a score map):
    /// - If both a and b are Some, returns the max of alternative matches for a
    ///   (excluding b) and for b (excluding a).
    /// - If only a is Some, returns max score over all (a, b').
    /// - If only b is Some, returns max score over all (a', b).
    /// At least one of a or b must be Some.
    fn opportunity_cost(
        &self,
        a: Option<usize>,
        b: Option<usize>,
        scores: &HashMap<(usize, usize), f64>,
        by_a: &HashMap<usize, Vec<usize>>,
        by_b: &HashMap<usize, Vec<usize>>,
    ) -> f64 {
        match (a, b) {
            (Some(a_idx), Some(b_idx)) => {
                let mut best = 0.0;
                if let Some(bs) = by_a.get(&a_idx) {
                    for &b2 in bs.iter() {
                        if b2 != b_idx {
                            let v = *scores.get(&(a_idx, b2)).unwrap_or(&0.0);
                            if v > best {
                                best = v;
                            }
                        }
                    }
                }
                if let Some(as_) = by_b.get(&b_idx) {
                    for &a2 in as_.iter() {
                        if a2 != a_idx {
                            let v = *scores.get(&(a2, b_idx)).unwrap_or(&0.0);
                            if v > best {
                                best = v;
                            }
                        }
                    }
                }
                best
            }
            (Some(a_idx), None) => by_a
                .get(&a_idx)
                .map(|bs| {
                    bs.iter()
                        .map(|&b2| *scores.get(&(a_idx, b2)).unwrap_or(&0.0))
                        .fold(0.0, f64::max)
                })
                .unwrap_or(0.0),
            (None, Some(b_idx)) => by_b
                .get(&b_idx)
                .map(|as_| {
                    as_.iter()
                        .map(|&a2| *scores.get(&(a2, b_idx)).unwrap_or(&0.0))
                        .fold(0.0, f64::max)
                })
                .unwrap_or(0.0),
            (None, None) => 0.0,
        }
    }
}

// Removed queue-based scheduling; heuristic selection is computed on demand.

impl<'a> MatchSelector for GreedyMatchSelector<'a> {
    fn add_ready_node(&mut self, node: ReadyNode) {
        match node.side {
            NodeSide::Old => {
                if matches!(
                    self.old.nodes[node.index].payload,
                    crate::ir::NodePayload::GetParam(_)
                ) {
                    return;
                }
                self.ready_old.insert(node.index);
            }
            NodeSide::New => {
                if matches!(
                    self.new.nodes[node.index].payload,
                    crate::ir::NodePayload::GetParam(_)
                ) {
                    return;
                }
                self.ready_new.insert(node.index);
            }
        }
    }

    fn select_next_match(&mut self) -> Option<MatchAction> {
        // Emit pending pre-seeded actions first (e.g., parameter matches).
        if let Some(a) = self.pending_actions.pop_front() {
            return Some(a);
        }

        // Build ready sets directly.
        let ready_old: Vec<usize> = self.ready_old.iter().copied().collect();
        let ready_new: Vec<usize> = self.ready_new.iter().copied().collect();

        if ready_old.is_empty() && ready_new.is_empty() {
            return None;
        }

        // Enumerate candidate same-shaped matches using hash correspondence.
        let mut candidates: Vec<(usize, usize)> = Vec::new();
        for &oa in ready_old.iter() {
            if let Some(news) = self.old_to_new_by_hash.get(&oa) {
                for &nb in news.iter() {
                    if self.ready_new.contains(&nb) {
                        candidates.push((oa, nb));
                    }
                }
            }
        }

        // Precompute forward and reverse scores for candidates.
        let mut mf: HashMap<(usize, usize), f64> = HashMap::new();
        let mut mr: HashMap<(usize, usize), f64> = HashMap::new();
        for &(a, b) in candidates.iter() {
            mf.insert((a, b), self.forward_match_score(a, b));
            mr.insert((a, b), *self.reverse_scores.get(&(a, b)).unwrap_or(&0.0));
        }

        let mf_of = |a: usize, b: usize, map: &HashMap<(usize, usize), f64>| -> f64 {
            *map.get(&(a, b)).unwrap_or(&0.0)
        };

        // Index candidates by a and by b.
        let mut by_a: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut by_b: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(a, b) in candidates.iter() {
            by_a.entry(a).or_default().push(b);
            by_b.entry(b).or_default().push(a);
        }

        // Opportunity cost for single-side actions.
        let mut ocf_a: HashMap<usize, f64> = HashMap::new();
        let mut ocr_a: HashMap<usize, f64> = HashMap::new();
        for &a in ready_old.iter() {
            ocf_a.insert(a, self.opportunity_cost(Some(a), None, &mf, &by_a, &by_b));
            ocr_a.insert(a, self.opportunity_cost(Some(a), None, &mr, &by_a, &by_b));
        }

        let mut ocf_b: HashMap<usize, f64> = HashMap::new();
        let mut ocr_b: HashMap<usize, f64> = HashMap::new();
        for &b in ready_new.iter() {
            ocf_b.insert(b, self.opportunity_cost(None, Some(b), &mf, &by_a, &by_b));
            ocr_b.insert(b, self.opportunity_cost(None, Some(b), &mr, &by_a, &by_b));
        }

        // Evaluate and pick best action.
        let mut best_score = f64::NEG_INFINITY;
        let mut best_action: Option<MatchAction> = None;

        // Match actions.
        for &(a, b) in candidates.iter() {
            let oc_f_pair = self.opportunity_cost(Some(a), Some(b), &mf, &by_a, &by_b);
            let oc_r_pair = self.opportunity_cost(Some(a), Some(b), &mr, &by_a, &by_b);

            let score = mf_of(a, b, &mf) + mf_of(a, b, &mr) - (oc_f_pair + oc_r_pair);
            if score > best_score {
                best_score = score;
                let is_ret = self
                    .new
                    .ret_node_ref
                    .map(|nr| nr.index)
                    .map_or(false, |ri| ri == b);
                best_action = Some(MatchAction::MatchNodes {
                    old_index: a,
                    new_index: b,
                    operand_substitutions: Vec::new(),
                    is_new_return: is_ret,
                });
            }
        }

        // Add actions for new ready nodes.
        for &b in ready_new.iter() {
            let oc = ocf_b.get(&b).copied().unwrap_or(0.0) + ocr_b.get(&b).copied().unwrap_or(0.0);
            let score = -oc;
            if score > best_score {
                best_score = score;
                let is_ret = self
                    .new
                    .ret_node_ref
                    .map(|nr| nr.index)
                    .map_or(false, |ri| ri == b);
                best_action = Some(MatchAction::AddNode {
                    new_index: b,
                    is_new_return: is_ret,
                });
            }
        }

        // Remove actions for old ready nodes.
        for &a in ready_old.iter() {
            let oc = ocf_a.get(&a).copied().unwrap_or(0.0) + ocr_a.get(&a).copied().unwrap_or(0.0);
            let score = -oc;
            if score > best_score {
                best_score = score;
                best_action = Some(MatchAction::DeleteNode { old_index: a });
            }
        }

        best_action
    }

    fn update_after_match(&mut self, edit: &MatchAction) {
        match edit {
            MatchAction::DeleteNode { old_index } => {
                self.ready_old.remove(old_index);
                self.handled_old.insert(*old_index);
            }
            MatchAction::AddNode { new_index, .. } => {
                self.ready_new.remove(new_index);
                self.handled_new.insert(*new_index);
            }
            MatchAction::MatchNodes {
                old_index,
                new_index,
                ..
            } => {
                self.ready_old.remove(old_index);
                self.ready_new.remove(new_index);
                self.matched_old_to_new.insert(*old_index, *new_index);
                self.handled_old.insert(*old_index);
                self.handled_new.insert(*new_index);
            }
        }
    }
}

/// Computes reverse-direction similarity scores M(A,B) for compatible node
/// pairs.
///
/// M(A,B) is 1.0 for sinks with no users and identical local shape.
/// Otherwise, for each user (u_a, i) of A, we take the maximum M(u_a, u_b)
/// over users (u_b, i) of B that are locally compatible; sum these per-user
/// scores and divide by max(|Users(A)|, |Users(B)|).
pub fn compute_reverse_matches(old: &Fn, new: &Fn) -> HashMap<(usize, usize), f64> {
    // Build dependency graphs for user/operand exploration.
    let old_graph = build_dependency_graph(old);
    let new_graph = build_dependency_graph(new);

    // Helper: local shape compatibility via local structural hash.
    let shapes_equal = |oi: usize, ni: usize| {
        compute_node_local_structural_hash(old, NodeRef { index: oi })
            == compute_node_local_structural_hash(new, NodeRef { index: ni })
    };

    // Scores map: best-known M(a,b) so far; initialize implicitly to 0.0 for all
    // pairs.
    let mut m_scores: HashMap<(usize, usize), f64> = HashMap::new();

    // Worklist seeded with the return-node pair if present and compatible.
    let mut worklist: VecDeque<(usize, usize)> = VecDeque::new();
    let mut on_worklist: HashSet<(usize, usize)> = HashSet::new();
    if let (Some(or), Some(nr)) = (old.ret_node_ref, new.ret_node_ref) {
        if shapes_equal(or.index, nr.index) {
            worklist.push_back((or.index, nr.index));
            on_worklist.insert((or.index, nr.index));
        }
    }

    // Recompute helper: compute M(a,b) from current scores of user pairs.
    let recompute_score = |a: usize, b: usize, m: &HashMap<(usize, usize), f64>| -> f64 {
        if !shapes_equal(a, b) {
            return 0.0;
        }
        let old_users = &old_graph.nodes[a].users;
        let new_users = &new_graph.nodes[b].users;
        let z = std::cmp::max(old_users.len(), new_users.len());
        if z == 0 {
            return 1.0;
        }
        let mut sum = 0.0f64;
        for &(u_a, slot) in old_users.iter() {
            let mut best = 0.0f64;
            for &(u_b, slot_b) in new_users.iter() {
                if slot_b != slot {
                    continue;
                }
                if !shapes_equal(u_a, u_b) {
                    continue;
                }
                if let Some(&val) = m.get(&(u_a, u_b)) {
                    if val > best {
                        best = val;
                    }
                }
            }
            sum += best;
        }
        sum / (z as f64)
    };

    // Process worklist: on improvement, propagate to operand pairs.
    while let Some((a, b)) = worklist.pop_front() {
        // Mark as no longer on the worklist so it can be re-enqueued upon future
        // improvements.
        on_worklist.remove(&(a, b));
        let new_score = recompute_score(a, b, &m_scores);
        let old_score = *m_scores.get(&(a, b)).unwrap_or(&0.0);
        if new_score > old_score {
            m_scores.insert((a, b), new_score);

            // Walk operands in lockstep; if counts match, enqueue compatible pairs.
            let a_ops: Vec<usize> = operands(&old.nodes[a].payload)
                .into_iter()
                .map(|r| r.index)
                .collect();
            let b_ops: Vec<usize> = operands(&new.nodes[b].payload)
                .into_iter()
                .map(|r| r.index)
                .collect();
            if a_ops.len() == b_ops.len() {
                for (op_a, op_b) in a_ops.into_iter().zip(b_ops.into_iter()) {
                    if shapes_equal(op_a, op_b) && !on_worklist.contains(&(op_a, op_b)) {
                        worklist.push_back((op_a, op_b));
                        on_worklist.insert((op_a, op_b));
                    }
                }
            }
        }
    }

    // Return map; optionally keep only non-zero entries to keep it compact.
    m_scores
}

/// Computes an edit set (distance) required to transform `old` into `new`.
/// Internally computes a match set, then converts matches to concrete edits.
pub fn compute_function_edit_distance(old: &Fn, new: &Fn) -> Result<IrEditSet, String> {
    crate::graph_edit::compute_function_edit_distance(old, new)
}

/// Computes the match actions required to transform `old` into `new`, using an
/// externally provided selector that controls priority and edit choice.
pub fn compute_function_edit<S: MatchSelector>(
    old: &Fn,
    new: &Fn,
    selector: &mut S,
) -> Result<IrMatchSet, String> {
    if old.get_type() != new.get_type() {
        return Err(format!(
            "Signature mismatch: old {} vs new {}",
            format_function_type(old),
            format_function_type(new)
        ));
    }

    // Verify parameter names match (order-sensitive) to avoid ambiguous operand
    // mapping.
    let old_param_names: Vec<&str> = old.params.iter().map(|p| p.name.as_str()).collect();
    let new_param_names: Vec<&str> = new.params.iter().map(|p| p.name.as_str()).collect();
    if old_param_names != new_param_names {
        return Err(format!(
            "Parameter names mismatch: old [{:?}] vs new [{:?}]",
            old_param_names, new_param_names
        ));
    }

    // Build dependency graphs for readiness tracking.
    let old_graph = build_dependency_graph(old);
    let new_graph = build_dependency_graph(new);

    // Remaining unhandled dependency counts per node.
    let mut old_remain: Vec<usize> = old_graph.nodes.iter().map(|n| n.operands.len()).collect();
    let mut new_remain: Vec<usize> = new_graph.nodes.iter().map(|n| n.operands.len()).collect();

    // (defined after seeding below)

    // Seed selector with all nodes that have no operands in each graph.
    for (i, &r) in old_remain.iter().enumerate() {
        if r == 0 {
            selector.add_ready_node(ReadyNode {
                side: NodeSide::Old,
                index: i,
            });
        }
    }
    for (i, &r) in new_remain.iter().enumerate() {
        if r == 0 {
            selector.add_ready_node(ReadyNode {
                side: NodeSide::New,
                index: i,
            });
        }
    }

    // Helper: decrement users' remaining counts and enqueue when they become ready.
    let mut update_ready = |idx: usize, side: NodeSide, selector: &mut S| match side {
        NodeSide::Old => {
            for &(user, _slot) in old_graph.nodes[idx].users.iter() {
                if old_remain[user] > 0 {
                    old_remain[user] -= 1;
                    if old_remain[user] == 0 {
                        selector.add_ready_node(ReadyNode { side, index: user });
                    }
                }
            }
        }
        NodeSide::New => {
            for &(user, _slot) in new_graph.nodes[idx].users.iter() {
                if new_remain[user] > 0 {
                    new_remain[user] -= 1;
                    if new_remain[user] == 0 {
                        selector.add_ready_node(ReadyNode { side, index: user });
                    }
                }
            }
        }
    };

    let mut matches: Vec<MatchAction> = Vec::new();
    while let Some(edit) = selector.select_next_match() {
        match edit.clone() {
            MatchAction::DeleteNode { old_index: index } => {
                matches.push(edit.clone());
                update_ready(index, NodeSide::Old, selector);
                selector.update_after_match(&edit);
            }
            MatchAction::AddNode {
                new_index: index, ..
            } => {
                let is_ret = new
                    .ret_node_ref
                    .map(|nr| nr.index)
                    .map_or(false, |ri| ri == index);
                matches.push(MatchAction::AddNode {
                    new_index: index,
                    is_new_return: is_ret,
                });
                update_ready(index, NodeSide::New, selector);
                selector.update_after_match(&edit);
            }
            MatchAction::MatchNodes {
                old_index,
                new_index,
                operand_substitutions,
                ..
            } => {
                // Record match edit.
                matches.push(MatchAction::MatchNodes {
                    old_index,
                    new_index,
                    operand_substitutions: operand_substitutions.clone(),
                    is_new_return: new.ret_node_ref.map(|nr| nr.index) == Some(new_index),
                });
                // Propagate readiness in old/new graphs.
                update_ready(old_index, NodeSide::Old, selector);
                update_ready(new_index, NodeSide::New, selector);
                selector.update_after_match(&edit);
            }
        }
    }

    Ok(IrMatchSet { matches })
}

/// Converts a set of `MatchAction`s into concrete `IrEdit`s, using `old` and
/// `new`.
pub fn convert_match_set_to_edit_set(old: &Fn, new: &Fn, m: &IrMatchSet) -> IrEditSet {
    // Build cross-reference maps between matched nodes for operand redirection.
    let mut new_to_old: HashMap<usize, usize> = HashMap::new();
    for action in m.matches.iter() {
        if let MatchAction::MatchNodes {
            old_index,
            new_index,
            operand_substitutions: _,
            is_new_return: _,
        } = action
        {
            new_to_old.insert(*new_index, *old_index);
        }
    }

    let mut edits: Vec<IrEdit> = Vec::new();
    for action in m.matches.iter() {
        match action {
            MatchAction::AddNode {
                new_index,
                is_new_return: _,
            } => {
                // Clone new node and remap operands that refer to matched new nodes to their
                // corresponding old indices. References to other new nodes remain as new
                // indices and are resolved during application as those nodes
                // are added earlier in order.
                let src = &new.nodes[*new_index];
                let remapped_payload = remap_payload_with(&src.payload, |nr: NodeRef| {
                    if let Some(&old_idx) = new_to_old.get(&nr.index) {
                        NodeRef { index: old_idx }
                    } else {
                        // Leave as original-new index; apply will resolve via new_to_patched.
                        nr
                    }
                });
                let cloned = Node {
                    text_id: src.text_id,
                    name: src.name.clone(),
                    ty: src.ty.clone(),
                    payload: remapped_payload,
                    pos: src.pos.clone(),
                };
                edits.push(IrEdit::AddNode {
                    new_index: *new_index,
                    node: cloned,
                });
            }
            MatchAction::DeleteNode { old_index } => {
                edits.push(IrEdit::DeleteNode { index: *old_index });
            }
            MatchAction::MatchNodes {
                old_index,
                new_index: _,
                operand_substitutions,
                is_new_return: _,
            } => {
                // For each substitution (old_operand_idx -> new_operand_idx), redirect
                // the operand slot(s) on the old user node to the mapped old target if known.
                for (old_operand_idx, new_operand_idx) in operand_substitutions.iter() {
                    if let Some(&mapped_old_target) = new_to_old.get(new_operand_idx) {
                        let user_node = &old.nodes[*old_index];
                        let user_operands = operands(&user_node.payload);
                        for (slot, nr) in user_operands.iter().enumerate() {
                            if nr.index == *old_operand_idx {
                                edits.push(IrEdit::SubstituteOperand {
                                    user_index: *old_index,
                                    operand_slot: slot,
                                    new_target_index: mapped_old_target,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    // Ensure the function return matches `new`.
    if let Some(nr) = new.ret_node_ref {
        if let Some(&old_idx) = new_to_old.get(&nr.index) {
            edits.push(IrEdit::SetReturn {
                index: old_idx,
                is_new: false,
            });
        } else {
            edits.push(IrEdit::SetReturn {
                index: nr.index,
                is_new: true,
            });
        }
    }
    IrEditSet { edits }
}

fn format_function_type(f: &Fn) -> String {
    let params = f
        .params
        .iter()
        .map(|p| format!("{}", p.ty))
        .collect::<Vec<String>>()
        .join(", ");
    format!("fn({}) -> {}", params, f.ret_ty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_edit::apply_function_edits;
    use crate::ir_parser::Parser;

    fn parse_ir_from_string(s: &str) -> crate::ir::Package {
        let mut parser = Parser::new(s);
        parser.parse_and_validate_package().unwrap()
    }

    #[test]
    fn compute_returns_adds_and_deletes_for_all_nodes() {
        let pkg = parse_ir_from_string(
            r#"package p
            top fn f(x: bits[8]) -> bits[8] {
                ret identity.2: bits[8] = identity(x, id=2)
            }
            "#,
        );
        let lhs = pkg.get_top().unwrap();

        let pkg2 = parse_ir_from_string(
            r#"package p2
            top fn f2(x: bits[8]) -> bits[8] {
                ret identity.2: bits[8] = identity(x, id=2)
            }
            "#,
        );
        let rhs = pkg2.get_top().unwrap();

        let edits = compute_function_edit_distance(lhs, rhs).unwrap();
        assert!(!edits.edits.is_empty());
        let patched = apply_function_edits(lhs, &edits).unwrap();
        assert!(crate::ir_isomorphism::is_ir_isomorphic(&patched, rhs));
    }

    #[test]
    fn compute_errors_on_mismatched_signatures() {
        let pkg = parse_ir_from_string(
            r#"package p
            top fn f(x: bits[8]) -> bits[8] {
                ret identity.2: bits[8] = identity(x, id=2)
            }
            "#,
        );
        let lhs = pkg.get_top().unwrap();

        let pkg2 = parse_ir_from_string(
            r#"package p2
            top fn f2(x: bits[9]) -> bits[9] {
                ret identity.2: bits[9] = identity(x, id=2)
            }
            "#,
        );
        let rhs = pkg2.get_top().unwrap();

        let result = compute_function_edit_distance(lhs, rhs);
        assert!(result.is_err());
    }

    #[test]
    fn apply_edits_literal_change_matches_new() {
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f() -> bits[8] {
                ret literal.1: bits[8] = literal(value=0, id=1)
            }
            "#,
        );
        let old_fn = pkg_old.get_top().unwrap();

        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f() -> bits[8] {
                ret literal.101: bits[8] = literal(value=1, id=101)
            }
            "#,
        );
        let new_fn = pkg_new.get_top().unwrap();

        let edits = compute_function_edit_distance(old_fn, new_fn).unwrap();
        let patched = apply_function_edits(old_fn, &edits).unwrap();
        assert!(crate::ir_isomorphism::is_ir_isomorphic(&patched, new_fn));
    }

    #[test]
    fn apply_edits_add_identity_of_literal() {
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f() -> bits[1] {
                ret literal.1: bits[1] = literal(value=0, id=1)
            }
            "#,
        );
        let old_fn = pkg_old.get_top().unwrap();

        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f() -> bits[1] {
                literal.11: bits[1] = literal(value=1, id=11)
                ret identity.22: bits[1] = identity(literal.11, id=22)
            }
            "#,
        );
        let new_fn = pkg_new.get_top().unwrap();

        let edits = compute_function_edit_distance(old_fn, new_fn).unwrap();
        let patched = apply_function_edits(old_fn, &edits).unwrap();
        assert!(crate::ir_isomorphism::is_ir_isomorphic(&patched, new_fn));
    }

    #[test]
    fn edit_distance_handles_different_return_nodes() {
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f() -> bits[1] {
                literal.1: bits[1] = literal(value=1, id=1)
                ret identity.2: bits[1] = identity(literal.1, id=2)
            }
            "#,
        );
        let old_fn = pkg_old.get_top().unwrap();

        // Same nodes, but return points to a different node (the literal).
        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f() -> bits[1] {
                ret literal.1: bits[1] = literal(value=1, id=1)
                identity.2: bits[1] = identity(literal.1, id=2)
            }
            "#,
        );
        let new_fn = pkg_new.get_top().unwrap();

        let edits = compute_function_edit_distance(old_fn, new_fn).unwrap();
        let patched = apply_function_edits(old_fn, &edits).unwrap();
        assert!(crate::ir_isomorphism::is_ir_isomorphic(&patched, new_fn));
    }

    #[test]
    fn edit_distance_multiple_parameters_identity_wrap() {
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
                ret add.3: bits[8] = add(a, b, id=3)
            }
            "#,
        );
        let old_fn = pkg_old.get_top().unwrap();

        // Different structure: wrap the add in an identity, keep params identical.
        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
                add.103: bits[8] = add(a, b, id=103)
                ret identity.104: bits[8] = identity(add.103, id=104)
            }
            "#,
        );
        let new_fn = pkg_new.get_top().unwrap();

        let edits = compute_function_edit_distance(old_fn, new_fn).unwrap();
        let patched = apply_function_edits(old_fn, &edits).unwrap();
        assert!(crate::ir_isomorphism::is_ir_isomorphic(&patched, new_fn));
    }
}
