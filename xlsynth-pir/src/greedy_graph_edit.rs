// SPDX-License-Identifier: Apache-2.0

//! Greedy edit-distance computation between two XLS IR functions.
//! Contains the matching machinery and conversion of matches into concrete
//! edits.

use crate::graph_edit::{
    DepGraph, MatchAction, MatchSelector, NewNodeRef, NodeSide, OldNodeRef, ReadyNode,
    build_dependency_graph,
};
use crate::node_hashing::FwdHash;
use log::{Level, debug, log_enabled, trace};
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

// compute_parameter_matches moved to graph_edit.rs

/// Greedy selector that pre-scores candidate matches using a reverse traversal
/// and then performs forward-direction matching guided by those scores.
pub struct GreedyMatchSelector {
    old_graph: DepGraph<OldNodeRef>,
    new_graph: DepGraph<NewNodeRef>,
    /// Per-node hash-correspondence indices.
    old_to_new_by_hash: HashMap<OldNodeRef, Vec<NewNodeRef>>,
    new_to_old_by_hash: HashMap<NewNodeRef, Vec<OldNodeRef>>,
    /// Reverse-direction similarity scores for candidate node pairs.
    reverse_scores: HashMap<(OldNodeRef, NewNodeRef), i32>,
    ready_old: BTreeSet<OldNodeRef>,
    ready_new: BTreeSet<NewNodeRef>,
    /// Mapping from matched old node index to new node index.
    matched_old_to_new: HashMap<OldNodeRef, NewNodeRef>,
    /// Nodes that have been handled (matched or added/removed) in old/new.
    handled_old: BTreeSet<OldNodeRef>,
    handled_new: BTreeSet<NewNodeRef>,
}

impl GreedyMatchSelector {
    /// Score scaling factor used for integer-based similarity.
    const SCORE_DENOMINATOR: i32 = 1000;
    const PERFECT_MATCH_SCORE: i32 = Self::SCORE_DENOMINATOR * 100;

    /// Generic opportunity cost using a supplied match score lambda.
    fn best_unready_match_score(&self, a: Option<OldNodeRef>, b: Option<NewNodeRef>) -> i32 {
        match (a, b) {
            (Some(a_idx), Some(b_idx)) => {
                assert!(self.ready_old.contains(&a_idx));
                assert!(self.ready_new.contains(&b_idx));
                let mut best = 0i32;
                if let Some(bs) = self.old_to_new_by_hash.get(&a_idx) {
                    for &b2 in bs.iter() {
                        if b2 == b_idx
                            || self.ready_new.contains(&b2)
                            || self.handled_new.contains(&b2)
                        {
                            continue;
                        }
                        let v = self.match_score(&a_idx, &b2).1;
                        if v > best {
                            best = v;
                        }
                    }
                }
                if let Some(as_) = self.new_to_old_by_hash.get(&b_idx) {
                    for &a2 in as_.iter() {
                        if a2 == a_idx
                            || self.ready_old.contains(&a2)
                            || self.handled_old.contains(&a2)
                        {
                            continue;
                        }
                        let v = self.match_score(&a2, &b_idx).1;
                        if v > best {
                            best = v;
                        }
                    }
                }
                best
            }
            (Some(a_idx), None) => {
                assert!(self.ready_old.contains(&a_idx));
                let mut best = 0i32;
                if let Some(bs) = self.old_to_new_by_hash.get(&a_idx) {
                    for &b2 in bs.iter() {
                        if self.ready_new.contains(&b2) || self.handled_new.contains(&b2) {
                            continue;
                        }
                        let v = self.match_score(&a_idx, &b2).1;
                        if v > best {
                            best = v;
                        }
                    }
                }
                best
            }
            (None, Some(b_idx)) => {
                assert!(self.ready_new.contains(&b_idx));
                let mut best = 0i32;
                if let Some(as_) = self.new_to_old_by_hash.get(&b_idx) {
                    for &a2 in as_.iter() {
                        if self.ready_old.contains(&a2) || self.handled_old.contains(&a2) {
                            continue;
                        }
                        let v = self.match_score(&a2, &b_idx).1;
                        if v > best {
                            best = v;
                        }
                    }
                }
                best
            }
            (None, None) => unreachable!(
                "opportunity cost called with (None, None); at least one side must be Some"
            ),
        }
    }

    /// Score for matching a ready old node `a` with a ready new node `b`.
    /// score = forward + reverse - (opportunity_cost_forward +
    /// opportunity_cost_reverse)
    fn match_score(&self, a: &OldNodeRef, b: &NewNodeRef) -> (bool, i32) {
        let fwd = self.forward_match_score(*a, *b);
        let fwd = if fwd == Self::SCORE_DENOMINATOR {
            Self::PERFECT_MATCH_SCORE
        } else {
            fwd
        };
        let rev = *self.reverse_scores.get(&(*a, *b)).unwrap_or(&0);
        let rev = if rev == Self::SCORE_DENOMINATOR {
            Self::PERFECT_MATCH_SCORE
        } else {
            rev
        };
        let is_perfect_match = fwd == Self::PERFECT_MATCH_SCORE || rev == Self::PERFECT_MATCH_SCORE;
        (is_perfect_match, fwd + rev)
    }
    fn net_match_score(&self, a: &OldNodeRef, b: &NewNodeRef) -> (bool, i32) {
        let (is_perfect_match, score) = self.match_score(a, b);
        (
            is_perfect_match,
            score - self.best_unready_match_score(Some(*a), Some(*b)),
        )
    }

    /// Score for deleting a ready old node `a`.
    /// score = -(best_forward_alt_for_a + best_reverse_alt_for_a)
    fn delete_node_score(&self, _a: &OldNodeRef) -> i32 {
        0
    }
    fn net_delete_node_score(&self, a: &OldNodeRef) -> i32 {
        self.delete_node_score(a) - self.best_unready_match_score(Some(*a), None)
    }

    /// Score for adding a ready new node `b`.
    /// score = -(best_forward_alt_for_b + best_reverse_alt_for_b)
    fn add_node_score(&self, _b: &NewNodeRef) -> i32 {
        0
    }
    fn net_add_node_score(&self, b: &NewNodeRef) -> i32 {
        self.add_node_score(b) - self.best_unready_match_score(None, Some(*b))
    }

    /// Build a MatchNodes action for (a, b).
    fn build_match_action(&self, a: &OldNodeRef, b: &NewNodeRef) -> MatchAction {
        let new_operands = self
            .new_graph
            .get_node(*b)
            .operands
            .iter()
            .copied()
            .collect::<Vec<NewNodeRef>>();
        MatchAction::MatchNodes {
            old_index: *a,
            new_index: *b,
            new_operands,
            is_new_return: *b == self.new_graph.return_value,
        }
    }

    /// Build a DeleteNode action for old node `a`.
    fn build_delete_node_action(&self, a: &OldNodeRef) -> MatchAction {
        MatchAction::DeleteNode { old_index: *a }
    }

    /// Build an AddNode action for new node `b`.
    fn build_add_node_action(&self, b: &NewNodeRef) -> MatchAction {
        MatchAction::AddNode {
            new_index: *b,
            is_return: *b == self.new_graph.return_value,
        }
    }
    pub fn new(old: &crate::ir::Fn, new: &crate::ir::Fn) -> Self {
        let old_graph = build_dependency_graph::<OldNodeRef>(old);
        let new_graph = build_dependency_graph::<NewNodeRef>(new);
        // Build hash->indices maps directly from the graphs, then per-node maps.
        let mut hash_to_old: HashMap<FwdHash, Vec<OldNodeRef>> = HashMap::new();
        for (idx, node) in old_graph.nodes.iter().enumerate() {
            hash_to_old
                .entry(node.structural_hash)
                .or_default()
                .push(OldNodeRef(idx));
        }
        let mut hash_to_new: HashMap<FwdHash, Vec<NewNodeRef>> = HashMap::new();
        for (idx, node) in new_graph.nodes.iter().enumerate() {
            hash_to_new
                .entry(node.structural_hash)
                .or_default()
                .push(NewNodeRef(idx));
        }
        let mut old_to_new_by_hash: HashMap<OldNodeRef, Vec<NewNodeRef>> = HashMap::new();
        for (oi_us, node) in old_graph.nodes.iter().enumerate() {
            let oi = OldNodeRef(oi_us);
            let list = hash_to_new
                .get(&node.structural_hash)
                .cloned()
                .unwrap_or_default();
            old_to_new_by_hash.insert(oi, list);
        }
        let mut new_to_old_by_hash: HashMap<NewNodeRef, Vec<OldNodeRef>> = HashMap::new();
        for (ni_us, node) in new_graph.nodes.iter().enumerate() {
            let ni = NewNodeRef(ni_us);
            let list = hash_to_old
                .get(&node.structural_hash)
                .cloned()
                .unwrap_or_default();
            new_to_old_by_hash.insert(ni, list);
        }
        let reverse_scores = compute_reverse_matches(&old_graph, &new_graph);
        Self {
            old_graph,
            new_graph,
            old_to_new_by_hash,
            new_to_old_by_hash,
            reverse_scores,
            ready_old: BTreeSet::new(),
            ready_new: BTreeSet::new(),
            matched_old_to_new: HashMap::new(),
            handled_old: BTreeSet::new(),
            handled_new: BTreeSet::new(),
        }
    }

    /// Computes forward-direction similarity score based on operand matches.
    /// Returns 0 unless the local shapes of (old_index, new_index) are equal.
    /// When equal, returns SCORE_DENOMINATOR * (#matched_operands / #operands).
    pub fn forward_match_score(&self, old_index: OldNodeRef, new_index: NewNodeRef) -> i32 {
        let old_ops: Vec<usize> = self
            .old_graph
            .get_node(old_index)
            .operands
            .iter()
            .copied()
            .map(|i| usize::from(i))
            .collect();
        let new_ops: Vec<usize> = self
            .new_graph
            .get_node(new_index)
            .operands
            .iter()
            .copied()
            .map(|i| usize::from(i))
            .collect();

        assert!(
            old_ops.len() == new_ops.len(),
            "forward_match_score expects equal operand counts for shape-equal nodes"
        );
        if old_ops.is_empty() {
            return Self::SCORE_DENOMINATOR;
        }

        let mut matches = 0usize;
        for (op_old, op_new) in old_ops.iter().zip(new_ops.iter()) {
            if self
                .matched_old_to_new
                .get(&OldNodeRef(*op_old))
                .map_or(false, |&mapped| mapped == NewNodeRef(*op_new))
            {
                matches += 1;
            }
        }
        ((matches as i32) * Self::SCORE_DENOMINATOR) / (old_ops.len() as i32)
    }

    fn select_best_action(&mut self) -> Option<(i32, MatchAction)> {
        if self.ready_old.is_empty() && self.ready_new.is_empty() {
            return None;
        }

        let mut best_action: Option<MatchAction> = None;
        let mut best_score: i32 = i32::MIN;

        // 1) Consider match actions for each ready old against compatible ready new.
        for &a in self.ready_old.iter() {
            if let Some(news) = self.old_to_new_by_hash.get(&a) {
                for &b in news.iter() {
                    if !self.ready_new.contains(&b) {
                        continue;
                    }
                    let (is_perfect_match, score) = self.net_match_score(&a, &b);
                    trace!(
                        "Considering match action: {} <-> {}, score={}, is_perfect_match={}",
                        self.old_graph.get_node(a).name,
                        self.new_graph.get_node(b).name,
                        score,
                        is_perfect_match
                    );
                    // if is_perfect_match || score > best_score {
                    //     trace!("New best score: {}", score);
                    //     best_score = score;
                    //     best_action = Some(self.build_match_action(&a, &b));
                    //     if is_perfect_match {
                    //         return Some((score, best_action.unwrap()));
                    //     }
                    // }
                    if score > best_score {
                        trace!("New best score: {}", score);
                        best_score = score;
                        best_action = Some(self.build_match_action(&a, &b));
                        // if is_perfect_match {
                        //return Some((score, best_action.unwrap()));
                        //}
                    }
                }
            }
        }

        // 2) Consider delete actions for each ready old.
        for &a in self.ready_old.iter() {
            let score = self.net_delete_node_score(&a);
            trace!(
                "Considering delete node action: {}, score={}",
                self.old_graph.get_node(a).name,
                score,
            );
            if score > best_score {
                trace!("New best score: {}", score);
                best_score = score;
                best_action = Some(self.build_delete_node_action(&a));
            }
        }

        // 3) Consider add actions for each ready new.
        for &b in self.ready_new.iter() {
            let score = self.net_add_node_score(&b);
            trace!(
                "Considering add node action: {}, score={}",
                self.new_graph.get_node(b).name,
                score,
            );
            if score > best_score {
                trace!("New best score: {}", score);
                best_score = score;
                best_action = Some(self.build_add_node_action(&b));
            }
        }
        assert!(best_action.is_some());
        Some((best_score, best_action.unwrap()))
    }
}

// Removed queue-based scheduling; heuristic selection is computed on demand.

impl MatchSelector for GreedyMatchSelector {
    fn add_ready_node(&mut self, node: ReadyNode) {
        trace!(
            "Adding ready node: {:?}",
            match node.side {
                NodeSide::Old => format!("old:{}", self.old_graph.nodes[node.index].name),
                NodeSide::New => format!("new:{}", self.new_graph.nodes[node.index].name),
            }
        );
        match node.side {
            NodeSide::Old => {
                self.ready_old.insert(OldNodeRef(node.index));
            }
            NodeSide::New => {
                self.ready_new.insert(NewNodeRef(node.index));
            }
        }
    }

    fn select_next_match(&mut self) -> Option<MatchAction> {
        trace!("Selecting next match");
        if log_enabled!(Level::Debug) {
            let ready_old_names: Vec<String> = self
                .ready_old
                .iter()
                .map(|o| self.old_graph.get_node(*o).name.clone())
                .collect();
            let ready_new_names: Vec<String> = self
                .ready_new
                .iter()
                .map(|n| self.new_graph.get_node(*n).name.clone())
                .collect();
            trace!("Ready old: [{}]", ready_old_names.join(", "));
            trace!("Ready new: [{}]", ready_new_names.join(", "));
        }
        match self.select_best_action() {
            Some((score, action)) => {
                //              println!("best_action (score={:.2}): {:?}", score, action);
                debug!(
                    "best_action (score={}): {}",
                    score,
                    crate::graph_edit::format_match_action(&action, |side, idx| match side {
                        NodeSide::Old =>
                            format!("old:{}", self.old_graph.get_node(OldNodeRef(idx)).name),
                        NodeSide::New =>
                            format!("new:{}", self.new_graph.get_node(NewNodeRef(idx)).name),
                    })
                );
                Some(action)
            }
            None => {
                debug!("No best action found");
                None
            }
        }
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
pub fn compute_reverse_matches(
    old_graph: &DepGraph<OldNodeRef>,
    new_graph: &DepGraph<NewNodeRef>,
) -> HashMap<(OldNodeRef, NewNodeRef), i32> {
    // Helper: local shape compatibility via stored structural hash.
    let shapes_equal = |oi: OldNodeRef, ni: NewNodeRef| {
        old_graph.get_node(oi).structural_hash == new_graph.get_node(ni).structural_hash
    };

    // Scores map: best-known M(a,b) so far; initialize implicitly to 0.0 for all
    // pairs.
    let mut m_scores: HashMap<(OldNodeRef, NewNodeRef), i32> = HashMap::new();

    // Worklist seeded with all compatible sink pairs (nodes with no users).
    let mut worklist: VecDeque<(OldNodeRef, NewNodeRef)> = VecDeque::new();
    let mut on_worklist: HashSet<(OldNodeRef, NewNodeRef)> = HashSet::new();
    // Just seed the worklist with the return values.
    worklist.push_back((old_graph.return_value, new_graph.return_value));
    on_worklist.insert((old_graph.return_value, new_graph.return_value));

    //for (oi_us, on) in old_graph.nodes.iter().enumerate() {
    // for (oi_us, on) in old_graph.nodes.iter().enumerate() {
    //     if !on.users.is_empty() {
    //         continue;
    //     }
    //     for (ni_us, nn) in new_graph.nodes.iter().enumerate() {
    //         if !nn.users.is_empty() {
    //             continue;
    //         }
    //         let oi = OldNodeRef(oi_us);
    //         let ni = NewNodeRef(ni_us);
    //         if shapes_equal(oi, ni) {
    //             if on_worklist.insert((oi, ni)) {
    //                 worklist.push_back((oi, ni));
    //             }
    //         }
    //     }
    // }

    // Compute liveness masks (reachable from return via operands) for old and new.
    // Recompute helper: compute M(a,b) from current scores of user pairs.
    let recompute_score =
        |a: OldNodeRef, b: NewNodeRef, m: &HashMap<(OldNodeRef, NewNodeRef), i32>| -> i32 {
            if !shapes_equal(a, b) {
                return 0;
            }
            let z = std::cmp::max(
                old_graph.get_node(a).users.len(),
                new_graph.get_node(b).users.len(),
            );
            if z == 0 {
                return GreedyMatchSelector::SCORE_DENOMINATOR;
            }
            let mut sum: i32 = 0;
            for &(u_a, slot) in old_graph.get_node(a).users.iter() {
                let mut best: i32 = 0;
                for &(u_b, slot_b) in new_graph.get_node(b).users.iter() {
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
            sum / (z as i32)
        };

    // Process worklist: on improvement, propagate to operand pairs.
    while let Some((a, b)) = worklist.pop_front() {
        // Mark as no longer on the worklist so it can be re-enqueued upon future
        // improvements.
        on_worklist.remove(&(a, b));
        let new_score = recompute_score(a, b, &m_scores);
        let old_score = *m_scores.get(&(a, b)).unwrap_or(&0);
        if new_score > old_score {
            m_scores.insert((a, b), new_score);

            // Walk operands in lockstep; if counts match, enqueue compatible pairs.
            let a_ops: Vec<OldNodeRef> = old_graph.get_node(a).operands.iter().copied().collect();
            let b_ops: Vec<NewNodeRef> = new_graph.get_node(b).operands.iter().copied().collect();
            if a_ops.len() == b_ops.len() {
                for (op_a, op_b) in a_ops.iter().zip(b_ops.iter()) {
                    if shapes_equal(*op_a, *op_b) && !on_worklist.contains(&(*op_a, *op_b)) {
                        worklist.push_back((*op_a, *op_b));
                        on_worklist.insert((*op_a, *op_b));
                    }
                }
            }
        }
    }

    // Convert to typed key map for external use.
    let mut out: HashMap<(OldNodeRef, NewNodeRef), i32> = HashMap::new();
    for (k, score) in m_scores.into_iter() {
        out.insert(k, score);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_edit::{
        IrEdit, MatchAction, NewNodeRef, NodeSide, OldNodeRef, ReadyNode, apply_function_edits,
        build_dependency_graph, compute_function_edit,
    };
    use crate::ir::{NodePayload, NodeRef};
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

        // Use GreedyMatchSelector-based matcher
        let mut selector = GreedyMatchSelector::new(lhs, rhs);
        let edits = compute_function_edit(lhs, rhs, &mut selector).unwrap();
        assert!(edits.edits.is_empty());
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

        let mut selector = GreedyMatchSelector::new(lhs, rhs);
        let result = compute_function_edit(lhs, rhs, &mut selector);
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

        let mut selector = GreedyMatchSelector::new(old_fn, new_fn);
        let edits = compute_function_edit(old_fn, new_fn, &mut selector).unwrap();
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

        let mut selector = GreedyMatchSelector::new(old_fn, new_fn);
        let edits = compute_function_edit(old_fn, new_fn, &mut selector).unwrap();
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

        let mut selector = GreedyMatchSelector::new(old_fn, new_fn);
        let edits = compute_function_edit(old_fn, new_fn, &mut selector).unwrap();
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

        let mut selector = GreedyMatchSelector::new(old_fn, new_fn);
        let edits = compute_function_edit(old_fn, new_fn, &mut selector).unwrap();
        let patched = apply_function_edits(old_fn, &edits).unwrap();
        assert!(crate::ir_isomorphism::is_ir_isomorphic(&patched, new_fn));
    }

    #[test]
    fn single_operand_substitution_between_adds() {
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f(x: bits[8] id=1, y: bits[8] id=2, z: bits[8] id=3) -> bits[8] {
                ret add.10: bits[8] = add(x, y, id=10)
            }
            "#,
        );
        let old_fn = pkg_old.get_top().unwrap();

        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f(x: bits[8] id=1, y: bits[8] id=2, z: bits[8] id=3) -> bits[8] {
                ret add.20: bits[8] = add(x, z, id=20)
            }
            "#,
        );
        let new_fn = pkg_new.get_top().unwrap();

        let mut selector = GreedyMatchSelector::new(old_fn, new_fn);
        let edits = compute_function_edit(old_fn, new_fn, &mut selector).unwrap();

        assert_eq!(
            edits.edits.len(),
            1,
            "expected exactly one edit, got {:?}",
            edits.edits
        );
        // Verify the edit substitutes to the parameter named 'z'.
        match edits.edits[0] {
            IrEdit::SubstituteOperand { new_operand, .. } => {
                let target_node = old_fn.get_node(new_operand);
                assert_eq!(target_node.name.as_deref(), Some("z"));
                assert!(matches!(target_node.payload, NodePayload::GetParam(_)));
            }
            ref other => panic!(
                "expected edit to be an operand substitution, got {:?}",
                other
            ),
        }
        let patched = apply_function_edits(old_fn, &edits).unwrap();
        assert!(crate::ir_isomorphism::is_ir_isomorphic(&patched, new_fn));
    }

    #[test]
    fn multiple_operand_substitutions_in_concat_permutation() {
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f(x: bits[8] id=1, y: bits[8] id=2, z: bits[8] id=3) -> bits[24] {
                ret concat.10: bits[24] = concat(x, y, z, id=10)
            }
            "#,
        );
        let old_fn = pkg_old.get_top().unwrap();

        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f(x: bits[8] id=1, y: bits[8] id=2, z: bits[8] id=3) -> bits[24] {
                ret concat.20: bits[24] = concat(z, x, y, id=20)
            }
            "#,
        );
        let new_fn = pkg_new.get_top().unwrap();

        let mut selector = GreedyMatchSelector::new(old_fn, new_fn);
        let edits = compute_function_edit(old_fn, new_fn, &mut selector).unwrap();

        // Expect exactly three operand substitutions, and patched result is isomorphic.
        assert_eq!(
            edits.edits.len(),
            3,
            "expected exactly three edits, got {:?}",
            edits.edits
        );
        assert!(
            edits
                .edits
                .iter()
                .all(|e| matches!(e, IrEdit::SubstituteOperand { .. })),
            "expected only SubstituteOperand edits, got {:?}",
            edits.edits
        );

        let patched = apply_function_edits(old_fn, &edits).unwrap();
        assert!(crate::ir_isomorphism::is_ir_isomorphic(&patched, new_fn));
    }

    #[test]
    fn reverse_matches_three_adds_under_or_with_param_permutation() {
        // Old: or(or(A(w,x), B(x,y)), C(y,z))
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f(w: bits[8] id=1, x: bits[8] id=2, y: bits[8] id=3, z: bits[8] id=4) -> bits[8] {
                a: bits[8] = add(w, x, id=10)
                b: bits[8] = add(x, y, id=11)
                c: bits[8] = add(y, z, id=12)
                foo: bits[8] = or(a, b, id=13)
                ret bar: bits[8] = or(foo, b, c, id=14)
            }
            "#,
        );
        let old_fn = pkg_old.get_top().unwrap();

        // New: or(or(A'(w,x), C'(y,z)), B'(x,y)) — A stays in inner slot0; B,C swap
        // levels.
        let pkg_new = parse_ir_from_string(
            r#"package p2
            top fn f2(w: bits[8] id=1, x: bits[8] id=2, y: bits[8] id=3, z: bits[8] id=4) -> bits[8] {
                a: bits[8] = add(w, x, id=110)
                b: bits[8] = add(x, y, id=111)
                c: bits[8] = add(y, z, id=112)
                foo: bits[8] = or(a, c, id=113)
                ret bar: bits[8] = or(foo, b, b, id=114)
            }
            "#,
        );
        let new_fn = pkg_new.get_top().unwrap();

        // Build dependency graphs and compute reverse matches.
        let old_graph = crate::graph_edit::build_dependency_graph::<OldNodeRef>(old_fn);
        let new_graph = crate::graph_edit::build_dependency_graph::<NewNodeRef>(new_fn);
        let reverse = compute_reverse_matches(&old_graph, &new_graph);
        let den = GreedyMatchSelector::SCORE_DENOMINATOR;

        // Helper to get node indices by name.
        let find_named = |f: &crate::ir::Fn, name: &str| -> usize {
            f.nodes
                .iter()
                .position(|n| n.name.as_deref() == Some(name))
                .expect("node by name not found")
        };
        // Helper to read reverse score by node names, or 0 if missing.
        // Panics if nodes are not found.
        // Usage: rev("old_name", "new_name")
        let rev = |old_name: &str, new_name: &str| -> i32 {
            let oi = find_named(old_fn, old_name);
            let ni = find_named(new_fn, new_name);
            reverse
                .get(&(OldNodeRef(oi), NewNodeRef(ni)))
                .copied()
                .unwrap_or(0)
        };

        assert_eq!(rev("bar", "bar"), den);
        assert_eq!(rev("foo", "bar"), 0);
        assert_eq!(rev("bar", "foo"), 0);
        assert_eq!(rev("a", "a"), den);
        assert_eq!(rev("b", "b"), den / 2);
        assert_eq!(rev("c", "c"), 0);
        assert_eq!(rev("w", "w"), den);
        assert_eq!(rev("x", "x"), 3 * den / 4);
        assert_eq!(rev("y", "y"), den / 4);
        assert_eq!(rev("z", "z"), 0);
    }

    #[test]
    fn forward_match_and_opportunity_costs_behave_as_expected() {
        // Reuse the same IR structure as the reverse-match test.
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f(w: bits[8] id=1, x: bits[8] id=2, y: bits[8] id=3, z: bits[8] id=4) -> bits[8] {
                a: bits[8] = add(w, x, id=10)
                b: bits[8] = add(x, y, id=11)
                c: bits[8] = add(x, z, id=12)
                ret foo: bits[8] = or(a, b, c, id=13)
            }
            "#,
        );
        let old_fn = pkg_old.get_top().unwrap();

        let pkg_new = parse_ir_from_string(
            r#"package p2
            top fn f2(w: bits[8] id=1, x: bits[8] id=2, y: bits[8] id=3, z: bits[8] id=4) -> bits[8] {
                a: bits[8] = add(w, x, id=110)
                not_y: bits[8] = not(y, id=111)
                b: bits[8] = add(x, not_y, id=112)
                c: bits[8] = add(x, z, id=114)
                ret foo: bits[8] = or(a, b, c, id=115)
            }
            "#,
        );
        let new_fn = pkg_new.get_top().unwrap();

        let mut sel = GreedyMatchSelector::new(old_fn, new_fn);
        let den = GreedyMatchSelector::SCORE_DENOMINATOR;
        let perf = GreedyMatchSelector::PERFECT_MATCH_SCORE;

        // Helper to get node indices by name.
        let find_named = |f: &crate::ir::Fn, name: &str| -> usize {
            f.nodes
                .iter()
                .position(|n| n.name.as_deref() == Some(name))
                .expect("node by name not found")
        };

        // Explicitly pre-match parameters by name using the MatchSelector API.
        for pname in ["w", "x", "y", "z"].iter() {
            let oi = find_named(old_fn, pname);
            let ni = find_named(new_fn, pname);
            sel.add_ready_node(ReadyNode {
                side: NodeSide::Old,
                index: oi,
            });
            sel.add_ready_node(ReadyNode {
                side: NodeSide::New,
                index: ni,
            });
            let action = MatchAction::MatchNodes {
                old_index: OldNodeRef(oi),
                new_index: NewNodeRef(ni),
                new_operands: Vec::new(),
                is_new_return: false,
            };
            sel.update_after_match(&action);
        }
        for n in ["a", "b", "c"].iter() {
            sel.add_ready_node(ReadyNode {
                side: NodeSide::Old,
                index: find_named(old_fn, n),
            });
        }
        for n in ["a", "c", "not_y"].iter() {
            sel.add_ready_node(ReadyNode {
                side: NodeSide::New,
                index: find_named(new_fn, n),
            });
        }

        // Forward match scores on adds should be perfect under param mapping.
        let match_score = |old_name: &str, new_name: &str| -> (bool, i32) {
            let oi = find_named(old_fn, old_name);
            let ni = find_named(new_fn, new_name);
            sel.match_score(&OldNodeRef(oi), &NewNodeRef(ni))
        };
        assert_eq!(match_score("a", "a"), (true, 2 * perf));
        assert_eq!(match_score("b", "b"), (true, perf + den / 2));
        assert_eq!(match_score("c", "c"), (true, 2 * perf));
        assert_eq!(match_score("foo", "foo"), (true, perf));

        // Helper: opportunity cost via names (use None to omit a side).
        let best_unready = |a: Option<&str>, b: Option<&str>| -> i32 {
            sel.best_unready_match_score(
                a.map(|n| OldNodeRef(find_named(old_fn, n))),
                b.map(|n| NewNodeRef(find_named(new_fn, n))),
            )
        };
        assert_eq!(best_unready(Some("a"), Some("a")), 0);
        assert_eq!(best_unready(Some("c"), Some("c")), den / 2);
        assert_eq!(best_unready(Some("a"), Some("c")), 0);
        assert_eq!(best_unready(Some("b"), Some("a")), perf + den / 2);

        // Helpers for name-based selector scores.
        let match_score = |old_name: &str, new_name: &str| -> (bool, i32) {
            let oi = OldNodeRef(find_named(old_fn, old_name));
            let ni = NewNodeRef(find_named(new_fn, new_name));
            sel.match_score(&oi, &ni)
        };
        assert_eq!(match_score("a", "a"), (true, 2 * perf));
        assert_eq!(match_score("b", "c"), (false, den / 2));
        assert_eq!(match_score("c", "c"), (true, 2 * perf));
    }
}
