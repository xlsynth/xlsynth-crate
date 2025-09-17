// SPDX-License-Identifier: Apache-2.0

//! Greedy edit-distance computation between two XLS IR functions.
//! Contains the matching machinery and conversion of matches into concrete
//! edits.

use crate::graph_edit::{
    DepGraph, MatchAction, MatchSelector, NewNodeRef, NodeSide, OldNodeRef, ReadyNode,
    build_dependency_graph,
};
use crate::node_hashing::FwdHash;
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
    reverse_scores: HashMap<(OldNodeRef, NewNodeRef), f64>,
    /// Ready sets for quick membership and stable iteration by node index.
    ready_old: BTreeSet<OldNodeRef>,
    ready_new: BTreeSet<NewNodeRef>,
    /// Mapping from matched old node index to new node index.
    matched_old_to_new: HashMap<OldNodeRef, NewNodeRef>,
    /// Nodes that have been handled (matched or added/removed) in old/new.
    handled_old: BTreeSet<OldNodeRef>,
    handled_new: BTreeSet<NewNodeRef>,
}

impl GreedyMatchSelector {
    /// Generic opportunity cost using a supplied match score lambda.
    fn opportunity_cost<F>(
        &self,
        a: Option<OldNodeRef>,
        b: Option<NewNodeRef>,
        get_match_score: F,
    ) -> f64
    where
        F: std::ops::Fn(OldNodeRef, NewNodeRef) -> f64,
    {
        match (a, b) {
            (Some(a_idx), Some(b_idx)) => {
                let mut best = 0.0f64;
                if let Some(bs) = self.old_to_new_by_hash.get(&a_idx) {
                    for &b2 in bs.iter() {
                        if b2 == b_idx || !self.ready_new.contains(&b2) {
                            continue;
                        }
                        let v = get_match_score(a_idx, b2);
                        if v > best {
                            best = v;
                        }
                    }
                }
                if let Some(as_) = self.new_to_old_by_hash.get(&b_idx) {
                    for &a2 in as_.iter() {
                        if a2 == a_idx || !self.ready_old.contains(&a2) {
                            continue;
                        }
                        let v = get_match_score(a2, b_idx);
                        if v > best {
                            best = v;
                        }
                    }
                }
                best
            }
            (Some(a_idx), None) => {
                let mut best = 0.0f64;
                if let Some(bs) = self.old_to_new_by_hash.get(&a_idx) {
                    for &b2 in bs.iter() {
                        if !self.ready_new.contains(&b2) {
                            continue;
                        }
                        let v = get_match_score(a_idx, b2);
                        if v > best {
                            best = v;
                        }
                    }
                }
                best
            }
            (None, Some(b_idx)) => {
                let mut best = 0.0f64;
                if let Some(as_) = self.new_to_old_by_hash.get(&b_idx) {
                    for &a2 in as_.iter() {
                        if !self.ready_old.contains(&a2) {
                            continue;
                        }
                        let v = get_match_score(a2, b_idx);
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
    fn compute_match_score(&self, a: &OldNodeRef, b: &NewNodeRef) -> f64 {
        let a_val = *a;
        let b_val = *b;
        let fwd = self.forward_match_score(a_val, b_val);
        let rev = *self.reverse_scores.get(&(a_val, b_val)).unwrap_or(&0.0);
        let oc_f = self.opportunity_cost(Some(a_val), Some(b_val), |oa, nb| {
            self.forward_match_score(oa, nb)
        });
        let oc_r = self.opportunity_cost(Some(a_val), Some(b_val), |oa, nb| {
            *self.reverse_scores.get(&(oa, nb)).unwrap_or(&0.0)
        });
        fwd + rev - (oc_f + oc_r) / 2.0
    }

    /// Score for deleting a ready old node `a`.
    /// score = -(best_forward_alt_for_a + best_reverse_alt_for_a)
    fn compute_delete_node_score(&self, a: &OldNodeRef) -> f64 {
        let a_val = *a;
        let oc_f =
            self.opportunity_cost(Some(a_val), None, |oa, nb| self.forward_match_score(oa, nb));
        let oc_r = self.opportunity_cost(Some(a_val), None, |oa, nb| {
            *self.reverse_scores.get(&(oa, nb)).unwrap_or(&0.0)
        });
        -(oc_f + oc_r) / 2.0
    }

    /// Score for adding a ready new node `b`.
    /// score = -(best_forward_alt_for_b + best_reverse_alt_for_b)
    fn compute_add_node_score(&self, b: &NewNodeRef) -> f64 {
        let b_val = *b;
        let oc_f =
            self.opportunity_cost(None, Some(b_val), |oa, nb| self.forward_match_score(oa, nb));
        let oc_r = self.opportunity_cost(None, Some(b_val), |oa, nb| {
            *self.reverse_scores.get(&(oa, nb)).unwrap_or(&0.0)
        });
        -(oc_f + oc_r) / 2.0
    }

    /// Build a MatchNodes action for (a, b).
    fn build_match_action(&self, a: &OldNodeRef, b: &NewNodeRef) -> Option<MatchAction> {
        Some(MatchAction::MatchNodes {
            old_index: *a,
            new_index: *b,
            operand_substitutions: Vec::new(),
            is_new_return: *b == self.new_graph.return_value,
        })
    }

    /// Build a DeleteNode action for old node `a`.
    fn build_delete_node_action(&self, a: &OldNodeRef) -> Option<MatchAction> {
        Some(MatchAction::DeleteNode { old_index: *a })
    }

    /// Build an AddNode action for new node `b`.
    fn build_add_node_action(&self, b: &NewNodeRef) -> Option<MatchAction> {
        Some(MatchAction::AddNode {
            new_index: *b,
            is_new_return: *b == self.new_graph.return_value,
        })
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
    /// Returns 0.0 unless the local shapes of (old_index, new_index) are equal.
    /// When equal, returns the fraction of operand positions that are matched
    /// according to `matched_old_to_new` (1.0 for a matched operand, 0.0
    /// otherwise), normalized by the number of operands (must be equal
    /// between nodes).
    pub fn forward_match_score(&self, old_index: OldNodeRef, new_index: NewNodeRef) -> f64 {
        let oi: usize = old_index.into();
        let ni: usize = new_index.into();
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
            return 1.0;
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
        (matches as f64) / (old_ops.len() as f64)
    }
}

// Removed queue-based scheduling; heuristic selection is computed on demand.

impl MatchSelector for GreedyMatchSelector {
    fn add_ready_node(&mut self, node: ReadyNode) {
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
        if self.ready_old.is_empty() && self.ready_new.is_empty() {
            return None;
        }

        let mut best_action: Option<MatchAction> = None;
        let mut best_score = f64::NEG_INFINITY;

        // 1) Consider match actions for each ready old against compatible ready new.
        for &a in self.ready_old.iter() {
            if let Some(news) = self.old_to_new_by_hash.get(&a) {
                for &b in news.iter() {
                    if !self.ready_new.contains(&b) {
                        continue;
                    }
                    let score = self.compute_match_score(&a, &b);
                    if score > best_score {
                        best_score = score;
                        best_action = self.build_match_action(&a, &b);
                    }
                }
            }
        }

        // 2) Consider delete actions for each ready old.
        for &a in self.ready_old.iter() {
            let score = self.compute_delete_node_score(&a);
            if score > best_score {
                best_score = score;
                best_action = self.build_delete_node_action(&a);
            }
        }

        // 3) Consider add actions for each ready new.
        for &b in self.ready_new.iter() {
            let score = self.compute_add_node_score(&b);
            if score > best_score {
                best_score = score;
                best_action = self.build_add_node_action(&b);
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
pub fn compute_reverse_matches(
    old_graph: &DepGraph<OldNodeRef>,
    new_graph: &DepGraph<NewNodeRef>,
) -> HashMap<(OldNodeRef, NewNodeRef), f64> {
    // Helper: local shape compatibility via stored structural hash.
    let shapes_equal = |oi: OldNodeRef, ni: NewNodeRef| {
        old_graph.get_node(oi).structural_hash == new_graph.get_node(ni).structural_hash
    };

    // Scores map: best-known M(a,b) so far; initialize implicitly to 0.0 for all
    // pairs.
    let mut m_scores: HashMap<(OldNodeRef, NewNodeRef), f64> = HashMap::new();

    // Worklist seeded with all compatible sink pairs (nodes with no users).
    let mut worklist: VecDeque<(OldNodeRef, NewNodeRef)> = VecDeque::new();
    let mut on_worklist: HashSet<(OldNodeRef, NewNodeRef)> = HashSet::new();
    for (oi_us, on) in old_graph.nodes.iter().enumerate() {
        if !on.users.is_empty() {
            continue;
        }
        for (ni_us, nn) in new_graph.nodes.iter().enumerate() {
            if !nn.users.is_empty() {
                continue;
            }
            let oi = OldNodeRef(oi_us);
            let ni = NewNodeRef(ni_us);
            if shapes_equal(oi, ni) {
                if on_worklist.insert((oi, ni)) {
                    worklist.push_back((oi, ni));
                }
            }
        }
    }

    // Recompute helper: compute M(a,b) from current scores of user pairs.
    let recompute_score =
        |a: OldNodeRef, b: NewNodeRef, m: &HashMap<(OldNodeRef, NewNodeRef), f64>| -> f64 {
            if !shapes_equal(a, b) {
                return 0.0;
            }
            let z = std::cmp::max(
                old_graph.get_node(a).users.len(),
                new_graph.get_node(b).users.len(),
            );
            if z == 0 {
                return 1.0;
            }
            let mut sum = 0.0f64;
            for &(u_a, slot) in old_graph.get_node(a).users.iter() {
                let mut best = 0.0f64;
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
    let mut out: HashMap<(OldNodeRef, NewNodeRef), f64> = HashMap::new();
    for (k, score) in m_scores.into_iter() {
        out.insert(k, score);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_edit::{IrEdit, apply_function_edits, compute_function_edit};
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
            IrEdit::SubstituteOperand {
                new_target_index, ..
            } => {
                let target_node = old_fn.get_node(NodeRef {
                    index: new_target_index,
                });
                assert_eq!(target_node.name.as_deref(), Some("z"));
                assert!(matches!(target_node.payload, NodePayload::GetParam(_)));
            }
            ref other => panic!(
                "expected edit to be an operand substitution, got {:?}",
                other
            ),
        }
    }
}
