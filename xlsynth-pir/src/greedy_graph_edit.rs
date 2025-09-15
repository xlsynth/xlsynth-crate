// SPDX-License-Identifier: Apache-2.0

//! Greedy edit-distance computation between two XLS IR functions.
//! Contains the matching machinery and conversion of matches into concrete
//! edits.

use crate::graph_edit::{IrEdit, IrEditSet};
use crate::ir::{Fn, Node, NodeRef};
use crate::ir_utils::{operands, remap_payload_with};
use crate::node_hashing::compute_node_local_structural_hash;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::VecDeque;

/// Represents an edit to transform one IR function into another.
///
/// This is a placeholder skeleton; variants will be expanded in future work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchAction {
    /// Delete a node present in the old function by index in `old_fn.nodes`.
    DeleteNode { old_index: usize },
    /// Add a node present in the new function by index in `new_fn.nodes`.
    AddNode { new_index: usize, is_return: bool },
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
        is_return: bool,
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

/// Default selector that mimics the previous behavior: a simple FIFO-like
/// priority over ready deletes/adds with stable ordering.
pub struct NaiveMatchSelector<'a> {
    order_counter: usize,
    heap: BinaryHeap<QueueEntry>,
    old: &'a Fn,
    new: &'a Fn,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QueueEntry {
    cost: u32,
    order: usize,
    action: MatchAction,
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse cost and order for min-heap behavior using BinaryHeap (which is
        // max-heap).
        (Reverse(self.cost), Reverse(self.order)).cmp(&(Reverse(other.cost), Reverse(other.order)))
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> NaiveMatchSelector<'a> {
    pub fn new(old: &'a Fn, new: &'a Fn) -> Self {
        let mut sel = Self {
            order_counter: 0,
            heap: BinaryHeap::new(),
            old,
            new,
        };

        // Seed parameter matches by name: map each parameter name to its
        // GetParam node index in old and new, and enqueue MatchNodes.
        let mut old_param_to_idx: HashMap<String, usize> = HashMap::new();
        for (idx, node) in old.nodes.iter().enumerate() {
            if let crate::ir::NodePayload::GetParam(pid) = node.payload {
                // Find the name for this pid
                if let Some(p) = old.params.iter().find(|p| p.id == pid) {
                    old_param_to_idx.insert(p.name.clone(), idx);
                }
            }
        }
        let mut new_param_to_idx: HashMap<String, usize> = HashMap::new();
        for (idx, node) in new.nodes.iter().enumerate() {
            if let crate::ir::NodePayload::GetParam(pid) = node.payload {
                if let Some(p) = new.params.iter().find(|p| p.id == pid) {
                    new_param_to_idx.insert(p.name.clone(), idx);
                }
            }
        }
        for op in old.params.iter() {
            if let (Some(&oi), Some(&ni)) = (
                old_param_to_idx.get(&op.name),
                new_param_to_idx.get(&op.name),
            ) {
                sel.push_action(MatchAction::MatchNodes {
                    old_index: oi,
                    new_index: ni,
                    operand_substitutions: Vec::new(),
                    is_return: false,
                });
            }
        }

        sel
    }

    fn push_action(&mut self, action: MatchAction) {
        let entry = QueueEntry {
            cost: 1,
            order: self.order_counter,
            action,
        };
        self.order_counter += 1;
        self.heap.push(entry);
    }
}

impl<'a> MatchSelector for NaiveMatchSelector<'a> {
    fn add_ready_node(&mut self, node: ReadyNode) {
        match node.side {
            NodeSide::Old => {
                if matches!(
                    self.old.nodes[node.index].payload,
                    crate::ir::NodePayload::GetParam(_)
                ) {
                    return;
                }
                self.push_action(MatchAction::DeleteNode {
                    old_index: node.index,
                });
            }
            NodeSide::New => {
                if matches!(
                    self.new.nodes[node.index].payload,
                    crate::ir::NodePayload::GetParam(_)
                ) {
                    return;
                }
                self.push_action(MatchAction::AddNode {
                    new_index: node.index,
                    is_return: self
                        .new
                        .ret_node_ref
                        .map(|nr| nr.index)
                        .map_or(false, |ri| ri == node.index),
                });
            }
        }
    }

    fn select_next_match(&mut self) -> Option<MatchAction> {
        let entry = self.heap.pop()?;
        Some(entry.action)
    }
}

/// Greedy selector that pre-scores candidate matches using a reverse traversal
/// and then performs forward-direction matching guided by those scores.
pub struct GreedyMatchSelector<'a> {
    old: &'a Fn,
    new: &'a Fn,
    /// Pairs of (old_index, new_index) that were reverse-identified as perfect
    /// matches (same structure/signature and recursively identical children).
    reverse_perfect_matches: Vec<(usize, usize)>,
    /// Pairs of (old_index, new_index) that were reverse-identified as strong
    /// matches (same local shape/signature; children may differ but
    /// compatible).
    reverse_strong_matches: Vec<(usize, usize)>,
    /// Forward priority queue for applying actions. Placeholder for now.
    heap: BinaryHeap<QueueEntry>,
    order_counter: usize,
}

impl<'a> GreedyMatchSelector<'a> {
    pub fn new(old: &'a Fn, new: &'a Fn) -> Self {
        let (reverse_perfect_matches, reverse_strong_matches) = compute_reverse_matches(old, new);
        Self {
            old,
            new,
            reverse_perfect_matches,
            reverse_strong_matches,
            heap: BinaryHeap::new(),
            order_counter: 0,
        }
    }

    fn push_action(&mut self, action: MatchAction) {
        let entry = QueueEntry {
            cost: 1,
            order: self.order_counter,
            action,
        };
        self.order_counter += 1;
        self.heap.push(entry);
    }
}

impl<'a> MatchSelector for GreedyMatchSelector<'a> {
    fn add_ready_node(&mut self, node: ReadyNode) {
        // Skeleton only: enqueue basic add/delete like the naive selector for now.
        match node.side {
            NodeSide::Old => {
                if matches!(
                    self.old.nodes[node.index].payload,
                    crate::ir::NodePayload::GetParam(_)
                ) {
                    return;
                }
                self.push_action(MatchAction::DeleteNode {
                    old_index: node.index,
                });
            }
            NodeSide::New => {
                if matches!(
                    self.new.nodes[node.index].payload,
                    crate::ir::NodePayload::GetParam(_)
                ) {
                    return;
                }
                self.push_action(MatchAction::AddNode {
                    new_index: node.index,
                    is_return: self
                        .new
                        .ret_node_ref
                        .map(|nr| nr.index)
                        .map_or(false, |ri| ri == node.index),
                });
            }
        }
    }

    fn select_next_match(&mut self) -> Option<MatchAction> {
        let entry = self.heap.pop()?;
        Some(entry.action)
    }
}

/// Computes reverse-direction candidate match pairs by walking old/new graphs
/// from returns to inputs. For now this is a stub that returns empty sets.
pub fn compute_reverse_matches(old: &Fn, new: &Fn) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let mut perfect_pairs: Vec<(usize, usize)> = Vec::new();
    let mut old_to_new: HashMap<usize, usize> = HashMap::new();
    let mut new_to_old: HashMap<usize, usize> = HashMap::new();

    // Build user info for reverse traversal checks.
    let old_graph = build_dependency_graph(old);
    let new_graph = build_dependency_graph(new);

    // Helper: check local shape compatibility via local structural hash.
    let shapes_equal = |oi: usize, ni: usize| {
        compute_node_local_structural_hash(old, NodeRef { index: oi })
            == compute_node_local_structural_hash(new, NodeRef { index: ni })
    };

    // Seed with return pair; assume both defined and same shape.
    let mut work_q: VecDeque<(usize, usize)> = VecDeque::new();
    let or = old
        .ret_node_ref
        .expect("compute_reverse_matches requires old return");
    let nr = new
        .ret_node_ref
        .expect("compute_reverse_matches requires new return");
    assert!(
        shapes_equal(or.index, nr.index),
        "Return nodes must have the same local shape"
    );
    old_to_new.insert(or.index, nr.index);
    new_to_old.insert(nr.index, or.index);
    perfect_pairs.push((or.index, nr.index));
    work_q.push_back((or.index, nr.index));

    // Helper function tests whether (oi, ni) can be declared perfectly
    // matched based on already-known perfect user pairs.
    fn are_nodes_perfect(
        _old: &crate::ir::Fn,
        _new: &crate::ir::Fn,
        old_graph: &DepGraph,
        new_graph: &DepGraph,
        old_to_new: &HashMap<usize, usize>,
        shapes_equal: &dyn std::ops::Fn(usize, usize) -> bool,
        oi: usize,
        ni: usize,
    ) -> bool {
        // Local shape must match.
        if !shapes_equal(oi, ni) {
            return false;
        }
        // Uses count must match exactly.
        let old_users = &old_graph.nodes[oi].users;
        let new_users = &new_graph.nodes[ni].users;
        if old_users.len() != new_users.len() {
            return false;
        }
        // For every user (user, slot) of old, ensure mapped new user uses ni at same
        // slot.
        for &(old_use, old_slot) in old_users.iter() {
            let Some(&new_use) = old_to_new.get(&old_use) else {
                return false;
            };
            if !new_users.contains(&(new_use, old_slot)) {
                return false;
            }
        }
        true
    }

    // Propagate reverse-perfect matches down to operands when criteria are met.
    while let Some((uo, un)) = work_q.pop_front() {
        let old_ops: Vec<usize> = operands(&old.nodes[uo].payload)
            .into_iter()
            .map(|r| r.index)
            .collect();
        let new_ops: Vec<usize> = operands(&new.nodes[un].payload)
            .into_iter()
            .map(|r| r.index)
            .collect();
        if old_ops.len() != new_ops.len() {
            continue;
        }
        for (oo, nn) in old_ops.into_iter().zip(new_ops.into_iter()) {
            // Respect already established mappings if present.
            if let Some(&mapped) = old_to_new.get(&oo) {
                if mapped != nn {
                    continue;
                }
            }
            if let Some(&mapped) = new_to_old.get(&nn) {
                if mapped != oo {
                    continue;
                }
            }
            if are_nodes_perfect(
                old,
                new,
                &old_graph,
                &new_graph,
                &old_to_new,
                &shapes_equal,
                oo,
                nn,
            ) {
                if old_to_new.insert(oo, nn).is_none() {
                    new_to_old.insert(nn, oo);
                    perfect_pairs.push((oo, nn));
                    work_q.push_back((oo, nn));
                }
            }
        }
    }

    (perfect_pairs, Vec::new())
}

/// Computes an edit set (distance) required to transform `old` into `new`.
/// Internally computes a match set, then converts matches to concrete edits.
pub fn compute_function_edit_distance(old: &Fn, new: &Fn) -> Result<IrEditSet, String> {
    let mut selector = NaiveMatchSelector::new(old, new);
    let matches = compute_function_edit(old, new, &mut selector)?;
    Ok(convert_match_set_to_edit_set(old, new, &matches))
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
                    is_return: is_ret,
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
                    is_return: old.ret_node_ref.map(|nr| nr.index) == Some(old_index)
                        && new.ret_node_ref.map(|nr| nr.index) == Some(new_index),
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
            is_return: _,
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
                is_return: _,
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
                is_return: _,
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
                lit.1: bits[1] = literal(value=1, id=1)
                ret identity.2: bits[1] = identity(lit.1, id=2)
            }
            "#,
        );
        let old_fn = pkg_old.get_top().unwrap();

        // Same nodes, but return points to a different node (the literal).
        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f() -> bits[1] {
                ret lit.1: bits[1] = literal(value=1, id=1)
                identity.2: bits[1] = identity(lit.1, id=2)
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
                s.103: bits[8] = add(a, b, id=103)
                ret id.104: bits[8] = identity(s.103, id=104)
            }
            "#,
        );
        let new_fn = pkg_new.get_top().unwrap();

        let edits = compute_function_edit_distance(old_fn, new_fn).unwrap();
        let patched = apply_function_edits(old_fn, &edits).unwrap();
        assert!(crate::xls_ir::ir_isomorphism::is_ir_isomorphic(
            &patched, new_fn
        ));
    }
}
