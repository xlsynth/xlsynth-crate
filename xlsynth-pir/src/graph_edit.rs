// SPDX-License-Identifier: Apache-2.0

//! Skeleton library for computing and applying IR graph edits between two
//! XLS IR functions. For now this returns an empty set of edits when computing,
//! and applying is a no-op that returns the input function unmodified.

use crate::ir::{Fn, Node, NodeRef};
use crate::ir_utils::{operands, remap_payload_with};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::greedy_graph_edit::MatchAction;
use std::collections::HashMap;
use std::collections::HashSet;

/// Represents a concrete IR edit operation to be applied to a function.
#[derive(Debug, Clone)]
pub enum IrEdit {
    /// Adds a concrete node; its payload operands should reference either
    /// existing old-node indices or previously added nodes by their original
    /// new indices (resolved during application).
    AddNode { new_index: usize, node: Node },
    /// Deletes a node from the "old" function by index in `old_fn.nodes`.
    DeleteNode { index: usize },
    /// Redirects a specific operand of a user node to a different target node.
    ///
    /// - `user_index`: index of the node whose operand will be rewritten
    /// - `operand_slot`: which operand position on the user node to rewrite
    /// - `new_target_index`: node index that the operand should reference
    SubstituteOperand {
        user_index: usize,
        operand_slot: usize,
        new_target_index: usize,
    },
    /// Sets the function return to reference either an existing old node index
    /// or a newly added node index (resolved during application via mapping).
    SetReturn { index: usize, is_new: bool },
}

/// A collection of edits that convert `old` into `new`.
#[derive(Debug, Clone, Default)]
pub struct IrEditSet {
    pub edits: Vec<IrEdit>,
}

/// A collection of match decisions produced by the matcher.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IrMatchSet {
    pub matches: Vec<MatchAction>,
}

/// Dependency information for a single node.
#[derive(Debug, Clone)]
pub struct DepNode {
    pub operands: Vec<usize>, // deps (by node index)
    pub users: Vec<usize>,    // use-list (by node index)
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
    let mut users_list: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (idx, ops) in operands_list.iter().enumerate() {
        for &d in ops {
            users_list[d].push(idx);
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

/// Computes MatchNodes actions that pair parameters (GetParam nodes) in `old`
/// and `new` functions by parameter name.
pub fn compute_parameter_matches(old: &Fn, new: &Fn) -> Vec<MatchAction> {
    let mut old_param_to_idx: HashMap<String, usize> = HashMap::new();
    for (idx, node) in old.nodes.iter().enumerate() {
        if let crate::ir::NodePayload::GetParam(pid) = node.payload {
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

    let mut matches: Vec<MatchAction> = Vec::new();
    for op in old.params.iter() {
        if let (Some(&oi), Some(&ni)) = (
            old_param_to_idx.get(&op.name),
            new_param_to_idx.get(&op.name),
        ) {
            matches.push(MatchAction::MatchNodes {
                old_index: oi,
                new_index: ni,
                operand_substitutions: Vec::new(),
                is_new_return: new
                    .ret_node_ref
                    .map(|nr| nr.index)
                    .map_or(false, |ri| ri == ni),
            });
        }
    }
    matches
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

        // Seed parameter matches by name and enqueue them.
        for m in compute_parameter_matches(old, new).into_iter() {
            sel.push_action(m);
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
                    is_new_return: self
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
            for &user in old_graph.nodes[idx].users.iter() {
                if old_remain[user] > 0 {
                    old_remain[user] -= 1;
                    if old_remain[user] == 0 {
                        selector.add_ready_node(ReadyNode { side, index: user });
                    }
                }
            }
        }
        NodeSide::New => {
            for &user in new_graph.nodes[idx].users.iter() {
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

/// Applies a sequence of IrEdits to `old`, using `new` as the source of truth
/// for any nodes that must be added. Returns the transformed function.
pub fn apply_function_edits(old: &Fn, edits: &IrEditSet) -> Result<Fn, String> {
    let mut patched = old.clone();

    // Map from original-new indices (carried in AddNode payloads) to patched
    // indices.
    let mut new_to_patched: HashMap<usize, usize> = HashMap::new();

    for e in edits.edits.iter() {
        match e.clone() {
            IrEdit::DeleteNode { index } => {
                if index >= patched.nodes.len() {
                    return Err(format!("DeleteNode index {} out of bounds", index));
                }
                patched.nodes[index].payload = crate::ir::NodePayload::Nil;
            }
            IrEdit::AddNode { new_index, node } => {
                // Remap payload operands: if operand index exists in new_to_patched, use
                // mapped; otherwise treat it as an old index that already
                // exists in `patched`.
                let remapped_payload = remap_payload_with(&node.payload, |nr: NodeRef| {
                    if let Some(&mapped) = new_to_patched.get(&nr.index) {
                        NodeRef { index: mapped }
                    } else {
                        nr
                    }
                });
                let cloned = crate::ir::Node {
                    text_id: node.text_id,
                    name: node.name.clone(),
                    ty: node.ty.clone(),
                    payload: remapped_payload,
                    pos: node.pos.clone(),
                };
                let patched_index = patched.nodes.len();
                patched.nodes.push(cloned);
                new_to_patched.insert(new_index, patched_index);
                // If this node originated from a new graph index, the
                // conversion should have embedded that index in
                // operand references of dependents, which will update
                // via new_to_patched when those dependents are applied.
            }
            IrEdit::SubstituteOperand {
                user_index: _,
                operand_slot: _,
                new_target_index: _,
            } => {
                // Not yet implemented; current matcher does not produce operand
                // redirects. Leave as a no-op for now.
            }
            IrEdit::SetReturn { index, is_new } => {
                let target_idx = if is_new {
                    *new_to_patched
                        .get(&index)
                        .expect("SetReturn refers to new node not yet added")
                } else {
                    index
                };
                patched.ret_node_ref = Some(NodeRef { index: target_idx });
            }
        }
    }

    Ok(patched)
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
                let cloned = crate::ir::Node {
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

/// Returns true if two functions are structurally isomorphic when traversed
/// from their return nodes. Names/ids may differ; operators, attributes,
/// types, and ordered operand relationships must match.
// Intentionally do not re-export isomorphism to avoid module import ordering
// issues in some build contexts; tests refer to it by full path.

#[cfg(test)]
mod tests {
    use super::*;
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

    fn apply_is_noop_for_now() {
        let pkg = parse_ir_from_string(
            r#"package p
            top fn f(x: bits[8]) -> bits[8] {
                ret identity.2: bits[8] = identity(x, id=2)
            }
            "#,
        );
        let f = pkg.get_top().unwrap();
        let edits = IrEditSet::default();
        let applied = apply_function_edits(f, &edits).unwrap();
        assert_eq!(applied.to_string(), f.to_string());
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
