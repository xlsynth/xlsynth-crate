// SPDX-License-Identifier: Apache-2.0

//! Skeleton library for computing and applying IR graph edits between two
//! XLS IR functions. For now this returns an empty set of edits when computing,
//! and applying is a no-op that returns the input function unmodified.

use crate::ir::{self, Fn, Node, NodeRef};
use crate::ir_utils::{get_topological, operands, remap_payload_with};
use crate::node_hashing::{
    FwdHash, compute_node_local_structural_hash, compute_node_structural_hash,
};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

// MatchAction is defined below in this module
use std::collections::HashMap;

/// Represents a concrete IR edit operation to be applied to a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OldNodeRef(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NewNodeRef(pub usize);

impl From<usize> for OldNodeRef {
    fn from(v: usize) -> Self {
        OldNodeRef(v)
    }
}

impl From<usize> for NewNodeRef {
    fn from(v: usize) -> Self {
        NewNodeRef(v)
    }
}

impl From<OldNodeRef> for usize {
    fn from(v: OldNodeRef) -> Self {
        v.0
    }
}

impl From<NewNodeRef> for usize {
    fn from(v: NewNodeRef) -> Self {
        v.0
    }
}

#[derive(Debug, Clone)]
pub enum IrEdit {
    /// Adds a concrete node; its payload operands must reference either
    /// existing old-node indices or previously added nodes by their patched
    /// indices. Application will push this node to the end of the nodes vec.
    AddNode { node: Node },
    /// Deletes a node from the "old" function by index in `old_fn.nodes`.
    DeleteNode { node: NodeRef },
    /// Redirects a specific operand of a node to a different target node.
    ///
    /// - `node`: reference of the node whose operand will be rewritten
    /// - `operand_slot`: which operand position on the node to rewrite
    /// - `new_operand`: reference of the node that the operand should target
    SubstituteOperand {
        node: NodeRef,
        operand_slot: usize,
        new_operand: NodeRef,
    },
    /// Sets the function return to reference either an existing old node index
    /// or a newly added node index (resolved during application via mapping).
    SetReturn { node: NodeRef },
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

/// Represents a high-level match/mapping between old and new graph nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchAction {
    /// Delete a node present in the old function.
    DeleteNode { old_index: OldNodeRef },
    /// Add a node present in the new function.
    AddNode {
        new_index: NewNodeRef,
        is_return: bool,
    },
    /// Pair an old node with a new node, with optional operand substitutions.
    MatchNodes {
        old_index: OldNodeRef,
        new_index: NewNodeRef,
        /// Full ordered operand list for the matched new node
        new_operands: Vec<NewNodeRef>,
        is_new_return: bool,
    },
}

/// Dependency information for a single node, parameterized by index type.
#[derive(Debug, Clone)]
pub struct DepNode<I> {
    pub operands: Vec<I>,       // deps (by node index)
    pub users: Vec<(I, usize)>, // (user index, operand slot)
    pub structural_hash: FwdHash,
    pub is_return: bool,
    pub name: String,
}

/// Dependency graph for a function: collection of per-node dependency info.
#[derive(Debug, Clone)]
pub struct DepGraph<I> {
    pub nodes: Vec<DepNode<I>>,
    pub return_value: I,
}

impl<I> DepGraph<I>
where
    I: Into<usize> + Copy,
{
    pub fn get_node(&self, index: I) -> &DepNode<I> {
        let idx: usize = index.into();
        &self.nodes[idx]
    }
}

/// Builds a dependency graph for the given function using a typed index `I`.
pub fn build_dependency_graph<I>(f: &Fn) -> DepGraph<I>
where
    I: From<usize> + Into<usize> + Copy,
{
    let n = f.nodes.len();
    let mut operands_list: Vec<Vec<I>> = vec![Vec::new(); n];
    for (i, node) in f.nodes.iter().enumerate() {
        operands_list[i] = operands(&node.payload)
            .into_iter()
            .map(|r| I::from(r.index))
            .collect();
    }
    let mut users_list: Vec<Vec<(I, usize)>> = vec![Vec::new(); n];
    for (idx, ops) in operands_list.iter().enumerate() {
        for (slot, &d) in ops.iter().enumerate() {
            let du: usize = d.into();
            users_list[du].push((I::from(idx), slot));
        }
    }
    let mut nodes: Vec<DepNode<I>> = Vec::with_capacity(n);
    let mut local_hashes: Vec<FwdHash> = Vec::with_capacity(n);
    for i in 0..n {
        local_hashes.push(compute_node_local_structural_hash(f, NodeRef { index: i }));
    }
    let ret_index_usize: Option<usize> = f.ret_node_ref.map(|nr| nr.index);
    for i in 0..n {
        nodes.push(DepNode {
            operands: operands_list[i].clone(),
            users: users_list[i].clone(),
            structural_hash: local_hashes[i],
            is_return: ret_index_usize == Some(i),
            name: crate::ir::node_textual_id(f, NodeRef { index: i }),
        });
    }
    assert!(
        ret_index_usize.is_some(),
        "build_dependency_graph: function has no return node"
    );
    let return_value: I = I::from(ret_index_usize.unwrap());
    DepGraph {
        nodes,
        return_value,
    }
}

/// Computes forward-equivalent nodes between two functions by hashing each node
/// using an ordered combination of its operands' hashes. Two nodes are forward
/// equivalent if:
/// - Their local structure (operator, type, attributes) is identical, and
/// - For every operand position i, the i-th operands are themselves forward
///   equivalent.
///
/// Returns:
/// - A map from every node in `old` to the vector of forward-equivalent nodes
///   in `new` (possibly empty).
/// - A map from every node in `new` to the vector of forward-equivalent nodes
///   in `old` (possibly empty).
pub fn compute_forward_equivalences(
    old: &Fn,
    new: &Fn,
) -> (
    HashMap<OldNodeRef, Vec<NewNodeRef>>,
    HashMap<NewNodeRef, Vec<OldNodeRef>>,
) {
    // Helper: compute forward structural hashes for all nodes via topo order.
    fn compute_forward_hashes(f: &Fn) -> Vec<FwdHash> {
        let order = get_topological(f);
        let mut hashes: Vec<Option<FwdHash>> = vec![None; f.nodes.len()];
        for nr in order {
            let child_hashes: Vec<FwdHash> = operands(&f.get_node(nr).payload)
                .into_iter()
                .map(|c| hashes[c.index].expect("child hash must be computed first"))
                .collect();
            let h = compute_node_structural_hash(f, nr, &child_hashes);
            hashes[nr.index] = Some(h);
        }
        hashes
            .into_iter()
            .map(|o| o.expect("hash must be set"))
            .collect()
    }

    let old_hashes = compute_forward_hashes(old);
    let new_hashes = compute_forward_hashes(new);

    // Build reverse indices by hash.
    let mut by_hash_old: HashMap<FwdHash, Vec<OldNodeRef>> = HashMap::new();
    for (idx, h) in old_hashes.iter().enumerate() {
        by_hash_old.entry(*h).or_default().push(OldNodeRef(idx));
    }
    let mut by_hash_new: HashMap<FwdHash, Vec<NewNodeRef>> = HashMap::new();
    for (idx, h) in new_hashes.iter().enumerate() {
        by_hash_new.entry(*h).or_default().push(NewNodeRef(idx));
    }

    // Produce dense maps including keys for nodes with no equivalents (empty vecs).
    let mut old_to_new: HashMap<OldNodeRef, Vec<NewNodeRef>> = HashMap::new();
    for (idx, h) in old_hashes.iter().enumerate() {
        let mut v = by_hash_new.get(h).cloned().unwrap_or_default();
        v.sort_by_key(|nr| nr.0);
        old_to_new.insert(OldNodeRef(idx), v);
    }

    let mut new_to_old: HashMap<NewNodeRef, Vec<OldNodeRef>> = HashMap::new();
    for (idx, h) in new_hashes.iter().enumerate() {
        let mut v = by_hash_old.get(h).cloned().unwrap_or_default();
        v.sort_by_key(|nr| nr.0);
        new_to_old.insert(NewNodeRef(idx), v);
    }

    (old_to_new, new_to_old)
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
                old_index: OldNodeRef(oi),
                new_index: NewNodeRef(ni),
                new_operands: Vec::new(),
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
                    old_index: OldNodeRef(node.index),
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
                    new_index: NewNodeRef(node.index),
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

/// Computes an edit set required to transform `old` into `new` using the
/// provided selector.
pub fn compute_function_edit<S: MatchSelector>(
    old: &Fn,
    new: &Fn,
    selector: &mut S,
) -> Result<IrEditSet, String> {
    let matches = compute_function_match(old, new, selector)?;
    Ok(convert_match_set_to_edit_set(old, new, &matches))
}

/// Computes the match actions required to transform `old` into `new`, using an
/// externally provided selector that controls priority and edit choice.
pub fn compute_function_match<S: MatchSelector>(
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
    let old_graph = build_dependency_graph::<OldNodeRef>(old);
    let new_graph = build_dependency_graph::<NewNodeRef>(new);

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
                let u: usize = user.into();
                if old_remain[u] > 0 {
                    old_remain[u] -= 1;
                    if old_remain[u] == 0 {
                        selector.add_ready_node(ReadyNode { side, index: u });
                    }
                }
            }
        }
        NodeSide::New => {
            for &(user, _slot) in new_graph.nodes[idx].users.iter() {
                let u: usize = user.into();
                if new_remain[u] > 0 {
                    new_remain[u] -= 1;
                    if new_remain[u] == 0 {
                        selector.add_ready_node(ReadyNode { side, index: u });
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
                update_ready(index.into(), NodeSide::Old, selector);
                selector.update_after_match(&edit);
            }
            MatchAction::AddNode {
                new_index: index, ..
            } => {
                let is_ret = new
                    .ret_node_ref
                    .map(|nr| nr.index)
                    .map_or(false, |ri| ri == usize::from(index));
                matches.push(MatchAction::AddNode {
                    new_index: index,
                    is_return: is_ret,
                });
                update_ready(index.into(), NodeSide::New, selector);
                selector.update_after_match(&edit);
            }
            MatchAction::MatchNodes {
                old_index,
                new_index,
                ..
            } => {
                // Record match edit with explicit new_operands and return flag.
                let ni: usize = new_index.into();
                let new_operands: Vec<NewNodeRef> = operands(&new.nodes[ni].payload)
                    .into_iter()
                    .map(|r| NewNodeRef(r.index))
                    .collect();
                matches.push(MatchAction::MatchNodes {
                    old_index,
                    new_index,
                    new_operands,
                    is_new_return: new.ret_node_ref.map(|nr| nr.index)
                        == Some(usize::from(new_index)),
                });
                // Propagate readiness in old/new graphs.
                update_ready(old_index.into(), NodeSide::Old, selector);
                update_ready(new_index.into(), NodeSide::New, selector);
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

    for e in edits.edits.iter() {
        match e.clone() {
            IrEdit::DeleteNode { node } => {
                if node.index >= patched.nodes.len() {
                    return Err(format!("DeleteNode index {} out of bounds", node.index));
                }
                patched.nodes[node.index].payload = crate::ir::NodePayload::Nil;
            }
            IrEdit::AddNode { node } => {
                // Payloads in AddNode must already use old or previously-added patched indices.
                patched.nodes.push(node);
            }
            IrEdit::SubstituteOperand {
                node,
                operand_slot,
                new_operand,
            } => {
                let idx = node.index;
                if idx >= patched.nodes.len() {
                    return Err(format!(
                        "SubstituteOperand user node index {} out of bounds",
                        idx
                    ));
                }
                // Validate slot
                let op_count = operands(&patched.nodes[idx].payload).len();
                if operand_slot >= op_count {
                    return Err(format!(
                        "SubstituteOperand slot {} out of bounds for node {} ({} operands)",
                        operand_slot, idx, op_count
                    ));
                }
                // Remap exactly the requested operand slot to new_operand, preserving others.
                let mut seen: usize = 0;
                let remapped_payload =
                    remap_payload_with(&patched.nodes[idx].payload, |nr: NodeRef| {
                        let out = if seen == operand_slot {
                            new_operand
                        } else {
                            nr
                        };
                        seen += 1;
                        out
                    });
                patched.nodes[idx].payload = remapped_payload;
            }
            IrEdit::SetReturn { node } => {
                patched.ret_node_ref = Some(NodeRef { index: node.index });
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
            ..
        } = action
        {
            new_to_old.insert((*new_index).into(), (*old_index).into());
        }
    }

    let mut edits: Vec<IrEdit> = Vec::new();
    let original_old_len = old.nodes.len();
    // Map from original-new indices to the eventual patched indices for added
    // nodes, assigned sequentially starting at original_old_len in the order
    // AddNode edits are emitted.
    let mut added_new_to_patched: HashMap<usize, usize> = HashMap::new();
    let mut add_count: usize = 0;
    for action in m.matches.iter() {
        match action {
            MatchAction::AddNode {
                new_index,
                is_return: _,
            } => {
                // Clone new node and remap operands that refer to matched new nodes to their
                // corresponding old indices. References to other new nodes must refer to
                // previously added nodes; remap those to their patched indices.
                let ni: usize = (*new_index).into();
                let src = &new.nodes[ni];
                let remapped_payload = remap_payload_with(&src.payload, |nr: NodeRef| {
                    if let Some(&old_idx) = new_to_old.get(&nr.index) {
                        NodeRef { index: old_idx }
                    } else {
                        if let Some(&patched_idx) = added_new_to_patched.get(&nr.index) {
                            NodeRef { index: patched_idx }
                        } else {
                            panic!(
                                "AddNode payload references future new node {} which is not yet added",
                                nr.index
                            );
                        }
                    }
                });
                let cloned = crate::ir::Node {
                    text_id: src.text_id,
                    name: src.name.clone(),
                    ty: src.ty.clone(),
                    payload: remapped_payload,
                    pos: src.pos.clone(),
                };
                // Assign patched index for this added node per protocol
                let patched_index = original_old_len + add_count;
                added_new_to_patched.insert(ni, patched_index);
                add_count += 1;
                edits.push(IrEdit::AddNode { node: cloned });
            }
            MatchAction::DeleteNode { old_index } => {
                let oi: usize = (*old_index).into();
                edits.push(IrEdit::DeleteNode {
                    node: NodeRef { index: oi },
                });
            }
            MatchAction::MatchNodes {
                old_index,
                new_index: _,
                new_operands,
                is_new_return: _,
            } => {
                // Infer operand substitutions using provided new_operands.
                let oi: usize = (*old_index).into();
                let old_ops = operands(&old.nodes[oi].payload);
                if old_ops.len() == new_operands.len() {
                    for (slot, (o_ref, n_new_ref)) in
                        old_ops.iter().zip(new_operands.iter()).enumerate()
                    {
                        let nidx: usize = (*n_new_ref).into();
                        if let Some(&mapped_old_target) = new_to_old.get(&nidx) {
                            if o_ref.index != mapped_old_target {
                                edits.push(IrEdit::SubstituteOperand {
                                    node: NodeRef { index: oi },
                                    operand_slot: slot,
                                    new_operand: NodeRef {
                                        index: mapped_old_target,
                                    },
                                });
                            }
                        } else if let Some(&patched_idx) = added_new_to_patched.get(&nidx) {
                            if o_ref.index != patched_idx {
                                edits.push(IrEdit::SubstituteOperand {
                                    node: NodeRef { index: oi },
                                    operand_slot: slot,
                                    new_operand: NodeRef { index: patched_idx },
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
            // Only emit SetReturn if the old function does not already return this node.
            if old.ret_node_ref.map(|r| r.index) != Some(old_idx) {
                edits.push(IrEdit::SetReturn {
                    node: NodeRef { index: old_idx },
                });
            }
        } else {
            // Return refers to a newly added node; map to its patched index assigned above.
            let patched_idx = *added_new_to_patched
                .get(&nr.index)
                .expect("SetReturn refers to a new node that was not added");
            edits.push(IrEdit::SetReturn {
                node: NodeRef { index: patched_idx },
            });
        }
    }
    IrEditSet { edits }
}

/// Formats a single IrEdit into a human-friendly string using names from `old`.
pub fn format_ir_edit(old: &Fn, e: &IrEdit) -> String {
    let name_for_index = |idx: usize| -> String {
        if idx < old.nodes.len() {
            ir::node_textual_id(old, NodeRef { index: idx })
        } else {
            format!("added.{}", idx)
        }
    };
    match e.clone() {
        IrEdit::DeleteNode { node } => {
            format!("DeleteNode: {}", name_for_index(node.index))
        }
        IrEdit::AddNode { node } => {
            // Use the same identifier style as DeleteNode: name if present, else
            // "op.text_id".
            let added_name = node
                .name
                .clone()
                .unwrap_or_else(|| format!("{}.{}", node.payload.get_operator(), node.text_id));
            format!("AddNode: {}", added_name)
        }
        IrEdit::SubstituteOperand {
            node,
            operand_slot,
            new_operand,
        } => {
            let node_idx = node.index;
            let payload = &old.nodes[node_idx].payload;
            let op = payload.get_operator();
            let orig_ops = operands(payload);
            let orig_name = if operand_slot < orig_ops.len() {
                name_for_index(orig_ops[operand_slot].index)
            } else {
                format!("<slot {} oob>", operand_slot)
            };
            let rendered = orig_ops
                .iter()
                .enumerate()
                .map(|(i, nr)| {
                    if i == operand_slot {
                        format!("**{}**", name_for_index(new_operand.index))
                    } else {
                        name_for_index(nr.index)
                    }
                })
                .collect::<Vec<String>>()
                .join(", ");
            format!(
                "SubstituteOperand: {}({}), was {}",
                ir::node_textual_id(old, NodeRef { index: node_idx }),
                rendered,
                orig_name
            )
        }
        IrEdit::SetReturn { node } => {
            format!("SetReturn: {}", name_for_index(node.index))
        }
    }
}

/// Formats an IrEditSet into a multiline string using names from `old`.
pub fn format_ir_edits(old: &Fn, edits: &IrEditSet) -> String {
    let mut lines: Vec<String> = Vec::new();
    for e in edits.edits.iter() {
        lines.push(format_ir_edit(old, e));
    }
    lines.join("\n")
}

/// Formats a single MatchAction into a human-friendly string using names from
/// both `old` and `new` functions.
pub fn format_match_action<F>(m: &MatchAction, to_string: F) -> String
where
    F: std::ops::Fn(NodeSide, usize) -> String,
{
    match m.clone() {
        MatchAction::DeleteNode { old_index } => {
            format!("DeleteNode: {}", to_string(NodeSide::Old, old_index.0))
        }
        MatchAction::AddNode {
            new_index,
            is_return,
        } => {
            let s = to_string(NodeSide::New, new_index.0);
            if is_return {
                format!("AddNode (ret): {}", s)
            } else {
                format!("AddNode: {}", s)
            }
        }
        MatchAction::MatchNodes {
            old_index,
            new_index,
            new_operands: _,
            is_new_return,
        } => {
            let old_s = to_string(NodeSide::Old, old_index.0);
            let new_s = to_string(NodeSide::New, new_index.0);
            if is_new_return {
                format!("MatchNodes (ret): {} <-> {}", old_s, new_s)
            } else {
                format!("MatchNodes: {} <-> {}", old_s, new_s)
            }
        }
    }
}

/// Formats a Match set into a multiline string using names from the provided
/// `old` and `new` functions.
pub fn format_match_set(old: &Fn, new: &Fn, ms: &IrMatchSet) -> String {
    let mut lines: Vec<String> = Vec::new();
    for m in ms.matches.iter() {
        lines.push(format_match_action(m, |side, idx| match side {
            NodeSide::Old => ir::node_textual_id(old, NodeRef { index: idx }),
            NodeSide::New => ir::node_textual_id(new, NodeRef { index: idx }),
        }));
    }
    lines.join("\n")
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

        let mut selector = NaiveMatchSelector::new(lhs, rhs);
        let edits = compute_function_edit(lhs, rhs, &mut selector).unwrap();
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

        let mut selector = NaiveMatchSelector::new(lhs, rhs);
        let result = compute_function_edit(lhs, rhs, &mut selector);
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

        let mut selector = crate::greedy_graph_edit::GreedyMatchSelector::new(old_fn, new_fn);
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

        let mut selector = NaiveMatchSelector::new(old_fn, new_fn);
        let edits = compute_function_edit(old_fn, new_fn, &mut selector).unwrap();
        let patched = apply_function_edits(old_fn, &edits).unwrap();
        assert!(crate::ir_isomorphism::is_ir_isomorphic(&patched, new_fn));
    }

    #[test]
    fn forward_equivalence_self_maps_each_node_to_itself() {
        let pkg = parse_ir_from_string(
            r#"package p
            top fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
                lit: bits[8] = literal(value=3, id=10)
                sum: bits[8] = add(a, lit, id=11)
                ret idnode: bits[8] = identity(sum, id=12)
            }
            "#,
        );
        let f = pkg.get_top().unwrap();

        let (old_to_new, new_to_old) = compute_forward_equivalences(f, f);

        for i in 0..f.nodes.len() {
            let oi = OldNodeRef(i);
            let ni = NewNodeRef(i);
            let targets = old_to_new.get(&oi).expect("key must exist");
            assert!(targets.contains(&ni), "node {} not equivalent to itself", i);

            let sources = new_to_old.get(&ni).expect("key must exist");
            assert!(
                sources.contains(&oi),
                "node {} not reverse-equivalent to itself",
                i
            );
        }
    }

    #[test]
    fn forward_equivalence_respects_operand_order_mismatch() {
        // Old: sum = add(a, b); New: sum = add(b, a)
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
                ret sum: bits[8] = add(a, b, id=10)
            }
            "#,
        );
        let old_fn = pkg_old.get_top().unwrap();

        let pkg_new = parse_ir_from_string(
            r#"package p2
            top fn f2(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
                ret sum: bits[8] = add(b, a, id=10)
            }
            "#,
        );
        let new_fn = pkg_new.get_top().unwrap();

        let find_named = |f: &crate::ir::Fn, name: &str| -> usize {
            f.nodes
                .iter()
                .position(|n| n.name.as_deref() == Some(name))
                .expect("node by name not found")
        };

        let (old_to_new, new_to_old) = compute_forward_equivalences(old_fn, new_fn);

        // Params map to themselves by name/ordinal.
        let a_old = OldNodeRef(find_named(old_fn, "a"));
        let b_old = OldNodeRef(find_named(old_fn, "b"));
        let a_new = NewNodeRef(find_named(new_fn, "a"));
        let b_new = NewNodeRef(find_named(new_fn, "b"));
        assert!(old_to_new.get(&a_old).unwrap().contains(&a_new));
        assert!(old_to_new.get(&b_old).unwrap().contains(&b_new));

        // The add nodes differ by operand order, so they are not forward equivalent.
        let sum_old = OldNodeRef(find_named(old_fn, "sum"));
        let sum_new = NewNodeRef(find_named(new_fn, "sum"));
        assert!(!old_to_new.get(&sum_old).unwrap().contains(&sum_new));
        assert!(!new_to_old.get(&sum_new).unwrap().contains(&sum_old));
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

        let mut selector = NaiveMatchSelector::new(old_fn, new_fn);
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

        let mut selector = NaiveMatchSelector::new(old_fn, new_fn);
        let edits = compute_function_edit(old_fn, new_fn, &mut selector).unwrap();
        let patched = apply_function_edits(old_fn, &edits).unwrap();
        assert!(crate::ir_isomorphism::is_ir_isomorphic(&patched, new_fn));
    }
}
