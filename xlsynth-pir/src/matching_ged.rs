// SPDX-License-Identifier: Apache-2.0

//! Library for computing heuristic graph edit distance (GED) between two XLS IR
//! functions using an incremental node matching approach.

use crate::ir::{self, BlockPortInfo, Fn, Node, NodeRef};
use crate::ir_utils::{compact_and_toposort_in_place, operands, remap_payload_with};
use crate::node_hashing::{FwdHash, compute_node_local_structural_hash};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum IrEdit {
    /// Add a node to the graph.
    AddNode { node: Node },
    /// Delete a node from the graph.
    DeleteNode { node: NodeRef },
    /// Substitute an operand of a node with a different target node.
    SubstituteOperand {
        node: NodeRef,
        operand_slot: usize,
        new_operand: NodeRef,
    },
    /// Sets the return value of the function.
    SetReturn { node: NodeRef },
}

#[derive(Debug, Clone, Default)]
pub struct IrEditSet {
    pub edits: Vec<IrEdit>,
}

/// Represents a single incremental match decision in the matching process.
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
        new_operands: Vec<NewNodeRef>,
        is_new_return: bool,
    },
}

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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IrMatchSet {
    pub matches: Vec<MatchAction>,
}
/// Abstraction representing a node in the IR graph distilling essential
/// dependency and structural information.
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

/// Abstraction for prioritizing and selecting match actions.

pub trait MatchSelector {
    /// Adds a node that has become ready to be handled.
    fn add_ready_node(&mut self, node: ReadyNode);

    /// Selects the next match action to apply, or `None` if none remain.
    fn select_next_match(&mut self) -> Option<MatchAction>;

    /// Notifies the selector that a match action has been observed/applied so
    /// it can update any internal state (e.g., statistics, heuristics).
    fn update_after_match(&mut self, _edit: &MatchAction) {}
}

/// Naive selector which matches parameter nodes but uses add/remove for all
/// other nodes.
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
pub fn compute_fn_edit<S: MatchSelector>(
    old: &Fn,
    new: &Fn,
    selector: &mut S,
) -> Result<IrEditSet, String> {
    let matches = compute_fn_match(old, new, selector)?;
    Ok(convert_match_set_to_edit_set(old, new, &matches, None))
}

/// Computes an edit set required to transform `old` Block into `new` Block
/// using the provided selector. Operates on the internal function of the block.
pub fn compute_block_edit<S: MatchSelector>(
    old: &crate::ir::PackageMember,
    new: &crate::ir::PackageMember,
    selector: &mut S,
) -> Result<IrEditSet, String> {
    match (old, new) {
        (
            crate::ir::PackageMember::Block {
                func: old_fn,
                port_info: old_port_info,
                ..
            },
            crate::ir::PackageMember::Block { func: new_fn, .. },
        ) => {
            let matches = compute_fn_match(old_fn, new_fn, selector)?;
            Ok(convert_match_set_to_edit_set(
                old_fn,
                new_fn,
                &matches,
                Some(old_port_info),
            ))
        }
        _ => Err("compute_block_edit requires Block package members".to_string()),
    }
}

/// Computes the match actions required to transform `old` into `new`, using an
/// externally provided selector that controls priority and edit choice.
pub fn compute_fn_match<S: MatchSelector>(
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

/// Computes the match actions required to transform `old` Block into `new`
/// Block, using an externally provided selector that controls priority and edit
/// choice. Operates on the internal function of the block.
pub fn compute_block_match<S: MatchSelector>(
    old: &crate::ir::PackageMember,
    new: &crate::ir::PackageMember,
    selector: &mut S,
) -> Result<IrMatchSet, String> {
    match (old, new) {
        (
            crate::ir::PackageMember::Block { func: old_fn, .. },
            crate::ir::PackageMember::Block { func: new_fn, .. },
        ) => compute_fn_match(old_fn, new_fn, selector),
        _ => Err("compute_block_match requires Block package members".to_string()),
    }
}

/// Applies a sequence of IrEdits to `old` and returns the result.
pub fn apply_fn_edits(old: &Fn, edits: &IrEditSet) -> Result<Fn, String> {
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
                let remapped_payload = remap_payload_with(
                    &patched.nodes[idx].payload,
                    |(slot, nr): (usize, NodeRef)| {
                        if slot == operand_slot {
                            new_operand
                        } else {
                            nr
                        }
                    },
                );
                patched.nodes[idx].payload = remapped_payload;
            }
            IrEdit::SetReturn { node } => {
                patched.ret_node_ref = Some(NodeRef { index: node.index });
            }
        }
    }
    compact_and_toposort_in_place(&mut patched)?;
    Ok(patched)
}

/// Applies a sequence of IrEdits to a Block and returns a new Block preserving
/// the original Block metadata (e.g., port info). Operates on the internal
/// function of the block.
pub fn apply_block_edits(
    old: &crate::ir::PackageMember,
    edits: &IrEditSet,
) -> Result<crate::ir::PackageMember, String> {
    match old {
        crate::ir::PackageMember::Block {
            func: old_fn,
            port_info,
        } => {
            let new_fn = apply_fn_edits(old_fn, edits)?;
            Ok(crate::ir::PackageMember::Block {
                func: new_fn,
                port_info: port_info.clone(),
            })
        }
        _ => Err("apply_block_edits requires a Block package member".to_string()),
    }
}

/// Converts a set of `MatchAction`s applied to a old/new pair into IrEdits on
/// old which produce a graph isomorphic to new.
pub fn convert_match_set_to_edit_set(
    old: &Fn,
    new: &Fn,
    m: &IrMatchSet,
    old_port_info: Option<&BlockPortInfo>,
) -> IrEditSet {
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

    // Node ids of new nodes will have to be remapped to avoid collisions.
    let mut next_text_id = 0usize;
    for node in old.nodes.iter() {
        next_text_id = std::cmp::max(next_text_id, node.text_id + 1);
    }
    if let Some(pi) = old_port_info {
        for port in pi.input_port_ids.iter() {
            next_text_id = std::cmp::max(next_text_id, port.1 + 1);
        }
        for port in pi.output_port_ids.iter() {
            next_text_id = std::cmp::max(next_text_id, port.1 + 1);
        }
    }
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
                let remapped_payload = remap_payload_with(
                    &src.payload,
                    |(_, nr): (usize, NodeRef)| {
                        if let Some(&old_idx) = new_to_old.get(&nr.index) {
                            NodeRef { index: old_idx }
                        } else if let Some(&patched_idx) = added_new_to_patched.get(&nr.index) {
                            NodeRef { index: patched_idx }
                        } else {
                            panic!(
                                "AddNode payload references future new node {} which is not yet added",
                                nr.index
                            );
                        }
                    },
                );
                let cloned = crate::ir::Node {
                    text_id: next_text_id,
                    name: src.name.clone(),
                    ty: src.ty.clone(),
                    payload: remapped_payload,
                    pos: src.pos.clone(),
                };
                // Assign patched index for this added node per protocol
                let patched_index = original_old_len + add_count;
                added_new_to_patched.insert(ni, patched_index);
                next_text_id += 1;
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
pub fn format_match_action<F>(m: &MatchAction, node_to_string: F) -> String
where
    F: std::ops::Fn(NodeSide, usize) -> String,
{
    match m.clone() {
        MatchAction::DeleteNode { old_index } => {
            format!("DeleteNode: {}", node_to_string(NodeSide::Old, old_index.0))
        }
        MatchAction::AddNode {
            new_index,
            is_return,
        } => {
            let s = node_to_string(NodeSide::New, new_index.0);
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
            let old_s = node_to_string(NodeSide::Old, old_index.0);
            let new_s = node_to_string(NodeSide::New, new_index.0);
            if is_new_return {
                format!("MatchNodes (ret): {} <-> {}", old_s, new_s)
            } else {
                format!("MatchNodes: {} <-> {}", old_s, new_s)
            }
        }
    }
}

/// Formats a sequence of MatchAction values into a multiline string using
/// names from the provided `old` and `new` functions.
pub fn format_match_actions(old: &Fn, new: &Fn, actions: &[MatchAction]) -> String {
    let lines: Vec<String> = actions
        .iter()
        .map(|m| {
            format_match_action(m, |side, idx| match side {
                NodeSide::Old => ir::node_textual_id(old, NodeRef { index: idx }),
                NodeSide::New => ir::node_textual_id(new, NodeRef { index: idx }),
            })
        })
        .collect();
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
        let lhs = pkg.get_top_fn().unwrap();

        let pkg2 = parse_ir_from_string(
            r#"package p2
            top fn f2(x: bits[8]) -> bits[8] {
                ret identity.2: bits[8] = identity(x, id=2)
            }
            "#,
        );
        let rhs = pkg2.get_top_fn().unwrap();

        let mut selector = NaiveMatchSelector::new(lhs, rhs);
        let edits = compute_fn_edit(lhs, rhs, &mut selector).unwrap();
        assert!(!edits.edits.is_empty());
        let patched = apply_fn_edits(lhs, &edits).unwrap();
        assert!(crate::node_hashing::functions_structurally_equivalent(
            &patched, rhs
        ));
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
        let lhs = pkg.get_top_fn().unwrap();

        let pkg2 = parse_ir_from_string(
            r#"package p2
            top fn f2(x: bits[9]) -> bits[9] {
                ret identity.2: bits[9] = identity(x, id=2)
            }
            "#,
        );
        let rhs = pkg2.get_top_fn().unwrap();

        let mut selector = NaiveMatchSelector::new(lhs, rhs);
        let result = compute_fn_edit(lhs, rhs, &mut selector);
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
        let f = pkg.get_top_fn().unwrap();
        let edits = IrEditSet::default();
        let applied = apply_fn_edits(f, &edits).unwrap();
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
        let old_fn = pkg_old.get_top_fn().unwrap();

        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f() -> bits[8] {
                ret literal.101: bits[8] = literal(value=1, id=101)
            }
            "#,
        );
        let new_fn = pkg_new.get_top_fn().unwrap();

        let mut selector = crate::greedy_matching_ged::GreedyMatchSelector::new(old_fn, new_fn);
        let edits = compute_fn_edit(old_fn, new_fn, &mut selector).unwrap();
        let patched = apply_fn_edits(old_fn, &edits).unwrap();
        assert!(crate::node_hashing::functions_structurally_equivalent(
            &patched, new_fn
        ));
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
        let old_fn = pkg_old.get_top_fn().unwrap();

        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f() -> bits[1] {
                literal.11: bits[1] = literal(value=1, id=11)
                ret identity.22: bits[1] = identity(literal.11, id=22)
            }
            "#,
        );
        let new_fn = pkg_new.get_top_fn().unwrap();

        let mut selector = NaiveMatchSelector::new(old_fn, new_fn);
        let edits = compute_fn_edit(old_fn, new_fn, &mut selector).unwrap();
        let patched = apply_fn_edits(old_fn, &edits).unwrap();
        assert!(crate::node_hashing::functions_structurally_equivalent(
            &patched, new_fn
        ));
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
        let old_fn = pkg_old.get_top_fn().unwrap();

        // Same nodes, but return points to a different node (the literal).
        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f() -> bits[1] {
                ret literal.1: bits[1] = literal(value=1, id=1)
                identity.2: bits[1] = identity(literal.1, id=2)
            }
            "#,
        );
        let new_fn = pkg_new.get_top_fn().unwrap();

        let mut selector = NaiveMatchSelector::new(old_fn, new_fn);
        let edits = compute_fn_edit(old_fn, new_fn, &mut selector).unwrap();
        let patched = apply_fn_edits(old_fn, &edits).unwrap();
        assert!(crate::node_hashing::functions_structurally_equivalent(
            &patched, new_fn
        ));
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
        let old_fn = pkg_old.get_top_fn().unwrap();

        // Different structure: wrap the add in an identity, keep params identical.
        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
                add.103: bits[8] = add(a, b, id=103)
                ret identity.104: bits[8] = identity(add.103, id=104)
            }
            "#,
        );
        let new_fn = pkg_new.get_top_fn().unwrap();

        let mut selector = NaiveMatchSelector::new(old_fn, new_fn);
        let edits = compute_fn_edit(old_fn, new_fn, &mut selector).unwrap();
        let patched = apply_fn_edits(old_fn, &edits).unwrap();
        assert!(crate::node_hashing::functions_structurally_equivalent(
            &patched, new_fn
        ));
    }
}
