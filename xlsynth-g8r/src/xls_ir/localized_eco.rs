// SPDX-License-Identifier: Apache-2.0

//! Computes a localized ECO (engineering change order) diff between two XLS IR
//! functions. The goal is to produce a minimal, path-localized set of edits to
//! transform "old" into the behavior of "new", focusing on structural
//! similarity with ordered operands.

use serde::Serialize;
use std::collections::HashSet;

use crate::xls_ir::ir::{Fn, NodePayload, NodeRef};
use crate::xls_ir::ir::{binop_to_operator, nary_op_to_operator, unop_to_operator};
use crate::xls_ir::ir_utils::get_topological;
use crate::xls_ir::node_hashing::{
    compute_node_local_structural_hash, compute_node_structural_hash, get_param_ordinal,
};

fn children_in_order(f: &Fn, node_ref: NodeRef) -> Vec<NodeRef> {
    let node = f.get_node(node_ref);
    match &node.payload {
        NodePayload::Nil => Vec::new(),
        NodePayload::GetParam(_) => Vec::new(),
        NodePayload::Tuple(elems)
        | NodePayload::Array(elems)
        | NodePayload::AfterAll(elems)
        | NodePayload::Nary(_, elems) => elems.clone(),
        NodePayload::TupleIndex { tuple, .. }
        | NodePayload::Unop(_, tuple)
        | NodePayload::Decode { arg: tuple, .. }
        | NodePayload::Encode { arg: tuple }
        | NodePayload::OneHot { arg: tuple, .. }
        | NodePayload::BitSlice { arg: tuple, .. } => vec![*tuple],
        NodePayload::Binop(_, a, b) => vec![*a, *b],
        NodePayload::SignExt { arg, .. } | NodePayload::ZeroExt { arg, .. } => vec![*arg],
        NodePayload::ArrayUpdate {
            array,
            value,
            indices,
            ..
        } => {
            let mut v = Vec::with_capacity(2 + indices.len());
            v.push(*array);
            v.push(*value);
            v.extend(indices.iter().copied());
            v
        }
        NodePayload::ArrayIndex { array, indices, .. } => {
            let mut v = Vec::with_capacity(1 + indices.len());
            v.push(*array);
            v.extend(indices.iter().copied());
            v
        }
        NodePayload::DynamicBitSlice { arg, start, .. } => vec![*arg, *start],
        NodePayload::BitSliceUpdate {
            arg,
            start,
            update_value,
        } => vec![*arg, *start, *update_value],
        NodePayload::Assert {
            token, activate, ..
        } => vec![*token, *activate],
        NodePayload::Trace {
            token,
            activated,
            operands,
            ..
        } => {
            let mut v = Vec::with_capacity(2 + operands.len());
            v.push(*token);
            v.push(*activated);
            v.extend(operands.iter().copied());
            v
        }
        NodePayload::Invoke { operands, .. } => operands.clone(),
        NodePayload::PrioritySel {
            selector,
            cases,
            default,
        }
        | NodePayload::Sel {
            selector,
            cases,
            default,
        } => {
            let mut v = Vec::with_capacity(1 + cases.len() + if default.is_some() { 1 } else { 0 });
            v.push(*selector);
            v.extend(cases.iter().copied());
            if let Some(d) = default {
                v.push(*d);
            }
            v
        }
        NodePayload::OneHotSel { selector, cases } => {
            let mut v = Vec::with_capacity(1 + cases.len());
            v.push(*selector);
            v.extend(cases.iter().copied());
            v
        }
        NodePayload::CountedFor {
            init,
            invariant_args,
            ..
        } => {
            let mut v = Vec::with_capacity(1 + invariant_args.len());
            v.push(*init);
            v.extend(invariant_args.iter().copied());
            v
        }
        NodePayload::Literal(_) => Vec::new(),
        NodePayload::Cover { predicate, .. } => vec![*predicate],
    }
}

fn collect_shape_hashes(f: &Fn) -> Vec<blake3::Hash> {
    let order = get_topological(f);
    let n = f.nodes.len();
    let mut hashes: Vec<blake3::Hash> = vec![blake3::Hash::from([0u8; 32]); n];
    for nr in order {
        let children = children_in_order(f, nr);
        let mut child_hashes: Vec<blake3::Hash> = Vec::with_capacity(children.len());
        for c in children.iter() {
            child_hashes.push(hashes[c.index]);
        }
        let h = compute_node_structural_hash(f, nr, &child_hashes);
        hashes[nr.index] = h;
    }
    hashes
}

fn collect_local_shape_hashes(f: &Fn) -> Vec<blake3::Hash> {
    let n = f.nodes.len();
    let mut hashes: Vec<blake3::Hash> = vec![blake3::Hash::from([0u8; 32]); n];
    for i in 0..n {
        let nr = NodeRef { index: i };
        hashes[i] = compute_node_local_structural_hash(f, nr);
    }
    hashes
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Path(pub Vec<usize>);

#[derive(Debug, Clone, Serialize)]
pub enum Edit {
    OperatorChanged {
        path: Path,
        old_op: String,
        new_op: String,
    },
    TypeChanged {
        path: Path,
        old_ty: String,
        new_ty: String,
    },
    ArityChanged {
        path: Path,
        old_arity: usize,
        new_arity: usize,
    },
    OperandInserted {
        path: Path,
        index: usize,
        new_signature: String,
    },
    OperandDeleted {
        path: Path,
        index: usize,
        old_signature: String,
    },
    OperandSubstituted {
        path: Path,
        index: usize,
        old_signature: String,
        new_signature: String,
    },
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct DiffStats {
    pub matched_nodes: usize,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct LocalizedEcoDiff {
    pub edits: Vec<Edit>,
    pub stats: DiffStats,
}

/// Computes a localized ECO diff: a set of path-addressed edits that, when
/// applied to `old`, yield the structure of `new`.
pub fn compute_localized_eco(old: &Fn, new: &Fn) -> LocalizedEcoDiff {
    assert_eq!(
        old.get_type(),
        new.get_type(),
        "Function signatures must match for ECO diff"
    );

    let old_hashes = collect_shape_hashes(old);
    let new_hashes = collect_shape_hashes(new);
    let old_local = collect_local_shape_hashes(old);
    let new_local = collect_local_shape_hashes(new);

    let mut edits: Vec<Edit> = Vec::new();
    let mut stats = DiffStats { matched_nodes: 0 };
    let mut visited_pairs: HashSet<(usize, usize)> = HashSet::new();

    let old_root = old
        .ret_node_ref
        .expect("old function must have a return node");
    let new_root = new
        .ret_node_ref
        .expect("new function must have a return node");

    align_nodes(
        old,
        new,
        &old_hashes,
        &new_hashes,
        &old_local,
        &new_local,
        old_root,
        new_root,
        &mut Vec::new(),
        &mut edits,
        &mut stats,
        &mut visited_pairs,
    );

    LocalizedEcoDiff { edits, stats }
}

/// Collects pairs of (old_idx, new_idx) whose subgraphs are structurally equal
/// (shape-hash equal), discovered via a DAG-aware operand alignment walk.
fn collect_equal_pairs(
    old: &Fn,
    new: &Fn,
    old_hashes: &[blake3::Hash],
    new_hashes: &[blake3::Hash],
    old_local: &[blake3::Hash],
    new_local: &[blake3::Hash],
    old_nr: NodeRef,
    new_nr: NodeRef,
    visited: &mut std::collections::HashSet<(usize, usize)>,
    out: &mut std::collections::HashSet<(usize, usize)>,
) {
    let key = (old_nr.index, new_nr.index);
    if visited.contains(&key) {
        return;
    }
    visited.insert(key);

    if old_hashes[old_nr.index] == new_hashes[new_nr.index] {
        out.insert((old_nr.index, new_nr.index));
        // When equal at parent, children arrays are the same length and equal by
        // position. Recurse pairwise.
        let old_children = children_in_order(old, old_nr);
        let new_children = children_in_order(new, new_nr);
        for (oc, nc) in old_children.iter().zip(new_children.iter()) {
            collect_equal_pairs(
                old, new, old_hashes, new_hashes, old_local, new_local, *oc, *nc, visited, out,
            );
        }
        return;
    }

    // Otherwise align children to find equal subgraphs below.
    let old_children = children_in_order(old, old_nr);
    let new_children = children_in_order(new, new_nr);
    let n = old_children.len();
    let m = new_children.len();
    let mut dp: Vec<Vec<usize>> = vec![vec![0; m + 1]; n + 1];
    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let equal =
                old_hashes[old_children[i - 1].index] == new_hashes[new_children[j - 1].index];
            let cost_sub = if equal { 0 } else { 1 };
            let del = dp[i - 1][j] + 1;
            let ins = dp[i][j - 1] + 1;
            let sub = dp[i - 1][j - 1] + cost_sub;
            dp[i][j] = del.min(ins).min(sub);
        }
    }
    // Backtrace: recurse on all diagonal moves to continue discovery of equal pairs
    // below.
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0 {
        if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            i -= 1;
        } else if j > 0 && dp[i][j] == dp[i][j - 1] + 1 {
            j -= 1;
        } else {
            let idx_old = i - 1;
            let idx_new = j - 1;
            let oc = old_children[idx_old];
            let nc = new_children[idx_new];
            // Recurse on this pair regardless of equality to find deeper equalities.
            collect_equal_pairs(
                old, new, old_hashes, new_hashes, old_local, new_local, oc, nc, visited, out,
            );
            i -= 1;
            j -= 1;
        }
    }
}

fn node_ref_at_path(f: &Fn, start: NodeRef, path: &[usize]) -> NodeRef {
    let mut nr = start;
    for &idx in path.iter() {
        let children = children_in_order(f, nr);
        assert!(idx < children.len(), "path index out of bounds");
        nr = children[idx];
    }
    nr
}

fn import_subtree(
    patched: &mut Fn,
    old: &Fn,
    new: &Fn,
    new_idx_to_old_idx: &std::collections::HashMap<usize, usize>,
    memo: &mut std::collections::HashMap<usize, NodeRef>,
    next_id: &mut usize,
    new_nr: NodeRef,
) -> NodeRef {
    if let Some(&existing) = memo.get(&new_nr.index) {
        return existing;
    }
    if let Some(&old_idx) = new_idx_to_old_idx.get(&new_nr.index) {
        let nr = NodeRef { index: old_idx };
        memo.insert(new_nr.index, nr);
        return nr;
    }
    let new_node = new.get_node(new_nr);
    let map_child = |nr_new: NodeRef,
                     patched: &mut Fn,
                     memo: &mut std::collections::HashMap<usize, NodeRef>,
                     next_id: &mut usize| {
        import_subtree(patched, old, new, new_idx_to_old_idx, memo, next_id, nr_new)
    };
    let payload = match &new_node.payload {
        NodePayload::Nil => NodePayload::Nil,
        NodePayload::GetParam(pid_new) => {
            // Map by ordinal to old param id to preserve param identities.
            let ord = get_param_ordinal(new, *pid_new);
            let pid_old = old.params[ord].id;
            NodePayload::GetParam(pid_old)
        }
        NodePayload::Tuple(children) => {
            let mapped: Vec<NodeRef> = children
                .iter()
                .map(|c| map_child(*c, patched, memo, next_id))
                .collect();
            NodePayload::Tuple(mapped)
        }
        NodePayload::Array(children) => {
            let mapped: Vec<NodeRef> = children
                .iter()
                .map(|c| map_child(*c, patched, memo, next_id))
                .collect();
            NodePayload::Array(mapped)
        }
        NodePayload::TupleIndex { tuple, index } => NodePayload::TupleIndex {
            tuple: map_child(*tuple, patched, memo, next_id),
            index: *index,
        },
        NodePayload::Binop(op, a, b) => NodePayload::Binop(
            *op,
            map_child(*a, patched, memo, next_id),
            map_child(*b, patched, memo, next_id),
        ),
        NodePayload::Unop(op, a) => NodePayload::Unop(*op, map_child(*a, patched, memo, next_id)),
        NodePayload::Literal(v) => NodePayload::Literal(v.clone()),
        NodePayload::SignExt { arg, new_bit_count } => NodePayload::SignExt {
            arg: map_child(*arg, patched, memo, next_id),
            new_bit_count: *new_bit_count,
        },
        NodePayload::ZeroExt { arg, new_bit_count } => NodePayload::ZeroExt {
            arg: map_child(*arg, patched, memo, next_id),
            new_bit_count: *new_bit_count,
        },
        NodePayload::ArrayUpdate {
            array,
            value,
            indices,
            assumed_in_bounds,
        } => {
            let mut mapped_indices: Vec<NodeRef> = Vec::with_capacity(indices.len());
            for idx in indices.iter() {
                mapped_indices.push(map_child(*idx, patched, memo, next_id));
            }
            NodePayload::ArrayUpdate {
                array: map_child(*array, patched, memo, next_id),
                value: map_child(*value, patched, memo, next_id),
                indices: mapped_indices,
                assumed_in_bounds: *assumed_in_bounds,
            }
        }
        NodePayload::ArrayIndex {
            array,
            indices,
            assumed_in_bounds,
        } => {
            let mut mapped_indices: Vec<NodeRef> = Vec::with_capacity(indices.len());
            for idx in indices.iter() {
                mapped_indices.push(map_child(*idx, patched, memo, next_id));
            }
            NodePayload::ArrayIndex {
                array: map_child(*array, patched, memo, next_id),
                indices: mapped_indices,
                assumed_in_bounds: *assumed_in_bounds,
            }
        }
        NodePayload::DynamicBitSlice { arg, start, width } => NodePayload::DynamicBitSlice {
            arg: map_child(*arg, patched, memo, next_id),
            start: map_child(*start, patched, memo, next_id),
            width: *width,
        },
        NodePayload::BitSlice { arg, start, width } => NodePayload::BitSlice {
            arg: map_child(*arg, patched, memo, next_id),
            start: *start,
            width: *width,
        },
        NodePayload::BitSliceUpdate {
            arg,
            start,
            update_value,
        } => NodePayload::BitSliceUpdate {
            arg: map_child(*arg, patched, memo, next_id),
            start: map_child(*start, patched, memo, next_id),
            update_value: map_child(*update_value, patched, memo, next_id),
        },
        NodePayload::Assert {
            token,
            activate,
            message,
            label,
        } => NodePayload::Assert {
            token: map_child(*token, patched, memo, next_id),
            activate: map_child(*activate, patched, memo, next_id),
            message: message.clone(),
            label: label.clone(),
        },
        NodePayload::Trace {
            token,
            activated,
            format,
            operands,
        } => {
            let mut mapped_ops: Vec<NodeRef> = Vec::with_capacity(operands.len());
            for o in operands.iter() {
                mapped_ops.push(map_child(*o, patched, memo, next_id));
            }
            NodePayload::Trace {
                token: map_child(*token, patched, memo, next_id),
                activated: map_child(*activated, patched, memo, next_id),
                format: format.clone(),
                operands: mapped_ops,
            }
        }
        NodePayload::AfterAll(children) => {
            let mapped: Vec<NodeRef> = children
                .iter()
                .map(|c| map_child(*c, patched, memo, next_id))
                .collect();
            NodePayload::AfterAll(mapped)
        }
        NodePayload::Nary(op, children) => {
            let mapped: Vec<NodeRef> = children
                .iter()
                .map(|c| map_child(*c, patched, memo, next_id))
                .collect();
            NodePayload::Nary(*op, mapped)
        }
        NodePayload::Invoke { to_apply, operands } => {
            let mapped: Vec<NodeRef> = operands
                .iter()
                .map(|c| map_child(*c, patched, memo, next_id))
                .collect();
            NodePayload::Invoke {
                to_apply: to_apply.clone(),
                operands: mapped,
            }
        }
        NodePayload::PrioritySel {
            selector,
            cases,
            default,
        } => {
            let mapped_selector = map_child(*selector, patched, memo, next_id);
            let mapped_cases: Vec<NodeRef> = cases
                .iter()
                .map(|c| map_child(*c, patched, memo, next_id))
                .collect();
            let mapped_default = default.map(|d| map_child(d, patched, memo, next_id));
            NodePayload::PrioritySel {
                selector: mapped_selector,
                cases: mapped_cases,
                default: mapped_default,
            }
        }
        NodePayload::OneHotSel { selector, cases } => {
            let mapped_selector = map_child(*selector, patched, memo, next_id);
            let mapped_cases: Vec<NodeRef> = cases
                .iter()
                .map(|c| map_child(*c, patched, memo, next_id))
                .collect();
            NodePayload::OneHotSel {
                selector: mapped_selector,
                cases: mapped_cases,
            }
        }
        NodePayload::OneHot { arg, lsb_prio } => NodePayload::OneHot {
            arg: map_child(*arg, patched, memo, next_id),
            lsb_prio: *lsb_prio,
        },
        NodePayload::Sel {
            selector,
            cases,
            default,
        } => {
            let mapped_selector = map_child(*selector, patched, memo, next_id);
            let mapped_cases: Vec<NodeRef> = cases
                .iter()
                .map(|c| map_child(*c, patched, memo, next_id))
                .collect();
            let mapped_default = default.map(|d| map_child(d, patched, memo, next_id));
            NodePayload::Sel {
                selector: mapped_selector,
                cases: mapped_cases,
                default: mapped_default,
            }
        }
        NodePayload::Cover { predicate, label } => NodePayload::Cover {
            predicate: map_child(*predicate, patched, memo, next_id),
            label: label.clone(),
        },
        NodePayload::Decode { arg, width } => NodePayload::Decode {
            arg: map_child(*arg, patched, memo, next_id),
            width: *width,
        },
        NodePayload::Encode { arg } => NodePayload::Encode {
            arg: map_child(*arg, patched, memo, next_id),
        },
        NodePayload::CountedFor {
            init,
            trip_count,
            stride,
            body,
            invariant_args,
        } => {
            let mapped_init = map_child(*init, patched, memo, next_id);
            let mapped_inv: Vec<NodeRef> = invariant_args
                .iter()
                .map(|c| map_child(*c, patched, memo, next_id))
                .collect();
            NodePayload::CountedFor {
                init: mapped_init,
                trip_count: *trip_count,
                stride: *stride,
                body: body.clone(),
                invariant_args: mapped_inv,
            }
        }
    };
    let new_text_id = *next_id;
    *next_id += 1;
    let new_node_cloned = crate::xls_ir::ir::Node {
        text_id: new_text_id,
        name: None,
        ty: new_node.ty.clone(),
        payload,
        pos: new_node.pos.clone(),
    };
    let new_index = patched.nodes.len();
    patched.nodes.push(new_node_cloned);
    let patched_nr = NodeRef { index: new_index };
    memo.insert(new_nr.index, patched_nr);
    patched_nr
}

fn rebuild_node_from_new_at_path(
    patched: &mut Fn,
    old: &Fn,
    new: &Fn,
    new_idx_to_old_idx: &std::collections::HashMap<usize, usize>,
    mut memo: &mut std::collections::HashMap<usize, NodeRef>,
    next_id: &mut usize,
    path: &[usize],
) {
    let old_root = patched.ret_node_ref.expect("patched must have ret node");
    let new_root = new.ret_node_ref.expect("new must have ret node");
    let patched_nr = node_ref_at_path(patched, old_root, path);
    let new_nr = node_ref_at_path(new, new_root, path);
    let new_node = new.get_node(new_nr);
    // Build payload by importing children.
    let map_child = |nr_new: NodeRef,
                     patched: &mut Fn,
                     memo: &mut std::collections::HashMap<usize, NodeRef>,
                     next_id: &mut usize| {
        import_subtree(patched, old, new, new_idx_to_old_idx, memo, next_id, nr_new)
    };
    let new_payload = match &new_node.payload {
        NodePayload::Nil => NodePayload::Nil,
        NodePayload::GetParam(pid_new) => {
            let ord = get_param_ordinal(new, *pid_new);
            let pid_old = old.params[ord].id;
            NodePayload::GetParam(pid_old)
        }
        NodePayload::Tuple(children)
        | NodePayload::Array(children)
        | NodePayload::AfterAll(children)
        | NodePayload::Nary(_, children) => {
            let mapped: Vec<NodeRef> = children
                .iter()
                .map(|c| map_child(*c, patched, &mut memo, next_id))
                .collect();
            match &new_node.payload {
                NodePayload::Tuple(_) => NodePayload::Tuple(mapped),
                NodePayload::Array(_) => NodePayload::Array(mapped),
                NodePayload::AfterAll(_) => NodePayload::AfterAll(mapped),
                NodePayload::Nary(op, _) => NodePayload::Nary(*op, mapped),
                _ => unreachable!(),
            }
        }
        NodePayload::TupleIndex { tuple, index } => NodePayload::TupleIndex {
            tuple: map_child(*tuple, patched, &mut memo, next_id),
            index: *index,
        },
        NodePayload::Binop(op, a, b) => NodePayload::Binop(
            *op,
            map_child(*a, patched, &mut memo, next_id),
            map_child(*b, patched, &mut memo, next_id),
        ),
        NodePayload::Unop(op, a) => {
            NodePayload::Unop(*op, map_child(*a, patched, &mut memo, next_id))
        }
        NodePayload::Literal(v) => NodePayload::Literal(v.clone()),
        NodePayload::SignExt { arg, new_bit_count } => NodePayload::SignExt {
            arg: map_child(*arg, patched, &mut memo, next_id),
            new_bit_count: *new_bit_count,
        },
        NodePayload::ZeroExt { arg, new_bit_count } => NodePayload::ZeroExt {
            arg: map_child(*arg, patched, &mut memo, next_id),
            new_bit_count: *new_bit_count,
        },
        NodePayload::ArrayUpdate {
            array,
            value,
            indices,
            assumed_in_bounds,
        } => {
            let mut mapped_indices: Vec<NodeRef> = Vec::with_capacity(indices.len());
            for idx in indices.iter() {
                mapped_indices.push(map_child(*idx, patched, &mut memo, next_id));
            }
            NodePayload::ArrayUpdate {
                array: map_child(*array, patched, &mut memo, next_id),
                value: map_child(*value, patched, &mut memo, next_id),
                indices: mapped_indices,
                assumed_in_bounds: *assumed_in_bounds,
            }
        }
        NodePayload::ArrayIndex {
            array,
            indices,
            assumed_in_bounds,
        } => {
            let mut mapped_indices: Vec<NodeRef> = Vec::with_capacity(indices.len());
            for idx in indices.iter() {
                mapped_indices.push(map_child(*idx, patched, &mut memo, next_id));
            }
            NodePayload::ArrayIndex {
                array: map_child(*array, patched, &mut memo, next_id),
                indices: mapped_indices,
                assumed_in_bounds: *assumed_in_bounds,
            }
        }
        NodePayload::DynamicBitSlice { arg, start, width } => NodePayload::DynamicBitSlice {
            arg: map_child(*arg, patched, &mut memo, next_id),
            start: map_child(*start, patched, &mut memo, next_id),
            width: *width,
        },
        NodePayload::BitSlice { arg, start, width } => NodePayload::BitSlice {
            arg: map_child(*arg, patched, &mut memo, next_id),
            start: *start,
            width: *width,
        },
        NodePayload::BitSliceUpdate {
            arg,
            start,
            update_value,
        } => NodePayload::BitSliceUpdate {
            arg: map_child(*arg, patched, &mut memo, next_id),
            start: map_child(*start, patched, &mut memo, next_id),
            update_value: map_child(*update_value, patched, &mut memo, next_id),
        },
        NodePayload::Assert {
            token,
            activate,
            message,
            label,
        } => NodePayload::Assert {
            token: map_child(*token, patched, &mut memo, next_id),
            activate: map_child(*activate, patched, &mut memo, next_id),
            message: message.clone(),
            label: label.clone(),
        },
        NodePayload::Trace {
            token,
            activated,
            format,
            operands,
        } => {
            let mut mapped_ops: Vec<NodeRef> = Vec::with_capacity(operands.len());
            for o in operands.iter() {
                mapped_ops.push(map_child(*o, patched, &mut memo, next_id));
            }
            NodePayload::Trace {
                token: map_child(*token, patched, &mut memo, next_id),
                activated: map_child(*activated, patched, &mut memo, next_id),
                format: format.clone(),
                operands: mapped_ops,
            }
        }
        NodePayload::Invoke { to_apply, operands } => {
            let mapped: Vec<NodeRef> = operands
                .iter()
                .map(|c| map_child(*c, patched, &mut memo, next_id))
                .collect();
            NodePayload::Invoke {
                to_apply: to_apply.clone(),
                operands: mapped,
            }
        }
        NodePayload::PrioritySel {
            selector,
            cases,
            default,
        } => {
            let mapped_selector = map_child(*selector, patched, &mut memo, next_id);
            let mapped_cases: Vec<NodeRef> = cases
                .iter()
                .map(|c| map_child(*c, patched, memo, next_id))
                .collect();
            let mapped_default = default.map(|d| map_child(d, patched, memo, next_id));
            NodePayload::PrioritySel {
                selector: mapped_selector,
                cases: mapped_cases,
                default: mapped_default,
            }
        }
        NodePayload::OneHotSel { selector, cases } => {
            let mapped_selector = map_child(*selector, patched, &mut memo, next_id);
            let mapped_cases: Vec<NodeRef> = cases
                .iter()
                .map(|c| map_child(*c, patched, memo, next_id))
                .collect();
            NodePayload::OneHotSel {
                selector: mapped_selector,
                cases: mapped_cases,
            }
        }
        NodePayload::OneHot { arg, lsb_prio } => NodePayload::OneHot {
            arg: map_child(*arg, patched, &mut memo, next_id),
            lsb_prio: *lsb_prio,
        },
        NodePayload::Sel {
            selector,
            cases,
            default,
        } => {
            let mapped_selector = map_child(*selector, patched, &mut memo, next_id);
            let mapped_cases: Vec<NodeRef> = cases
                .iter()
                .map(|c| map_child(*c, patched, &mut memo, next_id))
                .collect();
            let mapped_default = default.map(|d| map_child(d, patched, &mut memo, next_id));
            NodePayload::Sel {
                selector: mapped_selector,
                cases: mapped_cases,
                default: mapped_default,
            }
        }
        NodePayload::Cover { predicate, label } => NodePayload::Cover {
            predicate: map_child(*predicate, patched, &mut memo, next_id),
            label: label.clone(),
        },
        NodePayload::Decode { arg, width } => NodePayload::Decode {
            arg: map_child(*arg, patched, &mut memo, next_id),
            width: *width,
        },
        NodePayload::Encode { arg } => NodePayload::Encode {
            arg: map_child(*arg, patched, &mut memo, next_id),
        },
        NodePayload::CountedFor {
            init,
            trip_count,
            stride,
            body,
            invariant_args,
        } => {
            let mapped_init = map_child(*init, patched, &mut memo, next_id);
            let mapped_inv: Vec<NodeRef> = invariant_args
                .iter()
                .map(|c| map_child(*c, patched, &mut memo, next_id))
                .collect();
            NodePayload::CountedFor {
                init: mapped_init,
                trip_count: *trip_count,
                stride: *stride,
                body: body.clone(),
                invariant_args: mapped_inv,
            }
        }
    };
    // Mutate the existing node in place: preserve its text_id and name.
    let patched_node = patched.get_node_mut(patched_nr);
    patched_node.payload = new_payload;
    patched_node.ty = new_node.ty.clone();
    patched_node.pos = new_node.pos.clone();
}

/// Applies a LocalizedEcoDiff to `old`, using `new` as the source for any newly
/// inserted or substituted subgraphs. Preserves existing node IDs and only
/// allocates new IDs for imported nodes using the next available ordinal.
pub fn apply_localized_eco(old: &Fn, new: &Fn, diff: &LocalizedEcoDiff) -> Fn {
    let mut patched = old.clone();
    // Build mapping from new indices to old indices for equal subgraphs.
    let mut visited: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let mut equal_pairs: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    let old_hashes = collect_shape_hashes(old);
    let new_hashes = collect_shape_hashes(new);
    let old_local = collect_local_shape_hashes(old);
    let new_local = collect_local_shape_hashes(new);
    collect_equal_pairs(
        old,
        new,
        &old_hashes,
        &new_hashes,
        &old_local,
        &new_local,
        old.ret_node_ref.expect("old must have ret"),
        new.ret_node_ref.expect("new must have ret"),
        &mut visited,
        &mut equal_pairs,
    );
    let mut new_idx_to_old_idx: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for (old_idx, new_idx) in equal_pairs.into_iter() {
        new_idx_to_old_idx.insert(new_idx, old_idx);
    }
    // Compute next available id.
    let mut max_old_id = 0usize;
    for n in old.nodes.iter() {
        if n.text_id > max_old_id {
            max_old_id = n.text_id;
        }
    }
    let mut next_id = max_old_id + 1;

    // Collect unique paths to modify from diff.
    let mut paths: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();
    for e in diff.edits.iter() {
        match e {
            Edit::OperatorChanged { path, .. }
            | Edit::TypeChanged { path, .. }
            | Edit::ArityChanged { path, .. }
            | Edit::OperandInserted { path, .. }
            | Edit::OperandDeleted { path, .. }
            | Edit::OperandSubstituted { path, .. } => {
                paths.insert(path.0.clone());
            }
        }
    }
    // Apply modifications: rebuild nodes at these paths from `new`.
    let mut path_list: Vec<Vec<usize>> = paths.into_iter().collect();
    // Rebuilding parents first is sufficient since children are imported via
    // import_subtree. Order by path length ascending ensures parent nodes are
    // visited before deeper paths.
    path_list.sort_by_key(|p| p.len());
    let mut memo: std::collections::HashMap<usize, NodeRef> = std::collections::HashMap::new();
    for p in path_list.iter() {
        rebuild_node_from_new_at_path(
            &mut patched,
            old,
            new,
            &new_idx_to_old_idx,
            &mut memo,
            &mut next_id,
            p,
        );
    }

    // Re-topologize nodes so dependencies appear before uses.
    retopologize_in_place(&mut patched);

    patched
}

fn retopologize_in_place(f: &mut Fn) {
    let order = get_topological(f);
    let mut old_to_new: Vec<usize> = vec![0; f.nodes.len()];
    for (new_idx, nr) in order.iter().enumerate() {
        old_to_new[nr.index] = new_idx;
    }
    let mut remap_ref = |nr: NodeRef| NodeRef {
        index: old_to_new[nr.index],
    };
    fn remap_payload(
        payload: &NodePayload,
        remap: &mut dyn FnMut(NodeRef) -> NodeRef,
    ) -> NodePayload {
        match payload {
            NodePayload::Nil => NodePayload::Nil,
            NodePayload::GetParam(pid) => NodePayload::GetParam(*pid),
            NodePayload::Tuple(children) => {
                NodePayload::Tuple(children.iter().map(|c| remap(*c)).collect())
            }
            NodePayload::Array(children) => {
                NodePayload::Array(children.iter().map(|c| remap(*c)).collect())
            }
            NodePayload::TupleIndex { tuple, index } => NodePayload::TupleIndex {
                tuple: remap(*tuple),
                index: *index,
            },
            NodePayload::Binop(op, a, b) => NodePayload::Binop(*op, remap(*a), remap(*b)),
            NodePayload::Unop(op, a) => NodePayload::Unop(*op, remap(*a)),
            NodePayload::Literal(v) => NodePayload::Literal(v.clone()),
            NodePayload::SignExt { arg, new_bit_count } => NodePayload::SignExt {
                arg: remap(*arg),
                new_bit_count: *new_bit_count,
            },
            NodePayload::ZeroExt { arg, new_bit_count } => NodePayload::ZeroExt {
                arg: remap(*arg),
                new_bit_count: *new_bit_count,
            },
            NodePayload::ArrayUpdate {
                array,
                value,
                indices,
                assumed_in_bounds,
            } => NodePayload::ArrayUpdate {
                array: remap(*array),
                value: remap(*value),
                indices: indices.iter().map(|c| remap(*c)).collect(),
                assumed_in_bounds: *assumed_in_bounds,
            },
            NodePayload::ArrayIndex {
                array,
                indices,
                assumed_in_bounds,
            } => NodePayload::ArrayIndex {
                array: remap(*array),
                indices: indices.iter().map(|c| remap(*c)).collect(),
                assumed_in_bounds: *assumed_in_bounds,
            },
            NodePayload::DynamicBitSlice { arg, start, width } => NodePayload::DynamicBitSlice {
                arg: remap(*arg),
                start: remap(*start),
                width: *width,
            },
            NodePayload::BitSlice { arg, start, width } => NodePayload::BitSlice {
                arg: remap(*arg),
                start: *start,
                width: *width,
            },
            NodePayload::BitSliceUpdate {
                arg,
                start,
                update_value,
            } => NodePayload::BitSliceUpdate {
                arg: remap(*arg),
                start: remap(*start),
                update_value: remap(*update_value),
            },
            NodePayload::Assert {
                token,
                activate,
                message,
                label,
            } => NodePayload::Assert {
                token: remap(*token),
                activate: remap(*activate),
                message: message.clone(),
                label: label.clone(),
            },
            NodePayload::Trace {
                token,
                activated,
                format,
                operands,
            } => NodePayload::Trace {
                token: remap(*token),
                activated: remap(*activated),
                format: format.clone(),
                operands: operands.iter().map(|c| remap(*c)).collect(),
            },
            NodePayload::AfterAll(children) => {
                NodePayload::AfterAll(children.iter().map(|c| remap(*c)).collect())
            }
            NodePayload::Nary(op, children) => {
                NodePayload::Nary(*op, children.iter().map(|c| remap(*c)).collect())
            }
            NodePayload::Invoke { to_apply, operands } => NodePayload::Invoke {
                to_apply: to_apply.clone(),
                operands: operands.iter().map(|c| remap(*c)).collect(),
            },
            NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } => NodePayload::PrioritySel {
                selector: remap(*selector),
                cases: cases.iter().map(|c| remap(*c)).collect(),
                default: default.map(|d| remap(d)),
            },
            NodePayload::OneHotSel { selector, cases } => NodePayload::OneHotSel {
                selector: remap(*selector),
                cases: cases.iter().map(|c| remap(*c)).collect(),
            },
            NodePayload::OneHot { arg, lsb_prio } => NodePayload::OneHot {
                arg: remap(*arg),
                lsb_prio: *lsb_prio,
            },
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => NodePayload::Sel {
                selector: remap(*selector),
                cases: cases.iter().map(|c| remap(*c)).collect(),
                default: default.map(|d| remap(d)),
            },
            NodePayload::Cover { predicate, label } => NodePayload::Cover {
                predicate: remap(*predicate),
                label: label.clone(),
            },
            NodePayload::Decode { arg, width } => NodePayload::Decode {
                arg: remap(*arg),
                width: *width,
            },
            NodePayload::Encode { arg } => NodePayload::Encode { arg: remap(*arg) },
            NodePayload::CountedFor {
                init,
                trip_count,
                stride,
                body,
                invariant_args,
            } => NodePayload::CountedFor {
                init: remap(*init),
                trip_count: *trip_count,
                stride: *stride,
                body: body.clone(),
                invariant_args: invariant_args.iter().map(|c| remap(*c)).collect(),
            },
        }
    }
    let mut new_nodes: Vec<crate::xls_ir::ir::Node> = Vec::with_capacity(f.nodes.len());
    for nr in order.iter() {
        let old_node = &f.nodes[nr.index];
        let new_payload = remap_payload(&old_node.payload, &mut remap_ref);
        new_nodes.push(crate::xls_ir::ir::Node {
            text_id: old_node.text_id,
            name: old_node.name.clone(),
            ty: old_node.ty.clone(),
            payload: new_payload,
            pos: old_node.pos.clone(),
        });
    }
    f.nodes = new_nodes;
    if let Some(ret) = f.ret_node_ref {
        f.ret_node_ref = Some(NodeRef {
            index: old_to_new[ret.index],
        });
    }
}

fn node_operator_string(f: &Fn, nr: NodeRef) -> String {
    match &f.get_node(nr).payload {
        NodePayload::Binop(op, _, _) => binop_to_operator(*op).to_string(),
        NodePayload::Unop(op, _) => unop_to_operator(*op).to_string(),
        NodePayload::Nary(op, _) => nary_op_to_operator(*op).to_string(),
        p => p.get_operator().to_string(),
    }
}

fn node_type_string(f: &Fn, nr: NodeRef) -> String {
    format!("{}", f.get_node(nr).ty)
}

fn align_nodes(
    old: &Fn,
    new: &Fn,
    old_hashes: &[blake3::Hash],
    new_hashes: &[blake3::Hash],
    old_local: &[blake3::Hash],
    new_local: &[blake3::Hash],
    old_nr: NodeRef,
    new_nr: NodeRef,
    path: &mut Vec<usize>,
    edits: &mut Vec<Edit>,
    stats: &mut DiffStats,
    visited_pairs: &mut HashSet<(usize, usize)>,
) {
    let pair_key = (old_nr.index, new_nr.index);
    if visited_pairs.contains(&pair_key) {
        return;
    }
    visited_pairs.insert(pair_key);

    if old_hashes[old_nr.index] == new_hashes[new_nr.index] {
        stats.matched_nodes += 1;
        return;
    }

    // Record operator/type/arity changes at this node if any.
    let old_op = node_operator_string(old, old_nr);
    let new_op = node_operator_string(new, new_nr);
    if old_op != new_op {
        edits.push(Edit::OperatorChanged {
            path: Path(path.clone()),
            old_op,
            new_op,
        });
    }
    let old_ty = node_type_string(old, old_nr);
    let new_ty = node_type_string(new, new_nr);
    if old_ty != new_ty {
        edits.push(Edit::TypeChanged {
            path: Path(path.clone()),
            old_ty,
            new_ty,
        });
    }

    let old_children = children_in_order(old, old_nr);
    let new_children = children_in_order(new, new_nr);
    if old_children.len() != new_children.len() {
        edits.push(Edit::ArityChanged {
            path: Path(path.clone()),
            old_arity: old_children.len(),
            new_arity: new_children.len(),
        });
    }

    // Align ordered operands via Levenshtein DP on shape equality.
    let n = old_children.len();
    let m = new_children.len();
    let mut dp: Vec<Vec<usize>> = vec![vec![0; m + 1]; n + 1];
    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let equal =
                old_hashes[old_children[i - 1].index] == new_hashes[new_children[j - 1].index];
            let cost_sub = if equal { 0 } else { 1 };
            let del = dp[i - 1][j] + 1;
            let ins = dp[i][j - 1] + 1;
            let sub = dp[i - 1][j - 1] + cost_sub;
            dp[i][j] = del.min(ins).min(sub);
        }
    }

    // Backtrace to emit edits and collect matched/substituted pairs to recurse
    // into.
    let mut i = n;
    let mut j = m;
    let mut recurse_pairs: Vec<(usize, usize, usize)> = Vec::new();
    while i > 0 || j > 0 {
        if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            let idx = i - 1;
            let sig = old.get_node(old_children[idx]).to_signature_string(old);
            edits.push(Edit::OperandDeleted {
                path: Path(path.clone()),
                index: idx,
                old_signature: sig,
            });
            i -= 1;
        } else if j > 0 && dp[i][j] == dp[i][j - 1] + 1 {
            let idx = j - 1;
            let sig = new.get_node(new_children[idx]).to_signature_string(new);
            edits.push(Edit::OperandInserted {
                path: Path(path.clone()),
                index: idx,
                new_signature: sig,
            });
            j -= 1;
        } else {
            // substitution or match
            let idx_old = i - 1;
            let idx_new = j - 1;
            let equal =
                old_hashes[old_children[idx_old].index] == new_hashes[new_children[idx_new].index];
            if equal {
                // No edit; no need to recurse further.
            } else {
                // If the child's local shape (op/type/attrs) matches, do not emit a
                // substitution at this parent. Recurse into the child to localize the diff.
                let old_child_idx = old_children[idx_old].index;
                let new_child_idx = new_children[idx_new].index;
                let local_equal = old_local[old_child_idx] == new_local[new_child_idx];
                let single_input_parent = n == 1 && m == 1;
                if !local_equal && !single_input_parent {
                    let old_sig = old.get_node(old_children[idx_old]).to_signature_string(old);
                    let new_sig = new.get_node(new_children[idx_new]).to_signature_string(new);
                    edits.push(Edit::OperandSubstituted {
                        path: Path(path.clone()),
                        index: idx_new,
                        old_signature: old_sig,
                        new_signature: new_sig,
                    });
                }
                recurse_pairs.push((old_child_idx, new_child_idx, idx_new));
            }
            i -= 1;
            j -= 1;
        }
    }

    // Recurse into substituted pairs in forward index order to keep stable paths.
    recurse_pairs.reverse();
    for (old_child_idx, new_child_idx, at_index) in recurse_pairs.into_iter() {
        path.push(at_index);
        let old_child = NodeRef {
            index: old_child_idx,
        };
        let new_child = NodeRef {
            index: new_child_idx,
        };
        align_nodes(
            old,
            new,
            old_hashes,
            new_hashes,
            old_local,
            new_local,
            old_child,
            new_child,
            path,
            edits,
            stats,
            visited_pairs,
        );
        path.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xls_ir::ir_parser::Parser;

    fn parse_fn(ir: &str) -> Fn {
        let mut p = Parser::new(ir);
        p.parse_fn().unwrap()
    }

    // -- Tests

    #[test]
    fn identical_functions_produce_no_edits() {
        let f = parse_fn(
            "fn id(a: bits[1] id=1) -> bits[1] {\n  ret identity.2: bits[1] = identity(a, id=2)\n}",
        );
        let g = parse_fn(
            "fn id(a: bits[1] id=1) -> bits[1] {\n  ret identity.2: bits[1] = identity(a, id=2)\n}",
        );
        let diff = compute_localized_eco(&f, &g);
        assert!(diff.edits.is_empty());
        assert!(diff.stats.matched_nodes > 0);
    }

    #[test]
    fn tuple_extra_operand_localized_insert() {
        let old = parse_fn(
            "fn t(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {\n\
  ret or.3: bits[1] = or(a, b, id=3)\n\
}",
        );
        let new = parse_fn(
            "fn t(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {\n\
  ret or.3: bits[1] = or(a, b, b, id=3)\n\
}",
        );
        let diff = compute_localized_eco(&old, &new);
        // Expect at least one insertion at the return node path [].
        assert!(
            diff.edits
                .iter()
                .any(|e| matches!(e, Edit::OperandInserted { path, .. } if path.0.is_empty()))
        );
    }

    #[test]
    fn operator_change_localized() {
        let old =
            parse_fn("fn f(a: bits[1] id=1) -> bits[1] {\n  ret not.2: bits[1] = not(a, id=2)\n}");
        let new = parse_fn(
            "fn f(a: bits[1] id=1) -> bits[1] {\n  ret identity.2: bits[1] = identity(a, id=2)\n}",
        );
        let diff = compute_localized_eco(&old, &new);
        assert!(diff
            .edits
            .iter()
            .any(|e| matches!(e, Edit::OperatorChanged { path, old_op, new_op } if path.0.is_empty() && old_op == "not" && new_op == "identity")));
    }

    #[test]
    fn deep_chain_single_interior_change_backwards_trim() {
        // old: not(not(not(a)))
        // new: not(identity(not(a))) -- change at the middle node
        let old = parse_fn(
            "fn f(a: bits[1] id=1) -> bits[1] {\n  not.2: bits[1] = not(a, id=2)\n  not.3: bits[1] = not(not.2, id=3)\n  ret not.4: bits[1] = not(not.3, id=4)\n}",
        );
        let new = parse_fn(
            "fn f(a: bits[1] id=1) -> bits[1] {\n  not.2: bits[1] = not(a, id=2)\n  identity.3: bits[1] = identity(not.2, id=3)\n  ret not.4: bits[1] = not(identity.3, id=4)\n}",
        );
        let diff = compute_localized_eco(&old, &new);
        // Expect an operator change at the child (path [0]) and no substitution at the
        // root.
        assert!(diff.edits.iter().any(|e| matches!(e,
            Edit::OperatorChanged { path, old_op, new_op } if path.0 == vec![0] && old_op == "not" && new_op == "identity")));
        assert!(!diff.edits.iter().any(|e| matches!(e,
            Edit::OperandSubstituted { path, .. } if path.0.is_empty())));
        // Should not produce edits beyond that (e.g. no edits at the leaf 'not(a)')
        assert!(!diff.edits.iter().any(|e| matches!(e,
            Edit::OperatorChanged { path, .. } if path.0 == vec![0, 0])));
    }
}
