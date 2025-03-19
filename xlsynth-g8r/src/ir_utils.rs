// SPDX-License-Identifier: Apache-2.0

use crate::ir::{Fn, NodePayload, NodeRef};

/// Returns the list of operands for the provided node.
pub fn operands(payload: &NodePayload) -> Vec<NodeRef> {
    use NodePayload::*;
    match payload {
        Nil => vec![],
        GetParam(_) => vec![],
        Tuple(elems) => elems.clone(),
        Array(elems) => elems.clone(),
        TupleIndex { tuple, index: _ } => vec![*tuple],
        Binop(_, a, b) => vec![*a, *b],
        Unop(_, a) => vec![*a],
        Literal(_) => vec![],
        SignExt {
            arg,
            new_bit_count: _,
        } => vec![*arg],
        ZeroExt {
            arg,
            new_bit_count: _,
        } => vec![*arg],
        ArrayUpdate {
            array,
            value,
            indices,
        } => {
            let mut deps = vec![*array, *value];
            deps.extend(indices.iter().cloned());
            deps
        }
        ArrayIndex { array, indices } => {
            let mut deps = vec![*array];
            deps.extend(indices.iter().cloned());
            deps
        }
        DynamicBitSlice {
            arg,
            start,
            width: _,
        } => vec![*arg, *start],
        BitSlice {
            arg,
            start: _,
            width: _,
        } => vec![*arg],
        BitSliceUpdate {
            arg,
            start,
            update_value,
        } => vec![*arg, *start, *update_value],
        Assert {
            token,
            activate,
            message: _,
            label: _,
        } => vec![*token, *activate],
        Trace {
            token,
            activated,
            format: _,
            operands,
        } => {
            let mut deps = vec![*token, *activated];
            deps.extend(operands.iter().cloned());
            deps
        }
        AfterAll(elems) => elems.clone(),
        Nary(_, elems) => elems.clone(),
        Invoke {
            to_apply: _,
            operands,
        } => operands.clone(),
        PrioritySel {
            selector,
            cases,
            default,
        } => {
            let mut deps = vec![*selector];
            deps.extend(cases.iter().cloned());
            if let Some(d) = default {
                deps.push(*d);
            }
            deps
        }
        OneHotSel { selector, cases } => {
            let mut deps = vec![*selector];
            deps.extend(cases.iter().cloned());
            deps
        }
        OneHot { arg, lsb_prio: _ } => vec![*arg],
        Sel {
            selector,
            cases,
            default,
        } => {
            let mut deps = vec![*selector];
            deps.extend(cases.iter().cloned());
            if let Some(d) = default {
                deps.push(*d);
            }
            deps
        }
        Cover {
            predicate,
            label: _,
        } => vec![*predicate],
        Decode { arg, .. } => vec![*arg],
    }
}

/// Returns a topologically sorted list of node references for the given IR
/// function.
///
/// The ordering guarantees that for any node, all its dependency nodes will
/// appear before it in the returned vector.
pub fn get_topological(f: &Fn) -> Vec<NodeRef> {
    let n = f.nodes.len();
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);

    // Recursive DFS visit for each node.
    fn visit(node_ref: NodeRef, f: &Fn, visited: &mut [bool], order: &mut Vec<NodeRef>) {
        if visited[node_ref.index] {
            return;
        }
        visited[node_ref.index] = true;

        let node = &f.nodes[node_ref.index];
        for dep in operands(&node.payload) {
            visit(dep, f, visited, order);
        }
        order.push(node_ref);
    }

    // Visit all nodes in the function so that even non-reachable nodes are touched.
    for i in 0..n {
        let node_ref = NodeRef { index: i };
        if !visited[i] {
            visit(node_ref, f, &mut visited, &mut order);
        }
    }

    order
}
