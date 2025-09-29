// SPDX-License-Identifier: Apache-2.0

//! Utility functions for working with / on XLS IR.

use crate::ir::{Fn, Node, NodePayload, NodeRef};
use std::collections::{HashMap, HashSet};

/// Returns the list of operands for the provided node.
pub fn operands(payload: &NodePayload) -> Vec<NodeRef> {
    use NodePayload::*;

    match payload {
        Nil => vec![],
        GetParam(_) => vec![],
        Tuple(elems) => elems.clone(),
        Array(elems) => elems.clone(),
        ArraySlice {
            array,
            start,
            width: _,
        } => vec![*array, *start],
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
            assumed_in_bounds: _,
        } => {
            let mut deps = vec![*array, *value];
            deps.extend(indices.iter().cloned());
            deps
        }
        ArrayIndex {
            array,
            indices,
            assumed_in_bounds: _,
        } => {
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
        Decode { arg, .. } | Encode { arg, .. } => vec![*arg],
        CountedFor {
            init,
            invariant_args,
            ..
        } => {
            let mut deps = vec![*init];
            deps.extend(invariant_args.iter().cloned());
            deps
        }
    }
}

/// Returns a topologically sorted list of node references for the given IR
/// function.
///
/// The ordering guarantees that for any node, all its dependency nodes will
/// appear before it in the returned vector.
fn topo_from_nodes(nodes: &[Node]) -> Vec<NodeRef> {
    // Non-recursive DFS that preserves prior postorder semantics.
    let n = nodes.len();
    let mut visited: Vec<bool> = vec![false; n];
    let mut in_stack: Vec<bool> = vec![false; n];
    let mut order: Vec<NodeRef> = Vec::with_capacity(n);

    // Precompute dependency indices per node to avoid repeated operand walks
    let mut deps: Vec<Vec<usize>> = Vec::with_capacity(n);
    for node in nodes.iter() {
        deps.push(
            operands(&node.payload)
                .into_iter()
                .map(|r| r.index)
                .collect(),
        );
    }

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut stack: Vec<(usize, usize)> = Vec::new(); // (node_idx, next_child_pos)
        stack.push((start, 0));
        in_stack[start] = true;

        while let Some((node_idx, child_pos)) = stack.pop() {
            if visited[node_idx] {
                in_stack[node_idx] = false;
                continue;
            }
            if child_pos < deps[node_idx].len() {
                let next_child = deps[node_idx][child_pos];
                stack.push((node_idx, child_pos + 1));
                if !visited[next_child] {
                    assert!(
                        !in_stack[next_child],
                        "Cycle detected in IR graph; topological order impossible"
                    );
                    stack.push((next_child, 0));
                    in_stack[next_child] = true;
                }
                continue;
            }
            visited[node_idx] = true;
            in_stack[node_idx] = false;
            order.push(NodeRef { index: node_idx });
        }
    }
    assert!(
        order.len() == n,
        "Topological sort did not include all nodes"
    );
    order
}

pub fn get_topological(f: &Fn) -> Vec<NodeRef> {
    topo_from_nodes(&f.nodes)
}

/// Returns a topologically sorted list of node references for a standalone node
/// list. Useful when nodes are not yet wrapped in an `Fn`.
pub fn get_topological_nodes(nodes: &[Node]) -> Vec<NodeRef> {
    topo_from_nodes(nodes)
}

/// Returns a list of nodes that are dead (unreachable from the function's
/// return value by following operand edges).
///
/// The returned vector is sorted by node index ascending to ensure
/// deterministic ordering.
pub fn get_dead_nodes(f: &Fn) -> Vec<NodeRef> {
    let n = f.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    // Mark nodes reachable from the return node via operands.
    let mut live: Vec<bool> = vec![false; n];
    assert!(
        f.ret_node_ref.is_some(),
        "get_dead_nodes: function has no return node"
    );
    let ret = f.ret_node_ref.unwrap();
    let mut stack: Vec<NodeRef> = vec![ret];
    while let Some(nr) = stack.pop() {
        if live[nr.index] {
            continue;
        }
        live[nr.index] = true;
        let node = f.get_node(nr);
        for dep in operands(&node.payload) {
            if !live[dep.index] {
                stack.push(dep);
            }
        }
    }

    // Dead nodes are those never marked live.
    let mut dead: Vec<NodeRef> = Vec::new();
    for i in 0..n {
        if !live[i] {
            dead.push(NodeRef { index: i });
        }
    }
    dead
}

/// Returns a new function with dead nodes (unreachable from the return value)
/// removed and all remaining node indices compacted. Operand references are
/// remapped to the new indices. GetParam nodes are preserved even if they would
/// otherwise be considered dead, to satisfy validation rules requiring a
/// GetParam for each declared parameter.
pub fn remove_dead_nodes(f: &Fn) -> Fn {
    let n = f.nodes.len();
    assert!(n > 0, "remove_dead_nodes: function has no nodes");
    assert!(
        f.ret_node_ref.is_some(),
        "remove_dead_nodes: function has no return node"
    );

    // Compute liveness from the return.
    let mut live: Vec<bool> = vec![false; n];
    let mut stack: Vec<NodeRef> = vec![f.ret_node_ref.unwrap()];
    while let Some(nr) = stack.pop() {
        if live[nr.index] {
            continue;
        }
        live[nr.index] = true;
        let node = f.get_node(nr);
        for dep in operands(&node.payload) {
            if !live[dep.index] {
                stack.push(dep);
            }
        }
    }
    // Always keep GetParam nodes to satisfy validation rules.
    for (i, node) in f.nodes.iter().enumerate() {
        if matches!(node.payload, NodePayload::GetParam(_)) {
            live[i] = true;
        }
    }

    // Build mapping old index -> new index for live nodes.
    let mut mapping: Vec<Option<usize>> = vec![None; n];
    let mut next: usize = 0;
    for i in 0..n {
        if live[i] {
            mapping[i] = Some(next);
            next += 1;
        }
    }

    // Remap payloads using the mapping. Only live nodes are copied.
    let mut new_nodes: Vec<Node> = Vec::with_capacity(next);
    for (i, node) in f.nodes.iter().enumerate() {
        if !live[i] {
            continue;
        }
        let remapped_payload = remap_payload_with(&node.payload, |(_, nr): (usize, NodeRef)| {
            let ni = mapping[nr.index].expect("live node must not reference a dead operand");
            NodeRef { index: ni }
        });
        new_nodes.push(Node {
            text_id: node.text_id,
            name: node.name.clone(),
            ty: node.ty.clone(),
            payload: remapped_payload,
            pos: node.pos.clone(),
        });
    }

    // Remap return node.
    let ret_old = f.ret_node_ref.unwrap().index;
    let ret_new = mapping[ret_old].expect("return node must be live");

    Fn {
        name: f.name.clone(),
        params: f.params.clone(),
        ret_ty: f.ret_ty.clone(),
        nodes: new_nodes,
        ret_node_ref: Some(NodeRef { index: ret_new }),
        outer_attrs: f.outer_attrs.clone(),
        inner_attrs: f.inner_attrs.clone(),
    }
}

/// Computes the immediate users of each node in the function.
///
/// Returns a mapping from each `NodeRef` to the set of `NodeRef`s that
/// directly use it as an operand. Nodes with no users will map to an empty set.
pub fn compute_users(f: &Fn) -> HashMap<NodeRef, HashSet<NodeRef>> {
    let n = f.nodes.len();
    let mut users: HashMap<NodeRef, HashSet<NodeRef>> = HashMap::with_capacity(n);

    // Initialize all keys so even unreachable / sink nodes appear with empty sets.
    for i in 0..n {
        users.insert(NodeRef { index: i }, HashSet::new());
    }

    // For each node, add it as a user of each of its operands.
    for (i, node) in f.nodes.iter().enumerate() {
        let this_ref = NodeRef { index: i };
        for dep in operands(&node.payload) {
            users
                .get_mut(&dep)
                .expect("operand NodeRef must exist in users map")
                .insert(this_ref);
        }
    }

    users
}

pub fn remap_payload_with<FMap>(payload: &NodePayload, mut map: FMap) -> NodePayload
where
    // Map function takes the operand slot and the existing operand and returns the new operand.
    FMap: FnMut((usize, NodeRef)) -> NodeRef,
{
    match payload {
        NodePayload::Nil => NodePayload::Nil,
        NodePayload::GetParam(p) => NodePayload::GetParam(*p),
        NodePayload::Tuple(elems) => NodePayload::Tuple(
            elems
                .iter()
                .enumerate()
                .map(|(i, r)| map((i, *r)))
                .collect(),
        ),
        NodePayload::Array(elems) => NodePayload::Array(
            elems
                .iter()
                .enumerate()
                .map(|(i, r)| map((i, *r)))
                .collect(),
        ),
        NodePayload::TupleIndex { tuple, index } => NodePayload::TupleIndex {
            tuple: map((0, *tuple)),
            index: *index,
        },
        NodePayload::Binop(op, a, b) => NodePayload::Binop(*op, map((0, *a)), map((1, *b))),
        NodePayload::Unop(op, a) => NodePayload::Unop(*op, map((0, *a))),
        NodePayload::Literal(v) => NodePayload::Literal(v.clone()),
        NodePayload::SignExt { arg, new_bit_count } => NodePayload::SignExt {
            arg: map((0, *arg)),
            new_bit_count: *new_bit_count,
        },
        NodePayload::ZeroExt { arg, new_bit_count } => NodePayload::ZeroExt {
            arg: map((0, *arg)),
            new_bit_count: *new_bit_count,
        },
        NodePayload::ArrayUpdate {
            array,
            value,
            indices,
            assumed_in_bounds,
        } => NodePayload::ArrayUpdate {
            array: map((0, *array)),
            value: map((1, *value)),
            indices: indices
                .iter()
                .enumerate()
                .map(|(i, r)| map((i + 2, *r)))
                .collect(),
            assumed_in_bounds: *assumed_in_bounds,
        },
        NodePayload::ArrayIndex {
            array,
            indices,
            assumed_in_bounds,
        } => NodePayload::ArrayIndex {
            array: map((0, *array)),
            indices: indices
                .iter()
                .enumerate()
                .map(|(i, r)| map((i + 1, *r)))
                .collect(),
            assumed_in_bounds: *assumed_in_bounds,
        },
        NodePayload::ArraySlice {
            array,
            start,
            width,
        } => NodePayload::ArraySlice {
            array: map((0, *array)),
            start: map((1, *start)),
            width: *width,
        },
        NodePayload::DynamicBitSlice { arg, start, width } => NodePayload::DynamicBitSlice {
            arg: map((0, *arg)),
            start: map((1, *start)),
            width: *width,
        },
        NodePayload::BitSlice { arg, start, width } => NodePayload::BitSlice {
            arg: map((0, *arg)),
            start: *start,
            width: *width,
        },
        NodePayload::BitSliceUpdate {
            arg,
            start,
            update_value,
        } => NodePayload::BitSliceUpdate {
            arg: map((0, *arg)),
            start: map((1, *start)),
            update_value: map((2, *update_value)),
        },
        NodePayload::Assert {
            token,
            activate,
            message,
            label,
        } => NodePayload::Assert {
            token: map((0, *token)),
            activate: map((1, *activate)),
            message: message.clone(),
            label: label.clone(),
        },
        NodePayload::Trace {
            token,
            activated,
            format,
            operands,
        } => NodePayload::Trace {
            token: map((0, *token)),
            activated: map((1, *activated)),
            format: format.clone(),
            operands: operands
                .iter()
                .enumerate()
                .map(|(i, r)| map((i + 2, *r)))
                .collect(),
        },
        NodePayload::AfterAll(elems) => NodePayload::AfterAll(
            elems
                .iter()
                .enumerate()
                .map(|(i, r)| map((i, *r)))
                .collect(),
        ),
        NodePayload::Nary(op, elems) => NodePayload::Nary(
            *op,
            elems
                .iter()
                .enumerate()
                .map(|(i, r)| map((i, *r)))
                .collect(),
        ),
        NodePayload::Invoke { to_apply, operands } => NodePayload::Invoke {
            to_apply: to_apply.clone(),
            operands: operands
                .iter()
                .enumerate()
                .map(|(i, r)| map((i, *r)))
                .collect(),
        },
        NodePayload::PrioritySel {
            selector,
            cases,
            default,
        } => NodePayload::PrioritySel {
            selector: map((0, *selector)),
            cases: cases
                .iter()
                .enumerate()
                .map(|(i, r)| map((i + 1, *r)))
                .collect(),
            default: default.map(|d| map((cases.len() + 1, d))),
        },
        NodePayload::OneHotSel { selector, cases } => NodePayload::OneHotSel {
            selector: map((0, *selector)),
            cases: cases
                .iter()
                .enumerate()
                .map(|(i, r)| map((i + 1, *r)))
                .collect(),
        },
        NodePayload::OneHot { arg, lsb_prio } => NodePayload::OneHot {
            arg: map((0, *arg)),
            lsb_prio: *lsb_prio,
        },
        NodePayload::Sel {
            selector,
            cases,
            default,
        } => NodePayload::Sel {
            selector: map((0, *selector)),
            cases: cases
                .iter()
                .enumerate()
                .map(|(i, r)| map((i + 1, *r)))
                .collect(),
            default: default.map(|d| map((cases.len() + 1, d))),
        },
        NodePayload::Cover { predicate, label } => NodePayload::Cover {
            predicate: map((0, *predicate)),
            label: label.clone(),
        },
        NodePayload::Decode { arg, width } => NodePayload::Decode {
            arg: map((0, *arg)),
            width: *width,
        },
        NodePayload::Encode { arg } => NodePayload::Encode {
            arg: map((0, *arg)),
        },
        NodePayload::CountedFor {
            init,
            trip_count,
            stride,
            body,
            invariant_args,
        } => NodePayload::CountedFor {
            init: map((0, *init)),
            trip_count: *trip_count,
            stride: *stride,
            body: body.clone(),
            invariant_args: invariant_args
                .iter()
                .enumerate()
                .map(|(i, r)| map((i + 1, *r)))
                .collect(),
        },
    }
}

/// Returns true if `s` is a valid IR identifier `([_A-Za-z][_A-Za-z0-9]*)`;
/// i.e. can be used as a node name or parameter name.
pub fn is_valid_identifier_name(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c == '_' || c.is_ascii_alphabetic() => {}
        _ => return false,
    };
    for c in chars {
        if !(c == '_' || c.is_ascii_alphanumeric()) {
            return false;
        }
    }
    true
}

/// Sanitizes arbitrary text to a valid identifier name deterministically.
pub fn sanitize_text_id_to_identifier_name(s: &str) -> String {
    assert!(
        s.chars()
            .all(|c| c == '_' || c == '.' || c.is_ascii_alphanumeric())
    );
    // Replace dots with underscores.
    s.replace('.', "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::PackageMember;
    use crate::ir_parser::Parser;

    fn parse_fn(ir: &str) -> Fn {
        let pkg_text = format!("package test\n\n{}\n", ir);
        let mut p = Parser::new(&pkg_text);
        let pkg = p.parse_and_validate_package().unwrap();
        pkg.members
            .iter()
            .filter_map(|m| match m {
                PackageMember::Function(f) => Some(f.clone()),
                _ => None,
            })
            .next()
            .unwrap()
    }

    fn verify_topo_property(f: &Fn, order: &[NodeRef]) {
        // Build position map
        let mut pos: Vec<usize> = vec![0; f.nodes.len()];
        for (i, nr) in order.iter().enumerate() {
            pos[nr.index] = i;
        }
        for nr in order {
            let node = &f.nodes[nr.index];
            for dep in operands(&node.payload) {
                assert!(
                    pos[dep.index] < pos[nr.index],
                    "dependency must precede user"
                );
            }
        }
    }

    #[test]
    fn topo_linear_chain() {
        let f = parse_fn(
            r#"fn f() -> bits[1] {
  lit: bits[1] = literal(value=1, id=1)
  ret identity.2: bits[1] = identity(lit, id=2)
}"#,
        );
        let order = get_topological(&f);
        assert_eq!(order.len(), f.nodes.len());
        verify_topo_property(&f, &order);
    }

    #[test]
    fn topo_with_unreachable_node() {
        let f = parse_fn(
            r#"fn f() -> bits[1] {
  u: bits[1] = literal(value=0, id=1)
  lit: bits[1] = literal(value=1, id=2)
  ret identity.3: bits[1] = identity(lit, id=3)
}"#,
        );
        let order = get_topological(&f);
        assert_eq!(order.len(), f.nodes.len());
        verify_topo_property(&f, &order);
    }

    #[test]
    fn topo_large_chain_non_recursive() {
        // Build a long identity chain to stress recursion; here we expect non-recursive
        // handling.
        let mut ir = String::from("fn g(x: bits[1] id=1) -> bits[1] {\n");
        ir.push_str("  n2: bits[1] = identity(x, id=2)\n");
        let chain_len = 1024;
        for i in 3..(2 + chain_len) {
            let prev = i - 1;
            ir.push_str(&format!(
                "  n{}: bits[1] = identity(n{}, id={})\n",
                i, prev, i
            ));
        }
        let last = 1 + chain_len;
        ir.push_str(&format!(
            "  ret n{}: bits[1] = identity(n{}, id={})\n",
            last + 1,
            last,
            last + 1
        ));
        ir.push_str("}\n");
        let f = parse_fn(&ir);
        let order = get_topological(&f);
        assert_eq!(order.len(), f.nodes.len());
        verify_topo_property(&f, &order);
    }

    #[test]
    fn topo_two_independent_chains_depth_first_contiguous() {
        // Two independent chains A and B. DFS-based topo should list all A nodes
        // before all B nodes (no interleaving), then ret.
        let f = parse_fn(
            r#"fn f() -> bits[1] {
  a1: bits[1] = literal(value=1, id=1)
  a2: bits[1] = identity(a1, id=2)
  a3: bits[1] = identity(a2, id=3)
  b1: bits[1] = literal(value=0, id=4)
  b2: bits[1] = identity(b1, id=5)
  b3: bits[1] = identity(b2, id=6)
  ret r: bits[1] = identity(a3, id=7)
}"#,
        );
        let order = get_topological(&f);
        // Collect positions of nodes; indices 0..=2 are chain A, 3..=5 are chain B.
        let mut pos: Vec<usize> = vec![0; f.nodes.len()];
        for (i, nr) in order.iter().enumerate() {
            pos[nr.index] = i;
        }
        let a_indices = [0usize, 1, 2];
        let b_indices = [3usize, 4, 5];
        let max_a_pos = a_indices.iter().map(|i| pos[*i]).max().unwrap();
        let min_b_pos = b_indices.iter().map(|i| pos[*i]).min().unwrap();
        assert!(
            max_a_pos < min_b_pos,
            "DFS topo should not interleave independent chains"
        );
        // And ret should be last in topo (since it depends on a3 only).
        assert_eq!(order.last().unwrap().index, f.nodes.len() - 1);
    }

    #[test]
    fn remove_dead_nodes_keeps_params_and_live_graph() {
        let f = parse_fn(
            r#"fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  a: bits[8] = param(name=a, id=1)
  b: bits[8] = param(name=b, id=2)
  add.10: bits[8] = add(a, a, id=10)
  add.11: bits[8] = add(b, b, id=11)
  ret identity.12: bits[8] = identity(add.10, id=12)
}
"#,
        );
        let dead = get_dead_nodes(&f);
        assert!(
            dead.iter().any(|nr| {
                let n = &f.nodes[nr.index];
                matches!(n.payload, crate::ir::NodePayload::Binop(_, _, _)) && n.text_id == 11
            }),
            "expected add.11 to be dead"
        );
        let g = remove_dead_nodes(&f);
        // Validate function still has GetParam for both a and b (even if b was dead)
        let mut seen_params = 0usize;
        for node in g.nodes.iter() {
            if matches!(node.payload, crate::ir::NodePayload::GetParam(_)) {
                seen_params += 1;
            }
        }
        assert_eq!(
            seen_params, 2,
            "expected both GetParam nodes to be preserved"
        );
        // Ensure return remains and only live path nodes are present besides params.
        assert!(g.ret_node_ref.is_some());
        // Ensure no node references are out of bounds post-remap.
        for i in 0..g.nodes.len() {
            for dep in operands(&g.nodes[i].payload) {
                assert!(dep.index < g.nodes.len());
            }
        }
    }
}

#[cfg(test)]
mod remap_tests {
    use super::*;
    use crate::ir::{NodePayload, NodeRef};

    fn add(delta: usize) -> impl FnMut((usize, NodeRef)) -> NodeRef {
        move |(_i, nr): (usize, NodeRef)| NodeRef {
            index: nr.index + delta,
        }
    }

    #[test]
    fn remap_tuple_and_array_and_tuple_index() {
        let p_tuple = NodePayload::Tuple(vec![NodeRef { index: 1 }, NodeRef { index: 2 }]);
        let p_array = NodePayload::Array(vec![NodeRef { index: 3 }, NodeRef { index: 4 }]);
        let p_tidx = NodePayload::TupleIndex {
            tuple: NodeRef { index: 5 },
            index: 7,
        };

        let r_tuple = remap_payload_with(&p_tuple, add(10));
        let r_array = remap_payload_with(&p_array, add(10));
        let r_tidx = remap_payload_with(&p_tidx, add(10));

        match r_tuple {
            NodePayload::Tuple(v) => {
                assert_eq!(v, vec![NodeRef { index: 11 }, NodeRef { index: 12 }]);
            }
            _ => panic!("expected Tuple"),
        }
        match r_array {
            NodePayload::Array(v) => {
                assert_eq!(v, vec![NodeRef { index: 13 }, NodeRef { index: 14 }]);
            }
            _ => panic!("expected Array"),
        }
        match r_tidx {
            NodePayload::TupleIndex { tuple, index } => {
                assert_eq!(tuple, NodeRef { index: 15 });
                assert_eq!(index, 7);
            }
            _ => panic!("expected TupleIndex"),
        }
    }

    #[test]
    fn remap_sel_and_priority_sel_and_onehot_sel() {
        let p_sel = NodePayload::Sel {
            selector: NodeRef { index: 1 },
            cases: vec![NodeRef { index: 2 }, NodeRef { index: 3 }],
            default: Some(NodeRef { index: 4 }),
        };
        let p_psel = NodePayload::PrioritySel {
            selector: NodeRef { index: 5 },
            cases: vec![NodeRef { index: 6 }],
            default: None,
        };
        let p_ohsel = NodePayload::OneHotSel {
            selector: NodeRef { index: 7 },
            cases: vec![NodeRef { index: 8 }, NodeRef { index: 9 }],
        };

        let r_sel = remap_payload_with(&p_sel, add(100));
        let r_psel = remap_payload_with(&p_psel, add(100));
        let r_ohsel = remap_payload_with(&p_ohsel, add(100));

        match r_sel {
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                assert_eq!(selector, NodeRef { index: 101 });
                assert_eq!(cases, vec![NodeRef { index: 102 }, NodeRef { index: 103 }]);
                assert_eq!(default, Some(NodeRef { index: 104 }));
            }
            _ => panic!("expected Sel"),
        }
        match r_psel {
            NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } => {
                assert_eq!(selector, NodeRef { index: 105 });
                assert_eq!(cases, vec![NodeRef { index: 106 }]);
                assert_eq!(default, None);
            }
            _ => panic!("expected PrioritySel"),
        }
        match r_ohsel {
            NodePayload::OneHotSel { selector, cases } => {
                assert_eq!(selector, NodeRef { index: 107 });
                assert_eq!(cases, vec![NodeRef { index: 108 }, NodeRef { index: 109 }]);
            }
            _ => panic!("expected OneHotSel"),
        }
    }

    #[test]
    fn remap_dynamic_and_update_bit_slice_and_after_all() {
        let p_dyn = NodePayload::DynamicBitSlice {
            arg: NodeRef { index: 1 },
            start: NodeRef { index: 2 },
            width: 3,
        };
        let p_upd = NodePayload::BitSliceUpdate {
            arg: NodeRef { index: 4 },
            start: NodeRef { index: 5 },
            update_value: NodeRef { index: 6 },
        };
        let p_after = NodePayload::AfterAll(vec![NodeRef { index: 7 }, NodeRef { index: 8 }]);

        let r_dyn = remap_payload_with(&p_dyn, add(2));
        let r_upd = remap_payload_with(&p_upd, add(2));
        let r_after = remap_payload_with(&p_after, add(2));

        match r_dyn {
            NodePayload::DynamicBitSlice { arg, start, width } => {
                assert_eq!(arg, NodeRef { index: 3 });
                assert_eq!(start, NodeRef { index: 4 });
                assert_eq!(width, 3);
            }
            _ => panic!("expected DynamicBitSlice"),
        }
        match r_upd {
            NodePayload::BitSliceUpdate {
                arg,
                start,
                update_value,
            } => {
                assert_eq!(arg, NodeRef { index: 6 });
                assert_eq!(start, NodeRef { index: 7 });
                assert_eq!(update_value, NodeRef { index: 8 });
            }
            _ => panic!("expected BitSliceUpdate"),
        }
        match r_after {
            NodePayload::AfterAll(v) => {
                assert_eq!(v, vec![NodeRef { index: 9 }, NodeRef { index: 10 }]);
            }
            _ => panic!("expected AfterAll"),
        }
    }
}

#[cfg(test)]
mod users_tests {
    use super::*;
    use crate::ir::{NaryOp, PackageMember};
    use crate::ir_parser::Parser;

    fn parse_fn(ir: &str) -> Fn {
        let pkg_text = format!("package test\n\n{}\n", ir);
        let mut p = Parser::new(&pkg_text);
        let pkg = p.parse_and_validate_package().unwrap();
        pkg.members
            .iter()
            .filter_map(|m| match m {
                PackageMember::Function(f) => Some(f.clone()),
                _ => None,
            })
            .next()
            .unwrap()
    }

    #[test]
    fn users_linear_chain() {
        let f = parse_fn(
            r#"fn f() -> bits[1] {
  literal.1: bits[1] = literal(value=1, id=1)
  ret identity.2: bits[1] = identity(lit.1, id=2)
}"#,
        );
        let users = compute_users(&f);
        assert_eq!(users.len(), f.nodes.len());

        // Find literal and its sole user (the identity node) by payload relationships.
        let mut lit_ref: Option<NodeRef> = None;
        let mut idn_ref: Option<NodeRef> = None;
        for (i, node) in f.nodes.iter().enumerate() {
            match &node.payload {
                NodePayload::Literal(_) => lit_ref = Some(NodeRef { index: i }),
                NodePayload::Unop(_, arg) => {
                    // Identity will be the Unop consuming the literal.
                    if let Some(lr) = lit_ref {
                        if *arg == lr {
                            idn_ref = Some(NodeRef { index: i });
                        }
                    }
                }
                _ => {}
            }
        }
        let lit = lit_ref.expect("expected a literal node");
        let idn = idn_ref.expect("expected an identity node using the literal");
        assert!(users.get(&lit).unwrap().contains(&idn));
        assert!(users.get(&idn).unwrap().is_empty());
    }

    #[test]
    fn users_fanout_and_unreachable() {
        let f = parse_fn(
            r#"fn f() -> bits[1] {
  u: bits[1] = literal(value=0, id=1)
  a: bits[1] = literal(value=1, id=2)
  b: bits[1] = literal(value=0, id=3)
  and.4: bits[1] = and(a, b, id=4)
  ret identity.5: bits[1] = identity(and.4, id=5)
}"#,
        );
        let users = compute_users(&f);
        assert_eq!(users.len(), f.nodes.len());

        // Locate the key nodes via payload structure, not assumed indices.
        let mut u_ref: Option<NodeRef> = None;
        let mut a_ref: Option<NodeRef> = None;
        let mut b_ref: Option<NodeRef> = None;
        let mut and_ref: Option<NodeRef> = None;
        let mut ret_ref: Option<NodeRef> = None;

        // First, find the 'and' node and its two literal operands.
        for (i, node) in f.nodes.iter().enumerate() {
            if let NodePayload::Nary(NaryOp::And, elems) = &node.payload {
                assert_eq!(elems.len(), 2);
                and_ref = Some(NodeRef { index: i });
                a_ref = Some(elems[0]);
                b_ref = Some(elems[1]);
            }
        }
        let and = and_ref.expect("expected and node");
        let a = a_ref.expect("expected lhs operand");
        let b = b_ref.expect("expected rhs operand");

        // Find unreachable literal as the literal that is not an operand of 'and'.
        for (i, node) in f.nodes.iter().enumerate() {
            if let NodePayload::Literal(_) = &node.payload {
                let nr = NodeRef { index: i };
                if nr != a && nr != b {
                    u_ref = Some(nr);
                }
            }
        }
        let u = u_ref.expect("expected unreachable literal");

        // Find the ret identity node that consumes the 'and'.
        for (i, node) in f.nodes.iter().enumerate() {
            if let NodePayload::Unop(_, arg) = &node.payload {
                if *arg == and {
                    ret_ref = Some(NodeRef { index: i });
                }
            }
        }
        let ret = ret_ref.expect("expected ret identity using and");

        assert!(users.get(&u).unwrap().is_empty());
        assert!(users.get(&a).unwrap().contains(&and));
        assert!(users.get(&b).unwrap().contains(&and));
        assert!(users.get(&and).unwrap().contains(&ret));
        assert!(users.get(&ret).unwrap().is_empty());
    }

    #[test]
    fn dead_nodes_unreachable_literal() {
        let f = parse_fn(
            r#"fn f() -> bits[1] {
  u: bits[1] = literal(value=0, id=1)
  a: bits[1] = literal(value=1, id=2)
  b: bits[1] = literal(value=0, id=3)
  and.4: bits[1] = and(a, b, id=4)
  ret identity.5: bits[1] = identity(and.4, id=5)
}"#,
        );

        // Identify the unreachable literal node 'u' as the literal that is not
        // an operand of the 'and' node.
        let mut a_ref: Option<NodeRef> = None;
        let mut b_ref: Option<NodeRef> = None;
        let mut and_ref: Option<NodeRef> = None;
        for (i, node) in f.nodes.iter().enumerate() {
            if let NodePayload::Nary(NaryOp::And, elems) = &node.payload {
                assert_eq!(elems.len(), 2);
                and_ref = Some(NodeRef { index: i });
                a_ref = Some(elems[0]);
                b_ref = Some(elems[1]);
            }
        }
        let a = a_ref.expect("expected lhs operand");
        let b = b_ref.expect("expected rhs operand");

        let mut u_ref: Option<NodeRef> = None;
        for (i, node) in f.nodes.iter().enumerate() {
            if let NodePayload::Literal(_) = &node.payload {
                let nr = NodeRef { index: i };
                if nr != a && nr != b {
                    u_ref = Some(nr);
                }
            }
        }
        let u = u_ref.expect("expected unreachable literal");

        let dead = get_dead_nodes(&f);
        assert!(dead.contains(&u), "unreachable literal should be dead");
        assert!(!dead.contains(&a));
        assert!(!dead.contains(&b));
        assert!(!dead.contains(&and_ref.unwrap()));
    }
}
