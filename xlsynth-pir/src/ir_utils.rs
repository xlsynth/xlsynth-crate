// SPDX-License-Identifier: Apache-2.0

//! Utility functions for working with / on XLS IR.

use crate::ir::{Fn, Node, NodePayload, NodeRef, Package, PackageMember, Type};
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

pub fn next_text_id(pkg: &Package) -> usize {
    pkg.members
        .iter()
        .flat_map(|member| match member {
            PackageMember::Function(f) => &f.nodes,
            PackageMember::Block { func, .. } => &func.nodes,
        })
        .map(|n| n.text_id)
        .max()
        .unwrap_or(0)
        + 1
}

/// Returns the `NodeRef` corresponding to the `index`-th parameter of `f`, if
/// it exists.
pub fn param_node_ref_by_index(f: &Fn, param_index: usize) -> Option<NodeRef> {
    let param = f.params.get(param_index)?;
    f.nodes
        .iter()
        .enumerate()
        .find_map(|(idx, node)| match node.payload {
            NodePayload::GetParam(pid) if pid == param.id => Some(NodeRef { index: idx }),
            _ => None,
        })
}

/// Returns the `NodeRef` corresponding to the parameter named `param_name` in
/// `f`, if any.
pub fn param_node_ref_by_name(f: &Fn, param_name: &str) -> Option<NodeRef> {
    let (index, _) = f
        .params
        .iter()
        .enumerate()
        .find(|(_, param)| param.name == param_name)?;
    param_node_ref_by_index(f, index)
}

/// Returns the `Type` of the `index`-th parameter of `f`, if it exists.
pub fn param_type_by_index(f: &Fn, param_index: usize) -> Option<Type> {
    f.params.get(param_index).map(|param| param.ty.clone())
}

/// Returns the `Type` of the parameter named `param_name` in `f`, if any.
pub fn param_type_by_name(f: &Fn, param_name: &str) -> Option<Type> {
    f.params
        .iter()
        .find(|param| param.name == param_name)
        .map(|param| param.ty.clone())

/// Compacts and reorders the nodes of a function in place.
///
/// - Removes any nodes whose payload is `Nil`.
/// - Reorders remaining nodes into a topological order (dependencies before
///   users).
/// - Remaps all operand indices and the function's `ret_node_ref` to the new
///   indices.
///
/// Returns `Err` if remapping encounters a reference to a removed (Nil) node.
pub fn compact_and_toposort_in_place(f: &mut Fn) -> Result<(), String> {
    // Determine a topological order over the current node set.
    let topo_all: Vec<NodeRef> = get_topological(f);

    // Filter out Nil nodes (these will be removed).
    let mut kept_order: Vec<NodeRef> = Vec::with_capacity(topo_all.len());
    for nr in topo_all.into_iter() {
        if !matches!(f.get_node(nr).payload, NodePayload::Nil) {
            kept_order.push(nr);
        }
    }

    // Build old->new index mapping for remapping payloads.
    let old_len = f.nodes.len();
    let mut old_to_new: Vec<Option<usize>> = vec![None; old_len];
    for (new_idx, nr) in kept_order.iter().enumerate() {
        old_to_new[nr.index] = Some(new_idx);
    }

    // Construct new node vector with remapped payloads.
    let mut new_nodes: Vec<Node> = Vec::with_capacity(kept_order.len());
    for nr in kept_order.iter().copied() {
        let src = f.get_node(nr).clone();
        let remapped_payload = remap_payload_with(&src.payload, |(_, dep): (usize, NodeRef)| {
            match old_to_new.get(dep.index).and_then(|x| *x) {
                Some(new_index) => NodeRef { index: new_index },
                None => {
                    // Encountered a dependency that was removed (Nil). This indicates the
                    // function still references a deleted node; surface an error.
                    panic!(
                        "compact_and_toposort_in_place: dependency {} was removed (Nil)",
                        dep.index
                    );
                }
            }
        });
        new_nodes.push(Node {
            payload: remapped_payload,
            ..src
        });
    }

    // Remap return node ref, if present.
    if let Some(old_ret) = f.ret_node_ref {
        let mapped = old_to_new[old_ret.index].ok_or_else(|| {
            format!(
                "compact_and_toposort_in_place: return node {} was removed (Nil)",
                old_ret.index
            )
        })?;
        f.ret_node_ref = Some(NodeRef { index: mapped });
    }

    // Install new nodes.
    f.nodes = new_nodes;
    Ok(())
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
    use crate::ir::{FileTable, PackageMember};
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
    fn next_text_id_advances() {
        let f = parse_fn(
            r#"fn f(x: bits[1] id=1) -> bits[1] {
  ret identity.2: bits[1] = identity(x, id=2)
}"#,
        );
        let pkg = Package {
            name: "test".to_string(),
            file_table: FileTable::new(),
            members: vec![PackageMember::Function(f.clone())],
            top_name: None,
        };
        let max_id = f
            .nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .expect("function has nodes");
        assert_eq!(next_text_id(&pkg), max_id + 1);
    }

    #[test]
    fn next_text_id_handles_zero_ids() {
        let mut f = parse_fn(
            r#"fn f() -> bits[1] {
  lit: bits[1] = literal(value=0, id=1)
  ret identity.2: bits[1] = identity(lit, id=2)
}"#,
        );
        for node in f.nodes.iter_mut() {
            node.text_id = 0;
        }
        let pkg = Package {
            name: "test".to_string(),
            file_table: FileTable::new(),
            members: vec![PackageMember::Function(f)],
            top_name: None,
        };
        assert_eq!(next_text_id(&pkg), 1);
    }

    #[test]
    fn param_node_lookups() {
        let f = parse_fn(
            r#"fn g(a: bits[8] id=1, b: bits[1] id=2) -> bits[8] {
  ret identity.3: bits[8] = identity(a, id=3)
}"#,
        );

        let a_ref = param_node_ref_by_index(&f, 0).expect("param 0 node");
        match &f.nodes[a_ref.index].payload {
            NodePayload::GetParam(pid) => assert_eq!(*pid, f.params[0].id),
            other => panic!("expected get_param, found {other:?}"),
        }

        let b_ref = param_node_ref_by_name(&f, "b").expect("param b node");
        match &f.nodes[b_ref.index].payload {
            NodePayload::GetParam(pid) => assert_eq!(*pid, f.params[1].id),
            other => panic!("expected get_param, found {other:?}"),
        }

        assert!(matches!(param_type_by_index(&f, 0), Some(Type::Bits(8))));
        assert!(matches!(param_type_by_name(&f, "b"), Some(Type::Bits(1))));
        assert!(param_node_ref_by_index(&f, 2).is_none());
        assert!(param_node_ref_by_name(&f, "missing").is_none());
        assert!(param_type_by_index(&f, 2).is_none());
        assert!(param_type_by_name(&f, "missing").is_none());
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
}
