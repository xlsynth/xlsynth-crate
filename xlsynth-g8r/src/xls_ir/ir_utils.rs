// SPDX-License-Identifier: Apache-2.0

//! Utility functions for working with / on XLS IR.

use crate::xls_ir::ir::{Fn, Node, NodePayload, NodeRef};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xls_ir::ir::PackageMember;
    use crate::xls_ir::ir_parser::Parser;

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
  lit.1: bits[1] = literal(value=1, id=1)
  ret identity.2: bits[1] = identity(lit.1, id=2)
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
  u.1: bits[1] = literal(value=0, id=1)
  lit.2: bits[1] = literal(value=1, id=2)
  ret identity.3: bits[1] = identity(lit.2, id=3)
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
        ir.push_str("  n1.2: bits[1] = identity(x, id=2)\n");
        let chain_len = 1024;
        for i in 3..(2 + chain_len) {
            let prev = i - 1;
            ir.push_str(&format!(
                "  n{}.{}: bits[1] = identity(n{}.{}, id={})\n",
                i, i, prev, prev, i
            ));
        }
        let last = 1 + chain_len;
        ir.push_str(&format!(
            "  ret n{}.{}: bits[1] = identity(n{}.{}, id={})\n",
            last + 1,
            last + 1,
            last,
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
  a1.1: bits[1] = literal(value=1, id=1)
  a2.2: bits[1] = identity(a1.1, id=2)
  a3.3: bits[1] = identity(a2.2, id=3)
  b1.4: bits[1] = literal(value=0, id=4)
  b2.5: bits[1] = identity(b1.4, id=5)
  b3.6: bits[1] = identity(b2.5, id=6)
  ret r.7: bits[1] = identity(a3.3, id=7)
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
}

pub fn remap_payload_with<FMap>(payload: &NodePayload, mut map: FMap) -> NodePayload
where
    FMap: FnMut(NodeRef) -> NodeRef,
{
    match payload {
        NodePayload::Nil => NodePayload::Nil,
        NodePayload::GetParam(p) => NodePayload::GetParam(*p),
        NodePayload::Tuple(elems) => NodePayload::Tuple(elems.iter().map(|r| map(*r)).collect()),
        NodePayload::Array(elems) => NodePayload::Array(elems.iter().map(|r| map(*r)).collect()),
        NodePayload::TupleIndex { tuple, index } => NodePayload::TupleIndex {
            tuple: map(*tuple),
            index: *index,
        },
        NodePayload::Binop(op, a, b) => NodePayload::Binop(*op, map(*a), map(*b)),
        NodePayload::Unop(op, a) => NodePayload::Unop(*op, map(*a)),
        NodePayload::Literal(v) => NodePayload::Literal(v.clone()),
        NodePayload::SignExt { arg, new_bit_count } => NodePayload::SignExt {
            arg: map(*arg),
            new_bit_count: *new_bit_count,
        },
        NodePayload::ZeroExt { arg, new_bit_count } => NodePayload::ZeroExt {
            arg: map(*arg),
            new_bit_count: *new_bit_count,
        },
        NodePayload::ArrayUpdate {
            array,
            value,
            indices,
            assumed_in_bounds,
        } => NodePayload::ArrayUpdate {
            array: map(*array),
            value: map(*value),
            indices: indices.iter().map(|r| map(*r)).collect(),
            assumed_in_bounds: *assumed_in_bounds,
        },
        NodePayload::ArrayIndex {
            array,
            indices,
            assumed_in_bounds,
        } => NodePayload::ArrayIndex {
            array: map(*array),
            indices: indices.iter().map(|r| map(*r)).collect(),
            assumed_in_bounds: *assumed_in_bounds,
        },
        NodePayload::DynamicBitSlice { arg, start, width } => NodePayload::DynamicBitSlice {
            arg: map(*arg),
            start: map(*start),
            width: *width,
        },
        NodePayload::BitSlice { arg, start, width } => NodePayload::BitSlice {
            arg: map(*arg),
            start: *start,
            width: *width,
        },
        NodePayload::BitSliceUpdate {
            arg,
            start,
            update_value,
        } => NodePayload::BitSliceUpdate {
            arg: map(*arg),
            start: map(*start),
            update_value: map(*update_value),
        },
        NodePayload::Assert {
            token,
            activate,
            message,
            label,
        } => NodePayload::Assert {
            token: map(*token),
            activate: map(*activate),
            message: message.clone(),
            label: label.clone(),
        },
        NodePayload::Trace {
            token,
            activated,
            format,
            operands,
        } => NodePayload::Trace {
            token: map(*token),
            activated: map(*activated),
            format: format.clone(),
            operands: operands.iter().map(|r| map(*r)).collect(),
        },
        NodePayload::AfterAll(elems) => {
            NodePayload::AfterAll(elems.iter().map(|r| map(*r)).collect())
        }
        NodePayload::Nary(op, elems) => {
            NodePayload::Nary(*op, elems.iter().map(|r| map(*r)).collect())
        }
        NodePayload::Invoke { to_apply, operands } => NodePayload::Invoke {
            to_apply: to_apply.clone(),
            operands: operands.iter().map(|r| map(*r)).collect(),
        },
        NodePayload::PrioritySel {
            selector,
            cases,
            default,
        } => NodePayload::PrioritySel {
            selector: map(*selector),
            cases: cases.iter().map(|r| map(*r)).collect(),
            default: default.map(|d| map(d)),
        },
        NodePayload::OneHotSel { selector, cases } => NodePayload::OneHotSel {
            selector: map(*selector),
            cases: cases.iter().map(|r| map(*r)).collect(),
        },
        NodePayload::OneHot { arg, lsb_prio } => NodePayload::OneHot {
            arg: map(*arg),
            lsb_prio: *lsb_prio,
        },
        NodePayload::Sel {
            selector,
            cases,
            default,
        } => NodePayload::Sel {
            selector: map(*selector),
            cases: cases.iter().map(|r| map(*r)).collect(),
            default: default.map(|d| map(d)),
        },
        NodePayload::Cover { predicate, label } => NodePayload::Cover {
            predicate: map(*predicate),
            label: label.clone(),
        },
        NodePayload::Decode { arg, width } => NodePayload::Decode {
            arg: map(*arg),
            width: *width,
        },
        NodePayload::Encode { arg } => NodePayload::Encode { arg: map(*arg) },
        NodePayload::CountedFor {
            init,
            trip_count,
            stride,
            body,
            invariant_args,
        } => NodePayload::CountedFor {
            init: map(*init),
            trip_count: *trip_count,
            stride: *stride,
            body: body.clone(),
            invariant_args: invariant_args.iter().map(|r| map(*r)).collect(),
        },
    }
}
