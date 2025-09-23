// SPDX-License-Identifier: Apache-2.0

//! Dead-code elimination utilities for XLS IR functions.

use crate::ir::{Fn, NodeRef};
use crate::ir_utils::{operands, remap_payload_with};

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
        if matches!(node.payload, crate::ir::NodePayload::GetParam(_)) {
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
    let mut new_nodes: Vec<crate::ir::Node> = Vec::with_capacity(next);
    for (i, node) in f.nodes.iter().enumerate() {
        if !live[i] {
            continue;
        }
        let remapped_payload = remap_payload_with(&node.payload, |(_, nr): (usize, NodeRef)| {
            let ni = mapping[nr.index].expect("live node must not reference a dead operand");
            NodeRef { index: ni }
        });
        new_nodes.push(crate::ir::Node {
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

    crate::ir::Fn {
        name: f.name.clone(),
        params: f.params.clone(),
        ret_ty: f.ret_ty.clone(),
        nodes: new_nodes,
        ret_node_ref: Some(NodeRef { index: ret_new }),
        outer_attrs: f.outer_attrs.clone(),
        inner_attrs: f.inner_attrs.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{NaryOp, PackageMember};
    use crate::ir_parser::Parser;
    use crate::ir_utils::operands;

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
            if let crate::ir::NodePayload::Nary(NaryOp::And, elems) = &node.payload {
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
            if let crate::ir::NodePayload::Literal(_) = &node.payload {
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
