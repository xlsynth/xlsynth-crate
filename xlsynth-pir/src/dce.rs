// SPDX-License-Identifier: Apache-2.0

//! Dead-code elimination utilities for XLS IR functions.

use crate::ir::{Fn, NodeRef};
use crate::ir_utils::{compact_and_toposort_in_place, is_observable_effect_root, operands};

/// Computes nodes required by the return value or an observable effect.
fn compute_live_nodes(f: &Fn) -> Vec<bool> {
    let mut live: Vec<bool> = vec![false; f.nodes.len()];
    let mut stack: Vec<NodeRef> = vec![
        f.ret_node_ref
            .expect("DCE requires a function with a return node"),
    ];
    stack.extend(
        f.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| is_observable_effect_root(&node.payload))
            .map(|(index, _)| NodeRef { index }),
    );
    while let Some(nr) = stack.pop() {
        if live[nr.index] {
            continue;
        }
        live[nr.index] = true;
        for dep in operands(&f.get_node(nr).payload) {
            if !live[dep.index] {
                stack.push(dep);
            }
        }
    }
    live
}

/// Returns a list of nodes that are not required by the return value or any
/// observable effect root.
///
/// The returned vector is sorted by node index ascending to ensure
/// deterministic ordering.
pub fn get_dead_nodes(f: &Fn) -> Vec<NodeRef> {
    let n = f.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    assert!(
        f.ret_node_ref.is_some(),
        "get_dead_nodes: function has no return node"
    );
    let live = compute_live_nodes(f);

    // Dead nodes are those never marked live.
    let mut dead: Vec<NodeRef> = Vec::new();
    for i in 0..n {
        if !live[i] {
            dead.push(NodeRef { index: i });
        }
    }
    dead
}

/// Returns a new function with nodes irrelevant to its return value and
/// observable effects removed, and all remaining node indices compacted.
/// Operand references are remapped to the new indices. GetParam nodes are
/// preserved even if they would otherwise be considered dead, to satisfy
/// validation rules requiring a GetParam for each declared parameter.
pub fn remove_dead_nodes(f: &Fn) -> Fn {
    let n = f.nodes.len();
    assert!(n > 0, "remove_dead_nodes: function has no nodes");
    assert!(
        f.ret_node_ref.is_some(),
        "remove_dead_nodes: function has no return node"
    );

    let live = compute_live_nodes(f);

    // Always keep layout-invariant nodes:
    // - node[0] is reserved Nil
    // - params occupy indices 1..=params.len() in signature order
    //
    // We mark dead body nodes as Nil, then use `compact_and_toposort_in_place`
    // to remove those Nil nodes and remap indices while preserving the layout
    // invariants.
    let mut g: Fn = f.clone();
    let param_count = g.params.len();
    for i in 0..n {
        if i == 0 || (1..=param_count).contains(&i) {
            continue;
        }
        if !live[i] {
            g.nodes[i].payload = crate::ir::NodePayload::Nil;
        }
    }

    compact_and_toposort_in_place(&mut g).expect("remove_dead_nodes: compaction failed");
    debug_assert!(
        g.check_pir_layout_invariants().is_ok(),
        "remove_dead_nodes: PIR layout invariants violated for '{}'",
        g.name
    );
    g
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
        g.check_pir_layout_invariants().unwrap();
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

    #[test]
    fn remove_dead_nodes_preserves_effect_nodes_and_their_token_chain() {
        let f = parse_fn(
            r#"fn f(x: bits[1] id=1) -> bits[1] {
  x: bits[1] = param(name=x, id=1)
  tok: token = after_all(id=2)
  tr: token = trace(tok, x, format="x={}", data_operands=[x], id=3)
  cv: () = cover(x, label="hit", id=4)
  as: token = assert(tr, x, message="failed", label="a", id=5)
  dead: bits[1] = not(x, id=6)
  ret out: bits[1] = identity(x, id=7)
}"#,
        );

        let dead = get_dead_nodes(&f);
        assert!(dead.iter().any(|nr| f.get_node(*nr).text_id == 6));
        for effect_id in [2, 3, 4, 5] {
            assert!(
                dead.iter().all(|nr| f.get_node(*nr).text_id != effect_id),
                "effect/token dependency node {effect_id} was classified as dead"
            );
        }

        let g = remove_dead_nodes(&f);
        for effect_id in [2, 3, 4, 5] {
            assert!(g.nodes.iter().any(|node| node.text_id == effect_id));
        }
        assert!(g.nodes.iter().all(|node| node.text_id != 6));
    }

    #[test]
    fn remove_dead_nodes_may_remove_unused_assumed_in_bounds_accesses() {
        let f = parse_fn(
            r#"fn f(a: bits[8][2] id=1, v: bits[8] id=2, i: bits[2] id=3) -> bits[8] {
  a: bits[8][2] = param(name=a, id=1)
  v: bits[8] = param(name=v, id=2)
  i: bits[2] = param(name=i, id=3)
  dead_index: bits[8] = array_index(a, indices=[i], assumed_in_bounds=true, id=4)
  dead_update: bits[8][2] = array_update(a, v, indices=[i], assumed_in_bounds=true, id=5)
  ret out: bits[8] = identity(v, id=6)
}"#,
        );

        let dead = get_dead_nodes(&f);
        for id in [4, 5] {
            assert!(dead.iter().any(|nr| f.get_node(*nr).text_id == id));
        }
        let g = remove_dead_nodes(&f);
        assert!(g.nodes.iter().all(|node| !matches!(node.text_id, 4 | 5)));
    }
}
