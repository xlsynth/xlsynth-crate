// SPDX-License-Identifier: Apache-2.0

//! Simple rebasing utility: rebuilds a desired function on top of an existing
//! implementation, preserving all original nodes (and their text ids) from the
//! existing function wherever structurally equivalent subgraphs are found.

use std::collections::HashMap;
use std::collections::HashSet;

use crate::ir::{Fn as IrFn, Node, NodeRef, Type};
use crate::ir_utils::{get_topological, get_topological_nodes, remap_payload_with};
use crate::structural_similarity::collect_structural_entries;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DesiredNodeRef(NodeRef);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ResultNodeRef(NodeRef);

/// Rebase `desired` onto `existing`, preserving all original nodes (and their
/// text ids) from `existing` wherever structurally equivalent nodes exist.
///
/// Precondition: `desired` and `existing` must have identical function types.
pub fn rebase_onto(
    desired: &IrFn,
    existing: &IrFn,
    result_fn_name: &str,
    mut gen_new_id: impl FnMut() -> usize,
) -> IrFn {
    assert_eq!(
        desired.get_type(),
        existing.get_type(),
        "Function signatures must match"
    );

    // Prepare structural hash maps for existing and desired.
    let (existing_entries, _ed) = collect_structural_entries(existing);
    let (desired_entries, _dd) = collect_structural_entries(desired);

    // Hash -> queue of existing node indices (stable ascending order)
    let mut existing_by_hash: HashMap<[u8; 32], Vec<usize>> = HashMap::new();
    for (i, e) in existing_entries.iter().enumerate() {
        let mut k = [0u8; 32];
        k.copy_from_slice(e.hash.as_bytes());
        existing_by_hash.entry(k).or_default().push(i);
    }
    for v in existing_by_hash.values_mut() {
        v.sort_unstable();
    }

    // Start result function with a clone of `existing`'s parameters and nodes.
    let mut result_nodes: Vec<Node> = existing.nodes.clone();
    // Track used names in the result to avoid duplicate-name collisions.
    let mut used_names: HashSet<String> = HashSet::new();
    for n in result_nodes.iter() {
        if let Some(name) = &n.name {
            used_names.insert(name.clone());
        }
    }
    // For deterministic uniquification, track the next numeric suffix per base
    // name.
    let mut next_suffix_for: HashMap<String, usize> = HashMap::new();
    let mut make_unique_name = |base: &str| -> String {
        if !used_names.contains(base) {
            used_names.insert(base.to_string());
            return base.to_string();
        }
        let mut suffix = *next_suffix_for.get(base).unwrap_or(&1);
        loop {
            let candidate = format!("{}__{}", base, suffix);
            if !used_names.contains(&candidate) {
                used_names.insert(candidate.clone());
                next_suffix_for.insert(base.to_string(), suffix + 1);
                return candidate;
            }
            suffix += 1;
        }
    };
    let result_params = existing.params.clone();
    let result_ret_ty: Type = desired.ret_ty.clone();

    // Mapping from desired NodeRef -> result NodeRef (strongly-typed wrappers).
    let mut desired_to_result: HashMap<DesiredNodeRef, ResultNodeRef> = HashMap::new();

    // Walk `desired` in topological order as the work queue for what we need to
    // build; reuse equivalent existing nodes when possible.
    let topo_desired = get_topological(desired)
        .into_iter()
        .map(|nr| DesiredNodeRef(nr))
        .collect::<Vec<_>>();
    for d_ref in topo_desired.into_iter() {
        // Note: NodeRef.index is a dense index into `Fn.nodes` (0..nodes.len()),
        // not the textual node id. `desired_entries` was built by enumerating
        // `desired.nodes`, so `desired_entries[d_idx]` corresponds to this index.
        let d_idx = d_ref.0.index;
        let d_entry = &desired_entries[d_idx];
        let mut key = [0u8; 32];
        key.copy_from_slice(d_entry.hash.as_bytes());

        // Try to consume an equivalent node from existing by structural hash.
        let reuse: Option<usize> = existing_by_hash.get_mut(&key).and_then(|v| {
            if v.is_empty() {
                None
            } else {
                Some(v.remove(0))
            }
        });

        if let Some(e_idx) = reuse {
            desired_to_result.insert(
                DesiredNodeRef(NodeRef { index: d_idx }),
                ResultNodeRef(NodeRef { index: e_idx }),
            );
            continue;
        }

        // Otherwise, create a new node by remapping desired's payload operands
        // through the mapping we've built so far.
        let d_node = &desired.nodes[d_idx];
        let mapper = |r: NodeRef| -> NodeRef {
            if let Some(&ResultNodeRef(mapped)) = desired_to_result.get(&DesiredNodeRef(r)) {
                mapped
            } else {
                r
            }
        };
        let new_payload = remap_payload_with(&d_node.payload, mapper);
        // Ensure any new node name does not collide with names already present in the
        // result.
        let new_name: Option<String> = match &d_node.name {
            Some(s) => Some(make_unique_name(s)),
            None => None,
        };
        let new_node = Node {
            text_id: gen_new_id(),
            name: new_name,
            ty: d_node.ty.clone(),
            payload: new_payload,
            pos: d_node.pos.clone(),
        };
        let new_index = result_nodes.len();
        result_nodes.push(new_node);
        desired_to_result.insert(
            DesiredNodeRef(NodeRef { index: d_idx }),
            ResultNodeRef(NodeRef { index: new_index }),
        );
    }

    // Determine the return node reference in the result space.
    let result_ret_ref: Option<NodeRef> = desired.ret_node_ref.map(|nr| {
        desired_to_result
            .get(&DesiredNodeRef(nr))
            .copied()
            .unwrap_or(ResultNodeRef(nr))
            .0
    });

    // Topologically sort the resulting nodes
    let order = get_topological_nodes(&result_nodes)
        .into_iter()
        .map(|nr| ResultNodeRef(nr))
        .collect::<Vec<_>>();
    let mut old_to_new: Vec<usize> = vec![0; result_nodes.len()];
    for (new_idx, nr) in order.iter().enumerate() {
        old_to_new[nr.0.index] = new_idx;
    }

    let mut remapped_nodes: Vec<Node> = Vec::with_capacity(result_nodes.len());
    for nr in order.into_iter() {
        let old = &result_nodes[nr.0.index];
        let mapper = |r: NodeRef| -> NodeRef {
            NodeRef {
                index: old_to_new[r.index],
            }
        };
        let new_payload = remap_payload_with(&old.payload, mapper);
        remapped_nodes.push(Node {
            text_id: old.text_id,
            name: old.name.clone(),
            ty: old.ty.clone(),
            payload: new_payload,
            pos: old.pos.clone(),
        });
    }
    let remapped_ret = result_ret_ref.map(|nr| NodeRef {
        index: old_to_new[nr.index],
    });

    IrFn {
        name: result_fn_name.to_string(),
        params: result_params,
        ret_ty: result_ret_ty,
        nodes: remapped_nodes,
        ret_node_ref: remapped_ret,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Binop, NaryOp, NodePayload, PackageMember};
    use crate::ir_parser::Parser;

    fn parse_fn(ir: &str) -> IrFn {
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

    fn check_desired_equivalence(desired: &IrFn, result: &IrFn) {
        let res =
            xlsynth_g8r::equiv::prove_equiv_via_toolchain::prove_ir_fn_strings_equiv_via_toolchain(
                &desired.to_string(),
                &result.to_string(),
            );
        assert!(
            matches!(res, xlsynth_g8r::equiv::types::EquivResult::Proved),
            "Toolchain IR equivalence failed: {:?}",
            res
        );
    }

    #[test]
    fn identity_with_param_renames_full_reuse() {
        let existing = parse_fn(
            r#"fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret add.3: bits[8] = add(a, b, id=3)
}"#,
        );
        let desired = parse_fn(
            r#"fn f(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  ret add.3: bits[8] = add(x, y, id=3)
}"#,
        );
        let mut next_id = 1000usize;
        let result = rebase_onto(&desired, &existing, "f_rebased", || {
            let id = next_id;
            next_id += 1;
            id
        });
        assert_eq!(result.nodes.len(), existing.nodes.len());
        let existing_ret_idx = existing.ret_node_ref.unwrap().index;
        let result_ret_idx = result.ret_node_ref.unwrap().index;
        assert_eq!(
            result.nodes[result_ret_idx].text_id,
            existing.nodes[existing_ret_idx].text_id
        );
        // Topological orders should match existing since no new nodes were added.
        let order_existing: Vec<usize> = get_topological(&existing)
            .into_iter()
            .map(|nr| nr.index)
            .collect();
        let order_result: Vec<usize> = get_topological(&result)
            .into_iter()
            .map(|nr| nr.index)
            .collect();
        assert_eq!(order_result, order_existing);

        let expected = r#"fn f_rebased(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret add.3: bits[8] = add(a, b, id=3)
}"#;
        assert_eq!(result.to_string(), expected);

        // Semantic equivalence: desired == result
        check_desired_equivalence(&desired, &result);
    }

    #[test]
    fn basic_partial_reuse_add_reused_mul_added() {
        // existing: t = add(a,b); ret t; c is unused param
        let existing = parse_fn(
            r#"fn g(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  ret add.4: bits[8] = add(a, b, id=4)
}"#,
        );
        // desired: umul(add(a,b), c)
        let desired = parse_fn(
            r#"fn g(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  add.4: bits[8] = add(a, b, id=4)
  ret umul.5: bits[8] = umul(add.4, c, id=5)
}"#,
        );
        let mut next_id = 2000usize;
        let result = rebase_onto(&desired, &existing, "g_rebased", || {
            let id = next_id;
            next_id += 1;
            id
        });
        assert_eq!(result.nodes.len(), existing.nodes.len() + 1);
        // Ensure the mul node exists and depends on the existing add node (not a new
        // add)
        let mut mul_idx: Option<usize> = None;
        for (i, n) in result.nodes.iter().enumerate() {
            if let NodePayload::Binop(Binop::Umul, a, _c) = &n.payload {
                mul_idx = Some(i);
                // The operand 'a' should refer to a reused existing add node via text_id
                let existing_add_text_id = existing
                    .nodes
                    .iter()
                    .find_map(|en| match &en.payload {
                        NodePayload::Binop(Binop::Add, _, _) => Some(en.text_id),
                        _ => None,
                    })
                    .unwrap();
                assert_eq!(result.nodes[a.index].text_id, existing_add_text_id);
            }
        }
        assert!(mul_idx.is_some());

        let expected = r#"fn g_rebased(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  add.4: bits[8] = add(a, b, id=4)
  ret umul.2000: bits[8] = umul(add.4, c, id=2000)
}"#;
        assert_eq!(result.to_string(), expected);

        // Semantic equivalence: desired == result
        check_desired_equivalence(&desired, &result);
    }

    #[test]
    fn operand_order_mismatch_no_reuse() {
        // existing: add(b,a)
        let existing = parse_fn(
            r#"fn h(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret add.3: bits[8] = add(b, a, id=3)
}"#,
        );
        // desired: add(a,b)
        let desired = parse_fn(
            r#"fn h(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret add.3: bits[8] = add(a, b, id=3)
}"#,
        );
        let new_id = 3001usize; // deterministic id for new node
        let mut issued = false;
        let result = rebase_onto(&desired, &existing, "h_rebased", || {
            if issued {
                0
            } else {
                issued = true;
                new_id
            }
        });
        assert_eq!(result.nodes.len(), existing.nodes.len() + 1);
        // The return node must be the newly created add with text_id = new_id
        let ret_idx = result.ret_node_ref.unwrap().index;
        assert_eq!(result.nodes[ret_idx].text_id, new_id);

        let expected = r#"fn h_rebased(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  add.3: bits[8] = add(b, a, id=3)
  ret add.3001: bits[8] = add(a, b, id=3001)
}"#;
        assert_eq!(result.to_string(), expected);

        // Semantic equivalence: desired == result
        check_desired_equivalence(&desired, &result);
    }

    #[test]
    fn attribute_sensitive_mismatch_slice_bounds_differ() {
        // existing: bit_slice(x, start=0, width=2)
        let existing = parse_fn(
            r#"fn s(x: bits[8] id=1) -> bits[2] {
  ret bit_slice.2: bits[2] = bit_slice(x, start=0, width=2, id=2)
}"#,
        );
        // desired: bit_slice(x, start=1, width=2)
        let desired = parse_fn(
            r#"fn s(x: bits[8] id=1) -> bits[2] {
  ret bit_slice.2: bits[2] = bit_slice(x, start=1, width=2, id=2)
}"#,
        );
        let new_id = 4001usize;
        let mut issued = false;
        let result = rebase_onto(&desired, &existing, "s_rebased", || {
            if issued {
                0
            } else {
                issued = true;
                new_id
            }
        });
        assert_eq!(result.nodes.len(), existing.nodes.len() + 1);
        let ret_idx = result.ret_node_ref.unwrap().index;
        assert_eq!(result.nodes[ret_idx].text_id, new_id);

        let expected = r#"fn s_rebased(x: bits[8] id=1) -> bits[2] {
  bit_slice.2: bits[2] = bit_slice(x, start=0, width=2, id=2)
  ret bit_slice.4001: bits[2] = bit_slice(x, start=1, width=2, id=4001)
}"#;
        assert_eq!(result.to_string(), expected);

        // Semantic equivalence: desired == result
        check_desired_equivalence(&desired, &result);
    }

    #[test]
    fn duplicate_subgraphs_tiebreak_deterministic_lowest_index() {
        // existing has two identical adds
        let existing = parse_fn(
            r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  add.3: bits[8] = add(a, b, id=3)
  ret add.4: bits[8] = add(a, b, id=4)
}"#,
        );
        // desired needs one add(a,b)
        let desired = parse_fn(
            r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret add.3: bits[8] = add(a, b, id=3)
}"#,
        );
        let result = rebase_onto(&desired, &existing, "t_rebased", || 5000);
        // No new nodes
        assert_eq!(result.nodes.len(), existing.nodes.len());
        // The chosen return should be the lower of the two identical add indices
        // (deterministic)
        let add_indices: Vec<usize> = existing
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| match n.payload {
                NodePayload::Binop(Binop::Add, _, _) => Some(i),
                _ => None,
            })
            .collect();
        let chosen = add_indices.into_iter().min().unwrap();
        assert_eq!(result.ret_node_ref.unwrap().index, chosen);

        let expected = r#"fn t_rebased(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret add.3: bits[8] = add(a, b, id=3)
  add.4: bits[8] = add(a, b, id=4)
}"#;
        assert_eq!(result.to_string(), expected);

        // Semantic equivalence: desired == result
        check_desired_equivalence(&desired, &result);
    }

    #[test]
    fn name_collision_new_node_uniquified() {
        let _ = env_logger::builder().is_test(true).try_init();
        // existing has a named node 'foo'
        let existing = parse_fn(
            r#"fn c(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  ret foo: bits[8] = add(x, y, id=3)
}"#,
        );
        // desired reuses the add node (with a different name) and adds a NEW node also
        // named 'foo'. This would collide with existing's 'foo' unless we
        // uniquify new-node names on rebase.
        let desired = parse_fn(
            r#"fn c(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  t: bits[8] = add(x, y, id=3)
  ret foo: bits[8] = umul(t, y, id=4)
}"#,
        );
        let mut next_id = 7000usize;
        let result = rebase_onto(&desired, &existing, "c_rebased", || {
            let id = next_id;
            next_id += 1;
            id
        });

        let expected = r#"fn c_rebased(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  foo: bits[8] = add(x, y, id=3)
  ret foo__1: bits[8] = umul(foo, y, id=7000)
}"#;
        assert_eq!(result.to_string(), expected);

        // Verify parse/validate of the pretty-printed result succeeds (no duplicate
        // names).
        let pkg_text = format!("package test\n\n top {}\n", result.to_string());
        let mut p = Parser::new(&pkg_text);
        let parsed_pkg = p.parse_and_validate_package();
        assert!(
            parsed_pkg.is_ok(),
            "rebased function should be parseable without duplicate-name errors"
        );

        // Ensure there are no duplicate names among nodes in the result.
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for n in result.nodes.iter() {
            if let Some(name) = &n.name {
                assert!(
                    seen.insert(name.clone()),
                    "duplicate node name found in result: {}",
                    name
                );
            }
        }
        // And the added node does not keep the conflicting base name 'foo' if it was
        // already used. There should be at most one exact 'foo' in the result.
        let foo_count = result
            .nodes
            .iter()
            .filter(|n| n.name.as_deref() == Some("foo"))
            .count();
        assert!(foo_count <= 1);
    }

    #[test]
    fn preserve_all_with_dead_code_plus_new_work() {
        // existing: dead or node plus live and
        let existing = parse_fn(
            r#"fn u(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  or.5: bits[8] = or(a, b, id=5)
  ret and.4: bits[8] = and(a, b, id=4)
}"#,
        );
        // desired: xor(and(a,b), c)
        let desired = parse_fn(
            r#"fn u(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  and.4: bits[8] = and(a, b, id=4)
  ret xor.6: bits[8] = xor(and.4, c, id=6)
}"#,
        );
        let mut next_id = 6000usize;
        let result = rebase_onto(&desired, &existing, "u_rebased", || {
            let id = next_id;
            next_id += 1;
            id
        });
        // All existing nodes preserved, one new node added
        assert_eq!(result.nodes.len(), existing.nodes.len() + 1);
        // Existing text_ids preserved
        let existing_ids: std::collections::HashSet<usize> =
            existing.nodes.iter().map(|n| n.text_id).collect();
        let result_ids: std::collections::HashSet<usize> =
            result.nodes.iter().map(|n| n.text_id).collect();
        for id in existing_ids.iter() {
            assert!(result_ids.contains(id));
        }
        // Ensure the new XOR node has a generated id in the expected range
        let gen_min = 6000usize;
        assert!(
            result
                .nodes
                .iter()
                .any(|n| matches!(n.payload, NodePayload::Nary(NaryOp::Xor, _))
                    && n.text_id >= gen_min)
        );
        // And ret is reachable
        assert!(result.ret_node_ref.is_some());

        let expected = r#"fn u_rebased(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  or.5: bits[8] = or(a, b, id=5)
  and.4: bits[8] = and(a, b, id=4)
  ret xor.6000: bits[8] = xor(and.4, c, id=6000)
}"#;
        assert_eq!(result.to_string(), expected);

        // Semantic equivalence: desired == result
        check_desired_equivalence(&desired, &result);
    }
}
