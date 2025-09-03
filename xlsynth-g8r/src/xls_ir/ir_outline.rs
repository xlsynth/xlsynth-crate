// SPDX-License-Identifier: Apache-2.0

//! Outlining transformation for XLS IR functions.
//!
//! Given an `IrFn` and a set of nodes to outline, builds a new inner function
//! consisting of that subgraph and rewrites the outer function to invoke it.
//! The function signature of the inner is discovered from boundary inputs and
//! outputs of the selected subgraph.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::xls_ir::ir::{
    Fn as IrFn, Node, NodePayload, NodeRef, Package, PackageMember, Param, ParamId, Type,
};
use crate::xls_ir::ir_utils::{
    get_topological, get_topological_nodes, operands, remap_payload_with,
};

#[derive(Debug, Clone)]
pub struct OutlineResult {
    pub outer: IrFn,
    pub inner: IrFn,
}

fn next_text_id(nodes: &[Node]) -> usize {
    let mut max_id: usize = 0;
    for n in nodes.iter() {
        if n.text_id > max_id {
            max_id = n.text_id;
        }
    }
    max_id.saturating_add(1)
}

/// Computes a stable name for an external argument derived from a source node.
fn derive_arg_name(n: &Node, fallback_index: usize, used: &mut HashSet<String>) -> String {
    let base = n
        .name
        .clone()
        .unwrap_or_else(|| format!("arg_{}", fallback_index));
    if !used.contains(&base) {
        used.insert(base.clone());
        return base;
    }
    // Deterministically uniquify by appending __<k>
    let mut k: usize = 1;
    loop {
        let candidate = format!("{}__{}", base, k);
        if !used.contains(&candidate) {
            used.insert(candidate.clone());
            return candidate;
        }
        k += 1;
    }
}

/// Returns true if any node in `users` uses `target` as an operand.
fn is_used_by_any(users: &[Node], target: NodeRef) -> bool {
    for u in users.iter() {
        if operands(&u.payload).iter().any(|&r| r == target) {
            return true;
        }
    }
    false
}

/// Outline `to_outline` from `outer` into a new inner function. Returns the
/// rewritten outer and the new inner; also registers both into `package`.
///
/// Preconditions/invariants asserted:
/// - `to_outline` is a subset of nodes in `outer`.
///
/// This transformation does not introduce recursion: the inner function is a
/// pure extraction of the selected subgraph and the outer simply invokes it.
pub fn outline(
    outer: &IrFn,
    to_outline: &HashSet<NodeRef>,
    new_outer_name: &str,
    new_inner_name: &str,
    package: &mut Package,
) -> OutlineResult {
    // Basic sanity on indices
    for nr in to_outline.iter() {
        assert!(
            nr.index < outer.nodes.len(),
            "NodeRef out of bounds: {:?}",
            nr
        );
    }

    // Stable sets for boundary analysis
    let to_outline_set: HashSet<usize> = to_outline.iter().map(|r| r.index).collect();
    let mut inputs_ext: BTreeSet<usize> = BTreeSet::new();
    let mut outputs_boundary: BTreeSet<usize> = BTreeSet::new();

    // Identify external inputs: any operand of a selected node that is not
    // selected.
    for &idx in to_outline_set.iter() {
        let n = &outer.nodes[idx];
        for dep in operands(&n.payload).into_iter() {
            if !to_outline_set.contains(&dep.index) {
                inputs_ext.insert(dep.index);
            }
        }
    }

    // Identify boundary outputs: any selected node used by a node outside the set.
    let mut outside_nodes: Vec<Node> = Vec::new();
    for (i, n) in outer.nodes.iter().enumerate() {
        if !to_outline_set.contains(&i) {
            outside_nodes.push(n.clone());
        }
    }
    for &idx in to_outline_set.iter() {
        let target = NodeRef { index: idx };
        if is_used_by_any(&outside_nodes, target) {
            outputs_boundary.insert(idx);
        }
    }
    if let Some(ret_nr) = outer.ret_node_ref {
        if to_outline_set.contains(&ret_nr.index) {
            outputs_boundary.insert(ret_nr.index);
        }
    }

    // Collect all ParamIds used by the subgraph (whether the GetParam node is
    // inside or outside the subgraph). We'll pull names/types from the outer
    // signature by id.
    let mut used_param_ids: BTreeMap<usize, (String, Type)> = BTreeMap::new();
    // Build a quick map from ParamId -> (name, ty) from the outer signature.
    let mut outer_param_info: HashMap<usize, (String, Type)> = HashMap::new();
    for p in outer.params.iter() {
        outer_param_info.insert(p.id.get_wrapped_id(), (p.name.clone(), p.ty.clone()));
    }

    // Any GetParam node inside the region contributes its ParamId.
    for &idx in to_outline_set.iter() {
        if let NodePayload::GetParam(pid) = outer.nodes[idx].payload {
            let key = pid.get_wrapped_id();
            let (nm, ty) = outer_param_info
                .get(&key)
                .cloned()
                .expect("outer params must include referenced ParamId");
            used_param_ids.entry(key).or_insert((nm, ty));
        }
    }
    // Any GetParam node used as an external input contributes its ParamId as well.
    for &idx in inputs_ext.iter() {
        if let NodePayload::GetParam(pid) = outer.nodes[idx].payload {
            let key = pid.get_wrapped_id();
            let (nm, ty) = outer_param_info
                .get(&key)
                .cloned()
                .expect("outer params must include referenced ParamId");
            used_param_ids.entry(key).or_insert((nm, ty));
        }
    }

    // External non-parameter inputs (must be passed positionally to inner)
    let mut ext_nonparam_inputs: Vec<usize> = inputs_ext
        .iter()
        .copied()
        .filter(|idx| !matches!(outer.nodes[*idx].payload, NodePayload::GetParam(_)))
        .collect();
    ext_nonparam_inputs.sort_unstable();

    // Build the inner param list: first all ParamIds in ascending id order, then
    // all external non-param inputs in ascending node index order.
    let mut inner_params: Vec<Param> = Vec::new();
    let mut outer_paramid_to_inner: HashMap<usize, ParamId> = HashMap::new();
    let mut used_names: HashSet<String> = HashSet::new();
    let mut next_param_pos: usize = 1;
    for (pid_num, (nm, ty)) in used_param_ids.iter() {
        let inner_id = ParamId::new(next_param_pos);
        next_param_pos += 1;
        used_names.insert(nm.clone());
        inner_params.push(Param {
            name: nm.clone(),
            ty: ty.clone(),
            id: inner_id,
        });
        outer_paramid_to_inner.insert(*pid_num, inner_id);
    }
    // We'll also need an index for naming fallbacks of external inputs.
    for (pos, idx) in ext_nonparam_inputs.iter().enumerate() {
        let src_node = &outer.nodes[*idx];
        let name = derive_arg_name(src_node, pos, &mut used_names);
        let inner_id = ParamId::new(next_param_pos);
        next_param_pos += 1;
        inner_params.push(Param {
            name,
            ty: src_node.ty.clone(),
            id: inner_id,
        });
    }

    // Build inner nodes: create GetParam nodes for any ParamIds that do not already
    // exist inside the selected set (i.e. when the GetParam node itself is
    // external), and for all external non-param inputs. Track a map from outer
    // NodeRef -> inner NodeRef for remapping payload operands.
    let mut inner_nodes: Vec<Node> = Vec::new();
    let mut ext_ref_to_inner_ref: HashMap<usize, NodeRef> = HashMap::new();
    let mut next_inner_text_id: usize = next_text_id(&outer.nodes);

    // First, create GetParam nodes for ParamIds referenced where the GetParam node
    // is not part of the outlined set.
    let mut param_ids_with_node_in_set: HashSet<usize> = HashSet::new();
    for &idx in to_outline_set.iter() {
        if let NodePayload::GetParam(pid) = outer.nodes[idx].payload {
            param_ids_with_node_in_set.insert(pid.get_wrapped_id());
        }
    }
    // For each used ParamId that does not have a GetParam node copied from set,
    // synthesize a GetParam node in inner.
    for (pid_num, (nm, ty)) in used_param_ids.iter() {
        if !param_ids_with_node_in_set.contains(pid_num) {
            let inner_pid = *outer_paramid_to_inner
                .get(pid_num)
                .expect("inner ParamId mapping must exist");
            inner_nodes.push(Node {
                text_id: next_inner_text_id,
                name: Some(nm.clone()),
                ty: ty.clone(),
                payload: NodePayload::GetParam(inner_pid),
                pos: None,
            });
            let new_ref = NodeRef {
                index: inner_nodes.len() - 1,
            };
            // Find any external GetParam node references in outer and map them to this.
            for &ext_idx in inputs_ext.iter() {
                if let NodePayload::GetParam(ep) = outer.nodes[ext_idx].payload {
                    if ep.get_wrapped_id() == *pid_num {
                        ext_ref_to_inner_ref.insert(ext_idx, new_ref);
                    }
                }
            }
            next_inner_text_id += 1;
        }
    }

    // Now create GetParam nodes for each external non-param input, in the same
    // order they were appended to inner_params after the ParamIds.
    let start_nonparam_param_pos = inner_params
        .iter()
        .map(|p| p.id.get_wrapped_id())
        .max()
        .unwrap_or(0)
        - (ext_nonparam_inputs.len());
    // We need a map from (relative position among nonparam inputs) -> ParamId
    // Compute the ParamIds assigned to nonparam inputs in order.
    let mut nonparam_param_ids: Vec<ParamId> = Vec::new();
    let base = inner_params.len() - ext_nonparam_inputs.len();
    for i in 0..ext_nonparam_inputs.len() {
        nonparam_param_ids.push(inner_params[base + i].id);
    }
    for (i, ext_idx) in ext_nonparam_inputs.iter().enumerate() {
        let src_node = &outer.nodes[*ext_idx];
        inner_nodes.push(Node {
            text_id: next_inner_text_id,
            name: src_node.name.clone(),
            ty: src_node.ty.clone(),
            payload: NodePayload::GetParam(nonparam_param_ids[i]),
            pos: src_node.pos.clone(),
        });
        let new_ref = NodeRef {
            index: inner_nodes.len() - 1,
        };
        ext_ref_to_inner_ref.insert(*ext_idx, new_ref);
        next_inner_text_id += 1;
    }

    // Clone outlined nodes into inner, remapping operands:
    let topo_outer = get_topological(outer);
    let topo_filtered: Vec<NodeRef> = topo_outer
        .into_iter()
        .filter(|nr| to_outline_set.contains(&nr.index))
        .collect();
    let mut outlined_to_inner: HashMap<usize, NodeRef> = HashMap::new();
    for nr in topo_filtered.into_iter() {
        let old = &outer.nodes[nr.index];
        // Map operands: internal operands map to already-cloned nodes; external
        // operands map to synthesized GetParam nodes (must exist).
        let mapper = |r: NodeRef| -> NodeRef {
            if to_outline_set.contains(&r.index) {
                outlined_to_inner
                    .get(&r.index)
                    .copied()
                    .unwrap_or_else(|| panic!("missing mapping for internal operand {:?}", r))
            } else {
                ext_ref_to_inner_ref
                    .get(&r.index)
                    .copied()
                    .unwrap_or_else(|| panic!("missing mapping for external operand {:?}", r))
            }
        };
        let mut new_payload = remap_payload_with(&old.payload, mapper);
        // If cloning a GetParam, remap its ParamId to the inner-assigned id.
        if let NodePayload::GetParam(pid) = old.payload {
            let new_pid = *outer_paramid_to_inner
                .get(&pid.get_wrapped_id())
                .expect("inner param id must exist for cloned get_param");
            new_payload = NodePayload::GetParam(new_pid);
        }
        let new_node = Node {
            text_id: old.text_id,
            name: old.name.clone(),
            ty: old.ty.clone(),
            payload: new_payload,
            pos: old.pos.clone(),
        };
        inner_nodes.push(new_node);
        let new_ref = NodeRef {
            index: inner_nodes.len() - 1,
        };
        outlined_to_inner.insert(nr.index, new_ref);
    }

    // Determine inner return node and type from boundary outputs (in ascending
    // order).
    assert!(
        !outputs_boundary.is_empty(),
        "outlined region must produce at least one boundary output (or ret)"
    );
    let mut inner_ret_ref: Option<NodeRef> = None;
    let mut inner_ret_ty: Type;
    let mut ret_candidates: Vec<NodeRef> = outputs_boundary
        .iter()
        .map(|idx| outlined_to_inner.get(idx).copied().unwrap())
        .collect();
    // Stable order by inner node index
    ret_candidates.sort_by_key(|nr| nr.index);
    if ret_candidates.len() == 1 {
        inner_ret_ref = Some(ret_candidates[0]);
        inner_ret_ty = inner_nodes[ret_candidates[0].index].ty.clone();
    } else {
        let tuple_elems = ret_candidates.clone();
        let tuple_types: Vec<Type> = tuple_elems
            .iter()
            .map(|nr| inner_nodes[nr.index].ty.clone())
            .collect();
        inner_ret_ty = Type::Tuple(tuple_types.into_iter().map(|t| Box::new(t)).collect());
        // Create the tuple node
        let tuple_node = Node {
            text_id: next_inner_text_id,
            name: None,
            ty: inner_ret_ty.clone(),
            payload: NodePayload::Tuple(tuple_elems.clone()),
            pos: None,
        };
        inner_nodes.push(tuple_node);
        inner_ret_ref = Some(NodeRef {
            index: inner_nodes.len() - 1,
        });
        next_inner_text_id += 1;
    }

    // Topologically order inner nodes and remap indices accordingly
    let order_inner = get_topological_nodes(&inner_nodes);
    let mut old_to_new_inner: Vec<usize> = vec![0; inner_nodes.len()];
    for (new_idx, nr) in order_inner.iter().enumerate() {
        old_to_new_inner[nr.index] = new_idx;
    }
    let mut remapped_inner_nodes: Vec<Node> = Vec::with_capacity(inner_nodes.len());
    for nr in order_inner.iter() {
        let old = &inner_nodes[nr.index];
        let mapper = |r: NodeRef| -> NodeRef {
            NodeRef {
                index: old_to_new_inner[r.index],
            }
        };
        let new_payload = remap_payload_with(&old.payload, mapper);
        remapped_inner_nodes.push(Node {
            text_id: old.text_id,
            name: old.name.clone(),
            ty: old.ty.clone(),
            payload: new_payload,
            pos: old.pos.clone(),
        });
    }
    let remapped_inner_ret = inner_ret_ref.map(|nr| NodeRef {
        index: old_to_new_inner[nr.index],
    });

    let inner_fn = IrFn {
        name: new_inner_name.to_string(),
        params: inner_params.clone(),
        ret_ty: inner_ret_ty,
        nodes: remapped_inner_nodes,
        ret_node_ref: remapped_inner_ret,
    };

    // Build the new outer: clone and then splice in an invoke that replaces the
    // outlined boundary outputs.
    let mut outer_nodes = outer.nodes.clone();

    // Build invoke operands in the exact order of inner params.
    // Map from outer ParamId number -> NodeRef for the corresponding GetParam node.
    let mut paramid_to_node_ref: HashMap<usize, NodeRef> = HashMap::new();
    for (i, n) in outer.nodes.iter().enumerate() {
        if let NodePayload::GetParam(pid) = n.payload {
            paramid_to_node_ref.insert(pid.get_wrapped_id(), NodeRef { index: i });
        }
    }
    let mut invoke_operands: Vec<NodeRef> = Vec::with_capacity(inner_params.len());
    // First params originating from outer ParamIds, in used_param_ids order
    for (pid_num, _info) in used_param_ids.iter() {
        let nr = *paramid_to_node_ref
            .get(pid_num)
            .expect("outer must have a GetParam node for each param id");
        invoke_operands.push(nr);
    }
    // Then external non-param inputs, in ext_nonparam_inputs order
    for idx in ext_nonparam_inputs.iter() {
        invoke_operands.push(NodeRef { index: *idx });
    }

    // Create the invoke node in outer
    let mut next_outer_text_id: usize = next_text_id(&outer_nodes);
    let invoke_node_index = {
        let ret_ty = inner_fn.ret_ty.clone();
        let new_node = Node {
            text_id: next_outer_text_id,
            name: None,
            ty: ret_ty,
            payload: NodePayload::Invoke {
                to_apply: new_inner_name.to_string(),
                operands: invoke_operands,
            },
            pos: None,
        };
        outer_nodes.push(new_node);
        next_outer_text_id += 1;
        outer_nodes.len() - 1
    };
    let invoke_ref = NodeRef {
        index: invoke_node_index,
    };

    // For multiple outputs, synthesize tuple_index nodes and build a mapping from
    // old outlined outputs to their replacements. For a single output, the
    // invoke result replaces directly.
    let mut replacement_map: HashMap<usize, NodeRef> = HashMap::new();
    let mut ordered_outputs: Vec<usize> = outputs_boundary.iter().copied().collect();
    ordered_outputs.sort_unstable();
    if ordered_outputs.len() == 1 {
        replacement_map.insert(ordered_outputs[0], invoke_ref);
    } else {
        for (i, old_idx) in ordered_outputs.iter().enumerate() {
            let ty = outer.nodes[*old_idx].ty.clone();
            let tidx = Node {
                text_id: next_outer_text_id,
                name: None,
                ty,
                payload: NodePayload::TupleIndex {
                    tuple: invoke_ref,
                    index: i,
                },
                pos: None,
            };
            outer_nodes.push(tidx);
            let new_ref = NodeRef {
                index: outer_nodes.len() - 1,
            };
            replacement_map.insert(*old_idx, new_ref);
            next_outer_text_id += 1;
        }
    }

    // Rewrite all nodes outside the outlined set to refer to replacements where
    // needed.
    for (i, node) in outer_nodes.iter_mut().enumerate() {
        if to_outline_set.contains(&i) {
            continue;
        }
        let mapper = |r: NodeRef| -> NodeRef {
            if let Some(&nr) = replacement_map.get(&r.index) {
                nr
            } else {
                r
            }
        };
        let new_payload = remap_payload_with(&node.payload, mapper);
        node.payload = new_payload;
    }

    // Clobber outlined nodes' payloads with Nil (except GetParam nodes, which may
    // be used as invoke operands)
    for &idx in to_outline_set.iter() {
        if !matches!(outer_nodes[idx].payload, NodePayload::GetParam(_)) {
            outer_nodes[idx].payload = NodePayload::Nil;
        }
    }

    // Update outer return if it pointed to an outlined node
    let mut outer_ret_ref = outer.ret_node_ref.map(|nr| {
        if let Some(&rep) = replacement_map.get(&nr.index) {
            rep
        } else {
            nr
        }
    });

    // Topologically reorder outer nodes and remap operands accordingly
    let order_outer = get_topological_nodes(&outer_nodes);
    let mut old_to_new_outer: Vec<usize> = vec![0; outer_nodes.len()];
    for (new_idx, nr) in order_outer.iter().enumerate() {
        old_to_new_outer[nr.index] = new_idx;
    }
    let mut remapped_outer_nodes: Vec<Node> = Vec::with_capacity(outer_nodes.len());
    for nr in order_outer.iter() {
        let old = &outer_nodes[nr.index];
        let mapper = |r: NodeRef| -> NodeRef {
            NodeRef {
                index: old_to_new_outer[r.index],
            }
        };
        let new_payload = remap_payload_with(&old.payload, mapper);
        remapped_outer_nodes.push(Node {
            text_id: old.text_id,
            name: old.name.clone(),
            ty: old.ty.clone(),
            payload: new_payload,
            pos: old.pos.clone(),
        });
    }
    outer_ret_ref = outer_ret_ref.map(|nr| NodeRef {
        index: old_to_new_outer[nr.index],
    });

    let new_outer = IrFn {
        name: new_outer_name.to_string(),
        params: outer.params.clone(),
        ret_ty: outer.ret_ty.clone(),
        nodes: remapped_outer_nodes,
        ret_node_ref: outer_ret_ref,
    };

    // Register both functions in the package for caller convenience.
    package
        .members
        .push(PackageMember::Function(new_outer.clone()));
    package
        .members
        .push(PackageMember::Function(inner_fn.clone()));

    OutlineResult {
        outer: new_outer,
        inner: inner_fn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xls_ir::ir::{NodePayload, PackageMember};
    use crate::xls_ir::ir_parser::Parser;

    fn parse_single_fn(ir: &str) -> (Package, IrFn) {
        let pkg_text = format!("package test\n\n{}\n", ir);
        let mut p = Parser::new(&pkg_text);
        let pkg = p.parse_and_validate_package().unwrap();
        let f = pkg
            .members
            .iter()
            .filter_map(|m| match m {
                PackageMember::Function(f) => Some(f.clone()),
                _ => None,
            })
            .next()
            .unwrap();
        (pkg, f)
    }

    #[test]
    fn outline_basic_region_with_params() {
        let ir = r#"fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  t: bits[8] = add(a, b, id=3)
  ret id.4: bits[8] = identity(t, id=4)
}"#;
        let (mut pkg, f) = parse_single_fn(ir);
        let before_pkg = pkg.to_string();
        // Select nodes: indices 2 (add named t) and 3 (identity)
        // Build selection by payload ids
        let mut to_sel: HashSet<NodeRef> = HashSet::new();
        for (i, n) in f.nodes.iter().enumerate() {
            if matches!(n.payload, NodePayload::Binop(_, _, _))
                || matches!(n.payload, NodePayload::Unop(_, _))
            {
                to_sel.insert(NodeRef { index: i });
            }
        }
        let res = outline(&f, &to_sel, "f_out", "f_inner", &mut pkg);
        let after_pkg = pkg.to_string();
        println!("BEFORE:\n{}\n\nAFTER:\n{}", before_pkg, after_pkg);
        // Outer should invoke inner and return its result
        let outer_s = res.outer.to_string();
        // Expect exact package print before
        let expected_before = r#"package test

fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  t: bits[8] = add(a, b, id=3)
  ret identity.4: bits[8] = identity(t, id=4)
}
"#;
        assert_eq!(before_pkg, expected_before);

        // Expect exact package print after (original, new outer, new inner)
        let expected_after = r#"package test

fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  t: bits[8] = add(a, b, id=3)
  ret identity.4: bits[8] = identity(t, id=4)
}

fn f_out(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret invoke.5: bits[8] = invoke(a, b, to_apply=f_inner, id=5)
}

fn f_inner(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  t: bits[8] = add(a, b, id=3)
  ret identity.4: bits[8] = identity(t, id=4)
}
"#;
        assert_eq!(after_pkg, expected_after);
        // Also confirm inner structure and signature
        assert_eq!(res.inner.params.len(), 2);
        assert_eq!(res.inner.name, "f_inner");
        assert!(matches!(res.inner.ret_ty, Type::Bits(8)));
    }

    #[test]
    fn outline_external_nonparam_inputs() {
        let ir = r#"fn g(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  t: bits[8] = add(a, b, id=3)
  u: bits[8] = not(a, id=4)
  ret m: bits[8] = umul(t, u, id=5)
}"#;
        let (mut pkg, f) = parse_single_fn(ir);
        let before_pkg = pkg.to_string();
        // Outline only the mul node (index 4 or 5 depending on params); detect by
        // operator
        let mut to_sel: HashSet<NodeRef> = HashSet::new();
        for (i, n) in f.nodes.iter().enumerate() {
            if let NodePayload::Binop(op, _, _) = n.payload {
                if crate::xls_ir::ir::binop_to_operator(op) == "umul" {
                    to_sel.insert(NodeRef { index: i });
                }
            }
        }
        let res = outline(&f, &to_sel, "g_out", "g_inner", &mut pkg);
        let after_pkg = pkg.to_string();
        println!("BEFORE:\n{}\n\nAFTER:\n{}", before_pkg, after_pkg);
        // Expect exact package strings for determinism
        let expected_before = r#"package test

fn g(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  t: bits[8] = add(a, b, id=3)
  u: bits[8] = not(a, id=4)
  ret m: bits[8] = umul(t, u, id=5)
}
"#;
        assert_eq!(before_pkg, expected_before);

        let expected_after = r#"package test

fn g(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  t: bits[8] = add(a, b, id=3)
  u: bits[8] = not(a, id=4)
  ret m: bits[8] = umul(t, u, id=5)
}

fn g_out(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  t: bits[8] = add(a, b, id=3)
  u: bits[8] = not(a, id=4)
  ret invoke.6: bits[8] = invoke(t, u, to_apply=g_inner, id=6)
}

fn g_inner(t: bits[8] id=1, u: bits[8] id=2) -> bits[8] {
  ret m: bits[8] = umul(t, u, id=5)
}
"#;
        assert_eq!(after_pkg, expected_after);
        // Inner should have two params (corresponding to t and u)
        assert_eq!(res.inner.params.len(), 2);
    }
}
