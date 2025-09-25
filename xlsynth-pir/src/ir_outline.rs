// SPDX-License-Identifier: Apache-2.0

//! Outlining transformation for XLS IR functions.
//!
//! Given an `IrFn` and a set of nodes to outline, builds a new inner function
//! consisting of that subgraph and rewrites the outer function to invoke it.
//! The function signature of the inner is discovered from boundary inputs and
//! outputs of the selected subgraph.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::ir::{
    Fn as IrFn, Node, NodePayload, NodeRef, Package, PackageMember, Param, ParamId, Type,
};
use crate::ir_utils::{get_topological, get_topological_nodes, operands, remap_payload_with};

#[derive(Debug, Clone)]
pub struct OutlineResult {
    pub outer: IrFn,
    pub inner: IrFn,
}

#[derive(Debug, Clone)]
pub struct OutlineOrdering {
    pub params: Vec<OutlineParamSpec>,
    pub returns: Vec<OutlineReturnSpec>,
}

#[derive(Debug, Clone)]
pub struct OutlineParamSpec {
    pub node: NodeRef,
    pub rename: Option<String>,
}

#[derive(Debug, Clone)]
pub struct OutlineReturnSpec {
    pub node: NodeRef,
    pub rename: Option<String>,
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
fn derive_arg_name(n: &Node, fallback_index: usize, used: &HashSet<String>) -> String {
    let base = n
        .name
        .clone()
        .unwrap_or_else(|| format!("arg_{}", fallback_index));
    if !used.contains(&base) {
        return base;
    }
    // Deterministically uniquify by appending __<k>
    let mut k: usize = 1;
    loop {
        let candidate = format!("{}__{}", base, k);
        if !used.contains(&candidate) {
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
fn compute_inputs_and_boundary(
    outer: &IrFn,
    to_outline_set: &HashSet<usize>,
) -> (BTreeSet<usize>, BTreeSet<usize>) {
    let mut inputs_ext: BTreeSet<usize> = BTreeSet::new();
    let mut outputs_boundary: BTreeSet<usize> = BTreeSet::new();

    for &idx in to_outline_set.iter() {
        let n = &outer.nodes[idx];
        for dep in operands(&n.payload).into_iter() {
            if !to_outline_set.contains(&dep.index) {
                inputs_ext.insert(dep.index);
            }
        }
    }

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
    (inputs_ext, outputs_boundary)
}

pub fn compute_default_ordering(outer: &IrFn, to_outline: &HashSet<NodeRef>) -> OutlineOrdering {
    for nr in to_outline.iter() {
        assert!(
            nr.index < outer.nodes.len(),
            "NodeRef out of bounds: {:?}",
            nr
        );
    }
    let to_outline_set: HashSet<usize> = to_outline.iter().map(|r| r.index).collect();
    let (inputs_ext, outputs_boundary) = compute_inputs_and_boundary(outer, &to_outline_set);

    let mut used_param_ids: BTreeMap<usize, (String, Type)> = BTreeMap::new();
    let mut outer_param_info: HashMap<usize, (String, Type)> = HashMap::new();
    for p in outer.params.iter() {
        outer_param_info.insert(p.id.get_wrapped_id(), (p.name.clone(), p.ty.clone()));
    }
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

    let mut params: Vec<OutlineParamSpec> = Vec::new();
    // Prefer the GetParam node inside the selection if present for each ParamId,
    // otherwise reference the external GetParam node driving the region.
    let mut param_id_to_internal_node: BTreeMap<usize, usize> = BTreeMap::new();
    for &idx in to_outline_set.iter() {
        if let NodePayload::GetParam(pid) = outer.nodes[idx].payload {
            param_id_to_internal_node.insert(pid.get_wrapped_id(), idx);
        }
    }
    for (pid_num, _info) in used_param_ids.iter() {
        if let Some(&internal_idx) = param_id_to_internal_node.get(pid_num) {
            params.push(OutlineParamSpec {
                node: NodeRef {
                    index: internal_idx,
                },
                rename: None,
            });
        } else {
            // Find an external GetParam node with this id among inputs_ext
            let mut chosen: Option<usize> = None;
            for &ext_idx in inputs_ext.iter() {
                if let NodePayload::GetParam(pid) = outer.nodes[ext_idx].payload {
                    if pid.get_wrapped_id() == *pid_num {
                        chosen = Some(ext_idx);
                        break;
                    }
                }
            }
            let idx = chosen.expect("external GetParam node must exist for used ParamId");
            params.push(OutlineParamSpec {
                node: NodeRef { index: idx },
                rename: None,
            });
        }
    }
    let mut ext_nonparam_inputs: Vec<usize> = inputs_ext
        .iter()
        .copied()
        .filter(|idx| !matches!(outer.nodes[*idx].payload, NodePayload::GetParam(_)))
        .collect();
    ext_nonparam_inputs.sort_unstable();
    for idx in ext_nonparam_inputs.into_iter() {
        params.push(OutlineParamSpec {
            node: NodeRef { index: idx },
            rename: None,
        });
    }

    let mut returns: Vec<OutlineReturnSpec> = outputs_boundary
        .iter()
        .copied()
        .collect::<Vec<usize>>()
        .into_iter()
        .map(|idx| OutlineReturnSpec {
            node: NodeRef { index: idx },
            rename: None,
        })
        .collect();
    returns.sort_by_key(|r| r.node.index);

    OutlineOrdering { params, returns }
}

pub fn outline_with_ordering(
    outer: &IrFn,
    to_outline: &HashSet<NodeRef>,
    new_outer_name: &str,
    new_inner_name: &str,
    ordering: &OutlineOrdering,
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
    let (inputs_ext, outputs_boundary) = compute_inputs_and_boundary(outer, &to_outline_set);

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

    // Validate ordering coverage and build inner params according to ordering
    // Compute required sets
    let mut required_param_ids: BTreeSet<usize> = used_param_ids.keys().copied().collect();
    let mut required_nonparam_inputs: BTreeSet<usize> = inputs_ext
        .iter()
        .copied()
        .filter(|idx| !matches!(outer.nodes[*idx].payload, NodePayload::GetParam(_)))
        .collect();

    // Determine which outer nodes are used as passthrough returns (outside
    // selection)
    let mut passthrough_sources: HashSet<usize> = HashSet::new();
    for r in ordering.returns.iter() {
        if !to_outline_set.contains(&r.node.index) {
            passthrough_sources.insert(r.node.index);
        }
    }

    // Track duplicates and coverage
    let mut seen_param_ids: HashSet<usize> = HashSet::new();
    let mut seen_ext_nodes: HashSet<usize> = HashSet::new();

    // Build inner params in specified order
    let mut inner_params: Vec<Param> = Vec::new();
    let mut outer_paramid_to_inner: HashMap<usize, ParamId> = HashMap::new();
    let mut param_index_to_inner_param_id: Vec<ParamId> = Vec::new();
    let mut param_index_to_source_outer_node: Vec<Option<usize>> = Vec::new();
    let mut used_names: HashSet<String> = HashSet::new();
    let mut next_param_pos: usize = 1;
    for (idx, ps) in ordering.params.iter().enumerate() {
        let node = ps.node;
        let rename = &ps.rename;
        let src_node = &outer.nodes[node.index];
        match src_node.payload {
            NodePayload::GetParam(pid) => {
                let id_number = pid.get_wrapped_id();
                if !seen_param_ids.insert(id_number) {
                    panic!("duplicate ParamId in params ordering: {}", id_number);
                }
                let (nm, ty) = outer_param_info
                    .get(&id_number)
                    .cloned()
                    .expect("outer params must include referenced ParamId");
                let name = rename.clone().unwrap_or(nm.clone());
                if used_names.contains(&name) {
                    panic!("duplicate param name: {}", name);
                }
                used_names.insert(name.clone());
                let inner_id = ParamId::new(next_param_pos);
                next_param_pos += 1;
                inner_params.push(Param {
                    name,
                    ty: ty.clone(),
                    id: inner_id,
                });
                outer_paramid_to_inner.insert(id_number, inner_id);
                param_index_to_inner_param_id.push(inner_id);
                param_index_to_source_outer_node.push(Some(node.index));
                if required_param_ids.contains(&id_number) {
                    required_param_ids.remove(&id_number);
                }
            }
            _ => {
                if !seen_ext_nodes.insert(node.index) {
                    panic!("duplicate ExternalNode in params ordering: {}", node.index);
                }
                if to_outline_set.contains(&node.index) {
                    panic!("External input must be outside selection: {}", node.index);
                }
                if !inputs_ext.contains(&node.index) && !passthrough_sources.contains(&node.index) {
                    panic!(
                        "External input not required and not used as passthrough: {}",
                        node.index
                    );
                }
                let fallback_name = derive_arg_name(src_node, idx, &used_names);
                let name = rename.clone().unwrap_or(fallback_name);
                if used_names.contains(&name) {
                    panic!("duplicate param name: {}", name);
                }
                used_names.insert(name.clone());
                let inner_id = ParamId::new(next_param_pos);
                next_param_pos += 1;
                inner_params.push(Param {
                    name,
                    ty: src_node.ty.clone(),
                    id: inner_id,
                });
                param_index_to_inner_param_id.push(inner_id);
                param_index_to_source_outer_node.push(Some(node.index));
                if required_nonparam_inputs.contains(&node.index) {
                    required_nonparam_inputs.remove(&node.index);
                }
            }
        }
    }
    // All required items must be covered
    assert!(
        required_param_ids.is_empty(),
        "missing required OuterParamId(s) in params ordering: {:?}",
        required_param_ids
    );
    assert!(
        required_nonparam_inputs.is_empty(),
        "missing required ExternalNode(s) in params ordering: {:?}",
        required_nonparam_inputs
    );

    // Build inner nodes: create GetParam nodes for any ParamIds that do not already
    // exist inside the selected set (i.e. when the GetParam node itself is
    // external), and for all external non-param inputs. Track a map from outer
    // NodeRef -> inner NodeRef for remapping payload operands.
    let mut inner_nodes: Vec<Node> = Vec::new();
    let mut ext_ref_to_inner_ref: HashMap<usize, NodeRef> = HashMap::new();
    let mut next_inner_text_id: usize = next_text_id(&outer.nodes);

    // Create synthesized GetParam nodes in inner for any external GetParam inputs.
    for (pidx, ps) in ordering.params.iter().enumerate() {
        let node = ps.node;
        if !to_outline_set.contains(&node.index) {
            if let NodePayload::GetParam(_) = outer.nodes[node.index].payload {
                let inner_pid = param_index_to_inner_param_id[pidx];
                let src_node = &outer.nodes[node.index];
                inner_nodes.push(Node {
                    text_id: next_inner_text_id,
                    name: Some(inner_params[pidx].name.clone()),
                    ty: src_node.ty.clone(),
                    payload: NodePayload::GetParam(inner_pid),
                    pos: src_node.pos.clone(),
                });
                let new_ref = NodeRef {
                    index: inner_nodes.len() - 1,
                };
                ext_ref_to_inner_ref.insert(node.index, new_ref);
                next_inner_text_id += 1;
            }
        }
    }

    // Now create GetParam nodes for each external non-param input, in the same
    // order they were appended to inner_params after the ParamIds.
    // Compute the ParamIds (and names) assigned to nonparam inputs in order.
    for (pidx, ps) in ordering.params.iter().enumerate() {
        let node = ps.node;
        if !to_outline_set.contains(&node.index) {
            if !matches!(outer.nodes[node.index].payload, NodePayload::GetParam(_)) {
                let src_node = &outer.nodes[node.index];
                inner_nodes.push(Node {
                    text_id: next_inner_text_id,
                    name: Some(inner_params[pidx].name.clone()),
                    ty: src_node.ty.clone(),
                    payload: NodePayload::GetParam(param_index_to_inner_param_id[pidx]),
                    pos: src_node.pos.clone(),
                });
                let new_ref = NodeRef {
                    index: inner_nodes.len() - 1,
                };
                ext_ref_to_inner_ref.insert(node.index, new_ref);
                next_inner_text_id += 1;
            }
        }
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

    // No separate param_index_to_inner_ref needed with NodeRef-based returns.

    // Determine inner return node and type from boundary outputs (in ascending
    // order).
    assert!(
        !ordering.returns.is_empty(),
        "must specify at least one return in ordering"
    );
    // Validate that returns include all boundary outputs exactly once
    let mut required_boundary: BTreeSet<usize> = outputs_boundary.clone();
    let mut seen_boundary: HashSet<usize> = HashSet::new();
    for r in ordering.returns.iter() {
        let node = r.node;
        if to_outline_set.contains(&node.index) {
            if !outputs_boundary.contains(&node.index) {
                panic!("Outlined output must be a boundary output: {}", node.index);
            }
            if !seen_boundary.insert(node.index) {
                panic!("duplicate outlined output in returns: {}", node.index);
            }
            required_boundary.remove(&node.index);
        }
    }
    assert!(
        required_boundary.is_empty(),
        "missing boundary outputs in returns ordering: {:?}",
        required_boundary
    );

    // Build return element NodeRefs in the order specified
    let mut ret_elem_refs: Vec<NodeRef> = Vec::new();
    let mut ret_elem_tys: Vec<Type> = Vec::new();
    for r in ordering.returns.iter() {
        let node = r.node;
        if to_outline_set.contains(&node.index) {
            let ir = *outlined_to_inner
                .get(&node.index)
                .expect("inner ref for outlined output");
            ret_elem_tys.push(inner_nodes[ir.index].ty.clone());
            ret_elem_refs.push(ir);
        } else {
            let ir = *ext_ref_to_inner_ref
                .get(&node.index)
                .expect("inner ref for passthrough source");
            ret_elem_tys.push(inner_nodes[ir.index].ty.clone());
            ret_elem_refs.push(ir);
        }
    }

    #[allow(unused_assignments)]
    let (inner_ret_ref, inner_ret_ty) = if ret_elem_refs.len() == 1 {
        (Some(ret_elem_refs[0]), ret_elem_tys.remove(0))
    } else {
        let tuple_ty = Type::Tuple(ret_elem_tys.into_iter().map(|t| Box::new(t)).collect());
        let tuple_node = Node {
            text_id: next_inner_text_id,
            name: None,
            ty: tuple_ty.clone(),
            payload: NodePayload::Tuple(ret_elem_refs.clone()),
            pos: None,
        };
        inner_nodes.push(tuple_node);
        let rref = Some(NodeRef {
            index: inner_nodes.len() - 1,
        });
        next_inner_text_id += 1;
        (rref, tuple_ty)
    };

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
        outer_attrs: Vec::new(),
        inner_attrs: Vec::new(),
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
    for ps in ordering.params.iter() {
        invoke_operands.push(ps.node);
    }

    // Create the invoke node in outer
    let mut next_outer_text_id: usize = next_text_id(&outer_nodes);
    let invoke_operands_for_node = invoke_operands.clone();
    let invoke_node_index = {
        let ret_ty = inner_fn.ret_ty.clone();
        let new_node = Node {
            text_id: next_outer_text_id,
            name: None,
            ty: ret_ty,
            payload: NodePayload::Invoke {
                to_apply: new_inner_name.to_string(),
                operands: invoke_operands_for_node,
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

    // Materialize return projections in the specified order
    let multi_return = ordering.returns.len() > 1;
    let mut return_value_refs: Vec<NodeRef> = Vec::with_capacity(ordering.returns.len());
    if !multi_return {
        return_value_refs.push(invoke_ref);
    } else {
        for (i, r) in ordering.returns.iter().enumerate() {
            let ty = outer.nodes[r.node.index].ty.clone();
            let name = r.rename.clone();
            let tidx = Node {
                text_id: next_outer_text_id,
                name,
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
            return_value_refs.push(new_ref);
            next_outer_text_id += 1;
        }
    }

    // Build replacement map for outlined outputs and passthroughs
    let mut replacement_map: HashMap<usize, NodeRef> = HashMap::new();
    for (i, r) in ordering.returns.iter().enumerate() {
        let rep = if multi_return {
            return_value_refs[i]
        } else {
            invoke_ref
        };
        replacement_map.insert(r.node.index, rep);
    }

    // Compute a protected operand producer cone for the invoke's operands to avoid
    // creating cycles: skip rewriting inside this cone.
    let mut protected: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut stack: Vec<usize> = Vec::new();
    // The operands vector we built earlier refers to indices in `outer` (and thus
    // the cloned `outer_nodes`). Walk reverse-transitively via operands.
    for op in invoke_operands.iter() {
        if protected.insert(op.index) {
            stack.push(op.index);
        }
    }
    while let Some(idx) = stack.pop() {
        for d in operands(&outer_nodes[idx].payload).into_iter() {
            if protected.insert(d.index) {
                stack.push(d.index);
            }
        }
    }

    // Rewrite all nodes outside the outlined set to refer to replacements where
    // needed, but do not rewrite inside the protected cone or the invoke node
    // itself.
    for (i, node) in outer_nodes.iter_mut().enumerate() {
        if to_outline_set.contains(&i) || i == invoke_node_index || protected.contains(&i) {
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
        outer_attrs: Vec::new(),
        inner_attrs: Vec::new(),
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

pub fn outline(
    outer: &IrFn,
    to_outline: &HashSet<NodeRef>,
    new_outer_name: &str,
    new_inner_name: &str,
    package: &mut Package,
) -> OutlineResult {
    let ordering = compute_default_ordering(outer, to_outline);
    outline_with_ordering(
        outer,
        to_outline,
        new_outer_name,
        new_inner_name,
        &ordering,
        package,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{self, NodePayload, PackageMember};
    use crate::ir_parser::Parser;
    use crate::prove_equiv_via_toolchain;
    use crate::prove_equiv_via_toolchain::ToolchainEquivResult;

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

    fn assert_equiv_pkg(orig: &IrFn, outlined_outer: &IrFn, outlined_inner: Option<&IrFn>) {
        // If the outlined outer contains invokes, the external tool requires a
        // multi-function package with globally unique node ids, which our
        // pretty-printer does not guarantee across functions. Skip equivalence in
        // that case for unit tests; fuzz target covers semantic checks separately.
        if outlined_outer
            .nodes
            .iter()
            .any(|n| matches!(n.payload, NodePayload::Invoke { .. }))
        {
            return;
        }
        let lhs_pkg = format!("package lhs\n\ntop {}\n", orig.to_string());
        let rhs_pkg = match outlined_inner {
            Some(inner) => format!(
                "package rhs\n\n{}\n\ntop {}\n",
                inner.to_string(),
                outlined_outer.to_string()
            ),
            None => format!("package rhs\n\ntop {}\n", outlined_outer.to_string()),
        };
        let tool_dir = std::env::var("XLSYNTH_TOOLS")
            .expect("XLSYNTH_TOOLS must be set for equivalence tests");
        let res = prove_equiv_via_toolchain::prove_ir_pkg_equiv_with_tool_dir(
            &lhs_pkg, &rhs_pkg, None, tool_dir,
        );
        assert!(
            matches!(res, ToolchainEquivResult::Proved),
            "Outlining equivalence failed: {:?}",
            res
        );
    }

    #[test]
    fn outline_param_inside_selection() {
        let ir = r#"fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  t: bits[8] = add(a, b, id=3)
  ret identity.4: bits[8] = identity(t, id=4)
}"#;
        let (mut pkg, f) = parse_single_fn(ir);
        // Select the 'a' parameter node and the add node to ensure a param node
        // inside the outlined region is handled correctly.
        let mut to_sel: HashSet<NodeRef> = HashSet::new();
        let mut add_idx: Option<usize> = None;
        let mut a_param_idx: Option<usize> = None;
        for (i, n) in f.nodes.iter().enumerate() {
            match n.payload {
                NodePayload::GetParam(pid) => {
                    if pid.get_wrapped_id() == 1 {
                        a_param_idx = Some(i);
                    }
                }
                NodePayload::Binop(ir::Binop::Add, _, _) => {
                    add_idx = Some(i);
                }
                _ => {}
            }
        }
        to_sel.insert(NodeRef {
            index: a_param_idx.expect("found param a"),
        });
        to_sel.insert(NodeRef {
            index: add_idx.expect("found add"),
        });

        let res = outline(&f, &to_sel, "f_out", "f_inner", &mut pkg);
        // Inner should still have two params named 'a' and 'b'.
        assert_eq!(res.inner.params.len(), 2);
        assert_eq!(res.inner.params[0].name, "a");
        assert_eq!(res.inner.params[1].name, "b");
        // Equivalence: original vs outlined outer
        assert_equiv_pkg(&f, &res.outer, Some(&res.inner));
    }

    #[test]
    fn outline_basic_region_with_params() {
        let ir = r#"fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  t: bits[8] = add(a, b, id=3)
  ret identity.4: bits[8] = identity(t, id=4)
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
        println!("AFTER:\n{}", after_pkg);
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
        // Equivalence: original vs outlined outer
        assert_equiv_pkg(&f, &res.outer, Some(&res.inner));
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
                if ir::binop_to_operator(op) == "umul" {
                    to_sel.insert(NodeRef { index: i });
                }
            }
        }
        let res = outline(&f, &to_sel, "g_out", "g_inner", &mut pkg);
        let after_pkg = pkg.to_string();
        println!("AFTER:\n{}", after_pkg);
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
        // Equivalence: original vs outlined outer
        assert_equiv_pkg(&f, &res.outer, Some(&res.inner));
    }

    #[test]
    fn outline_external_nonparam_inputs_unnamed_sources() {
        // External inputs to the outlined region are unnamed nodes (e.g., add.3,
        // not.4). This exercises GetParam synthesis and naming for non-param
        // external inputs.
        let ir = r#"fn g2(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  add.3: bits[8] = add(a, b, id=3)
  not.4: bits[8] = not(a, id=4)
  ret umul.5: bits[8] = umul(add.3, not.4, id=5)
}"#;
        let (mut pkg, f) = parse_single_fn(ir);

        // Outline only the mul node (unnamed sources feeding it are external
        // non-params)
        let mut to_sel: HashSet<NodeRef> = HashSet::new();
        for (i, n) in f.nodes.iter().enumerate() {
            if let NodePayload::Binop(op, _, _) = n.payload {
                if ir::binop_to_operator(op) == "umul" {
                    to_sel.insert(NodeRef { index: i });
                }
            }
        }
        // Expect exact package strings for determinism; inner params must be named
        let expected_before = r#"package test

fn g2(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  add.3: bits[8] = add(a, b, id=3)
  not.4: bits[8] = not(a, id=4)
  ret umul.5: bits[8] = umul(add.3, not.4, id=5)
}
"#;
        let before_pkg = pkg.to_string();
        assert_eq!(before_pkg, expected_before);

        let res = outline(&f, &to_sel, "g2_out", "g2_inner", &mut pkg);

        let expected_after = r#"package test

fn g2(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  add.3: bits[8] = add(a, b, id=3)
  not.4: bits[8] = not(a, id=4)
  ret umul.5: bits[8] = umul(add.3, not.4, id=5)
}

fn g2_out(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  add.3: bits[8] = add(a, b, id=3)
  not.4: bits[8] = not(a, id=4)
  ret invoke.6: bits[8] = invoke(add.3, not.4, to_apply=g2_inner, id=6)
}

fn g2_inner(arg_0: bits[8] id=1, arg_1: bits[8] id=2) -> bits[8] {
  ret umul.5: bits[8] = umul(arg_0, arg_1, id=5)
}
"#;
        // pkg now includes original f, new outer, and new inner
        let after_pkg = pkg.to_string();
        assert_eq!(after_pkg, expected_after);

        // Equivalence: original vs outlined outer (skips if invoke is present)
        assert_equiv_pkg(&f, &res.outer, Some(&res.inner));
    }
    #[test]
    fn outline_region_with_two_boundary_outputs_tuple_return() {
        let ir = r#"fn h(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  t1: bits[8] = add(a, b, id=3)
  t2: bits[8] = sub(a, b, id=4)
  ret xor.5: bits[8] = xor(t1, t2, id=5)
}"#;
        let (mut pkg, f) = parse_single_fn(ir);
        let before_pkg = pkg.to_string();
        // Select the two producers t1(add) and t2(sub) to outline together.
        let mut to_sel: HashSet<NodeRef> = HashSet::new();
        for (i, n) in f.nodes.iter().enumerate() {
            match n.payload {
                NodePayload::Binop(_, _, _) => {
                    // add/sub
                    if i == 1 || i == 2 || i == 3 || i == 4 { /* keep lint calm */ }
                    to_sel.insert(NodeRef { index: i });
                }
                _ => {}
            }
        }
        // Ensure we only captured the add/sub (ids 3 and 4). Filter to exclude xor.
        to_sel.retain(|nr| match f.nodes[nr.index].payload {
            NodePayload::Binop(ir::Binop::Add, _, _) | NodePayload::Binop(ir::Binop::Sub, _, _) => {
                true
            }
            _ => false,
        });

        let res = outline(&f, &to_sel, "h_out", "h_inner", &mut pkg);
        let after_pkg = pkg.to_string();
        println!("AFTER:\n{}", after_pkg);

        let expected_before = r#"package test

fn h(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  t1: bits[8] = add(a, b, id=3)
  t2: bits[8] = sub(a, b, id=4)
  ret xor.5: bits[8] = xor(t1, t2, id=5)
}
"#;
        assert_eq!(before_pkg, expected_before);

        let expected_after = r#"package test

fn h(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  t1: bits[8] = add(a, b, id=3)
  t2: bits[8] = sub(a, b, id=4)
  ret xor.5: bits[8] = xor(t1, t2, id=5)
}

fn h_out(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  invoke.6: (bits[8], bits[8]) = invoke(a, b, to_apply=h_inner, id=6)
  tuple_index.7: bits[8] = tuple_index(invoke.6, index=0, id=7)
  tuple_index.8: bits[8] = tuple_index(invoke.6, index=1, id=8)
  ret xor.5: bits[8] = xor(tuple_index.7, tuple_index.8, id=5)
}

fn h_inner(a: bits[8] id=1, b: bits[8] id=2) -> (bits[8], bits[8]) {
  t1: bits[8] = add(a, b, id=3)
  t2: bits[8] = sub(a, b, id=4)
  ret tuple.8: (bits[8], bits[8]) = tuple(t1, t2, id=8)
}
"#;
        assert_eq!(after_pkg, expected_after);
        // Equivalence: original vs outlined outer
        assert_equiv_pkg(&f, &res.outer, Some(&res.inner));
    }

    #[test]
    fn outline_multi_in_multi_out_with_postprocess() {
        // Pre-op "px" makes the outlined region interior: it appears before the invoke
        // and feeds the outlined add/umul, while xor/and/or remain after the
        // invoke.
        let ir = r#"fn k(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  px: bits[8] = xor(a, c, id=4)
  add.5: bits[8] = add(px, b, id=5)
  umul.6: bits[8] = umul(px, b, id=6)
  xor.7: bits[8] = xor(add.5, c, id=7)
  and.8: bits[8] = and(umul.6, c, id=8)
  ret or.9: bits[8] = or(xor.7, and.8, id=9)
}"#;
        let (mut pkg, f) = parse_single_fn(ir);
        let before_pkg = pkg.to_string();
        // Outline the add and umul producers together
        let mut to_sel: HashSet<NodeRef> = HashSet::new();
        for (i, n) in f.nodes.iter().enumerate() {
            if matches!(n.payload, NodePayload::Binop(ir::Binop::Add, _, _))
                || matches!(n.payload, NodePayload::Binop(ir::Binop::Umul, _, _))
            {
                to_sel.insert(NodeRef { index: i });
            }
        }
        let res = outline(&f, &to_sel, "k_out", "k_inner", &mut pkg);

        let expected_before_fn = r#"fn k(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  px: bits[8] = xor(a, c, id=4)
  add.5: bits[8] = add(px, b, id=5)
  umul.6: bits[8] = umul(px, b, id=6)
  xor.7: bits[8] = xor(add.5, c, id=7)
  and.8: bits[8] = and(umul.6, c, id=8)
  ret or.9: bits[8] = or(xor.7, and.8, id=9)
}"#;
        assert_eq!(f.to_string(), expected_before_fn);

        let expected_outer = r#"fn k_out(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  px: bits[8] = xor(a, c, id=4)
  invoke.10: (bits[8], bits[8]) = invoke(b, px, to_apply=k_inner, id=10)
  tuple_index.11: bits[8] = tuple_index(invoke.10, index=0, id=11)
  xor.7: bits[8] = xor(tuple_index.11, c, id=7)
  tuple_index.12: bits[8] = tuple_index(invoke.10, index=1, id=12)
  and.8: bits[8] = and(tuple_index.12, c, id=8)
  ret or.9: bits[8] = or(xor.7, and.8, id=9)
}"#;
        assert_eq!(res.outer.to_string(), expected_outer);

        // Equivalence: original vs outlined outer
        assert_equiv_pkg(&f, &res.outer, Some(&res.inner));
    }
}
