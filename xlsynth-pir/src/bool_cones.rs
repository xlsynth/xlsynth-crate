// SPDX-License-Identifier: Apache-2.0

//! Boolean cone extraction utilities.
//!
//! This module supports extracting transitive fan-in cones for `bits[1]`-typed
//! nodes (\"booleans\") from an IR function, filtering by:
//! - maximum expression depth (exclusive)
//! - maximum number of referenced boundary `ParamId`s (exclusive)
//!
//! The extracted cones are emitted as minimal one-function IR packages and are
//! content-addressed by a SHA-256 hash of the emitted text.

use crate::ir::{Fn as IrFn, Node, NodePayload, NodeRef, Param, ParamId, Type, emit_fn};
use crate::ir_utils::{get_topological, operands, remap_payload_with};
use sha2::Digest;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct BoolConeExtractOptions {
    /// Cones must satisfy depth < max_depth_exclusive.
    pub max_depth_exclusive: usize,
    /// Cones must satisfy param_count < max_params_exclusive.
    pub max_params_exclusive: usize,
}

#[derive(Debug, Clone)]
pub struct ExtractedBoolCone {
    pub fn_text: String,
    pub sha256_hex: String,
    pub depth: usize,
    pub param_count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct BoolConeExtractStats {
    pub roots: usize,
    pub extracted_unique: usize,
    pub skipped_unsupported: usize,
    pub pruned_by_depth: usize,
    pub pruned_by_params: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootOutcome {
    Extracted,
    SkippedUnsupported,
    PrunedByDepth,
    PrunedByParams,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WalkError {
    Unsupported,
    PrunedByDepth,
    PrunedByParams,
    CycleDetected,
}

struct ConeWalkState<'a> {
    f: &'a IrFn,
    opts: &'a BoolConeExtractOptions,
    /// Boundary params referenced so far (original ParamId numbers).
    boundary_param_ids: BTreeSet<usize>,
    /// Internal nodes in the cone (indices into f.nodes), excluding GetParam.
    region_nodes: HashSet<usize>,
    /// Memoized depths for internal nodes (by node index).
    depth_memo: Vec<Option<usize>>,
    /// DFS recursion stack tracking to detect cycles.
    in_stack: Vec<bool>,
}

impl<'a> ConeWalkState<'a> {
    fn new(f: &'a IrFn, opts: &'a BoolConeExtractOptions) -> Self {
        Self {
            f,
            opts,
            boundary_param_ids: BTreeSet::new(),
            region_nodes: HashSet::new(),
            depth_memo: vec![None; f.nodes.len()],
            in_stack: vec![false; f.nodes.len()],
        }
    }

    fn note_param_id_or_prune(&mut self, pid: ParamId) -> Result<(), WalkError> {
        self.boundary_param_ids.insert(pid.get_wrapped_id());
        if self.boundary_param_ids.len() >= self.opts.max_params_exclusive {
            return Err(WalkError::PrunedByParams);
        }
        Ok(())
    }

    fn compute_depth_for_node(&mut self, nr: NodeRef) -> Result<usize, WalkError> {
        let idx = nr.index;
        if idx >= self.f.nodes.len() {
            return Err(WalkError::Unsupported);
        }

        let node = self.f.get_node(nr);
        match &node.payload {
            NodePayload::GetParam(pid) => {
                self.note_param_id_or_prune(*pid)?;
                let depth = 0usize;
                if depth >= self.opts.max_depth_exclusive {
                    return Err(WalkError::PrunedByDepth);
                }
                Ok(depth)
            }
            NodePayload::Literal(_) => {
                // Literals are internal to extracted cones (not boundary inputs).
                self.region_nodes.insert(idx);
                self.depth_memo[idx] = Some(0);
                let depth = 0usize;
                if depth >= self.opts.max_depth_exclusive {
                    return Err(WalkError::PrunedByDepth);
                }
                Ok(depth)
            }
            NodePayload::Nil => Err(WalkError::Unsupported),
            NodePayload::Invoke { .. } | NodePayload::CountedFor { .. } => {
                Err(WalkError::Unsupported)
            }
            _ => {
                if let Some(d) = self.depth_memo[idx] {
                    return Ok(d);
                }
                if self.in_stack[idx] {
                    return Err(WalkError::CycleDetected);
                }
                self.in_stack[idx] = true;

                // This node is part of the cone region (internal node).
                self.region_nodes.insert(idx);

                let mut max_child_depth: usize = 0;
                for dep in operands(&node.payload).into_iter() {
                    let dep_depth = self.compute_depth_for_node(dep)?;
                    if dep_depth > max_child_depth {
                        max_child_depth = dep_depth;
                    }
                }
                let depth = 1usize.saturating_add(max_child_depth);
                self.in_stack[idx] = false;
                self.depth_memo[idx] = Some(depth);

                if depth >= self.opts.max_depth_exclusive {
                    return Err(WalkError::PrunedByDepth);
                }
                Ok(depth)
            }
        }
    }
}

fn sha256_hex_of_text(s: &str) -> String {
    let digest = sha2::Sha256::digest(s.as_bytes());
    format!("{digest:x}")
}

fn build_param_id_to_info_map(f: &IrFn) -> HashMap<usize, (String, Type)> {
    let mut m: HashMap<usize, (String, Type)> = HashMap::new();
    for p in f.params.iter() {
        m.insert(p.id.get_wrapped_id(), (p.name.clone(), p.ty.clone()));
    }
    m
}

fn make_one_fn_package_text(func: &IrFn, package_name: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!("package {}\n\n", package_name));
    out.push_str(&emit_fn(func, /* is_top= */ true));
    out.push_str("\n\n");
    out
}

fn extract_canonical_cone_fn(
    f: &IrFn,
    root: NodeRef,
    region_nodes: &HashSet<usize>,
    boundary_param_ids: &BTreeSet<usize>,
) -> Result<IrFn, String> {
    assert!(root.index < f.nodes.len(), "root NodeRef out of bounds");

    let param_id_to_info = build_param_id_to_info_map(f);

    // Deterministic param order by original ParamId number.
    let mut params: Vec<Param> = Vec::with_capacity(boundary_param_ids.len());
    let mut param_old_ids_in_order: Vec<usize> = Vec::with_capacity(boundary_param_ids.len());
    let mut next_param_pos: usize = 1;
    for pid_num in boundary_param_ids.iter().copied() {
        let (name, ty) = param_id_to_info
            .get(&pid_num)
            .cloned()
            .ok_or_else(|| format!("missing param info for ParamId {}", pid_num))?;
        let new_pid = ParamId::new(next_param_pos);
        next_param_pos += 1;
        param_old_ids_in_order.push(pid_num);
        params.push(Param {
            name,
            ty,
            id: new_pid,
        });
    }

    // Build initial nodes: one GetParam per param.
    let mut nodes: Vec<Node> = Vec::new();
    let mut old_param_id_to_getparam_ref: HashMap<usize, NodeRef> = HashMap::new();
    let mut next_text_id: usize = 1;
    for (old_pid_num, p) in param_old_ids_in_order.iter().copied().zip(params.iter()) {
        nodes.push(Node {
            text_id: next_text_id,
            name: Some(p.name.clone()),
            ty: p.ty.clone(),
            payload: NodePayload::GetParam(p.id),
            pos: None,
        });
        old_param_id_to_getparam_ref.insert(
            old_pid_num,
            NodeRef {
                index: nodes.len() - 1,
            },
        );
        next_text_id += 1;
    }

    // Clone internal region nodes in topological order (stable).
    let topo = get_topological(f);
    let mut old_node_to_new: HashMap<usize, NodeRef> = HashMap::new();
    for nr in topo.into_iter() {
        if !region_nodes.contains(&nr.index) {
            continue;
        }
        let old = f.get_node(nr);
        let mapper = |(_op_idx, dep): (usize, NodeRef)| -> NodeRef {
            if region_nodes.contains(&dep.index) {
                *old_node_to_new
                    .get(&dep.index)
                    .expect("internal operand must be cloned before use")
            } else {
                // Boundary values should only be GetParam nodes by construction.
                let dep_node = f.get_node(dep);
                match dep_node.payload {
                    NodePayload::GetParam(old_pid) => *old_param_id_to_getparam_ref
                        .get(&old_pid.get_wrapped_id())
                        .expect("missing GetParam mapping for boundary ParamId"),
                    NodePayload::Literal(_) => {
                        panic!("literal operand unexpectedly outside region_nodes")
                    }
                    _ => {
                        panic!("unexpected non-GetParam boundary operand outside region_nodes");
                    }
                }
            }
        };
        let new_payload = remap_payload_with(&old.payload, mapper);
        nodes.push(Node {
            text_id: next_text_id,
            name: None,
            ty: old.ty.clone(),
            payload: new_payload,
            pos: None,
        });
        old_node_to_new.insert(
            nr.index,
            NodeRef {
                index: nodes.len() - 1,
            },
        );
        next_text_id += 1;
    }

    // Determine the return NodeRef in the cloned function.
    let ret_ref: NodeRef = {
        let root_node = f.get_node(root);
        match root_node.payload {
            NodePayload::GetParam(pid) => *old_param_id_to_getparam_ref
                .get(&pid.get_wrapped_id())
                .ok_or_else(|| "root GetParam pid not in boundary set".to_string())?,
            NodePayload::Literal(_) => *old_node_to_new
                .get(&root.index)
                .ok_or_else(|| "root literal unexpectedly missing from region".to_string())?,
            _ => *old_node_to_new
                .get(&root.index)
                .ok_or_else(|| "root node unexpectedly missing from region".to_string())?,
        }
    };

    let ret_ty = f.get_node(root).ty.clone();
    Ok(IrFn {
        name: "cone".to_string(),
        params,
        ret_ty,
        nodes,
        ret_node_ref: Some(ret_ref),
        outer_attrs: Vec::new(),
        inner_attrs: Vec::new(),
    })
}

fn attempt_extract_one_root(
    f: &IrFn,
    root: NodeRef,
    opts: &BoolConeExtractOptions,
) -> Result<(RootOutcome, Option<ExtractedBoolCone>), String> {
    let mut state = ConeWalkState::new(f, opts);

    let depth_result = state.compute_depth_for_node(root);
    let (depth, outcome) = match depth_result {
        Ok(d) => (d, RootOutcome::Extracted),
        Err(WalkError::Unsupported) => return Ok((RootOutcome::SkippedUnsupported, None)),
        Err(WalkError::PrunedByDepth) => return Ok((RootOutcome::PrunedByDepth, None)),
        Err(WalkError::PrunedByParams) => return Ok((RootOutcome::PrunedByParams, None)),
        Err(WalkError::CycleDetected) => {
            return Err("cycle detected while walking cone (unexpected in valid IR)".to_string());
        }
    };

    let param_count = state.boundary_param_ids.len();
    if depth >= opts.max_depth_exclusive {
        return Ok((RootOutcome::PrunedByDepth, None));
    }
    if param_count >= opts.max_params_exclusive {
        return Ok((RootOutcome::PrunedByParams, None));
    }
    assert!(
        depth < opts.max_depth_exclusive,
        "depth pruning should have occurred before extraction"
    );
    assert!(
        param_count < opts.max_params_exclusive,
        "param pruning should have occurred before extraction"
    );

    // Emit a canonical one-function package.
    let extracted_fn =
        extract_canonical_cone_fn(f, root, &state.region_nodes, &state.boundary_param_ids)?;
    let fn_text = make_one_fn_package_text(&extracted_fn, "bool_cone_pkg");
    let sha256_hex = sha256_hex_of_text(&fn_text);

    Ok((
        outcome,
        Some(ExtractedBoolCone {
            fn_text,
            sha256_hex,
            depth,
            param_count,
        }),
    ))
}

/// Extracts all unique boolean cones from `f` using `opts`.
///
/// Behavior summary:
/// - Roots are every node with type exactly `bits[1]` in node index order.
/// - Traversal cuts at `GetParam` nodes; referenced `ParamId`s become the
///   extracted signature.
/// - Cones are pruned early as soon as they cannot satisfy `depth < D` or
///   `params < P`.
/// - Cones containing unsupported ops (e.g. `invoke`, `counted_for`) are
///   skipped rather than failing.
pub fn extract_bool_cones_from_fn(
    f: &IrFn,
    opts: &BoolConeExtractOptions,
) -> Result<(Vec<ExtractedBoolCone>, BoolConeExtractStats), String> {
    let mut stats = BoolConeExtractStats::default();

    let mut cones_by_sha: BTreeMap<String, ExtractedBoolCone> = BTreeMap::new();
    for i in 0..f.nodes.len() {
        let nr = NodeRef { index: i };
        if f.get_node(nr).ty != Type::Bits(1) {
            continue;
        }
        stats.roots += 1;
        let (outcome, extracted_opt) = attempt_extract_one_root(f, nr, opts)?;
        match outcome {
            RootOutcome::Extracted => {
                if let Some(extracted) = extracted_opt {
                    cones_by_sha
                        .entry(extracted.sha256_hex.clone())
                        .or_insert(extracted);
                }
            }
            RootOutcome::SkippedUnsupported => stats.skipped_unsupported += 1,
            RootOutcome::PrunedByDepth => stats.pruned_by_depth += 1,
            RootOutcome::PrunedByParams => stats.pruned_by_params += 1,
        }
    }

    stats.extracted_unique = cones_by_sha.len();
    Ok((cones_by_sha.into_values().collect(), stats))
}
