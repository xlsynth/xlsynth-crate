// SPDX-License-Identifier: Apache-2.0

//! Enumerates, ranks, and extracts IR-function MFFCs (maximal fanout-free
//! cones).
//!
//! This module is intentionally library-oriented: it performs no stdout/stderr
//! output and returns structured results to callers (e.g. `xlsynth-driver`).

use crate::ir::{
    self, MemberType, Node, NodePayload, NodeRef, Package, PackageMember, Param, ParamId,
};
use crate::ir_utils::{operands, remap_payload_with};
use sha2::Digest;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug)]
pub struct MffcConfig {
    /// Optional cap: emit only the top-N ranked MFFCs.
    pub max_mffcs: Option<usize>,
    /// Minimum number of non-literal internal nodes required to keep a
    /// candidate.
    pub min_internal_non_literal_count: usize,
    /// Optional cap on non-literal frontier size.
    pub max_frontier_non_literal_count: Option<usize>,
    /// Whether to retain `pos` metadata (and the package file table, if
    /// provided to extraction).
    pub emit_pos_data: bool,
}

#[derive(Clone, Debug)]
pub struct MffcSpec {
    /// Root node index (into `Fn.nodes`) used for this MFFC.
    pub root_node_index: usize,
    /// Root node text_id for stable external reporting.
    pub root_text_id: usize,
    /// Sorted internal node indices (non-param nodes inside the cone).
    pub internal_node_indices: Vec<usize>,
    /// Sorted frontier node indices (boundary leaves feeding the cone).
    pub frontier_leaf_indices: Vec<usize>,
    /// Internal-node count excluding literals.
    pub internal_non_literal_count: usize,
    /// Frontier-leaf count excluding literals.
    pub frontier_non_literal_count: usize,
    /// Numerator for the "meat" ranking score.
    ///
    /// Current score: `internal_non_literal_count^2 /
    /// (frontier_non_literal_count + 1)`.
    pub score_numerator: usize,
    /// Denominator for the "meat" ranking score.
    pub score_denominator: usize,
}

#[derive(Clone, Debug)]
pub struct ExtractedMffc {
    pub package: Package,
    pub sha256_hex: String,
    pub root_node_index: usize,
    pub root_text_id: usize,
    pub included_node_count: usize,
    pub frontier_non_literal_count: usize,
    pub score_numerator: usize,
    pub score_denominator: usize,
}

fn is_literal_node(f: &ir::Fn, idx: usize) -> bool {
    matches!(f.nodes[idx].payload, NodePayload::Literal(_))
}

fn is_root_eligible(node: &Node) -> bool {
    !matches!(
        node.payload,
        NodePayload::Nil | NodePayload::GetParam(_) | NodePayload::Literal(_)
    )
}

/// Returns true if this function contains nodes that reference external callee
/// definitions (which MFFC extraction does not currently copy into the emitted
/// package).
pub fn has_nonlocal_callee_refs(f: &ir::Fn) -> bool {
    f.nodes.iter().any(|n| {
        matches!(
            n.payload,
            NodePayload::Invoke { .. } | NodePayload::CountedFor { .. }
        )
    })
}

/// Returns immediate users for each node index.
///
/// We deduplicate repeated operands from the same user node so each user is
/// counted once.
fn compute_users_by_index(f: &ir::Fn) -> Vec<Vec<usize>> {
    let mut users: Vec<Vec<usize>> = vec![Vec::new(); f.nodes.len()];
    for (user_idx, node) in f.nodes.iter().enumerate() {
        let mut deps: Vec<usize> = operands(&node.payload).iter().map(|nr| nr.index).collect();
        deps.sort_unstable();
        deps.dedup();
        for dep in deps {
            users[dep].push(user_idx);
        }
    }
    for u in users.iter_mut() {
        u.sort_unstable();
        u.dedup();
    }
    users
}

fn compute_single_mffc_spec(f: &ir::Fn, users: &[Vec<usize>], root_idx: usize) -> Option<MffcSpec> {
    if root_idx == 0 || root_idx >= f.nodes.len() {
        return None;
    }
    if !is_root_eligible(&f.nodes[root_idx]) {
        return None;
    }

    // Remaining live user counts while "deleting" the root cone.
    let mut remaining_users: Vec<usize> = users.iter().map(Vec::len).collect();
    let mut in_mffc: Vec<bool> = vec![false; f.nodes.len()];
    let mut frontier: BTreeSet<usize> = BTreeSet::new();
    let mut worklist: Vec<usize> = vec![root_idx];
    in_mffc[root_idx] = true;

    while let Some(dead_idx) = worklist.pop() {
        let mut preds: Vec<usize> = operands(&f.nodes[dead_idx].payload)
            .iter()
            .map(|nr| nr.index)
            .collect();
        preds.sort_unstable();
        preds.dedup();

        for pred_idx in preds {
            if in_mffc[pred_idx] {
                continue;
            }
            let pred = &f.nodes[pred_idx];
            match &pred.payload {
                NodePayload::Nil => {
                    // Nil is a reserved node and not considered a frontier
                    // leaf.
                }
                NodePayload::Literal(_) => {
                    // Inline literals in extracted cones, even if globally shared.
                    in_mffc[pred_idx] = true;
                    frontier.remove(&pred_idx);
                    worklist.push(pred_idx);
                }
                NodePayload::GetParam(_) => {
                    frontier.insert(pred_idx);
                }
                _ => {
                    if remaining_users[pred_idx] > 0 {
                        remaining_users[pred_idx] -= 1;
                    }
                    if remaining_users[pred_idx] == 0 {
                        in_mffc[pred_idx] = true;
                        frontier.remove(&pred_idx);
                        worklist.push(pred_idx);
                    } else {
                        frontier.insert(pred_idx);
                    }
                }
            }
        }
    }

    let internal_node_indices: Vec<usize> = (1..f.nodes.len())
        .filter(|&idx| {
            in_mffc[idx]
                && !matches!(
                    f.nodes[idx].payload,
                    NodePayload::Nil | NodePayload::GetParam(_)
                )
        })
        .collect();

    if internal_node_indices.is_empty() {
        return None;
    }

    let frontier_leaf_indices: Vec<usize> = frontier.into_iter().collect();

    let internal_non_literal_count = internal_node_indices
        .iter()
        .copied()
        .filter(|&idx| !is_literal_node(f, idx))
        .count();
    let frontier_non_literal_count = frontier_leaf_indices
        .iter()
        .copied()
        .filter(|&idx| !is_literal_node(f, idx))
        .count();

    let score_numerator = internal_non_literal_count.saturating_mul(internal_non_literal_count);
    let score_denominator = frontier_non_literal_count.saturating_add(1);

    Some(MffcSpec {
        root_node_index: root_idx,
        root_text_id: f.nodes[root_idx].text_id,
        internal_node_indices,
        frontier_leaf_indices,
        internal_non_literal_count,
        frontier_non_literal_count,
        score_numerator,
        score_denominator,
    })
}

/// Enumerates all MFFC candidates for eligible roots (before
/// ranking/filtering).
pub fn enumerate_all_mffc_specs(f: &ir::Fn) -> Vec<MffcSpec> {
    let users = compute_users_by_index(f);
    let mut out: Vec<MffcSpec> = Vec::new();
    for idx in 1..f.nodes.len() {
        if let Some(spec) = compute_single_mffc_spec(f, &users, idx) {
            out.push(spec);
        }
    }
    out
}

fn compare_meat_score_desc(a: &MffcSpec, b: &MffcSpec) -> Ordering {
    // Compare ratios `a.num/a.den` vs `b.num/b.den` without float instability.
    let a_cross: u128 = (a.score_numerator as u128) * (b.score_denominator as u128);
    let b_cross: u128 = (b.score_numerator as u128) * (a.score_denominator as u128);
    b_cross
        .cmp(&a_cross)
        .then_with(|| {
            b.internal_non_literal_count
                .cmp(&a.internal_non_literal_count)
        })
        .then_with(|| {
            a.frontier_non_literal_count
                .cmp(&b.frontier_non_literal_count)
        })
        .then_with(|| {
            b.internal_node_indices
                .len()
                .cmp(&a.internal_node_indices.len())
        })
        .then_with(|| a.root_node_index.cmp(&b.root_node_index))
}

/// Applies thresholding/ranking to an already-enumerated candidate set.
///
/// This function intentionally performs ranking *after* full enumeration to
/// enable "find the meatiest first" workflows.
pub fn rank_and_select_mffc_specs(mut specs: Vec<MffcSpec>, cfg: &MffcConfig) -> Vec<MffcSpec> {
    specs.retain(|s| s.internal_non_literal_count >= cfg.min_internal_non_literal_count);
    if let Some(max_frontier) = cfg.max_frontier_non_literal_count {
        specs.retain(|s| s.frontier_non_literal_count <= max_frontier);
    }
    specs.sort_by(compare_meat_score_desc);
    if let Some(max) = cfg.max_mffcs {
        if specs.len() > max {
            specs.truncate(max);
        }
    }
    specs
}

/// Enumerates, ranks, and selects MFFC specs according to `cfg`.
pub fn enumerate_mffc_specs(f: &ir::Fn, cfg: &MffcConfig) -> Vec<MffcSpec> {
    let all = enumerate_all_mffc_specs(f);
    rank_and_select_mffc_specs(all, cfg)
}

/// Extracts a single MFFC into a standalone package with one function.
///
/// - `pkg_file_table` is used only when `cfg.emit_pos_data == true`.
pub fn extract_mffc(
    f: &ir::Fn,
    pkg_file_table: Option<&ir::FileTable>,
    spec: &MffcSpec,
    cfg: &MffcConfig,
) -> ExtractedMffc {
    let root = spec.root_node_index;

    // Frontier params are all non-literal leaves.
    let mut param_leaf_indices: Vec<usize> = spec
        .frontier_leaf_indices
        .iter()
        .copied()
        .filter(|&idx| !is_literal_node(f, idx))
        .collect();
    param_leaf_indices.sort_unstable();
    param_leaf_indices.dedup();

    let mut included_internal: BTreeSet<usize> =
        spec.internal_node_indices.iter().copied().collect();
    // Include literal frontier leaves as constants (instead of parameters).
    for idx in spec.frontier_leaf_indices.iter().copied() {
        if is_literal_node(f, idx) {
            included_internal.insert(idx);
        }
    }

    // Build new params and their corresponding get_param nodes.
    let mut params: Vec<Param> = Vec::with_capacity(param_leaf_indices.len());
    let mut nodes: Vec<Node> = Vec::new();
    nodes.push(Node {
        text_id: 0,
        name: None,
        ty: ir::Type::nil(),
        payload: NodePayload::Nil,
        pos: None,
    });

    let mut old_to_new: HashMap<usize, usize> = HashMap::new();

    for (ordinal, old_idx) in param_leaf_indices.iter().copied().enumerate() {
        let param_id = ParamId::new(ordinal + 1);
        let param_name = format!("leaf_{}", old_idx);
        let param_ty = f.nodes[old_idx].ty.clone();
        params.push(Param {
            name: param_name.clone(),
            ty: param_ty.clone(),
            id: param_id,
        });
        nodes.push(Node {
            text_id: param_id.get_wrapped_id(),
            name: Some(param_name),
            ty: param_ty,
            payload: NodePayload::GetParam(param_id),
            pos: None,
        });
        old_to_new.insert(old_idx, nodes.len() - 1);
    }

    // Clone included nodes in ascending old-index order for deterministic
    // extraction and stable hashes.
    let mut next_text_id: usize = params.len() + 1;
    for old_idx in included_internal.iter().copied() {
        if old_to_new.contains_key(&old_idx) {
            continue;
        }
        let old_node = &f.nodes[old_idx];
        if matches!(
            old_node.payload,
            NodePayload::Invoke { .. } | NodePayload::CountedFor { .. }
        ) {
            panic!(
                "MFFC extraction does not support invoke/counted_for nodes because callee definitions are not copied into extracted packages"
            );
        }
        let new_payload = remap_payload_with(&old_node.payload, |(_slot, r): (usize, NodeRef)| {
            let new_index = *old_to_new.get(&r.index).unwrap_or_else(|| {
                panic!("missing mapping for operand {:?} while extracting MFFC", r)
            });
            NodeRef { index: new_index }
        });
        nodes.push(Node {
            text_id: next_text_id,
            name: None,
            ty: old_node.ty.clone(),
            payload: new_payload,
            pos: if cfg.emit_pos_data {
                old_node.pos.clone()
            } else {
                None
            },
        });
        old_to_new.insert(old_idx, nodes.len() - 1);
        next_text_id += 1;
    }

    let new_root_ref = NodeRef {
        index: *old_to_new
            .get(&root)
            .unwrap_or_else(|| panic!("root must be mapped while extracting MFFC: {}", root)),
    };

    let func = ir::Fn {
        name: "cone".to_string(),
        params,
        ret_ty: f.nodes[root].ty.clone(),
        nodes,
        ret_node_ref: Some(new_root_ref),
        outer_attrs: Vec::new(),
        inner_attrs: Vec::new(),
    };

    let package = Package {
        name: "fn_mffc".to_string(),
        file_table: if cfg.emit_pos_data {
            pkg_file_table.cloned().unwrap_or_else(ir::FileTable::new)
        } else {
            ir::FileTable::new()
        },
        members: vec![PackageMember::Function(func)],
        top: Some(("cone".to_string(), MemberType::Function)),
    };

    let text = package.to_string();
    let mut hasher = sha2::Sha256::new();
    hasher.update(text.as_bytes());
    let sha256_hex = format!("{:x}", hasher.finalize());

    ExtractedMffc {
        package,
        sha256_hex,
        root_node_index: spec.root_node_index,
        root_text_id: spec.root_text_id,
        included_node_count: included_internal.len(),
        frontier_non_literal_count: spec.frontier_non_literal_count,
        score_numerator: spec.score_numerator,
        score_denominator: spec.score_denominator,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser;

    fn parse_top_fn(pkg_text: &str) -> ir::Fn {
        let mut parser = ir_parser::Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        pkg.get_top_fn().unwrap().clone()
    }

    #[test]
    fn mffc_marks_shared_operand_as_frontier() {
        let pkg_text = r#"package p

top fn f(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  add.4: bits[8] = add(a, b, id=4)
  not.5: bits[8] = not(add.4, id=5)
  xor.6: bits[8] = xor(add.4, c, id=6)
  ret add.7: bits[8] = add(not.5, xor.6, id=7)
}
"#;
        let f = parse_top_fn(pkg_text);
        let specs = enumerate_all_mffc_specs(&f);
        let not_root = specs
            .iter()
            .find(|s| s.root_text_id == 5)
            .expect("expected MFFC spec for not.5");

        assert_eq!(not_root.internal_non_literal_count, 1);
        assert_eq!(not_root.frontier_leaf_indices, vec![4]);
        assert_eq!(not_root.frontier_non_literal_count, 1);
    }

    #[test]
    fn ranking_and_max_cap_pick_meatiest_first() {
        let pkg_text = r#"package p

top fn f(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  add.4: bits[8] = add(a, b, id=4)
  not.5: bits[8] = not(add.4, id=5)
  xor.6: bits[8] = xor(add.4, c, id=6)
  ret add.7: bits[8] = add(not.5, xor.6, id=7)
}
"#;
        let f = parse_top_fn(pkg_text);
        let cfg = MffcConfig {
            max_mffcs: Some(1),
            min_internal_non_literal_count: 2,
            max_frontier_non_literal_count: None,
            emit_pos_data: false,
        };
        let ranked = enumerate_mffc_specs(&f, &cfg);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].root_text_id, 7);
        assert!(ranked[0].internal_non_literal_count >= 2);
    }

    #[test]
    fn extract_mffc_is_deterministic_for_same_spec() {
        let pkg_text = r#"package p

top fn f(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  add.4: bits[8] = add(a, b, id=4)
  not.5: bits[8] = not(add.4, id=5)
  xor.6: bits[8] = xor(add.4, c, id=6)
  ret add.7: bits[8] = add(not.5, xor.6, id=7)
}
"#;
        let mut parser = ir_parser::Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_top_fn().unwrap();

        let cfg = MffcConfig {
            max_mffcs: None,
            min_internal_non_literal_count: 0,
            max_frontier_non_literal_count: None,
            emit_pos_data: false,
        };
        let specs = enumerate_mffc_specs(f, &cfg);
        let top_spec = specs
            .iter()
            .find(|s| s.root_text_id == 7)
            .expect("expected root add.7");

        let a = extract_mffc(f, Some(&pkg.file_table), top_spec, &cfg);
        let b = extract_mffc(f, Some(&pkg.file_table), top_spec, &cfg);

        assert_eq!(a.sha256_hex, b.sha256_hex);
        assert_eq!(a.package.to_string(), b.package.to_string());
    }

    #[test]
    fn has_nonlocal_callee_refs_detects_invoke_and_counted_for() {
        let invoke_pkg = r#"package p

fn g(x: bits[8] id=1) -> bits[8] {
  ret not.2: bits[8] = not(x, id=2)
}

top fn f(a: bits[8] id=10) -> bits[8] {
  ret invoke.11: bits[8] = invoke(a, to_apply=g, id=11)
}
"#;
        let invoke_f = parse_top_fn(invoke_pkg);
        assert!(has_nonlocal_callee_refs(&invoke_f));

        let counted_for_pkg = r#"package p

fn body(x: bits[8] id=1) -> bits[8] {
  ret add.2: bits[8] = add(x, x, id=2)
}

top fn f(a: bits[8] id=10) -> bits[8] {
  ret counted_for.11: bits[8] = counted_for(a, trip_count=2, stride=1, body=body, id=11)
}
"#;
        let counted_for_f = parse_top_fn(counted_for_pkg);
        assert!(has_nonlocal_callee_refs(&counted_for_f));

        let plain_pkg = r#"package p

top fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret add.3: bits[8] = add(a, b, id=3)
}
"#;
        let plain_f = parse_top_fn(plain_pkg);
        assert!(!has_nonlocal_callee_refs(&plain_f));
    }
}
