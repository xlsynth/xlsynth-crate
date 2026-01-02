// SPDX-License-Identifier: Apache-2.0

//! Enumerates and extracts boolean cones (constrained by k-feasible cuts) from
//! an XLS IR function.
//!
//! This module is intentionally library-oriented: it performs no stdout/stderr
//! output and returns structured results to callers (e.g. `xlsynth-driver`).

use crate::ir::{
    self, ArrayTypeData, MemberType, Node, NodePayload, NodeRef, Package, PackageMember, Param,
    ParamId, Type,
};
use crate::ir_utils::{operands, remap_payload_with};
use sha2::Digest;
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug)]
pub struct BoolConeConfig {
    /// Maximum frontier size (K) for cuts.
    pub k: usize,
    /// Safety cap: maximum number of cuts retained per node during enumeration.
    pub max_cuts_per_node: usize,
    /// Optional global cap: stop after enumerating this many cone specs.
    pub max_cones: Option<usize>,
    /// Whether to retain `pos` metadata (and the package file table, if
    /// provided to extraction).
    pub emit_pos_data: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cut {
    /// Sorted, unique node indices (into `Fn.nodes`) that form the frontier.
    pub leaves: Vec<usize>,
}

impl Cut {
    pub fn new(mut leaves: Vec<usize>) -> Self {
        leaves.sort_unstable();
        leaves.dedup();
        Self { leaves }
    }

    fn is_subset_of(&self, other: &Cut) -> bool {
        // Both are sorted unique.
        let mut i = 0usize;
        let mut j = 0usize;
        while i < self.leaves.len() && j < other.leaves.len() {
            let a = self.leaves[i];
            let b = other.leaves[j];
            if a == b {
                i += 1;
                j += 1;
            } else if a > b {
                j += 1;
            } else {
                // a < b: element a is not present in other.
                return false;
            }
        }
        i == self.leaves.len()
    }

    fn non_literal_cost(&self, f: &ir::Fn) -> usize {
        self.leaves
            .iter()
            .map(|&idx| match &f.nodes[idx].payload {
                NodePayload::Literal(_) => 0usize,
                _ => frontier_value_cost(&f.nodes[idx].ty),
            })
            .sum()
    }
}

#[derive(Clone, Debug)]
pub struct ConeSpec {
    pub sink: usize,
    pub cut: Cut,
}

#[derive(Clone, Debug)]
pub struct ExtractedCone {
    pub package: Package,
    pub sha256_hex: String,
    pub sink: usize,
    pub cut: Cut,
    pub included_node_count: usize,
    pub frontier_non_literal_count: usize,
}

/// Returns true if the extracted cone is a trivial passthrough of a single
/// parameter (i.e. the function body is just `ret <param> = param(...)`).
pub fn is_trivial_param_return_cone(cone: &ExtractedCone) -> bool {
    let member = match cone.package.members.as_slice() {
        [m] => m,
        _ => return false,
    };
    let f = match member {
        PackageMember::Function(f) => f,
        PackageMember::Block { .. } => return false,
    };
    if f.params.len() != 1 {
        return false;
    }
    if f.nodes.len() != 2 {
        // Nil + GetParam
        return false;
    }
    if f.ret_node_ref != Some(NodeRef { index: 1 }) {
        return false;
    }
    let p = &f.params[0];
    match &f.nodes[1].payload {
        NodePayload::GetParam(pid) => pid.get_wrapped_id() == p.id.get_wrapped_id(),
        _ => false,
    }
}

/// Returns true if the extracted cone is a trivial constant function with no
/// parameters, i.e. it returns a single literal node.
pub fn is_trivial_literal_return_cone(cone: &ExtractedCone) -> bool {
    let member = match cone.package.members.as_slice() {
        [m] => m,
        _ => return false,
    };
    let f = match member {
        PackageMember::Function(f) => f,
        PackageMember::Block { .. } => return false,
    };
    if !f.params.is_empty() {
        return false;
    }
    if f.nodes.len() != 2 {
        // Nil + Literal
        return false;
    }
    if f.ret_node_ref != Some(NodeRef { index: 1 }) {
        return false;
    }
    matches!(f.nodes[1].payload, NodePayload::Literal(_))
}

/// Returns `true` iff `nr` is a boolean-valued node (type exactly `bits[1]`).
pub fn is_bits1_node(f: &ir::Fn, nr: NodeRef) -> bool {
    matches!(f.get_node_ty(nr), Type::Bits(1))
}

fn is_literal_node(f: &ir::Fn, idx: usize) -> bool {
    matches!(f.nodes[idx].payload, NodePayload::Literal(_))
}

fn frontier_value_cost(ty: &Type) -> usize {
    match ty {
        Type::Token => 1,
        Type::Bits(_) => 1,
        Type::Tuple(elems) => elems.iter().map(|t| frontier_value_cost(t)).sum(),
        Type::Array(ArrayTypeData {
            element_type,
            element_count,
        }) => element_count.saturating_mul(frontier_value_cost(element_type)),
    }
}

fn union_sorted_unique(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(a.len() + b.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        let av = a[i];
        let bv = b[j];
        if av == bv {
            out.push(av);
            i += 1;
            j += 1;
        } else if av < bv {
            out.push(av);
            i += 1;
        } else {
            out.push(bv);
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out.dedup();
    out
}

fn prune_cuts_dedup_dominance_and_cap(
    mut cuts: Vec<Cut>,
    f: &ir::Fn,
    k: usize,
    cap: usize,
) -> Vec<Cut> {
    // Dedup via ordering.
    cuts.sort();
    cuts.dedup();

    // Stable ordering by (cost, leaf_count, lex leaves) for deterministic caps.
    cuts.sort_by(|a, b| {
        let ca = a.non_literal_cost(f);
        let cb = b.non_literal_cost(f);
        ca.cmp(&cb)
            .then_with(|| a.leaves.len().cmp(&b.leaves.len()))
            .then_with(|| a.cmp(b))
    });

    // Filter by K (literals cost 0).
    cuts.retain(|c| c.non_literal_cost(f) <= k);

    // Dominance: if we already have a cut whose leaves are a subset of this one,
    // drop the dominated (superset) cut.
    let mut kept: Vec<Cut> = Vec::new();
    'outer: for c in cuts.into_iter() {
        for kcut in kept.iter() {
            if kcut.is_subset_of(&c) {
                continue 'outer;
            }
        }
        kept.push(c);
        if kept.len() >= cap {
            break;
        }
    }

    kept
}

/// Computes k-feasible cuts for all nodes in `f`.
///
/// Cut cost counts **only non-literal** leaves.
pub fn compute_k_cuts(f: &ir::Fn, cfg: &BoolConeConfig) -> Vec<Vec<Cut>> {
    let n = f.nodes.len();
    let mut cuts_for: Vec<Vec<Cut>> = vec![Vec::new(); n];

    // Index 0 is the reserved Nil node; we don't enumerate cuts for it.
    if n > 0 {
        cuts_for[0] = Vec::new();
    }

    for idx in 1..n {
        let node = &f.nodes[idx];
        let deps = operands(&node.payload);
        let mut candidates: Vec<Cut> = Vec::new();

        // Always include the trivial cut {idx}.
        candidates.push(Cut::new(vec![idx]));

        if !deps.is_empty() {
            // Fold operand cut-sets via cross product.
            let mut partial: Vec<Cut> = vec![Cut::new(Vec::new())];
            for dep in deps.iter() {
                let dep_cuts = &cuts_for[dep.index];
                let mut next: Vec<Cut> = Vec::new();
                for p in partial.iter() {
                    for dc in dep_cuts.iter() {
                        let merged = union_sorted_unique(&p.leaves, &dc.leaves);
                        let c = Cut { leaves: merged };
                        if c.non_literal_cost(f) <= cfg.k {
                            next.push(c);
                        }
                    }
                }
                partial = prune_cuts_dedup_dominance_and_cap(next, f, cfg.k, cfg.max_cuts_per_node);
                if partial.is_empty() {
                    break;
                }
            }
            candidates.extend(partial.into_iter());
        }

        cuts_for[idx] =
            prune_cuts_dedup_dominance_and_cap(candidates, f, cfg.k, cfg.max_cuts_per_node);
    }

    cuts_for
}

/// Enumerates all `(sink, cut)` pairs where `sink` is a `bits[1]` node and
/// `cut` is a k-feasible cut (frontier size ≤ K, not counting literals).
pub fn enumerate_bool_cone_specs(f: &ir::Fn, cfg: &BoolConeConfig) -> Vec<ConeSpec> {
    let cuts_for = compute_k_cuts(f, cfg);
    let mut specs: Vec<ConeSpec> = Vec::new();

    for idx in 1..f.nodes.len() {
        if !is_bits1_node(f, NodeRef { index: idx }) {
            continue;
        }
        for c in cuts_for[idx].iter() {
            specs.push(ConeSpec {
                sink: idx,
                cut: c.clone(),
            });
            if let Some(max) = cfg.max_cones {
                if specs.len() >= max {
                    return specs;
                }
            }
        }
    }
    specs
}

fn collect_included_nodes(f: &ir::Fn, sink: usize, cut: &Cut) -> BTreeSet<usize> {
    let leaves_set: BTreeSet<usize> = cut.leaves.iter().copied().collect();
    let mut included: BTreeSet<usize> = BTreeSet::new();
    let mut stack: Vec<usize> = vec![sink];

    while let Some(idx) = stack.pop() {
        if included.contains(&idx) {
            continue;
        }
        if leaves_set.contains(&idx) && !is_literal_node(f, idx) {
            // Boundary: do not include the defining node of non-literal leaves.
            continue;
        }
        included.insert(idx);
        let deps = operands(&f.nodes[idx].payload);
        for d in deps {
            stack.push(d.index);
        }
    }

    included
}

/// Extracts a single cone into a standalone package with one function.
///
/// - `pkg_file_table` is used only when `cfg.emit_pos_data == true`.
pub fn extract_bool_cone(
    f: &ir::Fn,
    pkg_file_table: Option<&ir::FileTable>,
    spec: &ConeSpec,
    cfg: &BoolConeConfig,
) -> ExtractedCone {
    let sink = spec.sink;
    let cut = spec.cut.clone();

    let included = collect_included_nodes(f, sink, &cut);

    // Frontier params are all non-literal leaves.
    let mut param_leaf_indices: Vec<usize> = cut
        .leaves
        .iter()
        .copied()
        .filter(|&idx| !is_literal_node(f, idx))
        .collect();
    param_leaf_indices.sort_unstable();
    param_leaf_indices.dedup();

    let frontier_non_literal_count = param_leaf_indices.len();

    // Build new params and their corresponding GetParam nodes.
    let mut params: Vec<Param> = Vec::with_capacity(param_leaf_indices.len());
    let mut nodes: Vec<Node> = Vec::new();

    // Reserved Nil node at index 0.
    nodes.push(Node {
        text_id: 0,
        name: None,
        ty: Type::nil(),
        payload: NodePayload::Nil,
        pos: None,
    });

    // Map old node indices to new node indices for boundary leaves.
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

    // Clone included nodes in ascending original index order, remapping operands.
    // Assign new text ids deterministically after the param ids.
    let mut next_text_id: usize = params.len() + 1;
    for old_idx in included.iter().copied() {
        // Skip any included index that has already been mapped (should not
        // happen for non-literal leaves; but harmless).
        if old_to_new.contains_key(&old_idx) {
            continue;
        }
        let old_node = &f.nodes[old_idx];
        let mapper = |(_slot, r): (usize, NodeRef)| -> NodeRef {
            let new_index = *old_to_new
                .get(&r.index)
                .unwrap_or_else(|| panic!("missing old_to_new mapping for operand {:?}", r));
            NodeRef { index: new_index }
        };
        let new_payload = remap_payload_with(&old_node.payload, mapper);
        let new_node = Node {
            text_id: next_text_id,
            // Drop original names for canonicalization; params remain named.
            name: None,
            ty: old_node.ty.clone(),
            payload: new_payload,
            pos: if cfg.emit_pos_data {
                old_node.pos.clone()
            } else {
                None
            },
        };
        let new_index = nodes.len();
        nodes.push(new_node);
        old_to_new.insert(old_idx, new_index);
        next_text_id += 1;
    }

    let new_sink_ref = NodeRef {
        index: *old_to_new
            .get(&sink)
            .unwrap_or_else(|| panic!("sink must be mapped: {}", sink)),
    };

    let func = ir::Fn {
        name: "cone".to_string(),
        params,
        ret_ty: f.nodes[sink].ty.clone(),
        nodes,
        ret_node_ref: Some(new_sink_ref),
        outer_attrs: Vec::new(),
        inner_attrs: Vec::new(),
    };

    let package = Package {
        name: "bool_cone".to_string(),
        file_table: if cfg.emit_pos_data {
            pkg_file_table.cloned().unwrap_or_else(ir::FileTable::new)
        } else {
            ir::FileTable::new()
        },
        members: vec![PackageMember::Function(func)],
        top: Some(("cone".to_string(), MemberType::Function)),
    };

    // Compute sha256 of the exact printed package text.
    let text = package.to_string();
    let mut hasher = sha2::Sha256::new();
    hasher.update(text.as_bytes());
    let sha256_hex = format!("{:x}", hasher.finalize());

    ExtractedCone {
        package,
        sha256_hex,
        sink,
        cut,
        included_node_count: included.len(),
        frontier_non_literal_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser;
    use pretty_assertions::assert_eq;

    #[test]
    fn tuple_frontier_cost_counts_tuple_elements() {
        // If a cut uses the tuple-typed param node as a leaf, that should cost
        // the number of tuple elements (here: 3), not 1.
        let text = r#"fn f(x: (bits[1], bits[8], bits[7]) id=1) -> bits[1] {
  t0: bits[1] = tuple_index(x, index=0, id=2)
  ret not.3: bits[1] = not(t0, id=3)
}"#;
        let mut p = ir_parser::Parser::new(text);
        let f = p.parse_fn().unwrap();

        // With K=2, the cut {x} (leaf index 1) should be rejected (cost 3).
        let cfg_k2 = BoolConeConfig {
            k: 2,
            max_cuts_per_node: 128,
            max_cones: None,
            emit_pos_data: false,
        };
        let cuts_k2 = compute_k_cuts(&f, &cfg_k2);
        let tuple_index_ref = 2usize;
        assert!(
            !cuts_k2[tuple_index_ref]
                .iter()
                .any(|c| c.leaves == vec![1usize]),
            "expected cut {{x}} to be absent when K=2 (tuple cost should be 3)"
        );

        // With K=3, the cut {x} should be allowed.
        let cfg_k3 = BoolConeConfig {
            k: 3,
            max_cuts_per_node: 128,
            max_cones: None,
            emit_pos_data: false,
        };
        let cuts_k3 = compute_k_cuts(&f, &cfg_k3);
        assert!(
            cuts_k3[tuple_index_ref]
                .iter()
                .any(|c| c.leaves == vec![1usize]),
            "expected cut {{x}} to be present when K=3"
        );
    }

    #[test]
    fn k_cuts_do_not_count_literals_toward_k() {
        // bits[1] = or_reduce(literal bits[4]:0b0011) => still a sink bits[1].
        let text = r#"fn f() -> bits[1] {
  literal.1: bits[4] = literal(value=3, id=1)
  ret or_reduce.2: bits[1] = or_reduce(literal.1, id=2)
}"#;
        let mut p = ir_parser::Parser::new(text);
        let f = p.parse_fn().unwrap();

        let cfg = BoolConeConfig {
            k: 0,
            max_cuts_per_node: 128,
            max_cones: None,
            emit_pos_data: false,
        };
        let specs = enumerate_bool_cone_specs(&f, &cfg);
        assert!(
            !specs.is_empty(),
            "expected at least one cone spec for bits[1] sink"
        );
        // Ensure we can extract at least one cone without any non-literal leaves.
        let extracted = extract_bool_cone(&f, None, &specs[0], &cfg);
        assert_eq!(extracted.frontier_non_literal_count, 0);
        // Round-trip parse of the emitted package should succeed.
        let emitted_text = extracted.package.to_string();
        let mut p2 = ir_parser::Parser::new(&emitted_text);
        let pkg2 = p2.parse_and_validate_package().unwrap();
        assert_eq!(pkg2.to_string(), emitted_text);
    }

    #[test]
    fn extract_cone_is_deterministic_for_same_input() {
        let pkg_text = r#"package p

top fn f(x: bits[8] id=1) -> bits[1] {
  literal.2: bits[8] = literal(value=0, id=2)
  ne.3: bits[1] = ne(x, literal.2, id=3)
  ret not.4: bits[1] = not(ne.3, id=4)
}
"#;
        let mut parser = ir_parser::Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_top_fn().unwrap();

        let cfg = BoolConeConfig {
            k: 2,
            max_cuts_per_node: 256,
            max_cones: Some(10),
            emit_pos_data: false,
        };
        let specs = enumerate_bool_cone_specs(f, &cfg);
        assert!(!specs.is_empty());

        let a = extract_bool_cone(f, Some(&pkg.file_table), &specs[0], &cfg);
        let b = extract_bool_cone(f, Some(&pkg.file_table), &specs[0], &cfg);
        assert_eq!(a.sha256_hex, b.sha256_hex);
        assert_eq!(a.package.to_string(), b.package.to_string());
    }

    #[test]
    fn detects_trivial_param_return_cone() {
        let pkg_text = r#"package p

top fn f(x: bits[1] id=1) -> bits[1] {
  ret x: bits[1] = param(name=x, id=1)
}
"#;
        let mut parser = ir_parser::Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_top_fn().unwrap();

        let cfg = BoolConeConfig {
            k: 1,
            max_cuts_per_node: 128,
            max_cones: None,
            emit_pos_data: false,
        };
        let specs = enumerate_bool_cone_specs(f, &cfg);
        assert!(!specs.is_empty());

        // Find the cone spec where the sink is the param node itself.
        let sink_param_idx = 1usize;
        let spec = specs
            .iter()
            .find(|s| s.sink == sink_param_idx && s.cut.leaves == vec![sink_param_idx])
            .expect("expected trivial cut at param sink");
        let cone = extract_bool_cone(f, Some(&pkg.file_table), spec, &cfg);
        assert!(is_trivial_param_return_cone(&cone));
    }

    #[test]
    fn detects_trivial_literal_return_cone() {
        let pkg_text = r#"package p

top fn f() -> bits[1] {
  ret literal.1: bits[1] = literal(value=1, id=1)
}
"#;
        let mut parser = ir_parser::Parser::new(pkg_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_top_fn().unwrap();

        let cfg = BoolConeConfig {
            k: 0,
            max_cuts_per_node: 128,
            max_cones: None,
            emit_pos_data: false,
        };
        let specs = enumerate_bool_cone_specs(f, &cfg);
        assert!(!specs.is_empty());

        let spec = specs
            .iter()
            .find(|s| s.sink == 1usize && s.cut.leaves == vec![1usize])
            .expect("expected trivial cut at literal sink");
        let cone = extract_bool_cone(f, Some(&pkg.file_table), spec, &cfg);
        assert!(is_trivial_literal_return_cone(&cone));
    }
}
