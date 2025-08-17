// SPDX-License-Identifier: Apache-2.0

//! Computes structural similarity statistics between two XLS IR functions.

use std::collections::HashMap;

use crate::xls_ir::ir::{Fn, NodePayload, NodeRef, ParamId, Type};
use crate::xls_ir::ir_utils::get_topological;

fn update_hash_str(hasher: &mut blake3::Hasher, s: &str) {
    hasher.update(s.as_bytes());
}

fn update_hash_u64(hasher: &mut blake3::Hasher, x: u64) {
    hasher.update(&x.to_le_bytes());
}

fn update_hash_bool(hasher: &mut blake3::Hasher, x: bool) {
    update_hash_u64(hasher, if x { 1 } else { 0 });
}

fn update_hash_type(hasher: &mut blake3::Hasher, ty: &Type) {
    match ty {
        Type::Token => update_hash_str(hasher, "token"),
        Type::Bits(width) => {
            update_hash_str(hasher, "bits");
            update_hash_u64(hasher, *width as u64);
        }
        Type::Tuple(elems) => {
            update_hash_str(hasher, "tuple");
            update_hash_u64(hasher, elems.len() as u64);
            for e in elems.iter() {
                update_hash_type(hasher, e);
            }
        }
        Type::Array(arr) => {
            update_hash_str(hasher, "array");
            update_hash_u64(hasher, arr.element_count as u64);
            update_hash_type(hasher, &arr.element_type);
        }
    }
}

fn payload_tag(payload: &NodePayload) -> &'static str {
    match payload {
        NodePayload::Nil => "nil",
        NodePayload::GetParam(_) => "get_param",
        NodePayload::Tuple(_) => "tuple",
        NodePayload::Array(_) => "array",
        NodePayload::TupleIndex { .. } => "tuple_index",
        NodePayload::Binop(_, _, _) => "binop",
        NodePayload::Unop(_, _) => "unop",
        NodePayload::Literal(_) => "literal",
        NodePayload::SignExt { .. } => "sign_ext",
        NodePayload::ZeroExt { .. } => "zero_ext",
        NodePayload::ArrayUpdate { .. } => "array_update",
        NodePayload::ArrayIndex { .. } => "array_index",
        NodePayload::DynamicBitSlice { .. } => "dynamic_bit_slice",
        NodePayload::BitSlice { .. } => "bit_slice",
        NodePayload::BitSliceUpdate { .. } => "bit_slice_update",
        NodePayload::Assert { .. } => "assert",
        NodePayload::Trace { .. } => "trace",
        NodePayload::AfterAll(_) => "after_all",
        NodePayload::Nary(_, _) => "nary",
        NodePayload::Invoke { .. } => "invoke",
        NodePayload::PrioritySel { .. } => "priority_sel",
        NodePayload::OneHotSel { .. } => "one_hot_sel",
        NodePayload::OneHot { .. } => "one_hot",
        NodePayload::Sel { .. } => "sel",
        NodePayload::Cover { .. } => "cover",
        NodePayload::Decode { .. } => "decode",
        NodePayload::Encode { .. } => "encode",
        NodePayload::CountedFor { .. } => "counted_for",
    }
}

fn get_param_ordinal(f: &Fn, param_id: ParamId) -> usize {
    f.params
        .iter()
        .position(|p| p.id == param_id)
        .expect("ParamId must correspond to a function parameter")
}

fn hash_payload_attributes(f: &Fn, payload: &NodePayload, hasher: &mut blake3::Hasher) {
    match payload {
        NodePayload::Nil => {}
        NodePayload::GetParam(param_id) => {
            // Use stable ordinal position within the function signature, not the text id.
            let ordinal = get_param_ordinal(f, *param_id) as u64 + 1;
            update_hash_u64(hasher, ordinal);
        }
        NodePayload::Tuple(nodes) | NodePayload::Array(nodes) => {
            update_hash_u64(hasher, nodes.len() as u64);
        }
        NodePayload::TupleIndex { tuple: _, index } => update_hash_u64(hasher, *index as u64),
        NodePayload::Binop(op, _, _) => {
            update_hash_str(hasher, crate::xls_ir::ir::binop_to_operator(*op))
        }
        NodePayload::Unop(op, _) => {
            update_hash_str(hasher, crate::xls_ir::ir::unop_to_operator(*op))
        }
        NodePayload::Literal(value) => update_hash_str(hasher, &value.to_string()),
        NodePayload::SignExt { new_bit_count, .. } => {
            update_hash_u64(hasher, *new_bit_count as u64)
        }
        NodePayload::ZeroExt { new_bit_count, .. } => {
            update_hash_u64(hasher, *new_bit_count as u64)
        }
        NodePayload::ArrayUpdate {
            array: _,
            value: _,
            indices: _,
            assumed_in_bounds,
        } => update_hash_bool(hasher, *assumed_in_bounds),
        NodePayload::ArrayIndex {
            array: _,
            indices: _,
            assumed_in_bounds,
        } => update_hash_bool(hasher, *assumed_in_bounds),
        NodePayload::DynamicBitSlice { width, .. } => update_hash_u64(hasher, *width as u64),
        NodePayload::BitSlice { start, width, .. } => {
            update_hash_u64(hasher, *start as u64);
            update_hash_u64(hasher, *width as u64);
        }
        NodePayload::BitSliceUpdate {
            arg: _,
            start: _,
            update_value: _,
        } => {}
        NodePayload::Assert {
            token: _,
            activate: _,
            message: _,
            label: _,
        } => {}
        NodePayload::Trace {
            token: _,
            activated: _,
            format: _,
            operands: _,
        } => {}
        NodePayload::AfterAll(nodes) => update_hash_u64(hasher, nodes.len() as u64),
        NodePayload::Nary(op, nodes) => {
            update_hash_str(hasher, crate::xls_ir::ir::nary_op_to_operator(*op));
            update_hash_u64(hasher, nodes.len() as u64);
        }
        NodePayload::Invoke { to_apply, operands } => {
            update_hash_str(hasher, to_apply);
            update_hash_u64(hasher, operands.len() as u64);
        }
        NodePayload::PrioritySel {
            selector: _,
            default,
            cases,
        } => {
            update_hash_bool(hasher, default.is_some());
            update_hash_u64(hasher, cases.len() as u64);
        }
        NodePayload::OneHotSel { selector: _, cases } => {
            update_hash_u64(hasher, cases.len() as u64)
        }
        NodePayload::OneHot { arg: _, lsb_prio } => update_hash_bool(hasher, *lsb_prio),
        NodePayload::Sel {
            selector: _,
            default,
            cases,
        } => {
            update_hash_bool(hasher, default.is_some());
            update_hash_u64(hasher, cases.len() as u64);
        }
        NodePayload::Cover {
            predicate: _,
            label: _,
        } => {}
        NodePayload::Decode { arg: _, width } => update_hash_u64(hasher, *width as u64),
        NodePayload::Encode { arg: _ } => {}
        NodePayload::CountedFor {
            init: _,
            trip_count,
            stride,
            body,
            invariant_args,
        } => {
            update_hash_u64(hasher, *trip_count as u64);
            update_hash_u64(hasher, *stride as u64);
            update_hash_str(hasher, body);
            update_hash_u64(hasher, invariant_args.len() as u64);
        }
    }
}

fn compute_node_structural_hash(
    f: &Fn,
    node_ref: NodeRef,
    child_hashes: &[blake3::Hash],
) -> blake3::Hash {
    let node = f.get_node(node_ref);
    let mut hasher = blake3::Hasher::new();
    update_hash_str(&mut hasher, payload_tag(&node.payload));
    update_hash_type(&mut hasher, &node.ty);
    hash_payload_attributes(f, &node.payload, &mut hasher);
    for ch in child_hashes.iter() {
        hasher.update(ch.as_bytes());
    }
    hasher.finalize()
}

fn compute_node_depth(f: &Fn, node_ref: NodeRef, child_depths: &[usize]) -> usize {
    match &f.get_node(node_ref).payload {
        NodePayload::Nil
        | NodePayload::GetParam(_)
        | NodePayload::Literal(_)
        | NodePayload::Decode { .. }
        | NodePayload::Encode { .. } => 0,
        _ => 1 + child_depths.iter().copied().max().unwrap_or(0),
    }
}

pub struct StructuralEntry {
    pub hash: blake3::Hash,
    pub depth: usize,
    pub id: usize,
    pub signature: String,
}

/// Helper: walk a function and collect structural entries for each node.
pub fn collect_structural_entries(f: &Fn) -> (Vec<StructuralEntry>, Vec<usize>) {
    let order = get_topological(f);
    let n = f.nodes.len();
    let mut hashes: Vec<blake3::Hash> = vec![blake3::Hash::from([0u8; 32]); n];
    let mut depths: Vec<usize> = vec![0; n];

    for node_ref in order {
        // Gather child info in the same order as operands appear.
        let node = f.get_node(node_ref);
        let mut child_hashes: Vec<blake3::Hash> = Vec::new();
        let mut child_depths: Vec<usize> = Vec::new();
        match &node.payload {
            NodePayload::Nil => {}
            NodePayload::GetParam(_) => {}
            NodePayload::Tuple(elems)
            | NodePayload::Array(elems)
            | NodePayload::AfterAll(elems)
            | NodePayload::Nary(_, elems) => {
                for r in elems.iter() {
                    child_hashes.push(hashes[r.index]);
                    child_depths.push(depths[r.index]);
                }
            }
            NodePayload::TupleIndex { tuple, .. }
            | NodePayload::Unop(_, tuple)
            | NodePayload::Decode { arg: tuple, .. }
            | NodePayload::Encode { arg: tuple }
            | NodePayload::OneHot { arg: tuple, .. }
            | NodePayload::BitSlice { arg: tuple, .. } => {
                child_hashes.push(hashes[tuple.index]);
                child_depths.push(depths[tuple.index]);
            }
            NodePayload::Binop(_, a, b) => {
                child_hashes.push(hashes[a.index]);
                child_depths.push(depths[a.index]);
                child_hashes.push(hashes[b.index]);
                child_depths.push(depths[b.index]);
            }
            NodePayload::SignExt { arg, .. } | NodePayload::ZeroExt { arg, .. } => {
                child_hashes.push(hashes[arg.index]);
                child_depths.push(depths[arg.index]);
            }
            NodePayload::ArrayUpdate {
                array,
                value,
                indices,
                ..
            } => {
                child_hashes.push(hashes[array.index]);
                child_depths.push(depths[array.index]);
                child_hashes.push(hashes[value.index]);
                child_depths.push(depths[value.index]);
                for r in indices.iter() {
                    child_hashes.push(hashes[r.index]);
                    child_depths.push(depths[r.index]);
                }
            }
            NodePayload::ArrayIndex { array, indices, .. } => {
                child_hashes.push(hashes[array.index]);
                child_depths.push(depths[array.index]);
                for r in indices.iter() {
                    child_hashes.push(hashes[r.index]);
                    child_depths.push(depths[r.index]);
                }
            }
            NodePayload::DynamicBitSlice { arg, start, .. } => {
                child_hashes.push(hashes[arg.index]);
                child_depths.push(depths[arg.index]);
                child_hashes.push(hashes[start.index]);
                child_depths.push(depths[start.index]);
            }
            NodePayload::BitSliceUpdate {
                arg,
                start,
                update_value,
            } => {
                child_hashes.push(hashes[arg.index]);
                child_depths.push(depths[arg.index]);
                child_hashes.push(hashes[start.index]);
                child_depths.push(depths[start.index]);
                child_hashes.push(hashes[update_value.index]);
                child_depths.push(depths[update_value.index]);
            }
            NodePayload::Assert {
                token, activate, ..
            } => {
                child_hashes.push(hashes[token.index]);
                child_depths.push(depths[token.index]);
                child_hashes.push(hashes[activate.index]);
                child_depths.push(depths[activate.index]);
            }
            NodePayload::Trace {
                token,
                activated,
                operands,
                ..
            } => {
                child_hashes.push(hashes[token.index]);
                child_depths.push(depths[token.index]);
                child_hashes.push(hashes[activated.index]);
                child_depths.push(depths[activated.index]);
                for r in operands.iter() {
                    child_hashes.push(hashes[r.index]);
                    child_depths.push(depths[r.index]);
                }
            }
            NodePayload::Invoke { operands, .. } => {
                for r in operands.iter() {
                    child_hashes.push(hashes[r.index]);
                    child_depths.push(depths[r.index]);
                }
            }
            NodePayload::PrioritySel {
                selector,
                cases,
                default,
            }
            | NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                child_hashes.push(hashes[selector.index]);
                child_depths.push(depths[selector.index]);
                for r in cases.iter() {
                    child_hashes.push(hashes[r.index]);
                    child_depths.push(depths[r.index]);
                }
                if let Some(d) = default {
                    child_hashes.push(hashes[d.index]);
                    child_depths.push(depths[d.index]);
                }
            }
            NodePayload::OneHotSel { selector, cases } => {
                child_hashes.push(hashes[selector.index]);
                child_depths.push(depths[selector.index]);
                for r in cases.iter() {
                    child_hashes.push(hashes[r.index]);
                    child_depths.push(depths[r.index]);
                }
            }
            NodePayload::CountedFor {
                init,
                invariant_args,
                ..
            } => {
                child_hashes.push(hashes[init.index]);
                child_depths.push(depths[init.index]);
                for r in invariant_args.iter() {
                    child_hashes.push(hashes[r.index]);
                    child_depths.push(depths[r.index]);
                }
            }
            NodePayload::Literal(_) => {}
            NodePayload::Cover { predicate, .. } => {
                child_hashes.push(hashes[predicate.index]);
                child_depths.push(depths[predicate.index]);
            }
        }

        let h = compute_node_structural_hash(f, node_ref, &child_hashes);
        let d = compute_node_depth(f, node_ref, &child_depths);
        hashes[node_ref.index] = h;
        depths[node_ref.index] = d;
    }

    let mut entries: Vec<StructuralEntry> = Vec::with_capacity(n);
    for (i, node) in f.nodes.iter().enumerate() {
        let h = hashes[i];
        let d = depths[i];
        let id = node.text_id;
        let signature = node.to_signature_string(f);
        entries.push(StructuralEntry {
            hash: h,
            depth: d,
            id,
            signature,
        });
    }
    (entries, depths)
}

/// Computes a histogram mapping depth to discrepancy count between two IR
/// functions. Discrepancies are counted as the symmetric multiset difference of
/// structural hashes at each depth.
pub fn discrepancies_by_depth(lhs: &Fn, rhs: &Fn) -> HashMap<usize, usize> {
    // Validate signatures match.
    assert_eq!(
        lhs.get_type(),
        rhs.get_type(),
        "Function signatures must match for structural comparison"
    );

    let (lhs_entries, _lhs_depths) = collect_structural_entries(lhs);
    let (rhs_entries, _rhs_depths) = collect_structural_entries(rhs);

    // Build depth -> hash -> count maps.
    let mut lhs_map: HashMap<usize, HashMap<blake3::Hash, usize>> = HashMap::new();
    let mut rhs_map: HashMap<usize, HashMap<blake3::Hash, usize>> = HashMap::new();

    for e in lhs_entries.into_iter() {
        *lhs_map
            .entry(e.depth)
            .or_default()
            .entry(e.hash)
            .or_default() += 1;
    }
    for e in rhs_entries.into_iter() {
        *rhs_map
            .entry(e.depth)
            .or_default()
            .entry(e.hash)
            .or_default() += 1;
    }

    // Union of depths.
    let mut result: HashMap<usize, usize> = HashMap::new();
    let mut all_depths: Vec<usize> = lhs_map.keys().copied().collect();
    for d in rhs_map.keys() {
        if !all_depths.contains(d) {
            all_depths.push(*d);
        }
    }

    for d in all_depths.into_iter() {
        let mut total: usize = 0;
        let l = lhs_map.get(&d).cloned().unwrap_or_default();
        let r = rhs_map.get(&d).cloned().unwrap_or_default();

        let mut hashes: Vec<blake3::Hash> = l.keys().copied().collect();
        for h in r.keys() {
            if !hashes.contains(h) {
                hashes.push(*h);
            }
        }

        for h in hashes.into_iter() {
            let lc = *l.get(&h).unwrap_or(&0);
            let rc = *r.get(&h).unwrap_or(&0);
            let diff = if lc > rc { lc - rc } else { rc - lc };
            total += diff;
        }
        if total > 0 {
            result.insert(d, total);
        }
    }

    result
}

pub struct DepthDiscrepancy {
    pub depth: usize,
    pub lhs_only: Vec<(String, usize)>,
    pub rhs_only: Vec<(String, usize)>,
}

/// Computes detailed discrepancies grouped by depth, and the return-node depths
/// for LHS and RHS.
pub fn compute_structural_discrepancies(
    lhs: &Fn,
    rhs: &Fn,
) -> (Vec<DepthDiscrepancy>, usize, usize) {
    assert_eq!(
        lhs.get_type(),
        rhs.get_type(),
        "Function signatures must match for structural comparison",
    );

    let (lhs_entries, lhs_depths) = collect_structural_entries(lhs);
    let (rhs_entries, rhs_depths) = collect_structural_entries(rhs);

    let lhs_ret_depth: usize = match lhs.ret_node_ref {
        Some(nr) => lhs_depths[nr.index],
        None => lhs_depths.into_iter().max().unwrap_or(0),
    };
    let rhs_ret_depth: usize = match rhs.ret_node_ref {
        Some(nr) => rhs_depths[nr.index],
        None => rhs_depths.into_iter().max().unwrap_or(0),
    };

    // Build depth -> hash -> signature -> count maps.
    let mut lhs_map: HashMap<usize, HashMap<blake3::Hash, HashMap<String, usize>>> = HashMap::new();
    let mut rhs_map: HashMap<usize, HashMap<blake3::Hash, HashMap<String, usize>>> = HashMap::new();
    for e in lhs_entries {
        *lhs_map
            .entry(e.depth)
            .or_default()
            .entry(e.hash)
            .or_default()
            .entry(e.signature)
            .or_default() += 1;
    }
    for e in rhs_entries {
        *rhs_map
            .entry(e.depth)
            .or_default()
            .entry(e.hash)
            .or_default()
            .entry(e.signature)
            .or_default() += 1;
    }

    // Union of depths
    let mut all_depths: Vec<usize> = lhs_map.keys().copied().collect();
    for d in rhs_map.keys() {
        if !all_depths.contains(d) {
            all_depths.push(*d);
        }
    }
    all_depths.sort_unstable();

    let mut result: Vec<DepthDiscrepancy> = Vec::new();
    for d in all_depths.into_iter() {
        let mut lhs_only: HashMap<String, usize> = HashMap::new();
        let mut rhs_only: HashMap<String, usize> = HashMap::new();
        let l = lhs_map.get(&d);
        let r = rhs_map.get(&d);

        // Union of hashes at this depth
        let mut hashes: Vec<blake3::Hash> = Vec::new();
        if let Some(m) = l {
            hashes.extend(m.keys().copied());
        }
        if let Some(m) = r {
            for h in m.keys() {
                if !hashes.contains(h) {
                    hashes.push(*h);
                }
            }
        }

        for h in hashes.into_iter() {
            let l_sig = l.and_then(|m| m.get(&h)).cloned().unwrap_or_default();
            let r_sig = r.and_then(|m| m.get(&h)).cloned().unwrap_or_default();

            // Union of signatures for this hash
            let mut sigs: Vec<String> = l_sig.keys().cloned().collect();
            for s in r_sig.keys() {
                if !sigs.contains(s) {
                    sigs.push(s.clone());
                }
            }
            for s in sigs.into_iter() {
                let lc = *l_sig.get(&s).unwrap_or(&0);
                let rc = *r_sig.get(&s).unwrap_or(&0);
                if lc > rc {
                    *lhs_only.entry(s).or_default() += lc - rc;
                } else if rc > lc {
                    *rhs_only.entry(s).or_default() += rc - lc;
                }
            }
        }

        if !lhs_only.is_empty() || !rhs_only.is_empty() {
            let mut lhs_vec: Vec<(String, usize)> = lhs_only.into_iter().collect();
            let mut rhs_vec: Vec<(String, usize)> = rhs_only.into_iter().collect();
            lhs_vec.sort_by(|a, b| a.0.cmp(&b.0));
            rhs_vec.sort_by(|a, b| a.0.cmp(&b.0));
            result.push(DepthDiscrepancy {
                depth: d,
                lhs_only: lhs_vec,
                rhs_only: rhs_vec,
            });
        }
    }

    (result, lhs_ret_depth, rhs_ret_depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xls_ir::ir_parser::Parser;

    fn parse_fn(ir: &str) -> Fn {
        let mut p = Parser::new(ir);
        p.parse_fn().unwrap()
    }

    #[test]
    fn identical_functions_have_no_discrepancies() {
        let f = parse_fn(
            "fn id(a: bits[1] id=1) -> bits[1] {\n  ret identity.2: bits[1] = identity(a, id=2)\n}",
        );
        let g = parse_fn(
            "fn id(a: bits[1] id=1) -> bits[1] {\n  ret identity.2: bits[1] = identity(a, id=2)\n}",
        );
        let hist = discrepancies_by_depth(&f, &g);
        assert!(hist.is_empty());
    }

    #[test]
    fn single_op_difference_counts_at_correct_depth() {
        let f =
            parse_fn("fn n(a: bits[1] id=1) -> bits[1] {\n  ret not.2: bits[1] = not(a, id=2)\n}");
        let g = parse_fn(
            "fn i(a: bits[1] id=1) -> bits[1] {\n  ret identity.2: bits[1] = identity(a, id=2)\n}",
        );
        let hist = discrepancies_by_depth(&f, &g);
        // Param (depth 0) matches; op differs at depth 1. Symmetric multiset diff = 2.
        assert_eq!(hist.get(&1).copied(), Some(2));
        assert_eq!(hist.len(), 1);
    }
}
