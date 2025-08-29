// SPDX-License-Identifier: Apache-2.0

//! Computes structural similarity statistics between two XLS IR functions.

use std::collections::HashMap;

use crate::xls_ir::ir::{Fn, Node, NodePayload, NodeRef, Param, ParamId, Type};
use crate::xls_ir::ir_utils::{get_topological, operands};
use crate::xls_ir::node_hashing::{
    compute_node_local_structural_hash, compute_node_structural_hash,
};

/// Returns true if two IR functions are structurally equivalent (forward and
/// backward user-context) and have identical signatures.
pub fn structurally_equivalent_ir(lhs: &Fn, rhs: &Fn) -> bool {
    if lhs.get_type() != rhs.get_type() {
        return false;
    }
    discrepancies_by_depth(lhs, rhs).is_empty() && discrepancies_by_depth_bwd(lhs, rhs).is_empty()
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

/// Helper: build user lists with operand indices for each node in the function.
fn collect_users_with_operand_indices(f: &Fn) -> Vec<Vec<(NodeRef, usize)>> {
    let n = f.nodes.len();
    let mut users: Vec<Vec<(NodeRef, usize)>> = vec![Vec::new(); n];
    for (i, node) in f.nodes.iter().enumerate() {
        let this_ref = NodeRef { index: i };
        let ops = operands(&node.payload);
        for (operand_index, dep) in ops.into_iter().enumerate() {
            users[dep.index].push((this_ref, operand_index));
        }
    }
    users
}

/// Helper: walk a function and collect backward structural entries for each
/// node.
///
/// A node's backward hash is computed from its local structural hash combined
/// with a sorted vector of (user_backward_hash, user_operand_index) pairs.
pub fn collect_backward_structural_entries(f: &Fn) -> (Vec<StructuralEntry>, Vec<usize>) {
    let order = get_topological(f);
    let n = f.nodes.len();
    let users = collect_users_with_operand_indices(f);

    let mut bwd_hashes: Vec<blake3::Hash> = vec![blake3::Hash::from([0u8; 32]); n];
    let mut depths: Vec<usize> = vec![0; n];

    // Iterate in reverse topological order so users are processed before their
    // operands.
    for node_ref in order.into_iter().rev() {
        // Gather user info for this node.
        let mut user_pairs: Vec<(blake3::Hash, usize)> = Vec::new();
        let mut user_depths: Vec<usize> = Vec::new();
        for (user_ref, operand_index) in users[node_ref.index].iter().copied() {
            user_pairs.push((bwd_hashes[user_ref.index], operand_index));
            user_depths.push(depths[user_ref.index]);
        }

        // Sort by (hash bytes, operand index) for a stable characterization.
        user_pairs.sort_by(|a, b| {
            let ord = a.0.as_bytes().cmp(b.0.as_bytes());
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
            a.1.cmp(&b.1)
        });

        // Combine local hash with user context.
        let mut hasher = blake3::Hasher::new();
        let local = compute_node_local_structural_hash(f, node_ref);
        hasher.update(local.as_bytes());
        hasher.update(&u64::try_from(user_pairs.len()).unwrap_or(0).to_le_bytes());
        for (uh, idx) in user_pairs.into_iter() {
            hasher.update(uh.as_bytes());
            hasher.update(&u64::try_from(idx).unwrap_or(0).to_le_bytes());
        }
        bwd_hashes[node_ref.index] = hasher.finalize();

        // Backward depth: sinks (no users) are depth 0; else 1 + max(user depths).
        depths[node_ref.index] = match user_depths.iter().copied().max() {
            Some(m) => 1 + m,
            None => 0,
        };
    }

    let mut entries: Vec<StructuralEntry> = Vec::with_capacity(n);
    for (i, node) in f.nodes.iter().enumerate() {
        let h = bwd_hashes[i];
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

/// Computes a histogram mapping backward depth to discrepancy count between two
/// IR functions, using backward user-context structural hashes.
pub fn discrepancies_by_depth_bwd(lhs: &Fn, rhs: &Fn) -> HashMap<usize, usize> {
    assert_eq!(
        lhs.get_type(),
        rhs.get_type(),
        "Function signatures must match for structural comparison",
    );

    let (lhs_entries, _lhs_depths) = collect_backward_structural_entries(lhs);
    let (rhs_entries, _rhs_depths) = collect_backward_structural_entries(rhs);

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

/// Computes detailed discrepancies after matching nodes by either forward OR
/// backward structural equivalence. A node is considered matched if its forward
/// hash matches or its backward hash matches a counterpart on the other side.
/// Remaining unmatched nodes are grouped by forward depth and signature, and
/// the forward return-node depths are returned.
pub fn compute_structural_discrepancies_dual(
    lhs: &Fn,
    rhs: &Fn,
) -> (Vec<DepthDiscrepancy>, usize, usize) {
    assert_eq!(
        lhs.get_type(),
        rhs.get_type(),
        "Function signatures must match for structural comparison",
    );

    let (lhs_fwd_entries, lhs_fwd_depths) = collect_structural_entries(lhs);
    let (rhs_fwd_entries, rhs_fwd_depths) = collect_structural_entries(rhs);
    let (lhs_bwd_entries, _lhs_bwd_depths) = collect_backward_structural_entries(lhs);
    let (rhs_bwd_entries, _rhs_bwd_depths) = collect_backward_structural_entries(rhs);

    let lhs_ret_depth: usize = match lhs.ret_node_ref {
        Some(nr) => lhs_fwd_depths[nr.index],
        None => lhs_fwd_depths.into_iter().max().unwrap_or(0),
    };
    let rhs_ret_depth: usize = match rhs.ret_node_ref {
        Some(nr) => rhs_fwd_depths[nr.index],
        None => rhs_fwd_depths.into_iter().max().unwrap_or(0),
    };

    // Build maps from fwd/bwd hash -> rhs indices (unmatched pool).
    let mut rhs_fwd_map: HashMap<blake3::Hash, Vec<usize>> = HashMap::new();
    let mut rhs_bwd_map: HashMap<blake3::Hash, Vec<usize>> = HashMap::new();
    let rhs_len = rhs_fwd_entries.len();
    for i in 0..rhs_len {
        rhs_fwd_map
            .entry(rhs_fwd_entries[i].hash)
            .or_default()
            .push(i);
        rhs_bwd_map
            .entry(rhs_bwd_entries[i].hash)
            .or_default()
            .push(i);
    }
    // We'll pop from the back for O(1) removal; order does not matter.
    for v in rhs_fwd_map.values_mut() {
        v.shrink_to_fit();
    }
    for v in rhs_bwd_map.values_mut() {
        v.shrink_to_fit();
    }

    let mut rhs_matched: Vec<bool> = vec![false; rhs_len];
    let mut lhs_unmatched: Vec<usize> = Vec::new();

    // Greedy matching: forward-hash first, then backward-hash.
    for i in 0..lhs_fwd_entries.len() {
        let fwd_h = lhs_fwd_entries[i].hash;
        let bwd_h = lhs_bwd_entries[i].hash;
        let mut matched = false;
        if let Some(v) = rhs_fwd_map.get_mut(&fwd_h) {
            while let Some(j) = v.pop() {
                if !rhs_matched[j] {
                    rhs_matched[j] = true;
                    matched = true;
                    break;
                }
            }
        }
        if !matched {
            if let Some(v) = rhs_bwd_map.get_mut(&bwd_h) {
                while let Some(j) = v.pop() {
                    if !rhs_matched[j] {
                        rhs_matched[j] = true;
                        matched = true;
                        break;
                    }
                }
            }
        }
        if !matched {
            lhs_unmatched.push(i);
        }
    }

    // Remaining unmatched on RHS are those not marked matched.
    let mut rhs_unmatched: Vec<usize> = Vec::new();
    for (j, used) in rhs_matched.iter().enumerate() {
        if !*used {
            rhs_unmatched.push(j);
        }
    }

    // Group unmatched by forward depth -> signature -> count, mirroring forward
    // presentation.
    let mut lhs_map: HashMap<usize, HashMap<String, usize>> = HashMap::new();
    let mut rhs_map: HashMap<usize, HashMap<String, usize>> = HashMap::new();
    for i in lhs_unmatched.into_iter() {
        let d = lhs_fwd_entries[i].depth;
        let sig = lhs_fwd_entries[i].signature.clone();
        *lhs_map.entry(d).or_default().entry(sig).or_default() += 1;
    }
    for j in rhs_unmatched.into_iter() {
        let d = rhs_fwd_entries[j].depth;
        let sig = rhs_fwd_entries[j].signature.clone();
        *rhs_map.entry(d).or_default().entry(sig).or_default() += 1;
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
        let l = lhs_map.get(&d).cloned().unwrap_or_default();
        let r = rhs_map.get(&d).cloned().unwrap_or_default();

        // Union of signatures
        let mut sigs: Vec<String> = l.keys().cloned().collect();
        for s in r.keys() {
            if !sigs.contains(s) {
                sigs.push(s.clone());
            }
        }

        let mut lhs_only: Vec<(String, usize)> = Vec::new();
        let mut rhs_only: Vec<(String, usize)> = Vec::new();
        for s in sigs.into_iter() {
            let lc = *l.get(&s).unwrap_or(&0);
            let rc = *r.get(&s).unwrap_or(&0);
            if lc > rc {
                lhs_only.push((s, lc - rc));
            } else if rc > lc {
                rhs_only.push((s, rc - lc));
            }
        }

        if !lhs_only.is_empty() || !rhs_only.is_empty() {
            lhs_only.sort_by(|a, b| a.0.cmp(&b.0));
            rhs_only.sort_by(|a, b| a.0.cmp(&b.0));
            result.push(DepthDiscrepancy {
                depth: d,
                lhs_only,
                rhs_only,
            });
        }
    }

    (result, lhs_ret_depth, rhs_ret_depth)
}

fn remap_payload_with<FMap>(payload: &NodePayload, mut map: FMap) -> NodePayload
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

fn compute_dual_unmatched_masks(lhs: &Fn, rhs: &Fn) -> (Vec<bool>, Vec<bool>) {
    let (lhs_fwd_entries, _lhs_fwd_depths) = collect_structural_entries(lhs);
    let (rhs_fwd_entries, _rhs_fwd_depths) = collect_structural_entries(rhs);
    let (lhs_bwd_entries, _lhs_bwd_depths) = collect_backward_structural_entries(lhs);
    let (rhs_bwd_entries, _rhs_bwd_depths) = collect_backward_structural_entries(rhs);

    let rhs_len = rhs_fwd_entries.len();
    let mut rhs_fwd_map: HashMap<blake3::Hash, Vec<usize>> = HashMap::new();
    let mut rhs_bwd_map: HashMap<blake3::Hash, Vec<usize>> = HashMap::new();
    for j in 0..rhs_len {
        rhs_fwd_map
            .entry(rhs_fwd_entries[j].hash)
            .or_default()
            .push(j);
        rhs_bwd_map
            .entry(rhs_bwd_entries[j].hash)
            .or_default()
            .push(j);
    }
    let mut rhs_matched: Vec<bool> = vec![false; rhs_len];
    let mut lhs_unmatched_mask: Vec<bool> = vec![false; lhs_fwd_entries.len()];

    for i in 0..lhs_fwd_entries.len() {
        let fwd_h = lhs_fwd_entries[i].hash;
        let bwd_h = lhs_bwd_entries[i].hash;
        let mut matched = false;
        if let Some(v) = rhs_fwd_map.get_mut(&fwd_h) {
            while let Some(j) = v.pop() {
                if !rhs_matched[j] {
                    rhs_matched[j] = true;
                    matched = true;
                    break;
                }
            }
        }
        if !matched {
            if let Some(v) = rhs_bwd_map.get_mut(&bwd_h) {
                while let Some(j) = v.pop() {
                    if !rhs_matched[j] {
                        rhs_matched[j] = true;
                        matched = true;
                        break;
                    }
                }
            }
        }
        if !matched {
            lhs_unmatched_mask[i] = true;
        }
    }

    let mut rhs_unmatched_mask: Vec<bool> = vec![false; rhs_len];
    for j in 0..rhs_len {
        if !rhs_matched[j] {
            rhs_unmatched_mask[j] = true;
        }
    }

    (lhs_unmatched_mask, rhs_unmatched_mask)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ConsumerKey {
    local_hash: blake3::Hash,
    bwd_hash: blake3::Hash,
    operand_index: usize,
}

fn build_subgraph_fn_from_unmatched(
    orig: &Fn,
    unmatched_mask: &[bool],
    consumer_keys: Option<&[ConsumerKey]>,
) -> Fn {
    let n = orig.nodes.len();
    let mut include: Vec<bool> = vec![false; n];
    for i in 0..n {
        include[i] = if i < unmatched_mask.len() {
            unmatched_mask[i]
        } else {
            false
        };
    }

    // Determine external dependencies to become parameters.
    let mut external_sources: Vec<usize> = Vec::new();
    for (i, node) in orig.nodes.iter().enumerate() {
        if !include[i] {
            continue;
        }
        let deps = operands(&node.payload);
        for d in deps {
            if !include[d.index] && !external_sources.contains(&d.index) {
                external_sources.push(d.index);
            }
        }
    }
    external_sources.sort_unstable();

    // Create parameters and corresponding GetParam nodes.
    let mut params: Vec<Param> = Vec::new();
    let mut new_nodes: Vec<Node> = Vec::new();
    let mut next_param_id: usize = 1;
    let mut name_used: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut ext_to_new_node: HashMap<usize, usize> = HashMap::new();

    for src_idx in external_sources.iter().copied() {
        let src_node = &orig.nodes[src_idx];
        let mut pname = src_node
            .name
            .clone()
            .unwrap_or_else(|| format!("x{}", src_node.text_id));
        if name_used.contains(&pname) {
            let mut k = 1usize;
            while name_used.contains(&format!("{}__{}", pname, k)) {
                k += 1;
            }
            pname = format!("{}__{}", pname, k);
        }
        name_used.insert(pname.clone());

        let pid = ParamId::new(next_param_id);
        next_param_id += 1;
        params.push(Param {
            name: pname.clone(),
            ty: src_node.ty.clone(),
            id: pid,
        });
        let get_param_node = Node {
            text_id: new_nodes.len() + 1,
            name: Some(pname.clone()),
            ty: src_node.ty.clone(),
            payload: NodePayload::GetParam(pid),
            pos: src_node.pos.clone(),
        };
        let new_idx = new_nodes.len();
        new_nodes.push(get_param_node);
        ext_to_new_node.insert(src_idx, new_idx);
    }

    // Map from original included node index -> new node index
    let mut inc_to_new: HashMap<usize, usize> = HashMap::new();
    let order = get_topological(orig);
    for node_ref in order {
        let i = node_ref.index;
        if !include[i] {
            continue;
        }
        let old_node = &orig.nodes[i];
        let mapper = |r: NodeRef| -> NodeRef {
            if include[r.index] {
                let ni = *inc_to_new
                    .get(&r.index)
                    .expect("dep included must have been added");
                NodeRef { index: ni }
            } else {
                let ni = *ext_to_new_node
                    .get(&r.index)
                    .expect("external dep must be in param map");
                NodeRef { index: ni }
            }
        };
        let new_payload = remap_payload_with(&old_node.payload, mapper);
        let new_node = Node {
            text_id: new_nodes.len() + 1,
            name: old_node.name.clone(),
            ty: old_node.ty.clone(),
            payload: new_payload,
            pos: old_node.pos.clone(),
        };
        let new_idx = new_nodes.len();
        new_nodes.push(new_node);
        inc_to_new.insert(i, new_idx);
    }

    // Determine sinks in the included subgraph (used if no consumer_keys provided)
    let mut used_count: HashMap<usize, usize> = HashMap::new();
    for (i, node) in orig.nodes.iter().enumerate() {
        if !include[i] {
            continue;
        }
        for dep in operands(&node.payload) {
            if include[dep.index] {
                *used_count.entry(dep.index).or_insert(0) += 1;
            }
        }
    }
    let mut sink_new_refs: Vec<NodeRef> = Vec::new();
    for (i, inc) in include.iter().enumerate() {
        if *inc && used_count.get(&i).copied().unwrap_or(0) == 0 {
            let new_i = *inc_to_new.get(&i).expect("included node must be mapped");
            sink_new_refs.push(NodeRef { index: new_i });
        }
    }

    // Build the function return.
    let mut ret_ty: Type = Type::nil();
    let mut ret_node_ref: Option<NodeRef> = None;
    if let Some(keys) = consumer_keys {
        // Build outputs following the provided consumer keys. For each key, we
        // find a matching consumer in the original function that consumes some
        // source value and map that source into the new function, creating a
        // parameter if it is external.
        // Build a quick index: for each included node, enumerate users outside.
        let users = collect_users_with_operand_indices(orig);
        let (bwd_entries, _bwd_depths) = collect_backward_structural_entries(orig);
        let mut out_refs: Vec<NodeRef> = Vec::new();
        for key in keys.iter() {
            // Locate any consumer edge that matches this key.
            let mut found_src: Option<NodeRef> = None;
            'outer: for (src_idx, inc) in include.iter().enumerate() {
                if !*inc {
                    continue;
                }
                for (user_ref, op_index) in users[src_idx].iter().copied() {
                    if include[user_ref.index] {
                        continue;
                    }
                    if op_index != key.operand_index {
                        continue;
                    }
                    let local_h = compute_node_local_structural_hash(orig, user_ref);
                    let bwd_h = bwd_entries[user_ref.index].hash;
                    if local_h == key.local_hash && bwd_h == key.bwd_hash {
                        // The source feeding this operand is src_idx.
                        found_src = Some(NodeRef { index: src_idx });
                        break 'outer;
                    }
                }
            }
            if let Some(src) = found_src {
                // Map src into new function space; if external, add a param.
                let mapped = if include[src.index] {
                    let ni = *inc_to_new
                        .get(&src.index)
                        .expect("included node must be mapped");
                    NodeRef { index: ni }
                } else if let Some(&ni) = ext_to_new_node.get(&src.index) {
                    NodeRef { index: ni }
                } else {
                    // Create a new param for this external source
                    let src_node = &orig.nodes[src.index];
                    let mut pname = src_node
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("x{}", src_node.text_id));
                    if name_used.contains(&pname) {
                        let mut k = 1usize;
                        while name_used.contains(&format!("{}__{}", pname, k)) {
                            k += 1;
                        }
                        pname = format!("{}__{}", pname, k);
                    }
                    name_used.insert(pname.clone());
                    let pid = ParamId::new(next_param_id);
                    next_param_id += 1;
                    params.push(Param {
                        name: pname.clone(),
                        ty: src_node.ty.clone(),
                        id: pid,
                    });
                    let get_param_node = Node {
                        text_id: new_nodes.len() + 1,
                        name: Some(pname.clone()),
                        ty: src_node.ty.clone(),
                        payload: NodePayload::GetParam(pid),
                        pos: src_node.pos.clone(),
                    };
                    let new_idx = new_nodes.len();
                    new_nodes.push(get_param_node);
                    ext_to_new_node.insert(src.index, new_idx);
                    NodeRef { index: new_idx }
                };
                out_refs.push(mapped);
            } else {
                // If no match found on this side, pass through a unit literal as placeholder.
                // However, to maintain return arity, use a nil tuple element by skipping; we
                // choose to map to a zero-width tuple which is invalid; instead, fallback to
                // using a GetParam created above by adding a new param with nil type. Since
                // Type::Tuple(vec![]) is valid, but we have no literal Nil node; use an empty
                // tuple node.
                let nil_ty = Type::nil();
                let nil_node = Node {
                    text_id: new_nodes.len() + 1,
                    name: None,
                    ty: nil_ty.clone(),
                    payload: NodePayload::Tuple(vec![]),
                    pos: None,
                };
                let idx = new_nodes.len();
                new_nodes.push(nil_node);
                out_refs.push(NodeRef { index: idx });
            }
        }
        if out_refs.len() == 1 {
            let nref = out_refs[0];
            ret_ty = new_nodes[nref.index].ty.clone();
            ret_node_ref = Some(nref);
        } else {
            let elem_tys: Vec<Box<Type>> = out_refs
                .iter()
                .map(|r| Box::new(new_nodes[r.index].ty.clone()))
                .collect();
            let tuple_ty = Type::Tuple(elem_tys);
            let tuple_node = Node {
                text_id: new_nodes.len() + 1,
                name: None,
                ty: tuple_ty.clone(),
                payload: NodePayload::Tuple(out_refs.clone()),
                pos: None,
            };
            let tuple_idx = new_nodes.len();
            new_nodes.push(tuple_node);
            ret_ty = tuple_ty;
            ret_node_ref = Some(NodeRef { index: tuple_idx });
        }
    } else if sink_new_refs.len() == 1 {
        let nref = sink_new_refs[0];
        ret_ty = new_nodes[nref.index].ty.clone();
        ret_node_ref = Some(nref);
    } else if sink_new_refs.len() >= 2 {
        let elem_tys: Vec<Box<Type>> = sink_new_refs
            .iter()
            .map(|r| Box::new(new_nodes[r.index].ty.clone()))
            .collect();
        let tuple_ty = Type::Tuple(elem_tys);
        let tuple_node = Node {
            text_id: new_nodes.len() + 1,
            name: None,
            ty: tuple_ty.clone(),
            payload: NodePayload::Tuple(sink_new_refs.clone()),
            pos: None,
        };
        let tuple_idx = new_nodes.len();
        new_nodes.push(tuple_node);
        ret_ty = tuple_ty;
        ret_node_ref = Some(NodeRef { index: tuple_idx });
    }

    Fn {
        name: format!("{}_diff", orig.name),
        params,
        ret_ty,
        nodes: new_nodes,
        ret_node_ref,
    }
}

/// Extracts LHS/RHS subgraphs of unmatched nodes after dual (fwd OR bwd)
/// matching. External dependencies are turned into parameters with
/// corresponding GetParam nodes.
pub fn extract_dual_difference_subgraphs(lhs: &Fn, rhs: &Fn) -> (Fn, Fn) {
    assert_eq!(
        lhs.get_type(),
        rhs.get_type(),
        "Function signatures must match for structural comparison",
    );
    let (lhs_mask, rhs_mask) = compute_dual_unmatched_masks(lhs, rhs);

    // Compute common consumer keys across LHS and RHS from boundary edges.
    let users_lhs = collect_users_with_operand_indices(lhs);
    let users_rhs = collect_users_with_operand_indices(rhs);
    let (lhs_bwd_entries, _ld) = collect_backward_structural_entries(lhs);
    let (rhs_bwd_entries, _rd) = collect_backward_structural_entries(rhs);

    let mut lhs_keys: std::collections::HashSet<ConsumerKey> = std::collections::HashSet::new();
    for (i, inc) in lhs_mask.iter().enumerate() {
        if !*inc {
            continue;
        }
        for (user_ref, op_index) in users_lhs[i].iter().copied() {
            if lhs_mask[user_ref.index] {
                continue;
            }
            let local_h = compute_node_local_structural_hash(lhs, user_ref);
            let bwd_h = lhs_bwd_entries[user_ref.index].hash;
            lhs_keys.insert(ConsumerKey {
                local_hash: local_h,
                bwd_hash: bwd_h,
                operand_index: op_index,
            });
        }
    }

    let mut rhs_keys: std::collections::HashSet<ConsumerKey> = std::collections::HashSet::new();
    for (i, inc) in rhs_mask.iter().enumerate() {
        if !*inc {
            continue;
        }
        for (user_ref, op_index) in users_rhs[i].iter().copied() {
            if rhs_mask[user_ref.index] {
                continue;
            }
            let local_h = compute_node_local_structural_hash(rhs, user_ref);
            let bwd_h = rhs_bwd_entries[user_ref.index].hash;
            rhs_keys.insert(ConsumerKey {
                local_hash: local_h,
                bwd_hash: bwd_h,
                operand_index: op_index,
            });
        }
    }

    let mut common_keys: Vec<ConsumerKey> = lhs_keys.intersection(&rhs_keys).copied().collect();
    // Stable order: by local_hash bytes, then bwd_hash bytes, then operand index
    common_keys.sort_by(|a, b| {
        let ord1 = a.local_hash.as_bytes().cmp(b.local_hash.as_bytes());
        if ord1 != std::cmp::Ordering::Equal {
            return ord1;
        }
        let ord2 = a.bwd_hash.as_bytes().cmp(b.bwd_hash.as_bytes());
        if ord2 != std::cmp::Ordering::Equal {
            return ord2;
        }
        a.operand_index.cmp(&b.operand_index)
    });

    let lhs_sub = build_subgraph_fn_from_unmatched(lhs, &lhs_mask, Some(&common_keys));
    let rhs_sub = build_subgraph_fn_from_unmatched(rhs, &rhs_mask, Some(&common_keys));
    (lhs_sub, rhs_sub)
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

    #[test]
    fn identical_functions_have_no_bwd_discrepancies() {
        let f = parse_fn(
            "fn id(a: bits[1] id=1) -> bits[1] {\n  ret identity.2: bits[1] = identity(a, id=2)\n}",
        );
        let g = parse_fn(
            "fn id(a: bits[1] id=1) -> bits[1] {\n  ret identity.2: bits[1] = identity(a, id=2)\n}",
        );
        let hist = discrepancies_by_depth_bwd(&f, &g);
        assert!(hist.is_empty());
    }

    #[test]
    fn single_op_difference_counts_at_correct_bwd_depth() {
        let f =
            parse_fn("fn n(a: bits[1] id=1) -> bits[1] {\n  ret not.2: bits[1] = not(a, id=2)\n}");
        let g = parse_fn(
            "fn i(a: bits[1] id=1) -> bits[1] {\n  ret identity.2: bits[1] = identity(a, id=2)\n}",
        );
        let hist = discrepancies_by_depth_bwd(&f, &g);
        // Sink (op) differs at depth 0; the param's backward context differs at depth
        // 1.
        assert_eq!(hist.get(&1).copied(), Some(2));
        assert_eq!(hist.len(), 2);
    }

    #[test]
    fn fwd_tuple_operand_order_mismatch_discrepancy() {
        let f = parse_fn(
            "fn t(a: bits[1] id=1, b: bits[1] id=2) -> (bits[1], bits[1]) {
  ret tuple.3: (bits[1], bits[1]) = tuple(a, b, id=3)
}",
        );
        let g = parse_fn(
            "fn t(a: bits[1] id=1, b: bits[1] id=2) -> (bits[1], bits[1]) {
  ret tuple.3: (bits[1], bits[1]) = tuple(b, a, id=3)
}",
        );
        let hist = discrepancies_by_depth(&f, &g);
        // Only the tuple node at forward depth 1 differs; symmetric multiset diff = 2.
        assert_eq!(hist.get(&1).copied(), Some(2));
        assert_eq!(hist.len(), 1);
    }

    #[test]
    fn bwd_param_operand_position_mismatch_discrepancy() {
        let f = parse_fn(
            "fn u(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(a, b, id=3)
}",
        );
        let g = parse_fn(
            "fn u(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(b, a, id=3)
}",
        );
        let hist = discrepancies_by_depth_bwd(&f, &g);
        // The sink 'and' has identical user context (none), so depth 0 matches.
        // Each param's user-operand index differs, so both params differ at backward
        // depth 1. There are 2 params on each side but with distinct hashes;
        // symmetric diff = 4 at depth 1.
        assert_eq!(hist.get(&1).copied(), Some(4));
        assert_eq!(hist.get(&0).copied(), None);
    }

    #[test]
    fn bwd_user_presence_affects_operand_context() {
        let f = parse_fn(
            "fn v(a: bits[1] id=1) -> bits[1] {
  ret identity.2: bits[1] = identity(a, id=2)
}",
        );
        let g = parse_fn(
            "fn v(a: bits[1] id=1) -> bits[1] {
  not.2: bits[1] = not(a, id=2)
  ret identity.3: bits[1] = identity(not.2, id=3)
}",
        );
        let hist = discrepancies_by_depth_bwd(&f, &g);
        // The extra user 'not' on RHS changes the backward context of 'a' (depth 1).
        assert_eq!(hist.get(&1).copied(), Some(2));
    }
}
