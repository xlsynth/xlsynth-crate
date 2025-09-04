// SPDX-License-Identifier: Apache-2.0

//! Computes structural similarity statistics between two XLS IR functions.

use std::collections::HashMap;
use std::collections::HashSet;

use crate::xls_ir::ir::{Fn, NodePayload, NodeRef};
use crate::xls_ir::ir::{Package, PackageMember};
use crate::xls_ir::ir_outline::outline;
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
            NodePayload::ArraySlice { array, start, .. } => {
                child_hashes.push(hashes[array.index]);
                child_depths.push(depths[array.index]);
                child_hashes.push(hashes[start.index]);
                child_depths.push(depths[start.index]);
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

/// Extracts inner functions for the matched "interior" (nodes that are
/// equivalent by forward OR backward structural equivalence) using the
/// outlining transformation. Returns the inner functions from each side.
pub fn extract_dual_difference_subgraphs(lhs: &Fn, rhs: &Fn) -> (Fn, Fn) {
    assert_eq!(
        lhs.get_type(),
        rhs.get_type(),
        "Function signatures must match for structural comparison",
    );
    let (lhs_unmatched_mask, rhs_unmatched_mask) = compute_dual_unmatched_masks(lhs, rhs);

    // Select the unmatched nodes as the interior difference to outline.
    let mut lhs_interior: HashSet<NodeRef> = HashSet::new();
    for i in 0..lhs.nodes.len() {
        if i < lhs_unmatched_mask.len() && lhs_unmatched_mask[i] {
            lhs_interior.insert(NodeRef { index: i });
        }
    }
    let mut rhs_interior: HashSet<NodeRef> = HashSet::new();
    for i in 0..rhs.nodes.len() {
        if i < rhs_unmatched_mask.len() && rhs_unmatched_mask[i] {
            rhs_interior.insert(NodeRef { index: i });
        }
    }

    // Outline the interior on each side and return the resulting inner function.
    let mut lhs_pkg = Package {
        name: "outline_lhs".to_string(),
        file_table: crate::xls_ir::ir::FileTable::new(),
        members: vec![PackageMember::Function(lhs.clone())],
        top_name: None,
    };
    let mut rhs_pkg = Package {
        name: "outline_rhs".to_string(),
        file_table: crate::xls_ir::ir::FileTable::new(),
        members: vec![PackageMember::Function(rhs.clone())],
        top_name: None,
    };

    let lhs_names = (format!("{}_out", lhs.name), format!("{}_inner", lhs.name));
    let rhs_names = (format!("{}_out", rhs.name), format!("{}_inner", rhs.name));

    let lhs_res = outline(lhs, &lhs_interior, &lhs_names.0, &lhs_names.1, &mut lhs_pkg);
    let rhs_res = outline(rhs, &rhs_interior, &rhs_names.0, &rhs_names.1, &mut rhs_pkg);

    (lhs_res.inner, rhs_res.inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xls_ir::ir::PackageMember;
    use crate::xls_ir::ir_parser::Parser;

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
