// SPDX-License-Identifier: Apache-2.0

//! Computes structural similarity statistics between two XLS IR functions.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::ir::{Fn, Node, NodePayload, NodeRef, Param, ParamId, Type, node_textual_id};
use crate::ir_utils::{
    compute_users, get_topological, is_valid_identifier_name, operands, remap_payload_with,
    sanitize_text_id_to_identifier_name,
};
use crate::node_hashing::{compute_node_local_structural_hash, compute_node_structural_hash};
use xlsynth::IrValue;

fn zero_ir_value_for_type(ty: &Type) -> IrValue {
    match ty {
        Type::Token => IrValue::make_tuple(&[]),
        Type::Bits(width) => IrValue::make_ubits(*width, 0).unwrap(),
        Type::Tuple(types) => {
            let elems: Vec<IrValue> = types.iter().map(|t| zero_ir_value_for_type(t)).collect();
            IrValue::make_tuple(&elems)
        }
        Type::Array(arr) => {
            let elem = zero_ir_value_for_type(&arr.element_type);
            let elems: Vec<IrValue> = (0..arr.element_count).map(|_| elem.clone()).collect();
            IrValue::make_array(&elems).unwrap()
        }
    }
}

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

/// Computes the node sets that constitute the dual-difference regions on LHS
/// and RHS.
///
/// These are the nodes that remained unmatched when comparing using forward OR
/// backward structural hashes (same criterion used by
/// `extract_dual_difference_subgraphs`).
pub fn compute_dual_difference_regions(lhs: &Fn, rhs: &Fn) -> (HashSet<NodeRef>, HashSet<NodeRef>) {
    assert_eq!(
        lhs.get_type(),
        rhs.get_type(),
        "Function signatures must match for structural comparison",
    );
    let (lhs_unmatched_mask, rhs_unmatched_mask) = compute_dual_unmatched_masks(lhs, rhs);

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
    (lhs_interior, rhs_interior)
}

/// Summarizes the inbound operands and outbound users for a given region of
/// nodes.
///
/// Returns:
/// - A sorted list of unique textual ids for inbound operands (producers
///   outside `region` that feed any node inside `region`).
/// - For each boundary output (nodes in `region` used outside the region, or
///   the return if it lies inside the region), a pair of: (producer textual id,
///   sorted list of textual ids of outside users), ordered by ascending node
///   index to be stable.
fn compute_region_boundary_nodes(
    f: &Fn,
    region: &HashSet<NodeRef>,
    users_map: &HashMap<NodeRef, HashSet<NodeRef>>,
) -> Vec<NodeRef> {
    let mut boundary: Vec<NodeRef> = Vec::new();
    for nr in region.iter() {
        let users = users_map
            .get(nr)
            .map(|s| s.iter().copied().collect::<Vec<NodeRef>>())
            .unwrap_or_default();
        if users.iter().any(|u| !region.contains(u)) {
            boundary.push(*nr);
        }
    }
    if let Some(ret_nr) = f.ret_node_ref {
        if region.contains(&ret_nr) && !boundary.contains(&ret_nr) {
            boundary.push(ret_nr);
        }
    }
    boundary.sort_by_key(|nr| nr.index);
    boundary
}

// Helper: compute outbound boundary users summary for a region.
fn summarize_region_outbound(f: &Fn, region: &HashSet<NodeRef>) -> Vec<(String, Vec<String>)> {
    let users_map = compute_users(f);
    let boundary = compute_region_boundary_nodes(f, region, &users_map);
    let mut per_return: Vec<(String, Vec<String>)> = Vec::new();
    for nr in boundary.into_iter() {
        let producer_txt = node_textual_id(f, nr);
        let mut outside_users: Vec<String> = users_map
            .get(&nr)
            .map(|s| s.iter().copied().filter(|u| !region.contains(u)))
            .into_iter()
            .flatten()
            .map(|u| node_textual_id(f, u))
            .collect();
        outside_users.sort();
        outside_users.dedup();
        per_return.push((producer_txt, outside_users));
    }
    per_return
}

// Helper: collect outbound edges keyed by (consumer backward-hash, operand index)
// for boundary producers.
fn collect_outbound_edges_with_operand_indices(
    f: &Fn,
    region: &std::collections::HashSet<NodeRef>,
) -> BTreeMap<(String, usize), (NodeRef /* producer */, NodeRef /* consumer */)> {
    let users_map = compute_users(f);
    let boundary = compute_region_boundary_nodes(f, region, &users_map);
    let mut edges: BTreeMap<(String, usize), (NodeRef, NodeRef)> = BTreeMap::new();
    // Precompute backward hashes per node for stable consumer identity.
    let (bwd_entries, _bwd_depths) = collect_backward_structural_entries(f);
    let mut bwd_hex_by_index: Vec<String> = vec![String::new(); f.nodes.len()];
    for (i, e) in bwd_entries.into_iter().enumerate() {
        bwd_hex_by_index[i] = e.hash.to_hex().to_string();
    }
    for prod in boundary.into_iter() {
        if let Some(users) = users_map.get(&prod) {
            for user in users.iter().copied().filter(|u| !region.contains(u)) {
                let user_node = f.get_node(user);
                let deps = operands(&user_node.payload);
                for (op_index, dep) in deps.into_iter().enumerate() {
                    if dep.index == prod.index {
                        let key = (bwd_hex_by_index[user.index].clone(), op_index);
                        edges.insert(key, (prod, user));
                    }
                }
            }
        }
    }
    edges
}

// Helper: build textual-id -> NodeRef index for a function.
fn build_textual_id_index(f: &Fn) -> HashMap<String, NodeRef> {
    let mut m: HashMap<String, NodeRef> = HashMap::new();
    for (i, _n) in f.nodes.iter().enumerate() {
        let nr = NodeRef { index: i };
        let t = node_textual_id(f, nr);
        m.insert(t, nr);
    }
    m
}

fn make_slot_param_name(consumer_bwd_hash: &str, op_index: usize) -> String {
    format!("__slot__{}__op{}", consumer_bwd_hash, op_index)
}

// Helper: collect inbound edges (producer outside region to consumer inside)
// with operand indices and deterministically ordered by (consumer textual id,
// operand index).
fn collect_inbound_edges_with_operand_indices(
    f: &Fn,
    region: &HashSet<NodeRef>,
) -> BTreeMap<(String, usize), (NodeRef /* producer */, NodeRef /* consumer */)> {
    let mut edges: BTreeMap<(String, usize), (NodeRef, NodeRef)> = BTreeMap::new();
    for (i, _n) in f.nodes.iter().enumerate() {
        let cons = NodeRef { index: i };
        if !region.contains(&cons) {
            continue;
        }
        let deps = operands(&f.get_node(cons).payload);
        for (op_index, dep) in deps.into_iter().enumerate() {
            if !region.contains(&dep) {
                let key = (node_textual_id(f, cons), op_index);
                edges.insert(key, (dep, cons));
            }
        }
    }
    edges
}

fn to_hex_string(h: &blake3::Hash) -> String {
    h.to_hex().to_string()
}

fn short_hex(s: &str) -> String {
    let n = s.len();
    let take = if n >= 12 { 12 } else { n };
    s[..take].to_string()
}

struct InboundHashInfo {
    // Number of inbound arcs per forward-hash on this side.
    counts: BTreeMap<String /* full hex */, usize>,
    // Representative type per hash on this side.
    types: BTreeMap<String /* full hex */, Type>,
    // Deterministic assignment of each inbound producer NodeRef to a param name.
    // Name format: in_h_<short_hex>_<k>
    assignment: HashMap<usize /* producer index */, String>,
}

fn build_inbound_hash_info(f: &Fn, region: &HashSet<NodeRef>) -> InboundHashInfo {
    // Compute forward structural hashes per node index.
    let (entries, _depths) = collect_structural_entries(f);
    let mut hash_by_index: Vec<String> = vec![String::new(); f.nodes.len()];
    for (i, e) in entries.into_iter().enumerate() {
        hash_by_index[i] = to_hex_string(&e.hash);
    }

    // Collect inbound arcs in deterministic order.
    let inbound = collect_inbound_edges_with_operand_indices(f, region);
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut types: BTreeMap<String, Type> = BTreeMap::new();
    let mut per_hash_cursor: HashMap<String, usize> = HashMap::new();
    let mut assignment: HashMap<usize, String> = HashMap::new();

    for ((_cons_text, _op_index), (producer, _consumer)) in inbound.into_iter() {
        let h = hash_by_index[producer.index].clone();
        let sh = short_hex(&h);
        let cur = per_hash_cursor.entry(h.clone()).or_insert(0);
        let name = format!("fwd_{}_{}", sh, *cur);
        assignment.insert(producer.index, name);
        *cur += 1;
        *counts.entry(h.clone()).or_insert(0) += 1;
        // Record/verify representative type for this hash on this side.
        let ty = f.get_node(producer).ty.clone();
        match types.get(&h) {
            Some(existing) => {
                assert_eq!(
                    existing, &ty,
                    "Inbound hash has inconsistent types on one side"
                );
            }
            None => {
                types.insert(h, ty);
            }
        }
    }

    InboundHashInfo {
        counts,
        types,
        assignment,
    }
}

// Compute union params based on inbound forward-hash multisets across LHS/RHS.
fn compute_union_params_by_hash(
    lhs: &InboundHashInfo,
    rhs: &InboundHashInfo,
) -> Vec<(String, Type)> {
    let mut all_hashes: BTreeSet<String> = lhs.counts.keys().cloned().collect();
    for h in rhs.counts.keys() {
        all_hashes.insert(h.clone());
    }

    let mut result: Vec<(String, Type)> = Vec::new();
    for h in all_hashes.into_iter() {
        let lc = lhs.counts.get(&h).copied().unwrap_or(0);
        let rc = rhs.counts.get(&h).copied().unwrap_or(0);
        let count = if lc > rc { lc } else { rc };
        if count == 0 {
            continue;
        }
        // Determine type and ensure consistency across sides when present.
        let lty = lhs.types.get(&h);
        let rty = rhs.types.get(&h);
        let ty = match (lty, rty) {
            (Some(lt), Some(rt)) => {
                assert_eq!(lt, rt, "Inbound hash has mismatched types across sides");
                lt.clone()
            }
            (Some(lt), None) => lt.clone(),
            (None, Some(rt)) => rt.clone(),
            (None, None) => unreachable!(),
        };
        let sh = short_hex(&h);
        for k in 0..count {
            result.push((format!("fwd_{}_{}", sh, k), ty.clone()));
        }
    }
    result
}

// Helper: build inbound textual-id -> type map for a region.
fn compute_inbound_text_to_type_map(f: &Fn, region: &HashSet<NodeRef>) -> BTreeMap<String, Type> {
    let mut m: BTreeMap<String, Type> = BTreeMap::new();
    for nr in region.iter() {
        let node = f.get_node(*nr);
        for dep in operands(&node.payload).into_iter() {
            if !region.contains(&dep) {
                m.insert(node_textual_id(f, dep), f.get_node(dep).ty.clone());
            }
        }
    }
    m
}

// Helper: compute union inbound textual ids (sorted) and corresponding types.
fn compute_union_params(
    lhs_inbound_map: &BTreeMap<String, Type>,
    rhs_inbound_map: &BTreeMap<String, Type>,
) -> Vec<(String, Type)> {
    // Union textual ids, deterministic order by string.
    let mut union_texts: BTreeSet<String> = lhs_inbound_map.keys().cloned().collect();
    for k in rhs_inbound_map.keys() {
        union_texts.insert(k.clone());
    }

    // Build type map for union, asserting matching types when present on both
    // sides.
    let mut union_pairs: Vec<(String, Type)> = Vec::with_capacity(union_texts.len());
    for t in union_texts.into_iter() {
        let ty = match (lhs_inbound_map.get(&t), rhs_inbound_map.get(&t)) {
            (Some(lt), Some(rt)) => {
                assert_eq!(lt, rt, "Inbound textual id '{}' has mismatched types", t);
                lt.clone()
            }
            (Some(lt), None) => lt.clone(),
            (None, Some(rt)) => rt.clone(),
            (None, None) => unreachable!(),
        };
        union_pairs.push((t, ty));
    }
    union_pairs
}

// Helper: clone a region into a new inner function with fixed params from union
// lists.

// Variant: builds an inner function that returns a tuple matching the union of
// consumer-operand slots across LHS/RHS, using passthrough params when this
// side does not produce a given slot.
fn build_inner_with_union_user_slots(
    f: &Fn,
    region: &HashSet<NodeRef>,
    fname: &str,
    // Base union params (inbound to region).
    union_params: &[(String, Type)],
    // Deterministic slot order: (consumer textual id, operand index)
    slot_order: &[(String, usize)],
    // For this side only, outbound mapping from slot key -> boundary producer NodeRef.
    side_edges: &BTreeMap<(String, usize), (NodeRef, NodeRef)>,
    // For this side only, mapping from inbound external producer index -> assigned param name.
    inbound_assignment: &HashMap<usize, String>,
    // For this side only, mapping from consumer backward-hash -> NodeRef.
    consumer_by_bwd: &HashMap<String, NodeRef>,
) -> Fn {
    // 1) Create inner params: union of union_params + extra_passthrough_params,
    //    dedup’d by name, in deterministic order.
    let mut inner_params: Vec<Param> = Vec::new();
    let mut text_to_inner_param_ref: HashMap<String, NodeRef> = HashMap::new();
    let mut next_param_pos: usize = 1;
    let mut inner_nodes: Vec<Node> = Vec::new();
    let mut next_text_id: usize = {
        let mut max_id = 0usize;
        for n in f.nodes.iter() {
            if n.text_id > max_id {
                max_id = n.text_id;
            }
        }
        max_id.saturating_add(1)
    };

    let mut used_names: HashSet<String> = HashSet::new();
    for (raw_name, ty) in union_params.iter() {
        let pid = ParamId::new(next_param_pos);
        next_param_pos += 1;
        let mut name = if is_valid_identifier_name(raw_name) {
            raw_name.clone()
        } else {
            sanitize_text_id_to_identifier_name(raw_name)
        };
        if used_names.contains(&name) {
            let mut k: usize = 1;
            loop {
                let cand = format!("{}__{}", name, k);
                if !used_names.contains(&cand) {
                    name = cand;
                    break;
                }
                k += 1;
            }
        }
        used_names.insert(name.clone());
        inner_params.push(Param {
            name: name.clone(),
            ty: ty.clone(),
            id: pid,
        });
        // Synthesize a GetParam node for this param
        inner_nodes.push(Node {
            text_id: pid.get_wrapped_id(),
            name: Some(name.clone()),
            ty: ty.clone(),
            payload: NodePayload::GetParam(pid),
            pos: None,
        });
        let param_ref = NodeRef {
            index: inner_nodes.len() - 1,
        };
        text_to_inner_param_ref.insert(raw_name.clone(), param_ref);
    }

    // 2) Topologically clone region nodes mapping external operands to inner params
    let topo = get_topological(f);
    // Precompute forward hashes for error diagnostics.
    let (entries_for_diag, _depths_for_diag) = collect_structural_entries(f);
    let mut fwd_hash_hex_by_index: Vec<String> = vec![String::new(); f.nodes.len()];
    for (i, e) in entries_for_diag.into_iter().enumerate() {
        fwd_hash_hex_by_index[i] = to_hex_string(&e.hash);
    }
    let mut old_to_new: HashMap<usize, NodeRef> = HashMap::new();
    for nr in topo.into_iter() {
        if !region.contains(&nr) {
            continue;
        }
        let old = f.get_node(nr);
        let mapper = |r: NodeRef| -> NodeRef {
            if region.contains(&r) {
                *old_to_new.get(&r.index).expect("mapped internal ref")
            } else {
                // External operand must map to an inbound hash param.
                match inbound_assignment.get(&r.index) {
                    Some(name) => {
                        *text_to_inner_param_ref.get(name).unwrap_or_else(|| {
                            panic!(
                                "missing inbound param node for external operand; func={}, ext_op_text={}, fwd_hash={}",
                                f.name,
                                node_textual_id(f, r),
                                fwd_hash_hex_by_index[r.index]
                            )
                        })
                    }
                    None => panic!(
                        "missing inbound param assignment for external operand; func={}, ext_op_text={}, fwd_hash={}",
                        f.name,
                        node_textual_id(f, r),
                        fwd_hash_hex_by_index[r.index]
                    ),
                }
            }
        };
        let new_payload = remap_payload_with(&old.payload, mapper);
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
        old_to_new.insert(nr.index, new_ref);
    }

    // 3) Assemble return tuple in slot order
    let mut ret_elems: Vec<NodeRef> = Vec::with_capacity(slot_order.len());
    let mut ret_tys: Vec<Type> = Vec::with_capacity(slot_order.len());
    for (consumer_bwd_hash, op_index) in slot_order.iter() {
        if let Some((prod, _user)) = side_edges.get(&(consumer_bwd_hash.clone(), *op_index)) {
            let inner_ref = *old_to_new
                .get(&prod.index)
                .expect("inner ref for boundary producer");
            ret_tys.push(inner_nodes[inner_ref.index].ty.clone());
            ret_elems.push(inner_ref);
        } else {
            // Passthrough: use the original consumer operand value as a param via hash mapping.
            if let Some(consumer_ref) = consumer_by_bwd.get(consumer_bwd_hash) {
                let deps = operands(&f.get_node(*consumer_ref).payload);
                if *op_index < deps.len() {
                    let dep = deps[*op_index];
                    let pname = inbound_assignment.get(&dep.index).unwrap_or_else(|| {
                        panic!(
                            "missing inbound param assignment for passthrough; func={}, consumer_bwd_hash={}, op_index={}, dep_text={}, dep_fwd_hash={}",
                            f.name,
                            consumer_bwd_hash,
                            op_index,
                            node_textual_id(f, dep),
                            fwd_hash_hex_by_index[dep.index]
                        )
                    });
                    let param_ref = *text_to_inner_param_ref.get(pname).unwrap_or_else(|| {
                        panic!(
                            "missing inbound param node for passthrough; func={}, consumer_bwd_hash={}, op_index={}, pname={}",
                            f.name,
                            consumer_bwd_hash,
                            op_index,
                            pname
                        )
                    });
                    ret_tys.push(inner_nodes[param_ref.index].ty.clone());
                    ret_elems.push(param_ref);
                } else {
                    panic!(
                        "slot op index out of range while building passthrough; func={}, consumer_bwd_hash={}, op_index={}, deps_len={}",
                        f.name,
                        consumer_bwd_hash,
                        op_index,
                        deps.len()
                    );
                }
            } else {
                panic!(
                    "consumer missing while building passthrough; func={}, consumer_bwd_hash={}",
                    f.name, consumer_bwd_hash
                );
            }
        }
    }

    let (ret_ref_opt, ret_ty) = if ret_elems.len() == 1 {
        (Some(ret_elems[0]), ret_tys.remove(0))
    } else {
        let tuple_ty = Type::Tuple(ret_tys.into_iter().map(|t| Box::new(t)).collect());
        let tuple_node = Node {
            text_id: next_text_id,
            name: None,
            ty: tuple_ty.clone(),
            payload: NodePayload::Tuple(ret_elems.clone()),
            pos: None,
        };
        inner_nodes.push(tuple_node);
        let rref = Some(NodeRef {
            index: inner_nodes.len() - 1,
        });
        (rref, tuple_ty)
    };

    Fn {
        name: fname.to_string(),
        params: inner_params,
        ret_ty,
        nodes: inner_nodes,
        ret_node_ref: ret_ref_opt,
    }
}
pub struct DualDifferenceExtraction {
    pub lhs_inner: Fn,
    pub rhs_inner: Fn,
    pub lhs_region: HashSet<NodeRef>,
    pub rhs_region: HashSet<NodeRef>,
    pub union_params: Vec<(String, Type)>,
    pub lhs_inbound_texts: Vec<String>,
    pub rhs_inbound_texts: Vec<String>,
    pub lhs_outbound: Vec<(String, Vec<String>)>,
    pub rhs_outbound: Vec<(String, Vec<String>)>,
    pub slot_order: Vec<(String, usize)>,
    pub lhs_inbound_assignment: HashMap<usize, String>,
    pub rhs_inbound_assignment: HashMap<usize, String>,
}

pub fn extract_dual_difference_subgraphs_with_shared_params_and_metadata(
    lhs: &Fn,
    rhs: &Fn,
) -> DualDifferenceExtraction {
    assert_eq!(
        lhs.get_type(),
        rhs.get_type(),
        "Function signatures must match for structural comparison",
    );

    let (lhs_interior, rhs_interior) = compute_dual_difference_regions(lhs, rhs);

    let lhs_inbound_map = compute_inbound_text_to_type_map(lhs, &lhs_interior);
    let rhs_inbound_map = compute_inbound_text_to_type_map(rhs, &rhs_interior);

    // Hash-based inbound identification and union param construction.
    let lhs_inbound_hash = build_inbound_hash_info(lhs, &lhs_interior);
    let rhs_inbound_hash = build_inbound_hash_info(rhs, &rhs_interior);
    let union_params = compute_union_params_by_hash(&lhs_inbound_hash, &rhs_inbound_hash);

    // Compute outbound edges with operand indices on both sides.
    let lhs_edges = collect_outbound_edges_with_operand_indices(lhs, &lhs_interior);
    let rhs_edges = collect_outbound_edges_with_operand_indices(rhs, &rhs_interior);

    // Deterministic slot order = union of keys (no intersection filtering).
    let mut slot_keys: BTreeSet<(String, usize)> = lhs_edges.keys().cloned().collect();
    for k in rhs_edges.keys() {
        slot_keys.insert(k.clone());
    }
    // Build bwd-hash lookup for consumers on each side.
    let (lhs_bwd_entries, _ld) = collect_backward_structural_entries(lhs);
    let (rhs_bwd_entries, _rd) = collect_backward_structural_entries(rhs);
    let mut lhs_consumer_by_bwd: HashMap<String, NodeRef> = HashMap::new();
    for (i, e) in lhs_bwd_entries.into_iter().enumerate() {
        lhs_consumer_by_bwd.insert(e.hash.to_hex().to_string(), NodeRef { index: i });
    }
    let mut rhs_consumer_by_bwd: HashMap<String, NodeRef> = HashMap::new();
    for (i, e) in rhs_bwd_entries.into_iter().enumerate() {
        rhs_consumer_by_bwd.insert(e.hash.to_hex().to_string(), NodeRef { index: i });
    }
    let mut slot_order: Vec<(String, usize)> = slot_keys.into_iter().collect();
    slot_order.sort_by(|a, b| {
        let ord = a.0.as_bytes().cmp(&b.0.as_bytes());
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
        a.1.cmp(&b.1)
    });

    // Note: Do not construct cross-side passthrough params. All inbound values
    // must originate from the common region and are represented as forward-hash
    // params.

    // Build each inner with fixed + passthrough params and the unified slot tuple.
    let lhs_inner = build_inner_with_union_user_slots(
        lhs,
        &lhs_interior,
        &format!("{}_inner", lhs.name),
        &union_params,
        &slot_order,
        &lhs_edges,
        &lhs_inbound_hash.assignment,
        &lhs_consumer_by_bwd,
    );
    let rhs_inner = build_inner_with_union_user_slots(
        rhs,
        &rhs_interior,
        &format!("{}_inner", rhs.name),
        &union_params,
        &slot_order,
        &rhs_edges,
        &rhs_inbound_hash.assignment,
        &rhs_consumer_by_bwd,
    );

    let lhs_inbound_texts: Vec<String> = lhs_inbound_map.keys().cloned().collect();
    let rhs_inbound_texts: Vec<String> = rhs_inbound_map.keys().cloned().collect();

    let lhs_outbound = summarize_region_outbound(lhs, &lhs_interior);
    let rhs_outbound = summarize_region_outbound(rhs, &rhs_interior);

    DualDifferenceExtraction {
        lhs_inner,
        rhs_inner,
        lhs_region: lhs_interior,
        rhs_region: rhs_interior,
        union_params,
        lhs_inbound_texts,
        rhs_inbound_texts,
        lhs_outbound,
        rhs_outbound,
        slot_order,
        lhs_inbound_assignment: lhs_inbound_hash.assignment,
        rhs_inbound_assignment: rhs_inbound_hash.assignment,
    }
}

fn parse_slot_param_name(name: &str) -> Option<(String, usize)> {
    if !name.starts_with("__slot__") {
        return None;
    }
    // Format: __slot__{consumer_bwd_hash}__op{index}
    let rest = &name[8..];
    if let Some(pos) = rest.rfind("__op") {
        let consumer = &rest[..pos];
        let idx_str = &rest[pos + 4..];
        if let Ok(idx) = idx_str.parse::<usize>() {
            return Some((consumer.to_string(), idx));
        }
    }
    None
}

fn build_merged_params_order(
    union_params: &[(String, Type)],
    extra_passthrough_params: &[(String, Type)],
) -> Vec<(String, Type)> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut merged: Vec<(String, Type)> = Vec::new();
    for (t, ty) in union_params.iter() {
        if seen.insert(t.clone()) {
            merged.push((t.clone(), ty.clone()));
        }
    }
    for (t, ty) in extra_passthrough_params.iter() {
        if seen.insert(t.clone()) {
            merged.push((t.clone(), ty.clone()));
        }
    }
    merged
}

fn build_name_to_inbound_producer_map(
    inbound_assignment: &HashMap<usize, String>,
) -> HashMap<String, NodeRef> {
    let mut m: HashMap<String, NodeRef> = HashMap::new();
    for (idx, name) in inbound_assignment.iter() {
        m.insert(name.clone(), NodeRef { index: *idx });
    }
    m
}

fn build_slot_to_index_map(slot_order: &[(String, usize)]) -> BTreeMap<(String, usize), usize> {
    let mut m: BTreeMap<(String, usize), usize> = BTreeMap::new();
    for (i, (t, k)) in slot_order.iter().enumerate() {
        m.insert((t.clone(), *k), i);
    }
    m
}

fn build_prod_to_slot_index_map(
    side_edges: &BTreeMap<(String, usize), (NodeRef, NodeRef)>,
    slot_to_index: &BTreeMap<(String, usize), usize>,
) -> HashMap<usize, usize> {
    let mut m: HashMap<usize, usize> = HashMap::new();
    for (key, (prod, _cons)) in side_edges.iter() {
        if let Some(sidx) = slot_to_index.get(key) {
            m.entry(prod.index)
                .and_modify(|v| {
                    if *sidx < *v {
                        *v = *sidx
                    }
                })
                .or_insert(*sidx);
        }
    }
    m
}

pub fn build_common_wrapper_for_side(
    f: &Fn,
    region: &HashSet<NodeRef>,
    inner_name: &str,
    union_params: &[(String, Type)],
    inbound_assignment: &HashMap<usize, String>,
    slot_order: &[(String, usize)],
    side_edges: &BTreeMap<(String, usize), (NodeRef, NodeRef)>,
    inner_ret_ty: &Type,
) -> Fn {
    // New function basics: same name/signature as original.
    let mut new_params: Vec<Param> = Vec::new();
    let mut new_nodes: Vec<Node> = Vec::new();
    let mut old_to_new: HashMap<usize, NodeRef> = HashMap::new();

    // Start text_id after max existing.
    let mut next_text_id: usize = {
        let mut max_id = 0usize;
        for n in f.nodes.iter() {
            if n.text_id > max_id {
                max_id = n.text_id;
            }
        }
        max_id.saturating_add(1)
    };

    // Clone params and synthesize GetParam nodes.
    let mut next_param_pos: usize = 1;
    for p in f.params.iter() {
        let pid = ParamId::new(next_param_pos);
        next_param_pos += 1;
        new_params.push(Param {
            name: p.name.clone(),
            ty: p.ty.clone(),
            id: pid,
        });
        new_nodes.push(Node {
            text_id: pid.get_wrapped_id(),
            name: Some(p.name.clone()),
            ty: p.ty.clone(),
            payload: NodePayload::GetParam(pid),
            pos: None,
        });
        old_to_new.insert(
            f.nodes
                .iter()
                .position(|n| n.name.as_ref() == Some(&p.name))
                .expect("original GetParam node present"),
            NodeRef {
                index: new_nodes.len() - 1,
            },
        );
        next_text_id += 1;
    }

    let order = get_topological(f);

    // Pass 1: clone outside-of-region nodes that do not depend on region.
    for nr in order.iter().copied() {
        if region.contains(&nr) {
            continue;
        }
        let node = f.get_node(nr);
        let deps = operands(&node.payload);
        if deps.iter().any(|d| region.contains(d)) {
            continue;
        }
        let mapper = |r: NodeRef| -> NodeRef {
            *old_to_new
                .get(&r.index)
                .expect("dependency should have been cloned")
        };
        let new_payload = remap_payload_with(&node.payload, mapper);
        new_nodes.push(Node {
            text_id: next_text_id,
            name: node.name.clone(),
            ty: node.ty.clone(),
            payload: new_payload,
            pos: node.pos.clone(),
        });
        old_to_new.insert(
            nr.index,
            NodeRef {
                index: new_nodes.len() - 1,
            },
        );
        next_text_id += 1;
    }

    // Build invoke operands in original space, then map via old_to_new.
    let merged = build_merged_params_order(union_params, &[]);
    let name_to_prod = build_name_to_inbound_producer_map(inbound_assignment);
    let text_index = build_textual_id_index(f);
    let slot_to_index = build_slot_to_index_map(slot_order);
    let resolve_original = |name: &str| -> Option<NodeRef> {
        if let Some(prod) = name_to_prod.get(name) {
            return Some(*prod);
        }
        if let Some((cons_text, op_index)) = parse_slot_param_name(name) {
            if let Some(cons_ref) = text_index.get(&cons_text) {
                let deps = operands(&f.get_node(*cons_ref).payload);
                if op_index < deps.len() {
                    return Some(deps[op_index]);
                }
            }
            return None;
        }
        // Fall back to textual id mapping.
        text_index.get(name).copied()
    };
    let mut invoke_operands_new: Vec<NodeRef> = Vec::new();
    for (n, _ty) in merged.iter() {
        let orig = resolve_original(n).expect("resolve original producer for invoke operand");
        let mapped = *old_to_new
            .get(&orig.index)
            .expect("invoke operand producer should be cloned in pass 1");
        invoke_operands_new.push(mapped);
    }

    // Create invoke node.
    let invoke_node = Node {
        text_id: next_text_id,
        name: None,
        ty: inner_ret_ty.clone(),
        payload: NodePayload::Invoke {
            to_apply: inner_name.to_string(),
            operands: invoke_operands_new,
        },
        pos: None,
    };
    new_nodes.push(invoke_node);
    let invoke_ref = NodeRef {
        index: new_nodes.len() - 1,
    };
    next_text_id += 1;

    // Create tuple extractors per slot index (or use invoke directly if single
    // value).
    let mut tuple_elems: Vec<NodeRef> = Vec::new();
    match inner_ret_ty {
        Type::Tuple(elem_tys) => {
            for (i, ety) in elem_tys.iter().enumerate() {
                let ti = Node {
                    text_id: next_text_id,
                    name: None,
                    ty: *ety.clone(),
                    payload: NodePayload::TupleIndex {
                        tuple: invoke_ref,
                        index: i,
                    },
                    pos: None,
                };
                new_nodes.push(ti);
                tuple_elems.push(NodeRef {
                    index: new_nodes.len() - 1,
                });
                next_text_id += 1;
            }
        }
        _ => {
            tuple_elems.push(invoke_ref);
        }
    }

    // Build producer->slot index map to choose a representative tuple elem for each
    // internal producer.
    let prod_to_slot = build_prod_to_slot_index_map(side_edges, &slot_to_index);

    // Pass 2: clone remaining outside-of-region nodes, mapping region deps to tuple
    // elems.
    for nr in order.into_iter() {
        if region.contains(&nr) {
            continue;
        }
        if old_to_new.contains_key(&nr.index) {
            continue;
        }
        let node = f.get_node(nr);
        let mapper = |r: NodeRef| -> NodeRef {
            if region.contains(&r) {
                let sidx = *prod_to_slot
                    .get(&r.index)
                    .expect("slot index for producer in region");
                return tuple_elems[sidx];
            }
            *old_to_new.get(&r.index).expect("mapped earlier dependency")
        };
        let new_payload = remap_payload_with(&node.payload, mapper);
        new_nodes.push(Node {
            text_id: next_text_id,
            name: node.name.clone(),
            ty: node.ty.clone(),
            payload: new_payload,
            pos: node.pos.clone(),
        });
        old_to_new.insert(
            nr.index,
            NodeRef {
                index: new_nodes.len() - 1,
            },
        );
        next_text_id += 1;
    }

    // Map return ref.
    let new_ret = f
        .ret_node_ref
        .map(|r| *old_to_new.get(&r.index).expect("mapped return"));

    Fn {
        name: f.name.clone(),
        params: new_params,
        ret_ty: f.ret_ty.clone(),
        nodes: new_nodes,
        ret_node_ref: new_ret,
    }
}

pub fn build_common_wrapper_lhs(lhs: &Fn, meta: &DualDifferenceExtraction) -> Fn {
    build_common_wrapper_for_side(
        lhs,
        &meta.lhs_region,
        &meta.lhs_inner.name,
        &meta.union_params,
        &meta.lhs_inbound_assignment,
        &meta.slot_order,
        &collect_outbound_edges_with_operand_indices(lhs, &meta.lhs_region),
        &meta.lhs_inner.ret_ty,
    )
}

pub fn build_common_wrapper_rhs(rhs: &Fn, meta: &DualDifferenceExtraction) -> Fn {
    build_common_wrapper_for_side(
        rhs,
        &meta.rhs_region,
        &meta.rhs_inner.name,
        &meta.union_params,
        &meta.rhs_inbound_assignment,
        &meta.slot_order,
        &collect_outbound_edges_with_operand_indices(rhs, &meta.rhs_region),
        &meta.rhs_inner.ret_ty,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::PackageMember;
    use crate::ir_parser::Parser;

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
