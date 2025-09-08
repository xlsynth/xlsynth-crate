// SPDX-License-Identifier: Apache-2.0

//! Helpers for computing structural hashes of XLS IR nodes.

use crate::ir::{self, Fn, NodePayload, NodeRef, ParamId, Type};

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub struct FwdHash(pub blake3::Hash);

impl FwdHash {
    pub fn as_bytes(&self) -> &[u8; 32] {
        self.0.as_bytes()
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub struct BwdHash(pub blake3::Hash);

impl BwdHash {
    pub fn as_bytes(&self) -> &[u8; 32] {
        self.0.as_bytes()
    }
}

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

pub(crate) fn get_param_ordinal(f: &Fn, param_id: ParamId) -> usize {
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
        NodePayload::Binop(op, _, _) => update_hash_str(hasher, ir::binop_to_operator(*op)),
        NodePayload::Unop(op, _) => update_hash_str(hasher, ir::unop_to_operator(*op)),
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
        NodePayload::ArraySlice { width, .. } => update_hash_u64(hasher, *width as u64),
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
            update_hash_str(hasher, ir::nary_op_to_operator(*op));
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

pub(crate) fn compute_node_structural_hash(
    f: &Fn,
    node_ref: NodeRef,
    child_hashes: &[FwdHash],
) -> FwdHash {
    let node = f.get_node(node_ref);
    let mut hasher = blake3::Hasher::new();
    update_hash_str(&mut hasher, node.payload.get_operator());
    update_hash_type(&mut hasher, &node.ty);
    hash_payload_attributes(f, &node.payload, &mut hasher);
    for ch in child_hashes.iter() {
        hasher.update(ch.as_bytes());
    }
    FwdHash(hasher.finalize())
}

pub(crate) fn compute_node_local_structural_hash(f: &Fn, node_ref: NodeRef) -> FwdHash {
    // Hash operator tag + type + payload attributes only; ignore children.
    let node = f.get_node(node_ref);
    let mut hasher = blake3::Hasher::new();
    update_hash_str(&mut hasher, node.payload.get_operator());
    update_hash_type(&mut hasher, &node.ty);
    hash_payload_attributes(f, &node.payload, &mut hasher);
    FwdHash(hasher.finalize())
}

/// Computes a node's backward structural hash by combining its local
/// structural hash with its users' backward hashes and the operand indices
/// at which this node appears. The user pairs are sorted by (hash bytes,
/// operand index) to produce a stable characterization.
pub(crate) fn compute_node_backward_structural_hash(
    f: &Fn,
    node_ref: NodeRef,
    user_pairs: &[(BwdHash, usize)],
) -> BwdHash {
    let mut pairs: Vec<(BwdHash, usize)> = user_pairs.to_vec();
    pairs.sort_by(|a, b| {
        let ord = a.0.as_bytes().cmp(b.0.as_bytes());
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
        a.1.cmp(&b.1)
    });

    let mut hasher = blake3::Hasher::new();
    let local = compute_node_local_structural_hash(f, node_ref);
    hasher.update(local.as_bytes());
    hasher.update(&(u64::try_from(pairs.len()).unwrap_or(0)).to_le_bytes());
    for (uh, idx) in pairs.into_iter() {
        hasher.update(uh.as_bytes());
        hasher.update(&(u64::try_from(idx).unwrap_or(0)).to_le_bytes());
    }
    BwdHash(hasher.finalize())
}
