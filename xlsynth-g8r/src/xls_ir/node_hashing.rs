// SPDX-License-Identifier: Apache-2.0

//! Helpers for computing structural hashes of XLS IR nodes.

use crate::xls_ir::ir::{Fn, NodePayload, NodeRef, ParamId, Type};

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
        NodePayload::ArraySlice { .. } => "array_slice",
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

pub(crate) fn compute_node_structural_hash(
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

pub(crate) fn compute_node_local_structural_hash(f: &Fn, node_ref: NodeRef) -> blake3::Hash {
    // Hash operator tag + type + payload attributes only; ignore children.
    let node = f.get_node(node_ref);
    let mut hasher = blake3::Hasher::new();
    update_hash_str(&mut hasher, payload_tag(&node.payload));
    update_hash_type(&mut hasher, &node.ty);
    hash_payload_attributes(f, &node.payload, &mut hasher);
    hasher.finalize()
}

/// Returns a short hexadecimal prefix of the local structural hash for
/// `node_ref` in `f`. This avoids exposing the hash type to downstream crates.
pub fn local_structural_hash_hex_prefix(f: &Fn, node_ref: NodeRef, nbytes: usize) -> String {
    let h = compute_node_local_structural_hash(f, node_ref);
    let bytes = h.as_bytes();
    let take = core::cmp::min(nbytes, bytes.len());
    let mut s = String::with_capacity(take * 2);
    for b in bytes.iter().take(take) {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
