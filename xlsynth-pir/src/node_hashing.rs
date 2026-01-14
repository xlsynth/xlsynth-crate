// SPDX-License-Identifier: Apache-2.0

//! Helpers for computing structural hashes of XLS IR nodes.

use crate::ir::{self, Fn, NodePayload, NodeRef, ParamId, Type};
use crate::ir_utils::{get_topological, operands};

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
        NodePayload::ExtCarryOut { .. } => {}
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

pub fn compute_node_structural_hash(
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

pub fn compute_node_local_structural_hash(f: &Fn, node_ref: NodeRef) -> FwdHash {
    // Hash operator tag + type + payload attributes only; ignore children.
    let node = f.get_node(node_ref);
    let mut hasher = blake3::Hasher::new();
    update_hash_str(&mut hasher, node.payload.get_operator());
    update_hash_type(&mut hasher, &node.ty);
    hash_payload_attributes(f, &node.payload, &mut hasher);
    FwdHash(hasher.finalize())
}

/// Returns a human-readable node signature that reflects the inputs that affect
/// structural hashing (operator, types, and hashing-relevant payload
/// attributes).
///
/// This is intended for diagnostics / reporting (e.g. fuzzing and corpus
/// analysis), and should stay aligned with what `hash_payload_attributes()`
/// uses.
pub fn node_structural_signature_string(f: &Fn, node_ref: NodeRef) -> String {
    let node = f.get_node(node_ref);
    let operand_tys: Vec<String> = operands(&node.payload)
        .iter()
        .map(|o| f.get_node(*o).ty.to_string())
        .collect();

    let mut attrs: Vec<String> = Vec::new();
    match &node.payload {
        NodePayload::Nil => {}
        NodePayload::GetParam(param_id) => {
            let ordinal = get_param_ordinal(f, *param_id) + 1;
            attrs.push(format!("param_ordinal={ordinal}"));
        }
        NodePayload::Tuple(nodes) => attrs.push(format!("len={}", nodes.len())),
        NodePayload::Array(nodes) => attrs.push(format!("len={}", nodes.len())),
        NodePayload::TupleIndex { index, .. } => attrs.push(format!("index={index}")),
        NodePayload::Literal(value) => attrs.push(format!("value={}", value.to_string())),
        NodePayload::SignExt { new_bit_count, .. } | NodePayload::ZeroExt { new_bit_count, .. } => {
            attrs.push(format!("new_bit_count={new_bit_count}"));
        }
        NodePayload::ArrayUpdate {
            assumed_in_bounds, ..
        }
        | NodePayload::ArrayIndex {
            assumed_in_bounds, ..
        } => attrs.push(format!("assumed_in_bounds={assumed_in_bounds}")),
        NodePayload::ArraySlice { width, .. } => attrs.push(format!("width={width}")),
        NodePayload::DynamicBitSlice { width, .. } => attrs.push(format!("width={width}")),
        NodePayload::BitSlice { start, width, .. } => {
            attrs.push(format!("start={start}"));
            attrs.push(format!("width={width}"));
        }
        NodePayload::OneHot { lsb_prio, .. } => attrs.push(format!("lsb_prio={lsb_prio}")),
        NodePayload::Decode { width, .. } => attrs.push(format!("width={width}")),
        NodePayload::Nary(_, nodes) => attrs.push(format!("len={}", nodes.len())),
        NodePayload::Invoke { to_apply, operands } => {
            attrs.push(format!("to_apply={to_apply}"));
            attrs.push(format!("len={}", operands.len()));
        }
        NodePayload::PrioritySel { cases, default, .. } => {
            attrs.push(format!("len={}", cases.len()));
            attrs.push(format!("has_default={}", default.is_some()));
        }
        NodePayload::OneHotSel { cases, .. } => attrs.push(format!("len={}", cases.len())),
        NodePayload::Sel { cases, default, .. } => {
            attrs.push(format!("len={}", cases.len()));
            attrs.push(format!("has_default={}", default.is_some()));
        }
        NodePayload::CountedFor {
            trip_count,
            stride,
            body,
            invariant_args,
            ..
        } => {
            attrs.push(format!("trip_count={trip_count}"));
            attrs.push(format!("stride={stride}"));
            attrs.push(format!("body={body}"));
            attrs.push(format!("invariant_args_len={}", invariant_args.len()));
        }
        _ => {}
    }

    let mut args: Vec<String> = Vec::new();
    args.extend(operand_tys);
    args.extend(attrs);
    let args_str = args.join(", ");
    format!(
        "{}({}) -> {}",
        node.payload.get_operator(),
        args_str,
        node.ty.to_string()
    )
}

/// Computes a node's backward structural hash by combining its local
/// structural hash with its users' backward hashes and the operand indices
/// at which this node appears. The user pairs are sorted by (hash bytes,
/// operand index) to produce a stable characterization.
pub fn compute_node_backward_structural_hash(
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

/// Returns true if two functions are structurally isomorphic between the
/// parameters and the return node. That is, the functions have the same
/// signature and the return value is equivalent in the CSE (common
/// subexpression elimination) sense.
pub fn functions_structurally_equivalent(lhs: &Fn, rhs: &Fn) -> bool {
    if lhs.get_type() != rhs.get_type() {
        return false;
    }
    if lhs.params.len() != rhs.params.len() {
        return false;
    }
    if lhs.ret_node_ref.is_none() || rhs.ret_node_ref.is_none() {
        return false;
    }

    fn compute_forward_hashes(f: &Fn) -> Vec<FwdHash> {
        let n = f.nodes.len();
        if n == 0 {
            return vec![];
        }
        let order = get_topological(f);
        let mut out: Vec<Option<FwdHash>> = vec![None; n];
        for nr in order {
            let deps = operands(&f.get_node(nr).payload);
            let child_hashes: Vec<FwdHash> = deps
                .into_iter()
                .map(|c| out[c.index].expect("child hash must be computed first"))
                .collect();
            let h = compute_node_structural_hash(f, nr, &child_hashes);
            out[nr.index] = Some(h);
        }
        out.into_iter()
            .map(|o| o.expect("hash must be computed for all nodes"))
            .collect()
    }

    let lhs_fwd = compute_forward_hashes(lhs);
    let rhs_fwd = compute_forward_hashes(rhs);
    lhs_fwd[lhs.ret_node_ref.unwrap().index] == rhs_fwd[rhs.ret_node_ref.unwrap().index]
}

/// Computes depth-limited forward structural hashes for all nodes in `f`.
///
/// Depth definition:
/// - depth=0: local structural hash only (operator + type + payload
///   attributes).
/// - depth>0: hash(local + ordered child hashes at depth-1), where child order
///   is the operand order in the PIR node payload.
///
/// This is useful for computing bounded “neighborhood signatures” (e.g.
/// depth=2).
pub fn compute_depth_limited_forward_hashes(f: &Fn, depth: usize) -> Vec<FwdHash> {
    let n = f.nodes.len();
    if n == 0 {
        return vec![];
    }

    let mut hashes: Vec<FwdHash> = (0..n)
        .map(|index| compute_node_local_structural_hash(f, NodeRef { index }))
        .collect();
    if depth == 0 {
        return hashes;
    }

    let order = get_topological(f);
    for _ in 0..depth {
        let prev = hashes;
        let mut next: Vec<Option<FwdHash>> = vec![None; n];
        for nr in order.iter().copied() {
            let deps = operands(&f.get_node(nr).payload);
            let mut child_hashes: Vec<FwdHash> = Vec::with_capacity(deps.len());
            for c in deps.into_iter() {
                child_hashes.push(prev[c.index]);
            }
            next[nr.index] = Some(compute_node_structural_hash(f, nr, &child_hashes));
        }
        hashes = next
            .into_iter()
            .map(|o| o.expect("hash must be computed for all nodes"))
            .collect();
    }
    hashes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;

    fn parse_top_fn(ir_pkg_text: &str) -> Fn {
        let mut p = Parser::new(ir_pkg_text);
        let pkg = p.parse_and_validate_package().unwrap();
        pkg.get_top_fn().unwrap().clone()
    }

    #[test]
    fn isomorphic_functions_return_true() {
        let lhs = parse_top_fn(
            r#"package p
            top fn f(a: bits[8]) -> bits[8] {
              ret add.3: bits[8] = add(a, a)
            }
            "#,
        );
        let rhs = parse_top_fn(
            r#"package q
            top fn g(a: bits[8]) -> bits[8] {
              ret add.3: bits[8] = add(a, a)
            }
            "#,
        );
        assert!(functions_structurally_equivalent(&lhs, &rhs));
    }

    #[test]
    fn extra_parameter_is_notequivalent() {
        let lhs = parse_top_fn(
            r#"package p
            top fn f(a: bits[8]) -> bits[8] {
              ret identity.2: bits[8] = identity(a)
            }
            "#,
        );
        let rhs = parse_top_fn(
            r#"package q
            top fn g(a: bits[8], b: bits[8]) -> bits[8] {
              ret identity.3: bits[8] = identity(a)
            }
            "#,
        );
        assert!(!functions_structurally_equivalent(&lhs, &rhs));
    }

    #[test]
    fn different_graphs_but_dead_uses_ignored_are_equivalent() {
        let lhs = parse_top_fn(
            r#"package p
            top fn f(a: bits[8]) -> bits[8] {
              ret identity.2: bits[8] = identity(a)
            }
            "#,
        );
        let rhs = parse_top_fn(
            r#"package q
            top fn g(a: bits[8]) -> bits[8] {
              d: bits[8] = literal(value=7, id=2)
              u: bits[8] = add(a, d, id=3)
              ret x: bits[8] = identity(a, id=4)
              q: bits[8] = identity(x, id=5)
            }
            "#,
        );
        assert!(functions_structurally_equivalent(&lhs, &rhs));
    }
}
