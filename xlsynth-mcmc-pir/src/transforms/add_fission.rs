// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `add(a, b)` (bits[w]) ↔ `add(xor(a, b), shll(and(a, b), 1))`.
#[derive(Debug)]
pub struct AddFissionTransform;

impl AddFissionTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn mk_ubits_literal_node(f: &mut IrFn, w: usize, value: u64) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        let bits = IrBits::make_ubits(w, value).expect("make_ubits");
        let value = IrValue::from_bits(&bits);
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Literal(value),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_nary_bits_node(f: &mut IrFn, op: NaryOp, w: usize, ops: Vec<NodeRef>) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Nary(op, ops),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_binop_bits_node(f: &mut IrFn, op: Binop, w: usize, a: NodeRef, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Binop(op, a, b),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn literal_u64_value(f: &IrFn, r: NodeRef) -> Option<u64> {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return None;
        };
        v.to_u64().ok()
    }

    fn match_xor_and_pair(
        f: &IrFn,
        xor_ref: NodeRef,
        and_ref: NodeRef,
    ) -> Option<(NodeRef, NodeRef)> {
        let NodePayload::Nary(NaryOp::Xor, xor_ops) = &f.get_node(xor_ref).payload else {
            return None;
        };
        if xor_ops.len() != 2 {
            return None;
        }
        let NodePayload::Nary(NaryOp::And, and_ops) = &f.get_node(and_ref).payload else {
            return None;
        };
        if and_ops.len() != 2 {
            return None;
        }
        let (a, b) = (xor_ops[0], xor_ops[1]);
        let and_a = and_ops[0];
        let and_b = and_ops[1];
        if (and_a == a && and_b == b) || (and_a == b && and_b == a) {
            Some((a, b))
        } else {
            None
        }
    }

    fn match_fission_add(
        f: &IrFn,
        add_ref: NodeRef,
    ) -> Option<(NodeRef, NodeRef, NodeRef, NodeRef)> {
        let NodePayload::Binop(Binop::Add, lhs, rhs) = f.get_node(add_ref).payload else {
            return None;
        };
        let w = Self::bits_width(f, add_ref)?;
        let (xor_ref, shll_ref) =
            if matches!(f.get_node(lhs).payload, NodePayload::Nary(NaryOp::Xor, _))
                && matches!(
                    f.get_node(rhs).payload,
                    NodePayload::Binop(Binop::Shll, _, _)
                )
            {
                (lhs, rhs)
            } else if matches!(f.get_node(rhs).payload, NodePayload::Nary(NaryOp::Xor, _))
                && matches!(
                    f.get_node(lhs).payload,
                    NodePayload::Binop(Binop::Shll, _, _)
                )
            {
                (rhs, lhs)
            } else {
                return None;
            };

        if Self::bits_width(f, xor_ref) != Some(w) || Self::bits_width(f, shll_ref) != Some(w) {
            return None;
        }

        let NodePayload::Binop(Binop::Shll, and_ref, amount_ref) = f.get_node(shll_ref).payload
        else {
            return None;
        };
        if Self::bits_width(f, and_ref) != Some(w) {
            return None;
        }
        let Some(amount) = Self::literal_u64_value(f, amount_ref) else {
            return None;
        };
        if amount != 1 {
            return None;
        }

        let (a, b) = Self::match_xor_and_pair(f, xor_ref, and_ref)?;
        Some((a, b, xor_ref, shll_ref))
    }

    fn add_has_add_user(f: &IrFn, add_ref: NodeRef) -> bool {
        let users = compute_users(f);
        let Some(user_refs) = users.get(&add_ref) else {
            return false;
        };
        user_refs.iter().any(|nr| {
            matches!(
                f.get_node(*nr).payload,
                NodePayload::Binop(Binop::Add, _, _)
            )
        })
    }
}

impl PirTransform for AddFissionTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::AddFission
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            if let NodePayload::Binop(Binop::Add, _, _) = f.get_node(nr).payload {
                if Self::match_fission_add(f, nr).is_some() || Self::add_has_add_user(f, nr) {
                    out.push(TransformLocation::Node(nr));
                }
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "AddFissionTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let w = Self::bits_width(f, target_ref)
            .ok_or_else(|| "AddFissionTransform: output must be bits[w]".to_string())?;

        if let Some((a, b, _xor_ref, _shll_ref)) = Self::match_fission_add(f, target_ref) {
            f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Add, a, b);
            return Ok(());
        }

        let NodePayload::Binop(Binop::Add, a, b) = f.get_node(target_ref).payload else {
            return Err("AddFissionTransform: expected add".to_string());
        };
        if Self::bits_width(f, a) != Some(w) || Self::bits_width(f, b) != Some(w) {
            return Err("AddFissionTransform: operands must be bits[w]".to_string());
        }

        let xor_ref = Self::mk_nary_bits_node(f, NaryOp::Xor, w, vec![a, b]);
        let and_ref = Self::mk_nary_bits_node(f, NaryOp::And, w, vec![a, b]);
        let shift_one = Self::mk_ubits_literal_node(f, w, 1);
        let shll_ref = Self::mk_binop_bits_node(f, Binop::Shll, w, and_ref, shift_one);

        f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Add, xor_ref, shll_ref);
        Ok(())
    }
}
