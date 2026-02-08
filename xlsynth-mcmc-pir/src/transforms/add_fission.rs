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

    fn literal_is_one(f: &IrFn, r: NodeRef) -> bool {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return false;
        };
        let bits = match v.to_bits() {
            Ok(bits) => bits,
            Err(_) => return false,
        };
        // Width-agnostic check: accept any bits[N] literal that equals integer 1.
        //
        // NOTE: Do not use `IrValue::to_u64()` here; it fails for widths > 64.
        if bits.get_bit_count() == 0 {
            return false;
        }
        let bytes = match bits.to_bytes() {
            Ok(bytes) => bytes,
            Err(_) => return false,
        };
        if bytes.is_empty() {
            return false;
        }
        if bytes[0] != 1 {
            return false;
        }
        bytes.iter().skip(1).all(|b| *b == 0)
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
        if !Self::literal_is_one(f, amount_ref) {
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
                // `bits[0]` is permitted in the IR, but this transform materializes a
                // shift-amount literal `1`, which cannot be represented as `bits[0]`.
                // Skip zero-width adds rather than risking panics in literal creation.
                if Self::bits_width(f, nr) == Some(0) {
                    continue;
                }
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

        if w == 0 {
            // `bits[0]` is permitted elsewhere in the IR, but this transform requires
            // constructing a shift-amount literal `1`. Avoid panicking on
            // `IrBits::make_ubits(0, 1)` by treating this as a non-applicable site.
            return Err("AddFissionTransform: zero-width bits are not supported".to_string());
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn add_fission_rejects_zero_width_add_instead_of_panicking() {
        let ir_text = r#"fn t(a: bits[0] id=1, b: bits[0] id=2) -> bits[0] {
  ret add.3: bits[0] = add(a, b, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let add_ref = f
            .node_refs()
            .into_iter()
            .find(|nr| {
                matches!(
                    f.get_node(*nr).payload,
                    NodePayload::Binop(Binop::Add, _, _)
                )
            })
            .expect("expected add node");

        let t = AddFissionTransform;
        let err = t
            .apply(&mut f, &TransformLocation::Node(add_ref))
            .expect_err("expected zero-width add to be rejected");
        assert!(err.contains("zero-width"));
    }

    #[test]
    fn add_fission_folds_wide_shift_one_literal() {
        // Regression test: for widths > 64, `IrValue::to_u64()` fails, so we
        // must match the shift amount literal via bits inspection.
        let ir_text = r#"fn t(a: bits[128] id=1, b: bits[128] id=2) -> bits[128] {
  xor.3: bits[128] = xor(a, b, id=3)
  and.4: bits[128] = and(a, b, id=4)
  literal.5: bits[128] = literal(value=1, id=5)
  shll.6: bits[128] = shll(and.4, literal.5, id=6)
  ret add.7: bits[128] = add(xor.3, shll.6, id=7)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let add_ref = f
            .node_refs()
            .into_iter()
            .find(|nr| f.get_node(*nr).text_id == 7)
            .expect("expected add node");

        let t = AddFissionTransform;
        t.apply(&mut f, &TransformLocation::Node(add_ref))
            .expect("apply");

        let NodePayload::Binop(Binop::Add, lhs, rhs) = f.get_node(add_ref).payload else {
            panic!("expected add payload after fold");
        };
        assert!(matches!(f.get_node(lhs).payload, NodePayload::GetParam(_)));
        assert!(matches!(f.get_node(rhs).payload, NodePayload::GetParam(_)));
    }
}
