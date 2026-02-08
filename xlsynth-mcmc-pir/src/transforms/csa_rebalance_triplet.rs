// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform that compresses a 3-operand add chain into
/// a carry-save (3:2) form and back.
#[derive(Debug)]
pub struct CsaRebalanceTripletTransform;

impl CsaRebalanceTripletTransform {
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

    fn literal_is_one(f: &IrFn, r: NodeRef) -> bool {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return false;
        };
        v.bits_equals_u64_value(1)
    }

    fn match_add_chain(f: &IrFn, add_ref: NodeRef) -> Option<(NodeRef, NodeRef, NodeRef)> {
        let NodePayload::Binop(Binop::Add, lhs, rhs) = f.get_node(add_ref).payload else {
            return None;
        };
        if let NodePayload::Binop(Binop::Add, a, b) = f.get_node(lhs).payload {
            return Some((a, b, rhs));
        }
        if let NodePayload::Binop(Binop::Add, a, b) = f.get_node(rhs).payload {
            return Some((a, b, lhs));
        }
        None
    }

    fn match_majority(f: &IrFn, w: usize, carry_ref: NodeRef, operands: &[NodeRef]) -> Option<()> {
        let NodePayload::Nary(NaryOp::Or, or_ops) = &f.get_node(carry_ref).payload else {
            return None;
        };
        if or_ops.len() != 3 {
            return None;
        }
        if Self::bits_width(f, carry_ref) != Some(w) {
            return None;
        }
        let mut pairs: Vec<(NodeRef, NodeRef)> = Vec::new();
        for op_ref in or_ops {
            let NodePayload::Nary(NaryOp::And, and_ops) = &f.get_node(*op_ref).payload else {
                return None;
            };
            if and_ops.len() != 2 {
                return None;
            }
            let a = and_ops[0];
            let b = and_ops[1];
            if !operands.contains(&a) || !operands.contains(&b) || a == b {
                return None;
            }
            if Self::bits_width(f, *op_ref) != Some(w) {
                return None;
            }
            let ordered = if a.index <= b.index { (a, b) } else { (b, a) };
            pairs.push(ordered);
        }
        pairs.sort_by_key(|(a, b)| (a.index, b.index));
        pairs.dedup();
        if pairs.len() != 3 {
            return None;
        }
        let mut expected: Vec<(NodeRef, NodeRef)> = Vec::new();
        expected.push(if operands[0].index <= operands[1].index {
            (operands[0], operands[1])
        } else {
            (operands[1], operands[0])
        });
        expected.push(if operands[0].index <= operands[2].index {
            (operands[0], operands[2])
        } else {
            (operands[2], operands[0])
        });
        expected.push(if operands[1].index <= operands[2].index {
            (operands[1], operands[2])
        } else {
            (operands[2], operands[1])
        });
        expected.sort_by_key(|(a, b)| (a.index, b.index));
        if pairs == expected { Some(()) } else { None }
    }

    fn match_csa_add(f: &IrFn, add_ref: NodeRef) -> Option<(NodeRef, NodeRef, NodeRef, NodeRef)> {
        let NodePayload::Binop(Binop::Add, lhs, rhs) = f.get_node(add_ref).payload else {
            return None;
        };
        let w = Self::bits_width(f, add_ref)?;
        let (sum_ref, shll_ref) =
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

        let NodePayload::Nary(NaryOp::Xor, sum_ops) = &f.get_node(sum_ref).payload else {
            return None;
        };
        if sum_ops.len() != 3 {
            return None;
        }
        if Self::bits_width(f, sum_ref) != Some(w) {
            return None;
        }

        let NodePayload::Binop(Binop::Shll, carry_ref, amount_ref) = f.get_node(shll_ref).payload
        else {
            return None;
        };
        if Self::bits_width(f, shll_ref) != Some(w) || Self::bits_width(f, carry_ref) != Some(w) {
            return None;
        }
        if !Self::literal_is_one(f, amount_ref) {
            return None;
        }

        Self::match_majority(f, w, carry_ref, sum_ops)?;
        Some((sum_ops[0], sum_ops[1], sum_ops[2], carry_ref))
    }
}

impl PirTransform for CsaRebalanceTripletTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::CsaRebalanceTriplet
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Add, _, _)) {
                // `bits[0]` is permitted in the IR, but this transform materializes a
                // shift-amount literal `1`, which cannot be represented as `bits[0]`.
                // Skip zero-width adds rather than risking panics in literal creation.
                if Self::bits_width(f, nr) == Some(0) {
                    continue;
                }
                if Self::match_add_chain(f, nr).is_some() || Self::match_csa_add(f, nr).is_some() {
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
                    "CsaRebalanceTripletTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let w = Self::bits_width(f, target_ref)
            .ok_or_else(|| "CsaRebalanceTripletTransform: output must be bits[w]".to_string())?;

        if w == 0 {
            // `bits[0]` is permitted elsewhere in the IR, but this transform requires
            // constructing a shift-amount literal `1`. Avoid panicking on
            // `IrBits::make_ubits(0, 1)` by treating this as a non-applicable site.
            return Err(
                "CsaRebalanceTripletTransform: zero-width bits are not supported".to_string(),
            );
        }

        if let Some((a, b, c, _carry_ref)) = Self::match_csa_add(f, target_ref) {
            if Self::bits_width(f, a) != Some(w)
                || Self::bits_width(f, b) != Some(w)
                || Self::bits_width(f, c) != Some(w)
            {
                return Err("CsaRebalanceTripletTransform: width mismatch".to_string());
            }
            let inner_add = Self::mk_binop_bits_node(f, Binop::Add, w, a, b);
            f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Add, inner_add, c);
            return Ok(());
        }

        let Some((a, b, c)) = Self::match_add_chain(f, target_ref) else {
            return Err("CsaRebalanceTripletTransform: expected add chain".to_string());
        };
        if Self::bits_width(f, a) != Some(w)
            || Self::bits_width(f, b) != Some(w)
            || Self::bits_width(f, c) != Some(w)
        {
            return Err("CsaRebalanceTripletTransform: width mismatch".to_string());
        }

        let sum_ref = Self::mk_nary_bits_node(f, NaryOp::Xor, w, vec![a, b, c]);
        let and_ab = Self::mk_nary_bits_node(f, NaryOp::And, w, vec![a, b]);
        let and_ac = Self::mk_nary_bits_node(f, NaryOp::And, w, vec![a, c]);
        let and_bc = Self::mk_nary_bits_node(f, NaryOp::And, w, vec![b, c]);
        let carry_ref = Self::mk_nary_bits_node(f, NaryOp::Or, w, vec![and_ab, and_ac, and_bc]);
        let shift_one = Self::mk_ubits_literal_node(f, w, 1);
        let shll_ref = Self::mk_binop_bits_node(f, Binop::Shll, w, carry_ref, shift_one);

        f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Add, sum_ref, shll_ref);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn csa_rebalance_triplet_rejects_zero_width_add_chain_instead_of_panicking() {
        let ir_text = r#"fn t(a: bits[0] id=1, b: bits[0] id=2, c: bits[0] id=3) -> bits[0] {
  add.4: bits[0] = add(a, b, id=4)
  ret add.5: bits[0] = add(add.4, c, id=5)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let outer_add_ref = f
            .node_refs()
            .into_iter()
            .find(|nr| f.get_node(*nr).text_id == 5)
            .expect("expected outer add node");

        let t = CsaRebalanceTripletTransform;
        let err = t
            .apply(&mut f, &TransformLocation::Node(outer_add_ref))
            .expect_err("expected zero-width add chain to be rejected");
        assert!(err.contains("zero-width"));
    }

    #[test]
    fn csa_rebalance_triplet_matches_wide_shift_one_literal() {
        // Regression test: for widths > 64, `IrValue::to_u64()` fails, so we
        // must match the shift amount literal via bits inspection.
        let ir_text = r#"fn t(a: bits[128] id=1, b: bits[128] id=2, c: bits[128] id=3) -> bits[128] {
  xor.4: bits[128] = xor(a, b, c, id=4)
  and.5: bits[128] = and(a, b, id=5)
  and.6: bits[128] = and(a, c, id=6)
  and.7: bits[128] = and(b, c, id=7)
  or.8: bits[128] = or(and.5, and.6, and.7, id=8)
  literal.9: bits[128] = literal(value=1, id=9)
  shll.10: bits[128] = shll(or.8, literal.9, id=10)
  ret add.11: bits[128] = add(xor.4, shll.10, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let add_ref = f
            .node_refs()
            .into_iter()
            .find(|nr| f.get_node(*nr).text_id == 11)
            .expect("expected add node");

        let t = CsaRebalanceTripletTransform;
        t.apply(&mut f, &TransformLocation::Node(add_ref))
            .expect("apply");

        // CSA form should expand back to a 2-deep add chain.
        let NodePayload::Binop(Binop::Add, inner_add, c_ref) = f.get_node(add_ref).payload else {
            panic!("expected outer add after CSA expansion");
        };
        let NodePayload::Binop(Binop::Add, a_ref, b_ref) = f.get_node(inner_add).payload else {
            panic!("expected inner add after CSA expansion");
        };
        assert!(matches!(
            f.get_node(a_ref).payload,
            NodePayload::GetParam(_)
        ));
        assert!(matches!(
            f.get_node(b_ref).payload,
            NodePayload::GetParam(_)
        ));
        assert!(matches!(
            f.get_node(c_ref).payload,
            NodePayload::GetParam(_)
        ));
    }
}
