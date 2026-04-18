// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `add(zero_ext(a,2), zero_ext(b,2)) ↔ concat(and(a,b), xor(a,b))`
#[derive(Debug)]
pub struct OneBitCarryCountTransform;

impl OneBitCarryCountTransform {
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

    fn is_u1(f: &IrFn, r: NodeRef) -> bool {
        Self::bits_width(f, r) == Some(1)
    }

    fn is_zero_literal_node(f: &IrFn, r: NodeRef, w: usize) -> bool {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return false;
        };
        let expected = IrValue::from_bits(&IrBits::make_ubits(w, 0).expect("make_ubits"));
        *v == expected
    }

    fn zext_u1_to_u2_parts(f: &IrFn, r: NodeRef) -> Option<NodeRef> {
        match &f.get_node(r).payload {
            NodePayload::ZeroExt { arg, new_bit_count } => {
                if *new_bit_count == 2 && Self::bits_width(f, r) == Some(2) && Self::is_u1(f, *arg)
                {
                    Some(*arg)
                } else {
                    None
                }
            }
            NodePayload::Nary(NaryOp::Concat, ops) if ops.len() == 2 => {
                let hi = ops[0];
                let lo = ops[1];
                if Self::bits_width(f, r) == Some(2)
                    && Self::bits_width(f, hi) == Some(1)
                    && Self::is_u1(f, lo)
                    && Self::is_zero_literal_node(f, hi, 1)
                {
                    Some(lo)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn match_and_xor_pair(
        f: &IrFn,
        and_ref: NodeRef,
        xor_ref: NodeRef,
    ) -> Option<(NodeRef, NodeRef)> {
        let NodePayload::Nary(NaryOp::And, and_ops) = &f.get_node(and_ref).payload else {
            return None;
        };
        let NodePayload::Nary(NaryOp::Xor, xor_ops) = &f.get_node(xor_ref).payload else {
            return None;
        };
        if and_ops.len() != 2 || xor_ops.len() != 2 {
            return None;
        }
        let (a, b) = (and_ops[0], and_ops[1]);
        if ((xor_ops[0] == a && xor_ops[1] == b) || (xor_ops[0] == b && xor_ops[1] == a))
            && Self::is_u1(f, a)
            && Self::is_u1(f, b)
            && Self::bits_width(f, and_ref) == Some(1)
            && Self::bits_width(f, xor_ref) == Some(1)
        {
            Some((a, b))
        } else {
            None
        }
    }

    fn mk_zero_ext_u1_to_u2_node(f: &mut IrFn, arg: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(2),
            payload: NodePayload::ZeroExt {
                arg,
                new_bit_count: 2,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_nary_u1_node(f: &mut IrFn, op: NaryOp, a: NodeRef, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(1),
            payload: NodePayload::Nary(op, vec![a, b]),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for OneBitCarryCountTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::OneBitCarryCount
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let always_equivalent = true;
        let mut out = Vec::<TransformCandidate>::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Binop(Binop::Add, lhs, rhs) => {
                    if Self::bits_width(f, nr) != Some(2) {
                        continue;
                    }
                    if Self::zext_u1_to_u2_parts(f, *lhs).is_some()
                        && Self::zext_u1_to_u2_parts(f, *rhs).is_some()
                    {
                        out.push(TransformCandidate {
                            location: TransformLocation::Node(nr),
                            always_equivalent,
                        });
                    }
                }
                NodePayload::Nary(NaryOp::Concat, ops) if ops.len() == 2 => {
                    if Self::bits_width(f, nr) == Some(2)
                        && Self::match_and_xor_pair(f, ops[0], ops[1]).is_some()
                    {
                        out.push(TransformCandidate {
                            location: TransformLocation::Node(nr),
                            always_equivalent,
                        });
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "OneBitCarryCountTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        if Self::bits_width(f, target_ref) != Some(2) {
            return Err("OneBitCarryCountTransform: output must be bits[2]".to_string());
        }

        let payload = f.get_node(target_ref).payload.clone();
        match payload {
            NodePayload::Binop(Binop::Add, lhs, rhs) => {
                let a = Self::zext_u1_to_u2_parts(f, lhs).ok_or_else(|| {
                    "OneBitCarryCountTransform: expected lhs zero_ext bits[1] to bits[2]"
                        .to_string()
                })?;
                let b = Self::zext_u1_to_u2_parts(f, rhs).ok_or_else(|| {
                    "OneBitCarryCountTransform: expected rhs zero_ext bits[1] to bits[2]"
                        .to_string()
                })?;
                let carry = Self::mk_nary_u1_node(f, NaryOp::And, a, b);
                let sum = Self::mk_nary_u1_node(f, NaryOp::Xor, a, b);
                f.get_node_mut(target_ref).payload =
                    NodePayload::Nary(NaryOp::Concat, vec![carry, sum]);
                Ok(())
            }
            NodePayload::Nary(NaryOp::Concat, ops) => {
                if ops.len() != 2 {
                    return Err("OneBitCarryCountTransform: expected 2-operand concat".to_string());
                }
                let (a, b) = Self::match_and_xor_pair(f, ops[0], ops[1]).ok_or_else(|| {
                    "OneBitCarryCountTransform: expected concat(and(a,b), xor(a,b))".to_string()
                })?;
                let a_ext = Self::mk_zero_ext_u1_to_u2_node(f, a);
                let b_ext = Self::mk_zero_ext_u1_to_u2_node(f, b);
                f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Add, a_ext, b_ext);
                Ok(())
            }
            _ => Err("OneBitCarryCountTransform: expected add(...) or concat(...)".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    fn find_payload(f: &IrFn, pred: impl Fn(&NodePayload) -> bool) -> NodeRef {
        f.node_refs()
            .into_iter()
            .find(|nr| pred(&f.get_node(*nr).payload))
            .expect("expected node")
    }

    #[test]
    fn one_bit_carry_count_expands_add_of_zext_u1s() {
        let ir_text = r#"fn t(a: bits[1] id=1, b: bits[1] id=2) -> bits[2] {
  zero_ext.10: bits[2] = zero_ext(a, new_bit_count=2, id=10)
  zero_ext.11: bits[2] = zero_ext(b, new_bit_count=2, id=11)
  ret add.20: bits[2] = add(zero_ext.10, zero_ext.11, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let add_ref = find_payload(&f, |p| matches!(p, NodePayload::Binop(Binop::Add, _, _)));

        let mut t = OneBitCarryCountTransform;
        assert_eq!(t.find_candidates(&f).len(), 1);
        t.apply(&mut f, &TransformLocation::Node(add_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(add_ref).payload,
            NodePayload::Nary(NaryOp::Concat, _)
        ));
    }

    #[test]
    fn one_bit_carry_count_folds_concat_and_xor() {
        let ir_text = r#"fn t(a: bits[1] id=1, b: bits[1] id=2) -> bits[2] {
  and.10: bits[1] = and(a, b, id=10)
  xor.11: bits[1] = xor(b, a, id=11)
  ret concat.20: bits[2] = concat(and.10, xor.11, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let concat_ref = find_payload(&f, |p| matches!(p, NodePayload::Nary(NaryOp::Concat, _)));

        let t = OneBitCarryCountTransform;
        t.apply(&mut f, &TransformLocation::Node(concat_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(concat_ref).payload,
            NodePayload::Binop(Binop::Add, _, _)
        ));
    }

    #[test]
    fn one_bit_carry_count_accepts_concat_zero_ext_inputs() {
        let ir_text = r#"fn t(a: bits[1] id=1, b: bits[1] id=2) -> bits[2] {
  literal.9: bits[1] = literal(value=0, id=9)
  concat.10: bits[2] = concat(literal.9, a, id=10)
  concat.11: bits[2] = concat(literal.9, b, id=11)
  ret add.20: bits[2] = add(concat.10, concat.11, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();

        let mut t = OneBitCarryCountTransform;
        assert_eq!(t.find_candidates(&f).len(), 1);
    }

    #[test]
    fn one_bit_carry_count_rejects_add_of_plain_bits2_operands() {
        let ir_text = r#"fn t(a: bits[2] id=1, b: bits[2] id=2) -> bits[2] {
  ret add.20: bits[2] = add(a, b, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();

        let mut t = OneBitCarryCountTransform;
        assert!(t.find_candidates(&f).is_empty());
    }

    #[test]
    fn one_bit_carry_count_rejects_mismatched_and_xor_operands() {
        let ir_text = r#"fn t(a: bits[1] id=1, b: bits[1] id=2, c: bits[1] id=3) -> bits[2] {
  and.10: bits[1] = and(a, b, id=10)
  xor.11: bits[1] = xor(a, c, id=11)
  ret concat.20: bits[2] = concat(and.10, xor.11, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();

        let mut t = OneBitCarryCountTransform;
        assert!(t.find_candidates(&f).is_empty());
    }
}
