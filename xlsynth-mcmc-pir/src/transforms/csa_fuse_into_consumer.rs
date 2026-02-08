// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing reassociation to keep the
/// carry-save portion live when an add feeds another add.
#[derive(Debug)]
pub struct CsaFuseIntoConsumerTransform;

impl CsaFuseIntoConsumerTransform {
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

    fn match_shifted_and(f: &IrFn, shll_ref: NodeRef) -> Option<(NodeRef, NodeRef, NodeRef)> {
        let w = Self::bits_width(f, shll_ref)?;
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
        let NodePayload::Nary(NaryOp::And, _) = f.get_node(and_ref).payload else {
            return None;
        };
        Some((and_ref, amount_ref, shll_ref))
    }

    fn match_fission_add(
        f: &IrFn,
        add_ref: NodeRef,
    ) -> Option<(NodeRef, NodeRef, NodeRef, NodeRef)> {
        let NodePayload::Binop(Binop::Add, lhs, rhs) = f.get_node(add_ref).payload else {
            return None;
        };
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

        let (and_ref, _amount_ref, _shll_ref) = Self::match_shifted_and(f, shll_ref)?;
        let (a, b) = Self::match_xor_and_pair(f, xor_ref, and_ref)?;
        Some((a, b, xor_ref, shll_ref))
    }

    fn match_xor_add_with_other(f: &IrFn, add_ref: NodeRef) -> Option<(NodeRef, NodeRef, NodeRef)> {
        let NodePayload::Binop(Binop::Add, lhs, rhs) = f.get_node(add_ref).payload else {
            return None;
        };
        if matches!(f.get_node(lhs).payload, NodePayload::Nary(NaryOp::Xor, _)) {
            Some((lhs, rhs, add_ref))
        } else if matches!(f.get_node(rhs).payload, NodePayload::Nary(NaryOp::Xor, _)) {
            Some((rhs, lhs, add_ref))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn csa_fuse_into_consumer_matches_wide_shift_one_literal() {
        // Regression test: for widths > 64, `IrValue::to_u64()` fails, so we
        // must match the shift amount literal via bits inspection.
        let ir_text = r#"fn t(a: bits[128] id=1, b: bits[128] id=2, c: bits[128] id=3) -> bits[128] {
  xor.4: bits[128] = xor(a, b, id=4)
  and.5: bits[128] = and(a, b, id=5)
  literal.6: bits[128] = literal(value=1, id=6)
  shll.7: bits[128] = shll(and.5, literal.6, id=7)
  add.8: bits[128] = add(xor.4, shll.7, id=8)
  ret add.9: bits[128] = add(add.8, c, id=9)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let outer_add_ref = f
            .node_refs()
            .into_iter()
            .find(|nr| f.get_node(*nr).text_id == 9)
            .expect("expected outer add node");

        let t = CsaFuseIntoConsumerTransform;
        t.apply(&mut f, &TransformLocation::Node(outer_add_ref))
            .expect("apply");

        // Expect: add(add(xor(a,b), c), shll(and(a,b), 1))
        let NodePayload::Binop(Binop::Add, inner_add_ref, shll_ref) =
            f.get_node(outer_add_ref).payload
        else {
            panic!("expected outer add after reassociation");
        };
        assert_eq!(f.get_node(shll_ref).text_id, 7);

        let NodePayload::Binop(Binop::Add, xor_ref, c_ref) = f.get_node(inner_add_ref).payload
        else {
            panic!("expected inner add after reassociation");
        };
        assert_eq!(f.get_node(xor_ref).text_id, 4);
        assert!(matches!(
            f.get_node(c_ref).payload,
            NodePayload::GetParam(_)
        ));
    }
}

impl PirTransform for CsaFuseIntoConsumerTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::CsaFuseIntoConsumer
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            if !matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Add, _, _)) {
                continue;
            }
            let mut matches = false;
            if let NodePayload::Binop(Binop::Add, lhs, rhs) = f.get_node(nr).payload {
                if Self::match_fission_add(f, lhs).is_some()
                    || Self::match_fission_add(f, rhs).is_some()
                {
                    matches = true;
                }
                if let Some((xor_ref, _other_ref, _)) = Self::match_xor_add_with_other(f, lhs) {
                    if Self::match_shifted_and(f, rhs).is_some()
                        && Self::match_shifted_and(f, rhs)
                            .and_then(|(and_ref, _, _)| {
                                Self::match_xor_and_pair(f, xor_ref, and_ref)
                            })
                            .is_some()
                    {
                        matches = true;
                    }
                }
                if let Some((xor_ref, _other_ref, _)) = Self::match_xor_add_with_other(f, rhs) {
                    if Self::match_shifted_and(f, lhs).is_some()
                        && Self::match_shifted_and(f, lhs)
                            .and_then(|(and_ref, _, _)| {
                                Self::match_xor_and_pair(f, xor_ref, and_ref)
                            })
                            .is_some()
                    {
                        matches = true;
                    }
                }
            }
            if matches {
                out.push(TransformLocation::Node(nr));
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "CsaFuseIntoConsumerTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let w = Self::bits_width(f, target_ref)
            .ok_or_else(|| "CsaFuseIntoConsumerTransform: output must be bits[w]".to_string())?;

        let NodePayload::Binop(Binop::Add, lhs, rhs) = f.get_node(target_ref).payload else {
            return Err("CsaFuseIntoConsumerTransform: expected add".to_string());
        };

        if let Some((_a, _b, xor_ref, shll_ref)) = Self::match_fission_add(f, lhs) {
            if Self::bits_width(f, xor_ref) != Some(w) || Self::bits_width(f, rhs) != Some(w) {
                return Err("CsaFuseIntoConsumerTransform: width mismatch".to_string());
            }
            let new_inner = Self::mk_binop_bits_node(f, Binop::Add, w, xor_ref, rhs);
            f.get_node_mut(target_ref).payload =
                NodePayload::Binop(Binop::Add, new_inner, shll_ref);
            return Ok(());
        }
        if let Some((_a, _b, xor_ref, shll_ref)) = Self::match_fission_add(f, rhs) {
            if Self::bits_width(f, xor_ref) != Some(w) || Self::bits_width(f, lhs) != Some(w) {
                return Err("CsaFuseIntoConsumerTransform: width mismatch".to_string());
            }
            let new_inner = Self::mk_binop_bits_node(f, Binop::Add, w, xor_ref, lhs);
            f.get_node_mut(target_ref).payload =
                NodePayload::Binop(Binop::Add, new_inner, shll_ref);
            return Ok(());
        }

        if let Some((xor_ref, d_ref, _inner_ref)) = Self::match_xor_add_with_other(f, lhs) {
            if let Some((and_ref, _amount_ref, shll_ref)) = Self::match_shifted_and(f, rhs) {
                if Self::match_xor_and_pair(f, xor_ref, and_ref).is_some() {
                    let new_inner = Self::mk_binop_bits_node(f, Binop::Add, w, xor_ref, shll_ref);
                    f.get_node_mut(target_ref).payload =
                        NodePayload::Binop(Binop::Add, new_inner, d_ref);
                    return Ok(());
                }
            }
        }
        if let Some((xor_ref, d_ref, _inner_ref)) = Self::match_xor_add_with_other(f, rhs) {
            if let Some((and_ref, _amount_ref, shll_ref)) = Self::match_shifted_and(f, lhs) {
                if Self::match_xor_and_pair(f, xor_ref, and_ref).is_some() {
                    let new_inner = Self::mk_binop_bits_node(f, Binop::Add, w, xor_ref, shll_ref);
                    f.get_node_mut(target_ref).payload =
                        NodePayload::Binop(Binop::Add, new_inner, d_ref);
                    return Ok(());
                }
            }
        }

        Err("CsaFuseIntoConsumerTransform: pattern did not match".to_string())
    }
}
