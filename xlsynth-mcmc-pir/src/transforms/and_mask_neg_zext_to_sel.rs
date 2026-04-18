// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `and(x, neg(zero_ext(p,w))) ↔ sel(p, cases=[0_w, x])`
/// `and(x, sub(0_w, zero_ext(p,w))) ↔ sel(p, cases=[0_w, x])`
#[derive(Debug)]
pub struct AndMaskNegZextToSelTransform;

impl AndMaskNegZextToSelTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_u1(f: &IrFn, r: NodeRef) -> bool {
        matches!(f.get_node(r).ty, Type::Bits(1))
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn zero_value(w: usize) -> IrValue {
        IrValue::from_bits(&IrBits::make_ubits(w, 0).expect("make_ubits"))
    }

    fn is_zero_literal_node(f: &IrFn, r: NodeRef, w: usize) -> bool {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return false;
        };
        *v == Self::zero_value(w)
    }

    fn zero_ext_u1_parts(f: &IrFn, r: NodeRef) -> Option<(NodeRef, usize)> {
        match &f.get_node(r).payload {
            NodePayload::ZeroExt { arg, new_bit_count } => {
                if *new_bit_count == 0 || !Self::is_u1(f, *arg) {
                    return None;
                }
                if Self::bits_width(f, r) != Some(*new_bit_count) {
                    return None;
                }
                Some((*arg, *new_bit_count))
            }
            NodePayload::Nary(NaryOp::Concat, ops) if ops.len() == 2 => {
                let hi = ops[0];
                let lo = ops[1];
                let w = Self::bits_width(f, r)?;
                if w == 0 || !Self::is_u1(f, lo) {
                    return None;
                }
                if Self::bits_width(f, hi) != Some(w - 1) {
                    return None;
                }
                if !Self::is_zero_literal_node(f, hi, w - 1) {
                    return None;
                }
                Some((lo, w))
            }
            _ => None,
        }
    }

    fn neg_zext_mask_parts(f: &IrFn, r: NodeRef) -> Option<(NodeRef, usize)> {
        match &f.get_node(r).payload {
            NodePayload::Unop(Unop::Neg, zext) => {
                let (p, w) = Self::zero_ext_u1_parts(f, *zext)?;
                if Self::bits_width(f, r) != Some(w) {
                    return None;
                }
                Some((p, w))
            }
            NodePayload::Binop(Binop::Sub, zero, zext) => {
                let (p, w) = Self::zero_ext_u1_parts(f, *zext)?;
                if Self::bits_width(f, r) != Some(w) {
                    return None;
                }
                if !Self::is_zero_literal_node(f, *zero, w) {
                    return None;
                }
                Some((p, w))
            }
            _ => None,
        }
    }

    fn mk_literal_node(f: &mut IrFn, w: usize, value: IrValue) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Literal(value),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_zero_literal_node(f: &mut IrFn, w: usize) -> NodeRef {
        Self::mk_literal_node(f, w, Self::zero_value(w))
    }

    fn mk_zero_ext_u1_node(f: &mut IrFn, w: usize, p: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::ZeroExt {
                arg: p,
                new_bit_count: w,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_sub_node(f: &mut IrFn, w: usize, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Binop(Binop::Sub, lhs, rhs),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for AndMaskNegZextToSelTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::AndMaskNegZextToSel
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let always_equivalent = true;
        let mut out = Vec::<TransformCandidate>::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Nary(NaryOp::And, ops) if ops.len() == 2 => {
                    let mut ok = false;
                    for (x, mask) in [(ops[0], ops[1]), (ops[1], ops[0])] {
                        let Some((_p, w)) = Self::neg_zext_mask_parts(f, mask) else {
                            continue;
                        };
                        if Self::bits_width(f, x) == Some(w) && Self::bits_width(f, nr) == Some(w) {
                            ok = true;
                            break;
                        }
                    }
                    if ok {
                        out.push(TransformCandidate {
                            location: TransformLocation::Node(nr),
                            always_equivalent,
                        });
                    }
                }
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 2 || default.is_some() || !Self::is_u1(f, *selector) {
                        continue;
                    }
                    let Some(w) = Self::bits_width(f, nr) else {
                        continue;
                    };
                    if w == 0 {
                        continue;
                    }
                    if !Self::is_zero_literal_node(f, cases[0], w) {
                        continue;
                    }
                    if Self::bits_width(f, cases[1]) != Some(w) {
                        continue;
                    }
                    out.push(TransformCandidate {
                        location: TransformLocation::Node(nr),
                        always_equivalent,
                    });
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
                    "AndMaskNegZextToSelTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let payload = f.get_node(target_ref).payload.clone();
        match payload {
            NodePayload::Nary(NaryOp::And, ops) => {
                if ops.len() != 2 {
                    return Err("AndMaskNegZextToSelTransform: expected 2-operand and".to_string());
                }
                let mut matched: Option<(NodeRef, NodeRef, usize)> = None;
                for (x, mask) in [(ops[0], ops[1]), (ops[1], ops[0])] {
                    let Some((p, w)) = Self::neg_zext_mask_parts(f, mask) else {
                        continue;
                    };
                    if Self::bits_width(f, x) == Some(w)
                        && Self::bits_width(f, target_ref) == Some(w)
                    {
                        matched = Some((x, p, w));
                        break;
                    }
                }
                let Some((x, p, w)) = matched else {
                    return Err(
                        "AndMaskNegZextToSelTransform: expected and(x, neg/sub-zero zext(p,w)) pattern"
                            .to_string(),
                    );
                };
                let zero = Self::mk_zero_literal_node(f, w);
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: p,
                    cases: vec![zero, x],
                    default: None,
                };
                Ok(())
            }
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                if cases.len() != 2 || default.is_some() {
                    return Err(
                        "AndMaskNegZextToSelTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1(f, selector) {
                    return Err(
                        "AndMaskNegZextToSelTransform: selector must be bits[1]".to_string()
                    );
                }
                let w = Self::bits_width(f, target_ref).ok_or_else(|| {
                    "AndMaskNegZextToSelTransform: output must be bits[w]".to_string()
                })?;
                if w == 0 {
                    return Err(
                        "AndMaskNegZextToSelTransform: zero-width masks are not supported"
                            .to_string(),
                    );
                }
                if !Self::is_zero_literal_node(f, cases[0], w) {
                    return Err(
                        "AndMaskNegZextToSelTransform: expected sel case0 to be 0_w literal"
                            .to_string(),
                    );
                }
                let x = cases[1];
                if Self::bits_width(f, x) != Some(w) {
                    return Err(
                        "AndMaskNegZextToSelTransform: expected sel case1 to be bits[w]"
                            .to_string(),
                    );
                }

                let zero = Self::mk_zero_literal_node(f, w);
                let zext = Self::mk_zero_ext_u1_node(f, w, selector);
                let mask = Self::mk_sub_node(f, w, zero, zext);
                f.get_node_mut(target_ref).payload = NodePayload::Nary(NaryOp::And, vec![x, mask]);
                Ok(())
            }
            _ => Err(
                "AndMaskNegZextToSelTransform: expected and(...) or sel(...) payload".to_string(),
            ),
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
    fn and_mask_neg_zext_to_sel_expands_neg_zext_and() {
        let ir_text = r#"fn t(p: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  zero_ext.10: bits[8] = zero_ext(p, new_bit_count=8, id=10)
  neg.11: bits[8] = neg(zero_ext.10, id=11)
  ret and.20: bits[8] = and(x, neg.11, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let and_ref = find_payload(&f, |p| matches!(p, NodePayload::Nary(NaryOp::And, _)));

        let mut t = AndMaskNegZextToSelTransform;
        assert_eq!(t.find_candidates(&f).len(), 1);
        t.apply(&mut f, &TransformLocation::Node(and_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(and_ref).payload,
            NodePayload::Sel { .. }
        ));
    }

    #[test]
    fn and_mask_neg_zext_to_sel_expands_sub_zero_zext_and() {
        let ir_text = r#"fn t(p: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  literal.9: bits[8] = literal(value=0, id=9)
  zero_ext.10: bits[8] = zero_ext(p, new_bit_count=8, id=10)
  sub.11: bits[8] = sub(literal.9, zero_ext.10, id=11)
  ret and.20: bits[8] = and(sub.11, x, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let and_ref = find_payload(&f, |p| matches!(p, NodePayload::Nary(NaryOp::And, _)));

        let mut t = AndMaskNegZextToSelTransform;
        assert_eq!(t.find_candidates(&f).len(), 1);
        t.apply(&mut f, &TransformLocation::Node(and_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(and_ref).payload,
            NodePayload::Sel { .. }
        ));
    }

    #[test]
    fn and_mask_neg_zext_to_sel_folds_sel_to_sub_zero_zext_and() {
        let ir_text = r#"fn t(p: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  literal.10: bits[8] = literal(value=0, id=10)
  ret sel.20: bits[8] = sel(p, cases=[literal.10, x], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let sel_ref = find_payload(&f, |p| matches!(p, NodePayload::Sel { .. }));

        let t = AndMaskNegZextToSelTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        let NodePayload::Nary(NaryOp::And, ops) = &f.get_node(sel_ref).payload else {
            panic!("expected and");
        };
        assert!(matches!(
            f.get_node(ops[1]).payload,
            NodePayload::Binop(Binop::Sub, _, _)
        ));
    }

    #[test]
    fn and_mask_neg_zext_to_sel_does_not_match_plain_zext_mask() {
        let ir_text = r#"fn t(p: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  zero_ext.10: bits[8] = zero_ext(p, new_bit_count=8, id=10)
  ret and.20: bits[8] = and(x, zero_ext.10, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();

        let mut t = AndMaskNegZextToSelTransform;
        assert!(t.find_candidates(&f).is_empty());
    }
}
