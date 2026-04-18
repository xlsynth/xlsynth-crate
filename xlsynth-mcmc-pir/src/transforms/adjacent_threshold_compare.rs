// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThresholdSide {
    Lhs,
    Rhs,
}

/// A semantics-preserving transform implementing:
///
/// `cmp(x, add(k, zero_ext(p,w))) ↔ sel(p, [cmp(x,k), cmp(x,k+1)])`
/// `cmp(add(k, zero_ext(p,w)), x) ↔ sel(p, [cmp(k,x), cmp(k+1,x)])`
#[derive(Debug)]
pub struct AdjacentThresholdCompareTransform;

impl AdjacentThresholdCompareTransform {
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

    fn is_cmp_op(op: Binop) -> bool {
        matches!(
            op,
            Binop::Eq
                | Binop::Ne
                | Binop::Ult
                | Binop::Ule
                | Binop::Ugt
                | Binop::Uge
                | Binop::Slt
                | Binop::Sle
                | Binop::Sgt
                | Binop::Sge
        )
    }

    fn is_literal(f: &IrFn, r: NodeRef) -> bool {
        matches!(f.get_node(r).payload, NodePayload::Literal(_))
    }

    fn literal_bits(f: &IrFn, r: NodeRef, w: usize) -> Option<IrBits> {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return None;
        };
        let bits = v.to_bits().ok()?;
        if bits.get_bit_count() != w {
            return None;
        }
        Some(bits)
    }

    fn literal_plus_one_value(f: &IrFn, r: NodeRef, w: usize) -> Option<IrValue> {
        if w == 0 {
            return None;
        }
        let bits = Self::literal_bits(f, r, w)?;
        let one = IrBits::make_ubits(w, 1).ok()?;
        Some(IrValue::from_bits(&bits.add(&one)))
    }

    fn is_literal_plus_one_of(f: &IrFn, candidate: NodeRef, base: NodeRef, w: usize) -> bool {
        let Some(expected) = Self::literal_plus_one_value(f, base, w) else {
            return false;
        };
        let NodePayload::Literal(v) = &f.get_node(candidate).payload else {
            return false;
        };
        *v == expected
    }

    fn zero_ext_u1_parts(f: &IrFn, r: NodeRef) -> Option<(NodeRef, usize)> {
        let NodePayload::ZeroExt { arg, new_bit_count } = &f.get_node(r).payload else {
            return None;
        };
        if *new_bit_count == 0 || !Self::is_u1(f, *arg) {
            return None;
        }
        if Self::bits_width(f, r) != Some(*new_bit_count) {
            return None;
        }
        Some((*arg, *new_bit_count))
    }

    fn add_threshold_parts(f: &IrFn, r: NodeRef) -> Option<(NodeRef, NodeRef, usize)> {
        let NodePayload::Binop(Binop::Add, lhs, rhs) = f.get_node(r).payload else {
            return None;
        };
        for (k, zext) in [(lhs, rhs), (rhs, lhs)] {
            if !Self::is_literal(f, k) {
                continue;
            }
            let Some((p, w)) = Self::zero_ext_u1_parts(f, zext) else {
                continue;
            };
            if Self::bits_width(f, k) == Some(w) && Self::bits_width(f, r) == Some(w) {
                return Some((k, p, w));
            }
        }
        None
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

    fn mk_binop_node(f: &mut IrFn, op: Binop, ty: Type, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty,
            payload: NodePayload::Binop(op, lhs, rhs),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn sel_fold_parts(
        f: &IrFn,
        case0: NodeRef,
        case1: NodeRef,
    ) -> Option<(Binop, ThresholdSide, NodeRef, NodeRef, usize)> {
        let NodePayload::Binop(op0, lhs0, rhs0) = f.get_node(case0).payload else {
            return None;
        };
        let NodePayload::Binop(op1, lhs1, rhs1) = f.get_node(case1).payload else {
            return None;
        };
        if op0 != op1 || !Self::is_cmp_op(op0) {
            return None;
        }

        if rhs0 == rhs1 && Self::is_literal(f, lhs0) && Self::is_literal(f, lhs1) {
            let w = Self::bits_width(f, lhs0)?;
            if w != 0
                && Self::bits_width(f, lhs1) == Some(w)
                && Self::bits_width(f, rhs0) == Some(w)
                && Self::is_literal_plus_one_of(f, lhs1, lhs0, w)
            {
                return Some((op0, ThresholdSide::Lhs, rhs0, lhs0, w));
            }
        }

        if lhs0 == lhs1 && Self::is_literal(f, rhs0) && Self::is_literal(f, rhs1) {
            let w = Self::bits_width(f, rhs0)?;
            if w != 0
                && Self::bits_width(f, rhs1) == Some(w)
                && Self::bits_width(f, lhs0) == Some(w)
                && Self::is_literal_plus_one_of(f, rhs1, rhs0, w)
            {
                return Some((op0, ThresholdSide::Rhs, lhs0, rhs0, w));
            }
        }

        None
    }
}

impl PirTransform for AdjacentThresholdCompareTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::AdjacentThresholdCompare
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let always_equivalent = true;
        let mut out = Vec::<TransformCandidate>::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Binop(op, lhs, rhs) if Self::is_cmp_op(*op) => {
                    if Self::bits_width(f, nr) != Some(1) {
                        continue;
                    }
                    let rhs_ok = Self::add_threshold_parts(f, *rhs)
                        .is_some_and(|(_k, _p, w)| Self::bits_width(f, *lhs) == Some(w));
                    let lhs_ok = Self::add_threshold_parts(f, *lhs)
                        .is_some_and(|(_k, _p, w)| Self::bits_width(f, *rhs) == Some(w));
                    if rhs_ok || lhs_ok {
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
                    if Self::bits_width(f, nr) != Some(1)
                        || Self::bits_width(f, cases[0]) != Some(1)
                        || Self::bits_width(f, cases[1]) != Some(1)
                    {
                        continue;
                    }
                    if Self::sel_fold_parts(f, cases[0], cases[1]).is_some() {
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
                    "AdjacentThresholdCompareTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let payload = f.get_node(target_ref).payload.clone();
        match payload {
            NodePayload::Binop(op, lhs, rhs) if Self::is_cmp_op(op) => {
                if Self::bits_width(f, target_ref) != Some(1) {
                    return Err(
                        "AdjacentThresholdCompareTransform: comparison output must be bits[1]"
                            .to_string(),
                    );
                }

                let matched = if let Some((k, p, w)) = Self::add_threshold_parts(f, rhs) {
                    if Self::bits_width(f, lhs) != Some(w) {
                        None
                    } else {
                        Some((ThresholdSide::Rhs, lhs, k, p, w))
                    }
                } else if let Some((k, p, w)) = Self::add_threshold_parts(f, lhs) {
                    if Self::bits_width(f, rhs) != Some(w) {
                        None
                    } else {
                        Some((ThresholdSide::Lhs, rhs, k, p, w))
                    }
                } else {
                    None
                };
                let Some((side, x, k, p, w)) = matched else {
                    return Err(
                        "AdjacentThresholdCompareTransform: expected cmp with adjacent threshold add"
                            .to_string(),
                    );
                };
                let k1 = Self::mk_literal_node(
                    f,
                    w,
                    Self::literal_plus_one_value(f, k, w).ok_or_else(|| {
                        "AdjacentThresholdCompareTransform: could not increment threshold literal"
                            .to_string()
                    })?,
                );
                let (case0, case1) = match side {
                    ThresholdSide::Rhs => (
                        Self::mk_binop_node(f, op, Type::Bits(1), x, k),
                        Self::mk_binop_node(f, op, Type::Bits(1), x, k1),
                    ),
                    ThresholdSide::Lhs => (
                        Self::mk_binop_node(f, op, Type::Bits(1), k, x),
                        Self::mk_binop_node(f, op, Type::Bits(1), k1, x),
                    ),
                };
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: p,
                    cases: vec![case0, case1],
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
                        "AdjacentThresholdCompareTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1(f, selector) {
                    return Err(
                        "AdjacentThresholdCompareTransform: selector must be bits[1]".to_string(),
                    );
                }
                if Self::bits_width(f, target_ref) != Some(1) {
                    return Err(
                        "AdjacentThresholdCompareTransform: sel output must be bits[1]".to_string(),
                    );
                }
                let Some((op, side, x, k, w)) = Self::sel_fold_parts(f, cases[0], cases[1]) else {
                    return Err(
                        "AdjacentThresholdCompareTransform: expected sel of adjacent compares"
                            .to_string(),
                    );
                };
                let zext = Self::mk_zero_ext_u1_node(f, w, selector);
                let threshold = Self::mk_binop_node(f, Binop::Add, Type::Bits(w), k, zext);
                f.get_node_mut(target_ref).payload = match side {
                    ThresholdSide::Rhs => NodePayload::Binop(op, x, threshold),
                    ThresholdSide::Lhs => NodePayload::Binop(op, threshold, x),
                };
                Ok(())
            }
            _ => Err(
                "AdjacentThresholdCompareTransform: expected cmp(...) or sel(...) payload"
                    .to_string(),
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
    fn adjacent_threshold_compare_expands_rhs_threshold() {
        let ir_text = r#"fn t(p: bits[1] id=1, x: bits[8] id=2) -> bits[1] {
  literal.10: bits[8] = literal(value=7, id=10)
  zero_ext.11: bits[8] = zero_ext(p, new_bit_count=8, id=11)
  add.12: bits[8] = add(literal.10, zero_ext.11, id=12)
  ret ult.20: bits[1] = ult(x, add.12, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let cmp_ref = find_payload(&f, |p| matches!(p, NodePayload::Binop(Binop::Ult, _, _)));

        let mut t = AdjacentThresholdCompareTransform;
        assert_eq!(t.find_candidates(&f).len(), 1);
        t.apply(&mut f, &TransformLocation::Node(cmp_ref))
            .expect("apply");

        let NodePayload::Sel { cases, .. } = &f.get_node(cmp_ref).payload else {
            panic!("expected sel");
        };
        assert_eq!(cases.len(), 2);
        assert!(matches!(
            f.get_node(cases[0]).payload,
            NodePayload::Binop(Binop::Ult, _, _)
        ));
    }

    #[test]
    fn adjacent_threshold_compare_expands_lhs_threshold() {
        let ir_text = r#"fn t(p: bits[1] id=1, x: bits[8] id=2) -> bits[1] {
  literal.10: bits[8] = literal(value=7, id=10)
  zero_ext.11: bits[8] = zero_ext(p, new_bit_count=8, id=11)
  add.12: bits[8] = add(zero_ext.11, literal.10, id=12)
  ret uge.20: bits[1] = uge(add.12, x, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let cmp_ref = find_payload(&f, |p| matches!(p, NodePayload::Binop(Binop::Uge, _, _)));

        let mut t = AdjacentThresholdCompareTransform;
        assert_eq!(t.find_candidates(&f).len(), 1);
        t.apply(&mut f, &TransformLocation::Node(cmp_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(cmp_ref).payload,
            NodePayload::Sel { .. }
        ));
    }

    #[test]
    fn adjacent_threshold_compare_folds_sel_of_adjacent_compares() {
        let ir_text = r#"fn t(p: bits[1] id=1, x: bits[8] id=2) -> bits[1] {
  literal.10: bits[8] = literal(value=7, id=10)
  literal.11: bits[8] = literal(value=8, id=11)
  ult.12: bits[1] = ult(x, literal.10, id=12)
  ult.13: bits[1] = ult(x, literal.11, id=13)
  ret sel.20: bits[1] = sel(p, cases=[ult.12, ult.13], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let sel_ref = find_payload(&f, |p| matches!(p, NodePayload::Sel { .. }));

        let t = AdjacentThresholdCompareTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        let NodePayload::Binop(Binop::Ult, _x, rhs) = f.get_node(sel_ref).payload else {
            panic!("expected ult");
        };
        assert!(matches!(
            f.get_node(rhs).payload,
            NodePayload::Binop(Binop::Add, _, _)
        ));
    }

    #[test]
    fn adjacent_threshold_compare_handles_literal_wraparound() {
        let ir_text = r#"fn t(p: bits[1] id=1, x: bits[2] id=2) -> bits[1] {
  literal.10: bits[2] = literal(value=3, id=10)
  literal.11: bits[2] = literal(value=0, id=11)
  eq.12: bits[1] = eq(x, literal.10, id=12)
  eq.13: bits[1] = eq(x, literal.11, id=13)
  ret sel.20: bits[1] = sel(p, cases=[eq.12, eq.13], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();

        let mut t = AdjacentThresholdCompareTransform;
        assert_eq!(t.find_candidates(&f).len(), 1);
    }

    #[test]
    fn adjacent_threshold_compare_handles_wide_literals_without_to_u64() {
        let ir_text = r#"fn t(p: bits[1] id=1, x: bits[128] id=2) -> bits[1] {
  literal.10: bits[128] = literal(value=0xffffffffffffffffffffffffffffffff, id=10)
  literal.11: bits[128] = literal(value=0, id=11)
  ne.12: bits[1] = ne(x, literal.10, id=12)
  ne.13: bits[1] = ne(x, literal.11, id=13)
  ret sel.20: bits[1] = sel(p, cases=[ne.12, ne.13], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();

        let mut t = AdjacentThresholdCompareTransform;
        assert_eq!(t.find_candidates(&f).len(), 1);
    }

    #[test]
    fn adjacent_threshold_compare_rejects_reversed_sel_arms() {
        let ir_text = r#"fn t(p: bits[1] id=1, x: bits[8] id=2) -> bits[1] {
  literal.10: bits[8] = literal(value=7, id=10)
  literal.11: bits[8] = literal(value=8, id=11)
  ult.12: bits[1] = ult(x, literal.10, id=12)
  ult.13: bits[1] = ult(x, literal.11, id=13)
  ret sel.20: bits[1] = sel(p, cases=[ult.13, ult.12], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();

        let mut t = AdjacentThresholdCompareTransform;
        assert!(t.find_candidates(&f).is_empty());
    }
}
