// SPDX-License-Identifier: Apache-2.0

use super::macro_utils as mu;
use super::*;
use xlsynth_pir::ir_match::MatchCtx;

/// Splits two-field concat comparisons into fieldwise predicates and folds
/// them.
#[derive(Debug)]
pub struct ConcatCompareSplitMergeTransform;

#[derive(Clone, Copy)]
struct ConcatCompareParts {
    op: Binop,
    a_hi: NodeRef,
    a_lo: NodeRef,
    b_hi: NodeRef,
    b_lo: NodeRef,
    hi_width: usize,
    lo_width: usize,
}

impl ConcatCompareSplitMergeTransform {
    fn is_supported_compare(op: Binop) -> bool {
        matches!(
            op,
            Binop::Eq
                | Binop::Ne
                | Binop::Ugt
                | Binop::Uge
                | Binop::Ult
                | Binop::Ule
                | Binop::Sgt
                | Binop::Sge
                | Binop::Slt
                | Binop::Sle
        )
    }

    fn compare_parts(f: &IrFn, nr: NodeRef) -> Option<(Binop, NodeRef, NodeRef)> {
        if !mu::is_u1(f, nr) {
            return None;
        }
        let NodePayload::Binop(op, lhs, rhs) = f.get_node(nr).payload else {
            return None;
        };
        Self::is_supported_compare(op).then_some((op, lhs, rhs))
    }

    fn forward_parts(f: &IrFn, nr: NodeRef) -> Option<ConcatCompareParts> {
        let (op, lhs, rhs) = Self::compare_parts(f, nr)?;
        let (a_hi, a_lo) = mu::concat2_parts(f, lhs)?;
        let (b_hi, b_lo) = mu::concat2_parts(f, rhs)?;
        let hi_width = mu::bits_width(f, a_hi)?;
        let lo_width = mu::bits_width(f, a_lo)?;
        if hi_width == 0
            || lo_width == 0
            || mu::bits_width(f, b_hi) != Some(hi_width)
            || mu::bits_width(f, b_lo) != Some(lo_width)
        {
            return None;
        }
        Some(ConcatCompareParts {
            op,
            a_hi,
            a_lo,
            b_hi,
            b_lo,
            hi_width,
            lo_width,
        })
    }

    fn nary_operands(f: &IrFn, nr: NodeRef, op: NaryOp) -> Option<Vec<NodeRef>> {
        let ctx = MatchCtx::new(f);
        let operands = ctx.flattened_nary_operands(nr, op)?;
        (operands.len() == 2).then_some(operands)
    }

    fn eq_like_reverse_parts(
        f: &IrFn,
        nr: NodeRef,
        nary_op: NaryOp,
        cmp_op: Binop,
    ) -> Option<ConcatCompareParts> {
        let mut operands = Self::nary_operands(f, nr, nary_op)?;
        operands.sort_by_key(|r| r.index);
        let (op0, a_hi, b_hi) = Self::compare_parts(f, operands[0])?;
        let (op1, a_lo, b_lo) = Self::compare_parts(f, operands[1])?;
        if op0 != cmp_op || op1 != cmp_op {
            return None;
        }
        let hi_width = mu::bits_width(f, a_hi)?;
        let lo_width = mu::bits_width(f, a_lo)?;
        if hi_width == 0
            || lo_width == 0
            || mu::bits_width(f, b_hi) != Some(hi_width)
            || mu::bits_width(f, b_lo) != Some(lo_width)
        {
            return None;
        }
        Some(ConcatCompareParts {
            op: cmp_op,
            a_hi,
            a_lo,
            b_hi,
            b_lo,
            hi_width,
            lo_width,
        })
    }

    fn strict_and_low_to_original_op(strict_op: Binop, low_op: Binop) -> Option<Binop> {
        match (strict_op, low_op) {
            (Binop::Ugt, Binop::Ugt) => Some(Binop::Ugt),
            (Binop::Ugt, Binop::Uge) => Some(Binop::Uge),
            (Binop::Ult, Binop::Ult) => Some(Binop::Ult),
            (Binop::Ult, Binop::Ule) => Some(Binop::Ule),
            (Binop::Sgt, Binop::Ugt) => Some(Binop::Sgt),
            (Binop::Sgt, Binop::Uge) => Some(Binop::Sge),
            (Binop::Slt, Binop::Ult) => Some(Binop::Slt),
            (Binop::Slt, Binop::Ule) => Some(Binop::Sle),
            _ => None,
        }
    }

    fn split_order_ops(op: Binop) -> Option<(Binop, Binop)> {
        match op {
            Binop::Ugt => Some((Binop::Ugt, Binop::Ugt)),
            Binop::Uge => Some((Binop::Ugt, Binop::Uge)),
            Binop::Ult => Some((Binop::Ult, Binop::Ult)),
            Binop::Ule => Some((Binop::Ult, Binop::Ule)),
            Binop::Sgt => Some((Binop::Sgt, Binop::Ugt)),
            Binop::Sge => Some((Binop::Sgt, Binop::Uge)),
            Binop::Slt => Some((Binop::Slt, Binop::Ult)),
            Binop::Sle => Some((Binop::Slt, Binop::Ule)),
            _ => None,
        }
    }

    fn eq_matches_oriented(f: &IrFn, nr: NodeRef, lhs: NodeRef, rhs: NodeRef) -> bool {
        let Some((Binop::Eq, a, b)) = Self::compare_parts(f, nr) else {
            return false;
        };
        (a == lhs && b == rhs) || (a == rhs && b == lhs)
    }

    fn order_reverse_from_pair(
        f: &IrFn,
        strict_nr: NodeRef,
        and_nr: NodeRef,
    ) -> Option<ConcatCompareParts> {
        let Some((strict_op, a_hi, b_hi)) = Self::compare_parts(f, strict_nr) else {
            return None;
        };
        let and_operands = Self::nary_operands(f, and_nr, NaryOp::And)?;
        for (eq_nr, low_nr) in [
            (and_operands[0], and_operands[1]),
            (and_operands[1], and_operands[0]),
        ] {
            if !Self::eq_matches_oriented(f, eq_nr, a_hi, b_hi) {
                continue;
            }
            let Some((low_op, a_lo, b_lo)) = Self::compare_parts(f, low_nr) else {
                continue;
            };
            let Some(op) = Self::strict_and_low_to_original_op(strict_op, low_op) else {
                continue;
            };
            let hi_width = mu::bits_width(f, a_hi)?;
            let lo_width = mu::bits_width(f, a_lo)?;
            if hi_width == 0
                || lo_width == 0
                || mu::bits_width(f, b_hi) != Some(hi_width)
                || mu::bits_width(f, b_lo) != Some(lo_width)
            {
                continue;
            }
            return Some(ConcatCompareParts {
                op,
                a_hi,
                a_lo,
                b_hi,
                b_lo,
                hi_width,
                lo_width,
            });
        }
        None
    }

    fn ordering_reverse_parts(f: &IrFn, nr: NodeRef) -> Option<ConcatCompareParts> {
        let operands = Self::nary_operands(f, nr, NaryOp::Or)?;
        Self::order_reverse_from_pair(f, operands[0], operands[1])
            .or_else(|| Self::order_reverse_from_pair(f, operands[1], operands[0]))
    }

    fn reverse_parts(f: &IrFn, nr: NodeRef) -> Option<ConcatCompareParts> {
        if !mu::is_u1(f, nr) {
            return None;
        }
        Self::eq_like_reverse_parts(f, nr, NaryOp::And, Binop::Eq)
            .or_else(|| Self::eq_like_reverse_parts(f, nr, NaryOp::Or, Binop::Ne))
            .or_else(|| Self::ordering_reverse_parts(f, nr))
    }

    fn build_concat_compare(f: &mut IrFn, parts: ConcatCompareParts) -> NodePayload {
        let lhs = mu::mk_nary(
            f,
            NaryOp::Concat,
            Type::Bits(parts.hi_width + parts.lo_width),
            vec![parts.a_hi, parts.a_lo],
        );
        let rhs = mu::mk_nary(
            f,
            NaryOp::Concat,
            Type::Bits(parts.hi_width + parts.lo_width),
            vec![parts.b_hi, parts.b_lo],
        );
        NodePayload::Binop(parts.op, lhs, rhs)
    }
}

impl PirTransform for ConcatCompareSplitMergeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ConcatCompareSplitMerge
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            if Self::forward_parts(f, nr).is_some() || Self::reverse_parts(f, nr).is_some() {
                out.push(TransformCandidate {
                    location: TransformLocation::Node(nr),
                    always_equivalent: true,
                });
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err("ConcatCompareSplitMergeTransform: expected node location".to_string());
            }
        };
        if let Some(parts) = Self::forward_parts(f, target) {
            match parts.op {
                Binop::Eq => {
                    let hi = mu::mk_binop(f, Binop::Eq, Type::Bits(1), parts.a_hi, parts.b_hi);
                    let lo = mu::mk_binop(f, Binop::Eq, Type::Bits(1), parts.a_lo, parts.b_lo);
                    f.get_node_mut(target).payload = NodePayload::Nary(NaryOp::And, vec![hi, lo]);
                    return Ok(());
                }
                Binop::Ne => {
                    let hi = mu::mk_binop(f, Binop::Ne, Type::Bits(1), parts.a_hi, parts.b_hi);
                    let lo = mu::mk_binop(f, Binop::Ne, Type::Bits(1), parts.a_lo, parts.b_lo);
                    f.get_node_mut(target).payload = NodePayload::Nary(NaryOp::Or, vec![hi, lo]);
                    return Ok(());
                }
                _ => {
                    let (strict_op, low_op) = Self::split_order_ops(parts.op).ok_or_else(|| {
                        "ConcatCompareSplitMergeTransform: unsupported compare".to_string()
                    })?;
                    let strict_hi =
                        mu::mk_binop(f, strict_op, Type::Bits(1), parts.a_hi, parts.b_hi);
                    let eq_hi = mu::mk_binop(f, Binop::Eq, Type::Bits(1), parts.a_hi, parts.b_hi);
                    let low_cmp = mu::mk_binop(f, low_op, Type::Bits(1), parts.a_lo, parts.b_lo);
                    let eq_and_low =
                        mu::mk_nary(f, NaryOp::And, Type::Bits(1), vec![eq_hi, low_cmp]);
                    f.get_node_mut(target).payload =
                        NodePayload::Nary(NaryOp::Or, vec![strict_hi, eq_and_low]);
                    return Ok(());
                }
            }
        }
        if let Some(parts) = Self::reverse_parts(f, target) {
            let payload = Self::build_concat_compare(f, parts);
            f.get_node_mut(target).payload = payload;
            return Ok(());
        }
        Err("ConcatCompareSplitMergeTransform: unsupported target".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
    use xlsynth_pir::ir_parser;
    use xlsynth_pir::ir_utils::compact_and_toposort_in_place;

    fn parse_fn(ir_text: &str) -> IrFn {
        ir_parser::Parser::new(ir_text).parse_fn().unwrap()
    }

    fn eval_bits(f: &IrFn, values: &[u64], widths: &[usize]) -> IrValue {
        let args = values
            .iter()
            .zip(widths.iter())
            .map(|(value, width)| IrValue::from_bits(&IrBits::make_ubits(*width, *value).unwrap()))
            .collect::<Vec<_>>();
        match eval_fn(f, &args) {
            FnEvalResult::Success(s) => s.value,
            FnEvalResult::Failure(f) => panic!("eval failed: {f:?}"),
        }
    }

    #[test]
    fn splits_unsigned_compare_of_concats() {
        let mut f = parse_fn(
            r#"fn t(a_hi: bits[2] id=1, a_lo: bits[3] id=2, b_hi: bits[2] id=3, b_lo: bits[3] id=4) -> bits[1] {
  a: bits[5] = concat(a_hi, a_lo, id=5)
  b: bits[5] = concat(b_hi, b_lo, id=6)
  ret r: bits[1] = ugt(a, b, id=7)
}"#,
        );
        let target = f.ret_node_ref.unwrap();
        ConcatCompareSplitMergeTransform
            .apply(&mut f, &TransformLocation::Node(target))
            .unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Nary(NaryOp::Or, _)
        ));
    }

    #[test]
    fn folds_eq_fieldwise_compare() {
        let mut f = parse_fn(
            r#"fn t(a_hi: bits[2] id=1, a_lo: bits[3] id=2, b_hi: bits[2] id=3, b_lo: bits[3] id=4) -> bits[1] {
  hi: bits[1] = eq(a_hi, b_hi, id=5)
  lo: bits[1] = eq(a_lo, b_lo, id=6)
  ret r: bits[1] = and(lo, hi, id=7)
}"#,
        );
        let target = f.ret_node_ref.unwrap();
        ConcatCompareSplitMergeTransform
            .apply(&mut f, &TransformLocation::Node(target))
            .unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Binop(Binop::Eq, _, _)
        ));
    }

    #[test]
    fn rejects_mismatched_concat_fields() {
        let f = parse_fn(
            r#"fn t(a_hi: bits[2] id=1, a_lo: bits[3] id=2, b_hi: bits[3] id=3, b_lo: bits[2] id=4) -> bits[1] {
  a: bits[5] = concat(a_hi, a_lo, id=5)
  b: bits[5] = concat(b_hi, b_lo, id=6)
  ret r: bits[1] = ugt(a, b, id=7)
}"#,
        );
        let mut t = ConcatCompareSplitMergeTransform;
        assert!(t.find_candidates(&f).is_empty());
    }

    #[test]
    fn candidate_order_is_node_order() {
        let f = parse_fn(
            r#"fn t(a_hi: bits[1] id=1, a_lo: bits[1] id=2, b_hi: bits[1] id=3, b_lo: bits[1] id=4) -> bits[1] {
  a0: bits[2] = concat(a_hi, a_lo, id=5)
  b0: bits[2] = concat(b_hi, b_lo, id=6)
  c0: bits[1] = eq(a0, b0, id=7)
  a1: bits[2] = concat(b_hi, b_lo, id=8)
  b1: bits[2] = concat(a_hi, a_lo, id=9)
  ret c1: bits[1] = ne(a1, b1, id=10)
}"#,
        );
        let mut t = ConcatCompareSplitMergeTransform;
        let candidates = t.find_candidates(&f);
        let node_indexes = candidates
            .iter()
            .map(|c| match c.location {
                TransformLocation::Node(nr) => nr.index,
                TransformLocation::RewireOperand { .. } => panic!("expected node candidate"),
            })
            .collect::<Vec<_>>();
        assert!(node_indexes.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn exhaustive_small_width_forward_equivalence() {
        let mut before = parse_fn(
            r#"fn t(a_hi: bits[1] id=1, a_lo: bits[1] id=2, b_hi: bits[1] id=3, b_lo: bits[1] id=4) -> bits[1] {
  a: bits[2] = concat(a_hi, a_lo, id=5)
  b: bits[2] = concat(b_hi, b_lo, id=6)
  ret r: bits[1] = sge(a, b, id=7)
}"#,
        );
        let mut after = before.clone();
        let target = after.ret_node_ref.unwrap();
        ConcatCompareSplitMergeTransform
            .apply(&mut after, &TransformLocation::Node(target))
            .unwrap();
        compact_and_toposort_in_place(&mut before).unwrap();
        compact_and_toposort_in_place(&mut after).unwrap();

        for a_hi in 0..2 {
            for a_lo in 0..2 {
                for b_hi in 0..2 {
                    for b_lo in 0..2 {
                        let values = [a_hi, a_lo, b_hi, b_lo];
                        let widths = [1, 1, 1, 1];
                        assert_eq!(
                            eval_bits(&before, &values, &widths),
                            eval_bits(&after, &values, &widths)
                        );
                    }
                }
            }
        }
    }
}
