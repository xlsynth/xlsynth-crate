// SPDX-License-Identifier: Apache-2.0

use super::macro_utils as mu;
use super::*;
use xlsynth_pir::ir_match::MatchCtx;

/// Splits reductions over two-operand concat and merges them back.
#[derive(Debug)]
pub struct ReduceConcatSplitMergeTransform;

impl ReduceConcatSplitMergeTransform {
    fn is_reduce(op: Unop) -> bool {
        matches!(op, Unop::OrReduce | Unop::AndReduce | Unop::XorReduce)
    }

    fn nary_for_reduce(op: Unop) -> Option<NaryOp> {
        match op {
            Unop::OrReduce => Some(NaryOp::Or),
            Unop::AndReduce => Some(NaryOp::And),
            Unop::XorReduce => Some(NaryOp::Xor),
            _ => None,
        }
    }

    fn reduce_for_nary(op: NaryOp) -> Option<Unop> {
        match op {
            NaryOp::Or => Some(Unop::OrReduce),
            NaryOp::And => Some(Unop::AndReduce),
            NaryOp::Xor => Some(Unop::XorReduce),
            _ => None,
        }
    }

    fn forward_parts(f: &IrFn, nr: NodeRef) -> Option<(Unop, NodeRef, NodeRef)> {
        let NodePayload::Unop(op, concat) = f.get_node(nr).payload else {
            return None;
        };
        if !Self::is_reduce(op) || !mu::is_u1(f, nr) {
            return None;
        }
        let (a, b) = mu::concat2_parts(f, concat)?;
        if mu::bits_width(f, a)? == 0 || mu::bits_width(f, b)? == 0 {
            return None;
        }
        Some((op, a, b))
    }

    fn reverse_parts(f: &IrFn, nr: NodeRef) -> Option<(Unop, NodeRef, NodeRef)> {
        if !mu::is_u1(f, nr) {
            return None;
        }
        let NodePayload::Nary(nary_op, _) = f.get_node(nr).payload else {
            return None;
        };
        let reduce_op = Self::reduce_for_nary(nary_op)?;
        let ctx = MatchCtx::new(f);
        let mut ops = ctx.flattened_nary_operands(nr, nary_op)?;
        if ops.len() != 2 {
            return None;
        }
        ops.sort_by_key(|r| r.index);
        let NodePayload::Unop(op0, arg0) = f.get_node(ops[0]).payload else {
            return None;
        };
        let NodePayload::Unop(op1, arg1) = f.get_node(ops[1]).payload else {
            return None;
        };
        if op0 != reduce_op || op1 != reduce_op || !mu::is_u1(f, ops[0]) || !mu::is_u1(f, ops[1]) {
            return None;
        }
        if mu::bits_width(f, arg0)? == 0 || mu::bits_width(f, arg1)? == 0 {
            return None;
        }
        Some((reduce_op, arg0, arg1))
    }
}

impl PirTransform for ReduceConcatSplitMergeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ReduceConcatSplitMerge
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
                return Err("ReduceConcatSplitMergeTransform: expected node location".to_string());
            }
        };
        if let Some((op, a, b)) = Self::forward_parts(f, target) {
            let nary_op = Self::nary_for_reduce(op)
                .ok_or_else(|| "ReduceConcatSplitMergeTransform: unsupported reduce".to_string())?;
            let pred_a = mu::mk_unop(f, op, Type::Bits(1), a);
            let pred_b = mu::mk_unop(f, op, Type::Bits(1), b);
            f.get_node_mut(target).payload = NodePayload::Nary(nary_op, vec![pred_a, pred_b]);
            return Ok(());
        }
        if let Some((op, a, b)) = Self::reverse_parts(f, target) {
            let wa = mu::bits_width(f, a)
                .ok_or_else(|| "ReduceConcatSplitMergeTransform: a must be bits".to_string())?;
            let wb = mu::bits_width(f, b)
                .ok_or_else(|| "ReduceConcatSplitMergeTransform: b must be bits".to_string())?;
            let concat = mu::mk_nary(f, NaryOp::Concat, Type::Bits(wa + wb), vec![a, b]);
            f.get_node_mut(target).payload = NodePayload::Unop(op, concat);
            return Ok(());
        }
        Err("ReduceConcatSplitMergeTransform: unsupported target".to_string())
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
    fn splits_or_reduce_of_concat() {
        let mut f = parse_fn(
            r#"fn t(a: bits[3] id=1, b: bits[2] id=2) -> bits[1] {
  concat.3: bits[5] = concat(a, b, id=3)
  ret r: bits[1] = or_reduce(concat.3, id=4)
}"#,
        );
        let target = f.ret_node_ref.unwrap();
        ReduceConcatSplitMergeTransform
            .apply(&mut f, &TransformLocation::Node(target))
            .unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Nary(NaryOp::Or, _)
        ));
    }

    #[test]
    fn merges_xor_of_xor_reduces() {
        let mut f = parse_fn(
            r#"fn t(a: bits[3] id=1, b: bits[2] id=2) -> bits[1] {
  ra: bits[1] = xor_reduce(a, id=3)
  rb: bits[1] = xor_reduce(b, id=4)
  ret r: bits[1] = xor(rb, ra, id=5)
}"#,
        );
        let target = f.ret_node_ref.unwrap();
        ReduceConcatSplitMergeTransform
            .apply(&mut f, &TransformLocation::Node(target))
            .unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Unop(Unop::XorReduce, _)
        ));
    }

    #[test]
    fn rejects_zero_width_concat_part() {
        let f = parse_fn(
            r#"fn t(a: bits[0] id=1, b: bits[2] id=2) -> bits[1] {
  concat.3: bits[2] = concat(a, b, id=3)
  ret r: bits[1] = or_reduce(concat.3, id=4)
}"#,
        );
        let mut t = ReduceConcatSplitMergeTransform;
        assert!(t.find_candidates(&f).is_empty());
    }

    #[test]
    fn candidate_order_is_node_order() {
        let f = parse_fn(
            r#"fn t(a: bits[2] id=1, b: bits[2] id=2) -> bits[1] {
  c0: bits[4] = concat(a, b, id=3)
  r0: bits[1] = or_reduce(c0, id=4)
  c1: bits[4] = concat(b, a, id=5)
  ret r1: bits[1] = and_reduce(c1, id=6)
}"#,
        );
        let mut t = ReduceConcatSplitMergeTransform;
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
            r#"fn t(a: bits[2] id=1, b: bits[1] id=2) -> bits[1] {
  concat.3: bits[3] = concat(a, b, id=3)
  ret r: bits[1] = xor_reduce(concat.3, id=4)
}"#,
        );
        let mut after = before.clone();
        let target = after.ret_node_ref.unwrap();
        ReduceConcatSplitMergeTransform
            .apply(&mut after, &TransformLocation::Node(target))
            .unwrap();
        compact_and_toposort_in_place(&mut before).unwrap();
        compact_and_toposort_in_place(&mut after).unwrap();

        for a in 0..4 {
            for b in 0..2 {
                let values = [a, b];
                let widths = [2, 1];
                assert_eq!(
                    eval_bits(&before, &values, &widths),
                    eval_bits(&after, &values, &widths)
                );
            }
        }
    }
}
