// SPDX-License-Identifier: Apache-2.0

use super::macro_utils as mu;
use super::*;

/// Distributes reduction predicates over two-arm selects and folds them back.
#[derive(Debug)]
pub struct ReduceSelDistributeTransform;

impl ReduceSelDistributeTransform {
    fn is_reduce(op: Unop) -> bool {
        matches!(op, Unop::OrReduce | Unop::AndReduce | Unop::XorReduce)
    }

    fn forward_parts(f: &IrFn, nr: NodeRef) -> Option<(Unop, NodeRef, NodeRef, NodeRef, usize)> {
        let NodePayload::Unop(op, sel) = f.get_node(nr).payload else {
            return None;
        };
        if !Self::is_reduce(op) || !mu::is_u1(f, nr) {
            return None;
        }
        let (selector, case0, case1) = mu::sel2_parts(f, sel)?;
        if !mu::is_u1(f, selector) {
            return None;
        }
        let width = mu::bits_width(f, case0)?;
        if width == 0 || mu::bits_width(f, case1) != Some(width) {
            return None;
        }
        Some((op, selector, case0, case1, width))
    }

    fn reverse_parts(f: &IrFn, nr: NodeRef) -> Option<(Unop, NodeRef, NodeRef, NodeRef, usize)> {
        if !mu::is_u1(f, nr) {
            return None;
        }
        let (selector, case0, case1) = mu::sel2_parts(f, nr)?;
        if !mu::is_u1(f, selector) {
            return None;
        }
        let NodePayload::Unop(op0, arg0) = f.get_node(case0).payload else {
            return None;
        };
        let NodePayload::Unop(op1, arg1) = f.get_node(case1).payload else {
            return None;
        };
        if op0 != op1 || !Self::is_reduce(op0) || !mu::is_u1(f, case0) || !mu::is_u1(f, case1) {
            return None;
        }
        let width = mu::bits_width(f, arg0)?;
        if width == 0 || mu::bits_width(f, arg1) != Some(width) {
            return None;
        }
        Some((op0, selector, arg0, arg1, width))
    }
}

impl PirTransform for ReduceSelDistributeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ReduceSelDistribute
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
                return Err("ReduceSelDistributeTransform: expected node location".to_string());
            }
        };
        if let Some((op, selector, case0, case1, _)) = Self::forward_parts(f, target) {
            let pred0 = mu::mk_unop(f, op, Type::Bits(1), case0);
            let pred1 = mu::mk_unop(f, op, Type::Bits(1), case1);
            f.get_node_mut(target).payload = NodePayload::Sel {
                selector,
                cases: vec![pred0, pred1],
                default: None,
            };
            return Ok(());
        }
        if let Some((op, selector, arg0, arg1, width)) = Self::reverse_parts(f, target) {
            let sel = mu::mk_sel2(f, selector, Type::Bits(width), arg0, arg1);
            f.get_node_mut(target).payload = NodePayload::Unop(op, sel);
            return Ok(());
        }
        Err("ReduceSelDistributeTransform: unsupported target".to_string())
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
    fn expands_or_reduce_of_sel() {
        let mut f = parse_fn(
            r#"fn t(p: bits[1] id=1, x: bits[8] id=2, y: bits[8] id=3) -> bits[1] {
  sel.4: bits[8] = sel(p, cases=[x, y], id=4)
  ret r: bits[1] = or_reduce(sel.4, id=5)
}"#,
        );
        let target = f.ret_node_ref.unwrap();
        let t = ReduceSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Sel { .. }
        ));
    }

    #[test]
    fn folds_sel_of_and_reduces() {
        let mut f = parse_fn(
            r#"fn t(p: bits[1] id=1, x: bits[8] id=2, y: bits[8] id=3) -> bits[1] {
  rx: bits[1] = and_reduce(x, id=4)
  ry: bits[1] = and_reduce(y, id=5)
  ret r: bits[1] = sel(p, cases=[rx, ry], id=6)
}"#,
        );
        let target = f.ret_node_ref.unwrap();
        let t = ReduceSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Unop(Unop::AndReduce, _)
        ));
    }

    #[test]
    fn rejects_mismatched_case_widths() {
        let f = parse_fn(
            r#"fn t(p: bits[1] id=1, x: bits[8] id=2, y: bits[7] id=3) -> bits[1] {
  sel.4: bits[8] = sel(p, cases=[x, y], id=4)
  ret r: bits[1] = or_reduce(sel.4, id=5)
}"#,
        );
        let mut t = ReduceSelDistributeTransform;
        assert!(t.find_candidates(&f).is_empty());
    }

    #[test]
    fn candidate_order_is_node_order() {
        let f = parse_fn(
            r#"fn t(p: bits[1] id=1, x: bits[2] id=2, y: bits[2] id=3) -> bits[1] {
  s0: bits[2] = sel(p, cases=[x, y], id=4)
  r0: bits[1] = or_reduce(s0, id=5)
  s1: bits[2] = sel(p, cases=[y, x], id=6)
  ret r1: bits[1] = xor_reduce(s1, id=7)
}"#,
        );
        let mut t = ReduceSelDistributeTransform;
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
            r#"fn t(p: bits[1] id=1, x: bits[2] id=2, y: bits[2] id=3) -> bits[1] {
  sel.4: bits[2] = sel(p, cases=[x, y], id=4)
  ret r: bits[1] = xor_reduce(sel.4, id=5)
}"#,
        );
        let mut after = before.clone();
        let target = after.ret_node_ref.unwrap();
        ReduceSelDistributeTransform
            .apply(&mut after, &TransformLocation::Node(target))
            .unwrap();
        compact_and_toposort_in_place(&mut before).unwrap();
        compact_and_toposort_in_place(&mut after).unwrap();

        for p in 0..2 {
            for x in 0..4 {
                for y in 0..4 {
                    let values = [p, x, y];
                    let widths = [1, 2, 2];
                    assert_eq!(
                        eval_bits(&before, &values, &widths),
                        eval_bits(&after, &values, &widths)
                    );
                }
            }
        }
    }
}
