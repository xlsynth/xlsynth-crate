// SPDX-License-Identifier: Apache-2.0

use super::signbit_sub_compare::{only_sub_user_is_signbit, signbit_sub_compare_parts};
use super::*;

/// Rewrites signbit-only subtraction predicates to direct unsigned compares.
#[derive(Debug)]
pub struct SubSignbitToUnsignedCompareTransform;

impl PirTransform for SubSignbitToUnsignedCompareTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::SubSignbitToUnsignedCompare
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            let Some(parts) = signbit_sub_compare_parts(f, nr) else {
                continue;
            };
            if !only_sub_user_is_signbit(f, &parts) {
                continue;
            }
            out.push(TransformCandidate {
                location: TransformLocation::Node(nr),
                always_equivalent: parts.always_equivalent,
            });
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "SubSignbitToUnsignedCompareTransform: expected node location".to_string(),
                );
            }
        };
        let parts = signbit_sub_compare_parts(f, target).ok_or_else(|| {
            "SubSignbitToUnsignedCompareTransform: expected signbit(sub(...)) predicate".to_string()
        })?;
        if !only_sub_user_is_signbit(f, &parts) {
            return Err(
                "SubSignbitToUnsignedCompareTransform: subtract result has non-signbit users"
                    .to_string(),
            );
        }
        f.get_node_mut(target).payload =
            NodePayload::Binop(parts.polarity.binop(), parts.lhs, parts.rhs);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth::{IrBits, IrValue};
    use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
    use xlsynth_pir::ir_parser;

    fn eval_u1(f: &IrFn, x: u64, y: u64, width: usize) -> bool {
        let args = [
            IrValue::from_bits(&IrBits::make_ubits(width, x).unwrap()),
            IrValue::from_bits(&IrBits::make_ubits(width, y).unwrap()),
        ];
        let FnEvalResult::Success(result) = eval_fn(f, &args) else {
            panic!("expected value result");
        };
        let bits = result.value.to_bits().unwrap();
        bits.get_bit(0).unwrap()
    }

    #[test]
    fn direct_signbit_sub_candidate_is_oracle_backed_for_full_width_inputs() {
        let ir_text = r#"fn t(x: bits[2] id=1, y: bits[2] id=2) -> bits[1] {
  sub.3: bits[2] = sub(x, y, id=3)
  ret out: bits[1] = bit_slice(sub.3, start=1, width=1, id=4)
}"#;
        let f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let mut t = SubSignbitToUnsignedCompareTransform;
        let candidates = t.find_candidates(&f);
        assert_eq!(candidates.len(), 1);
        assert!(!candidates[0].always_equivalent);
    }

    #[test]
    fn rewritten_full_width_signbit_sub_is_not_universally_equivalent() {
        let ir_text = r#"fn t(x: bits[2] id=1, y: bits[2] id=2) -> bits[1] {
  sub.3: bits[2] = sub(x, y, id=3)
  ret out: bits[1] = bit_slice(sub.3, start=1, width=1, id=4)
}"#;
        let mut f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        let t = SubSignbitToUnsignedCompareTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();

        assert!(eval_u1(&f, 0, 1, 2));
        assert_eq!(eval_u1(&f, 2, 0, 2), false);
    }

    #[test]
    fn zero_extended_signbit_sub_candidate_is_always_equivalent() {
        let ir_text = r#"fn t(x: bits[3] id=1, y: bits[3] id=2) -> bits[1] {
  zero_ext.3: bits[4] = zero_ext(x, new_bit_count=4, id=3)
  zero_ext.4: bits[4] = zero_ext(y, new_bit_count=4, id=4)
  sub.5: bits[4] = sub(zero_ext.3, zero_ext.4, id=5)
  ret out: bits[1] = bit_slice(sub.5, start=3, width=1, id=6)
}"#;
        let f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let mut t = SubSignbitToUnsignedCompareTransform;
        let candidates = t.find_candidates(&f);
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].always_equivalent);
    }

    #[test]
    fn rewrites_direct_and_inverted_signbit_predicates() {
        let direct_ir = r#"fn t(x: bits[3] id=1, y: bits[3] id=2) -> bits[1] {
  zero_ext.3: bits[4] = zero_ext(x, new_bit_count=4, id=3)
  zero_ext.4: bits[4] = zero_ext(y, new_bit_count=4, id=4)
  sub.5: bits[4] = sub(zero_ext.3, zero_ext.4, id=5)
  ret out: bits[1] = bit_slice(sub.5, start=3, width=1, id=6)
}"#;
        let mut direct = ir_parser::Parser::new(direct_ir).parse_fn().unwrap();
        let direct_target = direct.ret_node_ref.unwrap();
        let t = SubSignbitToUnsignedCompareTransform;
        t.apply(&mut direct, &TransformLocation::Node(direct_target))
            .unwrap();
        assert!(matches!(
            direct.get_node(direct_target).payload,
            NodePayload::Binop(Binop::Ult, _, _)
        ));

        let inverted_ir = r#"fn t(x: bits[3] id=1, y: bits[3] id=2) -> bits[1] {
  zero_ext.3: bits[4] = zero_ext(x, new_bit_count=4, id=3)
  zero_ext.4: bits[4] = zero_ext(y, new_bit_count=4, id=4)
  sub.5: bits[4] = sub(zero_ext.3, zero_ext.4, id=5)
  bit_slice.6: bits[1] = bit_slice(sub.5, start=3, width=1, id=6)
  ret out: bits[1] = not(bit_slice.6, id=7)
}"#;
        let mut inverted = ir_parser::Parser::new(inverted_ir).parse_fn().unwrap();
        let inverted_target = inverted.ret_node_ref.unwrap();
        t.apply(&mut inverted, &TransformLocation::Node(inverted_target))
            .unwrap();
        assert!(matches!(
            inverted.get_node(inverted_target).payload,
            NodePayload::Binop(Binop::Uge, _, _)
        ));
    }
}
