// SPDX-License-Identifier: Apache-2.0

use super::macro_utils as mu;
use super::*;

const MAX_STATIC_SHIFT_CASES: usize = 16;

/// Converts bounded dynamic right-shift slices to static slice candidates.
#[derive(Debug)]
pub struct BoundedDynamicShiftToStaticCandidatesTransform;

impl BoundedDynamicShiftToStaticCandidatesTransform {
    fn amount_bound(f: &IrFn, amount: NodeRef) -> Option<usize> {
        let amount = mu::unwrap_identity(f, amount);
        let amount_bits = mu::bits_width(f, amount)?;
        if amount_bits < usize::BITS as usize {
            let cases = 1usize.checked_shl(amount_bits as u32)?;
            if cases <= MAX_STATIC_SHIFT_CASES {
                return Some(cases.saturating_sub(1));
            }
        }
        Self::canonical_clamp_bound(f, amount)
    }

    fn canonical_clamp_bound(f: &IrFn, amount: NodeRef) -> Option<usize> {
        let NodePayload::Sel {
            selector,
            cases,
            default,
        } = &f.get_node(amount).payload
        else {
            return None;
        };
        if default.is_some() || cases.len() != 2 {
            return None;
        }
        let NodePayload::Binop(Binop::Ult, x, max_lit) = f.get_node(*selector).payload else {
            return None;
        };
        if cases[0] != max_lit || cases[1] != x {
            return None;
        }
        mu::literal_usize(f, max_lit)
    }

    fn slices_in_bounds(x_width: usize, start: usize, width: usize, max_amount: usize) -> bool {
        start
            .checked_add(max_amount)
            .and_then(|v| v.checked_add(width))
            .is_some_and(|limit| limit <= x_width)
    }

    fn static_candidate_parts(
        f: &IrFn,
        nr: NodeRef,
    ) -> Option<(NodeRef, NodeRef, usize, usize, usize)> {
        let NodePayload::OneHotSel { selector, cases } = &f.get_node(nr).payload else {
            return None;
        };
        let NodePayload::Decode { arg: amount, width } = f.get_node(*selector).payload else {
            return None;
        };
        if width != cases.len() || cases.is_empty() || cases.len() > MAX_STATIC_SHIFT_CASES {
            return None;
        }
        let max_amount = cases.len() - 1;
        if Self::amount_bound(f, amount)? != max_amount {
            return None;
        }
        let (x, start, slice_width) = mu::bit_slice_parts(f, cases[0])?;
        for (i, case) in cases.iter().enumerate() {
            let (case_x, case_start, case_width) = mu::bit_slice_parts(f, *case)?;
            if case_x != x || case_start != start + i || case_width != slice_width {
                return None;
            }
        }
        let x_width = mu::bits_width(f, x)?;
        if !Self::slices_in_bounds(x_width, start, slice_width, max_amount) {
            return None;
        }
        Some((x, amount, start, slice_width, x_width))
    }
}

impl PirTransform for BoundedDynamicShiftToStaticCandidatesTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::BoundedDynamicShiftToStaticCandidates
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::BitSlice { arg, start, width } => {
                    let NodePayload::Binop(Binop::Shrl, x, amount) = f.get_node(*arg).payload
                    else {
                        continue;
                    };
                    let Some(x_width) = mu::bits_width(f, x) else {
                        continue;
                    };
                    let Some(max_amount) = Self::amount_bound(f, amount) else {
                        continue;
                    };
                    if max_amount >= MAX_STATIC_SHIFT_CASES {
                        continue;
                    }
                    if mu::bits_width(f, *arg) == Some(x_width)
                        && mu::bits_width(f, nr) == Some(*width)
                        && Self::slices_in_bounds(x_width, *start, *width, max_amount)
                    {
                        out.push(TransformCandidate {
                            location: TransformLocation::Node(nr),
                            always_equivalent: true,
                        });
                    }
                }
                NodePayload::OneHotSel { .. } => {
                    if Self::static_candidate_parts(f, nr).is_some() {
                        out.push(TransformCandidate {
                            location: TransformLocation::Node(nr),
                            always_equivalent: true,
                        });
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "BoundedDynamicShiftToStaticCandidatesTransform: expected node location"
                        .to_string(),
                );
            }
        };
        match f.get_node(target).payload.clone() {
            NodePayload::BitSlice { arg, start, width } => {
                let NodePayload::Binop(Binop::Shrl, x, amount) = f.get_node(arg).payload else {
                    return Err("BoundedDynamicShiftToStaticCandidatesTransform: expected bit_slice(shrl(...))".to_string());
                };
                let x_width = mu::bits_width(f, x).ok_or_else(|| {
                    "BoundedDynamicShiftToStaticCandidatesTransform: x must be bits".to_string()
                })?;
                let max_amount = Self::amount_bound(f, amount).ok_or_else(|| {
                    "BoundedDynamicShiftToStaticCandidatesTransform: amount not bounded".to_string()
                })?;
                if max_amount >= MAX_STATIC_SHIFT_CASES
                    || !Self::slices_in_bounds(x_width, start, width, max_amount)
                {
                    return Err("BoundedDynamicShiftToStaticCandidatesTransform: shift expansion out of bounds".to_string());
                }
                let decode = mu::mk_decode(f, amount, max_amount + 1);
                let cases = (0..=max_amount)
                    .map(|shift| mu::mk_bit_slice(f, x, start + shift, width))
                    .collect();
                f.get_node_mut(target).payload = NodePayload::OneHotSel {
                    selector: decode,
                    cases,
                };
                Ok(())
            }
            NodePayload::OneHotSel { .. } => {
                let (x, amount, start, width, x_width) =
                    Self::static_candidate_parts(f, target).ok_or_else(|| {
                        "BoundedDynamicShiftToStaticCandidatesTransform: expected static candidate select".to_string()
                    })?;
                let shrl = mu::mk_binop(f, Binop::Shrl, Type::Bits(x_width), x, amount);
                f.get_node_mut(target).payload = NodePayload::BitSlice {
                    arg: shrl,
                    start,
                    width,
                };
                Ok(())
            }
            _ => Err(
                "BoundedDynamicShiftToStaticCandidatesTransform: unsupported target".to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn expands_and_folds_small_dynamic_shift_slice() {
        let ir_text = r#"fn t(x: bits[8] id=1, amount: bits[2] id=2) -> bits[3] {
  sh: bits[8] = shrl(x, amount, id=3)
  ret out: bits[3] = bit_slice(sh, start=1, width=3, id=4)
}"#;
        let mut f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        let t = BoundedDynamicShiftToStaticCandidatesTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::OneHotSel { .. }
        ));
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::BitSlice { .. }
        ));
    }

    #[test]
    fn rejects_static_candidates_when_decode_does_not_cover_amount_bound() {
        let ir_text = r#"fn t(x: bits[8] id=1, amount: bits[4] id=2) -> bits[4] {
  decode.3: bits[4] = decode(amount, width=4, id=3)
  s0: bits[4] = bit_slice(x, start=0, width=4, id=4)
  s1: bits[4] = bit_slice(x, start=1, width=4, id=5)
  s2: bits[4] = bit_slice(x, start=2, width=4, id=6)
  s3: bits[4] = bit_slice(x, start=3, width=4, id=7)
  ret out: bits[4] = one_hot_sel(decode.3, cases=[s0, s1, s2, s3], id=8)
}"#;
        let f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        assert!(
            BoundedDynamicShiftToStaticCandidatesTransform::static_candidate_parts(&f, target)
                .is_none()
        );
    }

    #[test]
    fn rejects_huge_clamp_bound_without_overflowing_case_count() {
        let ir_text = r#"fn t(x: bits[128] id=1, amount: bits[64] id=2) -> bits[8] {
  max: bits[64] = literal(value=18446744073709551615, id=3)
  pred: bits[1] = ult(amount, max, id=4)
  clamped: bits[64] = sel(pred, cases=[max, amount], id=5)
  sh: bits[128] = shrl(x, clamped, id=6)
  ret out: bits[8] = bit_slice(sh, start=0, width=8, id=7)
}"#;
        let f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let mut t = BoundedDynamicShiftToStaticCandidatesTransform;
        assert!(t.find_candidates(&f).is_empty());
    }
}
