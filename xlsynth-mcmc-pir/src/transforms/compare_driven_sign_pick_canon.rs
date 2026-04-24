// SPDX-License-Identifier: Apache-2.0

use super::macro_utils as mu;
use super::signbit_sub_compare::signbit_sub_compare_parts;
use super::*;

/// Rewrites signbit-sub-driven selection to direct compare-driven selection.
#[derive(Debug)]
pub struct CompareDrivenSignPickCanonTransform;

impl CompareDrivenSignPickCanonTransform {
    fn case_matches_sub_pair(f: &IrFn, nr: NodeRef, lhs: NodeRef, rhs: NodeRef) -> bool {
        matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Sub, a, b) if a == lhs && b == rhs)
    }

    fn candidate_parts(f: &IrFn, nr: NodeRef) -> Option<(NodeRef, NodeRef, bool)> {
        let NodePayload::Sel {
            selector,
            cases,
            default,
        } = &f.get_node(nr).payload
        else {
            return None;
        };
        if default.is_some() || cases.len() != 2 {
            return None;
        }
        let parts = signbit_sub_compare_parts(f, *selector)?;
        let direct_xy = (cases[0] == parts.lhs && cases[1] == parts.rhs)
            || (cases[0] == parts.rhs && cases[1] == parts.lhs);
        let rev_sub = (
            Self::case_matches_sub_pair(f, cases[0], parts.lhs, parts.rhs)
                && Self::case_matches_sub_pair(f, cases[1], parts.rhs, parts.lhs),
            Self::case_matches_sub_pair(f, cases[0], parts.rhs, parts.lhs)
                && Self::case_matches_sub_pair(f, cases[1], parts.lhs, parts.rhs),
        );
        if direct_xy || rev_sub.0 || rev_sub.1 {
            Some((parts.lhs, parts.rhs, parts.always_equivalent))
        } else {
            None
        }
    }
}

impl PirTransform for CompareDrivenSignPickCanonTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::CompareDrivenSignPickCanon
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            let Some((_lhs, _rhs, always_equivalent)) = Self::candidate_parts(f, nr) else {
                continue;
            };
            out.push(TransformCandidate {
                location: TransformLocation::Node(nr),
                always_equivalent,
            });
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "CompareDrivenSignPickCanonTransform: expected node location".to_string(),
                );
            }
        };
        let NodePayload::Sel {
            selector,
            cases,
            default,
        } = f.get_node(target).payload.clone()
        else {
            return Err("CompareDrivenSignPickCanonTransform: expected sel".to_string());
        };
        if default.is_some() || cases.len() != 2 {
            return Err("CompareDrivenSignPickCanonTransform: expected 2-case sel".to_string());
        }
        let parts = signbit_sub_compare_parts(f, selector).ok_or_else(|| {
            "CompareDrivenSignPickCanonTransform: selector must match signbit(sub(...))".to_string()
        })?;
        Self::candidate_parts(f, target).ok_or_else(|| {
            "CompareDrivenSignPickCanonTransform: unsupported case shape".to_string()
        })?;
        f.get_node_mut(target).payload = NodePayload::Sel {
            selector: mu::mk_binop(
                f,
                parts.polarity.binop(),
                Type::Bits(1),
                parts.lhs,
                parts.rhs,
            ),
            cases,
            default: None,
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn rewrites_only_selector_for_operand_pick_shape() {
        let ir_text = r#"fn t(x: bits[3] id=1, y: bits[3] id=2) -> bits[4] {
  zero_ext.3: bits[4] = zero_ext(x, new_bit_count=4, id=3)
  zero_ext.4: bits[4] = zero_ext(y, new_bit_count=4, id=4)
  sub.5: bits[4] = sub(zero_ext.3, zero_ext.4, id=5)
  bit_slice.6: bits[1] = bit_slice(sub.5, start=3, width=1, id=6)
  ret out: bits[4] = sel(bit_slice.6, cases=[zero_ext.3, zero_ext.4], id=7)
}"#;
        let mut f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        let original_cases = match &f.get_node(target).payload {
            NodePayload::Sel { cases, .. } => cases.clone(),
            _ => unreachable!(),
        };
        let t = CompareDrivenSignPickCanonTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        let NodePayload::Sel {
            selector, cases, ..
        } = &f.get_node(target).payload
        else {
            panic!("expected sel");
        };
        assert_eq!(*cases, original_cases);
        assert!(matches!(
            f.get_node(*selector).payload,
            NodePayload::Binop(Binop::Ult, _, _)
        ));
    }
}
