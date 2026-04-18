// SPDX-License-Identifier: Apache-2.0

use super::macro_utils as mu;
use super::*;

const MAX_NORMALIZATION_CASES: usize = 32;

/// Converts bounded CLZ-driven normalization shifts into explicit candidates.
#[derive(Debug)]
pub struct BoundedNormalizationToCandidateSelectTransform;

impl BoundedNormalizationToCandidateSelectTransform {
    fn clz_of_arg(f: &IrFn, amount: NodeRef) -> Option<NodeRef> {
        match f.get_node(amount).payload {
            NodePayload::ExtClz { arg } => Some(arg),
            NodePayload::Encode { arg: one_hot } => {
                let NodePayload::OneHot {
                    arg: reversed,
                    lsb_prio: true,
                } = f.get_node(one_hot).payload
                else {
                    return None;
                };
                let NodePayload::Unop(Unop::Reverse, x) = f.get_node(reversed).payload else {
                    return None;
                };
                Some(x)
            }
            _ => None,
        }
    }

    fn candidate_parts(f: &IrFn, nr: NodeRef) -> Option<(Binop, NodeRef, NodeRef, usize, usize)> {
        let NodePayload::OneHotSel { selector, cases } = &f.get_node(nr).payload else {
            return None;
        };
        let NodePayload::Decode { arg: amount, width } = f.get_node(*selector).payload else {
            return None;
        };
        if cases.len() != width || cases.is_empty() || cases.len() > MAX_NORMALIZATION_CASES {
            return None;
        }
        let x = Self::clz_of_arg(f, amount)?;
        let x_width = mu::bits_width(f, x)?;
        if cases.len() != x_width + 1 {
            return None;
        }
        let mut op: Option<Binop> = None;
        for (i, case) in cases.iter().enumerate() {
            let NodePayload::Binop(case_op, case_x, lit) = f.get_node(*case).payload else {
                return None;
            };
            if case_op != Binop::Shll && case_op != Binop::Shrl {
                return None;
            }
            if case_x != x || mu::literal_usize(f, lit)? != i {
                return None;
            }
            if let Some(prev) = op {
                if prev != case_op {
                    return None;
                }
            } else {
                op = Some(case_op);
            }
        }
        Some((op?, x, amount, x_width, mu::bits_width(f, amount)?))
    }
}

impl PirTransform for BoundedNormalizationToCandidateSelectTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::BoundedNormalizationToCandidateSelect
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            match f.get_node(nr).payload {
                NodePayload::Binop(op @ (Binop::Shll | Binop::Shrl), x, amount) => {
                    let Some(clz_x) = Self::clz_of_arg(f, amount) else {
                        continue;
                    };
                    let Some(x_width) = mu::bits_width(f, x) else {
                        continue;
                    };
                    if x == clz_x
                        && mu::bits_width(f, nr) == Some(x_width)
                        && x_width + 1 <= MAX_NORMALIZATION_CASES
                        && mu::bits_width(f, amount).is_some()
                        && matches!(op, Binop::Shll | Binop::Shrl)
                    {
                        out.push(TransformCandidate {
                            location: TransformLocation::Node(nr),
                            always_equivalent: true,
                        });
                    }
                }
                NodePayload::OneHotSel { .. } => {
                    if Self::candidate_parts(f, nr).is_some() {
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
                    "BoundedNormalizationToCandidateSelectTransform: expected node location"
                        .to_string(),
                );
            }
        };
        match f.get_node(target).payload.clone() {
            NodePayload::Binop(op @ (Binop::Shll | Binop::Shrl), x, amount) => {
                let clz_x = Self::clz_of_arg(f, amount).ok_or_else(|| {
                    "BoundedNormalizationToCandidateSelectTransform: amount is not clz idiom"
                        .to_string()
                })?;
                let x_width = mu::bits_width(f, x).ok_or_else(|| {
                    "BoundedNormalizationToCandidateSelectTransform: x must be bits".to_string()
                })?;
                let amount_width = mu::bits_width(f, amount).ok_or_else(|| {
                    "BoundedNormalizationToCandidateSelectTransform: amount must be bits"
                        .to_string()
                })?;
                if x != clz_x || x_width + 1 > MAX_NORMALIZATION_CASES {
                    return Err("BoundedNormalizationToCandidateSelectTransform: invalid normalization shape".to_string());
                }
                let decode = mu::mk_decode(f, amount, x_width + 1);
                let cases = (0..=x_width)
                    .map(|shift| mu::constant_shift(f, op, x, x_width, amount_width, shift))
                    .collect();
                f.get_node_mut(target).payload = NodePayload::OneHotSel {
                    selector: decode,
                    cases,
                };
                Ok(())
            }
            NodePayload::OneHotSel { .. } => {
                let (op, x, amount, x_width, _amount_width) =
                    Self::candidate_parts(f, target).ok_or_else(|| {
                        "BoundedNormalizationToCandidateSelectTransform: expected normalization candidate select".to_string()
                    })?;
                f.get_node_mut(target).payload = NodePayload::Binop(op, x, amount);
                if mu::bits_width(f, target) != Some(x_width) {
                    return Err(
                        "BoundedNormalizationToCandidateSelectTransform: target width mismatch"
                            .to_string(),
                    );
                }
                Ok(())
            }
            _ => Err(
                "BoundedNormalizationToCandidateSelectTransform: unsupported target".to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn expands_ext_clz_shift_to_candidates() {
        let ir_text = r#"fn t(x: bits[4] id=1) -> bits[4] {
  c: bits[3] = ext_clz(x, id=2)
  ret out: bits[4] = shll(x, c, id=3)
}"#;
        let mut f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        let t = BoundedNormalizationToCandidateSelectTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::OneHotSel { .. }
        ));
    }
}
