// SPDX-License-Identifier: Apache-2.0

use super::macro_utils as mu;
use super::*;

/// Converts priority_sel into explicit mutually-exclusive masked candidates.
#[derive(Debug)]
pub struct PrioritySelToMaskedCandidatesTransform;

impl PrioritySelToMaskedCandidatesTransform {
    fn mk_case_pred(f: &mut IrFn, selector: NodeRef, i: usize) -> NodeRef {
        let this_bit = mu::mk_bit_slice(f, selector, i, 1);
        if i == 0 {
            return this_bit;
        }
        let prior = mu::mk_bit_slice(f, selector, 0, i);
        let prior_any = mu::mk_unop(f, Unop::OrReduce, Type::Bits(1), prior);
        let no_prior = mu::mk_unop(f, Unop::Not, Type::Bits(1), prior_any);
        mu::mk_nary_and(f, Type::Bits(1), vec![this_bit, no_prior])
    }

    fn mk_default_pred(f: &mut IrFn, selector: NodeRef) -> NodeRef {
        let any = mu::mk_unop(f, Unop::OrReduce, Type::Bits(1), selector);
        mu::mk_unop(f, Unop::Not, Type::Bits(1), any)
    }

    fn case_pred_parts(f: &IrFn, pred: NodeRef) -> Option<(NodeRef, usize)> {
        if let Some((selector, bit_index)) = mu::selector_bit(f, pred) {
            return (bit_index == 0).then_some((selector, 0));
        }
        let NodePayload::Nary(NaryOp::And, ops) = &f.get_node(pred).payload else {
            return None;
        };
        if ops.len() != 2 {
            return None;
        }
        for (this_bit, no_prior) in [(ops[0], ops[1]), (ops[1], ops[0])] {
            let (selector, bit_index) = mu::selector_bit(f, this_bit)?;
            if bit_index == 0 {
                continue;
            }
            let NodePayload::Unop(Unop::Not, prior_any) = f.get_node(no_prior).payload else {
                continue;
            };
            let NodePayload::Unop(Unop::OrReduce, prior_slice) = f.get_node(prior_any).payload
            else {
                continue;
            };
            let (prior_selector, start, width) = mu::bit_slice_parts(f, prior_slice)?;
            if prior_selector == selector && start == 0 && width == bit_index {
                return Some((selector, bit_index));
            }
        }
        None
    }

    fn default_pred_selector(f: &IrFn, pred: NodeRef) -> Option<NodeRef> {
        let NodePayload::Unop(Unop::Not, any) = f.get_node(pred).payload else {
            return None;
        };
        let NodePayload::Unop(Unop::OrReduce, selector) = f.get_node(any).payload else {
            return None;
        };
        Some(selector)
    }

    fn decode_masked_operands(
        f: &IrFn,
        operands: &[NodeRef],
        width: usize,
    ) -> Option<(NodeRef, Vec<NodeRef>, NodeRef)> {
        if operands.len() < 2 {
            return None;
        }
        let mut cases_by_index: Vec<(usize, NodeRef)> = Vec::new();
        let mut selector: Option<NodeRef> = None;
        let mut default: Option<NodeRef> = None;
        for operand in operands {
            let (case, pred) = mu::masked_case_parts(f, *operand, width)?;
            if let Some((sel, bit_index)) = Self::case_pred_parts(f, pred) {
                if let Some(prev) = selector {
                    if prev != sel {
                        return None;
                    }
                } else {
                    selector = Some(sel);
                }
                cases_by_index.push((bit_index, case));
            } else {
                let sel = Self::default_pred_selector(f, pred)?;
                if let Some(prev) = selector {
                    if prev != sel {
                        return None;
                    }
                } else {
                    selector = Some(sel);
                }
                if default.replace(case).is_some() {
                    return None;
                }
            }
        }
        let selector = selector?;
        let default = default?;
        cases_by_index.sort_by_key(|(bit_index, _case)| *bit_index);
        for (expected, (bit_index, _case)) in cases_by_index.iter().enumerate() {
            if *bit_index != expected {
                return None;
            }
        }
        if mu::bits_width(f, selector) != Some(cases_by_index.len()) {
            return None;
        }
        Some((
            selector,
            cases_by_index.into_iter().map(|(_, case)| case).collect(),
            default,
        ))
    }
}

impl PirTransform for PrioritySelToMaskedCandidatesTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::PrioritySelToMaskedCandidates
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::PrioritySel {
                    selector,
                    cases,
                    default: Some(default),
                } => {
                    let Some(width) = mu::bits_width(f, nr) else {
                        continue;
                    };
                    if width == 0
                        || cases.is_empty()
                        || mu::bits_width(f, *selector) != Some(cases.len())
                    {
                        continue;
                    }
                    if mu::bits_width(f, *default) == Some(width)
                        && cases
                            .iter()
                            .all(|case| mu::bits_width(f, *case) == Some(width))
                    {
                        out.push(TransformCandidate {
                            location: TransformLocation::Node(nr),
                            always_equivalent: true,
                        });
                    }
                }
                _ => {
                    let Some(width) = mu::bits_width(f, nr) else {
                        continue;
                    };
                    if let Some(ops) = mu::or_operands(f, nr) {
                        if Self::decode_masked_operands(f, &ops, width).is_some() {
                            out.push(TransformCandidate {
                                location: TransformLocation::Node(nr),
                                always_equivalent: true,
                            });
                        }
                    }
                }
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "PrioritySelToMaskedCandidatesTransform: expected node location".to_string(),
                );
            }
        };
        let width = mu::bits_width(f, target).ok_or_else(|| {
            "PrioritySelToMaskedCandidatesTransform: target must be bits".to_string()
        })?;
        match f.get_node(target).payload.clone() {
            NodePayload::PrioritySel {
                selector,
                cases,
                default: Some(default),
            } => {
                if width == 0
                    || cases.is_empty()
                    || mu::bits_width(f, selector) != Some(cases.len())
                {
                    return Err(
                        "PrioritySelToMaskedCandidatesTransform: invalid priority_sel shape"
                            .to_string(),
                    );
                }
                let mut masked = Vec::with_capacity(cases.len() + 1);
                for (i, case) in cases.into_iter().enumerate() {
                    if mu::bits_width(f, case) != Some(width) {
                        return Err(
                            "PrioritySelToMaskedCandidatesTransform: case width mismatch"
                                .to_string(),
                        );
                    }
                    let pred = Self::mk_case_pred(f, selector, i);
                    let mask = mu::mk_sign_ext_mask(f, pred, width);
                    masked.push(mu::mk_nary_and(f, Type::Bits(width), vec![case, mask]));
                }
                let default_pred = Self::mk_default_pred(f, selector);
                let default_mask = mu::mk_sign_ext_mask(f, default_pred, width);
                masked.push(mu::mk_nary_and(
                    f,
                    Type::Bits(width),
                    vec![default, default_mask],
                ));
                f.get_node_mut(target).payload = NodePayload::Nary(NaryOp::Or, masked);
                Ok(())
            }
            _ => {
                let ops = mu::or_operands(f, target).ok_or_else(|| {
                    "PrioritySelToMaskedCandidatesTransform: expected masked OR".to_string()
                })?;
                let (selector, cases, default) =
                    Self::decode_masked_operands(f, &ops, width).ok_or_else(|| {
                        "PrioritySelToMaskedCandidatesTransform: masked OR did not match exact priority shape".to_string()
                    })?;
                f.get_node_mut(target).payload = NodePayload::PrioritySel {
                    selector,
                    cases,
                    default: Some(default),
                };
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn expands_and_folds_priority_sel() {
        let ir_text = r#"fn t(sel: bits[2] id=1, a: bits[4] id=2, b: bits[4] id=3, d: bits[4] id=4) -> bits[4] {
  ret ps: bits[4] = priority_sel(sel, cases=[a, b], default=d, id=5)
}"#;
        let mut f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        let t = PrioritySelToMaskedCandidatesTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Nary(NaryOp::Or, _)
        ));
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::PrioritySel { .. }
        ));
    }
}
