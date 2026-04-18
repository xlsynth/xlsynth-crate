// SPDX-License-Identifier: Apache-2.0

use super::macro_utils as mu;
use super::*;

/// Converts one-hot select into explicit masked candidates and back.
#[derive(Debug)]
pub struct OneHotSelToMaskedOrTransform;

impl OneHotSelToMaskedOrTransform {
    fn candidate_from_masked_or(f: &IrFn, nr: NodeRef) -> Option<()> {
        let width = mu::bits_width(f, nr)?;
        if width == 0 {
            return None;
        }
        let operands = mu::or_operands(f, nr)?;
        if operands.is_empty() {
            return None;
        }
        Self::decode_masked_operands(f, &operands, width).map(|_| ())
    }

    fn decode_masked_operands(
        f: &IrFn,
        operands: &[NodeRef],
        width: usize,
    ) -> Option<(NodeRef, Vec<NodeRef>)> {
        let mut by_index: Vec<(usize, NodeRef)> = Vec::with_capacity(operands.len());
        let mut selector: Option<NodeRef> = None;
        for operand in operands {
            let (case, pred) = mu::masked_case_parts(f, *operand, width)?;
            let (sel, bit_index) = mu::selector_bit(f, pred)?;
            if let Some(prev) = selector {
                if prev != sel {
                    return None;
                }
            } else {
                selector = Some(sel);
            }
            by_index.push((bit_index, case));
        }
        by_index.sort_by_key(|(bit_index, _case)| *bit_index);
        for (expected, (bit_index, _case)) in by_index.iter().enumerate() {
            if *bit_index != expected {
                return None;
            }
        }
        let selector = selector?;
        if mu::bits_width(f, selector) != Some(by_index.len()) {
            return None;
        }
        Some((
            selector,
            by_index.into_iter().map(|(_, case)| case).collect(),
        ))
    }
}

impl PirTransform for OneHotSelToMaskedOrTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::OneHotSelToMaskedOr
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::OneHotSel { selector, cases } => {
                    let Some(width) = mu::bits_width(f, nr) else {
                        continue;
                    };
                    if width == 0
                        || cases.is_empty()
                        || mu::bits_width(f, *selector) != Some(cases.len())
                    {
                        continue;
                    }
                    if cases
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
                    if Self::candidate_from_masked_or(f, nr).is_some() {
                        out.push(TransformCandidate {
                            location: TransformLocation::Node(nr),
                            always_equivalent: true,
                        });
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
                return Err("OneHotSelToMaskedOrTransform: expected node location".to_string());
            }
        };
        let width = mu::bits_width(f, target)
            .ok_or_else(|| "OneHotSelToMaskedOrTransform: target must be bits".to_string())?;
        let payload = f.get_node(target).payload.clone();
        match payload {
            NodePayload::OneHotSel { selector, cases } => {
                if width == 0
                    || cases.is_empty()
                    || mu::bits_width(f, selector) != Some(cases.len())
                {
                    return Err(
                        "OneHotSelToMaskedOrTransform: invalid one_hot_sel shape".to_string()
                    );
                }
                let mut masked = Vec::with_capacity(cases.len());
                for (i, case) in cases.into_iter().enumerate() {
                    if mu::bits_width(f, case) != Some(width) {
                        return Err("OneHotSelToMaskedOrTransform: case width mismatch".to_string());
                    }
                    let bit = mu::mk_bit_slice(f, selector, i, 1);
                    let mask = mu::mk_sign_ext_mask(f, bit, width);
                    masked.push(mu::mk_nary_and(f, Type::Bits(width), vec![case, mask]));
                }
                f.get_node_mut(target).payload = NodePayload::Nary(NaryOp::Or, masked);
                Ok(())
            }
            _ => {
                let operands = mu::or_operands(f, target).ok_or_else(|| {
                    "OneHotSelToMaskedOrTransform: expected masked OR".to_string()
                })?;
                let (selector, cases) = Self::decode_masked_operands(f, &operands, width)
                    .ok_or_else(|| {
                        "OneHotSelToMaskedOrTransform: masked OR did not match exact one_hot shape"
                            .to_string()
                    })?;
                f.get_node_mut(target).payload = NodePayload::OneHotSel { selector, cases };
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
    fn expands_and_folds_one_hot_sel() {
        let ir_text = r#"fn t(sel: bits[2] id=1, a: bits[4] id=2, b: bits[4] id=3) -> bits[4] {
  ret oh: bits[4] = one_hot_sel(sel, cases=[a, b], id=4)
}"#;
        let mut f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        let t = OneHotSelToMaskedOrTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Nary(NaryOp::Or, _)
        ));
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::OneHotSel { .. }
        ));
    }
}
