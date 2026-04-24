// SPDX-License-Identifier: Apache-2.0

use super::macro_utils as mu;
use super::*;

/// Rewrites zero-arm selects into mask gating and back.
#[derive(Debug)]
pub struct SelZeroToAndMaskTransform;

impl SelZeroToAndMaskTransform {
    fn forward_parts(f: &IrFn, nr: NodeRef) -> Option<(NodeRef, NodeRef, bool, usize)> {
        let (selector, case0, case1) = mu::sel2_parts(f, nr)?;
        let width = mu::bits_width(f, nr)?;
        if width == 0 || !mu::is_u1(f, selector) {
            return None;
        }
        if mu::bits_width(f, case0) != Some(width) || mu::bits_width(f, case1) != Some(width) {
            return None;
        }
        if mu::is_zero_literal(f, case0, width) {
            Some((selector, case1, false, width))
        } else if mu::is_zero_literal(f, case1, width) {
            Some((selector, case0, true, width))
        } else {
            None
        }
    }

    fn reverse_parts(f: &IrFn, nr: NodeRef) -> Option<(NodeRef, NodeRef, bool, usize)> {
        let width = mu::bits_width(f, nr)?;
        if width == 0 {
            return None;
        }
        let NodePayload::Nary(NaryOp::And, ops) = &f.get_node(nr).payload else {
            return None;
        };
        if ops.len() != 2 {
            return None;
        }
        for (x, mask) in [(ops[0], ops[1]), (ops[1], ops[0])] {
            if mu::bits_width(f, x) != Some(width) {
                continue;
            }
            let NodePayload::SignExt { arg, new_bit_count } = f.get_node(mask).payload else {
                continue;
            };
            if new_bit_count != width || !mu::is_bits_w(f, mask, width) {
                continue;
            }
            if let NodePayload::Unop(Unop::Not, pred) = f.get_node(arg).payload {
                if mu::is_u1(f, pred) {
                    return Some((pred, x, true, width));
                }
            }
            if mu::is_u1(f, arg) {
                return Some((arg, x, false, width));
            }
        }
        None
    }
}

impl PirTransform for SelZeroToAndMaskTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::SelZeroToAndMask
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
                return Err("SelZeroToAndMaskTransform: expected node location".to_string());
            }
        };
        if let Some((selector, x, invert, width)) = Self::forward_parts(f, target) {
            let pred = if invert {
                mu::mk_unop(f, Unop::Not, Type::Bits(1), selector)
            } else {
                selector
            };
            let mask = mu::mk_sign_ext_mask(f, pred, width);
            f.get_node_mut(target).payload = NodePayload::Nary(NaryOp::And, vec![x, mask]);
            return Ok(());
        }
        if let Some((selector, x, invert, width)) = Self::reverse_parts(f, target) {
            let zero = mu::mk_literal_ubits(f, width, 0);
            let (case0, case1) = if invert { (x, zero) } else { (zero, x) };
            f.get_node_mut(target).payload = NodePayload::Sel {
                selector,
                cases: vec![case0, case1],
                default: None,
            };
            return Ok(());
        }
        Err("SelZeroToAndMaskTransform: unsupported target".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn rewrites_both_zero_arm_polarities_and_folds_back() {
        let left_zero = r#"fn t(p: bits[1] id=1, x: bits[4] id=2) -> bits[4] {
  literal.3: bits[4] = literal(value=0, id=3)
  ret out: bits[4] = sel(p, cases=[literal.3, x], id=4)
}"#;
        let mut f = ir_parser::Parser::new(left_zero).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        let t = SelZeroToAndMaskTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Nary(NaryOp::And, _)
        ));
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        let NodePayload::Sel { cases, .. } = &f.get_node(target).payload else {
            panic!("expected sel");
        };
        assert!(matches!(
            f.get_node(cases[0]).payload,
            NodePayload::Literal(_)
        ));

        let right_zero = r#"fn t(p: bits[1] id=1, x: bits[4] id=2) -> bits[4] {
  literal.3: bits[4] = literal(value=0, id=3)
  ret out: bits[4] = sel(p, cases=[x, literal.3], id=4)
}"#;
        let mut f = ir_parser::Parser::new(right_zero).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        let NodePayload::Nary(NaryOp::And, ops) = &f.get_node(target).payload else {
            panic!("expected and");
        };
        let mask = ops
            .iter()
            .copied()
            .find(|op| !matches!(f.get_node(*op).payload, NodePayload::GetParam(_)))
            .unwrap();
        let NodePayload::SignExt { arg, .. } = f.get_node(mask).payload else {
            panic!("expected sign_ext mask");
        };
        assert!(matches!(
            f.get_node(arg).payload,
            NodePayload::Unop(Unop::Not, _)
        ));
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        let NodePayload::Sel { cases, .. } = &f.get_node(target).payload else {
            panic!("expected sel");
        };
        assert!(matches!(
            f.get_node(cases[1]).payload,
            NodePayload::Literal(_)
        ));
    }
}
