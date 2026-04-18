// SPDX-License-Identifier: Apache-2.0

use super::macro_utils as mu;
use super::*;

/// Shares the datapath for a selected add/sub pair.
#[derive(Debug)]
pub struct SelectedAddSubUnificationTransform;

impl SelectedAddSubUnificationTransform {
    pub(super) fn build_unified_add_sub(
        f: &mut IrFn,
        pred_sub: NodeRef,
        a: NodeRef,
        b: NodeRef,
        width: usize,
    ) -> NodeRef {
        let mask = mu::mk_sign_ext_mask(f, pred_sub, width);
        let b2 = mu::push_node(
            f,
            Type::Bits(width),
            NodePayload::Nary(NaryOp::Xor, vec![b, mask]),
        );
        let carry = mu::mk_zero_ext(f, pred_sub, width);
        let inner = mu::mk_binop(f, Binop::Add, Type::Bits(width), a, b2);
        mu::mk_binop(f, Binop::Add, Type::Bits(width), inner, carry)
    }

    fn sel_add_sub_parts(
        f: &IrFn,
        nr: NodeRef,
    ) -> Option<(NodeRef, NodeRef, NodeRef, bool, usize)> {
        let NodePayload::Sel {
            selector,
            cases,
            default,
        } = &f.get_node(nr).payload
        else {
            return None;
        };
        if default.is_some() || cases.len() != 2 || !mu::is_u1(f, *selector) {
            return None;
        }
        let NodePayload::Binop(op0, a0, b0) = f.get_node(cases[0]).payload else {
            return None;
        };
        let NodePayload::Binop(op1, a1, b1) = f.get_node(cases[1]).payload else {
            return None;
        };
        if a0 != a1 || b0 != b1 {
            return None;
        }
        let width = mu::bits_width(f, nr)?;
        if mu::bits_width(f, a0) != Some(width) || mu::bits_width(f, b0) != Some(width) {
            return None;
        }
        match (op0, op1) {
            (Binop::Add, Binop::Sub) => Some((*selector, a0, b0, false, width)),
            (Binop::Sub, Binop::Add) => Some((*selector, a0, b0, true, width)),
            _ => None,
        }
    }

    fn unified_parts(f: &IrFn, nr: NodeRef) -> Option<(NodeRef, NodeRef, NodeRef, usize)> {
        let width = mu::bits_width(f, nr)?;
        let NodePayload::Binop(Binop::Add, inner, carry) = f.get_node(nr).payload else {
            return None;
        };
        let NodePayload::ZeroExt {
            arg: pred,
            new_bit_count,
        } = f.get_node(carry).payload
        else {
            return None;
        };
        if new_bit_count != width || !mu::is_u1(f, pred) {
            return None;
        }
        let NodePayload::Binop(Binop::Add, a, b2) = f.get_node(inner).payload else {
            return None;
        };
        let NodePayload::Nary(NaryOp::Xor, xor_ops) = &f.get_node(b2).payload else {
            return None;
        };
        if xor_ops.len() != 2 {
            return None;
        }
        for (b, mask) in [(xor_ops[0], xor_ops[1]), (xor_ops[1], xor_ops[0])] {
            if mu::sign_ext_mask_bit(f, mask, width) == Some(pred)
                && mu::bits_width(f, a) == Some(width)
                && mu::bits_width(f, b) == Some(width)
            {
                return Some((pred, a, b, width));
            }
        }
        None
    }
}

impl PirTransform for SelectedAddSubUnificationTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::SelectedAddSubUnification
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            if Self::sel_add_sub_parts(f, nr).is_some() || Self::unified_parts(f, nr).is_some() {
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
                return Err(
                    "SelectedAddSubUnificationTransform: expected node location".to_string()
                );
            }
        };
        if let Some((selector, a, b, reversed, width)) = Self::sel_add_sub_parts(f, target) {
            let pred_sub = if reversed {
                mu::mk_unop(f, Unop::Not, Type::Bits(1), selector)
            } else {
                selector
            };
            let unified = Self::build_unified_add_sub(f, pred_sub, a, b, width);
            f.get_node_mut(target).payload = NodePayload::Unop(Unop::Identity, unified);
            return Ok(());
        }
        if let Some((pred, a, b, width)) = Self::unified_parts(f, target) {
            let add = mu::mk_binop(f, Binop::Add, Type::Bits(width), a, b);
            let sub = mu::mk_binop(f, Binop::Sub, Type::Bits(width), a, b);
            f.get_node_mut(target).payload = NodePayload::Sel {
                selector: pred,
                cases: vec![add, sub],
                default: None,
            };
            return Ok(());
        }
        Err("SelectedAddSubUnificationTransform: unsupported target".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn unifies_selected_add_sub() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  add.4: bits[8] = add(a, b, id=4)
  sub.5: bits[8] = sub(a, b, id=5)
  ret out: bits[8] = sel(p, cases=[add.4, sub.5], id=6)
}"#;
        let mut f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        let t = SelectedAddSubUnificationTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Unop(Unop::Identity, _)
        ));
    }
}
