// SPDX-License-Identifier: Apache-2.0

use super::macro_utils as mu;
use super::*;

/// Converts `(1 << count) - 1` low-mask idioms to `ext_mask_low(count)`.
#[derive(Debug)]
pub struct ShiftOneMinusOneToMaskLowTransform;

impl ShiftOneMinusOneToMaskLowTransform {
    fn shift_one_count(f: &IrFn, r: NodeRef, width: usize) -> Option<NodeRef> {
        if width == 0 || !mu::is_bits_w(f, r, width) {
            return None;
        }
        let NodePayload::Binop(Binop::Shll, one, count) = f.get_node(r).payload else {
            return None;
        };
        if !mu::is_literal_one(f, one, width) || !matches!(f.get_node(count).ty, Type::Bits(_)) {
            return None;
        }
        Some(count)
    }

    fn idiom_count(f: &IrFn, nr: NodeRef) -> Option<NodeRef> {
        let width = mu::bits_width(f, nr)?;
        let NodePayload::Binop(op, lhs, rhs) = f.get_node(nr).payload else {
            return None;
        };
        match op {
            Binop::Sub if mu::is_literal_one(f, rhs, width) => Self::shift_one_count(f, lhs, width),
            Binop::Add if mu::is_literal_all_ones(f, rhs, width) => {
                Self::shift_one_count(f, lhs, width)
            }
            Binop::Add if mu::is_literal_all_ones(f, lhs, width) => {
                Self::shift_one_count(f, rhs, width)
            }
            _ => None,
        }
    }

    fn ext_mask_low_count(f: &IrFn, nr: NodeRef) -> Option<NodeRef> {
        if mu::bits_width(f, nr)? == 0 {
            return None;
        }
        let NodePayload::ExtMaskLow { count } = f.get_node(nr).payload else {
            return None;
        };
        if matches!(f.get_node(count).ty, Type::Bits(_)) {
            Some(count)
        } else {
            None
        }
    }
}

impl PirTransform for ShiftOneMinusOneToMaskLowTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ShiftOneMinusOneToMaskLow
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            if Self::idiom_count(f, nr).is_some() || Self::ext_mask_low_count(f, nr).is_some() {
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
                    "ShiftOneMinusOneToMaskLowTransform: expected node location".to_string()
                );
            }
        };
        if let Some(count) = Self::idiom_count(f, target) {
            f.get_node_mut(target).payload = NodePayload::ExtMaskLow { count };
            return Ok(());
        }
        if let Some(count) = Self::ext_mask_low_count(f, target) {
            let width = mu::bits_width(f, target).ok_or_else(|| {
                "ShiftOneMinusOneToMaskLowTransform: target must be bits".to_string()
            })?;
            let one_a = mu::mk_literal_ubits(f, width, 1);
            let one_b = mu::mk_literal_ubits(f, width, 1);
            let shll = mu::mk_binop(f, Binop::Shll, Type::Bits(width), one_a, count);
            f.get_node_mut(target).payload = NodePayload::Binop(Binop::Sub, shll, one_b);
            return Ok(());
        }
        Err("ShiftOneMinusOneToMaskLowTransform: unsupported target".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn rewrites_sub_shift_one_minus_one_to_ext_mask_low() {
        let ir_text = r#"fn t(count: bits[4] id=1) -> bits[8] {
  one: bits[8] = literal(value=1, id=2)
  sh: bits[8] = shll(one, count, id=3)
  one2: bits[8] = literal(value=1, id=4)
  ret out: bits[8] = sub(sh, one2, id=5)
}"#;
        let mut f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        let t = ShiftOneMinusOneToMaskLowTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::ExtMaskLow { .. }
        ));
    }

    #[test]
    fn reverses_ext_mask_low_to_sub_shift_one_minus_one() {
        let ir_text = r#"fn t(count: bits[4] id=1) -> bits[8] {
  ret out: bits[8] = ext_mask_low(count, id=2)
}"#;
        let mut f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        let t = ShiftOneMinusOneToMaskLowTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Binop(Binop::Sub, _, _)
        ));
    }
}
