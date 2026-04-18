// SPDX-License-Identifier: Apache-2.0

use super::macro_utils as mu;
use super::*;

/// Converts shifted all-ones masks to `not(ext_mask_low(count))`.
#[derive(Debug)]
pub struct ShiftedAllOnesToNotMaskLowTransform;

impl ShiftedAllOnesToNotMaskLowTransform {
    fn shifted_all_ones_count(f: &IrFn, nr: NodeRef) -> Option<NodeRef> {
        let width = mu::bits_width(f, nr)?;
        if width == 0 {
            return None;
        }
        let NodePayload::Binop(Binop::Shll, lhs, count) = f.get_node(nr).payload else {
            return None;
        };
        if !mu::is_literal_all_ones(f, lhs, width) || !matches!(f.get_node(count).ty, Type::Bits(_))
        {
            return None;
        }
        Some(count)
    }

    fn not_mask_low_count(f: &IrFn, nr: NodeRef) -> Option<NodeRef> {
        if mu::bits_width(f, nr)? == 0 {
            return None;
        }
        let NodePayload::Unop(Unop::Not, arg) = f.get_node(nr).payload else {
            return None;
        };
        let NodePayload::ExtMaskLow { count } = f.get_node(arg).payload else {
            return None;
        };
        if f.get_node(arg).ty == f.get_node(nr).ty && matches!(f.get_node(count).ty, Type::Bits(_))
        {
            Some(count)
        } else {
            None
        }
    }
}

impl PirTransform for ShiftedAllOnesToNotMaskLowTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ShiftedAllOnesToNotMaskLow
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            if Self::shifted_all_ones_count(f, nr).is_some()
                || Self::not_mask_low_count(f, nr).is_some()
            {
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
                    "ShiftedAllOnesToNotMaskLowTransform: expected node location".to_string(),
                );
            }
        };
        if let Some(count) = Self::shifted_all_ones_count(f, target) {
            let width = mu::bits_width(f, target).ok_or_else(|| {
                "ShiftedAllOnesToNotMaskLowTransform: target must be bits".to_string()
            })?;
            let mask = mu::mk_ext_mask_low(f, count, width);
            f.get_node_mut(target).payload = NodePayload::Unop(Unop::Not, mask);
            return Ok(());
        }
        if let Some(count) = Self::not_mask_low_count(f, target) {
            let width = mu::bits_width(f, target).ok_or_else(|| {
                "ShiftedAllOnesToNotMaskLowTransform: target must be bits".to_string()
            })?;
            let all_ones = mu::mk_literal_all_ones(f, width);
            f.get_node_mut(target).payload = NodePayload::Binop(Binop::Shll, all_ones, count);
            return Ok(());
        }
        Err("ShiftedAllOnesToNotMaskLowTransform: unsupported target".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn rewrites_shifted_all_ones_to_not_ext_mask_low() {
        let ir_text = r#"fn t(count: bits[4] id=1) -> bits[8] {
  ones: bits[8] = literal(value=255, id=2)
  ret out: bits[8] = shll(ones, count, id=3)
}"#;
        let mut f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        let t = ShiftedAllOnesToNotMaskLowTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Unop(Unop::Not, _)
        ));
    }

    #[test]
    fn reverses_not_ext_mask_low_to_shifted_all_ones() {
        let ir_text = r#"fn t(count: bits[4] id=1) -> bits[8] {
  mask: bits[8] = ext_mask_low(count, id=2)
  ret out: bits[8] = not(mask, id=3)
}"#;
        let mut f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        let t = ShiftedAllOnesToNotMaskLowTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Binop(Binop::Shll, _, _)
        ));
    }
}
