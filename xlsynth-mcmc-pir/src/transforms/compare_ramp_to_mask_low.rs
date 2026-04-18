// SPDX-License-Identifier: Apache-2.0

use super::macro_utils as mu;
use super::*;

/// Converts exact `count > bit_index` mask ramps to `ext_mask_low(count)`.
#[derive(Debug)]
pub struct CompareRampToMaskLowTransform;

impl CompareRampToMaskLowTransform {
    fn host_value_fits_in_bits_width(value: usize, width: usize) -> bool {
        if width == 0 {
            return value == 0;
        }
        u32::try_from(width)
            .ok()
            .and_then(|w| 1usize.checked_shl(w))
            .map(|limit| value < limit)
            .unwrap_or(true)
    }

    fn cmp_bit_parts(f: &IrFn, nr: NodeRef, expected_index: usize) -> Option<NodeRef> {
        if !mu::is_u1(f, nr) {
            return None;
        }
        let NodePayload::Binop(op, lhs, rhs) = f.get_node(nr).payload else {
            return None;
        };
        match op {
            Binop::Ugt => {
                if f.get_node(lhs).ty != f.get_node(rhs).ty
                    || !matches!(f.get_node(lhs).ty, Type::Bits(_))
                    || mu::literal_usize(f, rhs) != Some(expected_index)
                {
                    return None;
                }
                Some(lhs)
            }
            Binop::Ult => {
                if f.get_node(lhs).ty != f.get_node(rhs).ty
                    || !matches!(f.get_node(rhs).ty, Type::Bits(_))
                    || mu::literal_usize(f, lhs) != Some(expected_index)
                {
                    return None;
                }
                Some(rhs)
            }
            _ => None,
        }
    }

    fn ramp_count(f: &IrFn, nr: NodeRef) -> Option<NodeRef> {
        let width = mu::bits_width(f, nr)?;
        if width == 0 {
            return None;
        }
        if width == 1 {
            return Self::cmp_bit_parts(f, nr, 0);
        }
        let NodePayload::Nary(NaryOp::Concat, ops) = &f.get_node(nr).payload else {
            return None;
        };
        if ops.len() != width {
            return None;
        }
        let mut count: Option<NodeRef> = None;
        for (pos, op_ref) in ops.iter().enumerate() {
            let bit_index = width - 1 - pos;
            let bit_count = Self::cmp_bit_parts(f, *op_ref, bit_index)?;
            if !matches!(f.get_node(bit_count).ty, Type::Bits(_)) {
                return None;
            }
            match count {
                Some(existing) if existing != bit_count => return None,
                Some(_) => {}
                None => count = Some(bit_count),
            }
        }
        count
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

    fn build_ramp(f: &mut IrFn, count: NodeRef, width: usize) -> Result<NodePayload, String> {
        let count_width = mu::bits_width(f, count)
            .ok_or_else(|| "CompareRampToMaskLowTransform: count must be bits".to_string())?;
        for bit_index in 0..width {
            if !Self::host_value_fits_in_bits_width(bit_index, count_width) {
                return Err(format!(
                    "CompareRampToMaskLowTransform: bit index {bit_index} does not fit count width {count_width}"
                ));
            }
        }
        let mut ops = Vec::with_capacity(width);
        for bit_index in (0..width).rev() {
            let lit = mu::mk_literal_usize(f, count_width, bit_index);
            let cmp = mu::mk_binop(f, Binop::Ugt, Type::Bits(1), count, lit);
            ops.push(cmp);
        }
        if width == 1 {
            Ok(f.get_node(ops[0]).payload.clone())
        } else {
            Ok(NodePayload::Nary(NaryOp::Concat, ops))
        }
    }
}

impl PirTransform for CompareRampToMaskLowTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::CompareRampToMaskLow
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            if Self::ramp_count(f, nr).is_some() || Self::ext_mask_low_count(f, nr).is_some() {
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
                return Err("CompareRampToMaskLowTransform: expected node location".to_string());
            }
        };
        if let Some(count) = Self::ramp_count(f, target) {
            f.get_node_mut(target).payload = NodePayload::ExtMaskLow { count };
            return Ok(());
        }
        if let Some(count) = Self::ext_mask_low_count(f, target) {
            let width = mu::bits_width(f, target)
                .ok_or_else(|| "CompareRampToMaskLowTransform: target must be bits".to_string())?;
            let payload = Self::build_ramp(f, count, width)?;
            f.get_node_mut(target).payload = payload;
            return Ok(());
        }
        Err("CompareRampToMaskLowTransform: unsupported target".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn rewrites_compare_ramp_to_ext_mask_low() {
        let ir_text = r#"fn t(count: bits[3] id=1) -> bits[4] {
  l3: bits[3] = literal(value=3, id=2)
  b3: bits[1] = ugt(count, l3, id=3)
  l2: bits[3] = literal(value=2, id=4)
  b2: bits[1] = ult(l2, count, id=5)
  l1: bits[3] = literal(value=1, id=6)
  b1: bits[1] = ugt(count, l1, id=7)
  l0: bits[3] = literal(value=0, id=8)
  b0: bits[1] = ult(l0, count, id=9)
  ret out: bits[4] = concat(b3, b2, b1, b0, id=10)
}"#;
        let mut f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        let t = CompareRampToMaskLowTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::ExtMaskLow { .. }
        ));
    }

    #[test]
    fn reverses_ext_mask_low_to_compare_ramp() {
        let ir_text = r#"fn t(count: bits[3] id=1) -> bits[4] {
  ret out: bits[4] = ext_mask_low(count, id=2)
}"#;
        let mut f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        let t = CompareRampToMaskLowTransform;
        t.apply(&mut f, &TransformLocation::Node(target)).unwrap();
        assert!(matches!(
            f.get_node(target).payload,
            NodePayload::Nary(NaryOp::Concat, _)
        ));
    }

    #[test]
    fn rejects_noncontiguous_ramp() {
        let ir_text = r#"fn t(count: bits[3] id=1) -> bits[2] {
  l1: bits[3] = literal(value=1, id=2)
  b1: bits[1] = ugt(count, l1, id=3)
  l0: bits[3] = literal(value=0, id=4)
  b0: bits[1] = ugt(count, l0, id=5)
  ret out: bits[2] = concat(b0, b1, id=6)
}"#;
        let f = ir_parser::Parser::new(ir_text).parse_fn().unwrap();
        let target = f.ret_node_ref.unwrap();
        assert!(CompareRampToMaskLowTransform::ramp_count(&f, target).is_none());
    }
}
