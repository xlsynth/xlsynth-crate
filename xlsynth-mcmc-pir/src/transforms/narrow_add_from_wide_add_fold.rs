// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrBits, IrValue};
use xlsynth_pir::ir::{Binop, Fn as IrFn, NaryOp, NodePayload, NodeRef, Type};

use super::{PirTransform, PirTransformKind, TransformLocation};

/// A non-always-equivalent transform that folds a narrow add from a matching
/// wide add on zero-extended operands.
///
/// Match:
///   wide = add(zext(a), zext(b))
///   narrow = add(a, b)
/// Rewrite:
///   narrow = bit_slice(wide, 0, Wn)
#[derive(Debug)]
pub struct NarrowAddFromWideAddFoldTransform;

#[derive(Clone, Copy)]
struct ExtendInfo {
    base: NodeRef,
    wide_width: usize,
}

impl NarrowAddFromWideAddFoldTransform {
    fn bits_width(ty: &Type) -> Option<usize> {
        match ty {
            Type::Bits(w) => Some(*w),
            _ => None,
        }
    }

    fn is_zero_literal_node(f: &IrFn, r: NodeRef, w: usize) -> bool {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return false;
        };
        let bits = IrBits::make_ubits(w, 0).expect("make_ubits");
        let expected = IrValue::from_bits(&bits);
        *v == expected
    }

    fn zero_extend_info(f: &IrFn, r: NodeRef) -> Option<ExtendInfo> {
        match &f.get_node(r).payload {
            NodePayload::ZeroExt { arg, new_bit_count } => Some(ExtendInfo {
                base: *arg,
                wide_width: *new_bit_count,
            }),
            NodePayload::Nary(NaryOp::Concat, ops) if ops.len() == 2 => {
                let zero = ops[0];
                let base = ops[1];
                let zero_w = Self::bits_width(&f.get_node(zero).ty)?;
                let base_w = Self::bits_width(&f.get_node(base).ty)?;
                let concat_w = Self::bits_width(&f.get_node(r).ty)?;
                if zero_w + base_w != concat_w {
                    return None;
                }
                if !Self::is_zero_literal_node(f, zero, zero_w) {
                    return None;
                }
                Some(ExtendInfo {
                    base,
                    wide_width: concat_w,
                })
            }
            _ => None,
        }
    }

    fn add_parts(payload: &NodePayload) -> Option<(NodeRef, NodeRef)> {
        match payload {
            NodePayload::Binop(Binop::Add, lhs, rhs) => Some((*lhs, *rhs)),
            _ => None,
        }
    }

    fn find_matching_wide_add(
        f: &IrFn,
        a: NodeRef,
        b: NodeRef,
        narrow_w: usize,
    ) -> Option<(NodeRef, usize)> {
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            let Some((lhs, rhs)) = Self::add_parts(&node.payload) else {
                continue;
            };
            let Some(lhs_ext) = Self::zero_extend_info(f, lhs) else {
                continue;
            };
            let Some(rhs_ext) = Self::zero_extend_info(f, rhs) else {
                continue;
            };
            if lhs_ext.wide_width != rhs_ext.wide_width {
                continue;
            }
            let wide_w = lhs_ext.wide_width;
            if Self::bits_width(&node.ty) != Some(wide_w) {
                continue;
            }
            let matches_direct = lhs_ext.base == a && rhs_ext.base == b;
            let matches_swapped = lhs_ext.base == b && rhs_ext.base == a;
            if !matches_direct && !matches_swapped {
                continue;
            }
            if wide_w < narrow_w {
                // We can still extend later, but prefer wider matches first.
                return Some((nr, wide_w));
            }
            return Some((nr, wide_w));
        }
        None
    }
}

impl PirTransform for NarrowAddFromWideAddFoldTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::NarrowAddFromWideAddFold
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            let Some((lhs, rhs)) = Self::add_parts(&node.payload) else {
                continue;
            };
            let Some(narrow_w) = Self::bits_width(&node.ty) else {
                continue;
            };
            if Self::bits_width(&f.get_node(lhs).ty) != Some(narrow_w)
                || Self::bits_width(&f.get_node(rhs).ty) != Some(narrow_w)
            {
                continue;
            }
            if Self::find_matching_wide_add(f, lhs, rhs, narrow_w).is_some() {
                out.push(TransformLocation::Node(nr));
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "NarrowAddFromWideAddFoldTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        let Some((lhs, rhs)) = Self::add_parts(&target_payload) else {
            return Err("NarrowAddFromWideAddFoldTransform: expected add payload".to_string());
        };
        let Some(narrow_w) = Self::bits_width(&f.get_node(target_ref).ty) else {
            return Err("NarrowAddFromWideAddFoldTransform: target must be bits[w]".to_string());
        };
        if Self::bits_width(&f.get_node(lhs).ty) != Some(narrow_w)
            || Self::bits_width(&f.get_node(rhs).ty) != Some(narrow_w)
        {
            return Err(
                "NarrowAddFromWideAddFoldTransform: add operands must match output width"
                    .to_string(),
            );
        }

        let Some((wide_ref, wide_w)) = Self::find_matching_wide_add(f, lhs, rhs, narrow_w) else {
            return Err(
                "NarrowAddFromWideAddFoldTransform: no matching wide add found".to_string(),
            );
        };

        if wide_w >= narrow_w {
            f.get_node_mut(target_ref).payload = NodePayload::BitSlice {
                arg: wide_ref,
                start: 0,
                width: narrow_w,
            };
            Ok(())
        } else {
            f.get_node_mut(target_ref).payload = NodePayload::ZeroExt {
                arg: wide_ref,
                new_bit_count: narrow_w,
            };
            Ok(())
        }
    }

    fn always_equivalent(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use xlsynth_pir::ir::{Binop, NodePayload, NodeRef, Type};
    use xlsynth_pir::ir_parser;

    use super::NarrowAddFromWideAddFoldTransform;
    use crate::transforms::{PirTransform, TransformLocation};

    fn find_add_node(f: &xlsynth_pir::ir::Fn, width: usize) -> NodeRef {
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Add, _, _))
                && matches!(f.get_node(nr).ty, Type::Bits(w) if w == width)
            {
                return nr;
            }
        }
        panic!("expected add node");
    }

    #[test]
    fn narrow_add_from_wide_add_with_zext() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  zero_ext.10: bits[16] = zero_ext(a, new_bit_count=16, id=10)
  zero_ext.11: bits[16] = zero_ext(b, new_bit_count=16, id=11)
  add.12: bits[16] = add(zero_ext.10, zero_ext.11, id=12)
  ret add.20: bits[8] = add(a, b, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let narrow_ref = find_add_node(&f, 8);
        let t = NarrowAddFromWideAddFoldTransform;
        t.apply(&mut f, &TransformLocation::Node(narrow_ref))
            .expect("apply");

        let NodePayload::BitSlice { arg, start, width } = f.get_node(narrow_ref).payload else {
            panic!("expected bit_slice after rewrite");
        };
        assert_eq!(start, 0);
        assert_eq!(width, 8);
        assert!(matches!(
            f.get_node(arg).payload,
            NodePayload::Binop(Binop::Add, _, _)
        ));
    }

    #[test]
    fn narrow_add_from_wide_add_with_concat_zero() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  literal.9: bits[8] = literal(value=0, id=9)
  concat.10: bits[16] = concat(literal.9, a, id=10)
  concat.11: bits[16] = concat(literal.9, b, id=11)
  add.12: bits[16] = add(concat.10, concat.11, id=12)
  ret add.20: bits[8] = add(a, b, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let narrow_ref = find_add_node(&f, 8);
        let t = NarrowAddFromWideAddFoldTransform;
        t.apply(&mut f, &TransformLocation::Node(narrow_ref))
            .expect("apply");

        let NodePayload::BitSlice { arg, start, width } = f.get_node(narrow_ref).payload else {
            panic!("expected bit_slice after rewrite");
        };
        assert_eq!(start, 0);
        assert_eq!(width, 8);
        assert!(matches!(
            f.get_node(arg).payload,
            NodePayload::Binop(Binop::Add, _, _)
        ));
    }

    #[test]
    fn narrow_add_from_wide_add_with_concat_one_bit_zero() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  literal.9: bits[1] = literal(value=0, id=9)
  concat.10: bits[9] = concat(literal.9, a, id=10)
  concat.11: bits[9] = concat(literal.9, b, id=11)
  add.12: bits[9] = add(concat.10, concat.11, id=12)
  ret add.20: bits[8] = add(a, b, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let narrow_ref = find_add_node(&f, 8);
        let t = NarrowAddFromWideAddFoldTransform;
        t.apply(&mut f, &TransformLocation::Node(narrow_ref))
            .expect("apply");

        let NodePayload::BitSlice { arg, start, width } = f.get_node(narrow_ref).payload else {
            panic!("expected bit_slice after rewrite");
        };
        assert_eq!(start, 0);
        assert_eq!(width, 8);
        assert!(matches!(
            f.get_node(arg).payload,
            NodePayload::Binop(Binop::Add, _, _)
        ));
    }
}
