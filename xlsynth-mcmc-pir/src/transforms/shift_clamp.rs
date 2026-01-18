// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrBits, IrValue};
use xlsynth_pir::ir::{Binop, Fn as IrFn, Node, NodePayload, NodeRef, Type};

use super::{PirTransform, PirTransformKind, TransformLocation};

/// A non-always-equivalent transform that clamps shift amounts:
///
/// `sh{ll,rl,ra}(x, amount) -> sh{ll,rl,ra}(x, min(amount, max_amount))`
///
/// where `max_amount = min(width(x) - 1, (2^amount_bits) - 1)`.
#[derive(Debug)]
pub struct ShiftClampTransform;

impl ShiftClampTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn mk_ubits_literal_node(f: &mut IrFn, w: usize, value: u64) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        let bits = IrBits::make_ubits(w, value).expect("make_ubits");
        let value = IrValue::from_bits(&bits);
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Literal(value),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_binop_node(f: &mut IrFn, op: Binop, ty: Type, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty,
            payload: NodePayload::Binop(op, lhs, rhs),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_sel2(f: &mut IrFn, ty: Type, selector: NodeRef, a: NodeRef, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty,
            payload: NodePayload::Sel {
                selector,
                cases: vec![a, b],
                default: None,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn is_shift_op(op: Binop) -> bool {
        matches!(op, Binop::Shll | Binop::Shrl | Binop::Shra)
    }

    fn compute_max_amount(x_width: usize, amount_bits: usize) -> Option<u64> {
        if x_width == 0 || amount_bits == 0 {
            return None;
        }
        let max_by_x = x_width.saturating_sub(1) as u64;
        let max_by_amount = if amount_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << amount_bits).saturating_sub(1)
        };
        Some(std::cmp::min(max_by_x, max_by_amount))
    }
}

impl PirTransform for ShiftClampTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ShiftClamp
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let NodePayload::Binop(op, x, amount) = f.get_node(nr).payload else {
                continue;
            };
            if !Self::is_shift_op(op) {
                continue;
            }
            let Some(x_width) = Self::bits_width(f, x) else {
                continue;
            };
            let Some(amount_bits) = Self::bits_width(f, amount) else {
                continue;
            };
            if Self::bits_width(f, nr) != Some(x_width) {
                continue;
            }
            if Self::compute_max_amount(x_width, amount_bits).is_none() {
                continue;
            }
            out.push(TransformLocation::Node(nr));
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "ShiftClampTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let NodePayload::Binop(op, x, amount) = f.get_node(target_ref).payload.clone() else {
            return Err("ShiftClampTransform: expected binop payload".to_string());
        };
        if !Self::is_shift_op(op) {
            return Err("ShiftClampTransform: expected sh{ll,rl,ra} binop".to_string());
        }

        let x_width = Self::bits_width(f, x)
            .ok_or_else(|| "ShiftClampTransform: x must be bits[w]".to_string())?;
        let amount_bits = Self::bits_width(f, amount)
            .ok_or_else(|| "ShiftClampTransform: amount must be bits[k]".to_string())?;
        if Self::bits_width(f, target_ref) != Some(x_width) {
            return Err("ShiftClampTransform: output width must match x".to_string());
        }

        let max_amount = Self::compute_max_amount(x_width, amount_bits)
            .ok_or_else(|| "ShiftClampTransform: cannot compute clamp bound".to_string())?;
        let max_lit = Self::mk_ubits_literal_node(f, amount_bits, max_amount);
        let pred = Self::mk_binop_node(f, Binop::Ult, Type::Bits(1), amount, max_lit);
        let clamped = Self::mk_sel2(f, Type::Bits(amount_bits), pred, max_lit, amount);

        f.get_node_mut(target_ref).payload = NodePayload::Binop(op, x, clamped);
        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::ShiftClampTransform;
    use xlsynth_pir::ir::{Binop, NodePayload, NodeRef, Type};
    use xlsynth_pir::ir_parser;

    use crate::transforms::{PirTransform, TransformLocation};

    fn find_shift_node(f: &xlsynth_pir::ir::Fn) -> NodeRef {
        for nr in f.node_refs() {
            if let NodePayload::Binop(Binop::Shll, _, _) = f.get_node(nr).payload {
                return nr;
            }
        }
        panic!("expected shll node");
    }

    fn literal_u64_value(f: &xlsynth_pir::ir::Fn, r: NodeRef) -> u64 {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            panic!("expected literal");
        };
        v.to_u64().expect("literal to u64")
    }

    #[test]
    fn shift_clamp_uses_x_width_minus_one() {
        let ir_text = r#"fn t(x: bits[5] id=1, amt: bits[3] id=2) -> bits[5] {
  ret shll.3: bits[5] = shll(x, amt, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let shll_ref = find_shift_node(&f);
        let t = ShiftClampTransform;
        t.apply(&mut f, &TransformLocation::Node(shll_ref))
            .expect("apply");

        let NodePayload::Binop(Binop::Shll, _x, new_amt) = f.get_node(shll_ref).payload else {
            panic!("expected shll binop");
        };
        let NodePayload::Sel {
            selector,
            ref cases,
            ..
        } = f.get_node(new_amt).payload
        else {
            panic!("expected sel");
        };
        assert_eq!(cases.len(), 2);
        assert!(matches!(f.get_node(selector).ty, Type::Bits(1)));
        assert_eq!(literal_u64_value(&f, cases[0]), 4);
    }

    #[test]
    fn shift_clamp_uses_amount_bitwidth_limit() {
        let ir_text = r#"fn t(x: bits[10] id=1, amt: bits[2] id=2) -> bits[10] {
  ret shll.3: bits[10] = shll(x, amt, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let shll_ref = find_shift_node(&f);
        let t = ShiftClampTransform;
        t.apply(&mut f, &TransformLocation::Node(shll_ref))
            .expect("apply");

        let NodePayload::Binop(Binop::Shll, _x, new_amt) = f.get_node(shll_ref).payload else {
            panic!("expected shll binop");
        };
        let NodePayload::Sel { ref cases, .. } = f.get_node(new_amt).payload else {
            panic!("expected sel");
        };
        assert_eq!(cases.len(), 2);
        assert_eq!(literal_u64_value(&f, cases[0]), 3);
    }
}
