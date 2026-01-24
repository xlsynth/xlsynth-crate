// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrBits, IrValue};
use xlsynth_pir::ir::{Binop, Fn as IrFn, Node, NodePayload, NodeRef, Type};

use super::{PirTransform, PirTransformKind, TransformLocation};

/// A non-always-equivalent transform that replaces a complicated shift-amount
/// expression with a shallow canonical clamp.
///
/// This is intended to eliminate deep “shift amount clamp chain” patterns,
/// relying on an external equivalence oracle to accept or reject the rewrite.
#[derive(Debug)]
pub struct ShiftReclampTransform;

impl ShiftReclampTransform {
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

    fn unwrap_identity(f: &IrFn, r: NodeRef) -> NodeRef {
        match f.get_node(r).payload {
            NodePayload::Unop(xlsynth_pir::ir::Unop::Identity, arg) => arg,
            _ => r,
        }
    }

    fn looks_like_clamped_amount(f: &IrFn, amount: NodeRef) -> bool {
        let amount = Self::unwrap_identity(f, amount);
        matches!(
            f.get_node(amount).payload,
            NodePayload::Sel { .. } | NodePayload::PrioritySel { .. }
        )
    }

    fn is_canonical_shift_clamp(f: &IrFn, amount: NodeRef) -> bool {
        let amount = Self::unwrap_identity(f, amount);
        let NodePayload::Sel {
            selector,
            cases,
            default,
        } = &f.get_node(amount).payload
        else {
            return false;
        };
        if default.is_some() || cases.len() != 2 {
            return false;
        }
        // selector = ult(x, max_lit), cases=[max_lit, x]
        let NodePayload::Binop(Binop::Ult, x, max_lit) = f.get_node(*selector).payload else {
            return false;
        };
        if cases[0] != max_lit || cases[1] != x {
            return false;
        }
        matches!(f.get_node(max_lit).payload, NodePayload::Literal(_))
    }
}

impl PirTransform for ShiftReclampTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ShiftReclamp
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out = Vec::new();
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
            if !Self::looks_like_clamped_amount(f, amount) {
                continue;
            }
            if Self::is_canonical_shift_clamp(f, amount) {
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
                    "ShiftReclampTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let NodePayload::Binop(op, x, amount) = f.get_node(target_ref).payload.clone() else {
            return Err("ShiftReclampTransform: expected binop payload".to_string());
        };
        if !Self::is_shift_op(op) {
            return Err("ShiftReclampTransform: expected shift binop".to_string());
        }
        let x_width = Self::bits_width(f, x)
            .ok_or_else(|| "ShiftReclampTransform: x must be bits[w]".to_string())?;
        let amount_bits = Self::bits_width(f, amount)
            .ok_or_else(|| "ShiftReclampTransform: amount must be bits[k]".to_string())?;
        let max_amount = Self::compute_max_amount(x_width, amount_bits)
            .ok_or_else(|| "ShiftReclampTransform: cannot compute clamp bound".to_string())?;

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
