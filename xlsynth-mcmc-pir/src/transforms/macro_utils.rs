// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) fn next_text_id(f: &IrFn) -> usize {
    f.nodes
        .iter()
        .map(|n| n.text_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

pub(super) fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
    match f.get_node(r).ty {
        Type::Bits(w) => Some(w),
        _ => None,
    }
}

pub(super) fn is_bits_w(f: &IrFn, r: NodeRef, w: usize) -> bool {
    bits_width(f, r) == Some(w)
}

pub(super) fn is_u1(f: &IrFn, r: NodeRef) -> bool {
    bits_width(f, r) == Some(1)
}

pub(super) fn push_node(f: &mut IrFn, ty: Type, payload: NodePayload) -> NodeRef {
    let new_index = f.nodes.len();
    f.nodes.push(Node {
        text_id: next_text_id(f),
        name: None,
        ty,
        payload,
        pos: None,
    });
    NodeRef { index: new_index }
}

pub(super) fn mk_bit_slice(f: &mut IrFn, arg: NodeRef, start: usize, width: usize) -> NodeRef {
    push_node(
        f,
        Type::Bits(width),
        NodePayload::BitSlice { arg, start, width },
    )
}

pub(super) fn mk_sign_ext_mask(f: &mut IrFn, bit: NodeRef, width: usize) -> NodeRef {
    push_node(
        f,
        Type::Bits(width),
        NodePayload::SignExt {
            arg: bit,
            new_bit_count: width,
        },
    )
}

pub(super) fn mk_zero_ext(f: &mut IrFn, arg: NodeRef, width: usize) -> NodeRef {
    push_node(
        f,
        Type::Bits(width),
        NodePayload::ZeroExt {
            arg,
            new_bit_count: width,
        },
    )
}

pub(super) fn mk_decode(f: &mut IrFn, arg: NodeRef, width: usize) -> NodeRef {
    push_node(f, Type::Bits(width), NodePayload::Decode { arg, width })
}

pub(super) fn mk_ext_mask_low(f: &mut IrFn, count: NodeRef, width: usize) -> NodeRef {
    push_node(f, Type::Bits(width), NodePayload::ExtMaskLow { count })
}

#[allow(dead_code)]
pub(super) fn mk_one_hot_sel(
    f: &mut IrFn,
    ty: Type,
    selector: NodeRef,
    cases: Vec<NodeRef>,
) -> NodeRef {
    push_node(f, ty, NodePayload::OneHotSel { selector, cases })
}

pub(super) fn mk_binop(f: &mut IrFn, op: Binop, ty: Type, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
    push_node(f, ty, NodePayload::Binop(op, lhs, rhs))
}

pub(super) fn mk_unop(f: &mut IrFn, op: Unop, ty: Type, arg: NodeRef) -> NodeRef {
    push_node(f, ty, NodePayload::Unop(op, arg))
}

pub(super) fn mk_nary_and(f: &mut IrFn, ty: Type, operands: Vec<NodeRef>) -> NodeRef {
    push_node(f, ty, NodePayload::Nary(NaryOp::And, operands))
}

#[allow(dead_code)]
pub(super) fn mk_nary_or_or_identity(f: &mut IrFn, ty: Type, operands: Vec<NodeRef>) -> NodeRef {
    if operands.len() == 1 {
        return operands[0];
    }
    push_node(f, ty, NodePayload::Nary(NaryOp::Or, operands))
}

pub(super) fn mk_literal_ubits(f: &mut IrFn, width: usize, value: u64) -> NodeRef {
    let bits = IrBits::make_ubits(width, value).expect("make_ubits");
    push_node(
        f,
        Type::Bits(width),
        NodePayload::Literal(IrValue::from_bits(&bits)),
    )
}

pub(super) fn mk_literal_usize(f: &mut IrFn, width: usize, value: usize) -> NodeRef {
    let bits = (0..width)
        .map(|i| {
            if i < usize::BITS as usize {
                ((value >> i) & 1) != 0
            } else {
                false
            }
        })
        .collect::<Vec<_>>();
    push_node(
        f,
        Type::Bits(width),
        NodePayload::Literal(IrValue::from_bits(&IrBits::from_lsb_is_0(&bits))),
    )
}

pub(super) fn mk_literal_all_ones(f: &mut IrFn, width: usize) -> NodeRef {
    push_node(
        f,
        Type::Bits(width),
        NodePayload::Literal(IrValue::from_bits(&IrBits::from_lsb_is_0(&vec![
            true;
            width
        ]))),
    )
}

pub(super) fn is_literal_one(f: &IrFn, r: NodeRef, width: usize) -> bool {
    if !is_bits_w(f, r, width) {
        return false;
    }
    let NodePayload::Literal(value) = &f.get_node(r).payload else {
        return false;
    };
    value.bits_equals_u64_value(1)
}

pub(super) fn is_literal_all_ones(f: &IrFn, r: NodeRef, width: usize) -> bool {
    if !is_bits_w(f, r, width) {
        return false;
    }
    let NodePayload::Literal(value) = &f.get_node(r).payload else {
        return false;
    };
    let Ok(bits) = value.to_bits() else {
        return false;
    };
    bits.get_bit_count() == width && (0..width).all(|i| bits.get_bit(i).unwrap_or(false))
}

pub(super) fn literal_usize(f: &IrFn, r: NodeRef) -> Option<usize> {
    let NodePayload::Literal(v) = &f.get_node(r).payload else {
        return None;
    };
    let bits = v.to_bits().ok()?;
    let mut value = 0usize;
    for i in 0..bits.get_bit_count() {
        if !bits.get_bit(i).ok()? {
            continue;
        }
        if i >= usize::BITS as usize {
            return None;
        }
        {
            value |= 1usize.checked_shl(i as u32)?;
        }
    }
    Some(value)
}

pub(super) fn sign_ext_mask_bit(f: &IrFn, r: NodeRef, width: usize) -> Option<NodeRef> {
    let NodePayload::SignExt { arg, new_bit_count } = f.get_node(r).payload else {
        return None;
    };
    if new_bit_count == width && is_u1(f, arg) && is_bits_w(f, r, width) {
        Some(arg)
    } else {
        None
    }
}

pub(super) fn bit_slice_parts(f: &IrFn, r: NodeRef) -> Option<(NodeRef, usize, usize)> {
    let NodePayload::BitSlice { arg, start, width } = f.get_node(r).payload else {
        return None;
    };
    Some((arg, start, width))
}

pub(super) fn selector_bit(f: &IrFn, r: NodeRef) -> Option<(NodeRef, usize)> {
    let (selector, start, width) = bit_slice_parts(f, r)?;
    if width == 1 {
        Some((selector, start))
    } else {
        None
    }
}

pub(super) fn masked_case_parts(f: &IrFn, r: NodeRef, width: usize) -> Option<(NodeRef, NodeRef)> {
    let NodePayload::Nary(NaryOp::And, ops) = &f.get_node(r).payload else {
        return None;
    };
    if ops.len() != 2 || !is_bits_w(f, r, width) {
        return None;
    }
    for (case, mask) in [(ops[0], ops[1]), (ops[1], ops[0])] {
        if !is_bits_w(f, case, width) {
            continue;
        }
        if let Some(pred) = sign_ext_mask_bit(f, mask, width) {
            return Some((case, pred));
        }
    }
    None
}

pub(super) fn or_operands(f: &IrFn, r: NodeRef) -> Option<Vec<NodeRef>> {
    match &f.get_node(r).payload {
        NodePayload::Nary(NaryOp::Or, ops) => Some(ops.clone()),
        _ => None,
    }
}

pub(super) fn unwrap_identity(f: &IrFn, r: NodeRef) -> NodeRef {
    match f.get_node(r).payload {
        NodePayload::Unop(Unop::Identity, arg) => arg,
        _ => r,
    }
}

pub(super) fn constant_shift(
    f: &mut IrFn,
    op: Binop,
    x: NodeRef,
    x_width: usize,
    amount_width: usize,
    amount: usize,
) -> NodeRef {
    let lit = mk_literal_ubits(f, amount_width, amount as u64);
    mk_binop(f, op, Type::Bits(x_width), x, lit)
}
