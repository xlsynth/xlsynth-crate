// SPDX-License-Identifier: Apache-2.0

use super::macro_utils as mu;
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SignbitComparePolarity {
    Ult,
    Uge,
}

impl SignbitComparePolarity {
    pub(super) fn binop(self) -> Binop {
        match self {
            Self::Ult => Binop::Ult,
            Self::Uge => Binop::Uge,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SignbitSubCompareParts {
    pub(super) lhs: NodeRef,
    pub(super) rhs: NodeRef,
    pub(super) sub: NodeRef,
    pub(super) signbit: NodeRef,
    pub(super) width: usize,
    pub(super) polarity: SignbitComparePolarity,
    pub(super) always_equivalent: bool,
}

fn literal_u1_value(f: &IrFn, r: NodeRef) -> Option<bool> {
    if !mu::is_u1(f, r) {
        return None;
    }
    let NodePayload::Literal(value) = &f.get_node(r).payload else {
        return None;
    };
    if value.bits_equals_u64_value(0) {
        Some(false)
    } else if value.bits_equals_u64_value(1) {
        Some(true)
    } else {
        None
    }
}

fn literal_has_zero_msb(f: &IrFn, r: NodeRef, width: usize) -> bool {
    let NodePayload::Literal(value) = &f.get_node(r).payload else {
        return false;
    };
    let Ok(bits) = value.to_bits() else {
        return false;
    };
    bits.get_bit_count() == width && !bits.get_bit(width - 1).unwrap_or(true)
}

fn locally_proves_zero_msb(f: &IrFn, r: NodeRef, width: usize) -> bool {
    if width == 0 || mu::bits_width(f, r) != Some(width) {
        return false;
    }
    match f.get_node(r).payload {
        NodePayload::ZeroExt { arg, new_bit_count } => {
            new_bit_count == width && mu::bits_width(f, arg).is_some_and(|arg_w| arg_w < width)
        }
        NodePayload::Literal(_) => literal_has_zero_msb(f, r, width),
        NodePayload::Unop(Unop::Identity, arg) => locally_proves_zero_msb(f, arg, width),
        _ => false,
    }
}

fn direct_signbit_sub_parts(f: &IrFn, r: NodeRef) -> Option<(NodeRef, NodeRef, NodeRef, usize)> {
    let NodePayload::BitSlice { arg, start, width } = f.get_node(r).payload else {
        return None;
    };
    if width != 1 {
        return None;
    }
    let sub_width = mu::bits_width(f, arg)?;
    if sub_width == 0 || start != sub_width - 1 {
        return None;
    }
    let NodePayload::Binop(Binop::Sub, lhs, rhs) = f.get_node(arg).payload else {
        return None;
    };
    if mu::bits_width(f, lhs) != Some(sub_width) || mu::bits_width(f, rhs) != Some(sub_width) {
        return None;
    }
    Some((lhs, rhs, arg, sub_width))
}

fn finish_parts(
    f: &IrFn,
    signbit: NodeRef,
    polarity: SignbitComparePolarity,
) -> Option<SignbitSubCompareParts> {
    let (lhs, rhs, sub, width) = direct_signbit_sub_parts(f, signbit)?;
    let always_equivalent =
        locally_proves_zero_msb(f, lhs, width) && locally_proves_zero_msb(f, rhs, width);
    Some(SignbitSubCompareParts {
        lhs,
        rhs,
        sub,
        signbit,
        width,
        polarity,
        always_equivalent,
    })
}

pub(super) fn signbit_sub_compare_parts(f: &IrFn, root: NodeRef) -> Option<SignbitSubCompareParts> {
    if let Some(parts) = finish_parts(f, root, SignbitComparePolarity::Ult) {
        return Some(parts);
    }

    match f.get_node(root).payload {
        NodePayload::Unop(Unop::Not, arg) => finish_parts(f, arg, SignbitComparePolarity::Uge),
        NodePayload::Binop(Binop::Eq, lhs, rhs) => {
            for (signbit, lit) in [(lhs, rhs), (rhs, lhs)] {
                match literal_u1_value(f, lit) {
                    Some(false) => return finish_parts(f, signbit, SignbitComparePolarity::Uge),
                    Some(true) => return finish_parts(f, signbit, SignbitComparePolarity::Ult),
                    None => {}
                }
            }
            None
        }
        NodePayload::Binop(Binop::Ne, lhs, rhs) => {
            for (signbit, lit) in [(lhs, rhs), (rhs, lhs)] {
                match literal_u1_value(f, lit) {
                    Some(false) => return finish_parts(f, signbit, SignbitComparePolarity::Ult),
                    Some(true) => return finish_parts(f, signbit, SignbitComparePolarity::Uge),
                    None => {}
                }
            }
            None
        }
        _ => None,
    }
}

pub(super) fn only_sub_user_is_signbit(f: &IrFn, parts: &SignbitSubCompareParts) -> bool {
    let users = compute_users(f);
    users
        .get(&parts.sub)
        .is_some_and(|sub_users| sub_users.len() == 1 && sub_users.contains(&parts.signbit))
}
