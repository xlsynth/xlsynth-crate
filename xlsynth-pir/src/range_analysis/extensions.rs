// SPDX-License-Identifier: Apache-2.0

//! Direct extension transfers; no lowered graph is allocated during analysis.

use super::policy::{
    EXACT_VALUES, NORMALIZE_BIT_BUDGET, NORMALIZE_INTERMEDIATE_INTERVALS, RESULT_INTERVALS,
};
use super::{IntervalSet, RangeValue, ops};
use crate::IrBits;
use crate::ir::{self, Binop, NodePayload, Type, Unop};
use crate::known_bits::{self, KnownBits};

fn zero(width: usize) -> IntervalSet {
    IntervalSet::singleton(IrBits::zero(width))
}

/// Counts leading zeros directly from packed limbs, including `bits[0]`.
fn leading_zeros(value: &IrBits) -> usize {
    for (index, &limb) in value.limbs().iter().enumerate().rev() {
        if limb != 0 {
            return value.get_bit_count() - (index * 64 + 64 - limb.leading_zeros() as usize);
        }
    }
    value.get_bit_count()
}

/// Returns the first set bit, using the input width as the zero sentinel.
fn trailing_zeros(value: &IrBits) -> usize {
    for (index, &limb) in value.limbs().iter().enumerate() {
        if limb != 0 {
            return index * 64 + limb.trailing_zeros() as usize;
        }
    }
    value.get_bit_count()
}

/// CLZ is antitone on unsigned intervals; offsets are added before truncation.
fn clz(arg: &IntervalSet, offset: usize, width: usize) -> IntervalSet {
    let count_width = (usize::BITS as usize + 1).max(width);
    let offset = ops::index_bits(count_width, offset);
    let intervals = arg.intervals().iter().map(|interval| {
        (
            ops::index_bits(count_width, leading_zeros(interval.upper())).add(&offset),
            ops::index_bits(count_width, leading_zeros(interval.lower())).add(&offset),
        )
    });
    let counts = IntervalSet::from_intervals(count_width, intervals).unwrap();
    ops::truncate(&counts, width)
}

/// Encodes feasible first-set positions, including the zero-input sentinel.
fn priority_encode(arg: &IntervalSet, lsb_prio: bool, width: usize) -> IntervalSet {
    if !lsb_prio {
        let counts = clz(arg, 0, usize::BITS as usize);
        // Nonzero inputs reverse the CLZ order; zero uses the width sentinel.
        let nonzero_counts = counts.intersect(
            &IntervalSet::from_intervals(
                usize::BITS as usize,
                [(
                    ops::index_bits(usize::BITS as usize, 0),
                    ops::index_bits(usize::BITS as usize, arg.width().saturating_sub(1)),
                )],
            )
            .unwrap(),
        );
        let mut result = if !nonzero_counts.is_empty() && arg.width() != 0 {
            let top =
                IntervalSet::singleton(ops::index_bits(usize::BITS as usize, arg.width() - 1));
            ops::truncate(
                &ops::binop(Binop::Sub, &top, &nonzero_counts, usize::BITS as usize),
                width,
            )
        } else {
            IntervalSet::empty(width)
        };
        if arg.contains_usize(0) {
            result = result.union(&IntervalSet::singleton(ops::index_bits(width, arg.width())));
        }
        return result.into_minimized(RESULT_INTERVALS);
    }
    let mut result = Vec::with_capacity(arg.intervals().len().saturating_mul(3));
    let one = ops::index_bits(arg.width(), 1);
    for interval in arg.intervals() {
        if interval.lower() == interval.upper() {
            let value = ops::index_bits(width, trailing_zeros(interval.lower()));
            result.push((value.clone(), value));
            continue;
        }
        let length = interval.upper().sub(interval.lower()).add(&one);
        if length.is_zero() {
            // Only the full input domain has cardinality 2^input_width.
            result.push((IrBits::zero(width), ops::index_bits(width, arg.width())));
            continue;
        }
        let low_count = arg.width() - leading_zeros(&length) - 1;
        if low_count != 0 {
            // An interval of length L contains every first-set position below
            // floor(log2 L), since each such residue repeats at most every L.
            result.push((IrBits::zero(width), ops::index_bits(width, low_count - 1)));
        }
        // All remaining possibilities are multiples of 2^low_count. Since
        // L < 2^(low_count+1), at most two such multiples lie in this interval.
        let low_bits = low_mask(arg.width(), low_count);
        let step = low_bits.add(&one);
        let aligned = interval.lower().and(&low_bits.not());
        let mut candidate = if &aligned == interval.lower() {
            aligned
        } else {
            aligned.add(&step)
        };
        if candidate.ult(interval.lower()) {
            // Rounding upward overflowed the input width; no multiple fits.
            continue;
        }
        for _ in 0..2 {
            if candidate.ugt(interval.upper()) {
                break;
            }
            let value = ops::index_bits(width, trailing_zeros(&candidate));
            result.push((value.clone(), value));
            let next = candidate.add(&step);
            if next.ule(&candidate) {
                // The next multiple is outside the nonwrapping interval.
                break;
            }
            candidate = next;
        }
    }
    IntervalSet::from_intervals(width, result)
        .unwrap()
        .into_minimized(RESULT_INTERVALS)
}

/// Builds a low mask in byte-sized chunks rather than visiting every bit.
fn low_mask(width: usize, count: usize) -> IrBits {
    let count = count.min(width);
    let mut bytes = vec![0; width.div_ceil(8)];
    bytes[..count / 8].fill(u8::MAX);
    if count % 8 != 0 {
        bytes[count / 8] = (1 << (count % 8)) - 1;
    }
    IrBits::from_le_bytes(width, &bytes).unwrap()
}

/// Saturates count ranges before constructing masks; small images are exact.
fn mask_low(count: &IntervalSet, width: usize) -> IntervalSet {
    if let Some(values) = count.values_up_to(EXACT_VALUES) {
        return IntervalSet::from_intervals(
            width,
            values.into_iter().map(|value| {
                let value = low_mask(width, ops::saturating_index(&value, width));
                (value.clone(), value)
            }),
        )
        .unwrap();
    }
    // A numeric hull is sound for the monotone mask function. Intersection with
    // its cheap packed-mask facts retains guaranteed low ones and high zeros.
    let intervals = count.intervals().iter().map(|interval| {
        (
            low_mask(width, ops::saturating_index(interval.lower(), width)),
            low_mask(width, ops::saturating_index(interval.upper(), width)),
        )
    });
    let result = IntervalSet::from_intervals(width, intervals)
        .unwrap()
        .into_minimized(RESULT_INTERVALS);
    let minimum = count
        .unsigned_min()
        .map(|v| ops::saturating_index(v, width))
        .unwrap_or(0);
    let maximum = count
        .unsigned_max()
        .map(|v| ops::saturating_index(v, width))
        .unwrap_or(0);
    let facts = KnownBits::from_mask_value(
        IrBits::from_lsb_fn(width, |bit| bit < minimum || bit >= maximum),
        low_mask(width, minimum),
    )
    .unwrap();
    result.intersect(&IntervalSet::from_known_bits(&facts))
}

/// Conditions each fixed normalization on its leading one, with bounded work.
fn normalize_left(
    node: &ir::Node,
    arg: &IntervalSet,
    offset: usize,
    width: usize,
    clz_width: Option<usize>,
) -> RangeValue {
    let counts = clz(arg, 0, usize::BITS as usize);
    let cost = width
        .max(arg.width())
        .max(1)
        .saturating_mul(arg.intervals().len().max(1));
    let candidates = counts.values_up_to((NORMALIZE_BIT_BUDGET / cost).max(1));
    let normalized = if let Some(candidates) = candidates {
        let mut result = IntervalSet::empty(width);
        for candidate in candidates {
            let count = ops::saturating_index(&candidate, arg.width());
            let shift = count.saturating_add(offset);
            let next = if count == arg.width() || shift >= width || i64::try_from(shift).is_err() {
                zero(width)
            } else {
                let first = arg.width() - 1 - count;
                let condition = IntervalSet::from_intervals(
                    arg.width(),
                    [(
                        IrBits::from_lsb_fn(arg.width(), |bit| bit == first),
                        IrBits::from_lsb_fn(arg.width(), |bit| bit <= first),
                    )],
                )
                .unwrap();
                let restricted = arg.intersect(&condition);
                let extended = ops::extend(&restricted, width, false);
                ops::binop(
                    Binop::Shll,
                    &extended,
                    &IntervalSet::singleton(ops::index_bits(usize::BITS as usize, shift)),
                    width,
                )
            };
            result = result.union(&next);
            if result.intervals().len() > NORMALIZE_INTERMEDIATE_INTERVALS {
                result = result.into_minimized(RESULT_INTERVALS);
            }
        }
        result.into_minimized(RESULT_INTERVALS)
    } else {
        // The existing packed-mask extension transfer also bounds uncertain
        // normalization work. It never assumes correlations between inputs.
        let bits = arg.to_known_bits();
        let fallback = known_bits::extensions::evaluate(node, &[&bits]);
        let normalized_path: &[usize] = if clz_width.is_some() { &[0] } else { &[] };
        IntervalSet::from_known_bits(
            fallback
                .leaf(normalized_path)
                .expect("normalization result is bits or the first leaf of a pair"),
        )
    };
    match clz_width {
        Some(count_width) => RangeValue::Tuple(vec![
            RangeValue::Bits(normalized),
            RangeValue::Bits(ops::truncate(&counts, count_width)),
        ]),
        None => RangeValue::Bits(normalized),
    }
}

/// Evaluates extension operands in `ir_utils::operands` order.
pub(super) fn evaluate(node: &ir::Node, operands: &[&IntervalSet]) -> RangeValue {
    let width = || match node.ty {
        Type::Bits(width) => width,
        _ => unreachable!("validated scalar extension"),
    };
    RangeValue::Bits(match &node.payload {
        NodePayload::ExtCarryOut { .. } => {
            let input_width = operands[0].width();
            let width = input_width.checked_add(1).expect("validated carry width");
            let lhs = ops::extend(operands[0], width, false);
            let rhs = ops::extend(operands[1], width, false);
            let carry = ops::extend(operands[2], width, false);
            let sum = ops::binop(
                Binop::Add,
                &ops::binop(Binop::Add, &lhs, &rhs, width),
                &carry,
                width,
            );
            ops::bit_slice(&sum, input_width, 1)
        }
        NodePayload::ExtPrioEncode { lsb_prio, .. } => {
            priority_encode(operands[0], *lsb_prio, width())
        }
        NodePayload::ExtClz {
            offset,
            new_bit_count,
            ..
        } => clz(operands[0], *offset, *new_bit_count),
        NodePayload::ExtMaskLow { .. } => mask_low(operands[0], width()),
        NodePayload::ExtNormalizeLeft {
            shift_offset,
            normalized_bit_count,
            clz_bit_count,
            ..
        } => {
            return normalize_left(
                node,
                operands[0],
                *shift_offset,
                *normalized_bit_count,
                *clz_bit_count,
            );
        }
        NodePayload::ExtNaryAdd { terms, arch: _ } => {
            // Architecture changes lowering, not the modular sum contract.
            let mut result = zero(width());
            for (term, operand) in terms.iter().zip(operands) {
                let value = ops::extend(operand, width(), term.signed);
                let value = if term.negated {
                    ops::unop(Unop::Neg, &value, width())
                } else {
                    value
                };
                result = ops::binop(Binop::Add, &result, &value, width());
            }
            result
        }
        _ => unreachable!("extension transfer applied to a standard node"),
    })
}

#[cfg(test)]
mod tests {
    use super::priority_encode;
    use crate::IrBits;
    use crate::range_analysis::IntervalSet;

    #[test]
    fn priority_encoding_matches_every_small_interval_exactly() {
        for width in 0usize..=6 {
            let size = 1usize << width;
            let output_width = usize::BITS as usize - width.leading_zeros() as usize;
            for lower in 0..size {
                for upper in 0..size {
                    let input = IntervalSet::from_intervals(
                        width,
                        [(
                            IrBits::make_ubits(width, lower as u64).unwrap(),
                            IrBits::make_ubits(width, upper as u64).unwrap(),
                        )],
                    )
                    .unwrap();
                    for lsb_prio in [false, true] {
                        let expected = IntervalSet::from_intervals(
                            output_width,
                            (0..size)
                                .filter(|&value| input.contains_usize(value))
                                .map(|value| {
                                    let index = if value == 0 {
                                        width
                                    } else if lsb_prio {
                                        value.trailing_zeros() as usize
                                    } else {
                                        usize::BITS as usize - value.leading_zeros() as usize - 1
                                    };
                                    let value =
                                        IrBits::make_ubits(output_width, index as u64).unwrap();
                                    (value.clone(), value)
                                }),
                        )
                        .unwrap();
                        assert_eq!(
                            priority_encode(&input, lsb_prio, output_width),
                            expected,
                            "input width={width} [{lower},{upper}], lsb_prio={lsb_prio}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn wide_lsb_priority_encoding_uses_interval_structure() {
        let width = 65_537;
        let output_width = 17;
        let position = 32_768;
        let power = IrBits::from_lsb_fn(width, |bit| bit == position);
        let singleton = IntervalSet::singleton(power.clone());
        assert_eq!(
            priority_encode(&singleton, true, output_width),
            IntervalSet::singleton(IrBits::make_ubits(output_width, position as u64).unwrap())
        );
        let interval = IntervalSet::from_intervals(
            width,
            [(
                power.clone(),
                power.add(&IrBits::make_ubits(width, 7).unwrap()),
            )],
        )
        .unwrap();
        let index = IrBits::make_ubits(output_width, position as u64).unwrap();
        assert_eq!(
            priority_encode(&interval, true, output_width),
            IntervalSet::from_intervals(
                output_width,
                [
                    (
                        IrBits::zero(output_width),
                        IrBits::make_ubits(output_width, 2).unwrap()
                    ),
                    (index.clone(), index),
                ],
            )
            .unwrap()
        );
        assert_eq!(
            priority_encode(&IntervalSet::full(width), true, output_width),
            IntervalSet::from_intervals(
                output_width,
                [(
                    IrBits::zero(output_width),
                    IrBits::make_ubits(output_width, width as u64).unwrap()
                )]
            )
            .unwrap()
        );
    }
}
