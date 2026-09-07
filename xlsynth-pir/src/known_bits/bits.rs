// SPDX-License-Identifier: Apache-2.0

//! Width-checked known bits and scalar three-valued transfer functions.

use std::cmp::Ordering;
use std::collections::VecDeque;

use crate::ir::{Binop, NaryOp, Unop};
use crate::{IrBits, ValueError};

type Bit = Option<bool>;

/// Unconditional bit facts; unknown positions are zero in the value mask.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KnownBits {
    mask: IrBits,
    value: IrBits,
}

impl KnownBits {
    /// Creates a value with no known bits.
    pub fn unknown(width: usize) -> Self {
        Self {
            mask: IrBits::zero(width),
            value: IrBits::zero(width),
        }
    }

    /// Creates a fully known value.
    pub fn constant(value: &IrBits) -> Self {
        Self {
            mask: IrBits::all_ones(value.get_bit_count()),
            value: value.clone(),
        }
    }

    /// Checks widths and clears value bits outside the supplied known mask.
    pub fn from_mask_value(mask: IrBits, value: IrBits) -> Result<Self, ValueError> {
        if mask.get_bit_count() != value.get_bit_count() {
            return Err(ValueError(format!(
                "known-bits mask/value widths differ: {} and {}",
                mask.get_bit_count(),
                value.get_bit_count()
            )));
        }
        let value = value.and(&mask);
        Ok(Self { mask, value })
    }

    pub fn bit_count(&self) -> usize {
        self.mask.get_bit_count()
    }

    pub fn mask(&self) -> &IrBits {
        &self.mask
    }

    pub fn value(&self) -> &IrBits {
        &self.value
    }

    pub fn known_bit_count(&self) -> usize {
        self.mask
            .limbs()
            .iter()
            .map(|x| x.count_ones() as usize)
            .sum()
    }

    pub fn is_fully_known(&self) -> bool {
        self.known_bit_count() == self.bit_count()
    }

    /// Returns whether a concrete value satisfies these facts.
    pub fn contains(&self, value: &IrBits) -> bool {
        value.get_bit_count() == self.bit_count() && value.and(&self.mask) == self.value
    }

    /// Formats the most-significant bit first, using `X` for unknown bits.
    pub fn to_ternary_string(&self) -> String {
        self.lsb_bits()
            .into_iter()
            .rev()
            .map(|bit| match bit {
                Some(false) => '0',
                Some(true) => '1',
                None => 'X',
            })
            .collect()
    }

    pub(crate) fn from_lsb_bits(bits: &[Bit]) -> Self {
        Self {
            mask: IrBits::from_lsb_fn(bits.len(), |i| bits[i].is_some()),
            value: IrBits::from_lsb_fn(bits.len(), |i| bits[i].unwrap_or(false)),
        }
    }

    pub(crate) fn lsb_bits(&self) -> Vec<Bit> {
        (0..self.bit_count())
            .map(|i| {
                self.mask
                    .get_bit(i)
                    .unwrap()
                    .then(|| self.value.get_bit(i).unwrap())
            })
            .collect()
    }

    /// Keeps only equal facts shared by both alternative values.
    pub(crate) fn join(&self, other: &Self) -> Self {
        assert_eq!(self.bit_count(), other.bit_count());
        let mask = self
            .mask
            .and(&other.mask)
            .and(&self.value.xor(&other.value).not());
        let value = self.value.and(&mask);
        Self { mask, value }
    }
}

fn not(a: Bit) -> Bit {
    a.map(|a| !a)
}

fn and(a: Bit, b: Bit) -> Bit {
    match (a, b) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

fn or(a: Bit, b: Bit) -> Bit {
    match (a, b) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

fn xor(a: Bit, b: Bit) -> Bit {
    a.zip(b).map(|(a, b)| a ^ b)
}

fn select_bit(condition: Bit, yes: Bit, no: Bit) -> Bit {
    match condition {
        Some(true) => yes,
        Some(false) => no,
        None if yes == no => yes,
        None => None,
    }
}

fn select_bits(condition: Bit, yes: &[Bit], no: &[Bit]) -> Vec<Bit> {
    assert_eq!(yes.len(), no.len());
    yes.iter()
        .zip(no)
        .map(|(&yes, &no)| select_bit(condition, yes, no))
        .collect()
}

pub(super) fn full_adder(a: Bit, b: Bit, carry: Bit) -> (Bit, Bit) {
    // Majority preserves cases such as 1 + X + 1 that two chained half
    // adders lose, without assuming correlations among any unknown inputs.
    (
        xor(xor(a, b), carry),
        or(and(a, b), or(and(a, carry), and(b, carry))),
    )
}

fn add_bits(a: &[Bit], b: &[Bit]) -> Vec<Bit> {
    assert_eq!(a.len(), b.len());
    let mut carry = Some(false);
    a.iter()
        .zip(b)
        .map(|(&a, &b)| {
            let (sum, next) = full_adder(a, b, carry);
            carry = next;
            sum
        })
        .collect()
}

fn sub_bits(a: &[Bit], b: &[Bit]) -> Vec<Bit> {
    assert_eq!(a.len(), b.len());
    let mut borrow = Some(false);
    a.iter()
        .zip(b)
        .map(|(&a, &b)| {
            let difference = xor(xor(a, b), borrow);
            borrow = or(and(not(a), borrow), or(and(not(a), b), and(b, borrow)));
            difference
        })
        .collect()
}

fn negate_bits(a: &[Bit]) -> Vec<Bit> {
    sub_bits(&vec![Some(false); a.len()], a)
}

fn resize_bits(a: &[Bit], width: usize, signed: bool) -> Vec<Bit> {
    let fill = if signed {
        a.last().copied().unwrap_or(Some(false))
    } else {
        Some(false)
    };
    let mut result = a[..a.len().min(width)].to_vec();
    result.resize(width, fill);
    result
}

/// Compares independent extrema in unsigned order, or with the sign bit
/// inverted.
fn compare_extrema(
    a: &[Bit],
    a_unknown: bool,
    b: &[Bit],
    b_unknown: bool,
    signed: bool,
) -> Ordering {
    assert_eq!(a.len(), b.len());
    for i in (0..a.len()).rev() {
        let sign = signed && i == a.len() - 1;
        let av = a[i].map(|v| v ^ sign).unwrap_or(a_unknown);
        let bv = b[i].map(|v| v ^ sign).unwrap_or(b_unknown);
        match av.cmp(&bv) {
            Ordering::Equal => { /* Less-significant bits break the tie. */ }
            ordering => return ordering,
        }
    }
    Ordering::Equal
}

fn less_than(a: &[Bit], b: &[Bit], signed: bool) -> Bit {
    if compare_extrema(a, true, b, false, signed) == Ordering::Less {
        Some(true)
    } else if compare_extrema(a, false, b, true, signed) != Ordering::Less {
        Some(false)
    } else {
        None
    }
}

fn equal_bits(a: &[Bit], b: &[Bit]) -> Bit {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(Some(true), |result, (&a, &b)| and(result, not(xor(a, b))))
}

/// Evaluates the eager XLS Dadda reduction schedule, retaining only needed
/// columns.
fn multiply_bits(a: &[Bit], b: &[Bit], width: usize) -> Vec<Bit> {
    let mut columns = vec![VecDeque::new(); width];
    for (i, &a) in a.iter().enumerate().take(width) {
        for (j, &b) in b.iter().enumerate().take(width - i) {
            columns[i + j].push_back(and(a, b));
        }
    }
    let mut height = 2usize;
    while height < a.len().max(b.len()) {
        height = height.saturating_add(height / 2);
    }
    while height > 2 {
        height = height / 3 * 2 + (height % 3 * 2).div_ceil(3);
        for col in 0..columns.len() {
            let (low, high) = columns.split_at_mut(col + 1);
            let column = &mut low[col];
            let mut next = high.first_mut();
            while column.len() > height + 1 {
                let a = column.pop_front().unwrap();
                let b = column.pop_front().unwrap();
                let c = column.pop_front().unwrap();
                let (sum, carry) = full_adder(a, b, c);
                column.push_back(sum);
                if let Some(next) = next.as_mut() {
                    next.push_back(carry);
                }
            }
            if column.len() > height {
                let a = column.pop_front().unwrap();
                let b = column.pop_front().unwrap();
                column.push_back(xor(a, b));
                if let Some(next) = next.as_mut() {
                    next.push_back(and(a, b));
                }
            }
        }
    }
    let mut result = Vec::with_capacity(width);
    for col in 0..columns.len() {
        let (low, high) = columns.split_at_mut(col + 1);
        let column = &mut low[col];
        let height = column.len();
        let (sum, carry) = match height {
            0 => (Some(false), Some(false)),
            1 => (column.pop_front().unwrap(), Some(false)),
            2 => {
                let a = column.pop_front().unwrap();
                let b = column.pop_front().unwrap();
                (xor(a, b), and(a, b))
            }
            3 => full_adder(
                column.pop_front().unwrap(),
                column.pop_front().unwrap(),
                column.pop_front().unwrap(),
            ),
            _ => unreachable!("Dadda columns have at most three remaining entries"),
        };
        result.push(sum);
        // Do not add a synthetic zero carry from a column that had no adder:
        // retaining the original queue schedule preserves eager XLS precision.
        if let Some(next) = high.first_mut() {
            if height >= 2 {
                next.push_back(carry);
            }
        }
    }
    result
}

/// Computes restoring long division, including XLS's zero-divisor results.
fn unsigned_divmod(numerator: &[Bit], denominator: &[Bit]) -> (Vec<Bit>, Vec<Bit>) {
    let nonzero = denominator.iter().copied().fold(Some(false), or);
    if nonzero == Some(false) {
        return (
            vec![Some(true); numerator.len()],
            vec![Some(false); denominator.len()],
        );
    }
    let divisor = resize_bits(denominator, denominator.len() + 1, false);
    let negative_divisor = negate_bits(&divisor);
    let mut quotient = vec![Some(false); numerator.len()];
    let mut remainder = vec![Some(false); denominator.len()];
    for i in (0..numerator.len()).rev() {
        remainder.insert(0, numerator[i]);
        quotient[i] = not(less_than(&remainder, &divisor, false));
        remainder = select_bits(
            quotient[i],
            &add_bits(&remainder, &negative_divisor),
            &remainder,
        );
        remainder.pop();
    }
    (
        select_bits(nonzero, &quotient, &vec![Some(true); quotient.len()]),
        select_bits(nonzero, &remainder, &vec![Some(false); remainder.len()]),
    )
}

fn signed_divmod(numerator: &[Bit], denominator: &[Bit]) -> (Vec<Bit>, Vec<Bit>) {
    let n_sign = numerator.last().copied().unwrap_or(Some(false));
    let d_sign = denominator.last().copied().unwrap_or(Some(false));
    let n_abs = select_bits(n_sign, &negate_bits(numerator), numerator);
    let d_abs = select_bits(d_sign, &negate_bits(denominator), denominator);
    let (quotient, remainder) = unsigned_divmod(&n_abs, &d_abs);
    let quotient = select_bits(xor(n_sign, d_sign), &negate_bits(&quotient), &quotient);
    let remainder = select_bits(n_sign, &negate_bits(&remainder), &remainder);
    let mut positive = vec![Some(true); numerator.len()];
    let mut negative = vec![Some(false); numerator.len()];
    if let (Some(positive), Some(negative)) = (positive.last_mut(), negative.last_mut()) {
        *positive = Some(false);
        *negative = Some(true);
    }
    let nonzero = denominator.iter().copied().fold(Some(false), or);
    let zero_divisor_result = select_bits(n_sign, &negative, &positive);
    (
        select_bits(nonzero, &quotient, &zero_divisor_result),
        remainder,
    )
}

/// Tests a host-sized index without narrowing an arbitrary-width operand.
fn can_equal_index(bits: &[Bit], index: usize) -> bool {
    if bits.len() < usize::BITS as usize && index >> bits.len() != 0 {
        return false;
    }
    bits.iter().enumerate().all(|(i, bit)| {
        let value = i < usize::BITS as usize && (index >> i) & 1 != 0;
        bit.is_none_or(|bit| bit == value)
    })
}

fn can_be_at_least(bits: &[Bit], bound: usize) -> bool {
    if bits
        .iter()
        .skip(usize::BITS as usize)
        .any(|bit| *bit != Some(false))
    {
        return true;
    }
    let max = bits
        .iter()
        .take(usize::BITS as usize)
        .enumerate()
        .fold(0usize, |max, (i, bit)| {
            max | (usize::from(*bit != Some(false)) << i)
        });
    max >= bound
}

fn join_bits(result: &mut Option<Vec<Bit>>, value: Vec<Bit>) {
    if let Some(result) = result {
        for (a, b) in result.iter_mut().zip(value) {
            if *a != b {
                *a = None;
            }
        }
    } else {
        *result = Some(value);
    }
}

/// Recognizes one exact amount or an entire domain beyond the saturation limit.
fn fixed_or_saturated_amount(bits: &[Bit], limit: usize) -> Option<usize> {
    let mut minimum = 0usize;
    let mut fully_known = true;
    for bit in bits.iter().rev() {
        fully_known &= bit.is_some();
        minimum = minimum
            .saturating_mul(2)
            .saturating_add(usize::from(bit.unwrap_or(false)))
            .min(limit);
    }
    (fully_known || minimum == limit).then_some(minimum)
}

/// Routes a fixed shift directly between packed facts, including fill bits.
pub(super) fn shift_fixed(
    a: &KnownBits,
    amount: usize,
    width: usize,
    right: bool,
    arithmetic: bool,
) -> KnownBits {
    let source = |i: usize| {
        if right {
            i.checked_add(amount)
        } else {
            i.checked_sub(amount)
        }
        .filter(|&i| i < a.bit_count())
    };
    let fill_known = !arithmetic || a.bit_count() == 0 || a.mask.msb();
    let fill_value = arithmetic && a.value.msb();
    KnownBits {
        mask: IrBits::from_lsb_fn(width, |i| {
            source(i).map_or(fill_known, |i| a.mask.get_bit(i).unwrap())
        }),
        value: IrBits::from_lsb_fn(width, |i| {
            source(i).map_or(fill_value, |i| a.value.get_bit(i).unwrap())
        }),
    }
}

fn shift_by(a: &[Bit], amount: usize, width: usize, right: bool, fill: Bit) -> Vec<Bit> {
    (0..width)
        .map(|i| {
            let source = if right {
                i.checked_add(amount)
            } else {
                i.checked_sub(amount)
            };
            source.and_then(|i| a.get(i).copied()).unwrap_or(fill)
        })
        .collect()
}

/// Joins all feasible shifts; this avoids losing selection correlations.
fn shifted(a: &[Bit], amount: &[Bit], width: usize, right: bool, arithmetic: bool) -> Vec<Bit> {
    if width == 0 {
        return Vec::new();
    }
    let fill = if arithmetic {
        a.last().copied().unwrap_or(Some(false))
    } else {
        Some(false)
    };
    if let Some(amount) = fixed_or_saturated_amount(amount, a.len()) {
        return shift_by(a, amount, width, right, fill);
    }
    let mut result = None;
    for shift in 0..a.len() {
        if !can_equal_index(amount, shift) {
            continue;
        }
        join_bits(&mut result, shift_by(a, shift, width, right, fill));
        if result
            .as_ref()
            .is_some_and(|bits| bits.iter().all(Option::is_none))
        {
            break;
        }
    }
    if can_be_at_least(amount, a.len()) {
        join_bits(&mut result, vec![fill; width]);
    }
    result.unwrap_or_else(|| vec![fill; width])
}

/// Handles linear-time shifts before imposing uncertain-shift work limits.
pub(super) fn try_shift_fixed(
    op: Binop,
    a: &KnownBits,
    amount: &KnownBits,
    width: usize,
) -> Option<KnownBits> {
    let arithmetic = op == Binop::Shra;
    let right = op != Binop::Shll;
    if width == 0 || (a.is_fully_known() && a.value.is_zero()) {
        return Some(KnownBits::constant(&IrBits::zero(width)));
    }
    if arithmetic && a.is_fully_known() && a.value == a.mask {
        return Some(KnownBits::constant(&IrBits::all_ones(width)));
    }
    let amount = fixed_or_saturated_amount(&amount.lsb_bits(), a.bit_count())?;
    Some(shift_fixed(a, amount, width, right, arithmetic))
}

/// Handles fixed-position and provably unchanged updates in linear time.
pub(super) fn try_slice_update_fixed(
    a: &KnownBits,
    start: &KnownBits,
    update: &KnownBits,
) -> Option<KnownBits> {
    if a.bit_count() == 0 || update.bit_count() == 0 {
        return Some(a.clone());
    }
    let offset = fixed_or_saturated_amount(&start.lsb_bits(), a.bit_count())?;
    if offset == a.bit_count() {
        return Some(a.clone());
    }
    let count = update.bit_count().min(a.bit_count() - offset);
    let update_index = |i: usize| i.checked_sub(offset).filter(|&i| i < count);
    Some(KnownBits {
        mask: IrBits::from_lsb_fn(a.bit_count(), |i| {
            update_index(i).map_or_else(
                || a.mask.get_bit(i).unwrap(),
                |i| update.mask.get_bit(i).unwrap(),
            )
        }),
        value: IrBits::from_lsb_fn(a.bit_count(), |i| {
            update_index(i).map_or_else(
                || a.value.get_bit(i).unwrap(),
                |i| update.value.get_bit(i).unwrap(),
            )
        }),
    })
}

pub(super) fn slice(a: &KnownBits, start: usize, width: usize) -> KnownBits {
    let input_index = |i| start.checked_add(i).filter(|&i| i < a.bit_count());
    KnownBits {
        // Bits beyond the input, including overflowing indices, are known
        // zeros rather than unknown bits.
        mask: IrBits::from_lsb_fn(width, |i| {
            input_index(i).is_none_or(|i| a.mask.get_bit(i).unwrap())
        }),
        value: IrBits::from_lsb_fn(width, |i| {
            input_index(i).is_some_and(|i| a.value.get_bit(i).unwrap())
        }),
    }
}

pub(super) fn resize(a: &KnownBits, width: usize, signed: bool) -> KnownBits {
    let sign_known = !signed || a.bit_count() == 0 || a.mask.msb();
    let sign_value = signed && a.value.msb();
    KnownBits {
        mask: IrBits::from_lsb_fn(width, |i| {
            if i < a.bit_count() {
                a.mask.get_bit(i).unwrap()
            } else {
                sign_known
            }
        }),
        value: IrBits::from_lsb_fn(width, |i| {
            if i < a.bit_count() {
                a.value.get_bit(i).unwrap()
            } else {
                sign_value
            }
        }),
    }
}

pub(super) fn dynamic_slice(a: &KnownBits, start: &KnownBits, width: usize) -> KnownBits {
    KnownBits::from_lsb_bits(&shifted(
        &a.lsb_bits(),
        &start.lsb_bits(),
        width,
        true,
        false,
    ))
}

/// Joins feasible in-bounds updates and the out-of-bounds no-op case.
pub(crate) fn slice_update(a: &KnownBits, start: &KnownBits, update: &KnownBits) -> KnownBits {
    let mut a = a.lsb_bits();
    let start = start.lsb_bits();
    let update = update.lsb_bits();
    if update.is_empty() {
        return KnownBits::from_lsb_bits(&a);
    }
    if let Some(offset) = fixed_or_saturated_amount(&start, a.len()) {
        let count = update.len().min(a.len() - offset);
        a[offset..offset + count].copy_from_slice(&update[..count]);
        return KnownBits::from_lsb_bits(&a);
    }
    let mut result = None;
    for offset in 0..a.len() {
        if !can_equal_index(&start, offset) {
            continue;
        }
        let mut value = a.clone();
        let count = update.len().min(a.len() - offset);
        value[offset..offset + count].copy_from_slice(&update[..count]);
        join_bits(&mut result, value);
        if result
            .as_ref()
            .is_some_and(|bits| bits.iter().all(Option::is_none))
        {
            break;
        }
    }
    if can_be_at_least(&start, a.len()) {
        join_bits(&mut result, a.clone());
    }
    KnownBits::from_lsb_bits(&result.unwrap_or(a))
}

/// Tracks the first set bit and the extra all-zero indicator.
pub(super) fn one_hot(a: &KnownBits, lsb_prio: bool) -> KnownBits {
    let a = a.lsb_bits();
    let mut result = vec![Some(false); a.len() + 1];
    let mut all_zero = Some(true);
    for step in 0..a.len() {
        let i = if lsb_prio { step } else { a.len() - 1 - step };
        result[i] = and(all_zero, a[i]);
        all_zero = and(all_zero, not(a[i]));
    }
    result[a.len()] = all_zero;
    KnownBits::from_lsb_bits(&result)
}

/// Determines each decoded bit by matching its index against the input facts.
pub(super) fn decode(a: &KnownBits, width: usize) -> KnownBits {
    let bits = a.lsb_bits();
    let fully_known = a.is_fully_known();
    KnownBits::from_lsb_bits(
        &(0..width)
            .map(|index| {
                if !can_equal_index(&bits, index) {
                    Some(false)
                } else if fully_known {
                    Some(true)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>(),
    )
}

/// ORs the indices of every potentially selected input bit.
pub(crate) fn encode(a: &KnownBits) -> KnownBits {
    let input = a.lsb_bits();
    let width = if input.len() <= 1 {
        0
    } else {
        usize::BITS as usize - (input.len() - 1).leading_zeros() as usize
    };
    let mut result = vec![Some(false); width];
    for (i, &bit) in input.iter().enumerate() {
        for (j, result) in result.iter_mut().enumerate() {
            if (i >> j) & 1 != 0 {
                *result = or(*result, bit);
            }
        }
    }
    KnownBits::from_lsb_bits(&result)
}

/// Evaluates a scalar unary operation over three-valued input bits.
pub(crate) fn unop(op: Unop, a: &KnownBits) -> KnownBits {
    match op {
        Unop::Identity => a.clone(),
        Unop::Not => KnownBits {
            mask: a.mask.clone(),
            value: a.mask.xor(&a.value),
        },
        Unop::Neg => KnownBits::from_lsb_bits(&negate_bits(&a.lsb_bits())),
        Unop::Reverse => KnownBits {
            mask: IrBits::from_lsb_fn(a.bit_count(), |i| {
                a.mask.get_bit(a.bit_count() - 1 - i).unwrap()
            }),
            value: IrBits::from_lsb_fn(a.bit_count(), |i| {
                a.value.get_bit(a.bit_count() - 1 - i).unwrap()
            }),
        },
        Unop::OrReduce => {
            let has_one = !a.value.is_zero();
            KnownBits {
                mask: IrBits::bool(has_one || a.is_fully_known()),
                value: IrBits::bool(has_one),
            }
        }
        Unop::AndReduce => {
            let has_zero = a.mask != a.value;
            let known = has_zero || a.is_fully_known();
            KnownBits {
                mask: IrBits::bool(known),
                value: IrBits::bool(known && !has_zero),
            }
        }
        Unop::XorReduce => {
            let known = a.is_fully_known();
            let parity = known
                && a.value
                    .limbs()
                    .iter()
                    .fold(0, |parity, limb| parity ^ (limb.count_ones() & 1))
                    != 0;
            KnownBits {
                mask: IrBits::bool(known),
                value: IrBits::bool(parity),
            }
        }
    }
}

/// Evaluates bitwise operations or concatenates operands in IR order.
pub(crate) fn nary(op: NaryOp, args: &[&KnownBits], width: usize) -> KnownBits {
    if op == NaryOp::Concat {
        let width = args.iter().fold(0usize, |width, arg| {
            width
                .checked_add(arg.bit_count())
                .expect("concat width overflow")
        });
        let mut masks = args
            .iter()
            .rev()
            .flat_map(|arg| (0..arg.bit_count()).map(move |i| arg.mask.get_bit(i).unwrap()));
        let mut values = args
            .iter()
            .rev()
            .flat_map(|arg| (0..arg.bit_count()).map(move |i| arg.value.get_bit(i).unwrap()));
        return KnownBits {
            mask: IrBits::from_lsb_fn(width, |_| masks.next().unwrap()),
            value: IrBits::from_lsb_fn(width, |_| values.next().unwrap()),
        };
    }
    let identity = matches!(op, NaryOp::And | NaryOp::Nand);
    let mut args = args.iter();
    let mut result = match args.next() {
        Some(first) => {
            assert_eq!(first.bit_count(), width);
            (*first).clone()
        }
        None => KnownBits::constant(&if identity {
            IrBits::all_ones(width)
        } else {
            IrBits::zero(width)
        }),
    };
    for a in args {
        assert_eq!(a.bit_count(), width);
        result = match op {
            NaryOp::And | NaryOp::Nand => {
                let value = result.value.and(&a.value);
                let zeros = result.mask.xor(&result.value).or(&a.mask.xor(&a.value));
                KnownBits {
                    mask: zeros.or(&value),
                    value,
                }
            }
            NaryOp::Or | NaryOp::Nor => {
                let value = result.value.or(&a.value);
                let zeros = result.mask.xor(&result.value).and(&a.mask.xor(&a.value));
                KnownBits {
                    mask: zeros.or(&value),
                    value,
                }
            }
            NaryOp::Xor => {
                let mask = result.mask.and(&a.mask);
                KnownBits {
                    value: result.value.xor(&a.value).and(&mask),
                    mask,
                }
            }
            NaryOp::Concat => unreachable!("concat handled above"),
        };
    }
    if matches!(op, NaryOp::Nand | NaryOp::Nor) {
        result.value = result.mask.xor(&result.value);
    }
    result
}

/// Evaluates arithmetic, comparisons, shifts, and gating without assumptions.
pub(super) fn binop(op: Binop, a: &KnownBits, b: &KnownBits, out_width: usize) -> KnownBits {
    if a.is_fully_known() && b.is_fully_known() {
        let result = match op {
            Binop::Udiv => Some(a.value.udiv(&b.value)),
            Binop::Umod => Some(a.value.umod(&b.value)),
            Binop::Sdiv => Some(a.value.sdiv(&b.value)),
            Binop::Smod => Some(a.value.smod(&b.value)),
            _ => None,
        };
        if let Some(result) = result {
            return KnownBits::constant(&result);
        }
    }
    let av = a.lsb_bits();
    let bv = b.lsb_bits();
    let result = match op {
        Binop::Add => add_bits(&av, &bv),
        Binop::Sub => sub_bits(&av, &bv),
        Binop::Eq => vec![equal_bits(&av, &bv)],
        Binop::Ne => vec![not(equal_bits(&av, &bv))],
        Binop::Ult => vec![less_than(&av, &bv, false)],
        Binop::Ule => vec![not(less_than(&bv, &av, false))],
        Binop::Ugt => vec![less_than(&bv, &av, false)],
        Binop::Uge => vec![not(less_than(&av, &bv, false))],
        Binop::Slt => vec![less_than(&av, &bv, true)],
        Binop::Sle => vec![not(less_than(&bv, &av, true))],
        Binop::Sgt => vec![less_than(&bv, &av, true)],
        Binop::Sge => vec![not(less_than(&av, &bv, true))],
        Binop::Gate => bv.iter().map(|&bit| and(av[0], bit)).collect(),
        Binop::Shll => shifted(&av, &bv, out_width, false, false),
        Binop::Shrl => shifted(&av, &bv, out_width, true, false),
        Binop::Shra => shifted(&av, &bv, out_width, true, true),
        Binop::Umul | Binop::Smul => {
            let signed = op == Binop::Smul;
            let natural_width = av.len().saturating_add(bv.len());
            let width = out_width.min(natural_width);
            let product = if a.is_fully_known() && b.is_fully_known() {
                let product = if signed {
                    a.value.smul(&b.value)
                } else {
                    a.value.umul(&b.value)
                };
                KnownBits::constant(&product).lsb_bits()
            } else if signed && !av.is_empty() && !bv.is_empty() {
                let extended_width = av.len().max(bv.len()).saturating_mul(2);
                multiply_bits(
                    &resize_bits(&av, extended_width, true),
                    &resize_bits(&bv, extended_width, true),
                    width,
                )
            } else {
                multiply_bits(&av, &bv, width)
            };
            resize_bits(&product, out_width, signed)
        }
        Binop::Udiv => unsigned_divmod(&av, &bv).0,
        Binop::Umod => unsigned_divmod(&av, &bv).1,
        Binop::Sdiv => signed_divmod(&av, &bv).0,
        Binop::Smod => signed_divmod(&av, &bv).1,
        Binop::Umulp | Binop::Smulp => vec![None; out_width],
    };
    debug_assert_eq!(result.len(), out_width);
    KnownBits::from_lsb_bits(&result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(width: usize, value: u64) -> IrBits {
        IrBits::make_ubits(width, value).unwrap()
    }

    fn ternary(text: &str) -> KnownBits {
        KnownBits::from_lsb_bits(
            &text
                .chars()
                .rev()
                .map(|c| match c {
                    '0' => Some(false),
                    '1' => Some(true),
                    'X' => None,
                    _ => panic!("invalid test ternary"),
                })
                .collect::<Vec<_>>(),
        )
    }

    fn abstract_values(width: usize) -> Vec<KnownBits> {
        (0..3usize.pow(width as u32))
            .map(|mut value| {
                let bits = (0..width)
                    .map(|_| {
                        let bit = match value % 3 {
                            0 => Some(false),
                            1 => Some(true),
                            _ => None,
                        };
                        value /= 3;
                        bit
                    })
                    .collect::<Vec<_>>();
                KnownBits::from_lsb_bits(&bits)
            })
            .collect()
    }

    /// Retains the per-bit formulation as an independent packed-path reference.
    fn scalar_unop(op: Unop, a: &KnownBits) -> Vec<Bit> {
        let a = a.lsb_bits();
        match op {
            Unop::Identity => a,
            Unop::Not => a.into_iter().map(not).collect(),
            Unop::Neg => negate_bits(&a),
            Unop::Reverse => a.into_iter().rev().collect(),
            Unop::OrReduce => vec![a.into_iter().fold(Some(false), or)],
            Unop::AndReduce => vec![a.into_iter().fold(Some(true), and)],
            Unop::XorReduce => vec![a.into_iter().fold(Some(false), xor)],
        }
    }

    /// Retains the per-bit formulation, including empty-operand identities.
    fn scalar_nary(op: NaryOp, args: &[&KnownBits], width: usize) -> Vec<Bit> {
        if op == NaryOp::Concat {
            return args.iter().rev().flat_map(|a| a.lsb_bits()).collect();
        }
        let mut result = vec![Some(matches!(op, NaryOp::And | NaryOp::Nand)); width];
        for arg in args {
            for (result, bit) in result.iter_mut().zip(arg.lsb_bits()) {
                *result = match op {
                    NaryOp::And | NaryOp::Nand => and(*result, bit),
                    NaryOp::Or | NaryOp::Nor => or(*result, bit),
                    NaryOp::Xor => xor(*result, bit),
                    NaryOp::Concat => unreachable!("concat handled above"),
                };
            }
        }
        if matches!(op, NaryOp::Nand | NaryOp::Nor) {
            result.iter_mut().for_each(|bit| *bit = not(*bit));
        }
        result
    }

    /// Checks both transfer precision and the canonical packed representation.
    fn assert_packed_result(actual: KnownBits, expected: &[Bit]) {
        assert_eq!(actual.bit_count(), expected.len());
        assert_eq!(actual.mask.get_bit_count(), actual.value.get_bit_count());
        assert_eq!(actual.value.and(&actual.mask), actual.value);
        assert_eq!(actual.lsb_bits(), expected);
    }

    #[test]
    fn packed_bitwise_transfers_match_scalar_exhaustively() {
        let nary_ops = [
            NaryOp::And,
            NaryOp::Nand,
            NaryOp::Or,
            NaryOp::Nor,
            NaryOp::Xor,
        ];
        let unary_ops = [
            Unop::Identity,
            Unop::Not,
            Unop::Reverse,
            Unop::AndReduce,
            Unop::OrReduce,
            Unop::XorReduce,
        ];
        for width in 0..=3 {
            let values = abstract_values(width);
            for op in nary_ops {
                assert_packed_result(nary(op, &[], width), &scalar_nary(op, &[], width));
            }
            for a in &values {
                for op in unary_ops {
                    assert_packed_result(unop(op, a), &scalar_unop(op, a));
                }
                for op in nary_ops {
                    assert_packed_result(nary(op, &[a], width), &scalar_nary(op, &[a], width));
                }
                for b in &values {
                    for op in nary_ops {
                        assert_packed_result(
                            nary(op, &[a, b], width),
                            &scalar_nary(op, &[a, b], width),
                        );
                        if width <= 2 {
                            for c in &values {
                                assert_packed_result(
                                    nary(op, &[a, b, c], width),
                                    &scalar_nary(op, &[a, b, c], width),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn packed_reindexing_matches_scalar_exhaustively() {
        for input_width in 0..=4 {
            for a in abstract_values(input_width) {
                let bits = a.lsb_bits();
                for width in 0..=7 {
                    for signed in [false, true] {
                        assert_packed_result(
                            resize(&a, width, signed),
                            &resize_bits(&bits, width, signed),
                        );
                    }
                    for start in (0..=input_width + 2).chain([usize::MAX - 1, usize::MAX]) {
                        let expected = (0..width)
                            .map(|i| {
                                start
                                    .checked_add(i)
                                    .and_then(|i| bits.get(i).copied())
                                    .unwrap_or(Some(false))
                            })
                            .collect::<Vec<_>>();
                        assert_packed_result(slice(&a, start, width), &expected);
                    }
                }
            }
        }
        for a_width in 0..=3 {
            for b_width in 0..=3 {
                for a in abstract_values(a_width) {
                    for b in abstract_values(b_width) {
                        let args = [&a, &b];
                        assert_packed_result(
                            nary(NaryOp::Concat, &args, a_width + b_width),
                            &scalar_nary(NaryOp::Concat, &args, a_width + b_width),
                        );
                    }
                }
            }
        }
        assert_packed_result(nary(NaryOp::Concat, &[], 0), &[]);
    }

    #[test]
    fn packed_transfers_cover_limb_boundaries_and_metamorphic_identities() {
        for width in [0, 1, 7, 8, 63, 64, 65, 127, 128, 129, 257] {
            for seed in 0..6 {
                let values = (0..3)
                    .map(|operand| {
                        KnownBits::from_lsb_bits(
                            &(0..width)
                                .map(|i| match (i * (operand + 1) + seed + operand) % 3 {
                                    0 => Some(false),
                                    1 => Some(true),
                                    _ => None,
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>();
                let args = values.iter().collect::<Vec<_>>();
                let a = &values[0];
                let b = &values[1];
                for op in [
                    NaryOp::And,
                    NaryOp::Nand,
                    NaryOp::Or,
                    NaryOp::Nor,
                    NaryOp::Xor,
                ] {
                    assert_packed_result(nary(op, &args, width), &scalar_nary(op, &args, width));
                    assert_eq!(nary(op, &[a, b], width), nary(op, &[b, a], width));
                }
                for op in [
                    Unop::Identity,
                    Unop::Not,
                    Unop::Reverse,
                    Unop::AndReduce,
                    Unop::OrReduce,
                    Unop::XorReduce,
                ] {
                    assert_packed_result(unop(op, a), &scalar_unop(op, a));
                }
                assert_eq!(unop(Unop::Not, &unop(Unop::Not, a)), *a);
                assert_eq!(unop(Unop::Reverse, &unop(Unop::Reverse, a)), *a);
                let not_a = unop(Unop::Not, a);
                let not_b = unop(Unop::Not, b);
                assert_eq!(
                    nary(NaryOp::Nand, &[a, b], width),
                    nary(NaryOp::Or, &[&not_a, &not_b], width),
                );
                assert_eq!(
                    nary(NaryOp::Nor, &[a, b], width),
                    nary(NaryOp::And, &[&not_a, &not_b], width),
                );
                let joined = nary(NaryOp::Concat, &args, width * args.len());
                assert_packed_result(
                    joined.clone(),
                    &scalar_nary(NaryOp::Concat, &args, width * args.len()),
                );
                for (index, original) in args.iter().rev().enumerate() {
                    assert_eq!(slice(&joined, index * width, width), **original);
                }
                for target_width in [0, width / 2, width, width + 1, width + 65] {
                    for signed in [false, true] {
                        assert_packed_result(
                            resize(a, target_width, signed),
                            &resize_bits(&a.lsb_bits(), target_width, signed),
                        );
                    }
                }
                for start in [0, 1, 63, 64, 65, width, usize::MAX] {
                    let bits = a.lsb_bits();
                    let expected = (0..width + 1)
                        .map(|i| {
                            start
                                .checked_add(i)
                                .and_then(|i| bits.get(i).copied())
                                .unwrap_or(Some(false))
                        })
                        .collect::<Vec<_>>();
                    assert_packed_result(slice(a, start, width + 1), &expected);
                }
            }
        }
    }

    fn concretizations(value: &KnownBits) -> Vec<IrBits> {
        assert!(value.bit_count() <= 8);
        (0..1u64 << value.bit_count())
            .map(|v| bits(value.bit_count(), v))
            .filter(|v| value.contains(v))
            .collect()
    }

    fn expected(op: Binop, a: &IrBits, b: &IrBits, width: usize) -> IrBits {
        let result = match op {
            Binop::Add => a.add(b),
            Binop::Sub => a.sub(b),
            Binop::Umul => a.umul(b),
            Binop::Smul => a.smul(b),
            Binop::Udiv => a.udiv(b),
            Binop::Sdiv => a.sdiv(b),
            Binop::Umod => a.umod(b),
            Binop::Smod => a.smod(b),
            Binop::Eq => IrBits::bool(a == b),
            Binop::Ne => IrBits::bool(a != b),
            Binop::Ult => IrBits::bool(a.ult(b)),
            Binop::Ule => IrBits::bool(a.ule(b)),
            Binop::Ugt => IrBits::bool(a.ugt(b)),
            Binop::Uge => IrBits::bool(a.uge(b)),
            Binop::Slt => IrBits::bool(a.slt(b)),
            Binop::Sle => IrBits::bool(a.sle(b)),
            Binop::Sgt => IrBits::bool(a.sgt(b)),
            Binop::Sge => IrBits::bool(a.sge(b)),
            Binop::Shll => a.shll(b.to_u64().unwrap() as i64),
            Binop::Shrl => a.shrl(b.to_u64().unwrap() as i64),
            Binop::Shra => a.shra(b.to_u64().unwrap() as i64),
            _ => panic!("operation not included in this test"),
        };
        resize(&KnownBits::constant(&result), width, op == Binop::Smul)
            .value
            .clone()
    }

    #[test]
    fn domain_normalizes_and_checks_widths() {
        let value = KnownBits::from_mask_value(bits(4, 0b0101), bits(4, 0b1111)).unwrap();
        assert_eq!(value.to_ternary_string(), "X1X1");
        assert_eq!(value.known_bit_count(), 2);
        assert!(value.contains(&bits(4, 0b1101)));
        assert!(!value.contains(&bits(4, 0b1100)));
        assert!(!value.contains(&bits(5, 0b1101)));
        assert!(KnownBits::from_mask_value(bits(4, 0), bits(5, 0)).is_err());
        assert_eq!(value.join(&ternary("11X1")), value);
        assert!(KnownBits::unknown(0).is_fully_known());
        assert_eq!(KnownBits::unknown(0).to_ternary_string(), "");
    }

    #[test]
    fn arithmetic_preserves_decided_carries_and_remainders() {
        // The possible sums are 100 and 110. At bit 1, a known-one operand
        // and carry force the next carry even though the other operand is X.
        assert_eq!(
            binop(Binop::Add, &ternary("011"), &ternary("0X1"), 3).to_ternary_string(),
            "1X0"
        );
        // 7 % 5 == 2 and 7 % 7 == 0; both endpoint bits are known zeros.
        assert_eq!(
            binop(Binop::Umod, &ternary("111"), &ternary("1X1"), 3).to_ternary_string(),
            "0X0"
        );
    }

    #[test]
    fn comparisons_and_updates_retain_unconditional_facts() {
        // A signed one-bit value is either -1 or 0, so both satisfy x >= -1.
        assert_eq!(
            binop(Binop::Sge, &ternary("X"), &ternary("1"), 1).to_ternary_string(),
            "1"
        );
        // Replacing either bit of 11 with a one leaves the value unchanged.
        assert_eq!(
            slice_update(&ternary("11"), &ternary("X"), &ternary("1")).to_ternary_string(),
            "11"
        );
    }

    fn assert_refines(coarse: &KnownBits, refined: &KnownBits) {
        assert_eq!(refined.mask.and(&coarse.mask), coarse.mask);
        assert_eq!(refined.value.and(&coarse.mask), coarse.value);
    }

    #[test]
    fn binary_transfers_are_monotone_for_single_bit_refinements() {
        let operations = [
            Binop::Add,
            Binop::Sub,
            Binop::Umul,
            Binop::Smul,
            Binop::Udiv,
            Binop::Sdiv,
            Binop::Umod,
            Binop::Smod,
            Binop::Eq,
            Binop::Ne,
            Binop::Ult,
            Binop::Ule,
            Binop::Ugt,
            Binop::Uge,
            Binop::Slt,
            Binop::Sle,
            Binop::Sgt,
            Binop::Sge,
            Binop::Shll,
            Binop::Shrl,
            Binop::Shra,
        ];
        for width in 0..=2 {
            let values = abstract_values(width);
            for a in &values {
                for b in &values {
                    for op in operations {
                        let result_width = if matches!(
                            op,
                            Binop::Eq
                                | Binop::Ne
                                | Binop::Ult
                                | Binop::Ule
                                | Binop::Ugt
                                | Binop::Uge
                                | Binop::Slt
                                | Binop::Sle
                                | Binop::Sgt
                                | Binop::Sge
                        ) {
                            1
                        } else {
                            width
                        };
                        let coarse = binop(op, a, b, result_width);
                        for operand in 0..2 {
                            let source = if operand == 0 { a } else { b };
                            for (index, bit) in source.lsb_bits().iter().enumerate() {
                                if bit.is_some() {
                                    continue;
                                }
                                for value in [false, true] {
                                    let mut refined = source.lsb_bits();
                                    refined[index] = Some(value);
                                    let refined = KnownBits::from_lsb_bits(&refined);
                                    let refined = if operand == 0 {
                                        binop(op, &refined, b, result_width)
                                    } else {
                                        binop(op, a, &refined, result_width)
                                    };
                                    assert_refines(&coarse, &refined);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fixed_or_saturated_amount_is_sound_for_every_small_domain() {
        for width in 0..=4 {
            for value in abstract_values(width) {
                for limit in [0, 1, 3, 129] {
                    if let Some(amount) = fixed_or_saturated_amount(&value.lsb_bits(), limit) {
                        for concrete in concretizations(&value) {
                            assert_eq!((concrete.to_u64().unwrap() as usize).min(limit), amount);
                        }
                    }
                }
            }
        }
        let mut wide = vec![None; 129];
        wide[128] = Some(true);
        assert_eq!(fixed_or_saturated_amount(&wide, 257), Some(257));
        let known_zero = KnownBits::constant(&IrBits::zero(129));
        let data = KnownBits::from_lsb_bits(&wide);
        assert_eq!(binop(Binop::Shll, &data, &known_zero, 129), data);
        assert_eq!(binop(Binop::Shrl, &data, &known_zero, 129), data);
        assert_eq!(binop(Binop::Shra, &data, &known_zero, 129), data);
        assert_eq!(slice_update(&data, &data, &known_zero), data);
        assert_eq!(binop(Binop::Shrl, &data, &data, 129), known_zero);
        assert_eq!(
            binop(Binop::Shra, &data, &data, 129),
            KnownBits::constant(&IrBits::all_ones(129))
        );
    }

    #[test]
    fn arithmetic_and_comparisons_are_sound_exhaustively() {
        let operations = [
            Binop::Add,
            Binop::Sub,
            Binop::Umul,
            Binop::Smul,
            Binop::Udiv,
            Binop::Sdiv,
            Binop::Umod,
            Binop::Smod,
            Binop::Eq,
            Binop::Ne,
            Binop::Ult,
            Binop::Ule,
            Binop::Ugt,
            Binop::Uge,
            Binop::Slt,
            Binop::Sle,
            Binop::Sgt,
            Binop::Sge,
        ];
        for width in 0..=3 {
            let values = abstract_values(width);
            let concrete = values.iter().map(concretizations).collect::<Vec<_>>();
            for (ai, a) in values.iter().enumerate() {
                for (bi, b) in values.iter().enumerate() {
                    for op in operations {
                        let output_width = if matches!(
                            op,
                            Binop::Eq
                                | Binop::Ne
                                | Binop::Ult
                                | Binop::Ule
                                | Binop::Ugt
                                | Binop::Uge
                                | Binop::Slt
                                | Binop::Sle
                                | Binop::Sgt
                                | Binop::Sge
                        ) {
                            1
                        } else {
                            width
                        };
                        let result = binop(op, a, b, output_width);
                        for av in &concrete[ai] {
                            for bv in &concrete[bi] {
                                assert!(
                                    result.contains(&expected(op, av, bv, output_width)),
                                    "{op:?}({}, {}) = {} excludes ({av}, {bv})",
                                    a.to_ternary_string(),
                                    b.to_ternary_string(),
                                    result.to_ternary_string()
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn multiply_widths_and_sign_extension_are_sound() {
        for a_width in 0..=3 {
            for b_width in 0..=3 {
                for a in abstract_values(a_width) {
                    for b in abstract_values(b_width) {
                        for output_width in [0, 1, a_width + b_width, a_width + b_width + 2] {
                            for op in [Binop::Umul, Binop::Smul] {
                                let result = binop(op, &a, &b, output_width);
                                for av in concretizations(&a) {
                                    for bv in concretizations(&b) {
                                        assert!(
                                            result.contains(&expected(op, &av, &bv, output_width)),
                                            "{op:?}: {a:?} {b:?} -> {result:?}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn shifts_match_exact_joins_with_different_amount_widths() {
        for width in 0..=3 {
            for a in abstract_values(width) {
                for amount_width in 0..=3 {
                    for b in abstract_values(amount_width) {
                        for op in [Binop::Shll, Binop::Shrl, Binop::Shra] {
                            let result = binop(op, &a, &b, width);
                            let mut exact: Option<KnownBits> = None;
                            for av in concretizations(&a) {
                                for bv in concretizations(&b) {
                                    let value = KnownBits::constant(&expected(op, &av, &bv, width));
                                    exact =
                                        Some(exact.map_or(value.clone(), |previous| {
                                            previous.join(&value)
                                        }));
                                }
                            }
                            assert_eq!(result, exact.unwrap(), "{op:?} {a:?} {b:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn dynamic_operations_are_sound_exhaustively() {
        for a in abstract_values(3) {
            for start in abstract_values(3) {
                let slice = dynamic_slice(&a, &start, 2);
                for av in concretizations(&a) {
                    for sv in concretizations(&start) {
                        let amount = sv.to_u64().unwrap() as i64;
                        assert!(slice.contains(&av.shrl(amount).width_slice(0, 2)));
                    }
                }
                for update in abstract_values(2) {
                    let result = slice_update(&a, &start, &update);
                    for av in concretizations(&a) {
                        for sv in concretizations(&start) {
                            for uv in concretizations(&update) {
                                let offset = sv.to_u64().unwrap() as usize;
                                let mut expected =
                                    (0..3).map(|i| av.get_bit(i).unwrap()).collect::<Vec<_>>();
                                for i in 0..2 {
                                    if offset + i < expected.len() {
                                        expected[offset + i] = uv.get_bit(i).unwrap();
                                    }
                                }
                                assert!(result.contains(&IrBits::from_lsb_is_0(&expected)));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn encoders_and_one_hot_are_sound_exhaustively() {
        for width in 0..=4 {
            for a in abstract_values(width) {
                for av in concretizations(&a) {
                    let raw = av.to_u64().unwrap();
                    for lsb_prio in [false, true] {
                        let index = if raw == 0 {
                            width
                        } else if lsb_prio {
                            raw.trailing_zeros() as usize
                        } else {
                            u64::BITS as usize - 1 - raw.leading_zeros() as usize
                        };
                        assert!(one_hot(&a, lsb_prio).contains(&bits(width + 1, 1 << index)));
                    }
                    for output_width in 0..=8 {
                        let expected = if raw < output_width as u64 {
                            1 << raw
                        } else {
                            0
                        };
                        assert!(decode(&a, output_width).contains(&bits(output_width, expected)));
                    }
                    let encoded = encode(&a);
                    let expected = (0..width)
                        .filter(|&i| raw & (1 << i) != 0)
                        .fold(0usize, |v, i| v | i);
                    assert!(encoded.contains(&bits(encoded.bit_count(), expected as u64)));
                }
            }
        }
    }

    #[test]
    fn wide_facts_and_zero_divisors_do_not_narrow_through_host_integers() {
        for width in [65, 129, 257] {
            let zero = KnownBits::constant(&IrBits::zero(width));
            let all_ones = KnownBits::constant(&IrBits::all_ones(width));
            let unknown = KnownBits::unknown(width);
            assert_eq!(binop(Binop::Udiv, &unknown, &zero, width), all_ones);
            assert_eq!(binop(Binop::Umod, &unknown, &zero, width), zero);
            assert_eq!(binop(Binop::Smod, &unknown, &zero, width), zero);
            let signed_zero_division = binop(Binop::Sdiv, &unknown, &zero, width);
            assert!(signed_zero_division.contains(&IrBits::signed_min_value(width)));
            assert!(signed_zero_division.contains(&IrBits::signed_max_value(width)));
            assert_eq!(binop(Binop::Umul, &unknown, &zero, width), zero);
            let mut amount = vec![Some(false); width];
            amount[width - 1] = Some(true);
            let huge_amount = KnownBits::from_lsb_bits(&amount);
            assert_eq!(binop(Binop::Shll, &unknown, &huge_amount, width), zero);
            assert_eq!(binop(Binop::Shrl, &unknown, &huge_amount, width), zero);
            assert_eq!(slice_update(&unknown, &huge_amount, &all_ones), unknown);
        }
        assert_eq!(
            binop(Binop::Add, &ternary("01X"), &ternary("011"), 3).to_ternary_string(),
            "1XX"
        );
    }
}
