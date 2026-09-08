// SPDX-License-Identifier: Apache-2.0

//! Bounded, width-independent scalar interval transfers.

use crate::IrBits;
use crate::ir::{Binop, NaryOp, Unop};
use crate::known_bits::{KnownBits, bits as known};

use super::IntervalSet;
use super::interval_set::GapDistance;
use super::policy::{BIT_WORK_BUDGET, EXACT_VALUES, INTERVAL_COMBINATIONS, RESULT_INTERVALS};

#[cfg(test)]
#[path = "ops_small_domain_test.rs"]
mod small_domain_tests;

/// Resizes a bit pattern without converting arbitrary-width values to integers.
pub(crate) fn resize(value: &IrBits, width: usize, signed: bool) -> IrBits {
    let old_width = value.get_bit_count();
    let fill = signed && value.is_negative();
    IrBits::from_lsb_fn(width, |i| {
        if i < old_width {
            value.get_bit(i).unwrap()
        } else {
            fill
        }
    })
}

/// Creates a structural index, truncating only when requested by IR semantics.
pub(crate) fn index_bits(width: usize, value: usize) -> IrBits {
    IrBits::from_lsb_fn(width, |i| {
        i < usize::BITS as usize && ((value >> i) & 1) != 0
    })
}

/// Converts a shift/index only after bounding it by a structural host size.
pub(crate) fn saturating_index(value: &IrBits, limit: usize) -> usize {
    let mut result = 0usize;
    for (i, &limb) in value.limbs().iter().enumerate() {
        if limb == 0 {
            continue;
        }
        if i != 0 || limb > limit as u64 {
            return limit;
        }
        result = limb as usize;
    }
    result.min(limit)
}

fn make(width: usize, intervals: impl IntoIterator<Item = (IrBits, IrBits)>) -> IntervalSet {
    IntervalSet::from_intervals(width, intervals).expect("transfer endpoints have matching widths")
}

fn zero(width: usize) -> IntervalSet {
    IntervalSet::singleton(IrBits::zero(width))
}

fn bool_result(value: bool) -> IntervalSet {
    IntervalSet::singleton(IrBits::bool(value))
}

fn is_empty(value: &IntervalSet) -> bool {
    value.intervals().is_empty()
}

fn exact_unary(
    a: &IntervalSet,
    width: usize,
    f: impl Fn(&IrBits) -> IrBits,
) -> Option<IntervalSet> {
    Some(make(
        width,
        a.values_up_to(EXACT_VALUES)?.into_iter().map(|v| {
            let value = f(&v);
            (value.clone(), value)
        }),
    ))
}

fn exact_binary(
    a: &IntervalSet,
    b: &IntervalSet,
    width: usize,
    f: impl Fn(&IrBits, &IrBits) -> IrBits,
) -> Option<IntervalSet> {
    let count = a.cardinality_up_to(EXACT_VALUES)?;
    if count == 0 {
        return Some(IntervalSet::empty(width));
    }
    let rhs = b.values_up_to(EXACT_VALUES / count)?;
    let lhs = a.values_up_to(EXACT_VALUES)?;
    let mut points = Vec::with_capacity(lhs.len() * rhs.len());
    for a in lhs {
        for b in &rhs {
            let value = f(&a, b);
            points.push((value.clone(), value));
        }
    }
    Some(make(width, points))
}

/// Matches XLS's least-gap fragmentation reduction, including tie-breaking.
fn reduce_fragmentation(args: &[&IntervalSet]) -> Vec<IntervalSet> {
    let mut result: Vec<_> = args.iter().map(|x| (*x).clone()).collect();
    if result.iter().any(is_empty) {
        return result;
    }
    loop {
        if result
            .iter()
            .try_fold(1usize, |p, x| p.checked_mul(x.intervals().len()))
            .is_some_and(|p| p <= INTERVAL_COMBINATIONS)
        {
            return result;
        }
        let mut best: Option<(usize, GapDistance)> = None;
        for (i, set) in result.iter().enumerate() {
            for pair in set.intervals().windows(2) {
                let gap = GapDistance::between(pair[0].upper(), pair[1].lower());
                if best.as_ref().is_none_or(|(_, distance)| gap < *distance) {
                    best = Some((i, gap));
                }
            }
        }
        let i = best.expect("excessive product has a fragmented operand").0;
        result[i] = result[i].minimize(result[i].intervals().len() - 1);
    }
}

struct Endpoint {
    value: IrBits,
    overflow: bool,
    twice: bool,
}

impl Endpoint {
    fn plain(value: IrBits) -> Self {
        Self {
            value,
            overflow: false,
            twice: false,
        }
    }
}

/// Evaluates independent endpoint combinations with XLS's overflow policy.
fn endpoints(
    a: &IntervalSet,
    b: &IntervalSet,
    width: usize,
    rhs_antitone: bool,
    f: impl Fn(&IrBits, &IrBits) -> Endpoint,
) -> IntervalSet {
    let args = reduce_fragmentation(&[a, b]);
    endpoints_prepared(&args[0], &args[1], width, rhs_antitone, f).into_minimized(RESULT_INTERVALS)
}

/// Evaluates already-budgeted partitions without independently coarsening them.
fn endpoints_prepared(
    a: &IntervalSet,
    b: &IntervalSet,
    width: usize,
    rhs_antitone: bool,
    f: impl Fn(&IrBits, &IrBits) -> Endpoint,
) -> IntervalSet {
    let mut result = Vec::new();
    for lhs in a.intervals() {
        for rhs in b.intervals() {
            let low = f(
                lhs.lower(),
                if rhs_antitone {
                    rhs.upper()
                } else {
                    rhs.lower()
                },
            );
            let high = f(
                lhs.upper(),
                if rhs_antitone {
                    rhs.lower()
                } else {
                    rhs.upper()
                },
            );
            let lhs_precise = lhs.lower() == lhs.upper();
            let rhs_precise = rhs.lower() == rhs.upper();
            // These checks apply to this interval pair, not just the whole
            // operand sets. A tighter upstream fact can split an interval into
            // points; overflowing constant pairs still have an exact result.
            if (!low.overflow && !high.overflow) || (lhs_precise && rhs_precise) {
                result.push((low.value, high.value));
            } else if (low.overflow && high.overflow)
                || low.twice
                || high.twice
                || high.value.ugt(&low.value)
            {
                return IntervalSet::full(width);
            } else {
                result.push((low.value, IrBits::all_ones(width)));
                result.push((IrBits::zero(width), high.value));
            }
        }
    }
    make(width, result)
}

/// Returns positive nonzero values and absolute negative values independently.
pub(crate) fn signed_parts(a: &IntervalSet) -> (IntervalSet, IntervalSet) {
    let w = a.width();
    if w == 0 {
        return (IntervalSet::empty(0), IntervalSet::empty(0));
    }
    let positive = a.intersect(&make(w, [(index_bits(w, 1), IrBits::signed_max_value(w))]));
    // bits[1] has no positive nonzero value: the above interval would wrap.
    let positive = if w == 1 {
        IntervalSet::empty(w)
    } else {
        positive
    };
    let negative = a.intersect(&make(
        w,
        [(IrBits::signed_min_value(w), IrBits::all_ones(w))],
    ));
    (positive, negate(&negative))
}

fn negate(a: &IntervalSet) -> IntervalSet {
    make(
        a.width(),
        a.intervals()
            .iter()
            .map(|i| (i.upper().negate(), i.lower().negate())),
    )
}

fn unsigned_multiply(a: &IntervalSet, b: &IntervalSet, width: usize) -> IntervalSet {
    if let Some(result) = exact_binary(a, b, width, |x, y| resize(&x.umul(y), width, false)) {
        return result;
    }
    endpoints(a, b, width, false, |x, y| {
        let product = x.umul(y);
        let high = product.width_slice(
            width as i64,
            product.get_bit_count().saturating_sub(width) as i64,
        );
        Endpoint {
            value: resize(&product, width, false),
            overflow: !high.is_zero(),
            twice: !high.is_zero() && !high.equals_u64_value(1),
        }
    })
}

fn unsigned_divide(a: &IntervalSet, b: &IntervalSet) -> IntervalSet {
    let w = a.width();
    if let Some(result) = exact_binary(a, b, w, IrBits::udiv) {
        return result;
    }
    let nonzero = if w == 0 {
        IntervalSet::empty(w)
    } else {
        b.intersect(&make(w, [(index_bits(w, 1), IrBits::all_ones(w))]))
    };
    // Excluding division by zero can bring the Cartesian domain below the
    // exact-enumeration budget, even when the original divisor set did not fit.
    let mut result = exact_binary(a, &nonzero, w, IrBits::udiv)
        .unwrap_or_else(|| endpoints(a, &nonzero, w, true, |x, y| Endpoint::plain(x.udiv(y))));
    if b.contains(&IrBits::zero(w)) && !is_empty(a) {
        result = result.union(&IntervalSet::singleton(IrBits::all_ones(w)));
    }
    result
}

/// Computes constant unsigned remainders exactly for each input interval.
fn constant_remainder(a: &IntervalSet, divisor: &IrBits) -> IntervalSet {
    let w = a.width();
    if divisor.is_zero() {
        return zero(w);
    }
    let last = divisor.sub(&index_bits(w, 1));
    let mut result = Vec::new();
    for i in a.intervals() {
        if i.upper().sub(i.lower()).uge(&last) {
            return make(w, [(IrBits::zero(w), last)]);
        }
        let lo = i.lower().umod(divisor);
        let hi = i.upper().umod(divisor);
        if lo.ule(&hi) {
            result.push((lo, hi));
        } else {
            result.push((lo, last.clone()));
            result.push((IrBits::zero(w), hi));
        }
    }
    make(w, result)
}

fn unsigned_remainder(a: &IntervalSet, b: &IntervalSet) -> IntervalSet {
    if let Some(divisor) = b.singleton_value() {
        return constant_remainder(a, divisor);
    }
    if let Some(result) = exact_binary(a, b, a.width(), IrBits::umod) {
        return result;
    }
    let upper = b.unsigned_max().unwrap().sub(&index_bits(b.width(), 1));
    let upper = if a.unsigned_max().unwrap().ult(&upper) {
        a.unsigned_max().unwrap().clone()
    } else {
        upper
    };
    make(a.width(), [(IrBits::zero(a.width()), upper)])
}

fn signed_arithmetic(op: Binop, a: &IntervalSet, b: &IntervalSet, width: usize) -> IntervalSet {
    if let Some(result) = exact_binary(a, b, width, |x, y| match op {
        Binop::Smul => resize(&x.smul(y), width, true),
        Binop::Sdiv => x.sdiv(y),
        Binop::Smod => x.smod(y),
        _ => unreachable!("signed arithmetic operation"),
    }) {
        return result;
    }
    let (ap, an) = signed_parts(a);
    let (bp, bn) = signed_parts(b);
    if op == Binop::Smod {
        let magnitude = bp.union(&bn);
        let mut result = IntervalSet::empty(width);
        if !is_empty(&magnitude) {
            if !is_empty(&ap) {
                result = result.union(&unsigned_remainder(&ap, &magnitude));
            }
            if !is_empty(&an) {
                result = result.union(&negate(&unsigned_remainder(&an, &magnitude)));
            }
        }
        if a.contains(&IrBits::zero(a.width())) || b.contains(&IrBits::zero(b.width())) {
            result = result.union(&zero(width));
        }
        return result;
    }
    let apply = |x: &IntervalSet, y: &IntervalSet| {
        if op == Binop::Smul {
            unsigned_multiply(x, y, width)
        } else {
            unsigned_divide(x, y)
        }
    };
    let positive = apply(&ap, &bp).union(&apply(&an, &bn));
    let negative = negate(&apply(&ap, &bn).union(&apply(&an, &bp)));
    let mut result = positive.union(&negative);
    let a_zero = a.contains(&IrBits::zero(a.width()));
    let b_zero = b.contains(&IrBits::zero(b.width()));
    if (a_zero && (op == Binop::Smul || !is_empty(&bp) || !is_empty(&bn)))
        || (op == Binop::Smul && b_zero)
    {
        result = result.union(&zero(width));
    }
    if op == Binop::Sdiv && b_zero {
        if !is_empty(&ap) || a_zero {
            result = result.union(&IntervalSet::singleton(IrBits::signed_max_value(width)));
        }
        if !is_empty(&an) {
            result = result.union(&IntervalSet::singleton(IrBits::signed_min_value(width)));
        }
    }
    result
}

fn shift_value(op: Binop, value: &IrBits, amount: &IrBits) -> IrBits {
    let shift = saturating_index(amount, value.get_bit_count()) as i64;
    match op {
        Binop::Shll => value.shll(shift),
        Binop::Shrl => value.shrl(shift),
        Binop::Shra => value.shra(shift),
        _ => unreachable!("shift operation"),
    }
}

fn shift(op: Binop, a: &IntervalSet, b: &IntervalSet) -> IntervalSet {
    let w = a.width();
    if let Some(result) = exact_binary(a, b, w, |x, y| shift_value(op, x, y)) {
        return result;
    }
    if b.unsigned_min()
        .is_some_and(|x| saturating_index(x, w) == w)
    {
        if op != Binop::Shra {
            return zero(w);
        }
        let mut result = IntervalSet::empty(w);
        if !a.unsigned_min().unwrap().is_negative() {
            result = result.union(&zero(w));
        }
        if a.unsigned_max().unwrap().is_negative() {
            result = result.union(&IntervalSet::singleton(IrBits::all_ones(w)));
        }
        return result;
    }
    match op {
        Binop::Shrl => endpoints(a, b, w, true, |x, y| Endpoint::plain(shift_value(op, x, y))),
        Binop::Shra => {
            if w == 0 {
                return zero(w);
            }
            let args = reduce_fragmentation(&[a, b]);
            let a = &args[0];
            let b = &args[1];
            let positive = a.intersect(&make(w, [(IrBits::zero(w), IrBits::signed_max_value(w))]));
            let negative = a.intersect(&make(
                w,
                [(IrBits::signed_min_value(w), IrBits::all_ones(w))],
            ));
            // XLS splits every operand at its sign boundary when any operand
            // has sign-sensitive behavior, even the unsigned shift amount.
            // Retain those extra partitions: merging them before evaluation
            // can lose holes between ranges for adjacent shift amounts.
            let amounts = if b.width() == 0 {
                vec![b.clone()]
            } else {
                vec![
                    b.intersect(&make(
                        b.width(),
                        [(IrBits::zero(b.width()), IrBits::signed_max_value(b.width()))],
                    )),
                    b.intersect(&make(
                        b.width(),
                        [(
                            IrBits::signed_min_value(b.width()),
                            IrBits::all_ones(b.width()),
                        )],
                    )),
                ]
            };
            let mut result = IntervalSet::empty(w);
            for amount in &amounts {
                result = result.union(&endpoints_prepared(&positive, amount, w, true, |x, y| {
                    Endpoint::plain(shift_value(op, x, y))
                }));
                result = result.union(&endpoints_prepared(&negative, amount, w, false, |x, y| {
                    Endpoint::plain(shift_value(op, x, y))
                }));
            }
            result.into_minimized(RESULT_INTERVALS)
        }
        Binop::Shll => endpoints(a, b, w, false, |x, y| {
            let amount = saturating_index(y, w.saturating_add(1));
            let (overflow, twice) = if amount >= w {
                (
                    !x.is_zero(),
                    !x.is_zero() && (amount > w || !x.equals_u64_value(1)),
                )
            } else {
                let out = x.width_slice((w - amount) as i64, amount as i64);
                (!out.is_zero(), !out.is_zero() && !out.equals_u64_value(1))
            };
            Endpoint {
                value: shift_value(op, x, y),
                overflow,
                twice,
            }
        }),
        _ => unreachable!("shift operation"),
    }
}

/// Computes exact modular images of interval sums and differences.
fn add_sub(op: Binop, a: &IntervalSet, b: &IntervalSet) -> IntervalSet {
    let width = a.width();
    if a.is_full() || b.is_full() {
        return IntervalSet::full(width);
    }
    let args = reduce_fragmentation(&[a, b]);
    let mut result = Vec::new();
    let last = resize(&IrBits::all_ones(width), width + 1, false);
    for a in args[0].intervals() {
        let span_a = resize(&a.upper().sub(a.lower()), width + 1, false);
        for b in args[1].intervals() {
            let span_b = resize(&b.upper().sub(b.lower()), width + 1, false);
            // The sum/difference of two dense integer intervals is itself
            // dense. A span covering the bitvector modulus reaches all values;
            // otherwise the reduced endpoints precisely describe its wrap.
            if span_a.add(&span_b).uge(&last) {
                return IntervalSet::full(width);
            }
            result.push(if op == Binop::Add {
                (a.lower().add(b.lower()), a.upper().add(b.upper()))
            } else {
                (a.lower().sub(b.upper()), a.upper().sub(b.lower()))
            });
        }
    }
    make(width, result).into_minimized(RESULT_INTERVALS)
}

/// Evaluates arithmetic, comparisons, shifts, and scalar gating.
pub(crate) fn binop(op: Binop, a: &IntervalSet, b: &IntervalSet, width: usize) -> IntervalSet {
    if is_empty(a) || is_empty(b) {
        return IntervalSet::empty(width);
    }
    match op {
        Binop::Add | Binop::Sub => add_sub(op, a, b),
        Binop::Umul => unsigned_multiply(a, b, width),
        Binop::Udiv => unsigned_divide(a, b),
        Binop::Umod => unsigned_remainder(a, b),
        Binop::Smul | Binop::Sdiv | Binop::Smod => signed_arithmetic(op, a, b, width),
        Binop::Shll | Binop::Shrl | Binop::Shra => shift(op, a, b),
        Binop::Eq | Binop::Ne => {
            let result = if let (Some(x), Some(y)) = (a.singleton_value(), b.singleton_value()) {
                Some(x == y)
            } else if is_empty(&a.intersect(b)) {
                Some(false)
            } else {
                None
            };
            result.map_or_else(
                || IntervalSet::full(1),
                |x| bool_result(x ^ (op == Binop::Ne)),
            )
        }
        Binop::Ult
        | Binop::Ule
        | Binop::Ugt
        | Binop::Uge
        | Binop::Slt
        | Binop::Sle
        | Binop::Sgt
        | Binop::Sge => {
            let signed = matches!(op, Binop::Slt | Binop::Sle | Binop::Sgt | Binop::Sge);
            let (a, b) = if matches!(op, Binop::Ugt | Binop::Uge | Binop::Sgt | Binop::Sge) {
                (b, a)
            } else {
                (a, b)
            };
            let inclusive = matches!(op, Binop::Ule | Binop::Uge | Binop::Sle | Binop::Sge);
            let extrema = |x: &IntervalSet| {
                if signed {
                    (x.signed_min().unwrap(), x.signed_max().unwrap())
                } else {
                    (
                        x.unsigned_min().unwrap().clone(),
                        x.unsigned_max().unwrap().clone(),
                    )
                }
            };
            let (alo, ahi) = extrema(a);
            let (blo, bhi) = extrema(b);
            let cmp = |x: &IrBits, y: &IrBits| match (signed, inclusive) {
                (true, true) => x.sle(y),
                (true, false) => x.slt(y),
                (false, true) => x.ule(y),
                (false, false) => x.ult(y),
            };
            if cmp(&ahi, &blo) {
                bool_result(true)
            } else if !cmp(&alo, &bhi) {
                bool_result(false)
            } else {
                IntervalSet::full(1)
            }
        }
        Binop::Gate => {
            if a.singleton_value().is_some_and(IrBits::is_zero) {
                zero(width)
            } else if a.contains(&IrBits::zero(a.width())) {
                b.union(&zero(width))
            } else {
                b.clone()
            }
        }
        Binop::Smulp | Binop::Umulp => {
            // Partial-product components are deliberately opaque, as in XLS.
            IntervalSet::full(width)
        }
    }
}

/// Evaluates unary operators, preserving exact interval negation and inversion.
pub(crate) fn unop(op: Unop, a: &IntervalSet, width: usize) -> IntervalSet {
    if is_empty(a) {
        return IntervalSet::empty(width);
    }
    match op {
        Unop::Identity => a.clone(),
        Unop::Neg => negate(a),
        Unop::Not => make(
            width,
            a.intervals()
                .iter()
                .map(|i| (i.upper().not(), i.lower().not())),
        ),
        Unop::OrReduce => {
            if !a.contains(&IrBits::zero(a.width())) {
                bool_result(true)
            } else if a.singleton_value().is_some() {
                bool_result(false)
            } else {
                IntervalSet::full(1)
            }
        }
        Unop::AndReduce => {
            if !a.contains(&IrBits::all_ones(a.width())) {
                bool_result(false)
            } else if a.singleton_value().is_some() {
                bool_result(true)
            } else {
                IntervalSet::full(1)
            }
        }
        Unop::XorReduce => {
            if a.intervals().iter().all(|i| i.lower() == i.upper()) {
                let parity =
                    |v: &IrBits| (v.limbs().iter().map(|x| x.count_ones()).sum::<u32>() & 1) != 0;
                let first = parity(a.unsigned_min().unwrap());
                if a.intervals().iter().all(|i| parity(i.lower()) == first) {
                    return bool_result(first);
                }
            }
            exact_unary(a, 1, |v| {
                IrBits::bool(v.limbs().iter().fold(0u32, |p, x| p ^ x.count_ones()) & 1 != 0)
            })
            .unwrap_or_else(|| IntervalSet::from_known_bits(&known::unop(op, &a.to_known_bits())))
        }
        Unop::Reverse => exact_unary(a, width, |v| {
            IrBits::from_lsb_fn(width, |i| v.get_bit(width - 1 - i).unwrap())
        })
        .unwrap_or_else(|| IntervalSet::from_known_bits(&known::unop(op, &a.to_known_bits()))),
    }
}

fn bitwise(op: NaryOp, a: &IntervalSet, b: &IntervalSet) -> IntervalSet {
    let w = a.width();
    if is_empty(a) || is_empty(b) {
        return IntervalSet::empty(w);
    }
    let identity = if op == NaryOp::And {
        IrBits::all_ones(w)
    } else {
        IrBits::zero(w)
    };
    if a.singleton_value() == Some(&identity) {
        return b.clone();
    }
    if b.singleton_value() == Some(&identity) {
        return a.clone();
    }
    let endpoints_only = |v: &IntervalSet| {
        v.cardinality_up_to(2) == Some(2)
            && v.contains(&IrBits::zero(w))
            && v.contains(&IrBits::all_ones(w))
    };
    if matches!(op, NaryOp::And | NaryOp::Or) {
        let extra = if op == NaryOp::And {
            zero(w)
        } else {
            IntervalSet::singleton(IrBits::all_ones(w))
        };
        if endpoints_only(a) {
            return b.union(&extra);
        }
        if endpoints_only(b) {
            return a.union(&extra);
        }
    }
    if op == NaryOp::Xor {
        let ones = IrBits::all_ones(w);
        if a.singleton_value() == Some(&ones) {
            return unop(Unop::Not, b, w);
        }
        if b.singleton_value() == Some(&ones) {
            return unop(Unop::Not, a, w);
        }
    }
    if let Some(result) = exact_binary(a, b, w, |x, y| match op {
        NaryOp::And => x.and(y),
        NaryOp::Or => x.or(y),
        NaryOp::Xor => x.xor(y),
        _ => unreachable!("basic bitwise operation"),
    }) {
        return result;
    }
    IntervalSet::from_known_bits(&known::nary(
        op,
        &[&a.to_known_bits(), &b.to_known_bits()],
        w,
    ))
}

/// Folds bitwise operands in IR order, matching the XLS range visitor.
pub(crate) fn nary(op: NaryOp, args: &[&IntervalSet], width: usize) -> IntervalSet {
    if op == NaryOp::Concat {
        return concat(args);
    }
    let base = match op {
        NaryOp::Nand => NaryOp::And,
        NaryOp::Nor => NaryOp::Or,
        other => other,
    };
    let mut result = if base == NaryOp::And {
        IntervalSet::singleton(IrBits::all_ones(width))
    } else {
        zero(width)
    };
    for arg in args {
        result = bitwise(base, &result, arg);
    }
    if matches!(op, NaryOp::Nand | NaryOp::Nor) {
        unop(Unop::Not, &result, width)
    } else {
        result
    }
}

struct ConcatOperand<'a> {
    range: &'a IntervalSet,
    offset: usize,
}

/// Packs borrowed operands supplied from least to most significant.
fn concat_values_lsb_first<'a>(values: impl Iterator<Item = &'a IrBits>, width: usize) -> IrBits {
    let mut bits = values
        .flat_map(|value| (0..value.get_bit_count()).map(move |bit| value.get_bit(bit).unwrap()));
    IrBits::from_lsb_fn(width, |_| bits.next().unwrap())
}

/// Preserves holes exactly when the complete concrete operand product fits.
fn exact_concat(args: &[&IntervalSet], width: usize) -> Option<IntervalSet> {
    let mut combinations = 1;
    for arg in args {
        if arg.singleton_value().is_some() {
            continue;
        }
        let count = arg.cardinality_up_to(EXACT_VALUES / combinations)?;
        if count == 0 {
            return Some(IntervalSet::empty(width));
        }
        combinations *= count;
    }
    if combinations == 1 {
        return Some(IntervalSet::singleton(concat_values_lsb_first(
            args.iter().rev().map(|arg| arg.singleton_value().unwrap()),
            width,
        )));
    }
    // Bound repeated packing as well as the number of values. Zero-width
    // operands still cost traversal, while constant-only packing stays cheap.
    if width
        .saturating_add(args.len())
        .saturating_mul(combinations)
        > BIT_WORK_BUDGET
    {
        return None;
    }

    // The product check precedes any enumeration. Only nonconstant operands
    // need owned choices, so their count is also bounded by log2(EXACT_VALUES).
    let choices: Vec<_> = args
        .iter()
        .filter(|arg| arg.singleton_value().is_none())
        .map(|arg| {
            arg.values_up_to(EXACT_VALUES)
                .expect("the complete concrete product fits the enumeration budget")
        })
        .collect();
    let mut points = Vec::with_capacity(combinations);
    for combination in 0..combinations {
        let mut remaining = combination;
        let mut variable_choices = choices.iter().rev();
        let values = args.iter().rev().map(|arg| {
            if let Some(value) = arg.singleton_value() {
                value
            } else {
                let choices = variable_choices.next().unwrap();
                let index = remaining % choices.len();
                remaining /= choices.len();
                &choices[index]
            }
        });
        let value = concat_values_lsb_first(values, width);
        points.push((value.clone(), value));
    }
    Some(make(width, points))
}

struct ConcatContext<'a> {
    operands: Vec<ConcatOperand<'a>>,
    base: IrBits,
    minimum: IrBits,
    maximum: IrBits,
}

/// Appends endpoints in unsigned order, coalescing their overlap immediately.
fn append_concat_interval(result: &mut Vec<(IrBits, IrBits)>, lo: IrBits, hi: IrBits) {
    if let Some((_, previous_hi)) = result.last_mut() {
        if lo.ule(previous_hi) || lo.sub(previous_hi).equals_u64_value(1) {
            if hi.ugt(previous_hi) {
                *previous_hi = hi;
            }
            return;
        }
    }
    result.push((lo, hi));
}

impl ConcatContext<'_> {
    /// Combines the already-chosen prefix, current interval, and suffix bound.
    fn endpoint(&self, prefix: &IrBits, value: &IrBits, offset: usize, suffix: &IrBits) -> IrBits {
        IrBits::from_lsb_fn(self.base.get_bit_count(), |bit| {
            if bit < offset {
                suffix.get_bit(bit).unwrap()
            } else if bit - offset < value.get_bit_count() {
                value.get_bit(bit - offset).unwrap()
            } else {
                prefix.get_bit(bit).unwrap()
            }
        })
    }

    /// Skips a suffix once all of its endpoint intervals necessarily overlap.
    fn visit(&self, index: usize, prefix: &IrBits, result: &mut Vec<(IrBits, IrBits)>) {
        let operand = &self.operands[index];
        for interval in operand.range.intervals() {
            if interval.lower() != interval.upper() {
                // A non-singleton prefix spans at least one entire suffix
                // domain. Consequently every suffix-choice interval overlaps
                // every other one; their union is precisely this single hull.
                let lo = self.endpoint(prefix, interval.lower(), operand.offset, &self.minimum);
                let hi = self.endpoint(prefix, interval.upper(), operand.offset, &self.maximum);
                append_concat_interval(result, lo, hi);
            } else {
                let value = self.endpoint(prefix, interval.lower(), operand.offset, &self.base);
                if index + 1 == self.operands.len() {
                    append_concat_interval(result, value.clone(), value);
                } else {
                    // Precise operands were removed. Continuing thus requires
                    // a choice among at least two intervals; the Cartesian
                    // work limit also bounds recursion depth logarithmically.
                    self.visit(index + 1, &value, result);
                }
            }
        }
    }
}

/// Concatenates small concrete products exactly, with bounded endpoint
/// fallback.
pub(crate) fn concat(args: &[&IntervalSet]) -> IntervalSet {
    let width: usize = args.iter().map(|x| x.width()).sum();
    if args.iter().any(|x| is_empty(x)) {
        return IntervalSet::empty(width);
    }
    if args.is_empty() || width == 0 {
        return zero(width);
    }
    exact_concat(args, width).unwrap_or_else(|| concat_endpoint_hulls(args, width))
}

/// Concatenates endpoint hulls with bounded fragmentation and suffix pruning.
fn concat_endpoint_hulls(args: &[&IntervalSet], width: usize) -> IntervalSet {
    let args = reduce_fragmentation(args);
    let pattern = |bound: Option<bool>| {
        let mut bits = args.iter().rev().flat_map(|arg| {
            let value = match bound {
                Some(false) => arg.unsigned_min(),
                Some(true) => arg.unsigned_max(),
                None => arg.singleton_value(),
            };
            (0..arg.width()).map(move |i| value.is_some_and(|value| value.get_bit(i).unwrap()))
        });
        IrBits::from_lsb_fn(width, |_| bits.next().unwrap())
    };
    let mut offset = width;
    let operands: Vec<_> = args
        .iter()
        .filter_map(|range| {
            offset -= range.width();
            range
                .singleton_value()
                .is_none()
                .then_some(ConcatOperand { range, offset })
        })
        .collect();
    let base = pattern(None);
    if operands.is_empty() {
        return IntervalSet::singleton(base);
    }
    let context = ConcatContext {
        operands,
        base,
        minimum: pattern(Some(false)),
        maximum: pattern(Some(true)),
    };
    let mut result = Vec::new();
    context.visit(0, &context.base, &mut result);
    make(width, result).into_minimized(RESULT_INTERVALS)
}

/// Projects low bits, including modulo wrap and full-domain coverage.
pub(crate) fn truncate(a: &IntervalSet, width: usize) -> IntervalSet {
    if width >= a.width() {
        return extend(a, width, false);
    }
    let last = resize(&IrBits::all_ones(width), a.width(), false);
    let mut result = Vec::with_capacity(a.intervals().len() + 1);
    for i in a.intervals() {
        if i.upper().sub(i.lower()).uge(&last) {
            return IntervalSet::full(width);
        }
        result.push((
            resize(i.lower(), width, false),
            resize(i.upper(), width, false),
        ));
    }
    make(width, result)
}

/// Slices with zero padding and exact modulo projection of shifted bounds.
pub(crate) fn bit_slice(a: &IntervalSet, start: usize, width: usize) -> IntervalSet {
    if is_empty(a) {
        return IntervalSet::empty(width);
    }
    if start >= a.width() {
        return zero(width);
    }
    let shifted = make(
        a.width(),
        a.intervals()
            .iter()
            .map(|i| (i.lower().shrl(start as i64), i.upper().shrl(start as i64))),
    );
    truncate(&shifted, width)
}

/// Enumerates small dynamic starts with bounded shift-stage coarsening.
pub(crate) fn dynamic_bit_slice(a: &IntervalSet, start: &IntervalSet, width: usize) -> IntervalSet {
    if is_empty(a) || is_empty(start) {
        return IntervalSet::empty(width);
    }
    if let Some(value) = start.singleton_value() {
        return bit_slice(a, saturating_index(value, a.width()), width);
    }
    if let Some(values) = start.values_up_to(EXACT_VALUES) {
        let mut result = IntervalSet::empty(width);
        for value in values {
            // Preserve XLS's shift-stage coarsening before projection. Using
            // the tighter static slice here can change which gaps survive the
            // final union reduction, producing an incomparable interval set.
            let shifted = shift(Binop::Shrl, a, &IntervalSet::singleton(value));
            result = result.union(&truncate(&shifted, width));
        }
        return result.into_minimized(RESULT_INTERVALS);
    }
    truncate(&shift(Binop::Shrl, a, start), width)
}

/// Applies a fixed-position update with clipping at the end of the base value.
fn fixed_bit_slice_update(a: &IntervalSet, position: usize, update: &IntervalSet) -> IntervalSet {
    if position >= a.width() || update.width() == 0 {
        return a.clone();
    }
    let used = update.width().min(a.width() - position);
    let high = bit_slice(a, position + used, a.width() - position - used);
    let mid = bit_slice(update, 0, used);
    let low = bit_slice(a, 0, position);
    concat(&[&high, &mid, &low])
}

/// Joins a bounded set of fixed starts while preserving packed bit facts.
pub(crate) fn bit_slice_update(
    a: &IntervalSet,
    start: &IntervalSet,
    update: &IntervalSet,
) -> IntervalSet {
    if is_empty(a) || is_empty(start) || is_empty(update) {
        return IntervalSet::empty(a.width());
    }
    if start
        .unsigned_min()
        .is_some_and(|x| saturating_index(x, a.width()) == a.width())
        || update.width() == 0
    {
        return a.clone();
    }
    if let Some(position) = start.singleton_value() {
        return fixed_bit_slice_update(a, saturating_index(position, a.width()), update);
    }
    // XLS leaves variable slice updates unconstrained. Bound this optional
    // refinement independently of the standard interval arithmetic budget.
    if a.width()
        .saturating_mul(a.width().saturating_add(start.width()))
        > BIT_WORK_BUDGET
    {
        return IntervalSet::full(a.width());
    }
    let fallback = IntervalSet::from_known_bits(&known::slice_update(
        &a.to_known_bits(),
        &start.to_known_bits(),
        &update.to_known_bits(),
    ));
    let Some(count) = start.cardinality_up_to(EXACT_VALUES) else {
        return fallback;
    };
    let work = a
        .width()
        .saturating_mul(
            a.width()
                .saturating_add(start.width())
                .saturating_add(update.width()),
        )
        .saturating_mul(count);
    if work > BIT_WORK_BUDGET {
        return fallback;
    }
    let mut result = IntervalSet::empty(a.width());
    for position in start.values_up_to(EXACT_VALUES).unwrap() {
        let next = fixed_bit_slice_update(a, saturating_index(&position, a.width()), update);
        result = result.union(&next).into_minimized(RESULT_INTERVALS);
    }
    // Fixed-position interval decomposition can lose per-bit facts retained
    // by packed known bits. Intersection retains both kinds of information;
    // never coarsen afterward across one of the fallback's known-bit gaps.
    let refined = result.intersect(&fallback);
    if refined.intervals().len() <= RESULT_INTERVALS {
        refined
    } else {
        fallback
    }
}

/// Extends ranges, splitting at the sign boundary for signed extension.
pub(crate) fn extend(a: &IntervalSet, width: usize, signed: bool) -> IntervalSet {
    if width < a.width() {
        return truncate(a, width);
    }
    if width == a.width() {
        return a.clone();
    }
    let mut result = Vec::with_capacity(a.intervals().len() + 1);
    for i in a.intervals() {
        if signed && !i.lower().is_negative() && i.upper().is_negative() {
            result.push((
                resize(i.lower(), width, true),
                resize(&IrBits::signed_max_value(a.width()), width, true),
            ));
            result.push((
                resize(&IrBits::signed_min_value(a.width()), width, true),
                resize(i.upper(), width, true),
            ));
        } else {
            result.push((
                resize(i.lower(), width, signed),
                resize(i.upper(), width, signed),
            ));
        }
    }
    make(width, result)
}

/// Decodes only structurally in-range positions, plus the overshift zero.
pub(crate) fn decode(a: &IntervalSet, width: usize) -> IntervalSet {
    if is_empty(a) {
        return IntervalSet::empty(width);
    }
    let mut result = Vec::new();
    for i in 0..width {
        if a.contains_usize(i) {
            let value = IrBits::from_lsb_fn(width, |bit| bit == i);
            result.push((value.clone(), value));
        }
    }
    if saturating_index(a.unsigned_max().unwrap(), width) == width {
        result.push((IrBits::zero(width), IrBits::zero(width)));
    }
    make(width, result).into_minimized(RESULT_INTERVALS)
}

/// Encodes small value domains exactly, then uses OR-based ternary encoding.
pub(crate) fn encode(a: &IntervalSet, width: usize) -> IntervalSet {
    if is_empty(a) {
        return IntervalSet::empty(width);
    }
    exact_unary(a, width, |v| {
        let index = (0..a.width())
            .filter(|&i| v.get_bit(i).unwrap())
            .fold(0, |acc, i| acc | i);
        index_bits(width, index)
    })
    .unwrap_or_else(|| {
        let bits = known::encode(&a.to_known_bits());
        IntervalSet::from_known_bits(
            &KnownBits::from_mask_value(
                resize(bits.mask(), width, false),
                resize(bits.value(), width, false),
            )
            .unwrap(),
        )
    })
}

/// Tests each first-set position against interval membership of a bit pattern.
pub(crate) fn one_hot(a: &IntervalSet, lsb_prio: bool) -> IntervalSet {
    let width = a.width() + 1;
    if let Some(result) = exact_unary(a, width, |value| {
        let position = if lsb_prio {
            (0..a.width()).find(|&i| value.get_bit(i).unwrap())
        } else {
            (0..a.width()).rev().find(|&i| value.get_bit(i).unwrap())
        }
        .unwrap_or(a.width());
        IrBits::from_lsb_fn(width, |i| i == position)
    }) {
        return result;
    }
    let mut result = Vec::new();
    for i in 0..a.width() {
        let mask = IrBits::from_lsb_fn(a.width(), |bit| if lsb_prio { bit <= i } else { bit >= i });
        let value = IrBits::from_lsb_fn(a.width(), |bit| bit == i);
        if a.intersects_known_bits(&KnownBits::from_mask_value(mask, value).unwrap()) {
            let value = IrBits::from_lsb_fn(width, |bit| bit == i);
            result.push((value.clone(), value));
        }
    }
    if a.contains(&IrBits::zero(a.width())) {
        let value = IrBits::from_lsb_fn(width, |bit| bit == a.width());
        result.push((value.clone(), value));
    }
    make(width, result).into_minimized(RESULT_INTERVALS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(width: usize, lo: usize, hi: usize) -> IntervalSet {
        make(width, [(index_bits(width, lo), index_bits(width, hi))])
    }

    fn concrete_binop(op: Binop, a: &IrBits, b: &IrBits, width: usize) -> IrBits {
        match op {
            Binop::Add => a.add(b),
            Binop::Sub => a.sub(b),
            Binop::Umul => resize(&a.umul(b), width, false),
            Binop::Smul => resize(&a.smul(b), width, true),
            Binop::Udiv => a.udiv(b),
            Binop::Sdiv => a.sdiv(b),
            Binop::Umod => a.umod(b),
            Binop::Smod => a.smod(b),
            Binop::Shll | Binop::Shrl | Binop::Shra => shift_value(op, a, b),
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
            _ => unreachable!("tested binary operation"),
        }
    }

    #[test]
    fn all_small_interval_arithmetic_and_comparisons_are_sound() {
        let operations = [
            Binop::Add,
            Binop::Sub,
            Binop::Umul,
            Binop::Smul,
            Binop::Udiv,
            Binop::Sdiv,
            Binop::Umod,
            Binop::Smod,
            Binop::Shll,
            Binop::Shrl,
            Binop::Shra,
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
            let values: Vec<_> = (0..1usize << width).map(|v| index_bits(width, v)).collect();
            for lo in 0..values.len() {
                for hi in lo..values.len() {
                    let a = interval(width, lo, hi);
                    for blo in 0..values.len() {
                        for bhi in blo..values.len() {
                            let b = interval(width, blo, bhi);
                            for op in operations {
                                let comparison = matches!(
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
                                );
                                let output_width = if comparison { 1 } else { width };
                                let result = binop(op, &a, &b, output_width);
                                for x in &values[lo..=hi] {
                                    for y in &values[blo..=bhi] {
                                        let concrete = concrete_binop(op, x, y, output_width);
                                        assert!(
                                            result.contains(&concrete),
                                            "{op:?}({a:?}, {b:?}) = {result:?} excludes {concrete:?} for {x:?}, {y:?}"
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
    fn wider_products_signed_minima_and_zero_divisors_are_sound() {
        for width in 1..=5 {
            let candidates = [
                IntervalSet::full(width),
                interval(width, 0, 1),
                interval(width, (1 << (width - 1)) - 1, 1 << (width - 1)),
                interval(width, 1 << (width - 1), (1 << width) - 1),
            ];
            for a in &candidates {
                for b in &candidates {
                    for out_width in [0, 1, width, width + 1, width * 2 + 1] {
                        for op in [Binop::Umul, Binop::Smul] {
                            let result = binop(op, a, b, out_width);
                            for x in a.values_up_to(32).unwrap() {
                                for y in b.values_up_to(32).unwrap() {
                                    assert!(
                                        result.contains(&concrete_binop(op, &x, &y, out_width)),
                                        "{op:?} {a:?} {b:?} width={out_width} result={result:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cheap_precision_improvements_preserve_exact_ranges() {
        assert_eq!(unop(Unop::Not, &interval(4, 3, 5), 4), interval(4, 10, 12));
        assert_eq!(
            binop(Binop::Umod, &interval(4, 10, 12), &interval(4, 8, 8), 4),
            interval(4, 2, 4)
        );
        assert_eq!(
            binop(Binop::Umod, &interval(8, 14, 18), &interval(8, 8, 8), 8),
            make(
                8,
                [
                    (index_bits(8, 0), index_bits(8, 2)),
                    (index_bits(8, 6), index_bits(8, 7))
                ]
            )
        );
        assert_eq!(
            dynamic_bit_slice(&interval(8, 0, 15), &IntervalSet::full(64), 8),
            interval(8, 0, 15)
        );
        assert_eq!(
            binop(
                Binop::Shll,
                &IntervalSet::full(257),
                &interval(257, 257, 260),
                257
            ),
            zero(257)
        );
        let shifted = binop(Binop::Shra, &interval(8, 128, 133), &interval(2, 1, 3), 8);
        assert!(!shifted.contains_usize(200));
        assert!(shifted.contains_usize(192));
        assert!(shifted.contains_usize(240));
        let a = make(
            7,
            [
                (index_bits(7, 0), index_bits(7, 0)),
                (index_bits(7, 7), index_bits(7, 7)),
            ],
        );
        let b = make(
            7,
            [
                (index_bits(7, 0), index_bits(7, 0)),
                (index_bits(7, 127), index_bits(7, 127)),
            ],
        );
        assert_eq!(
            binop(Binop::Add, &a, &b, 7),
            make(
                7,
                [
                    (index_bits(7, 0), index_bits(7, 0)),
                    (index_bits(7, 6), index_bits(7, 7)),
                    (index_bits(7, 127), index_bits(7, 127))
                ]
            )
        );
        let a = make(
            17,
            [
                (index_bits(17, 0), index_bits(17, 0)),
                (index_bits(17, 511), index_bits(17, 511)),
            ],
        );
        assert_eq!(
            binop(Binop::Sub, &a, &a, 17),
            make(
                17,
                [
                    (index_bits(17, 0), index_bits(17, 0)),
                    (index_bits(17, 511), index_bits(17, 511)),
                    (index_bits(17, 130561), index_bits(17, 130561))
                ]
            )
        );
    }

    #[test]
    fn modular_add_and_division_partition_refinements_are_precise() {
        let a = make(
            31,
            [
                (index_bits(31, 0), index_bits(31, 15)),
                (index_bits(31, (1 << 31) - 16), IrBits::all_ones(31)),
            ],
        );
        let b = make(
            31,
            [
                (index_bits(31, 100), index_bits(31, 107)),
                (index_bits(31, 1000), index_bits(31, 1007)),
            ],
        );
        assert_eq!(
            binop(Binop::Add, &a, &b, 31),
            make(
                31,
                [
                    (index_bits(31, 84), index_bits(31, 122)),
                    (index_bits(31, 984), index_bits(31, 1022))
                ]
            )
        );
        let numerator = make(4, [2, 7, 13].map(|v| (index_bits(4, v), index_bits(4, v))));
        let divisor = make(
            4,
            [
                (index_bits(4, 0), index_bits(4, 3)),
                (index_bits(4, 14), index_bits(4, 15)),
            ],
        );
        assert_eq!(
            binop(Binop::Udiv, &numerator, &divisor, 4),
            make(
                4,
                [
                    (index_bits(4, 0), index_bits(4, 4)),
                    (index_bits(4, 6), index_bits(4, 7)),
                    (index_bits(4, 13), index_bits(4, 13)),
                    (index_bits(4, 15), index_bits(4, 15))
                ]
            )
        );
        assert_eq!(
            binop(Binop::Udiv, &interval(8, 252, 255), &interval(8, 0, 4), 8),
            make(
                8,
                [
                    (index_bits(8, 63), index_bits(8, 63)),
                    (index_bits(8, 84), index_bits(8, 85)),
                    (index_bits(8, 126), index_bits(8, 127)),
                    (index_bits(8, 252), index_bits(8, 255))
                ]
            )
        );
    }

    #[test]
    fn bitwise_routing_and_encoding_are_sound_for_small_intervals() {
        for width in 0..=4 {
            for lo in 0..1usize << width {
                for hi in lo..1usize << width {
                    let a = interval(width, lo, hi);
                    for op in [
                        Unop::Identity,
                        Unop::Not,
                        Unop::Neg,
                        Unop::Reverse,
                        Unop::AndReduce,
                        Unop::OrReduce,
                        Unop::XorReduce,
                    ] {
                        let out_width =
                            if matches!(op, Unop::AndReduce | Unop::OrReduce | Unop::XorReduce) {
                                1
                            } else {
                                width
                            };
                        let result = unop(op, &a, out_width);
                        for v in a.values_up_to(16).unwrap() {
                            let concrete = known::unop(op, &KnownBits::constant(&v));
                            assert!(result.contains(concrete.value()), "{op:?} {a:?}");
                        }
                    }
                    for start in 0..=width + 1 {
                        for out_width in [0, 1, width, width + 1] {
                            let result = bit_slice(&a, start, out_width);
                            for v in a.values_up_to(16).unwrap() {
                                assert!(
                                    result.contains(&v.width_slice(start as i64, out_width as i64))
                                );
                            }
                        }
                    }
                    for lsb in [true, false] {
                        let result = one_hot(&a, lsb);
                        for v in a.values_up_to(16).unwrap() {
                            let i = if lsb {
                                (0..width).find(|&i| v.get_bit(i).unwrap())
                            } else {
                                (0..width).rev().find(|&i| v.get_bit(i).unwrap())
                            }
                            .unwrap_or(width);
                            assert!(result.contains(&IrBits::from_lsb_fn(width + 1, |j| j == i)));
                        }
                    }
                    for out_width in [0, 1, width, width + 1] {
                        let result = decode(&a, out_width);
                        for v in lo..=hi {
                            assert!(result.contains(&IrBits::from_lsb_fn(out_width, |i| i == v)));
                        }
                    }
                    for signed in [true, false] {
                        let result = extend(&a, width + 3, signed);
                        for v in a.values_up_to(16).unwrap() {
                            assert!(result.contains(&resize(&v, width + 3, signed)));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn nary_concat_and_dynamic_updates_are_sound() {
        for width in 0..=4 {
            let last = (1usize << width) - 1;
            let candidates = [
                IntervalSet::full(width),
                interval(width, 0, 0),
                interval(width, last, last),
                interval(width, 0, last / 2),
                interval(width, last / 2, last),
            ];
            for a in &candidates {
                let out_width = if width <= 1 {
                    0
                } else {
                    usize::BITS as usize - (width - 1).leading_zeros() as usize
                };
                let encoded = encode(a, out_width);
                for v in a.values_up_to(16).unwrap() {
                    let index = (0..width)
                        .filter(|&i| v.get_bit(i).unwrap())
                        .fold(0usize, |x, y| x | y);
                    assert!(encoded.contains(&index_bits(out_width, index)));
                }
                for b in &candidates {
                    for op in [
                        NaryOp::And,
                        NaryOp::Or,
                        NaryOp::Xor,
                        NaryOp::Nand,
                        NaryOp::Nor,
                        NaryOp::Concat,
                    ] {
                        let result = nary(
                            op,
                            &[a, b],
                            if op == NaryOp::Concat {
                                width * 2
                            } else {
                                width
                            },
                        );
                        for x in a.values_up_to(16).unwrap() {
                            for y in b.values_up_to(16).unwrap() {
                                let concrete = match op {
                                    NaryOp::And => x.and(&y),
                                    NaryOp::Or => x.or(&y),
                                    NaryOp::Xor => x.xor(&y),
                                    NaryOp::Nand => x.and(&y).not(),
                                    NaryOp::Nor => x.or(&y).not(),
                                    NaryOp::Concat => resize(&x, width * 2, false)
                                        .shll(width as i64)
                                        .or(&resize(&y, width * 2, false)),
                                };
                                assert!(result.contains(&concrete), "{op:?} {a:?} {b:?}");
                            }
                        }
                    }
                    let starts = interval(8, 0, width + 2);
                    let result = bit_slice_update(a, &starts, b);
                    for x in a.values_up_to(16).unwrap() {
                        for y in b.values_up_to(16).unwrap() {
                            for start in 0..=width + 2 {
                                let concrete = IrBits::from_lsb_fn(width, |i| {
                                    if i >= start && i - start < width {
                                        y.get_bit(i - start).unwrap()
                                    } else {
                                        x.get_bit(i).unwrap()
                                    }
                                });
                                assert!(result.contains(&concrete));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fragmentation_budget_preserves_xls_gap_tie_breaking() {
        let points = make(
            8,
            (0..16).map(|i| {
                let value = index_bits(8, i * 2);
                (value.clone(), value)
            }),
        );
        let args = [&points, &points, &points, &points, &points];
        let reduced = reduce_fragmentation(&args);
        assert_eq!(
            reduced
                .iter()
                .map(|x| x.intervals().len())
                .collect::<Vec<_>>(),
            vec![15, 16, 16, 16, 16]
        );
        assert_eq!(reduced[0].intervals()[0].lower(), &index_bits(8, 0));
        assert_eq!(reduced[0].intervals()[0].upper(), &index_bits(8, 2));
        assert!(
            reduced
                .iter()
                .map(|x| x.intervals().len())
                .product::<usize>()
                <= INTERVAL_COMBINATIONS
        );
        for (original, reduced) in args.iter().zip(reduced) {
            assert!(original.is_subset_of(&reduced));
        }
    }

    #[test]
    fn fragmentation_budget_compares_gap_magnitudes_across_endpoint_widths() {
        for widths in [[257, 8, 129, 64, 193], [8, 257, 64, 129, 193]] {
            for first_step in [2, 4] {
                let points: Vec<_> = widths
                    .iter()
                    .enumerate()
                    .map(|(index, &width)| {
                        let step = if index == 0 { first_step } else { 2 };
                        make(
                            width,
                            (0..16).map(|i| {
                                let value = index_bits(width, i * step);
                                (value.clone(), value)
                            }),
                        )
                    })
                    .collect();
                let args: Vec<_> = points.iter().collect();
                let reduced = reduce_fragmentation(&args);
                // Equal magnitudes choose the first operand, independently of
                // its width. A larger first gap must instead choose operand1.
                let selected = usize::from(first_step != 2);
                for index in 0..points.len() {
                    assert_eq!(
                        reduced[index].intervals().len(),
                        if index == selected { 15 } else { 16 }
                    );
                    assert!(points[index].is_subset_of(&reduced[index]));
                }
                assert_eq!(
                    reduced[selected].intervals()[0].lower(),
                    &index_bits(widths[selected], 0)
                );
                assert_eq!(
                    reduced[selected].intervals()[0].upper(),
                    &index_bits(widths[selected], 2)
                );
            }
        }
    }

    /// Keeps an unpruned endpoint-product implementation as a precision oracle.
    fn reference_three_way_concat(args: [&IntervalSet; 3]) -> IntervalSet {
        let width = args.iter().map(|a| a.width()).sum();
        let combine = |a: &IrBits, b: &IrBits, c: &IrBits| {
            resize(a, width, false)
                .shll((b.get_bit_count() + c.get_bit_count()) as i64)
                .or(&resize(b, width, false).shll(c.get_bit_count() as i64))
                .or(&resize(c, width, false))
        };
        let mut intervals = Vec::new();
        for a in args[0].intervals() {
            for b in args[1].intervals() {
                for c in args[2].intervals() {
                    intervals.push((
                        combine(a.lower(), b.lower(), c.lower()),
                        combine(a.upper(), b.upper(), c.upper()),
                    ));
                }
            }
        }
        make(width, intervals).minimize(RESULT_INTERVALS)
    }

    #[test]
    fn concat_suffix_pruning_preserves_endpoint_product_precision() {
        let values = [
            interval(3, 0, 0),
            interval(3, 2, 5),
            IntervalSet::full(3),
            make(
                3,
                [
                    (index_bits(3, 0), index_bits(3, 1)),
                    (index_bits(3, 4), index_bits(3, 5)),
                ],
            ),
            make(
                3,
                [
                    (index_bits(3, 1), index_bits(3, 1)),
                    (index_bits(3, 3), index_bits(3, 3)),
                    (index_bits(3, 6), index_bits(3, 7)),
                ],
            ),
        ];
        for a in &values {
            for b in &values {
                for c in &values {
                    assert_eq!(
                        concat_endpoint_hulls(&[a, b, c], 9),
                        reference_three_way_concat([a, b, c])
                    );
                }
            }
        }
    }

    #[test]
    fn high_fan_in_concat_prunes_budgeted_suffixes() {
        let prefix = IntervalSet::singleton(index_bits(5, 17));
        let alternatives = make(
            3,
            [
                (IrBits::zero(3), IrBits::zero(3)),
                (IrBits::all_ones(3), IrBits::all_ones(3)),
            ],
        );
        for count in [24, 48, 96] {
            let mut args = vec![&prefix];
            args.extend(std::iter::repeat_n(&alternatives, count));
            let width = 5 + count * 3;
            let lower =
                resize(prefix.singleton_value().unwrap(), width, false).shll((count * 3) as i64);
            let upper = lower.or(&resize(&IrBits::all_ones(count * 3), width, false));
            assert_eq!(concat(&args), make(width, [(lower, upper)]));
        }
    }
}
