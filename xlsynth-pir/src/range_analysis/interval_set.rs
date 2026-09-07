// SPDX-License-Identifier: Apache-2.0

//! Canonical arbitrary-width unions of unsigned, inclusive intervals.

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt;

use smallvec::SmallVec;

use crate::known_bits::KnownBits;
use crate::{IrBits, IrFormatPreference, ValueError};

use super::policy::MAX_KNOWN_BITS_SPLITS;

/// One nonwrapping interval; endpoints have the same width and `lower <=
/// upper`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interval {
    lower: IrBits,
    upper: IrBits,
}

impl Interval {
    pub fn lower(&self) -> &IrBits {
        &self.lower
    }

    pub fn upper(&self) -> &IrBits {
        &self.upper
    }
}

/// A width-checked union with sorted, disjoint, nonadjacent intervals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntervalSet {
    width: usize,
    intervals: Vec<Interval>,
}

fn unsigned_cmp(a: &IrBits, b: &IrBits) -> Ordering {
    debug_assert_eq!(a.get_bit_count(), b.get_bit_count());
    a.limbs().iter().rev().cmp(b.limbs().iter().rev())
}

/// Subtracts nonwrapping endpoints in LSB-first order without temporary values.
fn distance_limbs<'a>(lower: &'a IrBits, upper: &'a IrBits) -> impl Iterator<Item = u64> + 'a {
    debug_assert_eq!(lower.get_bit_count(), upper.get_bit_count());
    debug_assert!(unsigned_cmp(lower, upper).is_le());
    let count = lower.limbs().len();
    let mut borrow = false;
    lower
        .limbs()
        .iter()
        .zip(upper.limbs())
        .enumerate()
        .map(move |(index, (&lower, &upper))| {
            let (difference, first_borrow) = upper.overflowing_sub(lower);
            let (difference, second_borrow) = difference.overflowing_sub(u64::from(borrow));
            borrow = first_borrow || second_borrow;
            debug_assert!(index + 1 != count || !borrow);
            difference
        })
}

/// Returns the exact endpoint distance only when it fits the supplied bound.
fn distance_up_to(lower: &IrBits, upper: &IrBits, limit: usize) -> Option<usize> {
    let mut limbs = distance_limbs(lower, upper);
    let low = usize::try_from(limbs.next().unwrap_or(0)).ok()?;
    if low > limit || limbs.any(|limb| limb != 0) {
        return None;
    }
    Some(low)
}

/// Exact unsigned gap magnitude, omitting high zeros regardless of input width.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct GapDistance(SmallVec<[u64; 1]>);

impl GapDistance {
    pub(super) fn between(lower: &IrBits, upper: &IrBits) -> Self {
        let mut limbs = SmallVec::new();
        for (index, limb) in distance_limbs(lower, upper).enumerate() {
            if limb != 0 {
                // Do not allocate for high zero limbs: even wide endpoints can
                // have a small gap that fits entirely in the inline limb.
                limbs.resize(index + 1, 0);
                limbs[index] = limb;
            }
        }
        Self(limbs)
    }
}

impl Ord for GapDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .len()
            .cmp(&other.0.len())
            .then_with(|| self.0.iter().rev().cmp(other.0.iter().rev()))
    }
}

impl PartialOrd for GapDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Appends an interval in lower-bound order, coalescing overlap and adjacency.
fn append_normalized(intervals: &mut Vec<Interval>, interval: Interval) {
    if let Some(previous) = intervals.last_mut() {
        if interval.lower.ule(&previous.upper)
            || distance_up_to(&previous.upper, &interval.lower, 1) == Some(1)
        {
            if interval.upper.ugt(&previous.upper) {
                previous.upper = interval.upper;
            }
            return;
        }
    }
    intervals.push(interval);
}

impl IntervalSet {
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn empty(width: usize) -> Self {
        Self {
            width,
            intervals: Vec::new(),
        }
    }

    /// Returns the full domain, including the singleton domain of `bits[0]`.
    pub fn full(width: usize) -> Self {
        Self {
            width,
            intervals: vec![Interval {
                lower: IrBits::zero(width),
                upper: IrBits::all_ones(width),
            }],
        }
    }

    pub fn singleton(value: IrBits) -> Self {
        Self {
            width: value.get_bit_count(),
            intervals: vec![Interval {
                lower: value.clone(),
                upper: value,
            }],
        }
    }

    /// Checks widths, splits wrapping intervals, then sorts and coalesces them.
    pub fn from_intervals(
        width: usize,
        intervals: impl IntoIterator<Item = (IrBits, IrBits)>,
    ) -> Result<Self, ValueError> {
        let mut proper = Vec::new();
        for (lower, upper) in intervals {
            if lower.get_bit_count() != width || upper.get_bit_count() != width {
                return Err(ValueError(format!(
                    "interval endpoints must have width {width}, got {} and {}",
                    lower.get_bit_count(),
                    upper.get_bit_count()
                )));
            }
            if lower.ugt(&upper) {
                proper.push(Interval {
                    lower: IrBits::zero(width),
                    upper,
                });
                proper.push(Interval {
                    lower,
                    upper: IrBits::all_ones(width),
                });
            } else {
                proper.push(Interval { lower, upper });
            }
        }
        proper.sort_unstable_by(|a, b| unsigned_cmp(&a.lower, &b.lower));
        let mut normalized = Vec::with_capacity(proper.len());
        for interval in proper {
            append_normalized(&mut normalized, interval);
        }
        Ok(Self {
            width,
            intervals: normalized,
        })
    }

    pub fn intervals(&self) -> &[Interval] {
        &self.intervals
    }

    pub fn contains(&self, value: &IrBits) -> bool {
        if value.get_bit_count() != self.width {
            return false;
        }
        let index = self
            .intervals
            .partition_point(|interval| interval.lower.ule(value));
        index != 0 && value.ule(&self.intervals[index - 1].upper)
    }

    /// Tests an index without truncating values that do not fit the domain.
    pub fn contains_usize(&self, value: usize) -> bool {
        if self.width < usize::BITS as usize && value >> self.width != 0 {
            return false;
        }
        self.contains(&IrBits::from_lsb_fn(self.width, |bit| {
            bit < usize::BITS as usize && (value >> bit) & 1 != 0
        }))
    }

    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.intervals.len() == 1
            && self.intervals[0].lower.is_zero()
            && self.intervals[0].upper == IrBits::all_ones(self.width)
    }

    pub fn singleton_value(&self) -> Option<&IrBits> {
        if self.intervals.len() == 1 && self.intervals[0].lower == self.intervals[0].upper {
            Some(&self.intervals[0].lower)
        } else {
            None
        }
    }

    pub fn unsigned_min(&self) -> Option<&IrBits> {
        self.intervals.first().map(Interval::lower)
    }

    pub fn unsigned_max(&self) -> Option<&IrBits> {
        self.intervals.last().map(Interval::upper)
    }

    /// Returns the lowest value in two's-complement order.
    pub fn signed_min(&self) -> Option<IrBits> {
        if self.width != 0 {
            let sign_boundary = IrBits::signed_min_value(self.width);
            if let Some(interval) = self
                .intervals
                .iter()
                .find(|interval| interval.upper.uge(&sign_boundary))
            {
                return Some(if interval.lower.ult(&sign_boundary) {
                    sign_boundary
                } else {
                    interval.lower.clone()
                });
            }
        }
        self.unsigned_min().cloned()
    }

    /// Returns the highest value in two's-complement order.
    pub fn signed_max(&self) -> Option<IrBits> {
        if self.width != 0 {
            let signed_max = IrBits::signed_max_value(self.width);
            if let Some(interval) = self
                .intervals
                .iter()
                .rev()
                .find(|interval| interval.lower.ule(&signed_max))
            {
                return Some(if interval.upper.ugt(&signed_max) {
                    signed_max
                } else {
                    interval.upper.clone()
                });
            }
        }
        self.unsigned_max().cloned()
    }

    /// Computes an exact union; any work-limit coarsening is explicit.
    pub fn union(&self, other: &Self) -> Self {
        assert_eq!(self.width, other.width, "interval-set width mismatch");
        let mut result = Vec::with_capacity(self.intervals.len() + other.intervals.len());
        let mut left = self.intervals.iter().peekable();
        let mut right = other.intervals.iter().peekable();
        while let (Some(a), Some(b)) = (left.peek(), right.peek()) {
            let interval = if a.lower.ule(&b.lower) {
                left.next().unwrap()
            } else {
                right.next().unwrap()
            };
            append_normalized(&mut result, interval.clone());
        }
        for interval in left.chain(right) {
            append_normalized(&mut result, interval.clone());
        }
        Self {
            width: self.width,
            intervals: result,
        }
    }

    /// Computes an exact intersection in linear time in the interval counts.
    pub fn intersect(&self, other: &Self) -> Self {
        assert_eq!(self.width, other.width, "interval-set width mismatch");
        let mut result = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < self.intervals.len() && j < other.intervals.len() {
            let a = &self.intervals[i];
            let b = &other.intervals[j];
            let lower = if a.lower.ugt(&b.lower) {
                &a.lower
            } else {
                &b.lower
            };
            let upper = if a.upper.ult(&b.upper) {
                &a.upper
            } else {
                &b.upper
            };
            if lower.ule(upper) {
                result.push(Interval {
                    lower: lower.clone(),
                    upper: upper.clone(),
                });
            }
            match unsigned_cmp(&a.upper, &b.upper) {
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
            }
        }
        Self {
            width: self.width,
            intervals: result,
        }
    }

    /// Checks exact set containment, including holes rather than only extrema.
    pub fn is_subset_of(&self, other: &Self) -> bool {
        if self.width != other.width {
            return false;
        }
        let mut j = 0;
        for interval in &self.intervals {
            while j < other.intervals.len() && other.intervals[j].upper.ult(&interval.lower) {
                j += 1;
            }
            if j == other.intervals.len()
                || other.intervals[j].lower.ugt(&interval.lower)
                || other.intervals[j].upper.ult(&interval.upper)
            {
                return false;
            }
        }
        true
    }

    /// Keeps the largest `limit - 1` gaps, preserving later gaps on ties.
    pub fn minimize(&self, limit: usize) -> Self {
        assert!(limit > 0, "interval limit must be positive");
        if self.intervals.len() <= limit {
            return self.clone();
        }
        if limit == 1 {
            return Self {
                width: self.width,
                intervals: vec![Interval {
                    lower: self.intervals.first().unwrap().lower.clone(),
                    upper: self.intervals.last().unwrap().upper.clone(),
                }],
            };
        }
        let mut result = Vec::with_capacity(limit);
        let mut start = 0;
        for (_, end) in self.retained_gaps(limit) {
            result.push(Interval {
                lower: self.intervals[start].lower.clone(),
                upper: self.intervals[end].upper.clone(),
            });
            start = end + 1;
        }
        result.push(Interval {
            lower: self.intervals[start].lower.clone(),
            upper: self.intervals.last().unwrap().upper.clone(),
        });
        Self {
            width: self.width,
            intervals: result,
        }
    }

    /// Coarsens an owned set without cloning its surviving endpoints.
    pub(crate) fn into_minimized(self, limit: usize) -> Self {
        assert!(limit > 0, "interval limit must be positive");
        if self.intervals.len() <= limit {
            return self;
        }
        let mut gaps = self.retained_gaps(limit).into_iter().peekable();
        let mut intervals = self.intervals.into_iter().enumerate();
        let (_, mut current) = intervals.next().unwrap();
        let mut result = Vec::with_capacity(limit);
        for (index, interval) in intervals {
            if gaps.peek().is_some_and(|(_, end)| *end == index - 1) {
                result.push(current);
                current = interval;
                gaps.next();
            } else {
                current.upper = interval.upper;
            }
        }
        result.push(current);
        Self {
            width: self.width,
            intervals: result,
        }
    }

    /// Selects retained gap boundaries in input order using the shared tie
    /// rule.
    fn retained_gaps(&self, limit: usize) -> Vec<(GapDistance, usize)> {
        debug_assert!(limit > 0 && limit < self.intervals.len());
        if limit == 1 {
            return Vec::new();
        }
        let mut gaps: Vec<_> = self
            .intervals
            .windows(2)
            .enumerate()
            .map(|(index, pair)| (GapDistance::between(&pair[0].upper, &pair[1].lower), index))
            .collect();
        gaps.select_nth_unstable_by(limit - 1, |(a, ai), (b, bi)| {
            b.cmp(a).then_with(|| bi.cmp(ai))
        });
        gaps.truncate(limit - 1);
        gaps.sort_unstable_by_key(|(_, index)| *index);
        gaps
    }

    /// Returns exact cardinality only when it does not exceed `limit`.
    pub fn cardinality_up_to(&self, limit: usize) -> Option<usize> {
        let mut result = 0;
        for interval in &self.intervals {
            let remaining = limit.checked_sub(result)?.checked_sub(1)?;
            let distance = distance_up_to(&interval.lower, &interval.upper, remaining)?;
            result += distance + 1;
        }
        Some(result)
    }

    /// Enumerates in unsigned order only when the entire set fits the limit.
    pub fn values_up_to(&self, limit: usize) -> Option<Vec<IrBits>> {
        let count = self.cardinality_up_to(limit)?;
        let mut result = Vec::with_capacity(count);
        let one = IrBits::from_lsb_fn(self.width, |bit| bit == 0);
        for interval in &self.intervals {
            let mut value = interval.lower.clone();
            loop {
                result.push(value.clone());
                if value == interval.upper {
                    break;
                }
                value = value.add(&one);
            }
        }
        Some(result)
    }

    /// Extracts the exact bit facts shared by all represented values.
    ///
    /// Empty sets map to unknown because `KnownBits` has no bottom element.
    pub fn to_known_bits(&self) -> KnownBits {
        let mut result = None;
        for interval in &self.intervals {
            let difference = interval.lower.xor(&interval.upper);
            let first_unknown = difference
                .limbs()
                .iter()
                .enumerate()
                .rev()
                .find(|(_, limb)| **limb != 0)
                .map(|(index, limb)| index * 64 + (64 - limb.leading_zeros() as usize));
            let mask = IrBits::from_lsb_fn(self.width, |bit| {
                first_unknown.is_none_or(|count| bit >= count)
            });
            let known = KnownBits::from_mask_value(mask, interval.lower.clone()).unwrap();
            result = Some(match result {
                None => known,
                Some(previous) => KnownBits::join(&previous, &known),
            });
        }
        result.unwrap_or_else(|| KnownBits::unknown(self.width))
    }

    /// Bounds a bit pattern using XLS's four high unknown-interval-bit policy.
    pub fn from_known_bits(known: &KnownBits) -> Self {
        let width = known.bit_count();
        if known.is_fully_known() {
            return Self::singleton(known.value().clone());
        }
        let lower_bound = known.value();
        let upper_bound = known.value().or(&known.mask().not());
        let mut trailing = (0..width)
            .find(|&bit| known.mask().get_bit(bit).unwrap())
            .unwrap_or(width);
        let mut unknown = VecDeque::with_capacity(MAX_KNOWN_BITS_SPLITS + 1);
        for bit in trailing..width {
            if !known.mask().get_bit(bit).unwrap() {
                unknown.push_back(bit);
                if unknown.len() > MAX_KNOWN_BITS_SPLITS + 1 {
                    unknown.pop_front();
                }
            }
        }
        if unknown.len() > MAX_KNOWN_BITS_SPLITS {
            trailing = unknown.pop_front().unwrap() + 1;
            while unknown.front() == Some(&trailing) {
                trailing += 1;
                unknown.pop_front();
            }
        }
        let low_mask = IrBits::from_lsb_fn(width, |bit| bit < trailing);
        let mut intervals = Vec::with_capacity(1 << unknown.len());
        for combination in 0..(1usize << unknown.len()) {
            let lower = IrBits::from_lsb_fn(width, |bit| {
                if bit < trailing {
                    return false;
                }
                if let Some(index) = unknown.iter().position(|&unknown_bit| unknown_bit == bit) {
                    (combination >> index) & 1 != 0
                } else {
                    known.value().get_bit(bit).unwrap()
                }
            });
            let upper = lower.or(&low_mask);
            intervals.push((
                if lower.ult(lower_bound) {
                    lower_bound.clone()
                } else {
                    lower
                },
                if upper.ugt(&upper_bound) {
                    upper_bound.clone()
                } else {
                    upper
                },
            ));
        }
        Self::from_intervals(width, intervals).unwrap()
    }

    /// Tests whether any represented value satisfies the supplied bit pattern.
    pub fn intersects_known_bits(&self, known: &KnownBits) -> bool {
        if self.width != known.bit_count() {
            return false;
        }
        if known.is_fully_known() {
            return self.contains(known.value());
        }
        if known.mask().is_zero() {
            return !self.is_empty();
        }
        self.intervals.iter().any(|interval| {
            // A four-state digit DP tracks whether the prefix equals each
            // endpoint. Strictly interior prefixes can accept either next bit.
            let mut states = [false, false, false, true];
            for bit in (0..self.width).rev() {
                let lower = interval.lower.get_bit(bit).unwrap();
                let upper = interval.upper.get_bit(bit).unwrap();
                let fixed = known.mask().get_bit(bit).unwrap();
                let value = known.value().get_bit(bit).unwrap();
                let mut next = [false; 4];
                for (state, &possible) in states.iter().enumerate() {
                    if !possible {
                        continue;
                    }
                    let equal_lower = state & 1 != 0;
                    let equal_upper = state & 2 != 0;
                    for choice in [false, true] {
                        if (fixed && choice != value)
                            || (equal_lower && lower && !choice)
                            || (equal_upper && !upper && choice)
                        {
                            continue;
                        }
                        let next_state = usize::from(equal_lower && choice == lower)
                            | (usize::from(equal_upper && choice == upper) << 1);
                        next[next_state] = true;
                    }
                }
                states = next;
                if states[0] {
                    // A prefix strictly between both bounds admits every
                    // suffix, including one satisfying the remaining pattern.
                    return true;
                }
                if !states.iter().any(|&state| state) {
                    return false;
                }
            }
            states.iter().any(|&state| state)
        })
    }
}

impl fmt::Display for IntervalSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bits[{}]:{{", self.width)?;
        for (index, interval) in self.intervals.iter().enumerate() {
            if index != 0 {
                write!(f, ", ")?;
            }
            write!(
                f,
                "[0x{}, 0x{}]",
                interval
                    .lower
                    .to_string_fmt(IrFormatPreference::ZeroPaddedHex, false),
                interval
                    .upper
                    .to_string_fmt(IrFormatPreference::ZeroPaddedHex, false)
            )?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use std::panic::catch_unwind;

    use num_bigint::BigUint;
    use rand::{RngCore, SeedableRng, rngs::StdRng};

    use super::{GapDistance, IntervalSet, distance_up_to};
    use crate::IrBits;

    fn as_biguint(value: &IrBits) -> BigUint {
        BigUint::from_bytes_le(&value.to_le_bytes())
    }

    fn random_bits(rng: &mut StdRng, width: usize) -> IrBits {
        let mut bytes = vec![0; width.div_ceil(8)];
        rng.fill_bytes(&mut bytes);
        if width % 8 != 0 {
            *bytes.last_mut().unwrap() &= (1u8 << (width % 8)) - 1;
        }
        IrBits::from_le_bytes(width, &bytes).unwrap()
    }

    /// Checks every narrow set, then fragmented arbitrary-width domains.
    #[test]
    fn consuming_minimization_matches_borrowed_for_narrow_and_wide_sets() {
        for width in 0..=3 {
            let size = 1usize << width;
            for membership in 0..(1usize << size) {
                let set = IntervalSet::from_intervals(
                    width,
                    (0..size)
                        .filter(|&value| (membership >> value) & 1 != 0)
                        .map(|value| {
                            let bits = IrBits::make_ubits(width, value as u64).unwrap();
                            (bits.clone(), bits)
                        }),
                )
                .unwrap();
                for limit in (1..=size + 1).chain([usize::MAX]) {
                    assert_eq!(set.clone().into_minimized(limit), set.minimize(limit));
                }
            }
        }
        let mut rng = StdRng::seed_from_u64(0x4d49_4e49_4d49_5a45);
        for width in [63, 64, 65, 127, 128, 129, 257, 513, 4097] {
            for _ in 0..24 {
                let set = IntervalSet::from_intervals(
                    width,
                    (0..24).map(|_| {
                        let bits = random_bits(&mut rng, width);
                        (bits.clone(), bits)
                    }),
                )
                .unwrap();
                for limit in [1, 2, 3, 4, 15, 16, 17, 24, 25, usize::MAX] {
                    assert_eq!(set.clone().into_minimized(limit), set.minimize(limit));
                }
            }
        }
    }

    /// An owned no-op retains the existing interval allocation, including empty
    /// and zero-width domains.
    #[test]
    fn consuming_minimization_reuses_no_op_storage() {
        for width in [0, 1, 63, 64, 65, 257] {
            for limit in [1, 16, usize::MAX] {
                for set in [IntervalSet::empty(width), IntervalSet::full(width)] {
                    let storage = set.intervals.as_ptr();
                    let capacity = set.intervals.capacity();
                    let result = set.into_minimized(limit);
                    assert_eq!(result.intervals.as_ptr(), storage);
                    assert_eq!(result.intervals.capacity(), capacity);
                }
            }
        }
    }

    /// Later equal gaps survive, and retained wide endpoints move rather than
    /// allocate replacement limb buffers.
    #[test]
    fn consuming_minimization_preserves_ties_and_endpoint_allocations() {
        let set = IntervalSet::from_intervals(
            257,
            [0, 2, 4, 6].map(|value| {
                let bits = IrBits::make_ubits(257, value).unwrap();
                (bits.clone(), bits)
            }),
        )
        .unwrap();
        let retained_buffers = [
            set.intervals[0].lower.limbs().as_ptr(),
            set.intervals[2].upper.limbs().as_ptr(),
            set.intervals[3].lower.limbs().as_ptr(),
            set.intervals[3].upper.limbs().as_ptr(),
        ];
        let result = set.into_minimized(2);
        assert_eq!(result.intervals.len(), 2);
        for (interval, (lower, upper)) in result.intervals.iter().zip([(0, 4), (6, 6)]) {
            assert!(interval.lower.equals_u64_value(lower));
            assert!(interval.upper.equals_u64_value(upper));
        }
        assert_eq!(
            [
                result.intervals[0].lower.limbs().as_ptr(),
                result.intervals[0].upper.limbs().as_ptr(),
                result.intervals[1].lower.limbs().as_ptr(),
                result.intervals[1].upper.limbs().as_ptr(),
            ],
            retained_buffers
        );
    }

    #[test]
    fn both_minimization_forms_reject_zero_limit_for_empty_sets() {
        for width in [0, 257] {
            let set = IntervalSet::empty(width);
            assert!(catch_unwind(|| set.minimize(0)).is_err());
            assert!(catch_unwind(|| set.into_minimized(0)).is_err());
        }
    }

    /// Uses independent unbounded arithmetic for exact and bounded differences.
    #[test]
    fn endpoint_distances_match_biguint_across_widths_and_bounds() {
        let mut rng = StdRng::seed_from_u64(0x4741_505f_4449_5354);
        let mut previous: Option<(GapDistance, BigUint)> = None;
        for width in [
            0, 1, 7, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 513,
        ] {
            for _ in 0..128 {
                let a = random_bits(&mut rng, width);
                let b = random_bits(&mut rng, width);
                let (lower, upper) = if as_biguint(&a) <= as_biguint(&b) {
                    (a, b)
                } else {
                    (b, a)
                };
                let expected = as_biguint(&upper) - as_biguint(&lower);
                let gap = GapDistance::between(&lower, &upper);
                let actual = gap.0.iter().rev().fold(BigUint::from(0u8), |acc, &limb| {
                    (acc << 64usize) + BigUint::from(limb)
                });
                assert_eq!(actual, expected, "width={width}");
                for limit in [0, 1, 2, 15, 16, 17, usize::MAX / 2, usize::MAX] {
                    let bounded = (expected <= BigUint::from(limit)).then(|| {
                        usize::try_from(expected.to_u64_digits().first().copied().unwrap_or(0))
                            .unwrap()
                    });
                    assert_eq!(distance_up_to(&lower, &upper, limit), bounded);
                }
                if let Some((previous_gap, previous_value)) = &previous {
                    assert_eq!(gap.cmp(previous_gap), expected.cmp(previous_value));
                }
                previous = Some((gap, expected));
            }
        }
    }

    #[test]
    fn endpoint_borrows_and_narrow_gaps_do_not_depend_on_endpoint_width() {
        let one = GapDistance::between(&IrBits::zero(1), &IrBits::bool(true));
        for boundary in [64, 128, 192, 256, 512] {
            let width = boundary + 1;
            let lower = IrBits::from_lsb_fn(width, |bit| bit < boundary);
            let upper = IrBits::from_lsb_fn(width, |bit| bit == boundary);
            let gap = GapDistance::between(&lower, &upper);
            assert_eq!(gap, one);
            assert!(!gap.0.spilled());
            assert_eq!(distance_up_to(&lower, &upper, 0), None);
            assert_eq!(distance_up_to(&lower, &upper, 1), Some(1));
            assert_eq!(distance_up_to(&lower, &lower, 0), Some(0));

            let low_one = IrBits::make_ubits(width, 1).unwrap();
            let expected = as_biguint(&upper) - BigUint::from(1u8);
            let borrow_chain = GapDistance::between(&low_one, &upper);
            let actual = borrow_chain
                .0
                .iter()
                .rev()
                .fold(BigUint::from(0u8), |acc, &limb| {
                    (acc << 64usize) + BigUint::from(limb)
                });
            assert_eq!(actual, expected);
            assert_eq!(
                distance_up_to(&low_one, &upper, usize::MAX),
                if boundary == usize::BITS as usize {
                    Some(usize::MAX)
                } else {
                    None
                }
            );
        }
        let zero = IrBits::zero(0);
        assert_eq!(distance_up_to(&zero, &zero, 0), Some(0));
        assert!(GapDistance::between(&zero, &zero) < one);

        let zero = IrBits::zero(65);
        let last_u32 = IrBits::make_ubits(65, u64::from(u32::MAX)).unwrap();
        let past_u32 = IrBits::make_ubits(65, 1u64 << 32).unwrap();
        assert_eq!(
            distance_up_to(&zero, &last_u32, u32::MAX as usize),
            Some(u32::MAX as usize)
        );
        assert_eq!(distance_up_to(&zero, &past_u32, u32::MAX as usize), None);
        assert_eq!(
            distance_up_to(&zero, &past_u32, usize::MAX),
            usize::try_from(1u64 << 32).ok()
        );
    }
}
