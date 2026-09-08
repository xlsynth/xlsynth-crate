// SPDX-License-Identifier: Apache-2.0

use rand::{Rng, SeedableRng, rngs::StdRng};
use xlsynth_pir::IrBits;
use xlsynth_pir::known_bits::KnownBits;
use xlsynth_pir::range_analysis::IntervalSet;

fn bits(width: usize, value: usize) -> IrBits {
    IrBits::make_ubits(width, value as u64).unwrap()
}

fn set(width: usize, intervals: &[(usize, usize)]) -> IntervalSet {
    IntervalSet::from_intervals(
        width,
        intervals
            .iter()
            .map(|&(lower, upper)| (bits(width, lower), bits(width, upper))),
    )
    .unwrap()
}

/// Checks canonical form independently of the interval constructor.
fn assert_canonical(set: &IntervalSet) {
    for interval in set.intervals() {
        assert_eq!(interval.lower().get_bit_count(), set.width());
        assert_eq!(interval.upper().get_bit_count(), set.width());
        assert!(interval.lower().ule(interval.upper()));
    }
    for pair in set.intervals().windows(2) {
        assert!(pair[0].upper().ult(pair[1].lower()));
        assert!(!pair[1].lower().sub(pair[0].upper()).equals_u64_value(1));
    }
}

/// Exhausts endpoints and concrete values, not the powerset of the domain.
#[test]
fn all_small_intervals_normalize_and_extract_exact_common_bits() {
    for width in 0..=6 {
        let size = 1usize << width;
        for lower in 0..size {
            for upper in 0..size {
                let range = set(width, &[(lower, upper)]);
                assert_canonical(&range);
                let mut common_ones = size - 1;
                let mut possible_ones = 0;
                let mut count = 0;
                let mut signed_min = None::<IrBits>;
                let mut signed_max = None::<IrBits>;
                for value in 0..size {
                    let expected = if lower <= upper {
                        value >= lower && value <= upper
                    } else {
                        value >= lower || value <= upper
                    };
                    assert_eq!(range.contains_usize(value), expected);
                    if expected {
                        count += 1;
                        common_ones &= value;
                        possible_ones |= value;
                        let value = bits(width, value);
                        if signed_min.as_ref().is_none_or(|min| value.slt(min)) {
                            signed_min = Some(value.clone());
                        }
                        if signed_max.as_ref().is_none_or(|max| value.sgt(max)) {
                            signed_max = Some(value);
                        }
                    }
                }
                assert_eq!(range.signed_min(), signed_min);
                assert_eq!(range.signed_max(), signed_max);
                let expected = KnownBits::from_mask_value(
                    bits(width, (size - 1) ^ (possible_ones ^ common_ones)),
                    bits(width, common_ones),
                )
                .unwrap();
                assert_eq!(range.to_known_bits(), expected);
                assert_eq!(range.cardinality_up_to(size), Some(count));
                assert_eq!(range.cardinality_up_to(count - 1), None);
                assert_eq!(range.values_up_to(size).unwrap().len(), count);
            }
        }
    }
}

#[test]
fn random_unions_have_exact_algebra_and_sound_coarsening() {
    let mut rng = StdRng::seed_from_u64(0x5241_4e47_4553);
    for width in 0..=6 {
        let size = 1usize << width;
        for _ in 0..400 {
            let mut generate = || {
                let count = rng.gen_range(0..=12);
                let intervals: Vec<_> = (0..count)
                    .map(|_| {
                        let lower = rng.gen_range(0..size);
                        let upper = match rng.gen_range(0..4) {
                            0 | 1 => lower,
                            2 => (lower + rng.gen_range(0..=3)) % size,
                            _ => rng.gen_range(0..size),
                        };
                        (lower, upper)
                    })
                    .collect();
                set(width, &intervals)
            };
            let a = generate();
            let b = generate();
            let union = a.union(&b);
            let intersection = a.intersect(&b);
            assert_canonical(&a);
            assert_canonical(&b);
            assert_canonical(&union);
            assert_canonical(&intersection);
            assert_eq!(union, b.union(&a));
            assert_eq!(intersection, b.intersect(&a));
            assert_eq!(a.union(&a), a);
            assert_eq!(a.intersect(&a), a);
            assert!(a.is_subset_of(&IntervalSet::from_known_bits(&a.to_known_bits())));
            let mut subset = true;
            for value in 0..size {
                let in_a = a.contains_usize(value);
                let in_b = b.contains_usize(value);
                assert_eq!(union.contains_usize(value), in_a || in_b);
                assert_eq!(intersection.contains_usize(value), in_a && in_b);
                subset &= !in_a || in_b;
            }
            assert_eq!(a.is_subset_of(&b), subset);
            for limit in 1..=5 {
                let minimized = a.minimize(limit);
                assert_canonical(&minimized);
                assert!(a.is_subset_of(&minimized));
                assert!(minimized.intervals().len() <= limit);
                assert_eq!(minimized.minimize(limit), minimized);
                assert_eq!(a.unsigned_min(), minimized.unsigned_min());
                assert_eq!(a.unsigned_max(), minimized.unsigned_max());
            }
        }
    }
}

#[test]
fn minimize_retains_largest_gaps_and_later_ties() {
    let evenly_spaced = set(5, &[(0, 0), (2, 2), (4, 4), (6, 6)]);
    assert_eq!(evenly_spaced.minimize(2), set(5, &[(0, 4), (6, 6)]));
    assert_eq!(evenly_spaced.minimize(3), set(5, &[(0, 2), (4, 4), (6, 6)]));
    let gaps = set(5, &[(0, 0), (2, 2), (12, 12), (15, 15)]);
    assert_eq!(gaps.minimize(2), set(5, &[(0, 2), (12, 15)]));
}

fn known_pattern(width: usize, mut pattern: usize) -> KnownBits {
    let mut mask = 0;
    let mut value = 0;
    for bit in 0..width {
        match pattern % 3 {
            0 => {
                // Digit zero denotes an unknown bit.
            }
            1 => mask |= 1 << bit,
            2 => {
                mask |= 1 << bit;
                value |= 1 << bit;
            }
            _ => unreachable!("base-three digit"),
        }
        pattern /= 3;
    }
    KnownBits::from_mask_value(bits(width, mask), bits(width, value)).unwrap()
}

#[test]
fn bit_pattern_conversion_is_sound_and_pattern_feasibility_is_exact() {
    for width in 0..=6 {
        let size = 1usize << width;
        for pattern in 0..3usize.pow(width as u32) {
            let known = known_pattern(width, pattern);
            let range = IntervalSet::from_known_bits(&known);
            assert_canonical(&range);
            assert!(range.intervals().len() <= 16);
            for value in 0..size {
                let concrete = bits(width, value);
                if known.contains(&concrete) {
                    assert!(range.contains(&concrete));
                }
            }
            if width <= 4 {
                for lower in 0..size {
                    for upper in 0..size {
                        let candidate = set(width, &[(lower, upper)]);
                        let expected = (0..size).any(|value| {
                            let concrete = bits(width, value);
                            known.contains(&concrete) && candidate.contains(&concrete)
                        });
                        assert_eq!(candidate.intersects_known_bits(&known), expected);
                    }
                }
            }
        }
    }
}

#[test]
fn ternary_relaxation_preserves_xls_extrema_and_contiguous_unknown_policy() {
    // Five high unknown bits exceed the budget. XLS folds the contiguous
    // unknown region into one interval, retaining the original global minimum.
    let odd = KnownBits::from_mask_value(bits(6, 1), bits(6, 1)).unwrap();
    assert_eq!(IntervalSet::from_known_bits(&odd), set(6, &[(1, 63)]));
    // Four high unknowns fit exactly and retain all the holes.
    let odd = KnownBits::from_mask_value(bits(5, 1), bits(5, 1)).unwrap();
    let exact: Vec<_> = (1..32).step_by(2).map(|value| (value, value)).collect();
    assert_eq!(IntervalSet::from_known_bits(&odd), set(5, &exact));
    // Noncontiguous unknowns produce sixteen intervals after relaxation.
    let mask = bits(12, 0b0101_0101_0101);
    let alternating = KnownBits::from_mask_value(mask.clone(), mask.clone()).unwrap();
    let relaxed = IntervalSet::from_known_bits(&alternating);
    assert_eq!(relaxed.intervals().len(), 16);
    assert_eq!(relaxed.unsigned_min(), Some(&mask));
    assert_eq!(relaxed.unsigned_max(), Some(&IrBits::all_ones(12)));
    for value in 0..4096 {
        let concrete = bits(12, value);
        assert!(!alternating.contains(&concrete) || relaxed.contains(&concrete));
    }
}

#[test]
fn known_bit_overlap_respects_population_bounds_without_enumerating_wide_domains() {
    for width in 0..=3 {
        let size = 1usize << width;
        for pattern in 0..3usize.pow(width as u32) {
            let base = known_pattern(width, pattern);
            for min in 0..=width {
                for max in min..=width {
                    let Ok(known) = base.clone().with_popcount_bounds(min, max) else {
                        // A contradictory mask/count intersection represents no
                        // valid KnownBits value and is rejected by
                        // construction.
                        continue;
                    };
                    for lower in 0..size {
                        for upper in 0..size {
                            let candidate = set(width, &[(lower, upper)]);
                            let expected = (0..size).any(|value| {
                                candidate.contains_usize(value)
                                    && known.contains(&bits(width, value))
                            });
                            assert_eq!(candidate.intersects_known_bits(&known), expected);
                        }
                    }
                }
            }
        }
    }
    for width in [65, 129, 257, 4097] {
        let one_hot = KnownBits::unknown(width)
            .with_popcount_bounds(1, 1)
            .unwrap();
        assert!(!IntervalSet::singleton(IrBits::zero(width)).intersects_known_bits(&one_hot));
        assert!(!IntervalSet::singleton(IrBits::all_ones(width)).intersects_known_bits(&one_hot));
        assert!(IntervalSet::full(width).intersects_known_bits(&one_hot));
        let no_powers =
            IntervalSet::from_intervals(width, [(bits(width, 5), bits(width, 7))]).unwrap();
        assert!(!no_powers.intersects_known_bits(&one_hot));
        let with_power =
            IntervalSet::from_intervals(width, [(bits(width, 5), bits(width, 9))]).unwrap();
        assert!(with_power.intersects_known_bits(&one_hot));
    }
}

#[test]
fn zero_width_bottom_and_mismatched_queries_are_distinct() {
    let zero = IrBits::zero(0);
    let empty = IntervalSet::empty(0);
    let full = IntervalSet::full(0);
    assert!(empty.is_empty());
    assert!(!empty.is_full());
    assert!(!empty.contains(&zero));
    assert!(full.is_full());
    assert!(!full.is_empty());
    assert_eq!(full.singleton_value(), Some(&zero));
    assert_eq!(full.cardinality_up_to(1), Some(1));
    assert_eq!(full.cardinality_up_to(0), None);
    assert_eq!(empty.cardinality_up_to(0), Some(0));
    assert_eq!(empty.values_up_to(0), Some(vec![]));
    assert_eq!(full.values_up_to(1), Some(vec![zero.clone()]));
    assert!(full.contains_usize(0));
    assert!(!full.contains_usize(1));
    assert!(full.intersects_known_bits(&KnownBits::unknown(0)));
    assert!(!empty.intersects_known_bits(&KnownBits::unknown(0)));
    assert!(empty.is_subset_of(&full));
    assert!(!full.is_subset_of(&empty));
    assert_eq!(full.signed_min(), Some(zero.clone()));
    assert_eq!(full.signed_max(), Some(zero));
    assert_eq!(empty.signed_min(), None);
    assert_eq!(empty.signed_max(), None);
    assert!(!full.contains(&bits(1, 0)));
    assert!(!full.is_subset_of(&IntervalSet::full(1)));
    assert!(!full.intersects_known_bits(&KnownBits::unknown(1)));
    assert!(IntervalSet::from_intervals(1, [(bits(1, 0), bits(2, 1))]).is_err());
}

#[test]
fn arbitrary_width_bounds_counts_and_patterns_do_not_truncate() {
    for width in [63, 64, 65, 127, 128, 129, 257] {
        let maximum = IrBits::all_ones(width);
        let one = bits(width, 1);
        let lower = maximum.sub(&bits(width, 3));
        let top_four =
            IntervalSet::from_intervals(width, [(lower.clone(), maximum.clone())]).unwrap();
        assert_eq!(top_four.cardinality_up_to(4), Some(4));
        assert_eq!(top_four.cardinality_up_to(3), None);
        assert_eq!(top_four.values_up_to(4).unwrap().last(), Some(&maximum));
        assert_eq!(top_four.signed_min(), Some(lower));
        assert_eq!(top_four.signed_max(), Some(maximum.clone()));
        assert!(!top_four.contains_usize(0));
        let full_cardinality = IntervalSet::full(width).cardinality_up_to(usize::MAX);
        if width < usize::BITS as usize {
            assert_eq!(full_cardinality, Some(1usize << width));
        } else {
            assert_eq!(full_cardinality, None);
        }
        let wrapped = IntervalSet::from_intervals(width, [(maximum.clone(), one.clone())]).unwrap();
        assert_eq!(wrapped.cardinality_up_to(3), Some(3));
        assert_eq!(
            wrapped.values_up_to(3).unwrap(),
            vec![IrBits::zero(width), one, maximum]
        );
        assert!(wrapped.intersects_known_bits(&KnownBits::unknown(width)));
        assert!(!top_four.intersects_known_bits(&KnownBits::constant(&IrBits::zero(width))));
    }
    let wide = IrBits::from_lsb_is_0(&(0..257).map(|bit| bit == 200).collect::<Vec<_>>());
    let interval = IntervalSet::from_intervals(257, [(IrBits::zero(257), wide)]).unwrap();
    assert_eq!(interval.cardinality_up_to(usize::MAX), None);
    assert_eq!(interval.values_up_to(16), None);
    assert!(!IntervalSet::full(3).contains_usize(8));
}

#[test]
fn display_is_canonical_and_deterministic() {
    assert_eq!(
        set(8, &[(8, 9), (1, 2), (2, 3)]).to_string(),
        "bits[8]:{[0x01, 0x03], [0x08, 0x09]}"
    );
    assert_eq!(IntervalSet::empty(8).to_string(), "bits[8]:{}");
    assert_eq!(IntervalSet::full(0).to_string(), "bits[0]:{[0x0, 0x0]}");
}

#[test]
fn endpoint_arithmetic_preserves_wide_normalization_counts_and_gap_ties() {
    for boundary in [64, 128, 192, 256] {
        let width = boundary + 1;
        let before =
            IrBits::from_lsb_is_0(&(0..width).map(|bit| bit < boundary).collect::<Vec<_>>());
        let after =
            IrBits::from_lsb_is_0(&(0..width).map(|bit| bit == boundary).collect::<Vec<_>>());
        let adjacent = IntervalSet::from_intervals(
            width,
            [
                (before.clone(), before.clone()),
                (after.clone(), after.clone()),
            ],
        )
        .unwrap();
        assert_eq!(adjacent.intervals().len(), 1);
        assert_eq!(adjacent.unsigned_min(), Some(&before));
        assert_eq!(adjacent.unsigned_max(), Some(&after));
        assert_eq!(adjacent.cardinality_up_to(1), None);
        assert_eq!(adjacent.cardinality_up_to(2), Some(2));
        let zero = IrBits::zero(width);
        let maximum = IrBits::all_ones(width);
        let separated =
            IntervalSet::from_intervals(width, [(maximum.clone(), maximum), (zero.clone(), zero)])
                .unwrap();
        assert_eq!(separated.intervals().len(), 2);
        assert_eq!(separated.cardinality_up_to(2), Some(2));
    }

    let width = 257;
    let base = IrBits::from_lsb_is_0(&(0..width).map(|bit| bit == 192).collect::<Vec<_>>());
    let point = |offset| base.add(&IrBits::make_ubits(width, offset).unwrap());
    let tied = IntervalSet::from_intervals(
        width,
        [0, 2, 4, 6].map(|offset| {
            let point = point(offset);
            (point.clone(), point)
        }),
    )
    .unwrap();
    assert_eq!(
        tied.minimize(2),
        IntervalSet::from_intervals(width, [(point(0), point(4)), (point(6), point(6))]).unwrap()
    );
    let largest_host_distance = IntervalSet::from_intervals(
        width,
        [(
            IrBits::zero(width),
            IrBits::make_ubits(width, usize::MAX as u64).unwrap(),
        )],
    )
    .unwrap();
    assert_eq!(largest_host_distance.cardinality_up_to(usize::MAX), None);
}
