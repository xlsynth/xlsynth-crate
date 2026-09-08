// SPDX-License-Identifier: Apache-2.0

//! Concrete-domain checks for bounded concatenation and slice-update transfers.

use crate::IrBits;
use crate::known_bits::bits as known;

use super::{IntervalSet, bit_slice_update, concat, concat_endpoint_hulls, index_bits};

fn points(width: usize, values: impl IntoIterator<Item = IrBits>) -> IntervalSet {
    IntervalSet::from_intervals(
        width,
        values.into_iter().map(|value| (value.clone(), value)),
    )
    .unwrap()
}

fn interval(width: usize, lower: usize, upper: usize) -> IntervalSet {
    IntervalSet::from_intervals(
        width,
        [(index_bits(width, lower), index_bits(width, upper))],
    )
    .unwrap()
}

/// Enumerates every subset, including bottom, of a tiny concrete bit domain.
fn tiny_domains(width: usize) -> Vec<IntervalSet> {
    let value_count = 1usize << width;
    (0..1usize << value_count)
        .map(|mask| {
            points(
                width,
                (0..value_count)
                    .filter(|value| mask & (1 << value) != 0)
                    .map(|value| index_bits(width, value)),
            )
        })
        .collect()
}

/// Copies concrete operands in their IR-defined most-significant-first order.
fn concrete_concat(values: &[&IrBits]) -> IrBits {
    let width = values.iter().map(|value| value.get_bit_count()).sum();
    let mut bits = values.iter().rev().flat_map(|value| {
        (0..value.get_bit_count()).map(move |index| value.get_bit(index).unwrap())
    });
    IrBits::from_lsb_fn(width, |_| bits.next().unwrap())
}

/// Enumerates small Cartesian domains independently of interval endpoints.
fn concrete_concat_domain(args: &[&IntervalSet]) -> IntervalSet {
    let mut values = vec![IrBits::zero(0)];
    for arg in args {
        let choices = arg.values_up_to(16).unwrap();
        values = values
            .iter()
            .flat_map(|prefix| {
                choices
                    .iter()
                    .map(move |choice| concrete_concat(&[prefix, choice]))
            })
            .collect();
    }
    points(args.iter().map(|arg| arg.width()).sum(), values)
}

/// Treats unrepresentably large concrete starts as out of bounds.
fn concrete_start(value: &IrBits) -> usize {
    let mut result = 0;
    for bit in 0..value.get_bit_count() {
        if value.get_bit(bit).unwrap() {
            if bit >= usize::BITS as usize {
                return usize::MAX;
            }
            result |= 1usize << bit;
        }
    }
    result
}

/// Applies replacement bit by bit, including clipping and out-of-bounds no-ops.
fn concrete_update(base: &IrBits, start: &IrBits, update: &IrBits) -> IrBits {
    let start = concrete_start(start);
    IrBits::from_lsb_fn(base.get_bit_count(), |bit| {
        if let Some(offset) = bit
            .checked_sub(start)
            .filter(|offset| *offset < update.get_bit_count())
        {
            update.get_bit(offset).unwrap()
        } else {
            base.get_bit(bit).unwrap()
        }
    })
}

/// Computes exact finite update results without using range-transfer helpers.
fn concrete_update_domain(
    base: &IntervalSet,
    starts: &IntervalSet,
    update: &IntervalSet,
) -> IntervalSet {
    let mut values = Vec::new();
    for base in base.values_up_to(16).unwrap() {
        for start in starts.values_up_to(32).unwrap() {
            for update in update.values_up_to(16).unwrap() {
                values.push(concrete_update(&base, &start, &update));
            }
        }
    }
    points(base.width(), values)
}

fn packed_update_fallback(
    base: &IntervalSet,
    starts: &IntervalSet,
    update: &IntervalSet,
) -> IntervalSet {
    IntervalSet::from_known_bits(&known::slice_update(
        &base.to_known_bits(),
        &starts.to_known_bits(),
        &update.to_known_bits(),
    ))
}

#[test]
fn concat_is_exact_for_every_pair_of_tiny_domains() {
    let domains: Vec<_> = (0..=2).flat_map(tiny_domains).collect();
    for a in &domains {
        for b in &domains {
            let args = [a, b];
            let result = concat(&args);
            assert_eq!(result, concrete_concat_domain(&args), "{a:?} {b:?}");
            assert!(result.is_subset_of(&concat_endpoint_hulls(&args, a.width() + b.width())));
        }
    }
}

#[test]
fn concat_preserves_sparse_points_through_the_sixteen_value_boundary() {
    let suffix = IntervalSet::singleton(IrBits::zero(3));
    let bit = IntervalSet::full(1);
    assert_eq!(
        concat(&[&bit, &suffix]),
        points(4, [index_bits(4, 0), index_bits(4, 8)])
    );

    let sixteen = interval(5, 0, 15);
    let exact = concat(&[&sixteen, &suffix]);
    assert_eq!(
        exact,
        points(8, (0..16).map(|value| index_bits(8, value << 3)))
    );
    assert_eq!(exact.intervals().len(), 16);
    assert!(exact.is_subset_of(&concat_endpoint_hulls(&[&sixteen, &suffix], 8)));

    let seventeen = interval(5, 0, 16);
    let fallback = concat(&[&seventeen, &suffix]);
    assert_eq!(fallback, concat_endpoint_hulls(&[&seventeen, &suffix], 8));
    for value in 0..17 {
        assert!(fallback.contains_usize(value << 3));
    }
}

#[test]
fn concat_handles_empty_zero_width_and_wide_operands() {
    let unit = IntervalSet::full(0);
    let empty = IntervalSet::empty(0);
    let bit = IntervalSet::full(1);
    assert_eq!(concat(&[]), unit);
    assert_eq!(concat(&[&unit, &unit]), unit);
    assert_eq!(concat(&[&bit, &empty]), IntervalSet::empty(1));
    assert_eq!(concat(&[&empty, &unit]), empty);

    for width in [64, 65, 129, 257] {
        let suffix = IntervalSet::singleton(IrBits::from_lsb_fn(width, |bit| bit % 3 == 0));
        let args = [&unit, &bit, &suffix, &unit];
        let result = concat(&args);
        assert_eq!(result, concrete_concat_domain(&args));
        assert!(result.is_subset_of(&concat_endpoint_hulls(&args, width + 1)));
    }
}

#[test]
fn concat_handles_many_operands_without_recursive_enumeration() {
    let variable_indices = [0, 1, 2048, 4095];
    let args: Vec<_> = (0..4096)
        .map(|index| {
            if variable_indices.contains(&index) {
                IntervalSet::full(1)
            } else if index % 3 == 0 {
                IntervalSet::full(0)
            } else {
                IntervalSet::singleton(IrBits::bool(index % 2 == 0))
            }
        })
        .collect();
    let refs: Vec<_> = args.iter().collect();
    let width = args.iter().map(IntervalSet::width).sum();
    let expected = points(
        width,
        (0..16).map(|selection| {
            let values: Vec<_> = args
                .iter()
                .enumerate()
                .map(|(index, arg)| {
                    if let Some(variable) = variable_indices.iter().position(|&i| i == index) {
                        IrBits::bool(selection & (1 << variable) != 0)
                    } else {
                        arg.singleton_value().unwrap().clone()
                    }
                })
                .collect();
            concrete_concat(&values.iter().collect::<Vec<_>>())
        }),
    );
    assert_eq!(concat(&refs), expected);

    let full_bit = IntervalSet::full(1);
    let over_budget = vec![&full_bit; 4096];
    assert_eq!(concat(&over_budget), IntervalSet::full(4096));
}

#[test]
fn concat_bounds_repeated_packing_but_keeps_constant_cases() {
    let suffix = IntervalSet::singleton(IrBits::zero(65_536));
    let prefix = IntervalSet::full(4);
    let args = [&prefix, &suffix];
    let width = prefix.width() + suffix.width();
    assert!(super::exact_concat(&args, width).is_none());
    assert_eq!(concat(&args), concat_endpoint_hulls(&args, width));

    let constant = IntervalSet::singleton(IrBits::zero(width));
    assert_eq!(concat(&[&constant]), constant);
}

#[test]
fn small_start_updates_are_sound_for_every_tiny_domain_combination() {
    let domains: Vec<_> = (0..=2).flat_map(tiny_domains).collect();
    let starts = tiny_domains(2);
    for base in &domains {
        for update in &domains {
            for starts in &starts {
                let result = bit_slice_update(base, starts, update);
                let exact = concrete_update_domain(base, starts, update);
                assert!(
                    exact.is_subset_of(&result),
                    "base={base:?} starts={starts:?} update={update:?} result={result:?}"
                );
                if base.singleton_value().is_some() {
                    assert_eq!(result, exact);
                }
                if !base.intervals().is_empty()
                    && !update.intervals().is_empty()
                    && starts.cardinality_up_to(4).is_some_and(|count| count > 1)
                {
                    assert!(result.is_subset_of(&packed_update_fallback(base, starts, update)));
                }
            }
        }
    }
}

#[test]
fn small_start_updates_preserve_packed_facts_when_slice_correlations_are_lost() {
    let base = IntervalSet::full(8);
    let starts = interval(2, 2, 3);
    let update = IntervalSet::singleton(IrBits::zero(3));
    let result = bit_slice_update(&base, &starts, &update);
    let fallback = packed_update_fallback(&base, &starts, &update);
    assert!(result.is_subset_of(&fallback));
    for value in 0..256 {
        for position in 2..=3 {
            assert!(result.contains(&concrete_update(
                &index_bits(8, value),
                &index_bits(2, position),
                &IrBits::zero(3),
            )));
        }
    }
    assert!(!result.contains_usize(1 << 3));
    assert!(!result.contains_usize(1 << 4));

    let correlated = points(5, [index_bits(5, 0), index_bits(5, 17)]);
    let starts = interval(2, 1, 2);
    let update = IntervalSet::singleton(IrBits::bool(true));
    let result = bit_slice_update(&correlated, &starts, &update);
    assert!(concrete_update_domain(&correlated, &starts, &update).is_subset_of(&result));
    assert!(result.is_subset_of(&packed_update_fallback(&correlated, &starts, &update)));
}

#[test]
fn small_start_update_budget_includes_sixteen_but_not_seventeen_values() {
    let base = IntervalSet::singleton(IrBits::zero(32));
    let update = IntervalSet::singleton(IrBits::bool(true));
    let sixteen = interval(5, 0, 15);
    let exact = bit_slice_update(&base, &sixteen, &update);
    assert_eq!(exact, concrete_update_domain(&base, &sixteen, &update));
    assert!(!exact.contains_usize(0));
    assert!(!exact.contains_usize(3));
    assert!(exact.is_subset_of(&packed_update_fallback(&base, &sixteen, &update)));

    let seventeen = interval(5, 0, 16);
    let fallback = bit_slice_update(&base, &seventeen, &update);
    assert_eq!(fallback, packed_update_fallback(&base, &seventeen, &update));
    assert!(concrete_update_domain(&base, &seventeen, &update).is_subset_of(&fallback));
}

#[test]
fn small_start_update_work_limits_preserve_fallbacks_and_cheap_noops() {
    let base = IntervalSet::singleton(IrBits::zero(257));
    let starts = interval(5, 0, 15);
    let update = IntervalSet::singleton(IrBits::bool(true));
    let result = bit_slice_update(&base, &starts, &update);
    assert_eq!(result, packed_update_fallback(&base, &starts, &update));
    assert!(concrete_update_domain(&base, &starts, &update).is_subset_of(&result));

    let wide_base = IntervalSet::singleton(IrBits::zero(4097));
    let variable_start = interval(129, 0, 1);
    assert_eq!(
        bit_slice_update(&wide_base, &variable_start, &update),
        IntervalSet::full(4097)
    );
    assert_eq!(
        bit_slice_update(&wide_base, &variable_start, &IntervalSet::full(0)),
        wide_base
    );
    assert_eq!(
        bit_slice_update(&wide_base, &interval(129, 4097, 4098), &update),
        wide_base
    );
    assert_eq!(
        bit_slice_update(&wide_base, &interval(129, 0, 0), &update),
        IntervalSet::singleton(index_bits(4097, 1))
    );
}

#[test]
fn small_start_updates_handle_wide_indices_clipping_and_noop_cases() {
    let base = IntervalSet::singleton(IrBits::from_lsb_fn(257, |bit| bit % 3 == 0));
    let update = IntervalSet::singleton(IrBits::all_ones(129));
    let starts = points(
        257,
        [
            index_bits(257, 0),
            index_bits(257, 128),
            index_bits(257, 256),
            index_bits(257, 300),
            IrBits::from_lsb_fn(257, |bit| bit == 256),
        ],
    );
    let result = bit_slice_update(&base, &starts, &update);
    assert_eq!(result, concrete_update_domain(&base, &starts, &update));
    assert!(result.is_subset_of(&packed_update_fallback(&base, &starts, &update)));

    let zero_width_start = IntervalSet::full(0);
    assert_eq!(
        bit_slice_update(&base, &zero_width_start, &update),
        concrete_update_domain(&base, &zero_width_start, &update)
    );

    let zero_width_update = IntervalSet::full(0);
    assert_eq!(bit_slice_update(&base, &starts, &zero_width_update), base);
    let out_of_bounds = points(257, [index_bits(257, 257), IrBits::all_ones(257)]);
    assert_eq!(bit_slice_update(&base, &out_of_bounds, &update), base);
    assert_eq!(
        bit_slice_update(&IntervalSet::full(0), &starts, &update),
        IntervalSet::full(0)
    );
    assert_eq!(
        bit_slice_update(&base, &IntervalSet::empty(257), &update),
        IntervalSet::empty(257)
    );
    assert_eq!(
        bit_slice_update(&base, &starts, &IntervalSet::empty(0)),
        IntervalSet::empty(257)
    );
    assert_eq!(
        bit_slice_update(&IntervalSet::empty(0), &starts, &update),
        IntervalSet::empty(0)
    );
}
