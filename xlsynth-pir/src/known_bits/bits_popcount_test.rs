// SPDX-License-Identifier: Apache-2.0

//! Exhaustive concrete checks of mask/count refinements and local transfers.

use super::{
    KnownBits, binop, decode, dynamic_slice, nary, one_hot, resize, slice, slice_update, unop,
};
use crate::IrBits;
use crate::ir::{Binop, NaryOp, Unop};

fn bits(width: usize, value: u64) -> IrBits {
    IrBits::make_ubits(width, value).unwrap()
}

fn count(value: &IrBits) -> usize {
    value
        .to_le_bytes()
        .iter()
        .map(|byte| byte.count_ones() as usize)
        .sum()
}

fn mask_accepts(facts: &KnownBits, value: &IrBits) -> bool {
    value.and(facts.mask()) == *facts.value()
}

/// Checks mask and count claims separately from the domain membership method.
fn assert_contains(facts: &KnownBits, value: &IrBits) {
    assert_eq!(facts.bit_count(), value.get_bit_count());
    assert!(mask_accepts(facts, value), "{facts:?} excludes {value:?}");
    let ones = count(value);
    assert!(
        facts.min_ones() <= ones && ones <= facts.max_ones(),
        "{facts:?} excludes {value:?}"
    );
    assert!(facts.contains(value));
    assert_eq!(facts.min_zeros(), facts.bit_count() - facts.max_ones());
    assert_eq!(facts.max_zeros(), facts.bit_count() - facts.min_ones());
}

/// Enumerates every consistent narrow mask/count domain, removing duplicates
/// introduced by saturation of the mask.
fn domains(width: usize) -> Vec<(KnownBits, Vec<IrBits>)> {
    let mut result = Vec::new();
    for mut pattern in 0..3usize.pow(width as u32) {
        let base = KnownBits::from_lsb_bits(
            &(0..width)
                .map(|_| {
                    let bit = [None, Some(false), Some(true)][pattern % 3];
                    pattern /= 3;
                    bit
                })
                .collect::<Vec<_>>(),
        );
        for min in 0..=width {
            for max in min..=width {
                let Ok(facts) = base.clone().with_popcount_bounds(min, max) else {
                    continue;
                };
                if result.iter().any(|(previous, _)| previous == &facts) {
                    continue;
                }
                let values = (0..1u64 << width)
                    .map(|value| bits(width, value))
                    .filter(|value| {
                        mask_accepts(&base, value) && (min..=max).contains(&count(value))
                    })
                    .collect::<Vec<_>>();
                assert!(!values.is_empty());
                result.push((facts, values));
            }
        }
    }
    result
}

#[test]
fn refinements_intersect_and_reduce_without_admitting_or_losing_values() {
    for width in 0..=4 {
        for (facts, values) in domains(width) {
            assert!(facts.min_ones() <= facts.max_ones());
            assert!(facts.max_ones() <= width);
            for value in 0..1u64 << width {
                let value = bits(width, value);
                assert_eq!(facts.contains(&value), values.contains(&value));
            }
            for min in 0..=width {
                for max in min..=width {
                    let expected = values
                        .iter()
                        .filter(|v| (min..=max).contains(&count(v)))
                        .collect::<Vec<_>>();
                    let refined = facts.clone().with_popcount_bounds(min, max);
                    assert_eq!(refined.is_err(), expected.is_empty());
                    if let Ok(refined) = refined {
                        for value in &values {
                            assert_eq!(refined.contains(value), expected.contains(&value));
                        }
                        assert!(refined.min_ones() >= facts.min_ones());
                        assert!(refined.max_ones() <= facts.max_ones());
                    }
                }
            }
        }
    }
    assert!(KnownBits::unknown(3).with_popcount_bounds(2, 1).is_err());
    assert!(KnownBits::unknown(3).with_popcount_bounds(0, 4).is_err());
    assert!(
        KnownBits::constant(&bits(3, 7))
            .with_popcount_bounds(0, 2)
            .is_err()
    );
    let exact = KnownBits::unknown(4).with_popcount_bounds(2, 2).unwrap();
    assert_eq!(exact.to_ternary_string(), "XXXX");
    assert_eq!(
        unop(Unop::XorReduce, &exact),
        KnownBits::constant(&IrBits::bool(false))
    );
}

#[test]
fn refined_unary_routing_and_decode_transfers_are_sound() {
    for width in 0..=3 {
        for (a, values) in domains(width) {
            for op in [
                Unop::Identity,
                Unop::Not,
                Unop::Neg,
                Unop::Reverse,
                Unop::OrReduce,
                Unop::AndReduce,
                Unop::XorReduce,
            ] {
                let result = unop(op, &a);
                for value in &values {
                    let concrete = match op {
                        Unop::Identity => value.clone(),
                        Unop::Not => value.not(),
                        Unop::Neg => IrBits::zero(width).sub(value),
                        Unop::Reverse => {
                            IrBits::from_lsb_fn(width, |i| value.get_bit(width - i - 1).unwrap())
                        }
                        Unop::OrReduce => IrBits::bool(count(value) > 0),
                        Unop::AndReduce => IrBits::bool(count(value) == width),
                        Unop::XorReduce => IrBits::bool(count(value) % 2 != 0),
                    };
                    assert_contains(&result, &concrete);
                }
            }
            for output_width in 0..=width + 2 {
                for signed in [false, true] {
                    let result = resize(&a, output_width, signed);
                    for value in &values {
                        let concrete = IrBits::from_lsb_fn(output_width, |i| {
                            if i < width {
                                value.get_bit(i).unwrap()
                            } else {
                                signed && width != 0 && value.get_bit(width - 1).unwrap()
                            }
                        });
                        assert_contains(&result, &concrete);
                    }
                }
                for start in (0..=width + 1).chain([usize::MAX]) {
                    let result = slice(&a, start, output_width);
                    for value in &values {
                        let concrete = IrBits::from_lsb_fn(output_width, |i| {
                            start
                                .checked_add(i)
                                .filter(|&i| i < width)
                                .is_some_and(|i| value.get_bit(i).unwrap())
                        });
                        assert_contains(&result, &concrete);
                    }
                }
            }
            for lsb_prio in [false, true] {
                let result = one_hot(&a, lsb_prio);
                assert_eq!((result.min_ones(), result.max_ones()), (1, 1));
                for value in &values {
                    let selected = if lsb_prio {
                        (0..width).find(|&i| value.get_bit(i).unwrap())
                    } else {
                        (0..width).rev().find(|&i| value.get_bit(i).unwrap())
                    }
                    .unwrap_or(width);
                    assert_contains(&result, &IrBits::from_lsb_fn(width + 1, |i| i == selected));
                }
            }
            for output_width in 0..=(1 << width) + 1 {
                let result = decode(&a, output_width);
                for value in &values {
                    let selected = value.to_u64().unwrap() as usize;
                    assert_contains(
                        &result,
                        &IrBits::from_lsb_fn(output_width, |i| i == selected),
                    );
                }
            }
        }
    }
}

#[test]
fn refined_bitwise_concat_join_and_gate_transfers_are_sound() {
    for width in 0..=3 {
        let values = domains(width);
        for (a, av) in &values {
            for (b, bv) in &values {
                let joined = a.join(b);
                assert_eq!(
                    (joined.min_ones(), joined.max_ones()),
                    (
                        a.min_ones().min(b.min_ones()),
                        a.max_ones().max(b.max_ones())
                    )
                );
                for value in av.iter().chain(bv) {
                    assert_contains(&joined, value);
                }
                let concat = nary(NaryOp::Concat, &[a, b], width * 2);
                for op in [
                    NaryOp::And,
                    NaryOp::Nand,
                    NaryOp::Or,
                    NaryOp::Nor,
                    NaryOp::Xor,
                ] {
                    let result = nary(op, &[a, b], width);
                    for x in av {
                        for y in bv {
                            let concrete = match op {
                                NaryOp::And => x.and(y),
                                NaryOp::Nand => x.and(y).not(),
                                NaryOp::Or => x.or(y),
                                NaryOp::Nor => x.or(y).not(),
                                NaryOp::Xor => x.xor(y),
                                NaryOp::Concat => unreachable!("concat tested separately"),
                            };
                            assert_contains(&result, &concrete);
                            assert_contains(
                                &concat,
                                &IrBits::from_lsb_fn(width * 2, |i| {
                                    if i < width {
                                        y.get_bit(i).unwrap()
                                    } else {
                                        x.get_bit(i - width).unwrap()
                                    }
                                }),
                            );
                        }
                    }
                }
            }
            for (condition, conditions) in domains(1) {
                let result = binop(Binop::Gate, &condition, a, width);
                for c in conditions {
                    for value in av {
                        assert_contains(
                            &result,
                            &if c.is_zero() {
                                IrBits::zero(width)
                            } else {
                                value.clone()
                            },
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn refined_shifts_and_updates_are_sound_for_exact_and_uncertain_starts() {
    for width in 0..=2 {
        let data = domains(width);
        for (a, av) in &data {
            for (start, starts) in domains(3) {
                for op in [Binop::Shll, Binop::Shrl, Binop::Shra] {
                    let result = binop(op, a, &start, width);
                    for value in av {
                        for s in &starts {
                            let shift = s.to_u64().unwrap() as i64;
                            let concrete = match op {
                                Binop::Shll => value.shll(shift),
                                Binop::Shrl => value.shrl(shift),
                                Binop::Shra => value.shra(shift),
                                _ => unreachable!("shift operation"),
                            };
                            assert_contains(&result, &concrete);
                        }
                    }
                }
                let sliced = dynamic_slice(a, &start, width + 1);
                for value in av {
                    for s in &starts {
                        let offset = s.to_u64().unwrap() as usize;
                        assert_contains(
                            &sliced,
                            &IrBits::from_lsb_fn(width + 1, |i| {
                                value.get_bit(i + offset).unwrap_or(false)
                            }),
                        );
                    }
                }
                for (update, updates) in &data {
                    let result = slice_update(a, &start, update);
                    for value in av {
                        for s in &starts {
                            let offset = s.to_u64().unwrap() as usize;
                            for update in updates {
                                let concrete = IrBits::from_lsb_fn(width, |i| {
                                    i.checked_sub(offset)
                                        .filter(|&i| i < update.get_bit_count())
                                        .map_or_else(
                                            || value.get_bit(i).unwrap(),
                                            |i| update.get_bit(i).unwrap(),
                                        )
                                });
                                assert_contains(&result, &concrete);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn wide_counts_saturate_and_routing_does_not_preserve_invalid_nonzero_claims() {
    for width in [0, 1, 63, 64, 65, 127, 128, 129, 257, 4097] {
        let unknown = KnownBits::unknown(width);
        assert_eq!((unknown.min_ones(), unknown.max_ones()), (0, width));
        assert_eq!(
            unknown.clone().with_popcount_bounds(0, 0).unwrap(),
            KnownBits::constant(&IrBits::zero(width))
        );
        assert_eq!(
            unknown.clone().with_popcount_bounds(width, width).unwrap(),
            KnownBits::constant(&IrBits::all_ones(width))
        );
        let hot = one_hot(&unknown, true);
        assert_eq!((hot.min_ones(), hot.max_ones()), (1, 1));
        assert_eq!(
            unop(Unop::OrReduce, &hot),
            KnownBits::constant(&IrBits::bool(true))
        );
        assert_eq!(
            unop(Unop::XorReduce, &hot),
            KnownBits::constant(&IrBits::bool(true))
        );
        let extended = resize(&hot, hot.bit_count() + 65, false);
        assert_eq!((extended.min_ones(), extended.max_ones()), (1, 1));
        assert_eq!(slice(&extended, 0, hot.bit_count()), hot);
        assert_eq!(resize(&extended, hot.bit_count(), false), hot);
        assert_eq!(
            nary(
                NaryOp::Xor,
                &[&hot, &KnownBits::constant(&IrBits::zero(hot.bit_count()))],
                hot.bit_count(),
            ),
            hot
        );
        assert_eq!(slice(&hot, 0, 0), KnownBits::constant(&IrBits::zero(0)));
        assert_eq!(
            slice(&hot, usize::MAX, 7),
            KnownBits::constant(&IrBits::zero(7))
        );
        if width != 0 {
            assert_eq!(slice(&hot, 0, width).min_ones(), 0);
            assert_eq!(resize(&hot, width, false).min_ones(), 0);
        }
        let zero_amount = KnownBits::constant(&IrBits::zero(129));
        assert_eq!(binop(Binop::Shll, &hot, &zero_amount, hot.bit_count()), hot);
        let huge_amount = KnownBits::constant(&IrBits::from_lsb_fn(129, |i| i == 128));
        assert_eq!(
            binop(Binop::Shrl, &hot, &huge_amount, hot.bit_count()),
            KnownBits::constant(&IrBits::zero(hot.bit_count()))
        );
        assert_eq!(slice_update(&hot, &huge_amount, &hot), hot);
    }
}
