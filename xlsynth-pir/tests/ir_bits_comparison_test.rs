// SPDX-License-Identifier: Apache-2.0

use num_bigint::{BigInt, BigUint, Sign};
use rand::{RngCore, SeedableRng, rngs::StdRng};
use xlsynth_pir::IrBits;

/// Keeps independently decoded arithmetic references beside the public value.
struct ReferenceValue {
    bits: IrBits,
    unsigned: BigUint,
    signed: BigInt,
}

impl ReferenceValue {
    fn from_bytes(width: usize, bytes: &[u8]) -> Self {
        let bits = IrBits::from_le_bytes(width, bytes).unwrap();
        let unsigned = BigUint::from_bytes_le(bytes);
        let signed = BigInt::from_biguint(Sign::Plus, unsigned.clone());
        let signed = if width != 0 && bytes[(width - 1) / 8] & (1 << ((width - 1) % 8)) != 0 {
            signed - (BigInt::from(1u8) << width)
        } else {
            signed
        };
        Self {
            bits,
            unsigned,
            signed,
        }
    }

    fn from_unsigned(width: usize, value: BigUint) -> Self {
        let mut bytes = value.to_bytes_le();
        bytes.resize(width.div_ceil(8), 0);
        Self::from_bytes(width, &bytes)
    }
}

/// Exercises all eight predicates against independent arbitrary-width integers.
fn check_pair(lhs: &ReferenceValue, rhs: &ReferenceValue) {
    let unsigned = lhs.unsigned.cmp(&rhs.unsigned);
    let signed = lhs.signed.cmp(&rhs.signed);
    assert_eq!(
        [
            lhs.bits.ult(&rhs.bits),
            lhs.bits.ule(&rhs.bits),
            lhs.bits.ugt(&rhs.bits),
            lhs.bits.uge(&rhs.bits),
            lhs.bits.slt(&rhs.bits),
            lhs.bits.sle(&rhs.bits),
            lhs.bits.sgt(&rhs.bits),
            lhs.bits.sge(&rhs.bits),
        ],
        [
            unsigned.is_lt(),
            unsigned.is_le(),
            unsigned.is_gt(),
            unsigned.is_ge(),
            signed.is_lt(),
            signed.is_le(),
            signed.is_gt(),
            signed.is_ge(),
        ],
        "comparison mismatch: lhs={:?}, rhs={:?}",
        lhs.bits,
        rhs.bits,
    );
}

/// Produces canonical random payloads, including partially used final bytes.
fn random_bytes(width: usize, rng: &mut StdRng) -> Vec<u8> {
    let mut bytes = vec![0; width.div_ceil(8)];
    rng.fill_bytes(&mut bytes);
    if width % 8 != 0 {
        *bytes.last_mut().unwrap() &= (1u8 << (width % 8)) - 1;
    }
    bytes
}

#[test]
fn comparison_predicates_exhaustive_through_eight_bits() {
    for width in 0..=8 {
        let values: Vec<_> = (0u64..1 << width)
            .map(|value| ReferenceValue::from_unsigned(width, BigUint::from(value)))
            .collect();
        for lhs in &values {
            for rhs in &values {
                check_pair(lhs, rhs);
            }
        }
    }
}

#[test]
fn comparison_predicates_match_big_integers_at_wide_boundaries() {
    let mut rng = StdRng::seed_from_u64(0xc061_9a7e_83a2_5b19);
    for width in [
        9usize, 15, 16, 17, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 191, 192, 193, 255,
        256, 257, 511, 512, 513, 1023, 1024, 1025, 4095, 4096, 4097,
    ] {
        let one = BigUint::from(1u8);
        let sign_bit = &one << (width - 1);
        let max_unsigned = (&one << width) - &one;
        let mut values = vec![
            BigUint::from(0u8),
            one.clone(),
            &sign_bit - &one,
            sign_bit.clone(),
            &sign_bit + &one,
            &max_unsigned - &one,
            max_unsigned,
        ]
        .into_iter()
        .map(|value| ReferenceValue::from_unsigned(width, value))
        .collect::<Vec<_>>();
        for _ in 0..32 {
            values.push(ReferenceValue::from_bytes(
                width,
                &random_bytes(width, &mut rng),
            ));
        }
        // Includes equality, both operand orders, opposite signs, and same-sign
        // negative values. Leading zero limbs remain present in the bitvectors.
        for lhs in &values {
            for rhs in &values {
                check_pair(lhs, rhs);
            }
        }
    }
}

#[test]
fn comparison_predicates_find_a_difference_in_every_bit_position() {
    let mut rng = StdRng::seed_from_u64(0x7913_38b5_c062_4ade);
    for width in [1usize, 63, 64, 65, 127, 128, 129, 257, 1025, 4097] {
        let bytes = random_bytes(width, &mut rng);
        let original = ReferenceValue::from_bytes(width, &bytes);
        for index in 0..width {
            let mut changed = bytes.clone();
            changed[index / 8] ^= 1u8 << (index % 8);
            let changed = ReferenceValue::from_bytes(width, &changed);
            check_pair(&original, &changed);
            check_pair(&changed, &original);
        }
    }
}

#[test]
fn comparison_predicates_keep_width_mismatch_panics() {
    type Predicate = fn(&IrBits, &IrBits) -> bool;
    let predicates: [(&str, Predicate); 8] = [
        ("ult", IrBits::ult),
        ("ule", IrBits::ule),
        ("ugt", IrBits::ugt),
        ("uge", IrBits::uge),
        ("slt", IrBits::slt),
        ("sle", IrBits::sle),
        ("sgt", IrBits::sgt),
        ("sge", IrBits::sge),
    ];
    for (left_width, right_width) in [(0, 1), (1, 0), (63, 64), (64, 65), (129, 128)] {
        // Equal numeric values must still reject unequal widths; also use
        // opposite signs to ensure signed fast paths do not bypass the check.
        for (lhs, rhs) in [
            (IrBits::zero(left_width), IrBits::zero(right_width)),
            (IrBits::all_ones(left_width), IrBits::zero(right_width)),
            (IrBits::zero(left_width), IrBits::all_ones(right_width)),
        ] {
            for (name, predicate) in predicates {
                let panic = std::panic::catch_unwind(|| predicate(&lhs, &rhs))
                    .expect_err("comparison must panic on mismatched widths");
                let message = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .expect("assertion panic must contain a diagnostic");
                assert!(message.contains("bit width mismatch:"), "{name}: {message}");
            }
        }
    }
}
