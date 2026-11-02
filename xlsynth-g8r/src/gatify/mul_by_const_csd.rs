// SPDX-License-Identifier: Apache-2.0

//! Helpers for lowering unsigned multiply-by-constant into shift/add/sub terms.
//!
//! This module uses Canonical Signed Digit (CSD) / Non-Adjacent Form (NAF)
//! style decomposition: represent a constant as a sum of signed powers of two,
//! i.e. terms of the form `+/- (1 << k)`, while minimizing non-zero digits.
//! Fewer non-zero terms generally means fewer add/sub inputs in the lowered
//! network compared to naive bit-by-bit expansion.

use xlsynth_pir::IrBits;

/// Signed digit polarity used by Canonical Signed Digit (CSD) / Non-Adjacent
/// Form (NAF) decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignedDigitSign {
    Plus,
    Minus,
}

/// One signed term in a shift-add/sub decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignedDigitTerm {
    pub sign: SignedDigitSign,
    pub shift: usize,
}

/// Returns a deterministic Canonical Signed Digit (CSD) / Non-Adjacent Form
/// (NAF) style decomposition for an unsigned constant.
///
/// The returned terms satisfy:
/// - each shift is `< output_bit_count`
/// - signs are in `{+1, -1}`
/// - summing all terms modulo `2^output_bit_count` equals the original constant
pub fn decompose_umul_const_terms(
    constant_bits: &IrBits,
    output_bit_count: usize,
) -> Vec<SignedDigitTerm> {
    if output_bit_count == 0 {
        return Vec::new();
    }
    let bit_count = constant_bits.get_bit_count();
    if bit_count == 0 {
        return Vec::new();
    }

    let mut terms = Vec::new();
    let mut i = 0usize;
    while i < bit_count {
        if !constant_bits
            .get_bit(i)
            .expect("constant bit index in range during decomposition")
        {
            i += 1;
            continue;
        }

        let mut run_end = i;
        while run_end + 1 < bit_count
            && constant_bits
                .get_bit(run_end + 1)
                .expect("constant bit index in range during run scan")
        {
            run_end += 1;
        }

        if run_end == i {
            if i < output_bit_count {
                terms.push(SignedDigitTerm {
                    sign: SignedDigitSign::Plus,
                    shift: i,
                });
            }
        } else {
            if i < output_bit_count {
                terms.push(SignedDigitTerm {
                    sign: SignedDigitSign::Minus,
                    shift: i,
                });
            }
            let plus_shift = run_end.saturating_add(1);
            if plus_shift < output_bit_count {
                terms.push(SignedDigitTerm {
                    sign: SignedDigitSign::Plus,
                    shift: plus_shift,
                });
            }
        }
        i = run_end.saturating_add(1);
    }

    terms
}

/// A minimum-weight signed-digit expansion modulo the width of the residue.
pub fn decompose_modular_const_terms(residue: &IrBits) -> Vec<SignedDigitTerm> {
    let width = residue.get_bit_count();
    let mut carry = false;
    let mut terms = Vec::new();
    for i in 0..width {
        let bit = residue.get_bit(i).expect("index within the residue");
        if bit == carry {
            carry = bit;
            continue;
        }
        // The odd suffix has two signed choices. Leave its next digit zero
        // whenever possible, discarding a carry past the output width.
        carry = i + 1 < width && residue.get_bit(i + 1).expect("index within the residue");
        let sign = if carry {
            SignedDigitSign::Minus
        } else {
            SignedDigitSign::Plus
        };
        terms.push(SignedDigitTerm { sign, shift: i });
    }
    terms
}

#[cfg(test)]
mod tests {
    use super::{
        IrBits, SignedDigitSign, decompose_modular_const_terms, decompose_umul_const_terms,
    };

    fn eval_terms_mod_width(constant_width: usize, terms: &[super::SignedDigitTerm]) -> u128 {
        assert!(constant_width <= 128);
        let mut acc = 0u128;
        for term in terms {
            assert!(term.shift < constant_width);
            let bit = 1u128 << term.shift;
            acc = match term.sign {
                SignedDigitSign::Plus => acc.wrapping_add(bit),
                SignedDigitSign::Minus => acc.wrapping_sub(bit),
            };
        }
        if constant_width == 128 {
            acc
        } else {
            acc & ((1u128 << constant_width) - 1)
        }
    }

    #[test]
    fn modular_digits_use_minimum_weight() {
        for width in 0..=8 {
            let modulus = 1usize << width;
            let mut minimum = vec![width + 1; modulus];
            minimum[0] = 0;
            // Try all three digit values at each position, independently of the
            // recoder.
            for shift in 0..width {
                let bit = 1usize << shift;
                let previous = minimum.clone();
                for (value, weight) in previous.into_iter().enumerate() {
                    for next in [(value + bit) % modulus, (value + modulus - bit) % modulus] {
                        minimum[next] = minimum[next].min(weight + 1);
                    }
                }
            }
            for (value, weight) in minimum.into_iter().enumerate() {
                let literal = IrBits::make_ubits(width, value as u64).unwrap();
                let terms = decompose_modular_const_terms(&literal);
                assert_eq!(eval_terms_mod_width(width, &terms), value as u128);
                assert_eq!(terms.len(), weight, "coefficient {value} at width {width}");
            }
        }
    }

    #[test]
    fn modular_digits_preserve_top_bit_and_wraparound() {
        for width in [64usize, 127, 128] {
            let top = 1u128 << (width - 1);
            for value in [top - 1, top, top | 1, top | (top - 1)] {
                let literal = IrBits::from_lsb_fn(width, |i| value & (1u128 << i) != 0);
                let terms = decompose_modular_const_terms(&literal);
                assert_eq!(eval_terms_mod_width(width, &terms), value);
            }
        }
    }

    fn decompose_u64(width: usize, value: u64) -> Vec<super::SignedDigitTerm> {
        let bits = xlsynth_pir::IrBits::make_ubits(width, value).expect("u64 constant");
        decompose_umul_const_terms(&bits, width)
    }

    #[test]
    fn decomposes_dense_runs_to_signed_terms() {
        let terms_3 = decompose_u64(4, 0b0011);
        assert_eq!(
            terms_3,
            vec![
                super::SignedDigitTerm {
                    sign: SignedDigitSign::Minus,
                    shift: 0
                },
                super::SignedDigitTerm {
                    sign: SignedDigitSign::Plus,
                    shift: 2
                }
            ]
        );

        let terms_7 = decompose_u64(4, 0b0111);
        assert_eq!(
            terms_7,
            vec![
                super::SignedDigitTerm {
                    sign: SignedDigitSign::Minus,
                    shift: 0
                },
                super::SignedDigitTerm {
                    sign: SignedDigitSign::Plus,
                    shift: 3
                }
            ]
        );
    }

    #[test]
    fn handles_all_ones_via_modulo_truncation() {
        let terms = decompose_u64(8, 0xff);
        assert_eq!(
            terms,
            vec![super::SignedDigitTerm {
                sign: SignedDigitSign::Minus,
                shift: 0
            }]
        );
    }

    #[test]
    fn preserves_value_modulo_width_for_representative_constants() {
        let samples: &[(usize, u64)] = &[
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 5),
            (8, 11),
            (8, 0x7f),
            (8, 0xff),
            (16, 0x8001),
            (16, 0xf0f3),
        ];
        for (width, value) in samples {
            let terms = decompose_u64(*width, *value);
            let eval = eval_terms_mod_width(*width, &terms);
            let expected = (*value as u128) & ((1u128 << width) - 1);
            assert_eq!(
                eval, expected,
                "value mismatch for width={} value={:#x} terms={:?}",
                width, value, terms
            );
        }
    }

    #[test]
    fn returns_stable_shift_order() {
        let terms = decompose_u64(16, 0b1110_1111_0011_1000);
        for pair in terms.windows(2) {
            assert!(pair[0].shift <= pair[1].shift);
        }
    }
}
