// SPDX-License-Identifier: Apache-2.0

//! Helpers for lowering unsigned multiply-by-constant into shift/add/sub terms.
//!
//! This module uses Canonical Signed Digit (CSD) / Non-Adjacent Form (NAF)
//! style decomposition: represent a constant as a sum of signed powers of two,
//! i.e. terms of the form `+/- (1 << k)`, while minimizing non-zero digits.
//! Fewer non-zero terms generally means fewer add/sub inputs in the lowered
//! network compared to naive bit-by-bit expansion.

use xlsynth::IrBits;

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

#[cfg(test)]
mod tests {
    use super::{SignedDigitSign, decompose_umul_const_terms};

    fn eval_terms_mod_width(constant_width: usize, terms: &[super::SignedDigitTerm]) -> u128 {
        assert!(constant_width <= 128);
        let modulus = if constant_width == 128 {
            u128::MAX
        } else {
            (1u128 << constant_width) - 1
        };
        let mut acc: i128 = 0;
        for term in terms {
            let val = 1i128 << term.shift;
            match term.sign {
                SignedDigitSign::Plus => {
                    acc += val;
                }
                SignedDigitSign::Minus => {
                    acc -= val;
                }
            }
        }
        let m = if constant_width == 128 {
            i128::MAX
        } else {
            1i128 << constant_width
        };
        let acc_mod = ((acc % m) + m) % m;
        (acc_mod as u128) & modulus
    }

    fn decompose_u64(width: usize, value: u64) -> Vec<super::SignedDigitTerm> {
        let bits = xlsynth::IrBits::make_ubits(width, value).expect("u64 constant");
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
