// SPDX-License-Identifier: Apache-2.0

//! 4-input NPN canonicalization.
//!
//! NPN equivalence considers:
//! - input negations (N)
//! - input permutations (P)
//! - output negation (N)
//!
//! We compute a canonical representative for a `TruthTable16` by enumerating
//! all 768 transforms and selecting the smallest transformed `u16` value.

use serde::{Deserialize, Serialize};

use crate::cut_db::tt16::{TruthTable16, decode_assignment, encode_assignment};

/// A 4-element permutation expressed as an array of indices.
///
/// Semantics: `perm[i]` indicates which original variable index (0..=3) the new
/// variable `i` drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Perm4(pub [u8; 4]);

impl Perm4 {
    pub const fn identity() -> Self {
        Self([0, 1, 2, 3])
    }
}

/// An NPN transform over 4 variables.
///
/// Semantics for transforming a function `f` into `f'`:
/// - For a new input assignment `y` (canonical domain), compute an original
///   assignment `x` (original domain) such that for each i in 0..4: `x[perm[i]]
///   = y[i] XOR input_neg(i)`.
/// - Then `f'(y) = f(x) XOR output_neg`.
///
/// This definition is convenient because applying the same transform to a
/// canonical recipe corresponds to substituting canonical inputs with permuted
/// (and optionally negated) original inputs, and optionally negating the
/// output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NpnTransform {
    pub perm: Perm4,
    pub input_neg_mask: u8, // low 4 bits used
    pub output_neg: bool,
}

impl NpnTransform {
    pub const fn identity() -> Self {
        Self {
            perm: Perm4::identity(),
            input_neg_mask: 0,
            output_neg: false,
        }
    }

    #[inline]
    pub const fn input_negated(self, i: usize) -> bool {
        ((self.input_neg_mask >> i) & 1) != 0
    }

    /// Returns the inverse transform `inv` such that for all `tt`:
    ///
    /// `transform_tt16(transform_tt16(tt, self), inv) == tt`.
    pub fn inverse(self) -> Self {
        let p = self.perm.0;
        let mut inv_p = [0u8; 4];
        for i in 0..4usize {
            inv_p[p[i] as usize] = i as u8;
        }

        // If the original transform applies input negation on new index `i`, the
        // inverse must apply that negation on new index `p[i]` (see derivation in
        // module docs).
        let mut inv_neg_mask: u8 = 0;
        for i in 0..4usize {
            let old_index = p[i] as usize;
            if ((self.input_neg_mask >> i) & 1) != 0 {
                inv_neg_mask |= 1u8 << old_index;
            }
        }

        Self {
            perm: Perm4(inv_p),
            input_neg_mask: inv_neg_mask,
            output_neg: self.output_neg,
        }
    }
}

/// Packs an `NpnTransform` into a `u16` suitable for dense table storage.
///
/// Layout:
/// - bits 0..=4: perm index (0..23)
/// - bits 5..=8: input_neg_mask (4 bits)
/// - bit 9: output_neg
pub fn pack_npn_transform(x: NpnTransform) -> u16 {
    let perms = all_perms4();
    let mut perm_idx: Option<u16> = None;
    for (i, p) in perms.iter().enumerate() {
        if p.0 == x.perm.0 {
            perm_idx = Some(i as u16);
            break;
        }
    }
    let perm_idx = perm_idx.expect("perm must be one of the 24 Perm4 values");
    let in_mask = (x.input_neg_mask & 0xF) as u16;
    let out_bit: u16 = if x.output_neg { 1 } else { 0 };
    perm_idx | (in_mask << 5) | (out_bit << 9)
}

/// Unpacks a `u16` produced by `pack_npn_transform` back into an
/// `NpnTransform`.
pub fn unpack_npn_transform(packed: u16) -> NpnTransform {
    let perms = all_perms4();
    let perm_idx = (packed & 0x1F) as usize;
    assert!(
        perm_idx < perms.len(),
        "packed perm index out of range: {perm_idx}"
    );
    let input_neg_mask = ((packed >> 5) & 0xF) as u8;
    let output_neg = ((packed >> 9) & 1) != 0;
    NpnTransform {
        perm: perms[perm_idx],
        input_neg_mask,
        output_neg,
    }
}

/// Applies `xform` to `tt` per the semantics documented on `NpnTransform`.
pub fn transform_tt16(tt: TruthTable16, xform: NpnTransform) -> TruthTable16 {
    let mut out = TruthTable16::const0();
    for y in 0u8..16 {
        let y_bits = decode_assignment(y);

        let mut x_bits = [false; 4];
        for i in 0..4usize {
            let orig_var = xform.perm.0[i] as usize;
            let mut v = y_bits[i];
            if xform.input_negated(i) {
                v = !v;
            }
            x_bits[orig_var] = v;
        }

        let x = encode_assignment(x_bits);
        let mut bit = tt.get_bit(x);
        if xform.output_neg {
            bit = !bit;
        }
        out.set_bit(y, bit);
    }
    out
}

fn all_perms4() -> [Perm4; 24] {
    // Hard-coded deterministic ordering of all 4! permutations.
    // (Lexicographic over the [u8;4] array.)
    [
        Perm4([0, 1, 2, 3]),
        Perm4([0, 1, 3, 2]),
        Perm4([0, 2, 1, 3]),
        Perm4([0, 2, 3, 1]),
        Perm4([0, 3, 1, 2]),
        Perm4([0, 3, 2, 1]),
        Perm4([1, 0, 2, 3]),
        Perm4([1, 0, 3, 2]),
        Perm4([1, 2, 0, 3]),
        Perm4([1, 2, 3, 0]),
        Perm4([1, 3, 0, 2]),
        Perm4([1, 3, 2, 0]),
        Perm4([2, 0, 1, 3]),
        Perm4([2, 0, 3, 1]),
        Perm4([2, 1, 0, 3]),
        Perm4([2, 1, 3, 0]),
        Perm4([2, 3, 0, 1]),
        Perm4([2, 3, 1, 0]),
        Perm4([3, 0, 1, 2]),
        Perm4([3, 0, 2, 1]),
        Perm4([3, 1, 0, 2]),
        Perm4([3, 1, 2, 0]),
        Perm4([3, 2, 0, 1]),
        Perm4([3, 2, 1, 0]),
    ]
}

/// Returns the NPN-canonical representative for `tt`, and the transform used.
///
/// The returned `xform` is the transform that was applied to `tt` to obtain the
/// returned canonical truth table:
///
/// `canon_tt = transform_tt16(tt, xform)`.
pub fn canon_tt16(tt: TruthTable16) -> (TruthTable16, NpnTransform) {
    let perms = all_perms4();

    let mut best_tt = TruthTable16(u16::MAX);
    let mut best_xform = NpnTransform::identity();

    for perm in perms {
        for input_neg_mask in 0u8..16 {
            for &output_neg in &[false, true] {
                let xform = NpnTransform {
                    perm,
                    input_neg_mask,
                    output_neg,
                };
                let cand = transform_tt16(tt, xform);
                // Deterministic tie-break: (cand_tt, output_neg, input_neg_mask, perm-array)
                // Note: cand_tt primary ensures stable canonical representative by numeric
                // order.
                if cand.0 < best_tt.0
                    || (cand.0 == best_tt.0
                        && (output_neg, input_neg_mask, perm.0)
                            < (
                                best_xform.output_neg,
                                best_xform.input_neg_mask,
                                best_xform.perm.0,
                            ))
                {
                    best_tt = cand;
                    best_xform = xform;
                }
            }
        }
    }

    (best_tt, best_xform)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_perms_for_test() -> [Perm4; 24] {
        // Mirror ordering used in implementation.
        super::all_perms4()
    }

    fn all_transforms_for_test() -> Vec<NpnTransform> {
        let mut v = Vec::new();
        for perm in all_perms_for_test() {
            for input_neg_mask in 0u8..16 {
                for &output_neg in &[false, true] {
                    v.push(NpnTransform {
                        perm,
                        input_neg_mask,
                        output_neg,
                    });
                }
            }
        }
        v
    }

    #[test]
    fn test_transform_identity_is_noop() {
        let tt = TruthTable16(0x1234);
        let got = transform_tt16(tt, NpnTransform::identity());
        assert_eq!(got, tt);
    }

    #[test]
    fn test_canon_is_idempotent() {
        let tt = TruthTable16(0x5A3C);
        let (canon1, _x1) = canon_tt16(tt);
        let (canon2, _x2) = canon_tt16(canon1);
        assert_eq!(canon2, canon1);
    }

    #[test]
    fn test_inverse_round_trip() {
        let tt = TruthTable16(0x9E37);
        let xform = NpnTransform {
            perm: Perm4([2, 0, 3, 1]),
            input_neg_mask: 0b1010,
            output_neg: true,
        };
        let inv = xform.inverse();
        let tt2 = transform_tt16(tt, xform);
        let tt3 = transform_tt16(tt2, inv);
        assert_eq!(tt3, tt);
    }

    #[test]
    fn test_canon_is_stable_under_all_transforms_for_sample() {
        let tt = TruthTable16(0x3C5A);
        let (canon, _) = canon_tt16(tt);
        for x in all_transforms_for_test() {
            let tt2 = transform_tt16(tt, x);
            let (canon2, _) = canon_tt16(tt2);
            assert_eq!(canon2, canon);
        }
    }
}
