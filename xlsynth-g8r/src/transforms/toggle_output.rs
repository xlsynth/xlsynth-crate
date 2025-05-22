// SPDX-License-Identifier: Apache-2.0

use rand::seq::SliceRandom;
use rand::Rng;

use crate::gate::{AigOperand, GateFn};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputBitLoc {
    pub out_idx: usize,
    pub bit_idx: usize,
}

/// Toggles the negation flag on a specific output bit.
///
/// Returns an error if the indices are out of range.
pub fn toggle_output_bit(g: &mut GateFn, loc: OutputBitLoc) -> Result<(), &'static str> {
    if loc.out_idx >= g.outputs.len() {
        return Err("toggle_output_bit: out_idx out of range");
    }
    let bv = &mut g.outputs[loc.out_idx].bit_vector;
    if loc.bit_idx >= bv.get_bit_count() {
        return Err("toggle_output_bit: bit_idx out of range");
    }
    let op = *bv.get_lsb(loc.bit_idx);
    let mut new_op = op;
    new_op.negated = !new_op.negated;
    bv.set_lsb(loc.bit_idx, new_op);
    Ok(())
}

/// Picks a random output bit and toggles its negation flag.
///
/// Returns the location of the toggled bit on success.
pub fn toggle_output_bit_rand<R: Rng + ?Sized>(
    g: &mut GateFn,
    rng: &mut R,
) -> Result<OutputBitLoc, &'static str> {
    if g.outputs.is_empty() {
        return Err("toggle_output_bit_rand: no outputs");
    }
    let mut candidates = Vec::new();
    for (out_idx, out) in g.outputs.iter().enumerate() {
        for bit_idx in 0..out.get_bit_count() {
            candidates.push(OutputBitLoc { out_idx, bit_idx });
        }
    }
    if candidates.is_empty() {
        return Err("toggle_output_bit_rand: no bits to toggle");
    }
    let loc = *candidates.choose(rng).unwrap();
    toggle_output_bit(g, loc)?;
    Ok(loc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_toggle_output_bit_self_inverse() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let a = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), a.into());
        let g1 = gb.build();

        let mut g2 = g1.clone();
        let loc = OutputBitLoc {
            out_idx: 0,
            bit_idx: 0,
        };
        toggle_output_bit(&mut g2, loc).unwrap();
        toggle_output_bit(&mut g2, loc).unwrap();
        assert_eq!(g1.to_string(), g2.to_string());
    }

    #[test]
    fn test_toggle_output_bit_rand_round_trip() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let a = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), a.into());
        let mut g = gb.build();
        let pre = g.to_string();
        let mut rng = StdRng::seed_from_u64(123);
        let loc = toggle_output_bit_rand(&mut g, &mut rng).unwrap();
        toggle_output_bit(&mut g, loc).unwrap();
        let post = g.to_string();
        assert_eq!(pre, post);
    }

    #[test]
    fn test_toggle_output_bit_invalid_indices() {
        let mut gb = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        gb.add_output("o".to_string(), i0.into());
        let mut g = gb.build();
        let res = toggle_output_bit(
            &mut g,
            OutputBitLoc {
                out_idx: 1,
                bit_idx: 0,
            },
        );
        assert!(res.is_err());
        let res = toggle_output_bit(
            &mut g,
            OutputBitLoc {
                out_idx: 0,
                bit_idx: 1,
            },
        );
        assert!(res.is_err());
    }
}
