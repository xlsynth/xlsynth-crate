// SPDX-License-Identifier: Apache-2.0

use xlsynth::ir_value::IrBits;

use crate::random_inputs::generate_uniform_irbits_with_rng;

/// Generates uniformly distributed bits using the provided RNG.
///
/// Prefer `random_inputs::generate_biased_irbits_with_rng` for semantic
/// checking.
/// Uniform samples remain useful for workload-oriented measurements such as
/// toggle estimation.
pub fn arbitrary_irbits<R: rand::Rng>(rng: &mut R, width: usize) -> IrBits {
    generate_uniform_irbits_with_rng(rng, width)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    #[test]
    fn arbitrary_irbits_handles_u64_boundary_widths() {
        let mut rng = StdRng::seed_from_u64(0);

        for width in [0usize, 1, 63, 64, 65] {
            let bits = arbitrary_irbits(&mut rng, width);
            assert_eq!(bits.get_bit_count(), width);
        }
    }
}
