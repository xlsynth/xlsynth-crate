// SPDX-License-Identifier: Apache-2.0

use xlsynth::ir_value::IrBits;

/// Generates an arbitrary IrBits value of the given width using the provided
/// random number generator.
pub fn arbitrary_irbits<R: rand::Rng>(rng: &mut R, width: usize) -> IrBits {
    if width == 0 {
        return IrBits::make_ubits(0, 0).unwrap();
    }
    if width <= 64 {
        let value = rng.r#gen::<u64>();
        let value_masked = if width == u64::BITS as usize {
            value
        } else {
            value & ((1u64 << width) - 1)
        };
        // Unwrapping is safe since we masked the value to fit in the width.
        IrBits::make_ubits(width, value_masked).unwrap()
    } else {
        // Make a random sequence of bools of the requested width.
        let mut bools = Vec::with_capacity(width);
        for _ in 0..width {
            bools.push(rng.gen_bool(0.5));
        }
        IrBits::from_lsb_is_0(&bools)
    }
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
