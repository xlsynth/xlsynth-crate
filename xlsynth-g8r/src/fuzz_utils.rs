// SPDX-License-Identifier: Apache-2.0

use crate::ir_value_utils;
use xlsynth::ir_value::IrBits;

/// Generates an arbitrary IrBits value of the given width using the provided
/// random number generator.
pub fn arbitrary_irbits<R: rand::Rng>(rng: &mut R, width: usize) -> IrBits {
    if width == 0 {
        return IrBits::make_ubits(0, 0).unwrap();
    }
    if width <= 64 {
        let value = rng.r#gen::<u64>();
        let value_masked = value & ((1u64 << width) - 1);
        // Unwrapping is safe since we masked the value to fit in the width.
        IrBits::make_ubits(width, value_masked).unwrap()
    } else {
        // Make a random sequence of bools of the requested width.
        let mut bools = Vec::with_capacity(width);
        for _ in 0..width {
            bools.push(rng.gen_bool(0.5));
        }
        ir_value_utils::ir_bits_from_lsb_is_0(&bools)
    }
}
