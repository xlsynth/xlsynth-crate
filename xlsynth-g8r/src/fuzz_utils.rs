// SPDX-License-Identifier: Apache-2.0

use xlsynth::ir_value::{IrBits, IrValue};
use xlsynth::xlsynth_error::XlsynthError;

/// Generates an arbitrary IrBits value of the given width using the provided
/// random number generator. The width must be <= 64.
pub fn arbitrary_irbits<R: rand::Rng>(rng: &mut R, width: usize) -> Result<IrBits, XlsynthError> {
    assert!(width > 0, "width must be positive, got {}", width);
    if width > 64 {
        // We build a string and then parse that. Inefficient, replace with better APIs
        // when they're available.
        let mut s = String::new();
        for _bit_index in 0..width {
            s.push(if rng.gen_bool(0.5) { '1' } else { '0' });
        }
        let value = IrValue::parse_typed(&format!("bits[{}]:0b{}", width, s))?;
        return value.to_bits();
    }
    let value = rng.gen::<u64>();
    let value_masked = value & ((1u64 << width) - 1);
    IrBits::make_ubits(width, value_masked)
}
