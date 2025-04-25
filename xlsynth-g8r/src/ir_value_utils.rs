// SPDX-License-Identifier: Apache-2.0

use xlsynth::ir_value::IrBits;

/// Converts a `&[bool]` slice into an IR `Bits` value.
///
/// ```
/// use xlsynth::ir_value::IrFormatPreference;
/// use xlsynth::IrBits;
/// use xlsynth_g8r::ir_value_utils::ir_bits_from_lsb_is_0;
///
/// let bools = vec![true, false, true, false]; // LSB is bools[0]
/// let ir_bits: IrBits = ir_bits_from_lsb_is_0(&bools);
/// assert_eq!(ir_bits.to_string_fmt(IrFormatPreference::Binary, false), "0b101");
/// assert_eq!(ir_bits.get_bit_count(), 4);
/// assert_eq!(ir_bits.get_bit(0).unwrap(), true); // LSB
/// assert_eq!(ir_bits.get_bit(1).unwrap(), false);
/// assert_eq!(ir_bits.get_bit(2).unwrap(), true);
/// assert_eq!(ir_bits.get_bit(3).unwrap(), false); // MSB
/// ```
pub fn ir_bits_from_lsb_is_0(bits: &[bool]) -> IrBits {
    assert!(bits.len() <= 64);
    let mut u64_value = 0;
    for (i, bit) in bits.iter().enumerate() {
        if *bit {
            u64_value |= 1 << i;
        }
    }
    IrBits::make_ubits(bits.len(), u64_value).unwrap()
}
