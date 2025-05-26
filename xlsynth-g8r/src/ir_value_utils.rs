// SPDX-License-Identifier: Apache-2.0

use bitvec::vec::BitVec;
use xlsynth::{ir_value::IrBits, IrValue};

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
    if bits.is_empty() {
        return IrBits::make_ubits(0, 0).unwrap();
    }
    let mut s: String = format!("bits[{}]:0b", bits.len());
    for b in bits.iter().rev() {
        s.push(if *b { '1' } else { '0' });
    }
    IrValue::parse_typed(&s).unwrap().to_bits().unwrap()
}

pub fn ir_bits_from_bitvec_lsb_is_0(bv: &BitVec) -> IrBits {
    if bv.is_empty() {
        return IrBits::make_ubits(0, 0).unwrap();
    }
    let mut s: String = format!("bits[{}]:0b", bv.len());
    for b in bv.iter().rev() {
        s.push(if *b { '1' } else { '0' });
    }
    IrValue::parse_typed(&s).unwrap().to_bits().unwrap()
}

pub fn ir_bits_from_bitvec_msb_is_0(bv: &BitVec) -> IrBits {
    if bv.is_empty() {
        return IrBits::make_ubits(0, 0).unwrap();
    }
    let mut s: String = format!("bits[{}]:0b", bv.len());
    for b in bv.iter() {
        s.push(if *b { '1' } else { '0' });
    }
    IrValue::parse_typed(&s).unwrap().to_bits().unwrap()
}

/// Turns a boolean slice into an IR `Bits` value under the assumption that
/// index 0 in the slice is the most significant bit (MSb).
pub fn ir_bits_from_msb_is_0(bits: &[bool]) -> IrBits {
    if bits.is_empty() {
        return IrBits::make_ubits(0, 0).unwrap();
    }
    let mut s: String = format!("bits[{}]:0b", bits.len());
    for b in bits {
        s.push(if *b { '1' } else { '0' });
    }
    IrValue::parse_typed(&s).unwrap().to_bits().unwrap()
}

pub fn ir_value_from_bits_with_type(
    bits: &IrBits,
    ty: &crate::xls_ir::ir::Type,
) -> xlsynth::IrValue {
    match ty {
        crate::xls_ir::ir::Type::Bits(width) => {
            assert_eq!(bits.get_bit_count(), *width);
            xlsynth::IrValue::from_bits(bits)
        }
        crate::xls_ir::ir::Type::Array(array_ty) => {
            let elem_width = array_ty.element_type.bit_count();
            let mut elements = Vec::with_capacity(array_ty.element_count);
            for i in 0..array_ty.element_count {
                let start = i * elem_width;
                let end = start + elem_width;
                let mut elem_bits_vec = Vec::with_capacity(elem_width);
                for j in start..end {
                    elem_bits_vec.push(bits.get_bit(j).unwrap());
                }
                let elem_bits = ir_bits_from_lsb_is_0(&elem_bits_vec);
                let elem_value = ir_value_from_bits_with_type(&elem_bits, &array_ty.element_type);
                elements.push(elem_value);
            }
            xlsynth::IrValue::make_array(&elements).unwrap()
        }
        crate::xls_ir::ir::Type::Tuple(types) => {
            let mut elements = Vec::with_capacity(types.len());
            let mut offset = 0;
            for t in types.iter() {
                let t_width = t.bit_count();
                let mut elem_bits_vec = Vec::with_capacity(t_width);
                for j in offset..offset + t_width {
                    elem_bits_vec.push(bits.get_bit(j).unwrap());
                }
                let elem_bits = ir_bits_from_lsb_is_0(&elem_bits_vec);
                let elem_value = ir_value_from_bits_with_type(&elem_bits, t);
                elements.push(elem_value);
                offset += t_width;
            }
            xlsynth::IrValue::make_tuple(&elements)
        }
        crate::xls_ir::ir::Type::Token => xlsynth::IrValue::make_tuple(&[]), // Tokens are zero bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::vec::BitVec;
    use xlsynth::IrBits;

    #[test]
    fn test_ir_bits_from_lsb_is_0() {
        // Small: bits: [true, false, true, false] => 0b0101 (LSB at index 0)
        let bits = [true, false, true, false];
        let ir = ir_bits_from_lsb_is_0(&bits);
        assert_eq!(ir, IrBits::make_ubits(4, 0b0101).unwrap());

        // Large: 100 bits, alternating true/false, LSB at index 0
        let mut bits = Vec::with_capacity(100);
        for i in 0..100 {
            bits.push(i % 2 == 0);
        }
        let ir = ir_bits_from_lsb_is_0(&bits);
        // Check a few bits
        for i in 0..100 {
            assert_eq!(ir.get_bit(i).unwrap(), i % 2 == 0, "bit {}", i);
        }
        assert_eq!(ir.get_bit_count(), 100);
    }

    #[test]
    fn test_ir_bits_from_bitvec_lsb_is_0() {
        // Small: BitVec: [true, false, true, false] => 0b0101 (LSB at index 0)
        let mut bv = BitVec::new();
        bv.push(true); // LSB
        bv.push(false);
        bv.push(true);
        bv.push(false); // MSB
        let ir = ir_bits_from_bitvec_lsb_is_0(&bv);
        assert_eq!(ir, IrBits::make_ubits(4, 0b0101).unwrap());

        // Large: 100 bits, alternating true/false, LSB at index 0
        let mut bv = BitVec::new();
        for i in 0..100 {
            bv.push(i % 2 == 0);
        }
        let ir = ir_bits_from_bitvec_lsb_is_0(&bv);
        for i in 0..100 {
            assert_eq!(ir.get_bit(i).unwrap(), i % 2 == 0, "bit {}", i);
        }
        assert_eq!(ir.get_bit_count(), 100);
    }

    #[test]
    fn test_ir_bits_from_bitvec_msb_is_0() {
        // Small: BitVec: [true, false, true, false] => 0b1010 (MSB at index 0)
        let mut bv = BitVec::new();
        bv.push(true); // MSB
        bv.push(false);
        bv.push(true);
        bv.push(false); // LSB
        let ir = ir_bits_from_bitvec_msb_is_0(&bv);
        assert_eq!(ir, IrBits::make_ubits(4, 0b1010).unwrap());

        // Large: 100 bits, alternating true/false, MSB at index 0
        let mut bv = BitVec::new();
        for i in 0..100 {
            bv.push(i % 2 == 0);
        }
        let ir = ir_bits_from_bitvec_msb_is_0(&bv);
        // In this case, bit 0 is MSB, so ir.get_bit(99) is bv[0], ir.get_bit(0) is
        // bv[99]
        for i in 0..100 {
            assert_eq!(ir.get_bit(99 - i).unwrap(), i % 2 == 0, "bit {}", 99 - i);
        }
        assert_eq!(ir.get_bit_count(), 100);
    }

    #[test]
    fn test_ir_bits_from_lsb_is_0_zero_bits() {
        let bits: [bool; 0] = [];
        let ir = ir_bits_from_lsb_is_0(&bits);
        assert_eq!(ir, IrBits::make_ubits(0, 0).unwrap());
        assert_eq!(ir.get_bit_count(), 0);
    }

    #[test]
    fn test_ir_bits_from_bitvec_lsb_is_0_zero_bits() {
        let bv = BitVec::new();
        let ir = ir_bits_from_bitvec_lsb_is_0(&bv);
        assert_eq!(ir, IrBits::make_ubits(0, 0).unwrap());
        assert_eq!(ir.get_bit_count(), 0);
    }

    #[test]
    fn test_ir_bits_from_bitvec_msb_is_0_zero_bits() {
        let bv = BitVec::new();
        let ir = ir_bits_from_bitvec_msb_is_0(&bv);
        assert_eq!(ir, IrBits::make_ubits(0, 0).unwrap());
        assert_eq!(ir.get_bit_count(), 0);
    }

    #[test]
    fn test_ir_bits_from_msb_is_0_zero_bits() {
        let bits: [bool; 0] = [];
        let ir = ir_bits_from_msb_is_0(&bits);
        assert_eq!(ir, IrBits::make_ubits(0, 0).unwrap());
        assert_eq!(ir.get_bit_count(), 0);
    }
}
