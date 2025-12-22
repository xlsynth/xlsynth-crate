// SPDX-License-Identifier: Apache-2.0

use crate::ir;
use bitvec::vec::BitVec;
use xlsynth::{IrValue, ir_value::IrBits};

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

pub fn ir_value_from_bits_with_type(bits: &IrBits, ty: &ir::Type) -> IrValue {
    match ty {
        ir::Type::Bits(width) => {
            assert_eq!(bits.get_bit_count(), *width);
            IrValue::from_bits(bits)
        }
        ir::Type::Array(array_ty) => {
            let elem_width = array_ty.element_type.bit_count();
            let mut elements = Vec::with_capacity(array_ty.element_count);
            for i in 0..array_ty.element_count {
                let start = i * elem_width;
                let end = start + elem_width;
                let mut elem_bits_vec = Vec::with_capacity(elem_width);
                for j in start..end {
                    elem_bits_vec.push(bits.get_bit(j).unwrap());
                }
                let elem_bits = IrBits::from_lsb_is_0(&elem_bits_vec);
                let elem_value = ir_value_from_bits_with_type(&elem_bits, &array_ty.element_type);
                elements.push(elem_value);
            }
            IrValue::make_array(&elements).unwrap()
        }
        ir::Type::Tuple(types) => {
            let mut elements = Vec::with_capacity(types.len());
            let mut offset = 0;
            for t in types.iter() {
                let t_width = t.bit_count();
                let mut elem_bits_vec = Vec::with_capacity(t_width);
                for j in offset..offset + t_width {
                    elem_bits_vec.push(bits.get_bit(j).unwrap());
                }
                let elem_bits = IrBits::from_lsb_is_0(&elem_bits_vec);
                let elem_value = ir_value_from_bits_with_type(&elem_bits, t);
                elements.push(elem_value);
                offset += t_width;
            }
            IrValue::make_tuple(&elements)
        }
        ir::Type::Token => IrValue::make_tuple(&[]), // Tokens are zero bits
    }
}

pub fn ir_bits_from_value_with_type(value: &IrValue, ty: &ir::Type) -> IrBits {
    match ty {
        ir::Type::Bits(width) => {
            let bits = value.to_bits().expect("bits value must be bits");
            assert_eq!(
                bits.get_bit_count(),
                *width,
                "bits width mismatch: expected {} got {}",
                width,
                bits.get_bit_count()
            );
            bits
        }
        ir::Type::Array(array_ty) => {
            let elements = value.get_elements().expect("array value must be array");
            assert_eq!(
                elements.len(),
                array_ty.element_count,
                "array length mismatch: expected {} got {}",
                array_ty.element_count,
                elements.len()
            );
            let mut out: Vec<bool> = Vec::with_capacity(ty.bit_count());
            for elem in elements.iter() {
                let elem_bits = ir_bits_from_value_with_type(elem, &array_ty.element_type);
                for i in 0..elem_bits.get_bit_count() {
                    out.push(elem_bits.get_bit(i).unwrap());
                }
            }
            IrBits::from_lsb_is_0(&out)
        }
        ir::Type::Tuple(types) => {
            let elements = value.get_elements().expect("tuple value must be tuple");
            assert_eq!(
                elements.len(),
                types.len(),
                "tuple arity mismatch: expected {} got {}",
                types.len(),
                elements.len()
            );
            let mut out: Vec<bool> = Vec::with_capacity(ty.bit_count());
            for (elem, elem_ty) in elements.iter().zip(types.iter()) {
                let elem_bits = ir_bits_from_value_with_type(elem, elem_ty);
                for i in 0..elem_bits.get_bit_count() {
                    out.push(elem_bits.get_bit(i).unwrap());
                }
            }
            IrBits::from_lsb_is_0(&out)
        }
        ir::Type::Token => IrBits::make_ubits(0, 0).unwrap(),
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
        let ir = IrBits::from_lsb_is_0(&bits);
        assert_eq!(ir, IrBits::make_ubits(4, 0b0101).unwrap());

        // Large: 100 bits, alternating true/false, LSB at index 0
        let mut bits = Vec::with_capacity(100);
        for i in 0..100 {
            bits.push(i % 2 == 0);
        }
        let ir = IrBits::from_lsb_is_0(&bits);
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
        let ir = IrBits::from_lsb_is_0(&bits);
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
        let ir = IrBits::from_msb_is_0(&bits);
        assert_eq!(ir, IrBits::make_ubits(0, 0).unwrap());
        assert_eq!(ir.get_bit_count(), 0);
    }

    #[test]
    fn test_ir_bits_from_value_with_type_round_trips_tuple_and_array() {
        let ty = ir::Type::Tuple(vec![
            Box::new(ir::Type::Bits(4)),
            Box::new(ir::Type::new_array(ir::Type::Bits(2), 3)),
        ]);
        let v = IrValue::make_tuple(&[
            IrValue::make_ubits(4, 0b1010).unwrap(),
            IrValue::make_array(&[
                IrValue::make_ubits(2, 0).unwrap(),
                IrValue::make_ubits(2, 1).unwrap(),
                IrValue::make_ubits(2, 2).unwrap(),
            ])
            .unwrap(),
        ]);
        let bits = ir_bits_from_value_with_type(&v, &ty);
        assert_eq!(bits.get_bit_count(), ty.bit_count());
        let v2 = ir_value_from_bits_with_type(&bits, &ty);
        assert_eq!(v2, v);
    }
}
