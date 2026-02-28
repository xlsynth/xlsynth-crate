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

pub fn zero_ir_value_for_type(ty: &ir::Type) -> IrValue {
    match ty {
        ir::Type::Token => IrValue::make_token(),
        ir::Type::Bits(width) => IrValue::make_ubits(*width, 0).expect("bits zero must construct"),
        ir::Type::Tuple(member_types) => {
            let elements: Vec<IrValue> = member_types
                .iter()
                .map(|member_ty| zero_ir_value_for_type(member_ty))
                .collect();
            IrValue::make_tuple(&elements)
        }
        ir::Type::Array(array_ty) => {
            let elements: Vec<IrValue> = (0..array_ty.element_count)
                .map(|_| zero_ir_value_for_type(&array_ty.element_type))
                .collect();
            IrValue::make_array(&elements).expect("zero array elements must share the same type")
        }
    }
}

/// Recursively ORs two values according to the provided PIR type.
pub fn deep_or_ir_values_for_type(ty: &ir::Type, lhs: &IrValue, rhs: &IrValue) -> IrValue {
    match ty {
        ir::Type::Token => IrValue::make_token(),
        ir::Type::Bits(width) => {
            let lhs_bits = lhs
                .to_bits()
                .expect("one_hot_sel bits case lhs must be bits");
            let rhs_bits = rhs
                .to_bits()
                .expect("one_hot_sel bits case rhs must be bits");
            assert_eq!(
                lhs_bits.get_bit_count(),
                *width,
                "one_hot_sel lhs width must match node type"
            );
            assert_eq!(
                rhs_bits.get_bit_count(),
                *width,
                "one_hot_sel rhs width must match node type"
            );
            IrValue::from_bits(&lhs_bits.or(&rhs_bits))
        }
        ir::Type::Tuple(member_types) => {
            let lhs_count = lhs
                .get_element_count()
                .expect("one_hot_sel tuple lhs must have elements");
            let rhs_count = rhs
                .get_element_count()
                .expect("one_hot_sel tuple rhs must have elements");
            assert_eq!(
                lhs_count,
                member_types.len(),
                "one_hot_sel tuple lhs arity must match node type"
            );
            assert_eq!(
                rhs_count,
                member_types.len(),
                "one_hot_sel tuple rhs arity must match node type"
            );
            let elements: Vec<IrValue> = member_types
                .iter()
                .enumerate()
                .map(|(i, member_ty)| {
                    let lhs_elem = lhs
                        .get_element(i)
                        .expect("one_hot_sel tuple lhs element must exist");
                    let rhs_elem = rhs
                        .get_element(i)
                        .expect("one_hot_sel tuple rhs element must exist");
                    deep_or_ir_values_for_type(member_ty, &lhs_elem, &rhs_elem)
                })
                .collect();
            IrValue::make_tuple(&elements)
        }
        ir::Type::Array(array_ty) => {
            let lhs_count = lhs
                .get_element_count()
                .expect("one_hot_sel array lhs must have elements");
            let rhs_count = rhs
                .get_element_count()
                .expect("one_hot_sel array rhs must have elements");
            assert_eq!(
                lhs_count, array_ty.element_count,
                "one_hot_sel array lhs length must match node type"
            );
            assert_eq!(
                rhs_count, array_ty.element_count,
                "one_hot_sel array rhs length must match node type"
            );
            let elements: Vec<IrValue> = (0..array_ty.element_count)
                .map(|i| {
                    let lhs_elem = lhs
                        .get_element(i)
                        .expect("one_hot_sel array lhs element must exist");
                    let rhs_elem = rhs
                        .get_element(i)
                        .expect("one_hot_sel array rhs element must exist");
                    deep_or_ir_values_for_type(&array_ty.element_type, &lhs_elem, &rhs_elem)
                })
                .collect();
            IrValue::make_array(&elements)
                .expect("one_hot_sel array deep-or elements must share the same type")
        }
    }
}

/// Converts an `IrBits` value to `usize` when all set bits are representable on
/// the current host width, and returns `None` when any set bit lies above the
/// host `usize` width.
pub fn ir_bits_to_usize(bits: &IrBits) -> Option<usize> {
    let usize_width = usize::BITS as usize;
    for i in usize_width..bits.get_bit_count() {
        if bits.get_bit(i).expect("bit index is in bounds") {
            return None;
        }
    }

    let mut value = 0usize;
    let low_width = std::cmp::min(bits.get_bit_count(), usize_width);
    for i in 0..low_width {
        if bits.get_bit(i).expect("bit index is in bounds") {
            value |= 1usize << i;
        }
    }
    Some(value)
}

/// Converts an `IrBits` value to `usize` when it is strictly less than the
/// given exclusive upper bound, and returns `None` when the value is not
/// representable as a host `usize` or is out of range.
pub fn ir_bits_to_usize_in_range(bits: &IrBits, upper_bound_exclusive: usize) -> Option<usize> {
    let value = ir_bits_to_usize(bits)?;
    (value < upper_bound_exclusive).then_some(value)
}

/// Flattens `value` into a single `IrBits`, using `ty` to define and validate
/// the shape.
///
/// This is the (checked) inverse of [`ir_value_from_bits_with_type`]: the
/// produced `IrBits` will round-trip via `ir_value_from_bits_with_type(&bits,
/// ty)` when `value` conforms to `ty`.
///
/// For composite types, elements are concatenated in order (tuple element order
/// and array index order) into a flat bit-vector where bit index 0 is the
/// overall LSb (matching `IrBits` APIs).
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

/// Flattens `value` into LSB-first bits following PIR/Gatify lowering layout.
///
/// This differs from [`ir_bits_from_value_with_type`] for tuples/arrays:
/// tuple/array tails occupy the least-significant bits, matching the layout
/// used when lowering PIR params into gate-level inputs.
pub fn flatten_ir_value_to_lsb0_bits_for_type(
    value: &IrValue,
    ty: &ir::Type,
    out: &mut Vec<bool>,
) -> Result<(), String> {
    match ty {
        ir::Type::Token => Ok(()),
        ir::Type::Bits(width) => {
            let bits = value.to_bits().map_err(|e| e.to_string())?;
            if bits.get_bit_count() != *width {
                return Err(format!(
                    "bits width mismatch: value has bits[{}] but type is bits[{}]",
                    bits.get_bit_count(),
                    width
                ));
            }
            for i in 0..*width {
                out.push(bits.get_bit(i).map_err(|e| e.to_string())?);
            }
            Ok(())
        }
        ir::Type::Tuple(elem_types) => {
            let elems = value.get_elements().map_err(|e| e.to_string())?;
            if elems.len() != elem_types.len() {
                return Err(format!(
                    "tuple arity mismatch: value has {} elems but type expects {}",
                    elems.len(),
                    elem_types.len()
                ));
            }
            for (elem, elem_ty) in elems.iter().rev().zip(elem_types.iter().rev()) {
                flatten_ir_value_to_lsb0_bits_for_type(elem, elem_ty, out)?;
            }
            Ok(())
        }
        ir::Type::Array(ir::ArrayTypeData {
            element_type,
            element_count,
        }) => {
            let got_count = value.get_element_count().map_err(|e| e.to_string())?;
            if got_count != *element_count {
                return Err(format!(
                    "array length mismatch: value has {} elems but type expects {}",
                    got_count, element_count
                ));
            }
            for i in (0..*element_count).rev() {
                let elem = value.get_element(i).map_err(|e| e.to_string())?;
                flatten_ir_value_to_lsb0_bits_for_type(&elem, element_type, out)?;
            }
            Ok(())
        }
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

    #[test]
    fn test_flatten_ir_value_to_lsb0_bits_for_type_tuple_tail_first() {
        let ty = ir::Type::Tuple(vec![
            Box::new(ir::Type::Bits(2)),
            Box::new(ir::Type::Bits(3)),
        ]);
        let v = IrValue::make_tuple(&[
            IrValue::make_ubits(2, 0b10).unwrap(),
            IrValue::make_ubits(3, 0b101).unwrap(),
        ]);
        let mut out = Vec::new();
        flatten_ir_value_to_lsb0_bits_for_type(&v, &ty, &mut out).unwrap();
        // Tail-first flattening:
        // bits[3]:101 contributes [1,0,1], then bits[2]:10 contributes [0,1].
        assert_eq!(out, vec![true, false, true, false, true]);
    }

    #[test]
    fn test_flatten_ir_value_to_lsb0_bits_for_type_rejects_mismatch() {
        let ty = ir::Type::Bits(4);
        let v = IrValue::make_ubits(3, 0b111).unwrap();
        let mut out = Vec::new();
        let err = flatten_ir_value_to_lsb0_bits_for_type(&v, &ty, &mut out).unwrap_err();
        assert!(err.contains("bits width mismatch"));
    }
}
