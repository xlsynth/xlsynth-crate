// SPDX-License-Identifier: Apache-2.0

//! Explicit conversions between native PIR values and libxls-backed values.
//!
//! Keep these conversions at upstream XLS boundaries so ordinary PIR parsing,
//! rewriting, evaluation, and JIT execution do not allocate through libxls.

use crate::ir::Type;
use xlsynth::{XlsIrBits, XlsIrValue};

use crate::{IrBits, IrValue, ValueError};

/// Copies a libxls bitvector into canonical native PIR storage.
pub fn bits_from_libxls(bits: &XlsIrBits) -> Result<IrBits, ValueError> {
    let bytes = bits
        .to_le_bytes()
        .map_err(|error| ValueError(format!("libxls bits to bytes failed: {error}")))?;
    IrBits::from_le_bytes(bits.get_bit_count(), &bytes)
}

/// Copies native PIR bits into a libxls-owned bitvector.
pub fn bits_to_libxls(bits: &IrBits) -> Result<XlsIrBits, ValueError> {
    // The byte-oriented libxls constructor rejects an empty byte slice even
    // though bits[0] is a valid XLS value. Its scalar constructor does accept
    // width zero, so use that representation for this one degenerate width.
    if bits.get_bit_count() == 0 {
        return XlsIrBits::make_ubits(0, 0)
            .map_err(|error| ValueError(format!("native bits to libxls failed: {error}")));
    }
    let bytes = bits.to_le_bytes();
    XlsIrBits::from_le_bytes(bits.get_bit_count(), &bytes)
        .map_err(|error| ValueError(format!("native bits to libxls failed: {error}")))
}

/// Copies a typed libxls value into a native PIR value.
pub fn value_from_libxls(value: &XlsIrValue, ty: &Type) -> Result<IrValue, ValueError> {
    match ty {
        Type::Token => {
            if value.to_string() != "token" {
                return Err(ValueError(format!("expected token, got {value}")));
            }
            Ok(IrValue::make_token())
        }
        Type::Bits(width) => {
            let bits = value
                .to_bits()
                .map_err(|error| ValueError(format!("libxls value to bits failed: {error}")))?;
            if bits.get_bit_count() != *width {
                return Err(ValueError(format!("expected {ty}, got {value}")));
            }
            Ok(IrValue::from_bits(&bits_from_libxls(&bits)?))
        }
        Type::Tuple(member_types) => {
            check_aggregate_shape(value, ty, '(', member_types.len())?;
            let mut elements = Vec::with_capacity(member_types.len());
            for (index, member_type) in member_types.iter().enumerate() {
                let element = value
                    .get_element(index)
                    .map_err(|error| ValueError(format!("libxls tuple element failed: {error}")))?;
                elements.push(value_from_libxls(&element, member_type)?);
            }
            Ok(IrValue::make_tuple(&elements))
        }
        Type::Array(array) => {
            check_aggregate_shape(value, ty, '[', array.element_count)?;
            let mut elements = Vec::with_capacity(array.element_count);
            for index in 0..array.element_count {
                let element = value
                    .get_element(index)
                    .map_err(|error| ValueError(format!("libxls array element failed: {error}")))?;
                elements.push(value_from_libxls(&element, &array.element_type)?);
            }
            IrValue::make_array_typed((*array.element_type).clone(), &elements)
        }
    }
}

/// Checks the aggregate kind and arity before copying any children.
fn check_aggregate_shape(
    value: &XlsIrValue,
    ty: &Type,
    opener: char,
    count: usize,
) -> Result<(), ValueError> {
    // The Rust wrapper exposes aggregate elements but not a value-kind query.
    // Canonical XLS text distinguishes tuples from arrays at this boundary.
    if !value.to_string().starts_with(opener) || value.get_element_count().ok() != Some(count) {
        return Err(ValueError(format!("expected {ty}, got {value}")));
    }
    Ok(())
}

/// Copies a native PIR value into a libxls-owned value.
pub fn value_to_libxls(value: &IrValue) -> Result<XlsIrValue, ValueError> {
    match value {
        IrValue::Token => Ok(XlsIrValue::make_token()),
        IrValue::Bits(bits) => {
            let bits = bits_to_libxls(bits)?;
            Ok(XlsIrValue::from_bits(&bits))
        }
        IrValue::Tuple(elements) => {
            let elements = elements
                .iter()
                .map(value_to_libxls)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(XlsIrValue::make_tuple(&elements))
        }
        IrValue::Array(array) => {
            let elements = array.elements();
            if elements.is_empty() {
                return Err(ValueError(
                    "libxls XlsIrValue cannot represent a typed empty array".to_string(),
                ));
            }
            let elements = elements
                .iter()
                .map(value_to_libxls)
                .collect::<Result<Vec<_>, _>>()?;
            XlsIrValue::make_array(&elements)
                .map_err(|error| ValueError(format!("native array to libxls failed: {error}")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IrFormatPreference;
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};
    use xlsynth::ir_value::IrFormatPreference as XlsFormat;

    #[test]
    fn conversion_rejects_mismatched_types_without_dropping_children() {
        let scalar = XlsIrValue::make_ubits(8, 1).unwrap();
        let tuple = XlsIrValue::make_tuple(&[scalar.clone(), scalar.clone()]);
        assert!(value_from_libxls(&scalar, &Type::Token).is_err());
        assert!(value_from_libxls(&scalar, &Type::Bits(7)).is_err());
        assert!(value_from_libxls(&tuple, &Type::Tuple(vec![Box::new(Type::Bits(8))])).is_err());
        assert!(value_from_libxls(&tuple, &Type::new_array(Type::Bits(8), 2)).is_err());
    }

    #[test]
    fn native_bits_match_libxls_across_limb_boundaries() {
        let mut rng = StdRng::seed_from_u64(0x5eed);
        for width in [0usize, 1, 7, 8, 31, 32, 63, 64, 65, 127, 128, 129, 257] {
            let mut samples = vec![
                IrBits::zero(width),
                IrBits::all_ones(width),
                IrBits::signed_min_value(width),
            ];
            for _ in 0..8 {
                let mut bytes = vec![0; width.div_ceil(8)];
                rng.fill_bytes(&mut bytes);
                if width % 8 != 0 {
                    *bytes.last_mut().unwrap() &= (1 << (width % 8)) - 1;
                }
                samples.push(IrBits::from_le_bytes(width, &bytes).unwrap());
            }
            for a in &samples {
                let xa = bits_to_libxls(a).unwrap();
                assert_eq!(bits_from_libxls(&xa).unwrap(), *a);
                assert_eq!(a.to_string(), xa.to_string(), "width={width}");
                for (native, xls) in [
                    (a.not(), xa.not()),
                    (a.negate(), xa.negate()),
                    (a.abs(), xa.abs()),
                ] {
                    assert_eq!(native, bits_from_libxls(&xls).unwrap());
                }
                for start in [0, width / 2, width] {
                    let count = width - start;
                    assert_eq!(
                        a.width_slice(start as i64, count as i64),
                        bits_from_libxls(&xa.width_slice(start as i64, count as i64)).unwrap(),
                    );
                }
                for prefix in [false, true] {
                    for (fmt, xfmt) in [
                        (IrFormatPreference::Default, XlsFormat::Default),
                        (IrFormatPreference::Binary, XlsFormat::Binary),
                        (IrFormatPreference::Hex, XlsFormat::Hex),
                        (IrFormatPreference::PlainBinary, XlsFormat::PlainBinary),
                        (IrFormatPreference::PlainHex, XlsFormat::PlainHex),
                        (
                            IrFormatPreference::ZeroPaddedBinary,
                            XlsFormat::ZeroPaddedBinary,
                        ),
                        (IrFormatPreference::ZeroPaddedHex, XlsFormat::ZeroPaddedHex),
                        (IrFormatPreference::SignedDecimal, XlsFormat::SignedDecimal),
                        (
                            IrFormatPreference::UnsignedDecimal,
                            XlsFormat::UnsignedDecimal,
                        ),
                    ] {
                        assert_eq!(
                            a.to_string_fmt(fmt, prefix),
                            xa.to_string_fmt(xfmt, prefix),
                            "width={width}, format={fmt:?}"
                        );
                    }
                }
                for shift in [0, 1, width as i64, width as i64 + 1] {
                    assert_eq!(a.shll(shift), bits_from_libxls(&xa.shll(shift)).unwrap());
                    assert_eq!(a.shrl(shift), bits_from_libxls(&xa.shrl(shift)).unwrap());
                    assert_eq!(a.shra(shift), bits_from_libxls(&xa.shra(shift)).unwrap());
                }
                for b in &samples {
                    let xb = bits_to_libxls(b).unwrap();
                    for (result, reference) in [
                        (a.add(b), xa.add(&xb)),
                        (a.sub(b), xa.sub(&xb)),
                        (a.umul(b), xa.umul(&xb)),
                        (a.smul(b), xa.smul(&xb)),
                        (a.udiv(b), xa.udiv(&xb)),
                        (a.umod(b), xa.umod(&xb)),
                        (a.sdiv(b), xa.sdiv(&xb)),
                        (a.smod(b), xa.smod(&xb)),
                        (a.and(b), xa.and(&xb)),
                        (a.or(b), xa.or(&xb)),
                        (a.xor(b), xa.xor(&xb)),
                    ] {
                        assert_eq!(
                            result,
                            bits_from_libxls(&reference).unwrap(),
                            "width={width}, a={a}, b={b}"
                        );
                    }
                    assert_eq!(
                        (a.ult(b), a.ule(b), a.slt(b), a.sle(b)),
                        (xa.ult(&xb), xa.ule(&xb), xa.slt(&xb), xa.sle(&xb))
                    );
                }
            }
        }
    }

    #[test]
    fn native_parser_matches_libxls_for_signed_and_aggregate_values() {
        let mut mismatches = Vec::new();
        for text in [
            "bits[0]:0",
            "bits[0]:-1",
            "bits[1]:-2",
            "bits[8]:-128",
            "bits[8]:-129",
            "bits[8]:-256",
            "bits[8]:-257",
            "bits[8]: 42",
            "bits[8]:0xff",
            "bits[8]:256",
            "bits[8]:-0x80",
            "bits[8]:-0b1000_0000",
            "bits[8]:0Xff",
            "bits [ 8 ] : 42",
            "bits[8]:+1",
            "bits[8]:",
            "bits[8]:0b2",
            "bits[65]:0x1_ffff_ffff_ffff_ffff",
            "(bits[4]:3, token)",
            "[bits[4]:3, bits[4]:2]",
            "(bits[4]:3,)",
            "[bits[4]:3,]",
            "()",
            "[]",
            "(token, (), [bits[0]:0, bits[0]:0])",
            "[bits[4]:3, bits[8]:2]",
        ] {
            let native = IrValue::parse_typed(text);
            let reference = XlsIrValue::parse_typed(text);
            if native.is_ok() != reference.is_ok() {
                mismatches.push((text, native.is_ok(), reference.is_ok()));
            }
            if let (Ok(native), Ok(reference)) = (native, reference) {
                assert_eq!(value_to_libxls(&native).unwrap(), reference, "{text}");
            }
        }
        assert_eq!(mismatches, vec![]);
    }

    #[test]
    fn aggregate_values_round_trip_across_the_libxls_boundary() {
        let array = IrValue::make_array(&[
            IrValue::make_ubits(8, 3).unwrap(),
            IrValue::make_ubits(8, 5).unwrap(),
        ])
        .unwrap();
        let native = IrValue::make_tuple(&[
            IrValue::make_token(),
            IrValue::make_ubits(0, 0).unwrap(),
            array,
        ]);
        let ty = native.type_();

        let libxls = value_to_libxls(&native).unwrap();
        let round_trip = value_from_libxls(&libxls, &ty).unwrap();

        assert_eq!(round_trip, native);
    }
}
