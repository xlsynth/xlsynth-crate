// SPDX-License-Identifier: Apache-2.0

use xlsynth_sys::{CIrBits, CIrValue};

use crate::{
    lib_support::{
        xls_bits_make_sbits, xls_bits_make_ubits, xls_bits_to_debug_str, xls_bits_to_string,
        xls_format_preference_from_string, xls_value_eq, xls_value_free, xls_value_get_bits,
        xls_value_get_element, xls_value_get_element_count, xls_value_make_array,
        xls_value_make_sbits, xls_value_make_tuple, xls_value_make_ubits, xls_value_to_string,
        xls_value_to_string_format_preference,
    },
    xls_parse_typed_value,
    xlsynth_error::XlsynthError,
};

pub struct IrBits {
    #[allow(dead_code)]
    pub(crate) ptr: *mut CIrBits,
}

impl IrBits {
    pub fn make_ubits(bit_count: usize, value: u64) -> Result<Self, XlsynthError> {
        xls_bits_make_ubits(bit_count, value)
    }

    pub fn make_sbits(bit_count: usize, value: i64) -> Result<Self, XlsynthError> {
        xls_bits_make_sbits(bit_count, value)
    }

    pub fn get_bit_count(&self) -> usize {
        let bit_count = unsafe { xlsynth_sys::xls_bits_get_bit_count(self.ptr) };
        assert!(bit_count >= 0);
        bit_count as usize
    }

    pub fn to_debug_str(&self) -> String {
        xls_bits_to_debug_str(self.ptr)
    }

    /// Note: index 0 is the least significant bit (LSb).
    pub fn get_bit(&self, index: usize) -> Result<bool, XlsynthError> {
        if self.get_bit_count() <= index {
            return Err(XlsynthError(format!(
                "Index {} out of bounds for bits[{}]:{}",
                index,
                self.get_bit_count(),
                self.to_debug_str()
            )));
        }
        let bit = unsafe { xlsynth_sys::xls_bits_get_bit(self.ptr, index as i64) };
        Ok(bit)
    }

    pub fn to_string_fmt(&self, format: IrFormatPreference, include_bit_count: bool) -> String {
        let fmt_pref: xlsynth_sys::XlsFormatPreference =
            xls_format_preference_from_string(format.to_string()).unwrap();
        xls_bits_to_string(self.ptr, fmt_pref, include_bit_count).unwrap()
    }

    #[allow(dead_code)]
    fn to_hex_string(&self) -> String {
        let value = self.to_string_fmt(IrFormatPreference::Hex, false);
        format!("bits[{}]:{}", self.get_bit_count(), value)
    }

    pub fn umul(&self, rhs: &IrBits) -> IrBits {
        let result = unsafe { xlsynth_sys::xls_bits_umul(self.ptr, rhs.ptr) };
        IrBits { ptr: result }
    }

    pub fn smul(&self, rhs: &IrBits) -> IrBits {
        let result = unsafe { xlsynth_sys::xls_bits_smul(self.ptr, rhs.ptr) };
        IrBits { ptr: result }
    }

    pub fn negate(&self) -> IrBits {
        let result = unsafe { xlsynth_sys::xls_bits_negate(self.ptr) };
        IrBits { ptr: result }
    }

    pub fn abs(&self) -> IrBits {
        let result = unsafe { xlsynth_sys::xls_bits_abs(self.ptr) };
        IrBits { ptr: result }
    }

    pub fn msb(&self) -> bool {
        self.get_bit(self.get_bit_count() - 1).unwrap()
    }

    pub fn shll(&self, shift_amount: i64) -> IrBits {
        let result = unsafe { xlsynth_sys::xls_bits_shift_left_logical(self.ptr, shift_amount) };
        IrBits { ptr: result }
    }

    pub fn shrl(&self, shift_amount: i64) -> IrBits {
        let result = unsafe { xlsynth_sys::xls_bits_shift_right_logical(self.ptr, shift_amount) };
        IrBits { ptr: result }
    }

    pub fn shra(&self, shift_amount: i64) -> IrBits {
        let result =
            unsafe { xlsynth_sys::xls_bits_shift_right_arithmetic(self.ptr, shift_amount) };
        IrBits { ptr: result }
    }

    pub fn width_slice(&self, start: i64, width: i64) -> IrBits {
        let result = unsafe { xlsynth_sys::xls_bits_width_slice(self.ptr, start, width) };
        IrBits { ptr: result }
    }

    pub fn not(&self) -> IrBits {
        let result = unsafe { xlsynth_sys::xls_bits_not(self.ptr) };
        IrBits { ptr: result }
    }

    pub fn and(&self, rhs: &IrBits) -> IrBits {
        let result = unsafe { xlsynth_sys::xls_bits_and(self.ptr, rhs.ptr) };
        IrBits { ptr: result }
    }

    pub fn or(&self, rhs: &IrBits) -> IrBits {
        let result = unsafe { xlsynth_sys::xls_bits_or(self.ptr, rhs.ptr) };
        IrBits { ptr: result }
    }

    pub fn xor(&self, rhs: &IrBits) -> IrBits {
        let result = unsafe { xlsynth_sys::xls_bits_xor(self.ptr, rhs.ptr) };
        IrBits { ptr: result }
    }
}

impl Clone for IrBits {
    fn clone(&self) -> Self {
        // TODO(cdleary): 2025-04-14 Right now we don't have a direct clone API for
        // IrBits. Adding one would make this more efficient.
        let value = IrValue::from_bits(self);
        let clone = value.clone();
        clone.to_bits().unwrap()
    }
}

impl Drop for IrBits {
    fn drop(&mut self) {
        unsafe { xlsynth_sys::xls_bits_free(self.ptr) }
    }
}

impl std::cmp::PartialEq for IrBits {
    fn eq(&self, other: &Self) -> bool {
        unsafe { xlsynth_sys::xls_bits_eq(self.ptr, other.ptr) }
    }
}

impl std::fmt::Debug for IrBits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_debug_str())
    }
}

impl From<&IrBits> for IrValue {
    fn from(bits: &IrBits) -> Self {
        IrValue::from_bits(bits)
    }
}

impl std::fmt::Display for IrBits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "bits[{}]:{}",
            self.get_bit_count(),
            self.to_string_fmt(IrFormatPreference::Default, false)
        )
    }
}

impl std::ops::Add for IrBits {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let result = unsafe { xlsynth_sys::xls_bits_add(self.ptr, rhs.ptr) };
        Self { ptr: result }
    }
}

impl std::ops::BitAnd for IrBits {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        let result = unsafe { xlsynth_sys::xls_bits_and(self.ptr, rhs.ptr) };
        Self { ptr: result }
    }
}

impl std::ops::BitOr for IrBits {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        IrBits::or(&self, &rhs)
    }
}

impl std::ops::BitXor for IrBits {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        IrBits::xor(&self, &rhs)
    }
}

impl std::ops::Not for IrBits {
    type Output = Self;

    fn not(self) -> Self::Output {
        IrBits::not(&self)
    }
}

// --

pub enum IrFormatPreference {
    Default,
    Binary,
    SignedDecimal,
    UnsignedDecimal,
    Hex,
    PlainBinary,
    ZeroPaddedBinary,
    PlainHex,
    ZeroPaddedHex,
}

impl IrFormatPreference {
    pub fn to_string(&self) -> &'static str {
        match self {
            IrFormatPreference::Default => "default",
            IrFormatPreference::Binary => "binary",
            IrFormatPreference::SignedDecimal => "signed_decimal",
            IrFormatPreference::UnsignedDecimal => "unsigned_decimal",
            IrFormatPreference::Hex => "hex",
            IrFormatPreference::PlainBinary => "plain_binary",
            IrFormatPreference::ZeroPaddedBinary => "zero_padded_binary",
            IrFormatPreference::PlainHex => "plain_hex",
            IrFormatPreference::ZeroPaddedHex => "zero_padded_hex",
        }
    }
}

pub struct IrValue {
    pub(crate) ptr: *mut CIrValue,
}

impl IrValue {
    pub fn make_tuple(elements: &[IrValue]) -> Self {
        xls_value_make_tuple(elements)
    }

    /// Returns an error if the elements do not all have the same type.
    pub fn make_array(elements: &[IrValue]) -> Result<Self, XlsynthError> {
        xls_value_make_array(elements)
    }

    pub fn from_bits(bits: &IrBits) -> Self {
        let ptr = unsafe { xlsynth_sys::xls_value_from_bits(bits.ptr) };
        Self { ptr }
    }

    pub fn parse_typed(s: &str) -> Result<Self, XlsynthError> {
        xls_parse_typed_value(s)
    }

    pub fn bool(value: bool) -> Self {
        xls_value_make_ubits(value as u64, 1).unwrap()
    }

    pub fn u32(value: u32) -> Self {
        // Unwrap should be ok since the u32 always fits.
        xls_value_make_ubits(value as u64, 32).unwrap()
    }

    pub fn u64(value: u64) -> Self {
        // Unwrap should be ok since the u64 always fits.
        xls_value_make_ubits(value as u64, 64).unwrap()
    }

    pub fn make_ubits(bit_count: usize, value: u64) -> Result<Self, XlsynthError> {
        xls_value_make_ubits(value as u64, bit_count)
    }

    pub fn make_sbits(bit_count: usize, value: i64) -> Result<Self, XlsynthError> {
        xls_value_make_sbits(value, bit_count)
    }

    pub fn bit_count(&self) -> Result<usize, XlsynthError> {
        // TODO(cdleary): 2024-06-23 Expose a more efficient API for this from libxls.so
        let bits = self.to_bits()?;
        Ok(bits.get_bit_count())
    }

    pub fn to_string_fmt(&self, format: IrFormatPreference) -> Result<String, XlsynthError> {
        let fmt_pref: xlsynth_sys::XlsFormatPreference =
            xls_format_preference_from_string(format.to_string())?;
        xls_value_to_string_format_preference(self.ptr, fmt_pref)
    }

    pub fn to_string_fmt_no_prefix(
        &self,
        format: IrFormatPreference,
    ) -> Result<String, XlsynthError> {
        let s = self.to_string_fmt(format)?;
        if s.starts_with("bits[") {
            let parts: Vec<&str> = s.split(':').collect();
            Ok(parts[1].to_string())
        } else {
            Ok(s)
        }
    }

    pub fn to_bool(&self) -> Result<bool, XlsynthError> {
        let bits = self.to_bits()?;
        if bits.get_bit_count() != 1 {
            return Err(XlsynthError(format!(
                "IrValue {} is not single-bit; must be bits[1] to convert to bool",
                self.to_string()
            )));
        }
        bits.get_bit(0)
    }

    pub fn to_i64(&self) -> Result<i64, XlsynthError> {
        let string = self.to_string_fmt(IrFormatPreference::SignedDecimal)?;
        let number = string.split(':').nth(1).expect("split success");
        match number.parse::<i64>() {
            Ok(i) => Ok(i),
            Err(e) => Err(XlsynthError(format!(
                "IrValue::to_i64() failed to parse i64 from string: {}",
                e
            ))),
        }
    }

    pub fn to_u64(&self) -> Result<u64, XlsynthError> {
        let string = self.to_string_fmt(IrFormatPreference::UnsignedDecimal)?;
        let number = string.split(':').nth(1).expect("split success");
        match number.parse::<u64>() {
            Ok(i) => Ok(i),
            Err(e) => Err(XlsynthError(format!(
                "IrValue::to_u64() failed to parse u64 from string: {}",
                e
            ))),
        }
    }

    pub fn to_u32(&self) -> Result<u32, XlsynthError> {
        let string = self.to_string_fmt(IrFormatPreference::UnsignedDecimal)?;
        let number = string.split(':').nth(1).expect("split success");
        match number.parse::<u32>() {
            Ok(i) => Ok(i),
            Err(e) => Err(XlsynthError(format!(
                "IrValue::to_u32() failed to parse u32 from string: {}",
                e
            ))),
        }
    }

    /// Attempts to extract the bits contents underlying this value.
    ///
    /// If this value is not a bits type, an error is returned.
    pub fn to_bits(&self) -> Result<IrBits, XlsynthError> {
        xls_value_get_bits(self.ptr)
    }

    pub fn get_element(&self, index: usize) -> Result<IrValue, XlsynthError> {
        xls_value_get_element(self.ptr, index)
    }

    pub fn get_element_count(&self) -> Result<usize, XlsynthError> {
        xls_value_get_element_count(self.ptr)
    }

    pub fn get_elements(&self) -> Result<Vec<IrValue>, XlsynthError> {
        let count = self.get_element_count()?;
        let mut elements = Vec::with_capacity(count);
        for i in 0..count {
            let element = self.get_element(i)?;
            elements.push(element);
        }
        Ok(elements)
    }
}

unsafe impl Send for IrValue {}
unsafe impl Sync for IrValue {}

impl Into<IrValue> for bool {
    fn into(self) -> IrValue {
        IrValue::bool(self)
    }
}

impl Into<IrValue> for u32 {
    fn into(self) -> IrValue {
        IrValue::u32(self)
    }
}

impl Into<IrValue> for u64 {
    fn into(self) -> IrValue {
        IrValue::u64(self)
    }
}

impl std::cmp::PartialEq for IrValue {
    fn eq(&self, other: &Self) -> bool {
        xls_value_eq(self.ptr, other.ptr).expect("eq success")
    }
}

impl std::fmt::Display for IrValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            xls_value_to_string(self.ptr).expect("stringify success")
        )
    }
}

impl std::fmt::Debug for IrValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            xls_value_to_string(self.ptr).expect("stringify success")
        )
    }
}

impl Drop for IrValue {
    fn drop(&mut self) {
        xls_value_free(self.ptr)
    }
}

impl Clone for IrValue {
    fn clone(&self) -> Self {
        let ptr = unsafe { xlsynth_sys::xls_value_clone(self.ptr) };
        Self { ptr }
    }
}

/// Typed wrapper around an `IrBits` value that has a particular
/// compile-time-known bit width and whose type notes the value
/// should be treated as unsigned.
pub struct IrUBits<const BIT_COUNT: usize> {
    #[allow(dead_code)]
    wrapped: IrBits,
}

impl<const BIT_COUNT: usize> IrUBits<BIT_COUNT> {
    pub const SIGNEDNESS: bool = false;

    pub fn new(wrapped: IrBits) -> Result<Self, XlsynthError> {
        if wrapped.get_bit_count() != BIT_COUNT {
            return Err(XlsynthError(format!(
                "Expected {} bits, got {}",
                BIT_COUNT,
                wrapped.get_bit_count()
            )));
        }
        Ok(Self { wrapped })
    }
}

/// Typed wrapper around an `IrBits` value that has a particular
/// compile-time-known bit width and whose type notes the value
/// should be treated as signed.
pub struct IrSBits<const BIT_COUNT: usize> {
    #[allow(dead_code)]
    wrapped: IrBits,
}

impl<const BIT_COUNT: usize> IrSBits<BIT_COUNT> {
    pub const SIGNEDNESS: bool = true;

    pub fn new(wrapped: IrBits) -> Result<Self, XlsynthError> {
        if wrapped.get_bit_count() != BIT_COUNT {
            return Err(XlsynthError(format!(
                "Expected {} bits, got {}",
                BIT_COUNT,
                wrapped.get_bit_count()
            )));
        }
        Ok(Self { wrapped })
    }
}

impl std::cmp::Eq for IrBits {}

impl std::hash::Hash for IrBits {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Since IrBits has a pointer field, we need to hash the actual bits
        // We can use the debug string representation as a stable hash
        self.to_debug_str().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_value_eq() {
        let v1 = IrValue::parse_typed("bits[32]:42").expect("parse success");
        let v2 = IrValue::parse_typed("bits[32]:42").expect("parse success");
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_ir_value_eq_fail() {
        let v1 = IrValue::parse_typed("bits[32]:42").expect("parse success");
        let v2 = IrValue::parse_typed("bits[32]:43").expect("parse success");
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_ir_value_display() {
        let v = IrValue::parse_typed("bits[32]:42").expect("parse success");
        assert_eq!(format!("{}", v), "bits[32]:42");
    }

    #[test]
    fn test_ir_value_debug() {
        let v = IrValue::parse_typed("bits[32]:42").expect("parse success");
        assert_eq!(format!("{:?}", v), "bits[32]:42");
    }

    #[test]
    fn test_ir_value_drop() {
        let v = IrValue::parse_typed("bits[32]:42").expect("parse success");
        drop(v);
    }

    #[test]
    fn test_ir_value_fmt_pref() {
        let v = IrValue::parse_typed("bits[32]:42").expect("parse success");
        assert_eq!(
            v.to_string_fmt(IrFormatPreference::Default)
                .expect("fmt success"),
            "bits[32]:42"
        );
        assert_eq!(
            v.to_string_fmt(IrFormatPreference::Binary)
                .expect("fmt success"),
            "bits[32]:0b10_1010"
        );
        assert_eq!(
            v.to_string_fmt(IrFormatPreference::SignedDecimal)
                .expect("fmt success"),
            "bits[32]:42"
        );
        assert_eq!(
            v.to_string_fmt(IrFormatPreference::UnsignedDecimal)
                .expect("fmt success"),
            "bits[32]:42"
        );
        assert_eq!(
            v.to_string_fmt(IrFormatPreference::Hex)
                .expect("fmt success"),
            "bits[32]:0x2a"
        );
        assert_eq!(
            v.to_string_fmt(IrFormatPreference::PlainBinary)
                .expect("fmt success"),
            "bits[32]:101010"
        );
        assert_eq!(
            v.to_string_fmt(IrFormatPreference::PlainHex)
                .expect("fmt success"),
            "bits[32]:2a"
        );
    }

    #[test]
    fn test_ir_value_from_rust() {
        let v = IrValue::u64(42);

        // Check formatting for default stringification.
        assert_eq!(
            v.to_string_fmt(IrFormatPreference::Default)
                .expect("fmt success"),
            "bits[64]:42"
        );
        // Check the bit count is as we specified.
        assert_eq!(v.bit_count().unwrap(), 64);

        // Check we can't convert a 64-bit value to a bool.
        v.to_bool()
            .expect_err("bool conversion should error for u64");

        let v_i64 = v.to_i64().expect("i64 conversion success");
        assert_eq!(v_i64, 42);

        let f = IrValue::parse_typed("bits[1]:0").expect("parse success");
        assert_eq!(f.to_bool().unwrap(), false);

        let t = IrValue::parse_typed("bits[1]:1").expect("parse success");
        assert_eq!(t.to_bool().unwrap(), true);
    }

    #[test]
    fn test_ir_value_get_bits() {
        let v = IrValue::parse_typed("bits[32]:42").expect("parse success");
        let bits = v.to_bits().expect("to_bits success");

        // Equality comparison.
        let v2 = IrValue::make_ubits(32, 42).expect("make_ubits success");
        assert_eq!(v, v2);

        // Getting at bit values; 42 = 0b101010.
        assert_eq!(bits.get_bit(0).unwrap(), false);
        assert_eq!(bits.get_bit(1).unwrap(), true);
        assert_eq!(bits.get_bit(2).unwrap(), false);
        assert_eq!(bits.get_bit(3).unwrap(), true);
        assert_eq!(bits.get_bit(4).unwrap(), false);
        assert_eq!(bits.get_bit(5).unwrap(), true);
        assert_eq!(bits.get_bit(6).unwrap(), false);
        for i in 7..32 {
            assert_eq!(bits.get_bit(i).unwrap(), false);
        }
        assert!(
            bits.get_bit(32).is_err(),
            "Expected an error for out of bounds index"
        );
        assert!(bits
            .get_bit(32)
            .unwrap_err()
            .to_string()
            .contains("Index 32 out of bounds for bits[32]:0b00000000000000000000000000101010"));

        let debug_fmt = format!("{:?}", bits);
        assert_eq!(debug_fmt, "0b00000000000000000000000000101010");
    }

    #[test]
    fn test_ir_value_make_bits() {
        let zero_u2 = IrValue::make_ubits(2, 0).expect("make_ubits success");
        assert_eq!(
            zero_u2
                .to_string_fmt(IrFormatPreference::Default)
                .expect("fmt success"),
            "bits[2]:0"
        );

        let three_u2 = IrValue::make_ubits(2, 3).expect("make_ubits success");
        assert_eq!(
            three_u2
                .to_string_fmt(IrFormatPreference::Default)
                .expect("fmt success"),
            "bits[2]:3"
        );
    }

    #[test]
    fn test_ir_value_parse_array_value() {
        let text = "[bits[32]:1, bits[32]:2]";
        let v = IrValue::parse_typed(text).expect("parse success");
        assert_eq!(v.to_string(), text);
    }

    #[test]
    fn test_ir_value_parse_2d_array_value() {
        let text = "[[bits[32]:1, bits[32]:2], [bits[32]:3, bits[32]:4], [bits[32]:5, bits[32]:6]]";
        let v = IrValue::parse_typed(text).expect("parse success");
        assert_eq!(v.to_string(), text);
    }

    #[test]
    fn test_ir_bits_add_two_plus_three() {
        let two = IrBits::make_ubits(32, 2).expect("make_ubits success");
        let three = IrBits::make_ubits(32, 3).expect("make_ubits success");
        let sum = two + three;
        assert_eq!(sum.to_string(), "bits[32]:5");
    }

    #[test]
    fn test_ir_bits_umul_two_times_three() {
        let two = IrBits::make_ubits(32, 2).expect("make_ubits success");
        let three = IrBits::make_ubits(32, 3).expect("make_ubits success");
        let product = two.umul(&three);
        assert_eq!(product.to_string(), "bits[64]:6");
    }

    #[test]
    fn test_ir_bits_smul_two_times_neg_three() {
        let two = IrBits::make_ubits(32, 2).expect("make_ubits success");
        let neg_three = IrBits::make_ubits(32, 3)
            .expect("make_ubits success")
            .negate();
        let product = two.smul(&neg_three);
        assert_eq!(product.msb(), true);
        assert_eq!(product.abs().to_string(), "bits[64]:6");
    }

    #[test]
    fn test_ir_bits_width_slice() {
        let bits = IrBits::make_ubits(32, 0x12345678).expect("make_ubits success");
        let slice = bits.width_slice(8, 16);
        assert_eq!(slice.to_hex_string(), "bits[16]:0x3456");
    }

    #[test]
    fn test_ir_bits_shll() {
        let bits = IrBits::make_ubits(32, 0x12345678).expect("make_ubits success");
        let shifted = bits.shll(8);
        assert_eq!(shifted.to_hex_string(), "bits[32]:0x3456_7800");
    }

    #[test]
    fn test_ir_bits_shrl() {
        let bits = IrBits::make_ubits(32, 0x12345678).expect("make_ubits success");
        let shifted = bits.shrl(8);
        assert_eq!(shifted.to_hex_string(), "bits[32]:0x12_3456");
    }

    #[test]
    fn test_ir_bits_shra() {
        let bits = IrBits::make_ubits(32, 0x92345678).expect("make_ubits success");
        let shifted = bits.shra(8);
        assert_eq!(shifted.to_hex_string(), "bits[32]:0xff92_3456");
    }

    #[test]
    fn test_ir_bits_and() {
        let lhs = IrBits::make_ubits(32, 0x5a5a5a5a).expect("make_ubits success");
        let rhs = IrBits::make_ubits(32, 0xa5a5a5a5).expect("make_ubits success");
        assert_eq!(lhs.and(&rhs).to_hex_string(), "bits[32]:0x0");
        assert_eq!(lhs.and(&rhs.not()).to_hex_string(), "bits[32]:0x5a5a_5a5a");
    }

    #[test]
    fn test_ir_bits_or() {
        let lhs = IrBits::make_ubits(32, 0x5a5a5a5a).expect("make_ubits success");
        let rhs = IrBits::make_ubits(32, 0xa5a5a5a5).expect("make_ubits success");
        assert_eq!(lhs.or(&rhs).to_hex_string(), "bits[32]:0xffff_ffff");
        assert_eq!(lhs.or(&rhs.not()).to_hex_string(), "bits[32]:0x5a5a_5a5a");
    }

    #[test]
    fn test_ir_bits_xor() {
        let lhs = IrBits::make_ubits(32, 0x5a5a5a5a).expect("make_ubits success");
        let rhs = IrBits::make_ubits(32, 0xa5a5a5a5).expect("make_ubits success");
        assert_eq!(lhs.xor(&rhs).to_hex_string(), "bits[32]:0xffff_ffff");
        assert_eq!(lhs.xor(&rhs.not()).to_hex_string(), "bits[32]:0x0");
    }

    #[test]
    fn test_make_tuple_and_get_elements() {
        let _ = env_logger::builder().is_test(true).try_init();
        let b1_v0 = IrValue::make_ubits(1, 0).expect("make_ubits success");
        let b2_v1 = IrValue::make_ubits(2, 1).expect("make_ubits success");
        let b3_v2 = IrValue::make_ubits(3, 2).expect("make_ubits success");
        let tuple = IrValue::make_tuple(&[b1_v0.clone(), b2_v1.clone(), b3_v2.clone()]);
        let elements = tuple.get_elements().expect("get_elements success");
        assert_eq!(elements.len(), 3);
        assert_eq!(elements[0].to_string(), "bits[1]:0");
        assert_eq!(elements[0], b1_v0);
        assert_eq!(elements[1].to_string(), "bits[2]:1");
        assert_eq!(elements[1], b2_v1);
        assert_eq!(elements[2].to_string(), "bits[3]:2");
        assert_eq!(elements[2], b3_v2);
    }

    #[test]
    fn test_make_ir_value_bits_that_does_not_fit() {
        let result = IrValue::make_ubits(1, 2);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("0x2 requires 2 bits to fit in an unsigned datatype, but attempting to fit in 1 bit"), "got: {}", error);

        let result = IrValue::make_sbits(1, -2);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("0xfffffffffffffffe requires 2 bits to fit in an signed datatype, but attempting to fit in 1 bit"), "got: {}", error);
    }

    #[test]
    fn test_make_ir_value_array() {
        let b2_v0 = IrValue::make_ubits(2, 0).expect("make_ubits success");
        let b2_v1 = IrValue::make_ubits(2, 1).expect("make_ubits success");
        let b2_v2 = IrValue::make_ubits(2, 2).expect("make_ubits success");
        let b2_v3 = IrValue::make_ubits(2, 3).expect("make_ubits success");
        let array = IrValue::make_array(&[b2_v0, b2_v1, b2_v2, b2_v3]).expect("make_array success");
        assert_eq!(
            array.to_string(),
            "[bits[2]:0, bits[2]:1, bits[2]:2, bits[2]:3]"
        );
    }

    #[test]
    fn test_make_ir_value_empty_array() {
        IrValue::make_array(&[]).expect_err("make_array should fail for empty array");
    }

    #[test]
    fn test_make_ir_value_array_with_mixed_types() {
        let b2_v0 = IrValue::make_ubits(2, 0).expect("make_ubits success");
        let b3_v1 = IrValue::make_ubits(3, 1).expect("make_ubits success");
        let result = IrValue::make_array(&[b2_v0, b3_v1]);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("SameTypeAs"));
    }
}
