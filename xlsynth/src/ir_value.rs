// SPDX-License-Identifier: Apache-2.0

use xlsynth_sys::{CIrBits, CIrValue};

use crate::{
    lib_support::{
        xls_bits_make_sbits, xls_bits_make_ubits, xls_bits_to_bytes, xls_bits_to_debug_str,
        xls_bits_to_int64, xls_bits_to_string, xls_bits_to_uint64,
        xls_format_preference_from_string, xls_value_eq, xls_value_free, xls_value_get_bits,
        xls_value_get_element, xls_value_get_element_count, xls_value_make_array,
        xls_value_make_sbits, xls_value_make_token, xls_value_make_tuple, xls_value_make_ubits,
        xls_value_to_string, xls_value_to_string_format_preference,
    },
    xls_parse_typed_value,
    xlsynth_error::XlsynthError,
};

pub struct IrBits {
    pub(crate) ptr: *mut CIrBits,
}

impl IrBits {
    fn from_raw(ptr: *mut CIrBits) -> Self {
        Self { ptr }
    }

    fn apply_binary_op(
        &self,
        rhs: &IrBits,
        op: unsafe extern "C" fn(*const CIrBits, *const CIrBits) -> *mut CIrBits,
    ) -> Self {
        let result = unsafe { op(self.ptr, rhs.ptr) };
        Self::from_raw(result)
    }

    fn apply_unary_op(&self, op: unsafe extern "C" fn(*const CIrBits) -> *mut CIrBits) -> Self {
        let result = unsafe { op(self.ptr) };
        Self::from_raw(result)
    }

    /// Converts a `&[bool]` slice into an IR `Bits` value.
    ///
    /// ```
    /// use xlsynth::ir_value::IrFormatPreference;
    /// use xlsynth::IrBits;
    ///
    /// let bools = vec![true, false, true, false]; // LSB is bools[0]
    /// let ir_bits: IrBits = IrBits::from_lsb_is_0(&bools);
    /// assert_eq!(ir_bits.to_string_fmt(IrFormatPreference::Binary, false), "0b101");
    /// assert_eq!(ir_bits.get_bit_count(), 4);
    /// assert_eq!(ir_bits.get_bit(0).unwrap(), true); // LSB
    /// assert_eq!(ir_bits.get_bit(1).unwrap(), false);
    /// assert_eq!(ir_bits.get_bit(2).unwrap(), true);
    /// assert_eq!(ir_bits.get_bit(3).unwrap(), false); // MSB
    /// ```
    pub fn from_lsb_is_0(bits: &[bool]) -> Self {
        if bits.is_empty() {
            return IrBits::make_ubits(0, 0).unwrap();
        }
        let mut s: String = format!("bits[{}]:0b", bits.len());
        for b in bits.iter().rev() {
            s.push(if *b { '1' } else { '0' });
        }
        IrValue::parse_typed(&s).unwrap().to_bits().unwrap()
    }

    /// Turns a boolean slice into an IR `Bits` value under the assumption that
    /// index 0 in the slice is the most significant bit (MSb).
    pub fn from_msb_is_0(bits: &[bool]) -> Self {
        if bits.is_empty() {
            return IrBits::make_ubits(0, 0).unwrap();
        }
        let mut s: String = format!("bits[{}]:0b", bits.len());
        for b in bits {
            s.push(if *b { '1' } else { '0' });
        }
        IrValue::parse_typed(&s).unwrap().to_bits().unwrap()
    }

    pub fn make_ubits(bit_count: usize, value: u64) -> Result<Self, XlsynthError> {
        xls_bits_make_ubits(bit_count, value)
    }

    pub fn make_sbits(bit_count: usize, value: i64) -> Result<Self, XlsynthError> {
        xls_bits_make_sbits(bit_count, value)
    }

    pub fn u32(value: u32) -> Self {
        // Unwrap should be ok since the u32 always fits.
        Self::make_ubits(32, value as u64).unwrap()
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

    pub fn equals(&self, rhs: &IrBits) -> bool {
        unsafe { xlsynth_sys::xls_bits_eq(self.ptr, rhs.ptr) }
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

    pub fn to_u64(&self) -> Result<u64, XlsynthError> {
        xls_bits_to_uint64(self.ptr)
    }

    pub fn to_i64(&self) -> Result<i64, XlsynthError> {
        xls_bits_to_int64(self.ptr)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, XlsynthError> {
        xls_bits_to_bytes(self.ptr)
    }

    pub fn add(&self, rhs: &IrBits) -> IrBits {
        self.apply_binary_op(rhs, xlsynth_sys::xls_bits_add)
    }

    pub fn sub(&self, rhs: &IrBits) -> IrBits {
        self.apply_binary_op(rhs, xlsynth_sys::xls_bits_sub)
    }

    pub fn umul(&self, rhs: &IrBits) -> IrBits {
        self.apply_binary_op(rhs, xlsynth_sys::xls_bits_umul)
    }

    pub fn smul(&self, rhs: &IrBits) -> IrBits {
        self.apply_binary_op(rhs, xlsynth_sys::xls_bits_smul)
    }

    pub fn negate(&self) -> IrBits {
        self.apply_unary_op(xlsynth_sys::xls_bits_negate)
    }

    pub fn abs(&self) -> IrBits {
        self.apply_unary_op(xlsynth_sys::xls_bits_abs)
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
        self.apply_unary_op(xlsynth_sys::xls_bits_not)
    }

    pub fn and(&self, rhs: &IrBits) -> IrBits {
        self.apply_binary_op(rhs, xlsynth_sys::xls_bits_and)
    }

    pub fn or(&self, rhs: &IrBits) -> IrBits {
        self.apply_binary_op(rhs, xlsynth_sys::xls_bits_or)
    }

    pub fn xor(&self, rhs: &IrBits) -> IrBits {
        self.apply_binary_op(rhs, xlsynth_sys::xls_bits_xor)
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
        self.equals(other)
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

// SAFETY: `IrBits` is a thin wrapper around a raw pointer to an immutable
// FFI object managed by the underlying XLS library. The pointer value itself
// may be freely transferred across thread boundaries so long as the
// underlying object is not concurrently mutated through other avenues. The
// XLS C API does not provide any mutation operations that would violate this
// assumption for the usage patterns within xlsynth, so it is sound to mark
// `IrBits` as `Send` and `Sync`.

unsafe impl Send for IrBits {}
unsafe impl Sync for IrBits {}

impl std::ops::Add for IrBits {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        IrBits::add(&self, &rhs)
    }
}

impl std::ops::Sub for IrBits {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        IrBits::sub(&self, &rhs)
    }
}

impl std::ops::BitAnd for IrBits {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        IrBits::and(&self, &rhs)
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
    pub fn make_token() -> Self {
        xls_value_make_token()
    }

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
        xls_value_make_ubits(value, 64).unwrap()
    }

    pub fn make_ubits(bit_count: usize, value: u64) -> Result<Self, XlsynthError> {
        xls_value_make_ubits(value, bit_count)
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
        // Use a regex for now to strip out all `bits[N]:` prefixes.
        let re = regex::Regex::new(r"bits\[[0-9]+\]:").unwrap();
        Ok(re.replace_all(&s, "").to_string())
    }

    pub fn to_bool(&self) -> Result<bool, XlsynthError> {
        let bits = self.to_bits()?;
        if bits.get_bit_count() != 1 {
            return Err(XlsynthError(format!(
                "IrValue {self} is not single-bit; must be bits[1] to convert to bool"
            )));
        }
        bits.get_bit(0)
    }

    pub fn to_i64(&self) -> Result<i64, XlsynthError> {
        let bits = self.to_bits()?;
        let width = bits.get_bit_count();
        if width > 64 {
            return Err(XlsynthError(format!(
                "IrValue::to_i64(): width {width} exceeds 64 bits"
            )));
        }
        let unsigned = self.to_u64()?;
        let shift = 64 - width;
        Ok(((unsigned << shift) as i64) >> shift)
    }

    pub fn to_u64(&self) -> Result<u64, XlsynthError> {
        let bits = self.to_bits()?;
        let width = bits.get_bit_count();
        if width > 64 {
            return Err(XlsynthError(format!(
                "IrValue::to_u64(): width {width} exceeds 64 bits"
            )));
        }
        let mut val: u64 = 0;
        for idx in (0..width).rev() {
            val = (val << 1) | bits.get_bit(idx)? as u64;
        }
        Ok(val)
    }

    pub fn to_u32(&self) -> Result<u32, XlsynthError> {
        let val = self.to_u64()?;
        if val > u32::MAX as u64 {
            return Err(XlsynthError(format!(
                "IrValue::to_u32() value {val} does not fit in u32"
            )));
        }
        Ok(val as u32)
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

impl From<bool> for IrValue {
    fn from(val: bool) -> Self {
        IrValue::bool(val)
    }
}
impl From<u32> for IrValue {
    fn from(val: u32) -> Self {
        IrValue::u32(val)
    }
}
impl From<u64> for IrValue {
    fn from(val: u64) -> Self {
        IrValue::u64(val)
    }
}

impl std::cmp::PartialEq for IrValue {
    fn eq(&self, other: &Self) -> bool {
        xls_value_eq(self.ptr, other.ptr).expect("eq success")
    }
}

// `Eq` is safe because XLS values are immutable and `xls_value_eq` defines a
// proper equivalence relation.
impl Eq for IrValue {}

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
        assert_eq!(format!("{v}"), "bits[32]:42");
    }

    #[test]
    fn test_ir_value_debug() {
        let v = IrValue::parse_typed("bits[32]:42").expect("parse success");
        assert_eq!(format!("{v:?}"), "bits[32]:42");
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
        assert!(!f.to_bool().unwrap());

        let t = IrValue::parse_typed("bits[1]:1").expect("parse success");
        assert!(t.to_bool().unwrap());
    }

    #[test]
    fn test_ir_value_get_bits() {
        let v = IrValue::parse_typed("bits[32]:42").expect("parse success");
        let bits = v.to_bits().expect("to_bits success");

        // Equality comparison.
        let v2 = IrValue::u32(42);
        assert_eq!(v, v2);

        // Getting at bit values; 42 = 0b101010.
        assert!(!bits.get_bit(0).unwrap());
        assert!(bits.get_bit(1).unwrap());
        assert!(!bits.get_bit(2).unwrap());
        assert!(bits.get_bit(3).unwrap());
        assert!(!bits.get_bit(4).unwrap());
        assert!(bits.get_bit(5).unwrap());
        assert!(!bits.get_bit(6).unwrap());
        for i in 7..32 {
            assert!(!bits.get_bit(i).unwrap());
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

        let debug_fmt = format!("{bits:?}");
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
    fn test_ir_bits_add() {
        let two = IrBits::u32(2);
        let three = IrBits::u32(3);
        let sum = two.add(&three);
        assert_eq!(sum.to_string(), "bits[32]:5");

        // Ensure the originals were not consumed by the add method.
        assert_eq!(two.to_string(), "bits[32]:2");
        assert_eq!(three.to_string(), "bits[32]:3");

        // The `+` operator should behave equivalently.
        let sum_op = two.clone() + three.clone();
        assert_eq!(sum, sum_op);
    }

    #[test]
    fn test_ir_bits_sub() {
        let five = IrBits::u32(5);
        let two = IrBits::u32(2);
        let diff = five.sub(&two);
        assert_eq!(diff.to_string(), "bits[32]:3");

        // Ensure the originals were not consumed by the sub method.
        assert_eq!(five.to_string(), "bits[32]:5");
        assert_eq!(two.to_string(), "bits[32]:2");

        // The `-` operator should behave equivalently.
        let diff_op = five.clone() - two.clone();
        assert_eq!(diff, diff_op);
    }

    #[test]
    fn test_ir_bits_eq() {
        let a = IrBits::u32(0x12345678);
        let b = IrBits::u32(0x12345678);
        let c = IrBits::u32(0x87654321);
        assert!(a.equals(&b));
        assert!(!a.equals(&c));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_ir_bits_umul_two_times_three() {
        let two = IrBits::u32(2);
        let three = IrBits::u32(3);
        let product = two.umul(&three);
        assert_eq!(product.to_string(), "bits[64]:6");
    }

    #[test]
    fn test_ir_bits_smul_two_times_neg_three() {
        let two = IrBits::u32(2);
        let neg_three = IrBits::u32(3).negate();
        let product = two.smul(&neg_three);
        assert!(product.msb());
        assert_eq!(product.abs().to_string(), "bits[64]:6");
    }

    #[test]
    fn test_ir_bits_negate() {
        let three = IrBits::u32(3);
        let neg = three.negate();
        assert_eq!(neg.to_hex_string(), "bits[32]:0xffff_fffd");
    }

    #[test]
    fn test_ir_bits_abs() {
        let neg_three = IrBits::u32(3).negate();
        let abs = neg_three.abs();
        assert_eq!(abs.to_string(), "bits[32]:3");
    }

    #[test]
    fn test_ir_bits_width_slice() {
        let bits = IrBits::u32(0x12345678);
        let slice = bits.width_slice(8, 16);
        assert_eq!(slice.to_hex_string(), "bits[16]:0x3456");
    }

    #[test]
    fn test_ir_bits_shll() {
        let bits = IrBits::u32(0x12345678);
        let shifted = bits.shll(8);
        assert_eq!(shifted.to_hex_string(), "bits[32]:0x3456_7800");
    }

    #[test]
    fn test_ir_bits_shrl() {
        let bits = IrBits::u32(0x12345678);
        let shifted = bits.shrl(8);
        assert_eq!(shifted.to_hex_string(), "bits[32]:0x12_3456");
    }

    #[test]
    fn test_ir_bits_shra() {
        let bits = IrBits::u32(0x92345678);
        let shifted = bits.shra(8);
        assert_eq!(shifted.to_hex_string(), "bits[32]:0xff92_3456");
    }

    #[test]
    fn test_ir_bits_u32() {
        let bits = IrBits::u32(0x12345678);
        assert_eq!(bits.to_hex_string(), "bits[32]:0x1234_5678");
    }

    #[test]
    fn test_ir_bits_get_bit() {
        let bits = IrBits::u32(0b1010);
        assert!(!bits.get_bit(0).unwrap());
        assert!(bits.get_bit(1).unwrap());
        assert!(bits.get_bit(3).unwrap());
        assert!(bits.get_bit(32).is_err());
    }

    #[test]
    fn test_ir_bits_to_u64() {
        let bits = IrBits::u32(0x12345678);
        assert_eq!(bits.to_u64().unwrap(), 0x12345678);
    }

    #[test]
    fn test_ir_bits_to_i64() {
        let bits = IrBits::make_sbits(8, -5).expect("make_sbits success");
        assert_eq!(bits.to_i64().unwrap(), -5);
    }

    #[test]
    fn test_ir_bits_to_bytes() {
        let bits = IrBits::u32(0x12345678);
        assert_eq!(bits.to_bytes().unwrap(), vec![0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_ir_bits_and() {
        let lhs = IrBits::u32(0x5a5a5a5a);
        let rhs = IrBits::u32(0xa5a5a5a5);
        assert_eq!(lhs.and(&rhs).to_hex_string(), "bits[32]:0x0");
        assert_eq!(lhs.and(&rhs.not()).to_hex_string(), "bits[32]:0x5a5a_5a5a");
    }

    #[test]
    fn test_ir_bits_or() {
        let lhs = IrBits::u32(0x5a5a5a5a);
        let rhs = IrBits::u32(0xa5a5a5a5);
        assert_eq!(lhs.or(&rhs).to_hex_string(), "bits[32]:0xffff_ffff");
        assert_eq!(lhs.or(&rhs.not()).to_hex_string(), "bits[32]:0x5a5a_5a5a");
    }

    #[test]
    fn test_ir_bits_xor() {
        let lhs = IrBits::u32(0x5a5a5a5a);
        let rhs = IrBits::u32(0xa5a5a5a5);
        assert_eq!(lhs.xor(&rhs).to_hex_string(), "bits[32]:0xffff_ffff");
        assert_eq!(lhs.xor(&rhs.not()).to_hex_string(), "bits[32]:0x0");
    }

    #[test]
    fn test_ir_bits_not() {
        let zero = IrBits::u32(0);
        assert_eq!(zero.not().to_hex_string(), "bits[32]:0xffff_ffff");
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
