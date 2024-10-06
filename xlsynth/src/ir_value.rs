// SPDX-License-Identifier: Apache-2.0

use xlsynth_sys::{CIrBits, CIrValue};

use crate::{
    xls_format_preference_from_string, xls_parse_typed_value, xls_value_eq, xls_value_free,
    xls_value_get_bits, xls_value_to_string, xls_value_to_string_format_preference,
    xlsynth_error::XlsynthError,
};

pub struct IrBits {
    #[allow(dead_code)]
    pub(crate) ptr: *mut CIrBits,
}

pub enum IrFormatPreference {
    Default,
    Binary,
    SignedDecimal,
    UnsignedDecimal,
    Hex,
    PlainBinary,
    PlainHex,
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
            IrFormatPreference::PlainHex => "plain_hex",
        }
    }
}

pub struct IrValue {
    pub(crate) ptr: *mut CIrValue,
}

impl IrValue {
    pub fn parse_typed(s: &str) -> Result<Self, XlsynthError> {
        xls_parse_typed_value(s)
    }

    pub fn u64(value: u64) -> Result<Self, XlsynthError> {
        // TODO(cdleary): 2024-06-23 Expose a more efficient API for this.
        Self::parse_typed(std::format!("bits[64]:{}", value).as_str())
    }

    pub fn make_bits(bit_count: usize, value: u64) -> Result<Self, XlsynthError> {
        // TODO(cdleary): 2024-10-06 Expose a more efficient API for this.
        Self::parse_typed(std::format!("bits[{}]:{}", bit_count, value).as_str())
    }

    pub fn bit_count(&self) -> usize {
        // TODO(cdleary): 2024-06-23 Expose a more efficient API for this.
        let s = self
            .to_string_fmt(IrFormatPreference::Default)
            .expect("fmt success");
        // Look at the decimal value in the formatted result; e.g. `bits[7]:42` => `7`
        let parts: Vec<&str> = s
            .split(':')
            .nth(0)
            .expect("split success")
            .split('[')
            .nth(1)
            .expect("split success")
            .split(']')
            .collect();
        return parts[0].parse::<usize>().expect("parse success");
    }

    pub fn to_string_fmt(&self, format: IrFormatPreference) -> Result<String, XlsynthError> {
        let fmt_pref: xlsynth_sys::XlsFormatPreference =
            xls_format_preference_from_string(format.to_string())?;
        xls_value_to_string_format_preference(self.ptr, fmt_pref)
    }

    pub fn to_bool(&self) -> Result<bool, XlsynthError> {
        if self.bit_count() != 1 {
            return Err(XlsynthError(format!(
                "IrValue {} is not single-bit; must be bits[1] to convert to bool",
                self.to_string()
            )));
        }
        // TODO(cdleary): 2024-06-23 Expose a more efficient API for this.
        let s = self.to_string_fmt(IrFormatPreference::PlainBinary)?;
        let v = s.split(':').nth(1).expect("split success");
        match v {
            "0" => return Ok(false),
            "1" => return Ok(true),
            _ => panic!("Unexpected stringified value for single-bit IrValue: {}", s),
        }
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

    /// Attempts to extract the bits contents underlying this value.
    ///
    /// If this value is not a bits type, an error is returned.
    pub fn to_bits(&self) -> Result<IrBits, XlsynthError> {
        xls_value_get_bits(self.ptr)
    }
}

unsafe impl Send for IrValue {}
unsafe impl Sync for IrValue {}

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
        xls_value_free(self.ptr).expect("dealloc success");
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
        let v = IrValue::u64(42).expect("u64 success");

        // Check formatting for default stringification.
        assert_eq!(
            v.to_string_fmt(IrFormatPreference::Default)
                .expect("fmt success"),
            "bits[64]:42"
        );
        // Check the bit count is as we specified.
        assert_eq!(v.bit_count(), 64);

        // Check we can't convert a 64-bit value to a bool.
        v.to_bool().expect_err("bool conversion fail for u64");

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
        let _bits = v.to_bits().expect("to_bits success");
        // TODO(cdleary): 2024-09-15 No APIs exposed to do anything directly
        // with bits yet.
    }

    #[test]
    fn test_ir_value_make_bits() {
        let zero_u2 = IrValue::make_bits(2, 0).expect("make_bits success");
        assert_eq!(
            zero_u2
                .to_string_fmt(IrFormatPreference::Default)
                .expect("fmt success"),
            "bits[2]:0"
        );

        let three_u2 = IrValue::make_bits(2, 3).expect("make_bits success");
        assert_eq!(
            three_u2
                .to_string_fmt(IrFormatPreference::Default)
                .expect("fmt success"),
            "bits[2]:3"
        );
    }
}
