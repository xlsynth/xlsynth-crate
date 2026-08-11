// SPDX-License-Identifier: Apache-2.0

//! Width-aware parsing and formatting of Verilog integer literals.

use num_bigint::{BigInt, BigUint, Sign};

use crate::{LiteralFormat, VastError};

/// Bounds both the integer allocation and the longest padded literal output.
const MAX_LITERAL_BIT_WIDTH: usize = 1 << 20;

/// Parsed unsigned bits; negative input is converted to two's complement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Literal {
    pub(crate) width: usize,
    pub(crate) value: BigUint,
}

impl Literal {
    /// Parses the XLS typed-value spelling `bits[N]:value` without native code.
    pub(crate) fn parse(text: &str) -> Result<Self, VastError> {
        let (width_text, value_text) = text
            .strip_prefix("bits[")
            .and_then(|rest| rest.split_once("]:"))
            .ok_or_else(|| VastError(format!("expected a typed bits literal, got `{text}`")))?;
        if width_text.is_empty() || !width_text.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(VastError(format!("invalid bit width in literal `{text}`")));
        }
        let width = width_text
            .parse::<usize>()
            .map_err(|error| VastError(format!("invalid bit width `{width_text}`: {error}")))?;
        if width == 0 {
            return Err(VastError(
                "literal bit width must be greater than zero".into(),
            ));
        }
        if width > MAX_LITERAL_BIT_WIDTH {
            return Err(VastError(format!(
                "literal bit width {width} exceeds maximum supported width {MAX_LITERAL_BIT_WIDTH}"
            )));
        }

        let negative = value_text.starts_with('-');
        let unsigned_text = value_text
            .strip_prefix('-')
            .or_else(|| value_text.strip_prefix('+'))
            .unwrap_or(value_text);
        let (radix, digits) = if let Some(digits) = unsigned_text
            .strip_prefix("0x")
            .or_else(|| unsigned_text.strip_prefix("0X"))
        {
            (16, digits)
        } else if let Some(digits) = unsigned_text
            .strip_prefix("0b")
            .or_else(|| unsigned_text.strip_prefix("0B"))
        {
            (2, digits)
        } else if let Some(digits) = unsigned_text
            .strip_prefix("0o")
            .or_else(|| unsigned_text.strip_prefix("0O"))
        {
            (8, digits)
        } else {
            (10, unsigned_text)
        };
        let digits: String = digits
            .chars()
            .filter(|character| *character != '_')
            .collect();
        let magnitude = BigUint::parse_bytes(digits.as_bytes(), radix)
            .ok_or_else(|| VastError(format!("invalid digits in literal `{text}`")))?;

        let value = if negative {
            if magnitude == BigUint::from(0u8) {
                magnitude
            } else if magnitude.bits() > width as u64 + 1 {
                return Err(VastError(format!(
                    "literal `{text}` does not fit in {width} bits"
                )));
            } else {
                let limit = BigUint::from(1u8) << width;
                if magnitude > limit {
                    return Err(VastError(format!(
                        "literal `{text}` does not fit in {width} bits"
                    )));
                }
                limit - magnitude
            }
        } else {
            if magnitude.bits() > width as u64 {
                return Err(VastError(format!(
                    "literal `{text}` does not fit in {width} bits"
                )));
            }
            magnitude
        };
        Ok(Self { width, value })
    }

    /// Emits a literal with the same width and grouping conventions as XLS
    /// VAST.
    pub(crate) fn format(&self, format: LiteralFormat) -> String {
        if self.width > 1024 && self.value == BigUint::from(0u8) {
            let specifier = match format {
                LiteralFormat::Binary | LiteralFormat::ZeroPaddedBinary => "b",
                LiteralFormat::Hex | LiteralFormat::ZeroPaddedHex => "h",
                LiteralFormat::Default
                | LiteralFormat::PlainBinary
                | LiteralFormat::PlainHex
                | LiteralFormat::SignedDecimal
                | LiteralFormat::UnsignedDecimal => "d",
            };
            return format!("{}'{}0", self.width, specifier);
        }

        match format {
            LiteralFormat::Default if self.width > 32 => {
                format!("{}'d{}", self.width, self.value.to_str_radix(10))
            }
            LiteralFormat::Default => self.value.to_str_radix(10),
            LiteralFormat::UnsignedDecimal => {
                format!("{}'d{}", self.width, self.value.to_str_radix(10))
            }
            LiteralFormat::SignedDecimal => {
                let signed = if self.width > 0 && self.value.bit((self.width - 1) as u64) {
                    BigInt::from_biguint(Sign::Plus, self.value.clone())
                        - (BigInt::from(1u8) << self.width)
                } else {
                    BigInt::from_biguint(Sign::Plus, self.value.clone())
                };
                let digits = signed.to_str_radix(10);
                if let Some(magnitude) = digits.strip_prefix('-') {
                    format!("-{}'sd{magnitude}", self.width)
                } else {
                    format!("{}'sd{digits}", self.width)
                }
            }
            LiteralFormat::Binary | LiteralFormat::ZeroPaddedBinary => {
                let digits = pad_digits(self.value.to_str_radix(2), self.width.max(1));
                format!("{}'b{}", self.width, group_digits(&digits))
            }
            LiteralFormat::PlainBinary => {
                format!("'b{}", self.value.to_str_radix(2))
            }
            LiteralFormat::Hex | LiteralFormat::ZeroPaddedHex => {
                let digit_count = self.width.div_ceil(4).max(1);
                let digits = pad_digits(self.value.to_str_radix(16), digit_count);
                format!("{}'h{}", self.width, group_digits(&digits))
            }
            LiteralFormat::PlainHex => {
                format!("'h{}", group_digits(&self.value.to_str_radix(16)))
            }
        }
    }
}

/// Left-pads radix digits to their declared bit width.
fn pad_digits(digits: String, width: usize) -> String {
    if digits.len() >= width {
        digits
    } else {
        format!("{}{}", "0".repeat(width - digits.len()), digits)
    }
}

/// Inserts separators between groups of four radix digits, from the right.
fn group_digits(digits: &str) -> String {
    let mut grouped = String::with_capacity(digits.len() + digits.len().saturating_sub(1) / 4);
    for (index, digit) in digits.char_indices() {
        if index != 0 && (digits.len() - index).is_multiple_of(4) {
            grouped.push('_');
        }
        grouped.push(digit);
    }
    grouped
}

#[cfg(test)]
mod tests {
    use super::{Literal, MAX_LITERAL_BIT_WIDTH};
    use crate::LiteralFormat;

    #[test]
    fn wide_hex_literals_keep_leading_zeroes_and_groups() {
        let literal = Literal::parse("bits[128]:0xFFEEDDCCBBAA99887766554433221100")
            .expect("valid 128-bit value");
        assert_eq!(
            literal.format(LiteralFormat::Hex),
            "128'hffee_ddcc_bbaa_9988_7766_5544_3322_1100"
        );
    }

    #[test]
    fn plain_hex_literals_keep_grouping_without_a_width() {
        let literal = Literal::parse("bits[32]:0x10000").expect("valid literal");
        assert_eq!(literal.format(LiteralFormat::PlainHex), "'h1_0000");
    }

    #[test]
    fn binary_literals_are_padded_and_grouped() {
        let literal = Literal::parse("bits[9]:0b11").expect("valid literal");
        assert_eq!(literal.format(LiteralFormat::Binary), "9'b0_0000_0011");
        assert_eq!(literal.format(LiteralFormat::PlainBinary), "'b11");
    }

    #[test]
    fn overly_wide_values_are_rejected() {
        assert!(Literal::parse("bits[2]:4").is_err());
    }

    #[test]
    fn zero_and_excessive_literal_widths_are_rejected_before_allocation() {
        assert_eq!(
            Literal::parse("bits[0]:0")
                .expect_err("zero-width Verilog literals are invalid")
                .to_string(),
            "literal bit width must be greater than zero"
        );

        let excessive_width = MAX_LITERAL_BIT_WIDTH + 1;
        assert_eq!(
            Literal::parse(&format!("bits[{excessive_width}]:0"))
                .expect_err("literal allocations must have a practical bound")
                .to_string(),
            format!(
                "literal bit width {excessive_width} exceeds maximum supported width \
                 {MAX_LITERAL_BIT_WIDTH}"
            )
        );

        let maximum_machine_width = usize::MAX;
        assert_eq!(
            Literal::parse(&format!("bits[{maximum_machine_width}]:1"))
                .expect_err("machine-sized widths must not reach a BigUint allocation")
                .to_string(),
            format!(
                "literal bit width {maximum_machine_width} exceeds maximum supported width \
                 {MAX_LITERAL_BIT_WIDTH}"
            )
        );
    }

    #[test]
    fn maximum_width_positive_literals_do_not_require_a_full_width_integer() {
        let zero = Literal::parse(&format!("bits[{MAX_LITERAL_BIT_WIDTH}]:0"))
            .expect("the supported maximum width is valid");
        assert_eq!(
            zero.format(LiteralFormat::Hex),
            format!("{MAX_LITERAL_BIT_WIDTH}'h0")
        );

        let one = Literal::parse(&format!("bits[{MAX_LITERAL_BIT_WIDTH}]:1"))
            .expect("a narrow positive value fits in the maximum width");
        assert_eq!(
            one.format(LiteralFormat::UnsignedDecimal),
            format!("{MAX_LITERAL_BIT_WIDTH}'d1")
        );
    }

    #[test]
    fn negative_range_boundaries_preserve_twos_complement_behavior() {
        let minimum = Literal::parse("bits[8]:-256")
            .expect("the prior parser accepts the negative full-range boundary");
        assert_eq!(minimum.format(LiteralFormat::Hex), "8'h00");
        assert!(Literal::parse("bits[8]:-257").is_err());

        let negative_zero = Literal::parse(&format!("bits[{MAX_LITERAL_BIT_WIDTH}]:-0"))
            .expect("negative zero does not require a full-width two's-complement value");
        assert_eq!(
            negative_zero.format(LiteralFormat::UnsignedDecimal),
            format!("{MAX_LITERAL_BIT_WIDTH}'d0")
        );
    }

    #[test]
    fn huge_zero_literals_are_compact() {
        let literal = Literal::parse("bits[4096]:0").expect("valid literal");
        assert_eq!(literal.format(LiteralFormat::Hex), "4096'h0");
        assert_eq!(literal.format(LiteralFormat::Binary), "4096'b0");
        assert_eq!(literal.format(LiteralFormat::PlainBinary), "4096'd0");
        assert_eq!(literal.format(LiteralFormat::PlainHex), "4096'd0");
        assert_eq!(literal.format(LiteralFormat::SignedDecimal), "4096'd0");
    }

    #[test]
    fn wide_default_literals_preserve_their_declared_width() {
        let narrow = Literal::parse("bits[32]:42").expect("valid narrow literal");
        let wide = Literal::parse("bits[96]:42").expect("valid wide literal");

        assert_eq!(narrow.format(LiteralFormat::Default), "42");
        assert_eq!(wide.format(LiteralFormat::Default), "96'd42");
    }

    #[test]
    fn negative_literals_use_twos_complement() {
        let literal = Literal::parse("bits[8]:-1").expect("valid negative literal");
        assert_eq!(literal.format(LiteralFormat::Hex), "8'hff");
        assert_eq!(literal.format(LiteralFormat::SignedDecimal), "-8'sd1");
    }

    #[test]
    fn signed_decimal_literals_place_negative_sign_before_the_width() {
        let negative = Literal::parse("bits[8]:0xff").expect("valid negative bit pattern");
        let minimum = Literal::parse("bits[8]:0x80").expect("valid minimum bit pattern");
        let positive = Literal::parse("bits[8]:0x7f").expect("valid positive bit pattern");

        assert_eq!(negative.format(LiteralFormat::SignedDecimal), "-8'sd1");
        assert_eq!(minimum.format(LiteralFormat::SignedDecimal), "-8'sd128");
        assert_eq!(positive.format(LiteralFormat::SignedDecimal), "8'sd127");
    }
}
