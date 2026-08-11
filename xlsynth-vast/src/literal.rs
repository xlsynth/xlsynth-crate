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

    /// Rejects unsized values whose magnitude is not portable across tools.
    pub(crate) fn validate_format(&self, format: LiteralFormat) -> Result<(), VastError> {
        match format {
            LiteralFormat::UnsizedDecimal if self.value > BigUint::from(i32::MAX as u32) => {
                Err(VastError(format!(
                    "unsized decimal literal magnitude {} exceeds maximum {}",
                    self.value,
                    i32::MAX
                )))
            }
            LiteralFormat::UnsizedBinary | LiteralFormat::UnsizedHex
                if self.value > BigUint::from(u32::MAX) =>
            {
                let radix = if matches!(format, LiteralFormat::UnsizedBinary) {
                    "binary"
                } else {
                    "hexadecimal"
                };
                Err(VastError(format!(
                    "unsized {radix} literal magnitude {} exceeds maximum {}",
                    self.value,
                    u32::MAX
                )))
            }
            _ => Ok(()),
        }
    }

    /// Emits a literal with the same width and grouping conventions as XLS
    /// VAST.
    pub(crate) fn format(&self, format: LiteralFormat) -> String {
        if self.width > 1024
            && self.value == BigUint::from(0u8)
            && matches!(format, LiteralFormat::Binary | LiteralFormat::Hex)
        {
            let specifier = if matches!(format, LiteralFormat::Binary) {
                "b"
            } else {
                "h"
            };
            return format!("{}'{}0", self.width, specifier);
        }

        match format {
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
            LiteralFormat::Binary => {
                let digits = pad_digits(self.value.to_str_radix(2), self.width.max(1));
                format!("{}'b{}", self.width, group_digits(&digits))
            }
            LiteralFormat::UnsizedBinary => {
                format!("'b{}", self.value.to_str_radix(2))
            }
            LiteralFormat::Hex => {
                let digit_count = self.width.div_ceil(4).max(1);
                let digits = pad_digits(self.value.to_str_radix(16), digit_count);
                format!("{}'h{}", self.width, group_digits(&digits))
            }
            LiteralFormat::UnsizedDecimal => self.value.to_str_radix(10),
            LiteralFormat::UnsizedHex => {
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
    use crate::{LiteralFormat, VastFile, VastFileType};

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
    fn unsized_hex_literals_keep_grouping_without_a_width() {
        let literal = Literal::parse("bits[32]:0x10000").expect("valid literal");
        assert_eq!(literal.format(LiteralFormat::UnsizedHex), "'h1_0000");
    }

    #[test]
    fn binary_literals_are_padded_and_grouped() {
        let literal = Literal::parse("bits[9]:0b11").expect("valid literal");
        assert_eq!(literal.format(LiteralFormat::Binary), "9'b0_0000_0011");
        assert_eq!(literal.format(LiteralFormat::UnsizedBinary), "'b11");
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
    fn huge_zero_literals_preserve_each_format_sizing_and_signedness() {
        let literal = Literal::parse("bits[4096]:0").expect("valid literal");
        assert_eq!(literal.format(LiteralFormat::Hex), "4096'h0");
        assert_eq!(literal.format(LiteralFormat::Binary), "4096'b0");
        assert_eq!(literal.format(LiteralFormat::SignedDecimal), "4096'sd0");
        assert_eq!(literal.format(LiteralFormat::UnsignedDecimal), "4096'd0");
        assert_eq!(literal.format(LiteralFormat::UnsizedBinary), "'b0");
        assert_eq!(literal.format(LiteralFormat::UnsizedDecimal), "0");
        assert_eq!(literal.format(LiteralFormat::UnsizedHex), "'h0");
    }

    #[test]
    fn decimal_formats_explicitly_choose_sized_or_unsized_emission() {
        let narrow = Literal::parse("bits[32]:42").expect("valid narrow literal");
        let wide = Literal::parse("bits[96]:42").expect("valid wide literal");

        assert_eq!(narrow.format(LiteralFormat::UnsizedDecimal), "42");
        assert_eq!(narrow.format(LiteralFormat::UnsignedDecimal), "32'd42");
        assert_eq!(wide.format(LiteralFormat::UnsizedDecimal), "42");
        assert_eq!(wide.format(LiteralFormat::UnsignedDecimal), "96'd42");
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

    #[test]
    fn every_format_emits_its_explicit_sizing_and_radix() {
        let literal = Literal::parse("bits[8]:0xa5").expect("valid literal");

        for (format, expected) in [
            (LiteralFormat::Binary, "8'b1010_0101"),
            (LiteralFormat::Hex, "8'ha5"),
            (LiteralFormat::SignedDecimal, "-8'sd91"),
            (LiteralFormat::UnsignedDecimal, "8'd165"),
            (LiteralFormat::UnsizedBinary, "'b10100101"),
            (LiteralFormat::UnsizedDecimal, "165"),
            (LiteralFormat::UnsizedHex, "'ha5"),
        ] {
            assert_eq!(literal.format(format), expected);
        }
    }

    #[test]
    fn unsized_decimal_validation_accepts_the_signed_32_bit_maximum() {
        let maximum = Literal::parse(&format!("bits[1025]:{}", i32::MAX))
            .expect("a small magnitude fits a wide source type");
        maximum
            .validate_format(LiteralFormat::UnsizedDecimal)
            .expect("the maximum portable decimal magnitude is valid");
        assert_eq!(
            maximum.format(LiteralFormat::UnsizedDecimal),
            i32::MAX.to_string()
        );

        let excessive =
            Literal::parse("bits[32]:2147483648").expect("the magnitude fits its source bit width");
        assert_eq!(
            excessive
                .validate_format(LiteralFormat::UnsizedDecimal)
                .expect_err("decimal literals must fit in a signed 32-bit value")
                .to_string(),
            "unsized decimal literal magnitude 2147483648 exceeds maximum 2147483647"
        );
        excessive
            .validate_format(LiteralFormat::UnsignedDecimal)
            .expect("sized decimal literals retain arbitrary precision");
    }

    #[test]
    fn unsized_based_validation_accepts_the_unsigned_32_bit_maximum() {
        let maximum = Literal::parse("bits[4096]:0xffff_ffff")
            .expect("a 32-bit magnitude fits a wide source type");
        for format in [LiteralFormat::UnsizedBinary, LiteralFormat::UnsizedHex] {
            maximum
                .validate_format(format)
                .expect("all unsigned 32-bit magnitudes are portable");
        }
        assert_eq!(
            maximum.format(LiteralFormat::UnsizedBinary),
            "'b11111111111111111111111111111111"
        );
        assert_eq!(maximum.format(LiteralFormat::UnsizedHex), "'hffff_ffff");

        let excessive =
            Literal::parse("bits[33]:4294967296").expect("the magnitude fits its source bit width");
        assert_eq!(
            excessive
                .validate_format(LiteralFormat::UnsizedBinary)
                .expect_err("unsized binary magnitudes must fit in 32 bits")
                .to_string(),
            "unsized binary literal magnitude 4294967296 exceeds maximum 4294967295"
        );
        assert_eq!(
            excessive
                .validate_format(LiteralFormat::UnsizedHex)
                .expect_err("unsized hexadecimal magnitudes must fit in 32 bits")
                .to_string(),
            "unsized hexadecimal literal magnitude 4294967296 exceeds maximum 4294967295"
        );
        excessive
            .validate_format(LiteralFormat::Binary)
            .expect("sized binary literals retain arbitrary precision");
        excessive
            .validate_format(LiteralFormat::Hex)
            .expect("sized hexadecimal literals retain arbitrary precision");
    }

    #[test]
    fn typed_negative_literals_are_validated_by_their_unsigned_bit_pattern() {
        let small = Literal::parse("bits[8]:-1").expect("valid two's-complement literal");
        small
            .validate_format(LiteralFormat::UnsizedDecimal)
            .expect("the eight-bit unsigned magnitude is portable");
        assert_eq!(small.format(LiteralFormat::UnsizedDecimal), "255");

        let wide = Literal::parse("bits[33]:-1").expect("valid wide two's-complement literal");
        assert!(wide.validate_format(LiteralFormat::UnsizedBinary).is_err());
        assert!(wide.validate_format(LiteralFormat::UnsizedHex).is_err());
        assert!(wide.validate_format(LiteralFormat::UnsizedDecimal).is_err());
    }

    #[test]
    fn public_literal_builder_rejects_nonportable_unsized_magnitudes() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let before = file.ast.expressions.len();

        assert!(
            file.make_literal("bits[32]:2147483648", &LiteralFormat::UnsizedDecimal)
                .is_err()
        );
        assert!(
            file.make_literal("bits[64]:4294967296", &LiteralFormat::UnsizedBinary)
                .is_err()
        );
        assert!(
            file.make_literal("bits[64]:4294967296", &LiteralFormat::UnsizedHex)
                .is_err()
        );
        assert_eq!(file.ast.expressions.len(), before);

        let decimal = file
            .make_literal("bits[4096]:42", &LiteralFormat::UnsizedDecimal)
            .expect("wide source widths do not restrict small unsized magnitudes");
        assert_eq!(file.emit_expression(&decimal), "42");
    }

    #[test]
    fn dedicated_unsized_decimal_builder_accepts_the_complete_i32_range() {
        let mut file = VastFile::new(VastFileType::Verilog);

        for (value, expected) in [
            (i32::MIN, "-2147483648"),
            (-1, "-1"),
            (0, "0"),
            (1, "1"),
            (i32::MAX, "2147483647"),
        ] {
            let literal = file.make_unsized_decimal_literal(value);
            assert_eq!(file.emit_expression(&literal), expected);
        }
    }

    #[test]
    fn compact_zero_optimization_starts_above_1024_bits() {
        let threshold = Literal::parse("bits[1024]:0").expect("valid threshold literal");
        let padded_hex = threshold.format(LiteralFormat::Hex);
        assert_eq!(padded_hex, format!("1024'h{}", "0000_".repeat(63) + "0000"));

        let compact = Literal::parse("bits[1025]:0").expect("valid compact literal");
        assert_eq!(compact.format(LiteralFormat::Binary), "1025'b0");
        assert_eq!(compact.format(LiteralFormat::Hex), "1025'h0");
        assert_eq!(compact.format(LiteralFormat::SignedDecimal), "1025'sd0");
        assert_eq!(compact.format(LiteralFormat::UnsignedDecimal), "1025'd0");
        assert_eq!(compact.format(LiteralFormat::UnsizedBinary), "'b0");
        assert_eq!(compact.format(LiteralFormat::UnsizedDecimal), "0");
        assert_eq!(compact.format(LiteralFormat::UnsizedHex), "'h0");
    }
}
