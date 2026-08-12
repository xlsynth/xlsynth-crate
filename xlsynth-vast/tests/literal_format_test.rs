// SPDX-License-Identifier: Apache-2.0

//! Public golden tests for sized and unsized Verilog integer literals.

use xlsynth_vast::{LiteralFormat, VastFile, VastFileType};

/// Asserts the complete emitted spelling of one parsed typed-bits literal.
fn assert_literal(file: &mut VastFile, source: &str, format: LiteralFormat, expected: &str) {
    let expression = file
        .make_literal(source, &format)
        .expect("source value should fit its requested literal format");
    assert_eq!(file.emit_expression(&expression), expected);
}

#[test]
fn all_seven_formats_preserve_sized_and_unsized_spellings_at_common_widths() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("literal_formats");
    let formats = [
        ("Binary", LiteralFormat::Binary),
        ("Hex", LiteralFormat::Hex),
        ("SignedDecimal", LiteralFormat::SignedDecimal),
        ("UnsignedDecimal", LiteralFormat::UnsignedDecimal),
        ("UnsizedBinary", LiteralFormat::UnsizedBinary),
        ("UnsizedDecimal", LiteralFormat::UnsizedDecimal),
        ("UnsizedHex", LiteralFormat::UnsizedHex),
    ];

    for (prefix, source) in [
        ("One", "bits[1]:1"),
        ("Eight", "bits[8]:42"),
        ("ThirtyTwo", "bits[32]:42"),
    ] {
        for (suffix, format) in formats {
            let expression = file
                .make_literal(source, &format)
                .expect("common-width literal should fit");
            file.add_localparam(module, &format!("{prefix}{suffix}"), &expression);
        }
    }

    let expected = r#"module literal_formats;
  localparam OneBinary = 1'b1;
  localparam OneHex = 1'h1;
  localparam OneSignedDecimal = -1'sd1;
  localparam OneUnsignedDecimal = 1'd1;
  localparam OneUnsizedBinary = 'b1;
  localparam OneUnsizedDecimal = 1;
  localparam OneUnsizedHex = 'h1;
  localparam EightBinary = 8'b0010_1010;
  localparam EightHex = 8'h2a;
  localparam EightSignedDecimal = 8'sd42;
  localparam EightUnsignedDecimal = 8'd42;
  localparam EightUnsizedBinary = 'b101010;
  localparam EightUnsizedDecimal = 42;
  localparam EightUnsizedHex = 'h2a;
  localparam ThirtyTwoBinary = 32'b0000_0000_0000_0000_0000_0000_0010_1010;
  localparam ThirtyTwoHex = 32'h0000_002a;
  localparam ThirtyTwoSignedDecimal = 32'sd42;
  localparam ThirtyTwoUnsignedDecimal = 32'd42;
  localparam ThirtyTwoUnsizedBinary = 'b101010;
  localparam ThirtyTwoUnsizedDecimal = 42;
  localparam ThirtyTwoUnsizedHex = 'h2a;
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn zero_literals_remain_correct_across_the_wide_compaction_boundary() {
    let mut file = VastFile::new(VastFileType::Verilog);

    for width in [1024, 1025, 4096] {
        let source = format!("bits[{width}]:0");
        let binary = if width == 1024 {
            format!("1024'b{}", vec!["0000"; 256].join("_"))
        } else {
            format!("{width}'b0")
        };
        let hex = if width == 1024 {
            format!("1024'h{}", vec!["0000"; 64].join("_"))
        } else {
            format!("{width}'h0")
        };

        assert_literal(&mut file, &source, LiteralFormat::Binary, &binary);
        assert_literal(&mut file, &source, LiteralFormat::Hex, &hex);
        assert_literal(
            &mut file,
            &source,
            LiteralFormat::SignedDecimal,
            &format!("{width}'sd0"),
        );
        assert_literal(
            &mut file,
            &source,
            LiteralFormat::UnsignedDecimal,
            &format!("{width}'d0"),
        );
        assert_literal(&mut file, &source, LiteralFormat::UnsizedBinary, "'b0");
        assert_literal(&mut file, &source, LiteralFormat::UnsizedDecimal, "0");
        assert_literal(&mut file, &source, LiteralFormat::UnsizedHex, "'h0");
    }
}

#[test]
fn all_formats_accept_zero_at_the_maximum_source_width() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let source = "bits[1048576]:0";

    for (format, expected) in [
        (LiteralFormat::Binary, "1048576'b0"),
        (LiteralFormat::Hex, "1048576'h0"),
        (LiteralFormat::SignedDecimal, "1048576'sd0"),
        (LiteralFormat::UnsignedDecimal, "1048576'd0"),
        (LiteralFormat::UnsizedBinary, "'b0"),
        (LiteralFormat::UnsizedDecimal, "0"),
        (LiteralFormat::UnsizedHex, "'h0"),
    ] {
        assert_literal(&mut file, source, format, expected);
    }
}

#[test]
fn unsized_formats_accept_small_nonzero_values_from_huge_source_widths() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);

    for width in [1025, 4096, 1_048_576] {
        let source = format!("bits[{width}]:42");
        assert_literal(&mut file, &source, LiteralFormat::UnsizedBinary, "'b101010");
        assert_literal(&mut file, &source, LiteralFormat::UnsizedDecimal, "42");
        assert_literal(&mut file, &source, LiteralFormat::UnsizedHex, "'h2a");
        assert_literal(
            &mut file,
            &source,
            LiteralFormat::UnsignedDecimal,
            &format!("{width}'d42"),
        );
    }
}

#[test]
fn dedicated_unsized_decimal_constructor_preserves_the_full_signed_i32_range() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("signed_unsized_decimals");

    for (name, value) in [
        ("Minimum", i32::MIN),
        ("Negative", -1),
        ("Zero", 0),
        ("Positive", 1),
        ("Maximum", i32::MAX),
    ] {
        let expression = file.make_unsized_decimal_literal(value);
        file.add_localparam(module, name, &expression);
    }

    let expected = r#"module signed_unsized_decimals;
  localparam Minimum = -2147483648;
  localparam Negative = -1;
  localparam Zero = 0;
  localparam Positive = 1;
  localparam Maximum = 2147483647;
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn unsized_decimal_accepts_i32_maximum_and_rejects_the_next_positive_value() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);

    for source in [
        "bits[31]:2147483647",
        "bits[32]:2147483647",
        "bits[1025]:2147483647",
        "bits[1048576]:2147483647",
    ] {
        assert_literal(
            &mut file,
            source,
            LiteralFormat::UnsizedDecimal,
            "2147483647",
        );
    }

    for source in [
        "bits[32]:2147483648",
        "bits[1025]:2147483648",
        "bits[1048576]:2147483648",
    ] {
        assert!(
            file.make_literal(source, &LiteralFormat::UnsizedDecimal)
                .is_err(),
            "{source} exceeds the positive unsized-decimal range"
        );
    }
}

#[test]
fn unsized_hex_accepts_u32_maximum_and_rejects_the_next_positive_value() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);

    for source in [
        "bits[32]:0xffff_ffff",
        "bits[4096]:4294967295",
        "bits[1048576]:0xffff_ffff",
    ] {
        assert_literal(&mut file, source, LiteralFormat::UnsizedHex, "'hffff_ffff");
    }

    for source in [
        "bits[33]:0x1_0000_0000",
        "bits[4096]:4294967296",
        "bits[1048576]:0x1_0000_0000",
    ] {
        assert!(
            file.make_literal(source, &LiteralFormat::UnsizedHex)
                .is_err(),
            "{source} exceeds the unsized-based-literal range"
        );
    }
}

#[test]
fn unsized_binary_accepts_u32_maximum_and_rejects_the_next_positive_value() {
    let mut file = VastFile::new(VastFileType::Verilog);

    for source in [
        "bits[32]:4294967295",
        "bits[4096]:0xffff_ffff",
        "bits[1048576]:4294967295",
    ] {
        assert_literal(
            &mut file,
            source,
            LiteralFormat::UnsizedBinary,
            "'b11111111111111111111111111111111",
        );
    }

    for source in [
        "bits[33]:4294967296",
        "bits[4096]:0x1_0000_0000",
        "bits[1048576]:4294967296",
    ] {
        assert!(
            file.make_literal(source, &LiteralFormat::UnsizedBinary)
                .is_err(),
            "{source} exceeds the unsized-based-literal range"
        );
    }
}

#[test]
fn sized_formats_preserve_arbitrary_widths_without_unsized_value_limits() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let source = "bits[130]:0x1_0000_0000";
    let binary_digits = format!("{:0130b}", 1u64 << 32);
    let grouped_binary = binary_digits.char_indices().fold(
        String::with_capacity(binary_digits.len() + binary_digits.len() / 4),
        |mut grouped, (index, digit)| {
            if index != 0 && (binary_digits.len() - index) % 4 == 0 {
                grouped.push('_');
            }
            grouped.push(digit);
            grouped
        },
    );

    assert_literal(
        &mut file,
        source,
        LiteralFormat::Binary,
        &format!("130'b{grouped_binary}"),
    );
    assert_literal(
        &mut file,
        source,
        LiteralFormat::Hex,
        "130'h0_0000_0000_0000_0000_0000_0001_0000_0000",
    );
    assert_literal(
        &mut file,
        source,
        LiteralFormat::SignedDecimal,
        "130'sd4294967296",
    );
    assert_literal(
        &mut file,
        source,
        LiteralFormat::UnsignedDecimal,
        "130'd4294967296",
    );

    assert_literal(
        &mut file,
        "bits[4096]:4294967296",
        LiteralFormat::UnsignedDecimal,
        "4096'd4294967296",
    );
}

#[test]
fn signed_decimal_interprets_the_declared_sign_bit_while_unsized_based_formats_do_not() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let source = "bits[8]:255";

    assert_literal(&mut file, source, LiteralFormat::SignedDecimal, "-8'sd1");
    assert_literal(&mut file, source, LiteralFormat::UnsignedDecimal, "8'd255");
    assert_literal(
        &mut file,
        source,
        LiteralFormat::UnsizedBinary,
        "'b11111111",
    );
    assert_literal(&mut file, source, LiteralFormat::UnsizedDecimal, "255");
    assert_literal(&mut file, source, LiteralFormat::UnsizedHex, "'hff");
}

#[test]
fn unsized_hex_groups_radix_digits_without_reintroducing_a_source_width() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);

    for source in ["bits[17]:0x1_0000", "bits[4096]:0x1_0000"] {
        assert_literal(&mut file, source, LiteralFormat::UnsizedHex, "'h1_0000");
        assert_literal(
            &mut file,
            source,
            LiteralFormat::UnsizedBinary,
            "'b10000000000000000",
        );
    }
}
