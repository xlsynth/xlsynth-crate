// SPDX-License-Identifier: Apache-2.0

//! Expression and data-type golden tests adapted from upstream XLS VAST.

use xlsynth_vast::{DataKind, Expr, LiteralFormat, VastFile, VastFileType, VastModule};

/// Creates three eight-bit operands for upstream-style expression tests.
fn named_operands(file: &mut VastFile) -> (VastModule, Expr, Expr, Expr) {
    let module = file.add_module("expressions");
    let byte = file.make_bit_vector_type(8, false);
    let a = file.add_wire(module, "a", &byte).to_expr();
    let b = file.add_wire(module, "b", &byte).to_expr();
    let c = file.add_wire(module, "c", &byte).to_expr();
    (module, a, b, c)
}

/// Appends an assignment through the public two-stage construction API.
fn assign(file: &mut VastFile, module: VastModule, lhs: &Expr, rhs: &Expr) {
    let assignment = file.make_continuous_assignment(lhs, rhs);
    file.add_member_continuous_assignment(module, assignment);
}

#[test]
fn scalar_and_single_bit_vectors_match_upstream_type_properties() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("scalars");
    let scalar = file.make_scalar_type();
    let unsigned = file.make_bit_vector_type(1, false);
    let signed = file.make_bit_vector_type(1, true);

    for data_type in [scalar, unsigned, signed] {
        assert_eq!(file.type_width_as_int64(data_type), Ok(1));
        assert_eq!(file.type_flat_bit_count_as_int64(data_type), Ok(1));
        assert_eq!(file.type_width_expr(data_type), None);
    }
    assert!(!file.type_is_signed(scalar));
    assert!(!file.type_is_signed(unsigned));
    assert!(file.type_is_signed(signed));

    file.add_input(module, "scalar", &scalar);
    file.add_input(module, "unsigned_bit", &unsigned);
    file.add_input(module, "signed_bit", &signed);
    let expected = r#"module scalars(
  input wire scalar,
  input wire unsigned_bit,
  input wire signed signed_bit
);

endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn signed_and_unsigned_vectors_preserve_declared_widths() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("vectors");
    let unsigned = file.make_bit_vector_type(2, false);
    let signed = file.make_bit_vector_type(32, true);

    assert_eq!(file.type_width_as_int64(unsigned), Ok(2));
    assert_eq!(file.type_flat_bit_count_as_int64(unsigned), Ok(2));
    assert_eq!(file.type_width_as_int64(signed), Ok(32));
    assert_eq!(file.type_flat_bit_count_as_int64(signed), Ok(32));
    assert!(file.type_is_signed(signed));
    let signed_width = file.type_width_expr(signed).expect("vector has width");
    assert_eq!(file.emit_expression(&signed_width), "32");

    file.add_input(module, "u2", &unsigned);
    file.add_output(module, "s32", &signed);
    let expected = r#"module vectors(
  input wire [1:0] u2,
  output wire signed [31:0] s32
);

endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn packed_arrays_preserve_element_width_and_total_flat_width() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("packed");
    let element = file.make_bit_vector_type(10, false);
    let signed_element = file.make_bit_vector_type(10, true);
    let unsigned_array = file.make_packed_array_type(element, &[3, 2]);
    let signed_array = file.make_packed_array_type(signed_element, &[3, 2]);

    for array in [unsigned_array, signed_array] {
        assert_eq!(file.type_width_as_int64(array), Ok(10));
        assert_eq!(file.type_flat_bit_count_as_int64(array), Ok(60));
        let width = file.type_width_expr(array).expect("element has width");
        assert_eq!(file.emit_expression(&width), "10");
    }
    assert!(!file.type_is_signed(unsigned_array));
    assert!(file.type_is_signed(signed_array));

    file.add_wire(module, "unsigned_array", &unsigned_array);
    file.add_wire(module, "signed_array", &signed_array);
    let expected = r#"module packed;
  wire [2:0][1:0][9:0] unsigned_array;
  wire signed [2:0][1:0][9:0] signed_array;
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn unpacked_arrays_with_packed_elements_match_both_upstream_dialects() {
    for file_type in [VastFileType::Verilog, VastFileType::SystemVerilog] {
        let mut file = VastFile::new(file_type);
        let module = file.add_module("arrays");
        let element = file.make_bit_vector_type(4, false);
        let packed = file.make_packed_array_type(element, &[42, 7]);
        let unpacked = file.make_unpacked_array_type(packed, &[8, 64]);
        let output_type = file.make_bit_vector_type(64, false);
        let output = file.add_output(module, "out", &output_type);
        let array = file
            .add_reg(module, "arr", &unpacked)
            .expect("array register is valid");
        let row = file.make_index(&array.to_indexable_expr(), 2);
        let element = file.make_index(&row.to_indexable_expr(), 1);
        assign(&mut file, module, &output.to_expr(), &element.to_expr());

        assert_eq!(file.type_width_as_int64(unpacked), Ok(4));
        assert_eq!(
            file.type_flat_bit_count_as_int64(unpacked),
            Ok(4 * 42 * 7 * 8 * 64)
        );

        let dimensions = match file_type {
            VastFileType::Verilog => "[0:7][0:63]",
            VastFileType::SystemVerilog => "[8][64]",
        };
        let expected = format!(
            "module arrays(\n  output wire [63:0] out\n);\n  \
             reg [41:0][6:0][3:0] arr{dimensions};\n  \
             assign out = arr[2][1];\nendmodule\n"
        );
        assert_eq!(file.emit(), expected);
    }
}

#[test]
fn symbolic_width_expression_matches_upstream_multiplication_example() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("symbolic");
    let ten = file.make_unsized_decimal_literal(10);
    let five = file.make_unsized_decimal_literal(5);
    let width = file.make_mul(&ten, &five);
    let data_type = file.make_bit_vector_type_expr(&width, false);
    file.add_wire(module, "foo", &data_type);

    assert_eq!(file.type_width_expr(data_type), Some(width));
    assert_eq!(file.emit_expression(&width), "10 * 5");
    assert_eq!(
        file.type_width_as_int64(data_type)
            .expect_err("compound widths are not folded")
            .to_string(),
        "Width is not a literal: 10 * 5"
    );
    assert_eq!(
        file.type_flat_bit_count_as_int64(data_type)
            .expect_err("compound widths have unknown flat width")
            .to_string(),
        "Width is not a literal: 10 * 5"
    );
    assert_eq!(
        file.emit(),
        "module symbolic;\n  wire [10 * 5 - 1:0] foo;\nendmodule\n"
    );
}

#[test]
fn symbolic_low_precedence_widths_are_parenthesized_before_subtraction() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let (module, a, b, _) = named_operands(&mut file);
    let width = file.make_logical_or(&a, &b);
    let data_type = file.make_bit_vector_type_expr(&width, true);
    file.add_wire(module, "result", &data_type);

    let expected = r#"module expressions;
  wire [7:0] a;
  wire [7:0] b;
  wire [7:0] c;
  wire signed [(a || b) - 1:0] result;
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn literal_matrix_matches_upstream_values_and_grouping() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let cases = [
        ("bits[32]:44", LiteralFormat::UnsignedDecimal, "32'd44"),
        ("bits[1]:1", LiteralFormat::Binary, "1'b1"),
        ("bits[4]:10", LiteralFormat::Binary, "4'b1010"),
        ("bits[42]:12345", LiteralFormat::Hex, "42'h000_0000_3039"),
        ("bits[1]:0", LiteralFormat::UnsignedDecimal, "1'd0"),
        ("bits[3]:2", LiteralFormat::Hex, "3'h2"),
        ("bits[3]:2", LiteralFormat::Binary, "3'b010"),
        ("bits[32]:55", LiteralFormat::UnsizedDecimal, "55"),
        (
            "bits[32]:55",
            LiteralFormat::Binary,
            "32'b0000_0000_0000_0000_0000_0000_0011_0111",
        ),
        ("bits[55]:1234", LiteralFormat::Hex, "55'h00_0000_0000_04d2"),
        ("bits[55]:1234", LiteralFormat::UnsignedDecimal, "55'd1234"),
    ];

    for (typed_value, format, expected) in cases {
        let literal = file
            .make_literal(typed_value, &format)
            .expect("upstream literal is representable");
        assert_eq!(file.emit_expression(&literal), expected, "{typed_value}");
    }
}

#[test]
fn literal_input_radices_and_underscores_normalize_correctly() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let cases = [
        ("bits[16]:0X00_FF", LiteralFormat::Hex, "16'h00ff"),
        (
            "bits[12]:0B1010_0101",
            LiteralFormat::Binary,
            "12'b0000_1010_0101",
        ),
        ("bits[12]:0o755", LiteralFormat::UnsignedDecimal, "12'd493"),
        ("bits[16]:+42", LiteralFormat::UnsignedDecimal, "16'd42"),
        ("bits[16]:1_000", LiteralFormat::UnsignedDecimal, "16'd1000"),
    ];

    for (typed_value, format, expected) in cases {
        let literal = file
            .make_literal(typed_value, &format)
            .expect("alternate-radix literal is representable");
        assert_eq!(file.emit_expression(&literal), expected, "{typed_value}");
    }
}

#[test]
fn huge_zero_literals_switch_to_compact_upstream_spelling() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let cases = [
        (LiteralFormat::Binary, "1025'b0"),
        (LiteralFormat::Hex, "1025'h0"),
        (LiteralFormat::SignedDecimal, "1025'sd0"),
        (LiteralFormat::UnsignedDecimal, "1025'd0"),
        (LiteralFormat::UnsizedBinary, "'b0"),
        (LiteralFormat::UnsizedDecimal, "0"),
        (LiteralFormat::UnsizedHex, "'h0"),
    ];

    for (format, expected) in cases {
        let zero = file
            .make_literal("bits[1025]:0", &format)
            .expect("zero fits any positive bit width");
        assert_eq!(file.emit_expression(&zero), expected);
    }

    let threshold = file
        .make_literal("bits[1024]:0", &LiteralFormat::Hex)
        .expect("threshold-width zero is representable");
    let expected = format!("1024'h{}", vec!["0000"; 64].join("_"));
    assert_eq!(file.emit_expression(&threshold), expected);
}

#[test]
fn unsized_and_sized_radix_formats_preserve_width_and_separator_rules() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let unsized_hex = file
        .make_literal("bits[32]:0x10000", &LiteralFormat::UnsizedHex)
        .expect("valid unsized hexadecimal value");
    let unsized_binary = file
        .make_literal("bits[32]:0b10100101", &LiteralFormat::UnsizedBinary)
        .expect("valid unsized binary value");
    let sized_hex = file
        .make_literal("bits[17]:0x55", &LiteralFormat::Hex)
        .expect("valid sized hexadecimal value");
    let sized_binary = file
        .make_literal("bits[9]:3", &LiteralFormat::Binary)
        .expect("valid sized binary value");

    assert_eq!(file.emit_expression(&unsized_hex), "'h1_0000");
    assert_eq!(file.emit_expression(&unsized_binary), "'b10100101");
    assert_eq!(file.emit_expression(&sized_hex), "17'h0_0055");
    assert_eq!(file.emit_expression(&sized_binary), "9'b0_0000_0011");
}

#[test]
fn signed_decimal_literals_cover_positive_negative_and_minimum_values() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let cases = [
        ("bits[8]:0", "8'sd0"),
        ("bits[8]:127", "8'sd127"),
        ("bits[8]:128", "-8'sd128"),
        ("bits[8]:255", "-8'sd1"),
        ("bits[8]:-1", "-8'sd1"),
        ("bits[8]:-128", "-8'sd128"),
        (
            "bits[128]:0x80000000000000000000000000000000",
            "-128'sd170141183460469231731687303715884105728",
        ),
    ];

    for (typed_value, expected) in cases {
        let literal = file
            .make_literal(typed_value, &LiteralFormat::SignedDecimal)
            .expect("signed value is representable");
        assert_eq!(file.emit_expression(&literal), expected, "{typed_value}");
    }
}

#[test]
fn upstream_unary_precedence_and_nested_operators_are_preserved() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let (_, a, b, _) = named_operands(&mut file);

    let negative = file.make_negate(&a);
    assert_eq!(file.emit_expression(&negative), "-a");

    let inverted = file.make_bitwise_not(&a);
    let negative_inverted = file.make_negate(&inverted);
    assert_eq!(file.emit_expression(&negative_inverted), "-(~a)");

    let disjunction = file.make_bitwise_or(&a, &b);
    let negative_disjunction = file.make_negate(&disjunction);
    assert_eq!(file.emit_expression(&negative_disjunction), "-(a | b)");

    let twice_negated = file.make_negate(&negative);
    let three_times_negated = file.make_negate(&twice_negated);
    assert_eq!(file.emit_expression(&three_times_negated), "-(-(-a))");

    let sum = file.make_add(&a, &b);
    let inverted_sum = file.make_bitwise_not(&sum);
    let negative_inverted_sum = file.make_negate(&inverted_sum);
    let expression = file.make_sub(&negative_inverted_sum, &b);
    assert_eq!(file.emit_expression(&expression), "-(~(a + b)) - b");
}

#[test]
fn upstream_binary_precedence_and_associativity_examples_are_preserved() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let (_, a, b, c) = named_operands(&mut file);

    let b_plus_c = file.make_add(&b, &c);
    let right_associative = file.make_add(&a, &b_plus_c);
    assert_eq!(file.emit_expression(&right_associative), "a + (b + c)");

    let a_plus_b = file.make_add(&a, &b);
    let left_associative = file.make_add(&a_plus_b, &c);
    assert_eq!(file.emit_expression(&left_associative), "a + b + c");

    let b_times_c = file.make_mul(&b, &c);
    let addition = file.make_add(&a, &b_times_c);
    assert_eq!(file.emit_expression(&addition), "a + b * c");

    let product = file.make_mul(&a, &b_plus_c);
    assert_eq!(file.emit_expression(&product), "a * (b + c)");

    let power = file.make_power(&a, &b_times_c);
    assert_eq!(file.emit_expression(&power), "a ** (b * c)");

    let logical_or = file.make_logical_or(&a_plus_b, &b);
    let mixed_or = file.make_bitwise_or(&a, &logical_or);
    assert_eq!(file.emit_expression(&mixed_or), "a | (a + b || b)");

    let a_times_b = file.make_mul(&a, &b);
    let b_and_c = file.make_bitwise_and(&b, &c);
    let shift = file.make_shll(&a_times_b, &b_and_c);
    assert_eq!(file.emit_expression(&shift), "a * b << (b & c)");
}

#[test]
fn every_binary_operator_uses_its_expected_verilog_spelling() {
    type BinaryBuilder = fn(&mut VastFile, &Expr, &Expr) -> Expr;
    let cases: &[(&str, BinaryBuilder)] = &[
        ("+", VastFile::make_add),
        ("-", VastFile::make_sub),
        ("*", VastFile::make_mul),
        ("/", VastFile::make_div),
        ("%", VastFile::make_mod),
        ("**", VastFile::make_power),
        ("&", VastFile::make_bitwise_and),
        ("|", VastFile::make_bitwise_or),
        ("^", VastFile::make_bitwise_xor),
        ("<<", VastFile::make_shll),
        (">>>", VastFile::make_shra),
        (">>", VastFile::make_shrl),
        ("!=", VastFile::make_ne),
        ("!==", VastFile::make_case_ne),
        ("==", VastFile::make_eq),
        ("===", VastFile::make_case_eq),
        (">=", VastFile::make_ge),
        (">", VastFile::make_gt),
        ("<=", VastFile::make_le),
        ("<", VastFile::make_lt),
        ("&&", VastFile::make_logical_and),
        ("||", VastFile::make_logical_or),
        ("!==", VastFile::make_ne_x),
        ("===", VastFile::make_eq_x),
    ];
    let mut file = VastFile::new(VastFileType::Verilog);
    let (_, a, b, _) = named_operands(&mut file);

    for (spelling, builder) in cases {
        let expression = builder(&mut file, &a, &b);
        assert_eq!(file.emit_expression(&expression), format!("a {spelling} b"));
    }
}

#[test]
fn reductions_receive_upstream_required_binary_parentheses() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let (_, a, b, c) = named_operands(&mut file);
    let reduced_a = file.make_or_reduce(&a);
    let and_reduced_a = file.make_and_reduce(&a);
    let reduced_c = file.make_xor_reduce(&c);

    assert_eq!(file.emit_expression(&reduced_a), "|a");
    assert_eq!(file.emit_expression(&and_reduced_a), "&a");

    let logical = file.make_logical_and(&b, &reduced_c);
    assert_eq!(file.emit_expression(&logical), "b && (^c)");

    let xor_a = file.make_xor_reduce(&a);
    let first_addition = file.make_add(&xor_a, &b);
    let or_c = file.make_or_reduce(&c);
    let second_addition = file.make_add(&first_addition, &or_c);
    assert_eq!(file.emit_expression(&second_addition), "(^a) + b + (|c)");
}

#[test]
fn ternary_expressions_parenthesize_every_nested_ternary_operand() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let (_, a, b, c) = named_operands(&mut file);
    let nested = file.make_ternary(&a, &b, &c);

    let conditional = file.make_ternary(&nested, &b, &c);
    let consequent = file.make_ternary(&a, &nested, &c);
    let alternate = file.make_ternary(&a, &b, &nested);
    assert_eq!(file.emit_expression(&conditional), "(a ? b : c) ? b : c");
    assert_eq!(file.emit_expression(&consequent), "a ? (a ? b : c) : c");
    assert_eq!(file.emit_expression(&alternate), "a ? b : (a ? b : c)");
}

#[test]
fn concatenations_match_singleton_mixed_and_empty_upstream_forms() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let (_, a, b, _) = named_operands(&mut file);
    let forty_two = file
        .make_literal("bits[32]:42", &LiteralFormat::Hex)
        .expect("valid singleton literal");
    let one_twenty_three = file
        .make_literal("bits[8]:123", &LiteralFormat::Hex)
        .expect("valid mixed literal");

    let singleton = file.make_concat(&[&forty_two]);
    let mixed = file.make_concat(&[&a, &one_twenty_three, &b]);
    let empty = file.make_concat(&[]);
    assert_eq!(file.emit_expression(&singleton), "{32'h0000_002a}");
    assert_eq!(file.emit_expression(&mixed), "{a, 8'h7b, b}");
    assert_eq!(file.emit_expression(&empty), "{}");
}

#[test]
fn replicated_concatenations_preserve_expression_and_constant_counts() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let (_, a, b, _) = named_operands(&mut file);
    let one_twenty_three = file
        .make_literal("bits[8]:123", &LiteralFormat::Hex)
        .expect("valid concatenation literal");
    let forty_two = file.make_unsized_decimal_literal(42);
    let by_expression = file.make_replicated_concat(&forty_two, &[&a, &one_twenty_three, &b]);
    let by_constant = file.make_replicated_concat_i64(42, &[&a, &one_twenty_three, &b]);

    assert_eq!(file.emit_expression(&by_expression), "{42{a, 8'h7b, b}}");
    assert_eq!(file.emit_expression(&by_constant), "{42{a, 8'h7b, b}}");
}

#[test]
fn nested_array_assignment_patterns_match_upstream_examples() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let (module, a, b, _) = named_operands(&mut file);
    let byte = file.make_bit_vector_type(8, false);
    let foo = file.add_wire(module, "foo", &byte).to_expr();
    let bar = file.add_wire(module, "bar", &byte).to_expr();
    let baz = file.add_wire(module, "baz", &byte).to_expr();
    let qux = file.add_wire(module, "qux", &byte).to_expr();
    let literal = file
        .make_literal("bits[32]:123", &LiteralFormat::Hex)
        .expect("valid array pattern literal");

    let mixed = file.make_array_assignment_pattern(&[&a, &literal, &b]);
    assert_eq!(file.emit_expression(&mixed), "'{a, 32'h0000_007b, b}");

    let first_row = file.make_array_assignment_pattern(&[&foo, &bar]);
    let second_row = file.make_array_assignment_pattern(&[&baz, &qux]);
    let nested = file.make_array_assignment_pattern(&[&first_row, &second_row]);
    assert_eq!(file.emit_expression(&nested), "'{'{foo, bar}, '{baz, qux}}");
}

#[test]
fn upstream_width_cast_module_covers_literal_parameter_and_expression_widths() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("top");
    let byte = file.make_bit_vector_type(8, false);
    let nibble = file.make_bit_vector_type(4, false);
    let twelve_bits = file.make_bit_vector_type(12, false);
    let sixteen_bits = file.make_bit_vector_type(16, false);
    let a = file.add_input(module, "a", &byte);
    let b = file.add_input(module, "b", &nibble);
    let out_literal = file.add_output(module, "out_literal", &byte);
    let out_param = file.add_output(module, "out_param", &twelve_bits);
    let out_expr = file.add_output(module, "out_expr", &sixteen_bits);
    let twelve = file.make_unsized_decimal_literal(12);
    let width_param = file.add_parameter(module, "WidthParam", &twelve);

    let eight = file.make_unsized_decimal_literal(8);
    let one = file.make_unsized_decimal_literal(1);
    let plus_one = file.make_add(&a.to_expr(), &one);
    let cast_literal = file.make_width_cast(&eight, &plus_one);
    assign(&mut file, module, &out_literal.to_expr(), &cast_literal);

    let concatenation = file.make_concat(&[&a.to_expr(), &b.to_expr()]);
    let cast_param = file.make_width_cast(&width_param.to_expr(), &concatenation);
    assign(&mut file, module, &out_param.to_expr(), &cast_param);

    let four = file.make_unsized_decimal_literal(4);
    let complex_width = file.make_add(&width_param.to_expr(), &four);
    let three = file
        .make_literal("bits[2]:3", &LiteralFormat::Hex)
        .expect("valid two-bit literal");
    let rhs_concat = file.make_concat(&[&b.to_expr(), &three]);
    let complex_value = file.make_bitwise_xor(&a.to_expr(), &rhs_concat);
    let cast_expr = file.make_width_cast(&complex_width, &complex_value);
    assign(&mut file, module, &out_expr.to_expr(), &cast_expr);

    let expected = r#"module top(
  input wire [7:0] a,
  input wire [3:0] b,
  output wire [7:0] out_literal,
  output wire [11:0] out_param,
  output wire [15:0] out_expr
);
  parameter WidthParam = 12;
  assign out_literal = 8'(a + 1);
  assign out_param = WidthParam'({a, b});
  assign out_expr = (WidthParam + 4)'(a ^ {b, 2'h3});
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn width_cast_identifier_rules_match_upstream_digits_and_dollar_identifiers() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("casts");
    let scalar = file.make_scalar_type();
    let value = file.add_input(module, "value", &scalar).to_expr();
    let parameter_value = file.make_unsized_decimal_literal(4);
    let identifier = file.add_parameter(module, "WIDTH$value", &parameter_value);
    let from_identifier = file.make_width_cast(&identifier.to_expr(), &value);
    let from_digits = file.make_width_cast(&parameter_value, &value);

    assert_eq!(
        file.emit_expression(&from_identifier),
        "WIDTH$value'(value)"
    );
    assert_eq!(file.emit_expression(&from_digits), "4'(value)");
}

#[test]
fn upstream_type_cast_module_handles_local_and_package_qualified_types() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("top");
    let byte = file.make_bit_vector_type(8, false);
    let local = file.make_extern_type("foobar");
    let qualified = file.make_extern_package_type("my_pkg", "my_type_t");
    let input = file.add_input(module, "a", &byte);
    let out_local = file.add_output(module, "out_foo", &local);
    let out_package = file.add_output(module, "out_pkg", &qualified);
    let one = file.make_unsized_decimal_literal(1);
    let incremented = file.make_add(&input.to_expr(), &one);
    let local_cast = file.make_type_cast(&local, &incremented);
    let package_cast = file.make_type_cast(&qualified, &input.to_expr());
    assign(&mut file, module, &out_local.to_expr(), &local_cast);
    assign(&mut file, module, &out_package.to_expr(), &package_cast);

    let expected = r#"module top(
  input wire [7:0] a,
  output foobar out_foo,
  output my_pkg::my_type_t out_pkg
);
  assign out_foo = foobar'(a + 1);
  assign out_pkg = my_pkg::my_type_t'(a);
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn empty_package_names_preserve_global_scope_type_cast_spelling() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let (_, value, _, _) = named_operands(&mut file);
    let globally_qualified = file.make_extern_package_type("", "payload_t");
    let cast = file.make_type_cast(&globally_qualified, &value);

    assert_eq!(file.emit_expression(&cast), "::payload_t'(a)");
}

#[test]
fn integer_and_int_types_preserve_signedness_and_fixed_flat_width() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("integers");
    let signed_integer = file.make_integer_type(true);
    let unsigned_integer = file.make_integer_type(false);
    let signed_int = file.make_int_type(true);
    let unsigned_int = file.make_int_type(false);
    let one = file.make_unsized_decimal_literal(1);
    let definitions = [
        ("SI", DataKind::Integer, signed_integer),
        ("UI", DataKind::Integer, unsigned_integer),
        ("S2", DataKind::Int, signed_int),
        ("U2", DataKind::Int, unsigned_int),
    ];

    for (name, kind, data_type) in definitions {
        assert_eq!(file.type_flat_bit_count_as_int64(data_type), Ok(32));
        assert_eq!(file.type_width_expr(data_type), None);
        assert!(file.type_width_as_int64(data_type).is_err());
        let definition = file.make_def(name, kind, &data_type);
        file.add_parameter_with_def(module, &definition, &one);
    }
    assert!(file.type_is_signed(signed_integer));
    assert!(!file.type_is_signed(unsigned_integer));
    assert!(file.type_is_signed(signed_int));
    assert!(!file.type_is_signed(unsigned_int));

    let expected = r#"module integers;
  parameter integer SI = 1;
  parameter integer unsigned UI = 1;
  parameter int S2 = 1;
  parameter int unsigned U2 = 1;
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn scalar_zero_selects_disappear_while_vectors_retain_their_selects() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("selects");
    let scalar = file.make_scalar_type();
    let vector = file.make_bit_vector_type(2, false);
    let bit = file.add_input(module, "bit", &scalar);
    let bits = file.add_input(module, "bits", &vector);

    let scalar_index = file.make_index(&bit.to_indexable_expr(), 0);
    let scalar_slice = file.make_slice(&bit.to_indexable_expr(), 0, 0);
    let vector_index = file.make_index(&bits.to_indexable_expr(), 0);
    let vector_slice = file.make_slice(&bits.to_indexable_expr(), 0, 0);
    assert_eq!(file.emit_expression(&scalar_index.to_expr()), "bit");
    assert_eq!(file.emit_expression(&scalar_slice.to_expr()), "bit");
    assert_eq!(file.emit_expression(&vector_index.to_expr()), "bits[0]");
    assert_eq!(file.emit_expression(&vector_slice.to_expr()), "bits[0:0]");
}

#[test]
fn large_constant_indices_switch_representation_at_the_i32_boundary() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("indices");
    let vector = file.make_bit_vector_type(8, false);
    let signal = file.add_input(module, "signal", &vector);
    let small = file.make_index(&signal.to_indexable_expr(), i64::from(i32::MAX));
    let large = file.make_index(&signal.to_indexable_expr(), i64::from(i32::MAX) + 1);
    let slice = file.make_slice(
        &signal.to_indexable_expr(),
        i64::from(i32::MAX) + 2,
        i64::from(i32::MAX) + 1,
    );

    assert_eq!(file.emit_expression(&small.to_expr()), "signal[2147483647]");
    assert_eq!(
        file.emit_expression(&large.to_expr()),
        "signal[64'h0000_0000_8000_0000]"
    );
    assert_eq!(
        file.emit_expression(&slice.to_expr()),
        "signal[64'h0000_0000_8000_0001:64'h0000_0000_8000_0000]"
    );
}

#[test]
fn expression_based_indices_and_slices_keep_arithmetic_expressions() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let (_, a, b, c) = named_operands(&mut file);
    let one = file.make_unsized_decimal_literal(1);
    let hi = file.make_add(&b, &one);
    let lo = file.make_sub(&c, &one);
    let index = file.make_mul(&b, &c);
    let subject = {
        let module = file.add_module("subjects");
        let vector = file.make_bit_vector_type(16, false);
        file.add_input(module, "subject", &vector)
    };
    let dynamic_slice = file.make_slice_expr(&subject.to_indexable_expr(), &hi, &lo);
    let dynamic_index = file.make_index_expr(&subject.to_indexable_expr(), &index);
    let result = file.make_add(&a, &dynamic_index.to_expr());

    assert_eq!(
        file.emit_expression(&dynamic_slice.to_expr()),
        "subject[b + 1:c - 1]"
    );
    assert_eq!(
        file.emit_expression(&dynamic_index.to_expr()),
        "subject[b * c]"
    );
    assert_eq!(file.emit_expression(&result), "a + subject[b * c]");
}

#[test]
fn packed_multidimensional_slices_match_upstream_assignment_golden() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("top");
    let two_bits = file.make_bit_vector_type(2, false);
    let a_type = file.make_packed_array_type(two_bits, &[3, 5]);
    let c_type = file.make_bit_vector_type(3, false);
    let a = file.add_wire(module, "a", &a_type);
    let b = file.add_wire(module, "b", &two_bits);
    let c = file.add_wire(module, "c", &c_type);

    let first_dimension = file.make_index(&a.to_indexable_expr(), 1);
    let second_dimension = file.make_index(&first_dimension.to_indexable_expr(), 2);
    let first_lhs = file.make_slice(&second_dimension.to_indexable_expr(), 3, 4);
    let first_rhs = file.make_slice(&b.to_indexable_expr(), 1, 0);
    assign(
        &mut file,
        module,
        &first_lhs.to_expr(),
        &first_rhs.to_expr(),
    );

    let second_lhs = file.make_slice(&a.to_indexable_expr(), 3, 4);
    let second_rhs = file.make_slice(&c.to_indexable_expr(), 2, 1);
    assign(
        &mut file,
        module,
        &second_lhs.to_expr(),
        &second_rhs.to_expr(),
    );

    let expected = r#"module top;
  wire [2:0][4:0][1:0] a;
  wire [1:0] b;
  wire [2:0] c;
  assign a[1][2][3:4] = b[1:0];
  assign a[3:4] = c[2:1];
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn upstream_tick_zero_one_and_unknown_parameters_emit_in_order() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("top");
    let zero = file.make_unsized_zero_literal();
    let one = file.make_unsized_one_literal();
    let unknown = file.make_unsized_x_literal();
    file.add_parameter(module, "P0", &zero);
    file.add_parameter(module, "P1", &one);
    file.add_parameter(module, "P2", &unknown);

    let expected = r#"module top;
  parameter P0 = '0;
  parameter P1 = '1;
  parameter P2 = 'X;
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn upstream_localparams_cover_plain_decimal_hexadecimal_and_binary() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("top");
    let decimal = file.make_unsized_decimal_literal(42);
    let hex = file
        .make_literal("bits[32]:0x42", &LiteralFormat::UnsizedHex)
        .expect("valid hexadecimal localparam");
    let binary = file
        .make_literal("bits[32]:0b1010", &LiteralFormat::UnsizedBinary)
        .expect("valid binary localparam");
    file.add_localparam(module, "PlainDecimal", &decimal);
    file.add_localparam(module, "PlainHex", &hex);
    file.add_localparam(module, "PlainBinary", &binary);

    let expected = r#"module top;
  localparam PlainDecimal = 42;
  localparam PlainHex = 'h42;
  localparam PlainBinary = 'b1010;
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn upstream_array_parameter_patterns_preserve_nested_types_and_dimensions() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("top");
    let scalar = file.make_scalar_type();
    let byte = file.make_bit_vector_type(8, false);
    let one = file.make_unsized_decimal_literal(1);
    let two = file.make_unsized_decimal_literal(2);
    let three = file.make_unsized_decimal_literal(3);
    let four = file.make_unsized_decimal_literal(4);
    let five = file.make_unsized_decimal_literal(5);
    let six = file.make_unsized_decimal_literal(6);

    let p1_type = file.make_unpacked_array_type(scalar, &[3]);
    let p1_definition = file.make_def("P1", DataKind::Int, &p1_type);
    let p1_value = file.make_array_assignment_pattern(&[&one, &two, &three]);
    file.add_parameter_with_def(module, &p1_definition, &p1_value);

    let p2_type = file.make_unpacked_array_type(byte, &[2]);
    let p2_definition = file.make_def("P2", DataKind::Logic, &p2_type);
    let hex_42 = file
        .make_literal("bits[8]:0x42", &LiteralFormat::Hex)
        .expect("valid byte literal");
    let hex_43 = file
        .make_literal("bits[8]:0x43", &LiteralFormat::Hex)
        .expect("valid byte literal");
    let p2_value = file.make_array_assignment_pattern(&[&hex_42, &hex_43]);
    file.add_parameter_with_def(module, &p2_definition, &p2_value);

    let p4_type = file.make_unpacked_array_type(scalar, &[2, 3]);
    let p4_definition = file.make_def("P4", DataKind::Int, &p4_type);
    let first_row = file.make_array_assignment_pattern(&[&one, &two, &three]);
    let second_row = file.make_array_assignment_pattern(&[&four, &five, &six]);
    let p4_value = file.make_array_assignment_pattern(&[&first_row, &second_row]);
    file.add_parameter_with_def(module, &p4_definition, &p4_value);

    let expected = r#"module top;
  parameter int P1[3] = '{1, 2, 3};
  parameter logic [7:0] P2[2] = '{8'h42, 8'h43};
  parameter int P4[2][3] = '{'{1, 2, 3}, '{4, 5, 6}};
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn parameter_ports_without_io_match_the_upstream_complete_header() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("top");
    let byte = file.make_bit_vector_type(8, false);
    let five = file.make_unsized_decimal_literal(5);
    let seven = file.make_unsized_decimal_literal(7);
    file.add_typed_parameter_port(module, "TypedParam", &byte, &five);
    file.add_parameter_port(module, "UntypedParam", &seven);

    let expected = r#"module top #(
  parameter logic [7:0] TypedParam = 5,
  parameter UntypedParam = 7
);

endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn package_types_and_packed_arrays_suppress_wire_port_keywords() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("top");
    let external = file.make_extern_type("color_e");
    let package = file.make_extern_package_type("mypack", "mystruct_t");
    let packed = file.make_packed_array_type(package, &[2, 3, 4]);
    file.add_input(module, "color", &external);
    file.add_output(module, "payload", &packed);

    let expected = r#"module top(
  input color_e color,
  output mypack::mystruct_t [1:0][2:0][3:0] payload
);

endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn posedge_and_negedge_sensitivity_expressions_emit_exactly() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("edges");
    let scalar = file.make_scalar_type();
    let clock = file.add_input(module, "clk", &scalar);
    let reset = file.add_input(module, "rst_n", &scalar);
    let rising = file.make_pos_edge(&clock.to_expr());
    let falling = file.make_neg_edge(&reset.to_expr());

    assert_eq!(file.emit_expression(&rising), "posedge clk");
    assert_eq!(file.emit_expression(&falling), "negedge rst_n");
    file.add_always_ff(module, &[&rising, &falling])
        .expect("edge expressions are valid sensitivity elements");
    let expected = r#"module edges(
  input wire clk,
  input wire rst_n
);
  always_ff @ (posedge clk or negedge rst_n) begin end
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn macro_references_distinguish_missing_and_empty_argument_lists() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let absent = file.make_macro_ref("NO_ARGS");
    let empty = file.make_macro_ref_with_args("EMPTY_ARGS", &[]);
    let one = file.make_unsized_decimal_literal(1);
    let two = file.make_unsized_decimal_literal(2);
    let populated = file.make_macro_ref_with_args("VALUES", &[&one, &two]);

    assert_eq!(file.emit_expression(&absent.to_expr()), "`NO_ARGS");
    assert_eq!(file.emit_expression(&empty.to_expr()), "`EMPTY_ARGS()");
    assert_eq!(file.emit_expression(&populated.to_expr()), "`VALUES(1, 2)");
}
