// SPDX-License-Identifier: Apache-2.0

//! Public-API golden coverage for Verilog operator precedence and
//! associativity.

use xlsynth_vast::{Expr, VastFile, VastFileType, VastModule};

/// Groups one module and its four independently addressable operands.
struct Operands {
    module: VastModule,
    a: Expr,
    b: Expr,
    c: Expr,
    d: Expr,
}

/// Creates a module containing four identically typed named inputs.
fn named_operands(file: &mut VastFile) -> Operands {
    let module = file.add_module("precedence");
    let byte = file.make_bit_vector_type(8, false);
    Operands {
        module,
        a: file.add_input(module, "a", &byte).to_expr(),
        b: file.add_input(module, "b", &byte).to_expr(),
        c: file.add_input(module, "c", &byte).to_expr(),
        d: file.add_input(module, "d", &byte).to_expr(),
    }
}

/// Appends a named eight-bit output and its complete continuous assignment.
fn add_output_assignment(file: &mut VastFile, module: VastModule, name: &str, value: &Expr) {
    let byte = file.make_bit_vector_type(8, false);
    let output = file.add_output(module, name, &byte);
    let assignment = file.make_continuous_assignment(&output.to_expr(), value);
    file.add_member_continuous_assignment(module, assignment);
}

#[test]
fn unary_operators_apply_to_cast_values_not_cast_widths() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let Operands { a, .. } = named_operands(&mut file);
    let width = file.make_unsized_decimal_literal(4);
    let cast = file.make_width_cast(&width, &a);
    for (expression, expected) in [
        (file.make_negate(&cast), "-(4'(a))"),
        (file.make_bitwise_not(&cast), "~(4'(a))"),
        (file.make_logical_not(&cast), "!(4'(a))"),
        (file.make_and_reduce(&cast), "&(4'(a))"),
        (file.make_or_reduce(&cast), "|(4'(a))"),
        (file.make_xor_reduce(&cast), "^(4'(a))"),
    ] {
        assert_eq!(file.emit_expression(&expression), expected);
    }
}

#[test]
fn exponentiation_is_left_associative_in_both_verilog_dialects() {
    for file_type in [VastFileType::Verilog, VastFileType::SystemVerilog] {
        let mut file = VastFile::new(file_type);
        let Operands { a, b, c, .. } = named_operands(&mut file);

        let left_power = file.make_power(&a, &b);
        let left_nested = file.make_power(&left_power, &c);
        let right_power = file.make_power(&b, &c);
        let right_nested = file.make_power(&a, &right_power);

        assert_eq!(file.emit_expression(&left_nested), "a ** b ** c");
        assert_eq!(file.emit_expression(&right_nested), "a ** (b ** c)");
    }
}

#[test]
fn exponentiation_chains_preserve_every_explicit_nondefault_grouping() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let Operands { a, b, c, d, .. } = named_operands(&mut file);

    let a_power_b = file.make_power(&a, &b);
    let left_three = file.make_power(&a_power_b, &c);
    let left_four = file.make_power(&left_three, &d);
    assert_eq!(file.emit_expression(&left_four), "a ** b ** c ** d");

    let c_power_d = file.make_power(&c, &d);
    let right_three = file.make_power(&b, &c_power_d);
    let right_four = file.make_power(&a, &right_three);
    assert_eq!(file.emit_expression(&right_four), "a ** (b ** (c ** d))");

    let balanced = file.make_power(&a_power_b, &c_power_d);
    assert_eq!(file.emit_expression(&balanced), "a ** b ** (c ** d)");

    let b_power_c = file.make_power(&b, &c);
    let nested_right_then_left = file.make_power(&a, &b_power_c);
    let nested_right_then_left = file.make_power(&nested_right_then_left, &d);
    assert_eq!(
        file.emit_expression(&nested_right_then_left),
        "a ** (b ** c) ** d"
    );

    let nested_left_then_right = file.make_power(&b_power_c, &d);
    let nested_left_then_right = file.make_power(&a, &nested_left_then_right);
    assert_eq!(
        file.emit_expression(&nested_left_then_right),
        "a ** (b ** c ** d)"
    );
}

#[test]
fn literal_exponentiation_distinguishes_left_and_right_evaluation_shapes() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let two = file.make_unsized_decimal_literal(2);
    let three = file.make_unsized_decimal_literal(3);

    let two_power_three = file.make_power(&two, &three);
    let left_nested = file.make_power(&two_power_three, &two);
    let three_power_two = file.make_power(&three, &two);
    let right_nested = file.make_power(&two, &three_power_two);

    assert_eq!(file.emit_expression(&left_nested), "2 ** 3 ** 2");
    assert_eq!(file.emit_expression(&right_nested), "2 ** (3 ** 2)");
}

#[test]
fn exponentiation_binds_more_tightly_than_multiplication_and_division() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let Operands { a, b, c, .. } = named_operands(&mut file);

    let a_power_b = file.make_power(&a, &b);
    let power_times_c = file.make_mul(&a_power_b, &c);
    let power_divided_by_c = file.make_div(&a_power_b, &c);
    assert_eq!(file.emit_expression(&power_times_c), "a ** b * c");
    assert_eq!(file.emit_expression(&power_divided_by_c), "a ** b / c");

    let b_power_c = file.make_power(&b, &c);
    let a_times_power = file.make_mul(&a, &b_power_c);
    let a_divided_by_power = file.make_div(&a, &b_power_c);
    assert_eq!(file.emit_expression(&a_times_power), "a * b ** c");
    assert_eq!(file.emit_expression(&a_divided_by_power), "a / b ** c");

    let a_times_b = file.make_mul(&a, &b);
    let product_base = file.make_power(&a_times_b, &c);
    let b_times_c = file.make_mul(&b, &c);
    let product_exponent = file.make_power(&a, &b_times_c);
    assert_eq!(file.emit_expression(&product_base), "(a * b) ** c");
    assert_eq!(file.emit_expression(&product_exponent), "a ** (b * c)");

    let a_divided_by_b = file.make_div(&a, &b);
    let quotient_base = file.make_power(&a_divided_by_b, &c);
    let b_divided_by_c = file.make_div(&b, &c);
    let quotient_exponent = file.make_power(&a, &b_divided_by_c);
    assert_eq!(file.emit_expression(&quotient_base), "(a / b) ** c");
    assert_eq!(file.emit_expression(&quotient_exponent), "a ** (b / c)");
}

#[test]
fn exponentiation_binds_more_tightly_than_addition_and_subtraction() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let Operands { a, b, c, .. } = named_operands(&mut file);

    let a_power_b = file.make_power(&a, &b);
    let power_plus_c = file.make_add(&a_power_b, &c);
    let power_minus_c = file.make_sub(&a_power_b, &c);
    assert_eq!(file.emit_expression(&power_plus_c), "a ** b + c");
    assert_eq!(file.emit_expression(&power_minus_c), "a ** b - c");

    let b_power_c = file.make_power(&b, &c);
    let a_plus_power = file.make_add(&a, &b_power_c);
    let a_minus_power = file.make_sub(&a, &b_power_c);
    assert_eq!(file.emit_expression(&a_plus_power), "a + b ** c");
    assert_eq!(file.emit_expression(&a_minus_power), "a - b ** c");

    let a_plus_b = file.make_add(&a, &b);
    let sum_base = file.make_power(&a_plus_b, &c);
    let b_plus_c = file.make_add(&b, &c);
    let sum_exponent = file.make_power(&a, &b_plus_c);
    assert_eq!(file.emit_expression(&sum_base), "(a + b) ** c");
    assert_eq!(file.emit_expression(&sum_exponent), "a ** (b + c)");
}

#[test]
fn unary_operations_preserve_their_precedence_around_exponentiation() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let Operands { a, b, c, .. } = named_operands(&mut file);

    let negative_a = file.make_negate(&a);
    let negative_base = file.make_power(&negative_a, &b);
    let negative_b = file.make_negate(&b);
    let negative_exponent = file.make_power(&a, &negative_b);
    assert_eq!(file.emit_expression(&negative_base), "-a ** b");
    assert_eq!(file.emit_expression(&negative_exponent), "a ** -b");

    let a_power_b = file.make_power(&a, &b);
    let negated_power = file.make_negate(&a_power_b);
    let inverted_power = file.make_bitwise_not(&a_power_b);
    let logically_inverted_power = file.make_logical_not(&a_power_b);
    assert_eq!(file.emit_expression(&negated_power), "-(a ** b)");
    assert_eq!(file.emit_expression(&inverted_power), "~(a ** b)");
    assert_eq!(file.emit_expression(&logically_inverted_power), "!(a ** b)");

    let b_power_c = file.make_power(&b, &c);
    let negative_power = file.make_negate(&b_power_c);
    let power_with_negative_power = file.make_power(&a, &negative_power);
    assert_eq!(
        file.emit_expression(&power_with_negative_power),
        "a ** -(b ** c)"
    );
}

#[test]
fn reduction_operands_keep_their_mandatory_binary_parentheses() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let Operands { a, b, c, .. } = named_operands(&mut file);

    let reduced_a = file.make_or_reduce(&a);
    let reduced_b = file.make_and_reduce(&b);
    let reduced_c = file.make_xor_reduce(&c);

    let reduced_base = file.make_power(&reduced_a, &b);
    let reduced_exponent = file.make_power(&a, &reduced_b);
    let two_reductions = file.make_power(&reduced_a, &reduced_b);
    assert_eq!(file.emit_expression(&reduced_base), "(|a) ** b");
    assert_eq!(file.emit_expression(&reduced_exponent), "a ** (&b)");
    assert_eq!(file.emit_expression(&two_reductions), "(|a) ** (&b)");

    let a_power_b = file.make_power(&a, &b);
    let reduced_power = file.make_or_reduce(&a_power_b);
    assert_eq!(file.emit_expression(&reduced_power), "|(a ** b)");

    let reduction_power = file.make_power(&reduced_b, &reduced_c);
    let nested_reduction_power = file.make_power(&a, &reduction_power);
    assert_eq!(
        file.emit_expression(&nested_reduction_power),
        "a ** ((&b) ** (^c))"
    );
}

#[test]
fn conditional_operands_are_grouped_only_inside_exponentiation() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let Operands { a, b, c, d, .. } = named_operands(&mut file);

    let a_power_b = file.make_power(&a, &b);
    let power_condition = file.make_ternary(&a_power_b, &c, &d);
    let b_power_c = file.make_power(&b, &c);
    let power_consequent = file.make_ternary(&a, &b_power_c, &d);
    let c_power_d = file.make_power(&c, &d);
    let power_alternate = file.make_ternary(&a, &b, &c_power_d);
    assert_eq!(file.emit_expression(&power_condition), "a ** b ? c : d");
    assert_eq!(file.emit_expression(&power_consequent), "a ? b ** c : d");
    assert_eq!(file.emit_expression(&power_alternate), "a ? b : c ** d");

    let conditional = file.make_ternary(&a, &b, &c);
    let conditional_base = file.make_power(&conditional, &d);
    let conditional_exponent = file.make_power(&a, &conditional);
    assert_eq!(file.emit_expression(&conditional_base), "(a ? b : c) ** d");
    assert_eq!(
        file.emit_expression(&conditional_exponent),
        "a ** (a ? b : c)"
    );
}

#[test]
fn subtraction_and_division_remain_left_associative() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let Operands { a, b, c, .. } = named_operands(&mut file);

    let a_minus_b = file.make_sub(&a, &b);
    let left_difference = file.make_sub(&a_minus_b, &c);
    let b_minus_c = file.make_sub(&b, &c);
    let right_difference = file.make_sub(&a, &b_minus_c);
    assert_eq!(file.emit_expression(&left_difference), "a - b - c");
    assert_eq!(file.emit_expression(&right_difference), "a - (b - c)");

    let a_divided_by_b = file.make_div(&a, &b);
    let left_quotient = file.make_div(&a_divided_by_b, &c);
    let b_divided_by_c = file.make_div(&b, &c);
    let right_quotient = file.make_div(&a, &b_divided_by_c);
    assert_eq!(file.emit_expression(&left_quotient), "a / b / c");
    assert_eq!(file.emit_expression(&right_quotient), "a / (b / c)");
}

#[test]
fn mixed_equal_precedence_operators_keep_only_required_parentheses() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let Operands { a, b, c, .. } = named_operands(&mut file);

    let a_plus_b = file.make_add(&a, &b);
    let left_sum_difference = file.make_sub(&a_plus_b, &c);
    let b_plus_c = file.make_add(&b, &c);
    let right_sum_difference = file.make_sub(&a, &b_plus_c);
    assert_eq!(file.emit_expression(&left_sum_difference), "a + b - c");
    assert_eq!(file.emit_expression(&right_sum_difference), "a - (b + c)");

    let a_minus_b = file.make_sub(&a, &b);
    let left_difference_sum = file.make_add(&a_minus_b, &c);
    let b_minus_c = file.make_sub(&b, &c);
    let right_difference_sum = file.make_add(&a, &b_minus_c);
    assert_eq!(file.emit_expression(&left_difference_sum), "a - b + c");
    assert_eq!(file.emit_expression(&right_difference_sum), "a + (b - c)");

    let a_times_b = file.make_mul(&a, &b);
    let left_product_quotient = file.make_div(&a_times_b, &c);
    let b_times_c = file.make_mul(&b, &c);
    let right_product_quotient = file.make_div(&a, &b_times_c);
    assert_eq!(file.emit_expression(&left_product_quotient), "a * b / c");
    assert_eq!(file.emit_expression(&right_product_quotient), "a / (b * c)");

    let a_divided_by_b = file.make_div(&a, &b);
    let left_quotient_product = file.make_mul(&a_divided_by_b, &c);
    let b_divided_by_c = file.make_div(&b, &c);
    let right_quotient_product = file.make_mul(&a, &b_divided_by_c);
    assert_eq!(file.emit_expression(&left_quotient_product), "a / b * c");
    assert_eq!(file.emit_expression(&right_quotient_product), "a * (b / c)");

    let a_modulo_b = file.make_mod(&a, &b);
    let left_remainder_product = file.make_mul(&a_modulo_b, &c);
    let b_modulo_c = file.make_mod(&b, &c);
    let right_remainder_product = file.make_mul(&a, &b_modulo_c);
    assert_eq!(file.emit_expression(&left_remainder_product), "a % b * c");
    assert_eq!(
        file.emit_expression(&right_remainder_product),
        "a * (b % c)"
    );
}

#[test]
fn complete_module_preserves_associativity_and_mixed_operator_precedence() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let Operands { module, a, b, c, d } = named_operands(&mut file);

    let a_power_b = file.make_power(&a, &b);
    let left_power = file.make_power(&a_power_b, &c);
    let b_power_c = file.make_power(&b, &c);
    let right_power = file.make_power(&a, &b_power_c);
    let sum = file.make_add(&a, &b);
    let sum_power = file.make_power(&sum, &c);
    let mixed_power = file.make_mul(&sum_power, &d);
    let reduced_a = file.make_or_reduce(&a);
    let reduced_b = file.make_and_reduce(&b);
    let reduced_power = file.make_power(&reduced_a, &reduced_b);
    let conditional_power = file.make_ternary(&a_power_b, &c, &d);

    add_output_assignment(&mut file, module, "left_power", &left_power);
    add_output_assignment(&mut file, module, "right_power", &right_power);
    add_output_assignment(&mut file, module, "mixed_power", &mixed_power);
    add_output_assignment(&mut file, module, "reduced_power", &reduced_power);
    add_output_assignment(&mut file, module, "conditional_power", &conditional_power);

    let expected = r#"module precedence(
  input wire [7:0] a,
  input wire [7:0] b,
  input wire [7:0] c,
  input wire [7:0] d,
  output wire [7:0] left_power,
  output wire [7:0] right_power,
  output wire [7:0] mixed_power,
  output wire [7:0] reduced_power,
  output wire [7:0] conditional_power
);
  assign left_power = a ** b ** c;
  assign right_power = a ** (b ** c);
  assign mixed_power = (a + b) ** c * d;
  assign reduced_power = (|a) ** (&b);
  assign conditional_power = a ** b ? c : d;
endmodule
"#;
    assert_eq!(file.emit(), expected);
}
