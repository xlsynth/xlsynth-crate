// SPDX-License-Identifier: Apache-2.0

//! Public regressions for type emission, literal bounds, and helper validation.

use xlsynth_vast::helpers::{
    RegisterDefinition, RegisterScope, add_registers, bitwise_or_reduce,
    bitwise_or_reduce_array_elements, logical_and_reduce, logical_or_reduce,
};
use xlsynth_vast::{Expr, IndexableExpr, LiteralFormat, VastFile, VastFileType};

/// Adds a named scalar expression belonging to the supplied file.
fn named_scalar(file: &mut VastFile, name: &str) -> Expr {
    let module = file.add_module("source");
    let scalar = file.make_scalar_type();
    file.add_input(module, name, &scalar).to_expr()
}

/// Adds a two-dimensional packed-array input with dimensions `[2][3]`.
fn packed_array_input(file: &mut VastFile) -> IndexableExpr {
    let module = file.add_module("array_source");
    let element = file.make_bit_vector_type(8, false);
    let array = file.make_packed_array_type(element, &[2, 3]);
    file.add_input(module, "array", &array).to_indexable_expr()
}

#[test]
fn zero_width_literals_are_rejected() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);

    assert!(file.make_literal("bits[0]:0", &LiteralFormat::Hex).is_err());
    assert!(
        file.make_literal("bits[0]:1", &LiteralFormat::UnsignedDecimal)
            .is_err()
    );
}

#[test]
fn excessive_literal_widths_are_rejected_without_large_allocations() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);

    assert!(
        file.make_literal("bits[1048577]:0", &LiteralFormat::UnsignedDecimal)
            .is_err()
    );
    assert!(
        file.make_literal("bits[18446744073709551616]:0", &LiteralFormat::Hex)
            .is_err()
    );
}

#[test]
fn maximum_width_zero_and_small_positive_literals_remain_compact() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let zero = file
        .make_literal("bits[1048576]:0", &LiteralFormat::UnsignedDecimal)
        .expect("the maximum supported zero literal is representable");
    let positive = file
        .make_literal("bits[1048576]:1", &LiteralFormat::UnsignedDecimal)
        .expect("the maximum supported positive literal is representable");

    assert_eq!(file.emit_expression(&zero), "1048576'd0");
    assert_eq!(file.emit_expression(&positive), "1048576'd1");
}

#[test]
#[should_panic(expected = "bit-vector width must be greater than zero")]
fn expression_based_vector_types_reject_zero_literal_widths() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let zero = file.make_unsized_decimal_literal(0);

    file.make_bit_vector_type_expr(&zero, false);
}

#[test]
#[should_panic(expected = "bit-vector width must be greater than zero")]
fn expression_based_vector_types_reject_negative_literal_widths() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let negative = file.make_unsized_decimal_literal(-1);

    file.make_bit_vector_type_expr(&negative, false);
}

#[test]
fn expression_based_single_bit_vectors_preserve_explicit_ranges() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("explicit_one_bit");
    let one = file.make_unsized_decimal_literal(1);
    let vector = file.make_bit_vector_type_expr(&one, false);
    file.add_input(module, "data", &vector);

    let expected = r#"module explicit_one_bit(
  input wire [0:0] data
);

endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn typed_parameter_ports_preserve_builtin_and_user_defined_type_identity() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("typed_parameters");
    let scalar = file.make_scalar_type();
    let signed_vector = file.make_bit_vector_type(8, true);
    let signed_integer = file.make_integer_type(true);
    let unsigned_integer = file.make_integer_type(false);
    let signed_int = file.make_int_type(true);
    let unsigned_int = file.make_int_type(false);
    let local_type = file.make_extern_type("payload_t");
    let package_type = file.make_extern_package_type("bus_pkg", "payload_t");
    let one = file.make_unsized_decimal_literal(1);

    for (name, data_type) in [
        ("Flag", scalar),
        ("SignedVector", signed_vector),
        ("LegacyInteger", signed_integer),
        ("LegacyUnsigned", unsigned_integer),
        ("SystemInt", signed_int),
        ("SystemUnsigned", unsigned_int),
        ("LocalPayload", local_type),
        ("PackagePayload", package_type),
    ] {
        file.add_typed_parameter_port(module, name, &data_type, &one);
    }

    let expected = r#"module typed_parameters #(
  parameter logic Flag = 1,
  parameter logic signed [7:0] SignedVector = 1,
  parameter integer LegacyInteger = 1,
  parameter integer unsigned LegacyUnsigned = 1,
  parameter int SystemInt = 1,
  parameter int unsigned SystemUnsigned = 1,
  parameter payload_t LocalPayload = 1,
  parameter bus_pkg::payload_t PackagePayload = 1
);

endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn builtin_and_external_type_casts_preserve_type_and_signedness() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let value = named_scalar(&mut file, "value");
    let scalar = file.make_scalar_type();
    let signed_scalar = file.make_bit_vector_type(1, true);
    let unsigned_vector = file.make_bit_vector_type(8, false);
    let signed_vector = file.make_bit_vector_type(8, true);
    let signed_integer = file.make_integer_type(true);
    let unsigned_integer = file.make_integer_type(false);
    let signed_int = file.make_int_type(true);
    let unsigned_int = file.make_int_type(false);
    let local_type = file.make_extern_type("payload_t");
    let package_type = file.make_extern_package_type("bus_pkg", "payload_t");

    let cases = [
        (scalar, "logic'(value)"),
        (signed_scalar, "signed'(logic'(value))"),
        (unsigned_vector, "unsigned'(8'(value))"),
        (signed_vector, "signed'(8'(value))"),
        (signed_integer, "integer'(value)"),
        (unsigned_integer, "unsigned'(integer'(value))"),
        (signed_int, "int'(value)"),
        (unsigned_int, "unsigned'(int'(value))"),
        (local_type, "payload_t'(value)"),
        (package_type, "bus_pkg::payload_t'(value)"),
    ];

    for (data_type, expected) in cases {
        let cast = file.make_type_cast(&data_type, &value);
        assert_eq!(file.emit_expression(&cast), expected);
    }
}

#[test]
fn vector_type_casts_preserve_symbolic_width_expressions() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("symbolic_cast");
    let default_width = file.make_unsized_decimal_literal(8);
    let width = file.add_parameter_port(module, "WIDTH", &default_width);
    let one = file.make_unsized_decimal_literal(1);
    let width_plus_one = file.make_add(&width.to_expr(), &one);
    let signed_width = file.make_bit_vector_type_expr(&width.to_expr(), true);
    let unsigned_width_plus_one = file.make_bit_vector_type_expr(&width_plus_one, false);
    let value = file.add_input(module, "value", &signed_width);

    let signed_cast = file.make_type_cast(&signed_width, &value.to_expr());
    let unsigned_cast = file.make_type_cast(&unsigned_width_plus_one, &value.to_expr());

    assert_eq!(file.emit_expression(&signed_cast), "signed'(WIDTH'(value))");
    assert_eq!(
        file.emit_expression(&unsigned_cast),
        "unsigned'((WIDTH + 1)'(value))"
    );
}

#[test]
fn register_groups_without_required_resets_return_errors_without_mutation() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("missing_reset");
    let scalar = file.make_scalar_type();
    let clock = file.add_input(module, "clock", &scalar);
    let data = file.add_input(module, "data", &scalar);
    let register = file.add_logic(module, "state", &scalar).unwrap();
    let reset_value = file.make_literal("bits[1]:0", &LiteralFormat::Hex).unwrap();
    let before = file.emit();

    let error = add_registers(
        &clock.to_expr(),
        None,
        &[RegisterDefinition {
            reg: register.to_expr(),
            next: data.to_expr(),
            reset_value: Some(reset_value),
            enable: None,
        }],
        RegisterScope::Module(module),
        &mut file,
    )
    .expect_err("registers with reset values require reset signals");

    assert_eq!(
        error.to_string(),
        "reset signal is required when a register has a reset value"
    );
    assert_eq!(file.emit(), before);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn singleton_logical_or_reductions_reject_foreign_handles() {
    let mut source = VastFile::new(VastFileType::SystemVerilog);
    let foreign = named_scalar(&mut source, "foreign");
    let mut destination = VastFile::new(VastFileType::SystemVerilog);

    let _ = logical_or_reduce(&[foreign], false, &mut destination);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn singleton_logical_and_reductions_reject_foreign_handles() {
    let mut source = VastFile::new(VastFileType::SystemVerilog);
    let foreign = named_scalar(&mut source, "foreign");
    let mut destination = VastFile::new(VastFileType::SystemVerilog);

    let _ = logical_and_reduce(&[foreign], false, &mut destination);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn singleton_bitwise_or_reductions_reject_foreign_handles() {
    let mut source = VastFile::new(VastFileType::SystemVerilog);
    let foreign = named_scalar(&mut source, "foreign");
    let mut destination = VastFile::new(VastFileType::SystemVerilog);

    let _ = bitwise_or_reduce(&[foreign], &mut destination);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn array_reductions_reject_foreign_subjects_before_validating_dimensions() {
    let mut source = VastFile::new(VastFileType::SystemVerilog);
    let foreign = packed_array_input(&mut source);
    let mut destination = VastFile::new(VastFileType::SystemVerilog);

    let _ = bitwise_or_reduce_array_elements(&foreign, &[], &mut destination);
}

#[test]
fn array_reductions_reject_empty_zero_and_negative_dimensions() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let array = packed_array_input(&mut file);

    let empty = bitwise_or_reduce_array_elements(&array, &[], &mut file)
        .expect_err("packed-array reductions require explicit dimensions");
    assert_eq!(
        empty.to_string(),
        "packed-array reduction requires at least one dimension"
    );

    for invalid_dimensions in [[0, 3], [2, -1]] {
        let error = bitwise_or_reduce_array_elements(&array, &invalid_dimensions, &mut file)
            .expect_err("packed-array dimensions must be strictly positive");
        assert_eq!(
            error.to_string(),
            "packed-array dimensions must be greater than zero"
        );
    }
}

#[test]
fn array_reductions_reject_shapes_that_disagree_with_the_declared_type() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let array = packed_array_input(&mut file);

    let error = bitwise_or_reduce_array_elements(&array, &[2, 4], &mut file)
        .expect_err("reduction shape must match the declared packed array");

    assert_eq!(
        error.to_string(),
        "packed-array dimensions do not match expression type: expected [2, 3], got [2, 4]"
    );
}
