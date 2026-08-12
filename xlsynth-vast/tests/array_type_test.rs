// SPDX-License-Identifier: Apache-2.0

//! Public golden tests for nested array types, dimensions, and cast validity.

use std::panic::{AssertUnwindSafe, catch_unwind};

use xlsynth_vast::helpers::bitwise_or_reduce_array_elements;
use xlsynth_vast::{VastFile, VastFileType};

#[test]
fn nested_unpacked_array_wrappers_emit_outer_dimensions_first_in_both_dialects() {
    for (dialect, expected_dimensions) in [
        (VastFileType::Verilog, "[0:3][0:4][0:1][0:2]"),
        (VastFileType::SystemVerilog, "[4][5][2][3]"),
    ] {
        let mut file = VastFile::new(dialect);
        let module = file.add_module("nested_unpacked");
        let byte = file.make_bit_vector_type(8, false);
        let inner = file.make_unpacked_array_type(byte, &[2, 3]);
        let outer = file.make_unpacked_array_type(inner, &[4, 5]);
        file.add_wire(module, "data", &outer);

        let expected = format!(
            "module nested_unpacked;\n  wire [7:0] data{expected_dimensions};\nendmodule\n"
        );
        assert_eq!(file.emit(), expected);
    }
}

#[test]
fn nested_packed_array_wrappers_emit_outer_dimensions_first_in_both_dialects() {
    for dialect in [VastFileType::Verilog, VastFileType::SystemVerilog] {
        let mut file = VastFile::new(dialect);
        let module = file.add_module("nested_packed");
        let byte = file.make_bit_vector_type(8, false);
        let inner = file.make_packed_array_type(byte, &[3]);
        let outer = file.make_packed_array_type(inner, &[2]);
        file.add_wire(module, "data", &outer);

        let expected = r#"module nested_packed;
  wire [1:0][2:0][7:0] data;
endmodule
"#;
        assert_eq!(file.emit(), expected);
    }
}

#[test]
fn nested_packed_arrays_preserve_signed_vector_and_scalar_element_types() {
    for dialect in [VastFileType::Verilog, VastFileType::SystemVerilog] {
        let mut file = VastFile::new(dialect);
        let module = file.add_module("signed_arrays");

        let signed_byte = file.make_bit_vector_type(8, true);
        let byte_inner = file.make_packed_array_type(signed_byte, &[3]);
        let byte_outer = file.make_packed_array_type(byte_inner, &[2]);
        file.add_wire(module, "signed_vector", &byte_outer);

        let signed_scalar = file.make_bit_vector_type(1, true);
        let scalar_inner = file.make_packed_array_type(signed_scalar, &[3]);
        let scalar_outer = file.make_packed_array_type(scalar_inner, &[2]);
        file.add_wire(module, "signed_scalar", &scalar_outer);

        assert!(file.type_is_signed(byte_outer));
        assert!(file.type_is_signed(scalar_outer));

        let expected = r#"module signed_arrays;
  wire signed [1:0][2:0][7:0] signed_vector;
  wire signed [1:0][2:0] signed_scalar;
endmodule
"#;
        assert_eq!(file.emit(), expected);
    }
}

#[test]
fn nested_packed_external_types_keep_one_type_separator_and_contiguous_dimensions() {
    for dialect in [VastFileType::Verilog, VastFileType::SystemVerilog] {
        let mut file = VastFile::new(dialect);
        let module = file.add_module("external_arrays");

        let local = file.make_extern_type("payload_t");
        let local_inner = file.make_packed_array_type(local, &[3]);
        let local_outer = file.make_packed_array_type(local_inner, &[2]);
        file.add_input(module, "request", &local_outer);

        let package = file.make_extern_package_type("bus_pkg", "response_t");
        let package_inner = file.make_packed_array_type(package, &[5]);
        let package_outer = file.make_packed_array_type(package_inner, &[4]);
        file.add_output(module, "response", &package_outer);
        file.add_wire(module, "internal", &package_outer);

        let expected = r#"module external_arrays(
  input payload_t [1:0][2:0] request,
  output bus_pkg::response_t [3:0][4:0] response
);
  wire bus_pkg::response_t [3:0][4:0] internal;
endmodule
"#;
        assert_eq!(file.emit(), expected);
    }
}

#[test]
fn multidimensional_wrappers_keep_packed_and_unpacked_dimensions_outermost_first() {
    for (dialect, expected_dimensions) in [
        (VastFileType::Verilog, "[0:6][0:7][0:4][0:5]"),
        (VastFileType::SystemVerilog, "[7][8][5][6]"),
    ] {
        let mut file = VastFile::new(dialect);
        let module = file.add_module("mixed_arrays");
        let byte = file.make_bit_vector_type(8, false);
        let packed_inner = file.make_packed_array_type(byte, &[2]);
        let packed_outer = file.make_packed_array_type(packed_inner, &[3, 4]);
        let unpacked_inner = file.make_unpacked_array_type(packed_outer, &[5, 6]);
        let unpacked_outer = file.make_unpacked_array_type(unpacked_inner, &[7, 8]);
        file.add_wire(module, "data", &unpacked_outer);

        let expected = format!(
            "module mixed_arrays;\n  wire [2:0][3:0][1:0][7:0] \
             data{expected_dimensions};\nendmodule\n"
        );
        assert_eq!(file.emit(), expected);
    }
}

#[test]
fn nested_array_dimensions_preserve_symbolic_signed_element_widths() {
    for (dialect, expected_dimensions) in [
        (VastFileType::Verilog, "[0:4][0:3]"),
        (VastFileType::SystemVerilog, "[5][4]"),
    ] {
        let mut file = VastFile::new(dialect);
        let module = file.add_module("symbolic_arrays");
        let nine = file.make_unsized_decimal_literal(9);
        let width = file.add_parameter_port(module, "WIDTH", &nine);
        let element = file.make_bit_vector_type_expr(&width.to_expr(), true);
        let packed_inner = file.make_packed_array_type(element, &[3]);
        let packed_outer = file.make_packed_array_type(packed_inner, &[2]);
        let unpacked_inner = file.make_unpacked_array_type(packed_outer, &[4]);
        let unpacked_outer = file.make_unpacked_array_type(unpacked_inner, &[5]);
        file.add_wire(module, "data", &unpacked_outer);

        assert!(file.type_is_signed(unpacked_outer));
        assert_eq!(file.type_width_expr(unpacked_outer), Some(width.to_expr()));
        assert!(file.type_width_as_int64(unpacked_outer).is_err());

        let expected = format!(
            "module symbolic_arrays #(\n  parameter WIDTH = 9\n);\n  \
             wire signed [1:0][2:0][WIDTH - 1:0] \
             data{expected_dimensions};\nendmodule\n"
        );
        assert_eq!(file.emit(), expected);
    }
}

#[test]
fn nested_array_indexing_follows_outermost_dimensions_before_inner_dimensions() {
    for (dialect, expected_dimensions) in [
        (VastFileType::Verilog, "[0:3][0:4][0:1][0:2]"),
        (VastFileType::SystemVerilog, "[4][5][2][3]"),
    ] {
        let mut file = VastFile::new(dialect);
        let module = file.add_module("indexed_arrays");
        let byte = file.make_bit_vector_type(8, false);
        let output = file.add_output(module, "out", &byte);
        let inner = file.make_unpacked_array_type(byte, &[2, 3]);
        let outer = file.make_unpacked_array_type(inner, &[4, 5]);
        let array = file
            .add_reg(module, "data", &outer)
            .expect("array register name is unique");
        let first = file.make_index(&array.to_indexable_expr(), 3);
        let second = file.make_index(&first.to_indexable_expr(), 4);
        let third = file.make_index(&second.to_indexable_expr(), 1);
        let fourth = file.make_index(&third.to_indexable_expr(), 2);
        let assignment = file.make_continuous_assignment(&output.to_expr(), &fourth.to_expr());
        file.add_member_continuous_assignment(module, assignment);

        let expected = format!(
            "module indexed_arrays(\n  output wire [7:0] out\n);\n  \
             reg [7:0] data{expected_dimensions};\n  \
             assign out = data[3][4][1][2];\nendmodule\n"
        );
        assert_eq!(file.emit(), expected);
    }
}

#[test]
fn nested_array_introspection_keeps_element_width_sign_and_total_flattened_width() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let element = file.make_bit_vector_type(5, true);
    let packed_inner = file.make_packed_array_type(element, &[2, 3]);
    let packed_outer = file.make_packed_array_type(packed_inner, &[4]);
    let unpacked_inner = file.make_unpacked_array_type(packed_outer, &[6]);
    let unpacked_outer = file.make_unpacked_array_type(unpacked_inner, &[7, 8]);

    assert_eq!(file.type_width_as_int64(unpacked_outer), Ok(5));
    assert_eq!(
        file.type_flat_bit_count_as_int64(unpacked_outer),
        Ok(5 * 2 * 3 * 4 * 6 * 7 * 8)
    );
    assert!(file.type_is_signed(unpacked_outer));
    let width = file
        .type_width_expr(unpacked_outer)
        .expect("nested arrays retain the element width");
    assert_eq!(file.emit_expression(&width), "5");
}

#[test]
fn nested_packed_array_reduction_uses_the_same_outer_first_dimension_order() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("nested_reduction");
    let scalar = file.make_scalar_type();
    let inner = file.make_packed_array_type(scalar, &[3]);
    let outer = file.make_packed_array_type(inner, &[2]);
    let input = file.add_input(module, "elements", &outer);
    let output = file.add_output(module, "reduced", &scalar);

    let reversed = bitwise_or_reduce_array_elements(&input.to_indexable_expr(), &[3, 2], &mut file)
        .expect_err("the shape must follow outermost dimensions first");
    assert_eq!(
        reversed.to_string(),
        "packed-array dimensions do not match expression type: expected [2, 3], got [3, 2]"
    );

    let reduction =
        bitwise_or_reduce_array_elements(&input.to_indexable_expr(), &[2, 3], &mut file)
            .expect("nested packed wrappers expose outer-first dimensions");
    let assignment = file.make_continuous_assignment(&output.to_expr(), &reduction);
    file.add_member_continuous_assignment(module, assignment);

    let expected = r#"module nested_reduction(
  input wire [1:0][2:0] elements,
  output wire reduced
);
  assign reduced = elements[0][0] | elements[0][1] | elements[0][2] | elements[1][0] | elements[1][1] | elements[1][2];
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn indexed_nested_packed_array_reduction_consumes_the_outer_dimension() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("indexed_reduction");
    let scalar = file.make_scalar_type();
    let inner = file.make_packed_array_type(scalar, &[3]);
    let outer = file.make_packed_array_type(inner, &[2]);
    let input = file.add_input(module, "elements", &outer);
    let row = file.make_index(&input.to_indexable_expr(), 1);

    let reduction = bitwise_or_reduce_array_elements(&row.to_indexable_expr(), &[3], &mut file)
        .expect("one array index consumes the outermost packed dimension");

    assert_eq!(
        file.emit_expression(&reduction),
        "elements[1][0] | elements[1][1] | elements[1][2]"
    );
}

#[test]
fn nested_packed_external_parameter_ports_keep_outermost_dimension_order() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("parameter_arrays");
    let package = file.make_extern_package_type("bus_pkg", "payload_t");
    let inner = file.make_packed_array_type(package, &[3]);
    let outer = file.make_packed_array_type(inner, &[2]);
    let zero = file.make_unsized_zero_literal();
    file.add_typed_parameter_port(module, "Payload", &outer, &zero);

    let expected = r#"module parameter_arrays #(
  parameter bus_pkg::payload_t [1:0][2:0] Payload = '0
);

endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn nested_unpacked_parameter_ports_keep_outermost_dimension_order() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("parameter_arrays");
    let byte = file.make_bit_vector_type(8, false);
    let inner = file.make_unpacked_array_type(byte, &[2]);
    let outer = file.make_unpacked_array_type(inner, &[3]);
    let one = file.make_unsized_decimal_literal(1);
    let two = file.make_unsized_decimal_literal(2);
    let row = file.make_array_assignment_pattern(&[&one, &two]);
    let matrix = file.make_array_assignment_pattern(&[&row, &row, &row]);
    file.add_typed_parameter_port(module, "Samples", &outer, &matrix);

    let expected = r#"module parameter_arrays #(
  parameter logic [7:0] Samples[3][2] = '{'{1, 2}, '{1, 2}, '{1, 2}}
);

endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn builtin_and_named_external_type_casts_remain_unchanged() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let value = file.make_unsized_decimal_literal(7);
    let scalar = file.make_scalar_type();
    let signed_vector = file.make_bit_vector_type(8, true);
    let signed_integer = file.make_integer_type(true);
    let unsigned_int = file.make_int_type(false);
    let local = file.make_extern_type("payload_t");
    let package = file.make_extern_package_type("bus_pkg", "payload_t");

    for (data_type, expected) in [
        (scalar, "logic'(7)"),
        (signed_vector, "signed'(8'(7))"),
        (signed_integer, "integer'(7)"),
        (unsigned_int, "unsigned'(int'(7))"),
        (local, "payload_t'(7)"),
        (package, "bus_pkg::payload_t'(7)"),
    ] {
        let cast = file.make_type_cast(&data_type, &value);
        assert_eq!(file.emit_expression(&cast), expected);
    }
}

#[test]
fn rejected_unpacked_scalar_and_vector_casts_leave_the_file_unchanged() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("unchanged_after_invalid_cast");
    let scalar = file.make_scalar_type();
    file.add_wire(module, "data", &scalar);
    let byte = file.make_bit_vector_type(8, false);
    let unpacked_scalar = file.make_unpacked_array_type(scalar, &[2]);
    let unpacked_vector = file.make_unpacked_array_type(byte, &[3]);
    let value = file.make_unsized_decimal_literal(7);
    let before = file.emit();

    for target in [unpacked_scalar, unpacked_vector] {
        let panic = catch_unwind(AssertUnwindSafe(|| file.make_type_cast(&target, &value)))
            .expect_err("unpacked arrays cannot be expression-cast target types");
        let message = panic
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
            .expect("cast rejection contains a panic message");
        assert_eq!(message, "unpacked array types cannot be used in type casts");
        assert_eq!(file.emit(), before);
    }
}

#[test]
fn rejected_packed_scalar_vector_nested_and_external_casts_leave_the_file_unchanged() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("unchanged_after_invalid_packed_cast");
    let scalar = file.make_scalar_type();
    file.add_wire(module, "data", &scalar);
    let signed_byte = file.make_bit_vector_type(8, true);
    let external = file.make_extern_package_type("bus_pkg", "payload_t");
    let packed_scalar = file.make_packed_array_type(scalar, &[2]);
    let packed_vector = file.make_packed_array_type(signed_byte, &[3]);
    let nested_packed = file.make_packed_array_type(packed_vector, &[4]);
    let packed_external = file.make_packed_array_type(external, &[5]);
    let value = file.make_unsized_decimal_literal(7);
    let before = file.emit();

    for target in [packed_scalar, packed_vector, nested_packed, packed_external] {
        let panic = catch_unwind(AssertUnwindSafe(|| file.make_type_cast(&target, &value)))
            .expect_err("anonymous packed arrays cannot be expression-cast target types");
        let message = panic
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
            .expect("cast rejection contains a panic message");
        assert_eq!(
            message,
            "packed array types cannot be used in type casts; use a named typedef instead"
        );
        assert_eq!(file.emit(), before);
    }
}

#[test]
#[should_panic(expected = "unpacked array types cannot be used in type casts")]
fn nested_unpacked_array_casts_are_rejected() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let byte = file.make_bit_vector_type(8, false);
    let inner = file.make_unpacked_array_type(byte, &[2]);
    let outer = file.make_unpacked_array_type(inner, &[3]);
    let value = file.make_unsized_decimal_literal(7);

    file.make_type_cast(&outer, &value);
}

#[test]
#[should_panic(expected = "packed arrays cannot contain unpacked array elements")]
fn packed_array_constructors_reject_unpacked_element_types() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let byte = file.make_bit_vector_type(8, false);
    let unpacked = file.make_unpacked_array_type(byte, &[2]);

    file.make_packed_array_type(unpacked, &[3]);
}

#[test]
fn packed_array_constructors_reject_integer_and_int_elements_without_mutation() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("unchanged_after_invalid_integer_array");
    let scalar = file.make_scalar_type();
    file.add_wire(module, "data", &scalar);
    let signed_integer = file.make_integer_type(true);
    let unsigned_integer = file.make_integer_type(false);
    let signed_int = file.make_int_type(true);
    let unsigned_int = file.make_int_type(false);
    let before = file.emit();

    for element in [signed_integer, unsigned_integer, signed_int, unsigned_int] {
        let panic = catch_unwind(AssertUnwindSafe(|| {
            file.make_packed_array_type(element, &[2])
        }))
        .expect_err("integer atom types cannot have packed dimensions");
        let message = panic
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
            .expect("constructor rejection contains a panic message");
        assert_eq!(
            message,
            "packed arrays cannot contain integer or int elements"
        );
        assert_eq!(file.emit(), before);
    }
}

#[test]
fn integer_and_int_elements_remain_valid_for_unpacked_array_types() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("integer_arrays");
    let signed_integer = file.make_integer_type(true);
    let unsigned_integer = file.make_integer_type(false);
    let signed_int = file.make_int_type(true);
    let unsigned_int = file.make_int_type(false);
    let signed_integer_array = file.make_unpacked_array_type(signed_integer, &[2]);
    let unsigned_integer_array = file.make_unpacked_array_type(unsigned_integer, &[2]);
    let signed_int_array = file.make_unpacked_array_type(signed_int, &[3]);
    let unsigned_int_array = file.make_unpacked_array_type(unsigned_int, &[3]);
    let zero = file.make_unsized_decimal_literal(0);

    file.add_typed_parameter_port(module, "SignedInteger", &signed_integer_array, &zero);
    file.add_typed_parameter_port(module, "UnsignedInteger", &unsigned_integer_array, &zero);
    file.add_typed_parameter_port(module, "SignedInt", &signed_int_array, &zero);
    file.add_typed_parameter_port(module, "UnsignedInt", &unsigned_int_array, &zero);

    assert_eq!(
        file.type_flat_bit_count_as_int64(signed_integer_array),
        Ok(64)
    );
    assert_eq!(
        file.type_flat_bit_count_as_int64(unsigned_integer_array),
        Ok(64)
    );
    assert_eq!(file.type_flat_bit_count_as_int64(signed_int_array), Ok(96));
    assert_eq!(
        file.type_flat_bit_count_as_int64(unsigned_int_array),
        Ok(96)
    );

    let expected = r#"module integer_arrays #(
  parameter integer SignedInteger[2] = 0,
  parameter integer unsigned UnsignedInteger[2] = 0,
  parameter int SignedInt[3] = 0,
  parameter int unsigned UnsignedInt[3] = 0
);

endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn packed_array_constructors_check_foreign_elements_before_array_validity() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let mut other = VastFile::new(VastFileType::SystemVerilog);
    let foreign_scalar = other.make_scalar_type();
    let foreign_unpacked = other.make_unpacked_array_type(foreign_scalar, &[2]);

    file.make_packed_array_type(foreign_unpacked, &[3]);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn packed_array_constructors_check_foreign_integers_before_element_validity() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let mut other = VastFile::new(VastFileType::SystemVerilog);
    let foreign_integer = other.make_integer_type(true);

    file.make_packed_array_type(foreign_integer, &[3]);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn unpacked_cast_target_ownership_is_checked_before_array_validity() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let mut other = VastFile::new(VastFileType::SystemVerilog);
    let foreign_scalar = other.make_scalar_type();
    let foreign_unpacked = other.make_unpacked_array_type(foreign_scalar, &[2]);
    let local_value = file.make_unsized_decimal_literal(7);

    file.make_type_cast(&foreign_unpacked, &local_value);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn packed_cast_target_ownership_is_checked_before_array_validity() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let mut other = VastFile::new(VastFileType::SystemVerilog);
    let foreign_byte = other.make_bit_vector_type(8, false);
    let foreign_packed = other.make_packed_array_type(foreign_byte, &[2]);
    let local_value = file.make_unsized_decimal_literal(7);

    file.make_type_cast(&foreign_packed, &local_value);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn cast_value_ownership_is_checked_before_unpacking_the_local_target_type() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let mut other = VastFile::new(VastFileType::SystemVerilog);
    let scalar = file.make_scalar_type();
    let unpacked = file.make_unpacked_array_type(scalar, &[2]);
    let foreign_value = other.make_unsized_decimal_literal(7);

    file.make_type_cast(&unpacked, &foreign_value);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn cast_value_ownership_is_checked_before_rejecting_the_local_packed_target() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let mut other = VastFile::new(VastFileType::SystemVerilog);
    let scalar = file.make_scalar_type();
    let packed = file.make_packed_array_type(scalar, &[2]);
    let foreign_value = other.make_unsized_decimal_literal(7);

    file.make_type_cast(&packed, &foreign_value);
}
