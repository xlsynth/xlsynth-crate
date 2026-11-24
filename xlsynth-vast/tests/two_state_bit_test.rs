// SPDX-License-Identifier: Apache-2.0

//! Native two-state SystemVerilog bit types and their emission semantics.

use xlsynth_vast::{ParameterKind, SvArgument, VastFile, VastFileType};

#[test]
fn bit_parameters_ports_and_array_declarations_preserve_the_two_state_keyword() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("TwoState");
    let logic = file.make_scalar_type();
    let bit = file.make_bit_type();
    let signed = file.make_two_state_bit_vector_type(8, true).unwrap();
    let array = file.make_packed_array_type(signed, &[2]).unwrap();
    let unknown = file.make_unsized_x_literal();
    file.add_value_parameter_port(
        module,
        "FOUR",
        ParameterKind::Parameter,
        Some(&logic),
        Some(&unknown),
    )
    .unwrap();
    file.add_value_parameter_port(
        module,
        "TWO",
        ParameterKind::Parameter,
        Some(&bit),
        Some(&unknown),
    )
    .unwrap();
    file.add_type_parameter_port(module, "Word", ParameterKind::Parameter, Some(&array))
        .unwrap();
    file.add_input(module, "request", &bit).unwrap();
    file.add_output(module, "response", &signed).unwrap();
    file.add_logic(module, "internal", &array).unwrap();
    assert_eq!(
        file.emit(),
        r#"module TwoState #(
  parameter logic FOUR = 'X,
  parameter bit TWO = 'X,
  parameter type Word = bit signed [1:0][7:0]
) (
  input bit request,
  output bit signed [7:0] response
);
  bit signed [1:0][7:0] internal;
endmodule
"#
    );
    assert_eq!(file.type_width_as_int64(bit), Ok(1));
    assert_eq!(file.type_flat_bit_count_as_int64(array), Ok(16));
    assert!(file.type_is_signed(signed));
    assert!(!file.type_is_signed(bit));
}

#[test]
fn bit_vector_casts_convert_unknown_bits_before_returning_the_target_width() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let source = file.make_four_state_binary_literal("10xz").unwrap();
    let bit = file.make_bit_type();
    let vector = file.make_two_state_bit_vector_type(4, true).unwrap();
    let scalar_cast = file.make_type_cast(&bit, &source).unwrap();
    let vector_cast = file.make_type_cast(&vector, &source).unwrap();
    assert_eq!(file.emit_expression(&scalar_cast), "bit'(4'b10xz)");
    assert_eq!(
        file.emit_expression(&vector_cast),
        "signed'(type(bit signed [3:0])'(4'b10xz))"
    );
    let bits = file
        .make_function_call_with_arguments("$bits", &[SvArgument::Type(vector)])
        .unwrap();
    assert_eq!(file.emit_expression(&bits), "$bits(bit signed [3:0])");
}

#[test]
fn symbolic_bit_vector_widths_keep_their_expression_and_sign() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Symbolic");
    let four = file.make_unsized_decimal_literal(4);
    let width = file.add_parameter_port(module, "WIDTH", &four).unwrap();
    let width_expression = width.to_expr();
    let data_type = file
        .make_two_state_bit_vector_type_expr(&width_expression, false)
        .unwrap();
    file.add_output(module, "data", &data_type).unwrap();
    assert_eq!(file.type_width_expr(data_type), Some(width_expression));
    assert_eq!(file.port_width(file.module_ports(module)[0]), 0);
    assert_eq!(
        file.emit(),
        r#"module Symbolic #(
  parameter WIDTH = 4
) (
  output bit [WIDTH - 1:0] data
);

endmodule
"#
    );
}

#[test]
fn bit_wire_nets_are_rejected() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Invalid");
    let bit = file.make_bit_type();
    let error = file.add_wire(module, "bad", &bit).unwrap_err();
    assert!(error.to_string().contains("cannot be used as a wire net"));
}

#[test]
fn bit_inout_ports_are_rejected() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Invalid");
    let bit = file.make_bit_type();
    let array = file.make_unpacked_array_type(bit, &[2]).unwrap();
    let error = file.add_inout(module, "bad", &array).unwrap_err();
    assert!(error.to_string().contains("cannot be used as an inout net"));
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn bit_vector_width_from_another_file_is_rejected() {
    let mut source = VastFile::new(VastFileType::SystemVerilog);
    let width = source.make_unsized_decimal_literal(8);
    let mut target = VastFile::new(VastFileType::SystemVerilog);
    let _ = target.make_two_state_bit_vector_type_expr(&width, false);
}

/// The existing int atom is not a legal net type either.
#[test]
fn int_wire_is_rejected() {
    let mut f = VastFile::new(VastFileType::SystemVerilog);
    let m = f.add_module("m");
    let t = f.make_int_type(true);
    let error = f.add_wire(m, "n", &t).unwrap_err();
    assert!(error.to_string().contains("cannot be used as a wire net"));
}

/// An int inout would be invalid SystemVerilog.
#[test]
fn int_inout_is_rejected() {
    let mut f = VastFile::new(VastFileType::SystemVerilog);
    let m = f.add_module("m");
    let t = f.make_int_type(true);
    let error = f.add_inout(m, "n", &t).unwrap_err();
    assert!(error.to_string().contains("cannot be used as an inout net"));
}

/// A logic-vector target must explicitly convert a two-state operand to
/// four-state.
#[test]
fn logic_vector_cast_targets_four_state_logic() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let m = file.add_module("Casts");
    let bit = file.make_two_state_bit_vector_type(4, false).unwrap();
    let logic = file.make_bit_vector_type(4, false).unwrap();
    let b = file.add_input(m, "b", &bit).unwrap();
    let cast = file.make_type_cast(&logic, &b.to_expr()).unwrap();
    assert_eq!(
        file.emit_expression(&cast),
        "unsigned'(type(logic [3:0])'(b))"
    );
}
