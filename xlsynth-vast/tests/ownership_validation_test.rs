// SPDX-License-Identifier: Apache-2.0

//! File ownership, handle validation, and invalid-AST construction tests.

use xlsynth_vast::{DataKind, LiteralFormat, ModulePortDirection, VastFile, VastFileType};

#[test]
fn mixed_port_directions_keep_declaration_order_and_independent_filters() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("directions");
    let scalar = file.make_scalar_type();
    let byte = file.make_bit_vector_type(8, false);
    file.add_output(module, "first_output", &byte);
    let first_input = file.add_input(module, "first_input", &scalar);
    file.add_inout(module, "bidirectional", &byte);
    file.add_input(module, "second_input", &byte);
    file.add_output(module, "second_output", &scalar);

    let input_names = file
        .module_input_ports(module)
        .iter()
        .map(|port| file.port_name(*port))
        .collect::<Vec<_>>();
    let output_names = file
        .module_output_ports(module)
        .iter()
        .map(|port| file.port_name(*port))
        .collect::<Vec<_>>();
    let all_ports = file.module_ports(module);

    assert_eq!(input_names, ["first_input", "second_input"]);
    assert_eq!(output_names, ["first_output", "second_output"]);
    assert_eq!(
        file.port_direction(all_ports[2]),
        ModulePortDirection::InOut
    );
    assert_eq!(file.port_data_type(all_ports[2]), byte);
    assert_eq!(file.port_width(all_ports[2]), 8);
    assert_eq!(file.logic_ref_name(first_input), "first_input");
    assert_eq!(
        file.emit(),
        r#"module directions(
  output wire [7:0] first_output,
  input wire first_input,
  inout wire [7:0] bidirectional,
  input wire [7:0] second_input,
  output wire second_output
);

endmodule
"#
    );
}

#[test]
fn separate_modules_have_independent_declaration_namespaces() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let first = file.add_module("first");
    let second = file.add_module("second");
    let scalar = file.make_scalar_type();
    file.add_input(first, "shared_name", &scalar);
    file.add_input(second, "shared_name", &scalar);
    file.add_reg(first, "state", &scalar)
        .expect("first module may declare state");
    file.add_reg(second, "state", &scalar)
        .expect("second module may independently declare state");

    assert_eq!(
        file.emit(),
        r#"module first(
  input wire shared_name
);
  reg state;
endmodule
module second(
  input wire shared_name
);
  reg state;
endmodule
"#
    );
}

#[test]
fn rejected_duplicate_declarations_do_not_modify_the_module() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("duplicates");
    let scalar = file.make_scalar_type();
    file.add_input(module, "existing", &scalar);
    let before = file.emit();

    let error = file
        .add_reg(module, "existing", &scalar)
        .expect_err("a register must not reuse an input port name");
    assert_eq!(
        error.to_string(),
        "FAILED_PRECONDITION: Attempted to declare reg with name 'existing' multiple times in the same module. Already defined: [existing]"
    );
    assert_eq!(file.emit(), before);

    file.add_logic(module, "new_name", &scalar)
        .expect("the failed declaration must not poison future declarations");
    assert_eq!(
        file.emit(),
        r#"module duplicates(
  input wire existing
);
  logic new_name;
endmodule
"#
    );
}

#[test]
fn symbolic_widths_report_unavailable_integer_and_port_widths() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("symbolic");
    let four = file.make_plain_literal(4, &LiteralFormat::UnsignedDecimal);
    let width = file.add_parameter_port(module, "WIDTH", &four);
    let symbolic = file.make_bit_vector_type_expr(&width.to_expr(), false);
    file.add_input(module, "data", &symbolic);

    assert_eq!(
        file.type_width_as_int64(symbolic)
            .expect_err("symbolic width has no integer value")
            .to_string(),
        "Width is not a literal: WIDTH"
    );
    assert_eq!(
        file.type_flat_bit_count_as_int64(symbolic)
            .expect_err("symbolic flat width has no integer value")
            .to_string(),
        "Width is not a literal: WIDTH"
    );
    assert_eq!(file.port_width(file.module_ports(module)[0]), 0);
    assert_eq!(
        file.emit_expression(&file.type_width_expr(symbolic).expect("width expression")),
        "WIDTH"
    );
    assert_eq!(
        file.emit(),
        r#"module symbolic #(
  parameter WIDTH = 4
) (
  input wire [WIDTH - 1:0] data
);

endmodule
"#
    );
}

#[test]
fn nested_array_introspection_preserves_element_width_and_signedness() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let element = file.make_bit_vector_type(5, true);
    let packed = file.make_packed_array_type(element, &[3, 2]);
    let nested = file.make_unpacked_array_type(packed, &[4]);

    assert_eq!(file.type_width_as_int64(nested), Ok(5));
    assert_eq!(file.type_flat_bit_count_as_int64(nested), Ok(120));
    assert!(file.type_is_signed(nested));
    assert_eq!(
        file.emit_expression(&file.type_width_expr(nested).expect("element width")),
        "5"
    );
}

#[test]
fn overflowing_array_flat_width_returns_a_structured_error() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let element = file.make_bit_vector_type(2, false);
    let array = file.make_packed_array_type(element, &[i64::MAX, 2]);

    assert_eq!(
        file.type_flat_bit_count_as_int64(array)
            .expect_err("an overflowing flattened width must be rejected")
            .to_string(),
        "array bit count exceeds i64"
    );
}

#[test]
fn integer_types_have_flat_width_without_a_vector_width_expression() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let signed = file.make_integer_type(true);
    let unsigned = file.make_int_type(false);

    assert_eq!(file.type_flat_bit_count_as_int64(signed), Ok(32));
    assert_eq!(file.type_flat_bit_count_as_int64(unsigned), Ok(32));
    assert!(file.type_width_as_int64(signed).is_err());
    assert!(file.type_width_as_int64(unsigned).is_err());
    assert_eq!(file.type_width_expr(signed), None);
    assert_eq!(file.type_width_expr(unsigned), None);
    assert!(file.type_is_signed(signed));
    assert!(!file.type_is_signed(unsigned));
}

#[test]
fn external_type_width_errors_identify_qualified_and_unqualified_types() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let local = file.make_extern_type("payload_t");
    let qualified = file.make_extern_package_type("bus_pkg", "payload_t");

    assert_eq!(
        file.type_flat_bit_count_as_int64(local)
            .expect_err("external widths are not known")
            .to_string(),
        "external type `payload_t` has no known bit width"
    );
    assert_eq!(
        file.type_flat_bit_count_as_int64(qualified)
            .expect_err("package-qualified widths are not known")
            .to_string(),
        "external type `bus_pkg::payload_t` has no known bit width"
    );
}

#[test]
fn malformed_and_out_of_range_literals_return_errors() {
    let mut file = VastFile::new(VastFileType::Verilog);

    for invalid in [
        "",
        "bits[]:0",
        "bits[-1]:0",
        "bits[width]:0",
        "bits[8]:",
        "bits[8]:0xgg",
        "bits[2]:4",
        "bits[8]:-257",
        "u8:1",
    ] {
        assert!(
            file.make_literal(invalid, &LiteralFormat::Hex).is_err(),
            "literal {invalid:?} should be rejected"
        );
    }
    assert_eq!(file.emit(), "");
}

#[test]
fn invalid_always_sensitivity_is_rejected_without_adding_a_block() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("events");
    let scalar = file.make_scalar_type();
    let signal = file.add_input(module, "signal", &scalar).to_expr();
    let invalid = file.make_logical_not(&signal);

    let ff_error = file
        .add_always_ff(module, &[&invalid])
        .expect_err("a unary expression is not a sensitivity event");
    assert_eq!(
        ff_error.to_string(),
        "Unsupported expression type passed to sensitivity list for always_ff at index 0. Only posedge, negedge, or logic-reference expressions are supported."
    );
    let at_error = file
        .add_always_at(module, &[&invalid])
        .expect_err("a unary expression is not a sensitivity event");
    assert_eq!(
        at_error.to_string(),
        "Unsupported expression type passed to sensitivity list for always @ at index 0. Only posedge, negedge, or logic-reference expressions are supported."
    );
    assert_eq!(
        file.emit(),
        "module events(\n  input wire signal\n);\n\nendmodule\n"
    );
}

#[test]
fn invalid_generate_sensitivity_is_rejected_without_adding_a_block() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("generated_events");
    let zero = file.make_plain_literal(0, &LiteralFormat::Default);
    let two = file.make_plain_literal(2, &LiteralFormat::Default);
    let generate = file.add_generate_loop(module, "i", &zero, &two, None);

    assert!(file.generate_add_always_ff(generate, &[&two]).is_err());
    assert!(file.generate_add_always_at(generate, &[&two]).is_err());
    assert_eq!(
        file.emit(),
        r#"module generated_events;
  for (genvar i = 0; i < 2; i = i + 1) begin
  end
endmodule
"#
    );
}

#[test]
#[should_panic(expected = "cannot add an else-if arm after an unconditional else")]
fn conditionals_reject_else_if_after_unconditional_else() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("conditionals");
    let scalar = file.make_scalar_type();
    let condition = file.add_input(module, "condition", &scalar).to_expr();
    let conditional = file.add_conditional(module, &condition);
    file.conditional_add_else(conditional);

    file.conditional_add_else_if(conditional, &condition);
}

#[test]
#[should_panic(expected = "cannot add another arm after an unconditional else")]
fn conditionals_reject_multiple_unconditional_else_arms() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("conditionals");
    let scalar = file.make_scalar_type();
    let condition = file.add_input(module, "condition", &scalar).to_expr();
    let conditional = file.add_conditional(module, &condition);
    file.conditional_add_else(conditional);

    file.conditional_add_else(conditional);
}

#[test]
#[should_panic(expected = "bit-vector width must be greater than zero")]
fn zero_bit_vector_width_is_rejected() {
    let mut file = VastFile::new(VastFileType::Verilog);
    file.make_bit_vector_type(0, false);
}

#[test]
#[should_panic(expected = "bit-vector width must be greater than zero")]
fn negative_bit_vector_width_is_rejected() {
    let mut file = VastFile::new(VastFileType::Verilog);
    file.make_bit_vector_type(-1, false);
}

#[test]
#[should_panic(expected = "packed array requires at least one dimension")]
fn packed_arrays_require_at_least_one_dimension() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let scalar = file.make_scalar_type();
    file.make_packed_array_type(scalar, &[]);
}

#[test]
#[should_panic(expected = "packed-array dimensions must be greater than zero")]
fn packed_arrays_reject_zero_dimensions() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let scalar = file.make_scalar_type();
    file.make_packed_array_type(scalar, &[2, 0]);
}

#[test]
#[should_panic(expected = "unpacked array requires at least one dimension")]
fn unpacked_arrays_require_at_least_one_dimension() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let scalar = file.make_scalar_type();
    file.make_unpacked_array_type(scalar, &[]);
}

#[test]
#[should_panic(expected = "unpacked-array dimensions must be greater than zero")]
fn unpacked_arrays_reject_negative_dimensions() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let scalar = file.make_scalar_type();
    file.make_unpacked_array_type(scalar, &[2, -1]);
}

#[test]
#[should_panic(expected = "bit and part-select indices must be nonnegative")]
fn constant_slices_reject_negative_upper_indices() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("slices");
    let vector = file.make_bit_vector_type(8, false);
    let signal = file.add_input(module, "signal", &vector);

    file.make_slice(&signal.to_indexable_expr(), -1, 0);
}

#[test]
#[should_panic(expected = "bit and part-select indices must be nonnegative")]
fn constant_slices_reject_negative_lower_indices() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("slices");
    let vector = file.make_bit_vector_type(8, false);
    let signal = file.add_input(module, "signal", &vector);

    file.make_slice(&signal.to_indexable_expr(), 7, -1);
}

#[test]
#[should_panic(
    expected = "module-instantiation parameter names and values must have equal lengths"
)]
fn module_instantiations_reject_mismatched_parameter_lists() {
    let mut file = VastFile::new(VastFileType::Verilog);
    file.make_instantiation("child", "instance", &["WIDTH"], &[], &[], &[]);
}

#[test]
#[should_panic(expected = "module-instantiation port names and values must have equal lengths")]
fn module_instantiations_reject_mismatched_connection_lists() {
    let mut file = VastFile::new(VastFileType::Verilog);
    file.make_instantiation("child", "instance", &[], &[], &["input"], &[]);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn expression_emission_rejects_handles_from_another_file() {
    let file = VastFile::new(VastFileType::Verilog);
    let mut other = VastFile::new(VastFileType::Verilog);
    let expression = other.make_plain_literal(1, &LiteralFormat::Default);

    file.emit_expression(&expression);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn module_introspection_rejects_handles_from_another_file() {
    let file = VastFile::new(VastFileType::Verilog);
    let mut other = VastFile::new(VastFileType::Verilog);
    let module = other.add_module("foreign");

    file.module_name(module);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn port_introspection_rejects_handles_from_another_file() {
    let file = VastFile::new(VastFileType::Verilog);
    let mut other = VastFile::new(VastFileType::Verilog);
    let module = other.add_module("foreign");
    let scalar = other.make_scalar_type();
    other.add_input(module, "foreign", &scalar);
    let port = other.module_ports(module)[0];

    file.port_direction(port);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn declarations_reject_data_types_from_another_file() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let mut other = VastFile::new(VastFileType::Verilog);
    let foreign_type = other.make_scalar_type();

    file.make_def("foreign", DataKind::Wire, &foreign_type);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn instantiations_reject_connection_expressions_from_another_file() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let mut other = VastFile::new(VastFileType::Verilog);
    let foreign = other.make_plain_literal(1, &LiteralFormat::Default);

    file.make_instantiation("child", "instance", &[], &[], &["input"], &[Some(&foreign)]);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn case_statements_reject_patterns_from_another_file() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("cases");
    let scalar = file.make_scalar_type();
    let selector = file.add_input(module, "selector", &scalar).to_expr();
    let always = file.add_always_comb(module).expect("valid always block");
    let case = file.block_add_case(file.statement_block(always), &selector);

    let mut other = VastFile::new(VastFileType::SystemVerilog);
    let foreign = other.make_plain_literal(1, &LiteralFormat::Default);
    file.case_add_item(case, &foreign);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn generate_loops_reject_bounds_from_another_file() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("generate");
    let start = file.make_plain_literal(0, &LiteralFormat::Default);
    let mut other = VastFile::new(VastFileType::SystemVerilog);
    let foreign_end = other.make_plain_literal(2, &LiteralFormat::Default);

    file.add_generate_loop(module, "i", &start, &foreign_end, None);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn always_blocks_reject_sensitivity_expressions_from_another_file() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("events");
    let mut other = VastFile::new(VastFileType::SystemVerilog);
    let other_module = other.add_module("other");
    let scalar = other.make_scalar_type();
    let foreign = other.add_input(other_module, "foreign", &scalar).to_expr();

    let _ = file.add_always_ff(module, &[&foreign]);
}
