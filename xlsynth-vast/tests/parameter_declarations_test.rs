// SPDX-License-Identifier: Apache-2.0

//! Public golden and validation coverage for module parameter declarations.

use xlsynth_vast::{ParameterKind, SvArgument, VastFile, VastFileType};

#[test]
fn structured_header_preserves_parameter_kinds_types_and_defaults() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let leaf = file.add_module("Leaf");
    file.add_module_import(leaf, "p", "*");
    let int = file.make_int_type(true);
    let word = file.make_extern_type("word_t");
    let port_word = file.make_extern_type("T");
    let mode = file.make_unsized_decimal_literal(0);
    let one = file.make_unsized_decimal_literal(1);
    let width =
        file.add_value_parameter_port(leaf, "W", ParameterKind::Parameter, Some(&int), None);
    let depth = file.make_add(&width.to_expr(), &one);
    file.add_value_parameter_port(leaf, "MODE", ParameterKind::Parameter, None, Some(&mode));
    file.add_type_parameter_port(leaf, "T", ParameterKind::Parameter, Some(&word));
    file.add_type_parameter_port(leaf, "RequiredT", ParameterKind::Parameter, None);
    file.add_value_parameter_port(
        leaf,
        "DEPTH",
        ParameterKind::Localparam,
        Some(&int),
        Some(&depth),
    );
    file.add_output(leaf, "q", &port_word);
    file.add_type_parameter(leaf, "BodyT", ParameterKind::Parameter, &word);
    file.add_type_parameter(leaf, "HiddenT", ParameterKind::Localparam, &word);

    assert_eq!(
        file.emit(),
        r#"module Leaf import p::*; #(
  parameter int W,
  parameter MODE = 0,
  parameter type T = word_t,
  parameter type RequiredT,
  localparam int DEPTH = W + 1
) (
  output T q
);
  parameter type BodyT = word_t;
  localparam type HiddenT = word_t;
endmodule
"#
    );
}

#[test]
fn type_parameter_overrides_use_typed_arguments() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Assembly");
    let byte = file.make_bit_vector_type(8, false);
    let width = file.make_unsized_decimal_literal(8);
    let instantiation = file.make_instantiation_with_arguments(
        "Leaf",
        "u",
        &["W", "T"],
        &[SvArgument::Value(width), SvArgument::Type(byte)],
        &[],
        &[],
    );
    file.add_member_instantiation(module, instantiation);
    assert_eq!(
        file.emit(),
        r#"module Assembly;
  Leaf #(
    .W(8),
    .T(logic [7:0])
  ) u (

  );
endmodule
"#
    );
}

#[test]
fn imports_without_parameters_or_ports_get_an_explicit_empty_port_list() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Imported");
    file.add_module_import(module, "p", "*");
    file.add_module_import(module, "q", "word_t");
    assert_eq!(
        file.emit(),
        "module Imported import p::*; import q::word_t; ();\n\nendmodule\n"
    );
}

#[test]
#[should_panic(expected = "invalid package identifier")]
fn module_import_package_is_an_identifier_not_a_header_fragment() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Invalid");
    file.add_module_import(module, "p::*; #(parameter W=4)", "*");
}

#[test]
#[should_panic(expected = "invalid import item")]
fn body_import_item_is_an_identifier_or_wildcard() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Invalid");
    file.add_member_import(module, "p", "x; parameter W=4");
}

#[test]
fn structured_imports_and_type_arguments_build_a_parameter_header_and_body() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Leaf");
    file.add_module_import(module, "p", "*");
    let word = file.make_extern_type("word_t");
    let bits = file.make_function_call_with_arguments("$bits", &[SvArgument::Type(word)]);
    file.add_value_parameter_port(module, "W", ParameterKind::Parameter, None, Some(&bits));
    file.add_output(module, "q", &word);
    file.add_member_import(module, "q", "body_t");
    let body = file.make_extern_type("body_t");
    file.add_type_parameter(module, "BodyT", ParameterKind::Parameter, &body);
    assert_eq!(
        file.emit(),
        r#"module Leaf import p::*; #(
  parameter W = $bits(word_t)
) (
  output word_t q
);
  import q::body_t;
  parameter type BodyT = body_t;
endmodule
"#
    );
}

#[test]
fn import_without_parameters_can_supply_a_port_type() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Direct");
    file.add_module_import(module, "p", "*");
    let word = file.make_extern_type("word_t");
    file.add_output(module, "q", &word);
    assert_eq!(
        file.emit(),
        "module Direct import p::*; (\n  output word_t q\n);\n\nendmodule\n"
    );
}

#[test]
fn compilation_unit_import_precedes_the_affected_module() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    file.add_file_import("p", "word_t");
    let module = file.add_module("UnitImported");
    let word = file.make_extern_type("word_t");
    file.add_output(module, "q", &word);
    assert_eq!(
        file.emit(),
        "import p::word_t;\nmodule UnitImported(\n  output word_t q\n);\n\nendmodule\n"
    );
}

#[test]
fn declaration_hoisting_does_not_cross_body_imports() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Body");
    let before = file.make_comment("before import");
    file.add_member_comment(module, before);
    file.add_member_import(module, "p", "word_t");
    let after = file.make_comment("after import");
    file.add_member_comment(module, after);
    let word = file.make_extern_type("word_t");
    file.add_wire(module, "value", &word);
    file.hoist_module_declarations(module, 0);
    assert_eq!(
        file.emit(),
        r#"module Body;
  // before import
  import p::word_t;
  wire word_t value;
  // after import
endmodule
"#
    );
}

#[test]
#[should_panic]
fn local_parameter_without_a_default_is_rejected() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Invalid");
    file.add_value_parameter_port(module, "W", ParameterKind::Localparam, None, None);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn parameter_default_from_another_file_is_rejected() {
    let mut source = VastFile::new(VastFileType::SystemVerilog);
    let default = source.make_unsized_decimal_literal(1);
    let mut destination = VastFile::new(VastFileType::SystemVerilog);
    let module = destination.add_module("Invalid");
    destination.add_value_parameter_port(
        module,
        "W",
        ParameterKind::Parameter,
        None,
        Some(&default),
    );
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn type_parameter_default_from_another_file_is_rejected() {
    let mut source = VastFile::new(VastFileType::SystemVerilog);
    let default = source.make_scalar_type();
    let mut destination = VastFile::new(VastFileType::SystemVerilog);
    let module = destination.add_module("Invalid");
    destination.add_type_parameter_port(module, "T", ParameterKind::Parameter, Some(&default));
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn type_argument_from_another_file_is_rejected() {
    let mut source = VastFile::new(VastFileType::SystemVerilog);
    let data_type = source.make_scalar_type();
    let mut destination = VastFile::new(VastFileType::SystemVerilog);
    destination.make_function_call_with_arguments("$bits", &[SvArgument::Type(data_type)]);
}

#[test]
#[should_panic]
fn local_type_parameter_without_a_default_is_rejected() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Invalid");
    file.add_type_parameter_port(module, "T", ParameterKind::Localparam, None);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn body_type_parameter_default_from_another_file_is_rejected() {
    let mut source = VastFile::new(VastFileType::SystemVerilog);
    let default = source.make_scalar_type();
    let mut destination = VastFile::new(VastFileType::SystemVerilog);
    let module = destination.add_module("Invalid");
    destination.add_type_parameter(module, "T", ParameterKind::Parameter, &default);
}

#[test]
#[should_panic(expected = "anonymous unpacked array types require a named typedef")]
fn anonymous_unpacked_type_default_is_rejected_instead_of_becoming_packed() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Invalid");
    let byte = file.make_bit_vector_type(8, false);
    let array = file.make_unpacked_array_type(byte, &[4]);
    file.add_type_parameter_port(module, "T", ParameterKind::Parameter, Some(&array));
}

#[test]
#[should_panic(expected = "anonymous unpacked array types require a named typedef")]
fn anonymous_unpacked_type_override_requires_a_typedef() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let byte = file.make_bit_vector_type(8, false);
    let array = file.make_unpacked_array_type(byte, &[4]);
    file.make_instantiation_with_arguments(
        "Leaf",
        "u",
        &["T"],
        &[SvArgument::Type(array)],
        &[],
        &[],
    );
}

/// Hoisting a signal must preserve the body declarations that define its type
/// and width.
#[test]
fn hoisting_respects_body_parameter_dependencies() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let m = file.add_module("Body");
    let byte = file.make_bit_vector_type(8, false);
    let eight = file.make_unsized_decimal_literal(8);
    let word = file.add_type_parameter(m, "T", ParameterKind::Parameter, &byte);
    let width = file.add_parameter(m, "W", &eight);
    let vector = file.make_bit_vector_type_expr(&width.to_expr(), false);
    file.add_logic(m, "typed", &word).unwrap();
    file.add_logic(m, "sized", &vector).unwrap();
    file.hoist_module_declarations(m, 0);
    assert_eq!(
        file.emit(),
        r#"module Body;
  parameter type T = logic [7:0];
  parameter W = 8;
  T typed;
  logic [W - 1:0] sized;
endmodule
"#
    );
}
