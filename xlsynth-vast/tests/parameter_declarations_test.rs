// SPDX-License-Identifier: Apache-2.0

//! Public golden and validation coverage for module parameter declarations.

use xlsynth_vast::{
    ParameterKind, SvArgument, VastDataType, VastError, VastFile, VastFileType, VastModule,
};

#[test]
fn structured_header_preserves_parameter_kinds_types_and_defaults() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let leaf = file.add_module("Leaf");
    file.add_module_import(leaf, "p", "*").unwrap();
    let int = file.make_int_type(true);
    let word = file.make_extern_type("word_t");
    let port_word = file.make_extern_type("T");
    let mode = file.make_unsized_decimal_literal(0);
    let one = file.make_unsized_decimal_literal(1);
    let width = file
        .add_value_parameter_port(leaf, "W", ParameterKind::Parameter, Some(&int), None)
        .unwrap();
    let depth = file.make_add(&width.to_expr(), &one);
    file.add_value_parameter_port(leaf, "MODE", ParameterKind::Parameter, None, Some(&mode))
        .unwrap();
    file.add_type_parameter_port(leaf, "T", ParameterKind::Parameter, Some(&word))
        .unwrap();
    file.add_type_parameter_port(leaf, "RequiredT", ParameterKind::Parameter, None)
        .unwrap();
    file.add_value_parameter_port(
        leaf,
        "DEPTH",
        ParameterKind::Localparam,
        Some(&int),
        Some(&depth),
    )
    .unwrap();
    file.add_output(leaf, "q", &port_word).unwrap();
    file.add_type_parameter(leaf, "BodyT", ParameterKind::Parameter, &word)
        .unwrap();
    file.add_type_parameter(leaf, "HiddenT", ParameterKind::Localparam, &word)
        .unwrap();

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
    let byte = file.make_bit_vector_type(8, false).unwrap();
    let width = file.make_unsized_decimal_literal(8);
    let instantiation = file
        .make_instantiation_with_arguments(
            "Leaf",
            "u",
            &["W", "T"],
            &[SvArgument::Value(width), SvArgument::Type(byte)],
            &[],
            &[],
        )
        .unwrap();
    file.add_member_instantiation(module, instantiation)
        .unwrap();
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
    file.add_module_import(module, "p", "*").unwrap();
    file.add_module_import(module, "q", "word_t").unwrap();
    assert_eq!(
        file.emit(),
        "module Imported import p::*; import q::word_t; ();\n\nendmodule\n"
    );
}

#[test]
fn module_import_package_is_an_identifier_not_a_header_fragment() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Invalid");
    let before = file.emit();
    let error = file
        .add_module_import(module, "p::*; #(parameter W=4)", "*")
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid package identifier: p::*; #(parameter W=4)"
    );
    assert_eq!(file.emit(), before);
    file.add_module_import(module, "p", "*").unwrap();
}

#[test]
fn body_import_item_is_an_identifier_or_wildcard() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Invalid");
    let error = file
        .add_member_import(module, "p", "x; parameter W=4")
        .unwrap_err();
    assert_eq!(error.to_string(), "invalid import item: x; parameter W=4");
}

#[test]
fn structured_imports_and_type_arguments_build_a_parameter_header_and_body() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Leaf");
    file.add_module_import(module, "p", "*").unwrap();
    let word = file.make_extern_type("word_t");
    let bits = file
        .make_function_call_with_arguments("$bits", &[SvArgument::Type(word)])
        .unwrap();
    file.add_value_parameter_port(module, "W", ParameterKind::Parameter, None, Some(&bits))
        .unwrap();
    file.add_output(module, "q", &word).unwrap();
    file.add_member_import(module, "q", "body_t").unwrap();
    let body = file.make_extern_type("body_t");
    file.add_type_parameter(module, "BodyT", ParameterKind::Parameter, &body)
        .unwrap();
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
    file.add_module_import(module, "p", "*").unwrap();
    let word = file.make_extern_type("word_t");
    file.add_output(module, "q", &word).unwrap();
    assert_eq!(
        file.emit(),
        "module Direct import p::*; (\n  output word_t q\n);\n\nendmodule\n"
    );
}

#[test]
fn compilation_unit_import_precedes_the_affected_module() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    file.add_file_import("p", "word_t").unwrap();
    let module = file.add_module("UnitImported");
    let word = file.make_extern_type("word_t");
    file.add_output(module, "q", &word).unwrap();
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
    file.add_member_import(module, "p", "word_t").unwrap();
    let after = file.make_comment("after import");
    file.add_member_comment(module, after);
    let word = file.make_extern_type("word_t");
    file.add_wire(module, "value", &word).unwrap();
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
fn local_parameter_without_a_default_is_rejected() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Recovered");
    let before = file.emit();
    let error = file
        .add_value_parameter_port(module, "W", ParameterKind::Localparam, None, None)
        .unwrap_err();
    assert_eq!(error.to_string(), "local parameters require a default");
    assert_eq!(file.emit(), before);
    let value = file.make_unsized_decimal_literal(8);
    file.add_value_parameter_port(module, "W", ParameterKind::Localparam, None, Some(&value))
        .unwrap();
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn parameter_default_from_another_file_is_rejected() {
    let mut source = VastFile::new(VastFileType::SystemVerilog);
    let default = source.make_unsized_decimal_literal(1);
    let mut destination = VastFile::new(VastFileType::SystemVerilog);
    let module = destination.add_module("Invalid");
    let _ = destination.add_value_parameter_port(
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
    let _ =
        destination.add_type_parameter_port(module, "T", ParameterKind::Parameter, Some(&default));
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn type_argument_from_another_file_is_rejected() {
    let mut source = VastFile::new(VastFileType::SystemVerilog);
    let data_type = source.make_scalar_type();
    let mut destination = VastFile::new(VastFileType::SystemVerilog);
    let _ = destination.make_function_call_with_arguments("$bits", &[SvArgument::Type(data_type)]);
}

#[test]
fn local_type_parameter_without_a_default_is_rejected() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Invalid");
    let error = file
        .add_type_parameter_port(module, "T", ParameterKind::Localparam, None)
        .unwrap_err();
    assert_eq!(error.to_string(), "local parameters require a default");
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn body_type_parameter_default_from_another_file_is_rejected() {
    let mut source = VastFile::new(VastFileType::SystemVerilog);
    let default = source.make_scalar_type();
    let mut destination = VastFile::new(VastFileType::SystemVerilog);
    let module = destination.add_module("Invalid");
    let _ = destination.add_type_parameter(module, "T", ParameterKind::Parameter, &default);
}

#[test]
fn anonymous_unpacked_type_default_is_rejected_instead_of_becoming_packed() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Invalid");
    let byte = file.make_bit_vector_type(8, false).unwrap();
    let array = file.make_unpacked_array_type(byte, &[4]).unwrap();
    let error = file
        .add_type_parameter_port(module, "T", ParameterKind::Parameter, Some(&array))
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "anonymous unpacked array types require a named typedef"
    );
}

#[test]
fn anonymous_unpacked_type_override_requires_a_typedef() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let byte = file.make_bit_vector_type(8, false).unwrap();
    let array = file.make_unpacked_array_type(byte, &[4]).unwrap();
    let error = file
        .make_instantiation_with_arguments(
            "Leaf",
            "u",
            &["T"],
            &[SvArgument::Type(array)],
            &[],
            &[],
        )
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "anonymous unpacked array types require a named typedef"
    );
}

/// Hoisting a signal must preserve the body declarations that define its type
/// and width.
#[test]
fn hoisting_respects_body_parameter_dependencies() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let m = file.add_module("Body");
    let byte = file.make_bit_vector_type(8, false).unwrap();
    let eight = file.make_unsized_decimal_literal(8);
    let word = file
        .add_type_parameter(m, "T", ParameterKind::Parameter, &byte)
        .unwrap();
    let width = file.add_parameter(m, "W", &eight).unwrap();
    let vector = file
        .make_bit_vector_type_expr(&width.to_expr(), false)
        .unwrap();
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

#[derive(Clone, Copy, Debug)]
enum NamedModuleItem {
    HeaderValue,
    HeaderType,
    BodyValue,
    BodyLocalValue,
    BodyType,
    BodyLocalType,
    Input,
    Logic,
    Instance,
    GenerateLabel,
}

impl NamedModuleItem {
    /// Adds a representative declaration of this kind under a shared name.
    fn add(
        self,
        file: &mut VastFile,
        module: VastModule,
        name: &str,
        data_type: VastDataType,
        value: xlsynth_vast::Expr,
    ) -> Result<(), VastError> {
        match self {
            Self::HeaderValue => file.add_parameter_port(module, name, &value).map(|_| ()),
            Self::HeaderType => file
                .add_type_parameter_port(module, name, ParameterKind::Parameter, Some(&data_type))
                .map(|_| ()),
            Self::BodyValue => file.add_parameter(module, name, &value).map(|_| ()),
            Self::BodyLocalValue => file.add_localparam(module, name, &value).map(|_| ()),
            Self::BodyType => file
                .add_type_parameter(module, name, ParameterKind::Parameter, &data_type)
                .map(|_| ()),
            Self::BodyLocalType => file
                .add_type_parameter(module, name, ParameterKind::Localparam, &data_type)
                .map(|_| ()),
            Self::Input => file.add_input(module, name, &data_type).map(|_| ()),
            Self::Logic => file.add_logic(module, name, &data_type).map(|_| ()),
            Self::Instance => {
                let instance = file
                    .make_instantiation("Leaf", name, &[], &[], &[], &[])
                    .unwrap();
                file.add_member_instantiation(module, instance)
            }
            Self::GenerateLabel => file
                .add_generate_loop(module, "i", &value, &value, Some(name))
                .map(|_| ()),
        }
    }

    /// Returns the declaration kind reported for a duplicate-name error.
    fn kind(self) -> &'static str {
        match self {
            Self::HeaderValue | Self::BodyValue => "parameter",
            Self::HeaderType | Self::BodyType => "type parameter",
            Self::BodyLocalValue => "localparam",
            Self::BodyLocalType => "local type parameter",
            Self::Input => "input",
            Self::Logic => "logic",
            Self::Instance => "instance",
            Self::GenerateLabel => "generate block",
        }
    }
}

#[test]
fn parameters_share_the_module_namespace_with_types_ports_and_variables() {
    use NamedModuleItem::*;

    for (left, right) in [
        (HeaderValue, HeaderValue),
        (HeaderValue, HeaderType),
        (HeaderValue, BodyLocalValue),
        (HeaderType, BodyType),
        (BodyValue, BodyLocalType),
        (HeaderValue, Input),
        (HeaderType, Logic),
        (BodyLocalValue, Input),
        (BodyType, Logic),
        (BodyValue, Instance),
        (HeaderType, GenerateLabel),
    ] {
        for (first, duplicate) in [(left, right), (right, left)] {
            let mut file = VastFile::new(VastFileType::SystemVerilog);
            let module = file.add_module("Namespace");
            let data_type = file.make_scalar_type();
            let value = file.make_unsized_decimal_literal(1);
            first
                .add(&mut file, module, "ITEM", data_type, value)
                .unwrap();
            let before = file.emit();

            let error = duplicate
                .add(&mut file, module, "ITEM", data_type, value)
                .unwrap_err();
            assert_eq!(
                error.to_string(),
                format!(
                    "FAILED_PRECONDITION: Attempted to declare {} with name 'ITEM' multiple times in the same module. Already defined: [ITEM]",
                    duplicate.kind()
                ),
                "first={first:?}, duplicate={duplicate:?}"
            );
            assert_eq!(
                file.emit(),
                before,
                "first={first:?}, duplicate={duplicate:?}"
            );
            duplicate
                .add(&mut file, module, "OTHER", data_type, value)
                .unwrap();
        }
    }
}

#[test]
fn generate_loop_genvar_can_shadow_a_module_parameter() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("GenvarScope");
    let zero = file.make_unsized_decimal_literal(0);
    let one = file.make_unsized_decimal_literal(1);
    file.add_parameter_port(module, "i", &zero).unwrap();
    file.add_generate_loop(module, "i", &zero, &one, Some("lanes"))
        .unwrap();
    assert_eq!(
        file.emit(),
        "module GenvarScope #(\n  parameter i = 0\n);\n  for (genvar i = 0; i < 1; i = i + 1) begin : lanes\n  end\nendmodule\n"
    );
}

/// Adds an import at the requested module-header or module-body location.
fn add_import(
    file: &mut VastFile,
    module: VastModule,
    in_header: bool,
    package: &str,
    item: &str,
) -> Result<(), VastError> {
    if in_header {
        file.add_module_import(module, package, item)
    } else {
        file.add_member_import(module, package, item)
    }
}

#[test]
fn explicit_imports_share_the_module_namespace_with_parameters_and_ports() {
    for in_header in [true, false] {
        for local in [NamedModuleItem::HeaderValue, NamedModuleItem::Input] {
            for import_first in [true, false] {
                let mut file = VastFile::new(VastFileType::SystemVerilog);
                let module = file.add_module("Imports");
                let scalar = file.make_scalar_type();
                let value = file.make_unsized_decimal_literal(1);
                if import_first {
                    add_import(&mut file, module, in_header, "pkg", "P").unwrap();
                } else {
                    local.add(&mut file, module, "P", scalar, value).unwrap();
                }
                let before = file.emit();
                let error = if import_first {
                    local
                        .add(&mut file, module, "P", scalar, value)
                        .unwrap_err()
                } else {
                    add_import(&mut file, module, in_header, "pkg", "P").unwrap_err()
                };
                let kind = if import_first { local.kind() } else { "import" };
                assert_eq!(
                    error.to_string(),
                    format!(
                        "FAILED_PRECONDITION: Attempted to declare {kind} with name 'P' multiple times in the same module. Already defined: [P]"
                    )
                );
                assert_eq!(file.emit(), before);
            }
        }
    }
}

#[test]
fn repeated_identical_explicit_imports_are_legal_but_conflicting_packages_are_not() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Imports");
    file.add_module_import(module, "pkg", "P").unwrap();
    file.add_member_import(module, "pkg", "P").unwrap();
    let before = file.emit();
    let error = file
        .add_member_import(module, "other_pkg", "P")
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "FAILED_PRECONDITION: Attempted to declare import with name 'P' multiple times in the same module. Already defined: [P]"
    );
    assert_eq!(file.emit(), before);
    assert_eq!(
        before,
        "module Imports import pkg::P; ();\n  import pkg::P;\nendmodule\n"
    );
}

#[test]
fn wildcard_imports_do_not_reserve_individual_module_names() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Wildcard");
    let value = file.make_unsized_decimal_literal(1);
    file.add_module_import(module, "pkg", "*").unwrap();
    file.add_member_import(module, "other_pkg", "*").unwrap();
    file.add_parameter(module, "P", &value).unwrap();
}
