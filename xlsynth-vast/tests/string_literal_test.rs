// SPDX-License-Identifier: Apache-2.0

//! Golden tests for Verilog strings and string-valued parameter expressions.

use xlsynth_vast::{DataKind, VastFile, VastFileType};

#[test]
fn simple_and_empty_strings_are_quoted_in_both_verilog_dialects() {
    for dialect in [VastFileType::Verilog, VastFileType::SystemVerilog] {
        let mut file = VastFile::new(dialect);
        let simple = file.make_string_literal("ExampleValue");
        let empty = file.make_string_literal("");

        assert_eq!(file.emit_expression(&simple), r#""ExampleValue""#);
        assert_eq!(file.emit_expression(&empty), r#""""#);
    }
}

#[test]
fn quotes_and_literal_backslashes_are_escaped_without_reinterpreting_input() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let expression = file.make_string_literal(r#"say "example" from path\to\value"#);

    assert_eq!(
        file.emit_expression(&expression),
        r#""say \"example\" from path\\to\\value""#
    );
}

#[test]
fn newline_carriage_return_and_tab_use_named_verilog_escapes() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let expression = file.make_string_literal("first\nsecond\rthird\tfourth");

    assert_eq!(
        file.emit_expression(&expression),
        r#""first\nsecond\rthird\tfourth""#
    );
}

#[test]
fn nul_and_other_ascii_controls_use_fixed_width_octal_escapes() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let expression = file.make_string_literal("\0\u{1}\u{7}\u{b}\u{c}\u{1f}\u{7f}");

    assert_eq!(
        file.emit_expression(&expression),
        r#""\000\001\007\013\014\037\177""#
    );
}

#[test]
fn ordinary_unicode_is_preserved_without_byte_or_codepoint_escaping() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let expression = file.make_string_literal("café • 文字 μ 🦀");

    assert_eq!(file.emit_expression(&expression), r#""café • 文字 μ 🦀""#);
}

#[test]
fn instance_parameter_overrides_accept_quoted_string_values() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("wrapper");
    let label = file.make_string_literal("ExampleValue");
    let instance = file.make_instantiation(
        "example_module",
        "example_instance",
        &["label"],
        &[&label],
        &["clk"],
        &[None],
    );
    file.add_member_instantiation(module, instance);

    let expected = r#"module wrapper;
  example_module #(
    .label("ExampleValue")
  ) example_instance (
    .clk()
  );
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn mixed_string_and_numeric_instance_parameters_preserve_insertion_order() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("wrapper");
    let scalar = file.make_scalar_type();
    let clock = file.add_input(module, "clock", &scalar).to_expr();
    let label = file.make_string_literal("example");
    let width = file.make_unsized_decimal_literal(64);
    let path = file.make_string_literal(r"nested\value");
    let instance = file.make_instantiation(
        "example_module",
        "example_instance",
        &["label", "Width", "path"],
        &[&label, &width, &path],
        &["clk", "unused"],
        &[Some(&clock), None],
    );
    file.add_member_instantiation(module, instance);

    let expected = r#"module wrapper(
  input wire clock
);
  example_module #(
    .label("example"),
    .Width(64),
    .path("nested\\value")
  ) example_instance (
    .clk(clock),
    .unused()
  );
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn untyped_module_ports_body_parameters_and_localparams_accept_strings() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("configured");
    let label = file.make_string_literal("example");
    let mode = file.make_string_literal("standard");
    let setting = file.make_string_literal("default");
    file.add_parameter_port(module, "label", &label);
    file.add_parameter(module, "mode", &mode);
    file.add_localparam(module, "setting", &setting);

    let expected = r#"module configured #(
  parameter label = "example"
);
  parameter mode = "standard";
  localparam setting = "default";
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn typed_string_parameters_use_the_existing_external_type_representation() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("configured");
    let string_type = file.make_extern_type("string");
    let label = file.make_string_literal("example");
    let mode = file.make_string_literal("standard");
    let setting = file.make_string_literal("default");
    file.add_typed_parameter_port(module, "label", &string_type, &label);
    let mode_definition = file.make_def("mode", DataKind::User, &string_type);
    file.add_parameter_with_def(module, &mode_definition, &mode);
    let setting_definition = file.make_def("setting", DataKind::User, &string_type);
    file.add_typed_localparam(module, &setting_definition, &setting);

    let expected = r#"module configured #(
  parameter string label = "example"
);
  parameter string mode = "standard";
  localparam string setting = "default";
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn instance_parameter_overrides_reject_string_handles_from_another_file() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let mut other = VastFile::new(VastFileType::SystemVerilog);
    let foreign_label = other.make_string_literal("example");

    file.make_instantiation(
        "example_module",
        "example_instance",
        &["label"],
        &[&foreign_label],
        &[],
        &[],
    );
}
