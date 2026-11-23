// SPDX-License-Identifier: Apache-2.0

//! Structured external references and named struct patterns in parameter
//! defaults.

use xlsynth_vast::{ParameterKind, VastFile, VastFileType};

#[test]
fn structured_parameter_defaults_retain_type_and_member_names() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Leaf");
    file.add_module_import(module, "p", "DEFAULT_CFG");
    let cfg_t = file.make_extern_package_type("p", "cfg_t");
    let config =
        file.add_type_parameter_port(module, "Config", ParameterKind::Parameter, Some(&cfg_t));
    let one = file.make_unsized_decimal_literal(1);
    let five = file.make_unsized_decimal_literal(5);
    let reset = file.make_struct_assignment_pattern(&[("valid", &one), ("tag", &five)]);
    file.add_value_parameter_port(
        module,
        "RESET",
        ParameterKind::Parameter,
        Some(&config),
        Some(&reset),
    );
    let state_t = file.make_extern_package_type("p", "state_t");
    let state =
        file.add_type_parameter_port(module, "State", ParameterKind::Parameter, Some(&state_t));
    let busy = file.make_extern_package_value_ref("p", "Busy");
    file.add_value_parameter_port(
        module,
        "S",
        ParameterKind::Parameter,
        Some(&state),
        Some(&busy),
    );
    let default_cfg = file.make_extern_value_ref("DEFAULT_CFG");
    file.add_value_parameter_port(
        module,
        "ALT",
        ParameterKind::Parameter,
        Some(&config),
        Some(&default_cfg),
    );

    assert_eq!(
        file.emit(),
        r#"module Leaf import p::DEFAULT_CFG; #(
  parameter type Config = p::cfg_t,
  parameter Config RESET = '{valid: 1, tag: 5},
  parameter type State = p::state_t,
  parameter State S = p::Busy,
  parameter Config ALT = DEFAULT_CFG
);

endmodule
"#
    );
}

#[test]
fn package_value_reference_is_an_atomic_expression() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let value = file.make_extern_package_value_ref("p", "MODE");
    let one = file.make_unsized_decimal_literal(1);
    let sum = file.make_add(&value, &one);
    assert_eq!(file.emit_expression(&sum), "p::MODE + 1");
}

#[test]
fn struct_pattern_can_supply_a_default_for_unspecified_members() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let one = file.make_unsized_decimal_literal(1);
    let zero = file.make_unsized_decimal_literal(0);
    let expression = file.make_struct_assignment_pattern(&[("valid", &one), ("default", &zero)]);
    assert_eq!(file.emit_expression(&expression), "'{valid: 1, default: 0}");
}

#[test]
#[should_panic(expected = "duplicate struct member valid")]
fn duplicate_struct_members_are_rejected() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let value = file.make_unsized_decimal_literal(1);
    file.make_struct_assignment_pattern(&[("valid", &value), ("valid", &value)]);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn struct_members_from_another_file_are_rejected() {
    let mut source = VastFile::new(VastFileType::SystemVerilog);
    let value = source.make_unsized_decimal_literal(1);
    let mut destination = VastFile::new(VastFileType::SystemVerilog);
    destination.make_struct_assignment_pattern(&[("valid", &value)]);
}

#[test]
#[should_panic(expected = "struct assignment pattern must not be empty")]
fn empty_struct_assignment_patterns_are_rejected() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    file.make_struct_assignment_pattern(&[]);
}

#[test]
#[should_panic(expected = "invalid package identifier")]
fn empty_package_name_is_rejected() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    file.make_extern_package_value_ref("", "Busy");
}

#[test]
#[should_panic(expected = "invalid external value identifier")]
fn external_reference_cannot_contain_an_arbitrary_expression() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    file.make_extern_value_ref("p::Busy + 1");
}

#[test]
#[should_panic(expected = "invalid struct assignment member identifier")]
fn struct_pattern_key_cannot_inject_a_declaration_fragment() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let one = file.make_unsized_decimal_literal(1);
    file.make_struct_assignment_pattern(&[("tag:0, other", &one)]);
}
