// SPDX-License-Identifier: Apache-2.0

use xlsynth_vast::{
    ParameterKind, SvArgument, TimeScaleMagnitude, TimeScaleValue, TimeUnit, VastFile, VastFileType,
};

#[test]
fn timed_and_string_parameters_keep_native_types_and_module_time_scale() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let leaf = file.add_module("Leaf");
    let string = file.make_string_type();
    let time = file.make_time_type(false);
    let longint = file.make_longint_type(true);
    let initial_text = file.make_string_literal("abc");
    let text = file.make_type_cast(&string, &initial_text).unwrap();
    let period = file
        .make_time_literal("1.5", TimeUnit::Nanoseconds)
        .unwrap();
    let count = file.make_unsized_decimal_literal(1);
    file.add_value_parameter_port(
        leaf,
        "TEXT",
        ParameterKind::Parameter,
        Some(&string),
        Some(&text),
    )
    .unwrap();
    file.add_value_parameter_port(
        leaf,
        "PERIOD",
        ParameterKind::Parameter,
        Some(&time),
        Some(&period),
    )
    .unwrap();
    file.add_value_parameter_port(
        leaf,
        "MaxCount",
        ParameterKind::Parameter,
        Some(&longint),
        Some(&count),
    )
    .unwrap();
    file.add_input(leaf, "message", &string).unwrap();
    file.add_output(leaf, "period", &time).unwrap();
    file.add_timeunit(
        leaf,
        TimeScaleValue {
            magnitude: TimeScaleMagnitude::One,
            unit: TimeUnit::Picoseconds,
        },
    )
    .unwrap();
    file.add_timeprecision(
        leaf,
        TimeScaleValue {
            magnitude: TimeScaleMagnitude::Hundred,
            unit: TimeUnit::Femtoseconds,
        },
    )
    .unwrap();
    assert_eq!(
        file.emit(),
        r#"module Leaf #(
  parameter string TEXT = string'("abc"),
  parameter time PERIOD = 1.5ns,
  parameter longint MaxCount = 1
) (
  input string message,
  output wire time period
);
  timeunit 1ps;
  timeprecision 100fs;
endmodule
"#
    );
}

#[test]
fn native_integer_atoms_report_width_and_signedness() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let byte = file.make_byte_type(true);
    let shortint = file.make_shortint_type(false);
    let longint = file.make_longint_type(true);
    let time = file.make_time_type(false);
    let signed_time = file.make_time_type(true);
    let string = file.make_string_type();
    for (ty, width, signed) in [
        (byte, 8, true),
        (shortint, 16, false),
        (longint, 64, true),
        (time, 64, false),
        (signed_time, 64, true),
    ] {
        assert_eq!(file.type_flat_bit_count_as_int64(ty).unwrap(), width);
        assert_eq!(file.type_is_signed(ty), signed);
    }
    assert!(
        file.type_flat_bit_count_as_int64(string)
            .unwrap_err()
            .to_string()
            .contains("dynamic")
    );
    let zero = file.make_unsized_decimal_literal(0);
    let cast = file.make_type_cast(&signed_time, &zero).unwrap();
    assert_eq!(file.emit_expression(&cast), "signed'(time'(0))");
    let bits = file
        .make_function_call_with_arguments("$bits", &[SvArgument::Type(signed_time)])
        .unwrap();
    assert_eq!(file.emit_expression(&bits), "$bits(time signed)");
}

#[test]
fn time_literal_magnitudes_are_validated_without_reformatting() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    for (magnitude, expected) in [
        ("1", "1ns"),
        ("1.5", "1.5ns"),
        ("1_000", "1_000ns"),
        ("1_", "1_ns"),
        ("1_.5", "1_.5ns"),
        ("1.5_", "1.5_ns"),
    ] {
        let literal = file
            .make_time_literal(magnitude, TimeUnit::Nanoseconds)
            .unwrap();
        assert_eq!(file.emit_expression(&literal), expected);
    }
    for magnitude in ["", "-1", "1ns", "1e3", ".5", "1.", "_1", "1.2.3", "1 "] {
        assert!(
            file.make_time_literal(magnitude, TimeUnit::Nanoseconds)
                .is_err(),
            "{magnitude:?}"
        );
    }
}

#[test]
fn string_cannot_be_an_inout_net() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let leaf = file.add_module("Leaf");
    let string = file.make_string_type();
    let error = file.add_inout(leaf, "invalid", &string).unwrap_err();
    assert!(error.to_string().contains("cannot be used as an inout net"));
}

#[test]
fn precision_cannot_be_coarser_than_the_local_unit() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let leaf = file.add_module("Leaf");
    file.add_timeunit(
        leaf,
        TimeScaleValue {
            magnitude: TimeScaleMagnitude::One,
            unit: TimeUnit::Nanoseconds,
        },
    )
    .unwrap();
    let before = file.emit();
    let error = file
        .add_timeprecision(
            leaf,
            TimeScaleValue {
                magnitude: TimeScaleMagnitude::Ten,
                unit: TimeUnit::Nanoseconds,
            },
        )
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "time precision cannot be coarser than time unit"
    );
    assert_eq!(file.emit(), before);
    file.add_timeprecision(
        leaf,
        TimeScaleValue {
            magnitude: TimeScaleMagnitude::One,
            unit: TimeUnit::Picoseconds,
        },
    )
    .unwrap();
}

#[test]
fn time_scale_cannot_follow_a_body_declaration() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let leaf = file.add_module("Leaf");
    let one = file.make_unsized_decimal_literal(1);
    file.add_parameter(leaf, "N", &one).unwrap();
    let before = file.emit();
    let error = file
        .add_timeunit(
            leaf,
            TimeScaleValue {
                magnitude: TimeScaleMagnitude::One,
                unit: TimeUnit::Nanoseconds,
            },
        )
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "time-scale declarations must precede other module members"
    );
    assert_eq!(file.emit(), before);
}

#[test]
fn hoisting_signal_declarations_keeps_time_scale_first() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let leaf = file.add_module("Leaf");
    file.add_timeunit(
        leaf,
        TimeScaleValue {
            magnitude: TimeScaleMagnitude::One,
            unit: TimeUnit::Nanoseconds,
        },
    )
    .unwrap();
    let scalar = file.make_scalar_type();
    file.add_wire(leaf, "w", &scalar).unwrap();
    file.hoist_module_declarations(leaf, 0);
    assert_eq!(
        file.emit(),
        "module Leaf;\n  timeunit 1ns;\n  wire w;\nendmodule\n"
    );
}

/// Native integral atom ports and nets must retain their declared widths.
#[test]
fn int_and_integer_ports_preserve_atom_types() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Integral");
    let int = file.make_int_type(true);
    let integer = file.make_integer_type(true);
    let unsigned_integer = file.make_integer_type(false);
    file.add_input(module, "input_int", &int).unwrap();
    file.add_output(module, "output_int", &int).unwrap();
    file.add_input(module, "input_integer", &integer).unwrap();
    file.add_inout(module, "inout_integer", &unsigned_integer)
        .unwrap();
    file.add_wire(module, "wire_integer", &integer).unwrap();
    file.add_logic(module, "logic_int", &int).unwrap();
    assert_eq!(
        file.emit(),
        r#"module Integral(
  input int input_int,
  output int output_int,
  input wire integer input_integer,
  inout wire integer unsigned inout_integer
);
  wire integer wire_integer;
  int logic_int;
endmodule
"#
    );
}
