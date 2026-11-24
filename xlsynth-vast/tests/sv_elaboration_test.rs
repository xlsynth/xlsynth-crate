// SPDX-License-Identifier: Apache-2.0

//! Semantic validation of emitted parameterized SystemVerilog with Slang.

use std::path::Path;
use std::process::{Command, Output};

use xlsynth_vast::{
    ParameterKind, SvArgument, TimeScaleMagnitude, TimeScaleValue, TimeUnit, VastFile, VastFileType,
};

/// Elaborates emitted SV and verifies that a false semantic assertion is
/// rejected.
#[test]
fn parameterized_sv_preserves_types_values_and_widths() {
    let Some(slang) = std::env::var_os("SLANG_PATH") else {
        eprintln!("Skipping SV elaboration: set SLANG_PATH to a Slang executable");
        return;
    };
    assert!(!slang.is_empty(), "SLANG_PATH is set but empty");
    let dir = tempfile::tempdir().expect("temporary SV directory");
    let path = dir.path().join("parameters.sv");
    let source = representative_design();
    std::fs::write(&path, &source).expect("write SV fixture");
    let output = elaborate(Path::new(&slang), &path);
    assert!(
        output.status.success(),
        "Slang rejected emitted SV:\n{}",
        diagnostics(&output)
    );

    for (before, after, expected_diagnostic) in [
        (".EXPECTED_MODE(0)", ".EXPECTED_MODE(1)", "BAD_MODE"),
        (".Cfg(p::wide_cfg_t)", ".Cfg(p::cfg_t)", "BAD_TYPE_ARGUMENT"),
    ] {
        let negative = source.replace(before, after);
        assert_ne!(negative, source);
        std::fs::write(&path, negative).expect("write negative control");
        let output = elaborate(Path::new(&slang), &path);
        assert!(
            !output.status.success(),
            "incorrect assertion unexpectedly elaborated"
        );
        assert!(
            diagnostics(&output).contains(expected_diagnostic),
            "unexpected negative-control failure:\n{}",
            diagnostics(&output)
        );
    }
}

/// Runs the configured compiler against the generated top.
fn elaborate(slang: &Path, source: &Path) -> Output {
    Command::new(slang)
        .args(["--top", "Top"])
        .arg(source)
        .output()
        .expect("run SLANG_PATH")
}

/// Combines compiler diagnostics for actionable test failures.
fn diagnostics(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// Builds two differently parameterized instances and independent semantic
/// checks.
fn representative_design() -> String {
    let mut f = VastFile::new(VastFileType::SystemVerilog);
    let leaf = f.add_module("Leaf");
    f.add_module_import(leaf, "p", "*").unwrap();
    let int = f.make_int_type(true);
    let logic = f.make_scalar_type();
    let bit = f.make_bit_type();
    let string = f.make_string_type();
    let time = f.make_time_type(false);
    let cfg_t = f.make_extern_type("cfg_t");
    let eight = f.make_unsized_decimal_literal(8);
    let zero = f.make_unsized_decimal_literal(0);
    let one = f.make_unsized_decimal_literal(1);
    let five = f.make_unsized_decimal_literal(5);
    let x = f.make_unsized_x_literal();
    let width = f
        .add_value_parameter_port(
            leaf,
            "WIDTH",
            ParameterKind::Parameter,
            Some(&int),
            Some(&eight),
        )
        .unwrap();
    f.add_value_parameter_port(
        leaf,
        "MODE",
        ParameterKind::Parameter,
        Some(&int),
        Some(&zero),
    )
    .unwrap();
    f.add_value_parameter_port(leaf, "REQUIRED", ParameterKind::Parameter, Some(&int), None)
        .unwrap();
    let cfg = f
        .add_type_parameter_port(leaf, "Cfg", ParameterKind::Parameter, Some(&cfg_t))
        .unwrap();
    let reset = f
        .make_struct_assignment_pattern(&[("valid", &one), ("tag", &five)])
        .unwrap();
    f.add_value_parameter_port(
        leaf,
        "RESET",
        ParameterKind::Parameter,
        Some(&cfg),
        Some(&reset),
    )
    .unwrap();
    f.add_value_parameter_port(leaf, "TWO", ParameterKind::Parameter, Some(&bit), Some(&x))
        .unwrap();
    f.add_value_parameter_port(
        leaf,
        "FOUR",
        ParameterKind::Parameter,
        Some(&logic),
        Some(&x),
    )
    .unwrap();
    let period = f.make_time_literal("1.5", TimeUnit::Nanoseconds).unwrap();
    f.add_value_parameter_port(
        leaf,
        "PERIOD",
        ParameterKind::Parameter,
        Some(&time),
        Some(&period),
    )
    .unwrap();
    let text = f.make_string_literal("a\rb");
    f.add_value_parameter_port(
        leaf,
        "TEXT",
        ParameterKind::Parameter,
        Some(&string),
        Some(&text),
    )
    .unwrap();
    let bits_ty = f.make_two_state_bit_vector_type(4, false).unwrap();
    let raw = f.make_four_state_binary_literal("10xz").unwrap();
    let bits = f
        .add_value_parameter_port(
            leaf,
            "BITS",
            ParameterKind::Parameter,
            Some(&bits_ty),
            Some(&raw),
        )
        .unwrap();
    let port_type = f
        .make_bit_vector_type_expr(&width.to_expr(), false)
        .unwrap();
    f.add_output(leaf, "q", &port_type).unwrap();
    f.add_timeunit(
        leaf,
        TimeScaleValue {
            magnitude: TimeScaleMagnitude::One,
            unit: TimeUnit::Picoseconds,
        },
    )
    .unwrap();
    f.add_timeprecision(
        leaf,
        TimeScaleValue {
            magnitude: TimeScaleMagnitude::Hundred,
            unit: TimeUnit::Femtoseconds,
        },
    )
    .unwrap();
    let cfg_bits = f
        .make_function_call_with_arguments("$bits", &[SvArgument::Type(cfg)])
        .unwrap();
    f.add_localparam(leaf, "CFG_BITS", &cfg_bits).unwrap();
    let macro_bits = f
        .make_macro_ref_with_arguments(
            "BITS_PLUS",
            &[SvArgument::Type(cfg), SvArgument::Value(one)],
        )
        .unwrap();
    f.add_localparam(leaf, "MACRO_BITS", &macro_bits.to_expr())
        .unwrap();
    let logic_vec = f.make_bit_vector_type(4, false).unwrap();
    let cast = f.make_type_cast(&logic_vec, &bits.to_expr()).unwrap();
    let type_name = f.make_function_call("$typename", &[&cast]).unwrap();
    f.add_localparam(leaf, "CAST_TYPE", &type_name).unwrap();
    let cast_raw = f.make_type_cast(&logic_vec, &raw).unwrap();
    let has_x = f.make_function_call("$isunknown", &[&cast_raw]).unwrap();
    f.add_localparam(leaf, "CAST_HAS_X", &has_x).unwrap();

    let top = f.add_module("Top");
    let q8 = f.make_bit_vector_type(8, false).unwrap();
    let q12 = f.make_bit_vector_type(12, false).unwrap();
    let qa = f.add_wire(top, "qa", &q8).unwrap();
    let qb = f.add_wire(top, "qb", &q12).unwrap();
    let required = f.make_unsized_decimal_literal(42);
    let twelve = f.make_unsized_decimal_literal(12);
    let qualified_cfg = f.make_extern_package_type("p", "wide_cfg_t");
    let a = f
        .make_instantiation(
            "Leaf",
            "a",
            &["REQUIRED"],
            &[&required],
            &["q"],
            &[Some(&qa.to_expr())],
        )
        .unwrap();
    f.add_member_instantiation(top, a).unwrap();
    let b = f
        .make_instantiation_with_arguments(
            "Leaf",
            "b",
            &["MODE", "WIDTH", "REQUIRED", "Cfg"],
            &[
                SvArgument::Value(one),
                SvArgument::Value(twelve),
                SvArgument::Value(required),
                SvArgument::Type(qualified_cfg),
            ],
            &["q"],
            &[Some(&qb.to_expr())],
        )
        .unwrap();
    f.add_member_instantiation(top, b).unwrap();
    format!("{}\n{}\n{}", PACKAGE, f.emit(), CHECKS)
}

const PACKAGE: &str = r#"`timescale 1ps/100fs
`define BITS_PLUS(type_, value_) ($bits(type_) + value_)
package p;
  typedef struct packed { logic valid; logic [2:0] tag; } cfg_t;
  typedef struct packed { logic valid; logic [3:0] tag; } wide_cfg_t;
endpackage"#;

const CHECKS: &str = r#"module CheckLeaf #(
  parameter int MODE=0, EXPECTED_MODE=0, WIDTH=0, EXPECTED_WIDTH=0, REQUIRED=0,
  parameter bit RESET_VALID=0, parameter int RESET_TAG=0,
  parameter bit TWO=1, parameter logic FOUR=0,
  parameter time PERIOD=0, parameter string TEXT="", CAST_TYPE="",
  parameter bit CAST_HAS_X=0, parameter bit [3:0] BITS='1, parameter int PORT_WIDTH=0,
  parameter int CFG_BITS=0, EXPECTED_CFG_BITS=0, MACRO_BITS=0
);
  if (MODE != EXPECTED_MODE) BAD_MODE bad_mode();
  if (WIDTH != EXPECTED_WIDTH || PORT_WIDTH != EXPECTED_WIDTH) BAD_WIDTH bad_width();
  if (REQUIRED != 42 || RESET_VALID != 1 || RESET_TAG != 5) BAD_PARAMS bad_params();
  if (TWO !== 0 || FOUR !== 1'bx || BITS !== 4'b1000) BAD_STATE bad_state();
  if (PERIOD != 1500 || TEXT.len() != 3 || TEXT[1] != 8'h0d) BAD_SCALARS bad_scalars();
  if (CAST_TYPE != "logic[3:0]" || CAST_HAS_X != 1) BAD_CAST bad_cast();
  if (CFG_BITS != EXPECTED_CFG_BITS || MACRO_BITS != EXPECTED_CFG_BITS + 1) BAD_TYPE_ARGUMENT bad_type_argument();
endmodule
bind Top.a CheckLeaf #(
  .MODE(MODE), .EXPECTED_MODE(0), .WIDTH(WIDTH), .EXPECTED_WIDTH(8),
  .REQUIRED(REQUIRED), .RESET_VALID(RESET.valid), .RESET_TAG(RESET.tag), .TWO(TWO), .FOUR(FOUR),
  .PERIOD(PERIOD), .TEXT(TEXT), .CAST_TYPE(CAST_TYPE), .CAST_HAS_X(CAST_HAS_X),
  .BITS(BITS), .PORT_WIDTH($bits(q)), .CFG_BITS(CFG_BITS), .EXPECTED_CFG_BITS(4), .MACRO_BITS(MACRO_BITS)
) check_a();
bind Top.b CheckLeaf #(
  .MODE(MODE), .EXPECTED_MODE(1), .WIDTH(WIDTH), .EXPECTED_WIDTH(12),
  .REQUIRED(REQUIRED), .RESET_VALID(RESET.valid), .RESET_TAG(RESET.tag), .TWO(TWO), .FOUR(FOUR),
  .PERIOD(PERIOD), .TEXT(TEXT), .CAST_TYPE(CAST_TYPE), .CAST_HAS_X(CAST_HAS_X),
  .BITS(BITS), .PORT_WIDTH($bits(q)), .CFG_BITS(CFG_BITS), .EXPECTED_CFG_BITS(5), .MACRO_BITS(MACRO_BITS)
) check_b();"#;
