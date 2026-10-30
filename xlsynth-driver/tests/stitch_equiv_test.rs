// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

const SPEC: &str = r#"
dslx_input_file = "pipe.x"
dslx_top = "pipe"
flop_inputs = false
flop_outputs = true
"#;

const LIBERTY: &str = r#"
format_magic: 5496997758228255311
cells: {
  name: "DFF"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4 }
  sequential: { state_var: "IQ" next_state: "D" clock_expr: "CLK" kind: SEQUENTIAL_KIND_FF }
}
interned_strings: ["D", "CLK", "Q", "IQ"]
"#;

const NETLIST: &str = r#"
module mapped(out, clk, x);
  input x, clk;
  output out;
  wire middle;
  DFF r0 (.D(x), .CLK(clk), .Q(middle));
  DFF r1 (.D(middle), .CLK(clk), .Q(out));
endmodule
"#;

fn driver(directory: &Path, args: &[&str]) -> std::io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(directory)
        .args(args)
        .output()
}

/// Both emitters consume the same relative-path spec and preserve their
/// flag-based output.
#[test]
fn stitch_generators_load_spec_from_another_directory() {
    let temp = tempfile::tempdir().unwrap();
    let design = temp.path().join("design");
    fs::create_dir(&design).unwrap();
    fs::write(design.join("pipe.stitch.toml"), SPEC).unwrap();
    fs::write(design.join("pipe.x"), "fn pipe_cycle0(x: u2) -> u2 { x }").unwrap();
    for command in ["dslx-stitch-pipeline", "dslx-stitch-g8r-pipeline"] {
        let from_spec =
            driver(temp.path(), &[command, "--spec", "design/pipe.stitch.toml"]).unwrap();
        let from_flags = driver(
            temp.path(),
            &[
                command,
                "--dslx_input_file",
                "design/pipe.x",
                "--dslx_top",
                "pipe",
                "--flop_inputs=false",
                "--flop_outputs=true",
            ],
        )
        .unwrap();
        for output in [&from_spec, &from_flags] {
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        assert_eq!(from_spec.stdout, from_flags.stdout);
        let conflicting = driver(
            temp.path(),
            &[
                command,
                "--spec",
                "design/pipe.stitch.toml",
                "--flop_inputs=true",
            ],
        )
        .unwrap();
        assert_eq!(conflicting.status.code(), Some(2));
    }
}

/// A working-directory module can change the generated logic even when the
/// shadowed transitive import contributes only a constant to the stage.
#[test]
fn stitch_spec_imports_are_independent_of_the_working_directory() {
    let temp = tempfile::tempdir().unwrap();
    let design = temp.path().join("design");
    let imports = design.join("imports");
    let callers = temp.path().join("callers");
    let clean = callers.join("clean");
    let direct = callers.join("direct");
    let transitive = callers.join("transitive");
    fs::create_dir_all(&imports).unwrap();
    fs::create_dir_all(&clean).unwrap();
    fs::create_dir(&direct).unwrap();
    fs::create_dir(&transitive).unwrap();
    fs::write(
        design.join("pipe.toml"),
        format!("{SPEC}\ndslx_path = [\"imports\"]\n"),
    )
    .unwrap();
    fs::write(
        design.join("pipe.x"),
        r#"import stitch_helper;
fn pipe_cycle0(x: bool) -> bool { stitch_helper::apply(x) }
"#,
    )
    .unwrap();
    fs::write(
        imports.join("stitch_helper.x"),
        r#"import stitch_constant;
pub fn apply(x: bool) -> bool { x ^ stitch_constant::INVERT }
"#,
    )
    .unwrap();
    fs::write(
        imports.join("stitch_constant.x"),
        "pub const INVERT = false;",
    )
    .unwrap();
    fs::write(
        direct.join("stitch_helper.x"),
        "pub fn apply(x: bool) -> bool { !x }",
    )
    .unwrap();
    fs::write(
        transitive.join("stitch_constant.x"),
        "pub const INVERT = true;",
    )
    .unwrap();

    fs::write(
        callers.join("pipe.gv"),
        r#"module pipe(input clk, input x, output out);
  DFF r0 (.D(x), .CLK(clk), .Q(out));
endmodule"#,
    )
    .unwrap();
    fs::write(callers.join("cells.textproto"), LIBERTY).unwrap();
    for command in [
        "dslx-stitch-pipeline",
        "dslx-stitch-g8r-pipeline",
        "stitch-equiv",
    ] {
        let mut args = vec![command, "--spec", "../../design/pipe.toml"];
        if command == "dslx-stitch-g8r-pipeline" {
            args.extend(["--bin-out", "pipe.g8rbin"]);
        }
        if command == "stitch-equiv" {
            args.extend([
                "--netlist",
                "../pipe.gv",
                "--liberty-proto",
                "../cells.textproto",
            ]);
        }
        let expected = driver(&clean, &args).unwrap();
        assert!(
            expected.status.success(),
            "{command}: {}",
            String::from_utf8_lossy(&expected.stderr)
        );
        for cwd in [&direct, &transitive] {
            let shadowed = driver(cwd, &args).unwrap();
            assert!(
                shadowed.status.success(),
                "{command}: {}",
                String::from_utf8_lossy(&shadowed.stderr)
            );
            assert_eq!(shadowed.stdout, expected.stdout, "{command} from {cwd:?}");
            if command == "dslx-stitch-g8r-pipeline" {
                assert_eq!(
                    fs::read(cwd.join("pipe.g8rbin")).unwrap(),
                    fs::read(clean.join("pipe.g8rbin")).unwrap()
                );
            }
        }
    }
}

/// A valid-enabled design needs a different observation contract even when
/// the mapped implementation uses the same control names.
#[test]
fn stitch_equiv_reports_configured_valid_and_reset_ports() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("pipe.x"),
        "fn pipe_cycle0(x: bool) -> bool { x }",
    )
    .unwrap();
    fs::write(
        temp.path().join("pipe.toml"),
        format!(
            "{SPEC}\n{}",
            r#"input_valid_signal = "vin"
output_valid_signal = "vout"
reset = "rst_n"
reset_active_low = true"#
        ),
    )
    .unwrap();
    let result = driver(
        temp.path(),
        &[
            "stitch-equiv",
            "--spec",
            "pipe.toml",
            "--netlist",
            "unused.gv",
            "--liberty-proto",
            "unused.proto",
        ],
    )
    .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stderr).unwrap(),
        r#"xlsynth-driver: stitch-equiv: unsupported stitch controls for 'pipe': input_valid_signal = "vin", output_valid_signal = "vout", reset = "rst_n". stitch-equiv checks free-running pipelines and compares data on every cycle after fill. Valid-qualified observations and reset sequences are not yet supported
"#
    );
}

/// Every source stage must resolve before proving the latency of the selected
/// pipeline, including stages lowered with the implicit-token convention.
#[test]
fn stitch_equiv_requires_a_data_entry_point_for_every_stage() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("pipe.toml"), SPEC).unwrap();
    let source = temp.path().join("pipe.x");
    let private_stage = r#"fn pipe_cycle0(x: bool) -> bool { x }
fn pipe_cycle1(x: bool) -> bool { assert!(x, "OK"); x }
"#;
    fs::write(&source, private_stage).unwrap();
    let args = [
        "stitch-equiv",
        "--spec",
        "pipe.toml",
        "--netlist",
        "mapped.gv",
        "--liberty-proto",
        "cells.textproto",
        "--module-name",
        "mapped",
    ];
    // One flop exactly implements the silently truncated first stage.
    fs::write(
        temp.path().join("mapped.gv"),
        r#"module mapped(input x, input clk, output out);
  DFF r0 (.D(x), .CLK(clk), .Q(out));
endmodule"#,
    )
    .unwrap();
    fs::write(temp.path().join("cells.textproto"), LIBERTY).unwrap();
    let result = driver(temp.path(), &args).unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stderr).unwrap(),
        "xlsynth-driver: stitch-equiv: xlsynth error: stage 'pipe_cycle1' uses the implicit-token calling convention and has no data-only IR entry point. Declare the stage pub to emit a wrapper for its packed data ports\n"
    );

    // XLS emits a data-only wrapper for the public stage. Both stages must
    // contribute to the proved data path and its latency.
    fs::write(
        &source,
        private_stage.replace("fn pipe_cycle1", "pub fn pipe_cycle1"),
    )
    .unwrap();
    fs::write(temp.path().join("mapped.gv"), NETLIST).unwrap();
    let result = driver(temp.path(), &args).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        result.stdout,
        b"Proved pipe equivalent to mapped after 2 rising clock edges.\n"
    );
}

/// Reports the selected DSLX stage and IR operation when gatification fails.
#[test]
fn stitch_equiv_reports_unsupported_stage_operation() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("pipe.x"),
        "fn pipe_cycle0(x: u4, y: u4) -> (u4, u4) { umulp(x, y) }",
    )
    .unwrap();
    fs::write(
        temp.path().join("pipe.toml"),
        SPEC.replace("flop_outputs = true", "flop_outputs = false"),
    )
    .unwrap();
    fs::write(
        temp.path().join("pipe.gv"),
        "module pipe(input [3:0] x, input [3:0] y, output [7:0] out); assign out = {x, y}; endmodule",
    )
    .unwrap();
    fs::write(temp.path().join("cells.textproto"), LIBERTY).unwrap();
    let result = driver(
        temp.path(),
        &[
            "stitch-equiv",
            "--spec",
            "pipe.toml",
            "--netlist",
            "pipe.gv",
            "--liberty-proto",
            "cells.textproto",
        ],
    )
    .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(result.stdout.is_empty());
    assert_eq!(
        String::from_utf8(result.stderr).unwrap(),
        concat!(
            "xlsynth-driver: stitch-equiv: xlsynth error: failed to lower pipeline stage ",
            "'pipe_cycle0' to g8r: Unsupported node payload Binop(Umulp, ",
            "NodeRef { index: 1 }, NodeRef { index: 2 })\n",
        )
    );
}

/// The post-fill proof detects a bypassed register even when the arithmetic is
/// identical.
#[test]
fn stitch_equiv_checks_clocked_netlist() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("pipe.stitch.toml"), SPEC).unwrap();
    fs::write(
        temp.path().join("pipe.x"),
        r#"
fn pipe_cycle0(x: u1) -> u1 { x }
fn pipe_cycle1(x: u1) -> u1 { x }
"#,
    )
    .unwrap();
    fs::write(temp.path().join("cells.textproto"), LIBERTY).unwrap();
    fs::write(temp.path().join("mapped.gv"), NETLIST).unwrap();
    let args = [
        "stitch-equiv",
        "--spec",
        "pipe.stitch.toml",
        "--netlist",
        "mapped.gv",
        "--liberty-proto",
        "cells.textproto",
        "--module-name",
        "mapped",
    ];
    let result = driver(temp.path(), &args).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        result.stdout,
        b"Proved pipe equivalent to mapped after 2 rising clock edges.\n"
    );

    fs::write(
        temp.path().join("mapped.gv"),
        NETLIST.replace(".D(middle)", ".D(x)"),
    )
    .unwrap();
    let wrong_latency = driver(temp.path(), &args).unwrap();
    assert_eq!(wrong_latency.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&wrong_latency.stderr).contains("post-fill output mismatch"));
}

/// The stage boundary gives the first parameter the low bit and the second
/// parameter the next two bits of the registered tuple.
#[test]
fn stitch_equiv_checks_unequal_width_stage_parameter_slices() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("pipe.x"),
        r#"fn pipe_cycle0(x: u3) -> (u1, u2) { (x[2+:u1], x[0+:u2]) }
fn pipe_cycle1(a: u1, b: u2) -> u3 { a ++ b }
"#,
    )
    .unwrap();
    fs::write(
        temp.path().join("pipe.toml"),
        SPEC.replace("flop_outputs = true", "flop_outputs = false"),
    )
    .unwrap();
    fs::write(temp.path().join("cells.textproto"), LIBERTY).unwrap();
    let mapped = temp.path().join("pipe.gv");
    // The first stage packs {x[2], x[1:0]}, so its registered result is q.
    // The second stage concatenates its two parameter slices in order.
    let gv = r#"module pipe(input clk, input [2:0] x, output [2:0] out);
  wire [2:0] q;
  DFF r0 (.CLK(clk), .D(x[0]), .Q(q[0]));
  DFF r1 (.CLK(clk), .D(x[1]), .Q(q[1]));
  DFF r2 (.CLK(clk), .D(x[2]), .Q(q[2]));
  assign out = {q[0], q[2:1]};
endmodule"#;
    let args = [
        "stitch-equiv",
        "--spec",
        "pipe.toml",
        "--netlist",
        "pipe.gv",
        "--liberty-proto",
        "cells.textproto",
    ];
    fs::write(&mapped, gv).unwrap();
    let result = driver(temp.path(), &args).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        result.stdout,
        b"Proved pipe equivalent to pipe after 1 rising clock edge.\n"
    );

    fs::write(&mapped, gv.replace("{q[0], q[2:1]}", "{q[2], q[1:0]}")).unwrap();
    let reversed = driver(temp.path(), &args).unwrap();
    assert_eq!(reversed.status.code(), Some(1));
    assert!(reversed.stdout.is_empty());
    // The solver can choose any input whose rotated value differs from x.
    assert_eq!(
        String::from_utf8(reversed.stderr).unwrap().lines().next(),
        Some("xlsynth-driver: stitch-equiv: post-fill output mismatch")
    );
}

/// Native clockless mapping and an RTL wrapper's unused clock have the same
/// zero-edge contract.
#[test]
fn stitch_equiv_checks_gate_semantics_with_and_without_a_clock_port() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("pipe.x"),
        "fn pipe_cycle0(x: bool, en: bool) -> bool { gate!(en, x) }",
    )
    .unwrap();
    fs::write(
        temp.path().join("pipe.toml"),
        SPEC.replace("flop_outputs = true", "flop_outputs = false"),
    )
    .unwrap();
    fs::write(
        temp.path().join("cells.textproto"),
        r#"
format_magic: 5496997758228255311
cells: {
  name: "AND2"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: INPUT }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4 }
}

interned_strings: ["A", "B", "Y", "A & B"]
"#,
    )
    .unwrap();
    for (clock, out) in [("", "output_value"), ("input clk, ", "out")] {
        fs::write(temp.path().join("pipe.gv"), format!(
            "module pipe({clock}input en, input x, output {out}); AND2 g(.A(en), .B(x), .Y({out})); endmodule"
        )).unwrap();
        let result = driver(
            temp.path(),
            &[
                "stitch-equiv",
                "--spec",
                "pipe.toml",
                "--netlist",
                "pipe.gv",
                "--liberty-proto",
                "cells.textproto",
            ],
        )
        .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            result.stdout,
            b"Proved pipe equivalent to pipe after 0 rising clock edges.\n"
        );
    }
}

/// The CLI preserves the Liberty decoder's remediation for an obsolete proto.
#[test]
fn stitch_equiv_reports_unsupported_liberty_header() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("pipe.toml"), SPEC).unwrap();
    fs::write(
        temp.path().join("pipe.x"),
        "fn pipe_cycle0(x: u1) -> u1 { x }",
    )
    .unwrap();
    fs::write(temp.path().join("mapped.gv"), NETLIST).unwrap();
    fs::write(temp.path().join("old.proto"), []).unwrap();
    let result = driver(
        temp.path(),
        &[
            "stitch-equiv",
            "--spec",
            "pipe.toml",
            "--netlist",
            "mapped.gv",
            "--liberty-proto",
            "old.proto",
            "--module-name",
            "mapped",
        ],
    )
    .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(result.stdout.is_empty());
    assert_eq!(
        String::from_utf8(result.stderr).unwrap(),
        concat!(
            "xlsynth-driver: stitch-equiv: decoding Liberty proto 'old.proto': ",
            "invalid Liberty proto header: magic=0x0000000000000000; ",
            "regenerate from source Liberty with the current importer (older protos may omit functional semantics)\n",
        )
    );
}

/// An input boundary without data or valid adds no fill edge, including in a
/// pipeline that still has registers between stages.
#[test]
fn stitch_equiv_observes_nullary_stages_at_their_actual_latency() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("pipe.x");
    let mapped = temp.path().join("pipe.gv");
    let nullary_stage = "fn pipe_cycle0() -> bool { true }";
    fs::write(&source, nullary_stage).unwrap();
    fs::write(
        temp.path().join("pipe.toml"),
        r#"dslx_input_file = "pipe.x"
dslx_top = "pipe"
flop_inputs = true
flop_outputs = false"#,
    )
    .unwrap();
    fs::write(temp.path().join("cells.textproto"), LIBERTY).unwrap();
    let args = [
        "stitch-equiv",
        "--spec",
        "pipe.toml",
        "--netlist",
        "pipe.gv",
        "--liberty-proto",
        "cells.textproto",
    ];
    for clock in ["", "input clk, "] {
        fs::write(
            &mapped,
            format!("module pipe({clock}output out); assign out = 1'b1; endmodule"),
        )
        .unwrap();
        let result = driver(temp.path(), &args).unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            result.stdout,
            b"Proved pipe equivalent to pipe after 0 rising clock edges.\n"
        );
    }

    fs::write(
        &mapped,
        r#"module pipe(input clk, output out);
  DFF r0 (.D(1'b1), .CLK(clk), .Q(out));
endmodule"#,
    )
    .unwrap();
    let spurious_flop = driver(temp.path(), &args).unwrap();
    assert_eq!(spurious_flop.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(spurious_flop.stderr).unwrap(),
        r#"xlsynth-driver: stitch-equiv: post-fill output mismatch
Reference initial state {}
Mapped initial state {"r0__IQ": bits[1]:0}
Frame 0 inputs {}
Reference outputs {"out": bits[1]:1}
Mapped outputs {"out": bits[1]:0}
"#
    );

    fs::write(
        &source,
        format!("{nullary_stage}\nfn pipe_cycle1(x: bool) -> bool {{ x }}"),
    )
    .unwrap();
    let one_boundary = driver(temp.path(), &args).unwrap();
    assert!(
        one_boundary.status.success(),
        "{}",
        String::from_utf8_lossy(&one_boundary.stderr)
    );
    assert_eq!(
        one_boundary.stdout,
        b"Proved pipe equivalent to pipe after 1 rising clock edge.\n"
    );

    fs::write(
        &mapped,
        r#"module pipe(input clk, output out);
  wire m;
  DFF r0 (.D(1'b1), .CLK(clk), .Q(m));
  DFF r1 (.D(m), .CLK(clk), .Q(out));
endmodule"#,
    )
    .unwrap();
    let extra_edge = driver(temp.path(), &args).unwrap();
    assert_eq!(extra_edge.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&extra_edge.stderr).contains("post-fill output mismatch"));
}
