// SPDX-License-Identifier: Apache-2.0

#![cfg(unix)]

use std::fmt::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

use prost::Message;
use xlsynth_g8r::aig::{AigBitVector, GateBuilder, GateBuilderOptions, SequentialGateFn};
use xlsynth_g8r::aig_serdes::g8r::{emit_g8r, encode_g8r_binary};
use xlsynth_g8r::block2sequential::block_ir_to_sequential_gate_fn_with_controls;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_g8r::liberty::model::library_to_proto;
use xlsynth_g8r::liberty::parser::parse_liberty_files;
use xlsynth_g8r::liberty_proto::Library;
use xlsynth_g8r::netlist::io::read_gv_from_str;
use xlsynth_g8r::test_utils::{interesting_ir_roundtrip_cases, load_interesting_ir_roundtrip_case};
use xlsynth_test_helpers::compare_golden_text;

/// Configures a fake ABC with the correct scalar interface, not a logic oracle.
fn fixture(dir: &Path) -> anyhow::Result<Library> {
    let raw = dir.join("cells.lib");
    std::fs::write(
        &raw,
        r#"library(test) {
        capacitive_load_unit(1, pf);
        cell(BUF) { area: 1; pin(A) { direction: input; }
            pin(Y) { direction: output; function: "A"; } }
        cell(INV) { area: 1; pin(A) { direction: input; }
            pin(Y) { direction: output; function: "!A"; } }
        cell(DFF) { area: 2; pin(D) { direction: input; } pin(CLK) { clock: true; }
            pin(Q) { direction: output; function: "IQ"; }
            ff(IQ,IQN) { clocked_on: "CLK"; next_state: "D"; } }
        cell(DFFN) { area: 3; pin(D) { direction: input; } pin(CLK) { clock: true; }
            pin(QN) { direction: output; function: "IQN"; }
            ff(IQ,IQN) { clocked_on: "CLK"; next_state: "D"; } }
    }"#,
    )?;
    let mut library = parse_liberty_files(&[&raw]).map_err(anyhow::Error::msg)?;
    library.reference_buffer_cell = Some("BUF".into());
    library.default_output_load = Some(0.004);
    let library = library_to_proto(library)?;
    std::fs::write(dir.join("cells.proto"), library.encode_to_vec())?;
    let abc = dir.join("fake abc");
    std::fs::write(
        &abc,
        r#"#!/bin/sh
if [ "$2" = "-c" ]; then echo 'fake ABC'; exit 0; fi
test "$1" = '-s' && test "$2" = '-f' && test "$3" = 'map.abc' || exit 2
test -s transition.aig && test -s timing.constr && test -f library_0.lib || exit 3
read -r magic m inputs l outputs a < transition.aig
printf '%s' "${XLSYNTH_TEST_ABC_STDOUT-}"
printf '%s' "${XLSYNTH_TEST_ABC_STDERR-}" >&2
exec > mapped.v
printf 'module mapped('
sep=''
i=0
while [ "$i" -lt "$inputs" ]; do printf '%si%s' "$sep" "$i"; sep=','; i=$((i+1)); done
i=0
while [ "$i" -lt "$outputs" ]; do printf '%so%s' "$sep" "$i"; sep=','; i=$((i+1)); done
printf ');\n'
i=0
while [ "$i" -lt "$inputs" ]; do printf 'input i%s;\n' "$i"; i=$((i+1)); done
i=0
while [ "$i" -lt "$outputs" ]; do printf "output o%s; assign o%s=1'b0;\n" "$i" "$i"; i=$((i+1)); done
printf 'endmodule\n'
"#,
    )?;
    std::fs::set_permissions(&abc, std::fs::Permissions::from_mode(0o755))?;
    Ok(library)
}

/// Builds the CLI invocation while keeping all external tools explicit.
fn command(dir: &Path, input: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
    command
        .arg("g8r2gv")
        .arg(input)
        .arg("--liberty")
        .arg(dir.join("cells.lib"))
        .arg("--liberty-proto")
        .arg(dir.join("cells.proto"))
        .arg("--abc-path")
        .arg(dir.join("fake abc"))
        .env("NO_COLOR", "1")
        .env_remove("RUST_LOG");
    command
}

#[test]
fn shared_signatures_accept_both_serializations_and_preserve_ports() {
    for case in interesting_ir_roundtrip_cases() {
        let design =
            SequentialGateFn::from_gate_fn(load_interesting_ir_roundtrip_case(case).gate_fn);
        let dir = tempfile::tempdir().unwrap();
        fixture(dir.path()).unwrap();
        let mut results = Vec::new();
        for (extension, bytes) in [
            ("g8r", emit_g8r(&design).unwrap().into_bytes()),
            ("g8rbin", encode_g8r_binary(&design).unwrap()),
        ] {
            let input = dir.path().join(format!("input.{extension}"));
            std::fs::write(&input, bytes).unwrap();
            let result = command(dir.path(), &input).output().unwrap();
            assert!(
                result.status.success(),
                "{}: {}",
                case.name,
                String::from_utf8_lossy(&result.stderr)
            );
            assert!(result.stderr.is_empty());
            let text = String::from_utf8(result.stdout).unwrap();
            let parsed = read_gv_from_str(&text).unwrap();
            let actual = parsed.modules[0]
                .ports
                .iter()
                .map(|p| {
                    (
                        parsed.interner.resolve(p.name).unwrap().to_owned(),
                        p.width.map_or(1, |(a, b)| (a - b + 1) as usize),
                    )
                })
                .collect::<Vec<_>>();
            let expected = design
                .inputs
                .iter()
                .map(|id| {
                    let p = &design.transition.inputs[id.index()];
                    (p.name.clone(), p.get_bit_count())
                })
                .chain(design.outputs.iter().map(|id| {
                    let p = &design.transition.outputs[id.index()];
                    (p.name.clone(), p.get_bit_count())
                }))
                .filter(|(_, width)| *width != 0)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "{}", case.name);
            assert!(parsed.modules[0].instances.is_empty());
            results.push(text);
        }
        assert_eq!(results[0], results[1]);
    }
}

#[test]
fn boundary_registers_script_constraints_and_diagnostics() {
    let mut builder = GateBuilder::new("top".into(), GateBuilderOptions::no_opt());
    let a = builder.add_input("a".into(), 2);
    let and = builder.add_and_binary(*a.get_lsb(0), *a.get_lsb(1));
    builder.add_output(
        "y".into(),
        AigBitVector::from_lsb_is_index_0(&[and, and.negate()]),
    );
    let design = SequentialGateFn::from_gate_fn(builder.build());
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path()).unwrap();
    let input = dir.path().join("input.g8r");
    std::fs::write(&input, emit_g8r(&design).unwrap()).unwrap();
    let script = dir.path().join("passes.abc");
    std::fs::write(&script, "&get -n; &nf; &put\n# no trailing newline").unwrap();
    let diagnostics = dir.path().join("diagnostics.json");
    let abc_stdout = dir.path().join("abc.stdout");
    let abc_stderr = dir.path().join("abc.stderr");
    let mut cmd = command(dir.path(), &input);
    cmd.args([
        "--add-clk-port",
        "clk",
        "--flop-inputs",
        "--flop-outputs",
        "--module-name",
        "mapped_top",
        "--register-cell",
        "DFFN",
        "--output-load-ff",
        "7",
    ])
    .arg("--abc-script")
    .arg(&script)
    .arg("--diagnostics-out")
    .arg(&diagnostics)
    .arg("--abc-stdout-out")
    .arg(&abc_stdout)
    .arg("--abc-stderr-out")
    .arg(&abc_stderr)
    .env("XLSYNTH_TEST_ABC_STDOUT", "elapsed: 0.01s\n")
    .env("XLSYNTH_TEST_ABC_STDERR", "peak memory: 100\n");
    let result = cmd.output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stderr.is_empty());
    compare_golden_text(
        &String::from_utf8(result.stdout.clone()).unwrap(),
        "tests/g8r2gv_boundary.golden.v",
    );
    assert_eq!(
        std::fs::read_to_string(&abc_stdout).unwrap(),
        "elapsed: 0.01s\n"
    );
    assert_eq!(
        std::fs::read_to_string(&abc_stderr).unwrap(),
        "peak memory: 100\n"
    );
    let first_diagnostics = std::fs::read(&diagnostics).unwrap();
    // Raw tool logs may vary between otherwise identical invocations. Neither
    // the netlist nor the diagnostics JSON should change with them.
    let repeated = cmd
        .env("XLSYNTH_TEST_ABC_STDOUT", "elapsed: 0.02s\n")
        .env("XLSYNTH_TEST_ABC_STDERR", "peak memory: 200\n")
        .output()
        .unwrap();
    assert!(
        repeated.status.success(),
        "{}",
        String::from_utf8_lossy(&repeated.stderr)
    );
    assert!(repeated.stderr.is_empty());
    assert_eq!(result.stdout, repeated.stdout);
    assert_eq!(first_diagnostics, std::fs::read(&diagnostics).unwrap());
    assert_eq!(
        std::fs::read_to_string(&abc_stdout).unwrap(),
        "elapsed: 0.02s\n"
    );
    assert_eq!(
        std::fs::read_to_string(&abc_stderr).unwrap(),
        "peak memory: 200\n"
    );
    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&diagnostics).unwrap()).unwrap();
    json["executable"] = "<abc>".into();
    json["raw_liberty_files"] = serde_json::json!(["<liberty>"]);
    compare_golden_text(
        &(serde_json::to_string_pretty(&json).unwrap() + "\n"),
        "tests/g8r2gv_diagnostics.golden.txt",
    );

    std::fs::remove_file(&diagnostics).unwrap();
    let binary = cmd.args(["--netlist-format", "pb"]).output().unwrap();
    assert!(
        binary.status.success(),
        "{}",
        String::from_utf8_lossy(&binary.stderr)
    );
    assert!(binary.stderr.is_empty());
    assert_eq!(first_diagnostics, std::fs::read(&diagnostics).unwrap());
    let pb = dir.path().join("mapped.pb");
    std::fs::write(&pb, binary.stdout).unwrap();
    let restored = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("pb2gv")
        .arg(&pb)
        .output()
        .unwrap();
    assert!(
        restored.status.success(),
        "{}",
        String::from_utf8_lossy(&restored.stderr)
    );
    assert_eq!(result.stdout, restored.stdout);
}

#[test]
fn explicit_controls_and_clock_validation() {
    let design = block_ir_to_sequential_gate_fn_with_controls(
        r#"package controls
top block top(clk: clock, rst: bits[1], data: bits[2], en: bits[1], out: bits[2]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg state(bits[2], reset_value=3)
  rst: bits[1] = input_port(name=rst, id=1)
  data: bits[2] = input_port(name=data, id=2)
  en: bits[1] = input_port(name=en, id=3)
  q: bits[2] = register_read(register=state, id=4)
  d: () = register_write(data, register=state, load_enable=en, reset=rst, id=5)
  out: () = output_port(q, name=out, id=6)
}
"#,
        GatifyOptions::all_opts_disabled(),
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path()).unwrap();
    let input = dir.path().join("input.g8rbin");
    std::fs::write(&input, encode_g8r_binary(&design).unwrap()).unwrap();
    let result = command(dir.path(), &input).output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let parsed = read_gv_from_str(&String::from_utf8(result.stdout).unwrap()).unwrap();
    assert_eq!(parsed.modules[0].instances.len(), 2);
    let rejected = command(dir.path(), &input)
        .args(["--add-clk-port", "other"])
        .output()
        .unwrap();
    assert_eq!(rejected.status.code(), Some(1));
    assert!(rejected.stdout.is_empty());
    assert_eq!(
        String::from_utf8(rejected.stderr).unwrap(),
        "xlsynth-driver: g8r2gv: --add-clk-port 'other' does not match the stored design clock 'clk'\n"
    );
}

#[test]
fn cleanup_registers_is_opt_in_and_merges_duplicate_state() {
    let design = block_ir_to_sequential_gate_fn_with_controls(
        r#"package duplicates
top block top(clk: clock, data: bits[1], out: bits[2]) {
  reg first(bits[1])
  reg second(bits[1])
  data: bits[1] = input_port(name=data, id=1)
  q0: bits[1] = register_read(register=first, id=2)
  q1: bits[1] = register_read(register=second, id=3)
  d0: () = register_write(data, register=first, id=4)
  d1: () = register_write(data, register=second, id=5)
  value: bits[2] = concat(q0, q1, id=6)
  out: () = output_port(value, name=out, id=7)
}
"#,
        GatifyOptions::all_opts_disabled(),
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path()).unwrap();
    let input = dir.path().join("input.g8r");
    std::fs::write(&input, emit_g8r(&design).unwrap()).unwrap();
    for (cleanup, expected_registers) in [(false, 2), (true, 1)] {
        let mut cmd = command(dir.path(), &input);
        if cleanup {
            cmd.arg("--cleanup-registers");
        }
        let result = cmd.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(result.stderr.is_empty());
        let parsed = read_gv_from_str(&String::from_utf8(result.stdout).unwrap()).unwrap();
        let module = &parsed.modules[0];
        // This design is all wires and registers, so no combinational cell can
        // hide a missing cleanup or an accidental polarity-repair inverter.
        assert_eq!(module.instances.len(), expected_registers);
        for instance in &module.instances {
            assert_eq!(parsed.interner.resolve(instance.type_name), Some("DFF"));
        }
        let output = module
            .ports
            .iter()
            .find(|p| parsed.interner.resolve(p.name) == Some("out"))
            .unwrap();
        assert_eq!(output.width, Some((1, 0)));
    }
}

#[test]
fn defaults_resolve_independently_and_invalid_options_fail_cleanly() {
    let mut builder = GateBuilder::new("top".into(), GateBuilderOptions::no_opt());
    let a = builder.add_input("a".into(), 1);
    builder.add_output("y".into(), a);
    let design = SequentialGateFn::from_gate_fn(builder.build());
    let dir = tempfile::tempdir().unwrap();
    let library = fixture(dir.path()).unwrap();
    let input = dir.path().join("input.g8r");
    std::fs::write(&input, emit_g8r(&design).unwrap()).unwrap();
    for (missing_driver, missing_load, flags) in [
        (true, false, vec!["--driving-cell", "BUF"]),
        (false, true, vec!["--output-load-ff", "9"]),
        (
            true,
            true,
            vec!["--driving-cell", "BUF", "--output-load-ff", "9"],
        ),
    ] {
        let mut lib = library.clone();
        if missing_driver {
            lib.reference_buffer_cell = None;
        }
        if missing_load {
            lib.default_output_load = None;
            lib.units = None;
        }
        std::fs::write(dir.path().join("cells.proto"), lib.encode_to_vec()).unwrap();
        let result = command(dir.path(), &input).args(flags).output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let missing = command(dir.path(), &input).output().unwrap();
        assert_eq!(missing.status.code(), Some(1));
        assert!(missing.stdout.is_empty());
    }
    std::fs::write(dir.path().join("cells.proto"), library.encode_to_vec()).unwrap();
    let mut errors = String::new();
    for flags in [
        vec!["--output-load-ff", "0"],
        vec!["--output-load-ff", "NaN"],
        vec!["--driving-cell", "INV"],
        vec!["--flop-inputs"],
    ] {
        let result = command(dir.path(), &input).args(&flags).output().unwrap();
        assert_eq!(result.status.code(), Some(1));
        assert!(result.stdout.is_empty());
        writeln!(
            errors,
            "{}\n{}",
            flags.join(" "),
            String::from_utf8(result.stderr).unwrap()
        )
        .unwrap();
    }
    compare_golden_text(
        &format!("{}\n", errors.trim_end()),
        "tests/g8r2gv_errors.golden.txt",
    );
    assert!(
        !command(dir.path(), &input)
            .args(["--timeout-seconds", "0"])
            .output()
            .unwrap()
            .status
            .success()
    );
}
