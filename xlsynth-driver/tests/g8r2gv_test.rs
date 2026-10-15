// SPDX-License-Identifier: Apache-2.0

#![cfg(unix)]

use std::fmt::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

use prost::Message;
use xlsynth_g8r::aig::{
    ClockPort, GateBuilder, GateBuilderOptions, SequentialGateFn, add_input_registers,
    add_output_registers,
};
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
fn fixture(dir: &Path, design: &SequentialGateFn) -> anyhow::Result<Library> {
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
    let design = design
        .legalize_register_controls()
        .map_err(anyhow::Error::msg)?;
    let ni: usize = design
        .transition
        .inputs
        .iter()
        .map(|p| p.get_bit_count())
        .sum();
    let no: usize = design
        .outputs
        .iter()
        .map(|id| design.transition.outputs[id.index()].get_bit_count())
        .sum::<usize>()
        + design
            .registers
            .iter()
            .map(|r| design.transition.inputs[r.q.index()].get_bit_count())
            .sum::<usize>();
    let ports = (0..ni)
        .map(|i| format!("i{i}"))
        .chain((0..no).map(|i| format!("o{i}")))
        .collect::<Vec<_>>();
    let mut gv = format!("module mapped({});\n", ports.join(","));
    for i in 0..ni {
        writeln!(gv, "input i{i};")?;
    }
    for i in 0..no {
        writeln!(gv, "output o{i}; assign o{i}=1'b0;")?;
    }
    gv.push_str("endmodule\n");
    let abc = dir.join("fake abc");
    std::fs::write(
        &abc,
        format!(
            r#"#!/bin/sh
if [ "$2" = "-c" ]; then echo 'fake ABC'; exit 0; fi
test "$1" = '-s' && test "$2" = '-f' && test "$3" = 'map.abc' || exit 2
test -s transition.aig && test -s timing.constr && test -f library_0.lib || exit 3
cat <<'NETLIST' > mapped.v
{gv}NETLIST
"#
        ),
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
        fixture(dir.path(), &design).unwrap();
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
    builder.add_output("y".into(), a);
    let design = SequentialGateFn::from_gate_fn(builder.build());
    let clock = ClockPort { name: "clk".into() };
    let wrapped =
        add_output_registers(&add_input_registers(&design, clock.clone()).unwrap(), clock).unwrap();
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path(), &wrapped).unwrap();
    let input = dir.path().join("input.g8r");
    std::fs::write(&input, emit_g8r(&design).unwrap()).unwrap();
    let script = dir.path().join("passes.abc");
    std::fs::write(&script, "&get -n; &nf; &put\n# no trailing newline").unwrap();
    let diagnostics = dir.path().join("diagnostics.json");
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
    .arg(&diagnostics);
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
    assert_eq!(result.stdout, cmd.output().unwrap().stdout);
    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&diagnostics).unwrap()).unwrap();
    json["executable"] = "<abc>".into();
    json["raw_liberty_files"] = serde_json::json!(["<liberty>"]);
    compare_golden_text(
        &(serde_json::to_string_pretty(&json).unwrap() + "\n"),
        "tests/g8r2gv_diagnostics.golden.txt",
    );
}

#[test]
fn explicit_controls_cleanup_and_clock_validation() {
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
    fixture(dir.path(), &design).unwrap();
    let input = dir.path().join("input.g8rbin");
    std::fs::write(&input, encode_g8r_binary(&design).unwrap()).unwrap();
    for cleanup in [false, true] {
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
        let parsed = read_gv_from_str(&String::from_utf8(result.stdout).unwrap()).unwrap();
        assert_eq!(parsed.modules[0].instances.len(), 2);
    }
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
fn defaults_resolve_independently_and_invalid_options_fail_cleanly() {
    let mut builder = GateBuilder::new("top".into(), GateBuilderOptions::no_opt());
    let a = builder.add_input("a".into(), 1);
    builder.add_output("y".into(), a);
    let design = SequentialGateFn::from_gate_fn(builder.build());
    let dir = tempfile::tempdir().unwrap();
    let library = fixture(dir.path(), &design).unwrap();
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
