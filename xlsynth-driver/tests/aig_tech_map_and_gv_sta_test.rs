// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::ffi::OsStr;
#[cfg(unix)]
use std::io::Read;
#[cfg(unix)]
use std::os::fd::OwnedFd;
#[cfg(target_os = "linux")]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process::Command;
#[cfg(unix)]
use std::process::Stdio;
#[cfg(unix)]
use std::time::{Duration, Instant};

use test_case::test_case;
use xlsynth_g8r::aig::AigBitVector;
use xlsynth_g8r::aig::gate::GateFn;
use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
use xlsynth_g8r::aig_sim::gate_sim::{Collect, eval};
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty;
use xlsynth_g8r::netlist::io::{load_liberty_from_path, read_gv_from_str};
use xlsynth_pir::IrBits;

/// Builds the two-input AND fixture used by the structural mapping smoke test.
fn make_two_input_and_aig() -> String {
    let mut gb = GateBuilder::new("and2_demo".to_string(), GateBuilderOptions::no_opt());
    let a = gb.add_input("a".to_string(), 1);
    let b = gb.add_input("b".to_string(), 1);
    let y = gb.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
    gb.add_output("y".to_string(), AigBitVector::from_bit(y));
    emit_aiger(&gb.build(), true).expect("emit_aiger should succeed")
}

/// Provides INV/NAND2 cells with complete scalar timing data and known areas.
fn make_timing_enabled_inv_nand2_liberty_textproto() -> &'static str {
    r#"
format_magic: 5496997758228255311
cells: {
  name: "INV"
  pins: { name_string_id: 1 direction: INPUT capacitance: 0.01 }
  pins: {
    name_string_id: 2
    direction: OUTPUT
    function_string_id: 3
    timing_arcs: {
      related_pin_string_id: 1
      timing_sense: TIMING_SENSE_NEGATIVE_UNATE
      timing_type: TIMING_TYPE_COMBINATIONAL
      tables: { kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: 1.0 }
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 1.1 }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 0.3 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: 0.4 }
    }
  }
  area: 1.0
}
cells: {
  name: "NAND2"
  pins: { name_string_id: 1 direction: INPUT capacitance: 0.02 }
  pins: { name_string_id: 4 direction: INPUT capacitance: 0.02 }
  pins: {
    name_string_id: 2
    direction: OUTPUT
    function_string_id: 5
    timing_arcs: {
      related_pin_string_id: 1
      timing_sense: TIMING_SENSE_NEGATIVE_UNATE
      timing_type: TIMING_TYPE_COMBINATIONAL
      tables: { kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: 2.0 }
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 2.1 }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 0.5 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: 0.6 }
    }
    timing_arcs: {
      related_pin_string_id: 4
      timing_sense: TIMING_SENSE_NEGATIVE_UNATE
      timing_type: TIMING_TYPE_COMBINATIONAL
      tables: { kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: 2.0 }
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 2.1 }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 0.5 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: 0.6 }
    }
  }
  area: 2.0
}
default_output_load: 0.01
units: { time_unit: "1ps" capacitance_unit: "1pf" }
interned_strings: ["A", "Y", "!A", "B", "!(A*B)"]
lut_shapes: {}
"#
}

#[test]
fn aig_tech_map_then_gv_sta_smoke() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let temp_dir = tempfile::tempdir().expect("create temp dir");

    let aig_path = temp_dir.path().join("in.aag");
    let liberty_path = temp_dir.path().join("lib.textproto");
    let netlist_path = temp_dir.path().join("mapped.gv");

    std::fs::write(&aig_path, make_two_input_and_aig()).expect("write aiger");
    std::fs::write(
        &liberty_path,
        make_timing_enabled_inv_nand2_liberty_textproto(),
    )
    .expect("write liberty");

    let map_output = Command::new(driver)
        .arg("aig-tech-map")
        .arg(aig_path.as_os_str())
        .arg("--liberty_proto")
        .arg(liberty_path.as_os_str())
        .arg("--netlist_out")
        .arg(netlist_path.as_os_str())
        .output()
        .expect("aig-tech-map invocation should run");

    assert!(
        map_output.status.success(),
        "aig-tech-map failed: status={:?}\nstdout={}\nstderr={}",
        map_output.status,
        String::from_utf8_lossy(&map_output.stdout),
        String::from_utf8_lossy(&map_output.stderr)
    );

    let mapped_text = std::fs::read_to_string(&netlist_path).expect("read mapped netlist");
    assert!(mapped_text.contains("NAND2"));
    assert!(mapped_text.contains("INV"));

    let sta_output = Command::new(driver)
        .arg("gv-sta")
        .arg("--netlist")
        .arg(netlist_path.as_os_str())
        .arg("--liberty_proto")
        .arg(liberty_path.as_os_str())
        .output()
        .expect("gv-sta invocation should run");

    assert!(
        sta_output.status.success(),
        "gv-sta failed: status={:?}\nstdout={}\nstderr={}",
        sta_output.status,
        String::from_utf8_lossy(&sta_output.stdout),
        String::from_utf8_lossy(&sta_output.stderr)
    );

    let stdout = String::from_utf8_lossy(&sta_output.stdout);
    assert!(stdout.contains("worst_output_arrival:"));

    let worst_line = stdout
        .lines()
        .find(|line| line.starts_with("worst_output_arrival:"))
        .expect("expected worst_output_arrival line");
    let value: f64 = worst_line
        .split(':')
        .nth(1)
        .expect("line should have ':'")
        .trim()
        .parse()
        .expect("worst_output_arrival should parse as f64");
    assert!(value > 0.0, "expected positive arrival, got {}", value);
}

#[test_case(&[]; "default_nand2")]
#[test_case(&["--strategy", "area"]; "area_with_fanout_optimization")]
#[test_case(&["--strategy", "delay", "--primary-input-transition", "0.01", "--module-output-load", "0.01"]; "delay_with_area_recovery")]
fn aig_tech_map_protobuf_matches_gv_for_file_and_stdout(
    strategy_args: &[&str],
) -> anyhow::Result<()> {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let dir = tempfile::tempdir()?;
    let aig = dir.path().join("input.aag");
    let liberty = dir.path().join("cells.textproto");
    let pb = dir.path().join("mapped.pb");
    let report = dir.path().join("mapping.json");
    std::fs::write(&aig, make_two_input_and_aig())?;
    std::fs::write(&liberty, make_timing_enabled_inv_nand2_liberty_textproto())?;

    let map = || {
        let mut cmd = Command::new(driver);
        cmd.arg("aig-tech-map")
            .arg(&aig)
            .args(strategy_args)
            .arg("--liberty_proto")
            .arg(&liberty)
            .args(["--module_name", "mapped_top"])
            .env_remove("RUST_LOG");
        cmd
    };
    let gv = map().output()?;
    let binary = map().args(["-o", "-", "--netlist-format", "pb"]).output()?;
    let file = map()
        .arg("--output")
        .arg(&pb)
        .arg("--json_out")
        .arg(&report)
        .output()?;
    for result in [&gv, &binary, &file] {
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    assert!(file.stdout.is_empty());
    assert_eq!(gv.stderr, binary.stderr);
    assert_eq!(std::fs::read(&pb)?, binary.stdout);
    let summary: serde_json::Value = serde_json::from_slice(&std::fs::read(report)?)?;
    assert_eq!(summary["module"], "mapped_top");

    let restored = Command::new(driver).arg("pb2gv").arg(&pb).output()?;
    assert!(
        restored.status.success(),
        "{}",
        String::from_utf8_lossy(&restored.stderr)
    );
    assert_eq!(gv.stdout, restored.stdout);

    // Format selection is shared across strategies, so exercise the remaining
    // destination and override cases once with the default mapper.
    if strategy_args.is_empty() {
        for (filename, flags, expected) in [
            (OsStr::new("mapped.PrOtO"), &[][..], &binary.stdout),
            // The default macOS filesystem only accepts UTF-8 filenames.
            #[cfg(target_os = "linux")]
            (
                OsStr::from_bytes(b"mapped-\xff.pb"),
                &[][..],
                &binary.stdout,
            ),
            (
                OsStr::new("verilog.pb"),
                &["--netlist-format", "gv"][..],
                &gv.stdout,
            ),
            (OsStr::new("mapped.netlist"), &[][..], &gv.stdout),
            (OsStr::new("mapped"), &[][..], &gv.stdout),
        ] {
            let path = dir.path().join(filename);
            let result = map().arg("-o").arg(&path).args(flags).output()?;
            assert!(
                result.status.success(),
                "{filename:?}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            assert!(result.stdout.is_empty(), "{filename:?}");
            assert_eq!(&std::fs::read(path)?, expected, "{filename:?}");
        }
    }
    Ok(())
}

#[test]
fn aig_tech_map_rejects_protobuf_output_aliasing_stderr() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let pb = dir.path().join("mapped.pb");
    let json = dir.path().join("report.json");
    let map = || {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
        cmd.args([
            "aig-tech-map",
            "unused.aig",
            "--liberty_proto",
            "unused.proto",
        ])
        .env_remove("RUST_LOG");
        cmd
    };
    let error = b"aig-tech-map error: protobuf netlist output must not refer to stderr\n";

    // Append diagnostics to an existing file to check that the mapper never
    // truncates it or writes the sidecar after detecting the alias.
    let existing = b"existing contents\n";
    std::fs::write(&pb, existing)?;
    let stderr = std::fs::OpenOptions::new().append(true).open(&pb)?;
    let result = map()
        .arg("-o")
        .arg(&pb)
        .arg("--json_out")
        .arg(&json)
        .stderr(stderr)
        .output()?;
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert_eq!(
        std::fs::read(&pb)?,
        [existing.as_slice(), error.as_slice()].concat()
    );
    assert!(!json.exists());

    let merged = std::fs::File::create(&pb)?;
    let result = map()
        .args(["--netlist-format", "pb"])
        .stdout(merged.try_clone()?)
        .stderr(merged)
        .output()?;
    assert!(!result.status.success());
    assert_eq!(std::fs::read(&pb)?, error);

    #[cfg(unix)]
    {
        let result = map()
            .args(["-o", "/dev/stderr", "--netlist-format", "pb"])
            .output()?;
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
        assert_eq!(result.stderr, error);
    }
    Ok(())
}

#[test]
fn nand2_mapping_does_not_require_timing_but_sta_does() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let aig_path = temp_dir.path().join("in.aag");
    let source_path = temp_dir.path().join("cells.lib");
    let untimed_path = temp_dir.path().join("untimed.proto");
    let timed_path = temp_dir.path().join("timed.proto");
    let netlist_path = temp_dir.path().join("mapped.gv");
    let report_path = temp_dir.path().join("mapping.json");

    let mut builder = GateBuilder::new("and2_demo".into(), GateBuilderOptions::no_opt());
    let a = *builder.add_input("a".into(), 1).get_lsb(0);
    let b = *builder.add_input("b".into(), 1).get_lsb(0);
    let y = builder.add_and_binary(a, b);
    builder.add_output("y".into(), y.into());
    let aig = builder.build();
    std::fs::write(&aig_path, emit_aiger(&aig, true).unwrap()).unwrap();
    // Only the canonical INV has timing data, and it is incomplete. The
    // higher-drive candidates have none. Mapping should ignore both facts.
    std::fs::write(
        &source_path,
        r#"
library (test) {
  time_unit : "1ns";
  lu_table_template (tmpl) { variable_1 : input_net_transition; index_1 ("0.1"); }
  cell (INV) {
    pin (A) { direction : input; }
    pin (Y) { direction : output; function : "!A";
      timing () { related_pin : "A"; timing_type : combinational;
        timing_sense : negative_unate; cell_rise (tmpl) { values ("0.4"); } }
    }
  }
  cell (INVx2) {
    pin (A) { direction : input; }
    pin (Y) { direction : output; function : "!A"; }
  }
  cell (INVx8) {
    use_for_size_only : true;
    pin (A) { direction : input; }
    pin (Y) { direction : output; function : "!A"; }
  }
  cell (INVx9) {
    dont_use : true;
    pin (A) { direction : input; }
    pin (Y) { direction : output; function : "!A"; }
  }
  cell (NAND2) {
    pin (A) { direction : input; } pin (B) { direction : input; }
    pin (Y) { direction : output; function : "!(A*B)"; }
  }
  cell (NAND2x2) {
    pin (A) { direction : input; } pin (B) { direction : input; }
    pin (Y) { direction : output; function : "!(A*B)"; }
  }
  cell (NAND2x8) {
    use_for_size_only : true;
    pin (A) { direction : input; } pin (B) { direction : input; }
    pin (Y) { direction : output; function : "!(A*B)"; }
  }
  cell (NAND2x9) {
    dont_use : true;
    pin (A) { direction : input; } pin (B) { direction : input; }
    pin (Y) { direction : output; function : "!(A*B)"; }
  }
}
"#,
    )
    .unwrap();

    for (path, include_timing) in [(&untimed_path, false), (&timed_path, true)] {
        let mut command = Command::new(driver);
        command.arg("lib2proto").arg("--output").arg(path);
        if include_timing {
            command.arg("--include-timing");
        }
        let output = command.arg(&source_path).output().unwrap();
        assert!(
            output.status.success(),
            "lib2proto failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let lib = load_liberty_from_path(path).unwrap();
        let has_timing = lib
            .cells
            .iter()
            .flat_map(|cell| &cell.pins)
            .any(|pin| !pin.timing_arcs.is_empty());
        assert_eq!(has_timing, include_timing);
    }

    for (policy, inv, nand2) in [
        ("small-normal-vt", "INV", "NAND2"),
        ("max-speed", "INVx2", "NAND2x2"),
    ] {
        let mut mapped = Vec::new();
        for lib_path in [&untimed_path, &timed_path] {
            let output = Command::new(driver)
                .arg("aig-tech-map")
                .arg(&aig_path)
                .arg("--liberty_proto")
                .arg(lib_path)
                .arg("--netlist_out")
                .arg(&netlist_path)
                .args(["--strategy", "nand2", "--cell_policy", policy])
                .arg("--json_out")
                .arg(&report_path)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "aig-tech-map failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let text = std::fs::read_to_string(&netlist_path).unwrap();
            let report: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
            assert_eq!(report["inv_cell"], inv);
            assert_eq!(report["nand2_cell"], nand2);
            assert_mapping_equivalent(&aig, &text, &aig_path, lib_path);
            mapped.push((text, report));
        }
        assert_eq!(mapped[0], mapped[1]);
    }

    // STA must diagnose missing timing both when the entire proto is untimed
    // and when the proto has timing, but not for the selected max-speed cells.
    for (lib_path, expected_error) in [
        (&untimed_path, "has no timing payloads"),
        (&timed_path, "has no combinational timing arcs"),
    ] {
        let sta_output = Command::new(driver)
            .arg("gv-sta")
            .arg("--netlist")
            .arg(&netlist_path)
            .arg("--liberty_proto")
            .arg(lib_path)
            .output()
            .expect("gv-sta invocation should run");
        assert!(!sta_output.status.success());
        assert!(
            String::from_utf8_lossy(&sta_output.stderr).contains(expected_error),
            "unexpected gv-sta error: {}",
            String::from_utf8_lossy(&sta_output.stderr)
        );
    }
}

#[test]
fn nand2_mapping_accepts_missing_input_arc_but_sta_reports_it() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let aig_path = temp_dir.path().join("in.aag");
    let liberty_path = temp_dir.path().join("lib.textproto");
    let netlist_path = temp_dir.path().join("mapped.gv");
    let nand_b_arc = r#"
    timing_arcs: {
      related_pin_string_id: 4
      timing_sense: TIMING_SENSE_NEGATIVE_UNATE
      timing_type: TIMING_TYPE_COMBINATIONAL
      tables: { kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: 2.0 }
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 2.1 }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 0.5 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: 0.6 }
    }"#;
    let timed_lib = make_timing_enabled_inv_nand2_liberty_textproto();
    assert_eq!(timed_lib.matches(nand_b_arc).count(), 1);
    std::fs::write(&liberty_path, timed_lib.replace(nand_b_arc, ""))
        .expect("write partially characterized library");
    std::fs::write(&aig_path, make_two_input_and_aig()).expect("write aiger");

    let map_output = Command::new(driver)
        .arg("aig-tech-map")
        .arg(&aig_path)
        .arg("--liberty_proto")
        .arg(&liberty_path)
        .arg("--netlist_out")
        .arg(&netlist_path)
        .output()
        .expect("aig-tech-map invocation should run");
    assert!(
        map_output.status.success(),
        "aig-tech-map failed: {}",
        String::from_utf8_lossy(&map_output.stderr)
    );

    let sta_output = Command::new(driver)
        .arg("gv-sta")
        .arg("--netlist")
        .arg(&netlist_path)
        .arg("--liberty_proto")
        .arg(&liberty_path)
        .output()
        .expect("gv-sta invocation should run");
    assert!(!sta_output.status.success());
    let stderr = String::from_utf8_lossy(&sta_output.stderr);
    assert!(
        stderr.contains("NAND2") && stderr.contains("B"),
        "expected an error identifying the missing NAND2 B arc, got: {}",
        stderr
    );
}

#[test]
fn area_mapping_and_sta_check_three_input_cell_coverage() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let dir = tempfile::tempdir().expect("create temp dir");
    let aig_path = dir.path().join("and3.aag");
    let mut builder = GateBuilder::new("and3_demo".into(), GateBuilderOptions::no_opt());
    let a = *builder.add_input("a".into(), 1).get_lsb(0);
    let b = *builder.add_input("b".into(), 1).get_lsb(0);
    let c = *builder.add_input("c".into(), 1).get_lsb(0);
    let ab = builder.add_and_binary(a, b);
    let abc = builder.add_and_binary(ab, c);
    builder.add_output("y".into(), abc.into());
    std::fs::write(&aig_path, emit_aiger(&builder.build(), true).unwrap()).unwrap();

    let mut mapped_netlists = Vec::new();
    for missing_c_arc in [false, true] {
        let arcs = ["A", "B", "C"]
            .into_iter()
            .filter(|pin| !missing_c_arc || *pin != "C")
            .map(|pin| {
                format!(
                    r#"
      timing () {{
        related_pin : "{pin}"; timing_type : combinational; timing_sense : positive_unate;
        cell_rise (tmpl) {{ values ("1.0"); }}
        cell_fall (tmpl) {{ values ("1.1"); }}
        rise_transition (tmpl) {{ values ("0.3"); }}
        fall_transition (tmpl) {{ values ("0.4"); }}
      }}"#
                )
            })
            .collect::<String>();
        // AND3 has the same Boolean function and direct pin arcs as the ASAP7
        // three-input AND; omit C's arc only for the negative case.
        let liberty = format!(
            r#"
library (test) {{
  time_unit : "1ns";
  lu_table_template (tmpl) {{ variable_1 : input_net_transition; index_1 ("0.1"); }}
  cell (INV) {{ area : 1.0;
    pin (A) {{ direction : input; }}
    pin (Y) {{ direction : output; function : "!A"; }}
  }}
  cell (AND3) {{ area : 1.0;
    pin (A) {{ direction : input; capacitance : 0.01; }}
    pin (B) {{ direction : input; capacitance : 0.01; }}
    pin (C) {{ direction : input; capacitance : 0.01; }}
    pin (Y) {{ direction : output; function : "(A * B * C)"; {arcs}
    }}
  }}
}}
"#
        );
        let source_path = dir.path().join("cells.lib");
        let liberty_path = dir.path().join("cells.proto");
        let netlist_path = dir.path().join("mapped.gv");
        let report_path = dir.path().join("mapping.json");
        std::fs::write(&source_path, liberty).unwrap();
        let import = Command::new(driver)
            .arg("lib2proto")
            .arg("--include-timing")
            .arg("--output")
            .arg(&liberty_path)
            .arg(&source_path)
            .output()
            .unwrap();
        assert!(
            import.status.success(),
            "lib2proto failed: {}",
            String::from_utf8_lossy(&import.stderr)
        );
        let mapped = Command::new(driver)
            .arg("aig-tech-map")
            .arg(&aig_path)
            .args([
                "--strategy",
                "area",
                "--fanout-opt=false",
                "--liberty_proto",
            ])
            .arg(&liberty_path)
            .arg("--netlist_out")
            .arg(&netlist_path)
            .arg("--json_out")
            .arg(&report_path)
            .output()
            .unwrap();
        assert!(
            mapped.status.success(),
            "aig-tech-map failed: {}",
            String::from_utf8_lossy(&mapped.stderr)
        );
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
        assert_eq!(
            report["stats"]["cells_by_type"],
            serde_json::json!({"AND3": 1})
        );
        mapped_netlists.push(std::fs::read_to_string(&netlist_path).unwrap());

        let sta = Command::new(driver)
            .arg("gv-sta")
            .arg("--netlist")
            .arg(&netlist_path)
            .arg("--liberty_proto")
            .arg(&liberty_path)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&sta.stderr);
        if missing_c_arc {
            assert!(!sta.status.success(), "missing C arc must fail STA");
            assert!(
                stderr.contains("functional input 'C'") && stderr.contains("AND3"),
                "unexpected gv-sta error: {stderr}"
            );
        } else {
            assert!(sta.status.success(), "gv-sta failed: {stderr}");
        }
    }
    assert_eq!(mapped_netlists[0], mapped_netlists[1]);
}

/// Checks emitted cells by exhaustive simulation and, when enabled, explicit
/// Bitwuzla CLI proof.
fn assert_mapping_equivalent(aig: &GateFn, text: &str, aig_path: &Path, liberty_path: &Path) {
    let lib = load_liberty_from_path(liberty_path).expect("load mapping library");
    let parsed = read_gv_from_str(text).expect("parse CLI netlist output");
    let projected = project_gatefn_from_netlist_and_liberty(
        &parsed.modules[0],
        &parsed.nets,
        &parsed.interner,
        &lib,
        &HashSet::new(),
        &HashSet::new(),
    )
    .expect("project CLI netlist");
    for assignment in 0..1usize << aig.inputs.len() {
        let inputs = (0..aig.inputs.len())
            .map(|i| IrBits::make_ubits(1, ((assignment >> i) & 1) as u64).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            eval(aig, &inputs, Collect::None).outputs,
            eval(&projected, &inputs, Collect::None).outputs
        );
    }
    #[cfg(feature = "has-bitwuzla")]
    {
        let dir = tempfile::tempdir().expect("create proof directory");
        let netlist_path = dir.path().join("mapped.v");
        let mapped_aig = dir.path().join("mapped.aig");
        std::fs::write(&netlist_path, text).expect("write proof input");
        let convert = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .arg("gv2aig")
            .arg("--netlist")
            .arg(&netlist_path)
            .arg("--liberty_proto")
            .arg(liberty_path)
            .arg("--aiger-out")
            .arg(&mapped_aig)
            .output()
            .unwrap();
        assert!(
            convert.status.success(),
            "{}",
            String::from_utf8_lossy(&convert.stderr)
        );
        let proof = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .arg("aig-equiv")
            .arg(aig_path)
            .arg(&mapped_aig)
            .arg("--solver=bitwuzla")
            .output()
            .unwrap();
        assert!(
            proof.status.success(),
            "stdout={} stderr={}",
            String::from_utf8_lossy(&proof.stdout),
            String::from_utf8_lossy(&proof.stderr)
        );
    }
    #[cfg(not(feature = "has-bitwuzla"))]
    let _ = aig_path;
}

#[test]
fn strategies_preserve_default_and_cli_semantics() {
    let dir = tempfile::tempdir().unwrap();
    let aig_path = dir.path().join("input.aag");
    let lib_path = dir.path().join("cells.textproto");
    let mut builder = GateBuilder::new("strategy_demo".into(), GateBuilderOptions::no_opt());
    let a = *builder.add_input("a".into(), 1).get_lsb(0);
    let b = *builder.add_input("b".into(), 1).get_lsb(0);
    let and = builder.add_and_binary(a, b);
    builder.add_output("y".into(), and.negate().into());
    let aig = builder.build();
    std::fs::write(&aig_path, emit_aiger(&aig, true).unwrap()).unwrap();
    std::fs::write(&lib_path, make_timing_enabled_inv_nand2_liberty_textproto()).unwrap();
    let mut nand2_outputs = Vec::new();
    for strategy in [None, Some("nand2"), Some("area")] {
        let netlist_path = dir
            .path()
            .join(format!("{}.v", strategy.unwrap_or("default")));
        let report_path = netlist_path.with_extension("json");
        let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
        command
            .arg("aig-tech-map")
            .arg(&aig_path)
            .arg("--liberty_proto")
            .arg(&lib_path)
            .arg("--netlist_out")
            .arg(&netlist_path)
            .args(["--module_name", "mapped_top"])
            .arg("--json_out")
            .arg(&report_path);
        if let Some(strategy) = strategy {
            command.args(["--strategy", strategy]);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty());
        let text = std::fs::read_to_string(&netlist_path).unwrap();
        let parsed = read_gv_from_str(&text).unwrap();
        assert_eq!(
            parsed.interner.resolve(parsed.modules[0].name),
            Some("mapped_top")
        );
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(report_path).unwrap()).unwrap();
        assert_eq!(report["strategy"], strategy.unwrap_or("nand2"));
        assert!(report.get("mapper").is_none());
        assert_eq!(report["module"], "mapped_top");
        let expected_count = if strategy == Some("area") { 1 } else { 3 };
        assert_eq!(report["cell_count"], expected_count);
        assert_eq!(parsed.modules[0].instances.len(), expected_count as usize);
        assert!(report.get("library_s").is_none());
        assert!(report.get("mapping_s").is_none());
        if strategy == Some("area") {
            assert_eq!(report["stats"]["cell_area"], 2.0);
            assert_eq!(
                report["stats"]["cells_by_type"],
                serde_json::json!({"NAND2": 1})
            );
            assert_eq!(report["stats"]["inverter_count"], 0);
            assert_eq!(report["stats"]["duplicated_ands"], 0);
        } else {
            assert!(report.get("fanout_optimization").is_none());
            nand2_outputs.push(text.clone());
        }
        assert_mapping_equivalent(&aig, &text, &aig_path, &lib_path);
    }
    assert_eq!(nand2_outputs[0], nand2_outputs[1]);
}

/// Checks default optimization, explicit overrides, budget, and equivalence.
#[test]
fn area_fanout_cli_preserves_equivalence_budget_and_determinism() {
    let dir = tempfile::tempdir().unwrap();
    let aig_path = dir.path().join("input.aag");
    let lib_path = dir.path().join("cells.textproto");
    let netlist_path = dir.path().join("mapped.v");
    let report_path = dir.path().join("mapping.json");
    let mut builder = GateBuilder::new("and2_demo".into(), GateBuilderOptions::no_opt());
    let a = *builder.add_input("a".into(), 1).get_lsb(0);
    let b = *builder.add_input("b".into(), 1).get_lsb(0);
    let and = builder.add_and_binary(a, b);
    builder.add_output("y".into(), and.into());
    let aig = builder.build();
    std::fs::write(&aig_path, emit_aiger(&aig, true).unwrap()).unwrap();

    // Cheap NAND2 violates its drive limit; a slightly larger equivalent cell
    // improves delay within the default 5% total-area budget.
    let base = make_timing_enabled_inv_nand2_liberty_textproto();
    let start = base.find("cells: {\n  name: \"NAND2\"").unwrap();
    let end = base.find("default_output_load:").unwrap();
    let weak = base[start..end].replace(
        "function_string_id: 5",
        "function_string_id: 5\n    max_capacitance: 0.001",
    );
    let fast = weak
        .replace("name: \"NAND2\"", "name: \"NAND_FAST\"")
        .replace("area: 2.0", "area: 2.1")
        .replace("values: 2.0", "values: 0.2")
        .replace("values: 2.1", "values: 0.3");
    std::fs::write(
        &lib_path,
        format!("{}{}{}{}", &base[..start], weak, fast, &base[end..]),
    )
    .unwrap();

    let mut outputs = Vec::new();
    let mut reports = Vec::new();
    for mode in [
        "disabled",
        "zero-budget",
        "default",
        "bare-enable",
        "explicit-enable",
    ] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
        command
            .arg("aig-tech-map")
            .arg(&aig_path)
            .args(["--strategy", "area", "--liberty_proto"])
            .arg(&lib_path)
            .arg("--netlist_out")
            .arg(&netlist_path)
            .arg("--json_out")
            .arg(&report_path);
        match mode {
            "disabled" => {
                command.arg("--fanout-opt=false");
            }
            "bare-enable" => {
                command.arg("--fanout-opt");
            }
            "explicit-enable" => {
                command.arg("--fanout-opt=true");
            }
            _ => {} // Exercise implicit area-mode enablement.
        }
        if mode == "zero-budget" {
            command.args(["--fanout-max-area-growth", "0"]);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{mode}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = std::fs::read_to_string(&netlist_path).unwrap();
        let report_text = std::fs::read_to_string(&report_path).unwrap();
        assert!(!report_text.contains("_seconds"));
        let report: serde_json::Value = serde_json::from_str(&report_text).unwrap();
        let parsed = read_gv_from_str(&text).unwrap();
        assert_eq!(report["cell_count"], parsed.modules[0].instances.len());
        assert_eq!(report["net_count"], parsed.nets.len());
        assert_eq!(report["stats"]["cell_area"], 3.0);
        assert_mapping_equivalent(&aig, &text, &aig_path, &lib_path);
        outputs.push(text);
        reports.push(report);
    }
    assert_eq!(outputs[0], outputs[1]);
    assert!(reports[0].get("fanout_optimization").is_none());
    let blocked = &reports[1]["fanout_optimization"];
    assert_eq!(blocked["before"], blocked["after"]);
    assert!(blocked["decisions"].as_array().unwrap().is_empty());
    assert!(
        blocked["trials"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["reason"] == "area budget")
    );
    let optimized = &reports[2]["fanout_optimization"];
    assert!(!optimized["decisions"].as_array().unwrap().is_empty());
    assert!(
        optimized["after"]["worst_delay"].as_f64().unwrap()
            < optimized["before"]["worst_delay"].as_f64().unwrap()
    );
    assert!(optimized["after"]["area"].as_f64().unwrap() <= 3.0 * 1.05);
    assert!(outputs[2].contains("NAND_FAST"));
    assert_eq!(outputs[2], outputs[3]);
    assert_eq!(reports[2], reports[3]);
    assert_eq!(outputs[2], outputs[4]);
    assert_eq!(reports[2], reports[4]);
}

#[test]
fn area_fanout_cli_rejects_options_without_area_optimization() {
    for (extra, expected) in [
        (
            vec!["--fanout-opt"],
            "--fanout-opt requires --strategy area",
        ),
        (
            vec!["--fanout-max-area-growth", "0.1"],
            "fanout tuning options",
        ),
        (
            vec![
                "--strategy",
                "area",
                "--fanout-opt=false",
                "--module_output_load",
                "0",
            ],
            "fanout tuning options",
        ),
        (
            vec!["--fanout-opt=true"],
            "--fanout-opt requires --strategy area",
        ),
    ] {
        // These errors must be detected before attempting to open either input.
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .args([
                "aig-tech-map",
                "missing.aag",
                "--liberty_proto",
                "missing.proto",
                "--netlist_out",
                "-",
            ])
            .args(extra)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(expected), "{stderr}");
        assert!(!stderr.contains("failed to load"), "{stderr}");
    }
}

#[test]
fn mapping_reports_are_deterministic_for_both_strategies() {
    let dir = tempfile::tempdir().unwrap();
    let aig_path = dir.path().join("input.aag");
    let lib_path = dir.path().join("cells.textproto");
    let netlist_path = dir.path().join("mapped.v");
    let report_path = dir.path().join("mapping.json");
    std::fs::write(&aig_path, make_two_input_and_aig()).unwrap();
    std::fs::write(&lib_path, make_timing_enabled_inv_nand2_liberty_textproto()).unwrap();

    for strategy in ["nand2", "area"] {
        let mut reports = Vec::new();
        for _ in 0..3 {
            let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
                .arg("aig-tech-map")
                .arg(&aig_path)
                .args(["--strategy", strategy])
                .arg("--liberty_proto")
                .arg(&lib_path)
                .arg("--netlist_out")
                .arg(&netlist_path)
                .arg("--json_out")
                .arg(&report_path)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "strategy={strategy}, stderr={}",
                String::from_utf8_lossy(&output.stderr)
            );
            reports.push(std::fs::read(&report_path).unwrap());
        }
        assert_eq!(reports[0], reports[1], "strategy={strategy}");
        assert_eq!(reports[0], reports[2], "strategy={strategy}");
    }
}

#[test]
fn area_cli_supports_functional_library_stdout_sharing_and_constants() {
    let dir = tempfile::tempdir().unwrap();
    let aig_path = dir.path().join("input.aag");
    let lib_path = dir.path().join("functional.textproto");
    let report_path = dir.path().join("mapping.json");
    std::fs::write(
        &lib_path,
        r#"
format_magic: 5496997758228255311
cells: {
  name: "INV" area: 1.0
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: OUTPUT function_string_id: 3 }
}
cells: {
  name: "NAND2" area: 2.0
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 4 direction: INPUT }
  pins: { name_string_id: 2 direction: OUTPUT function_string_id: 5 }
}
interned_strings: ["A", "Y", "!A", "B", "!(A*B)"]
"#,
    )
    .unwrap();
    let mut builder = GateBuilder::new("sharing_demo".into(), GateBuilderOptions::no_opt());
    let a = *builder.add_input("a".into(), 1).get_lsb(0);
    let b = *builder.add_input("b".into(), 1).get_lsb(0);
    let and = builder.add_and_binary(a, b);
    for (index, operand) in [
        and,
        and.negate(),
        and,
        a,
        builder.get_false(),
        builder.get_true(),
    ]
    .into_iter()
    .enumerate()
    {
        builder.add_output(format!("o{index}"), operand.into());
    }
    let aig = builder.build();
    std::fs::write(&aig_path, emit_aiger(&aig, true).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("aig-tech-map")
        .arg(&aig_path)
        .args(["--strategy", "area", "--fanout-opt=false"])
        .arg("--liberty_proto")
        .arg(&lib_path)
        .args(["--netlist_out", "-"])
        .arg("--json_out")
        .arg(&report_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(report_path).unwrap()).unwrap();
    assert_eq!(report["stats"]["cell_area"], 3.0);
    assert_eq!(report["stats"]["cell_count"], 2);
    assert_eq!(report["stats"]["inverter_count"], 1);
    assert_eq!(report["stats"]["shared_opposite_polarities"], 1);
    assert_eq!(report["stats"]["duplicated_ands"], 0);
    assert_eq!(report["stats"]["covered_ands"], 1);
    assert_mapping_equivalent(&aig, &text, &aig_path, &lib_path);
}

#[test]
fn area_cli_rejects_explicit_nand2_policy() {
    for policy in ["small-normal-vt", "max-speed"] {
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .args([
                "aig-tech-map",
                "unused.aig",
                "--strategy",
                "area",
                "--cell_policy",
                policy,
                "--liberty_proto",
                "unused.proto",
                "--netlist_out",
                "-",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            "aig-tech-map error: --cell_policy applies only to --strategy nand2\n"
        );
    }
}

#[test]
fn area_cli_matches_six_input_cells() {
    let dir = tempfile::tempdir().unwrap();
    let aig_path = dir.path().join("input.aag");
    let lib_path = dir.path().join("functional.textproto");
    let report_path = dir.path().join("mapping.json");
    std::fs::write(
        &lib_path,
        r#"
format_magic: 5496997758228255311
cells: {
  name: "INV" area: 1.0
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: OUTPUT function_string_id: 3 }
}
cells: {
  name: "AND2" area: 1.0
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 4 direction: INPUT }
  pins: { name_string_id: 2 direction: OUTPUT function_string_id: 5 }
}
cells: {
  name: "AND6" area: 1.0
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 4 direction: INPUT }
  pins: { name_string_id: 6 direction: INPUT }
  pins: { name_string_id: 7 direction: INPUT }
  pins: { name_string_id: 8 direction: INPUT }
  pins: { name_string_id: 9 direction: INPUT }
  pins: { name_string_id: 2 direction: OUTPUT function_string_id: 10 }
}
interned_strings: ["A", "Y", "!A", "B", "A*B", "C", "D", "E", "F", "A*B*C*D*E*F"]
"#,
    )
    .unwrap();
    let mut builder = GateBuilder::new("and6_demo".into(), GateBuilderOptions::no_opt());
    let inputs = (0..6)
        .map(|i| *builder.add_input(format!("i{i}"), 1).get_lsb(0))
        .collect::<Vec<_>>();
    let mut output = inputs[0];
    for input in &inputs[1..] {
        output = builder.add_and_binary(output, *input);
    }
    builder.add_output("y".into(), output.into());
    let aig = builder.build();
    std::fs::write(&aig_path, emit_aiger(&aig, true).unwrap()).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("aig-tech-map")
        .arg(&aig_path)
        .args(["--strategy", "area", "--fanout-opt=false"])
        .arg("--liberty_proto")
        .arg(&lib_path)
        .args(["--netlist_out", "-"])
        .arg("--json_out")
        .arg(&report_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(
        report["stats"]["cells_by_type"],
        serde_json::json!({"AND6": 1})
    );
    assert_eq!(report["stats"]["cell_area"], 1.0);
    assert_mapping_equivalent(&aig, &text, &aig_path, &lib_path);
}

#[test]
fn cli_rejects_aliased_outputs_without_clobbering() {
    let dir = tempfile::tempdir().unwrap();
    let output_path = dir.path().join("output.v");
    std::fs::write(&output_path, "keep this netlist\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(dir.path())
        .args([
            "aig-tech-map",
            "unused.aig",
            "--strategy",
            "area",
            "--liberty_proto",
            "unused.proto",
            "--netlist_out",
            "output.v",
            "--json_out",
        ])
        .arg(&output_path)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        "aig-tech-map error: --json_out and --output must name different files\n"
    );
    assert_eq!(
        std::fs::read_to_string(&output_path).unwrap(),
        "keep this netlist\n"
    );
}

#[test]
fn cli_rejects_json_aliasing_redirected_stdout() {
    let dir = tempfile::tempdir().unwrap();
    let aig_path = dir.path().join("input.aag");
    let lib_path = dir.path().join("cells.textproto");
    let output_path = dir.path().join("output");
    std::fs::write(&aig_path, make_two_input_and_aig()).unwrap();
    std::fs::write(&lib_path, make_timing_enabled_inv_nand2_liberty_textproto()).unwrap();
    std::fs::write(&output_path, "keep existing contents\n").unwrap();
    let stdout = std::fs::OpenOptions::new()
        .write(true)
        .open(&output_path)
        .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("aig-tech-map")
        .arg(&aig_path)
        .args(["--strategy", "area", "--liberty_proto"])
        .arg(&lib_path)
        .args(["--netlist_out", "-", "--json_out"])
        .arg(&output_path)
        .stdout(stdout)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        "aig-tech-map error: --json_out must not refer to stdout when the netlist is written to stdout\n"
    );
    assert_eq!(
        std::fs::read_to_string(&output_path).unwrap(),
        "keep existing contents\n"
    );
}

#[cfg(unix)]
#[test]
fn cli_rejects_json_aliasing_stdout_pipe() {
    let dir = tempfile::tempdir().unwrap();
    let aig_path = dir.path().join("input.aag");
    let lib_path = dir.path().join("cells.textproto");
    let stdout_alias = dir.path().join("stdout-alias");
    std::fs::write(&aig_path, make_two_input_and_aig()).unwrap();
    std::fs::write(&lib_path, make_timing_enabled_inv_nand2_liberty_textproto()).unwrap();
    std::os::unix::fs::symlink("/dev/stdout", &stdout_alias).unwrap();
    for json_path in [
        Path::new("/dev/stdout"),
        Path::new("/dev/fd/1"),
        &stdout_alias,
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .arg("aig-tech-map")
            .arg(&aig_path)
            .args(["--strategy", "area", "--liberty_proto"])
            .arg(&lib_path)
            .args(["--netlist_out", "-", "--json_out"])
            .arg(json_path)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            "aig-tech-map error: --json_out must not refer to stdout when the netlist is written to stdout\n"
        );
    }
}

#[cfg(unix)]
#[test]
fn cli_rejects_stdout_device_alias_with_redirected_file() {
    let dir = tempfile::tempdir().unwrap();
    let aig_path = dir.path().join("input.aag");
    let lib_path = dir.path().join("cells.textproto");
    let output_path = dir.path().join("output");
    std::fs::write(&aig_path, make_two_input_and_aig()).unwrap();
    std::fs::write(&lib_path, make_timing_enabled_inv_nand2_liberty_textproto()).unwrap();
    for json_path in ["/dev/stdout", "/dev/fd/1"] {
        std::fs::write(&output_path, "keep existing contents\n").unwrap();
        let stdout = std::fs::OpenOptions::new()
            .write(true)
            .open(&output_path)
            .unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .arg("aig-tech-map")
            .arg(&aig_path)
            .args(["--strategy", "area", "--liberty_proto"])
            .arg(&lib_path)
            .args(["--netlist_out", "-", "--json_out", json_path])
            .stdout(stdout)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            "aig-tech-map error: --json_out must not refer to stdout when the netlist is written to stdout\n"
        );
        assert_eq!(
            std::fs::read_to_string(&output_path).unwrap(),
            "keep existing contents\n"
        );
    }
}

#[cfg(unix)]
#[test]
fn cli_rejects_json_alias_to_stderr() {
    let dir = tempfile::tempdir().unwrap();
    let aig_path = dir.path().join("input.aag");
    let lib_path = dir.path().join("cells.textproto");
    let netlist_path = dir.path().join("netlist.v");
    let report_path = dir.path().join("report.json");
    std::fs::write(&aig_path, make_two_input_and_aig()).unwrap();
    std::fs::write(&lib_path, make_timing_enabled_inv_nand2_liberty_textproto()).unwrap();

    for json_path in [Path::new("/dev/stderr"), Path::new("/dev/fd/2")] {
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .arg("aig-tech-map")
            .arg(&aig_path)
            .args(["--strategy", "area", "--liberty_proto"])
            .arg(&lib_path)
            .arg("--netlist_out")
            .arg(&netlist_path)
            .arg("--json_out")
            .arg(json_path)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            "aig-tech-map error: --json_out must not refer to stderr\n"
        );
        assert!(!netlist_path.exists());
    }

    // Exercise an inherited descriptor rather than only the conventional
    // stderr device names.
    let output = Command::new("sh")
        .args(["-c", "exec 3>&2; exec \"$@\"", "sh"])
        .arg(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("aig-tech-map")
        .arg(&aig_path)
        .args(["--strategy", "area", "--liberty_proto"])
        .arg(&lib_path)
        .arg("--netlist_out")
        .arg(&netlist_path)
        .args(["--json_out", "/dev/fd/3"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        "aig-tech-map error: --json_out must not refer to stderr\n"
    );
    assert!(!netlist_path.exists());

    let stderr = std::fs::File::create(&report_path).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("aig-tech-map")
        .arg(&aig_path)
        .args(["--strategy", "area", "--liberty_proto"])
        .arg(&lib_path)
        .arg("--netlist_out")
        .arg(&netlist_path)
        .arg("--json_out")
        .arg(&report_path)
        .stderr(Stdio::from(stderr))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(
        std::fs::read_to_string(&report_path).unwrap(),
        "aig-tech-map error: --json_out must not refer to stderr\n"
    );
    assert!(!netlist_path.exists());
}

#[cfg(unix)]
#[test]
fn cli_rejects_stdout_device_alias_with_socket() {
    let dir = tempfile::tempdir().unwrap();
    let aig_path = dir.path().join("input.aag");
    let lib_path = dir.path().join("cells.textproto");
    std::fs::write(&aig_path, make_two_input_and_aig()).unwrap();
    std::fs::write(&lib_path, make_timing_enabled_inv_nand2_liberty_textproto()).unwrap();
    for json_path in ["/dev/stdout", "/dev/fd/1"] {
        let (socket, mut peer) = UnixStream::pair().unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .arg("aig-tech-map")
            .arg(&aig_path)
            .args(["--strategy", "area", "--liberty_proto"])
            .arg(&lib_path)
            .args(["--netlist_out", "-", "--json_out", json_path])
            .stdout(Stdio::from(OwnedFd::from(socket)))
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            "aig-tech-map error: --json_out must not refer to stdout when the netlist is written to stdout\n"
        );
        peer.set_nonblocking(true).unwrap();
        let mut byte = [0];
        // Another process may briefly hold an inherited socket descriptor;
        // both EOF and WouldBlock establish that no output bytes were sent.
        let read = peer.read(&mut byte);
        assert!(
            matches!(&read, Ok(0))
                || matches!(&read, Err(error) if error.kind() == std::io::ErrorKind::WouldBlock),
            "unexpected socket read: {read:?}"
        );
    }
}

#[cfg(unix)]
#[test]
fn cli_rejects_json_fifo_with_stdout_stream_without_deadlock() {
    let dir = tempfile::tempdir().unwrap();
    let aig_path = dir.path().join("input.aag");
    let lib_path = dir.path().join("cells.textproto");
    let fifo = dir.path().join("report.fifo");
    let stdout_alias = dir.path().join("stdout-alias");
    std::fs::write(&aig_path, make_two_input_and_aig()).unwrap();
    std::fs::write(&lib_path, make_timing_enabled_inv_nand2_liberty_textproto()).unwrap();
    std::os::unix::fs::symlink("/dev/stdout", &stdout_alias).unwrap();
    assert!(
        Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success()
    );

    // No reader opens the report FIFO until netlist stdout reaches EOF.
    for netlist_path in [
        Path::new("-"),
        Path::new("/dev/stdout"),
        Path::new("/dev/fd/1"),
        &stdout_alias,
    ] {
        let (socket, peer) = UnixStream::pair().unwrap();
        for (stdout, peer) in [
            (Stdio::piped(), None),
            (Stdio::from(OwnedFd::from(socket)), Some(peer)),
        ] {
            let mut child = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
                .arg("aig-tech-map")
                .arg(&aig_path)
                .args(["--strategy", "area", "--liberty_proto"])
                .arg(&lib_path)
                .arg("--netlist_out")
                .arg(netlist_path)
                .arg("--json_out")
                .arg(&fifo)
                .stdout(stdout)
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(10);
            while child.try_wait().unwrap().is_none() {
                if Instant::now() >= deadline {
                    child.kill().unwrap();
                    child.wait().unwrap();
                    panic!(
                        "mapping blocked while opening the JSON FIFO for {}",
                        netlist_path.display()
                    );
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            let output = child.wait_with_output().unwrap();
            assert!(!output.status.success());
            assert!(output.stdout.is_empty());
            assert_eq!(
                String::from_utf8(output.stderr).unwrap(),
                "aig-tech-map error: --json_out cannot be a FIFO when stdout is a pipe or socket\n"
            );
            if let Some(mut peer) = peer {
                // Avoid blocking if a descriptor remains open, without relying
                // on socket read timeouts, which are
                // unsupported for AF_UNIX on macOS.
                peer.set_nonblocking(true).unwrap();
                let mut byte = [0; 1];
                let read = peer.read(&mut byte);
                assert!(
                    matches!(&read, Ok(0))
                        || matches!(&read, Err(error) if error.kind() == std::io::ErrorKind::WouldBlock),
                    "unexpected socket read: {read:?}"
                );
            }
        }
    }
}

#[test]
fn cli_handles_case_dependent_output_aliases() {
    let dir = tempfile::tempdir().unwrap();
    let aig_path = dir.path().join("input.aag");
    let lib_path = dir.path().join("cells.textproto");
    std::fs::write(&aig_path, make_two_input_and_aig()).unwrap();
    std::fs::write(&lib_path, make_timing_enabled_inv_nand2_liberty_textproto()).unwrap();

    // Probe the actual output directory: case sensitivity can vary by
    // filesystem.
    let probe = dir.path().join("CaseProbe");
    std::fs::write(&probe, "probe").unwrap();
    let case_insensitive = dir.path().join("caseprobe").exists();
    std::fs::remove_file(probe).unwrap();

    let netlist_path = dir.path().join("Mapped.v");
    let report_path = dir.path().join("mapped.v");
    for exists in [false, true] {
        if exists {
            std::fs::write(&netlist_path, "keep existing output\n").unwrap();
        }
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .arg("aig-tech-map")
            .arg(&aig_path)
            .args(["--strategy", "area"])
            .arg("--liberty_proto")
            .arg(&lib_path)
            .arg("--netlist_out")
            .arg(&netlist_path)
            .arg("--json_out")
            .arg(&report_path)
            .output()
            .unwrap();

        if case_insensitive {
            assert!(!output.status.success());
            assert_eq!(
                String::from_utf8(output.stderr).unwrap(),
                "aig-tech-map error: --json_out and --output must name different files\n"
            );
            if exists {
                assert_eq!(
                    std::fs::read_to_string(&netlist_path).unwrap(),
                    "keep existing output\n"
                );
                std::fs::remove_file(&netlist_path).unwrap();
            } else {
                assert!(!netlist_path.exists());
                assert!(!report_path.exists());
            }
        } else {
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let netlist = std::fs::read_to_string(&netlist_path).unwrap();
            read_gv_from_str(&netlist).unwrap();
            let report: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
            assert_eq!(report["strategy"], "area");
            std::fs::remove_file(&netlist_path).unwrap();
            std::fs::remove_file(&report_path).unwrap();
        }
    }
}

#[cfg(unix)]
#[test]
fn cli_writes_fifo_outputs_for_sequential_consumers() {
    let dir = tempfile::tempdir().unwrap();
    let aig_path = dir.path().join("input.aag");
    let lib_path = dir.path().join("cells.textproto");
    let netlist_fifo = dir.path().join("netlist.fifo");
    let report_fifo = dir.path().join("report.fifo");
    std::fs::write(&aig_path, make_two_input_and_aig()).unwrap();
    std::fs::write(&lib_path, make_timing_enabled_inv_nand2_liberty_textproto()).unwrap();
    assert!(
        Command::new("mkfifo")
            .args([&netlist_fifo, &report_fifo])
            .status()
            .unwrap()
            .success()
    );

    let reader_netlist = netlist_fifo.clone();
    let reader_report = report_fifo.clone();
    let reader = std::thread::spawn(move || {
        let netlist = std::fs::read_to_string(reader_netlist).unwrap();
        let report = std::fs::read_to_string(reader_report).unwrap();
        (netlist, report)
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("aig-tech-map")
        .arg(&aig_path)
        .args(["--strategy", "area"])
        .arg("--liberty_proto")
        .arg(&lib_path)
        .arg("--netlist_out")
        .arg(&netlist_fifo)
        .arg("--json_out")
        .arg(&report_fifo)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("mapping did not finish with a sequential FIFO reader");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let (netlist, report) = reader.join().unwrap();
    read_gv_from_str(&netlist).unwrap();
    let report: serde_json::Value = serde_json::from_str(&report).unwrap();
    assert_eq!(report["strategy"], "area");
}

#[cfg(unix)]
#[test]
fn cli_rejects_symlinked_output_aliases() {
    let dir = tempfile::tempdir().unwrap();
    let output_path = dir.path().join("output.v");
    let alias = dir.path().join("alias.json");
    std::os::unix::fs::symlink(&output_path, &alias).unwrap();
    for exists in [false, true] {
        if exists {
            std::fs::write(&output_path, "keep this netlist\n").unwrap();
        }
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .args([
                "aig-tech-map",
                "unused.aig",
                "--strategy",
                "area",
                "--liberty_proto",
                "unused.proto",
                "--netlist_out",
            ])
            .arg(&output_path)
            .arg("--json_out")
            .arg(&alias)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            "aig-tech-map error: --json_out and --output must name different files\n"
        );
        if exists {
            assert_eq!(
                std::fs::read_to_string(&output_path).unwrap(),
                "keep this netlist\n"
            );
        } else {
            assert!(!output_path.exists());
        }
    }
}

#[test]
fn cli_rejects_hard_linked_output_aliases() {
    let dir = tempfile::tempdir().unwrap();
    let output_path = dir.path().join("output.v");
    let alias = dir.path().join("alias.json");
    std::fs::write(&output_path, "keep this netlist\n").unwrap();
    std::fs::hard_link(&output_path, &alias).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .args([
            "aig-tech-map",
            "unused.aig",
            "--strategy",
            "area",
            "--liberty_proto",
            "unused.proto",
            "--netlist_out",
        ])
        .arg(&output_path)
        .arg("--json_out")
        .arg(&alias)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        "aig-tech-map error: --json_out and --output must name different files\n"
    );
    assert_eq!(
        std::fs::read_to_string(output_path).unwrap(),
        "keep this netlist\n"
    );
    assert_eq!(
        std::fs::read_to_string(alias).unwrap(),
        "keep this netlist\n"
    );
}

/// Default optimization must fail visibly when its required data is absent.
#[test]
fn area_fanout_requires_timing_and_load_unless_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let aig = dir.path().join("input.aag");
    let lib = dir.path().join("library.textproto");
    let netlist = dir.path().join("out.v");
    let report = dir.path().join("out.json");
    std::fs::write(&aig, make_two_input_and_aig()).unwrap();
    let timingless = r#"
format_magic: 5496997758228255311
cells: {
  name: "INV" area: 1.0
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: OUTPUT function_string_id: 3 }
}
cells: {
  name: "NAND2" area: 2.0
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 4 direction: INPUT }
  pins: { name_string_id: 2 direction: OUTPUT function_string_id: 5 }
}
default_output_load: 0.01
interned_strings: ["A", "Y", "!A", "B", "!(A*B)"]
"#;
    for (library, error) in [
        (timingless.to_owned(), "no timing payloads"),
        (
            make_timing_enabled_inv_nand2_liberty_textproto()
                .replace("default_output_load: 0.01", ""),
            "requires an output load",
        ),
    ] {
        std::fs::write(&lib, library).unwrap();
        for enable in [None, Some("--fanout-opt"), Some("--fanout-opt=true")] {
            std::fs::write(&netlist, "preserve netlist").unwrap();
            std::fs::write(&report, "preserve report").unwrap();
            let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
            command
                .arg("aig-tech-map")
                .arg(&aig)
                .args(["--strategy", "area", "--liberty_proto"])
                .arg(&lib)
                .arg("--netlist_out")
                .arg(&netlist)
                .arg("--json_out")
                .arg(&report);
            if let Some(flag) = enable {
                command.arg(flag);
            }
            let output = command.output().unwrap();
            assert!(!output.status.success());
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(stderr.contains(error), "{stderr}");
            assert!(stderr.contains("--fanout-opt=false"), "{stderr}");
            assert!(output.stdout.is_empty());
            assert_eq!(
                std::fs::read_to_string(&netlist).unwrap(),
                "preserve netlist"
            );
            assert_eq!(std::fs::read_to_string(&report).unwrap(), "preserve report");
        }
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .arg("aig-tech-map")
            .arg(&aig)
            .args([
                "--strategy",
                "area",
                "--fanout-opt=false",
                "--liberty_proto",
            ])
            .arg(&lib)
            .arg("--netlist_out")
            .arg(&netlist)
            .arg("--json_out")
            .arg(&report)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
        assert!(json.get("fanout_optimization").is_none());
        assert!(!String::from_utf8_lossy(&output.stderr).contains("fanout delay"));
        if error == "requires an output load" {
            let mut previous = None;
            for (load_flag, slew_flag) in [
                ("--module-output-load", "--primary-input-transition"),
                ("--module_output_load", "--primary_input_transition"),
            ] {
                let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
                    .arg("aig-tech-map")
                    .arg(&aig)
                    .args([
                        "--strategy",
                        "area",
                        load_flag,
                        "0.02",
                        slew_flag,
                        "0.03",
                        "--liberty_proto",
                    ])
                    .arg(&lib)
                    .arg("--netlist_out")
                    .arg(&netlist)
                    .arg("--json_out")
                    .arg(&report)
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let json: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
                assert_eq!(
                    json["fanout_optimization"]["options"]["module_output_load"],
                    0.02
                );
                assert_eq!(
                    json["fanout_optimization"]["options"]["primary_input_transition"],
                    0.03
                );
                let netlist_text = std::fs::read_to_string(&netlist).unwrap();
                if let Some((old_report, old_netlist)) = previous {
                    assert_eq!(json, old_report);
                    assert_eq!(netlist_text, old_netlist);
                }
                previous = Some((json, netlist_text));
            }
        }
    }
}

/// Composed recovery must match an explicit native resize with the same model.
#[test]
fn delay_area_recovery_composes_after_fanout_with_shared_model_and_budgets() {
    let dir = tempfile::tempdir().unwrap();
    let aig = dir.path().join("input.aag");
    let lib = dir.path().join("library.textproto");
    std::fs::write(&aig, make_two_input_and_aig()).unwrap();
    let text = make_timing_enabled_inv_nand2_liberty_textproto();
    // A size-only cell is deliberately unavailable to covering, but recovery
    // must select its smaller, equally fast implementation.
    let recovery_cell = text
        .split("cells: {")
        .nth(2)
        .unwrap()
        .split("default_output_load:")
        .next()
        .unwrap()
        .replace(
            "name: \"NAND2\"",
            "name: \"NAND2_SMALL\" use_for_size_only: true",
        )
        .replace("area: 2.0", "area: 1.0");
    std::fs::write(&lib, format!("{text}\ncells: {{{recovery_cell}")).unwrap();
    for repair in [false, true] {
        let run_map = |filename: &str, extra: &[&str]| {
            let netlist = dir.path().join(filename);
            let report = netlist.with_extension("json");
            let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
                .arg("aig-tech-map")
                .arg(&aig)
                .args(["--strategy", "delay", "--liberty_proto"])
                .arg(&lib)
                .arg(format!("--fanout-opt={repair}"))
                .args([
                    "--primary-input-transition",
                    "0.03",
                    "--module-output-load",
                    "0.02",
                    "--netlist_out",
                ])
                .arg(&netlist)
                .arg("--json_out")
                .arg(&report)
                .args(extra)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let json: serde_json::Value =
                serde_json::from_slice(&std::fs::read(report).unwrap()).unwrap();
            (netlist, json)
        };
        let (raw, raw_report) = run_map("raw.gv", &["--area-recovery=false"]);
        assert!(raw_report.get("area_recovery").is_none());
        let (zero, zero_report) = run_map("zero.gv", &["--area-recovery-evaluation-budget", "0"]);
        assert_eq!(std::fs::read(&raw).unwrap(), std::fs::read(zero).unwrap());
        assert_eq!(
            zero_report["area_recovery"]["work_limits"]["recovery_evaluations"],
            0
        );
        assert_eq!(
            zero_report["area_recovery"]["downsize"]["propagated_instances"],
            0
        );
        let (_, rounds_report) = run_map("rounds.gv", &["--area-recovery-rounds", "0"]);
        assert_eq!(
            rounds_report["area_recovery"]["downsize"]["stop_reason"],
            "disabled"
        );

        let wire = [
            "--wire-capacitance-constant-ff",
            "500",
            "--wire-capacitance-per-sink-ff",
            "250",
        ];
        let (before, _) = if repair {
            run_map(
                "before.gv",
                &["--area-recovery=false", wire[0], wire[1], wire[2], wire[3]],
            )
        } else {
            (raw, raw_report)
        };
        let (after, report) = run_map("after.gv", &wire);
        let explicit = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .arg("gv-resize")
            .arg("--netlist")
            .arg(before)
            .arg("--liberty_proto")
            .arg(&lib)
            .args([
                "--primary_input_transition",
                "0.03",
                "--module_output_load",
                "0.02",
                "--upsize-rounds",
                "0",
                "--max-area-growth",
                "0",
            ])
            .args(wire)
            .output()
            .unwrap();
        assert!(
            explicit.status.success(),
            "{}",
            String::from_utf8_lossy(&explicit.stderr)
        );
        let optimized_gv = std::fs::read(after).unwrap();
        assert_eq!(optimized_gv, explicit.stdout);
        assert_eq!(
            report["post_mapping_timing"]["wire_load"]["constant_capacitance"],
            0.5
        );
        assert_eq!(
            report["post_mapping_timing"]["wire_load"]["capacitance_per_sink"],
            0.25
        );
        assert_eq!(report["area_recovery_rounds"], 2);
        assert_eq!(report["area_recovery"]["initial_area"], 3.0);
        assert_eq!(report["area_recovery"]["final_area"], 2.0);
        assert_eq!(report["area_recovery"]["downsize"]["accepted"], 1);
        assert_eq!(report["area_recovery"]["upsize"]["stop_reason"], "disabled");
        assert_eq!(
            report["area_recovery"]["work_limits"]["timing_evaluations"],
            0
        );
        if repair {
            assert_eq!(
                report["fanout_optimization"]["options"]["wire_load"],
                report["post_mapping_timing"]["wire_load"]
            );
            assert_eq!(
                report["fanout_optimization"]["after"]["area"],
                report["area_recovery"]["initial_area"]
            );
        }
        let (pb, repeated) = run_map("repeat.pb", &wire);
        assert_eq!(report, repeated);
        let decoded = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .arg("pb2gv")
            .arg(pb)
            .output()
            .unwrap();
        assert!(
            decoded.status.success(),
            "{}",
            String::from_utf8_lossy(&decoded.stderr)
        );
        assert_eq!(decoded.stdout, optimized_gv);
    }
}

#[test]
fn recovery_flags_reject_inapplicable_stages_before_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let aig = dir.path().join("unused.aag");
    for (extra, error) in [
        (
            vec!["--area-recovery=true"],
            "--area-recovery requires --strategy area or delay",
        ),
        (
            vec!["--area-recovery-rounds", "0"],
            "area recovery tuning options require --area-recovery=true",
        ),
        (
            vec!["--wire-capacitance-constant-ff", "0"],
            "wire capacitance options require fanout optimization or area recovery",
        ),
        (
            vec![
                "--strategy",
                "delay",
                "--area-recovery=false",
                "--area-recovery-evaluation-budget",
                "1",
            ],
            "area recovery tuning options require --area-recovery=true",
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .arg("aig-tech-map")
            .arg(&aig)
            .args(["--liberty_proto", "unused.proto", "--netlist_out", "-"])
            .args(extra)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            format!("aig-tech-map error: {error}\n")
        );
        assert!(output.stdout.is_empty());
    }
}
