// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
#[cfg(unix)]
use std::io::Read;
#[cfg(unix)]
use std::os::fd::OwnedFd;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process::Command;
#[cfg(unix)]
use std::process::Stdio;
#[cfg(unix)]
use std::time::{Duration, Instant};

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
    for assignment in 0..4 {
        let inputs = (0..2)
            .map(|i| IrBits::make_ubits(1, (assignment >> i) & 1).unwrap())
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
            nand2_outputs.push(text.clone());
        }
        assert_mapping_equivalent(&aig, &text, &aig_path, &lib_path);
    }
    assert_eq!(nand2_outputs[0], nand2_outputs[1]);
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
        .args(["--strategy", "area"])
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
        "aig-tech-map error: --json_out and --netlist_out must name different files\n"
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
        "aig-tech-map error: --json_out must not refer to stdout when --netlist_out is '-'\n"
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
            "aig-tech-map error: --json_out must not refer to stdout when --netlist_out is '-'\n"
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
            "aig-tech-map error: --json_out must not refer to stdout when --netlist_out is '-'\n"
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
            "aig-tech-map error: --json_out must not refer to stdout when --netlist_out is '-'\n"
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
                "aig-tech-map error: --json_out and --netlist_out must name different files\n"
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
            "aig-tech-map error: --json_out and --netlist_out must name different files\n"
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
        "aig-tech-map error: --json_out and --netlist_out must name different files\n"
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
