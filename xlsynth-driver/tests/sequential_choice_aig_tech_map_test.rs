// SPDX-License-Identifier: Apache-2.0

//! Driver-facing coverage for transition AIG and native register mapping.

use std::collections::BTreeSet;
use std::io::Cursor;
use std::path::PathBuf;
use std::process::{Command, Output};

use tempfile::TempDir;
use xlsynth_g8r::aig::{
    AigBitVector, ClockPort, RegisterBinding, SequentialGateFn, TransitionInputId,
    TransitionOutputId,
};
use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
use xlsynth_g8r::aig_serdes::g8r::{emit_g8r, encode_g8r_binary};
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::netlist::parse::{Parser as NetlistParser, TokenScanner};

const SEQUENTIAL_LIBERTY_TEXTPROTO: &str = r#"
format_magic: 5496997758177923663
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
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 1.0 }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 0.1 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: 0.1 }
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
    function_string_id: 8
    timing_arcs: {
      related_pin_string_id: 1
      timing_sense: TIMING_SENSE_NEGATIVE_UNATE
      timing_type: TIMING_TYPE_COMBINATIONAL
      tables: { kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: 2.0 }
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 2.0 }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 0.1 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: 0.1 }
    }
    timing_arcs: {
      related_pin_string_id: 4
      timing_sense: TIMING_SENSE_NEGATIVE_UNATE
      timing_type: TIMING_TYPE_COMBINATIONAL
      tables: { kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: 2.0 }
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 2.0 }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 0.1 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: 0.1 }
    }
  }
  area: 2.0
}
cells: {
  name: "DFF"
  pins: {
    name_string_id: 6
    direction: INPUT
    capacitance: 0.02
    timing_arcs: {
      related_pin_string_id: 5
      timing_type: TIMING_TYPE_SETUP_RISING
      tables: { kind: TIMING_TABLE_KIND_RISE_CONSTRAINT shape_id: 1 values: 0.25 }
      tables: { kind: TIMING_TABLE_KIND_FALL_CONSTRAINT shape_id: 1 values: 0.25 }
    }
  }
  pins: {
    name_string_id: 5
    direction: INPUT
    is_clocking_pin: true
    capacitance: 0.01
  }
  pins: {
    name_string_id: 7
    direction: OUTPUT
    function_string_id: 7
    timing_arcs: {
      related_pin_string_id: 5
      timing_sense: TIMING_SENSE_NON_UNATE
      timing_type: TIMING_TYPE_RISING_EDGE
      tables: { kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: 0.5 }
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 0.5 }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 0.1 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: 0.1 }
    }
  }
  area: 4.0
  sequential: {
    state_var: "Q"
    next_state: "D"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
  }
}
units: { time_unit: "1ps" capacitance_unit: "1pf" }
interned_strings: ["A", "Y", "!A", "B", "CLK", "D", "Q", "!(A*B)"]
lut_shapes: {}
"#;

struct SequentialMappingFixture {
    temp_dir: TempDir,
    aiger_path: PathBuf,
    liberty_path: PathBuf,
    sequential_design_path: PathBuf,
    netlist_path: PathBuf,
}

/// Constructs a one-register design with both feedback and output logic.
fn make_sequential_design() -> SequentialGateFn {
    let mut builder = GateBuilder::new(
        "pipeline__transition".to_string(),
        GateBuilderOptions::no_opt(),
    );
    let data = builder.add_input("data".to_string(), 1);
    let state_q = builder.add_input("state__q".to_string(), 1);
    let visible_output = builder.add_not(*state_q.get_lsb(0));
    let next_state = builder.add_and_binary(*data.get_lsb(0), *state_q.get_lsb(0));
    builder.add_output("out".to_string(), AigBitVector::from_bit(visible_output));
    builder.add_output("state__d".to_string(), AigBitVector::from_bit(next_state));

    SequentialGateFn::new(
        "pipeline".to_string(),
        builder.build(),
        vec![TransitionInputId::new(0)],
        vec![TransitionOutputId::new(0)],
        Some(ClockPort {
            name: "clk".to_string(),
        }),
        vec![RegisterBinding {
            name: "state".to_string(),
            q: TransitionInputId::new(1),
            d: TransitionOutputId::new(1),
            initial_value: None,
        }],
    )
    .expect("build one-register sequential mapping fixture")
}

/// Writes matching ordinary AIGER, Liberty, and native design artifacts.
fn make_fixture(native_extension: &str) -> SequentialMappingFixture {
    let temp_dir = tempfile::tempdir().expect("create sequential mapping tempdir");
    let design = make_sequential_design();
    let aiger_path = temp_dir.path().join("pipeline.transition.aag");
    let liberty_path = temp_dir.path().join("sequential.textproto");
    let sequential_design_path = temp_dir.path().join(format!("pipeline.{native_extension}"));
    let netlist_path = temp_dir.path().join("pipeline.mapped.gv");

    std::fs::write(
        &aiger_path,
        emit_aiger(&design.transition, true).expect("serialize transition AIGER"),
    )
    .expect("write transition AIGER");
    std::fs::write(&liberty_path, SEQUENTIAL_LIBERTY_TEXTPROTO)
        .expect("write sequential Liberty proto");
    match native_extension {
        "g8r" => std::fs::write(&sequential_design_path, emit_g8r(&design))
            .expect("write native text design"),
        "g8rbin" => std::fs::write(
            &sequential_design_path,
            encode_g8r_binary(&design).expect("encode native binary design"),
        )
        .expect("write native binary design"),
        _ => panic!("unsupported fixture extension '{native_extension}'"),
    }

    SequentialMappingFixture {
        temp_dir,
        aiger_path,
        liberty_path,
        sequential_design_path,
        netlist_path,
    }
}

/// Invokes the mapper with a shared sequential test fixture.
fn run_choice_mapping(fixture: &SequentialMappingFixture, extra_args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("choice-aig-tech-map")
        .arg(&fixture.aiger_path)
        .arg("--liberty_proto")
        .arg(&fixture.liberty_path)
        .arg("--netlist_out")
        .arg(&fixture.netlist_path)
        .args(extra_args)
        .output()
        .expect("invoke choice-AIG technology mapper")
}

/// Verifies that a native design becomes a mapped sequential cell netlist.
fn assert_sequential_mapping(native_extension: &str) {
    let fixture = make_fixture(native_extension);
    let design_path = fixture
        .sequential_design_path
        .to_str()
        .expect("UTF-8 temporary native design path");
    let output = run_choice_mapping(
        &fixture,
        &["--sequential-design", design_path, "--clock-period", "10"],
    );
    assert!(
        output.status.success(),
        "sequential {native_extension} mapping failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let netlist = std::fs::read(&fixture.netlist_path).expect("read mapped sequential netlist");
    let scanner = TokenScanner::with_line_lookup(Cursor::new(netlist), Box::new(|_| None));
    let mut parser = NetlistParser::new(scanner);
    let modules = parser
        .parse_file()
        .expect("parse mapped sequential netlist");
    assert_eq!(modules.len(), 1);
    let module = &modules[0];
    assert_eq!(parser.interner.resolve(module.name), Some("pipeline"));

    let port_names = module
        .ports
        .iter()
        .map(|port| {
            parser
                .interner
                .resolve(port.name)
                .expect("resolve sequential module port")
                .to_string()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        port_names,
        BTreeSet::from(["clk".to_string(), "data".to_string(), "out".to_string()])
    );

    let register_count = module
        .instances
        .iter()
        .filter(|instance| parser.interner.resolve(instance.type_name) == Some("DFF"))
        .count();
    assert_eq!(register_count, 1);

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("registers=1"),
        "sequential diagnostics did not report the register: {stderr}"
    );
    assert!(
        stderr.contains("timing-model=nf-liberty"),
        "sequential mapping did not use representative Liberty timing: {stderr}"
    );

    let stats_path = fixture.temp_dir.path().join("pipeline.gv_stats.json");
    let stats_output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("gv-stats")
        .arg("--netlist")
        .arg(&fixture.netlist_path)
        .arg("--liberty_proto")
        .arg(&fixture.liberty_path)
        .arg("--json_out")
        .arg(&stats_path)
        .output()
        .expect("invoke register-aware gv-stats");
    assert!(
        stats_output.status.success(),
        "mapped {native_extension} netlist failed register-aware STA:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&stats_output.stdout),
        String::from_utf8_lossy(&stats_output.stderr)
    );
    let stats: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&stats_path).expect("read register-aware gv-stats JSON"),
    )
    .expect("parse register-aware gv-stats JSON");
    assert_eq!(stats["sequential_cell_area"], 4.0);
    for metric in [
        "max_input_to_register_delay",
        "max_register_to_register_delay",
        "max_register_to_output_delay",
    ] {
        assert!(
            stats[metric]
                .as_f64()
                .is_some_and(|arrival| arrival.is_finite() && arrival > 0.0),
            "mapped sequential netlist has no valid {metric}: {stats}"
        );
    }
}

#[test]
fn choice_aig_tech_map_loads_native_text_sequential_design() {
    assert_sequential_mapping("g8r");
}

#[test]
fn choice_aig_tech_map_loads_native_binary_sequential_design() {
    assert_sequential_mapping("g8rbin");
}

#[test]
fn choice_aig_tech_map_accepts_register_aware_buffering() {
    let fixture = make_fixture("g8rbin");
    let design_path = fixture
        .sequential_design_path
        .to_str()
        .expect("UTF-8 temporary native design path");
    let output = run_choice_mapping(
        &fixture,
        &[
            "--sequential-design",
            design_path,
            "--clock-period",
            "10",
            "--buffer",
            "true",
        ],
    );

    assert!(
        output.status.success(),
        "register-aware buffered mapping failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    assert!(diagnostics.contains("registers=1"));
    assert!(diagnostics.contains("buffers="));
    assert!(fixture.netlist_path.is_file());
}

#[test]
fn choice_aig_tech_map_accepts_register_aware_resizing() {
    let fixture = make_fixture("g8rbin");
    let design_path = fixture
        .sequential_design_path
        .to_str()
        .expect("UTF-8 temporary native design path");
    let output = run_choice_mapping(
        &fixture,
        &[
            "--sequential-design",
            design_path,
            "--clock-period",
            "10",
            "--resize",
            "true",
            "--resize-rounds",
            "2",
        ],
    );

    assert!(
        output.status.success(),
        "register-aware resized mapping failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    assert!(diagnostics.contains("registers=1"));
    assert!(diagnostics.contains("upsizes="));
    assert!(fixture.netlist_path.is_file());
}

#[test]
fn choice_aig_tech_map_accepts_registered_buffering_and_resizing() {
    let fixture = make_fixture("g8rbin");
    let design_path = fixture
        .sequential_design_path
        .to_str()
        .expect("UTF-8 temporary native design path");
    let output = run_choice_mapping(
        &fixture,
        &[
            "--sequential-design",
            design_path,
            "--clock-period",
            "10",
            "--buffer",
            "true",
            "--resize",
            "true",
        ],
    );

    assert!(
        output.status.success(),
        "register-aware buffered and resized mapping failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    assert!(diagnostics.contains("registers=1"));
    assert!(diagnostics.contains("buffers="));
    assert!(diagnostics.contains("upsizes="));
    assert!(fixture.netlist_path.is_file());
}

#[test]
fn choice_aig_tech_map_remains_combinational_without_sequential_design() {
    let fixture = make_fixture("g8r");
    let output = run_choice_mapping(&fixture, &[]);
    assert!(
        output.status.success(),
        "combinational mapping failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let netlist = std::fs::read(&fixture.netlist_path).expect("read mapped combinational netlist");
    let scanner = TokenScanner::with_line_lookup(Cursor::new(netlist), Box::new(|_| None));
    let mut parser = NetlistParser::new(scanner);
    let modules = parser
        .parse_file()
        .expect("parse mapped combinational transition");
    assert_eq!(modules.len(), 1);
    let module = &modules[0];
    assert!(
        module
            .instances
            .iter()
            .all(|instance| parser.interner.resolve(instance.type_name) != Some("DFF")),
        "combinational mode must not instantiate a flip-flop"
    );
    let port_names = module
        .ports
        .iter()
        .map(|port| {
            parser
                .interner
                .resolve(port.name)
                .expect("resolve combinational module port")
                .to_string()
        })
        .collect::<BTreeSet<_>>();
    assert!(port_names.contains("state__q"));
    assert!(port_names.contains("state__d"));
    assert!(!port_names.contains("clk"));
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    assert!(
        diagnostics.contains("timing-model=nf-liberty"),
        "combinational mapping did not use representative Liberty timing: {diagnostics}"
    );
    assert!(!diagnostics.contains("registers="));
}

#[test]
fn choice_aig_tech_map_panics_when_nf_unit_is_selected() {
    let fixture = make_fixture("g8rbin");
    let output = run_choice_mapping(&fixture, &["--timing-model", "nf-unit"]);

    assert_eq!(output.status.code(), Some(101));
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    assert!(
        diagnostics.contains(
            "nf-unit technology mapping is disabled; use nf-liberty representative Liberty pin delays"
        ),
        "unit-delay mapping did not produce the expected panic: {diagnostics}"
    );
    assert!(!fixture.netlist_path.exists());
}

#[test]
fn choice_aig_tech_map_clock_period_requires_sequential_design() {
    let fixture = make_fixture("g8r");
    let output = run_choice_mapping(&fixture, &["--clock-period", "10"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--clock-period") && stderr.contains("--sequential-design"),
        "unexpected missing sequential-design error: {stderr}"
    );
    assert!(!fixture.netlist_path.exists());
}

#[test]
fn choice_aig_tech_map_rejects_non_positive_or_non_finite_clock_periods() {
    let fixture = make_fixture("g8r");
    let design_path = fixture
        .sequential_design_path
        .to_str()
        .expect("UTF-8 temporary native design path");

    for invalid_period in ["0", "-1", "NaN", "inf", "-inf"] {
        let period_argument = format!("--clock-period={invalid_period}");
        let output = run_choice_mapping(
            &fixture,
            &["--sequential-design", design_path, &period_argument],
        );
        assert!(
            !output.status.success(),
            "clock period '{invalid_period}' was unexpectedly accepted"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("finite positive"),
            "unexpected error for clock period '{invalid_period}': {stderr}"
        );
        assert!(!fixture.netlist_path.exists());
    }
}

#[test]
fn choice_aig_tech_map_rejects_mismatched_sequential_transition() {
    let fixture = make_fixture("g8rbin");
    let mut builder = GateBuilder::new(
        "mismatched_transition".to_string(),
        GateBuilderOptions::no_opt(),
    );
    let unrelated_input = builder.add_input("unrelated".to_string(), 1);
    builder.add_output("unrelated_out".to_string(), unrelated_input);
    std::fs::write(
        &fixture.aiger_path,
        emit_aiger(&builder.build(), true).expect("encode mismatched transition"),
    )
    .expect("write mismatched transition");

    let design_path = fixture
        .sequential_design_path
        .to_str()
        .expect("UTF-8 temporary native design path");
    let output = run_choice_mapping(&fixture, &["--sequential-design", design_path]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("sequential choice-AIG technology mapping failed"),
        "unexpected transition-interface error: {stderr}"
    );
    assert!(!fixture.netlist_path.exists());
}
