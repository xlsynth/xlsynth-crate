// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{Result, ensure};
use prost::Message;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use xlsynth_g8r::aig::{
    AigBitVector, ClockPort, GateBuilder, GateBuilderOptions, SequentialGateFn,
    add_input_registers, add_output_registers,
};
use xlsynth_g8r::aig_serdes::g8r::{encode_g8r_binary, load_sequential_gate_fn_from_path};
use xlsynth_g8r::liberty::load::TimingLibrarySnapshot;
use xlsynth_g8r::liberty::model::library_to_proto;
use xlsynth_g8r::liberty::parser::parse_liberty_files;
use xlsynth_g8r::netlist::emit::emit_netlist_as_text;
use xlsynth_g8r::netlist::fanout_opt::FanoutOptimizationOptions;
use xlsynth_g8r::netlist::resize::{RecoveryCellMenu, ResizeOptions};
use xlsynth_g8r::netlist::sequential_mapping::{
    SequentialMappingOptions, map_sequential_to_native_netlist,
};
use xlsynth_g8r::netlist::sta::StaOptions;
use xlsynth_g8r::techmap::delay::DelayMappingOptions;
use xlsynth_g8r::techmap::postprocess::{CoverStrategy, PostMappingOptions};

const DRIVER: &str = env!("CARGO_BIN_EXE_xlsynth-driver");

// Build the reference by running standalone sizing on each emitted
// checkpoint. The integrated command must match both gate bytes and the
// phase reports, including the skipped area-only phase.
#[test]
fn default_registered_mapping_matches_separate_resize_commands() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let dir = temp.path();
    let digest = write_fixture(dir)?;

    let initial = map_before_adaptive_sizing(dir)?;
    let mapped_sha256 = format!("{:x}", Sha256::digest(&initial));
    let (expected, phases) = resize_to_stability(dir, initial)?;
    let mut default_map = native_map(dir);
    default_map
        .arg("--adaptive-resize-report")
        .arg(dir.join("finish.json"));
    assert_eq!(success(&mut default_map)?, expected);
    let finish: Value = serde_json::from_slice(&fs::read(dir.join("finish.json"))?)?;
    for phase in ["area", "mixed"] {
        assert_eq!(finish[phase]["epochs"], phases[phase]["epochs"], "{phase}");
        assert_eq!(finish[phase]["stop"], phases[phase]["stop"], "{phase}");
    }
    assert_eq!(finish["sizing_liberty_sha256"], digest);
    assert_eq!(
        finish["output_sha256"],
        format!("{:x}", Sha256::digest(&expected))
    );
    assert_eq!(finish["mapped_sha256"], mapped_sha256);
    Ok(())
}

/// Construct the initial registered netlist through the library so the test
/// can run the finishing phases independently with gv-resize.
fn map_before_adaptive_sizing(dir: &Path) -> Result<Vec<u8>> {
    let design =
        load_sequential_gate_fn_from_path(&dir.join("and.g8rbin")).map_err(anyhow::Error::msg)?;
    let clock = ClockPort { name: "clk".into() };
    let launched = add_input_registers(&design, clock.clone()).map_err(anyhow::Error::msg)?;
    let mut design = add_output_registers(&launched, clock).map_err(anyhow::Error::msg)?;
    design.name = "top".into();
    let library = TimingLibrarySnapshot::from_path(&dir.join("cells.proto"))?.library;
    let timing = StaOptions {
        primary_input_transition: 0.01,
        module_output_load: 0.01,
        ..Default::default()
    };
    let sizing = ResizeOptions {
        timing,
        recovery_cell_menu: RecoveryCellMenu::ElectricalMinArea,
        recovery_cell_limit: Some(12),
        ..Default::default()
    };
    let mut mapped = map_sequential_to_native_netlist(
        &design,
        &library,
        &SequentialMappingOptions {
            cleanup_registers: true,
            clock_gating: None,
            ..Default::default()
        },
        CoverStrategy::Delay(DelayMappingOptions {
            reference_transition: timing.primary_input_transition,
            reference_load: timing.module_output_load,
        }),
        &PostMappingOptions {
            fanout: Some(FanoutOptimizationOptions {
                primary_input_transition: timing.primary_input_transition,
                module_output_load: timing.module_output_load,
                ..Default::default()
            }),
            recovery: Some(ResizeOptions {
                timing,
                max_area_growth: 0.0,
                max_upsize_rounds: 0,
                timing_evaluation_budget: Some(0),
                ..Default::default()
            }),
        },
    )?;
    xlsynth_g8r::netlist::resize::resize_netlist(
        &mut mapped.modules[0],
        &mapped.nets,
        &mut mapped.interner,
        &library,
        &sizing,
    )?;
    Ok(emit_netlist_as_text(&mapped)?.into_bytes())
}

/// Exercise the map and the mixed phase independently of the driver's
/// integrated flow. A single registered AND reaches a fixed point quickly.
fn resize_to_stability(dir: &Path, mut previous: Vec<u8>) -> Result<(Vec<u8>, Value)> {
    let mut reports = json!({"area": {"epochs": [], "stop": "skipped"}});
    for (name, cap, phase) in [(
        "mixed",
        46,
        &[
            "--upsize-rounds=10",
            "--downsize-rounds=4",
            "--max-area-growth=0.02",
            "--timing-evaluation-budget=250000",
            "--timing-cell-menu=delay-bands-fastest-first",
        ][..],
    )] {
        let mut epochs = Vec::<Value>::new();
        for number in 1..=cap {
            fs::write(dir.join("epoch.gv"), &previous)?;
            let next = success(
                resize_command(dir)
                    .arg("--monotone-recovery-checkpoint")
                    .args(phase)
                    .arg("--report-json")
                    .arg(dir.join("epoch.json")),
            )?;
            epochs.push(serde_json::from_slice(&fs::read(dir.join("epoch.json"))?)?);
            if next == previous {
                break;
            }
            ensure!(number < cap, "fixture did not converge within {cap} epochs");
            previous = next;
        }
        reports[name] = json!({"epochs": epochs, "stop": "identical-gate"});
    }
    Ok((previous, reports))
}

fn native_map(dir: &Path) -> Command {
    let mut cmd = Command::new(DRIVER);
    cmd.arg("g8r2gv")
        .arg(dir.join("and.g8rbin"))
        .arg("--liberty-proto")
        .arg(dir.join("cells.proto"))
        .args([
            "--module-name=top",
            "--add-clk-port=clk",
            "--flop-inputs",
            "--flop-outputs",
            "--cleanup-registers",
            "--clock-gating=false",
        ]);
    cmd
}

fn resize_command(dir: &Path) -> Command {
    let mut cmd = Command::new(DRIVER);
    cmd.arg("gv-resize")
        .arg("--netlist")
        .arg(dir.join("epoch.gv"))
        .arg("--liberty_proto")
        .arg(dir.join("cells.proto"))
        .args([
            "--module_name=top",
            "--primary_input_transition=0.01",
            "--module_output_load=0.01",
            "--wire-capacitance-constant-ff=0",
            "--wire-capacitance-per-sink-ff=0",
            "--no-register-resize",
            "--recovery-cell-menu=electrical-min-area",
            "--recovery-cell-limit=12",
        ]);
    cmd
}

fn success(cmd: &mut Command) -> Result<Vec<u8>> {
    let out = cmd.output()?;
    ensure!(
        out.status.success(),
        "{cmd:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    Ok(out.stdout)
}

/// Make the function and library visible to both the mapper and the
/// independent resize commands.
fn write_fixture(dir: &Path) -> Result<String> {
    let lib_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/testdata/native_registered_timing.lib");
    let mut library = parse_liberty_files(&[lib_path]).map_err(anyhow::Error::msg)?;
    library.default_output_load = Some(0.01);
    let proto = dir.join("cells.proto");
    fs::write(&proto, library_to_proto(library)?.encode_to_vec())?;

    let mut builder = GateBuilder::new("test".into(), GateBuilderOptions::no_opt());
    let a = *builder.add_input("a".into(), 1).get_lsb(0);
    let b = *builder.add_input("b".into(), 1).get_lsb(0);
    let and = builder.add_and_binary(a, b);
    builder.add_output("y".into(), AigBitVector::from_lsb_is_index_0(&[and]));
    let design = SequentialGateFn::from_gate_fn(builder.build());
    fs::write(
        dir.join("and.g8rbin"),
        encode_g8r_binary(&design).map_err(anyhow::Error::msg)?,
    )?;
    Ok(TimingLibrarySnapshot::from_path(&proto)?.sha256)
}

const VIEW: &str = include_str!("../testdata/native_registered_timing.lib");

/// A separate sizing view may add cheaper masters and spell an equivalent
/// function differently. The meaning of the mapped circuit must survive.
#[test]
fn sizing_view_checks_cell_behavior_and_module_name() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let dir = temp.path();
    write_fixture(dir)?;
    let alternate = dir.join("alternate.proto");
    let mut map = native_map(dir);
    map.arg("--resize-liberty-proto").arg(&alternate);

    let body = VIEW.trim_end().strip_suffix('}').unwrap();
    let and = &body[body.find("cell (AND2)").unwrap()..];
    let small = and
        .replace("AND2", "AND2_SMALL")
        .replace("area: 1", "area: 0.5");
    // The cheapest AND in the sizing view is an OR in the mapping view.
    // Skipping that master lets the search choose AND2_SMALL.
    let conflict = and
        .replace("AND2", "CONFLICT")
        .replace("area: 1", "area: 0.25");
    let mapping = format!("{body}{}}}", conflict.replace("A & B", "A | B"));
    write_view(&dir.join("cells.proto"), &mapping)?;
    let compatible = format!("{body}{small}{conflict}}}").replace("A & B", "B & A");
    write_view(&alternate, &compatible)?;
    let gates = String::from_utf8(success(&mut map)?)?;
    assert!(gates.contains("AND2_SMALL "), "{gates}");

    for (view, diagnostic) in [
        (
            compatible.replace("B & A", "B | A"),
            "sizing library has incompatible behavior or pins for cell",
        ),
        (
            compatible.replace(r#"next_state: "D""#, r#"next_state: "!D""#),
            "sizing library has incompatible behavior or pins for cell",
        ),
        (
            compatible.replace("AND2_SMALL", "top"),
            "top module name conflicts with a Liberty cell name",
        ),
    ] {
        write_view(&alternate, &view)?;
        let out = map.output()?;
        let stderr = String::from_utf8(out.stderr)?;
        assert!(
            !out.status.success() && stderr.contains(diagnostic),
            "expected {diagnostic}, got {stderr}"
        );
        assert!(out.stdout.is_empty());
    }
    Ok(())
}

fn write_view(proto: &Path, text: &str) -> Result<()> {
    let lib = proto.with_extension("lib");
    fs::write(&lib, text)?;
    let mut library = parse_liberty_files(&[lib]).map_err(anyhow::Error::msg)?;
    library.default_output_load = Some(0.1);
    fs::write(proto, library_to_proto(library)?.encode_to_vec())?;
    Ok(())
}

/// The input registers' setup arcs and output register's clock-to-Q arc
/// expose the physical slew and load used by sizing, in ps and fF.
#[test]
fn sizing_view_uses_physical_constraints_from_mapping() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let dir = temp.path();
    write_fixture(dir)?;
    write_view(&dir.join("cells.proto"), VIEW)?;
    let alternate = dir.join("alternate.proto");
    let sizing = VIEW
        .replace("1ns", "1ps")
        .replace("capacitive_load_unit (1, pf);", r#"
            capacitive_load_unit (1, ff);
            lu_table_template (setup) { variable_1: constrained_pin_transition; index_1 ("0,20"); }
            lu_table_template (load) { variable_1: total_output_net_capacitance; index_1 ("0,100"); }
        "#)
        .replace(r#"constraint (scalar) { values ("0.1"); }"#, r#"constraint (setup) { values ("0,20"); }"#)
        .replace(r#"(scalar) { values ("1"); }"#, r#"(load) { values ("0,100"); }"#);
    write_view(&alternate, &sizing)?;
    let report = dir.join("sizing.json");
    for (flags, slew_ps, load_ff) in [
        (vec![], 10.0, 100.0),
        (
            vec!["--primary-input-transition=0.01", "--output-load-ff=100"],
            10.0,
            100.0,
        ),
        (
            vec!["--primary-input-transition=0.02", "--output-load-ff=50"],
            20.0,
            50.0,
        ),
    ] {
        success(
            native_map(dir)
                .arg("--resize-liberty-proto")
                .arg(&alternate)
                .arg("--adaptive-resize-report")
                .arg(&report)
                .args(&flags),
        )?;
        let json: Value = serde_json::from_slice(&fs::read(&report)?)?;
        let delays = &json["mixed"]["epochs"].as_array().unwrap().last().unwrap()["final_delay"];
        let setup = delays["input_to_register"].as_f64().unwrap();
        let cq = delays["register_to_output"].as_f64().unwrap();
        assert!((setup - slew_ps).abs() < 1e-5, "{flags:?} {delays}");
        assert!((cq - load_ff).abs() < 1e-5, "{flags:?} {delays}");
    }
    Ok(())
}
