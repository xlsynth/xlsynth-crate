// SPDX-License-Identifier: Apache-2.0

//! Driver-facing coverage for bit-granular sequential register cleanup.

use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::json;
use tempfile::TempDir;
use xlsynth::IrBits;
use xlsynth_g8r::aig::{
    AigBitVector, ClockPort, GateBuilder, GateBuilderOptions, RegisterBinding, SequentialGateFn,
    TransitionInputId, TransitionOutputId,
};
use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
use xlsynth_g8r::aig_serdes::g8r::{encode_g8r_binary, load_sequential_gate_fn_from_path};
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::aig_sim::sequential::{SequentialState, simulate};

struct RegisterCleanupFixture {
    _temp_dir: TempDir,
    design: SequentialGateFn,
    source: PathBuf,
    checkpoint: PathBuf,
    cleaned_design: PathBuf,
    cleaned_transition: PathBuf,
    stats: PathBuf,
}

/// Creates a visible state with one constant and two equivalent data bits.
fn make_fixture() -> RegisterCleanupFixture {
    let temp_dir = tempfile::tempdir().expect("create register-cleanup tempdir");
    let mut builder = GateBuilder::new(
        "cleanup_pipeline__transition".to_string(),
        GateBuilderOptions::no_opt(),
    );
    let data = builder.add_input("data".to_string(), 1);
    let state = builder.add_input("state__q".to_string(), 3);
    let data_bit = *data.get_lsb(0);
    let constant_one = builder.get_true();
    builder.add_output("out".to_string(), state);
    builder.add_output(
        "state__d".to_string(),
        AigBitVector::from_lsb_is_index_0(&[data_bit, constant_one, data_bit]),
    );
    let design = SequentialGateFn::new(
        "cleanup_pipeline".to_string(),
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
    .expect("construct cleanup pipeline fixture");

    let source = temp_dir.path().join("pipeline.g8rbin");
    let checkpoint = temp_dir.path().join("pipeline.optimized.aag");
    let cleaned_design = temp_dir.path().join("pipeline.cleaned.g8rbin");
    let cleaned_transition = temp_dir.path().join("pipeline.cleaned.aig");
    let stats = temp_dir.path().join("pipeline.cleanup.json");
    std::fs::write(
        &source,
        encode_g8r_binary(&design).expect("encode register-cleanup fixture"),
    )
    .expect("write register-cleanup fixture");
    std::fs::write(
        &checkpoint,
        emit_aiger(&design.transition, true).expect("encode optimized transition checkpoint"),
    )
    .expect("write transition checkpoint");

    RegisterCleanupFixture {
        _temp_dir: temp_dir,
        design,
        source,
        checkpoint,
        cleaned_design,
        cleaned_transition,
        stats,
    }
}

/// Invokes the cleanup CLI with fully specified native checkpoint outputs.
fn run_cleanup(fixture: &RegisterCleanupFixture, extra_args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("g8r-cleanup-registers")
        .arg(&fixture.source)
        .arg("--optimized-transition")
        .arg(&fixture.checkpoint)
        .arg("--quiet=true")
        .arg("--bin-out")
        .arg(&fixture.cleaned_design)
        .arg("--transition-aiger-out")
        .arg(&fixture.cleaned_transition)
        .arg("--stats-out")
        .arg(&fixture.stats)
        .args(extra_args)
        .output()
        .expect("invoke sequential register cleanup")
}

#[test]
fn cleanup_rebinds_optimized_aig_and_emits_consistent_state_artifacts() {
    let fixture = make_fixture();
    let output = run_cleanup(
        &fixture,
        &["--initialization-policy", "uninitialized-dont-care"],
    );
    assert!(
        output.status.success(),
        "register cleanup failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty(), "quiet cleanup must suppress g8r");

    let cleaned = load_sequential_gate_fn_from_path(&fixture.cleaned_design)
        .expect("reload cleaned native sequential metadata");
    cleaned.validate().expect("validate cleaned native design");
    assert_eq!(cleaned.registers.len(), 1);
    assert_eq!(
        cleaned.transition.inputs[cleaned.registers[0].q.index()].get_bit_count(),
        1
    );
    assert_eq!(cleaned.transition.outputs[0].get_bit_count(), 3);

    let transition =
        load_aiger_auto_from_path(&fixture.cleaned_transition, GateBuilderOptions::no_opt())
            .expect("reload cleaned binary transition");
    assert_eq!(transition.gate_fn.inputs.len(), 2);
    assert_eq!(transition.gate_fn.outputs.len(), 4);

    let stats: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&fixture.stats).expect("read register-cleanup statistics"),
    )
    .expect("parse register-cleanup statistics");
    assert_eq!(
        stats,
        json!({
            "initial_register_bits": 3,
            "final_register_bits": 1,
            "dead_register_bits": 0,
            "constant_register_bits": 1,
            "merged_register_bits": 1,
            "initial_and_nodes": 0,
            "final_and_nodes": 0,
            "iterations": 2,
        })
    );

    let inputs = [false, true, true, false, true, false]
        .into_iter()
        .map(|bit| vec![IrBits::from_lsb_is_0(&[bit])])
        .collect::<Vec<_>>();
    let original = simulate(
        &fixture.design,
        &inputs,
        SequentialState::all_zeros(&fixture.design),
    )
    .expect("simulate original hardware pipeline");
    let actual = simulate(&cleaned, &inputs, SequentialState::all_zeros(&cleaned))
        .expect("simulate cleaned hardware pipeline");
    assert_eq!(
        &actual.external_outputs()[1..],
        &original.external_outputs()[1..]
    );
}

#[test]
fn cleanup_defaults_to_preserving_all_zero_cycle_behavior() {
    let fixture = make_fixture();
    let output = run_cleanup(&fixture, &[]);
    assert!(
        output.status.success(),
        "cycle-zero-preserving cleanup failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cleaned = load_sequential_gate_fn_from_path(&fixture.cleaned_design)
        .expect("reload cycle-zero-preserving result");
    assert_eq!(
        cleaned.transition.inputs[cleaned.registers[0].q.index()].get_bit_count(),
        2,
        "the initially zero constant-one bit must survive cycle zero"
    );

    let inputs = [false, true, false, true]
        .into_iter()
        .map(|bit| vec![IrBits::from_lsb_is_0(&[bit])])
        .collect::<Vec<_>>();
    let original = simulate(
        &fixture.design,
        &inputs,
        SequentialState::all_zeros(&fixture.design),
    )
    .expect("simulate original all-zero pipeline");
    let actual = simulate(&cleaned, &inputs, SequentialState::all_zeros(&cleaned))
        .expect("simulate cycle-zero-preserving cleanup");
    assert_eq!(actual.external_outputs(), original.external_outputs());
}

#[test]
fn cleanup_rejects_an_unrelated_transition_without_creating_outputs() {
    let fixture = make_fixture();
    let mut builder = GateBuilder::new(
        "unrelated__transition".to_string(),
        GateBuilderOptions::no_opt(),
    );
    let input = builder.add_input("unrelated".to_string(), 1);
    builder.add_output("unrelated_out".to_string(), input);
    std::fs::write(
        &fixture.checkpoint,
        emit_aiger(&builder.build(), true).expect("encode unrelated transition"),
    )
    .expect("replace fixture with unrelated transition");

    let output = run_cleanup(&fixture, &[]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("optimized transition input interface"),
        "unexpected transition-boundary rejection: {stderr}"
    );
    assert!(!fixture.cleaned_design.exists());
    assert!(!fixture.cleaned_transition.exists());
    assert!(!fixture.stats.exists());
}
