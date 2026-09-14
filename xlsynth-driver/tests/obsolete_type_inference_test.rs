// SPDX-License-Identifier: Apache-2.0

//! Old commands and saved configurations remain usable without selecting a
//! different typechecker or forwarding obsolete options to child commands.

use std::process::{Command, Output};
use test_case::test_case;
use xlsynth_driver::prover_config::{ProverPlan, ToDriverCommand};

const WARNING: &str = "type_inference_v2 is obsolete and ignored";

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Checks the default warning text, including its deterministic prefix.
fn assert_compatibility_warning(output: &Output, expected: bool) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let warnings: Vec<_> = stderr
        .lines()
        .filter(|line| line.contains(WARNING))
        .collect();
    if expected {
        assert_eq!(
            warnings,
            [format!(
                "[WARN  xlsynth_driver::obsolete_options] {WARNING}"
            )]
        );
    } else {
        assert!(warnings.is_empty());
    }
}

// Verifies: old CLI/TOML values warn and preserve each command's output.
// Catches: obsolete guards, hidden or timestamped warnings, and forwarding.
#[test_case("dslx2ir", false, false; "linked_cli")]
#[test_case("dslx2ir", true, false; "external_cli")]
#[test_case("dslx2ir", false, true; "linked_toml")]
#[test_case("dslx2ir", true, true; "external_toml")]
#[test_case("dslx2pipeline", false, false; "linked_pipeline")]
#[test_case("dslx2pipeline", true, false; "external_pipeline")]
#[test_case("dslx-g8r-stats", false, false; "linked_g8r")]
#[test_case("dslx-g8r-stats", true, false; "external_g8r")]
#[test_case("dslx-equiv", false, false; "linked_equiv")]
#[test_case("dslx-equiv", true, false; "external_equiv")]
#[test_case("dslx2pipeline-eco", true, false; "external_eco")]
fn type_inference_v2_conversion_ignores_legacy_input(command: &str, external: bool, in_toml: bool) {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("identity.x");
    std::fs::write(&source, "fn identity(x: u32) -> u32 { x }").unwrap();
    let baseline = dir.path().join("baseline.ir");
    if command == "dslx2pipeline-eco" {
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .arg("dslx2ir")
            .arg("--dslx_input_file")
            .arg(&source)
            .args(["--dslx_top", "identity"])
            .output()
            .unwrap();
        assert_success(&output);
        std::fs::write(&baseline, output.stdout).unwrap();
    }
    let config = dir.path().join("toolchain.toml");
    let toolchain = if external {
        format!(
            "[toolchain]\ntool_path = {:?}\n",
            std::env::var("XLSYNTH_TOOLS").unwrap()
        )
    } else {
        "[toolchain]\n".to_string()
    };
    let mut expected_stdout = None;
    for value in [None, Some("true"), Some("false"), Some("obsolete-value")] {
        let mut contents = toolchain.clone();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
        cmd.env_remove("RUST_LOG")
            .args(["--toolchain", config.to_str().unwrap(), command])
            .args(["--dslx_top", "identity"]);
        if command == "dslx-equiv" {
            // Disabling enum assumptions exercises actual external conversion
            // when a tool path is provided, rather than the linked fallback.
            cmd.arg(&source)
                .arg(&source)
                .arg("--assume-enum-in-bound=false");
        } else {
            cmd.arg("--dslx_input_file").arg(&source);
        }
        if command == "dslx2pipeline" || command == "dslx2pipeline-eco" {
            cmd.args(["--delay_model", "unit", "--pipeline_stages", "1"]);
        }
        if command == "dslx2pipeline-eco" {
            cmd.arg("--baseline_unopt_ir")
                .arg(&baseline)
                .args(["--flop_inputs=false", "--flop_outputs=false"]);
        }
        if let Some(value) = value {
            if in_toml {
                let toml_value = if value == "true" || value == "false" {
                    value.to_string()
                } else {
                    format!("{value:?}")
                };
                contents.push_str(&format!(
                    "[toolchain.dslx]\ntype_inference_v2 = {toml_value}\n"
                ));
            } else if value == "false" {
                cmd.args(["--type_inference_v2", value]);
            } else {
                cmd.arg(format!("--type_inference_v2={value}"));
            }
        }
        std::fs::write(&config, contents).unwrap();
        let output = cmd.output().unwrap();
        assert_success(&output);
        assert_compatibility_warning(&output, value.is_some());
        // Equivalence reports include elapsed time; compare the result lines.
        let stdout: Vec<_> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .filter(|line| !line.starts_with("[dslx-equiv] Time taken:"))
            .map(str::to_string)
            .collect();
        if let Some(expected) = &expected_stdout {
            assert_eq!(&stdout, expected);
        } else {
            expected_stdout = Some(stdout);
        }
    }
}

// Verifies: normal help omits the obsolete option on all five commands.
// Catches: advertising a selectable typechecker after selection was removed.
#[test]
fn type_inference_v2_is_hidden_from_help() {
    for command in [
        "dslx2ir",
        "dslx2pipeline",
        "dslx-g8r-stats",
        "dslx2pipeline-eco",
        "dslx-equiv",
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .args([command, "--help"])
            .output()
            .unwrap();
        assert_success(&output);
        assert!(!String::from_utf8_lossy(&output.stdout).contains("type_inference_v2"));
    }
}

// Verifies: saved JSON omits obsolete keys and child flags.
// Catches: forwarding ignored inputs from prover plans.
#[test]
fn type_inference_v2_json_is_not_serialized_or_forwarded() {
    for value in [
        serde_json::json!(true),
        serde_json::json!(false),
        serde_json::json!("obsolete-value"),
    ] {
        let json = serde_json::json!({
            "kind": "dslx-equiv", "lhs_dslx_file": "lhs.x", "rhs_dslx_file": "rhs.x",
            "dslx_top": "identity", "type_inference_v2": value
        });
        let plan: ProverPlan = serde_json::from_value(json).unwrap();
        assert!(
            !serde_json::to_string(&plan)
                .unwrap()
                .contains("type_inference_v2")
        );
        if let ProverPlan::Task { task, .. } = plan {
            assert!(
                !task
                    .to_command()
                    .get_args()
                    .any(|arg| arg == "--type_inference_v2")
            );
        } else {
            panic!("expected one equivalence task");
        }
    }
}

// Negative test: invalid live config values still produce errors.
#[test]
fn type_inference_v2_does_not_hide_other_config_errors() {
    assert!(
        serde_json::from_value::<ProverPlan>(serde_json::json!({
            "kind": "dslx-equiv", "lhs_dslx_file": "lhs.x", "rhs_dslx_file": "rhs.x",
            "type_inference_v2": false, "solver": "invalid-solver"
        }))
        .is_err()
    );
    assert!(
        toml::from_str::<xlsynth_driver::toolchain_config::ToolchainConfig>(
            "[dslx]\ntype_inference_v2 = false\nwarnings_as_errors = 'invalid-bool'"
        )
        .is_err()
    );
}

// Verifies: real plan loading warns on old keys and still runs the proof.
// Catches: silent handling or child failures caused by obsolete options.
#[test]
fn type_inference_v2_json_warns_in_the_prover_process() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("identity.x");
    std::fs::write(&source, "fn identity(x: u32) -> u32 { x }").unwrap();
    let config = dir.path().join("plan.json");
    for value in [None, Some(true), Some(false)] {
        let mut plan = serde_json::json!({
            "kind": "dslx-equiv", "lhs_dslx_file": source, "rhs_dslx_file": source,
            "dslx_top": "identity", "solver": "bitwuzla"
        });
        if let Some(value) = value {
            plan["type_inference_v2"] = value.into();
        }
        std::fs::write(&config, serde_json::to_vec(&plan).unwrap()).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .env_remove("RUST_LOG")
            .args(["prover", "--plan_json_file"])
            .arg(&config)
            .output()
            .unwrap();
        assert_success(&output);
        assert_compatibility_warning(&output, value.is_some());
    }
}
