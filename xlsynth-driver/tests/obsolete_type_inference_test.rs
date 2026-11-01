// SPDX-License-Identifier: Apache-2.0

//! Removed typechecker selection is rejected before conversion or proving.

use std::process::{Command, Output};
use test_case::test_case;
use xlsynth_driver::prover_config::{ProverPlan, ToDriverCommand};
use xlsynth_driver::toolchain_config::ToolchainConfig;

const REMOVED_ERROR: &str =
    "type_inference_v2 has been removed; V2 is always used; remove the option entirely";

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Rejection must explain the removed input, rather than fail later or panic.
fn assert_rejected(output: &Output, message: &str) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "unexpected success: {output:?}");
    assert!(stderr.contains(message), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(output.stdout.is_empty(), "{output:?}");
}

/// Every former CLI spelling is an unknown option, regardless of its value.
#[test_case("dslx2ir")]
#[test_case("dslx2pipeline")]
#[test_case("dslx-g8r-stats")]
#[test_case("dslx2pipeline-eco")]
#[test_case("dslx-equiv")]
fn type_inference_v2_cli_is_rejected(command: &str) {
    let dir = tempfile::tempdir().unwrap();
    for value in ["true", "false", "obsolete-value"] {
        for split in [false, true] {
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
            cmd.current_dir(dir.path()).arg(command);
            if split {
                cmd.args(["--type_inference_v2", value]);
            } else {
                cmd.arg(format!("--type_inference_v2={value}"));
            }
            let output = cmd.output().unwrap();
            assert_eq!(output.status.code(), Some(2));
            assert!(output.stdout.is_empty(), "{output:?}");
            assert_eq!(
                String::from_utf8_lossy(&output.stderr).lines().next(),
                Some("error: unexpected argument '--type_inference_v2' found"),
                "{output:?}"
            );
        }
    }
}

/// Present TOML keys fail even when their values have an unexpected type.
#[test_case("true"; "true")]
#[test_case("false"; "false")]
#[test_case("'obsolete-value'"; "string")]
#[test_case("0"; "integer")]
#[test_case("1.5"; "float")]
#[test_case("[]"; "array")]
#[test_case("{}"; "table")]
#[test_case("2026-10-27"; "date")]
fn type_inference_v2_toml_rejects_before_conversion(value: &str) {
    let body = format!("[dslx]\ntype_inference_v2 = {value}\n");
    let error = toml::from_str::<ToolchainConfig>(&body).unwrap_err();
    assert_eq!(error.message(), REMOVED_ERROR);
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("toolchain.toml");
    std::fs::write(
        &config,
        format!("[toolchain.dslx]\ntype_inference_v2 = {value}\n"),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(dir.path())
        .arg("--toolchain")
        .arg(&config)
        .args([
            "dslx2ir",
            "--dslx_input_file",
            "missing.x",
            "--dslx_top",
            "identity",
        ])
        .output()
        .unwrap();
    assert_rejected(&output, REMOVED_ERROR);
}

/// Omitting the removed key keeps ordinary TOML conversion working.
#[test]
fn type_inference_v2_absent_config_preserves_conversion() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("identity.x");
    std::fs::write(&source, "fn identity(x: u32) -> u32 { x }").unwrap();
    let config = dir.path().join("explicit-toolchain.toml");
    std::fs::write(&config, "[toolchain.dslx]\nwarnings_as_errors = true\n").unwrap();
    let mut outputs = Vec::new();
    for with_config in [false, true] {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
        cmd.current_dir(dir.path());
        if with_config {
            cmd.arg("--toolchain").arg(&config);
        }
        let output = cmd
            .args(["dslx2ir", "--dslx_input_file"])
            .arg(&source)
            .args(["--dslx_top", "identity"])
            .output()
            .unwrap();
        assert_success(&output);
        assert!(!output.stdout.is_empty());
        outputs.push(output.stdout);
    }
    assert_eq!(outputs[0], outputs[1]);
}

/// No command advertises the removed typechecker selection.
#[test]
fn type_inference_v2_is_absent_from_help() {
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

/// Valid plans neither serialize the removed key nor forward its child flag.
#[test]
fn type_inference_v2_json_is_not_serialized_or_forwarded() {
    let plan: ProverPlan = serde_json::from_value(serde_json::json!({
        "kind": "dslx-equiv", "lhs_dslx_file": "lhs.x", "rhs_dslx_file": "rhs.x", "dslx_top": "identity"
    })).unwrap();
    assert!(
        serde_json::to_value(&plan)
            .unwrap()
            .get("type_inference_v2")
            .is_none()
    );
    if let ProverPlan::Task { task, .. } = plan {
        assert!(
            !task
                .to_command()
                .unwrap()
                .get_args()
                .any(|arg| arg.to_string_lossy().starts_with("--type_inference_v2"))
        );
    } else {
        panic!("expected one equivalence task");
    }
}

/// Removing compatibility handling does not weaken validation of live fields.
#[test]
fn type_inference_v2_absent_config_still_rejects_other_errors() {
    assert!(serde_json::from_value::<ProverPlan>(serde_json::json!({
        "kind": "dslx-equiv", "lhs_dslx_file": "lhs.x", "rhs_dslx_file": "rhs.x", "solver": "invalid-solver"
    })).is_err());
    assert!(
        toml::from_str::<ToolchainConfig>("[dslx]\nwarnings_as_errors = 'invalid-bool'").is_err()
    );
}

/// Every present JSON value fails in direct and nested plans before proving.
#[test_case("true"; "true")]
#[test_case("false"; "false")]
#[test_case(r#""obsolete-value""#; "string")]
#[test_case("null"; "null")]
#[test_case("0"; "number")]
#[test_case("[]"; "array")]
#[test_case("{}"; "object")]
#[test_case(r#"true, "type_inference_v2": false"#; "repeated_key")]
fn type_inference_v2_json_rejects_before_proving(value: &str) {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("plan.json");
    let task = format!(
        r#"{{"kind":"dslx-equiv","lhs_dslx_file":"missing.x","rhs_dslx_file":"missing.x","type_inference_v2":{value}}}"#
    );
    for plan in [
        task.clone(),
        format!(r#"{{"kind":"all","tasks":[{task}]}}"#),
    ] {
        let error = serde_json::from_str::<ProverPlan>(&plan).unwrap_err();
        assert!(error.to_string().contains(REMOVED_ERROR), "{error}");
        std::fs::write(&config, plan).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .current_dir(dir.path())
            .args(["prover", "--plan_json_file"])
            .arg(&config)
            .output()
            .unwrap();
        assert_rejected(&output, REMOVED_ERROR);
    }
}

/// Omitting the removed key permits a real equivalence proof.
#[cfg(feature = "has-bitwuzla")]
#[test]
fn type_inference_v2_absent_plan_runs_proof() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("identity.x");
    std::fs::write(&source, "fn identity(x: u32) -> u32 { x }").unwrap();
    let config = dir.path().join("plan.json");
    let plan = serde_json::json!({
        "kind": "dslx-equiv", "lhs_dslx_file": source, "rhs_dslx_file": source,
        "dslx_top": "identity", "solver": "bitwuzla"
    });
    std::fs::write(&config, serde_json::to_vec(&plan).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(dir.path())
        .args(["prover", "--plan_json_file"])
        .arg(&config)
        .output()
        .unwrap();
    assert_success(&output);
}
