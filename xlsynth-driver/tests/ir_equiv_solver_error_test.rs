// SPDX-License-Identifier: Apache-2.0

#![cfg(all(unix, feature = "has-easy-smt"))]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;

use serde_json::Value;
use test_case::test_case;

#[test_case("single-threaded", true; "missing_solver")]
#[test_case("output-bits", true; "missing_solver_in_output_bit_worker")]
#[test_case("input-bit-split", true; "missing_solver_in_input_bit_split")]
#[test_case("single-threaded", false; "failed_check")]
#[test_case("output-bits", false; "failed_check_in_output_bit_worker")]
#[test_case("input-bit-split", false; "failed_check_in_input_bit_split")]
fn solver_failure_replaces_prior_proof(strategy: &str, missing_solver: bool) {
    let dir = tempfile::tempdir().unwrap();
    let solver = dir.path().join("z3");
    fs::write(
        &solver,
        r#"#!/bin/sh
while IFS= read -r command; do
    case "$command" in
        '(check-sat)') echo "$CHECK_RESPONSE" ;;
        *) echo success ;;
    esac
done
"#,
    )
    .unwrap();
    fs::set_permissions(&solver, fs::Permissions::from_mode(0o755)).unwrap();
    fs::write(
        dir.path().join("main.ir"),
        r#"package p
top fn main(x: bits[2] id=1) -> bits[2] {
  ret result: bits[2] = identity(x, id=2)
}"#,
    )
    .unwrap();
    let mut driver = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
    driver
        .current_dir(dir.path())
        .env("PATH", dir.path())
        .env("CHECK_RESPONSE", "unsat")
        .args(["ir-equiv", "main.ir", "main.ir", "--solver=z3-binary"])
        .args([
            "--parallelism-strategy",
            strategy,
            "--output_json=result.json",
        ]);
    let output = driver.output().unwrap();
    assert!(output.status.success(), "{output:?}");
    let report = dir.path().join("result.json");
    let previous: Value = serde_json::from_slice(&fs::read(&report).unwrap()).unwrap();
    assert_eq!(previous["status"], "proved");

    let expected = if missing_solver {
        fs::remove_file(solver).unwrap();
        let missing = fs::metadata(dir.path().join("z3")).unwrap_err();
        format!("Failed to create equivalence solver: {missing}")
    } else {
        driver.env("CHECK_RESPONSE", "unexpected-response");
        "Equivalence solver failed: Unexpected result from solver: unexpected-response".to_string()
    };
    let output = driver.output().unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let actual: Value = serde_json::from_slice(&fs::read(report).unwrap()).unwrap();
    assert_eq!(actual["status"], "execution-error");
    assert_eq!(actual["success"], false);
    assert_eq!(actual["error_str"], expected);
}
