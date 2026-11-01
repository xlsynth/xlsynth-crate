// SPDX-License-Identifier: Apache-2.0

#![cfg(all(unix, feature = "has-easy-smt"))]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use test_case::test_case;
use xlsynth_prover::proof_phases::ProofPhases;

/// The stage at which the second solver invocation fails.
enum Failure {
    MissingSolver,
    DefineFun,
    CheckSat,
    GetValueResponse(&'static str),
    ModelValue(&'static str),
}

/// Installs a fake Z3 whose declarations, checks and model replies are
/// independently controlled by the test.
fn write_model_solver(dir: &Path) -> std::io::Result<PathBuf> {
    let solver = dir.join("z3");
    fs::write(
        &solver,
        r#"#!/bin/sh
while IFS= read -r command; do
    case "$command" in
        '(define-fun '*) echo "$DEFINE_RESPONSE" ;;
        '(check-sat)') echo "$CHECK_RESPONSE" ;;
        '(get-value ('*'))')
            if [ -n "$MODEL_VALUE" ]; then
                term=${command#'(get-value ('}
                term=${term%'))'}
                echo "(($term $MODEL_VALUE))"
            else
                echo "$GET_VALUE_RESPONSE"
            fi ;;
        *) echo success ;;
    esac
done
"#,
    )?;
    fs::set_permissions(&solver, fs::Permissions::from_mode(0o755))?;
    Ok(solver)
}

/// Checks which stages of the fake SMT protocol the reported invocation
/// reached.
fn assert_query_phases(report: &Value, encoded: bool, check_count: u64, read_model: bool) {
    let phases: ProofPhases = serde_json::from_value(report["timing"]["phases"].clone()).unwrap();
    assert!(phases.prepare_work_micros.is_some(), "{report}");
    assert_eq!(phases.smt_encode_work_micros.is_some(), encoded, "{report}");
    assert_eq!(phases.solver_check_count, check_count, "{report}");
    assert_eq!(
        phases.solver_check_work_micros.is_some(),
        check_count != 0,
        "{report}"
    );
    assert_eq!(
        phases.counterexample_read_work_micros.is_some(),
        read_model,
        "{report}"
    );
    assert_eq!(phases.external_tool_work_micros, None, "{report}");
}

#[test_case("single-threaded", Failure::MissingSolver; "missing_solver")]
#[test_case("output-bits", Failure::MissingSolver; "missing_solver_in_output_bit_worker")]
#[test_case("input-bit-split", Failure::MissingSolver; "missing_solver_in_input_bit_split")]
#[test_case("single-threaded", Failure::CheckSat; "failed_check")]
#[test_case("output-bits", Failure::CheckSat; "failed_check_in_output_bit_worker")]
#[test_case("input-bit-split", Failure::CheckSat; "failed_check_in_input_bit_split")]
#[test_case("single-threaded", Failure::DefineFun; "failed_define_fun")]
#[test_case("single-threaded", Failure::GetValueResponse(r#"(error "model unavailable")"#); "failed_get_value")]
#[test_case("single-threaded", Failure::ModelValue("#b0Z"); "invalid_binary_digit")]
#[test_case("single-threaded", Failure::GetValueResponse("((other #b01))"); "wrong_model_term")]
fn solver_failure_replaces_prior_proof(strategy: &str, failure: Failure) {
    let dir = tempfile::tempdir().unwrap();
    let solver = write_model_solver(dir.path()).unwrap();
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
        .env("MODEL_VALUE", "")
        .env("DEFINE_RESPONSE", "success")
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
    // Both bits of the output, or both values of the split input, must prove.
    let partitions = if strategy == "single-threaded" { 1 } else { 2 };
    assert_query_phases(&previous, true, partitions, false);

    let expected = match failure {
        Failure::MissingSolver => {
            fs::remove_file(solver).unwrap();
            let missing = fs::metadata(dir.path().join("z3")).unwrap_err();
            format!("Failed to create equivalence solver: {missing}")
        }
        Failure::CheckSat => {
            driver.env("CHECK_RESPONSE", "unexpected-response");
            "Equivalence solver failed: Unexpected result from solver: unexpected-response".into()
        }
        Failure::DefineFun => {
            driver.env("DEFINE_RESPONSE", r#"(error "definition rejected")"#);
            r#"(error "definition rejected")"#.into()
        }
        Failure::GetValueResponse(response) => {
            driver
                .env("CHECK_RESPONSE", "sat")
                .env("GET_VALUE_RESPONSE", response);
            response.into()
        }
        Failure::ModelValue(value) => {
            driver
                .env("CHECK_RESPONSE", "sat")
                .env("MODEL_VALUE", value);
            format!(" {value}))")
        }
    };
    let output = driver.output().unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let actual: Value = serde_json::from_slice(&fs::read(report).unwrap()).unwrap();
    assert_eq!(actual["status"], "execution-error");
    assert_eq!(actual["success"], false);
    match failure {
        Failure::MissingSolver => assert_query_phases(&actual, false, 0, false),
        Failure::DefineFun => assert_query_phases(&actual, true, 0, false),
        Failure::CheckSat => assert_query_phases(&actual, true, partitions, false),
        Failure::GetValueResponse(_) | Failure::ModelValue(_) => {
            assert_query_phases(&actual, true, 1, true);
        }
    }
    let diagnostic = actual["error_str"].as_str().unwrap();
    match failure {
        Failure::MissingSolver | Failure::CheckSat => assert_eq!(diagnostic, expected),
        Failure::DefineFun | Failure::GetValueResponse(_) | Failure::ModelValue(_) => {
            assert!(diagnostic.ends_with(&expected), "{diagnostic}");
        }
    }
}

/// A satisfiable property query can fail while reading the witness.
#[test]
fn quickcheck_reports_model_failure_for_the_property() {
    let dir = tempfile::tempdir().unwrap();
    write_model_solver(dir.path()).unwrap();
    fs::write(
        dir.path().join("qc.x"),
        "#[quickcheck] fn p(x: u2) -> bool { x == u2:0 }",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(dir.path())
        .env("PATH", dir.path())
        .env("DEFINE_RESPONSE", "success")
        .env("CHECK_RESPONSE", "sat")
        .env("MODEL_VALUE", "")
        .env("GET_VALUE_RESPONSE", r#"(error "model unavailable")"#)
        .args([
            "prove-quickcheck",
            "--dslx_input_file=qc.x",
            "--solver=z3-binary",
            "--output_json=result.json",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let report: Value =
        serde_json::from_slice(&fs::read(dir.path().join("result.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "execution-error", "{report}");
    let test = &report["tests"][0];
    assert_eq!(test["name"], "p");
    assert_eq!(test["status"], "execution-error");
    assert_eq!(test["success"], false);
    assert_eq!(test["counterexample"], Value::Null);
    assert_query_phases(test, true, 1, true);
    assert_query_phases(&report, true, 1, true);
    assert!(
        test["error_str"]
            .as_str()
            .unwrap()
            .ends_with(r#"(error "model unavailable")"#),
        "{report}"
    );
}

/// A witness at x=1 disproves equivalence even if the solver could not decide
/// x=0.
#[test_case("unknown"; "unknown_on_zero")]
#[test_case("unexpected-response"; "solver_error_on_zero")]
fn input_split_counterexample_survives_failed_partition(first_response: &str) {
    let dir = tempfile::tempdir().unwrap();
    let solver = dir.path().join("z3");
    fs::write(
        &solver,
        r#"#!/bin/sh
while IFS= read -r command; do
    case "$command" in
        '(check-sat)')
            if [ -f checked-zero ]; then
                echo sat
            else
                : > checked-zero
                echo "$FIRST_RESPONSE"
            fi ;;
        '(get-value ('*'))')
            term=${command#'(get-value ('}
            term=${term%'))'}
            case "$term" in '#b0') value='#b0' ;; *) value='#b1' ;; esac
            echo "(($term $value))" ;;
        *) echo success ;;
    esac
done
"#,
    )
    .unwrap();
    fs::set_permissions(&solver, fs::Permissions::from_mode(0o755)).unwrap();
    for (file, result) in [
        ("lhs.ir", "identity(x, id=2)"),
        ("rhs.ir", "literal(value=0, id=2)"),
    ] {
        let ir = format!(
            r#"package p
top fn main(x: bits[1] id=1) -> bits[1] {{
  ret y: bits[1] = {result}
}}
"#
        );
        fs::write(dir.path().join(file), ir).unwrap();
    }
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(dir.path())
        .env("PATH", dir.path())
        .env("FIRST_RESPONSE", first_response)
        .args([
            "ir-equiv",
            "lhs.ir",
            "rhs.ir",
            "--solver=z3-binary",
            "--parallelism-strategy=input-bit-split",
            "--output_json=result.json",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let report: Value =
        serde_json::from_slice(&fs::read(dir.path().join("result.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "disproved", "{report}");
    assert_eq!(report["success"], false);
    // The failed or unknown check for x=0 and the SAT check for x=1 both count.
    assert_query_phases(&report, true, 2, true);
}
