// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;
use std::process::Command;
use test_case::test_case;

/// Both native proof paths must report invalid assertion filters to their
/// callers.
#[cfg(feature = "has-bitwuzla")]
#[test_case(&["dslx-equiv", "entry.x", "entry.x", "--dslx_top=main"]; "equivalence")]
#[test_case(&["dslx-fn-prove-assertions", "--dslx_input_file=entry.x", "--dslx_top=main"]; "assertions")]
fn invalid_assertion_filter_reports_execution_error(args: &[&str]) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("entry.x"), "pub fn main(x: u1) -> u1 { x }").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(dir.path())
        .args(args)
        .args([
            "--solver=bitwuzla",
            "--assert-label-filter=[",
            "--output_json=result.json",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let report: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("result.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "execution-error");
    assert_eq!(report["success"], false);
    assert_eq!(
        report["error_str"],
        "invalid regular expression in assert label filter: regex parse error:\n    [\n    ^\nerror: unclosed character class"
    );
}

/// Setup failures must reach the scheduler as execution errors, even when proof
/// never starts.
#[test_case("missing.x", false; "missing_direct")]
#[test_case("missing.x", true; "missing_tactic")]
#[test_case("valid.x", false; "unknown_top_direct")]
#[test_case("valid.x", true; "unknown_top_tactic")]
fn dslx_equiv_reports_input_errors(lhs: &str, tactic: bool) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("valid.x"), "pub fn main(x: u1) -> u1 { x }").unwrap();
    std::fs::write(
        dir.path().join("solve.json"),
        r#"[{"selector":["root"],"command":"Solve"}]"#,
    )
    .unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
    command.current_dir(dir.path()).args([
        "dslx-equiv",
        lhs,
        "valid.x",
        "--dslx_top=unknown",
        "--output_json=result.json",
    ]);
    if tactic {
        command.arg("--tactic_json=solve.json");
    }
    let output = command.output().unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let report: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("result.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "execution-error", "{report}");
    assert_eq!(report["success"], false, "{report}");
    if tactic && lhs == "valid.x" {
        // A root Solve ran a worker: its typed error must survive the process
        // boundary.
        assert_eq!(
            report["report"]["indefinite"][0]["reason"], "ExecutionError",
            "{report}"
        );
    } else {
        assert!(report["error_str"].is_string(), "{report}");
    }
}

/// Incompatible signatures must return a worker error, including through a root
/// Solve.
#[cfg(feature = "has-bitwuzla")]
#[test_case(false; "direct")]
#[test_case(true; "root_solve")]
fn incompatible_signatures_return_execution_error(tactic: bool) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("lhs.x"), "fn main(x: u8) -> u8 { x }").unwrap();
    std::fs::write(
        dir.path().join("rhs.x"),
        "fn main(x: u16) -> u8 { x as u8 }",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("solve.json"),
        r#"[{"selector":["root"],"command":"Solve"}]"#,
    )
    .unwrap();
    let result_path = dir.path().join("result.json");
    std::fs::write(&result_path, r#"{"status":"proved","success":true}"#).unwrap();
    let mut cli = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
    cli.current_dir(dir.path()).args([
        "dslx-equiv",
        "lhs.x",
        "rhs.x",
        "--dslx_top=main",
        "--solver=bitwuzla",
        "--output_json=result.json",
    ]);
    if tactic {
        cli.arg("--tactic_json=solve.json");
    }
    let output = cli.output().unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let report: Value = serde_json::from_slice(&std::fs::read(result_path).unwrap()).unwrap();
    assert_eq!(report["status"], "execution-error", "{report}");
    assert_eq!(report["success"], false);
    if tactic {
        assert_eq!(
            report["report"]["indefinite"][0]["reason"], "ExecutionError",
            "{report}"
        );
    }
}

/// A malformed script is a failed proof invocation, even if the DSLX inputs are
/// valid.
#[test]
fn malformed_tactic_script_writes_execution_error() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("input.x"), "pub fn main(x: u1) -> u1 { x }").unwrap();
    std::fs::write(dir.path().join("script.jsonl"), "not JSON").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(dir.path())
        .args([
            "dslx-equiv",
            "input.x",
            "input.x",
            "--dslx_top=main",
            "--tactic_jsonl=script.jsonl",
            "--output_json=result.json",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let report: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("result.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "execution-error");
    assert_eq!(report["success"], false);
    assert_eq!(
        report["error_str"],
        "tactic script (jsonl) parse error on line 1: expected ident at line 1 column 2"
    );
}
