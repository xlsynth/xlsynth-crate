// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use serde_json::Value;

/// All scheduler entry points must communicate conclusions independently of
/// exit codes.
#[cfg(feature = "has-bitwuzla")]
#[test]
fn scheduled_producers_report_proved_and_disproved_statuses() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("lhs.ir"),
        "package lhs\ntop fn main(x: bits[1] id=1) -> bits[1] { ret identity.2: bits[1] = identity(x, id=2) }",
    )
    .unwrap();
    std::fs::write(dir.path().join("lhs.x"), "fn main(x: u1) -> u1 { x }").unwrap();
    std::fs::write(
        dir.path().join("solve.json"),
        r#"[{"selector":["root"],"command":"Solve"}]"#,
    )
    .unwrap();
    for proved in [true, false] {
        let ir_op = if proved { "identity" } else { "not" };
        std::fs::write(
            dir.path().join("rhs.ir"),
            format!("package rhs\ntop fn main(x: bits[1] id=1) -> bits[1] {{ ret {ir_op}.2: bits[1] = {ir_op}(x, id=2) }}"),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("rhs.x"),
            if proved {
                "fn main(x: u1) -> u1 { x }"
            } else {
                "fn main(x: u1) -> u1 { !x }"
            },
        )
        .unwrap();
        let condition = if proved { "x == x" } else { "x == u1:0" };
        std::fs::write(
            dir.path().join("assertions.x"),
            format!("fn main(x: u1) -> u1 {{ assert!({condition}, \"property\"); x }}"),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("quickcheck.x"),
            format!("#[quickcheck] fn property(x: u1) -> bool {{ {condition} }}"),
        )
        .unwrap();
        let invocations: &[&[&str]] = &[
            &["ir-equiv", "lhs.ir", "rhs.ir", "--top=main"],
            &["dslx-equiv", "lhs.x", "rhs.x", "--dslx_top=main"],
            &[
                "dslx-equiv",
                "lhs.x",
                "rhs.x",
                "--dslx_top=main",
                "--tactic_json=solve.json",
            ],
            &[
                "dslx-fn-prove-assertions",
                "--dslx_input_file=assertions.x",
                "--dslx_top=main",
            ],
            &["prove-quickcheck", "--dslx_input_file=quickcheck.x"],
            &[
                "prove-quickcheck",
                "--dslx_input_file=quickcheck.x",
                "--tactic_json=solve.json",
            ],
        ];
        for args in invocations {
            let result_path = dir.path().join("result.json");
            // Prevent a missing report from being confused with the previous
            // invocation.
            if result_path.exists() {
                std::fs::remove_file(&result_path).unwrap();
            }
            let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
                .current_dir(dir.path())
                .args(*args)
                .args(["--solver=bitwuzla", "--output_json=result.json"])
                .output()
                .unwrap();
            assert_eq!(
                output.status.code(),
                Some(if proved { 0 } else { 1 }),
                "{args:?}: {}",
                String::from_utf8_lossy(&output.stderr),
            );
            let report: Value =
                serde_json::from_slice(&std::fs::read(result_path).unwrap()).unwrap();
            assert_eq!(
                report["status"],
                if proved { "proved" } else { "disproved" },
                "{args:?}"
            );
            assert_eq!(report["success"], proved, "{args:?}");
            if let Some(tests) = report["tests"].as_array() {
                assert_eq!(tests.len(), 1);
                assert_eq!(tests[0]["status"], report["status"]);
                assert_eq!(tests[0]["success"], proved);
                assert_eq!(tests[0]["error_str"], Value::Null);
                assert_eq!(tests[0]["counterexample"].is_null(), proved);
            }
        }
    }
}

/// A malformed request is an execution error, never a logical counterexample.
#[test]
fn invalid_proof_arguments_write_the_requested_report() {
    let invalid_uf =
        "invalid uninterpreted function specification 'typo'; expected <function-name>:<uf-name>";
    let cases: &[(&[&str], &str)] = &[
        (
            &["dslx-equiv", "a.x", "a.x", "--dslx_top=f", "--lhs_uf=typo"],
            invalid_uf,
        ),
        (
            &["prove-quickcheck", "--dslx_input_file=a.x", "--uf=typo"],
            invalid_uf,
        ),
        (
            &[
                "dslx-fn-prove-assertions",
                "--dslx_input_file=a.x",
                "--dslx_top=f",
                "--uf=typo",
            ],
            invalid_uf,
        ),
        (
            &["dslx-equiv", "a.x", "a.x"],
            "a top function must be specified via --dslx_top or both --lhs_dslx_top/--rhs_dslx_top",
        ),
        (
            &["dslx-equiv", "a.x", "a.x", "--lhs_dslx_top=f"],
            "--lhs_dslx_top and --rhs_dslx_top must be used together",
        ),
        (
            &["ir-equiv", "a.ir", "a.ir", "--top=f", "--lhs_ir_top=g"],
            "--ir_top and --lhs_ir_top/--rhs_ir_top cannot be used together",
        ),
    ];
    for (args, diagnostic) in cases {
        let dir = tempfile::tempdir().unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .current_dir(dir.path())
            .args(*args)
            .arg("--output_json=result.json")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{args:?}: {output:?}");
        let report: Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("result.json")).unwrap())
                .unwrap();
        assert_eq!(report["status"], "execution-error", "{args:?}");
        assert_eq!(report["success"], false, "{args:?}");
        assert_eq!(report["error_str"], *diagnostic, "{args:?}");
    }
}

/// A malformed IR package reaches the same worker protocol as setup errors.
#[cfg(feature = "has-bitwuzla")]
#[test]
fn ir_producer_reports_request_errors_with_typed_status() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("invalid.ir"), "not an IR package").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(dir.path())
        .args([
            "ir-equiv",
            "invalid.ir",
            "invalid.ir",
            "--top=main",
            "--solver=bitwuzla",
            "--output_json=result.json",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let report: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("result.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "execution-error");
    assert_eq!(report["success"], false);
    assert!(report["error_str"].is_string());
}

#[test]
fn unreadable_ir_and_assertion_inputs_write_execution_error_reports() {
    for (args, diagnostic, assertions) in [
        (
            &["ir-equiv", "missing", "missing", "--top=main"][..],
            "Failed to read lhs IR file",
            false,
        ),
        (
            &[
                "dslx-fn-prove-assertions",
                "--dslx_input_file=missing",
                "--dslx_top=main",
            ][..],
            "Failed to read DSLX file",
            true,
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let missing = std::fs::read(dir.path().join("missing")).unwrap_err();
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .current_dir(dir.path())
            .args(args)
            .arg("--output_json=result.json")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{args:?}");
        let mut report: Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("result.json")).unwrap())
                .unwrap();
        assert!(
            report
                .as_object_mut()
                .unwrap()
                .remove("time_micros")
                .unwrap()
                .is_u64()
        );
        let mut expected = serde_json::json!({
            "status": "execution-error",
            "success": false,
            "error_str": format!("{diagnostic}: {missing}"),
        });
        if assertions {
            expected["counterexample"] = Value::Null;
        }
        assert_eq!(report, expected);
    }
}
