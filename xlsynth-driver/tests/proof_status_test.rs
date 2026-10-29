// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use serde_json::Value;

/// Runs the saved plan and checks both its structured result and the process
/// exit.
#[cfg(feature = "has-bitwuzla")]
fn assert_saved_plan_outcome(directory: &std::path::Path, plan: &Value, expected: &str) {
    std::fs::write(directory.join("plan.json"), plan.to_string()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(directory)
        .args([
            "prover",
            "--plan_json_file=plan.json",
            "--output_json=result.json",
        ])
        .output()
        .unwrap();
    let report: Value =
        serde_json::from_slice(&std::fs::read(directory.join("result.json")).unwrap()).unwrap();
    assert_eq!(report["status"], expected, "{plan}: {report}");
    assert_eq!(
        output.status.code(),
        Some(i32::from(expected != "proved")),
        "{plan}: {output:?}"
    );
}

/// Both workers must preserve the distinction between two helper assumptions
/// when reading a saved plan.
#[cfg(feature = "has-bitwuzla")]
#[test]
fn saved_plans_do_not_merge_distinct_uf_symbols() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("helpers.x"),
        r#"fn g(x: u8) -> u8 { x + x }
fn h(x: u8) -> u8 { x - x }
fn left(x: u8) -> u8 { g(x) }
fn right(x: u8) -> u8 { h(x) }
#[quickcheck] fn same(x: u8) -> bool { g(x) == h(x) }
"#,
    )
    .unwrap();
    for (symbol, expected) in [
        ("F", "proved"),
        ("G", "disproved"),
        ("F ", "execution-error"),
    ] {
        for plan in [
            serde_json::json!({
                "kind": "dslx-equiv", "solver": "bitwuzla",
                "lhs_dslx_file": "helpers.x", "rhs_dslx_file": "helpers.x",
                "lhs_dslx_top": "left", "rhs_dslx_top": "right",
                "lhs_uf": ["g:F"], "rhs_uf": [format!("h:{symbol}")],
            }),
            serde_json::json!({
                "kind": "prove-quickcheck", "solver": "bitwuzla",
                "dslx_input_file": "helpers.x", "uf": ["g:F", format!("h:{symbol}")],
            }),
        ] {
            assert_saved_plan_outcome(dir.path(), &plan, expected);
        }
    }
    for (specs, expected) in [
        (["g:F", "g:F"], "proved"),
        (["g:F ", "g:F"], "execution-error"),
        (["g:G", "g:F"], "execution-error"),
    ] {
        let plan = serde_json::json!({
            "kind": "dslx-equiv", "solver": "bitwuzla",
            "lhs_dslx_file": "helpers.x", "rhs_dslx_file": "helpers.x",
            "lhs_dslx_top": "left", "rhs_dslx_top": "right",
            "lhs_uf": specs, "rhs_uf": ["h:F"],
        });
        assert_saved_plan_outcome(dir.path(), &plan, expected);
    }
}

/// A saved plan must never drop two parameters when it names just one.
#[cfg(feature = "has-bitwuzla")]
#[test]
fn planned_equivalence_does_not_split_a_parameter_name() {
    let dir = tempfile::tempdir().unwrap();
    for (side, params) in [("lhs", ["a", "x"]), ("rhs", ["x", "b"])] {
        let [first, second] = params;
        std::fs::write(
            dir.path().join(format!("{side}.x")),
            format!("fn main({first}: u1, {second}: u1) -> u1 {{ x }}"),
        )
        .unwrap();
        std::fs::write(
            dir.path().join(format!("{side}.ir")),
            format!(
                "package p\ntop fn main({first}: bits[1], {second}: bits[1]) -> bits[1] {{ ret result: bits[1] = identity(x, id=3) }}"
            ),
        )
        .unwrap();
    }
    for mut plan in [
        serde_json::json!({"kind":"ir-equiv","lhs_ir_file":"lhs.ir","rhs_ir_file":"rhs.ir","top":"main"}),
        serde_json::json!({"kind":"dslx-equiv","lhs_dslx_file":"lhs.x","rhs_dslx_file":"rhs.x","dslx_top":"main"}),
    ] {
        plan["solver"] = "bitwuzla".into();
        for (names, expected) in [
            (vec![], "disproved"),
            (vec!["a", "b"], "proved"),
            (vec!["a,b"], "execution-error"),
        ] {
            plan["drop_params"] = serde_json::json!(names);
            assert_saved_plan_outcome(dir.path(), &plan, expected);
        }
    }
}

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

/// Discovery runs before the tactic script clock starts, even when it fails.
#[test]
fn tactic_preparation_error_wall_time_includes_quickcheck_discovery() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("module.x"),
        "fn identity(x: u8) -> u8 { x }",
    )
    .unwrap();
    std::fs::write(dir.path().join("script.json"), "[]").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(dir.path())
        .args([
            "prove-quickcheck",
            "--dslx_input_file=module.x",
            "--tactic_json=script.json",
            "--output_json=result.json",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let report: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("result.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "execution-error");
    assert_eq!(
        report["error_str"],
        "No matching quickcheck functions found"
    );
    let wall = report["timing"]["wall_micros"].as_u64().unwrap();
    let script_time = report["time_micros"].as_u64().unwrap();
    assert!(wall > script_time, "{report}");
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
        let timing = report.as_object_mut().unwrap().remove("timing").unwrap();
        assert!(timing["wall_micros"].is_u64());
        if assertions {
            expected["counterexample"] = Value::Null;
        }
        assert_eq!(report, expected);
    }
}
