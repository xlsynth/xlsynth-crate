// SPDX-License-Identifier: Apache-2.0

#[cfg(any(feature = "has-bitwuzla", feature = "with-z3-binary-test"))]
use serde_json::Value;
use std::process::Command;

/// The configuration error is reported before attempting to use any solver.
#[test]
fn quickcheck_uf_requires_explicit_opt_false() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("qc.x"), "").unwrap();
    for opt in [None, Some("--opt=true")] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
        command.current_dir(dir.path()).env("NO_COLOR", "1").args([
            "prove-quickcheck",
            "--dslx_input_file=qc.x",
            "--uf=g:F",
            "--output_json=result.json",
        ]);
        if let Some(opt) = opt {
            command.arg(opt);
        }
        let output = command.output().unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert_eq!(output.stdout, b"");
        let result: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("result.json")).unwrap())
                .unwrap();
        assert_eq!(result["status"], "execution-error");
        assert_eq!(result["success"], false);
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            include_str!("testdata/quickcheck/uf_opt_error.golden.txt")
        );
    }
}

/// A UF alias containing SMT commands must not turn a false property into a
/// proof, even when it includes the responses expected by the SMT client.
#[cfg(feature = "with-z3-binary-test")]
#[test]
fn quickcheck_uf_alias_cannot_inject_solver_commands() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("qc.x"),
        r#"fn helper(x: u1) -> u1 { x }
#[quickcheck] fn p() -> bool { false }
"#,
    )
    .unwrap();
    let alias = r#"F ((_ BitVec 1)) (_ BitVec 1)) (set-option :print-success false) (assert false) (echo "success") (echo "success") (echo "success") (echo "success") ;"#;
    let output = xlsynth::external_tool::run_with_timeout(
        Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .current_dir(dir.path())
            .args([
                "prove-quickcheck",
                "--dslx_input_file=qc.x",
                "--opt=false",
                "--solver=z3-binary",
                "--output_json=result.json",
            ])
            .arg(format!("--uf=helper:{alias}")),
        dir.path(),
        "quickcheck",
        std::time::Duration::from_secs(10),
    )
    .unwrap();
    let report: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("result.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "disproved", "{output:?}");
    assert_eq!(output.status.code(), Some(1));
}

/// Default/explicit optimization and unoptimized proofs have the same readable
/// output and JSON structure, including continuing to a property after failure.
#[cfg(feature = "has-bitwuzla")]
#[test]
fn quickcheck_progress_and_counterexample_goldens() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("qc.x"),
        r#"
#[quickcheck] fn passing(x: u8) -> bool { x == x }
#[quickcheck] fn zero_is_forbidden(a: u1, b: u2) -> bool {
    a != u1:0 || b != u2:0
}
#[quickcheck] fn passing_after_failure(x: u8) -> bool { x + u8:0 == x }
"#,
    )
    .unwrap();
    for opt in [None, Some("--opt=true"), Some("--opt=false")] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
        command.current_dir(dir.path()).env("NO_COLOR", "1").args([
            "prove-quickcheck",
            "--dslx_input_file=qc.x",
            "--solver=bitwuzla",
            "--output_json=result.json",
        ]);
        if let Some(opt) = opt {
            command.arg(opt);
        }
        let output = command.output().unwrap();
        assert_eq!(output.status.code(), Some(1));
        xlsynth_test_helpers::compare_golden_text(
            &String::from_utf8(output.stderr).unwrap(),
            "tests/testdata/quickcheck/mixed.golden.txt",
        );
        assert_eq!(
            output.stdout,
            b"Failure: Some QuickChecks did not succeed\n"
        );
        let report: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("result.json")).unwrap())
                .unwrap();
        assert_eq!(report["success"], false);
        let tests = report["tests"].as_array().unwrap();
        assert_eq!(tests.len(), 3);
        for (test, (name, success)) in tests.iter().zip([
            ("passing", true),
            ("zero_is_forbidden", false),
            ("passing_after_failure", true),
        ]) {
            assert_eq!(test["name"], name);
            assert_eq!(test["success"], success);
            assert!(test["time_micros"].is_u64());
            if success {
                assert!(test["counterexample"].is_null());
            }
        }
        assert_eq!(
            tests[1]["counterexample"],
            "inputs: [FnInput { name: \"a\", value: bits[1]:0 }, FnInput { name: \"b\", value: bits[2]:0 }], output: FnOutput { value: bits[1]:0, assertion_violation: None }"
        );
    }
}

#[cfg(feature = "has-bitwuzla")]
#[test]
fn quickcheck_uf_with_opt_false_preserves_abstraction() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("qc.x"),
        r#"
fn g(x: u8) -> u8 { x + u8:1 }
fn h(x: u8) -> u8 { x + u8:2 }
#[quickcheck] fn abstract_equivalence(x: u8) -> bool { g(x) == h(x) }
"#,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(dir.path())
        .env("NO_COLOR", "1")
        .args([
            "prove-quickcheck",
            "--dslx_input_file=qc.x",
            "--solver=bitwuzla",
            "--opt=false",
            "--uf=g:F",
            "--uf=h:F",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stderr, b"[ RUN QUICKCHECK        ] abstract_equivalence\n[                    OK ] abstract_equivalence\n");
    assert_eq!(output.stdout, b"Success: All QuickChecks proved\n");
}

/// Invalid input and filters produce actionable diagnostics before any RUN
/// boundary, never a Rust panic or a spurious failed property.
#[cfg(feature = "has-bitwuzla")]
#[test]
fn quickcheck_preparation_cli_goldens() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("qc.x"),
        "#[quickcheck] fn check(x: u8) -> bool { true }",
    )
    .unwrap();
    for (input, filter, golden) in [
        ("missing.x", None, "missing_file"),
        ("qc.x", Some("["), "bad_filter"),
    ] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
        command.current_dir(dir.path()).env("NO_COLOR", "1").args([
            "prove-quickcheck",
            "--dslx_input_file",
            input,
            "--solver=bitwuzla",
        ]);
        if let Some(filter) = filter {
            command.args(["--test_filter", filter]);
        }
        let output = command.output().unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert_eq!(output.stdout, b"");
        // OS error numbers/text differ by platform. Keep that detail covered
        // by the typed library error test; normalize it for the CLI golden.
        let stderr = String::from_utf8(output.stderr).unwrap();
        let stderr = if golden == "missing_file" {
            let prefix = "Failed to read DSLX input file: ";
            let start = stderr.find(prefix).unwrap() + prefix.len();
            format!("{}<I/O error>\n", &stderr[..start])
        } else {
            stderr
        };
        xlsynth_test_helpers::compare_golden_text(
            &stderr,
            &format!("tests/testdata/quickcheck/{golden}.golden.txt"),
        );
    }
}

/// Running the root obligation in a worker must prove the same property as a
/// direct run.
#[cfg(all(unix, feature = "has-bitwuzla"))]
#[test]
fn quickcheck_tactic_workers_preserve_assertions_and_uf_assumptions() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("qc.x"),
        r#"
fn g(x: u8) -> u8 { x + u8:1 }
fn h(x: u8) -> u8 { x + u8:2 }
#[quickcheck] fn assertions(x: u8) -> bool {
    if x == u8:0 { fail!("reject_zero", true) } else { true }
}
#[quickcheck] fn abstract_eq(x: u8) -> bool { g(x) == h(x) }
"#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("solve.json"),
        r#"[{"selector":["root"],"command":"Solve"}]"#,
    )
    .unwrap();

    for (case, (property, flags, expected)) in [
        ("assertions", vec![], "disproved"),
        ("assertions", vec!["--assertion-semantics=ignore"], "proved"),
        (
            "assertions",
            vec!["--assert-label-filter=other_label"],
            "proved",
        ),
        ("abstract_eq", vec!["--uf=g:F", "--uf=h:F"], "proved"),
    ]
    .into_iter()
    .enumerate()
    {
        for scripted in [false, true] {
            let result_path = dir
                .path()
                .join(format!("case_{case}_scripted_{scripted}.json"));
            let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
            command
                .current_dir(dir.path())
                .args([
                    "prove-quickcheck",
                    "--dslx_input_file=qc.x",
                    "--solver=bitwuzla",
                    "--test_filter",
                    property,
                ])
                .args(&flags)
                .arg("--output_json")
                .arg(&result_path);
            if scripted {
                command.arg("--tactic_json=solve.json");
            } else if property == "abstract_eq" {
                command.arg("--opt=false");
            }
            let output = command.output().unwrap();
            let report: Value =
                serde_json::from_slice(&std::fs::read(result_path).unwrap()).unwrap();
            assert_eq!(report["status"], expected, "{command:?}: {report}");
            assert_eq!(
                output.status.success(),
                expected == "proved",
                "{command:?}: {report}"
            );
        }
    }
}
