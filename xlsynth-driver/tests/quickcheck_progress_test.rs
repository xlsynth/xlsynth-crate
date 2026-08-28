// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "has-bitwuzla")]
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
        ]);
        if let Some(opt) = opt {
            command.arg(opt);
        }
        let output = command.output().unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert_eq!(output.stdout, b"");
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            include_str!("testdata/quickcheck/uf_opt_error.golden.txt")
        );
    }
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
        assert_eq!(
            String::from_utf8(output.stderr).unwrap(),
            include_str!("testdata/quickcheck/mixed.golden.txt"),
            "opt={opt:?}"
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
