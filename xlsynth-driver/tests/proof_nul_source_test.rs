// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;
use std::process::Command;
use test_case::test_case;

/// A NUL after a valid module must not truncate the property or crash the proof
/// worker.
#[test_case(&["dslx-equiv", "input.x", "input.x", "--dslx_top=main"]; "equivalence")]
#[test_case(&["dslx-fn-prove-assertions", "--dslx_input_file=input.x", "--dslx_top=main"]; "assertions")]
#[test_case(&["prove-quickcheck", "--dslx_input_file=input.x", "--solver=toolchain"]; "quickcheck")]
fn nul_in_dslx_source_is_an_execution_error(args: &[&str]) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("input.x"),
        b"pub fn main(x: u8) -> u8 { x }\0#[quickcheck] fn prop(x: u8) -> bool { false }",
    )
    .unwrap();
    // QuickCheck resolves the solver executable before discovery, but malformed
    // input must be rejected before any tool is invoked.
    std::fs::write(dir.path().join("prove_quickcheck_main"), "").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(dir.path())
        .env("XLSYNTH_TOOLS", dir.path())
        .args(args)
        .arg("--output_json=result.json")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let report: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("result.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "execution-error", "{report}");
    assert_eq!(report["success"], false, "{report}");
    assert!(
        report["error_str"]
            .as_str()
            .unwrap()
            .ends_with("DSLX source contains a NUL byte at offset 30"),
        "{report}"
    );
}

#[test]
fn nul_in_configured_warning_is_an_execution_error() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("input.x"), "pub fn main(x: u8) -> u8 { x }").unwrap();
    std::fs::write(
        dir.path().join("tools.toml"),
        r#"[toolchain.dslx]
enable_warnings = ["bad\u0000warning"]
"#,
    )
    .unwrap();
    let result_path = dir.path().join("result.json");
    std::fs::write(&result_path, r#"{"status":"proved","success":true}"#).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(dir.path())
        .args([
            "--toolchain=tools.toml",
            "dslx-equiv",
            "input.x",
            "input.x",
            "--dslx_top=main",
            "--output_json=result.json",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let report: Value = serde_json::from_slice(&std::fs::read(result_path).unwrap()).unwrap();
    assert_eq!(report["status"], "execution-error", "{report}");
    assert_eq!(report["success"], false, "{report}");
    assert_eq!(
        report["error_str"], "xlsynth error: enable_warnings[0] contains a NUL byte at offset 3",
        "{report}"
    );
}
