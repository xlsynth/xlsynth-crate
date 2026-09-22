// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

fn ir_file(dir: &tempfile::TempDir, source: &str) -> std::path::PathBuf {
    let path = dir.path().join("design.ir");
    std::fs::write(&path, source).unwrap();
    path
}

const DESIGN: &str = r#"package gated

top fn main(x: bits[8] id=1, when_gate: bits[1] id=2) -> bits[8] {
  ret difference: bits[8] = xor(x, x, id=3)
}"#;

#[test]
fn invalid_slice_fails_before_starting_solver() {
    let dir = tempfile::tempdir().unwrap();
    let input = ir_file(&dir, DESIGN);
    let result = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .args(["ir-prove-operand-gate"])
        .arg(input)
        .args([
            "--when",
            "when_gate",
            "--consumer",
            "difference",
            "--operand",
            "0",
            "--start",
            "8",
            "--width",
            "1",
            "--clamp",
            "0",
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("invalid slice"));
}

#[cfg(feature = "has-bitwuzla")]
#[test]
fn multi_site_json_proves_with_deterministic_text() {
    let dir = tempfile::tempdir().unwrap();
    let input = ir_file(&dir, DESIGN);
    let sites = dir.path().join("sites.json");
    std::fs::write(
        &sites,
        r#"[
      {"consumer":"difference","operand":0,"clamp":"0"},
      {"consumer":"difference","operand":1,"clamp":"0"}
    ]"#,
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("ir-prove-operand-gate")
        .arg(input)
        .args(["--when", "when_gate", "--sites-json"])
        .arg(sites)
        .output()
        .unwrap();
    assert_eq!(
        result.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "ir-prove-operand-gate: proved (main)\npredicate reachability: reachable\nvacuous: false\n"
    );
}

#[cfg(feature = "has-bitwuzla")]
#[test]
fn counterexample_json_reports_inputs_and_distinct_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let input = ir_file(&dir, DESIGN);
    let result = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("ir-prove-operand-gate")
        .arg(input)
        .args([
            "--top",
            "main",
            "--when",
            "when_gate",
            "--consumer",
            "difference",
            "--operand",
            "0",
            "--clamp",
            "0",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert_eq!(
        result.status.code(),
        Some(1),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["status"], "counterexample");
    assert_eq!(value["predicate_reachability"], "reachable");
    assert_eq!(value["inputs"][1]["name"], "when_gate");
    assert_eq!(value["inputs"][1]["value"], "bits[1]:0x1");
    assert_ne!(value["original_output"], value["gated_output"]);
}
