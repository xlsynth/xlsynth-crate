// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::Path;
use std::process::Command;

use serde_json::Value;
use test_case::test_case;

/// Runs the proof command with a fresh result file and checks its logical and
/// process outcomes.
fn assert_proof_status(dir: &Path, args: &[&str], solver: &str, expected: &str) {
    let result = tempfile::NamedTempFile::new_in(dir).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(dir)
        .args(args)
        .args([
            "--solver",
            solver,
            "--assertion-semantics=ignore",
            "--output_json",
        ])
        .arg(result.path())
        .output()
        .unwrap();
    let report: Value = serde_json::from_slice(&fs::read(result.path()).unwrap())
        .unwrap_or_else(|err| panic!("missing proof result ({err}): {output:?}"));
    let proved = expected == "proved";
    assert_eq!(report["status"], expected, "{solver} {args:?}: {report}");
    assert_eq!(report["success"], proved, "{report}");
    assert_eq!(
        output.status.code(),
        Some(if proved { 0 } else { 1 }),
        "{output:?}"
    );
}

/// Both functions return their second input before dropping a. Afterwards the
/// left returns its first remaining input and the right returns its second.
#[test_case(&["ir-equiv", "lhs.ir", "rhs.ir"]; "ir")]
#[test_case(&["dslx-equiv", "lhs.x", "rhs.x", "--dslx_top=main"]; "dslx")]
#[test_case(&["dslx-equiv", "lhs.x", "rhs.x", "--dslx_top=main", "--tactic_json=solve.json"]; "dslx_root_solve")]
fn external_checker_uses_the_requested_parameter_list(args: &[&str]) {
    let dir = tempfile::tempdir().unwrap();
    for (side, names) in [("lhs", ["a", "x", "z"]), ("rhs", ["c", "x", "a"])] {
        let [first, second, third] = names;
        fs::write(
            dir.path().join(format!("{side}.x")),
            format!("pub fn main({first}: u1, {second}: u1, {third}: u1) -> u1 {{ x }}"),
        )
        .unwrap();
        fs::write(
            dir.path().join(format!("{side}.ir")),
            format!(
                r#"package {side}
top fn main({first}: bits[1], {second}: bits[1], {third}: bits[1]) -> bits[1] {{
  ret y: bits[1] = identity(x, id=4)
}}
"#
            ),
        )
        .unwrap();
    }
    fs::write(
        dir.path().join("solve.json"),
        r#"[{"selector":["root"],"command":"Solve"}]"#,
    )
    .unwrap();

    assert_proof_status(dir.path(), args, "toolchain", "proved");
    let mut dropped = args.to_vec();
    dropped.push("--drop_params=a");
    assert_proof_status(dir.path(), &dropped, "toolchain", "execution-error");
    #[cfg(feature = "has-bitwuzla")]
    assert_proof_status(dir.path(), &dropped, "bitwuzla", "disproved");

    // A supported parameter drop can still be proved when both sides agree.
    dropped[2] = dropped[1];
    assert_proof_status(dir.path(), &dropped, "toolchain", "proved");
}

/// Selecting two differently named tops must preserve opcodes with the same
/// names as those functions.
#[test_case("add", "proved"; "same_operation")]
#[test_case("sub", "execution-error"; "different_operation")]
fn external_checker_preserves_opcodes_when_selecting_tops(rhs_op: &str, expected: &str) {
    let dir = tempfile::tempdir().unwrap();
    for (file, top, op) in [("lhs.ir", "add", "add"), ("rhs.ir", "sub", rhs_op)] {
        fs::write(
            dir.path().join(file),
            format!(
                r#"package source
top fn {top}(x: bits[4], y: bits[4]) -> bits[4] {{
  ret result: bits[4] = {op}(x, y, id=3)
}}
"#
            ),
        )
        .unwrap();
    }
    assert_proof_status(
        dir.path(),
        &[
            "ir-equiv",
            "lhs.ir",
            "rhs.ir",
            "--lhs_ir_top=add",
            "--rhs_ir_top=sub",
        ],
        "toolchain",
        expected,
    );
}
