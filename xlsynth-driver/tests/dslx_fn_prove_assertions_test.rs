// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

fn run_driver(dslx: &str, top: &str, solver: &str, extra_args: &[&str]) -> std::process::Output {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("sample.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let mut command = Command::new(driver);
    command
        .arg("dslx-fn-prove-assertions")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg(top)
        .arg("--solver")
        .arg(solver);
    for arg in extra_args {
        command.arg(arg);
    }
    command.output().unwrap()
}

#[test]
fn dslx_fn_prove_assertions_rejects_toolchain_solver() {
    let dslx = r#"fn main(x: u1) -> u1 { assert!(x == x, "self_eq"); x }"#;
    let output = run_driver(dslx, "main", "toolchain", &[]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("toolchain") && stderr.contains("not supported"),
        "stderr: {}",
        stderr
    );
}

#[cfg_attr(feature = "has-boolector", test_case::test_case("boolector"; "boolector"))]
#[cfg_attr(feature = "has-bitwuzla", test_case::test_case("bitwuzla"; "bitwuzla"))]
#[cfg_attr(feature = "with-z3-binary-test", test_case::test_case("z3-binary"; "z3_binary"))]
#[cfg_attr(feature = "with-bitwuzla-binary-test", test_case::test_case("bitwuzla-binary"; "bitwuzla_binary"))]
#[cfg_attr(feature = "with-boolector-binary-test", test_case::test_case("boolector-binary"; "boolector_binary"))]
#[allow(dead_code)]
fn dslx_fn_prove_assertions_success(solver: &str) {
    let dslx = r#"fn main(x: u32) -> u32 { assert!(x == x, "self_eq"); x }"#;
    let output = run_driver(dslx, "main", solver, &[]);
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("All selected assertions proved"),
        "stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[cfg_attr(feature = "has-boolector", test_case::test_case("boolector"; "boolector"))]
#[cfg_attr(feature = "has-bitwuzla", test_case::test_case("bitwuzla"; "bitwuzla"))]
#[cfg_attr(feature = "with-z3-binary-test", test_case::test_case("z3-binary"; "z3_binary"))]
#[cfg_attr(feature = "with-bitwuzla-binary-test", test_case::test_case("bitwuzla-binary"; "bitwuzla_binary"))]
#[cfg_attr(feature = "with-boolector-binary-test", test_case::test_case("boolector-binary"; "boolector_binary"))]
#[allow(dead_code)]
fn dslx_fn_prove_assertions_counterexample_json(solver: &str) {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("sample.x");
    let json_path = temp_dir.path().join("assertions.json");
    let dslx = r#"fn main(x: u32) -> u32 { assert!(x < u32:10, "x_lt_10"); x }"#;
    std::fs::write(&dslx_path, dslx).unwrap();

    let output = Command::new(driver)
        .arg("dslx-fn-prove-assertions")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .arg("--solver")
        .arg(solver)
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "expected assertion proof to fail; stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("x_lt_10"), "stderr: {}", stderr);

    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&json_path).unwrap()).unwrap();
    assert_eq!(json["success"].as_bool(), Some(false));
    assert_eq!(
        json["counterexample"]["output"]["assertion_label"].as_str(),
        Some("x_lt_10")
    );
    let inputs = json["counterexample"]["inputs"].as_array().unwrap();
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0]["name"].as_str(), Some("x"));
}

#[cfg_attr(feature = "has-boolector", test_case::test_case("boolector"; "boolector"))]
#[cfg_attr(feature = "has-bitwuzla", test_case::test_case("bitwuzla"; "bitwuzla"))]
#[cfg_attr(feature = "with-z3-binary-test", test_case::test_case("z3-binary"; "z3_binary"))]
#[cfg_attr(feature = "with-bitwuzla-binary-test", test_case::test_case("bitwuzla-binary"; "bitwuzla_binary"))]
#[cfg_attr(feature = "with-boolector-binary-test", test_case::test_case("boolector-binary"; "boolector_binary"))]
#[allow(dead_code)]
fn dslx_fn_prove_assertions_label_filter(solver: &str) {
    let dslx = r#"
fn main(x: u1) -> u1 {
  assert!(x == u1:1, "red");
  assert!(x == x, "blue");
  x
}
"#;
    let output = run_driver(dslx, "main", solver, &[]);
    assert!(!output.status.success());

    let output = run_driver(dslx, "main", solver, &["--assert-label-filter", "blue"]);
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg_attr(feature = "has-boolector", test_case::test_case("boolector"; "boolector"))]
#[cfg_attr(feature = "has-bitwuzla", test_case::test_case("bitwuzla"; "bitwuzla"))]
#[cfg_attr(feature = "with-z3-binary-test", test_case::test_case("z3-binary"; "z3_binary"))]
#[cfg_attr(feature = "with-bitwuzla-binary-test", test_case::test_case("bitwuzla-binary"; "bitwuzla_binary"))]
#[cfg_attr(feature = "with-boolector-binary-test", test_case::test_case("boolector-binary"; "boolector_binary"))]
#[allow(dead_code)]
fn dslx_fn_prove_assertions_helper_and_uf(solver: &str) {
    let dslx = r#"
fn helper(x: u1) -> u1 { assert!(x == u1:1, "helper_ok"); x }
fn main(x: u1) -> u1 { helper(x) }
"#;
    let output = run_driver(dslx, "main", solver, &[]);
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("helper_ok"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = run_driver(dslx, "main", solver, &["--uf", "helper:F"]);
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg_attr(feature = "has-boolector", test_case::test_case("boolector"; "boolector"))]
#[cfg_attr(feature = "has-bitwuzla", test_case::test_case("bitwuzla"; "bitwuzla"))]
#[cfg_attr(feature = "with-z3-binary-test", test_case::test_case("z3-binary"; "z3_binary"))]
#[cfg_attr(feature = "with-bitwuzla-binary-test", test_case::test_case("bitwuzla-binary"; "bitwuzla_binary"))]
#[cfg_attr(feature = "with-boolector-binary-test", test_case::test_case("boolector-binary"; "boolector_binary"))]
#[allow(dead_code)]
fn dslx_fn_prove_assertions_enum_domains(solver: &str) {
    let dslx = r#"
enum E : u2 {
  A = 0,
  B = 1,
}

fn main(e: E) -> u2 {
  let raw = e as u2;
  assert!(raw < u2:2, "declared_enum_value");
  raw
}
"#;
    let output = run_driver(dslx, "main", solver, &[]);
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let output = run_driver(dslx, "main", solver, &["--assume-enum-in-bound", "false"]);
    assert!(
        !output.status.success(),
        "expected invalid enum bit patterns to fail"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("declared_enum_value"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
