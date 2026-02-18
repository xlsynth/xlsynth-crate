// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn test_dslx_list_fns_jsonl_concrete_and_parametric() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    let tmp = xlsynth_test_helpers::make_test_tmpdir("dslx_list_fns_jsonl");
    let dir = tmp.path();

    let dslx_path = dir.join("list_me.x");
    let dslx_src = r#"
pub fn add(a: u32, b: u32) -> u32 {
    a + b
}

pub fn id<N: u32 = {32}>(x: bits[N]) -> bits[N] {
    x
}
"#;
    std::fs::write(&dslx_path, dslx_src).expect("write dslx");

    let output = Command::new(driver)
        .arg("dslx-list-fns")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .output()
        .expect("run driver");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().filter(|line| !line.is_empty()).collect();
    assert_eq!(lines.len(), 2, "stdout: {}", stdout);

    let add: serde_json::Value = serde_json::from_str(lines[0]).expect("valid JSONL record");
    assert_eq!(add["name"], "add");
    assert_eq!(add["is_parametric"], false);
    assert_eq!(add["return_type"], "uN[32]");
    assert_eq!(add["function_type"], "(uN[32], uN[32]) -> uN[32]");
    assert_eq!(add["params"][0]["name"], "a");
    assert_eq!(add["params"][0]["type"], "uN[32]");
    assert_eq!(add["params"][1]["name"], "b");
    assert_eq!(add["params"][1]["type"], "uN[32]");

    let id: serde_json::Value = serde_json::from_str(lines[1]).expect("valid JSONL record");
    assert_eq!(id["name"], "id");
    assert_eq!(id["is_parametric"], true);
    assert!(id["return_type"].is_null());
    assert!(id["function_type"].is_null());
    assert_eq!(id["parametric_bindings"][0]["name"], "N");
    let default_expr = id["parametric_bindings"][0]["default_expr"]
        .as_str()
        .expect("default expr should be a string");
    assert!(
        default_expr.contains("32"),
        "default expr should mention 32; got {}",
        default_expr
    );
}

#[test]
fn test_dslx_list_fns_json_array_format() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    let tmp = xlsynth_test_helpers::make_test_tmpdir("dslx_list_fns_json");
    let dir = tmp.path();

    let dslx_path = dir.join("single_fn.x");
    let dslx_src = "fn neg(x: s32) -> s32 { -x }";
    std::fs::write(&dslx_path, dslx_src).expect("write dslx");

    let output = Command::new(driver)
        .arg("dslx-list-fns")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--format")
        .arg("json")
        .output()
        .expect("run driver");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let value: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    let arr = value.as_array().expect("top-level array");
    assert_eq!(arr.len(), 1, "stdout: {}", stdout);
    let rec = &arr[0];
    assert_eq!(rec["name"], "neg");
    assert_eq!(rec["is_parametric"], false);
    assert_eq!(rec["return_type"], "sN[32]");
    assert_eq!(rec["function_type"], "(sN[32]) -> sN[32]");
}

#[test]
fn test_dslx_list_fns_excludes_test_and_quickcheck_annotated_functions() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    let tmp = xlsynth_test_helpers::make_test_tmpdir("dslx_list_fns_filter_attrs");
    let dir = tmp.path();

    let dslx_path = dir.join("attrs.x");
    let dslx_src = r#"
fn ordinary(x: u8) -> u8 { x }

#[test]
fn unit_test() {
    assert_eq(ordinary(u8:1), u8:1)
}

#[quickcheck]
fn qc_reflexive(x: u8) -> bool { ordinary(x) == x }
"#;
    std::fs::write(&dslx_path, dslx_src).expect("write dslx");

    let output = Command::new(driver)
        .arg("dslx-list-fns")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .output()
        .expect("run driver");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let records: Vec<serde_json::Value> = stdout
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).expect("valid JSONL record"))
        .collect();

    assert_eq!(records.len(), 1, "stdout: {}", stdout);
    assert_eq!(records[0]["name"], "ordinary");
}

#[test]
fn test_dslx_list_fns_handles_missing_return_type_annotation() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    let tmp =
        xlsynth_test_helpers::make_test_tmpdir("dslx_list_fns_missing_return_type_annotation");
    let dir = tmp.path();

    let dslx_path = dir.join("inferred_ret.x");
    let dslx_src = "fn inferred(x: u32) { let _y = x; }";
    std::fs::write(&dslx_path, dslx_src).expect("write dslx");

    let output = Command::new(driver)
        .arg("dslx-list-fns")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .output()
        .expect("run driver");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let records: Vec<serde_json::Value> = stdout
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).expect("valid JSONL record"))
        .collect();

    assert_eq!(records.len(), 1, "stdout: {}", stdout);
    assert_eq!(records[0]["name"], "inferred");
    assert!(records[0]["return_type"].is_null());
    assert!(records[0]["function_type"].is_null());
}
