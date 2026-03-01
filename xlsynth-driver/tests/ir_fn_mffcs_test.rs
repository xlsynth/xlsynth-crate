// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn ir_fn_mffcs_emits_ranked_manifest_and_outputs() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    let pkg_ir = r#"package p

top fn f(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  add.4: bits[8] = add(a, b, id=4)
  not.5: bits[8] = not(add.4, id=5)
  xor.6: bits[8] = xor(add.4, c, id=6)
  ret add.7: bits[8] = add(not.5, xor.6, id=7)
}
"#;

    let temp_dir = tempfile::tempdir().expect("create tempdir");
    let ir_path = temp_dir.path().join("in.ir");
    std::fs::write(&ir_path, pkg_ir).expect("write package IR");
    let out_dir = temp_dir.path().join("mffcs_out");

    let output = Command::new(driver)
        .arg("ir-fn-mffcs")
        .arg(ir_path.as_os_str())
        .arg("--output_dir")
        .arg(out_dir.as_os_str())
        .arg("--max_mffcs")
        .arg("8")
        .arg("--min_internal_non_literal")
        .arg("2")
        .output()
        .expect("ir-fn-mffcs invocation should run");

    assert!(
        output.status.success(),
        "ir-fn-mffcs failed: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let manifest_path = out_dir.join("manifest.jsonl");
    let manifest_text = std::fs::read_to_string(&manifest_path).expect("read manifest");
    let manifest_lines: Vec<&str> = manifest_text
        .lines()
        .filter(|line| !line.is_empty())
        .collect();
    assert!(
        !manifest_lines.is_empty(),
        "expected at least one manifest line; got empty manifest"
    );

    let first: serde_json::Value =
        serde_json::from_str(manifest_lines[0]).expect("parse first manifest line");
    assert_eq!(
        first["root_text_id"].as_u64(),
        Some(7),
        "expected meatiest root first in manifest"
    );
    let sha = first["sha256"]
        .as_str()
        .expect("sha256 should be present in manifest");
    let emitted_ir_path = out_dir.join(format!("{}.ir", sha));
    assert!(
        emitted_ir_path.exists(),
        "expected emitted MFFC file to exist: {}",
        emitted_ir_path.display()
    );
}
