// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::process::{Command, Output};

fn compile(dir: &Path, top: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(dir)
        .args([
            "dslx2lean",
            "counter.x",
            "--top",
            top,
            "--output-dir",
            "model",
        ])
        .output()
        .unwrap()
}

#[test]
fn selecting_a_proc_reports_the_required_function_and_creates_no_bundle() {
    let directory = tempfile::tempdir().unwrap();
    let dir = directory.path();
    std::fs::write(
        dir.join("counter.x"),
        "proc Counter { config() { () } init { u1:0 } next(state: u1) { state } }",
    )
    .unwrap();
    let output = compile(dir, "Counter");
    assert!(!output.status.success());
    let diagnostic = String::from_utf8(output.stderr).unwrap();
    assert!(
        diagnostic.contains("Counter") && diagnostic.contains("function"),
        "{diagnostic}"
    );
    assert!(!dir.join("model").exists());
}

#[test]
fn an_existing_output_directory_is_preserved() {
    let directory = tempfile::tempdir().unwrap();
    let dir = directory.path();
    std::fs::write(
        dir.join("counter.x"),
        "pub fn tick(x: u1, __reg_count: u1) -> (u1, u1) { (__reg_count, x) }",
    )
    .unwrap();
    let model = dir.join("model");
    std::fs::create_dir(&model).unwrap();
    std::fs::write(model.join("sentinel"), "keep").unwrap();
    let output = compile(dir, "tick");
    assert!(!output.status.success());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("creating new output directory")
    );
    assert_eq!(std::fs::read_dir(&model).unwrap().count(), 1);
    assert_eq!(
        std::fs::read_to_string(model.join("sentinel")).unwrap(),
        "keep"
    );
}
