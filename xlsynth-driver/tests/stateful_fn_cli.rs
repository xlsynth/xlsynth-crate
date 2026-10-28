// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::process::{Command, Output};

fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn source_and_ir_commands_produce_the_same_block_and_raw_model() {
    let dir = tempfile::tempdir().unwrap();
    let dir = dir.path();
    let path = dir.join("counter.x");
    let source = "struct Outputs { seen: u1, valid: u1 }
pub fn tick(x: u1, __reg_count: u1) -> (Outputs, u1) {
    (Outputs { seen: __reg_count, valid: x }, x)
}";
    std::fs::write(&path, source).unwrap();
    let ir = xlsynth::convert_dslx_to_ir_text(source, &path, &Default::default())
        .unwrap()
        .ir;
    std::fs::write(dir.join("counter.ir"), ir).unwrap();
    let from_dslx = run(
        dir,
        &[
            "dslx-stateful-fn-to-block",
            path.to_str().unwrap(),
            "--top",
            "tick",
        ],
    );
    assert!(from_dslx.status.success(), "{from_dslx:?}");
    let from_ir = run(
        dir,
        &[
            "ir-stateful-fn-to-block",
            "counter.ir",
            "--top",
            "__counter__tick",
        ],
    );
    assert!(from_ir.status.success(), "{from_ir:?}");
    let from_ir = String::from_utf8(from_ir.stdout).unwrap();
    let from_dslx = String::from_utf8(from_dslx.stdout).unwrap();
    let raw = |ir: &str| {
        std::fs::write(dir.join("saved.ir"), ir).unwrap();
        let output = run(dir, &["block2lean", "saved.ir"]);
        assert!(output.status.success(), "{output:?}");
        String::from_utf8(output.stdout).unwrap()
    };
    assert_eq!(raw(&from_ir), raw(&from_dslx));
    let lean = run(
        dir,
        &[
            "dslx2lean",
            path.to_str().unwrap(),
            "--top",
            "tick",
            "--output-dir",
            "model",
        ],
    );
    assert!(lean.status.success(), "{lean:?}");
    assert_eq!(
        std::fs::read_to_string(dir.join("model/Generated.lean")).unwrap(),
        raw(&from_dslx)
    );
    assert!(dir.join("model/ReadableCorrespondence.lean").is_file());
    let representation =
        std::fs::read_to_string(dir.join("model/ReadableRepresentation.lean")).unwrap();
    assert!(representation.contains("«out0» :="));
    assert!(representation.contains("x.«out0»"));
}

#[test]
fn malformed_register_suffix_leaves_no_model() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("bad.x"),
        "pub fn f(__reg_count: u1, x: u1) -> (u1, u1) { (x, __reg_count) }",
    )
    .unwrap();
    let output = run(
        dir.path(),
        &["dslx2lean", "bad.x", "--top", "f", "--output-dir", "model"],
    );
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("suffix"),
        "{output:?}"
    );
    assert!(!dir.path().join("model").exists());
}
