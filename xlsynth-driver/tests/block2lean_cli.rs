// SPDX-License-Identifier: Apache-2.0

//! Process-boundary checks for argument handling and output publication.
//! The library owns block semantics.

use std::path::Path;
use std::process::{Command, Output};

use xlsynth_lean::block;

/// Shared swap-register fixture with reset, enable, and assertion behavior.
const BLOCK: &str = include_str!("../../xlsynth-lean/tests/fixtures/block_step.ir");
/// Runs the real command without inherited logging noise.
fn invoke(directory: &Path, arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(directory)
        .env_remove("RUST_LOG")
        .arg("block2lean")
        .args(arguments)
        .output()
        .unwrap()
}

// Verifies: Raw stdout and file output match the qualified sequential golden.
// Catches: Missing dispatch, output contamination, and changed model text.
#[test]
fn raw_stdout_and_file_match_the_sequential_golden() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("input.ir"), BLOCK).unwrap();
    let golden = include_str!("../../xlsynth-lean/tests/fixtures/block_step.lean")
        .strip_prefix("-- SPDX-License-Identifier: Apache-2.0\n\n")
        .unwrap();
    let stdout = invoke(directory.path(), &["input.ir"]);
    assert!(stdout.status.success(), "{:?}", stdout);
    assert!(stdout.stderr.is_empty());
    assert_eq!(String::from_utf8(stdout.stdout).unwrap(), golden);

    let file = invoke(directory.path(), &["input.ir", "--output", "model.lean"]);
    assert!(file.status.success(), "{:?}", file);
    assert!(file.stdout.is_empty());
    assert!(file.stderr.is_empty());
    assert_eq!(
        std::fs::read_to_string(directory.path().join("model.lean")).unwrap(),
        golden
    );
}

// Verifies: Explicit selection overrides the top and resolves multiple blocks.
// Catches: Ignored top flags and silently selecting an ambiguous block.
#[test]
fn raw_top_overrides_the_package_top_and_resolves_ambiguity() {
    let directory = tempfile::tempdir().unwrap();
    let other = BLOCK
        .split_once("top block")
        .unwrap()
        .1
        .replace("swap(", "other(")
        .replace("id=", "id=10");
    let multiple = format!("{BLOCK}\nblock{other}");
    for input in [&multiple, &multiple.replace("top block", "block")] {
        std::fs::write(directory.path().join("input.ir"), input).unwrap();
        let output = invoke(directory.path(), &["input.ir", "--top", "other"]);
        assert!(output.status.success(), "{:?}", output);
        assert!(output.stderr.is_empty());
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            block::emit(&block::parse(input, Some("other")).unwrap()).unwrap()
        );
    }
    let ambiguous = invoke(directory.path(), &["input.ir"]);
    assert_eq!(ambiguous.status.code(), Some(1));
    assert!(ambiguous.stdout.is_empty());
    assert!(
        String::from_utf8(ambiguous.stderr)
            .unwrap()
            .contains("multiple blocks; supply a block name")
    );
}

#[test]
fn invalid_block_preserves_existing_output() {
    let directory = tempfile::tempdir().unwrap();
    let directory = directory.path();
    std::fs::write(directory.join("model.lean"), "keep").unwrap();
    std::fs::write(directory.join("input.ir"), "not an IR package").unwrap();
    let output = invoke(directory, &["input.ir", "--output", "model.lean"]);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(
        std::fs::read_to_string(directory.join("model.lean")).unwrap(),
        "keep"
    );
}
