// SPDX-License-Identifier: Apache-2.0

//! Explicit native checks for generated readable block correspondence proofs.

use std::{fs, path::Path, process::Command};

use xlsynth_lean::block::{Attachment, collect_source_map, emit_readable, validate_source_map};

// Verifies: Nested source records, aliases, undeclared enum encodings, and
// selected records/fields have complete Lean correspondence proofs.
// Catches: Well-formed generated text whose proof tactics cannot close the
// equalities between readable and raw execution.
#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn readable_machine_and_nested_selections_compile() {
    let lean = std::env::var_os("XLSYNTH_LEAN")
        .expect("set XLSYNTH_LEAN to the actual Lean executable path");
    let source = include_str!("fixtures/readable_machine/machine.x");
    let source_path = Path::new("machine.x");
    for (fixture, block) in [
        (
            "machine",
            include_str!("fixtures/readable_machine/machine.block.ir"),
        ),
        (
            "nested selections",
            include_str!("fixtures/readable_machine/selections.block.ir"),
        ),
    ] {
        let map = collect_source_map(
            block,
            source,
            source_path,
            &[Attachment {
                register: "machine".into(),
                source_type: "Machine".into(),
            }],
        )
        .unwrap();
        let checked = validate_source_map(&map, block, source, source_path).unwrap();
        let bundle = emit_readable(&checked).unwrap();
        let directory = tempfile::tempdir().unwrap();
        for (name, source) in bundle.files {
            let path = directory.path().join(name);
            fs::write(&path, source).unwrap();
            let output = Command::new(&lean)
                .arg("-o")
                .arg(path.with_extension("olean"))
                .arg(&path)
                .current_dir(directory.path())
                .env("LEAN_PATH", directory.path())
                .env_remove("LEAN_SYSROOT")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "Lean failed for {fixture}, {}:\n{}\n{}",
                path.display(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
    }
}
