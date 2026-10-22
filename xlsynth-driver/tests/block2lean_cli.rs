// SPDX-License-Identifier: Apache-2.0

//! Process-boundary checks for argument handling and output publication.
//! The library owns block semantics and source-map validation.

use std::path::Path;
use std::process::{Command, Output};

use xlsynth_lean::block::{self, Attachment};

/// Shared swap-register fixture with reset, enable, and assertion behavior.
const BLOCK: &str = include_str!("../../xlsynth-lean/tests/fixtures/block_step.ir");
/// Source declarations covering aliases, enums, and nested structs.
const SOURCE: &str = include_str!("../../xlsynth-lean/tests/fixtures/readable_machine/machine.x");
/// A block retaining field projections needed for a checked struct attachment.
const MACHINE: &str =
    include_str!("../../xlsynth-lean/tests/fixtures/readable_machine/machine.block.ir");

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

// Verifies: Readable CLI files match library output with all attachments.
// Catches: Dropped repeated attachments, missing modules, and changed file
// text.
#[test]
fn readable_bundle_matches_library_with_structs_and_repeated_attachments() {
    for (ir, pairs) in [
        (MACHINE, vec![("machine", "Machine")]),
        (BLOCK, vec![("a", "Word"), ("b", "Word")]),
    ] {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("input.ir"), ir).unwrap();
        std::fs::write(directory.path().join("machine.x"), SOURCE).unwrap();
        let attachments: Vec<_> = pairs
            .iter()
            .map(|(register, source_type)| Attachment {
                register: (*register).to_owned(),
                source_type: (*source_type).to_owned(),
            })
            .collect();
        let attachment_flags: Vec<_> = pairs
            .iter()
            .map(|(register, source_type)| format!("--type-attachment={register}={source_type}"))
            .collect();
        let arguments: Vec<_> = ["input.ir", "--dslx-source=machine.x", "--output-dir=bundle"]
            .into_iter()
            .chain(attachment_flags.iter().map(String::as_str))
            .collect();
        let output = invoke(directory.path(), &arguments);
        assert!(output.status.success(), "{:?}", output);
        assert!(output.stdout.is_empty());
        assert!(output.stderr.is_empty());

        let source_path = directory.path().join("machine.x");
        let map = block::collect_source_map(ir, SOURCE, &source_path, &attachments).unwrap();
        let checked = block::validate_source_map(&map, ir, SOURCE, &source_path).unwrap();
        let expected = block::emit_readable(&checked).unwrap();
        let bundle = directory.path().join("bundle");
        assert_eq!(std::fs::read_dir(&bundle).unwrap().count(), 5);
        for (name, text) in expected.files {
            assert_eq!(std::fs::read_to_string(bundle.join(name)).unwrap(), text);
        }
    }
}

// Negative test: Incomplete or conflicting flags trigger CLI error handling
// before IO.
#[test]
fn readable_options_are_complete_and_exclusive_before_reading_inputs() {
    let directory = tempfile::tempdir().unwrap();
    let readable = [
        "--dslx-source=missing.x",
        "--type-attachment=a=Word",
        "--output-dir=bundle",
    ];
    for mask in 1..7 {
        let args: Vec<_> = std::iter::once("missing.ir")
            .chain(
                readable
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| mask & (1 << index) != 0)
                    .map(|(_, flag)| *flag),
            )
            .collect();
        let output = invoke(directory.path(), &args);
        assert_eq!(output.status.code(), Some(2), "{args:?}: {output:?}");
        assert!(output.stdout.is_empty());
    }
    for conflict in ["--top=swap", "--output=model.lean"] {
        let args: Vec<_> = std::iter::once("missing.ir")
            .chain(readable)
            .chain([conflict])
            .collect();
        let output = invoke(directory.path(), &args);
        assert_eq!(output.status.code(), Some(2), "{args:?}: {output:?}");
        assert!(output.stdout.is_empty());
    }
    assert!(!directory.path().join("bundle").exists());
    assert!(!directory.path().join("model.lean").exists());
}

// Negative test: Malformed attachments trigger CLI error handling before IO.
#[test]
fn malformed_attachments_are_cli_errors() {
    let directory = tempfile::tempdir().unwrap();
    for attachment in ["a", "=Word", "a=", "a=Word=Other"] {
        let flag = format!("--type-attachment={attachment}");
        let output = invoke(
            directory.path(),
            &[
                "missing.ir",
                "--dslx-source=missing.x",
                "--output-dir=bundle",
                &flag,
            ],
        );
        assert_eq!(output.status.code(), Some(2), "{output:?}");
        assert!(output.stdout.is_empty());
        assert!(
            String::from_utf8(output.stderr)
                .unwrap()
                .contains("REGISTER=TYPE")
        );
        assert!(!directory.path().join("bundle").exists());
    }
}

// Negative test: Parse and emission error handling preserves existing and
// absent outputs.
#[test]
fn invalid_input_does_not_truncate_raw_output_or_create_a_bundle() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("model.lean"), "keep this output").unwrap();
    std::fs::write(directory.path().join("machine.x"), SOURCE).unwrap();
    let unsupported = BLOCK.replace("asynchronous=false", "asynchronous=true");
    for input in ["not IR", unsupported.as_str()] {
        std::fs::write(directory.path().join("input.ir"), input).unwrap();
        for arguments in [
            vec!["input.ir", "--output=model.lean"],
            vec![
                "input.ir",
                "--dslx-source=machine.x",
                "--type-attachment=a=Word",
                "--output-dir=bundle",
            ],
        ] {
            let output = invoke(directory.path(), &arguments);
            assert_eq!(output.status.code(), Some(1), "{output:?}");
            assert!(output.stdout.is_empty());
            assert!(
                String::from_utf8(output.stderr)
                    .unwrap()
                    .starts_with("error:")
            );
            assert_eq!(
                std::fs::read_to_string(directory.path().join("model.lean")).unwrap(),
                "keep this output"
            );
            assert!(!directory.path().join("bundle").exists());
        }
    }
}

// Negative test: Missing-input and invalid-binding error handling creates no
// outputs.
#[test]
fn missing_files_and_invalid_bindings_do_not_create_a_bundle() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("input.ir"), BLOCK).unwrap();
    std::fs::write(directory.path().join("machine.x"), SOURCE).unwrap();
    for arguments in [
        vec!["missing.ir", "--output=model.lean"],
        vec![
            "input.ir",
            "--dslx-source=missing.x",
            "--type-attachment=a=Word",
            "--output-dir=bundle",
        ],
        vec![
            "input.ir",
            "--dslx-source=machine.x",
            "--type-attachment=absent=Word",
            "--output-dir=bundle",
        ],
        vec![
            "input.ir",
            "--dslx-source=machine.x",
            "--type-attachment=a=Word",
            "--type-attachment=a=Word",
            "--output-dir=bundle",
        ],
    ] {
        let output = invoke(directory.path(), &arguments);
        assert_eq!(output.status.code(), Some(1), "{output:?}");
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
        assert!(!directory.path().join("bundle").exists());
        assert!(!directory.path().join("model.lean").exists());
    }
}

// Negative test: Existing-directory error handling preserves its files without
// additions.
#[test]
fn readable_output_refuses_an_existing_directory() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("input.ir"), MACHINE).unwrap();
    std::fs::write(directory.path().join("machine.x"), SOURCE).unwrap();
    let bundle = directory.path().join("bundle");
    std::fs::create_dir(&bundle).unwrap();
    std::fs::write(bundle.join("Generated.lean"), "keep this output").unwrap();
    let output = invoke(
        directory.path(),
        &[
            "input.ir",
            "--dslx-source=machine.x",
            "--type-attachment=machine=Machine",
            "--output-dir=bundle",
        ],
    );
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(output.stdout.is_empty());
    assert_eq!(std::fs::read_dir(&bundle).unwrap().count(), 1);
    assert_eq!(
        std::fs::read_to_string(bundle.join("Generated.lean")).unwrap(),
        "keep this output"
    );
}
