// SPDX-License-Identifier: Apache-2.0

use cargo_metadata::MetadataCommand;
use std::fs;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

fn check_spdx_identifier(file_path: &Path) -> bool {
    let filename = file_path.file_name().unwrap().to_str().unwrap();
    let comment_prefix = if filename.ends_with(".yml")
        || filename.ends_with(".yaml")
        || filename.ends_with(".py")
        || filename.ends_with(".sh")
        || filename == "requirements.txt"
    {
        "#"
    } else {
        "//"
    };
    let expect_shebang = filename.ends_with(".py") || filename.ends_with(".sh");
    let expected_spdx_identifier = format!("{comment_prefix} SPDX-License-Identifier: Apache-2.0");

    let file = fs::File::open(file_path).unwrap();
    let reader = io::BufReader::new(file);
    let mut lines = reader.lines();
    let ok = if expect_shebang {
        let first_line = lines.next().unwrap().unwrap();
        if first_line.starts_with("#!") {
            lines
                .next()
                .unwrap()
                .unwrap()
                .starts_with(&expected_spdx_identifier)
        } else {
            first_line.starts_with(&expected_spdx_identifier)
        }
    } else {
        let first_line = lines.next();
        if let Some(Ok(line)) = first_line {
            line.starts_with(&expected_spdx_identifier)
        } else {
            false
        }
    };
    if ok {
        println!("Found SPDX identifier in file: {file_path:?}");
    } else {
        eprintln!("Missing SPDX identifier in file: {file_path:?}");
    }
    ok
}

fn find_missing_spdx_files(root: &Path) -> Vec<PathBuf> {
    let mut missing_spdx_files = Vec::new();
    let mut dir_worklist: Vec<PathBuf> = vec![root.into()];

    while let Some(dir) = dir_worklist.pop() {
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();

            if path.is_dir() {
                // Exclude directories that should not be checked
                if entry.file_name() != "target"
                    && entry.file_name() != ".git"
                    && entry.file_name() != ".venv"
                    && entry.file_name() != "xlsynth_tools"
                    && entry.file_name() != "__pycache__"
                    && entry.file_name() != ".pytest_cache"
                    && entry.file_name() != ".mypy_cache"
                    && entry.file_name() != ".ruff_cache"
                    && entry.file_name() != ".vscode"
                {
                    println!("Adding to directory worklist: {path:?}");
                    dir_worklist.push(path.clone());
                }
                continue;
            }

            // For golden comparison files (i.e. ones we compare to literally for code
            // generation facilities) we don't require SPDX identifiers.
            let path_str = path.as_os_str().to_str().unwrap();

            // Exclude fuzz corpora; these are inputs, not source files.
            if path_str.contains("/fuzz/corpus/") || path_str.contains("/fuzz/artifacts/") {
                continue;
            }

            // Exclude benchmark input data.
            if path_str.contains("/benchdata/") {
                continue;
            }

            let filename = path.file_name().unwrap().to_str().unwrap();
            if filename == "abc.history" {
                continue;
            }
            if path_str.ends_with(".golden.sv")
                || path_str.ends_with(".golden.v")
                || path_str.ends_with(".golden.txt")
                || path_str.ends_with(".golden.ir")
                || (filename.starts_with("generated_") && filename.ends_with(".json"))
            {
                continue;
            }

            if path.file_name().unwrap().to_str().unwrap() == "estimator_model.proto" {
                continue;
            }

            // DSOs are binary files don't check for SPDX.
            if path.file_name().unwrap().to_str().unwrap().ends_with(".so") {
                continue;
            }
            if path
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .ends_with(".bin")
            {
                continue;
            }

            if let Some(extension) = path.extension() {
                if extension == "md"
                    || extension == "lock"
                    || extension == "toml"
                    || extension == "supp"
                {
                    continue;
                }
                // Check all source files, not just Rust files
                if !check_spdx_identifier(&path) {
                    missing_spdx_files.push(path);
                }
            }
        }
    }
    missing_spdx_files
}

#[test]
fn test_finds_rust_file_missing_spdx_in_tempdir() {
    let temp_dir = tempfile::tempdir().unwrap();
    let temp_dir_path = temp_dir.path();

    // Write one file that does have the SPDX identifier.
    let has_spdx_file = temp_dir_path.join("has_spdx.rs");
    fs::write(has_spdx_file, "// SPDX-License-Identifier: Apache-2.0\n").unwrap();

    // Write one file that does not have the SPDX identifier.
    let missing_spdx_file = temp_dir_path.join("missing_spdx.rs");
    fs::write(missing_spdx_file.clone(), "").unwrap();

    let missing_spdx_files = find_missing_spdx_files(temp_dir_path);
    assert_eq!(missing_spdx_files.len(), 1);
    assert!(missing_spdx_files.contains(&missing_spdx_file));
}

#[test]
fn check_all_rust_files_for_spdx() {
    // Use cargo_metadata to get the workspace root
    let metadata = MetadataCommand::new().exec().unwrap();
    let workspace_dir = metadata.workspace_root;
    let missing_spdx_files = find_missing_spdx_files(workspace_dir.as_std_path());
    if !missing_spdx_files.is_empty() {
        eprintln!(
            "\nSummary of files missing SPDX identifiers ({}):",
            missing_spdx_files.len()
        );
        for path in &missing_spdx_files {
            eprintln!("  - {}", path.display());
        }
        eprintln!("\n");
    }
    assert!(
        missing_spdx_files.is_empty(),
        "The following files are missing SPDX identifiers: {:?}",
        missing_spdx_files
    );
}
