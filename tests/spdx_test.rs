// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

fn check_spdx_identifier(file_path: &Path) -> bool {
    let file = fs::File::open(file_path).unwrap();
    let reader = io::BufReader::new(file);
    if let Some(Ok(first_line)) = reader.lines().next() {
        return first_line.starts_with("// SPDX-License-Identifier: Apache-2.0");
    }
    false
}

#[test]
fn check_all_rust_files_for_spdx() {
    let project_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut missing_spdx_files = Vec::new();
    let mut dir_worklist: Vec<PathBuf> = vec![project_dir.to_path_buf()];

    loop {
        let dir = match dir_worklist.pop() {
            Some(dir) => dir,
            None => break,
        };
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();

            if path.is_dir() {
                dir_worklist.push(path.clone());
                continue;
            }

            if let Some(extension) = path.extension() {
                if extension == "rs" || extension == "x" {
                    if !check_spdx_identifier(&path) {
                        missing_spdx_files.push(path);
                    }
                }
            }
        }
    }

    if !missing_spdx_files.is_empty() {
        panic!(
            "The following files are missing SPDX identifiers: {:?}",
            missing_spdx_files
        );
    }
}
