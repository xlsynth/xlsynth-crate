// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::ReadDir;
use std::path::PathBuf;
use std::process::Command;
use syn::{ForeignItem, Item, Visibility};

fn list_dso_files_in_dir(dir: &str) -> Vec<PathBuf> {
    let read_dir: ReadDir = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("Failed to read XLS_DSO_PATH directory {}: {}", dir, e));
    let mut results: Vec<PathBuf> = Vec::new();
    for entry_result in read_dir {
        let entry = entry_result.unwrap();
        let path: PathBuf = entry.path();
        if !path.is_file() {
            continue;
        }
        let file_name_opt = path.file_name().and_then(OsStr::to_str);
        if let Some(file_name) = file_name_opt {
            let is_linux = cfg!(target_os = "linux");
            let is_macos = cfg!(target_os = "macos");
            let matches_platform = (is_linux && file_name.ends_with(".so"))
                || (is_macos && file_name.ends_with(".dylib"));
            if matches_platform && file_name.starts_with("libxls-") {
                results.push(path);
            }
        }
    }
    results
}

fn parse_rust_sys_binding_names(sys_lib_rs_source: &str) -> BTreeSet<String> {
    let file: syn::File =
        syn::parse_file(sys_lib_rs_source).expect("parse sys lib.rs source as Rust file");
    let mut names: BTreeSet<String> = BTreeSet::new();
    for item in file.items {
        if let Item::ForeignMod(fm) = item {
            for fitem in fm.items {
                if let ForeignItem::Fn(f) = fitem {
                    if matches!(f.vis, Visibility::Public(_)) {
                        let ident = f.sig.ident.to_string();
                        if ident.starts_with("xls_") {
                            names.insert(ident);
                        }
                    }
                }
            }
        }
    }
    names
}

#[test]
fn all_dso_xls_symbols_are_bound_in_sys() {
    // Only implemented/validated on Linux for now.
    if !cfg!(target_os = "linux") {
        eprintln!("Skipping DSO symbol coverage test on non-Linux target.");
        return;
    }

    // XLS_DSO_PATH may be a directory (common in CI where build.rs downloads
    // artifacts) or a file path (when environment provides a concrete DSO).
    // Handle both cases.
    let input_path = PathBuf::from(xlsynth_sys::XLS_DSO_PATH);
    let chosen: PathBuf = if input_path.is_dir() {
        let dso_dir: &str = xlsynth_sys::XLS_DSO_PATH;
        let mut candidates: Vec<PathBuf> = list_dso_files_in_dir(dso_dir);
        assert!(
            !candidates.is_empty(),
            "No libxls shared objects found in XLS_DSO_PATH directory: {}",
            dso_dir
        );
        candidates.sort();
        candidates
            .last()
            .expect("expected at least one candidate shared object")
            .to_path_buf()
    } else if input_path.is_file() {
        input_path
    } else {
        panic!(
            "XLS_DSO_PATH should be a directory or a shared object file; got: {}",
            input_path.display()
        );
    };
    let dso_path: &PathBuf = &chosen;

    // Use nm to enumerate exported symbols. We rely on nm being available in CI.
    let output = Command::new("nm")
        .arg("-D")
        .arg("--defined-only")
        .arg(dso_path)
        .output()
        .expect("failed to execute nm to list DSO symbols");
    assert!(
        output.status.success(),
        "nm failed with status {:?}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let nm_stdout = String::from_utf8_lossy(&output.stdout);
    let mut dso_symbols: BTreeSet<String> = BTreeSet::new();
    for line in nm_stdout.lines() {
        // Format: "<addr> <type> <name>" or variations; take last field.
        let maybe_name = line.split_whitespace().last();
        if let Some(name) = maybe_name {
            if name.starts_with("xls_") {
                dso_symbols.insert(name.to_string());
            }
        }
    }
    assert!(
        !dso_symbols.is_empty(),
        "No xls_* symbols found in DSO {}; nm output was:\n{}",
        dso_path.display(),
        nm_stdout
    );

    // Load the sys bindings file and extract all pub extern fn xls_* names.
    let sys_lib_rs_source: &str = include_str!("../src/lib.rs");
    let binding_names: BTreeSet<String> = parse_rust_sys_binding_names(sys_lib_rs_source);
    assert!(
        !binding_names.is_empty(),
        "No xls_* pub fn bindings found in xlsynth-sys/src/lib.rs"
    );

    // Compute difference: symbols present in DSO but not declared in Rust externs.
    let missing: Vec<&String> = dso_symbols
        .iter()
        .filter(|name| !binding_names.contains(*name))
        .collect();

    if !missing.is_empty() {
        let mut report = String::new();
        report.push_str("Missing xls_* symbols (present in DSO, absent in Rust externs):\n");
        for name in &missing {
            report.push_str("  ");
            report.push_str(name);
            report.push('\n');
        }
        report.push_str("\nDSO path: ");
        report.push_str(&dso_path.display().to_string());
        report.push('\n');
        panic!("{}", report);
    }
}
