// SPDX-License-Identifier: Apache-2.0

#![cfg(unix)]

use std::path::Path;
use std::process::Command;
use std::time::Duration;
use xlsynth::external_tool::run_with_timeout;

/// Runs the actual fuzz entry point and verifies an infrastructure exit.
fn assert_setup_failure(binary: &str, environment: &[(&str, &Path)], expected: &str) {
    let artifacts = tempfile::tempdir().unwrap();
    let mut command = Command::new(binary);
    command
        .current_dir(artifacts.path())
        .args(["-runs=1", "-artifact_prefix=./"])
        .env_remove("XLSYNTH_LIBERTY_FILES");
    for (name, value) in environment {
        command.env(name, value);
    }
    let diagnostics = tempfile::tempdir().unwrap();
    let result = run_with_timeout(
        &mut command,
        diagnostics.path(),
        "preflight",
        Duration::from_secs(30),
    )
    .unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert_eq!(result.status.code(), Some(2), "{stderr}");
    assert!(
        stderr.contains("fuzz setup failed:") && stderr.contains(expected),
        "{stderr}"
    );
    assert!(!stderr.contains("panicked at"), "{stderr}");
    assert_eq!(std::fs::read_dir(artifacts.path()).unwrap().count(), 0);
}

#[test]
fn missing_liberty_is_a_startup_error_not_a_crashing_sample() {
    assert_setup_failure(
        env!("CARGO_BIN_EXE_fuzz_codegen_yosys_combo"),
        &[("XLSYNTH_YOSYS_PATH", Path::new("/usr/bin/true"))],
        "Set XLSYNTH_LIBERTY_FILES=/path/to/combo.lib,/path/to/seq.lib",
    );
}

#[test]
fn missing_icarus_is_reported_before_decoding_samples() {
    let directory = tempfile::tempdir().unwrap();
    assert_setup_failure(
        env!("CARGO_BIN_EXE_fuzz_codegen_native_semantics"),
        &[(
            "XLSYNTH_IVERILOG_PATH",
            &directory.path().join("missing-iverilog"),
        )],
        "missing-iverilog",
    );
}

#[test]
fn sequential_target_requires_flip_flops_before_mapping_samples() {
    let directory = tempfile::tempdir().unwrap();
    let library = directory.path().join("combo.lib");
    std::fs::write(
        &library,
        r#"library(combo) {
      cell(INV) {
        area: 1;
        pin(A) { direction: input; }
        pin(Y) { direction: output; function: "!A"; }
      }
    }
"#,
    )
    .unwrap();
    assert_setup_failure(
        env!("CARGO_BIN_EXE_fuzz_codegen_yosys_sequential"),
        &[
            ("XLSYNTH_YOSYS_PATH", Path::new("/usr/bin/true")),
            ("XLSYNTH_LIBERTY_FILES", &library),
        ],
        "include a sequential Liberty file",
    );
}
