// SPDX-License-Identifier: Apache-2.0

//! Thin wrappers around XLS toolchain binaries. Each function simply spawns the
//! corresponding executable and returns its textual output or an error string.

use std::path::Path;
use std::process::{Command, Output};

use log::info;
use tempfile::NamedTempFile;

fn run_command(command: &mut Command, description: &str) -> Result<Output, String> {
    info!("Running {}: {:?}", description, command);
    command
        .output()
        .map_err(|e| format!("Failed to spawn {}: {}", description, e))
}

/// Runs `ir_converter_main` from the XLS toolchain to convert DSLX to IR.
pub fn run_ir_converter_main(
    tool_path: &Path,
    input_file: &Path,
    dslx_top: Option<&str>,
    dslx_stdlib_path: Option<&Path>,
    additional_search_paths: &[&Path],
    enable_warnings: Option<&[String]>,
    disable_warnings: Option<&[String]>,
    type_inference_v2: Option<bool>,
) -> Result<String, String> {
    let mut command = Command::new(tool_path.join("ir_converter_main"));
    command.arg(input_file);
    if let Some(top) = dslx_top {
        command.arg("--top").arg(top);
    }
    if let Some(stdlib) = dslx_stdlib_path {
        command.arg("--dslx_stdlib_path").arg(stdlib);
    }
    if let Some(joined) = join_paths(additional_search_paths) {
        command.arg("--dslx_path").arg(joined);
    }
    if let Some(enable) = enable_warnings {
        if !enable.is_empty() {
            command.arg("--enable_warnings").arg(enable.join(","));
        }
    }
    if let Some(disable) = disable_warnings {
        if !disable.is_empty() {
            command.arg("--disable_warnings").arg(disable.join(","));
        }
    }
    command.arg("--convert_tests=false");
    if let Some(value) = type_inference_v2 {
        command.arg(format!(
            "--type_inference_v2={}",
            if value { "true" } else { "false" }
        ));
    }

    let output = run_command(&mut command, "ir_converter_main")?;
    if !output.status.success() {
        return Err(format!(
            "ir_converter_main failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Runs `opt_main` from the XLS toolchain to optimize IR.
pub fn run_opt_main(tool_path: &Path, ir_text: &str, top: &str) -> Result<String, String> {
    let tmp = NamedTempFile::new()
        .map_err(|e| format!("Failed to create temp file for opt_main: {}", e))?;
    std::fs::write(tmp.path(), ir_text)
        .map_err(|e| format!("Failed to write temp IR file for opt_main: {}", e))?;
    let mut command = Command::new(tool_path.join("opt_main"));
    command.arg(tmp.path()).arg("--top").arg(top);
    let output = run_command(&mut command, "opt_main")?;
    if !output.status.success() {
        return Err(format!(
            "opt_main failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Runs `prove_quickcheck_main` from the XLS toolchain.
pub fn run_prove_quickcheck_main(
    exe: &Path,
    entry_file: &Path,
    dslx_stdlib_path: Option<&Path>,
    additional_search_paths: &[&Path],
    test_filter: &str,
) -> Result<String, String> {
    if !exe.exists() {
        return Err(format!(
            "prove_quickcheck_main not found: {}",
            exe.display()
        ));
    }
    let mut cmd = Command::new(exe);
    cmd.arg("--test_filter").arg(format!(".*{}.*", test_filter));
    cmd.arg(entry_file);
    if let Some(stdlib) = dslx_stdlib_path {
        cmd.arg("--dslx_stdlib_path").arg(stdlib);
    }
    if let Some(joined) = join_paths(additional_search_paths) {
        cmd.arg("--dslx_path").arg(joined);
    }

    let output = run_command(&mut cmd, "prove_quickcheck_main")?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        let mut msg = String::new();
        if !output.stdout.is_empty() {
            msg.push_str(&String::from_utf8_lossy(&output.stdout));
        }
        if msg.trim().is_empty() && !output.stderr.is_empty() {
            msg.push_str(&String::from_utf8_lossy(&output.stderr));
        }
        if msg.trim().is_empty() {
            msg = format!("prove_quickcheck_main failed with status {}", output.status);
        }
        Err(msg.trim().to_string())
    }
}

fn join_paths(paths: &[&Path]) -> Option<String> {
    if paths.is_empty() {
        None
    } else {
        Some(
            paths
                .iter()
                .map(|p| p.to_string_lossy())
                .collect::<Vec<_>>()
                .join(";"),
        )
    }
}
