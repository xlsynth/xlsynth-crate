// SPDX-License-Identifier: Apache-2.0

//! Helpers for invoking external XLS toolchain binaries from the prover.

use std::collections::HashMap;
use std::path::Path;
use std::process::{Command, Output};

use log::info;
use tempfile::NamedTempFile;

use crate::prove_quickcheck::load_quickcheck_context;
use crate::prover::ExternalProver;
use crate::types::{BoolPropertyResult, QuickCheckAssertionSemantics, QuickCheckRunResult};

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

fn run_prove_quickcheck_main(
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

/// Prove a single DSLX quickcheck function using a specific toolchain
/// executable.
pub fn prove_dslx_quickcheck_with_tool_exe<P: AsRef<Path>>(
    tool_exe: P,
    entry_file: &Path,
    dslx_stdlib_path: Option<&Path>,
    additional_search_paths: &[&Path],
    test_filter: &str,
) -> BoolPropertyResult {
    match run_prove_quickcheck_main(
        tool_exe.as_ref(),
        entry_file,
        dslx_stdlib_path,
        additional_search_paths,
        test_filter,
    ) {
        Ok(_) => BoolPropertyResult::Proved,
        Err(msg) => BoolPropertyResult::ToolchainDisproved(truncate_message(&msg)),
    }
}

/// Prove a single DSLX quickcheck function by locating `prove_quickcheck_main`
/// in `tool_dir`.
pub fn prove_dslx_quickcheck_with_tool_dir<P: AsRef<Path>>(
    tool_dir: P,
    entry_file: &Path,
    dslx_stdlib_path: Option<&Path>,
    additional_search_paths: &[&Path],
    test_filter: &str,
) -> BoolPropertyResult {
    let exe = tool_dir.as_ref().join("prove_quickcheck_main");
    if !exe.exists() {
        return BoolPropertyResult::ToolchainDisproved(format!(
            "prove_quickcheck_main not found in {}",
            tool_dir.as_ref().display()
        ));
    }
    match run_prove_quickcheck_main(
        &exe,
        entry_file,
        dslx_stdlib_path,
        additional_search_paths,
        test_filter,
    ) {
        Ok(_) => BoolPropertyResult::Proved,
        Err(msg) => BoolPropertyResult::ToolchainDisproved(truncate_message(&msg)),
    }
}

/// Reads `XLSYNTH_TOOLS` to locate the toolchain directory and runs a single
/// quickcheck.
pub fn prove_dslx_quickcheck_via_toolchain(
    entry_file: &Path,
    dslx_stdlib_path: Option<&Path>,
    additional_search_paths: &[&Path],
    test_filter: &str,
) -> BoolPropertyResult {
    match std::env::var("XLSYNTH_TOOLS") {
        Ok(dir) => prove_dslx_quickcheck_with_tool_dir(
            dir,
            entry_file,
            dslx_stdlib_path,
            additional_search_paths,
            test_filter,
        ),
        Err(_) => BoolPropertyResult::ToolchainDisproved(
            "XLSYNTH_TOOLS is not set; cannot run toolchain quickcheck".to_string(),
        ),
    }
}

pub(crate) fn prove_dslx_quickcheck_full_via_toolchain(
    prover: &ExternalProver,
    entry_file: &Path,
    dslx_stdlib_path: Option<&Path>,
    additional_search_paths: &[&Path],
    test_filter: Option<&str>,
    assertion_semantics: QuickCheckAssertionSemantics,
    assert_label_filter: Option<&str>,
    uf_map: &HashMap<String, String>,
) -> Vec<QuickCheckRunResult> {
    let (_, quickchecks) = load_quickcheck_context(
        entry_file,
        dslx_stdlib_path,
        additional_search_paths,
        test_filter,
    );
    if quickchecks.is_empty() {
        return Vec::new();
    }

    if assert_label_filter.is_some() {
        return quickchecks
            .into_iter()
            .map(|(name, _)| QuickCheckRunResult {
                name,
                duration: std::time::Duration::default(),
                result: BoolPropertyResult::ToolchainDisproved(
                    "External quickcheck does not support assertion label filters".to_string(),
                ),
            })
            .collect();
    }
    if !uf_map.is_empty() {
        return quickchecks
            .into_iter()
            .map(|(name, _)| QuickCheckRunResult {
                name,
                duration: std::time::Duration::default(),
                result: BoolPropertyResult::ToolchainDisproved(
                    "External quickcheck does not support uninterpreted functions".to_string(),
                ),
            })
            .collect();
    }

    let mut results = Vec::with_capacity(quickchecks.len());
    for (quickcheck_name, _) in quickchecks {
        let start_time = std::time::Instant::now();
        let filter = format!("^{}$", regex::escape(quickcheck_name.as_str()));
        let result = match prover {
            ExternalProver::ToolExe(path) => prove_dslx_quickcheck_with_tool_exe(
                path,
                entry_file,
                dslx_stdlib_path,
                additional_search_paths,
                filter.as_str(),
            ),
            ExternalProver::ToolDir(path) => prove_dslx_quickcheck_with_tool_dir(
                path,
                entry_file,
                dslx_stdlib_path,
                additional_search_paths,
                filter.as_str(),
            ),
            ExternalProver::Toolchain => prove_dslx_quickcheck_via_toolchain(
                entry_file,
                dslx_stdlib_path,
                additional_search_paths,
                filter.as_str(),
            ),
        };

        results.push(QuickCheckRunResult {
            name: quickcheck_name,
            duration: start_time.elapsed(),
            result,
        });
    }

    let _ = assertion_semantics;

    results
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

fn truncate_message(msg: &str) -> String {
    msg.chars().take(512).collect::<String>()
}
