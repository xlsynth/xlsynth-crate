// SPDX-License-Identifier: Apache-2.0

//! QuickCheck-style proving via external toolchain binary
//! `prove_quickcheck_main`.
//!
//! This invokes the upstream tool directly and maps its exit status to a
//! library-friendly `BoolPropertyResult` without printing.

use crate::types::BoolPropertyResult;

fn prove_quickcheck_with_exe_internal(
    exe: &std::path::Path,
    entry_file: &std::path::Path,
    quickcheck_name: &str,
) -> BoolPropertyResult {
    if !exe.exists() {
        return BoolPropertyResult::ToolchainDisproved(format!(
            "prove_quickcheck_main not found: {}",
            exe.display()
        ));
    }
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--test_filter").arg(quickcheck_name);
    cmd.arg(entry_file);

    match cmd.output() {
        Ok(output) if output.status.success() => BoolPropertyResult::Proved,
        Ok(output) => {
            let mut msg = String::new();
            if !output.stdout.is_empty() {
                msg.push_str(&String::from_utf8_lossy(&output.stdout));
            }
            if msg.trim().is_empty() && !output.stderr.is_empty() {
                msg.push_str(&String::from_utf8_lossy(&output.stderr));
            }
            let snippet: String = msg.chars().take(512).collect();
            BoolPropertyResult::ToolchainDisproved(snippet.trim().to_string())
        }
        Err(e) => BoolPropertyResult::ToolchainDisproved(format!("spawn failed: {}", e)),
    }
}

/// Prove a single DSLX quickcheck function using a specific
/// `prove_quickcheck_main` executable.
pub fn prove_dslx_quickcheck_with_tool_exe<P: AsRef<std::path::Path>>(
    entry_file: &std::path::Path,
    quickcheck_name: &str,
    tool_exe: P,
) -> BoolPropertyResult {
    prove_quickcheck_with_exe_internal(tool_exe.as_ref(), entry_file, quickcheck_name)
}

/// Prove a single DSLX quickcheck function by locating `prove_quickcheck_main`
/// in `tool_dir`.
pub fn prove_dslx_quickcheck_with_tool_dir<P: AsRef<std::path::Path>>(
    entry_file: &std::path::Path,
    quickcheck_name: &str,
    tool_dir: P,
) -> BoolPropertyResult {
    let exe = tool_dir.as_ref().join("prove_quickcheck_main");
    if !exe.exists() {
        return BoolPropertyResult::ToolchainDisproved(format!(
            "prove_quickcheck_main not found in {}",
            tool_dir.as_ref().display()
        ));
    }
    prove_quickcheck_with_exe_internal(&exe, entry_file, quickcheck_name)
}

/// Reads `XLSYNTH_TOOLS` to locate the toolchain directory and runs a single
/// quickcheck.
pub fn prove_dslx_quickcheck_via_toolchain(
    entry_file: &std::path::Path,
    quickcheck_name: &str,
) -> BoolPropertyResult {
    match std::env::var("XLSYNTH_TOOLS") {
        Ok(dir) => prove_dslx_quickcheck_with_tool_dir(entry_file, quickcheck_name, dir),
        Err(_) => BoolPropertyResult::ToolchainDisproved(
            "XLSYNTH_TOOLS is not set; cannot run toolchain quickcheck".to_string(),
        ),
    }
}
