// SPDX-License-Identifier: Apache-2.0

//! QuickCheck-style proving via external toolchain binary
//! `prove_quickcheck_main`.
//!
//! This invokes the upstream tool directly and maps its exit status to a
//! library-friendly `BoolPropertyResult` without printing.

use crate::prove_quickcheck::load_quickcheck_context;
use crate::prover::ExternalProver;
use crate::types::{BoolPropertyResult, QuickCheckAssertionSemantics, QuickCheckRunResult};
use std::collections::HashMap;
use std::path::Path;

fn prove_quickcheck_with_exe_internal(
    exe: &std::path::Path,
    entry_file: &std::path::Path,
    dslx_stdlib_path: Option<&std::path::Path>,
    additional_search_paths: &[&std::path::Path],
    test_filter: &str,
) -> BoolPropertyResult {
    if !exe.exists() {
        return BoolPropertyResult::ToolchainDisproved(format!(
            "prove_quickcheck_main not found: {}",
            exe.display()
        ));
    }
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--test_filter").arg(format!(".*{}.*", test_filter));
    cmd.arg(entry_file);
    if let Some(stdlib) = dslx_stdlib_path {
        cmd.arg("--dslx_stdlib_path").arg(stdlib);
    }
    if !additional_search_paths.is_empty() {
        let joined = additional_search_paths
            .iter()
            .map(|p| p.to_string_lossy())
            .collect::<Vec<_>>()
            .join(";");
        cmd.arg("--dslx_path").arg(joined);
    }

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
    tool_exe: P,
    entry_file: &std::path::Path,
    dslx_stdlib_path: Option<&std::path::Path>,
    additional_search_paths: &[&std::path::Path],
    test_filter: &str,
) -> BoolPropertyResult {
    prove_quickcheck_with_exe_internal(
        tool_exe.as_ref(),
        entry_file,
        dslx_stdlib_path,
        additional_search_paths,
        test_filter,
    )
}

/// Prove a single DSLX quickcheck function by locating `prove_quickcheck_main`
/// in `tool_dir`.
pub fn prove_dslx_quickcheck_with_tool_dir<P: AsRef<std::path::Path>>(
    tool_dir: P,
    entry_file: &std::path::Path,
    dslx_stdlib_path: Option<&std::path::Path>,
    additional_search_paths: &[&std::path::Path],
    test_filter: &str,
) -> BoolPropertyResult {
    let exe = tool_dir.as_ref().join("prove_quickcheck_main");
    if !exe.exists() {
        return BoolPropertyResult::ToolchainDisproved(format!(
            "prove_quickcheck_main not found in {}",
            tool_dir.as_ref().display()
        ));
    }
    prove_quickcheck_with_exe_internal(
        &exe,
        entry_file,
        dslx_stdlib_path,
        additional_search_paths,
        test_filter,
    )
}

/// Reads `XLSYNTH_TOOLS` to locate the toolchain directory and runs a single
/// quickcheck.
pub fn prove_dslx_quickcheck_via_toolchain(
    entry_file: &std::path::Path,
    dslx_stdlib_path: Option<&std::path::Path>,
    additional_search_paths: &[&std::path::Path],
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
