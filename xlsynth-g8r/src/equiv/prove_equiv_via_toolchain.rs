// SPDX-License-Identifier: Apache-2.0

//! IR equivalence via external toolchain `check_ir_equivalence_main`.

use crate::xls_ir::ir::Fn;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EquivResult {
    Proved,
    OtherProcessError(String),
}

/// Prove equivalence by invoking an external toolchain binary.
/// The `tool_dir` must contain `check_ir_equivalence_main`.
///
/// Note: this function captures the tool's stdout/stderr to avoid printing from
/// a library routine. Callers may choose to log or print the returned info.
pub fn prove_ir_pkg_equiv_with_tool_exe<P: AsRef<std::path::Path>>(
    lhs_pkg_text: &str,
    rhs_pkg_text: &str,
    top: Option<&str>,
    tool_exe: P,
) -> EquivResult {
    let exe = tool_exe.as_ref();
    if !exe.exists() {
        return EquivResult::OtherProcessError(format!("tool not found: {}", exe.display()));
    }
    let lhs_tmp = tempfile::NamedTempFile::new().unwrap();
    let rhs_tmp = tempfile::NamedTempFile::new().unwrap();
    if std::fs::write(lhs_tmp.path(), lhs_pkg_text).is_err()
        || std::fs::write(rhs_tmp.path(), rhs_pkg_text).is_err()
    {
        return EquivResult::OtherProcessError("failed to write temp files".to_string());
    }
    let mut cmd = std::process::Command::new(exe);
    // Note: flag like "--alsologtostderr" can be added for extra logs when
    // debugging, but it is not required for functionality.
    cmd.arg(lhs_tmp.path());
    cmd.arg(rhs_tmp.path());
    if let Some(t) = top {
        cmd.arg("--top");
        cmd.arg(t);
    }
    match cmd.output() {
        Ok(output) if output.status.success() => EquivResult::Proved,
        Ok(output) => {
            // Include stderr snippet to aid debugging, but do not print directly.
            let mut msg = format!("tool exited with status {}", output.status);
            if !output.stderr.is_empty() {
                let stderr_str = String::from_utf8_lossy(&output.stderr);
                // Truncate to a reasonable length to avoid huge error strings.
                let snippet: String = stderr_str.chars().take(512).collect();
                msg.push_str(": ");
                msg.push_str(&snippet);
            }
            EquivResult::OtherProcessError(msg)
        }
        Err(e) => EquivResult::OtherProcessError(format!("spawn failed: {}", e)),
    }
}

pub fn prove_ir_pkg_equiv_with_tool_dir<P: AsRef<std::path::Path>>(
    lhs_pkg_text: &str,
    rhs_pkg_text: &str,
    top: Option<&str>,
    tool_dir: P,
) -> EquivResult {
    let exe = tool_dir.as_ref().join("check_ir_equivalence_main");
    if !exe.exists() {
        return EquivResult::OtherProcessError(format!(
            "check_ir_equivalence_main not found in {}",
            tool_dir.as_ref().display()
        ));
    }
    prove_ir_pkg_equiv_with_tool_exe(lhs_pkg_text, rhs_pkg_text, top, &exe)
}

pub fn prove_ir_fn_equiv_with_tool_dir<P: AsRef<std::path::Path>>(
    lhs: &Fn,
    rhs: &Fn,
    tool_dir: P,
) -> EquivResult {
    let exe = tool_dir.as_ref().join("check_ir_equivalence_main");
    if !exe.exists() {
        return EquivResult::OtherProcessError(format!(
            "check_ir_equivalence_main not found in {}",
            tool_dir.as_ref().display()
        ));
    }
    let lhs_pkg = format!("package lhs\n\ntop {}\n", lhs.to_string());
    let rhs_pkg = format!("package rhs\n\ntop {}\n", rhs.to_string());
    prove_ir_pkg_equiv_with_tool_exe(&lhs_pkg, &rhs_pkg, None, &exe)
}

/// Convenience wrapper: reads `XLSYNTH_TOOLS` env var to locate the toolchain.
pub fn prove_ir_fn_equiv_via_toolchain(lhs: &Fn, rhs: &Fn) -> EquivResult {
    match std::env::var("XLSYNTH_TOOLS") {
        Ok(dir) => prove_ir_fn_equiv_with_tool_dir(lhs, rhs, dir),
        Err(_) => EquivResult::OtherProcessError(
            "XLSYNTH_TOOLS is not set; cannot run toolchain equivalence".to_string(),
        ),
    }
}
