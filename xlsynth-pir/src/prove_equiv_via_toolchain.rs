// SPDX-License-Identifier: Apache-2.0

//! IR equivalence via external toolchain `check_ir_equivalence_main`.
//!
//! Note that this lives in the `xlsynth-pir` crate to avoid a circular
//! dependency with the solver code -- we use this in some tests and it is
//! assumed that the $XLSYNTH_TOOLS are available for use in testing generally
//! throughout the xlsynth-crate codebase.

use std::io::{Read, Seek, SeekFrom};
use std::process::Stdio;
use std::time::{Duration, Instant};

use crate::ir::Fn;

/// Environment variable overriding the XLS checker watchdog in milliseconds.
pub const TOOLCHAIN_EQUIV_TIMEOUT_MS_ENV: &str = "XLSYNTH_TOOLCHAIN_EQUIV_TIMEOUT_MS";

/// Default watchdog for explicit XLS checker invocations.
pub const DEFAULT_TOOLCHAIN_EQUIV_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug, PartialEq, Clone)]
pub enum ToolchainEquivResult {
    Proved,
    Disproved(String),
    TimedOutOrInterrupted(String),
    Error(String),
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
) -> ToolchainEquivResult {
    prove_ir_pkg_equiv_with_tool_exe_and_timeout(
        lhs_pkg_text,
        rhs_pkg_text,
        top,
        tool_exe,
        configured_toolchain_equiv_timeout(),
    )
}

/// Proves equivalence with an explicit watchdog timeout.
pub fn prove_ir_pkg_equiv_with_tool_exe_and_timeout<P: AsRef<std::path::Path>>(
    lhs_pkg_text: &str,
    rhs_pkg_text: &str,
    top: Option<&str>,
    tool_exe: P,
    timeout: Duration,
) -> ToolchainEquivResult {
    let exe = tool_exe.as_ref();
    if !exe.exists() {
        return ToolchainEquivResult::Error(format!("tool not found: {}", exe.display()));
    }
    let lhs_tmp = tempfile::NamedTempFile::new().unwrap();
    let rhs_tmp = tempfile::NamedTempFile::new().unwrap();
    if std::fs::write(lhs_tmp.path(), lhs_pkg_text).is_err()
        || std::fs::write(rhs_tmp.path(), rhs_pkg_text).is_err()
    {
        return ToolchainEquivResult::Error("failed to write temp files".to_string());
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
    let mut stdout_file = tempfile::tempfile().unwrap();
    let mut stderr_file = tempfile::tempfile().unwrap();
    cmd.stdout(Stdio::from(stdout_file.try_clone().unwrap()));
    cmd.stderr(Stdio::from(stderr_file.try_clone().unwrap()));
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => return ToolchainEquivResult::Error(format!("spawn failed: {}", e)),
    };
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = read_spooled_output(&mut stdout_file);
                let stderr = read_spooled_output(&mut stderr_file);
                return classify_tool_output(status, &stdout, &stderr);
            }
            Ok(None) if start.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return ToolchainEquivResult::TimedOutOrInterrupted(format!(
                    "tool timed out after {} ms",
                    timeout.as_millis()
                ));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return ToolchainEquivResult::Error(format!("wait failed: {}", e));
            }
        }
    }
}

fn configured_toolchain_equiv_timeout() -> Duration {
    std::env::var(TOOLCHAIN_EQUIV_TIMEOUT_MS_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_TOOLCHAIN_EQUIV_TIMEOUT)
}

fn read_spooled_output(file: &mut std::fs::File) -> Vec<u8> {
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut output = Vec::new();
    file.read_to_end(&mut output).unwrap();
    output
}

fn classify_tool_output(
    status: std::process::ExitStatus,
    stdout: &[u8],
    stderr: &[u8],
) -> ToolchainEquivResult {
    if status.success() {
        return ToolchainEquivResult::Proved;
    }
    // Include stderr snippet to aid debugging, but do not print directly.
    let mut msg = format!("tool exited with status {}", status);
    if !stdout.is_empty() {
        let stdout_str = String::from_utf8_lossy(stdout);
        msg.push_str(": ");
        msg.push_str(&stdout_str);
    }
    if !stderr.is_empty() {
        let stderr_str = String::from_utf8_lossy(stderr);
        // Truncate to a reasonable length to avoid huge error strings.
        let snippet: String = stderr_str.chars().take(512).collect();
        msg.push_str(": ");
        msg.push_str(&snippet);
    }
    let lowered = msg.to_ascii_lowercase();
    if lowered.contains("deadline_exceeded")
        || lowered.contains("sigint")
        || lowered.contains("interrupted")
    {
        ToolchainEquivResult::TimedOutOrInterrupted(msg)
    } else {
        ToolchainEquivResult::Disproved(msg)
    }
}

pub fn prove_ir_pkg_equiv_with_tool_dir<P: AsRef<std::path::Path>>(
    lhs_pkg_text: &str,
    rhs_pkg_text: &str,
    top: Option<&str>,
    tool_dir: P,
) -> ToolchainEquivResult {
    let exe = tool_dir.as_ref().join("check_ir_equivalence_main");
    if !exe.exists() {
        return ToolchainEquivResult::Error(format!(
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
) -> ToolchainEquivResult {
    let exe = tool_dir.as_ref().join("check_ir_equivalence_main");
    if !exe.exists() {
        return ToolchainEquivResult::Error(format!(
            "check_ir_equivalence_main not found in {}",
            tool_dir.as_ref().display()
        ));
    }
    let lhs_pkg = format!("package lhs\n\ntop {}\n", lhs.to_string());
    let rhs_pkg = format!("package rhs\n\ntop {}\n", rhs.to_string());
    prove_ir_pkg_equiv_with_tool_exe(&lhs_pkg, &rhs_pkg, None, &exe)
}

/// As above, but where the caller has already done `lhs.to_string()` and
/// `rhs.to_string()` on the `ir::Fn` objects.
pub fn prove_ir_fn_strings_equiv_via_toolchain(lhs: &str, rhs: &str) -> ToolchainEquivResult {
    let lhs_pkg = format!("package lhs\n\ntop {}\n", lhs);
    let rhs_pkg = format!("package rhs\n\ntop {}\n", rhs);
    match std::env::var("XLSYNTH_TOOLS") {
        Ok(dir) => prove_ir_pkg_equiv_with_tool_dir(&lhs_pkg, &rhs_pkg, None, dir),
        Err(_) => ToolchainEquivResult::Error(
            "XLSYNTH_TOOLS is not set; cannot run toolchain equivalence".to_string(),
        ),
    }
}

/// Convenience wrapper: reads `XLSYNTH_TOOLS` env var to locate the toolchain.
pub fn prove_ir_fn_equiv_via_toolchain(lhs: &Fn, rhs: &Fn) -> ToolchainEquivResult {
    prove_ir_fn_strings_equiv_via_toolchain(&lhs.to_string(), &rhs.to_string())
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    #[test]
    fn toolchain_checker_watchdog_interrupts_slow_process() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("slow-checker");
        std::fs::write(&exe, "#!/bin/sh\nexec sleep 10\n").unwrap();
        let mut permissions = std::fs::metadata(&exe).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&exe, permissions).unwrap();

        let result = prove_ir_pkg_equiv_with_tool_exe_and_timeout(
            "package lhs\n",
            "package rhs\n",
            None,
            exe,
            Duration::from_millis(10),
        );
        assert!(matches!(
            result,
            ToolchainEquivResult::TimedOutOrInterrupted(msg)
                if msg.contains("tool timed out after 10 ms")
        ));
    }
}
