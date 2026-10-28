// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsString;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::LeanError;

/// Captured output of one successful Lean invocation.
pub(crate) struct ProcessOutput {
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

/// Resolves an explicit executable path before changing the working directory.
pub(crate) fn resolve_lean_executable(executable: &Path) -> Result<PathBuf, LeanError> {
    std::path::absolute(executable).map_err(|source| LeanError::Io {
        operation: "resolving Lean executable path",
        path: executable.to_path_buf(),
        source,
    })
}

/// Runs Lean and captures process output in temporary files.
pub(crate) fn run_lean_process(
    executable: &Path,
    working_directory: &Path,
    module_roots: &[&Path],
    operation: &'static str,
    args: &[OsString],
) -> Result<ProcessOutput, LeanError> {
    let lean_path = std::env::join_paths(module_roots).map_err(LeanError::ModuleSearchPath)?;
    let capture = tempfile::Builder::new()
        .prefix("xlsynth-lean-process-")
        .tempdir()
        .map_err(LeanError::TemporaryDirectory)?;
    let stdout_path = capture.path().join("stdout");
    let stderr_path = capture.path().join("stderr");
    let stdout_file = File::create(&stdout_path).map_err(|source| LeanError::Io {
        operation: "creating Lean stdout capture",
        path: stdout_path.clone(),
        source,
    })?;
    let stderr_file = File::create(&stderr_path).map_err(|source| LeanError::Io {
        operation: "creating Lean stderr capture",
        path: stderr_path.clone(),
        source,
    })?;
    let mut child = Command::new(executable)
        .args(args)
        .current_dir(working_directory)
        .env("LEAN_PATH", &lean_path)
        .env("LEAN_SRC_PATH", &lean_path)
        .env_remove("LEAN_SYSROOT")
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .spawn()
        .map_err(|source| LeanError::Io {
            operation: "starting Lean executable",
            path: executable.to_path_buf(),
            source,
        })?;
    let status = match child.wait() {
        Ok(status) => status,
        Err(source) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(LeanError::Io {
                operation: "waiting for Lean executable",
                path: executable.to_path_buf(),
                source,
            });
        }
    };
    let stdout = read_capture(&stdout_path)?;
    let stderr = read_capture(&stderr_path)?;
    if status.success() {
        Ok(ProcessOutput { stdout, stderr })
    } else {
        Err(LeanError::ProcessFailed {
            operation,
            exit_code: status.code(),
            stdout,
            stderr,
        })
    }
}

/// Reads captured process bytes without requiring valid UTF-8 diagnostics.
fn read_capture(path: &Path) -> Result<String, LeanError> {
    let bytes = fs::read(path).map_err(|source| LeanError::Io {
        operation: "reading Lean process output",
        path: path.to_path_buf(),
        source,
    })?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}
