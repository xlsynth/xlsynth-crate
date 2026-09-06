// SPDX-License-Identifier: Apache-2.0

//! Shared path resolution and bounded execution of external tools.

use std::ffi::OsStr;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Output};
use std::time::{Duration, Instant};

/// Distinguishes tool execution limits from errors in the supplied design.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolFailureKind {
    Failure,
    Timeout,
    ResourceExhausted,
    /// SIGKILL does not identify its sender; in particular, it is not proof of
    /// OOM.
    Killed,
}

impl ToolFailureKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Failure => "failure",
            Self::Timeout => "timeout",
            Self::ResourceExhausted => "resource-exhausted",
            Self::Killed => "killed",
        }
    }
}

/// Captured tool failure whose category survives added invocation context.
#[derive(Clone, Debug)]
pub struct ToolError {
    pub kind: ToolFailureKind,
    pub tool: String,
    message: String,
}

impl ToolError {
    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            kind: ToolFailureKind::Failure,
            tool: "harness".into(),
            message: message.into(),
        }
    }

    pub fn timeout(tool: &str, timeout: Duration) -> Self {
        Self {
            kind: ToolFailureKind::Timeout,
            tool: tool.into(),
            message: format!("{tool} exceeded {timeout:?}"),
        }
    }

    /// A resource-limited or externally killed child has no semantic verdict.
    pub fn is_resource_failure(&self) -> bool {
        self.kind != ToolFailureKind::Failure
    }

    pub fn reason_key(&self) -> String {
        format!("{}:{}", self.tool, self.kind.as_str())
    }

    pub fn with_context(mut self, context: impl std::fmt::Display) -> Self {
        self.message = format!("{}\n{context}", self.message);
        self
    }

    /// Classifies only unsuccessful exits; ordinary diagnostics stay fatal.
    pub fn from_exit_status(tool: &str, status: ExitStatus, stdout: &str, stderr: &str) -> Self {
        let mut kind = ToolFailureKind::Failure;
        #[cfg(unix)]
        match status.signal() {
            Some(libc::SIGKILL) => kind = ToolFailureKind::Killed,
            Some(libc::SIGXCPU | libc::SIGXFSZ) => kind = ToolFailureKind::ResourceExhausted,
            _ => { /* Other signals can expose tool or input bugs, not resource limits. */ }
        }
        if !status.success() {
            if resource_diagnostic(stdout) || resource_diagnostic(stderr) {
                kind = ToolFailureKind::ResourceExhausted;
            } else if status.code() == Some(137) {
                // Shell tool wrappers conventionally encode SIGKILL as 128 + 9.
                kind = ToolFailureKind::Killed;
            } else if stdout.lines().chain(stderr.lines()).any(|line| {
                let line = line.trim().to_ascii_lowercase();
                line.starts_with("error: abc:") && line.contains("return code 137")
            }) {
                kind = ToolFailureKind::Killed;
            }
        }
        Self {
            kind,
            tool: tool.into(),
            message: format!("{tool} failed ({status})\n{stdout}\n{stderr}"),
        }
    }

    /// Only process creation errors known to represent exhaustion are
    /// recoverable.
    pub fn spawn(tool: &str, error: std::io::Error) -> Self {
        let mut kind = ToolFailureKind::Failure;
        #[cfg(unix)]
        if matches!(error.raw_os_error(), Some(libc::ENOMEM | libc::EAGAIN)) {
            kind = ToolFailureKind::ResourceExhausted;
        }
        Self {
            kind,
            tool: tool.into(),
            message: format!("{tool}: {error}"),
        }
    }
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ToolError {}

impl From<String> for ToolError {
    fn from(message: String) -> Self {
        Self::failure(message)
    }
}

impl From<&str> for ToolError {
    fn from(message: &str) -> Self {
        Self::failure(message)
    }
}

impl From<std::io::Error> for ToolError {
    fn from(error: std::io::Error) -> Self {
        Self::failure(error.to_string())
    }
}

/// Matches diagnostic lines, not arbitrary occurrences in echoed RTL/scripts.
fn resource_diagnostic(text: &str) -> bool {
    text.lines().any(|line| {
        let line = line.trim().to_ascii_lowercase();
        let diagnostic = line.starts_with("error:")
            || line.starts_with("abc:")
            || line.starts_with("fatal:")
            || line.starts_with("llvm error:")
            || line.starts_with("what():")
            || line.starts_with("terminate called")
            || line.starts_with("out of memory")
            || line.starts_with("std::bad_alloc")
            || line.starts_with("memory allocation failed");
        diagnostic
            && [
                "std::bad_alloc",
                "out of memory",
                "cannot allocate memory",
                "memory allocation failed",
                "memory allocation failure",
                "failed to allocate memory",
                "resource temporarily unavailable",
                "no space left on device",
                "file size limit exceeded",
                "cpu time limit exceeded",
            ]
            .iter()
            .any(|message| line.contains(message))
    })
}

/// Resolves a tool name or explicit path before a caller changes directory.
///
/// Preserves symlinks because tool wrappers may dispatch based on the
/// invocation path rather than the executable's canonical path.
pub fn resolve_executable(program: &Path) -> Result<PathBuf, String> {
    resolve_executable_with_path(program, &std::env::var_os("PATH").unwrap_or_default())
}

/// Resolves against an explicit search path without changing process state.
fn resolve_executable_with_path(program: &Path, search_path: &OsStr) -> Result<PathBuf, String> {
    let candidate = if program.components().count() > 1 || program.is_absolute() {
        program.to_owned()
    } else {
        std::env::split_paths(search_path)
            .map(|directory| directory.join(program))
            .find(|path| is_executable_file(path))
            .ok_or_else(|| format!("executable `{}` not found on PATH", program.display()))?
    };
    if !candidate.is_file() {
        return Err(format!("executable is not a file: {}", candidate.display()));
    }
    if !is_executable_file(&candidate) {
        return Err(format!("file is not executable: {}", candidate.display()));
    }
    std::path::absolute(&candidate)
        .map_err(|error| format!("resolve executable `{}`: {error}", candidate.display()))
}

/// Checks regular-file and, on Unix, executable permission bits through
/// symlinks.
fn is_executable_file(path: &Path) -> bool {
    path.metadata().is_ok_and(|metadata| {
        if !metadata.is_file() {
            return false;
        }
        #[cfg(unix)]
        {
            metadata.permissions().mode() & 0o111 != 0
        }
        #[cfg(not(unix))]
        {
            true
        }
    })
}

/// Runs a tool with file-backed diagnostics and a wall-clock timeout.
///
/// On Unix, the tool and its children share a fresh process group which is
/// terminated on completion, failure, or timeout. Callers supply a temporary
/// directory and a unique label for the captured stdout/stderr files.
pub fn run_checked(
    command: &mut Command,
    directory: &Path,
    label: &str,
    timeout: Duration,
) -> Result<String, String> {
    run_checked_detailed(command, directory, label, timeout).map_err(|error| error.to_string())
}

/// Runs a tool while preserving resource-failure categories for its caller.
pub fn run_checked_detailed(
    command: &mut Command,
    directory: &Path,
    label: &str,
    timeout: Duration,
) -> Result<String, ToolError> {
    let result = run_with_timeout_detailed(command, directory, label, timeout)?;
    if result.status.success() {
        String::from_utf8(result.stdout).map_err(|error| ToolError::failure(error.to_string()))
    } else {
        Err(ToolError::from_exit_status(
            label,
            result.status,
            &String::from_utf8_lossy(&result.stdout),
            &String::from_utf8_lossy(&result.stderr),
        ))
    }
}

/// Captures a bounded tool invocation, including an unsuccessful exit status.
pub fn run_with_timeout(
    command: &mut Command,
    directory: &Path,
    label: &str,
    timeout: Duration,
) -> Result<Output, String> {
    run_with_timeout_detailed(command, directory, label, timeout).map_err(|error| error.to_string())
}

/// Captures a tool without converting timeout/resource errors into strings.
pub fn run_with_timeout_detailed(
    command: &mut Command,
    directory: &Path,
    label: &str,
    timeout: Duration,
) -> Result<Output, ToolError> {
    let stdout_path = directory.join(format!("{label}.stdout"));
    let stderr_path = directory.join(format!("{label}.stderr"));
    command.stdout(std::fs::File::create(&stdout_path).map_err(|e| e.to_string())?);
    command.stderr(std::fs::File::create(&stderr_path).map_err(|e| e.to_string())?);
    set_process_group(command);
    let mut child = command.spawn().map_err(|e| ToolError::spawn(label, e))?;
    let start = Instant::now();
    let outcome = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => { /* The external tool is still running. */ }
            Err(error) => break Err(ToolError::failure(format!("{label}: {error}"))),
        }
        if start.elapsed() >= timeout {
            break Err(ToolError::timeout(label, timeout));
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    kill_process_group(&mut child);
    let stdout = std::fs::read(&stdout_path).map_err(|e| e.to_string())?;
    let stderr = std::fs::read(&stderr_path).map_err(|e| e.to_string())?;
    match outcome {
        Ok(status) => Ok(Output {
            status,
            stdout,
            stderr,
        }),
        Err(error) => Err(error.with_context(format!(
            "{}\n{}",
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr)
        ))),
    }
}

/// Gives a tool its own Unix process group before spawning it.
pub fn set_process_group(command: &mut Command) {
    #[cfg(unix)]
    {
        command.process_group(0);
    }
    #[cfg(not(unix))]
    let _ = command;
}

/// Terminates a tool's process group and reaps the direct child.
pub fn kill_process_group(child: &mut Child) {
    #[cfg(unix)]
    {
        // The group may still contain descendants after its leader exits.
        // Only use this with children spawned via set_process_group.
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::symlink;

    use super::*;

    /// Creates a tool wrapper that dispatches only under its symlink's name.
    fn create_dispatch_wrapper(directory: &Path) -> PathBuf {
        let wrapper = directory.join("tool-wrapper");
        std::fs::write(
            &wrapper,
            r#"#!/bin/sh
case "${0##*/}" in
  example-tool) printf 'dispatched' ;;
  *) exit 7 ;;
esac
"#,
        )
        .unwrap();
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
        let bin = directory.join("bin");
        std::fs::create_dir(&bin).unwrap();
        let tool = bin.join("example-tool");
        symlink("../tool-wrapper", &tool).unwrap();
        tool
    }

    #[test]
    fn preserves_symlinked_tool_dispatch_for_explicit_and_search_paths() {
        let directory = tempfile::tempdir().unwrap();
        let tool = create_dispatch_wrapper(directory.path());
        let execution_directory = tempfile::tempdir().unwrap();
        let search_path = directory.path().join("bin");
        for resolved in [
            resolve_executable(&tool).unwrap(),
            resolve_executable_with_path(Path::new("example-tool"), search_path.as_os_str())
                .unwrap(),
        ] {
            assert_eq!(resolved, tool);
            assert_eq!(
                run_checked(
                    Command::new(resolved).current_dir(execution_directory.path()),
                    execution_directory.path(),
                    "dispatch",
                    Duration::from_secs(5),
                )
                .unwrap(),
                "dispatched"
            );
        }
    }

    #[test]
    fn resolves_relative_tool_and_search_paths_before_changing_directory() {
        let current_dir = std::env::current_dir().unwrap();
        let directory = tempfile::tempdir_in(&current_dir).unwrap();
        let tool = create_dispatch_wrapper(directory.path());
        let relative_tool = tool.strip_prefix(&current_dir).unwrap();
        let execution_directory = tempfile::tempdir().unwrap();
        for resolved in [
            resolve_executable(relative_tool).unwrap(),
            resolve_executable_with_path(
                Path::new("example-tool"),
                relative_tool.parent().unwrap().as_os_str(),
            )
            .unwrap(),
        ] {
            assert_eq!(resolved, tool);
            assert_eq!(
                run_checked(
                    Command::new(resolved).current_dir(execution_directory.path()),
                    execution_directory.path(),
                    "relative-dispatch",
                    Duration::from_secs(5),
                )
                .unwrap(),
                "dispatched"
            );
        }
    }

    #[test]
    fn preserves_parent_components_after_symlinked_directories() {
        let directory = tempfile::tempdir().unwrap();
        let tool = create_dispatch_wrapper(directory.path());
        std::fs::create_dir(directory.path().join("bin/child")).unwrap();
        symlink("bin/child", directory.path().join("alias")).unwrap();
        let invocation_path = directory.path().join("alias/../example-tool");
        assert_eq!(
            resolve_executable(&invocation_path).unwrap(),
            invocation_path
        );
        assert_eq!(
            std::fs::canonicalize(&invocation_path).unwrap(),
            std::fs::canonicalize(tool).unwrap()
        );
        assert_eq!(
            run_checked(
                &mut Command::new(resolve_executable(&invocation_path).unwrap()),
                directory.path(),
                "parent-dispatch",
                Duration::from_secs(5),
            )
            .unwrap(),
            "dispatched"
        );
    }

    #[test]
    fn rejects_missing_paths_directories_and_non_executable_files() {
        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("missing");
        for path in [missing.as_path(), directory.path()] {
            assert_eq!(
                resolve_executable(path).unwrap_err(),
                format!("executable is not a file: {}", path.display())
            );
        }
        let non_executable = directory.path().join("example-tool");
        std::fs::write(&non_executable, "not an executable").unwrap();
        std::fs::set_permissions(&non_executable, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert_eq!(
            resolve_executable(&non_executable).unwrap_err(),
            format!("file is not executable: {}", non_executable.display())
        );
        assert_eq!(
            resolve_executable_with_path(Path::new("example-tool"), directory.path().as_os_str())
                .unwrap_err(),
            "executable `example-tool` not found on PATH"
        );
        let tool = create_dispatch_wrapper(directory.path());
        let search_path = std::env::join_paths([directory.path(), tool.parent().unwrap()]).unwrap();
        assert_eq!(
            resolve_executable_with_path(Path::new("example-tool"), &search_path).unwrap(),
            tool
        );
    }

    #[test]
    fn captures_output_and_diagnostics_on_failure_and_timeout() {
        let directory = tempfile::tempdir().unwrap();
        assert_eq!(
            run_checked(
                Command::new("sh").args(["-c", "printf success"]),
                directory.path(),
                "success",
                Duration::from_secs(5)
            )
            .unwrap(),
            "success"
        );
        let error = run_checked(
            Command::new("sh").args(["-c", "printf output; printf diagnostic >&2; exit 7"]),
            directory.path(),
            "failure",
            Duration::from_secs(5),
        )
        .unwrap_err();
        assert!(
            error.contains("output") && error.contains("diagnostic") && error.contains('7'),
            "{error}"
        );
        let error = run_checked(
            Command::new("sh").args(["-c", "printf started; sleep 30 & wait"]),
            directory.path(),
            "timeout",
            Duration::from_millis(100),
        )
        .unwrap_err();
        assert!(
            error.contains("exceeded") && error.contains("started"),
            "{error}"
        );
    }

    #[test]
    fn classifies_only_explicit_resource_interruptions_as_inconclusive() {
        let directory = tempfile::tempdir().unwrap();
        let allocation = run_checked_detailed(
            Command::new("sh").args(["-c", "printf 'error: std::bad_alloc\\n' >&2; exit 1"]),
            directory.path(),
            "allocation",
            Duration::from_secs(5),
        )
        .unwrap_err();
        assert_eq!(allocation.kind, ToolFailureKind::ResourceExhausted);

        let killed = run_checked_detailed(
            Command::new("sh").args(["-c", "kill -9 $$"]),
            directory.path(),
            "killed",
            Duration::from_secs(5),
        )
        .unwrap_err();
        assert_eq!(killed.kind, ToolFailureKind::Killed);

        let ordinary = run_checked_detailed(
            Command::new("sh").args(["-c", r#"printf 'assign x = "out of memory";\n'; exit 1"#]),
            directory.path(),
            "ordinary",
            Duration::from_secs(5),
        )
        .unwrap_err();
        assert_eq!(ordinary.kind, ToolFailureKind::Failure);

        let timeout = run_checked_detailed(
            Command::new("sh").args(["-c", "sleep 30"]),
            directory.path(),
            "timeout-detailed",
            Duration::from_millis(100),
        )
        .unwrap_err();
        assert_eq!(timeout.kind, ToolFailureKind::Timeout);
    }

    #[test]
    fn cleans_up_descendants_even_after_the_leader_exits() {
        let directory = tempfile::tempdir().unwrap();
        run_checked(
            Command::new("sh")
                .current_dir(directory.path())
                .args(["-c", "(sleep 0.3; touch leaked) & exit 0"]),
            directory.path(),
            "orphan",
            Duration::from_secs(5),
        )
        .unwrap();
        std::thread::sleep(Duration::from_millis(600));
        assert!(!directory.path().join("leaked").exists());
    }
}
