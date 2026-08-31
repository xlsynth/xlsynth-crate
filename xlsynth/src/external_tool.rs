// SPDX-License-Identifier: Apache-2.0

//! Shared path resolution and bounded execution of external tools.

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
pub fn resolve_executable(program: &Path) -> Result<PathBuf, String> {
    let candidate = if program.components().count() > 1 || program.is_absolute() {
        program.to_owned()
    } else {
        std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
            .map(|directory| directory.join(program))
            .find(|path| path.is_file())
            .ok_or_else(|| format!("executable `{}` not found on PATH", program.display()))?
    };
    if !candidate.is_file() {
        return Err(format!("executable is not a file: {}", candidate.display()));
    }
    candidate
        .canonicalize()
        .map_err(|error| format!("resolve executable `{}`: {error}", candidate.display()))
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
    use super::*;

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
