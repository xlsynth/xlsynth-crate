// SPDX-License-Identifier: Apache-2.0

//! Validated iverilog compilation and simulation shared by RTL tests.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use xlsynth::external_tool::{ToolError, resolve_executable, run_checked, run_checked_detailed};

/// An iverilog installation; neither executable is needed to compile this API.
#[derive(Clone, Debug)]
pub struct IcarusToolchain {
    iverilog: PathBuf,
    vvp: PathBuf,
}

impl IcarusToolchain {
    /// Validates both executables before a simulation session is started.
    pub fn new(iverilog: impl AsRef<Path>, vvp: impl AsRef<Path>) -> Result<Self, String> {
        let result = Self {
            iverilog: resolve_executable(iverilog.as_ref())?,
            vvp: resolve_executable(vvp.as_ref())?,
        };
        let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
        for (name, path) in [("iverilog", &result.iverilog), ("vvp", &result.vvp)] {
            run_checked(
                Command::new(path).arg("-V"),
                directory.path(),
                name,
                Duration::from_secs(10),
            )
            .map_err(|error| {
                format!(
                    "required iverilog tool `{name}` at {} is unavailable: {error}",
                    path.display()
                )
            })?;
        }
        Ok(result)
    }

    /// Uses explicit environment overrides, falling back to PATH lookup.
    pub fn from_env() -> Result<Self, String> {
        Self::new(
            std::env::var_os("XLSYNTH_IVERILOG_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("iverilog")),
            std::env::var_os("XLSYNTH_VVP_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("vvp")),
        )
    }

    pub fn iverilog_path(&self) -> &Path {
        &self.iverilog
    }

    pub fn vvp_path(&self) -> &Path {
        &self.vvp
    }

    /// Compiles SystemVerilog sources into `sim.vvp` in the given directory.
    /// Relative source paths are resolved against `directory`; the returned
    /// executable path is absolute. Calls sharing a directory must be serial.
    /// Defines use `NAME` or `NAME=value`, without the `-D` prefix.
    pub fn compile(
        &self,
        directory: &Path,
        sources: &[impl AsRef<Path>],
        top: &str,
        defines: &[&str],
        timeout: Duration,
    ) -> Result<PathBuf, ToolError> {
        let directory = directory.canonicalize().map_err(|e| e.to_string())?;
        let executable = directory.join("sim.vvp");
        let mut command = self.compile_command(defines);
        command
            .current_dir(&directory)
            .arg("-s")
            .arg(top)
            .arg("-o")
            .arg(&executable)
            .args(sources.iter().map(AsRef::as_ref));
        run_checked_detailed(&mut command, &directory, "iverilog", timeout)?;
        Ok(executable)
    }

    /// Checks SystemVerilog syntax, allowing unresolved modules and producing
    /// no simulation executable. Source paths and defines follow `compile`.
    pub fn check_syntax(
        &self,
        directory: &Path,
        sources: &[impl AsRef<Path>],
        defines: &[&str],
        timeout: Duration,
    ) -> Result<(), ToolError> {
        let mut command = self.compile_command(defines);
        command
            .current_dir(directory)
            .args(["-i", "-t", "null"])
            .args(sources.iter().map(AsRef::as_ref));
        run_checked_detailed(&mut command, directory, "iverilog-syntax", timeout).map(|_| ())
    }

    /// Runs a compiled simulation to completion and returns captured stdout.
    /// Relative executable paths are resolved against `directory`. Timeouts
    /// and resource failures retain their categories and captured diagnostics.
    pub fn run(
        &self,
        directory: &Path,
        executable: &Path,
        timeout: Duration,
    ) -> Result<String, ToolError> {
        run_checked_detailed(
            Command::new(&self.vvp)
                .current_dir(directory)
                .arg(executable),
            directory,
            "vvp",
            timeout,
        )
    }

    /// Builds common compiler arguments without selecting an execution mode.
    fn compile_command(&self, defines: &[&str]) -> Command {
        let mut command = Command::new(&self.iverilog);
        command.arg("-g2012");
        for define in defines {
            command.arg(format!("-D{define}"));
        }
        command
    }
}

static ICARUS: OnceLock<Result<IcarusToolchain, String>> = OnceLock::new();

/// Resolves required tools once per test/fuzz process, preserving failures.
pub fn required_iverilog_toolchain() -> Result<&'static IcarusToolchain, &'static str> {
    ICARUS
        .get_or_init(IcarusToolchain::from_env)
        .as_ref()
        .map_err(String::as_str)
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::time::Duration;
    use xlsynth::external_tool::ToolFailureKind;

    use super::IcarusToolchain;

    /// Provides a successful version probe and a caller-selected tool body.
    fn fake_tool(directory: &Path, name: &str, body: &str) -> PathBuf {
        let path = directory.join(name);
        std::fs::write(
            &path,
            format!("#!/bin/sh\nif [ \"$1\" = \"-V\" ]; then exit 0; fi\n{body}\n"),
        )
        .unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[test]
    fn compile_and_run_resolve_relative_paths_and_preserve_arguments() {
        let cwd = std::env::current_dir().unwrap();
        // Keep transient source/diagnostic files out of concurrent SPDX scans.
        let target = cwd.join("target");
        std::fs::create_dir_all(&target).unwrap();
        let directory = tempfile::tempdir_in(target).unwrap();
        let tool = fake_tool(
            directory.path(),
            "tool",
            "test -f source.sv || exit 1\nprintf '%s\\n' \"$@\"",
        );
        std::fs::write(directory.path().join("source.sv"), "").unwrap();
        let relative = directory.path().strip_prefix(&cwd).unwrap();
        let tools = IcarusToolchain::new(relative.join("tool"), &tool).unwrap();
        let executable = tools
            .compile(
                relative,
                &["source.sv"],
                "top",
                &["SYNTHESIS", "VALUE=2"],
                Duration::from_secs(1),
            )
            .unwrap();
        assert_eq!(
            executable,
            directory.path().canonicalize().unwrap().join("sim.vvp")
        );
        assert_eq!(
            std::fs::read_to_string(directory.path().join("iverilog.stdout")).unwrap(),
            format!(
                "-g2012\n-DSYNTHESIS\n-DVALUE=2\n-s\ntop\n-o\n{}\nsource.sv\n",
                executable.display()
            ),
        );
        tools
            .check_syntax(
                relative,
                &["source.sv"],
                &["SYNTHESIS"],
                Duration::from_secs(1),
            )
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(directory.path().join("iverilog-syntax.stdout")).unwrap(),
            "-g2012\n-DSYNTHESIS\n-i\n-t\nnull\nsource.sv\n",
        );
        assert_eq!(
            tools
                .run(relative, Path::new("sim.vvp"), Duration::from_secs(1))
                .unwrap(),
            "sim.vvp\n"
        );
    }

    #[test]
    fn compilation_and_simulation_preserve_failure_categories_and_diagnostics() {
        let directory = tempfile::tempdir().unwrap();
        for (body, expected) in [
            ("echo diagnostic >&2; exit 1", ToolFailureKind::Failure),
            (
                "echo 'out of memory' >&2; exit 1",
                ToolFailureKind::ResourceExhausted,
            ),
            ("echo diagnostic >&2; sleep 10", ToolFailureKind::Timeout),
        ] {
            let tool = fake_tool(directory.path(), "tool", body);
            let tools = IcarusToolchain::new(&tool, &tool).unwrap();
            let timeout = Duration::from_secs(1);
            for error in [
                tools
                    .compile(directory.path(), &["source.sv"], "top", &[], timeout)
                    .unwrap_err(),
                tools
                    .run(directory.path(), Path::new("sim.vvp"), timeout)
                    .unwrap_err(),
            ] {
                assert_eq!(error.kind, expected, "{error}");
                let expected_diagnostic = if expected == ToolFailureKind::ResourceExhausted {
                    "out of memory"
                } else {
                    "diagnostic"
                };
                assert!(error.to_string().contains(expected_diagnostic), "{error}");
            }
        }
    }

    #[test]
    fn validates_both_executables_without_requiring_an_installation() {
        let directory = tempfile::tempdir().unwrap();
        let working = directory.path().join("working");
        let broken = directory.path().join("broken");
        let missing = directory.path().join("missing");
        for (path, script) in [
            (&working, "#!/bin/sh\nexit 0\n"),
            (&broken, "#!/bin/sh\necho broken >&2\nexit 1\n"),
        ] {
            std::fs::write(path, script).unwrap();
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        IcarusToolchain::new(&working, &working).unwrap();
        for (compiler, runtime, name) in
            [(&broken, &working, "iverilog"), (&working, &broken, "vvp")]
        {
            let error = IcarusToolchain::new(compiler, runtime).unwrap_err();
            assert!(error.contains(name) && error.contains("broken"), "{error}");
        }
        assert!(IcarusToolchain::new(&missing, &working).is_err());
        assert!(IcarusToolchain::new(&working, &missing).is_err());
    }
}
