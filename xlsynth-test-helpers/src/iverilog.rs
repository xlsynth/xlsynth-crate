// SPDX-License-Identifier: Apache-2.0

//! Validated Icarus compiler/runtime configuration shared by RTL tests.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use xlsynth::external_tool::{resolve_executable, run_checked};

/// An Icarus installation; neither executable is needed to compile this API.
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
                    "required Icarus tool `{name}` at {} is unavailable: {error}",
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

    use super::IcarusToolchain;

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
