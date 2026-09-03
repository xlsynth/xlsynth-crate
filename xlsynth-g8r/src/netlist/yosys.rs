// SPDX-License-Identifier: Apache-2.0

//! Validated Yosys execution and Liberty-backed technology mapping.
//! Syntax checks and proofs use `YosysToolchain` without any Liberty
//! dependency; mapping uses `YosysEnvironment`, optionally with a parsed
//! `YosysMappingContext`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use crate::liberty::parser::{LibertyPayloadOptions, parse_liberty_files_with_payload_options};
use crate::liberty_model::{Library, SequentialKind};
use crate::netlist::gv_eval::{
    GvEvalOptions, load_labeled_netlist_aig_with_liberty,
    load_labeled_sequential_netlist_aig_with_liberty,
};

use xlsynth::external_tool::{ToolError, resolve_executable, run_checked, run_checked_detailed};

/// Environment variable naming the Yosys executable used for external mapping.
pub const YOSYS_PATH_ENV: &str = "XLSYNTH_YOSYS_PATH";

/// Environment variable containing comma-separated Liberty paths for Yosys.
pub const LIBERTY_FILES_ENV: &str = "XLSYNTH_LIBERTY_FILES";

const LIBERTY_FILES_HELP: &str = "Set XLSYNTH_LIBERTY_FILES=/path/to/combo.lib,/path/to/seq.lib to comma-separated installed Liberty files from one compatible library/corner; include flip-flop cells for sequential mapping.";

/// A validated Yosys executable, independent of Liberty or mapping
/// configuration.
#[derive(Clone, Debug)]
pub struct YosysToolchain {
    path: PathBuf,
}

impl YosysToolchain {
    /// Resolves and probes the executable before changing working directories.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = resolve_executable(path.as_ref()).map_err(|error| format!("Yosys {error}"))?;
        let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
        run_checked(
            Command::new(&path).arg("-V"),
            directory.path(),
            "yosys-version",
            Duration::from_secs(10),
        )?;
        Ok(Self { path })
    }

    /// Uses XLSYNTH_YOSYS_PATH when set, otherwise resolves yosys on PATH.
    pub fn from_env() -> Result<Self, String> {
        Self::new(std::env::var_os(YOSYS_PATH_ENV).unwrap_or_else(|| "yosys".into()))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Runs a caller-supplied script with captured diagnostics and
    /// process-group cleanup. Script paths are relative to directory; no
    /// Liberty is required.
    pub fn run_script(
        &self,
        directory: &Path,
        program: &str,
        timeout: Duration,
    ) -> Result<String, ToolError> {
        run_checked_detailed(
            Command::new(&self.path)
                .current_dir(directory)
                .args(["-Q", "-p", program]),
            directory,
            "yosys",
            timeout,
        )
        .map_err(|error| error.with_context(format_yosys_invocation_context(&self.path, program)))
    }
}

/// Frontend used to parse the input RTL, independently of mapping mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YosysInputLanguage {
    Verilog,
    SystemVerilog,
}

/// Whether synthesis must also map state elements using dfflibmap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YosysMappingKind {
    Combinational,
    Sequential,
}

/// Validated external Yosys executable and Liberty-library configuration.
pub struct YosysEnvironment {
    toolchain: YosysToolchain,
    liberty_files: YosysLibertyFileSet,
}

impl YosysEnvironment {
    /// Validates an explicit external Yosys setup.
    pub fn new<P: AsRef<Path>>(
        yosys_path: P,
        liberty_files: YosysLibertyFileSet,
    ) -> Result<Self, String> {
        Ok(Self {
            toolchain: YosysToolchain::new(yosys_path)?,
            liberty_files,
        })
    }

    /// Reads and validates the external Yosys setup from environment variables.
    pub fn from_env() -> Result<Self, String> {
        Self::new(yosys_path_from_env()?, YosysLibertyFileSet::from_env()?)
    }

    /// Returns the validated Yosys executable path.
    pub fn yosys_path(&self) -> &Path {
        self.toolchain.path()
    }

    /// Returns the validated Liberty files used by Yosys and ABC.
    pub fn liberty_files(&self) -> &YosysLibertyFileSet {
        &self.liberty_files
    }

    /// Maps one combinational Verilog module with this external Yosys setup.
    pub fn synthesize_verilog_to_gv(
        &self,
        verilog: &str,
        top_module: &str,
    ) -> Result<String, String> {
        self.synthesize_verilog_to_gv_detailed(verilog, top_module)
            .map_err(|error| error.to_string())
    }

    /// Maps combinational RTL while preserving external resource-failure
    /// categories.
    pub fn synthesize_verilog_to_gv_detailed(
        &self,
        verilog: &str,
        top_module: &str,
    ) -> Result<String, ToolError> {
        self.synthesize_to_gv(
            verilog,
            top_module,
            YosysInputLanguage::Verilog,
            YosysMappingKind::Combinational,
        )
    }

    /// Maps one sequential Verilog module with this external Yosys setup.
    ///
    /// Internal Yosys FFs are mapped through dfflibmap before ABC maps the
    /// remaining combinational logic.
    pub fn synthesize_sequential_verilog_to_gv(
        &self,
        verilog: &str,
        top_module: &str,
    ) -> Result<String, String> {
        self.synthesize_sequential_verilog_to_gv_detailed(verilog, top_module)
            .map_err(|error| error.to_string())
    }

    /// Maps sequential RTL while preserving external resource-failure
    /// categories.
    pub fn synthesize_sequential_verilog_to_gv_detailed(
        &self,
        verilog: &str,
        top_module: &str,
    ) -> Result<String, ToolError> {
        self.synthesize_to_gv(
            verilog,
            top_module,
            YosysInputLanguage::SystemVerilog,
            YosysMappingKind::Sequential,
        )
    }

    /// Maps RTL with an explicit frontend and mapping mode, preserving typed
    /// tool failures. Uses the same synthesis passes for both input languages.
    pub fn synthesize_to_gv(
        &self,
        source: &str,
        top_module: &str,
        language: YosysInputLanguage,
        kind: YosysMappingKind,
    ) -> Result<String, ToolError> {
        let program =
            render_synthesis_program(top_module, self.liberty_files.paths(), language, kind);
        run_yosys_synthesis_program(&self.toolchain, source, top_module, &program, kind)
    }
}

/// Parsed Liberty semantics and matching mapping configuration. This is
/// reusable library state; callers decide whether to cache it or run a startup
/// probe.
pub struct YosysMappingContext {
    pub liberty: Library,
    pub yosys: YosysEnvironment,
}

impl YosysMappingContext {
    /// Loads Liberty cell semantics without timing/power payloads.
    pub fn new(yosys: YosysEnvironment) -> Result<Self, String> {
        let liberty = parse_liberty_files_with_payload_options(
            yosys.liberty_files().paths(),
            LibertyPayloadOptions {
                include_timing: false,
                include_power: false,
            },
        )
        .map_err(|error| {
            format!("parse Liberty inputs configured by XLSYNTH_LIBERTY_FILES: {error}")
        })?;
        if liberty.cells.is_empty() {
            return Err(
            "XLSYNTH_LIBERTY_FILES contains no cells; supply installed standard-cell Liberty files"
                .into(),
        );
        }
        Ok(Self { liberty, yosys })
    }

    /// Requires an explicitly configured mapper and installed Liberty files.
    pub fn from_env() -> Result<Self, String> {
        Self::new(YosysEnvironment::from_env()?)
    }

    /// Exercises mapping and netlist import with a small infrastructure probe.
    /// This does not impose any fuzzing error-handling or caching policy.
    pub fn preflight(&self, kind: YosysMappingKind) -> Result<(), ToolError> {
        let sequential = kind == YosysMappingKind::Sequential;
        if sequential
            && !self.liberty.cells.iter().any(|cell| {
                cell.sequential
                    .iter()
                    .any(|state| state.kind == SequentialKind::Ff as i32)
            })
        {
            return Err("XLSYNTH_LIBERTY_FILES has no flip-flop cells; include a sequential Liberty file for this target".into());
        }
        let source = if sequential {
            "module preflight(input clk, input a, input b, output reg q); always @(posedge clk) q <= a ^ b; endmodule\n"
        } else {
            "module preflight(input a, input b, output y); assign y = a ^ b; endmodule\n"
        };
        let netlist = if sequential {
            self.yosys
                .synthesize_sequential_verilog_to_gv_detailed(source, "preflight")
        } else {
            self.yosys
                .synthesize_verilog_to_gv_detailed(source, "preflight")
        }
        .map_err(|error| error.with_context("Yosys/ABC startup mapping check failed"))?;
        let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
        let path = directory.path().join("mapped.gv");
        std::fs::write(&path, netlist).map_err(|error| error.to_string())?;
        let options = GvEvalOptions {
            module_name: Some("preflight".into()),
            clock_port_name: sequential.then(|| "clk".into()),
        };
        if sequential {
            load_labeled_sequential_netlist_aig_with_liberty(&path, &self.liberty, &options)
                .map(|_| ())
        } else {
            load_labeled_netlist_aig_with_liberty(&path, &self.liberty, &options).map(|_| ())
        }
        .map_err(|error| {
            ToolError::failure(format!(
                "startup mapped-netlist import failed for XLSYNTH_LIBERTY_FILES: {error}"
            ))
        })
    }
}

/// A validated set of Liberty files passed directly to Yosys and ABC.
pub struct YosysLibertyFileSet {
    paths: Vec<PathBuf>,
}

impl YosysLibertyFileSet {
    /// Reads comma-separated source Liberty paths from the environment.
    pub fn from_env() -> Result<Self, String> {
        std::env::var(LIBERTY_FILES_ENV)
            .map_err(|error| match error {
                std::env::VarError::NotPresent => format!("{LIBERTY_FILES_ENV} is not set"),
                std::env::VarError::NotUnicode(_) => {
                    format!("{LIBERTY_FILES_ENV} must contain UTF-8 paths")
                }
            })
            .and_then(|raw_paths| Self::from_comma_separated_paths(&raw_paths))
            .map_err(|error| format!("{error}. {LIBERTY_FILES_HELP}"))
    }

    /// Validates and canonicalizes source Liberty files while preserving order.
    pub fn new<P: AsRef<Path>>(liberty_files: &[P]) -> Result<Self, String> {
        if liberty_files.is_empty() {
            return Err("Yosys Liberty file set cannot be empty".to_string());
        }

        let mut paths = Vec::with_capacity(liberty_files.len());
        for path in liberty_files {
            let path = path.as_ref();
            if !path.is_file() {
                return Err(format!(
                    "Yosys Liberty input is not a file: {}",
                    path.display()
                ));
            }
            if path.extension().is_some_and(|extension| extension == "7z") {
                return Err(format!(
                    "Yosys Liberty input is an archive; provide installed .lib files: {}",
                    path.display()
                ));
            }
            let canonical_path = path.canonicalize().map_err(|e| {
                format!("canonicalize Yosys Liberty input '{}': {e}", path.display())
            })?;
            paths.push(canonical_path);
        }
        Ok(Self { paths })
    }

    /// Returns source Liberty paths in the order Yosys and ABC should load
    /// them.
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    fn from_comma_separated_paths(raw_paths: &str) -> Result<Self, String> {
        if raw_paths.trim().is_empty() {
            return Err(format!("{LIBERTY_FILES_ENV} is empty"));
        }
        let mut paths = Vec::new();
        for raw_path in raw_paths.split(',') {
            let path = raw_path.trim();
            if path.is_empty() {
                return Err(format!(
                    "{LIBERTY_FILES_ENV} contains an empty comma-separated entry"
                ));
            }
            paths.push(PathBuf::from(path));
        }
        Self::new(&paths)
    }
}

fn yosys_path_from_env() -> Result<PathBuf, String> {
    std::env::var_os(YOSYS_PATH_ENV)
        .map(PathBuf::from)
        .ok_or_else(|| {
            format!("{YOSYS_PATH_ENV} is not set. Set XLSYNTH_YOSYS_PATH=/path/to/yosys; mapping also requires {LIBERTY_FILES_ENV}.")
        })
}

/// Stages source and retrieves the mapped netlist using the shared runner.
fn run_yosys_synthesis_program(
    toolchain: &YosysToolchain,
    verilog: &str,
    top_module: &str,
    yosys_program: &str,
    mapping_kind: YosysMappingKind,
) -> Result<String, ToolError> {
    if !is_simple_yosys_identifier(top_module) {
        return Err(format!("Yosys top module must be a simple identifier: '{top_module}'").into());
    }

    let temp_dir =
        tempfile::tempdir().map_err(|e| format!("create temporary Yosys directory: {e}"))?;
    let input_path = temp_dir.path().join("dut.v");
    let output_path = temp_dir.path().join("mapped.gv");
    std::fs::write(&input_path, verilog)
        .map_err(|e| format!("write temporary Yosys input Verilog: {e}"))?;
    let invocation_context = format_yosys_invocation_context(toolchain.path(), yosys_program);

    toolchain
        .run_script(temp_dir.path(), yosys_program, Duration::from_secs(120))
        .map_err(|error| {
            error.with_context(format!("Yosys {mapping_kind:?} technology mapping"))
        })?;

    std::fs::read_to_string(&output_path).map_err(|e| {
        ToolError::failure(format!(
            "read Yosys mapped netlist '{}': {e}\n{invocation_context}",
            output_path.display()
        ))
    })
}

fn format_yosys_invocation_context(yosys_path: &Path, yosys_program: &str) -> String {
    format!(
        "Yosys executable: {}\nYosys program:\n{}",
        yosys_path.display(),
        yosys_program.trim_end()
    )
}

/// Renders the common mapping script with only frontend and FF mapping varying.
fn render_synthesis_program(
    top_module: &str,
    liberty_paths: &[PathBuf],
    language: YosysInputLanguage,
    kind: YosysMappingKind,
) -> String {
    let read_liberty_commands = liberty_paths
        .iter()
        .map(|path| format!("read_liberty -lib {}", quote_yosys_path(path)))
        .collect::<Vec<_>>()
        .join("\n");
    let liberty_arguments = liberty_paths
        .iter()
        .map(|path| format!("-liberty {}", quote_yosys_path(path)))
        .collect::<Vec<_>>()
        .join(" ");
    let frontend = match language {
        YosysInputLanguage::Verilog => "read_verilog dut.v",
        YosysInputLanguage::SystemVerilog => "read_verilog -sv dut.v",
    };
    let map_registers = match kind {
        YosysMappingKind::Combinational => String::new(),
        YosysMappingKind::Sequential => format!("dfflibmap {liberty_arguments}\nopt\n"),
    };
    format!(
        "{read_liberty_commands}\n\
         {frontend}\n\
         hierarchy -check -top {top_module}\n\
         proc\n\
         flatten\n\
         opt\n\
         techmap\n\
         opt\n\
         {map_registers}abc {liberty_arguments}\n\
         clean -purge\n\
         write_verilog -noattr mapped.gv\n"
    )
}

fn quote_yosys_path(path: &Path) -> String {
    format!(
        "\"{}\"",
        path.to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
    )
}

fn is_simple_yosys_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '$')
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(unix)]
    use xlsynth::external_tool::ToolFailureKind;

    /// Keeps relative-path fixtures out of concurrent source/license scans.
    fn relative_test_directory(cwd: &Path) -> tempfile::TempDir {
        let target = cwd.join("target");
        std::fs::create_dir_all(&target).unwrap();
        tempfile::tempdir_in(target).unwrap()
    }

    #[cfg(unix)]
    #[test]
    fn script_runner_needs_no_liberty_and_preserves_timeout_diagnostics() {
        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("yosys");
        std::fs::write(
            &executable,
            r#"#!/bin/sh
if [ "$1" = "-V" ]; then exit 0; fi
echo diagnostic >&2
sleep 10
"#,
        )
        .unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
        let toolchain = YosysToolchain::new(&executable).unwrap();
        let error = toolchain
            .run_script(directory.path(), "help", Duration::from_secs(1))
            .unwrap_err();
        assert_eq!(error.kind, ToolFailureKind::Timeout);
        assert!(error.to_string().contains("diagnostic"), "{error}");
        assert!(
            error.to_string().contains("Yosys program:\nhelp"),
            "{error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn mapping_context_rejects_empty_cell_libraries() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("empty.lib");
        std::fs::write(&path, "library(empty) {}").unwrap();
        let environment =
            YosysEnvironment::new("/usr/bin/true", YosysLibertyFileSet::new(&[path]).unwrap())
                .unwrap();
        let error = YosysMappingContext::new(environment).err().unwrap();
        assert_eq!(
            error,
            "XLSYNTH_LIBERTY_FILES contains no cells; supply installed standard-cell Liberty files"
        );
    }

    #[test]
    fn input_language_changes_only_the_frontend() {
        let libraries = [PathBuf::from("cells.lib")];
        for kind in [
            YosysMappingKind::Combinational,
            YosysMappingKind::Sequential,
        ] {
            let verilog =
                render_synthesis_program("top", &libraries, YosysInputLanguage::Verilog, kind);
            let system_verilog = render_synthesis_program(
                "top",
                &libraries,
                YosysInputLanguage::SystemVerilog,
                kind,
            );
            assert_eq!(
                system_verilog,
                verilog.replace("read_verilog dut.v", "read_verilog -sv dut.v")
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn relative_yosys_path_survives_synthesis_working_directory_change() {
        let cwd = std::env::current_dir().unwrap();
        let directory = relative_test_directory(&cwd);
        let executable = directory.path().join("yosys");
        std::fs::write(
            &executable,
            "#!/bin/sh\nif [ \"$1\" = \"-V\" ]; then exit 0; fi\nprintf 'mapped\\n' > mapped.gv\n",
        )
        .unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
        let liberty = directory.path().join("cells.lib");
        std::fs::write(&liberty, "library (cells) {}\n").unwrap();
        let environment = YosysEnvironment::new(
            executable.strip_prefix(&cwd).unwrap(),
            YosysLibertyFileSet::new(&[liberty]).unwrap(),
        )
        .unwrap();
        assert!(environment.yosys_path().is_absolute());
        assert_eq!(
            environment
                .synthesize_verilog_to_gv("module top; endmodule", "top")
                .unwrap(),
            "mapped\n"
        );
    }

    #[test]
    fn liberty_file_set_preserves_files_in_order() {
        let source_dir = tempfile::tempdir().unwrap();
        let first = source_dir.path().join("first.lib");
        let second = source_dir.path().join("second.lib");
        std::fs::write(&first, "library (first) {}\n").unwrap();
        std::fs::write(&second, "library (second) {}\n").unwrap();

        let set = YosysLibertyFileSet::new(&[&first, &second]).unwrap();
        assert_eq!(
            set.paths(),
            &[
                first.canonicalize().unwrap(),
                second.canonicalize().unwrap()
            ]
        );
    }

    #[test]
    fn liberty_file_set_canonicalizes_relative_paths() {
        let current_dir = std::env::current_dir().unwrap();
        let source_dir = relative_test_directory(&current_dir);
        let absolute_path = source_dir.path().join("cells.lib");
        std::fs::write(&absolute_path, "library (cells) {}\n").unwrap();
        let relative_path = absolute_path.strip_prefix(&current_dir).unwrap();
        assert!(!relative_path.is_absolute());

        let set = YosysLibertyFileSet::new(&[relative_path]).unwrap();
        assert_eq!(set.paths(), &[absolute_path.canonicalize().unwrap()]);
    }

    #[test]
    fn synthesis_program_passes_each_liberty_file_to_yosys_and_abc() {
        let liberty_paths = vec![PathBuf::from("first.lib"), PathBuf::from("second.lib")];
        let program = render_synthesis_program(
            "top",
            &liberty_paths,
            YosysInputLanguage::Verilog,
            YosysMappingKind::Combinational,
        );
        assert_eq!(
            program,
            "read_liberty -lib \"first.lib\"\n\
             read_liberty -lib \"second.lib\"\n\
             read_verilog dut.v\n\
             hierarchy -check -top top\n\
             proc\n\
             flatten\n\
             opt\n\
             techmap\n\
             opt\n\
             abc -liberty \"first.lib\" -liberty \"second.lib\"\n\
             clean -purge\n\
             write_verilog -noattr mapped.gv\n"
        );
    }

    #[test]
    fn sequential_synthesis_program_maps_ffs_before_abc() {
        let liberty_paths = vec![PathBuf::from("first.lib"), PathBuf::from("second.lib")];
        let program = render_synthesis_program(
            "top",
            &liberty_paths,
            YosysInputLanguage::SystemVerilog,
            YosysMappingKind::Sequential,
        );
        assert_eq!(
            program,
            "read_liberty -lib \"first.lib\"\n\
             read_liberty -lib \"second.lib\"\n\
             read_verilog -sv dut.v\n\
             hierarchy -check -top top\n\
             proc\n\
             flatten\n\
             opt\n\
             techmap\n\
             opt\n\
             dfflibmap -liberty \"first.lib\" -liberty \"second.lib\"\n\
             opt\n\
             abc -liberty \"first.lib\" -liberty \"second.lib\"\n\
             clean -purge\n\
             write_verilog -noattr mapped.gv\n"
        );
    }

    #[test]
    fn invocation_context_includes_executable_and_program() {
        let context =
            format_yosys_invocation_context(Path::new("/path/to/yosys"), "abc -liberty cells.lib");
        assert_eq!(
            context,
            "Yosys executable: /path/to/yosys\nYosys program:\nabc -liberty cells.lib"
        );
    }

    #[test]
    fn liberty_file_set_rejects_an_empty_input_list() {
        let error = YosysLibertyFileSet::new::<&Path>(&[]).err().unwrap();
        assert_eq!(error, "Yosys Liberty file set cannot be empty");
    }

    #[test]
    fn liberty_file_set_parses_comma_separated_paths() {
        let source_dir = tempfile::tempdir().unwrap();
        let first = source_dir.path().join("first.lib");
        let second = source_dir.path().join("second.lib");
        std::fs::write(&first, "library (first) {}\n").unwrap();
        std::fs::write(&second, "library (second) {}\n").unwrap();
        let raw_paths = format!("{}, {}", first.display(), second.display());

        let set = YosysLibertyFileSet::from_comma_separated_paths(&raw_paths).unwrap();
        assert_eq!(
            set.paths(),
            &[
                first.canonicalize().unwrap(),
                second.canonicalize().unwrap()
            ]
        );
    }

    #[test]
    fn liberty_file_set_rejects_an_empty_comma_separated_entry() {
        let error = YosysLibertyFileSet::from_comma_separated_paths("first.lib,")
            .err()
            .unwrap();
        assert_eq!(
            error,
            "XLSYNTH_LIBERTY_FILES contains an empty comma-separated entry"
        );
    }

    #[test]
    fn liberty_file_set_rejects_empty_environment_values() {
        for raw_paths in ["", "  "] {
            let error = YosysLibertyFileSet::from_comma_separated_paths(raw_paths)
                .err()
                .unwrap();
            assert_eq!(error, "XLSYNTH_LIBERTY_FILES is empty");
        }
    }

    #[test]
    fn liberty_file_set_requires_installed_files_instead_of_archives() {
        let source_dir = tempfile::tempdir().unwrap();
        let archive = source_dir.path().join("cells.lib.7z");
        std::fs::write(&archive, []).unwrap();
        let error = YosysLibertyFileSet::new(&[&archive]).err().unwrap();
        assert_eq!(
            error,
            format!(
                "Yosys Liberty input is an archive; provide installed .lib files: {}",
                archive.display()
            )
        );
    }

    #[test]
    fn environment_rejects_a_missing_programmatic_yosys_path() {
        let source_dir = tempfile::tempdir().unwrap();
        let liberty_path = source_dir.path().join("cells.lib");
        std::fs::write(&liberty_path, "library (cells) {}\n").unwrap();
        let liberty_files = YosysLibertyFileSet::new(&[&liberty_path]).unwrap();
        let yosys_path = source_dir.path().join("missing-yosys");

        let error = YosysEnvironment::new(&yosys_path, liberty_files)
            .err()
            .unwrap();
        assert_eq!(
            error,
            format!("Yosys executable is not a file: {}", yosys_path.display())
        );
    }

    #[test]
    fn simple_yosys_identifier_rejects_script_syntax() {
        assert!(is_simple_yosys_identifier("random_block_0"));
        assert!(!is_simple_yosys_identifier(""));
        assert!(!is_simple_yosys_identifier("random-block"));
        assert!(!is_simple_yosys_identifier("top; shell echo nope"));
    }
}
