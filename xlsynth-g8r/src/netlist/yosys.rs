// SPDX-License-Identifier: Apache-2.0

//! External Yosys helpers for combinational Liberty technology mapping.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Environment variable naming the Yosys executable used for external mapping.
pub const YOSYS_PATH_ENV: &str = "XLSYNTH_YOSYS_PATH";

/// Environment variable containing comma-separated Liberty paths for Yosys.
pub const LIBERTY_FILES_ENV: &str = "XLSYNTH_LIBERTY_FILES";

/// Validated external Yosys executable and Liberty-library configuration.
pub struct YosysEnvironment {
    yosys_path: PathBuf,
    liberty_files: YosysLibertyFileSet,
}

impl YosysEnvironment {
    /// Validates an explicit external Yosys setup.
    pub fn new<P: AsRef<Path>>(
        yosys_path: P,
        liberty_files: YosysLibertyFileSet,
    ) -> Result<Self, String> {
        let yosys_path = yosys_path.as_ref().to_path_buf();
        validate_yosys_executable_path(&yosys_path)?;
        Ok(Self {
            yosys_path,
            liberty_files,
        })
    }

    /// Reads and validates the external Yosys setup from environment variables.
    pub fn from_env() -> Result<Self, String> {
        Self::new(yosys_path_from_env()?, YosysLibertyFileSet::from_env()?)
    }

    /// Returns the validated Yosys executable path.
    pub fn yosys_path(&self) -> &Path {
        &self.yosys_path
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
        synthesize_verilog_to_gv_with_yosys(
            &self.yosys_path,
            verilog,
            top_module,
            &self.liberty_files,
        )
    }
}

/// A validated set of Liberty files passed directly to Yosys and ABC.
pub struct YosysLibertyFileSet {
    paths: Vec<PathBuf>,
}

impl YosysLibertyFileSet {
    /// Reads comma-separated source Liberty paths from the environment.
    pub fn from_env() -> Result<Self, String> {
        let raw_paths = std::env::var(LIBERTY_FILES_ENV)
            .map_err(|_| format!("{LIBERTY_FILES_ENV} is not set"))?;
        Self::from_comma_separated_paths(&raw_paths)
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
        if paths.is_empty() {
            return Err(format!("{LIBERTY_FILES_ENV} is empty"));
        }
        Self::new(&paths)
    }
}

fn yosys_path_from_env() -> Result<PathBuf, String> {
    std::env::var_os(YOSYS_PATH_ENV)
        .map(PathBuf::from)
        .ok_or_else(|| format!("{YOSYS_PATH_ENV} is not set"))
}

fn validate_yosys_executable_path(path: &Path) -> Result<(), String> {
    if !path.is_file() {
        return Err(format!(
            "Yosys executable is not a file: {}",
            path.display()
        ));
    }
    let output = Command::new(path)
        .arg("-V")
        .output()
        .map_err(|e| format!("run Yosys executable '{}': {e}", path.display()))?;
    if !output.status.success() {
        return Err(format!(
            "Yosys executable did not run successfully: {}",
            path.display()
        ));
    }
    Ok(())
}

/// Maps one combinational Verilog module through Yosys and its default ABC
/// executable, returning the emitted Liberty-backed gate-level Verilog.
fn synthesize_verilog_to_gv_with_yosys(
    yosys_path: &Path,
    verilog: &str,
    top_module: &str,
    liberty_files: &YosysLibertyFileSet,
) -> Result<String, String> {
    if !is_simple_yosys_identifier(top_module) {
        return Err(format!(
            "Yosys top module must be a simple identifier: '{top_module}'"
        ));
    }

    let temp_dir =
        tempfile::tempdir().map_err(|e| format!("create temporary Yosys directory: {e}"))?;
    let input_path = temp_dir.path().join("dut.v");
    let output_path = temp_dir.path().join("mapped.gv");
    std::fs::write(&input_path, verilog)
        .map_err(|e| format!("write temporary Yosys input Verilog: {e}"))?;
    let yosys_program = render_combo_synthesis_program(top_module, liberty_files.paths());
    let invocation_context = format_yosys_invocation_context(yosys_path, &yosys_program);

    let output = Command::new(yosys_path)
        .current_dir(temp_dir.path())
        .arg("-Q")
        .arg("-p")
        .arg(&yosys_program)
        .output()
        .map_err(|e| format!("run Yosys: {e}\n{invocation_context}"))?;
    if !output.status.success() {
        return Err(format!(
            "Yosys combinational technology mapping failed\n{invocation_context}\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ));
    }

    std::fs::read_to_string(&output_path).map_err(|e| {
        format!(
            "read Yosys mapped netlist '{}': {e}\n{invocation_context}",
            output_path.display()
        )
    })
}

fn format_yosys_invocation_context(yosys_path: &Path, yosys_program: &str) -> String {
    format!(
        "Yosys executable: {}\nYosys program:\n{}",
        yosys_path.display(),
        yosys_program.trim_end()
    )
}

fn render_combo_synthesis_program(top_module: &str, liberty_paths: &[PathBuf]) -> String {
    let read_liberty_commands = liberty_paths
        .iter()
        .map(|path| format!("read_liberty -lib {}", quote_yosys_path(path)))
        .collect::<Vec<_>>()
        .join("\n");
    let abc_liberty_arguments = liberty_paths
        .iter()
        .map(|path| format!("-liberty {}", quote_yosys_path(path)))
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "{read_liberty_commands}\n\
         read_verilog dut.v\n\
         hierarchy -check -top {top_module}\n\
         proc\n\
         flatten\n\
         opt\n\
         techmap\n\
         opt\n\
         abc {abc_liberty_arguments}\n\
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
        let source_dir = tempfile::tempdir_in(&current_dir).unwrap();
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
        let program = render_combo_synthesis_program("top", &liberty_paths);
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
