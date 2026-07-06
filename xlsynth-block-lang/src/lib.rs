// SPDX-License-Identifier: Apache-2.0

//! Experimental DSLX-adjacent frontend for explicitly authored XLS blocks.

mod compile;
mod parse;
mod sv;

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub use compile::{ToolRunOutput, compile_block_module, run_tool_with_limits};
pub use sv::{
    Xls53ExternCodegenPlan, apply_xls53_extern_codegen, is_valid_system_verilog_identifier,
    prepare_xls53_extern_codegen, rename_package_block, reorder_system_verilog_module_ports,
    reorder_system_verilog_package_ports,
};

/// Permitted combinational optimization freedom for a compiled block.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CombinationalOptimization {
    #[default]
    Free,
    PreserveNames,
    PreserveNamesAndFunctions,
}

/// One concrete value supplied for a block parametric binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParametricBinding {
    pub name: String,
    /// A typed DSLX constexpr expression, for example `u32:8` or `false`.
    pub value: String,
}

/// Options controlling block parsing, elaboration, and lowering.
#[derive(Debug, Clone)]
pub struct BlockCompileOptions {
    pub top: Option<String>,
    pub parametric_bindings: Vec<ParametricBinding>,
    pub combinational_optimization: CombinationalOptimization,
    pub dslx_stdlib_path: Option<PathBuf>,
    pub additional_search_paths: Vec<PathBuf>,
    pub enable_warnings: Option<Vec<String>>,
    pub disable_warnings: Option<Vec<String>>,
    /// Directory containing official XLS tools. Required only by reachable
    /// proc instances in the MVP.
    pub tool_path: Option<PathBuf>,
    /// Maximum wall time for one external XLS tool invocation.
    pub external_tool_timeout: Duration,
    /// Maximum stdout/stderr bytes retained per external XLS tool stream.
    pub max_tool_output_bytes: usize,
    /// Maximum size of a lossless external-tool artifact before the tool is
    /// terminated and compilation fails.
    pub max_tool_artifact_bytes: usize,
}

impl Default for BlockCompileOptions {
    fn default() -> Self {
        Self {
            top: None,
            parametric_bindings: Vec::new(),
            combinational_optimization: CombinationalOptimization::Free,
            dslx_stdlib_path: None,
            additional_search_paths: Vec::new(),
            enable_warnings: None,
            disable_warnings: None,
            tool_path: None,
            external_tool_timeout: Duration::from_secs(60),
            max_tool_output_bytes: 1024 * 1024,
            max_tool_artifact_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Successful block compilation result.
#[derive(Debug, Clone)]
pub struct BlockCompileOutput {
    pub package: xlsynth_pir::ir::Package,
    pub ir_text: String,
    pub warnings: Vec<String>,
}

/// Source-oriented frontend diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockDiagnostic {
    pub path: PathBuf,
    pub offset: Option<usize>,
    pub message: String,
}

impl BlockDiagnostic {
    pub(crate) fn new(path: &Path, offset: Option<usize>, message: impl Into<String>) -> Self {
        Self {
            path: path.to_path_buf(),
            offset,
            message: message.into(),
        }
    }
}

impl fmt::Display for BlockDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(offset) = self.offset {
            write!(
                f,
                "{}:byte-offset-{}: {}",
                self.path.display(),
                offset,
                self.message
            )
        } else {
            write!(f, "{}: {}", self.path.display(), self.message)
        }
    }
}

impl std::error::Error for BlockDiagnostic {}
