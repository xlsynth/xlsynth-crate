// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

/// Error returned while exporting or evaluating a gate graph in Lean.
#[derive(Debug, thiserror::Error)]
pub enum LeanError {
    #[error("invalid Lean module name: {0}")]
    InvalidModuleName(String),

    #[error("invalid GateFn: {0}")]
    InvalidGraph(String),

    #[error("input case {case_index} has {actual} ports; expected {expected}")]
    InputCount {
        case_index: usize,
        expected: usize,
        actual: usize,
    },

    #[error("input case {case_index}, port {input_index} has width {actual}; expected {expected}")]
    InputWidth {
        case_index: usize,
        input_index: usize,
        expected: usize,
        actual: usize,
    },

    #[error("{operation} at {path}: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to create temporary directory: {0}")]
    TemporaryDirectory(#[source] std::io::Error),

    #[error("failed to construct Lean module search path: {0}")]
    ModuleSearchPath(#[source] std::env::JoinPathsError),

    #[error(
        "Lean {operation} failed with exit code {exit_code:?}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )]
    ProcessFailed {
        operation: &'static str,
        exit_code: Option<i32>,
        stdout: String,
        stderr: String,
    },

    #[error("invalid Lean evaluation response: {0}")]
    Protocol(String),

    #[error("failed to encode or decode JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("invalid bitvector value: {0}")]
    Value(#[from] xlsynth_ir_value::ValueError),
}
