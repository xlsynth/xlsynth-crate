// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::path::PathBuf;

/// Direction of a port in a circuit interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceDirection {
    /// Input ports.
    Input,
    /// Output ports.
    Output,
}

impl fmt::Display for InterfaceDirection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Input => "input",
            Self::Output => "output",
        })
    }
}

/// Error returned while exporting, evaluating, or checking a proof in Lean.
#[derive(Debug, thiserror::Error)]
pub enum LeanError {
    #[error("invalid Lean module name: {0}")]
    InvalidModuleName(String),

    #[error("invalid GateFn: {0}")]
    InvalidGraph(String),

    #[error("invalid Lean proof sources: {0}")]
    InvalidProofSources(String),

    #[error("Lean proof work directory is not empty: {0}")]
    WorkDirectoryNotEmpty(PathBuf),

    #[error("{direction} port count mismatch: lhs has {lhs}, rhs has {rhs}")]
    InterfacePortCountMismatch {
        direction: InterfaceDirection,
        lhs: usize,
        rhs: usize,
    },

    #[error("{direction} port {port_index} name mismatch: lhs is {lhs:?}, rhs is {rhs:?}")]
    InterfacePortNameMismatch {
        direction: InterfaceDirection,
        port_index: usize,
        lhs: String,
        rhs: String,
    },

    #[error("{direction} port {port_index} width mismatch: lhs is {lhs}, rhs is {rhs}")]
    InterfacePortWidthMismatch {
        direction: InterfaceDirection,
        port_index: usize,
        lhs: usize,
        rhs: usize,
    },

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
