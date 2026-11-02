// SPDX-License-Identifier: Apache-2.0

//! Exports pure XLS IR functions and combinational gate graphs to executable
//! Lean.
//!
//! The word-level exporter preserves the XLS function signature and is checked
//! against the PIR evaluator. Word-level proof targets expose each IR node and
//! check pointwise equivalence over the full signature. Gate proofs use an
//! evaluator proved equivalent to Lean's stock AIG semantics.

mod error;
mod evaluate;
mod gate_fn;
mod ir_fn;
mod ir_proof;
mod process;
mod proof;
mod support;

pub use error::{InterfaceDirection, LeanError};
pub use evaluate::{LeanEvaluator, LeanFunctionInputs};
pub use gate_fn::{
    CircuitInterface, CircuitPort, LeanCircuitBundle, LeanExportOptions, LeanModuleName,
    WrittenLeanCircuit, export_gate_fn,
};
pub use ir_fn::{LeanFunction, export_ir_function};
pub use ir_proof::{LeanIrFunctionEquivalenceTarget, prepare_ir_function_equivalence_target};
pub use proof::{
    LeanEquivalenceTarget, LeanProofFailure, LeanProofOutput, LeanProofSources,
    check_equivalence_proof, prepare_equivalence_target, run_equivalence_proof,
    run_ir_function_equivalence_proof,
};
pub use support::write_lean_support;

/// Lean release used to validate the companion library.
pub const TESTED_LEAN_VERSION: &str = "4.34.1";
