// SPDX-License-Identifier: Apache-2.0

//! Exports combinational XLSynth gate graphs to Lean, evaluates them there, and
//! checks user-supplied equivalence proofs.
//!
//! The Rust exporter and native g8r reader are part of the conversion trust
//! boundary. The companion Lean library checks the exported graph structure and
//! evaluates it using a function proved equivalent to Lean's stock AIG
//! semantics.

mod error;
mod evaluate;
mod gate_fn;
mod process;
mod proof;
mod support;

pub use error::{InterfaceDirection, LeanError};
pub use evaluate::LeanEvaluator;
pub use gate_fn::{
    CircuitInterface, CircuitPort, LeanCircuitBundle, LeanExportOptions, LeanModuleName,
    WrittenLeanCircuit, export_gate_fn,
};
pub use proof::{
    LeanEquivalenceTarget, LeanProofFailure, LeanProofOutput, LeanProofSources,
    check_equivalence_proof, prepare_equivalence_target, run_equivalence_proof,
};
pub use support::write_lean_support;

/// Lean release used to validate the companion library.
pub const TESTED_LEAN_VERSION: &str = "4.34.1";
