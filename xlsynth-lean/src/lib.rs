// SPDX-License-Identifier: Apache-2.0

//! Exports XLSynth gate graphs and XLS block IR to Lean.
//!
//! The Rust exporter and native g8r reader are part of the conversion trust
//! boundary. The companion Lean library checks the exported graph structure and
//! evaluates it using a function proved equivalent to Lean's stock AIG
//! semantics.
//!
//! [`block`] emits transition equations from block IR. For stateful DSLX
//! functions it also emits a typed interface and a proof relating the models.

pub mod block;

mod error;
mod evaluate;
mod gate_fn;
mod support;

pub use error::LeanError;
pub use evaluate::LeanEvaluator;
pub use gate_fn::{
    CircuitInterface, CircuitPort, LeanCircuitBundle, LeanExportOptions, LeanModuleName,
    WrittenLeanCircuit, export_gate_fn,
};
pub use support::write_lean_support;

/// Lean release used to validate the companion library.
pub const TESTED_LEAN_VERSION: &str = "4.34.1";
