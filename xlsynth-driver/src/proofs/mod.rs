// SPDX-License-Identifier: Apache-2.0

pub mod plans;
pub mod script;
pub use xlsynth_prover::dslx_tactics::obligations;
// Preserve the library import path even though the CLI uses the script engine.
#[allow(unused_imports)]
pub use xlsynth_prover::dslx_tactics::tactics;
