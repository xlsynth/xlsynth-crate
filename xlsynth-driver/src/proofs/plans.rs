// SPDX-License-Identifier: Apache-2.0

//! Compatibility exports for obligation plan construction.

// Used by library consumers; the CLI binary no longer constructs plans here.
#[allow(unused_imports)]
pub use xlsynth_prover::dslx_tactics::plans::*;
