// SPDX-License-Identifier: Apache-2.0

//! Compatibility shim re-exporting the Varisat implementation so that legacy
//! `crate::validate_equiv::*` imports continue to compile.

pub use crate::prove_gate_fn_equiv_varisat::{
    prove_gate_fn_equiv, validate_equivalence_classes, Ctx, EquivResult,
};
