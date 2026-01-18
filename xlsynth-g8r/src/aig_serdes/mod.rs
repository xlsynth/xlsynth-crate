// SPDX-License-Identifier: Apache-2.0

pub mod emit_aiger;
pub mod emit_aiger_binary;
pub mod emit_netlist;
pub mod gate2ir;
pub mod gate_parser;
pub mod load_aiger;
pub mod load_aiger_auto;
pub mod load_aiger_binary;
// NOTE: `aig_serdes` is intended for (de)serialization and external textual
// formats. Gatification/lowering lives under `crate::gatify`. We keep these
// re-exports as a temporary compatibility shim while call sites migrate.
pub use crate::gatify::ir2gate;
pub use crate::gatify::prep_for_gatify;
