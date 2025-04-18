// SPDX-License-Identifier: Apache-2.0

#![feature(map_try_insert)]
#![feature(let_chains)]
#![feature(custom_test_frameworks)]

pub mod aig_hasher;
mod aig_simplify;
pub mod check_equivalence;
mod emit_netlist;
pub mod find_structures;
pub mod gate;
pub mod gate2ir;
pub mod gate_builder;
pub mod gate_sim;
pub mod get_summary_stats;
pub mod ir2gate;
pub mod ir2gate_utils;
pub mod liberty;
pub mod process_ir_path;
pub mod propose_equiv;
pub mod test_utils;
pub mod use_count;
pub mod validate_equiv;
pub mod xls_ir;
