// SPDX-License-Identifier: Apache-2.0

#![feature(map_try_insert)]
#![feature(let_chains)]
#![feature(custom_test_frameworks)]

pub mod aig_hasher;
mod aig_simplify;
pub mod bulk_replace;
pub mod check_equivalence;
pub mod count_toggles;
pub mod dce;
mod emit_netlist;
pub mod fanout;
pub mod find_structures;
pub mod fraig;
pub mod fuzz_utils;
pub mod gate;
pub mod gate2ir;
pub mod gate_builder;
pub mod gate_sim;
pub mod get_summary_stats;
pub mod graph_logical_effort;
pub mod ir2gate;
pub mod ir2gate_utils;
pub mod ir_equiv_boolector;
pub mod ir_value_utils;
pub mod liberty;
pub mod logical_effort;
pub mod process_ir_path;
pub mod propose_equiv;
pub mod test_utils;
pub mod topo;
pub mod transforms;
pub mod use_count;
pub mod validate_equiv;
pub mod xls_ir;
