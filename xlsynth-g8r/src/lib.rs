// SPDX-License-Identifier: Apache-2.0

#![feature(map_try_insert)]
#![feature(custom_test_frameworks)]
#![feature(portable_simd)]

pub mod aig_hasher;
mod aig_simplify;
pub mod bulk_replace;
pub mod check_equivalence;
pub mod count_toggles;
pub mod dce;
pub mod dslx_stitch_pipeline;
pub mod emit_netlist;
pub mod emitted_netlist_parser;
pub mod equiv;
pub mod fanout;
pub mod find_structures;
pub mod fraig;
pub mod gate;
pub mod gate2ir;
pub mod gate_builder;
pub mod gate_parser;
pub mod gate_sim;
pub mod gate_simd;
pub mod get_summary_stats;
pub mod graph_logical_effort;
pub mod ir2gate;
pub mod ir2gate_utils;
pub mod ir_equiv_boolector;
pub mod ir_value_utils;
pub mod liberty;
pub mod match_and_rewrite;
pub mod verilog_version;
pub mod liberty_proto {
    include!(concat!(env!("OUT_DIR"), "/liberty.rs"));
}
pub mod emit_aiger;
pub mod load_aiger;
pub mod logical_effort;
pub mod mcmc_logic;
pub mod netlist;
pub mod process_ir_path;
pub mod propose_equiv;
pub mod prove_gate_fn_equiv_common;
pub mod prove_gate_fn_equiv_varisat;
#[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
pub mod prove_gate_fn_equiv_z3;
pub mod test_utils;
pub mod topo;
pub mod transforms;
pub mod use_count;
