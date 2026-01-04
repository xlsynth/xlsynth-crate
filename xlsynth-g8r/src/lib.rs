// SPDX-License-Identifier: Apache-2.0

#![feature(map_try_insert)]
#![feature(custom_test_frameworks)]
#![feature(portable_simd)]

pub mod aig;
pub mod aig_serdes;
pub mod aig_sim;

pub mod check_equivalence;
pub mod cut_db;
pub mod cut_db_cli_defaults;
pub mod dslx_stitch_pipeline;
pub mod gate_builder;
pub mod ir2gate_utils;
pub mod ir2gates;
pub mod liberty;
pub mod verilog_version;
pub mod liberty_proto {
    include!(concat!(env!("OUT_DIR"), "/liberty.rs"));
}
pub mod mcmc_logic;
pub mod netlist;
pub mod process_ir_path;
pub mod propose_equiv;
pub mod prove_gate_fn_equiv_common;
pub mod prove_gate_fn_equiv_varisat;
#[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
pub mod prove_gate_fn_equiv_z3;
pub mod test_utils;
pub mod transforms;
pub mod use_count;
