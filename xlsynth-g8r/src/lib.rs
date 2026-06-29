// SPDX-License-Identifier: Apache-2.0

#![feature(portable_simd)]

pub mod aig;
pub mod aig_serdes;
pub mod aig_sim;
pub mod block2sequential;
pub mod gatify;
pub mod ir_aig_sharing;

pub mod check_equivalence;
pub mod cut_db;
pub mod cut_db_cli_defaults;
pub mod diverse_samples;
pub mod dslx_stitch_pipeline;
pub mod gate_builder;
pub mod gate_fn_equiv_report;
pub mod gate_fn_optimize;
pub mod ir2gate_utils;
pub mod ir2gates;
pub mod liberty;
pub(crate) mod prefix_scan_utils;
pub mod verilog_version;
pub mod liberty_proto {
    include!(concat!(env!("OUT_DIR"), "/liberty.rs"));
}
/// Normalized in-memory Liberty model used by evaluators and transforms.
pub mod liberty_model {
    pub use crate::liberty::model::{
        AxisId, Cell, InternalPower, Library, LibraryBuilder, LuTableTemplate, LutShape,
        LutTemplateKind, LutValueRange, LutVariable, Pin, PowerTable, StringId, TimingArc,
        TimingSense, TimingTable, TimingType,
    };
    pub use crate::liberty_proto::{
        ClockGate, LibraryUnits, PinDirection, PowerTransition, Sequential, SequentialKind,
    };
}
pub mod result_proto {
    include!(concat!(env!("OUT_DIR"), "/g8r_results.rs"));
}
pub mod mcmc_logic;
pub mod netlist;
pub mod process_ir_path;
pub mod propose_equiv;
pub mod prove_gate_fn_equiv_common;
pub mod prove_gate_fn_equiv_sat;
#[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
pub mod prove_gate_fn_equiv_z3;
pub mod test_utils;
pub mod transforms;
pub mod use_count;
