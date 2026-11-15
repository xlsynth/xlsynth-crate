// SPDX-License-Identifier: Apache-2.0

pub mod aig_hasher;
pub mod aig_simplify;
pub mod bulk_replace;
pub mod cut_db_rewrite;
pub mod dce;
pub mod dynamic_depth;
pub mod dynamic_structural_hash;
pub mod fanout;
pub mod find_structures;
pub mod fraig;
pub mod gate;
pub mod get_summary_stats;
pub mod graph_logical_effort;
pub(crate) mod import;
pub mod logical_effort;
pub mod match_and_rewrite;
pub mod mffc;
pub mod reassociation;
mod register_controls;
pub mod sequential_cleanup;
pub mod sequential_gate;
pub mod sequential_pipeline;
pub(crate) mod structural_hash_cons;
pub mod table;
pub mod topo;

pub use crate::aig::gate::{AigBitVector, AigNode, AigOperand, AigRef, GateFn, Input, Output};
pub use crate::aig::sequential_gate::{
    ClockPort, RegisterBinding, RegisterControl, SequentialGateFn, SynchronousReset,
    TransitionInputId, TransitionOutputId, add_input_registers, add_output_registers,
};
pub use crate::aig::sequential_pipeline::{
    SequentialPipelineOptions, stitch_gate_fns_into_pipeline,
    stitch_gate_fns_into_pipeline_with_controls,
};
pub use crate::gate_builder::{GateBuilder, GateBuilderOptions};
