// SPDX-License-Identifier: Apache-2.0

use std::fs;

use clap::Parser;
use prost::Message;
use xlsynth_g8r::aig::cut_db_rewrite::CutDbRewriteMode;
use xlsynth_g8r::cut_db_cli_defaults::{
    CUT_DB_REWRITE_MAX_CUTS_PER_NODE_CLI, CUT_DB_REWRITE_MAX_ITERATIONS_CLI,
};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::process_ir_path::{
    DEFAULT_MAX_FRAIG_SIM_SAMPLES, Options, process_ir_path_for_cli,
};
use xlsynth_g8r::prove_gate_fn_equiv_common::GateFormalBackend;
use xlsynth_g8r::result_proto;

/// Simple program to parse an XLS IR file and emit a Verilog netlist.
#[derive(Parser, Debug)]
struct Args {
    /// Whether to perform AIG folding optimizations
    #[arg(long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    fold: bool,

    /// Whether to hash the AIG nodes.
    #[arg(long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    hash: bool,

    /// Whether to run "fraiging" optimization.
    #[arg(long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    fraig: bool,

    #[arg(long)]
    fraig_max_iterations: Option<usize>,

    #[arg(
        long = "max-fraig-sim-samples",
        alias = "fraig-sim-samples",
        default_value_t = DEFAULT_MAX_FRAIG_SIM_SAMPLES
    )]
    max_fraig_sim_samples: usize,

    /// Formal backend for gate-level proof steps.
    #[arg(long, default_value = GateFormalBackend::DEFAULT_CLI_VALUE, value_parser = GateFormalBackend::CLI_VALUES)]
    gate_formal_backend: String,

    /// Whether to check equivalence between the IR and the gate function.
    #[arg(long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    check_equivalence: bool,

    /// Whether to emit the netlist.
    #[arg(long, default_value_t = false)]
    #[arg(action = clap::ArgAction::Set)]
    emit_netlist: bool,

    /// Number of random input samples for toggle stats (0 disables)
    #[arg(long, default_value_t = 0)]
    toggle_sample_count: usize,

    /// Seed for random toggle stimulus (default 0)
    #[arg(long, default_value_t = 0)]
    toggle_sample_seed: u64,

    /// Whether to compute the graph logical effort worst case delay.
    #[arg(long, default_value_t = false)]
    #[arg(action = clap::ArgAction::Set)]
    compute_graph_logical_effort: bool,

    /// The beta1 value for the graph logical effort computation.
    #[arg(long, default_value_t = 1.0)]
    #[arg(action = clap::ArgAction::Set)]
    graph_logical_effort_beta1: f64,

    /// The beta2 value for the graph logical effort computation.
    #[arg(long, default_value_t = 0.0)]
    #[arg(action = clap::ArgAction::Set)]
    graph_logical_effort_beta2: f64,

    /// Cut-db rewrite mode: delay, balanced, or area.
    #[arg(
        long = "cut-db-rewrite-mode",
        default_value = CutDbRewriteMode::DEFAULT_CLI_VALUE,
        value_parser = clap::builder::PossibleValuesParser::new(CutDbRewriteMode::CLI_VALUES)
    )]
    cut_db_rewrite_mode: String,

    #[arg(long)]
    #[arg(action = clap::ArgAction::Set)]
    result_proto: Option<String>,

    /// The path to the XLS IR file.
    input: String,
}

fn main() {
    let _ = env_logger::builder().try_init();
    let args = Args::parse();
    let gate_formal_backend = GateFormalBackend::parse(&args.gate_formal_backend)
        .expect("validated by clap value_parser");
    let cut_db_rewrite_mode =
        CutDbRewriteMode::parse(&args.cut_db_rewrite_mode).expect("validated by clap value_parser");
    let cut_db = Some(xlsynth_g8r::cut_db::loader::CutDb::load_default());

    let options = Options {
        check_equivalence: args.check_equivalence,
        fold: args.fold,
        hash: args.hash,
        enable_rewrite_carry_out: false,
        enable_rewrite_prio_encode: false,
        enable_rewrite_nary_add: false,
        enable_rewrite_mask_low: false,
        adder_mapping: AdderMapping::default(),
        mul_adder_mapping: None,
        fraig: args.fraig,
        emit_independent_op_stats: false,
        ir_top: None,
        emit_netlist: args.emit_netlist,
        quiet: args.emit_netlist,
        toggle_sample_count: args.toggle_sample_count,
        toggle_sample_seed: args.toggle_sample_seed,
        compute_graph_logical_effort: args.compute_graph_logical_effort,
        graph_logical_effort_beta1: args.graph_logical_effort_beta1,
        graph_logical_effort_beta2: args.graph_logical_effort_beta2,
        fraig_max_iterations: args.fraig_max_iterations,
        max_fraig_sim_samples: Some(args.max_fraig_sim_samples),
        gate_formal_backend,
        cut_db,
        cut_db_rewrite_max_iterations: CUT_DB_REWRITE_MAX_ITERATIONS_CLI,
        cut_db_rewrite_max_cuts_per_node: CUT_DB_REWRITE_MAX_CUTS_PER_NODE_CLI,
        cut_db_enable_large_cone_rewrite: true,
        cut_db_rewrite_mode,
        prepared_ir_out: None,
    };
    let input_path = std::path::Path::new(&args.input);
    let stats = process_ir_path_for_cli(input_path, &options);
    if let Some(file_name) = args.result_proto {
        let out: result_proto::Ir2GatesSummaryStats = stats.into();
        let data = out.encode_to_vec();
        fs::write(file_name, data).expect("Could not write");
    }
}
