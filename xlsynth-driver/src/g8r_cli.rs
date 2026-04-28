// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use xlsynth_g8r::cut_db::loader::CutDb;
use xlsynth_g8r::cut_db_cli_defaults::{
    CUT_DB_REWRITE_MAX_CANDIDATE_EVALS_PER_ROUND_CLI, CUT_DB_REWRITE_MAX_CUTS_PER_NODE_CLI,
    CUT_DB_REWRITE_MAX_ITERATIONS_CLI, CUT_DB_REWRITE_MAX_REWRITES_PER_ROUND_CLI,
};
use xlsynth_g8r::gatify::prep_for_gatify::PrepForGatifyOptions;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::process_ir_path;
use xlsynth_g8r::process_ir_path::DEFAULT_MAX_FRAIG_SIM_SAMPLES;
use xlsynth_g8r::prove_gate_fn_equiv_common::GateFormalBackend;

fn parse_adder_mapping(value: Option<&str>) -> AdderMapping {
    match value {
        Some("ripple-carry") => AdderMapping::RippleCarry,
        Some("brent-kung") => AdderMapping::BrentKung,
        Some("kogge-stone") => AdderMapping::KoggeStone,
        _ => AdderMapping::default(),
    }
}

fn parse_adder_mappings(matches: &ArgMatches) -> (AdderMapping, Option<AdderMapping>) {
    let adder_mapping = parse_adder_mapping(
        matches
            .get_one::<String>("adder_mapping")
            .map(|s| s.as_str()),
    );
    let mul_adder_mapping = match matches
        .get_one::<String>("mul_adder_mapping")
        .map(|s| s.as_str())
    {
        Some("ripple-carry") => Some(AdderMapping::RippleCarry),
        Some("brent-kung") => Some(AdderMapping::BrentKung),
        Some("kogge-stone") => Some(AdderMapping::KoggeStone),
        _ => None,
    };
    (adder_mapping, mul_adder_mapping)
}

fn parse_bool(matches: &ArgMatches, name: &str, default: bool) -> bool {
    match matches.get_one::<String>(name).map(|s| s.as_str()) {
        Some("true") => true,
        Some("false") => false,
        _ => default,
    }
}

fn parse_f64_default(matches: &ArgMatches, name: &str, default: f64) -> f64 {
    matches
        .get_one::<String>(name)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(default)
}

fn parse_usize_default(matches: &ArgMatches, name: &str, default: usize) -> usize {
    matches
        .get_one::<String>(name)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_usize_or_exit(
    matches: &ArgMatches,
    name: &str,
    flag_name_for_error: &str,
    default: usize,
) -> usize {
    let Some(value) = matches.get_one::<String>(name) else {
        return default;
    };
    match value.parse::<usize>() {
        Ok(n) => n,
        Err(_) => {
            eprintln!("Invalid {flag_name_for_error}: {value:?}");
            std::process::exit(1);
        }
    }
}

fn parse_u64_default(matches: &ArgMatches, name: &str, default: u64) -> u64 {
    matches
        .get_one::<String>(name)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(default)
}

fn parse_optional_usize_or_exit(
    matches: &ArgMatches,
    name: &str,
    flag_name_for_error: &str,
) -> Option<usize> {
    let value = matches.get_one::<String>(name);
    let parsed = value.map(|s| s.parse::<usize>());
    match parsed {
        Some(Ok(n)) => Some(n),
        Some(Err(_)) => {
            eprintln!(
                "Invalid {flag_name_for_error}: {:?}",
                matches.get_one::<String>(name).unwrap()
            );
            std::process::exit(1);
        }
        None => None,
    }
}

fn parse_gate_formal_backend_or_exit(matches: &ArgMatches) -> GateFormalBackend {
    let value = matches
        .get_one::<String>("gate_formal_backend")
        .map(|s| s.as_str())
        .unwrap_or(GateFormalBackend::default().as_str());
    match GateFormalBackend::parse(value) {
        Ok(backend) => backend,
        Err(_) => {
            eprintln!("Invalid --gate-formal-backend: {:?}", value);
            std::process::exit(1);
        }
    }
}

pub(crate) struct G8rCliOptions {
    pub(crate) fold: bool,
    pub(crate) hash: bool,
    pub(crate) enable_rewrite_carry_out: bool,
    pub(crate) enable_rewrite_prio_encode: bool,
    pub(crate) enable_rewrite_nary_add: bool,
    pub(crate) enable_rewrite_mask_low: bool,
    pub(crate) adder_mapping: AdderMapping,
    pub(crate) mul_adder_mapping: Option<AdderMapping>,
    pub(crate) fraig: bool,
    pub(crate) toggle_sample_count: usize,
    pub(crate) toggle_sample_seed: u64,
    pub(crate) compute_graph_logical_effort: bool,
    pub(crate) graph_logical_effort_beta1: f64,
    pub(crate) graph_logical_effort_beta2: f64,
    pub(crate) fraig_max_iterations: Option<usize>,
    pub(crate) max_fraig_sim_samples: usize,
    pub(crate) gate_formal_backend: GateFormalBackend,
}

pub(crate) fn parse_g8r_cli_options(matches: &ArgMatches) -> G8rCliOptions {
    let fold = parse_bool(matches, "fold", /* default= */ true);
    let hash = parse_bool(matches, "hash", /* default= */ true);
    let fraig = parse_bool(matches, "fraig", /* default= */ true);
    let prep_defaults = PrepForGatifyOptions::all_opts_enabled();
    let enable_rewrite_carry_out = parse_bool(
        matches,
        "enable-rewrite-carry-out",
        prep_defaults.enable_rewrite_carry_out,
    );
    let enable_rewrite_prio_encode = parse_bool(
        matches,
        "enable-rewrite-prio-encode",
        prep_defaults.enable_rewrite_prio_encode,
    );
    let enable_rewrite_nary_add = parse_bool(
        matches,
        "enable-rewrite-nary-add",
        prep_defaults.enable_rewrite_nary_add,
    );
    let enable_rewrite_mask_low = parse_bool(
        matches,
        "enable-rewrite-mask-low",
        prep_defaults.enable_rewrite_mask_low,
    );
    let (adder_mapping, mul_adder_mapping) = parse_adder_mappings(matches);
    let toggle_sample_count =
        parse_usize_default(matches, "toggle_sample_count", /* default= */ 0);
    let toggle_sample_seed =
        parse_u64_default(matches, "toggle_sample_seed", /* default= */ 0);
    let compute_graph_logical_effort = parse_bool(
        matches,
        "compute_graph_logical_effort",
        /* default= */ true,
    );
    let graph_logical_effort_beta1 = parse_f64_default(
        matches,
        "graph_logical_effort_beta1",
        /* default= */ 1.0,
    );
    let graph_logical_effort_beta2 = parse_f64_default(
        matches,
        "graph_logical_effort_beta2",
        /* default= */ 0.0,
    );
    let fraig_max_iterations =
        parse_optional_usize_or_exit(matches, "fraig_max_iterations", "--fraig-max-iterations");
    let max_fraig_sim_samples = parse_usize_or_exit(
        matches,
        "max_fraig_sim_samples",
        "--max-fraig-sim-samples",
        DEFAULT_MAX_FRAIG_SIM_SAMPLES,
    );
    let gate_formal_backend = parse_gate_formal_backend_or_exit(matches);

    G8rCliOptions {
        fold,
        hash,
        enable_rewrite_carry_out,
        enable_rewrite_prio_encode,
        enable_rewrite_nary_add,
        enable_rewrite_mask_low,
        adder_mapping,
        mul_adder_mapping,
        fraig,
        toggle_sample_count,
        toggle_sample_seed,
        compute_graph_logical_effort,
        graph_logical_effort_beta1,
        graph_logical_effort_beta2,
        fraig_max_iterations,
        max_fraig_sim_samples,
        gate_formal_backend,
    }
}

pub(crate) fn build_process_ir_path_options_for_cli(
    matches: &ArgMatches,
    quiet: bool,
    emit_netlist: bool,
    emit_independent_op_stats: bool,
    ir_top: Option<&str>,
    prepared_ir_out: Option<&std::path::Path>,
) -> process_ir_path::Options {
    let cli = parse_g8r_cli_options(matches);
    process_ir_path::Options {
        check_equivalence: false,
        fold: cli.fold,
        hash: cli.hash,
        enable_rewrite_carry_out: cli.enable_rewrite_carry_out,
        enable_rewrite_prio_encode: cli.enable_rewrite_prio_encode,
        enable_rewrite_nary_add: cli.enable_rewrite_nary_add,
        enable_rewrite_mask_low: cli.enable_rewrite_mask_low,
        adder_mapping: cli.adder_mapping,
        mul_adder_mapping: cli.mul_adder_mapping,
        fraig: cli.fraig,
        emit_independent_op_stats,
        ir_top: ir_top.map(|s| s.to_string()),
        fraig_max_iterations: cli.fraig_max_iterations,
        max_fraig_sim_samples: Some(cli.max_fraig_sim_samples),
        gate_formal_backend: cli.gate_formal_backend,
        quiet,
        emit_netlist,
        toggle_sample_count: cli.toggle_sample_count,
        toggle_sample_seed: cli.toggle_sample_seed,
        compute_graph_logical_effort: cli.compute_graph_logical_effort,
        graph_logical_effort_beta1: cli.graph_logical_effort_beta1,
        graph_logical_effort_beta2: cli.graph_logical_effort_beta2,
        cut_db: Some(CutDb::load_default()),
        cut_db_rewrite_max_iterations: CUT_DB_REWRITE_MAX_ITERATIONS_CLI,
        cut_db_rewrite_max_candidate_evals_per_round:
            CUT_DB_REWRITE_MAX_CANDIDATE_EVALS_PER_ROUND_CLI,
        cut_db_rewrite_max_rewrites_per_round: CUT_DB_REWRITE_MAX_REWRITES_PER_ROUND_CLI,
        cut_db_rewrite_max_cuts_per_node: CUT_DB_REWRITE_MAX_CUTS_PER_NODE_CLI,
        prepared_ir_out: prepared_ir_out.map(|p| p.to_path_buf()),
    }
}
