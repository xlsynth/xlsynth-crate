// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use xlsynth_g8r::aig::cut_db_rewrite::CutDbRewriteMode;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::process_ir_path;
use xlsynth_g8r::process_ir_path::{CanonicalG8rOptions, DEFAULT_MAX_FRAIG_SIM_SAMPLES};
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

fn parse_cut_db_rewrite_mode_or_exit(matches: &ArgMatches) -> CutDbRewriteMode {
    let value = matches
        .get_one::<String>("cut_db_rewrite_mode")
        .map(|s| s.as_str())
        .unwrap_or(CutDbRewriteMode::DEFAULT_CLI_VALUE);
    match CutDbRewriteMode::parse(value) {
        Some(mode) => mode,
        None => {
            eprintln!("Invalid --cut-db-rewrite-mode: {:?}", value);
            std::process::exit(1);
        }
    }
}

pub(crate) fn parse_g8r_cli_options(matches: &ArgMatches) -> CanonicalG8rOptions {
    let defaults = CanonicalG8rOptions::default();
    let fold = parse_bool(matches, "fold", defaults.fold);
    let hash = parse_bool(matches, "hash", defaults.hash);
    let fraig = parse_bool(matches, "fraig", defaults.fraig);
    let enable_rewrite_carry_out = parse_bool(
        matches,
        "enable-rewrite-carry-out",
        defaults.enable_rewrite_carry_out,
    );
    let enable_rewrite_prio_encode = parse_bool(
        matches,
        "enable-rewrite-prio-encode",
        defaults.enable_rewrite_prio_encode,
    );
    let enable_rewrite_nary_add = parse_bool(
        matches,
        "enable-rewrite-nary-add",
        defaults.enable_rewrite_nary_add,
    );
    let enable_rewrite_mask_low = parse_bool(
        matches,
        "enable-rewrite-mask-low",
        defaults.enable_rewrite_mask_low,
    );
    let unsafe_gatify_gate_operation = parse_bool(
        matches,
        "unsafe-gatify-gate-operation",
        defaults.unsafe_gatify_gate_operation,
    );
    let (adder_mapping, mul_adder_mapping) = parse_adder_mappings(matches);
    let toggle_sample_count =
        parse_usize_default(matches, "toggle_sample_count", defaults.toggle_sample_count);
    let toggle_sample_seed =
        parse_u64_default(matches, "toggle_sample_seed", defaults.toggle_sample_seed);
    let compute_graph_logical_effort = parse_bool(
        matches,
        "compute_graph_logical_effort",
        defaults.compute_graph_logical_effort,
    );
    let graph_logical_effort_beta1 = parse_f64_default(
        matches,
        "graph_logical_effort_beta1",
        defaults.graph_logical_effort_beta1,
    );
    let graph_logical_effort_beta2 = parse_f64_default(
        matches,
        "graph_logical_effort_beta2",
        defaults.graph_logical_effort_beta2,
    );
    let cut_db_enable_large_cone_rewrite = parse_bool(
        matches,
        "cut-db-enable-large-cone-rewrite",
        defaults.cut_db_enable_large_cone_rewrite,
    );
    let cut_db_rewrite_mode = parse_cut_db_rewrite_mode_or_exit(matches);
    let fraig_max_iterations =
        parse_optional_usize_or_exit(matches, "fraig_max_iterations", "--fraig-max-iterations");
    let max_fraig_sim_samples = parse_usize_or_exit(
        matches,
        "max_fraig_sim_samples",
        "--max-fraig-sim-samples",
        DEFAULT_MAX_FRAIG_SIM_SAMPLES,
    );
    let gate_formal_backend = parse_gate_formal_backend_or_exit(matches);

    CanonicalG8rOptions {
        fold,
        hash,
        enable_rewrite_carry_out,
        enable_rewrite_prio_encode,
        enable_rewrite_nary_add,
        enable_rewrite_mask_low,
        unsafe_gatify_gate_operation,
        adder_mapping,
        mul_adder_mapping,
        fraig,
        toggle_sample_count,
        toggle_sample_seed,
        compute_graph_logical_effort,
        graph_logical_effort_beta1,
        graph_logical_effort_beta2,
        cut_db_enable_large_cone_rewrite,
        cut_db_rewrite_mode,
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
    cli.to_process_ir_path_options(
        ir_top,
        quiet,
        emit_netlist,
        emit_independent_op_stats,
        prepared_ir_out,
    )
}
