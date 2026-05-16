// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use clap::ArgMatches;
use serde::{Deserialize, Serialize};
use xlsynth_g8r::process_ir_path::CanonicalG8rOptions;
use xlsynth_mcmc_pir::{
    canonical_g8r_scoring_input_for_pir_fn,
    cost_with_effort_options_toggle_stimulus_extension_mode_evaluator_and_g8r_options,
    lower_toggle_stimulus_for_fn, parse_irvals_tuple_file, Cost, G8rEvaluationMode, Objective,
};
use xlsynth_pir::ir_parser;

use crate::report_cli_error::report_cli_error_and_exit;

#[derive(Debug, Deserialize)]
struct RawStatsJson {
    live_nodes: usize,
    deepest_path: usize,
    graph_logical_effort_worst_case_delay: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct AigStatsJson {
    and_nodes: usize,
    depth: usize,
    graph_logical_effort_worst_case_delay: Option<f64>,
}

#[derive(Debug, Serialize)]
struct AlignmentCommands {
    ir2g8r: Vec<String>,
    postprocess: Option<Vec<String>>,
    aig_stats: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct AlignmentComparison {
    objective: String,
    mcmc_origin_cost: Cost,
    external_origin_cost: Cost,
    mcmc_score: String,
    external_score: String,
    raw_nodes_match: bool,
    raw_depth_match: bool,
    raw_graph_logical_effort_milli_match: bool,
    post_and_nodes_match: Option<bool>,
    post_depth_match: Option<bool>,
    post_graph_logical_effort_milli_match: Option<bool>,
    objective_score_match: bool,
}

fn graph_le_delay_to_milli(delay: Option<f64>) -> usize {
    let Some(delay) = delay else {
        return 0;
    };
    if !delay.is_finite() {
        return usize::MAX;
    }
    if delay <= 0.0 {
        return 0;
    }
    let scaled = delay * 1000.0;
    if scaled >= usize::MAX as f64 {
        usize::MAX
    } else {
        scaled.round() as usize
    }
}

fn bool_arg(value: bool) -> String {
    if value { "true" } else { "false" }.to_string()
}

fn canonical_g8r_flag_args(options: &CanonicalG8rOptions) -> Vec<String> {
    let mut args = vec![
        "--fold".to_string(),
        bool_arg(options.fold),
        "--hash".to_string(),
        bool_arg(options.hash),
        "--enable-rewrite-carry-out".to_string(),
        bool_arg(options.enable_rewrite_carry_out),
        "--enable-rewrite-prio-encode".to_string(),
        bool_arg(options.enable_rewrite_prio_encode),
        "--enable-rewrite-nary-add".to_string(),
        bool_arg(options.enable_rewrite_nary_add),
        "--enable-rewrite-mask-low".to_string(),
        bool_arg(options.enable_rewrite_mask_low),
        "--unsafe-gatify-gate-operation".to_string(),
        bool_arg(options.unsafe_gatify_gate_operation),
        "--adder-mapping".to_string(),
        options.adder_mapping.to_string(),
        "--fraig".to_string(),
        bool_arg(options.fraig),
        "--reassociation".to_string(),
        bool_arg(options.reassociation),
        "--max-fraig-sim-samples".to_string(),
        options.max_fraig_sim_samples.to_string(),
        "--gate-formal-backend".to_string(),
        options.gate_formal_backend.to_string(),
        "--compute_graph_logical_effort".to_string(),
        bool_arg(options.compute_graph_logical_effort),
        "--graph-logical-effort-beta1".to_string(),
        options.graph_logical_effort_beta1.to_string(),
        "--graph-logical-effort-beta2".to_string(),
        options.graph_logical_effort_beta2.to_string(),
        "--toggle-sample-count".to_string(),
        options.toggle_sample_count.to_string(),
        "--toggle-seed".to_string(),
        options.toggle_sample_seed.to_string(),
    ];
    if let Some(mapping) = options.mul_adder_mapping {
        args.extend(["--mul-adder-mapping".to_string(), mapping.to_string()]);
    }
    if let Some(iterations) = options.fraig_max_iterations {
        args.extend(["--fraig-max-iterations".to_string(), iterations.to_string()]);
    }
    args
}

fn run_checked(command: &mut Command, label: &str) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|e| format!("failed to run {label}: {e}"))?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "{label} failed with status {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    ))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|e| format!("failed to serialize {}: {e}", path.display()))?;
    std::fs::write(path, text).map_err(|e| format!("failed to write {}: {e}", path.display()))
}

fn compute_cost(
    top_fn: &xlsynth_pir::ir::Fn,
    objective: Objective,
    toggle_stimulus: Option<&[Vec<xlsynth::IrBits>]>,
    weighted_switching_options: &xlsynth_g8r::aig_sim::count_toggles::WeightedSwitchingOptions,
    cli: &xlsynth_mcmc_pir::driver_cli::PirMcmcCliArgs,
    g8r_evaluation_mode: &G8rEvaluationMode,
) -> Result<Cost, String> {
    cost_with_effort_options_toggle_stimulus_extension_mode_evaluator_and_g8r_options(
        top_fn,
        objective,
        toggle_stimulus,
        weighted_switching_options,
        cli.extension_costing_mode,
        g8r_evaluation_mode,
        &cli.canonical_g8r_options,
    )
    .map_err(|e| e.to_string())
}

fn verify_origin_alignment(
    cli: &xlsynth_mcmc_pir::driver_cli::PirMcmcCliArgs,
) -> Result<(), String> {
    let output_dir = cli.output.as_ref().ok_or_else(|| {
        "--verify-origin-alignment requires a concrete output directory".to_string()
    })?;
    let output_dir = PathBuf::from(output_dir);
    let alignment_dir = output_dir.join("origin-alignment");
    std::fs::create_dir_all(&alignment_dir)
        .map_err(|e| format!("failed to create {}: {e}", alignment_dir.display()))?;

    let mut pkg = ir_parser::parse_and_validate_path_to_package(Path::new(&cli.input_path))
        .map_err(|e| format!("failed to parse PIR package: {e:?}"))?;
    if let Some(top) = cli.top.as_deref() {
        pkg.set_top_fn(top)
            .map_err(|e| format!("failed to set top function '{top}': {e}"))?;
    }
    let top_fn = pkg
        .get_top_fn()
        .cloned()
        .ok_or_else(|| "No top function found in PIR package".to_string())?;

    let toggle_stimulus_values = cli
        .toggle_stimulus
        .as_ref()
        .map(|path| parse_irvals_tuple_file(Path::new(path)))
        .transpose()
        .map_err(|e| e.to_string())?;
    let prepared_toggle_stimulus = toggle_stimulus_values
        .as_ref()
        .map(|samples| lower_toggle_stimulus_for_fn(samples, &top_fn))
        .transpose()
        .map_err(|e| e.to_string())?;
    let weighted_switching_options =
        xlsynth_g8r::aig_sim::count_toggles::WeightedSwitchingOptions {
            beta1: cli.switching_beta1,
            beta2: cli.switching_beta2,
            primary_output_load: cli.switching_primary_output_load,
        };
    let g8r_evaluation_mode = match cli.g8r_postprocess_program.clone() {
        Some(program) => G8rEvaluationMode::ExternalPostprocess { program }
            .canonicalized_for_persistence()
            .map_err(|e| e.to_string())?,
        None => G8rEvaluationMode::Builtin,
    };
    let mut mcmc_origin_cost = compute_cost(
        &top_fn,
        cli.metric,
        prepared_toggle_stimulus.as_deref(),
        &weighted_switching_options,
        cli,
        &g8r_evaluation_mode,
    )?;
    let verify_graph_logical_effort = cli.canonical_g8r_options.compute_graph_logical_effort
        || cli.metric.needs_graph_logical_effort();
    let raw_structural_cost = compute_cost(
        &top_fn,
        Objective::G8rNodesTimesDepth,
        None,
        &weighted_switching_options,
        cli,
        &g8r_evaluation_mode,
    )?;
    mcmc_origin_cost.g8r_nodes = raw_structural_cost.g8r_nodes;
    mcmc_origin_cost.g8r_depth = raw_structural_cost.g8r_depth;
    if verify_graph_logical_effort {
        let raw_graph_cost = compute_cost(
            &top_fn,
            Objective::G8rLeGraph,
            None,
            &weighted_switching_options,
            cli,
            &g8r_evaluation_mode,
        )?;
        mcmc_origin_cost.g8r_le_graph_milli = raw_graph_cost.g8r_le_graph_milli;
    } else {
        mcmc_origin_cost.g8r_le_graph_milli = 0;
    }
    if matches!(
        g8r_evaluation_mode,
        G8rEvaluationMode::ExternalPostprocess { .. }
    ) {
        let post_structural_cost = compute_cost(
            &top_fn,
            Objective::G8rPostAndNodesTimesDepth,
            None,
            &weighted_switching_options,
            cli,
            &g8r_evaluation_mode,
        )?;
        mcmc_origin_cost.g8r_post_and_nodes = post_structural_cost.g8r_post_and_nodes;
        mcmc_origin_cost.g8r_post_depth = post_structural_cost.g8r_post_depth;
        if verify_graph_logical_effort {
            let post_graph_cost = compute_cost(
                &top_fn,
                Objective::G8rPostLeGraph,
                None,
                &weighted_switching_options,
                cli,
                &g8r_evaluation_mode,
            )?;
            mcmc_origin_cost.g8r_post_le_graph_milli = post_graph_cost.g8r_post_le_graph_milli;
        } else {
            mcmc_origin_cost.g8r_post_le_graph_milli = 0;
        }
    }

    let scoring_input = canonical_g8r_scoring_input_for_pir_fn(&top_fn, cli.extension_costing_mode)
        .map_err(|e| e.to_string())?;
    let scored_ir = alignment_dir.join("scored.ir");
    std::fs::write(&scored_ir, scoring_input.ir_text)
        .map_err(|e| format!("failed to write {}: {e}", scored_ir.display()))?;
    let raw_aig = alignment_dir.join("raw.aig");
    let raw_stats = alignment_dir.join("raw.stats.json");
    let exe = std::env::current_exe().map_err(|e| format!("failed to resolve current exe: {e}"))?;
    let mut external_g8r_options = cli.canonical_g8r_options.clone();
    external_g8r_options.compute_graph_logical_effort = verify_graph_logical_effort;
    let mut ir2g8r_args = vec![
        exe.display().to_string(),
        "ir2g8r".to_string(),
        scored_ir.display().to_string(),
        "--top".to_string(),
        scoring_input.top_fn.name,
        "--aiger-out".to_string(),
        raw_aig.display().to_string(),
        "--stats-out".to_string(),
        raw_stats.display().to_string(),
    ];
    ir2g8r_args.extend(canonical_g8r_flag_args(&external_g8r_options));
    let mut ir2g8r_cmd = Command::new(&exe);
    ir2g8r_cmd.args(&ir2g8r_args[1..]).stdout(Stdio::null());
    run_checked(&mut ir2g8r_cmd, "ir2g8r alignment command")?;

    let raw_stats_json: RawStatsJson = read_json(&raw_stats)?;
    let mut external_origin_cost = mcmc_origin_cost;
    external_origin_cost.g8r_nodes = raw_stats_json.live_nodes;
    external_origin_cost.g8r_depth = raw_stats_json.deepest_path;
    external_origin_cost.g8r_le_graph_milli =
        graph_le_delay_to_milli(raw_stats_json.graph_logical_effort_worst_case_delay);

    let mut postprocess_args = None;
    let mut aig_stats_args = None;
    let mut post_stats_json = None;
    if let G8rEvaluationMode::ExternalPostprocess { program } = &g8r_evaluation_mode {
        let post_aig = alignment_dir.join("post.aig");
        let post_stats = alignment_dir.join("post.stats.json");
        let args = vec![
            program.to_string(),
            raw_aig.display().to_string(),
            "--output-path".to_string(),
            post_aig.display().to_string(),
        ];
        let mut post_cmd = Command::new(program);
        post_cmd.args(&args[1..]);
        run_checked(&mut post_cmd, "g8r postprocess alignment command")?;
        postprocess_args = Some(args);

        let args = vec![
            exe.display().to_string(),
            "aig-stats".to_string(),
            "--compute-graph-logical-effort".to_string(),
            bool_arg(verify_graph_logical_effort),
            "--graph-logical-effort-beta1".to_string(),
            cli.canonical_g8r_options
                .graph_logical_effort_beta1
                .to_string(),
            "--graph-logical-effort-beta2".to_string(),
            cli.canonical_g8r_options
                .graph_logical_effort_beta2
                .to_string(),
            "--output_json".to_string(),
            post_stats.display().to_string(),
            post_aig.display().to_string(),
        ];
        let mut stats_cmd = Command::new(&exe);
        stats_cmd.args(&args[1..]);
        run_checked(&mut stats_cmd, "aig-stats alignment command")?;
        aig_stats_args = Some(args);
        let stats: AigStatsJson = read_json(&post_stats)?;
        external_origin_cost.g8r_post_and_nodes = stats.and_nodes;
        external_origin_cost.g8r_post_depth = stats.depth;
        external_origin_cost.g8r_post_le_graph_milli =
            graph_le_delay_to_milli(stats.graph_logical_effort_worst_case_delay);
        post_stats_json = Some(stats);
    }

    write_json(
        &alignment_dir.join("commands.json"),
        &AlignmentCommands {
            ir2g8r: ir2g8r_args,
            postprocess: postprocess_args,
            aig_stats: aig_stats_args,
        },
    )?;

    let mcmc_score = cli.metric.metric(&mcmc_origin_cost);
    let external_score = cli.metric.metric(&external_origin_cost);
    let raw_graph_match =
        mcmc_origin_cost.g8r_le_graph_milli == external_origin_cost.g8r_le_graph_milli;
    let comparison = AlignmentComparison {
        objective: cli.metric.value_name().to_string(),
        mcmc_origin_cost,
        external_origin_cost,
        mcmc_score: mcmc_score.to_string(),
        external_score: external_score.to_string(),
        raw_nodes_match: mcmc_origin_cost.g8r_nodes == external_origin_cost.g8r_nodes,
        raw_depth_match: mcmc_origin_cost.g8r_depth == external_origin_cost.g8r_depth,
        raw_graph_logical_effort_milli_match: raw_graph_match,
        post_and_nodes_match: post_stats_json.as_ref().map(|_| {
            mcmc_origin_cost.g8r_post_and_nodes == external_origin_cost.g8r_post_and_nodes
        }),
        post_depth_match: post_stats_json
            .as_ref()
            .map(|_| mcmc_origin_cost.g8r_post_depth == external_origin_cost.g8r_post_depth),
        post_graph_logical_effort_milli_match: post_stats_json.as_ref().map(|_| {
            mcmc_origin_cost.g8r_post_le_graph_milli == external_origin_cost.g8r_post_le_graph_milli
        }),
        objective_score_match: mcmc_score == external_score,
    };
    write_json(&alignment_dir.join("comparison.json"), &comparison)?;
    let post_matches = comparison.post_and_nodes_match.unwrap_or(true)
        && comparison.post_depth_match.unwrap_or(true)
        && comparison
            .post_graph_logical_effort_milli_match
            .unwrap_or(true);
    if comparison.raw_nodes_match
        && comparison.raw_depth_match
        && comparison.raw_graph_logical_effort_milli_match
        && post_matches
        && comparison.objective_score_match
    {
        Ok(())
    } else {
        Err(format!(
            "origin alignment mismatch; see {}",
            alignment_dir.join("comparison.json").display()
        ))
    }
}

pub fn handle_ir_mcmc_opt(matches: &ArgMatches) {
    let should_verify_origin_alignment = matches.get_flag("verify_origin_alignment");
    let mut cli = xlsynth_mcmc_pir::driver_cli::parse_pir_mcmc_args(matches);
    if should_verify_origin_alignment && cli.metric.needs_toggle_stimulus() {
        report_cli_error_and_exit(
            &format!(
                "--verify-origin-alignment does not support toggle-dependent objective {}",
                cli.metric.value_name()
            ),
            Some("ir-mcmc-opt"),
            vec![],
        );
    }
    if should_verify_origin_alignment && cli.output.is_none() {
        match tempfile::Builder::new()
            .prefix("pir_mcmc_alignment_output_")
            .tempdir()
        {
            Ok(temp_dir) => {
                cli.output = Some(temp_dir.keep().display().to_string());
            }
            Err(e) => report_cli_error_and_exit(
                &format!("failed to create verification output directory: {e}"),
                Some("ir-mcmc-opt"),
                vec![],
            ),
        }
    }
    if let Err(e) =
        xlsynth_mcmc_pir::driver_cli::run_pir_mcmc_driver(cli.clone(), |msg| eprintln!("{msg}"))
    {
        let message = e.to_string();
        report_cli_error_and_exit(&message, Some("ir-mcmc-opt"), vec![]);
    }
    if should_verify_origin_alignment {
        if let Err(message) = verify_origin_alignment(&cli) {
            report_cli_error_and_exit(&message, Some("ir-mcmc-opt"), vec![]);
        }
        eprintln!(
            "Verified origin alignment artifacts in {}",
            Path::new(cli.output.as_ref().expect("verification output"))
                .join("origin-alignment")
                .display()
        );
    }
}
