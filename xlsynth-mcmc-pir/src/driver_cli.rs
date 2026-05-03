// SPDX-License-Identifier: Apache-2.0

//! Shared CLI wiring for PIR MCMC optimization entry points.
//!
//! Both `pir-mcmc-driver` and `xlsynth-driver ir-mcmc-opt` use this module so
//! their flag surface and behavior stay in sync.

use anyhow::Result;
use clap::{Arg, ArgAction, ArgMatches, Command, ValueEnum};
use num_cpus;
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;
use tempfile::Builder;
use xlsynth_g8r::aig::gate::GateFn;
use xlsynth_g8r::aig::get_summary_stats;
use xlsynth_g8r::aig::get_summary_stats::AigStats;
use xlsynth_g8r::aig::get_summary_stats::SummaryStats;
use xlsynth_g8r::aig_serdes::gate2ir::GateFnInterfaceSchema;
use xlsynth_g8r::gatify::ir2gate;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_mcmc::multichain::ChainStrategy;
use xlsynth_pir::ir::{Package, PackageMember};
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_utils::compact_and_toposort_in_place;

use crate::{
    Best, CheckpointKind, CheckpointMsg, ConstraintLimits, ExtensionCostingMode, G8rEvaluationMode,
    Objective, PirMcmcBudgetFrontierOptions, PirMcmcBudgetWitness, PirMcmcPrefixMinimizeOptions,
    RunOptions, cost_with_effort_options_toggle_stimulus_extension_mode_and_evaluator,
    effective_constraint_limits, format_search_score, lower_toggle_stimulus_for_fn,
    minimize_winning_prefix, parse_irvals_tuple_file, postprocess_gate_fn_for_artifact,
    read_pir_mcmc_artifact_dir, run_pir_mcmc_with_artifact_and_observers,
    run_pir_mcmc_with_shared_best, search_score, search_winning_budget_frontier,
    validate_constraint_configuration, validate_pir_mcmc_artifact_run_options,
    write_pir_mcmc_artifact_dir,
};

#[derive(ValueEnum, Debug, Clone, Copy)]
enum CliChainStrategy {
    Independent,
    ExploreExploit,
}

impl From<CliChainStrategy> for ChainStrategy {
    fn from(v: CliChainStrategy) -> Self {
        match v {
            CliChainStrategy::Independent => ChainStrategy::Independent,
            CliChainStrategy::ExploreExploit => ChainStrategy::ExploreExploit,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PirMcmcCliArgs {
    pub input_path: String,
    pub iters: u64,
    pub seed: u64,
    pub output: Option<String>,
    pub metric: Objective,
    pub extension_costing_mode: ExtensionCostingMode,
    pub g8r_postprocess_program: Option<String>,
    pub max_delay: Option<usize>,
    pub max_area: Option<usize>,
    pub toggle_stimulus: Option<String>,
    pub initial_temperature: f64,
    pub threads: u64,
    pub checkpoint_iters: u64,
    pub progress_iters: u64,
    pub formal_oracle: bool,
    pub switching_beta1: f64,
    pub switching_beta2: f64,
    pub switching_primary_output_load: f64,
    chain_strategy: CliChainStrategy,
}

#[derive(Debug, Clone)]
pub struct PirMcmcMinimizeCliArgs {
    pub run_dir: String,
    pub retained_win_fraction: Option<f64>,
    pub budget_step: Option<usize>,
    pub max_actions: Option<usize>,
    pub rollouts_per_budget: Option<usize>,
    pub seed: Option<u64>,
    pub witness_kind_boost: f64,
    pub proposal_attempts_per_rewrite: usize,
    pub allow_artifact_postprocess_program: bool,
    pub output: String,
}

/// Adds the PIR MCMC arguments to the given command.
pub fn add_pir_mcmc_args(command: Command) -> Command {
    command
        .arg(
            Arg::new("input_path")
                .help("Input IR file (.ir) to optimize.")
                .required(true)
                .index(1)
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("iters")
                .short('n')
                .long("iters")
                .value_name("ITERS")
                .help("Number of MCMC iterations to perform.")
                .required(true)
                .value_parser(clap::value_parser!(u64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("seed")
                .short('S')
                .long("seed")
                .value_name("SEED")
                .help("Random seed.")
                .default_value("1")
                .value_parser(clap::value_parser!(u64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("OUTPUT_DIR")
                .help(
                    "Output directory for artifacts like best.ir, orig.opt.ir, best.stats.json, and trajectory.cNNN.jsonl.",
                )
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("metric")
                .long("metric")
                .value_name("METRIC")
                .help("Metric to optimize.")
                .value_parser(clap::builder::EnumValueParser::<Objective>::new())
                .default_value("nodes")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("extension_costing_mode")
                .long("extension-costing-mode")
                .value_name("MODE")
                .help(
                    "How PIR extension ops are projected before XLS optimization/g8r costing.",
                )
                .value_parser(clap::builder::EnumValueParser::<ExtensionCostingMode>::new())
                .default_value("preserve")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("g8r_postprocess_program")
                .long("g8r-postprocess-program")
                .value_name("PATH")
                .help(
                    "External postprocessor invoked as: <program> <input.aig> --output-path <output.aig>.",
                )
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("max_delay")
                .long("max-delay")
                .value_name("LEVELS")
                .help("Optional hard cap on g8r depth for g8r-based objectives.")
                .value_parser(clap::value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("max_area")
                .long("max-area")
                .value_name("GATES")
                .help("Optional hard cap on g8r live-node count for g8r-based objectives.")
                .value_parser(clap::value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("toggle_stimulus")
                .long("toggle-stimulus")
                .value_name("IRVALS_PATH")
                .help(
                    "Path to .irvals stimulus (one typed tuple per line) for toggle-based objective.",
                )
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("initial_temperature")
                .long("initial-temperature")
                .value_name("INITIAL_TEMPERATURE")
                .help("Initial temperature for MCMC (default: 5.0).")
                .default_value("5.0")
                .value_parser(clap::value_parser!(f64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("threads")
                .long("threads")
                .value_name("THREADS")
                .help("Number of parallel MCMC chains to run.")
                .value_parser(clap::value_parser!(u64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("chain_strategy")
                .long("chain-strategy")
                .value_name("CHAIN_STRATEGY")
                .help("Strategy for running multiple MCMC chains.")
                .value_parser(clap::builder::EnumValueParser::<CliChainStrategy>::new())
                .default_value("independent")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("checkpoint_iters")
                .long("checkpoint-iters")
                .value_name("CHECKPOINT_ITERS")
                .help("Iterations per synchronization segment in explore/exploit mode.")
                .default_value("5000")
                .value_parser(clap::value_parser!(u64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("progress_iters")
                .long("progress-iters")
                .value_name("PROGRESS_ITERS")
                .help("Progress logging interval in iterations (0 disables progress logs).")
                .default_value("1000")
                .value_parser(clap::value_parser!(u64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("formal_oracle")
                .long("formal-oracle")
                .value_name("BOOL")
                .help(
                    "Enable a formal equivalence oracle in addition to the interpreter oracle.",
                )
                .default_value("true")
                .value_parser(clap::value_parser!(bool))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("switching_beta1")
                .long("switching-beta1")
                .value_name("BETA1")
                .help("Linear load coefficient for weighted switching objectives.")
                .default_value("1.0")
                .value_parser(clap::value_parser!(f64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("switching_beta2")
                .long("switching-beta2")
                .value_name("BETA2")
                .help("Quadratic load coefficient for weighted switching objectives.")
                .default_value("0.0")
                .value_parser(clap::value_parser!(f64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("switching_primary_output_load")
                .long("switching-primary-output-load")
                .value_name("LOAD")
                .help("Additional load per primary-output use in weighted switching.")
                .default_value("1.0")
                .value_parser(clap::value_parser!(f64))
                .action(ArgAction::Set),
        )
}

pub fn parse_pir_mcmc_args(matches: &ArgMatches) -> PirMcmcCliArgs {
    PirMcmcCliArgs {
        input_path: matches.get_one::<String>("input_path").unwrap().to_string(),
        iters: *matches.get_one::<u64>("iters").unwrap(),
        seed: *matches.get_one::<u64>("seed").unwrap(),
        output: matches.get_one::<String>("output").cloned(),
        metric: *matches.get_one::<Objective>("metric").unwrap(),
        extension_costing_mode: *matches
            .get_one::<ExtensionCostingMode>("extension_costing_mode")
            .unwrap(),
        g8r_postprocess_program: matches
            .get_one::<String>("g8r_postprocess_program")
            .cloned(),
        max_delay: matches.get_one::<usize>("max_delay").copied(),
        max_area: matches.get_one::<usize>("max_area").copied(),
        toggle_stimulus: matches.get_one::<String>("toggle_stimulus").cloned(),
        initial_temperature: *matches.get_one::<f64>("initial_temperature").unwrap(),
        threads: matches
            .get_one::<u64>("threads")
            .copied()
            .unwrap_or(num_cpus::get() as u64),
        checkpoint_iters: *matches.get_one::<u64>("checkpoint_iters").unwrap(),
        progress_iters: *matches.get_one::<u64>("progress_iters").unwrap(),
        formal_oracle: *matches.get_one::<bool>("formal_oracle").unwrap(),
        switching_beta1: *matches.get_one::<f64>("switching_beta1").unwrap(),
        switching_beta2: *matches.get_one::<f64>("switching_beta2").unwrap(),
        switching_primary_output_load: *matches
            .get_one::<f64>("switching_primary_output_load")
            .unwrap(),
        chain_strategy: *matches
            .get_one::<CliChainStrategy>("chain_strategy")
            .unwrap(),
    }
}

/// Adds arguments for reducing stored winning provenance to an earlier prefix.
pub fn add_pir_mcmc_minimize_args(command: Command) -> Command {
    command
        .arg(
            Arg::new("run_dir")
                .help("MCMC output directory containing winning-provenance artifacts.")
                .required(true)
                .index(1)
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("retained_win_fraction")
                .long("retain-win-fraction")
                .value_name("FRACTION")
                .help("Fraction of the discovered objective win to retain.")
                .conflicts_with_all(["budget_step", "max_actions", "rollouts_per_budget"])
                .value_parser(clap::value_parser!(f64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("budget_step")
                .long("budget-step")
                .value_name("N")
                .help("Provenance-action spacing for frontier budgets.")
                .requires_all(["max_actions", "rollouts_per_budget"])
                .conflicts_with("retained_win_fraction")
                .value_parser(clap::value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("max_actions")
                .long("max-actions")
                .value_name("N")
                .help("Largest provenance-action budget to evaluate in frontier mode.")
                .requires_all(["budget_step", "rollouts_per_budget"])
                .conflicts_with("retained_win_fraction")
                .value_parser(clap::value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("rollouts_per_budget")
                .long("rollouts-per-budget")
                .value_name("N")
                .help("Independent guided short rollouts per frontier budget.")
                .requires_all(["budget_step", "max_actions"])
                .conflicts_with("retained_win_fraction")
                .value_parser(clap::value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .value_name("SEED")
                .help("Optional frontier search seed override (defaults to artifact seed).")
                .value_parser(clap::value_parser!(u64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("witness_kind_boost")
                .long("witness-kind-boost")
                .value_name("BOOST")
                .help(
                    "Extra proposal weight per winning-provenance occurrence of a transform kind.",
                )
                .default_value("4.0")
                .value_parser(clap::value_parser!(f64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("proposal_attempts_per_rewrite")
                .long("proposal-attempts-per-rewrite")
                .value_name("N")
                .help("Proposal-attempt cap per accepted rewrite in each frontier rollout.")
                .default_value("64")
                .value_parser(clap::value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("allow_artifact_postprocess_program")
                .long("allow-artifact-postprocess-program")
                .help(
                    "Allow execution of an external g8r postprocessor path persisted in the artifact manifest.",
                )
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("OUTPUT_DIR")
                .help("Directory to write minimized witness artifacts.")
                .required(true)
                .action(ArgAction::Set),
        )
}

pub fn parse_pir_mcmc_minimize_args(matches: &ArgMatches) -> PirMcmcMinimizeCliArgs {
    PirMcmcMinimizeCliArgs {
        run_dir: matches.get_one::<String>("run_dir").unwrap().to_string(),
        retained_win_fraction: matches.get_one::<f64>("retained_win_fraction").copied(),
        budget_step: matches.get_one::<usize>("budget_step").copied(),
        max_actions: matches.get_one::<usize>("max_actions").copied(),
        rollouts_per_budget: matches.get_one::<usize>("rollouts_per_budget").copied(),
        seed: matches.get_one::<u64>("seed").copied(),
        witness_kind_boost: *matches.get_one::<f64>("witness_kind_boost").unwrap(),
        proposal_attempts_per_rewrite: *matches
            .get_one::<usize>("proposal_attempts_per_rewrite")
            .unwrap(),
        allow_artifact_postprocess_program: matches.get_flag("allow_artifact_postprocess_program"),
        output: matches.get_one::<String>("output").unwrap().to_string(),
    }
}

fn optimize_ir_text(
    ir_text: &str,
    top: &str,
    extension_costing_mode: ExtensionCostingMode,
) -> Result<String> {
    let mut p = xlsynth_pir::ir_parser::Parser::new(ir_text);
    let pir_pkg = p
        .parse_and_validate_package()
        .map_err(|e| anyhow::anyhow!("PIR parse_and_validate_package failed: {:?}", e))?;
    let optimized_pir_pkg = super::optimize_pir_package_via_xls_with_extension_mode(
        &pir_pkg,
        top,
        extension_costing_mode,
    )?;
    Ok(optimized_pir_pkg.to_string())
}

fn emit_pkg_text_toposorted(pkg: &Package) -> Result<String> {
    let mut pkg = pkg.clone();
    for member in pkg.members.iter_mut() {
        match member {
            PackageMember::Function(f) => {
                compact_and_toposort_in_place(f)
                    .map_err(|e| anyhow::anyhow!("compact_and_toposort_in_place failed: {}", e))?;
            }
            PackageMember::Block { func, .. } => {
                compact_and_toposort_in_place(func)
                    .map_err(|e| anyhow::anyhow!("compact_and_toposort_in_place failed: {}", e))?;
            }
        }
    }
    Ok(pkg.to_string())
}

struct GatifiedArtifacts {
    g8r_text: String,
    raw_stats: SummaryStats,
    gate_fn: GateFn,
    schema: GateFnInterfaceSchema,
}

#[derive(Debug, serde::Serialize)]
struct PostprocessStatsOutput {
    and_nodes: usize,
    depth: usize,
    fanout_histogram: std::collections::BTreeMap<usize, usize>,
}

impl From<&AigStats> for PostprocessStatsOutput {
    fn from(value: &AigStats) -> Self {
        Self {
            and_nodes: value.and_nodes,
            depth: value.max_depth,
            fanout_histogram: value.fanout_histogram.clone(),
        }
    }
}

fn gatify_ir_text_to_artifacts(ir_text: &str) -> Result<GatifiedArtifacts> {
    let mut parser = ir_parser::Parser::new(ir_text);
    let pir_pkg = parser
        .parse_and_validate_package()
        .map_err(|e| anyhow::anyhow!("PIR parse_and_validate_package failed: {:?}", e))?;
    let top_fn = pir_pkg
        .get_top_fn()
        .ok_or_else(|| anyhow::anyhow!("No top function found in PIR package"))?;

    let gatify_options = GatifyOptions {
        fold: true,
        hash: true,
        check_equivalence: false,
        adder_mapping: AdderMapping::default(),
        array_index_lowering_strategy: Default::default(),
        mul_adder_mapping: None,
        range_info: None,
        enable_rewrite_carry_out: false,
        enable_rewrite_prio_encode: false,
        enable_rewrite_nary_add: false,
        enable_rewrite_mask_low: false,
    };
    let gatify_output = ir2gate::gatify(top_fn, gatify_options)
        .map_err(|e| anyhow::anyhow!("ir2gate::gatify failed: {}", e))?;
    let gate_fn = gatify_output.gate_fn;
    let raw_stats = get_summary_stats::get_summary_stats(&gate_fn);
    let schema = GateFnInterfaceSchema::from_pir_fn(top_fn)
        .map_err(|e| anyhow::anyhow!("failed to derive gate interface schema: {}", e))?;
    Ok(GatifiedArtifacts {
        g8r_text: gate_fn.to_string(),
        raw_stats,
        gate_fn,
        schema,
    })
}

fn maybe_write_postprocess_artifacts(
    output_dir: &PathBuf,
    stem: &str,
    gate_fn: &GateFn,
    schema: &GateFnInterfaceSchema,
    g8r_evaluation_mode: &G8rEvaluationMode,
) -> Result<()> {
    if g8r_evaluation_mode.external_postprocess_program().is_none() {
        return Ok(());
    }
    let post = postprocess_gate_fn_for_artifact(gate_fn, schema, g8r_evaluation_mode)?;
    let post_aig_path = output_dir.join(format!("{stem}.post.aig"));
    std::fs::write(&post_aig_path, &post.bytes)
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", post_aig_path.display(), e))?;
    let post_stats_path = output_dir.join(format!("{stem}.post.stats.json"));
    let post_stats_json = serde_json::to_string_pretty(&PostprocessStatsOutput::from(&post.stats))
        .expect("serialize postprocess AIG stats");
    std::fs::write(&post_stats_path, post_stats_json.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", post_stats_path.display(), e))?;
    Ok(())
}

fn write_witness_artifacts(
    output_dir: &PathBuf,
    package_template: &Package,
    top_fn_name: &str,
    witness_fn: &xlsynth_pir::ir::Fn,
    extension_costing_mode: ExtensionCostingMode,
    g8r_evaluation_mode: &G8rEvaluationMode,
) -> Result<()> {
    std::fs::create_dir_all(output_dir).map_err(|e| {
        anyhow::anyhow!(
            "Failed to create minimization output directory {}: {}",
            output_dir.display(),
            e
        )
    })?;

    let mut witness_pkg = package_template.clone();
    let witness_top = witness_pkg.get_fn_mut(top_fn_name).ok_or_else(|| {
        anyhow::anyhow!(
            "Top function '{}' not found in artifact package template",
            top_fn_name
        )
    })?;
    *witness_top = witness_fn.clone();

    let witness_ir_text = emit_pkg_text_toposorted(&witness_pkg)?;
    let witness_ir_path = output_dir.join("witness.ir");
    std::fs::write(&witness_ir_path, witness_ir_text.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", witness_ir_path.display(), e))?;

    let witness_opt_ir_text =
        optimize_ir_text(&witness_ir_text, top_fn_name, extension_costing_mode)?;
    let witness_opt_ir_path = output_dir.join("witness.opt.ir");
    std::fs::write(&witness_opt_ir_path, witness_opt_ir_text.as_bytes()).map_err(|e| {
        anyhow::anyhow!("Failed to write {}: {:?}", witness_opt_ir_path.display(), e)
    })?;

    let witness_artifacts = gatify_ir_text_to_artifacts(&witness_opt_ir_text)?;
    let witness_g8r_path = output_dir.join("witness.g8r");
    std::fs::write(&witness_g8r_path, witness_artifacts.g8r_text.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", witness_g8r_path.display(), e))?;
    let witness_stats_path = output_dir.join("witness.stats.json");
    let witness_stats_json =
        serde_json::to_string_pretty(&witness_artifacts.raw_stats).expect("serialize SummaryStats");
    std::fs::write(&witness_stats_path, witness_stats_json.as_bytes()).map_err(|e| {
        anyhow::anyhow!("Failed to write {}: {:?}", witness_stats_path.display(), e)
    })?;
    maybe_write_postprocess_artifacts(
        output_dir,
        "witness",
        &witness_artifacts.gate_fn,
        &witness_artifacts.schema,
        g8r_evaluation_mode,
    )?;
    Ok(())
}

fn budget_witness_json(witness: &PirMcmcBudgetWitness) -> serde_json::Value {
    serde_json::json!({
        "provenance_action_count": witness.provenance_action_count,
        "metric": witness.metric,
        "absolute_win": witness.absolute_win,
        "win_percent_vs_origin": witness.win_percent_vs_origin,
        "retained_win_fraction": witness.retained_win_fraction,
        "cost": {
            "pir_nodes": witness.witness_cost.pir_nodes,
            "g8r_nodes": witness.witness_cost.g8r_nodes,
            "g8r_depth": witness.witness_cost.g8r_depth,
            "g8r_le_graph_milli": witness.witness_cost.g8r_le_graph_milli,
            "g8r_gate_output_toggles": witness.witness_cost.g8r_gate_output_toggles,
            "g8r_weighted_switching_milli": witness.witness_cost.g8r_weighted_switching_milli,
            "g8r_post_and_nodes": witness.witness_cost.g8r_post_and_nodes,
            "g8r_post_depth": witness.witness_cost.g8r_post_depth,
            "g8r_post_le_graph_milli": witness.witness_cost.g8r_post_le_graph_milli,
            "g8r_post_gate_output_toggles": witness.witness_cost.g8r_post_gate_output_toggles,
            "g8r_post_weighted_switching_milli": witness.witness_cost.g8r_post_weighted_switching_milli,
        },
    })
}

pub fn run_pir_mcmc_driver<F>(cli: PirMcmcCliArgs, mut report: F) -> Result<()>
where
    F: FnMut(String),
{
    report(format!("PIR MCMC Driver started with args: {:?}", cli));
    report(format!(
        "PIR MCMC extension costing mode: {}",
        cli.extension_costing_mode.value_name()
    ));
    let g8r_evaluation_mode = match cli.g8r_postprocess_program.clone() {
        Some(program) => {
            G8rEvaluationMode::ExternalPostprocess { program }.canonicalized_for_persistence()?
        }
        None => G8rEvaluationMode::Builtin,
    };

    // Parse IR package.
    let input_path = PathBuf::from(&cli.input_path);
    let mut pkg = ir_parser::parse_and_validate_path_to_package(&input_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to parse PIR package from '{}': {:?}",
            input_path.display(),
            e
        )
    })?;

    let top_fn = pkg
        .get_top_fn()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("No top function found in PIR package"))?;

    let toggle_stimulus_values = match &cli.toggle_stimulus {
        Some(path) => Some(parse_irvals_tuple_file(PathBuf::from(path).as_path())?),
        None => None,
    };
    if cli.metric.needs_toggle_stimulus() && toggle_stimulus_values.is_none() {
        return Err(anyhow::anyhow!(
            "--toggle-stimulus is required when --metric={}",
            cli.metric.value_name()
        ));
    }
    if !cli.metric.needs_toggle_stimulus() && toggle_stimulus_values.is_some() {
        return Err(anyhow::anyhow!(
            "--toggle-stimulus is not valid with --metric={}",
            cli.metric.value_name()
        ));
    }
    validate_constraint_configuration(
        cli.metric,
        ConstraintLimits {
            max_delay: cli.max_delay,
            max_area: cli.max_area,
        },
    )?;
    let prepared_toggle_stimulus = toggle_stimulus_values
        .as_ref()
        .map(|samples| lower_toggle_stimulus_for_fn(samples, &top_fn))
        .transpose()?;

    let weighted_switching_options =
        xlsynth_g8r::aig_sim::count_toggles::WeightedSwitchingOptions {
            beta1: cli.switching_beta1,
            beta2: cli.switching_beta2,
            primary_output_load: cli.switching_primary_output_load,
        };
    let initial_cost = cost_with_effort_options_toggle_stimulus_extension_mode_and_evaluator(
        &top_fn,
        cli.metric,
        prepared_toggle_stimulus.as_deref(),
        &weighted_switching_options,
        cli.extension_costing_mode,
        &g8r_evaluation_mode,
    )?;
    let effective_constraints = effective_constraint_limits(
        cli.metric,
        ConstraintLimits {
            max_delay: cli.max_delay,
            max_area: cli.max_area,
        },
        &initial_cost,
    );
    let initial_score = search_score(&initial_cost, cli.metric, effective_constraints);
    if cli.metric.uses_postprocessed_costing() && cli.metric.needs_weighted_switching() {
        report(format!(
            "Successfully loaded top function. Initial stats: pir_nodes={}, g8r_post_and_nodes={}, g8r_post_depth={}, g8r_post_weighted_switching_milli={}, score={}",
            initial_cost.pir_nodes,
            initial_cost.g8r_post_and_nodes,
            initial_cost.g8r_post_depth,
            initial_cost.g8r_post_weighted_switching_milli,
            format_search_score(initial_score),
        ));
    } else if cli.metric.uses_postprocessed_costing() && cli.metric.needs_toggle_stimulus() {
        report(format!(
            "Successfully loaded top function. Initial stats: pir_nodes={}, g8r_post_and_nodes={}, g8r_post_depth={}, g8r_post_gate_output_toggles={}, score={}",
            initial_cost.pir_nodes,
            initial_cost.g8r_post_and_nodes,
            initial_cost.g8r_post_depth,
            initial_cost.g8r_post_gate_output_toggles,
            format_search_score(initial_score),
        ));
    } else if cli.metric.uses_postprocessed_costing() {
        report(format!(
            "Successfully loaded top function. Initial stats: pir_nodes={}, g8r_post_and_nodes={}, g8r_post_depth={}, score={}",
            initial_cost.pir_nodes,
            initial_cost.g8r_post_and_nodes,
            initial_cost.g8r_post_depth,
            format_search_score(initial_score),
        ));
    } else if cli.metric.needs_weighted_switching() {
        report(format!(
            "Successfully loaded top function. Initial stats: pir_nodes={}, g8r_nodes={}, g8r_depth={}, g8r_weighted_switching_milli={}, score={}",
            initial_cost.pir_nodes,
            initial_cost.g8r_nodes,
            initial_cost.g8r_depth,
            initial_cost.g8r_weighted_switching_milli,
            format_search_score(initial_score),
        ));
    } else if cli.metric.needs_toggle_stimulus() {
        report(format!(
            "Successfully loaded top function. Initial stats: pir_nodes={}, g8r_nodes={}, g8r_depth={}, g8r_gate_output_toggles={}, score={}",
            initial_cost.pir_nodes,
            initial_cost.g8r_nodes,
            initial_cost.g8r_depth,
            initial_cost.g8r_gate_output_toggles,
            format_search_score(initial_score),
        ));
    } else {
        report(format!(
            "Successfully loaded top function. Initial stats: pir_nodes={}, g8r_nodes={}, g8r_depth={}, score={}",
            initial_cost.pir_nodes,
            initial_cost.g8r_nodes,
            initial_cost.g8r_depth,
            format_search_score(initial_score),
        ));
    }

    // Determine output directory and paths.
    //
    // We always treat `--output` as a directory so we have a stable place to
    // write the full set of artifacts.
    let output_ir_filename = "best.ir";

    let (output_dir, _temp_dir_holder): (PathBuf, Option<tempfile::TempDir>) = match &cli.output {
        Some(path_str) => {
            let dir = PathBuf::from(path_str);
            std::fs::create_dir_all(&dir)?;
            (dir, None)
        }
        None => {
            let temp_dir = Builder::new().prefix("pir_mcmc_output_").tempdir()?;
            report(format!(
                "No output path specified, using temp dir: {}",
                temp_dir.path().display()
            ));
            (temp_dir.path().to_path_buf(), Some(temp_dir))
        }
    };

    let output_ir_path = output_dir.join(output_ir_filename);
    let extension_costing_mode = cli.extension_costing_mode;

    let opts = RunOptions {
        max_iters: cli.iters,
        threads: cli.threads,
        chain_strategy: cli.chain_strategy.into(),
        checkpoint_iters: cli.checkpoint_iters,
        progress_iters: cli.progress_iters,
        seed: cli.seed,
        initial_temperature: cli.initial_temperature,
        objective: cli.metric,
        max_allowed_depth: cli.max_delay,
        max_allowed_area: cli.max_area,
        weighted_switching_options,
        extension_costing_mode,
        g8r_evaluation_mode: g8r_evaluation_mode.clone(),
        enable_formal_oracle: cli.formal_oracle,
        trajectory_dir: Some(output_dir.clone()),
        toggle_stimulus: toggle_stimulus_values,
    };

    // Optional checkpoint writer: overwrites best.* artifacts periodically so
    // users can inspect best-so-far while the run is still running.
    let (checkpoint_tx, checkpoint_rx) = if cli.checkpoint_iters > 0 {
        let (tx, rx) = mpsc::channel::<CheckpointMsg>();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    let shared_best = if cli.checkpoint_iters > 0 {
        Some(Arc::new(Best::new(initial_score, top_fn.clone())))
    } else {
        None
    };

    // Emit original artifacts.
    let orig_ir_text = emit_pkg_text_toposorted(&pkg)?;
    let orig_ir_path = output_dir.join("orig.ir");
    std::fs::write(&orig_ir_path, orig_ir_text.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", orig_ir_path.display(), e))?;

    let orig_top_name = top_fn.name.clone();
    let orig_opt_ir_text = optimize_ir_text(&orig_ir_text, &orig_top_name, extension_costing_mode)?;
    let orig_opt_ir_path = output_dir.join("orig.opt.ir");
    std::fs::write(&orig_opt_ir_path, orig_opt_ir_text.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", orig_opt_ir_path.display(), e))?;

    let orig_artifacts = gatify_ir_text_to_artifacts(&orig_opt_ir_text)?;
    let orig_g8r_path = output_dir.join("orig.g8r");
    std::fs::write(&orig_g8r_path, orig_artifacts.g8r_text.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", orig_g8r_path.display(), e))?;
    let orig_stats_path = output_dir.join("orig.stats.json");
    let orig_stats_json =
        serde_json::to_string_pretty(&orig_artifacts.raw_stats).expect("serialize SummaryStats");
    std::fs::write(&orig_stats_path, orig_stats_json.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", orig_stats_path.display(), e))?;
    maybe_write_postprocess_artifacts(
        &output_dir,
        "orig",
        &orig_artifacts.gate_fn,
        &orig_artifacts.schema,
        &g8r_evaluation_mode,
    )?;

    let pkg_template = Arc::new(pkg.clone());
    let output_dir_for_thread = output_dir.clone();
    let orig_top_name_for_thread = orig_top_name.clone();
    let extension_costing_mode_for_thread = extension_costing_mode;
    let g8r_evaluation_mode_for_thread = g8r_evaluation_mode.clone();

    let writer_handle = if let (Some(best), Some(rx)) = (shared_best.clone(), checkpoint_rx) {
        Some(std::thread::spawn(move || {
            // Track last written metric to reduce redundant writes when multiple
            // chains hit the same checkpoint boundary.
            let mut last_written: Option<crate::SearchScore> = None;
            // Monotonic snapshot write index (per run) so filenames sort by
            // checkpoint write order across chains.
            let mut write_index: u64 = 0;
            // Track the last "new global best" message so snapshots use the
            // chain/iteration that actually produced the improvement, even if a
            // later periodic tick triggers the write.
            let mut last_best_update_msg: Option<CheckpointMsg> = None;
            while let Ok(msg) = rx.recv() {
                if msg.kind == CheckpointKind::GlobalBestUpdate {
                    last_best_update_msg = Some(msg);
                }
                let cur_score = best.score();
                if last_written == Some(cur_score) {
                    continue;
                }
                last_written = Some(cur_score);

                let best_fn = best.get();
                let mut pkg = (*pkg_template).clone();
                if let Some(top_mut) = pkg.get_fn_mut(&best_fn.name) {
                    *top_mut = best_fn;
                } else {
                    // If the function is missing, skip this checkpoint write.
                    continue;
                }

                let best_ir_text = match emit_pkg_text_toposorted(&pkg) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let best_ir_path = output_dir_for_thread.join("best.ir");
                let _ = std::fs::write(&best_ir_path, best_ir_text.as_bytes());

                let best_opt_ir_text = match optimize_ir_text(
                    &best_ir_text,
                    &orig_top_name_for_thread,
                    extension_costing_mode_for_thread,
                ) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let best_opt_ir_path = output_dir_for_thread.join("best.opt.ir");
                let _ = std::fs::write(&best_opt_ir_path, best_opt_ir_text.as_bytes());

                // Also snapshot each new "best so far" optimized IR so users can
                // inspect the trajectory of improvements over time.
                //
                // We prefer the chain/iter from the most recent GlobalBestUpdate
                // message; if not available, fall back to the message that
                // triggered this write.
                let snapshot_msg = last_best_update_msg.unwrap_or(msg);
                write_index = write_index.saturating_add(1);
                let best_opt_ir_snapshot_path = output_dir_for_thread.join(format!(
                    "best.w{:06}.c{:03}-i{:06}.opt.ir",
                    write_index, snapshot_msg.chain_no, snapshot_msg.global_iter
                ));
                let _ = std::fs::write(&best_opt_ir_snapshot_path, best_opt_ir_text.as_bytes());

                let best_artifacts = match gatify_ir_text_to_artifacts(&best_opt_ir_text) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let best_g8r_path = output_dir_for_thread.join("best.g8r");
                let _ = std::fs::write(&best_g8r_path, best_artifacts.g8r_text.as_bytes());

                let best_stats_path = output_dir_for_thread.join("best.stats.json");
                if let Ok(stats_json) = serde_json::to_string_pretty(&best_artifacts.raw_stats) {
                    let _ = std::fs::write(&best_stats_path, stats_json.as_bytes());

                    let best_stats_snapshot_path = output_dir_for_thread.join(format!(
                        "best.w{:06}.c{:03}-i{:06}.stats.json",
                        write_index, snapshot_msg.chain_no, snapshot_msg.global_iter
                    ));
                    let _ = std::fs::write(&best_stats_snapshot_path, stats_json.as_bytes());
                }
                let _ = maybe_write_postprocess_artifacts(
                    &output_dir_for_thread,
                    "best",
                    &best_artifacts.gate_fn,
                    &best_artifacts.schema,
                    &g8r_evaluation_mode_for_thread,
                );
                let snapshot_stem = format!(
                    "best.w{:06}.c{:03}-i{:06}",
                    write_index, snapshot_msg.chain_no, snapshot_msg.global_iter
                );
                let _ = maybe_write_postprocess_artifacts(
                    &output_dir_for_thread,
                    &snapshot_stem,
                    &best_artifacts.gate_fn,
                    &best_artifacts.schema,
                    &g8r_evaluation_mode_for_thread,
                );
            }
        }))
    } else {
        None
    };

    let (result, recorded_artifact) = match validate_pir_mcmc_artifact_run_options(&opts) {
        Ok(()) => {
            let run_output = run_pir_mcmc_with_artifact_and_observers(
                top_fn,
                opts,
                shared_best.clone(),
                checkpoint_tx,
            )?;
            (run_output.result, Some(run_output.artifact))
        }
        Err(e) => {
            report(format!(
                "No minimizable winning-provenance artifact emitted for this run: {}",
                e
            ));
            (
                run_pir_mcmc_with_shared_best(
                    top_fn,
                    opts,
                    shared_best.clone(),
                    checkpoint_tx,
                    None,
                )?,
                None,
            )
        }
    };

    // Stop checkpoint writer cleanly before final artifact emission.
    if let Some(h) = writer_handle {
        let _ = h.join();
    }

    match cli.metric {
        Objective::Nodes => {
            report(format!(
                "PIR MCMC finished. Best stats: pir_nodes={}",
                result.best_cost.pir_nodes
            ));
        }
        Objective::G8rNodes => {
            report(format!(
                "PIR MCMC finished. Best stats: g8r_nodes={}",
                result.best_cost.g8r_nodes
            ));
        }
        Objective::G8rNodesTimesDepth => {
            let product = (result.best_cost.g8r_nodes as u128)
                .saturating_mul(result.best_cost.g8r_depth as u128);
            report(format!(
                "PIR MCMC finished. Best stats: g8r_nodes={}, g8r_depth={}, product={}",
                result.best_cost.g8r_nodes, result.best_cost.g8r_depth, product,
            ));
        }
        Objective::G8rNodesTimesDepthTimesToggles => {
            let nd = (result.best_cost.g8r_nodes as u128)
                .saturating_mul(result.best_cost.g8r_depth as u128);
            let metric = nd.saturating_mul(result.best_cost.g8r_gate_output_toggles as u128);
            report(format!(
                "PIR MCMC finished. Best stats: g8r_nodes={}, g8r_depth={}, g8r_gate_output_toggles={}, metric={}",
                result.best_cost.g8r_nodes,
                result.best_cost.g8r_depth,
                result.best_cost.g8r_gate_output_toggles,
                metric,
            ));
        }
        Objective::G8rLeGraph => {
            report(format!(
                "PIR MCMC finished. Best stats: g8r_le_graph={:.3} (metric_milli={})",
                (result.best_cost.g8r_le_graph_milli as f64) / 1000.0,
                result.best_cost.g8r_le_graph_milli,
            ));
        }
        Objective::G8rLeGraphTimesNodes => {
            let metric = (result.best_cost.g8r_le_graph_milli as u128)
                .saturating_mul(result.best_cost.g8r_nodes as u128);
            report(format!(
                "PIR MCMC finished. Best stats: g8r_le_graph={:.3}, g8r_nodes={}, metric={}",
                (result.best_cost.g8r_le_graph_milli as f64) / 1000.0,
                result.best_cost.g8r_nodes,
                metric,
            ));
        }
        Objective::G8rLeGraphTimesProduct => {
            let product = (result.best_cost.g8r_nodes as u128)
                .saturating_mul(result.best_cost.g8r_depth as u128);
            let metric = (result.best_cost.g8r_le_graph_milli as u128).saturating_mul(product);
            report(format!(
                "PIR MCMC finished. Best stats: g8r_le_graph={:.3}, g8r_nodes={}, g8r_depth={}, product={}, metric={}",
                (result.best_cost.g8r_le_graph_milli as f64) / 1000.0,
                result.best_cost.g8r_nodes,
                result.best_cost.g8r_depth,
                product,
                metric,
            ));
        }
        Objective::G8rWeightedSwitching => {
            report(format!(
                "PIR MCMC finished. Best stats: g8r_weighted_switching_milli={}",
                result.best_cost.g8r_weighted_switching_milli,
            ));
        }
        Objective::G8rNodesTimesWeightedSwitchingNoDepthRegress => {
            let metric = (result.best_cost.g8r_nodes as u128)
                .saturating_mul(result.best_cost.g8r_weighted_switching_milli);
            report(format!(
                "PIR MCMC finished. Best stats: g8r_nodes={}, g8r_depth={} (non-regressing), g8r_weighted_switching_milli={}, metric={}",
                result.best_cost.g8r_nodes,
                result.best_cost.g8r_depth,
                result.best_cost.g8r_weighted_switching_milli,
                metric,
            ));
        }
        Objective::G8rPostAndNodes => {
            report(format!(
                "PIR MCMC finished. Best stats: g8r_post_and_nodes={}",
                result.best_cost.g8r_post_and_nodes
            ));
        }
        Objective::G8rPostAndNodesTimesDepth => {
            let product = (result.best_cost.g8r_post_and_nodes as u128)
                .saturating_mul(result.best_cost.g8r_post_depth as u128);
            report(format!(
                "PIR MCMC finished. Best stats: g8r_post_and_nodes={}, g8r_post_depth={}, product={}",
                result.best_cost.g8r_post_and_nodes, result.best_cost.g8r_post_depth, product,
            ));
        }
        Objective::G8rPostAndNodesTimesDepthTimesToggles => {
            let nd = (result.best_cost.g8r_post_and_nodes as u128)
                .saturating_mul(result.best_cost.g8r_post_depth as u128);
            let metric = nd.saturating_mul(result.best_cost.g8r_post_gate_output_toggles as u128);
            report(format!(
                "PIR MCMC finished. Best stats: g8r_post_and_nodes={}, g8r_post_depth={}, g8r_post_gate_output_toggles={}, metric={}",
                result.best_cost.g8r_post_and_nodes,
                result.best_cost.g8r_post_depth,
                result.best_cost.g8r_post_gate_output_toggles,
                metric,
            ));
        }
        Objective::G8rPostLeGraph => {
            report(format!(
                "PIR MCMC finished. Best stats: g8r_post_le_graph={:.3} (metric_milli={})",
                (result.best_cost.g8r_post_le_graph_milli as f64) / 1000.0,
                result.best_cost.g8r_post_le_graph_milli,
            ));
        }
        Objective::G8rPostLeGraphTimesAndNodes => {
            let metric = (result.best_cost.g8r_post_le_graph_milli as u128)
                .saturating_mul(result.best_cost.g8r_post_and_nodes as u128);
            report(format!(
                "PIR MCMC finished. Best stats: g8r_post_le_graph={:.3}, g8r_post_and_nodes={}, metric={}",
                (result.best_cost.g8r_post_le_graph_milli as f64) / 1000.0,
                result.best_cost.g8r_post_and_nodes,
                metric,
            ));
        }
        Objective::G8rPostLeGraphTimesProduct => {
            let product = (result.best_cost.g8r_post_and_nodes as u128)
                .saturating_mul(result.best_cost.g8r_post_depth as u128);
            let metric = (result.best_cost.g8r_post_le_graph_milli as u128).saturating_mul(product);
            report(format!(
                "PIR MCMC finished. Best stats: g8r_post_le_graph={:.3}, g8r_post_and_nodes={}, g8r_post_depth={}, product={}, metric={}",
                (result.best_cost.g8r_post_le_graph_milli as f64) / 1000.0,
                result.best_cost.g8r_post_and_nodes,
                result.best_cost.g8r_post_depth,
                product,
                metric,
            ));
        }
        Objective::G8rPostWeightedSwitching => {
            report(format!(
                "PIR MCMC finished. Best stats: g8r_post_weighted_switching_milli={}",
                result.best_cost.g8r_post_weighted_switching_milli,
            ));
        }
        Objective::G8rPostAndNodesTimesWeightedSwitchingNoDepthRegress => {
            let metric = (result.best_cost.g8r_post_and_nodes as u128)
                .saturating_mul(result.best_cost.g8r_post_weighted_switching_milli);
            report(format!(
                "PIR MCMC finished. Best stats: g8r_post_and_nodes={}, g8r_post_depth={} (non-regressing), g8r_post_weighted_switching_milli={}, metric={}",
                result.best_cost.g8r_post_and_nodes,
                result.best_cost.g8r_post_depth,
                result.best_cost.g8r_post_weighted_switching_milli,
                metric,
            ));
        }
    }

    let final_score = search_score(&result.best_cost, cli.metric, effective_constraints);
    if !final_score.feasible() {
        report(format!(
            "No feasible solution found; best result remains infeasible: {}",
            format_search_score(final_score),
        ));
    }

    // Replace the top function in the package with the optimized version.
    {
        let top_name = result.best_fn.name.clone();
        let top_mut = pkg.get_fn_mut(&top_name).ok_or_else(|| {
            anyhow::anyhow!("Top function '{}' not found for replacement", top_name)
        })?;
        *top_mut = result.best_fn;
    }

    report(format!(
        "Writing optimized PIR package to {}",
        output_ir_path.display()
    ));
    let mut f_ir = std::fs::File::create(&output_ir_path)?;
    let pkg_text_out = emit_pkg_text_toposorted(&pkg)?;
    f_ir.write_all(pkg_text_out.as_bytes())?;
    report(format!(
        "Successfully wrote optimized PIR to {}",
        output_ir_path.display()
    ));

    // Emit best artifacts (alongside best.ir even if output_ir_path was
    // customized).
    let best_ir_text = emit_pkg_text_toposorted(&pkg)?;
    let best_ir_path = output_dir.join("best.ir");
    std::fs::write(&best_ir_path, best_ir_text.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", best_ir_path.display(), e))?;

    let best_top_name = orig_top_name;
    let best_opt_ir_text = optimize_ir_text(&best_ir_text, &best_top_name, extension_costing_mode)?;
    let best_opt_ir_path = output_dir.join("best.opt.ir");
    std::fs::write(&best_opt_ir_path, best_opt_ir_text.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", best_opt_ir_path.display(), e))?;

    let best_artifacts = gatify_ir_text_to_artifacts(&best_opt_ir_text)?;
    let best_g8r_path = output_dir.join("best.g8r");
    std::fs::write(&best_g8r_path, best_artifacts.g8r_text.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", best_g8r_path.display(), e))?;
    let best_stats_path = output_dir.join("best.stats.json");
    let best_stats_json =
        serde_json::to_string_pretty(&best_artifacts.raw_stats).expect("serialize SummaryStats");
    std::fs::write(&best_stats_path, best_stats_json.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", best_stats_path.display(), e))?;
    maybe_write_postprocess_artifacts(
        &output_dir,
        "best",
        &best_artifacts.gate_fn,
        &best_artifacts.schema,
        &g8r_evaluation_mode,
    )?;

    if let Some(artifact) = recorded_artifact.as_ref() {
        let artifact_dir = write_pir_mcmc_artifact_dir(artifact, &pkg, &output_dir)?;
        report(format!(
            "Wrote minimizable winning-provenance artifact to {}",
            artifact_dir.display()
        ));
    }

    Ok(())
}

/// Loads stored winning provenance, minimizes it to an earlier prefix, and
/// emits package-level artifacts for the selected witness.
pub fn run_pir_mcmc_minimize_driver<F>(cli: PirMcmcMinimizeCliArgs, mut report: F) -> Result<()>
where
    F: FnMut(String),
{
    let run_dir = PathBuf::from(&cli.run_dir);
    let loaded = read_pir_mcmc_artifact_dir(&run_dir)?;
    if let Some(program) = loaded
        .artifact
        .run_options
        .g8r_evaluation_mode
        .external_postprocess_program()
        && !cli.allow_artifact_postprocess_program
    {
        return Err(anyhow::anyhow!(
            "artifact requests external g8r postprocessor '{}'; rerun with --allow-artifact-postprocess-program to execute it",
            program
        ));
    }
    let frontier_mode = match (cli.budget_step, cli.max_actions, cli.rollouts_per_budget) {
        (Some(budget_step), Some(max_actions), Some(rollouts_per_budget)) => {
            Some((budget_step, max_actions, rollouts_per_budget))
        }
        (None, None, None) => None,
        _ => {
            return Err(anyhow::anyhow!(
                "frontier mode requires --budget-step, --max-actions, and --rollouts-per-budget together"
            ));
        }
    };
    if cli.retained_win_fraction.is_some() == frontier_mode.is_some() {
        return Err(anyhow::anyhow!(
            "choose exactly one minimization mode: --retain-win-fraction or frontier budget flags"
        ));
    }

    let output_dir = PathBuf::from(&cli.output);
    let extension_costing_mode = loaded.artifact.run_options.extension_costing_mode;
    let g8r_evaluation_mode = &loaded.artifact.run_options.g8r_evaluation_mode;

    if let Some((budget_step, max_actions, rollouts_per_budget)) = frontier_mode {
        std::fs::create_dir_all(&output_dir).map_err(|e| {
            anyhow::anyhow!(
                "Failed to create minimization output directory {}: {}",
                output_dir.display(),
                e
            )
        })?;
        let frontier = search_winning_budget_frontier(
            &loaded.artifact,
            PirMcmcBudgetFrontierOptions {
                budget_step,
                max_actions,
                rollouts_per_budget,
                seed: cli.seed.unwrap_or(loaded.artifact.run_options.seed),
                witness_kind_boost: cli.witness_kind_boost,
                proposal_attempts_per_rewrite: cli.proposal_attempts_per_rewrite,
            },
        )?;

        let mut point_summaries = Vec::with_capacity(frontier.points.len());
        for point in frontier.points.iter() {
            let point_dir = output_dir.join(format!("budget-{:04}", point.action_budget));
            write_witness_artifacts(
                &point_dir,
                &loaded.package_template,
                &loaded.top_fn_name,
                &point.guided.witness_fn,
                extension_costing_mode,
                g8r_evaluation_mode,
            )?;
            point_summaries.push(serde_json::json!({
                "action_budget": point.action_budget,
                "guided": budget_witness_json(&point.guided),
                "prefix_baseline": budget_witness_json(&point.prefix_baseline),
                "artifact_dir": point_dir
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default(),
            }));
        }
        let summary = serde_json::json!({
            "mode": "budget_frontier",
            "origin_metric": frontier.origin_metric,
            "winner_metric": frontier.winner_metric,
            "original_winning_provenance_len": frontier.original_winning_provenance_len,
            "search": {
                "budget_step": budget_step,
                "max_actions": max_actions,
                "rollouts_per_budget": rollouts_per_budget,
                "seed": cli.seed.unwrap_or(loaded.artifact.run_options.seed),
                "witness_kind_boost": cli.witness_kind_boost,
                "proposal_attempts_per_rewrite": cli.proposal_attempts_per_rewrite,
            },
            "points": point_summaries,
        });
        let summary_path = output_dir.join("summary.json");
        let summary_json =
            serde_json::to_string_pretty(&summary).expect("serialize frontier summary");
        std::fs::write(&summary_path, summary_json.as_bytes())
            .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", summary_path.display(), e))?;
        report(format!(
            "PIR MCMC guided frontier searched {} budgets from {} to {} provenance actions",
            frontier.points.len(),
            budget_step,
            max_actions
        ));
        for point in frontier.points.iter() {
            report(format!(
                "budget <= {:>4}: guided metric={} retained={:.6} actions={} | prefix metric={} retained={:.6} actions={}",
                point.action_budget,
                point.guided.metric,
                point.guided.retained_win_fraction,
                point.guided.provenance_action_count,
                point.prefix_baseline.metric,
                point.prefix_baseline.retained_win_fraction,
                point.prefix_baseline.provenance_action_count,
            ));
        }
        report(format!(
            "Wrote frontier witness artifacts to {}",
            output_dir.display()
        ));
        return Ok(());
    }

    let minimized = minimize_winning_prefix(
        &loaded.artifact,
        PirMcmcPrefixMinimizeOptions {
            retained_win_fraction: cli.retained_win_fraction.unwrap(),
        },
    )?;

    write_witness_artifacts(
        &output_dir,
        &loaded.package_template,
        &loaded.top_fn_name,
        &minimized.witness_fn,
        extension_costing_mode,
        g8r_evaluation_mode,
    )?;

    let summary = serde_json::json!({
        "mode": "retain_win_fraction",
        "requested_retained_win_fraction": minimized.requested_retained_win_fraction,
        "actual_retained_win_fraction": minimized.actual_retained_win_fraction,
        "provenance_action_count": minimized.provenance_action_count,
        "original_winning_provenance_len": minimized.original_winning_provenance_len,
        "origin_metric": minimized.origin_metric,
        "winner_metric": minimized.winner_metric,
        "witness_metric": minimized.witness_metric,
        "witness_cost": {
            "pir_nodes": minimized.witness_cost.pir_nodes,
            "g8r_nodes": minimized.witness_cost.g8r_nodes,
            "g8r_depth": minimized.witness_cost.g8r_depth,
            "g8r_le_graph_milli": minimized.witness_cost.g8r_le_graph_milli,
            "g8r_gate_output_toggles": minimized.witness_cost.g8r_gate_output_toggles,
            "g8r_weighted_switching_milli": minimized.witness_cost.g8r_weighted_switching_milli,
            "g8r_post_and_nodes": minimized.witness_cost.g8r_post_and_nodes,
            "g8r_post_depth": minimized.witness_cost.g8r_post_depth,
            "g8r_post_le_graph_milli": minimized.witness_cost.g8r_post_le_graph_milli,
            "g8r_post_gate_output_toggles": minimized.witness_cost.g8r_post_gate_output_toggles,
            "g8r_post_weighted_switching_milli": minimized.witness_cost.g8r_post_weighted_switching_milli,
        },
    });
    let summary_path = output_dir.join("summary.json");
    let summary_json =
        serde_json::to_string_pretty(&summary).expect("serialize minimization summary");
    std::fs::write(&summary_path, summary_json.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", summary_path.display(), e))?;

    report(format!(
        "PIR MCMC prefix minimization selected {} provenance actions from a {}-action winning provenance",
        minimized.provenance_action_count, minimized.original_winning_provenance_len
    ));
    report(format!(
        "Retained win fraction: requested={:.6}, actual={:.6}; metrics origin={} winner={} witness={}",
        minimized.requested_retained_win_fraction,
        minimized.actual_retained_win_fraction,
        minimized.origin_metric,
        minimized.winner_metric,
        minimized.witness_metric,
    ));
    report(format!(
        "Wrote minimized witness artifacts to {}",
        output_dir.display()
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CliChainStrategy, ExtensionCostingMode, Objective, PirMcmcCliArgs, PirMcmcMinimizeCliArgs,
        run_pir_mcmc_driver, run_pir_mcmc_minimize_driver,
    };
    use crate::{
        Cost, G8rEvaluationMode, PirMcmcArtifact, PirMcmcBudgetFrontierOptions,
        PirMcmcProvenanceAction, RunOptions, transforms::PirTransformKind,
        write_pir_mcmc_artifact_dir,
    };
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use tempfile::tempdir;
    use xlsynth_g8r::aig_sim::count_toggles::WeightedSwitchingOptions;
    use xlsynth_mcmc::multichain::ChainStrategy;
    use xlsynth_pir::ir::Package;
    use xlsynth_pir::ir_parser;

    fn parse_pkg(ir_text: &str) -> Package {
        let mut parser = ir_parser::Parser::new(ir_text);
        parser.parse_and_validate_package().unwrap()
    }

    fn write_executable_script(
        dir: &std::path::Path,
        name: &str,
        body: &str,
    ) -> std::path::PathBuf {
        let path = dir.join(name);
        fs::write(&path, body).unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).unwrap();
        path
    }

    fn cost_with_pir_nodes(pir_nodes: usize) -> Cost {
        Cost {
            pir_nodes,
            g8r_nodes: pir_nodes,
            g8r_depth: pir_nodes,
            g8r_le_graph_milli: 0,
            g8r_gate_output_toggles: 0,
            g8r_weighted_switching_milli: 0,
            g8r_post_and_nodes: 0,
            g8r_post_depth: 0,
            g8r_post_le_graph_milli: 0,
            g8r_post_gate_output_toggles: 0,
            g8r_post_weighted_switching_milli: 0,
        }
    }

    fn minimize_test_artifact() -> (Package, PirMcmcArtifact) {
        let origin_pkg = parse_pkg(
            r#"package sample

top fn main(x: bits[8] id=1) -> bits[8] {
  dead_a: bits[8] = identity(x, id=2)
  dead_b: bits[8] = identity(x, id=3)
  ret live: bits[8] = identity(x, id=4)
}
"#,
        );
        let step1_pkg = parse_pkg(
            r#"package sample

top fn main(x: bits[8] id=1) -> bits[8] {
  dead_b: bits[8] = identity(x, id=3)
  ret live: bits[8] = identity(x, id=4)
}
"#,
        );
        let step2_pkg = parse_pkg(
            r#"package sample

top fn main(x: bits[8] id=1) -> bits[8] {
  ret live: bits[8] = identity(x, id=4)
}
"#,
        );
        let origin_fn = origin_pkg.get_fn("main").unwrap().clone();
        let step1_fn = step1_pkg.get_fn("main").unwrap().clone();
        let step2_fn = step2_pkg.get_fn("main").unwrap().clone();
        let artifact = PirMcmcArtifact {
            origin_fn,
            origin_cost: cost_with_pir_nodes(5),
            run_options: RunOptions {
                max_iters: 2,
                threads: 1,
                chain_strategy: ChainStrategy::Independent,
                checkpoint_iters: 0,
                progress_iters: 0,
                seed: 1,
                initial_temperature: 1.0,
                objective: Objective::Nodes,
                extension_costing_mode: ExtensionCostingMode::Preserve,
                g8r_evaluation_mode: G8rEvaluationMode::Builtin,
                max_allowed_depth: None,
                max_allowed_area: None,
                weighted_switching_options: WeightedSwitchingOptions::default(),
                enable_formal_oracle: false,
                trajectory_dir: None,
                toggle_stimulus: None,
            },
            raw_winner_fn: step2_fn.clone(),
            raw_winner_cost: cost_with_pir_nodes(3),
            winning_provenance: vec![
                PirMcmcProvenanceAction::AcceptedRewrite {
                    action_index: 1,
                    chain_no: 0,
                    global_iter: 1,
                    transform_kind: PirTransformKind::NotNotCancel,
                    state: step1_fn,
                    cost: cost_with_pir_nodes(4),
                },
                PirMcmcProvenanceAction::AcceptedRewrite {
                    action_index: 2,
                    chain_no: 0,
                    global_iter: 2,
                    transform_kind: PirTransformKind::NegNegCancel,
                    state: step2_fn,
                    cost: cost_with_pir_nodes(3),
                },
            ],
        };
        (origin_pkg, artifact)
    }

    #[test]
    fn reports_when_best_result_remains_infeasible() {
        let input_dir = tempdir().unwrap();
        let output_dir = tempdir().unwrap();
        let input_path = input_dir.path().join("sample.ir");
        fs::write(
            &input_path,
            r#"package sample

top fn main(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(a, b, id=3)
}
"#,
        )
        .unwrap();

        let cli = PirMcmcCliArgs {
            input_path: input_path.display().to_string(),
            iters: 1,
            seed: 1,
            output: Some(output_dir.path().display().to_string()),
            metric: Objective::G8rNodes,
            extension_costing_mode: ExtensionCostingMode::Preserve,
            g8r_postprocess_program: None,
            max_delay: Some(0),
            max_area: None,
            toggle_stimulus: None,
            initial_temperature: 1.0,
            threads: 2,
            checkpoint_iters: 0,
            progress_iters: 0,
            formal_oracle: false,
            switching_beta1: 1.0,
            switching_beta2: 0.0,
            switching_primary_output_load: 1.0,
            chain_strategy: CliChainStrategy::Independent,
        };

        let mut messages = Vec::new();
        run_pir_mcmc_driver(cli, |msg| messages.push(msg)).unwrap();

        assert!(messages.iter().any(|msg| {
            msg.contains("No feasible solution found; best result remains infeasible:")
        }));
    }

    #[test]
    fn supported_run_emits_minimizable_artifact_directory() {
        let input_dir = tempdir().unwrap();
        let output_dir = tempdir().unwrap();
        let input_path = input_dir.path().join("sample.ir");
        fs::write(
            &input_path,
            r#"package sample

top fn main(x: bits[1] id=1) -> bits[1] {
  ret identity.2: bits[1] = identity(x, id=2)
}
"#,
        )
        .unwrap();

        let cli = PirMcmcCliArgs {
            input_path: input_path.display().to_string(),
            iters: 0,
            seed: 1,
            output: Some(output_dir.path().display().to_string()),
            metric: Objective::G8rNodes,
            extension_costing_mode: ExtensionCostingMode::Preserve,
            g8r_postprocess_program: None,
            max_delay: None,
            max_area: None,
            toggle_stimulus: None,
            initial_temperature: 1.0,
            threads: 2,
            checkpoint_iters: 0,
            progress_iters: 0,
            formal_oracle: false,
            switching_beta1: 1.0,
            switching_beta2: 0.0,
            switching_primary_output_load: 1.0,
            chain_strategy: CliChainStrategy::Independent,
        };

        let mut messages = Vec::new();
        run_pir_mcmc_driver(cli, |msg| messages.push(msg)).unwrap();
        assert!(
            output_dir
                .path()
                .join("winning-lineage")
                .join("manifest.json")
                .exists()
        );
        assert!(
            messages
                .iter()
                .any(|msg| msg.contains("Wrote minimizable winning-provenance artifact"))
        );
    }

    #[test]
    fn postprocessed_run_emits_post_artifacts_and_lineage() {
        let input_dir = tempdir().unwrap();
        let output_dir = tempdir().unwrap();
        let hook_dir = tempdir().unwrap();
        let hook = write_executable_script(
            hook_dir.path(),
            "identity.sh",
            "#!/bin/sh\ncp \"$1\" \"$3\"\n",
        );
        let input_path = input_dir.path().join("sample.ir");
        fs::write(
            &input_path,
            r#"package sample

top fn main(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(a, b, id=3)
}
"#,
        )
        .unwrap();

        let cli = PirMcmcCliArgs {
            input_path: input_path.display().to_string(),
            iters: 0,
            seed: 1,
            output: Some(output_dir.path().display().to_string()),
            metric: Objective::G8rPostAndNodes,
            extension_costing_mode: ExtensionCostingMode::Preserve,
            g8r_postprocess_program: Some(hook.display().to_string()),
            max_delay: None,
            max_area: None,
            toggle_stimulus: None,
            initial_temperature: 1.0,
            threads: 1,
            checkpoint_iters: 0,
            progress_iters: 0,
            formal_oracle: false,
            switching_beta1: 1.0,
            switching_beta2: 0.0,
            switching_primary_output_load: 1.0,
            chain_strategy: CliChainStrategy::Independent,
        };

        run_pir_mcmc_driver(cli, |_| {}).unwrap();
        assert!(output_dir.path().join("orig.post.aig").exists());
        assert!(output_dir.path().join("orig.post.stats.json").exists());
        assert!(output_dir.path().join("best.post.aig").exists());
        assert!(output_dir.path().join("best.post.stats.json").exists());
        assert!(
            output_dir
                .path()
                .join("winning-lineage/manifest.json")
                .exists()
        );
    }

    #[test]
    fn unsupported_run_reports_artifact_skip() {
        let input_dir = tempdir().unwrap();
        let output_dir = tempdir().unwrap();
        let input_path = input_dir.path().join("sample.ir");
        fs::write(
            &input_path,
            r#"package sample

top fn main(x: bits[1] id=1) -> bits[1] {
  ret identity.2: bits[1] = identity(x, id=2)
}
"#,
        )
        .unwrap();

        let cli = PirMcmcCliArgs {
            input_path: input_path.display().to_string(),
            iters: 0,
            seed: 1,
            output: Some(output_dir.path().display().to_string()),
            metric: Objective::G8rNodes,
            extension_costing_mode: ExtensionCostingMode::Preserve,
            g8r_postprocess_program: None,
            max_delay: Some(1),
            max_area: None,
            toggle_stimulus: None,
            initial_temperature: 1.0,
            threads: 2,
            checkpoint_iters: 0,
            progress_iters: 0,
            formal_oracle: false,
            switching_beta1: 1.0,
            switching_beta2: 0.0,
            switching_primary_output_load: 1.0,
            chain_strategy: CliChainStrategy::Independent,
        };

        let mut messages = Vec::new();
        run_pir_mcmc_driver(cli, |msg| messages.push(msg)).unwrap();
        assert!(
            messages
                .iter()
                .any(|msg| msg.contains("No minimizable winning-provenance artifact emitted"))
        );
        assert!(!output_dir.path().join("winning-lineage").exists());
    }

    #[test]
    fn minimize_driver_writes_witness_outputs_and_summary() {
        let run_dir = tempdir().unwrap();
        let output_dir = tempdir().unwrap();
        let (pkg, artifact) = minimize_test_artifact();
        write_pir_mcmc_artifact_dir(&artifact, &pkg, run_dir.path()).unwrap();

        let cli = PirMcmcMinimizeCliArgs {
            run_dir: run_dir.path().display().to_string(),
            retained_win_fraction: Some(0.5),
            budget_step: None,
            max_actions: None,
            rollouts_per_budget: None,
            seed: None,
            witness_kind_boost: PirMcmcBudgetFrontierOptions::DEFAULT_WITNESS_KIND_BOOST,
            proposal_attempts_per_rewrite:
                PirMcmcBudgetFrontierOptions::DEFAULT_PROPOSAL_ATTEMPTS_PER_REWRITE,
            allow_artifact_postprocess_program: false,
            output: output_dir.path().display().to_string(),
        };
        let mut messages = Vec::new();
        run_pir_mcmc_minimize_driver(cli, |msg| messages.push(msg)).unwrap();

        assert!(output_dir.path().join("witness.ir").exists());
        assert!(output_dir.path().join("witness.opt.ir").exists());
        assert!(output_dir.path().join("witness.g8r").exists());
        assert!(output_dir.path().join("witness.stats.json").exists());
        let summary_text = fs::read_to_string(output_dir.path().join("summary.json")).unwrap();
        let summary: serde_json::Value = serde_json::from_str(&summary_text).unwrap();
        assert_eq!(summary["provenance_action_count"], 1);
        assert_eq!(summary["witness_metric"], 4);
        assert!(
            messages
                .iter()
                .any(|msg| msg.contains("selected 1 provenance actions from a 2-action"))
        );
    }

    #[test]
    fn minimize_driver_replays_persisted_postprocessor() {
        let run_dir = tempdir().unwrap();
        let output_dir = tempdir().unwrap();
        let hook_dir = tempdir().unwrap();
        let marker = hook_dir.path().join("invoked");
        let hook = write_executable_script(
            hook_dir.path(),
            "identity.sh",
            &format!(
                "#!/bin/sh\nprintf invoked > \"{}\"\ncp \"$1\" \"$3\"\n",
                marker.display()
            ),
        );
        let (pkg, mut artifact) = minimize_test_artifact();
        artifact.run_options.g8r_evaluation_mode = G8rEvaluationMode::ExternalPostprocess {
            program: hook.display().to_string(),
        };
        write_pir_mcmc_artifact_dir(&artifact, &pkg, run_dir.path()).unwrap();

        let cli = PirMcmcMinimizeCliArgs {
            run_dir: run_dir.path().display().to_string(),
            retained_win_fraction: Some(0.5),
            budget_step: None,
            max_actions: None,
            rollouts_per_budget: None,
            seed: None,
            witness_kind_boost: PirMcmcBudgetFrontierOptions::DEFAULT_WITNESS_KIND_BOOST,
            proposal_attempts_per_rewrite:
                PirMcmcBudgetFrontierOptions::DEFAULT_PROPOSAL_ATTEMPTS_PER_REWRITE,
            allow_artifact_postprocess_program: false,
            output: output_dir.path().display().to_string(),
        };
        let err = run_pir_mcmc_minimize_driver(cli.clone(), |_| {}).unwrap_err();
        assert!(
            err.to_string()
                .contains("--allow-artifact-postprocess-program"),
            "unexpected error: {err}"
        );
        assert!(!marker.exists());

        run_pir_mcmc_minimize_driver(
            PirMcmcMinimizeCliArgs {
                allow_artifact_postprocess_program: true,
                ..cli
            },
            |_| {},
        )
        .unwrap();

        assert!(output_dir.path().join("witness.post.aig").exists());
        assert!(output_dir.path().join("witness.post.stats.json").exists());
        assert!(marker.exists());
    }

    #[test]
    fn minimize_driver_rejects_invalid_fraction_and_missing_artifact() {
        let missing_run_dir = tempdir().unwrap();
        let output_dir = tempdir().unwrap();
        let err = run_pir_mcmc_minimize_driver(
            PirMcmcMinimizeCliArgs {
                run_dir: missing_run_dir.path().display().to_string(),
                retained_win_fraction: Some(0.5),
                budget_step: None,
                max_actions: None,
                rollouts_per_budget: None,
                seed: None,
                witness_kind_boost: PirMcmcBudgetFrontierOptions::DEFAULT_WITNESS_KIND_BOOST,
                proposal_attempts_per_rewrite:
                    PirMcmcBudgetFrontierOptions::DEFAULT_PROPOSAL_ATTEMPTS_PER_REWRITE,
                allow_artifact_postprocess_program: false,
                output: output_dir.path().display().to_string(),
            },
            |_| {},
        )
        .unwrap_err();
        assert!(err.to_string().contains("manifest.json"));

        let run_dir = tempdir().unwrap();
        let (pkg, artifact) = minimize_test_artifact();
        write_pir_mcmc_artifact_dir(&artifact, &pkg, run_dir.path()).unwrap();
        let err = run_pir_mcmc_minimize_driver(
            PirMcmcMinimizeCliArgs {
                run_dir: run_dir.path().display().to_string(),
                retained_win_fraction: Some(1.1),
                budget_step: None,
                max_actions: None,
                rollouts_per_budget: None,
                seed: None,
                witness_kind_boost: PirMcmcBudgetFrontierOptions::DEFAULT_WITNESS_KIND_BOOST,
                proposal_attempts_per_rewrite:
                    PirMcmcBudgetFrontierOptions::DEFAULT_PROPOSAL_ATTEMPTS_PER_REWRITE,
                allow_artifact_postprocess_program: false,
                output: output_dir.path().display().to_string(),
            },
            |_| {},
        )
        .unwrap_err();
        assert!(err.to_string().contains("retained_win_fraction"));
    }

    #[test]
    fn minimize_driver_frontier_writes_budget_subtrees_and_summary() {
        let run_dir = tempdir().unwrap();
        let output_dir = tempdir().unwrap();
        let (pkg, artifact) = minimize_test_artifact();
        write_pir_mcmc_artifact_dir(&artifact, &pkg, run_dir.path()).unwrap();

        let cli = PirMcmcMinimizeCliArgs {
            run_dir: run_dir.path().display().to_string(),
            retained_win_fraction: None,
            budget_step: Some(1),
            max_actions: Some(2),
            rollouts_per_budget: Some(1),
            seed: Some(1),
            witness_kind_boost: 4.0,
            proposal_attempts_per_rewrite: 1,
            allow_artifact_postprocess_program: false,
            output: output_dir.path().display().to_string(),
        };
        let mut messages = Vec::new();
        run_pir_mcmc_minimize_driver(cli, |msg| messages.push(msg)).unwrap();

        assert!(output_dir.path().join("budget-0001/witness.ir").exists());
        assert!(output_dir.path().join("budget-0002/witness.ir").exists());
        let summary_text = fs::read_to_string(output_dir.path().join("summary.json")).unwrap();
        let summary: serde_json::Value = serde_json::from_str(&summary_text).unwrap();
        assert_eq!(summary["mode"], "budget_frontier");
        assert_eq!(summary["points"].as_array().unwrap().len(), 2);
        assert!(
            messages
                .iter()
                .any(|msg| msg.contains("guided frontier searched 2 budgets"))
        );
    }

    #[test]
    fn minimize_driver_rejects_mixed_or_missing_modes() {
        let run_dir = tempdir().unwrap();
        let output_dir = tempdir().unwrap();
        let (pkg, artifact) = minimize_test_artifact();
        write_pir_mcmc_artifact_dir(&artifact, &pkg, run_dir.path()).unwrap();

        let err = run_pir_mcmc_minimize_driver(
            PirMcmcMinimizeCliArgs {
                run_dir: run_dir.path().display().to_string(),
                retained_win_fraction: Some(0.5),
                budget_step: Some(1),
                max_actions: Some(2),
                rollouts_per_budget: Some(1),
                seed: None,
                witness_kind_boost: 4.0,
                proposal_attempts_per_rewrite: 1,
                allow_artifact_postprocess_program: false,
                output: output_dir.path().display().to_string(),
            },
            |_| {},
        )
        .unwrap_err();
        assert!(err.to_string().contains("exactly one minimization mode"));

        let err = run_pir_mcmc_minimize_driver(
            PirMcmcMinimizeCliArgs {
                run_dir: run_dir.path().display().to_string(),
                retained_win_fraction: None,
                budget_step: None,
                max_actions: None,
                rollouts_per_budget: None,
                seed: None,
                witness_kind_boost: 4.0,
                proposal_attempts_per_rewrite: 1,
                allow_artifact_postprocess_program: false,
                output: output_dir.path().display().to_string(),
            },
            |_| {},
        )
        .unwrap_err();
        assert!(err.to_string().contains("exactly one minimization mode"));
    }
}
