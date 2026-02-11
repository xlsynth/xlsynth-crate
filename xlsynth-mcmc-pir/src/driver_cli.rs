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
use xlsynth_g8r::aig::get_summary_stats;
use xlsynth_g8r::aig::get_summary_stats::SummaryStats;
use xlsynth_g8r::gatify::ir2gate;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_mcmc::Best;
use xlsynth_mcmc::multichain::ChainStrategy;
use xlsynth_pir::desugar_extensions;
use xlsynth_pir::ir::{Package, PackageMember};
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_utils::compact_and_toposort_in_place;

use crate::{
    CheckpointKind, CheckpointMsg, Objective, RunOptions, cost, cost_with_toggle_stimulus,
    lower_toggle_stimulus_for_fn, parse_irvals_tuple_file, run_pir_mcmc_with_shared_best,
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
    pub toggle_stimulus: Option<String>,
    pub initial_temperature: f64,
    pub threads: u64,
    pub checkpoint_iters: u64,
    pub progress_iters: u64,
    pub formal_oracle: bool,
    chain_strategy: CliChainStrategy,
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
}

pub fn parse_pir_mcmc_args(matches: &ArgMatches) -> PirMcmcCliArgs {
    PirMcmcCliArgs {
        input_path: matches.get_one::<String>("input_path").unwrap().to_string(),
        iters: *matches.get_one::<u64>("iters").unwrap(),
        seed: *matches.get_one::<u64>("seed").unwrap(),
        output: matches.get_one::<String>("output").cloned(),
        metric: *matches.get_one::<Objective>("metric").unwrap(),
        toggle_stimulus: matches.get_one::<String>("toggle_stimulus").cloned(),
        initial_temperature: *matches.get_one::<f64>("initial_temperature").unwrap(),
        threads: matches
            .get_one::<u64>("threads")
            .copied()
            .unwrap_or(num_cpus::get() as u64),
        checkpoint_iters: *matches.get_one::<u64>("checkpoint_iters").unwrap(),
        progress_iters: *matches.get_one::<u64>("progress_iters").unwrap(),
        formal_oracle: *matches.get_one::<bool>("formal_oracle").unwrap(),
        chain_strategy: *matches
            .get_one::<CliChainStrategy>("chain_strategy")
            .unwrap(),
    }
}

fn optimize_ir_text(ir_text: &str, top: &str) -> Result<String> {
    // The CLI accepts PIR text which may contain extension ops (e.g.
    // ext_carry_out), but upstream XLS does not. Desugar extensions before
    // invoking upstream.
    let lowered_ir_text = {
        let mut p = xlsynth_pir::ir_parser::Parser::new(ir_text);
        let mut pir_pkg = p
            .parse_package()
            .map_err(|e| anyhow::anyhow!("PIR parse_package failed: {:?}", e))?;
        desugar_extensions::desugar_extensions_in_package(&mut pir_pkg)
            .map_err(|e| anyhow::anyhow!("desugar_extensions_in_package failed: {}", e))?;
        pir_pkg.to_string()
    };

    let ir_pkg = xlsynth::IrPackage::parse_ir(&lowered_ir_text, None)
        .map_err(|e| anyhow::anyhow!("IrPackage::parse_ir failed: {:?}", e))?;
    let optimized_ir_pkg = xlsynth::optimize_ir(&ir_pkg, top)
        .map_err(|e| anyhow::anyhow!("optimize_ir failed: {:?}", e))?;
    Ok(optimized_ir_pkg.to_string())
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

fn gatify_ir_text_to_g8r_text_and_stats(ir_text: &str) -> Result<(String, SummaryStats)> {
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
        mul_adder_mapping: None,
        range_info: None,
        enable_rewrite_carry_out: false,
        enable_rewrite_prio_encode: false,
    };
    let gatify_output = ir2gate::gatify(top_fn, gatify_options)
        .map_err(|e| anyhow::anyhow!("ir2gate::gatify failed: {}", e))?;
    let gate_fn = gatify_output.gate_fn;
    let stats = get_summary_stats::get_summary_stats(&gate_fn);
    Ok((gate_fn.to_string(), stats))
}

pub fn run_pir_mcmc_driver<F>(cli: PirMcmcCliArgs, mut report: F) -> Result<()>
where
    F: FnMut(String),
{
    report(format!("PIR MCMC Driver started with args: {:?}", cli));

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
            "--toggle-stimulus is only valid with --metric={}",
            Objective::G8rNodesTimesDepthTimesToggles.value_name()
        ));
    }
    let prepared_toggle_stimulus = toggle_stimulus_values
        .as_ref()
        .map(|samples| lower_toggle_stimulus_for_fn(samples, &top_fn))
        .transpose()?;

    let initial_cost = if cli.metric.needs_toggle_stimulus() {
        cost_with_toggle_stimulus(&top_fn, cli.metric, prepared_toggle_stimulus.as_deref())?
    } else {
        cost(&top_fn, cli.metric)?
    };
    if cli.metric.needs_toggle_stimulus() {
        report(format!(
            "Successfully loaded top function. Initial stats: pir_nodes={}, g8r_nodes={}, g8r_depth={}, g8r_gate_output_toggles={}",
            initial_cost.pir_nodes,
            initial_cost.g8r_nodes,
            initial_cost.g8r_depth,
            initial_cost.g8r_gate_output_toggles
        ));
    } else {
        report(format!(
            "Successfully loaded top function. Initial stats: pir_nodes={}, g8r_nodes={}, g8r_depth={}",
            initial_cost.pir_nodes, initial_cost.g8r_nodes, initial_cost.g8r_depth
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

    let opts = RunOptions {
        max_iters: cli.iters,
        threads: cli.threads,
        chain_strategy: cli.chain_strategy.into(),
        checkpoint_iters: cli.checkpoint_iters,
        progress_iters: cli.progress_iters,
        seed: cli.seed,
        initial_temperature: cli.initial_temperature,
        objective: cli.metric,
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
        let metric_u128 = cli.metric.metric(&initial_cost);
        Some(Arc::new(Best::new(metric_u128, top_fn.clone())))
    } else {
        None
    };

    // Emit original artifacts.
    let orig_ir_text = emit_pkg_text_toposorted(&pkg)?;
    let orig_ir_path = output_dir.join("orig.ir");
    std::fs::write(&orig_ir_path, orig_ir_text.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", orig_ir_path.display(), e))?;

    let orig_top_name = top_fn.name.clone();
    let orig_opt_ir_text = optimize_ir_text(&orig_ir_text, &orig_top_name)?;
    let orig_opt_ir_path = output_dir.join("orig.opt.ir");
    std::fs::write(&orig_opt_ir_path, orig_opt_ir_text.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", orig_opt_ir_path.display(), e))?;

    let (orig_g8r_text, orig_stats) = gatify_ir_text_to_g8r_text_and_stats(&orig_opt_ir_text)?;
    let orig_g8r_path = output_dir.join("orig.g8r");
    std::fs::write(&orig_g8r_path, orig_g8r_text.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", orig_g8r_path.display(), e))?;
    let orig_stats_path = output_dir.join("orig.stats.json");
    let orig_stats_json =
        serde_json::to_string_pretty(&orig_stats).expect("serialize SummaryStats");
    std::fs::write(&orig_stats_path, orig_stats_json.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", orig_stats_path.display(), e))?;

    let pkg_template = Arc::new(pkg.clone());
    let output_dir_for_thread = output_dir.clone();
    let orig_top_name_for_thread = orig_top_name.clone();

    let writer_handle = if let (Some(best), Some(rx)) = (shared_best.clone(), checkpoint_rx) {
        Some(std::thread::spawn(move || {
            // Track last written metric to reduce redundant writes when multiple
            // chains hit the same checkpoint boundary.
            let mut last_written: Option<u128> = None;
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
                let cur_metric = best.cost.load(std::sync::atomic::Ordering::SeqCst);
                if last_written == Some(cur_metric) {
                    continue;
                }
                last_written = Some(cur_metric);

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

                let best_opt_ir_text =
                    match optimize_ir_text(&best_ir_text, &orig_top_name_for_thread) {
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

                let (best_g8r_text, best_stats) =
                    match gatify_ir_text_to_g8r_text_and_stats(&best_opt_ir_text) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                let best_g8r_path = output_dir_for_thread.join("best.g8r");
                let _ = std::fs::write(&best_g8r_path, best_g8r_text.as_bytes());

                let best_stats_path = output_dir_for_thread.join("best.stats.json");
                if let Ok(stats_json) = serde_json::to_string_pretty(&best_stats) {
                    let _ = std::fs::write(&best_stats_path, stats_json.as_bytes());

                    let best_stats_snapshot_path = output_dir_for_thread.join(format!(
                        "best.w{:06}.c{:03}-i{:06}.stats.json",
                        write_index, snapshot_msg.chain_no, snapshot_msg.global_iter
                    ));
                    let _ = std::fs::write(&best_stats_snapshot_path, stats_json.as_bytes());
                }
            }
        }))
    } else {
        None
    };

    let result =
        run_pir_mcmc_with_shared_best(top_fn, opts, shared_best.clone(), checkpoint_tx, None)?;

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
    let best_opt_ir_text = optimize_ir_text(&best_ir_text, &best_top_name)?;
    let best_opt_ir_path = output_dir.join("best.opt.ir");
    std::fs::write(&best_opt_ir_path, best_opt_ir_text.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", best_opt_ir_path.display(), e))?;

    let (best_g8r_text, best_stats) = gatify_ir_text_to_g8r_text_and_stats(&best_opt_ir_text)?;
    let best_g8r_path = output_dir.join("best.g8r");
    std::fs::write(&best_g8r_path, best_g8r_text.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", best_g8r_path.display(), e))?;
    let best_stats_path = output_dir.join("best.stats.json");
    let best_stats_json =
        serde_json::to_string_pretty(&best_stats).expect("serialize SummaryStats");
    std::fs::write(&best_stats_path, best_stats_json.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {:?}", best_stats_path.display(), e))?;

    Ok(())
}
