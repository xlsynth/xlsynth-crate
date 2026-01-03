// SPDX-License-Identifier: Apache-2.0

//! PIR MCMC driver.
//!
//! This binary parses an XLS IR file into PIR, runs a single-chain MCMC
//! optimization over the top function using `xlsynth-mcmc-pir`, and writes the
//! resulting optimized IR back out as text.

use anyhow::Result;
use clap::Parser;
use clap::ValueEnum;
use num_cpus;
use std::io::Write as IoWrite;
use std::path::PathBuf;
use tempfile::Builder;

use std::sync::Arc;
use std::sync::mpsc;
use xlsynth_g8r::aig::get_summary_stats;
use xlsynth_g8r::aig::get_summary_stats::SummaryStats;
use xlsynth_g8r::aig_serdes::ir2gate;
use xlsynth_g8r::aig_serdes::ir2gate::GatifyOptions;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_mcmc::Best;
use xlsynth_mcmc::multichain::ChainStrategy;
use xlsynth_mcmc_pir::{
    CheckpointKind, CheckpointMsg, Objective, RunOptions, cost, run_pir_mcmc_with_shared_best,
};
use xlsynth_pir::ir::{Package, PackageMember};
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_utils::compact_and_toposort_in_place;

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

#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]
struct CliArgs {
    /// Input IR file (.ir) to optimize.
    input_path: String,

    /// Number of MCMC iterations to perform.
    #[clap(short = 'n', long, value_parser)]
    iters: u64,

    /// Random seed.
    #[clap(short = 'S', long, value_parser, default_value_t = 1)]
    seed: u64,

    /// Output file or directory. If a directory, 'best.ir' is saved there.
    /// If not specified, output goes to a new temporary directory.
    #[clap(short, long, value_parser)]
    output: Option<String>,

    /// Metric to optimize.
    #[clap(long, value_enum, default_value_t = Objective::Nodes)]
    metric: Objective,

    /// Initial temperature for MCMC (default: 5.0).
    #[clap(long, value_parser, default_value_t = 5.0)]
    initial_temperature: f64,

    /// Number of parallel MCMC chains to run.
    #[clap(long, value_parser, default_value_t = num_cpus::get() as u64)]
    threads: u64,

    /// Strategy for running multiple MCMC chains.
    #[clap(long, value_enum, default_value_t = CliChainStrategy::Independent)]
    chain_strategy: CliChainStrategy,

    /// Iterations per synchronization segment in explore/exploit mode.
    #[clap(long, value_parser, default_value_t = 5000)]
    checkpoint_iters: u64,

    /// Progress logging interval in iterations (0 disables progress logs).
    #[clap(long, value_parser, default_value_t = 1000)]
    progress_iters: u64,

    /// Enable a formal equivalence oracle (in addition to the interpreter-based
    /// oracle) for transforms that are not marked always-equivalent.
    ///
    /// Defaults to `true`. Disable with `--formal-oracle=false` if you want max
    /// throughput (but note that non-always-equivalent transforms are then
    /// disabled for safety).
    #[clap(long, default_value_t = true)]
    formal_oracle: bool,
}

fn optimize_ir_text(ir_text: &str, top: &str) -> Result<String> {
    let ir_pkg = xlsynth::IrPackage::parse_ir(ir_text, None)
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
    };
    let gatify_output = ir2gate::gatify(top_fn, gatify_options)
        .map_err(|e| anyhow::anyhow!("ir2gate::gatify failed: {}", e))?;
    let gate_fn = gatify_output.gate_fn;
    let stats = get_summary_stats::get_summary_stats(&gate_fn);
    Ok((gate_fn.to_string(), stats))
}

fn main() -> Result<()> {
    let _ = env_logger::try_init();

    let cli = CliArgs::parse();
    println!("PIR MCMC Driver started with args: {:?}", cli);

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

    let initial_cost = cost(&top_fn, cli.metric)?;
    println!(
        "Successfully loaded top function. Initial stats: pir_nodes={}, g8r_nodes={}, g8r_depth={}",
        initial_cost.pir_nodes, initial_cost.g8r_nodes, initial_cost.g8r_depth
    );

    // Determine output paths.
    let output_ir_filename = "best.ir";

    let (output_ir_path, output_dir, _temp_dir_holder): (
        PathBuf,
        PathBuf,
        Option<tempfile::TempDir>,
    ) = match &cli.output {
        Some(path_str) => {
            let p = PathBuf::from(path_str);
            let dir = if p.is_dir() || path_str.ends_with('/') || path_str.ends_with('\\') {
                std::fs::create_dir_all(&p)?;
                p.clone()
            } else {
                if let Some(parent) = p.parent() {
                    if !parent.exists() {
                        std::fs::create_dir_all(parent)?;
                    }
                }
                p.parent().unwrap_or(&p).to_path_buf()
            };
            let ir = if p.is_dir() || path_str.ends_with('/') || path_str.ends_with('\\') {
                p.join(output_ir_filename)
            } else {
                p.clone()
            };
            (ir, dir, None)
        }
        None => {
            let temp_dir = Builder::new().prefix("pir_mcmc_output_").tempdir()?;
            println!(
                "No output path specified, using temp dir: {}",
                temp_dir.path().display()
            );
            let base_path = temp_dir.path();
            (
                base_path.join(output_ir_filename),
                base_path.to_path_buf(),
                Some(temp_dir),
            )
        }
    };

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
        let metric_u64 = cli.metric.metric(&initial_cost);
        let metric_usize = usize::try_from(metric_u64).unwrap_or(usize::MAX);
        Some(Arc::new(Best::new(metric_usize, top_fn.clone())))
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
            let mut last_written: Option<usize> = None;
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
                let best_opt_ir_snapshot_path = output_dir_for_thread.join(format!(
                    "best.c{:03}-i{:06}.opt.ir",
                    snapshot_msg.chain_no, snapshot_msg.global_iter
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
                        "best.c{:03}-i{:06}.stats.json",
                        snapshot_msg.chain_no, snapshot_msg.global_iter
                    ));
                    let _ = std::fs::write(&best_stats_snapshot_path, stats_json.as_bytes());
                }
            }
        }))
    } else {
        None
    };

    let result = run_pir_mcmc_with_shared_best(top_fn, opts, shared_best.clone(), checkpoint_tx)?;

    // Stop checkpoint writer cleanly before final artifact emission.
    if let Some(h) = writer_handle {
        let _ = h.join();
    }

    match cli.metric {
        Objective::Nodes => {
            println!(
                "PIR MCMC finished. Best stats: pir_nodes={}",
                result.best_cost.pir_nodes
            );
        }
        Objective::G8rNodes => {
            println!(
                "PIR MCMC finished. Best stats: g8r_nodes={}",
                result.best_cost.g8r_nodes
            );
        }
        Objective::G8rNodesTimesDepth => {
            println!(
                "PIR MCMC finished. Best stats: g8r_nodes={}, g8r_depth={}, product={}",
                result.best_cost.g8r_nodes,
                result.best_cost.g8r_depth,
                (result.best_cost.g8r_nodes as u64)
                    .saturating_mul(result.best_cost.g8r_depth as u64),
            );
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

    println!(
        "Writing optimized PIR package to {}",
        output_ir_path.display()
    );
    let mut f_ir = std::fs::File::create(&output_ir_path)?;
    let pkg_text_out = emit_pkg_text_toposorted(&pkg)?;
    f_ir.write_all(pkg_text_out.as_bytes())?;
    println!(
        "Successfully wrote optimized PIR to {}",
        output_ir_path.display()
    );

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
