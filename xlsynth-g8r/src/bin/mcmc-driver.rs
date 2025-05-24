// SPDX-License-Identifier: Apache-2.0

//! XLS‑IR → `gate::GateFn` → single‑thread MCMC

use anyhow::Result;
use clap::Parser;
use num_cpus;
use serde_json;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tempfile::Builder;
use xlsynth_g8r::gate::GateFn;
use xlsynth_g8r::get_summary_stats::SummaryStats;

use xlsynth_g8r::get_summary_stats;
use xlsynth_g8r::mcmc_logic::{cost, load_start, mcmc, Best, Objective};

#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]
struct CliArgs {
    /// Input file (.ir) or sample (sample://name)
    input_path: String,

    /// Number of MCMC iterations to perform.
    #[clap(short = 'n', long, value_parser)]
    iters: u64,

    /// Random seed
    #[clap(short = 'S', long, value_parser, default_value_t = 1)]
    seed: u64,

    /// Output file or directory. If a directory, 'best.g8r' is saved there.
    /// If not specified, output goes to a new temporary directory.
    #[clap(short, long, value_parser)]
    output: Option<String>,

    /// List of transform kinds to disable (e.g., SwapOp, DblNeg).
    #[clap(long, value_delimiter = ',', num_args = 0.., use_value_delimiter = true)]
    disabled_transforms: Option<Vec<String>>,

    /// Print every MCMC iteration and action (for debugging hangs)
    #[clap(long)]
    verbose: bool,

    /// Metric to optimize: nodes, depth, or product (nodes*depth)
    #[clap(long, value_enum, default_value_t = Objective::Product)]
    metric: Objective,

    /// Verify equivalence for every accepted edit even if the transform is
    /// marked always_equivalent, and abort if any equivalence failure is
    /// detected.
    #[clap(long)]
    paranoid: bool,

    /// Iterations between checkpoints written to disk (0 to disable).
    #[clap(long, value_parser, default_value_t = 5000)]
    checkpoint_iters: u64,

    /// Number of parallel MCMC chains to run.
    #[clap(long, value_parser, default_value_t = num_cpus::get() as u64)]
    threads: u64,
}

fn run_chain(
    cfg: Arc<CliArgs>,
    seed: u64,
    start: GateFn,
    running: Arc<AtomicBool>,
    best: Arc<Best>,
) {
    mcmc(
        start,
        cfg.iters,
        seed,
        running,
        cfg.disabled_transforms.clone().unwrap_or_default(),
        cfg.verbose,
        cfg.metric,
        None,
        cfg.paranoid,
        cfg.checkpoint_iters,
        Some(best),
    );
}

fn main() -> Result<()> {
    let _ = env_logger::try_init();

    let cli = CliArgs::parse();
    println!("MCMC Driver started with args: {:?}", cli);

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        println!("\nCtrl+C received, attempting to shut down gracefully...");
    })
    .expect("Error setting Ctrl-C handler");

    let start_gfn = match load_start(&cli.input_path) {
        Ok(gfn) => gfn,
        Err(e) => {
            eprintln!("Error loading start program: {:?}", e);
            return Err(e);
        }
    };

    let initial_stats = get_summary_stats::get_summary_stats(&start_gfn);
    println!(
        "Successfully loaded start program. Initial stats: nodes={}, depth={}",
        initial_stats.live_nodes, initial_stats.deepest_path
    );

    // Determine output paths early so that we can periodically dump during MCMC.
    let output_g8r_filename = "best.g8r";
    let output_stats_filename = "best.stats.json";

    let (output_g8r_path, output_stats_path, _temp_dir_holder): (
        PathBuf,
        PathBuf,
        Option<tempfile::TempDir>,
    ) = match &cli.output {
        Some(path_str) => {
            let p = PathBuf::from(path_str);
            if p.is_dir() || path_str.ends_with('/') || path_str.ends_with('\\') {
                fs::create_dir_all(&p)?;
                (
                    p.join(output_g8r_filename),
                    p.join(output_stats_filename),
                    None,
                )
            } else {
                if let Some(parent) = p.parent() {
                    if !parent.exists() {
                        fs::create_dir_all(parent)?;
                    }
                }
                let stats_p = p.with_file_name(
                    p.file_stem()
                        .unwrap_or_default()
                        .to_str()
                        .unwrap_or("best")
                        .to_owned()
                        + ".stats.json",
                );
                (p, stats_p, None)
            }
        }
        None => {
            let temp_dir = Builder::new().prefix("mcmc_output_").tempdir()?;
            println!(
                "No output path specified, using temp dir: {}",
                temp_dir.path().display()
            );
            let base_path = temp_dir.path();
            (
                base_path.join(output_g8r_filename),
                base_path.join(output_stats_filename),
                Some(temp_dir),
            )
        }
    };

    let init_cost = cost(&start_gfn);
    let init_metric = match cli.metric {
        Objective::Nodes => init_cost.nodes as usize,
        Objective::Depth => init_cost.depth as usize,
        Objective::Product => init_cost.nodes * init_cost.depth,
    };

    let best = Arc::new(Best::new(init_metric, start_gfn.clone()));

    let mut handles = Vec::new();
    for i in 0..cli.threads {
        let cfg = cli.clone();
        let running_cl = running.clone();
        let best_cl = best.clone();
        let start_cl = start_gfn.clone();
        let seed_i = cli.seed ^ i as u64;
        handles.push(std::thread::spawn(move || {
            run_chain(Arc::new(cfg), seed_i, start_cl, running_cl, best_cl);
        }));
    }

    for h in handles {
        let _ = h.join();
    }

    if !running.load(Ordering::SeqCst) {
        println!("MCMC process was interrupted.");
    }

    let best_gfn = best.get();
    let final_summary_stats: SummaryStats = get_summary_stats::get_summary_stats(&best_gfn);
    println!(
        "MCMC finished. Final best GateFn stats: nodes={}, depth={}",
        final_summary_stats.live_nodes, final_summary_stats.deepest_path
    );

    println!(
        "Dumping best GateFn as text to: {}",
        output_g8r_path.display()
    );
    let mut f_g8r = fs::File::create(&output_g8r_path)?;
    f_g8r.write_all(best_gfn.to_string().as_bytes())?;
    println!("Successfully wrote output to {}", output_g8r_path.display());

    println!(
        "Dumping best GateFn stats as JSON to: {}",
        output_stats_path.display()
    );
    let mut f_stats = fs::File::create(&output_stats_path)?;
    let stats_json = serde_json::to_string_pretty(&final_summary_stats)?;
    f_stats.write_all(stats_json.as_bytes())?;
    println!(
        "Successfully wrote stats to {}",
        output_stats_path.display()
    );

    Ok(())
}
