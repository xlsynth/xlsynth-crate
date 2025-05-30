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
use xlsynth_g8r::mcmc_logic::{cost, load_start, mcmc, Best, McmcOptions, Objective};

use std::time::{Duration, Instant};

#[cfg(target_os = "linux")]
use libc;

use atty::Stream;
use colored::*;

/// Returns the current resident-set size in MiB (Linux-only). On other
/// platforms this always returns `None`.
#[cfg(target_os = "linux")]
fn rss_megabytes() -> Option<u64> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    if let Ok(f) = File::open("/proc/self/statm") {
        let mut line = String::new();
        if BufReader::new(f).read_line(&mut line).is_ok() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(resident_pages) = parts[1].parse::<u64>() {
                    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as u64 };
                    return Some(resident_pages * page_size / (1024 * 1024));
                }
            }
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn rss_megabytes() -> Option<u64> {
    None
}

#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]
struct CliArgs {
    /// Input file (.ir or .g8r) or sample (sample://name)
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

    /// Iterations between progress updates written to progress.jsonl (0 to
    /// disable).
    #[clap(long, value_parser, default_value_t = 1000)]
    progress_iters: u64,

    /// Number of parallel MCMC chains to run.
    #[clap(long, value_parser, default_value_t = num_cpus::get() as u64)]
    threads: u64,

    /// Initial temperature for MCMC (default: 5.0)
    #[clap(long, value_parser, default_value_t = 5.0)]
    initial_temperature: f64,
}

fn run_chain(
    cfg: Arc<CliArgs>,
    seed: u64,
    start: GateFn,
    running: Arc<AtomicBool>,
    best: Arc<Best>,
    chain_no: usize,
    periodic_dump_dir: Option<PathBuf>,
    progress_interval: u64,
) -> Result<(), anyhow::Error> {
    let options = McmcOptions {
        sat_reset_interval: 20000, // or make this configurable
        initial_temperature: cfg.initial_temperature,
    };
    mcmc(
        start,
        cfg.iters,
        seed,
        running,
        cfg.disabled_transforms.clone().unwrap_or_default(),
        cfg.verbose || cfg.paranoid,
        cfg.metric,
        periodic_dump_dir,
        cfg.paranoid,
        cfg.checkpoint_iters,
        progress_interval,
        Some(best),
        Some(chain_no),
        options,
    )?;
    Ok(())
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

    let (output_g8r_path, output_stats_path, output_dir, _temp_dir_holder): (
        PathBuf,
        PathBuf,
        PathBuf,
        Option<tempfile::TempDir>,
    ) = match &cli.output {
        Some(path_str) => {
            let p = PathBuf::from(path_str);
            let dir = if p.is_dir() || path_str.ends_with('/') || path_str.ends_with('\\') {
                fs::create_dir_all(&p)?;
                p.clone()
            } else {
                if let Some(parent) = p.parent() {
                    if !parent.exists() {
                        fs::create_dir_all(parent)?;
                    }
                }
                p.parent().unwrap_or(&p).to_path_buf()
            };
            let g8r = if p.is_dir() || path_str.ends_with('/') || path_str.ends_with('\\') {
                p.join(output_g8r_filename)
            } else {
                p.clone()
            };
            let stats_p = if p.is_dir() || path_str.ends_with('/') || path_str.ends_with('\\') {
                p.join(output_stats_filename)
            } else {
                p.with_file_name(
                    p.file_stem()
                        .unwrap_or_default()
                        .to_str()
                        .unwrap_or("best")
                        .to_owned()
                        + ".stats.json",
                )
            };
            (g8r, stats_p, dir, None)
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
                base_path.to_path_buf(),
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
    use std::sync::Mutex;
    let error_flag = Arc::new(Mutex::new(None));
    for i in 0..cli.threads {
        let cfg = cli.clone();
        let running_cl = running.clone();
        let best_cl = best.clone();
        let start_cl = start_gfn.clone();
        let seed_i = cli.seed ^ i as u64;
        let error_flag_cl = error_flag.clone();
        let output_dir_cl = output_dir.clone();
        handles.push(std::thread::spawn(move || {
            let progress_interval = cfg.progress_iters;
            if let Err(e) = run_chain(
                Arc::new(cfg),
                seed_i,
                start_cl,
                running_cl.clone(),
                best_cl,
                i as usize,
                Some(output_dir_cl.clone()),
                progress_interval,
            ) {
                let mut guard = error_flag_cl.lock().unwrap();
                *guard = Some(e);
                running_cl.store(false, Ordering::SeqCst);
            }
        }));
    }

    let mut last_print = Instant::now();
    let print_interval = Duration::from_secs(20); // Make global print less frequent
    for h in handles {
        while !h.is_finished() {
            if last_print.elapsed() > print_interval {
                let best_gfn = best.get();
                let stats = get_summary_stats::get_summary_stats(&best_gfn);
                let rss_mb = rss_megabytes().unwrap_or(0) as f64;
                let best_cost = match cli.metric {
                    Objective::Nodes => stats.live_nodes as usize,
                    Objective::Depth => stats.deepest_path as usize,
                    Objective::Product => stats.live_nodes * stats.deepest_path,
                };
                let improvement = if init_metric == 0 {
                    0.0
                } else {
                    100.0 * (init_metric as f64 - best_cost as f64) / (init_metric as f64)
                };
                let (_improvement_str, colorized_improvement) = if atty::is(Stream::Stdout) {
                    if best_cost < init_metric {
                        (
                            format!("{:.2}%", improvement),
                            format!("{:.2}%", improvement).green(),
                        )
                    } else if best_cost > init_metric {
                        (
                            format!("{:.2}%", improvement),
                            format!("{:.2}%", improvement).red(),
                        )
                    } else {
                        ("0.00%".to_string(), "0.00%".normal())
                    }
                } else {
                    let s = format!("{:.2}%", improvement);
                    (s.clone(), s.normal())
                };
                println!(
                    "[mcmc] [main] interim global best: nodes={}, depth={}, rss={:.3} GiB\n  original: nodes={}, depth={}, objective={}\n  global best: nodes={}, depth={}, objective={}\n  objective improvement: {} ({} mode)",
                    stats.live_nodes,
                    stats.deepest_path,
                    rss_mb / 1024.0,
                    initial_stats.live_nodes,
                    initial_stats.deepest_path,
                    init_metric,
                    stats.live_nodes,
                    stats.deepest_path,
                    best_cost,
                    colorized_improvement,
                    format!("{:?}", cli.metric)
                );
                last_print = Instant::now();
            }
            // If any thread reported an error, break out early
            if error_flag.lock().unwrap().is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = h.join();
    }
    if let Some(e) = error_flag.lock().unwrap().as_ref() {
        eprintln!(
            "[mcmc] ERROR: Aborting due to error in one of the chains: {}",
            e
        );
        std::process::exit(1);
    }

    if !running.load(Ordering::SeqCst) {
        let best_gfn = best.get();
        let stats = get_summary_stats::get_summary_stats(&best_gfn);
        println!("[mcmc] process was interrupted.");
        println!(
            "[mcmc] Global best at interruption: nodes={}, depth={}, rss={} MiB",
            stats.live_nodes,
            stats.deepest_path,
            rss_megabytes().unwrap_or(0)
        );
        println!(
            "[mcmc] Will write output to: {} and stats to: {}",
            output_g8r_path.display(),
            output_stats_path.display()
        );
    }

    println!("[mcmc] All MCMC chains joined. Writing final output...");

    // We are back on the main thread; write out the results once.
    let best_gfn = best.get();
    let final_summary_stats: SummaryStats = get_summary_stats::get_summary_stats(&best_gfn);
    println!(
        "[mcmc] finished. Final best GateFn stats: nodes={}, depth={}, rss={} MiB",
        final_summary_stats.live_nodes,
        final_summary_stats.deepest_path,
        rss_megabytes().unwrap_or(0)
    );

    println!(
        "[mcmc] Dumping best GateFn as text to: {}",
        output_g8r_path.display()
    );
    let mut f_g8r = fs::File::create(&output_g8r_path)?;
    f_g8r.write_all(best_gfn.to_string().as_bytes())?;
    println!(
        "[mcmc] Successfully wrote output to {}",
        output_g8r_path.display()
    );

    println!(
        "[mcmc] Dumping best GateFn stats as JSON to: {}",
        output_stats_path.display()
    );
    let mut f_stats = fs::File::create(&output_stats_path)?;
    let stats_json = serde_json::to_string_pretty(&final_summary_stats)?;
    f_stats.write_all(stats_json.as_bytes())?;
    println!(
        "[mcmc] Successfully wrote stats to {}",
        output_stats_path.display()
    );

    Ok(())
}
