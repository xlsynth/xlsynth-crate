// SPDX-License-Identifier: Apache-2.0
//! PIR MCMC sampler.
//!
//! This binary parses an XLS IR file into PIR, runs multi-chain MCMC over the
//! top function, and writes a deduplicated corpus of accepted equivalent
//! samples into an output directory:
//!
//!   samples/<structural_digest>.ir
//!
//! The digest is a deterministic BLAKE3-based structural hash of the function
//! return node's live cone (dead nodes do not affect the digest).
//!
//! Uniqueness is defined by the XLS-optimized IR: for each accepted sample we
//! run the XLS optimizer (via the same pipeline used for g8r costing) and then
//! structural-hash that optimized form.
//!
//! This tool strips position metadata (`pos=` attributes and file table
//! entries) from the input before running MCMC.

use anyhow::Result;
use clap::Parser;
use clap::ValueEnum;
use num_cpus;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write as IoWrite;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;
use std::time::Instant;
use tempfile::Builder;

use serde_json::json;
use xlsynth_mcmc::multichain::ChainStrategy;
use xlsynth_mcmc_pir::AcceptedSampleMsg;
use xlsynth_mcmc_pir::Objective;
use xlsynth_mcmc_pir::RunOptions;
use xlsynth_mcmc_pir::run_pir_mcmc_with_shared_best;
use xlsynth_pir::ir::{Package, PackageMember};
use xlsynth_pir::ir_parser::ParseOptions;
use xlsynth_pir::ir_utils::compact_and_toposort_in_place;
use xlsynth_pir::ir_validate;

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
    /// Input PIR IR file (.ir) to sample from.
    input_path: String,

    /// Number of MCMC iterations to perform per chain.
    #[clap(short = 'n', long, value_parser)]
    iters: u64,

    /// Random seed.
    #[clap(short = 'S', long, value_parser, default_value_t = 1)]
    seed: u64,

    /// Output directory. Samples are written under `samples/` within this dir.
    ///
    /// If not specified, output goes to a new temporary directory.
    #[clap(short, long, value_parser)]
    output: Option<String>,

    /// Metric to optimize (used by the MCMC acceptance criterion).
    #[clap(long, value_enum)]
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

    /// Progress logging interval in seconds for sampler corpus emission.
    ///
    /// Set to 0 to disable progress logs.
    #[clap(long, value_parser, default_value_t = 10)]
    progress_seconds: u64,

    /// When true, write per-chain append-only JSONL trajectory logs under the
    /// output directory.
    ///
    /// Files are written to `trajectory/trajectory.cXXX.jsonl`.
    ///
    /// Defaults to `true`. Disable with `--trajectory=false`.
    #[clap(long, default_value_t = true)]
    trajectory: bool,

    /// Override trajectory directory (implies `--trajectory=true`).
    #[clap(long, value_parser)]
    trajectory_dir: Option<String>,

    /// Enable a formal equivalence oracle (in addition to the interpreter-based
    /// oracle) for transforms that are not marked always-equivalent.
    ///
    /// Defaults to `true`. Disable with `--formal-oracle=false` if you want max
    /// throughput (but note that non-always-equivalent transforms are then
    /// disabled for safety).
    #[clap(long, default_value_t = true)]
    formal_oracle: bool,
}

fn hash_to_hex(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes.iter() {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn parse_hex_32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        let byte_str = &s[(2 * i)..(2 * i + 2)];
        let b = u8::from_str_radix(byte_str, 16).ok()?;
        out[i] = b;
    }
    Some(out)
}

fn resolve_output_dir(output: &Option<String>) -> Result<(PathBuf, Option<tempfile::TempDir>)> {
    match output {
        Some(path_str) => {
            let dir = PathBuf::from(path_str);
            std::fs::create_dir_all(&dir)?;
            Ok((dir, None))
        }
        None => {
            let temp_dir = Builder::new()
                .prefix("pir_mcmc_sampler_output_")
                .tempdir()?;
            Ok((temp_dir.path().to_path_buf(), Some(temp_dir)))
        }
    }
}

fn make_pkg_template_toposorted(pkg: &Package) -> Result<Package> {
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
    Ok(pkg)
}

fn replace_fn_in_pkg(pkg: &mut Package, new_fn: xlsynth_pir::ir::Fn) {
    if let Some(f) = pkg.get_fn_mut(&new_fn.name) {
        *f = new_fn;
        return;
    }
    if let Some(block) = pkg.get_block_mut(&new_fn.name) {
        match block {
            PackageMember::Block { func, .. } => {
                *func = new_fn;
                return;
            }
            PackageMember::Function(_) => unreachable!("get_block_mut must return a block"),
        }
    }
    panic!(
        "Expected to find function/block '{}' in template package",
        new_fn.name
    );
}

struct SampleWriter {
    pkg_template: Arc<Package>,
    samples_dir: PathBuf,
    manifest_path: PathBuf,
    trajectory_dir: Option<PathBuf>,
    seen: HashSet<[u8; 32]>,
    start: Instant,
    unique_written: u64,
    total_msgs: u64,
    last_report: Instant,
    report_interval: Duration,
}

impl SampleWriter {
    fn new(
        pkg_template: Arc<Package>,
        output_dir: &Path,
        trajectory_dir: Option<PathBuf>,
        report_interval: Duration,
    ) -> Result<Self> {
        let samples_dir = output_dir.join("samples");
        std::fs::create_dir_all(&samples_dir)?;
        let manifest_path = samples_dir.join("manifest.jsonl");

        let mut seen: HashSet<[u8; 32]> = HashSet::new();
        for entry in std::fs::read_dir(&samples_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("ir") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Some(d) = parse_hex_32(stem) else {
                continue;
            };
            seen.insert(d);
        }

        let now = Instant::now();
        Ok(SampleWriter {
            pkg_template,
            samples_dir,
            manifest_path,
            trajectory_dir,
            seen,
            start: now,
            unique_written: 0,
            total_msgs: 0,
            last_report: now,
            report_interval,
        })
    }

    fn write_if_new(&mut self, msg: AcceptedSampleMsg) -> Result<()> {
        self.total_msgs = self.total_msgs.saturating_add(1);
        if self.seen.contains(&msg.digest) {
            return Ok(());
        }

        let hex = hash_to_hex(&msg.digest);
        let final_path = self.samples_dir.join(format!("{}.ir", hex));
        if final_path.exists() {
            // Resume behavior: treat preexisting files as already-seen.
            self.seen.insert(msg.digest);
            return Ok(());
        }

        let mut pkg = (*self.pkg_template).clone();
        replace_fn_in_pkg(&mut pkg, msg.func);
        let pkg_text = pkg.to_string();

        let tmp_path = self.samples_dir.join(format!(
            ".{}.c{:03}-i{:06}.tmp",
            hex, msg.chain_no, msg.global_iter
        ));
        let mut f = File::create(&tmp_path)?;
        f.write_all(pkg_text.as_bytes())?;
        drop(f);
        std::fs::rename(&tmp_path, &final_path)?;

        // Record a stable mapping from (chain, iter) to the digest/filename.
        // This is append-only so users can correlate samples with trajectories.
        let rec = json!({
            "chain_no": msg.chain_no,
            "global_iter": msg.global_iter,
            "digest": hex,
            "path": format!("samples/{}.ir", hex),
            "pir_nodes": msg.cost.pir_nodes,
            "g8r_nodes": msg.cost.g8r_nodes,
            "g8r_depth": msg.cost.g8r_depth,
            "g8r_le_graph_milli": msg.cost.g8r_le_graph_milli,
        });
        let mut mf = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.manifest_path)?;
        writeln!(mf, "{}", rec.to_string())?;

        // Also emit an explicit "write event" record under the trajectory
        // directory (if enabled) so users can resolve written filenames from a
        // subset of the trajectory outputs without dedup ambiguity.
        if let Some(dir) = &self.trajectory_dir {
            std::fs::create_dir_all(dir)?;
            let path = dir.join(format!("writes.c{:03}.jsonl", msg.chain_no));
            let write_rec = json!({
                "chain_no": msg.chain_no,
                "global_iter": msg.global_iter,
                "digest": hex,
                "path": format!("samples/{}.ir", hex),
            });
            let mut wf = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)?;
            writeln!(wf, "{}", write_rec.to_string())?;
        }

        self.seen.insert(msg.digest);
        self.unique_written = self.unique_written.saturating_add(1);
        Ok(())
    }

    fn maybe_report_progress(&mut self) {
        if self.report_interval.is_zero() {
            return;
        }
        if self.last_report.elapsed() < self.report_interval {
            return;
        }
        self.last_report = Instant::now();

        let elapsed_secs = self.start.elapsed().as_secs_f64();
        let unique_per_sec = if elapsed_secs > 0.0 {
            (self.unique_written as f64) / elapsed_secs
        } else {
            0.0
        };
        log::info!(
            "[pir-mcmc-sampler] unique_written={} total_msgs={} unique_samples_per_sec={:.2}",
            self.unique_written,
            self.total_msgs,
            unique_per_sec
        );
    }
}

fn main() -> Result<()> {
    let _ = env_logger::try_init();

    let cli = CliArgs::parse();
    println!("PIR MCMC Sampler started with args: {:?}", cli);

    let input_path = PathBuf::from(&cli.input_path);
    let pkg = {
        let file_content = std::fs::read_to_string(&input_path)
            .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", input_path.display(), e))?;
        let opts = ParseOptions {
            retain_pos_data: false,
        };
        let mut parser = xlsynth_pir::ir_parser::Parser::new_with_options(&file_content, opts);
        let mut pkg = parser
            .parse_package()
            .map_err(|e| anyhow::anyhow!("PIR parse_package failed: {:?}", e))?;
        ir_validate::validate_package(&pkg)
            .map_err(|e| anyhow::anyhow!("PIR validate_package failed: {:?}", e))?;
        // Drop the file table so `to_string()` does not emit `file_number` lines.
        pkg.file_table.id_to_path.clear();
        pkg
    };

    let top_fn = pkg
        .get_top_fn()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("No top function found in PIR package"))?;

    let (output_dir, _temp_dir_holder) = resolve_output_dir(&cli.output)?;
    println!("Writing samples under: {}", output_dir.display());

    let pkg_template = Arc::new(make_pkg_template_toposorted(&pkg)?);

    let (tx, rx) = mpsc::channel::<AcceptedSampleMsg>();
    let output_dir_for_thread = output_dir.clone();
    let pkg_template_for_thread = pkg_template.clone();
    let report_interval = if cli.progress_seconds == 0 {
        Duration::from_secs(0)
    } else {
        Duration::from_secs(cli.progress_seconds)
    };
    let trajectory_dir_for_writer = cli.trajectory_dir.as_ref().map(PathBuf::from).or_else(|| {
        if cli.trajectory {
            Some(output_dir.join("trajectory"))
        } else {
            None
        }
    });
    let writer_handle = std::thread::spawn(move || -> Result<()> {
        let mut writer = SampleWriter::new(
            pkg_template_for_thread,
            &output_dir_for_thread,
            trajectory_dir_for_writer,
            report_interval,
        )?;
        loop {
            match rx.recv_timeout(Duration::from_secs(1)) {
                Ok(msg) => {
                    writer.write_if_new(msg)?;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            writer.maybe_report_progress();
        }
        Ok(())
    });

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
        trajectory_dir: cli.trajectory_dir.as_ref().map(PathBuf::from).or_else(|| {
            if cli.trajectory {
                Some(output_dir.join("trajectory"))
            } else {
                None
            }
        }),
    };

    let _result = run_pir_mcmc_with_shared_best(top_fn, opts, None, None, Some(tx))?;

    // Close the writer channel and join.
    let _ = writer_handle
        .join()
        .expect("sample writer thread panicked")?;

    Ok(())
}
