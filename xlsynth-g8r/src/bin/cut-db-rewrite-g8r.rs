// SPDX-License-Identifier: Apache-2.0

//! Runs only the cut-db rewrite pass on a serialized GateFn.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use clap::Parser;
use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig::cut_db_rewrite::{RewriteOptions, rewrite_gatefn_with_cut_db};
use xlsynth_g8r::aig::get_summary_stats::get_summary_stats;
use xlsynth_g8r::cut_db::loader::CutDb;
use xlsynth_g8r::cut_db_cli_defaults::{
    CUT_DB_REWRITE_MAX_CANDIDATE_EVALS_PER_ROUND_CLI, CUT_DB_REWRITE_MAX_CUTS_PER_NODE_CLI,
    CUT_DB_REWRITE_MAX_ITERATIONS_CLI, CUT_DB_REWRITE_MAX_REWRITES_PER_ROUND_CLI,
};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Input GateFn file (.g8r or .g8rbin).
    input: PathBuf,

    /// Optional output GateFn file. Uses bincode for .g8rbin, text otherwise.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Maximum global recompute rounds. 0 means run to convergence.
    #[arg(long, default_value_t = CUT_DB_REWRITE_MAX_ITERATIONS_CLI)]
    max_iterations: usize,

    /// Maximum cuts retained per node during cut enumeration. 0 is unbounded.
    #[arg(long, default_value_t = CUT_DB_REWRITE_MAX_CUTS_PER_NODE_CLI)]
    max_cuts_per_node: usize,

    /// Maximum cheap candidate depth evaluations per global recompute round. 0
    /// is unbounded.
    #[arg(long, default_value_t = CUT_DB_REWRITE_MAX_CANDIDATE_EVALS_PER_ROUND_CLI)]
    max_candidate_evals_per_round: usize,

    /// Maximum accepted replacements per global recompute round. 0 is
    /// unbounded.
    #[arg(long, default_value_t = CUT_DB_REWRITE_MAX_REWRITES_PER_ROUND_CLI)]
    max_rewrites_per_round: usize,
}

fn is_g8rbin(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("g8rbin"))
        .unwrap_or(false)
}

fn load_gate_fn(path: &Path) -> anyhow::Result<GateFn> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    if is_g8rbin(path) {
        bincode::deserialize(&bytes)
            .with_context(|| format!("failed to bincode-deserialize {}", path.display()))
    } else {
        let text = String::from_utf8(bytes)
            .with_context(|| format!("failed to decode utf8 from {}", path.display()))?;
        GateFn::try_from(text.as_str())
            .map_err(|e| anyhow::anyhow!("failed to parse {}: {}", path.display(), e))
    }
}

fn write_gate_fn(path: &Path, gate_fn: &GateFn) -> anyhow::Result<()> {
    if is_g8rbin(path) {
        let bytes = bincode::serialize(gate_fn)
            .with_context(|| format!("failed to serialize {}", path.display()))?;
        fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
    } else {
        fs::write(path, gate_fn.to_string())
            .with_context(|| format!("failed to write {}", path.display()))
    }
}

fn main() -> anyhow::Result<()> {
    let _ = env_logger::builder().try_init();
    let args = Args::parse();
    let gate_fn = load_gate_fn(&args.input)?;
    let before = get_summary_stats(&gate_fn);

    let cut_db = CutDb::load_default();
    let start = Instant::now();
    let rewritten = rewrite_gatefn_with_cut_db(
        &gate_fn,
        &cut_db,
        RewriteOptions {
            max_cuts_per_node: args.max_cuts_per_node,
            max_iterations: args.max_iterations,
            max_candidate_evals_per_round: args.max_candidate_evals_per_round,
            max_rewrites_per_round: args.max_rewrites_per_round,
        },
    );
    let elapsed = start.elapsed();
    let after = get_summary_stats(&rewritten);

    eprintln!(
        "cut-db rewrite: elapsed={:?} live_nodes {} -> {} depth {} -> {}",
        elapsed, before.live_nodes, after.live_nodes, before.deepest_path, after.deepest_path
    );

    if let Some(output) = args.output.as_ref() {
        write_gate_fn(output, &rewritten)?;
    }

    Ok(())
}
