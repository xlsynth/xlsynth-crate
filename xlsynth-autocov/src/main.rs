// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use clap::Parser;
use xlsynth_autocov::{IrFnAutocovRunConfig, run_ir_fn_autocov_with_writers};

#[derive(Debug, Parser)]
#[command(name = "xlsynth-autocov")]
#[command(about = "Coverage-guided corpus growth for an XLS IR function")]
struct Args {
    /// Path to an XLS IR text file (package).
    #[arg(long)]
    ir_file: PathBuf,

    /// Name of the function within the IR package to evaluate.
    #[arg(long)]
    entry_fn: String,

    /// Newline-delimited corpus file. Each line is a typed IR tuple value whose
    /// elements are the function arguments.
    #[arg(long)]
    corpus_file: PathBuf,

    /// PRNG seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Maximum number of candidates to evaluate; if omitted, run until
    /// interrupted (SIGINT / ^C).
    #[arg(long)]
    max_iters: Option<u64>,

    /// Emit a progress line every N iterations (and also on each accepted
    /// sample).
    ///
    /// Set to 0 to disable periodic progress (new-sample progress still
    /// prints).
    #[arg(long, default_value_t = 10_000)]
    progress_every: u64,

    /// Disable printing the full mux-space summary at startup.
    #[arg(long, default_value_t = false)]
    no_mux_space: bool,

    /// Number of worker threads to use for candidate evaluation.
    ///
    /// If not provided, defaults to `std::thread::available_parallelism()`.
    #[arg(long)]
    threads: Option<usize>,

    /// Seed the corpus with structured bit patterns: all-zeros, all-ones, all
    /// one-hot, and all two-hot (subject to --seed-two-hot-max-bits).
    #[arg(long, default_value_t = true)]
    seed_structured: bool,

    /// Upper bound on bit-width for generating all two-hot seeds (quadratic).
    #[arg(long, default_value_t = 4096)]
    seed_two_hot_max_bits: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let config = IrFnAutocovRunConfig {
        ir_input_file: args.ir_file,
        entry_fn: args.entry_fn,
        corpus_file: args.corpus_file,
        seed: args.seed,
        max_iters: args.max_iters,
        progress_every: args.progress_every,
        no_mux_space: args.no_mux_space,
        threads: args.threads,
        seed_structured: args.seed_structured,
        seed_two_hot_max_bits: args.seed_two_hot_max_bits,
        install_signal_handlers: true,
    };

    let stdout = std::io::stdout();
    let stderr = std::io::stderr();
    let mut stdout = stdout.lock();
    let mut stderr = stderr.lock();

    run_ir_fn_autocov_with_writers(&config, &mut stdout, &mut stderr)
        .map_err(anyhow::Error::msg)?;
    Ok(())
}
