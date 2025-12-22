// SPDX-License-Identifier: Apache-2.0

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::Parser;

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
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        ctrlc::set_handler(move || {
            stop.store(true, Ordering::Relaxed);
        })?;
    }

    let cfg = xlsynth_autocov::AutocovConfig {
        seed: args.seed,
        max_iters: args.max_iters,
    };

    let mut engine =
        xlsynth_autocov::AutocovEngine::from_ir_path(&args.ir_file, &args.entry_fn, cfg)
            .map_err(|e| anyhow::anyhow!(e))?;
    engine.set_stop_flag(stop.clone());

    // Load existing corpus file (if any).
    if let Ok(f) = std::fs::File::open(&args.corpus_file) {
        let rdr = BufReader::new(f);
        for line in rdr.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v = xlsynth::IrValue::parse_typed(line)?;
            engine
                .add_corpus_sample_from_arg_tuple(&v)
                .map_err(|e| anyhow::anyhow!(e))?;
        }
    }

    // Open for append (create if missing).
    let f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.corpus_file)?;
    let mut w = BufWriter::new(f);

    struct FileSink<'a, W: Write> {
        w: &'a mut W,
    }

    impl<W: Write> xlsynth_autocov::CorpusSink for FileSink<'_, W> {
        fn on_new_sample(&mut self, tuple_value: &xlsynth::IrValue) {
            // Newline-delimited typed IR tuples.
            writeln!(self.w, "{}", tuple_value).expect("corpus write should succeed");
        }
    }

    let mut sink = FileSink { w: &mut w };
    let report = engine.run_with_sink(Some(&mut sink));
    w.flush()?;
    println!(
        "iters={} corpus_len={} mux_features_set={} path_features_set={}",
        report.iters, report.corpus_len, report.mux_features_set, report.path_features_set
    );

    Ok(())
}
