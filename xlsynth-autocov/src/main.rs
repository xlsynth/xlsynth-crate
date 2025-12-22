// SPDX-License-Identifier: Apache-2.0

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

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

    if !args.no_mux_space {
        let summary = engine.get_mux_space_summary();
        eprintln!(
            "mux_space mux_count={} total_mux_feature_possibilities={} implied_log10_path_space_upper_bound={:.3}",
            summary.muxes.len(),
            summary.total_mux_feature_possibilities,
            summary.log10_path_space_upper_bound
        );
        for m in summary.muxes.iter() {
            let kind = match m.kind {
                xlsynth_autocov::MuxNodeKind::Sel => "sel",
                xlsynth_autocov::MuxNodeKind::PrioritySel => "priority_sel",
                xlsynth_autocov::MuxNodeKind::OneHotSel => "one_hot_sel",
            };
            eprintln!(
                "mux node_id={} kind={} cases_len={} has_default={} feature_possibilities={} log10_path_poss_upper_bound={:.3}",
                m.node_text_id,
                kind,
                m.cases_len,
                m.has_default,
                m.feature_possibilities(),
                m.log10_path_possibilities_upper_bound(),
            );
        }
    }

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
    struct StderrProgressSink {
        start: Instant,
        last_report: Instant,
        last_report_iters: u64,
    }

    impl xlsynth_autocov::ProgressSink for StderrProgressSink {
        fn on_progress(&mut self, p: xlsynth_autocov::AutocovProgress) {
            let now = Instant::now();
            let total_secs = now.duration_since(self.start).as_secs_f64().max(1e-9);
            let interval_secs = now.duration_since(self.last_report).as_secs_f64().max(1e-9);
            let total_sps = (p.iters as f64) / total_secs;
            let delta_iters = p.iters.saturating_sub(self.last_report_iters);
            let interval_sps = (delta_iters as f64) / interval_secs;

            if p.last_iter_added {
                eprintln!(
                    "new_coverage iters={} corpus_len={} mux_features_set={} path_features_set={} samples_per_sec={:.1} interval_samples_per_sec={:.1}",
                    p.iters,
                    p.corpus_len,
                    p.mux_features_set,
                    p.path_features_set,
                    total_sps,
                    interval_sps
                );
            } else {
                eprintln!(
                    "progress iters={} corpus_len={} mux_features_set={} path_features_set={} samples_per_sec={:.1} interval_samples_per_sec={:.1}",
                    p.iters,
                    p.corpus_len,
                    p.mux_features_set,
                    p.path_features_set,
                    total_sps,
                    interval_sps
                );
            }

            self.last_report = now;
            self.last_report_iters = p.iters;
        }
    }

    let now = Instant::now();
    let mut progress = StderrProgressSink {
        start: now,
        last_report: now,
        last_report_iters: 0,
    };
    let report = engine.run_with_sinks(
        Some(&mut sink),
        Some(&mut progress),
        Some(args.progress_every),
    );
    w.flush()?;
    println!(
        "iters={} corpus_len={} mux_features_set={} path_features_set={}",
        report.iters, report.corpus_len, report.mux_features_set, report.path_features_set
    );

    Ok(())
}
