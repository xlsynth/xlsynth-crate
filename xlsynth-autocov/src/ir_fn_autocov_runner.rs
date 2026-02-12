// SPDX-License-Identifier: Apache-2.0

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use signal_hook::SigId;
use signal_hook::consts::signal::{SIGHUP, SIGINT, SIGTERM};
use signal_hook::low_level as siglow;
use xlsynth::IrValue;
use xlsynth_pir::ir_parser::Parser;

use crate::{
    AutocovConfig, AutocovEngine, AutocovProgress, AutocovReport, CorpusSink, MuxNodeKind,
    ProgressSink,
};

#[derive(Debug, Clone)]
pub struct IrFnAutocovRunConfig {
    pub ir_input_file: PathBuf,
    pub entry_fn: String,
    pub corpus_file: PathBuf,
    pub seed: u64,
    pub max_iters: Option<u64>,
    pub progress_every: u64,
    pub no_mux_space: bool,
    pub threads: Option<usize>,
    pub seed_structured: bool,
    pub seed_two_hot_max_bits: usize,
    pub install_signal_handlers: bool,
}

pub fn resolve_entry_fn_from_ir_path(
    ir_input_file: &Path,
    top: Option<&str>,
) -> Result<String, String> {
    let ir_text = std::fs::read_to_string(ir_input_file)
        .map_err(|e| format!("Failed to read {}: {}", ir_input_file.display(), e))?;
    let mut parser = Parser::new(&ir_text);
    let mut pkg = parser
        .parse_and_validate_package()
        .map_err(|e| format!("Failed to parse/validate IR package: {}", e))?;
    if let Some(top) = top {
        pkg.set_top_fn(top)
            .map_err(|e| format!("Failed to set --top: {}", e))?;
    }
    let top_fn = pkg
        .get_top_fn()
        .ok_or_else(|| "No top function found in package; provide --top".to_string())?;
    Ok(top_fn.name.clone())
}

fn mux_kind_name(kind: MuxNodeKind) -> &'static str {
    match kind {
        MuxNodeKind::Sel => "sel",
        MuxNodeKind::PrioritySel => "priority_sel",
        MuxNodeKind::OneHotSel => "one_hot_sel",
    }
}

struct SignalHandlers {
    sig_ids: Vec<SigId>,
}

impl Drop for SignalHandlers {
    fn drop(&mut self) {
        for sig_id in self.sig_ids.drain(..) {
            let _ = siglow::unregister(sig_id);
        }
    }
}

fn install_signal_handlers<W: Write>(
    stop: Arc<AtomicBool>,
    stderr: &mut W,
) -> Result<SignalHandlers, String> {
    let mut sig_ids = Vec::new();
    for sig in [SIGINT, SIGTERM, SIGHUP] {
        let stop = Arc::clone(&stop);
        // Set run-scoped stop flag so Ctrl-C exits via normal report/flush path.
        match unsafe { siglow::register(sig, move || stop.store(true, Ordering::Relaxed)) } {
            Ok(sig_id) => sig_ids.push(sig_id),
            Err(e) => {
                writeln!(
                    stderr,
                    "warning: failed to register signal handler for signal {}: {}",
                    sig, e
                )
                .map_err(|e| format!("Failed writing warning to stderr: {}", e))?;
            }
        }
    }
    Ok(SignalHandlers { sig_ids })
}

struct FileSink<'a, W: Write> {
    writer: &'a mut W,
    stop: Arc<AtomicBool>,
    write_error: Option<String>,
}

impl<W: Write> FileSink<'_, W> {
    fn new(writer: &mut W, stop: Arc<AtomicBool>) -> FileSink<'_, W> {
        FileSink {
            writer,
            stop,
            write_error: None,
        }
    }

    fn take_error(&mut self) -> Option<String> {
        self.write_error.take()
    }
}

impl<W: Write> CorpusSink for FileSink<'_, W> {
    fn on_new_sample(&mut self, tuple_value: &IrValue) {
        if self.write_error.is_some() {
            return;
        }
        if let Err(e) = writeln!(self.writer, "{}", tuple_value) {
            self.write_error = Some(format!("Failed to append corpus sample: {}", e));
            self.stop.store(true, Ordering::Relaxed);
        }
    }
}

struct WriterProgressSink<'a, W: Write> {
    writer: &'a mut W,
    start: Instant,
    last_report: Instant,
    last_report_iters: u64,
    write_error: Option<String>,
}

impl<W: Write> WriterProgressSink<'_, W> {
    fn new(writer: &mut W, now: Instant) -> WriterProgressSink<'_, W> {
        WriterProgressSink {
            writer,
            start: now,
            last_report: now,
            last_report_iters: 0,
            write_error: None,
        }
    }

    fn take_error(&mut self) -> Option<String> {
        self.write_error.take()
    }
}

impl<W: Write> ProgressSink for WriterProgressSink<'_, W> {
    fn on_progress(&mut self, p: AutocovProgress) {
        if self.write_error.is_some() {
            return;
        }

        let now = Instant::now();
        let total_secs = now.duration_since(self.start).as_secs_f64().max(1e-9);
        let interval_secs = now.duration_since(self.last_report).as_secs_f64().max(1e-9);
        let total_sps = (p.iters as f64) / total_secs;
        let delta_iters = p.iters.saturating_sub(self.last_report_iters);
        let interval_sps = (delta_iters as f64) / interval_secs;

        let write_result = if p.last_iter_added {
            let new_cov = p
                .new_coverage
                .map(|c| format!("[{}]", c.kind_names().join(",")))
                .unwrap_or_else(|| "[]".to_string());
            writeln!(
                self.writer,
                "new_coverage {} iters={} corpus_len={} mux_features_set={} path_features_set={} bools_features_set={} corner_features_set={} compare_distance_features_set={} failure_features_set={} mux_outcomes_observed={} mux_outcomes_possible={} mux_outcomes_missing={} samples_per_sec={:.1} interval_samples_per_sec={:.1}",
                new_cov,
                p.iters,
                p.corpus_len,
                p.mux_features_set,
                p.path_features_set,
                p.bools_features_set,
                p.corner_features_set,
                p.compare_distance_features_set,
                p.failure_features_set,
                p.mux_outcomes_observed,
                p.mux_outcomes_possible,
                p.mux_outcomes_missing,
                total_sps,
                interval_sps
            )
        } else {
            writeln!(
                self.writer,
                "progress iters={} corpus_len={} mux_features_set={} path_features_set={} bools_features_set={} corner_features_set={} compare_distance_features_set={} failure_features_set={} mux_outcomes_observed={} mux_outcomes_possible={} mux_outcomes_missing={} samples_per_sec={:.1} interval_samples_per_sec={:.1}",
                p.iters,
                p.corpus_len,
                p.mux_features_set,
                p.path_features_set,
                p.bools_features_set,
                p.corner_features_set,
                p.compare_distance_features_set,
                p.failure_features_set,
                p.mux_outcomes_observed,
                p.mux_outcomes_possible,
                p.mux_outcomes_missing,
                total_sps,
                interval_sps
            )
        };
        if let Err(e) = write_result {
            self.write_error = Some(format!("Failed writing progress line: {}", e));
        }

        self.last_report = now;
        self.last_report_iters = p.iters;
    }
}

pub fn run_ir_fn_autocov_with_writers<WOut: Write, WErr: Write>(
    config: &IrFnAutocovRunConfig,
    stdout: &mut WOut,
    stderr: &mut WErr,
) -> Result<AutocovReport, String> {
    let cfg = AutocovConfig {
        seed: config.seed,
        max_iters: config.max_iters,
    };
    let mut engine = AutocovEngine::from_ir_path(&config.ir_input_file, &config.entry_fn, cfg)
        .map_err(|e| format!("Failed to initialize autocov engine: {}", e))?;

    let stop = Arc::new(AtomicBool::new(false));
    engine.set_stop_flag(Arc::clone(&stop));
    let _signal_handlers = if config.install_signal_handlers {
        Some(install_signal_handlers(Arc::clone(&stop), stderr)?)
    } else {
        None
    };

    if !config.no_mux_space {
        let summary = engine.get_mux_space_summary();
        writeln!(
            stderr,
            "mux_space mux_count={} total_mux_feature_possibilities={} implied_log10_path_space_upper_bound={:.3}",
            summary.muxes.len(),
            summary.total_mux_feature_possibilities,
            summary.log10_path_space_upper_bound
        )
        .map_err(|e| format!("Failed writing mux_space summary: {}", e))?;
        for mux in &summary.muxes {
            writeln!(
                stderr,
                "mux node_id={} kind={} cases_len={} has_default={} feature_possibilities={} log10_path_poss_upper_bound={:.3}",
                mux.node_text_id,
                mux_kind_name(mux.kind),
                mux.cases_len,
                mux.has_default,
                mux.feature_possibilities(),
                mux.log10_path_possibilities_upper_bound(),
            )
            .map_err(|e| format!("Failed writing mux_space entry: {}", e))?;
        }
    }

    if let Ok(file) = std::fs::File::open(&config.corpus_file) {
        writeln!(
            stderr,
            "corpus_replay_begin path={}",
            config.corpus_file.display()
        )
        .map_err(|e| format!("Failed writing corpus replay begin line: {}", e))?;
        let reader = BufReader::new(file);
        let replay_start = Instant::now();
        let mut replay_last = replay_start;
        let mut replay_lines: u64 = 0;

        for (line_index, line_result) in reader.lines().enumerate() {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let line = line_result
                .map_err(|e| format!("Failed reading corpus line {}: {}", line_index + 1, e))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let value = IrValue::parse_typed(line).map_err(|e| {
                format!(
                    "Invalid typed value on corpus line {}: {}",
                    line_index + 1,
                    e
                )
            })?;
            engine
                .add_corpus_sample_from_arg_tuple(&value)
                .map_err(|e| {
                    format!(
                        "Failed to replay corpus line {} into engine: {}",
                        line_index + 1,
                        e
                    )
                })?;
            replay_lines += 1;
            if replay_lines % 10_000 == 0 {
                let now = Instant::now();
                let total_s = now.duration_since(replay_start).as_secs_f64().max(1e-9);
                let interval_s = now.duration_since(replay_last).as_secs_f64().max(1e-9);
                let total_rate = (replay_lines as f64) / total_s;
                let interval_rate = 10_000f64 / interval_s;
                writeln!(
                    stderr,
                    "corpus_replay progress lines={} corpus_len={} mux_features_set={} path_features_set={} bools_features_set={} corner_features_set={} compare_distance_features_set={} failure_features_set={} lines_per_sec={:.1} interval_lines_per_sec={:.1}",
                    replay_lines,
                    engine.corpus_len(),
                    engine.mux_features_set(),
                    engine.path_features_set(),
                    engine.bools_features_set(),
                    engine.corner_features_set(),
                    engine.compare_distance_features_set(),
                    engine.failure_features_set(),
                    total_rate,
                    interval_rate
                )
                .map_err(|e| format!("Failed writing corpus replay progress line: {}", e))?;
                replay_last = now;
            }
        }

        let replay_total_s = Instant::now()
            .duration_since(replay_start)
            .as_secs_f64()
            .max(1e-9);
        writeln!(
            stderr,
            "corpus_replay_end lines={} corpus_len={} seconds={:.3} lines_per_sec={:.1}",
            replay_lines,
            engine.corpus_len(),
            replay_total_s,
            (replay_lines as f64) / replay_total_s
        )
        .map_err(|e| format!("Failed writing corpus replay end line: {}", e))?;
    }

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&config.corpus_file)
        .map_err(|e| {
            format!(
                "Failed to open corpus file for append {}: {}",
                config.corpus_file.display(),
                e
            )
        })?;
    let mut writer = BufWriter::new(file);
    let mut sink = FileSink::new(&mut writer, Arc::clone(&stop));

    if config.seed_structured && !stop.load(Ordering::Relaxed) {
        let added = engine.seed_structured_corpus(config.seed_two_hot_max_bits, Some(&mut sink));
        if let Some(e) = sink.take_error() {
            return Err(e);
        }
        writeln!(
            stderr,
            "seed_structured added={} two_hot_max_bits={}",
            added, config.seed_two_hot_max_bits
        )
        .map_err(|e| format!("Failed writing seed_structured line: {}", e))?;
        sink.writer
            .flush()
            .map_err(|e| format!("Failed to flush corpus writer after structured seed: {}", e))?;
    }

    {
        let report = engine.get_mux_outcome_report();
        writeln!(
            stderr,
            "mux_outcomes_summary total_possible={} total_missing={}",
            report.total_possible, report.total_missing
        )
        .map_err(|e| format!("Failed writing mux_outcomes_summary line: {}", e))?;
        for entry in &report.entries {
            writeln!(
                stderr,
                "mux_outcomes node_id={} kind={} observed={}/{} missing={:?}",
                entry.node_text_id,
                mux_kind_name(entry.kind),
                entry.observed_count,
                entry.possible_count,
                entry.missing
            )
            .map_err(|e| format!("Failed writing mux_outcomes entry line: {}", e))?;
        }
    }

    let report = {
        let now = Instant::now();
        let mut progress = WriterProgressSink::new(stderr, now);
        let threads = config
            .threads
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
        let report = if threads <= 1 {
            engine.run_with_sinks(
                Some(&mut sink),
                Some(&mut progress),
                Some(config.progress_every),
            )
        } else {
            engine.run_parallel_with_sinks(
                threads,
                Some(&mut sink),
                Some(&mut progress),
                Some(config.progress_every),
            )
        };
        if let Some(e) = progress.take_error() {
            return Err(e);
        }
        report
    };

    if let Some(e) = sink.take_error() {
        return Err(e);
    }
    writer.flush().map_err(|e| {
        format!(
            "Failed to flush corpus output {}: {}",
            config.corpus_file.display(),
            e
        )
    })?;

    {
        let report = engine.get_mux_outcome_report();
        writeln!(
            stderr,
            "mux_outcomes_summary_end total_possible={} total_missing={}",
            report.total_possible, report.total_missing
        )
        .map_err(|e| format!("Failed writing mux_outcomes_summary_end line: {}", e))?;
    }

    writeln!(
        stdout,
        "iters={} corpus_len={} mux_features_set={} path_features_set={} bools_features_set={} corner_features_set={} compare_distance_features_set={} failure_features_set={} mux_outcomes_observed={} mux_outcomes_possible={} mux_outcomes_missing={}",
        report.iters,
        report.corpus_len,
        report.mux_features_set,
        report.path_features_set,
        report.bools_features_set,
        report.corner_features_set,
        report.compare_distance_features_set,
        report.failure_features_set,
        report.mux_outcomes_observed,
        report.mux_outcomes_possible,
        report.mux_outcomes_missing,
    )
    .map_err(|e| format!("Failed writing final summary line: {}", e))?;

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    struct AlwaysFailWriter;

    impl Write for AlwaysFailWriter {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(
                io::ErrorKind::StorageFull,
                "simulated ENOSPC",
            ))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn file_sink_sets_stop_flag_on_write_error() {
        let stop = Arc::new(AtomicBool::new(false));
        let mut writer = AlwaysFailWriter;
        let mut sink = FileSink::new(&mut writer, Arc::clone(&stop));
        let tuple_value = IrValue::make_tuple(&[
            IrValue::parse_typed("bits[1]:0").expect("literal should parse"),
            IrValue::parse_typed("bits[1]:1").expect("literal should parse"),
        ]);

        sink.on_new_sample(&tuple_value);

        assert!(
            stop.load(Ordering::Relaxed),
            "write failure should immediately request stop"
        );
        let err = sink.take_error().expect("write error should be captured");
        assert!(
            err.contains("Failed to append corpus sample"),
            "unexpected error message: {err}"
        );
    }
}
