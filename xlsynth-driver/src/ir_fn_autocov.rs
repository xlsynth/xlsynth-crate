// SPDX-License-Identifier: Apache-2.0

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::ArgMatches;
use signal_hook::consts::signal::{SIGHUP, SIGINT, SIGTERM};
use signal_hook::low_level as siglow;
use xlsynth::IrValue;
use xlsynth_autocov::{
    AutocovConfig, AutocovEngine, AutocovProgress, CorpusSink, MuxNodeKind, ProgressSink,
};
use xlsynth_pir::ir_parser::Parser;

use crate::common::parse_bool_flag_or;
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;

fn mux_kind_name(kind: MuxNodeKind) -> &'static str {
    match kind {
        MuxNodeKind::Sel => "sel",
        MuxNodeKind::PrioritySel => "priority_sel",
        MuxNodeKind::OneHotSel => "one_hot_sel",
    }
}

pub fn handle_ir_fn_autocov(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let ir_input_file = matches
        .get_one::<String>("ir_input_file")
        .expect("ir_input_file is required");
    let corpus_file = matches
        .get_one::<String>("corpus_file")
        .expect("corpus_file is required");
    let seed = *matches
        .get_one::<u64>("seed")
        .expect("seed has a default value");
    let max_iters = matches.get_one::<u64>("max_iters").copied();
    let progress_every = *matches
        .get_one::<u64>("progress_every")
        .expect("progress_every has a default value");
    let threads = matches.get_one::<usize>("threads").copied();
    let seed_two_hot_max_bits = *matches
        .get_one::<usize>("seed_two_hot_max_bits")
        .expect("seed_two_hot_max_bits has a default value");
    let no_mux_space = parse_bool_flag_or(matches, "no_mux_space", false);
    let seed_structured = parse_bool_flag_or(matches, "seed_structured", true);

    let ir_text = match std::fs::read_to_string(ir_input_file) {
        Ok(content) => content,
        Err(e) => {
            report_cli_error_and_exit(
                &format!("Failed to read {}: {}", ir_input_file, e),
                Some("ir-fn-autocov"),
                vec![],
            );
        }
    };

    let mut parser = Parser::new(&ir_text);
    let mut pkg = match parser.parse_and_validate_package() {
        Ok(pkg) => pkg,
        Err(e) => {
            report_cli_error_and_exit(
                &format!("Failed to parse/validate IR package: {}", e),
                Some("ir-fn-autocov"),
                vec![],
            );
        }
    };

    if let Some(top) = matches.get_one::<String>("ir_top") {
        if let Err(e) = pkg.set_top_fn(top) {
            report_cli_error_and_exit(
                &format!("Failed to set --top: {}", e),
                Some("ir-fn-autocov"),
                vec![],
            );
        }
    }

    let entry_fn = match pkg.get_top_fn() {
        Some(f) => f.name.clone(),
        None => {
            report_cli_error_and_exit(
                "No top function found in package; provide --top",
                Some("ir-fn-autocov"),
                vec![],
            );
        }
    };

    let cfg = AutocovConfig { seed, max_iters };
    let mut engine = match AutocovEngine::from_ir_path(Path::new(ir_input_file), &entry_fn, cfg) {
        Ok(engine) => engine,
        Err(e) => {
            report_cli_error_and_exit(
                &format!("Failed to initialize autocov engine: {}", e),
                Some("ir-fn-autocov"),
                vec![],
            );
        }
    };

    if !no_mux_space {
        let summary = engine.get_mux_space_summary();
        eprintln!(
            "mux_space mux_count={} total_mux_feature_possibilities={} implied_log10_path_space_upper_bound={:.3}",
            summary.muxes.len(),
            summary.total_mux_feature_possibilities,
            summary.log10_path_space_upper_bound
        );
        for mux in &summary.muxes {
            eprintln!(
                "mux node_id={} kind={} cases_len={} has_default={} feature_possibilities={} log10_path_poss_upper_bound={:.3}",
                mux.node_text_id,
                mux_kind_name(mux.kind),
                mux.cases_len,
                mux.has_default,
                mux.feature_possibilities(),
                mux.log10_path_possibilities_upper_bound(),
            );
        }
    }

    let corpus_path = PathBuf::from(corpus_file);
    if let Ok(file) = std::fs::File::open(&corpus_path) {
        eprintln!("corpus_replay_begin path={}", corpus_path.display());
        let reader = BufReader::new(file);
        let replay_start = Instant::now();
        let mut replay_last = replay_start;
        let mut replay_lines: u64 = 0;
        for (line_index, line_result) in reader.lines().enumerate() {
            let line = match line_result {
                Ok(line) => line,
                Err(e) => {
                    report_cli_error_and_exit(
                        &format!("Failed reading corpus line {}: {}", line_index + 1, e),
                        Some("ir-fn-autocov"),
                        vec![],
                    );
                }
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let value = match IrValue::parse_typed(line) {
                Ok(v) => v,
                Err(e) => {
                    report_cli_error_and_exit(
                        &format!(
                            "Invalid typed value on corpus line {}: {}",
                            line_index + 1,
                            e
                        ),
                        Some("ir-fn-autocov"),
                        vec![],
                    );
                }
            };
            if let Err(e) = engine.add_corpus_sample_from_arg_tuple(&value) {
                report_cli_error_and_exit(
                    &format!(
                        "Failed to replay corpus line {} into engine: {}",
                        line_index + 1,
                        e
                    ),
                    Some("ir-fn-autocov"),
                    vec![],
                );
            }
            replay_lines += 1;
            if replay_lines % 10_000 == 0 {
                let now = Instant::now();
                let total_s = now.duration_since(replay_start).as_secs_f64().max(1e-9);
                let interval_s = now.duration_since(replay_last).as_secs_f64().max(1e-9);
                let total_rate = (replay_lines as f64) / total_s;
                let interval_rate = 10_000f64 / interval_s;
                eprintln!(
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
                );
                replay_last = now;
            }
        }
        let replay_total_s = Instant::now()
            .duration_since(replay_start)
            .as_secs_f64()
            .max(1e-9);
        eprintln!(
            "corpus_replay_end lines={} corpus_len={} seconds={:.3} lines_per_sec={:.1}",
            replay_lines,
            engine.corpus_len(),
            replay_total_s,
            (replay_lines as f64) / replay_total_s
        );
    }

    let file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&corpus_path)
    {
        Ok(file) => file,
        Err(e) => {
            report_cli_error_and_exit(
                &format!(
                    "Failed to open corpus file for append {}: {}",
                    corpus_path.display(),
                    e
                ),
                Some("ir-fn-autocov"),
                vec![],
            );
        }
    };
    let mut writer = BufWriter::new(file);

    struct FileSink<'a, W: Write> {
        writer: &'a mut W,
    }

    impl<W: Write> CorpusSink for FileSink<'_, W> {
        fn on_new_sample(&mut self, tuple_value: &IrValue) {
            writeln!(self.writer, "{}", tuple_value).expect("corpus write should succeed");
        }
    }

    let mut sink = FileSink {
        writer: &mut writer,
    };

    if seed_structured {
        let added = engine.seed_structured_corpus(seed_two_hot_max_bits, Some(&mut sink));
        eprintln!(
            "seed_structured added={} two_hot_max_bits={}",
            added, seed_two_hot_max_bits
        );
        if let Err(e) = sink.writer.flush() {
            report_cli_error_and_exit(
                &format!("Failed to flush corpus writer after structured seed: {}", e),
                Some("ir-fn-autocov"),
                vec![],
            );
        }
    }

    {
        let report = engine.get_mux_outcome_report();
        eprintln!(
            "mux_outcomes_summary total_possible={} total_missing={}",
            report.total_possible, report.total_missing
        );
        for entry in &report.entries {
            eprintln!(
                "mux_outcomes node_id={} kind={} observed={}/{} missing={:?}",
                entry.node_text_id,
                mux_kind_name(entry.kind),
                entry.observed_count,
                entry.possible_count,
                entry.missing
            );
        }
    }

    struct StderrProgressSink {
        start: Instant,
        last_report: Instant,
        last_report_iters: u64,
    }

    impl ProgressSink for StderrProgressSink {
        fn on_progress(&mut self, p: AutocovProgress) {
            let now = Instant::now();
            let total_secs = now.duration_since(self.start).as_secs_f64().max(1e-9);
            let interval_secs = now.duration_since(self.last_report).as_secs_f64().max(1e-9);
            let total_sps = (p.iters as f64) / total_secs;
            let delta_iters = p.iters.saturating_sub(self.last_report_iters);
            let interval_sps = (delta_iters as f64) / interval_secs;

            if p.last_iter_added {
                let new_cov = p
                    .new_coverage
                    .map(|c| format!("[{}]", c.kind_names().join(",")))
                    .unwrap_or_else(|| "[]".to_string());
                eprintln!(
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
                );
            } else {
                eprintln!(
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
    let stop = Arc::new(AtomicBool::new(false));
    engine.set_stop_flag(Arc::clone(&stop));
    let mut sig_ids = Vec::new();
    for sig in [SIGINT, SIGTERM, SIGHUP] {
        let stop = Arc::clone(&stop);
        // Set the run-scoped stop flag so Ctrl-C exits via normal report/flush path.
        match unsafe { siglow::register(sig, move || stop.store(true, Ordering::Relaxed)) } {
            Ok(id) => sig_ids.push(id),
            Err(e) => eprintln!(
                "warning: failed to register signal handler for signal {}: {}",
                sig, e
            ),
        }
    }

    let threads = threads.unwrap_or_else(|| std::thread::available_parallelism().unwrap().get());
    let report = if threads <= 1 {
        engine.run_with_sinks(Some(&mut sink), Some(&mut progress), Some(progress_every))
    } else {
        engine.run_parallel_with_sinks(
            threads,
            Some(&mut sink),
            Some(&mut progress),
            Some(progress_every),
        )
    };
    for id in sig_ids.drain(..) {
        let _ = siglow::unregister(id);
    }
    if let Err(e) = writer.flush() {
        report_cli_error_and_exit(
            &format!(
                "Failed to flush corpus output {}: {}",
                corpus_path.display(),
                e
            ),
            Some("ir-fn-autocov"),
            vec![],
        );
    }

    {
        let report = engine.get_mux_outcome_report();
        eprintln!(
            "mux_outcomes_summary_end total_possible={} total_missing={}",
            report.total_possible, report.total_missing
        );
    }

    println!(
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
        report.mux_outcomes_missing
    );
}
