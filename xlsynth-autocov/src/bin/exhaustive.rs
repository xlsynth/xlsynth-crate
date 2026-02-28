// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::time::Instant;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "xlsynth-autocov-exhaustive")]
#[command(
    about = "Exhaustively enumerates an XLS IR function input space and reports achievable coverage"
)]
struct Args {
    /// Path to an XLS IR text file (package).
    #[arg(long)]
    ir_file: PathBuf,

    /// Name of the function within the IR package to evaluate.
    #[arg(long)]
    entry_fn: String,

    /// Refuse to run if the total flattened argument bits exceed this bound.
    #[arg(long, default_value_t = 24)]
    max_bits: usize,

    /// Optional early stop after this many candidates (for debugging).
    #[arg(long)]
    max_iters: Option<u64>,

    /// Emit a progress line every N iterations.
    #[arg(long, default_value_t = 1_000_000)]
    progress_every: u64,

    /// Number of worker threads to use for candidate evaluation.
    ///
    /// If not provided, defaults to `std::thread::available_parallelism()`.
    #[arg(long)]
    threads: Option<usize>,
}

fn bits_from_counter(nbits: usize, mut ctr: u128) -> xlsynth::IrBits {
    let mut v: Vec<bool> = Vec::with_capacity(nbits);
    for _ in 0..nbits {
        v.push((ctr & 1) != 0);
        ctr >>= 1;
    }
    xlsynth::IrBits::from_lsb_is_0(&v)
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
        seed: 0,
        max_iters: None,
        max_corpus_len: None,
    };
    let mut engine =
        xlsynth_autocov::AutocovEngine::from_ir_path(&args.ir_file, &args.entry_fn, cfg)
            .map_err(|e| anyhow::anyhow!(e))?;
    engine.set_stop_flag(stop.clone());

    let nbits = engine.args_bit_count();
    if nbits > args.max_bits {
        return Err(anyhow::anyhow!(
            "refusing to enumerate: args_bit_count={} exceeds --max-bits={}",
            nbits,
            args.max_bits
        ));
    }
    if nbits >= 128 {
        return Err(anyhow::anyhow!(
            "refusing to enumerate: args_bit_count={} too large for u128 counter",
            nbits
        ));
    }

    let total: u128 = 1u128 << nbits;
    let max_iters_u128: u128 = args.max_iters.map(u128::from).unwrap_or(total);
    let limit = std::cmp::min(total, max_iters_u128);

    eprintln!(
        "exhaustive_begin args_bit_count={} total_space={} limit={}",
        nbits, total, limit
    );

    let threads = args
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().unwrap().get());
    assert!(threads > 0, "threads must be > 0");

    let start = Instant::now();
    let mut last = start;
    let mut last_iters: u128 = 0;
    let mut iters: u128 = 0;
    let mut successes: u128 = 0;
    let mut failures: u128 = 0;

    if threads <= 1 {
        for i in 0..limit {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let bits = bits_from_counter(nbits, i);
            let ok = engine.observe_candidate(&bits);
            iters = i + 1;
            if ok {
                successes += 1;
            } else {
                failures += 1;
            }
            if args.progress_every > 0 && (iters % (args.progress_every as u128) == 0) {
                let now = Instant::now();
                let total_s = now.duration_since(start).as_secs_f64().max(1e-9);
                let interval_s = now.duration_since(last).as_secs_f64().max(1e-9);
                let total_rate = (iters as f64) / total_s;
                let interval_rate = ((iters - last_iters) as f64) / interval_s;
                eprintln!(
                    "exhaustive_progress iters={} successes={} failures={} mux_features_set={} path_features_set={} bools_features_set={} corner_features_set={} compare_distance_features_set={} failure_features_set={} mux_outcomes_observed={} mux_outcomes_possible={} mux_outcomes_missing={} iters_per_sec={:.1} interval_iters_per_sec={:.1}",
                    iters,
                    successes,
                    failures,
                    engine.mux_features_set(),
                    engine.path_features_set(),
                    engine.bools_features_set(),
                    engine.corner_features_set(),
                    engine.compare_distance_features_set(),
                    engine.failure_features_set(),
                    engine.mux_outcomes_observed(),
                    engine.mux_outcomes_possible(),
                    engine.mux_outcomes_missing(),
                    total_rate,
                    interval_rate,
                );
                last = now;
                last_iters = iters;
            }
        }
    } else {
        #[derive(Debug)]
        #[allow(dead_code)]
        struct WorkItem {
            ctr: u128,
        }

        #[derive(Debug)]
        struct WorkResult {
            ctr: u128,
            obs: xlsynth_autocov::CandidateObservation,
        }

        let work_cap = std::cmp::max(threads * 16, 64);
        let (work_tx, work_rx): (SyncSender<WorkItem>, Receiver<WorkItem>) = sync_channel(work_cap);
        let (res_tx, res_rx) = sync_channel::<WorkResult>(work_cap);
        let work_rx = Arc::new(Mutex::new(work_rx));

        // Clone a minimal engine per worker to evaluate observations. This reuses
        // parsing/type information but keeps maps centralized in the
        // coordinator.
        //
        // Note: AutocovEngine is not cheaply cloneable (contains maps), so workers
        // reconstruct a lightweight evaluator by parsing the IR text once up
        // front.
        let ir_text = std::fs::read_to_string(&args.ir_file)?;
        let entry_fn = args.entry_fn.clone();

        fn spawn_worker(
            ir_text: Arc<String>,
            entry_fn: Arc<String>,
            work_rx: Arc<Mutex<Receiver<WorkItem>>>,
            res_tx: SyncSender<WorkResult>,
            nbits: usize,
        ) -> std::thread::JoinHandle<()> {
            std::thread::spawn(move || {
                let cfg = xlsynth_autocov::AutocovConfig {
                    seed: 0,
                    max_iters: None,
                    max_corpus_len: None,
                };
                let engine = xlsynth_autocov::AutocovEngine::from_ir_text(
                    ir_text.as_str(),
                    None,
                    entry_fn.as_str(),
                    cfg,
                )
                .expect("engine construct");
                loop {
                    let item = {
                        let rx = work_rx.lock().unwrap();
                        rx.recv()
                    };
                    let item = match item {
                        Ok(v) => v,
                        Err(_) => break,
                    };
                    let bits = bits_from_counter(nbits, item.ctr);
                    let obs = engine.evaluate_observation(&bits);
                    let _ = res_tx.send(WorkResult { ctr: item.ctr, obs });
                }
            })
        }

        let ir_text = Arc::new(ir_text);
        let entry_fn = Arc::new(entry_fn);
        let mut workers = Vec::with_capacity(threads);
        for _ in 0..threads {
            workers.push(spawn_worker(
                ir_text.clone(),
                entry_fn.clone(),
                work_rx.clone(),
                res_tx.clone(),
                nbits,
            ));
        }
        drop(res_tx);

        let mut next_send: u128 = 0;
        let mut inflight: usize = 0;

        // Prime the pipeline.
        while inflight < work_cap && next_send < limit {
            let _ = work_tx.send(WorkItem { ctr: next_send });
            inflight += 1;
            next_send += 1;
        }

        while inflight > 0 {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let r = match res_rx.recv() {
                Ok(r) => r,
                Err(_) => break,
            };
            let _ = r.ctr;
            inflight -= 1;
            iters += 1;
            if r.obs.ok {
                successes += 1;
            } else {
                failures += 1;
            }
            engine.apply_observation(&r.obs);

            if next_send < limit {
                let _ = work_tx.send(WorkItem { ctr: next_send });
                inflight += 1;
                next_send += 1;
            }

            if args.progress_every > 0 && (iters % (args.progress_every as u128) == 0) {
                let now = Instant::now();
                let total_s = now.duration_since(start).as_secs_f64().max(1e-9);
                let interval_s = now.duration_since(last).as_secs_f64().max(1e-9);
                let total_rate = (iters as f64) / total_s;
                let interval_rate = ((iters - last_iters) as f64) / interval_s;
                eprintln!(
                    "exhaustive_progress iters={} successes={} failures={} mux_features_set={} path_features_set={} bools_features_set={} corner_features_set={} compare_distance_features_set={} failure_features_set={} mux_outcomes_observed={} mux_outcomes_possible={} mux_outcomes_missing={} iters_per_sec={:.1} interval_iters_per_sec={:.1}",
                    iters,
                    successes,
                    failures,
                    engine.mux_features_set(),
                    engine.path_features_set(),
                    engine.bools_features_set(),
                    engine.corner_features_set(),
                    engine.compare_distance_features_set(),
                    engine.failure_features_set(),
                    engine.mux_outcomes_observed(),
                    engine.mux_outcomes_possible(),
                    engine.mux_outcomes_missing(),
                    total_rate,
                    interval_rate,
                );
                last = now;
                last_iters = iters;
            }
        }

        drop(work_tx);
        for h in workers {
            let _ = h.join();
        }
    }

    println!(
        "iters={} successes={} failures={} mux_features_set={} path_features_set={} bools_features_set={} corner_features_set={} compare_distance_features_set={} failure_features_set={} mux_outcomes_observed={} mux_outcomes_possible={} mux_outcomes_missing={}",
        iters,
        successes,
        failures,
        engine.mux_features_set(),
        engine.path_features_set(),
        engine.bools_features_set(),
        engine.corner_features_set(),
        engine.compare_distance_features_set(),
        engine.failure_features_set(),
        engine.mux_outcomes_observed(),
        engine.mux_outcomes_possible(),
        engine.mux_outcomes_missing(),
    );

    Ok(())
}
