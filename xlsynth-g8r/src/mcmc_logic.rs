// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;
use rand::prelude::{Rng, SeedableRng, SliceRandom};
use rand_pcg::Pcg64Mcg;

use xlsynth_mcmc::Best as SharedBest;
use xlsynth_mcmc::McmcIterationOutput as SharedMcmcIterationOutput;
use xlsynth_mcmc::McmcOptions as SharedMcmcOptions;
use xlsynth_mcmc::McmcStats as SharedMcmcStats;
use xlsynth_mcmc::metropolis_accept;

// Imports from the xlsynth_g8r crate
use crate::aig::gate::GateFn;
use crate::aig::{dce, get_summary_stats};
use crate::aig_serdes::ir2gate::{self, GatifyOptions};
use crate::aig_sim::gate_simd::{self, Vec256};

#[cfg(not(any(feature = "with-z3-system", feature = "with-z3-built")))]
use crate::check_equivalence::prove_same_gate_fn_via_ir;

#[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
use crate::prove_gate_fn_equiv_common::EquivResult;
#[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
use crate::prove_gate_fn_equiv_z3::{self, prove_gate_fn_equiv as prove_gate_fn_equiv_z3};

use crate::test_utils::{
    Opt as SampleOpt, load_bf16_add_sample, load_bf16_mul_sample, make_ripple_carry_adder,
};
use crate::transforms::get_all_transforms;
use crate::transforms::transform_trait::{TransformDirection, TransformKind};
use clap::ValueEnum;
use core::simd::u64x4;
use serde_json;
use std::fs::{self, OpenOptions};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use xlsynth_mcmc::MIN_TEMPERATURE_RATIO;
use xlsynth_pir::ir_parser;
const STATS_PRINT_ITERATION_INTERVAL: u64 = 1000;
const STATS_PRINT_TIME_INTERVAL_SECS: u64 = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cost {
    pub nodes: usize,
    pub depth: usize,
}

/// Calculates the cost of a GateFn based on its live nodes and depth.
pub fn cost(g: &GateFn) -> Cost {
    let stats = get_summary_stats::get_summary_stats(g);
    Cost {
        nodes: stats.live_nodes,
        depth: stats.deepest_path,
    }
}

/// Checks equivalence of two `GateFn`s using the external IR-based checker.
pub fn oracle_equiv_sat(lhs: &GateFn, rhs: &GateFn) -> bool {
    #[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
    {
        let mut ctx = prove_gate_fn_equiv_z3::Ctx::new();
        return matches!(
            prove_gate_fn_equiv_z3(lhs, rhs, &mut ctx),
            EquivResult::Proved
        );
    }

    // If we don't have Z3 we do via IR equivalence.
    #[cfg(not(any(feature = "with-z3-system", feature = "with-z3-built")))]
    {
        return prove_same_gate_fn_via_ir(lhs, rhs).is_ok();
    }
}

/// Type aliases specializing the generic MCMC helpers from `xlsynth-mcmc` to
/// the `xlsynth-g8r` AIG world.
pub type McmcStats = SharedMcmcStats<TransformKind>;
pub type Best = SharedBest<GateFn>;
pub type IterationOutcomeDetails = xlsynth_mcmc::IterationOutcomeDetails<TransformKind>;
pub type McmcIterationOutput = SharedMcmcIterationOutput<GateFn, Cost, TransformKind>;
pub type McmcOptions = SharedMcmcOptions;

/// Context for an MCMC iteration, holding shared resources.
pub struct McmcContext<'a> {
    pub rng: &'a mut Pcg64Mcg,
    pub all_transforms: Vec<Box<dyn crate::transforms::transform_trait::Transform>>,
    pub weights: Vec<f64>,
}

/// Objective used to evaluate cost improvements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Objective {
    Nodes,
    Depth,
    Product,
}

impl Objective {
    fn metric(self, c: &Cost) -> u64 {
        match self {
            Objective::Nodes => c.nodes as u64,
            Objective::Depth => c.depth as u64,
            Objective::Product => (c.nodes as u64) * (c.depth as u64),
        }
    }
}

/// Performs a single iteration of the MCMC process.
#[allow(clippy::too_many_arguments)]
pub fn mcmc_iteration(
    current_gfn: GateFn, /* Takes ownership, becomes the basis for candidate or returned if no
                          * change */
    current_cost: Cost,
    best_gfn: &mut GateFn, // Mutated if new best is found
    best_cost: &mut Cost,  // Mutated if new best is found
    context: &mut McmcContext,
    temp: f64,
    objective: Objective,
    paranoid: bool,
    simd_inputs: &[Vec256],
    baseline_outputs: &Vec<Vec256>,
) -> McmcIterationOutput {
    let mut iteration_best_gfn_updated = false;

    if context.all_transforms.is_empty() {
        // No transforms available to apply
        return McmcIterationOutput {
            output_state: current_gfn,
            output_cost: current_cost,
            best_updated: false,
            outcome: IterationOutcomeDetails::CandidateFailure, /* Or a new outcome? For now,
                                                                 * CandidateFailure */
            oracle_time_micros: 0,
            transform_always_equivalent: true,
            transform: None,
        };
    }

    let dist = WeightedIndex::new(&context.weights).expect("non-empty weights");
    let chosen_transform_idx = dist.sample(context.rng);
    let chosen_transform = &mut context.all_transforms[chosen_transform_idx];
    let current_transform_kind = chosen_transform.kind();

    let direction = if context.rng.r#gen::<bool>() {
        TransformDirection::Forward
    } else {
        TransformDirection::Backward
    };

    let candidate_locations = chosen_transform.find_candidates(&current_gfn, direction);

    log::trace!(
        "Found {} candidates for {:?} ({:?})",
        candidate_locations.len(),
        current_transform_kind,
        direction
    );

    if candidate_locations.is_empty() {
        return McmcIterationOutput {
            output_state: current_gfn,
            output_cost: current_cost,
            best_updated: false,
            outcome: IterationOutcomeDetails::CandidateFailure,
            oracle_time_micros: 0,
            transform_always_equivalent: true,
            transform: Some(current_transform_kind),
        };
    }

    let chosen_location = candidate_locations.choose(context.rng).unwrap();

    log::trace!("Chosen location: {:?}", chosen_location);

    let mut candidate_gfn = current_gfn.clone();

    log::trace!(
        "Applying transform {:?} ({:?}) to {:?}",
        current_transform_kind,
        direction,
        chosen_location
    );

    match chosen_transform.apply(&mut candidate_gfn, chosen_location, direction) {
        Ok(()) => {
            log::trace!("Transform applied successfully; determining cost...");
            // We'll compute cost after potential DCE later.
            let mut oracle_time_micros = 0u128;
            let is_equiv = if chosen_transform.always_equivalent() && !paranoid {
                true
            } else {
                let sim_start = Instant::now();
                let candidate_out = gate_simd::eval(&candidate_gfn, simd_inputs).outputs;
                let sim_equiv = *baseline_outputs == candidate_out;
                let sim_time_micros = sim_start.elapsed().as_micros();
                if !sim_equiv {
                    return McmcIterationOutput {
                        output_state: current_gfn,
                        output_cost: current_cost,
                        best_updated: false,
                        outcome: IterationOutcomeDetails::SimFailure,
                        oracle_time_micros: sim_time_micros,
                        transform_always_equivalent: chosen_transform.always_equivalent(),
                        transform: Some(current_transform_kind),
                    };
                }
                oracle_time_micros = sim_time_micros;
                let sat_res = oracle_equiv_sat(&current_gfn, &candidate_gfn);
                if paranoid {
                    let external_res = crate::check_equivalence::prove_same_gate_fn_via_ir(
                        &current_gfn,
                        &candidate_gfn,
                    )
                    .is_ok();
                    if sat_res != external_res {
                        panic!(
                            "[mcmc] ERROR: SAT oracle and external check_equivalence_with_top DISAGREE in mcmc_iteration: SAT oracle: {}, external: {}",
                            sat_res, external_res
                        );
                    }
                }
                sat_res
            };
            log::trace!("is_equiv: {:?}", is_equiv);

            if !is_equiv {
                McmcIterationOutput {
                    output_state: current_gfn,
                    output_cost: current_cost,
                    best_updated: false,
                    outcome: IterationOutcomeDetails::OracleFailure,
                    oracle_time_micros,
                    transform_always_equivalent: chosen_transform.always_equivalent(),
                    transform: Some(current_transform_kind),
                }
            } else {
                // Apply DCE to remove any dead nodes, reducing memory footprint.
                let candidate_gfn_dce = dce::dce(&candidate_gfn);

                let new_candidate_cost = cost(&candidate_gfn_dce);

                let curr_metric = objective.metric(&current_cost) as f64;
                let new_metric = objective.metric(&new_candidate_cost) as f64;
                let accept = metropolis_accept(curr_metric, new_metric, temp, context.rng);

                if accept {
                    if new_candidate_cost < *best_cost {
                        *best_gfn = candidate_gfn_dce.clone();
                        *best_cost = new_candidate_cost;
                        iteration_best_gfn_updated = true;
                    }
                    McmcIterationOutput {
                        output_state: candidate_gfn_dce,
                        output_cost: new_candidate_cost,
                        best_updated: iteration_best_gfn_updated,
                        outcome: IterationOutcomeDetails::Accepted {
                            kind: current_transform_kind,
                        },
                        oracle_time_micros,
                        transform_always_equivalent: chosen_transform.always_equivalent(),
                        transform: Some(current_transform_kind),
                    }
                } else {
                    McmcIterationOutput {
                        output_state: current_gfn,
                        output_cost: current_cost,
                        best_updated: false,
                        outcome: IterationOutcomeDetails::MetropolisReject,
                        oracle_time_micros,
                        transform_always_equivalent: chosen_transform.always_equivalent(),
                        transform: Some(current_transform_kind),
                    }
                }
            }
        }
        Err(e) => {
            log::debug!(
                "Error applying transform {:?}: {:?}",
                current_transform_kind,
                e
            );
            McmcIterationOutput {
                output_state: current_gfn,
                output_cost: current_cost,
                best_updated: false,
                outcome: IterationOutcomeDetails::ApplyFailure,
                oracle_time_micros: 0,
                transform_always_equivalent: true,
                transform: Some(current_transform_kind),
            }
        }
    }
}

/// Runs the MCMC optimization process.
pub fn mcmc(
    initial_gfn_param: GateFn,
    max_iters: u64,
    seed: u64,
    running: Arc<AtomicBool>,
    disabled_transform_names: Vec<String>,
    verbose: bool,
    objective: Objective,
    periodic_dump_dir: Option<PathBuf>,
    paranoid: bool,
    checkpoint_interval: u64,
    progress_interval: u64,
    shared_best: Option<Arc<Best>>,
    chain_no: Option<usize>,
    options: McmcOptions,
) -> Result<GateFn> {
    println!("Ticker Legend: F=ApplyFail, O=OracleFail, M=MetropolisReject, CF=CandidateFail");
    let mut iteration_rng = Pcg64Mcg::seed_from_u64(seed);

    let mut all_available_transforms = get_all_transforms();
    let disabled_kinds_set: std::collections::HashSet<String> =
        disabled_transform_names.into_iter().collect();

    all_available_transforms.retain(|t| {
        let kind_name = t.kind().to_string();
        if disabled_kinds_set.contains(&kind_name) {
            println!("Disabling transform: {}", kind_name);
            false
        } else {
            true
        }
    });

    if all_available_transforms.is_empty() {
        println!("Warning: All transforms have been disabled. No MCMC will be performed.");
        return Ok(initial_gfn_param);
    }

    println!(
        "Loaded {} transform kinds ({} total instances after filtering).",
        all_available_transforms
            .iter()
            .map(|t| t.kind())
            .collect::<std::collections::HashSet<_>>()
            .len(),
        all_available_transforms.len()
    );

    let original_gfn_for_check = initial_gfn_param.clone(); // Keep original for later equivalence checks

    let mut current_gfn = original_gfn_for_check.clone();
    let mut current_cost = cost(&current_gfn);
    let mut best_gfn = initial_gfn_param;
    let mut best_cost = current_cost;

    if let Some(ref b) = shared_best {
        b.try_update(objective.metric(&best_cost) as usize, best_gfn.clone());
    }

    let mut stats = McmcStats::default();

    let weights = build_transform_weights(&all_available_transforms, objective);

    // Pre-construct a 256-wide random input batch for fast SIMD equivalence
    // checking *before* we hand out a mutable borrow of `iteration_rng` to
    // `mcmc_context`.
    let simd_inputs = generate_simd_inputs(&original_gfn_for_check, &mut iteration_rng);

    // Compute baseline outputs for the initial GateFn.
    let baseline_outputs = gate_simd::eval(&original_gfn_for_check, &simd_inputs).outputs;

    let mut mcmc_context = McmcContext {
        rng: &mut iteration_rng,
        all_transforms: all_available_transforms,
        weights,
    };

    let start_time = Instant::now();
    let mut iterations_count: u64 = 0;
    let mut last_print_time = Instant::now();

    // Set up progress reporting channel and writer thread if needed
    let (progress_tx, progress_rx) = mpsc::channel::<ProgressEntry>();
    let writer_handle = if let Some(ref dump_dir) = periodic_dump_dir {
        let progress_path = dump_dir.join("progress.jsonl");
        Some(thread::spawn(move || {
            let mut f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(progress_path)
                .expect("Failed to open progress.jsonl for writing");
            for entry in progress_rx {
                let json = serde_json::to_string(&entry).expect("serialize progress");
                writeln!(f, "{}", json).expect("write progress");
            }
        }))
    } else {
        None
    };

    while running.load(Ordering::SeqCst) && iterations_count < max_iters {
        iterations_count += 1;

        let progress_ratio = match options.total_iters {
            Some(total) => {
                let done = options.start_iteration + iterations_count;
                (done as f64) / (total as f64)
            }
            None => 0.0, // constant temperature – explorer chain
        };
        let current_temp =
            options.initial_temperature * (1.0 - progress_ratio).max(MIN_TEMPERATURE_RATIO);

        // Print action about to be taken if verbose
        if verbose {
            // Print transform, direction, and candidate info
            if !mcmc_context.all_transforms.is_empty() {
                let chosen_transform_idx = mcmc_context
                    .rng
                    .gen_range(0..mcmc_context.all_transforms.len());
                let chosen_transform = &mut mcmc_context.all_transforms[chosen_transform_idx];
                let current_transform_kind = chosen_transform.kind();
                let direction = if mcmc_context.rng.r#gen::<bool>() {
                    TransformDirection::Forward
                } else {
                    TransformDirection::Backward
                };
                let candidate_locations = chosen_transform.find_candidates(&current_gfn, direction);
                if !candidate_locations.is_empty() {
                    let chosen_location = candidate_locations.choose(mcmc_context.rng).unwrap();
                    if let Some(chain) = chain_no {
                        println!(
                            "[mcmc][verbose] c{:03}:i{:06}: About to apply {:?} ({:?}) at {:?}",
                            chain,
                            options.start_iteration + iterations_count,
                            current_transform_kind,
                            direction,
                            chosen_location
                        );
                    } else {
                        println!(
                            "[mcmc][verbose] iter {}: About to apply {:?} ({:?}) at {:?}",
                            options.start_iteration + iterations_count,
                            current_transform_kind,
                            direction,
                            chosen_location
                        );
                    }
                } else {
                    if let Some(chain) = chain_no {
                        println!(
                            "[mcmc][verbose] c{:03}:i{:06}: No candidates for {:?} ({:?})",
                            chain,
                            options.start_iteration + iterations_count,
                            current_transform_kind,
                            direction
                        );
                    } else {
                        println!(
                            "[mcmc][verbose] iter {}: No candidates for {:?} ({:?})",
                            options.start_iteration + iterations_count,
                            current_transform_kind,
                            direction
                        );
                    }
                }
            }
        }

        log::trace!(
            "Starting MCMC iteration: {:?}",
            options.start_iteration + iterations_count
        );
        let iteration_output = mcmc_iteration(
            current_gfn, // current_gfn is moved in
            current_cost,
            &mut best_gfn,
            &mut best_cost,
            &mut mcmc_context,
            current_temp,
            objective,
            paranoid,
            &simd_inputs,
            &baseline_outputs,
        );
        log::trace!(
            "MCMC iteration completed: {:?}",
            options.start_iteration + iterations_count
        );

        // Update current_gfn and baseline outputs depending on acceptance.
        current_gfn = iteration_output.output_state.clone();
        current_cost = iteration_output.output_cost;
        stats.update_for_iteration(
            &iteration_output,
            paranoid,
            options.start_iteration + iterations_count,
        );

        if iteration_output.best_updated {
            if let Some(ref b) = shared_best {
                let _ = b.try_update(objective.metric(&best_cost) as usize, best_gfn.clone());
            }
        }

        if iterations_count % STATS_PRINT_ITERATION_INTERVAL == 0
            || Instant::now() - last_print_time
                > Duration::from_secs(STATS_PRINT_TIME_INTERVAL_SECS)
        {
            let elapsed_secs = start_time.elapsed().as_secs_f64();
            let samples_per_sec = if elapsed_secs > 0.0 {
                iterations_count as f64 / elapsed_secs
            } else {
                0.0
            };
            let avg_oracle_ms = if stats.oracle_verified + stats.rejected_oracle > 0 {
                (stats.total_oracle_time_micros as f64
                    / (stats.oracle_verified + stats.rejected_oracle) as f64)
                    / 1000.0
            } else {
                0.0
            };
            let avg_sim_ms = if stats.rejected_sim_fail > 0 {
                (stats.total_sim_time_micros as f64 / stats.rejected_sim_fail as f64) / 1000.0
            } else {
                0.0
            };

            let mut sorted_edits_vec: Vec<(&TransformKind, &usize)> =
                stats.accepted_edits_by_kind.iter().collect();
            sorted_edits_vec.sort_by_key(|&(k, _)| k);

            let accepted_edits_str = sorted_edits_vec
                .iter()
                .map(|(kind, count)| format!("{}:{}", kind, count))
                .collect::<Vec<String>>()
                .join(", ");

            if let Some(chain) = chain_no {
                println!(
                    "[mcmc] c{:03}:i{:06} | Best: (n={}, d={}) | Cur: (n={}, d={}) | Temp: {:.2e} | Samples/s: {:.2} | Rejected (AF/CF/SIM/O/M): {}/{}/{}/{}/{} | Oracle Ok: {} | Avg Oracle (ms): {:.3} | Avg Sim (ms): {:.3} | Accepted: {} ({})         ",
                    chain,
                    options.start_iteration + iterations_count,
                    best_cost.nodes,
                    best_cost.depth,
                    current_cost.nodes,
                    current_cost.depth,
                    current_temp,
                    samples_per_sec,
                    stats.rejected_apply_fail,
                    stats.rejected_candidate_fail,
                    stats.rejected_sim_fail,
                    stats.rejected_oracle,
                    stats.rejected_metro,
                    stats.oracle_verified,
                    avg_oracle_ms,
                    avg_sim_ms,
                    stats.accepted_overall,
                    if accepted_edits_str.is_empty() {
                        "-"
                    } else {
                        &accepted_edits_str
                    },
                );
            } else {
                println!(
                    "[mcmc] iter: {} | Best: (n={}, d={}) | Cur: (n={}, d={}) | Temp: {:.2e} | Samples/s: {:.2} | Rejected (AF/CF/SIM/O/M): {}/{}/{}/{}/{} | Oracle Ok: {} | Avg Oracle (ms): {:.3} | Avg Sim (ms): {:.3} | Accepted: {} ({})         ",
                    options.start_iteration + iterations_count,
                    best_cost.nodes,
                    best_cost.depth,
                    current_cost.nodes,
                    current_cost.depth,
                    current_temp,
                    samples_per_sec,
                    stats.rejected_apply_fail,
                    stats.rejected_candidate_fail,
                    stats.rejected_sim_fail,
                    stats.rejected_oracle,
                    stats.rejected_metro,
                    stats.oracle_verified,
                    avg_oracle_ms,
                    avg_sim_ms,
                    stats.accepted_overall,
                    if accepted_edits_str.is_empty() {
                        "-"
                    } else {
                        &accepted_edits_str
                    },
                );
            }
            let _ = std::io::stdout().flush();
            last_print_time = Instant::now();
        }

        if progress_interval > 0 {
            if let Some(_) = periodic_dump_dir {
                if iterations_count % progress_interval == 0 {
                    let elapsed_secs = start_time.elapsed().as_secs_f64();
                    let proposed_per_sec = if elapsed_secs > 0.0 {
                        iterations_count as f64 / elapsed_secs
                    } else {
                        0.0
                    };
                    let accepted_per_sec = if elapsed_secs > 0.0 {
                        stats.accepted_overall as f64 / elapsed_secs
                    } else {
                        0.0
                    };
                    let progress_entry = ProgressEntry {
                        utc_time_secs: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                        chain_number: chain_no.unwrap_or(0),
                        iterations: options.start_iteration + iterations_count,
                        current_depth: current_cost.depth,
                        current_nodes: current_cost.nodes,
                        proposed_samples_per_sec: proposed_per_sec,
                        accepted_samples_per_sec: accepted_per_sec,
                        temperature: current_temp,
                    };
                    let _ = progress_tx.send(progress_entry);
                }
            }
        }

        // Periodic dump of current best GateFn and its stats.
        if checkpoint_interval > 0 {
            if let Some(ref dump_dir) = periodic_dump_dir {
                if iterations_count % checkpoint_interval == 0 {
                    // Ensure directory exists.
                    let _ = std::fs::create_dir_all(dump_dir);
                    let prefix = if let Some(chain) = chain_no {
                        format!("c{}-", chain)
                    } else {
                        String::new()
                    };
                    let global_iter = options.start_iteration + iterations_count;
                    let g8r_path =
                        dump_dir.join(format!("{}best_iter_{}.g8r", prefix, global_iter));
                    let stats_path =
                        dump_dir.join(format!("{}best_iter_{}.stats.json", prefix, global_iter));
                    write_checkpoint(
                        &g8r_path,
                        &stats_path,
                        &original_gfn_for_check,
                        &best_gfn,
                        global_iter,
                        "Iter checkpoint",
                    )?;
                }
            }
        }
    }

    // Final checkpoint (for main thread write-out)
    if let Some(ref dump_dir) = periodic_dump_dir {
        let prefix = if let Some(chain) = chain_no {
            format!("c{}-", chain)
        } else {
            String::new()
        };
        let g8r_path = dump_dir.join(format!("{}final_best.g8r", prefix));
        let stats_path = dump_dir.join(format!("{}final_best.stats.json", prefix));
        write_checkpoint(
            &g8r_path,
            &stats_path,
            &original_gfn_for_check,
            &best_gfn,
            options.start_iteration + iterations_count,
            "Final checkpoint",
        )?;
        if progress_interval > 0 {
            let elapsed_secs = start_time.elapsed().as_secs_f64();
            let proposed_per_sec = if elapsed_secs > 0.0 {
                iterations_count as f64 / elapsed_secs
            } else {
                0.0
            };
            let accepted_per_sec = if elapsed_secs > 0.0 {
                stats.accepted_overall as f64 / elapsed_secs
            } else {
                0.0
            };
            let progress_ratio = (iterations_count as f64) / (max_iters as f64);
            let current_temp =
                options.initial_temperature * (1.0 - progress_ratio).max(MIN_TEMPERATURE_RATIO);
            let progress_entry = ProgressEntry {
                utc_time_secs: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                chain_number: chain_no.unwrap_or(0),
                iterations: options.start_iteration + iterations_count,
                current_depth: current_cost.depth,
                current_nodes: current_cost.nodes,
                proposed_samples_per_sec: proposed_per_sec,
                accepted_samples_per_sec: accepted_per_sec,
                temperature: current_temp,
            };
            let _ = progress_tx.send(progress_entry);
        }
    }
    // Close the progress channel and join the writer thread if it was spawned
    drop(progress_tx);
    if let Some(handle) = writer_handle {
        let _ = handle.join();
    }
    Ok(best_gfn)
}

/// Loads the starting `GateFn` from either a sample, a `.g8r` file, or an IR
/// file.
pub fn load_start<P: AsRef<Path>>(p_generic: P) -> Result<GateFn> {
    let p_str = p_generic.as_ref().to_str().unwrap_or_default();

    if p_str.starts_with("sample://") {
        let sample_name = p_str.trim_start_matches("sample://");
        match sample_name {
            "bf16_add" => {
                let loaded_sample = load_bf16_add_sample(SampleOpt::Yes);
                let sample_cost = cost(&loaded_sample.gate_fn);
                println!(
                    "Sample '{}' loaded. Initial stats: nodes={}, depth={}",
                    sample_name, sample_cost.nodes, sample_cost.depth
                );
                Ok(loaded_sample.gate_fn)
            }
            "bf16_mul" => {
                let loaded_sample = load_bf16_mul_sample(SampleOpt::Yes);
                let sample_cost = cost(&loaded_sample.gate_fn);
                println!(
                    "Sample '{}' loaded. Initial stats: nodes={}, depth={}",
                    sample_name, sample_cost.nodes, sample_cost.depth
                );
                Ok(loaded_sample.gate_fn)
            }
            _ if sample_name.starts_with("ripple_carry_adder:") => {
                let bits_str = sample_name.trim_start_matches("ripple_carry_adder:");
                let bits: usize = bits_str.parse().map_err(|_| {
                    anyhow::anyhow!("Invalid bit width '{}', expected integer", bits_str)
                })?;
                let gfn = make_ripple_carry_adder(bits);
                let sample_cost = cost(&gfn);
                println!(
                    "Sample '{}' loaded. Initial stats: nodes={}, depth={}",
                    sample_name, sample_cost.nodes, sample_cost.depth
                );
                Ok(gfn)
            }
            _ => Err(anyhow::anyhow!("Unknown sample name: {}", sample_name)),
        }
    } else {
        let path = p_generic.as_ref();
        match path.extension().and_then(|e| e.to_str()) {
            Some("g8r") => {
                println!("Loading GateFn from path: {}", p_str);
                let contents = fs::read_to_string(path).map_err(|e| {
                    anyhow::anyhow!("Failed to read GateFn file '{}': {}", p_str, e)
                })?;
                let gfn = GateFn::try_from(contents.as_str()).map_err(|e| {
                    anyhow::anyhow!("Failed to parse GateFn from '{}': {}", p_str, e)
                })?;
                let g_cost = cost(&gfn);
                println!(
                    "Loaded GateFn. Initial stats: nodes={}, depth={}",
                    g_cost.nodes, g_cost.depth
                );
                Ok(gfn)
            }
            _ => {
                println!("Loading IR from path: {}", p_str);
                let package = ir_parser::parse_and_validate_path_to_package(path).map_err(|e| {
                    anyhow::anyhow!("Failed to parse IR package '{}': {:?}", p_str, e)
                })?;

                let top_entity = package.get_top_fn().ok_or_else(|| {
                    anyhow::anyhow!("No top entity found in IR package '{}'", p_str)
                })?;
                println!("Found top function: {}", top_entity.name);

                let gatify_options = GatifyOptions {
                    fold: true,
                    hash: true,
                    check_equivalence: false,
                    adder_mapping: crate::ir2gate_utils::AdderMapping::default(),
                    mul_adder_mapping: None,
                    range_info: None,
                };
                let gatify_output = ir2gate::gatify(top_entity, gatify_options)
                    .map_err(|e| anyhow::anyhow!("Failed to gatify IR from '{}': {}", p_str, e))?;
                println!("Successfully gatified main function into GateFn.");
                Ok(gatify_output.gate_fn)
            }
        }
    }
}

/// Returns a vector of weights for the given transforms and objective.
pub fn build_transform_weights<
    T: AsRef<[Box<dyn crate::transforms::transform_trait::Transform>]>,
>(
    transforms: T,
    objective: Objective,
) -> Vec<f64> {
    use crate::transforms::transform_trait::TransformKind;
    fn weight_for_kind(k: TransformKind, obj: Objective) -> f64 {
        use TransformKind::*;
        match obj {
            Objective::Nodes | Objective::Product => match k {
                RemoveRedundantAnd | RemoveFalseAnd | RemoveTrueAnd | UnduplicateGate
                | MergeFanout => 3.0,
                InsertRedundantAnd | InsertFalseAnd | InsertTrueAnd | DuplicateGate
                | SplitFanout | UnfactorSharedAnd => 0.5,
                _ => 1.0,
            },
            Objective::Depth => match k {
                RotateAndRight | RotateAndLeft | BalanceAndTree | UnbalanceAndTree
                | SplitFanout | FactorSharedAnd => 3.0,
                _ => 1.0,
            },
        }
    }
    transforms
        .as_ref()
        .iter()
        .map(|t| weight_for_kind(t.kind(), objective))
        .collect()
}

fn write_checkpoint(
    g8r_path: &Path,
    stats_path: &Path,
    original_gfn: &GateFn,
    best_gfn: &GateFn,
    iter: u64,
    context: &str,
) -> Result<()> {
    // Cross-check equivalence
    let equiv_ok_sat = oracle_equiv_sat(original_gfn, best_gfn);
    use crate::check_equivalence::{IrCheckResult, prove_same_gate_fn_via_ir_status};

    let ir_status = prove_same_gate_fn_via_ir_status(original_gfn, best_gfn);
    let equiv_ok_external = matches!(ir_status, IrCheckResult::Equivalent);

    if equiv_ok_sat != equiv_ok_external || !equiv_ok_sat {
        // Ensure we persist the disagreeing pair for offline triage.
        if let Some(parent_dir) = g8r_path.parent() {
            let dump_dir = parent_dir.join("equiv_failures");
            let _ = std::fs::create_dir_all(&dump_dir);
            let orig_dump = dump_dir.join(format!("orig_iter_{}.g8r", iter));
            let cand_dump = dump_dir.join(format!("cand_iter_{}.g8r", iter));
            let orig_bin_dump = dump_dir.join(format!("orig_iter_{}.g8rbin", iter));
            let cand_bin_dump = dump_dir.join(format!("cand_iter_{}.g8rbin", iter));
            let _ = std::fs::write(&orig_dump, original_gfn.to_string());
            let _ = std::fs::write(&cand_dump, best_gfn.to_string());
            let _ = std::fs::write(&orig_bin_dump, bincode::serialize(original_gfn).unwrap());
            let _ = std::fs::write(&cand_bin_dump, bincode::serialize(best_gfn).unwrap());
            eprintln!(
                "[mcmc] Disagreeing GateFns dumped to {} and {} (text), {} and {} (bincode)",
                orig_dump.display(),
                cand_dump.display(),
                orig_bin_dump.display(),
                cand_bin_dump.display()
            );
        }

        return Err(anyhow::anyhow!(
            "[mcmc] ERROR: Equivalence disagreement at iter {} (sat:{}, external:{})",
            iter,
            equiv_ok_sat,
            equiv_ok_external
        ));
    }
    match ir_status {
        IrCheckResult::Equivalent => {}
        IrCheckResult::TimedOutOrInterrupted => {
            eprintln!(
                "[mcmc] Warning: External IR equivalence check timed out or was interrupted (iteration {}). Proceeding with SAT oracle result only.",
                iter
            );
        }
        IrCheckResult::OtherProcessError(ref msg) => {
            eprintln!(
                "[mcmc] Warning: External IR equivalence checker failed at iter {}: {}",
                iter, msg
            );
        }
        IrCheckResult::NotEquivalent => {}
    }
    if let Err(e) = std::fs::write(g8r_path, best_gfn.to_string()) {
        eprintln!(
            "[mcmc] Warning: Failed to write {} checkpoint to {}: {:?}",
            context,
            g8r_path.display(),
            e
        );
    }
    let stats = get_summary_stats::get_summary_stats(best_gfn);
    match serde_json::to_string_pretty(&stats) {
        Ok(json) => {
            if let Err(e) = std::fs::write(stats_path, json) {
                eprintln!(
                    "[mcmc] Warning: Failed to write {} stats checkpoint to {}: {:?}",
                    context,
                    stats_path.display(),
                    e
                );
            } else {
                println!(
                    "[mcmc] {}: checkpoint written to {} and {} | Equivalence: OK",
                    context,
                    g8r_path.display(),
                    stats_path.display()
                );
            }
        }
        Err(e) => {
            eprintln!(
                "[mcmc] Warning: Failed to serialize stats for {} checkpoint at iteration {}: {:?}",
                context, iter, e
            );
        }
    }
    Ok(())
}

/// Generates a fixed set of 256-wide random inputs matching the bit-width of
/// `gate_fn`'s declared inputs.
fn generate_simd_inputs(gate_fn: &GateFn, rng: &mut impl rand::Rng) -> Vec<Vec256> {
    const LANES: usize = 256;
    let total_bits: usize = gate_fn.inputs.iter().map(|i| i.get_bit_count()).sum();
    let mut words_per_bit = vec![[0u64; 4]; total_bits];

    for lane in 0..LANES {
        let mut bit_cursor = 0;
        for input in &gate_fn.inputs {
            let rand_val =
                xlsynth_pir::fuzz_utils::arbitrary_irbits(rng, input.bit_vector.get_bit_count());
            for bit_idx in 0..input.bit_vector.get_bit_count() {
                if rand_val.get_bit(bit_idx).unwrap() {
                    let limb = lane / 64;
                    let offset = lane % 64;
                    words_per_bit[bit_cursor + bit_idx][limb] |= 1u64 << offset;
                }
            }
            bit_cursor += input.bit_vector.get_bit_count();
        }
    }

    words_per_bit
        .into_iter()
        .map(|w| Vec256(u64x4::from_array(w)))
        .collect()
}

#[derive(serde::Serialize)]
struct ProgressEntry {
    utc_time_secs: u64,
    chain_number: usize,
    iterations: u64,
    current_depth: usize,
    current_nodes: usize,
    proposed_samples_per_sec: f64,
    accepted_samples_per_sec: f64,
    temperature: f64,
}
