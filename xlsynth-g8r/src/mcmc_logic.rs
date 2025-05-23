// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;
use rand::prelude::{Rng, SeedableRng, SliceRandom};
use rand_pcg::Pcg64Mcg;

// Imports from the xlsynth_g8r crate
use crate::check_equivalence::validate_same_gate_fn;
use crate::gate::GateFn; // Cost is now defined in this file
use crate::get_summary_stats;
use crate::ir2gate::{self, GatifyOptions};
use crate::test_utils::{
    load_bf16_add_sample, load_bf16_mul_sample, LoadedSample, Opt as SampleOpt,
};
use crate::transforms::get_all_transforms;
use crate::transforms::transform_trait::{TransformDirection, TransformKind};
use crate::xls_ir::ir_parser;
use clap::ValueEnum;

const INITIAL_TEMPERATURE: f64 = 5.0;
const MIN_TEMPERATURE_RATIO: f64 = 0.00001;
const STATS_PRINT_ITERATION_INTERVAL: u64 = 1000;
const STATS_PRINT_TIME_INTERVAL_SECS: u64 = 1;

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

/// Placeholder for SAT-based equivalence check.
/// Currently uses simulation-based check via validate_same_gate_fn.
pub fn oracle_equiv_sat(lhs: &GateFn, rhs: &GateFn) -> bool {
    validate_same_gate_fn(lhs, rhs).is_ok()
}

/// Holds MCMC iteration statistics.
#[derive(Debug)]
pub struct McmcStats {
    pub accepted_overall: usize,
    pub rejected_apply_fail: usize,
    pub rejected_candidate_fail: usize,
    pub rejected_oracle: usize,
    pub rejected_metro: usize,
    pub oracle_verified: usize,
    pub total_oracle_time_micros: u128,
    pub accepted_edits_by_kind: HashMap<TransformKind, usize>,
}

impl Default for McmcStats {
    fn default() -> Self {
        McmcStats {
            accepted_overall: 0,
            rejected_apply_fail: 0,
            rejected_candidate_fail: 0,
            rejected_oracle: 0,
            rejected_metro: 0,
            oracle_verified: 0,
            total_oracle_time_micros: 0,
            accepted_edits_by_kind: HashMap::new(),
        }
    }
}

/// Context for an MCMC iteration, holding shared resources.
pub struct McmcContext<'a> {
    pub rng: &'a mut Pcg64Mcg,
    pub all_transforms: Vec<Box<dyn crate::transforms::transform_trait::Transform>>,
    pub weights: Vec<f64>,
}

/// Details of what occurred during a single MCMC iteration attempt.
pub enum IterationOutcomeDetails {
    CandidateFailure,
    ApplyFailure,
    OracleFailure,
    MetropolisReject,
    Accepted { kind: TransformKind },
}

/// Output of a single MCMC iteration.
pub struct McmcIterationOutput {
    pub output_gfn: GateFn, // The GateFn to be used as current_gfn for the next iteration
    pub output_cost: Cost,  // Cost of output_gfn
    pub best_gfn_updated: bool,
    pub outcome: IterationOutcomeDetails,
    pub oracle_time_micros: u128, // Time spent in oracle, 0 if not run
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
) -> McmcIterationOutput {
    let mut iteration_best_gfn_updated = false;

    if context.all_transforms.is_empty() {
        // No transforms available to apply
        return McmcIterationOutput {
            output_gfn: current_gfn,
            output_cost: current_cost,
            best_gfn_updated: false,
            outcome: IterationOutcomeDetails::CandidateFailure, /* Or a new outcome? For now,
                                                                 * CandidateFailure */
            oracle_time_micros: 0,
        };
    }

    let dist = WeightedIndex::new(&context.weights).expect("non-empty weights");
    let chosen_transform_idx = dist.sample(context.rng);
    let chosen_transform = &mut context.all_transforms[chosen_transform_idx];
    let current_transform_kind = chosen_transform.kind();

    let direction = if context.rng.gen::<bool>() {
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
            output_gfn: current_gfn,
            output_cost: current_cost,
            best_gfn_updated: false,
            outcome: IterationOutcomeDetails::CandidateFailure,
            oracle_time_micros: 0,
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
            let new_candidate_cost = cost(&candidate_gfn);
            log::trace!("new_candidate_cost: {:?}", new_candidate_cost);
            let mut oracle_time_micros = 0u128;
            let is_equiv = if chosen_transform.always_equivalent() {
                true
            } else {
                let oracle_start_time = Instant::now();
                let res = oracle_equiv_sat(&current_gfn, &candidate_gfn);
                oracle_time_micros = oracle_start_time.elapsed().as_micros();
                res
            };
            log::trace!("is_equiv: {:?}", is_equiv);

            if !is_equiv {
                McmcIterationOutput {
                    output_gfn: current_gfn,
                    output_cost: current_cost,
                    best_gfn_updated: false,
                    outcome: IterationOutcomeDetails::OracleFailure,
                    oracle_time_micros,
                }
            } else {
                let curr_metric = objective.metric(&current_cost) as f64;
                let new_metric = objective.metric(&new_candidate_cost) as f64;
                let better = new_metric < curr_metric;
                let accept_prob = ((curr_metric - new_metric) / temp).exp();
                let metropolis = context.rng.gen::<f64>() < accept_prob;

                if better || metropolis {
                    if new_candidate_cost < *best_cost {
                        *best_gfn = candidate_gfn.clone();
                        *best_cost = new_candidate_cost;
                        iteration_best_gfn_updated = true;
                    }
                    McmcIterationOutput {
                        output_gfn: candidate_gfn,
                        output_cost: new_candidate_cost,
                        best_gfn_updated: iteration_best_gfn_updated,
                        outcome: IterationOutcomeDetails::Accepted {
                            kind: current_transform_kind,
                        },
                        oracle_time_micros,
                    }
                } else {
                    McmcIterationOutput {
                        output_gfn: current_gfn,
                        output_cost: current_cost,
                        best_gfn_updated: false,
                        outcome: IterationOutcomeDetails::MetropolisReject,
                        oracle_time_micros,
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
                output_gfn: current_gfn,
                output_cost: current_cost,
                best_gfn_updated: false,
                outcome: IterationOutcomeDetails::ApplyFailure,
                oracle_time_micros: 0,
            }
        }
    }
}

/// Runs the MCMC optimization process.
pub fn mcmc(
    initial_gfn_param: GateFn,
    secs: u64,
    seed: u64,
    running: Arc<AtomicBool>,
    disabled_transform_names: Vec<String>,
    verbose: bool,
    objective: Objective,
) -> GateFn {
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
        return initial_gfn_param;
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

    let mut current_gfn = initial_gfn_param.clone();
    let mut current_cost = cost(&current_gfn);
    let mut best_gfn = initial_gfn_param;
    let mut best_cost = current_cost;

    let mut stats = McmcStats::default();

    let weights = build_transform_weights(&all_available_transforms, objective);

    let mut mcmc_context = McmcContext {
        rng: &mut iteration_rng,
        all_transforms: all_available_transforms,
        weights,
    };

    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(secs);
    let annealing_duration_secs = secs;

    let mut iterations_count: u64 = 0;
    let mut last_print_time = Instant::now();

    while running.load(Ordering::SeqCst) && Instant::now() < end_time {
        iterations_count += 1;

        let time_since_start_secs = (Instant::now() - start_time).as_secs_f64();
        let current_temp = INITIAL_TEMPERATURE
            * (1.0 - time_since_start_secs / annealing_duration_secs as f64)
                .max(MIN_TEMPERATURE_RATIO);

        // Print action about to be taken if verbose
        if verbose {
            // Print transform, direction, and candidate info
            if !mcmc_context.all_transforms.is_empty() {
                let chosen_transform_idx = mcmc_context
                    .rng
                    .gen_range(0..mcmc_context.all_transforms.len());
                let chosen_transform = &mut mcmc_context.all_transforms[chosen_transform_idx];
                let current_transform_kind = chosen_transform.kind();
                let direction = if mcmc_context.rng.gen::<bool>() {
                    TransformDirection::Forward
                } else {
                    TransformDirection::Backward
                };
                let candidate_locations = chosen_transform.find_candidates(&current_gfn, direction);
                if !candidate_locations.is_empty() {
                    let chosen_location = candidate_locations.choose(mcmc_context.rng).unwrap();
                    println!(
                        "[mcmc][verbose] iter {}: About to apply {:?} ({:?}) at {:?}",
                        iterations_count, current_transform_kind, direction, chosen_location
                    );
                } else {
                    println!(
                        "[mcmc][verbose] iter {}: No candidates for {:?} ({:?})",
                        iterations_count, current_transform_kind, direction
                    );
                }
            }
        }

        log::trace!("Starting MCMC iteration: {:?}", iterations_count);
        let iteration_output = mcmc_iteration(
            current_gfn, // current_gfn is moved in
            current_cost,
            &mut best_gfn,
            &mut best_cost,
            &mut mcmc_context, // Pass context here
            current_temp,
            objective,
        );
        log::trace!("MCMC iteration completed: {:?}", iterations_count);

        current_gfn = iteration_output.output_gfn; // new current_gfn obtained from output
        current_cost = iteration_output.output_cost;
        stats.total_oracle_time_micros += iteration_output.oracle_time_micros;

        match iteration_output.outcome {
            IterationOutcomeDetails::Accepted { kind } => {
                stats.accepted_overall += 1;
                *stats.accepted_edits_by_kind.entry(kind).or_insert(0) += 1;
                if iteration_output.oracle_time_micros > 0 {
                    stats.oracle_verified += 1;
                }
            }
            IterationOutcomeDetails::CandidateFailure => {
                stats.rejected_candidate_fail += 1;
            }
            IterationOutcomeDetails::ApplyFailure => {
                stats.rejected_apply_fail += 1;
            }
            IterationOutcomeDetails::OracleFailure => {
                stats.rejected_oracle += 1;
                // oracle_time_micros > 0 when this outcome occurs
            }
            IterationOutcomeDetails::MetropolisReject => {
                stats.rejected_metro += 1;
                if iteration_output.oracle_time_micros > 0 {
                    stats.oracle_verified += 1;
                }
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

            let mut sorted_edits_vec: Vec<(&TransformKind, &usize)> =
                stats.accepted_edits_by_kind.iter().collect();
            sorted_edits_vec.sort_by_key(|&(k, _)| k);

            let accepted_edits_str = sorted_edits_vec
                .iter()
                .map(|(kind, count)| format!("{}:{}", kind, count))
                .collect::<Vec<String>>()
                .join(", ");

            println!(
                "[mcmc] iter: {} | Best: (n={}, d={}) | Cur: (n={}, d={}) | Temp: {:.2e} | Samples/s: {:.2} | Rejected (AF/CF/O/M): {}/{}/{}/{} | Oracle Ok: {} | Avg Oracle (ms): {:.3} | Accepted: {} ({})         ",
                iterations_count, best_cost.nodes, best_cost.depth, current_cost.nodes, current_cost.depth, current_temp, samples_per_sec,
                stats.rejected_apply_fail, stats.rejected_candidate_fail, stats.rejected_oracle, stats.rejected_metro,
                stats.oracle_verified, avg_oracle_ms,
                stats.accepted_overall, if accepted_edits_str.is_empty() { "-" } else { &accepted_edits_str },
            );
            let _ = std::io::stdout().flush();
            last_print_time = Instant::now();
        }
    }
    best_gfn
}

/// Loads the starting `GateFn` from either a sample or an IR file.
pub fn load_start<P: AsRef<Path>>(p_generic: P) -> Result<GateFn> {
    let p_str = p_generic.as_ref().to_str().unwrap_or_default();

    if p_str.starts_with("sample://") {
        let sample_name = p_str.trim_start_matches("sample://");
        let loaded_sample_res: Result<LoadedSample, anyhow::Error> = match sample_name {
            "bf16_add" => Ok(load_bf16_add_sample(SampleOpt::Yes)),
            "bf16_mul" => Ok(load_bf16_mul_sample(SampleOpt::Yes)),
            _ => Err(anyhow::anyhow!("Unknown sample name: {}", sample_name)),
        };
        let loaded_sample = loaded_sample_res?;
        let sample_cost = cost(&loaded_sample.gate_fn);
        println!(
            "Sample '{}' loaded. Initial stats: nodes={}, depth={}",
            sample_name, sample_cost.nodes, sample_cost.depth
        );
        Ok(loaded_sample.gate_fn)
    } else {
        println!("Loading IR from path: {}", p_str);
        let package = ir_parser::parse_path_to_package(p_generic.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to parse IR package '{}': {:?}", p_str, e))?;

        let top_entity = package
            .get_top()
            .ok_or_else(|| anyhow::anyhow!("No top entity found in IR package '{}'", p_str))?;
        println!("Found top function: {}", top_entity.name);

        let gatify_options = GatifyOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
        };
        let gatify_output = ir2gate::gatify(top_entity, gatify_options)
            .map_err(|e| anyhow::anyhow!("Failed to gatify IR from '{}': {}", p_str, e))?;
        println!("Successfully gatified main function into GateFn.");
        Ok(gatify_output.gate_fn)
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
                RemoveRedundantAnd | RemoveFalseAnd | RemoveTrueAnd | UnduplicateGate => 3.0,
                InsertRedundantAnd | InsertFalseAnd | InsertTrueAnd | DuplicateGate => 0.5,
                _ => 1.0,
            },
            Objective::Depth => match k {
                RotateAndRight | RotateAndLeft => 3.0,
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
