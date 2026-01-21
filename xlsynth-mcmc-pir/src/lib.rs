// SPDX-License-Identifier: Apache-2.0

//! PIR-based MCMC optimization using the shared `xlsynth-mcmc` engine.
//!
//! This crate wires the XLSynth PIR IR (`xlsynth_pir::ir`) into the generic
//! MCMC statistics and Metropolis helpers in `xlsynth-mcmc`.  It provides a
//! small library API (`run_pir_mcmc`) that runs a single-chain MCMC over a
//! single PIR function.

use anyhow::Result;
use rand::Rng;
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;
use rand::prelude::SliceRandom;
use rand_pcg::Pcg64Mcg;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::time::Instant;

use clap::ValueEnum;

use xlsynth::IrBits;
use xlsynth::IrPackage;
use xlsynth::IrValue;
use xlsynth_g8r::aig::get_summary_stats;
use xlsynth_g8r::gatify::ir2gate::{self, GatifyOptions};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_mcmc::Best as SharedBest;
use xlsynth_mcmc::MIN_TEMPERATURE_RATIO;
use xlsynth_mcmc::McmcIterationOutput as SharedMcmcIterationOutput;
use xlsynth_mcmc::McmcOptions as SharedMcmcOptions;
use xlsynth_mcmc::McmcStats as SharedMcmcStats;
use xlsynth_mcmc::metropolis_accept;
use xlsynth_mcmc::multichain::{ChainRole, ChainStrategy, SegmentOutcome, SegmentRunParams};
use xlsynth_mcmc::multichain::{SegmentRunner, run_multichain};
use xlsynth_pir::desugar_extensions;
use xlsynth_pir::fuzz_utils::arbitrary_irbits;
use xlsynth_pir::ir::Fn as IrFn;
use xlsynth_pir::ir::Type as PirType;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_utils::compact_and_toposort_in_place;

pub mod transforms;

use crate::transforms::{
    PirTransform, PirTransformKind, build_transform_weights, get_all_pir_transforms,
};

const DEFAULT_ORACLE_RANDOM_SAMPLES: usize = 32;

// We want invalid-IR candidates (esp. bit_slice bounds) to be visible, since
// they often indicate a bug in a transform. But they can also happen frequently
// during exploration, which can drown out more important warnings. So: warn a
// few times, then downgrade.
static INVALID_BIT_SLICE_WARN_COUNT: AtomicUsize = AtomicUsize::new(0);
const INVALID_BIT_SLICE_WARN_LIMIT: usize = 8;

use xlsynth_prover::prover::prove_ir_fn_equiv;
use xlsynth_prover::prover::types::EquivResult;

/// Simple cost model for PIR MCMC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cost {
    /// Number of PIR nodes in the function.
    pub pir_nodes: usize,
    /// Number of gates in the corresponding `GateFn` after running the XLS
    /// optimizer and gatifying.
    pub g8r_nodes: usize,
    /// Depth of the corresponding `GateFn` (deepest path) after running the XLS
    /// optimizer and gatifying.
    pub g8r_depth: usize,
}

/// Calculates the cost of a PIR function.
///
/// When the objective is g8r-based, this runs the XLS optimizer and gatify
/// pipeline to obtain live gate count and depth. Failures are returned as an
/// error (callers can choose to reject the candidate).
pub fn cost(f: &IrFn, objective: Objective) -> Result<Cost> {
    let pir_nodes = f.nodes.len();

    let (g8r_nodes, g8r_depth) = if matches!(
        objective,
        Objective::G8rNodes | Objective::G8rNodesTimesDepth
    ) {
        compute_g8r_stats_for_pir_fn(f)?
    } else {
        (pir_nodes, pir_nodes)
    };

    Ok(Cost {
        pir_nodes,
        g8r_nodes,
        g8r_depth,
    })
}

/// Produces the XLS-optimized PIR function for `f`.
///
/// This uses the same PIR → text → XLS optimize → PIR parsing pipeline used by
/// g8r-costing. We prefer storing "best" candidates in this optimized form so
/// emitted artifacts (`best.ir`) reflect the canonical optimized IR rather than
/// an intermediate exploration state.
fn optimize_pir_fn_via_xls(f: &IrFn) -> Result<IrFn> {
    // The pipeline assumes the IR is a DAG and that textual IR references only
    // previously-defined names. MCMC exploration can transiently violate that;
    // callers can choose how to handle errors.
    let mut fn_for_text = f.clone();
    compact_and_toposort_in_place(&mut fn_for_text)
        .map_err(|e| anyhow::anyhow!("compact_and_toposort_in_place failed: {}", e))?;

    // Upstream XLS IR does not understand PIR extension ops; desugar them before
    // round-tripping through the XLS optimizer.
    desugar_extensions::desugar_extensions_in_fn(&mut fn_for_text)
        .map_err(|e| anyhow::anyhow!("desugar_extensions_in_fn failed: {}", e))?;

    let pkg_text = format!("package pir_mcmc\n\n{}\n", fn_for_text);

    let ir_pkg = IrPackage::parse_ir(&pkg_text, None)
        .map_err(|e| anyhow::anyhow!("IrPackage::parse_ir failed: {:?}", e))?;
    let optimized_ir_pkg = xlsynth::optimize_ir(&ir_pkg, &f.name)
        .map_err(|e| anyhow::anyhow!("optimize_ir failed: {:?}", e))?;
    let optimized_ir_text = optimized_ir_pkg.to_string();

    let mut parser = ir_parser::Parser::new(&optimized_ir_text);
    let pir_pkg = parser
        .parse_and_validate_package()
        .map_err(|e| anyhow::anyhow!("PIR parse_and_validate_package failed: {:?}", e))?;
    let top_fn = pir_pkg
        .get_top_fn()
        .ok_or_else(|| anyhow::anyhow!("No top function found in optimized PIR package"))?;
    Ok(top_fn.clone())
}

/// Objective used to evaluate cost improvements for PIR MCMC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Objective {
    Nodes,
    G8rNodes,
    G8rNodesTimesDepth,
}

impl Objective {
    pub fn metric(self, c: &Cost) -> u64 {
        match self {
            Objective::Nodes => c.pir_nodes as u64,
            Objective::G8rNodes => c.g8r_nodes as u64,
            Objective::G8rNodesTimesDepth => {
                (c.g8r_nodes as u64).saturating_mul(c.g8r_depth as u64)
            }
        }
    }
}

/// Computes the g8r node count for the given PIR function by:
///   1) Wrapping it in a one-function IR package.
///   2) Running the XLS optimizer.
///   3) Parsing the optimized IR back into PIR.
///   4) Gatifying the top function into a `GateFn` and counting live nodes.
fn compute_g8r_stats_for_pir_fn(f: &IrFn) -> Result<(usize, usize)> {
    // The PIR → text → XLS → optimize → PIR → gatify pipeline assumes a DAG.
    // Random rewiring transforms can (transiently) create cycles; if that happens
    // we treat it as a candidate failure and fall back to PIR node count.
    //
    // Note: we intentionally catch panics here because some PIR utilities
    // currently panic on cycle detection.
    let result = catch_unwind(AssertUnwindSafe(|| compute_g8r_stats_for_pir_fn_impl(f)));
    match result {
        Ok(r) => r,
        Err(_panic) => Err(anyhow::anyhow!(
            "panic during g8r-stats pipeline (likely a cycle)"
        )),
    }
}

fn compute_g8r_stats_for_pir_fn_impl(f: &IrFn) -> Result<(usize, usize)> {
    // 1-3) Optimize the PIR function via the XLS pipeline.
    let top_fn = optimize_pir_fn_via_xls(f)?;

    // 4) Gatify and measure live gate count.
    let gatify_options = GatifyOptions {
        fold: true,
        hash: true,
        check_equivalence: false,
        adder_mapping: AdderMapping::default(),
        mul_adder_mapping: None,
        range_info: None,
        enable_rewrite_carry_out: false,
    };
    let gatify_output = ir2gate::gatify(&top_fn, gatify_options)
        .map_err(|e| anyhow::anyhow!("ir2gate::gatify failed: {}", e))?;
    let gate_fn = gatify_output.gate_fn;
    let stats = get_summary_stats::get_summary_stats(&gate_fn);
    Ok((stats.live_nodes, stats.deepest_path))
}

/// Type aliases specializing the generic MCMC helpers from `xlsynth-mcmc` to
/// the PIR world.
pub type McmcStats = SharedMcmcStats<PirTransformKind>;
pub type Best = SharedBest<IrFn>;
pub type IterationOutcomeDetails = xlsynth_mcmc::IterationOutcomeDetails<PirTransformKind>;
pub type McmcIterationOutput = SharedMcmcIterationOutput<IrFn, Cost, PirTransformKind>;
pub type McmcOptions = SharedMcmcOptions;

/// Context for a PIR MCMC iteration, holding shared resources.
pub struct PirMcmcContext<'a> {
    pub rng: &'a mut Pcg64Mcg,
    pub all_transforms: Vec<Box<dyn PirTransform>>,
    pub weights: Vec<f64>,
    pub enable_formal_oracle: bool,
}

/// Options controlling a PIR MCMC run.
#[derive(Clone, Debug)]
pub struct RunOptions {
    /// Maximum number of iterations to perform.
    pub max_iters: u64,
    /// Number of parallel chains to run.
    pub threads: u64,
    /// Strategy for running multiple chains.
    pub chain_strategy: ChainStrategy,
    /// Segment size (iterations) for explore/exploit synchronization.
    ///
    /// Only used when `chain_strategy=ExploreExploit`.
    pub checkpoint_iters: u64,
    /// Progress logging interval in iterations (0 disables progress logs).
    pub progress_iters: u64,
    /// RNG seed for the Markov chain.
    pub seed: u64,
    /// Initial temperature for MCMC.
    pub initial_temperature: f64,
    /// Objective to optimize.
    pub objective: Objective,
    /// When true, and when the crate is built with a formal solver feature
    /// (e.g. `--features with-boolector-built`), run a formal equivalence
    /// oracle after the fast interpreter-based oracle for
    /// non-always-equivalent transforms.
    pub enable_formal_oracle: bool,
}

/// Message sent from the PIR MCMC engine to an optional checkpoint writer.
///
/// This is used by the `xlsynth-mcmc-pir` binary to keep on-disk best artifacts
/// up-to-date during long runs and (optionally) to snapshot the improvement
/// trajectory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointKind {
    /// A periodic checkpoint tick (e.g. every N iterations).
    Periodic,
    /// A new global best was found.
    GlobalBestUpdate,
}

/// A checkpoint writer notification, including the chain and iteration that
/// triggered the event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointMsg {
    pub chain_no: usize,
    pub global_iter: u64,
    pub kind: CheckpointKind,
}

/// Result of a PIR MCMC run.
pub struct PirMcmcResult {
    pub best_fn: IrFn,
    pub best_cost: Cost,
    pub stats: McmcStats,
}

struct PirSegmentRunner {
    objective: Objective,
    initial_temperature: f64,
    enable_formal_oracle: bool,
    progress_iters: u64,
    checkpoint_iters: u64,
    checkpoint_tx: Option<Sender<CheckpointMsg>>,
    shared_best: Option<Arc<SharedBest<IrFn>>>,
    baseline_metric: usize,
}

/// Performs a single iteration of the PIR MCMC process.
pub fn mcmc_iteration(
    current_fn: IrFn, /* Takes ownership, becomes the basis for candidate or returned if no
                       * change */
    current_cost: Cost,
    best_fn: &mut IrFn,   // Mutated if new best is found
    best_cost: &mut Cost, // Mutated if new best is found
    context: &mut PirMcmcContext,
    temp: f64,
    objective: Objective,
) -> McmcIterationOutput {
    let mut iteration_best_updated = false;

    if context.all_transforms.is_empty() {
        // No transforms available to apply.
        return McmcIterationOutput {
            output_state: current_fn,
            output_cost: current_cost,
            best_updated: false,
            outcome: IterationOutcomeDetails::CandidateFailure,
            oracle_time_micros: 0,
            transform_always_equivalent: true,
            transform: None,
        };
    }

    let dist = WeightedIndex::new(&context.weights).expect("non-empty weights");
    let chosen_transform_idx = dist.sample(context.rng);
    let chosen_transform = &mut context.all_transforms[chosen_transform_idx];
    let current_transform_kind = chosen_transform.kind();

    let candidate_locations = chosen_transform.find_candidates(&current_fn);

    log::trace!(
        "Found {} PIR candidates for {:?}",
        candidate_locations.len(),
        current_transform_kind,
    );

    if candidate_locations.is_empty() {
        return McmcIterationOutput {
            output_state: current_fn,
            output_cost: current_cost,
            best_updated: false,
            outcome: IterationOutcomeDetails::CandidateFailure,
            oracle_time_micros: 0,
            transform_always_equivalent: true,
            transform: Some(current_transform_kind),
        };
    }

    let chosen_location = candidate_locations.choose(context.rng).unwrap();

    log::trace!("Chosen PIR location: {:?}", chosen_location);

    let mut candidate_fn = current_fn.clone();

    log::trace!(
        "Applying PIR transform {:?} at {:?}",
        current_transform_kind,
        chosen_location
    );

    match chosen_transform.apply(&mut candidate_fn, chosen_location) {
        Ok(()) => {
            log::trace!("PIR transform applied successfully; determining cost...");
            let (is_equiv, oracle_time_micros) = if chosen_transform.always_equivalent() {
                // Always-equivalent transforms can skip equivalence checks.
                (true, 0u128)
            } else {
                let oracle_start = Instant::now();
                // For transforms that are not guaranteed to preserve semantics, we run a
                // lightweight equivalence oracle:
                // - Deterministic corner cases (all-zeros, all-ones)
                // - A small number of randomized samples (seeded by the run's RNG)
                //
                // If evaluation fails (e.g. due to a cycle or unsupported node kinds),
                // we treat that as non-equivalence for this candidate and reject it.
                let ok = pir_equiv_oracle(
                    &current_fn,
                    &candidate_fn,
                    context.rng,
                    DEFAULT_ORACLE_RANDOM_SAMPLES,
                    context.enable_formal_oracle,
                );
                let micros = oracle_start.elapsed().as_micros();
                (ok, micros)
            };

            if !is_equiv {
                McmcIterationOutput {
                    output_state: current_fn,
                    output_cost: current_cost,
                    best_updated: false,
                    outcome: IterationOutcomeDetails::OracleFailure,
                    oracle_time_micros,
                    transform_always_equivalent: chosen_transform.always_equivalent(),
                    transform: Some(current_transform_kind),
                }
            } else {
                let cost_start = Instant::now();
                let new_candidate_cost = match cost(&candidate_fn, objective) {
                    Ok(c) => c,
                    Err(e) => {
                        let sim_micros = cost_start.elapsed().as_micros();
                        let msg = e.to_string();
                        let is_invalid_bit_slice = msg.contains("Expected operand 0 of bit_slice")
                            || msg.contains("invalid bit_slice");

                        if is_invalid_bit_slice {
                            // Not a sample failure: we sometimes propose structurally invalid
                            // candidates (e.g. bit_slice bounds violations) while exploring.
                            // These are rejected.
                            //
                            // Still, keep this loud for a bit to catch regressions in transforms.
                            let n = INVALID_BIT_SLICE_WARN_COUNT
                                .fetch_add(1, Ordering::Relaxed)
                                .saturating_add(1);
                            if n <= INVALID_BIT_SLICE_WARN_LIMIT {
                                log::warn!(
                                    "[pir-mcmc] cost evaluation failed for '{}' under {:?}: {}; rejecting candidate (invalid bit_slice; warning {}/{})",
                                    candidate_fn.name,
                                    objective,
                                    e,
                                    n,
                                    INVALID_BIT_SLICE_WARN_LIMIT
                                );
                            } else {
                                log::debug!(
                                    "[pir-mcmc] cost evaluation failed for '{}' under {:?}: {}; rejecting candidate (invalid bit_slice; further occurrences silenced to debug)",
                                    candidate_fn.name,
                                    objective,
                                    e
                                );
                            }
                        } else {
                            log::warn!(
                                "[pir-mcmc] cost evaluation failed for '{}' under {:?}: {}; rejecting candidate",
                                candidate_fn.name,
                                objective,
                                e
                            );
                        }
                        return McmcIterationOutput {
                            output_state: current_fn,
                            output_cost: current_cost,
                            best_updated: false,
                            outcome: IterationOutcomeDetails::SimFailure,
                            oracle_time_micros: sim_micros,
                            transform_always_equivalent: chosen_transform.always_equivalent(),
                            transform: Some(current_transform_kind),
                        };
                    }
                };

                let curr_metric_u64 = objective.metric(&current_cost);
                let new_metric_u64 = objective.metric(&new_candidate_cost);
                let accept = if new_metric_u64 == curr_metric_u64
                    && new_candidate_cost.pir_nodes > current_cost.pir_nodes
                {
                    // Equal objective metric but PIR nodes grew: only accept if
                    // the temperature still allows it.
                    metropolis_accept(
                        current_cost.pir_nodes as f64,
                        new_candidate_cost.pir_nodes as f64,
                        temp,
                        context.rng,
                    )
                } else {
                    metropolis_accept(
                        curr_metric_u64 as f64,
                        new_metric_u64 as f64,
                        temp,
                        context.rng,
                    )
                };

                if accept {
                    let best_metric_u64 = objective.metric(best_cost);
                    let new_metric_u64 = objective.metric(&new_candidate_cost);
                    if new_metric_u64 < best_metric_u64 {
                        // When storing a new global best, prefer the optimized IR form so
                        // artifacts (and subsequent segments via shared best) are based on
                        // the canonical optimized representation, not the raw exploration
                        // state.
                        *best_fn = match optimize_pir_fn_via_xls(&candidate_fn) {
                            Ok(opt) => opt,
                            Err(e) => {
                                log::warn!(
                                    "[pir-mcmc] failed to optimize new best candidate '{}': {}; storing unoptimized function",
                                    candidate_fn.name,
                                    e
                                );
                                candidate_fn.clone()
                            }
                        };
                        *best_cost = new_candidate_cost;
                        iteration_best_updated = true;
                    }
                    McmcIterationOutput {
                        output_state: candidate_fn,
                        output_cost: new_candidate_cost,
                        best_updated: iteration_best_updated,
                        outcome: IterationOutcomeDetails::Accepted {
                            kind: current_transform_kind,
                        },
                        oracle_time_micros,
                        transform_always_equivalent: chosen_transform.always_equivalent(),
                        transform: Some(current_transform_kind),
                    }
                } else {
                    McmcIterationOutput {
                        output_state: current_fn,
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
                "Error applying PIR transform {:?}: {:?}",
                current_transform_kind,
                e
            );
            McmcIterationOutput {
                output_state: current_fn,
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

fn make_all_zeros_value(ty: &PirType) -> IrValue {
    match ty {
        PirType::Token => IrValue::make_token(),
        PirType::Bits(width) => {
            if *width == 0 {
                IrValue::from_bits(&IrBits::make_ubits(0, 0).unwrap())
            } else {
                IrValue::from_bits(&IrBits::make_ubits(*width, 0).unwrap())
            }
        }
        PirType::Tuple(elem_types) => {
            let elems: Vec<IrValue> = elem_types.iter().map(|t| make_all_zeros_value(t)).collect();
            IrValue::make_tuple(&elems)
        }
        PirType::Array(arr) => {
            let mut elems: Vec<IrValue> = Vec::with_capacity(arr.element_count);
            for _ in 0..arr.element_count {
                elems.push(make_all_zeros_value(&arr.element_type));
            }
            IrValue::make_array(&elems).expect("array elements must be same-typed")
        }
    }
}

fn make_all_ones_value(ty: &PirType) -> IrValue {
    match ty {
        PirType::Token => IrValue::make_token(),
        PirType::Bits(width) => {
            if *width == 0 {
                IrValue::from_bits(&IrBits::make_ubits(0, 0).unwrap())
            } else if *width <= 64 {
                let mask = if *width == 64 {
                    u64::MAX
                } else {
                    (1u64 << *width) - 1
                };
                IrValue::from_bits(&IrBits::make_ubits(*width, mask).unwrap())
            } else {
                // Build bits by parsing a typed value string via IrBits helper.
                let ones: Vec<bool> = vec![true; *width];
                IrValue::from_bits(&IrBits::from_lsb_is_0(&ones))
            }
        }
        PirType::Tuple(elem_types) => {
            let elems: Vec<IrValue> = elem_types.iter().map(|t| make_all_ones_value(t)).collect();
            IrValue::make_tuple(&elems)
        }
        PirType::Array(arr) => {
            let mut elems: Vec<IrValue> = Vec::with_capacity(arr.element_count);
            for _ in 0..arr.element_count {
                elems.push(make_all_ones_value(&arr.element_type));
            }
            IrValue::make_array(&elems).expect("array elements must be same-typed")
        }
    }
}

fn arbitrary_value_for_type<R: Rng>(rng: &mut R, ty: &PirType) -> IrValue {
    match ty {
        PirType::Token => IrValue::make_token(),
        PirType::Bits(width) => {
            let bits = arbitrary_irbits(rng, *width);
            IrValue::from_bits(&bits)
        }
        PirType::Tuple(elem_types) => {
            let elems: Vec<IrValue> = elem_types
                .iter()
                .map(|t| arbitrary_value_for_type(rng, t))
                .collect();
            IrValue::make_tuple(&elems)
        }
        PirType::Array(arr) => {
            let mut elems: Vec<IrValue> = Vec::with_capacity(arr.element_count);
            for _ in 0..arr.element_count {
                elems.push(arbitrary_value_for_type(rng, &arr.element_type));
            }
            IrValue::make_array(&elems).expect("array elements must be same-typed")
        }
    }
}

fn eval_fn_safe(f: &IrFn, args: &[IrValue]) -> Result<IrValue, ()> {
    // Note: `xlsynth_pir::ir_eval` uses internal `expect` / `unwrap` paths for
    // invariants; rewiring transforms may temporarily violate those (cycles,
    // missing package context for invoke, etc.). We treat any such failure as a
    // rejection signal for the candidate, not a crash.
    let result = catch_unwind(AssertUnwindSafe(|| eval_fn(f, args)));
    match result {
        Ok(FnEvalResult::Success(s)) => Ok(s.value),
        Ok(FnEvalResult::Failure(_f)) => Err(()),
        Err(_panic) => Err(()),
    }
}

fn pir_equiv_oracle<R: Rng>(
    lhs: &IrFn,
    rhs: &IrFn,
    rng: &mut R,
    random_samples: usize,
    enable_formal_oracle: bool,
) -> bool {
    if lhs.params.len() != rhs.params.len() || lhs.ret_ty != rhs.ret_ty {
        return false;
    }
    for (lp, rp) in lhs.params.iter().zip(rhs.params.iter()) {
        if lp.ty != rp.ty {
            return false;
        }
    }

    // Deterministic corner cases first: all-zeros and all-ones.
    let zeros_args: Vec<IrValue> = lhs
        .params
        .iter()
        .map(|p| make_all_zeros_value(&p.ty))
        .collect();
    let ones_args: Vec<IrValue> = lhs
        .params
        .iter()
        .map(|p| make_all_ones_value(&p.ty))
        .collect();
    for args in [&zeros_args, &ones_args] {
        let l = eval_fn_safe(lhs, args);
        let r = eval_fn_safe(rhs, args);
        match (l, r) {
            (Ok(lv), Ok(rv)) => {
                if lv != rv {
                    return false;
                }
            }
            _ => return false,
        }
    }

    // Randomized sampling.
    for _ in 0..random_samples {
        let args: Vec<IrValue> = lhs
            .params
            .iter()
            .map(|p| arbitrary_value_for_type(rng, &p.ty))
            .collect();
        let l = eval_fn_safe(lhs, &args);
        let r = eval_fn_safe(rhs, &args);
        match (l, r) {
            (Ok(lv), Ok(rv)) => {
                if lv != rv {
                    return false;
                }
            }
            _ => return false,
        }
    }

    if enable_formal_oracle {
        {
            match prove_ir_fn_equiv(lhs, rhs) {
                EquivResult::Proved => true,
                EquivResult::Disproved { .. } | EquivResult::ToolchainDisproved(_) => false,
                EquivResult::Error(msg) => {
                    log::warn!(
                        "[pir-mcmc] formal oracle error for '{}' vs '{}': {}; rejecting candidate",
                        lhs.name,
                        rhs.name,
                        msg
                    );
                    false
                }
            }
        }
    } else {
        true
    }
}

impl SegmentRunner<IrFn, Cost, PirTransformKind> for PirSegmentRunner {
    type Error = anyhow::Error;

    fn run_segment(
        &self,
        start_state: IrFn,
        params: SegmentRunParams,
    ) -> Result<SegmentOutcome<IrFn, Cost, PirTransformKind>, Self::Error> {
        let mut iteration_rng = Pcg64Mcg::seed_from_u64(params.seed);
        let mut all_transforms = get_all_pir_transforms();
        if !self.enable_formal_oracle {
            // Safety-by-default: non-always-equivalent transforms are only enabled when a
            // formal equivalence oracle is enabled. This prevents accidentally accepting
            // incorrect rewrites due to sampling-only equivalence.
            all_transforms.retain(|t| t.always_equivalent());
        }
        let weights = build_transform_weights(&all_transforms);

        let mut context = PirMcmcContext {
            rng: &mut iteration_rng,
            all_transforms,
            weights,
            enable_formal_oracle: self.enable_formal_oracle,
        };

        let mut current_fn = start_state.clone();
        let mut current_cost = cost(&current_fn, self.objective).map_err(|e| {
            anyhow::anyhow!(
                "failed to evaluate initial cost for '{}' under {:?}: {}",
                current_fn.name,
                self.objective,
                e
            )
        })?;
        let mut best_fn = start_state;
        let mut best_cost = current_cost;
        let mut stats = McmcStats::default();

        let seg_start_time = Instant::now();

        let mut iterations_count: u64 = 0;
        while iterations_count < params.segment_iters {
            iterations_count += 1;
            let global_iter = params.iter_offset + iterations_count;

            let temp = match params.role {
                ChainRole::Explorer => self.initial_temperature * 10.0,
                ChainRole::Exploit => {
                    let progress_ratio = if params.total_iters > 0 {
                        (global_iter as f64) / (params.total_iters as f64)
                    } else {
                        0.0
                    };
                    let progress_ratio = progress_ratio.min(1.0);
                    self.initial_temperature * (1.0 - progress_ratio).max(MIN_TEMPERATURE_RATIO)
                }
            };

            let iteration_output = mcmc_iteration(
                current_fn,
                current_cost,
                &mut best_fn,
                &mut best_cost,
                &mut context,
                temp,
                self.objective,
            );

            current_fn = iteration_output.output_state.clone();
            current_cost = iteration_output.output_cost;

            if iteration_output.best_updated {
                if let Some(ref shared_best) = self.shared_best {
                    let before = shared_best.cost.load(Ordering::SeqCst);
                    let metric_u64 = self.objective.metric(&best_cost);
                    let metric_usize = usize::try_from(metric_u64).unwrap_or(usize::MAX);
                    let _ = shared_best.try_update(metric_usize, best_fn.clone());
                    let after = shared_best.cost.load(Ordering::SeqCst);
                    if after < before {
                        let improvement_pct = if self.baseline_metric > 0 {
                            ((self.baseline_metric as f64) - (after as f64)) * 100.0
                                / (self.baseline_metric as f64)
                        } else {
                            0.0
                        };
                        log::info!(
                            "[pir-mcmc] GLOBAL BEST UPDATE c{:03}:i{:06} | metric {} -> {} | improvement={:+.2}%",
                            params.chain_no,
                            global_iter,
                            before,
                            after,
                            improvement_pct,
                        );
                        // Best-effort: if a checkpoint writer is active, trigger an
                        // immediate update so the monotone global best is visible on disk.
                        if let Some(ref tx) = self.checkpoint_tx {
                            let _ = tx.send(CheckpointMsg {
                                chain_no: params.chain_no,
                                global_iter,
                                kind: CheckpointKind::GlobalBestUpdate,
                            });
                        }
                    }
                }
            }

            if self.checkpoint_iters > 0 && global_iter % self.checkpoint_iters == 0 {
                if let Some(ref tx) = self.checkpoint_tx {
                    // Best-effort: if the receiver is gone, stop sending.
                    let _ = tx.send(CheckpointMsg {
                        chain_no: params.chain_no,
                        global_iter,
                        kind: CheckpointKind::Periodic,
                    });
                }
            }

            stats.update_for_iteration(&iteration_output, /* paranoid= */ false, global_iter);

            if self.progress_iters > 0
                && (global_iter % self.progress_iters == 0
                    || global_iter == params.total_iters
                    || iterations_count == params.segment_iters)
            {
                let elapsed_secs = seg_start_time.elapsed().as_secs_f64();
                let samples_per_sec = if elapsed_secs > 0.0 {
                    iterations_count as f64 / elapsed_secs
                } else {
                    0.0
                };
                log::info!(
                    "PIR MCMC c{:03}:i{:06} | GBestM={} | LBest (pir={}, g8r_n={}, g8r_d={}, m={}) | Cur (pir={}, g8r_n={}, g8r_d={}, m={}) | Temp={:.2e} | Samples/s={:.2}",
                    params.chain_no,
                    global_iter,
                    self.shared_best
                        .as_ref()
                        .map(|b| b.cost.load(Ordering::SeqCst))
                        .unwrap_or(usize::MAX),
                    best_cost.pir_nodes,
                    best_cost.g8r_nodes,
                    best_cost.g8r_depth,
                    self.objective.metric(&best_cost),
                    current_cost.pir_nodes,
                    current_cost.g8r_nodes,
                    current_cost.g8r_depth,
                    self.objective.metric(&current_cost),
                    temp,
                    samples_per_sec,
                );
            }
        }

        Ok(SegmentOutcome {
            end_state: current_fn,
            end_cost: current_cost,
            best_state: best_fn,
            best_cost,
            stats,
        })
    }
}

/// Runs a single-chain MCMC optimization over a PIR function.
///
/// This function is deterministic for fixed `start_fn` and `options`.
pub fn run_pir_mcmc(start_fn: IrFn, options: RunOptions) -> Result<PirMcmcResult> {
    run_pir_mcmc_with_shared_best(start_fn, options, None, None)
}

pub fn run_pir_mcmc_with_shared_best(
    start_fn: IrFn,
    options: RunOptions,
    shared_best: Option<Arc<SharedBest<IrFn>>>,
    checkpoint_tx: Option<Sender<CheckpointMsg>>,
) -> Result<PirMcmcResult> {
    let initial_cost = cost(&start_fn, options.objective)?;
    let initial_metric_u64 = options.objective.metric(&initial_cost);
    let baseline_metric = usize::try_from(initial_metric_u64).unwrap_or(usize::MAX);
    let runner = PirSegmentRunner {
        objective: options.objective,
        initial_temperature: options.initial_temperature,
        enable_formal_oracle: options.enable_formal_oracle,
        progress_iters: options.progress_iters,
        checkpoint_iters: options.checkpoint_iters,
        checkpoint_tx,
        shared_best,
        baseline_metric,
    };

    let objective = options.objective;
    let threshold = options.initial_temperature as u64;

    let (best_fn, best_cost, stats) = run_multichain(
        start_fn,
        options.max_iters,
        options.seed,
        options.threads.max(1) as usize,
        options.chain_strategy,
        options.checkpoint_iters,
        Arc::new(runner),
        move |c: &Cost| objective.metric(c),
        |f: &IrFn| f.to_string(),
        move |cur_cost: &Cost, global_best_cost: &Cost| {
            objective.metric(cur_cost)
                > objective.metric(global_best_cost).saturating_add(threshold)
        },
    )?;

    Ok(PirMcmcResult {
        best_fn,
        best_cost,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;
    use xlsynth_pir::ir_utils::remap_payload_with;

    #[test]
    fn pir_mcmc_runs_and_is_deterministic_on_simple_add() {
        let ir_text = r#"fn add(x: bits[8] id=10, y: bits[8] id=20) -> bits[8] {
  ret add.42: bits[8] = add(x, y, id=42)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_fn = parser.parse_fn().unwrap();

        let opts = RunOptions {
            max_iters: 10,
            threads: 1,
            chain_strategy: ChainStrategy::Independent,
            checkpoint_iters: 5000,
            progress_iters: 0,
            seed: 1,
            initial_temperature: 5.0,
            objective: Objective::Nodes,
            enable_formal_oracle: false,
        };

        let res1 = run_pir_mcmc(ir_fn.clone(), opts.clone()).unwrap();
        let res2 = run_pir_mcmc(ir_fn.clone(), opts).unwrap();

        assert_eq!(res1.best_cost.pir_nodes, ir_fn.nodes.len());
        assert_eq!(res2.best_cost.pir_nodes, ir_fn.nodes.len());

        // Determinism: for fixed seed and options, we should get the same best
        // function text.
        assert_eq!(res1.best_fn.to_string(), res2.best_fn.to_string());
    }

    #[test]
    fn pir_equiv_oracle_rejects_obviously_non_equivalent_rewire() {
        // Build a tiny function with a literal so the rewire can substitute a
        // same-typed node but change semantics.
        let ir_text = r#"fn add_lit(x: bits[8] id=10, y: bits[8] id=20) -> bits[8] {
  literal.30: bits[8] = literal(value=0, id=30)
  ret add.42: bits[8] = add(x, y, id=42)
}"#;
        let mut parser1 = ir_parser::Parser::new(ir_text);
        let orig_fn = parser1.parse_fn().unwrap();
        let mut parser2 = ir_parser::Parser::new(ir_text);
        let mut rewired_fn = parser2.parse_fn().unwrap();

        // Rewire the RHS operand of add.42 from y to literal.30 (same type).
        // This should change semantics for the all-ones test vector.
        let mut add_ref = None;
        let mut lit_ref = None;
        for nr in rewired_fn.node_refs() {
            let node = rewired_fn.get_node(nr);
            match &node.payload {
                xlsynth_pir::ir::NodePayload::Literal(_) => {
                    lit_ref = Some(nr);
                }
                xlsynth_pir::ir::NodePayload::Binop(xlsynth_pir::ir::Binop::Add, _, _) => {
                    add_ref = Some(nr);
                }
                _ => {}
            }
        }
        let add_ref = add_ref.expect("expected add node");
        let lit_ref = lit_ref.expect("expected literal node");

        let old_add_payload = rewired_fn.get_node(add_ref).payload.clone();
        rewired_fn.get_node_mut(add_ref).payload = remap_payload_with(
            &old_add_payload,
            |(slot, dep)| {
                if slot == 1 { lit_ref } else { dep }
            },
        );

        let mut rng = Pcg64Mcg::seed_from_u64(1);
        assert!(!pir_equiv_oracle(
            &orig_fn,
            &rewired_fn,
            &mut rng,
            4,
            /* enable_formal_oracle= */ false,
        ));
    }
}
