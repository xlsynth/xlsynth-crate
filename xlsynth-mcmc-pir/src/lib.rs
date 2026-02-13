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
use serde_json::json;
use std::io::Write as IoWrite;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
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
use xlsynth_g8r::aig::graph_logical_effort::GraphLogicalEffortOptions;
use xlsynth_g8r::aig::graph_logical_effort::analyze_graph_logical_effort;
use xlsynth_g8r::aig_sim::count_toggles;
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
use xlsynth_pir::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type;
use xlsynth_pir::structural_similarity::collect_structural_entries;

pub mod driver_cli;
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
    /// Graph logical-effort worst-case delay, scaled by 1e3 and rounded.
    ///
    /// This is populated when objective=`g8r-le-graph`; otherwise it is `0`.
    pub g8r_le_graph_milli: usize,
    /// Number of interior AIG gate output toggles across a provided input
    /// stimulus sequence.
    ///
    /// This is populated when objective=`g8r-nodes-times-depth-times-toggles`;
    /// otherwise it is `0`.
    pub g8r_gate_output_toggles: usize,
    /// Load-weighted switching activity (`alpha*C`) proxy, scaled by 1e3.
    ///
    /// This is populated for weighted-switching objectives; otherwise it is
    /// `0`.
    pub g8r_weighted_switching_milli: u128,
}

/// Calculates the cost of a PIR function.
///
/// When the objective is g8r-based, this runs the XLS optimizer and gatify
/// pipeline to obtain live gate count and depth. Failures are returned as an
/// error (callers can choose to reject the candidate).
pub fn cost(f: &IrFn, objective: Objective) -> Result<Cost> {
    cost_with_effort_options_and_toggle_stimulus(
        f,
        objective,
        None,
        &count_toggles::WeightedSwitchingOptions::default(),
    )
}

/// Calculates cost and, for toggle-based objectives, evaluates the candidate on
/// a fixed gate-level toggle stimulus.
pub fn cost_with_toggle_stimulus(
    f: &IrFn,
    objective: Objective,
    toggle_stimulus: Option<&[Vec<IrBits>]>,
) -> Result<Cost> {
    cost_with_effort_options_and_toggle_stimulus(
        f,
        objective,
        toggle_stimulus,
        &count_toggles::WeightedSwitchingOptions::default(),
    )
}

/// Calculates cost with explicit load-weighting options.
pub fn cost_with_effort_options_and_toggle_stimulus(
    f: &IrFn,
    objective: Objective,
    toggle_stimulus: Option<&[Vec<IrBits>]>,
    weighted_switching_options: &count_toggles::WeightedSwitchingOptions,
) -> Result<Cost> {
    let pir_nodes = f.nodes.len();

    if objective.needs_toggle_stimulus() && toggle_stimulus.is_none() {
        return Err(anyhow::anyhow!(
            "objective {} requires toggle stimulus",
            objective.value_name()
        ));
    }
    if !objective.needs_toggle_stimulus() && toggle_stimulus.is_some() {
        return Err(anyhow::anyhow!(
            "toggle stimulus provided but objective {} does not use toggles",
            objective.value_name()
        ));
    }
    if objective.needs_weighted_switching() {
        validate_weighted_switching_options(weighted_switching_options)?;
    }

    let (
        g8r_nodes,
        g8r_depth,
        g8r_le_graph_milli,
        g8r_gate_output_toggles,
        g8r_weighted_switching_milli,
    ) = if objective.uses_g8r_costing() {
        compute_g8r_stats_for_pir_fn(
            f,
            objective.needs_graph_logical_effort(),
            objective.needs_gate_output_toggles(),
            objective.needs_weighted_switching(),
            toggle_stimulus,
            weighted_switching_options,
        )?
    } else {
        (pir_nodes, pir_nodes, 0, 0, 0u128)
    };

    Ok(Cost {
        pir_nodes,
        g8r_nodes,
        g8r_depth,
        g8r_le_graph_milli,
        g8r_gate_output_toggles,
        g8r_weighted_switching_milli,
    })
}

fn validate_weighted_switching_options(
    options: &count_toggles::WeightedSwitchingOptions,
) -> Result<()> {
    let checks = [
        ("beta1", options.beta1),
        ("beta2", options.beta2),
        ("primary_output_load", options.primary_output_load),
    ];
    for (name, value) in checks {
        if !value.is_finite() {
            return Err(anyhow::anyhow!(
                "weighted switching option '{}' must be finite; got {}",
                name,
                value
            ));
        }
    }
    Ok(())
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
    #[value(name = "g8r-nodes-times-depth-times-toggles")]
    G8rNodesTimesDepthTimesToggles,
    #[value(
        name = "g8r-le-graph",
        alias = "g8r-graph-le",
        alias = "g8r-graph-logical-effort"
    )]
    G8rLeGraph,
    #[value(name = "g8r-le-graph-times-product")]
    G8rLeGraphTimesProduct,
    #[value(name = "g8r-weighted-switching")]
    G8rWeightedSwitching,
    #[value(name = "g8r-nodes-times-weighted-switching-no-depth-regress")]
    G8rNodesTimesWeightedSwitchingNoDepthRegress,
}

impl Objective {
    pub fn uses_g8r_costing(self) -> bool {
        matches!(
            self,
            Objective::G8rNodes
                | Objective::G8rNodesTimesDepth
                | Objective::G8rNodesTimesDepthTimesToggles
                | Objective::G8rLeGraph
                | Objective::G8rLeGraphTimesProduct
                | Objective::G8rWeightedSwitching
                | Objective::G8rNodesTimesWeightedSwitchingNoDepthRegress
        )
    }

    pub fn needs_graph_logical_effort(self) -> bool {
        matches!(
            self,
            Objective::G8rLeGraph | Objective::G8rLeGraphTimesProduct
        )
    }

    pub fn needs_toggle_stimulus(self) -> bool {
        self.needs_gate_output_toggles() || self.needs_weighted_switching()
    }

    pub fn needs_gate_output_toggles(self) -> bool {
        matches!(self, Objective::G8rNodesTimesDepthTimesToggles)
    }

    pub fn needs_weighted_switching(self) -> bool {
        matches!(
            self,
            Objective::G8rWeightedSwitching
                | Objective::G8rNodesTimesWeightedSwitchingNoDepthRegress
        )
    }

    pub fn enforces_non_regressing_depth(self) -> bool {
        matches!(
            self,
            Objective::G8rNodesTimesWeightedSwitchingNoDepthRegress
        )
    }

    pub fn value_name(self) -> &'static str {
        match self {
            Objective::Nodes => "nodes",
            Objective::G8rNodes => "g8r-nodes",
            Objective::G8rNodesTimesDepth => "g8r-nodes-times-depth",
            Objective::G8rNodesTimesDepthTimesToggles => "g8r-nodes-times-depth-times-toggles",
            Objective::G8rLeGraph => "g8r-le-graph",
            Objective::G8rLeGraphTimesProduct => "g8r-le-graph-times-product",
            Objective::G8rWeightedSwitching => "g8r-weighted-switching",
            Objective::G8rNodesTimesWeightedSwitchingNoDepthRegress => {
                "g8r-nodes-times-weighted-switching-no-depth-regress"
            }
        }
    }

    pub fn metric(self, c: &Cost) -> u128 {
        match self {
            Objective::Nodes => c.pir_nodes as u128,
            Objective::G8rNodes => c.g8r_nodes as u128,
            Objective::G8rNodesTimesDepth => {
                (c.g8r_nodes as u128).saturating_mul(c.g8r_depth as u128)
            }
            Objective::G8rNodesTimesDepthTimesToggles => (c.g8r_nodes as u128)
                .saturating_mul(c.g8r_depth as u128)
                .saturating_mul(c.g8r_gate_output_toggles as u128),
            Objective::G8rLeGraph => c.g8r_le_graph_milli as u128,
            Objective::G8rLeGraphTimesProduct => {
                let product = (c.g8r_nodes as u128).saturating_mul(c.g8r_depth as u128);
                (c.g8r_le_graph_milli as u128).saturating_mul(product)
            }
            Objective::G8rWeightedSwitching => c.g8r_weighted_switching_milli,
            Objective::G8rNodesTimesWeightedSwitchingNoDepthRegress => {
                (c.g8r_nodes as u128).saturating_mul(c.g8r_weighted_switching_milli)
            }
        }
    }
}

/// Computes the g8r node count for the given PIR function by:
///   1) Wrapping it in a one-function IR package.
///   2) Running the XLS optimizer.
///   3) Parsing the optimized IR back into PIR.
///   4) Gatifying the top function into a `GateFn` and counting live nodes.
fn compute_g8r_stats_for_pir_fn(
    f: &IrFn,
    compute_graph_logical_effort: bool,
    compute_gate_output_toggles: bool,
    compute_weighted_switching: bool,
    toggle_stimulus: Option<&[Vec<IrBits>]>,
    weighted_switching_options: &count_toggles::WeightedSwitchingOptions,
) -> Result<(usize, usize, usize, usize, u128)> {
    // The PIR → text → XLS → optimize → PIR → gatify pipeline assumes a DAG.
    // Random rewiring transforms can (transiently) create cycles; if that happens
    // we treat it as a candidate failure and fall back to PIR node count.
    //
    // Note: we intentionally catch panics here because some PIR utilities
    // currently panic on cycle detection.
    let result = catch_unwind(AssertUnwindSafe(|| {
        compute_g8r_stats_for_pir_fn_impl(
            f,
            compute_graph_logical_effort,
            compute_gate_output_toggles,
            compute_weighted_switching,
            toggle_stimulus,
            weighted_switching_options,
        )
    }));
    match result {
        Ok(r) => r,
        Err(_panic) => Err(anyhow::anyhow!(
            "panic during g8r-stats pipeline (likely a cycle)"
        )),
    }
}

fn graph_le_delay_to_milli(delay: f64) -> usize {
    if !delay.is_finite() {
        return usize::MAX;
    }
    if delay <= 0.0 {
        return 0;
    }
    let scaled = delay * 1000.0;
    if scaled >= usize::MAX as f64 {
        usize::MAX
    } else {
        scaled.round() as usize
    }
}

fn compute_g8r_stats_for_pir_fn_impl(
    f: &IrFn,
    compute_graph_logical_effort: bool,
    compute_gate_output_toggles: bool,
    compute_weighted_switching: bool,
    toggle_stimulus: Option<&[Vec<IrBits>]>,
    weighted_switching_options: &count_toggles::WeightedSwitchingOptions,
) -> Result<(usize, usize, usize, usize, u128)> {
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
        enable_rewrite_prio_encode: false,
    };
    let gatify_output = ir2gate::gatify(&top_fn, gatify_options)
        .map_err(|e| anyhow::anyhow!("ir2gate::gatify failed: {}", e))?;
    let gate_fn = gatify_output.gate_fn;
    let stats = get_summary_stats::get_summary_stats(&gate_fn);
    let g8r_le_graph_milli = if compute_graph_logical_effort {
        let graph_le = analyze_graph_logical_effort(
            &gate_fn,
            &GraphLogicalEffortOptions {
                beta1: 1.0,
                beta2: 0.0,
            },
        );
        graph_le_delay_to_milli(graph_le.delay)
    } else {
        0
    };
    let (g8r_gate_output_toggles, g8r_weighted_switching_milli) = if compute_gate_output_toggles
        || compute_weighted_switching
    {
        let batch = toggle_stimulus.ok_or_else(|| {
            anyhow::anyhow!("toggle-based objective requires prepared toggle stimulus")
        })?;
        if batch.len() < 2 {
            return Err(anyhow::anyhow!(
                "toggle stimulus must contain at least two samples; got {}",
                batch.len()
            ));
        }
        let expected_input_count = gate_fn.inputs.len();
        for (sample_idx, sample) in batch.iter().enumerate() {
            if sample.len() != expected_input_count {
                return Err(anyhow::anyhow!(
                    "toggle sample {} has {} inputs, expected {}",
                    sample_idx + 1,
                    sample.len(),
                    expected_input_count
                ));
            }
            for (input_idx, (bits, gate_input)) in
                sample.iter().zip(gate_fn.inputs.iter()).enumerate()
            {
                let expected_width = gate_input.get_bit_count();
                if bits.get_bit_count() != expected_width {
                    return Err(anyhow::anyhow!(
                        "toggle sample {} input {} has width {}, expected {}",
                        sample_idx + 1,
                        input_idx,
                        bits.get_bit_count(),
                        expected_width
                    ));
                }
            }
        }
        let mut gate_output_toggles = 0usize;
        let mut weighted_switching_milli = 0u128;
        if compute_weighted_switching {
            let weighted_stats = count_toggles::count_weighted_switching(
                &gate_fn,
                batch,
                weighted_switching_options,
            );
            weighted_switching_milli = weighted_stats.weighted_switching_milli;
            if compute_gate_output_toggles {
                gate_output_toggles = weighted_stats.gate_output_toggles;
            }
        }
        if compute_gate_output_toggles && !compute_weighted_switching {
            gate_output_toggles = count_toggles::count_toggles(&gate_fn, batch).gate_output_toggles;
        }
        (gate_output_toggles, weighted_switching_milli)
    } else {
        (0, 0u128)
    };
    Ok((
        stats.live_nodes,
        stats.deepest_path,
        g8r_le_graph_milli,
        g8r_gate_output_toggles,
        g8r_weighted_switching_milli,
    ))
}

/// Parses `.irvals`-style stimulus text where each line is one typed tuple
/// value.
pub fn parse_irvals_tuple_lines(irvals_text: &str) -> Result<Vec<IrValue>> {
    let mut values = Vec::new();
    for (lineno, line) in irvals_text.lines().enumerate() {
        let line_no = lineno + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Err(anyhow::anyhow!(
                "empty line {} in stimulus file is not allowed",
                line_no
            ));
        }
        let tuple_val = IrValue::parse_typed(trimmed).map_err(|e| {
            anyhow::anyhow!("failed to parse stimulus tuple at line {}: {}", line_no, e)
        })?;
        tuple_val.get_elements().map_err(|e| {
            anyhow::anyhow!("stimulus line {} is not a tuple value: {}", line_no, e)
        })?;
        values.push(tuple_val);
    }
    Ok(values)
}

/// Reads and parses a `.irvals` stimulus file.
pub fn parse_irvals_tuple_file(path: &Path) -> Result<Vec<IrValue>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {}", path.display(), e))?;
    parse_irvals_tuple_lines(&text)
}

/// Validates tuple-valued stimulus samples against `f`'s parameter signature
/// and lowers each sample to GateFn input vectors (`Vec<IrBits>`).
pub fn lower_toggle_stimulus_for_fn(samples: &[IrValue], f: &IrFn) -> Result<Vec<Vec<IrBits>>> {
    if samples.len() < 2 {
        return Err(anyhow::anyhow!(
            "toggle stimulus must contain at least two samples; got {}",
            samples.len()
        ));
    }

    let mut lowered: Vec<Vec<IrBits>> = Vec::with_capacity(samples.len());
    for (sample_idx, tuple_val) in samples.iter().enumerate() {
        let elems = tuple_val.get_elements().map_err(|e| {
            anyhow::anyhow!("sample {} is not a tuple value: {}", sample_idx + 1, e)
        })?;
        if elems.len() != f.params.len() {
            return Err(anyhow::anyhow!(
                "sample {} tuple arity mismatch: expected {}, got {}",
                sample_idx + 1,
                f.params.len(),
                elems.len()
            ));
        }

        let mut sample_bits = Vec::with_capacity(f.params.len());
        for (param_idx, (elem, param)) in elems.iter().zip(f.params.iter()).enumerate() {
            let mut flat_bits: Vec<bool> = Vec::with_capacity(param.ty.bit_count());
            flatten_ir_value_to_lsb0_bits_for_type(elem, &param.ty, &mut flat_bits).map_err(
                |e| {
                    anyhow::anyhow!(
                        "sample {} param {} ('{}') incompatible with {}: {}",
                        sample_idx + 1,
                        param_idx,
                        param.name,
                        param.ty.to_string(),
                        e
                    )
                },
            )?;
            if flat_bits.len() != param.ty.bit_count() {
                return Err(anyhow::anyhow!(
                    "sample {} param {} ('{}') flattened width mismatch: expected {}, got {}",
                    sample_idx + 1,
                    param_idx,
                    param.name,
                    param.ty.bit_count(),
                    flat_bits.len()
                ));
            }
            sample_bits.push(IrBits::from_lsb_is_0(&flat_bits));
        }
        lowered.push(sample_bits);
    }
    Ok(lowered)
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
    /// Parameters used to convert per-node fanout to load weighting when
    /// computing weighted-switching objectives.
    pub weighted_switching_options: count_toggles::WeightedSwitchingOptions,
    /// When true, and when the crate is built with a formal solver feature
    /// (e.g. `--features with-boolector-built`), run a formal equivalence
    /// oracle after the fast interpreter-based oracle for
    /// non-always-equivalent transforms.
    pub enable_formal_oracle: bool,

    /// Optional directory for writing per-chain trajectory logs as JSONL.
    ///
    /// When set, each chain appends one JSON record per iteration to:
    ///   `trajectory.c{chain_no:03}.jsonl`
    pub trajectory_dir: Option<PathBuf>,

    /// Optional toggle stimulus samples in `.irvals` tuple form (one tuple per
    /// sample) used by toggle-based objectives.
    pub toggle_stimulus: Option<Vec<IrValue>>,
}

/// Message sent from the PIR MCMC engine to an optional checkpoint writer.
///
/// This is used by the `pir-mcmc-driver` binary to keep on-disk best artifacts
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

/// Message sent from the PIR MCMC engine to an optional accepted-sample writer.
///
/// This is used by the `xlsynth-mcmc-pir-sampler` binary to build a
/// deduplicated corpus of accepted equivalent samples.
#[derive(Clone, Debug)]
pub struct AcceptedSampleMsg {
    pub chain_no: usize,
    pub global_iter: u64,
    pub digest: [u8; 32],
    pub cost: Cost,
    pub func: IrFn,
}

fn compute_fn_structural_digest(f: &IrFn) -> Option<[u8; 32]> {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let ret = f
            .ret_node_ref
            .expect("PIR functions must have a return node");
        let (entries, _depths) = collect_structural_entries(f);
        let h = entries[ret.index].hash.as_bytes();
        let mut out = [0u8; 32];
        out.copy_from_slice(h);
        out
    }));
    match result {
        Ok(v) => Some(v),
        Err(_panic) => None,
    }
}

fn canonicalize_fn_for_sample(f: &IrFn) -> Result<IrFn> {
    let mut f = f.clone();
    compact_and_toposort_in_place(&mut f)
        .map_err(|e| anyhow::anyhow!("compact_and_toposort_in_place failed: {}", e))?;
    desugar_extensions::desugar_extensions_in_fn(&mut f)
        .map_err(|e| anyhow::anyhow!("desugar_extensions_in_fn failed: {}", e))?;
    Ok(f)
}

fn iteration_outcome_tag<K>(o: &xlsynth_mcmc::IterationOutcomeDetails<K>) -> &'static str {
    match o {
        xlsynth_mcmc::IterationOutcomeDetails::CandidateFailure => "CandidateFailure",
        xlsynth_mcmc::IterationOutcomeDetails::ApplyFailure => "ApplyFailure",
        xlsynth_mcmc::IterationOutcomeDetails::SimFailure => "SimFailure",
        xlsynth_mcmc::IterationOutcomeDetails::OracleFailure => "OracleFailure",
        xlsynth_mcmc::IterationOutcomeDetails::MetropolisReject => "MetropolisReject",
        xlsynth_mcmc::IterationOutcomeDetails::Accepted { .. } => "Accepted",
    }
}

fn hash_to_hex(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes.iter() {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

struct PirSegmentRunner {
    objective: Objective,
    weighted_switching_options: count_toggles::WeightedSwitchingOptions,
    initial_temperature: f64,
    max_allowed_depth: Option<usize>,
    enable_formal_oracle: bool,
    progress_iters: u64,
    checkpoint_iters: u64,
    checkpoint_tx: Option<Sender<CheckpointMsg>>,
    accepted_sample_tx: Option<Sender<AcceptedSampleMsg>>,
    shared_best: Option<Arc<SharedBest<IrFn>>>,
    baseline_metric: u128,
    trajectory_dir: Option<PathBuf>,
    prepared_toggle_stimulus: Option<Arc<Vec<Vec<IrBits>>>>,
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
    toggle_stimulus: Option<&[Vec<IrBits>]>,
    weighted_switching_options: &count_toggles::WeightedSwitchingOptions,
    max_allowed_depth: Option<usize>,
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
                let new_candidate_cost = match cost_with_effort_options_and_toggle_stimulus(
                    &candidate_fn,
                    objective,
                    toggle_stimulus,
                    weighted_switching_options,
                ) {
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

                if let Some(max_depth) = max_allowed_depth
                    && new_candidate_cost.g8r_depth > max_depth
                {
                    return McmcIterationOutput {
                        output_state: current_fn,
                        output_cost: current_cost,
                        best_updated: false,
                        outcome: IterationOutcomeDetails::MetropolisReject,
                        oracle_time_micros,
                        transform_always_equivalent: chosen_transform.always_equivalent(),
                        transform: Some(current_transform_kind),
                    };
                }

                let curr_metric_u128 = objective.metric(&current_cost);
                let new_metric_u128 = objective.metric(&new_candidate_cost);
                let accept = if new_metric_u128 == curr_metric_u128
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
                        curr_metric_u128 as f64,
                        new_metric_u128 as f64,
                        temp,
                        context.rng,
                    )
                };

                if accept {
                    let best_metric_u128 = objective.metric(best_cost);
                    let new_metric_u128 = objective.metric(&new_candidate_cost);
                    if new_metric_u128 < best_metric_u128 {
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
        let mut trajectory_writer: Option<std::io::BufWriter<std::fs::File>> =
            if let Some(dir) = &self.trajectory_dir {
                std::fs::create_dir_all(dir).map_err(|e| {
                    anyhow::anyhow!("failed to create trajectory dir {}: {}", dir.display(), e)
                })?;
                let path = dir.join(format!("trajectory.c{:03}.jsonl", params.chain_no));
                let f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .map_err(|e| anyhow::anyhow!("failed to open {}: {}", path.display(), e))?;
                Some(std::io::BufWriter::new(f))
            } else {
                None
            };

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

        let toggle_stimulus = self.prepared_toggle_stimulus.as_ref().map(|v| v.as_slice());

        let mut current_fn = start_state.clone();
        let mut current_cost = cost_with_effort_options_and_toggle_stimulus(
            &current_fn,
            self.objective,
            toggle_stimulus,
            &self.weighted_switching_options,
        )
        .map_err(|e| {
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
                toggle_stimulus,
                &self.weighted_switching_options,
                self.max_allowed_depth,
            );

            let mut accepted_digest: Option<[u8; 32]> = None;
            let mut accepted_sample_sent = false;

            if let IterationOutcomeDetails::Accepted { .. } = iteration_output.outcome {
                if let Some(ref tx) = self.accepted_sample_tx {
                    match canonicalize_fn_for_sample(&iteration_output.output_state) {
                        Ok(canon) => match optimize_pir_fn_via_xls(&canon) {
                            Ok(mut opt) => {
                                let _ = compact_and_toposort_in_place(&mut opt);
                                match compute_fn_structural_digest(&opt) {
                                    Some(digest) => {
                                        accepted_digest = Some(digest);
                                        accepted_sample_sent = tx
                                            .send(AcceptedSampleMsg {
                                                chain_no: params.chain_no,
                                                global_iter,
                                                digest,
                                                cost: iteration_output.output_cost,
                                                func: opt,
                                            })
                                            .is_ok();
                                    }
                                    None => {
                                        log::warn!(
                                            "[pir-mcmc] failed to compute structural digest for accepted sample '{}' after XLS optimize (c{:03}:i{:06}); skipping sample emission",
                                            iteration_output.output_state.name,
                                            params.chain_no,
                                            global_iter
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                // The sampler wants uniqueness defined by the XLS-optimized form.
                                // If we cannot obtain it, skip emission rather than falling back
                                // to the pre-optimized state.
                                log::warn!(
                                    "[pir-mcmc] failed to XLS-optimize accepted sample '{}' (c{:03}:i{:06}): {}; skipping sample emission",
                                    iteration_output.output_state.name,
                                    params.chain_no,
                                    global_iter,
                                    e
                                );
                            }
                        },
                        Err(e) => {
                            log::warn!(
                                "[pir-mcmc] failed to canonicalize accepted sample '{}' (c{:03}:i{:06}): {}; skipping sample emission",
                                iteration_output.output_state.name,
                                params.chain_no,
                                global_iter,
                                e
                            );
                        }
                    }
                }
            }

            if let Some(w) = trajectory_writer.as_mut() {
                let metric_u128 = self.objective.metric(&iteration_output.output_cost);
                let rec = json!({
                    "chain_no": params.chain_no,
                    "role": format!("{:?}", params.role),
                    "global_iter": global_iter,
                    "temp": temp,
                    "outcome": iteration_outcome_tag(&iteration_output.outcome),
                    "best_updated": iteration_output.best_updated,
                    "objective": format!("{:?}", self.objective),
                    "metric": metric_u128,
                    "pir_nodes": iteration_output.output_cost.pir_nodes,
                    "g8r_nodes": iteration_output.output_cost.g8r_nodes,
                    "g8r_depth": iteration_output.output_cost.g8r_depth,
                    "g8r_le_graph_milli": iteration_output.output_cost.g8r_le_graph_milli,
                    "g8r_gate_output_toggles": iteration_output.output_cost.g8r_gate_output_toggles,
                    "g8r_weighted_switching_milli": iteration_output.output_cost.g8r_weighted_switching_milli,
                    "oracle_time_micros": iteration_output.oracle_time_micros,
                    "transform": iteration_output.transform.map(|k| format!("{:?}", k)),
                    "transform_always_equivalent": iteration_output.transform_always_equivalent,
                    "accepted_digest": accepted_digest.map(|d| hash_to_hex(&d)),
                    "accepted_sample_sent": accepted_sample_sent,
                });
                // Best-effort: if trajectory logging fails, abort the segment. This should
                // never happen and indicates an infrastructure issue (disk full, permissions,
                // etc.).
                writeln!(w, "{}", rec.to_string())?;
                if global_iter % 1000 == 0 {
                    w.flush()?;
                }
            }

            current_fn = iteration_output.output_state.clone();
            current_cost = iteration_output.output_cost;

            if iteration_output.best_updated {
                if let Some(ref shared_best) = self.shared_best {
                    let before = shared_best.cost.load(Ordering::SeqCst);
                    let metric_u128 = self.objective.metric(&best_cost);
                    let _ = shared_best.try_update(metric_u128, best_fn.clone());
                    let after = shared_best.cost.load(Ordering::SeqCst);
                    if after < before {
                        let improvement_pct = if self.baseline_metric > 0 {
                            (self.baseline_metric.saturating_sub(after) as f64) * 100.0
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
                        .unwrap_or(u128::MAX),
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

        if let Some(mut w) = trajectory_writer {
            w.flush()?;
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
    run_pir_mcmc_with_shared_best(start_fn, options, None, None, None)
}

pub fn run_pir_mcmc_with_shared_best(
    start_fn: IrFn,
    options: RunOptions,
    shared_best: Option<Arc<SharedBest<IrFn>>>,
    checkpoint_tx: Option<Sender<CheckpointMsg>>,
    accepted_sample_tx: Option<Sender<AcceptedSampleMsg>>,
) -> Result<PirMcmcResult> {
    if !options.objective.needs_toggle_stimulus() && options.toggle_stimulus.is_some() {
        return Err(anyhow::anyhow!(
            "toggle stimulus is not valid with objective {}",
            options.objective.value_name()
        ));
    }

    let prepared_toggle_stimulus = if options.objective.needs_toggle_stimulus() {
        let samples = options.toggle_stimulus.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "objective {} requires toggle stimulus",
                options.objective.value_name()
            )
        })?;
        Some(Arc::new(lower_toggle_stimulus_for_fn(samples, &start_fn)?))
    } else {
        None
    };

    let initial_cost = cost_with_effort_options_and_toggle_stimulus(
        &start_fn,
        options.objective,
        prepared_toggle_stimulus.as_ref().map(|v| v.as_slice()),
        &options.weighted_switching_options,
    )?;
    let baseline_metric = options.objective.metric(&initial_cost);
    let runner = PirSegmentRunner {
        objective: options.objective,
        weighted_switching_options: options.weighted_switching_options,
        initial_temperature: options.initial_temperature,
        max_allowed_depth: if options.objective.enforces_non_regressing_depth() {
            Some(initial_cost.g8r_depth)
        } else {
            None
        },
        enable_formal_oracle: options.enable_formal_oracle,
        progress_iters: options.progress_iters,
        checkpoint_iters: options.checkpoint_iters,
        checkpoint_tx,
        accepted_sample_tx,
        shared_best,
        baseline_metric,
        trajectory_dir: options.trajectory_dir.clone(),
        prepared_toggle_stimulus,
    };

    let objective = options.objective;
    let threshold = options.initial_temperature as u128;

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
    use count_toggles::WeightedSwitchingOptions;
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
            weighted_switching_options: WeightedSwitchingOptions::default(),
            enable_formal_oracle: false,
            trajectory_dir: None,
            toggle_stimulus: None,
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

    #[test]
    fn objective_metric_toggles_product_saturates() {
        let c = Cost {
            pir_nodes: 0,
            g8r_nodes: usize::MAX,
            g8r_depth: usize::MAX,
            g8r_le_graph_milli: 0,
            g8r_gate_output_toggles: 2,
            g8r_weighted_switching_milli: 0,
        };
        assert_eq!(
            Objective::G8rNodesTimesDepthTimesToggles.metric(&c),
            u128::MAX
        );
    }

    #[test]
    fn objective_metric_nodes_times_weighted_switching_saturates() {
        let c = Cost {
            pir_nodes: 0,
            g8r_nodes: usize::MAX,
            g8r_depth: 0,
            g8r_le_graph_milli: 0,
            g8r_gate_output_toggles: 0,
            g8r_weighted_switching_milli: u128::MAX,
        };
        assert_eq!(
            Objective::G8rNodesTimesWeightedSwitchingNoDepthRegress.metric(&c),
            u128::MAX
        );
    }

    #[test]
    fn parse_irvals_tuple_lines_accepts_valid_tuples() {
        let text = "(bits[1]:0, bits[1]:1)\n(bits[1]:1, bits[1]:1)\n";
        let got = parse_irvals_tuple_lines(text).unwrap();
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn parse_irvals_tuple_lines_rejects_invalid_or_non_tuple_lines() {
        let bad_parse = parse_irvals_tuple_lines("not_a_value\n").unwrap_err();
        assert!(
            bad_parse.to_string().contains("line 1"),
            "expected line number in parse error"
        );

        let non_tuple = parse_irvals_tuple_lines("bits[1]:1\n").unwrap_err();
        assert!(
            non_tuple.to_string().contains("not a tuple"),
            "expected tuple-specific error"
        );
    }

    #[test]
    fn lower_toggle_stimulus_rejects_arity_and_type_mismatch() {
        let mut parser = ir_parser::Parser::new(
            r#"fn f(a: bits[1] id=1, b: bits[2] id=2) -> bits[1] {
  ret identity.3: bits[1] = identity(a, id=3)
}"#,
        );
        let f = parser.parse_fn().unwrap();

        let arity_bad = vec![
            IrValue::parse_typed("(bits[1]:0, bits[2]:0, bits[1]:0)").unwrap(),
            IrValue::parse_typed("(bits[1]:1, bits[2]:1, bits[1]:1)").unwrap(),
        ];
        assert!(lower_toggle_stimulus_for_fn(&arity_bad, &f).is_err());

        let type_bad = vec![
            IrValue::parse_typed("(bits[1]:0, bits[1]:0)").unwrap(),
            IrValue::parse_typed("(bits[1]:1, bits[1]:1)").unwrap(),
        ];
        assert!(lower_toggle_stimulus_for_fn(&type_bad, &f).is_err());
    }

    #[test]
    fn run_pir_mcmc_rejects_invalid_toggle_stimulus_usage() {
        let mut parser = ir_parser::Parser::new(
            r#"fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(a, b, id=3)
}"#,
        );
        let f = parser.parse_fn().unwrap();

        let opts_missing = RunOptions {
            max_iters: 1,
            threads: 1,
            chain_strategy: ChainStrategy::Independent,
            checkpoint_iters: 1,
            progress_iters: 0,
            seed: 1,
            initial_temperature: 1.0,
            objective: Objective::G8rNodesTimesDepthTimesToggles,
            weighted_switching_options: WeightedSwitchingOptions::default(),
            enable_formal_oracle: false,
            trajectory_dir: None,
            toggle_stimulus: None,
        };
        assert!(run_pir_mcmc(f.clone(), opts_missing).is_err());

        let opts_wrong_objective = RunOptions {
            max_iters: 1,
            threads: 1,
            chain_strategy: ChainStrategy::Independent,
            checkpoint_iters: 1,
            progress_iters: 0,
            seed: 1,
            initial_temperature: 1.0,
            objective: Objective::Nodes,
            weighted_switching_options: WeightedSwitchingOptions::default(),
            enable_formal_oracle: false,
            trajectory_dir: None,
            toggle_stimulus: Some(vec![
                IrValue::parse_typed("(bits[1]:0, bits[1]:0)").unwrap(),
                IrValue::parse_typed("(bits[1]:1, bits[1]:1)").unwrap(),
            ]),
        };
        assert!(run_pir_mcmc(f, opts_wrong_objective).is_err());
    }

    #[test]
    fn cost_with_toggle_objective_populates_toggle_count() {
        let mut parser = ir_parser::Parser::new(
            r#"fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(a, b, id=3)
}"#,
        );
        let f = parser.parse_fn().unwrap();
        let samples = vec![
            IrValue::parse_typed("(bits[1]:0, bits[1]:0)").unwrap(),
            IrValue::parse_typed("(bits[1]:1, bits[1]:1)").unwrap(),
            IrValue::parse_typed("(bits[1]:0, bits[1]:0)").unwrap(),
        ];
        let lowered = lower_toggle_stimulus_for_fn(&samples, &f).unwrap();
        let c = cost_with_toggle_stimulus(
            &f,
            Objective::G8rNodesTimesDepthTimesToggles,
            Some(&lowered),
        )
        .unwrap();
        assert!(
            c.g8r_gate_output_toggles > 0,
            "expected positive interior toggle count"
        );
    }

    #[test]
    fn cost_with_weighted_switching_objective_populates_weighted_metric() {
        let mut parser = ir_parser::Parser::new(
            r#"fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(a, b, id=3)
}"#,
        );
        let f = parser.parse_fn().unwrap();
        let samples = vec![
            IrValue::parse_typed("(bits[1]:0, bits[1]:0)").unwrap(),
            IrValue::parse_typed("(bits[1]:1, bits[1]:1)").unwrap(),
            IrValue::parse_typed("(bits[1]:0, bits[1]:0)").unwrap(),
        ];
        let lowered = lower_toggle_stimulus_for_fn(&samples, &f).unwrap();
        let c = cost_with_effort_options_and_toggle_stimulus(
            &f,
            Objective::G8rWeightedSwitching,
            Some(&lowered),
            &WeightedSwitchingOptions::default(),
        )
        .unwrap();
        assert!(
            c.g8r_weighted_switching_milli > 0,
            "expected positive weighted switching estimate"
        );
    }

    #[test]
    fn cost_with_weighted_switching_rejects_non_finite_options() {
        let mut parser = ir_parser::Parser::new(
            r#"fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(a, b, id=3)
}"#,
        );
        let f = parser.parse_fn().unwrap();
        let samples = vec![
            IrValue::parse_typed("(bits[1]:0, bits[1]:0)").unwrap(),
            IrValue::parse_typed("(bits[1]:1, bits[1]:1)").unwrap(),
        ];
        let lowered = lower_toggle_stimulus_for_fn(&samples, &f).unwrap();

        let err = cost_with_effort_options_and_toggle_stimulus(
            &f,
            Objective::G8rWeightedSwitching,
            Some(&lowered),
            &WeightedSwitchingOptions {
                beta1: f64::NAN,
                beta2: 0.0,
                primary_output_load: 1.0,
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("must be finite"),
            "expected finite-coefficient validation error, got: {}",
            err
        );
    }

    #[test]
    fn trajectory_json_preserves_large_u128_metrics() {
        let rec = serde_json::json!({
            "metric": u128::MAX,
            "g8r_weighted_switching_milli": u128::MAX,
        });
        let s = serde_json::to_string(&rec).unwrap();
        assert!(
            s.contains(&u128::MAX.to_string()),
            "expected full u128 JSON number in serialized output"
        );
    }
}
