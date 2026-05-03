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
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::io::Write as IoWrite;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
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
use xlsynth_mcmc::MIN_TEMPERATURE_RATIO;
use xlsynth_mcmc::McmcIterationOutput as SharedMcmcIterationOutput;
use xlsynth_mcmc::McmcOptions as SharedMcmcOptions;
use xlsynth_mcmc::McmcStats as SharedMcmcStats;
use xlsynth_mcmc::metropolis_accept;
use xlsynth_mcmc::multichain::{ChainRole, ChainStrategy, SegmentOutcome, SegmentRunParams};
use xlsynth_mcmc::multichain::{SegmentRunner, run_multichain};
use xlsynth_pir::desugar_extensions::{self, ExtensionEmitMode};
use xlsynth_pir::fuzz_utils::arbitrary_irbits;
use xlsynth_pir::ir::FileTable as PirFileTable;
use xlsynth_pir::ir::Fn as IrFn;
use xlsynth_pir::ir::Package as PirPackage;
use xlsynth_pir::ir::PackageMember as PirPackageMember;
use xlsynth_pir::ir::Param as PirParam;
use xlsynth_pir::ir::Type as PirType;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_assuming_node_index_topological};
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
    /// This is populated for graph-logical-effort objectives; otherwise it is
    /// `0`.
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

/// How PIR extension ops are projected before XLS optimization and g8r costing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ExtensionCostingMode {
    /// Preserve extension ops through XLS optimization using FFI wrappers, then
    /// reconstruct extension ops when reparsing the optimized IR.
    #[value(name = "preserve")]
    Preserve,
    /// Desugar extension ops to ordinary XLS IR before optimization, so costs
    /// and best artifacts are grounded in standard non-extension IR.
    #[value(name = "desugar")]
    Desugar,
}

impl ExtensionCostingMode {
    pub fn value_name(self) -> &'static str {
        match self {
            ExtensionCostingMode::Preserve => "preserve",
            ExtensionCostingMode::Desugar => "desugar",
        }
    }

    fn from_value_name(value: &str) -> Result<Self> {
        match value {
            "preserve" => Ok(ExtensionCostingMode::Preserve),
            "desugar" => Ok(ExtensionCostingMode::Desugar),
            _ => Err(anyhow::anyhow!(
                "unknown extension costing mode in artifact: {}",
                value
            )),
        }
    }

    fn extension_emit_mode(self) -> ExtensionEmitMode {
        match self {
            ExtensionCostingMode::Preserve => ExtensionEmitMode::AsFfiFunction,
            ExtensionCostingMode::Desugar => ExtensionEmitMode::Desugared,
        }
    }
}

/// Optional hard caps applied to gate-level cost components during PIR MCMC.
///
/// At most one cap may be active in a run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConstraintLimits {
    pub max_delay: Option<usize>,
    pub max_area: Option<usize>,
}

/// Detailed violation information for an infeasible candidate.
///
/// At most one cap is active in any given run, so at most one of these fields
/// is populated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstraintViolationScore {
    pub delay_over: Option<usize>,
    pub area_over: Option<usize>,
}

/// Ordered score used for selecting best-so-far states under optional caps.
///
/// `violation=None` means the candidate is feasible. Feasible candidates always
/// beat infeasible ones; among infeasible candidates we minimize raw overage
/// under the single active cap first, then fall back to the objective only as a
/// final tiebreak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchScore {
    pub objective: u128,
    pub violation: Option<ConstraintViolationScore>,
}

impl SearchScore {
    /// Returns true when the candidate satisfies all active hard caps.
    pub fn feasible(self) -> bool {
        self.violation.is_none()
    }
}

impl Ord for SearchScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.violation, other.violation) {
            (None, None) => self.objective.cmp(&other.objective),
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            (Some(lhs), Some(rhs)) => lhs
                .delay_over
                .cmp(&rhs.delay_over)
                .then_with(|| lhs.area_over.cmp(&rhs.area_over))
                .then_with(|| self.objective.cmp(&other.objective)),
        }
    }
}

impl PartialOrd for SearchScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone)]
struct BestState {
    score: SearchScore,
    value: IrFn,
}

/// Shared best-so-far PIR function using structured feasibility-first scoring.
pub struct Best {
    inner: Mutex<BestState>,
}

impl Best {
    pub fn new(initial_score: SearchScore, value: IrFn) -> Self {
        Self {
            inner: Mutex::new(BestState {
                score: initial_score,
                value,
            }),
        }
    }

    pub fn try_update(&self, new_score: SearchScore, new_value: IrFn) -> bool {
        let mut guard = self.inner.lock().unwrap();
        if new_score < guard.score {
            *guard = BestState {
                score: new_score,
                value: new_value,
            };
            true
        } else {
            false
        }
    }

    pub fn get(&self) -> IrFn {
        self.inner.lock().unwrap().value.clone()
    }

    pub fn score(&self) -> SearchScore {
        self.inner.lock().unwrap().score
    }
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
    cost_with_effort_options_toggle_stimulus_and_extension_mode(
        f,
        objective,
        toggle_stimulus,
        weighted_switching_options,
        ExtensionCostingMode::Preserve,
    )
}

/// Calculates cost with explicit load-weighting and extension projection
/// options.
pub fn cost_with_effort_options_toggle_stimulus_and_extension_mode(
    f: &IrFn,
    objective: Objective,
    toggle_stimulus: Option<&[Vec<IrBits>]>,
    weighted_switching_options: &count_toggles::WeightedSwitchingOptions,
    extension_costing_mode: ExtensionCostingMode,
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
            extension_costing_mode,
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

/// Optimizes a PIR package through XLS using the selected extension projection.
pub(crate) fn optimize_pir_package_via_xls_with_extension_mode(
    pkg: &PirPackage,
    top: &str,
    extension_costing_mode: ExtensionCostingMode,
) -> Result<PirPackage> {
    let wrapped_ir_text = desugar_extensions::emit_package_with_extension_mode(
        pkg,
        extension_costing_mode.extension_emit_mode(),
    )
    .map_err(|e| anyhow::anyhow!("emit_package_with_extension_mode failed: {}", e))?;

    let ir_pkg = IrPackage::parse_ir(&wrapped_ir_text, None)
        .map_err(|e| anyhow::anyhow!("IrPackage::parse_ir failed: {:?}", e))?;
    let optimized_ir_pkg = xlsynth::optimize_ir(&ir_pkg, top)
        .map_err(|e| anyhow::anyhow!("optimize_ir failed: {:?}", e))?;
    let optimized_ir_text = optimized_ir_pkg.to_string();

    let mut parser = ir_parser::Parser::new(&optimized_ir_text);
    parser
        .parse_and_validate_package()
        .map_err(|e| anyhow::anyhow!("PIR parse_and_validate_package failed: {:?}", e))
}

/// Produces the XLS-optimized PIR function for `f` using the selected extension
/// projection mode.
pub(crate) fn optimize_pir_fn_via_xls_with_extension_mode(
    f: &IrFn,
    extension_costing_mode: ExtensionCostingMode,
) -> Result<IrFn> {
    // The pipeline assumes the IR is a DAG and that textual IR references only
    // previously-defined names. MCMC exploration can transiently violate that;
    // callers can choose how to handle errors.
    let mut fn_for_text = f.clone();
    compact_and_toposort_in_place(&mut fn_for_text)
        .map_err(|e| anyhow::anyhow!("compact_and_toposort_in_place failed: {}", e))?;

    let mut pir_pkg = PirPackage {
        name: "pir_mcmc".to_string(),
        file_table: PirFileTable::new(),
        members: vec![PirPackageMember::Function(fn_for_text)],
        top: None,
    };
    pir_pkg
        .set_top_fn(&f.name)
        .map_err(|e| anyhow::anyhow!("set_top_fn failed: {}", e))?;

    let pir_pkg = optimize_pir_package_via_xls_with_extension_mode(
        &pir_pkg,
        &f.name,
        extension_costing_mode,
    )?;
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
    #[value(name = "g8r-le-graph-times-nodes")]
    G8rLeGraphTimesNodes,
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
                | Objective::G8rLeGraphTimesNodes
                | Objective::G8rLeGraphTimesProduct
                | Objective::G8rWeightedSwitching
                | Objective::G8rNodesTimesWeightedSwitchingNoDepthRegress
        )
    }

    pub fn needs_graph_logical_effort(self) -> bool {
        matches!(
            self,
            Objective::G8rLeGraph
                | Objective::G8rLeGraphTimesNodes
                | Objective::G8rLeGraphTimesProduct
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
            Objective::G8rLeGraphTimesNodes => "g8r-le-graph-times-nodes",
            Objective::G8rLeGraphTimesProduct => "g8r-le-graph-times-product",
            Objective::G8rWeightedSwitching => "g8r-weighted-switching",
            Objective::G8rNodesTimesWeightedSwitchingNoDepthRegress => {
                "g8r-nodes-times-weighted-switching-no-depth-regress"
            }
        }
    }

    fn from_value_name(value: &str) -> Result<Self> {
        match value {
            "nodes" => Ok(Objective::Nodes),
            "g8r-nodes" => Ok(Objective::G8rNodes),
            "g8r-nodes-times-depth" => Ok(Objective::G8rNodesTimesDepth),
            "g8r-nodes-times-depth-times-toggles" => Ok(Objective::G8rNodesTimesDepthTimesToggles),
            "g8r-le-graph" => Ok(Objective::G8rLeGraph),
            "g8r-le-graph-times-nodes" => Ok(Objective::G8rLeGraphTimesNodes),
            "g8r-le-graph-times-product" => Ok(Objective::G8rLeGraphTimesProduct),
            "g8r-weighted-switching" => Ok(Objective::G8rWeightedSwitching),
            "g8r-nodes-times-weighted-switching-no-depth-regress" => {
                Ok(Objective::G8rNodesTimesWeightedSwitchingNoDepthRegress)
            }
            _ => Err(anyhow::anyhow!("unknown objective in artifact: {}", value)),
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
            Objective::G8rLeGraphTimesNodes => {
                (c.g8r_le_graph_milli as u128).saturating_mul(c.g8r_nodes as u128)
            }
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

pub(crate) fn validate_constraint_configuration(
    objective: Objective,
    limits: ConstraintLimits,
) -> Result<()> {
    if limits.max_delay.is_some() && limits.max_area.is_some() {
        return Err(anyhow::anyhow!(
            "at most one of --max-delay and --max-area may be specified"
        ));
    }
    if !objective.uses_g8r_costing() && (limits.max_delay.is_some() || limits.max_area.is_some()) {
        return Err(anyhow::anyhow!(
            "area/delay caps require a g8r-based objective; got {}",
            objective.value_name()
        ));
    }
    if objective.enforces_non_regressing_depth() && limits.max_area.is_some() {
        return Err(anyhow::anyhow!(
            "--max-area is not compatible with objective {} because it already enforces a non-regressing depth cap",
            objective.value_name()
        ));
    }
    Ok(())
}

/// Computes the active constraint violation details for the given cost.
pub fn constraint_violation(
    c: &Cost,
    limits: ConstraintLimits,
) -> Option<ConstraintViolationScore> {
    let delay_over = limits
        .max_delay
        .map(|max_delay| c.g8r_depth.saturating_sub(max_delay))
        .filter(|over| *over > 0);
    let area_over = limits
        .max_area
        .map(|max_area| c.g8r_nodes.saturating_sub(max_area))
        .filter(|over| *over > 0);

    if delay_over.is_none() && area_over.is_none() {
        return None;
    }

    Some(ConstraintViolationScore {
        delay_over,
        area_over,
    })
}

/// Computes the ordered search score for best-state tracking and multichain
/// synchronization under optional area/delay caps.
pub fn search_score(c: &Cost, objective: Objective, limits: ConstraintLimits) -> SearchScore {
    SearchScore {
        objective: objective.metric(c),
        violation: constraint_violation(c, limits),
    }
}

pub(crate) fn effective_constraint_limits(
    objective: Objective,
    user_limits: ConstraintLimits,
    initial_cost: &Cost,
) -> ConstraintLimits {
    let limits = ConstraintLimits {
        max_delay: match (
            user_limits.max_delay,
            objective.enforces_non_regressing_depth(),
        ) {
            (Some(user_cap), true) => Some(user_cap.min(initial_cost.g8r_depth)),
            (Some(user_cap), false) => Some(user_cap),
            (None, true) => Some(initial_cost.g8r_depth),
            (None, false) => None,
        },
        max_area: user_limits.max_area,
    };
    debug_assert!(
        limits.max_delay.is_none() || limits.max_area.is_none(),
        "effective constraints must keep at most one active cap"
    );
    limits
}

fn repair_energy(v: ConstraintViolationScore) -> u128 {
    match (v.delay_over, v.area_over) {
        (Some(over), None) => over as u128,
        (None, Some(over)) => over as u128,
        (Some(_), Some(_)) => unreachable!("constraint configuration validation rejects dual caps"),
        (None, None) => 0,
    }
}

pub(crate) fn format_search_score(score: SearchScore) -> String {
    match score.violation {
        None => format!("feasible(obj={})", score.objective),
        Some(v) => format!(
            "infeasible(obj={}, delay_over={:?}, area_over={:?})",
            score.objective, v.delay_over, v.area_over,
        ),
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
    extension_costing_mode: ExtensionCostingMode,
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
            extension_costing_mode,
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
    extension_costing_mode: ExtensionCostingMode,
) -> Result<(usize, usize, usize, usize, u128)> {
    // 1-3) Optimize the PIR function via the XLS pipeline.
    let top_fn = optimize_pir_fn_via_xls_with_extension_mode(f, extension_costing_mode)?;

    // 4) Gatify and measure live gate count.
    let gatify_options = GatifyOptions {
        fold: true,
        hash: true,
        check_equivalence: false,
        adder_mapping: AdderMapping::default(),
        array_index_lowering_strategy: Default::default(),
        mul_adder_mapping: None,
        range_info: None,
        enable_rewrite_carry_out: false,
        enable_rewrite_prio_encode: false,
        enable_rewrite_nary_add: false,
        enable_rewrite_mask_low: false,
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
pub type IterationOutcomeDetails = xlsynth_mcmc::IterationOutcomeDetails<PirTransformKind>;
pub type McmcIterationOutput = SharedMcmcIterationOutput<IrFn, Cost, PirTransformKind>;
pub type McmcOptions = SharedMcmcOptions;

/// Cached oracle inputs and expected baseline `eval_fn` results.
///
/// MCMC oracle checks compare many candidate functions against the same
/// accepted baseline. This cache stores the deterministic/random sample
/// arguments and the baseline return value for each sample so candidates only
/// need to evaluate their rewritten graph.
#[derive(Default)]
pub struct EvalFnBaselineResults {
    samples: Vec<Vec<IrValue>>,
    expected_values: Vec<Result<IrValue, ()>>,
    random_samples: usize,
    param_types: Vec<PirType>,
}

impl EvalFnBaselineResults {
    fn clear(&mut self) {
        self.samples.clear();
        self.expected_values.clear();
        self.random_samples = 0;
        self.param_types.clear();
    }

    fn matches_signature(&self, baseline: &IrFn, random_samples: usize) -> bool {
        self.random_samples == random_samples
            && self.param_types.len() == baseline.params.len()
            && self
                .param_types
                .iter()
                .zip(baseline.params.iter())
                .all(|(cached_ty, param)| cached_ty == &param.ty)
    }

    fn populate_from_baseline<R: Rng>(
        &mut self,
        baseline: &IrFn,
        rng: &mut R,
        random_samples: usize,
    ) -> Result<()> {
        self.clear();
        self.random_samples = random_samples;
        self.param_types = baseline.params.iter().map(|p| p.ty.clone()).collect();

        // Deterministic corner cases first: all-zeros and all-ones.
        self.samples.push(make_oracle_args(
            &baseline.params,
            "all-zeros",
            make_all_zeros_value,
        )?);
        self.samples.push(make_oracle_args(
            &baseline.params,
            "all-ones",
            make_all_ones_value,
        )?);

        for _ in 0..random_samples {
            self.samples
                .push(make_oracle_args(&baseline.params, "random", |ty| {
                    arbitrary_value_for_type(rng, ty)
                })?);
        }

        self.expected_values = self
            .samples
            .iter()
            .map(|args| eval_fn_safe(baseline, args))
            .collect();
        Ok(())
    }

    fn ensure_populated<R: Rng>(
        &mut self,
        baseline_if_empty: &IrFn,
        rng: &mut R,
        random_samples: usize,
    ) -> Result<()> {
        if !self.matches_signature(baseline_if_empty, random_samples) || self.samples.is_empty() {
            self.populate_from_baseline(baseline_if_empty, rng, random_samples)?;
        }
        Ok(())
    }
}

/// Context for a PIR MCMC iteration, holding shared resources.
pub struct PirMcmcContext<'a> {
    pub rng: &'a mut Pcg64Mcg,
    pub all_transforms: Vec<Box<dyn PirTransform>>,
    pub weights: Vec<f64>,
    pub enable_formal_oracle: bool,
    pub oracle_baseline_cache: EvalFnBaselineResults,
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
    /// How extension ops are projected before XLS optimization and g8r costing.
    pub extension_costing_mode: ExtensionCostingMode,
    /// Optional hard cap on gate depth (`g8r_depth`) for g8r-based objectives.
    pub max_allowed_depth: Option<usize>,
    /// Optional hard cap on gate count (`g8r_nodes`) for g8r-based objectives.
    pub max_allowed_area: Option<usize>,
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

/// One accepted rewrite on the exact lineage that led to an MCMC winner.
#[derive(Clone, Debug)]
pub struct AcceptedLineageStep {
    /// One-based count of accepted rewrites from the origin to this state.
    pub accepted_rewrite_index: usize,
    /// One-based global MCMC iteration that accepted this rewrite.
    pub global_iter: u64,
    /// Transform accepted at this step.
    pub transform_kind: PirTransformKind,
    /// Raw accepted PIR state after the rewrite.
    pub state: IrFn,
    /// Cost of `state` under the run's objective semantics.
    pub cost: Cost,
}

/// In-memory single-chain artifact for minimizing a discovered MCMC witness.
#[derive(Clone, Debug)]
pub struct PirMcmcArtifact {
    /// Canonicalized function used as the exact rollout origin.
    pub origin_fn: IrFn,
    /// Cost of `origin_fn`.
    pub origin_cost: Cost,
    /// Options used to produce this artifact.
    pub run_options: RunOptions,
    /// Raw accepted state that produced the final strict best update.
    pub raw_winner_fn: IrFn,
    /// Cost of `raw_winner_fn`.
    pub raw_winner_cost: Cost,
    /// Exact accepted rewrite sequence from `origin_fn` to `raw_winner_fn`.
    pub winning_lineage: Vec<AcceptedLineageStep>,
}

/// Options for reducing a winning lineage to an earliest useful prefix.
#[derive(Clone, Copy, Debug)]
pub struct PirMcmcPrefixMinimizeOptions {
    /// Fraction of the discovered objective improvement that must be retained.
    pub retained_win_fraction: f64,
}

/// Result of reducing a winning lineage to an earliest useful prefix.
#[derive(Clone, Debug)]
pub struct PirMcmcPrefixMinimizeResult {
    /// Earliest lineage-prefix state satisfying the requested retained win.
    pub witness_fn: IrFn,
    /// Cost of `witness_fn`.
    pub witness_cost: Cost,
    /// Number of accepted rewrites needed to reach `witness_fn`.
    pub accepted_rewrite_count: usize,
    /// Number of accepted rewrites in the original winning lineage.
    pub original_winning_lineage_len: usize,
    /// Fraction requested by the caller.
    pub requested_retained_win_fraction: f64,
    /// Fraction actually retained by the selected witness.
    pub actual_retained_win_fraction: f64,
    /// Origin objective metric.
    pub origin_metric: u128,
    /// Final winner objective metric.
    pub winner_metric: u128,
    /// Selected witness objective metric.
    pub witness_metric: u128,
}

/// Options for witness-guided short-witness frontier search.
#[derive(Clone, Copy, Debug)]
pub struct PirMcmcBudgetFrontierOptions {
    /// Requested budget spacing; budgets are `step, 2*step, ... <= max`.
    pub budget_step: usize,
    /// Largest accepted-rewrite budget to evaluate.
    pub max_rewrites: usize,
    /// Number of independent short rollouts attempted per requested budget.
    pub rollouts_per_budget: usize,
    /// Search seed. Use the source artifact's seed by default.
    pub seed: u64,
    /// Extra proposal weight per winning-lineage occurrence of a transform
    /// kind.
    pub witness_kind_boost: f64,
    /// Proposal-attempt cap per accepted rewrite in each rollout.
    pub proposal_attempts_per_rewrite: usize,
}

impl PirMcmcBudgetFrontierOptions {
    pub const DEFAULT_WITNESS_KIND_BOOST: f64 = 4.0;
    pub const DEFAULT_PROPOSAL_ATTEMPTS_PER_REWRITE: usize = 64;
}

/// One witness on a frontier, either searched or historical-prefix baseline.
#[derive(Clone, Debug)]
pub struct PirMcmcBudgetWitness {
    pub witness_fn: IrFn,
    pub witness_cost: Cost,
    pub accepted_rewrite_count: usize,
    pub metric: u128,
    pub absolute_win: u128,
    pub win_percent_vs_origin: f64,
    pub retained_win_fraction: f64,
}

/// One requested short-witness budget point.
#[derive(Clone, Debug)]
pub struct PirMcmcBudgetFrontierPoint {
    pub rewrite_budget: usize,
    pub guided: PirMcmcBudgetWitness,
    pub prefix_baseline: PirMcmcBudgetWitness,
}

/// Result of witness-guided short-witness frontier search.
#[derive(Clone, Debug)]
pub struct PirMcmcBudgetFrontierResult {
    pub origin_metric: u128,
    pub winner_metric: u128,
    pub original_winning_lineage_len: usize,
    pub points: Vec<PirMcmcBudgetFrontierPoint>,
}

struct PirMcmcArtifactRunOutput {
    result: PirMcmcResult,
    artifact: PirMcmcArtifact,
}

const PIR_MCMC_ARTIFACT_DIR_NAME: &str = "winning-lineage";
const PIR_MCMC_ARTIFACT_MANIFEST_FILE: &str = "manifest.json";
const PIR_MCMC_ARTIFACT_STATES_DIR_NAME: &str = "states";
const PIR_MCMC_ARTIFACT_SCHEMA_VERSION: u32 = 1;

/// Durable PIR MCMC artifact loaded from a run directory.
pub struct LoadedPirMcmcArtifact {
    pub artifact: PirMcmcArtifact,
    pub package_template: PirPackage,
    pub top_fn_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedPirMcmcArtifactManifest {
    schema_version: u32,
    top_fn_name: String,
    run_options: PersistedRunOptions,
    origin: PersistedArtifactState,
    raw_winner: PersistedArtifactState,
    winning_lineage: Vec<PersistedAcceptedLineageStep>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedRunOptions {
    max_iters: u64,
    threads: u64,
    chain_strategy: String,
    checkpoint_iters: u64,
    progress_iters: u64,
    seed: u64,
    initial_temperature: f64,
    objective: String,
    extension_costing_mode: String,
    max_allowed_depth: Option<usize>,
    max_allowed_area: Option<usize>,
    switching_beta1: f64,
    switching_beta2: f64,
    switching_primary_output_load: f64,
    enable_formal_oracle: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedArtifactState {
    file: String,
    cost: Cost,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedAcceptedLineageStep {
    accepted_rewrite_index: usize,
    global_iter: u64,
    transform_kind: PirTransformKind,
    state: PersistedArtifactState,
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

#[derive(Default)]
struct WinningLineageTracker {
    current_lineage: Vec<AcceptedLineageStep>,
    winning_lineage: Vec<AcceptedLineageStep>,
}

impl WinningLineageTracker {
    /// Records one accepted rewrite and snapshots the full path when it
    /// improves the best-so-far result.
    fn record_accepted_step(
        &mut self,
        global_iter: u64,
        transform_kind: PirTransformKind,
        state: IrFn,
        cost: Cost,
        best_updated: bool,
    ) {
        let accepted_rewrite_index = self.current_lineage.len() + 1;
        self.current_lineage.push(AcceptedLineageStep {
            accepted_rewrite_index,
            global_iter,
            transform_kind,
            state,
            cost,
        });
        if best_updated {
            self.winning_lineage = self.current_lineage.clone();
        }
    }
}

struct PirSegmentRunner {
    objective: Objective,
    extension_costing_mode: ExtensionCostingMode,
    weighted_switching_options: count_toggles::WeightedSwitchingOptions,
    initial_temperature: f64,
    constraints: ConstraintLimits,
    enable_formal_oracle: bool,
    progress_iters: u64,
    checkpoint_iters: u64,
    checkpoint_tx: Option<Sender<CheckpointMsg>>,
    accepted_sample_tx: Option<Sender<AcceptedSampleMsg>>,
    shared_best: Option<Arc<Best>>,
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
    best_score: &mut SearchScore,
    context: &mut PirMcmcContext,
    temp: f64,
    objective: Objective,
    extension_costing_mode: ExtensionCostingMode,
    toggle_stimulus: Option<&[Vec<IrBits>]>,
    weighted_switching_options: &count_toggles::WeightedSwitchingOptions,
    constraints: ConstraintLimits,
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

    let mut candidates = chosen_transform.find_candidates(&current_fn);
    if !context.enable_formal_oracle {
        candidates.retain(|c| c.always_equivalent);
    }

    log::trace!(
        "Found {} PIR candidates for {:?}",
        candidates.len(),
        current_transform_kind,
    );

    if candidates.is_empty() {
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

    let chosen_candidate = candidates.choose(context.rng).unwrap();

    log::trace!("Chosen PIR candidate: {:?}", chosen_candidate);

    let mut candidate_fn = current_fn.clone();

    log::trace!(
        "Applying PIR transform {:?} at {:?}",
        current_transform_kind,
        chosen_candidate.location
    );

    match chosen_transform.apply(&mut candidate_fn, &chosen_candidate.location) {
        Ok(()) => {
            // Transform implementations may emit acyclic def-after-use graphs.
            // Canonicalize centrally here so the oracle and cost paths see the
            // same normalized IR without each transform paying a local
            // topo-sort/compaction cost.
            if let Err(e) = compact_and_toposort_in_place(&mut candidate_fn) {
                log::debug!(
                    "[pir-mcmc] compact/toposort failed for '{}' after {:?} at {:?}: {}; \
                     rejecting candidate",
                    candidate_fn.name,
                    current_transform_kind,
                    chosen_candidate.location,
                    e
                );
                return McmcIterationOutput {
                    output_state: current_fn,
                    output_cost: current_cost,
                    best_updated: false,
                    outcome: IterationOutcomeDetails::CandidateFailure,
                    oracle_time_micros: 0,
                    transform_always_equivalent: chosen_candidate.always_equivalent,
                    transform: Some(current_transform_kind),
                };
            }

            log::trace!("PIR transform applied successfully; determining cost...");
            let (is_equiv, oracle_time_micros) = if chosen_candidate.always_equivalent {
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
                    &mut context.oracle_baseline_cache,
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
                    transform_always_equivalent: chosen_candidate.always_equivalent,
                    transform: Some(current_transform_kind),
                }
            } else {
                let cost_start = Instant::now();
                let new_candidate_cost =
                    match cost_with_effort_options_toggle_stimulus_and_extension_mode(
                        &candidate_fn,
                        objective,
                        toggle_stimulus,
                        weighted_switching_options,
                        extension_costing_mode,
                    ) {
                        Ok(c) => c,
                        Err(e) => {
                            let sim_micros = cost_start.elapsed().as_micros();
                            let msg = e.to_string();
                            let is_invalid_bit_slice = msg
                                .contains("Expected operand 0 of bit_slice")
                                || msg.contains("invalid bit_slice");

                            if is_invalid_bit_slice {
                                // Not a sample failure: we sometimes propose structurally invalid
                                // candidates (e.g. bit_slice bounds violations) while exploring.
                                // These are rejected.
                                //
                                // Still, keep this loud for a bit to catch regressions in
                                // transforms.
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
                                transform_always_equivalent: chosen_candidate.always_equivalent,
                                transform: Some(current_transform_kind),
                            };
                        }
                    };

                let current_score = search_score(&current_cost, objective, constraints);
                let new_score = search_score(&new_candidate_cost, objective, constraints);
                let curr_metric_u128 = objective.metric(&current_cost);
                let new_metric_u128 = objective.metric(&new_candidate_cost);
                let accept = match (current_score.violation, new_score.violation) {
                    (Some(_), None) => true,
                    (None, Some(_)) => false,
                    (Some(curr_violation), Some(new_violation)) => metropolis_accept(
                        repair_energy(curr_violation) as f64,
                        repair_energy(new_violation) as f64,
                        temp,
                        context.rng,
                    ),
                    (None, None) => {
                        if new_metric_u128 == curr_metric_u128
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
                        }
                    }
                };

                if accept {
                    if new_score < *best_score {
                        // When storing a new global best, prefer the optimized IR form so
                        // artifacts (and subsequent segments via shared best) are based on
                        // the canonical optimized representation, not the raw exploration
                        // state.
                        *best_fn = match optimize_pir_fn_via_xls_with_extension_mode(
                            &candidate_fn,
                            extension_costing_mode,
                        ) {
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
                        *best_score = new_score;
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
                        transform_always_equivalent: chosen_candidate.always_equivalent,
                        transform: Some(current_transform_kind),
                    }
                } else {
                    McmcIterationOutput {
                        output_state: current_fn,
                        output_cost: current_cost,
                        best_updated: false,
                        outcome: IterationOutcomeDetails::MetropolisReject,
                        oracle_time_micros,
                        transform_always_equivalent: chosen_candidate.always_equivalent,
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
                transform_always_equivalent: chosen_candidate.always_equivalent,
                transform: Some(current_transform_kind),
            }
        }
    }
}

fn make_all_zeros_value(ty: &PirType) -> Result<IrValue> {
    match ty {
        PirType::Token => Ok(IrValue::make_token()),
        PirType::Bits(width) => {
            if *width == 0 {
                Ok(IrValue::from_bits(&IrBits::make_ubits(0, 0).unwrap()))
            } else {
                Ok(IrValue::from_bits(&IrBits::make_ubits(*width, 0).unwrap()))
            }
        }
        PirType::Tuple(elem_types) => {
            let elems: Result<Vec<IrValue>> =
                elem_types.iter().map(|t| make_all_zeros_value(t)).collect();
            Ok(IrValue::make_tuple(&elems?))
        }
        PirType::Array(arr) => {
            if arr.element_count == 0 {
                return Err(anyhow::anyhow!(
                    "cannot construct all-zeros oracle sample for zero-length array type {}",
                    ty
                ));
            }
            let mut elems: Vec<IrValue> = Vec::with_capacity(arr.element_count);
            for _ in 0..arr.element_count {
                elems.push(make_all_zeros_value(&arr.element_type)?);
            }
            IrValue::make_array(&elems).map_err(|e| {
                anyhow::anyhow!("failed to construct all-zeros array oracle sample: {}", e)
            })
        }
    }
}

fn make_all_ones_value(ty: &PirType) -> Result<IrValue> {
    match ty {
        PirType::Token => Ok(IrValue::make_token()),
        PirType::Bits(width) => {
            if *width == 0 {
                Ok(IrValue::from_bits(&IrBits::make_ubits(0, 0).unwrap()))
            } else if *width <= 64 {
                let mask = if *width == 64 {
                    u64::MAX
                } else {
                    (1u64 << *width) - 1
                };
                Ok(IrValue::from_bits(
                    &IrBits::make_ubits(*width, mask).unwrap(),
                ))
            } else {
                // Build the bit vector directly to avoid fixed-width integer limits.
                let ones: Vec<bool> = vec![true; *width];
                Ok(IrValue::from_bits(&IrBits::from_lsb_is_0(&ones)))
            }
        }
        PirType::Tuple(elem_types) => {
            let elems: Result<Vec<IrValue>> =
                elem_types.iter().map(|t| make_all_ones_value(t)).collect();
            Ok(IrValue::make_tuple(&elems?))
        }
        PirType::Array(arr) => {
            if arr.element_count == 0 {
                return Err(anyhow::anyhow!(
                    "cannot construct all-ones oracle sample for zero-length array type {}",
                    ty
                ));
            }
            let mut elems: Vec<IrValue> = Vec::with_capacity(arr.element_count);
            for _ in 0..arr.element_count {
                elems.push(make_all_ones_value(&arr.element_type)?);
            }
            IrValue::make_array(&elems).map_err(|e| {
                anyhow::anyhow!("failed to construct all-ones array oracle sample: {}", e)
            })
        }
    }
}

fn arbitrary_value_for_type<R: Rng>(rng: &mut R, ty: &PirType) -> Result<IrValue> {
    match ty {
        PirType::Token => Ok(IrValue::make_token()),
        PirType::Bits(width) => {
            let bits = arbitrary_irbits(rng, *width);
            Ok(IrValue::from_bits(&bits))
        }
        PirType::Tuple(elem_types) => {
            let elems: Result<Vec<IrValue>> = elem_types
                .iter()
                .map(|t| arbitrary_value_for_type(rng, t))
                .collect();
            Ok(IrValue::make_tuple(&elems?))
        }
        PirType::Array(arr) => {
            if arr.element_count == 0 {
                return Err(anyhow::anyhow!(
                    "cannot construct random oracle sample for zero-length array type {}",
                    ty
                ));
            }
            let mut elems: Vec<IrValue> = Vec::with_capacity(arr.element_count);
            for _ in 0..arr.element_count {
                elems.push(arbitrary_value_for_type(rng, &arr.element_type)?);
            }
            IrValue::make_array(&elems).map_err(|e| {
                anyhow::anyhow!("failed to construct random array oracle sample: {}", e)
            })
        }
    }
}

fn eval_fn_safe(f: &IrFn, args: &[IrValue]) -> Result<IrValue, ()> {
    // Note: `xlsynth_pir::ir_eval` uses internal `expect` / `unwrap` paths for
    // invariants; rewiring transforms may temporarily violate those (cycles,
    // missing package context for invoke, etc.). We treat any such failure as a
    // rejection signal for the candidate, not a crash.
    //
    // The MCMC state is compacted/toposorted at initialization and after each
    // successful candidate application, so the oracle can skip the evaluator's
    // per-call node-order check.
    let result = catch_unwind(AssertUnwindSafe(|| {
        eval_fn_assuming_node_index_topological(f, args)
    }));
    match result {
        Ok(FnEvalResult::Success(s)) => Ok(s.value),
        Ok(FnEvalResult::Failure(_f)) => Err(()),
        Err(_panic) => Err(()),
    }
}

fn make_oracle_args<F>(params: &[PirParam], label: &str, mut make_value: F) -> Result<Vec<IrValue>>
where
    F: FnMut(&PirType) -> Result<IrValue>,
{
    params
        .iter()
        .map(|p| make_value(&p.ty))
        .collect::<Result<Vec<_>>>()
        .map_err(|e| anyhow::anyhow!("failed to construct {} oracle sample args: {}", label, e))
}

fn pir_equiv_oracle<R: Rng>(
    lhs: &IrFn,
    rhs: &IrFn,
    rng: &mut R,
    random_samples: usize,
    enable_formal_oracle: bool,
    baseline_cache: &mut EvalFnBaselineResults,
) -> bool {
    if lhs.params.len() != rhs.params.len() || lhs.ret_ty != rhs.ret_ty {
        return false;
    }
    for (lp, rp) in lhs.params.iter().zip(rhs.params.iter()) {
        if lp.ty != rp.ty {
            return false;
        }
    }

    // The accepted-state invariant says each current `lhs` is equivalent to the
    // initial baseline. Populate this once for the first oracle check in a
    // chain/segment, then keep comparing candidates to those expected return
    // values instead of re-evaluating `lhs`.
    if let Err(e) = baseline_cache.ensure_populated(lhs, rng, random_samples) {
        log::debug!(
            "[pir-mcmc] failed to populate oracle baseline cache: {}; rejecting candidate",
            e
        );
        return false;
    }
    for (args, expected_value) in baseline_cache
        .samples
        .iter()
        .zip(baseline_cache.expected_values.iter())
    {
        let Ok(expected_value) = expected_value else {
            return false;
        };
        let Ok(rhs_value) = eval_fn_safe(rhs, args) else {
            return false;
        };
        if expected_value != &rhs_value {
            return false;
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

fn get_pir_transforms_for_run(enable_formal_oracle: bool) -> Vec<Box<dyn PirTransform>> {
    let mut all_transforms = get_all_pir_transforms();
    if !enable_formal_oracle {
        all_transforms.retain(|t| t.can_emit_always_equivalent_candidates());
    }
    all_transforms
}

struct PreparedRun {
    start_fn: IrFn,
    prepared_toggle_stimulus: Option<Arc<Vec<Vec<IrBits>>>>,
    initial_cost: Cost,
    effective_constraints: ConstraintLimits,
}

/// Validates a run and computes the canonicalized origin artifacts shared by
/// ordinary MCMC runs and lineage-producing runs.
fn prepare_run_start(mut start_fn: IrFn, options: &RunOptions) -> Result<PreparedRun> {
    if !options.objective.needs_toggle_stimulus() && options.toggle_stimulus.is_some() {
        return Err(anyhow::anyhow!(
            "toggle stimulus is not valid with objective {}",
            options.objective.value_name()
        ));
    }
    validate_constraint_configuration(
        options.objective,
        ConstraintLimits {
            max_delay: options.max_allowed_depth,
            max_area: options.max_allowed_area,
        },
    )?;
    compact_and_toposort_in_place(&mut start_fn)
        .map_err(|e| anyhow::anyhow!("compact_and_toposort_in_place failed: {}", e))?;

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

    let initial_cost = cost_with_effort_options_toggle_stimulus_and_extension_mode(
        &start_fn,
        options.objective,
        prepared_toggle_stimulus.as_ref().map(|v| v.as_slice()),
        &options.weighted_switching_options,
        options.extension_costing_mode,
    )?;
    let effective_constraints = effective_constraint_limits(
        options.objective,
        ConstraintLimits {
            max_delay: options.max_allowed_depth,
            max_area: options.max_allowed_area,
        },
        &initial_cost,
    );

    Ok(PreparedRun {
        start_fn,
        prepared_toggle_stimulus,
        initial_cost,
        effective_constraints,
    })
}

/// Validates whether a run shape can produce a v1 winning-lineage artifact.
pub fn validate_pir_mcmc_artifact_run_options(options: &RunOptions) -> Result<()> {
    if options.threads != 1 {
        return Err(anyhow::anyhow!(
            "run_pir_mcmc_with_artifact currently supports only single-chain runs; got threads={}",
            options.threads
        ));
    }
    if options.chain_strategy != ChainStrategy::Independent {
        return Err(anyhow::anyhow!(
            "run_pir_mcmc_with_artifact currently supports only chain_strategy=Independent"
        ));
    }
    if options.max_allowed_depth.is_some() || options.max_allowed_area.is_some() {
        return Err(anyhow::anyhow!(
            "run_pir_mcmc_with_artifact currently supports only unconstrained runs"
        ));
    }
    if options.objective.enforces_non_regressing_depth() {
        return Err(anyhow::anyhow!(
            "run_pir_mcmc_with_artifact does not yet support objectives with implicit feasibility caps; got {}",
            options.objective.value_name()
        ));
    }
    Ok(())
}

fn validate_prefix_minimization_artifact(artifact: &PirMcmcArtifact) -> Result<()> {
    validate_pir_mcmc_artifact_run_options(&artifact.run_options)?;
    Ok(())
}

fn retained_win_fraction_for_metric(origin_metric: u128, winner_metric: u128, metric: u128) -> f64 {
    let total_win = origin_metric.saturating_sub(winner_metric);
    if total_win == 0 {
        return 0.0;
    }
    let retained_win = origin_metric.saturating_sub(metric);
    retained_win as f64 / total_win as f64
}

fn win_percent_vs_origin_for_metric(origin_metric: u128, metric: u128) -> f64 {
    if origin_metric == 0 {
        return 0.0;
    }
    100.0 * origin_metric.saturating_sub(metric) as f64 / origin_metric as f64
}

fn objective_supports_budget_frontier_search(objective: Objective) -> bool {
    !objective.needs_toggle_stimulus()
        && !matches!(
            objective,
            Objective::G8rNodesTimesWeightedSwitchingNoDepthRegress
        )
}

fn make_budget_witness(
    witness_fn: IrFn,
    witness_cost: Cost,
    accepted_rewrite_count: usize,
    objective: Objective,
    origin_metric: u128,
    winner_metric: u128,
) -> PirMcmcBudgetWitness {
    let metric = objective.metric(&witness_cost);
    PirMcmcBudgetWitness {
        witness_fn,
        witness_cost,
        accepted_rewrite_count,
        metric,
        absolute_win: origin_metric.saturating_sub(metric),
        win_percent_vs_origin: win_percent_vs_origin_for_metric(origin_metric, metric),
        retained_win_fraction: retained_win_fraction_for_metric(
            origin_metric,
            winner_metric,
            metric,
        ),
    }
}

fn better_budget_witness(
    candidate: &PirMcmcBudgetWitness,
    incumbent: &PirMcmcBudgetWitness,
) -> bool {
    candidate.metric < incumbent.metric
        || (candidate.metric == incumbent.metric
            && candidate.accepted_rewrite_count < incumbent.accepted_rewrite_count)
}

fn frontier_budgets(options: PirMcmcBudgetFrontierOptions) -> Result<Vec<usize>> {
    if options.budget_step == 0 {
        return Err(anyhow::anyhow!("budget_step must be > 0"));
    }
    if options.max_rewrites == 0 {
        return Err(anyhow::anyhow!("max_rewrites must be > 0"));
    }
    if options.budget_step > options.max_rewrites {
        return Err(anyhow::anyhow!(
            "budget_step must be <= max_rewrites; got step={} max={}",
            options.budget_step,
            options.max_rewrites
        ));
    }
    if options.rollouts_per_budget == 0 {
        return Err(anyhow::anyhow!("rollouts_per_budget must be > 0"));
    }
    if options.proposal_attempts_per_rewrite == 0 {
        return Err(anyhow::anyhow!("proposal_attempts_per_rewrite must be > 0"));
    }
    if !options.witness_kind_boost.is_finite() || options.witness_kind_boost < 0.0 {
        return Err(anyhow::anyhow!(
            "witness_kind_boost must be finite and >= 0; got {}",
            options.witness_kind_boost
        ));
    }

    let mut budgets = Vec::new();
    let mut budget = options.budget_step;
    while budget <= options.max_rewrites {
        budgets.push(budget);
        match budget.checked_add(options.budget_step) {
            Some(next) => budget = next,
            None => break,
        }
    }
    if budgets.last().copied() != Some(options.max_rewrites) {
        budgets.push(options.max_rewrites);
    }
    Ok(budgets)
}

fn build_witness_guided_transform_weights(
    transforms: &[Box<dyn PirTransform>],
    artifact: &PirMcmcArtifact,
    witness_kind_boost: f64,
) -> Vec<f64> {
    let mut counts = BTreeMap::<PirTransformKind, usize>::new();
    for step in artifact.winning_lineage.iter() {
        *counts.entry(step.transform_kind).or_insert(0) += 1;
    }
    transforms
        .iter()
        .map(|transform| {
            1.0 + witness_kind_boost * counts.get(&transform.kind()).copied().unwrap_or(0) as f64
        })
        .collect()
}

fn prefix_baseline_for_budget(
    artifact: &PirMcmcArtifact,
    rewrite_budget: usize,
    origin_metric: u128,
    winner_metric: u128,
) -> PirMcmcBudgetWitness {
    let objective = artifact.run_options.objective;
    let mut best = make_budget_witness(
        artifact.origin_fn.clone(),
        artifact.origin_cost,
        0,
        objective,
        origin_metric,
        winner_metric,
    );
    for step in artifact
        .winning_lineage
        .iter()
        .take_while(|step| step.accepted_rewrite_index <= rewrite_budget)
    {
        let candidate = make_budget_witness(
            step.state.clone(),
            step.cost,
            step.accepted_rewrite_index,
            objective,
            origin_metric,
            winner_metric,
        );
        if better_budget_witness(&candidate, &best) {
            best = candidate;
        }
    }
    best
}

fn chain_strategy_value_name(strategy: ChainStrategy) -> &'static str {
    match strategy {
        ChainStrategy::Independent => "independent",
        ChainStrategy::ExploreExploit => "explore-exploit",
    }
}

fn chain_strategy_from_value_name(value: &str) -> Result<ChainStrategy> {
    match value {
        "independent" => Ok(ChainStrategy::Independent),
        "explore-exploit" => Ok(ChainStrategy::ExploreExploit),
        _ => Err(anyhow::anyhow!(
            "unknown chain strategy in artifact: {}",
            value
        )),
    }
}

fn emit_pkg_text_toposorted(pkg: &PirPackage) -> Result<String> {
    let mut pkg = pkg.clone();
    for member in pkg.members.iter_mut() {
        match member {
            PirPackageMember::Function(f) => {
                compact_and_toposort_in_place(f)
                    .map_err(|e| anyhow::anyhow!("compact_and_toposort_in_place failed: {}", e))?;
            }
            PirPackageMember::Block { func, .. } => {
                compact_and_toposort_in_place(func)
                    .map_err(|e| anyhow::anyhow!("compact_and_toposort_in_place failed: {}", e))?;
            }
        }
    }
    Ok(pkg.to_string())
}

fn package_with_replaced_fn(
    package_template: &PirPackage,
    top_fn_name: &str,
    replacement: &IrFn,
) -> Result<PirPackage> {
    let mut pkg = package_template.clone();
    let top_mut = pkg.get_fn_mut(top_fn_name).ok_or_else(|| {
        anyhow::anyhow!(
            "top function '{}' not found in artifact package template",
            top_fn_name
        )
    })?;
    *top_mut = replacement.clone();
    Ok(pkg)
}

fn write_artifact_state_package(
    artifact_dir: &Path,
    package_template: &PirPackage,
    top_fn_name: &str,
    relative_file: &str,
    state_fn: &IrFn,
) -> Result<()> {
    let pkg = package_with_replaced_fn(package_template, top_fn_name, state_fn)?;
    let text = emit_pkg_text_toposorted(&pkg)?;
    let path = artifact_dir.join(relative_file);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            anyhow::anyhow!(
                "failed to create artifact state directory {}: {}",
                parent.display(),
                e
            )
        })?;
    }
    std::fs::write(&path, text.as_bytes())
        .map_err(|e| anyhow::anyhow!("failed to write {}: {}", path.display(), e))
}

fn load_artifact_state_fn(
    artifact_dir: &Path,
    relative_file: &str,
    top_fn_name: &str,
) -> Result<(PirPackage, IrFn)> {
    let path = artifact_dir.join(relative_file);
    let pkg = ir_parser::parse_and_validate_path_to_package(&path).map_err(|e| {
        anyhow::anyhow!(
            "failed to parse artifact state package {}: {:?}",
            path.display(),
            e
        )
    })?;
    let state_fn = pkg.get_fn(top_fn_name).cloned().ok_or_else(|| {
        anyhow::anyhow!(
            "artifact state package {} does not contain top function '{}'",
            path.display(),
            top_fn_name
        )
    })?;
    Ok((pkg, state_fn))
}

impl PersistedRunOptions {
    fn from_run_options(options: &RunOptions) -> Self {
        Self {
            max_iters: options.max_iters,
            threads: options.threads,
            chain_strategy: chain_strategy_value_name(options.chain_strategy).to_string(),
            checkpoint_iters: options.checkpoint_iters,
            progress_iters: options.progress_iters,
            seed: options.seed,
            initial_temperature: options.initial_temperature,
            objective: options.objective.value_name().to_string(),
            extension_costing_mode: options.extension_costing_mode.value_name().to_string(),
            max_allowed_depth: options.max_allowed_depth,
            max_allowed_area: options.max_allowed_area,
            switching_beta1: options.weighted_switching_options.beta1,
            switching_beta2: options.weighted_switching_options.beta2,
            switching_primary_output_load: options.weighted_switching_options.primary_output_load,
            enable_formal_oracle: options.enable_formal_oracle,
        }
    }

    fn into_run_options(self) -> Result<RunOptions> {
        Ok(RunOptions {
            max_iters: self.max_iters,
            threads: self.threads,
            chain_strategy: chain_strategy_from_value_name(&self.chain_strategy)?,
            checkpoint_iters: self.checkpoint_iters,
            progress_iters: self.progress_iters,
            seed: self.seed,
            initial_temperature: self.initial_temperature,
            objective: Objective::from_value_name(&self.objective)?,
            extension_costing_mode: ExtensionCostingMode::from_value_name(
                &self.extension_costing_mode,
            )?,
            max_allowed_depth: self.max_allowed_depth,
            max_allowed_area: self.max_allowed_area,
            weighted_switching_options: count_toggles::WeightedSwitchingOptions {
                beta1: self.switching_beta1,
                beta2: self.switching_beta2,
                primary_output_load: self.switching_primary_output_load,
            },
            enable_formal_oracle: self.enable_formal_oracle,
            trajectory_dir: None,
            toggle_stimulus: None,
        })
    }
}

/// Writes a durable winning-lineage artifact under `run_dir/winning-lineage`.
pub fn write_pir_mcmc_artifact_dir(
    artifact: &PirMcmcArtifact,
    package_template: &PirPackage,
    run_dir: &Path,
) -> Result<PathBuf> {
    validate_prefix_minimization_artifact(artifact)?;
    let artifact_dir = run_dir.join(PIR_MCMC_ARTIFACT_DIR_NAME);
    std::fs::create_dir_all(&artifact_dir).map_err(|e| {
        anyhow::anyhow!(
            "failed to create artifact directory {}: {}",
            artifact_dir.display(),
            e
        )
    })?;

    let top_fn_name = artifact.origin_fn.name.clone();
    let origin_file = format!("{}/origin.ir", PIR_MCMC_ARTIFACT_STATES_DIR_NAME);
    let raw_winner_file = format!("{}/raw-winner.ir", PIR_MCMC_ARTIFACT_STATES_DIR_NAME);
    write_artifact_state_package(
        &artifact_dir,
        package_template,
        &top_fn_name,
        &origin_file,
        &artifact.origin_fn,
    )?;
    write_artifact_state_package(
        &artifact_dir,
        package_template,
        &top_fn_name,
        &raw_winner_file,
        &artifact.raw_winner_fn,
    )?;

    let mut winning_lineage = Vec::with_capacity(artifact.winning_lineage.len());
    for step in artifact.winning_lineage.iter() {
        let state_file = format!(
            "{}/step-{:06}.ir",
            PIR_MCMC_ARTIFACT_STATES_DIR_NAME, step.accepted_rewrite_index
        );
        write_artifact_state_package(
            &artifact_dir,
            package_template,
            &top_fn_name,
            &state_file,
            &step.state,
        )?;
        winning_lineage.push(PersistedAcceptedLineageStep {
            accepted_rewrite_index: step.accepted_rewrite_index,
            global_iter: step.global_iter,
            transform_kind: step.transform_kind,
            state: PersistedArtifactState {
                file: state_file,
                cost: step.cost,
            },
        });
    }

    let manifest = PersistedPirMcmcArtifactManifest {
        schema_version: PIR_MCMC_ARTIFACT_SCHEMA_VERSION,
        top_fn_name,
        run_options: PersistedRunOptions::from_run_options(&artifact.run_options),
        origin: PersistedArtifactState {
            file: origin_file,
            cost: artifact.origin_cost,
        },
        raw_winner: PersistedArtifactState {
            file: raw_winner_file,
            cost: artifact.raw_winner_cost,
        },
        winning_lineage,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| anyhow::anyhow!("failed to serialize artifact manifest: {}", e))?;
    let manifest_path = artifact_dir.join(PIR_MCMC_ARTIFACT_MANIFEST_FILE);
    std::fs::write(&manifest_path, manifest_json.as_bytes())
        .map_err(|e| anyhow::anyhow!("failed to write {}: {}", manifest_path.display(), e))?;
    Ok(artifact_dir)
}

/// Loads a durable winning-lineage artifact from `run_dir/winning-lineage`.
pub fn read_pir_mcmc_artifact_dir(run_dir: &Path) -> Result<LoadedPirMcmcArtifact> {
    let artifact_dir = run_dir.join(PIR_MCMC_ARTIFACT_DIR_NAME);
    let manifest_path = artifact_dir.join(PIR_MCMC_ARTIFACT_MANIFEST_FILE);
    let manifest_text = std::fs::read_to_string(&manifest_path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {}", manifest_path.display(), e))?;
    let manifest: PersistedPirMcmcArtifactManifest = serde_json::from_str(&manifest_text)
        .map_err(|e| anyhow::anyhow!("failed to parse {}: {}", manifest_path.display(), e))?;
    if manifest.schema_version != PIR_MCMC_ARTIFACT_SCHEMA_VERSION {
        return Err(anyhow::anyhow!(
            "unsupported PIR MCMC artifact schema version {}; expected {}",
            manifest.schema_version,
            PIR_MCMC_ARTIFACT_SCHEMA_VERSION
        ));
    }

    let top_fn_name = manifest.top_fn_name.clone();
    let (package_template, origin_fn) =
        load_artifact_state_fn(&artifact_dir, &manifest.origin.file, &top_fn_name)?;
    let (_, raw_winner_fn) =
        load_artifact_state_fn(&artifact_dir, &manifest.raw_winner.file, &top_fn_name)?;

    let mut winning_lineage = Vec::with_capacity(manifest.winning_lineage.len());
    for step in manifest.winning_lineage.into_iter() {
        let (_, state) = load_artifact_state_fn(&artifact_dir, &step.state.file, &top_fn_name)?;
        winning_lineage.push(AcceptedLineageStep {
            accepted_rewrite_index: step.accepted_rewrite_index,
            global_iter: step.global_iter,
            transform_kind: step.transform_kind,
            state,
            cost: step.state.cost,
        });
    }

    let artifact = PirMcmcArtifact {
        origin_fn,
        origin_cost: manifest.origin.cost,
        run_options: manifest.run_options.into_run_options()?,
        raw_winner_fn,
        raw_winner_cost: manifest.raw_winner.cost,
        winning_lineage,
    };
    validate_prefix_minimization_artifact(&artifact)?;
    Ok(LoadedPirMcmcArtifact {
        artifact,
        package_template,
        top_fn_name,
    })
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
        let all_transforms = get_pir_transforms_for_run(self.enable_formal_oracle);
        let weights = build_transform_weights(&all_transforms);

        let mut context = PirMcmcContext {
            rng: &mut iteration_rng,
            all_transforms,
            weights,
            enable_formal_oracle: self.enable_formal_oracle,
            oracle_baseline_cache: EvalFnBaselineResults::default(),
        };

        let toggle_stimulus = self.prepared_toggle_stimulus.as_ref().map(|v| v.as_slice());

        let mut current_fn = start_state.clone();
        let mut current_cost = cost_with_effort_options_toggle_stimulus_and_extension_mode(
            &current_fn,
            self.objective,
            toggle_stimulus,
            &self.weighted_switching_options,
            self.extension_costing_mode,
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
        let mut best_score = search_score(&best_cost, self.objective, self.constraints);
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
                &mut best_score,
                &mut context,
                temp,
                self.objective,
                self.extension_costing_mode,
                toggle_stimulus,
                &self.weighted_switching_options,
                self.constraints,
            );

            let mut accepted_digest: Option<[u8; 32]> = None;
            let mut accepted_sample_sent = false;

            if let IterationOutcomeDetails::Accepted { .. } = iteration_output.outcome {
                if let Some(ref tx) = self.accepted_sample_tx {
                    match canonicalize_fn_for_sample(&iteration_output.output_state) {
                        Ok(canon) => match optimize_pir_fn_via_xls_with_extension_mode(
                            &canon,
                            self.extension_costing_mode,
                        ) {
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
                let iter_score = search_score(
                    &iteration_output.output_cost,
                    self.objective,
                    self.constraints,
                );
                let iter_violation = iter_score.violation;
                let rec = json!({
                    "chain_no": params.chain_no,
                    "role": format!("{:?}", params.role),
                    "global_iter": global_iter,
                    "temp": temp,
                    "outcome": iteration_outcome_tag(&iteration_output.outcome),
                    "best_updated": iteration_output.best_updated,
                    "objective": format!("{:?}", self.objective),
                    "extension_costing_mode": self.extension_costing_mode.value_name(),
                    "metric": metric_u128,
                    "pir_nodes": iteration_output.output_cost.pir_nodes,
                    "g8r_nodes": iteration_output.output_cost.g8r_nodes,
                    "g8r_depth": iteration_output.output_cost.g8r_depth,
                    "g8r_le_graph_milli": iteration_output.output_cost.g8r_le_graph_milli,
                    "g8r_gate_output_toggles": iteration_output.output_cost.g8r_gate_output_toggles,
                    "g8r_weighted_switching_milli": iteration_output.output_cost.g8r_weighted_switching_milli,
                    "feasible": iter_score.feasible(),
                    "delay_over": iter_violation.and_then(|v| v.delay_over),
                    "area_over": iter_violation.and_then(|v| v.area_over),
                    "oracle_time_micros": iteration_output.oracle_time_micros,
                    "transform": iteration_output.transform.map(|k| format!("{:?}", k)),
                    "transform_mechanism": iteration_output.transform.map(|k| k.mechanism_hint()),
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
                    let before = shared_best.score();
                    let _ = shared_best.try_update(best_score, best_fn.clone());
                    let after = shared_best.score();
                    if after < before {
                        log::info!(
                            "[pir-mcmc] GLOBAL BEST UPDATE c{:03}:i{:06} | {} -> {}",
                            params.chain_no,
                            global_iter,
                            format_search_score(before),
                            format_search_score(after),
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
                    "PIR MCMC c{:03}:i{:06} | GBest={} | LBest (pir={}, g8r_n={}, g8r_d={}, score={}) | Cur (pir={}, g8r_n={}, g8r_d={}, score={}) | Temp={:.2e} | Samples/s={:.2}",
                    params.chain_no,
                    global_iter,
                    self.shared_best
                        .as_ref()
                        .map(|b| format_search_score(b.score()))
                        .unwrap_or_else(|| "none".to_string()),
                    best_cost.pir_nodes,
                    best_cost.g8r_nodes,
                    best_cost.g8r_depth,
                    format_search_score(best_score),
                    current_cost.pir_nodes,
                    current_cost.g8r_nodes,
                    current_cost.g8r_depth,
                    format_search_score(search_score(
                        &current_cost,
                        self.objective,
                        self.constraints
                    )),
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

/// Runs a single-chain MCMC search while retaining the exact accepted lineage
/// that led to the final raw winning state.
pub fn run_pir_mcmc_with_artifact(start_fn: IrFn, options: RunOptions) -> Result<PirMcmcArtifact> {
    validate_pir_mcmc_artifact_run_options(&options)?;
    let all_transforms = get_pir_transforms_for_run(options.enable_formal_oracle);
    Ok(run_pir_mcmc_with_artifact_using_transforms(start_fn, options, all_transforms)?.artifact)
}

fn run_pir_mcmc_with_artifact_using_transforms(
    start_fn: IrFn,
    options: RunOptions,
    all_transforms: Vec<Box<dyn PirTransform>>,
) -> Result<PirMcmcArtifactRunOutput> {
    run_pir_mcmc_with_artifact_using_transforms_and_observers(
        start_fn,
        options,
        all_transforms,
        None,
        None,
    )
}

fn run_pir_mcmc_with_artifact_and_observers(
    start_fn: IrFn,
    options: RunOptions,
    shared_best: Option<Arc<Best>>,
    checkpoint_tx: Option<Sender<CheckpointMsg>>,
) -> Result<PirMcmcArtifactRunOutput> {
    validate_pir_mcmc_artifact_run_options(&options)?;
    let all_transforms = get_pir_transforms_for_run(options.enable_formal_oracle);
    run_pir_mcmc_with_artifact_using_transforms_and_observers(
        start_fn,
        options,
        all_transforms,
        shared_best,
        checkpoint_tx,
    )
}

fn run_pir_mcmc_with_artifact_using_transforms_and_observers(
    start_fn: IrFn,
    options: RunOptions,
    all_transforms: Vec<Box<dyn PirTransform>>,
    shared_best: Option<Arc<Best>>,
    checkpoint_tx: Option<Sender<CheckpointMsg>>,
) -> Result<PirMcmcArtifactRunOutput> {
    let prepared = prepare_run_start(start_fn, &options)?;

    let mut trajectory_writer: Option<std::io::BufWriter<std::fs::File>> =
        if let Some(dir) = &options.trajectory_dir {
            std::fs::create_dir_all(dir).map_err(|e| {
                anyhow::anyhow!("failed to create trajectory dir {}: {}", dir.display(), e)
            })?;
            let path = dir.join("trajectory.c000.jsonl");
            let f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| anyhow::anyhow!("failed to open {}: {}", path.display(), e))?;
            Some(std::io::BufWriter::new(f))
        } else {
            None
        };

    let toggle_stimulus = prepared
        .prepared_toggle_stimulus
        .as_ref()
        .map(|v| v.as_slice());
    let weights = build_transform_weights(&all_transforms);
    let mut iteration_rng = Pcg64Mcg::seed_from_u64(options.seed);
    let mut context = PirMcmcContext {
        rng: &mut iteration_rng,
        all_transforms,
        weights,
        enable_formal_oracle: options.enable_formal_oracle,
        oracle_baseline_cache: EvalFnBaselineResults::default(),
    };

    let origin_fn = prepared.start_fn.clone();
    let origin_cost = prepared.initial_cost;
    let mut current_fn = prepared.start_fn.clone();
    let mut current_cost = prepared.initial_cost;
    // `mcmc_iteration()` maintains the existing public optimized-best
    // behavior. The lineage artifact tracks the raw accepted winner separately.
    let mut best_fn_for_iteration = prepared.start_fn.clone();
    let mut best_cost_for_iteration = prepared.initial_cost;
    let mut best_score = search_score(
        &best_cost_for_iteration,
        options.objective,
        prepared.effective_constraints,
    );
    let mut raw_winner_fn = prepared.start_fn;
    let mut raw_winner_cost = prepared.initial_cost;
    let mut lineage = WinningLineageTracker::default();
    let mut stats = McmcStats::default();
    let run_start_time = Instant::now();

    let mut global_iter = 0u64;
    while global_iter < options.max_iters {
        global_iter += 1;
        let progress_ratio = if options.max_iters > 0 {
            (global_iter as f64) / (options.max_iters as f64)
        } else {
            0.0
        };
        let temp = options.initial_temperature
            * (1.0 - progress_ratio.min(1.0)).max(MIN_TEMPERATURE_RATIO);

        let iteration_output = mcmc_iteration(
            current_fn,
            current_cost,
            &mut best_fn_for_iteration,
            &mut best_cost_for_iteration,
            &mut best_score,
            &mut context,
            temp,
            options.objective,
            options.extension_costing_mode,
            toggle_stimulus,
            &options.weighted_switching_options,
            prepared.effective_constraints,
        );

        if let IterationOutcomeDetails::Accepted { kind } = &iteration_output.outcome {
            lineage.record_accepted_step(
                global_iter,
                *kind,
                iteration_output.output_state.clone(),
                iteration_output.output_cost,
                iteration_output.best_updated,
            );
        }

        if let Some(w) = trajectory_writer.as_mut() {
            let metric_u128 = options.objective.metric(&iteration_output.output_cost);
            let iter_score = search_score(
                &iteration_output.output_cost,
                options.objective,
                prepared.effective_constraints,
            );
            let iter_violation = iter_score.violation;
            let rec = json!({
                "chain_no": 0,
                "role": format!("{:?}", ChainRole::Exploit),
                "global_iter": global_iter,
                "temp": temp,
                "outcome": iteration_outcome_tag(&iteration_output.outcome),
                "best_updated": iteration_output.best_updated,
                "objective": format!("{:?}", options.objective),
                "extension_costing_mode": options.extension_costing_mode.value_name(),
                "metric": metric_u128,
                "pir_nodes": iteration_output.output_cost.pir_nodes,
                "g8r_nodes": iteration_output.output_cost.g8r_nodes,
                "g8r_depth": iteration_output.output_cost.g8r_depth,
                "g8r_le_graph_milli": iteration_output.output_cost.g8r_le_graph_milli,
                "g8r_gate_output_toggles": iteration_output.output_cost.g8r_gate_output_toggles,
                "g8r_weighted_switching_milli": iteration_output.output_cost.g8r_weighted_switching_milli,
                "feasible": iter_score.feasible(),
                "delay_over": iter_violation.and_then(|v| v.delay_over),
                "area_over": iter_violation.and_then(|v| v.area_over),
                "oracle_time_micros": iteration_output.oracle_time_micros,
                "transform": iteration_output.transform.map(|k| format!("{:?}", k)),
                "transform_mechanism": iteration_output.transform.map(|k| k.mechanism_hint()),
                "transform_always_equivalent": iteration_output.transform_always_equivalent,
                "accepted_digest": Option::<String>::None,
                "accepted_sample_sent": false,
            });
            writeln!(w, "{}", rec.to_string())?;
            if global_iter % 1000 == 0 {
                w.flush()?;
            }
        }

        current_fn = iteration_output.output_state.clone();
        current_cost = iteration_output.output_cost;
        if iteration_output.best_updated {
            raw_winner_fn = current_fn.clone();
            raw_winner_cost = current_cost;
            if let Some(ref shared_best) = shared_best {
                let before = shared_best.score();
                let _ = shared_best.try_update(best_score, best_fn_for_iteration.clone());
                let after = shared_best.score();
                if after < before {
                    log::info!(
                        "[pir-mcmc] GLOBAL BEST UPDATE c000:i{:06} | {} -> {}",
                        global_iter,
                        format_search_score(before),
                        format_search_score(after),
                    );
                    if let Some(ref tx) = checkpoint_tx {
                        let _ = tx.send(CheckpointMsg {
                            chain_no: 0,
                            global_iter,
                            kind: CheckpointKind::GlobalBestUpdate,
                        });
                    }
                }
            }
        }

        if options.checkpoint_iters > 0 && global_iter % options.checkpoint_iters == 0 {
            if let Some(ref tx) = checkpoint_tx {
                let _ = tx.send(CheckpointMsg {
                    chain_no: 0,
                    global_iter,
                    kind: CheckpointKind::Periodic,
                });
            }
        }

        stats.update_for_iteration(&iteration_output, /* paranoid= */ false, global_iter);

        if options.progress_iters > 0
            && (global_iter % options.progress_iters == 0 || global_iter == options.max_iters)
        {
            let elapsed_secs = run_start_time.elapsed().as_secs_f64();
            let samples_per_sec = if elapsed_secs > 0.0 {
                global_iter as f64 / elapsed_secs
            } else {
                0.0
            };
            log::info!(
                "PIR MCMC c000:i{:06} | GBest={} | LBest (pir={}, g8r_n={}, g8r_d={}, score={}) | Cur (pir={}, g8r_n={}, g8r_d={}, score={}) | Temp={:.2e} | Samples/s={:.2}",
                global_iter,
                shared_best
                    .as_ref()
                    .map(|b| format_search_score(b.score()))
                    .unwrap_or_else(|| "none".to_string()),
                best_cost_for_iteration.pir_nodes,
                best_cost_for_iteration.g8r_nodes,
                best_cost_for_iteration.g8r_depth,
                format_search_score(best_score),
                current_cost.pir_nodes,
                current_cost.g8r_nodes,
                current_cost.g8r_depth,
                format_search_score(search_score(
                    &current_cost,
                    options.objective,
                    prepared.effective_constraints
                )),
                temp,
                samples_per_sec,
            );
        }
    }

    if let Some(mut w) = trajectory_writer {
        w.flush()?;
    }

    Ok(PirMcmcArtifactRunOutput {
        result: PirMcmcResult {
            best_fn: best_fn_for_iteration,
            best_cost: best_cost_for_iteration,
            stats,
        },
        artifact: PirMcmcArtifact {
            origin_fn,
            origin_cost,
            run_options: options,
            raw_winner_fn,
            raw_winner_cost,
            winning_lineage: lineage.winning_lineage,
        },
    })
}

/// Selects the earliest accepted winning-lineage prefix that retains the
/// requested fraction of the discovered objective improvement.
pub fn minimize_winning_prefix(
    artifact: &PirMcmcArtifact,
    options: PirMcmcPrefixMinimizeOptions,
) -> Result<PirMcmcPrefixMinimizeResult> {
    validate_prefix_minimization_artifact(artifact)?;
    if !options.retained_win_fraction.is_finite()
        || !(0.0..=1.0).contains(&options.retained_win_fraction)
    {
        return Err(anyhow::anyhow!(
            "retained_win_fraction must be finite and in [0, 1]; got {}",
            options.retained_win_fraction
        ));
    }

    let objective = artifact.run_options.objective;
    let origin_metric = objective.metric(&artifact.origin_cost);
    let winner_metric = objective.metric(&artifact.raw_winner_cost);
    if winner_metric >= origin_metric {
        return Err(anyhow::anyhow!(
            "artifact does not contain a positive objective win: origin_metric={}, winner_metric={}",
            origin_metric,
            winner_metric
        ));
    }

    if options.retained_win_fraction == 0.0 {
        return Ok(PirMcmcPrefixMinimizeResult {
            witness_fn: artifact.origin_fn.clone(),
            witness_cost: artifact.origin_cost,
            accepted_rewrite_count: 0,
            original_winning_lineage_len: artifact.winning_lineage.len(),
            requested_retained_win_fraction: options.retained_win_fraction,
            actual_retained_win_fraction: 0.0,
            origin_metric,
            winner_metric,
            witness_metric: origin_metric,
        });
    }

    for step in artifact.winning_lineage.iter() {
        let witness_metric = objective.metric(&step.cost);
        let actual_retained_win_fraction =
            retained_win_fraction_for_metric(origin_metric, winner_metric, witness_metric);
        if actual_retained_win_fraction >= options.retained_win_fraction {
            return Ok(PirMcmcPrefixMinimizeResult {
                witness_fn: step.state.clone(),
                witness_cost: step.cost,
                accepted_rewrite_count: step.accepted_rewrite_index,
                original_winning_lineage_len: artifact.winning_lineage.len(),
                requested_retained_win_fraction: options.retained_win_fraction,
                actual_retained_win_fraction,
                origin_metric,
                winner_metric,
                witness_metric,
            });
        }
    }

    Err(anyhow::anyhow!(
        "winning lineage did not contain a prefix retaining requested win fraction {}",
        options.retained_win_fraction
    ))
}

struct GuidedRolloutResult {
    best_witness: PirMcmcBudgetWitness,
}

fn run_witness_guided_rollout(
    artifact: &PirMcmcArtifact,
    rewrite_budget: usize,
    rollout_seed: u64,
    weights: Vec<f64>,
    proposal_attempts_per_rewrite: usize,
    transforms: Vec<Box<dyn PirTransform>>,
    origin_metric: u128,
    winner_metric: u128,
) -> Result<GuidedRolloutResult> {
    let objective = artifact.run_options.objective;
    let mut iteration_rng = Pcg64Mcg::seed_from_u64(rollout_seed);
    let mut context = PirMcmcContext {
        rng: &mut iteration_rng,
        all_transforms: transforms,
        weights,
        enable_formal_oracle: artifact.run_options.enable_formal_oracle,
        oracle_baseline_cache: EvalFnBaselineResults::default(),
    };
    let mut current_fn = artifact.origin_fn.clone();
    let mut current_cost = artifact.origin_cost;
    let mut best_fn_for_iteration = artifact.origin_fn.clone();
    let mut best_cost_for_iteration = artifact.origin_cost;
    let mut best_score = search_score(
        &best_cost_for_iteration,
        objective,
        ConstraintLimits::default(),
    );
    let mut best_witness = make_budget_witness(
        artifact.origin_fn.clone(),
        artifact.origin_cost,
        0,
        objective,
        origin_metric,
        winner_metric,
    );
    let max_proposals = rewrite_budget.saturating_mul(proposal_attempts_per_rewrite);
    let mut accepted_rewrites = 0usize;
    let mut proposal_attempts = 0usize;

    while accepted_rewrites < rewrite_budget && proposal_attempts < max_proposals {
        proposal_attempts += 1;
        let progress_ratio = accepted_rewrites as f64 / rewrite_budget.max(1) as f64;
        let temp = artifact.run_options.initial_temperature
            * (1.0 - progress_ratio.min(1.0)).max(MIN_TEMPERATURE_RATIO);
        let iteration_output = mcmc_iteration(
            current_fn,
            current_cost,
            &mut best_fn_for_iteration,
            &mut best_cost_for_iteration,
            &mut best_score,
            &mut context,
            temp,
            objective,
            artifact.run_options.extension_costing_mode,
            None,
            &artifact.run_options.weighted_switching_options,
            ConstraintLimits::default(),
        );
        current_fn = iteration_output.output_state.clone();
        current_cost = iteration_output.output_cost;
        if matches!(
            iteration_output.outcome,
            IterationOutcomeDetails::Accepted { .. }
        ) {
            accepted_rewrites += 1;
            let candidate = make_budget_witness(
                current_fn.clone(),
                current_cost,
                accepted_rewrites,
                objective,
                origin_metric,
                winner_metric,
            );
            if better_budget_witness(&candidate, &best_witness) {
                best_witness = candidate;
            }
        }
    }

    Ok(GuidedRolloutResult { best_witness })
}

/// Searches for best-found short witnesses at a schedule of accepted-rewrite
/// budgets, using the long witness to bias transform proposals.
pub fn search_winning_budget_frontier(
    artifact: &PirMcmcArtifact,
    options: PirMcmcBudgetFrontierOptions,
) -> Result<PirMcmcBudgetFrontierResult> {
    search_winning_budget_frontier_with_rollout(
        artifact,
        options,
        |rewrite_budget, _rollout_idx, rollout_seed, origin_metric, winner_metric| {
            let transforms = get_pir_transforms_for_run(artifact.run_options.enable_formal_oracle);
            let weights = build_witness_guided_transform_weights(
                &transforms,
                artifact,
                options.witness_kind_boost,
            );
            let rollout = run_witness_guided_rollout(
                artifact,
                rewrite_budget,
                rollout_seed,
                weights,
                options.proposal_attempts_per_rewrite,
                transforms,
                origin_metric,
                winner_metric,
            )?;
            Ok(rollout.best_witness)
        },
    )
}

fn search_winning_budget_frontier_with_rollout<F>(
    artifact: &PirMcmcArtifact,
    options: PirMcmcBudgetFrontierOptions,
    mut run_rollout: F,
) -> Result<PirMcmcBudgetFrontierResult>
where
    F: FnMut(usize, usize, u64, u128, u128) -> Result<PirMcmcBudgetWitness>,
{
    validate_prefix_minimization_artifact(artifact)?;
    if !objective_supports_budget_frontier_search(artifact.run_options.objective) {
        return Err(anyhow::anyhow!(
            "budget frontier search currently supports only objectives that can be recomputed without stored toggle stimulus; got {}",
            artifact.run_options.objective.value_name()
        ));
    }
    let budgets = frontier_budgets(options)?;
    let objective = artifact.run_options.objective;
    let origin_metric = objective.metric(&artifact.origin_cost);
    let winner_metric = objective.metric(&artifact.raw_winner_cost);
    if winner_metric >= origin_metric {
        return Err(anyhow::anyhow!(
            "artifact does not contain a positive objective win: origin_metric={}, winner_metric={}",
            origin_metric,
            winner_metric
        ));
    }

    let mut carried_guided = make_budget_witness(
        artifact.origin_fn.clone(),
        artifact.origin_cost,
        0,
        objective,
        origin_metric,
        winner_metric,
    );
    let mut points = Vec::with_capacity(budgets.len());
    for rewrite_budget in budgets {
        let prefix_baseline =
            prefix_baseline_for_budget(artifact, rewrite_budget, origin_metric, winner_metric);
        let mut best_for_budget = carried_guided.clone();
        for rollout_idx in 0..options.rollouts_per_budget {
            let rollout_seed = options
                .seed
                .wrapping_add((rewrite_budget as u64).wrapping_mul(1_000_003))
                .wrapping_add(rollout_idx as u64);
            let rollout_witness = run_rollout(
                rewrite_budget,
                rollout_idx,
                rollout_seed,
                origin_metric,
                winner_metric,
            )?;
            if better_budget_witness(&rollout_witness, &best_for_budget) {
                best_for_budget = rollout_witness;
            }
        }
        carried_guided = best_for_budget.clone();
        points.push(PirMcmcBudgetFrontierPoint {
            rewrite_budget,
            guided: best_for_budget,
            prefix_baseline,
        });
    }

    Ok(PirMcmcBudgetFrontierResult {
        origin_metric,
        winner_metric,
        original_winning_lineage_len: artifact.winning_lineage.len(),
        points,
    })
}

pub fn run_pir_mcmc_with_shared_best(
    start_fn: IrFn,
    options: RunOptions,
    shared_best: Option<Arc<Best>>,
    checkpoint_tx: Option<Sender<CheckpointMsg>>,
    accepted_sample_tx: Option<Sender<AcceptedSampleMsg>>,
) -> Result<PirMcmcResult> {
    let prepared = prepare_run_start(start_fn, &options)?;
    let runner = PirSegmentRunner {
        objective: options.objective,
        extension_costing_mode: options.extension_costing_mode,
        weighted_switching_options: options.weighted_switching_options,
        initial_temperature: options.initial_temperature,
        constraints: prepared.effective_constraints,
        enable_formal_oracle: options.enable_formal_oracle,
        progress_iters: options.progress_iters,
        checkpoint_iters: options.checkpoint_iters,
        checkpoint_tx,
        accepted_sample_tx,
        shared_best,
        trajectory_dir: options.trajectory_dir.clone(),
        prepared_toggle_stimulus: prepared.prepared_toggle_stimulus,
    };

    let objective = options.objective;
    let threshold = options.initial_temperature as u128;
    let constraints = prepared.effective_constraints;

    let (best_fn, best_cost, stats) = run_multichain(
        prepared.start_fn,
        options.max_iters,
        options.seed,
        options.threads.max(1) as usize,
        options.chain_strategy,
        options.checkpoint_iters,
        Arc::new(runner),
        move |c: &Cost| search_score(c, objective, constraints),
        |f: &IrFn| f.to_string(),
        move |cur_cost: &Cost, global_best_cost: &Cost| {
            let cur_score = search_score(cur_cost, objective, constraints);
            let global_best_score = search_score(global_best_cost, objective, constraints);
            match (cur_score.violation, global_best_score.violation) {
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (Some(cur_violation), Some(best_violation)) => {
                    repair_energy(cur_violation)
                        > repair_energy(best_violation).saturating_add(threshold)
                }
                (None, None) => {
                    objective.metric(cur_cost)
                        > objective.metric(global_best_cost).saturating_add(threshold)
                }
            }
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
    use std::collections::HashSet;
    use std::fs;

    use super::*;
    use count_toggles::WeightedSwitchingOptions;
    use tempfile::tempdir;
    use xlsynth_pir::ir::{ExtNaryAddArchitecture, NodePayload};
    use xlsynth_pir::ir_parser;
    use xlsynth_pir::ir_utils::remap_payload_with;

    fn parse_fn(ir_text: &str) -> IrFn {
        let mut parser = ir_parser::Parser::new(ir_text);
        parser.parse_fn().unwrap()
    }

    fn parse_pkg(ir_text: &str) -> PirPackage {
        let mut parser = ir_parser::Parser::new(ir_text);
        parser.parse_and_validate_package().unwrap()
    }

    fn test_run_options(objective: Objective) -> RunOptions {
        RunOptions {
            max_iters: 1,
            threads: 1,
            chain_strategy: ChainStrategy::Independent,
            checkpoint_iters: 100,
            progress_iters: 0,
            seed: 1,
            initial_temperature: 1.0,
            objective,
            extension_costing_mode: ExtensionCostingMode::Preserve,
            max_allowed_depth: None,
            max_allowed_area: None,
            weighted_switching_options: WeightedSwitchingOptions::default(),
            enable_formal_oracle: false,
            trajectory_dir: None,
            toggle_stimulus: None,
        }
    }

    fn cost_with_pir_nodes(pir_nodes: usize) -> Cost {
        Cost {
            pir_nodes,
            g8r_nodes: pir_nodes,
            g8r_depth: pir_nodes,
            g8r_le_graph_milli: 0,
            g8r_gate_output_toggles: 0,
            g8r_weighted_switching_milli: 0,
        }
    }

    fn renamed_state(base: &IrFn, name: &str) -> IrFn {
        let mut f = base.clone();
        f.name = name.to_string();
        f
    }

    fn manual_prefix_artifact() -> PirMcmcArtifact {
        let origin = parse_fn(
            r#"fn origin(x: bits[8] id=1) -> bits[8] {
  ret identity.2: bits[8] = identity(x, id=2)
}"#,
        );
        let step1_state = renamed_state(&origin, "step1");
        let step2_state = renamed_state(&origin, "step2");
        let step3_state = renamed_state(&origin, "step3");
        PirMcmcArtifact {
            origin_fn: origin.clone(),
            origin_cost: cost_with_pir_nodes(100),
            run_options: test_run_options(Objective::Nodes),
            raw_winner_fn: step3_state.clone(),
            raw_winner_cost: cost_with_pir_nodes(50),
            winning_lineage: vec![
                AcceptedLineageStep {
                    accepted_rewrite_index: 1,
                    global_iter: 2,
                    transform_kind: PirTransformKind::NotNotCancel,
                    state: step1_state,
                    cost: cost_with_pir_nodes(90),
                },
                AcceptedLineageStep {
                    accepted_rewrite_index: 2,
                    global_iter: 4,
                    transform_kind: PirTransformKind::NegNegCancel,
                    state: step2_state,
                    cost: cost_with_pir_nodes(70),
                },
                AcceptedLineageStep {
                    accepted_rewrite_index: 3,
                    global_iter: 7,
                    transform_kind: PirTransformKind::SelSameArmsFold,
                    state: step3_state,
                    cost: cost_with_pir_nodes(50),
                },
            ],
        }
    }

    #[derive(Debug)]
    struct RemoveDeadNodeTestTransform;

    impl PirTransform for RemoveDeadNodeTestTransform {
        fn kind(&self) -> PirTransformKind {
            PirTransformKind::NotNotCancel
        }

        fn find_candidates(&mut self, f: &IrFn) -> Vec<crate::transforms::TransformCandidate> {
            f.node_refs()
                .into_iter()
                .filter(|nr| {
                    f.get_node(*nr)
                        .name
                        .as_deref()
                        .map(|name| name.starts_with("dead"))
                        .unwrap_or(false)
                })
                .map(|nr| crate::transforms::TransformCandidate {
                    location: crate::transforms::TransformLocation::Node(nr),
                    always_equivalent: true,
                })
                .collect()
        }

        fn apply(
            &self,
            f: &mut IrFn,
            loc: &crate::transforms::TransformLocation,
        ) -> Result<(), String> {
            let crate::transforms::TransformLocation::Node(nr) = loc else {
                return Err("RemoveDeadNodeTestTransform expects a node location".to_string());
            };
            f.get_node_mut(*nr).payload = NodePayload::Nil;
            Ok(())
        }
    }

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
            extension_costing_mode: ExtensionCostingMode::Preserve,
            max_allowed_depth: None,
            max_allowed_area: None,
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
    fn winning_lineage_tracker_preserves_non_best_detours_before_later_win() {
        let origin = parse_fn(
            r#"fn origin(x: bits[8] id=1) -> bits[8] {
  ret identity.2: bits[8] = identity(x, id=2)
}"#,
        );
        let mut tracker = WinningLineageTracker::default();
        tracker.record_accepted_step(
            1,
            PirTransformKind::NotNotCancel,
            renamed_state(&origin, "best1"),
            cost_with_pir_nodes(90),
            true,
        );
        tracker.record_accepted_step(
            2,
            PirTransformKind::NegNegCancel,
            renamed_state(&origin, "detour"),
            cost_with_pir_nodes(95),
            false,
        );
        tracker.record_accepted_step(
            3,
            PirTransformKind::SelSameArmsFold,
            renamed_state(&origin, "best2"),
            cost_with_pir_nodes(80),
            true,
        );

        assert_eq!(tracker.winning_lineage.len(), 3);
        assert_eq!(tracker.winning_lineage[0].state.name, "best1");
        assert_eq!(tracker.winning_lineage[1].state.name, "detour");
        assert_eq!(tracker.winning_lineage[2].state.name, "best2");
        assert_eq!(tracker.winning_lineage[2].accepted_rewrite_index, 3);
    }

    #[test]
    fn prefix_minimization_selects_earliest_prefix_meeting_requested_win() {
        let artifact = manual_prefix_artifact();

        let retain_all = minimize_winning_prefix(
            &artifact,
            PirMcmcPrefixMinimizeOptions {
                retained_win_fraction: 1.0,
            },
        )
        .unwrap();
        assert_eq!(retain_all.accepted_rewrite_count, 3);
        assert_eq!(retain_all.witness_metric, 50);

        let retain_most = minimize_winning_prefix(
            &artifact,
            PirMcmcPrefixMinimizeOptions {
                retained_win_fraction: 0.6,
            },
        )
        .unwrap();
        assert_eq!(retain_most.accepted_rewrite_count, 2);
        assert_eq!(retain_most.witness_fn.name, "step2");
        assert_eq!(retain_most.witness_metric, 70);
        assert_eq!(retain_most.original_winning_lineage_len, 3);
        assert!((retain_most.actual_retained_win_fraction - 0.6).abs() < 1e-12);

        let retain_none = minimize_winning_prefix(
            &artifact,
            PirMcmcPrefixMinimizeOptions {
                retained_win_fraction: 0.0,
            },
        )
        .unwrap();
        assert_eq!(retain_none.accepted_rewrite_count, 0);
        assert_eq!(retain_none.witness_fn.name, "origin");
        assert_eq!(retain_none.witness_metric, 100);
    }

    #[test]
    fn prefix_minimization_rejects_invalid_fraction_and_non_winning_artifacts() {
        let artifact = manual_prefix_artifact();
        for retained_win_fraction in [f64::NAN, -0.1, 1.1] {
            let err = minimize_winning_prefix(
                &artifact,
                PirMcmcPrefixMinimizeOptions {
                    retained_win_fraction,
                },
            )
            .unwrap_err();
            assert!(
                err.to_string().contains("retained_win_fraction"),
                "unexpected error: {err}"
            );
        }

        let mut no_win = artifact.clone();
        no_win.raw_winner_cost = no_win.origin_cost;
        let err = minimize_winning_prefix(
            &no_win,
            PirMcmcPrefixMinimizeOptions {
                retained_win_fraction: 0.5,
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("positive objective win"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn frontier_schedule_validates_step_and_max() {
        let opts = PirMcmcBudgetFrontierOptions {
            budget_step: 4,
            max_rewrites: 16,
            rollouts_per_budget: 1,
            seed: 1,
            witness_kind_boost: PirMcmcBudgetFrontierOptions::DEFAULT_WITNESS_KIND_BOOST,
            proposal_attempts_per_rewrite:
                PirMcmcBudgetFrontierOptions::DEFAULT_PROPOSAL_ATTEMPTS_PER_REWRITE,
        };
        assert_eq!(frontier_budgets(opts).unwrap(), vec![4, 8, 12, 16]);
        assert_eq!(
            frontier_budgets(PirMcmcBudgetFrontierOptions {
                max_rewrites: 10,
                ..opts
            })
            .unwrap(),
            vec![4, 8, 10]
        );
        assert!(
            frontier_budgets(PirMcmcBudgetFrontierOptions {
                budget_step: 0,
                ..opts
            })
            .is_err()
        );
        assert!(
            frontier_budgets(PirMcmcBudgetFrontierOptions {
                max_rewrites: 0,
                ..opts
            })
            .is_err()
        );
        assert!(
            frontier_budgets(PirMcmcBudgetFrontierOptions {
                budget_step: 8,
                max_rewrites: 4,
                ..opts
            })
            .is_err()
        );
    }

    #[test]
    fn win_percent_vs_origin_reports_percentage_points() {
        assert!(
            (win_percent_vs_origin_for_metric(1205, 1173) - 2.655_601_659_751_037).abs() < 1e-12
        );
    }

    #[test]
    fn witness_guided_weights_favor_lineage_kinds_without_excluding_others() {
        #[derive(Debug)]
        struct KindOnlyTransform(PirTransformKind);
        impl PirTransform for KindOnlyTransform {
            fn kind(&self) -> PirTransformKind {
                self.0
            }

            fn find_candidates(&mut self, _f: &IrFn) -> Vec<crate::transforms::TransformCandidate> {
                Vec::new()
            }

            fn apply(
                &self,
                _f: &mut IrFn,
                _loc: &crate::transforms::TransformLocation,
            ) -> Result<(), String> {
                Ok(())
            }
        }

        let artifact = manual_prefix_artifact();
        let transforms: Vec<Box<dyn PirTransform>> = vec![
            Box::new(KindOnlyTransform(PirTransformKind::NotNotCancel)),
            Box::new(KindOnlyTransform(PirTransformKind::CmpSwap)),
        ];
        let weights = build_witness_guided_transform_weights(&transforms, &artifact, 4.0);
        assert_eq!(weights, vec![5.0, 1.0]);
    }

    #[test]
    fn frontier_reports_prefix_baseline_and_carries_guided_points_forward() {
        let artifact = manual_prefix_artifact();
        let opts = PirMcmcBudgetFrontierOptions {
            budget_step: 1,
            max_rewrites: 3,
            rollouts_per_budget: 1,
            seed: 7,
            witness_kind_boost: PirMcmcBudgetFrontierOptions::DEFAULT_WITNESS_KIND_BOOST,
            proposal_attempts_per_rewrite:
                PirMcmcBudgetFrontierOptions::DEFAULT_PROPOSAL_ATTEMPTS_PER_REWRITE,
        };
        let result = search_winning_budget_frontier_with_rollout(
            &artifact,
            opts,
            |budget, _, _, origin_metric, winner_metric| {
                let (state_name, cost) = match budget {
                    1 => ("guided1", cost_with_pir_nodes(95)),
                    2 => ("guided2", cost_with_pir_nodes(80)),
                    3 => ("guided3", cost_with_pir_nodes(85)),
                    _ => unreachable!(),
                };
                Ok(make_budget_witness(
                    renamed_state(&artifact.origin_fn, state_name),
                    cost,
                    budget,
                    Objective::Nodes,
                    origin_metric,
                    winner_metric,
                ))
            },
        )
        .unwrap();

        assert_eq!(result.points.len(), 3);
        assert_eq!(result.points[0].prefix_baseline.metric, 90);
        assert_eq!(result.points[1].prefix_baseline.metric, 70);
        assert_eq!(result.points[2].prefix_baseline.metric, 50);
        assert_eq!(result.points[0].guided.metric, 95);
        assert_eq!(result.points[1].guided.metric, 80);
        assert_eq!(
            result.points[2].guided.metric, 80,
            "worse later searches must carry forward the prior frontier point"
        );
    }

    #[test]
    fn frontier_uses_origin_fallback_when_rollouts_do_not_improve() {
        let artifact = manual_prefix_artifact();
        let opts = PirMcmcBudgetFrontierOptions {
            budget_step: 2,
            max_rewrites: 4,
            rollouts_per_budget: 1,
            seed: 1,
            witness_kind_boost: PirMcmcBudgetFrontierOptions::DEFAULT_WITNESS_KIND_BOOST,
            proposal_attempts_per_rewrite:
                PirMcmcBudgetFrontierOptions::DEFAULT_PROPOSAL_ATTEMPTS_PER_REWRITE,
        };
        let result = search_winning_budget_frontier_with_rollout(
            &artifact,
            opts,
            |budget, _, _, origin_metric, winner_metric| {
                Ok(make_budget_witness(
                    renamed_state(&artifact.origin_fn, &format!("noop{budget}")),
                    artifact.origin_cost,
                    budget,
                    Objective::Nodes,
                    origin_metric,
                    winner_metric,
                ))
            },
        )
        .unwrap();
        assert_eq!(result.points[0].guided.accepted_rewrite_count, 0);
        assert_eq!(result.points[0].guided.metric, 100);
        assert_eq!(result.points[1].guided.accepted_rewrite_count, 0);
        assert_eq!(result.points[1].guided.metric, 100);
    }

    #[test]
    fn frontier_rejects_no_win_and_toggle_objectives() {
        let opts = PirMcmcBudgetFrontierOptions {
            budget_step: 1,
            max_rewrites: 1,
            rollouts_per_budget: 1,
            seed: 1,
            witness_kind_boost: PirMcmcBudgetFrontierOptions::DEFAULT_WITNESS_KIND_BOOST,
            proposal_attempts_per_rewrite:
                PirMcmcBudgetFrontierOptions::DEFAULT_PROPOSAL_ATTEMPTS_PER_REWRITE,
        };
        let mut no_win = manual_prefix_artifact();
        no_win.raw_winner_cost = no_win.origin_cost;
        assert!(search_winning_budget_frontier(&no_win, opts).is_err());

        let mut toggle_artifact = manual_prefix_artifact();
        toggle_artifact.run_options.objective = Objective::G8rNodesTimesDepthTimesToggles;
        toggle_artifact.raw_winner_cost.g8r_gate_output_toggles = 1;
        assert!(search_winning_budget_frontier(&toggle_artifact, opts).is_err());
    }

    #[test]
    fn artifact_api_rejects_unsupported_run_shapes() {
        let f = parse_fn(
            r#"fn f(x: bits[1] id=1) -> bits[1] {
  ret identity.2: bits[1] = identity(x, id=2)
}"#,
        );

        let mut multichain = test_run_options(Objective::Nodes);
        multichain.threads = 2;
        assert!(run_pir_mcmc_with_artifact(f.clone(), multichain).is_err());

        let mut constrained = test_run_options(Objective::G8rNodes);
        constrained.max_allowed_depth = Some(4);
        assert!(run_pir_mcmc_with_artifact(f.clone(), constrained).is_err());

        let mut implicit_constraint =
            test_run_options(Objective::G8rNodesTimesWeightedSwitchingNoDepthRegress);
        implicit_constraint.toggle_stimulus = Some(vec![
            IrValue::parse_typed("(bits[1]:0)").unwrap(),
            IrValue::parse_typed("(bits[1]:1)").unwrap(),
        ]);
        assert!(run_pir_mcmc_with_artifact(f, implicit_constraint).is_err());
    }

    #[test]
    fn artifact_run_is_deterministic_and_captures_raw_winning_lineage() {
        let f = parse_fn(
            r#"fn f(x: bits[8] id=1) -> bits[8] {
  dead: bits[8] = identity(x, id=2)
  ret live: bits[8] = identity(x, id=3)
}"#,
        );
        let mut opts = test_run_options(Objective::Nodes);
        opts.max_iters = 1;
        opts.seed = 7;
        let transforms1: Vec<Box<dyn PirTransform>> = vec![Box::new(RemoveDeadNodeTestTransform)];
        let transforms2: Vec<Box<dyn PirTransform>> = vec![Box::new(RemoveDeadNodeTestTransform)];

        let artifact1 =
            run_pir_mcmc_with_artifact_using_transforms(f.clone(), opts.clone(), transforms1)
                .unwrap()
                .artifact;
        let artifact2 = run_pir_mcmc_with_artifact_using_transforms(f, opts, transforms2)
            .unwrap()
            .artifact;

        assert!(
            artifact1.raw_winner_cost.pir_nodes < artifact1.origin_cost.pir_nodes,
            "expected a real node-count win in the deterministic fixture"
        );
        assert_eq!(
            artifact1.origin_fn.to_string(),
            artifact2.origin_fn.to_string()
        );
        assert_eq!(artifact1.origin_cost, artifact2.origin_cost);
        assert_eq!(artifact1.raw_winner_cost, artifact2.raw_winner_cost);
        assert_eq!(
            artifact1.raw_winner_fn.to_string(),
            artifact2.raw_winner_fn.to_string()
        );
        assert_eq!(
            artifact1.winning_lineage.len(),
            artifact2.winning_lineage.len()
        );
        assert!(
            !artifact1.winning_lineage.is_empty(),
            "expected a non-empty winning lineage"
        );
        let last1 = artifact1.winning_lineage.last().unwrap();
        let last2 = artifact2.winning_lineage.last().unwrap();
        assert_eq!(last1.cost, artifact1.raw_winner_cost);
        assert_eq!(last1.state.to_string(), artifact1.raw_winner_fn.to_string());
        assert_eq!(last1.cost, last2.cost);
        assert_eq!(last1.state.to_string(), last2.state.to_string());
        assert_eq!(last1.transform_kind, last2.transform_kind);
    }

    #[test]
    fn artifact_lineage_can_be_minimized_end_to_end() {
        let f = parse_fn(
            r#"fn f(x: bits[8] id=1) -> bits[8] {
  dead_a: bits[8] = identity(x, id=2)
  dead_b: bits[8] = identity(x, id=3)
  ret live: bits[8] = identity(x, id=4)
}"#,
        );
        let mut opts = test_run_options(Objective::Nodes);
        opts.max_iters = 2;
        let transforms: Vec<Box<dyn PirTransform>> = vec![Box::new(RemoveDeadNodeTestTransform)];

        let artifact = run_pir_mcmc_with_artifact_using_transforms(f, opts, transforms)
            .unwrap()
            .artifact;
        assert_eq!(artifact.origin_cost.pir_nodes, 5);
        assert_eq!(artifact.raw_winner_cost.pir_nodes, 3);
        assert_eq!(artifact.winning_lineage.len(), 2);

        let minimized = minimize_winning_prefix(
            &artifact,
            PirMcmcPrefixMinimizeOptions {
                retained_win_fraction: 0.5,
            },
        )
        .unwrap();
        assert_eq!(minimized.accepted_rewrite_count, 1);
        assert_eq!(minimized.original_winning_lineage_len, 2);
        assert_eq!(minimized.origin_metric, 5);
        assert_eq!(minimized.winner_metric, 3);
        assert_eq!(minimized.witness_metric, 4);
        assert!((minimized.actual_retained_win_fraction - 0.5).abs() < 1e-12);
    }

    #[test]
    fn durable_artifact_round_trips_and_minimizes_identically() {
        let pkg = parse_pkg(
            r#"package sample

top fn f(x: bits[8] id=1) -> bits[8] {
  dead_a: bits[8] = identity(x, id=2)
  dead_b: bits[8] = identity(x, id=3)
  ret live: bits[8] = identity(x, id=4)
}
"#,
        );
        let f = pkg.get_fn("f").unwrap().clone();
        let mut opts = test_run_options(Objective::Nodes);
        opts.max_iters = 2;
        let transforms: Vec<Box<dyn PirTransform>> = vec![Box::new(RemoveDeadNodeTestTransform)];
        let artifact = run_pir_mcmc_with_artifact_using_transforms(f, opts, transforms)
            .unwrap()
            .artifact;
        let before = minimize_winning_prefix(
            &artifact,
            PirMcmcPrefixMinimizeOptions {
                retained_win_fraction: 0.5,
            },
        )
        .unwrap();

        let run_dir = tempdir().unwrap();
        write_pir_mcmc_artifact_dir(&artifact, &pkg, run_dir.path()).unwrap();
        let loaded = read_pir_mcmc_artifact_dir(run_dir.path()).unwrap();
        let after = minimize_winning_prefix(
            &loaded.artifact,
            PirMcmcPrefixMinimizeOptions {
                retained_win_fraction: 0.5,
            },
        )
        .unwrap();

        assert_eq!(loaded.top_fn_name, "f");
        assert_eq!(loaded.artifact.origin_cost, artifact.origin_cost);
        assert_eq!(
            loaded.artifact.origin_fn.to_string(),
            artifact.origin_fn.to_string()
        );
        assert_eq!(
            loaded.artifact.raw_winner_fn.to_string(),
            artifact.raw_winner_fn.to_string()
        );
        assert_eq!(
            loaded.artifact.winning_lineage.len(),
            artifact.winning_lineage.len()
        );
        assert_eq!(
            loaded.artifact.winning_lineage[0].transform_kind,
            artifact.winning_lineage[0].transform_kind
        );
        assert_eq!(after.accepted_rewrite_count, before.accepted_rewrite_count);
        assert_eq!(after.witness_metric, before.witness_metric);
        assert_eq!(after.witness_fn.to_string(), before.witness_fn.to_string());
    }

    #[test]
    fn durable_artifact_manifest_is_deterministic_for_fixed_run() {
        let pkg = parse_pkg(
            r#"package sample

top fn f(x: bits[8] id=1) -> bits[8] {
  dead_a: bits[8] = identity(x, id=2)
  dead_b: bits[8] = identity(x, id=3)
  ret live: bits[8] = identity(x, id=4)
}
"#,
        );
        let f = pkg.get_fn("f").unwrap().clone();
        let mut opts = test_run_options(Objective::Nodes);
        opts.max_iters = 2;
        let artifact1 = run_pir_mcmc_with_artifact_using_transforms(
            f.clone(),
            opts.clone(),
            vec![Box::new(RemoveDeadNodeTestTransform)],
        )
        .unwrap()
        .artifact;
        let artifact2 = run_pir_mcmc_with_artifact_using_transforms(
            f,
            opts,
            vec![Box::new(RemoveDeadNodeTestTransform)],
        )
        .unwrap()
        .artifact;

        let run_dir1 = tempdir().unwrap();
        let run_dir2 = tempdir().unwrap();
        let artifact_dir1 = write_pir_mcmc_artifact_dir(&artifact1, &pkg, run_dir1.path()).unwrap();
        let artifact_dir2 = write_pir_mcmc_artifact_dir(&artifact2, &pkg, run_dir2.path()).unwrap();
        let manifest1 =
            fs::read_to_string(artifact_dir1.join(PIR_MCMC_ARTIFACT_MANIFEST_FILE)).unwrap();
        let manifest2 =
            fs::read_to_string(artifact_dir2.join(PIR_MCMC_ARTIFACT_MANIFEST_FILE)).unwrap();
        assert_eq!(manifest1, manifest2);
    }

    #[test]
    fn durable_artifact_rejects_malformed_or_incomplete_manifests() {
        let malformed_dir = tempdir().unwrap();
        let malformed_artifact_dir = malformed_dir.path().join(PIR_MCMC_ARTIFACT_DIR_NAME);
        fs::create_dir_all(&malformed_artifact_dir).unwrap();
        fs::write(
            malformed_artifact_dir.join(PIR_MCMC_ARTIFACT_MANIFEST_FILE),
            "{not json",
        )
        .unwrap();
        assert!(read_pir_mcmc_artifact_dir(malformed_dir.path()).is_err());

        let incomplete_dir = tempdir().unwrap();
        let incomplete_artifact_dir = incomplete_dir.path().join(PIR_MCMC_ARTIFACT_DIR_NAME);
        fs::create_dir_all(&incomplete_artifact_dir).unwrap();
        fs::write(
            incomplete_artifact_dir.join(PIR_MCMC_ARTIFACT_MANIFEST_FILE),
            r#"{
  "schema_version": 1,
  "top_fn_name": "f",
  "run_options": {
    "max_iters": 1,
    "threads": 1,
    "chain_strategy": "independent",
    "checkpoint_iters": 1,
    "progress_iters": 0,
    "seed": 1,
    "initial_temperature": 1.0,
    "objective": "nodes",
    "extension_costing_mode": "preserve",
    "max_allowed_depth": null,
    "max_allowed_area": null,
    "switching_beta1": 1.0,
    "switching_beta2": 0.0,
    "switching_primary_output_load": 1.0,
    "enable_formal_oracle": false
  },
  "origin": {"file": "states/origin.ir", "cost": {"pir_nodes": 2, "g8r_nodes": 2, "g8r_depth": 2, "g8r_le_graph_milli": 0, "g8r_gate_output_toggles": 0, "g8r_weighted_switching_milli": 0}},
  "raw_winner": {"file": "states/raw-winner.ir", "cost": {"pir_nodes": 1, "g8r_nodes": 1, "g8r_depth": 1, "g8r_le_graph_milli": 0, "g8r_gate_output_toggles": 0, "g8r_weighted_switching_milli": 0}},
  "winning_lineage": []
}"#,
        )
        .unwrap();
        assert!(read_pir_mcmc_artifact_dir(incomplete_dir.path()).is_err());
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
        let mut baseline_cache = EvalFnBaselineResults::default();
        assert!(!pir_equiv_oracle(
            &orig_fn,
            &rewired_fn,
            &mut rng,
            4,
            /* enable_formal_oracle= */ false,
            &mut baseline_cache,
        ));
    }

    #[test]
    fn pir_equiv_oracle_rejects_zero_length_array_params_without_panic() {
        let ir_text = r#"fn zero_len(a: bits[8][0] id=10) -> bits[1] {
  ret literal.20: bits[1] = literal(value=0, id=20)
}"#;
        let mut parser1 = ir_parser::Parser::new(ir_text);
        let lhs = parser1.parse_fn().unwrap();
        let mut parser2 = ir_parser::Parser::new(ir_text);
        let rhs = parser2.parse_fn().unwrap();

        let mut rng = Pcg64Mcg::seed_from_u64(1);
        let mut baseline_cache = EvalFnBaselineResults::default();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pir_equiv_oracle(
                &lhs,
                &rhs,
                &mut rng,
                4,
                /* enable_formal_oracle= */ false,
                &mut baseline_cache,
            )
        }));

        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn optimize_pir_fn_via_xls_preserves_ext_nary_add_arch_via_ffi_wrappers() {
        let mut parser = ir_parser::Parser::new(
            r#"fn f(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  ret sum: bits[8] = ext_nary_add(a, b, c, signed=[false, false, false], negated=[false, false, false], arch=brent_kung, id=4)
}"#,
        );
        let f = parser.parse_fn().unwrap();

        let optimized =
            optimize_pir_fn_via_xls_with_extension_mode(&f, ExtensionCostingMode::Preserve)
                .unwrap();
        let ext_nodes = optimized
            .nodes
            .iter()
            .filter(|node| {
                matches!(
                    &node.payload,
                    NodePayload::ExtNaryAdd {
                        arch: Some(ExtNaryAddArchitecture::BrentKung),
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            ext_nodes, 1,
            "expected optimized PIR to reconstruct brent_kung ext_nary_add:\n{}",
            optimized
        );
    }

    #[test]
    fn optimize_pir_fn_via_xls_can_desugar_ext_nary_add_to_standard_ir() {
        let mut parser = ir_parser::Parser::new(
            r#"fn f(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  ret sum: bits[8] = ext_nary_add(a, b, c, signed=[false, false, false], negated=[false, false, false], arch=brent_kung, id=4)
}"#,
        );
        let f = parser.parse_fn().unwrap();

        let optimized =
            optimize_pir_fn_via_xls_with_extension_mode(&f, ExtensionCostingMode::Desugar).unwrap();
        let ext_nodes = optimized
            .nodes
            .iter()
            .filter(|node| matches!(&node.payload, NodePayload::ExtNaryAdd { .. }))
            .count();
        assert_eq!(
            ext_nodes, 0,
            "expected desugared optimized PIR to contain no ext_nary_add nodes:\n{}",
            optimized
        );
        assert!(
            !optimized.to_string().contains("ext_nary_add"),
            "expected desugared optimized PIR text to contain no extension op spelling:\n{}",
            optimized
        );

        let cost = cost_with_effort_options_toggle_stimulus_and_extension_mode(
            &f,
            Objective::G8rNodes,
            None,
            &WeightedSwitchingOptions::default(),
            ExtensionCostingMode::Desugar,
        )
        .unwrap();
        assert!(cost.g8r_nodes > 0);
    }

    #[test]
    fn get_pir_transforms_for_run_prunes_unsafe_only_classes_without_formal_oracle() {
        let no_oracle_kinds: HashSet<PirTransformKind> =
            get_pir_transforms_for_run(/* enable_formal_oracle= */ false)
                .into_iter()
                .map(|t| t.kind())
                .collect();
        let oracle_kinds: HashSet<PirTransformKind> =
            get_pir_transforms_for_run(/* enable_formal_oracle= */ true)
                .into_iter()
                .map(|t| t.kind())
                .collect();

        assert!(!no_oracle_kinds.contains(&PirTransformKind::ShiftHoist));
        assert!(!no_oracle_kinds.contains(&PirTransformKind::MaskOperandHighBit));
        assert!(!no_oracle_kinds.contains(&PirTransformKind::RewireOperandToSameType));
        assert!(!no_oracle_kinds.contains(&PirTransformKind::GuardedPredicateRewire));
        assert!(no_oracle_kinds.contains(&PirTransformKind::AbsorbAddOperandIntoExtNaryAdd));
        assert!(no_oracle_kinds.contains(&PirTransformKind::AddToExtNaryAdd));
        assert!(no_oracle_kinds.contains(&PirTransformKind::ReduceSelDistribute));

        assert!(oracle_kinds.contains(&PirTransformKind::ShiftHoist));
        assert!(oracle_kinds.contains(&PirTransformKind::MaskOperandHighBit));
        assert!(oracle_kinds.contains(&PirTransformKind::RewireOperandToSameType));
        assert!(oracle_kinds.contains(&PirTransformKind::GuardedPredicateRewire));
    }

    #[test]
    fn ext_nary_add_arch_reaches_g8r_tags_after_xls_round_trip() {
        let mut parser = ir_parser::Parser::new(
            r#"fn f(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  ret sum: bits[8] = ext_nary_add(a, b, c, signed=[false, false, false], negated=[false, false, false], arch=brent_kung, id=4)
}"#,
        );
        let f = parser.parse_fn().unwrap();

        let optimized =
            optimize_pir_fn_via_xls_with_extension_mode(&f, ExtensionCostingMode::Preserve)
                .unwrap();
        let ext_text_id = optimized
            .nodes
            .iter()
            .find_map(|node| match &node.payload {
                NodePayload::ExtNaryAdd {
                    arch: Some(ExtNaryAddArchitecture::BrentKung),
                    ..
                } => Some(node.text_id),
                _ => None,
            })
            .expect("expected reconstructed brent_kung ext_nary_add");

        let gatify_output = ir2gate::gatify(
            &optimized,
            GatifyOptions {
                fold: true,
                hash: true,
                check_equivalence: false,
                adder_mapping: AdderMapping::default(),
                array_index_lowering_strategy: Default::default(),
                mul_adder_mapping: None,
                range_info: None,
                enable_rewrite_carry_out: false,
                enable_rewrite_prio_encode: false,
                enable_rewrite_nary_add: false,
                enable_rewrite_mask_low: false,
            },
        )
        .unwrap();
        let gate_fn_text = gatify_output.gate_fn.to_string();
        assert!(
            gate_fn_text.contains(&format!(
                "ext_nary_add_{}_brent_kung_output_bit_",
                ext_text_id
            )),
            "expected gatify tags to reflect the ext_nary_add arch after XLS round-trip:\n{}",
            gate_fn_text
        );
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
    fn objective_metric_graph_logical_effort_times_nodes_multiplies_nodes() {
        let c = Cost {
            pir_nodes: 0,
            g8r_nodes: 7,
            g8r_depth: 11,
            g8r_le_graph_milli: 13,
            g8r_gate_output_toggles: 0,
            g8r_weighted_switching_milli: 0,
        };
        assert_eq!(Objective::G8rLeGraphTimesNodes.metric(&c), 91);
    }

    #[test]
    fn objective_metric_graph_logical_effort_times_nodes_handles_large_values() {
        let c = Cost {
            pir_nodes: 0,
            g8r_nodes: usize::MAX,
            g8r_depth: 0,
            g8r_le_graph_milli: usize::MAX,
            g8r_gate_output_toggles: 0,
            g8r_weighted_switching_milli: 0,
        };
        assert_eq!(
            Objective::G8rLeGraphTimesNodes.metric(&c),
            (usize::MAX as u128) * (usize::MAX as u128)
        );
    }

    #[test]
    fn graph_logical_effort_times_nodes_uses_graph_effort_without_toggles() {
        let objective = Objective::G8rLeGraphTimesNodes;
        assert!(objective.uses_g8r_costing());
        assert!(objective.needs_graph_logical_effort());
        assert!(!objective.needs_toggle_stimulus());
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
    fn search_score_prefers_feasible_then_lower_violation_then_objective() {
        let feasible = search_score(
            &Cost {
                pir_nodes: 0,
                g8r_nodes: 10,
                g8r_depth: 10,
                g8r_le_graph_milli: 0,
                g8r_gate_output_toggles: 0,
                g8r_weighted_switching_milli: 0,
            },
            Objective::G8rNodes,
            ConstraintLimits {
                max_delay: Some(12),
                max_area: None,
            },
        );
        let mildly_infeasible = search_score(
            &Cost {
                pir_nodes: 0,
                g8r_nodes: 10,
                g8r_depth: 13,
                g8r_le_graph_milli: 0,
                g8r_gate_output_toggles: 0,
                g8r_weighted_switching_milli: 0,
            },
            Objective::G8rNodes,
            ConstraintLimits {
                max_delay: Some(12),
                max_area: None,
            },
        );
        let badly_infeasible = search_score(
            &Cost {
                pir_nodes: 0,
                g8r_depth: 16,
                g8r_nodes: 10,
                g8r_le_graph_milli: 0,
                g8r_gate_output_toggles: 0,
                g8r_weighted_switching_milli: 0,
            },
            Objective::G8rNodes,
            ConstraintLimits {
                max_delay: Some(12),
                max_area: None,
            },
        );

        assert!(feasible < mildly_infeasible);
        assert!(mildly_infeasible < badly_infeasible);
    }

    #[test]
    fn effective_constraint_limits_respects_non_regressing_depth() {
        let initial = Cost {
            pir_nodes: 0,
            g8r_nodes: 10,
            g8r_depth: 17,
            g8r_le_graph_milli: 0,
            g8r_gate_output_toggles: 0,
            g8r_weighted_switching_milli: 0,
        };
        let got = effective_constraint_limits(
            Objective::G8rNodesTimesWeightedSwitchingNoDepthRegress,
            ConstraintLimits {
                max_delay: Some(20),
                max_area: None,
            },
            &initial,
        );
        assert_eq!(got.max_delay, Some(17));
        assert_eq!(got.max_area, None);
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
            extension_costing_mode: ExtensionCostingMode::Preserve,
            max_allowed_depth: None,
            max_allowed_area: None,
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
            extension_costing_mode: ExtensionCostingMode::Preserve,
            max_allowed_depth: None,
            max_allowed_area: None,
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
    fn run_pir_mcmc_rejects_caps_with_nodes_objective() {
        let mut parser = ir_parser::Parser::new(
            r#"fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(a, b, id=3)
}"#,
        );
        let f = parser.parse_fn().unwrap();

        let opts = RunOptions {
            max_iters: 1,
            threads: 1,
            chain_strategy: ChainStrategy::Independent,
            checkpoint_iters: 1,
            progress_iters: 0,
            seed: 1,
            initial_temperature: 1.0,
            objective: Objective::Nodes,
            extension_costing_mode: ExtensionCostingMode::Preserve,
            max_allowed_depth: Some(10),
            max_allowed_area: None,
            weighted_switching_options: WeightedSwitchingOptions::default(),
            enable_formal_oracle: false,
            trajectory_dir: None,
            toggle_stimulus: None,
        };
        assert!(run_pir_mcmc(f, opts).is_err());
    }

    #[test]
    fn run_pir_mcmc_rejects_dual_caps() {
        let mut parser = ir_parser::Parser::new(
            r#"fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(a, b, id=3)
}"#,
        );
        let f = parser.parse_fn().unwrap();

        let opts = RunOptions {
            max_iters: 1,
            threads: 1,
            chain_strategy: ChainStrategy::Independent,
            checkpoint_iters: 1,
            progress_iters: 0,
            seed: 1,
            initial_temperature: 1.0,
            objective: Objective::G8rNodes,
            extension_costing_mode: ExtensionCostingMode::Preserve,
            max_allowed_depth: Some(10),
            max_allowed_area: Some(10),
            weighted_switching_options: WeightedSwitchingOptions::default(),
            enable_formal_oracle: false,
            trajectory_dir: None,
            toggle_stimulus: None,
        };
        assert!(run_pir_mcmc(f, opts).is_err());
    }

    #[test]
    fn run_pir_mcmc_rejects_area_cap_with_non_regressing_depth_objective() {
        let mut parser = ir_parser::Parser::new(
            r#"fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(a, b, id=3)
}"#,
        );
        let f = parser.parse_fn().unwrap();

        let opts = RunOptions {
            max_iters: 1,
            threads: 1,
            chain_strategy: ChainStrategy::Independent,
            checkpoint_iters: 1,
            progress_iters: 0,
            seed: 1,
            initial_temperature: 1.0,
            objective: Objective::G8rNodesTimesWeightedSwitchingNoDepthRegress,
            extension_costing_mode: ExtensionCostingMode::Preserve,
            max_allowed_depth: None,
            max_allowed_area: Some(10),
            weighted_switching_options: WeightedSwitchingOptions::default(),
            enable_formal_oracle: false,
            trajectory_dir: None,
            toggle_stimulus: Some(vec![
                IrValue::parse_typed("(bits[1]:0, bits[1]:0)").unwrap(),
                IrValue::parse_typed("(bits[1]:1, bits[1]:1)").unwrap(),
            ]),
        };
        assert!(run_pir_mcmc(f, opts).is_err());
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

    #[test]
    fn trajectory_json_emits_transform_mechanism() {
        let mut parser = ir_parser::Parser::new(
            r#"fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(a, b, id=3)
}"#,
        );
        let f = parser.parse_fn().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();
        let opts = RunOptions {
            max_iters: 1,
            threads: 1,
            chain_strategy: ChainStrategy::Independent,
            checkpoint_iters: 100,
            progress_iters: 0,
            seed: 1,
            initial_temperature: 1.0,
            objective: Objective::Nodes,
            extension_costing_mode: ExtensionCostingMode::Preserve,
            max_allowed_depth: None,
            max_allowed_area: None,
            weighted_switching_options: WeightedSwitchingOptions::default(),
            enable_formal_oracle: false,
            trajectory_dir: Some(temp_dir.path().to_path_buf()),
            toggle_stimulus: None,
        };

        let _ = run_pir_mcmc(f, opts).unwrap();
        let path = temp_dir.path().join("trajectory.c000.jsonl");
        let text = std::fs::read_to_string(path).unwrap();
        let first_line = text.lines().next().expect("trajectory line");
        let value: serde_json::Value = serde_json::from_str(first_line).unwrap();
        assert!(
            value.get("transform_mechanism").is_some(),
            "expected transform_mechanism in trajectory JSON: {first_line}"
        );
    }
}
