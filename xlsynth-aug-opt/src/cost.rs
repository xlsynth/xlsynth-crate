// SPDX-License-Identifier: Apache-2.0

//! Profitability costs for the optimizer's explicit local regions.

use xlsynth_g8r::aig::dce::dce;
use xlsynth_g8r::aig::gate::{AigNode, GateFn};
use xlsynth_g8r::aig::graph_logical_effort::{
    GraphLogicalEffortOptions, analyze_graph_logical_effort,
};
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, IrRegion, gatify_prepared_fn, gatify_region};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::process_ir_path::CanonicalG8rOptions;
use xlsynth_pir::ir;

use crate::constant_shift_choices::ShiftChoiceCostGraph;
use crate::ir_cost::{IrCost, ShiftChoiceCostEvaluator};

/// Measures the AND count and Graph LE of one constant-shift alternative.
///
/// The region's boundary is independent of external logic and loads. g8r lowers
/// its explicit node list and removes dead gates before these measurements.
pub struct GateBuilderCostEvaluator {
    options: GateBuilderOptions,
    graph_le_options: GraphLogicalEffortOptions,
}

impl Default for GateBuilderCostEvaluator {
    fn default() -> Self {
        let defaults = CanonicalG8rOptions::default();
        Self {
            options: GateBuilderOptions {
                fold: defaults.fold,
                hash: defaults.hash,
            },
            graph_le_options: GraphLogicalEffortOptions {
                beta1: defaults.graph_logical_effort_beta1,
                beta2: defaults.graph_logical_effort_beta2,
            },
        }
    }
}

impl GateBuilderCostEvaluator {
    /// Uses the caller's gate folding and sharing settings with this model.
    pub fn with_gate_builder_options(mut self, options: GateBuilderOptions) -> Self {
        self.options = options;
        self
    }
}

impl ShiftChoiceCostEvaluator for GateBuilderCostEvaluator {
    fn estimate(&mut self, graph: &ShiftChoiceCostGraph<'_>) -> Result<IrCost, String> {
        let gate_fn = gatify_region(
            IrRegion {
                function: graph.function(),
                nodes: graph.nodes(),
                inputs: graph.inputs(),
                outputs: graph.outputs(),
            },
            self.options,
        )?;
        measure_gate_cost(&gate_fn, &self.graph_le_options)
    }
}

/// Measures a lowered graph with a consistent no-logic delay convention.
fn measure_gate_cost(
    gate_fn: &GateFn,
    graph_le_options: &GraphLogicalEffortOptions,
) -> Result<IrCost, String> {
    let area = gate_fn
        .gates
        .iter()
        .filter(|node| matches!(node, AigNode::And2 { .. }))
        .count();
    let delay = if area == 0 {
        // The analyzer uses -1 when no logic gates remain.
        0.0
    } else {
        analyze_graph_logical_effort(gate_fn, graph_le_options).delay
    };
    if !delay.is_finite() || delay < 0.0 {
        return Err("gate cost requires finite nonnegative graph logical effort".to_string());
    }
    Ok(IrCost { area, delay })
}

/// Costs whole-function alternatives, retaining external loads and sharing.
///
/// The reference profile uses Brent-Kung mapping without range information,
/// gate DCE, and graph LE with beta1=1/beta2=0. Only private clones are
/// prepared; the optimizer's returned candidate remains ordinary XLS IR.
pub(crate) struct G8rFunctionCostEvaluator {
    gate_options: GateBuilderOptions,
}

impl Default for G8rFunctionCostEvaluator {
    fn default() -> Self {
        Self {
            gate_options: GateBuilderOptions {
                fold: true,
                hash: true,
            },
        }
    }
}

impl G8rFunctionCostEvaluator {
    pub(crate) fn with_gate_builder_options(mut self, options: GateBuilderOptions) -> Self {
        self.gate_options = options;
        self
    }

    pub(crate) fn estimate(&mut self, f: &ir::Fn) -> Result<IrCost, String> {
        let prepared = prep_for_gatify(f, None, PrepForGatifyOptions::all_opts_enabled());
        let mut options = GatifyOptions::all_opts_enabled();
        options.fold = self.gate_options.fold;
        options.hash = self.gate_options.hash;
        options.adder_mapping = AdderMapping::BrentKung;
        let graph = dce(&gatify_prepared_fn(&prepared, options)?.gate_fn);
        measure_gate_cost(
            &graph,
            &GraphLogicalEffortOptions {
                beta1: 1.0,
                beta2: 0.0,
            },
        )
    }
}
