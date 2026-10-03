// SPDX-License-Identifier: Apache-2.0

//! Profitability costs for the optimizer's explicit local regions.

use xlsynth_g8r::aig::gate::AigNode;
use xlsynth_g8r::aig::graph_logical_effort::{
    GraphLogicalEffortOptions, analyze_graph_logical_effort,
};
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::gatify::ir2gate::{IrRegion, gatify_region};
use xlsynth_g8r::process_ir_path::CanonicalG8rOptions;

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
        let area = gate_fn
            .gates
            .iter()
            .filter(|node| matches!(node, AigNode::And2 { .. }))
            .count();
        let delay = if area == 0 {
            // The shared analyzer uses -1 when there are no logic gates.
            0.0
        } else {
            analyze_graph_logical_effort(&gate_fn, &self.graph_le_options).delay
        };
        if !delay.is_finite() || delay < 0.0 {
            return Err(
                "local cost: Graph LE did not produce a finite nonnegative delay".to_string(),
            );
        }
        Ok(IrCost { area, delay })
    }
}
