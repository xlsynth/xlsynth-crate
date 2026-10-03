// SPDX-License-Identifier: Apache-2.0

//! Local shift-choice costing through the ordinary g8r node lowering.

use super::{GateEnv, GateOrVec, GatifyOptions, gatify_concat, gatify_node};
use crate::aig::dce::dce;
use crate::aig::gate::{AigBitVector, AigNode};
use crate::aig::graph_logical_effort::{GraphLogicalEffortOptions, analyze_graph_logical_effort};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::process_ir_path::CanonicalG8rOptions;
use xlsynth_pir::constant_shift_choices::ShiftChoiceCostGraph;
use xlsynth_pir::ir::NodePayload;
use xlsynth_pir::ir_cost::{IrCost, ShiftChoiceCostEvaluator};

/// Measures the AND count and Graph LE of a single shift-choice alternative.
///
/// The rewrite borrows a bounded region with shared boundary inputs and outputs
/// for retained values. Logic and loads outside that graph are not modeled.
/// No preparation, range analysis, or gate rewriting runs during costing.
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
        let f = graph.function();
        let mut builder = GateBuilder::new("shift_choice_cost".to_string(), self.options);
        let mut env = GateEnv::for_region(f, graph.nodes(), graph.outputs());
        // Structural peepholes must stay disabled: boundary nodes are
        // independent inputs even when their original payload is an operation.
        let options = GatifyOptions {
            fold: self.options.fold,
            hash: self.options.hash,
            ..GatifyOptions::all_opts_disabled()
        };
        for (index, &input) in graph.inputs().iter().enumerate() {
            let node = f.get_node(input);
            let bits = builder.add_input(format!("input_{index}"), node.ty.bit_count());
            env.add(input, GateOrVec::BitVector(bits));
        }
        // The listed nodes are in dependency order and exclude the boundary
        // inputs; no traversal of the enclosing function occurs.
        for &nr in graph.nodes() {
            let node = f.get_node(nr);
            let direct = if node.ty.bit_count() == 0 {
                Some(AigBitVector::zeros(0))
            } else {
                match &node.payload {
                    NodePayload::Sel { cases, default, .. } if cases.is_empty() => Some(
                        env.get_bit_vector(default.expect("a default-only select has a default"))?,
                    ),
                    NodePayload::Sel {
                        selector, cases, ..
                    } if f.get_node(*selector).ty.bit_count() == 0 => {
                        Some(env.get_bit_vector(cases[0])?)
                    }
                    _ => None, // Ordinary nodes use the shared lowering.
                }
            };
            if let Some(bits) = direct {
                env.add(nr, GateOrVec::BitVector(bits));
            } else {
                gatify_node(f, nr, node, &mut builder, &mut env, &options)?;
            }
        }
        let outputs = graph
            .outputs()
            .iter()
            .map(|&output| env.get_bit_vector(output))
            .collect::<Result<Vec<_>, _>>()?;
        builder.add_output("out".to_string(), gatify_concat(&outputs));
        // Discarded bits must not inflate fanout during Graph LE analysis.
        let gate_fn = dce(&builder.build());
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
