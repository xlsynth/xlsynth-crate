// SPDX-License-Identifier: Apache-2.0

//! Shared PIR block evaluation helpers for random-block fuzz targets.

use xlsynth_pir::block_eval::{BlockEvaluator, BlockState};
use xlsynth_pir::ir::{Block, NodeRef, Type};
use xlsynth_pir::ir_eval::{EvalObserver, SelectEvent};
use xlsynth_pir::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type;
use xlsynth_pir::{IrBits, IrValue};

/// Cycle results with sampled values indexed by node in the original block.
pub struct ObservedBlockCycle {
    pub outputs: Vec<IrValue>,
    pub next_state: BlockState,
    pub node_values: Vec<Option<IrValue>>,
}

/// Records nodes for range analysis and behavioral coverage.
struct NodeValues(Vec<Option<IrValue>>);

impl EvalObserver for NodeValues {
    fn on_select(&mut self, _event: SelectEvent) {
        // Consumers classify select behavior from the original operand values.
    }

    fn on_node_value(&mut self, node_ref: NodeRef, _id: usize, value: &IrValue) {
        self.0[node_ref.index] = Some(value.clone());
    }
}

/// Retains node values in the original top block for one sampled cycle.
pub fn evaluate_block_cycle_observed(
    evaluator: &BlockEvaluator,
    block: &Block,
    inputs: &[IrValue],
    state: &BlockState,
) -> ObservedBlockCycle {
    let mut observer = NodeValues(vec![None; block.nodes.len()]);
    let cycle = evaluator
        .step_with_observer(state, inputs, &mut observer)
        .unwrap_or_else(|err| panic!("generated block evaluation failed: {err}\n{block}"));
    ObservedBlockCycle {
        outputs: cycle.outputs,
        next_state: cycle.next_state,
        node_values: observer.0,
    }
}

/// Returns visible block output values in port declaration order.
pub fn block_output_refs(block: &Block) -> Vec<NodeRef> {
    block
        .output_ports()
        .map(|port| block.output_value(port))
        .collect()
}

/// Returns visible block output types in port declaration order.
pub fn block_output_types(block: &Block) -> Vec<&Type> {
    block
        .output_ports()
        .map(|port| block.port_type(port))
        .collect()
}

/// Flattens one generated PIR value into the packed LSB-first AIG convention.
pub fn flatten_value(value: &IrValue, ty: &Type) -> IrBits {
    let mut bits = Vec::with_capacity(ty.bit_count());
    flatten_ir_value_to_lsb0_bits_for_type(value, ty, &mut bits)
        .expect("generated value should match its PIR type");
    IrBits::from_lsb_is_0(&bits)
}
