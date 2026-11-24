// SPDX-License-Identifier: Apache-2.0

//! Experimental timing checks on lowered, unoptimized XLS Block IR.

use std::collections::BTreeMap;
use xlsynth_pir::ir::{Block, NodePayload};
use xlsynth_pir::ir_utils::get_topological;

use crate::ast::OutputCycles;

/// The input and output cycles promised in the BX block's signature.
pub(crate) struct Contract<'a> {
    pub input_cycles: &'a BTreeMap<String, u64>,
    pub output_cycles: &'a OutputCycles,
}

/// Whether the experimental checker could verify this block's contract.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum CheckResult {
    /// We could check every operation, and every output matches its promise.
    Checked,
    /// We do not know enough to check the entire block yet.
    Unsupported,
}

/// Either a value belongs to a particular cycle, or it is a constant.
#[derive(Clone, Copy)]
enum Timing {
    Cycle(u64),
    Constant,
}

/// Check whether a block's operations and outputs have the promised timing.
pub(crate) fn check_block(block: &Block, contract: &Contract<'_>) -> Result<CheckResult, String> {
    let OutputCycles::Uniform(expected) = contract.output_cycles else {
        // For now we only support signatures like `-> u32 @2`.
        // Each output having its own cycle, or a cycle chosen by a function,
        // will need separate rules.
        return Ok(CheckResult::Unsupported);
    };

    // Give every IR node a slot. Start with None (we do not know its cycle),
    // then fill in the input cycles from annotations like `x: u32 @2`.
    let mut timings = vec![None; block.nodes.len()];
    for input in block.input_ports() {
        timings[input.index] = contract
            .input_cycles
            .get(block.port_name(input))
            .copied()
            .map(Timing::Cycle);
    }

    // Work out x and y before handling x + y. The graph's storage order does
    // not necessarily put them first, so ask PIR for dependency order.
    let mut result = CheckResult::Checked;
    for node_ref in get_topological(&block.graph) {
        let node = block.get_node(node_ref);
        timings[node_ref.index] = match &node.payload {
            // `u32:1` is available in every cycle.
            NodePayload::Literal(_) => Some(Timing::Constant),
            // Addition, comparison, and other two-input operations all need
            // their changing inputs to belong to the same cycle.
            NodePayload::Binop(_, lhs, rhs) => match (timings[lhs.index], timings[rhs.index]) {
                // `x@0 + y@1` mixes cycles, so reject it.
                (Some(Timing::Cycle(a)), Some(Timing::Cycle(b))) if a != b => {
                    return Err(format!(
                        "timing mismatch at {}.{}: operands are at stages {a} and {b}",
                        node.payload.get_operator(),
                        node.text_id
                    ));
                }
                // `x@2 + 1` has x's cycle. `1 + 2` remains a constant.
                (Some(Timing::Constant), other) | (other, Some(Timing::Constant)) => other,
                // We already rejected unequal cycles, so @2 + @2 gives @2.
                (Some(a), Some(_)) => Some(a),
                // Without both input timings, we cannot work out the result.
                _ => None,
            },
            NodePayload::Nil | NodePayload::InputPort { .. } | NodePayload::OutputPort { .. } => {
                // Nil is an empty graph slot (index 0, or a deleted node),
                // not hardware. Inputs were handled above; outputs are
                // checked below against the values connected to them.
                continue;
            }
            _ => {
                // We have not written a rule for this operation yet. Even if
                // nobody uses its result, we cannot say the whole block passed.
                result = CheckResult::Unsupported;
                continue;
            }
        };
        if timings[node_ref.index].is_none() {
            // An unchecked operation means we cannot report Checked, even if
            // the final output happens to have a known cycle.
            result = CheckResult::Unsupported;
        }
    }

    // For `output_port(add(x, y))`, check the addition's cycle. The output
    // port itself just connects that value to the outside of the block.
    let mut outputs = block.output_ports().peekable();
    if outputs.peek().is_none() {
        return Ok(CheckResult::Unsupported);
    }
    for output in outputs {
        let value = block.output_value(output);
        match timings[value.index] {
            // The signature promised @2, for example, but this value is @1.
            Some(Timing::Cycle(actual)) if actual != *expected => {
                return Err(format!(
                    "output '{}' is at stage {actual}, expected stage {expected}",
                    block.port_name(output)
                ));
            }
            Some(Timing::Cycle(_) | Timing::Constant) => {
                // This output matches its promise, or is a constant that can
                // be used in any cycle.
            }
            // Keep checking other outputs: one may have a definite error.
            None => result = CheckResult::Unsupported,
        }
    }
    Ok(result)
}
