// SPDX-License-Identifier: Apache-2.0

//! Experimental timing checks on lowered, unoptimized XLS Block IR.
//!
//! A register's stage describes its normal update path. Holding or resetting
//! the register does not guarantee fresh data or a particular transaction.

use std::collections::BTreeMap;
use xlsynth_pir::ir::{Block, NodePayload, Unop};
use xlsynth_pir::ir_utils::{TopologicalSortError, get_topological_with_extra_dependencies};

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

impl Timing {
    /// Combine values used together: cycles must agree, and constants fit any
    /// cycle.
    fn join(a: Option<Self>, b: Option<Self>) -> Result<Option<Self>, (u64, u64)> {
        match (a, b) {
            // x@0 and y@1 cannot be used by the same operation.
            (Some(Self::Cycle(a)), Some(Self::Cycle(b))) if a != b => Err((a, b)),
            // x@2 + 1 is @2; combining two constants stays constant.
            (Some(Self::Constant), other) | (other, Some(Self::Constant)) => Ok(other),
            // Both cycles are known and equal.
            (Some(a), Some(_)) => Ok(Some(a)),
            // At least one value has unknown timing.
            _ => Ok(None),
        }
    }
}

/// If q captures a value at @N, reading q gives @N+1.
pub(super) fn register_cycle(register: &str, input_cycle: u64) -> Result<u64, String> {
    input_cycle
        .checked_add(1)
        .ok_or_else(|| format!("cycle offset overflow at register '{register}'"))
}

/// Check whether a block's operations and outputs have the promised timing.
/// The block must already have passed PIR's structural verification.
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

    // Build the read/write lookup once for all registers in this block.
    let registers = block.register_accesses();
    let active_low_reset_port = block
        .reset
        .as_ref()
        .filter(|reset| reset.active_low)
        .map(|reset| reset.port);
    let mut result = CheckResult::Checked;

    // A register read has no operands in Block IR. For timing, it also needs
    // its write to have been visited: q := x + y must check the addition
    // before assigning q its cycle. This adds an edge only to our traversal.
    let write_dependency = |node_ref| match &block.get_node(node_ref).payload {
        NodePayload::RegisterRead { register } => registers.unique_write(register).ok(),
        _ => None,
    };
    let order = match get_topological_with_extra_dependencies(&block.graph, write_dependency) {
        Ok(order) => order,
        // Feedback such as q := q + x has no fixed dependency order. We
        // cannot check the rest of the block until we can order its nodes.
        Err(TopologicalSortError::Cycle { .. }) => return Ok(CheckResult::Unsupported),
        Err(error) => return Err(error.to_string()),
    };

    for node_ref in order {
        let node = block.get_node(node_ref);

        timings[node_ref.index] = match &node.payload {
            // `u32:1` is available in every cycle.
            NodePayload::Literal(_) => Some(Timing::Constant),
            NodePayload::Unop(Unop::Not, arg) if Some(*arg) == active_low_reset_port => {
                // BX normalizes an active-low reset with !rst_n. This is
                // still a reset control, not a value with a pipeline cycle.
                // Leave its timing unknown if ordinary logic uses it.
                continue;
            }
            // Addition, comparison, and other two-input operations all need
            // their changing inputs to belong to the same cycle.
            NodePayload::Binop(_, lhs, rhs) => Timing::join(timings[lhs.index], timings[rhs.index])
                .map_err(|(a, b)| {
                    format!(
                        "timing mismatch at {}.{}: operands are at stages {a} and {b}",
                        node.payload.get_operator(),
                        node.text_id
                    )
                })?,
            // Nested `if` guards become an AND of their conditions. AND, OR,
            // and XOR can have any number of operands; all must align.
            NodePayload::Nary(_, operands) => operands
                .iter()
                .try_fold(Some(Timing::Constant), |cycle, operand| {
                    Timing::join(cycle, timings[operand.index])
                })
                .map_err(|(a, b)| {
                    format!(
                        "timing mismatch at {}.{}: operands are at stages {a} and {b}",
                        node.payload.get_operator(),
                        node.text_id
                    )
                })?,
            NodePayload::RegisterRead { register } => {
                let Ok(write) = registers.unique_write(register) else {
                    // We need exactly one write. Multiple enabled writes are
                    // valid hardware, but we cannot check them yet.
                    result = CheckResult::Unsupported;
                    continue;
                };
                let NodePayload::RegisterWrite {
                    arg,
                    load_enable,
                    reset,
                    ..
                } = &block.get_node(write).payload
                else {
                    unreachable!("register lookup returned a non-write node");
                };

                // No reset is fine, and the block's designated reset port
                // does not need to align with the data. A different signal,
                // such as rst | error, may have timing requirements we do
                // not know how to check yet.
                if let Some(signal) = reset
                    && block.reset.as_ref().map(|reset| reset.port) != Some(*signal)
                {
                    result = CheckResult::Unsupported;
                    continue;
                }

                let data = timings[arg.index];
                let capture = match load_enable {
                    // `if en { q := x; }`: en and x meet before the edge.
                    Some(enable) => Timing::join(data, timings[enable.index]).map_err(
                        |(data, enable)| {
                            format!(
                                "write to '{register}' has enable at stage {enable} and data at stage {data}"
                            )
                        },
                    )?,
                    None => data,
                };

                match capture {
                    // If q captures x@2, q is @3.
                    Some(Timing::Cycle(cycle)) => {
                        Some(Timing::Cycle(register_cycle(register, cycle)?))
                    }
                    // A constant alone gives us no starting cycle to count
                    // from, and an unknown input stays unknown.
                    Some(Timing::Constant) | None => None,
                }
            }
            NodePayload::RegisterWrite { .. } => {
                // A verified block has one read for every register. That read
                // checks this write's data and adds the register's cycle.
                continue;
            }
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
