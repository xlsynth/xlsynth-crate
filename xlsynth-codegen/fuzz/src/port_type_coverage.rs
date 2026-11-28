// SPDX-License-Identifier: Apache-2.0

//! Checked semantic coverage for the annotated-port fuzz target.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use xlsynth::external_tool::ToolError;
use xlsynth_pir::ir::{Block, NodePayload, NodeRef, Package};
use xlsynth_pir::ir_utils::operands;

use crate::coverage::{CoverageReport, Outcome, live_nodes};
use crate::semantics::Trace;

#[derive(Default)]
struct PortTypeCoverage {
    graph: CoverageReport,
    generated_port_types: BTreeMap<String, u64>,
    checked_port_types: BTreeMap<String, u64>,
    checked_input_type_consumers: BTreeMap<String, u64>,
    checked_output_type_drivers: BTreeMap<String, u64>,
    checked_output_chain_depths: BTreeMap<usize, u64>,
}

impl PortTypeCoverage {
    /// Counts only successful semantic checks as checked coverage; external
    /// tool resource interruptions remain generated-but-inconclusive cases.
    fn record(
        &mut self,
        package: &Package,
        choices: &[u8],
        trace: &Trace,
        result: &Result<(), ToolError>,
    ) -> bool {
        let outcome = match result {
            Ok(()) => Outcome::Checked,
            Err(error) if error.is_resource_failure() => Outcome::Inconclusive(error),
            // Ordinary tool failures are sample bugs, not inconclusive
            // coverage.
            Err(_) => return false,
        };
        let checked = result.is_ok();
        self.graph.record(package, outcome);
        if checked {
            self.graph.record_trace(trace);
        }

        let block = crate::top_block(package);
        let live = live_nodes(block);
        if checked {
            *self
                .checked_output_chain_depths
                .entry(longest_output_chain(block))
                .or_default() += 1;
        }
        for (index, port) in block
            .input_ports()
            .chain(block.output_ports())
            .filter(|port| block.port_type(*port).bit_count() != 0)
            .enumerate()
        {
            let kind = crate::dslx_types::type_kind(choices.get(index * 4).copied().unwrap_or(0));
            let direction = match &block.get_node(port).payload {
                NodePayload::InputPort { .. } => "input",
                NodePayload::OutputPort { .. } => "output",
                _ => unreachable!("block data port has an input/output payload"),
            };
            let key = format!("{direction}:{kind}");
            *self.generated_port_types.entry(key.clone()).or_default() += 1;
            if !checked {
                continue;
            }
            *self.checked_port_types.entry(key).or_default() += 1;
            if direction == "input" {
                for (node_index, node) in block.nodes.iter().enumerate().skip(1) {
                    if live[node_index] && operands(&node.payload).contains(&port) {
                        *self
                            .checked_input_type_consumers
                            .entry(format!("{kind} -> {}", node.payload.get_operator()))
                            .or_default() += 1;
                    }
                }
            } else {
                let driver = block.get_node(block.output_value(port));
                *self
                    .checked_output_type_drivers
                    .entry(format!("{kind} <- {}", driver.payload.get_operator()))
                    .or_default() += 1;
            }
        }
        true
    }

    /// Emits target-specific metadata rather than the native target's input
    /// format and generation limits.
    fn json(&self) -> String {
        let operations = crate::operations(true);
        let missing_checked_live_operations: Vec<_> = operations
            .iter()
            .map(|operation| operation.name())
            .filter(|operation| !self.graph.checked_live_operations.contains_key(*operation))
            .collect();
        serde_json::to_string(&serde_json::json!({
            "target": "fuzz_codegen_port_type_semantics",
            "coverage": self.graph,
            "generated_port_types": self.generated_port_types,
            "checked_port_types": self.checked_port_types,
            "checked_input_type_consumers": self.checked_input_type_consumers,
            "checked_output_type_drivers": self.checked_output_type_drivers,
            "checked_output_chain_depths": self.checked_output_chain_depths,
            "chain_depth_scope": "longest operation dependency path to any output; inputs and literals contribute zero",
            "missing_checked_live_operations": missing_checked_live_operations,
            "checked_scope": "cases passing PIR-versus-Icarus comparison",
            "input_consumer_scope": "direct operand edges into output-live operations",
            "output_driver_scope": "immediate output-value producer",
            "unique_graph_scope": "Block IR text; DSLX type-definition bodies excluded",
        }))
        .expect("coverage consists of JSON-compatible counters")
    }
}

/// Counts operations on the longest dependency path to an output value.
fn longest_output_chain(block: &Block) -> usize {
    fn depth(block: &Block, node: NodeRef, memo: &mut [Option<usize>]) -> usize {
        if let Some(value) = memo[node.index] {
            return value;
        }
        let payload = &block.get_node(node).payload;
        let value = if matches!(
            payload,
            NodePayload::InputPort { .. } | NodePayload::Literal(_)
        ) {
            0
        } else {
            1 + operands(payload)
                .into_iter()
                .map(|operand| depth(block, operand, memo))
                .max()
                .unwrap_or(0)
        };
        memo[node.index] = Some(value);
        value
    }

    let mut memo = vec![None; block.nodes.len()];
    block
        .output_ports()
        .map(|port| depth(block, block.output_value(port), &mut memo))
        .max()
        .unwrap_or(0)
}

thread_local! {
    static REPORT: RefCell<(PortTypeCoverage, Instant)> = RefCell::new((PortTypeCoverage::default(), Instant::now()));
}

/// Returns the cumulative annotated-port coverage collected on this thread.
pub fn snapshot() -> String {
    REPORT.with(|report| report.borrow().0.json())
}

/// Returns a cumulative snapshot after the first case, an inconclusive check,
/// every 4,096 cases, or 30 seconds at a completed case. `choices` is the
/// annotated-port target's four-byte-per-port type-choice region. Ordinary
/// tool failures remain for the caller to report as failing samples.
pub fn record_progress(
    package: &Package,
    choices: &[u8],
    trace: &Trace,
    result: &Result<(), ToolError>,
) -> Option<String> {
    REPORT.with(|report| {
        let mut report = report.borrow_mut();
        let (coverage, last) = &mut *report;
        if !coverage.record(package, choices, trace, result) {
            return None;
        }
        if result.is_err()
            || coverage.graph.generated_cases == 1
            || coverage.graph.generated_cases % 4096 == 0
            || last.elapsed() >= Duration::from_secs(30)
        {
            *last = Instant::now();
            Some(coverage.json())
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::PortTypeCoverage;
    use crate::semantics::Trace;
    use xlsynth::external_tool::ToolError;

    #[test]
    fn checked_boundary_counts_exclude_dead_operations_and_inconclusive_vectors() {
        let package = crate::parse_reference(
            r#"package census
top block census(x: bits[8], y: bits[8], out: bits[8]) {
  x: bits[8] = input_port(name=x, id=1)
  y: bits[8] = input_port(name=y, id=2)
  sum: bits[8] = add(x, y, id=3)
  dead: bits[8] = xor(x, y, id=4)
  inverted: bits[8] = not(sum, id=5)
  out: () = output_port(inverted, name=out, id=6)
}
"#,
        );
        let choices = [1, 0, 0, 0, 0, 0, 0, 0, 4];
        let trace = Trace::with_seed(&package, [0; 32]);
        let mut report = PortTypeCoverage::default();
        assert!(report.record(&package, &choices, &trace, &Ok(())));
        assert_eq!(report.checked_input_type_consumers["signed -> add"], 1);
        assert!(
            !report
                .checked_input_type_consumers
                .contains_key("signed -> xor")
        );
        assert_eq!(report.checked_output_type_drivers["enum <- not"], 1);
        assert_eq!(report.graph.checked_live_pairs["add -> not"], 1);
        assert_eq!(report.checked_output_chain_depths[&2], 1);
        assert_eq!(report.graph.checked_vectors, 16);
        let timeout = Err(ToolError::timeout("iverilog", std::time::Duration::ZERO));
        assert!(report.record(&package, &choices, &trace, &timeout));
        assert_eq!(report.generated_port_types["input:signed"], 2);
        assert_eq!(report.checked_port_types["input:signed"], 1);
        assert_eq!(report.graph.checked_vectors, 16);
        assert_eq!(report.graph.inconclusive_cases.values().sum::<u64>(), 1);
    }
}
