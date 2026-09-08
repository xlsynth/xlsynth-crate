// SPDX-License-Identifier: Apache-2.0

//! Shared local validation for read-only numerical graph analyses.

use std::collections::HashSet;
use std::fmt;

use crate::ir::{NodeGraph, NodePayload, Type};
use crate::{ValueError, ir_deduce, ir_utils, ir_verify};

/// An invalid graph, type, or requested combination of abstract values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisError(pub(crate) String);

impl AnalysisError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AnalysisError {}

impl From<ValueError> for AnalysisError {
    fn from(error: ValueError) -> Self {
        Self(error.to_string())
    }
}

impl From<String> for AnalysisError {
    fn from(error: String) -> Self {
        Self(error)
    }
}

/// Validates bounds once, followed by graph shape and local operation
/// semantics.
pub(crate) fn validate_graph(graph: &NodeGraph) -> Result<(), AnalysisError> {
    ir_verify::verify_graph_operand_indices_in_bounds(graph)?;
    ir_utils::verify_no_cycle(graph)?;
    verify_local_types(graph)?;
    ir_verify::verify_graph_xls_node_semantics_after_bounds_check(graph)?;
    Ok(())
}

/// Checks local operand/result types without requiring package call metadata.
fn verify_local_types(graph: &NodeGraph) -> Result<(), AnalysisError> {
    let mut ids = HashSet::new();
    for node in &graph.nodes {
        if matches!(node.payload, NodePayload::Nil) {
            // Removed nodes need not retain a meaningful type or unique ID.
            continue;
        }
        if !ids.insert(node.text_id) {
            return Err(AnalysisError::new(format!(
                "duplicate node id={}",
                node.text_id
            )));
        }
        if node.ty.checked_bit_count().is_none() {
            return Err(AnalysisError::new(format!(
                "node id={} type width overflows usize",
                node.text_id
            )));
        }
        let operands = ir_utils::operands(&node.payload);
        let operand_types = operands
            .iter()
            .map(|reference| {
                let operand = graph.get_node(*reference);
                if matches!(operand.payload, NodePayload::Nil) {
                    Err(AnalysisError::new(format!(
                        "node id={} references a Nil slot at index {}",
                        node.text_id, reference.index
                    )))
                } else {
                    Ok(operand.ty.clone())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let deduced = if matches!(&node.payload, NodePayload::Array(elements) if elements.is_empty())
        {
            // An empty array has no element from which to infer its type.
            if !matches!(&node.ty, Type::Array(array) if array.element_count == 0) {
                return Err(AnalysisError::new(
                    "empty array requires a zero-length array type",
                ));
            }
            None
        } else {
            ir_deduce::deduce_result_type(&node.payload, &operand_types)
                .map_err(|e| AnalysisError::new(format!("node id={}: {e}", node.text_id)))?
        };
        if let Some(deduced) = deduced {
            if deduced != node.ty {
                return Err(AnalysisError::new(format!(
                    "node id={} type mismatch: deduced {deduced}, declared {}",
                    node.text_id, node.ty
                )));
            }
        }
    }
    Ok(())
}
