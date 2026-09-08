// SPDX-License-Identifier: Apache-2.0

//! Forward interval-set analysis for functions and blocks.
//!
//! Facts hold for every input and current-register assignment. Resets and
//! assertions are not assumptions. Calls and instance outputs are opaque.
//! Results borrow the immutable graph; analysis neither rewrites it nor invokes
//! XLS, a solver, or an external process.
//!
//! Each scalar leaf is a normalized union of unsigned inclusive intervals.
//! Signed operations interpret those bit patterns in two's-complement order.

use crate::ir::{self, NodeGraph, NodePayload, NodeRef, Type};
use crate::{IrValue, ir_utils, ir_verify};

mod eval;
mod extensions;
mod interval_set;
mod ops;
mod policy;

pub use crate::analysis_utils::AnalysisError;
pub use interval_set::{Interval, IntervalSet};

/// Type-shaped range facts; tuple/array paths retain their IR element order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RangeValue {
    Bits(IntervalSet),
    Tuple(Vec<RangeValue>),
    Array(Vec<RangeValue>),
    Token,
}

impl RangeValue {
    /// Creates unconstrained bits at each leaf of an IR type.
    pub fn unknown(ty: &Type) -> Self {
        match ty {
            Type::Bits(width) => Self::Bits(IntervalSet::full(*width)),
            Type::Tuple(types) => Self::Tuple(types.iter().map(|ty| Self::unknown(ty)).collect()),
            Type::Array(array) => Self::Array(
                (0..array.element_count)
                    .map(|_| Self::unknown(&array.element_type))
                    .collect(),
            ),
            Type::Token => Self::Token,
        }
    }

    /// Creates exact range facts for every bits leaf of a concrete value.
    pub fn constant(value: &IrValue) -> Self {
        match value {
            IrValue::Bits(bits) => Self::Bits(IntervalSet::singleton(bits.clone())),
            IrValue::Tuple(elements) => Self::Tuple(elements.iter().map(Self::constant).collect()),
            IrValue::Array(array) => {
                Self::Array(array.elements().iter().map(Self::constant).collect())
            }
            IrValue::Token => Self::Token,
        }
    }

    /// Borrows scalar facts, distinguishing aggregates from unconstrained bits.
    pub fn as_bits(&self) -> Option<&IntervalSet> {
        match self {
            Self::Bits(bits) => Some(bits),
            _ => None,
        }
    }

    /// Borrows aggregate children without allocating or cloning them.
    pub fn elements(&self) -> Option<&[RangeValue]> {
        match self {
            Self::Tuple(elements) | Self::Array(elements) => Some(elements),
            _ => None,
        }
    }

    /// Queries a bits leaf by tuple/array indices; scalar leaves use `[]`.
    pub fn leaf(&self, path: &[usize]) -> Option<&IntervalSet> {
        let mut current = self;
        for &index in path {
            current = current.elements()?.get(index)?;
        }
        current.as_bits()
    }

    /// Checks shape and leaf widths against a graph's declared type.
    pub fn matches_type(&self, ty: &Type) -> bool {
        match (self, ty) {
            (Self::Bits(bits), Type::Bits(width)) => bits.width() == *width,
            (Self::Tuple(elements), Type::Tuple(types)) => {
                elements.len() == types.len()
                    && elements.iter().zip(types).all(|(v, ty)| v.matches_type(ty))
            }
            (Self::Array(elements), Type::Array(array)) => {
                elements.len() == array.element_count
                    && elements.iter().all(|v| v.matches_type(&array.element_type))
            }
            (Self::Token, Type::Token) => true,
            _ => false,
        }
    }

    /// Checks whether a concrete value belongs to every leaf's interval set.
    pub fn contains(&self, value: &IrValue) -> bool {
        match (self, value) {
            (Self::Bits(known), IrValue::Bits(bits)) => known.contains(bits),
            (Self::Tuple(known), IrValue::Tuple(elements)) => {
                known.len() == elements.len()
                    && known
                        .iter()
                        .zip(elements.iter())
                        .all(|(k, v)| k.contains(v))
            }
            (Self::Array(known), IrValue::Array(array)) => {
                known.len() == array.elements().len()
                    && known
                        .iter()
                        .zip(array.elements())
                        .all(|(k, v)| k.contains(v))
            }
            (Self::Token, IrValue::Token) => true,
            _ => false,
        }
    }

    /// Forms the exact union of each leaf's alternatives without coarsening.
    pub fn join(&self, other: &Self) -> Result<Self, AnalysisError> {
        match (self, other) {
            (Self::Bits(a), Self::Bits(b)) if a.width() == b.width() => Ok(Self::Bits(a.union(b))),
            (Self::Tuple(a), Self::Tuple(b)) | (Self::Array(a), Self::Array(b))
                if a.len() == b.len() =>
            {
                let mut elements = Vec::with_capacity(a.len());
                for (a, b) in a.iter().zip(b) {
                    elements.push(a.join(b)?);
                }
                Ok(if matches!(self, Self::Tuple(_)) {
                    Self::Tuple(elements)
                } else {
                    Self::Array(elements)
                })
            }
            (Self::Token, Self::Token) => Ok(Self::Token),
            _ => Err(AnalysisError::new(
                "cannot join differently shaped range values",
            )),
        }
    }
}

/// Unconditional facts for one immutable operation graph, including dead nodes.
///
/// Facts cannot remain in use across a mutation of the analyzed graph:
///
/// ```compile_fail
/// use xlsynth_pir::{ir, range_analysis::analyze_fn};
/// fn stale_facts(function: &mut ir::Fn) {
///     let facts = analyze_fn(function).unwrap();
///     function.nodes.clear();
///     assert!(facts.iter().next().is_some());
/// }
/// ```
#[derive(Debug)]
pub struct RangeAnalysis<'ir> {
    graph: &'ir NodeGraph,
    values: Vec<Option<RangeValue>>,
}

impl<'ir> RangeAnalysis<'ir> {
    /// Returns the immutable graph to which these facts belong.
    pub fn graph(&self) -> &'ir NodeGraph {
        self.graph
    }

    /// Borrows a node's facts; invalid references and Nil slots return `None`.
    pub fn get(&self, node: NodeRef) -> Option<&RangeValue> {
        self.values.get(node.index)?.as_ref()
    }

    /// Queries one scalar node, retaining unconstrained values as `Some`.
    pub fn bits(&self, node: NodeRef) -> Option<&IntervalSet> {
        self.get(node)?.as_bits()
    }

    /// Queries an aggregate bits leaf using its original IR index path.
    pub fn leaf(&self, node: NodeRef, path: &[usize]) -> Option<&IntervalSet> {
        self.get(node)?.leaf(path)
    }

    /// Iterates in graph storage order, excluding reserved/deleted Nil slots.
    pub fn iter(&self) -> impl Iterator<Item = (NodeRef, &RangeValue)> {
        self.values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.as_ref().map(|v| (NodeRef { index }, v)))
    }
}

/// Analyzes a function with unconstrained parameters, without rewriting it.
pub fn analyze_fn(function: &ir::Fn) -> Result<RangeAnalysis<'_>, AnalysisError> {
    ir_verify::verify_function_signature(function)
        .map_err(|e| AnalysisError::new(e.to_string()))?;
    let ret = function
        .ret_node_ref
        .and_then(|r| function.nodes.get(r.index))
        .ok_or_else(|| AnalysisError::new("function has no valid return node"))?;
    if ret.ty != function.ret_ty || matches!(ret.payload, NodePayload::Nil) {
        return Err(AnalysisError::new(
            "function return does not match its signature",
        ));
    }
    analyze_graph(&function.graph)
}

/// Analyzes block dataflow over arbitrary inputs and current register values.
///
/// Instance outputs are opaque unknown sources. Package-level hierarchy and
/// resource validity remain the responsibility of the package verifier.
pub fn analyze_block(block: &ir::Block) -> Result<RangeAnalysis<'_>, AnalysisError> {
    if !block
        .nodes
        .first()
        .is_some_and(|node| matches!(node.payload, NodePayload::Nil))
    {
        return Err(AnalysisError::new(
            "block graph must start with a Nil sentinel",
        ));
    }
    analyze_graph(&block.graph)
}

/// Validates local graph contracts before evaluating all nodes once.
fn analyze_graph(graph: &NodeGraph) -> Result<RangeAnalysis<'_>, AnalysisError> {
    crate::analysis_utils::validate_graph(graph)?;
    let mut values = vec![None; graph.nodes.len()];
    for reference in ir_utils::get_topological(graph) {
        let node = &graph.nodes[reference.index];
        if matches!(node.payload, NodePayload::Nil) {
            // Transforms leave typed holes; these are not operations or values.
            continue;
        }
        let value = eval::evaluate(node, &values).map_err(|e| {
            AnalysisError::new(format!(
                "ranges for {} node id={} ({}): {e}",
                graph.name,
                node.text_id,
                node.payload.get_operator()
            ))
        })?;
        if !value.matches_type(&node.ty) {
            return Err(AnalysisError::new(format!(
                "range result shape disagrees with node id={} type {}",
                node.text_id, node.ty
            )));
        }
        values[reference.index] = Some(value);
    }
    Ok(RangeAnalysis { graph, values })
}
