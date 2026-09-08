// SPDX-License-Identifier: Apache-2.0

//! Forward known-bit and population-count analysis for functions and blocks.
//!
//! Every reported fact holds for all input values and all current register
//! values. Register resets and program assertions are not assumptions. The
//! result borrows its graph so it cannot be reused across graph mutations.
//! This analysis does not invoke XLS, a solver, or any external executable.

use crate::ir::{self, NodeGraph, NodePayload, NodeRef, Type};
use crate::{IrValue, ir_utils, ir_verify};

pub use crate::analysis_utils::AnalysisError;

pub(crate) mod bits;
mod eval;
pub(crate) mod extensions;

pub use bits::KnownBits;

/// Type-shaped knowledge; tuple/array paths retain their IR element order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KnownValue {
    Bits(KnownBits),
    Tuple(Vec<KnownValue>),
    Array(Vec<KnownValue>),
    Token,
}

impl KnownValue {
    /// Creates unconstrained bits at each leaf of an IR type.
    pub fn unknown(ty: &Type) -> Self {
        match ty {
            Type::Bits(width) => Self::Bits(KnownBits::unknown(*width)),
            Type::Tuple(types) => Self::Tuple(types.iter().map(|ty| Self::unknown(ty)).collect()),
            Type::Array(array) => Self::Array(
                (0..array.element_count)
                    .map(|_| Self::unknown(&array.element_type))
                    .collect(),
            ),
            Type::Token => Self::Token,
        }
    }

    /// Creates exact knowledge for every bits leaf of a concrete value.
    pub fn constant(value: &IrValue) -> Self {
        match value {
            IrValue::Bits(bits) => Self::Bits(KnownBits::constant(bits)),
            IrValue::Tuple(elements) => Self::Tuple(elements.iter().map(Self::constant).collect()),
            IrValue::Array(array) => {
                Self::Array(array.elements().iter().map(Self::constant).collect())
            }
            IrValue::Token => Self::Token,
        }
    }

    /// Borrows scalar knowledge, distinguishing aggregates from unknown bits.
    pub fn as_bits(&self) -> Option<&KnownBits> {
        match self {
            Self::Bits(bits) => Some(bits),
            _ => None,
        }
    }

    /// Borrows aggregate children without allocating or cloning them.
    pub fn elements(&self) -> Option<&[KnownValue]> {
        match self {
            Self::Tuple(elements) | Self::Array(elements) => Some(elements),
            _ => None,
        }
    }

    /// Queries a bits leaf by tuple/array indices; scalar leaves use `[]`.
    pub fn leaf(&self, path: &[usize]) -> Option<&KnownBits> {
        let mut current = self;
        for &index in path {
            current = current.elements()?.get(index)?;
        }
        current.as_bits()
    }

    /// Checks shape and leaf widths against a graph's declared type.
    pub fn matches_type(&self, ty: &Type) -> bool {
        match (self, ty) {
            (Self::Bits(bits), Type::Bits(width)) => bits.bit_count() == *width,
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

    /// Checks whether a concrete value satisfies every mask and population
    /// bound.
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

    /// Retains only knowledge shared by both alternatives.
    pub fn join(&self, other: &Self) -> Result<Self, AnalysisError> {
        match (self, other) {
            (Self::Bits(a), Self::Bits(b)) if a.bit_count() == b.bit_count() => {
                Ok(Self::Bits(a.join(b)))
            }
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
                "cannot join differently shaped known values",
            )),
        }
    }
}

/// Unconditional facts for one immutable operation graph, including dead nodes.
///
/// Facts cannot remain in use across a mutation of the analyzed graph:
///
/// ```compile_fail
/// use xlsynth_pir::{ir, known_bits::analyze_fn};
/// fn stale_facts(function: &mut ir::Fn) {
///     let facts = analyze_fn(function).unwrap();
///     function.nodes.clear();
///     assert!(facts.iter().next().is_some());
/// }
/// ```
#[derive(Debug)]
pub struct KnownBitsAnalysis<'ir> {
    graph: &'ir NodeGraph,
    values: Vec<Option<KnownValue>>,
}

impl<'ir> KnownBitsAnalysis<'ir> {
    /// Returns the immutable graph to which these facts belong.
    pub fn graph(&self) -> &'ir NodeGraph {
        self.graph
    }

    /// Borrows a node's facts; invalid references and Nil slots return `None`.
    pub fn get(&self, node: NodeRef) -> Option<&KnownValue> {
        self.values.get(node.index)?.as_ref()
    }

    /// Queries one scalar node, retaining all-unknown values as `Some`.
    pub fn bits(&self, node: NodeRef) -> Option<&KnownBits> {
        self.get(node)?.as_bits()
    }

    /// Queries an aggregate bits leaf using its original IR index path.
    pub fn leaf(&self, node: NodeRef, path: &[usize]) -> Option<&KnownBits> {
        self.get(node)?.leaf(path)
    }

    /// Iterates in graph storage order, excluding reserved/deleted Nil slots.
    pub fn iter(&self) -> impl Iterator<Item = (NodeRef, &KnownValue)> {
        self.values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.as_ref().map(|v| (NodeRef { index }, v)))
    }
}

/// Analyzes a function with unconstrained parameters, without rewriting it.
pub fn analyze_fn(function: &ir::Fn) -> Result<KnownBitsAnalysis<'_>, AnalysisError> {
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
pub fn analyze_block(block: &ir::Block) -> Result<KnownBitsAnalysis<'_>, AnalysisError> {
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
fn analyze_graph(graph: &NodeGraph) -> Result<KnownBitsAnalysis<'_>, AnalysisError> {
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
                "known bits for {} node id={} ({}): {e}",
                graph.name,
                node.text_id,
                node.payload.get_operator()
            ))
        })?;
        if !value.matches_type(&node.ty) {
            return Err(AnalysisError::new(format!(
                "known-bits result shape disagrees with node id={} type {}",
                node.text_id, node.ty
            )));
        }
        values[reference.index] = Some(value);
    }
    Ok(KnownBitsAnalysis { graph, values })
}

#[cfg(test)]
mod tests {
    use super::{KnownBits, KnownValue};
    use crate::IrBits;

    fn constant(width: usize, value: u64) -> KnownValue {
        KnownValue::Bits(KnownBits::constant(
            &IrBits::make_ubits(width, value).unwrap(),
        ))
    }

    #[test]
    fn aggregate_join_preserves_wide_empty_and_token_leaves() {
        let mut wide_bits = vec![false; 129];
        wide_bits[128] = true;
        wide_bits[0] = true;
        let wide_a = KnownValue::Bits(KnownBits::constant(&IrBits::from_lsb_is_0(&wide_bits)));
        wide_bits[1] = true;
        let wide_b = KnownValue::Bits(KnownBits::constant(&IrBits::from_lsb_is_0(&wide_bits)));
        let a = KnownValue::Tuple(vec![
            wide_a,
            KnownValue::Array(vec![constant(4, 1), constant(4, 2)]),
            KnownValue::Array(vec![]),
            KnownValue::Tuple(vec![]),
            KnownValue::Token,
            constant(0, 0),
        ]);
        let b = KnownValue::Tuple(vec![
            wide_b,
            KnownValue::Array(vec![constant(4, 3), constant(4, 6)]),
            KnownValue::Array(vec![]),
            KnownValue::Tuple(vec![]),
            KnownValue::Token,
            constant(0, 0),
        ]);
        let joined = a.join(&b).unwrap();
        assert_eq!(joined, b.join(&a).unwrap());
        assert_eq!(a.join(&a).unwrap(), a);
        assert_eq!(
            joined.leaf(&[0]).unwrap().to_ternary_string(),
            format!("1{}X1", "0".repeat(126)),
        );
        assert_eq!(joined.leaf(&[1, 0]).unwrap().to_ternary_string(), "00X1");
        assert_eq!(joined.leaf(&[1, 1]).unwrap().to_ternary_string(), "0X10");
        assert_eq!(joined.elements().unwrap()[2], KnownValue::Array(vec![]));
        assert_eq!(joined.elements().unwrap()[3], KnownValue::Tuple(vec![]));
        assert_eq!(joined.elements().unwrap()[4], KnownValue::Token);
        assert_eq!(joined.leaf(&[5]).unwrap().bit_count(), 0);
    }

    #[test]
    fn aggregate_join_rejects_shape_errors_without_changing_inputs() {
        let pairs = [
            (constant(4, 1), constant(8, 1)),
            (KnownValue::Tuple(vec![]), KnownValue::Array(vec![])),
            (
                KnownValue::Array(vec![constant(4, 1)]),
                KnownValue::Array(vec![]),
            ),
            (
                KnownValue::Tuple(vec![
                    constant(129, 1),
                    KnownValue::Array(vec![constant(4, 2), KnownValue::Token]),
                ]),
                KnownValue::Tuple(vec![
                    constant(129, 3),
                    KnownValue::Array(vec![constant(4, 6), KnownValue::Tuple(vec![])]),
                ]),
            ),
        ];
        for (a, b) in pairs {
            let original_a = a.clone();
            let original_b = b.clone();
            assert_eq!(
                a.join(&b).unwrap_err().to_string(),
                "cannot join differently shaped known values",
            );
            assert_eq!(a, original_a);
            assert_eq!(b, original_b);
        }
    }
}
