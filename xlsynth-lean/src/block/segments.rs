// SPDX-License-Identifier: Apache-2.0

//! Split a block's topological order into bounded Lean definitions.
//!
//! A cut is the set of wires live between two definitions. Both evaluators
//! pack these wires in graph-index order into the same balanced product, so
//! the proof can compare a pair of leaves on an arbitrary input of that type.
//! The public result can read input and register fields directly.

use std::{collections::BTreeSet, fmt::Write};

use anyhow::{Context, Result};
use xlsynth_pir::{
    ir::{NodeGraph, NodePayload, NodeRef},
    ir_utils,
};

use super::{BlockContext, raw};

/// Maximum number of graph computations emitted as nested lets in one leaf.
///
/// Larger leaves reduce wide intermediate tuples, but a correspondence proof
/// for 192 Boolean operations exceeded Lean's default recursion limit.
const MAX_LEAF_OPERATIONS: usize = 160;

/// Recognize a bare input or register read, optionally followed by tuple
/// projections.
pub(super) fn is_interface_projection(graph: &NodeGraph, mut wire: NodeRef) -> bool {
    while let NodePayload::TupleIndex { tuple, .. } = &graph.get_node(wire).payload {
        wire = *tuple;
    }
    matches!(
        graph.get_node(wire).payload,
        NodePayload::InputPort { .. } | NodePayload::RegisterRead { .. }
    )
}

/// Operations and live wires shared by both evaluators and their correspondence
/// proof.
pub(super) struct Segments {
    /// Each inner list contains the computations emitted by one leaf, in
    /// dependency order.
    pub(super) leaves: Vec<Vec<NodeRef>>,
    /// Cut `i` is the sorted input to leaf `i`, and the extra final cut feeds
    /// the result.
    pub(super) cuts: Vec<Vec<NodeRef>>,
}

impl Segments {
    /// Plan leaf boundaries for blocks whose total node count exceeds the leaf
    /// limit.
    ///
    /// The backward walk starts with operands needed by the clock-step result.
    /// Crossing a definition removes its wire and adds its operands. Even a
    /// large block with no leaf computations gets one leaf, so the root exists.
    pub(super) fn new(block: &BlockContext<'_>) -> Option<Self> {
        let graph = &block.graph;
        // Small blocks keep their single readable definition. Include port
        // and effect bindings in this check since they also nest in that form.
        if graph.nodes.len() <= MAX_LEAF_OPERATIONS {
            return None;
        }
        let mut live = BTreeSet::new();
        let mut leaves = Vec::new();
        let mut leaf = Vec::new();
        for reference in ir_utils::get_topological_nodes(&graph.nodes) {
            match &graph.get_node(reference).payload {
                NodePayload::InputPort { .. }
                | NodePayload::RegisterRead { .. }
                | NodePayload::Nil => continue,
                payload @ (NodePayload::OutputPort { .. } | NodePayload::RegisterWrite { .. }) => {
                    live.extend(
                        ir_utils::operands(payload)
                            .iter()
                            .map(|signal| signal.index),
                    );
                    continue;
                }
                NodePayload::Assert { activate, .. } => {
                    live.insert(activate.index);
                }
                _ => {
                    // This operation goes into the leaf below. The reverse walk
                    // collects its operands. Only outputs, register writes, and
                    // assertions seed the wires needed for the final result.
                }
            }
            leaf.push(reference);
            if leaf.len() == MAX_LEAF_OPERATIONS {
                leaves.push(std::mem::take(&mut leaf));
            }
        }
        if !leaf.is_empty() || leaves.is_empty() {
            leaves.push(leaf);
        }

        let snapshot = |live: &BTreeSet<usize>| {
            live.iter()
                .map(|&index| NodeRef { index })
                .collect::<Vec<_>>()
        };
        // The result wrapper reads these fields directly. Reverse traversal
        // still adds them to the cuts required by equations that use them.
        live.retain(|&index| !is_interface_projection(graph, NodeRef { index }));
        let mut cuts = vec![snapshot(&live)];
        for leaf in leaves.iter().rev() {
            for &reference in leaf.iter().rev() {
                live.remove(&reference.index);
                live.extend(
                    ir_utils::operands(&graph.get_node(reference).payload)
                        .iter()
                        .map(|arg| arg.index),
                );
            }
            cuts.push(snapshot(&live));
        }
        cuts.reverse();
        Some(Self { leaves, cuts })
    }

    /// Render a cut's product of physical wire types using this plan's block.
    pub(super) fn cut_type(&self, block: &BlockContext<'_>, cut: usize) -> Result<String> {
        bundle(&self.cuts[cut], "L.Unit", " × ", &|wire| {
            raw::lean_type(block.graph.get_node_ty(wire))
                .with_context(|| raw::node_context(block, wire))
        })
    }

    /// Emit stage signatures and joins in dependency order.
    ///
    /// The renderer writes each leaf body with its input bound to `w` and
    /// returns the physical wires prescribed by the following cut.
    pub(super) fn write_definitions(
        &self,
        out: &mut String,
        block: &BlockContext<'_>,
        write_leaf: impl Fn(&mut String, usize) -> Result<()>,
    ) -> Result<()> {
        let types = (0..self.cuts.len())
            .map(|cut| self.cut_type(block, cut))
            .collect::<Result<Vec<_>>>()?;
        for stage in self.stages() {
            writeln!(
                out,
                "def {} (w : {}) : {} :=",
                stage.name(),
                types[stage.start],
                types[stage.end]
            )?;
            if let Some((left, right)) = stage.children() {
                writeln!(out, "  {} ({} w)\n", right.name(), left.name())?;
            } else {
                write_leaf(out, stage.start)?;
            }
        }
        Ok(())
    }

    /// The evaluator calls the stage spanning the entry cut through the result
    /// cut.
    pub(super) fn root(&self) -> Stage {
        Stage {
            start: 0,
            end: self.leaves.len(),
        }
    }

    /// Visit children before their parent so each emitted reference resolves in
    /// Lean.
    pub(super) fn stages(&self) -> Vec<Stage> {
        fn visit(stage: Stage, ordered: &mut Vec<Stage>) {
            if let Some((left, right)) = stage.children() {
                visit(left, ordered);
                visit(right, ordered);
            }
            ordered.push(stage);
        }
        let mut ordered = Vec::with_capacity(2 * self.leaves.len() - 1);
        visit(self.root(), &mut ordered);
        ordered
    }
}

/// Half-open range of consecutive leaves evaluated by one Lean definition.
#[derive(Clone, Copy)]
pub(super) struct Stage {
    pub(super) start: usize,
    pub(super) end: usize,
}

impl Stage {
    /// Both evaluator namespaces and the proof use the same range-based name.
    pub(super) fn name(self) -> String {
        format!("xls_segment_{}_{}", self.start, self.end)
    }

    /// Split a non-leaf stage at the middle cut, which connects the two
    /// children.
    pub(super) fn children(self) -> Option<(Self, Self)> {
        let mid = self.start + (self.end - self.start) / 2;
        (mid != self.start).then_some((
            Self {
                start: self.start,
                end: mid,
            },
            Self {
                start: mid,
                end: self.end,
            },
        ))
    }
}

/// Render the balanced product of crossing wires, leaving each wire's XLS shape
/// intact.
fn bundle(
    fields: &[NodeRef],
    empty: &str,
    separator: &str,
    render: &impl Fn(NodeRef) -> Result<String>,
) -> Result<String> {
    bundle_reusing(fields, empty, separator, render, &|_| None)
}

/// Render a bundle, letting the caller supply a whole product for any matching
/// subtree.
fn bundle_reusing(
    fields: &[NodeRef],
    empty: &str,
    separator: &str,
    render: &impl Fn(NodeRef) -> Result<String>,
    subtree: &impl Fn(&[NodeRef]) -> Option<String>,
) -> Result<String> {
    match fields {
        [] => Ok(empty.to_string()),
        [field] => render(*field),
        _ => {
            if let Some(existing) = subtree(fields) {
                return Ok(existing);
            }
            let (left, right) = fields.split_at(fields.len() / 2);
            Ok(format!(
                "({}{}{})",
                bundle_reusing(left, empty, separator, render, subtree)?,
                separator,
                bundle_reusing(right, empty, separator, render, subtree)?
            ))
        }
    }
}

/// Pack ordered cut wires as a Lean value with the same nesting as `cut_type`.
///
/// An empty cut is unit and a singleton is the wire itself.
pub(super) fn bundle_value(
    fields: &[NodeRef],
    render: impl Fn(NodeRef) -> Result<String>,
) -> Result<String> {
    bundle(fields, "()", ", ", &render)
}

impl Segments {
    /// Forward an unchanged product directly from the input when its wires
    /// occupy exactly the same balanced subtree in the output.
    pub(super) fn leaf_result(
        &self,
        leaf: usize,
        render: impl Fn(NodeRef) -> Result<String>,
    ) -> Result<String> {
        let incoming = &self.cuts[leaf];
        bundle_reusing(&self.cuts[leaf + 1], "()", ", ", &render, &|fields| {
            let index = incoming
                .binary_search_by_key(&fields[0].index, |wire| wire.index)
                .ok()?;
            let (path, range) = subtree_projection(incoming.len(), index, fields.len(), "w");
            (incoming[range] == *fields).then_some(path)
        })
    }
}

/// Select wire `index < len` from a balanced cut value at its use site.
///
/// Binding every incoming wire to a local first can exhaust elaboration depth
/// before any operation in the leaf.
pub(super) fn projection(len: usize, index: usize, root: &str) -> String {
    subtree_projection(len, index, 1, root).0
}

/// Descend toward a wire until the subtree fits the requested size. Return
/// its actual range, since consecutive wires can straddle a product boundary.
fn subtree_projection(
    mut len: usize,
    mut index: usize,
    size: usize,
    root: &str,
) -> (String, std::ops::Range<usize>) {
    debug_assert!(index < len);
    let mut start = 0;
    let mut expr = root.to_owned();
    while len > size {
        let mid = len / 2;
        if index < mid {
            expr.push_str(".1");
            len = mid;
        } else {
            expr.push_str(".2");
            start += mid;
            index -= mid;
            len -= mid;
        }
    }
    (expr, start..start + len)
}
