// SPDX-License-Identifier: Apache-2.0

//! Exact routing of selected bit ranges and aggregate elements through PIR
//! operations.

use std::collections::HashSet;
use std::rc::Rc;

use smallvec::{SmallVec, smallvec};

use crate::IrValue;
use crate::ir::{Binop, NaryOp, NodeGraph, NodePayload, NodeRef, Type, Unop};
use crate::ir_utils::literal_offset;
use crate::ir_value_utils::ir_bits_to_usize;

/// A field path followed by an optional half-open range of the selected bits.
#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct Selection {
    pub(crate) path: SmallVec<[usize; 2]>,
    pub(crate) bits: Option<(usize, usize)>,
}

impl Selection {
    /// Selects the full payload of a bits value, array, tuple, or token.
    pub(crate) fn whole(ty: &Type) -> Self {
        let bits = match ty {
            Type::Bits(width) => Some((0, *width)),
            _ => None,
        };
        Self {
            path: SmallVec::new(),
            bits,
        }
    }
}

/// Reading a port sink's PIR result yields `()`. Traversing the corresponding
/// hierarchy connection reads the operand driving the port instead.
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum DemandNode {
    Value(usize),
    PortDriver(usize),
}

/// A particular field or bit range of a definition-local value or port driver.
#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct Demand {
    target: DemandNode,
    pub(crate) part: Selection,
}

impl Demand {
    /// Selects a part of the node's PIR result.
    pub(crate) fn value(node: NodeRef, part: Selection) -> Self {
        Self {
            target: DemandNode::Value(node.index),
            part,
        }
    }

    pub(crate) fn whole(graph: &NodeGraph, node: NodeRef) -> Self {
        Self::value(node, Selection::whole(graph.get_node_ty(node)))
    }

    /// Selects data crossing an output port or an instantiation input.
    pub(crate) fn port_driver(sink: NodeRef, part: Selection) -> Self {
        Self {
            target: DemandNode::PortDriver(sink.index),
            part,
        }
    }

    /// Starts a physical connectivity trace at a whole-node cycle candidate.
    pub(crate) fn whole_connection(graph: &NodeGraph, node: NodeRef) -> Self {
        match graph.get_node(node).payload {
            NodePayload::InstantiationInput { arg, .. } | NodePayload::OutputPort { arg, .. } => {
                Self::port_driver(node, Selection::whole(graph.get_node_ty(arg)))
            }
            _ => Self::whole(graph, node),
        }
    }

    pub(crate) fn node(&self) -> NodeRef {
        let (DemandNode::Value(index) | DemandNode::PortDriver(index)) = self.target;
        NodeRef { index }
    }

    /// The selected driver, if this demand traverses a hierarchy connection.
    pub(crate) fn driver(&self, graph: &NodeGraph) -> Option<NodeRef> {
        let DemandNode::PortDriver(index) = self.target else {
            return None;
        };
        match graph.nodes[index].payload {
            NodePayload::InstantiationInput { arg, .. } | NodePayload::OutputPort { arg, .. } => {
                Some(arg)
            }
            _ => unreachable!("only port sinks have a driver"),
        }
    }

    /// Retains the requested field and bits when following a value operand.
    pub(crate) fn at(&self, node: NodeRef) -> Self {
        Self::value(node, self.part.clone())
    }
}

/// A nonempty half-open bit range of one graph value, numbered from the LSB.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BitRange {
    pub node: NodeRef,
    pub lo: usize,
    pub hi: usize,
}

/// An exact expression for a range of the original result.
pub(crate) enum BitProjection {
    Forward(BitRange),
    Zero,
    Extend { arg: BitRange, signed: bool },
    PadLow { arg: BitRange, zeros: usize },
    Unary(Unop, BitRange),
    Nary(NaryOp, Vec<BitRange>),
}

impl BitProjection {
    /// Builds the operation for this recipe after its operand ranges have
    /// been resolved. A left shift can also request a low padding literal.
    pub(crate) fn into_payload<E>(
        self,
        width: usize,
        mut operand: impl FnMut(BitRange) -> NodeRef,
        zero: impl FnOnce(usize) -> Result<NodeRef, E>,
    ) -> Result<NodePayload, E> {
        Ok(match self {
            Self::Forward(part) => NodePayload::Unop(Unop::Identity, operand(part)),
            Self::Zero => NodePayload::Literal(crate::IrValue::Bits(crate::IrBits::zero(width))),
            Self::Extend { arg, signed } => {
                let arg = operand(arg);
                if signed {
                    NodePayload::SignExt {
                        arg,
                        new_bit_count: width,
                    }
                } else {
                    NodePayload::ZeroExt {
                        arg,
                        new_bit_count: width,
                    }
                }
            }
            Self::PadLow { arg, zeros } => {
                NodePayload::Nary(NaryOp::Concat, vec![operand(arg), zero(zeros)?])
            }
            Self::Unary(op, arg) => NodePayload::Unop(op, operand(arg)),
            Self::Nary(op, args) => NodePayload::Nary(op, args.into_iter().map(operand).collect()),
        })
    }

    /// Visits selected operands in the same order as the original operation.
    pub(crate) fn inputs(&self) -> &[BitRange] {
        match self {
            Self::Zero => &[],
            Self::Forward(arg)
            | Self::Unary(_, arg)
            | Self::Extend { arg, .. }
            | Self::PadLow { arg, .. } => std::slice::from_ref(arg),
            Self::Nary(_, args) => args,
        }
    }
}

impl BitRange {
    /// Pushes the selection into the operands where the operation permits it.
    /// The lookup includes literals bound to formal parameters during call
    /// analysis.
    pub(crate) fn project(
        self,
        graph: &NodeGraph,
        mut control: impl FnMut(NodeRef) -> Option<Rc<IrValue>>,
    ) -> Option<BitProjection> {
        let Self { node, lo, hi } = self;
        Some(match &graph.get_node(node).payload {
            NodePayload::Unop(Unop::Identity, arg) => {
                BitProjection::Forward(Self { node: *arg, lo, hi })
            }
            NodePayload::Unop(Unop::Not, arg) => {
                BitProjection::Unary(Unop::Not, Self { node: *arg, lo, hi })
            }
            NodePayload::Unop(Unop::Reverse, arg) => {
                let Type::Bits(width) = graph.get_node_ty(*arg) else {
                    return None;
                };
                BitProjection::Unary(
                    Unop::Reverse,
                    Self {
                        node: *arg,
                        lo: width - hi,
                        hi: width - lo,
                    },
                )
            }
            NodePayload::BitSlice { arg, start, .. } => BitProjection::Forward(Self {
                node: *arg,
                lo: start.checked_add(lo)?,
                hi: start.checked_add(hi)?,
            }),
            NodePayload::DynamicBitSlice { arg, start, .. } => {
                let Type::Bits(width) = graph.get_node_ty(*arg) else {
                    return None;
                };
                shift_right(
                    Self { node: *arg, lo, hi },
                    *width,
                    index_value(graph, *start, &mut control)?,
                    false,
                )
            }
            NodePayload::BitSliceUpdate {
                arg,
                start,
                update_value,
            } => {
                let Type::Bits(width) = graph.get_node_ty(*update_value) else {
                    return None;
                };
                update(
                    Self { node: *arg, lo, hi },
                    index_value(graph, *start, &mut control)?,
                    *update_value,
                    *width,
                )
            }
            NodePayload::ZeroExt { arg, .. } | NodePayload::SignExt { arg, .. } => {
                let Type::Bits(width) = graph.get_node_ty(*arg) else {
                    return None;
                };
                shift_right(
                    Self { node: *arg, lo, hi },
                    *width,
                    0,
                    matches!(graph.get_node(node).payload, NodePayload::SignExt { .. }),
                )
            }
            NodePayload::Binop(op @ (Binop::Shll | Binop::Shrl | Binop::Shra), arg, amount) => {
                let shift = index_value(graph, *amount, &mut control)?;
                let Type::Bits(width) = graph.get_node_ty(*arg) else {
                    return None;
                };
                match op {
                    Binop::Shll if shift >= hi => BitProjection::Zero,
                    Binop::Shll if shift <= lo => BitProjection::Forward(Self {
                        node: *arg,
                        lo: lo - shift,
                        hi: hi - shift,
                    }),
                    Binop::Shll => BitProjection::PadLow {
                        arg: Self {
                            node: *arg,
                            lo: 0,
                            hi: hi - shift,
                        },
                        zeros: shift - lo,
                    },
                    _ => shift_right(
                        Self { node: *arg, lo, hi },
                        *width,
                        shift,
                        *op == Binop::Shra,
                    ),
                }
            }
            NodePayload::TupleIndex { .. } | NodePayload::ArrayIndex { .. } => {
                BitProjection::Forward(Self {
                    node: selected_aggregate_operand(graph, node, control)?,
                    lo,
                    hi,
                })
            }
            NodePayload::Sel { .. }
            | NodePayload::PrioritySel { .. }
            | NodePayload::OneHotSel { .. } => match project_select(graph, node, &mut control)? {
                SelectedCases::One(arg) => BitProjection::Forward(Self { node: arg, lo, hi }),
                SelectedCases::Or(arms) if arms.is_empty() => BitProjection::Zero,
                SelectedCases::Or(arms) => BitProjection::Nary(
                    NaryOp::Or,
                    arms.into_iter().map(|node| Self { node, lo, hi }).collect(),
                ),
            },
            NodePayload::Nary(NaryOp::Concat, args) => {
                let mut base = 0usize;
                let mut inputs = Vec::new();
                for &arg in args.iter().rev() {
                    let Type::Bits(width) = graph.get_node_ty(arg) else {
                        return None;
                    };
                    let end = base.checked_add(*width)?;
                    let start = lo.max(base);
                    let limit = hi.min(end);
                    if start < limit {
                        inputs.push(Self {
                            node: arg,
                            lo: start - base,
                            hi: limit - base,
                        });
                    }
                    if end >= hi {
                        break;
                    }
                    base = end;
                }
                inputs.reverse();
                concat(inputs)
            }
            NodePayload::Nary(
                op @ (NaryOp::And | NaryOp::Or | NaryOp::Xor | NaryOp::Nand | NaryOp::Nor),
                args,
            ) => BitProjection::Nary(
                *op,
                args.iter().map(|&node| Self { node, lo, hi }).collect(),
            ),
            _ => return None,
        })
    }
}

/// Reads the requested range after a right shift, padding above the input.
/// Comparing against the available width before adding also handles huge
/// offsets.
fn shift_right(range: BitRange, width: usize, amount: usize, signed: bool) -> BitProjection {
    let BitRange { node, lo, hi } = range;
    let available = width.saturating_sub(amount);
    if lo >= available {
        return if signed && width != 0 {
            extended(
                BitRange {
                    node,
                    lo: width - 1,
                    hi: width,
                },
                hi - lo,
                true,
            )
        } else {
            BitProjection::Zero
        };
    }
    let arg = BitRange {
        node,
        lo: amount + lo,
        hi: amount + hi.min(available),
    };
    extended(arg, hi - lo, signed)
}

/// Concatenates ranges in most-significant-first order.
fn concat(parts: Vec<BitRange>) -> BitProjection {
    if let [part] = parts.as_slice() {
        BitProjection::Forward(*part)
    } else {
        BitProjection::Nary(NaryOp::Concat, parts)
    }
}

/// Intersects the requested range with the overwritten field. Bits above and
/// below the overlap retain their positions in the base word.
fn update(base: BitRange, start: usize, value: NodeRef, width: usize) -> BitProjection {
    let lo = base.lo.max(start);
    let hi = base.hi.min(start.saturating_add(width));
    if lo >= hi {
        return BitProjection::Forward(base);
    }
    let mut parts = Vec::with_capacity(3);
    if hi < base.hi {
        parts.push(BitRange { lo: hi, ..base });
    }
    parts.push(BitRange {
        node: value,
        lo: lo - start,
        hi: hi - start,
    });
    if base.lo < lo {
        parts.push(BitRange { hi: lo, ..base });
    }
    concat(parts)
}

/// Avoids introducing an extension when the selected input already has the
/// requested width.
fn extended(arg: BitRange, width: usize, signed: bool) -> BitProjection {
    if arg.hi - arg.lo == width {
        BitProjection::Forward(arg)
    } else {
        BitProjection::Extend { arg, signed }
    }
}

/// Empty data has one possible value. Tokens retain ordering dependencies even
/// though their flattened bit count is zero.
pub(crate) fn is_empty_data(ty: &Type) -> bool {
    ty.checked_bit_count() == Some(0) && !ty.contains_token()
}

/// Returns the operand and static element indices from outermost to innermost.
/// Out-of-range array reads select the last element, as in PIR evaluation.
pub(crate) fn aggregate_selection(
    graph: &NodeGraph,
    node: NodeRef,
    mut control: impl FnMut(NodeRef) -> Option<Rc<IrValue>>,
) -> Option<(NodeRef, SmallVec<[usize; 2]>)> {
    match &graph.get_node(node).payload {
        NodePayload::TupleIndex { tuple, index } => Some((*tuple, smallvec![*index])),
        NodePayload::ArrayIndex {
            array,
            indices,
            assumed_in_bounds,
        } => {
            let mut ty = graph.get_node_ty(*array);
            let mut path = SmallVec::new();
            for &index in indices {
                let Type::Array(dim) = ty else { return None };
                let index = index_value(graph, index, &mut control)?;
                if *assumed_in_bounds && index >= dim.element_count {
                    return None;
                }
                path.push(index.min(dim.element_count.checked_sub(1)?));
                ty = &dim.element_type;
            }
            Some((*array, path))
        }
        _ => None,
    }
}

/// The selected value forwarded by an aggregate operation, or the number of
/// elements required to describe it when different lanes have different
/// drivers.
pub(crate) enum AggregateProjection {
    Forward {
        node: NodeRef,
        path: SmallVec<[usize; 2]>,
    },
    Elements(usize),
    /// Elementwise OR of the selected cases, restricted to the requested path.
    Or(Vec<NodeRef>),
}

/// Converts an index to the host width only at an indexing operation.
fn index_value(
    graph: &NodeGraph,
    node: NodeRef,
    control: &mut impl FnMut(NodeRef) -> Option<Rc<IrValue>>,
) -> Option<usize> {
    literal_offset(graph, node).or_else(|| {
        let value = control(node)?;
        Some(ir_bits_to_usize(value.as_bits().ok()?).unwrap_or(usize::MAX))
    })
}

/// A fixed one-hot selector may enable zero, one, or multiple cases.
enum SelectedCases {
    One(NodeRef),
    Or(Vec<NodeRef>),
}

/// Selects live cases for a known control. Mask bits are numbered from the
/// least significant bit and keep their full PIR width.
fn project_select(
    graph: &NodeGraph,
    node: NodeRef,
    control: &mut impl FnMut(NodeRef) -> Option<Rc<IrValue>>,
) -> Option<SelectedCases> {
    let payload = &graph.get_node(node).payload;
    let (selector, cases, default) = match payload {
        NodePayload::Sel {
            selector,
            cases,
            default,
        } => {
            let index = index_value(graph, *selector, control)?;
            return Some(SelectedCases::One(*cases.get(index).or(default.as_ref())?));
        }
        NodePayload::PrioritySel {
            selector,
            cases,
            default,
        } => (*selector, cases, *default),
        NodePayload::OneHotSel { selector, cases } => (*selector, cases, None),
        _ => return None,
    };
    let value = control(selector)?;
    let mask = value.as_bits().ok()?;
    let mut live = cases.iter().enumerate().filter_map(|(i, &case)| {
        mask.get_bit(i)
            .expect("verified selector width")
            .then_some(case)
    });
    if let Some(default) = default {
        return Some(SelectedCases::One(live.next().unwrap_or(default)));
    }
    let arms: Vec<_> = live.collect();
    Some(match arms.as_slice() {
        [one] => SelectedCases::One(*one),
        _ => SelectedCases::Or(arms),
    })
}

/// Type reached by a sequence of in-bounds array and tuple indices.
pub(crate) fn element_type<'a>(mut ty: &'a Type, path: &[usize]) -> Option<&'a Type> {
    for &index in path {
        ty = match ty {
            Type::Array(dim) if index < dim.element_count => &dim.element_type,
            Type::Tuple(fields) => fields.get(index)?,
            _ => return None,
        };
    }
    Some(ty)
}

/// Applies one routing step to a statically selected element. Paths are in
/// outermost-first order. A prefix of a nested update can select a whole row
/// whose elements come from both the update value and the original array.
pub(crate) fn project_aggregate(
    graph: &NodeGraph,
    node: NodeRef,
    path: &[usize],
    mut control: impl FnMut(NodeRef) -> Option<Rc<IrValue>>,
) -> Option<AggregateProjection> {
    let same_path = |node| AggregateProjection::Forward {
        node,
        path: SmallVec::from_slice(path),
    };
    let split_array = || {
        let Type::Array(dim) = element_type(graph.get_node_ty(node), path)? else {
            return None;
        };
        // PIR does not permit an array_index result containing a token.
        (!dim.element_type.contains_token())
            .then_some(AggregateProjection::Elements(dim.element_count))
    };
    Some(match &graph.get_node(node).payload {
        NodePayload::Tuple(elems) | NodePayload::Array(elems) => {
            let (&index, rest) = path.split_first()?;
            AggregateProjection::Forward {
                node: *elems.get(index)?,
                path: SmallVec::from_slice(rest),
            }
        }
        NodePayload::Unop(Unop::Identity, arg) => same_path(*arg),
        NodePayload::Sel { .. }
        | NodePayload::PrioritySel { .. }
        | NodePayload::OneHotSel { .. } => match project_select(graph, node, &mut control)? {
            SelectedCases::One(node) => same_path(node),
            SelectedCases::Or(arms) => AggregateProjection::Or(arms),
        },
        NodePayload::ArrayIndex { .. } | NodePayload::TupleIndex { .. } => {
            let (source, mut selected) = aggregate_selection(graph, node, &mut control)?;
            selected.extend_from_slice(path);
            AggregateProjection::Forward {
                node: source,
                path: selected,
            }
        }
        NodePayload::ArrayConcat(arrays) => {
            let Some((&first, rest)) = path.split_first() else {
                return split_array();
            };
            let mut index = first;
            for &array in arrays {
                let Type::Array(dim) = graph.get_node_ty(array) else {
                    return None;
                };
                if index < dim.element_count {
                    let mut path = smallvec![index];
                    path.extend_from_slice(rest);
                    return Some(AggregateProjection::Forward { node: array, path });
                }
                index -= dim.element_count;
            }
            return None;
        }
        NodePayload::ArraySlice { array, start, .. } => {
            let start = index_value(graph, *start, &mut control)?;
            let Some((&index, rest)) = path.split_first() else {
                return split_array();
            };
            let Type::Array(dim) = graph.get_node_ty(*array) else {
                return None;
            };
            let source = start
                .saturating_add(index)
                .min(dim.element_count.checked_sub(1)?);
            let mut path = smallvec![source];
            path.extend_from_slice(rest);
            AggregateProjection::Forward { node: *array, path }
        }
        NodePayload::ArrayUpdate {
            array,
            value,
            indices,
            assumed_in_bounds,
        } => {
            let mut ty = graph.get_node_ty(*array);
            let mut matches = true;
            for (depth, &index) in indices.iter().enumerate() {
                let Type::Array(dim) = ty else { return None };
                let index = index_value(graph, index, &mut control)?;
                if index >= dim.element_count {
                    return (!assumed_in_bounds).then(|| same_path(*array));
                }
                matches &= path.get(depth).is_none_or(|&selected| selected == index);
                ty = &dim.element_type;
            }
            if !matches {
                same_path(*array)
            } else if path.len() < indices.len() {
                return split_array();
            } else {
                AggregateProjection::Forward {
                    node: *value,
                    path: SmallVec::from_slice(&path[indices.len()..]),
                }
            }
        }
        _ => return None,
    })
}

/// Resolves a read, or a constant select, when its result already has a node.
/// A repeated selection in the flattened graph can represent genuine feedback.
pub(crate) fn selected_aggregate_operand(
    graph: &NodeGraph,
    root: NodeRef,
    mut control: impl FnMut(NodeRef) -> Option<Rc<IrValue>>,
) -> Option<NodeRef> {
    if !matches!(
        graph.get_node(root).payload,
        NodePayload::ArrayIndex { .. }
            | NodePayload::TupleIndex { .. }
            | NodePayload::Sel { .. }
            | NodePayload::PrioritySel { .. }
            | NodePayload::OneHotSel { .. }
    ) {
        return None;
    }
    let mut node = root;
    let mut path = SmallVec::new();
    let mut seen = HashSet::new();
    loop {
        let AggregateProjection::Forward {
            node: source,
            path: next,
        } = project_aggregate(graph, node, &path, &mut control)?
        else {
            return None;
        };
        if next.is_empty() {
            return Some(source);
        }
        if !seen.insert((source, next.clone())) {
            return None;
        }
        node = source;
        path = next;
    }
}
