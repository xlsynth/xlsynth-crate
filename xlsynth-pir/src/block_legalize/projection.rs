// SPDX-License-Identifier: Apache-2.0

//! Exact recipes for routing static selections through wiring operations.

use crate::ir::{NaryOp, NodeGraph, NodePayload, NodeRef, Type, Unop};

/// A field path followed by an optional half-open range of the selected bits.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct Selection {
    pub(super) path: Vec<usize>,
    pub(super) bits: Option<(usize, usize)>,
}

impl Selection {
    /// Selects an entire value without expanding aggregate fields.
    pub(super) fn whole(ty: &Type) -> Self {
        Self {
            path: Vec::new(),
            bits: match ty {
                Type::Bits(width) => Some((0, *width)),
                _ => None,
            },
        }
    }

    /// Returns the selected result type, rejecting invalid paths and ranges.
    pub(super) fn selected_type(&self, mut ty: &Type) -> Option<Type> {
        for &index in &self.path {
            ty = match ty {
                Type::Tuple(fields) => fields.get(index)?,
                Type::Array(array) if index < array.element_count => &array.element_type,
                _ => return None,
            };
        }
        match (self.bits, ty) {
            (Some((lo, hi)), Type::Bits(width)) if lo <= hi && hi <= *width => {
                Some(Type::Bits(hi - lo))
            }
            (None, _) => Some(ty.clone()),
            _ => None,
        }
    }

    /// Adds an aggregate field at the front of this selection.
    fn prepend(&self, index: usize) -> Self {
        let mut path = Vec::with_capacity(self.path.len() + 1);
        path.push(index);
        path.extend_from_slice(&self.path);
        Self {
            path,
            bits: self.bits,
        }
    }
}

/// A selection of one original node's value.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct Demand {
    pub(super) node: NodeRef,
    pub(super) part: Selection,
}

impl Demand {
    /// Demands a complete original graph value.
    pub(super) fn whole(block: &NodeGraph, node: NodeRef) -> Self {
        Self {
            node,
            part: Selection::whole(block.get_node_ty(node)),
        }
    }

    /// Forwards this selection to a different node.
    fn at(&self, node: NodeRef) -> Self {
        Self {
            node,
            part: self.part.clone(),
        }
    }

    /// Selects a half-open bit range of an operand.
    fn bits(node: NodeRef, lo: usize, hi: usize) -> Self {
        Self {
            node,
            part: Selection {
                path: Vec::new(),
                bits: Some((lo, hi)),
            },
        }
    }
}

/// An exact node expression after the operand selections have been resolved.
#[derive(Debug)]
pub(super) enum Recipe {
    Forward(Demand),
    Concat(Vec<Demand>),
    Reverse(Demand),
    Zero(usize),
    Extend {
        arg: Demand,
        signed: bool,
        width: usize,
    },
    Tuple(Vec<Demand>),
    Array(Vec<Demand>),
}

impl Recipe {
    /// Returns the selected operands in expression order.
    pub(super) fn inputs(&self) -> &[Demand] {
        match self {
            Self::Forward(arg) | Self::Reverse(arg) | Self::Extend { arg, .. } => {
                std::slice::from_ref(arg)
            }
            Self::Concat(args) | Self::Tuple(args) | Self::Array(args) => args,
            Self::Zero(_) => &[],
        }
    }
}

/// Reports whether projecting this selection must materialize array elements.
///
/// Callers can defer this potentially large rewrite until inexpensive scalar
/// and field projections have failed to remove the same feedback component.
pub(super) fn needs_aggregate_expansion(block: &NodeGraph, demand: &Demand) -> bool {
    let Some(node) = block.nodes.get(demand.node.index) else {
        return false;
    };
    if !matches!(demand.part.selected_type(&node.ty), Some(Type::Array(_))) {
        return false;
    }
    match &node.payload {
        NodePayload::ArraySlice { start, .. } => {
            demand.part.path.is_empty() && literal_index(block, *start).is_some()
        }
        NodePayload::ArrayUpdate {
            array,
            indices,
            assumed_in_bounds,
            ..
        } => {
            matches!(literal_indices(block, *array, indices, *assumed_in_bounds, false), Indices::Known(indices) if demand.part.path.len() < indices.len() && !indices.iter().zip(&demand.part.path).any(|(a, b)| a != b))
        }
        _ => false,
    }
}

/// Projects a static selection exactly; other operations remain whole nodes.
pub(super) fn project(block: &NodeGraph, demand: &Demand) -> Option<Recipe> {
    let node = block.nodes.get(demand.node.index)?;
    demand.part.selected_type(&node.ty)?;
    let part = &demand.part;
    let make = |node, part| Demand { node, part };
    Some(match &node.payload {
        NodePayload::Unop(Unop::Identity, arg) => Recipe::Forward(demand.at(*arg)),
        NodePayload::BitSlice { arg, start, .. } if part.path.is_empty() => {
            let (lo, hi) = part.bits?;
            Recipe::Forward(Demand::bits(
                *arg,
                start.checked_add(lo)?,
                start.checked_add(hi)?,
            ))
        }
        NodePayload::Unop(Unop::Reverse, arg) if part.path.is_empty() => {
            let (lo, hi) = part.bits?;
            let Type::Bits(width) = block.get_node_ty(*arg) else {
                return None;
            };
            Recipe::Reverse(Demand::bits(
                *arg,
                width.checked_sub(hi)?,
                width.checked_sub(lo)?,
            ))
        }
        NodePayload::Nary(NaryOp::Concat, args) if part.path.is_empty() => {
            let (lo, hi) = part.bits?;
            let mut base = 0usize;
            let mut selected = Vec::new();
            for &arg in args.iter().rev() {
                let Type::Bits(width) = block.get_node_ty(arg) else {
                    return None;
                };
                let end = base.checked_add(*width)?;
                let first = lo.max(base);
                let last = hi.min(end);
                if first < last {
                    selected.push(Demand::bits(arg, first - base, last - base));
                }
                base = end;
                if base >= hi {
                    break;
                }
            }
            selected.reverse();
            match selected.len() {
                0 => Recipe::Zero(hi - lo),
                1 => Recipe::Forward(selected.pop()?),
                _ => Recipe::Concat(selected),
            }
        }
        NodePayload::ZeroExt { arg, .. } | NodePayload::SignExt { arg, .. }
            if part.path.is_empty() =>
        {
            let (lo, hi) = part.bits?;
            let Type::Bits(source_width) = block.get_node_ty(*arg) else {
                return None;
            };
            let signed = matches!(node.payload, NodePayload::SignExt { .. });
            if lo == hi {
                Recipe::Zero(0)
            } else if hi <= *source_width {
                Recipe::Forward(Demand::bits(*arg, lo, hi))
            } else if lo < *source_width {
                Recipe::Extend {
                    arg: Demand::bits(*arg, lo, *source_width),
                    signed,
                    width: hi - lo,
                }
            } else if signed && *source_width != 0 {
                Recipe::Extend {
                    arg: Demand::bits(*arg, source_width - 1, *source_width),
                    signed: true,
                    width: hi - lo,
                }
            } else {
                Recipe::Zero(hi - lo)
            }
        }
        NodePayload::Tuple(values) | NodePayload::Array(values) => {
            if let Some((&index, rest)) = part.path.split_first() {
                Recipe::Forward(make(
                    *values.get(index)?,
                    Selection {
                        path: rest.to_vec(),
                        bits: part.bits,
                    },
                ))
            } else {
                let values = values
                    .iter()
                    .map(|&arg| Demand::whole(block, arg))
                    .collect();
                if matches!(node.payload, NodePayload::Tuple(_)) {
                    Recipe::Tuple(values)
                } else {
                    Recipe::Array(values)
                }
            }
        }
        NodePayload::TupleIndex { tuple, index } => {
            Recipe::Forward(make(*tuple, part.prepend(*index)))
        }
        NodePayload::ArrayConcat(arrays) => {
            let (&index, rest) = part.path.split_first()?;
            let mut offset = index;
            let mut found = None;
            for &array in arrays {
                let Type::Array(ty) = block.get_node_ty(array) else {
                    return None;
                };
                if offset < ty.element_count {
                    let mut path = vec![offset];
                    path.extend_from_slice(rest);
                    found = Some(make(
                        array,
                        Selection {
                            path,
                            bits: part.bits,
                        },
                    ));
                    break;
                }
                offset -= ty.element_count;
            }
            Recipe::Forward(found?)
        }
        NodePayload::ArrayIndex {
            array,
            indices,
            assumed_in_bounds,
        } => {
            let Indices::Known(mut path) =
                literal_indices(block, *array, indices, *assumed_in_bounds, true)
            else {
                return None;
            };
            path.extend_from_slice(&part.path);
            Recipe::Forward(make(
                *array,
                Selection {
                    path,
                    bits: part.bits,
                },
            ))
        }
        NodePayload::ArraySlice { array, start, .. } => {
            let start = literal_index(block, *start)?;
            if let Some((&index, rest)) = part.path.split_first() {
                let Type::Array(ty) = block.get_node_ty(*array) else {
                    return None;
                };
                let last = ty.element_count.checked_sub(1)?;
                let mut path = vec![start.saturating_add(index).min(last)];
                path.extend_from_slice(rest);
                Recipe::Forward(make(
                    *array,
                    Selection {
                        path,
                        bits: part.bits,
                    },
                ))
            } else {
                split_array(block, demand)?
            }
        }
        NodePayload::ArrayUpdate {
            array,
            value,
            indices,
            assumed_in_bounds,
        } => match literal_indices(block, *array, indices, *assumed_in_bounds, false) {
            Indices::OutOfBounds => Recipe::Forward(demand.at(*array)),
            Indices::Unknown => return None,
            Indices::Known(indices) => {
                if indices.iter().zip(&part.path).any(|(a, b)| a != b) {
                    Recipe::Forward(demand.at(*array))
                } else if part.path.len() < indices.len() {
                    split_array(block, demand)?
                } else {
                    Recipe::Forward(make(
                        *value,
                        Selection {
                            path: part.path[indices.len()..].to_vec(),
                            bits: part.bits,
                        },
                    ))
                }
            }
        },
        _ => return None,
    })
}

/// Expands only a genuinely mixed aggregate selection into its element demands.
fn split_array(block: &NodeGraph, demand: &Demand) -> Option<Recipe> {
    let Type::Array(ty) = demand.part.selected_type(block.get_node_ty(demand.node))? else {
        return None;
    };
    let mut elements = Vec::with_capacity(ty.element_count);
    for index in 0..ty.element_count {
        let mut path = demand.part.path.clone();
        path.push(index);
        let mut part = Selection::whole(&ty.element_type);
        part.path = path;
        elements.push(Demand {
            node: demand.node,
            part,
        });
    }
    Some(Recipe::Array(elements))
}

/// Decodes a literal index; values wider than usize behave as enormous indices.
fn literal_index(block: &NodeGraph, index: NodeRef) -> Option<usize> {
    let NodePayload::Literal(value) = &block.get_node(index).payload else {
        return None;
    };
    let bits = value.as_bits().ok()?;
    Some(crate::ir_value_utils::ir_bits_to_usize(bits).unwrap_or(usize::MAX))
}

enum Indices {
    Known(Vec<usize>),
    OutOfBounds,
    Unknown,
}

/// Applies XLS's clamping-read and ignored-update rules to literal indices.
fn literal_indices(
    block: &NodeGraph,
    array: NodeRef,
    indices: &[NodeRef],
    assumed: bool,
    clamp: bool,
) -> Indices {
    let mut ty = block.get_node_ty(array);
    let mut result = Vec::with_capacity(indices.len());
    for &index in indices {
        let Type::Array(dim) = ty else {
            return Indices::Unknown;
        };
        let Some(raw) = literal_index(block, index) else {
            return Indices::Unknown;
        };
        if raw >= dim.element_count {
            if assumed {
                return Indices::Unknown;
            }
            if !clamp {
                return Indices::OutOfBounds;
            }
        }
        let Some(last) = dim.element_count.checked_sub(1) else {
            return Indices::Unknown;
        };
        result.push(raw.min(last));
        ty = &dim.element_type;
    }
    Indices::Known(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Block;
    use crate::ir_parser::Parser;

    /// Returns a named node in the small verified projection fixture.
    fn named(block: &Block, name: &str) -> NodeRef {
        NodeRef {
            index: block
                .nodes
                .iter()
                .position(|node| node.name.as_deref() == Some(name))
                .unwrap(),
        }
    }

    /// Builds independent read, update, and extension operations for boundary
    /// checks.
    fn fixture() -> Block {
        let text = r#"package projection

top block root(seed: bits[1], out: bits[1]) {
  seed: bits[1] = input_port(name=seed, id=1)
  pair: bits[1][2] = array(seed, seed, id=2)
  oob: bits[2] = literal(value=3, id=3)
  one: bits[2] = literal(value=1, id=4)
  read: bits[1] = array_index(pair, indices=[oob], id=5)
  unchanged: bits[1][2] = array_update(pair, seed, indices=[oob], id=6)
  changed: bits[1][2] = array_update(pair, seed, indices=[one], id=7)
  extended: bits[4] = sign_ext(seed, new_bit_count=4, id=8)
  out: () = output_port(read, name=out, id=9)
}"#;
        Parser::new(text)
            .parse_and_verify_package()
            .unwrap()
            .get_top_block()
            .unwrap()
            .clone()
    }

    #[test]
    fn out_of_bounds_read_clamps_but_update_is_ignored() {
        let block = fixture();
        let read = Demand::whole(&block, named(&block, "read"));
        let Recipe::Forward(selected) = project(&block, &read).unwrap() else {
            panic!("expected a static read");
        };
        assert_eq!(
            selected,
            Demand {
                node: named(&block, "pair"),
                part: Selection {
                    path: vec![1],
                    bits: Some((0, 1))
                }
            }
        );

        let unchanged = Demand::whole(&block, named(&block, "unchanged"));
        assert!(!needs_aggregate_expansion(&block, &unchanged));
        let Recipe::Forward(original) = project(&block, &unchanged).unwrap() else {
            panic!("expected an ignored update");
        };
        assert_eq!(original, Demand::whole(&block, named(&block, "pair")));

        let changed = Demand::whole(&block, named(&block, "changed"));
        assert!(needs_aggregate_expansion(&block, &changed));
    }

    #[test]
    fn selecting_only_sign_extension_padding_reuses_the_sign_bit() {
        let block = fixture();
        let padded = Demand {
            node: named(&block, "extended"),
            part: Selection {
                path: Vec::new(),
                bits: Some((2, 4)),
            },
        };
        let Recipe::Extend { arg, signed, width } = project(&block, &padded).unwrap() else {
            panic!("expected a sign-bit extension");
        };
        assert_eq!(arg, Demand::whole(&block, named(&block, "seed")));
        assert!(signed);
        assert_eq!(width, 2);
    }
}
