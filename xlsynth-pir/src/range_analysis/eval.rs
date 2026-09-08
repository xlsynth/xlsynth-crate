// SPDX-License-Identifier: Apache-2.0

//! Type-shaped graph transfers over normalized interval sets.

use super::policy::RESULT_INTERVALS;
use super::{AnalysisError, IntervalSet, RangeValue, extensions, ops};
use crate::IrBits;
use crate::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use crate::ir_utils;
use crate::known_bits::KnownBits;

/// Borrows an operand already evaluated in topological order.
fn value_at(values: &[Option<RangeValue>], node: NodeRef) -> Result<&RangeValue, AnalysisError> {
    values
        .get(node.index)
        .and_then(Option::as_ref)
        .ok_or_else(|| AnalysisError::new(format!("missing range operand at index {}", node.index)))
}

fn bits_at(values: &[Option<RangeValue>], node: NodeRef) -> Result<&IntervalSet, AnalysisError> {
    value_at(values, node)?
        .as_bits()
        .ok_or_else(|| AnalysisError::new("range operand is not bits-typed"))
}

/// Applies a scalar transfer to each leaf without flattening aggregate paths.
pub(super) fn map_bits(value: &RangeValue, f: &impl Fn(&IntervalSet) -> IntervalSet) -> RangeValue {
    match value {
        RangeValue::Bits(bits) => RangeValue::Bits(f(bits)),
        RangeValue::Tuple(elements) => {
            RangeValue::Tuple(elements.iter().map(|v| map_bits(v, f)).collect())
        }
        RangeValue::Array(elements) => {
            RangeValue::Array(elements.iter().map(|v| map_bits(v, f)).collect())
        }
        RangeValue::Token => RangeValue::Token,
    }
}

/// Combines corresponding leaves, rejecting incompatible aggregate shapes.
pub(super) fn zip_bits(
    lhs: &RangeValue,
    rhs: &RangeValue,
    f: &impl Fn(&IntervalSet, &IntervalSet) -> IntervalSet,
) -> Result<RangeValue, AnalysisError> {
    match (lhs, rhs) {
        (RangeValue::Bits(a), RangeValue::Bits(b)) if a.width() == b.width() => {
            Ok(RangeValue::Bits(f(a, b)))
        }
        (RangeValue::Tuple(a), RangeValue::Tuple(b))
        | (RangeValue::Array(a), RangeValue::Array(b))
            if a.len() == b.len() =>
        {
            let mut elements = Vec::with_capacity(a.len());
            for (a, b) in a.iter().zip(b) {
                elements.push(zip_bits(a, b, f)?);
            }
            Ok(if matches!(lhs, RangeValue::Tuple(_)) {
                RangeValue::Tuple(elements)
            } else {
                RangeValue::Array(elements)
            })
        }
        (RangeValue::Token, RangeValue::Token) => Ok(RangeValue::Token),
        _ => Err(AnalysisError::new("range-value aggregate shapes differ")),
    }
}

/// Computes aggregate equality, including vacuously equal empty shapes.
fn equal_values(lhs: &RangeValue, rhs: &RangeValue) -> Result<IntervalSet, AnalysisError> {
    match (lhs, rhs) {
        (RangeValue::Bits(a), RangeValue::Bits(b)) if a.width() == b.width() => {
            Ok(ops::binop(Binop::Eq, a, b, 1))
        }
        (RangeValue::Tuple(a), RangeValue::Tuple(b))
        | (RangeValue::Array(a), RangeValue::Array(b))
            if a.len() == b.len() =>
        {
            let mut result = IntervalSet::singleton(IrBits::bool(true));
            for (a, b) in a.iter().zip(b) {
                result = ops::nary(NaryOp::And, &[&result, &equal_values(a, b)?], 1);
            }
            Ok(result)
        }
        (RangeValue::Token, RangeValue::Token) => Ok(IntervalSet::singleton(IrBits::bool(true))),
        _ => Err(AnalysisError::new(
            "equality operands have different aggregate shapes",
        )),
    }
}

/// Moves the first owned alternative rather than cloning its aggregate again.
fn join_owned(result: &mut Option<RangeValue>, next: RangeValue) -> Result<(), AnalysisError> {
    *result = Some(match result.take() {
        Some(previous) => previous.join(&next)?,
        None => next,
    });
    Ok(())
}

fn join_borrowed(result: &mut Option<RangeValue>, next: &RangeValue) -> Result<(), AnalysisError> {
    *result = Some(match result.take() {
        Some(previous) => previous.join(next)?,
        None => next.clone(),
    });
    Ok(())
}

/// Tests structural bounds without truncating wide numeric indices.
fn maximum_index(index: &IntervalSet) -> usize {
    index
        .unsigned_max()
        .map(|v| ops::saturating_index(v, usize::MAX))
        .unwrap_or(0)
}

/// Reads feasible physical positions, independently clamping each dimension.
fn array_index(
    array: &RangeValue,
    indices: &[&IntervalSet],
    result_type: &Type,
) -> Result<RangeValue, AnalysisError> {
    let Some((index, remaining)) = indices.split_first() else {
        return Ok(array.clone());
    };
    let RangeValue::Array(elements) = array else {
        return Err(AnalysisError::new("array_index operand is not an array"));
    };
    if elements.is_empty() {
        return Ok(RangeValue::unknown(result_type));
    }
    if let Some(position) = index.singleton_value() {
        let position = ops::saturating_index(position, elements.len() - 1);
        return array_index(&elements[position], remaining, result_type);
    }
    let mut result = None;
    for (position, element) in elements.iter().enumerate() {
        if index.contains_usize(position)
            || (position == elements.len() - 1 && maximum_index(index) >= position)
        {
            join_owned(&mut result, array_index(element, remaining, result_type)?)?;
        }
    }
    // Physical scanning bounds work by the array shape, not the numeric index
    // domain. Like XLS, array routing preserves exact leaf unions.
    result.ok_or_else(|| AnalysisError::new("array_index has no feasible position"))
}

/// Replaces exact writes; uncertain or out-of-bounds alternatives retain data.
fn array_update(
    array: &mut RangeValue,
    indices: &[&IntervalSet],
    update: &RangeValue,
    exact: bool,
) -> Result<(), AnalysisError> {
    let Some((index, remaining)) = indices.split_first() else {
        *array = if exact {
            update.clone()
        } else {
            array.join(update)?
        };
        return Ok(());
    };
    let RangeValue::Array(elements) = array else {
        return Err(AnalysisError::new("array_update operand is not an array"));
    };
    if let Some(position) = index.singleton_value() {
        let position = ops::saturating_index(position, elements.len());
        if let Some(element) = elements.get_mut(position) {
            array_update(element, remaining, update, exact)?;
        }
        return Ok(());
    }
    for (position, element) in elements.iter_mut().enumerate() {
        if index.contains_usize(position) {
            array_update(element, remaining, update, exact)?;
        }
    }
    Ok(())
}

/// Applies one operation to validated operand facts.
pub(super) fn evaluate(
    node: &ir::Node,
    values: &[Option<RangeValue>],
) -> Result<RangeValue, AnalysisError> {
    let width = node
        .ty
        .checked_bit_count()
        .ok_or_else(|| AnalysisError::new("node width overflows usize"))?;
    if node.payload.is_extension_op() {
        let refs = ir_utils::operands(&node.payload);
        let operands = refs
            .iter()
            .map(|r| bits_at(values, *r))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(extensions::evaluate(node, &operands));
    }
    Ok(match &node.payload {
        NodePayload::Literal(value) => RangeValue::constant(value),
        NodePayload::Tuple(operands) | NodePayload::Array(operands) => {
            let elements = operands
                .iter()
                .map(|r| value_at(values, *r).cloned())
                .collect::<Result<Vec<_>, _>>()?;
            if matches!(node.payload, NodePayload::Tuple(_)) {
                RangeValue::Tuple(elements)
            } else {
                RangeValue::Array(elements)
            }
        }
        NodePayload::TupleIndex { tuple, index } => {
            let RangeValue::Tuple(elements) = value_at(values, *tuple)? else {
                return Err(AnalysisError::new("tuple_index operand is not a tuple"));
            };
            elements
                .get(*index)
                .cloned()
                .ok_or_else(|| AnalysisError::new("tuple_index out of bounds"))?
        }
        NodePayload::ArrayConcat(operands) => {
            let Type::Array(array_type) = &node.ty else {
                unreachable!("validated array_concat has an array result")
            };
            let mut elements = Vec::with_capacity(array_type.element_count);
            for operand in operands {
                let RangeValue::Array(part) = value_at(values, *operand)? else {
                    return Err(AnalysisError::new("array_concat operand is not an array"));
                };
                elements.extend(part.iter().cloned());
            }
            RangeValue::Array(elements)
        }
        NodePayload::ArrayIndex { array, indices, .. }
        | NodePayload::ArrayUpdate { array, indices, .. } => {
            let indices = indices
                .iter()
                .map(|r| bits_at(values, *r))
                .collect::<Result<Vec<_>, _>>()?;
            let array = value_at(values, *array)?;
            if let NodePayload::ArrayUpdate { value, .. } = &node.payload {
                let mut result = array.clone();
                array_update(
                    &mut result,
                    &indices,
                    value_at(values, *value)?,
                    indices.iter().all(|v| v.singleton_value().is_some()),
                )?;
                result
            } else {
                array_index(array, &indices, &node.ty)?
            }
        }
        NodePayload::ArraySlice {
            array,
            start,
            width,
        } => {
            let start = bits_at(values, *start)?;
            let RangeValue::Array(elements) = value_at(values, *array)? else {
                return Err(AnalysisError::new("array_slice operand is not an array"));
            };
            if elements.is_empty() {
                return Ok(RangeValue::unknown(&node.ty));
            }
            if let Some(start) = start.singleton_value() {
                let start = ops::saturating_index(start, elements.len() - 1);
                return Ok(RangeValue::Array(
                    (0..*width)
                        .map(|offset| {
                            elements[start.saturating_add(offset).min(elements.len() - 1)].clone()
                        })
                        .collect(),
                ));
            }
            let mut result = None;
            for position in 0..elements.len() {
                if start.contains_usize(position)
                    || (position == elements.len() - 1 && maximum_index(start) >= position)
                {
                    let slice = RangeValue::Array(
                        (0..*width)
                            .map(|offset| {
                                elements[position.saturating_add(offset).min(elements.len() - 1)]
                                    .clone()
                            })
                            .collect(),
                    );
                    join_owned(&mut result, slice)?;
                }
            }
            result.ok_or_else(|| AnalysisError::new("array_slice has no feasible start"))?
        }
        NodePayload::Binop(Binop::Eq | Binop::Ne, lhs, rhs) => {
            let eq = equal_values(value_at(values, *lhs)?, value_at(values, *rhs)?)?;
            RangeValue::Bits(
                if matches!(node.payload, NodePayload::Binop(Binop::Ne, ..)) {
                    ops::unop(Unop::Not, &eq, 1)
                } else {
                    eq
                },
            )
        }
        NodePayload::Binop(Binop::Gate, condition, data) => {
            let condition = bits_at(values, *condition)?;
            map_bits(value_at(values, *data)?, &|data| {
                ops::binop(Binop::Gate, condition, data, data.width())
            })
        }
        NodePayload::Binop(Binop::Umulp | Binop::Smulp, ..) => RangeValue::unknown(&node.ty),
        NodePayload::Binop(op, lhs, rhs) => RangeValue::Bits(ops::binop(
            *op,
            bits_at(values, *lhs)?,
            bits_at(values, *rhs)?,
            width,
        )),
        NodePayload::Unop(Unop::Identity, operand) => value_at(values, *operand)?.clone(),
        NodePayload::Unop(op, operand) => {
            RangeValue::Bits(ops::unop(*op, bits_at(values, *operand)?, width))
        }
        NodePayload::Nary(op, operands) => {
            let operands = operands
                .iter()
                .map(|r| bits_at(values, *r))
                .collect::<Result<Vec<_>, _>>()?;
            RangeValue::Bits(ops::nary(*op, &operands, width))
        }
        NodePayload::SignExt { arg, new_bit_count }
        | NodePayload::ZeroExt { arg, new_bit_count } => RangeValue::Bits(ops::extend(
            bits_at(values, *arg)?,
            *new_bit_count,
            matches!(node.payload, NodePayload::SignExt { .. }),
        )),
        NodePayload::BitSlice { arg, start, width } => {
            RangeValue::Bits(ops::bit_slice(bits_at(values, *arg)?, *start, *width))
        }
        NodePayload::DynamicBitSlice { arg, start, width } => RangeValue::Bits(
            ops::dynamic_bit_slice(bits_at(values, *arg)?, bits_at(values, *start)?, *width),
        ),
        NodePayload::BitSliceUpdate {
            arg,
            start,
            update_value,
        } => RangeValue::Bits(ops::bit_slice_update(
            bits_at(values, *arg)?,
            bits_at(values, *start)?,
            bits_at(values, *update_value)?,
        )),
        NodePayload::OneHot { arg, lsb_prio } => {
            RangeValue::Bits(ops::one_hot(bits_at(values, *arg)?, *lsb_prio))
        }
        NodePayload::Decode { arg, width } => {
            RangeValue::Bits(ops::decode(bits_at(values, *arg)?, *width))
        }
        NodePayload::Encode { arg } => RangeValue::Bits(ops::encode(bits_at(values, *arg)?, width)),
        NodePayload::Sel {
            selector,
            cases,
            default,
        } => {
            let selector = bits_at(values, *selector)?;
            let mut result = None;
            for (index, case) in cases.iter().enumerate() {
                if selector.contains_usize(index) {
                    join_borrowed(&mut result, value_at(values, *case)?)?;
                }
            }
            if let Some(default) = default.filter(|_| maximum_index(selector) >= cases.len()) {
                join_borrowed(&mut result, value_at(values, default)?)?;
            }
            map_bits(
                &result.unwrap_or_else(|| RangeValue::unknown(&node.ty)),
                &|leaf| leaf.minimize(RESULT_INTERVALS),
            )
        }
        NodePayload::PrioritySel {
            selector,
            cases,
            default,
        } => {
            let selector = bits_at(values, *selector)?;
            let mut result = None;
            if selector.contains_usize(0) {
                let default =
                    default.ok_or_else(|| AnalysisError::new("priority_sel requires a default"))?;
                join_borrowed(&mut result, value_at(values, default)?)?;
            }
            for (index, case) in cases.iter().enumerate() {
                let pattern = KnownBits::from_mask_value(
                    IrBits::from_lsb_fn(selector.width(), |bit| bit <= index),
                    IrBits::from_lsb_fn(selector.width(), |bit| bit == index),
                )
                .unwrap();
                if selector.intersects_known_bits(&pattern) {
                    join_borrowed(&mut result, value_at(values, *case)?)?;
                }
            }
            map_bits(
                &result.unwrap_or_else(|| RangeValue::unknown(&node.ty)),
                &|leaf| leaf.minimize(RESULT_INTERVALS),
            )
        }
        NodePayload::OneHotSel { selector, cases } => {
            let selector = bits_at(values, *selector)?.to_known_bits();
            let mut result = None;
            for (index, case) in cases.iter().enumerate() {
                let known = selector.mask().get_bit(index).unwrap();
                let bit = selector.value().get_bit(index).unwrap();
                if known && !bit {
                    continue;
                }
                let case = value_at(values, *case)?;
                let gated = if known {
                    case.clone()
                } else {
                    map_bits(case, &|leaf| {
                        leaf.union(&IntervalSet::singleton(IrBits::zero(leaf.width())))
                    })
                };
                result = Some(match result {
                    None => gated,
                    Some(previous) => zip_bits(&previous, &gated, &|a, b| {
                        ops::nary(NaryOp::Or, &[a, b], a.width())
                    })?,
                });
            }
            let result = result.unwrap_or_else(|| {
                map_bits(&RangeValue::unknown(&node.ty), &|leaf| {
                    IntervalSet::singleton(IrBits::zero(leaf.width()))
                })
            });
            map_bits(&result, &|leaf| leaf.minimize(RESULT_INTERVALS))
        }
        NodePayload::OutputPort { .. }
        | NodePayload::RegisterWrite { .. }
        | NodePayload::InstantiationInput { .. }
        | NodePayload::Cover { .. } => RangeValue::Tuple(Vec::new()),
        NodePayload::Assert { .. } | NodePayload::Trace { .. } | NodePayload::AfterAll(_) => {
            RangeValue::Token
        }
        // Opaque sources are unconstrained; reset and write data do not impose
        // restrictions on a register's arbitrary current value.
        NodePayload::Nil
        | NodePayload::Param
        | NodePayload::InputPort { .. }
        | NodePayload::RegisterRead { .. }
        | NodePayload::InstantiationOutput { .. }
        | NodePayload::Invoke { .. }
        | NodePayload::CountedFor { .. } => RangeValue::unknown(&node.ty),
        NodePayload::ExtCarryOut { .. }
        | NodePayload::ExtPrioEncode { .. }
        | NodePayload::ExtClz { .. }
        | NodePayload::ExtNormalizeLeft { .. }
        | NodePayload::ExtMaskLow { .. }
        | NodePayload::ExtNaryAdd { .. } => unreachable!("extensions are dispatched above"),
    })
}
