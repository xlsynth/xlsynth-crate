// SPDX-License-Identifier: Apache-2.0

//! Graph transfers, including type-shaped aggregate and block values.

use super::{AnalysisError, KnownBits, KnownValue, bits, extensions};
use crate::IrBits;
use crate::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use crate::ir_utils;

/// Obtains an operand that has already been visited in topological order.
fn value_at(values: &[Option<KnownValue>], node: NodeRef) -> Result<&KnownValue, AnalysisError> {
    values
        .get(node.index)
        .and_then(Option::as_ref)
        .ok_or_else(|| {
            AnalysisError::new(format!(
                "missing known-bits operand at index {}",
                node.index
            ))
        })
}

fn bits_at(values: &[Option<KnownValue>], node: NodeRef) -> Result<&KnownBits, AnalysisError> {
    value_at(values, node)?
        .as_bits()
        .ok_or_else(|| AnalysisError::new("known-bits operand is not bits-typed"))
}

/// Applies one scalar transfer independently to every aggregate leaf.
fn map_bits(value: &KnownValue, f: &impl Fn(&KnownBits) -> KnownBits) -> KnownValue {
    match value {
        KnownValue::Bits(value) => KnownValue::Bits(f(value)),
        KnownValue::Tuple(elements) => {
            KnownValue::Tuple(elements.iter().map(|value| map_bits(value, f)).collect())
        }
        KnownValue::Array(elements) => {
            KnownValue::Array(elements.iter().map(|value| map_bits(value, f)).collect())
        }
        KnownValue::Token => KnownValue::Token,
    }
}

/// Combines corresponding leaves without flattening tuple and array order.
fn zip_bits(
    lhs: &KnownValue,
    rhs: &KnownValue,
    f: &impl Fn(&KnownBits, &KnownBits) -> KnownBits,
) -> Result<KnownValue, AnalysisError> {
    let is_tuple = matches!(lhs, KnownValue::Tuple(_));
    match (lhs, rhs) {
        (KnownValue::Bits(lhs), KnownValue::Bits(rhs)) => Ok(KnownValue::Bits(f(lhs, rhs))),
        (KnownValue::Tuple(lhs), KnownValue::Tuple(rhs))
        | (KnownValue::Array(lhs), KnownValue::Array(rhs))
            if lhs.len() == rhs.len() =>
        {
            let mut elements = Vec::with_capacity(lhs.len());
            for (lhs, rhs) in lhs.iter().zip(rhs) {
                elements.push(zip_bits(lhs, rhs, f)?);
            }
            Ok(if is_tuple {
                KnownValue::Tuple(elements)
            } else {
                KnownValue::Array(elements)
            })
        }
        (KnownValue::Token, KnownValue::Token) => Ok(KnownValue::Token),
        _ => Err(AnalysisError::new("known-value aggregate shapes differ")),
    }
}

/// Reduces leaf equalities, including vacuously equal empty aggregates.
fn equal_values(lhs: &KnownValue, rhs: &KnownValue) -> Result<KnownBits, AnalysisError> {
    match (lhs, rhs) {
        (KnownValue::Bits(lhs), KnownValue::Bits(rhs)) => Ok(bits::binop(Binop::Eq, lhs, rhs, 1)),
        (KnownValue::Tuple(lhs), KnownValue::Tuple(rhs))
        | (KnownValue::Array(lhs), KnownValue::Array(rhs))
            if lhs.len() == rhs.len() =>
        {
            let mut result = KnownBits::constant(&IrBits::bool(true));
            for (lhs, rhs) in lhs.iter().zip(rhs) {
                let equal = equal_values(lhs, rhs)?;
                result = bits::nary(NaryOp::And, &[&result, &equal], 1);
            }
            Ok(result)
        }
        (KnownValue::Token, KnownValue::Token) => Ok(KnownBits::constant(&IrBits::bool(true))),
        _ => Err(AnalysisError::new(
            "equality operands have different aggregate shapes",
        )),
    }
}

fn join_into(result: &mut Option<KnownValue>, next: &KnownValue) -> Result<(), AnalysisError> {
    *result = Some(match result.take() {
        Some(previous) => previous.join(next)?,
        None => next.clone(),
    });
    Ok(())
}

/// Moves an already-owned first alternative into the accumulator.
fn join_owned_into(result: &mut Option<KnownValue>, next: KnownValue) -> Result<(), AnalysisError> {
    *result = Some(match result.take() {
        Some(previous) => previous.join(&next)?,
        None => next,
    });
    Ok(())
}

/// Returns whether the abstract index contains a particular host-sized index.
fn contains_index(index: &KnownBits, candidate: usize) -> bool {
    if index.bit_count() < usize::BITS as usize && candidate >> index.bit_count() != 0 {
        return false;
    }
    (0..index.bit_count()).all(|position| {
        !index.mask().get_bit(position).unwrap()
            || index.value().get_bit(position).unwrap()
                == (position < usize::BITS as usize && ((candidate >> position) & 1) != 0)
    })
}

/// Saturation retains all comparisons against structural array/case counts.
fn maximum_index(index: &KnownBits) -> usize {
    let mut result = 0usize;
    for position in 0..index.bit_count() {
        if !index.mask().get_bit(position).unwrap() || index.value().get_bit(position).unwrap() {
            if position >= usize::BITS as usize {
                return usize::MAX;
            }
            result |= 1usize << position;
        }
    }
    result
}

/// Enumerates a bounded abstract index; large numeric values saturate to OOB.
fn possible_indices(index: &KnownBits) -> Vec<usize> {
    let mut base = 0usize;
    let mut unknown = Vec::new();
    for (position, bit) in index.lsb_bits().iter().enumerate() {
        match bit {
            Some(true) => {
                if position >= usize::BITS as usize {
                    return vec![usize::MAX];
                }
                base |= 1usize << position;
            }
            None => unknown.push(position),
            Some(false) => {
                // Known-zero positions do not contribute to the index.
            }
        }
    }
    debug_assert!(unknown.len() < 10);
    let mut result = Vec::with_capacity(1usize << unknown.len());
    for assignment in 0..(1usize << unknown.len()) {
        let mut value = base;
        for (variable, position) in unknown.iter().enumerate() {
            if assignment & (1usize << variable) != 0 {
                if *position >= usize::BITS as usize {
                    value = usize::MAX;
                    break;
                }
                value |= 1usize << position;
            }
        }
        result.push(value);
    }
    result.sort_unstable();
    result.dedup();
    result
}

/// Mirrors the eager XLS engine's bound on uncertain array-index bits.
fn index_scan_is_expensive(indices: &[&KnownBits]) -> bool {
    indices
        .iter()
        .map(|index| index.bit_count() - index.known_bit_count())
        .fold(0usize, usize::saturating_add)
        >= 10
}

/// Scans a small physical array instead of enumerating a wide index domain.
fn bounded_array_positions(
    index: &KnownBits,
    length: usize,
    result_width: usize,
    clamp: bool,
) -> Option<Vec<usize>> {
    // Bound both index membership checks and the following value joins. This
    // alternate path is intentionally limited to one-dimensional accesses.
    let per_position = index.bit_count().saturating_add(result_width).max(1);
    if length > 512 || length.saturating_mul(per_position) > 65_536 {
        return None;
    }
    let maximum = maximum_index(index);
    Some(
        (0..length)
            .filter(|&position| {
                (clamp && position == length - 1 && maximum >= position)
                    || contains_index(index, position)
            })
            .collect(),
    )
}

/// Reads all feasible multidimensional positions, clamping each dimension.
fn array_index(
    array: &KnownValue,
    indices: &[Vec<usize>],
    result_type: &Type,
) -> Result<KnownValue, AnalysisError> {
    let Some((index, remaining)) = indices.split_first() else {
        return Ok(array.clone());
    };
    let KnownValue::Array(elements) = array else {
        return Err(AnalysisError::new("array index operand is not an array"));
    };
    if elements.is_empty() {
        return Ok(KnownValue::unknown(result_type));
    }
    let mut result = None;
    for &index in index {
        let selected = &elements[index.min(elements.len() - 1)];
        join_owned_into(&mut result, array_index(selected, remaining, result_type)?)?;
        if index >= elements.len() - 1 {
            break;
        }
    }
    result.ok_or_else(|| AnalysisError::new("array index has no feasible index"))
}

/// Writes exactly known positions, or joins the update into every possible one.
fn array_update(
    array: &mut KnownValue,
    indices: &[Vec<usize>],
    update: &KnownValue,
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
    let KnownValue::Array(elements) = array else {
        return Err(AnalysisError::new("array update operand is not an array"));
    };
    for &index in index {
        let Some(element) = elements.get_mut(index) else {
            // Unlike reads, out-of-bounds array writes leave the array alone.
            break;
        };
        array_update(element, remaining, update, exact)?;
    }
    Ok(())
}

/// Evaluates one validated node from already available operand facts.
pub(super) fn evaluate(
    node: &ir::Node,
    values: &[Option<KnownValue>],
    graph: &ir::NodeGraph,
) -> Result<KnownValue, AnalysisError> {
    let output_width = node
        .ty
        .checked_bit_count()
        .ok_or_else(|| AnalysisError::new("node width overflows usize"))?;
    if node.payload.is_extension_op() {
        let references = ir_utils::operands(&node.payload);
        let operands = references
            .iter()
            .map(|reference| bits_at(values, *reference))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(extensions::evaluate(node, &operands));
    }
    // These transfers only copy/reindex existing facts or construct literals;
    // they do not enumerate uncertain alternatives as aggregate operations can.
    let cheap_aggregate = match &node.payload {
        NodePayload::Literal(_)
        | NodePayload::Unop(Unop::Identity, _)
        | NodePayload::Tuple(_)
        | NodePayload::Array(_)
        | NodePayload::TupleIndex { .. }
        | NodePayload::ArrayConcat(_) => true,
        NodePayload::ArrayIndex { indices, .. } | NodePayload::ArrayUpdate { indices, .. } => {
            indices.iter().try_fold(true, |known, index| {
                Ok::<_, AnalysisError>(known && bits_at(values, *index)?.is_fully_known())
            })?
        }
        NodePayload::ArraySlice { start, .. } => bits_at(values, *start)?.is_fully_known(),
        _ => false,
    };
    if !cheap_aggregate && !matches!(node.ty, Type::Bits(_)) && output_width > 65_536 {
        return Ok(KnownValue::unknown(&node.ty));
    }
    match &node.payload {
        NodePayload::Binop(op @ (Binop::Shll | Binop::Shrl | Binop::Shra), lhs, rhs) => {
            if let Some(result) = bits::try_shift_fixed(
                *op,
                bits_at(values, *lhs)?,
                bits_at(values, *rhs)?,
                output_width,
            ) {
                return Ok(KnownValue::Bits(result));
            }
            if output_width > 256 {
                return Ok(KnownValue::unknown(&node.ty));
            }
        }
        NodePayload::BitSliceUpdate {
            arg,
            start,
            update_value,
        } => {
            if let Some(result) = bits::try_slice_update_fixed(
                bits_at(values, *arg)?,
                bits_at(values, *start)?,
                bits_at(values, *update_value)?,
            ) {
                return Ok(KnownValue::Bits(result));
            }
            if output_width > 256 {
                return Ok(KnownValue::unknown(&node.ty));
            }
        }
        _ => {
            // Other operations retain their existing transfer and work limits.
        }
    }
    let result = match &node.payload {
        NodePayload::Literal(value) => KnownValue::constant(value),
        NodePayload::Tuple(operands) | NodePayload::Array(operands) => {
            let elements = operands
                .iter()
                .map(|operand| value_at(values, *operand).cloned())
                .collect::<Result<Vec<_>, _>>()?;
            if matches!(node.payload, NodePayload::Tuple(_)) {
                KnownValue::Tuple(elements)
            } else {
                KnownValue::Array(elements)
            }
        }
        NodePayload::TupleIndex { tuple, index } => {
            let KnownValue::Tuple(elements) = value_at(values, *tuple)? else {
                return Err(AnalysisError::new("tuple_index operand is not a tuple"));
            };
            elements
                .get(*index)
                .cloned()
                .ok_or_else(|| AnalysisError::new("tuple_index is out of bounds"))?
        }
        NodePayload::ArrayConcat(operands) => {
            let mut elements = Vec::new();
            for operand in operands {
                let KnownValue::Array(part) = value_at(values, *operand)? else {
                    return Err(AnalysisError::new("array_concat operand is not an array"));
                };
                elements.extend(part.iter().cloned());
            }
            KnownValue::Array(elements)
        }
        NodePayload::ArrayIndex { array, indices, .. }
        | NodePayload::ArrayUpdate { array, indices, .. } => {
            let indices = indices
                .iter()
                .map(|index| bits_at(values, *index))
                .collect::<Result<Vec<_>, _>>()?;
            let array = value_at(values, *array)?;
            let exact = indices.iter().all(|index| index.is_fully_known());
            let possibilities = if index_scan_is_expensive(&indices) {
                let (KnownValue::Array(elements), [index]) = (array, indices.as_slice()) else {
                    // Large multidimensional domains retain the bounded
                    // fallback.
                    return Ok(KnownValue::unknown(&node.ty));
                };
                let clamp = matches!(node.payload, NodePayload::ArrayIndex { .. });
                let Some(positions) =
                    bounded_array_positions(index, elements.len(), output_width, clamp)
                else {
                    // Scanning the actual array would exceed the work budget
                    // too.
                    return Ok(KnownValue::unknown(&node.ty));
                };
                vec![positions]
            } else {
                indices
                    .iter()
                    .map(|index| possible_indices(index))
                    .collect::<Vec<_>>()
            };
            if let NodePayload::ArrayUpdate { value, .. } = &node.payload {
                let mut result = array.clone();
                array_update(
                    &mut result,
                    &possibilities,
                    value_at(values, *value)?,
                    exact,
                )?;
                result
            } else {
                array_index(array, &possibilities, &node.ty)?
            }
        }
        NodePayload::ArraySlice {
            array,
            start,
            width,
        } => {
            let start = bits_at(values, *start)?;
            let KnownValue::Array(elements) = value_at(values, *array)? else {
                return Err(AnalysisError::new("array_slice operand is not an array"));
            };
            if elements.is_empty() {
                return Ok(KnownValue::unknown(&node.ty));
            }
            let starts = if index_scan_is_expensive(&[start]) {
                let Some(starts) =
                    bounded_array_positions(start, elements.len(), output_width, true)
                else {
                    // Both index-domain enumeration and physical scanning are
                    // costly.
                    return Ok(KnownValue::unknown(&node.ty));
                };
                starts
            } else {
                possible_indices(start)
            };
            let mut result = None;
            for start in starts {
                let slice = KnownValue::Array(
                    (0..*width)
                        .map(|offset| {
                            elements[start.saturating_add(offset).min(elements.len() - 1)].clone()
                        })
                        .collect(),
                );
                join_owned_into(&mut result, slice)?;
                if start >= elements.len() - 1 {
                    break;
                }
            }
            result.ok_or_else(|| AnalysisError::new("array_slice has no feasible start"))?
        }
        NodePayload::Binop(Binop::Eq | Binop::Ne, lhs, rhs) => {
            let mut equal = equal_values(value_at(values, *lhs)?, value_at(values, *rhs)?)?;
            if matches!(node.payload, NodePayload::Binop(Binop::Ne, ..)) {
                equal = bits::unop(Unop::Not, &equal);
            }
            KnownValue::Bits(equal)
        }
        NodePayload::Binop(Binop::Gate, condition, data) => {
            let condition = bits_at(values, *condition)?;
            map_bits(value_at(values, *data)?, &|data| {
                bits::binop(Binop::Gate, condition, data, data.bit_count())
            })
        }
        NodePayload::Binop(Binop::Umulp | Binop::Smulp, ..) => KnownValue::unknown(&node.ty),
        NodePayload::Binop(op, lhs, rhs) => KnownValue::Bits(bits::binop(
            *op,
            bits_at(values, *lhs)?,
            bits_at(values, *rhs)?,
            output_width,
        )),
        NodePayload::Unop(Unop::Identity, operand) => value_at(values, *operand)?.clone(),
        NodePayload::Unop(op, operand) => {
            KnownValue::Bits(bits::unop(*op, bits_at(values, *operand)?))
        }
        NodePayload::Nary(op, operands) => {
            let operands = operands
                .iter()
                .map(|operand| bits_at(values, *operand))
                .collect::<Result<Vec<_>, _>>()?;
            KnownValue::Bits(bits::nary(*op, &operands, output_width))
        }
        NodePayload::SignExt { arg, new_bit_count }
        | NodePayload::ZeroExt { arg, new_bit_count } => KnownValue::Bits(bits::resize(
            bits_at(values, *arg)?,
            *new_bit_count,
            matches!(node.payload, NodePayload::SignExt { .. }),
        )),
        NodePayload::BitSlice { arg, start, width } => {
            KnownValue::Bits(bits::slice(bits_at(values, *arg)?, *start, *width))
        }
        NodePayload::DynamicBitSlice { arg, start, width } => KnownValue::Bits(
            bits::dynamic_slice(bits_at(values, *arg)?, bits_at(values, *start)?, *width),
        ),
        NodePayload::BitSliceUpdate {
            arg,
            start,
            update_value,
        } => KnownValue::Bits(bits::slice_update(
            bits_at(values, *arg)?,
            bits_at(values, *start)?,
            bits_at(values, *update_value)?,
        )),
        NodePayload::OneHot { arg, lsb_prio } => {
            KnownValue::Bits(bits::one_hot(bits_at(values, *arg)?, *lsb_prio))
        }
        NodePayload::Decode { arg, width } => {
            KnownValue::Bits(bits::decode(bits_at(values, *arg)?, *width))
        }
        NodePayload::Encode { arg } => KnownValue::Bits(bits::encode(bits_at(values, *arg)?)),
        NodePayload::Sel {
            selector,
            cases,
            default,
        } => {
            let selector = bits_at(values, *selector)?;
            let mut result = None;
            for (index, case) in cases.iter().enumerate() {
                if contains_index(selector, index) {
                    join_into(&mut result, value_at(values, *case)?)?;
                }
            }
            if let Some(default) = default.filter(|_| maximum_index(selector) >= cases.len()) {
                join_into(&mut result, value_at(values, default)?)?;
            }
            result.unwrap_or_else(|| KnownValue::unknown(&node.ty))
        }
        NodePayload::PrioritySel {
            selector,
            cases,
            default,
        } => {
            let known_nonzero = matches!(
                graph.get_node(*selector).payload,
                NodePayload::OneHot { .. }
            );
            let selector = bits_at(values, *selector)?.lsb_bits();
            let mut result = None;
            let mut can_reach_default = true;
            for (index, case) in cases.iter().enumerate() {
                if selector[index] != Some(false) {
                    join_into(&mut result, value_at(values, *case)?)?;
                }
                if selector[index] == Some(true) {
                    can_reach_default = false;
                    break;
                }
            }
            if can_reach_default && !known_nonzero {
                let default =
                    default.ok_or_else(|| AnalysisError::new("priority_sel requires a default"))?;
                join_into(&mut result, value_at(values, default)?)?;
            }
            result.unwrap_or_else(|| KnownValue::unknown(&node.ty))
        }
        NodePayload::OneHotSel { selector, cases } => {
            let known_nonzero = matches!(
                graph.get_node(*selector).payload,
                NodePayload::OneHot { .. }
            );
            let selector = bits_at(values, *selector)?.lsb_bits();
            let mut result = map_bits(&KnownValue::unknown(&node.ty), &|leaf| {
                KnownBits::constant(&IrBits::zero(leaf.bit_count()))
            });
            let mut common = None;
            for (index, case) in cases.iter().enumerate() {
                let case = value_at(values, *case)?;
                let condition = KnownBits::from_lsb_bits(&[selector[index]]);
                let gated = map_bits(case, &|leaf| {
                    bits::binop(Binop::Gate, &condition, leaf, leaf.bit_count())
                });
                result = zip_bits(&result, &gated, &|lhs, rhs| {
                    bits::nary(NaryOp::Or, &[lhs, rhs], lhs.bit_count())
                })?;
                if known_nonzero && selector[index] != Some(false) {
                    common = Some(match common {
                        Some(previous) => zip_bits(&previous, case, &|lhs, rhs| {
                            bits::nary(NaryOp::And, &[lhs, rhs], lhs.bit_count())
                        })?,
                        None => case.clone(),
                    });
                }
            }
            if let Some(common) = common {
                result = zip_bits(&result, &common, &|lhs, rhs| {
                    bits::nary(NaryOp::Or, &[lhs, rhs], lhs.bit_count())
                })?;
            }
            result
        }
        NodePayload::OutputPort { .. }
        | NodePayload::RegisterWrite { .. }
        | NodePayload::InstantiationInput { .. }
        | NodePayload::Cover { .. } => KnownValue::Tuple(Vec::new()),
        NodePayload::Assert { .. } | NodePayload::Trace { .. } | NodePayload::AfterAll(_) => {
            KnownValue::Token
        }
        // Sources and opaque calls/instances have no local transfer.
        // Reset/write data never constrain a register's arbitrary current value.
        NodePayload::Nil
        | NodePayload::Param
        | NodePayload::InputPort { .. }
        | NodePayload::RegisterRead { .. }
        | NodePayload::InstantiationOutput { .. }
        | NodePayload::Invoke { .. }
        | NodePayload::CountedFor { .. } => KnownValue::unknown(&node.ty),
        NodePayload::ExtCarryOut { .. }
        | NodePayload::ExtPrioEncode { .. }
        | NodePayload::ExtClz { .. }
        | NodePayload::ExtNormalizeLeft { .. }
        | NodePayload::ExtMaskLow { .. }
        | NodePayload::ExtNaryAdd { .. } => {
            unreachable!("extensions are dispatched before work limits")
        }
    };
    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::ir_parser::Parser;
    use crate::known_bits::analyze_fn;

    fn constant(value: u64) -> KnownValue {
        KnownValue::Bits(KnownBits::constant(&IrBits::make_ubits(4, value).unwrap()))
    }

    #[test]
    fn array_reads_clamp_each_dimension_and_merge_possible_leaves() {
        let array = KnownValue::Array(vec![
            KnownValue::Array(vec![constant(1), constant(3)]),
            KnownValue::Array(vec![constant(5), constant(7)]),
        ]);
        let clamped = array_index(&array, &[vec![1, usize::MAX], vec![0]], &Type::Bits(4)).unwrap();
        assert_eq!(clamped, constant(5));
        let merged = array_index(&array, &[vec![0, 1], vec![0, 1]], &Type::Bits(4)).unwrap();
        assert_eq!(merged.as_bits().unwrap().to_ternary_string(), "0XX1");
        assert_eq!(array_index(&array, &[], &Type::Bits(4)).unwrap(), array);
    }

    #[test]
    fn array_writes_replace_exact_locations_but_join_uncertain_ones() {
        let original = KnownValue::Array(vec![constant(1), constant(2)]);
        let mut exact = original.clone();
        array_update(&mut exact, &[vec![1]], &constant(15), true).unwrap();
        assert_eq!(exact, KnownValue::Array(vec![constant(1), constant(15)]));

        let mut out_of_bounds = original.clone();
        array_update(&mut out_of_bounds, &[vec![usize::MAX]], &constant(15), true).unwrap();
        assert_eq!(out_of_bounds, original);

        let mut uncertain = original;
        array_update(&mut uncertain, &[vec![0, 1]], &constant(15), false).unwrap();
        assert_eq!(uncertain.leaf(&[0]).unwrap().to_ternary_string(), "XXX1");
        assert_eq!(uncertain.leaf(&[1]).unwrap().to_ternary_string(), "XX1X");
    }

    #[test]
    fn wide_indices_saturate_without_truncation_and_scan_limit_is_inclusive() {
        let mut value = vec![false; 129];
        value[128] = true;
        let index = KnownBits::constant(&IrBits::from_lsb_is_0(&value));
        assert_eq!(possible_indices(&index), vec![usize::MAX]);
        assert_eq!(maximum_index(&index), usize::MAX);
        assert!(!contains_index(&index, 0));
        assert!(!contains_index(&index, usize::MAX));
        assert!(!index_scan_is_expensive(&[&KnownBits::unknown(9)]));
        assert!(index_scan_is_expensive(&[&KnownBits::unknown(10)]));
        assert!(index_scan_is_expensive(&[
            &KnownBits::unknown(6),
            &KnownBits::unknown(4)
        ]));
    }

    #[test]
    fn physical_array_scan_matches_small_enumerated_domains() {
        for width in 0..=6 {
            for pattern in 0..3usize.pow(width as u32) {
                let mut digits = pattern;
                let index = KnownBits::from_lsb_bits(
                    &(0..width)
                        .map(|_| {
                            let bit = [Some(false), Some(true), None][digits % 3];
                            digits /= 3;
                            bit
                        })
                        .collect::<Vec<_>>(),
                );
                let enumerated = possible_indices(&index);
                for length in 0..=8 {
                    for clamp in [false, true] {
                        let mut expected = enumerated
                            .iter()
                            .filter_map(|&position| {
                                if length == 0 {
                                    None
                                } else if clamp {
                                    Some(position.min(length - 1))
                                } else {
                                    (position < length).then_some(position)
                                }
                            })
                            .collect::<Vec<_>>();
                        expected.sort_unstable();
                        expected.dedup();
                        assert_eq!(
                            bounded_array_positions(&index, length, 4, clamp),
                            Some(expected),
                            "width={width} pattern={pattern} length={length} clamp={clamp}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn physical_array_scan_is_bounded_and_preserves_wide_oob_indices() {
        let unknown = KnownBits::unknown(129);
        assert_eq!(
            bounded_array_positions(&unknown, 3, 4, true),
            Some(vec![0, 1, 2])
        );
        assert_eq!(bounded_array_positions(&unknown, 0, 4, false), Some(vec![]));
        assert_eq!(bounded_array_positions(&unknown, 513, 0, true), None);
        assert_eq!(bounded_array_positions(&unknown, 2, 65_536, true), None);
        let mut bits = vec![None; 129];
        bits[128] = Some(true);
        let out_of_bounds = KnownBits::from_lsb_bits(&bits);
        assert_eq!(
            bounded_array_positions(&out_of_bounds, 3, 4, true),
            Some(vec![2])
        );
        assert_eq!(
            bounded_array_positions(&out_of_bounds, 3, 4, false),
            Some(vec![])
        );
    }

    #[test]
    fn aggregate_leaf_operations_preserve_empty_shapes_and_tokens() {
        let value = KnownValue::Tuple(vec![
            KnownValue::Array(Vec::new()),
            KnownValue::Tuple(Vec::new()),
            KnownValue::Token,
            constant(5),
        ]);
        let joined = zip_bits(&value, &value, &|lhs, rhs| {
            bits::nary(NaryOp::Or, &[lhs, rhs], lhs.bit_count())
        })
        .unwrap();
        assert_eq!(joined, value);
        assert_eq!(
            equal_values(&value, &value).unwrap().to_ternary_string(),
            "1"
        );
    }

    #[test]
    fn owned_join_seed_retains_aggregate_storage_and_join_errors() {
        for value in [
            KnownValue::Tuple(vec![constant(5), KnownValue::Array(vec![constant(1); 17])]),
            KnownValue::Array(vec![constant(3); 17]),
        ] {
            let original_storage = value.elements().unwrap().as_ptr();
            let expected = value.clone();
            let mut result = None;
            join_owned_into(&mut result, value).unwrap();
            assert_eq!(
                result.as_ref().unwrap().elements().unwrap().as_ptr(),
                original_storage,
            );
            assert_eq!(result.as_ref(), Some(&expected));
            join_owned_into(&mut result, expected.clone()).unwrap();
            assert_eq!(result, Some(expected));
            assert!(join_owned_into(&mut result, KnownValue::Token).is_err());
            assert!(result.is_none());
        }
    }

    #[test]
    fn aggregate_zip_preserves_wide_leaves_and_stops_at_nested_shape_error() {
        let wide = KnownValue::Bits(KnownBits::constant(&IrBits::make_ubits(129, 7).unwrap()));
        let value = KnownValue::Tuple(vec![
            wide.clone(),
            KnownValue::Array(vec![wide.clone(); 17]),
            KnownValue::Array(vec![]),
            KnownValue::Tuple(vec![]),
            KnownValue::Token,
            KnownValue::Bits(KnownBits::unknown(0)),
        ]);
        assert_eq!(zip_bits(&value, &value, &KnownBits::join).unwrap(), value);

        let lhs = KnownValue::Tuple(vec![
            wide.clone(),
            KnownValue::Array(vec![KnownValue::Tuple(vec![])]),
            wide.clone(),
        ]);
        let rhs = KnownValue::Tuple(vec![
            wide.clone(),
            KnownValue::Array(vec![KnownValue::Array(vec![])]),
            wide,
        ]);
        let visits = Cell::new(0);
        let error = zip_bits(&lhs, &rhs, &|a, b| {
            visits.set(visits.get() + 1);
            a.join(b)
        })
        .unwrap_err();
        assert_eq!(error.to_string(), "known-value aggregate shapes differ");
        assert_eq!(
            visits.get(),
            1,
            "later leaves must not be visited after an error"
        );
    }

    #[test]
    fn multidimensional_reads_join_wide_aggregates_and_clamp_each_dimension() {
        let element = |value| {
            KnownValue::Tuple(vec![
                KnownValue::Bits(KnownBits::constant(
                    &IrBits::make_ubits(129, value).unwrap(),
                )),
                KnownValue::Array(vec![]),
                KnownValue::Tuple(vec![]),
                KnownValue::Bits(KnownBits::unknown(0)),
            ])
        };
        let element_type = Type::Tuple(vec![
            Box::new(Type::Bits(129)),
            Box::new(Type::new_array(Type::Bits(8), 0)),
            Box::new(Type::nil()),
            Box::new(Type::Bits(0)),
        ]);
        let array = KnownValue::Array(vec![
            KnownValue::Array(vec![element(1), element(3)]),
            KnownValue::Array(vec![element(5), element(7)]),
        ]);
        assert_eq!(
            array_index(&array, &[vec![1], vec![0]], &element_type).unwrap(),
            element(5),
        );
        assert_eq!(
            array_index(&array, &[vec![usize::MAX], vec![usize::MAX]], &element_type).unwrap(),
            element(7),
        );
        let merged =
            array_index(&array, &[vec![0, 1], vec![0, 1, usize::MAX]], &element_type).unwrap();
        assert!(merged.matches_type(&element_type));
        assert_eq!(
            merged.leaf(&[0]).unwrap().to_ternary_string(),
            format!("{}XX1", "0".repeat(126)),
        );
        let row_type = Type::new_array(element_type.clone(), 2);
        assert_eq!(
            array_index(&array, &[vec![usize::MAX]], &row_type).unwrap(),
            KnownValue::Array(vec![element(5), element(7)]),
        );
        let empty = KnownValue::Array(vec![]);
        assert_eq!(
            array_index(&empty, &[vec![0]], &element_type).unwrap(),
            KnownValue::unknown(&element_type),
        );
    }

    #[test]
    fn direct_one_hot_selectors_prove_nonzero_for_aggregate_selects() {
        let text = r#"package selectors
top fn sample(x: bits[2] id=1, arbitrary: bits[3] id=2) -> ((bits[4], ()), (bits[4], ()), (bits[4], ())) {
  selector: bits[3] = one_hot(x, lsb_prio=true, id=3)
  ones: bits[4] = literal(value=15, id=4)
  zero: bits[4] = literal(value=0, id=5)
  empty: () = tuple(id=6)
  all_ones: (bits[4], ()) = tuple(ones, empty, id=7)
  all_zero: (bits[4], ()) = tuple(zero, empty, id=8)
  selected: (bits[4], ()) = one_hot_sel(selector, cases=[all_ones, all_ones, all_ones], id=9)
  priority: (bits[4], ()) = priority_sel(selector, cases=[all_ones, all_ones, all_ones], default=all_zero, id=10)
  may_be_zero: (bits[4], ()) = one_hot_sel(arbitrary, cases=[all_ones, all_ones, all_ones], id=11)
  ret result: ((bits[4], ()), (bits[4], ()), (bits[4], ())) = tuple(selected, priority, may_be_zero, id=12)
}
"#;
        let package = Parser::new(text).parse_and_validate_package().unwrap();
        let function = package.get_top_fn().unwrap();
        let analysis = analyze_fn(function).unwrap();
        let result = analysis.get(function.ret_node_ref.unwrap()).unwrap();
        assert_eq!(result.leaf(&[0, 0]).unwrap().to_ternary_string(), "1111");
        assert_eq!(result.leaf(&[1, 0]).unwrap().to_ternary_string(), "1111");
        assert_eq!(result.leaf(&[2, 0]).unwrap().to_ternary_string(), "XXXX");
    }

    #[test]
    fn one_hot_select_combines_every_asserted_case() {
        let text = r#"package multi_hot
top fn sample() -> bits[4] {
  selector: bits[3] = literal(value=3, id=1)
  one: bits[4] = literal(value=1, id=2)
  two: bits[4] = literal(value=2, id=3)
  eight: bits[4] = literal(value=8, id=4)
  ret result: bits[4] = one_hot_sel(selector, cases=[one, two, eight], id=5)
}
"#;
        let package = Parser::new(text).parse_and_validate_package().unwrap();
        let function = package.get_top_fn().unwrap();
        let analysis = analyze_fn(function).unwrap();
        assert_eq!(
            analysis
                .bits(function.ret_node_ref.unwrap())
                .unwrap()
                .to_ternary_string(),
            "0011"
        );
    }
}
