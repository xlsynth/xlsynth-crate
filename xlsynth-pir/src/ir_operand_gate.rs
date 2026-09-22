// SPDX-License-Identifier: Apache-2.0

//! Conditional replacement of selected bits at exact function operand sites.

use std::collections::BTreeMap;

use crate::IrValue;
use crate::ir::{self, NaryOp, Node, NodePayload, NodeRef, Type, Unop};
use crate::ir_utils;
use crate::ir_verify;

/// One operand slice to clamp when the shared predicate is true.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperandGateSite {
    pub consumer: String,
    pub operand: usize,
    /// Both absent means the entire operand.
    pub start: Option<usize>,
    pub width: Option<usize>,
    /// Unsigned decimal, hexadecimal (`0x`), or binary (`0b`) literal.
    pub clamp: String,
}

/// A validated transformation and the original predicate node reference.
#[derive(Debug)]
pub struct OperandGateTransform {
    pub function: ir::Fn,
    pub predicate: NodeRef,
}

struct ResolvedSite {
    start: usize,
    width: usize,
    value: IrValue,
}

/// Resolves a printed IR name or numeric text ID within a function.
pub fn resolve_node(f: &ir::Fn, selector: &str) -> Result<NodeRef, String> {
    let id = selector.parse::<usize>().ok();
    let hits: Vec<_> = f
        .node_refs()
        .into_iter()
        .filter(|&nr| {
            nr.index != 0
                && (id == Some(f.get_node(nr).text_id) || ir::node_textual_id(f, nr) == selector)
        })
        .collect();
    match hits.as_slice() {
        [nr] => Ok(*nr),
        [] => Err(format!(
            "node {selector:?} not found in function {:?}",
            f.name
        )),
        _ => Err(format!(
            "node {selector:?} is ambiguous in function {:?}",
            f.name
        )),
    }
}

fn ancestors(f: &ir::Fn, root: NodeRef) -> Vec<bool> {
    let mut seen = vec![false; f.nodes.len()];
    let mut pending = vec![root];
    while let Some(nr) = pending.pop() {
        if !seen[nr.index] {
            seen[nr.index] = true;
            pending.extend(ir_utils::operands(&f.get_node(nr).payload));
        }
    }
    seen
}

/// Appends a well-typed, unnamed node with a fresh function-local textual ID.
fn append_node(f: &mut ir::Fn, next_id: &mut usize, ty: Type, payload: NodePayload) -> NodeRef {
    let nr = NodeRef {
        index: f.nodes.len(),
    };
    f.nodes.push(Node {
        text_id: *next_id,
        name: None,
        ty,
        payload,
        pos: None,
    });
    *next_id += 1;
    nr
}

fn slice(f: &mut ir::Fn, next_id: &mut usize, arg: NodeRef, start: usize, width: usize) -> NodeRef {
    append_node(
        f,
        next_id,
        Type::Bits(width),
        NodePayload::BitSlice { arg, start, width },
    )
}

/// Produces a new function with all sites clamped simultaneously under `when`.
pub fn gate_operands(
    original: &ir::Fn,
    when: &str,
    sites: &[OperandGateSite],
) -> Result<OperandGateTransform, String> {
    gate_operands_with_context(original, None, when, sites)
}

/// Gates a function whose calls may require validation against its package.
pub fn gate_operands_in_package(
    original: &ir::Fn,
    package: &ir::Package,
    when: &str,
    sites: &[OperandGateSite],
) -> Result<OperandGateTransform, String> {
    gate_operands_with_context(original, Some(package), when, sites)
}

fn gate_operands_with_context(
    original: &ir::Fn,
    package: Option<&ir::Package>,
    when: &str,
    sites: &[OperandGateSite],
) -> Result<OperandGateTransform, String> {
    if sites.is_empty() {
        return Err("at least one operand site is required".to_string());
    }
    let predicate = resolve_node(original, when)?;
    if original.get_node_ty(predicate) != &Type::Bits(1) {
        return Err(format!("predicate {when:?} must have type bits[1]"));
    }
    let predicate_deps = ancestors(original, predicate);
    let return_ref = original
        .ret_node_ref
        .ok_or_else(|| "selected function has no return value".to_string())?;
    let observable = ancestors(original, return_ref);
    let mut grouped: BTreeMap<(usize, usize), Vec<ResolvedSite>> = BTreeMap::new();
    for site in sites {
        let consumer = resolve_node(original, &site.consumer)?;
        if predicate_deps[consumer.index] {
            return Err(format!("predicate depends on consumer {:?}", site.consumer));
        }
        if !observable[consumer.index] {
            return Err(format!(
                "consumer {:?} cannot affect the function return",
                site.consumer
            ));
        }
        let operands = ir_utils::operands(&original.get_node(consumer).payload);
        let source = operands.get(site.operand).copied().ok_or_else(|| {
            format!(
                "consumer {:?} has no operand {}",
                site.consumer, site.operand
            )
        })?;
        let Type::Bits(source_width) = original.get_node_ty(source) else {
            return Err(format!(
                "consumer {:?} operand {} is not bits-typed",
                site.consumer, site.operand
            ));
        };
        let (start, width) = match (site.start, site.width) {
            (None, None) => (0, *source_width),
            (Some(start), Some(width)) => (start, width),
            _ => return Err("--start and --width must be supplied together".to_string()),
        };
        if width == 0
            || start
                .checked_add(width)
                .is_none_or(|end| end > *source_width)
        {
            return Err(format!(
                "invalid slice start={start}, width={width} for bits[{source_width}]"
            ));
        }
        if site.clamp.starts_with('-') {
            return Err("clamp must be nonnegative".to_string());
        }
        let value = IrValue::parse_typed(&format!("bits[{width}]:{}", site.clamp))
            .map_err(|error| format!("invalid clamp {:?}: {error}", site.clamp))?;
        grouped
            .entry((consumer.index, site.operand))
            .or_default()
            .push(ResolvedSite {
                start,
                width,
                value,
            });
    }
    for group in grouped.values_mut() {
        group.sort_by_key(|site| site.start);
        let mut end = 0;
        for site in group {
            if site.start < end {
                return Err("overlapping or duplicate slices on the same operand".to_string());
            }
            end = site.start + site.width;
        }
    }

    let mut function = original.clone();
    let mut next_id = original
        .nodes
        .iter()
        .map(|node| node.text_id)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| "node ID space exhausted".to_string())?;
    for ((consumer_index, operand), group) in grouped {
        let consumer = NodeRef {
            index: consumer_index,
        };
        let source = ir_utils::operands(&original.get_node(consumer).payload)[operand];
        let Type::Bits(source_width) = original.get_node_ty(source) else {
            unreachable!("validated operand type");
        };
        let mut pieces_low_first = Vec::new();
        let mut offset = 0;
        for site in group {
            if offset < site.start {
                pieces_low_first.push(slice(
                    &mut function,
                    &mut next_id,
                    source,
                    offset,
                    site.start - offset,
                ));
            }
            let original_bits = if site.start == 0 && site.width == *source_width {
                source
            } else {
                slice(&mut function, &mut next_id, source, site.start, site.width)
            };
            let constant = append_node(
                &mut function,
                &mut next_id,
                Type::Bits(site.width),
                NodePayload::Literal(site.value),
            );
            let gated = append_node(
                &mut function,
                &mut next_id,
                Type::Bits(site.width),
                NodePayload::Sel {
                    selector: predicate,
                    cases: vec![original_bits, constant],
                    default: None,
                },
            );
            pieces_low_first.push(gated);
            offset = site.start + site.width;
        }
        if offset < *source_width {
            pieces_low_first.push(slice(
                &mut function,
                &mut next_id,
                source,
                offset,
                *source_width - offset,
            ));
        }
        pieces_low_first.reverse();
        let replacement = if pieces_low_first.len() == 1 {
            pieces_low_first[0]
        } else {
            append_node(
                &mut function,
                &mut next_id,
                Type::Bits(*source_width),
                NodePayload::Nary(NaryOp::Concat, pieces_low_first),
            )
        };
        ir_utils::replace_operand_with_ref(&mut function, consumer, operand, replacement)?;
    }
    let mapping = ir_utils::compact_and_toposort_with_mapping_in_place(&mut function)?;
    match package {
        Some(package) => ir_verify::verify_function_in_package(&function, package),
        None => ir_verify::verify_function(&function),
    }
    .map_err(|error| error.to_string())?;
    Ok(OperandGateTransform {
        function,
        predicate: mapping[predicate.index].expect("the gated function retains its predicate"),
    })
}

/// Constructs a one-bit property that is true precisely when `when` is false.
pub fn predicate_is_false_property(original: &ir::Fn, when: &str) -> Result<ir::Fn, String> {
    predicate_is_false_property_with_context(original, None, when)
}

/// Builds a predicate property whose calls are validated against `package`.
pub fn predicate_is_false_property_in_package(
    original: &ir::Fn,
    package: &ir::Package,
    when: &str,
) -> Result<ir::Fn, String> {
    predicate_is_false_property_with_context(original, Some(package), when)
}

fn predicate_is_false_property_with_context(
    original: &ir::Fn,
    package: Option<&ir::Package>,
    when: &str,
) -> Result<ir::Fn, String> {
    let predicate = resolve_node(original, when)?;
    if original.get_node_ty(predicate) != &Type::Bits(1) {
        return Err(format!("predicate {when:?} must have type bits[1]"));
    }
    let mut function = original.clone();
    let mut next_id = original
        .nodes
        .iter()
        .map(|node| node.text_id)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| "node ID space exhausted".to_string())?;
    let inverted = append_node(
        &mut function,
        &mut next_id,
        Type::Bits(1),
        NodePayload::Unop(Unop::Not, predicate),
    );
    function.ret_node_ref = Some(inverted);
    function.ret_ty = Type::Bits(1);
    ir_utils::compact_and_toposort_in_place(&mut function)?;
    match package {
        Some(package) => ir_verify::verify_function_in_package(&function, package),
        None => ir_verify::verify_function(&function),
    }
    .map_err(|error| error.to_string())?;
    Ok(function)
}

#[cfg(test)]
mod tests {
    use super::{OperandGateSite, gate_operands, predicate_is_false_property};
    use crate::IrValue;
    use crate::ir_eval::{FnEvalResult, eval_fn};
    use crate::ir_parser::Parser;

    const IR: &str = r#"fn main(x: bits[8] id=1, y: bits[8] id=2, enabled: bits[1] id=3) -> (bits[8], bits[8]) {
  sum: bits[8] = add(x, y, id=4)
  ret result: (bits[8], bits[8]) = tuple(sum, x, id=5)
}"#;

    fn site(start: Option<usize>, width: Option<usize>, clamp: &str) -> OperandGateSite {
        OperandGateSite {
            consumer: "sum".to_string(),
            operand: 0,
            start,
            width,
            clamp: clamp.to_string(),
        }
    }

    fn output(function: &crate::ir::Fn, enabled: u64) -> IrValue {
        let args = [
            IrValue::make_ubits(8, 0xab).unwrap(),
            IrValue::make_ubits(8, 1).unwrap(),
            IrValue::make_ubits(1, enabled).unwrap(),
        ];
        match eval_fn(function, &args) {
            FnEvalResult::Success(success) => success.value,
            other => panic!("unexpected evaluation result: {other:?}"),
        }
    }

    #[test]
    fn slices_and_other_users_remain_independent() {
        let original = Parser::new(IR).parse_fn().unwrap();
        let gated = gate_operands(&original, "enabled", &[site(Some(4), Some(4), "0")])
            .unwrap()
            .function;
        assert_eq!(output(&gated, 0), output(&original, 0));
        let want = IrValue::make_tuple(&[
            IrValue::make_ubits(8, 0x0c).unwrap(),
            IrValue::make_ubits(8, 0xab).unwrap(),
        ]);
        assert_eq!(output(&gated, 1), want);
        let reparsed = Parser::new(&gated.to_string()).parse_fn().unwrap();
        assert_eq!(output(&reparsed, 1), want);
    }

    #[test]
    fn disjoint_slices_and_entire_operand() {
        let original = Parser::new(IR).parse_fn().unwrap();
        let gated = gate_operands(
            &original,
            "enabled",
            &[site(Some(0), Some(4), "0x5"), site(Some(4), Some(4), "0xc")],
        )
        .unwrap()
        .function;
        assert_eq!(output(&gated, 0), output(&original, 0));
        assert_eq!(
            output(&gated, 1),
            IrValue::make_tuple(&[
                IrValue::make_ubits(8, 0xc6).unwrap(),
                IrValue::make_ubits(8, 0xab).unwrap(),
            ])
        );
        let entire = gate_operands(&original, "enabled", &[site(None, None, "0")]).unwrap();
        assert_eq!(
            output(&entire.function, 1),
            IrValue::make_tuple(&[
                IrValue::make_ubits(8, 1).unwrap(),
                IrValue::make_ubits(8, 0xab).unwrap(),
            ])
        );
    }

    #[test]
    fn validation_rejects_overlap_and_cycles_and_overflow() {
        let original = Parser::new(IR).parse_fn().unwrap();
        assert!(
            gate_operands(
                &original,
                "enabled",
                &[site(Some(0), Some(4), "0"), site(Some(3), Some(2), "0")]
            )
            .unwrap_err()
            .contains("overlapping")
        );
        assert!(
            gate_operands(&original, "enabled", &[site(Some(7), Some(2), "0")])
                .unwrap_err()
                .contains("invalid slice")
        );
        assert!(
            gate_operands(&original, "enabled", &[site(Some(0), Some(4), "0x10")])
                .unwrap_err()
                .contains("does not fit")
        );
        assert!(
            gate_operands(&original, "enabled", &[site(Some(0), None, "0")])
                .unwrap_err()
                .contains("together")
        );
        let dependent = Parser::new(
            r#"fn main(x: bits[1] id=1) -> bits[1] {
  not_x: bits[1] = not(x, id=2)
  ret consumer: bits[1] = and(not_x, x, id=3)
}"#,
        )
        .parse_fn()
        .unwrap();
        assert!(
            gate_operands(
                &dependent,
                "consumer",
                &[OperandGateSite {
                    consumer: "consumer".to_string(),
                    operand: 0,
                    start: None,
                    width: None,
                    clamp: "0".to_string(),
                }]
            )
            .unwrap_err()
            .contains("predicate depends")
        );
    }

    #[test]
    fn reachability_property_reads_original_predicate() {
        let original = Parser::new(IR).parse_fn().unwrap();
        let property = predicate_is_false_property(&original, "enabled").unwrap();
        let args = [
            IrValue::make_ubits(8, 0).unwrap(),
            IrValue::make_ubits(8, 0).unwrap(),
            IrValue::make_ubits(1, 1).unwrap(),
        ];
        let FnEvalResult::Success(value) = eval_fn(&property, &args) else {
            panic!("property evaluation failed");
        };
        assert_eq!(value.value, IrValue::make_ubits(1, 0).unwrap());
    }

    #[test]
    fn wide_clamp_and_connected_sites() {
        let original = Parser::new(
            r#"fn main(x: bits[129] id=1, p: bits[1] id=2) -> bits[129] {
  inverted: bits[129] = not(x, id=3)
  ret result: bits[129] = xor(inverted, inverted, id=4)
}"#,
        )
        .parse_fn()
        .unwrap();
        let gated = gate_operands(
            &original,
            "p",
            &[
                OperandGateSite {
                    consumer: "inverted".to_string(),
                    operand: 0,
                    start: Some(64),
                    width: Some(65),
                    clamp: "0x10000000000000000".to_string(),
                },
                OperandGateSite {
                    consumer: "result".to_string(),
                    operand: 0,
                    start: None,
                    width: None,
                    clamp: "0".to_string(),
                },
            ],
        )
        .unwrap()
        .function;
        let args = [
            IrValue::parse_typed("bits[129]:0x1234").unwrap(),
            IrValue::make_ubits(1, 0).unwrap(),
        ];
        let FnEvalResult::Success(original_off) = eval_fn(&original, &args) else {
            panic!("original eval failed");
        };
        let FnEvalResult::Success(gated_off) = eval_fn(&gated, &args) else {
            panic!("gated eval failed");
        };
        assert_eq!(original_off.value, gated_off.value);
        let args = [args[0].clone(), IrValue::make_ubits(1, 1).unwrap()];
        let FnEvalResult::Success(gated_on) = eval_fn(&gated, &args) else {
            panic!("gated eval failed");
        };
        assert_ne!(gated_on.value, original_off.value);
    }
}
