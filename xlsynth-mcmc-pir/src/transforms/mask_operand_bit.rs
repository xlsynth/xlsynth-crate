// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrBits, IrValue};
use xlsynth_pir::ir::{Fn as IrFn, NaryOp, Node, NodePayload, NodeRef, Type};
use xlsynth_pir::ir_utils::remap_payload_with;

use super::{PirTransform, PirTransformKind, TransformLocation};

#[derive(Debug)]
enum MaskBitPosition {
    Low,
    High,
}

fn next_text_id(f: &IrFn) -> usize {
    f.nodes
        .iter()
        .map(|n| n.text_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
    match f.get_node(r).ty {
        Type::Bits(w) => Some(w),
        _ => None,
    }
}

fn operand_pairs(payload: &NodePayload) -> Vec<(usize, NodeRef)> {
    let mut pairs: Vec<(usize, NodeRef)> = Vec::new();
    let _ = remap_payload_with(payload, |(slot, dep)| {
        pairs.push((slot, dep));
        dep
    });
    pairs
}

fn mask_value(w: usize, position: &MaskBitPosition) -> IrValue {
    let mut bits = vec![true; w];
    match position {
        MaskBitPosition::Low => bits[0] = false,
        MaskBitPosition::High => bits[w.saturating_sub(1)] = false,
    }
    IrValue::from_bits(&IrBits::from_lsb_is_0(&bits))
}

fn make_literal_mask_node(f: &mut IrFn, w: usize, position: &MaskBitPosition) -> NodeRef {
    let text_id = next_text_id(f);
    let new_index = f.nodes.len();
    f.nodes.push(Node {
        text_id,
        name: None,
        ty: Type::Bits(w),
        payload: NodePayload::Literal(mask_value(w, position)),
        pos: None,
    });
    NodeRef { index: new_index }
}

fn make_and_node(f: &mut IrFn, w: usize, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
    let text_id = next_text_id(f);
    let new_index = f.nodes.len();
    f.nodes.push(Node {
        text_id,
        name: None,
        ty: Type::Bits(w),
        payload: NodePayload::Nary(NaryOp::And, vec![lhs, rhs]),
        pos: None,
    });
    NodeRef { index: new_index }
}

fn try_unmask_dep(f: &IrFn, dep: NodeRef, w: usize, position: &MaskBitPosition) -> Option<NodeRef> {
    let (a, b) = match &f.get_node(dep).payload {
        NodePayload::Nary(NaryOp::And, ops) if ops.len() == 2 => (ops[0], ops[1]),
        _ => return None,
    };
    let expected = mask_value(w, position);

    let a_node = f.get_node(a);
    let b_node = f.get_node(b);

    let a_is_expected_mask = matches!(&a_node.payload, NodePayload::Literal(v) if *v == expected)
        && a_node.ty == Type::Bits(w);
    let b_is_expected_mask = matches!(&b_node.payload, NodePayload::Literal(v) if *v == expected)
        && b_node.ty == Type::Bits(w);

    if a_is_expected_mask && b_node.ty == Type::Bits(w) {
        return Some(b);
    }
    if b_is_expected_mask && a_node.ty == Type::Bits(w) {
        return Some(a);
    }
    None
}

fn apply_mask_transform(
    f: &mut IrFn,
    loc: &TransformLocation,
    position: MaskBitPosition,
    transform_name: &str,
) -> Result<(), String> {
    let (node_ref, operand_slot) = match loc {
        TransformLocation::RewireOperand {
            node,
            operand_slot,
            new_operand: _,
        } => (*node, *operand_slot),
        TransformLocation::Node(_) => {
            return Err(format!(
                "{transform_name}: expected TransformLocation::RewireOperand, got Node"
            ));
        }
    };

    if node_ref.index >= f.nodes.len() {
        return Err(format!("{transform_name}: node ref out of bounds"));
    }

    let old_payload = f.get_node(node_ref).payload.clone();
    let pairs = operand_pairs(&old_payload);
    let (_, old_dep) = pairs
        .iter()
        .find(|(slot, _)| *slot == operand_slot)
        .copied()
        .ok_or_else(|| format!("{transform_name}: operand slot not found"))?;

    let Some(w) = bits_width(f, old_dep) else {
        return Err(format!("{transform_name}: operand is not bits-typed"));
    };
    if w < 2 {
        return Err(format!(
            "{transform_name}: width must be >=2 for bit masking (got {w})"
        ));
    }

    let replacement_dep = if let Some(unmasked) = try_unmask_dep(f, old_dep, w, &position) {
        unmasked
    } else {
        let lit = make_literal_mask_node(f, w, &position);
        make_and_node(f, w, old_dep, lit)
    };

    let new_payload = remap_payload_with(&old_payload, |(slot, dep)| {
        if slot == operand_slot {
            replacement_dep
        } else {
            dep
        }
    });
    f.get_node_mut(node_ref).payload = new_payload;
    Ok(())
}

fn find_candidates_for_mask_transform(f: &IrFn, max_candidates: usize) -> Vec<TransformLocation> {
    let mut out: Vec<TransformLocation> = Vec::new();
    for i in 0..f.nodes.len() {
        if out.len() >= max_candidates {
            break;
        }
        let node_ref = NodeRef { index: i };
        let pairs = operand_pairs(&f.get_node(node_ref).payload);
        for (slot, dep) in pairs {
            if out.len() >= max_candidates {
                break;
            }
            if matches!(bits_width(f, dep), Some(w) if w >= 2) {
                out.push(TransformLocation::RewireOperand {
                    node: node_ref,
                    operand_slot: slot,
                    new_operand: dep, // placeholder; ignored by this transform
                });
            }
        }
    }
    out
}

/// Non-always-equivalent transform that toggles masking of an operand's high
/// bit by inserting/removing `and(operand, mask_with_high_bit_cleared)`.
#[derive(Debug)]
pub struct MaskOperandHighBitTransform;

impl PirTransform for MaskOperandHighBitTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::MaskOperandHighBit
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        find_candidates_for_mask_transform(f, 2000)
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        apply_mask_transform(f, loc, MaskBitPosition::High, "MaskOperandHighBitTransform")
    }

    fn always_equivalent(&self) -> bool {
        false
    }
}

/// Non-always-equivalent transform that toggles masking of an operand's low
/// bit by inserting/removing `and(operand, mask_with_low_bit_cleared)`.
#[derive(Debug)]
pub struct MaskOperandLowBitTransform;

impl PirTransform for MaskOperandLowBitTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::MaskOperandLowBit
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        find_candidates_for_mask_transform(f, 2000)
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        apply_mask_transform(f, loc, MaskBitPosition::Low, "MaskOperandLowBitTransform")
    }

    fn always_equivalent(&self) -> bool {
        false
    }
}
