// SPDX-License-Identifier: Apache-2.0

//! Prove returned bits of affine priority ordinals, retaining their ordinary
//! implementation through whole-function rewriting.

use super::GateEnv;
use crate::aig::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::aig::get_summary_stats::get_output_cone_stats;
use crate::gate_builder::GateBuilder;
use xlsynth_pir::IrBits;
use xlsynth_pir::ir::{Binop, Fn, NaryOp, NodePayload, NodeRef, Unop};

#[derive(Clone, Copy)]
pub(super) enum Order {
    TrailingZeros,
    LeadingZeros,
    HighestSetIndex,
}

pub(super) struct Ordinal {
    pub(super) input: NodeRef,
    pub(super) order: Order,
    pub(super) leading_user: Option<NodeRef>,
}

/// Recognizes an ordinal and its one-hot encoding when the consumer owns both.
pub(super) fn owned(f: &Fn, env: &GateEnv, encoded: NodeRef) -> Option<Ordinal> {
    if !env.has_single_use(encoded) {
        return None;
    }
    let (input, lsb_first) = match f.get_node(encoded).payload {
        NodePayload::ExtClz { arg, offset: 0, .. } => {
            return Some(Ordinal {
                input: arg,
                order: Order::LeadingZeros,
                leading_user: Some(encoded),
            });
        }
        NodePayload::ExtPrioEncode { arg, lsb_prio } => (arg, lsb_prio),
        NodePayload::Encode { arg: hot } if env.has_single_use(hot) => {
            let NodePayload::OneHot { arg, lsb_prio } = f.get_node(hot).payload else {
                return None;
            };
            (arg, lsb_prio)
        }
        _ => return None,
    };
    let (input, order, leading_user) = if !lsb_first {
        (input, Order::HighestSetIndex, None)
    } else if let NodePayload::Unop(Unop::Reverse, source) = f.get_node(input).payload {
        (
            source,
            Order::LeadingZeros,
            env.has_single_use(input).then_some(input),
        )
    } else {
        (input, Order::TrailingZeros, None)
    };
    Some(Ordinal {
        input,
        order,
        leading_user,
    })
}

// Coordinates survive native cut rewriting, unlike a mapped gate ID.
enum Pin {
    Constant(bool),
    Input {
        name: String,
        bit: usize,
        inverted: bool,
    },
}

impl Pin {
    fn from_operand(graph: &GateFn, operand: AigOperand) -> Option<Self> {
        match &graph.gates[operand.node.id] {
            AigNode::Literal { value, .. } => Some(Self::Constant(value ^ operand.negated)),
            AigNode::Input {
                name, lsb_index, ..
            } => Some(Self::Input {
                name: name.clone(),
                bit: *lsb_index,
                inverted: operand.negated,
            }),
            AigNode::And2 { .. } => None,
        }
    }

    fn resolve(&self, graph: &GateFn) -> Option<AigOperand> {
        match self {
            Self::Constant(value) => Some(AigOperand {
                node: AigRef { id: 0 },
                negated: *value,
            }),
            Self::Input {
                name,
                bit,
                inverted,
            } => {
                let input = graph
                    .inputs
                    .iter()
                    .find(|p| p.name == *name && *bit < p.get_bit_count())?;
                let operand = *input.bit_vector.get_lsb(*bit);
                Some(if *inverted { operand.negate() } else { operand })
            }
        }
    }
}

pub(crate) struct ReturnBit {
    bit: usize,
    width: usize,
    inverted: bool,
    lhs: Pin,
    mask: Option<(NaryOp, Pin)>,
}

pub(super) fn capture(f: &Fn, env: &GateEnv, graph: &GateFn, plans: &mut Vec<ReturnBit>) {
    let (Some(root), [output]) = (f.ret_node_ref, graph.outputs.as_slice()) else {
        return;
    };
    // Walk only return wiring. An equivalent internal bit cannot authorize
    // replacing a computed value at an unrelated consumer.
    let mut work = vec![(root, 0, false)];
    while let Some((node, bit, inverted)) = work.pop() {
        match &f.get_node(node).payload {
            NodePayload::Unop(Unop::Not, arg) => work.push((*arg, bit, !inverted)),
            NodePayload::Tuple(parts) | NodePayload::Nary(NaryOp::Concat, parts) => {
                let mut offset = bit;
                for part in parts.iter().rev() {
                    work.push((*part, offset, inverted));
                    offset += f.get_node_ty(*part).bit_count();
                }
            }
            payload @ NodePayload::BitSlice { width: 1, .. } => {
                if let Some((lhs, mask)) = return_expression(f, payload, env, graph) {
                    if env.get_bit_vector(node).ok().is_some_and(|v| {
                        let mapped = *v.get_lsb(0);
                        let signed = if inverted { mapped.negate() } else { mapped };
                        signed == *output.bit_vector.get_lsb(bit)
                    }) {
                        plans.push(ReturnBit {
                            bit,
                            width: output.get_bit_count(),
                            inverted,
                            lhs,
                            mask,
                        });
                    }
                }
            }
            _ => {}
        }
    }
}

pub(crate) fn apply(stock: GateFn, plans: &[ReturnBit]) -> GateFn {
    let Some(first) = plans.first() else {
        return stock;
    };
    let [output] = stock.outputs.as_slice() else {
        return stock;
    };
    if output.name != "output_value" || output.get_bit_count() != first.width {
        return stock;
    }
    let mut trial = GateBuilder::append_to(stock.clone());
    let mut changed = false;
    for plan in plans {
        let Some(lhs) = plan.lhs.resolve(&stock) else {
            return stock;
        };
        let replacement = match &plan.mask {
            None => lhs,
            Some((op, rhs)) => {
                let Some(rhs) = rhs.resolve(&stock) else {
                    return stock;
                };
                match op {
                    NaryOp::And => trial.add_and_binary(lhs, rhs),
                    NaryOp::Nand => trial.add_and_binary(lhs, rhs).negate(),
                    NaryOp::Or => trial.add_or_binary(lhs, rhs),
                    NaryOp::Nor => trial.add_or_binary(lhs, rhs).negate(),
                    NaryOp::Xor => trial.add_xor_binary(lhs, rhs),
                    NaryOp::Concat => unreachable!(),
                }
            }
        };
        let replacement = if plan.inverted {
            replacement.negate()
        } else {
            replacement
        };
        let old = *output.bit_vector.get_lsb(plan.bit);
        if replacement != old {
            trial.add_pir_node_ids(
                replacement.node,
                stock.gates[old.node.id].get_pir_node_ids(),
            );
            trial.outputs[0].bit_vector.set_lsb(plan.bit, replacement);
            changed = true;
        }
    }
    if !changed {
        return stock;
    }
    let trial = trial.build();
    // Price the union of live outputs. An otherwise cheap wire can retain a
    // shared cone, and an output with slack can still have an external
    // deadline.
    let before = get_output_cone_stats(&stock);
    let after = get_output_cone_stats(&trial);
    if after.and_nodes < before.and_nodes
        && after
            .root_depths
            .iter()
            .zip(&before.root_depths)
            .all(|(new, old)| new <= old)
    {
        log::debug!("priority ordinal substituted {} returned bits", plans.len());
        crate::aig::dce::dce(&trial)
    } else {
        stock
    }
}

fn return_expression(
    f: &Fn,
    node: &NodePayload,
    env: &GateEnv,
    graph: &GateFn,
) -> Option<(Pin, Option<(NaryOp, Pin)>)> {
    let NodePayload::BitSlice {
        arg,
        start,
        width: 1,
    } = *node
    else {
        return None;
    };
    // Reusing a bit at one sink cannot remove a word retained by another sink.
    if !env.has_single_use(arg) {
        return None;
    }
    if let Some(wire) = affine_wire(f, env, graph, arg, start) {
        return Some((Pin::from_operand(graph, wire)?, None));
    }
    let NodePayload::Nary(op, args) = &f.get_node(arg).payload else {
        return None;
    };
    let [lhs, rhs] = args.as_slice() else {
        return None;
    };
    if *op == NaryOp::Concat {
        return None;
    }
    let (wire, other) = affine_wire(f, env, graph, *lhs, start)
        .map(|wire| (wire, *rhs))
        .or_else(|| affine_wire(f, env, graph, *rhs, start).map(|wire| (wire, *lhs)))?;
    let other = *env.get_bit_vector(other).ok()?.get_lsb(start);
    Some((
        Pin::from_operand(graph, wire)?,
        Some((*op, Pin::from_operand(graph, other)?)),
    ))
}

fn affine_wire(
    f: &Fn,
    env: &GateEnv,
    graph: &GateFn,
    value: NodeRef,
    bit: usize,
) -> Option<AigOperand> {
    if !env.has_single_use(value) {
        return None;
    }
    let (literal, mut ordinal, subtract, signed_literal, signed_ordinal) =
        match &f.get_node(value).payload {
            NodePayload::ExtNaryAdd { terms, arch: None } => {
                let [a, b] = terms.as_slice() else {
                    return None;
                };
                let (k, o) = if matches!(f.get_node(a.operand).payload, NodePayload::Literal(_)) {
                    (a, b)
                } else {
                    (b, a)
                };
                if k.negated {
                    return None;
                }
                (k.operand, o.operand, o.negated, k.signed, o.signed)
            }
            NodePayload::Binop(Binop::Sub, a, b) => (*a, *b, true, false, false),
            NodePayload::Binop(Binop::Add, a, b) => {
                if matches!(f.get_node(*a).payload, NodePayload::Literal(_)) {
                    (*a, *b, false, false, false)
                } else {
                    (*b, *a, false, false, false)
                }
            }
            _ => return None,
        };
    let NodePayload::Literal(k) = &f.get_node(literal).payload else {
        return None;
    };
    let operand_width = f.get_node_ty(ordinal).bit_count();
    loop {
        if !env.has_single_use(ordinal) {
            return None;
        }
        match &f.get_node(ordinal).payload {
            NodePayload::ZeroExt { arg, .. } => ordinal = *arg,
            NodePayload::Nary(NaryOp::Concat, parts) => {
                let (low, high) = parts.split_last()?;
                let zero_padding = high.iter().all(|r| {
                    matches!(&f.get_node(*r).payload, NodePayload::Literal(v)
                        if v.to_bits().is_ok_and(|bits| bits.is_zero()))
                });
                if !zero_padding {
                    return None;
                }
                ordinal = *low;
            }
            _ => break,
        }
    }
    let source = owned(f, env, ordinal)?;
    let input = source.input;
    let width = f.get_node_ty(input).bit_count();
    if width == 0 {
        return None;
    }
    let required = xlsynth_pir::math::ceil_log2(width.checked_add(1)?);
    // The all-zero input has ordinal width. A wrapped sentinel or signed
    // extension into the observed bit changes its arithmetic preimage.
    if f.get_node_ty(ordinal).bit_count() < required
        || (signed_ordinal && operand_width <= required && bit >= operand_width)
    {
        return None;
    }
    let (first_index, first_wire) = match source.order {
        Order::TrailingZeros => (0, 0),
        Order::LeadingZeros => (0, width - 1),
        Order::HighestSetIndex => (width - 1, width - 1),
    };
    let literal = k.to_bits().ok()?;
    let inverted = singleton_preimage(&literal, signed_literal, bit, subtract, width, first_index)?;
    let wire = *env.get_bit_vector(input).ok()?.get_lsb(first_wire);
    // A computed bit has its own cone. Bypassing the ordinal may expose that
    // cone even when the encoded arithmetic can simplify its predicate.
    if matches!(graph.gates[wire.node.id], AigNode::And2 { .. }) {
        return None;
    }
    Some(if inverted { wire.negate() } else { wire })
}

/// Checks whether bit b of K ± ordinal equals (ordinal == winner) or its
/// complement on the entire inclusive domain [0, width], including the
/// sentinel.
fn singleton_preimage(
    k: &IrBits,
    signed: bool,
    bit: usize,
    subtract: bool,
    width: usize,
    winner: usize,
) -> Option<bool> {
    let kw = k.get_bit_count();
    let ext = signed && kw > 0 && k.get_bit(kw - 1).ok()?;
    let kbit = |i| if i < kw { k.get_bit(i).ok() } else { Some(ext) };
    // The first flip is one plus the low b bits of K (subtract) or !K (add).
    // Subsequent flips are 2^b apart. Flips beyond the source width are
    // immaterial.
    let mut first = 1usize;
    for i in 0..bit {
        if kbit(i)? == subtract {
            first = first.checked_add(1usize.checked_shl(u32::try_from(i).ok()?)?)?;
            if first > width {
                return None;
            }
        }
    }
    if first != if winner == 0 { 1 } else { winner } {
        return None;
    }
    let stride = u32::try_from(bit).ok().and_then(|b| 1usize.checked_shl(b));
    let second = stride.and_then(|s| first.checked_add(s));
    if winner == 0 {
        if second.is_some_and(|v| v <= width) {
            return None;
        }
    } else if second != winner.checked_add(1)
        || stride
            .and_then(|s| second?.checked_add(s))
            .is_some_and(|v| v <= width)
    {
        return None;
    }
    Some(kbit(bit)? == (winner != 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_matches_direct_word_arithmetic() {
        for constant in 0..32 {
            let k = IrBits::make_ubits(5, constant).unwrap();
            for signed in [false, true] {
                let value = if signed && constant >= 16 {
                    i128::from(constant) - 32
                } else {
                    i128::from(constant)
                };
                for subtract in [false, true] {
                    for bit in [0, 1, 2, 3, 4, 5, 8, 63, 65] {
                        for width in 1..=40 {
                            for winner in [0, width - 1] {
                                let expected = [false, true].into_iter().find(|inverted| {
                                    (0..=width).all(|ordinal| {
                                        let offset = ordinal as i128;
                                        let result = if subtract {
                                            value - offset
                                        } else {
                                            value + offset
                                        };
                                        ((result >> bit) & 1 != 0)
                                            == ((ordinal == winner) ^ inverted)
                                    })
                                });
                                assert_eq!(
                                    singleton_preimage(&k, signed, bit, subtract, width, winner),
                                    expected,
                                    "K={value}, subtract={subtract}, bit={bit}, width={width}, winner={winner}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
