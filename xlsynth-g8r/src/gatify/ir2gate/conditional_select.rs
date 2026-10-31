// SPDX-License-Identifier: Apache-2.0

//! Prove that a specialized select arm duplicates its general arm on a small
//! finite domain. The proof evaluates the already lowered AIG under all
//! assignments admitted by the guard. Unknown gates keep their identity.

use super::{GateEnv, GateOrVec, UseCounts};
use crate::aig::aig_simplify::or_contains_operand;
use crate::aig::gate::{AigBitVector, AigNode, AigOperand, AigRef};
use crate::gate_builder::GateBuilder;
use std::collections::HashMap;
use xlsynth_pir::IrBits;
use xlsynth_pir::ir::{Binop, Fn, NodePayload, NodeRef};

/// Eight cofactors cover a guard leaving up to three bits of its word free.
/// A query can inspect at most this many nodes including failed proofs.
const MAX_COFACTORS: usize = 8;
const MAX_VISITS: usize = 4096;

/// Reuses the general arm when both mapped arms agree under the slice equality.
pub(super) fn equivalent_general_arm(
    f: &Fn,
    env: &mut GateEnv,
    gb: &mut GateBuilder,
    selector: NodeRef,
    selector_bits: &AigBitVector,
    arms: &[AigBitVector],
    has_default: bool,
) -> Option<AigBitVector> {
    // Region inputs are independent, even if the enclosing function gives
    // them a definition that would imply a range for another region input.
    if !gb.options.fold
        || has_default
        || arms.len() != 2
        || !matches!(env.use_counts, UseCounts::Function(_))
    {
        return None;
    }
    let NodePayload::Binop(op @ (Binop::Eq | Binop::Ne), lhs, rhs) = f.get_node(selector).payload
    else {
        return None;
    };
    // Let the ordinary binary select fold to the actual selected arm.
    if selector_bits.get_lsb(0).node == gb.get_false().node {
        return None;
    }
    let (arg, start, width, literal) =
        [(lhs, rhs), (rhs, lhs)]
            .into_iter()
            .find_map(|(slice, constant)| {
                let NodePayload::BitSlice { arg, start, width } = f.get_node(slice).payload else {
                    return None;
                };
                let NodePayload::Literal(ref value) = f.get_node(constant).payload else {
                    return None;
                };
                Some((arg, start, width, value))
            })?;
    let word_width = f.get_node(arg).ty.bit_count();
    let free_bits = word_width.checked_sub(width)?;
    if free_bits == 0 || free_bits > MAX_COFACTORS.ilog2() as usize || start > free_bits {
        return None;
    }
    // All queries in one lowering share a budget linear in the mapped graph.
    let mut budget = MAX_VISITS.min(
        gb.gates
            .len()
            .saturating_mul(MAX_COFACTORS)
            .saturating_sub(env.select_proof_visits),
    );
    let initial = budget;
    if word_width.max(arms[0].get_bit_count()) > budget {
        return None;
    }
    // Borrow the guard data. Each budgeted visit reads just its word bit and
    // corresponding literal bit, even when a wide guard conflicts early.
    let word = match env.ir_to_g8.get(&arg)? {
        GateOrVec::BitVector(word) => word,
        GateOrVec::Gate(bit) => &AigBitVector::from_bit(*bit),
    };
    let fixed = literal.as_bits().ok()?;
    let result = all_cofactors_agree(gb, word, start, fixed, free_bits, arms, &mut budget);
    let visited = initial - budget;
    env.select_proof_visits += visited;
    log::debug!(
        "conditional select: guard={} cofactors={} visits={} bypass={}",
        f.get_node(selector).text_id,
        1 << free_bits,
        visited,
        result == Some(true)
    );
    if result != Some(true) {
        return None;
    }
    let selector = *selector_bits.get_lsb(0);
    let (enable, general, specialized) = if op == Binop::Eq {
        (selector.negate(), &arms[0], &arms[1])
    } else {
        (selector, &arms[1], &arms[0])
    };
    let mut output = general.clone();
    for (i, &special_bit) in specialized.iter_lsb_to_msb().enumerate() {
        let general_bit = *general.get_lsb(i);
        // The ordinary mux mask absorbs enable & (enable | rest) to enable.
        // With a constant or selector on the other arm the mux is then a wire
        // or a constant. Keep that fold without building speculative gates.
        let bit = if (special_bit.node == gb.get_false().node || special_bit.node == enable.node)
            && or_contains_operand(gb, general_bit, enable)
        {
            gb.add_mux2(enable, gb.get_true(), special_bit)
        } else {
            general_bit
        };
        output.set_lsb(i, gb.union_current_pir_node_id_into_operand(bit));
    }
    Some(output)
}

fn all_cofactors_agree(
    gb: &GateBuilder,
    word: &AigBitVector,
    start: usize,
    fixed: &IrBits,
    free_bits: usize,
    arms: &[AigBitVector],
    budget: &mut usize,
) -> Option<bool> {
    let zero = gb.get_false();
    let width = fixed.get_bit_count();
    for assignment in 0..1usize << free_bits {
        let mut facts = HashMap::from([(zero.node, zero)]);
        let mut feasible = true;
        for (i, bit) in word.iter_lsb_to_msb().enumerate() {
            *budget = budget.checked_sub(1)?;
            let value = if i < start {
                (assignment >> i) & 1 != 0
            } else if i - start < width {
                fixed.get_bit(i - start).ok()?
            } else {
                (assignment >> (i - width)) & 1 != 0
            };
            let base = AigOperand {
                node: zero.node,
                negated: value ^ bit.negated,
            };
            if facts.insert(bit.node, base).is_some_and(|old| old != base) {
                feasible = false;
                break;
            }
        }
        if !feasible {
            continue;
        }
        let mut pending = Vec::new();
        for (a, b) in arms[0].iter_lsb_to_msb().zip(arms[1].iter_lsb_to_msb()) {
            *budget = budget.checked_sub(1)?;
            if a != b
                && cofactor(*a, gb, &mut facts, &mut pending, budget)?
                    != cofactor(*b, gb, &mut facts, &mut pending, budget)?
            {
                return Some(false);
            }
        }
    }
    Some(true)
}

fn cofactor(
    output: AigOperand,
    gb: &GateBuilder,
    facts: &mut HashMap<AigRef, AigOperand>,
    pending: &mut Vec<(AigRef, bool)>,
    budget: &mut usize,
) -> Option<AigOperand> {
    let known = |facts: &HashMap<AigRef, AigOperand>, x: AigOperand| {
        facts
            .get(&x.node)
            .map(|&v| if x.negated { v.negate() } else { v })
    };
    pending.push((output.node, false));
    while let Some((node, expanded)) = pending.pop() {
        if facts.contains_key(&node) {
            continue;
        }
        if !expanded {
            *budget = budget.checked_sub(1)?;
        }
        let mut value = AigOperand::from(node);
        if let AigNode::And2 { a, b, .. } = gb.gates[node.id] {
            if !expanded {
                pending.extend([(node, true), (b.node, false), (a.node, false)]);
                continue;
            }
            let x = known(facts, a)?;
            let y = known(facts, b)?;
            value = if gb.is_known_false(x) || gb.is_known_false(y) || x == y.negate() {
                gb.get_false()
            } else if gb.is_known_true(x) || x == y {
                y
            } else if gb.is_known_true(y) {
                x
            } else {
                value
            };
        }
        facts.insert(node, value);
    }
    known(facts, output)
}
