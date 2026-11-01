// SPDX-License-Identifier: Apache-2.0

//! Shorten an increment only where the actual enabled carry cannot reach a
//! high suffix of existing wires. Escaping enables and borrow remain live.

use super::{GateEnv, GatifyOptions, UseCounts};
use crate::aig::gate::{AigBitVector, AigNode};
use crate::gate_builder::GateBuilder;
use std::collections::HashSet;
use xlsynth_pir::ir;

pub(super) fn try_lower(
    f: &ir::Fn,
    env: &mut GateEnv,
    terms: &[ir::ExtNaryAddTerm],
    width: usize,
    options: &GatifyOptions,
    gb: &mut GateBuilder,
    id: usize,
) -> Option<AigBitVector> {
    if width < 2 || !(2..=3).contains(&terms.len()) {
        return None;
    }
    let mut base = None;
    let mut up = None;
    let mut down = None;
    for t in terms {
        let w = f.get_node_ty(t.operand).bit_count();
        if w == width && !t.negated && base.is_none() {
            base = Some(t.operand);
        } else if w == 1 {
            let bit = *env.get_bit_vector(t.operand).ok()?.get_lsb(0);
            let slot = if t.signed ^ t.negated {
                &mut down
            } else {
                &mut up
            };
            if slot.replace(bit).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }
    let (base, up) = (base?, up?);
    let word = env.get_bit_vector(base).ok()?;
    if gb.is_known_false(up)
        || gb.is_known_true(up)
        || down.is_some_and(|d| d == up || gb.is_known_true(d))
    {
        return None;
    }
    let down = down.filter(|d| !gb.is_known_false(*d));
    let bits = word.iter_lsb_to_msb().copied().collect::<Vec<_>>();
    // A computed high operand may be economical only inside the original
    // arithmetic. Forward a high region composed exclusively of existing input
    // pins and constants.
    let wire_floor = bits
        .iter()
        .rposition(|bit| matches!(gb.gates[bit.node.id], AigNode::And2 { .. }))
        .map_or(1, |i| i + 1);
    if wire_floor == width {
        log::debug!(target: "xlsynth_g8r::guarded_unit", "guarded-unit held node={id} width={width} reason=computed-high");
        return None;
    }
    let mut guards = vec![up];
    if let Some(d) = down {
        guards.push(d.negate());
    }
    let start = log::log_enabled!(target: "xlsynth_g8r::guarded_unit", log::Level::Debug)
        .then(std::time::Instant::now);

    // Reuse already available range/bit facts. A zero in any low bit breaks the
    // all-ones prefix. An independently exposed region input, or an undemanded
    // bit of a partially mapped word, need not satisfy the enclosing PIR's
    // range.
    let complete_word = matches!(env.use_counts, UseCounts::Function(_))
        && env
            .demanded_bits
            .as_ref()
            .and_then(|m| m.get(&base))
            .is_some_and(|mask| mask.iter().all(|b| *b));
    let info = options
        .range_info
        .as_ref()
        .filter(|_| complete_word)
        .and_then(|r| r.get(f.get_node(base).text_id));
    let mut best = width;
    if let Some(upper) = info.and_then(|v| v.unsigned_max.as_ref()) {
        // The least k with max < 2^k - 1 is its bit length, plus one if
        // all significant bits are ones. This also handles max = 0.
        let length = (0..upper.get_bit_count())
            .rfind(|&i| upper.get_bit(i) == Ok(true))
            .map_or(0, |i| i + 1);
        let all_ones = (0..length).all(|i| upper.get_bit(i) == Ok(true));
        best = best.min(length + usize::from(all_ones));
    }
    for (i, &bit) in bits[..best.saturating_sub(1)].iter().enumerate() {
        let known_zero = info
            .and_then(|v| v.known_bits.as_ref())
            .is_some_and(|k| k.mask().get_bit(i) == Ok(true) && k.value().get_bit(i) == Ok(false));
        if known_zero || gb.is_known_false(bit) || guards.contains(&bit.negate()) {
            best = i + 1;
            break;
        }
    }
    best = best.max(wire_floor);
    let cheap = best;
    // Distinct, unconstrained input pins admit every low-prefix assignment.
    // This common case needs no truth table, CNF, or graph traversal.
    if best == width
        && guards
            .iter()
            .all(|r| matches!(gb.gates[r.node.id], AigNode::Input { .. }))
    {
        let mut pins = HashSet::new();
        if bits[..width - 1].iter().all(|b| {
            matches!(gb.gates[b.node.id], AigNode::Input { .. })
                && pins.insert(b.node)
                && !guards.iter().any(|g| g.node == b.node)
        }) {
            log::trace!(target:"xlsynth_g8r::guarded_unit","guarded-unit independent node={id} width={width}");
            return None;
        }
    }
    let facts = &mut env.guarded_unit_facts;
    let old_visits = facts.visits;
    let old_queries = facts.queries;
    let (best, strategy, reason, retries) =
        facts.shortest_impossible_prefix(gb, &guards, &bits, wire_floor, best);
    log::debug!(target: "xlsynth_g8r::guarded_unit",
        "guarded-unit node={id} width={width} scalar_decrement={} wire_floor={wire_floor} cheap={cheap} cap={best} method={strategy} result={reason} retries={retries} visits={} queries={} proof_us={}",
        down.is_some(), facts.visits - old_visits, facts.queries - old_queries,
        start.map_or(0, |t| t.elapsed().as_micros()));
    if best == width {
        return None;
    }
    let before = gb.gates.len();
    let Some(down) = down else {
        let low = super::gatify_ext_nary_add(f, env, id, terms, None, best, options, gb);
        let result = low
            .iter_lsb_to_msb()
            .copied()
            .chain(bits[best..].iter().copied())
            .collect::<Vec<_>>();
        log::debug!(target:"xlsynth_g8r::guarded_unit","guarded-unit emitted node={id} new_ands={} cap={best} construction=ordinary-low",gb.gates.len()-before);
        return Some(AigBitVector::from_lsb_is_index_0(&result));
    };
    // A positive and negative unit cancel when both are asserted. Only an
    // ascending run of ones can propagate an increment, and only a descending
    // run of zeroes can propagate a decrement. Neither path changes its input
    // enables.
    let mut carry = gb.add_and_binary(up, down.negate());
    let mut borrow = gb.add_and_binary(down, up.negate());
    let mut out = Vec::with_capacity(width);
    for (i, &bit) in bits.iter().enumerate() {
        let toggle = if i == 0 {
            gb.add_xor_binary(up, down)
        } else if i >= best {
            borrow
        } else {
            gb.add_or_binary(carry, borrow)
        };
        out.push(gb.add_xor_binary(bit, toggle));
        if i + 1 < best {
            carry = gb.add_and_binary(carry, bit);
        }
        if i + 1 < width {
            borrow = gb.add_and_binary(borrow, bit.negate());
        }
    }
    log::debug!(target:"xlsynth_g8r::guarded_unit","guarded-unit emitted node={id} new_ands={} cap={best}",gb.gates.len()-before);
    Some(AigBitVector::from_lsb_is_index_0(&out))
}
