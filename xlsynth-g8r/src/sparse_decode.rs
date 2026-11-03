// SPDX-License-Identifier: Apache-2.0

//! Share address enables for equal output cofactors of a small decoded key.
use crate::aig::dce::dce;
use crate::aig::gate::{AigBitVector, AigNode, AigOperand, AigRef, GateFn};
use crate::aig::get_summary_stats::get_output_cone_stats;
use crate::aig::retained_rails::{CutBit, CutIndex, RailBudget};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use std::collections::BTreeMap;
use xlsynth_pir::ir::{Binop, Fn, NodePayload};
use xlsynth_pir::ir_utils::operands;

mod refactor;

/// Try banked output cofactors while preserving output timing and observable
/// signals.
pub(crate) fn rewrite(f: &Fn, old: &GateFn) -> Option<GateFn> {
    let inputs: usize = old.inputs.iter().map(|p| p.get_bit_count()).sum();
    let outputs: usize = old.outputs.iter().map(|p| p.get_bit_count()).sum();
    if !(3..=6).contains(&inputs)
        || !(2..=16).contains(&outputs)
        || old.gates.len() > 512
        || f.nodes.len() > 1024
    {
        return None;
    }
    let (offset, width) = decoded_key(f, old)?;
    let coords: Vec<_> = old
        .inputs
        .iter()
        .flat_map(|p| {
            (0..p.get_bit_count()).map(move |bit| CutBit {
                port: p.name.clone(),
                width: p.get_bit_count(),
                bit,
            })
        })
        .collect();
    // Each bank enable and each residual function fits in an 8-row truth table.
    let mut budget = RailBudget::default();
    let incumbent = CutIndex::build(old, &coords, &mut budget)?;
    let roots = |f: &GateFn| {
        f.outputs
            .iter()
            .flat_map(|p| p.bit_vector.iter_lsb_to_msb().copied())
            .collect::<Vec<_>>()
    };
    let old_roots = roots(old);
    let row_count = 1usize << inputs;
    let full = u64::MAX >> (64 - row_count);
    let mut covers = Vec::with_capacity(outputs);
    for &root in &old_roots {
        let rows = incumbent.signature(root)?.rows();
        let complement = rows ^ full;
        covers.push(OutputCover {
            rows,
            sparse: if complement.count_ones() < rows.count_ones() {
                complement
            } else {
                rows
            },
            factors: Vec::new(),
        });
    }
    let info = get_output_cone_stats(old);
    let eligible_area = |trial: &GateFn| {
        let stats = get_output_cone_stats(trial);
        (stats.and_nodes < info.and_nodes
            && stats
                .root_depths
                .iter()
                .zip(&info.root_depths)
                .all(|(a, b)| a <= b))
        .then_some(stats.and_nodes)
    };
    let mut best = None;
    let mut best_area = info.and_nodes;
    for bank_len in 1..=width.min(3) {
        if inputs - bank_len > 3 {
            continue;
        }
        let bank: Vec<_> = (offset + width - bank_len..offset + width).collect();
        let word: Vec<_> = (0..inputs).filter(|p| !bank.contains(p)).collect();
        let trial = build_banked(old, &covers, &bank, &word);
        if let Some(area) = eligible_area(&trial)
            && area < best_area
        {
            best_area = area;
            best = Some((trial, bank, word));
        }
    }
    let (mut trial, bank, word) = best?;
    let mut alternate = CutIndex::build(&trial, &coords, &mut budget)?;
    // A minimized bank can absorb a named equality into a larger enable.
    // Try one decomposition of the same partition with explicit address terms.
    let mut missing = old
        .gates
        .iter()
        .enumerate()
        .filter(|(_, node)| node.get_tags().is_some_and(|tags| !tags.is_empty()))
        // Only reachable nodes have a signature on the full input cut.
        .filter_map(|(id, _)| incumbent.signature(AigRef { id }.into()))
        .filter(|sig| {
            !alternate
                .resolve(sig)
                .is_some_and(|e| !e.negated && matches!(trial.get(e.node), AigNode::And2 { .. }))
        });
    let key_mask = ((1usize << width) - 1) << offset;
    let mut refactor_mapping = None;
    if let Some(first) = missing.next() {
        let placed = std::iter::once(first)
            .chain(missing)
            .all(|sig| place_address_term(&mut covers, sig.rows(), key_mask, row_count).is_some());
        let banked = placed.then(|| build_banked(old, &covers, &bank, &word));
        trial = banked.filter(|g| eligible_area(g).is_some()).or_else(|| {
            let rebuilt =
                refactor::reuse_key_products(old, &incumbent, offset, width, &mut budget)?;
            eligible_area(&rebuilt.graph)?;
            refactor_mapping = Some(rebuilt.mapped);
            Some(rebuilt.graph)
        })?;
        alternate = CutIndex::build(&trial, &coords, &mut budget)?;
    }
    // An automatic replacement must preserve the value and order of every
    // output. Seed each actual output even if resolve picks an equivalent copy.
    let mut old_output_roots = vec![false; old.gates.len()];
    for (a, b) in old_roots.into_iter().zip(roots(&trial)) {
        if incumbent.signature(a)?.rows() != alternate.signature(b)?.rows() {
            return None;
        }
        trial.gates[b.node.id].union_pir_node_ids_from_node(old.get(a.node));
        old_output_roots[a.node.id] = true;
    }
    // Source attribution on retained values is phase-independent. A named
    // signal tag, in contrast, denotes the positive node and can only move in
    // that phase.
    for (id, node) in old.gates.iter().enumerate() {
        let tags = node.get_tags().filter(|tags| !tags.is_empty());
        // An output's IDs already went to its actual implementation. A truth
        // lookup could also charge them to a separately built complement.
        let needs_ids = !old_output_roots[id] && !node.get_pir_node_ids().is_empty();
        if !needs_ids && tags.is_none() {
            continue;
        }
        let Some(sig) = incumbent.signature(AigRef { id }.into()) else {
            // The cut covers every input, so only dead gates lack a signature.
            continue;
        };
        let target = alternate.resolve(&sig);
        if let Some(tags) = tags {
            let positive = target.filter(|edge| !edge.negated)?;
            if !trial.gates[positive.node.id].try_add_tags(tags) {
                return None;
            }
        }
        if needs_ids {
            // A surviving implementation owns its IDs even when a shallower
            // rail has the same truth table. A bypassed copy has no signature.
            let mapped = refactor_mapping
                .as_ref()
                .and_then(|mapping| mapping[id])
                .filter(|&edge| alternate.signature(edge).is_some());
            if let Some(producer) = mapped.or(target) {
                trial.gates[producer.node.id].union_pir_node_ids_from_node(node);
            }
        }
    }
    // An exact producer owns its implementation up to the next recovered
    // subfunction. Shared new gates inherit all such enclosing producers.
    let retained: Vec<_> = trial
        .gates
        .iter()
        .map(|n| !n.get_pir_node_ids().is_empty())
        .collect();
    // Both builders emit each AND after its fanins, so all users have
    // contributed before their shared fanin is visited.
    for id in (0..trial.gates.len()).rev() {
        let (fanins, rest) = trial.gates.split_at_mut(id);
        let node = &rest[0];
        let ids = node.get_pir_node_ids();
        if !ids.is_empty() {
            for child in node.args() {
                if !retained[child.id] && matches!(fanins[child.id], AigNode::And2 { .. }) {
                    fanins[child.id].try_add_pir_node_ids(ids);
                }
            }
        }
    }
    // Drop bypassed ANDs after attribution has consumed the rebuild's node IDs.
    Some(if refactor_mapping.is_some() {
        dce(&trial)
    } else {
        trial
    })
}

/// Find the unique input compared with at least two distinct literals in the
/// return cone.
fn decoded_key(f: &Fn, old: &GateFn) -> Option<(usize, usize)> {
    let mut comparisons = Vec::new();
    let mut seen = vec![false; f.nodes.len()];
    let mut pending = vec![f.ret_node_ref?];
    while let Some(nr) = pending.pop() {
        if seen[nr.index] {
            continue;
        }
        seen[nr.index] = true;
        let payload = &f.get_node(nr).payload;
        pending.extend(operands(payload));
        if let NodePayload::Binop(Binop::Eq, left, right) = *payload {
            comparisons.push((left, right));
        }
    }
    let mut offset = 0;
    let mut address = None;
    for port in &old.inputs {
        let width = port.get_bit_count();
        if width > 1 {
            let param = f
                .params
                .iter()
                .find(|p| f.get_node(**p).param_name() == port.name)?;
            let mut first_literal = None;
            for &(left, right) in &comparisons {
                let literal = if left == *param {
                    right
                } else if right == *param {
                    left
                } else {
                    continue;
                };
                if let NodePayload::Literal(value) = &f.get_node(literal).payload
                    && *first_literal.get_or_insert(value) != value
                {
                    if address.replace((offset, width)).is_some() {
                        return None;
                    }
                    break;
                }
            }
        }
        offset += width;
    }
    address
}

/// The full output table and the disjoint address terms to expose in its sparse
/// phase.
struct OutputCover {
    rows: u64,
    sparse: u64,
    factors: Vec<u64>,
}

/// Expose one full-key minterm on a compatible output. Distinct minterms give
/// at most one extra term per address across the whole bank.
fn place_address_term(
    covers: &mut [OutputCover],
    rows: u64,
    key_mask: usize,
    row_count: usize,
) -> Option<()> {
    let address = rows.trailing_zeros() as usize & key_mask;
    // The equality must fix every key bit and be independent of all other
    // inputs.
    if !(0..row_count).all(|r| (rows >> r & 1 != 0) == (r & key_mask == address)) {
        return None;
    }
    // Choose the sparse output phase where this equality covers the largest
    // fraction of its asserted rows. Equalities for distinct keys are disjoint.
    let cover = covers
        .iter_mut()
        .filter(|c| rows & !c.sparse == 0)
        .min_by_key(|c| c.sparse.count_ones())?;
    if !cover.factors.contains(&rows) {
        cover.factors.push(rows);
    }
    Some(())
}

/// Build all output bits in port order, sharing the bank and word decodes.
fn build_banked(old: &GateFn, covers: &[OutputCover], bank: &[usize], word: &[usize]) -> GateFn {
    let (mut g, pins) = decoder_inputs(old);
    let mut cache = BTreeMap::new();
    let mut covers = covers.iter();
    for port in &old.outputs {
        let mut result = Vec::new();
        for cover in covers.by_ref().take(port.get_bit_count()) {
            let mut emit = |rows| banked_bit(&mut g, &pins, bank, word, rows, &mut cache);
            if cover.factors.is_empty() {
                result.push(emit(cover.rows));
                continue;
            }
            let mut residual = cover.sparse;
            let mut terms = Vec::new();
            for &rows in &cover.factors {
                terms.push(emit(rows));
                residual &= !rows;
            }
            terms.push(emit(residual));
            let value = combine(&mut g, terms, true);
            result.push(if cover.sparse == cover.rows {
                value
            } else {
                value.negate()
            });
        }
        g.add_output(
            port.name.clone(),
            AigBitVector::from_lsb_is_index_0(&result),
        );
    }
    g.build()
}
/// Copy the decoder interface, including provenance on unused bits and the
/// constant.
fn decoder_inputs(old: &GateFn) -> (GateBuilder, Vec<AigOperand>) {
    let mut g = GateBuilder::new(old.name.clone(), GateBuilderOptions::opt());
    // The native format exposes %0's provenance even outside the output cone.
    g.add_pir_node_ids(g.get_false().node, old.gates[0].get_pir_node_ids());
    let mut pins = Vec::new();
    for port in &old.inputs {
        let input = g.add_input(port.name.clone(), port.get_bit_count());
        // Even pins outside the output cone retain their interface provenance.
        for (prior, pin) in port
            .bit_vector
            .iter_lsb_to_msb()
            .zip(input.iter_lsb_to_msb())
        {
            g.add_pir_node_ids(pin.node, old.get(prior.node).get_pir_node_ids());
            pins.push(*pin);
        }
    }
    (g, pins)
}
type TableCache = BTreeMap<(Vec<usize>, u8), AigOperand>;
fn banked_bit(
    g: &mut GateBuilder,
    pins: &[AigOperand],
    bank: &[usize],
    word: &[usize],
    t: u64,
    cache: &mut TableCache,
) -> AigOperand {
    let w = 1usize << word.len();
    let full = ((1u16 << w) - 1) as u8;
    let rows: Vec<u8> = (0..1usize << bank.len())
        .map(|i| {
            (0..w).fold(0, |v, j| {
                let a = deposit(i, bank) | deposit(j, word);
                v | ((((t >> a) & 1) as u8) << j)
            })
        })
        .collect();
    let invert =
        rows.iter().filter(|&&r| r == full).count() > rows.iter().filter(|&&r| r == 0).count();
    let mut groups = BTreeMap::<u8, u8>::new();
    for (i, row) in rows.into_iter().enumerate() {
        let r = if invert { row ^ full } else { row };
        if r != 0 {
            *groups.entry(r).or_default() |= 1 << i;
        }
    }
    let mut terms = Vec::new();
    for (r, mask) in groups {
        let enable = small_function(g, pins, bank, mask, cache);
        let val = small_function(g, pins, word, r, cache);
        terms.push(g.add_and_binary(enable, val));
    }
    let y = combine(g, terms, true);
    if invert { y.negate() } else { y }
}
fn deposit(x: usize, pins: &[usize]) -> usize {
    pins.iter()
        .enumerate()
        .fold(0, |v, (i, p)| v | (((x >> i) & 1) << p))
}
fn combine(g: &mut GateBuilder, mut v: Vec<AigOperand>, or: bool) -> AigOperand {
    if v.is_empty() {
        return if or { g.get_false() } else { g.get_true() };
    }
    v.sort_by_key(|x| (x.node.id, x.negated));
    v.dedup();
    while v.len() > 1 {
        v.sort_by_key(|x| (g.aig_depth(*x), x.node.id, x.negated));
        let b = v.remove(1);
        let a = v.remove(0);
        let t = if or {
            g.add_or_binary(a, b)
        } else {
            g.add_and_binary(a, b)
        };
        v.push(t);
    }
    v[0]
}
type Cube = (u8, u8);
fn minimum_cube_cover(p: usize, table: u8) -> Vec<Cube> {
    let full = (1usize << p) - 1;
    // At three inputs there are 27 cubes and at most 256 uncovered-row sets.
    let mut cubes = Vec::new();
    for mask in 0..=full {
        for pattern in 0..=full {
            if pattern & !mask != 0 {
                continue;
            }
            let truth = (0..=full).fold(0u8, |u, i| u | (u8::from(i & mask == pattern) << i));
            if truth & !table == 0 {
                cubes.push((mask as u8, pattern as u8, truth));
            }
        }
    }
    let primes: Vec<_> = cubes
        .iter()
        .filter(|a| !cubes.iter().any(|b| a.2 != b.2 && a.2 & !b.2 == 0))
        .collect();
    let mut solutions = vec![None; usize::from(table) + 1];
    solutions[0] = Some(Vec::<Cube>::new());
    for left in 1..=table {
        if left & !table != 0 {
            continue;
        }
        let bit = 1 << left.trailing_zeros();
        for &&(mask, value, t) in &primes {
            if bit & t == 0 {
                continue;
            }
            let mut item = solutions[(left & !t) as usize].as_ref().unwrap().clone();
            item.push((mask, value));
            item.sort_unstable();
            let key = |a: &Vec<Cube>| (a.len(), a.iter().map(|x| x.0.count_ones()).sum::<u32>());
            let dst = &mut solutions[left as usize];
            if dst
                .as_ref()
                .is_none_or(|v| (key(&item), &item) < (key(v), v))
            {
                *dst = Some(item);
            }
        }
    }
    solutions[usize::from(table)].take().unwrap()
}
fn small_function(
    g: &mut GateBuilder,
    pins: &[AigOperand],
    coords: &[usize],
    t: u8,
    cache: &mut TableCache,
) -> AigOperand {
    let key = (coords.to_vec(), t);
    if let Some(&x) = cache.get(&key) {
        return x;
    }
    let full = ((1u16 << (1usize << coords.len())) - 1) as u8;
    if t == 0 {
        return g.get_false();
    }
    if t == full {
        return g.get_true();
    }
    let yes = minimum_cube_cover(coords.len(), t);
    let no = minimum_cube_cover(coords.len(), t ^ full);
    let cost = |v: &Vec<Cube>| (v.iter().map(|c| c.0.count_ones()).sum::<u32>(), v.len());
    let inv = cost(&no) < cost(&yes);
    let mut cs = Vec::new();
    for (m, v) in if inv { no } else { yes } {
        let terms = coords
            .iter()
            .enumerate()
            .filter(|(i, _)| m & (1 << i) != 0)
            .map(|(i, &p)| {
                if v & (1 << i) == 0 {
                    pins[p].negate()
                } else {
                    pins[p]
                }
            })
            .collect();
        cs.push(combine(g, terms, false));
    }
    let o = combine(g, cs, true);
    let o = if inv { o.negate() } else { o };
    cache.insert(key, o);
    o
}

#[cfg(test)]
mod tests;
