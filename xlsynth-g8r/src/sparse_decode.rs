// SPDX-License-Identifier: Apache-2.0

//! Share address enables for equal output cofactors of a small decoded key.
use crate::aig::gate::{AigBitVector, AigOperand, GateFn};
use crate::aig::get_summary_stats::get_output_cone_stats;
use crate::aig::retained_rails::{CutBit, CutIndex, RailBudget};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use std::collections::{BTreeMap, BTreeSet};
use xlsynth_pir::ir::{Binop, Fn, NodePayload};

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
    let info = get_output_cone_stats(old);
    let mut best = None;
    let mut best_area = info.and_nodes;
    for bank_len in 1..=width.min(3) {
        if inputs - bank_len > 3 {
            continue;
        }
        let bank: Vec<_> = (offset + width - bank_len..offset + width).collect();
        let word: Vec<_> = (0..inputs).filter(|p| !bank.contains(p)).collect();
        let trial = build_banked(old, &incumbent, &bank, &word)?;
        let stats = get_output_cone_stats(&trial);
        if stats.and_nodes < best_area
            && stats
                .root_depths
                .iter()
                .zip(&info.root_depths)
                .all(|(after, before)| after <= before)
        {
            best_area = stats.and_nodes;
            best = Some(trial);
        }
    }
    let mut trial = best?;
    let alternate = CutIndex::build(&trial, &coords, &mut budget)?;
    let roots = |f: &GateFn| {
        f.outputs
            .iter()
            .flat_map(|p| p.bit_vector.iter_lsb_to_msb().copied())
            .collect::<Vec<_>>()
    };
    let pairs = roots(old)
        .into_iter()
        .zip(roots(&trial))
        .collect::<Vec<_>>();
    // An automatic replacement must preserve the value and order of every
    // output.
    for &(a, b) in &pairs {
        if incumbent.signature(a)?.rows() != alternate.signature(b)?.rows() {
            return None;
        }
    }
    // Source attribution on retained values is phase-independent. A named
    // signal tag, in contrast, denotes the positive node and can only move in
    // that phase.
    for edge in old.post_order_operands(false) {
        let node = &old.gates[edge.node.id];
        if node.get_pir_node_ids().is_empty() && node.get_tags().is_none() {
            continue;
        }
        let positive = AigOperand {
            node: edge.node,
            negated: false,
        };
        let sig = incumbent.signature(positive)?;
        if let Some(target) = alternate.resolve(&sig) {
            let new = &mut trial.gates[target.node.id];
            new.union_pir_node_ids_from_node(node);
            if !target.negated {
                if let Some(tags) = node.get_tags() {
                    new.try_add_tags(tags);
                }
            }
        }
    }
    // Every new gate in a returned cone implements that source observation.
    // More precise origin sets on equal old subfunctions are retained above.
    for (prior, port) in pairs {
        let ids = old.gates[prior.node.id].get_pir_node_ids();
        if ids.is_empty() {
            continue;
        }
        let mut queue = vec![port.node];
        let mut seen = BTreeSet::new();
        while let Some(id) = queue.pop() {
            if seen.insert(id) {
                let nd = &mut trial.gates[id.id];
                if !matches!(nd, crate::aig::gate::AigNode::And2 { .. }) {
                    continue;
                }
                nd.try_add_pir_node_ids(ids);
                queue.extend(nd.args());
            }
        }
    }
    Some(trial)
}
fn decoded_key(f: &Fn, old: &GateFn) -> Option<(usize, usize)> {
    let mut offset = 0;
    let mut address = None;
    for port in &old.inputs {
        let width = port.get_bit_count();
        if width > 1 {
            let param = f
                .params
                .iter()
                .find(|p| f.get_node(**p).param_name() == port.name)?;
            let mut keys = BTreeSet::new();
            for n in &f.nodes {
                if let NodePayload::Binop(Binop::Eq, left, right) = n.payload {
                    let literal = if left == *param {
                        right
                    } else if right == *param {
                        left
                    } else {
                        continue;
                    };
                    if let NodePayload::Literal(v) = &f.get_node(literal).payload {
                        if let Ok(value) = v.to_u64() {
                            keys.insert(value);
                        }
                    }
                }
            }
            if keys.len() >= 2 {
                if address.is_some() {
                    return None;
                }
                address = Some((offset, width));
            }
        }
        offset += width;
    }
    address
}
fn build_banked(
    old: &GateFn,
    incumbent: &CutIndex,
    bank: &[usize],
    word: &[usize],
) -> Option<GateFn> {
    let mut g = GateBuilder::new(old.name.clone(), GateBuilderOptions::opt());
    let pins: Vec<_> = old
        .inputs
        .iter()
        .flat_map(|p| {
            g.add_input(p.name.clone(), p.get_bit_count())
                .iter_lsb_to_msb()
                .copied()
                .collect::<Vec<_>>()
        })
        .collect();
    let mut cache = BTreeMap::new();
    for port in &old.outputs {
        let mut result = Vec::new();
        for &root in port.bit_vector.iter_lsb_to_msb() {
            result.push(banked_bit(
                &mut g,
                &pins,
                bank,
                word,
                incumbent.signature(root)?.rows(),
                &mut cache,
            ));
        }
        g.add_output(
            port.name.clone(),
            AigBitVector::from_lsb_is_index_0(&result),
        );
    }
    Some(g.build())
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
