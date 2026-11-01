// SPDX-License-Identifier: Apache-2.0

use super::{MAX_GATES, MAX_LANES, MAX_OUTPUT_BITS};
use crate::aig::get_summary_stats::{AigConeStats, get_aig_cone_stats};
use crate::aig::topo::topo_sort_refs;
use crate::aig::{AigNode, AigOperand, AigRef, GateFn};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Label {
    X,
    Y,
    Constant,
}

pub(crate) struct Group {
    pub coords: Vec<AigOperand>,
    pub words: Vec<Word>,
}
pub(crate) struct Word {
    pub table: Vec<Label>,
    pub lanes: Vec<Lane>,
}
pub(crate) struct Lane {
    pub output: usize,
    pub bit: usize,
    pub selected: AigOperand,
    pub x: AigOperand,
    pub y: AigOperand,
    pub constant: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Form {
    Flat,
    Active,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Route {
    Direct,
    Shannon(usize),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Admission {
    SafeOnly,
    Diagnostic,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Verdict {
    Eligible,
    Area,
    Arrival,
    AreaAndArrival,
    Unchanged,
    Duplicate(usize),
}

#[derive(Debug)]
pub(crate) struct Report {
    pub route: Route,
    pub form: Form,
    pub keep_late: bool,
    pub verdict: Verdict,
    pub and_nodes: usize,
    pub depths: Vec<usize>,
    pub changed: Vec<usize>,
    pub late: Vec<usize>,
}
#[derive(Default, Debug)]
pub(crate) struct Stats {
    pub declined: Option<&'static str>,
    pub stock: Option<AigConeStats>,
    pub reports: Vec<Report>,
    pub scalar_queries: usize,
    pub reusable_hits: usize,
    pub cube_candidates: usize,
    pub constructed_ands: usize,
    pub construction_limited: bool,
    pub prices: usize,
    pub elapsed: std::time::Duration,
}
pub(crate) struct Candidate {
    pub graph: GateFn,
    pub report: usize,
}

/// Table rows use the signed coordinate values in LSB-first order. The caller
/// proves the original source selection, the payload bindings, and the complete
/// table. All other outputs and consumers retain their exact stock functions.
pub(crate) fn alternatives(
    stock: &GateFn,
    group: &Group,
    admission: Admission,
    stats: &mut Stats,
) -> Vec<Candidate> {
    *stats = Stats::default();
    let start = std::time::Instant::now();
    let result = generate(stock, group, admission, stats, 4096);
    stats.elapsed = start.elapsed();
    match result {
        Ok(values) => values,
        Err(reason) => {
            stats.declined = Some(reason);
            Vec::new()
        }
    }
}

fn generate(
    stock: &GateFn,
    group: &Group,
    admission: Admission,
    stats: &mut Stats,
    max_constructed_ands: usize,
) -> Result<Vec<Candidate>, &'static str> {
    if !(1..=6).contains(&group.coords.len()) || !(2..=4).contains(&group.words.len()) {
        return Err("coordinate/word bound");
    }
    let total_lanes: usize = group.words.iter().map(|w| w.lanes.len()).sum();
    let noutputs: usize = stock.outputs.iter().map(|o| o.get_bit_count()).sum();
    if !(2..=MAX_LANES).contains(&total_lanes)
        || noutputs > MAX_OUTPUT_BITS
        || stock.gates.len() > MAX_GATES
    {
        return Err("graph/slot work bound");
    }
    let roots: Vec<_> = stock
        .outputs
        .iter()
        .flat_map(|o| o.bit_vector.iter_lsb_to_msb().copied())
        .collect();
    let mut offsets = vec![0];
    for o in &stock.outputs {
        offsets.push(offsets.last().unwrap() + o.get_bit_count());
    }
    let mut used = HashSet::new();
    let mut arena = Arena::new(stock)?;
    let paid = |p: AigOperand| arena.paid.get(p.node.id).copied().unwrap_or(false);
    if group
        .coords
        .iter()
        .any(|&e| !paid(e) || !used.insert(e.node))
    {
        return Err("unbound or repeated coordinate");
    }
    let mut slots = HashSet::new();
    for w in &group.words {
        if w.lanes.is_empty() || w.table.len() != (1usize << group.coords.len()) {
            return Err("incomplete word/table");
        }
        for l in &w.lanes {
            let o = stock.outputs.get(l.output).ok_or("unknown output")?;
            if l.bit >= o.get_bit_count()
                || o.bit_vector.get_lsb(l.bit).node != l.selected.node
                || !paid(l.x)
                || !paid(l.y)
                || !slots.insert((l.output, l.bit))
            {
                return Err("unbound or repeated slot/data");
            }
        }
    }
    let base = get_aig_cone_stats(&stock.gates, &roots);
    stats.stock = Some(base.clone());
    let mut seen = HashMap::<Vec<AigOperand>, usize>::new();
    let mut results = Vec::new();
    for split in std::iter::once(None).chain((0..group.coords.len()).map(Some)) {
        let route = split.map_or(Route::Direct, Route::Shannon);
        let axes: Vec<_> = group
            .coords
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(i, p)| (Some(i) != split).then_some(p))
            .collect();
        let mut solver = Solver::new(&arena, &axes);
        for form in [Form::Flat, Form::Active] {
            let mut pins = roots.clone();
            for w in &group.words {
                let grants =
                    |solver: &mut Solver, arena: &mut Arena, high: bool, stats: &mut Stats| {
                        let mut masks = [0u64; 2];
                        for i in 0..(1usize << axes.len()) {
                            let row = if let Some(j) = split {
                                let low = (1 << j) - 1;
                                (i & low) | ((i & !low) << 1) | (usize::from(high) << j)
                            } else {
                                i
                            };
                            match w.table[row] {
                                Label::X => masks[0] |= 1 << i,
                                Label::Y => masks[1] |= 1 << i,
                                Label::Constant => {}
                            }
                        }
                        solver.grants(arena, masks, form, stats)
                    };
                let low = grants(&mut solver, &mut arena, false, stats);
                let high = split.map(|_| grants(&mut solver, &mut arena, true, stats));
                for l in &w.lanes {
                    let off = arena.routed(l, low);
                    let positive = if let (Some(j), Some(gr)) = (split, high) {
                        let on = arena.routed(l, gr);
                        arena.mux(group.coords[j], off, on)
                    } else {
                        off
                    };
                    let slot = offsets[l.output] + l.bit;
                    let old = roots[slot];
                    let replacement = phase(positive, old.negated != l.selected.negated);
                    // Tags observe the positive AND node, independent of the
                    // output phase. Keep untransferable roots when pricing.
                    if stock.gates[old.node.id]
                        .get_tags()
                        .is_some_and(|tags| !tags.is_empty())
                        && (old.negated != replacement.negated
                            || !matches!(arena.g.gates[replacement.node.id], AigNode::And2 { .. }))
                    {
                        continue;
                    }
                    pins[slot] = replacement;
                }
            }
            stats.constructed_ands = arena.g.gates.len() - arena.old_len;
            if stats.constructed_ands > max_constructed_ands {
                // Completed candidates own their graphs. Discard only the
                // unfinished form and its speculative arena when the next
                // construction exceeds the cap.
                stats.construction_limited = true;
                return Ok(results);
            }
            let mut keep_late = false;
            loop {
                let previous = seen.get(&pins).copied();
                let price = if let Some(i) = previous {
                    let report = &stats.reports[i];
                    AigConeStats {
                        and_nodes: report.and_nodes,
                        root_depths: report.depths.clone(),
                    }
                } else if pins == roots {
                    base.clone()
                } else {
                    stats.prices += 1;
                    get_aig_cone_stats(&arena.g.gates, &pins)
                };
                let changed: Vec<_> = (0..pins.len()).filter(|&i| pins[i] != roots[i]).collect();
                let late: Vec<_> = (0..pins.len())
                    .filter(|&i| price.root_depths[i] > base.root_depths[i])
                    .collect();
                let verdict = if changed.is_empty() {
                    Verdict::Unchanged
                } else if let Some(index) = previous {
                    Verdict::Duplicate(index)
                } else {
                    match (price.and_nodes < base.and_nodes, late.is_empty()) {
                        (true, true) => Verdict::Eligible,
                        (true, false) => Verdict::Arrival,
                        (false, true) => Verdict::Area,
                        (false, false) => Verdict::AreaAndArrival,
                    }
                };
                let index = stats.reports.len();
                seen.entry(pins.clone()).or_insert(index);
                if verdict == Verdict::Eligible
                    || (admission == Admission::Diagnostic
                        && previous.is_none()
                        && !changed.is_empty())
                {
                    results.push(Candidate {
                        graph: arena.finish(&pins),
                        report: index,
                    });
                }
                let retry = !keep_late && !late.is_empty();
                if retry {
                    for &slot in &late {
                        pins[slot] = roots[slot];
                    }
                }
                stats.reports.push(Report {
                    route,
                    form,
                    keep_late,
                    verdict,
                    and_nodes: price.and_nodes,
                    depths: price.root_depths,
                    changed,
                    late,
                });
                if !retry {
                    break;
                }
                keep_late = true;
            }
        }
    }
    Ok(results)
}

fn phase(p: AigOperand, invert: bool) -> AigOperand {
    if invert { p.negate() } else { p }
}
fn full(n: usize) -> u64 {
    u64::MAX >> (64 - (1 << n))
}
fn axis(n: usize, j: usize) -> u64 {
    (0..(1usize << n)).fold(0, |v, i| v | (((i >> j & 1) as u64) << i))
}

struct Arena {
    g: GateFn,
    old_len: usize,
    order: Vec<AigRef>,
    paid: Vec<bool>,
    depth: Vec<usize>,
    zero: AigOperand,
    ands: HashMap<[AigOperand; 2], AigOperand>,
}
impl Arena {
    fn new(g: &GateFn) -> Result<Self, &'static str> {
        let mut paid: Vec<_> = g
            .gates
            .iter()
            .map(|gate| matches!(gate, AigNode::Input { .. } | AigNode::Literal { .. }))
            .collect();
        for p in g.post_order_operands(false) {
            paid[p.node.id] = true;
        }
        let order = topo_sort_refs(&g.gates);
        let mut depth = vec![0; g.gates.len()];
        let mut ands = HashMap::new();
        let mut zero = None;
        for &r in &order {
            match g.gates[r.id] {
                AigNode::Literal { value, .. } => {
                    zero.get_or_insert(phase(r.into(), value));
                }
                AigNode::And2 { a, b, .. } => {
                    let mut k = [a, b];
                    k.sort_unstable();
                    if paid[r.id] {
                        ands.entry(k).or_insert(r.into());
                    }
                    depth[r.id] = 1 + depth[a.node.id].max(depth[b.node.id]);
                }
                _ => {}
            }
        }
        Ok(Self {
            g: g.clone(),
            old_len: g.gates.len(),
            order,
            paid,
            depth,
            zero: zero.ok_or("missing literal")?,
            ands,
        })
    }
    fn and(&mut self, a: AigOperand, b: AigOperand) -> AigOperand {
        for (p, other) in [(a, b), (b, a)] {
            if let AigNode::Literal { value, .. } = self.g.gates[p.node.id] {
                return if value ^ p.negated { other } else { self.zero };
            }
        }
        if a == b {
            return a;
        }
        if a == b.negate() {
            return self.zero;
        }
        let mut key = [a, b];
        key.sort_unstable();
        if let Some(p) = self.ands.get(&key) {
            return *p;
        }
        let p: AigOperand = AigRef {
            id: self.g.gates.len(),
        }
        .into();
        self.g.gates.push(AigNode::And2 {
            a: key[0],
            b: key[1],
            tags: None,
            pir_node_ids: Default::default(),
        });
        self.depth
            .push(1 + self.depth[a.node.id].max(self.depth[b.node.id]));
        self.ands.insert(key, p);
        p
    }
    fn or(&mut self, a: AigOperand, b: AigOperand) -> AigOperand {
        self.and(a.negate(), b.negate()).negate()
    }
    fn mux(&mut self, s: AigOperand, off: AigOperand, on: AigOperand) -> AigOperand {
        if off == on {
            return off;
        }
        let a = self.and(s.negate(), off);
        let b = self.and(s, on);
        self.or(a, b)
    }
    fn routed(&mut self, l: &Lane, grants: [AigOperand; 2]) -> AigOperand {
        let a = self.and(grants[0], phase(l.x, l.constant));
        let b = self.and(grants[1], phase(l.y, l.constant));
        let v = self.or(a, b);
        phase(v, l.constant)
    }
    /// Materializes the retained cone without renumbering earlier groups'
    /// gates.
    fn finish(&self, pins: &[AigOperand]) -> GateFn {
        // Preserve the entire original ID space and prune only speculative
        // appended gates.
        let mut live = vec![false; self.g.gates.len() - self.old_len];
        for p in pins {
            if p.node.id >= self.old_len {
                live[p.node.id - self.old_len] = true;
            }
        }
        for i in (0..live.len()).rev() {
            if live[i] {
                for p in self.g.gates[i + self.old_len].operands() {
                    if p.node.id >= self.old_len {
                        live[p.node.id - self.old_len] = true;
                    }
                }
            }
        }
        let mut g = GateFn {
            name: self.g.name.clone(),
            inputs: self.g.inputs.clone(),
            outputs: self.g.outputs.clone(),
            gates: self.g.gates[..self.old_len].to_vec(),
        };
        let mut ids = vec![0; live.len()];
        let remap = |mut p: AigOperand, ids: &[usize]| {
            if p.node.id >= self.old_len {
                p.node.id = ids[p.node.id - self.old_len];
            }
            p
        };
        for (i, needed) in live.into_iter().enumerate() {
            if needed {
                let AigNode::And2 { a, b, .. } = self.g.gates[self.old_len + i] else {
                    unreachable!()
                };
                ids[i] = g.gates.len();
                g.gates.push(AigNode::And2 {
                    a: remap(a, &ids),
                    b: remap(b, &ids),
                    tags: None,
                    pir_node_ids: Default::default(),
                });
            }
        }
        let mut replacements = pins.iter();
        for o in &mut g.outputs {
            for i in 0..o.get_bit_count() {
                let p = remap(*replacements.next().unwrap(), &ids);
                o.bit_vector.set_lsb(i, p);
            }
        }
        g
    }
}

/// Transfers output observations and attributes shared gates once all source
/// groups have chosen their output cones.
pub(super) fn attribute_replacements(stock: &GateFn, graph: &mut GateFn) {
    let mut transferred_tags = HashSet::new();
    let mut propagate_ids = false;
    let stock_len = stock.gates.len();
    for (before, after) in stock.outputs.iter().zip(&graph.outputs) {
        for (old, new) in before
            .bit_vector
            .iter_lsb_to_msb()
            .zip(after.bit_vector.iter_lsb_to_msb())
        {
            if old.node == new.node {
                continue;
            }
            let origin = &stock.gates[old.node.id];
            let replacement = &mut graph.gates[new.node.id];
            if let Some(tags) = origin.get_tags()
                && !tags.is_empty()
                && transferred_tags.insert(old.node)
            {
                debug_assert_eq!(old.negated, new.negated);
                assert!(replacement.try_add_tags(tags));
            }
            let ids = origin.get_pir_node_ids();
            if !ids.is_empty() {
                replacement.try_add_pir_node_ids(ids);
                propagate_ids |= new.node.id >= stock_len;
            }
        }
    }
    if propagate_ids {
        // Every group appends after the original incumbent in topological
        // order. Visit users first so a reused grant gets every lane's IDs.
        for i in (stock_len..graph.gates.len()).rev() {
            let (earlier, rest) = graph.gates.split_at_mut(i);
            let gate = &rest[0];
            for arg in gate.args().filter(|arg| arg.id >= stock_len) {
                earlier[arg.id].try_add_pir_node_ids(gate.get_pir_node_ids());
            }
        }
    }
}

struct Solver {
    coords: Vec<AigOperand>,
    mask: u64,
    axes: Vec<u64>,
    wires: BTreeMap<u64, AigOperand>,
    cubes: Vec<(u64, u8, u8)>,
    memo: HashMap<(u64, u64, u8), AigOperand>,
}
impl Solver {
    fn new(a: &Arena, coords: &[AigOperand]) -> Self {
        let n = coords.len();
        let mask = full(n);
        let axes: Vec<_> = (0..n).map(|i| axis(n, i)).collect();
        let mut values = vec![None; a.old_len];
        let mut wires = BTreeMap::new();
        for (j, p) in coords.iter().enumerate() {
            values[p.node.id] = Some(if p.negated { mask ^ axes[j] } else { axes[j] });
        }
        for &r in &a.order {
            if !a.paid[r.id] {
                continue;
            }
            let t = match values[r.id] {
                Some(t) => Some(t),
                None => match a.g.gates[r.id] {
                    AigNode::Literal { value, .. } => Some(if value { mask } else { 0 }),
                    AigNode::And2 {
                        a: left, b: right, ..
                    } => values[left.node.id]
                        .zip(values[right.node.id])
                        .map(|(x, y)| {
                            (if left.negated { mask ^ x } else { x })
                                & (if right.negated { mask ^ y } else { y })
                        }),
                    _ => None,
                },
            };
            if let Some(t) = t {
                values[r.id] = Some(t);
                for (v, p) in [
                    (t, AigOperand::from(r)),
                    (mask ^ t, AigOperand::from(r).negate()),
                ] {
                    let q = wires.entry(v).or_insert(p);
                    if (a.depth[p.node.id], p) < (a.depth[q.node.id], *q) {
                        *q = p;
                    }
                }
            }
        }
        let mut cubes = Vec::new();
        for i in 0..3usize.pow(n as u32) {
            let mut k = i;
            let mut on = mask;
            let (mut use_bits, mut neg) = (0, 0);
            for (j, t) in axes.iter().enumerate() {
                match k % 3 {
                    1 => {
                        on &= *t;
                        use_bits |= 1 << j
                    }
                    2 => {
                        on &= mask ^ t;
                        use_bits |= 1 << j;
                        neg |= 1 << j
                    }
                    _ => {}
                }
                k /= 3;
            }
            cubes.push((on, use_bits, neg));
        }
        Self {
            coords: coords.to_vec(),
            mask,
            axes,
            wires,
            cubes,
            memo: HashMap::new(),
        }
    }
    fn grants(
        &mut self,
        a: &mut Arena,
        [x, y]: [u64; 2],
        f: Form,
        stats: &mut Stats,
    ) -> [AigOperand; 2] {
        let dims = (1u8 << self.coords.len()) - 1;
        if f == Form::Flat {
            [
                self.boolean(a, x, self.mask, dims, stats),
                self.boolean(a, y, self.mask, dims, stats),
            ]
        } else {
            let active = x | y;
            let valid = self.boolean(a, active, self.mask, dims, stats);
            let select_x = self.boolean(a, x, active, dims, stats);
            [a.and(valid, select_x), a.and(valid, select_x.negate())]
        }
    }
    fn boolean(
        &mut self,
        a: &mut Arena,
        on: u64,
        care: u64,
        dims: u8,
        stats: &mut Stats,
    ) -> AigOperand {
        stats.scalar_queries += 1;
        if on == 0 {
            return a.zero;
        }
        if on == care {
            return a.zero.negate();
        }
        let key = (on, care, dims);
        if let Some(&p) = self.memo.get(&key) {
            return p;
        }
        let found = if care == self.mask {
            self.wires.get(&on).copied()
        } else {
            self.wires
                .iter()
                .filter(|(t, _)| (**t ^ on) & care == 0)
                .min_by_key(|(_, p)| (a.depth[p.node.id], **p))
                .map(|(_, p)| *p)
        };
        if let Some(p) = found {
            stats.reusable_hits += 1;
            self.memo.insert(key, p);
            return p;
        }
        let mut best = None;
        for &(table, bits, neg) in &self.cubes {
            if bits & !dims != 0 {
                continue;
            }
            stats.cube_candidates += 1;
            for inverted in [false, true] {
                let t = if inverted { self.mask ^ table } else { table };
                if ((t ^ on) & care) == 0 {
                    let d = (0..self.coords.len())
                        .filter(|j| bits & (1 << j) != 0)
                        .map(|j| a.depth[self.coords[j].node.id])
                        .max()
                        .unwrap_or(0);
                    let rank = (bits.count_ones(), d, bits, neg, inverted);
                    if best.is_none_or(|q| rank < q) {
                        best = Some(rank);
                    }
                }
            }
        }
        let p = if let Some((_, _, bits, neg, inv)) = best {
            let mut terms: Vec<_> = (0..self.coords.len())
                .filter(|j| bits & (1 << j) != 0)
                .map(|j| phase(self.coords[j], neg & (1 << j) != 0))
                .collect();
            while terms.len() > 1 {
                terms.sort_by_key(|p| (a.depth[p.node.id], *p));
                let b = a.and(terms[0], terms[1]);
                terms.drain(..2);
                terms.push(b);
            }
            phase(terms.first().copied().unwrap_or(a.zero.negate()), inv)
        } else {
            let j = (0..self.coords.len())
                .filter(|j| dims & (1 << j) != 0)
                .max_by_key(|&j| (a.depth[self.coords[j].node.id], j))
                .unwrap();
            let stride = 1usize << j;
            let axis = self.axes[j];
            let mask = self.mask;
            let split = |table: u64, high: bool| {
                if high {
                    let h = table & axis;
                    h | (h >> stride)
                } else {
                    let l = table & (mask ^ axis);
                    l | (l << stride)
                }
            };
            let next = dims & !(1 << j);
            let lo = self.boolean(a, split(on, false), split(care, false), next, stats);
            let hi = self.boolean(a, split(on, true), split(care, true), next, stats);
            a.mux(self.coords[j], lo, hi)
        };
        self.memo.insert(key, p);
        p
    }
}

#[cfg(test)]
#[path = "trial/test_cases.rs"]
mod tests;
