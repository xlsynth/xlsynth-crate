// SPDX-License-Identifier: Apache-2.0
//! Share complementary decisions between multiple observed data words.
mod domain;
pub(crate) mod trial;
use super::GateEnv;
use crate::aig::get_summary_stats::get_aig_cone_stats;
use crate::aig::{AigNode, AigOperand, AigRef, GateFn};
use crate::process_ir_path::WordChoice;
use domain::Domain;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use trial::{Admission, Group, Label, Lane, Route, Stats, Word};
use xlsynth_pir::ir::{self, NodePayload, NodeRef, Type};

const MAX_LANES: usize = 256;
const MAX_OUTPUT_BITS: usize = 4096;
const MAX_GATES: usize = 32768;

struct Budget(usize);
impl Default for Budget {
    fn default() -> Self {
        Self(32 * 1024)
    }
}
impl Budget {
    fn take(&mut self, n: usize) -> bool {
        if self.0 < n {
            return false;
        }
        self.0 -= n;
        true
    }
}
/// Source operands for one selected bit, before expanding its output fanout.
struct Binding {
    selected: AigOperand,
    x: AigOperand,
    y: AigOperand,
    constant: bool,
}

/// A select word with at least one potential output binding.
struct Observed {
    axes: Vec<AigOperand>,
    take_x: u64,
    take_y: u64,
    bindings: Vec<Binding>,
}

/// Words with the same data sources and their competing claims on output gates.
#[derive(Default)]
struct Siblings {
    words: Vec<Observed>,
    repeated: HashMap<AigRef, bool>,
}

pub(crate) struct CapturedGroup {
    source_ids: [usize; 2],
    group: Group,
}

pub(super) fn capture(f: &ir::Fn, env: &GateEnv, graph: &GateFn) -> Vec<CapturedGroup> {
    let output_bits: usize = graph.outputs.iter().map(|port| port.get_bit_count()).sum();
    if graph.gates.len() > MAX_GATES || output_bits > MAX_OUTPUT_BITS {
        return Vec::new();
    }
    let mut outputs: HashMap<AigRef, Vec<(usize, usize)>> = HashMap::new();
    for (i, port) in graph.outputs.iter().enumerate() {
        for (j, bit) in port.bit_vector.iter_lsb_to_msb().enumerate() {
            if !matches!(graph.gates[bit.node.id], AigNode::Literal { .. }) {
                outputs.entry(bit.node).or_default().push((i, j));
            }
        }
    }
    let Some(demands) = &env.demanded_bits else {
        return Vec::new();
    };
    let demanded_word = |n: NodeRef| Some((env.get_bit_vector(n).ok()?, demands.get(&n)?));
    let mut budget = Budget::default();
    let mut siblings: BTreeMap<(usize, usize), Siblings> = BTreeMap::new();
    for (i, node) in f.nodes.iter().enumerate() {
        if !matches!(node.ty, Type::Bits(w) if w>1)
            || !matches!(
                node.payload,
                NodePayload::Sel { .. } | NodePayload::PrioritySel { .. }
            )
        {
            continue;
        }
        let root = NodeRef { index: i };
        let Some((v, demanded)) = demanded_word(root) else {
            continue;
        };
        if !v.iter_lsb_to_msb().any(|e| outputs.contains_key(&e.node)) {
            continue;
        }
        let Some(mut d) = Domain::analyze(f, root, &mut budget) else {
            continue;
        };
        if d.x.index > d.y.index {
            std::mem::swap(&mut d.x, &mut d.y);
            std::mem::swap(&mut d.take_x, &mut d.take_y);
        }
        let Some(axes) = d
            .axes
            .iter()
            .map(|&(n, bit)| {
                let (v, mask) = demanded_word(n)?;
                (mask.get(bit) == Some(&true) && bit < v.get_bit_count()).then(|| *v.get_lsb(bit))
            })
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let (Some((xs, x_demand)), Some((ys, y_demand))) = (demanded_word(d.x), demanded_word(d.y))
        else {
            continue;
        };
        let pair = siblings.entry((d.x.index, d.y.index)).or_default();
        let mut bindings = Vec::new();
        let data_width = xs.get_bit_count().min(ys.get_bit_count());
        for (bit, &selected) in v.iter_lsb_to_msb().take(data_width).enumerate() {
            if !outputs.contains_key(&selected.node)
                || demanded.get(bit) != Some(&true)
                || x_demand.get(bit) != Some(&true)
                || y_demand.get(bit) != Some(&true)
            {
                continue;
            }
            // Multiple selected bits (even in the same word) can map to the
            // same gate. Keep one binding and mark that gate ambiguous.
            let repeated = pair
                .repeated
                .entry(selected.node)
                .and_modify(|repeated| *repeated = true)
                .or_insert(false);
            if !*repeated {
                bindings.push(Binding {
                    selected,
                    x: *xs.get_lsb(bit),
                    y: *ys.get_lsb(bit),
                    constant: d.constant[bit],
                });
            }
        }
        if !bindings.is_empty() {
            pair.words.push(Observed {
                axes,
                take_x: d.take_x,
                take_y: d.take_y,
                bindings,
            });
        }
    }
    siblings
        .into_iter()
        .filter_map(|((x, y), pair)| {
            Some(CapturedGroup {
                source_ids: [f.nodes[x].text_id, f.nodes[y].text_id],
                group: group(graph, pair, &outputs)?,
            })
        })
        .collect()
}

/// Expand only uniquely bound gates in groups that fit the trial's lane budget.
fn group(
    graph: &GateFn,
    pair: Siblings,
    outputs: &HashMap<AigRef, Vec<(usize, usize)>>,
) -> Option<Group> {
    let mut records = pair.words;
    records.retain_mut(|r| {
        r.bindings.retain(|b| !pair.repeated[&b.selected.node]);
        !r.bindings.is_empty()
    });
    let lane_count: usize = records
        .iter()
        .flat_map(|r| &r.bindings)
        .map(|b| outputs[&b.selected.node].len())
        .sum();
    if lane_count > MAX_LANES || !(2..=4).contains(&records.len()) {
        return None;
    }
    let mut ids = BTreeSet::new();
    for e in records.iter().flat_map(|r| &r.axes) {
        if !matches!(graph.gates[e.node.id], AigNode::Literal { .. }) {
            ids.insert(e.node);
        }
    }
    if ids.is_empty() || ids.len() > 6 {
        return None;
    }
    let coords: Vec<_> = ids
        .into_iter()
        .map(|node| AigOperand {
            node,
            negated: false,
        })
        .collect();
    let mut words = Vec::new();
    for r in records {
        let mut table = Vec::new();
        for row in 0..(1usize << coords.len()) {
            let mut idx = 0;
            for (i, e) in r.axes.iter().enumerate() {
                let val = match graph.gates[e.node.id] {
                    AigNode::Literal { value, .. } => value,
                    _ => row >> coords.iter().position(|v| v.node == e.node)? & 1 != 0,
                } ^ e.negated;
                idx |= usize::from(val) << i;
            }
            table.push(if r.take_x >> idx & 1 != 0 {
                Label::X
            } else if r.take_y >> idx & 1 != 0 {
                Label::Y
            } else {
                Label::Constant
            });
        }
        if ![Label::X, Label::Y, Label::Constant]
            .iter()
            .all(|l| table.contains(l))
        {
            return None;
        }
        let mut lanes = Vec::new();
        for b in r.bindings {
            for &(output, bit) in &outputs[&b.selected.node] {
                lanes.push(Lane {
                    output,
                    bit,
                    selected: b.selected,
                    x: b.x,
                    y: b.y,
                    constant: b.constant,
                });
            }
        }
        words.push(Word { table, lanes });
    }
    // A common data edge alone does not establish a shared decision between
    // receivers.
    if !has_shared_cofactor(&words, coords.len()) {
        return None;
    }
    Some(Group { coords, words })
}

fn has_shared_cofactor(words: &[Word], axes: usize) -> bool {
    for axis in 0..axes {
        let mut previous = BTreeSet::new();
        for w in words {
            let mut here = Vec::new();
            for value in 0..2 {
                let reduced: Vec<_> = (0..w.table.len())
                    .filter(|r| r & (1 << axis) == 0)
                    .map(|r| w.table[r | (value << axis)] as u8)
                    .collect();
                if reduced.windows(2).any(|p| p[0] != p[1]) {
                    if previous.contains(&reduced) {
                        return true;
                    }
                    here.push(reduced);
                }
            }
            previous.extend(here);
        }
    }
    false
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Objective {
    DirectArea,
    CofactorArea,
    LatePin,
}

pub(crate) fn offers(
    stock: &GateFn,
    groups: &[CapturedGroup],
    construction_limited: &mut bool,
) -> Vec<WordChoice> {
    let mut offers = Vec::new();
    for objective in [
        Objective::DirectArea,
        Objective::CofactorArea,
        Objective::LatePin,
    ] {
        if let Some(trial) = compose(
            stock,
            groups.iter().take(4),
            objective,
            "all",
            construction_limited,
        ) {
            offers.push(trial);
        }
    }
    if groups.len() > 1 {
        for group in groups.iter().take(4) {
            if let Some(trial) = compose(
                stock,
                std::iter::once(group),
                Objective::LatePin,
                "source-pair",
                construction_limited,
            ) {
                offers.push(trial);
            }
        }
    }
    offers
}

fn compose<'a>(
    stock: &GateFn,
    groups: impl Iterator<Item = &'a CapturedGroup>,
    objective: Objective,
    scope: &'static str,
    construction_limited: &mut bool,
) -> Option<WordChoice> {
    let mut chosen = None;
    let mut output_bits = BTreeSet::new();
    let mut source_pairs = Vec::new();
    for capture in groups {
        let g = &capture.group;
        let base = chosen.as_ref().unwrap_or(stock);
        let mut stats = Stats::default();
        let mode = if objective == Objective::DirectArea {
            Admission::Diagnostic
        } else {
            Admission::SafeOnly
        };
        let arrivals = get_aig_cone_stats(&base.gates, &g.coords).root_depths;
        let alternatives = trial::alternatives(base, g, mode, &mut stats);
        *construction_limited |= stats.construction_limited;
        log::debug!(
            "joint word {:?} axes={:?} words={} declined={:?} hits={} queries={} cubes={} adds={} speculative={:?}",
            objective,
            arrivals,
            g.words.len(),
            stats.declined,
            stats.reusable_hits,
            stats.scalar_queries,
            stats.cube_candidates,
            stats.constructed_ands,
            stats.elapsed
        );
        let Some(stock_price) = stats.stock.as_ref() else {
            continue;
        };
        let mut best = None;
        let mut best_key = None;
        for c in alternatives {
            let report = &stats.reports[c.report];
            log::trace!(
                "joint word {:?} {:?} retained_pins={} result={:?} area={} later={} changed={}",
                report.route,
                report.form,
                report.keep_late,
                report.verdict,
                report.and_nodes,
                report.late.len(),
                report.changed.len()
            );
            if matches!(report.route, Route::Direct) != (objective == Objective::DirectArea)
                || report.and_nodes >= stock_price.and_nodes
            {
                continue;
            }
            let rank = if let (Objective::LatePin, Route::Shannon(i)) = (objective, report.route) {
                (
                    usize::MAX - arrivals[i],
                    report
                        .changed
                        .iter()
                        .map(|&j| report.depths[j])
                        .max()
                        .unwrap_or(0),
                    report.depths.iter().sum(),
                )
            } else {
                (0, 0, 0)
            };
            let key = (
                !report.late.is_empty(),
                rank,
                report.and_nodes,
                report.depths.iter().copied().max().unwrap_or(0),
                c.report,
            );
            if best_key.is_none_or(|k| key < k) {
                best_key = Some(key);
                best = Some((c.graph, &report.changed));
            }
        }
        log::debug!("joint word picked {:?} key={:?}", objective, best_key);
        if let Some((graph, changed)) = best {
            chosen = Some(graph);
            output_bits.extend(changed.iter().copied());
            source_pairs.push(capture.source_ids);
        }
    }
    let mode = match objective {
        Objective::DirectArea => "direct-area",
        Objective::CofactorArea => "cofactor-area",
        Objective::LatePin => "late-pin",
    };
    let mut graph = chosen?;
    trial::attribute_replacements(stock, &mut graph);
    Some(WordChoice {
        graph,
        mode,
        scope,
        source_pairs,
        output_bits: output_bits.into_iter().collect(),
    })
}

#[cfg(test)]
mod tests;
