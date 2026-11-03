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
struct Observed {
    axes: Vec<AigOperand>,
    take_x: u64,
    take_y: u64,
    lanes: Vec<Lane>,
}

pub(crate) struct CapturedGroup {
    source_ids: [usize; 2],
    group: Group,
}

pub(super) fn capture(f: &ir::Fn, env: &GateEnv, graph: &GateFn) -> Vec<CapturedGroup> {
    if graph.gates.len() > 32768 {
        return Vec::new();
    }
    let mut outputs: HashMap<AigRef, Vec<(usize, usize)>> = HashMap::new();
    for (i, port) in graph.outputs.iter().enumerate() {
        for (j, bit) in port.bit_vector.iter_lsb_to_msb().enumerate() {
            outputs.entry(bit.node).or_default().push((i, j));
        }
    }
    let get = |n: NodeRef, i: usize| {
        if !env
            .demanded_bits
            .as_ref()?
            .get(&n)?
            .get(i)
            .copied()
            .unwrap_or(false)
        {
            return None;
        }
        let v = env.get_bit_vector(n).ok()?;
        (i < v.get_bit_count()).then(|| *v.get_lsb(i))
    };
    let mut budget = Budget::default();
    let mut siblings: BTreeMap<(usize, usize), Vec<Observed>> = BTreeMap::new();
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
        let Ok(v) = env.get_bit_vector(root) else {
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
            .map(|&(n, b)| get(n, b))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let mut lanes = Vec::new();
        for (bit, &selected) in v.iter_lsb_to_msb().enumerate() {
            let Some(targets) = outputs.get(&selected.node) else {
                continue;
            };
            let (Some(_), Some(x), Some(y)) = (get(root, bit), get(d.x, bit), get(d.y, bit)) else {
                continue;
            };
            for &(output, po_bit) in targets {
                lanes.push(Lane {
                    output,
                    bit: po_bit,
                    selected,
                    x,
                    y,
                    constant: d.constant[bit],
                });
            }
        }
        if !lanes.is_empty() {
            siblings
                .entry((d.x.index, d.y.index))
                .or_default()
                .push(Observed {
                    axes,
                    take_x: d.take_x,
                    take_y: d.take_y,
                    lanes,
                });
        }
    }
    siblings
        .into_iter()
        .filter_map(|((x, y), records)| {
            Some(CapturedGroup {
                source_ids: [f.nodes[x].text_id, f.nodes[y].text_id],
                group: group(graph, records)?,
            })
        })
        .collect()
}

fn group(graph: &GateFn, records: Vec<Observed>) -> Option<Group> {
    if !(2..=4).contains(&records.len()) {
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
        words.push(Word {
            table,
            lanes: r.lanes,
        });
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
    Some(WordChoice {
        graph: chosen?,
        mode,
        scope,
        source_pairs,
        output_bits: output_bits.into_iter().collect(),
    })
}
