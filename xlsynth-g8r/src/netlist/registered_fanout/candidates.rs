// SPDX-License-Identifier: Apache-2.0

use super::proposal::TopologyProposal;
use super::{buffer_choices, edit, graph, plan, severity};
use crate::liberty_model::Library;
use crate::netlist::fanout_opt::FanoutOptimizationOptions;
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::sta::incremental::IncrementalSta;
use crate::netlist::sta::{PreparedStaLibrary, StaOptions};
use anyhow::Result;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TopologyTrial {
    Forest,
    ElectricalRetry(Option<TimingConeTrial>),
    TimingCone(TimingConeTrial),
}

#[derive(Clone, Copy, PartialEq, Eq)]
/// Original timing window and number of buildable roots in the joint plan.
pub(crate) struct TimingConeTrial {
    pub window: usize,
    pub roots: usize,
}

impl TopologyTrial {
    pub(crate) fn timing_cone(self) -> Option<TimingConeTrial> {
        match self {
            Self::TimingCone(cone) | Self::ElectricalRetry(Some(cone)) => Some(cone),
            _ => None,
        }
    }
}

/// A buffer-tree candidate that would be acceptable if the listed existing
/// output pins were also repaired.
pub(crate) struct ForestRetry {
    pub rank: (u8, f64),
    pub outputs: Vec<(usize, String)>,
}

/// Generate wiring candidates for the registered stage. The caller evaluates
/// timing and electrical limits and selects a winner. Each candidate starts
/// from `original`; the count includes every candidate passed to the caller.
pub(crate) fn visit_topology_candidates<'lib>(
    original: &MappedNetlist,
    library: &Library,
    prepared: &PreparedStaLibrary<'_>,
    timing: StaOptions,
    options: &FanoutOptimizationOptions,
    sta: &mut IncrementalSta<'lib>,
    timing_cones: Option<usize>,
    mut visit: impl FnMut(
        &mut IncrementalSta<'lib>,
        TopologyTrial,
        TopologyProposal<'_>,
    ) -> Result<Option<ForestRetry>>,
) -> Result<usize> {
    if options.drivers_per_round == 0 {
        return Ok(0);
    }
    let threshold = options.fanout_trigger.max(2);
    let mut roots = graph::collect_roots(
        original,
        library,
        timing.module_output_load,
        Some(threshold),
    )?;
    roots.retain(|r| r.sinks.len() >= 2);
    if roots.is_empty() {
        return Ok(0);
    }
    let observed: Vec<_> = roots.iter().map(|r| r.source_bit).collect();
    let launches = graph::launches(original, library)?;
    let (_, slews) = prepared.analyze_register_boundary_with_slews(
        &original.module,
        &original.nets,
        &original.interner,
        timing,
        true,
        &launches,
        &observed,
    )?;
    // Use electrical stress and fanout to pick a bounded set of drivers worth
    // timing. The registered STA result decides whether a plan actually helps.
    let mut ranked = Vec::new();
    let mut cone_eligible = Vec::new();
    for (i, r) in roots.iter().enumerate() {
        let Some(slew) = slews[i] else { continue };
        let electrical = severity(r, Some(slew), timing);
        let large_internal =
            library.cells[r.cell].sequential.is_empty() && r.sinks.len() >= threshold;
        if timing_cones.is_some() && (electrical > 1.0 + 1e-9 || r.sinks.len() >= threshold) {
            cone_eligible.push(i);
        }
        if electrical > 1.0 + 1e-9 || large_internal {
            ranked.push((i, electrical));
        }
    }
    ranked.sort_unstable_by(|&(i, a), &(j, b)| {
        b.total_cmp(&a)
            .then(roots[j].sinks.len().cmp(&roots[i].sinks.len()))
            .then(i.cmp(&j))
    });
    let cones = match timing_cones {
        Some(boundary) => {
            super::selection::TimingCones::prepare(sta, &roots, &cone_eligible, boundary)?
        }
        None => None,
    };
    if ranked.is_empty() && cones.is_none() {
        return Ok(0);
    }
    const WIDTHS: [usize; 2] = [16, 64];
    let buffers = graph::buffers(library, prepared);
    let choices = buffer_choices(&buffers, library);
    let cones = super::selection::plan_windows(cones, &roots, &choices, &WIDTHS, timing);
    // A single proposal can repair equally critical paths. Bound the number
    // of individual alternatives separately from the width of that batch.
    const MAX_SINGLE: usize = 8;
    const MAX_GROUP: usize = 64;
    let single = ranked.len().min(options.drivers_per_round).min(MAX_SINGLE);
    // Reserve a place for each kind of driver: a bank of equally overloaded
    // Qs must not hide the strongest internal branch, or vice versa.
    let drivers: Vec<_> = match ranked.first() {
        Some(&(first, _)) => {
            let is_internal = |i: usize| library.cells[roots[i].cell].sequential.is_empty();
            let remaining = ranked.iter().skip(1).map(|&(i, _)| i);
            let other = remaining
                .clone()
                .find(|&i| is_internal(i) != is_internal(first));
            std::iter::once(first)
                .chain(other)
                .chain(remaining.filter(|&i| Some(i) != other))
                .take(MAX_GROUP)
                .collect()
        }
        None => Vec::new(),
    };
    let mut retry = None;
    let mut visited = 0;
    for buffer in choices {
        let mut previous = Vec::new();
        for width in WIDTHS {
            let plans: Vec<_> = drivers
                .iter()
                .filter_map(|&i| plan::tree(&roots[i], buffer, width, timing).map(|p| (i, p)))
                .collect();
            let mut emit = |p: &[(usize, Vec<edit::Branch>)], kind: TopologyTrial| -> Result<()> {
                if !p.is_empty() && !previous.iter().any(|old: &Vec<_>| old.as_slice() == p) {
                    let proposal = TopologyProposal::forest(&roots, buffer, p);
                    if let Some(request) = visit(sta, kind, proposal)? {
                        let priority = (request.outputs.len(), request.rank);
                        if retry
                            .as_ref()
                            .is_none_or(|(best, _, _, _)| priority < *best)
                        {
                            let additional: Option<Vec<_>> = request
                                .outputs
                                .iter()
                                .map(|(inst, pin)| {
                                    let i = roots
                                        .iter()
                                        .position(|r| r.instance == *inst && r.pin_name == *pin)?;
                                    if p.iter().any(|&(j, _)| j == i) {
                                        return None;
                                    }
                                    Some((i, plan::tree(&roots[i], buffer, width, timing)?))
                                })
                                .collect();
                            if let Some(extra) = additional {
                                let mut expanded = p.to_vec();
                                expanded.extend(extra);
                                retry = Some((priority, buffer, expanded, kind.timing_cone()));
                            }
                        }
                    }
                    visited += 1;
                    previous.push(p.to_vec());
                }
                Ok(())
            };
            for p in plans
                .iter()
                .filter(|(id, _)| drivers[..single].contains(id))
            {
                emit(std::slice::from_ref(p), TopologyTrial::Forest)?;
            }
            emit(&plans, TopologyTrial::Forest)?;
            if let Some(windows) = cones.get(&(buffer.cell, width)) {
                for (window, cone) in windows.iter().enumerate() {
                    // The ordinary forest may already have evaluated this group
                    // in a different instance order. Preserve its generated
                    // names and score.
                    if cone.is_empty()
                        || (cone.len() == plans.len()
                            && plans
                                .iter()
                                .all(|(i, _)| cone.binary_search_by_key(i, |(id, _)| *id).is_ok()))
                    {
                        continue;
                    }
                    emit(
                        cone,
                        TopologyTrial::TimingCone(TimingConeTrial {
                            window,
                            roots: cone.len(),
                        }),
                    )?;
                }
            }
        }
    }
    // Evaluate at most one additional repair proposal. Prefer repairing fewer
    // other outputs, then use the original delay and area ranking.
    if let Some((_, buffer, plans, parent)) = retry {
        visit(
            sta,
            TopologyTrial::ElectricalRetry(parent),
            TopologyProposal::forest(&roots, buffer, &plans),
        )?;
        visited += 1;
    }
    Ok(visited)
}
