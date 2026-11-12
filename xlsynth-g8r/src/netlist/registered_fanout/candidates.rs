// SPDX-License-Identifier: Apache-2.0

use super::{buffer_choices, capacitance_severity, edit, graph, plan, severity};
use crate::liberty_model::Library;
use crate::netlist::fanout_opt::FanoutOptimizationOptions;
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::sta::{PreparedStaLibrary, StaOptions};
use anyhow::Result;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TopologyTrial {
    Forest,
    ElectricalRetry,
}

/// A request to try another candidate with the listed outputs buffered.
pub(crate) struct ForestRetry {
    pub rank: (u8, f64),
    pub outputs: Vec<(usize, String)>,
}

/// Generate wiring candidates for the registered stage. The caller evaluates
/// timing and area and selects a winner. Each candidate starts
/// from `original`; the count includes every candidate passed to the caller.
pub(crate) fn visit_topology_candidates(
    original: &MappedNetlist,
    library: &Library,
    prepared: &PreparedStaLibrary<'_>,
    timing: StaOptions,
    options: &FanoutOptimizationOptions,
    mut visit: impl FnMut(MappedNetlist, TopologyTrial) -> Result<Option<ForestRetry>>,
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
        false,
    )?;
    roots.retain(|r| !r.sinks.is_empty());
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
    for (i, r) in roots.iter().enumerate() {
        let Some(slew) = slews[i] else { continue };
        let internal = library.cells[r.cell].sequential.is_empty();
        let electrical = if internal {
            severity(r, Some(slew), timing)
        } else {
            capacitance_severity(r, timing)
        };
        let large_internal = internal && r.sinks.len() >= threshold;
        if electrical > 1.0 + 1e-9 || large_internal {
            ranked.push((i, electrical));
        }
    }
    ranked.sort_unstable_by(|&(i, a), &(j, b)| {
        b.total_cmp(&a)
            .then(roots[j].sinks.len().cmp(&roots[i].sinks.len()))
            .then(i.cmp(&j))
    });
    if ranked.is_empty() {
        return Ok(0);
    }
    // A single proposal can repair equally critical paths. Bound the number
    // of individual alternatives separately from the width of that batch.
    const MAX_SINGLE: usize = 8;
    const MAX_GROUP: usize = 64;
    let single = ranked.len().min(options.drivers_per_round).min(MAX_SINGLE);
    // Reserve a place for each kind of driver: a bank of equally overloaded
    // Qs must not hide the strongest internal branch, or vice versa.
    let first = ranked[0].0;
    let is_internal = |i: usize| library.cells[roots[i].cell].sequential.is_empty();
    let remaining = ranked.iter().skip(1).map(|&(i, _)| i);
    let other = remaining
        .clone()
        .find(|&i| is_internal(i) != is_internal(first));
    let drivers: Vec<_> = std::iter::once(first)
        .chain(other)
        .chain(remaining.filter(|&i| Some(i) != other))
        .take(MAX_GROUP)
        .collect();
    let mut retry = None;
    let mut visited = 0;
    let buffers = graph::buffers(library, prepared);
    for buffer in buffer_choices(&buffers, library) {
        let mut previous = Vec::new();
        for width in [16, 64] {
            let plans: Vec<_> = drivers
                .iter()
                .filter_map(|&i| plan::tree(&roots[i], buffer, width, timing).map(|p| (i, p)))
                .collect();
            let mut emit = |p: &[(usize, Vec<edit::Branch>)]| -> Result<()> {
                if !p.is_empty() && !previous.iter().any(|old: &Vec<_>| old.as_slice() == p) {
                    let mut trial = original.clone();
                    edit::apply_forest(&mut trial, library, buffer, &roots, p)?;
                    if let Some(request) = visit(trial, TopologyTrial::Forest)? {
                        let priority = (request.outputs.len(), request.rank);
                        if retry.as_ref().is_none_or(|(best, _, _)| priority < *best) {
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
                                retry = Some((priority, buffer, expanded));
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
                emit(std::slice::from_ref(p))?;
            }
            emit(&plans)?;
        }
    }
    // Evaluate at most one additional repair proposal. Prefer repairing fewer
    // other outputs, then use the original delay and area ranking.
    if let Some((_, buffer, plans)) = retry {
        let mut trial = original.clone();
        edit::apply_forest(&mut trial, library, buffer, &roots, &plans)?;
        visit(trial, TopologyTrial::ElectricalRetry)?;
        visited += 1;
    }
    Ok(visited)
}
