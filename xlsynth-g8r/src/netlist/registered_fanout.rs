// SPDX-License-Identifier: Apache-2.0

//! Buffer-tree candidates for the registered optimizer.
//!
//! The combinational mapper sees stage inputs as ideal sources. After assembly
//! the actual launch cell, downstream pin loads and capture setup arcs are
//! known. This pass keeps register masters, port connections and logic
//! functions fixed. It measures proposed trees with the same Liberty library
//! and wire model used for the original registered netlist. It does not
//! estimate routed delay or hold.

mod candidates;
mod edit;
mod graph;
mod plan;
#[cfg(test)]
mod tests;

pub(crate) use self::candidates::{ForestRetry, TopologyTrial, visit_topology_candidates};
use self::graph::{Buffer, Root};
use crate::liberty_model::Library;
use crate::netlist::sta::StaOptions;

fn buffer_choices<'a>(buffers: &'a [Buffer], library: &Library) -> Vec<&'a Buffer> {
    let mut legal: Vec<_> = buffers
        .iter()
        .filter(|b| {
            b.max_capacitance.is_some_and(|c| c.is_finite() && c > 0.0)
                && b.input_cap.iter().all(|&c| c.is_finite() && c > 0.0)
                && library.cells[b.cell].area.is_finite()
                && library.cells[b.cell].area > 0.0
        })
        .collect();
    // Try the characterized reference master and a high electrical-gain option.
    // Prefer small area, then lexical master name for equal-gain choices.
    legal.sort_by(|a, b| {
        let gain = |c: &Buffer| c.max_capacitance.unwrap() / c.input_cap[0].max(c.input_cap[1]);
        gain(b)
            .total_cmp(&gain(a))
            .then(
                library.cells[a.cell]
                    .area
                    .total_cmp(&library.cells[b.cell].area),
            )
            .then(library.cells[a.cell].name.cmp(&library.cells[b.cell].name))
    });
    let mut selected = Vec::new();
    if let Ok(name) = crate::liberty::io_defaults::resolve_reference_buffer(library)
        && let Some(b) = legal.iter().find(|b| library.cells[b.cell].name == name)
    {
        selected.push(*b);
    }
    if let Some(b) = legal
        .into_iter()
        .find(|b| selected.first().is_none_or(|r| r.cell != b.cell))
    {
        selected.push(b);
    }
    selected
}

fn root_load(root: &Root, timing: StaOptions) -> [f64; 2] {
    // fixed_load includes unrewirable sink pins and external output port loads.
    let wire = timing.wire_load.constant_capacitance
        + timing.wire_load.capacitance_per_sink * (root.fixed_sink_count + root.sinks.len()) as f64;
    root.sinks
        .iter()
        .fold(root.fixed_load.map(|c| c + wire), |caps, s| {
            [caps[0] + s.capacitance[0], caps[1] + s.capacitance[1]]
        })
}

fn fanout(r: &Root) -> f64 {
    r.fixed_fanout_load + r.sinks.iter().map(|s| s.fanout_load).sum::<f64>()
}
/// Nominate a register output using its characterized load limit.
fn capacitance_severity(root: &Root, timing: StaOptions) -> f64 {
    root.max_capacitance
        .filter(|limit| limit.is_finite() && *limit > 0.0)
        .map_or(0.0, |limit| {
            let [rise, fall] = root_load(root, timing);
            rise.max(fall) / limit
        })
}

fn severity(root: &Root, slew: Option<f64>, timing: StaOptions) -> f64 {
    let [rise, fall] = root_load(root, timing);
    let ratio = |value: Option<f64>, limit: Option<f64>| {
        value
            .zip(limit)
            .filter(|(_, v)| v.is_finite() && *v > 0.0)
            .map_or(0.0, |(value, v)| value / v)
    };
    ratio(Some(rise.max(fall)), root.max_capacitance)
        .max(ratio(slew, root.max_transition))
        .max(ratio(Some(fanout(root)), root.max_fanout))
}
