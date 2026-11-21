// SPDX-License-Identifier: Apache-2.0

use super::{buffers::Buffer, edit::Branch};
use crate::netlist::editability::EditableOutput as Root;
use crate::netlist::sta::StaOptions;

/// Build from the sinks upward, then number the branches in driver-to-load
/// order.
pub(super) fn tree(
    root: &Root,
    buffer: &Buffer,
    width: usize,
    timing: StaOptions,
) -> Option<Vec<Branch>> {
    if width < 2 || root.sinks.is_empty() {
        return None;
    }
    let wire = timing.wire_load;
    let wire_cap =
        |pins: usize| wire.constant_capacitance + wire.capacitance_per_sink * pins as f64;
    let capacity = |limit: Option<f64>| limit.unwrap_or(f64::INFINITY) * super::CAPACITY_TOLERANCE;
    let buffer_limit = capacity(buffer.max_capacitance);
    let root_limit = capacity(super::root_cap_ceiling(root));
    let buffer_fanout_limit = capacity(buffer.max_fanout);
    let root_fanout_limit = capacity(root.electrical.limits.fanout);
    let mut original = root.fixed_load;
    for sink in &root.sinks {
        for (cap, pin) in original.iter_mut().zip(sink.capacitance) {
            *cap += pin;
        }
    }
    let original = original.map(|v| v + wire_cap(root.electrical.load.pins));
    if original.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // Deal the largest sinks first. Alternating direction balances successive
    // rows across leaves while preserving a deterministic order for
    // equal-capacitance pins.
    let mut order: Vec<usize> = (0..root.sinks.len()).collect();
    let peak = |i: usize| root.sinks[i].capacitance[0].max(root.sinks[i].capacitance[1]);
    order.sort_unstable_by(|&a, &b| peak(b).total_cmp(&peak(a)).then(a.cmp(&b)));
    let mut leaf_count = order.len().div_ceil(width);
    let leaves = loop {
        let mut leaves: Vec<Branch> = (0..leaf_count)
            .map(|_| Branch {
                parent: None,
                sinks: Vec::new(),
            })
            .collect();
        let mut loads = vec![[0.0; 2]; leaf_count];
        for (row, chunk) in order.chunks(leaf_count).enumerate() {
            for (i, &idx) in chunk.iter().enumerate() {
                let leaf = if row % 2 == 0 { i } else { leaf_count - 1 - i };
                leaves[leaf].sinks.push(idx);
                for (cap, pin) in loads[leaf].iter_mut().zip(root.sinks[idx].capacitance) {
                    *cap += pin;
                }
            }
        }
        if leaves.iter().zip(loads).all(|(leaf, caps)| {
            let wire = wire_cap(leaf.sinks.len());
            let fanout: f64 = leaf.sinks.iter().map(|&k| root.sinks[k].fanout_load).sum();
            fanout <= buffer_fanout_limit
                && caps
                    .into_iter()
                    .all(|cap| (cap + wire).is_finite() && cap + wire <= buffer_limit)
        }) {
            break leaves;
        }
        if leaf_count == order.len() {
            return None;
        }
        leaf_count = leaf_count.saturating_mul(2).min(order.len());
    };

    let repeated_load = |pins: usize, edge: usize| buffer.input_cap[edge] * pins as f64;
    let root_fits = |pins: usize| {
        let fixed = wire_cap(root.fixed_sink_count + pins);
        let load = std::array::from_fn::<_, 2, _>(|edge| {
            root.fixed_load[edge] + fixed + repeated_load(pins, edge)
        });
        pins <= width
            && root.fixed_fanout_load + buffer.input_fanout_load * pins as f64 <= root_fanout_limit
            && (0..2).all(|edge| load[edge] <= root_limit && load[edge] <= original[edge])
            && (0..2).any(|edge| load[edge] < original[edge])
    };
    let mut counts = vec![leaves.len()];
    while !root_fits(*counts.last()?) {
        let children = *counts.last()?;
        if counts.len() == 3 || children <= 1 {
            return None;
        }
        let fanout = (2..=width.min(children)).rev().find(|&n| {
            buffer.input_fanout_load * n as f64 <= buffer_fanout_limit
                && (0..2).all(|edge| {
                    let cap = repeated_load(n, edge) + wire_cap(n);
                    cap.is_finite() && cap <= buffer_limit
                })
        })?;
        counts.push(children.div_ceil(fanout));
    }

    counts.reverse();
    let mut branches = Vec::new();
    let mut leaf_iter = leaves.into_iter();
    let mut preceding_count = 0;
    let mut preceding_start = 0;
    for (level, &count) in counts.iter().enumerate() {
        let start = branches.len();
        for i in 0..count {
            let mut branch = if level + 1 == counts.len() {
                leaf_iter.next()?
            } else {
                Branch {
                    parent: None,
                    sinks: Vec::new(),
                }
            };
            if level > 0 {
                branch.parent = Some(preceding_start + i % preceding_count);
            }
            branches.push(branch);
        }
        preceding_start = start;
        preceding_count = count;
    }
    Some(branches)
}
