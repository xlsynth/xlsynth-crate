// SPDX-License-Identifier: Apache-2.0

use super::buffers::InsertableBufferCell;
use crate::netlist::editability::EditableOutput;
use crate::netlist::electrical::RiseFall;
use crate::netlist::sta::StaOptions;

/// A proposed buffer tree for one entry in the candidate driver list.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct BufferTreePlan {
    /// Index into the candidate driver list.
    pub(super) driver_index: usize,
    pub(super) buffers: Vec<PlannedBuffer>,
}

/// One buffer to insert, listed after any buffer driving its input.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct PlannedBuffer {
    /// Index into `BufferTreePlan::buffers`, or `None` for the original driver.
    ///
    /// ```text
    /// FF.Q --> BUF0 --> BUF1 --> gate.A
    /// BUF0: None       BUF1: Some(0)
    /// ```
    pub(super) driving_buffer_index: Option<usize>,
    /// Indices into `driver.sinks` identifying input pins to reconnect to
    /// this buffer's output.
    pub(super) sink_indices: Vec<usize>,
}

/// Plans a buffer tree connecting a driver to selected sink pins.
/// Returns `None` if no tree meets the load and fanout bounds while
/// reducing the original driver's capacitive load.
pub(super) fn build_buffer_tree(
    drivers: &[EditableOutput],
    driver_index: usize,
    buffer: &InsertableBufferCell,
    width: usize,
    timing: StaOptions,
) -> Option<BufferTreePlan> {
    let driver = &drivers[driver_index];
    if width < 2 || driver.sinks.is_empty() {
        return None;
    }
    let wire = timing.wire_load;
    let wire_cap =
        |pins: usize| wire.constant_capacitance + wire.capacitance_per_sink * pins as f64;
    let capacity = |limit: Option<f64>| limit.unwrap_or(f64::INFINITY) * super::CAPACITY_TOLERANCE;
    let buffer_limit = capacity(buffer.output_load_ceiling);
    let driver_load_limit = capacity(super::driver_output_load_ceiling(driver));
    let buffer_fanout_limit = capacity(buffer.max_fanout);
    let driver_fanout_limit = capacity(driver.electrical.limits.fanout);
    let original = super::driver_load_capacitance(driver, timing);
    if !original.rise.is_finite() || !original.fall.is_finite() {
        return None;
    }

    let fits_buffer = |pins, load: RiseFall<f64>, fanout: f64| {
        let wire = wire_cap(pins);
        fanout <= buffer_fanout_limit
            && [load.rise + wire, load.fall + wire]
                .into_iter()
                .all(|cap| cap.is_finite() && cap <= buffer_limit)
    };
    // Distribute sinks round-robin in netlist order. Experiments
    // found no benefit from sorting them by capacitance first.
    // Double the number of buffers until every group fits.
    let sinks = driver.sinks.len();
    let mut leaf_count = sinks.div_ceil(width);
    let leaves = loop {
        let mut leaves = vec![PlannedBuffer::default(); leaf_count];
        for i in 0..sinks {
            leaves[i % leaf_count].sink_indices.push(i);
        }
        if leaves.iter().all(|leaf| {
            let mut load = RiseFall::both(0.);
            let mut fanout = 0.;
            for &i in &leaf.sink_indices {
                let sink = &driver.sinks[i];
                load.rise += sink.capacitance.rise;
                load.fall += sink.capacitance.fall;
                fanout += sink.fanout_load;
            }
            fits_buffer(leaf.sink_indices.len(), load, fanout)
        }) {
            break leaves;
        }
        if leaf_count == sinks {
            return None;
        }
        leaf_count = leaf_count.saturating_mul(2).min(sinks);
    };

    let repeated_load = |pins: usize| RiseFall {
        rise: buffer.input_capacitance.rise * pins as f64,
        fall: buffer.input_capacitance.fall * pins as f64,
    };
    // Buffering must reduce the original driver's capacitive load
    // for at least one transition without increasing it for the other.
    // Otherwise, the tree provides no local load reduction to justify
    // its additional cells and delay.
    let driver_fits = |pins: usize| {
        let fixed = wire_cap(driver.fixed_sink_count + pins);
        let input_load = repeated_load(pins);
        let rise = driver.fixed_load.rise + fixed + input_load.rise;
        let fall = driver.fixed_load.fall + fixed + input_load.fall;
        pins <= width
            && driver.fixed_fanout_load + buffer.input_fanout_load * pins as f64
                <= driver_fanout_limit
            && rise <= driver_load_limit
            && fall <= driver_load_limit
            && rise <= original.rise
            && fall <= original.fall
            && (rise < original.rise || fall < original.fall)
    };
    // Build the tree from sinks toward the driver, adding levels
    // until the driver can handle their combined input load.
    //
    //               +--> BUF --> sinks
    // FF.Q --> BUF -+
    //               +--> BUF --> sinks
    //
    // Each additional level must reduce the number of buffer
    // inputs connected directly to the driver. Fixed connections remain
    // attached to the original driver.
    let mut counts = vec![leaves.len()];
    while !driver_fits(*counts.last()?) {
        let children = *counts.last()?;
        if children <= 1 {
            return None;
        }
        let fanout = (2..=width.min(children))
            .rev()
            .find(|&n| fits_buffer(n, repeated_load(n), buffer.input_fanout_load * n as f64))?;
        counts.push(children.div_ceil(fanout));
    }

    // List each buffer after the buffer driving it. Distribute children
    // round-robin.
    let mut buffers = Vec::with_capacity(counts.iter().sum());
    let mut driving_buffer_range = 0..0;
    for count in counts.into_iter().rev() {
        let start = buffers.len();
        for i in 0..count {
            buffers.push(PlannedBuffer {
                driving_buffer_index: (!driving_buffer_range.is_empty())
                    .then(|| driving_buffer_range.start + i % driving_buffer_range.len()),
                sink_indices: Vec::new(),
            });
        }
        driving_buffer_range = start..buffers.len();
    }
    for (planned_buffer, leaf) in buffers[driving_buffer_range].iter_mut().zip(leaves) {
        planned_buffer.sink_indices = leaf.sink_indices;
    }
    Some(BufferTreePlan {
        driver_index,
        buffers,
    })
}
