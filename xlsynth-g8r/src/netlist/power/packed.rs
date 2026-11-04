// SPDX-License-Identifier: Apache-2.0

//! Replay Liberty selectors over batches of 256 ordered, settled timestamps.

use super::{
    BoundConditionTerm, ConditionValueSlot, EdgeValue, GvDynamicPowerDiagnostics, InstanceTopology,
    InternalEventKey, ObservedActivity, PowerInstance, PowerModelView, PowerSignal, PowerTopology,
    PowerTransitionKind, ResolvedSource, Signal, SignalId, TimingCauseKey, attribute_candidates,
    is_sequential_edge_timing_type, simulate_chunk, timing_sense_matches,
    timing_source_drives_output, timing_type_allows_output, transition_table,
};
use crate::aig::AigOperand;
use crate::aig_sim::gate_simd::Vec256;
use crate::liberty_model::{Library, Pin, PinDirection};
use std::collections::BTreeMap;
use xlsynth_pir::IrBits;

const EMPTY: Vec256 = Vec256::zero();

/// The two endpoints of every transition in a batch, earliest sample in bit 0.
#[derive(Clone, Copy)]
struct Samples {
    before: Vec256,
    now: Vec256,
}

impl Samples {
    fn from_batch(now: Vec256, predecessor: bool) -> Self {
        // Bit 0 in each word follows bit 63 of the preceding word.
        let [a, b, c, d] = now.to_array();
        let before = Vec256::from_words([
            (a << 1) | u64::from(predecessor),
            (b << 1) | (a >> 63),
            (c << 1) | (b >> 63),
            (d << 1) | (c >> 63),
        ]);
        Self { before, now }
    }

    fn inverted(self) -> Self {
        Self {
            before: !self.before,
            now: !self.now,
        }
    }

    fn changed(self) -> Vec256 {
        self.before ^ self.now
    }

    /// Selects timestamps with the given settled level from a transition mask.
    fn level(self, high: bool, mask: Vec256) -> Vec256 {
        mask & if high { self.now } else { !self.now }
    }
}

/// Settled values of each distinct low-transparent ICG latch across a batch.
pub(super) struct GateLatchSamples {
    by_enable: BTreeMap<AigOperand, Samples>,
}

impl GateLatchSamples {
    pub(super) fn new(topology: &PowerTopology) -> Self {
        let by_enable = topology
            .signals
            .iter()
            .filter_map(|signal| {
                let PowerSignal::GatedClock(enable) = signal.source else {
                    return None;
                };
                Some((
                    enable,
                    Samples {
                        before: EMPTY,
                        now: EMPTY,
                    },
                ))
            })
            .collect();
        Self { by_enable }
    }

    /// Each high sample is immediately preceded by the low pre-edge sample.
    /// During that high phase, register updates may change the live enable
    /// while the latch retains the bit that admitted the clock pulse.
    pub(super) fn capture(
        &mut self,
        nodes: &[Vec256],
        clock: Option<Vec256>,
    ) -> Result<(), String> {
        for (&enable, latch) in &mut self.by_enable {
            let high =
                clock.ok_or_else(|| "power sample omitted selected clock value".to_string())?;
            let raw = nodes
                .get(enable.node.id)
                .ok_or_else(|| format!("AIG node {} is out of range", enable.node.id))?
                .apply_neg(enable.negated);
            // Every nonfinal batch has 256 lanes. If this batch starts high,
            // the previous batch ended low, with the latch tracking its enable.
            let predecessor = latch.now.get_lane(255);
            let input = Samples::from_batch(raw, predecessor);
            let held = (high & input.before) | (!high & input.now);
            *latch = Samples::from_batch(held, predecessor);
        }
        Ok(())
    }

    pub(super) fn value_at(&self, enable: AigOperand, lane: usize) -> Result<bool, String> {
        self.by_enable
            .get(&enable)
            .map(|sample| sample.now.get_lane(lane))
            .ok_or_else(|| format!("no sampled clock-gate latch for enable {enable:?}"))
    }
}

/// Packs the selected root clock levels, or returns None for an unclocked
/// trace.
pub(super) fn clock_levels(values: &[Option<bool>]) -> Option<Vec256> {
    let mut words = [0u64; 4];
    for (lane, value) in values.iter().enumerate() {
        words[lane / 64] |= u64::from((*value)?) << (lane % 64);
    }
    Some(Vec256::from_words(words))
}

/// Packed simulator nodes, predecessor bits and clock phases for one batch.
struct SampleBatch<'a> {
    nodes: &'a [Vec256],
    carry: &'a [bool],
    valid: Vec256,
    clock: Option<Samples>,
    gate_latches: &'a GateLatchSamples,
    active_edges: EdgeValue<Vec256>,
}

impl SampleBatch<'_> {
    fn signal(&self, source: PowerSignal) -> Option<Samples> {
        match source {
            PowerSignal::Clock => self.clock,
            PowerSignal::GateLatch(enable) => self.gate_latches.by_enable.get(&enable).copied(),
            PowerSignal::GatedClock(enable) => {
                let latch = self.gate_latches.by_enable.get(&enable)?;
                let clock = self.clock?;
                Some(Samples {
                    before: clock.before & latch.before,
                    now: clock.now & latch.now,
                })
            }
            PowerSignal::Operand(op) => {
                let value =
                    Samples::from_batch(*self.nodes.get(op.node.id)?, self.carry[op.node.id]);
                Some(if op.negated { value.inverted() } else { value })
            }
        }
    }

    /// Uses the same timing-type rules as single-transition replay.
    fn timing_mask(&self, timing_type: &str, output_rise: bool, sequential: bool) -> Vec256 {
        if sequential {
            let mut allowed = EMPTY;
            for (clock_rise, mask) in [
                (true, self.active_edges.rise),
                (false, self.active_edges.fall),
            ] {
                if timing_type_allows_output(
                    timing_type,
                    output_rise,
                    PowerTransitionKind::ActiveClockEdge { clock_rise },
                    true,
                ) {
                    allowed |= mask;
                }
            }
            allowed
        } else if timing_type_allows_output(
            timing_type,
            output_rise,
            PowerTransitionKind::Combinational,
            false,
        ) {
            self.valid
        } else {
            EMPTY
        }
    }
}

impl BoundConditionTerm {
    /// Evaluates both endpoints. An unbound leaf requires the scalar
    /// interpreter.
    fn evaluate_batch(&self, pins: &[Samples], aliases: &[Samples]) -> Option<Samples> {
        let binary = |a: &Self, b: &Self, op: fn(Vec256, Vec256) -> Vec256| {
            let a = a.evaluate_batch(pins, aliases)?;
            let b = b.evaluate_batch(pins, aliases)?;
            Some(Samples {
                before: op(a.before, b.before),
                now: op(a.now, b.now),
            })
        };
        match self {
            Self::Input(Some(ConditionValueSlot::Pin(i))) => pins.get(*i).copied(),
            Self::Input(Some(ConditionValueSlot::StateAlias(i))) => aliases.get(*i).copied(),
            Self::Input(None) => None,
            Self::And(a, b) => binary(a, b, |x, y| x & y),
            Self::Or(a, b) => binary(a, b, |x, y| x | y),
            Self::Xor(a, b) => binary(a, b, |x, y| x ^ y),
            Self::Negate(inner) => Some(inner.evaluate_batch(pins, aliases)?.inverted()),
            Self::Constant(value) => {
                let bits = Vec256::splat(*value);
                Some(Samples {
                    before: bits,
                    now: bits,
                })
            }
        }
    }
}

/// Counts each successful selector request, even if its group charges no
/// energy.
fn eligible(
    requests: Vec256,
    condition: Option<usize>,
    conditions: &[Option<Samples>],
    diagnostics: &mut GvDynamicPowerDiagnostics,
) -> Option<Vec256> {
    let Some(index) = condition else {
        return Some(requests);
    };
    if requests == EMPTY {
        return Some(EMPTY);
    }
    let values = conditions[index]?;
    diagnostics.when_evaluation_count += requests.popcount();
    diagnostics.when_changed_during_transition_count += (requests & values.changed()).popcount();
    Some(requests & values.before)
}

/// Candidate lists for one instance. Each lane retains Liberty selector order.
struct Candidates<K> {
    lanes: [Vec<K>; 256],
}

impl<K: Copy> Candidates<K> {
    fn new() -> Self {
        Self {
            lanes: std::array::from_fn(|_| Vec::new()),
        }
    }

    fn add(&mut self, mask: Vec256, key: K) {
        for lane in mask.set_lanes() {
            self.lanes[lane].push(key);
        }
    }
}

/// Identifies one Liberty owner pin and its valid output edges in this batch.
struct OwnerPin<'a> {
    instance_index: usize,
    owner: usize,
    layout: &'a InstanceTopology,
    pin: &'a Pin,
    output_changed: Vec256,
}

/// Reused across cells. Packed pin, alias and formula values are
/// instance-local.
struct CellScratch {
    pins: Vec<Samples>,
    aliases: Vec<Samples>,
    conditions: Vec<Option<Samples>>,
    timing: Candidates<TimingCauseKey>,
    internal: Candidates<InternalEventKey>,
    rail_candidate_counts: Vec<usize>,
}

impl CellScratch {
    fn new() -> Self {
        Self {
            pins: Vec::new(),
            aliases: Vec::new(),
            conditions: Vec::new(),
            timing: Candidates::new(),
            internal: Candidates::new(),
            rail_candidate_counts: Vec::new(),
        }
    }

    /// Builds the same candidate lists as scalar replay, with requests masked
    /// before testing a condition and related-pin checks in their original
    /// order.
    fn observe_cell(
        &mut self,
        batch: &SampleBatch<'_>,
        instance_index: usize,
        instance: &PowerInstance,
        topology: &PowerTopology,
        library: &Library,
        result: &mut ObservedActivity,
    ) -> Option<()> {
        let layout = &topology.instances[instance_index];
        let bindings = &layout.bindings;
        self.pins.clear();
        let mut changed = EMPTY;
        for pin in &instance.pins {
            let value = batch.signal(pin.signal)?;
            self.pins.push(value);
            changed |= value.changed();
        }
        self.aliases.clear();
        for alias in &layout.state_aliases {
            let value = batch.signal(alias.signal)?;
            self.aliases.push(if alias.complemented {
                value.inverted()
            } else {
                value
            });
        }
        let changed = changed & batch.valid;
        if changed == EMPTY {
            return Some(());
        }
        self.conditions.clear();
        self.conditions
            .extend(bindings.conditions.iter().map(|condition| {
                condition
                    .term
                    .as_ref()
                    .ok()?
                    .evaluate_batch(&self.pins, &self.aliases)
            }));

        let cell = &library.cells[layout.cell];
        for (owner, &output) in layout.output_signals.iter().enumerate() {
            let output_changed = if output.is_some() {
                self.pins[owner].changed() & changed
            } else {
                EMPTY
            };
            let pin = OwnerPin {
                instance_index,
                owner,
                layout,
                pin: &cell.pins[layout.liberty_pins[owner]],
                output_changed,
            };
            if let Some(output) = output
                && output_changed != EMPTY
            {
                self.select_timing(&pin, output, batch, library, &topology.signals, result)?;
            }
            self.select_internal(&pin, changed, &mut result.diagnostics)?;
        }
        // Every fractional addition is still made separately in timestamp and
        // selector order. Only integer diagnostics above use popcount.
        for lane in changed.set_lanes() {
            let timing = &mut self.timing.lanes[lane];
            for causes in timing.chunk_by(|a, b| a.output_pin == b.output_pin) {
                attribute_candidates(
                    causes,
                    &mut result.timing_causes,
                    &mut result
                        .diagnostics
                        .multiply_attributed_output_transition_count,
                );
            }
            timing.clear();
            let internal = &mut self.internal.lanes[lane];
            for events in internal.chunk_by(|a, b| a.owner_pin == b.owner_pin) {
                super::attribute_internal_candidates(
                    events,
                    &bindings.pins[events[0].owner_pin],
                    &mut self.rail_candidate_counts,
                    result,
                );
            }
            internal.clear();
        }
        Some(())
    }

    /// Selects timing causes only for output edges allowed by the arc and
    /// phase.
    fn select_timing(
        &mut self,
        owner_pin: &OwnerPin<'_>,
        output: SignalId,
        batch: &SampleBatch<'_>,
        library: &Library,
        signals: &[Signal],
        result: &mut ObservedActivity,
    ) -> Option<()> {
        let OwnerPin {
            instance_index,
            owner,
            layout,
            pin,
            output_changed,
        } = *owner_pin;
        let owner_values = self.pins[owner];
        let mut attributed = EMPTY;
        for (arc_index, (arc, binding)) in pin
            .timing_arcs
            .iter()
            .zip(&layout.bindings.pins[owner].timing_arcs)
            .enumerate()
        {
            let timing_type = arc.timing_type_str(library);
            let edge_arc = layout.is_sequential && is_sequential_edge_timing_type(timing_type);
            for output_rise in [false, true] {
                let allowed = owner_values.level(output_rise, output_changed)
                    & batch.timing_mask(timing_type, output_rise, layout.is_sequential);
                if allowed == EMPTY || transition_table(&arc.tables, output_rise).is_none() {
                    continue;
                }
                let related = *binding.related_pin.as_ref().ok()?;
                let ResolvedSource::Signal(source) = layout.pin_sources[related] else {
                    continue;
                };
                if !timing_source_drives_output(
                    signals[source.0].source,
                    signals[output.0].source,
                    edge_arc,
                ) {
                    continue;
                }
                let source_values = self.pins[related];
                for source_rise in [false, true] {
                    if !edge_arc
                        && !timing_sense_matches(
                            arc.timing_sense_str(library),
                            source_rise,
                            output_rise,
                        )
                    {
                        continue;
                    }
                    let requests =
                        source_values.level(source_rise, allowed & source_values.changed());
                    let mask = eligible(
                        requests,
                        binding.condition,
                        &self.conditions,
                        &mut result.diagnostics,
                    )?;
                    attributed |= mask;
                    self.timing.add(
                        mask,
                        TimingCauseKey {
                            output,
                            source,
                            source_rise,
                            output_rise,
                            instance: instance_index,
                            output_pin: owner,
                            arc: arc_index,
                        },
                    );
                }
            }
        }
        for rise in [false, true] {
            let missed = owner_values
                .level(rise, output_changed & !attributed)
                .popcount();
            if missed != 0 {
                *result.unattributed.entry((output, rise)).or_default() += missed;
                result.diagnostics.unattributed_output_transition_count += missed;
            }
        }
        Some(())
    }

    /// Requests each group's condition before inspecting its related-pin edges.
    fn select_internal(
        &mut self,
        owner_pin: &OwnerPin<'_>,
        changed: Vec256,
        diagnostics: &mut GvDynamicPowerDiagnostics,
    ) -> Option<()> {
        let OwnerPin {
            instance_index,
            owner,
            layout,
            pin,
            output_changed,
        } = *owner_pin;
        let owner_values = self.pins[owner];
        let requests = if pin.direction == PinDirection::Output as i32 {
            output_changed
        } else {
            changed
        };
        for (group, binding) in layout.bindings.pins[owner].power_groups.iter().enumerate() {
            let mask = eligible(requests, binding.condition, &self.conditions, diagnostics)?;
            if mask == EMPTY {
                continue;
            }
            for &related in binding.related_pins.as_ref().ok()? {
                let ResolvedSource::Signal(source) = layout.pin_sources[related] else {
                    continue;
                };
                let source_values = self.pins[related];
                for source_rise in [false, true] {
                    let events = source_values.level(source_rise, mask & source_values.changed());
                    if events == EMPTY {
                        continue;
                    }
                    let key = InternalEventKey {
                        instance: instance_index,
                        owner_pin: owner,
                        group,
                        source,
                        source_rise,
                        owner_output_rise: None,
                    };
                    self.internal.add(events & !output_changed, key);
                    for rise in [false, true] {
                        self.internal.add(
                            owner_values.level(rise, events & output_changed),
                            InternalEventKey {
                                owner_output_rise: Some(rise),
                                ..key
                            },
                        );
                    }
                }
            }
        }
        Some(())
    }
}

/// Attempts packed replay. Commit edges only after success, so a scalar retry
/// starts from the original state and reports the first error in trace order.
pub(super) fn observe(
    model: &PowerModelView<'_>,
    library: &Library,
    inputs: &[Vec<IrBits>],
    clocks: &[Option<bool>],
    kinds: &[PowerTransitionKind],
    topology: &mut PowerTopology,
) -> Option<ObservedActivity> {
    let mut result = ObservedActivity::default();
    let mut edges = vec![EdgeValue::<usize>::default(); topology.signals.len()];
    let mut scratch = CellScratch::new();
    let live = vec![true; model.gate_fn.gates.len()];
    let mut carry = vec![false; live.len()];
    let mut nodes = Vec::new();
    let mut gate_latches = GateLatchSamples::new(topology);
    for start in (0..inputs.len()).step_by(256) {
        let len = (inputs.len() - start).min(256);
        simulate_chunk(model.gate_fn, inputs, start, len, &live, &mut nodes);
        let mut valid = [0u64; 4];
        let mut active_edges = EdgeValue::<[u64; 4]>::default();
        // The first sample of the trace has no predecessor. Padding never
        // enters a request mask, including for complemented signals.
        for i in usize::from(start == 0)..len {
            let bit = 1 << (i % 64);
            valid[i / 64] |= bit;
            if let PowerTransitionKind::ActiveClockEdge { clock_rise } = kinds[start + i - 1] {
                let edge = if clock_rise {
                    &mut active_edges.rise
                } else {
                    &mut active_edges.fall
                };
                edge[i / 64] |= bit;
            }
        }
        let clock = clock_levels(&clocks[start..start + len]);
        gate_latches.capture(&nodes, clock).ok()?;
        let previous_clock = start
            .checked_sub(1)
            .and_then(|i| clocks[i])
            .unwrap_or(false);
        let batch = SampleBatch {
            nodes: &nodes,
            carry: &carry,
            valid: Vec256::from_words(valid),
            clock: clock.map(|levels| Samples::from_batch(levels, previous_clock)),
            gate_latches: &gate_latches,
            active_edges: EdgeValue {
                rise: Vec256::from_words(active_edges.rise),
                fall: Vec256::from_words(active_edges.fall),
            },
        };
        for (signal, count) in topology.signals.iter().zip(&mut edges) {
            let value = batch.signal(signal.source)?;
            let mask = batch.valid & value.changed();
            count.rise += value.level(true, mask).popcount();
            count.fall += value.level(false, mask).popcount();
        }
        for (index, instance) in model.instances.iter().enumerate() {
            scratch.observe_cell(&batch, index, instance, topology, library, &mut result)?;
        }
        for (bit, node) in carry.iter_mut().zip(&nodes) {
            *bit = node.get_lane(len - 1);
        }
    }
    for (signal, count) in topology.signals.iter_mut().zip(edges) {
        signal.edges.rise += count.rise;
        signal.edges.fall += count.fall;
    }
    Some(result)
}
