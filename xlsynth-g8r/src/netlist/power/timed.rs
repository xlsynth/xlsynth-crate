// SPDX-License-Identifier: Apache-2.0

//! Edge-driven simulation of combinational mapped cells with Liberty timing and
//! power. Times and slews are in Liberty time units. The model uses full-swing
//! threshold crossings and cancels an output crossing if its input reverses
//! before it is due.

mod cell;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use xlsynth_pir::IrValue;

use super::{
    AigNode, BoundConditionTerm, ConditionValueSlot, ConditionValues, Library, PinDirection,
    PowerModelView, PowerSignal, PowerTopology, ResolvedSource, SignalId, SignalOwner,
    WireLoadModel, build_topology, combinational_power_view, nominal_voltage, parse_formula,
    report_units,
};
use crate::netlist::gatefn_from_netlist::{LabeledNetlistAig, TimedLabeledNetlist};
use cell::{CellContext, PinEdge, PowerEvent};

/// One primary-input threshold crossing at a physical port bit.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TimedInputEvent {
    pub port: String,
    pub bit: u32,
    pub time: f64,
    pub value: bool,
    pub slew: f64,
}

/// Identifies the physical driver of a signal, including logic-equivalent
/// drivers.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TimedSignal {
    Input { port: String, bit: u32 },
    CellOutput { instance: String, pin: String },
}

/// A committed full-swing transition and its load switching energy.
#[derive(Clone, Debug, Serialize)]
pub struct TimedTransition {
    pub signal: TimedSignal,
    pub time: f64,
    pub value: bool,
    pub slew: f64,
    pub switching_energy: f64,
}

/// A cell-internal energy contribution and the owning pin's value at that time.
#[derive(Clone, Debug, Serialize)]
pub struct TimedInternalEnergy {
    pub instance: String,
    pub pin: String,
    pub time: f64,
    pub value: bool,
    pub energy: f64,
}

/// Chronological transition and internal-energy records over the half-open
/// window [0, duration).
#[derive(Clone, Debug, Serialize)]
pub struct TimedRun {
    pub duration: f64,
    pub time_unit: String,
    pub energy_unit: String,
    pub power_unit: String,
    pub events: Vec<TimedTransition>,
    pub internal_events: Vec<TimedInternalEnergy>,
    pub switching_energy: f64,
    pub internal_energy: f64,
    pub total_dynamic_energy: f64,
    pub average_dynamic_power: f64,
}

/// Maps a physical input to both the external stimulus tuple and its signal
/// trace.
struct InputBinding {
    signal: SignalId,
    argument: usize,
    bit_offset: usize,
}

/// A Liberty function bound to this instance's input pin slots.
struct OutputFunction {
    pin: usize,
    signal: SignalId,
    function: BoundConditionTerm,
}

/// A pending crossing is committed once its due time is reached, or replaced on
/// reversal.
struct PendingEdge {
    edge: Edge,
    causes: OutputCauses,
}

/// The causal input batch and output function evaluated over its possible
/// intermediate pin assignments. Row bits identify changes already applied.
struct OutputCauses {
    edges: Vec<PinEdge>,
    prior: Vec<bool>,
    truth: Vec<bool>,
    target: bool,
}

impl OutputCauses {
    /// Sets the changed inputs to their values on entry to a candidate edge.
    fn apply_row(&self, row: usize, pins: &mut [bool]) {
        for (index, edge) in self.edges.iter().enumerate() {
            pins[edge.pin] = if row & (1 << index) == 0 {
                self.prior[edge.pin]
            } else {
                edge.value
            };
        }
    }

    /// Finds predecessors where this input alone causes the requested edge.
    /// Single input changes and the wide-batch approximation use row zero.
    fn predecessors(&self, index: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.truth.len().max(1)).filter(move |&row| {
            self.truth.is_empty() || {
                let bit = 1 << index;
                row & bit == 0
                    && self.truth[row] != self.target
                    && self.truth[row | bit] == self.target
            }
        })
    }
}

/// A threshold crossing in one physical signal's chronological history.
#[derive(Clone, Copy)]
struct Edge {
    time: f64,
    round: usize,
    value: bool,
    slew: f64,
}

impl Edge {
    fn at_pin(self, pin: usize) -> PinEdge {
        PinEdge {
            pin,
            time: self.time,
            round: self.round,
            value: self.value,
            slew: self.slew,
        }
    }
}

/// Initial settled value and all committed edges within the observation window.
#[derive(Clone, Default)]
struct SignalTrace {
    initial: bool,
    edges: Vec<Edge>,
}

/// A combinational simulation plan reusable across independent stimulus runs.
pub struct TimedNetlistEvaluator<'a> {
    model: &'a LabeledNetlistAig,
    library: &'a Library,
    view: PowerModelView<'a>,
    topology: PowerTopology,
    inputs: HashMap<(&'a str, u32), InputBinding>,
    signal_names: Vec<TimedSignal>,
    outputs: Vec<Vec<OutputFunction>>,
    voltage_squared: f64,
}

impl<'a> TimedNetlistEvaluator<'a> {
    /// Prepares pin connectivity, loads and output functions once for this
    /// mapped design.
    pub fn new(
        netlist: &'a TimedLabeledNetlist,
        library: &'a Library,
        module_output_load: f64,
        wire_load: WireLoadModel,
    ) -> Result<Self, String> {
        let model = netlist.settled_model();
        if !module_output_load.is_finite() || module_output_load < 0.0 {
            return Err("module output load must be finite and nonnegative".into());
        }
        wire_load.validate().map_err(|e| e.to_string())?;
        let voltage_squared = nominal_voltage(library)?.powi(2);
        if !voltage_squared.is_finite() {
            return Err("nominal voltage squared is outside the finite range".into());
        }
        if library.units.as_ref().is_none_or(|units| {
            units.time_unit.trim().is_empty()
                || units.voltage_unit.trim().is_empty()
                || units.capacitance_unit.trim().is_empty()
        }) {
            return Err("timed power requires Liberty time, voltage and capacitance units".into());
        }
        let view = combinational_power_view(model);
        let topology = build_topology(
            &view,
            library,
            module_output_load,
            wire_load,
            &netlist.wire_aliases,
            super::DriverFallback::VerifiedWiring,
        )?;
        let arguments: HashMap<_, _> = model
            .gate_fn
            .inputs
            .iter()
            .enumerate()
            .map(|(index, input)| (input.name.as_str(), index))
            .collect();
        let mut inputs = HashMap::new();
        let mut signal_names = Vec::with_capacity(topology.signals.len());
        for (id, signal) in topology.signals.iter().enumerate() {
            signal_names.push(match signal.owner {
                SignalOwner::ModuleInput { port, bit } => {
                    let port = &model.module_ports[port];
                    let name = port.name.as_str();
                    let number = port.bits_lsb_to_msb[bit].bit_number;
                    let argument = *arguments
                        .get(name)
                        .ok_or_else(|| format!("no stimulus argument for input port '{name}'"))?;
                    inputs.insert(
                        (name, number),
                        InputBinding {
                            signal: SignalId(id),
                            argument,
                            bit_offset: bit,
                        },
                    );
                    TimedSignal::Input {
                        port: name.to_owned(),
                        bit: number,
                    }
                }
                SignalOwner::CellOutput { instance, pin } => {
                    let cell = &view.instances[instance];
                    TimedSignal::CellOutput {
                        instance: cell.instance_name.clone(),
                        pin: cell.pins[pin].pin_name.clone(),
                    }
                }
                SignalOwner::Clock => {
                    return Err("unexpected clock in combinational netlist".into());
                }
            });
        }
        let mut outputs = Vec::with_capacity(view.instances.len());
        for (instance, layout) in view.instances.iter().zip(&topology.instances) {
            let slots = instance
                .pins
                .iter()
                .enumerate()
                .filter(|(_, pin)| pin.direction == PinDirection::Input)
                .map(|(i, pin)| (pin.pin_name.as_str(), ConditionValueSlot::Pin(i)))
                .collect();
            // Output power uses the causal input batch and the committed
            // output. A group driven by another output requires joint output
            // characterization.
            for (owner, bindings) in layout.bindings.pins.iter().enumerate() {
                for group in &bindings.power_groups {
                    if group.related_pins.as_ref().is_ok_and(|related| {
                        related.iter().any(|&pin| {
                            pin != owner && instance.pins[pin].direction == PinDirection::Output
                        })
                    }) {
                        return Err(format!(
                            "timed power on {}.{} has an unsupported related output pin",
                            instance.instance_name, instance.pins[owner].pin_name
                        ));
                    }
                }
            }
            let mut functions = Vec::new();
            for (pin, signal) in layout.output_signals.iter().enumerate() {
                let Some(signal) = *signal else { continue };
                let lib_pin = &library.cells[layout.cell].pins[layout.liberty_pins[pin]];
                let term = parse_formula(library.resolve_string(&lib_pin.function))?;
                let function = BoundConditionTerm::bind(&term, &slots);
                if !function.is_fully_bound() {
                    return Err(format!(
                        "unbound input in function for {}.{}",
                        instance.instance_name, instance.pins[pin].pin_name
                    ));
                }
                functions.push(OutputFunction {
                    pin,
                    signal,
                    function,
                });
            }
            outputs.push(functions);
        }
        Ok(Self {
            model,
            library,
            view,
            topology,
            inputs,
            signal_names,
            outputs,
            voltage_squared,
        })
    }

    /// Evaluates a settled input tuple followed by timed changes, counting
    /// energy in [0, duration).
    pub fn simulate(
        &self,
        initial_inputs: &IrValue,
        input_events: &[TimedInputEvent],
        duration: f64,
    ) -> Result<TimedRun, String> {
        if !duration.is_finite() || duration <= 0.0 {
            return Err("duration must be finite and positive".into());
        }
        let initial = self.model.lower_arg_tuple(initial_inputs)?;
        let mut traces = vec![SignalTrace::default(); self.signal_names.len()];
        for binding in self.inputs.values() {
            traces[binding.signal.0].initial = initial[binding.argument]
                .get_bit(binding.bit_offset)
                .map_err(|e| e.to_string())?;
        }
        for event in input_events {
            if !event.time.is_finite() || event.time < 0.0 || event.time >= duration {
                return Err(format!(
                    "input time {} is outside [0, {duration})",
                    event.time
                ));
            }
            if !event.slew.is_finite() || event.slew <= 0.0 {
                return Err(format!(
                    "input slew must be finite and positive, got {}",
                    event.slew
                ));
            }
            let binding = self
                .inputs
                .get(&(event.port.as_str(), event.bit))
                .ok_or_else(|| format!("unknown input bit '{}[{}]'", event.port, event.bit))?;
            traces[binding.signal.0].edges.push(Edge {
                // Canonicalize -0.0 for total ordering of simultaneous inputs.
                time: event.time.abs(),
                round: 0,
                value: event.value,
                slew: event.slew,
            });
        }
        for trace in &mut traces {
            trace.edges.sort_by(|a, b| a.time.total_cmp(&b.time));
            if trace.edges.windows(2).any(|p| p[0].time == p[1].time) {
                return Err("multiple updates to the same input bit at one time".into());
            }
            let mut value = trace.initial;
            trace.edges.retain(|edge| {
                let changed = edge.value != value;
                value = edge.value;
                changed
            });
        }

        let mut internal_events = Vec::new();
        for &index in &self.topology.instance_order {
            self.simulate_instance(index, &mut traces, &mut internal_events, duration)?;
        }
        let mut events = Vec::new();
        for (id, trace) in traces.into_iter().enumerate() {
            for edge in trace.edges {
                events.push(TimedTransition {
                    signal: self.signal_names[id].clone(),
                    time: edge.time,
                    value: edge.value,
                    slew: edge.slew,
                    switching_energy: 0.5
                        * self.load(SignalId(id), edge.value)
                        * self.voltage_squared,
                });
            }
        }
        events.sort_by(|a, b| {
            a.time
                .total_cmp(&b.time)
                .then_with(|| a.signal.cmp(&b.signal))
        });
        internal_events.sort_by(|a, b| {
            a.time
                .total_cmp(&b.time)
                .then_with(|| (&a.instance, &a.pin).cmp(&(&b.instance, &b.pin)))
        });
        let switching_energy: f64 = events.iter().map(|e| e.switching_energy).sum();
        let internal_energy: f64 = internal_events.iter().map(|e| e.energy).sum();
        let total_dynamic_energy = switching_energy + internal_energy;
        let average_dynamic_power = total_dynamic_energy / duration;
        if !average_dynamic_power.is_finite() || !total_dynamic_energy.is_finite() {
            return Err("nonfinite energy or average power for timed run".into());
        }
        let (time_unit, _, _, energy_unit, power_unit) = report_units(self.library, Some(duration));
        Ok(TimedRun {
            duration,
            time_unit,
            energy_unit,
            power_unit: power_unit.expect("duration supplied for timed run"),
            events,
            internal_events,
            switching_energy,
            internal_energy,
            total_dynamic_energy,
            average_dynamic_power,
        })
    }

    fn load(&self, id: SignalId, rise: bool) -> f64 {
        let load = self.topology.signals[id.0].load;
        if rise { load.rise } else { load.fall }
    }

    /// Merges all input traces of one physical cell, grouping simultaneous pin
    /// changes.
    fn simulate_instance(
        &self,
        index: usize,
        traces: &mut [SignalTrace],
        internal_events: &mut Vec<TimedInternalEnergy>,
        duration: f64,
    ) -> Result<(), String> {
        let instance = &self.view.instances[index];
        let layout = &self.topology.instances[index];
        let outputs = &self.outputs[index];
        let cell = CellContext {
            library: self.library,
            instance,
            topology: layout,
        };
        let mut pins = vec![false; instance.pins.len()];
        let mut arrivals = Vec::new();
        let mut input_energy_owners = Vec::new();
        for (pin, source) in layout.pin_sources.iter().enumerate() {
            if instance.pins[pin].direction != PinDirection::Input {
                continue;
            }
            if !layout.bindings.pins[pin].power_groups.is_empty() {
                input_energy_owners.push(pin);
            }
            match *source {
                ResolvedSource::Signal(id) => {
                    pins[pin] = traces[id.0].initial;
                    arrivals.extend(traces[id.0].edges.iter().map(|edge| edge.at_pin(pin)));
                }
                ResolvedSource::Literal => {
                    let PowerSignal::Operand(operand) = instance.pins[pin].signal else {
                        return Err("unexpected sequential pin in combinational evaluation".into());
                    };
                    let AigNode::Literal { value, .. } = self.model.gate_fn.get(operand.node)
                    else {
                        return Err(format!(
                            "unresolved input {}.{}",
                            instance.instance_name, instance.pins[pin].pin_name
                        ));
                    };
                    pins[pin] = *value ^ operand.negated;
                }
            }
        }
        for output in outputs {
            pins[output.pin] = output.evaluate(&pins)?;
            traces[output.signal.0].initial = pins[output.pin];
        }

        arrivals.sort_by(|a, b| {
            a.time
                .total_cmp(&b.time)
                .then(a.round.cmp(&b.round))
                .then(a.pin.cmp(&b.pin))
        });
        let mut pending: Vec<Option<PendingEdge>> = outputs.iter().map(|_| None).collect();
        // An empty final batch flushes crossings strictly inside the
        // observation window.
        let batches = arrivals
            .chunk_by(|a, b| a.time == b.time && a.round == b.round)
            .chain([&[][..]]);
        for inputs in batches {
            let time = inputs.first().map_or(duration, |e| e.time);
            let round = inputs.first().map_or(0, |e| e.round);
            for (out, slot) in outputs.iter().zip(&mut pending) {
                let Some(event) = slot.take_if(|p| {
                    p.edge.time < duration && (p.edge.time, p.edge.round) <= (time, round)
                }) else {
                    continue;
                };
                let edge = event.edge;
                let energy = cell.internal_energy(
                    out.pin,
                    PowerEvent::Output {
                        edge: edge.at_pin(out.pin),
                        causes: &event.causes,
                    },
                    self.load(out.signal, edge.value),
                )?;
                record_internal(
                    &cell,
                    out.pin,
                    edge.value,
                    edge.time,
                    energy,
                    internal_events,
                );
                traces[out.signal.0].edges.push(edge);
                pins[out.pin] = edge.value;
            }
            if inputs.is_empty() {
                break;
            }
            let prior = pins.clone();
            for input in inputs {
                pins[input.pin] = input.value;
            }
            for &owner in &input_energy_owners {
                let energy = cell.internal_energy(
                    owner,
                    PowerEvent::Inputs {
                        edges: inputs,
                        prior: &prior,
                    },
                    0.0,
                )?;
                record_internal(&cell, owner, pins[owner], time, energy, internal_events);
            }
            for (out, slot) in outputs.iter().zip(&mut pending) {
                let value = out.evaluate(&pins)?;
                let planned = slot.as_ref().map_or(pins[out.pin], |p| p.edge.value);
                if value == planned {
                    continue;
                }
                *slot = None;
                if value == pins[out.pin] {
                    continue;
                }
                let causes = out.causes(&prior, inputs, value)?;
                let response =
                    cell.response(out.pin, value, &causes, self.load(out.signal, value))?;
                let due = time + response.delay;
                if !due.is_finite() {
                    return Err("nonfinite output arrival time".into());
                }
                // Independent crossings share round zero. A zero-delay
                // reversal of an already committed edge at this time follows
                // it in the next round, including at every downstream pin.
                let output_round = if due == time {
                    traces[out.signal.0]
                        .edges
                        .last()
                        .filter(|edge| edge.time == time)
                        .map_or(round, |edge| round.max(edge.round + 1))
                } else {
                    0
                };
                *slot = Some(PendingEdge {
                    edge: Edge {
                        time: due,
                        round: output_round,
                        value,
                        slew: response.slew,
                    },
                    causes,
                });
            }
        }
        Ok(())
    }
}

impl OutputFunction {
    /// A pin can cause this output edge if its change produces the edge for
    /// some ordering of the other simultaneous changes. Hold every unchanged
    /// pin fixed.
    fn causes(
        &self,
        prior: &[bool],
        edges: &[PinEdge],
        value: bool,
    ) -> Result<OutputCauses, String> {
        let mut causes = OutputCauses {
            prior: prior.to_vec(),
            edges: edges.to_vec(),
            truth: Vec::new(),
            target: value,
        };
        // A single change has exactly one predecessor. For more than eight
        // simultaneous changes, approximate using every changed pin and the
        // batch's original assignment for selectors, bounding exponential work.
        if (2..=8).contains(&edges.len()) {
            let mut pins = prior.to_vec();
            let row_count = 1 << edges.len();
            causes.truth.reserve_exact(row_count);
            for row in 0..row_count {
                causes.apply_row(row, &mut pins);
                causes.truth.push(self.evaluate(&pins)?);
            }
        }
        Ok(causes)
    }

    fn evaluate(&self, pins: &[bool]) -> Result<bool, String> {
        self.function
            .evaluate_partial(ConditionValues {
                pins,
                state_aliases: &[],
            })
            .ok_or_else(|| "could not evaluate mapped output function".into())
    }
}

/// Records a nonzero contribution under its cell and pin, retaining signed
/// Liberty values.
fn record_internal(
    cell: &CellContext<'_>,
    pin: usize,
    value: bool,
    time: f64,
    energy: f64,
    result: &mut Vec<TimedInternalEnergy>,
) {
    if energy != 0.0 {
        result.push(TimedInternalEnergy {
            instance: cell.instance.instance_name.clone(),
            pin: cell.instance.pins[pin].pin_name.clone(),
            time,
            value,
            energy,
        });
    }
}

// A masked B transition must not win a delay lookup over the sensitized A->Y
// arc.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simultaneous_masked_pin_is_excluded_from_output_causes() {
        let slots = HashMap::from([
            ("A", ConditionValueSlot::Pin(0)),
            ("B", ConditionValueSlot::Pin(1)),
            ("C", ConditionValueSlot::Pin(2)),
        ]);
        let out = OutputFunction {
            pin: 3,
            signal: SignalId(0),
            function: BoundConditionTerm::bind(&parse_formula("A | (B & C)").unwrap(), &slots),
        };
        let rise = Edge {
            time: 0.0,
            round: 0,
            value: true,
            slew: 1.0,
        };
        let edges = [rise.at_pin(0), rise.at_pin(1)];
        let causes = out
            .causes(&[false, false, false, false], &edges, true)
            .unwrap();
        assert!(causes.predecessors(0).next().is_some());
        assert!(causes.predecessors(1).next().is_none());
    }
}
