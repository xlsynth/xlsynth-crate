// SPDX-License-Identifier: Apache-2.0

//! Compute required times backward from endpoints for one launch/capture
//! class, using the current circuit's Liberty pin slews and loads. Use them
//! to choose candidate edits, then rerun forward timing before accepting a
//! changed circuit.

use super::*;
use crate::netlist::sta::{
    StaTimingSense, StaTimingTableKind, arc_when_may_apply, evaluate_table_with_diagnostics,
    find_unique_table,
};

/// Smallest required time and least slack over observed rise and fall arcs.
/// The connection ordinal is the position in the mapped instance's pin list.
/// Paths outside the selected boundary or without a timed arc have no value.
pub(crate) struct PinTimingUrgency {
    pins: Vec<SmallVec<[[f64; 2]; 4]>>,
}

impl PinTimingUrgency {
    pub(crate) fn required_arrival(&self, instance: usize, connection: usize) -> Option<f64> {
        let value = self.pins.get(instance)?.get(connection)?[0];
        value.is_finite().then_some(value)
    }

    pub(crate) fn sink_slack(&self, instance: usize, connection: usize) -> Option<f64> {
        let value = self.pins.get(instance)?.get(connection)?[1];
        value.is_finite().then_some(value)
    }

    fn observe(&mut self, i: usize, pin: usize, required: f64, arrival: f64) {
        let entry = &mut self.pins[i][pin];
        entry[0] = entry[0].min(required);
        entry[1] = entry[1].min(required - arrival);
    }
}

impl IncrementalSta<'_> {
    /// Uses the worst endpoint arrival as a common reference. A clock budget
    /// would move all slacks by the same constant within this boundary.
    pub(crate) fn pin_timing_urgency(&self, boundary: usize) -> Result<PinTimingUrgency> {
        ensure!(boundary < 4, "unknown timing boundary {boundary}");
        let mode = boundary / 2;
        let mut result = PinTimingUrgency {
            pins: self
                .instances
                .iter()
                .map(|i| std::iter::repeat_n([f64::INFINITY; 2], i.pin_order.len()).collect())
                .collect(),
        };
        let Some((_, reference)) = self.endpoint_arrivals(boundary).next_back() else {
            return Ok(result);
        };
        let library = self.lib.library;
        let mut eval = TimingEvaluationContext::new(library);
        let mut required = vec![[f64::INFINITY; 2]; self.bit_count()];

        for (endpoint, _) in self.endpoint_arrivals(boundary) {
            match endpoint {
                TimingEndpoint::Output(bit) => required[bit] = [reference; 2],
                TimingEndpoint::Capture(i) => self.seed_capture_bounds(
                    i,
                    mode,
                    reference,
                    &mut required,
                    &mut result,
                    &mut eval,
                )?,
            }
        }
        for &i in self.topological_instances().iter().rev() {
            self.propagate_instance_bounds(i, mode, &mut required, &mut result, &mut eval)?;
        }
        Ok(result)
    }

    fn seed_capture_bounds(
        &self,
        i: usize,
        mode: usize,
        reference: f64,
        required: &mut [[f64; 2]],
        result: &mut PinTimingUrgency,
        eval: &mut TimingEvaluationContext<'_>,
    ) -> Result<()> {
        let library = self.lib.library;
        let inst = &self.instances[i];
        for (pos, name) in inst.pin_order.iter().enumerate() {
            let pin = self.lib.pin(self.cells[i], name).expect("validated pin");
            if pin.direction != PinDirection::Input as i32 || pin.is_clocking_pin {
                continue;
            }
            let [BitSource::Bit(bit)] = inst.pins[name].as_slice() else {
                continue;
            };
            let Some(data) = self.timing[mode][*bit].as_ref() else {
                continue;
            };
            let arcs: SmallVec<[&TimingArc; 4]> = pin
                .timing_arcs
                .iter()
                .filter(|a| StaTimingType::from_arc(a, library).is_setup())
                .collect();
            if arcs.is_empty() {
                continue;
            }
            for (edge, edges) in [&data.rise, &data.fall].into_iter().enumerate() {
                let Some(latest) = edges.max_arrival_edge() else {
                    continue;
                };
                let mut isolated = SignalTimingSet::default();
                if edge == 0 {
                    isolated.rise = edges.clone()
                } else {
                    isolated.fall = edges.clone()
                }
                if let Some(capture) = evaluate_register_setup_capture_arrival(
                    library,
                    &arcs,
                    &inst.known,
                    &isolated,
                    eval,
                    name,
                )? {
                    // Back out setup time to obtain a deadline at D. This is a
                    // ranking bound at the incumbent slew. A trial can change
                    // slew and must redo setup STA.
                    let deadline = reference - (capture.arrival - latest.arrival);
                    required[*bit][edge] = required[*bit][edge].min(deadline);
                    result.observe(i, pos, deadline, latest.arrival);
                }
            }
        }
        Ok(())
    }

    fn propagate_instance_bounds(
        &self,
        i: usize,
        mode: usize,
        required: &mut [[f64; 2]],
        result: &mut PinTimingUrgency,
        eval: &mut TimingEvaluationContext<'_>,
    ) -> Result<()> {
        let library = self.lib.library;
        let inst = &self.instances[i];
        let cell_idx = self.cells[i];
        let cell = &library.cells[cell_idx];
        if !inst
            .outputs
            .iter()
            .any(|&bit| required[bit].iter().any(|b| b.is_finite()))
        {
            return Ok(());
        }
        let hidden = (self.orders[i].len() > inst.outputs.len())
            .then(|| self.internal_pin_timing(i, mode, eval))
            .transpose()?;
        let mut internal_bounds = hidden
            .as_ref()
            .map(|pins| vec![[f64::INFINITY; 2]; pins.len()]);
        let wire = |name: &str| match inst.pins.get(name).map(Vec::as_slice) {
            Some([BitSource::Bit(bit)]) => Some(*bit),
            _ => None,
        };
        let timing = |bit: Option<usize>, pin: usize| match bit {
            Some(bit) => self.timing[mode][bit].as_ref(),
            None => hidden.as_ref().and_then(|pins| pins[pin].as_ref()),
        };
        // A related output can feed another output of the same standard
        // cell. Mirror the forward order in reverse to propagate its bound.
        for &out_idx in self.orders[i].iter().rev() {
            let out = &cell.pins[out_idx];
            let out_name = library.resolve_string(&out.name);
            let dst = wire(out_name);
            let bounds = match dst {
                Some(bit) => required[bit],
                None => internal_bounds.as_ref().expect("internal output")[out_idx],
            };
            if bounds.iter().all(|t| !t.is_finite()) {
                continue;
            }
            let Some(data) = timing(dst, out_idx) else {
                continue;
            };
            if let Some(pos) = inst.pin_order.iter().position(|n| n == out_name) {
                for (edge, set) in [&data.rise, &data.fall].into_iter().enumerate() {
                    if bounds[edge].is_finite()
                        && let Some(a) = set.max_arrival_edge()
                    {
                        result.observe(i, pos, bounds[edge], a.arrival);
                    }
                }
            }
            if inst.sequential {
                continue;
            }
            let load = dst.map(|bit| self.capacitances[bit]).unwrap_or_default();
            for arc in &out.timing_arcs {
                if !arc_when_may_apply(library, arc, &inst.known, eval, out_name)? {
                    continue;
                }
                for &name in self.lib.related_pin_names(arc.related_pin) {
                    let src_idx = self
                        .lib
                        .pin_index(cell_idx, name)
                        .expect("validated timing pin");
                    let src = wire(name);
                    if src.is_none() && cell.pins[src_idx].direction != PinDirection::Output as i32
                    {
                        continue;
                    }
                    let Some(data) = timing(src, src_idx) else {
                        continue;
                    };
                    let pos = inst.pin_order.iter().position(|n| n == name);
                    let net_bounds = match src {
                        Some(bit) => &mut required[bit],
                        None => {
                            &mut internal_bounds.as_mut().expect("internal timing pin")[src_idx]
                        }
                    };
                    let branch = propagate_arc(library, arc, data, load, bounds, eval, out_name)?;
                    for (edge, t) in branch.into_iter().enumerate() {
                        net_bounds[edge] = net_bounds[edge].min(t);
                        if t.is_finite()
                            && let Some(pin) = pos
                        {
                            let set = if edge == 0 { &data.rise } else { &data.fall };
                            if let Some(a) = set.max_arrival_edge() {
                                result.observe(i, pin, t, a.arrival);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// If a Liberty output feeds another output without a physical wire, run
    /// the ordinary forward kernel with one scratch bit per cell pin. This
    /// recovers its slew using the same clamped tables and output loads.
    fn internal_pin_timing(
        &self,
        i: usize,
        mode: usize,
        eval: &mut TimingEvaluationContext<'_>,
    ) -> Result<Vec<Option<SignalTimingSet>>> {
        let inst = &self.instances[i];
        let cell_idx = self.cells[i];
        let cell = &self.lib.library.cells[cell_idx];
        let mut times = vec![None; cell.pins.len()];
        let mut loads = vec![EdgeLoadCapacitance::default(); cell.pins.len()];
        let mut bindings = HashMap::new();
        for (idx, pin) in cell.pins.iter().enumerate() {
            let name = self.lib.library.resolve_string(&pin.name);
            let source = match inst.pins.get(name).and_then(|sources| sources.first()) {
                Some(BitSource::Bit(bit)) => {
                    times[idx] = self.timing[mode][*bit].clone();
                    loads[idx] = self.capacitances[*bit];
                    BitSource::Bit(idx)
                }
                _ if pin.direction == PinDirection::Output as i32 => BitSource::Bit(idx),
                Some(other) => *other,
                None => continue,
            };
            bindings.insert(name.to_owned(), vec![source]);
        }
        evaluate_instance_outputs(
            InstanceTimingInputs {
                lib: &self.lib,
                cell_idx,
                inst_pin_map: &bindings,
                known_pin_values: &inst.known,
                output_pin_order: &self.orders[i],
                instance_name: &inst.name,
                render_bit: &|b| {
                    self.lib
                        .library
                        .resolve_string(&cell.pins[b].name)
                        .to_owned()
                },
                register_boundaries: self.registered,
                is_sequential: false,
                launch_register: false,
            },
            &loads,
            &mut times,
            &self.literal,
            eval,
        )?;
        Ok(times)
    }
}

/// Pulls each output deadline through one characterized arc, retaining
/// separate source-edge bounds across inverting and non-unate arcs.
fn propagate_arc(
    library: &Library,
    arc: &TimingArc,
    data: &SignalTimingSet,
    load: EdgeLoadCapacitance,
    bounds: [f64; 2],
    eval: &mut TimingEvaluationContext<'_>,
    name: &str,
) -> Result<[f64; 2]> {
    let kind = StaTimingType::from_arc(arc, library);
    let sense = StaTimingSense::from_arc(arc, library);
    let mut input_bounds = [f64::INFINITY; 2];
    for (out_edge, table_kind, cap, active) in [
        (
            0,
            StaTimingTableKind::CellRise,
            load.rise,
            kind.produces_rise(),
        ),
        (
            1,
            StaTimingTableKind::CellFall,
            load.fall,
            kind.produces_fall(),
        ),
    ] {
        if !active || !bounds[out_edge].is_finite() {
            continue;
        }
        let table = find_unique_table(arc, table_kind, name)?;
        for (in_edge, timings) in [&data.rise, &data.fall].into_iter().enumerate() {
            let coupled = match sense {
                StaTimingSense::PositiveUnate => in_edge == out_edge,
                StaTimingSense::NegativeUnate => in_edge != out_edge,
                other => other.may_use_either_input_edge(),
            };
            if !coupled {
                continue;
            }
            for arrival in timings.iter().map(|v| v.timing) {
                let delay = evaluate_table_with_diagnostics(
                    library,
                    table,
                    arrival.transition,
                    cap,
                    eval,
                    name,
                )?;
                let deadline = bounds[out_edge] - delay;
                input_bounds[in_edge] = input_bounds[in_edge].min(deadline);
            }
        }
    }
    Ok(input_bounds)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty::model::LibraryBuilder;
    use crate::netlist::io::read_gv_from_str;
    use crate::netlist::resize::tests::library;

    #[test]
    fn setup_slack_reaches_launch_q_through_an_unwired_output() -> Result<()> {
        let mut b = LibraryBuilder::from_library(library()?);
        let y = b.intern_string("Y")?;
        let yn = b.intern_string("YN")?;
        let cell = b.cells.iter_mut().find(|c| c.name == "DUAL1").unwrap();
        cell.pins
            .iter_mut()
            .find(|p| p.name == yn)
            .unwrap()
            .timing_arcs[0]
            .related_pin = y;
        let lib = b.finish();
        let p = read_gv_from_str(
            "module m(input clk,input a,output y,output z);
             wire q,n;
             DFF1 src(.CLK(clk),.D(a),.Q(q));
             DUAL1 dual(.A(q),.YN(n));
             DFF1 slow(.CLK(clk),.D(n),.Q(y));
             DFF1 fast(.CLK(clk),.D(q),.Q(z));
             endmodule",
        )?;
        let sta = IncrementalSta::new(
            &p.modules[0],
            &p.nets,
            &p.interner,
            &lib,
            StaOptions::default(),
        )?;
        let urgency = sta.pin_timing_urgency(3)?;
        for (i, pin) in [(0, 2), (1, 0), (2, 1)] {
            assert!(urgency.sink_slack(i, pin).unwrap().abs() < 1e-9);
        }
        assert!(urgency.sink_slack(3, 1).unwrap() > 0.0);
        assert!(urgency.sink_slack(0, 1).is_none()); // PI-to-register class.
        assert!(urgency.sink_slack(2, 0).is_none()); // Capture clock.
        Ok(())
    }
}
