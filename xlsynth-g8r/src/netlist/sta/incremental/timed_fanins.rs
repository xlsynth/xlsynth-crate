// SPDX-License-Identifier: Apache-2.0

use super::{BitSource, IncrementalSta, PinDirection, StaTimingType, TimingArc, TimingEndpoint};
use smallvec::SmallVec;

impl IncrementalSta<'_> {
    /// Visits connected (instance, input pin, canonical source bit)
    /// predecessors of a capture or driven bit in one launch class. All
    /// potentially active timing arcs are included, even when their arrivals
    /// miss the envelope. Endpoint cones produced by this walk nominate edits
    /// for subsequent STA.
    pub(crate) fn visit_timed_fanins(
        &self,
        endpoint: TimingEndpoint,
        mode: usize,
        mut visit: impl FnMut(usize, &str, usize),
    ) {
        if mode >= self.launch_class_count() {
            return;
        }
        let (driver, output) = match endpoint {
            TimingEndpoint::Capture(i) if self.instances.get(i).is_some_and(|i| i.sequential) => {
                (i, None)
            }
            TimingEndpoint::Output(bit) => {
                let Some(BitSource::Bit(bit)) = self.canonical_source(bit) else {
                    return;
                };
                if self.timing[mode][bit].is_none() {
                    return;
                }
                let Some(i) = self.driver_for_bit(bit) else {
                    return;
                };
                if self.is_register(i) {
                    return;
                }
                let Some(pin) = self.output_pin_index(i, bit) else {
                    return;
                };
                (i, Some(pin))
            }
            TimingEndpoint::Capture(_) => return,
        };
        let inst = &self.instances[driver];
        let lib = self.lib.library;
        let cell = &lib.cells[self.cells[driver]];
        let possible = |arc: &TimingArc| {
            self.evaluation
                .when_formulas
                .get(&arc.when)
                .is_none_or(|when| when.evaluate(&inst.known) != Some(false))
        };
        let mut visit_input = |pin: usize| {
            let name = lib.resolve_string(&cell.pins[pin].name);
            if let Some([BitSource::Bit(bit)]) = inst.pins.get(name).map(Vec::as_slice)
                && self.timing[mode][*bit].is_some()
            {
                visit(driver, name, *bit);
            }
        };
        if let Some(output) = output {
            // Output-to-output arcs can pass through unwired sibling outputs.
            // Follow Liberty pin indices until the boundary is an actual input.
            let mut pending = SmallVec::<[usize; 4]>::from_slice(&[output]);
            let mut seen = SmallVec::<[usize; 8]>::from_slice(&[output]);
            while let Some(index) = pending.pop() {
                for arc in &cell.pins[index].timing_arcs {
                    if !StaTimingType::from_arc(arc, lib).is_combinational() || !possible(arc) {
                        continue;
                    }
                    for &name in self.lib.related_pin_names(arc.related_pin) {
                        let Some(pin) = self.lib.pin_index(self.cells[driver], name) else {
                            continue;
                        };
                        if seen.contains(&pin) {
                            continue;
                        }
                        seen.push(pin);
                        if cell.pins[pin].direction == PinDirection::Output as i32 {
                            pending.push(pin);
                        } else if cell.pins[pin].direction == PinDirection::Input as i32 {
                            visit_input(pin);
                        }
                    }
                }
            }
        } else {
            for (pin, data) in cell.pins.iter().enumerate() {
                if data.direction == PinDirection::Input as i32
                    && !data.is_clocking_pin
                    && data
                        .timing_arcs
                        .iter()
                        .any(|arc| StaTimingType::from_arc(arc, lib).is_setup() && possible(arc))
                {
                    visit_input(pin);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::io::read_gv_from_str;
    use crate::netlist::resize::tests::library;
    use crate::netlist::sta::{PreparedStaLibrary, StaOptions};
    use anyhow::Result;
    use std::sync::Arc;

    // Capture setup arcs define the active D pins. Walking through launch Q
    // into the logic that supplied its D would cross a second clock boundary.
    #[test]
    fn setup_checks_drive_capture_traversal_but_launch_register_outputs_stop_it() -> Result<()> {
        let lib = library()?;
        let parsed = read_gv_from_str(
            r#"module top(input clk, input a, input b, output q);
             wire launch, tied, mixed;
             DFF1 src(.CLK(clk), .D(a), .Q(launch));
             AND1 tied_gate(.A(launch), .B(launch), .Y(tied));
             AND1 mixed_gate(.A(tied), .B(b), .Y(mixed));
             DFF1 dest(.CLK(clk), .D(mixed), .Q(q));
             endmodule"#,
        )?;
        let sta = IncrementalSta::with_registered_library(
            &parsed.modules[0],
            &parsed.nets,
            &parsed.interner,
            Arc::new(PreparedStaLibrary::new(&lib)?),
            StaOptions::default(),
        )?;
        let fanins = |endpoint, mode| {
            let mut pins = Vec::new();
            sta.visit_timed_fanins(endpoint, mode, |i, pin, bit| {
                pins.push((
                    sta.instance(i).unwrap().name.to_owned(),
                    pin.to_owned(),
                    bit,
                ));
            });
            pins.sort();
            pins
        };
        let dest = (0..sta.instance_count())
            .find(|&i| sta.instance(i).unwrap().name == "dest")
            .unwrap();
        let capture = fanins(TimingEndpoint::Capture(dest), 1);
        assert_eq!(capture.len(), 1);
        assert_eq!((&capture[0].0[..], &capture[0].1[..]), ("dest", "D"));
        let and_inputs = fanins(TimingEndpoint::Output(capture[0].2), 1);
        assert_eq!(and_inputs.len(), 1);
        assert_eq!(
            (&and_inputs[0].0[..], &and_inputs[0].1[..]),
            ("mixed_gate", "A")
        );
        assert_eq!(fanins(TimingEndpoint::Output(capture[0].2), 0)[0].1, "B");
        let tied = fanins(TimingEndpoint::Output(and_inputs[0].2), 1);
        assert_eq!(
            tied.iter()
                .map(|(_, pin, _)| pin.as_str())
                .collect::<Vec<_>>(),
            ["A", "B"]
        );
        assert_eq!(tied[0].2, tied[1].2);
        assert!(fanins(TimingEndpoint::Output(tied[0].2), 1).is_empty());
        Ok(())
    }
}
