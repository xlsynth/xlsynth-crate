// SPDX-License-Identifier: Apache-2.0

//! Wiring of verified, low-transparent Liberty ICGs in a single clock domain.

use super::{
    ResolvedBitValues, SequentialAigSignal, SequentialInstanceAigBinding,
    build_sequential_instance_aig_binding,
};
use crate::aig::gate::AigOperand;
use crate::gate_builder::GateBuilder;
use crate::liberty_model::{Cell, Library};
use crate::liberty_proto::ClockGate;
use crate::netlist::normalized::{
    BitIndex, BitSource, NormalizedInstance, NormalizedNetlistModule,
};
use crate::netlist::parse::Net;
use std::collections::HashMap;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// One verified ICG's physical connections, before data signals are projected.
pub(super) struct GateInstance<'a> {
    pub cell: &'a Cell,
    pub spec: &'a ClockGate,
    pub clock: BitSource,
    pub output: BitIndex,
    enable: BitSource,
    test: Option<BitSource>,
}

impl<'a> GateInstance<'a> {
    /// Requires every functional clock-gate pin to have a scalar connection.
    pub fn new(
        cell: &'a Cell,
        spec: &'a ClockGate,
        inst: &NormalizedInstance,
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> Result<Self, String> {
        let instance_name = interner.resolve(inst.instance_name).unwrap_or("<unknown>");
        let pin_source = |pin_name: &str| -> Result<BitSource, String> {
            let bits = inst
                .connections
                .iter()
                .find(|c| interner.resolve(c.port) == Some(pin_name))
                .map(|c| c.bits.as_slice());
            match bits {
                Some([source]) if *source != BitSource::Unknown => Ok(*source),
                _ => Err(format!(
                    "clock-gate instance '{instance_name}' pin '{pin_name}' must connect exactly one known bit"
                )),
            }
        };
        let BitSource::Bit(output) = pin_source(&spec.output_pin)? else {
            return Err(format!(
                "clock-gate instance '{instance_name}' output must drive a net bit"
            ));
        };
        Ok(Self {
            cell,
            spec,
            clock: pin_source(&spec.clock_pin)?,
            output,
            enable: pin_source(&spec.enable_pins[0])?,
            test: spec
                .test_pins
                .first()
                .map(|name| pin_source(name))
                .transpose()?,
        })
    }

    /// Binds pre-edge enable and clock labels after the ordinary data cone has
    /// settled.
    pub fn bind(
        &self,
        inst: &NormalizedInstance,
        normalized: &NormalizedNetlistModule<'_>,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        library: &Library,
        resolved: &ResolvedBitValues,
        gb: &mut GateBuilder,
    ) -> Result<(AigOperand, SequentialInstanceAigBinding), String> {
        let instance_name = interner.resolve(inst.instance_name).unwrap_or("<unknown>");
        let resolve = |source, gb: &mut GateBuilder| {
            resolved.resolve_source(source, gb)?.ok_or_else(|| {
                format!(
                    "clock-gate instance '{instance_name}' has an unresolved enable or test input"
                )
            })
        };
        let enable = resolve(self.enable, gb)?;
        let mut inputs = HashMap::from([(self.spec.enable_pins[0].clone(), enable)]);
        if let Some(test) = self.test {
            let test = resolve(test, gb)?;
            if test != gb.get_false() {
                return Err(format!(
                    "clock-gate instance '{instance_name}' test pin '{}' must be tied inactive",
                    self.spec.test_pins[0]
                ));
            }
            inputs.insert(self.spec.test_pins[0].clone(), test);
        }
        let clocks = [
            (self.spec.clock_pin.as_str(), SequentialAigSignal::Clock),
            (
                self.spec.output_pin.as_str(),
                SequentialAigSignal::GatedClock(enable),
            ),
        ];
        let binding = build_sequential_instance_aig_binding(
            inst,
            self.cell,
            library,
            &self.cell.name,
            instance_name,
            interner,
            nets,
            normalized,
            &inputs,
            &HashMap::new(),
            &clocks,
            None,
        )?;
        Ok((enable, binding))
    }
}
