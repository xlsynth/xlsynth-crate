// SPDX-License-Identifier: Apache-2.0

//! Lower explicit register controls into effective-D combinational logic.

use std::collections::BTreeSet;

use crate::aig::dce::dce_safe;
use crate::aig::import::import_gate_fn;
use crate::aig::sequential_gate::{
    RegisterBinding, RegisterControl, SequentialGateFn, TransitionOutputId,
    canonical_register_d_name, uniquify_transition_port_name,
};
use crate::aig::{AigBitVector, AigOperand, GateFn};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};

impl SequentialGateFn {
    /// Lowers controls to effective-D logic without changing the state space.
    ///
    /// For the same arbitrary initial register values, every cycle (including
    /// cycle zero) is preserved. This does not remove or merge registers.
    /// When controls are present, only external outputs and effective D
    /// endpoints remain in the interface. Already-legalized graphs are
    /// returned unchanged.
    pub fn legalize_register_controls(&self) -> Result<SequentialGateFn, String> {
        self.validate()?;
        // Already-legalized designs retain their exact graph and interface.
        if !self.has_register_controls() {
            return Ok(self.clone());
        }
        let mut builder = GateBuilder::new(self.transition.name.clone(), GateBuilderOptions::opt());
        let bindings = self
            .transition
            .inputs
            .iter()
            .map(|input| builder.add_input(input.name.clone(), input.get_bit_count()))
            .collect::<Vec<_>>();
        let values = import_gate_fn(&self.transition, &bindings, &mut builder)?;
        let mut output_names = BTreeSet::new();
        let mut outputs = Vec::with_capacity(self.outputs.len());
        for id in &self.outputs {
            let output = &self.transition.outputs[id.index()];
            output_names.insert(output.name.clone());
            outputs.push(TransitionOutputId::new(builder.outputs.len()));
            builder.add_output(output.name.clone(), values[id.index()].clone());
        }
        let mut registers = Vec::with_capacity(self.registers.len());
        for register in &self.registers {
            let mut data = values[register.d.index()].clone();
            if let Some(enable) = register.enable {
                data = builder.add_mux2_vec(
                    &asserted_control(enable, &values),
                    &data,
                    &bindings[register.q.index()],
                );
            }
            if let Some(reset) = &register.reset {
                let reset_value = builder.add_literal(&reset.value);
                data = builder.add_mux2_vec(
                    &asserted_control(reset.control, &values),
                    &reset_value,
                    &data,
                );
            }
            let name = uniquify_transition_port_name(
                &canonical_register_d_name(&register.name),
                &mut output_names,
            );
            let d = TransitionOutputId::new(builder.outputs.len());
            builder.add_output(name, data);
            registers.push(RegisterBinding {
                enable: None,
                reset: None,
                name: register.name.clone(),
                q: register.q,
                d,
            });
        }
        // GateBuilder::build requires an output; a zero-output combinational
        // design is also valid at this interface.
        let transition = GateFn {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            gates: builder.gates,
        };
        SequentialGateFn::new(
            self.name.clone(),
            dce_safe(&transition),
            self.inputs.clone(),
            outputs,
            self.clock.clone(),
            registers,
        )
    }
}

/// Returns the active-high interpretation of a validated control endpoint.
fn asserted_control(control: RegisterControl, values: &[AigBitVector]) -> AigOperand {
    let bit = *values[control.signal.index()].get_lsb(0);
    if control.active_low {
        bit.negate()
    } else {
        bit
    }
}
