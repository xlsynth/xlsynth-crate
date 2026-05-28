// SPDX-License-Identifier: Apache-2.0

//! Sequential design metadata around a combinational one-cycle transition
//! function.

use std::collections::BTreeSet;

use xlsynth::IrBits;

use crate::aig::gate::{GateFn, Input, Output};

/// Identifies an input of a [`SequentialGateFn::transition`] function.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct TransitionInputId(usize);

impl TransitionInputId {
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    pub fn index(self) -> usize {
        self.0
    }
}

/// Identifies an output of a [`SequentialGateFn::transition`] function.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct TransitionOutputId(usize);

impl TransitionOutputId {
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    pub fn index(self) -> usize {
        self.0
    }
}

/// The single clock port of an XLS-compatible sequential design.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ClockPort {
    pub name: String,
}

/// Reset behavior of a register.
///
/// `signal` is an output of the transition function because reset may be a
/// combinational expression and must remain live during AIG optimization.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResetSpec {
    pub signal: TransitionOutputId,
    pub asynchronous: bool,
    pub active_low: bool,
    pub value: IrBits,
}

/// Binds one state element to the transition function interface.
///
/// Register `Q` is a source of combinational logic, represented as a
/// transition input. Register `D`, load enable, and reset are sequential-cell
/// inputs, represented as transition outputs.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RegisterBinding {
    pub name: String,
    pub q: TransitionInputId,
    pub d: TransitionOutputId,
    pub load_enable: Option<TransitionOutputId>,
    pub reset: Option<ResetSpec>,
    pub initial_value: Option<IrBits>,
}

/// A synchronous design represented as a combinational transition function
/// plus its sequential boundaries.
///
/// `transition.inputs` are classified as external `inputs` or register `Q`
/// values. `transition.outputs` are externally visible `outputs`, register
/// `D` values, or register control inputs. A design with no registers is valid
/// and may omit `clock`.
#[derive(Debug, Clone)]
pub struct SequentialGateFn {
    pub name: String,
    pub transition: GateFn,
    pub inputs: Vec<TransitionInputId>,
    pub outputs: Vec<TransitionOutputId>,
    pub clock: Option<ClockPort>,
    pub registers: Vec<RegisterBinding>,
}

impl SequentialGateFn {
    /// Wraps a combinational gate function in the canonical sequential design
    /// representation without adding state or clock metadata.
    pub fn from_gate_fn(gate_fn: GateFn) -> Self {
        let name = gate_fn.name.clone();
        let inputs = (0..gate_fn.inputs.len())
            .map(TransitionInputId::new)
            .collect();
        let outputs = (0..gate_fn.outputs.len())
            .map(TransitionOutputId::new)
            .collect();
        Self::new(name, gate_fn, inputs, outputs, None, vec![])
            .expect("identity bindings for a GateFn must form a valid SequentialGateFn")
    }

    /// Returns a combinational gate function when this design contains no
    /// sequential boundary metadata.
    pub fn try_into_gate_fn(self) -> Result<GateFn, String> {
        self.validate().map_err(|e| {
            format!(
                "cannot convert design '{}' to GateFn: invalid SequentialGateFn: {}",
                self.name, e
            )
        })?;
        match (&self.clock, self.registers.len()) {
            (Some(clock), register_count) if register_count != 0 => {
                return Err(format!(
                    "cannot convert design '{}' to GateFn: design contains {} register(s) and clock '{}'",
                    self.name, register_count, clock.name
                ));
            }
            (Some(clock), _) => {
                return Err(format!(
                    "cannot convert design '{}' to GateFn: design declares clock '{}'",
                    self.name, clock.name
                ));
            }
            (None, register_count) if register_count != 0 => {
                return Err(format!(
                    "cannot convert design '{}' to GateFn: design contains {} register(s)",
                    self.name, register_count
                ));
            }
            (None, _) => {}
        }

        let inputs = self
            .inputs
            .iter()
            .map(|id| self.transition.inputs[id.index()].clone())
            .collect();
        let outputs = self
            .outputs
            .iter()
            .map(|id| self.transition.outputs[id.index()].clone())
            .collect();
        let mut gate_fn = self.transition;
        gate_fn.name = self.name;
        gate_fn.inputs = inputs;
        gate_fn.outputs = outputs;
        Ok(gate_fn)
    }

    /// Constructs a sequential function after validating its interface
    /// bindings.
    pub fn new(
        name: String,
        transition: GateFn,
        inputs: Vec<TransitionInputId>,
        outputs: Vec<TransitionOutputId>,
        clock: Option<ClockPort>,
        registers: Vec<RegisterBinding>,
    ) -> Result<Self, String> {
        let result = Self {
            name,
            transition,
            inputs,
            outputs,
            clock,
            registers,
        };
        result.validate()?;
        Ok(result)
    }

    /// Validates sequential metadata and its bindings to the transition
    /// function.
    pub fn validate(&self) -> Result<(), String> {
        self.transition.check_invariants_with_debug_assert();

        if !self.registers.is_empty() && self.clock.is_none() {
            return Err("a SequentialGateFn with registers must have a clock".to_string());
        }

        let mut input_claims = vec![None; self.transition.inputs.len()];
        for id in &self.inputs {
            self.claim_transition_input(&mut input_claims, *id, "external input".to_string())?;
        }

        let mut output_is_bound = vec![false; self.transition.outputs.len()];
        let mut external_outputs = BTreeSet::new();
        for id in &self.outputs {
            self.transition_output(*id, "external output")?;
            if !external_outputs.insert(*id) {
                return Err(format!(
                    "transition output index {} is listed more than once as an external output",
                    id.index()
                ));
            }
            output_is_bound[id.index()] = true;
        }

        let mut register_names = BTreeSet::new();
        for register in &self.registers {
            if !register_names.insert(register.name.clone()) {
                return Err(format!("duplicate register name '{}'", register.name));
            }

            self.claim_transition_input(
                &mut input_claims,
                register.q,
                format!("register '{}' Q", register.name),
            )?;

            let q =
                self.transition_input(register.q, &format!("register '{}' Q", register.name))?;
            let d = self.bind_transition_output(
                &mut output_is_bound,
                register.d,
                &format!("register '{}' D", register.name),
            )?;
            let register_width = q.get_bit_count();
            if d.get_bit_count() != register_width {
                return Err(format!(
                    "register '{}' has Q width {} but D width {}",
                    register.name,
                    register_width,
                    d.get_bit_count()
                ));
            }

            if let Some(load_enable) = register.load_enable {
                let output = self.bind_transition_output(
                    &mut output_is_bound,
                    load_enable,
                    &format!("register '{}' load enable", register.name),
                )?;
                if output.get_bit_count() != 1 {
                    return Err(format!(
                        "register '{}' load enable must be bits[1], got bits[{}]",
                        register.name,
                        output.get_bit_count()
                    ));
                }
            }

            if let Some(reset) = &register.reset {
                let output = self.bind_transition_output(
                    &mut output_is_bound,
                    reset.signal,
                    &format!("register '{}' reset", register.name),
                )?;
                if output.get_bit_count() != 1 {
                    return Err(format!(
                        "register '{}' reset signal must be bits[1], got bits[{}]",
                        register.name,
                        output.get_bit_count()
                    ));
                }
                if reset.value.get_bit_count() != register_width {
                    return Err(format!(
                        "register '{}' reset value has width {} but register width is {}",
                        register.name,
                        reset.value.get_bit_count(),
                        register_width
                    ));
                }
            }

            if let Some(initial_value) = &register.initial_value
                && initial_value.get_bit_count() != register_width
            {
                return Err(format!(
                    "register '{}' initial value has width {} but register width is {}",
                    register.name,
                    initial_value.get_bit_count(),
                    register_width
                ));
            }
        }

        for (index, claim) in input_claims.iter().enumerate() {
            if claim.is_none() {
                return Err(format!(
                    "transition input '{}' at index {} is not bound as an external input or register Q",
                    self.transition.inputs[index].name, index
                ));
            }
        }
        for (index, is_bound) in output_is_bound.iter().enumerate() {
            if !is_bound {
                return Err(format!(
                    "transition output '{}' at index {} is not bound as an external output or register input",
                    self.transition.outputs[index].name, index
                ));
            }
        }

        Ok(())
    }

    fn claim_transition_input(
        &self,
        claims: &mut [Option<String>],
        id: TransitionInputId,
        claim: String,
    ) -> Result<(), String> {
        self.transition_input(id, &claim)?;
        if let Some(previous) = &claims[id.index()] {
            return Err(format!(
                "transition input index {} is bound both as {} and {}",
                id.index(),
                previous,
                claim
            ));
        }
        claims[id.index()] = Some(claim);
        Ok(())
    }

    fn transition_input(&self, id: TransitionInputId, context: &str) -> Result<&Input, String> {
        self.transition.inputs.get(id.index()).ok_or_else(|| {
            format!(
                "{} references transition input index {}, but only {} inputs exist",
                context,
                id.index(),
                self.transition.inputs.len()
            )
        })
    }

    fn bind_transition_output(
        &self,
        output_is_bound: &mut [bool],
        id: TransitionOutputId,
        context: &str,
    ) -> Result<&Output, String> {
        let output = self.transition_output(id, context)?;
        output_is_bound[id.index()] = true;
        Ok(output)
    }

    fn transition_output(&self, id: TransitionOutputId, context: &str) -> Result<&Output, String> {
        self.transition.outputs.get(id.index()).ok_or_else(|| {
            format!(
                "{} references transition output index {}, but only {} outputs exist",
                context,
                id.index(),
                self.transition.outputs.len()
            )
        })
    }
}
