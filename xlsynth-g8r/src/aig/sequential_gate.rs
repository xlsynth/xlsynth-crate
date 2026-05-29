// SPDX-License-Identifier: Apache-2.0

//! Sequential design metadata around a combinational one-cycle transition
//! function.

use std::collections::BTreeSet;

use xlsynth::IrBits;

use crate::aig::gate::{AigBitVector, AigNode, AigOperand, AigRef, GateFn, Input, Output};

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

/// Binds one state element to the transition function interface.
///
/// Register `Q` is a source of combinational logic, represented as a
/// transition input. Register `D` is the effective next-state value,
/// represented as a transition output. Synchronous load-enable and reset
/// behavior is part of that transition logic. Asynchronous reset is not
/// representable by this type.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RegisterBinding {
    pub name: String,
    pub q: TransitionInputId,
    pub d: TransitionOutputId,
    pub initial_value: Option<IrBits>,
}

/// A synchronous design represented as a combinational transition function
/// plus its sequential boundaries.
///
/// `transition.inputs` are classified as external `inputs`, including
/// synchronous controls, or register `Q` values. `transition.outputs` are
/// externally visible `outputs` or effective register `D` values. A design
/// with no registers is valid and may omit `clock`.
#[derive(Debug, Clone)]
pub struct SequentialGateFn {
    pub name: String,
    pub transition: GateFn,
    pub inputs: Vec<TransitionInputId>,
    pub outputs: Vec<TransitionOutputId>,
    pub clock: Option<ClockPort>,
    pub registers: Vec<RegisterBinding>,
}

/// Returns the canonical generated name for a sequential transition function.
pub(crate) fn canonical_transition_name(design_name: &str) -> String {
    format!("{design_name}__transition")
}

/// Returns the canonical generated transition input name for a register Q port.
pub(crate) fn canonical_register_q_name(register_name: &str) -> String {
    format!("{register_name}__q")
}

/// Returns the canonical generated transition output name for a register D
/// port.
pub(crate) fn canonical_register_d_name(register_name: &str) -> String {
    format!("{register_name}__d")
}

/// Makes a canonical generated transition port name unique in an interface.
pub(crate) fn uniquify_transition_port_name(
    preferred_name: &str,
    used_names: &mut BTreeSet<String>,
) -> String {
    if used_names.insert(preferred_name.to_string()) {
        return preferred_name.to_string();
    }
    for suffix in 1usize.. {
        let candidate = format!("{preferred_name}__{suffix}");
        if used_names.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!("unbounded suffix sequence must provide a unique name")
}

/// Adds one register layer between each external input and the existing
/// transition function.
pub fn add_input_registers(
    design: &SequentialGateFn,
    clock: ClockPort,
) -> Result<SequentialGateFn, String> {
    design.validate()?;
    let mut result = design.clone();
    adopt_clock(&mut result, &clock)?;
    let original_inputs = result.inputs.clone();
    let mut register_names = result
        .registers
        .iter()
        .map(|register| register.name.clone())
        .collect::<BTreeSet<String>>();
    let mut transition_input_names = result
        .transition
        .inputs
        .iter()
        .map(|input| input.name.clone())
        .collect::<BTreeSet<String>>();
    let mut transition_output_names = result
        .transition
        .outputs
        .iter()
        .map(|output| output.name.clone())
        .collect::<BTreeSet<String>>();

    for (external_index, old_input_id) in original_inputs.into_iter().enumerate() {
        let old_input = result.transition.inputs[old_input_id.index()].clone();
        let old_input_bit_count = old_input.get_bit_count();
        let register_name =
            uniquify_register_name(&format!("p0_{}", old_input.name), &mut register_names);
        let q_name = uniquify_transition_port_name(
            &canonical_register_q_name(&register_name),
            &mut transition_input_names,
        );
        rename_transition_input(&mut result.transition, old_input_id, &q_name);

        let external_input_id =
            append_transition_input(&mut result.transition, old_input.name, old_input_bit_count);
        result.inputs[external_index] = external_input_id;
        let d_name = uniquify_transition_port_name(
            &canonical_register_d_name(&register_name),
            &mut transition_output_names,
        );
        let d_bit_vector = result.transition.inputs[external_input_id.index()]
            .bit_vector
            .clone();
        let d_id = append_transition_output(&mut result.transition, d_name, d_bit_vector);
        result.registers.push(RegisterBinding {
            name: register_name,
            q: old_input_id,
            d: d_id,
            initial_value: None,
        });
    }

    result.validate()?;
    Ok(result)
}

/// Adds one register layer between each existing external output and the
/// visible output interface.
pub fn add_output_registers(
    design: &SequentialGateFn,
    clock: ClockPort,
) -> Result<SequentialGateFn, String> {
    design.validate()?;
    let mut result = design.clone();
    adopt_clock(&mut result, &clock)?;
    let original_outputs = result.outputs.clone();
    let mut register_names = result
        .registers
        .iter()
        .map(|register| register.name.clone())
        .collect::<BTreeSet<String>>();
    let mut transition_input_names = result
        .transition
        .inputs
        .iter()
        .map(|input| input.name.clone())
        .collect::<BTreeSet<String>>();
    let mut transition_output_names = result
        .transition
        .outputs
        .iter()
        .map(|output| output.name.clone())
        .collect::<BTreeSet<String>>();

    for (external_index, old_output_id) in original_outputs.into_iter().enumerate() {
        let old_output = result.transition.outputs[old_output_id.index()].clone();
        let register_name =
            uniquify_register_name(&format!("p0_{}", old_output.name), &mut register_names);
        let d_name = uniquify_transition_port_name(
            &format!("{}_comb", old_output.name),
            &mut transition_output_names,
        );
        result.transition.outputs[old_output_id.index()].name = d_name;

        let q_name = uniquify_transition_port_name(
            &canonical_register_q_name(&register_name),
            &mut transition_input_names,
        );
        let q_id =
            append_transition_input(&mut result.transition, q_name, old_output.get_bit_count());
        let visible_bit_vector = result.transition.inputs[q_id.index()].bit_vector.clone();
        let visible_output_id =
            append_transition_output(&mut result.transition, old_output.name, visible_bit_vector);
        result.outputs[external_index] = visible_output_id;
        result.registers.push(RegisterBinding {
            name: register_name,
            q: q_id,
            d: old_output_id,
            initial_value: None,
        });
    }

    result.validate()?;
    Ok(result)
}

fn adopt_clock(design: &mut SequentialGateFn, clock: &ClockPort) -> Result<(), String> {
    match &design.clock {
        Some(existing) if existing != clock => Err(format!(
            "cannot add boundary registers using clock '{}': design already declares clock '{}'",
            clock.name, existing.name
        )),
        Some(_) => Ok(()),
        None => {
            design.clock = Some(clock.clone());
            Ok(())
        }
    }
}

fn uniquify_register_name(preferred_name: &str, used_names: &mut BTreeSet<String>) -> String {
    if used_names.insert(preferred_name.to_string()) {
        return preferred_name.to_string();
    }
    for suffix in 1usize.. {
        let candidate = format!("{preferred_name}__{suffix}");
        if used_names.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!("unbounded suffix sequence must provide a unique register name")
}

fn rename_transition_input(transition: &mut GateFn, id: TransitionInputId, name: &str) {
    let bit_vector = transition.inputs[id.index()].bit_vector.clone();
    transition.inputs[id.index()].name = name.to_string();
    for bit in bit_vector.iter_lsb_to_msb() {
        match &mut transition.gates[bit.node.id] {
            AigNode::Input {
                name: node_name, ..
            } => *node_name = name.to_string(),
            _ => unreachable!("transition input bits must refer to AIG input nodes"),
        }
    }
}

fn append_transition_input(
    transition: &mut GateFn,
    name: String,
    bit_count: usize,
) -> TransitionInputId {
    let operands = (0..bit_count)
        .map(|lsb_index| {
            let node = AigRef {
                id: transition.gates.len(),
            };
            transition.gates.push(AigNode::Input {
                name: name.clone(),
                lsb_index,
                pir_node_ids: AigNode::with_pir_node_id(None),
            });
            AigOperand {
                node,
                negated: false,
            }
        })
        .collect::<Vec<AigOperand>>();
    let id = TransitionInputId::new(transition.inputs.len());
    transition.inputs.push(Input {
        name,
        bit_vector: AigBitVector::from_lsb_is_index_0(&operands),
    });
    id
}

fn append_transition_output(
    transition: &mut GateFn,
    name: String,
    bit_vector: AigBitVector,
) -> TransitionOutputId {
    let id = TransitionOutputId::new(transition.outputs.len());
    transition.outputs.push(Output { name, bit_vector });
    id
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
