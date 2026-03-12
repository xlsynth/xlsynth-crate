// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use crate::ComboEvalPlan;
use crate::CompiledPipelineModule;
use crate::Env;
use crate::LogicBit;
use crate::Result;
use crate::Signedness;
use crate::State;
use crate::Value4;
use crate::eval_combo_seeded;
use crate::packed::packed_index_selection_if_in_bounds;
use crate::plan_combo_eval;
use crate::step_pipeline_state_with_env;

#[derive(Debug, Clone)]
pub struct InputPortHandle {
    name: String,
    width: u32,
    signedness: Signedness,
    decl: crate::compiled_module::DeclInfo,
}

impl InputPortHandle {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn signedness(&self) -> Signedness {
        self.signedness
    }
}

#[derive(Debug, Clone)]
pub struct OutputPortHandle {
    name: String,
    width: u32,
    signedness: Signedness,
    decl: crate::compiled_module::DeclInfo,
}

impl OutputPortHandle {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn signedness(&self) -> Signedness {
        self.signedness
    }
}

pub struct PortBindings<'a> {
    module: &'a CompiledPipelineModule,
}

impl<'a> PortBindings<'a> {
    pub fn new(module: &'a CompiledPipelineModule) -> Self {
        Self { module }
    }

    pub fn maybe_input(&self, name: &str) -> Option<InputPortHandle> {
        self.module
            .combo
            .input_ports
            .iter()
            .find(|port| port.name == name)
            .and_then(|port| {
                self.module
                    .combo
                    .decls
                    .get(name)
                    .map(|decl| InputPortHandle {
                        name: name.to_string(),
                        width: port.width,
                        signedness: decl.signedness,
                        decl: decl.clone(),
                    })
            })
    }

    pub fn input(&self, name: &str) -> Result<InputPortHandle> {
        self.maybe_input(name)
            .ok_or_else(|| crate::Error::Parse(format!("input port `{name}` was not found")))
    }

    pub fn input_with_width(&self, name: &str, width: u32) -> Result<InputPortHandle> {
        let handle = self.input(name)?;
        if handle.width != width {
            return Err(crate::Error::Parse(format!(
                "input port `{name}` has width {}, expected {width}",
                handle.width
            )));
        }
        Ok(handle)
    }

    pub fn output(&self, name: &str) -> Result<OutputPortHandle> {
        self.maybe_output(name)
            .ok_or_else(|| crate::Error::Parse(format!("output port `{name}` was not found")))
    }

    pub fn output_with_width(&self, name: &str, width: u32) -> Result<OutputPortHandle> {
        let handle = self.output(name)?;
        if handle.width != width {
            return Err(crate::Error::Parse(format!(
                "output port `{name}` has width {}, expected {width}",
                handle.width
            )));
        }
        Ok(handle)
    }

    pub fn maybe_output(&self, name: &str) -> Option<OutputPortHandle> {
        self.module
            .combo
            .output_ports
            .iter()
            .find(|port| port.name == name)
            .and_then(|port| {
                self.module
                    .combo
                    .decls
                    .get(name)
                    .map(|decl| OutputPortHandle {
                        name: name.to_string(),
                        width: port.width,
                        signedness: decl.signedness,
                        decl: decl.clone(),
                    })
            })
    }
}

pub trait CycleFixture {
    fn bind(&mut self, ports: &PortBindings<'_>) -> Result<()>;

    fn drive_cycle_inputs(&mut self, _cycle: u64, _ctx: &mut DriveContext<'_>) -> Result<()> {
        Ok(())
    }

    fn observe_cycle_outputs(&mut self, _cycle: u64, _ctx: &ObserveContext<'_>) -> Result<()> {
        Ok(())
    }
}

pub struct DriveContext<'a> {
    inputs: &'a mut BTreeMap<String, Value4>,
    driven_bits: &'a mut BTreeMap<String, Vec<bool>>,
    previous_outputs: &'a BTreeMap<String, Value4>,
}

impl<'a> DriveContext<'a> {
    pub fn previous_output(&self, handle: &OutputPortHandle) -> Option<&Value4> {
        self.previous_outputs.get(handle.name())
    }

    pub fn set(&mut self, handle: &InputPortHandle, value: Value4) -> Result<()> {
        if value.width != handle.width {
            return Err(crate::Error::Parse(format!(
                "input port `{}` expects width {}, got {}",
                handle.name, handle.width, value.width
            )));
        }
        self.claim_range(
            handle.name(),
            handle.width,
            0,
            handle.width,
            format!("input port `{}`", handle.name()),
        )?;
        self.inputs.insert(
            handle.name.clone(),
            value.with_signedness(handle.signedness),
        );
        Ok(())
    }

    pub fn set_zero(&mut self, handle: &InputPortHandle) -> Result<()> {
        self.set(handle, Value4::zeros(handle.width, handle.signedness))
    }

    pub fn set_bool(&mut self, handle: &InputPortHandle, value: bool) -> Result<()> {
        self.set(
            handle,
            Value4::from_u64(handle.width, handle.signedness, u64::from(value)),
        )
    }

    pub fn set_u64(&mut self, handle: &InputPortHandle, value: u64) -> Result<()> {
        self.set(
            handle,
            Value4::from_u64(handle.width, handle.signedness, value),
        )
    }

    pub fn set_packed(
        &mut self,
        handle: &InputPortHandle,
        indices: &[u32],
        value: Value4,
    ) -> Result<()> {
        let (offset, width) = selected_range(&handle.decl, handle.name(), indices)?;
        if value.width != width {
            return Err(crate::Error::Parse(format!(
                "input port `{}` slice {:?} expects width {}, got {}",
                handle.name, indices, width, value.width
            )));
        }
        self.claim_range(
            handle.name(),
            handle.width,
            offset,
            width,
            format!("input port `{}` slice {:?}", handle.name(), indices),
        )?;
        let current = self
            .inputs
            .get(handle.name())
            .cloned()
            .unwrap_or_else(|| Value4::zeros(handle.width, handle.signedness));
        self.inputs.insert(
            handle.name.clone(),
            current.replace_slice(offset, &value.with_signedness(handle.signedness))?,
        );
        Ok(())
    }

    pub fn set_packed_u64(
        &mut self,
        handle: &InputPortHandle,
        indices: &[u32],
        value: u64,
    ) -> Result<()> {
        let (_, width) = selected_range(&handle.decl, handle.name(), indices)?;
        self.set_packed(
            handle,
            indices,
            Value4::from_u64(width, handle.signedness, value),
        )
    }

    pub fn set_packed_bool(
        &mut self,
        handle: &InputPortHandle,
        indices: &[u32],
        value: bool,
    ) -> Result<()> {
        self.set_packed_u64(handle, indices, u64::from(value))
    }

    fn claim_range(
        &mut self,
        name: &str,
        port_width: u32,
        offset: u32,
        width: u32,
        target_desc: String,
    ) -> Result<()> {
        let bits = self
            .driven_bits
            .entry(name.to_string())
            .or_insert_with(|| vec![false; port_width as usize]);
        let start = offset as usize;
        let end = (offset + width) as usize;
        if bits[start..end].iter().any(|bit| *bit) {
            return Err(crate::Error::Parse(format!(
                "{target_desc} was already driven earlier in the cycle"
            )));
        }
        for bit in &mut bits[start..end] {
            *bit = true;
        }
        Ok(())
    }
}

pub struct ObserveContext<'a> {
    inputs: &'a BTreeMap<String, Value4>,
    outputs: &'a BTreeMap<String, Value4>,
}

impl<'a> ObserveContext<'a> {
    pub fn input(&self, handle: &InputPortHandle) -> Result<&Value4> {
        self.inputs.get(handle.name()).ok_or_else(|| {
            crate::Error::Parse(format!(
                "input port `{}` has no driven value",
                handle.name()
            ))
        })
    }

    pub fn output(&self, handle: &OutputPortHandle) -> Result<&Value4> {
        self.outputs.get(handle.name()).ok_or_else(|| {
            crate::Error::Parse(format!(
                "output port `{}` has no sampled value",
                handle.name()
            ))
        })
    }

    pub fn output_u64_if_known(&self, handle: &OutputPortHandle) -> Result<Option<u64>> {
        Ok(self.output(handle)?.to_u64_if_known())
    }

    pub fn output_bool_if_known(&self, handle: &OutputPortHandle) -> Result<Option<bool>> {
        let value = self.output(handle)?;
        if value.width != 1 {
            return Err(crate::Error::Parse(format!(
                "output port `{}` is width {}, not width 1",
                handle.name(),
                value.width
            )));
        }
        Ok(match value.bit(0) {
            LogicBit::Zero => Some(false),
            LogicBit::One => Some(true),
            LogicBit::X | LogicBit::Z => None,
        })
    }

    pub fn output_packed(&self, handle: &OutputPortHandle, indices: &[u32]) -> Result<Value4> {
        let (offset, width) = selected_range(&handle.decl, handle.name(), indices)?;
        self.output(handle)?.slice_lsb_width(offset, width)
    }

    pub fn output_packed_u64_if_known(
        &self,
        handle: &OutputPortHandle,
        indices: &[u32],
    ) -> Result<Option<u64>> {
        Ok(self.output_packed(handle, indices)?.to_u64_if_known())
    }

    pub fn output_packed_bool_if_known(
        &self,
        handle: &OutputPortHandle,
        indices: &[u32],
    ) -> Result<Option<bool>> {
        let value = self.output_packed(handle, indices)?;
        if value.width != 1 {
            return Err(crate::Error::Parse(format!(
                "output port `{}` slice {:?} is width {}, not width 1",
                handle.name(),
                indices,
                value.width
            )));
        }
        Ok(match value.bit(0) {
            LogicBit::Zero => Some(false),
            LogicBit::One => Some(true),
            LogicBit::X | LogicBit::Z => None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct FixtureCycleResult {
    pub cycle: u64,
    pub inputs: BTreeMap<String, Value4>,
    pub pre_outputs: BTreeMap<String, Value4>,
    pub outputs: BTreeMap<String, Value4>,
}

pub struct FixtureRunner {
    module: CompiledPipelineModule,
    plan: ComboEvalPlan,
    state: State,
    cycle: u64,
    last_inputs: BTreeMap<String, Value4>,
    last_outputs: BTreeMap<String, Value4>,
}

impl FixtureRunner {
    pub fn new(module: &CompiledPipelineModule) -> Result<Self> {
        let module = module.clone();
        let plan = plan_combo_eval(&module.combo)?;
        Ok(Self {
            last_inputs: zero_inputs(&module)?,
            plan,
            state: module.initial_state_x(),
            cycle: 0,
            module,
            last_outputs: BTreeMap::new(),
        })
    }

    pub fn port_bindings(&self) -> PortBindings<'_> {
        PortBindings::new(&self.module)
    }

    pub fn bind_fixture(&self, fixture: &mut dyn CycleFixture) -> Result<()> {
        fixture.bind(&self.port_bindings())
    }

    pub fn bind_fixtures(&self, fixtures: &mut [&mut dyn CycleFixture]) -> Result<()> {
        let ports = self.port_bindings();
        for fixture in fixtures {
            fixture.bind(&ports)?;
        }
        Ok(())
    }

    pub fn cycle(&self) -> u64 {
        self.cycle
    }

    pub fn last_inputs(&self) -> &BTreeMap<String, Value4> {
        &self.last_inputs
    }

    pub fn last_outputs(&self) -> &BTreeMap<String, Value4> {
        &self.last_outputs
    }

    pub fn last_observe_context(&self) -> ObserveContext<'_> {
        ObserveContext {
            inputs: &self.last_inputs,
            outputs: &self.last_outputs,
        }
    }

    pub fn reset_state_x(&mut self) {
        self.state = self.module.initial_state_x();
        self.last_inputs.clear();
        self.last_outputs.clear();
        self.cycle = 0;
    }

    pub fn step_cycle(
        &mut self,
        fixtures: &mut [&mut dyn CycleFixture],
    ) -> Result<FixtureCycleResult> {
        self.step_cycle_with_drive(fixtures, |_ctx| Ok(()))
    }

    pub fn step_cycle_with_drive<F>(
        &mut self,
        fixtures: &mut [&mut dyn CycleFixture],
        extra_drive: F,
    ) -> Result<FixtureCycleResult>
    where
        F: FnOnce(&mut DriveContext<'_>) -> Result<()>,
    {
        let mut inputs = zero_inputs(&self.module)?;
        let mut driven_bits = BTreeMap::new();
        {
            let mut drive_ctx = DriveContext {
                inputs: &mut inputs,
                driven_bits: &mut driven_bits,
                previous_outputs: &self.last_outputs,
            };
            for fixture in fixtures.iter_mut() {
                fixture.drive_cycle_inputs(self.cycle, &mut drive_ctx)?;
            }
            extra_drive(&mut drive_ctx)?;
        }

        let (pre_outputs, outputs) = self.simulate_cycle(&inputs)?;
        let result = FixtureCycleResult {
            cycle: self.cycle,
            inputs: inputs.clone(),
            pre_outputs: pre_outputs.clone(),
            outputs: outputs.clone(),
        };
        {
            let observe_ctx = ObserveContext {
                inputs: &inputs,
                outputs: &outputs,
            };
            for fixture in fixtures.iter_mut() {
                fixture.observe_cycle_outputs(self.cycle, &observe_ctx)?;
            }
        }

        self.last_inputs = inputs;
        self.last_outputs = outputs;
        self.cycle += 1;
        Ok(result)
    }

    fn simulate_cycle(
        &mut self,
        inputs: &BTreeMap<String, Value4>,
    ) -> Result<(BTreeMap<String, Value4>, BTreeMap<String, Value4>)> {
        let high_seed = build_seed_env(&self.module, &self.state, inputs, true);
        let high_values = eval_combo_seeded(&self.module.combo, &self.plan, &high_seed)?;
        if self.module.seqs.is_empty() {
            return Ok((high_values.clone(), high_values));
        }
        let env = env_from_values(&high_values);
        self.state = step_pipeline_state_with_env(&self.module, &env, &self.state)?;
        let post_seed = build_seed_env(&self.module, &self.state, inputs, true);
        let post_values = eval_combo_seeded(&self.module.combo, &self.plan, &post_seed)?;
        Ok((high_values, post_values))
    }
}

fn zero_inputs(module: &CompiledPipelineModule) -> Result<BTreeMap<String, Value4>> {
    let mut out = BTreeMap::new();
    for port in &module.combo.input_ports {
        let decl = module.combo.decls.get(&port.name).ok_or_else(|| {
            crate::Error::Parse(format!("no decl info found for input `{}`", port.name))
        })?;
        out.insert(
            port.name.clone(),
            Value4::zeros(decl.width, decl.signedness),
        );
    }
    Ok(out)
}

fn build_seed_env(
    module: &CompiledPipelineModule,
    state: &State,
    inputs: &BTreeMap<String, Value4>,
    clk_high: bool,
) -> Env {
    let mut seed = Env::new();
    for (name, value) in state {
        seed.insert(name.clone(), value.clone());
    }
    for (name, value) in inputs {
        seed.insert(name.clone(), value.clone());
    }
    seed.insert(
        module.clk_name.clone(),
        Value4::from_u64(1, Signedness::Unsigned, u64::from(clk_high)),
    );
    seed
}

fn env_from_values(values: &BTreeMap<String, Value4>) -> Env {
    let mut env = Env::new();
    for (name, value) in values {
        env.insert(name.clone(), value.clone());
    }
    env
}

fn selected_range(
    decl: &crate::compiled_module::DeclInfo,
    name: &str,
    indices: &[u32],
) -> Result<(u32, u32)> {
    packed_index_selection_if_in_bounds(decl, indices)?.ok_or_else(|| {
        crate::Error::Parse(format!(
            "packed indices {:?} are out of bounds for `{name}`",
            indices
        ))
    })
}
