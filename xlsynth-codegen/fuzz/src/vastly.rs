// SPDX-License-Identifier: Apache-2.0

//! Replay a flat native-codegen block trace in Vastly's four-state RTL
//! simulator.

use xlsynth_test_helpers::rtl_sim::Bindings;
use xlsynth_vastly::{
    Env, Signedness, State, Value4, compile_pipeline_module_without_spans, eval_combo_seeded,
    irbits_to_value4, plan_combo_eval, step_pipeline_state_with_env,
};

use crate::semantics::Trace;

/// Binds one PIR stimulus to an RTL signal, rejecting width changes before
/// Vastly can resize the value to match its declaration.
fn drive(bindings: &Bindings, name: &str, width: u32) -> Result<Value4, String> {
    let value = bindings
        .get(name)
        .ok_or_else(|| format!("missing stimulus for RTL signal `{name}`"))?;
    if value.width() != width as usize {
        return Err(format!(
            "RTL signal `{name}` has width {width}, stimulus has width {}",
            value.width()
        ));
    }
    irbits_to_value4(&value.to_bits()?, Signedness::Unsigned)
        .map_err(|error| format!("driving `{name}`: {error:?}"))
}

/// Checks every packed bit, including X/Z, against the PIR expectation at a
/// particular point in the clock cycle.
fn compare<'a>(
    cycle: usize,
    kind: &str,
    actual: impl ExactSizeIterator<Item = (&'a String, &'a Value4)>,
    expected: &Bindings,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "cycle {cycle} {kind}: RTL has {} signals, expected {}",
            actual.len(),
            expected.len()
        ));
    }
    for (name, value) in actual {
        let expected = expected
            .get(name)
            .ok_or_else(|| format!("cycle {cycle} {kind}: unexpected RTL signal `{name}`"))?
            .to_bit_string_msb_first();
        let observed = value.to_bit_string_msb_first();
        if observed != expected {
            return Err(format!(
                "cycle {cycle} {kind} `{name}`: RTL {observed}, expected {expected}"
            ));
        }
    }
    Ok(())
}

/// Drives independently supplied register state and named inputs through the
/// emitted RTL. Each edge consumes the simulator's own previous register state.
pub fn check_rtl_trace(rtl: &str, trace: &Trace) -> Result<(), String> {
    let module = compile_pipeline_module_without_spans(rtl)
        .map_err(|error| format!("Vastly could not compile RTL: {error:?}"))?;
    let plan = plan_combo_eval(&module.combo)
        .map_err(|error| format!("Vastly could not schedule combinational logic: {error:?}"))?;

    // The generator uses distinct Verilog-safe register and port names.
    // Native codegen omits zero-width registers and ports from the RTL, as
    // does the trace from its bindings.
    let mut state: State = module.initial_state_x();
    if state.len() != trace.initial_state.len() {
        return Err(format!(
            "initial state: RTL has {} registers, trace has {}",
            state.len(),
            trace.initial_state.len()
        ));
    }
    for (name, value) in &mut state {
        *value = drive(&trace.initial_state, name, value.width)?.into_signedness(value.signedness);
    }
    for (cycle, sample) in trace.samples.iter().enumerate() {
        if module.combo.input_ports.len() != sample.inputs.len() {
            return Err(format!(
                "cycle {cycle}: RTL has {} inputs, trace has {}",
                module.combo.input_ports.len(),
                sample.inputs.len()
            ));
        }
        let mut seed = Env::new();
        for (name, value) in &state {
            seed.insert(name.clone(), value.clone());
        }
        for port in &module.combo.input_ports {
            seed.insert(
                port.name.clone(),
                drive(&sample.inputs, &port.name, port.width)?,
            );
        }
        // Native block RTL only uses its clock for the edge. Settle once with
        // old Q before executing the nonblocking register assignments.
        seed.insert(
            module.clk_name.clone(),
            Value4::from_u64(1, Signedness::Unsigned, 1),
        );
        let settled = eval_combo_seeded(&module.combo, &plan, &seed)
            .map_err(|error| format!("cycle {cycle} Vastly combinational evaluation: {error:?}"))?;
        compare(
            cycle,
            "outputs",
            module
                .combo
                .output_ports
                .iter()
                .map(|port| (&port.name, &settled[&port.name])),
            &sample.outputs,
        )
        .map_err(|error| format!("{error}; inputs: {:?}", sample.inputs))?;

        let expected = sample
            .next_state
            .as_ref()
            .ok_or_else(|| format!("cycle {cycle}: missing sequential trace state"))?;
        let mut edge = Env::new();
        for (name, value) in settled {
            edge.insert(name, value);
        }
        state = step_pipeline_state_with_env(&module, &edge, &state)
            .map_err(|error| format!("cycle {cycle} Vastly register update: {error:?}"))?;
        compare(cycle, "next state", state.iter(), expected)
            .map_err(|error| format!("{error}; inputs: {:?}", sample.inputs))?;
    }
    Ok(())
}
