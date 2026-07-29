// SPDX-License-Identifier: Apache-2.0

//! Register-aware final technology mapping for synchronous transition AIGs.

use super::{
    MappedNetlist, TechMapOptions, TechMapTimingConstraints, TechMapTimingModel,
    map_choice_aig_to_netlist, map_choice_aig_to_netlist_with_nf_constraints, scalar_bit_name,
};
use crate::aig::{ChoiceAig, GateFn, SequentialGateFn};
use crate::liberty::cell_formula::{Term, parse_formula};
use crate::liberty_model::{Cell, Library, PinDirection};
use crate::netlist::parse::{
    AssignExpr, Net, NetIndex, NetRef, NetlistInstance, NetlistModule, NetlistPort, PortDirection,
};
use crate::netlist::report::build_netlist_report_with_primary_input_arrivals;
use crate::netlist::sequential_liberty::{
    GvEvalSequentialCellSpec, get_gv_eval_sequential_cell_spec,
};
use crate::netlist::sta::{
    StaOptions, analyze_register_boundary_max_arrival,
    analyze_register_boundary_max_arrival_with_primary_input_arrivals,
    validate_output_pin_for_basic_sta,
};
use crate::netlist::utils::scalar_constant_output_assignments;
use anyhow::{Context, Result, anyhow, bail};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use string_interner::StringInterner;
use string_interner::backend::StringBackend;
use string_interner::symbol::SymbolU32;
use xlsynth::IrBits;

/// Numerical tolerance used when checking an exact register setup violation.
const REGISTER_SETUP_SLACK_EPSILON: f64 = 1e-9;

/// Controller-owned timing constraints for one synchronous mapped design.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SequentialTechMapConstraints {
    /// Arrival times of external, flattened primary-input bits.
    pub primary_input_arrivals: BTreeMap<String, f64>,
    /// Required times of external, flattened primary-output bits.
    pub primary_output_required: BTreeMap<String, f64>,
    /// Common positive-edge register clock period, in Liberty time units.
    pub clock_period: Option<f64>,
}

/// One metadata-verified, positively clocked standard-cell flip-flop.
#[derive(Clone, Debug)]
struct FlipFlopBinding {
    cell_name: String,
    clock_pin: String,
    data_pin: String,
    output_pin: String,
    tied_inputs: BTreeMap<String, bool>,
    invert_data: bool,
    area: f64,
    clock_to_output: f64,
    setup: f64,
}

/// A characterized one-input cell that can drive one packed constant bit.
#[derive(Clone, Debug)]
struct ConstantDriverBinding {
    cell_name: String,
    input_pin: String,
    output_pin: String,
    input_inverted: bool,
    area: f64,
}

/// Maps a one-clock transition function and reinstates physical flip-flops.
pub fn map_sequential_choice_aig_to_netlist(
    design: &SequentialGateFn,
    choice_aig: &ChoiceAig,
    library: &Library,
    constraints: &SequentialTechMapConstraints,
    options: &TechMapOptions,
) -> Result<MappedNetlist> {
    design
        .validate()
        .map_err(|error| anyhow!("invalid sequential design '{}': {error}", design.name))?;
    if options.buffer_options.is_some() || options.resize_options.is_some() {
        bail!(
            "sequential technology mapping does not support buffer_options or resize_options; the current post-mapping optimizer is combinational-only"
        );
    }
    validate_constraints(design, constraints)?;
    validate_transition_interface(&design.transition, choice_aig.graph())?;

    let mut effective_options = options.clone();
    if effective_options.module_name.is_none() {
        effective_options.module_name = Some(design.name.clone());
    }
    let sta_options = StaOptions {
        primary_input_transition: options.primary_input_transition,
        module_output_load: options.module_output_load,
    };

    if design.registers.is_empty() {
        let timing = TechMapTimingConstraints {
            primary_input_arrivals: constraints.primary_input_arrivals.clone(),
            primary_output_required: constraints.primary_output_required.clone(),
        };
        let mut mapped = map_transition(choice_aig, library, &timing, &effective_options)?;
        reinstate_sequential_boundary(&mut mapped, design, library, None)?;
        finalize_sequential_mapping(&mut mapped, library, constraints, sta_options)?;
        return Ok(mapped);
    }

    for register in &design.registers {
        if register.initial_value.is_some() {
            bail!(
                "register '{}' has an explicit initial value; sequential technology mapping does not support power-up initialization",
                register.name
            );
        }
    }

    let flip_flops = index_flip_flops(library, sta_options)?;
    let mut failures = Vec::new();
    for flip_flop in flip_flops {
        match map_with_flip_flop(
            design,
            choice_aig,
            library,
            constraints,
            &effective_options,
            sta_options,
            &flip_flop,
        ) {
            Ok(mapped) => return Ok(mapped),
            Err(error) => {
                if failures.len() < 4 {
                    failures.push(format!("{}: {error:#}", flip_flop.cell_name));
                }
            }
        }
    }
    bail!(
        "no usable positive-edge synchronous flip-flop produced a valid sequential mapping: {}",
        failures.join("; ")
    )
}

/// Validates non-negative, finite timing against actual external endpoints.
fn validate_constraints(
    design: &SequentialGateFn,
    constraints: &SequentialTechMapConstraints,
) -> Result<()> {
    if let Some(period) = constraints.clock_period {
        if !period.is_finite() || period <= 0.0 {
            bail!("clock period must be finite and strictly positive; got {period}");
        }
    }

    let external_inputs = design
        .inputs
        .iter()
        .flat_map(|id| {
            let port = &design.transition.inputs[id.index()];
            (0..port.get_bit_count())
                .map(move |bit| scalar_bit_name(&port.name, bit, port.get_bit_count()))
        })
        .collect::<BTreeSet<_>>();
    let external_outputs = design
        .outputs
        .iter()
        .flat_map(|id| {
            let port = &design.transition.outputs[id.index()];
            (0..port.get_bit_count())
                .map(move |bit| scalar_bit_name(&port.name, bit, port.get_bit_count()))
        })
        .collect::<BTreeSet<_>>();

    for (name, arrival) in &constraints.primary_input_arrivals {
        if !external_inputs.contains(name) {
            bail!("timing constraint names unknown external primary input '{name}'");
        }
        if !arrival.is_finite() || *arrival < 0.0 {
            bail!(
                "primary input arrival for '{name}' must be non-negative and finite; got {arrival}"
            );
        }
    }
    for (name, required) in &constraints.primary_output_required {
        if !external_outputs.contains(name) {
            bail!("timing constraint names unknown external primary output '{name}'");
        }
        if !required.is_finite() || *required < 0.0 {
            bail!(
                "primary output required time for '{name}' must be non-negative and finite; got {required}"
            );
        }
    }

    if let Some(clock) = &design.clock {
        if external_inputs.contains(&clock.name) || external_outputs.contains(&clock.name) {
            bail!(
                "clock name '{}' collides with an external data port",
                clock.name
            );
        }
    }

    let external_output_ids = design
        .outputs
        .iter()
        .map(|id| id.index())
        .collect::<BTreeSet<_>>();
    for register in &design.registers {
        if external_output_ids.contains(&register.d.index()) {
            bail!(
                "register '{}' D transition output is also an external output; sequential technology mapping requires distinct register and external endpoints",
                register.name
            );
        }
    }
    Ok(())
}

/// Requires the optimized AIG to preserve every transition bit and its order.
fn validate_transition_interface(original: &GateFn, optimized: &GateFn) -> Result<()> {
    let original_inputs = flatten_input_names(original)?;
    let optimized_inputs = flatten_input_names(optimized)?;
    check_flattened_interface("input", &original_inputs, &optimized_inputs)?;
    let original_outputs = flatten_output_names(original)?;
    let optimized_outputs = flatten_output_names(optimized)?;
    check_flattened_interface("output", &original_outputs, &optimized_outputs)
}

/// Collects transition inputs in AIGER-compatible, low-to-high bit order.
fn flatten_input_names(graph: &GateFn) -> Result<Vec<String>> {
    let mut names = Vec::new();
    let mut seen = BTreeSet::new();
    for input in &graph.inputs {
        if input.get_bit_count() == 0 {
            bail!("transition input '{}' has zero bits", input.name);
        }
        for (bit_index, bit) in input.bit_vector.iter_lsb_to_msb().enumerate() {
            if bit.negated {
                bail!(
                    "transition input '{}' has a negated input binding",
                    input.name
                );
            }
            let name = scalar_bit_name(&input.name, bit_index, input.get_bit_count());
            if !seen.insert(name.clone()) {
                bail!("transition has duplicate flattened input '{name}'");
            }
            names.push(name);
        }
    }
    Ok(names)
}

/// Collects transition outputs in AIGER-compatible, low-to-high bit order.
fn flatten_output_names(graph: &GateFn) -> Result<Vec<String>> {
    let mut names = Vec::new();
    let mut seen = BTreeSet::new();
    for output in &graph.outputs {
        if output.get_bit_count() == 0 {
            bail!("transition output '{}' has zero bits", output.name);
        }
        for bit_index in 0..output.get_bit_count() {
            let name = scalar_bit_name(&output.name, bit_index, output.get_bit_count());
            if !seen.insert(name.clone()) {
                bail!("transition has duplicate flattened output '{name}'");
            }
            names.push(name);
        }
    }
    Ok(names)
}

/// Reports the first changed bit instead of silently rebinding register state.
fn check_flattened_interface(kind: &str, expected: &[String], actual: &[String]) -> Result<()> {
    if expected.len() != actual.len() {
        bail!(
            "optimized transition {kind} interface has {} bits; original transition has {}",
            actual.len(),
            expected.len()
        );
    }
    for (index, (expected_name, actual_name)) in expected.iter().zip(actual).enumerate() {
        if expected_name != actual_name {
            bail!(
                "optimized transition {kind} bit {index} is '{actual_name}'; expected '{expected_name}'"
            );
        }
    }
    Ok(())
}

/// Selects the compact constrained NF implementation only for NF modes.
fn map_transition(
    choice_aig: &ChoiceAig,
    library: &Library,
    constraints: &TechMapTimingConstraints,
    options: &TechMapOptions,
) -> Result<MappedNetlist> {
    if matches!(
        options.timing_model,
        TechMapTimingModel::NfUnit | TechMapTimingModel::NfLiberty
    ) {
        map_choice_aig_to_netlist_with_nf_constraints(choice_aig, library, constraints, options)
    } else {
        map_choice_aig_to_netlist(choice_aig, library, constraints, options)
    }
}

/// Maps with one characterized flip-flop and then restores the clock boundary.
#[allow(clippy::too_many_arguments)]
fn map_with_flip_flop(
    design: &SequentialGateFn,
    choice_aig: &ChoiceAig,
    library: &Library,
    constraints: &SequentialTechMapConstraints,
    options: &TechMapOptions,
    sta_options: StaOptions,
    flip_flop: &FlipFlopBinding,
) -> Result<MappedNetlist> {
    if let Some(period) = constraints.clock_period {
        if period < flip_flop.setup {
            bail!(
                "clock period {period} is shorter than flip-flop '{}' setup requirement {}; a non-negative register required time cannot be constructed",
                flip_flop.cell_name,
                flip_flop.setup
            );
        }
    }

    let adjusted = adjust_register_data_phase(design, choice_aig, flip_flop.invert_data)?;
    let mut timing = TechMapTimingConstraints {
        primary_input_arrivals: constraints.primary_input_arrivals.clone(),
        primary_output_required: constraints.primary_output_required.clone(),
    };
    for register in &design.registers {
        let q = &design.transition.inputs[register.q.index()];
        let d = &design.transition.outputs[register.d.index()];
        for bit in 0..q.get_bit_count() {
            timing.primary_input_arrivals.insert(
                scalar_bit_name(&q.name, bit, q.get_bit_count()),
                flip_flop.clock_to_output,
            );
            if let Some(period) = constraints.clock_period {
                timing.primary_output_required.insert(
                    scalar_bit_name(&d.name, bit, d.get_bit_count()),
                    period - flip_flop.setup,
                );
            }
        }
    }

    let mut mapped = map_transition(&adjusted, library, &timing, options)
        .with_context(|| format!("mapping transition for flip-flop '{}'", flip_flop.cell_name))?;
    reinstate_sequential_boundary(&mut mapped, design, library, Some(flip_flop))?;
    finalize_sequential_mapping(&mut mapped, library, constraints, sta_options)?;
    Ok(mapped)
}

/// Inverts physical D roots without discarding ABC structural-choice links.
fn adjust_register_data_phase(
    design: &SequentialGateFn,
    choice_aig: &ChoiceAig,
    invert_data: bool,
) -> Result<ChoiceAig> {
    if !invert_data {
        return Ok(choice_aig.clone());
    }
    let mut graph = choice_aig.graph().clone();
    let mut positions = BTreeMap::new();
    for (output_index, output) in graph.outputs.iter().enumerate() {
        for bit_index in 0..output.get_bit_count() {
            let name = scalar_bit_name(&output.name, bit_index, output.get_bit_count());
            positions.insert(name, (output_index, bit_index));
        }
    }
    let mut inverted = BTreeSet::new();
    for register in &design.registers {
        let output = &design.transition.outputs[register.d.index()];
        for bit_index in 0..output.get_bit_count() {
            let name = scalar_bit_name(&output.name, bit_index, output.get_bit_count());
            if !inverted.insert(name.clone()) {
                continue;
            }
            let (port_index, optimized_bit) = positions.get(&name).copied().ok_or_else(|| {
                anyhow!("optimized transition is missing register D bit '{name}'")
            })?;
            let operand = *graph.outputs[port_index].bit_vector.get_lsb(optimized_bit);
            graph.outputs[port_index]
                .bit_vector
                .set_lsb(optimized_bit, operand.negate());
        }
    }
    ChoiceAig::new(graph, choice_aig.sibling_links().to_vec()).map_err(|error| {
        anyhow!("could not preserve choice links while phasing register D: {error}")
    })
}

/// Builds and timing-characterizes eligible Liberty FFs without cell-name
/// rules.
fn index_flip_flops(library: &Library, options: StaOptions) -> Result<Vec<FlipFlopBinding>> {
    let mut candidates = Vec::new();
    let mut rejected = BTreeSet::new();
    for cell in &library.cells {
        if cell.sequential.is_empty() || cell.dont_use == Some(true) {
            continue;
        }
        if !cell.area.is_finite() || cell.area < 0.0 {
            rejected.insert(format!("{} has invalid cell area", cell.name));
            continue;
        }
        let spec = match get_gv_eval_sequential_cell_spec(cell, library) {
            Ok(Some(spec)) => spec,
            Ok(None) => continue,
            Err(error) => {
                rejected.insert(error);
                continue;
            }
        };
        if spec.clock.is_negated {
            rejected.insert(format!("{} requires a negative clock edge", cell.name));
            continue;
        }
        match enumerate_flip_flop_bindings(cell, &spec, library, options) {
            Ok(bindings) if !bindings.is_empty() => candidates.extend(bindings),
            Ok(_) => {
                rejected.insert(format!(
                    "{} has no supported state output and data-input phase",
                    cell.name
                ));
            }
            Err(error) => {
                rejected.insert(format!("{}: {error:#}", cell.name));
            }
        }
    }
    candidates.sort_by(|lhs, rhs| {
        lhs.area
            .total_cmp(&rhs.area)
            .then_with(|| {
                (lhs.clock_to_output + lhs.setup).total_cmp(&(rhs.clock_to_output + rhs.setup))
            })
            .then_with(|| lhs.invert_data.cmp(&rhs.invert_data))
            .then_with(|| lhs.cell_name.cmp(&rhs.cell_name))
            .then_with(|| lhs.output_pin.cmp(&rhs.output_pin))
            .then_with(|| lhs.data_pin.cmp(&rhs.data_pin))
            .then_with(|| lhs.tied_inputs.cmp(&rhs.tied_inputs))
    });
    if candidates.is_empty() {
        let reasons = rejected.into_iter().take(6).collect::<Vec<_>>().join("; ");
        bail!(
            "Liberty has no usable positive-edge synchronous flip-flop with characterized clock-to-Q and setup timing{}",
            if reasons.is_empty() {
                String::new()
            } else {
                format!(": {reasons}")
            }
        );
    }
    Ok(candidates)
}

/// Enumerates state-output polarities and constant-safe scan/control settings.
fn enumerate_flip_flop_bindings(
    cell: &Cell,
    spec: &GvEvalSequentialCellSpec,
    library: &Library,
    options: StaOptions,
) -> Result<Vec<FlipFlopBinding>> {
    let mut input_names = cell
        .pins
        .iter()
        .filter(|pin| {
            pin.direction == PinDirection::Input as i32
                && !pin.is_clocking_pin
                && library.resolve_string(&pin.name) != spec.clock.pin_name
        })
        .map(|pin| library.resolve_string(&pin.name).to_string())
        .collect::<Vec<_>>();
    input_names.sort();
    if input_names.windows(2).any(|pair| pair[0] == pair[1]) {
        bail!("flip-flop contains duplicate data/control input pins");
    }
    if input_names.is_empty() {
        bail!("flip-flop has no data input");
    }
    if input_names.len() > 9 {
        bail!("flip-flop has more than eight constant-select control inputs");
    }

    let allowed_names = input_names
        .iter()
        .cloned()
        .chain(std::iter::once(spec.state_var.clone()))
        .chain(spec.complementary_state_var.iter().cloned())
        .collect::<BTreeSet<_>>();
    for name in spec.next_state.inputs() {
        if !allowed_names.contains(&name) {
            bail!("next_state references unsupported input or clock '{name}'");
        }
    }

    let mut output_choices = Vec::new();
    for pin in cell
        .pins
        .iter()
        .filter(|pin| pin.direction == PinDirection::Output as i32)
    {
        let pin_name = library.resolve_string(&pin.name);
        let formula_text = library.resolve_string(&pin.function);
        if formula_text.is_empty() {
            continue;
        }
        let Ok(term) = parse_formula(formula_text) else {
            continue;
        };
        if term.inputs().iter().any(|name| {
            name != &spec.state_var
                && spec
                    .complementary_state_var
                    .as_ref()
                    .is_none_or(|complement| name != complement)
        }) {
            continue;
        }
        let Some(output_inverted) = state_formula_polarity(&term, spec) else {
            continue;
        };
        output_choices.push((pin_name.to_string(), output_inverted));
    }

    let mut candidates = Vec::new();
    for data_pin in &input_names {
        let control_pins = input_names
            .iter()
            .filter(|name| *name != data_pin)
            .cloned()
            .collect::<Vec<_>>();
        for assignment in 0..(1usize << control_pins.len()) {
            let tied_inputs = control_pins
                .iter()
                .enumerate()
                .map(|(index, name)| (name.clone(), ((assignment >> index) & 1) != 0))
                .collect::<BTreeMap<_, _>>();
            let Some(data_inverted) =
                data_formula_polarity(&spec.next_state, spec, data_pin, &tied_inputs)
            else {
                continue;
            };
            for (output_pin, output_inverted) in &output_choices {
                let mut candidate = FlipFlopBinding {
                    cell_name: cell.name.clone(),
                    clock_pin: spec.clock.pin_name.clone(),
                    data_pin: data_pin.clone(),
                    output_pin: output_pin.clone(),
                    tied_inputs: tied_inputs.clone(),
                    invert_data: data_inverted ^ output_inverted,
                    area: cell.area,
                    clock_to_output: 0.0,
                    setup: 0.0,
                };
                if let Ok((clock_to_output, setup)) =
                    characterize_flip_flop(&candidate, library, options)
                {
                    candidate.clock_to_output = clock_to_output;
                    candidate.setup = setup;
                    candidates.push(candidate);
                }
            }
        }
    }
    Ok(candidates)
}

/// Recognizes a real state output independently of the output pin's spelling.
fn state_formula_polarity(term: &Term, spec: &GvEvalSequentialCellSpec) -> Option<bool> {
    let mut values = HashMap::new();
    values.insert(spec.state_var.clone(), false);
    if let Some(complement) = &spec.complementary_state_var {
        values.insert(complement.clone(), true);
    }
    let at_zero = term.evaluate_partial(&values)?;
    values.insert(spec.state_var.clone(), true);
    if let Some(complement) = &spec.complementary_state_var {
        values.insert(complement.clone(), false);
    }
    let at_one = term.evaluate_partial(&values)?;
    (at_zero != at_one).then_some(at_zero)
}

/// Verifies that tied controls reduce next-state logic to `D` or `!D`.
fn data_formula_polarity(
    term: &Term,
    spec: &GvEvalSequentialCellSpec,
    data_pin: &str,
    tied_inputs: &BTreeMap<String, bool>,
) -> Option<bool> {
    let mut values = tied_inputs
        .iter()
        .map(|(name, value)| (name.clone(), *value))
        .collect::<HashMap<_, _>>();
    let mut polarity = None;
    for old_state in [false, true] {
        values.insert(spec.state_var.clone(), old_state);
        if let Some(complement) = &spec.complementary_state_var {
            values.insert(complement.clone(), !old_state);
        }
        values.insert(data_pin.to_string(), false);
        let at_zero = term.evaluate_partial(&values)?;
        values.insert(data_pin.to_string(), true);
        let at_one = term.evaluate_partial(&values)?;
        if at_zero == at_one || polarity.is_some_and(|previous| previous != at_zero) {
            return None;
        }
        polarity = Some(at_zero);
    }
    polarity
}

/// Measures Liberty setup and clock-to-Q with the production STA engine.
fn characterize_flip_flop(
    binding: &FlipFlopBinding,
    library: &Library,
    options: StaOptions,
) -> Result<(f64, f64)> {
    let probe_options = StaOptions {
        module_output_load: 0.0,
        ..options
    };
    let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
    let data = interner.get_or_intern("__seq_probe_data");
    let clock = interner.get_or_intern("__seq_probe_clock");
    let output = interner.get_or_intern("__seq_probe_output");
    let nets = vec![
        Net {
            name: data,
            width: None,
        },
        Net {
            name: clock,
            width: None,
        },
        Net {
            name: output,
            width: None,
        },
    ];
    let mut connections = vec![
        (
            interner.get_or_intern(&binding.data_pin),
            NetRef::Simple(NetIndex(0)),
        ),
        (
            interner.get_or_intern(&binding.clock_pin),
            NetRef::Simple(NetIndex(1)),
        ),
        (
            interner.get_or_intern(&binding.output_pin),
            NetRef::Simple(NetIndex(2)),
        ),
    ];
    for (name, value) in &binding.tied_inputs {
        connections.push((interner.get_or_intern(name), literal_net_ref(*value)?));
    }
    connections.sort_by(|(lhs, _), (rhs, _)| interner.resolve(*lhs).cmp(&interner.resolve(*rhs)));
    let module = NetlistModule {
        name: interner.get_or_intern("__seq_probe"),
        net_index_range: 0..nets.len(),
        ports: vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: data,
            },
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: clock,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: output,
            },
        ],
        wires: vec![],
        assigns: vec![],
        instances: vec![NetlistInstance {
            type_name: interner.get_or_intern(&binding.cell_name),
            instance_name: interner.get_or_intern("__seq_probe_register"),
            connections,
            inst_lineno: 1,
            inst_colno: 1,
        }],
    };
    let launch = analyze_register_boundary_max_arrival(
        &module,
        &nets,
        &interner,
        library,
        probe_options,
        false,
        &[0],
    )?;
    let q_timing = launch
        .timing_for_net(NetIndex(2))
        .ok_or_else(|| anyhow!("flip-flop output has no characterized clock-to-Q timing"))?;
    let clock_to_output = q_timing.rise.arrival.max(q_timing.fall.arrival);
    let capture = analyze_register_boundary_max_arrival(
        &module,
        &nets,
        &interner,
        library,
        probe_options,
        true,
        &[],
    )?;
    let setup = capture
        .register_input_arrivals
        .first()
        .copied()
        .flatten()
        .ok_or_else(|| anyhow!("flip-flop data pin has no characterized setup timing"))?;
    if !clock_to_output.is_finite() || !setup.is_finite() {
        bail!("flip-flop setup and clock-to-Q timing must be finite");
    }
    Ok((clock_to_output, setup))
}

/// Produces an ordinary one-bit standard-cell constant connection.
fn literal_net_ref(value: bool) -> Result<NetRef> {
    IrBits::make_ubits(1, u64::from(value))
        .map(NetRef::Literal)
        .map_err(|error| anyhow!("could not construct one-bit flip-flop tie-off: {error}"))
}

/// Finds a complete Liberty identity or inverter for packed constant outputs.
fn select_constant_driver(library: &Library) -> Result<ConstantDriverBinding> {
    let mut bindings = Vec::new();
    for cell in &library.cells {
        if !cell.sequential.is_empty()
            || cell.clock_gate.is_some()
            || cell.dont_use == Some(true)
            || !cell.area.is_finite()
            || cell.area < 0.0
        {
            continue;
        }
        let inputs = cell
            .pins
            .iter()
            .filter(|pin| pin.direction == PinDirection::Input as i32)
            .collect::<Vec<_>>();
        if inputs.len() != 1 || inputs[0].is_clocking_pin {
            continue;
        }
        let input_name = library.resolve_string(&inputs[0].name);
        for output in cell
            .pins
            .iter()
            .filter(|pin| pin.direction == PinDirection::Output as i32)
        {
            let formula = library.resolve_string(&output.function);
            let Ok(term) = parse_formula(formula) else {
                continue;
            };
            if term.inputs().iter().any(|name| name != input_name) {
                continue;
            }
            let mut values = HashMap::new();
            values.insert(input_name.to_string(), false);
            let Some(at_zero) = term.evaluate_partial(&values) else {
                continue;
            };
            values.insert(input_name.to_string(), true);
            let Some(at_one) = term.evaluate_partial(&values) else {
                continue;
            };
            if at_zero == at_one
                || validate_output_pin_for_basic_sta(
                    library,
                    &cell.name,
                    output,
                    &[input_name.to_string()],
                )
                .is_err()
            {
                continue;
            }
            bindings.push(ConstantDriverBinding {
                cell_name: cell.name.clone(),
                input_pin: input_name.to_string(),
                output_pin: library.resolve_string(&output.name).to_string(),
                input_inverted: at_zero,
                area: cell.area,
            });
        }
    }
    bindings.sort_by(|lhs, rhs| {
        lhs.area
            .total_cmp(&rhs.area)
            .then_with(|| lhs.input_inverted.cmp(&rhs.input_inverted))
            .then_with(|| lhs.cell_name.cmp(&rhs.cell_name))
            .then_with(|| lhs.output_pin.cmp(&rhs.output_pin))
    });
    bindings.into_iter().next().ok_or_else(|| {
        anyhow!("a packed constant output requires a characterized Liberty buffer or inverter")
    })
}

/// Replaces exposed transition-state ports with internal FF-connected wires.
fn reinstate_sequential_boundary(
    mapped: &mut MappedNetlist,
    design: &SequentialGateFn,
    library: &Library,
    binding: Option<&FlipFlopBinding>,
) -> Result<()> {
    let mut port_nets = BTreeMap::new();
    for port in &mapped.module.ports {
        let name = mapped
            .interner
            .resolve(port.name)
            .ok_or_else(|| anyhow!("could not resolve mapped transition port"))?
            .to_string();
        let net = mapped
            .module
            .find_net_index(port.name, &mapped.nets)
            .ok_or_else(|| anyhow!("mapped transition port '{name}' has no net"))?;
        if port_nets.insert(name.clone(), net).is_some() {
            bail!("mapped transition contains duplicate port '{name}'");
        }
    }
    let constant_outputs = scalar_constant_output_assignments(&mapped.module, &mapped.nets)
        .context("validating mapped transition constant outputs")?;

    let mut external_names = BTreeSet::new();
    let mut hidden_scalar_nets = BTreeSet::new();
    let mut external_bit_refs = BTreeMap::new();
    let mut ports = Vec::new();
    for id in &design.inputs {
        let input = &design.transition.inputs[id.index()];
        if !external_names.insert(input.name.clone()) {
            bail!(
                "sequential module has duplicate external port '{}'",
                input.name
            );
        }
        ports.push(restore_external_port(
            mapped,
            &port_nets,
            &input.name,
            input.get_bit_count(),
            PortDirection::Input,
            &mut hidden_scalar_nets,
            &mut external_bit_refs,
        )?);
    }

    let clock_net = if let Some(clock) = &design.clock {
        if mapped
            .nets
            .iter()
            .any(|net| mapped.interner.resolve(net.name) == Some(clock.name.as_str()))
        {
            bail!("clock '{}' collides with a transition net", clock.name);
        }
        if !external_names.insert(clock.name.clone()) {
            bail!("clock '{}' collides with an external data port", clock.name);
        }
        let symbol = mapped.interner.get_or_intern(&clock.name);
        let index = NetIndex(mapped.nets.len());
        mapped.nets.push(Net {
            name: symbol,
            width: None,
        });
        ports.push(NetlistPort {
            direction: PortDirection::Input,
            width: None,
            name: symbol,
        });
        Some(index)
    } else {
        None
    };

    for id in &design.outputs {
        let output = &design.transition.outputs[id.index()];
        if !external_names.insert(output.name.clone()) {
            bail!(
                "sequential module has duplicate external port '{}'",
                output.name
            );
        }
        ports.push(restore_external_port(
            mapped,
            &port_nets,
            &output.name,
            output.get_bit_count(),
            PortDirection::Output,
            &mut hidden_scalar_nets,
            &mut external_bit_refs,
        )?);
    }

    for instance in &mut mapped.module.instances {
        for (_, reference) in &mut instance.connections {
            remap_external_net_ref(reference, &external_bit_refs)?;
        }
    }

    let register_d_nets = design
        .registers
        .iter()
        .flat_map(|register| {
            let output = &design.transition.outputs[register.d.index()];
            (0..output.get_bit_count())
                .map(move |bit| scalar_bit_name(&output.name, bit, output.get_bit_count()))
        })
        .filter_map(|name| port_nets.get(&name).copied())
        .map(|net| net.0)
        .collect::<BTreeSet<_>>();
    let mut packed_output_constants = Vec::new();
    for id in &design.outputs {
        let output = &design.transition.outputs[id.index()];
        if output.get_bit_count() == 1 {
            continue;
        }
        for bit in 0..output.get_bit_count() {
            let name = scalar_bit_name(&output.name, bit, output.get_bit_count());
            let net = *port_nets
                .get(&name)
                .ok_or_else(|| anyhow!("mapped transition is missing external output '{name}'"))?;
            if let Some(value) = constant_outputs.get(&net.0).copied() {
                let packed_ref = external_bit_refs
                    .get(&net.0)
                    .cloned()
                    .ok_or_else(|| anyhow!("packed constant output '{name}' has no bit binding"))?;
                packed_output_constants.push((name, packed_ref, value, net.0));
            }
        }
    }
    let packed_constant_nets = packed_output_constants
        .iter()
        .map(|(_, _, _, index)| *index)
        .collect::<BTreeSet<_>>();
    mapped.module.assigns.retain(|assign| {
        !matches!(
            assign.lhs,
            NetRef::Simple(net)
                if register_d_nets.contains(&net.0) || packed_constant_nets.contains(&net.0)
        )
    });
    for assign in &mut mapped.module.assigns {
        remap_external_net_ref(&mut assign.lhs, &external_bit_refs)?;
        remap_external_assign_expr(&mut assign.rhs, &external_bit_refs)?;
    }

    let mut used_instance_names = mapped
        .module
        .instances
        .iter()
        .map(|instance| {
            mapped
                .interner
                .resolve(instance.instance_name)
                .ok_or_else(|| anyhow!("could not resolve mapped instance name"))
                .map(str::to_string)
        })
        .collect::<Result<BTreeSet<_>>>()?;

    if !packed_output_constants.is_empty() {
        let driver = select_constant_driver(library)?;
        for (name, output, value, _) in packed_output_constants {
            let instance_name =
                unique_instance_name(&format!("u_const_{name}"), &mut used_instance_names);
            let input = literal_net_ref(value ^ driver.input_inverted)?;
            let mut connections = vec![
                (mapped.interner.get_or_intern(&driver.input_pin), input),
                (mapped.interner.get_or_intern(&driver.output_pin), output),
            ];
            connections.sort_by(|(lhs, _), (rhs, _)| {
                mapped
                    .interner
                    .resolve(*lhs)
                    .cmp(&mapped.interner.resolve(*rhs))
            });
            mapped.module.instances.push(NetlistInstance {
                type_name: mapped.interner.get_or_intern(&driver.cell_name),
                instance_name: mapped.interner.get_or_intern(&instance_name),
                connections,
                inst_lineno: 1,
                inst_colno: 1,
            });
        }
    }

    if let Some(flip_flop) = binding {
        let clock_net = clock_net.ok_or_else(|| {
            anyhow!("a registered sequential design must expose its positive-edge clock")
        })?;
        for register in &design.registers {
            let q = &design.transition.inputs[register.q.index()];
            let d = &design.transition.outputs[register.d.index()];
            for bit in 0..q.get_bit_count() {
                let q_name = scalar_bit_name(&q.name, bit, q.get_bit_count());
                let d_name = scalar_bit_name(&d.name, bit, d.get_bit_count());
                let q_net = *port_nets
                    .get(&q_name)
                    .ok_or_else(|| anyhow!("missing register Q net '{q_name}'"))?;
                let d_net = *port_nets
                    .get(&d_name)
                    .ok_or_else(|| anyhow!("missing register D net '{d_name}'"))?;
                let data = match constant_outputs.get(&d_net.0).copied() {
                    Some(value) => literal_net_ref(value)?,
                    None => NetRef::Simple(d_net),
                };
                let preferred = format!(
                    "u_reg_{}",
                    scalar_bit_name(&register.name, bit, q.get_bit_count())
                );
                let instance_name = unique_instance_name(&preferred, &mut used_instance_names);
                let mut connections = vec![
                    (
                        mapped.interner.get_or_intern(&flip_flop.clock_pin),
                        NetRef::Simple(clock_net),
                    ),
                    (mapped.interner.get_or_intern(&flip_flop.data_pin), data),
                    (
                        mapped.interner.get_or_intern(&flip_flop.output_pin),
                        NetRef::Simple(q_net),
                    ),
                ];
                for (pin, value) in &flip_flop.tied_inputs {
                    connections
                        .push((mapped.interner.get_or_intern(pin), literal_net_ref(*value)?));
                }
                connections.sort_by(|(lhs, _), (rhs, _)| {
                    mapped
                        .interner
                        .resolve(*lhs)
                        .cmp(&mapped.interner.resolve(*rhs))
                });
                mapped.module.instances.push(NetlistInstance {
                    type_name: mapped.interner.get_or_intern(&flip_flop.cell_name),
                    instance_name: mapped.interner.get_or_intern(&instance_name),
                    connections,
                    inst_lineno: 1,
                    inst_colno: 1,
                });
            }
        }
    }

    let public_nets = ports.iter().map(|port| port.name).collect::<BTreeSet<_>>();
    mapped.module.ports = ports;
    mapped.module.wires = mapped
        .nets
        .iter()
        .enumerate()
        .filter_map(|(index, net)| {
            (!public_nets.contains(&net.name)
                && !hidden_scalar_nets.contains(&index)
                && !(register_d_nets.contains(&index) && constant_outputs.contains_key(&index)))
            .then_some(NetIndex(index))
        })
        .collect();
    mapped.module.net_index_range = 0..mapped.nets.len();
    Ok(())
}

/// Restores one original scalar or packed external transition port.
fn restore_external_port(
    mapped: &mut MappedNetlist,
    port_nets: &BTreeMap<String, NetIndex>,
    name: &str,
    bit_count: usize,
    direction: PortDirection,
    hidden_scalar_nets: &mut BTreeSet<usize>,
    external_bit_refs: &mut BTreeMap<usize, NetRef>,
) -> Result<NetlistPort> {
    if bit_count == 0 {
        bail!("external port '{name}' has zero bits");
    }
    if bit_count == 1 {
        let net = *port_nets
            .get(name)
            .ok_or_else(|| anyhow!("mapped transition is missing external port '{name}'"))?;
        return Ok(NetlistPort {
            direction,
            width: None,
            name: mapped.nets[net.0].name,
        });
    }

    if mapped
        .nets
        .iter()
        .any(|net| mapped.interner.resolve(net.name) == Some(name))
    {
        bail!("packed external port '{name}' collides with an existing transition net");
    }
    let msb = u32::try_from(bit_count - 1)
        .map_err(|_| anyhow!("external port '{name}' is too wide for a Verilog packed range"))?;
    let symbol = mapped.interner.get_or_intern(name);
    let packed_net = NetIndex(mapped.nets.len());
    mapped.nets.push(Net {
        name: symbol,
        width: Some((msb, 0)),
    });

    for bit in 0..bit_count {
        let scalar = scalar_bit_name(name, bit, bit_count);
        let scalar_net = *port_nets
            .get(&scalar)
            .ok_or_else(|| anyhow!("mapped transition is missing external port bit '{scalar}'"))?;
        let bit = u32::try_from(bit)
            .map_err(|_| anyhow!("external port bit '{scalar}' exceeds a Verilog bit index"))?;
        if external_bit_refs
            .insert(scalar_net.0, NetRef::BitSelect(packed_net, bit))
            .is_some()
        {
            bail!("external port bit '{scalar}' is bound more than once");
        }
        hidden_scalar_nets.insert(scalar_net.0);
    }

    Ok(NetlistPort {
        direction,
        width: Some((msb, 0)),
        name: symbol,
    })
}

/// Rebinds old flattened external nets directly to packed Verilog bits.
fn remap_external_net_ref(
    reference: &mut NetRef,
    external_bit_refs: &BTreeMap<usize, NetRef>,
) -> Result<()> {
    match reference {
        NetRef::Simple(index) => {
            if let Some(replacement) = external_bit_refs.get(&index.0) {
                *reference = replacement.clone();
            }
        }
        NetRef::BitSelect(index, bit) => {
            if let Some(replacement) = external_bit_refs.get(&index.0) {
                if *bit != 0 {
                    bail!("flattened scalar external net uses invalid bit selection {bit}");
                }
                *reference = replacement.clone();
            }
        }
        NetRef::PartSelect(index, msb, lsb) => {
            if let Some(replacement) = external_bit_refs.get(&index.0) {
                if *msb != 0 || *lsb != 0 {
                    bail!("flattened scalar external net uses an invalid part selection");
                }
                *reference = replacement.clone();
            }
        }
        NetRef::Concat(parts) => {
            for part in parts {
                remap_external_net_ref(part, external_bit_refs)?;
            }
        }
        NetRef::Literal(_) | NetRef::UnknownLiteral(_) | NetRef::Unconnected => {
            // Constants and unconnected pins do not refer to a flattened port.
        }
    }
    Ok(())
}

/// Rebinds packed external references throughout a structural assignment.
fn remap_external_assign_expr(
    expression: &mut AssignExpr,
    external_bit_refs: &BTreeMap<usize, NetRef>,
) -> Result<()> {
    match expression {
        AssignExpr::Leaf(reference) => remap_external_net_ref(reference, external_bit_refs),
        AssignExpr::Not(inner) => remap_external_assign_expr(inner, external_bit_refs),
        AssignExpr::And(lhs, rhs) | AssignExpr::Or(lhs, rhs) | AssignExpr::Xor(lhs, rhs) => {
            remap_external_assign_expr(lhs, external_bit_refs)?;
            remap_external_assign_expr(rhs, external_bit_refs)
        }
    }
}

/// Selects a collision-free, deterministic physical register instance name.
fn unique_instance_name(preferred: &str, used: &mut BTreeSet<String>) -> String {
    if used.insert(preferred.to_string()) {
        return preferred.to_string();
    }
    for suffix in 1usize.. {
        let name = format!("{preferred}__{suffix}");
        if used.insert(name.clone()) {
            return name;
        }
    }
    unreachable!("an unbounded suffix sequence must yield a unique instance name")
}

/// Checks each output deadline against both exact physical launch classes.
fn validate_exact_output_requirements(
    mapped: &MappedNetlist,
    library: &Library,
    constraints: &SequentialTechMapConstraints,
    options: StaOptions,
) -> Result<()> {
    if constraints.primary_output_required.is_empty() {
        return Ok(());
    }

    let primary_launch = analyze_register_boundary_max_arrival_with_primary_input_arrivals(
        &mapped.module,
        mapped.nets.as_slice(),
        &mapped.interner,
        library,
        options,
        true,
        &[],
        &constraints.primary_input_arrivals,
    )
    .context("checking exact primary-input-to-output required times")?;

    let register_indices = mapped
        .module
        .instances
        .iter()
        .enumerate()
        .filter_map(|(index, instance)| {
            mapped
                .interner
                .resolve(instance.type_name)
                .and_then(|name| library.cells.iter().find(|cell| cell.name == name))
                .is_some_and(|cell| !cell.sequential.is_empty())
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let register_launch = if register_indices.is_empty() {
        None
    } else {
        Some(
            analyze_register_boundary_max_arrival(
                &mapped.module,
                mapped.nets.as_slice(),
                &mapped.interner,
                library,
                options,
                false,
                &register_indices,
            )
            .context("checking exact register-to-output required times")?,
        )
    };

    for (name, required) in &constraints.primary_output_required {
        let launch_timings = [
            ("primary-input", primary_launch.timing_for_output_bit(name)),
            (
                "register",
                register_launch
                    .as_ref()
                    .and_then(|report| report.timing_for_output_bit(name)),
            ),
        ];
        for (launch, timing) in launch_timings {
            let Some(timing) = timing else {
                // Constant outputs and unreachable launch classes have no
                // physical propagation path that can violate this deadline.
                continue;
            };
            let arrival = timing.rise.arrival.max(timing.fall.arrival);
            if arrival > required + REGISTER_SETUP_SLACK_EPSILON {
                bail!(
                    "primary output '{name}' violates required time {required}: exact {launch} launch arrival is {arrival}"
                );
            }
        }
    }

    Ok(())
}

/// Replaces transition-only estimates with exact full sequential STA and area.
fn finalize_sequential_mapping(
    mapped: &mut MappedNetlist,
    library: &Library,
    constraints: &SequentialTechMapConstraints,
    options: StaOptions,
) -> Result<()> {
    let report = build_netlist_report_with_primary_input_arrivals(
        &mapped.module,
        mapped.nets.as_slice(),
        &mapped.interner,
        library,
        options,
        &constraints.primary_input_arrivals,
    )
    .context("building exact register-aware mapped-netlist report")?;
    validate_exact_output_requirements(mapped, library, constraints, options)?;
    mapped.stats.selected_instance_count = report.cell_count;
    mapped.stats.selected_area = report.cell_area;
    mapped.stats.sequential_instance_count = mapped
        .module
        .instances
        .iter()
        .filter(|instance| {
            mapped
                .interner
                .resolve(instance.type_name)
                .and_then(|name| library.cells.iter().find(|cell| cell.name == name))
                .is_some_and(|cell| !cell.sequential.is_empty())
        })
        .count();
    mapped.stats.sequential_area = report.sequential_cell_area;
    mapped.stats.worst_input_to_register_arrival = report.max_input_to_register_delay;
    mapped.stats.worst_register_to_register_arrival = report.max_register_to_register_delay;
    mapped.stats.worst_register_to_output_arrival = report.max_register_to_output_delay;
    mapped.stats.clock_period = constraints.clock_period;
    let worst_capture = report
        .max_input_to_register_delay
        .into_iter()
        .chain(report.max_register_to_register_delay)
        .reduce(f64::max);
    let worst_register_slack = constraints
        .clock_period
        .zip(worst_capture)
        .map(|(period, arrival)| period - arrival);
    if let Some(slack) = worst_register_slack {
        if slack < -REGISTER_SETUP_SLACK_EPSILON {
            let period = constraints
                .clock_period
                .expect("register slack requires a clock period");
            let arrival = worst_capture.expect("register slack requires a capture arrival");
            bail!(
                "clock period {period} violates exact register setup timing: capture arrival {arrival} gives setup slack {slack}, below the -{REGISTER_SETUP_SLACK_EPSILON} tolerance"
            );
        }
    }
    mapped.stats.worst_register_slack = worst_register_slack;
    mapped.stats.worst_estimated_output_arrival = report
        .max_delay
        .into_iter()
        .chain(report.max_input_to_register_delay)
        .chain(report.max_register_to_register_delay)
        .chain(report.max_register_to_output_delay)
        .reduce(f64::max)
        .unwrap_or(0.0);
    mapped.stats.buffer_stats = None;
    mapped.stats.resize_stats = None;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{
        AigBitVector, AigOperand, ClockPort, RegisterBinding, TransitionInputId, TransitionOutputId,
    };
    use crate::aig_sim::sequential::{SequentialState, simulate};
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::liberty_model::{
        LibraryBuilder, LuTableTemplate, Pin, Sequential, SequentialKind, TimingArc, TimingTable,
    };
    use crate::liberty_proto::TimingTableKind;
    use crate::netlist::buffer::BufferOptions;
    use crate::netlist::emit::emit_module_as_netlist_text;
    use crate::netlist::gatefn_from_netlist::project_labeled_sequential_netlist_aig;
    use crate::netlist::resize::ResizeOptions;
    use crate::netlist::sequential_liberty::SequentialClockSpec;

    /// Builds one constant, fully characterized Liberty timing table.
    fn test_timing_table(
        builder: &mut LibraryBuilder,
        kind: TimingTableKind,
        value: f64,
    ) -> TimingTable {
        builder
            .add_timing_table_f64(kind, 0, vec![], vec![], vec![], vec![value], vec![], "")
            .expect("construct scalar sequential-mapper timing table")
    }

    /// Builds complete rise, fall, and slew characterization for one pin.
    fn test_output_arc(
        builder: &mut LibraryBuilder,
        related_pin: &str,
        sense: &str,
        timing_type: &str,
        delay: f64,
    ) -> TimingArc {
        let tables = vec![
            test_timing_table(builder, TimingTableKind::CellRise, delay),
            test_timing_table(builder, TimingTableKind::CellFall, delay),
            test_timing_table(builder, TimingTableKind::RiseTransition, 0.1),
            test_timing_table(builder, TimingTableKind::FallTransition, 0.1),
        ];
        builder
            .add_timing_arc(related_pin, sense, timing_type, "", tables)
            .expect("construct complete sequential-mapper output arc")
    }

    /// Builds a small, timing-complete positive-edge standard-cell library.
    fn test_library() -> Library {
        let mut builder = LibraryBuilder::new();
        let a = builder.intern_string("A").expect("intern A");
        let y = builder.intern_string("Y").expect("intern Y");
        let d = builder.intern_string("D").expect("intern D");
        let clock = builder.intern_string("CLK").expect("intern CLK");
        let q = builder.intern_string("Q").expect("intern Q");
        let identity = builder.intern_string("A").expect("intern identity");
        let inversion = builder.intern_string("!A").expect("intern inversion");
        let state = builder.intern_string("IQ").expect("intern state");
        let buffer_arc = test_output_arc(&mut builder, "A", "positive_unate", "combinational", 1.0);
        let inverter_arc =
            test_output_arc(&mut builder, "A", "negative_unate", "combinational", 1.0);
        let clock_arc = test_output_arc(&mut builder, "CLK", "non_unate", "rising_edge", 0.5);
        let setup_tables = vec![
            test_timing_table(&mut builder, TimingTableKind::RiseConstraint, 0.25),
            test_timing_table(&mut builder, TimingTableKind::FallConstraint, 0.25),
        ];
        let setup = builder
            .add_timing_arc("CLK", "", "setup_rising", "", setup_tables)
            .expect("construct setup arc");

        builder.cells = vec![
            Cell {
                name: "BUF".to_string(),
                pins: vec![
                    Pin {
                        name: a,
                        direction: PinDirection::Input as i32,
                        capacitance: Some(0.02),
                        ..Pin::default()
                    },
                    Pin {
                        name: y,
                        direction: PinDirection::Output as i32,
                        function: identity,
                        timing_arcs: vec![buffer_arc],
                        ..Pin::default()
                    },
                ],
                area: 1.0,
                ..Cell::default()
            },
            Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        name: a,
                        direction: PinDirection::Input as i32,
                        capacitance: Some(0.02),
                        ..Pin::default()
                    },
                    Pin {
                        name: y,
                        direction: PinDirection::Output as i32,
                        function: inversion,
                        timing_arcs: vec![inverter_arc],
                        ..Pin::default()
                    },
                ],
                area: 1.0,
                ..Cell::default()
            },
            Cell {
                name: "DFF".to_string(),
                pins: vec![
                    Pin {
                        name: d,
                        direction: PinDirection::Input as i32,
                        capacitance: Some(0.02),
                        timing_arcs: vec![setup],
                        ..Pin::default()
                    },
                    Pin {
                        name: clock,
                        direction: PinDirection::Input as i32,
                        is_clocking_pin: true,
                        capacitance: Some(0.01),
                        ..Pin::default()
                    },
                    Pin {
                        name: q,
                        direction: PinDirection::Output as i32,
                        function: state,
                        timing_arcs: vec![clock_arc],
                        ..Pin::default()
                    },
                ],
                area: 4.0,
                sequential: vec![Sequential {
                    state_var: "IQ".to_string(),
                    complementary_state_var: Some("IQN".to_string()),
                    next_state: "D".to_string(),
                    clock_expr: "CLK".to_string(),
                    kind: SequentialKind::Ff as i32,
                    ..Sequential::default()
                }],
                ..Cell::default()
            },
        ];
        builder.finish()
    }

    /// Makes flip-flop clock-to-Q observably depend on external output load.
    fn load_dependent_flip_flop_library() -> Library {
        let mut builder = LibraryBuilder::from_library(test_library());
        builder.lu_table_templates.push(LuTableTemplate {
            kind: "lu_table_template".to_string().into(),
            name: "seq_probe_output_load".to_string(),
            variable_1: "total_output_net_capacitance".to_string().into(),
            index_1: vec![0.0, 1.0],
            ..LuTableTemplate::default()
        });
        let mut tables = Vec::new();
        for (kind, values) in [
            (TimingTableKind::CellRise, vec![0.5, 10.5]),
            (TimingTableKind::CellFall, vec![0.5, 10.5]),
            (TimingTableKind::RiseTransition, vec![0.1, 0.2]),
            (TimingTableKind::FallTransition, vec![0.1, 0.2]),
        ] {
            tables.push(
                builder
                    .add_timing_table_f64(kind, 1, vec![], vec![], vec![], values, vec![2], "")
                    .expect("construct load-sensitive flip-flop timing table"),
            );
        }
        let clock_arc = builder
            .add_timing_arc("CLK", "non_unate", "rising_edge", "", tables)
            .expect("construct load-sensitive clock-to-Q arc");
        let q_name = builder.intern_string("Q").expect("intern Q");
        let cell = builder
            .cells
            .iter_mut()
            .find(|cell| cell.name == "DFF")
            .expect("find synthetic flip-flop");
        let output = cell
            .pins
            .iter_mut()
            .find(|pin| pin.name == q_name)
            .expect("find synthetic flip-flop state output");
        output.timing_arcs = vec![clock_arc];
        builder.finish()
    }

    /// Creates a two-bit registered transition with canonical external ports.
    fn test_design() -> SequentialGateFn {
        let mut builder = GateBuilder::new(
            "unit_pipeline__transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let data = builder.add_input("data".to_string(), 2);
        let q = builder.add_input("state__q".to_string(), 2);
        builder.add_output("out".to_string(), q);
        builder.add_output("state__d".to_string(), data);
        SequentialGateFn::new(
            "unit_pipeline".to_string(),
            builder.build(),
            vec![TransitionInputId::new(0)],
            vec![TransitionOutputId::new(0)],
            Some(ClockPort {
                name: "clk".to_string(),
            }),
            vec![RegisterBinding {
                name: "state".to_string(),
                q: TransitionInputId::new(1),
                d: TransitionOutputId::new(1),
                initial_value: None,
            }],
        )
        .expect("construct registered mapper test design")
    }

    /// Creates parsed Liberty state metadata without relying on pin spelling.
    fn state_spec(next_state: &str) -> GvEvalSequentialCellSpec {
        GvEvalSequentialCellSpec {
            state_var: "IQ".to_string(),
            complementary_state_var: Some("IQN".to_string()),
            next_state: parse_formula(next_state).expect("parse test next-state formula"),
            next_state_text: next_state.to_string(),
            clock: SequentialClockSpec {
                pin_name: "CLK".to_string(),
                is_negated: false,
            },
        }
    }

    #[test]
    fn recognizes_state_and_complementary_state_output_polarities() {
        let spec = state_spec("D");
        let positive = parse_formula("IQ").expect("parse state formula");
        let complementary = parse_formula("IQN").expect("parse complementary state formula");

        assert_eq!(state_formula_polarity(&positive, &spec), Some(false));
        assert_eq!(state_formula_polarity(&complementary, &spec), Some(true));
    }

    #[test]
    fn reduces_scan_and_enable_controls_to_a_deterministic_data_phase() {
        let scan = state_spec("(!D * !SE) + (SE * SI)");
        let scan_controls = BTreeMap::from([("SE".to_string(), false), ("SI".to_string(), false)]);
        assert_eq!(
            data_formula_polarity(&scan.next_state, &scan, "D", &scan_controls),
            Some(true)
        );

        let enabled = state_spec("(!EN * IQ) + (EN * D)");
        assert_eq!(
            data_formula_polarity(
                &enabled.next_state,
                &enabled,
                "D",
                &BTreeMap::from([("EN".to_string(), true)]),
            ),
            Some(false)
        );
        assert_eq!(
            data_formula_polarity(
                &enabled.next_state,
                &enabled,
                "D",
                &BTreeMap::from([("EN".to_string(), false)]),
            ),
            None
        );
    }

    #[test]
    fn register_data_phase_adjustment_preserves_structural_choice_links() {
        let mut builder = GateBuilder::new(
            "choice_transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let data: AigOperand = builder
            .add_input("data".to_string(), 1)
            .try_into()
            .expect("extract data bit");
        let q: AigOperand = builder
            .add_input("state__q".to_string(), 1)
            .try_into()
            .expect("extract state bit");
        let live = builder.add_and_binary(data, q);
        let sibling = builder.add_and_binary(data, q);
        builder.add_output("out".to_string(), q.into());
        builder.add_output("state__d".to_string(), live.into());
        let transition = builder.build();
        let design = SequentialGateFn::new(
            "choice_pipeline".to_string(),
            transition.clone(),
            vec![TransitionInputId::new(0)],
            vec![TransitionOutputId::new(0)],
            Some(ClockPort {
                name: "clk".to_string(),
            }),
            vec![RegisterBinding {
                name: "state".to_string(),
                q: TransitionInputId::new(1),
                d: TransitionOutputId::new(1),
                initial_value: None,
            }],
        )
        .expect("construct choice-rich registered design");
        let mut siblings = vec![None; transition.gates.len()];
        siblings[sibling.node.id] = Some(live.node);
        let choices = ChoiceAig::new(transition, siblings).expect("construct structural choices");

        let adjusted = adjust_register_data_phase(&design, &choices, true)
            .expect("adjust physical flip-flop data phase");

        assert_eq!(adjusted.sibling_links(), choices.sibling_links());
        assert_eq!(adjusted.sibling_link_count(), 1);
        assert_eq!(
            *adjusted.graph().outputs[1].bit_vector.get_lsb(0),
            choices.graph().outputs[1].bit_vector.get_lsb(0).negate()
        );
        assert_eq!(
            adjusted.graph().outputs[0].bit_vector.get_lsb(0),
            choices.graph().outputs[0].bit_vector.get_lsb(0)
        );
    }

    #[test]
    fn rejects_unknown_external_endpoint_constraints() {
        let design = test_design();
        let bad_input = SequentialTechMapConstraints {
            primary_input_arrivals: BTreeMap::from([("state__q_0".to_string(), 1.0)]),
            ..SequentialTechMapConstraints::default()
        };
        let input_error = validate_constraints(&design, &bad_input)
            .expect_err("internal register state must not be an external arrival");
        assert_eq!(
            input_error.to_string(),
            "timing constraint names unknown external primary input 'state__q_0'"
        );

        let bad_output = SequentialTechMapConstraints {
            primary_output_required: BTreeMap::from([("state__d_0".to_string(), 1.0)]),
            ..SequentialTechMapConstraints::default()
        };
        let output_error = validate_constraints(&design, &bad_output)
            .expect_err("internal register data must not be an external required endpoint");
        assert_eq!(
            output_error.to_string(),
            "timing constraint names unknown external primary output 'state__d_0'"
        );
    }

    #[test]
    fn rejects_nonpositive_and_nonfinite_clock_periods() {
        let design = test_design();
        for period in [0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let constraints = SequentialTechMapConstraints {
                clock_period: Some(period),
                ..SequentialTechMapConstraints::default()
            };
            let error = validate_constraints(&design, &constraints)
                .expect_err("sequential clock period must be finite and positive");
            assert!(error.to_string().contains("strictly positive"));
        }
    }

    #[test]
    fn rejects_nonfinite_external_endpoint_times() {
        let design = test_design();
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let input = SequentialTechMapConstraints {
                primary_input_arrivals: BTreeMap::from([("data_0".to_string(), value)]),
                ..SequentialTechMapConstraints::default()
            };
            let error =
                validate_constraints(&design, &input).expect_err("external arrival must be finite");
            assert!(error.to_string().contains("non-negative and finite"));

            let output = SequentialTechMapConstraints {
                primary_output_required: BTreeMap::from([("out_0".to_string(), value)]),
                ..SequentialTechMapConstraints::default()
            };
            let error = validate_constraints(&design, &output)
                .expect_err("external required time must be finite");
            assert!(error.to_string().contains("non-negative and finite"));
        }
    }

    #[test]
    fn rejects_negative_external_arrivals_and_required_times() {
        let design = test_design();
        let negative_arrival = SequentialTechMapConstraints {
            primary_input_arrivals: BTreeMap::from([("data_0".to_string(), -1.0)]),
            ..SequentialTechMapConstraints::default()
        };
        let arrival_error = validate_constraints(&design, &negative_arrival)
            .expect_err("negative primary-input arrival must be rejected before FF indexing");
        assert_eq!(
            arrival_error.to_string(),
            "primary input arrival for 'data_0' must be non-negative and finite; got -1"
        );

        let negative_required = SequentialTechMapConstraints {
            primary_output_required: BTreeMap::from([("out_0".to_string(), -1.0)]),
            ..SequentialTechMapConstraints::default()
        };
        let required_error = validate_constraints(&design, &negative_required)
            .expect_err("negative primary-output requirement must be rejected before FF indexing");
        assert_eq!(
            required_error.to_string(),
            "primary output required time for 'out_0' must be non-negative and finite; got -1"
        );
    }

    #[test]
    fn rejects_clock_period_shorter_than_characterized_flip_flop_setup() {
        let design = test_design();
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let constraints = SequentialTechMapConstraints {
            clock_period: Some(0.2),
            ..SequentialTechMapConstraints::default()
        };

        let error = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &test_library(),
            &constraints,
            &TechMapOptions::default(),
        )
        .expect_err("clock period below flip-flop setup cannot yield a valid required time");
        let diagnostic = format!("{error:#}");
        assert!(
            diagnostic.contains("setup requirement")
                && diagnostic.contains("non-negative register required time"),
            "unexpected infeasible setup diagnostic: {diagnostic}"
        );
    }

    #[test]
    fn characterizes_register_q_without_external_module_output_load() {
        let library = load_dependent_flip_flop_library();
        let candidates = index_flip_flops(
            &library,
            StaOptions {
                primary_input_transition: 0.01,
                module_output_load: 1.0,
            },
        )
        .expect("characterize an internally loaded flip-flop");
        let binding = candidates
            .iter()
            .find(|binding| binding.cell_name == "DFF")
            .expect("find load-sensitive synthetic flip-flop");

        assert!(
            (binding.clock_to_output - 0.5).abs() < 1e-12,
            "internal register Q incorrectly inherited the external output load: {}",
            binding.clock_to_output
        );
        assert!((binding.setup - 0.25).abs() < 1e-12);
    }

    #[test]
    fn rejects_exact_clock_setup_slack_below_numerical_tolerance() {
        let design = test_design();
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let library = test_library();
        let mut mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions::default(),
        )
        .expect("map an unconstrained sequential test design");
        let arrival = mapped
            .stats
            .worst_input_to_register_arrival
            .expect("synthetic design must have a register capture path");

        let near_constraints = SequentialTechMapConstraints {
            clock_period: Some(arrival - REGISTER_SETUP_SLACK_EPSILON / 2.0),
            ..SequentialTechMapConstraints::default()
        };
        finalize_sequential_mapping(
            &mut mapped,
            &library,
            &near_constraints,
            StaOptions::default(),
        )
        .expect("rounding-sized negative setup slack should remain within tolerance");
        assert!(
            mapped
                .stats
                .worst_register_slack
                .is_some_and(|slack| { slack < 0.0 && slack >= -REGISTER_SETUP_SLACK_EPSILON })
        );

        let violating_constraints = SequentialTechMapConstraints {
            clock_period: Some(arrival - 2.0 * REGISTER_SETUP_SLACK_EPSILON),
            ..SequentialTechMapConstraints::default()
        };
        let error = finalize_sequential_mapping(
            &mut mapped,
            &library,
            &violating_constraints,
            StaOptions::default(),
        )
        .expect_err("exact negative setup slack must reject the current flip-flop candidate");
        let diagnostic = error.to_string();
        assert!(
            diagnostic.contains("clock period")
                && diagnostic.contains("exact register setup")
                && diagnostic.contains("setup slack"),
            "unexpected exact setup violation diagnostic: {diagnostic}"
        );
    }

    #[test]
    fn checks_exact_output_required_times_for_register_and_input_launches() {
        let library = test_library();
        let register_design = test_design();
        let register_choices = ChoiceAig::without_choices(register_design.transition.clone());
        let register_mapped = map_sequential_choice_aig_to_netlist(
            &register_design,
            &register_choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions::default(),
        )
        .expect("map register-launched visible output");
        let register_arrival = register_mapped
            .stats
            .worst_register_to_output_arrival
            .expect("register output must have clock-to-Q timing");
        let register_constraints = SequentialTechMapConstraints {
            primary_output_required: BTreeMap::from([(
                "out_0".to_string(),
                register_arrival - 0.01,
            )]),
            ..SequentialTechMapConstraints::default()
        };
        let register_error = validate_exact_output_requirements(
            &register_mapped,
            &library,
            &register_constraints,
            StaOptions::default(),
        )
        .expect_err("exact register-to-output timing must satisfy its bit deadline");
        assert!(
            register_error.to_string().contains("register launch"),
            "unexpected register output deadline diagnostic: {register_error:#}"
        );

        let mut input_design = test_design();
        input_design.transition.outputs[0].bit_vector =
            input_design.transition.inputs[0].bit_vector.clone();
        let input_choices = ChoiceAig::without_choices(input_design.transition.clone());
        let input_mapped = map_sequential_choice_aig_to_netlist(
            &input_design,
            &input_choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions::default(),
        )
        .expect("map primary-input-launched visible output");
        let primary_arrivals =
            BTreeMap::from([("data_0".to_string(), 5.0), ("data_1".to_string(), 5.0)]);
        let primary_launch = analyze_register_boundary_max_arrival_with_primary_input_arrivals(
            &input_mapped.module,
            &input_mapped.nets,
            &input_mapped.interner,
            &library,
            StaOptions::default(),
            true,
            &[],
            &primary_arrivals,
        )
        .expect("analyze externally launched packed output");
        let output_timing = primary_launch
            .timing_for_output_bit("out_0")
            .expect("resolve packed output bit timing");
        let input_arrival = output_timing.rise.arrival.max(output_timing.fall.arrival);
        let input_constraints = SequentialTechMapConstraints {
            primary_input_arrivals: primary_arrivals,
            primary_output_required: BTreeMap::from([("out_0".to_string(), input_arrival - 0.01)]),
            ..SequentialTechMapConstraints::default()
        };
        let input_error = validate_exact_output_requirements(
            &input_mapped,
            &library,
            &input_constraints,
            StaOptions::default(),
        )
        .expect_err("exact primary-input-to-output timing must include source arrival");
        assert!(
            input_error.to_string().contains("primary-input launch"),
            "unexpected primary-input output deadline diagnostic: {input_error:#}"
        );
    }

    #[test]
    fn reports_renamed_or_missing_flattened_transition_bits() {
        let design = test_design();
        let mut renamed = design.transition.clone();
        renamed.inputs[0].name = "wrong".to_string();
        let error = validate_transition_interface(&design.transition, &renamed)
            .expect_err("optimized transition must preserve input names");
        assert_eq!(
            error.to_string(),
            "optimized transition input bit 0 is 'wrong_0'; expected 'data_0'"
        );

        let mut missing = design.transition.clone();
        missing.outputs.pop();
        let error = validate_transition_interface(&design.transition, &missing)
            .expect_err("optimized transition must preserve output bit count");
        assert_eq!(
            error.to_string(),
            "optimized transition output interface has 2 bits; original transition has 4"
        );
    }

    #[test]
    fn explicitly_rejects_register_unaware_buffering_and_resizing() {
        let design = test_design();
        let choices = ChoiceAig::without_choices(design.transition.clone());
        for options in [
            TechMapOptions {
                buffer_options: Some(BufferOptions::default()),
                ..TechMapOptions::default()
            },
            TechMapOptions {
                resize_options: Some(ResizeOptions::default()),
                ..TechMapOptions::default()
            },
        ] {
            let error = map_sequential_choice_aig_to_netlist(
                &design,
                &choices,
                &test_library(),
                &SequentialTechMapConstraints::default(),
                &options,
            )
            .expect_err("combinational-only optimization must not run on a sequential module");
            assert!(error.to_string().contains("combinational-only"));
        }
    }

    #[test]
    fn rejects_dont_use_negative_edge_asynchronous_and_latch_flip_flops() {
        for case in ["dont_use", "negative_edge", "asynchronous", "latch"] {
            let mut library = test_library();
            let cell = library
                .cells
                .iter_mut()
                .find(|cell| cell.name == "DFF")
                .expect("find synthetic flip-flop");
            match case {
                "dont_use" => cell.dont_use = Some(true),
                "negative_edge" => cell.sequential[0].clock_expr = "!CLK".to_string(),
                "asynchronous" => cell.sequential[0].clear_expr = "!RST".to_string(),
                "latch" => cell.sequential[0].kind = SequentialKind::Latch as i32,
                _ => unreachable!("all synthetic rejection cases are enumerated"),
            }
            let error = index_flip_flops(&library, StaOptions::default())
                .expect_err("unsupported sequential cells must not enter the FF index");
            assert!(
                error.to_string().contains("no usable positive-edge"),
                "unexpected {case} diagnostic: {error:#}"
            );
        }
    }

    #[test]
    fn maps_packed_constant_outputs_and_constant_register_inputs() {
        let mut builder = GateBuilder::new(
            "constant_pipeline__transition".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let _data = builder.add_input("data".to_string(), 2);
        let state = builder.add_input("state__q".to_string(), 2);
        let visible = AigBitVector::from_lsb_is_index_0(&[*state.get_lsb(0), builder.get_false()]);
        let next_state = builder
            .add_literal(&IrBits::make_ubits(2, 0b10).expect("construct constant register input"));
        builder.add_output("out".to_string(), visible);
        builder.add_output("state__d".to_string(), next_state);
        let design = SequentialGateFn::new(
            "constant_pipeline".to_string(),
            builder.build(),
            vec![TransitionInputId::new(0)],
            vec![TransitionOutputId::new(0)],
            Some(ClockPort {
                name: "clk".to_string(),
            }),
            vec![RegisterBinding {
                name: "state".to_string(),
                q: TransitionInputId::new(1),
                d: TransitionOutputId::new(1),
                initial_value: None,
            }],
        )
        .expect("construct constant-output registered design");
        let choices = ChoiceAig::without_choices(design.transition.clone());
        let library = test_library();

        let mapped = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &library,
            &SequentialTechMapConstraints::default(),
            &TechMapOptions::default(),
        )
        .expect("map packed constant outputs and directly tied register data");
        emit_module_as_netlist_text(&mapped.module, &mapped.nets, &mapped.interner)
            .expect("packed constant output must be serializable as a standard-cell netlist");
        assert_eq!(mapped.stats.sequential_instance_count, 2);
        assert!(mapped.module.assigns.is_empty());

        let projected = project_labeled_sequential_netlist_aig(
            &mapped.module,
            &mapped.nets,
            &mapped.interner,
            &library,
            Some("clk"),
        )
        .expect("project packed constant-output netlist");
        let inputs = [0, 1, 2]
            .into_iter()
            .map(|value| {
                vec![IrBits::make_ubits(2, value).expect("construct packed test stimulus")]
            })
            .collect::<Vec<_>>();
        let original = simulate(&design, &inputs, SequentialState::all_zeros(&design))
            .expect("simulate native constant-output transition");
        let actual = simulate(
            &projected.sequential_gate_fn,
            &inputs,
            SequentialState::all_zeros(&projected.sequential_gate_fn),
        )
        .expect("simulate physical constant-output netlist");
        assert_eq!(actual.external_outputs(), original.external_outputs());
    }

    #[test]
    fn avoids_collisions_when_naming_physical_register_instances() {
        let mut used = BTreeSet::from(["u_reg_state".to_string(), "u_reg_state__1".to_string()]);
        assert_eq!(
            unique_instance_name("u_reg_state", &mut used),
            "u_reg_state__2"
        );
        assert_eq!(
            unique_instance_name("u_reg_state", &mut used),
            "u_reg_state__3"
        );
    }
}
