// SPDX-License-Identifier: Apache-2.0

//! Electrical fanout of single-driver state and combinational data outputs.
//! Wiring aliases share loads, while each editable tap retains its original
//! Verilog pin binding.

use crate::liberty::cell_formula::{Term, parse_formula};
use crate::liberty_model::{Cell, Library, Pin, PinDirection, SequentialKind};
use crate::netlist::fanout_opt::pin_cap;
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::normalized::{BitExpr, BitIndex, BitSource, NormalizedNetlistModule};
use crate::netlist::parse::{NetRef, PortDirection, PortId};
use crate::netlist::sequential_liberty::classify_tied_ff_data;
use anyhow::{Result, anyhow, ensure};
use std::collections::{HashMap, HashSet};

/// A movable data pin, including the exact binding needed to validate an edit.
#[derive(Clone, Debug)]
pub(super) struct Sink {
    pub instance: usize,
    pub connection: usize,
    pub original: NetRef,
    pub capacitance: [f64; 2],
    pub fanout_load: f64,
}

/// A driver and its load split into movable sinks and connections that must
/// stay.
#[derive(Debug)]
pub(super) struct Root {
    pub instance: usize,
    pub connection: usize,
    pub cell: usize,
    pub source: NetRef,
    pub source_bit: crate::netlist::bit_ref::NetBit,
    pub instance_name: String,
    pub pin_name: String,
    pub sinks: Vec<Sink>,
    /// Rise/fall capacitance that remains physically connected after editing.
    pub fixed_load: [f64; 2],
    pub fixed_sink_count: usize,
    pub fixed_fanout_load: f64,
    pub max_fanout: Option<f64>,
    pub output_count: usize,
    pub max_capacitance: Option<f64>,
    pub max_transition: Option<f64>,
}

/// A legal non-inverting data buffer and the pin limits used to build a tree.
#[derive(Clone, Debug)]
pub(super) struct Buffer {
    pub cell: usize,
    pub input: String,
    pub output: String,
    pub input_cap: [f64; 2],
    pub input_fanout_load: f64,
    pub max_fanout: Option<f64>,
    pub max_capacitance: Option<f64>,
}

/// All edge-triggered instances, including uneditable launches. Timing must
/// compare the same population before and after each candidate edit.
pub(super) fn launches(netlist: &MappedNetlist, lib: &Library) -> Result<Vec<usize>> {
    let cells = indexed_cells(lib)?;
    let mut out = Vec::new();
    for (i, inst) in netlist.module.instances.iter().enumerate() {
        let name = netlist
            .interner
            .resolve(inst.type_name)
            .ok_or_else(|| anyhow!("unresolved cell type"))?;
        let cell = &lib.cells[*cells
            .get(name)
            .ok_or_else(|| anyhow!("missing Liberty cell {name}"))?];
        if cell
            .sequential
            .iter()
            .any(|seq| seq.kind == SequentialKind::Ff as i32)
        {
            out.push(i);
        }
    }
    Ok(out)
}

/// Collects freely movable scalar data taps and the loads that must remain on
/// their original source. A scalar pin may use a vector bit or a one-bit alias.
#[cfg(test)]
pub(super) fn roots(
    netlist: &MappedNetlist,
    lib: &Library,
    module_output_load: f64,
) -> Result<Vec<Root>> {
    collect_roots(netlist, lib, module_output_load, None)
}

/// Adds internal data drivers with at least the requested number of movable
/// pin loads. The standalone registered repair asks only for state outputs.
pub(super) fn collect_roots(
    netlist: &MappedNetlist,
    lib: &Library,
    module_output_load: f64,
    internal_threshold: Option<usize>,
) -> Result<Vec<Root>> {
    ensure!(
        module_output_load.is_finite() && module_output_load >= 0.,
        "invalid external load"
    );
    let n = NormalizedNetlistModule::new(&netlist.module, &netlist.nets, &netlist.interner)?;
    let count = n.bit_count();
    let mut parents = n.canonical_bits().to_vec();
    let mut written = vec![false; count];
    let mut unsafe_bits = Vec::new();
    let mut assigned_constants = Vec::new();
    // A directed assign drives its left-hand side. Only a direct bit copy can
    // be treated as an electrical alias. A compound expression has no Liberty
    // cell pins from which to obtain a physical load.
    for assign in &n.assigns {
        for (&dst, expr) in assign.lhs_bits.iter().zip(&assign.rhs_bits) {
            if std::mem::replace(&mut written[dst], true) {
                unsafe_bits.push(dst);
            }
            if let BitExpr::Source(BitSource::Bit(src)) = expr {
                let a = find(&mut parents, dst);
                let b = find(&mut parents, *src);
                if a == b {
                    unsafe_bits.push(dst);
                } else {
                    parents[a] = b;
                }
            } else if let BitExpr::Source(BitSource::Literal(level)) = expr {
                assigned_constants.push((dst, *level));
            } else {
                unsafe_bits.push(dst);
                expr.collect_source_bits(&mut unsafe_bits);
            }
        }
    }
    for bit in 0..count {
        parents[bit] = find(&mut parents, bit);
    }
    let group = |bit: BitIndex| parents[bit];
    let mut blocked = vec![false; count];
    let mut control: Vec<_> = unsafe_bits.into_iter().map(group).collect();
    for &g in &control {
        blocked[g] = true;
    }
    let mut outputs = HashMap::<usize, usize>::new();
    for port in &n.ports {
        for &bit in &port.bits {
            match port.direction {
                PortDirection::Output => *outputs.entry(group(bit)).or_default() += 1,
                PortDirection::Input | PortDirection::Inout => blocked[group(bit)] = true,
            }
        }
    }
    // Constant levels may specialize a scan mux only on unambiguously driven
    // bits. Every extra driver is detected as a second write to a canonical
    // alias destination, or as two primary sources for a common alias.
    let mut constants: HashMap<BitIndex, bool> = HashMap::new();
    for (bit, level) in assigned_constants {
        if constants.insert(group(bit), level).is_some() {
            blocked[group(bit)] = true;
        }
    }
    let catalog = pin_roles(netlist, lib, internal_threshold.is_some())?;
    // Large cones have many single-use gates. Allocate names, tap vectors and
    // observed STA bits only for nets with enough candidate pins.
    let mut tap_counts = vec![];
    if internal_threshold.is_some() {
        tap_counts.resize(count, 0_usize);
        for inst in &n.instances {
            for conn in &inst.connections {
                if let [BitSource::Bit(bit)] = conn.bits.as_slice()
                    && matches!(
                        catalog[&inst.type_name].pins.get(&conn.port),
                        Some(Role::Input { movable: true, .. })
                    )
                {
                    tap_counts[group(*bit)] += 1;
                }
            }
        }
    }
    let mut drivers = HashMap::new();
    let mut combinational_driver = HashMap::new();
    let mut seen = HashSet::new();
    for (i, inst) in n.instances.iter().enumerate() {
        let shape = &catalog[&inst.type_name];
        seen.clear();
        let duplicate_pin = inst.connections.iter().any(|c| !seen.insert(c.port));
        for (j, conn) in inst.connections.iter().enumerate() {
            let role = shape.pins.get(&conn.port);
            let [BitSource::Bit(bit)] = conn.bits.as_slice() else {
                // Liberty pins are scalar. A wider reference cannot be edited
                // or assigned capacitance to just one of its constituent bits.
                if conn.bits.len() > 1 {
                    for source in &conn.bits {
                        if let BitSource::Bit(b) = source {
                            blocked[group(*b)] = true;
                            control.push(group(*b));
                        }
                    }
                }
                continue;
            };
            let g = group(*bit);
            if duplicate_pin {
                blocked[g] = true;
                control.push(g);
            }
            match role {
                Some(Role::Input { control: true, .. }) => control.push(g),
                Some(Role::Input { .. }) => {}
                Some(Role::Output {
                    state,
                    internal_data,
                    constant,
                    max_capacitance,
                    max_transition,
                    max_fanout,
                }) => {
                    if std::mem::replace(&mut written[*bit], true) {
                        blocked[g] = true;
                    }
                    if let Some(level) = constant {
                        if constants.insert(g, *level).is_some() {
                            blocked[g] = true;
                        }
                    }
                    if shape.combinational {
                        ensure!(
                            combinational_driver.insert(g, i).is_none(),
                            "multiple combinational drivers for one electrical net"
                        );
                    }
                    if (*state && internal_threshold.is_none_or(|_| tap_counts[g] >= 2))
                        || (*internal_data
                            && internal_threshold.is_some_and(|min| tap_counts[g] >= min))
                    {
                        let root = Root {
                            instance: i,
                            connection: j,
                            cell: shape.cell,
                            source: netlist.module.instances[i].connections[j].1.clone(),
                            source_bit: n.bit(*bit),
                            instance_name: netlist
                                .interner
                                .resolve(inst.instance_name)
                                .ok_or_else(|| anyhow!("unresolved instance name"))?
                                .to_owned(),
                            pin_name: netlist
                                .interner
                                .resolve(conn.port)
                                .ok_or_else(|| anyhow!("unresolved output pin"))?
                                .to_owned(),
                            sinks: vec![],
                            fixed_load: [0.; 2],
                            fixed_sink_count: 0,
                            fixed_fanout_load: 0.,
                            max_fanout: *max_fanout,
                            output_count: outputs.get(&g).copied().unwrap_or(0),
                            max_capacitance: *max_capacitance,
                            max_transition: *max_transition,
                        };
                        if drivers.insert(g, root).is_some() {
                            blocked[g] = true;
                        }
                    }
                }
                None | Some(Role::Unknown) => {
                    blocked[g] = true;
                    control.push(g);
                }
            }
        }
    }
    constants.retain(|bit, _| !blocked[*bit]);
    // A conditional timing arc may be active at one gate and inactive at
    // another instance of the same master. Memoize by the actually bound
    // input bits, including tie-cell outputs and simple assigns.
    let mut specializations = HashMap::new();
    for inst in &n.instances {
        let shape = &catalog[&inst.type_name];
        if shape.combinational {
            continue;
        }
        let coverage = shape.conditional.as_ref().map(|next| {
            let mut low = 0u64;
            let mut high = 0u64;
            for conn in &inst.connections {
                let Some(&position) = next.positions.get(&conn.port) else {
                    continue;
                };
                let value = match conn.bits.as_slice() {
                    [BitSource::Literal(bit)] => Some(*bit),
                    [BitSource::Bit(bit)] => constants.get(&group(*bit)).copied(),
                    _ => None,
                };
                match value {
                    Some(false) => low |= 1u64 << position,
                    Some(true) => high |= 1u64 << position,
                    None => {}
                }
            }
            let cache = specializations
                .entry((shape.cell, low, high))
                .or_insert_with(|| {
                    let mut tied = HashMap::new();
                    for (i, name) in next.inputs.iter().enumerate() {
                        let mask = 1u64 << i;
                        if (high | low) & mask != 0 {
                            tied.insert(name.clone(), high & mask != 0);
                        }
                    }
                    let cell = &lib.cells[shape.cell];
                    let data = classify_tied_ff_data(&next.expr, &tied, &next.inputs)
                        .and_then(|name| netlist.interner.get(name));
                    let mut priced = HashSet::new();
                    for pin in &cell.pins {
                        let Some(port) = netlist.interner.get(lib.resolve_string(&pin.name)) else {
                            continue;
                        };
                        if (Some(port) == data
                            || matches!(
                                shape.pins.get(&port),
                                Some(Role::Output { state: true, .. })
                            ))
                            && crate::netlist::sta::has_complete_register_pin_timing(
                                lib, cell, pin, &tied,
                            )
                        {
                            priced.insert(port);
                        }
                    }
                    (data, priced)
                });
            &*cache
        });
        for conn in &inst.connections {
            let [BitSource::Bit(bit)] = conn.bits.as_slice() else {
                continue;
            };
            match shape.pins.get(&conn.port) {
                Some(Role::Output { state: true, .. })
                    if coverage.is_none_or(|(_, priced)| !priced.contains(&conn.port)) =>
                {
                    blocked[group(*bit)] = true
                }
                Some(Role::Input { .. })
                    if coverage.is_none_or(|(data, priced)| {
                        *data != Some(conn.port) || !priced.contains(&conn.port)
                    }) =>
                {
                    // Without both active setup edges this physical D can be
                    // absent from otherwise successful whole-module STA.
                    control.push(group(*bit));
                }
                _ => {}
            }
        }
    }
    // A generated clock/reset may reach its control pin through ordinary
    // inverters or buffers. Preserve the combinational fan-in to that pin.
    let mut visited = HashSet::new();
    while let Some(g) = control.pop() {
        if !visited.insert(g) {
            continue;
        }
        blocked[g] = true;
        if let Some(&i) = combinational_driver.get(&g) {
            let inst = &n.instances[i];
            let shape = &catalog[&inst.type_name];
            for conn in &inst.connections {
                if matches!(shape.pins.get(&conn.port), Some(Role::Input { .. })) {
                    for bit in &conn.bits {
                        if let BitSource::Bit(b) = bit {
                            control.push(group(*b));
                        }
                    }
                }
            }
        }
    }
    drivers.retain(|g, _| !blocked[*g]);
    for driver in drivers.values_mut() {
        driver.fixed_load = [module_output_load * driver.output_count as f64; 2];
    }
    for (i, inst) in n.instances.iter().enumerate() {
        let shape = &catalog[&inst.type_name];
        for (j, conn) in inst.connections.iter().enumerate() {
            let [BitSource::Bit(bit)] = conn.bits.as_slice() else {
                continue;
            };
            let Some(driver) = drivers.get_mut(&group(*bit)) else {
                continue;
            };
            if let Some(Role::Input {
                movable,
                cap,
                fanout_load,
                ..
            }) = shape.pins.get(&conn.port)
            {
                if *movable {
                    driver.sinks.push(Sink {
                        instance: i,
                        connection: j,
                        original: netlist.module.instances[i].connections[j].1.clone(),
                        capacitance: *cap,
                        fanout_load: *fanout_load,
                    });
                } else {
                    driver.fixed_load[0] += cap[0];
                    driver.fixed_load[1] += cap[1];
                    driver.fixed_sink_count += 1;
                    driver.fixed_fanout_load += fanout_load;
                }
            }
        }
    }
    let mut roots: Vec<_> = drivers.into_values().collect();
    roots.sort_by_key(|root| (root.instance, root.connection));
    Ok(roots)
}

/// Path halving keeps long generated port-alias chains off the call stack.
fn find(parents: &mut [usize], mut bit: usize) -> usize {
    while parents[bit] != bit {
        parents[bit] = parents[parents[bit]];
        bit = parents[bit];
    }
    bit
}

enum Role {
    Input {
        movable: bool,
        control: bool,
        cap: [f64; 2],
        fanout_load: f64,
    },
    Output {
        state: bool,
        internal_data: bool,
        constant: Option<bool>,
        max_fanout: Option<f64>,
        max_capacitance: Option<f64>,
        max_transition: Option<f64>,
    },
    Unknown,
}

struct Shape {
    cell: usize,
    combinational: bool,
    pins: HashMap<PortId, Role>,
    conditional: Option<ConditionalData>,
}

struct ConditionalData {
    expr: Term,
    inputs: Vec<String>,
    positions: HashMap<PortId, usize>,
}

fn indexed_cells(lib: &Library) -> Result<HashMap<&str, usize>> {
    let mut cells = HashMap::with_capacity(lib.cells.len());
    for (i, cell) in lib.cells.iter().enumerate() {
        ensure!(
            cells.insert(cell.name.as_str(), i).is_none(),
            "duplicate Liberty master {}",
            cell.name
        );
    }
    Ok(cells)
}

/// Decode each used master's pin functions and characterization once per graph.
fn pin_roles(
    netlist: &MappedNetlist,
    lib: &Library,
    include_internal: bool,
) -> Result<HashMap<PortId, Shape>> {
    let cells = indexed_cells(lib)?;
    let mut shapes = HashMap::new();
    for inst in &netlist.module.instances {
        if shapes.contains_key(&inst.type_name) {
            continue;
        }
        let name = netlist
            .interner
            .resolve(inst.type_name)
            .ok_or_else(|| anyhow!("unresolved cell type"))?;
        let i = *cells
            .get(name)
            .ok_or_else(|| anyhow!("missing Liberty cell {name}"))?;
        let cell = &lib.cells[i];
        let internal = include_internal
            && cell.sequential.is_empty()
            && cell.clock_gate.is_none()
            && !cell.pins.iter().any(|p| p.is_clocking_pin)
            && cell.validate_functional_semantics().is_ok();
        let mut pins = HashMap::new();
        for pin in &cell.pins {
            let name = lib.resolve_string(&pin.name);
            let Some(sym) = netlist.interner.get(name) else {
                continue;
            };
            let role = if pin.direction == PinDirection::Output as i32 {
                let state = state_output(lib, cell, pin);
                let internal_data = internal
                    && parse_formula(lib.resolve_string(&pin.function))
                        .is_ok_and(|term| !term.inputs().is_empty());
                Role::Output {
                    state,
                    internal_data,
                    constant: if cell.sequential.is_empty()
                        && cell.clock_gate.is_none()
                        && cell.validate_functional_semantics().is_ok()
                    {
                        parse_formula(lib.resolve_string(&pin.function))
                            .ok()
                            .and_then(|term| term.evaluate_partial(&HashMap::new()))
                    } else {
                        None
                    },
                    max_fanout: pin.max_fanout,
                    max_capacitance: if state || internal_data {
                        limit(lib, pin)
                    } else {
                        None
                    },
                    max_transition: pin.max_transition,
                }
            } else if pin.direction == PinDirection::Input as i32 {
                let cap = pin_cap(pin);
                let fanout_load = pin.fanout_load.unwrap_or(1.0);
                if cap.iter().all(|v| v.is_finite() && *v >= 0.)
                    && fanout_load.is_finite()
                    && fanout_load >= 0.
                {
                    Role::Input {
                        movable: potential_data_sink(lib, cell, pin),
                        control: control_sink(lib, cell, pin),
                        cap,
                        fanout_load,
                    }
                } else {
                    Role::Unknown
                }
            } else {
                Role::Unknown
            };
            ensure!(
                pins.insert(sym, role).is_none(),
                "duplicate Liberty pin {name} on {}",
                cell.name
            );
        }
        let conditional = match cell.sequential.as_slice() {
            [seq]
                if seq.kind == SequentialKind::Ff as i32
                    && matches!(
                        crate::netlist::sequential_liberty::get_sequential_ff_clock_spec(cell, lib),
                        Ok(Some(_))
                    ) =>
            {
                parse_formula(&seq.next_state).ok().and_then(|expr| {
                    let inputs: Vec<_> = cell
                        .pins
                        .iter()
                        .filter(|pin| pin.direction == PinDirection::Input as i32)
                        .map(|pin| lib.resolve_string(&pin.name).to_owned())
                        .collect();
                    // Larger or unsupported transition functions stay
                    // protected.
                    if inputs.len() > u64::BITS as usize {
                        return None;
                    }
                    let positions = inputs
                        .iter()
                        .enumerate()
                        .filter_map(|(i, name)| netlist.interner.get(name.as_str()).map(|s| (s, i)))
                        .collect();
                    Some(ConditionalData {
                        expr,
                        inputs,
                        positions,
                    })
                })
            }
            _ => None,
        };
        shapes.insert(
            inst.type_name,
            Shape {
                cell: i,
                combinational: cell.sequential.is_empty(),
                pins,
                conditional,
            },
        );
    }
    Ok(shapes)
}

/// Only offer newly instantiable non-inverting cells whose input-to-output
/// timing the same STA implementation can independently evaluate.
pub(super) fn buffers(
    lib: &Library,
    prepared: &crate::netlist::sta::PreparedStaLibrary<'_>,
) -> Vec<Buffer> {
    let mut options = (0..lib.cells.len())
        .filter_map(|idx| buffer_model(lib, idx))
        .filter(|buffer| {
            prepared
                .validate_independent_combinational_cell(buffer.cell)
                .is_ok()
        })
        .collect::<Vec<_>>();
    options.sort_by(|a, b| {
        let (a, b) = (&lib.cells[a.cell], &lib.cells[b.cell]);
        a.area.total_cmp(&b.area).then_with(|| a.name.cmp(&b.name))
    });
    options
}

pub(super) fn buffer_model(lib: &Library, cell: usize) -> Option<Buffer> {
    let candidate = lib.cells.get(cell)?;
    if candidate.pins.len() != 2
        || candidate.use_for_size_only == Some(true)
        || !candidate.area.is_finite()
        || candidate.area <= 0.0
        || candidate.validate_mapping_policy_coverage().is_err()
    {
        return None;
    }
    // Resolves a unique cell and checks the exact truth table, functional
    // coverage, dont_use, clock pins and both port directions.
    let input = crate::liberty::io_defaults::buffer_input_pin(lib, &candidate.name).ok()?;
    let output = candidate
        .pins
        .iter()
        .find(|p| p.direction == PinDirection::Output as i32)?;
    let cap = pin_cap(input);
    if !cap.iter().all(|c| c.is_finite() && *c >= 0.0)
        || !input.fanout_load.is_none_or(|v| v.is_finite() && v >= 0.)
        || !output.max_fanout.is_none_or(|v| v.is_finite() && v > 0.)
    {
        return None;
    }
    Some(Buffer {
        cell,
        input: lib.resolve_string(&input.name).to_owned(),
        output: lib.resolve_string(&output.name).to_owned(),
        input_cap: cap,
        input_fanout_load: input.fanout_load.unwrap_or(1.0),
        max_fanout: output.max_fanout,
        max_capacitance: limit(lib, output),
    })
}

/// Intersect the explicit electrical limit with the characterized load range
/// of each delay and slew table. A scalar or slew-only table adds no limit.
fn limit(lib: &Library, pin: &Pin) -> Option<f64> {
    let mut bound = pin.max_capacitance;
    for table in pin.timing_arcs.iter().flat_map(|arc| &arc.tables) {
        if !matches!(
            table.kind_str(),
            "cell_rise" | "cell_fall" | "rise_transition" | "fall_transition"
        ) {
            continue;
        }
        let shape = lib.timing_table_shape(table);
        let template = shape
            .template_id
            .checked_sub(1)
            .and_then(|i| lib.lu_table_templates.get(i as usize));
        let explicit = lib.timing_table_axes(table);
        for (axis, &knots) in explicit.iter().enumerate().take(shape.dimensions.len()) {
            let (var, inherited): (&str, &[f64]) = match (template, axis) {
                (Some(t), 0) => (t.variable_1_str(lib), &t.index_1),
                (Some(t), 1) => (t.variable_2_str(lib), &t.index_2),
                (Some(t), 2) => (t.variable_3_str(lib), &t.index_3),
                _ => ("", &[]),
            };
            // Same named and unnamed-axis conventions as liberty::lut.
            if matches!(
                var,
                "total_output_net_capacitance"
                    | "related_out_total_output_net_capacitance"
                    | "output_net_pin_cap"
                    | "related_out_output_net_pin_cap"
            ) || (var.is_empty() && axis == 1)
            {
                let knots = if knots.is_empty() { inherited } else { knots };
                if let Some(&hi) = knots.last() {
                    bound = Some(bound.map_or(hi, |b| b.min(hi)));
                }
            }
        }
    }
    bound
}

/// The output has to expose Q or its complement. The FF may implement a scan
/// mux in next_state; that changes its captured value, not its state outputs.
fn state_output(lib: &Library, cell: &Cell, pin: &Pin) -> bool {
    use crate::liberty::cell_formula::{Term, parse_formula};
    if pin.direction != PinDirection::Output as i32
        || cell.clock_gate.is_some()
        || !matches!(
            crate::netlist::sequential_liberty::get_sequential_ff_clock_spec(cell, lib),
            Ok(Some(_))
        )
    {
        return false;
    }
    let seq = &cell.sequential[0];
    if seq.state_var.is_empty() || parse_formula(&seq.next_state).is_err() {
        return false;
    }
    let Ok(term) = parse_formula(lib.resolve_string(&pin.function)) else {
        return false;
    };
    let signal = match &term {
        Term::Input(name) => name,
        Term::Negate(inner) => match inner.as_ref() {
            Term::Input(name) => name,
            _ => return false,
        },
        _ => return false,
    };
    signal == &seq.state_var || seq.complementary_state_var.as_deref() == Some(signal.as_str())
}

/// Control roles come from Liberty, including the operands of asynchronous
/// control expressions. A pin name is never enough to mark a clock or reset.
fn control_sink(lib: &Library, cell: &Cell, pin: &Pin) -> bool {
    use crate::liberty::cell_formula::parse_formula;
    if pin.direction != PinDirection::Input as i32 {
        return false;
    }
    let name = lib.resolve_string(&pin.name);
    if pin.is_clocking_pin
        || cell.clock_gate.as_ref().is_some_and(|gate| {
            gate.clock_pin == name
                || gate
                    .enable_pins
                    .iter()
                    .chain(&gate.test_pins)
                    .any(|p| p == name)
        })
    {
        return true;
    }
    if matches!(crate::netlist::sequential_liberty::get_sequential_ff_clock_spec(cell, lib),
                Ok(Some(spec)) if spec.pin_name == name)
    {
        return true;
    }
    cell.sequential.iter().any(|seq| {
        [&seq.clock_expr, &seq.clear_expr, &seq.preset_expr]
            .into_iter()
            .filter(|expr| !expr.is_empty())
            .any(|expr| {
                parse_formula(expr).is_ok_and(|term| term.inputs().iter().any(|p| p == name))
            })
    })
}

/// Preselect taps before instance ties are known. Each sequential input that
/// fails the instance data and timing checks blocks its entire driver cone
/// before any movable sinks are collected.
fn potential_data_sink(lib: &Library, cell: &Cell, pin: &Pin) -> bool {
    if pin.direction != PinDirection::Input as i32
        || cell.clock_gate.is_some()
        || cell.validate_functional_semantics().is_err()
        || control_sink(lib, cell, pin)
    {
        return false;
    }
    let name = lib.resolve_string(&pin.name);
    if cell.sequential.is_empty() {
        return !cell.pins.iter().any(|p| p.is_clocking_pin)
            && cell
                .pins
                .iter()
                .filter(|p| p.direction == PinDirection::Output as i32)
                .any(|out| {
                    parse_formula(lib.resolve_string(&out.function))
                        .is_ok_and(|term| term.inputs().iter().any(|p| p == name))
                });
    }
    true
}
