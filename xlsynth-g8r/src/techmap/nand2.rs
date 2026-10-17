// SPDX-License-Identifier: Apache-2.0

//! Structural technology mapping from `GateFn` (AIG) to gate-level netlists.
//!
//! The initial mapper is intentionally simple and deterministic:
//! - Each AIG `And2` node maps to `NAND2` followed by `INV`.
//! - Negated operands are materialized via explicit `INV` edge inverters.
//! - Output negations are materialized via explicit `INV` instances.
//! - Current scope is limited to direct structural lowering, not general
//!   cell-covering or optimization across wider cell libraries.
//!
//! Output is provided in parsed-netlist data structures so later flows (e.g.
//! STA) can consume it directly without reparsing text.

use crate::aig::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::liberty::cell_formula::{Term, parse_formula};
use crate::liberty_model::{Cell, Library, PinDirection};
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::parse::{
    Net, NetIndex, NetRef, NetlistInstance, NetlistModule, NetlistPort, PortDirection,
};
use anyhow::{Result, anyhow};
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};
use xlsynth_pir::IrBits;

/// Explicit cell and pin bindings for structural NAND2/INV lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralTechMapOptions {
    pub module_name: Option<String>,
    pub nand2_cell_name: String,
    pub nand2_input_pin_names: [String; 2],
    pub nand2_output_pin_name: String,
    pub inv_cell_name: String,
    pub inv_input_pin_name: String,
    pub inv_output_pin_name: String,
}

impl Default for StructuralTechMapOptions {
    /// Uses canonical INV/NAND2 cell names and A/B/Y pin bindings.
    fn default() -> Self {
        Self {
            module_name: None,
            nand2_cell_name: "NAND2".to_string(),
            nand2_input_pin_names: ["A".to_string(), "B".to_string()],
            nand2_output_pin_name: "Y".to_string(),
            inv_cell_name: "INV".to_string(),
            inv_input_pin_name: "A".to_string(),
            inv_output_pin_name: "Y".to_string(),
        }
    }
}

/// A mapped scalar signal or a constant awaiting structural lowering.
#[derive(Clone, Copy, Debug)]
enum TechSignal {
    Net(NetIndex),
    Literal(bool),
}

/// An output port bit and the AIG operand that must drive it.
#[derive(Clone, Debug)]
struct OutputBitPlan {
    port_name: String,
    port_net: NetIndex,
    operand: AigOperand,
}

/// Names a scalar port, adding a bit suffix only for multi-bit vectors.
fn scalar_bit_name(base: &str, bit_index: usize, bit_count: usize) -> String {
    if bit_count == 1 {
        base.to_string()
    } else {
        format!("{}_{}", base, bit_index)
    }
}

/// Creates a one-bit netlist literal for an internal signal.
fn literal_netref(value: bool) -> NetRef {
    NetRef::Literal(IrBits::make_ubits(1, if value { 1 } else { 0 }).expect("1-bit literal"))
}

/// Converts a resolved structural signal into a netlist reference.
fn signal_to_netref(signal: TechSignal) -> NetRef {
    match signal {
        TechSignal::Net(idx) => NetRef::Simple(idx),
        TechSignal::Literal(value) => literal_netref(value),
    }
}

/// Verifies that literal propagation has removed constants from mapped cell
/// inputs before structural lowering starts.
fn verify_folded_literal_operands(gate_fn: &GateFn) -> Result<()> {
    for (gate_id, node) in gate_fn.gates.iter().enumerate() {
        let AigNode::And2 { a, b, .. } = node else {
            continue;
        };
        for (operand_name, operand) in [("a", a), ("b", b)] {
            if matches!(gate_fn.get(operand.node), AigNode::Literal { .. }) {
                return Err(anyhow!(
                    "AIG literal operand reached structural tech mapping at And2 gate {} input {}; constants must be folded before mapping",
                    gate_id,
                    operand_name
                ));
            }
        }
    }
    Ok(())
}

/// Finds and reserves an internal name that cannot collide with a port.
fn fresh_internal_net_name(base: String, used_net_names: &mut HashSet<String>) -> String {
    if used_net_names.insert(base.clone()) {
        return base;
    }
    let mut suffix = 1usize;
    loop {
        let candidate = format!("{}__{}", base, suffix);
        if used_net_names.insert(candidate.clone()) {
            return candidate;
        }
        suffix += 1;
    }
}

#[allow(clippy::too_many_arguments)]
/// Interns a net and keeps port and internal-wire membership disjoint.
fn ensure_net(
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    nets: &mut Vec<Net>,
    net_index_by_sym: &mut HashMap<SymbolU32, NetIndex>,
    port_net_indices: &mut HashSet<NetIndex>,
    wire_net_indices: &mut Vec<NetIndex>,
    wire_net_set: &mut HashSet<NetIndex>,
    name: &str,
    is_port: bool,
) -> NetIndex {
    let sym = interner.get_or_intern(name);
    let idx = if let Some(existing) = net_index_by_sym.get(&sym).copied() {
        existing
    } else {
        let created = NetIndex(nets.len());
        nets.push(Net {
            name: sym,
            width: None,
        });
        net_index_by_sym.insert(sym, created);
        created
    };

    if is_port {
        if port_net_indices.insert(idx) && wire_net_set.remove(&idx) {
            wire_net_indices.retain(|x| *x != idx);
        }
    } else if !port_net_indices.contains(&idx) && wire_net_set.insert(idx) {
        wire_net_indices.push(idx);
    }

    idx
}

/// Appends a cell instance with deterministically sorted pin connections.
fn add_instance(
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    instances: &mut Vec<NetlistInstance>,
    type_name: &str,
    instance_name: &str,
    connections: Vec<(&str, NetRef)>,
) {
    let type_sym = interner.get_or_intern(type_name);
    let instance_sym = interner.get_or_intern(instance_name);

    let mut conn_vec: Vec<(SymbolU32, NetRef)> = connections
        .into_iter()
        .map(|(pin_name, netref)| (interner.get_or_intern(pin_name), netref))
        .collect();

    conn_vec.sort_by(|(a, _), (b, _)| {
        interner
            .resolve(*a)
            .unwrap_or("")
            .cmp(interner.resolve(*b).unwrap_or(""))
    });

    instances.push(NetlistInstance {
        type_name: type_sym,
        instance_name: instance_sym,
        connections: conn_vec,
        inst_lineno: 1,
        inst_colno: 1,
    });
}

/// Resolves a literal or a previously mapped AIG node to its signal.
fn resolve_node_signal(
    gate_fn: &GateFn,
    node_signals: &[Option<NetIndex>],
    node: AigRef,
) -> Result<TechSignal> {
    match gate_fn.get(node) {
        AigNode::Literal { value, .. } => Ok(TechSignal::Literal(*value)),
        AigNode::Input { .. } | AigNode::And2 { .. } => {
            let Some(net_idx) = node_signals.get(node.id).and_then(|x| *x) else {
                return Err(anyhow!("unmapped node signal for AIG node {}", node.id));
            };
            Ok(TechSignal::Net(net_idx))
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Emits an explicit edge inverter when a cell operand is negated.
fn materialize_operand_signal_for_cell_input(
    gate_fn: &GateFn,
    node_signals: &[Option<NetIndex>],
    operand: AigOperand,
    edge_inv_counter: &mut usize,
    options: &StructuralTechMapOptions,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    nets: &mut Vec<Net>,
    net_index_by_sym: &mut HashMap<SymbolU32, NetIndex>,
    port_net_indices: &mut HashSet<NetIndex>,
    wire_net_indices: &mut Vec<NetIndex>,
    wire_net_set: &mut HashSet<NetIndex>,
    used_net_names: &mut HashSet<String>,
    instances: &mut Vec<NetlistInstance>,
) -> Result<TechSignal> {
    let base = resolve_node_signal(gate_fn, node_signals, operand.node)?;
    if !operand.negated {
        return Ok(base);
    }

    match base {
        TechSignal::Literal(v) => Ok(TechSignal::Literal(!v)),
        TechSignal::Net(src) => {
            let inv_id = *edge_inv_counter;
            *edge_inv_counter += 1;

            let out_name =
                fresh_internal_net_name(format!("n_inv_edge_{}", inv_id), used_net_names);
            let out_net = ensure_net(
                interner,
                nets,
                net_index_by_sym,
                port_net_indices,
                wire_net_indices,
                wire_net_set,
                out_name.as_str(),
                /* is_port= */ false,
            );
            add_instance(
                interner,
                instances,
                options.inv_cell_name.as_str(),
                format!("u_inv_edge_{}", inv_id).as_str(),
                vec![
                    (options.inv_input_pin_name.as_str(), NetRef::Simple(src)),
                    (
                        options.inv_output_pin_name.as_str(),
                        NetRef::Simple(out_net),
                    ),
                ],
            );
            Ok(TechSignal::Net(out_net))
        }
    }
}

/// Lowers each AND to NAND2 plus INV, retaining explicit edge/output
/// inversions.
pub fn map_gatefn_to_structural_netlist(
    gate_fn: &GateFn,
    options: &StructuralTechMapOptions,
) -> Result<MappedNetlist> {
    gate_fn.check_invariants_with_debug_assert();
    verify_folded_literal_operands(gate_fn)?;

    let module_name = options
        .module_name
        .clone()
        .unwrap_or_else(|| gate_fn.name.clone());

    let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
    let module_name_sym = interner.get_or_intern(module_name.as_str());

    let mut nets: Vec<Net> = Vec::new();
    let mut net_index_by_sym: HashMap<SymbolU32, NetIndex> = HashMap::new();
    let mut port_net_indices: HashSet<NetIndex> = HashSet::new();
    let mut wire_net_indices: Vec<NetIndex> = Vec::new();
    let mut wire_net_set: HashSet<NetIndex> = HashSet::new();
    let mut ports: Vec<NetlistPort> = Vec::new();
    let mut output_plans: Vec<OutputBitPlan> = Vec::new();
    let mut port_names_seen: HashSet<String> = HashSet::new();
    let mut used_net_names: HashSet<String> = HashSet::new();

    let mut node_signals: Vec<Option<NetIndex>> = vec![None; gate_fn.gates.len()];

    for input in &gate_fn.inputs {
        let bit_count = input.get_bit_count();
        for (bit_index, bit) in input.bit_vector.iter_lsb_to_msb().enumerate() {
            let port_name = scalar_bit_name(input.name.as_str(), bit_index, bit_count);
            if !port_names_seen.insert(port_name.clone()) {
                return Err(anyhow!(
                    "duplicate mapped port name '{}' in input set",
                    port_name
                ));
            }
            used_net_names.insert(port_name.clone());
            let port_sym = interner.get_or_intern(port_name.as_str());
            let net_idx = ensure_net(
                &mut interner,
                &mut nets,
                &mut net_index_by_sym,
                &mut port_net_indices,
                &mut wire_net_indices,
                &mut wire_net_set,
                port_name.as_str(),
                /* is_port= */ true,
            );
            ports.push(NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: port_sym,
            });
            if bit.node.id >= node_signals.len() {
                return Err(anyhow!(
                    "input bit references out-of-range node {}",
                    bit.node.id
                ));
            }
            node_signals[bit.node.id] = Some(net_idx);
        }
    }

    for output in &gate_fn.outputs {
        let bit_count = output.get_bit_count();
        for (bit_index, bit) in output.bit_vector.iter_lsb_to_msb().enumerate() {
            let port_name = scalar_bit_name(output.name.as_str(), bit_index, bit_count);
            if !port_names_seen.insert(port_name.clone()) {
                return Err(anyhow!(
                    "duplicate mapped port name '{}' in output set",
                    port_name
                ));
            }
            used_net_names.insert(port_name.clone());
            let port_sym = interner.get_or_intern(port_name.as_str());
            let net_idx = ensure_net(
                &mut interner,
                &mut nets,
                &mut net_index_by_sym,
                &mut port_net_indices,
                &mut wire_net_indices,
                &mut wire_net_set,
                port_name.as_str(),
                /* is_port= */ true,
            );
            ports.push(NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: port_sym,
            });
            output_plans.push(OutputBitPlan {
                port_name,
                port_net: net_idx,
                operand: *bit,
            });
        }
    }

    let mut owner_output_name_by_node: HashMap<usize, String> = HashMap::new();
    for plan in &output_plans {
        if plan.operand.negated {
            continue;
        }
        if matches!(gate_fn.get(plan.operand.node), AigNode::Literal { .. }) {
            continue;
        }
        owner_output_name_by_node
            .entry(plan.operand.node.id)
            .or_insert_with(|| plan.port_name.clone());
    }

    let mut instances: Vec<NetlistInstance> = Vec::new();
    let mut edge_inv_counter = 0usize;

    for (gate_id, node) in gate_fn.gates.iter().enumerate() {
        match node {
            AigNode::Input { .. } | AigNode::Literal { .. } => {}
            AigNode::And2 { a, b, .. } => {
                let a_signal = materialize_operand_signal_for_cell_input(
                    gate_fn,
                    node_signals.as_slice(),
                    *a,
                    &mut edge_inv_counter,
                    options,
                    &mut interner,
                    &mut nets,
                    &mut net_index_by_sym,
                    &mut port_net_indices,
                    &mut wire_net_indices,
                    &mut wire_net_set,
                    &mut used_net_names,
                    &mut instances,
                )?;
                let b_signal = materialize_operand_signal_for_cell_input(
                    gate_fn,
                    node_signals.as_slice(),
                    *b,
                    &mut edge_inv_counter,
                    options,
                    &mut interner,
                    &mut nets,
                    &mut net_index_by_sym,
                    &mut port_net_indices,
                    &mut wire_net_indices,
                    &mut wire_net_set,
                    &mut used_net_names,
                    &mut instances,
                )?;

                let nand_out_name =
                    fresh_internal_net_name(format!("n_nand_g{}", gate_id), &mut used_net_names);
                let nand_out_net = ensure_net(
                    &mut interner,
                    &mut nets,
                    &mut net_index_by_sym,
                    &mut port_net_indices,
                    &mut wire_net_indices,
                    &mut wire_net_set,
                    nand_out_name.as_str(),
                    /* is_port= */ false,
                );
                add_instance(
                    &mut interner,
                    &mut instances,
                    options.nand2_cell_name.as_str(),
                    format!("u_nand_g{}", gate_id).as_str(),
                    vec![
                        (
                            options.nand2_input_pin_names[0].as_str(),
                            signal_to_netref(a_signal),
                        ),
                        (
                            options.nand2_input_pin_names[1].as_str(),
                            signal_to_netref(b_signal),
                        ),
                        (
                            options.nand2_output_pin_name.as_str(),
                            NetRef::Simple(nand_out_net),
                        ),
                    ],
                );

                let node_out_name = owner_output_name_by_node
                    .get(&gate_id)
                    .cloned()
                    .unwrap_or_else(|| {
                        fresh_internal_net_name(format!("n_g{}", gate_id), &mut used_net_names)
                    });
                let node_out_net = ensure_net(
                    &mut interner,
                    &mut nets,
                    &mut net_index_by_sym,
                    &mut port_net_indices,
                    &mut wire_net_indices,
                    &mut wire_net_set,
                    node_out_name.as_str(),
                    /* is_port= */ false,
                );
                add_instance(
                    &mut interner,
                    &mut instances,
                    options.inv_cell_name.as_str(),
                    format!("u_inv_g{}", gate_id).as_str(),
                    vec![
                        (
                            options.inv_input_pin_name.as_str(),
                            NetRef::Simple(nand_out_net),
                        ),
                        (
                            options.inv_output_pin_name.as_str(),
                            NetRef::Simple(node_out_net),
                        ),
                    ],
                );

                node_signals[gate_id] = Some(node_out_net);
            }
        }
    }

    for (output_index, output) in output_plans.iter().enumerate() {
        let base_signal =
            resolve_node_signal(gate_fn, node_signals.as_slice(), output.operand.node)?;

        match (base_signal, output.operand.negated) {
            (TechSignal::Literal(_), _) => {
                return Err(anyhow!(
                    "constant output '{}' is unsupported by structural tech mapping because INV/NAND2 mapping has no constant drivers",
                    output.port_name
                ));
            }
            (TechSignal::Net(src), true) => {
                add_instance(
                    &mut interner,
                    &mut instances,
                    options.inv_cell_name.as_str(),
                    format!("u_inv_out_neg_{}", output_index).as_str(),
                    vec![
                        (options.inv_input_pin_name.as_str(), NetRef::Simple(src)),
                        (
                            options.inv_output_pin_name.as_str(),
                            NetRef::Simple(output.port_net),
                        ),
                    ],
                );
            }
            (TechSignal::Net(src), false) => {
                if src != output.port_net {
                    // Keep the emitted netlist purely structural: when a
                    // distinct output port must mirror an existing signal,
                    // realize that connection through cells instead of a
                    // continuous assign.
                    let tmp_name = fresh_internal_net_name(
                        format!("n_out_buf_{}", output_index),
                        &mut used_net_names,
                    );
                    let tmp_net = ensure_net(
                        &mut interner,
                        &mut nets,
                        &mut net_index_by_sym,
                        &mut port_net_indices,
                        &mut wire_net_indices,
                        &mut wire_net_set,
                        tmp_name.as_str(),
                        /* is_port= */ false,
                    );
                    add_instance(
                        &mut interner,
                        &mut instances,
                        options.inv_cell_name.as_str(),
                        format!("u_inv_out_buf0_{}", output_index).as_str(),
                        vec![
                            (options.inv_input_pin_name.as_str(), NetRef::Simple(src)),
                            (
                                options.inv_output_pin_name.as_str(),
                                NetRef::Simple(tmp_net),
                            ),
                        ],
                    );
                    add_instance(
                        &mut interner,
                        &mut instances,
                        options.inv_cell_name.as_str(),
                        format!("u_inv_out_buf1_{}", output_index).as_str(),
                        vec![
                            (options.inv_input_pin_name.as_str(), NetRef::Simple(tmp_net)),
                            (
                                options.inv_output_pin_name.as_str(),
                                NetRef::Simple(output.port_net),
                            ),
                        ],
                    );
                }
            }
        }
    }

    let module = NetlistModule {
        name: module_name_sym,
        net_index_range: 0..nets.len(),
        ports,
        wires: wire_net_indices,
        assigns: vec![],
        instances,
    };

    Ok(MappedNetlist {
        module,
        nets,
        interner,
    })
}

/// VT and drive-strength policy for choosing INV/NAND2 cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellPolicy {
    SmallNormalVt,
    MaxSpeed,
}

impl CellPolicy {
    /// Returns the stable spelling used in cell-selection reports.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SmallNormalVt => "small-normal-vt",
            Self::MaxSpeed => "max-speed",
        }
    }
}

/// A selected cell with its functional input and output pin names.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SelectedCellBinding {
    cell_name: String,
    input_pin_names: Vec<String>,
    output_pin_name: String,
}

/// Collects cell input names in deterministic lexical order.
fn input_pin_names(lib: &Library, cell: &Cell) -> Vec<String> {
    let mut names = cell
        .pins
        .iter()
        .filter(|p| p.direction == PinDirection::Input as i32)
        .map(|p| lib.resolve_string(&p.name).to_string())
        .collect::<Vec<_>>();
    names.sort();
    names
}

/// Collects cell output names in deterministic lexical order.
fn output_pin_names(lib: &Library, cell: &Cell) -> Vec<String> {
    let mut names = cell
        .pins
        .iter()
        .filter(|p| p.direction == PinDirection::Output as i32)
        .map(|p| lib.resolve_string(&p.name).to_string())
        .collect::<Vec<_>>();
    names.sort();
    names
}

/// Looks up the Boolean function for a named output pin.
fn output_pin_function<'a>(
    lib: &'a Library,
    cell: &'a Cell,
    output_pin_name: &str,
) -> Option<&'a str> {
    cell.pins
        .iter()
        .find(|p| {
            lib.resolve_string(&p.name) == output_pin_name
                && p.direction == PinDirection::Output as i32
        })
        .map(|p| lib.resolve_string(&p.function))
}

/// Recognizes the inverter function for the selected input pin.
fn is_inv_formula(term: &Term, input_pin_name: &str) -> bool {
    *term == Term::Negate(Box::new(Term::Input(input_pin_name.to_string())))
}

/// Recognizes NAND2 in direct or De Morgan form with either input order.
fn is_nand2_formula(term: &Term, input_pin_names: &[String]) -> bool {
    let [a, b] = input_pin_names else {
        return false;
    };
    // Liberty libraries may spell NAND2 either directly or in equivalent SOP
    // form, so accept both exact two-input AST shapes after parsing.
    let expected_ab = Term::Negate(Box::new(Term::And(
        Box::new(Term::Input(a.clone())),
        Box::new(Term::Input(b.clone())),
    )));
    let expected_ba = Term::Negate(Box::new(Term::And(
        Box::new(Term::Input(b.clone())),
        Box::new(Term::Input(a.clone())),
    )));
    let expected_demorgan_ab = Term::Or(
        Box::new(Term::Negate(Box::new(Term::Input(a.clone())))),
        Box::new(Term::Negate(Box::new(Term::Input(b.clone())))),
    );
    let expected_demorgan_ba = Term::Or(
        Box::new(Term::Negate(Box::new(Term::Input(b.clone())))),
        Box::new(Term::Negate(Box::new(Term::Input(a.clone())))),
    );
    *term == expected_ab
        || *term == expected_ba
        || *term == expected_demorgan_ab
        || *term == expected_demorgan_ba
}

/// Recognizes a functional cell before checking its sizing-policy coverage.
fn cell_binding_without_policy_coverage(
    lib: &Library,
    cell: &Cell,
    family_prefix: &str,
    expected_input_count: usize,
) -> Option<SelectedCellBinding> {
    if !cell.name.starts_with(family_prefix) {
        return None;
    }
    // `dont_use` forbids use; `use_for_size_only` permits only resizing.
    if cell.dont_use == Some(true) || cell.use_for_size_only == Some(true) {
        return None;
    }
    // A combinational formula alone does not make a sequential or clocking
    // cell safe to use as a logic gate.
    if !cell.sequential.is_empty()
        || cell.clock_gate.is_some()
        || cell.pins.iter().any(|pin| pin.is_clocking_pin)
    {
        return None;
    }
    if let Err(error) = cell.validate_functional_semantics() {
        log::warn!(
            "skipping {} candidate '{}': {}",
            family_prefix,
            cell.name,
            error
        );
        return None;
    }
    let input_pin_names = input_pin_names(lib, cell);
    let output_pin_names = output_pin_names(lib, cell);
    if input_pin_names.len() != expected_input_count || output_pin_names.len() != 1 {
        return None;
    }
    let output_pin_name = &output_pin_names[0];
    // Named connections require nonempty and distinct input and output names.
    if output_pin_name.is_empty()
        || input_pin_names.iter().any(String::is_empty)
        || input_pin_names.windows(2).any(|pair| pair[0] == pair[1])
        || input_pin_names.contains(output_pin_name)
    {
        return None;
    }
    let output_function = output_pin_function(lib, cell, output_pin_name)?;
    let output_term = parse_formula(output_function).ok()?;
    let formula_matches = match family_prefix {
        "INV" => is_inv_formula(&output_term, &input_pin_names[0]),
        "NAND2" => is_nand2_formula(&output_term, &input_pin_names),
        _ => false,
    };
    if !formula_matches {
        return None;
    }
    Some(SelectedCellBinding {
        cell_name: cell.name.clone(),
        input_pin_names,
        output_pin_name: output_pin_name.clone(),
    })
}

/// Accepts functionally matching cells with known sizing-policy coverage.
fn cell_binding_if_shape_matches(
    lib: &Library,
    cell: &Cell,
    family_prefix: &str,
    expected_input_count: usize,
) -> Option<SelectedCellBinding> {
    let binding =
        cell_binding_without_policy_coverage(lib, cell, family_prefix, expected_input_count)?;
    if let Err(error) = cell.validate_mapping_policy_coverage() {
        log::warn!(
            "skipping {} candidate '{}': {}",
            family_prefix,
            cell.name,
            error
        );
        return None;
    }
    Some(binding)
}

/// Explains a missing candidate when matching cells have unknown policy
/// coverage.
fn missing_policy_coverage_error(
    lib: &Library,
    family_prefix: &str,
    expected_input_count: usize,
) -> Option<String> {
    lib.cells
        .iter()
        .filter(|cell| {
            cell_binding_without_policy_coverage(lib, cell, family_prefix, expected_input_count)
                .is_some()
        })
        .filter_map(|cell| cell.validate_mapping_policy_coverage().err())
        .min()
}

/// Checks whether a cell is an eligible inverter.
fn is_inv_candidate(lib: &Library, cell: &Cell) -> bool {
    cell_binding_if_shape_matches(lib, cell, "INV", 1).is_some()
}

/// Checks whether a cell is an eligible two-input NAND.
fn is_nand2_candidate(lib: &Library, cell: &Cell) -> bool {
    cell_binding_if_shape_matches(lib, cell, "NAND2", 2).is_some()
}

/// Returns the effective VT class for a cell when the library provides
/// ordering.
fn effective_vt_class(lib: &Library, cell: &Cell) -> Result<Option<i32>, String> {
    let group_id = if cell.threshold_voltage_group_id != 0 {
        cell.threshold_voltage_group_id
    } else {
        lib.default_threshold_voltage_group_id
    };
    if group_id == 0 {
        return Ok(None);
    }
    let group_index = group_id as usize - 1;
    if group_index >= lib.threshold_voltage_groups.len() {
        return Err(format!(
            "Cell '{}' effective threshold-voltage group ID {} is out of range",
            cell.name, group_id
        ));
    }
    if lib.threshold_voltage_group_class_indices.is_empty() {
        return Ok(None);
    }
    let Some(class_index) = lib
        .threshold_voltage_group_class_indices
        .get(group_index)
        .copied()
    else {
        return Err(format!(
            "Cell '{}' effective threshold-voltage group '{}' has no class index",
            cell.name, lib.threshold_voltage_groups[group_index]
        ));
    };
    Ok(Some(class_index))
}

/// Ranks canonical and x1 cell names ahead of other drive strengths.
fn drive_x1_priority(name: &str, base: &str) -> usize {
    if name == base {
        return 0;
    }
    let needle = format!("{}x1_", base);
    if name.contains(needle.as_str()) || name.ends_with(&format!("{}x1", base)) {
        1
    } else {
        2
    }
}

/// Parses integer or fractional x-suffixed drive strengths in thousandths.
fn parse_drive_strength_milli(name: &str, base: &str) -> Option<i64> {
    if !name.starts_with(base) {
        return None;
    }
    let rest = name.strip_prefix(base)?;
    let rest = rest.strip_prefix('x')?;
    let end = rest.find('_').unwrap_or(rest.len());
    let token = &rest[..end];
    if token.is_empty() {
        return None;
    }

    if let Some(frac_digits) = token.strip_prefix('p') {
        if frac_digits.is_empty() {
            return None;
        }
        let frac = frac_digits.parse::<i64>().ok()?;
        let exponent = u32::try_from(frac_digits.len()).ok()?;
        let denom = 10i64.checked_pow(exponent)?;
        return frac.checked_mul(1000)?.checked_div(denom);
    }

    let whole = token.parse::<i64>().ok()?;
    whole.checked_mul(1000)
}

/// Ranks candidates by the selected VT/drive policy with stable name
/// tie-breaking.
fn best_cell_name<'a>(
    lib: &Library,
    candidates: &[&'a Cell],
    exact_base: &str,
    policy: CellPolicy,
) -> Result<Option<&'a str>, String> {
    let candidates_with_class = candidates
        .iter()
        .map(|cell| effective_vt_class(lib, cell).map(|class_index| (*cell, class_index)))
        .collect::<Result<Vec<_>, _>>()?;
    let has_any_class = candidates_with_class
        .iter()
        .any(|(_, class_index)| class_index.is_some());
    let has_missing_class = candidates_with_class
        .iter()
        .any(|(_, class_index)| class_index.is_none());
    if has_any_class && has_missing_class {
        return Err(format!(
            "{} candidates mix ordered and unordered threshold-voltage metadata",
            exact_base
        ));
    }

    match policy {
        CellPolicy::SmallNormalVt => Ok(candidates_with_class
            .iter()
            .map(|(cell, class_index)| (cell.name.as_str(), *class_index))
            .min_by_key(|(name, class_index)| {
                (
                    if class_index.map_or(true, |class| class == 0) {
                        0usize
                    } else {
                        1usize
                    },
                    if *name == exact_base { 0usize } else { 1usize },
                    drive_x1_priority(name, exact_base),
                    *name,
                )
            })
            .map(|(name, _)| name)),
        CellPolicy::MaxSpeed => Ok(candidates_with_class
            .iter()
            .map(|(cell, class_index)| (cell.name.as_str(), *class_index))
            .min_by_key(|(name, class_index)| {
                let drive = parse_drive_strength_milli(name, exact_base);
                (
                    Reverse(class_index.unwrap_or(0)),
                    if drive.is_some() { 0usize } else { 1usize },
                    Reverse(drive.unwrap_or(0)),
                    *name,
                )
            })
            .map(|(name, _)| name)),
    }
}

/// Resolves and revalidates a selected cell before returning its pin binding.
fn selected_binding_for_name(
    lib: &Library,
    selected_name: &str,
    family_prefix: &str,
    expected_input_count: usize,
) -> Result<SelectedCellBinding, String> {
    let mut matching_cells = lib.cells.iter().filter(|cell| cell.name == selected_name);
    let cell = matching_cells
        .next()
        .ok_or_else(|| format!("selected cell '{}' disappeared", selected_name))?;
    if matching_cells.next().is_some() {
        return Err(format!(
            "selected cell name '{}' occurs more than once in the library",
            selected_name
        ));
    }
    cell_binding_if_shape_matches(lib, cell, family_prefix, expected_input_count).ok_or_else(|| {
        format!(
            "selected cell '{}' no longer matches expected {} shape",
            selected_name, family_prefix
        )
    })
}

/// Selects one inverter using the requested VT/drive policy.
fn select_inv_cell(
    lib: &crate::liberty_model::Library,
    policy: CellPolicy,
) -> Result<SelectedCellBinding, String> {
    let candidates: Vec<&Cell> = lib
        .cells
        .iter()
        .filter(|c| is_inv_candidate(lib, c))
        .collect();
    let Some(name) = best_cell_name(lib, candidates.as_slice(), "INV", policy)? else {
        return Err(missing_policy_coverage_error(lib, "INV", 1).unwrap_or_else(|| {
            "could not find an eligible INV candidate with one input pin, one output pin, and an inverter function".to_string()
        }));
    };
    selected_binding_for_name(lib, name, "INV", 1)
}

/// Selects one NAND2 using the requested VT/drive policy.
fn select_nand2_cell(
    lib: &crate::liberty_model::Library,
    policy: CellPolicy,
) -> Result<SelectedCellBinding, String> {
    let candidates: Vec<&Cell> = lib
        .cells
        .iter()
        .filter(|c| is_nand2_candidate(lib, c))
        .collect();
    let Some(name) = best_cell_name(lib, candidates.as_slice(), "NAND2", policy)? else {
        return Err(missing_policy_coverage_error(lib, "NAND2", 2).unwrap_or_else(|| {
            "could not find an eligible NAND2 candidate with two input pins, one output pin, and a NAND2 function".to_string()
        }));
    };
    selected_binding_for_name(lib, name, "NAND2", 2)
}

#[cfg(test)]
/// Returns the selected inverter name for policy assertions.
fn select_inv_cell_name(lib: &Library, policy: CellPolicy) -> Result<String, String> {
    select_inv_cell(lib, policy).map(|binding| binding.cell_name)
}

#[cfg(test)]
/// Returns the selected NAND2 name for policy assertions.
fn select_nand2_cell_name(lib: &Library, policy: CellPolicy) -> Result<String, String> {
    select_nand2_cell(lib, policy).map(|binding| binding.cell_name)
}

/// Selects functional INV/NAND2 bindings using the requested VT/drive policy.
/// Timing arcs are not required for mapping; STA requires timing data
/// separately.
pub fn structural_options_from_liberty(
    lib: &Library,
    policy: CellPolicy,
) -> Result<StructuralTechMapOptions> {
    let inv = select_inv_cell(lib, policy).map_err(anyhow::Error::msg)?;
    let nand2 = select_nand2_cell(lib, policy).map_err(anyhow::Error::msg)?;
    Ok(StructuralTechMapOptions {
        module_name: None,
        nand2_cell_name: nand2.cell_name,
        nand2_input_pin_names: [
            nand2.input_pin_names[0].clone(),
            nand2.input_pin_names[1].clone(),
        ],
        nand2_output_pin_name: nand2.output_pin_name,
        inv_cell_name: inv.cell_name,
        inv_input_pin_name: inv.input_pin_names[0].clone(),
        inv_output_pin_name: inv.output_pin_name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig_sim::gate_sim::{Collect, eval};
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::liberty_model::{Cell, Library as LibertyLibrary, Pin, PinDirection};
    use crate::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty;
    use std::collections::HashSet;

    /// Resolves the name of a mapped net for structural assertions.
    fn net_name(mapped: &MappedNetlist, idx: NetIndex) -> String {
        mapped
            .nets
            .get(idx.0)
            .and_then(|n| mapped.interner.resolve(n.name))
            .unwrap_or("<unknown>")
            .to_string()
    }

    /// Resolves a test symbol into its textual name.
    fn sym_name(interner: &StringInterner<StringBackend<SymbolU32>>, sym: SymbolU32) -> String {
        interner.resolve(sym).unwrap_or("<unknown>").to_string()
    }

    /// Builds the functional INV/NAND2 test library for netlist projection.
    fn make_inv_nand2_liberty() -> LibertyLibrary {
        let mut builder = crate::liberty_model::LibraryBuilder::new();
        let a = builder.intern_string("A").unwrap();
        let b = builder.intern_string("B").unwrap();
        let y = builder.intern_string("Y").unwrap();
        let inv_function = builder.intern_string("!A").unwrap();
        let nand_function = builder.intern_string("!(A*B)").unwrap();
        builder.cells = vec![
            Cell {
                name: "INV".to_string().into(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: a,
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: y,
                        function: inv_function,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
            Cell {
                name: "NAND2".to_string().into(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: a,
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: b,
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: y,
                        function: nand_function,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
        ];
        builder.finish()
    }

    /// Compares original and projected graphs by exhaustive input simulation.
    fn assert_projected_equivalent_for_all_inputs(lhs: &GateFn, rhs: &GateFn) {
        assert_eq!(lhs.inputs.len(), rhs.inputs.len());
        assert_eq!(lhs.outputs.len(), rhs.outputs.len());

        let total_input_bits: usize = lhs.inputs.iter().map(|i| i.get_bit_count()).sum();
        assert!(
            total_input_bits <= 16,
            "test harness only supports <=16 bits"
        );

        for input_index in 0..(1usize << total_input_bits) {
            let mut inputs: Vec<IrBits> = Vec::new();
            let mut shift = 0usize;
            for input in &lhs.inputs {
                let width = input.get_bit_count();
                let value = if width == 0 {
                    0usize
                } else {
                    (input_index >> shift) & ((1usize << width) - 1)
                };
                inputs.push(
                    IrBits::make_ubits(width, value as u64).expect("input stimulus should fit"),
                );
                shift += width;
            }

            let lhs_eval = eval(lhs, inputs.as_slice(), Collect::None);
            let rhs_eval = eval(rhs, inputs.as_slice(), Collect::None);
            assert_eq!(lhs_eval.outputs, rhs_eval.outputs);
        }
    }

    /// Renders stable cell and pin bindings for structural golden assertions.
    fn render_instance_summary(mapped: &MappedNetlist) -> Vec<String> {
        let mut lines = Vec::new();
        for inst in &mapped.module.instances {
            let type_name = sym_name(&mapped.interner, inst.type_name);
            let inst_name = sym_name(&mapped.interner, inst.instance_name);
            let mut conn_parts = Vec::new();
            for (pin_sym, netref) in &inst.connections {
                let pin_name = sym_name(&mapped.interner, *pin_sym);
                let rhs = match netref {
                    NetRef::Simple(idx) => net_name(mapped, *idx),
                    NetRef::Literal(bits) => {
                        let bit = bits.get_bit(0).unwrap_or(false);
                        if bit {
                            "1'b1".to_string()
                        } else {
                            "1'b0".to_string()
                        }
                    }
                    NetRef::UnknownLiteral(width) => format!("{}'hx", width),
                    NetRef::BitSelect(idx, bit) => format!("{}[{}]", net_name(mapped, *idx), bit),
                    NetRef::PartSelect(idx, msb, lsb) => {
                        format!("{}[{}:{}]", net_name(mapped, *idx), msb, lsb)
                    }
                    NetRef::Unconnected => "<unconnected>".to_string(),
                    NetRef::Concat(_) => "<concat>".to_string(),
                };
                conn_parts.push(format!("{}.{}", pin_name, rhs));
            }
            lines.push(format!(
                "{} {} {}",
                type_name,
                inst_name,
                conn_parts.join(" ")
            ));
        }
        lines
    }

    #[test]
    fn and_gate_maps_to_one_nand2_and_one_inv() {
        let mut gb = GateBuilder::new("and_gate".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1);
        let b = gb.add_input("b".to_string(), 1);
        let and_out = gb.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
        gb.add_output("y".to_string(), and_out.into());
        let gate_fn = gb.build();

        let mapped =
            map_gatefn_to_structural_netlist(&gate_fn, &StructuralTechMapOptions::default())
                .expect("mapping should succeed");

        assert_eq!(mapped.module.instances.len(), 2);
        let summary = render_instance_summary(&mapped);
        assert!(
            summary
                .iter()
                .any(|line| line.starts_with("NAND2 u_nand_g")),
            "expected NAND2 instance in {:?}",
            summary
        );
        assert!(
            summary.iter().any(|line| line.starts_with("INV u_inv_g")),
            "expected INV instance in {:?}",
            summary
        );
    }

    #[test]
    fn negated_edge_materializes_edge_inverter() {
        let mut gb = GateBuilder::new("neg_edge".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1);
        let b = gb.add_input("b".to_string(), 1);
        let and_out = gb.add_and_binary(a.get_lsb(0).negate(), *b.get_lsb(0));
        gb.add_output("y".to_string(), and_out.into());
        let gate_fn = gb.build();

        let mapped =
            map_gatefn_to_structural_netlist(&gate_fn, &StructuralTechMapOptions::default())
                .expect("mapping should succeed");

        let summary = render_instance_summary(&mapped);
        assert!(
            summary
                .iter()
                .any(|line| line.starts_with("INV u_inv_edge_0")),
            "expected edge inverter in {:?}",
            summary
        );
        assert_eq!(mapped.module.instances.len(), 3);
    }

    #[test]
    fn generated_internal_nets_do_not_reuse_port_names() {
        let mut gb = GateBuilder::new("collision".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("n_nand_g3".to_string(), 1);
        let b = gb.add_input("b".to_string(), 1);
        let and_out = gb.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
        gb.add_output("y".to_string(), and_out.into());
        let gate_fn = gb.build();

        let mapped =
            map_gatefn_to_structural_netlist(&gate_fn, &StructuralTechMapOptions::default())
                .expect("mapping should succeed");
        let input_port_names: HashSet<String> = mapped
            .module
            .ports
            .iter()
            .filter(|port| port.direction == PortDirection::Input)
            .map(|port| sym_name(&mapped.interner, port.name))
            .collect();
        let nand = mapped
            .module
            .instances
            .iter()
            .find(|inst| sym_name(&mapped.interner, inst.instance_name).starts_with("u_nand_g"))
            .expect("nand instance expected");
        let nand_output = nand
            .connections
            .iter()
            .find_map(|(pin, netref)| (sym_name(&mapped.interner, *pin) == "Y").then_some(netref))
            .expect("nand output connection expected");
        let NetRef::Simple(nand_output_net) = nand_output else {
            panic!("nand output should be a simple net");
        };
        let nand_output_name = net_name(&mapped, *nand_output_net);
        assert!(!input_port_names.contains(&nand_output_name));
        assert!(nand_output_name.starts_with("n_nand_g3"));
    }

    #[test]
    fn constant_outputs_are_rejected_instead_of_emitting_literal_tied_cells() {
        let mut gb = GateBuilder::new("const_out".to_string(), GateBuilderOptions::no_opt());
        gb.add_output("y".to_string(), gb.get_false().into());
        let gate_fn = gb.build();

        let err = map_gatefn_to_structural_netlist(&gate_fn, &StructuralTechMapOptions::default())
            .expect_err("constant outputs should be rejected");
        assert!(
            err.to_string()
                .contains("constant output 'y' is unsupported")
        );
    }

    #[test]
    fn literal_fed_gates_are_rejected_before_mapping() {
        let mut gb = GateBuilder::new("literal_input".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1);
        let y = gb.add_and_binary(*a.get_lsb(0), gb.get_true());
        gb.add_output("y".to_string(), y.into());
        let gate_fn = gb.build();

        let err = map_gatefn_to_structural_netlist(&gate_fn, &StructuralTechMapOptions::default())
            .expect_err("literal-fed gates should be rejected");
        assert!(
            err.to_string()
                .contains("constants must be folded before mapping")
        );
    }

    #[test]
    fn mapped_netlist_projects_back_equivalent_gatefn() {
        let mut gb = GateBuilder::new("and_nand_dual".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1);
        let b = gb.add_input("b".to_string(), 1);
        let and_out = gb.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
        let nand_out = gb.add_not(and_out);
        gb.add_output("y_and".to_string(), and_out.into());
        gb.add_output("y_nand".to_string(), nand_out.into());
        let original = gb.build();

        let mapped =
            map_gatefn_to_structural_netlist(&original, &StructuralTechMapOptions::default())
                .expect("mapping should succeed");

        let lib = make_inv_nand2_liberty();
        let dff_cells_identity: HashSet<String> = HashSet::new();
        let dff_cells_inverted: HashSet<String> = HashSet::new();
        let projected = project_gatefn_from_netlist_and_liberty(
            &mapped.module,
            mapped.nets.as_slice(),
            &mapped.interner,
            &lib,
            &dff_cells_identity,
            &dff_cells_inverted,
        )
        .expect("projected gatefn should build from mapped netlist");

        assert_projected_equivalent_for_all_inputs(&original, &projected);
    }

    #[test]
    fn mapping_is_deterministic() {
        let mut gb = GateBuilder::new("deterministic".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1);
        let b = gb.add_input("b".to_string(), 1);
        let and_out = gb.add_and_binary(a.get_lsb(0).negate(), b.get_lsb(0).negate());
        gb.add_output("y".to_string(), and_out.into());
        let gate_fn = gb.build();

        let first =
            map_gatefn_to_structural_netlist(&gate_fn, &StructuralTechMapOptions::default())
                .expect("first mapping should succeed");
        let second =
            map_gatefn_to_structural_netlist(&gate_fn, &StructuralTechMapOptions::default())
                .expect("second mapping should succeed");

        let first_summary = render_instance_summary(&first);
        let second_summary = render_instance_summary(&second);
        assert_eq!(first_summary, second_summary);

        let first_nets: Vec<String> = first
            .nets
            .iter()
            .map(|n| sym_name(&first.interner, n.name))
            .collect();
        let second_nets: Vec<String> = second
            .nets
            .iter()
            .map(|n| sym_name(&second.interner, n.name))
            .collect();
        assert_eq!(first_nets, second_nets);
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    use crate::liberty::model::{library_from_proto, library_to_proto};
    use crate::liberty::parser::parse_liberty_files;
    use crate::liberty_model::{LibraryBuilder, Pin, TimingTable};
    use prost::Message;

    /// Creates a test pin with interned name and function strings.
    fn make_pin(
        library: &mut LibraryBuilder,
        name: &str,
        direction: PinDirection,
        function: &str,
    ) -> Pin {
        Pin {
            name: library.intern_string(name).unwrap(),
            direction: direction as i32,
            function: library.intern_string(function).unwrap(),
            ..Default::default()
        }
    }

    /// Creates an output with complete scalar timing arcs for every related
    /// input.
    fn make_timed_output_pin(
        library: &mut LibraryBuilder,
        name: &str,
        function: &str,
        related_pins: &[&str],
    ) -> Pin {
        let mut timing_arcs = Vec::with_capacity(related_pins.len());
        for related_pin in related_pins {
            let tables = vec![
                scalar_table(library, "cell_rise", 1.0),
                scalar_table(library, "cell_fall", 1.0),
                scalar_table(library, "rise_transition", 0.1),
                scalar_table(library, "fall_transition", 0.1),
            ];
            timing_arcs.push(
                library
                    .add_timing_arc(related_pin, "", "combinational", "", tables)
                    .unwrap(),
            );
        }
        Pin {
            name: library.intern_string(name).unwrap(),
            direction: PinDirection::Output as i32,
            function: library.intern_string(function).unwrap(),
            timing_arcs,
            ..Default::default()
        }
    }

    /// Creates a one-point timing table for cell-selection tests.
    fn scalar_table(library: &mut LibraryBuilder, kind: &str, value: f64) -> TimingTable {
        let kind = match kind {
            "cell_rise" => crate::liberty_proto::TimingTableKind::CellRise,
            "cell_fall" => crate::liberty_proto::TimingTableKind::CellFall,
            "rise_transition" => crate::liberty_proto::TimingTableKind::RiseTransition,
            "fall_transition" => crate::liberty_proto::TimingTableKind::FallTransition,
            other => panic!("unsupported test timing-table kind {other}"),
        };
        library
            .add_timing_table_f64(kind, 0, vec![], vec![], vec![], vec![value], vec![], "")
            .unwrap()
    }

    /// Creates a timed inverter with canonical test pins and the requested VT
    /// group.
    fn make_inv_cell(
        library: &mut LibraryBuilder,
        name: &str,
        threshold_voltage_group_id: u32,
    ) -> Cell {
        make_inv_cell_with_pins(library, name, threshold_voltage_group_id, "A", "Y")
    }

    /// Creates a timed inverter with explicitly named input and output pins.
    fn make_inv_cell_with_pins(
        library: &mut LibraryBuilder,
        name: &str,
        threshold_voltage_group_id: u32,
        input_pin_name: &str,
        output_pin_name: &str,
    ) -> Cell {
        Cell {
            name: name.to_string().into(),
            threshold_voltage_group_id,
            pins: vec![
                make_pin(library, input_pin_name, PinDirection::Input, ""),
                make_timed_output_pin(
                    library,
                    output_pin_name,
                    format!("!{}", input_pin_name).as_str(),
                    &[input_pin_name],
                ),
            ],
            ..Default::default()
        }
    }

    /// Creates a timed NAND2 with canonical test pins and the requested VT
    /// group.
    fn make_nand2_cell(
        library: &mut LibraryBuilder,
        name: &str,
        threshold_voltage_group_id: u32,
    ) -> Cell {
        make_nand2_cell_with_pins(library, name, threshold_voltage_group_id, ["A", "B"], "Y")
    }

    /// Creates a timed NAND2 with explicitly named input and output pins.
    fn make_nand2_cell_with_pins(
        library: &mut LibraryBuilder,
        name: &str,
        threshold_voltage_group_id: u32,
        input_pin_names: [&str; 2],
        output_pin_name: &str,
    ) -> Cell {
        Cell {
            name: name.to_string().into(),
            threshold_voltage_group_id,
            pins: vec![
                make_pin(library, input_pin_names[0], PinDirection::Input, ""),
                make_pin(library, input_pin_names[1], PinDirection::Input, ""),
                make_timed_output_pin(
                    library,
                    output_pin_name,
                    format!("!({}*{})", input_pin_names[0], input_pin_names[1]).as_str(),
                    &input_pin_names,
                ),
            ],
            ..Default::default()
        }
    }

    /// Finishes a test library after populating cells through its shared
    /// interner.
    fn make_library_with_cells(
        build_cells: impl FnOnce(&mut LibraryBuilder) -> Vec<Cell>,
    ) -> crate::liberty_model::Library {
        let mut library = LibraryBuilder::new();
        library.cells = build_cells(&mut library);
        library.finish()
    }

    #[test]
    fn auto_selection_excludes_mapping_restricted_cells() {
        let mut lib = make_library_with_cells(|library| {
            let mut inv = make_inv_cell(library, "INV", 0);
            inv.dont_use = Some(true);
            let mut inv_size = make_inv_cell(library, "INVx1", 0);
            inv_size.use_for_size_only = Some(true);
            let mut inv_ok = make_inv_cell(library, "INVx2", 0);
            inv_ok.dont_use = Some(false);
            let mut nand = make_nand2_cell(library, "NAND2", 0);
            nand.dont_use = Some(true);
            let mut nand_size = make_nand2_cell(library, "NAND2x1", 0);
            nand_size.use_for_size_only = Some(true);
            let mut nand_ok = make_nand2_cell(library, "NAND2x2", 0);
            nand_ok.use_for_size_only = Some(false);
            vec![inv, inv_size, inv_ok, nand, nand_size, nand_ok]
        });
        for policy in [CellPolicy::SmallNormalVt, CellPolicy::MaxSpeed] {
            assert_eq!(select_inv_cell_name(&lib, policy).unwrap(), "INVx2");
            assert_eq!(select_nand2_cell_name(&lib, policy).unwrap(), "NAND2x2");
        }

        lib.cells.retain(|cell| !cell.name.ends_with("x2"));
        for policy in [CellPolicy::SmallNormalVt, CellPolicy::MaxSpeed] {
            assert_eq!(
                select_inv_cell_name(&lib, policy).unwrap_err(),
                "could not find an eligible INV candidate with one input pin, one output pin, and an inverter function"
            );
            assert_eq!(
                select_nand2_cell_name(&lib, policy).unwrap_err(),
                "could not find an eligible NAND2 candidate with two input pins, one output pin, and a NAND2 function"
            );
        }
    }

    #[test]
    fn structural_selection_rejects_unknown_legacy_sizing_policy() {
        let legacy = make_library_with_cells(|library| {
            vec![
                make_inv_cell(library, "INV", 0),
                make_nand2_cell(library, "NAND2", 0),
            ]
        });
        let mut wire = library_to_proto(legacy).unwrap();
        wire.format_magic = crate::liberty::model::LIBERTY_FORMAT_MAGIC_V5;
        let legacy = library_from_proto(wire).unwrap();
        assert!(legacy.cells.iter().all(|cell| {
            cell.validate_functional_semantics().is_ok()
                && cell.validate_mapping_policy_coverage().is_err()
        }));
        for policy in [CellPolicy::SmallNormalVt, CellPolicy::MaxSpeed] {
            assert_eq!(
                select_inv_cell_name(&legacy, policy).unwrap_err(),
                "cell 'INV' has unknown use_for_size_only source coverage; regenerate from source Liberty before mapping"
            );
            assert_eq!(
                select_nand2_cell_name(&legacy, policy).unwrap_err(),
                "cell 'NAND2' has unknown use_for_size_only source coverage; regenerate from source Liberty before mapping"
            );
        }

        let mut mixed = make_library_with_cells(|library| {
            vec![
                make_inv_cell(library, "INVx2", 0),
                make_nand2_cell(library, "NAND2x2", 0),
            ]
        });
        mixed.append_cells_and_luts(legacy).unwrap();
        for policy in [CellPolicy::SmallNormalVt, CellPolicy::MaxSpeed] {
            assert_eq!(select_inv_cell_name(&mixed, policy).unwrap(), "INVx2");
            assert_eq!(select_nand2_cell_name(&mixed, policy).unwrap(), "NAND2x2");
        }
    }

    /// Imports structural candidates with different functional semantics.
    fn import_structural_candidates(output_attribute: &str, extra_pin: &str) -> Library {
        let mut source = String::from("library(test) {");
        for (family, inputs, formula) in [
            ("INV", vec!["A"], "!A"),
            ("NAND2", vec!["A", "B"], "!(A * B)"),
        ] {
            for safe in [false, true] {
                let name = if safe {
                    format!("{family}_SAFE")
                } else {
                    family.into()
                };
                source.push_str(&format!("cell({name}) {{ area: 1;"));
                for input in &inputs {
                    source.push_str(&format!("pin({input}) {{ direction: input; }}"));
                }
                if !safe {
                    source.push_str(extra_pin);
                }
                source.push_str(&format!(
                    r#"pin(Y) {{ direction: output; function: "{formula}";"#
                ));
                if !safe {
                    source.push_str(output_attribute);
                }
                for input in &inputs {
                    source.push_str(&format!(r#"timing() {{ related_pin: "{input}"; timing_type: combinational;
                        cell_rise(scalar) {{ values("1"); }} cell_fall(scalar) {{ values("1"); }}
                        rise_transition(scalar) {{ values("0.1"); }} fall_transition(scalar) {{ values("0.1"); }}
                    }}"#));
                }
                source.push_str("} }");
            }
        }
        source.push('}');
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), source).unwrap();
        let wire = library_to_proto(parse_liberty_files(&[file.path()]).unwrap()).unwrap();
        library_from_proto(
            crate::liberty_proto::Library::decode(wire.encode_to_vec().as_slice()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn structural_selection_rejects_unsupported_imported_semantics() {
        for (output_attribute, extra_pin) in [
            (r#"three_state: "A";"#, ""),
            (r#"x_function: "A";"#, ""),
            ("", "pin(I) { direction: internal; }"),
            ("", "pin(IO) { direction: inout; }"),
        ] {
            let mut library = import_structural_candidates(output_attribute, extra_pin);
            for policy in [CellPolicy::SmallNormalVt, CellPolicy::MaxSpeed] {
                assert_eq!(select_inv_cell_name(&library, policy).unwrap(), "INV_SAFE");
                assert_eq!(
                    select_nand2_cell_name(&library, policy).unwrap(),
                    "NAND2_SAFE"
                );
            }
            library.cells.retain(|cell| !cell.name.ends_with("_SAFE"));
            for policy in [CellPolicy::SmallNormalVt, CellPolicy::MaxSpeed] {
                assert!(select_inv_cell_name(&library, policy).is_err());
                assert!(select_nand2_cell_name(&library, policy).is_err());
            }
        }
    }

    #[test]
    fn select_prefers_exact_base_names_in_small_normal_vt_mode() {
        let mut lib = make_library_with_cells(|library| {
            vec![
                make_inv_cell(library, "INV", 1),
                make_inv_cell(library, "INVx1_nominal", 1),
                make_nand2_cell(library, "NAND2", 1),
                make_nand2_cell(library, "NAND2x1_nominal", 1),
            ]
        });
        lib.threshold_voltage_groups = vec!["nominal".to_string()];
        lib.threshold_voltage_group_class_indices = vec![0];

        assert_eq!(
            select_inv_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "INV"
        );
        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "NAND2"
        );
    }

    #[test]
    fn select_prefers_x1_nominal_when_no_exact_base_in_small_normal_vt_mode() {
        let mut lib = make_library_with_cells(|library| {
            vec![
                make_inv_cell(library, "INVx2_fast", 2),
                make_inv_cell(library, "INVx1_nominal", 1),
                make_inv_cell(library, "INVx1_faster", 3),
                make_nand2_cell(library, "NAND2x2_fast", 2),
                make_nand2_cell(library, "NAND2x1_nominal", 1),
            ]
        });
        lib.threshold_voltage_groups = vec![
            "nominal".to_string(),
            "fast".to_string(),
            "faster".to_string(),
        ];
        lib.threshold_voltage_group_class_indices = vec![0, 1, 2];

        assert_eq!(
            select_inv_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "INVx1_nominal"
        );
        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "NAND2x1_nominal"
        );
    }

    #[test]
    fn select_prefers_max_drive_in_fastest_vt_in_max_speed_mode() {
        let mut lib = make_library_with_cells(|library| {
            vec![
                make_inv_cell(library, "INVx1_nominal", 1),
                make_inv_cell(library, "INVx2_faster", 3),
                make_inv_cell(library, "INVx4_faster", 3),
                make_nand2_cell(library, "NAND2x1_nominal", 1),
                make_nand2_cell(library, "NAND2x2_fast", 2),
                make_nand2_cell(library, "NAND2x4_faster", 3),
            ]
        });
        lib.threshold_voltage_groups = vec![
            "nominal".to_string(),
            "fast".to_string(),
            "faster".to_string(),
        ];
        lib.threshold_voltage_group_class_indices = vec![0, 1, 2];

        assert_eq!(
            select_inv_cell_name(&lib, CellPolicy::MaxSpeed).unwrap(),
            "INVx4_faster"
        );
        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::MaxSpeed).unwrap(),
            "NAND2x4_faster"
        );
    }

    #[test]
    fn select_uses_drive_strength_with_fractional_suffix_in_max_speed_mode() {
        let mut lib = make_library_with_cells(|library| {
            vec![
                make_inv_cell(library, "INVxp75_faster", 1),
                make_inv_cell(library, "INVx1_faster", 1),
                make_nand2_cell(library, "NAND2xp33_faster", 1),
                make_nand2_cell(library, "NAND2x1_faster", 1),
            ]
        });
        lib.threshold_voltage_groups = vec!["faster".to_string()];
        lib.threshold_voltage_group_class_indices = vec![2];

        assert_eq!(
            select_inv_cell_name(&lib, CellPolicy::MaxSpeed).unwrap(),
            "INVx1_faster"
        );
        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::MaxSpeed).unwrap(),
            "NAND2x1_faster"
        );
    }

    #[test]
    fn select_handles_extremely_long_fractional_drive_names() {
        let tiny = format!("INVxp{}1", "0".repeat(64));
        let near_one = format!("INVxp{}", "9".repeat(64));
        let lib = make_library_with_cells(|library| {
            vec![
                make_inv_cell(library, &tiny, 0),
                make_inv_cell(library, "INVxp1", 0),
                make_inv_cell(library, &near_one, 0),
            ]
        });
        assert_eq!(parse_drive_strength_milli(&tiny, "INV"), None);
        assert_eq!(parse_drive_strength_milli(&near_one, "INV"), None);
        assert_eq!(
            select_inv_cell_name(&lib, CellPolicy::MaxSpeed).unwrap(),
            "INVxp1"
        );
    }

    #[test]
    fn select_handles_extremely_large_integer_drive_names() {
        for large in [
            "INVx9223372036854776".to_string(),
            format!("INVx{}", "9".repeat(64)),
        ] {
            let lib = make_library_with_cells(|library| {
                vec![
                    make_inv_cell(library, "INVx1", 0),
                    make_inv_cell(library, &large, 0),
                ]
            });
            assert_eq!(parse_drive_strength_milli(&large, "INV"), None);
            assert_eq!(
                select_inv_cell_name(&lib, CellPolicy::MaxSpeed).unwrap(),
                "INVx1"
            );
        }
    }

    #[test]
    fn select_uses_default_threshold_voltage_group() {
        let mut lib = make_library_with_cells(|library| {
            vec![
                make_inv_cell(library, "INVx1_default", 0),
                make_inv_cell(library, "INVx2_fast", 2),
                make_nand2_cell(library, "NAND2x1_default", 0),
            ]
        });
        lib.threshold_voltage_groups = vec!["nominal".to_string(), "fast".to_string()];
        lib.default_threshold_voltage_group_id = 1;
        lib.threshold_voltage_group_class_indices = vec![0, 1];

        assert_eq!(
            select_inv_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "INVx1_default"
        );
        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "NAND2x1_default"
        );
    }

    #[test]
    fn select_discovers_nonstandard_pin_names() {
        let lib = make_library_with_cells(|library| {
            vec![
                make_inv_cell_with_pins(library, "INV", 0, "I", "ZN"),
                make_nand2_cell_with_pins(library, "NAND2", 0, ["I0", "I1"], "ZN"),
            ]
        });

        assert_eq!(
            select_inv_cell(&lib, CellPolicy::SmallNormalVt).unwrap(),
            SelectedCellBinding {
                cell_name: "INV".to_string().into(),
                input_pin_names: vec!["I".to_string()],
                output_pin_name: "ZN".to_string().into(),
            }
        );
        assert_eq!(
            select_nand2_cell(&lib, CellPolicy::SmallNormalVt).unwrap(),
            SelectedCellBinding {
                cell_name: "NAND2".to_string().into(),
                input_pin_names: vec!["I0".to_string(), "I1".to_string()],
                output_pin_name: "ZN".to_string().into(),
            }
        );
    }

    #[test]
    fn structural_selection_rejects_duplicate_pin_names() {
        let mut lib = make_library_with_cells(|library| {
            vec![
                make_inv_cell_with_pins(library, "INV", 0, "A", "A"),
                make_inv_cell(library, "INVx2", 0),
                make_nand2_cell_with_pins(library, "NAND2", 0, ["A", "A"], "Y"),
                make_nand2_cell_with_pins(library, "NAND2x1", 0, ["A", "B"], "A"),
                make_nand2_cell(library, "NAND2x2", 0),
            ]
        });
        for with_timing in [true, false] {
            if !with_timing {
                for cell in &mut lib.cells {
                    for pin in &mut cell.pins {
                        pin.timing_arcs.clear();
                    }
                }
            }
            assert!(!is_inv_candidate(&lib, &lib.cells[0]));
            assert!(!is_nand2_candidate(&lib, &lib.cells[2]));
            assert!(!is_nand2_candidate(&lib, &lib.cells[3]));
            for policy in [CellPolicy::SmallNormalVt, CellPolicy::MaxSpeed] {
                assert_eq!(select_inv_cell_name(&lib, policy).unwrap(), "INVx2");
                assert_eq!(select_nand2_cell_name(&lib, policy).unwrap(), "NAND2x2");
            }
        }
    }

    #[test]
    fn structural_selection_rejects_empty_output_pin_name() {
        let mut lib = make_library_with_cells(|library| {
            vec![
                make_inv_cell_with_pins(library, "INV", 0, "A", ""),
                make_nand2_cell_with_pins(library, "NAND2", 0, ["A", "B"], ""),
            ]
        });
        for with_timing in [true, false] {
            if !with_timing {
                for cell in &mut lib.cells {
                    for pin in &mut cell.pins {
                        pin.timing_arcs.clear();
                    }
                }
            }
            assert!(!is_inv_candidate(&lib, &lib.cells[0]));
            assert!(!is_nand2_candidate(&lib, &lib.cells[1]));
        }
    }

    #[test]
    fn structural_selection_rejects_sequential_and_clocking_cells() {
        for kind in 0..3 {
            let mut lib = make_library_with_cells(|library| {
                let mut inv = make_inv_cell(library, "INV", 0);
                let mut nand = make_nand2_cell(library, "NAND2", 0);
                match kind {
                    0 => {
                        inv.sequential.push(Default::default());
                        nand.sequential.push(Default::default());
                    }
                    1 => {
                        inv.clock_gate = Some(Default::default());
                        nand.clock_gate = Some(Default::default());
                    }
                    2 => {
                        inv.pins[0].is_clocking_pin = true;
                        nand.pins[0].is_clocking_pin = true;
                    }
                    _ => unreachable!(),
                }
                vec![
                    inv,
                    nand,
                    make_inv_cell(library, "INVx2", 0),
                    make_nand2_cell(library, "NAND2x2", 0),
                ]
            });
            for with_timing in [true, false] {
                if !with_timing {
                    for cell in &mut lib.cells {
                        for pin in &mut cell.pins {
                            pin.timing_arcs.clear();
                        }
                    }
                }
                assert!(!is_inv_candidate(&lib, &lib.cells[0]));
                assert!(!is_nand2_candidate(&lib, &lib.cells[1]));
                for policy in [CellPolicy::SmallNormalVt, CellPolicy::MaxSpeed] {
                    assert_eq!(select_inv_cell_name(&lib, policy).unwrap(), "INVx2");
                    assert_eq!(select_nand2_cell_name(&lib, policy).unwrap(), "NAND2x2");
                }
            }
        }
    }

    #[test]
    fn structural_selection_rejects_ambiguous_selected_cell_name() {
        for family in ["INV", "NAND2"] {
            for bad_first in [false, true] {
                let lib = make_library_with_cells(|library| {
                    let inv = make_inv_cell(library, "INV", 0);
                    let nand = make_nand2_cell(library, "NAND2", 0);
                    let mut bad = if family == "INV" {
                        make_inv_cell(library, "INV", 0)
                    } else {
                        make_nand2_cell(library, "NAND2", 0)
                    };
                    let wrong_function = if family == "INV" { "A" } else { "A*B" };
                    bad.pins.last_mut().unwrap().function =
                        library.intern_string(wrong_function).unwrap();
                    let mut cells = vec![inv, nand, bad];
                    if bad_first {
                        cells.swap(if family == "INV" { 0 } else { 1 }, 2);
                    }
                    cells
                });
                let error = structural_options_from_liberty(&lib, CellPolicy::SmallNormalVt)
                    .unwrap_err()
                    .to_string();
                assert_eq!(
                    error,
                    format!(
                        "selected cell name '{}' occurs more than once in the library",
                        family
                    )
                );
            }
        }
    }

    #[test]
    fn select_rejects_family_name_with_wrong_formula() {
        let lib = make_library_with_cells(|library| {
            vec![
                Cell {
                    name: "INV".to_string().into(),
                    pins: vec![
                        make_pin(library, "A", PinDirection::Input, ""),
                        make_timed_output_pin(library, "Y", "A", &["A"]),
                    ],
                    ..Default::default()
                },
                Cell {
                    name: "NAND2".to_string().into(),
                    pins: vec![
                        make_pin(library, "A", PinDirection::Input, ""),
                        make_pin(library, "B", PinDirection::Input, ""),
                        make_timed_output_pin(library, "Y", "A*B", &["A", "B"]),
                    ],
                    ..Default::default()
                },
            ]
        });

        assert!(select_inv_cell(&lib, CellPolicy::SmallNormalVt).is_err());
        assert!(select_nand2_cell(&lib, CellPolicy::SmallNormalVt).is_err());
    }

    #[test]
    fn select_accepts_demorgan_nand2_formula() {
        let lib = make_library_with_cells(|library| {
            vec![Cell {
                name: "NAND2".to_string().into(),
                pins: vec![
                    make_pin(library, "A", PinDirection::Input, ""),
                    make_pin(library, "B", PinDirection::Input, ""),
                    make_timed_output_pin(library, "Y", "(!A)+(!B)", &["A", "B"]),
                ],
                ..Default::default()
            }]
        });

        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "NAND2"
        );
    }

    #[test]
    fn select_is_deterministic_without_vt_metadata() {
        let lib = make_library_with_cells(|library| {
            vec![
                make_inv_cell(library, "INVx2", 0),
                make_inv_cell(library, "INVx1", 0),
                make_nand2_cell(library, "NAND2x2", 0),
                make_nand2_cell(library, "NAND2x1", 0),
            ]
        });

        assert_eq!(
            select_inv_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "INVx1"
        );
        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "NAND2x1"
        );
        assert_eq!(
            select_inv_cell_name(&lib, CellPolicy::MaxSpeed).unwrap(),
            "INVx2"
        );
        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::MaxSpeed).unwrap(),
            "NAND2x2"
        );
    }

    #[test]
    fn select_accepts_candidate_missing_timing_for_one_functional_input() {
        let lib = make_library_with_cells(|library| {
            vec![Cell {
                name: "NAND2".to_string().into(),
                pins: vec![
                    make_pin(library, "A", PinDirection::Input, ""),
                    make_pin(library, "B", PinDirection::Input, ""),
                    make_timed_output_pin(library, "Y", "!(A*B)", &["A"]),
                ],
                ..Default::default()
            }]
        });

        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "NAND2"
        );
    }

    #[test]
    fn select_uses_preferred_candidate_even_if_sta_incompatible() {
        let mut builder = LibraryBuilder::new();
        let mut invalid_fast = make_nand2_cell(&mut builder, "NAND2x2_fast", 2);
        let output_pin_idx = invalid_fast
            .pins
            .iter()
            .position(|pin| builder.resolve_string(&pin.name) == "Y")
            .expect("output pin");
        let tables = std::mem::take(&mut invalid_fast.pins[output_pin_idx].timing_arcs[0].tables);
        invalid_fast.pins[output_pin_idx].timing_arcs[0] = builder
            .add_timing_arc("A", "", "rising_edge", "", tables)
            .unwrap();
        let nominal = make_nand2_cell(&mut builder, "NAND2x1_nominal", 1);
        builder.cells = vec![invalid_fast, nominal];
        builder.threshold_voltage_groups = vec!["nominal".to_string(), "fast".to_string()];
        builder.threshold_voltage_group_class_indices = vec![0, 1];
        let lib = builder.finish();

        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::MaxSpeed).unwrap(),
            "NAND2x2_fast"
        );
    }

    #[test]
    fn structural_selection_is_independent_of_timing_data() {
        let mut lib = make_library_with_cells(|library| {
            vec![
                make_inv_cell(library, "INVx1_nominal", 1),
                make_inv_cell(library, "INVx2_fast", 2),
                make_nand2_cell(library, "NAND2x1_nominal", 1),
                make_nand2_cell(library, "NAND2x2_fast", 2),
            ]
        });
        lib.threshold_voltage_groups = vec!["nominal".to_string(), "fast".to_string()];
        lib.threshold_voltage_group_class_indices = vec![0, 1];
        let policies = [CellPolicy::SmallNormalVt, CellPolicy::MaxSpeed];
        let timed = policies.map(|policy| structural_options_from_liberty(&lib, policy).unwrap());
        assert_eq!(timed[0].inv_cell_name, "INVx1_nominal");
        assert_eq!(timed[0].nand2_cell_name, "NAND2x1_nominal");
        assert_eq!(timed[1].inv_cell_name, "INVx2_fast");
        assert_eq!(timed[1].nand2_cell_name, "NAND2x2_fast");

        for cell in &mut lib.cells {
            for pin in &mut cell.pins {
                pin.timing_arcs.clear();
            }
        }
        let untimed = policies.map(|policy| structural_options_from_liberty(&lib, policy).unwrap());
        assert_eq!(timed, untimed);
    }

    #[test]
    fn structural_selection_rejects_extra_untimed_output() {
        let lib = make_library_with_cells(|library| {
            let mut inv = make_inv_cell(library, "INV", 0);
            inv.pins
                .push(make_pin(library, "Z", PinDirection::Output, "!A"));
            let mut nand = make_nand2_cell(library, "NAND2", 0);
            nand.pins
                .push(make_pin(library, "Z", PinDirection::Output, "!(A*B)"));
            vec![inv, nand]
        });

        for policy in [CellPolicy::SmallNormalVt, CellPolicy::MaxSpeed] {
            assert!(select_inv_cell(&lib, policy).is_err());
            assert!(select_nand2_cell(&lib, policy).is_err());
        }
    }

    #[test]
    fn select_rejects_mixed_ordered_and_unordered_vt_metadata() {
        let mut lib = make_library_with_cells(|library| {
            vec![
                make_inv_cell(library, "INVx1_nominal", 1),
                make_inv_cell(library, "INVx2_unclassified", 0),
            ]
        });
        lib.threshold_voltage_groups = vec!["nominal".to_string()];
        lib.threshold_voltage_group_class_indices = vec![0];

        let err = select_inv_cell_name(&lib, CellPolicy::MaxSpeed).unwrap_err();
        assert!(err.contains("mix ordered and unordered threshold-voltage metadata"));
    }
}
