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
use crate::netlist::parse::{
    Net, NetIndex, NetRef, NetlistInstance, NetlistModule, NetlistPort, PortDirection,
};
use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};
use xlsynth::IrBits;

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

#[derive(Debug)]
pub struct StructuralMappedNetlist {
    pub module: NetlistModule,
    pub nets: Vec<Net>,
    pub interner: StringInterner<StringBackend<SymbolU32>>,
}

#[derive(Clone, Copy, Debug)]
enum TechSignal {
    Net(NetIndex),
    Literal(bool),
}

#[derive(Clone, Debug)]
struct OutputBitPlan {
    port_name: String,
    port_net: NetIndex,
    operand: AigOperand,
}

fn scalar_bit_name(base: &str, bit_index: usize, bit_count: usize) -> String {
    if bit_count == 1 {
        base.to_string()
    } else {
        format!("{}_{}", base, bit_index)
    }
}

fn literal_netref(value: bool) -> NetRef {
    NetRef::Literal(IrBits::make_ubits(1, if value { 1 } else { 0 }).expect("1-bit literal"))
}

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

pub fn map_gatefn_to_structural_netlist(
    gate_fn: &GateFn,
    options: &StructuralTechMapOptions,
) -> Result<StructuralMappedNetlist> {
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
                    "constant output '{}' is unsupported by structural tech mapping because the STA-compatible cell subset does not include constant drivers",
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

    Ok(StructuralMappedNetlist {
        module,
        nets,
        interner,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig_sim::gate_sim::{Collect, eval};
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::liberty_proto::{Cell, Library as LibertyLibrary, Pin, PinDirection};
    use crate::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty;
    use std::collections::HashSet;

    fn net_name(mapped: &StructuralMappedNetlist, idx: NetIndex) -> String {
        mapped
            .nets
            .get(idx.0)
            .and_then(|n| mapped.interner.resolve(n.name))
            .unwrap_or("<unknown>")
            .to_string()
    }

    fn sym_name(interner: &StringInterner<StringBackend<SymbolU32>>, sym: SymbolU32) -> String {
        interner.resolve(sym).unwrap_or("<unknown>").to_string()
    }

    fn make_inv_nand2_liberty() -> LibertyLibrary {
        LibertyLibrary {
            cells: vec![
                Cell {
                    name: "INV".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            function: "!A".to_string(),
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                Cell {
                    name: "NAND2".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "B".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            function: "!(A*B)".to_string(),
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

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

    fn render_instance_summary(mapped: &StructuralMappedNetlist) -> Vec<String> {
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
