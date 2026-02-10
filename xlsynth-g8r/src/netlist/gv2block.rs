// SPDX-License-Identifier: Apache-2.0

//! Convert a gate-level netlist + Liberty proto into PIR Block IR.

use crate::liberty::cell_formula::{EmitContext as FormulaEmitContext, Term, parse_formula};
use crate::liberty::indexed::IndexedLibrary;
use crate::liberty_proto::{Cell, PinDirection, SequentialKind};
use crate::netlist::io::{ParsedNetlist, load_liberty_from_path, parse_netlist_from_path};
use crate::netlist::parse::{Net, NetIndex, NetRef, NetlistInstance, NetlistModule};
use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use xlsynth::IrValue;
use xlsynth_pir::ir::{
    BlockMetadata, BlockResetMetadata, Fn as PirFn, Instantiation, NaryOp, Node, NodePayload,
    NodeRef, Package, PackageMember, Param, ParamId, Register, Type, Unop,
};

pub fn convert_gv2block_paths(netlist_path: &Path, liberty_proto_path: &Path) -> Result<Package> {
    let parsed = parse_netlist_from_path(netlist_path)?;
    if parsed.modules.len() != 1 {
        return Err(anyhow!(format!(
            "expected exactly one module, got {}",
            parsed.modules.len()
        )));
    }
    let module = &parsed.modules[0];
    let liberty_lib = load_liberty_from_path(liberty_proto_path)?;
    let lib_indexed = IndexedLibrary::new(liberty_lib);
    build_package_from_netlist(module, &parsed, &lib_indexed)
}

pub fn convert_gv2block_paths_to_string(
    netlist_path: &Path,
    liberty_proto_path: &Path,
) -> Result<String> {
    Ok(convert_gv2block_paths(netlist_path, liberty_proto_path)?.to_string())
}

/// Canonical clock/output pin names for a Liberty clock-gate cell.
#[derive(Clone)]
struct ClockGatePassthroughSpec {
    clock_pin: String,
    output_pin: String,
}

/// Net connections for one elided clock-gate instance in the netlist.
#[derive(Clone)]
struct ClockGatePassthroughInstance {
    instance_name: String,
    clock_ref: NetRef,
    output_ref: NetRef,
}

/// Aggregated clock-gate passthrough data used while building the top block.
struct ClockGatePassthroughAnalysis {
    /// Per-instance clock->output passthrough wiring for elided clock gates.
    passthroughs: Vec<ClockGatePassthroughInstance>,
    /// Clock gaten istance names that should be skipped from normal block
    /// instantiation.
    elided_instance_names: HashSet<String>,
    /// Alias map from clock-gate output net to its source clock net.
    net_aliases: HashMap<NetIndex, NetIndex>,
}

fn get_clock_gate_passthrough_spec(cell: &Cell) -> Result<Option<ClockGatePassthroughSpec>> {
    let Some(clock_gate) = cell.clock_gate.as_ref() else {
        return Ok(None);
    };

    let clock_pin = if !clock_gate.clock_pin.is_empty() {
        clock_gate.clock_pin.clone()
    } else {
        let candidates: Vec<String> = cell
            .pins
            .iter()
            .filter(|p| p.is_clocking_pin)
            .map(|p| p.name.clone())
            .collect();
        match candidates.as_slice() {
            [pin] => pin.clone(),
            [] => {
                return Err(anyhow!(format!(
                    "cell '{}' has clock_gate metadata but no clock_pin and no is_clocking_pin input",
                    cell.name
                )));
            }
            _ => {
                return Err(anyhow!(format!(
                    "cell '{}' has clock_gate metadata but ambiguous clock pin candidates {:?}",
                    cell.name, candidates
                )));
            }
        }
    };

    let output_pin = if !clock_gate.output_pin.is_empty() {
        clock_gate.output_pin.clone()
    } else {
        let candidates: Vec<String> = cell
            .pins
            .iter()
            .filter(|p| p.direction == PinDirection::Output as i32)
            .map(|p| p.name.clone())
            .collect();
        match candidates.as_slice() {
            [pin] => pin.clone(),
            [] => {
                return Err(anyhow!(format!(
                    "cell '{}' has clock_gate metadata but no output_pin and no output pins",
                    cell.name
                )));
            }
            _ => {
                return Err(anyhow!(format!(
                    "cell '{}' has clock_gate metadata but ambiguous output pin candidates {:?}",
                    cell.name, candidates
                )));
            }
        }
    };

    let clock_pin_ok = cell
        .pins
        .iter()
        .any(|p| p.name == clock_pin && p.direction == PinDirection::Input as i32);
    if !clock_pin_ok {
        return Err(anyhow!(format!(
            "cell '{}' clock_gate clock pin '{}' is missing or not INPUT",
            cell.name, clock_pin
        )));
    }

    let output_pin_ok = cell
        .pins
        .iter()
        .any(|p| p.name == output_pin && p.direction == PinDirection::Output as i32);
    if !output_pin_ok {
        return Err(anyhow!(format!(
            "cell '{}' clock_gate output pin '{}' is missing or not OUTPUT",
            cell.name, output_pin
        )));
    }

    Ok(Some(ClockGatePassthroughSpec {
        clock_pin,
        output_pin,
    }))
}

fn instance_connection_for_port(
    inst: &NetlistInstance,
    parsed: &ParsedNetlist,
    port_name: &str,
) -> Option<NetRef> {
    inst.connections.iter().find_map(|(port_id, net_ref)| {
        let Some(name) = parsed.interner.resolve(*port_id) else {
            return None;
        };
        if name == port_name {
            return Some(net_ref.clone());
        }
        None
    })
}

fn resolve_net_alias(idx: NetIndex, aliases: &HashMap<NetIndex, NetIndex>) -> Result<NetIndex> {
    let mut visited: HashSet<NetIndex> = HashSet::new();
    let mut cur = idx;
    while let Some(next) = aliases.get(&cur).copied() {
        if !visited.insert(cur) {
            return Err(anyhow!(format!(
                "clock-gate net alias cycle detected at {:?}",
                cur
            )));
        }
        cur = next;
    }
    Ok(cur)
}

fn net_ref_to_simple_index(net_ref: &NetRef) -> Option<NetIndex> {
    match net_ref {
        NetRef::Simple(idx) => Some(*idx),
        _ => None,
    }
}

fn collect_clock_gate_passthroughs(
    module: &NetlistModule,
    parsed: &ParsedNetlist,
    lib_indexed: &IndexedLibrary,
) -> Result<ClockGatePassthroughAnalysis> {
    let mut passthroughs: Vec<ClockGatePassthroughInstance> = Vec::new();
    let mut elided_instance_names: HashSet<String> = HashSet::new();
    let mut net_aliases: HashMap<NetIndex, NetIndex> = HashMap::new();

    for inst in &module.instances {
        let inst_name = parsed
            .interner
            .resolve(inst.instance_name)
            .unwrap_or("<unknown>")
            .to_string();
        let cell_name = parsed.interner.resolve(inst.type_name).unwrap_or("");
        let cell = lib_indexed
            .get_cell(cell_name)
            .ok_or_else(|| anyhow!(format!("cell '{}' not found in Liberty", cell_name)))?;
        let Some(spec) = get_clock_gate_passthrough_spec(cell)? else {
            continue;
        };

        let clock_ref =
            instance_connection_for_port(inst, parsed, &spec.clock_pin).ok_or_else(|| {
                anyhow!(format!(
                    "clock-gate instance '{}' missing connection for clock pin '{}'",
                    inst_name, spec.clock_pin
                ))
            })?;
        let output_ref =
            instance_connection_for_port(inst, parsed, &spec.output_pin).ok_or_else(|| {
                anyhow!(format!(
                    "clock-gate instance '{}' missing connection for output pin '{}'",
                    inst_name, spec.output_pin
                ))
            })?;

        let clock_idx = net_ref_to_simple_index(&clock_ref).ok_or_else(|| {
            anyhow!(format!(
                "clock-gate instance '{}' clock pin '{}' is not connected to a simple net",
                inst_name, spec.clock_pin
            ))
        })?;
        let output_idx = net_ref_to_simple_index(&output_ref).ok_or_else(|| {
            anyhow!(format!(
                "clock-gate instance '{}' output pin '{}' is not connected to a simple net",
                inst_name, spec.output_pin
            ))
        })?;

        if let Some(existing) = net_aliases.insert(output_idx, clock_idx) {
            if existing != clock_idx {
                return Err(anyhow!(format!(
                    "clock-gate output net '{}' is aliased to multiple clock nets",
                    net_name_by_index(output_idx, parsed)
                )));
            }
        }

        elided_instance_names.insert(inst_name.clone());
        passthroughs.push(ClockGatePassthroughInstance {
            instance_name: inst_name,
            clock_ref,
            output_ref,
        });
    }

    Ok(ClockGatePassthroughAnalysis {
        passthroughs,
        elided_instance_names,
        net_aliases,
    })
}

fn apply_clock_gate_passthroughs(
    passthroughs: &[ClockGatePassthroughInstance],
    parsed: &ParsedNetlist,
    net_drivers: &mut NetDrivers,
    net_widths: &HashMap<NetIndex, (usize, i64)>,
    b: &mut PirFnBuilder,
) -> Result<HashSet<NetIndex>> {
    let mut pending_passthroughs = passthroughs.to_vec();
    let mut unresolved_passthrough_output_nets: HashSet<NetIndex> = HashSet::new();

    while !pending_passthroughs.is_empty() {
        let mut progressed = false;
        let mut next_pending: Vec<ClockGatePassthroughInstance> = Vec::new();
        for passthrough in pending_passthroughs {
            if !net_ref_is_resolved(&passthrough.clock_ref, net_drivers, net_widths) {
                next_pending.push(passthrough);
                continue;
            }

            let source_node = net_ref_to_node(
                &passthrough.clock_ref,
                parsed,
                net_drivers,
                net_widths,
                b,
                1,
            )?;
            match &passthrough.output_ref {
                NetRef::Simple(net_idx) => {
                    net_drivers.set_whole(*net_idx, source_node, net_widths, parsed)?;
                }
                NetRef::BitSelect(net_idx, bit) => {
                    net_drivers.set_bit(
                        *net_idx,
                        i64::from(*bit),
                        source_node,
                        net_widths,
                        parsed,
                    )?;
                }
                NetRef::PartSelect(net_idx, msb, lsb) => {
                    if msb != lsb {
                        return Err(anyhow!(format!(
                            "clock-gate instance '{}' output connection is unsupported multi-bit part-select on net '{}'",
                            passthrough.instance_name,
                            net_name_by_index(*net_idx, parsed)
                        )));
                    }
                    net_drivers.set_bit(
                        *net_idx,
                        i64::from(*lsb),
                        source_node,
                        net_widths,
                        parsed,
                    )?;
                }
                NetRef::Unconnected => {}
                _ => {
                    return Err(anyhow!(format!(
                        "clock-gate instance '{}' output connection is unsupported net reference",
                        passthrough.instance_name
                    )));
                }
            }
            progressed = true;
        }
        if !progressed {
            for passthrough in &next_pending {
                if let Some(output_idx) = net_ref_to_simple_index(&passthrough.output_ref) {
                    unresolved_passthrough_output_nets.insert(output_idx);
                }
            }
            break;
        }
        pending_passthroughs = next_pending;
    }

    Ok(unresolved_passthrough_output_nets)
}

fn build_package_from_netlist(
    module: &NetlistModule,
    parsed: &ParsedNetlist,
    lib_indexed: &IndexedLibrary,
) -> Result<Package> {
    let module_name = parsed
        .interner
        .resolve(module.name)
        .unwrap_or("top")
        .to_string();

    let mut pkg = Package {
        name: module_name.clone(),
        file_table: xlsynth_pir::ir::FileTable::new(),
        members: Vec::new(),
        top: None,
    };

    let mut needed_cells: HashSet<String> = HashSet::new();
    for inst in &module.instances {
        if let Some(name) = parsed.interner.resolve(inst.type_name) {
            needed_cells.insert(name.to_string());
        }
    }

    let mut cell_blocks: Vec<(String, PirFn, BlockMetadata)> = Vec::new();
    for cell_name in needed_cells.iter() {
        let cell = lib_indexed
            .get_cell(cell_name)
            .ok_or_else(|| anyhow!(format!("cell '{}' not found in Liberty proto", cell_name)))?;
        if get_clock_gate_passthrough_spec(cell)?.is_some() {
            continue;
        }
        let (f, meta) = build_cell_block(cell, lib_indexed)?;
        cell_blocks.push((cell_name.to_string(), f, meta));
    }
    cell_blocks.sort_by(|a, b| a.0.cmp(&b.0));

    for (_name, f, meta) in &cell_blocks {
        pkg.members.push(PackageMember::Block {
            func: f.clone(),
            metadata: meta.clone(),
        });
    }

    let (top_fn, top_meta) = build_top_block(module, parsed, lib_indexed)?;
    pkg.members.push(PackageMember::Block {
        func: top_fn.clone(),
        metadata: top_meta,
    });
    pkg.set_top_block(&top_fn.name).map_err(|e| anyhow!(e))?;

    Ok(pkg)
}

struct PirFnBuilder {
    name: String,
    params: Vec<Param>,
    nodes: Vec<Node>,
    next_text_id: usize,
}

impl PirFnBuilder {
    fn new(name: &str) -> Self {
        let mut nodes = Vec::new();
        nodes.push(Node {
            text_id: 0,
            name: None,
            ty: Type::Tuple(vec![]),
            payload: NodePayload::Nil,
            pos: None,
        });
        Self {
            name: name.to_string(),
            params: Vec::new(),
            nodes,
            next_text_id: 1,
        }
    }

    fn add_param(&mut self, name: &str, ty: Type) -> NodeRef {
        let param_id = ParamId::new(self.params.len() + 1);
        self.params.push(Param {
            name: name.to_string(),
            ty: ty.clone(),
            id: param_id,
        });
        let node_ref = NodeRef {
            index: self.nodes.len(),
        };
        self.nodes.push(Node {
            text_id: self.next_text_id,
            name: Some(name.to_string()),
            ty,
            payload: NodePayload::GetParam(param_id),
            pos: None,
        });
        self.next_text_id += 1;
        node_ref
    }

    fn add_node(&mut self, payload: NodePayload, ty: Type, name: Option<&str>) -> NodeRef {
        let node_ref = NodeRef {
            index: self.nodes.len(),
        };
        self.nodes.push(Node {
            text_id: self.next_text_id,
            name: name.map(|s| s.to_string()),
            ty,
            payload,
            pos: None,
        });
        self.next_text_id += 1;
        node_ref
    }

    fn add_literal_bits(&mut self, width: usize, value: u64) -> NodeRef {
        let lit = IrValue::make_ubits(width, value).unwrap();
        self.add_node(NodePayload::Literal(lit), Type::Bits(width), None)
    }

    fn finish(self, ret_ty: Type, ret_node_ref: Option<NodeRef>) -> PirFn {
        PirFn {
            name: self.name,
            params: self.params,
            ret_ty,
            nodes: self.nodes,
            ret_node_ref,
            outer_attrs: Vec::new(),
            inner_attrs: Vec::new(),
        }
    }
}

fn build_cell_block(cell: &Cell, lib_indexed: &IndexedLibrary) -> Result<(PirFn, BlockMetadata)> {
    let mut b = PirFnBuilder::new(&cell.name);
    let mut input_map: HashMap<String, NodeRef> = HashMap::new();
    let mut output_names: Vec<String> = Vec::new();

    let inputs = lib_indexed
        .pins_for_dir(&cell.name, PinDirection::Input)
        .unwrap_or_default();
    let outputs = lib_indexed
        .pins_for_dir(&cell.name, PinDirection::Output)
        .unwrap_or_default();

    let mut clock_pin_from_seq: Option<String> = None;
    if let Some(seq) = cell.sequential.first() {
        if !seq.clock_expr.is_empty() {
            let (clk_pin_name, is_neg) = parse_simple_clock_expr(&seq.clock_expr)?;
            if is_neg {
                return Err(anyhow!(format!(
                    "cell '{}' clock expression '{}' uses negation (unsupported)",
                    cell.name, seq.clock_expr
                )));
            }
            clock_pin_from_seq = Some(clk_pin_name);
        }
    }
    let clock_pin_from_pin = inputs
        .iter()
        .find(|p| p.is_clocking_pin)
        .map(|p| p.name.clone());
    if let (Some(a), Some(b)) = (&clock_pin_from_seq, &clock_pin_from_pin) {
        if a != b {
            return Err(anyhow!(format!(
                "cell '{}' clock pin mismatch: seq '{}' vs pin '{}'",
                cell.name, a, b
            )));
        }
    }
    let clock_pin = clock_pin_from_seq.or(clock_pin_from_pin);

    for pin in inputs.iter() {
        if clock_pin.as_ref().is_some_and(|c| c == &pin.name) {
            continue;
        }
        let nr = b.add_param(&pin.name, Type::Bits(1));
        input_map.insert(pin.name.clone(), nr);
    }

    let mut registers: Vec<Register> = Vec::new();
    let mut reset_meta: Option<BlockResetMetadata> = None;
    let mut state_var_name: Option<String> = None;
    let mut state_ref: Option<NodeRef> = None;
    let mut reset_node_ref: Option<NodeRef> = None;

    if let Some(seq) = cell.sequential.first() {
        if seq.kind != SequentialKind::Ff as i32 {
            return Err(anyhow!(format!(
                "cell '{}' uses unsupported sequential kind {:?}",
                cell.name, seq.kind
            )));
        }
        let state_var = seq.state_var.clone();
        let reg_name = format!("{state_var}_reg");
        state_var_name = Some(state_var.clone());
        registers.push(Register {
            name: reg_name.clone(),
            ty: Type::Bits(1),
            reset_value: None,
        });
        let reg_read = b.add_node(
            NodePayload::RegisterRead {
                register: reg_name.clone(),
            },
            Type::Bits(1),
            Some(&format!("{}_q", state_var)),
        );
        state_ref = Some(reg_read);
        input_map.insert(state_var.clone(), reg_read);

        if !seq.clear_expr.is_empty() || !seq.preset_expr.is_empty() {
            let (expr, reset_value) = if !seq.clear_expr.is_empty() {
                (seq.clear_expr.as_str(), 0u64)
            } else {
                (seq.preset_expr.as_str(), 1u64)
            };
            let (port_name, active_low) = parse_simple_reset_expr(expr)?;
            if !input_map.contains_key(&port_name) {
                let nr = b.add_param(&port_name, Type::Bits(1));
                input_map.insert(port_name.clone(), nr);
            }
            let reset_ref = input_map
                .get(&port_name)
                .copied()
                .ok_or_else(|| anyhow!(format!("reset port '{}' not found", port_name)))?;
            reset_node_ref = Some(reset_ref);
            reset_meta = Some(BlockResetMetadata {
                port_name: port_name.clone(),
                asynchronous: true,
                active_low,
            });
            registers[0].reset_value = Some(IrValue::make_ubits(1, reset_value).unwrap());
        }

        if seq.clock_expr.is_empty() && clock_pin.is_none() {
            return Err(anyhow!(format!(
                "cell '{}' sequential entry missing clock expression",
                cell.name
            )));
        }
        if !seq.clock_expr.is_empty() && clock_pin.is_none() {
            return Err(anyhow!(format!(
                "cell '{}' clock expression missing resolved clock pin",
                cell.name
            )));
        }

        if !seq.next_state.is_empty() {
            let term = parse_formula(&seq.next_state).map_err(|e| anyhow!(e))?;
            let next_ref = emit_term_as_pir(
                &term,
                &mut b,
                &input_map,
                &FormulaEmitContext {
                    cell_name: &cell.name,
                    original_formula: &seq.next_state,
                    instance_name: None,
                    port_map: None,
                },
            )?;
            b.add_node(
                NodePayload::RegisterWrite {
                    arg: next_ref,
                    register: reg_name.clone(),
                    load_enable: None,
                    reset: reset_node_ref,
                },
                Type::Tuple(vec![]),
                Some(&format!("{}_d", state_var)),
            );
        }
    }

    let mut output_nodes: Vec<NodeRef> = Vec::new();
    for pin in outputs.iter() {
        output_names.push(pin.name.clone());
        let term = if pin.function.is_empty() {
            None
        } else {
            Some(parse_formula(&pin.function).map_err(|e| anyhow!(e))?)
        };
        let node = if let Some(t) = term {
            emit_term_as_pir(
                &t,
                &mut b,
                &input_map,
                &FormulaEmitContext {
                    cell_name: &cell.name,
                    original_formula: &pin.function,
                    instance_name: None,
                    port_map: None,
                },
            )?
        } else if let (Some(state_var), Some(state_ref)) = (&state_var_name, state_ref) {
            if pin.name == *state_var {
                state_ref
            } else {
                return Err(anyhow!(format!(
                    "cell '{}' output pin '{}' missing function",
                    cell.name, pin.name
                )));
            }
        } else {
            return Err(anyhow!(format!(
                "cell '{}' output pin '{}' missing function",
                cell.name, pin.name
            )));
        };
        output_nodes.push(node);
    }

    let (ret_ty, ret_node_ref) = build_return_node(&mut b, &output_nodes);

    let mut input_port_ids: HashMap<String, usize> = HashMap::new();
    for p in &b.params {
        input_port_ids.insert(p.name.clone(), p.id.get_wrapped_id());
    }

    let mut output_port_ids: HashMap<String, usize> = HashMap::new();
    let mut next_out_id = b.nodes.iter().map(|n| n.text_id).max().unwrap_or(0) + 1;
    for name in &output_names {
        output_port_ids.insert(name.clone(), next_out_id);
        next_out_id += 1;
    }

    let meta = BlockMetadata {
        clock_port_name: clock_pin,
        input_port_ids,
        output_port_ids,
        output_names,
        reset: reset_meta,
        registers,
        instantiations: Vec::new(),
    };

    Ok((b.finish(ret_ty, ret_node_ref), meta))
}

struct NetDrivers {
    whole: HashMap<NetIndex, NodeRef>,
    bits: HashMap<NetIndex, Vec<Option<NodeRef>>>,
}

impl NetDrivers {
    fn new() -> Self {
        Self {
            whole: HashMap::new(),
            bits: HashMap::new(),
        }
    }

    fn set_whole(
        &mut self,
        idx: NetIndex,
        node: NodeRef,
        net_widths: &HashMap<NetIndex, (usize, i64)>,
        parsed: &ParsedNetlist,
    ) -> Result<()> {
        if self.bits.contains_key(&idx) {
            let net_name = net_name_by_index(idx, parsed);
            return Err(anyhow!(format!(
                "multiple drivers for net '{}' (bit-level and whole-net)",
                net_name
            )));
        }
        if self.whole.insert(idx, node).is_some() {
            let net_name = net_name_by_index(idx, parsed);
            return Err(anyhow!(format!("multiple drivers for net '{}'", net_name)));
        }
        // Ensure widths map entry exists for later slices.
        let _ = net_widths.get(&idx).copied().unwrap_or((1, 0));
        Ok(())
    }

    fn set_bit(
        &mut self,
        idx: NetIndex,
        bit: i64,
        node: NodeRef,
        net_widths: &HashMap<NetIndex, (usize, i64)>,
        parsed: &ParsedNetlist,
    ) -> Result<()> {
        if self.whole.contains_key(&idx) {
            let net_name = net_name_by_index(idx, parsed);
            return Err(anyhow!(format!(
                "multiple drivers for net '{}' (whole-net and bit-level)",
                net_name
            )));
        }
        let (width, lsb) = net_widths.get(&idx).copied().unwrap_or((1, 0));
        let offset = (bit - lsb) as usize;
        if offset >= width {
            let net_name = net_name_by_index(idx, parsed);
            return Err(anyhow!(format!(
                "bit {} out of range for net '{}' (width {}, lsb {})",
                bit, net_name, width, lsb
            )));
        }
        let entry = self.bits.entry(idx).or_insert_with(|| vec![None; width]);
        if entry[offset].is_some() {
            let net_name = net_name_by_index(idx, parsed);
            return Err(anyhow!(format!("multiple drivers for net '{}'", net_name)));
        }
        entry[offset] = Some(node);
        Ok(())
    }

    fn get_whole(&self, idx: NetIndex) -> Option<NodeRef> {
        self.whole.get(&idx).copied()
    }

    fn get_bits(&self, idx: NetIndex) -> Option<&Vec<Option<NodeRef>>> {
        self.bits.get(&idx)
    }
}

fn build_top_block(
    module: &NetlistModule,
    parsed: &ParsedNetlist,
    lib_indexed: &IndexedLibrary,
) -> Result<(PirFn, BlockMetadata)> {
    let module_name = parsed
        .interner
        .resolve(module.name)
        .unwrap_or("top")
        .to_string();
    let mut b = PirFnBuilder::new(&module_name);

    let mut net_drivers = NetDrivers::new();
    let mut net_widths: HashMap<NetIndex, (usize, i64)> = HashMap::new();

    for (i, net) in parsed.nets.iter().enumerate() {
        let idx = NetIndex(i);
        net_widths.insert(idx, net_width_bits(net));
    }

    let ClockGatePassthroughAnalysis {
        passthroughs: clock_gate_passthroughs,
        elided_instance_names: elided_clock_gate_instances,
        net_aliases: clock_gate_net_aliases,
    } = collect_clock_gate_passthroughs(module, parsed, lib_indexed)?;

    // Identify a shared clock net name for DFFs (if any).
    let mut clock_net_name: Option<String> = None;
    let mut dff_clock_pins: Vec<String> = Vec::new();

    for inst in &module.instances {
        let cell_name = parsed.interner.resolve(inst.type_name).unwrap_or("");
        let cell = lib_indexed
            .get_cell(cell_name)
            .ok_or_else(|| anyhow!(format!("cell '{}' not found in Liberty", cell_name)))?;
        if let Some(seq) = cell.sequential.first() {
            if seq.kind == SequentialKind::Ff as i32 {
                for pin in &cell.pins {
                    if pin.is_clocking_pin {
                        dff_clock_pins.push(pin.name.clone());
                    }
                }
                if !seq.clock_expr.is_empty() {
                    let (clk_pin, _neg) = parse_simple_clock_expr(&seq.clock_expr)?;
                    dff_clock_pins.push(clk_pin);
                }
            }
        }
    }

    for inst in &module.instances {
        let cell_name = parsed.interner.resolve(inst.type_name).unwrap_or("");
        let cell = lib_indexed
            .get_cell(cell_name)
            .ok_or_else(|| anyhow!(format!("cell '{}' not found in Liberty", cell_name)))?;
        if cell
            .sequential
            .first()
            .is_some_and(|s| s.kind == SequentialKind::Ff as i32)
        {
            for (port_name, net_ref) in &inst.connections {
                let port = parsed.interner.resolve(*port_name).unwrap_or("");
                if dff_clock_pins.iter().any(|p| p == port) {
                    let net_idx = net_ref_to_simple_index(net_ref).ok_or_else(|| {
                        anyhow!(format!(
                            "clock pin '{}' on instance '{}' is not driven by a simple net",
                            port,
                            parsed
                                .interner
                                .resolve(inst.instance_name)
                                .unwrap_or("<unknown>")
                        ))
                    })?;
                    let resolved_net_idx = resolve_net_alias(net_idx, &clock_gate_net_aliases)?;
                    let net_name = net_name_by_index(resolved_net_idx, parsed);
                    match &clock_net_name {
                        None => clock_net_name = Some(net_name),
                        Some(existing) if existing != &net_name => {
                            return Err(anyhow!(format!(
                                "multiple clock nets detected: '{}' vs '{}'",
                                existing, net_name
                            )));
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    if let Some(ref clk_net) = clock_net_name {
        let is_top_input = module.ports.iter().any(|p| {
            p.direction == crate::netlist::parse::PortDirection::Input
                && parsed
                    .interner
                    .resolve(p.name)
                    .is_some_and(|name| name == clk_net)
        });
        if !is_top_input {
            return Err(anyhow!(format!(
                "derived clock '{}' is not a top-level input port",
                clk_net
            )));
        }
    }

    for port in &module.ports {
        if port.direction != crate::netlist::parse::PortDirection::Input {
            continue;
        }
        let name = parsed.interner.resolve(port.name).unwrap_or("").to_string();
        if clock_net_name.as_ref().is_some_and(|clk| clk == &name) {
            continue;
        }
        let width = port
            .width
            .map(|(msb, lsb)| (msb as i64 - lsb as i64).abs() as usize + 1)
            .unwrap_or(1);
        let nr = b.add_param(&name, Type::Bits(width));
        let net_idx = parsed
            .nets
            .iter()
            .position(|n| n.name == port.name)
            .map(NetIndex)
            .ok_or_else(|| anyhow!(format!("input port net '{}' not found", name)))?;
        net_drivers.set_whole(net_idx, nr, &net_widths, parsed)?;
    }

    let mut instantiations: Vec<Instantiation> = Vec::new();
    let mut instance_outputs: Vec<(String, Vec<(String, NodeRef)>)> = Vec::new();

    for inst in &module.instances {
        let inst_name = parsed
            .interner
            .resolve(inst.instance_name)
            .unwrap_or("")
            .to_string();
        if elided_clock_gate_instances.contains(&inst_name) {
            continue;
        }
        let cell_name = parsed
            .interner
            .resolve(inst.type_name)
            .unwrap_or("")
            .to_string();
        instantiations.push(Instantiation {
            name: inst_name.clone(),
            block: cell_name.clone(),
        });

        let outputs = lib_indexed
            .pins_for_dir(&cell_name, PinDirection::Output)
            .unwrap_or_default();
        let mut output_refs: Vec<(String, NodeRef)> = Vec::new();
        for pin in outputs {
            let output_node_name = format!("{}_{}", inst_name, pin.name);
            let nr = b.add_node(
                NodePayload::InstantiationOutput {
                    instantiation: inst_name.clone(),
                    port_name: pin.name.clone(),
                },
                Type::Bits(1),
                Some(&output_node_name),
            );
            output_refs.push((pin.name.clone(), nr));
        }
        instance_outputs.push((inst_name, output_refs));
    }

    // Map instance outputs to nets.
    for (inst_name, output_refs) in &instance_outputs {
        let inst = module
            .instances
            .iter()
            .find(|i| parsed.interner.resolve(i.instance_name).unwrap_or("") == inst_name)
            .ok_or_else(|| anyhow!(format!("instance '{}' not found", inst_name)))?;
        for (port_id, net_ref) in &inst.connections {
            let port_name = parsed.interner.resolve(*port_id).unwrap_or("").to_string();
            let Some(node_ref) = output_refs
                .iter()
                .find(|(n, _)| n == &port_name)
                .map(|(_, r)| *r)
            else {
                continue;
            };
            match net_ref {
                NetRef::Simple(net_idx) => {
                    net_drivers.set_whole(*net_idx, node_ref, &net_widths, parsed)?;
                }
                NetRef::BitSelect(net_idx, bit) => {
                    net_drivers.set_bit(*net_idx, *bit as i64, node_ref, &net_widths, parsed)?;
                }
                NetRef::PartSelect(net_idx, msb, lsb) => {
                    if msb != lsb {
                        return Err(anyhow!(format!(
                            "unsupported multi-bit part-select output for net '{}'",
                            net_name_by_index(*net_idx, parsed)
                        )));
                    }
                    net_drivers.set_bit(*net_idx, *lsb as i64, node_ref, &net_widths, parsed)?;
                }
                _ => {}
            }
        }
    }

    let unresolved_passthrough_output_nets = apply_clock_gate_passthroughs(
        &clock_gate_passthroughs,
        parsed,
        &mut net_drivers,
        &net_widths,
        &mut b,
    )?;

    // Emit instantiation inputs.
    for inst in &module.instances {
        let inst_name = parsed
            .interner
            .resolve(inst.instance_name)
            .unwrap_or("")
            .to_string();
        if elided_clock_gate_instances.contains(&inst_name) {
            continue;
        }
        let cell_name = parsed
            .interner
            .resolve(inst.type_name)
            .unwrap_or("")
            .to_string();
        let cell = lib_indexed
            .get_cell(&cell_name)
            .ok_or_else(|| anyhow!(format!("cell '{}' not found in Liberty", cell_name)))?;
        let inputs = lib_indexed
            .pins_for_dir(&cell_name, PinDirection::Input)
            .unwrap_or_default();
        let input_pin_names: HashSet<String> = inputs.iter().map(|p| p.name.clone()).collect();

        let mut clock_pin_names: HashSet<String> = inputs
            .iter()
            .filter(|p| p.is_clocking_pin)
            .map(|p| p.name.clone())
            .collect();
        if let Some(seq) = cell.sequential.first() {
            if !seq.clock_expr.is_empty() {
                let (clk_pin, _neg) = parse_simple_clock_expr(&seq.clock_expr)?;
                clock_pin_names.insert(clk_pin);
            }
        }

        for (port_id, net_ref) in &inst.connections {
            let port_name = parsed.interner.resolve(*port_id).unwrap_or("").to_string();
            if !input_pin_names.contains(&port_name) || clock_pin_names.contains(&port_name) {
                continue;
            }
            if let (Some(clock_net), Some(net_idx)) =
                (clock_net_name.as_ref(), net_ref_to_simple_index(net_ref))
            {
                let resolved_net_idx = resolve_net_alias(net_idx, &clock_gate_net_aliases)?;
                if net_name_by_index(resolved_net_idx, parsed) == *clock_net {
                    return Err(anyhow!(format!(
                        "clock net '{}' is connected to non-clock input '{}.{}' (cell '{}'); gv2block does not support feeding the selected clock into ordinary logic",
                        clock_net, inst_name, port_name, cell_name
                    )));
                }
            }
            let val = net_ref_to_node(net_ref, parsed, &net_drivers, &net_widths, &mut b, 1)?;
            let input_node_name = format!("{}_{}", inst_name, port_name);
            b.add_node(
                NodePayload::InstantiationInput {
                    instantiation: inst_name.clone(),
                    port_name: port_name.clone(),
                    arg: val,
                },
                Type::Tuple(vec![]),
                Some(&input_node_name),
            );
        }
    }

    // Collect module outputs.
    let mut output_names: Vec<String> = Vec::new();
    let mut output_nodes: Vec<NodeRef> = Vec::new();
    for port in &module.ports {
        if port.direction != crate::netlist::parse::PortDirection::Output {
            continue;
        }
        let name = parsed.interner.resolve(port.name).unwrap_or("").to_string();
        let net_idx = parsed
            .nets
            .iter()
            .position(|n| n.name == port.name)
            .map(NetIndex)
            .ok_or_else(|| anyhow!(format!("output port net '{}' not found", name)))?;
        let (width, _) = net_widths.get(&net_idx).copied().unwrap_or((1, 0));
        let node = if let Some(nr) = net_drivers.get_whole(net_idx) {
            nr
        } else if let Some(bits) = net_drivers.get_bits(net_idx) {
            let mut parts: Vec<NodeRef> = Vec::new();
            for bit in bits.iter().rev() {
                let nr = match bit {
                    Some(n) => *n,
                    None => b.add_literal_bits(1, 0),
                };
                parts.push(nr);
            }
            if parts.len() == 1 {
                parts[0]
            } else {
                b.add_node(
                    NodePayload::Nary(NaryOp::Concat, parts),
                    Type::Bits(width),
                    None,
                )
            }
        } else {
            if unresolved_passthrough_output_nets.contains(&net_idx) {
                return Err(anyhow!(format!(
                    "clock-gate output net '{}' is unresolved as data (clock-only passthrough)",
                    name
                )));
            }
            b.add_literal_bits(width, 0)
        };
        output_names.push(name);
        output_nodes.push(node);
    }

    let (ret_ty, ret_node_ref) = build_return_node(&mut b, &output_nodes);

    let mut input_port_ids: HashMap<String, usize> = HashMap::new();
    for p in &b.params {
        input_port_ids.insert(p.name.clone(), p.id.get_wrapped_id());
    }

    let meta = BlockMetadata {
        clock_port_name: clock_net_name,
        input_port_ids,
        output_port_ids: HashMap::new(),
        output_names,
        reset: None,
        registers: Vec::new(),
        instantiations,
    };

    Ok((b.finish(ret_ty, ret_node_ref), meta))
}

fn build_return_node(b: &mut PirFnBuilder, outputs: &[NodeRef]) -> (Type, Option<NodeRef>) {
    if outputs.is_empty() {
        return (Type::Tuple(vec![]), None);
    }
    if outputs.len() == 1 {
        let ty = b.nodes[outputs[0].index].ty.clone();
        return (ty, Some(outputs[0]));
    }
    let tys = outputs
        .iter()
        .map(|nr| b.nodes[nr.index].ty.clone())
        .map(|t| Box::new(t))
        .collect::<Vec<_>>();
    let tuple_ty = Type::Tuple(tys);
    let tuple_node = b.add_node(NodePayload::Tuple(outputs.to_vec()), tuple_ty.clone(), None);
    (tuple_ty, Some(tuple_node))
}

fn net_width_bits(net: &Net) -> (usize, i64) {
    if let Some((msb, lsb)) = net.width {
        let width = (msb as i64 - lsb as i64).abs() as usize + 1;
        (width, lsb as i64)
    } else {
        (1, 0)
    }
}

fn net_name_by_index(idx: NetIndex, parsed: &ParsedNetlist) -> String {
    parsed
        .interner
        .resolve(parsed.nets[idx.0].name)
        .unwrap_or("<unknown>")
        .to_string()
}

fn net_ref_is_resolved(
    net_ref: &NetRef,
    net_drivers: &NetDrivers,
    net_widths: &HashMap<NetIndex, (usize, i64)>,
) -> bool {
    match net_ref {
        NetRef::Simple(idx) => {
            if net_drivers.get_whole(*idx).is_some() {
                return true;
            }
            net_drivers
                .get_bits(*idx)
                .is_some_and(|bits| bits.iter().all(|b| b.is_some()))
        }
        NetRef::BitSelect(idx, bit) => {
            if net_drivers.get_whole(*idx).is_some() {
                return true;
            }
            let Some(bits) = net_drivers.get_bits(*idx) else {
                return false;
            };
            let (_, lsb) = net_widths.get(idx).copied().unwrap_or((1, 0));
            let offset_i64 = i64::from(*bit) - lsb;
            if offset_i64 < 0 {
                return false;
            }
            bits.get(offset_i64 as usize).is_some_and(|n| n.is_some())
        }
        NetRef::PartSelect(idx, msb, lsb_select) => {
            if msb < lsb_select {
                return false;
            }
            if net_drivers.get_whole(*idx).is_some() {
                return true;
            }
            let Some(bits) = net_drivers.get_bits(*idx) else {
                return false;
            };
            let (_, net_lsb) = net_widths.get(idx).copied().unwrap_or((1, 0));
            for bit in *lsb_select..=*msb {
                let offset_i64 = i64::from(bit) - net_lsb;
                if offset_i64 < 0 {
                    return false;
                }
                let Some(nr_opt) = bits.get(offset_i64 as usize) else {
                    return false;
                };
                if nr_opt.is_none() {
                    return false;
                }
            }
            true
        }
        NetRef::Literal(_) | NetRef::Unconnected => true,
        NetRef::Concat(elems) => elems
            .iter()
            .all(|e| net_ref_is_resolved(e, net_drivers, net_widths)),
    }
}

fn net_ref_to_node(
    net_ref: &NetRef,
    parsed: &ParsedNetlist,
    net_drivers: &NetDrivers,
    net_widths: &HashMap<NetIndex, (usize, i64)>,
    b: &mut PirFnBuilder,
    expected_width: usize,
) -> Result<NodeRef> {
    match net_ref {
        NetRef::Simple(idx) => {
            if let Some(nr) = net_drivers.get_whole(*idx) {
                return Ok(nr);
            }
            if let Some(bits) = net_drivers.get_bits(*idx) {
                let (width, _lsb) = net_widths.get(idx).copied().unwrap_or((1, 0));
                let mut parts: Vec<NodeRef> = Vec::new();
                for bit in bits.iter().rev() {
                    let nr = match bit {
                        Some(n) => *n,
                        None => b.add_literal_bits(1, 0),
                    };
                    parts.push(nr);
                }
                if width == 1 {
                    return Ok(parts[0]);
                }
                return Ok(b.add_node(
                    NodePayload::Nary(NaryOp::Concat, parts),
                    Type::Bits(width),
                    None,
                ));
            }
            Err(anyhow!(format!(
                "net '{}' has no driver",
                net_name_by_index(*idx, parsed)
            )))
        }
        NetRef::BitSelect(idx, bit) => {
            if let Some(nr) = net_drivers.get_whole(*idx) {
                let (_, lsb) = net_widths.get(idx).copied().unwrap_or((1, 0));
                let start = (*bit as i64 - lsb) as usize;
                return Ok(b.add_node(
                    NodePayload::BitSlice {
                        arg: nr,
                        start,
                        width: 1,
                    },
                    Type::Bits(1),
                    None,
                ));
            }
            if let Some(bits) = net_drivers.get_bits(*idx) {
                let (_, lsb) = net_widths.get(idx).copied().unwrap_or((1, 0));
                let start = (*bit as i64 - lsb) as usize;
                if start >= bits.len() {
                    return Err(anyhow!(format!(
                        "net '{}' has no driver",
                        net_name_by_index(*idx, parsed)
                    )));
                }
                return bits[start].ok_or_else(|| {
                    anyhow!(format!(
                        "net '{}' has no driver",
                        net_name_by_index(*idx, parsed)
                    ))
                });
            }
            Err(anyhow!(format!(
                "net '{}' has no driver",
                net_name_by_index(*idx, parsed)
            )))
        }
        NetRef::PartSelect(idx, msb, lsb) => {
            if let Some(nr) = net_drivers.get_whole(*idx) {
                let (_, net_lsb) = net_widths.get(idx).copied().unwrap_or((1, 0));
                let (hi, lo) = (*msb as i64, *lsb as i64);
                if hi < lo {
                    return Err(anyhow!(format!(
                        "unsupported part-select with msb<lsb for net '{}'",
                        net_name_by_index(*idx, parsed)
                    )));
                }
                let width = (hi - lo + 1) as usize;
                let start = (lo - net_lsb) as usize;
                return Ok(b.add_node(
                    NodePayload::BitSlice {
                        arg: nr,
                        start,
                        width,
                    },
                    Type::Bits(width),
                    None,
                ));
            }
            if let Some(bits) = net_drivers.get_bits(*idx) {
                let (_, net_lsb) = net_widths.get(idx).copied().unwrap_or((1, 0));
                let (hi, lo) = (*msb as i64, *lsb as i64);
                if hi < lo {
                    return Err(anyhow!(format!(
                        "unsupported part-select with msb<lsb for net '{}'",
                        net_name_by_index(*idx, parsed)
                    )));
                }
                let mut parts: Vec<NodeRef> = Vec::new();
                for bit in (lo..=hi).rev() {
                    let offset = (bit - net_lsb) as usize;
                    let nr = bits.get(offset).and_then(|n| *n).ok_or_else(|| {
                        anyhow!(format!(
                            "net '{}' has no driver",
                            net_name_by_index(*idx, parsed)
                        ))
                    })?;
                    parts.push(nr);
                }
                let width = (hi - lo + 1) as usize;
                if width == 1 {
                    return Ok(parts[0]);
                }
                return Ok(b.add_node(
                    NodePayload::Nary(NaryOp::Concat, parts),
                    Type::Bits(width),
                    None,
                ));
            }
            Err(anyhow!(format!(
                "net '{}' has no driver",
                net_name_by_index(*idx, parsed)
            )))
        }
        NetRef::Literal(bits) => {
            let lit = IrValue::from_bits(bits);
            let width = bits.get_bit_count();
            Ok(b.add_node(NodePayload::Literal(lit), Type::Bits(width), None))
        }
        NetRef::Unconnected => Ok(b.add_literal_bits(expected_width, 0)),
        NetRef::Concat(elems) => {
            let mut parts: Vec<NodeRef> = Vec::new();
            let mut total_width = 0usize;
            for e in elems {
                let nr = net_ref_to_node(e, parsed, net_drivers, net_widths, b, 1)?;
                let ty = b.nodes[nr.index].ty.clone();
                let width = match ty {
                    Type::Bits(w) => w,
                    _ => return Err(anyhow!("concat element must be bits, got {:?}", ty)),
                };
                total_width += width;
                parts.push(nr);
            }
            Ok(b.add_node(
                NodePayload::Nary(NaryOp::Concat, parts),
                Type::Bits(total_width),
                None,
            ))
        }
    }
}

fn emit_term_as_pir(
    term: &Term,
    b: &mut PirFnBuilder,
    input_map: &HashMap<String, NodeRef>,
    ctx: &FormulaEmitContext<'_>,
) -> Result<NodeRef> {
    match term {
        Term::Input(name) => input_map.get(name).copied().ok_or_else(|| {
            anyhow!(format!(
                "input '{}' not found in map for cell '{}' (formula: \"{}\")",
                name, ctx.cell_name, ctx.original_formula
            ))
        }),
        Term::And(a, b_term) => {
            let a_nr = emit_term_as_pir(a, b, input_map, ctx)?;
            let b_nr = emit_term_as_pir(b_term, b, input_map, ctx)?;
            Ok(b.add_node(
                NodePayload::Nary(NaryOp::And, vec![a_nr, b_nr]),
                Type::Bits(1),
                None,
            ))
        }
        Term::Or(a, b_term) => {
            let a_nr = emit_term_as_pir(a, b, input_map, ctx)?;
            let b_nr = emit_term_as_pir(b_term, b, input_map, ctx)?;
            Ok(b.add_node(
                NodePayload::Nary(NaryOp::Or, vec![a_nr, b_nr]),
                Type::Bits(1),
                None,
            ))
        }
        Term::Xor(a, b_term) => {
            let a_nr = emit_term_as_pir(a, b, input_map, ctx)?;
            let b_nr = emit_term_as_pir(b_term, b, input_map, ctx)?;
            Ok(b.add_node(
                NodePayload::Nary(NaryOp::Xor, vec![a_nr, b_nr]),
                Type::Bits(1),
                None,
            ))
        }
        Term::Negate(inner) => {
            let inner_nr = emit_term_as_pir(inner, b, input_map, ctx)?;
            Ok(b.add_node(NodePayload::Unop(Unop::Not, inner_nr), Type::Bits(1), None))
        }
        Term::Constant(value) => Ok(b.add_literal_bits(1, if *value { 1 } else { 0 })),
    }
}

fn parse_simple_reset_expr(expr: &str) -> Result<(String, bool)> {
    let term = parse_formula(expr).map_err(|e| anyhow!(e))?;
    match term {
        Term::Input(name) => Ok((name, false)),
        Term::Negate(inner) => match *inner {
            Term::Input(name) => Ok((name, true)),
            _ => Err(anyhow!(format!(
                "reset expression '{}' is not a simple input or negated input",
                expr
            ))),
        },
        _ => Err(anyhow!(format!(
            "reset expression '{}' is not a simple input or negated input",
            expr
        ))),
    }
}

fn parse_simple_clock_expr(expr: &str) -> Result<(String, bool)> {
    let term = parse_formula(expr).map_err(|e| anyhow!(e))?;
    match term {
        Term::Input(name) => Ok((name, false)),
        _ => Err(anyhow!(format!(
            "clock expression '{}' is not a simple input",
            expr
        ))),
    }
}
