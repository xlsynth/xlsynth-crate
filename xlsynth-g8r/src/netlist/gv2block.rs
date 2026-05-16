// SPDX-License-Identifier: Apache-2.0

//! Convert a gate-level netlist + Liberty proto into PIR Block IR.

use crate::liberty::cell_formula::{EmitContext as FormulaEmitContext, Term, parse_formula};
use crate::liberty::indexed::IndexedLibrary;
use crate::liberty_proto::{Cell, PinDirection, SequentialKind};
use crate::netlist::io::{ParsedNetlist, load_liberty_from_path, parse_netlist_from_path};
use crate::netlist::normalized::{
    BitExpr, BitIndex, BitSource, NormalizedInstance, NormalizedNetlistModule,
};
use crate::netlist::parse::NetlistModule;
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
    let normalized = NormalizedNetlistModule::new(module, &parsed.nets, &parsed.interner)?;
    build_package_from_normalized_netlist(module, &parsed, &lib_indexed, &normalized)
}

pub fn convert_gv2block_paths_to_string(
    netlist_path: &Path,
    liberty_proto_path: &Path,
) -> Result<String> {
    Ok(convert_gv2block_paths(netlist_path, liberty_proto_path)?.to_string())
}

fn sanitize_to_xls_identifier(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 1);
    for (idx, ch) in raw.chars().enumerate() {
        let is_valid = if idx == 0 {
            ch == '_' || ch.is_ascii_alphabetic()
        } else {
            ch == '_' || ch.is_ascii_alphanumeric()
        };
        if is_valid {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('_');
    } else if !out
        .as_bytes()
        .first()
        .is_some_and(|b| *b == b'_' || b.is_ascii_alphabetic())
    {
        out.insert(0, '_');
    }
    out
}

#[derive(Default)]
struct IdentifierLegalizer {
    raw_to_legal: HashMap<String, String>,
    used_legal_names: HashSet<String>,
}

impl IdentifierLegalizer {
    fn legalize(&mut self, raw: &str) -> String {
        if let Some(existing) = self.raw_to_legal.get(raw) {
            return existing.clone();
        }
        let base = sanitize_to_xls_identifier(raw);
        let mut candidate = base.clone();
        let mut suffix = 0usize;
        while self.used_legal_names.contains(&candidate) {
            suffix += 1;
            candidate = format!("{}_{}", base, suffix);
        }
        self.used_legal_names.insert(candidate.clone());
        self.raw_to_legal.insert(raw.to_string(), candidate.clone());
        candidate
    }
}

/// Canonical clock/output pin names for a Liberty clock-gate cell.
#[derive(Clone)]
struct ClockGatePassthroughSpec {
    clock_pin: String,
    output_pin: String,
}

#[derive(Clone)]
struct ClockGatePassthroughInstance {
    instance_name: String,
    clock_source: BitSource,
    output_bit: BitIndex,
}

struct ClockGatePassthroughAnalysis {
    passthroughs: Vec<ClockGatePassthroughInstance>,
    elided_instance_names: HashSet<String>,
    bit_aliases: HashMap<BitIndex, BitIndex>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolvedWiringSource {
    Bit(BitIndex),
    Literal(bool),
}

struct WiringAssignResolver {
    source_by_lhs: HashMap<BitIndex, BitSource>,
}

impl WiringAssignResolver {
    fn new(normalized: &NormalizedNetlistModule<'_>, parsed: &ParsedNetlist) -> Result<Self> {
        let mut source_by_lhs = HashMap::new();
        for assign in &normalized.assigns {
            for (lhs_bit, rhs_expr) in assign.lhs_bits.iter().copied().zip(&assign.rhs_bits) {
                let source = match rhs_expr {
                    BitExpr::Source(source) => *source,
                    BitExpr::Not(_)
                    | BitExpr::And(_, _)
                    | BitExpr::Or(_, _)
                    | BitExpr::Xor(_, _) => {
                        return Err(anyhow!(format!(
                            "gv2block only supports techmapped netlists; preserved combinational assign to '{}' at {} contains top-level logic; run technology mapping first",
                            normalized.render_bit(lhs_bit, &parsed.nets, &parsed.interner),
                            assign.span.to_human_string()
                        )));
                    }
                };
                if source_by_lhs.insert(lhs_bit, source).is_some() {
                    return Err(anyhow!(format!(
                        "gv2block found multiple preserved wiring assigns for '{}'",
                        normalized.render_bit(lhs_bit, &parsed.nets, &parsed.interner)
                    )));
                }
            }
        }
        let resolver = Self { source_by_lhs };
        for lhs_bit in resolver.source_by_lhs.keys().copied() {
            resolver.resolve_source(BitSource::Bit(lhs_bit), normalized, parsed)?;
        }
        Ok(resolver)
    }

    fn has_lhs(&self, bit_idx: BitIndex) -> bool {
        self.source_by_lhs.contains_key(&bit_idx)
    }

    fn resolve_source(
        &self,
        source: BitSource,
        normalized: &NormalizedNetlistModule<'_>,
        parsed: &ParsedNetlist,
    ) -> Result<ResolvedWiringSource> {
        let mut seen_bits = HashSet::new();
        let mut current = source;
        loop {
            match current {
                BitSource::Bit(bit_idx) => {
                    if !seen_bits.insert(bit_idx) {
                        return Err(anyhow!(format!(
                            "gv2block found a preserved wiring-assign cycle at '{}'",
                            normalized.render_bit(bit_idx, &parsed.nets, &parsed.interner)
                        )));
                    }
                    let Some(next_source) = self.source_by_lhs.get(&bit_idx).copied() else {
                        return Ok(ResolvedWiringSource::Bit(bit_idx));
                    };
                    current = next_source;
                }
                BitSource::Literal(value) => return Ok(ResolvedWiringSource::Literal(value)),
                BitSource::Unknown => {
                    return Err(anyhow!(
                        "gv2block does not support unknown literal preserved wiring"
                    ));
                }
            }
        }
    }
}

struct BitDrivers {
    drivers: Vec<Option<BitDriver>>,
}

/// One normalized bit driver; vector input bits stay lazy until consumed.
#[derive(Clone, Copy)]
enum BitDriver {
    Node(NodeRef),
    InputPortBit {
        port_node: NodeRef,
        bit_offset: usize,
        port_width: usize,
    },
}

impl BitDrivers {
    fn new(bit_count: usize) -> Self {
        Self {
            drivers: vec![None; bit_count],
        }
    }

    fn set_node_bit(
        &mut self,
        bit_idx: BitIndex,
        node: NodeRef,
        wiring: &WiringAssignResolver,
        normalized: &NormalizedNetlistModule<'_>,
        parsed: &ParsedNetlist,
    ) -> Result<()> {
        self.set_driver(bit_idx, BitDriver::Node(node), wiring, normalized, parsed)
    }

    fn set_input_bit(
        &mut self,
        bit_idx: BitIndex,
        port_node: NodeRef,
        bit_offset: usize,
        port_width: usize,
        wiring: &WiringAssignResolver,
        normalized: &NormalizedNetlistModule<'_>,
        parsed: &ParsedNetlist,
    ) -> Result<()> {
        self.set_driver(
            bit_idx,
            BitDriver::InputPortBit {
                port_node,
                bit_offset,
                port_width,
            },
            wiring,
            normalized,
            parsed,
        )
    }

    fn set_driver(
        &mut self,
        bit_idx: BitIndex,
        driver: BitDriver,
        wiring: &WiringAssignResolver,
        normalized: &NormalizedNetlistModule<'_>,
        parsed: &ParsedNetlist,
    ) -> Result<()> {
        if wiring.has_lhs(bit_idx) {
            return Err(anyhow!(format!(
                "multiple drivers for '{}': preserved wiring assign and explicit driver",
                normalized.render_bit(bit_idx, &parsed.nets, &parsed.interner)
            )));
        }
        let entry = self.drivers.get_mut(bit_idx).ok_or_else(|| {
            anyhow!(format!(
                "normalized bit index {} is out of range for gv2block driver table",
                bit_idx
            ))
        })?;
        if entry.replace(driver).is_some() {
            return Err(anyhow!(format!(
                "multiple drivers for '{}'",
                normalized.render_bit(bit_idx, &parsed.nets, &parsed.interner)
            )));
        }
        Ok(())
    }

    fn materialize_bit(&self, bit_idx: BitIndex, b: &mut PirFnBuilder) -> Option<NodeRef> {
        let driver = self.drivers.get(bit_idx).copied().flatten()?;
        Some(match driver {
            BitDriver::Node(node) => node,
            BitDriver::InputPortBit {
                port_node,
                bit_offset,
                port_width,
            } => {
                if port_width == 1 {
                    port_node
                } else {
                    b.add_node(
                        NodePayload::BitSlice {
                            arg: port_node,
                            start: bit_offset,
                            width: 1,
                        },
                        Type::Bits(1),
                        None,
                    )
                }
            }
        })
    }
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

fn normalized_connection_for_port<'a>(
    inst: &'a NormalizedInstance,
    parsed: &ParsedNetlist,
    port_name: &str,
) -> Option<&'a [BitSource]> {
    inst.connections.iter().find_map(|connection| {
        let Some(name) = parsed.interner.resolve(connection.port) else {
            return None;
        };
        if name == port_name {
            return Some(connection.bits.as_slice());
        }
        None
    })
}

fn source_bit_only(
    sources: &[BitSource],
    what: &str,
    wiring: &WiringAssignResolver,
    normalized: &NormalizedNetlistModule<'_>,
    parsed: &ParsedNetlist,
) -> Result<BitIndex> {
    let [source] = sources else {
        return Err(anyhow!(format!(
            "{} must connect exactly one bit in gv2block",
            what
        )));
    };
    match wiring.resolve_source(*source, normalized, parsed)? {
        ResolvedWiringSource::Bit(bit_idx) => Ok(bit_idx),
        ResolvedWiringSource::Literal(_) => Err(anyhow!(format!(
            "{} must be driven by a net bit in gv2block",
            what
        ))),
    }
}

fn target_bit_only(
    sources: &[BitSource],
    what: &str,
    normalized: &NormalizedNetlistModule<'_>,
    parsed: &ParsedNetlist,
) -> Result<BitIndex> {
    let [BitSource::Bit(bit_idx)] = sources else {
        return Err(anyhow!(format!(
            "{} must connect exactly one net bit in gv2block; got {}",
            what,
            normalized.render_sources(sources, &parsed.nets, &parsed.interner)
        )));
    };
    Ok(*bit_idx)
}

fn resolve_clock_bit(
    bit_idx: BitIndex,
    wiring: &WiringAssignResolver,
    clock_gate_bit_aliases: &HashMap<BitIndex, BitIndex>,
    normalized: &NormalizedNetlistModule<'_>,
    parsed: &ParsedNetlist,
) -> Result<BitIndex> {
    let mut seen_bits = HashSet::new();
    let mut current_bit = bit_idx;
    loop {
        if !seen_bits.insert(current_bit) {
            return Err(anyhow!(format!(
                "gv2block found a clock wiring cycle at '{}'",
                normalized.render_bit(current_bit, &parsed.nets, &parsed.interner)
            )));
        }
        current_bit =
            match wiring.resolve_source(BitSource::Bit(current_bit), normalized, parsed)? {
                ResolvedWiringSource::Bit(bit_idx) => bit_idx,
                ResolvedWiringSource::Literal(_) => {
                    return Err(anyhow!("gv2block clock pin is driven by a literal"));
                }
            };
        let Some(next_bit) = clock_gate_bit_aliases.get(&current_bit).copied() else {
            return Ok(current_bit);
        };
        current_bit = next_bit;
    }
}

fn render_clock_bit(
    bit_idx: BitIndex,
    normalized: &NormalizedNetlistModule<'_>,
    parsed: &ParsedNetlist,
) -> String {
    normalized.render_bit(bit_idx, &parsed.nets, &parsed.interner)
}

fn collect_clock_gate_passthroughs(
    normalized: &NormalizedNetlistModule<'_>,
    parsed: &ParsedNetlist,
    lib_indexed: &IndexedLibrary,
    wiring: &WiringAssignResolver,
) -> Result<ClockGatePassthroughAnalysis> {
    let mut passthroughs = Vec::new();
    let mut elided_instance_names = HashSet::new();
    let mut bit_aliases = HashMap::new();

    for inst in &normalized.instances {
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

        let clock_sources = normalized_connection_for_port(inst, parsed, &spec.clock_pin)
            .ok_or_else(|| {
                anyhow!(format!(
                    "clock-gate instance '{}' missing connection for clock pin '{}'",
                    inst_name, spec.clock_pin
                ))
            })?;
        let output_sources = normalized_connection_for_port(inst, parsed, &spec.output_pin)
            .ok_or_else(|| {
                anyhow!(format!(
                    "clock-gate instance '{}' missing connection for output pin '{}'",
                    inst_name, spec.output_pin
                ))
            })?;

        let [clock_source] = clock_sources else {
            return Err(anyhow!(format!(
                "clock-gate instance '{}' clock pin '{}' must connect exactly one bit",
                inst_name, spec.clock_pin
            )));
        };
        let output_bit = target_bit_only(
            output_sources,
            &format!(
                "clock-gate instance '{}' output pin '{}'",
                inst_name, spec.output_pin
            ),
            normalized,
            parsed,
        )?;
        let clock_bit = source_bit_only(
            clock_sources,
            &format!(
                "clock-gate instance '{}' clock pin '{}'",
                inst_name, spec.clock_pin
            ),
            wiring,
            normalized,
            parsed,
        )?;

        if let Some(existing) = bit_aliases.insert(output_bit, clock_bit) {
            if existing != clock_bit {
                return Err(anyhow!(format!(
                    "clock-gate output bit '{}' is aliased to multiple clock bits",
                    render_clock_bit(output_bit, normalized, parsed)
                )));
            }
        }

        elided_instance_names.insert(inst_name.clone());
        passthroughs.push(ClockGatePassthroughInstance {
            instance_name: inst_name,
            clock_source: *clock_source,
            output_bit,
        });
    }

    Ok(ClockGatePassthroughAnalysis {
        passthroughs,
        elided_instance_names,
        bit_aliases,
    })
}

fn apply_clock_gate_passthroughs(
    passthroughs: &[ClockGatePassthroughInstance],
    wiring: &WiringAssignResolver,
    normalized: &NormalizedNetlistModule<'_>,
    parsed: &ParsedNetlist,
    bit_drivers: &mut BitDrivers,
    b: &mut PirFnBuilder,
) -> Result<HashSet<BitIndex>> {
    let mut pending_passthroughs = passthroughs.to_vec();
    let mut unresolved_passthrough_output_bits = HashSet::new();

    while !pending_passthroughs.is_empty() {
        let mut progressed = false;
        let mut next_pending = Vec::new();
        for passthrough in pending_passthroughs {
            let ResolvedWiringSource::Bit(clock_bit) =
                wiring.resolve_source(passthrough.clock_source, normalized, parsed)?
            else {
                return Err(anyhow!(format!(
                    "clock-gate instance '{}' clock pin is driven by a literal",
                    passthrough.instance_name
                )));
            };
            let Some(source_node) = bit_drivers.materialize_bit(clock_bit, b) else {
                next_pending.push(passthrough);
                continue;
            };
            bit_drivers.set_node_bit(
                passthrough.output_bit,
                source_node,
                wiring,
                normalized,
                parsed,
            )?;
            progressed = true;
        }
        if !progressed {
            for passthrough in &next_pending {
                unresolved_passthrough_output_bits.insert(passthrough.output_bit);
            }
            break;
        }
        pending_passthroughs = next_pending;
    }

    Ok(unresolved_passthrough_output_bits)
}

fn build_package_from_normalized_netlist(
    module: &NetlistModule,
    parsed: &ParsedNetlist,
    lib_indexed: &IndexedLibrary,
    normalized: &NormalizedNetlistModule<'_>,
) -> Result<Package> {
    let wiring = WiringAssignResolver::new(normalized, parsed)?;
    let module_name =
        sanitize_to_xls_identifier(parsed.interner.resolve(module.name).unwrap_or("top"));

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

    let (top_fn, top_meta) = build_top_block(module, parsed, lib_indexed, normalized, &wiring)?;
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

fn add_input_bit_driver(
    bit_idx: BitIndex,
    bit_offset: usize,
    port_width: usize,
    port_node: NodeRef,
    bit_drivers: &mut BitDrivers,
    wiring: &WiringAssignResolver,
    normalized: &NormalizedNetlistModule<'_>,
    parsed: &ParsedNetlist,
) -> Result<()> {
    bit_drivers.set_input_bit(
        bit_idx, port_node, bit_offset, port_width, wiring, normalized, parsed,
    )
}

fn add_literal_bit(b: &mut PirFnBuilder, value: bool) -> NodeRef {
    b.add_literal_bits(1, if value { 1 } else { 0 })
}

fn resolved_wiring_source_to_node(
    source: ResolvedWiringSource,
    bit_drivers: &BitDrivers,
    unresolved_passthrough_output_bits: &HashSet<BitIndex>,
    normalized: &NormalizedNetlistModule<'_>,
    parsed: &ParsedNetlist,
    b: &mut PirFnBuilder,
    allow_undriven_zero: bool,
) -> Result<NodeRef> {
    match source {
        ResolvedWiringSource::Bit(bit_idx) => {
            if let Some(node) = bit_drivers.materialize_bit(bit_idx, b) {
                return Ok(node);
            }
            if unresolved_passthrough_output_bits.contains(&bit_idx) {
                return Err(anyhow!(format!(
                    "clock-gate output bit '{}' is unresolved as data (clock-only passthrough)",
                    normalized.render_bit(bit_idx, &parsed.nets, &parsed.interner)
                )));
            }
            if allow_undriven_zero {
                return Ok(add_literal_bit(b, false));
            }
            Err(anyhow!(format!(
                "net bit '{}' has no driver",
                normalized.render_bit(bit_idx, &parsed.nets, &parsed.interner)
            )))
        }
        ResolvedWiringSource::Literal(value) => Ok(add_literal_bit(b, value)),
    }
}

fn materialize_sources_to_node(
    sources: &[BitSource],
    empty_width: usize,
    wiring: &WiringAssignResolver,
    bit_drivers: &BitDrivers,
    unresolved_passthrough_output_bits: &HashSet<BitIndex>,
    normalized: &NormalizedNetlistModule<'_>,
    parsed: &ParsedNetlist,
    b: &mut PirFnBuilder,
    allow_undriven_zero: bool,
) -> Result<NodeRef> {
    if sources.is_empty() {
        return Ok(b.add_literal_bits(empty_width, 0));
    }

    let mut parts = Vec::with_capacity(sources.len());
    for source in sources.iter().rev().copied() {
        let resolved = wiring.resolve_source(source, normalized, parsed)?;
        parts.push(resolved_wiring_source_to_node(
            resolved,
            bit_drivers,
            unresolved_passthrough_output_bits,
            normalized,
            parsed,
            b,
            allow_undriven_zero,
        )?);
    }
    if parts.len() == 1 {
        Ok(parts[0])
    } else {
        Ok(b.add_node(
            NodePayload::Nary(NaryOp::Concat, parts),
            Type::Bits(sources.len()),
            None,
        ))
    }
}

fn materialize_bits_to_node(
    bits: &[BitIndex],
    wiring: &WiringAssignResolver,
    bit_drivers: &BitDrivers,
    unresolved_passthrough_output_bits: &HashSet<BitIndex>,
    normalized: &NormalizedNetlistModule<'_>,
    parsed: &ParsedNetlist,
    b: &mut PirFnBuilder,
    allow_undriven_zero: bool,
) -> Result<NodeRef> {
    let sources: Vec<BitSource> = bits.iter().copied().map(BitSource::Bit).collect();
    materialize_sources_to_node(
        sources.as_slice(),
        bits.len(),
        wiring,
        bit_drivers,
        unresolved_passthrough_output_bits,
        normalized,
        parsed,
        b,
        allow_undriven_zero,
    )
}

fn source_resolves_to_clock(
    source: BitSource,
    selected_clock_bit: BitIndex,
    wiring: &WiringAssignResolver,
    clock_gate_bit_aliases: &HashMap<BitIndex, BitIndex>,
    normalized: &NormalizedNetlistModule<'_>,
    parsed: &ParsedNetlist,
) -> Result<bool> {
    let ResolvedWiringSource::Bit(bit_idx) = wiring.resolve_source(source, normalized, parsed)?
    else {
        return Ok(false);
    };
    Ok(
        resolve_clock_bit(bit_idx, wiring, clock_gate_bit_aliases, normalized, parsed)?
            == selected_clock_bit,
    )
}

fn top_input_port_name_for_bit(
    bit_idx: BitIndex,
    normalized: &NormalizedNetlistModule<'_>,
    parsed: &ParsedNetlist,
) -> Option<String> {
    normalized
        .ports
        .iter()
        .find(|port| {
            port.direction == crate::netlist::parse::PortDirection::Input
                && port.bits.contains(&bit_idx)
        })
        .and_then(|port| parsed.interner.resolve(port.name))
        .map(|name| name.to_string())
}

fn build_top_block(
    module: &NetlistModule,
    parsed: &ParsedNetlist,
    lib_indexed: &IndexedLibrary,
    normalized: &NormalizedNetlistModule<'_>,
    wiring: &WiringAssignResolver,
) -> Result<(PirFn, BlockMetadata)> {
    let module_name_raw = parsed.interner.resolve(module.name).unwrap_or("top");
    let module_name = sanitize_to_xls_identifier(module_name_raw);
    let mut b = PirFnBuilder::new(&module_name);

    let mut bit_drivers = BitDrivers::new(normalized.bit_count());
    let mut top_port_name_legalizer = IdentifierLegalizer::default();
    let mut instance_name_legalizer = IdentifierLegalizer::default();

    let ClockGatePassthroughAnalysis {
        passthroughs: clock_gate_passthroughs,
        elided_instance_names: elided_clock_gate_instances,
        bit_aliases: clock_gate_bit_aliases,
    } = collect_clock_gate_passthroughs(normalized, parsed, lib_indexed, wiring)?;

    let mut selected_clock_bit: Option<BitIndex> = None;
    let mut selected_clock_port_name_raw: Option<String> = None;
    let mut dff_clock_pins = Vec::new();

    for inst in &normalized.instances {
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

    for inst in &normalized.instances {
        let cell_name = parsed.interner.resolve(inst.type_name).unwrap_or("");
        let cell = lib_indexed
            .get_cell(cell_name)
            .ok_or_else(|| anyhow!(format!("cell '{}' not found in Liberty", cell_name)))?;
        if cell
            .sequential
            .first()
            .is_some_and(|s| s.kind == SequentialKind::Ff as i32)
        {
            for connection in &inst.connections {
                let port = parsed.interner.resolve(connection.port).unwrap_or("");
                if !dff_clock_pins.iter().any(|clock_pin| clock_pin == port) {
                    continue;
                }
                let raw_clock_bit = source_bit_only(
                    connection.bits.as_slice(),
                    &format!(
                        "clock pin '{}' on instance '{}'",
                        port,
                        parsed
                            .interner
                            .resolve(inst.instance_name)
                            .unwrap_or("<unknown>")
                    ),
                    wiring,
                    normalized,
                    parsed,
                )?;
                let clock_bit = resolve_clock_bit(
                    raw_clock_bit,
                    wiring,
                    &clock_gate_bit_aliases,
                    normalized,
                    parsed,
                )?;
                match selected_clock_bit {
                    None => selected_clock_bit = Some(clock_bit),
                    Some(existing) if existing != clock_bit => {
                        return Err(anyhow!(format!(
                            "multiple clock nets detected: '{}' vs '{}'",
                            render_clock_bit(existing, normalized, parsed),
                            render_clock_bit(clock_bit, normalized, parsed)
                        )));
                    }
                    _ => {}
                }
            }
        }
    }

    if let Some(clock_bit) = selected_clock_bit {
        selected_clock_port_name_raw = top_input_port_name_for_bit(clock_bit, normalized, parsed);
        if selected_clock_port_name_raw.is_none() {
            return Err(anyhow!(format!(
                "derived clock '{}' is not a top-level input port",
                render_clock_bit(clock_bit, normalized, parsed)
            )));
        }
    }

    for port in &normalized.ports {
        if port.direction != crate::netlist::parse::PortDirection::Input {
            continue;
        }
        let name_raw = parsed.interner.resolve(port.name).unwrap_or("").to_string();
        if selected_clock_port_name_raw
            .as_ref()
            .is_some_and(|clock_name| clock_name == &name_raw)
        {
            continue;
        }
        let name = top_port_name_legalizer.legalize(&name_raw);
        let width = port.bits.len();
        let port_node = b.add_param(&name, Type::Bits(width));
        for (bit_offset, bit_idx) in port.bits.iter().copied().enumerate() {
            add_input_bit_driver(
                bit_idx,
                bit_offset,
                width,
                port_node,
                &mut bit_drivers,
                wiring,
                normalized,
                parsed,
            )?;
        }
    }

    let mut instantiations = Vec::new();
    let mut instance_outputs: Vec<(String, String, Vec<(String, NodeRef)>)> = Vec::new();

    for inst in &normalized.instances {
        let inst_name_raw = parsed
            .interner
            .resolve(inst.instance_name)
            .unwrap_or("")
            .to_string();
        if elided_clock_gate_instances.contains(&inst_name_raw) {
            continue;
        }
        let inst_name = instance_name_legalizer.legalize(&inst_name_raw);
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
        let mut output_refs = Vec::new();
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
        instance_outputs.push((inst_name_raw, inst_name, output_refs));
    }

    for (inst_name_raw, _inst_name_legal, output_refs) in &instance_outputs {
        let inst = normalized
            .instances
            .iter()
            .find(|inst| parsed.interner.resolve(inst.instance_name).unwrap_or("") == inst_name_raw)
            .ok_or_else(|| anyhow!(format!("instance '{}' not found", inst_name_raw)))?;
        for connection in &inst.connections {
            let port_name = parsed
                .interner
                .resolve(connection.port)
                .unwrap_or("")
                .to_string();
            let Some(node_ref) = output_refs
                .iter()
                .find(|(name, _)| name == &port_name)
                .map(|(_, node_ref)| *node_ref)
            else {
                continue;
            };
            if connection.bits.is_empty() {
                continue;
            }
            let output_bit = target_bit_only(
                connection.bits.as_slice(),
                &format!("instance output '{}.{}'", inst_name_raw, port_name),
                normalized,
                parsed,
            )?;
            bit_drivers.set_node_bit(output_bit, node_ref, wiring, normalized, parsed)?;
        }
    }

    let unresolved_passthrough_output_bits = apply_clock_gate_passthroughs(
        &clock_gate_passthroughs,
        wiring,
        normalized,
        parsed,
        &mut bit_drivers,
        &mut b,
    )?;

    for inst in &normalized.instances {
        let inst_name_raw = parsed
            .interner
            .resolve(inst.instance_name)
            .unwrap_or("")
            .to_string();
        if elided_clock_gate_instances.contains(&inst_name_raw) {
            continue;
        }
        let inst_name = instance_name_legalizer.legalize(&inst_name_raw);
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
        let input_pin_names: HashSet<String> = inputs.iter().map(|pin| pin.name.clone()).collect();

        let mut clock_pin_names: HashSet<String> = inputs
            .iter()
            .filter(|pin| pin.is_clocking_pin)
            .map(|pin| pin.name.clone())
            .collect();
        if let Some(seq) = cell.sequential.first() {
            if !seq.clock_expr.is_empty() {
                let (clk_pin, _neg) = parse_simple_clock_expr(&seq.clock_expr)?;
                clock_pin_names.insert(clk_pin);
            }
        }

        for connection in &inst.connections {
            let port_name = parsed
                .interner
                .resolve(connection.port)
                .unwrap_or("")
                .to_string();
            if !input_pin_names.contains(&port_name) || clock_pin_names.contains(&port_name) {
                continue;
            }
            if let Some(clock_bit) = selected_clock_bit {
                for source in connection.bits.iter().copied() {
                    if source_resolves_to_clock(
                        source,
                        clock_bit,
                        wiring,
                        &clock_gate_bit_aliases,
                        normalized,
                        parsed,
                    )? {
                        return Err(anyhow!(format!(
                            "clock net '{}' is connected to non-clock input '{}.{}' (cell '{}'); gv2block does not support feeding the selected clock into ordinary logic",
                            render_clock_bit(clock_bit, normalized, parsed),
                            inst_name,
                            port_name,
                            cell_name
                        )));
                    }
                }
            }
            let value = materialize_sources_to_node(
                connection.bits.as_slice(),
                1,
                wiring,
                &bit_drivers,
                &unresolved_passthrough_output_bits,
                normalized,
                parsed,
                &mut b,
                false,
            )?;
            let input_node_name = format!("{}_{}", inst_name, port_name);
            b.add_node(
                NodePayload::InstantiationInput {
                    instantiation: inst_name.clone(),
                    port_name: port_name.clone(),
                    arg: value,
                },
                Type::Tuple(vec![]),
                Some(&input_node_name),
            );
        }
    }

    let mut output_names = Vec::new();
    let mut output_nodes = Vec::new();
    for port in &normalized.ports {
        if port.direction != crate::netlist::parse::PortDirection::Output {
            continue;
        }
        let name_raw = parsed.interner.resolve(port.name).unwrap_or("").to_string();
        let name = top_port_name_legalizer.legalize(&name_raw);
        let node = materialize_bits_to_node(
            port.bits.as_slice(),
            wiring,
            &bit_drivers,
            &unresolved_passthrough_output_bits,
            normalized,
            parsed,
            &mut b,
            true,
        )?;
        output_names.push(name);
        output_nodes.push(node);
    }

    let (ret_ty, ret_node_ref) = build_return_node(&mut b, &output_nodes);

    let mut input_port_ids = HashMap::new();
    for param in &b.params {
        input_port_ids.insert(param.name.clone(), param.id.get_wrapped_id());
    }

    let clock_port_name = selected_clock_port_name_raw
        .as_ref()
        .map(|name| top_port_name_legalizer.legalize(name));

    let meta = BlockMetadata {
        clock_port_name,
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
