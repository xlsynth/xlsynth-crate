// SPDX-License-Identifier: Apache-2.0

//! Project a parsed netlist and Liberty proto into a GateFn.

use crate::aig::gate::{AigBitVector, AigOperand, GateFn};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::liberty::cell_formula::Term;
use crate::liberty_proto::Library;
use crate::netlist::parse::{Net, NetIndex, NetRef, NetlistModule};
use std::collections::HashMap;
use std::collections::HashSet;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

fn substitute_state_vars_in_term(
    term: &Term,
    sequential_terms: &HashMap<String, (Term, String)>,
) -> Term {
    match term {
        Term::Input(name) => sequential_terms
            .get(name)
            .map(|(replacement, _)| replacement.clone())
            .unwrap_or_else(|| Term::Input(name.clone())),
        Term::And(lhs, rhs) => Term::And(
            Box::new(substitute_state_vars_in_term(lhs, sequential_terms)),
            Box::new(substitute_state_vars_in_term(rhs, sequential_terms)),
        ),
        Term::Or(lhs, rhs) => Term::Or(
            Box::new(substitute_state_vars_in_term(lhs, sequential_terms)),
            Box::new(substitute_state_vars_in_term(rhs, sequential_terms)),
        ),
        Term::Xor(lhs, rhs) => Term::Xor(
            Box::new(substitute_state_vars_in_term(lhs, sequential_terms)),
            Box::new(substitute_state_vars_in_term(rhs, sequential_terms)),
        ),
        Term::Negate(inner) => Term::Negate(Box::new(substitute_state_vars_in_term(
            inner,
            sequential_terms,
        ))),
        Term::Constant(value) => Term::Constant(*value),
    }
}

fn build_cell_formula_map(
    liberty_lib: &Library,
    collapse_sequential: bool,
    used_cells: &HashSet<String>,
    dff_cells_identity: &HashSet<String>,
    dff_cells_inverted: &HashSet<String>,
) -> Result<HashMap<(String, String), (Term, String)>, String> {
    let mut cell_formula_map = HashMap::new();
    let override_cells: HashSet<String> = dff_cells_identity
        .iter()
        .cloned()
        .chain(dff_cells_inverted.iter().cloned())
        .collect();
    for cell in &liberty_lib.cells {
        if !used_cells.contains(&cell.name) {
            continue;
        }
        let collapse_for_cell = collapse_sequential && !override_cells.contains(&cell.name);
        let pin_names: HashSet<String> = cell.pins.iter().map(|pin| pin.name.clone()).collect();
        let state_vars: HashSet<String> = cell
            .sequential
            .iter()
            .map(|seq| seq.state_var.clone())
            .collect();
        let mut sequential_terms: HashMap<String, (crate::liberty::cell_formula::Term, String)> =
            HashMap::new();
        if collapse_for_cell {
            for seq in &cell.sequential {
                if seq.kind == crate::liberty_proto::SequentialKind::Latch as i32
                    && !seq.clock_expr.is_empty()
                {
                    return Err(format!(
                        "collapse_sequential does not support latches with enable; cell '{}' state '{}' enable '{}'",
                        cell.name, seq.state_var, seq.clock_expr
                    ));
                }
                if seq.kind == crate::liberty_proto::SequentialKind::Ff as i32
                    && (!seq.clear_expr.is_empty() || !seq.preset_expr.is_empty())
                {
                    return Err(format!(
                        "collapse_sequential does not support flops with async clear/preset; cell '{}' state '{}' clear '{}' preset '{}'",
                        cell.name, seq.state_var, seq.clear_expr, seq.preset_expr
                    ));
                }
                if seq.state_var.is_empty() || seq.next_state.is_empty() {
                    continue;
                }
                let term = crate::liberty::cell_formula::parse_formula(&seq.next_state).map_err(
                    |e| {
                        format!(
                            "Failed to parse next_state for cell '{}' state '{}' (next_state: \"{}\"): {}",
                            cell.name, seq.state_var, seq.next_state, e
                        )
                    },
                )?;
                let inputs = term.inputs();
                let mut state_refs: Vec<String> = inputs
                    .iter()
                    .filter(|name| state_vars.contains((*name).as_str()))
                    .cloned()
                    .collect();
                state_refs.sort();
                state_refs.dedup();
                if !state_refs.is_empty() {
                    return Err(format!(
                        "collapse_sequential does not support next_state formulas that reference state variables; cell '{}' state '{}' references [{}] in next_state \"{}\"",
                        cell.name,
                        seq.state_var,
                        state_refs.join(", "),
                        seq.next_state
                    ));
                }
                let mut non_pin_inputs: Vec<String> = inputs
                    .iter()
                    .filter(|name| !pin_names.contains((*name).as_str()))
                    .cloned()
                    .collect();
                non_pin_inputs.sort();
                non_pin_inputs.dedup();
                if !non_pin_inputs.is_empty() {
                    return Err(format!(
                        "collapse_sequential requires next_state inputs to be cell pins; cell '{}' state '{}' references non-pin inputs [{}] in next_state \"{}\"",
                        cell.name,
                        seq.state_var,
                        non_pin_inputs.join(", "),
                        seq.next_state
                    ));
                }
                let mut term = term;
                let mut next_state_string = seq.next_state.clone();
                if let Some(base_var) = seq.state_var.strip_suffix('N') {
                    if state_vars.contains(base_var) {
                        // Common Liberty convention: IQN is the complement of IQ.
                        term = crate::liberty::cell_formula::Term::Negate(Box::new(term));
                        next_state_string = format!("!({})", seq.next_state);
                    }
                }
                sequential_terms.insert(seq.state_var.clone(), (term, next_state_string));
            }
        }
        for pin in &cell.pins {
            if pin.direction == 1 && !pin.function.is_empty() {
                let original_formula_string = pin.function.clone();
                match crate::liberty::cell_formula::parse_formula(&pin.function) {
                    Ok(term) => {
                        let term = if collapse_for_cell {
                            let replaced = substitute_state_vars_in_term(&term, &sequential_terms);
                            let mut remaining_state_refs: Vec<String> = replaced
                                .inputs()
                                .into_iter()
                                .filter(|name| state_vars.contains(name))
                                .collect();
                            remaining_state_refs.sort();
                            remaining_state_refs.dedup();
                            if !remaining_state_refs.is_empty() {
                                return Err(format!(
                                    "collapse_sequential could not safely collapse state variables [{}] for cell '{}' output pin '{}' (function \"{}\")",
                                    remaining_state_refs.join(", "),
                                    cell.name,
                                    pin.name,
                                    pin.function
                                ));
                            }
                            replaced
                        } else {
                            term
                        };
                        cell_formula_map.insert(
                            (cell.name.clone(), pin.name.clone()),
                            (term, original_formula_string),
                        );
                    }
                    Err(e) => {
                        log::warn!(
                            "Failed to parse formula for cell '{}', pin '{}' (formula: \"{}\"): {}",
                            cell.name,
                            pin.name,
                            original_formula_string,
                            e
                        );
                    }
                }
            }
        }
    }
    Ok(cell_formula_map)
}

fn check_undriven_nets(
    used_as_input: &HashSet<SymbolU32>,
    driven: &HashSet<SymbolU32>,
    interner: &StringInterner<StringBackend<SymbolU32>>,
) {
    for net in used_as_input {
        if !driven.contains(net) {
            let net_name = interner.resolve(*net).unwrap_or("<unknown>");
            panic!(
                "Net '{}' is used as an input but is never driven by any instance or module input!",
                net_name
            );
        }
    }
}

fn net_name(
    idx: NetIndex,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> String {
    interner
        .resolve(nets[idx.0].name)
        .unwrap_or("<unknown>")
        .to_string()
}

fn select_width_bits(msb: u32, lsb: u32) -> usize {
    (u32::abs_diff(msb, lsb) as usize) + 1
}

fn select_bit_number(msb: u32, lsb: u32, bit_offset: usize) -> Option<u32> {
    if bit_offset >= select_width_bits(msb, lsb) {
        return None;
    }
    let offset = bit_offset as u32;
    if msb >= lsb {
        Some(lsb + offset)
    } else {
        Some(lsb - offset)
    }
}

struct ResolvedNetValues {
    values: HashMap<NetIndex, Vec<Option<AigOperand>>>,
}

impl ResolvedNetValues {
    fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    fn seed_input(&mut self, idx: NetIndex, bv: &AigBitVector) {
        self.values.insert(
            idx,
            bv.iter_lsb_to_msb()
                .map(|bit| Some(*bit))
                .collect::<Vec<_>>(),
        );
    }

    fn ensure_entry(&mut self, idx: NetIndex, width: usize) -> &mut Vec<Option<AigOperand>> {
        self.values.entry(idx).or_insert_with(|| vec![None; width])
    }

    fn resolve_bit(
        &self,
        idx: NetIndex,
        bit_number: u32,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> Result<Option<AigOperand>, String> {
        let Some(offset) = nets[idx.0].bit_offset(bit_number) else {
            return Err(format!(
                "bit {} out of range for net '{}'",
                bit_number,
                net_name(idx, nets, interner)
            ));
        };
        Ok(self
            .values
            .get(&idx)
            .and_then(|bits| bits.get(offset))
            .copied()
            .flatten())
    }

    fn materialize_ref(
        &self,
        net_ref: &NetRef,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        gb: &mut GateBuilder,
    ) -> Result<Option<AigBitVector>, String> {
        match net_ref {
            NetRef::Simple(idx) => {
                let net = &nets[idx.0];
                let mut bits = Vec::with_capacity(net.width_bits());
                for offset in 0..net.width_bits() {
                    let bit_number = net.bit_number(offset).ok_or_else(|| {
                        format!(
                            "internal error computing bit {} for net '{}'",
                            offset,
                            net_name(*idx, nets, interner)
                        )
                    })?;
                    let Some(bit) = self.resolve_bit(*idx, bit_number, nets, interner)? else {
                        return Ok(None);
                    };
                    bits.push(bit);
                }
                Ok(Some(AigBitVector::from_lsb_is_index_0(&bits)))
            }
            NetRef::BitSelect(idx, bit) => {
                let Some(bit) = self.resolve_bit(*idx, *bit, nets, interner)? else {
                    return Ok(None);
                };
                Ok(Some(AigBitVector::from_bit(bit)))
            }
            NetRef::PartSelect(idx, msb, lsb) => {
                let width = select_width_bits(*msb, *lsb);
                let mut bits = Vec::with_capacity(width);
                for offset in 0..width {
                    let bit_number = select_bit_number(*msb, *lsb, offset).ok_or_else(|| {
                        format!(
                            "invalid part-select [{}:{}] on net '{}'",
                            msb,
                            lsb,
                            net_name(*idx, nets, interner)
                        )
                    })?;
                    let Some(bit) = self.resolve_bit(*idx, bit_number, nets, interner)? else {
                        return Ok(None);
                    };
                    bits.push(bit);
                }
                Ok(Some(AigBitVector::from_lsb_is_index_0(&bits)))
            }
            NetRef::Literal(bits) => {
                let mut ops = Vec::with_capacity(bits.get_bit_count());
                for i in 0..bits.get_bit_count() {
                    let bit_is_one = bits.get_bit(i).unwrap_or(false);
                    ops.push(if bit_is_one {
                        gb.get_true()
                    } else {
                        gb.get_false()
                    });
                }
                Ok(Some(AigBitVector::from_lsb_is_index_0(&ops)))
            }
            NetRef::Unconnected | NetRef::Concat(_) => Ok(None),
        }
    }

    fn write_bit(
        &mut self,
        idx: NetIndex,
        bit_number: u32,
        value: AigOperand,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> Result<(), String> {
        let net = &nets[idx.0];
        let width = net.width_bits();
        let Some(offset) = net.bit_offset(bit_number) else {
            return Err(format!(
                "bit {} out of range for net '{}'",
                bit_number,
                net_name(idx, nets, interner)
            ));
        };
        let entry = self.ensure_entry(idx, width);
        if entry[offset].is_some() {
            return Err(format!(
                "net '{}' bit {} was assigned more than once during projection",
                net_name(idx, nets, interner),
                bit_number
            ));
        }
        entry[offset] = Some(value);
        Ok(())
    }

    fn write_ref(
        &mut self,
        dst_ref: &NetRef,
        src_bv: &AigBitVector,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> Result<(), String> {
        match dst_ref {
            NetRef::Simple(idx) => {
                let net = &nets[idx.0];
                let width = net.width_bits();
                if src_bv.get_bit_count() != width {
                    return Err(format!(
                        "width mismatch assigning to net '{}': expected {} bits but got {}",
                        net_name(*idx, nets, interner),
                        width,
                        src_bv.get_bit_count()
                    ));
                }
                for offset in 0..width {
                    let bit_number = net.bit_number(offset).ok_or_else(|| {
                        format!(
                            "internal error computing bit {} for net '{}'",
                            offset,
                            net_name(*idx, nets, interner)
                        )
                    })?;
                    self.write_bit(*idx, bit_number, *src_bv.get_lsb(offset), nets, interner)?;
                }
                Ok(())
            }
            NetRef::BitSelect(idx, bit) => {
                if src_bv.get_bit_count() != 1 {
                    return Err(format!(
                        "width mismatch assigning to net '{}[{}]': expected 1 bit but got {}",
                        net_name(*idx, nets, interner),
                        bit,
                        src_bv.get_bit_count()
                    ));
                }
                self.write_bit(*idx, *bit, *src_bv.get_lsb(0), nets, interner)
            }
            NetRef::PartSelect(idx, msb, lsb) => {
                let width = select_width_bits(*msb, *lsb);
                if src_bv.get_bit_count() != width {
                    return Err(format!(
                        "width mismatch assigning to net '{}[{}:{}]': expected {} bits but got {}",
                        net_name(*idx, nets, interner),
                        msb,
                        lsb,
                        width,
                        src_bv.get_bit_count()
                    ));
                }
                for offset in 0..width {
                    let bit_number = select_bit_number(*msb, *lsb, offset).ok_or_else(|| {
                        format!(
                            "invalid part-select [{}:{}] on net '{}'",
                            msb,
                            lsb,
                            net_name(*idx, nets, interner)
                        )
                    })?;
                    self.write_bit(*idx, bit_number, *src_bv.get_lsb(offset), nets, interner)?;
                }
                Ok(())
            }
            NetRef::Literal(_) => Err("output destination cannot be a literal".to_string()),
            NetRef::Unconnected => Ok(()),
            NetRef::Concat(_) => Err("concat output destination is not supported".to_string()),
        }
    }

    fn materialize_output_or_false(
        &self,
        idx: NetIndex,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        gb: &mut GateBuilder,
    ) -> Result<AigBitVector, String> {
        let net = &nets[idx.0];
        let mut bits = Vec::with_capacity(net.width_bits());
        for offset in 0..net.width_bits() {
            let bit_number = net.bit_number(offset).ok_or_else(|| {
                format!(
                    "internal error computing bit {} for net '{}'",
                    offset,
                    net_name(idx, nets, interner)
                )
            })?;
            bits.push(
                self.resolve_bit(idx, bit_number, nets, interner)?
                    .unwrap_or_else(|| gb.get_false()),
            );
        }
        Ok(AigBitVector::from_lsb_is_index_0(&bits))
    }
}

fn process_instance_outputs(
    inst: &crate::netlist::parse::NetlistInstance,
    type_name: &str,
    inst_name: &str,
    pin_directions: &HashMap<&str, i32>,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    nets: &[Net],
    gb: &mut GateBuilder,
    resolved: &mut ResolvedNetValues,
    dff_cells_identity: &std::collections::HashSet<String>,
    dff_cells_inverted: &std::collections::HashSet<String>,
    cell_formula_map: &HashMap<(String, String), (crate::liberty::cell_formula::Term, String)>,
    input_map: &HashMap<String, AigOperand>,
    port_map: &HashMap<String, String>,
) -> Result<bool, String> {
    let mut processed_any_output = false;
    for (port, netref) in &inst.connections {
        let port_name = interner.resolve(*port).unwrap();
        let pin_dir = *pin_directions.get(port_name).unwrap_or(&0); // 1=OUTPUT, 2=INPUT
        if pin_dir == 1 {
            if handle_dff_identity_override(
                type_name,
                inst_name,
                port_name,
                inst,
                interner,
                gb,
                resolved,
                dff_cells_identity,
                dff_cells_inverted,
                nets,
            )? {
                processed_any_output = true;
                continue;
            }
            let key = (type_name.to_string(), port_name.to_string());
            let (formula_ast, original_formula_str) = match cell_formula_map.get(&key) {
                Some(x) => x,
                None => continue,
            };
            let context = crate::liberty::cell_formula::EmitContext {
                cell_name: type_name,
                original_formula: original_formula_str.as_str(),
                instance_name: Some(inst_name),
                port_map: Some(port_map),
            };
            let out_op = formula_ast
                .emit_formula_term(gb, input_map, &context)
                .map_err(|e| {
                    format!(
                        "Failed to emit formula for cell '{}' instance '{}' pin '{}': {}",
                        type_name, inst_name, port_name, e
                    )
                })?;
            match netref {
                NetRef::Simple(_) | NetRef::BitSelect(_, _) | NetRef::PartSelect(_, _, _) => {
                    let src_bv = AigBitVector::from_bit(out_op);
                    resolved.write_ref(netref, &src_bv, nets, interner)?;
                }
                _ => {
                    // Only net destinations are supported for output pins.
                }
            }
            processed_any_output = true;
        }
    }
    Ok(processed_any_output)
}

pub fn project_gatefn_from_netlist_and_liberty(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    liberty_lib: &Library,
    dff_cells_identity: &std::collections::HashSet<String>,
    dff_cells_inverted: &std::collections::HashSet<String>,
) -> Result<GateFn, String> {
    let options = GateFnProjectOptions {
        collapse_sequential: false,
    };
    project_gatefn_from_netlist_and_liberty_with_options(
        module,
        nets,
        interner,
        liberty_lib,
        dff_cells_identity,
        dff_cells_inverted,
        &options,
    )
}

#[derive(Debug, Clone)]
pub struct GateFnProjectOptions {
    /// If true, collapse sequential state variables by substituting next_state.
    pub collapse_sequential: bool,
}

pub fn project_gatefn_from_netlist_and_liberty_with_options(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    liberty_lib: &Library,
    dff_cells_identity: &std::collections::HashSet<String>,
    dff_cells_inverted: &std::collections::HashSet<String>,
    options: &GateFnProjectOptions,
) -> Result<GateFn, String> {
    if !module.assigns.is_empty() {
        return Err(
            "Liberty-backed netlist projection does not support preserved continuous assigns"
                .to_string(),
        );
    }
    let used_cell_names: HashSet<String> = module
        .instances
        .iter()
        .filter_map(|inst| interner.resolve(inst.type_name).map(str::to_string))
        .collect();
    let cell_formula_map = build_cell_formula_map(
        liberty_lib,
        options.collapse_sequential,
        &used_cell_names,
        dff_cells_identity,
        dff_cells_inverted,
    )?;
    let module_name = interner.resolve(module.name).unwrap();
    let mut gb = GateBuilder::new(module_name.to_string(), GateBuilderOptions::no_opt());
    let mut resolved = ResolvedNetValues::new();
    collect_module_io_nets(module, nets, interner, &mut gb, &mut resolved);
    let mut used_as_input = HashSet::new();
    let mut driven = HashSet::new();
    for port in &module.ports {
        if port.direction == crate::netlist::parse::PortDirection::Input {
            driven.insert(port.name);
        }
    }
    for inst in &module.instances {
        let type_name = interner.resolve(inst.type_name).unwrap();
        let cell = liberty_lib
            .cells
            .iter()
            .find(|c| c.name == type_name)
            .ok_or_else(|| format!("Cell '{}' not found in liberty data", type_name))?;
        let mut pin_directions = std::collections::HashMap::new();
        for pin in &cell.pins {
            pin_directions.insert(pin.name.as_str(), pin.direction);
        }
        for (port, netref) in &inst.connections {
            let port_name = interner.resolve(*port).unwrap();
            let pin_dir = *pin_directions.get(port_name).unwrap_or(&0); // 1=OUTPUT, 2=INPUT
            match netref {
                crate::netlist::parse::NetRef::Simple(net_idx)
                | crate::netlist::parse::NetRef::BitSelect(net_idx, _) => {
                    if pin_dir == 1 {
                        driven.insert(nets[net_idx.0].name);
                    } else if pin_dir == 2 {
                        used_as_input.insert(nets[net_idx.0].name);
                    }
                }
                crate::netlist::parse::NetRef::Unconnected => {
                    // Do not count as driven/used.
                }
                _ => {}
            }
        }
    }
    check_undriven_nets(&used_as_input, &driven, interner);
    let mut unprocessed: Vec<_> = module.instances.iter().collect();
    let mut processed_any = true;
    while !unprocessed.is_empty() && processed_any {
        processed_any = false;
        let mut i = 0;
        while i < unprocessed.len() {
            let inst = unprocessed[i];
            let type_name = interner.resolve(inst.type_name).unwrap();
            let inst_name = interner.resolve(inst.instance_name).unwrap();
            let cell = liberty_lib
                .cells
                .iter()
                .find(|c| c.name == type_name)
                .ok_or_else(|| format!("Cell '{}' not found in Liberty", type_name))?;
            let mut pin_directions = HashMap::new();
            for pin in &cell.pins {
                pin_directions.insert(pin.name.as_str(), pin.direction);
            }
            let (input_map, missing_inputs, port_map) =
                build_instance_input_map(inst, &pin_directions, interner, nets, &resolved, &mut gb);
            if !missing_inputs.is_empty() {
                log::trace!(
                    "Skipping instance '{}' (cell '{}') due to missing input nets: {:?}",
                    inst_name,
                    type_name,
                    missing_inputs
                );
                i += 1;
                continue;
            }
            let processed = process_instance_outputs(
                inst,
                type_name,
                inst_name,
                &pin_directions,
                interner,
                nets,
                &mut gb,
                &mut resolved,
                dff_cells_identity,
                dff_cells_inverted,
                &cell_formula_map,
                &input_map,
                &port_map,
            )?;
            if processed {
                unprocessed.remove(i);
                processed_any = true;
            } else {
                i += 1;
            }
        }
        if !processed_any && !unprocessed.is_empty() {
            // Build a short, actionable diagnostic to help pinpoint why instances
            // could not be resolved. We show up to a bounded number of examples
            // with their missing inputs.
            let mut diag_lines: Vec<String> = Vec::new();
            let example_limit = 10usize.min(unprocessed.len());
            for inst in unprocessed.iter().take(example_limit) {
                let type_name = interner.resolve(inst.type_name).unwrap_or("<unknown>");
                let inst_name = interner.resolve(inst.instance_name).unwrap_or("<unknown>");
                // Determine pin directions for this cell type (to identify inputs).
                let mut pin_directions = HashMap::new();
                if let Some(cell) = liberty_lib.cells.iter().find(|c| c.name == type_name) {
                    for pin in &cell.pins {
                        pin_directions.insert(pin.name.as_str(), pin.direction);
                    }
                }
                // Recompute missing inputs for this instance w.r.t. current resolved nets.
                let (_input_map, missing_inputs, _port_map) = build_instance_input_map(
                    inst,
                    &pin_directions,
                    interner,
                    nets,
                    &resolved,
                    &mut gb,
                );
                if missing_inputs.is_empty() {
                    diag_lines.push(format!(
                        "- cell '{}' instance '{}' at {}:{} has no missing inputs detected; unresolved due to dependency cycle or unknown output formula",
                        type_name, inst_name, inst.inst_lineno, inst.inst_colno
                    ));
                } else {
                    diag_lines.push(format!(
                        "- cell '{}' instance '{}' at {}:{} missing inputs: [{}]",
                        type_name,
                        inst_name,
                        inst.inst_lineno,
                        inst.inst_colno,
                        missing_inputs.join(", ")
                    ));
                }
            }

            let mut msg = String::new();
            msg.push_str(&format!(
                "Could not resolve all instance dependencies (possible cycle or missing driver). Remaining instances: {}\n",
                unprocessed.len()
            ));
            msg.push_str(&format!(
                "Examples of unresolved instances (showing up to {}):\n",
                example_limit
            ));
            for line in diag_lines {
                msg.push_str(&line);
                msg.push('\n');
            }
            msg.push_str(r#"Hint: re-run with RUST_LOG=trace to log skipped instances and their missing nets during processing."#);
            return Err(msg);
        }
    }
    for port in &module.ports {
        if port.direction == crate::netlist::parse::PortDirection::Output {
            let net_idx = nets
                .iter()
                .position(|n| n.name == port.name)
                .map(NetIndex)
                .ok_or("output port net not found")?;
            let net = &nets[net_idx.0];
            let net_name = interner.resolve(net.name).unwrap();
            let width = net.width_bits();
            let bv = resolved.materialize_output_or_false(net_idx, nets, interner, &mut gb)?;
            assert_eq!(
                bv.get_bit_count(),
                width,
                "Output net '{}' width mismatch",
                net_name
            );
            gb.add_output(net_name.to_string(), bv.clone());
        }
    }
    Ok(gb.build())
}

fn collect_module_io_nets(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    gb: &mut GateBuilder,
    resolved: &mut ResolvedNetValues,
) {
    for port in &module.ports {
        if port.direction == crate::netlist::parse::PortDirection::Input {
            let net_idx = nets
                .iter()
                .position(|n| n.name == port.name)
                .map(NetIndex)
                .expect("input port net not found");
            let net = &nets[net_idx.0];
            let net_name = interner.resolve(net.name).unwrap();
            let width = net.width_bits();
            let bv = gb.add_input(net_name.to_string(), width);
            resolved.seed_input(net_idx, &bv);
        }
    }
}

fn build_instance_input_map(
    inst: &crate::netlist::parse::NetlistInstance,
    pin_directions: &HashMap<&str, i32>,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    nets: &[Net],
    resolved: &ResolvedNetValues,
    gb: &mut GateBuilder,
) -> (
    HashMap<String, AigOperand>,
    Vec<String>,
    HashMap<String, String>,
) {
    let mut input_map = HashMap::new();
    let mut missing_inputs = Vec::new();
    let mut port_map = HashMap::new();
    for (port, netref) in &inst.connections {
        let port_name = interner.resolve(*port).unwrap();
        let pin_dir = *pin_directions.get(port_name).unwrap_or(&0); // 1=OUTPUT, 2=INPUT
        let net_name_str = match netref {
            crate::netlist::parse::NetRef::Simple(net_idx) => interner
                .resolve(nets[net_idx.0].name)
                .unwrap_or("<unknown>")
                .to_string(),
            crate::netlist::parse::NetRef::BitSelect(net_idx, bit) => format!(
                "{}[{}]",
                interner
                    .resolve(nets[net_idx.0].name)
                    .unwrap_or("<unknown>"),
                bit
            ),
            crate::netlist::parse::NetRef::PartSelect(net_idx, msb, lsb) => format!(
                "{}[{}:{}]",
                interner
                    .resolve(nets[net_idx.0].name)
                    .unwrap_or("<unknown>"),
                msb,
                lsb
            ),
            crate::netlist::parse::NetRef::Literal(bits) => format!("{}", bits),
            crate::netlist::parse::NetRef::Unconnected => "<unconnected>".to_string(),
            crate::netlist::parse::NetRef::Concat(_) => "<concat>".to_string(),
        };
        port_map.insert(port_name.to_string(), net_name_str);
        if pin_dir == 2 {
            match netref {
                NetRef::Simple(net_idx) => {
                    let materialized = resolved.materialize_ref(netref, nets, interner, gb);
                    match materialized {
                        Ok(Some(bv)) => {
                            if bv.get_bit_count() == 1 {
                                input_map.insert(port_name.to_string(), *bv.get_lsb(0));
                            }
                        }
                        Ok(None) => {
                            missing_inputs.push(format!(
                                "{} (NetIndex({}), name='{}')",
                                port_name,
                                net_idx.0,
                                net_name(*net_idx, nets, interner)
                            ));
                        }
                        Err(e) => missing_inputs.push(format!("{} ({})", port_name, e)),
                    }
                }
                NetRef::BitSelect(net_idx, bit) => {
                    match resolved.materialize_ref(netref, nets, interner, gb) {
                        Ok(Some(bv)) => {
                            input_map.insert(port_name.to_string(), *bv.get_lsb(0));
                        }
                        Ok(None) => {
                            missing_inputs.push(format!(
                                "{} (NetIndex({}), name='{}', bit={})",
                                port_name,
                                net_idx.0,
                                net_name(*net_idx, nets, interner),
                                bit
                            ));
                        }
                        Err(e) => missing_inputs.push(format!("{} ({})", port_name, e)),
                    }
                }
                NetRef::PartSelect(net_idx, msb, lsb) => {
                    match resolved.materialize_ref(netref, nets, interner, gb) {
                        Ok(Some(bv)) => {
                            if bv.get_bit_count() == 1 {
                                input_map.insert(port_name.to_string(), *bv.get_lsb(0));
                            }
                        }
                        Ok(None) => {
                            missing_inputs.push(format!(
                                "{} (NetIndex({}), name='{}', part-select=[{}:{}])",
                                port_name,
                                net_idx.0,
                                net_name(*net_idx, nets, interner),
                                msb,
                                lsb
                            ));
                        }
                        Err(e) => missing_inputs.push(format!("{} ({})", port_name, e)),
                    }
                }
                NetRef::Literal(bits) => {
                    let bit_count = bits.get_bit_count();
                    assert_eq!(bit_count, 1);
                    let is_one = bits.get_bit(0).unwrap();
                    let val = if is_one {
                        gb.get_true()
                    } else {
                        gb.get_false()
                    };
                    input_map.insert(port_name.to_string(), val);
                }
                NetRef::Unconnected => {
                    // Treat as a hard missing input and surface clearly.
                    missing_inputs.push(format!("{} (<unconnected>)", port_name));
                }
                NetRef::Concat(_) => {
                    // Currently unsupported for cell inputs; surface clearly.
                    missing_inputs.push(format!("{} (<concat-unsupported>)", port_name));
                }
            }
        }
    }
    (input_map, missing_inputs, port_map)
}
// Helpers for DFF identity/inverted overrides.
fn invert_bv(gb: &mut GateBuilder, src: &AigBitVector) -> AigBitVector {
    let width = src.get_bit_count();
    let mut out = AigBitVector::from_lsb_is_index_0(&vec![gb.get_false(); width]);
    for i in 0..width {
        out.set_lsb(i, gb.add_not(*src.get_lsb(i)));
    }
    out
}

fn build_d_bv(
    d_netref: &NetRef,
    gb: &mut GateBuilder,
    resolved: &ResolvedNetValues,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    invert: bool,
) -> Result<Option<AigBitVector>, String> {
    let Some(bv) = resolved.materialize_ref(d_netref, nets, interner, gb)? else {
        return Ok(None);
    };
    Ok(Some(if invert { invert_bv(gb, &bv) } else { bv }))
}

fn write_bv_to_port_destination(
    inst: &crate::netlist::parse::NetlistInstance,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    resolved: &mut ResolvedNetValues,
    nets: &[Net],
    target_port_ci: &str,
    src_bv: &AigBitVector,
) -> Result<(), String> {
    for (p, dst_ref) in &inst.connections {
        let pname = interner.resolve(*p).unwrap();
        if !pname.eq_ignore_ascii_case(target_port_ci) {
            continue;
        }
        match dst_ref {
            NetRef::Simple(_) | NetRef::BitSelect(_, _) | NetRef::PartSelect(_, _, _) => {
                resolved.write_ref(dst_ref, src_bv, nets, interner)?;
            }
            NetRef::Literal(_) => {
                log::warn!(
                    "DFF override: destination output as literal not supported for port '{}'",
                    pname
                );
            }
            NetRef::Unconnected => {
                // Nothing to write.
            }
            NetRef::Concat(_) => {
                return Err("concat destination not supported in DFF override".to_string());
            }
        }
    }
    Ok(())
}

/// Implements DFF output overrides for identity (Q=D) and inverted (QN=NOT(D)).
///
/// This bypasses formula emission by directly wiring from the connected `D`
/// netref into the `Q`/`QN` destination, supporting Simple, BitSelect, and
/// PartSelect on the destination side. Destination bitvectors are sized to the
/// full declared width of the destination net, and part-selects are written at
/// the correct least-significant-bit offset.
///
/// Returns `true` when this function fully handled this output port; `false`
/// when the instance is not a recognized DFF and normal handling should
/// proceed.
fn handle_dff_identity_override(
    type_name: &str,
    inst_name: &str,
    port_name: &str,
    inst: &crate::netlist::parse::NetlistInstance,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    gb: &mut GateBuilder,
    resolved: &mut ResolvedNetValues,
    dff_cells_identity: &std::collections::HashSet<String>,
    dff_cells_inverted: &std::collections::HashSet<String>,
    nets: &[Net],
) -> Result<bool, String> {
    if !dff_cells_identity.contains(type_name) && !dff_cells_inverted.contains(type_name) {
        return Ok(false);
    }
    // Decide identity vs inverted based on the current output port name and
    // membership.
    let (target_port, invert) = if port_name.eq_ignore_ascii_case("q")
        && dff_cells_identity.contains(type_name)
    {
        ("q", false)
    } else if port_name.eq_ignore_ascii_case("qn") && dff_cells_inverted.contains(type_name) {
        ("qn", true)
    } else {
        log::warn!(
            "DFF identity override: unexpected D/Q pin mapping for cell '{}' instance '{}' output '{}'",
            type_name,
            inst_name,
            port_name
        );
        return Ok(true);
    };
    // Resolve D input
    let d_input = inst.connections.iter().find_map(|(p, nref)| {
        let pname = interner.resolve(*p).unwrap();
        if pname.eq_ignore_ascii_case("d") {
            Some(nref)
        } else {
            None
        }
    });
    if d_input.is_none() {
        return Err(format!(
            "DFF identity override: D input not found for cell '{}' instance '{}' (output '{}'). \
This cell was classified as DFF-like but does not expose a 'd' pin. \
Ensure the cell exposes a 'd' pin or do not classify it as DFF-like.",
            type_name, inst_name, target_port
        ));
    }
    // Build D (optionally inverted) and write to destination port.
    if let Some(d_bv) = build_d_bv(d_input.unwrap(), gb, resolved, nets, interner, invert)? {
        write_bv_to_port_destination(inst, interner, resolved, nets, target_port, &d_bv)?;
    } else {
        return Err(format!(
            "DFF override: D net not available for cell '{}' instance '{}' (output '{}'). \
This indicates missing drivers or a DFF classification mismatch; re-run with RUST_LOG=trace to diagnose.",
            type_name, inst_name, target_port
        ));
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig_sim::gate_sim::{self, Collect};
    use crate::liberty_proto::{Cell, Library, Pin, PinDirection};
    use crate::netlist::parse::{
        Net, NetRef, NetlistInstance, NetlistModule, NetlistPort, PortDirection,
    };
    use string_interner::{StringInterner, backend::StringBackend};
    use xlsynth::IrBits;

    fn input_pin(name: &str) -> Pin {
        Pin {
            name: name.to_string(),
            direction: PinDirection::Input as i32,
            function: String::new(),
            is_clocking_pin: false,
            ..Default::default()
        }
    }

    fn output_pin(name: &str, function: &str) -> Pin {
        Pin {
            name: name.to_string(),
            direction: PinDirection::Output as i32,
            function: function.to_string(),
            is_clocking_pin: false,
            ..Default::default()
        }
    }

    fn projection_order_test_liberty() -> Library {
        Library {
            cells: vec![
                Cell {
                    name: "BUF".to_string(),
                    pins: vec![input_pin("A"), output_pin("Y", "A")],
                    area: 1.0,
                    sequential: vec![],
                    clock_gate: None,
                    ..Default::default()
                },
                Cell {
                    name: "INV".to_string(),
                    pins: vec![input_pin("A"), output_pin("Y", "(!A)")],
                    area: 1.0,
                    sequential: vec![],
                    clock_gate: None,
                    ..Default::default()
                },
                Cell {
                    name: "AO21".to_string(),
                    pins: vec![
                        input_pin("A1"),
                        input_pin("A2"),
                        input_pin("B"),
                        output_pin("Y", "((A1 & A2) | B)"),
                    ],
                    area: 1.0,
                    sequential: vec![],
                    clock_gate: None,
                    ..Default::default()
                },
                // Similar to ASAP7 `DFFHQx1_ASAP7_75t_R`: a D flip-flop
                // whose Q output is modeled by the override path as Q = D.
                Cell {
                    name: "DFFHQ".to_string(),
                    pins: vec![input_pin("D"), input_pin("CLK"), output_pin("Q", "IQ")],
                    area: 1.0,
                    sequential: vec![],
                    clock_gate: None,
                    ..Default::default()
                },
                // Similar to ASAP7 `DFFHQNx1_ASAP7_75t_R`: a D flip-flop
                // whose QN output is modeled by the override path as QN = !D.
                Cell {
                    name: "DFFHQN".to_string(),
                    pins: vec![input_pin("D"), input_pin("CLK"), output_pin("QN", "IQN")],
                    area: 1.0,
                    sequential: vec![],
                    clock_gate: None,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    fn eval_single_output_bit(gate_fn: &GateFn, inputs: &[bool]) -> bool {
        let input_bits = inputs.iter().copied().map(IrBits::bool).collect::<Vec<_>>();
        let got = gate_sim::eval(gate_fn, &input_bits, Collect::None);
        got.outputs[0].get_bit(0).unwrap()
    }

    fn eval_output_by_name(gate_fn: &GateFn, inputs: Vec<IrBits>, output_name: &str) -> IrBits {
        let output_index = gate_fn
            .outputs
            .iter()
            .position(|output| output.name == output_name)
            .unwrap_or_else(|| panic!("output '{output_name}' not found"));
        let got = gate_sim::eval(gate_fn, &inputs, Collect::None);
        got.outputs.into_iter().nth(output_index).unwrap()
    }

    fn dffhq_identity_cells() -> HashSet<String> {
        let mut dff_cells_identity = HashSet::new();
        dff_cells_identity.insert("DFFHQ".to_string());
        dff_cells_identity
    }

    fn dffhqn_inverted_cells() -> HashSet<String> {
        let mut dff_cells_inverted = HashSet::new();
        dff_cells_inverted.insert("DFFHQN".to_string());
        dff_cells_inverted
    }

    fn simple_ref(idx: NetIndex) -> NetRef {
        NetRef::Simple(idx)
    }

    fn bit_ref(idx: NetIndex, bit: u32) -> NetRef {
        NetRef::BitSelect(idx, bit)
    }

    fn part_ref(idx: NetIndex, msb: u32, lsb: u32) -> NetRef {
        NetRef::PartSelect(idx, msb, lsb)
    }

    fn bool_ir(values: &[bool]) -> Vec<IrBits> {
        values.iter().copied().map(IrBits::bool).collect()
    }

    fn assert_bits(bits: &IrBits, expected_lsb_to_msb: &[bool]) {
        assert_eq!(bits.get_bit_count(), expected_lsb_to_msb.len());
        for (i, expected) in expected_lsb_to_msb.iter().enumerate() {
            assert_eq!(
                bits.get_bit(i).unwrap(),
                *expected,
                "bit {i} mismatch in {bits}"
            );
        }
    }

    struct NetlistFixture {
        interner: StringInterner<StringBackend<SymbolU32>>,
        nets: Vec<Net>,
        ports: Vec<NetlistPort>,
        wires: Vec<NetIndex>,
        instances: Vec<NetlistInstance>,
    }

    impl NetlistFixture {
        fn new() -> Self {
            Self {
                interner: StringInterner::new(),
                nets: Vec::new(),
                ports: Vec::new(),
                wires: Vec::new(),
                instances: Vec::new(),
            }
        }

        fn add_net(&mut self, name: &str, width: Option<(u32, u32)>) -> NetIndex {
            let idx = NetIndex(self.nets.len());
            let name = self.interner.get_or_intern(name);
            self.nets.push(Net { name, width });
            idx
        }

        fn input(&mut self, name: &str, width: Option<(u32, u32)>) -> NetIndex {
            let idx = self.add_net(name, width);
            self.ports.push(NetlistPort {
                direction: PortDirection::Input,
                width,
                name: self.nets[idx.0].name,
            });
            idx
        }

        fn output(&mut self, name: &str, width: Option<(u32, u32)>) -> NetIndex {
            let idx = self.add_net(name, width);
            self.ports.push(NetlistPort {
                direction: PortDirection::Output,
                width,
                name: self.nets[idx.0].name,
            });
            idx
        }

        fn wire(&mut self, name: &str, width: Option<(u32, u32)>) -> NetIndex {
            let idx = self.add_net(name, width);
            self.wires.push(idx);
            idx
        }

        fn inst(&mut self, type_name: &str, instance_name: &str, connections: Vec<(&str, NetRef)>) {
            let connections = connections
                .into_iter()
                .map(|(port, net_ref)| (self.interner.get_or_intern(port), net_ref))
                .collect();
            self.instances.push(NetlistInstance {
                type_name: self.interner.get_or_intern(type_name),
                instance_name: self.interner.get_or_intern(instance_name),
                connections,
                inst_lineno: 0,
                inst_colno: 0,
            });
        }

        fn module(&mut self) -> NetlistModule {
            NetlistModule {
                name: self.interner.get_or_intern("top"),
                net_index_range: 0..self.nets.len(),
                ports: self.ports.clone(),
                wires: self.wires.clone(),
                assigns: vec![],
                instances: self.instances.clone(),
            }
        }

        fn project(
            &mut self,
            dff_cells_identity: &HashSet<String>,
            dff_cells_inverted: &HashSet<String>,
        ) -> Result<GateFn, String> {
            let module = self.module();
            project_gatefn_from_netlist_and_liberty(
                &module,
                &self.nets,
                &self.interner,
                &projection_order_test_liberty(),
                dff_cells_identity,
                dff_cells_inverted,
            )
        }

        fn project_plain(&mut self) -> Result<GateFn, String> {
            self.project(&HashSet::new(), &HashSet::new())
        }
    }

    #[test]
    fn test_inverter_projection() {
        // Build a minimal netlist and Liberty proto for an inverter
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let y = interner.get_or_intern("y");
        let invx1 = interner.get_or_intern("INV");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: a,
                width: None,
            },
            Net {
                name: y,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: a,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: y,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: invx1,
            instance_name: u1,
            connections: vec![
                (
                    interner.get_or_intern("A"),
                    NetRef::Simple(crate::netlist::parse::NetIndex(0)),
                ),
                (
                    interner.get_or_intern("Y"),
                    NetRef::Simple(crate::netlist::parse::NetIndex(1)),
                ),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "INV".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "A".to_string(),
                        direction: 2,
                        function: "".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "Y".to_string(),
                        direction: 1,
                        function: "(!A)".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
        )
        .unwrap();
        let s = gate_fn.to_string();
        assert!(s.contains("not("), "GateFn output: {}", s);
    }

    #[test]
    fn test_unconnected_output_is_ignored() {
        // Cell with output Y unconnected should not error.
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let y = interner.get_or_intern("y");
        let invx1 = interner.get_or_intern("INV");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: a,
                width: None,
            },
            Net {
                name: y,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: a,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: y,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: invx1,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("A"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("Y"), NetRef::Unconnected),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "INV".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "A".to_string(),
                        direction: 2,
                        function: "".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "Y".to_string(),
                        direction: 1,
                        function: "(!A)".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let res = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
        );
        assert!(
            res.is_ok(),
            "projection should tolerate unconnected output: {:?}",
            res
        );
    }

    #[test]
    fn test_unconnected_input_errors_clearly() {
        // Cell with input A unconnected should surface a clear error.
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let y = interner.get_or_intern("y");
        let and2 = interner.get_or_intern("AND2");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![Net {
            name: y,
            width: None,
        }];
        let ports = vec![NetlistPort {
            direction: PortDirection::Output,
            width: None,
            name: y,
        }];
        let instances = vec![NetlistInstance {
            type_name: and2,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("A"), NetRef::Unconnected),
                (interner.get_or_intern("B"), NetRef::Unconnected),
                (interner.get_or_intern("Y"), NetRef::Simple(NetIndex(0))),
            ],
            inst_lineno: 10,
            inst_colno: 5,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "AND2".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "A".to_string(),
                        direction: 2,
                        function: "".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "B".to_string(),
                        direction: 2,
                        function: "".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "Y".to_string(),
                        direction: 1,
                        function: "(A & B)".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let err = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
        )
        .expect_err("should error on unconnected input");
        assert!(err.contains("missing inputs"), "{}", err);
        assert!(err.contains("<unconnected>"), "{}", err);
    }

    #[test]
    fn test_bitselect_output_projection() {
        // Build a netlist and Liberty proto for a buffer with bit-select output
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let n = interner.get_or_intern("n");
        let buf = interner.get_or_intern("BUF");
        let u1 = interner.get_or_intern("u1");
        // n is a 4-bit net
        let nets = vec![
            Net {
                name: a,
                width: None,
            },
            Net {
                name: n,
                width: Some((3, 0)),
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: a,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: Some((3, 0)),
                name: n,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: buf,
            instance_name: u1,
            connections: vec![
                (
                    interner.get_or_intern("A"),
                    NetRef::Simple(crate::netlist::parse::NetIndex(0)),
                ),
                (
                    interner.get_or_intern("Y"),
                    NetRef::BitSelect(crate::netlist::parse::NetIndex(1), 3),
                ),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "BUF".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "A".to_string(),
                        direction: 2,
                        function: "".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "Y".to_string(),
                        direction: 1,
                        function: "A".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
        )
        .expect("BitSelect output should be supported");
        let s = gate_fn.to_string();
        // n[3] should be assigned to a nonzero node (not literal false)
        assert!(
            s.contains("n[3] = %1") || s.contains("n[3] = %"),
            "GateFn output: {}",
            s
        );
    }

    #[test]
    fn test_liberty_projection_waits_for_selected_vector_bit_before_using_it() {
        let mut nl = NetlistFixture::new();
        nl.input("n0", None);
        let n1 = nl.input("n1", None);
        let n2 = nl.input("n2", None);
        let ctrl = nl.input("ctrl", None);
        let other = nl.input("other", None);
        let clk = nl.input("clk", None);
        let y = nl.output("y", None);
        let exp_bus = nl.wire("exp_bus", Some((7, 0)));

        nl.inst(
            "DFFHQ",
            "exp_reg_2",
            vec![
                ("D", simple_ref(n2)),
                ("CLK", simple_ref(clk)),
                ("Q", bit_ref(exp_bus, 2)),
            ],
        );
        nl.inst(
            "AO21",
            "use_bit_1",
            vec![
                ("A1", bit_ref(exp_bus, 1)),
                ("A2", simple_ref(ctrl)),
                ("B", simple_ref(other)),
                ("Y", simple_ref(y)),
            ],
        );
        nl.inst(
            "DFFHQN",
            "exp_reg_1",
            vec![
                ("D", simple_ref(n1)),
                ("CLK", simple_ref(clk)),
                ("QN", bit_ref(exp_bus, 1)),
            ],
        );
        let gate_fn = nl
            .project(&dffhq_identity_cells(), &dffhqn_inverted_cells())
            .expect("projection should wait for exp_bus[1] before AO21");

        // Inputs are n0, n1, n2, ctrl, other, clk. With n1=0, ctrl=1,
        // other=0, the AO21 should see exp_bus[1]=~n1 and drive y=1.
        assert!(eval_single_output_bit(
            &gate_fn,
            &[false, false, false, true, false, false]
        ));
    }

    #[test]
    fn test_liberty_projection_waits_for_combinational_vector_bit_before_using_it() {
        let mut nl = NetlistFixture::new();
        let n1 = nl.input("n1", None);
        let n2 = nl.input("n2", None);
        let ctrl = nl.input("ctrl", None);
        let other = nl.input("other", None);
        let y = nl.output("y", None);
        let exp_bus = nl.wire("exp_bus", Some((7, 0)));

        nl.inst(
            "BUF",
            "drive_bit_2",
            vec![("A", simple_ref(n2)), ("Y", bit_ref(exp_bus, 2))],
        );
        nl.inst(
            "AO21",
            "use_bit_1",
            vec![
                ("A1", bit_ref(exp_bus, 1)),
                ("A2", simple_ref(ctrl)),
                ("B", simple_ref(other)),
                ("Y", simple_ref(y)),
            ],
        );
        nl.inst(
            "INV",
            "drive_bit_1",
            vec![("A", simple_ref(n1)), ("Y", bit_ref(exp_bus, 1))],
        );
        let gate_fn = nl
            .project_plain()
            .expect("projection should wait for combinational exp_bus[1] driver");

        // Inputs are n1, n2, ctrl, other. With n1=0, ctrl=1, other=0,
        // the AO21 should see exp_bus[1]=~n1 and drive y=1.
        assert!(eval_single_output_bit(
            &gate_fn,
            &[false, false, true, false]
        ));
    }

    #[test]
    fn test_liberty_projection_rejects_duplicate_bit_drivers() {
        let mut nl = NetlistFixture::new();
        let a = nl.input("a", None);
        let b = nl.input("b", None);
        let y = nl.output("y", Some((3, 0)));

        nl.inst(
            "BUF",
            "u0",
            vec![("A", simple_ref(a)), ("Y", bit_ref(y, 1))],
        );
        nl.inst(
            "BUF",
            "u1",
            vec![("A", simple_ref(b)), ("Y", bit_ref(y, 1))],
        );
        let err = nl
            .project_plain()
            .expect_err("duplicate bit driver should be rejected");
        assert!(err.contains("y"), "unexpected error: {err}");
        assert!(err.contains("bit 1"), "unexpected error: {err}");
    }

    #[test]
    fn test_liberty_projection_waits_for_partselect_bits_before_dff_override() {
        let mut nl = NetlistFixture::new();
        let b1 = nl.input("b1", None);
        let b2 = nl.input("b2", None);
        let y = nl.output("y", Some((1, 0)));
        let bus = nl.wire("bus", Some((2, 0)));

        nl.inst(
            "BUF",
            "drive_bit_2",
            vec![("A", simple_ref(b2)), ("Y", bit_ref(bus, 2))],
        );
        nl.inst(
            "DFFHQ",
            "capture_slice",
            vec![("D", part_ref(bus, 2, 1)), ("Q", simple_ref(y))],
        );
        nl.inst(
            "BUF",
            "drive_bit_1",
            vec![("A", simple_ref(b1)), ("Y", bit_ref(bus, 1))],
        );
        let gate_fn = nl
            .project(&dffhq_identity_cells(), &HashSet::new())
            .expect("DFF override should wait for every selected D bit");

        let y_bits = eval_output_by_name(&gate_fn, bool_ir(&[true, false]), "y");
        assert_bits(&y_bits, &[true, false]);
    }

    #[test]
    fn test_liberty_projection_defaults_only_final_output_missing_bits() {
        let mut nl = NetlistFixture::new();
        let a = nl.input("a", None);
        let b = nl.input("b", None);
        let ctrl = nl.input("ctrl", None);
        let other = nl.input("other", None);
        let out_bus = nl.output("out_bus", Some((3, 0)));
        let y = nl.output("y", None);

        nl.inst(
            "BUF",
            "drive_output_bit_2",
            vec![("A", simple_ref(a)), ("Y", bit_ref(out_bus, 2))],
        );
        nl.inst(
            "AO21",
            "use_output_bit_1",
            vec![
                ("A1", bit_ref(out_bus, 1)),
                ("A2", simple_ref(ctrl)),
                ("B", simple_ref(other)),
                ("Y", simple_ref(y)),
            ],
        );
        nl.inst(
            "INV",
            "drive_output_bit_1",
            vec![("A", simple_ref(b)), ("Y", bit_ref(out_bus, 1))],
        );
        let gate_fn = nl
            .project_plain()
            .expect("internal output-net consumer should wait for the real selected bit");

        let inputs = bool_ir(&[true, false, true, false]);
        let out_bits = eval_output_by_name(&gate_fn, inputs.clone(), "out_bus");
        let y_bits = eval_output_by_name(&gate_fn, inputs, "y");
        assert_bits(&out_bits, &[false, true, true, false]);
        assert_bits(&y_bits, &[true]);
    }

    #[test]
    fn test_liberty_projection_honors_ascending_packed_range_bit_numbers() {
        let mut nl = NetlistFixture::new();
        let a = nl.input("a", None);
        let b = nl.input("b", None);
        let ctrl = nl.input("ctrl", None);
        let other = nl.input("other", None);
        let bus = nl.output("bus", Some((0, 3)));
        let y = nl.output("y", None);

        nl.inst(
            "BUF",
            "drive_bit_2",
            vec![("A", simple_ref(b)), ("Y", bit_ref(bus, 2))],
        );
        nl.inst(
            "AO21",
            "use_bit_1",
            vec![
                ("A1", bit_ref(bus, 1)),
                ("A2", simple_ref(ctrl)),
                ("B", simple_ref(other)),
                ("Y", simple_ref(y)),
            ],
        );
        nl.inst(
            "INV",
            "drive_bit_1",
            vec![("A", simple_ref(a)), ("Y", bit_ref(bus, 1))],
        );
        let gate_fn = nl
            .project_plain()
            .expect("ascending packed ranges should use declared bit numbers");

        let inputs = bool_ir(&[false, true, true, false]);
        let bus_bits = eval_output_by_name(&gate_fn, inputs.clone(), "bus");
        let y_bits = eval_output_by_name(&gate_fn, inputs, "y");
        assert_bits(&bus_bits, &[false, true, true, false]);
        assert_bits(&y_bits, &[true]);
    }

    #[test]
    fn test_liberty_projection_rejects_simple_output_overlap_with_bit_driver() {
        let mut nl = NetlistFixture::new();
        let x = nl.input("x", Some((3, 0)));
        let a = nl.input("a", None);
        let bus = nl.output("bus", Some((3, 0)));

        nl.inst(
            "DFFHQ",
            "drive_whole_bus",
            vec![("D", simple_ref(x)), ("Q", simple_ref(bus))],
        );
        nl.inst(
            "BUF",
            "drive_bit_1",
            vec![("A", simple_ref(a)), ("Y", bit_ref(bus, 1))],
        );
        let err = nl
            .project(&dffhq_identity_cells(), &HashSet::new())
            .expect_err("whole-vector write should conflict with later bit write");
        assert!(err.contains("bus"), "unexpected error: {err}");
        assert!(err.contains("bit 1"), "unexpected error: {err}");
    }

    #[test]
    fn test_liberty_projection_rejects_partselect_output_overlap_with_bit_driver() {
        let mut nl = NetlistFixture::new();
        let x = nl.input("x", Some((1, 0)));
        let a = nl.input("a", None);
        let bus = nl.output("bus", Some((3, 0)));

        nl.inst(
            "DFFHQ",
            "drive_bus_slice",
            vec![("D", simple_ref(x)), ("Q", part_ref(bus, 2, 1))],
        );
        nl.inst(
            "BUF",
            "drive_bit_1",
            vec![("A", simple_ref(a)), ("Y", bit_ref(bus, 1))],
        );
        let err = nl
            .project(&dffhq_identity_cells(), &HashSet::new())
            .expect_err("slice write should conflict with later overlapping bit write");
        assert!(err.contains("bus"), "unexpected error: {err}");
        assert!(err.contains("bit 1"), "unexpected error: {err}");
    }

    #[test]
    fn test_dff_override_reads_partselect_and_writes_bitselect() {
        let mut nl = NetlistFixture::new();
        let x = nl.input("x", Some((3, 0)));
        let y = nl.output("y", Some((3, 0)));

        nl.inst(
            "DFFHQ",
            "capture_single_bit_slice",
            vec![("D", part_ref(x, 1, 1)), ("Q", bit_ref(y, 2))],
        );
        let gate_fn = nl
            .project(&dffhq_identity_cells(), &HashSet::new())
            .expect("DFF override should read a one-bit part-select and write a bit-select");

        let y_bits =
            eval_output_by_name(&gate_fn, vec![IrBits::make_ubits(4, 0b0010).unwrap()], "y");
        assert_bits(&y_bits, &[false, false, true, false]);
    }

    #[test]
    fn test_liberty_projection_waits_for_full_vector_before_dff_override() {
        let mut nl = NetlistFixture::new();
        let low = nl.input("low", None);
        let high = nl.input("high", None);
        let y = nl.output("y", Some((1, 0)));
        let bus = nl.wire("bus", Some((1, 0)));

        nl.inst(
            "BUF",
            "drive_high",
            vec![("A", simple_ref(high)), ("Y", bit_ref(bus, 1))],
        );
        nl.inst(
            "DFFHQ",
            "capture_bus",
            vec![("D", simple_ref(bus)), ("Q", simple_ref(y))],
        );
        nl.inst(
            "BUF",
            "drive_low",
            vec![("A", simple_ref(low)), ("Y", bit_ref(bus, 0))],
        );
        let gate_fn = nl
            .project(&dffhq_identity_cells(), &HashSet::new())
            .expect("DFF override should wait for every bit of a full-vector D input");

        let y_bits = eval_output_by_name(&gate_fn, bool_ir(&[true, false]), "y");
        assert_bits(&y_bits, &[true, false]);
    }

    #[test]
    fn test_dff_identity_q_bitselect_expands_to_full_width() {
        // Build a netlist with a DFF cell where Q drives a single bit of a 4-bit output
        // net. Regression: ensure we size the destination vector to the full
        // net width (4), not just the bit index slice width (1), and that no
        // width-assertion fires.
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let y = interner.get_or_intern("y");
        let dff = interner.get_or_intern("DFF");
        let u1 = interner.get_or_intern("u1");

        // a: 1-bit input, y: 4-bit output
        let nets = vec![
            Net {
                name: a,
                width: None,
            },
            Net {
                name: y,
                width: Some((3, 0)),
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: a,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: Some((3, 0)),
                name: y,
            },
        ];
        // Connect D = a, Q = y[3]
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (
                    interner.get_or_intern("D"),
                    NetRef::Simple(crate::netlist::parse::NetIndex(0)),
                ),
                (
                    interner.get_or_intern("Q"),
                    NetRef::BitSelect(crate::netlist::parse::NetIndex(1), 3),
                ),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        // Liberty with a DFF cell having pins D (input) and Q (output)
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "DFF".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "D".to_string(),
                        direction: 2,
                        function: "".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "Q".to_string(),
                        direction: 1,
                        function: "Q".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut dff_cells = std::collections::HashSet::new();
        dff_cells.insert("DFF".to_string());

        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &dff_cells,
            &HashSet::new(),
        )
        .expect("projection should succeed without width mismatch");

        // Signature must show full 4-bit output width.
        assert_eq!(gate_fn.get_signature(), "fn top(a: bits[1]) -> bits[4]");
        // And we should be writing to y[3] (the selected bit) in the output mapping.
        let s = gate_fn.to_string();
        assert!(s.contains("  y[3] = "), "GateFn output: {}", s);
    }

    #[test]
    fn test_dff_identity_q_partselect_writes_at_offset() {
        // DFF cell where Q drives a 4-bit slice y[7:4] of an 8-bit output net.
        // Regression: ensure we size to full width (8) and write at lsb offset 4.
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let dff = interner.get_or_intern("DFF");
        let u1 = interner.get_or_intern("u1");

        // x: 8-bit input, y: 8-bit output
        let nets = vec![
            Net {
                name: x,
                width: Some((7, 0)),
            },
            Net {
                name: y,
                width: Some((7, 0)),
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: Some((7, 0)),
                name: x,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: Some((7, 0)),
                name: y,
            },
        ];
        // Connect D = x[7:4], Q = y[7:4]
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (
                    interner.get_or_intern("D"),
                    NetRef::PartSelect(crate::netlist::parse::NetIndex(0), 7, 4),
                ),
                (
                    interner.get_or_intern("Q"),
                    NetRef::PartSelect(crate::netlist::parse::NetIndex(1), 7, 4),
                ),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "DFF".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "D".to_string(),
                        direction: 2,
                        function: "".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "Q".to_string(),
                        direction: 1,
                        function: "Q".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut dff_cells = std::collections::HashSet::new();
        dff_cells.insert("DFF".to_string());

        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &dff_cells,
            &HashSet::new(),
        )
        .expect("projection should succeed without width mismatch");

        // Signature must show full 8-bit output width.
        assert_eq!(gate_fn.get_signature(), "fn top(x: bits[8]) -> bits[8]");
        // And we should be writing to y[4]..y[7] in the output mapping.
        let s = gate_fn.to_string();
        assert!(s.contains("  y[4] = "), "GateFn output: {}", s);
        assert!(s.contains("  y[7] = "), "GateFn output: {}", s);
    }

    #[test]
    fn test_dff_identity_override_simple() {
        // D and Q are both Simple netrefs
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let q = interner.get_or_intern("q");
        let dff = interner.get_or_intern("DFF");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: q,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: q,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("d"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("q"), NetRef::Simple(NetIndex(1))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "DFF".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "d".to_string(),
                        direction: 2,
                        function: "".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "q".to_string(),
                        direction: 1,
                        function: "d".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut dff_cells = std::collections::HashSet::new();
        dff_cells.insert("DFF".to_string());
        // Set up the input value for D
        let mut gb = GateBuilder::new("top".to_string(), GateBuilderOptions::no_opt());
        let d_bv = gb.add_input("d".to_string(), 1);
        let mut net_to_bv = HashMap::new();
        net_to_bv.insert(NetIndex(0), d_bv.clone());
        // Run the projection
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &dff_cells,
            &HashSet::new(),
        )
        .unwrap();
        // The output should be wired directly from D
        let q_input = gate_fn.inputs.iter().find(|i| i.name == "d").unwrap();
        let q_output = gate_fn.outputs.iter().find(|o| o.name == "q").unwrap();
        assert_eq!(
            q_input.bit_vector.get_lsb(0),
            q_output.bit_vector.get_lsb(0)
        );
    }

    #[test]
    fn test_dff_inverted_qn_simple() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let qn = interner.get_or_intern("qn");
        let dffn = interner.get_or_intern("DFFN");
        let u1 = interner.get_or_intern("u1");

        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: qn,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: qn,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dffn,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("D"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("QN"), NetRef::Simple(NetIndex(1))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "DFFN".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "D".to_string(),
                        direction: 2,
                        function: "".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "QN".to_string(),
                        direction: 1,
                        function: "IQN".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let dff_cells_identity = HashSet::new();
        let mut dff_cells_inverted = HashSet::new();
        dff_cells_inverted.insert("DFFN".to_string());
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &dff_cells_identity,
            &dff_cells_inverted,
        )
        .unwrap();
        let s = gate_fn.to_string();
        assert!(s.contains("qn[0] = not("), "GateFn output: {}", s);
    }

    #[test]
    fn test_dff_inverted_qn_bitselect() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let y = interner.get_or_intern("y");
        let dffn = interner.get_or_intern("DFFN");
        let u1 = interner.get_or_intern("u1");

        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: y,
                width: Some((3, 0)),
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: Some((3, 0)),
                name: y,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dffn,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("D"), NetRef::Simple(NetIndex(0))),
                (
                    interner.get_or_intern("QN"),
                    NetRef::BitSelect(NetIndex(1), 2),
                ),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "DFFN".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "D".to_string(),
                        direction: 2,
                        function: "".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "QN".to_string(),
                        direction: 1,
                        function: "IQN".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let dff_cells_identity = HashSet::new();
        let mut dff_cells_inverted = HashSet::new();
        dff_cells_inverted.insert("DFFN".to_string());
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &dff_cells_identity,
            &dff_cells_inverted,
        )
        .unwrap();
        let s = gate_fn.to_string();
        assert!(s.contains("  y[2] = not("), "GateFn output: {}", s);
    }

    #[test]
    fn test_dff_inverted_qn_partselect() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let dffn = interner.get_or_intern("DFFN");
        let u1 = interner.get_or_intern("u1");

        let nets = vec![
            Net {
                name: x,
                width: Some((7, 0)),
            },
            Net {
                name: y,
                width: Some((7, 0)),
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: Some((7, 0)),
                name: x,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: Some((7, 0)),
                name: y,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dffn,
            instance_name: u1,
            connections: vec![
                (
                    interner.get_or_intern("D"),
                    NetRef::PartSelect(NetIndex(0), 7, 4),
                ),
                (
                    interner.get_or_intern("QN"),
                    NetRef::PartSelect(NetIndex(1), 7, 4),
                ),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "DFFN".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "D".to_string(),
                        direction: 2,
                        function: "".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "QN".to_string(),
                        direction: 1,
                        function: "IQN".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let dff_cells_identity = HashSet::new();
        let mut dff_cells_inverted = HashSet::new();
        dff_cells_inverted.insert("DFFN".to_string());
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &dff_cells_identity,
            &dff_cells_inverted,
        )
        .unwrap();
        let s = gate_fn.to_string();
        assert!(s.contains("  y[4] = not("), "GateFn output: {}", s);
        assert!(s.contains("  y[7] = not("), "GateFn output: {}", s);
    }

    #[test]
    fn test_sequential_iqn_inverts_next_state_when_collapsing() {
        // Collapsing sequential state variables should treat IQN as the complement of
        // IQ.
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let q = interner.get_or_intern("q");
        let qn = interner.get_or_intern("qn");
        // Synthetic cell name, not an ASAP7-specific alias.
        let dff = interner.get_or_intern("DFFX");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: q,
                width: None,
            },
            Net {
                name: qn,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: q,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: qn,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("D"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("Q"), NetRef::Simple(NetIndex(1))),
                (interner.get_or_intern("QN"), NetRef::Simple(NetIndex(2))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "DFFX".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "D".to_string(),
                        direction: 2,
                        function: "".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "Q".to_string(),
                        direction: 1,
                        function: "IQ".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "QN".to_string(),
                        direction: 1,
                        function: "IQN".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![
                    crate::liberty_proto::Sequential {
                        state_var: "IQ".to_string(),
                        next_state: "D".to_string(),
                        clock_expr: "CLK".to_string(),
                        kind: crate::liberty_proto::SequentialKind::Ff as i32,
                        clear_expr: String::new(),
                        preset_expr: String::new(),
                    },
                    crate::liberty_proto::Sequential {
                        state_var: "IQN".to_string(),
                        next_state: "D".to_string(),
                        clock_expr: "CLK".to_string(),
                        kind: crate::liberty_proto::SequentialKind::Ff as i32,
                        clear_expr: String::new(),
                        preset_expr: String::new(),
                    },
                ],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let gate_fn = project_gatefn_from_netlist_and_liberty_with_options(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
            &GateFnProjectOptions {
                collapse_sequential: true,
            },
        )
        .unwrap();
        let d_input = gate_fn
            .inputs
            .iter()
            .find(|input| input.name == "d")
            .unwrap();
        let q_output = gate_fn
            .outputs
            .iter()
            .find(|output| output.name == "q")
            .unwrap();
        let qn_output = gate_fn
            .outputs
            .iter()
            .find(|output| output.name == "qn")
            .unwrap();
        let d_bit = *d_input.bit_vector.get_lsb(0);
        let q_bit = *q_output.bit_vector.get_lsb(0);
        let qn_bit = *qn_output.bit_vector.get_lsb(0);
        assert_eq!(q_bit, d_bit);
        assert_eq!(qn_bit, d_bit.negate());
    }

    #[test]
    fn test_collapse_ignores_unsupported_unused_cells() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let y = interner.get_or_intern("y");
        let inv = interner.get_or_intern("INV");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: a,
                width: None,
            },
            Net {
                name: y,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: a,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: y,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: inv,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("A"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("Y"), NetRef::Simple(NetIndex(1))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        // Include an unsupported sequential cell in the library, but do not instantiate
        // it.
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![
                crate::liberty_proto::Cell {
                    name: "INV".to_string(),
                    pins: vec![
                        crate::liberty_proto::Pin {
                            name: "A".to_string(),
                            direction: 2,
                            function: String::new(),
                            is_clocking_pin: false,
                            ..Default::default()
                        },
                        crate::liberty_proto::Pin {
                            name: "Y".to_string(),
                            direction: 1,
                            function: "!A".to_string(),
                            is_clocking_pin: false,
                            ..Default::default()
                        },
                    ],
                    area: 1.0,
                    sequential: vec![],
                    clock_gate: None,
                    ..Default::default()
                },
                crate::liberty_proto::Cell {
                    name: "DFFEN".to_string(),
                    pins: vec![
                        crate::liberty_proto::Pin {
                            name: "D".to_string(),
                            direction: 2,
                            function: String::new(),
                            is_clocking_pin: false,
                            ..Default::default()
                        },
                        crate::liberty_proto::Pin {
                            name: "EN".to_string(),
                            direction: 2,
                            function: String::new(),
                            is_clocking_pin: false,
                            ..Default::default()
                        },
                        crate::liberty_proto::Pin {
                            name: "CLK".to_string(),
                            direction: 2,
                            function: String::new(),
                            is_clocking_pin: true,
                            ..Default::default()
                        },
                        crate::liberty_proto::Pin {
                            name: "Q".to_string(),
                            direction: 1,
                            function: "IQ".to_string(),
                            is_clocking_pin: false,
                            ..Default::default()
                        },
                    ],
                    area: 1.0,
                    sequential: vec![crate::liberty_proto::Sequential {
                        state_var: "IQ".to_string(),
                        next_state: "(!EN * IQ) + (EN * D)".to_string(),
                        clock_expr: "CLK".to_string(),
                        kind: crate::liberty_proto::SequentialKind::Ff as i32,
                        clear_expr: String::new(),
                        preset_expr: String::new(),
                    }],
                    clock_gate: None,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let gate_fn = project_gatefn_from_netlist_and_liberty_with_options(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
            &GateFnProjectOptions {
                collapse_sequential: true,
            },
        )
        .expect("unused unsupported sequential cells should not cause collapse to fail");
        let s = gate_fn.to_string();
        assert!(s.contains("y[0] = not("), "GateFn output: {s}");
    }

    #[test]
    fn test_collapse_errors_on_async_flop_controls() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let rn = interner.get_or_intern("rn");
        let q = interner.get_or_intern("q");
        // Approximate ASAP7 async-reset flavor (e.g. DFFASRHQNx1_ASAP7_75t_R).
        let dff = interner.get_or_intern("DFFAR");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: rn,
                width: None,
            },
            Net {
                name: q,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: rn,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: q,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("D"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("RN"), NetRef::Simple(NetIndex(1))),
                (interner.get_or_intern("Q"), NetRef::Simple(NetIndex(2))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "DFFAR".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "D".to_string(),
                        direction: 2,
                        function: String::new(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "RN".to_string(),
                        direction: 2,
                        function: String::new(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "CLK".to_string(),
                        direction: 2,
                        function: String::new(),
                        is_clocking_pin: true,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "Q".to_string(),
                        direction: 1,
                        function: "IQ".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![crate::liberty_proto::Sequential {
                    state_var: "IQ".to_string(),
                    next_state: "D".to_string(),
                    clock_expr: "CLK".to_string(),
                    kind: crate::liberty_proto::SequentialKind::Ff as i32,
                    clear_expr: "!RN".to_string(),
                    preset_expr: String::new(),
                }],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let err = project_gatefn_from_netlist_and_liberty_with_options(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
            &GateFnProjectOptions {
                collapse_sequential: true,
            },
        )
        .expect_err("async flop controls should be rejected when collapsing");
        assert!(
            err.contains("async clear/preset"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_collapse_errors_on_latch_enable() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let clk = interner.get_or_intern("clk");
        let q = interner.get_or_intern("q");
        let lat = interner.get_or_intern("LATCH");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: clk,
                width: None,
            },
            Net {
                name: q,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: clk,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: q,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: lat,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("D"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("CLK"), NetRef::Simple(NetIndex(1))),
                (interner.get_or_intern("Q"), NetRef::Simple(NetIndex(2))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "LATCH".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "D".to_string(),
                        direction: 2,
                        function: String::new(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "CLK".to_string(),
                        direction: 2,
                        function: String::new(),
                        is_clocking_pin: true,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "Q".to_string(),
                        direction: 1,
                        function: "IQ".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![crate::liberty_proto::Sequential {
                    state_var: "IQ".to_string(),
                    next_state: "D".to_string(),
                    clock_expr: "CLK".to_string(),
                    kind: crate::liberty_proto::SequentialKind::Latch as i32,
                    clear_expr: String::new(),
                    preset_expr: String::new(),
                }],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let err = project_gatefn_from_netlist_and_liberty_with_options(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
            &GateFnProjectOptions {
                collapse_sequential: true,
            },
        )
        .expect_err("latch enable should be rejected when collapsing");
        assert!(
            err.contains("latches with enable"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_collapse_errors_on_state_var_references_in_next_state() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let en = interner.get_or_intern("en");
        let q = interner.get_or_intern("q");
        let dff = interner.get_or_intern("DFFEN");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: en,
                width: None,
            },
            Net {
                name: q,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: en,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: q,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("D"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("EN"), NetRef::Simple(NetIndex(1))),
                (interner.get_or_intern("Q"), NetRef::Simple(NetIndex(2))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "DFFEN".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "D".to_string(),
                        direction: 2,
                        function: String::new(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "EN".to_string(),
                        direction: 2,
                        function: String::new(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "CLK".to_string(),
                        direction: 2,
                        function: String::new(),
                        is_clocking_pin: true,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "Q".to_string(),
                        direction: 1,
                        function: "IQ".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![crate::liberty_proto::Sequential {
                    state_var: "IQ".to_string(),
                    next_state: "(!EN * IQ) + (EN * D)".to_string(),
                    clock_expr: "CLK".to_string(),
                    kind: crate::liberty_proto::SequentialKind::Ff as i32,
                    clear_expr: String::new(),
                    preset_expr: String::new(),
                }],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let err = project_gatefn_from_netlist_and_liberty_with_options(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
            &GateFnProjectOptions {
                collapse_sequential: true,
            },
        )
        .expect_err("state-var references should be rejected when collapsing");
        assert!(
            err.contains("reference state variables"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_collapse_errors_on_non_pin_inputs_in_next_state() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let q = interner.get_or_intern("q");
        // Approximate ASAP7 QN-only flop naming (e.g. DFFLQNx1_ASAP7_75t_R).
        let dff = interner.get_or_intern("DFFNP");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: q,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: q,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("D"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("Q"), NetRef::Simple(NetIndex(1))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "DFFNP".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "D".to_string(),
                        direction: 2,
                        function: String::new(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "CLK".to_string(),
                        direction: 2,
                        function: String::new(),
                        is_clocking_pin: true,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "Q".to_string(),
                        direction: 1,
                        function: "IQ".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![crate::liberty_proto::Sequential {
                    state_var: "IQ".to_string(),
                    next_state: "D & FOO".to_string(),
                    clock_expr: "CLK".to_string(),
                    kind: crate::liberty_proto::SequentialKind::Ff as i32,
                    clear_expr: String::new(),
                    preset_expr: String::new(),
                }],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let err = project_gatefn_from_netlist_and_liberty_with_options(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
            &GateFnProjectOptions {
                collapse_sequential: true,
            },
        )
        .expect_err("non-pin inputs should be rejected when collapsing");
        assert!(err.contains("non-pin inputs"), "unexpected error: {err}");
    }

    #[test]
    fn test_dff_override_allows_literal_d_input() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let q = interner.get_or_intern("q");
        let dff = interner.get_or_intern("DFF");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![Net {
            name: q,
            width: None,
        }];
        let ports = vec![NetlistPort {
            direction: PortDirection::Output,
            width: None,
            name: q,
        }];
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (
                    interner.get_or_intern("D"),
                    NetRef::Literal(xlsynth::IrBits::make_ubits(1, 1).unwrap()),
                ),
                (interner.get_or_intern("Q"), NetRef::Simple(NetIndex(0))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "DFF".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "D".to_string(),
                        direction: 2,
                        function: String::new(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "Q".to_string(),
                        direction: 1,
                        function: "IQ".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut dff_cells_identity = HashSet::new();
        dff_cells_identity.insert("DFF".to_string());
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &dff_cells_identity,
            &HashSet::new(),
        )
        .expect("DFF override should accept literal D inputs");
        let s = gate_fn.to_string();
        assert!(
            s.contains("literal(true)") || (s.contains("literal(false)") && s.contains("not(")),
            "expected constant-true behavior in gatefn, got: {s}"
        );
    }

    #[test]
    fn test_instance_input_tied_to_literal() {
        // Build a netlist and Liberty proto for a 2-input AND gate, with one input tied
        // to 1'b0
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let y = interner.get_or_intern("y");
        let and2 = interner.get_or_intern("AND2");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: a,
                width: None,
            },
            Net {
                name: y,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: a,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: y,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: and2,
            instance_name: u1,
            connections: vec![
                // .A(a), .B(1'b0), .Y(y)
                (interner.get_or_intern("A"), NetRef::Simple(NetIndex(0))),
                (
                    interner.get_or_intern("B"),
                    NetRef::Literal(xlsynth::IrBits::make_ubits(1, 0).unwrap()),
                ),
                (interner.get_or_intern("Y"), NetRef::Simple(NetIndex(1))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = crate::liberty_proto::Library {
            cells: vec![crate::liberty_proto::Cell {
                name: "AND2".to_string(),
                pins: vec![
                    crate::liberty_proto::Pin {
                        name: "A".to_string(),
                        direction: 2,
                        function: "".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "B".to_string(),
                        direction: 2,
                        function: "".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    crate::liberty_proto::Pin {
                        name: "Y".to_string(),
                        direction: 1,
                        function: "(A & B)".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
        )
        .unwrap();
        let s = gate_fn.to_string();
        // The output should be always false (since B is tied to 0)
        assert!(
            s.contains("literal(false)") || s.contains("= %0"),
            "GateFn output: {}",
            s
        );
    }
}
