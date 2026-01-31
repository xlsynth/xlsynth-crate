// SPDX-License-Identifier: Apache-2.0

//! Project a parsed netlist and Liberty proto into a GateFn.

use crate::aig::gate::{AigBitVector, AigOperand, GateFn};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::liberty_proto::Library;
use crate::netlist::parse::{Net, NetIndex, NetlistModule};
use std::collections::HashMap;
use std::collections::HashSet;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

fn build_cell_formula_map(
    liberty_lib: &Library,
) -> HashMap<(String, String), (crate::liberty::cell_formula::Term, String)> {
    let mut cell_formula_map = HashMap::new();
    for cell in &liberty_lib.cells {
        let mut sequential_terms: HashMap<String, (crate::liberty::cell_formula::Term, String)> =
            HashMap::new();
        for seq in &cell.sequential {
            if seq.state_var.is_empty() || seq.next_state.is_empty() {
                continue;
            }
            match crate::liberty::cell_formula::parse_formula(&seq.next_state) {
                Ok(term) => {
                    sequential_terms.insert(seq.state_var.clone(), (term, seq.next_state.clone()));
                }
                Err(e) => {
                    log::warn!(
                        "Failed to parse next_state for cell '{}' state '{}' (next_state: \"{}\"): {}",
                        cell.name,
                        seq.state_var,
                        seq.next_state,
                        e
                    );
                }
            }
        }
        for pin in &cell.pins {
            if pin.direction == 1 && !pin.function.is_empty() {
                // If the pin function references a sequential state variable (e.g. IQ),
                // prefer the sequential next_state formula as the driving logic.
                if let Some((term, next_state_string)) = sequential_terms.get(&pin.function) {
                    cell_formula_map.insert(
                        (cell.name.clone(), pin.name.clone()),
                        (term.clone(), next_state_string.clone()),
                    );
                    continue;
                }

                let original_formula_string = pin.function.clone();
                match crate::liberty::cell_formula::parse_formula(&pin.function) {
                    Ok(term) => {
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
    cell_formula_map
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

fn process_instance_outputs(
    inst: &crate::netlist::parse::NetlistInstance,
    type_name: &str,
    inst_name: &str,
    pin_directions: &HashMap<&str, i32>,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    nets: &[Net],
    gb: &mut GateBuilder,
    net_to_bv: &mut HashMap<NetIndex, AigBitVector>,
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
                net_to_bv,
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
                .unwrap();
            match netref {
                crate::netlist::parse::NetRef::Simple(net_idx) => {
                    if net_to_bv.contains_key(net_idx) {
                        panic!("Output net '{}' already assigned", net_idx.0);
                    }
                    net_to_bv.insert(*net_idx, AigBitVector::from_bit(out_op));
                }
                crate::netlist::parse::NetRef::BitSelect(net_idx, bit) => {
                    let width = nets[net_idx.0]
                        .width
                        .map(|(msb, lsb)| (msb as usize) - (lsb as usize) + 1)
                        .unwrap_or(1);
                    let mut bv = net_to_bv.remove(net_idx).unwrap_or_else(|| {
                        AigBitVector::from_lsb_is_index_0(&vec![gb.get_false(); width])
                    });
                    if (*bit as usize) >= width {
                        panic!(
                            "Bit-select out of range for output net '{}' (width {}) in instance '{}' port '{}', net_idx={:?}, bit={}",
                            interner.resolve(nets[net_idx.0].name).unwrap(),
                            width,
                            inst_name,
                            port_name,
                            net_idx,
                            bit
                        );
                    }
                    bv.set_lsb(*bit as usize, out_op);
                    net_to_bv.insert(*net_idx, bv);
                }
                _ => {
                    // Only simple/bitselect supported for output
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
    let cell_formula_map = build_cell_formula_map(liberty_lib);
    let module_name = interner.resolve(module.name).unwrap();
    let mut gb = GateBuilder::new(module_name.to_string(), GateBuilderOptions::no_opt());
    let mut net_to_bv: HashMap<NetIndex, AigBitVector> = HashMap::new();
    collect_module_io_nets(module, nets, interner, &mut gb, &mut net_to_bv);
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
            let (input_map, missing_inputs, port_map) = build_instance_input_map(
                inst,
                &pin_directions,
                interner,
                nets,
                &net_to_bv,
                &mut gb,
            );
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
                &mut net_to_bv,
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
                // Recompute missing inputs for this instance w.r.t. current net_to_bv.
                let (_input_map, missing_inputs, _port_map) = build_instance_input_map(
                    inst,
                    &pin_directions,
                    interner,
                    nets,
                    &net_to_bv,
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
            let width = net
                .width
                .map(|(msb, lsb)| (msb as usize) - (lsb as usize) + 1)
                .unwrap_or(1);
            let bv = match net_to_bv.get(&net_idx) {
                Some(bv) => bv.clone(),
                None => {
                    // Unconnected output: emit deterministic false vector of declared width.
                    AigBitVector::from_lsb_is_index_0(&vec![gb.get_false(); width])
                }
            };
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
    net_to_bv: &mut HashMap<NetIndex, AigBitVector>,
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
            let width = net
                .width
                .map(|(msb, lsb)| (msb as usize) - (lsb as usize) + 1)
                .unwrap_or(1);
            let bv = gb.add_input(net_name.to_string(), width);
            net_to_bv.insert(net_idx, bv.clone());
        }
    }
}

fn build_instance_input_map(
    inst: &crate::netlist::parse::NetlistInstance,
    pin_directions: &HashMap<&str, i32>,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    nets: &[Net],
    net_to_bv: &HashMap<NetIndex, AigBitVector>,
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
                crate::netlist::parse::NetRef::Simple(net_idx) => {
                    if let Some(bv) = net_to_bv.get(net_idx) {
                        assert_eq!(bv.get_bit_count(), 1);
                        input_map.insert(port_name.to_string(), *bv.get_lsb(0));
                    } else {
                        let net_name = interner
                            .resolve(nets[net_idx.0].name)
                            .unwrap_or("<unknown>");
                        missing_inputs.push(format!(
                            "{} (NetIndex({}), name='{}')",
                            port_name, net_idx.0, net_name
                        ));
                    }
                }
                crate::netlist::parse::NetRef::BitSelect(net_idx, bit) => {
                    if let Some(bv) = net_to_bv.get(net_idx) {
                        if (*bit as usize) >= bv.get_bit_count() {
                            missing_inputs.push(format!(
                                "{} (NetIndex({}), name='{}', bit={})",
                                port_name,
                                net_idx.0,
                                interner
                                    .resolve(nets[net_idx.0].name)
                                    .unwrap_or("<unknown>"),
                                bit
                            ));
                        } else {
                            input_map.insert(port_name.to_string(), *bv.get_lsb(*bit as usize));
                        }
                    } else {
                        let net_name = interner
                            .resolve(nets[net_idx.0].name)
                            .unwrap_or("<unknown>");
                        missing_inputs.push(format!(
                            "{} (NetIndex({}), name='{}', bit={})",
                            port_name, net_idx.0, net_name, bit
                        ));
                    }
                }
                crate::netlist::parse::NetRef::PartSelect(_, _, _) => {}
                crate::netlist::parse::NetRef::Literal(bits) => {
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
                crate::netlist::parse::NetRef::Unconnected => {
                    // Treat as a hard missing input and surface clearly.
                    missing_inputs.push(format!("{} (<unconnected>)", port_name));
                }
                crate::netlist::parse::NetRef::Concat(_) => {
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
    d_netref: &crate::netlist::parse::NetRef,
    gb: &mut GateBuilder,
    net_to_bv: &HashMap<NetIndex, AigBitVector>,
    invert: bool,
) -> Option<AigBitVector> {
    match d_netref {
        crate::netlist::parse::NetRef::Simple(d_netidx) => net_to_bv.get(d_netidx).map(|bv| {
            if invert {
                invert_bv(gb, bv)
            } else {
                bv.clone()
            }
        }),
        crate::netlist::parse::NetRef::BitSelect(d_netidx, bit) => {
            net_to_bv.get(d_netidx).map(|bv| {
                let mut bv1 = AigBitVector::from_lsb_is_index_0(&vec![gb.get_false(); 1]);
                let bit_op = *bv.get_lsb(*bit as usize);
                bv1.set_lsb(0, if invert { gb.add_not(bit_op) } else { bit_op });
                bv1
            })
        }
        crate::netlist::parse::NetRef::PartSelect(d_netidx, msb, lsb) => {
            net_to_bv.get(d_netidx).map(|bv| {
                let width = (*msb as usize) - (*lsb as usize) + 1;
                let mut bv_part = AigBitVector::from_lsb_is_index_0(&vec![gb.get_false(); width]);
                for i in 0..width {
                    let op = *bv.get_lsb(*lsb as usize + i);
                    bv_part.set_lsb(i, if invert { gb.add_not(op) } else { op });
                }
                bv_part
            })
        }
        crate::netlist::parse::NetRef::Literal(_) => None,
        crate::netlist::parse::NetRef::Unconnected => None,
        crate::netlist::parse::NetRef::Concat(_) => None,
    }
}

fn write_bv_to_port_destination(
    inst: &crate::netlist::parse::NetlistInstance,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    gb: &mut GateBuilder,
    net_to_bv: &mut HashMap<NetIndex, AigBitVector>,
    nets: &[Net],
    target_port_ci: &str,
    src_bv: &AigBitVector,
) {
    for (p, dst_ref) in &inst.connections {
        let pname = interner.resolve(*p).unwrap();
        if !pname.eq_ignore_ascii_case(target_port_ci) {
            continue;
        }
        match dst_ref {
            crate::netlist::parse::NetRef::Simple(q_netidx) => {
                let full_width = nets[q_netidx.0]
                    .width
                    .map(|(msb, lsb)| (msb as usize) - (lsb as usize) + 1)
                    .unwrap_or(1);
                assert_eq!(
                    src_bv.get_bit_count(),
                    full_width,
                    "DFF override: width mismatch assigning to simple; expected {} bits for net but got {}",
                    full_width,
                    src_bv.get_bit_count()
                );
                net_to_bv.insert(*q_netidx, src_bv.clone());
            }
            crate::netlist::parse::NetRef::BitSelect(q_netidx, bit) => {
                let full_width = nets[q_netidx.0]
                    .width
                    .map(|(msb, lsb)| (msb as usize) - (lsb as usize) + 1)
                    .unwrap_or(1);
                assert!((*bit as usize) < full_width);
                let mut bv = net_to_bv.remove(q_netidx).unwrap_or_else(|| {
                    AigBitVector::from_lsb_is_index_0(&vec![gb.get_false(); full_width])
                });
                if bv.get_bit_count() < full_width {
                    let mut new_bv =
                        AigBitVector::from_lsb_is_index_0(&vec![gb.get_false(); full_width]);
                    for i in 0..bv.get_bit_count() {
                        new_bv.set_lsb(i, *bv.get_lsb(i));
                    }
                    bv = new_bv;
                }
                bv.set_lsb(*bit as usize, *src_bv.get_lsb(0));
                net_to_bv.insert(*q_netidx, bv);
            }
            crate::netlist::parse::NetRef::PartSelect(q_netidx, msb, lsb) => {
                let slice_width = (*msb as usize) - (*lsb as usize) + 1;
                let full_width = nets[q_netidx.0]
                    .width
                    .map(|(w_msb, w_lsb)| (w_msb as usize) - (w_lsb as usize) + 1)
                    .unwrap_or(1);
                assert!(slice_width <= full_width);
                let mut bv = net_to_bv.remove(q_netidx).unwrap_or_else(|| {
                    AigBitVector::from_lsb_is_index_0(&vec![gb.get_false(); full_width])
                });
                if bv.get_bit_count() < full_width {
                    let mut new_bv =
                        AigBitVector::from_lsb_is_index_0(&vec![gb.get_false(); full_width]);
                    for i in 0..bv.get_bit_count() {
                        new_bv.set_lsb(i, *bv.get_lsb(i));
                    }
                    bv = new_bv;
                }
                for i in 0..slice_width {
                    bv.set_lsb((*lsb as usize) + i, *src_bv.get_lsb(i));
                }
                net_to_bv.insert(*q_netidx, bv);
            }
            crate::netlist::parse::NetRef::Literal(_) => {
                log::warn!(
                    "DFF override: destination output as literal not supported for port '{}'",
                    pname
                );
            }
            crate::netlist::parse::NetRef::Unconnected => {
                // Nothing to write.
            }
            crate::netlist::parse::NetRef::Concat(_) => {
                assert!(false, "concat destination not supported in DFF override");
            }
        }
    }
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
    net_to_bv: &mut HashMap<NetIndex, AigBitVector>,
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
Provide a more specific --dff_cells list or avoid formula-based DFF classification for this library.",
            type_name, inst_name, target_port
        ));
    }
    // Build D (optionally inverted) and write to destination port.
    if let Some(d_bv) = build_d_bv(d_input.unwrap(), gb, net_to_bv, invert) {
        write_bv_to_port_destination(inst, interner, gb, net_to_bv, nets, target_port, &d_bv);
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
    use crate::netlist::parse::{
        Net, NetRef, NetlistInstance, NetlistModule, NetlistPort, PortDirection,
    };
    use string_interner::{StringInterner, backend::StringBackend};

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
            ports,
            wires: vec![],
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
                    },
                    crate::liberty_proto::Pin {
                        name: "Y".to_string(),
                        direction: 1,
                        function: "(!A)".to_string(),
                        is_clocking_pin: false,
                    },
                ],
                area: 1.0,
                sequential: vec![],
            }],
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
            ports,
            wires: vec![],
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
                    },
                    crate::liberty_proto::Pin {
                        name: "Y".to_string(),
                        direction: 1,
                        function: "(!A)".to_string(),
                        is_clocking_pin: false,
                    },
                ],
                area: 1.0,
                sequential: vec![],
            }],
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
            ports,
            wires: vec![],
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
                    },
                    crate::liberty_proto::Pin {
                        name: "B".to_string(),
                        direction: 2,
                        function: "".to_string(),
                        is_clocking_pin: false,
                    },
                    crate::liberty_proto::Pin {
                        name: "Y".to_string(),
                        direction: 1,
                        function: "(A & B)".to_string(),
                        is_clocking_pin: false,
                    },
                ],
                area: 1.0,
                sequential: vec![],
            }],
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
            ports,
            wires: vec![],
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
                    },
                    crate::liberty_proto::Pin {
                        name: "Y".to_string(),
                        direction: 1,
                        function: "A".to_string(),
                        is_clocking_pin: false,
                    },
                ],
                area: 1.0,
                sequential: vec![],
            }],
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
            ports,
            wires: vec![],
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
                    },
                    crate::liberty_proto::Pin {
                        name: "Q".to_string(),
                        direction: 1,
                        function: "Q".to_string(),
                        is_clocking_pin: false,
                    },
                ],
                area: 1.0,
                sequential: vec![],
            }],
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
            ports,
            wires: vec![],
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
                    },
                    crate::liberty_proto::Pin {
                        name: "Q".to_string(),
                        direction: 1,
                        function: "Q".to_string(),
                        is_clocking_pin: false,
                    },
                ],
                area: 1.0,
                sequential: vec![],
            }],
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
            ports,
            wires: vec![],
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
                    },
                    crate::liberty_proto::Pin {
                        name: "q".to_string(),
                        direction: 1,
                        function: "d".to_string(),
                        is_clocking_pin: false,
                    },
                ],
                area: 1.0,
                sequential: vec![],
            }],
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
            ports,
            wires: vec![],
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
                    },
                    crate::liberty_proto::Pin {
                        name: "QN".to_string(),
                        direction: 1,
                        function: "IQN".to_string(),
                        is_clocking_pin: false,
                    },
                ],
                area: 1.0,
                sequential: vec![],
            }],
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
            ports,
            wires: vec![],
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
                    },
                    crate::liberty_proto::Pin {
                        name: "QN".to_string(),
                        direction: 1,
                        function: "IQN".to_string(),
                        is_clocking_pin: false,
                    },
                ],
                area: 1.0,
                sequential: vec![],
            }],
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
            ports,
            wires: vec![],
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
                    },
                    crate::liberty_proto::Pin {
                        name: "QN".to_string(),
                        direction: 1,
                        function: "IQN".to_string(),
                        is_clocking_pin: false,
                    },
                ],
                area: 1.0,
                sequential: vec![],
            }],
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
            ports,
            wires: vec![],
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
                    },
                    crate::liberty_proto::Pin {
                        name: "B".to_string(),
                        direction: 2,
                        function: "".to_string(),
                        is_clocking_pin: false,
                    },
                    crate::liberty_proto::Pin {
                        name: "Y".to_string(),
                        direction: 1,
                        function: "(A & B)".to_string(),
                        is_clocking_pin: false,
                    },
                ],
                area: 1.0,
                sequential: vec![],
            }],
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
