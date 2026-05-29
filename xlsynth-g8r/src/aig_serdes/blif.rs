// SPDX-License-Identifier: Apache-2.0

//! BLIF serialization for combinational and synchronous g8r designs.
//!
//! The emitted netlist uses ordinary `.names` combinational nodes and
//! rising-edge `.latch` statements. Port and register bit identities are
//! encoded in the net names, so no g8r-specific BLIF annotations are required.
//! Parsing targets this naming convention, while accepting uniform on-set or
//! off-set `.names` covers, including covers with more than two fanins.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use xlsynth::IrBits;

use crate::aig::sequential_gate::{
    canonical_register_d_name, canonical_register_q_name, canonical_transition_name,
    uniquify_transition_port_name,
};
use crate::aig::{
    AigBitVector, AigNode, AigOperand, ClockPort, GateFn, RegisterBinding, SequentialGateFn,
    TransitionInputId, TransitionOutputId,
};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};

#[derive(Debug)]
struct EmissionLayout {
    transition_inputs: Vec<BlifPort>,
    transition_outputs: Vec<BlifPort>,
    external_inputs: Vec<usize>,
    external_outputs: Vec<usize>,
    clock_net: Option<String>,
    registers: Vec<EmissionRegister>,
}

#[derive(Debug, Clone)]
struct BlifPort {
    name: String,
    nets: Vec<String>,
}

#[derive(Debug)]
struct EmissionRegister {
    name: String,
    q: usize,
    d: usize,
    initial_value: Option<Vec<bool>>,
}

#[derive(Debug)]
struct ParsedBlif {
    model: String,
    primary_inputs: Vec<String>,
    primary_outputs: Vec<String>,
    names: Vec<NamesTable>,
    latches: Vec<ParsedLatch>,
}

#[derive(Debug)]
struct NamesTable {
    inputs: Vec<String>,
    output: String,
    rows: Vec<Vec<String>>,
}

#[derive(Debug)]
struct ParsedCoverRow {
    pattern: String,
    output_value: bool,
}

#[derive(Debug)]
struct ParsedLatch {
    input: String,
    output: String,
    kind: Option<String>,
    control: Option<String>,
    initial: String,
}

#[derive(Debug)]
struct InferredClock {
    name: String,
    net: String,
}

#[derive(Debug)]
struct InferredRegister {
    name: String,
    q_nets: Vec<String>,
    d_nets: Vec<String>,
    initial_value: Option<IrBits>,
}

#[derive(Debug)]
struct InferredLayout {
    external_inputs: Vec<BlifPort>,
    external_outputs: Vec<BlifPort>,
    clock: Option<InferredClock>,
    registers: Vec<InferredRegister>,
}

/// Emits a synchronous design as Berkeley Logic Interchange Format.
///
/// Synchronous controls already present in the effective `D` transition logic
/// are emitted as combinational `.names` equations. A register bit is emitted
/// as `.latch <D> <Q> re <clock> <initial>`. A declared clock without
/// registers, or a zero-width port, cannot be represented without an
/// out-of-band annotation.
pub fn emit_blif(design: &SequentialGateFn) -> Result<String, String> {
    design
        .validate()
        .map_err(|e| format!("cannot emit BLIF for invalid SequentialGateFn: {}", e))?;
    if design.registers.is_empty() && design.clock.is_some() {
        return Err(format!(
            "cannot emit BLIF for design '{}': a clock without registers has no BLIF representation",
            design.name
        ));
    }
    let layout = make_emission_layout(design)?;
    let mut text = String::new();

    writeln!(&mut text, ".model {}", model_name(&design.name)?).unwrap();
    emit_net_list(&mut text, ".inputs", &primary_input_nets(&layout)?);
    emit_net_list(&mut text, ".outputs", &primary_output_nets(&layout)?);

    let mut node_nets = source_node_nets(design, &layout)?;
    for operand in design.transition.post_order_operands(true) {
        match design.transition.get(operand.node) {
            AigNode::Input { .. } => {}
            AigNode::Literal { value, .. } => {
                let output = logic_node_net(operand.node.id);
                writeln!(&mut text, ".names {}", output).unwrap();
                if *value {
                    writeln!(&mut text, "1").unwrap();
                }
                node_nets.insert(operand.node.id, output);
            }
            AigNode::And2 { a, b, .. } => {
                let a_net = operand_net(&node_nets, *a)?;
                let b_net = operand_net(&node_nets, *b)?;
                let output = logic_node_net(operand.node.id);
                writeln!(&mut text, ".names {} {} {}", a_net, b_net, output).unwrap();
                writeln!(
                    &mut text,
                    "{}{} 1",
                    if a.negated { '0' } else { '1' },
                    if b.negated { '0' } else { '1' }
                )
                .unwrap();
                node_nets.insert(operand.node.id, output);
            }
        }
    }

    for (port, output) in layout
        .transition_outputs
        .iter()
        .zip(design.transition.outputs.iter())
    {
        for (net, operand) in port.nets.iter().zip(output.bit_vector.iter_lsb_to_msb()) {
            let source = operand_net(&node_nets, *operand)?;
            writeln!(&mut text, ".names {} {}", source, net).unwrap();
            writeln!(&mut text, "{} 1", if operand.negated { '0' } else { '1' }).unwrap();
        }
    }

    for register in &layout.registers {
        let q_port = layout_input_port(&layout, register.q, "register Q")?;
        let d_port = layout_output_port(&layout, register.d, "register D")?;
        let clock_net = layout.clock_net.as_ref().ok_or_else(|| {
            format!(
                "register '{}' cannot be emitted without a clock",
                register.name
            )
        })?;
        for bit in 0..q_port.nets.len() {
            writeln!(
                &mut text,
                ".latch {} {} re {} {}",
                d_port.nets[bit],
                q_port.nets[bit],
                clock_net,
                format_initial_bit(register.initial_value.as_deref(), bit)
            )
            .unwrap();
        }
    }
    text.push_str(".end\n");
    Ok(text)
}

/// Emits a combinational gate function as BLIF.
pub fn emit_gate_fn_blif(gate_fn: &GateFn) -> Result<String, String> {
    emit_blif(&SequentialGateFn::from_gate_fn(gate_fn.clone()))
}

/// Parses BLIF emitted using the g8r flattened-net naming convention.
///
/// Transition interface ordering is normalized to external ports followed by
/// register ports. Register transition port names are reconstructed using the
/// same canonical convention as XLS block lowering. Compact `.latch D Q init`
/// records written by ABC are accepted by recovering the clock as the single
/// primary input that is not an encoded data port.
pub fn parse_blif(text: &str) -> Result<SequentialGateFn, String> {
    let parsed = parse_structural_blif(text)?;
    let design_name = parsed.model.clone();
    let layout = infer_layout(&parsed)?;
    let transition_name = if layout.registers.is_empty() {
        design_name.clone()
    } else {
        canonical_transition_name(&design_name)
    };
    let mut builder = GateBuilder::new(transition_name, GateBuilderOptions::no_opt());
    let mut net_to_operand = BTreeMap::new();
    let mut input_ids = Vec::with_capacity(layout.external_inputs.len());
    for port in &layout.external_inputs {
        let input_id = TransitionInputId::new(builder.inputs.len());
        let bits = builder.add_input(port.name.clone(), port.nets.len());
        for (net, operand) in port.nets.iter().zip(bits.iter_lsb_to_msb()) {
            insert_net_operand(&mut net_to_operand, net, *operand, "transition input")?;
        }
        input_ids.push(input_id);
    }
    let mut used_input_names = layout
        .external_inputs
        .iter()
        .map(|port| port.name.clone())
        .collect::<BTreeSet<String>>();
    let mut register_q_ids = Vec::with_capacity(layout.registers.len());
    for register in &layout.registers {
        let q_id = TransitionInputId::new(builder.inputs.len());
        let q_name = uniquify_transition_port_name(
            &canonical_register_q_name(&register.name),
            &mut used_input_names,
        );
        let bits = builder.add_input(q_name, register.q_nets.len());
        for (net, operand) in register.q_nets.iter().zip(bits.iter_lsb_to_msb()) {
            insert_net_operand(&mut net_to_operand, net, *operand, "register Q")?;
        }
        register_q_ids.push(q_id);
    }
    materialize_all_names(&parsed.names, &mut builder, &mut net_to_operand)?;

    let mut output_ids = Vec::with_capacity(layout.external_outputs.len());
    for port in &layout.external_outputs {
        let output_id = TransitionOutputId::new(builder.outputs.len());
        builder.add_output(
            port.name.clone(),
            operands_for_nets(&port.name, &port.nets, &net_to_operand)?,
        );
        output_ids.push(output_id);
    }
    let mut used_output_names = layout
        .external_outputs
        .iter()
        .map(|port| port.name.clone())
        .collect::<BTreeSet<String>>();
    let mut registers = Vec::with_capacity(layout.registers.len());
    for (register, q_id) in layout.registers.iter().zip(register_q_ids) {
        let d_id = TransitionOutputId::new(builder.outputs.len());
        let d_name = uniquify_transition_port_name(
            &canonical_register_d_name(&register.name),
            &mut used_output_names,
        );
        builder.add_output(
            d_name,
            operands_for_nets(&register.name, &register.d_nets, &net_to_operand)?,
        );
        registers.push(RegisterBinding {
            name: register.name.clone(),
            q: q_id,
            d: d_id,
            initial_value: register.initial_value.clone(),
        });
    }
    let transition = finish_builder(builder);
    let clock = layout.clock.map(|clock| ClockPort { name: clock.name });
    SequentialGateFn::new(
        design_name,
        transition,
        input_ids,
        output_ids,
        clock,
        registers,
    )
    .map_err(|e| format!("invalid SequentialGateFn reconstructed from BLIF: {}", e))
}

/// Parses a combinational BLIF design as a [`GateFn`].
pub fn parse_gate_fn_blif(text: &str) -> Result<GateFn, String> {
    parse_blif(text)?.try_into_gate_fn()
}

/// Reads and parses a BLIF design from disk.
pub fn load_blif_from_path(path: &Path) -> Result<SequentialGateFn, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    parse_blif(&text).map_err(|e| format!("failed to parse {}: {}", path.display(), e))
}

/// Constructs the deterministic physical-net layout used for BLIF emission.
fn make_emission_layout(design: &SequentialGateFn) -> Result<EmissionLayout, String> {
    let mut transition_inputs = vec![None; design.transition.inputs.len()];
    for (ordinal, id) in design.inputs.iter().enumerate() {
        let input = &design.transition.inputs[id.index()];
        reject_zero_width_port(&input.name, input.get_bit_count(), "input")?;
        transition_inputs[id.index()] = Some(BlifPort {
            name: input.name.clone(),
            nets: (0..input.get_bit_count())
                .map(|bit| external_input_net(input.name.as_str(), ordinal, bit))
                .collect(),
        });
    }

    let mut transition_outputs = vec![None; design.transition.outputs.len()];
    for (ordinal, id) in design.outputs.iter().enumerate() {
        let output = &design.transition.outputs[id.index()];
        reject_zero_width_port(&output.name, output.get_bit_count(), "output")?;
        transition_outputs[id.index()] = Some(BlifPort {
            name: output.name.clone(),
            nets: (0..output.get_bit_count())
                .map(|bit| external_output_net(output.name.as_str(), ordinal, bit))
                .collect(),
        });
    }

    let mut registers = Vec::with_capacity(design.registers.len());
    for register in &design.registers {
        let q = &design.transition.inputs[register.q.index()];
        let d = &design.transition.outputs[register.d.index()];
        reject_zero_width_port(&register.name, q.get_bit_count(), "register")?;
        transition_inputs[register.q.index()] = Some(BlifPort {
            name: q.name.clone(),
            nets: (0..q.get_bit_count())
                .map(|bit| register_q_net(register.name.as_str(), bit))
                .collect(),
        });
        transition_outputs[register.d.index()] = Some(BlifPort {
            name: d.name.clone(),
            nets: (0..d.get_bit_count())
                .map(|bit| register_d_net(register.name.as_str(), bit))
                .collect(),
        });
        registers.push(EmissionRegister {
            name: register.name.clone(),
            q: register.q.index(),
            d: register.d.index(),
            initial_value: register.initial_value.as_ref().map(|bits| {
                (0..bits.get_bit_count())
                    .map(|bit| bits.get_bit(bit).expect("bit index is in range"))
                    .collect()
            }),
        });
    }

    let layout = EmissionLayout {
        transition_inputs: collect_ports(transition_inputs, "input")?,
        transition_outputs: collect_ports(transition_outputs, "output")?,
        external_inputs: design.inputs.iter().map(|id| id.index()).collect(),
        external_outputs: design.outputs.iter().map(|id| id.index()).collect(),
        clock_net: design
            .clock
            .as_ref()
            .map(|clock| clock_net(&clock.name))
            .transpose()?,
        registers,
    };
    check_layout_nets(&layout)?;
    Ok(layout)
}

fn reject_zero_width_port(name: &str, bit_count: usize, kind: &str) -> Result<(), String> {
    if bit_count == 0 {
        return Err(format!(
            "cannot emit zero-width {} '{}': plain BLIF has no net with which to recover it",
            kind, name
        ));
    }
    Ok(())
}

fn collect_ports(ports: Vec<Option<BlifPort>>, kind: &str) -> Result<Vec<BlifPort>, String> {
    ports
        .into_iter()
        .enumerate()
        .map(|(index, port)| {
            port.ok_or_else(|| {
                format!(
                    "transition {} index {} is not bound in SequentialGateFn",
                    kind, index
                )
            })
        })
        .collect()
}

fn check_layout_nets(layout: &EmissionLayout) -> Result<(), String> {
    let mut nets = BTreeSet::new();
    for port in layout
        .transition_inputs
        .iter()
        .chain(layout.transition_outputs.iter())
    {
        for net in &port.nets {
            if !nets.insert(net.clone()) {
                return Err(format!(
                    "BLIF interface net '{}' is generated more than once",
                    net
                ));
            }
        }
    }
    if let Some(clock_net) = &layout.clock_net {
        if clock_net.starts_with("$g8r$node") {
            return Err(format!(
                "BLIF clock net '{}' uses the reserved internal-node prefix '$g8r$node'",
                clock_net
            ));
        }
        if !nets.insert(clock_net.clone()) {
            return Err(format!(
                "BLIF clock net '{}' conflicts with an interface net",
                clock_net
            ));
        }
    }
    Ok(())
}

fn source_node_nets(
    design: &SequentialGateFn,
    layout: &EmissionLayout,
) -> Result<BTreeMap<usize, String>, String> {
    let mut node_nets = BTreeMap::new();
    for (input, port) in design
        .transition
        .inputs
        .iter()
        .zip(layout.transition_inputs.iter())
    {
        for (operand, net) in input.bit_vector.iter_lsb_to_msb().zip(port.nets.iter()) {
            if operand.negated {
                return Err(format!(
                    "transition input '{}' has a negated source bit, which cannot be emitted as a BLIF input",
                    input.name
                ));
            }
            if !matches!(design.transition.get(operand.node), AigNode::Input { .. }) {
                return Err(format!(
                    "transition input '{}' bit references non-input AIG node %{}",
                    input.name, operand.node.id
                ));
            }
            if node_nets.insert(operand.node.id, net.clone()).is_some() {
                return Err(format!(
                    "AIG input node %{} is bound to multiple BLIF interface nets",
                    operand.node.id
                ));
            }
        }
    }
    Ok(node_nets)
}

fn operand_net(node_nets: &BTreeMap<usize, String>, operand: AigOperand) -> Result<&str, String> {
    node_nets
        .get(&operand.node.id)
        .map(String::as_str)
        .ok_or_else(|| format!("missing BLIF net for live AIG node %{}", operand.node.id))
}

/// Parses the BLIF directives used by the synchronous format.
fn parse_structural_blif(text: &str) -> Result<ParsedBlif, String> {
    let mut model = None;
    let mut primary_inputs = Vec::new();
    let mut primary_outputs = Vec::new();
    let mut names: Vec<NamesTable> = Vec::new();
    let mut latches = Vec::new();
    let mut current_names: Option<usize> = None;
    let mut saw_end = false;

    for (line_number, raw_line) in logical_blif_lines(text)? {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if saw_end {
            return Err(format!(
                "unexpected BLIF content after .end at line {}",
                line_number
            ));
        }
        if !line.starts_with('.') {
            let table_index = current_names.ok_or_else(|| {
                format!(
                    "unexpected BLIF cover row without preceding .names at line {}",
                    line_number
                )
            })?;
            names[table_index]
                .rows
                .push(line.split_whitespace().map(str::to_string).collect());
            continue;
        }
        current_names = None;
        let fields = line.split_whitespace().collect::<Vec<&str>>();
        match fields[0] {
            ".model" => {
                if fields.len() != 2 {
                    return Err(format!("invalid .model directive at line {}", line_number));
                }
                if model.replace(fields[1].to_string()).is_some() {
                    return Err("BLIF input contains multiple .model directives".to_string());
                }
            }
            ".inputs" => primary_inputs.extend(fields[1..].iter().map(|net| net.to_string())),
            ".outputs" => primary_outputs.extend(fields[1..].iter().map(|net| net.to_string())),
            ".names" => {
                if fields.len() < 2 {
                    return Err(format!(
                        ".names is missing an output at line {}",
                        line_number
                    ));
                }
                let output = fields
                    .last()
                    .expect(".names has at least an output")
                    .to_string();
                names.push(NamesTable {
                    inputs: fields[1..fields.len() - 1]
                        .iter()
                        .map(|net| net.to_string())
                        .collect(),
                    output,
                    rows: Vec::new(),
                });
                current_names = Some(names.len() - 1);
            }
            ".latch" => {
                let latch = match fields.len() {
                    4 => ParsedLatch {
                        input: fields[1].to_string(),
                        output: fields[2].to_string(),
                        kind: None,
                        control: None,
                        initial: fields[3].to_string(),
                    },
                    6 => ParsedLatch {
                        input: fields[1].to_string(),
                        output: fields[2].to_string(),
                        kind: Some(fields[3].to_string()),
                        control: Some(fields[4].to_string()),
                        initial: fields[5].to_string(),
                    },
                    _ => {
                        return Err(format!(
                            "only '.latch input output initial' and '.latch input output re clock initial' are supported at line {}",
                            line_number
                        ));
                    }
                };
                latches.push(latch);
            }
            ".end" => {
                if fields.len() != 1 {
                    return Err(format!("invalid .end directive at line {}", line_number));
                }
                saw_end = true;
            }
            other => {
                return Err(format!(
                    "unsupported BLIF directive '{}' at line {}",
                    other, line_number
                ));
            }
        }
    }
    if !saw_end {
        return Err("BLIF input is missing .end".to_string());
    }
    Ok(ParsedBlif {
        model: model.ok_or_else(|| "BLIF input is missing .model".to_string())?,
        primary_inputs,
        primary_outputs,
        names,
        latches,
    })
}

/// Coalesces standard BLIF backslash continuations before parsing directives.
fn logical_blif_lines(text: &str) -> Result<Vec<(usize, String)>, String> {
    let mut result = Vec::new();
    let mut pending: Option<(usize, String)> = None;
    for (line_index, raw_line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let non_comment = raw_line
            .split_once('#')
            .map_or(raw_line, |(before_comment, _)| before_comment);
        let trimmed_end = non_comment.trim_end();
        let continued = trimmed_end.ends_with('\\');
        let fragment = if continued {
            trimmed_end
                .strip_suffix('\\')
                .expect("continuation marker was found")
        } else {
            non_comment
        };
        if let Some((_, pending_text)) = &mut pending {
            pending_text.push(' ');
            pending_text.push_str(fragment.trim());
        } else if continued {
            pending = Some((line_number, fragment.trim_end().to_string()));
        } else {
            result.push((line_number, non_comment.to_string()));
        }
        if !continued && let Some(line) = pending.take() {
            result.push(line);
        }
    }
    if let Some((line_number, _)) = pending {
        return Err(format!(
            "unterminated BLIF line continuation starting at line {}",
            line_number
        ));
    }
    Ok(result)
}

/// Infers the flattened g8r interface represented by BLIF net names.
fn infer_layout(parsed: &ParsedBlif) -> Result<InferredLayout, String> {
    let clock = infer_clock(parsed)?;
    let external_input_nets = parsed
        .primary_inputs
        .iter()
        .filter(|net| clock.as_ref().is_none_or(|clock| *net != &clock.net))
        .cloned()
        .collect::<Vec<String>>();
    let external_inputs = infer_external_ports(&external_input_nets, "_input", "input")?;
    let external_outputs = infer_external_ports(&parsed.primary_outputs, "_output", "output")?;
    let registers = infer_registers(&parsed.latches)?;
    Ok(InferredLayout {
        external_inputs,
        external_outputs,
        clock,
        registers,
    })
}

fn infer_clock(parsed: &ParsedBlif) -> Result<Option<InferredClock>, String> {
    let Some(first_latch) = parsed.latches.first() else {
        return Ok(None);
    };
    let possible_clock_nets = parsed
        .primary_inputs
        .iter()
        .filter(|net| parse_port_net(net, "_input", "input").is_err())
        .collect::<Vec<&String>>();
    if possible_clock_nets.len() != 1 {
        return Err(format!(
            "BLIF with latches must have exactly one non-data primary input for its clock; found {:?}",
            possible_clock_nets
        ));
    }
    let clock_net = possible_clock_nets[0];
    let has_explicit_control = first_latch.control.is_some();
    for latch in &parsed.latches {
        if latch.control.is_some() != has_explicit_control {
            return Err("BLIF mixes explicit-control and compact .latch forms".to_string());
        }
        if let Some(kind) = &latch.kind
            && kind != "re"
        {
            return Err(format!(
                "BLIF latch output '{}' uses '{}'; only rising-edge 're' latches are supported",
                latch.output, kind
            ));
        }
        if let Some(control) = &latch.control
            && control != clock_net
        {
            return Err(format!(
                "BLIF latch clock '{}' does not match inferred clock primary input '{}'",
                control, clock_net
            ));
        }
    }
    let occurrences = parsed
        .primary_inputs
        .iter()
        .filter(|net| *net == clock_net)
        .count();
    if occurrences != 1 {
        return Err(format!(
            "BLIF latch clock net '{}' must occur exactly once in .inputs; found {} occurrence(s)",
            clock_net, occurrences
        ));
    }
    Ok(Some(InferredClock {
        name: clock_net.to_string(),
        net: clock_net.to_string(),
    }))
}

fn infer_external_ports(
    nets: &[String],
    marker: &str,
    kind: &str,
) -> Result<Vec<BlifPort>, String> {
    let mut groups: BTreeMap<usize, (String, BTreeMap<usize, String>)> = BTreeMap::new();
    for net in nets {
        let (name, port_index, bit_index) = parse_port_net(net, marker, kind)?;
        let entry = groups
            .entry(port_index)
            .or_insert_with(|| (name.clone(), BTreeMap::new()));
        if entry.0 != name {
            return Err(format!(
                "BLIF {} port index {} is associated with names '{}' and '{}'",
                kind, port_index, entry.0, name
            ));
        }
        if entry.1.insert(bit_index, net.clone()).is_some() {
            return Err(format!(
                "BLIF {} port '{}' contains duplicate bit {}",
                kind, name, bit_index
            ));
        }
    }
    groups
        .into_iter()
        .enumerate()
        .map(|(expected_index, (port_index, (name, bits)))| {
            if port_index != expected_index {
                return Err(format!(
                    "BLIF {} port indices are not contiguous: expected {}, found {}",
                    kind, expected_index, port_index
                ));
            }
            Ok(BlifPort {
                name,
                nets: contiguous_bits(bits, kind)?,
            })
        })
        .collect()
}

fn parse_port_net(net: &str, marker: &str, kind: &str) -> Result<(String, usize, usize), String> {
    let (stem, bit_index) = parse_bit_suffix(net).map_err(|e| {
        format!(
            "BLIF primary {} net '{}' does not match the g8r flattened-net convention: {}",
            kind, net, e
        )
    })?;
    let (encoded_name, port_index) = stem.rsplit_once(marker).ok_or_else(|| {
        format!(
            "BLIF primary {} net '{}' does not match '<name>{}<port>[<bit>]'",
            kind, net, marker
        )
    })?;
    let port_index = port_index.parse::<usize>().map_err(|_| {
        format!(
            "BLIF primary {} net '{}' has invalid port index '{}'",
            kind, net, port_index
        )
    })?;
    let name = decode_component(encoded_name)
        .map_err(|e| format!("cannot decode BLIF {} port net '{}': {}", kind, net, e))?;
    Ok((name, port_index, bit_index))
}

fn infer_registers(latches: &[ParsedLatch]) -> Result<Vec<InferredRegister>, String> {
    let mut ordered_names = Vec::new();
    let mut groups: BTreeMap<String, BTreeMap<usize, &ParsedLatch>> = BTreeMap::new();
    for latch in latches {
        let (name, bit_index) = parse_register_q_net(&latch.output)?;
        if !groups.contains_key(&name) {
            ordered_names.push(name.clone());
        }
        if groups
            .entry(name.clone())
            .or_default()
            .insert(bit_index, latch)
            .is_some()
        {
            return Err(format!(
                "BLIF register '{}' contains duplicate Q bit {}",
                name, bit_index
            ));
        }
    }
    let mut result = Vec::with_capacity(ordered_names.len());
    for name in ordered_names {
        let bits = groups
            .remove(&name)
            .expect("ordered register names were inserted into groups");
        let expected_width = bits.len();
        let mut q_nets = Vec::with_capacity(expected_width);
        let mut d_nets = Vec::with_capacity(expected_width);
        let mut initial_bits = Vec::with_capacity(expected_width);
        let mut initialized = None;
        for (expected_bit, (bit, latch)) in bits.into_iter().enumerate() {
            if bit != expected_bit {
                return Err(format!(
                    "BLIF register '{}' bit indices are not contiguous: expected {}, found {}",
                    name, expected_bit, bit
                ));
            }
            q_nets.push(latch.output.clone());
            d_nets.push(latch.input.clone());
            let bit_value = match latch.initial.as_str() {
                "0" => Some(false),
                "1" => Some(true),
                "2" => None,
                other => {
                    return Err(format!(
                        "BLIF register '{}' bit {} has unsupported initial value '{}'; expected 0, 1, or 2",
                        name, bit, other
                    ));
                }
            };
            match (initialized, bit_value) {
                (None, Some(value)) if initial_bits.is_empty() => {
                    initialized = Some(true);
                    initial_bits.push(value);
                }
                (None, None) if initial_bits.is_empty() => {
                    initialized = Some(false);
                }
                (Some(true), Some(value)) => initial_bits.push(value),
                (Some(false), None) => {}
                _ => {
                    return Err(format!(
                        "BLIF register '{}' mixes specified and unspecified initial bits",
                        name
                    ));
                }
            }
        }
        result.push(InferredRegister {
            name,
            q_nets,
            d_nets,
            initial_value: initialized.and_then(|is_initialized| {
                is_initialized.then(|| IrBits::from_lsb_is_0(&initial_bits))
            }),
        });
    }
    Ok(result)
}

fn parse_register_q_net(net: &str) -> Result<(String, usize), String> {
    let (stem, bit_index) = parse_bit_suffix(net).map_err(|e| {
        format!(
            "BLIF latch Q net '{}' does not match the g8r register convention: {}",
            net, e
        )
    })?;
    let encoded_name = stem.strip_suffix("_reg").ok_or_else(|| {
        format!(
            "BLIF latch Q net '{}' does not match '<register>_reg[<bit>]'",
            net
        )
    })?;
    let name = decode_component(encoded_name)
        .map_err(|e| format!("cannot decode BLIF register Q net '{}': {}", net, e))?;
    Ok((name, bit_index))
}

fn parse_bit_suffix(net: &str) -> Result<(&str, usize), String> {
    let (stem, suffix) = net
        .rsplit_once('[')
        .ok_or_else(|| "missing '[<bit>]' suffix".to_string())?;
    let bit = suffix
        .strip_suffix(']')
        .ok_or_else(|| "missing closing ']' in bit suffix".to_string())?;
    let bit_index = bit
        .parse::<usize>()
        .map_err(|_| format!("invalid bit index '{}'", bit))?;
    Ok((stem, bit_index))
}

fn contiguous_bits(bits: BTreeMap<usize, String>, kind: &str) -> Result<Vec<String>, String> {
    bits.into_iter()
        .enumerate()
        .map(|(expected_bit, (bit, net))| {
            if bit != expected_bit {
                return Err(format!(
                    "BLIF {} bit indices are not contiguous: expected {}, found {}",
                    kind, expected_bit, bit
                ));
            }
            Ok(net)
        })
        .collect()
}

/// Materializes `.names` truth tables once all of their driving nets exist.
fn materialize_all_names(
    tables: &[NamesTable],
    builder: &mut GateBuilder,
    net_to_operand: &mut BTreeMap<String, AigOperand>,
) -> Result<(), String> {
    let mut definitions = net_to_operand.keys().cloned().collect::<BTreeSet<String>>();
    for table in tables {
        if !definitions.insert(table.output.clone()) {
            return Err(format!(
                "BLIF net '{}' is defined more than once",
                table.output
            ));
        }
    }
    let mut completed = vec![false; tables.len()];
    let mut remaining = tables.len();
    while remaining != 0 {
        let mut progressed = false;
        for (index, table) in tables.iter().enumerate() {
            if completed[index]
                || table
                    .inputs
                    .iter()
                    .any(|net| !net_to_operand.contains_key(net))
            {
                continue;
            }
            materialize_names(table, builder, net_to_operand)?;
            completed[index] = true;
            remaining -= 1;
            progressed = true;
        }
        if !progressed {
            let table = tables
                .iter()
                .enumerate()
                .find_map(|(index, table)| (!completed[index]).then_some(table))
                .expect("remaining count means there is an incomplete table");
            let unresolved = table
                .inputs
                .iter()
                .filter(|net| !net_to_operand.contains_key(*net))
                .cloned()
                .collect::<Vec<String>>();
            return Err(format!(
                "cannot materialize BLIF net '{}'; unresolved inputs {:?} form a cycle or are undefined",
                table.output, unresolved
            ));
        }
    }
    Ok(())
}

/// Materializes one uniform on-set or off-set truth table into AIG operations.
fn materialize_names(
    table: &NamesTable,
    builder: &mut GateBuilder,
    net_to_operand: &mut BTreeMap<String, AigOperand>,
) -> Result<(), String> {
    let inputs = table
        .inputs
        .iter()
        .map(|net| {
            net_to_operand
                .get(net)
                .copied()
                .ok_or_else(|| format!(".names references undefined input net '{}'", net))
        })
        .collect::<Result<Vec<AigOperand>, String>>()?;
    let mut cubes = Vec::with_capacity(table.rows.len());
    let mut cover_value = None;
    for row in &table.rows {
        let parsed_row = parse_cover_row(table, row)?;
        if let Some(previous) = cover_value
            && previous != parsed_row.output_value
        {
            return Err(format!(
                ".names output '{}' mixes on-set and off-set rows, which is not supported",
                table.output
            ));
        }
        cover_value = Some(parsed_row.output_value);
        let terms = parsed_row
            .pattern
            .chars()
            .enumerate()
            .filter_map(|(index, value)| match value {
                '-' => None,
                '0' => Some(builder.add_not(inputs[index])),
                '1' => Some(inputs[index]),
                _ => unreachable!("parse_cover_row validates cube characters"),
            })
            .collect::<Vec<AigOperand>>();
        cubes.push(and_operands(builder, &terms));
    }
    let cover = or_operands(builder, &cubes);
    let operand = if cover_value == Some(false) {
        builder.add_not(cover)
    } else {
        cover
    };
    insert_net_operand(net_to_operand, &table.output, operand, ".names output")
}

fn parse_cover_row(table: &NamesTable, row: &[String]) -> Result<ParsedCoverRow, String> {
    if table.inputs.is_empty() {
        if row == ["0"] || row == ["1"] {
            return Ok(ParsedCoverRow {
                pattern: String::new(),
                output_value: row[0] == "1",
            });
        }
        return Err(format!(
            "constant .names output '{}' has unsupported row {:?}; expected '0' or '1'",
            table.output, row
        ));
    }
    if row.len() != 2 || !matches!(row[1].as_str(), "0" | "1") {
        return Err(format!(
            ".names output '{}' uses unsupported row {:?}; rows must end in '0' or '1'",
            table.output, row
        ));
    }
    if row[0].chars().count() != table.inputs.len()
        || row[0]
            .chars()
            .any(|value| !matches!(value, '0' | '1' | '-'))
    {
        return Err(format!(
            ".names output '{}' has invalid cube '{}' for {} input(s)",
            table.output,
            row[0],
            table.inputs.len()
        ));
    }
    Ok(ParsedCoverRow {
        pattern: row[0].clone(),
        output_value: row[1] == "1",
    })
}

fn and_operands(builder: &mut GateBuilder, operands: &[AigOperand]) -> AigOperand {
    let Some((first, rest)) = operands.split_first() else {
        return builder.get_true();
    };
    rest.iter().fold(*first, |result, operand| {
        builder.add_and_binary(result, *operand)
    })
}

fn or_operands(builder: &mut GateBuilder, operands: &[AigOperand]) -> AigOperand {
    let Some((first, rest)) = operands.split_first() else {
        return builder.get_false();
    };
    rest.iter().fold(*first, |result, operand| {
        builder.add_or_binary(result, *operand)
    })
}

fn operands_for_nets(
    name: &str,
    nets: &[String],
    net_to_operand: &BTreeMap<String, AigOperand>,
) -> Result<AigBitVector, String> {
    let operands = nets
        .iter()
        .map(|net| {
            net_to_operand.get(net).copied().ok_or_else(|| {
                format!(
                    "transition output '{}' references undefined BLIF net '{}'",
                    name, net
                )
            })
        })
        .collect::<Result<Vec<AigOperand>, String>>()?;
    Ok(AigBitVector::from_lsb_is_index_0(&operands))
}

fn finish_builder(builder: GateBuilder) -> GateFn {
    if builder.outputs.is_empty() {
        GateFn {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            gates: builder.gates,
        }
    } else {
        builder.build()
    }
}

fn insert_net_operand(
    net_to_operand: &mut BTreeMap<String, AigOperand>,
    net: &str,
    operand: AigOperand,
    kind: &str,
) -> Result<(), String> {
    if net_to_operand.insert(net.to_string(), operand).is_some() {
        return Err(format!("{} net '{}' is defined more than once", kind, net));
    }
    Ok(())
}

fn primary_input_nets(layout: &EmissionLayout) -> Result<Vec<String>, String> {
    let mut nets = Vec::new();
    for index in &layout.external_inputs {
        nets.extend(
            layout_input_port(layout, *index, "external input")?
                .nets
                .clone(),
        );
    }
    if let Some(clock_net) = &layout.clock_net {
        nets.push(clock_net.clone());
    }
    Ok(nets)
}

fn primary_output_nets(layout: &EmissionLayout) -> Result<Vec<String>, String> {
    let mut nets = Vec::new();
    for index in &layout.external_outputs {
        nets.extend(
            layout_output_port(layout, *index, "external output")?
                .nets
                .clone(),
        );
    }
    Ok(nets)
}

fn layout_input_port<'a>(
    layout: &'a EmissionLayout,
    index: usize,
    kind: &str,
) -> Result<&'a BlifPort, String> {
    layout
        .transition_inputs
        .get(index)
        .ok_or_else(|| format!("{} index {} is out of bounds", kind, index))
}

fn layout_output_port<'a>(
    layout: &'a EmissionLayout,
    index: usize,
    kind: &str,
) -> Result<&'a BlifPort, String> {
    layout
        .transition_outputs
        .get(index)
        .ok_or_else(|| format!("{} index {} is out of bounds", kind, index))
}

fn emit_net_list(text: &mut String, directive: &str, nets: &[String]) {
    text.push_str(directive);
    for net in nets {
        write!(text, " {}", net).unwrap();
    }
    text.push('\n');
}

fn format_initial_bit(initial_value: Option<&[bool]>, bit: usize) -> char {
    match initial_value {
        None => '2',
        Some(value) if value[bit] => '1',
        Some(_) => '0',
    }
}

fn model_name(name: &str) -> Result<String, String> {
    verbatim_token(name, "design name")
}

fn external_input_net(name: &str, index: usize, bit: usize) -> String {
    format!("{}_input{}[{}]", escape_component(name), index, bit)
}

fn external_output_net(name: &str, index: usize, bit: usize) -> String {
    format!("{}_output{}[{}]", escape_component(name), index, bit)
}

fn register_q_net(name: &str, bit: usize) -> String {
    format!("{}_reg[{}]", escape_component(name), bit)
}

fn register_d_net(name: &str, bit: usize) -> String {
    format!("{}_next[{}]", escape_component(name), bit)
}

fn clock_net(name: &str) -> Result<String, String> {
    verbatim_token(name, "clock port name")
}

fn verbatim_token(name: &str, kind: &str) -> Result<String, String> {
    if name.is_empty()
        || name
            .bytes()
            .any(|byte| byte.is_ascii_whitespace() || matches!(byte, b'#' | b'\\'))
    {
        return Err(format!(
            "{} '{}' cannot be emitted verbatim as a BLIF token",
            kind, name
        ));
    }
    Ok(name.to_string())
}

fn logic_node_net(node_id: usize) -> String {
    format!("$g8r$node{}", node_id)
}

/// Encodes one user-visible identifier component without using BLIF whitespace.
///
/// `_` is escaped as well as punctuation so the reserved `_input`, `_output`,
/// `_reg`, and `_next` separators remain unambiguous during import.
fn escape_component(name: &str) -> String {
    if name.is_empty() {
        return "_e".to_string();
    }
    let mut escaped = String::new();
    for byte in name.bytes() {
        if byte.is_ascii_alphanumeric() || byte == b'$' {
            escaped.push(byte as char);
        } else {
            write!(&mut escaped, "_x{:02x}", byte).unwrap();
        }
    }
    escaped
}

fn decode_component(component: &str) -> Result<String, String> {
    if component == "_e" {
        return Ok(String::new());
    }
    let bytes = component.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'_'
            && index + 3 < bytes.len()
            && bytes[index + 1] == b'x'
            && bytes[index + 2].is_ascii_hexdigit()
            && bytes[index + 3].is_ascii_hexdigit()
        {
            let value = u8::from_str_radix(&component[index + 2..index + 4], 16)
                .expect("hex digit check guarantees a valid byte");
            decoded.push(value);
            index += 4;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded)
        .map_err(|e| format!("escaped component is not valid UTF-8 after decoding: {}", e))
}
