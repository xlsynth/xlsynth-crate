// SPDX-License-Identifier: Apache-2.0

//! Text and binary serialization for native g8r designs.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use xlsynth_pir::{IrBits, IrFormatPreference, IrValue};

use crate::aig::{
    ClockPort, GateFn, RegisterBinding, RegisterControl, SequentialGateFn, SynchronousReset,
    TransitionInputId, TransitionOutputId,
};
use crate::aig_serdes::gate_parser::parse_gate_fn;

const TEXT_HEADER: &str = "g8r_v4";
const BINARY_HEADER: &[u8] = b"g8rbin_v4\n";

/// Emits either explicit-control or legalized designs as canonical `.g8r` text.
pub fn emit_g8r(design: &SequentialGateFn) -> Result<String, String> {
    design.validate()?;
    Ok(emit_sequential_g8r(design))
}

/// Parses a human-readable `.g8r` design.
pub fn parse_g8r(text: &str) -> Result<SequentialGateFn, String> {
    let (header, body) = text
        .split_once('\n')
        .ok_or_else(|| "g8r file is missing its header".to_string())?;
    if header != TEXT_HEADER {
        return Err(format!(
            "unsupported g8r header '{}'; expected '{}'",
            header, TEXT_HEADER
        ));
    }
    parse_sequential_g8r(body)
}

/// Serializes a native design as `.g8rbin`, preserving explicit controls.
///
/// The ASCII prefix lets tools identify the native format before decoding its
/// bincode body.
pub fn encode_g8r_binary(design: &SequentialGateFn) -> Result<Vec<u8>, String> {
    design.validate()?;
    let body = bincode::serialize(&BinarySequentialGateFn::from(design))
        .map_err(|e| format!("failed to serialize SequentialGateFn payload: {}", e))?;
    let mut bytes = Vec::with_capacity(BINARY_HEADER.len() + body.len());
    bytes.extend_from_slice(BINARY_HEADER);
    bytes.extend_from_slice(&body);
    Ok(bytes)
}

/// Decodes a `.g8rbin` design after inspecting its format prefix.
pub fn decode_g8r_binary(bytes: &[u8]) -> Result<SequentialGateFn, String> {
    let body = bytes
        .strip_prefix(BINARY_HEADER)
        .ok_or_else(|| "unsupported g8rbin header; expected 'g8rbin_v4'".to_string())?;
    let binary: BinarySequentialGateFn = bincode::deserialize(body)
        .map_err(|e| format!("failed to deserialize SequentialGateFn payload: {}", e))?;
    binary.try_into()
}

/// Loads a `.g8r` or `.g8rbin` sequential design from a path.
pub fn load_sequential_gate_fn_from_path(path: &Path) -> Result<SequentialGateFn, String> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("g8rbin") => {
            let bytes =
                fs::read(path).map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
            decode_g8r_binary(&bytes)
                .map_err(|e| format!("failed to decode {}: {}", path.display(), e))
        }
        Some("g8r") => {
            let text = fs::read_to_string(path)
                .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
            parse_g8r(&text).map_err(|e| format!("failed to parse {}: {}", path.display(), e))
        }
        _ => Err(format!(
            "expected a .g8r or .g8rbin input, got {}",
            path.display()
        )),
    }
}

/// Loads a native design for a consumer that operates only on combinational
/// `GateFn`s.
pub fn load_gate_fn_from_path(path: &Path) -> Result<GateFn, String> {
    load_sequential_gate_fn_from_path(path)?
        .try_into_gate_fn()
        .map_err(|e| format!("{}: {}", path.display(), e))
}

fn emit_sequential_g8r(sequential: &SequentialGateFn) -> String {
    let mut text = String::new();
    text.push_str(TEXT_HEADER);
    text.push('\n');
    text.push_str(&format!("name {}\n", sequential.name));
    match &sequential.clock {
        None => text.push_str("clock none\n"),
        Some(clock) => text.push_str(&format!("clock {}\n", clock.name)),
    }
    text.push_str(&format!(
        "inputs {}\n",
        format_input_ids(&sequential.inputs)
    ));
    text.push_str(&format!(
        "outputs {}\n",
        format_output_ids(&sequential.outputs)
    ));
    text.push_str(&format!("registers {}\n", sequential.registers.len()));
    for register in &sequential.registers {
        text.push_str(&format!(
            "register {} q={} d={}",
            register.name,
            register.q.index(),
            register.d.index()
        ));
        if let Some(enable) = register.enable {
            text.push_str(&format!(
                " enable={} enable_active_low={}",
                enable.signal.index(),
                enable.active_low
            ));
        }
        if let Some(reset) = &register.reset {
            text.push_str(&format!(
                " reset={} reset_active_low={} reset_value=bits[{}]:{}",
                reset.control.signal.index(),
                reset.control.active_low,
                reset.value.get_bit_count(),
                reset.value.to_string_fmt(IrFormatPreference::Hex, false)
            ));
        }
        text.push('\n');
    }
    text.push_str("transition\n");
    text.push_str(&sequential.transition.to_string_including_dead_nodes());
    text
}

fn parse_sequential_g8r(body: &str) -> Result<SequentialGateFn, String> {
    let (metadata, transition_text) = body
        .split_once("\ntransition\n")
        .ok_or_else(|| "SequentialGateFn payload is missing its transition section".to_string())?;
    let mut lines = metadata.lines();
    let name = parse_prefixed_line(
        lines
            .next()
            .ok_or_else(|| "SequentialGateFn payload is missing its name".to_string())?,
        "name ",
        "name",
    )?
    .to_string();
    let clock = parse_clock(
        lines
            .next()
            .ok_or_else(|| "SequentialGateFn payload is missing its clock".to_string())?,
    )?;
    let inputs = parse_input_ids(parse_prefixed_line(
        lines
            .next()
            .ok_or_else(|| "SequentialGateFn payload is missing its inputs".to_string())?,
        "inputs ",
        "inputs",
    )?)?;
    let outputs = parse_output_ids(parse_prefixed_line(
        lines
            .next()
            .ok_or_else(|| "SequentialGateFn payload is missing its outputs".to_string())?,
        "outputs ",
        "outputs",
    )?)?;
    let register_count: usize = parse_prefixed_line(
        lines
            .next()
            .ok_or_else(|| "SequentialGateFn payload is missing its register count".to_string())?,
        "registers ",
        "register count",
    )?
    .parse()
    .map_err(|e| format!("invalid SequentialGateFn register count: {}", e))?;
    let mut registers = Vec::with_capacity(register_count);
    for _ in 0..register_count {
        let line = lines.next().ok_or_else(|| {
            "SequentialGateFn payload has fewer registers than declared".to_string()
        })?;
        registers.push(parse_register(line)?);
    }
    if let Some(line) = lines.find(|line| !line.is_empty()) {
        return Err(format!(
            "unexpected SequentialGateFn metadata after registers: '{}'",
            line
        ));
    }
    let transition = parse_gate_fn(transition_text)
        .map_err(|e| format!("failed to parse SequentialGateFn transition: {}", e))?;
    SequentialGateFn::new(name, transition, inputs, outputs, clock, registers)
        .map_err(|e| format!("invalid SequentialGateFn payload: {}", e))
}

fn parse_prefixed_line<'a>(line: &'a str, prefix: &str, field: &str) -> Result<&'a str, String> {
    line.strip_prefix(prefix).ok_or_else(|| {
        format!(
            "invalid SequentialGateFn {} line '{}'; expected prefix '{}'",
            field, line, prefix
        )
    })
}

fn parse_clock(line: &str) -> Result<Option<ClockPort>, String> {
    let value = parse_prefixed_line(line, "clock ", "clock")?;
    if value == "none" {
        return Ok(None);
    }
    if value.is_empty() {
        return Err("SequentialGateFn clock line is missing its port name".to_string());
    }
    if value.split_whitespace().count() != 1 {
        return Err("SequentialGateFn clock line has trailing fields".to_string());
    }
    Ok(Some(ClockPort {
        name: value.to_string(),
    }))
}

fn format_input_ids(ids: &[TransitionInputId]) -> String {
    format_ids(ids.iter().map(|id| id.index()))
}

fn format_output_ids(ids: &[TransitionOutputId]) -> String {
    format_ids(ids.iter().map(|id| id.index()))
}

fn format_ids(ids: impl Iterator<Item = usize>) -> String {
    let text = ids
        .map(|id| id.to_string())
        .collect::<Vec<String>>()
        .join(",");
    if text.is_empty() {
        "-".to_string()
    } else {
        text
    }
}

fn parse_input_ids(text: &str) -> Result<Vec<TransitionInputId>, String> {
    parse_ids(text)
        .map(|ids| ids.into_iter().map(TransitionInputId::new).collect())
        .map_err(|e| format!("invalid SequentialGateFn inputs: {}", e))
}

fn parse_output_ids(text: &str) -> Result<Vec<TransitionOutputId>, String> {
    parse_ids(text)
        .map(|ids| ids.into_iter().map(TransitionOutputId::new).collect())
        .map_err(|e| format!("invalid SequentialGateFn outputs: {}", e))
}

fn parse_ids(text: &str) -> Result<Vec<usize>, String> {
    if text == "-" {
        return Ok(vec![]);
    }
    text.split(',')
        .map(|piece| {
            piece
                .parse::<usize>()
                .map_err(|e| format!("invalid transition index '{}': {}", piece, e))
        })
        .collect()
}

fn parse_register(line: &str) -> Result<RegisterBinding, String> {
    let mut parts = line.split_whitespace();
    if parts.next() != Some("register") {
        return Err(format!(
            "invalid SequentialGateFn register line '{}'; expected 'register'",
            line
        ));
    }
    let name = parts
        .next()
        .ok_or_else(|| "SequentialGateFn register is missing its name".to_string())?
        .to_string();
    let mut fields = BTreeMap::new();
    for part in parts {
        let (key, value) = part
            .split_once('=')
            .ok_or_else(|| format!("invalid register field '{}'", part))?;
        if fields.insert(key, value).is_some() {
            return Err(format!("duplicate register field '{}'", key));
        }
    }
    let q = TransitionInputId::new(parse_required_usize(&mut fields, "q")?);
    let d = TransitionOutputId::new(parse_required_usize(&mut fields, "d")?);
    let enable = parse_optional_control(&mut fields, "enable", "enable_active_low")?;
    let reset = parse_optional_control(&mut fields, "reset", "reset_active_low")?
        .map(|control| {
            let value = take_required_field(&mut fields, "reset_value")?;
            let value = IrValue::parse_typed(value)
                .map_err(|e| format!("invalid register reset_value: {e}"))?
                .to_bits()
                .map_err(|e| format!("register reset_value is not bits: {e}"))?;
            Ok::<_, String>(SynchronousReset { control, value })
        })
        .transpose()?;
    if let Some(field) = fields.keys().next() {
        return Err(format!("unknown register field '{}'", field));
    }
    Ok(RegisterBinding {
        enable,
        reset,
        name,
        q,
        d,
    })
}

/// Parses a complete optional control group, rejecting malformed fields.
fn parse_optional_control<'a>(
    fields: &mut BTreeMap<&'a str, &'a str>,
    signal_field: &str,
    polarity_field: &str,
) -> Result<Option<RegisterControl>, String> {
    if !fields.contains_key(signal_field) {
        return Ok(None);
    }
    let signal = TransitionOutputId::new(parse_required_usize(fields, signal_field)?);
    let polarity = take_required_field(fields, polarity_field)?;
    let active_low = polarity
        .parse::<bool>()
        .map_err(|e| format!("invalid register '{polarity_field}' value '{polarity}': {e}"))?;
    Ok(Some(RegisterControl { signal, active_low }))
}

fn take_required_field<'a>(
    fields: &mut BTreeMap<&'a str, &'a str>,
    name: &str,
) -> Result<&'a str, String> {
    fields
        .remove(name)
        .ok_or_else(|| format!("register is missing '{}' field", name))
}

fn parse_required_usize<'a>(
    fields: &mut BTreeMap<&'a str, &'a str>,
    name: &str,
) -> Result<usize, String> {
    let value = take_required_field(fields, name)?;
    value
        .parse()
        .map_err(|e| format!("invalid register '{}' value '{}': {}", name, value, e))
}

#[derive(Debug, Serialize, Deserialize)]
struct BinarySequentialGateFn {
    name: String,
    transition: GateFn,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    clock: Option<String>,
    registers: Vec<BinaryRegisterBinding>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BinaryRegisterBinding {
    name: String,
    q: usize,
    d: usize,
    enable: Option<BinaryRegisterControl>,
    reset: Option<BinarySynchronousReset>,
}

/// Serialized control endpoint and asserted polarity.
#[derive(Debug, Serialize, Deserialize)]
struct BinaryRegisterControl {
    signal: usize,
    active_low: bool,
}

/// Serialized arbitrary-width synchronous reset value.
#[derive(Debug, Serialize, Deserialize)]
struct BinarySynchronousReset {
    control: BinaryRegisterControl,
    value_lsb_is_0: Vec<bool>,
}

impl From<RegisterControl> for BinaryRegisterControl {
    fn from(value: RegisterControl) -> Self {
        Self {
            signal: value.signal.index(),
            active_low: value.active_low,
        }
    }
}

impl From<BinaryRegisterControl> for RegisterControl {
    fn from(value: BinaryRegisterControl) -> Self {
        Self {
            signal: TransitionOutputId::new(value.signal),
            active_low: value.active_low,
        }
    }
}

impl From<&SequentialGateFn> for BinarySequentialGateFn {
    fn from(value: &SequentialGateFn) -> Self {
        Self {
            name: value.name.clone(),
            transition: value.transition.clone(),
            inputs: value.inputs.iter().map(|id| id.index()).collect(),
            outputs: value.outputs.iter().map(|id| id.index()).collect(),
            clock: value.clock.as_ref().map(|clock| clock.name.clone()),
            registers: value
                .registers
                .iter()
                .map(BinaryRegisterBinding::from)
                .collect(),
        }
    }
}

impl From<&RegisterBinding> for BinaryRegisterBinding {
    fn from(value: &RegisterBinding) -> Self {
        Self {
            name: value.name.clone(),
            q: value.q.index(),
            d: value.d.index(),
            enable: value.enable.map(Into::into),
            reset: value.reset.as_ref().map(|reset| BinarySynchronousReset {
                control: reset.control.into(),
                value_lsb_is_0: (0..reset.value.get_bit_count())
                    .map(|i| reset.value.get_bit(i).expect("bit index is in range"))
                    .collect(),
            }),
        }
    }
}

impl TryFrom<BinarySequentialGateFn> for SequentialGateFn {
    type Error = String;

    fn try_from(value: BinarySequentialGateFn) -> Result<Self, Self::Error> {
        let clock = value.clock.map(|name| ClockPort { name });
        let registers = value
            .registers
            .into_iter()
            .map(|register| RegisterBinding {
                enable: register.enable.map(Into::into),
                reset: register.reset.map(|reset| SynchronousReset {
                    control: reset.control.into(),
                    value: IrBits::from_lsb_is_0(&reset.value_lsb_is_0),
                }),
                name: register.name,
                q: TransitionInputId::new(register.q),
                d: TransitionOutputId::new(register.d),
            })
            .collect();
        SequentialGateFn::new(
            value.name,
            value.transition,
            value
                .inputs
                .into_iter()
                .map(TransitionInputId::new)
                .collect(),
            value
                .outputs
                .into_iter()
                .map(TransitionOutputId::new)
                .collect(),
            clock,
            registers,
        )
        .map_err(|e| format!("invalid SequentialGateFn payload: {}", e))
    }
}
