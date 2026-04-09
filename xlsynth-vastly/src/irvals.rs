// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use xlsynth::IrBits;
use xlsynth::IrValue;
use xlsynth::XlsynthError;

use crate::Error;
use crate::LogicBit;
use crate::Result;
use crate::Signedness;
use crate::Value4;
use crate::combo_compile::Port;
use crate::compiled_module::DeclInfo;
use crate::pipeline_compile::CompiledPipelineModule;
use crate::pipeline_harness::PipelineCycle;

pub fn cycles_from_irvals_file(
    m: &CompiledPipelineModule,
    path: &std::path::Path,
    cycles_override: Option<u64>,
) -> Result<Vec<PipelineCycle>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Parse(format!("io error reading irvals `{}`: {e}", path.display())))?;
    let mut parsed: Vec<IrValue> = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let v = IrValue::parse_typed(line).map_err(|e| {
            Error::Parse(format!(
                "failed to parse irvals line {} `{}`: {e}",
                idx + 1,
                line
            ))
        })?;
        parsed.push(v);
    }
    if parsed.is_empty() {
        return Err(Error::Parse(format!(
            "no irvals parsed from `{}`",
            path.display()
        )));
    }

    let inferred_cycles: u64 = parsed.len() as u64;
    let cycles: u64 = cycles_override.unwrap_or(inferred_cycles);
    if cycles_override.is_some() && cycles < inferred_cycles {
        // For irvals input, allow `--cycles` to mean "run the first N vectors".
        parsed.truncate(cycles as usize);
    }

    let mut out: Vec<PipelineCycle> = Vec::with_capacity(cycles as usize);
    for cyc in 0..(cycles as usize) {
        if cyc < parsed.len() {
            out.push(pipeline_cycle_from_irvalue(m, &parsed[cyc])?);
        } else {
            out.push(zero_cycle(&m.combo.input_ports, &m.combo.decls)?);
        }
    }
    Ok(out)
}

/// Converts one XLS typed input value into a single simulation cycle.
pub fn pipeline_cycle_from_irvalue(
    m: &CompiledPipelineModule,
    v: &IrValue,
) -> Result<PipelineCycle> {
    cycle_from_irvalue(&m.combo.input_ports, &m.combo.decls, v)
}

fn cycle_from_irvalue(
    input_ports: &[Port],
    decls: &BTreeMap<String, DeclInfo>,
    v: &IrValue,
) -> Result<PipelineCycle> {
    let leaves =
        flatten_irvalue_to_bits(v).map_err(|e| Error::Parse(format!("irvals flatten: {e}")))?;
    if leaves.len() != input_ports.len() {
        return Err(Error::Parse(format!(
            "irvals element count {} does not match input port count {}",
            leaves.len(),
            input_ports.len()
        )));
    }

    let mut inputs: BTreeMap<String, Value4> = BTreeMap::new();
    for ((p, bits), idx) in input_ports.iter().zip(leaves.iter()).zip(0usize..) {
        let info = decls
            .get(&p.name)
            .ok_or_else(|| Error::Parse(format!("no decl for input `{}`", p.name)))?;
        let v4 = irbits_to_value4(bits, info.signedness)?;
        if v4.width != info.width {
            return Err(Error::Parse(format!(
                "irvals element {idx} width {} does not match input `{}` width {}",
                v4.width, p.name, info.width
            )));
        }
        inputs.insert(p.name.clone(), v4);
    }
    Ok(PipelineCycle { inputs })
}

fn zero_cycle(input_ports: &[Port], decls: &BTreeMap<String, DeclInfo>) -> Result<PipelineCycle> {
    let mut inputs: BTreeMap<String, Value4> = BTreeMap::new();
    for p in input_ports {
        let info = decls
            .get(&p.name)
            .ok_or_else(|| Error::Parse(format!("no decl for input `{}`", p.name)))?;
        inputs.insert(p.name.clone(), Value4::zeros(info.width, info.signedness));
    }
    Ok(PipelineCycle { inputs })
}

fn flatten_irvalue_to_bits(v: &IrValue) -> std::result::Result<Vec<IrBits>, XlsynthError> {
    if let Ok(bits) = v.to_bits() {
        return Ok(vec![bits]);
    }
    let elems = v.get_elements()?;
    let mut out: Vec<IrBits> = Vec::new();
    for e in elems {
        out.extend(flatten_irvalue_to_bits(&e)?);
    }
    Ok(out)
}

/// Converts an XLS `IrBits` value into a `Value4` with the given signedness.
pub fn irbits_to_value4(bits: &IrBits, signedness: Signedness) -> Result<Value4> {
    let width: usize = bits.get_bit_count();
    let width_u32: u32 = width
        .try_into()
        .map_err(|_| Error::Parse(format!("bit width too large: {width}")))?;
    let mut out_bits: Vec<LogicBit> = Vec::with_capacity(width);
    for i in 0..width {
        let b = bits
            .get_bit(i)
            .map_err(|e| Error::Parse(format!("read bit {i}: {e}")))?;
        out_bits.push(if b { LogicBit::One } else { LogicBit::Zero });
    }
    Ok(Value4::new(width_u32, signedness, out_bits))
}
