// SPDX-License-Identifier: Apache-2.0

//! Runs an emitted block for a sequence of clock edges in one Lean process.
//!
//! Lean carries register state between edges and writes one row per edge to
//! a trace file. Each row contains outputs followed by updated registers, all
//! in decimal. The block declarations supply their order and bit widths.

use std::{ffi::OsString, fmt::Write, fs, path::Path};

use anyhow::{Context, Result, bail, ensure};
use xlsynth_ir_value::{IrFormatPreference, IrValue};
use xlsynth_pir::ir::{Block, Package, Type};

use super::{LeanEvaluator, run_lean_process};
use crate::block::{emit, identifier, record_value};

/// The outputs observed before the clock edge and register values after it.
#[derive(Debug, PartialEq, Eq)]
pub struct BlockCycleResult {
    /// Output values in block port declaration order.
    pub outputs: Vec<IrValue>,
    /// Updated register values in register declaration order.
    pub next_state: Vec<IrValue>,
}

impl LeanEvaluator {
    /// Evaluates one clock edge per input vector for a verified block.
    ///
    /// Pass a block from `package` and its initial registers in declaration
    /// order. Each input vector follows input port declaration order, including
    /// any reset or enable input. The clock itself takes no value. Ports and
    /// registers must be bit vectors of their declared widths. Two inputs of
    /// equal width supplied in the wrong order will drive the wrong ports.
    ///
    /// Each result pairs the outputs computed from the current registers and
    /// inputs with the updated registers used for the following cycle.
    ///
    /// # Errors
    ///
    /// Returns an error if the block cannot be emitted, the register or port
    /// values do not match the interface, or the Lean trace cannot be run or
    /// decoded.
    pub fn eval_block_trace(
        &self,
        package: &Package,
        block: &Block,
        initial_state: &[IrValue],
        inputs: &[Vec<IrValue>],
    ) -> Result<Vec<BlockCycleResult>> {
        let mut source = emit(package, block)?;
        let registers = block.registers.iter().map(|r| (r.name.as_str(), &r.ty));
        let ports = |port| (block.port_name(port), block.port_type(port));
        let registers = bit_fields(registers)?;
        let input_fields = bit_fields(block.input_ports().map(ports))?;
        let outputs = bit_fields(block.output_ports().map(ports))?;
        let state = input_record(&registers, initial_state).context("initial registers")?;
        writeln!(source, "\ndef main : IO Unit := do")?;
        writeln!(source, "  let mut q : XlsBlock.State := {state}")?;
        writeln!(source, "  let xs : List XlsBlock.Input := [")?;
        for (cycle, values) in inputs.iter().enumerate() {
            let input = input_record(&input_fields, values)
                .with_context(|| format!("block inputs at cycle {cycle}"))?;
            writeln!(source, "    {input},")?;
        }
        source.push_str(
            r#"  ]
  let out ← IO.FS.Handle.mk "trace.txt" .write
  for x in xs do
    let e := XlsBlock.eval q x
    q := e.next
"#,
        );
        // Frame each cycle independently of Lean diagnostic output. Decimal
        // naturals round-trip without imposing a machine-word width limit.
        let projections = outputs
            .iter()
            .map(|(field, _)| format!("toString e.output.{field}.toNat"))
            .chain(
                registers
                    .iter()
                    .map(|(field, _)| format!("toString e.next.{field}.toNat")),
            )
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            source,
            "    out.putStrLn (String.intercalate \",\" [{projections}])"
        )?;

        let work = tempfile::tempdir()?;
        crate::support::write_file(work.path(), Path::new("RunBlock.lean"), source.as_bytes())?;
        run_lean_process(
            &self.lean_executable,
            work.path(),
            &[work.path(), self.support_root.path()],
            "block trace evaluation",
            &[OsString::from("--run"), OsString::from("RunBlock.lean")],
        )?;
        let trace = fs::read_to_string(work.path().join("trace.txt"))
            .context("reading Lean block trace")?;
        ensure!(
            trace.lines().count() == inputs.len(),
            "wrong number of Lean block cycles"
        );
        trace
            .lines()
            .map(|line| {
                let widths = outputs.iter().chain(&registers);
                let fields: Vec<_> = if line.is_empty() {
                    Vec::new()
                } else {
                    line.split(',').collect()
                };
                ensure!(
                    fields.len() == outputs.len() + registers.len(),
                    "wrong number of Lean block result fields"
                );
                let mut values = widths
                    .zip(fields)
                    .map(|((_, width), value)| {
                        IrValue::parse_typed(&format!("bits[{width}]:{value}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let next_state = values.split_off(outputs.len());
                Ok(BlockCycleResult {
                    outputs: values,
                    next_state,
                })
            })
            .collect()
    }
}

/// Converts PIR fields to Lean field names and bit widths in declaration order.
fn bit_fields<'a>(
    fields: impl Iterator<Item = (&'a str, &'a Type)>,
) -> Result<Vec<(String, usize)>> {
    fields
        .map(|(name, ty)| match ty {
            Type::Bits(width) => Ok((identifier(name)?, *width)),
            _ => bail!("expected a bits-valued block trace field, got {name}: {ty}"),
        })
        .collect()
}

/// Builds a Lean record after checking the value count and each field width.
fn input_record(fields: &[(String, usize)], values: &[IrValue]) -> Result<String> {
    ensure!(
        fields.len() == values.len(),
        "expected {} values, got {}",
        fields.len(),
        values.len()
    );
    let literals = fields
        .iter()
        .zip(values)
        .map(|((name, width), value)| {
            let bits = value
                .to_bits()
                .with_context(|| format!("expected bits[{width}] for {name}"))?;
            ensure!(
                bits.get_bit_count() == *width,
                "expected bits[{width}] for {name}, got {value}"
            );
            Ok((
                name.clone(),
                format!(
                    "0x{}",
                    bits.to_string_fmt(IrFormatPreference::PlainHex, false)
                ),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(record_value(&literals))
}
