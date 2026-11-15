// SPDX-License-Identifier: Apache-2.0

//! Run Lean block traces and unpack each clock's outputs and next state in IR
//! declaration order.

use std::{ffi::OsString, fmt::Write, fs, path::Path};

use anyhow::{Context, Result, ensure};
use xlsynth_ir_value::IrValue;
use xlsynth_pir::ir::{Block, Package, Type};
use xlsynth_pir::ir_value_utils::ir_value_from_lsb0_bits_with_layout;

use super::{LeanEvaluator, run_lean_process};
use crate::block::{emit, identifier, source_record_value};
use crate::local_expr::{LocalRenderer, bit_width, write_elaboration_limits};

/// The outputs observed before the clock edge and register values after it.
#[derive(Debug, PartialEq, Eq)]
pub struct BlockCycleResult {
    /// Output values in block port declaration order.
    pub outputs: Vec<IrValue>,
    /// Updated register values in register declaration order.
    pub next_state: Vec<IrValue>,
}

impl LeanEvaluator {
    /// Evaluates a verified block for one clock per input vector.
    ///
    /// Supply `initial_state` in register declaration order and each input
    /// vector in data port declaration order. Values must have their declared
    /// PIR types. For example, passing a tuple of two bits to an array port
    /// of type `bits[1][2]` fails even though both values occupy two bits.
    /// Each result contains the outputs computed from that clock's inputs and
    /// starting state, plus the register values carried into the next clock.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid field counts or types, or if emitting the
    /// block, running Lean, or decoding its trace fails.
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
        let registers = trace_fields(registers)?;
        let input_fields = trace_fields(block.input_ports().map(ports))?;
        let outputs = trace_fields(block.output_ports().map(ports))?;
        let state = input_record(&registers, initial_state).context("initial registers")?;
        write_elaboration_limits(&mut source, [&block.graph]);
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
            .map(|(field, _)| {
                format!("toString (_root_.XlsynthLean.IR.Packed.pack e.output.{field})")
            })
            .chain(registers.iter().map(|(field, _)| {
                format!("toString (_root_.XlsynthLean.IR.Packed.pack e.next.{field})")
            }))
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
                let fields: Vec<_> = if line.is_empty() {
                    Vec::new()
                } else {
                    line.split(',').collect()
                };
                ensure!(
                    fields.len() == outputs.len() + registers.len(),
                    "wrong number of Lean block result fields"
                );
                let mut values = outputs
                    .iter()
                    .chain(&registers)
                    .zip(fields)
                    .map(|((name, ty), value)| {
                        decode_field(ty, value).with_context(|| format!("Lean result for {name}"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let next_state = values.split_off(outputs.len());
                Ok(BlockCycleResult {
                    outputs: values,
                    next_state,
                })
            })
            .collect()
    }
}

/// Pairs Lean field names with their PIR types in declaration order.
fn trace_fields<'a>(
    fields: impl Iterator<Item = (&'a str, &'a Type)>,
) -> Result<Vec<(String, &'a Type)>> {
    fields
        .map(|(name, ty)| Ok((identifier(name)?, ty)))
        .collect()
}

/// Restores the PIR shape of a field printed by Lean's `Packed` instance.
///
/// The decimal field carries no type information. The PIR layout determines
/// how to divide its bits among array elements and tuple members, including
/// members that occupy zero bits.
fn decode_field(ty: &Type, value: &str) -> Result<IrValue> {
    let width = bit_width(ty).map_err(anyhow::Error::msg)?;
    let packed = IrValue::parse_typed(&format!("bits[{width}]:{value}"))?;
    if matches!(ty, Type::Bits(_)) {
        return Ok(packed);
    }
    let bits = packed.as_bits()?;
    let flat = (0..width)
        .map(|i| bits.get_bit(i))
        .collect::<Result<Vec<_>, _>>()?;
    ir_value_from_lsb0_bits_with_layout(ty, &flat).map_err(anyhow::Error::msg)
}

/// Build a Lean input or state literal from IR values in declaration order.
///
/// Check each physical type before printing. Records with a generated-name
/// collision bind values positionally.
fn input_record(fields: &[(String, &Type)], values: &[IrValue]) -> Result<String> {
    ensure!(
        fields.len() == values.len(),
        "expected {} values, got {}",
        fields.len(),
        values.len()
    );
    let literals = fields
        .iter()
        .zip(values)
        .map(|((name, ty), value)| {
            let actual = value.type_();
            ensure!(&actual == *ty, "expected {ty} for {name}, got {actual}");
            // The driver is a top-level function, outside the model namespace.
            Ok((name.clone(), LocalRenderer::Function.literal(value)))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(source_record_value(&literals))
}
