// SPDX-License-Identifier: Apache-2.0

//! Runs a whole input trace through the upstream XLS block interpreter.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use xlsynth::external_tool::{resolve_executable, run_checked};
use xlsynth_pir::IrValue;

/// Output port samples and newly committed register values from one XLS clock
/// edge.
pub struct OracleCycle {
    pub outputs: BTreeMap<String, IrValue>,
    pub registers: BTreeMap<String, IrValue>,
}

fn executable() -> Result<&'static Path, String> {
    static PROGRAM: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    PROGRAM.get_or_init(|| {
        let path = std::env::var_os("XLSYNTH_BLOCK_INTERPRETER_ORACLE")
            .unwrap_or_else(|| "xls_block_oracle".into());
        resolve_executable(Path::new(&path)).map_err(|e| format!(
            "{e}; build xls_block_oracle and set XLSYNTH_BLOCK_INTERPRETER_ORACLE (see xlsynth-pir/fuzz/FUZZ.md)"
        ))
    }).as_deref().map_err(Clone::clone)
}

/// Resolve the oracle before libFuzzer starts mutating inputs.
pub fn check_oracle() -> Result<(), String> {
    executable().map(|_| ())
}

/// Serializes the fuzz generator's simple identifiers and typed XLS values on a
/// single line.
fn write_map(out: &mut impl Write, values: &BTreeMap<String, IrValue>) -> std::io::Result<()> {
    for (index, (name, value)) in values.iter().enumerate() {
        if index != 0 {
            write!(out, "\t")?;
        }
        write!(out, "{name}={value}")?;
    }
    writeln!(out)
}

/// Decode a complete set of named values, retaining widths and aggregate
/// shapes.
fn read_map(line: &str) -> Result<BTreeMap<String, IrValue>, String> {
    let mut values = BTreeMap::new();
    if line.is_empty() {
        return Ok(values);
    }
    for binding in line.split('\t') {
        let (name, text) = binding
            .split_once('=')
            .ok_or_else(|| format!("malformed XLS result binding: {binding:?}"))?;
        let value = IrValue::parse_typed(text)
            .map_err(|e| format!("invalid XLS result for {name}: {text:?}: {e}"))?;
        if values.insert(name.to_owned(), value).is_some() {
            return Err(format!("duplicate XLS result name {name:?}"));
        }
    }
    Ok(values)
}

/// Initializes one upstream continuation and returns each cycle's outputs and
/// next registers.
pub fn evaluate_trace(
    ir: &str,
    top: &str,
    initial: &BTreeMap<String, IrValue>,
    inputs: &[BTreeMap<String, IrValue>],
) -> Result<Vec<OracleCycle>, String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let ir_path = dir.path().join("design.ir");
    fs::write(&ir_path, ir).map_err(|e| e.to_string())?;
    let trace_path = dir.path().join("inputs");
    let mut trace = BufWriter::new(File::create(&trace_path).map_err(|e| e.to_string())?);
    for values in std::iter::once(initial).chain(inputs) {
        write_map(&mut trace, values).map_err(|e| e.to_string())?;
    }
    trace.flush().map_err(|e| e.to_string())?;

    let mut command = Command::new(executable()?);
    command
        .arg(ir_path)
        .arg(top)
        .stdin(File::open(trace_path).map_err(|e| e.to_string())?);
    let output = run_checked(
        &mut command,
        dir.path(),
        "xls_block_oracle",
        Duration::from_secs(10),
    )?;
    let mut lines = output.lines();
    let mut cycles = Vec::with_capacity(inputs.len());
    for cycle in 0..inputs.len() {
        let mut next_map = |what| {
            let line = lines
                .next()
                .ok_or_else(|| format!("missing {what} from XLS at cycle {cycle}"))?;
            read_map(line).map_err(|e| format!("{what} from XLS at cycle {cycle}: {e}"))
        };
        cycles.push(OracleCycle {
            outputs: next_map("outputs")?,
            registers: next_map("registers")?,
        });
    }
    if lines.next().is_some() {
        return Err(format!(
            "unexpected XLS result after {} cycles",
            inputs.len()
        ));
    }
    Ok(cycles)
}
