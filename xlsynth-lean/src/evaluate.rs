// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tempfile::TempDir;
use xlsynth_g8r::aig::GateFn;
use xlsynth_ir_value::IrBits;

use crate::process::{resolve_lean_executable, run_lean_process};
use crate::support::{write_file, write_lean_support};
use crate::{LeanCircuitBundle, LeanError, LeanExportOptions, export_gate_fn};

/// Reusable evaluator that compiles the companion Lean support modules once.
///
/// Each evaluation batch compiles its generated circuit module and runs in a
/// separate temporary directory. The supplied executable should be the actual
/// `bin/lean` from a Lean distribution; the companion support modules are
/// tested with [`crate::TESTED_LEAN_VERSION`].
/// Initialization and evaluation wait for Lean to finish.
#[derive(Debug)]
pub struct LeanEvaluator {
    lean_executable: PathBuf,
    lean_version: String,
    support_root: TempDir,
}

/// JSON request sent to the Lean evaluator; strings contain LSB-first bits.
#[derive(Serialize)]
struct EvalRequest {
    cases: Vec<String>,
}

/// JSON response returned by the Lean evaluator; strings contain LSB-first
/// bits.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvalResponse {
    outputs: Vec<String>,
}

impl LeanEvaluator {
    /// Creates an evaluator and compiles its companion Lean support modules.
    pub fn new(lean_executable: impl Into<PathBuf>) -> Result<Self, LeanError> {
        let lean_executable = lean_executable.into();
        let lean_executable = resolve_lean_executable(&lean_executable)?;
        let support_root = tempfile::Builder::new()
            .prefix("xlsynth-lean-support-")
            .tempdir()
            .map_err(LeanError::TemporaryDirectory)?;
        write_lean_support(support_root.path())?;
        let version_output = run_lean_process(
            &lean_executable,
            support_root.path(),
            &[support_root.path()],
            "version query",
            &[OsString::from("--version")],
        )?;
        let lean_version = version_output.stdout.trim().to_string();
        for relative in [
            "XlsynthLean/Aig.lean",
            "XlsynthLean/Aag.lean",
            "EvalMain.lean",
        ] {
            let source = support_root.path().join(relative);
            let output = source.with_extension("olean");
            run_lean_process(
                &lean_executable,
                support_root.path(),
                &[support_root.path()],
                "support compilation",
                &[
                    OsString::from("-o"),
                    output.into_os_string(),
                    source.into_os_string(),
                ],
            )?;
        }
        Ok(Self {
            lean_executable,
            lean_version,
            support_root,
        })
    }

    /// Returns the executable used by this evaluator.
    pub fn lean_executable(&self) -> &Path {
        &self.lean_executable
    }

    /// Returns the version string reported by the selected Lean executable.
    pub fn lean_version(&self) -> &str {
        &self.lean_version
    }

    /// Exports and evaluates one input case, returning outputs in `GateFn`
    /// order.
    pub fn eval_gate_fn(
        &self,
        graph: &GateFn,
        inputs: &[IrBits],
    ) -> Result<Vec<IrBits>, LeanError> {
        let mut outputs = self.eval_gate_fn_batch(graph, &[inputs.to_vec()])?;
        outputs.pop().ok_or_else(|| {
            LeanError::Protocol("single-case evaluation returned no case".to_string())
        })
    }

    /// Exports and evaluates a batch through Lean.
    pub fn eval_gate_fn_batch(
        &self,
        graph: &GateFn,
        input_cases: &[Vec<IrBits>],
    ) -> Result<Vec<Vec<IrBits>>, LeanError> {
        let bundle = export_gate_fn(graph, &LeanExportOptions::default())?;
        self.eval_circuit_batch(&bundle, input_cases)
    }

    /// Compiles and evaluates an already exported circuit bundle.
    pub fn eval_circuit_batch(
        &self,
        bundle: &LeanCircuitBundle,
        input_cases: &[Vec<IrBits>],
    ) -> Result<Vec<Vec<IrBits>>, LeanError> {
        let cases = encode_input_cases(bundle, input_cases)?;
        let work = tempfile::Builder::new()
            .prefix("xlsynth-lean-eval-")
            .tempdir()
            .map_err(LeanError::TemporaryDirectory)?;
        let request_path = work.path().join("request.json");
        let response_path = work.path().join("response.json");
        let written = bundle.write_to(work.path())?;
        write_file(
            work.path(),
            Path::new("lean-toolchain"),
            include_bytes!("../lean/lean-toolchain"),
        )?;
        write_path(&request_path, &serde_json::to_vec(&EvalRequest { cases })?)?;
        let module_roots = [self.support_root.path(), work.path()];
        run_lean_process(
            &self.lean_executable,
            work.path(),
            &module_roots,
            "circuit compilation",
            &[
                OsString::from("-o"),
                written.source.with_extension("olean").into_os_string(),
                written.source.into_os_string(),
            ],
        )?;
        let module = bundle.module_name().lean_identifier();
        let main_source = format!(
            "-- SPDX-License-Identifier: Apache-2.0\n\nimport EvalMain\nimport {module}\n\ndef main (args : List String) : IO UInt32 :=\n  XlsynthLean.EvalMain.run {module}.circuit args\n"
        );
        let main_relative = Path::new("XlsynthLean/RunEval.lean");
        write_file(work.path(), main_relative, main_source.as_bytes())?;
        let main_path = work.path().join(main_relative);
        run_lean_process(
            &self.lean_executable,
            work.path(),
            &module_roots,
            "evaluation",
            &[
                OsString::from("--run"),
                main_path.into_os_string(),
                request_path.into_os_string(),
                response_path.clone().into_os_string(),
            ],
        )?;
        let response_bytes = fs::read(&response_path).map_err(|source| LeanError::Io {
            operation: "reading Lean evaluation response",
            path: response_path,
            source,
        })?;
        let response: EvalResponse = serde_json::from_slice(&response_bytes)?;
        decode_outputs(bundle, input_cases.len(), response)
    }
}

/// Validates vector shapes and flattens input cases into LSB-first strings.
fn encode_input_cases(
    bundle: &LeanCircuitBundle,
    input_cases: &[Vec<IrBits>],
) -> Result<Vec<String>, LeanError> {
    let interface = bundle.interface();
    input_cases
        .iter()
        .enumerate()
        .map(|(case_index, inputs)| {
            if inputs.len() != interface.inputs().len() {
                return Err(LeanError::InputCount {
                    case_index,
                    expected: interface.inputs().len(),
                    actual: inputs.len(),
                });
            }
            let mut flat = String::with_capacity(interface.input_bit_count());
            for (input_index, (value, port)) in inputs.iter().zip(interface.inputs()).enumerate() {
                if value.get_bit_count() != port.bit_count {
                    return Err(LeanError::InputWidth {
                        case_index,
                        input_index,
                        expected: port.bit_count,
                        actual: value.get_bit_count(),
                    });
                }
                for bit_index in 0..port.bit_count {
                    flat.push(if value.get_bit(bit_index)? { '1' } else { '0' });
                }
            }
            Ok(flat)
        })
        .collect()
}

/// Checks response dimensions and reconstructs the original output vectors.
fn decode_outputs(
    bundle: &LeanCircuitBundle,
    expected_cases: usize,
    response: EvalResponse,
) -> Result<Vec<Vec<IrBits>>, LeanError> {
    if response.outputs.len() != expected_cases {
        return Err(LeanError::Protocol(format!(
            "received {} cases; expected {expected_cases}",
            response.outputs.len()
        )));
    }
    let interface = bundle.interface();
    response
        .outputs
        .into_iter()
        .enumerate()
        .map(|(case_index, flat)| {
            let bits = flat.as_bytes();
            if bits.len() != interface.output_bit_count() {
                return Err(LeanError::Protocol(format!(
                    "case {case_index} has {} output bits; expected {}",
                    bits.len(),
                    interface.output_bit_count()
                )));
            }
            if let Some((bit_index, value)) = bits
                .iter()
                .enumerate()
                .find(|(_, value)| **value != b'0' && **value != b'1')
            {
                return Err(LeanError::Protocol(format!(
                    "case {case_index} output bit {bit_index} is invalid byte {value}"
                )));
            }
            let mut offset = 0usize;
            let mut outputs = Vec::with_capacity(interface.outputs().len());
            for port in interface.outputs() {
                outputs.push(IrBits::from_lsb_fn(port.bit_count, |index| {
                    bits[offset + index] == b'1'
                }));
                offset += port.bit_count;
            }
            Ok(outputs)
        })
        .collect()
}

/// Writes one temporary request artifact.
fn write_path(path: &Path, contents: &[u8]) -> Result<(), LeanError> {
    fs::write(path, contents).map_err(|source| LeanError::Io {
        operation: "writing Lean evaluation input",
        path: path.to_path_buf(),
        source,
    })
}
