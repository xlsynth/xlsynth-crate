// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;
use xlsynth_g8r::aig::GateFn;
use xlsynth_ir_value::{IrBits, IrValue};
use xlsynth_pir::ir_value_utils::{
    flatten_ir_value_to_lsb0_bits_for_type, ir_value_from_lsb0_bits_with_layout,
};

use crate::process::{resolve_lean_executable, run_lean_process};
use crate::support::{EVALUATOR_SUPPORT, write_file, write_toolchain};
use crate::{
    LeanCircuitBundle, LeanError, LeanExportOptions, LeanFunction, LeanModuleName, export_gate_fn,
};

/// Reusable evaluator that compiles the companion Lean support modules once.
///
/// Each evaluation batch runs in a separate temporary directory. The supplied
/// executable should be the actual `bin/lean` from a Lean distribution.
/// The companion support modules are
/// tested with [`crate::TESTED_LEAN_VERSION`].
/// Initialization and evaluation wait for Lean to finish.
#[derive(Debug)]
pub struct LeanEvaluator {
    lean_executable: PathBuf,
    lean_version: String,
    support_root: TempDir,
}

/// A word-level Lean function and the XLS argument lists to evaluate for it.
pub struct LeanFunctionInputs<'a> {
    /// Function whose signature describes each input case and output.
    pub function: &'a LeanFunction,
    /// One argument list per call, in XLS parameter order.
    pub input_cases: &'a [Vec<IrValue>],
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
        EVALUATOR_SUPPORT.write_to(support_root.path())?;
        let version_output = run_lean_process(
            &lean_executable,
            support_root.path(),
            &[support_root.path()],
            "version query",
            &[OsString::from("--version")],
        )?;
        let lean_version = version_output.stdout.trim().to_string();
        for relative in EVALUATOR_SUPPORT.module_paths() {
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
        let work = evaluation_directory()?;
        let source = bundle.write_to(work.path())?.source;
        let response = self.run_exported_function(
            work.path(),
            source,
            bundle.module_name(),
            "eval",
            &EvalRequest { cases },
        )?;
        validate_outputs(
            input_cases.len(),
            bundle.interface().output_bit_count(),
            &response,
        )?;
        Ok(decode_outputs(bundle, response.outputs))
    }

    /// Compiles and imports the exported module, then evaluates all typed XLS
    /// argument lists in a single request.
    pub fn eval_ir_function_batch(
        &self,
        function: &LeanFunction,
        input_cases: &[Vec<IrValue>],
    ) -> Result<Vec<IrValue>, LeanError> {
        let batch = LeanFunctionInputs {
            function,
            input_cases,
        };
        let request = encode_ir_request(&batch)?;
        let work = evaluation_directory()?;
        let source = function.write_to(work.path())?;
        let response = self.run_exported_function(
            work.path(),
            source,
            function.module_name(),
            "evalPacked",
            &request,
        )?;
        decode_ir_response(&batch, response)
    }

    /// Evaluates independently typed functions in one Lean invocation,
    /// returning results in function order and then in input order. Module
    /// names may repeat.
    pub fn eval_ir_functions(
        &self,
        batches: &[LeanFunctionInputs<'_>],
    ) -> Result<Vec<Vec<IrValue>>, LeanError> {
        if batches.is_empty() {
            return Ok(Vec::new());
        }
        let requests = batches
            .iter()
            .map(encode_ir_request)
            .collect::<Result<Vec<_>, _>>()?;
        let mut driver = String::from(
            "-- SPDX-License-Identifier: Apache-2.0\nimport XlsynthLean.IR\nimport EvalMain\n\n",
        );
        let mut entries = String::new();
        for (i, batch) in batches.iter().enumerate() {
            let module = batch.function.module_name().lean_identifier();
            // Isolate the exported namespace even when several packages use the
            // same name.
            writeln!(
                driver,
                "namespace Batch{i}\n{}end Batch{i}\n",
                batch.function.declarations()
            )
            .unwrap();
            writeln!(
                entries,
                "    XlsynthLean.EvalMain.evaluate Batch{i}.{module}.evalPacked,"
            )
            .unwrap();
        }
        writeln!(
            driver,
            r#"def main (args : List String) : IO UInt32 :=
  XlsynthLean.EvalMain.runFunctions #[
{entries}  ] args"#
        )
        .unwrap();
        let work = evaluation_directory()?;
        let responses: Vec<EvalResponse> = self.run_driver(work.path(), &driver, &requests)?;
        if responses.len() != batches.len() {
            return Err(LeanError::Protocol(format!(
                "received {} functions; expected {}",
                responses.len(),
                batches.len()
            )));
        }
        batches
            .iter()
            .zip(responses)
            .map(|(batch, response)| decode_ir_response(batch, response))
            .collect()
    }

    /// Compiles one standalone module and imports it into the evaluation
    /// driver.
    fn run_exported_function(
        &self,
        root: &Path,
        source: PathBuf,
        module_name: &LeanModuleName,
        entry: &str,
        request: &EvalRequest,
    ) -> Result<EvalResponse, LeanError> {
        run_lean_process(
            &self.lean_executable,
            root,
            &[self.support_root.path(), root],
            "module compilation",
            &[
                OsString::from("-o"),
                source.with_extension("olean").into_os_string(),
                source.into_os_string(),
            ],
        )?;
        let module = module_name.lean_identifier();
        let driver = format!(
            r#"-- SPDX-License-Identifier: Apache-2.0
import EvalMain
import {module}

def main (args : List String) : IO UInt32 :=
  XlsynthLean.EvalMain.runFunction {module}.{entry} args
"#
        );
        self.run_driver(root, &driver, request)
    }

    /// Runs one driver and exchanges its JSON request and response in the batch
    /// directory.
    fn run_driver<R: DeserializeOwned>(
        &self,
        root: &Path,
        driver: &str,
        request: &impl Serialize,
    ) -> Result<R, LeanError> {
        let main_relative = Path::new("XlsynthLean/RunEval.lean");
        write_file(root, main_relative, driver.as_bytes())?;
        let main_path = root.join(main_relative);
        let request_path = root.join("request.json");
        let response_path = root.join("response.json");
        write_path(&request_path, &serde_json::to_vec(request)?)?;
        run_lean_process(
            &self.lean_executable,
            root,
            &[self.support_root.path(), root],
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
        Ok(serde_json::from_slice(&response_bytes)?)
    }
}

/// Creates an isolated workspace for the generated modules and selected
/// toolchain.
fn evaluation_directory() -> Result<TempDir, LeanError> {
    let work = tempfile::Builder::new()
        .prefix("xlsynth-lean-eval-")
        .tempdir()
        .map_err(LeanError::TemporaryDirectory)?;
    write_toolchain(work.path())?;
    Ok(work)
}

/// Validates typed argument lists before packing them into the Lean wire
/// format.
fn encode_ir_request(batch: &LeanFunctionInputs<'_>) -> Result<EvalRequest, LeanError> {
    let signature = batch.function.signature();
    let input_cases = batch.input_cases;
    let mut cases = Vec::with_capacity(input_cases.len());
    for (case_index, args) in input_cases.iter().enumerate() {
        if args.len() != signature.param_types.len() {
            return Err(LeanError::InputCount {
                case_index,
                expected: signature.param_types.len(),
                actual: args.len(),
            });
        }
        let mut flat = Vec::new();
        for (i, (value, ty)) in args.iter().zip(&signature.param_types).enumerate() {
            if value.type_() != *ty {
                return Err(LeanError::InvalidIr(format!(
                    "case {case_index}, argument {i}: expected {ty}, got {}",
                    value.type_()
                )));
            }
            flatten_ir_value_to_lsb0_bits_for_type(value, ty, &mut flat)
                .map_err(LeanError::InvalidIr)?;
        }
        cases.push(
            flat.into_iter()
                .map(|b| if b { '1' } else { '0' })
                .collect(),
        );
    }
    Ok(EvalRequest { cases })
}

/// Reconstructs each typed IR result from the evaluation driver's packed bits.
fn decode_ir_response(
    batch: &LeanFunctionInputs<'_>,
    response: EvalResponse,
) -> Result<Vec<IrValue>, LeanError> {
    let ty = &batch.function.signature().return_type;
    validate_outputs(batch.input_cases.len(), ty.bit_count(), &response)?;
    response
        .outputs
        .iter()
        .map(|text| {
            let bits = text.bytes().map(|b| b == b'1').collect::<Vec<_>>();
            ir_value_from_lsb0_bits_with_layout(ty, &bits).map_err(LeanError::Protocol)
        })
        .collect()
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

/// Checks response dimensions and bit spelling before decoding typed values.
fn validate_outputs(
    expected_cases: usize,
    width: usize,
    response: &EvalResponse,
) -> Result<(), LeanError> {
    if response.outputs.len() != expected_cases {
        return Err(LeanError::Protocol(format!(
            "received {} cases; expected {expected_cases}",
            response.outputs.len()
        )));
    }
    for (case_index, flat) in response.outputs.iter().enumerate() {
        let bits = flat.as_bytes();
        if bits.len() != width {
            return Err(LeanError::Protocol(format!(
                "case {case_index} has {} output bits; expected {width}",
                bits.len(),
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
    }
    Ok(())
}

/// Regroups a validated response into the gate interface's individual ports.
fn decode_outputs(bundle: &LeanCircuitBundle, outputs: Vec<String>) -> Vec<Vec<IrBits>> {
    let interface = bundle.interface();
    outputs
        .into_iter()
        .map(|flat| {
            let bits = flat.as_bytes();
            let mut offset = 0usize;
            let mut outputs = Vec::with_capacity(interface.outputs().len());
            for port in interface.outputs() {
                outputs.push(IrBits::from_lsb_fn(port.bit_count, |index| {
                    bits[offset + index] == b'1'
                }));
                offset += port.bit_count;
            }
            outputs
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
