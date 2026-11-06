// SPDX-License-Identifier: Apache-2.0

//! Check source acceptance, DSLX bytecode results, and lowered hardware
//! separately.

use crate::eval::expr_eval::{BlockEvaluator, bx_source, compile_bundled_module, dslx_source};
use std::fmt::Write;
use std::{env, fs, path::PathBuf, process::Command};
use xlsynth::dslx::{self, ImportData, MatchableModuleMember};
use xlsynth_bx::ast::Type;
use xlsynth_bx::{check_module, parse_module};
use xlsynth_pir::ir_value_utils::ir_bits_to_usize_in_range;
use xlsynth_pir::libxls_bridge::value_to_libxls;
use xlsynth_pir::random_inputs::generate_argument_sets_from_seed;
use xlsynth_pir::{IrFormatPreference, IrValue};

/// Common function declarations and the value expression that drives the block.
pub(crate) struct FunctionProgram {
    pub functions: String,
    pub params: String,
    pub result: String,
    pub body: String,
    pub valid: bool,
    /// The pinned converter emits unparsable IR names for DSLX functions with
    /// punctuation. Such programs still exercise both source frontends and
    /// compare the lowered BX result to actual DSLX execution.
    pub dslx_ir: bool,
    /// Describes the operation whose runtime bounds are checked by native DSLX.
    pub dynamic_index: Option<DynamicIndex>,
}

/// Argument positions and array size for a generated runtime array access.
pub(crate) struct DynamicIndex {
    pub index_argument: usize,
    pub size: usize,
    pub operation: ArrayOperation,
}

/// Reads clamp in hardware, and an out-of-bounds update preserves its input
/// array.
pub(crate) enum ArrayOperation {
    Read,
    Update { array_argument: usize },
}

impl DynamicIndex {
    /// Check the hardware no-op for an update and identify the expected native
    /// error.
    fn expected_bounds_error(
        &self,
        args: &[IrValue],
        hardware: &IrValue,
    ) -> Result<Option<String>, String> {
        let index = args[self.index_argument]
            .as_bits()
            .map_err(|e| e.to_string())?;
        if ir_bits_to_usize_in_range(index, self.size).is_some() {
            return Ok(None);
        }
        let index = index.to_string_fmt(IrFormatPreference::UnsignedDecimal, false);
        let size = self.size;
        match self.operation {
            ArrayOperation::Read => Ok(Some(format!(
                "Index out of bounds; index: {index} >= {size} elements"
            ))),
            ArrayOperation::Update { array_argument } => {
                let original = &args[array_argument];
                if hardware != original {
                    return Err(format!(
                        "out-of-bounds update at index {index} changed {original} to {hardware}"
                    ));
                }
                Ok(Some(format!(
                    "Update index {index} is out of bounds; subject size: {size}"
                )))
            }
        }
    }
}

/// Native execution surface used to run concrete function calls.
pub(crate) enum DslxExecution {
    /// The in-process API exposes concrete function results through constants.
    ConcreteConstants,
    /// Run tests using the pinned interpreter executable, including failures.
    BytecodeTests,
}

/// A sampled call and the result obtained from the BX hardware block.
struct ConcreteCall {
    expression: String,
    hardware: IrValue,
    out_of_bounds: Option<String>,
}

/// Format the concrete value at its source type, retaining aggregate shape
/// and signed or zero-width bit patterns.
fn dslx_literal(ty: &Type, value: &IrValue) -> Result<String, String> {
    match ty {
        Type::Bits { .. } | Type::Bool => Ok(format!(
            "{ty}:{}",
            value.to_string_fmt_no_prefix(IrFormatPreference::Hex)
        )),
        Type::Array { element, .. } => {
            let elements = value
                .as_elements()
                .map_err(|e| e.to_string())?
                .iter()
                .map(|item| dslx_literal(element, item))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(format!("[{}]", elements.join(", ")))
        }
        Type::Tuple(types) => {
            let values = value.as_elements().map_err(|e| e.to_string())?;
            if types.len() != values.len() {
                return Err("tuple literal arity mismatch".into());
            }
            let elements = types
                .iter()
                .zip(values)
                .map(|(ty, v)| dslx_literal(ty, v))
                .collect::<Result<Vec<_>, _>>()?;
            let comma = if types.len() == 1 { "," } else { "" };
            Ok(format!("({}{comma})", elements.join(", ")))
        }
        Type::Named(name) => Err(format!("cannot render fuzz input of named type {name}")),
    }
}

/// Independently parse/typecheck both frontends, then evaluate accepted
/// programs.
pub(crate) fn compare_program(
    program: FunctionProgram,
    seed: u64,
    execution: DslxExecution,
) -> Result<(), String> {
    let bx = format!(
        "{}\n{}",
        program.functions,
        bx_source(&program.params, &program.result, &program.body)
    );
    let reference = format!(
        "{}\n{}",
        program.functions,
        dslx_source(&program.params, &program.result, &program.body)
    );
    let bx_check = parse_module(&bx).and_then(|m| check_module(&m));
    let dslx_check =
        dslx::parse_and_typecheck(&reference, "oracle.x", "oracle", &mut ImportData::default());
    let checked = match (bx_check, dslx_check) {
        (Ok(bx), Ok(_)) if program.valid => bx,
        // An intentionally invalid call or binding must fail source checking
        // on both sides. IR conversion errors never count as source rejection.
        (Err(_), Err(_)) if !program.valid => return Ok(()),
        (bx_result, native_result) => {
            return Err(format!(
                "expected source acceptance {}: BX {:?}, DSLX {:?}\n{bx}",
                program.valid,
                bx_result.err(),
                native_result.err()
            ));
        }
    };
    let evaluator = BlockEvaluator::lower(&checked, &bx)?;
    let native_ir = if program.dslx_ir {
        Some(
            compile_bundled_module(&reference, &[])
                .map_err(|e| format!("DSLX IR lowering failed: {e}\n{reference}"))?,
        )
    } else {
        None
    };
    let mut calls = Vec::new();
    for args in generate_argument_sets_from_seed(&evaluator.function, seed, 4) {
        let hardware = match &native_ir {
            Some(ir) => evaluator.compare_ir(ir, &args),
            None => evaluator.evaluate(&args),
        }
        .map_err(|e| format!("{e}\nseed {seed:#x}\n{bx}"))?;
        let literals = evaluator
            .params
            .iter()
            .zip(&args)
            .map(|(p, arg)| dslx_literal(&p.ty, arg))
            .collect::<Result<Vec<_>, _>>()?
            .join(", ");
        let out_of_bounds = match &program.dynamic_index {
            Some(access) => access
                .expected_bounds_error(&args, &hardware)
                .map_err(|e| format!("{e}\nseed {seed:#x}\n{bx}"))?,
            None => None,
        };
        calls.push(ConcreteCall {
            expression: format!("sample({literals})"),
            hardware,
            out_of_bounds,
        });
    }
    let outcome = match execution {
        DslxExecution::ConcreteConstants => compare_concrete_constants(reference, &calls),
        DslxExecution::BytecodeTests => {
            run_bytecode_tests(reference, &calls, &checked.blocks()[0].result)
        }
    };
    outcome.map_err(|error| format!("{error}\nseed {seed:#x}\n{bx}"))
}

/// Execute in-process calls independently of the DSLX-to-IR converter.
fn compare_concrete_constants(mut source: String, calls: &[ConcreteCall]) -> Result<(), String> {
    for (i, call) in calls.iter().enumerate() {
        writeln!(source, "\nconst RESULT_{i} = {};", call.expression).unwrap();
    }
    let native =
        dslx::parse_and_typecheck(&source, "oracle.x", "oracle", &mut ImportData::default())
            .map_err(|e| format!("DSLX concrete calls failed: {e}\n{source}"))?;
    let module = native.get_module();
    let info = native.get_type_info();
    let first = module.get_member_count() - calls.len();
    for (i, call) in calls.iter().enumerate() {
        let Some(MatchableModuleMember::ConstantDef(constant)) =
            module.get_member(first + i).to_matchable()
        else {
            return Err(format!("DSLX result constant missing\n{source}"));
        };
        let interpreted = info
            .get_const_expr(&constant.get_value())
            .and_then(|v| v.convert_to_ir())
            .map_err(|e| format!("DSLX failed to evaluate {}: {e}", call.expression))?;
        if interpreted != value_to_libxls(&call.hardware).map_err(|e| e.to_string())? {
            return Err(format!(
                "{}: DSLX returned {interpreted}, BX returned {}",
                call.expression, call.hardware
            ));
        }
    }
    Ok(())
}

/// Batch successful calls in one DSLX unit test. Each expected runtime error
/// gets a separate unit test so the interpreter reports its own diagnosis.
fn run_bytecode_tests(
    mut source: String,
    calls: &[ConcreteCall],
    result: &Type,
) -> Result<(), String> {
    source.push_str("\n#[test]\nfn compare_values() {\n");
    for call in calls.iter().filter(|c| c.out_of_bounds.is_none()) {
        let expected = dslx_literal(result, &call.hardware)?;
        writeln!(source, "    assert_eq({}, {expected});", call.expression).unwrap();
    }
    source.push_str("}\n");
    let mut expected_errors = Vec::new();
    for (i, call) in calls.iter().enumerate() {
        if let Some(error) = &call.out_of_bounds {
            writeln!(
                source,
                "#[test]\nfn bounds_{i}() {{ {}; }}",
                call.expression
            )
            .unwrap();
            expected_errors.push(format!(
                "[        FAILED ] bounds_{i}: internal error: INVALID_ARGUMENT: {error}"
            ));
        }
    }
    let tools = env::var_os("XLSYNTH_TOOLS")
        .ok_or("set XLSYNTH_TOOLS to the pinned XLS tools directory")?;
    let binary = PathBuf::from(tools).join("dslx_interpreter_main");
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let path = dir.path().join("oracle.x");
    fs::write(&path, &source).map_err(|e| e.to_string())?;
    let output = Command::new(&binary)
        .args([
            "--evaluator=dslx-interpreter",
            "--compare=none",
            "--warnings_as_errors=false",
        ])
        .arg(format!(
            "--dslx_stdlib_path={}",
            xlsynth::default_dslx_stdlib_path().display()
        ))
        .arg(path)
        .output()
        .map_err(|e| format!("could not launch {}: {e}", binary.display()))?;
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    let failed = expected_errors.len();
    let ran = failed + 1;
    let summary = format!("[===============] {ran} test(s) ran; {failed} failed; 0 skipped.");
    let code = if failed == 0 { 0 } else { 1 };
    if output.status.code() != Some(code)
        || !diagnostics.lines().any(|line| line == summary)
        || expected_errors
            .iter()
            .any(|error| !diagnostics.lines().any(|line| line.starts_with(error)))
    {
        return Err(format!(
            "DSLX runtime comparison failed ({})\n{diagnostics}\n{source}",
            output.status
        ));
    }
    Ok(())
}
