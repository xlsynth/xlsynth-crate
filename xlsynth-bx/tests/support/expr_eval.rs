// SPDX-License-Identifier: Apache-2.0

//! Independent expression evaluation for differential tests and fuzzing.

use std::path::Path;

use xlsynth::{DslxConvertOptions, IrFunction, convert_dslx_to_ir};
use xlsynth_bx::ast::{Param, Type};
use xlsynth_bx::{ResetBehavior, check_module, compile, empty_package, lower_module, parse_module};
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir::Fn;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::libxls_bridge::{value_from_libxls, value_to_libxls};
use xlsynth_pir::{IrFormatPreference, IrValue};

/// Wrap the shared body in a BX block, including its clock port.
pub fn bx_source(params: &str, result: &str, body: &str) -> String {
    let ports = if params.is_empty() {
        "clk".to_owned()
    } else {
        format!("clk, {params}")
    };
    format!("block sample({ports}) -> {result} {{\n{body}\n}}")
}

/// Wrap the shared body in a DSLX function with the same value signature.
pub fn dslx_source(params: &str, result: &str, body: &str) -> String {
    format!("fn sample({params}) -> {result} {{\n{body}\n}}")
}

/// Compile the reference expression using the workspace's pinned DSLX frontend.
pub fn compile_bundled_dslx(params: &str, result: &str, body: &str) -> Result<IrFunction, String> {
    let function = dslx_source(params, result, body);
    let package = convert_dslx_to_ir(
        &function,
        Path::new("expr_eval.x"),
        &DslxConvertOptions::default(),
    )
    .map_err(|e| e.to_string())?;
    package
        .ir
        .get_function("__expr_eval__sample")
        .map_err(|e| e.to_string())
}

/// A BX block projected to a PIR function and its native DSLX reference.
pub struct ExpressionEvaluator {
    pub bx: Fn,
    dslx: IrFunction,
    params: Vec<Param>,
    singleton_tuple_result: bool,
}

impl ExpressionEvaluator {
    /// Compile both frontends and report their acceptance separately.
    pub fn compile(
        params: &str,
        result: &str,
        body: &str,
        compile_dslx: fn(&str, &str, &str) -> Result<IrFunction, String>,
    ) -> Result<Self, String> {
        let block = bx_source(params, result, body);
        let checked = parse_module(&block).and_then(|module| check_module(&module));
        let reference = compile_dslx(params, result, body);
        let (checked, dslx) = match (checked, reference) {
            (Ok(checked), Ok(reference)) => (checked, reference),
            (bx, dslx) => {
                let bx =
                    bx.map_or_else(|error| format!("rejected: {error}"), |_| "accepted".into());
                let dslx =
                    dslx.map_or_else(|error| format!("rejected: {error}"), |_| "accepted".into());
                return Err(format!(
                    "Expression failed to compile\nBX {bx}\nDSLX {dslx}\n{block}"
                ));
            }
        };
        let package = lower_module(
            &checked,
            empty_package("expr_eval"),
            ResetBehavior::default(),
        )
        .map_err(|error| format!("BX lowering failed: {error}\n{block}"))?;
        #[cfg(test)]
        xlsynth::IrPackage::parse_ir(&package.to_string(), None)
            .map_err(|error| format!("BX emitted invalid IR: {error}\n{block}"))?;
        let block = package.get_block("sample").ok_or("BX block is missing")?;
        let bx = combinational_block_to_fn(block)
            .map_err(|error| format!("BX block projection failed: {error}"))?;
        let signature = &checked.blocks()[0];
        Ok(Self {
            bx,
            dslx,
            params: signature.params.clone(),
            singleton_tuple_result: matches!(&signature.result, Type::Tuple(fields) if fields.len() == 1),
        })
    }

    /// Run the same inputs in the PIR and native XLS interpreters and return
    /// their common typed value, or a counterexample.
    pub fn compare(&self, args: &[IrValue]) -> Result<IrValue, String> {
        let inputs = || IrValue::make_tuple(args);
        let reference_args = args
            .iter()
            .map(value_to_libxls)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("Cannot pass {} to DSLX: {error}", inputs()))?;
        let expected = self
            .dslx
            .interpret(&reference_args)
            .map_err(|error| format!("DSLX evaluation failed on {}: {error}", inputs()))?;
        let FnEvalResult::Success(result) = eval_fn(&self.bx, args) else {
            return Err(format!("BX evaluation failed on {}", inputs()));
        };
        // A block with a single output port projects to its driving value;
        // preserve the enclosing tuple when the source declared one.
        let actual = if self.singleton_tuple_result {
            IrValue::make_tuple(&[result.value])
        } else {
            result.value
        };
        let expected = value_from_libxls(&expected, &actual.type_()).map_err(|error| {
            format!(
                "Input {}: BX returned {actual}, DSLX returned {expected}: {error}",
                inputs()
            )
        })?;
        if actual != expected {
            return Err(format!(
                "Input {}: BX returned {actual}, DSLX returned {expected}",
                inputs()
            ));
        }
        Ok(expected)
    }

    /// Evaluate a BX reset initializer with scalar inputs bound to their exact
    /// bit patterns, and compare it to the native DSLX result for these inputs.
    pub fn compare_reset(
        &self,
        args: &[IrValue],
        result_type: &str,
        bindings: &str,
        expression: &str,
        expected: &IrValue,
    ) -> Result<(), String> {
        let inputs = self
            .params
            .iter()
            .zip(args)
            .map(|(param, value)| {
                let bits = value.to_string_fmt_no_prefix(IrFormatPreference::Hex);
                format!("let {} = {}:{bits};", param.name, param.ty)
            })
            .collect::<Vec<_>>()
            .join("\n");
        let source = format!(
            "block check_reset(clk, rst) -> {result_type} {{\n{inputs}\n{bindings}\nreg captured: {result_type} = {expression};\ncaptured := captured;\ncaptured\n}}"
        );
        let package = compile(
            &source,
            empty_package("expr_eval"),
            ResetBehavior::default(),
        )
        .map_err(|error| format!("BX reset initializer failed: {error}\n{source}"))?;
        let actual = package
            .get_block("check_reset")
            .and_then(|block| block.registers.first())
            .and_then(|reg| reg.reset_value.as_ref())
            .ok_or_else(|| format!("BX register is missing its reset value\n{source}"))?;
        if actual != expected {
            return Err(format!(
                "Input {}: BX reset returned {actual}, DSLX returned {expected}",
                IrValue::make_tuple(args)
            ));
        }
        Ok(())
    }
}
