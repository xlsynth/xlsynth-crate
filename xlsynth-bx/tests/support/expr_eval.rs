// SPDX-License-Identifier: Apache-2.0

//! Independent expression evaluation for differential tests and fuzzing.

use std::path::Path;

use xlsynth::{DslxConvertOptions, IrFunction, convert_dslx_to_ir};
use xlsynth_bx::ast::{Param, Type};
use xlsynth_bx::{
    CheckedModule, ImportOptions, ResetBehavior, check_module_with_options, compile, empty_package,
    lower_module, parse_module,
};
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir::{Fn, Package};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
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
    compile_bundled_module(&dslx_source(params, result, body), &[])
}

/// Compile a reference function with the same module search paths used by BX.
pub fn compile_bundled_module(source: &str, paths: &[&Path]) -> Result<IrFunction, String> {
    let options = DslxConvertOptions {
        additional_search_paths: paths.to_vec(),
        ..Default::default()
    };
    let package = convert_dslx_to_ir(source, Path::new("expr_eval.x"), &options)
        .map_err(|e| e.to_string())?;
    package
        .ir
        .get_function("__expr_eval__sample")
        .map_err(|e| e.to_string())
}

/// The BX block projected to a function, together with its local callees.
pub struct BlockEvaluator {
    pub function: Fn,
    package: Package,
    pub params: Vec<Param>,
    singleton_aggregate_result: bool,
}

impl BlockEvaluator {
    /// Lower the checked block and project its data interface for evaluation.
    pub fn lower(checked: &CheckedModule, block: &str) -> Result<Self, String> {
        let package = lower_module(
            checked,
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
        let signature = checked
            .blocks()
            .iter()
            .find(|block| block.name == "sample")
            .ok_or("checked BX sample block is missing")?;
        Ok(Self {
            function: bx,
            params: signature.params.clone(),
            singleton_aggregate_result: block.output_ports().count() == 1
                && matches!(&signature.result, Type::Tuple(_) | Type::Named(_)),
            package,
        })
    }

    /// Evaluate the projected block and restore its source-level result shape.
    pub fn evaluate(&self, args: &[IrValue]) -> Result<IrValue, String> {
        let FnEvalResult::Success(result) = eval_fn_in_package(&self.package, &self.function, args)
        else {
            return Err(format!(
                "BX evaluation failed on {}",
                IrValue::make_tuple(args)
            ));
        };
        // A block with a single output port projects to its driving value;
        // preserve the enclosing tuple for a one-element tuple or struct.
        let actual = if self.singleton_aggregate_result {
            IrValue::make_tuple(&[result.value])
        } else {
            result.value
        };
        Ok(actual)
    }

    /// Compare the block result with an independently lowered XLS IR function.
    pub fn compare_ir(&self, dslx: &IrFunction, args: &[IrValue]) -> Result<IrValue, String> {
        let inputs = || IrValue::make_tuple(args);
        let reference_args = args
            .iter()
            .map(value_to_libxls)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("Cannot pass {} to DSLX: {error}", inputs()))?;
        let expected = dslx
            .interpret(&reference_args)
            .map_err(|error| format!("DSLX evaluation failed on {}: {error}", inputs()))?;
        let actual = self.evaluate(args)?;
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
}

/// A compiled BX block and its native DSLX reference.
pub struct ExpressionEvaluator {
    pub bx: BlockEvaluator,
    dslx: IrFunction,
}

impl ExpressionEvaluator {
    /// Compile both frontends and report their acceptance separately.
    pub fn compile(
        params: &str,
        result: &str,
        body: &str,
        compile_dslx: fn(&str, &str, &str) -> Result<IrFunction, String>,
    ) -> Result<Self, String> {
        Self::compile_block(
            &bx_source(params, result, body),
            &ImportOptions::default(),
            compile_dslx(params, result, body),
        )
    }

    /// Compile a block and its imports alongside an independently compiled
    /// reference.
    pub fn compile_block(
        block: &str,
        options: &ImportOptions,
        reference: Result<IrFunction, String>,
    ) -> Result<Self, String> {
        let checked =
            parse_module(block).and_then(|module| check_module_with_options(&module, options));
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
        Ok(Self {
            bx: BlockEvaluator::lower(&checked, block)?,
            dslx,
        })
    }

    pub fn compare(&self, args: &[IrValue]) -> Result<IrValue, String> {
        self.bx.compare_ir(&self.dslx, args)
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
            .bx
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
