// SPDX-License-Identifier: Apache-2.0

//! Evaluate a DSLX function over a list of XLS IR value tuples.
//!
//! Core logic lives here; the CLI subcommand should be a thin shim over this
//! API.

use std::panic::AssertUnwindSafe;
use std::path::Path;

use std::io::Write;

struct DumpNodeValuesObserver<'a> {
    out: &'a mut dyn Write,
}

impl xlsynth_pir::ir_eval::EvalObserver for DumpNodeValuesObserver<'_> {
    fn on_select(&mut self, _ev: xlsynth_pir::ir_eval::SelectEvent) {}

    fn on_node_value(
        &mut self,
        _node_ref: xlsynth_pir::ir::NodeRef,
        node_text_id: usize,
        value: &xlsynth::IrValue,
    ) {
        writeln!(
            self.out,
            "pir_node_value node_text_id={} value={}",
            node_text_id, value
        )
        .expect("write pir_node_value line");
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalMode {
    Interp,
    Jit,
    PirInterp,
}

#[derive(Debug, Clone, Default)]
pub struct DslxFnEvalOptions<'a> {
    pub dslx_stdlib_path: Option<&'a Path>,
    pub additional_search_paths: Vec<&'a Path>,
    /// When true, force implicit-token calling convention during DSLX→IR.
    /// Default is false; normally this is inferred from the DSLX program.
    pub force_implicit_token_calling_convention: bool,
}

/// Returns whether the DSLX function requires the implicit-token calling
/// convention.
fn requires_implicit_token_via_dslx(
    dslx_src: &str,
    dslx_file: &Path,
    module_name: &str,
    top_function: &str,
    opts: &DslxFnEvalOptions,
) -> anyhow::Result<bool> {
    let dslx_path_str = dslx_file
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-utf8 dslx_file path"))?;
    let mut import_data =
        xlsynth::dslx::ImportData::new(opts.dslx_stdlib_path, &opts.additional_search_paths);
    let tcm =
        xlsynth::dslx::parse_and_typecheck(dslx_src, dslx_path_str, module_name, &mut import_data)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let module = tcm.get_module();
    let mut target_fn: Option<xlsynth::dslx::Function> = None;
    for i in 0..module.get_member_count() {
        if let Some(xlsynth::dslx::MatchableModuleMember::Function(f)) =
            module.get_member(i).to_matchable()
        {
            if f.get_identifier() == top_function {
                target_fn = Some(f);
                break;
            }
        }
    }
    let func = target_fn.ok_or_else(|| {
        anyhow::anyhow!(format!(
            "DSLX function '{}' not found in module {}",
            top_function, module_name
        ))
    })?;
    let type_info = tcm.get_type_info();
    let requires = type_info
        .requires_implicit_token(&func)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    Ok(requires)
}

fn unpack_return_value(
    pkg: &xlsynth::IrPackage,
    v: xlsynth::IrValue,
    is_itok: bool,
) -> xlsynth::IrValue {
    if !is_itok {
        return v;
    }
    let count = v
        .get_element_count()
        .expect("implicit-token return should be a tuple");
    assert!(
        count >= 2,
        "implicit-token return tuple must have at least 2 elements"
    );
    let first = v.get_element(0).expect("tuple el0 present");
    let tok_ty = pkg.get_type_for_value(&first).expect("type for el0");
    let token_ty = pkg.get_token_type();
    let is_token = pkg.types_eq(&tok_ty, &token_ty).expect("types_eq");
    assert!(
        is_token,
        "first tuple element must be token for itok return"
    );
    v.get_element(1).expect("tuple el1 present")
}

enum DslxFnEvaluator {
    Interp,
    Jit {
        jit: xlsynth::IrFunctionJit,
    },
    PirInterp {
        pir_pkg: xlsynth_pir::ir::Package,
        fn_name: String,
    },
}

impl DslxFnEvaluator {
    fn new(
        mode: EvalMode,
        pkg: &xlsynth::IrPackage,
        func: &xlsynth::IrFunction,
    ) -> anyhow::Result<Self> {
        Ok(match mode {
            EvalMode::Interp => DslxFnEvaluator::Interp,
            EvalMode::Jit => {
                let jit = xlsynth::IrFunctionJit::new(func)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                DslxFnEvaluator::Jit { jit }
            }
            EvalMode::PirInterp => {
                let ir_text = pkg.to_string();
                let mut parser = xlsynth_pir::ir_parser::Parser::new(&ir_text);
                let pir_pkg = parser
                    .parse_and_validate_package()
                    .map_err(|e| anyhow::anyhow!(format!("PIR parse: {}", e)))?;
                let fn_name = func.get_name();
                DslxFnEvaluator::PirInterp { pir_pkg, fn_name }
            }
        })
    }

    fn run(
        &self,
        args: &[xlsynth::IrValue],
        pkg: &xlsynth::IrPackage,
        func: &xlsynth::IrFunction,
        is_itok: bool,
        pir_dump_node_values: bool,
        out: &mut dyn Write,
    ) -> anyhow::Result<xlsynth::IrValue> {
        match self {
            DslxFnEvaluator::Interp => {
                let v = func
                    .interpret(args)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(unpack_return_value(pkg, v, is_itok))
            }
            DslxFnEvaluator::Jit { jit } => {
                let rr = jit.run(args).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                if !rr.assert_messages.is_empty() {
                    return Err(anyhow::anyhow!(format!(
                        "assertion failure(s): {}",
                        rr.assert_messages.join("; ")
                    )));
                }
                Ok(unpack_return_value(pkg, rr.value.clone(), is_itok))
            }
            DslxFnEvaluator::PirInterp { pir_pkg, fn_name } => {
                let pir_fn = pir_pkg
                    .get_fn(fn_name)
                    .ok_or_else(|| anyhow::anyhow!("PIR function lookup failed"))?;

                let mut maybe_observer = if pir_dump_node_values {
                    Some(DumpNodeValuesObserver { out })
                } else {
                    None
                };

                let eval_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    if let Some(observer) = maybe_observer.as_mut() {
                        xlsynth_pir::ir_eval::eval_fn_in_package_with_observer(
                            pir_pkg,
                            &pir_fn,
                            args,
                            Some(observer),
                        )
                    } else {
                        xlsynth_pir::ir_eval::eval_fn_in_package(pir_pkg, &pir_fn, args)
                    }
                }));
                match eval_result {
                    Ok(result) => match result {
                        xlsynth_pir::ir_eval::FnEvalResult::Success(s) => Ok(s.value),
                        xlsynth_pir::ir_eval::FnEvalResult::Failure(f) => {
                            let labels: Vec<String> = f
                                .assertion_failures
                                .iter()
                                .map(|a| format!("{}: {}", a.label, a.message))
                                .collect();
                            Err(anyhow::anyhow!(format!(
                                "assertion failure(s): {}",
                                labels.join("; ")
                            )))
                        }
                    },
                    Err(_payload) => Err(anyhow::anyhow!(
                        "assertion failure(s): pir-interp panic while evaluating function"
                    )),
                }
            }
        }
    }
}

fn build_args_for_call(
    pkg: &xlsynth::IrPackage,
    f: &xlsynth::IrFunction,
    logical_tuple: &xlsynth::IrValue,
    is_itok: bool,
) -> anyhow::Result<Vec<xlsynth::IrValue>> {
    let fty = f.get_type().map_err(|e| anyhow::anyhow!(e.to_string()))?;
    // Require input as tuple for uniformity.
    let logical_elems = logical_tuple
        .get_elements()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    // Activation (u1) immediately follows token when present in the IR signature.
    if is_itok {
        assert!(
            fty.param_count() >= 2,
            "implicit-token calling convention requires exactly (token, activation, ...) parameters"
        );
        let p1_ty = fty
            .param_type(1)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let one = xlsynth::IrValue::make_ubits(1, 1).expect("make u1");
        let bits1_ty = pkg.get_type_for_value(&one).expect("type for u1");
        let is_u1 = pkg
            .types_eq(&bits1_ty, &p1_ty)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        assert!(is_u1, "activation parameter must be bits[1]");
    }
    let logical_param_start = if is_itok { 2 } else { 0 };
    let logical_param_count = fty.param_count() - logical_param_start;
    if logical_elems.len() != logical_param_count {
        return Err(anyhow::anyhow!(format!(
            "arity mismatch: function expects {} logical params; got {}",
            logical_param_count,
            logical_elems.len()
        )));
    }

    // Type-check each logical arg against the function param type.
    for (i, val) in logical_elems.iter().enumerate() {
        let expected_ty = fty
            .param_type(logical_param_start + i)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let got_ty = pkg
            .get_type_for_value(val)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let matches = pkg
            .types_eq(&expected_ty, &got_ty)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        if !matches {
            return Err(anyhow::anyhow!(format!(
                "type mismatch for param {}: expected {}, observed {}",
                i, expected_ty, got_ty
            )));
        }
    }

    let mut args: Vec<xlsynth::IrValue> = Vec::new();
    if is_itok {
        // Prepend token and activation=1.
        args.push(xlsynth::IrValue::make_token());
        args.push(xlsynth::IrValue::make_ubits(1, 1).expect("u1:1"));
    }
    args.extend(logical_elems);
    Ok(args)
}

pub fn evaluate_dslx_function_over_ir_values(
    dslx_file: &Path,
    top_function: &str,
    input_values_ir_text: &[String],
    mode: EvalMode,
    opts: &DslxFnEvalOptions,
    pir_dump_node_values: bool,
    out: &mut dyn Write,
) -> anyhow::Result<()> {
    // Read DSLX source.
    let dslx_src =
        std::fs::read_to_string(dslx_file).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    // Convert DSLX → IR package.
    let mut conv_opts = xlsynth::DslxConvertOptions::default();
    conv_opts.dslx_stdlib_path = opts.dslx_stdlib_path;
    conv_opts.additional_search_paths = opts.additional_search_paths.clone();
    conv_opts.force_implicit_token_calling_convention =
        opts.force_implicit_token_calling_convention;

    let conv = xlsynth::convert_dslx_to_ir(&dslx_src, dslx_file, &conv_opts)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let pkg = conv.ir;

    // Determine module name and whether the DSLX top requires implicit-token.
    let module_name =
        xlsynth::dslx_path_to_module_name(dslx_file).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let requires_itok =
        requires_implicit_token_via_dslx(&dslx_src, dslx_file, module_name, top_function, opts)?;

    // Mangle with the precise calling convention and look up function.
    let mangled = xlsynth::mangle_dslx_name_with_calling_convention(
        module_name,
        top_function,
        if requires_itok {
            xlsynth::DslxCallingConvention::ImplicitToken
        } else {
            xlsynth::DslxCallingConvention::Typical
        },
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let func = pkg
        .get_function(&mangled)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    // Build an evaluator instance that encapsulates backend state.
    let evaluator = DslxFnEvaluator::new(mode, &pkg, &func)?;

    if pir_dump_node_values && mode != EvalMode::PirInterp {
        return Err(anyhow::anyhow!(
            "--pir_dump_node_values is only supported with --eval_mode=pir-interp"
        ));
    }

    // Parse each line as a tuple value; evaluate; write outputs.
    for (lineno, line) in input_values_ir_text.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Err(anyhow::anyhow!(format!(
                "empty line at {} not allowed",
                lineno + 1
            )));
        }
        let tuple_val = xlsynth::IrValue::parse_typed(trimmed)
            .map_err(|e| anyhow::anyhow!(format!("parse error at line {}: {}", lineno + 1, e)))?;
        // Enforce that the value is a tuple (unary requires (v,)).
        tuple_val.get_elements().map_err(|e| {
            anyhow::anyhow!(format!(
                "input at line {} is not a tuple: {}",
                lineno + 1,
                e
            ))
        })?;

        let args = build_args_for_call(&pkg, &func, &tuple_val, requires_itok)?;

        let out_val =
            evaluator.run(&args, &pkg, &func, requires_itok, pir_dump_node_values, out)?;

        writeln!(out, "{}", out_val).expect("write output value");
    }

    Ok(())
}
