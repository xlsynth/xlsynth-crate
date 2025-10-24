// SPDX-License-Identifier: Apache-2.0

//! Evaluate a DSLX function over a list of XLS IR value tuples.
//!
//! Core logic lives here; the CLI subcommand should be a thin shim over this
//! API.

use std::path::Path;

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

fn mangle_candidates(module_name: &str, top_function: &str) -> Vec<String> {
    let mut v = Vec::new();
    // Typical convention
    v.push(
        xlsynth::mangle_dslx_name_with_calling_convention(
            module_name,
            top_function,
            xlsynth::DslxCallingConvention::Typical,
        )
        .expect("mangle typical"),
    );
    // Implicit token convention
    v.push(
        xlsynth::mangle_dslx_name_with_calling_convention(
            module_name,
            top_function,
            xlsynth::DslxCallingConvention::ImplicitToken,
        )
        .expect("mangle itok"),
    );
    v
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
    f: &xlsynth::IrFunction,
    v: xlsynth::IrValue,
) -> xlsynth::IrValue {
    // If return type is (token, X), extract X; otherwise return v as-is.
    let fty = f.get_type().expect("get function type");
    let ret_ty = fty.return_type();
    // Check tuple of size >= 2 with first being token.
    // We don't have a direct tuple-introspection on IrType; check via value shape.
    if let Ok(count) = v.get_element_count() {
        if count >= 2 {
            let first = v.get_element(0).expect("tuple el0");
            let tok_ty = pkg.get_type_for_value(&first).expect("type for el0");
            let token_ty = pkg.get_token_type();
            if pkg.types_eq(&tok_ty, &token_ty).expect("types_eq") {
                return v.get_element(1).expect("tuple el1");
            }
        }
    }
    // Fallback: if return type text starts with "(token," also attempt extraction.
    let ret_ty_str = ret_ty.to_string();
    if ret_ty_str.starts_with("(token,") {
        if let Ok(val1) = v.get_element(1) {
            return val1;
        }
    }
    v
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
    let has_activation = if is_itok && fty.param_count() >= 2 {
        let p1_ty = fty
            .param_type(1)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let one = xlsynth::IrValue::make_ubits(1, 1).expect("make u1");
        let bits1_ty = pkg.get_type_for_value(&one).expect("type for u1");
        pkg.types_eq(&bits1_ty, &p1_ty)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
    } else {
        false
    };
    let logical_param_start = if is_itok {
        if has_activation {
            2
        } else {
            1
        }
    } else {
        0
    };
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
        // Prepend token and optional activation=1.
        args.push(xlsynth::IrValue::make_token());
        if has_activation {
            args.push(xlsynth::IrValue::make_ubits(1, 1).expect("u1:1"));
        }
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
) -> anyhow::Result<Vec<String>> {
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

    // Resolve mangled function name candidates.
    let module_name =
        xlsynth::dslx_path_to_module_name(dslx_file).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let candidates = mangle_candidates(module_name, top_function);

    // Find the function by trying candidates.
    let mut func_opt: Option<xlsynth::IrFunction> = None;
    for name in &candidates {
        if let Ok(f) = pkg.get_function(name) {
            func_opt = Some(f);
            break;
        }
    }
    let func = func_opt.ok_or_else(|| {
        anyhow::anyhow!(format!(
            "Function not found: tried [{}]",
            candidates.join(", ")
        ))
    })?;

    // Determine if the DSLX top requires implicit-token calling convention.
    let requires_itok =
        requires_implicit_token_via_dslx(&dslx_src, dslx_file, module_name, top_function, opts)?;

    // Parse each line as a tuple value; evaluate; stringify result.
    let mut outputs: Vec<String> = Vec::with_capacity(input_values_ir_text.len());
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

        let out_val = match mode {
            EvalMode::Interp => {
                let v = func
                    .interpret(&args)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                unpack_return_value(&pkg, &func, v)
            }
            EvalMode::Jit => {
                let jit = xlsynth::IrFunctionJit::new(&func)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let rr = jit.run(&args).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                if !rr.assert_messages.is_empty() {
                    return Err(anyhow::anyhow!(format!(
                        "assertion failure(s): {}",
                        rr.assert_messages.join("; ")
                    )));
                }
                unpack_return_value(&pkg, &func, rr.value)
            }
            EvalMode::PirInterp => {
                // Parse IR text to PIR and evaluate there (captures asserts/traces).
                let ir_text = pkg.to_string();
                let mut parser = xlsynth_pir::ir_parser::Parser::new(&ir_text);
                let pir_pkg = parser
                    .parse_and_validate_package()
                    .map_err(|e| anyhow::anyhow!(format!("PIR parse: {}", e)))?;
                let name = func.get_name();
                let pir_fn = pir_pkg
                    .get_fn(&name)
                    .ok_or_else(|| anyhow::anyhow!(format!("PIR function '{}' not found", name)))?;

                // Drop itok synthesized args when present for logical mapping? The PIR
                // evaluator expects exact param arity; our args include itok when present
                // in IR, so pass as-is.
                match std::panic::catch_unwind(|| xlsynth_pir::ir_eval::eval_fn(&pir_fn, &args)) {
                    Ok(result) => match result {
                        xlsynth_pir::ir_eval::FnEvalResult::Success(s) => s.value,
                        xlsynth_pir::ir_eval::FnEvalResult::Failure(f) => {
                            let labels: Vec<String> = f
                                .assertion_failures
                                .iter()
                                .map(|a| format!("{}: {}", a.label, a.message))
                                .collect();
                            return Err(anyhow::anyhow!(format!(
                                "assertion failure(s): {}",
                                labels.join("; ")
                            )));
                        }
                    },
                    Err(_payload) => {
                        return Err(anyhow::anyhow!(
                            "assertion failure(s): pir-interp panic while evaluating function"
                        ));
                    }
                }
            }
        };

        // Output formatting policy:
        // - If the input line contains a hex literal ("0x"), emit hex output.
        // - Otherwise, use default formatting (decimal) to mirror common CLI behavior.
        let prefer_hex = trimmed.contains("0x") || trimmed.contains("0X");
        if prefer_hex {
            let s = out_val
                .to_string_fmt(xlsynth::ir_value::IrFormatPreference::Hex)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            outputs.push(s);
        } else {
            outputs.push(out_val.to_string());
        }
    }

    Ok(outputs)
}
