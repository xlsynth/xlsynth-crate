// SPDX-License-Identifier: Apache-2.0

//! Stage IR, packed ABI and concrete interpretation shared by stitch checks.

use std::collections::{BTreeMap, HashMap, HashSet};

use xlsynth::ir_package::IrPackage;
use xlsynth_pir::IrBits;
use xlsynth_pir::ir::{self, NodePayload};
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_utils::external_function_references;
use xlsynth_pir::ir_value_utils::{
    flatten_ir_value_to_lsb0_bits_for_type, ir_value_from_lsb0_bits_with_layout,
};
use xlsynth_pir::ir_verify::verify_package;

use crate::aig::sequential_pipeline::{StitchedGatePipeline, stitch_gate_fns_with_layout};
use crate::aig::{AigBitVector, GateBuilder, GateBuilderOptions, SequentialPipelineOptions};

// Both translation and concrete evaluation descend into called functions.
const MAX_CALL_DEPTH: usize = 64;

/// Returns the longest callee chain, counting this function. A pending cache
/// entry denotes a function on the current path.
fn call_height<'a>(
    functions: &BTreeMap<&'a str, &'a ir::Fn>,
    function: &'a ir::Fn,
    remaining: usize,
    heights: &mut HashMap<&'a str, Option<usize>>,
) -> Result<usize, String> {
    let cached = heights.get(function.name.as_str()).copied();
    if remaining == 0 || cached.flatten().is_some_and(|height| height > remaining) {
        return Err(format!(
            "Stitch stage checks support at most {MAX_CALL_DEPTH} nested XLS functions including the stage. Inline helpers to shorten the call chain"
        ));
    }
    if let Some(height) = cached {
        return height.ok_or_else(|| format!("recursive IR call to function '{}'", function.name));
    }
    heights.insert(&function.name, None);
    let mut height = 1;
    for node in &function.nodes {
        if let NodePayload::Invoke { to_apply, .. }
        | NodePayload::Map { to_apply, .. }
        | NodePayload::CountedFor { body: to_apply, .. } = &node.payload
        {
            let callee = functions.get(to_apply.as_str()).ok_or_else(|| {
                format!(
                    "missing IR callee '{to_apply}' in function '{}'",
                    function.name
                )
            })?;
            height = height.max(1 + call_height(functions, callee, remaining - 1, heights)?);
        }
    }
    heights.insert(&function.name, Some(height));
    Ok(height)
}

/// Bounds recursive package verification and translation before accessing any
/// unverified operands. Check stage roots first to report the DSLX stage name.
fn check_call_depths(package: &ir::Package, names: &[(String, String)]) -> Result<(), String> {
    let functions: BTreeMap<_, _> = package
        .members
        .iter()
        .filter_map(|member| match member {
            ir::PackageMember::Function(f) => Some((f.name.as_str(), f)),
            ir::PackageMember::Block(_) => None,
        })
        .collect();
    let mut heights = HashMap::new();
    for (stage, name) in names {
        let root = functions
            .get(name.as_str())
            .ok_or_else(|| format!("stage '{stage}': no IR function '{name}'"))?;
        call_height(&functions, root, MAX_CALL_DEPTH, &mut heights)
            .map_err(|e| format!("stage '{stage}': {e}"))?;
    }
    // The package verifier also visits any functions outside the stage roots.
    for function in functions.values() {
        call_height(&functions, function, MAX_CALL_DEPTH, &mut heights)?;
    }
    Ok(())
}

/// Loads the original XLS declarations of the selected stages and their
/// callees.
pub(super) fn load_stage_package(
    source: &IrPackage,
    names: &[(String, String)],
) -> Result<ir::Package, String> {
    let mut members = Vec::new();
    let mut visited = HashSet::new();
    for (stage, root) in names {
        let mut pending = vec![root.clone()];
        while let Some(name) = pending.pop() {
            if !visited.insert(name.clone()) {
                continue;
            }
            let declaration = source
                .get_function(&name)
                .and_then(|f| f.to_ir_string())
                .map_err(|e| {
                    format!("stage '{stage}': could not read XLS function '{name}': {e}")
                })?;
            // Read the behavioral body even for FFI helpers. Extension
            // metadata describes an implementation that may disagree
            // with the XLS function being specified.
            let function = Parser::new(&declaration)
                .parse_function_declaration()
                .map_err(|e| {
                    format!("stage '{stage}': could not read XLS function '{name}': {e}")
                })?;
            pending.extend(external_function_references(&function));
            members.push(ir::PackageMember::Function(function));
        }
    }
    Ok(ir::Package {
        name: "behavioral_stages".into(),
        members,
        file_table: ir::FileTable::new(),
        top: None,
    })
}

/// Verifies the reachable IR before interpreting it or translating solver
/// terms.
pub(super) fn validate_stage_package(
    package: &ir::Package,
    names: &[(String, String)],
) -> Result<(), String> {
    if names.is_empty() {
        return Err("behavioral pipeline needs at least one stage".into());
    }
    check_call_depths(package, names)?;
    verify_package(package).map_err(|e| format!("invalid behavioral stage IR: {e}"))
}

/// A resolved function together with the DSLX stage name used in diagnostics.
pub(super) struct StageFunction<'a> {
    name: &'a str,
    function: &'a ir::Fn,
}

/// Looks up the ordered stage functions once before running a stream of
/// transactions.
pub(super) fn resolve_functions<'a>(
    package: &'a ir::Package,
    names: impl Iterator<Item = (&'a str, &'a str)>,
) -> Result<Vec<StageFunction<'a>>, String> {
    names
        .map(|(name, ir_name)| {
            let function = package
                .get_fn(ir_name)
                .ok_or_else(|| format!("stage '{name}': no IR function '{ir_name}'"))?;
            Ok(StageFunction { name, function })
        })
        .collect()
}

/// Builds the external ABI and state layout using only the stage signatures.
pub(super) fn interface_layout(
    stages: &[StageFunction<'_>],
    boundary: &SequentialPipelineOptions,
) -> Result<StitchedGatePipeline, String> {
    let mut interfaces = Vec::new();
    for &StageFunction {
        name: stage,
        function: func,
    } in stages
    {
        let mut builder = GateBuilder::new(stage.to_owned(), GateBuilderOptions::no_opt());
        for param in func.param_nodes() {
            builder.add_input(param.param_name().into(), param.ty.bit_count());
        }
        builder.add_output(
            "output_value".into(),
            AigBitVector::zeros(func.ret_ty.bit_count()),
        );
        interfaces.push(builder.build());
    }
    stitch_gate_fns_with_layout(&interfaces, boundary)
}

/// Interprets one transaction through all DSLX stages using the wrapper's
/// packed port order.
pub(super) fn compose_concrete(
    package: &ir::Package,
    stages: &[StageFunction<'_>],
    inputs: &BTreeMap<String, IrBits>,
) -> Result<IrBits, String> {
    let mut previous: Option<Vec<bool>> = None;
    for &StageFunction {
        name: stage,
        function: func,
    } in stages
    {
        let flat = if let Some(bits) = previous.take() {
            bits
        } else {
            let words = named_inputs(stage, func, inputs, IrBits::get_bit_count)?;
            let mut bits = Vec::new();
            for word in words {
                for i in 0..word.get_bit_count() {
                    bits.push(word.get_bit(i).map_err(|e| e.to_string())?);
                }
            }
            bits
        };
        let mut offset = 0;
        let args = func
            .param_nodes()
            .map(|p| {
                let end = offset + p.ty.bit_count();
                let bits = flat.get(offset..end).ok_or_else(|| {
                    format!(
                        "stage '{}': missing packed input bits for '{}'",
                        stage,
                        p.param_name()
                    )
                })?;
                offset = end;
                ir_value_from_lsb0_bits_with_layout(&p.ty, bits)
                    .map_err(|e| format!("stage '{}', parameter '{}': {e}", stage, p.param_name()))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let result = eval_fn_in_package_value(package, func, &args);
        let mut bits = Vec::with_capacity(func.ret_ty.bit_count());
        flatten_ir_value_to_lsb0_bits_for_type(&result, &func.ret_ty, &mut bits)
            .map_err(|e| format!("stage '{}': {e}", stage))?;
        previous = Some(bits);
    }
    previous
        .map(|bits| IrBits::from_lsb_is_0(&bits))
        .ok_or_else(|| "behavioral pipeline needs at least one stage".into())
}

/// Binds external ports by name, checking widths before either evaluator runs.
pub(super) fn named_inputs<T: Clone>(
    stage: &str,
    func: &ir::Fn,
    inputs: &BTreeMap<String, T>,
    width: impl Fn(&T) -> usize,
) -> Result<Vec<T>, String> {
    // A frame may also carry wrapper valid and reset inputs. Only the stage's
    // declared data parameters participate in the DSLX call.
    func.param_nodes()
        .map(|p| {
            let name = p.param_name();
            let value = inputs
                .get(name)
                .ok_or_else(|| format!("stage '{}': missing external input '{name}'", stage))?;
            if width(value) != p.ty.bit_count() {
                return Err(format!(
                    "stage '{}': external input '{name}' expects {} bits, got {}",
                    stage,
                    p.ty.bit_count(),
                    width(value)
                ));
            }
            Ok(value.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_callee_counts_toward_stage_call_depth() {
        let mut helpers = String::from(
            "package pipeline\nfn h0(x: bits[1] id=1) -> bits[1] { ret y: bits[1] = identity(x, id=2) }\n",
        );
        for i in 1..MAX_CALL_DEPTH {
            helpers.push_str(&format!(
                "fn h{i}(x: bits[1] id={}) -> bits[1] {{ ret y: bits[1] = invoke(x, to_apply=h{}, id={}) }}\n",
                2 * i + 1, i - 1, 2 * i + 2
            ));
        }
        for (last_helper, expected) in [
            (MAX_CALL_DEPTH - 2, Ok(())),
            (
                MAX_CALL_DEPTH - 1,
                Err(format!(
                    "stage 'stage': Stitch stage checks support at most {MAX_CALL_DEPTH} nested XLS functions including the stage. Inline helpers to shorten the call chain"
                )),
            ),
        ] {
            let source = format!(
                "{helpers}fn stage(a: bits[1][1] id={}) -> bits[1][1] {{ ret y: bits[1][1] = map(a, to_apply=h{last_helper}, id={}) }}",
                2 * MAX_CALL_DEPTH + 1,
                2 * MAX_CALL_DEPTH + 2
            );
            let package = Parser::new(&source).parse_package().unwrap();
            assert_eq!(
                validate_stage_package(&package, &[("stage".into(), "stage".into())]),
                expected
            );
        }
    }
}
