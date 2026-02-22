// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use crate::ir_equiv::{dispatch_ir_equiv, IrEquivRequest, IrModule};
use crate::toolchain_config::ToolchainConfig;
use xlsynth_g8r::aig::gate::{AigBitVector, Input};
use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_serdes::gate2ir::gate_fn_to_xlsynth_ir;
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivParallelism};
use xlsynth_prover::prover::SolverChoice;

const SUBCOMMAND: &str = "aig-ir-equiv";

fn load_aig_gate_fn(path: &Path) -> Result<GateFn, String> {
    load_aiger_auto_from_path(path, GateBuilderOptions::no_opt())
        .map(|res| res.gate_fn)
        .map_err(|e| format!("failed to load {}: {}", path.display(), e))
}

fn parse_pir_top_fn(
    ir_text: &str,
    ir_path: &Path,
    top: Option<&str>,
) -> Result<(ir::Package, ir::Fn), String> {
    let mut parser = ir_parser::Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().map_err(|e| {
        format!(
            "failed to parse/validate PIR package {}: {}",
            ir_path.display(),
            e
        )
    })?;

    let f = if let Some(name) = top {
        pkg.get_fn(name)
            .ok_or_else(|| format!("top function '{}' not found in {}", name, ir_path.display()))?
            .clone()
    } else {
        pkg.get_top_fn()
            .ok_or_else(|| format!("no top function found in {}", ir_path.display()))?
            .clone()
    };
    Ok((pkg, f))
}

fn repack_flat_aig_inputs_to_pir_params(pir_fn: &ir::Fn, mut gate_fn: GateFn) -> GateFn {
    let want_param_count = pir_fn.params.len();
    let want_total_bits: usize = pir_fn.params.iter().map(|p| p.ty.bit_count()).sum();
    let gate_total_bits: usize = gate_fn.inputs.iter().map(|i| i.get_bit_count()).sum();

    // The common AIGER form has one scalar input per bit. If widths match,
    // repack those scalar inputs to the PIR parameter grouping.
    let all_one_bit_inputs = gate_fn.inputs.iter().all(|i| i.get_bit_count() == 1);
    if !all_one_bit_inputs
        || gate_fn.inputs.len() != want_total_bits
        || gate_total_bits != want_total_bits
    {
        return gate_fn;
    }

    let mut flat_ops = Vec::with_capacity(want_total_bits);
    for inp in &gate_fn.inputs {
        flat_ops.push(*inp.bit_vector.get_lsb(0));
    }

    let mut new_inputs: Vec<Input> = Vec::with_capacity(want_param_count);
    let mut offset = 0usize;
    for p in &pir_fn.params {
        let width = p.ty.bit_count();
        let slice = &flat_ops[offset..offset + width];
        new_inputs.push(Input {
            name: p.name.clone(),
            bit_vector: AigBitVector::from_lsb_is_index_0(slice),
        });
        offset += width;
    }
    gate_fn.inputs = new_inputs;
    gate_fn
}

fn check_gate_fn_matches_function_type(
    gate_fn: &GateFn,
    function_type: &ir::FunctionType,
) -> Result<(), String> {
    if gate_fn.inputs.len() != function_type.param_types.len() {
        return Err(format!(
            "input count mismatch after repack: gate_fn inputs={} rhs params={}",
            gate_fn.inputs.len(),
            function_type.param_types.len()
        ));
    }

    for (index, (input, param_type)) in gate_fn
        .inputs
        .iter()
        .zip(function_type.param_types.iter())
        .enumerate()
    {
        let input_width = input.get_bit_count();
        let param_width = param_type.bit_count();
        if input_width != param_width {
            return Err(format!(
                "input {} width mismatch after repack: gate_fn bits[{}] rhs param bits[{}]",
                index, input_width, param_width
            ));
        }
    }

    let gate_output_bits: usize = gate_fn.outputs.iter().map(|o| o.get_bit_count()).sum();
    let rhs_return_bits = function_type.return_type.bit_count();
    if gate_output_bits != rhs_return_bits {
        return Err(format!(
            "output width mismatch: gate_fn total bits[{}] rhs return bits[{}]",
            gate_output_bits, rhs_return_bits
        ));
    }
    Ok(())
}

pub fn handle_aig_ir_equiv(matches: &clap::ArgMatches, config: &Option<ToolchainConfig>) {
    let aig_path = Path::new(matches.get_one::<String>("aig_file").unwrap());
    let rhs_ir_path = Path::new(matches.get_one::<String>("rhs_ir_file").unwrap());
    let rhs_top = matches.get_one::<String>("ir_top").map(|s| s.as_str());

    let assertion_semantics = matches
        .get_one::<AssertionSemantics>("assertion_semantics")
        .unwrap_or(&AssertionSemantics::Same);
    let solver: Option<SolverChoice> = matches
        .get_one::<String>("solver")
        .map(|s| s.parse().unwrap());
    let flatten_aggregates = matches
        .get_one::<String>("flatten_aggregates")
        .map(|s| s == "true")
        .unwrap_or(false);
    let drop_params: Vec<String> = matches
        .get_one::<String>("drop_params")
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_else(Vec::new);
    let strategy = matches
        .get_one::<String>("parallelism_strategy")
        .map(|s| s.parse().unwrap())
        .unwrap_or(EquivParallelism::SingleThreaded);
    let lhs_fixed_implicit_activation = matches
        .get_one::<String>("lhs_fixed_implicit_activation")
        .map(|s| s.parse().unwrap())
        .unwrap_or(false);
    let rhs_fixed_implicit_activation = matches
        .get_one::<String>("rhs_fixed_implicit_activation")
        .map(|s| s.parse().unwrap())
        .unwrap_or(false);
    let output_json = matches.get_one::<String>("output_json");
    let assert_label_filter = matches
        .get_one::<String>("assert_label_filter")
        .map(|s| s.as_str());

    let rhs_ir_text = std::fs::read_to_string(rhs_ir_path).unwrap_or_else(|e| {
        eprintln!(
            "{} error: failed to read {}: {}",
            SUBCOMMAND,
            rhs_ir_path.display(),
            e
        );
        std::process::exit(2)
    });
    let (_rhs_pkg, rhs_fn) =
        parse_pir_top_fn(&rhs_ir_text, rhs_ir_path, rhs_top).unwrap_or_else(|e| {
            eprintln!("{} error: {}", SUBCOMMAND, e);
            std::process::exit(2)
        });

    let gate_fn = load_aig_gate_fn(aig_path).unwrap_or_else(|e| {
        eprintln!("{} error: {}", SUBCOMMAND, e);
        std::process::exit(2)
    });
    let gate_fn = repack_flat_aig_inputs_to_pir_params(&rhs_fn, gate_fn);
    let rhs_fn_type = rhs_fn.get_type();
    check_gate_fn_matches_function_type(&gate_fn, &rhs_fn_type).unwrap_or_else(|e| {
        eprintln!("{} error: {}", SUBCOMMAND, e);
        std::process::exit(2)
    });

    let lifted_pkg = gate_fn_to_xlsynth_ir(&gate_fn, "aig", &rhs_fn_type).unwrap_or_else(|e| {
        eprintln!(
            "{} error: failed to convert {} to IR: {}",
            SUBCOMMAND,
            aig_path.display(),
            e
        );
        std::process::exit(2)
    });
    let lhs_ir_text = lifted_pkg.to_string();
    let lhs_top = gate_fn.name.as_str();
    let rhs_top_name = rhs_fn.name.as_str();

    let tool_path = config.as_ref().and_then(|c| c.tool_path.as_deref());
    let tool_path_ref = tool_path.map(Path::new);

    let request = IrEquivRequest::new(
        IrModule::new(&lhs_ir_text)
            .with_path(Some(aig_path))
            .with_top(Some(lhs_top))
            .with_fixed_implicit_activation(lhs_fixed_implicit_activation),
        IrModule::new(&rhs_ir_text)
            .with_path(Some(rhs_ir_path))
            .with_top(Some(rhs_top_name))
            .with_fixed_implicit_activation(rhs_fixed_implicit_activation),
    )
    .with_drop_params(&drop_params)
    .with_flatten_aggregates(flatten_aggregates)
    .with_parallelism(strategy)
    .with_assertion_semantics(*assertion_semantics)
    .with_assert_label_filter(assert_label_filter)
    .with_solver(solver)
    .with_tool_path(tool_path_ref);

    let outcome = dispatch_ir_equiv(&request, SUBCOMMAND);
    if let Some(path) = output_json {
        std::fs::write(path, serde_json::to_string(&outcome).unwrap()).unwrap_or_else(|e| {
            eprintln!(
                "{} error: failed to write output JSON {}: {}",
                SUBCOMMAND, path, e
            );
            std::process::exit(2)
        });
    }

    let dur = std::time::Duration::from_micros(outcome.time_micros as u64);
    if outcome.success {
        println!("[{}] Time taken: {:?}", SUBCOMMAND, dur);
        println!("[{}] success: Solver proved equivalence", SUBCOMMAND);
        std::process::exit(0);
    }

    eprintln!("[{}] Time taken: {:?}", SUBCOMMAND, dur);
    if let Some(err) = outcome.error_str.as_ref() {
        eprintln!("[{}] failure: {}", SUBCOMMAND, err);
    } else {
        eprintln!("[{}] failure", SUBCOMMAND);
    }
    std::process::exit(1);
}
