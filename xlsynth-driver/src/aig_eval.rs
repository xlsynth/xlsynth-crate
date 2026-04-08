// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use clap::ArgMatches;
use xlsynth::{IrBits, IrValue};
use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_serdes::gate2ir::{
    repack_gate_fn_interface_with_schema, GateFnInterfaceSchema,
};
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_pir::ir;
use xlsynth_pir::ir_value_utils::{
    flatten_ir_value_to_lsb0_bits_for_type, ir_value_from_lsb0_bits_with_layout,
};

use crate::fn_type_arg::parse_function_type_text;
use crate::toolchain_config::ToolchainConfig;

fn load_aig_gate_fn(path: &Path) -> Result<GateFn, String> {
    load_aiger_auto_from_path(path, GateBuilderOptions::no_opt())
        .map(|res| res.gate_fn)
        .map_err(|e| format!("failed to load {}: {}", path.display(), e))
}

fn flatten_outputs_lsb_is_0(outputs: &[IrBits]) -> Result<IrBits, String> {
    let total_bits: usize = outputs.iter().map(IrBits::get_bit_count).sum();
    let mut bits_lsb_first = Vec::with_capacity(total_bits);
    for output in outputs {
        for bit_index in 0..output.get_bit_count() {
            bits_lsb_first.push(
                output
                    .get_bit(bit_index)
                    .map_err(|e| format!("failed to read output bit {bit_index}: {e}"))?,
            );
        }
    }
    Ok(IrBits::from_lsb_is_0(&bits_lsb_first))
}

fn flatten_value_lsb_is_0_for_type(ty: &ir::Type, value: &IrValue) -> Result<IrBits, String> {
    let mut flat_lsb_is_0 = Vec::with_capacity(ty.bit_count());
    flatten_ir_value_to_lsb0_bits_for_type(value, ty, &mut flat_lsb_is_0)
        .map_err(|e| format!("failed to flatten value for type {}: {}", ty, e))?;
    Ok(IrBits::from_lsb_is_0(&flat_lsb_is_0))
}
fn value_from_type_and_flat_bits(ty: &ir::Type, flat_bits: &IrBits) -> Result<IrValue, String> {
    if flat_bits.get_bit_count() != ty.bit_count() {
        return Err(format!(
            "bit-width mismatch for result type {}: have {} bits",
            ty,
            flat_bits.get_bit_count()
        ));
    }
    let mut bits_lsb_is_0 = Vec::with_capacity(flat_bits.get_bit_count());
    for bit_index in 0..flat_bits.get_bit_count() {
        bits_lsb_is_0.push(
            flat_bits
                .get_bit(bit_index)
                .map_err(|e| format!("failed to read output bit {bit_index}: {e}"))?,
        );
    }
    ir_value_from_lsb0_bits_with_layout(ty, &bits_lsb_is_0)
}

pub fn handle_aig_eval(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let aig_file = matches.get_one::<String>("aig_file").unwrap();
    let arg_tuple = matches.get_one::<String>("arg_tuple").unwrap();
    let fn_type = matches
        .get_one::<String>("fn_type")
        .map(|s| parse_function_type_text(s))
        .transpose()
        .unwrap_or_else(|e| {
            eprintln!("aig-eval error: {e}");
            std::process::exit(1);
        });

    let gate_fn = load_aig_gate_fn(Path::new(aig_file)).unwrap_or_else(|e| {
        eprintln!("aig-eval error: {e}");
        std::process::exit(1);
    });
    let gate_fn = if let Some(fn_type) = fn_type.as_ref() {
        let schema = GateFnInterfaceSchema::from_function_type(fn_type).unwrap_or_else(|e| {
            eprintln!("aig-eval error: {e}");
            std::process::exit(1);
        });
        repack_gate_fn_interface_with_schema(gate_fn, &schema).unwrap_or_else(|e| {
            eprintln!("aig-eval error: {e}");
            std::process::exit(1);
        })
    } else {
        gate_fn
    };

    let args_value = IrValue::parse_typed(arg_tuple).unwrap_or_else(|e| {
        eprintln!("aig-eval error: failed to parse argument tuple: {e}");
        std::process::exit(1);
    });
    let args = args_value.get_elements().unwrap_or_else(|e| {
        eprintln!("aig-eval error: argument value is not a tuple: {e}");
        std::process::exit(1);
    });

    let arg_bits: Vec<IrBits> = if let Some(fn_type) = fn_type.as_ref() {
        if args.len() != fn_type.param_types.len() {
            eprintln!("aig-eval error: --fn-type parameter count mismatch with arg tuple");
            eprintln!("  fn-type param count: {}", fn_type.param_types.len());
            eprintln!("  arg tuple count: {}", args.len());
            std::process::exit(1);
        }
        args.iter()
            .zip(fn_type.param_types.iter())
            .enumerate()
            .map(|(i, (value, ty))| {
                flatten_value_lsb_is_0_for_type(ty, value).map_err(|e| {
                    format!(
                        "argument {i} does not match --fn-type parameter {}: {e}",
                        ty
                    )
                })
            })
            .collect::<Result<Vec<IrBits>, String>>()
            .unwrap_or_else(|e| {
                eprintln!("aig-eval error: {e}");
                std::process::exit(1);
            })
    } else {
        args.iter()
            .enumerate()
            .map(|(i, v)| {
                v.to_bits().map_err(|e| {
                    format!(
                        "argument {i} is not bits-typed (currently only bits params are supported without --fn-type): {e}"
                    )
                })
            })
            .collect::<Result<Vec<IrBits>, String>>()
            .unwrap_or_else(|e| {
                eprintln!("aig-eval error: {e}");
                std::process::exit(1);
            })
    };

    let gate_input_widths = gate_fn
        .inputs
        .iter()
        .map(|i| i.get_bit_count())
        .collect::<Vec<usize>>();
    let arg_widths = arg_bits
        .iter()
        .map(IrBits::get_bit_count)
        .collect::<Vec<usize>>();

    if gate_input_widths != arg_widths {
        if fn_type.is_some() {
            eprintln!("aig-eval error: input widths mismatch after applying --fn-type");
        } else {
            eprintln!(
                "aig-eval error: input widths mismatch; provide --fn-type to impose an explicit AIGER interface"
            );
        }
        eprintln!("  AIG input widths: {:?}", gate_input_widths);
        eprintln!("  arg tuple widths: {:?}", arg_widths);
        std::process::exit(1);
    }

    let sim_result = gate_sim::eval(&gate_fn, &arg_bits, Collect::None);
    if let Some(fn_type) = fn_type {
        let flat_output = flatten_outputs_lsb_is_0(&sim_result.outputs).unwrap_or_else(|e| {
            eprintln!("aig-eval error: {e}");
            std::process::exit(1);
        });
        if flat_output.get_bit_count() != fn_type.return_type.bit_count() {
            eprintln!("aig-eval error: --fn-type return width mismatch with AIG outputs");
            eprintln!(
                "  fn-type return width: {}",
                fn_type.return_type.bit_count()
            );
            eprintln!("  AIG output width: {}", flat_output.get_bit_count());
            std::process::exit(1);
        }
        let value = value_from_type_and_flat_bits(&fn_type.return_type, &flat_output)
            .unwrap_or_else(|e| {
                eprintln!("aig-eval error: {e}");
                std::process::exit(1);
            });
        println!("{value}");
    } else if sim_result.outputs.len() == 1 {
        println!("{}", IrValue::from_bits(&sim_result.outputs[0]));
    } else {
        let tuple = IrValue::make_tuple(
            &sim_result
                .outputs
                .iter()
                .map(IrValue::from_bits)
                .collect::<Vec<IrValue>>(),
        );
        println!("{tuple}");
    }
}
