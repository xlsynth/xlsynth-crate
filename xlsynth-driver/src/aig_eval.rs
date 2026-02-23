// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use clap::ArgMatches;
use xlsynth::{IrBits, IrValue};
use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;

use crate::toolchain_config::ToolchainConfig;

fn parse_type_text(text: &str) -> Result<ir::Type, String> {
    let mut parser = ir_parser::Parser::new(text);
    parser
        .parse_type()
        .map_err(|e| format!("failed to parse type `{text}`: {e}"))
}

fn parse_function_type_text(text: &str) -> Result<ir::FunctionType, String> {
    let (params_text, ret_text) = text
        .split_once("->")
        .ok_or_else(|| format!("expected `<param_tuple_type> -> <return_type>`, got `{text}`"))?;
    let params_ty = parse_type_text(params_text.trim())?;
    let ret_ty = parse_type_text(ret_text.trim())?;
    let ir::Type::Tuple(param_members) = params_ty else {
        return Err(format!(
            "expected parameter side to be a tuple type, got `{}`",
            params_ty
        ));
    };
    let param_types = param_members
        .into_iter()
        .map(|boxed| *boxed)
        .collect::<Vec<ir::Type>>();
    Ok(ir::FunctionType {
        param_types,
        return_type: ret_ty,
    })
}

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
    match ty {
        ir::Type::Bits(width) => {
            let bits = value
                .to_bits()
                .map_err(|e| format!("expected bits value for type {}: {e}", ty))?;
            if bits.get_bit_count() != *width {
                return Err(format!(
                    "bit-width mismatch for argument type {}: got {} bits",
                    ty,
                    bits.get_bit_count()
                ));
            }
            Ok(bits)
        }
        ir::Type::Tuple(member_tys) => {
            let elements = value
                .get_elements()
                .map_err(|e| format!("expected tuple value for type {}: {e}", ty))?;
            if elements.len() != member_tys.len() {
                return Err(format!(
                    "tuple arity mismatch for argument type {}: expected {} elements, got {}",
                    ty,
                    member_tys.len(),
                    elements.len()
                ));
            }
            let mut flat_lsb_is_0 = vec![false; ty.bit_count()];
            for index in 0..member_tys.len() {
                let member_bits =
                    flatten_value_lsb_is_0_for_type(&member_tys[index], &elements[index])?;
                let slice = ty
                    .tuple_get_flat_bit_slice_for_index(index)
                    .map_err(|e| format!("failed tuple slice extraction: {e}"))?;
                for bit_index in 0..member_bits.get_bit_count() {
                    flat_lsb_is_0[slice.start + bit_index] = member_bits
                        .get_bit(bit_index)
                        .map_err(|e| format!("failed to read tuple member bit {bit_index}: {e}"))?;
                }
            }
            Ok(IrBits::from_lsb_is_0(&flat_lsb_is_0))
        }
        ir::Type::Array(_) | ir::Type::Token => Err(format!(
            "unsupported argument type for `aig-eval`: {} (supported: bits and tuple-of-bits)",
            ty
        )),
    }
}
fn value_from_type_and_flat_bits(ty: &ir::Type, flat_bits: &IrBits) -> Result<IrValue, String> {
    match ty {
        ir::Type::Bits(width) => {
            if flat_bits.get_bit_count() != *width {
                return Err(format!(
                    "bit-width mismatch for result type {}: have {} bits",
                    ty,
                    flat_bits.get_bit_count()
                ));
            }
            Ok(IrValue::from_bits(flat_bits))
        }
        ir::Type::Tuple(members) => {
            let mut elements = Vec::with_capacity(members.len());
            for index in 0..members.len() {
                let slice = ty
                    .tuple_get_flat_bit_slice_for_index(index)
                    .map_err(|e| format!("failed tuple slice extraction: {e}"))?;
                let member_bits =
                    flat_bits.width_slice(slice.start as i64, (slice.limit - slice.start) as i64);
                let member = value_from_type_and_flat_bits(&members[index], &member_bits)?;
                elements.push(member);
            }
            Ok(IrValue::make_tuple(&elements))
        }
        ir::Type::Array(_) | ir::Type::Token => Err(format!(
            "unsupported result type for `aig-eval`: {} (supported: bits and tuple-of-bits)",
            ty
        )),
    }
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

    let eval_inputs = if gate_input_widths == arg_widths {
        arg_bits
    } else {
        let fn_type = fn_type.as_ref().unwrap_or_else(|| {
            eprintln!(
                "aig-eval error: input widths mismatch; provide --fn-type to repack flattened AIG inputs"
            );
            eprintln!("  AIG input widths: {:?}", gate_input_widths);
            eprintln!("  arg tuple widths: {:?}", arg_widths);
            std::process::exit(1);
        });
        let param_widths = fn_type
            .param_types
            .iter()
            .map(ir::Type::bit_count)
            .collect::<Vec<usize>>();
        if param_widths != arg_widths {
            eprintln!("aig-eval error: --fn-type parameter widths do not match arg tuple");
            eprintln!("  fn-type param widths: {:?}", param_widths);
            eprintln!("  arg tuple widths: {:?}", arg_widths);
            std::process::exit(1);
        }
        let gate_inputs_are_flat_one_bit = gate_input_widths.iter().all(|w| *w == 1);
        let expected_flat_bits: usize = param_widths.iter().sum();
        if !gate_inputs_are_flat_one_bit || gate_input_widths.len() != expected_flat_bits {
            eprintln!(
                "aig-eval error: input widths mismatch and no flatten-repack pattern detected"
            );
            eprintln!("  AIG input widths: {:?}", gate_input_widths);
            eprintln!("  fn-type param widths: {:?}", param_widths);
            std::process::exit(1);
        }
        let mut repacked = Vec::with_capacity(expected_flat_bits);
        for param in arg_bits {
            for bit_index in 0..param.get_bit_count() {
                let bit = param.get_bit(bit_index).unwrap_or_else(|e| {
                    eprintln!("aig-eval error: failed to read input bit {bit_index}: {e}");
                    std::process::exit(1);
                });
                repacked.push(IrBits::from_lsb_is_0(&[bit]));
            }
        }
        repacked
    };

    let sim_result = gate_sim::eval(&gate_fn, &eval_inputs, Collect::None);
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
