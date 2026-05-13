// SPDX-License-Identifier: Apache-2.0

use std::{io::Write, path::Path};

use clap::ArgMatches;
use xlsynth::{IrBits, IrValue};
use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_serdes::gate2ir::{
    repack_gate_fn_interface_with_schema, GateFnInterfaceSchema,
};
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::aig_sim::count_toggles;
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::aig_sim::gate_simd;
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_pir::ir;
use xlsynth_pir::ir_value_utils::{
    flatten_ir_value_to_lsb0_bits_for_type, ir_value_from_lsb0_bits_with_layout,
};
use xlsynth_pir::irvals::parse_irvals_tuple_file;

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

fn load_and_repack_gate_fn(
    aig_file: &str,
    fn_type: Option<&ir::FunctionType>,
) -> Result<GateFn, String> {
    let gate_fn = load_aig_gate_fn(Path::new(aig_file))?;
    if let Some(fn_type) = fn_type {
        let schema = GateFnInterfaceSchema::from_function_type(fn_type)?;
        repack_gate_fn_interface_with_schema(gate_fn, &schema)
    } else {
        Ok(gate_fn)
    }
}

fn parse_arg_tuple(arg_tuple: &str) -> Result<IrValue, String> {
    IrValue::parse_typed(arg_tuple).map_err(|e| format!("failed to parse argument tuple: {e}"))
}

fn lower_arg_tuple_to_bits(
    gate_fn: &GateFn,
    args_value: &IrValue,
    fn_type: Option<&ir::FunctionType>,
) -> Result<Vec<IrBits>, String> {
    let args = args_value
        .get_elements()
        .map_err(|e| format!("argument value is not a tuple: {e}"))?;

    let arg_bits: Vec<IrBits> = if let Some(fn_type) = fn_type {
        if args.len() != fn_type.param_types.len() {
            return Err(format!(
                "--fn-type parameter count mismatch with arg tuple: fn-type has {}, arg tuple has {}",
                fn_type.param_types.len(),
                args.len()
            ));
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
            .collect::<Result<Vec<IrBits>, String>>()?
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
            .collect::<Result<Vec<IrBits>, String>>()?
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
            return Err(format!(
                "input widths mismatch after applying --fn-type: AIG input widths {:?}, arg tuple widths {:?}",
                gate_input_widths, arg_widths
            ));
        } else {
            return Err(format!(
                "input widths mismatch; provide --fn-type to impose an explicit AIGER interface: AIG input widths {:?}, arg tuple widths {:?}",
                gate_input_widths, arg_widths
            ));
        }
    }
    Ok(arg_bits)
}

fn value_from_outputs(
    outputs: &[IrBits],
    fn_type: Option<&ir::FunctionType>,
) -> Result<IrValue, String> {
    if let Some(fn_type) = fn_type {
        let flat_output = flatten_outputs_lsb_is_0(outputs)?;
        if flat_output.get_bit_count() != fn_type.return_type.bit_count() {
            return Err(format!(
                "--fn-type return width mismatch with AIG outputs: fn-type return width {}, AIG output width {}",
                fn_type.return_type.bit_count(),
                flat_output.get_bit_count()
            ));
        }
        value_from_type_and_flat_bits(&fn_type.return_type, &flat_output)
    } else if outputs.len() == 1 {
        Ok(IrValue::from_bits(&outputs[0]))
    } else {
        Ok(IrValue::make_tuple(
            &outputs
                .iter()
                .map(IrValue::from_bits)
                .collect::<Vec<IrValue>>(),
        ))
    }
}

fn eval_batch_inputs(
    gate_fn: &GateFn,
    batch_inputs: &[Vec<IrBits>],
    fn_type: Option<&ir::FunctionType>,
    use_simd: bool,
) -> Result<Vec<IrValue>, String> {
    let batch_outputs = if use_simd {
        gate_simd::eval_ordered_batch(gate_fn, batch_inputs)?
    } else {
        batch_inputs
            .iter()
            .map(|arg_bits| gate_sim::eval(gate_fn, arg_bits, Collect::None).outputs)
            .collect::<Vec<Vec<IrBits>>>()
    };
    batch_outputs
        .iter()
        .map(|outputs| value_from_outputs(outputs, fn_type))
        .collect()
}

fn read_input_samples(matches: &ArgMatches) -> Result<Vec<IrValue>, String> {
    if let Some(arg_tuple) = matches.get_one::<String>("arg_tuple") {
        return Ok(vec![parse_arg_tuple(arg_tuple)?]);
    }
    let input_irvals = matches
        .get_one::<String>("input_irvals")
        .expect("clap requires either arg_tuple or input_irvals");
    parse_irvals_tuple_file(Path::new(input_irvals)).map_err(|e| e.to_string())
}

fn write_toggle_activity_json(
    path: &str,
    activity: &count_toggles::ToggleActivityStats,
) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|e| format!("failed to create --toggle-output-json {}: {}", path, e))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, activity)
        .map_err(|e| format!("failed to write --toggle-output-json {}: {}", path, e))?;
    writeln!(writer).map_err(|e| format!("failed to finalize --toggle-output-json {}: {}", path, e))
}

pub fn handle_aig_eval(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let aig_file = matches.get_one::<String>("aig_file").unwrap();
    let fn_type = matches
        .get_one::<String>("fn_type")
        .map(|s| parse_function_type_text(s))
        .transpose()
        .unwrap_or_else(|e| {
            eprintln!("aig-eval error: {e}");
            std::process::exit(1);
        });
    let gate_fn = load_and_repack_gate_fn(aig_file, fn_type.as_ref()).unwrap_or_else(|e| {
        eprintln!("aig-eval error: {e}");
        std::process::exit(1);
    });
    let input_samples = read_input_samples(matches).unwrap_or_else(|e| {
        eprintln!("aig-eval error: {e}");
        std::process::exit(1);
    });
    let batch_inputs = input_samples
        .iter()
        .enumerate()
        .map(|(sample_index, sample)| {
            lower_arg_tuple_to_bits(&gate_fn, sample, fn_type.as_ref())
                .map_err(|e| format!("input sample {}: {}", sample_index + 1, e))
        })
        .collect::<Result<Vec<Vec<IrBits>>, String>>()
        .unwrap_or_else(|e| {
            eprintln!("aig-eval error: {e}");
            std::process::exit(1);
        });
    if matches.get_one::<String>("toggle_output_json").is_some() && batch_inputs.len() < 2 {
        eprintln!(
            "aig-eval error: --toggle-output-json requires at least two --input-irvals samples"
        );
        std::process::exit(1);
    }

    let values = eval_batch_inputs(
        &gate_fn,
        &batch_inputs,
        fn_type.as_ref(),
        matches.get_one::<String>("input_irvals").is_some(),
    )
    .unwrap_or_else(|e| {
        eprintln!("aig-eval error: {e}");
        std::process::exit(1);
    });
    for value in values {
        println!("{value}");
    }

    if let Some(toggle_output_json) = matches.get_one::<String>("toggle_output_json") {
        let activity = count_toggles::count_toggle_activity(&gate_fn, &batch_inputs);
        write_toggle_activity_json(toggle_output_json, &activity).unwrap_or_else(|e| {
            eprintln!("aig-eval error: {e}");
            std::process::exit(1);
        });
    }
}
