// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_serdes::emit_netlist::{
    emit_netlist, emit_netlist_with_version_and_port_style, NetlistPortStyle,
};
use xlsynth_g8r::aig_serdes::gate2ir::{
    repack_gate_fn_interface_with_schema, GateFnInterfacePort, GateFnInterfaceSchema,
};
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::verilog_version::VerilogVersion;
use xlsynth_pir::ir;

use crate::fn_type_arg::parse_function_type_text;

fn load_aig_gate_fn(path: &Path) -> Result<GateFn, String> {
    load_aiger_auto_from_path(path, GateBuilderOptions::no_opt())
        .map(|res| res.gate_fn)
        .map_err(|e| format!("failed to load {}: {}", path.display(), e))
}

fn require_nonzero_bits_type(kind: &str, ty: &ir::Type) -> Result<(), String> {
    match ty {
        ir::Type::Bits(0) => Err(format!(
            "aig2v --fn-type {kind} has zero width; packed Verilog ports require bits[N] with N > 0"
        )),
        ir::Type::Bits(_) => Ok(()),
        _ => Err(format!(
            "aig2v --fn-type currently supports only top-level bits[N] parameters and a bits[M] return; {kind} has type {ty}"
        )),
    }
}

fn schema_from_supported_function_type(
    function_type: &ir::FunctionType,
) -> Result<GateFnInterfaceSchema, String> {
    for (index, param_type) in function_type.param_types.iter().enumerate() {
        require_nonzero_bits_type(&format!("parameter {index}"), param_type)?;
    }
    require_nonzero_bits_type("return value", &function_type.return_type)?;

    Ok(GateFnInterfaceSchema {
        input_ports: function_type
            .param_types
            .iter()
            .enumerate()
            .map(|(index, ty)| GateFnInterfacePort {
                name: format!("arg{index}"),
                ty: ty.clone(),
            })
            .collect::<Vec<GateFnInterfacePort>>(),
        output_ports: vec![GateFnInterfacePort {
            name: "output_value".to_string(),
            ty: function_type.return_type.clone(),
        }],
        return_type: function_type.return_type.clone(),
    })
}

pub fn handle_aig2v(matches: &clap::ArgMatches) -> Result<(), String> {
    let aig_input_file = matches.get_one::<String>("aig_input_file").unwrap();
    let module_name = matches.get_one::<String>("module-name").unwrap();
    let add_clk_port = matches.get_one::<String>("add-clk-port").cloned();
    let flop_inputs = matches.get_flag("flop-inputs");
    let flop_outputs = matches.get_flag("flop-outputs");
    let use_system_verilog = matches.get_flag("use-system-verilog");
    let fn_type = matches
        .get_one::<String>("fn_type")
        .map(|text| parse_function_type_text(text))
        .transpose()?;

    if (flop_inputs || flop_outputs) && add_clk_port.is_none() {
        return Err(
            "--add-clk-port <NAME> is required when --flop-inputs or --flop-outputs is used."
                .to_string(),
        );
    }

    let gate_fn = load_aig_gate_fn(Path::new(aig_input_file))?;
    let netlist_str = if let Some(function_type) = fn_type {
        let schema = schema_from_supported_function_type(&function_type)?;
        let gate_fn = repack_gate_fn_interface_with_schema(gate_fn, &schema)?;
        let version = if use_system_verilog {
            VerilogVersion::SystemVerilog
        } else {
            VerilogVersion::Verilog
        };
        emit_netlist_with_version_and_port_style(
            module_name,
            &gate_fn,
            flop_inputs,
            flop_outputs,
            version,
            add_clk_port,
            NetlistPortStyle::PackedBits,
        )?
    } else {
        emit_netlist(
            module_name,
            &gate_fn,
            flop_inputs,
            flop_outputs,
            use_system_verilog,
            add_clk_port,
        )?
    };

    println!("{}", netlist_str);
    Ok(())
}
