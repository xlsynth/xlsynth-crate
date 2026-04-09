// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_serdes::gate2ir::{
    gate_fn_to_xlsynth_ir, repack_gate_fn_interface_with_schema, GateFnInterfaceSchema,
};
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_pir::ir;

use crate::fn_type_arg::parse_function_type_text;
use crate::toolchain_config::ToolchainConfig;

fn load_aig_gate_fn(path: &Path) -> Result<GateFn, String> {
    load_aiger_auto_from_path(path, GateBuilderOptions::no_opt())
        .map(|res| res.gate_fn)
        .map_err(|e| format!("failed to load {}: {}", path.display(), e))
}

/// Formats a parseable CLI function type string.
fn format_function_type_text(function_type: &ir::FunctionType) -> String {
    let params_text = function_type
        .param_types
        .iter()
        .map(|param_type| param_type.to_string())
        .collect::<Vec<String>>()
        .join(", ");
    format!("({params_text}) -> {}", function_type.return_type)
}

pub fn handle_aig2ir(matches: &clap::ArgMatches, _config: &Option<ToolchainConfig>) {
    let aig_input_file = matches.get_one::<String>("aig_input_file").unwrap();
    let aig_path = Path::new(aig_input_file);

    let gate_fn = load_aig_gate_fn(aig_path).unwrap_or_else(|e| {
        eprintln!("aig2ir error: {}", e);
        std::process::exit(2)
    });
    let fn_type_text = matches.get_one::<String>("fn_type").unwrap_or_else(|| {
        let naive_type = format_function_type_text(&gate_fn.get_flat_type());
        eprintln!(
            "aig2ir error: --fn-type is required to interpret the raw AIGER interface before lifting.\n  naive type suggestion: `{}`",
            naive_type
        );
        std::process::exit(2)
    });
    let function_type = parse_function_type_text(fn_type_text).unwrap_or_else(|e| {
        eprintln!("aig2ir error: {e}");
        std::process::exit(2)
    });

    let schema = GateFnInterfaceSchema::from_function_type(&function_type).unwrap_or_else(|e| {
        eprintln!("aig2ir error: {}", e);
        std::process::exit(2)
    });
    let gate_fn = repack_gate_fn_interface_with_schema(gate_fn, &schema).unwrap_or_else(|e| {
        eprintln!("aig2ir error: {}", e);
        std::process::exit(2)
    });

    let ir_pkg = gate_fn_to_xlsynth_ir(&gate_fn, "gate", &function_type).unwrap_or_else(|e| {
        eprintln!(
            "aig2ir error: failed to convert AIGER {} to XLS IR: {}",
            aig_input_file, e
        );
        std::process::exit(2)
    });

    println!("{}", ir_pkg.to_string());
}
