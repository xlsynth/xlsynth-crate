// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_serdes::gate2ir::{
    gate_fn_to_xlsynth_ir, repack_gate_fn_interface_with_schema, GateFnInterfaceSchema,
};
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::gate_builder::GateBuilderOptions;

use crate::fn_type_arg::parse_function_type_text;
use crate::toolchain_config::ToolchainConfig;

fn load_aig_gate_fn(path: &Path) -> Result<GateFn, String> {
    load_aiger_auto_from_path(path, GateBuilderOptions::no_opt())
        .map(|res| res.gate_fn)
        .map_err(|e| format!("failed to load {}: {}", path.display(), e))
}

pub fn handle_aig2ir(matches: &clap::ArgMatches, _config: &Option<ToolchainConfig>) {
    let aig_input_file = matches.get_one::<String>("aig_input_file").unwrap();
    let aig_path = Path::new(aig_input_file);
    let fn_type = matches
        .get_one::<String>("fn_type")
        .map(|text| parse_function_type_text(text))
        .transpose()
        .unwrap_or_else(|e| {
            eprintln!("aig2ir error: {e}");
            std::process::exit(2)
        });

    let gate_fn = load_aig_gate_fn(aig_path).unwrap_or_else(|e| {
        eprintln!("aig2ir error: {}", e);
        std::process::exit(2)
    });

    let (gate_fn, function_type) = if let Some(function_type) = fn_type {
        let schema =
            GateFnInterfaceSchema::from_function_type(&function_type).unwrap_or_else(|e| {
                eprintln!("aig2ir error: {}", e);
                std::process::exit(2)
            });
        let gate_fn = repack_gate_fn_interface_with_schema(gate_fn, &schema).unwrap_or_else(|e| {
            eprintln!("aig2ir error: {}", e);
            std::process::exit(2)
        });
        (gate_fn, function_type)
    } else {
        let function_type = gate_fn.get_flat_type();
        (gate_fn, function_type)
    };

    let ir_pkg = gate_fn_to_xlsynth_ir(&gate_fn, "gate", &function_type).unwrap_or_else(|e| {
        eprintln!(
            "aig2ir error: failed to convert AIGER {} to XLS IR: {}",
            aig_input_file, e
        );
        std::process::exit(2)
    });

    println!("{}", ir_pkg.to_string());
}
