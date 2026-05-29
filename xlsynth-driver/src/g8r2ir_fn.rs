// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use xlsynth_g8r::aig_serdes::g8r::load_sequential_gate_fn_from_path;
use xlsynth_g8r::aig_serdes::gate2ir::gate_fn_to_xlsynth_ir;

pub fn handle_g8r2ir_fn(matches: &clap::ArgMatches) -> Result<(), String> {
    let input_file = matches.get_one::<String>("g8r_input_file").unwrap();
    let design = load_sequential_gate_fn_from_path(Path::new(input_file))?;
    let gate_fn = design.try_into_gate_fn().map_err(|e| {
        format!(
            "g8r2ir-fn error: cannot project SequentialGateFn in {} to an XLS function: {}",
            input_file, e
        )
    })?;
    let flat_type = gate_fn.get_flat_type();
    let package = gate_fn_to_xlsynth_ir(&gate_fn, "gate", &flat_type).map_err(|e| {
        format!(
            "g8r2ir-fn error: failed to convert GateFn to XLS function IR for {}: {}",
            input_file, e
        )
    })?;
    println!("{}", package);
    Ok(())
}
