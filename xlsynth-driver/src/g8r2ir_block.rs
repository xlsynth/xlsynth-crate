// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use xlsynth_g8r::aig_serdes::g8r::load_sequential_gate_fn_from_path;
use xlsynth_g8r::aig_serdes::sequential2ir::sequential_gate_fn_to_pir_block_package;

pub fn handle_g8r2ir_block(matches: &clap::ArgMatches) -> Result<(), String> {
    let input_file = matches.get_one::<String>("g8r_input_file").unwrap();
    let design = load_sequential_gate_fn_from_path(Path::new(input_file))?;
    let package = sequential_gate_fn_to_pir_block_package(&design, "gate").map_err(|e| {
        format!(
            "g8r2ir-block error: failed to convert SequentialGateFn to XLS block IR for {}: {}",
            input_file, e
        )
    })?;
    println!("{}", package);
    Ok(())
}
