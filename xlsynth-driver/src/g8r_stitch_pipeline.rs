// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use clap::ArgMatches;
use xlsynth_g8r::aig::{ClockPort, SequentialPipelineOptions, stitch_gate_fns_into_pipeline};
use xlsynth_g8r::aig_serdes::g8r::{
    emit_g8r, encode_g8r_binary, load_sequential_gate_fn_from_path,
};

use crate::common::{parse_bool_flag_or, write_stdout};

/// Handles the `g8r-stitch-pipeline` subcommand.
pub fn handle_g8r_stitch_pipeline(matches: &ArgMatches) -> Result<(), String> {
    let stage_gate_fns = matches
        .get_many::<String>("g8r_input_files")
        .expect("clap requires at least one g8r input file")
        .map(|input_file| {
            let input_path = Path::new(input_file);
            load_sequential_gate_fn_from_path(input_path)?
                .try_into_gate_fn()
                .map_err(|error| {
                    format!(
                        "pipeline stage '{}' must be clockless and register-free: {}",
                        input_path.display(),
                        error
                    )
                })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let output_design_name = matches.get_one::<String>("output_design_name").unwrap();
    let options = SequentialPipelineOptions {
        name: output_design_name.clone(),
        clock: ClockPort {
            name: matches
                .get_one::<String>("clock_name")
                .cloned()
                .unwrap_or_else(|| "clk".to_string()),
        },
        flop_inputs: parse_bool_flag_or(
            matches,
            "flop_inputs",
            crate::flag_defaults::CODEGEN_FLOP_INPUTS,
        ),
        flop_outputs: parse_bool_flag_or(
            matches,
            "flop_outputs",
            crate::flag_defaults::CODEGEN_FLOP_OUTPUTS,
        ),
        input_valid_signal: matches.get_one::<String>("input_valid_signal").cloned(),
        output_valid_signal: matches.get_one::<String>("output_valid_signal").cloned(),
        reset_signal: matches.get_one::<String>("reset").cloned(),
        reset_active_low: parse_bool_flag_or(matches, "reset_active_low", false),
    };
    let design = stitch_gate_fns_into_pipeline(&stage_gate_fns, &options)
        .map_err(|error| format!("could not stitch g8r pipeline: {error}"))?;

    write_stdout(&emit_g8r(&design));
    if let Some(bin_path) = matches.get_one::<String>("bin_out") {
        let binary = encode_g8r_binary(&design)
            .map_err(|error| format!("could not serialize g8r binary: {error}"))?;
        std::fs::write(bin_path, binary)
            .map_err(|error| format!("could not write g8r binary '{}': {error}", bin_path))?;
    }
    Ok(())
}
