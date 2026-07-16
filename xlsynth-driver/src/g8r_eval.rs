// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::path::Path;

use clap::ArgMatches;
use xlsynth_g8r::aig_serdes::g8r::load_sequential_gate_fn_from_path;
use xlsynth_g8r::aig_sim::sequential::{count_sequential_toggle_activity, simulate};

use crate::sequential_eval_io::{
    output_value, read_external_inputs, resolve_initial_state, write_final_state,
};

fn write_toggle_activity(
    path: &Path,
    activity: &xlsynth_g8r::aig_sim::sequential::SequentialToggleActivity,
) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|e| format!("failed to create {}: {e}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, activity)
        .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
    writeln!(writer).map_err(|e| format!("failed to finalize {}: {e}", path.display()))
}

pub fn handle_g8r_eval(matches: &ArgMatches) -> Result<(), String> {
    let g8r_file = matches
        .get_one::<String>("g8r_file")
        .expect("g8r_file is required by clap");
    let design = load_sequential_gate_fn_from_path(Path::new(g8r_file))?;
    let inputs = read_external_inputs(matches, &design)?;
    if matches.get_one::<String>("toggle_output_json").is_some() && inputs.len() < 2 {
        return Err(format!(
            "--toggle-output-json requires at least two --input-irvals cycles; got {}",
            inputs.len()
        ));
    }
    let initial_state = resolve_initial_state(
        matches, &design, /* allow_declared_initial_values= */ true, "G8R",
    )?;
    let trace = simulate(&design, &inputs, initial_state)?;

    for outputs in trace.external_outputs() {
        println!("{}", output_value(outputs));
    }
    if let Some(path) = matches.get_one::<String>("final_state_irvals") {
        if design.registers.is_empty() {
            return Err(
                "--final-state-irvals is invalid for a G8R design without registers".to_string(),
            );
        }
        write_final_state(Path::new(path), &design, trace.final_state())?;
    }
    if let Some(path) = matches.get_one::<String>("toggle_output_json") {
        let activity = count_sequential_toggle_activity(&design, &trace)?;
        write_toggle_activity(Path::new(path), &activity)?;
    }
    Ok(())
}
