// SPDX-License-Identifier: Apache-2.0

use std::{io::Write, path::Path};

use clap::ArgMatches;
use xlsynth::{IrValue, parse_ir_values_file};
use xlsynth_g8r::netlist::gv_eval::{GvEvalOptions, load_labeled_netlist_aig};

fn write_toggle_activity_json(
    path: &str,
    activity: &xlsynth_g8r::netlist::gv_eval::GvToggleActivity,
) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|e| format!("failed to create --toggle-output-json {}: {}", path, e))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, activity)
        .map_err(|e| format!("failed to write --toggle-output-json {}: {}", path, e))?;
    writeln!(writer).map_err(|e| format!("failed to finalize --toggle-output-json {}: {}", path, e))
}

pub fn handle_gv_eval(matches: &ArgMatches) -> Result<(), String> {
    let netlist_path = matches
        .get_one::<String>("netlist")
        .expect("netlist is required by clap");
    let liberty_proto_path = matches
        .get_one::<String>("liberty_proto")
        .expect("liberty_proto is required by clap");
    let options = GvEvalOptions {
        module_name: matches.get_one::<String>("module_name").cloned(),
    };
    let model = load_labeled_netlist_aig(
        Path::new(netlist_path),
        Path::new(liberty_proto_path),
        &options,
    )
    .map_err(|e| format!("failed to build evaluation model: {e:#}"))?;

    let samples = if let Some(arg_tuple) = matches.get_one::<String>("arg_tuple") {
        vec![
            IrValue::parse_typed(arg_tuple)
                .map_err(|e| format!("failed to parse argument tuple: {e}"))?,
        ]
    } else {
        let input_irvals = matches
            .get_one::<String>("input_irvals")
            .expect("clap requires either arg_tuple or input_irvals");
        let argument_names = model
            .gate_fn
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect::<Vec<_>>();
        parse_ir_values_file(Path::new(input_irvals))
            .and_then(|file| file.into_positional_values(&argument_names))
            .map_err(|e| e.to_string())?
    };
    if matches.get_one::<String>("toggle_output_json").is_some() && samples.len() < 2 {
        return Err(
            "--toggle-output-json requires at least two --input-irvals samples".to_string(),
        );
    }
    for result in model.evaluate_ir_values(&samples)? {
        println!("{result}");
    }
    if let Some(toggle_output_json) = matches.get_one::<String>("toggle_output_json") {
        let activity = model.count_toggle_activity(&samples)?;
        write_toggle_activity_json(toggle_output_json, &activity)?;
    }
    Ok(())
}
