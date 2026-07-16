// SPDX-License-Identifier: Apache-2.0

use std::{io::Write, path::Path};

use clap::ArgMatches;
use xlsynth::{IrValue, parse_ir_values_file};
use xlsynth_g8r::netlist::gv_eval::{
    GvEvalOptions, load_labeled_netlist_aig, load_labeled_sequential_netlist_aig,
};
use xlsynth_g8r::netlist::io::load_liberty_from_path;
use xlsynth_g8r::netlist::power::{GvDynamicPowerOptions, GvDynamicPowerReport};

use crate::sequential_eval_io::{
    output_value, read_external_inputs, resolve_initial_state, write_final_state,
};

fn write_toggle_activity_json<T: serde::Serialize>(path: &str, activity: &T) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|e| format!("failed to create --toggle-output-json {}: {}", path, e))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, activity)
        .map_err(|e| format!("failed to write --toggle-output-json {}: {}", path, e))?;
    writeln!(writer).map_err(|e| format!("failed to finalize --toggle-output-json {}: {}", path, e))
}

fn write_power_json(path: &str, report: &GvDynamicPowerReport) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|e| format!("failed to create --power-output-json {}: {}", path, e))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, report)
        .map_err(|e| format!("failed to write --power-output-json {}: {}", path, e))?;
    writeln!(writer).map_err(|e| format!("failed to finalize --power-output-json {}: {}", path, e))
}

pub fn handle_gv_eval(matches: &ArgMatches) -> Result<(), String> {
    if matches.get_flag("sequential") {
        handle_sequential_gv_eval(matches)
    } else {
        handle_combinational_gv_eval(matches)
    }
}

fn gv_eval_options(matches: &ArgMatches) -> GvEvalOptions {
    GvEvalOptions {
        module_name: matches.get_one::<String>("module_name").cloned(),
        clock_port_name: matches.get_one::<String>("clock_port_name").cloned(),
    }
}

fn handle_combinational_gv_eval(matches: &ArgMatches) -> Result<(), String> {
    let netlist_path = matches
        .get_one::<String>("netlist")
        .expect("netlist is required by clap");
    let liberty_proto_path = matches
        .get_one::<String>("liberty_proto")
        .expect("liberty_proto is required by clap");
    let options = gv_eval_options(matches);
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
    if matches.get_one::<String>("power_output_json").is_some() && samples.len() < 2 {
        return Err("--power-output-json requires at least two --input-irvals samples".to_string());
    }
    for result in model.evaluate_ir_values(&samples)? {
        println!("{result}");
    }
    if let Some(toggle_output_json) = matches.get_one::<String>("toggle_output_json") {
        let activity = model.count_toggle_activity(&samples)?;
        write_toggle_activity_json(toggle_output_json, &activity)?;
    }
    if let Some(power_output_json) = matches.get_one::<String>("power_output_json") {
        let library = load_liberty_from_path(Path::new(liberty_proto_path))
            .map_err(|e| format!("failed to reload Liberty power model: {e:#}"))?;
        let power_options = GvDynamicPowerOptions {
            primary_input_transition: *matches
                .get_one::<f64>("primary_input_transition")
                .expect("primary_input_transition has a default"),
            module_output_load: *matches
                .get_one::<f64>("module_output_load")
                .expect("module_output_load has a default"),
            cycle_time: matches.get_one::<f64>("cycle_time").copied(),
        };
        let report = model.analyze_dynamic_power(&library, &samples, power_options)?;
        write_power_json(power_output_json, &report)?;
    }
    Ok(())
}

fn handle_sequential_gv_eval(matches: &ArgMatches) -> Result<(), String> {
    if matches.get_one::<String>("power_output_json").is_some() {
        return Err("--power-output-json is not supported with --sequential yet".to_string());
    }
    let netlist_path = matches
        .get_one::<String>("netlist")
        .expect("netlist is required by clap");
    let liberty_proto_path = matches
        .get_one::<String>("liberty_proto")
        .expect("liberty_proto is required by clap");
    let model = load_labeled_sequential_netlist_aig(
        Path::new(netlist_path),
        Path::new(liberty_proto_path),
        &gv_eval_options(matches),
    )
    .map_err(|error| format!("failed to build sequential evaluation model: {error:#}"))?;
    let design = &model.sequential_gate_fn;
    let inputs = read_external_inputs(matches, design)?;
    if matches.get_one::<String>("toggle_output_json").is_some() && inputs.len() < 2 {
        return Err(format!(
            "--toggle-output-json requires at least two --input-irvals cycles; got {}",
            inputs.len()
        ));
    }
    let initial_state = resolve_initial_state(
        matches,
        design,
        /* allow_declared_initial_values= */ false,
        "mapped netlist",
    )?;
    let trace = model.simulate_bits(&inputs, initial_state)?;

    for outputs in trace.external_outputs() {
        println!("{}", output_value(outputs));
    }
    if let Some(path) = matches.get_one::<String>("final_state_irvals") {
        if design.registers.is_empty() {
            return Err(
                "--final-state-irvals is invalid for a sequential gv-eval design without registers"
                    .to_string(),
            );
        }
        write_final_state(Path::new(path), design, trace.final_state())?;
    }
    if let Some(path) = matches.get_one::<String>("toggle_output_json") {
        let activity = model.count_toggle_activity(&trace)?;
        write_toggle_activity_json(path, &activity)?;
    }
    Ok(())
}
