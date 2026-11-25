// SPDX-License-Identifier: Apache-2.0

use std::{io::Write, path::Path};

use clap::ArgMatches;
use serde::Deserialize;
use xlsynth_g8r::liberty_model::Library;
use xlsynth_g8r::netlist::gv_eval::{
    GvEvalOptions, load_labeled_netlist_aig_with_liberty,
    load_labeled_sequential_netlist_aig_with_liberty, load_timed_netlist_with_liberty,
};
use xlsynth_g8r::netlist::io::load_liberty_from_path;
use xlsynth_g8r::netlist::power::timed::{TimedInputEvent, TimedNetlistEvaluator};
use xlsynth_g8r::netlist::power::{GvDynamicPowerOptions, GvSequentialDynamicPowerOptions};
use xlsynth_pir::{IrValue, parse_ir_values_file};

use crate::sequential_eval_io::{
    output_value, read_external_inputs, resolve_initial_state, write_final_state,
};

/// Combinational timed stimulus as supplied by the CLI.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TimedStimulus {
    initial_inputs: String,
    input_events: Vec<TimedInputEvent>,
    duration: f64,
}

fn write_report_json<T: serde::Serialize>(
    flag: &str,
    path: &str,
    report: &T,
) -> Result<(), String> {
    let file =
        std::fs::File::create(path).map_err(|e| format!("failed to create {flag} {path}: {e}"))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, report)
        .map_err(|e| format!("failed to write {flag} {path}: {e}"))?;
    writeln!(writer)
        .and_then(|()| writer.flush())
        .map_err(|e| format!("failed to finalize {flag} {path}: {e}"))
}

fn dynamic_power_options(
    matches: &ArgMatches,
    library: &Library,
) -> Result<GvDynamicPowerOptions, String> {
    Ok(GvDynamicPowerOptions {
        primary_input_transition: *matches
            .get_one::<f64>("primary_input_transition")
            .expect("primary_input_transition has a default"),
        module_output_load: *matches
            .get_one::<f64>("module_output_load")
            .expect("module_output_load has a default"),
        cycle_time: matches.get_one::<f64>("cycle_time").copied(),
        wire_load: crate::gv_sta::wire_load_from_matches(matches, library)
            .map_err(|e| e.to_string())?,
    })
}

fn sequential_dynamic_power_options(
    matches: &ArgMatches,
    library: &Library,
) -> Result<GvSequentialDynamicPowerOptions, String> {
    let base = dynamic_power_options(matches, library)?;
    Ok(GvSequentialDynamicPowerOptions {
        primary_input_transition: base.primary_input_transition,
        clock_transition: matches
            .get_one::<f64>("clock_transition")
            .copied()
            .unwrap_or(base.primary_input_transition),
        module_output_load: base.module_output_load,
        cycle_time: base.cycle_time,
        wire_load: base.wire_load,
    })
}

pub fn handle_gv_eval(matches: &ArgMatches) -> Result<(), String> {
    let netlist_path = matches
        .get_one::<String>("netlist")
        .expect("netlist is required by clap");
    let liberty_proto_path = matches
        .get_one::<String>("liberty_proto")
        .expect("liberty_proto is required by clap");
    let library = load_liberty_from_path(Path::new(liberty_proto_path))
        .map_err(|error| format!("failed to load Liberty model: {error:#}"))?;
    let netlist_path = Path::new(netlist_path);
    let options = GvEvalOptions {
        module_name: matches.get_one::<String>("module_name").cloned(),
        clock_port_name: matches.get_one::<String>("clock_port_name").cloned(),
    };
    if matches.get_flag("sequential") {
        return handle_sequential_gv_eval(matches, netlist_path, &library, &options);
    }

    if let Some(path) = matches.get_one::<String>("timed_stimulus_json") {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("failed to open --timed-stimulus-json {path}: {e}"))?;
        let stimulus: TimedStimulus = serde_json::from_reader(std::io::BufReader::new(file))
            .map_err(|e| format!("failed to parse --timed-stimulus-json {path}: {e}"))?;
        let initial = IrValue::parse_typed(&stimulus.initial_inputs)
            .map_err(|e| format!("failed to parse timed initial_inputs: {e}"))?;
        let timed = load_timed_netlist_with_liberty(netlist_path, &library, &options)
            .map_err(|e| format!("failed to build timed evaluation model: {e:#}"))?;
        let options = dynamic_power_options(matches, &library)?;
        let evaluator = TimedNetlistEvaluator::new(
            &timed,
            &library,
            options.module_output_load,
            options.wire_load,
        )?;
        let report = evaluator.simulate(&initial, &stimulus.input_events, stimulus.duration)?;
        let report_path = matches
            .get_one::<String>("timed_output_json")
            .expect("timed stimulus requires a report path");
        return write_report_json("--timed-output-json", report_path, &report);
    }

    let model = load_labeled_netlist_aig_with_liberty(netlist_path, &library, &options)
        .map_err(|e| format!("failed to build evaluation model: {e:#}"))?;
    let samples = if let Some(arg_tuple) = matches.get_one::<String>("arg_tuple") {
        vec![
            IrValue::parse_typed(arg_tuple)
                .map_err(|e| format!("failed to parse argument tuple: {e}"))?,
        ]
    } else {
        let input_irvals = matches
            .get_one::<String>("input_irvals")
            .expect("settled evaluation requires arg_tuple or input_irvals");
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
        write_report_json("--toggle-output-json", toggle_output_json, &activity)?;
    }
    if let Some(power_output_json) = matches.get_one::<String>("power_output_json") {
        let power_options = dynamic_power_options(matches, &library)?;
        let report = model.analyze_dynamic_power(&library, &samples, power_options)?;
        write_report_json("--power-output-json", power_output_json, &report)?;
    }
    Ok(())
}

fn handle_sequential_gv_eval(
    matches: &ArgMatches,
    netlist_path: &Path,
    library: &Library,
    options: &GvEvalOptions,
) -> Result<(), String> {
    let model = load_labeled_sequential_netlist_aig_with_liberty(netlist_path, library, options)
        .map_err(|error| format!("failed to build sequential evaluation model: {error:#}"))?;
    let design = &model.sequential_gate_fn;
    let inputs = read_external_inputs(matches, design)?;
    if matches.get_one::<String>("power_output_json").is_some() && design.clock.is_none() {
        return Err("--power-output-json requires a selected clock in sequential mode".to_string());
    }
    if matches.get_one::<String>("toggle_output_json").is_some()
        && design.clock.is_none()
        && inputs.len() < 2
    {
        return Err(format!(
            "--toggle-output-json requires at least two --input-irvals cycles; got {}",
            inputs.len()
        ));
    }
    let initial_state = resolve_initial_state(matches, design, "mapped netlist")?;
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
        write_report_json("--toggle-output-json", path, &activity)?;
    }
    if let Some(path) = matches.get_one::<String>("power_output_json") {
        let report = model.analyze_dynamic_power(
            library,
            &trace,
            sequential_dynamic_power_options(matches, library)?,
        )?;
        write_report_json("--power-output-json", path, &report)?;
    }
    Ok(())
}
