// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use clap::{Arg, ArgAction, ArgMatches, Command};
use xlsynth_g8r::liberty::io_defaults::default_output_load;
use xlsynth_g8r::liberty::load::load_library_with_timing_data_from_path;
use xlsynth_g8r::netlist::emit::emit_module_as_netlist_text;
use xlsynth_g8r::netlist::io::{read_gv_from_path, select_module};
use xlsynth_g8r::netlist::resize::{ResizeOptions, TrialTolerance, resize_netlist};
use xlsynth_g8r::netlist::sta::StaOptions;

/// Exposes bounded native resizing independently of the mapping frontend.
pub fn command() -> Command {
    Command::new("gv-resize")
        .about("Improve timing then recover area by resizing compatible logic cells and FFs; emit GV on stdout")
        .arg(Arg::new("netlist").long("netlist").required(true))
        .arg(Arg::new("liberty_proto").long("liberty_proto").required(true))
        .arg(Arg::new("module_name").long("module_name"))
        .arg(Arg::new("primary_input_transition").long("primary_input_transition").default_value("0.01").value_parser(clap::value_parser!(f64)))
        .arg(Arg::new("module_output_load").long("module_output_load").value_parser(clap::value_parser!(f64)))
        .arg(Arg::new("max-area-growth").long("max-area-growth").default_value("0.02").value_parser(clap::value_parser!(f64)))
        .arg(Arg::new("max-trials").long("max-trials").default_value("1024").value_parser(clap::value_parser!(usize)))
        .arg(Arg::new("max-propagated-instances").long("max-propagated-instances").default_value("100000").value_parser(clap::value_parser!(usize)))
        .arg(Arg::new("upsize-rounds").long("upsize-rounds").default_value("4").value_parser(clap::value_parser!(usize)))
        .arg(Arg::new("downsize-rounds").long("downsize-rounds").default_value("2").value_parser(clap::value_parser!(usize)))
        .arg(Arg::new("no-register-resize").long("no-register-resize").action(ArgAction::SetTrue))
        .arg(Arg::new("trial-arrival-tolerance").long("trial-arrival-tolerance").default_value("0").value_parser(clap::value_parser!(f64)))
        .arg(Arg::new("trial-slew-tolerance").long("trial-slew-tolerance").default_value("0").value_parser(clap::value_parser!(f64)))
        .arg(Arg::new("report-json").long("report-json").help("Write deterministic before/after timing, area and search-work statistics"))
}

/// Reads one flat mapped module and delegates all optimization to the library.
pub fn handle(matches: &ArgMatches) -> Result<(), String> {
    let library = load_library_with_timing_data_from_path(Path::new(
        matches.get_one::<String>("liberty_proto").unwrap(),
    ))
    .map_err(|e| e.to_string())?;
    let mut parsed = read_gv_from_path(Path::new(matches.get_one::<String>("netlist").unwrap()))
        .map_err(|e| e.to_string())?;
    let mut module = select_module(
        &parsed,
        matches.get_one::<String>("module_name").map(String::as_str),
    )
    .map_err(|e| e.to_string())?
    .clone();
    let output_load = match matches.get_one::<f64>("module_output_load") {
        Some(load) => *load,
        None => default_output_load(&library)?,
    };
    let options = ResizeOptions {
        timing: StaOptions {
            primary_input_transition: *matches.get_one::<f64>("primary_input_transition").unwrap(),
            module_output_load: output_load,
        },
        max_area_growth: *matches.get_one::<f64>("max-area-growth").unwrap(),
        max_trials: *matches.get_one::<usize>("max-trials").unwrap(),
        max_propagated_instances: *matches
            .get_one::<usize>("max-propagated-instances")
            .unwrap(),
        max_upsize_rounds: *matches.get_one::<usize>("upsize-rounds").unwrap(),
        max_downsize_rounds: *matches.get_one::<usize>("downsize-rounds").unwrap(),
        resize_registers: !matches.get_flag("no-register-resize"),
        trial_tolerance: TrialTolerance {
            arrival: *matches.get_one::<f64>("trial-arrival-tolerance").unwrap(),
            slew: *matches.get_one::<f64>("trial-slew-tolerance").unwrap(),
        },
        ..ResizeOptions::default()
    };
    let report = resize_netlist(
        &mut module,
        &parsed.nets,
        &mut parsed.interner,
        &library,
        &options,
    )
    .map_err(|e| e.to_string())?;
    let verilog = emit_module_as_netlist_text(&module, &parsed.nets, &parsed.interner)
        .map_err(|e| e.to_string())?;
    if let Some(path) = matches.get_one::<String>("report-json") {
        let json = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
        std::fs::write(path, format!("{json}\n"))
            .map_err(|e| format!("writing resizing report: {e}"))?;
    }
    print!("{verilog}");
    Ok(())
}
