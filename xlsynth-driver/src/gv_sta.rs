// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use serde::Serialize;
use std::path::Path;
use xlsynth_g8r::netlist::io::{load_liberty_with_timing_data_from_path, parse_netlist_from_path};
use xlsynth_g8r::netlist::report::{
    build_sta_report, select_module, NetlistStaReport, OutputTimingRow,
};
use xlsynth_g8r::netlist::sta::StaOptions;

#[derive(Debug, Serialize)]
struct StaSummary {
    module: String,
    time_unit: String,
    primary_input_transition: f64,
    module_output_load: f64,
    worst_output_arrival: f64,
    outputs: Vec<OutputTimingRow>,
}

impl From<NetlistStaReport> for StaSummary {
    fn from(value: NetlistStaReport) -> Self {
        Self {
            module: value.module,
            time_unit: value.time_unit,
            primary_input_transition: value.primary_input_transition,
            module_output_load: value.module_output_load,
            worst_output_arrival: value.delay,
            outputs: value.outputs,
        }
    }
}

pub fn handle_gv_sta(matches: &ArgMatches) {
    let netlist_path = matches
        .get_one::<String>("netlist")
        .expect("netlist is required");
    let liberty_proto_path = matches
        .get_one::<String>("liberty_proto")
        .expect("liberty_proto is required");
    let module_name = matches.get_one::<String>("module_name").map(|s| s.as_str());
    let primary_input_transition = *matches
        .get_one::<f64>("primary_input_transition")
        .expect("primary_input_transition has default");
    let module_output_load = *matches
        .get_one::<f64>("module_output_load")
        .expect("module_output_load has default");
    let json_out = matches.get_one::<String>("json_out");

    let parsed = match parse_netlist_from_path(Path::new(netlist_path)) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "gv-sta error: failed to parse netlist '{}': {:#}",
                netlist_path, e
            );
            std::process::exit(1);
        }
    };

    let module = match select_module(&parsed, module_name) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("gv-sta error: {}", e);
            std::process::exit(1);
        }
    };

    let liberty = match load_liberty_with_timing_data_from_path(Path::new(liberty_proto_path)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!(
                "gv-sta error: failed to load timing-enabled Liberty proto '{}': {:#}",
                liberty_proto_path, e
            );
            std::process::exit(1);
        }
    };

    let summary = match build_sta_report(
        module,
        parsed.nets.as_slice(),
        &parsed.interner,
        &liberty,
        StaOptions {
            primary_input_transition,
            module_output_load,
        },
    ) {
        Ok(r) => StaSummary::from(r),
        Err(e) => {
            eprintln!("gv-sta error: STA failed: {:#}", e);
            std::process::exit(1);
        }
    };

    let shown_time_unit = if summary.time_unit.is_empty() {
        "<unspecified>"
    } else {
        summary.time_unit.as_str()
    };

    println!("module: {}", summary.module);
    println!("time_unit: {}", shown_time_unit);
    println!(
        "primary_input_transition: {:.6}",
        summary.primary_input_transition
    );
    println!("module_output_load: {:.6}", summary.module_output_load);
    println!("worst_output_arrival: {:.6}", summary.worst_output_arrival);
    for out in &summary.outputs {
        println!(
            "output {} rise_arrival={:.6} fall_arrival={:.6} rise_transition={:.6} fall_transition={:.6} worst_arrival={:.6}",
            out.output,
            out.rise_arrival,
            out.fall_arrival,
            out.rise_transition,
            out.fall_transition,
            out.worst_arrival
        );
    }

    if let Some(json_path) = json_out {
        if let Err(e) = std::fs::File::create(json_path)
            .map_err(|e| e.to_string())
            .and_then(|file| {
                serde_json::to_writer_pretty(file, &summary).map_err(|e| e.to_string())
            })
        {
            eprintln!(
                "gv-sta error: failed writing JSON summary to '{}': {}",
                json_path, e
            );
            std::process::exit(1);
        }
    }
}
