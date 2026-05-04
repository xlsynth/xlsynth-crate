// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use serde::Serialize;
use std::path::Path;
use xlsynth_g8r::netlist::io::{
    load_liberty_with_timing_data_from_path, parse_netlist_from_path, ParsedNetlist,
};
use xlsynth_g8r::netlist::parse::{NetlistModule, PortDirection};
use xlsynth_g8r::netlist::sta::{analyze_combinational_max_arrival, StaOptions};

#[derive(Debug, Serialize)]
struct OutputTimingRow {
    output: String,
    rise_arrival: f64,
    fall_arrival: f64,
    rise_transition: f64,
    fall_transition: f64,
    worst_arrival: f64,
}

#[derive(Debug, Serialize)]
struct StaSummary {
    module: String,
    time_unit: String,
    primary_input_transition: f64,
    module_output_load: f64,
    worst_output_arrival: f64,
    outputs: Vec<OutputTimingRow>,
}

fn resolve_symbol(
    parsed: &ParsedNetlist,
    sym: xlsynth_g8r::netlist::parse::PortId,
    what: &str,
) -> Result<String, String> {
    parsed
        .interner
        .resolve(sym)
        .map(|s| s.to_string())
        .ok_or_else(|| format!("could not resolve {} symbol", what))
}

fn select_module<'a>(
    parsed: &'a ParsedNetlist,
    module_name: Option<&str>,
) -> Result<&'a NetlistModule, String> {
    if let Some(name) = module_name {
        for module in &parsed.modules {
            let m_name = resolve_symbol(parsed, module.name, "module name")?;
            if m_name == name {
                return Ok(module);
            }
        }
        return Err(format!("module '{}' was not found in netlist", name));
    }

    if parsed.modules.len() == 1 {
        return Ok(&parsed.modules[0]);
    }

    let mut names: Vec<String> = Vec::with_capacity(parsed.modules.len());
    for module in &parsed.modules {
        names.push(resolve_symbol(parsed, module.name, "module name")?);
    }
    names.sort();
    Err(format!(
        "netlist contains {} modules; use --module_name; available modules: [{}]",
        parsed.modules.len(),
        names.join(", ")
    ))
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

    let module_name_text = match resolve_symbol(&parsed, module.name, "module name") {
        Ok(s) => s,
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

    let report = match analyze_combinational_max_arrival(
        module,
        parsed.nets.as_slice(),
        &parsed.interner,
        &liberty,
        StaOptions {
            primary_input_transition,
            module_output_load,
        },
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("gv-sta error: STA failed: {:#}", e);
            std::process::exit(1);
        }
    };

    let mut outputs: Vec<OutputTimingRow> = Vec::new();
    for port in &module.ports {
        if port.direction != PortDirection::Output {
            continue;
        }

        let output_name = match resolve_symbol(&parsed, port.name, "output port") {
            Ok(s) => s,
            Err(e) => {
                eprintln!("gv-sta error: {}", e);
                std::process::exit(1);
            }
        };

        let Some(net_idx) = module.find_net_index(port.name, parsed.nets.as_slice()) else {
            eprintln!(
                "gv-sta error: output '{}' does not resolve to a net",
                output_name
            );
            std::process::exit(1);
        };

        let Some(timing) = report.timing_for_net(net_idx) else {
            eprintln!(
                "gv-sta error: output '{}' has no computed timing",
                output_name
            );
            std::process::exit(1);
        };

        outputs.push(OutputTimingRow {
            output: output_name,
            rise_arrival: timing.rise.arrival,
            fall_arrival: timing.fall.arrival,
            rise_transition: timing.rise.transition,
            fall_transition: timing.fall.transition,
            worst_arrival: timing.rise.arrival.max(timing.fall.arrival),
        });
    }

    let time_unit = liberty
        .units
        .as_ref()
        .map(|u| u.time_unit.as_str())
        .unwrap_or("")
        .to_string();

    let summary = StaSummary {
        module: module_name_text,
        time_unit,
        primary_input_transition,
        module_output_load,
        worst_output_arrival: report.worst_output_arrival,
        outputs,
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
