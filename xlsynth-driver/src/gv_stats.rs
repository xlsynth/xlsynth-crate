// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use std::path::Path;
use xlsynth_g8r::netlist::io::{load_liberty_with_timing_data_from_path, parse_netlist_from_path};
use xlsynth_g8r::netlist::report::{build_netlist_report, select_module, NetlistReport};
use xlsynth_g8r::netlist::sta::StaOptions;

const SUBCOMMAND: &str = "gv-stats";

fn shown_time_unit(time_unit: &str) -> &str {
    if time_unit.is_empty() {
        "<unspecified>"
    } else {
        time_unit
    }
}

fn render_netlist_report(report: &NetlistReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("module: {}\n", report.module));
    out.push_str(&format!(
        "time_unit: {}\n",
        shown_time_unit(&report.time_unit)
    ));
    out.push_str(&format!(
        "primary_input_transition: {:.6}\n",
        report.primary_input_transition
    ));
    out.push_str(&format!(
        "module_output_load: {:.6}\n",
        report.module_output_load
    ));
    out.push_str(&format!("area: {:.6}\n", report.area));
    out.push_str(&format!("delay: {:.6}\n", report.delay));
    out.push_str(&format!("cell_count: {}\n", report.cell_count));
    out.push_str(&format!("cell_levels: {}\n", report.cell_levels));
    out.push_str("cell_counts:\n");
    for cell in &report.cells {
        out.push_str(&format!(
            "  {} count={} cell_area={:.6} total_area={:.6}\n",
            cell.cell, cell.count, cell.cell_area, cell.total_area
        ));
    }
    for output in &report.outputs {
        out.push_str(&format!(
            "output {} rise_arrival={:.6} fall_arrival={:.6} rise_transition={:.6} fall_transition={:.6} worst_arrival={:.6}\n",
            output.output,
            output.rise_arrival,
            output.fall_arrival,
            output.rise_transition,
            output.fall_transition,
            output.worst_arrival
        ));
    }
    out
}

pub fn handle_gv_stats(matches: &ArgMatches) {
    let netlist_path = Path::new(matches.get_one::<String>("netlist").unwrap());
    let liberty_proto_path = Path::new(matches.get_one::<String>("liberty_proto").unwrap());
    let module_name = matches.get_one::<String>("module_name").map(|s| s.as_str());
    let primary_input_transition = *matches
        .get_one::<f64>("primary_input_transition")
        .expect("primary_input_transition has default");
    let module_output_load = *matches
        .get_one::<f64>("module_output_load")
        .expect("module_output_load has default");
    let json_out = matches.get_one::<String>("json_out");

    let parsed = parse_netlist_from_path(netlist_path).unwrap_or_else(|e| {
        eprintln!(
            "{} error: failed to parse netlist '{}': {:#}",
            SUBCOMMAND,
            netlist_path.display(),
            e
        );
        std::process::exit(1)
    });
    let module = select_module(&parsed, module_name).unwrap_or_else(|e| {
        eprintln!("{} error: {}", SUBCOMMAND, e);
        std::process::exit(1)
    });
    let liberty = load_liberty_with_timing_data_from_path(liberty_proto_path).unwrap_or_else(|e| {
        eprintln!(
            "{} error: failed to load timing-enabled Liberty proto '{}': {:#}",
            SUBCOMMAND,
            liberty_proto_path.display(),
            e
        );
        std::process::exit(1)
    });
    let report = build_netlist_report(
        module,
        parsed.nets.as_slice(),
        &parsed.interner,
        &liberty,
        StaOptions {
            primary_input_transition,
            module_output_load,
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("{} error: failed to compute report: {:#}", SUBCOMMAND, e);
        std::process::exit(1)
    });

    print!("{}", render_netlist_report(&report));
    if let Some(json_path) = json_out {
        std::fs::File::create(json_path)
            .map_err(|e| e.to_string())
            .and_then(|file| serde_json::to_writer_pretty(file, &report).map_err(|e| e.to_string()))
            .unwrap_or_else(|e| {
                eprintln!(
                    "{} error: failed writing JSON summary to '{}': {}",
                    SUBCOMMAND, json_path, e
                );
                std::process::exit(1)
            });
    }
}

#[cfg(test)]
mod tests {
    use super::render_netlist_report;
    use xlsynth_g8r::netlist::report::{CellAreaRow, NetlistReport, OutputTimingRow};

    #[test]
    fn render_netlist_report_is_stable() {
        let report = NetlistReport {
            module: "top".to_string(),
            time_unit: "1ps".to_string(),
            primary_input_transition: 0.01,
            module_output_load: 0.0,
            area: 4.0,
            delay: 3.0,
            cell_count: 3,
            cell_levels: 2,
            cells: vec![CellAreaRow {
                cell: "INV".to_string(),
                count: 2,
                cell_area: 1.0,
                total_area: 2.0,
            }],
            outputs: vec![OutputTimingRow {
                output: "y".to_string(),
                rise_arrival: 2.0,
                fall_arrival: 3.0,
                rise_transition: 0.2,
                fall_transition: 0.3,
                worst_arrival: 3.0,
            }],
        };
        assert!(render_netlist_report(&report).contains("cell_levels: 2\n"));
        assert!(render_netlist_report(&report).contains("delay: 3.000000\n"));
    }
}
