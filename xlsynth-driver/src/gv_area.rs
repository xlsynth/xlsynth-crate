// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use std::path::Path;
use xlsynth_g8r::netlist::io::{load_liberty_from_path, parse_netlist_from_path};
use xlsynth_g8r::netlist::report::{build_area_report, select_module, NetlistAreaReport};

const SUBCOMMAND: &str = "gv-area";

fn render_area_report(report: &NetlistAreaReport) -> String {
    format!("module: {}\narea: {:.6}\n", report.module, report.area)
}

pub fn handle_gv_area(matches: &ArgMatches) {
    let netlist_path = Path::new(matches.get_one::<String>("netlist").unwrap());
    let liberty_proto_path = Path::new(matches.get_one::<String>("liberty_proto").unwrap());
    let module_name = matches.get_one::<String>("module_name").map(|s| s.as_str());
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
    let liberty = load_liberty_from_path(liberty_proto_path).unwrap_or_else(|e| {
        eprintln!(
            "{} error: failed to load Liberty proto '{}': {:#}",
            SUBCOMMAND,
            liberty_proto_path.display(),
            e
        );
        std::process::exit(1)
    });
    let report =
        build_area_report(module, &parsed.interner, liberty.as_proto()).unwrap_or_else(|e| {
            eprintln!("{} error: failed to compute cell area: {:#}", SUBCOMMAND, e);
            std::process::exit(1)
        });

    print!("{}", render_area_report(&report));
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
    use super::render_area_report;
    use xlsynth_g8r::netlist::report::NetlistAreaReport;

    #[test]
    fn render_area_report_is_stable() {
        let report = NetlistAreaReport {
            module: "top".to_string(),
            cell_count: 3,
            area: 4.0,
            cells: Vec::new(),
        };
        assert_eq!(render_area_report(&report), "module: top\narea: 4.000000\n");
    }
}
