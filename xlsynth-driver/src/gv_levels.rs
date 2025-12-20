// SPDX-License-Identifier: Apache-2.0

use crate::report_cli_error::report_cli_error_and_exit;
use clap::ArgMatches;
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;
use xlsynth_g8r::liberty::IndexedLibrary;
use xlsynth_g8r::netlist;
use xlsynth_g8r::netlist::dff::classify_dff_cells;
use xlsynth_g8r::netlist::levels::{compute_levels, LevelsCategory};

fn parse_format(s: &str) -> Result<&'static str, String> {
    match s {
        "text" => Ok("text"),
        "csv" => Ok("csv"),
        other => Err(format!(
            "invalid --format value '{}'; expected 'text' or 'csv'",
            other
        )),
    }
}

fn parse_dff_cells_csv(s: &str) -> HashSet<String> {
    s.split(',')
        .filter(|p| !p.is_empty())
        .map(|p| p.to_string())
        .collect()
}

fn print_text_report(
    report: &xlsynth_g8r::netlist::levels::LevelsReport,
) -> Result<(), std::io::Error> {
    let mut out = String::new();
    out.push_str(&format!("Instances: {}\n", report.num_instances));
    out.push_str(&format!("DFF instances: {}\n", report.num_dff_instances));
    out.push_str(&format!("Output ports: {}\n", report.num_output_ports));

    let cats = [
        LevelsCategory::InputToReg,
        LevelsCategory::RegToReg,
        LevelsCategory::RegToOutput,
        LevelsCategory::InputToOutput,
    ];
    for cat in cats {
        out.push_str(&format!("\n{}:\n", cat.as_str()));
        if let Some(h) = report.histograms.get(&cat) {
            for (depth, count) in h {
                out.push_str(&format!("  {}: {}\n", depth, count));
            }
        }
    }
    std::io::stdout().write_all(out.as_bytes())
}

fn print_csv_report(
    report: &xlsynth_g8r::netlist::levels::LevelsReport,
) -> Result<(), std::io::Error> {
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(true)
        .from_writer(std::io::stdout());
    wtr.write_record(["category", "depth", "count"])?;

    let cats = [
        LevelsCategory::InputToReg,
        LevelsCategory::RegToReg,
        LevelsCategory::RegToOutput,
        LevelsCategory::InputToOutput,
    ];
    for cat in cats {
        if let Some(h) = report.histograms.get(&cat) {
            for (depth, count) in h {
                wtr.write_record([cat.as_str(), &format!("{}", depth), &format!("{}", count)])?;
            }
        }
    }
    wtr.flush()?;
    Ok(())
}

pub fn handle_gv_levels(matches: &ArgMatches) {
    let netlist_path = matches
        .get_one::<String>("netlist")
        .expect("netlist path is required");
    let liberty_proto_path = matches
        .get_one::<String>("liberty_proto")
        .expect("liberty_proto is required");
    let module_name = matches.get_one::<String>("module_name").map(|s| s.as_str());

    let dff_cells: HashSet<String> = matches
        .get_one::<String>("dff_cells")
        .map(|s| parse_dff_cells_csv(s.as_str()))
        .unwrap_or_default();

    let dff_cell_formula: Option<&str> = matches
        .get_one::<String>("dff_cell_formula")
        .map(|s| s.as_str());
    let dff_cell_invert_formula: Option<&str> = matches
        .get_one::<String>("dff_cell_invert_formula")
        .map(|s| s.as_str());

    if dff_cells.is_empty() && dff_cell_formula.is_none() && dff_cell_invert_formula.is_none() {
        report_cli_error_and_exit(
            "must specify at least one DFF classification input (e.g. --dff_cells or --dff_cell_formula)",
            None,
            vec![],
        );
    }

    let format_str = matches
        .get_one::<String>("format")
        .map(|s| s.as_str())
        .unwrap_or("text");
    let format = match parse_format(format_str) {
        Ok(f) => f,
        Err(msg) => report_cli_error_and_exit(&msg, None, vec![("format", format_str)]),
    };

    let parsed_netlist = match netlist::io::parse_netlist_from_path(Path::new(netlist_path)) {
        Ok(p) => p,
        Err(e) => {
            report_cli_error_and_exit(
                "failed to parse gate-level netlist",
                Some(&format!("{}", e)),
                vec![("netlist", netlist_path.as_str())],
            );
        }
    };

    let liberty_lib = match netlist::io::load_liberty_from_path(Path::new(liberty_proto_path)) {
        Ok(l) => l,
        Err(e) => {
            report_cli_error_and_exit(
                "failed to parse Liberty proto",
                Some(&format!("{}", e)),
                vec![("liberty_proto", liberty_proto_path.as_str())],
            );
        }
    };
    let indexed_lib = IndexedLibrary::new(liberty_lib.clone());

    let module_port_dirs =
        netlist::connectivity::build_module_port_directions(&parsed_netlist.modules);

    let module: &netlist::parse::NetlistModule =
        match netlist::io::select_module(&parsed_netlist, module_name) {
            Ok(m) => m,
            Err(e) => {
                let mut details: Vec<(&str, &str)> = Vec::new();
                if let Some(name) = module_name {
                    details.push(("module_name", name));
                }
                report_cli_error_and_exit(
                    "failed to select module from netlist",
                    Some(&format!("{}", e)),
                    details,
                );
            }
        };

    let dff = match classify_dff_cells(
        &liberty_lib,
        &dff_cells,
        dff_cell_formula,
        dff_cell_invert_formula,
    ) {
        Ok(d) => d,
        Err(e) => {
            report_cli_error_and_exit(
                "failed to classify DFF cells",
                Some(&format!("{}", e)),
                vec![],
            );
        }
    };
    let dff_cell_types = dff.all_cell_types();

    let report = match compute_levels(
        module,
        &parsed_netlist.nets,
        &parsed_netlist.interner,
        &indexed_lib,
        &dff_cell_types,
        Some(&module_port_dirs),
    ) {
        Ok(r) => r,
        Err(e) => report_cli_error_and_exit(
            "failed to compute gv-levels histogram",
            Some(&format!("{}", e)),
            vec![],
        ),
    };

    let print_result = if format == "csv" {
        print_csv_report(&report)
    } else {
        print_text_report(&report)
    };
    if let Err(e) = print_result {
        report_cli_error_and_exit("failed to write output", Some(&format!("{}", e)), vec![]);
    }
}
