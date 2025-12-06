// SPDX-License-Identifier: Apache-2.0

use crate::report_cli_error::report_cli_error_and_exit;
use clap::ArgMatches;
use csv::WriterBuilder;
use std::collections::HashSet;
use std::io::BufWriter;
use std::path::Path;
use xlsynth_g8r::netlist::cone::{
    ConeError, ConeVisit, StopCondition, TraversalDirection,
};

fn parse_traversal_direction(dir: &str) -> Result<TraversalDirection, String> {
    match dir {
        "fanin" => Ok(TraversalDirection::Fanin),
        "fanout" => Ok(TraversalDirection::Fanout),
        other => Err(format!(
            "invalid --traverse value '{}'; expected 'fanin' or 'fanout'",
            other
        )),
    }
}

fn stop_condition_from_matches(matches: &ArgMatches) -> Result<StopCondition, String> {
    let levels = matches.get_one::<String>("stop-at-levels");
    let at_dff = matches.get_flag("stop-at-dff");
    let at_block_port = matches.get_flag("stop-at-block-port");

    let mut count = 0;
    if levels.is_some() {
        count += 1;
    }
    if at_dff {
        count += 1;
    }
    if at_block_port {
        count += 1;
    }
    if count != 1 {
        return Err(
            "exactly one of --stop-at-levels, --stop-at-dff, or --stop-at-block-port must be specified"
                .to_string(),
        );
    }

    if let Some(s) = levels {
        let n: u32 = s
            .parse()
            .map_err(|e| format!("invalid --stop-at-levels value '{}': {}", s, e))?;
        return Ok(StopCondition::Levels(n));
    }
    if at_dff {
        return Ok(StopCondition::AtDff);
    }
    if at_block_port {
        return Ok(StopCondition::AtBlockPort);
    }

    Err("internal error: no stop condition selected".to_string())
}

fn cone_error_to_report_message(err: ConeError) -> (String, Vec<(&'static str, String)>) {
    match err {
        ConeError::MissingInstance { name } => (
            "start instance not found in module".to_string(),
            vec![("instance", name)],
        ),
        ConeError::AmbiguousInstance { name, count } => (
            "start instance is ambiguous in module".to_string(),
            vec![
                ("instance", name),
                ("count", format!("{}", count)),
            ],
        ),
        ConeError::UnknownCellType { cell } => (
            "cell type from netlist is missing in Liberty library".to_string(),
            vec![("cell_type", cell)],
        ),
        ConeError::UnknownCellPin { cell, pin } => (
            "cell pin from netlist is missing in Liberty library".to_string(),
            vec![
                ("cell_type", cell),
                ("pin", pin),
            ],
        ),
        ConeError::NoModulesParsed { path } => (
            "no modules parsed from netlist".to_string(),
            vec![("netlist", path)],
        ),
        ConeError::ModuleNotFound { name } => (
            "requested module name was not found in netlist".to_string(),
            vec![("module_name", name)],
        ),
        ConeError::NetlistParse(msg) => (
            "failed to parse gate-level netlist".to_string(),
            vec![("detail", msg)],
        ),
        ConeError::Liberty(msg) => (
            "failed to parse Liberty proto".to_string(),
            vec![("detail", msg)],
        ),
        ConeError::Invariant(msg) => (
            "cone traversal invariant failed".to_string(),
            vec![("detail", msg)],
        ),
    }
}

pub fn handle_gv_dump_cone(matches: &ArgMatches) {
    let netlist_path = matches
        .get_one::<String>("netlist")
        .expect("netlist path is required");
    let liberty_proto_path = matches
        .get_one::<String>("liberty_proto")
        .expect("liberty_proto is required");
    let instance_name = matches
        .get_one::<String>("instance")
        .expect("instance name is required");

    let module_name = matches.get_one::<String>("module_name").map(|s| s.as_str());

    let traverse_str = matches
        .get_one::<String>("traverse")
        .expect("--traverse is required");
    let direction = match parse_traversal_direction(traverse_str) {
        Ok(d) => d,
        Err(msg) => {
            report_cli_error_and_exit(&msg, None, vec![]);
        }
    };

    let stop = match stop_condition_from_matches(matches) {
        Ok(s) => s,
        Err(msg) => {
            report_cli_error_and_exit(&msg, None, vec![]);
        }
    };

    let dff_cells: HashSet<String> = matches
        .get_one::<String>("dff_cells")
        .map(|s| {
            s.split(',')
                .filter(|p| !p.is_empty())
                .map(|p| p.to_string())
                .collect()
        })
        .unwrap_or_default();

    if matches!(stop, StopCondition::AtDff) && dff_cells.is_empty() {
        report_cli_error_and_exit(
            "when using --stop-at-dff you must also provide --dff_cells with one or more DFF cell names",
            None,
            vec![],
        );
    }

    let start_pins: Option<Vec<String>> = matches
        .get_one::<String>("start-pins")
        .map(|s| s.split(',').map(|p| p.to_string()).collect());

    let stdout = std::io::stdout();
    let handle = stdout.lock();
    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .from_writer(BufWriter::new(handle));

    // Header row.
    if let Err(e) = wtr.write_record(["instance_type", "instance_name", "traversal_pin"]) {
        report_cli_error_and_exit(
            "failed to write CSV header to stdout",
            Some(&format!("{}", e)),
            vec![],
        );
    }

    let visit_result = xlsynth_g8r::netlist::cone::visit_cone_from_paths(
        Path::new(netlist_path),
        Path::new(liberty_proto_path),
        module_name,
        instance_name.as_str(),
        start_pins.as_ref().map(|v| v.as_slice()),
        direction,
        stop,
        &dff_cells,
        |ConeVisit {
             instance_type,
             instance_name,
             traversal_pin,
         }| {
            wtr.write_record([instance_type, instance_name, traversal_pin])
                .map_err(|e| ConeError::Invariant(format!("failed to write CSV record: {}", e)))
        },
    );

    if let Err(err) = visit_result {
        let (msg, details) = cone_error_to_report_message(err);
        let mut kvs: Vec<(&str, &str)> = Vec::new();
        for (k, v) in &details {
            kvs.push((*k, v.as_str()));
        }
        report_cli_error_and_exit(&msg, None, kvs);
    }

    if let Err(e) = wtr.flush() {
        report_cli_error_and_exit(
            "failed to flush CSV output to stdout",
            Some(&format!("{}", e)),
            vec![],
        );
    }
}
