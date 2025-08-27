// SPDX-License-Identifier: Apache-2.0

//! Binary to parse a gate-level netlist and project it through a Liberty proto
//! to create a GateFn.

use clap::Parser;
use std::collections::HashMap;
use std::io::Read;
use std::{fs::File, path::PathBuf};
use xlsynth_g8r::liberty::cell_formula::{self, Term};

// Use the crate's prost-generated proto module
use xlsynth_g8r::liberty_proto;

use prost::Message;
use xlsynth_g8r::gate2ir::gate_fn_to_xlsynth_ir;
use xlsynth_g8r::netlist::parse::{Parser as NetlistParser, TokenScanner};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input gate-level netlist in .gv format
    netlist: PathBuf,
    /// Input Liberty proto file (binary)
    liberty_proto: PathBuf,
}

fn load_cell_formula_map(liberty_lib: &liberty_proto::Library) -> HashMap<String, Term> {
    use crate::liberty_proto::PinDirection;
    let mut map = HashMap::new();
    for cell in &liberty_lib.cells {
        // Find the output pin with a function
        if let Some(pin) = cell
            .pins
            .iter()
            .find(|p| p.direction == PinDirection::Output as i32 && !p.function.is_empty())
        {
            match cell_formula::parse_formula(&pin.function) {
                Ok(term) => {
                    map.insert(cell.name.clone(), term);
                }
                Err(e) => {
                    eprintln!(
                        "Failed to parse formula for cell '{}': {}\n  formula: {}",
                        cell.name, e, pin.function
                    );
                }
            }
        }
    }
    map
}

fn main() {
    let _ = env_logger::builder().try_init();
    let args = Args::parse();
    println!("Reading netlist from {}...", args.netlist.display());
    let file = File::open(&args.netlist).unwrap();
    let scanner = TokenScanner::from_file_with_path(file, args.netlist.clone());
    let mut parser = NetlistParser::new(scanner);
    // Read and decode Liberty proto once
    let mut liberty_file = File::open(&args.liberty_proto).expect("failed to open liberty proto");
    let mut buf = Vec::new();
    liberty_file
        .read_to_end(&mut buf)
        .expect("failed to read liberty proto");
    let liberty_lib =
        liberty_proto::Library::decode(&*buf).expect("failed to decode liberty proto");
    let modules = match parser.parse_file() {
        Ok(m) => m,
        Err(e) => {
            // Print error message, line context, and caret
            eprintln!("parse error: {} at {:?}", e.message, e.span);
            // Use the public get_line method from the parser
            let line = parser
                .get_line(e.span.start.lineno)
                .unwrap_or_else(|| "<line unavailable>".to_string());
            let col = (e.span.start.colno as usize).saturating_sub(1);
            eprintln!("{}", line);
            eprintln!("{}^", " ".repeat(col));
            std::process::exit(1);
        }
    };
    // Log all net names and indices
    for (i, net) in parser.nets.iter().enumerate() {
        let net_name = parser.interner.resolve(net.name).unwrap();
        log::info!("Net[{}]: {} width={:?}", i, net_name, net.width);
    }
    let total_instances: usize = modules.iter().map(|m| m.instances.len()).sum();
    println!(
        "Parsed {} modules, total {} instances.",
        modules.len(),
        total_instances
    );
    // Load Liberty proto and build cell formula map
    let cell_formula_map = load_cell_formula_map(&liberty_lib);
    println!(
        "Loaded {} cell formulas from Liberty proto.",
        cell_formula_map.len()
    );
    if modules.len() != 1 {
        eprintln!(
            "Error: Only single-module netlists are supported (got {}).",
            modules.len()
        );
        std::process::exit(1);
    }
    let module = &modules[0];
    let gate_fn =
        xlsynth_g8r::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty(
            module,
            &parser.nets,
            &parser.interner,
            &liberty_lib,
            &std::collections::HashSet::new(),
            &std::collections::HashSet::new(),
        )
        .unwrap_or_else(|e| {
            eprintln!("Error during netlist projection: {}", e);
            std::process::exit(1);
        });
    println!("GateFn:\n{}", gate_fn.to_string());
    // Convert to XLS IR and print
    let flat_type = gate_fn.get_flat_type();
    let ir_pkg = gate_fn_to_xlsynth_ir(&gate_fn, "gate", &flat_type).unwrap();
    println!("XLS IR:\n{}", ir_pkg.to_string());
    println!("Done parsing netlist.");
}
