// SPDX-License-Identifier: Apache-2.0

//! Binary to parse a gate-level netlist and project it through a Liberty proto
//! to create a GateFn.

use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;
use xlsynth_g8r::liberty::cell_formula::{self, Term};

// Use the crate's prost-generated proto module
use xlsynth_g8r::liberty_model;

use xlsynth_g8r::aig_serdes::gate2ir::gate_fn_to_pir;
use xlsynth_g8r::netlist::io::{load_liberty_from_path, read_gv_from_path};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input gate-level netlist in .gv format
    netlist: PathBuf,
    /// Input Liberty proto file (binary)
    liberty_proto: PathBuf,
}

fn load_cell_formula_map(liberty_lib: &liberty_model::Library) -> HashMap<String, Term> {
    use crate::liberty_model::PinDirection;
    let mut map = HashMap::new();
    for cell in &liberty_lib.cells {
        // Find the output pin with a function
        if let Some(pin) = cell.pins.iter().find(|p| {
            p.direction == PinDirection::Output as i32
                && !liberty_lib.resolve_string(&p.function).is_empty()
        }) {
            let function = liberty_lib.resolve_string(&pin.function);
            match cell_formula::parse_formula(function) {
                Ok(term) => {
                    map.insert(cell.name.clone(), term);
                }
                Err(e) => {
                    eprintln!(
                        "Failed to parse formula for cell '{}': {}\n  formula: {}",
                        cell.name, e, function
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
    let parsed = read_gv_from_path(&args.netlist).unwrap_or_else(|error| {
        eprintln!("{error:#}");
        std::process::exit(1);
    });
    // Read and decode Liberty once, skipping timing payloads by default.
    let liberty_lib = load_liberty_from_path(&args.liberty_proto).expect("failed to load liberty");
    // Log all net names and indices
    for (i, net) in parsed.nets.iter().enumerate() {
        let net_name = parsed.interner.resolve(net.name).unwrap();
        log::info!("Net[{}]: {} width={:?}", i, net_name, net.width);
    }
    let total_instances: usize = parsed.modules.iter().map(|m| m.instances.len()).sum();
    println!(
        "Parsed {} modules, total {} instances.",
        parsed.modules.len(),
        total_instances
    );
    // Load Liberty proto and build cell formula map
    let cell_formula_map = load_cell_formula_map(&liberty_lib);
    println!(
        "Loaded {} cell formulas from Liberty proto.",
        cell_formula_map.len()
    );
    if parsed.modules.len() != 1 {
        eprintln!(
            "Error: Only single-module netlists are supported (got {}).",
            parsed.modules.len()
        );
        std::process::exit(1);
    }
    let module = &parsed.modules[0];
    let gate_fn =
        xlsynth_g8r::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty(
            module,
            &parsed.nets,
            &parsed.interner,
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
    let ir_pkg = gate_fn_to_pir(&gate_fn, "gate", &flat_type).unwrap();
    println!("XLS IR:\n{}", ir_pkg.to_string());
    println!("Done parsing netlist.");
}
