// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use xlsynth_g8r::netlist::gv2ir::convert_gv2ir_paths;

pub fn handle_gv2ir(matches: &clap::ArgMatches) {
    let netlist_path = matches.get_one::<String>("netlist").unwrap();
    let liberty_proto_path = matches.get_one::<String>("liberty_proto").unwrap();

    // Parse dff_cells flag
    let dff_cells: std::collections::HashSet<String> = matches
        .get_one::<String>("dff_cells")
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    // Optional formula to auto-classify DFF cells by output pin function string
    let dff_cell_formula: Option<String> = matches
        .get_one::<String>("dff_cell_formula")
        .map(|s| s.to_string());
    let dff_cell_invert_formula: Option<String> = matches
        .get_one::<String>("dff_cell_invert_formula")
        .map(|s| s.to_string());

    match convert_gv2ir_paths(
        Path::new(netlist_path),
        Path::new(liberty_proto_path),
        &dff_cells,
        dff_cell_formula.as_deref(),
        dff_cell_invert_formula.as_deref(),
    ) {
        Ok(ir_text) => println!("{}", ir_text),
        Err(e) => {
            eprintln!("Failed to convert netlist to IR: {:#}", e);
            std::process::exit(1);
        }
    }
}
