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

    match convert_gv2ir_paths(
        Path::new(netlist_path),
        Path::new(liberty_proto_path),
        &dff_cells,
    ) {
        Ok(ir_text) => println!("{}", ir_text),
        Err(e) => {
            eprintln!("Failed to convert netlist to IR: {}", e);
            std::process::exit(1);
        }
    }
}
