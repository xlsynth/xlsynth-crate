// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use xlsynth_g8r::netlist::gv2block::convert_gv2block_paths_to_string;

pub fn handle_gv2block(matches: &clap::ArgMatches) {
    let netlist_path = matches.get_one::<String>("netlist").unwrap();
    let liberty_proto_path = matches.get_one::<String>("liberty_proto").unwrap();
    match convert_gv2block_paths_to_string(Path::new(netlist_path), Path::new(liberty_proto_path)) {
        Ok(block_ir) => println!("{}", block_ir),
        Err(e) => {
            eprintln!("Failed to convert netlist to block IR: {:#}", e);
            std::process::exit(1);
        }
    }
}
