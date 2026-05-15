// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use xlsynth_g8r::netlist::gv2ir::{convert_gv2ir_paths_with_options, Gv2IrOptions};

pub fn handle_gv2ir(matches: &clap::ArgMatches) {
    let netlist_path = matches.get_one::<String>("netlist").unwrap();
    let liberty_proto_path = matches.get_one::<String>("liberty_proto").unwrap();

    let collapse_sequential = matches
        .get_one::<bool>("collapse_sequential")
        .copied()
        .unwrap_or(true);

    let opts = Gv2IrOptions {
        module_name: matches.get_one::<String>("module_name").cloned(),
        collapse_sequential,
        output_function_name: matches.get_one::<String>("output_function_name").cloned(),
    };

    match convert_gv2ir_paths_with_options(
        Path::new(netlist_path),
        Path::new(liberty_proto_path),
        &opts,
    ) {
        Ok(ir_text) => println!("{}", ir_text),
        Err(e) => {
            eprintln!("Failed to convert netlist to IR: {:#}", e);
            std::process::exit(1);
        }
    }
}
