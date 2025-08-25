// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use std::path::Path;
use xlsynth_g8r::netlist::stats;

pub fn handle_gv_read_stats(matches: &ArgMatches) {
    let netlist_path = matches
        .get_one::<String>("netlist")
        .expect("netlist path is required");
    match stats::read_netlist_stats(Path::new(netlist_path)) {
        Ok(s) => {
            println!("Instances: {}", s.num_instances);
            println!("Nets: {}", s.num_nets);
            let approx_mib = (s.memory_bytes as f64) / (1024.0 * 1024.0);
            println!("Approx. memory: {:.2} MiB", approx_mib);
            println!("Parse time: {} ms", s.parse_duration.as_millis());
            println!("Cell counts:");
            for (cell, count) in s.cell_counts {
                println!("  {cell}: {count}");
            }
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}
