// SPDX-License-Identifier: Apache-2.0

//! Implements the `gv-instance-csv` driver subcommand: emit all
//! instance/cell-type pairs in a netlist as csv.gz.

use clap::ArgMatches;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use xlsynth_g8r::netlist::io::parse_netlist_from_path;
use xlsynth_g8r::netlist::utils::module_instance_names_and_types;

pub fn do_gv_instance_csv(opts: &ArgMatches) -> Result<(), String> {
    let input_path = opts
        .get_one::<String>("input")
        .ok_or("--input argument is required")?;
    let output_path = opts
        .get_one::<String>("output")
        .ok_or("--output argument is required")?;
    // Open and parse the gate-level netlist (supports plain and .gz inputs)
    // using the shared netlist I/O helper for consistent error reporting.
    let parsed = parse_netlist_from_path(Path::new(input_path))
        .map_err(|e| format!("failed to parse gate-level netlist: {e}"))?;
    // Extract (module_name, instance_name, cell_type) triples.
    let triples = module_instance_names_and_types(&parsed.modules, &parsed.interner);
    // Open .csv.gz output stream
    let outf = File::create(output_path).map_err(|e| format!("failed to open output: {e}"))?;
    let enc = GzEncoder::new(BufWriter::new(outf), Compression::default());
    let mut wtr = csv::Writer::from_writer(enc);
    // Header row.
    wtr.write_record(["module_name", "instance_name", "cell_type"])
        .map_err(|e| format!("failed to write csv header: {e}"))?;
    for (module_name, instance_name, cell_type) in triples {
        wtr.write_record([module_name, instance_name, cell_type])
            .map_err(|e| format!("failed to write csv: {e}"))?;
    }
    wtr.flush()
        .map_err(|e| format!("failed to flush csv: {e}"))?;
    Ok(())
}
