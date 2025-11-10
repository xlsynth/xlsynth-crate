// SPDX-License-Identifier: Apache-2.0

//! Implements the `gv-instance-csv` driver subcommand: emit all
//! instance/cell-type pairs in a netlist as csv.gz.

use clap::ArgMatches;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use xlsynth_g8r::netlist::parse::{Parser, TokenScanner};
use xlsynth_g8r::netlist::utils::instance_names_and_types;

pub fn do_gv_instance_csv(opts: &ArgMatches) -> Result<(), String> {
    let input_path = opts
        .get_one::<String>("input")
        .ok_or("--input argument is required")?;
    let output_path = opts
        .get_one::<String>("output")
        .ok_or("--output argument is required")?;
    // Open and parse the gate-level netlist.
    let f = File::open(input_path).map_err(|e| format!("failed to open input: {e}"))?;
    let scanner =
        TokenScanner::from_file_with_path(std::io::BufReader::new(f), PathBuf::from(input_path));
    let mut parser = Parser::new(scanner);
    let modules = parser
        .parse_file()
        .map_err(|e| format!("parse error: {e:?}"))?;
    // Extract (instance_name, cell_type) pairs.
    let pairs = instance_names_and_types(&modules, &parser.interner);
    // Open .csv.gz output stream
    let outf = File::create(output_path).map_err(|e| format!("failed to open output: {e}"))?;
    let enc = GzEncoder::new(BufWriter::new(outf), Compression::default());
    let mut wtr = csv::Writer::from_writer(enc);
    for (instance_name, cell_type) in pairs {
        wtr.write_record([instance_name, cell_type])
            .map_err(|e| format!("failed to write csv: {e}"))?;
    }
    wtr.flush()
        .map_err(|e| format!("failed to flush csv: {e}"))?;
    Ok(())
}
