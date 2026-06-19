// SPDX-License-Identifier: Apache-2.0

//! Binary to convert a Liberty file to a proto representation.

use clap::Parser;
use prost::Message;
use std::{fs::File, path::PathBuf};

// Import the generated proto code.
pub mod liberty_proto {
    include!(concat!(env!("OUT_DIR"), "/liberty.rs"));
}

use xlsynth_g8r::liberty::descriptor::liberty_proto_bytes_to_pretty_textproto;
use xlsynth_g8r::liberty::model::{library_to_proto, strip_timing_data};
use xlsynth_g8r::liberty::parser::{
    parse_liberty_files, parse_liberty_files_without_timing_validation,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input Liberty files (.lib or .lib.gz), variadic
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    /// Output proto file (binary)
    #[arg(short, long, required = true)]
    output: PathBuf,
    /// If true, drop all timing payloads (timing arcs/templates) from output.
    #[arg(long, default_value_t = false)]
    no_timing_data: bool,
}

fn main() {
    let _ = env_logger::builder().try_init();
    let args = Args::parse();
    println!("Parsing and consolidating cells from all libraries...");
    let mut proto_lib = if args.no_timing_data {
        parse_liberty_files_without_timing_validation(&args.inputs).unwrap()
    } else {
        parse_liberty_files(&args.inputs).unwrap()
    };
    if args.no_timing_data {
        strip_timing_data(&mut proto_lib);
    }
    let proto_lib = library_to_proto(proto_lib).expect("encode Liberty LUT data");
    println!("Writing output to {}...", args.output.display());
    let output_path = args.output.display().to_string();
    if output_path.ends_with(".textproto") {
        println!("Writing as textproto...");
        let textproto = liberty_proto_bytes_to_pretty_textproto(&proto_lib.encode_to_vec())
            .expect("format Liberty textproto");
        std::fs::write(&args.output, textproto).unwrap();
    } else {
        println!("Writing as binary proto...");
        let mut out_file = File::create(&args.output).unwrap();
        let mut buf = Vec::new();
        proto_lib.encode(&mut buf).unwrap();
        std::io::Write::write_all(&mut out_file, &buf).unwrap();
    }
    println!("Done! Wrote {} cells.", proto_lib.cells.len());
}

#[cfg(test)]
mod tests {
    use crate::liberty_proto::{Cell, Library, Pin, PinDirection};

    #[test]
    fn test_proto_structure() {
        let mut lib = Library {
            cells: vec![],
            ..Default::default()
        };
        lib.cells.push(Cell {
            area: 1.0,
            pins: vec![Pin {
                name: "A".to_string(),
                direction: PinDirection::Input as i32,
                function: "A".to_string(),
                is_clocking_pin: false,
                ..Default::default()
            }],
            name: "test_cell".to_string(),
            sequential: vec![],
            clock_gate: None,
            ..Default::default()
        });
        assert_eq!(lib.cells.len(), 1);
        assert_eq!(lib.cells[0].pins[0].direction, PinDirection::Input as i32);
    }
}
