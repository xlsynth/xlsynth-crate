// SPDX-License-Identifier: Apache-2.0

//! Binary to convert a Liberty file to a proto representation.

use clap::Parser;
use prost::Message;
use prost_reflect::DescriptorPool;
use std::{fs::File, path::PathBuf};

// Import the generated proto code.
pub mod liberty_proto {
    include!(concat!(env!("OUT_DIR"), "/liberty.rs"));
}

use xlsynth_g8r::liberty::liberty_to_proto::parse_liberty_files_to_proto;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input Liberty files (.lib or .lib.gz), variadic
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    /// Output proto file (binary)
    #[arg(short, long, required = true)]
    output: PathBuf,
}

fn main() {
    let _ = env_logger::builder().try_init();
    let args = Args::parse();
    println!("Parsing and consolidating cells from all libraries...");
    let proto_lib = parse_liberty_files_to_proto(&args.inputs).unwrap();
    println!("Writing output to {}...", args.output.display());
    let output_path = args.output.display().to_string();
    if output_path.ends_with(".textproto") {
        println!("Writing as textproto...");
        // Use prost-reflect to get a dynamic message and text format
        let descriptor_pool = DescriptorPool::decode(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/liberty.bin"
        )) as &[u8])
        .unwrap();
        let msg_desc = descriptor_pool
            .get_message_by_name("liberty.Library")
            .unwrap();
        let mut buf = Vec::new();
        proto_lib.encode(&mut buf).unwrap();
        let dyn_msg = prost_reflect::DynamicMessage::decode(msg_desc, &*buf).unwrap();
        let textproto = dyn_msg.to_text_format();
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
        let mut lib = Library { cells: vec![] };
        lib.cells.push(Cell {
            area: 1.0,
            pins: vec![Pin {
                name: "A".to_string(),
                direction: PinDirection::Input as i32,
                function: "A".to_string(),
                is_clocking_pin: false,
            }],
            name: "test_cell".to_string(),
            sequential: vec![],
            clock_gate: None,
        });
        assert_eq!(lib.cells.len(), 1);
        assert_eq!(lib.cells[0].pins[0].direction, PinDirection::Input as i32);
    }
}
