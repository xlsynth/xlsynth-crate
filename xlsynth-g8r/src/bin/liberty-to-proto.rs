// SPDX-License-Identifier: Apache-2.0

//! Binary to convert a Liberty file to a proto representation.

use clap::Parser;
use flate2::bufread::GzDecoder;
use prost::Message;
use prost_reflect::{prost::Message as _, DescriptorPool};
use std::io::BufReader;
use std::{fs::File, path::PathBuf};

// Import the generated proto code.
pub mod liberty_proto {
    include!(concat!(env!("OUT_DIR"), "/liberty.rs"));
}

use liberty_proto::*;
use xlsynth_g8r::liberty::liberty_to_proto::parse_liberty_files_to_proto;
use xlsynth_g8r::liberty::{CharReader, LibertyParser};

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

fn value_to_f64(value: &xlsynth_g8r::liberty::liberty_parser::Value) -> f64 {
    match value {
        xlsynth_g8r::liberty::liberty_parser::Value::Number(n) => *n,
        _ => panic!("Expected number for area attribute"),
    }
}

fn value_to_string(value: &xlsynth_g8r::liberty::liberty_parser::Value) -> String {
    match value {
        xlsynth_g8r::liberty::liberty_parser::Value::String(s) => s.clone(),
        xlsynth_g8r::liberty::liberty_parser::Value::Identifier(s) => s.clone(),
        _ => panic!("Expected string or identifier for function attribute"),
    }
}

fn direction_from_str(s: &str) -> i32 {
    match s {
        "input" => PinDirection::Input as i32,
        "output" => PinDirection::Output as i32,
        _ => PinDirection::Invalid as i32,
    }
}

fn block_to_proto_cells(block: &xlsynth_g8r::liberty::liberty_parser::Block) -> Vec<Cell> {
    let mut cells = Vec::new();
    for member in &block.members {
        if let xlsynth_g8r::liberty::liberty_parser::BlockMember::SubBlock(cell_block) = member {
            if cell_block.block_type != "cell" {
                continue;
            }
            let mut area = 0.0;
            let mut pins = Vec::new();
            let name = match cell_block.qualifiers.get(0) {
                Some(xlsynth_g8r::liberty::liberty_parser::Value::Identifier(s)) => s.clone(),
                Some(xlsynth_g8r::liberty::liberty_parser::Value::String(s)) => s.clone(),
                _ => String::new(),
            };
            for cell_member in &cell_block.members {
                match cell_member {
                    xlsynth_g8r::liberty::liberty_parser::BlockMember::BlockAttr(attr) => {
                        if attr.attr_name == "area" {
                            area = value_to_f64(&attr.value);
                        }
                    }
                    xlsynth_g8r::liberty::liberty_parser::BlockMember::SubBlock(pin_block) => {
                        if pin_block.block_type != "pin" {
                            continue;
                        }
                        let mut direction = PinDirection::Invalid as i32;
                        let mut function = String::new();
                        for pin_member in &pin_block.members {
                            if let xlsynth_g8r::liberty::liberty_parser::BlockMember::BlockAttr(
                                attr,
                            ) = pin_member
                            {
                                if attr.attr_name == "direction" {
                                    if let xlsynth_g8r::liberty::liberty_parser::Value::Identifier(
                                        s,
                                    ) = &attr.value
                                    {
                                        direction = direction_from_str(s);
                                    }
                                } else if attr.attr_name == "function" {
                                    function = value_to_string(&attr.value);
                                }
                            }
                        }
                        pins.push(Pin {
                            direction,
                            function,
                        });
                    }
                }
            }
            cells.push(Cell { area, pins, name });
        }
    }
    cells
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
    use super::*;
    #[test]
    fn test_proto_structure() {
        let mut lib = Library { cells: vec![] };
        lib.cells.push(Cell {
            area: 1.0,
            pins: vec![Pin {
                direction: PinDirection::Input as i32,
                function: "A".to_string(),
            }],
            name: "test_cell".to_string(),
        });
        assert_eq!(lib.cells.len(), 1);
        assert_eq!(lib.cells[0].pins[0].direction, PinDirection::Input as i32);
    }
}
