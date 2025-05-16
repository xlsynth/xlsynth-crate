// SPDX-License-Identifier: Apache-2.0

use crate::liberty::{CharReader, LibertyParser};
use crate::liberty_proto::{Cell, Library, Pin, PinDirection};
use flate2::bufread::GzDecoder;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

fn value_to_f64(value: &crate::liberty::liberty_parser::Value) -> f64 {
    match value {
        crate::liberty::liberty_parser::Value::Number(n) => *n,
        _ => panic!("Expected number for area attribute"),
    }
}

fn value_to_string(value: &crate::liberty::liberty_parser::Value) -> String {
    match value {
        crate::liberty::liberty_parser::Value::String(s) => s.clone(),
        crate::liberty::liberty_parser::Value::Identifier(s) => s.clone(),
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

fn block_to_proto_cells(block: &crate::liberty::liberty_parser::Block) -> Vec<Cell> {
    let mut cells = Vec::new();
    for member in &block.members {
        if let crate::liberty::liberty_parser::BlockMember::SubBlock(cell_block) = member {
            if cell_block.block_type != "cell" {
                continue;
            }
            let mut area = 0.0;
            let mut pins = Vec::new();
            let name = match cell_block.qualifiers.get(0) {
                Some(crate::liberty::liberty_parser::Value::Identifier(s)) => s.clone(),
                Some(crate::liberty::liberty_parser::Value::String(s)) => s.clone(),
                _ => String::new(),
            };
            for cell_member in &cell_block.members {
                match cell_member {
                    crate::liberty::liberty_parser::BlockMember::BlockAttr(attr) => {
                        if attr.attr_name == "area" {
                            area = value_to_f64(&attr.value);
                        }
                    }
                    crate::liberty::liberty_parser::BlockMember::SubBlock(pin_block) => {
                        if pin_block.block_type != "pin" {
                            continue;
                        }
                        let mut direction = PinDirection::Invalid as i32;
                        let mut function = String::new();
                        for pin_member in &pin_block.members {
                            if let crate::liberty::liberty_parser::BlockMember::BlockAttr(attr) =
                                pin_member
                            {
                                if attr.attr_name == "direction" {
                                    if let crate::liberty::liberty_parser::Value::Identifier(s) =
                                        &attr.value
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

pub fn parse_liberty_files_to_proto<P: AsRef<Path>>(paths: &[P]) -> Result<Library, String> {
    let mut libraries = Vec::new();
    for path in paths {
        log::info!("Parsing Liberty file: {}", path.as_ref().display());
        let file = File::open(&path)
            .map_err(|e| format!("Failed to open {}: {}", path.as_ref().display(), e))?;
        let streamer: Box<dyn std::io::Read> = if path
            .as_ref()
            .extension()
            .map(|e| e == "gz")
            .unwrap_or(false)
        {
            Box::new(GzDecoder::new(BufReader::with_capacity(256 * 1024, file)))
        } else {
            Box::new(BufReader::new(file))
        };
        let char_reader = CharReader::new(streamer);
        let mut parser = LibertyParser::new_from_iter(char_reader);
        let library = parser
            .parse()
            .map_err(|e| format!("Parse error in {}: {}", path.as_ref().display(), e))?;
        libraries.push(library);
    }
    let mut all_cells = Vec::new();
    for lib in &libraries {
        all_cells.extend(block_to_proto_cells(lib));
    }
    Ok(Library { cells: all_cells })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use prost_reflect::{DescriptorPool, DynamicMessage};
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_liberty_files_to_proto_and_pretty_print() {
        let liberty_text = r#"
        library (my_library) {
            cell (my_and) {
                area: 1.0;
                pin (Y) {
                    direction: output;
                    function: "(A * B)";
                }
                pin (A) {
                    direction: input;
                }
                pin (B) {
                    direction: input;
                }
            }
        }
        "#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{}", liberty_text).unwrap();
        let lib = parse_liberty_files_to_proto(&[tmp.path()]).unwrap();
        assert_eq!(lib.cells.len(), 1);
        assert_eq!(lib.cells[0].name, "my_and");
        assert_eq!(lib.cells[0].area, 1.0);
        assert_eq!(lib.cells[0].pins.len(), 3);
        // Pretty-print using prost-reflect
        let descriptor_pool = DescriptorPool::decode(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/liberty.bin"
        )) as &[u8])
        .unwrap();
        let msg_desc = descriptor_pool
            .get_message_by_name("liberty.Library")
            .unwrap();
        let mut buf = Vec::new();
        lib.encode(&mut buf).unwrap();
        let dyn_msg = DynamicMessage::decode(msg_desc, &*buf).unwrap();
        let textproto = dyn_msg.to_text_format();
        println!("Pretty-printed textproto:\n{}", textproto);
        assert!(textproto.contains("cells:["));
        assert!(textproto.contains("name:\"my_and\""));
    }
}
