// SPDX-License-Identifier: Apache-2.0

use crate::liberty::cell_formula::parse_formula;
use crate::liberty::util::human_readable_size;
use crate::liberty::{CharReader, LibertyParser};
use crate::liberty_proto::{Cell, Library, Pin, PinDirection, Sequential, SequentialKind};
use flate2::bufread::GzDecoder;
use std::collections::HashSet;
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

fn qualifier_to_string(value: &crate::liberty::liberty_parser::Value) -> Option<String> {
    match value {
        crate::liberty::liberty_parser::Value::Identifier(s)
        | crate::liberty::liberty_parser::Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn extract_sequential_blocks(
    cell_block: &crate::liberty::liberty_parser::Block,
) -> Vec<Sequential> {
    let mut sequential = Vec::new();
    for cell_member in &cell_block.members {
        let crate::liberty::liberty_parser::BlockMember::SubBlock(sub_block) = cell_member else {
            continue;
        };
        let kind = match sub_block.block_type.as_str() {
            "ff" => SequentialKind::Ff,
            "latch" => SequentialKind::Latch,
            _ => continue,
        } as i32;

        let Some(state_var) = sub_block.qualifiers.first().and_then(qualifier_to_string) else {
            continue;
        };

        let mut next_state = String::new();
        let mut data_in = String::new();
        let mut clock_expr = String::new();
        for seq_member in &sub_block.members {
            let crate::liberty::liberty_parser::BlockMember::BlockAttr(attr) = seq_member else {
                continue;
            };
            if attr.attr_name == "next_state" {
                next_state = value_to_string(&attr.value);
            } else if sub_block.block_type == "latch" && attr.attr_name == "data_in" {
                data_in = value_to_string(&attr.value);
            } else if (sub_block.block_type == "ff" && attr.attr_name == "clocked_on")
                || (sub_block.block_type == "latch" && attr.attr_name == "enable")
            {
                clock_expr = value_to_string(&attr.value);
            }
        }

        if next_state.is_empty() && kind == (SequentialKind::Latch as i32) && !data_in.is_empty() {
            next_state = data_in;
        }

        if next_state.is_empty() && clock_expr.is_empty() {
            continue;
        }

        sequential.push(Sequential {
            state_var,
            next_state,
            clock_expr,
            kind,
        });
    }
    sequential
}

fn block_to_proto_cells(block: &crate::liberty::liberty_parser::Block) -> Vec<Cell> {
    let mut cells = Vec::new();
    for member in &block.members {
        if let crate::liberty::liberty_parser::BlockMember::SubBlock(cell_block) = member {
            if cell_block.block_type != "cell" {
                continue;
            }
            let name = match cell_block.qualifiers.get(0) {
                Some(crate::liberty::liberty_parser::Value::Identifier(s)) => s.clone(),
                Some(crate::liberty::liberty_parser::Value::String(s)) => s.clone(),
                _ => String::new(),
            };
            let mut area = 0.0;
            let mut clocking_pins: HashSet<String> = HashSet::new();
            let sequential = extract_sequential_blocks(cell_block);

            // First pass: gather cell-level attributes (like area) and any clocking pins
            // referenced by sequential blocks via clock expressions.
            for cell_member in &cell_block.members {
                if let crate::liberty::liberty_parser::BlockMember::BlockAttr(attr) = cell_member {
                    if attr.attr_name == "area" {
                        area = value_to_f64(&attr.value);
                    }
                }
            }
            for seq in &sequential {
                if seq.clock_expr.is_empty() {
                    continue;
                }
                match parse_formula(&seq.clock_expr) {
                    Ok(term) => {
                        for input in term.inputs() {
                            clocking_pins.insert(input);
                        }
                    }
                    Err(e) => {
                        log::warn!(
                            "Failed to parse sequential clock expression {:?}: {}",
                            seq.clock_expr,
                            e
                        );
                    }
                }
            }

            // Second pass: build Pin protos, marking pins that are referred to by any
            // clocked_on attribute in an ff block as clocking pins.
            let mut pins = Vec::new();
            for cell_member in &cell_block.members {
                if let crate::liberty::liberty_parser::BlockMember::SubBlock(pin_block) =
                    cell_member
                {
                    if pin_block.block_type != "pin" {
                        continue;
                    }
                    let mut direction = PinDirection::Invalid as i32;
                    let mut function = String::new();
                    let mut pin_name = String::new();
                    if let Some(crate::liberty::liberty_parser::Value::Identifier(s)) =
                        pin_block.qualifiers.get(0)
                    {
                        pin_name = s.clone();
                    } else if let Some(crate::liberty::liberty_parser::Value::String(s)) =
                        pin_block.qualifiers.get(0)
                    {
                        pin_name = s.clone();
                    }
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
                    let is_clocking_pin = clocking_pins.contains(&pin_name);
                    pins.push(Pin {
                        direction,
                        function,
                        name: pin_name,
                        is_clocking_pin,
                    });
                }
            }
            cells.push(Cell {
                area,
                pins,
                name,
                sequential,
            });
        }
    }
    cells
}

pub fn parse_liberty_files_to_proto<P: AsRef<Path>>(paths: &[P]) -> Result<Library, String> {
    let mut libraries = Vec::new();
    for path in paths {
        log::info!("Parsing Liberty file: {}", path.as_ref().display());
        // Log the human-readable file size
        match std::fs::metadata(&path) {
            Ok(meta) => {
                let size = meta.len();
                let human_size = human_readable_size(size);
                log::info!("  Size: {}", human_size);
            }
            Err(e) => {
                log::warn!("  Could not stat file: {}", e);
            }
        }
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
    log::info!("Flattening cells from {} libraries...", libraries.len());
    let mut all_cells = Vec::new();
    for lib in &libraries {
        all_cells.extend(block_to_proto_cells(lib));
    }
    log::info!("Flattened {} cells", all_cells.len());
    Ok(Library { cells: all_cells })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty::descriptor::LIBERTY_DESCRIPTOR;
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
        assert!(!lib.cells[0].pins[0].is_clocking_pin);
        assert!(!lib.cells[0].pins[1].is_clocking_pin);
        assert!(!lib.cells[0].pins[2].is_clocking_pin);
        // Pretty-print using prost-reflect
        let descriptor_pool = DescriptorPool::decode(LIBERTY_DESCRIPTOR).unwrap();
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

    #[test]
    fn test_clocked_on_marks_clocking_pin() {
        let liberty_text = r#"
        library (my_library) {
            cell (my_ff) {
                area: 2.0;
                pin (CLK) {
                    direction: input;
                }
                pin (D) {
                    direction: input;
                }
                pin (Q) {
                    direction: output;
                }
                ff (IQN) {
                    clocked_on : "CLK";
                }
            }
        }
        "#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{}", liberty_text).unwrap();
        let lib = parse_liberty_files_to_proto(&[tmp.path()]).unwrap();
        assert_eq!(lib.cells.len(), 1);
        let cell = &lib.cells[0];
        assert_eq!(cell.name, "my_ff");
        assert_eq!(cell.pins.len(), 3);
        let mut clk_is_clocking = None;
        let mut d_is_clocking = None;
        let mut q_is_clocking = None;
        for pin in &cell.pins {
            match pin.name.as_str() {
                "CLK" => clk_is_clocking = Some(pin.is_clocking_pin),
                "D" => d_is_clocking = Some(pin.is_clocking_pin),
                "Q" => q_is_clocking = Some(pin.is_clocking_pin),
                other => panic!("Unexpected pin name in test: {}", other),
            }
        }
        assert_eq!(clk_is_clocking, Some(true));
        assert_eq!(d_is_clocking, Some(false));
        assert_eq!(q_is_clocking, Some(false));
    }

    #[test]
    fn test_clocked_on_multiple_clocks_all_marked() {
        let liberty_text = r#"
        library (my_library) {
            cell (my_ff) {
                area: 2.0;
                pin (CLK1) {
                    direction: input;
                }
                pin (CLK2) {
                    direction: input;
                }
                pin (D) {
                    direction: input;
                }
                pin (Q) {
                    direction: output;
                }
                ff (IQN) {
                    clocked_on : "CLK1 | CLK2";
                }
            }
        }
        "#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{}", liberty_text).unwrap();
        let lib = parse_liberty_files_to_proto(&[tmp.path()]).unwrap();
        assert_eq!(lib.cells.len(), 1);
        let cell = &lib.cells[0];
        assert_eq!(cell.name, "my_ff");
        assert_eq!(cell.pins.len(), 4);
        let mut clk1 = None;
        let mut clk2 = None;
        let mut d = None;
        let mut q = None;
        for pin in &cell.pins {
            match pin.name.as_str() {
                "CLK1" => clk1 = Some(pin.is_clocking_pin),
                "CLK2" => clk2 = Some(pin.is_clocking_pin),
                "D" => d = Some(pin.is_clocking_pin),
                "Q" => q = Some(pin.is_clocking_pin),
                other => panic!("Unexpected pin name in test: {}", other),
            }
        }
        assert_eq!(clk1, Some(true));
        assert_eq!(clk2, Some(true));
        assert_eq!(d, Some(false));
        assert_eq!(q, Some(false));
    }

    #[test]
    fn test_ff_next_state_is_captured() {
        // Inspired by the ASAP7 `SDFH*` scan-flop `ff { next_state: ... }` pattern.
        let liberty_text = r#"
        library (my_library) {
            cell (my_scan_ff) {
                area: 3.0;
                pin (CLK) {
                    direction: input;
                }
                pin (D) {
                    direction: input;
                }
                pin (SE) {
                    direction: input;
                }
                pin (SI) {
                    direction: input;
                }
                pin (Q) {
                    direction: output;
                    function: "IQ";
                }
                ff (IQ, IQN) {
                    clocked_on : "CLK";
                    next_state : "(!D * !SE) + (SE * SI)";
                }
            }
        }
        "#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{}", liberty_text).unwrap();
        let lib = parse_liberty_files_to_proto(&[tmp.path()]).unwrap();
        let cell = &lib.cells[0];
        assert_eq!(cell.name, "my_scan_ff");
        assert_eq!(cell.sequential.len(), 1);
        let seq = &cell.sequential[0];
        assert_eq!(seq.state_var, "IQ");
        assert_eq!(seq.next_state, "(!D * !SE) + (SE * SI)");
        assert_eq!(seq.clock_expr, "CLK");
        assert_eq!(seq.kind, SequentialKind::Ff as i32);
    }

    #[test]
    fn test_latch_enable_marks_clocking_pin_and_data_in_captured() {
        // Mirrors ASAP7 `DHLx2_ASAP7_75t_R` latch block with data_in/enable.
        let liberty_text = r#"
        library (my_library) {
            cell (my_latch) {
                area: 1.5;
                pin (CLK) {
                    direction: input;
                }
                pin (D) {
                    direction: input;
                }
                pin (Q) {
                    direction: output;
                    function: "IQ";
                }
                latch (IQ, IQN) {
                    data_in : "D";
                    enable : "CLK";
                    power_down_function : "(!VDD) + (VSS)";
                }
            }
        }
        "#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{}", liberty_text).unwrap();
        let lib = parse_liberty_files_to_proto(&[tmp.path()]).unwrap();
        assert_eq!(lib.cells.len(), 1);
        let cell = &lib.cells[0];
        assert_eq!(cell.name, "my_latch");
        assert_eq!(cell.sequential.len(), 1);
        let seq = &cell.sequential[0];
        assert_eq!(seq.state_var, "IQ");
        assert_eq!(seq.next_state, "D");
        assert_eq!(seq.clock_expr, "CLK");
        assert_eq!(seq.kind, SequentialKind::Latch as i32);
        let mut clk_is_clocking = None;
        let mut d_is_clocking = None;
        let mut q_is_clocking = None;
        for pin in &cell.pins {
            match pin.name.as_str() {
                "CLK" => clk_is_clocking = Some(pin.is_clocking_pin),
                "D" => d_is_clocking = Some(pin.is_clocking_pin),
                "Q" => q_is_clocking = Some(pin.is_clocking_pin),
                other => panic!("Unexpected pin name in test: {}", other),
            }
        }
        assert_eq!(clk_is_clocking, Some(true));
        assert_eq!(d_is_clocking, Some(false));
        assert_eq!(q_is_clocking, Some(false));
    }

    #[test]
    fn test_committed_liberty_bin_matches_generated() {
        // Check protoc version
        let output = std::process::Command::new("protoc")
            .arg("--version")
            .output()
            .expect("failed to run protoc");
        let version_str = String::from_utf8_lossy(&output.stdout);
        let expected_version = "libprotoc 3.21.12"; // Update to your canonical version
        if !version_str.trim().starts_with(expected_version) {
            eprintln!(
                "Skipping descriptor byte comparison: protoc version is '{}', expected '{}'",
                version_str.trim(),
                expected_version
            );
            return;
        }
        let committed = include_bytes!("../../proto/liberty.bin") as &[u8];
        // Generate a fresh descriptor set in a temp dir
        let tmp = tempfile::tempdir().unwrap();
        let descriptor_path = tmp.path().join("liberty.bin");
        // Use absolute path to proto/liberty.proto for robustness
        let proto_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("proto/liberty.proto");
        let proto_path_str = proto_path.to_str().unwrap();
        prost_build::Config::new()
            .file_descriptor_set_path(&descriptor_path)
            .compile_protos(
                &[proto_path_str],
                &[proto_path.parent().unwrap().to_str().unwrap()],
            )
            .expect("Failed to compile proto");
        let generated = std::fs::read(&descriptor_path).expect("read generated liberty.bin");
        assert_eq!(
            committed, generated,
            "Committed proto/liberty.bin is out of date; re-run build and commit the new file"
        );
    }
}
