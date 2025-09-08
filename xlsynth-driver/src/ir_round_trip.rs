// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use xlsynth_pir::{
    ir,
    ir_parser::{self, emit_fn_as_block, Parser},
};

/// Implements the "ir-round-trip" subcommand: parse IR and write it back to
/// stdout.
pub fn handle_ir_round_trip(matches: &ArgMatches) {
    let ir_path = std::path::Path::new(matches.get_one::<String>("ir_input_file").unwrap());
    let strip_pos = matches
        .get_one::<String>("strip_pos_attrs")
        .map(|s| s == "true")
        .unwrap_or(false);
    let ir_text = std::fs::read_to_string(ir_path).expect("read IR input should succeed");

    // Grammar-based prefix scanning: package vs block.
    let trimmed = ir_text.trim_start();
    if trimmed.starts_with("package") {
        match ir_parser::parse_path_to_package(ir_path) {
            Ok(mut pkg) => {
                if strip_pos {
                    pkg.file_table = ir::FileTable::new();
                    pkg.for_each_fn_mut(|f| {
                        for n in f.nodes.iter_mut() {
                            n.pos = None;
                        }
                    });
                }
                print!("{}", pkg);
            }
            Err(_e) => {
                // Attempt to recover by extracting the first block member in the file.
                if let Some(block_idx) = ir_text
                    .find("block ")
                    .or_else(|| ir_text.find("\nblock "))
                    .or_else(|| ir_text.find("\n#["))
                    .or_else(|| ir_text.find("\n#!["))
                {
                    let block_slice = &ir_text[block_idx..];
                    let mut parser = Parser::new(block_slice);
                    match parser.parse_block_to_fn_with_ports() {
                        Ok((mut f, port_info)) => {
                            if strip_pos {
                                for n in f.nodes.iter_mut() {
                                    n.pos = None;
                                }
                            }
                            let block_text = emit_fn_as_block(&f, None, Some(&port_info));
                            print!("{}", block_text);
                        }
                        Err(_e2) => {
                            // Could not parse a block member; pass through the original text.
                            print!("{}", ir_text);
                        }
                    }
                } else {
                    // No block found; pass through the original text.
                    print!("{}", ir_text);
                }
            }
        }
    } else {
        // Treat as a standalone block (allowing outer attributes).
        let mut parser = Parser::new(&ir_text);
        let (mut f, port_info) = parser
            .parse_block_to_fn_with_ports()
            .expect("parse block IR should succeed");
        if strip_pos {
            for n in f.nodes.iter_mut() {
                n.pos = None;
            }
        }
        let block_text = emit_fn_as_block(&f, None, Some(&port_info));
        print!("{}", block_text);
    }
}
