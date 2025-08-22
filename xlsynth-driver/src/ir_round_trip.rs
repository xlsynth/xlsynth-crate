// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use xlsynth_g8r::xls_ir::ir_parser::{emit_fn_as_block, Parser};

/// Implements the "ir-round-trip" subcommand: parse IR and write it back to
/// stdout.
pub fn handle_ir_round_trip(matches: &ArgMatches) {
    let ir_path = std::path::Path::new(matches.get_one::<String>("ir_input_file").unwrap());
    let strip_pos = matches
        .get_one::<String>("strip_pos_attrs")
        .map(|s| s == "true")
        .unwrap_or(false);
    let ir_text = std::fs::read_to_string(ir_path).expect("read IR input should succeed");

    // Heuristic: support both package IR and standalone block IR files.
    let trimmed = ir_text.trim_start();
    let looks_like_block = if trimmed.starts_with("package") {
        false
    } else {
        ir_path.to_string_lossy().ends_with(".block.ir")
            || trimmed.starts_with("block")
            || trimmed.starts_with("#[")
            || trimmed.starts_with("#![")
    };

    if looks_like_block {
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
    } else {
        match xlsynth_g8r::xls_ir::ir_parser::parse_path_to_package(ir_path) {
            Ok(mut pkg) => {
                if strip_pos {
                    pkg.file_table = xlsynth_g8r::xls_ir::ir::FileTable::new();
                    pkg.for_each_fn_mut(|f| {
                        for n in f.nodes.iter_mut() {
                            n.pos = None;
                        }
                    });
                }
                print!("{}", pkg);
            }
            Err(_e) => {
                // Fallback: the file may be a package wrapper with a block; extract the block.
                if let Some(block_idx) = ir_text.find("block ") {
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
                            // Could not parse block; just pass the original text as-is.
                            print!("{}", ir_text);
                        }
                    }
                } else {
                    // Surface the original error context if no block was found.
                    let mut pkg = xlsynth_g8r::xls_ir::ir_parser::parse_path_to_package(ir_path)
                        .expect("parse IR package should succeed");
                    if strip_pos {
                        pkg.file_table = xlsynth_g8r::xls_ir::ir::FileTable::new();
                        pkg.for_each_fn_mut(|f| {
                            for n in f.nodes.iter_mut() {
                                n.pos = None;
                            }
                        });
                    }
                    print!("{}", pkg);
                }
            }
        }
    }
}
