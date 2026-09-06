// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use xlsynth_pir::{
    ir,
    ir_parser::{self, Parser, emit_block},
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
        let mut pkg =
            ir_parser::parse_path_to_package(ir_path).expect("parse IR package should succeed");
        if strip_pos {
            pkg.file_table = ir::FileTable::new();
            for member in &mut pkg.members {
                for n in &mut member.graph_mut().nodes {
                    n.pos = None;
                }
            }
        }
        print!("{}", pkg);
    } else {
        // Treat as a standalone block (allowing outer attributes).
        let mut parser = Parser::new(&ir_text);
        let mut block = parser.parse_block().expect("parse block IR should succeed");
        if strip_pos {
            for n in block.nodes.iter_mut() {
                n.pos = None;
            }
        }
        let block_text = emit_block(&block, false);
        print!("{}", block_text);
    }
}
