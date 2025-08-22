// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

/// Implements the "ir-round-trip" subcommand: parse IR and write it back to
/// stdout.
pub fn handle_ir_round_trip(matches: &ArgMatches) {
    let ir_path = std::path::Path::new(matches.get_one::<String>("ir_input_file").unwrap());
    let strip_pos = matches
        .get_one::<String>("strip_pos_attrs")
        .map(|s| s == "true")
        .unwrap_or(false);
    let mut pkg = xlsynth_g8r::xls_ir::ir_parser::parse_path_to_package(ir_path)
        .expect("parse IR package should succeed");
    if strip_pos {
        pkg.file_table = xlsynth_g8r::xls_ir::ir::FileTable::new();
        for f in pkg.fns.iter_mut() {
            for n in f.nodes.iter_mut() {
                n.pos = None;
            }
        }
    }
    print!("{}", pkg);
}
