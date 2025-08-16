// SPDX-License-Identifier: Apache-2.0

use crate::toolchain_config::ToolchainConfig;

pub fn handle_ir_strip_pos_data(matches: &clap::ArgMatches, _config: &Option<ToolchainConfig>) {
    let ir_path = matches.get_one::<String>("ir_file").unwrap();

    let file_content = std::fs::read_to_string(ir_path).unwrap_or_else(|err| {
        eprintln!("Failed to read {}: {}", ir_path, err);
        std::process::exit(1);
    });

    let mut parser = xlsynth_g8r::xls_ir::ir_parser::Parser::new(&file_content);
    let mut package = parser.parse_package().unwrap_or_else(|err| {
        eprintln!("Error encountered parsing XLS IR package: {:?}", err);
        std::process::exit(1);
    });

    // Drop the file table so it is not emitted on print.
    package.file_table.id_to_path.clear();

    // Emit IR without position data (nodes are printed without pos attributes).
    print!("{}", package);
}
