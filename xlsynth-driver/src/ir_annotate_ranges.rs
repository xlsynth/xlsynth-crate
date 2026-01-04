// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::toolchain_config::ToolchainConfig;

pub fn handle_ir_annotate_ranges(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let ir_path = matches.get_one::<String>("ir_input_file").unwrap();
    let top = matches.get_one::<String>("ir_top").map(|s| s.as_str());

    let file_content = std::fs::read_to_string(ir_path).unwrap_or_else(|err| {
        eprintln!("Failed to read {}: {}", ir_path, err);
        std::process::exit(1);
    });

    let annotated =
        xlsynth_pir::ir_annotate_ranges::annotate_ranges_in_package_ir_text(&file_content, top)
            .unwrap_or_else(|e| {
                eprintln!("Error annotating ranges: {}", e);
                std::process::exit(1);
            });

    print!("{}", annotated);
}
