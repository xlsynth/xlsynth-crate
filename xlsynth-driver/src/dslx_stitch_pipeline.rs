// SPDX-License-Identifier: Apache-2.0

use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;

/// Handles the `dslx-stitch-pipeline` subcommand.
pub fn handle_dslx_stitch_pipeline(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input = matches.get_one::<String>("dslx_input_file").unwrap();
    let top = matches.get_one::<String>("dslx_top").unwrap();
    let dslx = std::fs::read_to_string(input).unwrap_or_else(|e| {
        report_cli_error_and_exit("could not read DSLX input", Some(&e.to_string()), vec![]);
        unreachable!()
    });
    match xlsynth_g8r::dslx_stitch_pipeline::stitch_pipeline(
        &dslx,
        std::path::Path::new(input),
        top,
    ) {
        Ok(sv) => println!("{}", sv),
        Err(e) => report_cli_error_and_exit("stitch error", Some(&e.0), vec![]),
    }
}
