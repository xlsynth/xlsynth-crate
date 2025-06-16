// SPDX-License-Identifier: Apache-2.0

use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;
use xlsynth_g8r::verilog_version::VerilogVersion;

/// Handles the `dslx-stitch-pipeline` subcommand.
pub fn handle_dslx_stitch_pipeline(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input = matches.get_one::<String>("dslx_input_file").unwrap();
    let top = matches.get_one::<String>("dslx_top").unwrap();
    let dslx = std::fs::read_to_string(input).unwrap_or_else(|e| {
        report_cli_error_and_exit("could not read DSLX input", Some(&e.to_string()), vec![]);
    });
    let use_system_verilog = matches
        .get_one::<String>("use_system_verilog")
        .map(|s| s == "true")
        .unwrap_or(true);
    let verilog_version = if use_system_verilog {
        VerilogVersion::SystemVerilog
    } else {
        VerilogVersion::Verilog
    };

    let stage_list: Option<Vec<String>> = matches
        .get_one::<String>("stages")
        .map(|csv| csv.split(',').map(|s| s.trim().to_string()).collect());
    match xlsynth_g8r::dslx_stitch_pipeline::stitch_pipeline(
        &dslx,
        std::path::Path::new(input),
        top,
        verilog_version,
        stage_list.as_ref().map(|v| v.as_slice()),
    ) {
        Ok(sv) => println!("{}", sv),
        Err(e) => report_cli_error_and_exit("stitch error", Some(&e.0), vec![]),
    }
}
