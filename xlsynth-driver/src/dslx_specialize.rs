// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use clap::ArgMatches;

use crate::common::get_dslx_paths;
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;

pub fn handle_dslx_specialize(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches
        .get_one::<String>("dslx_input_file")
        .expect("dslx_input_file arg missing");
    let top = matches
        .get_one::<String>("dslx_top")
        .expect("dslx_top arg missing");
    let input_path = Path::new(input_file);

    let dslx_contents = match std::fs::read_to_string(input_path) {
        Ok(contents) => contents,
        Err(err) => {
            report_cli_error_and_exit(
                "failed to read DSLX input file",
                Some("dslx-specialize"),
                vec![
                    ("path", input_file.as_str()),
                    ("error", &format!("{}", err)),
                ],
            );
        }
    };

    let crate::common::DslxPaths {
        stdlib_path,
        search_paths,
    } = get_dslx_paths(matches, config);
    let stdlib_path_ref = stdlib_path.as_deref();
    let additional_search_paths = search_paths;

    let specialized_text = match xlsynth_prover::dslx_specializer::specialize_dslx_module(
        &dslx_contents,
        input_path,
        top,
        stdlib_path_ref,
        &additional_search_paths,
    ) {
        Ok(output) => output,
        Err(err) => {
            report_cli_error_and_exit(
                "DSLX specialization failed",
                Some("dslx-specialize"),
                vec![
                    ("path", input_file.as_str()),
                    ("error", &format!("{}", err)),
                ],
            );
        }
    };

    println!("{}", specialized_text);
}
