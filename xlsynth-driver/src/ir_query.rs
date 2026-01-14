// SPDX-License-Identifier: Apache-2.0

use crate::common::parse_bool_flag_or;
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_query;

pub fn handle_ir_query(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches
        .get_one::<String>("ir_input_file")
        .expect("ir_input_file is required");
    let query_text = matches
        .get_one::<String>("query")
        .expect("query is required");

    // Parse the query up-front so malformed queries fail fast (before we do any IR
    // I/O).
    let query = match ir_query::parse_query(query_text) {
        Ok(query) => query,
        Err(e) => {
            let msg = format!("Failed to parse query: {}", e);
            report_cli_error_and_exit(&msg, Some("ir-query"), vec![]);
        }
    };

    if parse_bool_flag_or(matches, "check_query", false) {
        return;
    }

    let file_content = std::fs::read_to_string(input_file)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", input_file, e));
    let mut parser = ir_parser::Parser::new(&file_content);
    let mut pkg = parser
        .parse_and_validate_package()
        .unwrap_or_else(|e| panic!("Failed to parse/validate IR package: {}", e));

    if let Some(top) = matches.get_one::<String>("ir_top") {
        pkg.set_top_fn(top)
            .unwrap_or_else(|e| panic!("Failed to set --top: {}", e));
    }
    let top_fn = pkg
        .get_top_fn()
        .unwrap_or_else(|| panic!("No top function found in package"));

    let matches = ir_query::find_matching_nodes(top_fn, &query);

    for node_ref in matches {
        let node = top_fn.get_node(node_ref);
        if let Some(line) = node.to_string(top_fn) {
            println!("{}", line);
        } else {
            println!("{}", ir::node_textual_id(top_fn, node_ref));
        }
    }
}
