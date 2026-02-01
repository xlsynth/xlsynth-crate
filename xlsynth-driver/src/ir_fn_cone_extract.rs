// SPDX-License-Identifier: Apache-2.0

use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;
use xlsynth_pir::ir_fn_cone_extract;
use xlsynth_pir::ir_parser;

pub fn handle_ir_fn_cone_extract(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches
        .get_one::<String>("ir_input_file")
        .expect("ir_input_file is required");
    let sink = matches.get_one::<String>("sink").expect("sink is required");

    let emit_pos_data = match matches
        .get_one::<String>("emit_pos_data")
        .map(|s| s.as_str())
    {
        Some("true") => true,
        Some("false") => false,
        _ => false,
    };

    let file_content = match std::fs::read_to_string(input_file) {
        Ok(content) => content,
        Err(e) => {
            report_cli_error_and_exit(
                &format!("Failed to read {}: {}", input_file, e),
                Some("ir-fn-cone-extract"),
                vec![],
            );
        }
    };

    let parse_options = ir_parser::ParseOptions {
        retain_pos_data: emit_pos_data,
    };
    let mut parser = ir_parser::Parser::new_with_options(&file_content, parse_options);
    let mut pkg = match parser.parse_and_validate_package() {
        Ok(pkg) => pkg,
        Err(e) => {
            report_cli_error_and_exit(
                &format!("Failed to parse/validate IR package: {}", e),
                Some("ir-fn-cone-extract"),
                vec![],
            );
        }
    };

    if let Some(top) = matches.get_one::<String>("ir_top") {
        if let Err(e) = pkg.set_top_fn(top) {
            report_cli_error_and_exit(
                &format!("Failed to set --top: {}", e),
                Some("ir-fn-cone-extract"),
                vec![],
            );
        }
    }

    let top_fn = match pkg.get_top_fn() {
        Some(f) => f,
        None => {
            report_cli_error_and_exit(
                "No top function found in package",
                Some("ir-fn-cone-extract"),
                vec![],
            );
        }
    };

    let sink_selector = match ir_fn_cone_extract::parse_sink_selector(sink) {
        Ok(s) => s,
        Err(e) => {
            report_cli_error_and_exit(
                &format!("invalid sink selector: {}", e),
                Some("ir-fn-cone-extract"),
                vec![("sink", sink.as_str())],
            );
        }
    };

    let extracted = match ir_fn_cone_extract::extract_fn_cone_to_params(
        top_fn,
        Some(&pkg.file_table),
        sink_selector,
        emit_pos_data,
    ) {
        Ok(x) => x,
        Err(e) => {
            report_cli_error_and_exit(
                &e,
                Some("ir-fn-cone-extract"),
                vec![("sink", sink.as_str())],
            );
        }
    };

    print!("{}", extracted.package.to_string());
}
