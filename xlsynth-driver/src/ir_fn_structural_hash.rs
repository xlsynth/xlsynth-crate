// SPDX-License-Identifier: Apache-2.0

use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;
use xlsynth_pir::ir_parser;
use xlsynth_pir::node_hashing::compute_function_structural_hash;

fn hash_to_hex(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes.iter() {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

pub fn handle_ir_fn_structural_hash(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches
        .get_one::<String>("ir_input_file")
        .expect("ir_input_file is required");

    let file_content = match std::fs::read_to_string(input_file) {
        Ok(content) => content,
        Err(e) => {
            report_cli_error_and_exit(
                &format!("Failed to read {}: {}", input_file, e),
                Some("ir-fn-structural-hash"),
                vec![],
            );
        }
    };

    let mut parser = ir_parser::Parser::new(&file_content);
    let mut pkg = match parser.parse_and_validate_package() {
        Ok(pkg) => pkg,
        Err(e) => {
            report_cli_error_and_exit(
                &format!("Failed to parse/validate IR package: {}", e),
                Some("ir-fn-structural-hash"),
                vec![],
            );
        }
    };

    if let Some(top) = matches.get_one::<String>("ir_top") {
        if let Err(e) = pkg.set_top_fn(top) {
            report_cli_error_and_exit(
                &format!("Failed to set --top: {}", e),
                Some("ir-fn-structural-hash"),
                vec![],
            );
        }
    }

    let top_fn = match pkg.get_top_fn() {
        Some(f) => f,
        None => {
            report_cli_error_and_exit(
                "No top function found in package",
                Some("ir-fn-structural-hash"),
                vec![],
            );
        }
    };

    let hash = compute_function_structural_hash(top_fn);
    let hex = hash_to_hex(hash.as_bytes());
    let json = matches
        .get_one::<String>("json")
        .is_some_and(|v| v == "true");
    if json {
        println!("{{\"structural_hash\":\"{}\"}}", hex);
    } else {
        println!("{}", hex);
    }
}
