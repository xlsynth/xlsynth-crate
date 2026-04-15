// SPDX-License-Identifier: Apache-2.0

use crate::common::write_stdout;
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_rewrite::{
    apply_rule_to_package, MatchPattern, MatchRewritePackageError, MatchRewriteRule,
    RewriteApplyMode, RewriteTarget, RewriteTemplate,
};

fn extract_error_byte_offset(msg: &str) -> Option<usize> {
    let (_prefix, suffix) = msg.rsplit_once(" at byte ")?;
    suffix.parse::<usize>().ok()
}

fn format_parse_error(kind: &str, text: &str, err: &str) -> String {
    if let Some(pos) = extract_error_byte_offset(err) {
        let clamped = std::cmp::min(pos, text.len());
        format!(
            "Failed to parse {kind}: {err}\n  {text}\n  {}^",
            " ".repeat(clamped)
        )
    } else {
        format!("Failed to parse {kind}: {err}")
    }
}

/// Handles the `ir-rewrite` driver subcommand.
pub fn handle_ir_rewrite(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches
        .get_one::<String>("ir_input_file")
        .expect("ir_input_file is required");
    let match_text = matches
        .get_one::<String>("match")
        .expect("match is required");
    let replacement_text = matches
        .get_one::<String>("replacement")
        .expect("replacement is required");

    let match_pattern = MatchPattern::parse(match_text).unwrap_or_else(|e| {
        report_cli_error_and_exit(
            &format_parse_error("match pattern", match_text, &e.to_string()),
            Some("ir-rewrite"),
            vec![],
        )
    });
    let rewrite_template = RewriteTemplate::parse(replacement_text).unwrap_or_else(|e| {
        report_cli_error_and_exit(
            &format_parse_error("replacement", replacement_text, &e.to_string()),
            Some("ir-rewrite"),
            vec![],
        )
    });
    let rule = MatchRewriteRule::new(match_pattern, rewrite_template).unwrap_or_else(|e| {
        report_cli_error_and_exit(
            &format!("Invalid rewrite rule: {e}"),
            Some("ir-rewrite"),
            vec![],
        )
    });
    let mode = if let Some(target_text) = matches.get_one::<String>("target") {
        let target = target_text.parse::<RewriteTarget>().unwrap_or_else(|e| {
            report_cli_error_and_exit(
                &format!("Invalid rewrite target: {e}"),
                Some("ir-rewrite"),
                vec![("target", target_text.as_str())],
            )
        });
        RewriteApplyMode::Target(target)
    } else {
        RewriteApplyMode::AllMatchesSinglePass
    };

    let file_content = std::fs::read_to_string(input_file).unwrap_or_else(|e| {
        report_cli_error_and_exit(
            &format!("Failed to read {input_file}: {e}"),
            Some("ir-rewrite"),
            vec![],
        )
    });
    let mut parser = ir_parser::Parser::new(&file_content);
    let pkg = parser.parse_and_validate_package().unwrap_or_else(|e| {
        report_cli_error_and_exit(
            &format!("Failed to parse/validate IR package: {e}"),
            Some("ir-rewrite"),
            vec![],
        )
    });

    let target_fn_name = matches.get_one::<String>("ir_top").map(String::as_str);
    let outcome =
        apply_rule_to_package(&pkg, &rule, target_fn_name, mode).unwrap_or_else(|e| match e {
            MatchRewritePackageError::Apply(err) => {
                report_cli_error_and_exit(&err.to_string(), Some("ir-rewrite"), vec![])
            }
            other => report_cli_error_and_exit(&other.to_string(), Some("ir-rewrite"), vec![]),
        });

    if !matches.contains_id("target") && matches.get_flag("must-match") && !outcome.rewrote() {
        report_cli_error_and_exit("No matches found", Some("ir-rewrite"), vec![]);
    }

    write_stdout(&outcome.rewritten_package().to_string());
}
