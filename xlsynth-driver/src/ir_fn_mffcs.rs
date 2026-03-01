// SPDX-License-Identifier: Apache-2.0

use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;
use serde::Serialize;
use std::io::Write;
use xlsynth_pir::ir_fn_mffcs::{
    enumerate_all_mffc_specs, extract_mffc, has_nonlocal_callee_refs, rank_and_select_mffc_specs,
    MffcConfig,
};
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_utils::{classify_trivial_fn_body, TrivialFnBody};

#[derive(Serialize)]
struct MffcManifestLine {
    sha256: String,
    rank: usize,
    root_node_index: usize,
    root_text_id: usize,
    frontier_leaf_indices: Vec<usize>,
    frontier_non_literal_count: usize,
    internal_non_literal_count: usize,
    included_node_count: usize,
    score_numerator: usize,
    score_denominator: usize,
}

fn parse_positive_or_zero_usize(matches: &ArgMatches, key: &str, cmd: &str) -> usize {
    let raw = matches
        .get_one::<String>(key)
        .unwrap_or_else(|| panic!("{} is required by clap", key));
    match raw.parse::<usize>() {
        Ok(v) => v,
        Err(_) => report_cli_error_and_exit(
            &format!(
                "invalid --{} value: '{}'; expected a non-negative integer",
                key, raw
            ),
            Some(cmd),
            vec![(key, raw.as_str())],
        ),
    }
}

pub fn handle_ir_fn_mffcs(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let cmd = "ir-fn-mffcs";
    let input_file = matches
        .get_one::<String>("ir_input_file")
        .expect("ir_input_file is required");
    let output_dir = matches
        .get_one::<String>("output_dir")
        .expect("output_dir is required");

    let max_mffcs_raw = parse_positive_or_zero_usize(matches, "max_mffcs", cmd);
    let max_mffcs = if max_mffcs_raw == 0 {
        None
    } else {
        Some(max_mffcs_raw)
    };
    let min_internal_non_literal_count =
        parse_positive_or_zero_usize(matches, "min_internal_non_literal", cmd);
    let max_frontier_non_literal_raw =
        parse_positive_or_zero_usize(matches, "max_frontier_non_literal", cmd);
    let max_frontier_non_literal_count = if max_frontier_non_literal_raw == 0 {
        None
    } else {
        Some(max_frontier_non_literal_raw)
    };

    let emit_pos_data = match matches
        .get_one::<String>("emit_pos_data")
        .map(|s| s.as_str())
    {
        Some("true") => true,
        Some("false") => false,
        _ => false,
    };

    let parse_options = ir_parser::ParseOptions {
        retain_pos_data: emit_pos_data,
    };

    let file_content = match std::fs::read_to_string(input_file) {
        Ok(content) => content,
        Err(e) => report_cli_error_and_exit(
            &format!("Failed to read {}: {}", input_file, e),
            Some(cmd),
            vec![],
        ),
    };
    let mut parser = ir_parser::Parser::new_with_options(&file_content, parse_options);
    let mut pkg = match parser.parse_and_validate_package() {
        Ok(pkg) => pkg,
        Err(e) => report_cli_error_and_exit(
            &format!("Failed to parse/validate IR package: {}", e),
            Some(cmd),
            vec![],
        ),
    };

    if let Some(top) = matches.get_one::<String>("ir_top") {
        if let Err(e) = pkg.set_top_fn(top) {
            report_cli_error_and_exit(
                &format!("Failed to set --top: {}", e),
                Some(cmd),
                vec![("top", top.as_str())],
            );
        }
    }

    let top_fn = match pkg.get_top_fn() {
        Some(f) => f,
        None => report_cli_error_and_exit("No top function found in package", Some(cmd), vec![]),
    };

    if has_nonlocal_callee_refs(top_fn) {
        report_cli_error_and_exit(
            "ir-fn-mffcs does not support top functions containing invoke/counted_for nodes; run an inlining/optimization flow first",
            Some(cmd),
            vec![],
        );
    }

    let cfg = MffcConfig {
        max_mffcs,
        min_internal_non_literal_count,
        max_frontier_non_literal_count,
        emit_pos_data,
    };

    let all_specs = enumerate_all_mffc_specs(top_fn);
    let all_count = all_specs.len();
    let selected_specs = rank_and_select_mffc_specs(all_specs, &cfg);
    let selected_count = selected_specs.len();

    let out_dir_path = std::path::Path::new(output_dir);
    if let Err(e) = std::fs::create_dir_all(out_dir_path) {
        report_cli_error_and_exit(
            &format!("Failed to create output_dir {}: {}", output_dir, e),
            Some(cmd),
            vec![("output_dir", output_dir.as_str())],
        );
    }

    let manifest_path = match matches.get_one::<String>("manifest_jsonl") {
        Some(path) => std::path::PathBuf::from(path),
        None => out_dir_path.join("manifest.jsonl"),
    };

    let mut manifest_file = match std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&manifest_path)
    {
        Ok(f) => f,
        Err(e) => report_cli_error_and_exit(
            &format!("Failed to open manifest {}: {}", manifest_path.display(), e),
            Some(cmd),
            vec![],
        ),
    };

    let mut emitted = 0usize;
    for (rank, spec) in selected_specs.iter().enumerate() {
        let extracted = extract_mffc(top_fn, Some(&pkg.file_table), spec, &cfg);

        // Skip structurally trivial outputs; these are typically not useful as
        // mined "meaty" MFFCs.
        if extracted.package.members.len() == 1 {
            match &extracted.package.members[0] {
                xlsynth_pir::ir::PackageMember::Function(f) => {
                    if matches!(
                        classify_trivial_fn_body(f),
                        Some(
                            TrivialFnBody::Constant
                                | TrivialFnBody::SingleParamStructural { .. }
                                | TrivialFnBody::SingleBoolGate { .. }
                        )
                    ) {
                        continue;
                    }
                }
                xlsynth_pir::ir::PackageMember::Block { .. } => {
                    // Extraction here always emits a function package member.
                }
            }
        }

        let out_path = out_dir_path.join(format!("{}.ir", extracted.sha256_hex));
        if !out_path.exists() {
            let mut out = match std::fs::File::create(&out_path) {
                Ok(f) => f,
                Err(e) => report_cli_error_and_exit(
                    &format!("Failed to create {}: {}", out_path.display(), e),
                    Some(cmd),
                    vec![],
                ),
            };
            if let Err(e) = out.write_all(extracted.package.to_string().as_bytes()) {
                report_cli_error_and_exit(
                    &format!("Failed to write {}: {}", out_path.display(), e),
                    Some(cmd),
                    vec![],
                );
            }
        }

        let line = MffcManifestLine {
            sha256: extracted.sha256_hex,
            rank,
            root_node_index: spec.root_node_index,
            root_text_id: spec.root_text_id,
            frontier_leaf_indices: spec.frontier_leaf_indices.clone(),
            frontier_non_literal_count: spec.frontier_non_literal_count,
            internal_non_literal_count: spec.internal_non_literal_count,
            included_node_count: extracted.included_node_count,
            score_numerator: spec.score_numerator,
            score_denominator: spec.score_denominator,
        };
        let json = serde_json::to_string(&line).expect("manifest JSON serialization must succeed");
        writeln!(manifest_file, "{}", json).expect("manifest write must succeed");
        emitted += 1;
    }

    eprintln!(
        "ir-fn-mffcs: enumerated {} candidates, selected {}, emitted {} into {}; manifest: {}",
        all_count,
        selected_count,
        emitted,
        out_dir_path.display(),
        manifest_path.display()
    );
}
