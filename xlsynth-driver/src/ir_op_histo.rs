// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::path::Path;

use clap::ArgMatches;
use xlsynth_pir::ir::Package;
use xlsynth_pir::ir_corpus::walk_ir_files_sorted;
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_utils::op_histogram;
use xlsynth_pir::ir_utils::op_histogram_with_types;

use crate::common::parse_bool_flag_or;
use crate::common::write_stdout_line;
use crate::toolchain_config::ToolchainConfig;

fn add_hist(total: &mut BTreeMap<String, usize>, part: &BTreeMap<String, usize>) {
    for (op, count) in part.iter() {
        *total.entry(op.clone()).or_insert(0) += *count;
    }
}

fn format_hist(hist: &BTreeMap<String, usize>) -> String {
    if hist.is_empty() {
        return "{}".to_string();
    }
    let entries: Vec<String> = hist
        .iter()
        .map(|(op, count)| format!("{op}: {count}"))
        .collect();
    format!("{{{}}}", entries.join(", "))
}

fn parse_ir_package_file(path: &Path) -> Result<Package, String> {
    let file_content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let mut parser = ir_parser::Parser::new(&file_content);
    parser
        .parse_and_validate_package()
        .map_err(|e| format!("parse/validate failed for {}: {e}", path.display()))
}

fn histogram_for_ir_file(
    path: &Path,
    top: Option<&String>,
    include_types: bool,
) -> Result<BTreeMap<String, usize>, String> {
    let mut pkg = parse_ir_package_file(path)?;

    if let Some(top) = top {
        pkg.set_top_fn(top)
            .map_err(|e| format!("failed to set --top for {}: {e}", path.display()))?;
    }

    let top_fn = pkg
        .get_top_fn()
        .ok_or_else(|| format!("no top function found in {}", path.display()))?;

    Ok(if include_types {
        op_histogram_with_types(top_fn)
    } else {
        op_histogram(top_fn)
    })
}

pub fn handle_ir_op_histo(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let ir_file = matches
        .get_one::<String>("ir_input_file")
        .expect("ir_input_file is required");
    let include_types = parse_bool_flag_or(matches, "include-types", true);
    let top = matches.get_one::<String>("ir_top");

    let hist = histogram_for_ir_file(Path::new(ir_file), top, include_types).unwrap_or_else(|e| {
        eprintln!("error: ir-op-histo: {e}");
        std::process::exit(2);
    });
    write_stdout_line(&format_hist(&hist));
}

pub fn handle_ir_op_histo_corpus(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let corpus_dir = matches
        .get_one::<String>("corpus_dir")
        .expect("corpus_dir is required");
    let ignore_parse_errors = parse_bool_flag_or(matches, "ignore-parse-errors", true);
    let include_types = parse_bool_flag_or(matches, "include-types", true);
    let top = matches.get_one::<String>("ir_top");

    let max_files: Option<usize> = matches
        .get_one::<String>("max-files")
        .map(|s| {
            s.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("error: ir-op-histo-corpus: --max-files must be an integer");
                std::process::exit(2);
            })
        })
        .filter(|&n| n != 0);

    let root = Path::new(corpus_dir);
    let mut total_hist: BTreeMap<String, usize> = BTreeMap::new();

    if let Err(e) = walk_ir_files_sorted(root, max_files, |path| {
        let per_file_hist = match histogram_for_ir_file(&path, top, include_types) {
            Ok(hist) => hist,
            Err(e) => {
                if ignore_parse_errors {
                    return true;
                }
                eprintln!("error: ir-op-histo-corpus: {e}");
                std::process::exit(2);
            }
        };

        add_hist(&mut total_hist, &per_file_hist);
        write_stdout_line(&format!(
            "{}: {}",
            path.display(),
            format_hist(&per_file_hist)
        ));
        true
    }) {
        eprintln!(
            "error: ir-op-histo-corpus: failed to read corpus dir {}: {e}",
            root.display()
        );
        std::process::exit(2);
    }

    write_stdout_line(&format!("total: {}", format_hist(&total_hist)));
}
