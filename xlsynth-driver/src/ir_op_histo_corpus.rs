// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use clap::ArgMatches;
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_utils::op_histogram;

use crate::common::parse_bool_flag_or;
use crate::common::write_stdout_line;
use crate::toolchain_config::ToolchainConfig;

fn collect_ir_files_recursively(root: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let ty = entry.file_type()?;
            let is_ir = path.extension().and_then(|s| s.to_str()) == Some("ir");
            if ty.is_dir() {
                stack.push(path);
            } else if ty.is_file() {
                if is_ir {
                    out.push(path);
                }
            } else if ty.is_symlink() {
                if is_ir {
                    if let Ok(meta) = std::fs::metadata(&path) {
                        if meta.is_file() {
                            out.push(path);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

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

pub fn handle_ir_op_histo_corpus(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let corpus_dir = matches
        .get_one::<String>("corpus_dir")
        .expect("corpus_dir is required");
    let ignore_parse_errors = parse_bool_flag_or(matches, "ignore-parse-errors", true);

    let max_files: Option<usize> = matches
        .get_one::<String>("max-files")
        .map(|s| {
            s.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("error: ir-op-histo-corpus: --max-files must be an integer");
                std::process::exit(2);
            })
        })
        .filter(|&n| n != 0);

    // Deterministic scan order: gather and sort all candidate paths.
    let mut files: Vec<PathBuf> = Vec::new();
    let root = Path::new(corpus_dir);
    if let Err(e) = collect_ir_files_recursively(root, &mut files) {
        eprintln!(
            "error: ir-op-histo-corpus: failed to read corpus dir {}: {e}",
            root.display()
        );
        std::process::exit(2);
    }
    files.sort();

    let mut files_scanned = 0usize;
    let mut total_hist: BTreeMap<String, usize> = BTreeMap::new();

    for path in files {
        if let Some(limit) = max_files {
            if files_scanned >= limit {
                break;
            }
        }
        files_scanned += 1;

        let file_content = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                if ignore_parse_errors {
                    continue;
                }
                eprintln!(
                    "error: ir-op-histo-corpus: failed to read {}: {e}",
                    path.display()
                );
                std::process::exit(2);
            }
        };

        let mut parser = ir_parser::Parser::new(&file_content);
        let mut pkg = match parser.parse_and_validate_package() {
            Ok(pkg) => pkg,
            Err(e) => {
                if ignore_parse_errors {
                    continue;
                }
                eprintln!(
                    "error: ir-op-histo-corpus: parse/validate failed for {}: {e}",
                    path.display()
                );
                std::process::exit(2);
            }
        };

        if let Some(top) = matches.get_one::<String>("ir_top") {
            if let Err(e) = pkg.set_top_fn(top) {
                if ignore_parse_errors {
                    continue;
                }
                eprintln!(
                    "error: ir-op-histo-corpus: failed to set --top for {}: {e}",
                    path.display()
                );
                std::process::exit(2);
            }
        }

        let top_fn = match pkg.get_top_fn() {
            Some(f) => f,
            None => {
                if ignore_parse_errors {
                    continue;
                }
                eprintln!(
                    "error: ir-op-histo-corpus: no top function found in {}",
                    path.display()
                );
                std::process::exit(2);
            }
        };

        let per_file_hist = op_histogram(top_fn);
        add_hist(&mut total_hist, &per_file_hist);
        write_stdout_line(&format!(
            "{}: {}",
            path.display(),
            format_hist(&per_file_hist)
        ));
    }

    write_stdout_line(&format!("total: {}", format_hist(&total_hist)));
}
