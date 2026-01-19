// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use clap::ArgMatches;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_query;

use crate::common::parse_bool_flag_or;
use crate::toolchain_config::ToolchainConfig;

fn collect_ir_files_recursively(root: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let ty = entry.file_type()?;
            if ty.is_dir() {
                stack.push(path);
            } else if ty.is_file() {
                if path.extension().and_then(|s| s.to_str()) == Some("ir") {
                    out.push(path);
                }
            }
        }
    }
    Ok(())
}

fn required_opname_prefilter_tokens(query: &ir_query::QueryExpr) -> Vec<String> {
    fn walk(expr: &ir_query::QueryExpr, out: &mut Vec<String>) {
        match expr {
            ir_query::QueryExpr::Placeholder(_)
            | ir_query::QueryExpr::Number(_)
            | ir_query::QueryExpr::Ellipsis => {}
            ir_query::QueryExpr::Matcher(m) => {
                if let ir_query::MatcherKind::OpName(op) = &m.kind {
                    // Quick and effective heuristic: if the query contains an explicit
                    // operator match like `and(...)`, require the file text to contain
                    // `= and(`. This avoids PIR parsing for most non-matching files.
                    //
                    // Note: XLS IR node lines always contain `= <opname>(...)` for op nodes.
                    // Using `= <opname>(` is a stricter filter than `<opname>(`.
                    out.push(format!("= {op}("));
                }
                for a in &m.args {
                    walk(a, out);
                }
                for na in &m.named_args {
                    if let ir_query::NamedArgValue::Expr(e) = &na.value {
                        walk(e, out);
                    } else if let ir_query::NamedArgValue::ExprList(es) = &na.value {
                        for e in es {
                            walk(e, out);
                        }
                    }
                }
            }
        }
    }

    let mut toks = Vec::new();
    walk(query, &mut toks);
    toks.sort();
    toks.dedup();
    toks
}

pub fn handle_ir_query_corpus(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let corpus_dir = matches
        .get_one::<String>("corpus_dir")
        .expect("corpus_dir is required");
    let query_text = matches
        .get_one::<String>("query")
        .expect("query is required");

    let query = match ir_query::parse_query(query_text) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("error: ir-query-corpus: failed to parse query: {e}");
            std::process::exit(2);
        }
    };

    let show_ret = parse_bool_flag_or(matches, "show-ret", true);
    let prefilter = parse_bool_flag_or(matches, "prefilter", true);
    let ignore_parse_errors = parse_bool_flag_or(matches, "ignore-parse-errors", true);

    let max_files: Option<usize> = matches
        .get_one::<String>("max-files")
        .map(|s| {
            s.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("error: ir-query-corpus: --max-files must be an integer");
                std::process::exit(2);
            })
        })
        .filter(|&n| n != 0);

    let max_matches: Option<usize> = matches
        .get_one::<String>("max-matches")
        .map(|s| {
            s.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("error: ir-query-corpus: --max-matches must be an integer");
                std::process::exit(2);
            })
        })
        .filter(|&n| n != 0);

    let prefilter_tokens = if prefilter {
        required_opname_prefilter_tokens(&query)
    } else {
        Vec::new()
    };

    // Deterministic scan order: gather and sort all candidate paths.
    let mut files: Vec<PathBuf> = Vec::new();
    let root = Path::new(corpus_dir);
    if let Err(e) = collect_ir_files_recursively(root, &mut files) {
        eprintln!(
            "error: ir-query-corpus: failed to read corpus dir {}: {e}",
            root.display()
        );
        std::process::exit(2);
    }
    files.sort();

    let mut files_scanned = 0usize;
    let mut matches_emitted = 0usize;

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
                    "error: ir-query-corpus: failed to read {}: {e}",
                    path.display()
                );
                std::process::exit(2);
            }
        };

        if prefilter && !prefilter_tokens.is_empty() {
            let mut ok = true;
            for tok in &prefilter_tokens {
                if !file_content.contains(tok) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
        }

        let mut parser = ir_parser::Parser::new(&file_content);
        let mut pkg = match parser.parse_and_validate_package() {
            Ok(pkg) => pkg,
            Err(e) => {
                if ignore_parse_errors {
                    continue;
                }
                eprintln!(
                    "error: ir-query-corpus: parse/validate failed for {}: {e}",
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
                    "error: ir-query-corpus: failed to set --top for {}: {e}",
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
                    "error: ir-query-corpus: no top function found in {}",
                    path.display()
                );
                std::process::exit(2);
            }
        };

        let matches = ir_query::find_matching_nodes(top_fn, &query);
        if matches.is_empty() {
            continue;
        }

        for node_ref in matches {
            let node = top_fn.get_node(node_ref);
            let mut line = if let Some(line) = node.to_string(top_fn) {
                line
            } else {
                ir::node_textual_id(top_fn, node_ref)
            };
            if show_ret && top_fn.ret_node_ref == Some(node_ref) {
                line = format!("ret {}", line);
            }

            // Always show the file prefix in corpus mode.
            println!("{}: {}", path.display(), line);

            matches_emitted += 1;
            if let Some(limit) = max_matches {
                if matches_emitted >= limit {
                    return;
                }
            }
        }
    }
}
