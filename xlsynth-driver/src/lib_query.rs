// SPDX-License-Identifier: Apache-2.0

use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;
use serde_json::json;
use std::path::Path;
use xlsynth_g8r::liberty::liberty_parser::{Block, BlockMember, Value};
use xlsynth_g8r::liberty::query::{parse_liberty_file_to_ast, parse_query, run_query};

fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::String(s) => json!({ "kind": "string", "value": s }),
        Value::Identifier(s) => json!({ "kind": "identifier", "value": s }),
        Value::Number(n) => json!({ "kind": "number", "value": n }),
        Value::Tuple(xs) => json!({
            "kind": "tuple",
            "value": xs.iter().map(|x| value_to_json(x.as_ref())).collect::<Vec<_>>()
        }),
    }
}

fn block_to_json(block: &Block) -> serde_json::Value {
    let mut attrs = Vec::new();
    let mut subblocks = Vec::new();
    for member in &block.members {
        match member {
            BlockMember::BlockAttr(attr) => {
                attrs.push(json!({
                    "name": attr.attr_name,
                    "value": value_to_json(&attr.value)
                }));
            }
            BlockMember::SubBlock(sb) => {
                subblocks.push(block_to_json(sb.as_ref()));
            }
        }
    }
    json!({
        "block_type": block.block_type,
        "qualifiers": block.qualifiers.iter().map(value_to_json).collect::<Vec<_>>(),
        "attrs": attrs,
        "subblocks": subblocks,
    })
}

pub fn handle_lib_query(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let liberty_file = matches
        .get_one::<String>("liberty_file")
        .expect("liberty_file is required");
    let query = matches
        .get_one::<String>("query")
        .expect("query is required");
    let max_matches = *matches
        .get_one::<usize>("max_matches")
        .expect("max_matches has default");
    let path_only = *matches.get_one::<bool>("path_only").unwrap_or(&false);
    let jsonl = *matches.get_one::<bool>("jsonl").unwrap_or(&false);

    let ast = match parse_liberty_file_to_ast(Path::new(liberty_file)) {
        Ok(ast) => ast,
        Err(e) => {
            report_cli_error_and_exit(
                &format!("Failed to parse liberty file '{}': {}", liberty_file, e),
                Some("lib-query"),
                vec![("liberty_file", liberty_file)],
            );
        }
    };

    let steps = match parse_query(query) {
        Ok(steps) => steps,
        Err(e) => {
            report_cli_error_and_exit(
                &format!("Failed to parse query '{}': {}", query, e),
                Some("lib-query"),
                vec![("query", query.as_str())],
            );
        }
    };

    let matches_found = run_query(&ast, &steps);
    if jsonl {
        for m in matches_found.iter().take(max_matches) {
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "path": m.path,
                    "block": block_to_json(m.block),
                }))
                .expect("json serialization should not fail")
            );
        }
    } else {
        println!(
            "query={} total_matches={} max_matches={}",
            query,
            matches_found.len(),
            max_matches
        );
        for (idx, m) in matches_found.iter().take(max_matches).enumerate() {
            println!("match[{idx}] path={}", m.path);
            if !path_only {
                println!("{:#?}", m.block);
            }
        }
    }
}
