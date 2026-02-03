// SPDX-License-Identifier: Apache-2.0

use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;
use serde::Serialize;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_utils;

#[derive(Serialize)]
struct IrFnJsonOutput {
    package_name: String,
    selected_top: String,
    node_count: usize,
    return_type: String,
    pir: String,
    nodes: Vec<NodeRecord>,
}

#[derive(Serialize)]
struct PosRecord {
    fileno: usize,
    lineno: usize,
    colno: usize,
}

#[derive(Serialize)]
struct NodeRecord {
    index: usize,
    text_id: usize,
    name: Option<String>,
    op: String,
    ty: String,
    operands: Vec<usize>,
    is_param: bool,
    is_ret: bool,
    signature: String,
    text: Option<String>,
    pos: Option<Vec<PosRecord>>,
}

fn node_to_record(f: &ir::Fn, index: usize, node: &ir::Node) -> NodeRecord {
    let operands = ir_utils::operands(&node.payload)
        .into_iter()
        .map(|nr| nr.index)
        .collect();
    let pos = node.pos.as_ref().map(|positions| {
        positions
            .iter()
            .map(|p| PosRecord {
                fileno: p.fileno,
                lineno: p.lineno,
                colno: p.colno,
            })
            .collect()
    });

    NodeRecord {
        index,
        text_id: node.text_id,
        name: node.name.clone(),
        op: node.payload.get_operator().to_string(),
        ty: node.ty.to_string(),
        operands,
        is_param: matches!(node.payload, ir::NodePayload::GetParam(_)),
        is_ret: f.ret_node_ref.map(|nr| nr.index == index).unwrap_or(false),
        signature: node.to_signature_string(f),
        text: node.to_string(f),
        pos,
    }
}

pub fn handle_ir_fn_to_json(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches
        .get_one::<String>("ir_input_file")
        .expect("ir_input_file is required");

    let file_content = match std::fs::read_to_string(input_file) {
        Ok(content) => content,
        Err(e) => {
            report_cli_error_and_exit(
                &format!("Failed to read {}: {}", input_file, e),
                Some("ir-fn-to-json"),
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
                Some("ir-fn-to-json"),
                vec![],
            );
        }
    };

    if let Some(top) = matches.get_one::<String>("ir_top") {
        if let Err(e) = pkg.set_top_fn(top) {
            report_cli_error_and_exit(
                &format!("Failed to set --top: {}", e),
                Some("ir-fn-to-json"),
                vec![],
            );
        }
    }

    let top_fn = match pkg.get_top_fn() {
        Some(f) => f,
        None => {
            report_cli_error_and_exit(
                "No top function found in package",
                Some("ir-fn-to-json"),
                vec![],
            );
        }
    };

    let out = IrFnJsonOutput {
        package_name: pkg.name.clone(),
        selected_top: top_fn.name.clone(),
        node_count: top_fn.nodes.len(),
        return_type: top_fn.ret_ty.to_string(),
        pir: ir::emit_fn(top_fn, /* is_top= */ true),
        nodes: top_fn
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| node_to_record(top_fn, index, node))
            .collect(),
    };

    let json = serde_json::to_string_pretty(&out).expect("JSON serialization should not fail");
    println!("{json}");
}
