// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::toolchain_config::ToolchainConfig;

pub fn handle_ir_structural_similarity(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let lhs = matches.get_one::<String>("lhs_ir_file").unwrap();
    let lhs_path = std::path::Path::new(lhs);
    let rhs = matches.get_one::<String>("rhs_ir_file").unwrap();
    let rhs_path = std::path::Path::new(rhs);
    let lhs_ir_top = matches.get_one::<String>("lhs_ir_top");
    let rhs_ir_top = matches.get_one::<String>("rhs_ir_top");

    let lhs_pkg = xlsynth_g8r::xls_ir::ir_parser::parse_path_to_package(lhs_path).unwrap();
    let rhs_pkg = xlsynth_g8r::xls_ir::ir_parser::parse_path_to_package(rhs_path).unwrap();

    let lhs_fn = match lhs_ir_top {
        Some(top) => lhs_pkg.get_fn(top).unwrap(),
        None => lhs_pkg.get_top().unwrap(),
    };
    let rhs_fn = match rhs_ir_top {
        Some(top) => rhs_pkg.get_fn(top).unwrap(),
        None => rhs_pkg.get_top().unwrap(),
    };

    let (recs, lhs_ret_depth, rhs_ret_depth) =
        xlsynth_g8r::xls_ir::structural_similarity::compute_structural_discrepancies_dual(
            lhs_fn, rhs_fn,
        );

    println!("LHS return depth: {}", lhs_ret_depth);
    println!("RHS return depth: {}", rhs_ret_depth);
    let show_details = match matches
        .get_one::<String>("show_discrepancies")
        .map(|s| s.as_str())
    {
        Some("true") => true,
        Some("false") => false,
        _ => false,
    };
    for rec in recs {
        let lhs_total: usize = rec.lhs_only.iter().map(|(_, c)| *c).sum();
        let rhs_total: usize = rec.rhs_only.iter().map(|(_, c)| *c).sum();
        println!("depth {}: {}", rec.depth, lhs_total + rhs_total);
        // Always print concise opcode summaries for this depth.
        let mut lhs_op_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut rhs_op_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let extract_op = |sig: &str| -> String {
            match sig.find('(') {
                Some(idx) => sig[..idx].to_string(),
                None => sig.to_string(),
            }
        };
        for (sig, c) in rec.lhs_only.iter() {
            let op = extract_op(sig);
            *lhs_op_counts.entry(op).or_insert(0) += *c;
        }
        for (sig, c) in rec.rhs_only.iter() {
            let op = extract_op(sig);
            *rhs_op_counts.entry(op).or_insert(0) += *c;
        }
        let mut lhs_ops: Vec<(String, usize)> = lhs_op_counts.into_iter().collect();
        let mut rhs_ops: Vec<(String, usize)> = rhs_op_counts.into_iter().collect();
        lhs_ops.sort_by(|a, b| a.0.cmp(&b.0));
        rhs_ops.sort_by(|a, b| a.0.cmp(&b.0));
        let fmt_map = |items: &Vec<(String, usize)>| -> String {
            let parts: Vec<String> = items.iter().map(|(k, v)| format!("{}: {}", k, v)).collect();
            format!("{{{}}}", parts.join(", "))
        };
        println!("  lhs: {}", fmt_map(&lhs_ops));
        println!("  rhs: {}", fmt_map(&rhs_ops));
        if show_details {
            for (s, c) in rec.lhs_only.iter() {
                if *c == 1 {
                    println!("  LHS has `{}` not present in RHS", s);
                } else {
                    println!("  LHS has {}x `{}` not present in RHS", c, s);
                }
            }
            for (s, c) in rec.rhs_only.iter() {
                if *c == 1 {
                    println!("  RHS has `{}` not present in LHS", s);
                } else {
                    println!("  RHS has {}x `{}` not present in LHS", c, s);
                }
            }
        }
    }

    // Also emit minimized subgraphs and metadata for the unmatched parts (dual
    // matching).
    let meta = xlsynth_g8r::xls_ir::structural_similarity::
        extract_dual_difference_subgraphs_with_shared_params_and_metadata(lhs_fn, rhs_fn);
    let lhs_sub = meta.lhs_inner;
    let rhs_sub = meta.rhs_inner;
    println!(
        "\nLHS diff subgraph:\n{}",
        xlsynth_g8r::xls_ir::ir::emit_fn_with_human_pos_comments(&lhs_sub, &lhs_pkg.file_table)
    );
    println!(
        "LHS inbound textual ids (unique): [{}]",
        meta.lhs_inbound_texts.join(", ")
    );
    println!("LHS outbound users per return element:");
    for (prod, users) in meta.lhs_outbound.iter() {
        println!("  {} -> [{}]", prod, users.join(", "));
    }
    println!(
        "\nRHS diff subgraph:\n{}",
        xlsynth_g8r::xls_ir::ir::emit_fn_with_human_pos_comments(&rhs_sub, &rhs_pkg.file_table)
    );
    println!(
        "RHS inbound textual ids (unique): [{}]",
        meta.rhs_inbound_texts.join(", ")
    );
    println!("RHS outbound users per return element:");
    for (prod, users) in meta.rhs_outbound.iter() {
        println!("  {} -> [{}]", prod, users.join(", "));
    }
}
