// SPDX-License-Identifier: Apache-2.0

use blake3;
use clap::ArgMatches;
use comfy_table::{presets::ASCII_MARKDOWN, ContentArrangement, Table};
use xlsynth_pir::{
    ir, ir_parser,
    structural_similarity::{
        build_common_wrapper_lhs, collect_backward_structural_entries, collect_structural_entries,
        compute_structural_discrepancies_dual,
        extract_dual_difference_subgraphs_with_shared_params_and_metadata,
    },
};

use crate::toolchain_config::ToolchainConfig;

fn find_node_signature_by_textual_id(f: &ir::Fn, text: &str) -> Option<String> {
    for (i, _n) in f.nodes.iter().enumerate() {
        let nr = ir::NodeRef { index: i };
        let t = ir::node_textual_id(f, nr);
        if t == text {
            return Some(f.get_node(nr).to_signature_string(f));
        }
    }
    None
}

pub fn handle_ir_structural_similarity(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let lhs = matches.get_one::<String>("lhs_ir_file").unwrap();
    let lhs_path = std::path::Path::new(lhs);
    let rhs = matches.get_one::<String>("rhs_ir_file").unwrap();
    let rhs_path = std::path::Path::new(rhs);
    let lhs_ir_top = matches.get_one::<String>("lhs_ir_top");
    let rhs_ir_top = matches.get_one::<String>("rhs_ir_top");

    let lhs_pkg = match ir_parser::parse_and_validate_path_to_package(lhs_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "LHS IR failed to parse/validate ({}): {}",
                lhs_path.display(),
                e
            );
            std::process::exit(1);
        }
    };
    let rhs_pkg = match ir_parser::parse_and_validate_path_to_package(rhs_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "RHS IR failed to parse/validate ({}): {}",
                rhs_path.display(),
                e
            );
            std::process::exit(1);
        }
    };

    let lhs_fn = match lhs_ir_top {
        Some(top) => lhs_pkg.get_fn(top).unwrap(),
        None => lhs_pkg.get_top().unwrap(),
    };
    let rhs_fn = match rhs_ir_top {
        Some(top) => rhs_pkg.get_fn(top).unwrap(),
        None => rhs_pkg.get_top().unwrap(),
    };

    // Prepare deterministic diff artifacts directory keyed by input paths.
    let lhs_abs = std::fs::canonicalize(lhs_path).unwrap_or(lhs_path.to_path_buf());
    let rhs_abs = std::fs::canonicalize(rhs_path).unwrap_or(rhs_path.to_path_buf());
    let key = format!("{}::{}", lhs_abs.display(), rhs_abs.display());
    let hash_hex = blake3::hash(key.as_bytes()).to_hex().to_string();
    let dir_name = format!("xlsynth-structural-diff-{}", &hash_hex[..16]);
    let outdir_path = std::env::temp_dir().join(dir_name);
    let _ = std::fs::remove_dir_all(&outdir_path);
    let _ = std::fs::create_dir_all(&outdir_path);
    println!("Diff artifacts directory: {}", outdir_path.display());

    let (recs, lhs_ret_depth, rhs_ret_depth) =
        compute_structural_discrepancies_dual(lhs_fn, rhs_fn);

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
    let meta = extract_dual_difference_subgraphs_with_shared_params_and_metadata(lhs_fn, rhs_fn);
    let lhs_sub = meta.lhs_inner.clone();
    let rhs_sub = meta.rhs_inner.clone();
    // Unified return mapping before printing subgraphs.
    println!("\nUnified return type: {}", lhs_sub.ret_ty);
    println!("Unified return slots (index -> consumer[operand_index] : signature):");
    for (i, (cons, op)) in meta.slot_order.iter().enumerate() {
        let sig = find_node_signature_by_textual_id(lhs_fn, cons)
            .or_else(|| find_node_signature_by_textual_id(rhs_fn, cons))
            .unwrap_or_else(|| "<unknown signature>".to_string());
        println!("  {} -> {}[{}]  {}", i, cons, op, sig);
    }
    println!(
        "\nLHS diff subgraph:\n{}",
        ir::emit_fn_with_human_pos_comments(&lhs_sub, &lhs_pkg.file_table)
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
        ir::emit_fn_with_human_pos_comments(&rhs_sub, &rhs_pkg.file_table)
    );
    println!(
        "RHS inbound textual ids (unique): [{}]",
        meta.rhs_inbound_texts.join(", ")
    );
    println!("RHS outbound users per return element:");
    for (prod, users) in meta.rhs_outbound.iter() {
        println!("  {} -> [{}]", prod, users.join(", "));
    }

    // Append tables of all nodes with forward/backward hashes and diff-region flag.
    // Accumulate node table text to write to artifacts directory.
    let mut node_tables_text = String::new();

    // LHS graph table
    {
        let (fwd_entries, _fwd_depths) = collect_structural_entries(lhs_fn);
        let (bwd_entries, _bwd_depths) = collect_backward_structural_entries(lhs_fn);

        // Build stable ordering by textual id.
        let mut indices: Vec<usize> = (0..lhs_fn.nodes.len()).collect();
        indices.sort_by(|&a, &b| {
            let ta = ir::node_textual_id(lhs_fn, ir::NodeRef { index: a });
            let tb = ir::node_textual_id(lhs_fn, ir::NodeRef { index: b });
            ta.cmp(&tb)
        });

        println!("\nLHS graph");
        let mut table = Table::new();
        table
            .load_preset(ASCII_MARKDOWN)
            .set_content_arrangement(ContentArrangement::Disabled)
            .set_header(vec!["node_name", "hash", "rhash", "is_diff"]);
        for i in indices.into_iter() {
            let nref = ir::NodeRef { index: i };
            let name = ir::node_textual_id(lhs_fn, nref);
            if name == "reserved_zero_node" {
                continue;
            }
            let fh = fwd_entries[i].hash.to_hex().to_string();
            let rh = bwd_entries[i].hash.to_hex().to_string();
            let mark = if meta.lhs_region.contains(&nref) {
                "✓"
            } else {
                ""
            };
            table.add_row(vec![name, fh, rh, mark.to_string()]);
        }
        println!("{}", table);
        node_tables_text.push_str("LHS graph\n");
        node_tables_text.push_str(&format!("{}\n\n", table));
    }

    // RHS graph table
    {
        let (fwd_entries, _fwd_depths) = collect_structural_entries(rhs_fn);
        let (bwd_entries, _bwd_depths) = collect_backward_structural_entries(rhs_fn);

        let mut indices: Vec<usize> = (0..rhs_fn.nodes.len()).collect();
        indices.sort_by(|&a, &b| {
            let ta = ir::node_textual_id(rhs_fn, ir::NodeRef { index: a });
            let tb = ir::node_textual_id(rhs_fn, ir::NodeRef { index: b });
            ta.cmp(&tb)
        });

        println!("\nRHS graph");
        let mut table = Table::new();
        table
            .load_preset(ASCII_MARKDOWN)
            .set_content_arrangement(ContentArrangement::Disabled)
            .set_header(vec!["node_name", "hash", "rhash", "is_diff"]);
        for i in indices.into_iter() {
            let nref = ir::NodeRef { index: i };
            let name = ir::node_textual_id(rhs_fn, nref);
            if name == "reserved_zero_node" {
                continue;
            }
            let fh = fwd_entries[i].hash.to_hex().to_string();
            let rh = bwd_entries[i].hash.to_hex().to_string();
            let mark = if meta.rhs_region.contains(&nref) {
                "✓"
            } else {
                ""
            };
            table.add_row(vec![name, fh, rh, mark.to_string()]);
        }
        println!("{}", table);
        node_tables_text.push_str("RHS graph\n");
        node_tables_text.push_str(&format!("{}\n", table));
    }

    // Write node tables to artifacts directory.
    let node_tables_path = outdir_path.join("node_tables.txt");
    let _ = std::fs::write(&node_tables_path, node_tables_text);

    // Build common wrappers for LHS and RHS.
    let lhs_common = build_common_wrapper_lhs(lhs_fn, &meta);
    // Note: We currently only build LHS common; extend to RHS similarly if needed
    // elsewhere.
    let rhs_common = xlsynth_pir::structural_similarity::build_common_wrapper_rhs(rhs_fn, &meta);

    // Emit full IR packages for lhs_diff.ir and rhs_diff.ir with inner before
    // common wrapper.
    let mut lhs_pkg_out = ir::Package {
        name: format!("{}_diff", lhs_pkg.name),
        file_table: lhs_pkg.file_table.clone(),
        members: vec![
            ir::PackageMember::Function(lhs_sub.clone()),
            ir::PackageMember::Function(lhs_common.clone()),
        ],
        top_name: Some(lhs_common.name.clone()),
    };
    let mut rhs_pkg_out = ir::Package {
        name: format!("{}_diff", rhs_pkg.name),
        file_table: rhs_pkg.file_table.clone(),
        members: vec![
            ir::PackageMember::Function(rhs_sub.clone()),
            ir::PackageMember::Function(rhs_common.clone()),
        ],
        top_name: Some(rhs_common.name.clone()),
    };
    let lhs_pkg_text = lhs_pkg_out.to_string();
    let rhs_pkg_text = rhs_pkg_out.to_string();
    let lhs_path = outdir_path.join("lhs_diff.ir");
    let rhs_path = outdir_path.join("rhs_diff.ir");
    let _ = std::fs::write(&lhs_path, lhs_pkg_text.clone());
    let _ = std::fs::write(&rhs_path, rhs_pkg_text.clone());

    // Also emit the original packages for reference.
    let _ = std::fs::write(outdir_path.join("lhs_orig.ir"), lhs_pkg.to_string());
    let _ = std::fs::write(outdir_path.join("rhs_orig.ir"), rhs_pkg.to_string());

    // Verify generated packages parse and validate; print detailed errors.
    let lhs_result = {
        let mut p = ir_parser::Parser::new(&lhs_pkg_text);
        p.parse_and_validate_package()
    };
    let rhs_result = {
        let mut p = ir_parser::Parser::new(&rhs_pkg_text);
        p.parse_and_validate_package()
    };
    println!("\nGenerated IR verification:");
    match lhs_result {
        Ok(_) => println!("  LHS package: ok ({})", lhs_path.display()),
        Err(e) => println!("  LHS package: FAILED ({})\n    {}", lhs_path.display(), e),
    }
    match rhs_result {
        Ok(_) => println!("  RHS package: ok ({})", rhs_path.display()),
        Err(e) => println!("  RHS package: FAILED ({})\n    {}", rhs_path.display(), e),
    }
}
