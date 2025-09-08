// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use xlsynth_pir::{
    ir, ir_parser,
    ir_utils::get_topological,
    structural_similarity::{
        collect_backward_structural_entries, collect_structural_entries,
        compute_structural_discrepancies_dual,
        extract_dual_difference_subgraphs_with_shared_params_and_metadata,
    },
};

use crate::toolchain_config::ToolchainConfig;
use comfy_table::presets::ASCII_MARKDOWN;
use comfy_table::{ContentArrangement, Table};
use xlsynth_g8r::check_equivalence;

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

// Emit a node_table.txt summarizing each side's nodes with fwd/bwd hashes and
// diff flags.
fn hash_to_hex(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes.iter() {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
fn ir_fn_to_table(f: &ir::Fn, diff_region: &std::collections::HashSet<ir::NodeRef>) -> String {
    let (fwd_entries, _fwd_depths) = collect_structural_entries(f);
    let (bwd_entries, _bwd_depths) = collect_backward_structural_entries(f);
    let order = get_topological(f);
    let ret_idx_opt = f.ret_node_ref.map(|nr| nr.index);

    let mut table = Table::new();
    table.load_preset(ASCII_MARKDOWN);
    table.set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["node_name", "fwd_hash", "bwd_hash", "Δ"]);

    for nr in order.into_iter() {
        let name = ir::node_textual_id(f, nr);
        if name == "reserved_zero_node" {
            continue;
        }
        let is_ret = ret_idx_opt == Some(nr.index);
        let sigil = if is_ret { "*" } else { "" };
        let fwd_hex = hash_to_hex(fwd_entries[nr.index].hash.as_bytes());
        let bwd_hex = hash_to_hex(bwd_entries[nr.index].hash.as_bytes());
        let is_diff = if diff_region.contains(&nr) { "✓" } else { "" };
        table.add_row(vec![
            format!("{}{}", sigil, name),
            fwd_hex,
            bwd_hex,
            is_diff.to_string(),
        ]);
    }
    table.to_string()
}

fn print_equiv_result(label: &str, lhs_pkg_text: &str, rhs_pkg_text: &str, top_name: &str) {
    match check_equivalence::check_equivalence_with_top(
        lhs_pkg_text,
        rhs_pkg_text,
        Some(top_name),
        false,
    ) {
        Ok(()) => println!("  Equiv ({}): OK", label),
        Err(e) => println!("  Equiv ({}): FAILED: {}", label, e),
    }
}

pub fn handle_ir_structural_similarity(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let lhs = matches.get_one::<String>("lhs_ir_file").unwrap();
    let lhs_path = std::path::Path::new(lhs);
    let rhs = matches.get_one::<String>("rhs_ir_file").unwrap();
    let rhs_path = std::path::Path::new(rhs);
    let lhs_ir_top = matches.get_one::<String>("lhs_ir_top");
    let rhs_ir_top = matches.get_one::<String>("rhs_ir_top");

    // Prepare output directory: user-provided or a kept temp directory.
    let out_dir = if let Some(dir_str) = matches.get_one::<String>("output_dir") {
        let p = std::path::PathBuf::from(dir_str);
        if !p.exists() {
            std::fs::create_dir_all(&p).unwrap();
        }
        p
    } else {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().to_path_buf();
        std::mem::forget(td); // persist directory
        p
    };
    println!("  Output dir: {}", out_dir.display());
    // Copy original IR files for convenience/debugging.
    let lhs_copy_path = out_dir.join("lhs_orig.ir");
    let rhs_copy_path = out_dir.join("rhs_orig.ir");
    let _ = std::fs::copy(&lhs_path, &lhs_copy_path).expect("copy lhs IR");
    let _ = std::fs::copy(&rhs_path, &rhs_copy_path).expect("copy rhs IR");
    println!("  LHS IR copied to: {}", lhs_copy_path.display());
    println!("  RHS IR copied to: {}", rhs_copy_path.display());

    let lhs_pkg = ir_parser::parse_path_to_package(lhs_path).unwrap();
    let rhs_pkg = ir_parser::parse_path_to_package(rhs_path).unwrap();

    let lhs_fn = match lhs_ir_top {
        Some(top) => lhs_pkg.get_fn(top).unwrap(),
        None => lhs_pkg.get_top().unwrap(),
    };
    let rhs_fn = match rhs_ir_top {
        Some(top) => rhs_pkg.get_fn(top).unwrap(),
        None => rhs_pkg.get_top().unwrap(),
    };

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
    let lhs_sub = meta.lhs_inner;
    let rhs_sub = meta.rhs_inner;
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

    // Write diff packages: common outer (from LHS) + side-specific inner.
    let outer_text = ir::emit_fn_with_human_pos_comments(lhs_fn, &lhs_pkg.file_table);
    let lhs_inner_text = ir::emit_fn_with_human_pos_comments(&lhs_sub, &lhs_pkg.file_table);
    let rhs_inner_text = ir::emit_fn_with_human_pos_comments(&rhs_sub, &rhs_pkg.file_table);

    let lhs_diff_pkg = format!("package lhs_diff\n\n{}\n\n{}\n", outer_text, lhs_inner_text);
    let rhs_diff_pkg = format!("package rhs_diff\n\n{}\n\n{}\n", outer_text, rhs_inner_text);

    let lhs_diff_path = out_dir.join("lhs_diff.ir");
    let rhs_diff_path = out_dir.join("rhs_diff.ir");
    std::fs::write(&lhs_diff_path, lhs_diff_pkg.as_bytes()).unwrap();
    std::fs::write(&rhs_diff_path, rhs_diff_pkg.as_bytes()).unwrap();
    println!("  LHS diff IR written to: {}", lhs_diff_path.display());
    println!("  RHS diff IR written to: {}", rhs_diff_path.display());

    // Parse and verify the emitted diff packages; print results without panicking.
    {
        let mut p = ir_parser::Parser::new(&lhs_diff_pkg);
        match p.parse_and_validate_package() {
            Ok(_pkg) => println!("  LHS diff (PIR verify): OK"),
            Err(e) => println!("  LHS diff (PIR verify) FAILED: {}", e),
        }
    }
    {
        let mut p = ir_parser::Parser::new(&rhs_diff_pkg);
        match p.parse_and_validate_package() {
            Ok(_pkg) => println!("  RHS diff (PIR verify): OK"),
            Err(e) => println!("  RHS diff (PIR verify) FAILED: {}", e),
        }
    }

    // xlsynth parse + verify
    match xlsynth::IrPackage::parse_ir(&lhs_diff_pkg, None) {
        Ok(mut pkg) => {
            let _ = pkg.set_top_by_name(lhs_fn.name.as_str());
            match pkg.verify() {
                Ok(()) => println!("  LHS diff (xlsynth verify): OK"),
                Err(e) => println!("  LHS diff (xlsynth verify) FAILED: {}", e),
            }
        }
        Err(e) => println!("  LHS diff (xlsynth parse) FAILED: {}", e),
    }
    match xlsynth::IrPackage::parse_ir(&rhs_diff_pkg, None) {
        Ok(mut pkg) => {
            let _ = pkg.set_top_by_name(lhs_fn.name.as_str());
            match pkg.verify() {
                Ok(()) => println!("  RHS diff (xlsynth verify): OK"),
                Err(e) => println!("  RHS diff (xlsynth verify) FAILED: {}", e),
            }
        }
        Err(e) => println!("  RHS diff (xlsynth parse) FAILED: {}", e),
    }

    // Opportunistic equivalence checks using library-level equiv: (lhs_diff ≡ lhs_orig) and (rhs_diff ≡ rhs_orig).
    match std::fs::read_to_string(&lhs_copy_path) {
        Ok(lhs_orig_text) => {
            print_equiv_result(
                "lhs_diff ≡ lhs_orig",
                &lhs_diff_pkg,
                &lhs_orig_text,
                lhs_fn.name.as_str(),
            );
        }
        Err(e) => println!("  Equiv (lhs_diff ≡ lhs_orig): skipped (read error: {})", e),
    }
    match std::fs::read_to_string(&rhs_copy_path) {
        Ok(rhs_orig_text) => {
            print_equiv_result(
                "rhs_diff ≡ rhs_orig",
                &rhs_diff_pkg,
                &rhs_orig_text,
                lhs_fn.name.as_str(),
            );
        }
        Err(e) => println!("  Equiv (rhs_diff ≡ rhs_orig): skipped (read error: {})", e),
    }

    let lhs_table = ir_fn_to_table(lhs_fn, &meta.lhs_region);
    let rhs_table = ir_fn_to_table(rhs_fn, &meta.rhs_region);
    let mut table_text = String::new();
    table_text.push_str("LHS nodes:\n");
    table_text.push_str(&lhs_table);
    table_text.push_str("\n\nRHS nodes:\n");
    table_text.push_str(&rhs_table);
    let table_path = out_dir.join("node_table.txt");
    std::fs::write(&table_path, table_text.as_bytes()).unwrap();
    println!("  Node table written to: {}", table_path.display());
}
