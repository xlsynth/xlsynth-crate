// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};

use clap::ArgMatches;
use xlsynth_g8r::equiv::prove_equiv::EquivResult;
use xlsynth_g8r::xls_ir::ir_validate;
use xlsynth_g8r::xls_ir::{ir, ir_outline};
use xlsynth_g8r::xls_ir::{ir_parser, structural_similarity};

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

pub fn handle_ir_structural_similarity(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let lhs = matches.get_one::<String>("lhs_ir_file").unwrap();
    let lhs_path = std::path::Path::new(lhs);
    let rhs = matches.get_one::<String>("rhs_ir_file").unwrap();
    let rhs_path = std::path::Path::new(rhs);
    let lhs_ir_top = matches.get_one::<String>("lhs_ir_top");
    let rhs_ir_top = matches.get_one::<String>("rhs_ir_top");

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
        let mut lhs_op_counts: HashMap<String, usize> = HashMap::new();
        let mut rhs_op_counts: HashMap<String, usize> = HashMap::new();
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

    // Print a cross-side alignment for each unified return slot, grouped by
    // consumer reverse-CSE (backward) structural hash and operand index. This
    // is cut-invariant and deduplicates textual-id differences.
    println!("\nUnified return slots (grouped by consumer backward structural hash):");
    // Build quick textual-id -> NodeRef lookup for each side.
    let mut lhs_text_to_ref: HashMap<String, xlsynth_g8r::xls_ir::ir::NodeRef> = HashMap::new();
    for (i, _n) in lhs_fn.nodes.iter().enumerate() {
        let nr = xlsynth_g8r::xls_ir::ir::NodeRef { index: i };
        let t = xlsynth_g8r::xls_ir::ir::node_textual_id(lhs_fn, nr);
        lhs_text_to_ref.insert(t, nr);
    }
    let mut rhs_text_to_ref: HashMap<String, xlsynth_g8r::xls_ir::ir::NodeRef> = HashMap::new();
    for (i, _n) in rhs_fn.nodes.iter().enumerate() {
        let nr = xlsynth_g8r::xls_ir::ir::NodeRef { index: i };
        let t = xlsynth_g8r::xls_ir::ir::node_textual_id(rhs_fn, nr);
        rhs_text_to_ref.insert(t, nr);
    }
    // Build backward-hash arrays per side (index-aligned with nodes)
    let (lhs_bwd_entries, _lhs_bwd_depths) =
        structural_similarity::collect_backward_structural_entries(lhs_fn);
    let (rhs_bwd_entries, _rhs_bwd_depths) =
        structural_similarity::collect_backward_structural_entries(rhs_fn);
    let lhs_bwd_hex_by_index: std::collections::HashMap<usize, String> = lhs_bwd_entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let bytes = e.hash.as_bytes();
            let mut s = String::with_capacity(16);
            for b in bytes.iter().take(8) {
                s.push_str(&format!("{:02x}", b));
            }
            (i, s)
        })
        .collect();
    let rhs_bwd_hex_by_index: std::collections::HashMap<usize, String> = rhs_bwd_entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let bytes = e.hash.as_bytes();
            let mut s = String::with_capacity(16);
            for b in bytes.iter().take(8) {
                s.push_str(&format!("{:02x}", b));
            }
            (i, s)
        })
        .collect();

    #[derive(Default)]
    struct Row {
        lhs_text: Option<String>,
        rhs_text: Option<String>,
        sig: Option<String>,
    }
    let mut rows: BTreeMap<(String, usize), Row> = BTreeMap::new();
    for (cons_text, op) in meta.slot_order.iter() {
        if let Some(&nr) = lhs_text_to_ref.get(cons_text) {
            let hh = lhs_bwd_hex_by_index
                .get(&nr.index)
                .cloned()
                .unwrap_or_else(|| "-".to_string());
            let key = (hh, *op);
            let e = rows.entry(key).or_default();
            e.lhs_text = Some(cons_text.clone());
            if e.sig.is_none() {
                e.sig = find_node_signature_by_textual_id(lhs_fn, cons_text);
            }
        }
        if let Some(&nr) = rhs_text_to_ref.get(cons_text) {
            let hh = rhs_bwd_hex_by_index
                .get(&nr.index)
                .cloned()
                .unwrap_or_else(|| "-".to_string());
            let key = (hh, *op);
            let e = rows.entry(key).or_default();
            e.rhs_text = Some(cons_text.clone());
            if e.sig.is_none() {
                e.sig = find_node_signature_by_textual_id(rhs_fn, cons_text);
            }
        }
    }
    for (i, ((hash_hex, op), row)) in rows.into_iter().enumerate() {
        let lhs = row.lhs_text.map(|t| t).unwrap_or_else(|| "-".to_string());
        let rhs = row.rhs_text.map(|t| t).unwrap_or_else(|| "-".to_string());
        let sig = row.sig.unwrap_or_else(|| "<unknown signature>".to_string());
        println!(
            "  {} -> op={} hash={} lhs_cons={} rhs_cons={}  {}",
            i, op, hash_hex, lhs, rhs, sig
        );
    }

    // Build a common outer graph for LHS that invokes the unified-signature LHS
    // callee.
    {
        let ordering = ir_outline::build_outline_ordering_from_unified_hash_spec_for_side(
            lhs_fn,
            &meta.lhs_region,
            &lhs_sub,
            &meta.slot_order_bwd,
            &meta.lhs_consumer_bwd_map,
        );
        let lhs_commoned = ir_outline::build_outer_with_existing_callee(
            lhs_fn,
            &meta.lhs_region,
            &format!("{}_common", lhs_fn.name),
            &lhs_sub.name,
            &ordering,
        );
        // Construct and dump LHS/RHS commoned packages for inspection.
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros();

        // Build RHS commoned outer with the same outer name for side-by-side
        // comparison.
        let rhs_ordering = ir_outline::build_outline_ordering_from_unified_hash_spec_for_side(
            rhs_fn,
            &meta.rhs_region,
            &rhs_sub,
            &meta.slot_order_bwd,
            &meta.rhs_consumer_bwd_map,
        );
        let rhs_commoned = ir_outline::build_outer_with_existing_callee(
            rhs_fn,
            &meta.rhs_region,
            &format!("{}_common", lhs_fn.name),
            &rhs_sub.name,
            &rhs_ordering,
        );

        let lhs_common_pkg = ir::Package {
            name: "lhs_common".to_string(),
            file_table: lhs_pkg.file_table.clone(),
            members: vec![
                ir::PackageMember::Function(lhs_sub.clone()),
                ir::PackageMember::Function(lhs_commoned.clone()),
            ],
            top_name: Some(format!("{}_common", lhs_fn.name)),
        };
        let rhs_common_pkg = ir::Package {
            name: "rhs_common".to_string(),
            file_table: rhs_pkg.file_table.clone(),
            members: vec![
                ir::PackageMember::Function(rhs_sub.clone()),
                ir::PackageMember::Function(rhs_commoned.clone()),
            ],
            top_name: Some(format!("{}_common", lhs_fn.name)),
        };

        let lhs_common_path = std::env::temp_dir().join(format!("xlsynth_lhs_common_{}.ir", ts));
        let rhs_common_path = std::env::temp_dir().join(format!("xlsynth_rhs_common_{}.ir", ts));
        if let Err(werr) = std::fs::write(&lhs_common_path, lhs_common_pkg.to_string()) {
            eprintln!(
                "Warning: failed to write LHS commoned IR to {}: {}",
                lhs_common_path.display(),
                werr
            );
        } else {
            println!("Wrote LHS commoned IR to {}", lhs_common_path.display());
        }
        if let Err(werr) = std::fs::write(&rhs_common_path, rhs_common_pkg.to_string()) {
            eprintln!(
                "Warning: failed to write RHS commoned IR to {}: {}",
                rhs_common_path.display(),
                werr
            );
        } else {
            println!("Wrote RHS commoned IR to {}", rhs_common_path.display());
        }
        if let Err(e) = ir_validate::validate_package(&lhs_common_pkg) {
            eprintln!("IR validate failed for LHS commoned package: {}", e);
            return;
        }
        if let Err(e) = ir_validate::validate_package(&rhs_common_pkg) {
            eprintln!("IR validate failed for RHS commoned package: {}", e);
            return;
        }
        // Only package-level validation is required.

        // Prove: original LHS == commoned LHS outer. Include the inner callee in
        // the RHS package in case the outer contains an invoke.
        let lhs_pkg_text = format!(
            "package lhs\n\n{}\n",
            format!("top {}\n", lhs_fn.to_string())
        );
        let rhs_pkg_text = format!(
            "package rhs\n\n{}\n\n{}\n",
            lhs_sub.to_string(),
            format!("top {}\n", lhs_commoned.to_string())
        );
        let lhs_dump = std::env::temp_dir().join(format!("xlsynth_lhs_pkg_{}.ir", ts));
        let rhs_dump = std::env::temp_dir().join(format!("xlsynth_rhs_pkg_{}.ir", ts));
        if let Err(werr) = std::fs::write(&lhs_dump, &lhs_pkg_text) {
            eprintln!(
                "Warning: failed to write LHS pkg IR to {}: {}",
                lhs_dump.display(),
                werr
            );
        } else {
            println!("Wrote LHS pkg IR to {}", lhs_dump.display());
        }
        if let Err(werr) = std::fs::write(&rhs_dump, &rhs_pkg_text) {
            eprintln!(
                "Warning: failed to write RHS pkg IR to {}: {}",
                rhs_dump.display(),
                werr
            );
        } else {
            println!("Wrote RHS pkg IR to {}", rhs_dump.display());
        }
        let tool_dir: String = if let Some(dir) = config.as_ref().and_then(|c| c.tool_path.clone())
        {
            dir
        } else {
            match std::env::var("XLSYNTH_TOOLS") {
                Ok(dir) => dir,
                Err(_) => {
                    eprintln!(
                        "LHS commoned outer equivalence: error: XLSYNTH_TOOLS not set and no tool_path in config"
                    );
                    return;
                }
            }
        };
        let res = xlsynth_g8r::equiv::prove_equiv_via_toolchain::prove_ir_pkg_equiv_with_tool_dir(
            &lhs_pkg_text,
            &rhs_pkg_text,
            None,
            &tool_dir,
        );
        match res {
            EquivResult::Proved => {
                println!("LHS commoned outer equivalence: success (proof closed)");
            }
            EquivResult::Disproved { .. } => {
                eprintln!("LHS commoned outer equivalence: FAILED");
            }
            EquivResult::Error(msg) => {
                eprintln!("LHS commoned outer equivalence: error: {}", msg);
            }
        }

        // Prove: original RHS == commoned RHS outer. Include the inner callee in
        // the LHS package in case the outer contains an invoke.
        let rhs_orig_pkg_text = format!(
            "package rhs\n\n{}\n",
            format!("top {}\n", rhs_fn.to_string())
        );
        let rhs_common_pkg_text = format!(
            "package rhs_common\n\n{}\n\n{}\n",
            rhs_sub.to_string(),
            format!("top {}\n", rhs_commoned.to_string())
        );
        let res_rhs =
            xlsynth_g8r::equiv::prove_equiv_via_toolchain::prove_ir_pkg_equiv_with_tool_dir(
                &rhs_orig_pkg_text,
                &rhs_common_pkg_text,
                None,
                &tool_dir,
            );
        match res_rhs {
            EquivResult::Proved => {
                println!("RHS commoned outer equivalence: success (proof closed)");
            }
            EquivResult::Disproved { .. } => {
                eprintln!("RHS commoned outer equivalence: FAILED");
            }
            EquivResult::Error(msg) => {
                eprintln!("RHS commoned outer equivalence: error: {}", msg);
            }
        }
    }
}
