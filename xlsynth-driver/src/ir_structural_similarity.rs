// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::ir_equiv::{dispatch_ir_equiv, EquivInputs};
use crate::parallelism::ParallelismStrategy;
use crate::toolchain_config::ToolchainConfig;
use xlsynth_g8r::xls_ir::structural_similarity::build_common_packages_from_lhs_rhs;

pub fn handle_ir_structural_similarity(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
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

    // Also emit minimized subgraphs capturing only the unmatched parts (dual
    // matching).
    let (lhs_sub, rhs_sub) =
        xlsynth_g8r::xls_ir::structural_similarity::extract_dual_difference_subgraphs(
            lhs_fn, rhs_fn,
        );
    println!(
        "\nLHS diff subgraph:\n{}",
        xlsynth_g8r::xls_ir::ir::emit_fn_with_human_pos_comments(&lhs_sub, &lhs_pkg.file_table)
    );
    println!(
        "\nRHS diff subgraph:\n{}",
        xlsynth_g8r::xls_ir::ir::emit_fn_with_human_pos_comments(&rhs_sub, &rhs_pkg.file_table)
    );

    // Optional: write side-by-side commonized packages whose tops preserve the
    // original function names/signatures and call into side-specific inner
    // impls for differing regions.
    if let Some(out_dir) = matches.get_one::<String>("diff_output_dir") {
        let out_path = std::path::Path::new(out_dir);
        if !out_path.exists() {
            std::fs::create_dir_all(out_path).expect("create diff output dir");
        }
        let (lhs_pkg_common, rhs_pkg_common) = build_common_packages_from_lhs_rhs(lhs_fn, rhs_fn);
        let lhs_text = lhs_pkg_common.to_string();
        let rhs_text = rhs_pkg_common.to_string();
        let lhs_out = out_path.join("lhs.ir");
        let rhs_out = out_path.join("rhs.ir");
        std::fs::write(&lhs_out, lhs_text.as_bytes()).expect("write lhs.ir");
        std::fs::write(&rhs_out, rhs_text.as_bytes()).expect("write rhs.ir");
        println!(
            "Wrote diff packages: {} and {}",
            lhs_out.display(),
            rhs_out.display()
        );

        // If toolchain path is configured, prove original LHS == LHS_common wrapper
        // and original RHS == RHS_common wrapper using the shared wrapper top.
        if let Some(tc) = config {
            if let Some(tool_path) = tc.tool_path.as_deref() {
                // Tops are preserved as the original function names on each side.
                let wrapper_top = lhs_fn.name.as_str();
                // LHS proof
                let lhs_ir_text = std::fs::read_to_string(lhs_path).unwrap();
                // Always pass the actual top function we used for LHS above.
                let lhs_top_opt: Option<&str> = Some(lhs_fn.name.as_str());
                let lhs_out_str = lhs_out.to_string_lossy().to_string();
                let lhs_inputs = EquivInputs {
                    lhs_ir_text: &lhs_ir_text,
                    rhs_ir_text: &lhs_text,
                    lhs_top: lhs_top_opt,
                    rhs_top: Some(wrapper_top),
                    flatten_aggregates: false,
                    drop_params: &[],
                    strategy: ParallelismStrategy::SingleThreaded,
                    assertion_semantics: xlsynth_g8r::equiv::prove_equiv::AssertionSemantics::Same,
                    lhs_fixed_implicit_activation: false,
                    rhs_fixed_implicit_activation: false,
                    subcommand: "ir-structural-similarity",
                    lhs_origin: lhs.as_str(),
                    rhs_origin: &lhs_out_str,
                    lhs_param_domains: None,
                    rhs_param_domains: None,
                    lhs_uf_map: std::collections::HashMap::new(),
                    rhs_uf_map: std::collections::HashMap::new(),
                };
                let lhs_outcome = dispatch_ir_equiv(None, Some(tool_path), &lhs_inputs);
                if lhs_outcome.success {
                    println!(
                        "[ir-structural-similarity] Proved LHS == LHS_common ({})",
                        wrapper_top
                    );
                } else {
                    eprintln!(
                        "[ir-structural-similarity] Failed to prove LHS == LHS_common: {}",
                        lhs_outcome
                            .counterexample
                            .unwrap_or_else(|| "(no details)".to_string())
                    );
                }

                // RHS proof
                let rhs_ir_text = std::fs::read_to_string(rhs_path).unwrap();
                // Always pass the actual top function we used for RHS above.
                let rhs_top_opt: Option<&str> = Some(rhs_fn.name.as_str());
                let rhs_out_str = rhs_out.to_string_lossy().to_string();
                let rhs_inputs = EquivInputs {
                    lhs_ir_text: &rhs_ir_text,
                    rhs_ir_text: &rhs_text,
                    lhs_top: rhs_top_opt,
                    rhs_top: Some(wrapper_top),
                    flatten_aggregates: false,
                    drop_params: &[],
                    strategy: ParallelismStrategy::SingleThreaded,
                    assertion_semantics: xlsynth_g8r::equiv::prove_equiv::AssertionSemantics::Same,
                    lhs_fixed_implicit_activation: false,
                    rhs_fixed_implicit_activation: false,
                    subcommand: "ir-structural-similarity",
                    lhs_origin: rhs.as_str(),
                    rhs_origin: &rhs_out_str,
                    lhs_param_domains: None,
                    rhs_param_domains: None,
                    lhs_uf_map: std::collections::HashMap::new(),
                    rhs_uf_map: std::collections::HashMap::new(),
                };
                let rhs_outcome = dispatch_ir_equiv(None, Some(tool_path), &rhs_inputs);
                if rhs_outcome.success {
                    println!(
                        "[ir-structural-similarity] Proved RHS == RHS_common ({})",
                        wrapper_top
                    );
                } else {
                    eprintln!(
                        "[ir-structural-similarity] Failed to prove RHS == RHS_common: {}",
                        rhs_outcome
                            .counterexample
                            .unwrap_or_else(|| "(no details)".to_string())
                    );
                }
            } else {
                println!(
                    "[ir-structural-similarity] Toolchain path not configured; skipping equivalence proofs"
                );
            }
        }
    }
}
