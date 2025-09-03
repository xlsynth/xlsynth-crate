// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::ir_equiv::{dispatch_ir_equiv, EquivInputs};
use crate::parallelism::ParallelismStrategy;
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use rand::Rng;
use rand::SeedableRng;
use xlsynth::IrValue;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::equiv::prove_equiv::AssertionSemantics;
use xlsynth_g8r::xls_ir::ir::Type;
use xlsynth_g8r::xls_ir::ir::{self as ir_mod, BlockPortInfo, PackageMember};
use xlsynth_g8r::xls_ir::ir_parser::emit_fn_as_block;

#[derive(serde::Serialize)]
struct AddedOpsSummaryItem {
    op: String,
    count: usize,
}

#[derive(serde::Serialize)]
struct LocalizedEcoReport {
    added_node_count: usize,
    added_ops: Vec<AddedOpsSummaryItem>,
    edit_distance_old_to_patched: u64,
    text_edit_distance_old_to_patched: usize,
    rtl_text_edit_distance_old_to_patched: usize,
}

pub fn handle_ir_localized_eco(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let old_path = std::path::Path::new(matches.get_one::<String>("old_ir_file").unwrap());
    let new_path = std::path::Path::new(matches.get_one::<String>("new_ir_file").unwrap());
    let old_ir_top = matches.get_one::<String>("old_ir_top");
    let new_ir_top = matches.get_one::<String>("new_ir_top");

    // Read inputs to detect whether they are package IR or standalone block IR.
    let old_text = match std::fs::read_to_string(old_path) {
        Ok(s) => s,
        Err(e) => report_cli_error_and_exit(
            &format!(
                "could not read old IR file; path: {}; error: {}",
                old_path.display(),
                e
            ),
            Some("ir-localized-eco"),
            vec![],
        ),
    };
    let new_text = match std::fs::read_to_string(new_path) {
        Ok(s) => s,
        Err(e) => report_cli_error_and_exit(
            &format!(
                "could not read new IR file; path: {}; error: {}",
                new_path.display(),
                e
            ),
            Some("ir-localized-eco"),
            vec![],
        ),
    };
    let old_trimmed = old_text.trim_start();
    let new_trimmed = new_text.trim_start();
    if !old_trimmed.starts_with("package") || !new_trimmed.starts_with("package") {
        report_cli_error_and_exit(
            &format!(
                "expected package-form IR starting with 'package'; got old starts_with_package={} new starts_with_package={}",
                old_trimmed.starts_with("package"),
                new_trimmed.starts_with("package")
            ),
            Some("ir-localized-eco"),
            vec![],
        );
    }

    let old_pkg = match xlsynth_g8r::xls_ir::ir_parser::parse_path_to_package(old_path) {
        Ok(p) => p,
        Err(e) => {
            let path_str = old_path.display().to_string();
            let err_str = e.to_string();
            let trunc = truncate_for_cli(&err_str, 1024);
            let msg = format!(
                "failed to parse old IR package; path: {}; error: {}",
                path_str, trunc
            );
            report_cli_error_and_exit(&msg, Some("ir-localized-eco"), vec![])
        }
    };
    let new_pkg = match xlsynth_g8r::xls_ir::ir_parser::parse_path_to_package(new_path) {
        Ok(p) => p,
        Err(e) => {
            let path_str = new_path.display().to_string();
            let err_str = e.to_string();
            let trunc = truncate_for_cli(&err_str, 1024);
            let msg = format!(
                "failed to parse new IR package; path: {}; error: {}",
                path_str, trunc
            );
            report_cli_error_and_exit(&msg, Some("ir-localized-eco"), vec![])
        }
    };

    // If both packages contain at least one block member, operate on blocks.
    if let (Some((old_block_fn, old_ports)), Some((new_block_fn, new_ports))) = (
        select_block_from_package(&old_pkg, old_ir_top.as_deref().map(|x| x.as_str())),
        select_block_from_package(&new_pkg, new_ir_top.as_deref().map(|x| x.as_str())),
    ) {
        return handle_ir_localized_eco_blocks_in_packages(
            matches,
            old_path,
            new_path,
            &old_text,
            &new_text,
            old_block_fn,
            old_ports,
            new_block_fn,
            new_ports,
        );
    }

    let old_fn = match old_ir_top {
        Some(top) => match old_pkg.get_fn(top) {
            Some(f) => f,
            None => report_cli_error_and_exit(
                "old entry function not found",
                Some("ir-localized-eco"),
                vec![("name", top)],
            ),
        },
        None => match old_pkg.get_top() {
            Some(f) => f,
            None => {
                let msg = format!(
                    "no top function found in old package; path: {}",
                    old_path.display()
                );
                report_cli_error_and_exit(&msg, Some("ir-localized-eco"), vec![])
            }
        },
    };
    let new_fn = match new_ir_top {
        Some(top) => match new_pkg.get_fn(top) {
            Some(f) => f,
            None => report_cli_error_and_exit(
                "new entry function not found",
                Some("ir-localized-eco"),
                vec![("name", top)],
            ),
        },
        None => match new_pkg.get_top() {
            Some(f) => f,
            None => {
                let msg = format!(
                    "no top function found in new package; path: {}",
                    new_path.display()
                );
                report_cli_error_and_exit(&msg, Some("ir-localized-eco"), vec![])
            }
        },
    };

    // Build patched function via structural rebase; compute simple node-add count.
    let patched_for_count =
        xlsynth_g8r::xls_ir::localized_eco2::compute_localized_eco(old_fn, new_fn);
    let added_count: usize = patched_for_count
        .nodes
        .len()
        .saturating_sub(old_fn.nodes.len());
    let added_ops: Vec<AddedOpsSummaryItem> = Vec::new();

    let report = LocalizedEcoReport {
        added_node_count: added_count,
        added_ops,
        edit_distance_old_to_patched: 0,
        text_edit_distance_old_to_patched: 0, // placeholder; set below when we have texts
        rtl_text_edit_distance_old_to_patched: 0,
    };

    // Decide output directory: user-provided or temp directory we keep.
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

    // JSON path: use explicit --json_out if given, else write into
    // out_dir/eco_report.json
    let json_path = if let Some(json_out) = matches.get_one::<String>("json_out") {
        let path = std::path::PathBuf::from(json_out);
        // We'll serialize after computing text edit distance below; write later.
        // For now, just return the path.
        path
    } else {
        let path = out_dir.join("eco_report.json");
        // We'll serialize after computing text edit distance below.
        path
    };

    // Patched IR path: emit the NEW IR with IDs remapped to preserve old IDs where
    // subgraphs are structurally equal; allocate fresh IDs for new nodes.
    let patched_ir_path = out_dir.join("patched_old.ir");
    // Re-emit both old and patched packages to ensure a comparable formatting
    // baseline.
    let old_ir_text_emitted = old_pkg.to_string();
    // Build patched package by constructing a rebase-based patched function
    // that preserves existing node IDs where structurally reusable, allocating
    // new ones only for synthesized nodes.
    let mut patched_pkg = old_pkg.clone();
    if let Some(target_fn) = patched_pkg.get_fn_mut(&old_fn.name) {
        let applied = xlsynth_g8r::xls_ir::localized_eco2::compute_localized_eco(old_fn, new_fn);
        *target_fn = applied;
    }
    let patched_ir_text_emitted = patched_pkg.to_string();
    std::fs::write(&patched_ir_path, patched_ir_text_emitted.as_bytes()).unwrap();
    // Inform the user where outputs are going before expensive text diffing.
    println!("  Output dir: {}", out_dir.display());
    println!("  Patched IR written to: {}", patched_ir_path.display());
    // Copy input IRs into the output directory for convenience.
    let old_copy_path = out_dir.join("old.ir");
    let new_copy_path = out_dir.join("new.ir");
    std::fs::copy(&old_path, &old_copy_path).expect("copy old IR");
    std::fs::copy(&new_path, &new_copy_path).expect("copy new IR");
    println!("  Old IR copied to: {}", old_copy_path.display());
    println!("  New IR copied to: {}", new_copy_path.display());

    // Run local validations on the 'new' function (mirrors patched IR) to catch
    // common issues like duplicate IDs before invoking the external toolchain.
    if let Err(e) = xlsynth_g8r::xls_ir::ir_verify::verify_fn_unique_node_ids(new_fn) {
        println!("  WARNING: verification failed (duplicate IDs): {}", e);
    }
    if let Err(e) = xlsynth_g8r::xls_ir::ir_verify::verify_fn_operand_indices_in_bounds(new_fn) {
        println!("  WARNING: verification failed (operand indices): {}", e);
    }

    // Human-readable output
    println!("Localized ECO (rebase-based) summary");
    println!("  New nodes added: {}", report.added_node_count);
    println!(
        "  IR Graph Edit Distance (old → patched(old)): {}",
        report.edit_distance_old_to_patched
    );
    let compute_text_diff: bool = matches
        .get_one::<String>("compute_text_diff")
        .map(|s| s == "true")
        .unwrap_or(false);
    let mut text_diff_chars: usize = 0;
    let mut rtl_diff_chars: usize = 0;
    if compute_text_diff {
        let (ir_chars, rtl_chars) = compute_package_text_diffs(
            &old_ir_text_emitted,
            &patched_ir_text_emitted,
            &new_fn.name,
            &out_dir,
        );
        text_diff_chars = ir_chars;
        rtl_diff_chars = rtl_chars;
    }

    // Serialize report with both IR and RTL text edit distances now known.
    {
        let mut report_with_text = report;
        report_with_text.text_edit_distance_old_to_patched = text_diff_chars;
        report_with_text.rtl_text_edit_distance_old_to_patched = rtl_diff_chars;
        let s = serde_json::to_string_pretty(&report_with_text).unwrap();
        std::fs::write(&json_path, s).unwrap();
    }
    println!("  JSON written to: {}", json_path.display());

    // Optional: quick interpreter sanity check before expensive proof.
    let sanity_samples = matches
        .get_one::<String>("sanity_samples")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let sanity_seed = matches
        .get_one::<String>("sanity_seed")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    if sanity_samples > 0 {
        match sanity_check_interpret(
            &patched_ir_text_emitted,
            new_fn,
            &patched_ir_path,
            sanity_samples,
            sanity_seed,
        ) {
            Ok(()) => {
                // The function prints its own summary including skipped counts.
            }
            Err(e) => {
                println!("  Sanity check: FAILED: {}", e);
                println!("  Equivalence: skipped due to failing interpreter sanity check");
                return;
            }
        }
    }
    // Optional: Prove old vs new equivalence using toolchain if available.
    // Determine tool_path: prefer --toolchain config; otherwise use XLSYNTH_TOOLS
    // env var if set.
    let mut tool_path_opt: Option<String> = config.as_ref().and_then(|c| c.tool_path.clone());
    if tool_path_opt.is_none() {
        if let Ok(env_tools) = std::env::var("XLSYNTH_TOOLS") {
            if !env_tools.trim().is_empty() {
                tool_path_opt = Some(env_tools);
            }
        }
    }

    if let Some(tool_path) = tool_path_opt.as_deref() {
        // Prove: patched_old.ir ≡ new.ir
        let patched_ir_text = std::fs::read_to_string(&patched_ir_path).unwrap();
        let new_ir_text = std::fs::read_to_string(new_path).unwrap();
        // Use the new function's name as the top on both sides (patched equals new
        // package text).
        let lhs_top = Some(new_fn.name.as_str());
        let rhs_top = Some(new_fn.name.as_str());
        println!(
            "  Starting equivalence proof using toolchain at {}: patched='{}' top='{}' vs new='{}' top='{}'",
            tool_path,
            patched_ir_path.display(),
            lhs_top.unwrap_or("") ,
            new_path.display(),
            rhs_top.unwrap_or("")
        );
        let inputs = EquivInputs {
            lhs_ir_text: &patched_ir_text,
            rhs_ir_text: &new_ir_text,
            lhs_top,
            rhs_top,
            flatten_aggregates: false,
            drop_params: &[],
            strategy: ParallelismStrategy::SingleThreaded,
            assertion_semantics: AssertionSemantics::Same,
            lhs_fixed_implicit_activation: false,
            rhs_fixed_implicit_activation: false,
            subcommand: "ir-localized-eco",
            lhs_origin: old_path.to_str().unwrap_or(""),
            rhs_origin: new_path.to_str().unwrap_or(""),
            lhs_param_domains: None,
            rhs_param_domains: None,
            lhs_uf_map: std::collections::HashMap::new(),
            rhs_uf_map: std::collections::HashMap::new(),
        };
        let outcome = dispatch_ir_equiv(None, Some(tool_path), &inputs);
        let dur = std::time::Duration::from_micros(outcome.time_micros as u64);
        if outcome.success {
            println!("  Equivalence: proved (patched(old) ≡ new) in {:?}", dur);
        } else {
            println!("  Equivalence: FAILED (patched(old) vs new) in {:?}", dur);
            if let Some(cex) = outcome.counterexample {
                println!("    counterexample: {}", cex);
                // Attempt to replay the counterexample via interpreter.
                if let Some(input_idx) = cex.find("input:") {
                    let arg_text = cex[input_idx + "input:".len()..].trim();
                    match try_interpret_cex(&new_ir_text, new_fn, &patched_ir_path, arg_text) {
                        Ok(()) => {}
                        Err(e) => println!("    interpreter replay: skipped ({})", e),
                    }
                }
            }
        }
    } else {
        println!("  Equivalence: skipped (no --toolchain config and no XLSYNTH_TOOLS env var)");
    }
}

fn truncate_for_cli(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    // Find a valid UTF-8 boundary at or before max_len.
    let mut cut = 0;
    for (i, _) in s.char_indices() {
        if i <= max_len {
            cut = i;
        } else {
            break;
        }
    }
    format!("{} ... [{} bytes truncated]", &s[..cut], s.len() - cut)
}

fn get_output_types_for_emission(f: &ir_mod::Fn, expected_outputs: usize) -> Vec<ir_mod::Type> {
    if expected_outputs == 0 {
        return Vec::new();
    }
    if let Some(ret_nr) = f.ret_node_ref {
        let ret_node = f.get_node(ret_nr);
        if expected_outputs == 1 {
            return vec![ret_node.ty.clone()];
        }
        match &ret_node.payload {
            ir_mod::NodePayload::Tuple(_elems) => {
                if let ir_mod::Type::Tuple(tys) = &ret_node.ty {
                    return tys.iter().map(|t| (**t).clone()).collect();
                }
                vec![ret_node.ty.clone()]
            }
            _ => vec![ret_node.ty.clone()],
        }
    } else {
        Vec::new()
    }
}

fn select_block_from_package<'a>(
    pkg: &'a ir_mod::Package,
    name_opt: Option<&str>,
) -> Option<(&'a ir_mod::Fn, &'a BlockPortInfo)> {
    if let Some(name) = name_opt {
        for m in pkg.members.iter() {
            if let PackageMember::Block { func, port_info } = m {
                if func.name == name {
                    return Some((func, port_info));
                }
            }
        }
        return None;
    }
    if let Some(top_name) = &pkg.top_name {
        for m in pkg.members.iter() {
            if let PackageMember::Block { func, port_info } = m {
                if &func.name == top_name {
                    return Some((func, port_info));
                }
            }
        }
    }
    for m in pkg.members.iter() {
        if let PackageMember::Block { func, port_info } = m {
            return Some((func, port_info));
        }
    }
    None
}

fn handle_ir_localized_eco_blocks_in_packages(
    matches: &ArgMatches,
    _old_path: &std::path::Path,
    _new_path: &std::path::Path,
    _old_text: &str,
    _new_text: &str,
    old_fn: &ir_mod::Fn,
    old_ports: &BlockPortInfo,
    new_fn: &ir_mod::Fn,
    new_ports: &BlockPortInfo,
) {
    // Summaries (rebase-based): will compute added_count after building applied.
    let added_ops: Vec<AddedOpsSummaryItem> = Vec::new();

    // Prepare output directory.
    let out_dir = if let Some(dir_str) = matches.get_one::<String>("output_dir") {
        let p = std::path::PathBuf::from(dir_str);
        if !p.exists() {
            std::fs::create_dir_all(&p).unwrap();
        }
        p
    } else {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().to_path_buf();
        std::mem::forget(td);
        p
    };
    println!("  Output dir: {}", out_dir.display());

    // Build patched(old) via structural rebase.
    println!("  Building patched block via structural rebase...");
    let applied = xlsynth_g8r::xls_ir::localized_eco2::compute_localized_eco(old_fn, new_fn);
    let added_count: usize = applied.nodes.len().saturating_sub(old_fn.nodes.len());

    // Validate output arity compatibility with old block port info.
    let applied_out_types = get_output_types_for_emission(&applied, old_ports.output_names.len());
    if old_ports.output_names.len() != applied_out_types.len() {
        let msg = format!(
            "output arity mismatch: old block had output ports {:?}; function outputs are {:?} ({}).",
            old_ports.output_names,
            applied_out_types.iter().map(|t| t.to_string()).collect::<Vec<_>>(),
            applied_out_types.len()
        );
        report_cli_error_and_exit(&msg, Some("ir-localized-eco"), vec![]);
    }
    println!("  Emitting patched block text...");
    let patched_block_text = emit_fn_as_block(&applied, None, Some(old_ports));
    let patched_ir_path = out_dir.join("patched_old.block.ir");
    std::fs::write(&patched_ir_path, patched_block_text.as_bytes()).unwrap();
    println!("  Patched IR written to: {}", patched_ir_path.display());

    // Copy old/new for convenience: write ONLY the selected blocks.
    let old_copy_path = out_dir.join("old.ir");
    let new_copy_path = out_dir.join("new.ir");
    let old_block_text = emit_fn_as_block(old_fn, None, Some(old_ports));
    let new_block_text = emit_fn_as_block(new_fn, None, Some(new_ports));
    std::fs::write(&old_copy_path, old_block_text.as_bytes()).unwrap();
    std::fs::write(&new_copy_path, new_block_text.as_bytes()).unwrap();
    println!("  Old IR copied to: {}", old_copy_path.display());
    println!("  New IR copied to: {}", new_copy_path.display());

    // Human-readable summary.
    println!("Localized ECO (rebase-based) summary");
    println!("  New nodes added: {}", added_count);

    // Optional: compute simple text diff for block text.
    let compute_text_diff: bool = matches
        .get_one::<String>("compute_text_diff")
        .map(|s| s == "true")
        .unwrap_or(false);
    let text_diff_chars: usize = if compute_text_diff {
        compute_block_text_diff(&old_block_text, &patched_block_text)
    } else {
        0
    };

    // Serialize JSON report analogous to package path.
    let json_path = if let Some(json_out) = matches.get_one::<String>("json_out") {
        std::path::PathBuf::from(json_out)
    } else {
        out_dir.join("eco_report.json")
    };
    println!("  Serializing JSON report...");
    let report = LocalizedEcoReport {
        added_node_count: added_count,
        added_ops,
        edit_distance_old_to_patched: 0,
        text_edit_distance_old_to_patched: text_diff_chars,
        rtl_text_edit_distance_old_to_patched: 0,
    };
    let s = serde_json::to_string_pretty(&report).unwrap();
    std::fs::write(&json_path, s).unwrap();
    println!("  JSON written to: {}", json_path.display());
    println!("  Done.");

    // Attempt equivalence by wrapping the functions into minimal packages and
    // invoking the external checker, if tools are configured.
    match std::env::var("XLSYNTH_TOOLS") {
        Ok(p) if !p.trim().is_empty() => {
            let lhs_pkg = format!("package lhs\n\ntop {}", applied.to_string());
            let rhs_pkg = format!("package rhs\n\ntop {}", new_fn.to_string());
            let top_name = Some(new_fn.name.as_str());
            match check_equivalence::check_equivalence_with_top(&lhs_pkg, &rhs_pkg, top_name, false)
            {
                Ok(()) => println!("  Equivalence: proved (patched(old) ≡ new)"),
                Err(e) => println!("  Equivalence: FAILED: {}", e),
            }
        }
        _ => {
            println!("  Equivalence: skipped (no XLSYNTH_TOOLS env var)");
        }
    }
}

// Compute Levenshtein distance over bytes (ASCII-safe for IR text), O(n*m).
// Myers' O(ND) diff distance over bytes (insert+delete only); returns minimal
// number of inserted + deleted bytes to transform a into b.
fn myers_insdel_distance_bytes(a: &[u8], b: &[u8]) -> usize {
    let n = a.len() as isize;
    let m = b.len() as isize;
    if n == 0 {
        return m as usize;
    }
    if m == 0 {
        return n as usize;
    }
    let max = (n + m) as usize;
    let offset = max as isize;
    let mut v: Vec<isize> = vec![0; 2 * max + 1];
    for d in 0..=max {
        let d_isize = d as isize;
        let mut k = -d_isize;
        while k <= d_isize {
            let idx_plus = (k + 1 + offset) as usize;
            let idx_minus = (k - 1 + offset) as usize;
            let x = if k == -d_isize || (k != d_isize && v[idx_minus] < v[idx_plus]) {
                v[idx_plus]
            } else {
                v[idx_minus] + 1
            };
            let mut x_mut = x;
            let mut y_mut = x_mut - k;
            while x_mut < n && y_mut < m && a[x_mut as usize] == b[y_mut as usize] {
                x_mut += 1;
                y_mut += 1;
            }
            v[(k + offset) as usize] = x_mut;
            if x_mut >= n && y_mut >= m {
                return d;
            }
            k += 2;
        }
    }
    max
}

fn compute_block_text_diff(old_block_text: &str, patched_block_text: &str) -> usize {
    println!(
        "  Computing text diff {} bytes vs {} bytes...",
        old_block_text.as_bytes().len(),
        patched_block_text.as_bytes().len()
    );
    let a = old_block_text.as_bytes();
    let b = patched_block_text.as_bytes();
    let d = myers_insdel_distance_bytes(a, b);
    println!("  Text diff char count (old → patched(old)): {}", d);
    d
}

fn compute_package_text_diffs(
    old_ir_text_emitted: &str,
    patched_ir_text_emitted: &str,
    new_fn_name: &str,
    out_dir: &std::path::Path,
) -> (usize, usize) {
    println!(
        "  Computing text diff char count (Myers, inserts+deletes) for IR text old → patched(old)..."
    );
    let text_diff_chars = myers_insdel_distance_bytes(
        old_ir_text_emitted.as_bytes(),
        patched_ir_text_emitted.as_bytes(),
    );
    println!(
        "  Text diff char count (old → patched(old)): {}",
        text_diff_chars
    );

    let old_pkg_x = xlsynth::IrPackage::parse_ir(old_ir_text_emitted, None)
        .expect("parse old IR for RTL codegen");
    let patched_pkg_x = xlsynth::IrPackage::parse_ir(patched_ir_text_emitted, None)
        .expect("parse patched IR for RTL codegen");
    let mut old_pkg_x = old_pkg_x;
    let mut patched_pkg_x = patched_pkg_x;
    let _ = old_pkg_x.set_top_by_name(new_fn_name);
    let _ = patched_pkg_x.set_top_by_name(new_fn_name);
    let delay_model = "unit";
    let sched_proto = format!("delay_model: \"{}\"\npipeline_stages: 1", delay_model);
    let codegen_proto = format!(
        "register_merge_strategy: STRATEGY_IDENTITY_ONLY\ngenerator: GENERATOR_KIND_PIPELINE\nuse_system_verilog: true\nmodule_name: \"{}\"\ncodegen_version: 1",
        new_fn_name
    );
    let old_sv = xlsynth::schedule_and_codegen(&old_pkg_x, &sched_proto, &codegen_proto)
        .and_then(|res| res.get_verilog_text())
        .expect("schedule/codegen old IR");
    let patched_sv = xlsynth::schedule_and_codegen(&patched_pkg_x, &sched_proto, &codegen_proto)
        .and_then(|res| res.get_verilog_text())
        .expect("schedule/codegen patched IR");
    let old_sv_path = out_dir.join("rtl_old.sv");
    let patched_sv_path = out_dir.join("rtl_patched_old.sv");
    std::fs::write(&old_sv_path, old_sv.as_bytes()).expect("write rtl_old.sv");
    std::fs::write(&patched_sv_path, patched_sv.as_bytes()).expect("write rtl_patched_old.sv");
    println!("  RTL (old) written to: {}", old_sv_path.display());
    println!(
        "  RTL (patched(old)) written to: {}",
        patched_sv_path.display()
    );
    let rtl_diff_chars = myers_insdel_distance_bytes(old_sv.as_bytes(), patched_sv.as_bytes());
    println!(
        "  RTL text diff char count (old → patched(old)): {}",
        rtl_diff_chars
    );
    (text_diff_chars, rtl_diff_chars)
}

fn type_zero_value_text(ty: &Type) -> String {
    match ty {
        Type::Bits(w) => format!("bits[{}]:0", w),
        Type::Tuple(elems) => {
            let parts: Vec<String> = elems.iter().map(|t| type_zero_value_text(t)).collect();
            format!("({})", parts.join(", "))
        }
        Type::Array(arr) => {
            let part = type_zero_value_text(&arr.element_type);
            let parts = std::iter::repeat(part)
                .take(arr.element_count)
                .collect::<Vec<_>>();
            format!("[{}]", parts.join(", "))
        }
        Type::Token => "()".to_string(),
    }
}

fn ones_hex_for_width(width: usize) -> String {
    if width == 0 {
        return "0x0".to_string();
    }
    let full = width / 4;
    let rem = width % 4;
    let mut s = String::from("0x");
    if rem > 0 {
        let mask = (1u8 << rem) - 1;
        s.push_str(&format!("{:x}", mask));
    }
    if full > 0 {
        s.push_str(&"f".repeat(full));
    }
    s
}

fn type_ones_value_text(ty: &Type) -> String {
    match ty {
        Type::Bits(w) => format!("bits[{}]:{}", w, ones_hex_for_width(*w)),
        Type::Tuple(elems) => {
            let parts: Vec<String> = elems.iter().map(|t| type_ones_value_text(t)).collect();
            format!("({})", parts.join(", "))
        }
        Type::Array(arr) => {
            let part = type_ones_value_text(&arr.element_type);
            let parts = std::iter::repeat(part)
                .take(arr.element_count)
                .collect::<Vec<_>>();
            format!("[{}]", parts.join(", "))
        }
        Type::Token => "()".to_string(),
    }
}

fn rnd_hex_for_width(rng: &mut rand::rngs::StdRng, width: usize) -> String {
    if width == 0 {
        return "0x0".to_string();
    }
    let full = width / 4;
    let rem = width % 4;
    let mut s = String::from("0x");
    if rem > 0 {
        let max = (1u8 << rem) - 1;
        let val: u8 = rng.gen_range(0..=max);
        s.push_str(&format!("{:x}", val));
    }
    for _ in 0..full {
        let v: u8 = rng.gen_range(0..=15);
        s.push_str(&format!("{:x}", v));
    }
    s
}

fn type_random_value_text(ty: &Type, rng: &mut rand::rngs::StdRng) -> String {
    match ty {
        Type::Bits(w) => format!("bits[{}]:{}", w, rnd_hex_for_width(rng, *w)),
        Type::Tuple(elems) => {
            let parts: Vec<String> = elems
                .iter()
                .map(|t| type_random_value_text(t, rng))
                .collect();
            format!("({})", parts.join(", "))
        }
        Type::Array(arr) => {
            let part = type_random_value_text(&arr.element_type, rng);
            let parts = (0..arr.element_count)
                .map(|_| part.clone())
                .collect::<Vec<_>>();
            format!("[{}]", parts.join(", "))
        }
        Type::Token => "()".to_string(),
    }
}

fn has_token_param(f: &xlsynth_g8r::xls_ir::ir::Fn) -> bool {
    f.params.iter().any(|p| matches!(p.ty, Type::Token))
}

fn build_args_text(
    f: &xlsynth_g8r::xls_ir::ir::Fn,
    mode: &str,
    mut rng: Option<&mut rand::rngs::StdRng>,
) -> String {
    let parts: Vec<String> = f
        .params
        .iter()
        .map(|p| match mode {
            "zeros" => type_zero_value_text(&p.ty),
            "ones" => type_ones_value_text(&p.ty),
            _ => {
                let mut_ref = rng.as_deref_mut().unwrap();
                type_random_value_text(&p.ty, mut_ref)
            }
        })
        .collect();
    format!("({})", parts.join(", "))
}

fn sanity_check_interpret(
    new_ir_text: &str,
    new_fn: &xlsynth_g8r::xls_ir::ir::Fn,
    patched_ir_path: &std::path::Path,
    random_samples: usize,
    seed: u64,
) -> Result<(), String> {
    if has_token_param(new_fn) {
        return Err("token parameters not supported in interpreter sanity check".to_string());
    }
    let patched_pkg = xlsynth::IrPackage::parse_ir_from_path(patched_ir_path)
        .map_err(|e| format!("parse patched IR failed: {}", e))?;
    let new_pkg = xlsynth::IrPackage::parse_ir(new_ir_text, None)
        .map_err(|e| format!("parse new IR failed: {}", e))?;
    let top_name = &new_fn.name;
    let patched_f = patched_pkg
        .get_function(top_name)
        .map_err(|e| format!("get patched top failed: {}", e))?;
    let new_f = new_pkg
        .get_function(top_name)
        .map_err(|e| format!("get new top failed: {}", e))?;

    let zeros_text = build_args_text(new_fn, "zeros", None);
    let ones_text = build_args_text(new_fn, "ones", None);
    let zeros_tuple = xlsynth::IrValue::parse_typed(&zeros_text)
        .map_err(|e| format!("parse zeros args failed: {}", e))?;
    let ones_tuple = xlsynth::IrValue::parse_typed(&ones_text)
        .map_err(|e| format!("parse ones args failed: {}", e))?;
    let zeros_args = zeros_tuple
        .get_elements()
        .map_err(|e| format!("decompose zeros tuple failed: {}", e))?;
    let ones_args = ones_tuple
        .get_elements()
        .map_err(|e| format!("decompose ones tuple failed: {}", e))?;
    let mut skipped_due_to_asserts: usize = 0;
    // zeros
    let zv_p_res = patched_f.interpret(&zeros_args);
    let zv_n_res = new_f.interpret(&zeros_args);
    let zeros_assert = zv_p_res.is_err() || zv_n_res.is_err();
    if zeros_assert {
        skipped_due_to_asserts += 1;
    } else {
        let zv_p = zv_p_res.unwrap();
        let zv_n = zv_n_res.unwrap();
        if zv_p != zv_n {
            return Err(format!("mismatch on zeros: patched={} new={}", zv_p, zv_n));
        }
    }
    // ones
    let ov_p_res = patched_f.interpret(&ones_args);
    let ov_n_res = new_f.interpret(&ones_args);
    let ones_assert = ov_p_res.is_err() || ov_n_res.is_err();
    if ones_assert {
        skipped_due_to_asserts += 1;
    } else {
        let ov_p = ov_p_res.unwrap();
        let ov_n = ov_n_res.unwrap();
        if ov_p != ov_n {
            return Err(format!("mismatch on ones: patched={} new={}", ov_p, ov_n));
        }
    }

    if random_samples > 0 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut valid_done: usize = 0;
        let mut attempts: usize = 0;
        let max_attempts: usize = random_samples * 10 + 10;
        while valid_done < random_samples && attempts < max_attempts {
            attempts += 1;
            let arg_text = build_args_text(new_fn, "random", Some(&mut rng));
            let arg_tuple = xlsynth::IrValue::parse_typed(&arg_text)
                .map_err(|e| format!("parse random args failed: {}", e))?;
            let args = arg_tuple
                .get_elements()
                .map_err(|e| format!("decompose random tuple failed: {}", e))?;
            let pv_res = patched_f.interpret(&args);
            let nv_res = new_f.interpret(&args);
            if pv_res.is_err() || nv_res.is_err() {
                skipped_due_to_asserts += 1;
                continue;
            }
            let pv = pv_res.unwrap();
            let nv = nv_res.unwrap();
            if pv != nv {
                return Err(format!(
                    "mismatch on random input {}: patched={} new={}",
                    arg_text, pv, nv
                ));
            }
            valid_done += 1;
        }
        println!("  Sanity check: random valid samples: {}/{} (attempts: {}, skipped due to assertions: {})", valid_done, random_samples, attempts, skipped_due_to_asserts);
    }
    println!("  Sanity check: zeros/ones compared successfully");
    Ok(())
}

fn try_interpret_cex(
    new_ir_text: &str,
    new_fn: &xlsynth_g8r::xls_ir::ir::Fn,
    patched_ir_path: &std::path::Path,
    arg_text: &str,
) -> Result<(), String> {
    if has_token_param(new_fn) {
        return Err("token parameters present".to_string());
    }
    let patched_pkg = xlsynth::IrPackage::parse_ir_from_path(patched_ir_path)
        .map_err(|e| format!("parse patched IR failed: {}", e))?;
    let new_pkg = xlsynth::IrPackage::parse_ir(new_ir_text, None)
        .map_err(|e| format!("parse new IR failed: {}", e))?;
    let top_name = &new_fn.name;
    let patched_f = patched_pkg
        .get_function(top_name)
        .map_err(|e| format!("get patched top failed: {}", e))?;
    let new_f = new_pkg
        .get_function(top_name)
        .map_err(|e| format!("get new top failed: {}", e))?;
    // Extract the top-level tuple text after "input:" using balanced paren/bracket
    // parsing.
    let tuple_text = extract_tuple_text(arg_text)
        .ok_or_else(|| "could not extract tuple from counterexample text".to_string())?;
    let arg_tuple = xlsynth::IrValue::parse_typed(&tuple_text)
        .map_err(|e| format!("parse cex arg tuple failed: {}", e))?;
    // Prefer type-based mapping of the parsed value to function parameters.
    let f_type = new_f
        .get_type()
        .map_err(|e| format!("get new fn type failed: {}", e))?;
    let param_count = f_type.param_count();
    let expected_args_type: xlsynth::IrType = if param_count == 0 {
        new_pkg.get_tuple_type(&[])
    } else if param_count == 1 {
        f_type
            .param_type(0)
            .map_err(|e| format!("get param type failed: {}", e))?
    } else {
        let expected_param_types: Vec<xlsynth::IrType> = (0..param_count)
            .map(|i| {
                f_type
                    .param_type(i)
                    .map_err(|e| format!("get param type failed: {}", e))
            })
            .collect::<Result<_, _>>()?;
        new_pkg.get_tuple_type(&expected_param_types)
    };
    let candidate_type = new_pkg
        .get_type_for_value(&arg_tuple)
        .map_err(|e| format!("get type for cex value failed: {}", e))?;
    let args: Vec<IrValue> = if param_count == 0 {
        Vec::new()
    } else if param_count == 1 {
        // Single-parameter function: allow the whole value to be the argument.
        vec![arg_tuple]
    } else if new_pkg
        .types_eq(&candidate_type, &expected_args_type)
        .unwrap_or(false)
    {
        // Top value is a tuple exactly matching the param tuple type.
        arg_tuple
            .get_elements()
            .map_err(|e| format!("decompose cex tuple failed: {}", e))?
    } else if let Ok(top_elems) = arg_tuple.get_elements() {
        // If the top-level is a singleton that itself matches the arg tuple type,
        // unwrap it.
        if top_elems.len() == 1 {
            let inner = &top_elems[0];
            if let Ok(inner_ty) = new_pkg.get_type_for_value(inner) {
                if new_pkg
                    .types_eq(&inner_ty, &expected_args_type)
                    .unwrap_or(false)
                {
                    inner
                        .get_elements()
                        .map_err(|e| format!("decompose inner tuple failed: {}", e))?
                } else {
                    // Fallback: try reshaping from flat sequence.
                    reshape_args_to_params(&top_elems, &new_fn.params)
                        .map_err(|e| format!("reshape cex args failed: {}", e))?
                }
            } else {
                reshape_args_to_params(&top_elems, &new_fn.params)
                    .map_err(|e| format!("reshape cex args failed: {}", e))?
            }
        } else if top_elems.len() == new_fn.params.len() {
            top_elems
        } else {
            reshape_args_to_params(&top_elems, &new_fn.params)
                .map_err(|e| format!("reshape cex args failed: {}", e))?
        }
    } else {
        // Last resort: treat the parsed tuple elements as a flat list and try to
        // assemble by type.
        let flat = arg_tuple
            .get_elements()
            .map_err(|e| format!("decompose cex tuple failed: {}", e))?;
        reshape_args_to_params(&flat, &new_fn.params)
            .map_err(|e| format!("reshape cex args failed: {}", e))?
    };
    let pv = patched_f
        .interpret(&args)
        .map_err(|e| format!("patched interpret failed: {}", e))?;
    let nv = new_f
        .interpret(&args)
        .map_err(|e| format!("new interpret failed: {}", e))?;
    if pv == nv {
        println!("    interpreter replay: outputs equal: {}", pv);
    } else {
        println!(
            "    interpreter replay: mismatch: patched(old)={} new={}",
            pv, nv
        );
    }
    Ok(())
}

fn extract_tuple_text(s: &str) -> Option<String> {
    let bytes: Vec<char> = s.chars().collect();
    // Find first '(' character
    let mut i = 0usize;
    while i < bytes.len() && bytes[i] != '(' {
        i += 1;
    }
    if i == bytes.len() {
        return None;
    }
    let mut depth_paren: i32 = 0;
    let mut depth_bracket: i32 = 0;
    let start = i;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            '(' => depth_paren += 1,
            ')' => {
                depth_paren -= 1;
                if depth_paren == 0 && depth_bracket == 0 {
                    let end = i + 1;
                    return Some(bytes[start..end].iter().collect());
                }
            }
            '[' => depth_bracket += 1,
            ']' => depth_bracket -= 1,
            _ => {}
        }
        i += 1;
    }
    None
}

fn consume_value_for_type(
    expected: &Type,
    flat: &[IrValue],
    idx: &mut usize,
) -> Result<IrValue, String> {
    match expected {
        Type::Bits(_w) => {
            if *idx >= flat.len() {
                return Err("ran out of values while matching bits param".to_string());
            }
            let v = flat[*idx].clone();
            *idx += 1;
            Ok(v)
        }
        Type::Tuple(elems) => {
            // Build tuple by consuming elements for each field type.
            let mut fields: Vec<IrValue> = Vec::with_capacity(elems.len());
            for t in elems.iter() {
                let fv = consume_value_for_type(t, flat, idx)?;
                fields.push(fv);
            }
            Ok(IrValue::make_tuple(&fields))
        }
        Type::Array(arr) => {
            let mut elems: Vec<IrValue> = Vec::with_capacity(arr.element_count);
            for _ in 0..arr.element_count {
                let ev = consume_value_for_type(&arr.element_type, flat, idx)?;
                elems.push(ev);
            }
            IrValue::make_array(&elems).map_err(|e| format!("make_array failed: {}", e))
        }
        Type::Token => Err("token parameter not supported".to_string()),
    }
}

fn reshape_args_to_params(
    flat: &[IrValue],
    params: &[xlsynth_g8r::xls_ir::ir::Param],
) -> Result<Vec<IrValue>, String> {
    let mut idx: usize = 0;
    let mut out: Vec<IrValue> = Vec::with_capacity(params.len());
    for p in params.iter() {
        let v = consume_value_for_type(&p.ty, flat, &mut idx)?;
        out.push(v);
    }
    Ok(out)
}
