// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use xlsynth_pir::ir;
use xlsynth_pir::ir_verify;
use xlsynth_pir::localized_eco2;

use crate::ir_equiv::{IrEquivRequest, IrModule, dispatch_ir_equiv};
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use rand::SeedableRng;
use std::path::Path;
use xlsynth_g8r::check_equivalence;
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::Type;
use xlsynth_pir::ir::{self as ir_mod, BlockMetadata, MemberType, PackageMember};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_parser::{self, emit_fn_as_block};
use xlsynth_pir::random_inputs::{
    BitValuePattern, generate_biased_arguments_with_rng, generate_pattern_arguments,
};
use xlsynth_prover::prover::SolverChoice;

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
    let solver: SolverChoice = matches
        .get_one::<String>("solver")
        .unwrap()
        .parse()
        .unwrap();
    let tool_path = config
        .as_ref()
        .and_then(|c| c.tool_path.as_deref())
        .map(Path::new);

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

    let old_pkg = match ir_parser::parse_path_to_package(old_path) {
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
    let new_pkg = match ir_parser::parse_path_to_package(new_path) {
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
            solver,
            tool_path,
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
        None => match old_pkg.get_top_fn() {
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
        None => match new_pkg.get_top_fn() {
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

    // Build patched function via structural rebase; compute simple node-add
    // count.
    let patched_for_count = localized_eco2::compute_localized_eco(old_fn, new_fn);
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
        // We'll serialize after computing text edit distance below; write
        // later. For now, just return the path.
        path
    } else {
        let path = out_dir.join("eco_report.json");
        // We'll serialize after computing text edit distance below.
        path
    };

    // Patched IR path: emit the NEW IR with IDs remapped to preserve old IDs
    // where subgraphs are structurally equal; allocate fresh IDs for new
    // nodes.
    let patched_ir_path = out_dir.join("patched_old.ir");
    // Re-emit both old and patched packages to ensure a comparable formatting
    // baseline.
    let old_ir_text_emitted = old_pkg.to_string();
    // Build patched package by constructing a rebase-based patched function
    // that preserves existing node IDs where structurally reusable, allocating
    // new ones only for synthesized nodes.
    let mut patched_pkg = old_pkg.clone();
    if let Some(target_fn) = patched_pkg.get_fn_mut(&old_fn.name) {
        let applied = localized_eco2::compute_localized_eco(old_fn, new_fn);
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
    if let Err(e) = ir_verify::verify_fn_unique_node_ids(new_fn) {
        println!("  WARNING: verification failed (duplicate IDs): {}", e);
    }
    if let Err(e) = ir_verify::verify_fn_operand_indices_in_bounds(new_fn) {
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
    // Prove: patched_old.ir ≡ new.ir using the selected solver.
    let patched_ir_text = std::fs::read_to_string(&patched_ir_path).unwrap();
    let new_ir_text = std::fs::read_to_string(new_path).unwrap();
    // Use the new function's name as the top on both sides (patched equals new
    // package text).
    let lhs_top = Some(new_fn.name.as_str());
    let rhs_top = Some(new_fn.name.as_str());
    println!(
        "  Starting equivalence proof using solver '{}': patched='{}' top='{}' vs new='{}' top='{}'",
        solver,
        patched_ir_path.display(),
        lhs_top.unwrap_or(""),
        new_path.display(),
        rhs_top.unwrap_or("")
    );
    let request = IrEquivRequest::new(
        IrModule::new(&patched_ir_text)
            .with_path(Some(patched_ir_path.as_path()))
            .with_top(lhs_top),
        IrModule::new(&new_ir_text)
            .with_path(Some(new_path))
            .with_top(rhs_top),
    )
    .with_solver(Some(solver))
    .with_tool_path(tool_path);

    let outcome = dispatch_ir_equiv(&request, "ir-localized-eco");
    let dur = std::time::Duration::from_micros(outcome.time_micros as u64);
    if outcome.success {
        println!("  Equivalence: proved (patched(old) ≡ new) in {:?}", dur);
    } else {
        println!("  Equivalence: FAILED (patched(old) vs new) in {:?}", dur);
        if let Some(err) = outcome.error_str.as_ref() {
            println!("    error: {}", err);
            // Attempt to replay the counterexample via interpreter.
            if let Some(input_idx) = err.find("input:") {
                let arg_text = err[input_idx + "input:".len()..].trim();
                match try_interpret_cex(&new_ir_text, new_fn, &patched_ir_path, arg_text) {
                    Ok(()) => {}
                    Err(e) => println!("    interpreter replay: skipped ({})", e),
                }
            }
        }
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
) -> Option<(&'a ir_mod::Fn, &'a BlockMetadata)> {
    if let Some(name) = name_opt {
        for m in pkg.members.iter() {
            if let PackageMember::Block { func, metadata } = m {
                if func.name == name {
                    return Some((func, metadata));
                }
            }
        }
        return None;
    }
    if let Some((top_name, MemberType::Block)) = &pkg.top {
        for m in pkg.members.iter() {
            if let PackageMember::Block { func, metadata } = m {
                if &func.name == top_name {
                    return Some((func, metadata));
                }
            }
        }
    }
    for m in pkg.members.iter() {
        if let PackageMember::Block { func, metadata } = m {
            return Some((func, metadata));
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
    old_ports: &BlockMetadata,
    new_fn: &ir_mod::Fn,
    new_ports: &BlockMetadata,
    solver: SolverChoice,
    tool_path: Option<&Path>,
) {
    // Summaries (rebase-based): will compute added_count after building
    // applied.
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
    let applied = localized_eco2::compute_localized_eco(old_fn, new_fn);
    let added_count: usize = applied.nodes.len().saturating_sub(old_fn.nodes.len());

    // Validate output arity compatibility with old block port info.
    let applied_out_types = get_output_types_for_emission(&applied, old_ports.output_names.len());
    if old_ports.output_names.len() != applied_out_types.len() {
        let msg = format!(
            "output arity mismatch: old block had output ports {:?}; function outputs are {:?} ({}).",
            old_ports.output_names,
            applied_out_types
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>(),
            applied_out_types.len()
        );
        report_cli_error_and_exit(&msg, Some("ir-localized-eco"), vec![]);
    }
    println!("  Emitting patched block text...");
    let patched_block_text = emit_fn_as_block(&applied, None, Some(old_ports), false);
    let patched_ir_path = out_dir.join("patched_old.block.ir");
    std::fs::write(&patched_ir_path, patched_block_text.as_bytes()).unwrap();
    println!("  Patched IR written to: {}", patched_ir_path.display());

    // Copy old/new for convenience: write ONLY the selected blocks.
    let old_copy_path = out_dir.join("old.ir");
    let new_copy_path = out_dir.join("new.ir");
    let old_block_text = emit_fn_as_block(old_fn, None, Some(old_ports), false);
    let new_block_text = emit_fn_as_block(new_fn, None, Some(new_ports), false);
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
    // using the selected solver.
    let lhs_pkg = format!("package lhs\n\ntop {}", applied.to_string());
    let rhs_pkg = format!("package rhs\n\ntop {}", new_fn.to_string());
    let top_name = Some(new_fn.name.as_str());
    match check_equivalence::check_equivalence_with_top_and_solver(
        &lhs_pkg, &rhs_pkg, top_name, solver, tool_path,
    ) {
        Ok(()) => println!("  Equivalence: proved (patched(old) ≡ new)"),
        Err(e) => println!("  Equivalence: FAILED: {}", e),
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

fn has_token_param(f: &ir::Fn) -> bool {
    f.params.iter().any(|p| matches!(p.ty, Type::Token))
}

fn build_zero_args_value(f: &ir::Fn) -> IrValue {
    IrValue::make_tuple(&generate_pattern_arguments(f, BitValuePattern::Zero))
}

fn build_ones_args_value(f: &ir::Fn) -> IrValue {
    IrValue::make_tuple(&generate_pattern_arguments(f, BitValuePattern::AllOnes))
}

fn build_random_args_value(f: &ir::Fn, rng: &mut rand::rngs::StdRng) -> IrValue {
    IrValue::make_tuple(&generate_biased_arguments_with_rng(rng, f))
}

/// Checks replay arguments before invoking the native evaluator on verified IR.
fn eval_pir_for_replay(
    pkg: &ir::Package,
    f: &ir::Fn,
    args: &[IrValue],
) -> Result<FnEvalResult, String> {
    if args.len() != f.params.len() {
        return Err(format!(
            "function '{}' expects {} arguments, got {}",
            f.name,
            f.params.len(),
            args.len()
        ));
    }
    for (arg, param) in args.iter().zip(&f.params) {
        if arg.type_() != param.ty {
            return Err(format!(
                "argument '{}' expects {}, got {}",
                param.name,
                param.ty,
                arg.type_()
            ));
        }
    }
    Ok(eval_fn_in_package(pkg, f, args))
}

/// Compares native PIR results while excluding samples that violate runtime
/// contracts.
fn sanity_check_interpret(
    new_ir_text: &str,
    new_fn: &ir::Fn,
    patched_ir_path: &std::path::Path,
    random_samples: usize,
    seed: u64,
) -> Result<(), String> {
    if has_token_param(new_fn) {
        return Err("token parameters not supported in interpreter sanity check".to_string());
    }
    let patched_text = std::fs::read_to_string(patched_ir_path)
        .map_err(|e| format!("read patched IR failed: {}", e))?;
    let patched_pkg = ir_parser::Parser::new(&patched_text)
        .parse_and_verify_package()
        .map_err(|e| format!("parse patched IR failed: {}", e))?;
    let new_pkg = ir_parser::Parser::new(new_ir_text)
        .parse_and_verify_package()
        .map_err(|e| format!("parse new IR failed: {}", e))?;
    let top_name = &new_fn.name;
    let patched_f = patched_pkg
        .get_fn(top_name)
        .ok_or_else(|| format!("get patched top failed: function '{top_name}' not found"))?;
    let new_f = new_pkg
        .get_fn(top_name)
        .ok_or_else(|| format!("get new top failed: function '{top_name}' not found"))?;

    let zeros_tuple = build_zero_args_value(new_fn);
    let ones_tuple = build_ones_args_value(new_fn);
    let zeros_args = zeros_tuple
        .get_elements()
        .map_err(|e| format!("decompose zeros tuple failed: {}", e))?;
    let ones_args = ones_tuple
        .get_elements()
        .map_err(|e| format!("decompose ones tuple failed: {}", e))?;
    let mut skipped_due_to_asserts: usize = 0;
    // zeros
    match (
        eval_pir_for_replay(&patched_pkg, patched_f, &zeros_args)?,
        eval_pir_for_replay(&new_pkg, new_f, &zeros_args)?,
    ) {
        (FnEvalResult::Success(patched), FnEvalResult::Success(new)) => {
            if patched.value != new.value {
                return Err(format!(
                    "mismatch on zeros: patched={} new={}",
                    patched.value, new.value
                ));
            }
        }
        _ => {
            // Samples violating either function's runtime contracts are not
            // comparable.
            skipped_due_to_asserts += 1;
        }
    }
    // ones
    match (
        eval_pir_for_replay(&patched_pkg, patched_f, &ones_args)?,
        eval_pir_for_replay(&new_pkg, new_f, &ones_args)?,
    ) {
        (FnEvalResult::Success(patched), FnEvalResult::Success(new)) => {
            if patched.value != new.value {
                return Err(format!(
                    "mismatch on ones: patched={} new={}",
                    patched.value, new.value
                ));
            }
        }
        _ => {
            // Samples violating either function's runtime contracts are not
            // comparable.
            skipped_due_to_asserts += 1;
        }
    }

    if random_samples > 0 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut valid_done: usize = 0;
        let mut attempts: usize = 0;
        let max_attempts: usize = random_samples * 10 + 10;
        while valid_done < random_samples && attempts < max_attempts {
            attempts += 1;
            let arg_tuple = build_random_args_value(new_fn, &mut rng);
            let arg_text = arg_tuple.to_string();
            let args = arg_tuple
                .get_elements()
                .map_err(|e| format!("decompose random tuple failed: {}", e))?;
            match (
                eval_pir_for_replay(&patched_pkg, patched_f, &args)?,
                eval_pir_for_replay(&new_pkg, new_f, &args)?,
            ) {
                (FnEvalResult::Success(patched), FnEvalResult::Success(new)) => {
                    if patched.value != new.value {
                        return Err(format!(
                            "mismatch on random input {}: patched={} new={}",
                            arg_text, patched.value, new.value
                        ));
                    }
                }
                _ => {
                    // Assertion-failing samples are skipped without masking
                    // invalid argument types.
                    skipped_due_to_asserts += 1;
                    continue;
                }
            }
            valid_done += 1;
        }
        println!(
            "  Sanity check: random valid samples: {}/{} (attempts: {}, skipped due to assertions: {})",
            valid_done, random_samples, attempts, skipped_due_to_asserts
        );
    }
    println!("  Sanity check: zeros/ones compared successfully");
    Ok(())
}

/// Replays a typed counterexample against both verified PIR packages.
fn try_interpret_cex(
    new_ir_text: &str,
    new_fn: &ir::Fn,
    patched_ir_path: &std::path::Path,
    arg_text: &str,
) -> Result<(), String> {
    if has_token_param(new_fn) {
        return Err("token parameters present".to_string());
    }
    let patched_text = std::fs::read_to_string(patched_ir_path)
        .map_err(|e| format!("read patched IR failed: {}", e))?;
    let patched_pkg = ir_parser::Parser::new(&patched_text)
        .parse_and_verify_package()
        .map_err(|e| format!("parse patched IR failed: {}", e))?;
    let new_pkg = ir_parser::Parser::new(new_ir_text)
        .parse_and_verify_package()
        .map_err(|e| format!("parse new IR failed: {}", e))?;
    let top_name = &new_fn.name;
    let patched_f = patched_pkg
        .get_fn(top_name)
        .ok_or_else(|| format!("get patched top failed: function '{top_name}' not found"))?;
    let new_f = new_pkg
        .get_fn(top_name)
        .ok_or_else(|| format!("get new top failed: function '{top_name}' not found"))?;
    // Extract the top-level tuple text after "input:" using balanced
    // paren/bracket parsing.
    let tuple_text = extract_tuple_text(arg_text)
        .ok_or_else(|| "could not extract tuple from counterexample text".to_string())?;
    let arg_tuple = IrValue::parse_typed(&tuple_text)
        .map_err(|e| format!("parse cex arg tuple failed: {}", e))?;
    let args = counterexample_args(&arg_tuple, &new_fn.params)?;
    let pv = match eval_pir_for_replay(&patched_pkg, patched_f, &args)? {
        FnEvalResult::Success(success) => success.value,
        failure => return Err(format!("patched interpret failed: {failure:?}")),
    };
    let nv = match eval_pir_for_replay(&new_pkg, new_f, &args)? {
        FnEvalResult::Success(success) => success.value,
        failure => return Err(format!("new interpret failed: {failure:?}")),
    };
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

/// Maps a counterexample's tuple or flattened leaves to native function
/// arguments.
fn counterexample_args(value: &IrValue, params: &[ir::Param]) -> Result<Vec<IrValue>, String> {
    let expected = Type::Tuple(
        params
            .iter()
            .map(|param| Box::new(param.ty.clone()))
            .collect(),
    );
    if params.len() == 1 && value.type_() == params[0].ty {
        return Ok(vec![value.clone()]);
    }
    let IrValue::Tuple(elements) = value else {
        return Err(format!(
            "expected counterexample argument tuple, got {}",
            value.type_()
        ));
    };
    if value.type_() == expected {
        return Ok(elements.to_vec());
    }
    if elements.len() == 1 && elements[0].type_() == expected {
        return elements[0]
            .get_elements()
            .map_err(|error| error.to_string());
    }
    reshape_args_to_params(elements, params)
}

/// Reconstructs one native argument by consuming values according to its type.
fn consume_value_for_type(
    expected: &Type,
    flat: &[IrValue],
    idx: &mut usize,
) -> Result<IrValue, String> {
    if let Some(value) = flat.get(*idx) {
        if value.type_() == *expected {
            *idx += 1;
            return Ok(value.clone());
        }
    }
    match expected {
        Type::Bits(_w) => {
            if *idx >= flat.len() {
                return Err("ran out of values while matching bits param".to_string());
            }
            Err(format!("expected {expected}, got {}", flat[*idx].type_()))
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
            IrValue::make_array_typed((*arr.element_type).clone(), &elems)
                .map_err(|e| format!("make_array failed: {}", e))
        }
        Type::Token => Err("token parameter not supported".to_string()),
    }
}

/// Reconstructs native arguments and rejects unconsumed counterexample values.
fn reshape_args_to_params(flat: &[IrValue], params: &[ir::Param]) -> Result<Vec<IrValue>, String> {
    let mut idx: usize = 0;
    let mut out: Vec<IrValue> = Vec::with_capacity(params.len());
    for p in params.iter() {
        let v = consume_value_for_type(&p.ty, flat, &mut idx)?;
        out.push(v);
    }
    if idx != flat.len() {
        return Err(format!(
            "{} unconsumed counterexample values",
            flat.len() - idx
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use xlsynth_g8r::test_utils::interesting_ir_roundtrip_cases;
    use xlsynth_pir::IrBits;
    use xlsynth_pir::random_inputs::generate_argument_sets_from_seed;

    const REPLAY_CALLS_IR: &str = r#"package replay

fn inc(x: bits[129] id=1) -> bits[129] {
  one: bits[129] = literal(value=1, id=2)
  ret sum: bits[129] = add(x, one, id=3)
}

fn body(i: bits[2] id=4, carry: bits[129] id=5) -> bits[129] {
  ret next: bits[129] = invoke(carry, to_apply=inc, id=6)
}

top fn main(x: bits[129] id=7) -> bits[129] {
  initial: bits[129] = invoke(x, to_apply=inc, id=8)
  ret loop: bits[129] = counted_for(initial, trip_count=3, stride=1, body=body, id=9)
}
"#;

    const REPLAY_ASSERT_IR: &str = r#"package replay

fn checked(x: bits[1] id=1) -> bits[1] {
  t: token = after_all(id=2)
  checked: token = assert(t, x, message="x must be set", label="x_set", id=3)
  ret result: bits[1] = identity(x, id=4)
}

top fn main(x: bits[1] id=5) -> bits[1] {
  call: bits[1] = invoke(x, to_apply=checked, id=6)
  one: bits[1] = literal(value=1, id=7)
  ret result: bits[1] = identity(call, id=8)
}
"#;

    #[test]
    fn replay_evaluates_package_calls_loops_and_wide_values() {
        let pkg = ir_parser::Parser::new(REPLAY_CALLS_IR)
            .parse_and_verify_package()
            .unwrap();
        let f = pkg.get_top_fn().unwrap();
        let args = [
            IrValue::parse_typed("bits[129]:0x1_0000_0000_0000_0000_0000_0000_0000_0000").unwrap(),
        ];
        let FnEvalResult::Success(result) = eval_pir_for_replay(&pkg, f, &args).unwrap() else {
            panic!("valid replay must succeed");
        };
        assert_eq!(
            result.value,
            IrValue::parse_typed("bits[129]:0x1_0000_0000_0000_0000_0000_0000_0000_0004").unwrap()
        );
        let dir = tempdir().unwrap();
        let path = dir.path().join("patched.ir");
        std::fs::write(&path, REPLAY_CALLS_IR).unwrap();
        sanity_check_interpret(REPLAY_CALLS_IR, f, &path, 8, 42).unwrap();
        let cex = format!("input: {}", IrValue::make_tuple(&args));
        try_interpret_cex(REPLAY_CALLS_IR, f, &path, &cex).unwrap();

        std::fs::write(&path, REPLAY_CALLS_IR.replace("value=1", "value=2")).unwrap();
        assert_eq!(
            sanity_check_interpret(REPLAY_CALLS_IR, f, &path, 0, 42).unwrap_err(),
            "mismatch on zeros: patched=bits[129]:0x8 new=bits[129]:0x4"
        );
    }

    #[test]
    fn replay_skips_assertion_samples_but_rejects_invalid_arguments() {
        let pkg = ir_parser::Parser::new(REPLAY_ASSERT_IR)
            .parse_and_verify_package()
            .unwrap();
        let f = pkg.get_top_fn().unwrap();
        let failed = eval_pir_for_replay(&pkg, f, &[IrValue::make_ubits(1, 0).unwrap()]).unwrap();
        let FnEvalResult::Failure(failure) = failed else {
            panic!("callee assertion must be propagated");
        };
        assert_eq!(failure.assertion_failures.len(), 1);
        assert_eq!(failure.assertion_failures[0].label, "x_set");
        assert_eq!(
            eval_pir_for_replay(&pkg, f, &[]).unwrap_err(),
            "function 'main' expects 1 arguments, got 0"
        );
        assert_eq!(
            eval_pir_for_replay(&pkg, f, &[IrValue::make_ubits(2, 0).unwrap()]).unwrap_err(),
            "argument 'x' expects bits[1], got bits[2]"
        );

        let dir = tempdir().unwrap();
        let path = dir.path().join("patched.ir");
        let patched = REPLAY_ASSERT_IR.replace("identity(call, id=8)", "or(call, one, id=8)");
        std::fs::write(&path, &patched).unwrap();
        // Only assertion-failing inputs differ; they must be excluded from the
        // sanity check.
        sanity_check_interpret(REPLAY_ASSERT_IR, f, &path, 8, 42).unwrap();
        try_interpret_cex(REPLAY_ASSERT_IR, f, &path, "input: (bits[1]:1)").unwrap();
        let patched_pkg = ir_parser::Parser::new(&patched)
            .parse_and_verify_package()
            .unwrap();
        let patched_failure = eval_pir_for_replay(
            &patched_pkg,
            patched_pkg.get_top_fn().unwrap(),
            &[IrValue::make_ubits(1, 0).unwrap()],
        )
        .unwrap();
        assert_eq!(
            try_interpret_cex(REPLAY_ASSERT_IR, f, &path, "input: (bits[1]:0)").unwrap_err(),
            format!("patched interpret failed: {patched_failure:?}")
        );
    }

    fn params(types: &[Type]) -> Vec<ir::Param> {
        types
            .iter()
            .enumerate()
            .map(|(index, ty)| ir::Param {
                name: format!("p{index}"),
                ty: ty.clone(),
                id: ir::ParamId::new(index + 1),
            })
            .collect()
    }

    /// Flattens aggregates to typed leaves using only native value storage.
    fn leaves(value: &IrValue, out: &mut Vec<IrValue>) {
        match value {
            IrValue::Bits(_) | IrValue::Token => out.push(value.clone()),
            IrValue::Tuple(_) | IrValue::Array(_) => {
                for element in value.as_elements().unwrap() {
                    leaves(element, out);
                }
            }
        }
    }

    #[test]
    fn counterexamples_preserve_shared_scalar_and_aggregate_signatures() {
        for case in interesting_ir_roundtrip_cases() {
            let package = ir_parser::Parser::new(case.ir_text)
                .parse_and_validate_package()
                .unwrap();
            let f = package.get_top_fn().unwrap();
            for args in generate_argument_sets_from_seed(&f, 42, 8) {
                let tuple = IrValue::make_tuple(&args);
                let parsed = IrValue::parse_typed(&tuple.to_string()).unwrap();
                assert_eq!(
                    counterexample_args(&parsed, &f.params).unwrap(),
                    args,
                    "{}",
                    case.name
                );
                let wrapped = IrValue::make_tuple(&[tuple.clone()]);
                assert_eq!(
                    counterexample_args(&wrapped, &f.params).unwrap(),
                    args,
                    "{}",
                    case.name
                );
                let mut flat = Vec::new();
                leaves(&tuple, &mut flat);
                assert_eq!(
                    reshape_args_to_params(&flat, &f.params).unwrap(),
                    args,
                    "{}",
                    case.name
                );
            }
        }
    }

    #[test]
    fn reshapes_wide_nested_values_and_typed_empty_arrays() {
        let wide = IrValue::from_bits(&IrBits::all_ones(129));
        let element = IrValue::make_ubits(65, 7).unwrap();
        let array = IrValue::make_array(&[element.clone(), element]).unwrap();
        let nested = IrValue::make_tuple(&[wide.clone(), array.clone()]);
        let empty = IrValue::make_array_typed(Type::Bits(257), &[]).unwrap();
        let expected = vec![nested.clone(), empty];
        let params = params(&expected.iter().map(IrValue::type_).collect::<Vec<_>>());
        let mut flat = Vec::new();
        leaves(&nested, &mut flat);
        assert_eq!(reshape_args_to_params(&flat, &params).unwrap(), expected);
        assert_eq!(
            counterexample_args(&nested, &params[..1]).unwrap(),
            vec![nested]
        );
    }

    #[test]
    fn rejects_wrong_width_missing_extra_and_non_tuple_arguments() {
        let params = params(&[Type::Bits(129), Type::Bits(1)]);
        let wide = IrValue::make_ubits(129, 0).unwrap();
        let bit = IrValue::make_ubits(1, 0).unwrap();
        assert_eq!(
            reshape_args_to_params(&[bit.clone(), bit.clone()], &params).unwrap_err(),
            "expected bits[129], got bits[1]"
        );
        assert_eq!(
            reshape_args_to_params(&[wide.clone()], &params).unwrap_err(),
            "ran out of values while matching bits param"
        );
        assert_eq!(
            reshape_args_to_params(&[wide.clone(), bit.clone(), bit.clone()], &params).unwrap_err(),
            "1 unconsumed counterexample values"
        );
        assert_eq!(
            counterexample_args(&wide, &params).unwrap_err(),
            "expected counterexample argument tuple, got bits[129]"
        );
        assert_eq!(
            counterexample_args(&IrValue::make_tuple(&[]), &[]).unwrap(),
            Vec::<IrValue>::new()
        );
        assert_eq!(
            counterexample_args(&IrValue::make_tuple(&[bit]), &[]).unwrap_err(),
            "1 unconsumed counterexample values"
        );
    }
}
