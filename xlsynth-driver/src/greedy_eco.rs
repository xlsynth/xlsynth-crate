// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use std::path::Path;
use xlsynth_pir::greedy_matching_ged::GreedyMatchSelector;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::matching_ged::{apply_fn_edits, compute_fn_edit};

/// Implements the "greedy-eco" subcommand:
/// - Reads old/new IR package files
/// - Computes greedy graph-edit edits to transform old→new
/// - Applies edits to the old function
/// - Emits the patched package IR (stdout or --patched_out)
/// - Optionally writes the debug string of IrEdits to --edits_debug_out
pub fn handle_greedy_eco(matches: &ArgMatches) {
    let old_ir_path = Path::new(matches.get_one::<String>("old_ir_file").unwrap());
    let new_ir_path = Path::new(matches.get_one::<String>("new_ir_file").unwrap());

    let old_ir_text = std::fs::read_to_string(old_ir_path).expect("read old IR should succeed");
    let new_ir_text = std::fs::read_to_string(new_ir_path).expect("read new IR should succeed");

    let mut old_pkg = Parser::new(&old_ir_text)
        .parse_and_validate_package()
        .expect("parse old IR should succeed");
    let new_pkg = Parser::new(&new_ir_text)
        .parse_and_validate_package()
        .expect("parse new IR should succeed");

    let old_top_flag = matches.get_one::<String>("old_ir_top").map(|s| s.as_str());
    let new_top_flag = matches.get_one::<String>("new_ir_top").map(|s| s.as_str());

    let old_fn = match old_top_flag {
        Some(name) => old_pkg
            .get_fn(name)
            .unwrap_or_else(|| panic!("old package missing function '{}'", name)),
        None => old_pkg
            .get_top_fn()
            .expect("old package missing top function"),
    };
    let new_fn = match new_top_flag {
        Some(name) => new_pkg
            .get_fn(name)
            .unwrap_or_else(|| panic!("new package missing function '{}'", name)),
        None => new_pkg
            .get_top_fn()
            .expect("new package missing top function"),
    };

    // Compute edits using the greedy selector.
    let mut selector = GreedyMatchSelector::new(old_fn, new_fn);
    let edits = compute_fn_edit(old_fn, new_fn, &mut selector)
        .expect("compute_function_edit (greedy) failed");

    // Optionally write the debug string of edits.
    if let Some(path) = matches.get_one::<String>("edits_debug_out") {
        let dbg_str = format!("{:#?}\n", edits.edits);
        std::fs::write(path, dbg_str).expect("write edits_debug_out should succeed");
    }

    // Apply to old function and splice back into the old package.
    let patched_fn = apply_fn_edits(old_fn, &edits).expect("apply_fn_edits failed");

    if let Some(name) = old_top_flag {
        // Replace the named function in the old package.
        let slot = old_pkg
            .get_fn_mut(name)
            .unwrap_or_else(|| panic!("old package missing function '{}'", name));
        *slot = patched_fn;
    } else {
        // Replace top function.
        let slot = old_pkg
            .get_top_fn_mut()
            .expect("old package missing top function (mut)");
        *slot = patched_fn;
    }

    // Emit patched package IR.
    if let Some(path) = matches.get_one::<String>("patched_out") {
        std::fs::write(path, format!("{}", old_pkg)).expect("write patched_out should succeed");
    } else {
        print!("{}", old_pkg);
    }
}
