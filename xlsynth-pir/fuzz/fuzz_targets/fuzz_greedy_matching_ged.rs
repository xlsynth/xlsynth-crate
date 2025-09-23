// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use log::{debug, info};
use xlsynth::IrPackage;
<<<<<<< HEAD
use xlsynth_pir::dce::{get_dead_nodes, remove_dead_nodes};
use xlsynth_pir::greedy_matching_ged::GreedyMatchSelector;
use xlsynth_pir::ir_fuzz::{FuzzSampleSameTypedPair, generate_ir_fn};
=======
use xlsynth_pir::greedy_matching_ged::GreedyMatchSelector;
use xlsynth_pir::ir_fuzz::{FuzzSampleSameTypedPair, generate_ir_fn};
use xlsynth_pir::ir_utils::{get_dead_nodes, remove_dead_nodes};
>>>>>>> 5cee27d (Add a greedy ECO computation algorithm.)
use xlsynth_pir::matching_ged::{
    IrMatchSet, MatchAction, NodeSide, apply_function_edits, compute_function_edit,
    compute_function_match, convert_match_set_to_edit_set, format_ir_edits, format_match_actions,
};
use xlsynth_pir::{ir_parser, node_hashing::functions_structurally_equivalent};
use xlsynth_pir_fuzz::equiv::{
    compute_forward_equivalences, compute_reverse_equivalences_to_return,
};

// Returns the pairs of nodes which are CSE-equivalent between old and new
// graphs. Only unique pairs are returned. If a node is equivalent to multiple
// nodes in the other graph then it is not included.
fn unique_forward_equivalent_pairs(
    old_fn: &xlsynth_pir::ir::Fn,
    new_fn: &xlsynth_pir::ir::Fn,
) -> Vec<(usize, usize)> {
    let eq = compute_forward_equivalences(old_fn, new_fn);
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (oi, news) in eq.lhs_to_rhs.iter() {
        if news.len() != 1 {
            continue;
        }
        let ni: usize = news[0];
        if let Some(olds) = eq.rhs_to_lhs.get(&ni) {
            if olds.len() == 1 && olds[0] == *oi {
                pairs.push((*oi, ni));
            }
        }
    }
    pairs
}

// Returns the pairs of nodes which are reverse structurallyequivalent between
// old and new graphs. Only unique pairs are returned. If a node is equivalent
// to multiple nodes in the other graph then it is not included.
fn unique_reverse_equivalent_pairs(
    old_fn: &xlsynth_pir::ir::Fn,
    new_fn: &xlsynth_pir::ir::Fn,
) -> Vec<(usize, usize)> {
    let eq = compute_reverse_equivalences_to_return(old_fn, new_fn);
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (oi, news) in eq.lhs_to_rhs.iter() {
        if news.len() != 1 {
            continue;
        }
        let ni: usize = news[0];
        if let Some(olds) = eq.rhs_to_lhs.get(&ni) {
            if olds.len() == 1 && olds[0] == *oi {
                pairs.push((*oi, ni));
            }
        }
    }
    pairs
}

fn assert_equivalent_nodes_are_matched(
    old_fn: &xlsynth_pir::ir::Fn,
    new_fn: &xlsynth_pir::ir::Fn,
    matches: &IrMatchSet,
) {
    use std::collections::HashSet;
    let mut matched_pairs: HashSet<(usize, usize)> = HashSet::new();
    for m in matches.matches.iter() {
        if let MatchAction::MatchNodes {
            old_index,
            new_index,
            ..
        } = m
        {
            matched_pairs.insert((usize::from(*old_index), usize::from(*new_index)));
        }
    }
    // Precompute dead node sets (by index) for both functions.
    let dead_old: std::collections::HashSet<usize> = get_dead_nodes(old_fn)
        .into_iter()
        .map(|nr| nr.index)
        .collect();
    let dead_new: std::collections::HashSet<usize> = get_dead_nodes(new_fn)
        .into_iter()
        .map(|nr| nr.index)
        .collect();

    // Compute uniquely equivalent pairs (forward) and ensure matches exist,
    // skipping pairs where either node is dead.
    let fwd_pairs = unique_forward_equivalent_pairs(old_fn, new_fn);
    for (oi, ni) in fwd_pairs.into_iter() {
        if dead_old.contains(&oi) || dead_new.contains(&ni) {
            continue;
        }
        if !matched_pairs.contains(&(oi, ni)) {
            panic!(
                "missing MatchNodes for uniquely forward-equivalent pair: {} <-> {}",
                xlsynth_pir::ir::node_textual_id(old_fn, xlsynth_pir::ir::NodeRef { index: oi }),
                xlsynth_pir::ir::node_textual_id(new_fn, xlsynth_pir::ir::NodeRef { index: ni }),
            );
        }
    }
    // Compute uniquely equivalent pairs (reverse) and ensure matches exist.
    let rev_pairs = unique_reverse_equivalent_pairs(old_fn, new_fn);
    for (oi, ni) in rev_pairs.into_iter() {
        if dead_old.contains(&oi) || dead_new.contains(&ni) {
            continue;
        }
        if !matched_pairs.contains(&(oi, ni)) {
            panic!(
                "missing MatchNodes for uniquely reverse-equivalent pair: {} <-> {}",
                xlsynth_pir::ir::node_textual_id(old_fn, xlsynth_pir::ir::NodeRef { index: oi }),
                xlsynth_pir::ir::node_textual_id(new_fn, xlsynth_pir::ir::NodeRef { index: ni }),
            );
        }
    }
}

fuzz_target!(|pair: FuzzSampleSameTypedPair| {
    info!("Sample generated.");

    // Early-return on degenerate inputs.
    if pair.first.ops.is_empty() || pair.second.ops.is_empty() {
        return;
    }
    info!("Sample is not degenerate.");

    let _ = env_logger::builder().is_test(true).try_init();

    // Build two XLS IR functions via the C++ bindings
    let mut pkg1 = IrPackage::new("first").expect("IrPackage::new infra error");
    let _ = match generate_ir_fn(pair.first.ops.clone(), &mut pkg1, None) {
        Ok(f) => f,
        Err(_) => return, // unsupported generator outputs are skipped
    };

    let mut pkg2 = IrPackage::new("second").expect("IrPackage::new infra error");
    let _ = match generate_ir_fn(pair.second.ops.clone(), &mut pkg2, None) {
        Ok(f) => f,
        Err(_) => return,
    };

    info!("IR function pair generated.");

    // Convert both to text and parse via our Rust parser to obtain xls_ir::ir::Fn
    let pkg1_text = pkg1.to_string();
    let parsed1 = match ir_parser::Parser::new(&pkg1_text).parse_and_validate_package() {
        Ok(p) => p,
        Err(_) => return,
    };
    let old_fn = match parsed1.get_top() {
        Some(f) => f,
        None => return,
    };

    let pkg2_text = pkg2.to_string();
    let parsed2 = match ir_parser::Parser::new(&pkg2_text).parse_and_validate_package() {
        Ok(p) => p,
        Err(_) => return,
    };
    let new_fn = match parsed2.get_top() {
        Some(f) => f,
        None => return,
    };
    info!(
        "Parsable sample generated. sizes: old={}, new={}",
        old_fn.nodes.len(),
        new_fn.nodes.len()
    );
    debug!("OLD IR TEXT:\n{}", pkg1_text);
    debug!("NEW IR TEXT:\n{}", pkg2_text);
    info!(
        "old_fn.nodes.len() = {}, new_fn.nodes.len() = {}",
        old_fn.nodes.len(),
        new_fn.nodes.len()
    );

    let mut selector = GreedyMatchSelector::new(old_fn, new_fn);
    let edits =
        compute_function_edit(old_fn, new_fn, &mut selector).expect("compute_function_edit Err");

    let patched =
        apply_function_edits(old_fn, &edits).expect("apply_function_edits returned Err (greedy)");
    debug!("PATCHED IR TEXT:\n{}", patched);
    debug!(
        "IR EDITS ({}):\n{}",
        edits.edits.len(),
        format_ir_edits(old_fn, &edits)
    );
    assert!(functions_structurally_equivalent(&patched, new_fn));

    // Rerun matching after running DCE. This enables more precise testing of
    // expected invariants around equivalence.
    let old_dce = remove_dead_nodes(old_fn);
    let new_dce = remove_dead_nodes(new_fn);
    debug!("OLD IR (AFTER DCE):\n{}", old_dce);
    debug!("NEW IR (AFTER DCE):\n{}", new_dce);

    // Compute matches on DCE'd functions with greedy selector.
    let mut sel_dce = GreedyMatchSelector::new(&old_dce, &new_dce);
    let matches = compute_function_match(&old_dce, &new_dce, &mut sel_dce)
        .expect("compute_function_match Err (AFTER DCE)");
    debug!(
        "MATCHES (AFTER DCE) ({}):\n{}",
        matches.matches.len(),
        format_match_actions(&old_dce, &new_dce, &matches.matches)
    );
    assert_equivalent_nodes_are_matched(&old_dce, &new_dce, &matches);

    // Convert matches to edits and verify isomorphism on editted graph.
    let edits_dce = convert_match_set_to_edit_set(&old_dce, &new_dce, &matches);
    let patched_dce = apply_function_edits(&old_dce, &edits_dce)
        .expect("apply_function_edits returned Err (AFTER DCE)");
    debug!("PATCHED IR (AFTER DCE):\n{}", patched_dce);
    assert!(functions_structurally_equivalent(&patched_dce, &new_dce));
});
