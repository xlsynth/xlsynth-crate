// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use log::{debug, info};
use xlsynth::IrPackage;
use xlsynth_pir::graph_edit::{
    IrMatchSet, MatchAction, apply_function_edits, compute_forward_equivalences,
    compute_function_edit, compute_function_match, compute_reverse_equivalences_to_return,
    convert_match_set_to_edit_set, format_ir_edits, format_match_set,
};
use xlsynth_pir::greedy_graph_edit::GreedyMatchSelector;
use xlsynth_pir::ir_fuzz::{FuzzSampleSameTypedPair, generate_ir_fn};
use xlsynth_pir::ir_utils::{get_dead_nodes, remove_dead_nodes};
use xlsynth_pir::{ir_isomorphism::is_ir_isomorphic, ir_parser};

fn unique_forward_equivalent_pairs(
    old_fn: &xlsynth_pir::ir::Fn,
    new_fn: &xlsynth_pir::ir::Fn,
) -> Vec<(usize, usize)> {
    let (old_to_new_eq, new_to_old_eq) = compute_forward_equivalences(old_fn, new_fn);
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (old_ref, news) in old_to_new_eq.iter() {
        if news.len() != 1 {
            continue;
        }
        let oi: usize = usize::from(*old_ref);
        let ni: usize = usize::from(news[0]);
        if let Some(olds) = new_to_old_eq.get(&news[0]) {
            if olds.len() == 1 && usize::from(olds[0]) == oi {
                pairs.push((oi, ni));
            }
        }
    }
    pairs
}

fn unique_reverse_equivalent_pairs(
    old_fn: &xlsynth_pir::ir::Fn,
    new_fn: &xlsynth_pir::ir::Fn,
) -> Vec<(usize, usize)> {
    let (old_to_new_eq, new_to_old_eq) = compute_reverse_equivalences_to_return(old_fn, new_fn);
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (old_ref, news) in old_to_new_eq.iter() {
        if news.len() != 1 {
            continue;
        }
        let oi: usize = usize::from(*old_ref);
        let ni: usize = usize::from(news[0]);
        if let Some(olds) = new_to_old_eq.get(&news[0]) {
            if olds.len() == 1 && usize::from(olds[0]) == oi {
                pairs.push((oi, ni));
            }
        }
    }
    pairs
}

fn assert_pairs_have_matches(
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
    // Early-return on degenerate inputs.
    if pair.first.ops.is_empty()
        || pair.second.ops.is_empty()
        || pair.first.input_bits == 0
        || pair.second.input_bits == 0
    {
        return;
    }

    let _ = env_logger::builder().is_test(true).try_init();

    // Build two XLS IR functions via the C++ bindings
    let mut pkg1 = IrPackage::new("first").expect("IrPackage::new infra error");
    let _ = match generate_ir_fn(
        pair.first.input_bits,
        pair.first.ops.clone(),
        &mut pkg1,
        None,
    ) {
        Ok(f) => f,
        Err(_) => return, // unsupported generator outputs are skipped
    };

    let mut pkg2 = IrPackage::new("second").expect("IrPackage::new infra error");
    let _ = match generate_ir_fn(
        pair.second.input_bits,
        pair.second.ops.clone(),
        &mut pkg2,
        None,
    ) {
        Ok(f) => f,
        Err(_) => return,
    };

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
    // Compute matches with the greedy matcher, verify unique forward equivalences
    // are represented as matches, then convert to edits and verify isomorphism.
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
    assert!(is_ir_isomorphic(&patched, new_fn));

    // Additional step: run dead code elimination on both functions, then perform
    // greedy matching+edits on the DCE'd graphs. Verify isomorphism and that
    // uniquely equivalent pairs (both directions) have matches.
    let old_dce = remove_dead_nodes(old_fn);
    let new_dce = remove_dead_nodes(new_fn);
    debug!("OLD IR (AFTER DCE):\n{}", old_dce);
    debug!("NEW IR (AFTER DCE):\n{}", new_dce);

    // Compute matches on DCE'd functions with greedy selector.
    let mut sel_dce = GreedyMatchSelector::new(&old_dce, &new_dce);
    let matches = compute_function_match(&old_dce, &new_dce, &mut sel_dce)
        .expect("compute_function_match Err (AFTER DCE)");
    // Optionally debug print matches.
    debug!(
        "MATCHES (AFTER DCE) ({}):\n{}",
        matches.matches.len(),
        format_match_set(&old_dce, &new_dce, &matches)
    );
    assert_pairs_have_matches(&old_dce, &new_dce, &matches);

    // Convert matches to edits and verify isomorphism on DCE'd graphs.
    let edits_dce = convert_match_set_to_edit_set(&old_dce, &new_dce, &matches);
    let patched_dce = apply_function_edits(&old_dce, &edits_dce)
        .expect("apply_function_edits returned Err (AFTER DCE)");
    debug!("PATCHED IR (AFTER DCE):\n{}", patched_dce);
    assert!(is_ir_isomorphic(&patched_dce, &new_dce));

    // debug!(
    //     "MATCHES ({}):\n{}",
    //     matches.matches.len(),
    //     format_match_set(old_fn, new_fn, &matches)
    // );
    // assert_pairs_have_matches(old_fn, new_fn, &matches);
    // let edits = convert_match_set_to_edit_set(old_fn, new_fn, &matches);
});
