// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use log::{debug, info};
use xlsynth::IrPackage;
use xlsynth_pir::graph_edit::{
    IrEdit, IrEditSet, IrMatchSet, MatchAction, apply_function_edits, compute_forward_equivalences,
    compute_function_match, convert_match_set_to_edit_set, format_ir_edits, format_match_set,
};
use xlsynth_pir::greedy_graph_edit::GreedyMatchSelector;
use xlsynth_pir::ir_fuzz::{FuzzSampleSameTypedPair, generate_ir_fn};
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

fn assert_pairs_have_matches(
    old_fn: &xlsynth_pir::ir::Fn,
    new_fn: &xlsynth_pir::ir::Fn,
    pairs: &[(usize, usize)],
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
    for &(oi, ni) in pairs.iter() {
        if !matched_pairs.contains(&(oi, ni)) {
            panic!(
                "missing MatchNodes for uniquely forward-equivalent pair: {} <-> {}",
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
    let matches =
        compute_function_match(old_fn, new_fn, &mut selector).expect("compute_function_match Err");
    debug!(
        "MATCHES ({}):\n{}",
        matches.matches.len(),
        format_match_set(old_fn, new_fn, &matches)
    );
    let unique_pairs = unique_forward_equivalent_pairs(old_fn, new_fn);
    assert_pairs_have_matches(old_fn, new_fn, &unique_pairs, &matches);
    let edits = convert_match_set_to_edit_set(old_fn, new_fn, &matches);
    let patched =
        apply_function_edits(old_fn, &edits).expect("apply_function_edits returned Err (greedy)");
    debug!("PATCHED IR TEXT:\n{}", patched);
    debug!(
        "IR EDITS ({}):\n{}",
        edits.edits.len(),
        format_ir_edits(old_fn, &edits)
    );
    assert!(is_ir_isomorphic(&patched, new_fn));
});
