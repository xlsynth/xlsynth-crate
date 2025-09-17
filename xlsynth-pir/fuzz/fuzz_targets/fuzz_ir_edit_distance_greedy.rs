// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth::IrPackage;
use xlsynth_pir::graph_edit::{apply_function_edits, compute_function_edit};
use xlsynth_pir::greedy_graph_edit::GreedyMatchSelector;
use xlsynth_pir::ir_fuzz::{FuzzSampleSameTypedPair, generate_ir_fn};
use xlsynth_pir::{ir_isomorphism::is_ir_isomorphic, ir_parser};

fuzz_target!(|pair: FuzzSampleSameTypedPair| {
    println!("RUN!");

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
    println!(
        "*********************** GENERATED PARSABLE SAMPLE (size={}/{})",
        old_fn.nodes.len(),
        new_fn.nodes.len()
    );
    // Compute edits with the greedy matcher, apply, and verify isomorphism.
    let mut selector = GreedyMatchSelector::new(old_fn, new_fn);
    let edits = compute_function_edit(old_fn, new_fn, &mut selector)
        .expect("compute_function_edit returned Err (greedy)");
    let patched =
        apply_function_edits(old_fn, &edits).expect("apply_function_edits returned Err (greedy)");
    // Print IR texts for old sample, new sample, and the patched result.
    assert!(is_ir_isomorphic(&patched, new_fn));
});
