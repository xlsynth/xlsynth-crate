// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth::IrPackage;
use xlsynth_pir::ir_fuzz::{FuzzSampleSameTypedPair, generate_ir_fn};
use xlsynth_pir::matching_ged::apply_fn_edits;
use xlsynth_pir::{ir_parser, node_hashing::functions_structurally_equivalent};

fuzz_target!(|pair: FuzzSampleSameTypedPair| {
    // Skip degenerate samples early.
    if pair.first.ops.is_empty() || pair.second.ops.is_empty() {
        return;
    }

    let _ = env_logger::builder().is_test(true).try_init();

    // Build two XLS IR functions via the C++ bindings
    let mut pkg1 = IrPackage::new("first").expect("IrPackage::new infra error");
    let _func1 = match generate_ir_fn(pair.first.ops.clone(), &mut pkg1, None) {
        Ok(f) => f,
        Err(_) => return, // unsupported generator outputs are skipped
    };

    let mut pkg2 = IrPackage::new("second").expect("IrPackage::new infra error");
    let _func2 = match generate_ir_fn(pair.second.ops.clone(), &mut pkg2, None) {
        Ok(f) => f,
        Err(_) => return,
    };

    // Convert both to text and parse via our Rust parser to obtain xls_ir::ir::Fn
    let pkg1_text = pkg1.to_string();
    let parsed1 = match ir_parser::Parser::new(&pkg1_text).parse_and_validate_package() {
        Ok(p) => p,
        Err(_) => return,
    };
    let old_fn = match parsed1.get_top_fn() {
        Some(f) => f,
        None => return,
    };

    let pkg2_text = pkg2.to_string();
    let parsed2 = match ir_parser::Parser::new(&pkg2_text).parse_and_validate_package() {
        Ok(p) => p,
        Err(_) => return,
    };
    let new_fn = match parsed2.get_top_fn() {
        Some(f) => f,
        None => return,
    };

    // Compute edit distance, apply to old, and verify isomorphism.
    let mut selector = xlsynth_pir::matching_ged::NaiveMatchSelector::new(old_fn, new_fn);
    let edits = xlsynth_pir::matching_ged::compute_fn_edit(old_fn, new_fn, &mut selector)
        .expect("compute_function_edit returned Err");
    let patched = apply_fn_edits(old_fn, &edits).expect("apply_fn_edits returned Err");
    assert!(functions_structurally_equivalent(&patched, new_fn));
});
