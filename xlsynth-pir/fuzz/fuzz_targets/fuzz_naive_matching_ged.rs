// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth_pir::matching_ged::apply_fn_edits;
use xlsynth_pir::node_hashing::functions_structurally_equivalent;
use xlsynth_pir_fuzz::generate_full_random_pir_pair;

fuzz_target!(|data: &[u8]| {
    let _ = env_logger::builder().is_test(true).try_init();

    let (old_fn, new_fn) = generate_full_random_pir_pair(data);

    // Compute edit distance, apply to old, and verify isomorphism.
    let mut selector = xlsynth_pir::matching_ged::NaiveMatchSelector::new(&old_fn, &new_fn);
    let edits = xlsynth_pir::matching_ged::compute_fn_edit(&old_fn, &new_fn, &mut selector)
        .expect("compute_function_edit returned Err");
    let patched = apply_fn_edits(&old_fn, &edits).expect("apply_fn_edits returned Err");
    assert!(functions_structurally_equivalent(&patched, &new_fn));
});
