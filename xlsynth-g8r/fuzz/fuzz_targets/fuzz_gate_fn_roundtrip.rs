// SPDX-License-Identifier: Apache-2.0
#![no_main]
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::test_utils::structurally_equivalent;
use xlsynth_g8r_fuzz::{build_graph, FuzzGraph};

fuzz_target!(|graph: FuzzGraph| {
    let _ = env_logger::builder().is_test(true).try_init();
    let Some(g) = build_graph(&graph) else {
        return;
    };
    let text = g.to_string();
    let parsed = match GateFn::try_from(text.as_str()) {
        Ok(p) => p,
        Err(_) => return,
    };
    assert!(structurally_equivalent(&g, &parsed));
});
