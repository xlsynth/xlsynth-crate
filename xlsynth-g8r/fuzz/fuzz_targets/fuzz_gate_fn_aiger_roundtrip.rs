// SPDX-License-Identifier: Apache-2.0
#![no_main]
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::emit_aiger::emit_aiger;
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::load_aiger::load_aiger;
use xlsynth_g8r::test_utils::structurally_equivalent;
use xlsynth_g8r_fuzz::{build_graph, FuzzGraph};

fuzz_target!(|graph: FuzzGraph| {
    let _ = env_logger::builder().is_test(true).try_init();
    let Some(g) = build_graph(&graph) else {
        return;
    };

    let aiger = match emit_aiger(&g, true) {
        Ok(t) => t,
        Err(e) => panic!("emit_aiger failed: {}", e),
    };
    let loaded_fn = match load_aiger(&aiger, GateBuilderOptions::no_opt()) {
        Ok(res) => res.gate_fn,
        Err(e) => panic!("load_aiger failed: {}", e),
    };

    assert!(structurally_equivalent(&g, &loaded_fn));
});
