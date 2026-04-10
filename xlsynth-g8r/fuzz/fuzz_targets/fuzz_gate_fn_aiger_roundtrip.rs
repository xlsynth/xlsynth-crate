// SPDX-License-Identifier: Apache-2.0
#![no_main]
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::aig::GateBuilderOptions;
use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
use xlsynth_g8r::aig_serdes::gate2ir::{
    repack_gate_fn_interface_with_schema, GateFnInterfaceSchema,
};
use xlsynth_g8r::aig_serdes::load_aiger::load_aiger;
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
    let schema = GateFnInterfaceSchema::from_gate_fn(&g)
        .expect("original GateFn interface should produce a regroup schema");
    let repacked = repack_gate_fn_interface_with_schema(loaded_fn, &schema)
        .expect("loaded AIGER should regroup under the original GateFn schema");

    assert!(structurally_equivalent(&g, &repacked));
});
