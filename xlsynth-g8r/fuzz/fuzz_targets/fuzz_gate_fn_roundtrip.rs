// SPDX-License-Identifier: Apache-2.0
#![no_main]
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::gate_random::RandomGateOptions;
use xlsynth_g8r::test_utils::structurally_equivalent;
use xlsynth_g8r_fuzz::direct_g8r::generate_combinational;

fuzz_target!(|graph: &[u8]| {
    let _ = env_logger::builder().is_test(true).try_init();
    let g = generate_combinational(graph, &RandomGateOptions::default())
        .expect("valid bounded combinational generator options")
        .graph;
    let text = g.to_string_including_dead_nodes();
    let parsed = match GateFn::try_from(text.as_str()) {
        Ok(p) => p,
        Err(error) => panic!("generated graph text did not parse: {error}\n{text}"),
    };
    assert!(structurally_equivalent(&g, &parsed));
});
