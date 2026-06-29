// SPDX-License-Identifier: Apache-2.0

#![no_main]

use std::collections::HashSet;

use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::aig::gate::AigNode;
use xlsynth_g8r::gatify::ir2gate::{self, GatifyOptions};
use xlsynth_g8r_fuzz::generate_gatify_random_pir_package;

fuzz_target!(|data: &[u8]| {
    let _ = env_logger::builder().is_test(true).try_init();

    let package = generate_gatify_random_pir_package(data, "fuzz_node_provenance");
    let parsed_fn = package
        .get_top_fn()
        .expect("generated package should have a top function");
    let original_text_ids: HashSet<u32> = parsed_fn
        .nodes
        .iter()
        .map(|node| node.text_id as u32)
        .collect();

    let gatify_output = ir2gate::gatify(
        parsed_fn,
        GatifyOptions {
            fold: false,
            hash: false,
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .expect("generated standard PIR should lower successfully");

    for (index, node) in gatify_output.gate_fn.gates.iter().enumerate() {
        match node {
            AigNode::Literal {
                value,
                pir_node_ids,
            } => {
                assert_eq!(
                    index, 0,
                    "only the builder's constant-false literal should appear in gatify output"
                );
                assert!(
                    !*value,
                    "the builder's dedicated literal node should be constant false"
                );
                assert!(
                    pir_node_ids.windows(2).all(|pair| pair[0] < pair[1]),
                    "literal node provenance ids should be sorted and deduped: node={:?}",
                    node
                );
                for pir_node_id in pir_node_ids.iter() {
                    assert!(
                        original_text_ids.contains(pir_node_id),
                        "literal provenance id {} should correspond to an original pre-prep PIR node",
                        pir_node_id
                    );
                }
            }
            AigNode::Input { pir_node_ids, .. } | AigNode::And2 { pir_node_ids, .. } => {
                assert!(
                    !pir_node_ids.is_empty(),
                    "each lowered g8r node should carry at least one PIR provenance id: node={:?}",
                    node
                );
                assert!(
                    pir_node_ids.windows(2).all(|pair| pair[0] < pair[1]),
                    "lowered g8r node provenance ids should be sorted and deduped: node={:?}",
                    node
                );
                for pir_node_id in pir_node_ids.iter() {
                    assert!(
                        original_text_ids.contains(pir_node_id),
                        "provenance id {} should correspond to an original pre-prep PIR node",
                        pir_node_id
                    );
                }
            }
        }
    }
});
