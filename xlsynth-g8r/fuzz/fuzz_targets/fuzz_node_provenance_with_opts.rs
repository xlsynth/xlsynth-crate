// SPDX-License-Identifier: Apache-2.0

#![no_main]

use std::collections::HashSet;

use libfuzzer_sys::fuzz_target;
use rand::SeedableRng;
use rand::rngs::StdRng;
use xlsynth_g8r::aig::cut_db_rewrite::{RewriteOptions, rewrite_gatefn_with_cut_db};
use xlsynth_g8r::aig::fraig::fraig_optimize_with_backend_and_options;
use xlsynth_g8r::aig::gate::AigNode;
use xlsynth_g8r::cut_db::loader::CutDb;
use xlsynth_g8r::gatify::ir2gate::{self, GatifyOptions};
use xlsynth_g8r::prove_gate_fn_equiv_common::GateFormalBackend;
use xlsynth_g8r::prove_gate_fn_equiv_sat::ValidationError;
use xlsynth_g8r_fuzz::{fuzz_gate_formal_options, generate_gatify_random_pir_package};

fuzz_target!(|data: &[u8]| {
    let _ = env_logger::builder().is_test(true).try_init();

    let package = generate_gatify_random_pir_package(data, "fuzz_node_provenance_with_opts");
    let parsed_fn = package
        .get_top_fn()
        .expect("generated package should have a top function");
    let original_text_ids: HashSet<u32> = parsed_fn
        .nodes
        .iter()
        .map(|node| node.text_id as u32)
        .collect();

    let gatify_output = ir2gate::gatify(parsed_fn, GatifyOptions::all_opts_disabled())
        .expect("generated standard PIR should lower successfully");
    let mut rng = StdRng::seed_from_u64(0);
    let optimized_gate_fn = match fraig_optimize_with_backend_and_options(
        &gatify_output.gate_fn,
        64,
        GateFormalBackend::Cadical,
        fuzz_gate_formal_options(),
        &mut rng,
    ) {
        Ok(result) => result.optimized_fn,
        Err(err)
            if matches!(
                err.downcast_ref::<ValidationError>(),
                Some(ValidationError::CadicalSolveInterrupted)
            ) =>
        {
            // A configured solver timeout is expected to make some fuzz
            // samples inconclusive; those samples are not provenance failures.
            return;
        }
        Err(err) => panic!("fraig should not fail on successfully lowered GateFn: {err}"),
    };
    let optimized_gate_fn = rewrite_gatefn_with_cut_db(
        &optimized_gate_fn,
        CutDb::load_default().as_ref(),
        RewriteOptions {
            max_cuts_per_node: 32,
            max_iterations: 1,
            ..RewriteOptions::default()
        },
    );

    for (index, node) in optimized_gate_fn.gates.iter().enumerate() {
        let pir_node_ids = node.get_pir_node_ids();
        assert!(
            pir_node_ids.windows(2).all(|pair| pair[0] < pair[1]),
            "provenance ids should remain sorted and deduplicated: node={:?}",
            node
        );
        for pir_node_id in pir_node_ids {
            assert!(
                original_text_ids.contains(pir_node_id),
                "provenance id {} should correspond to an original pre-prep PIR node",
                pir_node_id
            );
        }
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
                let _ = pir_node_ids;
            }
            AigNode::Input { pir_node_ids, .. } | AigNode::And2 { pir_node_ids, .. } => {
                assert!(
                    !pir_node_ids.is_empty(),
                    "each surviving optimized g8r node should carry non-empty PIR provenance ids: node={:?}",
                    node
                );
            }
        }
    }
});
