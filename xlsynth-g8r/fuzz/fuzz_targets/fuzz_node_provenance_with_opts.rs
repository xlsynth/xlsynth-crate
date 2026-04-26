// SPDX-License-Identifier: Apache-2.0

#![no_main]

use std::collections::HashSet;

use libfuzzer_sys::fuzz_target;
use rand::SeedableRng;
use rand::rngs::StdRng;
use xlsynth_g8r::aig::cut_db_rewrite::{RewriteOptions, rewrite_gatefn_with_cut_db};
use xlsynth_g8r::aig::fraig::{IterationBounds, fraig_optimize};
use xlsynth_g8r::aig::gate::AigNode;
use xlsynth_g8r::cut_db::loader::CutDb;
use xlsynth_g8r::gatify::ir2gate::{self, GatifyOptions};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_pir::ir_fuzz::{FuzzSample, generate_ir_fn};
use xlsynth_pir::ir_parser;

fuzz_target!(|sample: FuzzSample| {
    let _ = env_logger::builder().is_test(true).try_init();

    // Degenerate samples with no ops do not exercise any meaningful gatify
    // provenance behavior, so skip them rather than biasing the corpus toward
    // trivial zero-work functions.
    if sample.ops.is_empty() {
        return;
    }

    let mut package = xlsynth::IrPackage::new("fuzz_node_provenance_with_opts")
        .expect("IrPackage::new should not fail for fuzz target setup");
    if let Err(_e) = generate_ir_fn(sample.ops, &mut package, None) {
        // The shared IR generator can intentionally explore edge-case programs
        // outside the current g8r lowering surface. Those are not sample
        // failures for this provenance property, which only applies once
        // gatification is valid.
        return;
    }

    let package_text = package.to_string();
    let parsed_package = ir_parser::Parser::new(&package_text)
        .parse_and_validate_package()
        .expect("C++-emitted IR should parse and validate in PIR");
    let parsed_fn = parsed_package
        .get_top_fn()
        .expect("generated package should have a top function");
    let original_text_ids: HashSet<u32> = parsed_fn
        .nodes
        .iter()
        .map(|node| node.text_id as u32)
        .collect();

    let gatify_output = match ir2gate::gatify(
        parsed_fn,
        GatifyOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            adder_mapping: AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            enable_rewrite_nary_add: false,
            enable_rewrite_mask_low: false,
            array_index_lowering_strategy: Default::default(),
        },
    ) {
        Ok(output) => output,
        // The generator spans more PIR than the current g8r lowering supports.
        // Unsupported samples are not provenance failures; skip them so the
        // target focuses on successful lowering plus optimization behavior.
        Err(_) => return,
    };
    let mut rng = StdRng::seed_from_u64(0);
    let optimized_gate_fn = fraig_optimize(
        &gatify_output.gate_fn,
        64,
        IterationBounds::MaxIterations(1),
        &mut rng,
    )
    .expect("fraig should not fail on successfully lowered GateFn")
    .0;
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
            AigNode::Literal { value, pir_node_ids } => {
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
