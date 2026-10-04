// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use mffc_corpus::{Expectations, Limits, MappingProfile};
use xlsynth_aug_opt::run_aug_opt_over_ir_text_with_stats;
use xlsynth_aug_opt::{AugOptMode, AugOptOptions};
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig::graph_logical_effort::{
    GraphLogicalEffortOptions, analyze_graph_logical_effort,
};
use xlsynth_g8r::check_equivalence::validate_same_fn_via_toolchain;
use xlsynth_g8r::process_ir_path::process_ir_text_with_gatefn;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::node_hashing::functions_structurally_equivalent;

pub mod mffc_corpus;

/// Records reachable gate cost after the same internal cleanup on both sides.
#[derive(Debug)]
struct MeasuredCost {
    and_nodes: usize,
    graph_le: f64,
    depth: usize,
}

/// Retains one serialization's results for comparison with its ordering
/// variant.
struct RecoveryResult<'a> {
    original: &'a ir::Fn,
    optimized: ir::Fn,
    disabled: MeasuredCost,
    enabled: MeasuredCost,
}

/// Maps and proves one optimizer output against its untouched fixture.
fn measure_cost(
    text: &str,
    original: &ir::Fn,
    mode: &MappingProfile,
) -> Result<MeasuredCost, String> {
    let mut options = mode.canonical_options.to_process_ir_path_options(
        Some(&original.name),
        true,
        false,
        false,
        None,
    );
    options.cut_db_rewrite_max_iterations = mode.cut_db_rewrite_max_iterations;
    options.cut_db_rewrite_max_cuts_per_node = mode.cut_db_rewrite_max_cuts_per_node;
    let (gate_fn, _) = process_ir_text_with_gatefn(text, &options)?;
    validate_same_fn_via_toolchain(original, &gate_fn)?;
    let stats = get_aig_stats(&gate_fn);
    let graph_le_options = GraphLogicalEffortOptions {
        beta1: mode.canonical_options.graph_logical_effort_beta1,
        beta2: mode.canonical_options.graph_logical_effort_beta2,
    };
    Ok(MeasuredCost {
        and_nodes: stats.and_nodes,
        graph_le: analyze_graph_logical_effort(&gate_fn, &graph_le_options).delay,
        depth: stats.max_depth,
    })
}

/// Checks absolute bounds so known area/delay tradeoffs remain reviewable.
fn assert_within_limits(cost: &MeasuredCost, limits: &Limits, tolerance: f64, context: &str) {
    assert!(
        cost.and_nodes <= limits.and_nodes_max
            && cost.graph_le <= limits.graph_le_max + tolerance
            && cost.depth <= limits.depth_max,
        "{context}: exceeded recorded QoR limits: {cost:?}",
    );
}

/// Checks fixed IR and aug-opt under one pinned mapping profile.
#[test]
fn mffc_corpus_respects_equivalence_and_qor_limits() {
    let corpus = mffc_corpus::load().unwrap();
    assert!(
        corpus
            .cases
            .iter()
            .any(|case| matches!(case.expectations, Expectations::Shift { .. }))
    );
    let mode = &corpus.profile;
    let tolerance = mode.graph_le_tolerance;
    assert!(!mode.canonical_options.fraig);

    for case in corpus.cases {
        let Expectations::Shift {
            require_improvement,
            limits,
        } = case.expectations
        else {
            // Add recovery has its own flag comparison and bounds below.
            continue;
        };
        let rewritten = run_aug_opt_over_ir_text_with_stats(
            &case.text,
            Some(&case.original.name),
            AugOptOptions {
                enable: true,
                rounds: 1,
                mode: AugOptMode::PirOnly,
                ..Default::default()
            },
        )
        .expect("aug-opt succeeds on corpus fixture");
        if require_improvement {
            assert!(rewritten.rewrite_stats.constant_shift_choices > 0);
        }
        let [fixed, augmented] = [
            ("fixed IR", case.text.as_str()),
            ("aug-opt", rewritten.output_text.as_str()),
        ]
        .map(|(pipeline, input)| {
            measure_cost(input, &case.original, mode)
                .unwrap_or_else(|error| panic!("{} ({pipeline}): {error}", case.name))
        });

        eprintln!("{}: fixed={fixed:?}, aug-opt={augmented:?}", case.name);
        assert_within_limits(&augmented, &limits, tolerance, &case.name);
        assert!(
            augmented.and_nodes <= fixed.and_nodes
                && augmented.graph_le <= fixed.graph_le + tolerance,
            "{}: aug-opt regressed: fixed={fixed:?}, augmented={augmented:?}",
            case.name,
        );
        if require_improvement {
            assert!(
                augmented.and_nodes < fixed.and_nodes
                    || augmented.graph_le < fixed.graph_le - tolerance,
                "{}: expected aug-opt to improve: fixed={fixed:?}, augmented={augmented:?}",
                case.name,
            );
        }
    }
}

/// Checks both add-recovery flag settings without imposing a Pareto policy.
#[test]
fn split_adder_corpus_respects_equivalence_limits_and_ordering() {
    let corpus = mffc_corpus::load().unwrap();
    assert!(
        corpus
            .cases
            .iter()
            .any(|case| matches!(case.expectations, Expectations::SplitAdder { .. }))
    );
    assert!(!corpus.profile.canonical_options.fraig);
    let tolerance = corpus.profile.graph_le_tolerance;

    for mode in [AugOptMode::PirOnly, AugOptMode::Sandwich] {
        let mut results = BTreeMap::new();
        for case in &corpus.cases {
            let Expectations::SplitAdder {
                disabled_limits,
                enabled_limits,
                ..
            } = &case.expectations
            else {
                // Shift fixtures are exercised with their own comparison above.
                continue;
            };
            let context = format!("{} ({mode:?})", case.name);
            let [disabled, enabled] = [false, true].map(|recover_split_adders| {
                run_aug_opt_over_ir_text_with_stats(
                    &case.text,
                    Some(&case.original.name),
                    AugOptOptions {
                        enable: true,
                        rounds: 1,
                        mode,
                        recover_split_adders,
                    },
                )
                .unwrap_or_else(|e| panic!("{context}: recovery={recover_split_adders}: {e}"))
            });
            assert_eq!(
                disabled.rewrite_stats.split_adders_recovered, 0,
                "{context}"
            );
            assert_eq!(enabled.rewrite_stats.split_adders_recovered, 1, "{context}");
            let optimized = Parser::new(&enabled.output_text)
                .parse_and_validate_package()
                .unwrap()
                .get_fn(&case.original.name)
                .unwrap()
                .clone();
            let disabled = measure_cost(&disabled.output_text, &case.original, &corpus.profile)
                .unwrap_or_else(|e| panic!("{context}: recovery disabled: {e}"));
            let enabled = measure_cost(&enabled.output_text, &case.original, &corpus.profile)
                .unwrap_or_else(|e| panic!("{context}: recovery enabled: {e}"));
            eprintln!(
                "{} {mode:?}: disabled={disabled:?}, enabled={enabled:?}",
                case.name
            );
            assert_within_limits(
                &disabled,
                disabled_limits,
                tolerance,
                &format!("{context}, disabled"),
            );
            assert_within_limits(
                &enabled,
                enabled_limits,
                tolerance,
                &format!("{context}, enabled"),
            );
            results.insert(
                case.name.as_str(),
                RecoveryResult {
                    original: &case.original,
                    optimized,
                    disabled,
                    enabled,
                },
            );
        }

        for case in &corpus.cases {
            let Expectations::SplitAdder {
                same_as: Some(same_as),
                ..
            } = &case.expectations
            else {
                // Only designated ordering variants need a peer comparison.
                continue;
            };
            let actual = &results[case.name.as_str()];
            let expected = &results[same_as.as_str()];
            assert!(functions_structurally_equivalent(
                actual.original,
                expected.original
            ));
            assert!(functions_structurally_equivalent(
                &actual.optimized,
                &expected.optimized
            ));
            for (actual, expected) in [
                (&actual.disabled, &expected.disabled),
                (&actual.enabled, &expected.enabled),
            ] {
                assert_eq!(actual.and_nodes, expected.and_nodes);
                assert_eq!(actual.depth, expected.depth);
                assert!((actual.graph_le - expected.graph_le).abs() <= tolerance);
            }
        }
    }
}
