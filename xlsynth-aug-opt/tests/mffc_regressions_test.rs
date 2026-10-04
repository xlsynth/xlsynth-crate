// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use xlsynth_aug_opt::run_aug_opt_over_ir_text_with_stats;
use xlsynth_aug_opt::{AugOptMode, AugOptOptions};
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig::graph_logical_effort::{
    GraphLogicalEffortOptions, analyze_graph_logical_effort,
};
use xlsynth_g8r::check_equivalence::validate_same_fn_via_toolchain;
use xlsynth_g8r::process_ir_path::{CanonicalG8rOptions, process_ir_text_with_gatefn};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::node_hashing::functions_structurally_equivalent;

/// Pins the shared mapping profile and expectations for each fixture family.
#[derive(Deserialize)]
struct Corpus {
    schema: String,
    measurement_mode: MeasurementMode,
    cases: Vec<Case>,
    split_adder_cases: Vec<SplitAdderCase>,
}

/// Keeps gate mapping identical across the compared optimizer configurations.
#[derive(Deserialize)]
struct MeasurementMode {
    input: String,
    abc: bool,
    canonical_options: CanonicalG8rOptions,
    cut_db_rewrite_max_iterations: usize,
    cut_db_rewrite_max_cuts_per_node: usize,
}

/// Identifies a fixed IR input and detects accidental changes to its contents.
#[derive(Deserialize)]
struct Fixture {
    name: String,
    file: String,
    top: String,
    sha256: String,
}

/// Retains the original shift corpus's strict nonregression expectations.
#[derive(Deserialize)]
struct Case {
    #[serde(flatten)]
    fixture: Fixture,
    require_improvement: bool,
    regression_limits: Limits,
}

/// Shares measured bounds for add recovery across exact-input and sandwich
/// modes.
#[derive(Deserialize)]
struct SplitAdderCase {
    #[serde(flatten)]
    fixture: Fixture,
    disabled_limits: Limits,
    enabled_limits: Limits,
    same_as: Option<String>,
}

/// Allows improvements while guarding each measured area and delay metric.
#[derive(Deserialize)]
struct Limits {
    and_nodes_max: usize,
    graph_le_max: f64,
    depth_max: usize,
    graph_le_tolerance: f64,
}

/// Records reachable gate cost after the same internal cleanup on both sides.
#[derive(Debug)]
struct MeasuredCost {
    and_nodes: usize,
    graph_le: f64,
    depth: usize,
}

/// Owns a verified fixture's text and parsed function for equivalence checking.
struct LoadedFixture {
    text: String,
    original: ir::Fn,
}

/// Retains one serialization's results for comparison with its ordering
/// variant.
struct RecoveryResult {
    original: ir::Fn,
    optimized: ir::Fn,
    disabled: MeasuredCost,
    enabled: MeasuredCost,
}

/// Reads a fixed input and checks its hash, syntax, and declared top function.
fn load_fixture(root: &Path, fixture: &Fixture) -> Result<LoadedFixture, String> {
    let text = std::fs::read_to_string(root.join(&fixture.file)).map_err(|e| e.to_string())?;
    let actual_hash = format!("{:x}", Sha256::digest(text.as_bytes()));
    if actual_hash != fixture.sha256 {
        return Err(format!("{}: fixture content changed", fixture.name));
    }
    let package = Parser::new(&text)
        .parse_and_validate_package()
        .map_err(|e| e.to_string())?;
    let original = package
        .get_fn(&fixture.top)
        .cloned()
        .ok_or_else(|| format!("{}: fixture top is missing", fixture.name))?;
    Ok(LoadedFixture { text, original })
}

/// Maps and proves one optimizer output against its untouched fixture.
fn measure_cost(
    text: &str,
    original: &ir::Fn,
    mode: &MeasurementMode,
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
fn assert_within_limits(cost: &MeasuredCost, limits: &Limits, context: &str) {
    assert!(limits.graph_le_tolerance.is_finite() && limits.graph_le_tolerance >= 0.0);
    assert!(
        cost.and_nodes <= limits.and_nodes_max
            && cost.graph_le <= limits.graph_le_max + limits.graph_le_tolerance
            && cost.depth <= limits.depth_max,
        "{context}: exceeded recorded QoR limits: {cost:?}",
    );
}

/// Checks fixed IR and aug-opt under one pinned mapping profile.
#[test]
fn mffc_corpus_respects_equivalence_and_qor_limits() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mffc_regressions");
    let corpus: Corpus =
        serde_json::from_str(include_str!("fixtures/mffc_regressions/manifest.json"))
            .expect("valid regression manifest");
    assert_eq!(corpus.schema, "g8r-standalone-ir-regression-corpus-v1");
    assert!(!corpus.cases.is_empty());
    let mode = &corpus.measurement_mode;
    assert_eq!(
        mode.input,
        "fixed IR baseline; one PIR-only aug-opt round for the candidate"
    );
    assert!(!mode.abc && !mode.canonical_options.fraig);

    for case in corpus.cases {
        let fixture = &case.fixture;
        let LoadedFixture { text, original } = load_fixture(&root, fixture).unwrap();
        let rewritten = run_aug_opt_over_ir_text_with_stats(
            &text,
            Some(&fixture.top),
            AugOptOptions {
                enable: true,
                rounds: 1,
                mode: AugOptMode::PirOnly,
                ..Default::default()
            },
        )
        .expect("aug-opt succeeds on corpus fixture");
        if case.require_improvement {
            assert!(rewritten.rewrite_stats.constant_shift_choices > 0);
        }
        let [fixed, augmented] = [
            ("fixed IR", text.as_str()),
            ("aug-opt", rewritten.output_text.as_str()),
        ]
        .map(|(pipeline, input)| {
            measure_cost(input, &original, mode)
                .unwrap_or_else(|error| panic!("{} ({pipeline}): {error}", fixture.name))
        });

        let limits = &case.regression_limits;
        let tolerance = limits.graph_le_tolerance;
        eprintln!("{}: fixed={fixed:?}, aug-opt={augmented:?}", fixture.name);
        assert_within_limits(&augmented, limits, &fixture.name);
        assert!(
            augmented.and_nodes <= fixed.and_nodes
                && augmented.graph_le <= fixed.graph_le + tolerance,
            "{}: aug-opt regressed: fixed={fixed:?}, augmented={augmented:?}",
            fixture.name,
        );
        if case.require_improvement {
            assert!(
                augmented.and_nodes < fixed.and_nodes
                    || augmented.graph_le < fixed.graph_le - tolerance,
                "{}: expected aug-opt to improve: fixed={fixed:?}, augmented={augmented:?}",
                fixture.name,
            );
        }
    }
}

/// Checks both add-recovery flag settings without imposing a Pareto policy.
#[test]
fn split_adder_corpus_respects_equivalence_limits_and_ordering() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mffc_regressions");
    let corpus: Corpus =
        serde_json::from_str(include_str!("fixtures/mffc_regressions/manifest.json"))
            .expect("valid regression manifest");
    assert!(!corpus.split_adder_cases.is_empty());
    assert!(!corpus.measurement_mode.abc && !corpus.measurement_mode.canonical_options.fraig);

    for mode in [AugOptMode::PirOnly, AugOptMode::Sandwich] {
        let mut results = BTreeMap::new();
        for case in &corpus.split_adder_cases {
            let fixture = &case.fixture;
            let context = format!("{} ({mode:?})", fixture.name);
            let LoadedFixture { text, original } = load_fixture(&root, fixture).unwrap();
            let [disabled, enabled] = [false, true].map(|recover_split_adders| {
                run_aug_opt_over_ir_text_with_stats(
                    &text,
                    Some(&fixture.top),
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
                .get_fn(&fixture.top)
                .unwrap()
                .clone();
            let disabled = measure_cost(&disabled.output_text, &original, &corpus.measurement_mode)
                .unwrap_or_else(|e| panic!("{context}: recovery disabled: {e}"));
            let enabled = measure_cost(&enabled.output_text, &original, &corpus.measurement_mode)
                .unwrap_or_else(|e| panic!("{context}: recovery enabled: {e}"));
            eprintln!(
                "{} {mode:?}: disabled={disabled:?}, enabled={enabled:?}",
                fixture.name
            );
            assert_within_limits(
                &disabled,
                &case.disabled_limits,
                &format!("{context}, disabled"),
            );
            assert_within_limits(
                &enabled,
                &case.enabled_limits,
                &format!("{context}, enabled"),
            );
            results.insert(
                fixture.name.as_str(),
                RecoveryResult {
                    original,
                    optimized,
                    disabled,
                    enabled,
                },
            );
        }

        for case in &corpus.split_adder_cases {
            let Some(same_as) = &case.same_as else {
                // Only designated ordering variants need a peer comparison.
                continue;
            };
            let actual = &results[case.fixture.name.as_str()];
            let expected = &results[same_as.as_str()];
            assert!(functions_structurally_equivalent(
                &actual.original,
                &expected.original
            ));
            assert!(functions_structurally_equivalent(
                &actual.optimized,
                &expected.optimized
            ));
            for (actual, expected, limits) in [
                (&actual.disabled, &expected.disabled, &case.disabled_limits),
                (&actual.enabled, &expected.enabled, &case.enabled_limits),
            ] {
                assert_eq!(actual.and_nodes, expected.and_nodes);
                assert_eq!(actual.depth, expected.depth);
                assert!((actual.graph_le - expected.graph_le).abs() <= limits.graph_le_tolerance);
            }
        }
    }
}
