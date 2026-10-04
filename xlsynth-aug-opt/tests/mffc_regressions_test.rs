// SPDX-License-Identifier: Apache-2.0

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
use xlsynth_pir::ir_parser::Parser;

#[derive(Deserialize)]
struct Corpus {
    schema: String,
    measurement_mode: MeasurementMode,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct MeasurementMode {
    input: String,
    abc: bool,
    canonical_options: CanonicalG8rOptions,
    cut_db_rewrite_max_iterations: usize,
    cut_db_rewrite_max_cuts_per_node: usize,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    file: String,
    top: String,
    sha256: String,
    require_improvement: bool,
    regression_limits: Limits,
}

#[derive(Deserialize)]
struct Limits {
    and_nodes_max: usize,
    graph_le_max: f64,
    depth_max: usize,
    graph_le_tolerance: f64,
}

#[derive(Debug)]
struct MeasuredCost {
    and_nodes: usize,
    graph_le: f64,
    depth: usize,
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
    let graph_le_options = GraphLogicalEffortOptions {
        beta1: mode.canonical_options.graph_logical_effort_beta1,
        beta2: mode.canonical_options.graph_logical_effort_beta2,
    };

    for case in corpus.cases {
        let text = std::fs::read_to_string(root.join(&case.file)).expect("read fixture");
        assert_eq!(
            format!("{:x}", Sha256::digest(text.as_bytes())),
            case.sha256,
            "{}: fixture content changed",
            case.name,
        );
        let package = Parser::new(&text)
            .parse_and_validate_package()
            .expect("valid fixture IR");
        let original = package.get_fn(&case.top).expect("fixture top exists");
        let rewritten = run_aug_opt_over_ir_text_with_stats(
            &text,
            Some(&case.top),
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
        let mut options = mode.canonical_options.to_process_ir_path_options(
            Some(&case.top),
            true,
            false,
            false,
            None,
        );
        options.cut_db_rewrite_max_iterations = mode.cut_db_rewrite_max_iterations;
        options.cut_db_rewrite_max_cuts_per_node = mode.cut_db_rewrite_max_cuts_per_node;
        let [fixed, augmented] = [
            ("fixed IR", text.as_str()),
            ("aug-opt", rewritten.output_text.as_str()),
        ]
        .map(|(pipeline, input)| {
            let context = format!("{} ({pipeline})", case.name);
            let (gate_fn, _) = process_ir_text_with_gatefn(input, &options)
                .unwrap_or_else(|error| panic!("{context}: mapping failed: {error}"));
            validate_same_fn_via_toolchain(original, &gate_fn)
                .unwrap_or_else(|error| panic!("{context}: equivalence failed: {error}"));
            let stats = get_aig_stats(&gate_fn);
            MeasuredCost {
                and_nodes: stats.and_nodes,
                graph_le: analyze_graph_logical_effort(&gate_fn, &graph_le_options).delay,
                depth: stats.max_depth,
            }
        });

        let limits = &case.regression_limits;
        let tolerance = limits.graph_le_tolerance;
        assert!(tolerance.is_finite() && tolerance >= 0.0);
        eprintln!("{}: fixed={fixed:?}, aug-opt={augmented:?}", case.name);
        assert!(
            augmented.and_nodes <= limits.and_nodes_max
                && augmented.graph_le <= limits.graph_le_max + tolerance
                && augmented.depth <= limits.depth_max,
            "{}: exceeded recorded QoR limits: {augmented:?}",
            case.name,
        );
        assert!(
            augmented.and_nodes <= fixed.and_nodes
                && augmented.graph_le <= fixed.graph_le + tolerance,
            "{}: aug-opt regressed: fixed={fixed:?}, augmented={augmented:?}",
            case.name,
        );
        if case.require_improvement {
            assert!(
                augmented.and_nodes < fixed.and_nodes
                    || augmented.graph_le < fixed.graph_le - tolerance,
                "{}: expected aug-opt to improve: fixed={fixed:?}, augmented={augmented:?}",
                case.name,
            );
        }
    }
}
