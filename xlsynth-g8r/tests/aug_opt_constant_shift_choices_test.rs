// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use serde::Deserialize;
use serde_json::json;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig::graph_logical_effort::{
    GraphLogicalEffortOptions, analyze_graph_logical_effort,
};
use xlsynth_g8r::check_equivalence::{
    check_equivalence_with_top_via_toolchain, validate_same_fn_via_toolchain,
};
use xlsynth_g8r::process_ir_path::{CanonicalG8rOptions, process_ir_text_with_gatefn};
use xlsynth_pir::aug_opt::{AugOptMode, AugOptOptions, run_aug_opt_over_ir_text_with_stats};
use xlsynth_pir::constant_shift_choices::{
    ConstantShiftChoiceLimits, constant_shift_choice_candidate,
};
use xlsynth_pir::local_cost::estimate_local_cost;
use xlsynth_pir::{ir, ir_parser::Parser, ir_utils::fn_node_count};

const GRAPH_LE_TOLERANCE: f64 = 1.0e-6;

#[derive(Deserialize)]
struct Corpus {
    measurement_mode: MappingProfile,
    cases: Vec<CorpusCase>,
}

#[derive(Deserialize)]
struct MappingProfile {
    canonical_options: CanonicalG8rOptions,
    cut_db_rewrite_max_iterations: usize,
    cut_db_rewrite_max_cuts_per_node: usize,
}

#[derive(Deserialize)]
struct CorpusCase {
    name: String,
    file: String,
    top: String,
    require_improvement: bool,
}

struct Fixture {
    name: String,
    text: String,
    top: String,
    expect_constant_shift_candidate: bool,
    expect_constant_shift_rewrite: bool,
    require_no_aug_opt_regression: bool,
    synthetic_qor_width: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct MeasuredCost {
    and_nodes: usize,
    graph_le: f64,
    depth: usize,
}

#[derive(Clone, Copy)]
enum Pipeline {
    FixedIr,
    ProjectionChoices,
    Libxls,
    AugOptPirOnly,
    AugOptSandwich,
}

impl Pipeline {
    fn label(self) -> &'static str {
        match self {
            Self::FixedIr => "fixed_ir",
            Self::ProjectionChoices => "projection_choices",
            Self::Libxls => "libxls",
            Self::AugOptPirOnly => "aug_opt_pir_only",
            Self::AugOptSandwich => "aug_opt_sandwich",
        }
    }
}

struct TransformedInput {
    text: String,
    total_rewrites: usize,
    constant_shift_rewrites: usize,
    projection_candidate: bool,
}

/// Exercises encoded shift choices and the fixed MFFC corpus with one profile.
fn fixtures(cases: Vec<CorpusCase>) -> Vec<Fixture> {
    let mut fixtures = Vec::new();
    for op in ["shll", "shrl"] {
        for width in [2, 3, 4, 5, 6, 8, 12, 16, 32] {
            fixtures.push(Fixture {
                name: format!("synthetic_{op}_{width}"),
                top: "main".to_string(),
                expect_constant_shift_candidate: true,
                expect_constant_shift_rewrite: width == 4,
                require_no_aug_opt_regression: matches!(width, 4 | 8 | 16),
                synthetic_qor_width: matches!(width, 4 | 8 | 16).then_some(width),
                text: format!(
                    r#"package synthetic_shift_choices

top fn main(x: bits[{width}] id=1, en: bits[1] id=2, p: bits[1] id=3, q: bits[1] id=4) -> bits[{width}] {{
  one_bit: bits[1] = literal(value=1, id=5)
  pair: bits[2] = concat(one_bit, q, id=6)
  one: bits[2] = literal(value=1, id=7)
  chosen: bits[2] = sel(p, cases=[pair, one], id=8)
  mask: bits[2] = sign_ext(en, new_bit_count=2, id=9)
  amount: bits[2] = and(chosen, mask, id=10)
  ret result: bits[{width}] = {op}(x, amount, id=11)
}}
"#,
                ),
            });
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mffc_regressions");
    fixtures.extend(cases.into_iter().map(|case| Fixture {
        text: std::fs::read_to_string(root.join(&case.file)).expect("read MFFC fixture"),
        name: case.name,
        top: case.top,
        expect_constant_shift_candidate: case.require_improvement,
        expect_constant_shift_rewrite: case.require_improvement,
        require_no_aug_opt_regression: case.require_improvement,
        synthetic_qor_width: None,
    }));
    fixtures.extend(context_fixtures());
    fixtures
}

/// Covers regular barrel choices and retained amount/data users at two widths.
fn context_fixtures() -> Vec<Fixture> {
    let mut fixtures = Vec::new();
    for (width, op) in [(4, "shll"), (8, "shrl")] {
        fixtures.push(Fixture {
            name: format!("natural_barrel_choices_{op}_{width}"),
            top: "main".to_string(),
            expect_constant_shift_candidate: true,
            expect_constant_shift_rewrite: false,
            require_no_aug_opt_regression: false,
            synthetic_qor_width: None,
            text: format!(
                r#"package natural_barrel_choices

top fn main(x: bits[{width}] id=1, p: bits[1] id=2, q: bits[1] id=3) -> bits[{width}] {{
  zero: bits[2] = literal(value=0, id=4)
  one: bits[2] = literal(value=1, id=5)
  two: bits[2] = literal(value=2, id=6)
  three: bits[2] = literal(value=3, id=7)
  low: bits[2] = sel(q, cases=[zero, one], id=8)
  high: bits[2] = sel(q, cases=[two, three], id=9)
  amount: bits[2] = sel(p, cases=[low, high], id=10)
  ret result: bits[{width}] = {op}(x, amount, id=11)
}}
"#,
            ),
        });
        fixtures.push(Fixture {
            name: format!("shared_amount_and_data_{op}_{width}"),
            top: "main".to_string(),
            expect_constant_shift_candidate: true,
            expect_constant_shift_rewrite: false,
            require_no_aug_opt_regression: false,
            synthetic_qor_width: None,
            text: format!(
                r#"package shared_amount_and_data

top fn main(a: bits[{width}] id=1, b: bits[{width}] id=2, en: bits[1] id=3, p: bits[1] id=4, q: bits[1] id=5) -> (bits[{width}], bits[2], bits[{width}]) {{
  x: bits[{width}] = xor(a, b, id=6)
  one_bit: bits[1] = literal(value=1, id=7)
  pair: bits[2] = concat(one_bit, q, id=8)
  one: bits[2] = literal(value=1, id=9)
  chosen: bits[2] = sel(p, cases=[pair, one], id=10)
  mask: bits[2] = sign_ext(en, new_bit_count=2, id=11)
  amount: bits[2] = and(chosen, mask, id=12)
  shifted: bits[{width}] = {op}(x, amount, id=13)
  other: bits[{width}] = and(x, a, id=14)
  ret result: (bits[{width}], bits[2], bits[{width}]) = tuple(shifted, amount, other, id=15)
}}
"#,
            ),
        });
    }
    fixtures
}

/// Runs the public optimizer entry points, including the actual opt sandwich.
fn transform(fixture: &Fixture, pipeline: Pipeline) -> TransformedInput {
    match pipeline {
        Pipeline::FixedIr => TransformedInput {
            text: fixture.text.clone(),
            total_rewrites: 0,
            constant_shift_rewrites: 0,
            projection_candidate: false,
        },
        Pipeline::ProjectionChoices => {
            let function = parse_function(&fixture.text, &fixture.top);
            let candidate =
                constant_shift_choice_candidate(&function, ConstantShiftChoiceLimits::default());
            TransformedInput {
                projection_candidate: candidate.is_some(),
                text: candidate.map_or_else(
                    || fixture.text.clone(),
                    |function| format!("package projection_choices\n\ntop {function}"),
                ),
                // These counters describe aug-opt passes, not this reference.
                total_rewrites: 0,
                constant_shift_rewrites: 0,
            }
        }
        Pipeline::Libxls => {
            let package = xlsynth::IrPackage::parse_ir(&fixture.text, None)
                .expect("libxls parses comparison fixture");
            TransformedInput {
                text: xlsynth::optimize_ir(&package, &fixture.top)
                    .expect("standard libxls optimization")
                    .to_string(),
                total_rewrites: 0,
                constant_shift_rewrites: 0,
                projection_candidate: false,
            }
        }
        Pipeline::AugOptPirOnly | Pipeline::AugOptSandwich => {
            let result = run_aug_opt_over_ir_text_with_stats(
                &fixture.text,
                Some(&fixture.top),
                AugOptOptions {
                    enable: true,
                    rounds: 1,
                    mode: if matches!(pipeline, Pipeline::AugOptPirOnly) {
                        AugOptMode::PirOnly
                    } else {
                        AugOptMode::Sandwich
                    },
                },
            )
            .expect("aug-opt comparison pipeline succeeds");
            TransformedInput {
                text: result.output_text,
                total_rewrites: result.total_rewrites,
                constant_shift_rewrites: result.rewrite_stats.constant_shift_choices,
                projection_candidate: false,
            }
        }
    }
}

fn parse_function(text: &str, top: &str) -> ir::Fn {
    Parser::new(text)
        .parse_and_validate_package()
        .expect("valid comparison IR")
        .get_fn(top)
        .expect("comparison top exists")
        .clone()
}

/// Classifies both objectives; graph depth is diagnostic, not an objective.
fn cost_relation(cost: MeasuredCost, baseline: MeasuredCost) -> &'static str {
    let better = cost.and_nodes < baseline.and_nodes
        || cost.graph_le < baseline.graph_le - GRAPH_LE_TOLERANCE;
    let worse = cost.and_nodes > baseline.and_nodes
        || cost.graph_le > baseline.graph_le + GRAPH_LE_TOLERANCE;
    match (better, worse) {
        (true, false) => "improved",
        (false, false) => "unchanged",
        (true, true) => "tradeoff",
        (false, true) => "regressed",
    }
}

/// Retains absolute limits so both the baseline and aug-opt cannot regress
/// together.
fn assert_quality_envelope(width: usize, cost: MeasuredCost) {
    let within = match width {
        4 => {
            cost.depth <= 4
                && ((cost.and_nodes <= 21 && cost.graph_le <= 18.316461 + GRAPH_LE_TOLERANCE)
                    || (cost.and_nodes <= 20 && cost.graph_le <= 18.985425 + GRAPH_LE_TOLERANCE))
        }
        8 => cost.and_nodes <= 45 && cost.graph_le <= 24.912359 + GRAPH_LE_TOLERANCE,
        16 => cost.and_nodes <= 93 && cost.graph_le <= 27.367808 + GRAPH_LE_TOLERANCE,
        _ => panic!("no recorded quality envelope for width {width}"),
    };
    assert!(
        within,
        "width {width} escaped the final-quality envelope: {cost:?}"
    );
}

/// Compares costed aug-opt and unconditional projection through normal mapping.
#[test]
fn compare_costed_aug_opt_shift_choice_pipelines() {
    let corpus: Corpus =
        serde_json::from_str(include_str!("fixtures/mffc_regressions/manifest.json"))
            .expect("valid MFFC manifest");
    let profile = &corpus.measurement_mode;
    let graph_le_options = GraphLogicalEffortOptions {
        beta1: profile.canonical_options.graph_logical_effort_beta1,
        beta2: profile.canonical_options.graph_logical_effort_beta2,
    };
    let mut summaries = BTreeMap::<&str, BTreeMap<&str, usize>>::new();
    for fixture in fixtures(corpus.cases) {
        let original = parse_function(&fixture.text, &fixture.top);
        let mut fixed_ir_cost = None;
        for pipeline in [
            Pipeline::FixedIr,
            Pipeline::ProjectionChoices,
            Pipeline::Libxls,
            Pipeline::AugOptPirOnly,
            Pipeline::AugOptSandwich,
        ] {
            let context = format!("{} / {}", fixture.name, pipeline.label());
            let start = Instant::now();
            let transformed = transform(&fixture, pipeline);
            let transform_us = start.elapsed().as_micros();
            let transformed_fn = parse_function(&transformed.text, &fixture.top);
            let start = Instant::now();
            let local_cost = estimate_local_cost(&transformed_fn);
            let local_cost_us = start.elapsed().as_micros();
            if !matches!(pipeline, Pipeline::FixedIr) {
                check_equivalence_with_top_via_toolchain(
                    &fixture.text,
                    &transformed.text,
                    Some(&fixture.top),
                )
                .unwrap_or_else(|error| panic!("{context}: IR equivalence failed: {error}"));
            }
            if matches!(pipeline, Pipeline::AugOptPirOnly) && fixture.expect_constant_shift_rewrite
            {
                assert!(
                    transformed.constant_shift_rewrites > 0,
                    "{context}: constant-shift rewrite did not run"
                );
            }
            if matches!(pipeline, Pipeline::ProjectionChoices) {
                assert_eq!(
                    transformed.projection_candidate, fixture.expect_constant_shift_candidate,
                    "{context}: unexpected constant-shift candidate eligibility"
                );
            }
            let mut options = profile.canonical_options.to_process_ir_path_options(
                Some(&fixture.top),
                true,
                false,
                false,
                None,
            );
            options.cut_db_rewrite_max_iterations = profile.cut_db_rewrite_max_iterations;
            options.cut_db_rewrite_max_cuts_per_node = profile.cut_db_rewrite_max_cuts_per_node;
            let start = Instant::now();
            let (gate_fn, _) = process_ir_text_with_gatefn(&transformed.text, &options)
                .unwrap_or_else(|error| panic!("{context}: mapping failed: {error}"));
            let map_us = start.elapsed().as_micros();
            validate_same_fn_via_toolchain(&original, &gate_fn)
                .unwrap_or_else(|error| panic!("{context}: graph equivalence failed: {error}"));
            let stats = get_aig_stats(&gate_fn);
            let cost = MeasuredCost {
                and_nodes: stats.and_nodes,
                graph_le: analyze_graph_logical_effort(&gate_fn, &graph_le_options).delay,
                depth: stats.max_depth,
            };
            assert!(cost.graph_le.is_finite(), "{context}: non-finite cost");
            // Timings are diagnostic metadata and never regression criteria.
            eprintln!(
                "shift_choice_comparison {}",
                json!({
                    "fixture": fixture.name,
                    "pipeline": pipeline.label(),
                    "ir_nodes": fn_node_count(&transformed_fn),
                    "total_rewrites": transformed.total_rewrites,
                    "constant_shift_rewrites": transformed.constant_shift_rewrites,
                    "projection_candidate": transformed.projection_candidate,
                    "ir_and_equivalents": local_cost.as_ref().map(|cost| cost.and_equivalents),
                    "ir_logic_depth": local_cost.as_ref().map(|cost| cost.logic_depth),
                    "local_cost_us": local_cost_us,
                    "and_nodes": cost.and_nodes,
                    "graph_le": cost.graph_le,
                    "depth": cost.depth,
                    "transform_us": transform_us,
                    "map_us": map_us,
                })
            );
            let baseline = *fixed_ir_cost.get_or_insert(cost);
            if matches!(pipeline, Pipeline::AugOptPirOnly | Pipeline::AugOptSandwich) {
                if fixture.require_no_aug_opt_regression {
                    assert!(
                        cost.and_nodes <= baseline.and_nodes
                            && cost.graph_le <= baseline.graph_le + GRAPH_LE_TOLERANCE,
                        "{context}: aug-opt regressed: fixed={baseline:?}, augmented={cost:?}"
                    );
                }
                if let Some(width) = fixture.synthetic_qor_width {
                    assert_quality_envelope(width, cost);
                }
            }
            *summaries
                .entry(pipeline.label())
                .or_default()
                .entry(cost_relation(cost, baseline))
                .or_default() += 1;
        }
    }
    for (pipeline, relations) in summaries {
        eprintln!(
            "shift_choice_summary {}",
            json!({
                "pipeline": pipeline,
                "baseline": "fixed_ir",
                "relations": relations,
            })
        );
    }
}
