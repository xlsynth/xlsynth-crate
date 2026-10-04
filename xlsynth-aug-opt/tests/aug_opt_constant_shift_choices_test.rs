// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};

use mffc_corpus::{Case, Expectations};
use serde_json::json;
use xlsynth_aug_opt::constant_shift_choices::{
    ConstantShiftChoiceLimits, ShiftChoiceCostGraph, constant_shift_choice_candidate,
    rewrite_constant_shift_choices_with_evaluator,
};
use xlsynth_aug_opt::cost::GateBuilderCostEvaluator;
use xlsynth_aug_opt::ir_cost::{IrCost, ShiftChoiceCostEvaluator};
use xlsynth_aug_opt::run_aug_opt_over_ir_text_with_stats;
use xlsynth_aug_opt::{AugOptMode, AugOptOptions};
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig::graph_logical_effort::{
    GraphLogicalEffortOptions, analyze_graph_logical_effort,
};
use xlsynth_g8r::check_equivalence::{
    check_equivalence_with_top_via_toolchain, validate_same_fn_via_toolchain,
};
use xlsynth_g8r::process_ir_path::process_ir_text_with_gatefn;
use xlsynth_pir::ir_eval::eval_fn;
use xlsynth_pir::ir_verify::verify_function;
use xlsynth_pir::{
    IrValue, ir,
    ir_parser::Parser,
    ir_utils::{self, fn_node_count},
};

const GRAPH_LE_TOLERANCE: f64 = 1.0e-6;

pub mod mffc_corpus;

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

/// Owns a compact copy of an evaluated region for subsequent equivalence
/// checks.
struct EvaluatedChoice {
    function: ir::Fn,
    cost: IrCost,
}

/// Records the region seen by the real g8r evaluator for semantic checks.
#[derive(Default)]
struct RecordingCostEvaluator {
    inner: GateBuilderCostEvaluator,
    evaluations: Vec<EvaluatedChoice>,
}

impl ShiftChoiceCostEvaluator for RecordingCostEvaluator {
    fn estimate(&mut self, graph: &ShiftChoiceCostGraph<'_>) -> Result<IrCost, String> {
        let cost = self.inner.estimate(graph)?;
        self.evaluations.push(EvaluatedChoice {
            function: materialize_cost_graph(graph),
            cost,
        });
        Ok(cost)
    }
}

/// Copies only a borrowed region into standalone IR for the equivalence tool.
fn materialize_cost_graph(graph: &ShiftChoiceCostGraph<'_>) -> ir::Fn {
    let mut function = ir::Fn {
        graph: ir::NodeGraph::new("shift_choice_cost"),
        params: Vec::new(),
        ret_ty: ir::Type::Bits(0),
        ret_node_ref: None,
    };
    let mut mapping = HashMap::new();
    for (index, &input) in graph.inputs().iter().enumerate() {
        let local = ir::NodeRef {
            index: function.nodes.len(),
        };
        function.nodes.push(ir::Node {
            text_id: local.index,
            name: Some(format!("input_{index}")),
            ty: graph.function().get_node(input).ty.clone(),
            payload: ir::NodePayload::Param,
            pos: None,
        });
        function.params.push(local);
        assert!(mapping.insert(input, local).is_none());
    }
    for &source in graph.nodes() {
        let local = ir::NodeRef {
            index: function.nodes.len(),
        };
        let source_node = graph.function().get_node(source);
        let payload = ir_utils::remap_payload_with(&source_node.payload, |(_, arg)| mapping[&arg]);
        function.nodes.push(ir::Node {
            text_id: local.index,
            name: None,
            ty: source_node.ty.clone(),
            payload,
            pos: None,
        });
        assert!(mapping.insert(source, local).is_none());
    }
    let outputs: Vec<_> = graph.outputs().iter().map(|nr| mapping[nr]).collect();
    let width = outputs
        .iter()
        .map(|nr| function.get_node(*nr).ty.bit_count())
        .sum();
    let result = ir::NodeRef {
        index: function.nodes.len(),
    };
    function.nodes.push(ir::Node {
        text_id: result.index,
        name: None,
        ty: ir::Type::Bits(width),
        payload: ir::NodePayload::Nary(ir::NaryOp::Concat, outputs),
        pos: None,
    });
    function.ret_ty = ir::Type::Bits(width);
    function.ret_node_ref = Some(result);
    verify_function(&function).unwrap();
    function
}

/// Proves each site's alternatives implement the same function of their cut.
fn prove_local_choices_equivalent(evaluator: &RecordingCostEvaluator) {
    assert!(!evaluator.evaluations.is_empty());
    let mut pairs = evaluator.evaluations.chunks_exact(2);
    for pair in &mut pairs {
        let original = &pair[0].function;
        let candidate = &pair[1].function;
        assert_eq!(original.name, candidate.name);
        check_equivalence_with_top_via_toolchain(
            &format!("package local_original\n\ntop {original}"),
            &format!("package local_candidate\n\ntop {candidate}"),
            Some(&original.name),
        )
        .unwrap();
    }
    assert!(pairs.remainder().is_empty());
}

#[derive(Default)]
struct EqualCostEvaluator {
    calls: usize,
}

impl ShiftChoiceCostEvaluator for EqualCostEvaluator {
    fn estimate(&mut self, _: &ShiftChoiceCostGraph<'_>) -> Result<IrCost, String> {
        self.calls += 1;
        Ok(IrCost {
            area: 10,
            delay: 5.0,
        })
    }
}

/// Exercises encoded shift choices and the fixed MFFC corpus with one profile.
fn fixtures(cases: Vec<Case>) -> Vec<Fixture> {
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
                text: masked_choice_text(width, op, false),
            });
        }
    }
    for case in cases {
        let Expectations::Shift {
            require_improvement,
            ..
        } = case.expectations
        else {
            // The add-recovery cases have a separate pipeline comparison.
            continue;
        };
        fixtures.push(Fixture {
            text: case.text,
            name: case.name,
            top: case.original.name.clone(),
            expect_constant_shift_candidate: require_improvement,
            expect_constant_shift_rewrite: require_improvement,
            require_no_aug_opt_regression: require_improvement,
            synthetic_qor_width: None,
        });
    }
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
                    ..Default::default()
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

/// Makes the shift amount optionally observable as a second output.
fn masked_choice_text(width: usize, op: &str, return_amount: bool) -> String {
    let return_type = if return_amount {
        format!("(bits[{width}], bits[2])")
    } else {
        format!("bits[{width}]")
    };
    let result = if return_amount {
        format!(
            "shifted: bits[{width}] = {op}(x, amount, id=11)\n  ret result: {return_type} = tuple(shifted, amount, id=12)"
        )
    } else {
        format!("ret result: bits[{width}] = {op}(x, amount, id=11)")
    };
    format!(
        r#"package synthetic_shift_choices

top fn main(x: bits[{width}] id=1, en: bits[1] id=2, p: bits[1] id=3, q: bits[1] id=4) -> {return_type} {{
  one_bit: bits[1] = literal(value=1, id=5)
  pair: bits[2] = concat(one_bit, q, id=6)
  one: bits[2] = literal(value=1, id=7)
  chosen: bits[2] = sel(p, cases=[pair, one], id=8)
  mask: bits[2] = sign_ext(en, new_bit_count=2, id=9)
  amount: bits[2] = and(chosen, mask, id=10)
  {result}
}}
"#,
    )
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

#[derive(Default)]
struct ComparisonSummary {
    relations: BTreeMap<&'static str, usize>,
    log_cost_ratio_sum: f64,
    count: usize,
}

/// Compares costed aug-opt and unconditional projection through normal mapping.
#[test]
fn compare_costed_aug_opt_shift_choice_pipelines() {
    let corpus = mffc_corpus::load().unwrap();
    let profile = &corpus.profile;
    let graph_le_options = GraphLogicalEffortOptions {
        beta1: profile.canonical_options.graph_logical_effort_beta1,
        beta2: profile.canonical_options.graph_logical_effort_beta2,
    };
    let mut summaries = BTreeMap::<&str, ComparisonSummary>::new();
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
            let transformed = transform(&fixture, pipeline);
            let transformed_fn = parse_function(&transformed.text, &fixture.top);
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
            let (gate_fn, _) = process_ir_text_with_gatefn(&transformed.text, &options)
                .unwrap_or_else(|error| panic!("{context}: mapping failed: {error}"));
            validate_same_fn_via_toolchain(&original, &gate_fn)
                .unwrap_or_else(|error| panic!("{context}: graph equivalence failed: {error}"));
            let stats = get_aig_stats(&gate_fn);
            let cost = MeasuredCost {
                and_nodes: stats.and_nodes,
                graph_le: analyze_graph_logical_effort(&gate_fn, &graph_le_options).delay,
                depth: stats.max_depth,
            };
            assert!(cost.graph_le.is_finite(), "{context}: non-finite cost");
            let baseline = *fixed_ir_cost.get_or_insert(cost);
            eprintln!(
                "shift_choice_comparison {}",
                json!({
                    "fixture": fixture.name,
                    "pipeline": pipeline.label(),
                    "ir_nodes": fn_node_count(&transformed_fn),
                    "total_rewrites": transformed.total_rewrites,
                    "constant_shift_rewrites": transformed.constant_shift_rewrites,
                    "projection_candidate": transformed.projection_candidate,
                    "and_nodes": cost.and_nodes,
                    "graph_le": cost.graph_le,
                    "depth": cost.depth,
                    "relation_to_fixed_ir": cost_relation(cost, baseline),
                })
            );
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
            let summary = summaries.entry(pipeline.label()).or_default();
            *summary
                .relations
                .entry(cost_relation(cost, baseline))
                .or_default() += 1;
            summary.log_cost_ratio_sum += ((cost.and_nodes as f64 * cost.graph_le)
                / (baseline.and_nodes as f64 * baseline.graph_le))
                .ln();
            summary.count += 1;
        }
    }
    for (pipeline, summary) in summaries {
        eprintln!(
            "shift_choice_summary {}",
            json!({
                "pipeline": pipeline,
                "baseline": "fixed_ir",
                "relations": summary.relations,
                "and_graph_le_geomean_ratio":
                    (summary.log_cost_ratio_sum / summary.count as f64).exp(),
            })
        );
    }
}

#[test]
fn profitability_rejects_area_tradeoffs_and_preserves_shared_amounts() {
    let limits = ConstantShiftChoiceLimits::default();
    let mut evaluator = GateBuilderCostEvaluator::default();
    for width in [4, 8, 16] {
        for op in ["shll", "shrl"] {
            for return_amount in [false, true] {
                let text = masked_choice_text(width, op, return_amount);
                let mut function = parse_function(&text, "main");
                let original = function.to_string();
                assert!(constant_shift_choice_candidate(&function, limits).is_some());
                let expected = usize::from(width == 4 && !return_amount);
                assert_eq!(
                    rewrite_constant_shift_choices_with_evaluator(
                        &mut function,
                        limits,
                        &mut evaluator,
                    )
                    .unwrap(),
                    expected,
                    "width={width}, op={op}, return_amount={return_amount}"
                );
                if expected == 0 {
                    assert_eq!(function.to_string(), original);
                } else {
                    check_equivalence_with_top_via_toolchain(
                        &text,
                        &format!("package rewritten\n\ntop {function}"),
                        Some("main"),
                    )
                    .unwrap();
                }
                assert_eq!(
                    rewrite_constant_shift_choices_with_evaluator(
                        &mut function,
                        limits,
                        &mut evaluator,
                    ),
                    Ok(0)
                );
            }
        }
    }
}

#[test]
fn profitability_preserves_input_on_equal_cost() {
    let limits = ConstantShiftChoiceLimits::default();
    let mut evaluator = EqualCostEvaluator::default();
    let mut function = parse_function(&masked_choice_text(4, "shll", false), "main");
    let original = function.to_string();
    assert!(constant_shift_choice_candidate(&function, limits).is_some());
    assert_eq!(
        rewrite_constant_shift_choices_with_evaluator(&mut function, limits, &mut evaluator),
        Ok(0)
    );
    assert_eq!(evaluator.calls, 2);
    assert_eq!(function.to_string(), original);
}

#[test]
fn rejected_local_choice_does_not_affect_the_next_site() {
    let text = r#"package independent_choices

top fn main(x: bits[8] id=1, y: bits[4] id=2, en: bits[1] id=3, p: bits[1] id=4, q: bits[1] id=5) -> (bits[8], bits[4]) {
  a_one_bit: bits[1] = literal(value=1, id=6)
  a_pair: bits[2] = concat(a_one_bit, q, id=7)
  a_one: bits[2] = literal(value=1, id=8)
  a_chosen: bits[2] = sel(p, cases=[a_pair, a_one], id=9)
  a_mask: bits[2] = sign_ext(en, new_bit_count=2, id=10)
  a_amount: bits[2] = and(a_chosen, a_mask, id=11)
  a_shifted: bits[8] = shll(x, a_amount, id=12)
  b_one_bit: bits[1] = literal(value=1, id=13)
  b_pair: bits[2] = concat(b_one_bit, q, id=14)
  b_one: bits[2] = literal(value=1, id=15)
  b_chosen: bits[2] = sel(p, cases=[b_pair, b_one], id=16)
  b_mask: bits[2] = sign_ext(en, new_bit_count=2, id=17)
  b_amount: bits[2] = and(b_chosen, b_mask, id=18)
  b_shifted: bits[4] = shll(y, b_amount, id=19)
  ret result: (bits[8], bits[4]) = tuple(a_shifted, b_shifted, id=20)
}
"#;
    let mut function = parse_function(text, "main");
    let mut evaluator = RecordingCostEvaluator::default();
    assert_eq!(
        rewrite_constant_shift_choices_with_evaluator(
            &mut function,
            ConstantShiftChoiceLimits::default(),
            &mut evaluator,
        ),
        Ok(1)
    );
    let [first, first_candidate, second, second_candidate] = evaluator.evaluations.as_slice()
    else {
        panic!("both independent sites must be costed");
    };
    assert_eq!(first.function.ret_ty, ir::Type::Bits(8));
    assert_eq!(second.function.ret_ty, ir::Type::Bits(4));
    assert!(!first_candidate.cost.is_pareto_improvement_on(first.cost));
    assert!(second_candidate.cost.is_pareto_improvement_on(second.cost));
    prove_local_choices_equivalent(&evaluator);
    check_equivalence_with_top_via_toolchain(
        text,
        &format!("package rewritten\n\ntop {function}"),
        Some("main"),
    )
    .unwrap();
    verify_function(&function).unwrap();
}

#[test]
fn local_costs_are_independent_of_node_storage_order() {
    let text = masked_choice_text(4, "shll", false);
    let original = parse_function(&text, "main");
    let mut reversed = original.clone();
    let node_count = reversed.nodes.len();
    let remap = |node: ir::NodeRef| ir::NodeRef {
        index: if node.index == 0 {
            0
        } else {
            node_count - node.index
        },
    };
    // Model an intermediate working graph before final compaction: appended
    // replacements can leave forward references and sparse parameter indices.
    reversed.nodes[1..].reverse();
    for node in &mut reversed.nodes[1..] {
        node.payload = ir_utils::remap_payload_with(&node.payload, |(_, arg)| remap(arg));
    }
    reversed.params = reversed.params.iter().copied().map(remap).collect();
    reversed.ret_node_ref = reversed.ret_node_ref.map(remap);
    ir_utils::verify_no_cycle(&reversed).unwrap();

    let mut costs = Vec::new();
    for mut function in [original, reversed] {
        let mut evaluator = RecordingCostEvaluator::default();
        assert_eq!(
            rewrite_constant_shift_choices_with_evaluator(
                &mut function,
                ConstantShiftChoiceLimits::default(),
                &mut evaluator,
            ),
            Ok(1)
        );
        prove_local_choices_equivalent(&evaluator);
        check_equivalence_with_top_via_toolchain(
            &text,
            &format!("package rewritten\n\ntop {function}"),
            Some("main"),
        )
        .unwrap();
        costs.push(evaluator.evaluations);
    }
    for (original, reversed) in costs[0].iter().zip(&costs[1]) {
        assert_eq!(original.cost.area, reversed.cost.area);
        assert!((original.cost.delay - reversed.cost.delay).abs() <= GRAPH_LE_TOLERANCE);
    }
}

#[test]
fn profitable_local_choice_is_independent_of_unrelated_arithmetic() {
    let isolated = masked_choice_text(4, "shll", false);
    let mut arithmetic = String::new();
    let mut previous = "a".to_string();
    for index in 0..64 {
        let next = format!("sum_{index}");
        arithmetic.push_str(&format!(
            "  {next}: bits[256] = add({previous}, b, id={})\n",
            index + 14,
        ));
        previous = next;
    }
    // The shifted data also has an arithmetic producer. Its boundary binding
    // must hide that payload as well as the unrelated wide output's logic.
    let embedded = format!(
        r#"package embedded_shift_choices

top fn main(x: bits[4] id=1, en: bits[1] id=2, p: bits[1] id=3, q: bits[1] id=4, a: bits[256] id=12, b: bits[256] id=13, bias: bits[4] id=79) -> (bits[4], bits[256]) {{
  data: bits[4] = add(x, bias, id=80)
  one_bit: bits[1] = literal(value=1, id=5)
  pair: bits[2] = concat(one_bit, q, id=6)
  one: bits[2] = literal(value=1, id=7)
  chosen: bits[2] = sel(p, cases=[pair, one], id=8)
  mask: bits[2] = sign_ext(en, new_bit_count=2, id=9)
  amount: bits[2] = and(chosen, mask, id=10)
  shifted: bits[4] = shll(data, amount, id=11)
{arithmetic}  ret result: (bits[4], bits[256]) = tuple(shifted, {previous}, id=78)
}}
"#,
    );
    let mut evaluations = Vec::new();
    for text in [&isolated, &embedded] {
        let mut function = parse_function(text, "main");
        let mut evaluator = RecordingCostEvaluator::default();
        assert_eq!(
            rewrite_constant_shift_choices_with_evaluator(
                &mut function,
                ConstantShiftChoiceLimits::default(),
                &mut evaluator,
            ),
            Ok(1)
        );
        prove_local_choices_equivalent(&evaluator);
        check_equivalence_with_top_via_toolchain(
            text,
            &format!("package rewritten\n\ntop {function}"),
            Some("main"),
        )
        .unwrap();
        evaluations.push(evaluator.evaluations);
    }
    for (isolated, embedded) in evaluations[0].iter().zip(&evaluations[1]) {
        assert_eq!(isolated.cost, embedded.cost);
        assert_eq!(isolated.function.nodes.len(), embedded.function.nodes.len());
        assert!(
            embedded
                .function
                .nodes
                .iter()
                .all(|node| !matches!(node.payload, ir::NodePayload::Binop(ir::Binop::Add, _, _)))
        );
    }
}

#[test]
fn local_cost_graphs_preserve_shared_amount_output() {
    let fixture = context_fixtures()
        .into_iter()
        .find(|fixture| fixture.name == "shared_amount_and_data_shll_4")
        .unwrap();
    let mut function = parse_function(&fixture.text, "main");
    let mut evaluator = RecordingCostEvaluator::default();
    rewrite_constant_shift_choices_with_evaluator(
        &mut function,
        ConstantShiftChoiceLimits::default(),
        &mut evaluator,
    )
    .unwrap();
    prove_local_choices_equivalent(&evaluator);
    for evaluation in &evaluator.evaluations {
        // Four result bits plus the two amount bits needed by its other user.
        assert_eq!(evaluation.function.ret_ty, ir::Type::Bits(6));
    }
}

#[test]
fn aug_opt_fuses_constant_choices_and_reaches_fixed_point() {
    for op in ["shll", "shrl"] {
        let text = masked_choice_text(4, op, false);
        let mut first_output = None;
        for rounds in [1, 3] {
            let result = run_aug_opt_over_ir_text_with_stats(
                &text,
                Some("main"),
                AugOptOptions {
                    enable: true,
                    rounds,
                    mode: AugOptMode::PirOnly,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(result.rewrite_stats.constant_shift_choices, 1);
            assert_eq!(result.total_rewrites, 1);
            check_equivalence_with_top_via_toolchain(&text, &result.output_text, Some("main"))
                .unwrap();
            if let Some(first) = &first_output {
                assert_eq!(&result.output_text, first);
            } else {
                first_output = Some(result.output_text);
            }
        }
    }
}

#[test]
fn aug_opt_constant_choices_preserve_package_calls_and_ids() {
    for data_op in ["identity(x, id=11)", "invoke(x, to_apply=helper, id=11)"] {
        let text = format!(
            r#"package package_choices

fn helper(x: bits[4] id=13) -> bits[4] {{
  ret result: bits[4] = not(x, id=14)
}}

top fn main(x: bits[4] id=1, en: bits[1] id=2, p: bits[1] id=3, q: bits[1] id=4) -> bits[4] {{
  one_bit: bits[1] = literal(value=1, id=5)
  pair: bits[2] = concat(one_bit, q, id=6)
  one: bits[2] = literal(value=1, id=7)
  chosen: bits[2] = sel(p, cases=[pair, one], id=8)
  mask: bits[2] = sign_ext(en, new_bit_count=2, id=9)
  amount: bits[2] = and(chosen, mask, id=10)
  data: bits[4] = {data_op}
  ret result: bits[4] = shll(data, amount, id=12)
}}
"#,
        );
        let mut first_output = None;
        for rounds in [1, 3] {
            let result = run_aug_opt_over_ir_text_with_stats(
                &text,
                Some("main"),
                AugOptOptions {
                    enable: true,
                    rounds,
                    mode: AugOptMode::PirOnly,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(result.rewrite_stats.constant_shift_choices, 1);
            assert_eq!(
                parse_function(&text, "helper").to_string(),
                parse_function(&result.output_text, "helper").to_string(),
            );
            // libxls also enforces uniqueness of node IDs across the package.
            xlsynth::IrPackage::parse_ir(&result.output_text, None).unwrap();
            check_equivalence_with_top_via_toolchain(&text, &result.output_text, Some("main"))
                .unwrap();
            if let Some(first) = &first_output {
                assert_eq!(&result.output_text, first);
            } else {
                first_output = Some(result.output_text);
            }
        }
    }
}

#[test]
fn aug_opt_does_not_expand_choices_with_variable_leaves() {
    for op in ["shll", "shrl"] {
        let text = format!(
            r#"package variable_choices

top fn main(x: bits[4] id=1, p: bits[1] id=2, amount: bits[65] id=3) -> bits[4] {{
  one: bits[65] = literal(value=1, id=4)
  chosen: bits[65] = sel(p, cases=[one, amount], id=5)
  ret result: bits[4] = {op}(x, chosen, id=6)
}}
"#,
        );
        let result = run_aug_opt_over_ir_text_with_stats(
            &text,
            Some("main"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                mode: AugOptMode::PirOnly,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.rewrite_stats.constant_shift_choices, 0);
        assert_eq!(
            parse_function(&result.output_text, "main").to_string(),
            parse_function(&text, "main").to_string(),
        );
    }
}

#[test]
fn profitable_rewrite_preserves_effects_and_shift_used_only_by_trace() {
    let original = parse_function(
        r#"package effects

top fn main(x: bits[4] id=1, selector: bits[1] id=2, enabled: bits[1] id=3) -> bits[1] {
  one: bits[2] = literal(value=1, id=4)
  two: bits[2] = literal(value=2, id=5)
  amount: bits[2] = sel(selector, cases=[one, two], id=6)
  shifted: bits[4] = shll(x, amount, id=7)
  tok: token = after_all(id=8)
  traced: token = trace(tok, enabled, format="shifted={}", data_operands=[shifted], verbosity=0, id=9)
  covered: () = cover(enabled, label="enabled", id=10)
  checked: token = assert(traced, enabled, message="disabled", label="check", id=11)
  ret out: bits[1] = literal(value=0, id=12)
}
"#,
        "main",
    );
    let mut rewritten = original.clone();
    assert_eq!(
        rewrite_constant_shift_choices_with_evaluator(
            &mut rewritten,
            ConstantShiftChoiceLimits::default(),
            &mut GateBuilderCostEvaluator::default(),
        ),
        Ok(1)
    );
    verify_function(&rewritten).unwrap();
    for x in 0..16 {
        for selector in 0..2 {
            for enabled in 0..2 {
                let args = [
                    IrValue::make_ubits(4, x).unwrap(),
                    IrValue::make_ubits(1, selector).unwrap(),
                    IrValue::make_ubits(1, enabled).unwrap(),
                ];
                // Include trace data, coverage, and assertion failures.
                assert_eq!(eval_fn(&rewritten, &args), eval_fn(&original, &args));
            }
        }
    }
}
