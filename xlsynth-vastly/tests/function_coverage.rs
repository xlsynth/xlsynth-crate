// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use xlsynth_vastly::CompiledPipelineModule;
use xlsynth_vastly::CoverageCounters;
use xlsynth_vastly::Env;
use xlsynth_vastly::LogicBit;
use xlsynth_vastly::PipelineCycle;
use xlsynth_vastly::PipelineStimulus;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::SourceText;
use xlsynth_vastly::Value4;
use xlsynth_vastly::compile_pipeline_module;
use xlsynth_vastly::eval_combo_seeded;
use xlsynth_vastly::eval_combo_seeded_with_coverage;
use xlsynth_vastly::plan_combo_eval;
use xlsynth_vastly::render_annotated_source;
use xlsynth_vastly::run_pipeline_and_collect_coverage;
use xlsynth_vastly::run_pipeline_and_collect_outputs;

fn vbits(width: u32, signedness: Signedness, msb: &str) -> Value4 {
    assert_eq!(msb.len(), width as usize);
    let mut bits = Vec::with_capacity(width as usize);
    for c in msb.chars().rev() {
        bits.push(match c {
            '0' => LogicBit::Zero,
            '1' => LogicBit::One,
            'x' | 'X' => LogicBit::X,
            'z' | 'Z' => LogicBit::Z,
            _ => panic!("bad bit char {c}"),
        });
    }
    Value4::new(width, signedness, bits)
}

fn output_snapshot(
    m: &CompiledPipelineModule,
    values: &BTreeMap<String, Value4>,
) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for p in &m.combo.output_ports {
        out.insert(
            p.name.clone(),
            values.get(&p.name).unwrap().to_bit_string_msb_first(),
        );
    }
    out
}

fn build_seed_env(
    m: &CompiledPipelineModule,
    inputs: &BTreeMap<String, Value4>,
    clk_msb: &str,
) -> Env {
    let mut seed = Env::new();
    seed.insert(m.clk_name.clone(), vbits(1, Signedness::Unsigned, clk_msb));
    for (k, v) in inputs {
        seed.insert(k.clone(), v.clone());
    }
    seed
}

fn eval_step_and_compare(
    m: &CompiledPipelineModule,
    plan: &xlsynth_vastly::ComboEvalPlan,
    inputs: &BTreeMap<String, Value4>,
    clk_msb: &str,
    src: &SourceText,
    cov: &mut CoverageCounters,
) -> BTreeMap<String, String> {
    let seed = build_seed_env(m, inputs, clk_msb);
    let plain = eval_combo_seeded(&m.combo, plan, &seed).unwrap();
    let covered =
        eval_combo_seeded_with_coverage(&m.combo, plan, &seed, src, cov, &m.fn_meta).unwrap();
    let plain_snapshot = output_snapshot(m, &plain);
    let covered_snapshot = output_snapshot(m, &covered);
    assert_eq!(covered_snapshot, plain_snapshot);
    covered_snapshot
}

fn run_combinational_pipeline_with_output_equivalence(
    m: &CompiledPipelineModule,
    stimulus: &PipelineStimulus,
    src: &SourceText,
) -> (Vec<BTreeMap<String, String>>, CoverageCounters) {
    assert!(
        m.seqs.is_empty(),
        "helper only supports combinational pipeline modules"
    );
    let plan = plan_combo_eval(&m.combo).unwrap();
    let mut cov = CoverageCounters::default();
    cov.register_functions(&m.fn_meta);
    let mut coverage_steps: Vec<BTreeMap<String, String>> = Vec::new();
    let mut cycle_low_outputs: Vec<BTreeMap<String, String>> = Vec::new();

    let c0 = stimulus.cycles.first().expect("stimulus must be non-empty");
    coverage_steps.push(eval_step_and_compare(
        m, &plan, &c0.inputs, "0", src, &mut cov,
    ));

    for (cyc_idx, cyc) in stimulus.cycles.iter().enumerate() {
        if cyc_idx != 0 {
            coverage_steps.push(eval_step_and_compare(
                m,
                &plan,
                &cyc.inputs,
                "0",
                src,
                &mut cov,
            ));
        }
        coverage_steps.push(eval_step_and_compare(
            m,
            &plan,
            &cyc.inputs,
            "1",
            src,
            &mut cov,
        ));
        let low = eval_step_and_compare(m, &plan, &cyc.inputs, "0", src, &mut cov);
        cycle_low_outputs.push(low.clone());
        coverage_steps.push(low);
    }

    let init = m.initial_state_x();
    let harness_low_outputs: Vec<BTreeMap<String, String>> =
        run_pipeline_and_collect_outputs(m, stimulus, &init)
            .unwrap()
            .into_iter()
            .map(|values| output_snapshot(m, &values))
            .collect();
    assert_eq!(cycle_low_outputs, harness_low_outputs);

    (coverage_steps, cov)
}

fn build_function_expr_ternary_module(lhs_decl: &str, op: &str, rhs: &str) -> String {
    format!(
        "module m(input logic clk, input {lhs_decl} sel, input logic a, input logic b, output logic y);\n  function automatic logic pick(input {lhs_decl} s, input logic t, input logic f);\n    begin\n      pick = ((s {op} {rhs}) ? t : f);\n    end\n  endfunction\n  assign y = pick(sel, a, b);\nendmodule\n"
    )
}

fn make_sel_cycle(sel: Value4, a_msb: &str, b_msb: &str) -> PipelineCycle {
    PipelineCycle {
        inputs: [
            ("sel".to_string(), sel),
            ("a".to_string(), vbits(1, Signedness::Unsigned, a_msb)),
            ("b".to_string(), vbits(1, Signedness::Unsigned, b_msb)),
        ]
        .into_iter()
        .collect(),
    }
}

#[test]
fn function_casez_arm_coverage_is_aggregated_at_definition() {
    let dut = concat!(
        "module m(input logic clk, input logic [1:0] sel, output logic y);\n",
        "  function automatic logic f (input reg [1:0] sel, input reg case0, input reg case1, input reg def);\n",
        "    begin\n",
        "      unique casez (sel)\n",
        "        2'b?1: begin f = case0; end\n",
        "        2'b10: begin f = case1; end\n",
        "        default: begin f = def; end\n",
        "      endcase\n",
        "    end\n",
        "  endfunction\n",
        "  assign y = f(sel, 1'b0, 1'b1, 1'b0);\n",
        "endmodule\n",
    );
    let cm = compile_pipeline_module(dut).unwrap();
    let src = SourceText::new(dut.to_string());

    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![
            PipelineCycle {
                inputs: [("sel".to_string(), vbits(2, Signedness::Unsigned, "01"))]
                    .into_iter()
                    .collect(),
            },
            PipelineCycle {
                inputs: [("sel".to_string(), vbits(2, Signedness::Unsigned, "10"))]
                    .into_iter()
                    .collect(),
            },
        ],
    };
    let init = BTreeMap::new();

    let mut cov = CoverageCounters::default();
    cov.register_functions(&cm.fn_meta);
    run_pipeline_and_collect_coverage(&cm, &stimulus, &init, &src, &mut cov).unwrap();

    // With 2 cycles and no seq, the harness evaluates combo 3*cycles times, so the
    // function call should be observed 6 times (one call per eval due to the
    // single `assign y = f(...)`).
    assert_eq!(cov.function_calls.get("f").copied().unwrap_or(0), 6);

    let rendered = render_annotated_source(&src, &cov, false);
    let expected = concat!(
        "SKP    1 | module m(input logic clk, input logic [1:0] sel, output logic y);\n",
        "SKP    2 |   function automatic logic f (input reg [1:0] sel, input reg case0, input reg case1, input reg def); // calls=6\n",
        "HIT    3 |     begin\n",
        "HIT    4 |       unique casez (sel)\n",
        "HIT    5 |         [A+]2'b?1: begin f = case0; end[/A]\n",
        "HIT    6 |         [A+]2'b10: begin f = case1; end[/A]\n",
        "MIS    7 |         [A-]default: begin f = def; end[/A]\n",
        "HIT    8 |       endcase\n",
        "HIT    9 |     end\n",
        "SKP   10 |   endfunction\n",
        "HIT   11 |   assign y = f(sel, 1'b0, 1'b1, 1'b0);\n",
        "SKP   12 | endmodule\n",
    );
    assert_eq!(rendered, expected);
}

#[test]
fn function_expr_ternary_coverage_is_tracked() {
    let dut = concat!(
        "module m(input logic clk, input logic sel, input logic a, input logic b, output logic y);\n",
        "  function automatic logic pick(input logic s, input logic t, input logic f);\n",
        "    begin\n",
        "      pick = s ? t : f;\n",
        "    end\n",
        "  endfunction\n",
        "  assign y = pick(sel, a, b);\n",
        "endmodule\n",
    );
    let cm = compile_pipeline_module(dut).unwrap();
    let src = SourceText::new(dut.to_string());
    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![
            PipelineCycle {
                inputs: [
                    ("sel".to_string(), vbits(1, Signedness::Unsigned, "0")),
                    ("a".to_string(), vbits(1, Signedness::Unsigned, "1")),
                    ("b".to_string(), vbits(1, Signedness::Unsigned, "0")),
                ]
                .into_iter()
                .collect(),
            },
            PipelineCycle {
                inputs: [
                    ("sel".to_string(), vbits(1, Signedness::Unsigned, "1")),
                    ("a".to_string(), vbits(1, Signedness::Unsigned, "1")),
                    ("b".to_string(), vbits(1, Signedness::Unsigned, "0")),
                ]
                .into_iter()
                .collect(),
            },
        ],
    };
    let init = BTreeMap::new();

    let mut cov = CoverageCounters::default();
    cov.register_functions(&cm.fn_meta);
    run_pipeline_and_collect_coverage(&cm, &stimulus, &init, &src, &mut cov).unwrap();

    assert_eq!(cov.ternary_branches.len(), 1);
    let counts = cov.ternary_branches.values().next().unwrap();
    assert_eq!(counts.f_taken, 3);
    assert_eq!(counts.t_taken, 3);
    assert_eq!(counts.cond_unknown, 0);
}

#[test]
fn function_expr_ternary_equality_recontexts_unbased_unsized_rhs() {
    let dut = concat!(
        "module m(input logic clk, input logic [1:0] sel, input logic a, input logic b, output logic y);\n",
        "  function automatic logic pick(input logic [1:0] s, input logic t, input logic f);\n",
        "    begin\n",
        "      pick = (s == '1) ? t : f;\n",
        "    end\n",
        "  endfunction\n",
        "  assign y = pick(sel, a, b);\n",
        "endmodule\n",
    );
    let cm = compile_pipeline_module(dut).unwrap();
    let src = SourceText::new(dut.to_string());
    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![
            PipelineCycle {
                inputs: [
                    ("sel".to_string(), vbits(2, Signedness::Unsigned, "11")),
                    ("a".to_string(), vbits(1, Signedness::Unsigned, "1")),
                    ("b".to_string(), vbits(1, Signedness::Unsigned, "0")),
                ]
                .into_iter()
                .collect(),
            },
            PipelineCycle {
                inputs: [
                    ("sel".to_string(), vbits(2, Signedness::Unsigned, "10")),
                    ("a".to_string(), vbits(1, Signedness::Unsigned, "1")),
                    ("b".to_string(), vbits(1, Signedness::Unsigned, "0")),
                ]
                .into_iter()
                .collect(),
            },
        ],
    };
    let init = BTreeMap::new();

    let mut cov = CoverageCounters::default();
    cov.register_functions(&cm.fn_meta);
    run_pipeline_and_collect_coverage(&cm, &stimulus, &init, &src, &mut cov).unwrap();

    assert_eq!(cov.ternary_branches.len(), 1);
    let counts = cov.ternary_branches.values().next().unwrap();
    assert_eq!(counts.t_taken, 3);
    assert_eq!(counts.f_taken, 3);
    assert_eq!(counts.cond_unknown, 0);
}

#[test]
fn function_expr_ternary_unbased_unsized_rhs_matrix_tracks_outputs_and_counts() {
    let lhs_decls = ["logic [3:0]", "logic signed [3:0]"];
    let known_cases = [
        ("==", "'0", "0000", "0010", "1", "0"),
        ("==", "'1", "1111", "0010", "1", "0"),
        ("!=", "'0", "0000", "0010", "0", "1"),
        ("!=", "'1", "1111", "0010", "0", "1"),
        ("===", "'0", "0000", "0010", "1", "0"),
        ("===", "'1", "1111", "0010", "1", "0"),
        ("!==", "'0", "0000", "0010", "0", "1"),
        ("!==", "'1", "1111", "0010", "0", "1"),
        ("===", "'x", "xxxx", "0000", "1", "0"),
        ("===", "'z", "zzzz", "0000", "1", "0"),
        ("!==", "'x", "xxxx", "0000", "0", "1"),
        ("!==", "'z", "zzzz", "0000", "0", "1"),
    ];

    for &lhs_decl in &lhs_decls {
        let lhs_signedness = if lhs_decl.contains("signed") {
            Signedness::Signed
        } else {
            Signedness::Unsigned
        };
        for &(op, rhs, true_sel, false_sel, true_out, false_out) in &known_cases {
            let dut = build_function_expr_ternary_module(lhs_decl, op, rhs);
            let cm = compile_pipeline_module(&dut).unwrap();
            let src = SourceText::new(dut);
            let stimulus = PipelineStimulus {
                half_period: 5,
                cycles: vec![
                    make_sel_cycle(vbits(4, lhs_signedness, true_sel), "1", "0"),
                    make_sel_cycle(vbits(4, lhs_signedness, false_sel), "1", "0"),
                ],
            };

            let (coverage_steps, cov) =
                run_combinational_pipeline_with_output_equivalence(&cm, &stimulus, &src);

            let ys: Vec<String> = coverage_steps
                .into_iter()
                .map(|snapshot| snapshot.get("y").unwrap().clone())
                .collect();
            let expected = vec![
                true_out.to_string(),
                true_out.to_string(),
                true_out.to_string(),
                false_out.to_string(),
                false_out.to_string(),
                false_out.to_string(),
            ];
            assert_eq!(ys, expected, "lhs_decl={lhs_decl} op={op} rhs={rhs}");

            assert_eq!(
                cov.ternary_branches.len(),
                1,
                "lhs_decl={lhs_decl} op={op} rhs={rhs}"
            );
            let counts = cov.ternary_branches.values().next().unwrap();
            assert_eq!(counts.t_taken, 3, "lhs_decl={lhs_decl} op={op} rhs={rhs}");
            assert_eq!(counts.f_taken, 3, "lhs_decl={lhs_decl} op={op} rhs={rhs}");
            assert_eq!(
                counts.cond_unknown, 0,
                "lhs_decl={lhs_decl} op={op} rhs={rhs}"
            );
        }
    }
}

#[test]
fn function_expr_ternary_unbased_unsized_rhs_unknown_conditions_track_cond_unknown() {
    let lhs_decls = ["logic [3:0]", "logic signed [3:0]"];
    let unknown_cases = [("==", "'x"), ("==", "'z"), ("!=", "'x"), ("!=", "'z")];

    for &lhs_decl in &lhs_decls {
        let lhs_signedness = if lhs_decl.contains("signed") {
            Signedness::Signed
        } else {
            Signedness::Unsigned
        };
        for &(op, rhs) in &unknown_cases {
            let dut = build_function_expr_ternary_module(lhs_decl, op, rhs);
            let cm = compile_pipeline_module(&dut).unwrap();
            let src = SourceText::new(dut);
            let stimulus = PipelineStimulus {
                half_period: 5,
                cycles: vec![make_sel_cycle(vbits(4, lhs_signedness, "0000"), "1", "0")],
            };

            let (coverage_steps, cov) =
                run_combinational_pipeline_with_output_equivalence(&cm, &stimulus, &src);

            let ys: Vec<String> = coverage_steps
                .into_iter()
                .map(|snapshot| snapshot.get("y").unwrap().clone())
                .collect();
            assert_eq!(
                ys,
                vec!["x".to_string(), "x".to_string(), "x".to_string()],
                "lhs_decl={lhs_decl} op={op} rhs={rhs}"
            );

            assert_eq!(
                cov.ternary_branches.len(),
                1,
                "lhs_decl={lhs_decl} op={op} rhs={rhs}"
            );
            let counts = cov.ternary_branches.values().next().unwrap();
            assert_eq!(counts.t_taken, 0, "lhs_decl={lhs_decl} op={op} rhs={rhs}");
            assert_eq!(counts.f_taken, 0, "lhs_decl={lhs_decl} op={op} rhs={rhs}");
            assert_eq!(
                counts.cond_unknown, 3,
                "lhs_decl={lhs_decl} op={op} rhs={rhs}"
            );
        }
    }
}
