// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use xlsynth_vastly::CoverageCounters;
use xlsynth_vastly::LogicBit;
use xlsynth_vastly::PipelineCycle;
use xlsynth_vastly::PipelineStimulus;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::SourceText;
use xlsynth_vastly::SpannedExprKind;
use xlsynth_vastly::Value4;
use xlsynth_vastly::compile_pipeline_module;
use xlsynth_vastly::parse_expr_spanned;
use xlsynth_vastly::render_annotated_source;
use xlsynth_vastly::run_pipeline_and_collect_coverage;

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

#[test]
fn spanned_parser_captures_ternary_branch_spans() {
    let s = "a ? (b ? c : d) : e";
    let e = parse_expr_spanned(s).unwrap();
    let SpannedExprKind::Ternary { t, f, .. } = &e.kind else {
        panic!("expected ternary");
    };
    assert_eq!(&s[t.span.start..t.span.end], "(b ? c : d)");
    assert_eq!(&s[f.span.start..f.span.end], "e");
}

#[test]
fn distinct_ternaries_on_same_line_are_disambiguated_by_span() {
    let dut = "module m(input logic clk, input logic a, output wire y);\n  assign y = a ? 1'b1 : 1'b0; assign y = a ? 1'b1 : 1'b0;\nendmodule\n";
    let cm = compile_pipeline_module(dut).unwrap();
    let mut cov = CoverageCounters::default();
    for a in &cm.combo.assigns {
        cov.register_ternaries_from_spanned_expr(
            a.rhs_spanned
                .as_ref()
                .expect("coverage registration requires spanned assign expressions"),
        );
    }
    assert_eq!(cov.ternary_branches.len(), 2);
    let mut spans: Vec<(usize, usize)> = cov
        .ternary_branches
        .keys()
        .map(|k| (k.start, k.end))
        .collect();
    spans.sort();
    assert_ne!(spans[0], spans[1]);
}

#[test]
fn annotated_source_plain_shows_branch_hits() {
    let dut = concat!(
        "module m(\n",
        "  input logic clk,\n",
        "  input logic a,\n",
        "  output wire y\n",
        ");\n",
        "  wire t;\n",
        "  assign t = a ? 1'b1 : 1'b0;\n",
        "  assign y = t;\n",
        "endmodule\n",
    );
    let cm = compile_pipeline_module(dut).unwrap();
    let src = SourceText::new(dut.to_string());
    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![
            PipelineCycle {
                inputs: [("a".to_string(), vbits(1, Signedness::Unsigned, "0"))]
                    .into_iter()
                    .collect(),
            },
            PipelineCycle {
                inputs: [("a".to_string(), vbits(1, Signedness::Unsigned, "1"))]
                    .into_iter()
                    .collect(),
            },
        ],
    };
    let init = BTreeMap::new();

    let mut cov = CoverageCounters::default();
    for a in &cm.combo.assigns {
        cov.register_ternaries_from_spanned_expr(
            a.rhs_spanned
                .as_ref()
                .expect("coverage registration requires spanned assign expressions"),
        );
    }
    run_pipeline_and_collect_coverage(&cm, &stimulus, &init, &src, &mut cov).unwrap();

    let rendered = render_annotated_source(&src, &cov, false);
    let expected = concat!(
        "SKP    1 | module m(\n",
        "SKP    2 |   input logic clk,\n",
        "SKP    3 |   input logic a,\n",
        "SKP    4 |   output wire y\n",
        "SKP    5 | );\n",
        "SKP    6 |   wire t;\n",
        "HIT    7 |   assign t = a ? [T+]1'b1[/T] : [F+]1'b0[/F];\n",
        "HIT    8 |   assign y = t;\n",
        "SKP    9 | endmodule\n",
    );
    assert_eq!(rendered, expected);
}

#[test]
fn nested_ternary_shows_leaf_level_branch_hits_and_annotation() {
    let dut = concat!(
        "module m(input logic clk, input logic a, input logic b, output wire [1:0] y);\n",
        "  assign y = a ? (b ? 2'b10 : 2'b11) : 2'b00;\n",
        "endmodule\n",
    );
    let cm = compile_pipeline_module(dut).unwrap();
    let src = SourceText::new(dut.to_string());

    // Always take outer true branch (a=1). Always take inner true branch (b=1).
    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![
            PipelineCycle {
                inputs: [
                    ("a".to_string(), vbits(1, Signedness::Unsigned, "1")),
                    ("b".to_string(), vbits(1, Signedness::Unsigned, "1")),
                ]
                .into_iter()
                .collect(),
            },
            PipelineCycle {
                inputs: [
                    ("a".to_string(), vbits(1, Signedness::Unsigned, "1")),
                    ("b".to_string(), vbits(1, Signedness::Unsigned, "1")),
                ]
                .into_iter()
                .collect(),
            },
        ],
    };
    let init = BTreeMap::new();

    let mut cov = CoverageCounters::default();
    for a in &cm.combo.assigns {
        cov.register_ternaries_from_spanned_expr(
            a.rhs_spanned
                .as_ref()
                .expect("coverage registration requires spanned assign expressions"),
        );
    }
    run_pipeline_and_collect_coverage(&cm, &stimulus, &init, &src, &mut cov).unwrap();

    // Two ternaries: the outer `a ? (...) : 2'b00` and the inner `b ? 2'b10 :
    // 2'b11`.
    assert_eq!(cov.ternary_branches.len(), 2);
    let mut by_span_len = cov
        .ternary_branches
        .iter()
        .map(|(k, v)| (k.end.saturating_sub(k.start), v))
        .collect::<Vec<_>>();
    by_span_len.sort_by_key(|(len, _)| *len);
    let inner = by_span_len[0].1;
    let outer = by_span_len[1].1;

    // Inner: b is always true => true branch taken, false branch never taken.
    assert!(inner.t_taken > 0);
    assert_eq!(inner.f_taken, 0);
    // Outer: a is always true => true branch taken, false branch never taken.
    assert!(outer.t_taken > 0);
    assert_eq!(outer.f_taken, 0);

    // Annotation should show leaf-level miss even though outer side is hit:
    // - inner false branch `2'b11` is [F-]
    // - outer false branch `2'b00` is [F-]
    let rendered = render_annotated_source(&src, &cov, false);
    let line2 = rendered.lines().nth(1).unwrap();
    let expected_line2 = concat!(
        "HIT    2 |   assign y = a ? ",
        "[T+](b ? [/T]",
        "[T+]2'b10[/T]",
        "[T+] : [/T]",
        "[F-]2'b11[/F]",
        "[T+])[/T]",
        " : ",
        "[F-]2'b00[/F]",
        ";"
    );
    assert_eq!(line2, expected_line2);
}
