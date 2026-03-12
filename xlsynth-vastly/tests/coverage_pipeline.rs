// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use xlsynth_vastly::CoverageCounters;
use xlsynth_vastly::LogicBit;
use xlsynth_vastly::PipelineCycle;
use xlsynth_vastly::PipelineStimulus;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::SourceText;
use xlsynth_vastly::Value4;
use xlsynth_vastly::compile_pipeline_module;
use xlsynth_vastly::compile_pipeline_module_with_defines;
use xlsynth_vastly::compile_pipeline_module_without_spans;
use xlsynth_vastly::run_pipeline_and_collect_coverage;
use xlsynth_vastly::run_pipeline_and_write_vcd;

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
fn collects_line_ternary_and_toggle_coverage() {
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

    // Line hits: two assigns evaluated 6 times each (time0 + (posedge+negedge)*2 +
    // input-change for cycle1).
    assert_eq!(cov.line_hits.get(&7).copied().unwrap_or(0), 6);
    assert_eq!(cov.line_hits.get(&8).copied().unwrap_or(0), 6);

    // Ternary decision coverage: 3 false decisions in cycle0, 3 true decisions in
    // cycle1.
    assert_eq!(cov.ternary_branches.len(), 1);
    let (_k, c) = cov.ternary_branches.iter().next().unwrap();
    assert_eq!(c.f_taken, 3);
    assert_eq!(c.t_taken, 3);
    assert_eq!(c.cond_unknown, 0);

    // Toggle coverage (0↔1 only):
    // - clk toggles 4 times across the 6 snapshots
    // - a toggles once (0->1) at cycle1 input-change snapshot
    // - t and y follow a, so they toggle once as well
    assert_eq!(cov.toggle_counts.get("clk").unwrap()[0], 4);
    assert_eq!(cov.toggle_counts.get("a").unwrap()[0], 1);
    assert_eq!(cov.toggle_counts.get("t").unwrap()[0], 1);
    assert_eq!(cov.toggle_counts.get("y").unwrap()[0], 1);
}

#[test]
fn pipeline_vcd_rejects_zero_half_period() {
    let dut = concat!(
        "module m(\n",
        "  input logic clk,\n",
        "  input logic a,\n",
        "  output wire y\n",
        ");\n",
        "  assign y = a;\n",
        "endmodule\n",
    );
    let cm = compile_pipeline_module(dut).unwrap();
    let stimulus = PipelineStimulus {
        half_period: 0,
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
    let out_path = std::env::temp_dir().join(format!(
        "xlsynth_vastly_half_period_zero_{}.vcd",
        std::process::id()
    ));
    let init = BTreeMap::new();
    let err = run_pipeline_and_write_vcd(&cm, &stimulus, &init, &out_path).unwrap_err();
    assert!(format!("{err:?}").contains("half_period must be > 0"));
}

#[test]
fn pipeline_observer_eval_failures_are_not_swallowed() {
    let dut = concat!(
        "module m(\n",
        "  input logic clk,\n",
        "  input logic a,\n",
        "  output wire y\n",
        ");\n",
        "  function automatic logic helper(input logic v);\n",
        "    begin\n",
        "      helper = v;\n",
        "    end\n",
        "  endfunction\n",
        "  assign y = a;\n",
        "`ifdef SIMULATION\n",
        "  always_ff @ (posedge clk) begin\n",
        "    if (helper(a)) begin\n",
        "      $display(\"a=%d\", a);\n",
        "    end\n",
        "  end\n",
        "`endif\n",
        "endmodule\n",
    );
    let src = SourceText::new(dut.to_string());
    let mut defs = std::collections::BTreeSet::new();
    defs.insert("SIMULATION".to_string());
    let cm = compile_pipeline_module_with_defines(dut, &defs).unwrap();
    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![PipelineCycle {
            inputs: [("a".to_string(), vbits(1, Signedness::Unsigned, "1"))]
                .into_iter()
                .collect(),
        }],
    };
    let init = BTreeMap::new();
    let mut cov = CoverageCounters::default();
    let err = run_pipeline_and_collect_coverage(&cm, &stimulus, &init, &src, &mut cov).unwrap_err();
    assert!(format!("{err:?}").contains("function call `helper` not supported"));
}

#[test]
fn coverage_attributes_stateful_always_ff_lines() {
    let dut = concat!(
        "module m(\n",
        "  input logic clk,\n",
        "  input logic en,\n",
        "  output logic [3:0] q\n",
        ");\n",
        "  always_ff @ (posedge clk) begin\n",
        "    if (en) begin\n",
        "      q <= q + 4'd1;\n",
        "    end\n",
        "  end\n",
        "endmodule\n",
    );
    let cm = compile_pipeline_module(dut).unwrap();
    let src = SourceText::new(dut.to_string());

    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![
            PipelineCycle {
                inputs: [("en".to_string(), vbits(1, Signedness::Unsigned, "0"))]
                    .into_iter()
                    .collect(),
            },
            PipelineCycle {
                inputs: [("en".to_string(), vbits(1, Signedness::Unsigned, "1"))]
                    .into_iter()
                    .collect(),
            },
        ],
    };
    let init = BTreeMap::new();
    let mut cov = CoverageCounters::default();
    run_pipeline_and_collect_coverage(&cm, &stimulus, &init, &src, &mut cov).unwrap();

    // The always_ff block executes at every posedge, so its source lines must
    // be attributed as hit even when the guard is false.
    assert!(cov.line_hits.get(&6).copied().unwrap_or(0) > 0);
    assert!(cov.line_hits.get(&7).copied().unwrap_or(0) > 0);
    assert!(cov.line_hits.get(&8).copied().unwrap_or(0) > 0);
}

#[test]
fn coverage_rejects_modules_compiled_without_spans() {
    let dut = concat!(
        "module m(\n",
        "  input logic clk,\n",
        "  input logic a,\n",
        "  output wire y\n",
        ");\n",
        "  assign y = a;\n",
        "endmodule\n",
    );
    let cm = compile_pipeline_module_without_spans(dut).unwrap();
    let src = SourceText::new(dut.to_string());
    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![PipelineCycle {
            inputs: [("a".to_string(), vbits(1, Signedness::Unsigned, "0"))]
                .into_iter()
                .collect(),
        }],
    };
    let init = BTreeMap::new();
    let mut cov = CoverageCounters::default();

    let err = run_pipeline_and_collect_coverage(&cm, &stimulus, &init, &src, &mut cov).unwrap_err();
    assert!(format!("{err:?}").contains("coverage requires a module compiled with spans"));
}
