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
