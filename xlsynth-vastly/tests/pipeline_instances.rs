// SPDX-License-Identifier: Apache-2.0

use xlsynth_vastly::{
    CoverageCounters, Error, FixtureRunner, LogicBit, PipelineCycle, PipelineStimulus, Signedness,
    SourceText, Value4, compile_pipeline_module_with_top, run_pipeline_and_collect_coverage,
};

// The helper's formal shadows the child wire. Its callee must still read the
// child wire, whose driver appears after the call in the source.
const TWO_INSTANCES: &str = r#"
module pair(input wire clk, input wire [3:0] a, input wire [3:0] b,
            output logic [7:0] result);
  wire [3:0] left_value;
  wire [3:0] right_value;
  stage left (.y(left_value), .sample(a));
  stage right (.sample({2'b01, b[1:0]}), .y(right_value));
  always @(posedge clk) result <= {left_value, right_value};
endmodule

module stage(input wire [3:0] sample, output wire [3:0] y);
  wire [3:0] inner;
  function automatic [3:0] captured(input reg [3:0] sample);
    begin
      captured = inner ^ sample;
    end
  endfunction
  function automatic [3:0] combine(input reg [3:0] inner);
    reg [3:0] tmp;
    begin
      tmp = captured(inner);
      combine = tmp;
    end
  endfunction
  assign y = combine(4'b1010);
  assign inner = sample;
endmodule

module unused(input wire a, output wire b);
  assign b = a;
endmodule
"#;

#[test]
fn repeated_child_has_independent_wires_and_helpers() {
    let module = compile_pipeline_module_with_top(TWO_INSTANCES, "pair").unwrap();
    let mut runner = FixtureRunner::new(&module).unwrap();
    let ports = runner.port_bindings();
    let a = ports.input_with_width("a", 4).unwrap();
    let b = ports.input_with_width("b", 4).unwrap();
    let edge = runner
        .step_cycle_with_drive(&mut [], |drive| {
            drive.set_u64(&a, 3)?;
            drive.set_u64(&b, 2)
        })
        .unwrap();
    assert_eq!(edge.outputs["result"].to_u64_if_known(), Some(0x9c));

    let four_state = Value4::new(
        4,
        Signedness::Unsigned,
        vec![LogicBit::Z, LogicBit::One, LogicBit::Zero, LogicBit::X],
    );
    let edge = runner
        .step_cycle_with_drive(&mut [], |drive| {
            drive.set(&a, four_state.clone())?;
            drive.set_u64(&b, 3)
        })
        .unwrap();
    assert_eq!(edge.outputs["left.inner"], four_state);
    assert_eq!(edge.outputs["result"].to_bit_string_msb_first(), "x00x1101");

    // Coverage uses the spanned expression tree and the original file offsets.
    let mut cov = CoverageCounters::default();
    let src = SourceText::new(TWO_INSTANCES.to_owned());
    run_pipeline_and_collect_coverage(
        &module,
        &PipelineStimulus {
            half_period: 1,
            cycles: vec![PipelineCycle {
                inputs: edge.inputs,
            }],
        },
        &module.initial_state_x(),
        &src,
        &mut cov,
    )
    .unwrap();
    assert!(cov.function_calls["left.combine"] > 0);
    assert!(cov.function_calls["right.combine"] > 0);
    let assign_line = TWO_INSTANCES
        .lines()
        .position(|l| l.trim() == "assign inner = sample;")
        .unwrap() as u32
        + 1;
    assert!(cov.line_hits[&assign_line] > 0);
    let unused_line = TWO_INSTANCES
        .lines()
        .position(|l| l.trim() == "assign b = a;")
        .unwrap() as u32
        + 1;
    let cover = cov.coverability(&src);
    assert!(cover.is_coverable(assign_line));
    assert!(!cover.is_coverable(unused_line));
}

#[test]
fn port_boundaries_apply_the_childs_declared_width_and_signedness() {
    let rtl = r#"
module top #(parameter W = 8)(input wire clk, input wire [W-1:0] data,
                               output wire [W-1:0] result);
  stage inst (.x(data), .y(result));
endmodule
module stage #(parameter W = 4)(input wire [W-1:0] x, output wire signed [W-1:0] y);
  assign y = x;
endmodule
"#;
    let module = compile_pipeline_module_with_top(rtl, "top").unwrap();
    let mut runner = FixtureRunner::new(&module).unwrap();
    let data = runner.port_bindings().input("data").unwrap();
    let edge = runner
        .step_cycle_with_drive(&mut [], |drive| drive.set_u64(&data, 0x7e))
        .unwrap();
    assert_eq!(edge.outputs["result"].to_u64_if_known(), Some(0xfe));
}

#[test]
fn malformed_instance_connections_are_rejected_at_compile_time() {
    for (source, error) in [
        (
            TWO_INSTANCES.replace(".y(left_value)", ".z(left_value)"),
            "missing port connection `left.y`",
        ),
        (
            TWO_INSTANCES.replace(".y(left_value)", ".y(left_value), .z(left_value)"),
            "unknown port `z` on instance `left` of `stage`",
        ),
        (
            TWO_INSTANCES.replace(".y(left_value)", ".y(left_value), .y(right_value)"),
            "duplicate port connection `left.y`",
        ),
        (
            TWO_INSTANCES.replace(".y(left_value)", ".y(4'b0)"),
            "output `left.y` must connect to a signal name",
        ),
        (
            TWO_INSTANCES.replace(".sample(a)", ".sample(absent)"),
            "input `left.sample` references unknown signal or parameter `absent`",
        ),
        (
            TWO_INSTANCES.replace("stage left", "other left"),
            "unknown child module `other` in instance `left`",
        ),
        (
            TWO_INSTANCES.replace("stage right", "stage left"),
            "duplicate instance `left`",
        ),
        (
            TWO_INSTANCES.replace(
                "assign inner = sample;",
                "always @(posedge sample) inner <= sample;",
            ),
            "instance `left` of `stage` must be a combinational leaf module",
        ),
    ] {
        assert_eq!(
            compile_pipeline_module_with_top(&source, "pair").unwrap_err(),
            Error::Parse(error.into())
        );
    }
    assert_eq!(
        compile_pipeline_module_with_top(TWO_INSTANCES, "missing").unwrap_err(),
        Error::Parse("unknown top module `missing`".into())
    );
}

#[test]
fn stitched_add_mul_rtl_holds_disabled_banks() {
    let rtl = include_str!("fixtures/stitch_enable_add_mul.golden.sv");
    let module = compile_pipeline_module_with_top(rtl, "pipe").unwrap();
    let mut runner = FixtureRunner::new(&module).unwrap();
    let ports = runner.port_bindings();
    let rst = ports.input_with_width("rst", 1).unwrap();
    let in_valid = ports.input_with_width("in_valid", 1).unwrap();
    let mul_op = ports.input_with_width("mul_op", 1).unwrap();
    let x = ports.input_with_width("x", 4).unwrap();
    let y = ports.input_with_width("y", 4).unwrap();
    let mut previous = (false, false);
    for (reset, valid, mul, a, b, expected_valid, expected_data) in [
        (true, false, false, 0, 0, 0, None),
        (false, true, true, 2, 3, 0, None),
        (false, true, false, 7, 3, 1, Some(6)),
        (false, false, true, 9, 1, 1, Some(10)),
        (false, true, true, 4, 2, 0, Some(10)),
        (true, true, false, 2, 3, 0, Some(8)),
        (false, false, false, 0, 0, 0, Some(8)),
        (false, true, false, 6, 5, 0, Some(8)),
        (false, false, true, 1, 1, 1, Some(11)),
        // Reset can change a stage-0 payload while clearing its valid bit.
        // The following bubble must keep that payload out of stage 1.
        (true, true, true, 3, 5, 0, Some(11)),
        (false, false, true, 0, 0, 0, Some(11)),
    ] {
        let edge = runner
            .step_cycle_with_drive(&mut [], |drive| {
                drive.set_bool(&rst, reset)?;
                drive.set_bool(&in_valid, valid)?;
                drive.set_bool(&mul_op, mul)?;
                drive.set_u64(&x, a)?;
                drive.set_u64(&y, b)
            })
            .unwrap();
        // The disabled payload is unobservable at "out". Inspect each bank
        // across the edge to catch an unconditional write in either layer.
        for (bank, load) in [
            ("p0_bank_at_0", valid && mul),
            ("p0_bank_at_5", valid && !mul),
            ("p1_bank_at_0", previous.0 && previous.1),
            ("p1_bank_at_5", previous.0 && !previous.1),
        ] {
            if !load {
                assert_eq!(
                    edge.outputs[bank], edge.pre_outputs[bank],
                    "{bank} changed on cycle {}",
                    edge.cycle
                );
            }
        }
        previous = (valid && !reset, mul);
        assert_eq!(
            edge.outputs["out_valid"].to_u64_if_known(),
            Some(expected_valid)
        );
        if let Some(expected) = expected_data {
            assert_eq!(edge.outputs["out"].to_u64_if_known(), Some(expected));
        }
    }
}
