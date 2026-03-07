// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use xlsynth_vastly::LogicBit;
use xlsynth_vastly::PipelineCycle;
use xlsynth_vastly::PipelineStimulus;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::State;
use xlsynth_vastly::Value4;
use xlsynth_vastly::compile_combo_module;
use xlsynth_vastly::compile_pipeline_module;
use xlsynth_vastly::eval_combo;
use xlsynth_vastly::plan_combo_eval;
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

#[test]
fn pipeline_module_elaborates_nested_generate_loops() {
    let dut = r#"
module m(
  input logic clk,
  input logic rst,
  input logic [1:0][1:0][3:0] in_data,
  output logic [1:0][1:0][3:0] out_data,
  output logic [1:0][1:0][3:0] reset_data
);
  logic [1:0][1:0][3:0] q;
  logic [1:0][1:0][3:0] reset_v;

  for (genvar i = 0; i < 2; i = i + 1) begin : gi
    for (genvar j = 0; j < 2; j = j + 1) begin : gj
      if (i == 0 && j == 0) begin
        assign reset_v[i][j] = 4'h1;
      end else if (i == 0) begin
        assign reset_v[i][j] = 4'h2;
      end else if (j == 0) begin
        assign reset_v[i][j] = 4'h3;
      end else begin
        assign reset_v[i][j] = 4'h4;
      end

      assign out_data[i][j] = q[i][j];

      always_ff @(posedge clk) begin
        if (rst) begin
          q[i][j] <= reset_v[i][j];
        end else begin
          q[i][j] <= in_data[i][j];
        end
      end
    end
  end

  assign reset_data = reset_v;
endmodule
"#;

    let m = compile_pipeline_module(dut).unwrap();
    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "1")),
                    (
                        "in_data".to_string(),
                        vbits(16, Signedness::Unsigned, "0000000000000000"),
                    ),
                ]
                .into_iter()
                .collect::<BTreeMap<_, _>>(),
            },
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "0")),
                    (
                        "in_data".to_string(),
                        vbits(16, Signedness::Unsigned, "1111111011011100"),
                    ),
                ]
                .into_iter()
                .collect::<BTreeMap<_, _>>(),
            },
        ],
    };

    let init: State = BTreeMap::new();
    let outputs = run_pipeline_and_collect_outputs(&m, &stimulus, &init).unwrap();

    assert_eq!(
        outputs[0]
            .get("reset_data")
            .unwrap()
            .to_bit_string_msb_first(),
        "0100001100100001"
    );
    assert_eq!(
        outputs[0]
            .get("out_data")
            .unwrap()
            .to_bit_string_msb_first(),
        "0100001100100001"
    );
    assert_eq!(
        outputs[1]
            .get("out_data")
            .unwrap()
            .to_bit_string_msb_first(),
        "1111111011011100"
    );
}

#[test]
fn combo_module_elaborates_generate_if_branches() {
    let dut = r#"
module m(
  input wire [1:0][3:0] in_data,
  output wire [1:0][3:0] out_data,
  output wire [1:0][3:0] mask_data
);
  wire [1:0][3:0] mask;

  for (genvar i = 0; i < 2; i = i + 1) begin : gi
    if (i == 0) begin
      assign mask[i] = 4'h3;
    end else begin
      assign mask[i] = 4'hc;
    end
    assign out_data[i] = in_data[i] ^ mask[i];
  end

  assign mask_data = mask;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();
    let out = eval_combo(
        &m,
        &plan,
        &[(
            "in_data".to_string(),
            vbits(8, Signedness::Unsigned, "01011010"),
        )]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();

    assert_eq!(out["mask_data"].to_bit_string_msb_first(), "11000011");
    assert_eq!(out["out_data"].to_bit_string_msb_first(), "10011001");
}

#[test]
fn combo_module_elaborates_nested_generate_loops() {
    let dut = r#"
module m(
  input wire [1:0][1:0][3:0] in_data,
  output wire [1:0][1:0][3:0] out_data,
  output wire [1:0][1:0][3:0] mask_data
);
  wire [1:0][1:0][3:0] mask;

  for (genvar i = 0; i < 2; i = i + 1) begin : gi
    for (genvar j = 0; j < 2; j = j + 1) begin : gj
      if (i == 0 && j == 0) begin
        assign mask[i][j] = 4'h1;
      end else if (i == 0) begin
        assign mask[i][j] = 4'h2;
      end else if (j == 0) begin
        assign mask[i][j] = 4'h3;
      end else begin
        assign mask[i][j] = 4'h4;
      end

      assign out_data[i][j] = in_data[i][j] ^ mask[i][j];
    end
  end

  assign mask_data = mask;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();
    let out = eval_combo(
        &m,
        &plan,
        &[(
            "in_data".to_string(),
            vbits(16, Signedness::Unsigned, "1111111011011100"),
        )]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();

    assert_eq!(
        out["mask_data"].to_bit_string_msb_first(),
        "0100001100100001"
    );
    assert_eq!(
        out["out_data"].to_bit_string_msb_first(),
        "1011110111111101"
    );
}

#[test]
fn combo_module_rejects_declaration_inside_generate_loop() {
    let dut = r#"
module m(
  input wire [1:0] in_data,
  output wire [1:0] out_data
);
  for (genvar i = 0; i < 2; i = i + 1) begin : gi
    wire t;
    assign t = in_data[i];
    assign out_data[i] = t;
  end
endmodule
"#;

    let err = compile_combo_module(dut).unwrap_err();
    assert!(format!("{err:?}").contains("declarations inside generate blocks are not supported"));
}

#[test]
fn pipeline_module_rejects_declaration_inside_generate_loop() {
    let dut = r#"
module m(
  input logic clk,
  input logic [1:0] in_data,
  output logic [1:0] out_data
);
  for (genvar i = 0; i < 2; i = i + 1) begin : gi
    logic t;
    assign t = in_data[i];
    assign out_data[i] = t;
  end

  always_ff @(posedge clk) begin
  end
endmodule
"#;

    let err = compile_pipeline_module(dut).unwrap_err();
    assert!(format!("{err:?}").contains("declarations inside generate blocks are not supported"));
}
