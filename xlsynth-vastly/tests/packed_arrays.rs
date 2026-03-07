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
fn combo_module_supports_packed_array_ports_and_decls() {
    let dut = r#"
module m(
  input logic [1:0][3:0] a,
  input logic sel,
  output logic [3:0] y
);
  logic [1:0][3:0] tmp;
  assign tmp = a;
  assign y = tmp[sel];
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    assert_eq!(m.decls.get("a").unwrap().width, 8);
    assert_eq!(m.decls.get("tmp").unwrap().width, 8);

    let plan = plan_combo_eval(&m).unwrap();
    let inputs0: BTreeMap<String, Value4> = [
        ("a".to_string(), vbits(8, Signedness::Unsigned, "10100011")),
        ("sel".to_string(), vbits(1, Signedness::Unsigned, "0")),
    ]
    .into_iter()
    .collect();
    let values0 = eval_combo(&m, &plan, &inputs0).unwrap();
    assert_eq!(values0.get("y").unwrap().to_bit_string_msb_first(), "0011");

    let inputs1: BTreeMap<String, Value4> = [
        ("a".to_string(), vbits(8, Signedness::Unsigned, "10100011")),
        ("sel".to_string(), vbits(1, Signedness::Unsigned, "1")),
    ]
    .into_iter()
    .collect();
    let values1 = eval_combo(&m, &plan, &inputs1).unwrap();
    assert_eq!(values1.get("y").unwrap().to_bit_string_msb_first(), "1010");
}

#[test]
fn pipeline_module_updates_packed_array_elements() {
    let dut = r#"
module m(
  input logic clk,
  input logic rst,
  input logic [1:0][1:0][3:0] in_data,
  output logic [1:0][1:0][3:0] out_data,
  output logic [3:0] tap
);
  logic [1:0][1:0][3:0] q;

  always_ff @(posedge clk) begin
    if (rst) begin
      q[0][0] <= 4'h1;
      q[0][1] <= 4'h2;
      q[1][0] <= 4'h3;
      q[1][1] <= 4'h4;
    end else begin
      q[0][0] <= in_data[1][1];
      q[0][1] <= in_data[1][0];
      q[1][0] <= in_data[0][1];
      q[1][1] <= in_data[0][0];
    end
  end

  assign out_data = q;
  assign tap = q[1][0];
endmodule
"#;

    let m = compile_pipeline_module(dut).unwrap();
    assert_eq!(m.combo.decls.get("in_data").unwrap().width, 16);
    assert_eq!(m.combo.decls.get("q").unwrap().width, 16);

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
                .collect(),
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
                .collect(),
            },
        ],
    };

    let init: State = BTreeMap::new();
    let outputs = run_pipeline_and_collect_outputs(&m, &stimulus, &init).unwrap();

    assert_eq!(
        outputs[0]
            .get("out_data")
            .unwrap()
            .to_bit_string_msb_first(),
        "0100001100100001"
    );
    assert_eq!(
        outputs[0].get("tap").unwrap().to_bit_string_msb_first(),
        "0011"
    );

    assert_eq!(
        outputs[1]
            .get("out_data")
            .unwrap()
            .to_bit_string_msb_first(),
        "1100110111101111"
    );
    assert_eq!(
        outputs[1].get("tap").unwrap().to_bit_string_msb_first(),
        "1101"
    );
}

#[test]
fn pipeline_module_supports_unpacked_array_of_packed_elements() {
    let dut = r#"
module m(
  input logic clk,
  input logic rst,
  input logic [1:0] in_data,
  output logic [1:0] out_data,
  output logic [3:0] snapshot
);
  logic [1:0] lanes[1:0];
  logic [1:0] q;

  assign lanes[0] = in_data;
  assign lanes[1] = ~in_data;
  assign snapshot = {lanes[1], lanes[0]};

  always_ff @(posedge clk) begin
    if (rst) begin
      q <= 2'b00;
    end else begin
      q <= lanes[1];
    end
  end

  assign out_data = q;
endmodule
"#;

    let m = compile_pipeline_module(dut).unwrap();
    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "1")),
                    ("in_data".to_string(), vbits(2, Signedness::Unsigned, "01")),
                ]
                .into_iter()
                .collect(),
            },
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "0")),
                    ("in_data".to_string(), vbits(2, Signedness::Unsigned, "01")),
                ]
                .into_iter()
                .collect(),
            },
        ],
    };

    let init: State = BTreeMap::new();
    let outputs = run_pipeline_and_collect_outputs(&m, &stimulus, &init).unwrap();
    assert_eq!(
        outputs[0]
            .get("snapshot")
            .unwrap()
            .to_bit_string_msb_first(),
        "1001"
    );
    assert_eq!(
        outputs[0]
            .get("out_data")
            .unwrap()
            .to_bit_string_msb_first(),
        "00"
    );
    assert_eq!(
        outputs[1]
            .get("snapshot")
            .unwrap()
            .to_bit_string_msb_first(),
        "1001"
    );
    assert_eq!(
        outputs[1]
            .get("out_data")
            .unwrap()
            .to_bit_string_msb_first(),
        "10"
    );
}

#[test]
fn pipeline_module_supports_nested_unpacked_arrays_of_packed_elements() {
    let dut = r#"
module m(
  input logic clk,
  input logic rst,
  input logic [1:0] in_data,
  output logic [1:0] out_data,
  output logic [3:0] tap
);
  logic [1:0] table[1:0][1:0];
  logic [1:0] pick;
  logic [1:0] q;

  assign table[0][0] = in_data;
  assign table[0][1] = ~in_data;
  assign table[1][0] = 2'b11;
  assign table[1][1] = 2'b00;
  assign pick = table[0][1];
  assign tap = {table[1][0], table[0][1]};

  always_ff @(posedge clk) begin
    if (rst) begin
      q <= 2'b00;
    end else begin
      q <= pick;
    end
  end

  assign out_data = q;
endmodule
"#;

    let m = compile_pipeline_module(dut).unwrap();
    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "1")),
                    ("in_data".to_string(), vbits(2, Signedness::Unsigned, "01")),
                ]
                .into_iter()
                .collect(),
            },
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "0")),
                    ("in_data".to_string(), vbits(2, Signedness::Unsigned, "01")),
                ]
                .into_iter()
                .collect(),
            },
        ],
    };

    let init: State = BTreeMap::new();
    let outputs = run_pipeline_and_collect_outputs(&m, &stimulus, &init).unwrap();
    assert_eq!(
        outputs[0].get("tap").unwrap().to_bit_string_msb_first(),
        "1110"
    );
    assert_eq!(
        outputs[0]
            .get("out_data")
            .unwrap()
            .to_bit_string_msb_first(),
        "00"
    );
    assert_eq!(
        outputs[1].get("tap").unwrap().to_bit_string_msb_first(),
        "1110"
    );
    assert_eq!(
        outputs[1]
            .get("out_data")
            .unwrap()
            .to_bit_string_msb_first(),
        "10"
    );
}

#[test]
fn combo_module_supports_dynamic_index_on_unpacked_array_of_packed_values() {
    let dut = r#"
module m(
  input logic [1:0][3:0] a0,
  input logic [1:0][3:0] a1,
  input logic sel,
  output logic [1:0][3:0] y
);
  logic [1:0][3:0] arr[1:0];
  assign arr[0] = a0;
  assign arr[1] = a1;
  assign y = arr[sel];
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let out_sel0 = eval_combo(
        &m,
        &plan,
        &[
            ("a0".to_string(), vbits(8, Signedness::Unsigned, "00111100")),
            ("a1".to_string(), vbits(8, Signedness::Unsigned, "10100101")),
            ("sel".to_string(), vbits(1, Signedness::Unsigned, "0")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_sel0["y"].to_bit_string_msb_first(), "00111100");

    let out_sel1 = eval_combo(
        &m,
        &plan,
        &[
            ("a0".to_string(), vbits(8, Signedness::Unsigned, "00111100")),
            ("a1".to_string(), vbits(8, Signedness::Unsigned, "10100101")),
            ("sel".to_string(), vbits(1, Signedness::Unsigned, "1")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_sel1["y"].to_bit_string_msb_first(), "10100101");
}

#[test]
fn pipeline_module_supports_dynamic_index_on_unpacked_array_of_packed_values() {
    let dut = r#"
module m(
  input logic clk,
  input logic rst,
  input logic [1:0][3:0] a0,
  input logic [1:0][3:0] a1,
  input logic sel,
  output logic [1:0][3:0] out_data
);
  logic [1:0][3:0] arr[1:0];
  logic [1:0][3:0] q;

  assign arr[0] = a0;
  assign arr[1] = a1;

  always_ff @(posedge clk) begin
    if (rst) begin
      q <= 8'h00;
    end else begin
      q <= arr[sel];
    end
  end

  assign out_data = q;
endmodule
"#;

    let m = compile_pipeline_module(dut).unwrap();
    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "1")),
                    ("a0".to_string(), vbits(8, Signedness::Unsigned, "00111100")),
                    ("a1".to_string(), vbits(8, Signedness::Unsigned, "10100101")),
                    ("sel".to_string(), vbits(1, Signedness::Unsigned, "0")),
                ]
                .into_iter()
                .collect(),
            },
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "0")),
                    ("a0".to_string(), vbits(8, Signedness::Unsigned, "00111100")),
                    ("a1".to_string(), vbits(8, Signedness::Unsigned, "10100101")),
                    ("sel".to_string(), vbits(1, Signedness::Unsigned, "0")),
                ]
                .into_iter()
                .collect(),
            },
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "0")),
                    ("a0".to_string(), vbits(8, Signedness::Unsigned, "00111100")),
                    ("a1".to_string(), vbits(8, Signedness::Unsigned, "10100101")),
                    ("sel".to_string(), vbits(1, Signedness::Unsigned, "1")),
                ]
                .into_iter()
                .collect(),
            },
        ],
    };

    let init: State = BTreeMap::new();
    let outputs = run_pipeline_and_collect_outputs(&m, &stimulus, &init).unwrap();
    assert_eq!(
        outputs[0]
            .get("out_data")
            .unwrap()
            .to_bit_string_msb_first(),
        "00000000"
    );
    assert_eq!(
        outputs[1]
            .get("out_data")
            .unwrap()
            .to_bit_string_msb_first(),
        "00111100"
    );
    assert_eq!(
        outputs[2]
            .get("out_data")
            .unwrap()
            .to_bit_string_msb_first(),
        "10100101"
    );
}

#[test]
fn pipeline_oob_lhs_indexed_write_is_noop_for_unpacked_array_of_packed_values() {
    let dut = r#"
module m(
  input logic clk,
  input logic rst,
  input logic [1:0] in0,
  input logic [1:0] in1,
  output logic [3:0] snapshot
);
  logic [1:0] lanes[1:0];

  always_ff @(posedge clk) begin
    if (rst) begin
      lanes[0] <= in0;
      lanes[1] <= in1;
    end else begin
      lanes[2] <= 2'b11;
    end
  end

  assign snapshot = {lanes[1], lanes[0]};
endmodule
"#;

    let m = compile_pipeline_module(dut).unwrap();
    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "1")),
                    ("in0".to_string(), vbits(2, Signedness::Unsigned, "01")),
                    ("in1".to_string(), vbits(2, Signedness::Unsigned, "10")),
                ]
                .into_iter()
                .collect(),
            },
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "0")),
                    ("in0".to_string(), vbits(2, Signedness::Unsigned, "00")),
                    ("in1".to_string(), vbits(2, Signedness::Unsigned, "00")),
                ]
                .into_iter()
                .collect(),
            },
        ],
    };

    let init: State = BTreeMap::new();
    let outputs = run_pipeline_and_collect_outputs(&m, &stimulus, &init).unwrap();
    assert_eq!(
        outputs[0]
            .get("snapshot")
            .unwrap()
            .to_bit_string_msb_first(),
        "1001"
    );
    assert_eq!(
        outputs[1]
            .get("snapshot")
            .unwrap()
            .to_bit_string_msb_first(),
        "1001"
    );
}

#[test]
fn pipeline_oob_rhs_index_read_returns_x_for_unpacked_array_of_packed_values() {
    let dut = r#"
module m(
  input logic clk,
  input logic rst,
  input logic [1:0] in_data,
  input logic [1:0] sel,
  output logic [1:0] out_data
);
  logic [1:0] lanes[1:0];
  logic [1:0] q;

  assign lanes[0] = in_data;
  assign lanes[1] = ~in_data;

  always_ff @(posedge clk) begin
    if (rst) begin
      q <= 2'b00;
    end else begin
      q <= lanes[sel];
    end
  end

  assign out_data = q;
endmodule
"#;

    let m = compile_pipeline_module(dut).unwrap();
    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "1")),
                    ("in_data".to_string(), vbits(2, Signedness::Unsigned, "01")),
                    ("sel".to_string(), vbits(2, Signedness::Unsigned, "00")),
                ]
                .into_iter()
                .collect(),
            },
            PipelineCycle {
                inputs: [
                    ("rst".to_string(), vbits(1, Signedness::Unsigned, "0")),
                    ("in_data".to_string(), vbits(2, Signedness::Unsigned, "01")),
                    ("sel".to_string(), vbits(2, Signedness::Unsigned, "10")),
                ]
                .into_iter()
                .collect(),
            },
        ],
    };

    let init: State = BTreeMap::new();
    let outputs = run_pipeline_and_collect_outputs(&m, &stimulus, &init).unwrap();
    assert_eq!(
        outputs[0]
            .get("out_data")
            .unwrap()
            .to_bit_string_msb_first(),
        "00"
    );
    assert_eq!(
        outputs[1]
            .get("out_data")
            .unwrap()
            .to_bit_string_msb_first(),
        "xx"
    );
}
