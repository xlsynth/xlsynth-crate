// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use xlsynth_vastly::CoverageCounters;
use xlsynth_vastly::Env;
use xlsynth_vastly::PipelineCycle;
use xlsynth_vastly::PipelineStimulus;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::SourceText;
use xlsynth_vastly::Value4;
use xlsynth_vastly::compile_combo_module;
use xlsynth_vastly::compile_pipeline_module;
use xlsynth_vastly::eval_combo;
use xlsynth_vastly::eval_combo_seeded_with_coverage;
use xlsynth_vastly::eval_expr;
use xlsynth_vastly::plan_combo_eval;
use xlsynth_vastly::run_pipeline_and_collect_outputs;

fn ubits(width: u32, token: &str) -> Value4 {
    Value4::parse_numeric_token(width, Signedness::Unsigned, token).unwrap()
}

#[test]
fn eval_sized_cast_with_identifier_width() {
    let mut env = Env::new();
    env.insert(
        "AddrWidth",
        Value4::parse_numeric_token(32, Signedness::Signed, "2").unwrap(),
    );
    let got = eval_expr("AddrWidth'(64'hf)", &env).unwrap().value;
    assert_eq!(got.width, 2);
    assert_eq!(got.to_bit_string_msb_first(), "11");
}

#[test]
fn compile_and_eval_parameterized_combo_module() {
    let sv = r#"
module mycombo #(
  parameter BusWidth = 8,
  parameter logic [BusWidth - 1:0] Mask = 8'h0f
) (
  input logic [BusWidth - 1:0] x,
  output logic [BusWidth - 1:0] y
);
  assign y = x & Mask;
endmodule
"#;

    let m = compile_combo_module(sv).unwrap();
    assert_eq!(m.decls.get("x").unwrap().width, 8);
    assert_eq!(m.decls.get("y").unwrap().width, 8);
    assert_eq!(
        m.consts
            .get("BusWidth")
            .unwrap()
            .to_decimal_string_if_known(),
        Some("8".to_string())
    );
    assert_eq!(
        m.consts.get("Mask").unwrap().to_bit_string_msb_first(),
        "00001111"
    );

    let plan = plan_combo_eval(&m).unwrap();
    let inputs = BTreeMap::from([("x".to_string(), ubits(8, "170"))]);
    let values = eval_combo(&m, &plan, &inputs).unwrap();
    assert_eq!(
        values.get("y").unwrap().to_bit_string_msb_first(),
        "00001010"
    );
}

#[test]
fn parameterized_function_can_read_module_parameter() {
    let sv = r#"
module mycombo_with_fn #(
  parameter BusWidth = 8,
  parameter logic [BusWidth - 1:0] Mask = 8'h0f
) (
  input logic [BusWidth - 1:0] x,
  output logic [BusWidth - 1:0] y
);
  function automatic logic [BusWidth - 1:0] apply_mask(input logic [BusWidth - 1:0] v);
    begin
      apply_mask = v & Mask;
    end
  endfunction
  assign y = apply_mask(x);
endmodule
"#;

    let m = compile_combo_module(sv).unwrap();
    let plan = plan_combo_eval(&m).unwrap();
    let inputs = BTreeMap::from([("x".to_string(), ubits(8, "170"))]);

    let values = eval_combo(&m, &plan, &inputs).unwrap();
    assert_eq!(
        values.get("y").unwrap().to_bit_string_msb_first(),
        "00001010"
    );

    // Also exercise the coverage path, which evaluates function bodies via the
    // spanned-expression evaluator.
    let mut cov = CoverageCounters::default();
    let src = SourceText::new(sv.to_string());
    let mut seed = Env::new();
    seed.insert("x", ubits(8, "170"));
    let values_cov =
        eval_combo_seeded_with_coverage(&m, &plan, &seed, &src, &mut cov, &BTreeMap::new())
            .unwrap();
    assert_eq!(
        values_cov.get("y").unwrap().to_bit_string_msb_first(),
        "00001010"
    );
}

#[test]
fn compile_and_eval_parameterized_pipeline_module_with_pipeline_api() {
    let sv = r#"
module mymodule #(
  parameter BusWidth = 32,
  parameter AddrWidth = 2,
  parameter logic [AddrWidth - 1:0] BlockOffset = '0
) (
  input logic clk,
  input logic req_valid,
  input logic [AddrWidth - 1:0] req_addr,
  input logic [BusWidth / 8 - 1:0] req_wstrb,
  output logic hit
);
  logic [AddrWidth - 1:0] req_addr_q;

  always_ff @ (posedge clk) begin
    req_addr_q <= req_valid ? req_addr : req_addr_q;
  end

  assign hit = req_addr_q == BlockOffset + AddrWidth'(64'h0000_0000_0000_0000);
endmodule
"#;

    let m = compile_pipeline_module(sv).unwrap();
    assert_eq!(m.seqs.len(), 1);
    assert_eq!(m.combo.decls.get("req_addr").unwrap().width, 2);
    assert_eq!(m.combo.decls.get("req_wstrb").unwrap().width, 4);
    assert_eq!(
        m.combo
            .consts
            .get("BusWidth")
            .unwrap()
            .to_decimal_string_if_known(),
        Some("32".to_string())
    );
    assert_eq!(
        m.combo
            .consts
            .get("AddrWidth")
            .unwrap()
            .to_decimal_string_if_known(),
        Some("2".to_string())
    );

    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles: vec![
            PipelineCycle {
                inputs: BTreeMap::from([
                    ("req_valid".to_string(), ubits(1, "1")),
                    ("req_addr".to_string(), ubits(2, "0")),
                    ("req_wstrb".to_string(), ubits(4, "0")),
                ]),
            },
            PipelineCycle {
                inputs: BTreeMap::from([
                    ("req_valid".to_string(), ubits(1, "1")),
                    ("req_addr".to_string(), ubits(2, "1")),
                    ("req_wstrb".to_string(), ubits(4, "0")),
                ]),
            },
        ],
    };

    let outputs = run_pipeline_and_collect_outputs(&m, &stimulus, &m.initial_state_x()).unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(
        outputs[0].get("hit").unwrap().to_bit_string_msb_first(),
        "1"
    );
    assert_eq!(
        outputs[1].get("hit").unwrap().to_bit_string_msb_first(),
        "0"
    );
}

#[test]
fn parameter_without_default_reports_helpful_error() {
    let sv = r#"
module missing_default #(
  parameter AddrWidth
) (
  input logic [7:0] x,
  output logic [7:0] y
);
  assign y = x;
endmodule
"#;

    let err = compile_combo_module(sv).unwrap_err();
    assert_eq!(
        err,
        xlsynth_vastly::Error::Parse(
            "parameter `AddrWidth` must have a default value via `=`; parameter overrides are not supported yet"
                .to_string()
        )
    );
}
