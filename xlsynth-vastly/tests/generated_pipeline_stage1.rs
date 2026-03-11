// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "irvals")]

use std::collections::BTreeMap;

use xlsynth_vastly::LogicBit;
use xlsynth_vastly::PipelineCycle;
use xlsynth_vastly::PipelineStimulus;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::Value4;
use xlsynth_vastly::compile_pipeline_module;
use xlsynth_vastly::run_pipeline_and_collect_outputs;

const SIMPLE_ADD_IR: &str = r#"package stage1_pipeline_tests

top fn main(x: bits[8]) -> bits[8] {
  one: bits[8] = literal(value=1, id=2)
  ret add.3: bits[8] = add(x, one, id=3)
}
"#;

const PIPELINE_MUL_BSU_IR: &str = r#"package pipeline_codegen_regressions

top fn fuzz_test(p0: bits[5]) -> bits[12] {
  zero_ext.3: bits[10] = zero_ext(p0, new_bit_count=10, id=3)
  smul.4: bits[10] = smul(zero_ext.3, zero_ext.3, id=4)
  bit_slice_update.5: bits[10] = bit_slice_update(smul.4, smul.4, smul.4, id=5)
  sel.2: bits[5] = sel(p0, cases=[p0, p0, p0, p0], default=p0, id=2)
  ret zero_ext.6: bits[12] = zero_ext(bit_slice_update.5, new_bit_count=12, id=6)
}
"#;

fn bit1(bit: LogicBit) -> Value4 {
    Value4::new(1, Signedness::Unsigned, vec![bit])
}

fn vbits(width: u32, msb: &str) -> Value4 {
    assert_eq!(msb.len(), width as usize);
    let bits = msb
        .chars()
        .rev()
        .map(|c| match c {
            '0' => LogicBit::Zero,
            '1' => LogicBit::One,
            'x' | 'X' => LogicBit::X,
            'z' | 'Z' => LogicBit::Z,
            _ => panic!("bad bit char {c}"),
        })
        .collect();
    Value4::new(width, Signedness::Unsigned, bits)
}

fn canonical_ir(ir_text: &str) -> String {
    xlsynth::IrPackage::parse_ir(ir_text, None)
        .unwrap()
        .to_string()
}

fn codegen_pipeline(
    ir_text: &str,
    module_name: &str,
    pipeline_stages: u32,
) -> Result<String, xlsynth::XlsynthError> {
    let package = xlsynth::IrPackage::parse_ir(ir_text, None)?;
    let sched_proto = format!("delay_model: \"unit\"\npipeline_stages: {pipeline_stages}");
    let codegen_proto = format!(
        "register_merge_strategy: STRATEGY_IDENTITY_ONLY\n\
generator: GENERATOR_KIND_PIPELINE\n\
use_system_verilog: true\n\
module_name: \"{module_name}\"\n\
input_valid_signal: \"input_valid\"\n\
output_valid_signal: \"output_valid\"\n\
flop_inputs: false\n\
flop_outputs: false\n\
reset: \"rst\"\n\
reset_active_low: false\n\
reset_asynchronous: false\n\
reset_data_path: true\n\
add_invariant_assertions: false\n\
codegen_version: 1"
    );
    let result = xlsynth::schedule_and_codegen(&package, &sched_proto, &codegen_proto)?;
    result.get_verilog_text()
}

fn build_stimulus(
    input_name: &str,
    payload_cycles: &[Value4],
    flush_cycles: usize,
) -> PipelineStimulus {
    let zero_payload = Value4::zeros(payload_cycles[0].width, Signedness::Unsigned);
    let mut cycles = Vec::new();
    for _ in 0..2 {
        cycles.push(PipelineCycle {
            inputs: BTreeMap::from([
                ("rst".to_string(), bit1(LogicBit::One)),
                ("input_valid".to_string(), bit1(LogicBit::Zero)),
                (input_name.to_string(), zero_payload.clone()),
            ]),
        });
    }
    for x in payload_cycles {
        cycles.push(PipelineCycle {
            inputs: BTreeMap::from([
                ("rst".to_string(), bit1(LogicBit::Zero)),
                ("input_valid".to_string(), bit1(LogicBit::One)),
                (input_name.to_string(), x.clone()),
            ]),
        });
    }
    for _ in 0..flush_cycles {
        cycles.push(PipelineCycle {
            inputs: BTreeMap::from([
                ("rst".to_string(), bit1(LogicBit::Zero)),
                ("input_valid".to_string(), bit1(LogicBit::Zero)),
                (input_name.to_string(), zero_payload.clone()),
            ]),
        });
    }
    PipelineStimulus {
        half_period: 5,
        cycles,
    }
}

fn retired_outputs(
    m: &xlsynth_vastly::CompiledPipelineModule,
    stimulus: &PipelineStimulus,
) -> Vec<String> {
    let init = m.initial_state_x();
    run_pipeline_and_collect_outputs(m, stimulus, &init)
        .unwrap()
        .into_iter()
        .filter_map(|cycle_out| {
            let valid = cycle_out.get("output_valid")?;
            match valid.bits_lsb_first().first().copied() {
                Some(LogicBit::One) => {
                    Some(cycle_out.get("out").unwrap().to_bit_string_msb_first())
                }
                _ => None,
            }
        })
        .collect()
}

#[test]
fn generated_stage1_pipeline_retires_combo_values_after_reset() {
    let ir_text = canonical_ir(SIMPLE_ADD_IR);
    let src = codegen_pipeline(&ir_text, "generated_stage1", 1).unwrap();
    let m = compile_pipeline_module(&src).unwrap();
    let payload = vec![
        vbits(8, "00000001"),
        vbits(8, "00000010"),
        vbits(8, "00000011"),
    ];
    let stimulus = build_stimulus("x", &payload, 1);
    let retired = retired_outputs(&m, &stimulus);

    assert_eq!(retired, vec!["00000010", "00000011", "00000100"]);
}

#[test]
fn generated_stage2_pipeline_retires_same_values_as_stage1() {
    let ir_text = canonical_ir(SIMPLE_ADD_IR);
    let payload = vec![
        vbits(8, "00000001"),
        vbits(8, "00000010"),
        vbits(8, "00000011"),
    ];

    let src1 = codegen_pipeline(&ir_text, "generated_stage1", 1).unwrap();
    let src2 = codegen_pipeline(&ir_text, "generated_stage2", 2).unwrap();
    let m1 = compile_pipeline_module(&src1).unwrap();
    let m2 = compile_pipeline_module(&src2).unwrap();

    let retired1 = retired_outputs(&m1, &build_stimulus("x", &payload, 1));
    let retired2 = retired_outputs(&m2, &build_stimulus("x", &payload, 2));

    assert_eq!(retired1, vec!["00000010", "00000011", "00000100"]);
    assert_eq!(retired2, retired1);
}

#[test]
fn multi_always_ff_pipeline_retires_expected_values() {
    let dut = r#"
module m(
  input logic clk,
  input logic rst,
  input logic input_valid,
  input logic [7:0] x,
  output logic output_valid,
  output logic [7:0] out
);
  logic s0_valid;
  logic [7:0] s0_x;
  logic s1_valid;
  logic [7:0] s1_out;

  always_ff @(posedge clk) begin
    if (rst) begin
      s0_valid <= 1'b0;
      s0_x <= 8'h00;
    end else begin
      s0_valid <= input_valid;
      s0_x <= input_valid ? x : s0_x;
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      s1_valid <= 1'b0;
      s1_out <= 8'h00;
    end else begin
      s1_valid <= s0_valid;
      s1_out <= s0_valid ? (s0_x + 8'h01) : s1_out;
    end
  end

  assign output_valid = s1_valid;
  assign out = s1_out;
endmodule
"#;
    let m = compile_pipeline_module(dut).unwrap();
    assert_eq!(m.seqs.len(), 2);

    let payload = vec![
        vbits(8, "00000001"),
        vbits(8, "00000010"),
        vbits(8, "00000011"),
    ];
    let retired = retired_outputs(&m, &build_stimulus("x", &payload, 2));
    assert_eq!(retired, vec!["00000010", "00000011", "00000100"]);
}

#[test]
fn finding_run26_stage1_pipeline_mul_helper_matches_ir() {
    let ir_text = canonical_ir(PIPELINE_MUL_BSU_IR);
    let src = codegen_pipeline(&ir_text, "finding_run26_pipe1", 1).unwrap();
    let m = compile_pipeline_module(&src).unwrap();
    let retired = retired_outputs(&m, &build_stimulus("p0", &[vbits(5, "01011")], 1));

    assert_eq!(retired, vec!["000001111001"]);
}
