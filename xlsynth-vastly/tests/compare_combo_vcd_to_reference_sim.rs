// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "reference-sim-tests")]

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::io::Write;
use std::time::SystemTime;

use xlsynth_vastly::LogicBit;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::Value4;
use xlsynth_vastly::Vcd;
use xlsynth_vastly::VcdDiffOptions;
use xlsynth_vastly::compile_combo_module;
use xlsynth_vastly::diff_vcd_exact;
use xlsynth_vastly::plan_combo_eval;
use xlsynth_vastly::run_combo_and_write_vcd;
use xlsynth_vastly::run_iverilog_combo_and_collect_vcd;

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

fn assert_common_var_timelines_match(a: &Vcd, b: &Vcd) {
    assert_eq!(a.timescale, b.timescale, "timescale mismatch");
    assert_eq!(a.times(), b.times(), "timestamp mismatch");

    let common: BTreeSet<String> = a
        .var_names()
        .intersection(&b.var_names())
        .cloned()
        .collect();
    assert!(
        common.iter().any(|name| name.starts_with("tb.dut.")),
        "expected at least one common DUT var, common={common:?}"
    );

    let am = a.materialize().unwrap();
    let bm = b.materialize().unwrap();
    for ((ta, va), (tb, vb)) in am.iter().zip(bm.iter()) {
        assert_eq!(ta, tb, "time mismatch");
        for name in &common {
            assert_eq!(
                va.get(name),
                vb.get(name),
                "value mismatch for {name} at time {ta}"
            );
        }
    }
}

#[test]
fn combo_vcd_matches_reference_sim_for_casez_function() {
    let dut = r#"
module m(
  input wire [1:0] sel,
  input wire a,
  input wire b,
  output wire out
);
  function automatic logic pick (input reg [1:0] sel, input reg case0, input reg case1, input reg default_value);
    begin
      unique casez (sel)
        2'b?1: begin
          pick = case0;
        end
        2'b10: begin
          pick = case1;
        end
        2'b00: begin
          pick = default_value;
        end
        default: begin
          pick = 'X;
        end
      endcase
    end
  endfunction

  wire w;
  assign w = pick(sel, a, b, 1'b0);
  assign out = w;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let vectors: Vec<BTreeMap<String, Value4>> = vec![
        // sel=01 => ?1 arm, pick=a
        [
            ("sel".to_string(), vbits(2, Signedness::Unsigned, "01")),
            ("a".to_string(), vbits(1, Signedness::Unsigned, "1")),
            ("b".to_string(), vbits(1, Signedness::Unsigned, "0")),
        ]
        .into_iter()
        .collect(),
        // sel=10 => 10 arm, pick=b
        [
            ("sel".to_string(), vbits(2, Signedness::Unsigned, "10")),
            ("a".to_string(), vbits(1, Signedness::Unsigned, "0")),
            ("b".to_string(), vbits(1, Signedness::Unsigned, "1")),
        ]
        .into_iter()
        .collect(),
        // sel=00 => default_value = 0
        [
            ("sel".to_string(), vbits(2, Signedness::Unsigned, "00")),
            ("a".to_string(), vbits(1, Signedness::Unsigned, "1")),
            ("b".to_string(), vbits(1, Signedness::Unsigned, "1")),
        ]
        .into_iter()
        .collect(),
    ];

    let td = mk_temp_dir();
    let dut_path = td.join("dut.v");
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("reference_sim.vcd");

    {
        let mut f = std::fs::File::create(&dut_path).unwrap();
        f.write_all(dut.as_bytes()).unwrap();
    }

    run_combo_and_write_vcd(&m, &plan, &vectors, &ours_vcd).unwrap();
    run_iverilog_combo_and_collect_vcd(&dut_path, &m, &vectors, &iv_vcd).unwrap();

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    assert_common_var_timelines_match(&ours, &iv);
}

#[test]
fn helper_function_with_locals_and_signed_casts_matches_reference_sim() {
    let dut = r#"
module fuzz_codegen_v(
  input wire [6:0] p0,
  output wire [7:0] out
);
  // lint_off SIGNED_TYPE
  // lint_off MULTIPLY
  function automatic [6:0] smul7b_7b_x_7b (input reg [6:0] lhs, input reg [6:0] rhs);
    reg signed [6:0] signed_lhs;
    reg signed [6:0] signed_rhs;
    reg signed [6:0] signed_result;
    begin
      signed_lhs = $signed(lhs);
      signed_rhs = $signed(rhs);
      signed_result = signed_lhs * signed_rhs;
      smul7b_7b_x_7b = $unsigned(signed_result);
    end
  endfunction
  // lint_on MULTIPLY
  // lint_on SIGNED_TYPE
  wire [6:0] smul_5;
  wire [7:0] one_hot_6;
  assign smul_5 = smul7b_7b_x_7b(p0, p0);
  assign one_hot_6 = {smul_5[6:0] == 7'h00, smul_5[6], smul_5[5] && !smul_5[6], smul_5[4] && smul_5[6:5] == 2'h0, smul_5[3] && smul_5[6:4] == 3'h0, smul_5[2] && smul_5[6:3] == 4'h0, smul_5[1] && smul_5[6:2] == 5'h00, smul_5[0] && smul_5[6:1] == 6'h00};
  assign out = one_hot_6;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let vectors: Vec<BTreeMap<String, Value4>> = vec![
        [("p0".to_string(), vbits(7, Signedness::Unsigned, "0000001"))]
            .into_iter()
            .collect(),
        [("p0".to_string(), vbits(7, Signedness::Unsigned, "1111111"))]
            .into_iter()
            .collect(),
        [("p0".to_string(), vbits(7, Signedness::Unsigned, "1000000"))]
            .into_iter()
            .collect(),
    ];

    let td = mk_temp_dir();
    let dut_path = td.join("dut.v");
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("reference_sim.vcd");

    {
        let mut f = std::fs::File::create(&dut_path).unwrap();
        f.write_all(dut.as_bytes()).unwrap();
    }

    run_combo_and_write_vcd(&m, &plan, &vectors, &ours_vcd).unwrap();
    run_iverilog_combo_and_collect_vcd(&dut_path, &m, &vectors, &iv_vcd).unwrap();

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    assert_common_var_timelines_match(&ours, &iv);
}

#[test]
fn assignment_context_sized_ops_match_iverilog() {
    let dut = r#"
module ctx_ops_v(
  input wire [4:0] a,
  output wire [9:0] mul_out,
  output wire [9:0] nested_out,
  output wire [7:0] add_out,
  output wire [7:0] sub_out,
  output wire [7:0] div_out,
  output wire [7:0] mod_out,
  output wire [7:0] neg_out,
  output wire [7:0] not_out,
  output wire [7:0] shl_out,
  output wire [7:0] lshr_out,
  output wire signed [7:0] sshr_out,
  output wire signed [7:0] bor_ss_out,
  output wire [7:0] bor_su_out
);
  assign mul_out = a * a;
  assign nested_out = (a * a) + 10'd1;
  assign add_out = 4'd15 + 4'd1;
  assign sub_out = 4'd0 - 4'd1;
  assign div_out = 4'd13 / 4'd2;
  assign mod_out = 4'd13 % 4'd2;
  assign neg_out = -4'd1;
  assign not_out = ~4'd1;
  assign shl_out = 4'sb1000 << 1;
  assign lshr_out = 4'sb1000 >> 1;
  assign sshr_out = 4'sb1000 >>> 1;
  assign bor_ss_out = 4'sb1000 | 4'sb0001;
  assign bor_su_out = 4'sb1111 | 4'b0001;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let vectors: Vec<BTreeMap<String, Value4>> = vec![
        [("a".to_string(), vbits(5, Signedness::Unsigned, "01011"))]
            .into_iter()
            .collect(),
    ];

    let td = mk_temp_dir();
    let dut_path = td.join("dut.v");
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("reference_sim.vcd");

    {
        let mut f = std::fs::File::create(&dut_path).unwrap();
        f.write_all(dut.as_bytes()).unwrap();
    }

    run_combo_and_write_vcd(&m, &plan, &vectors, &ours_vcd).unwrap();
    run_iverilog_combo_and_collect_vcd(&dut_path, &m, &vectors, &iv_vcd).unwrap();

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    assert_common_var_timelines_match(&ours, &iv);
}

#[test]
fn function_helper_assignment_context_sized_ops_match_iverilog() {
    let dut = r#"
module ctx_helper_v(
  input wire [4:0] a,
  output wire [9:0] direct_mul_out,
  output wire [9:0] nested_mul_out,
  output wire [9:0] tmp_mul_out
);
  function automatic [9:0] widen_mul_direct (input reg [4:0] lhs, input reg [4:0] rhs);
    begin
      widen_mul_direct = lhs * rhs;
    end
  endfunction
  function automatic [9:0] widen_mul_nested (input reg [4:0] lhs, input reg [4:0] rhs);
    begin
      widen_mul_nested = (lhs * rhs) + 10'd1;
    end
  endfunction
  function automatic [9:0] widen_mul_tmp (input reg [4:0] lhs, input reg [4:0] rhs);
    reg [9:0] tmp;
    begin
      tmp = lhs * rhs;
      widen_mul_tmp = tmp;
    end
  endfunction
  assign direct_mul_out = widen_mul_direct(a, a);
  assign nested_mul_out = widen_mul_nested(a, a);
  assign tmp_mul_out = widen_mul_tmp(a, a);
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let vectors: Vec<BTreeMap<String, Value4>> = vec![
        [("a".to_string(), vbits(5, Signedness::Unsigned, "01011"))]
            .into_iter()
            .collect(),
    ];

    let td = mk_temp_dir();
    let dut_path = td.join("dut.v");
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("reference_sim.vcd");

    {
        let mut f = std::fs::File::create(&dut_path).unwrap();
        f.write_all(dut.as_bytes()).unwrap();
    }

    run_combo_and_write_vcd(&m, &plan, &vectors, &ours_vcd).unwrap();
    run_iverilog_combo_and_collect_vcd(&dut_path, &m, &vectors, &iv_vcd).unwrap();

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    assert_common_var_timelines_match(&ours, &iv);
}

#[test]
fn priority_sel_helper_with_decimal_x_default_matches_reference_sim() {
    let dut = r#"
module fuzz_codegen_v(
  input wire [7:0] p0,
  output wire out
);
  function automatic [7:0] priority_sel_8b_8way (input reg [7:0] sel, input reg [7:0] case0, input reg [7:0] case1, input reg [7:0] case2, input reg [7:0] case3, input reg [7:0] case4, input reg [7:0] case5, input reg [7:0] case6, input reg [7:0] case7, input reg [7:0] default_value);
    begin
      casez (sel)
        8'b???????1: begin
          priority_sel_8b_8way = case0;
        end
        8'b??????10: begin
          priority_sel_8b_8way = case1;
        end
        8'b?????100: begin
          priority_sel_8b_8way = case2;
        end
        8'b????1000: begin
          priority_sel_8b_8way = case3;
        end
        8'b???10000: begin
          priority_sel_8b_8way = case4;
        end
        8'b??100000: begin
          priority_sel_8b_8way = case5;
        end
        8'b?1000000: begin
          priority_sel_8b_8way = case6;
        end
        8'b10000000: begin
          priority_sel_8b_8way = case7;
        end
        8'b0000_0000: begin
          priority_sel_8b_8way = default_value;
        end
        default: begin
          priority_sel_8b_8way = 8'dx;
        end
      endcase
    end
  endfunction
  wire [7:0] priority_sel_5;
  wire decode_6;
  assign priority_sel_5 = priority_sel_8b_8way(p0, p0, p0, p0, p0, p0, p0, p0, p0, p0);
  assign decode_6 = priority_sel_5 >= 8'h01 ? 1'h0 : 1'h1 << priority_sel_5;
  assign out = decode_6;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let vectors: Vec<BTreeMap<String, Value4>> = vec![
        [("p0".to_string(), vbits(8, Signedness::Unsigned, "00000000"))]
            .into_iter()
            .collect(),
        [("p0".to_string(), vbits(8, Signedness::Unsigned, "00000001"))]
            .into_iter()
            .collect(),
        [("p0".to_string(), vbits(8, Signedness::Unsigned, "10000000"))]
            .into_iter()
            .collect(),
    ];

    let td = mk_temp_dir();
    let dut_path = td.join("dut.v");
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("reference_sim.vcd");

    {
        let mut f = std::fs::File::create(&dut_path).unwrap();
        f.write_all(dut.as_bytes()).unwrap();
    }

    run_combo_and_write_vcd(&m, &plan, &vectors, &ours_vcd).unwrap();
    run_iverilog_combo_and_collect_vcd(&dut_path, &m, &vectors, &iv_vcd).unwrap();

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    assert_common_var_timelines_match(&ours, &iv);
}

#[test]
fn dynamic_bit_slice_helper_matches_reference_sim() {
    let dut = r#"
module dbs_mod_v(
  input wire [7:0] x,
  input wire [7:0] s,
  output wire [3:0] out
);
  function automatic [3:0] dynamic_bit_slice_w4_8b_8b (input reg [7:0] operand, input reg [7:0] start);
    reg [11:0] extended_operand;
    begin
      extended_operand = {4'h0, operand};
      dynamic_bit_slice_w4_8b_8b = start >= 4'h8 ? 4'h0 : extended_operand[start +: 4];
    end
  endfunction
  wire [3:0] dbs;
  assign dbs = dynamic_bit_slice_w4_8b_8b(x, s);
  assign out = dbs;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let vectors: Vec<BTreeMap<String, Value4>> = vec![
        [
            ("x".to_string(), vbits(8, Signedness::Unsigned, "10110011")),
            ("s".to_string(), vbits(8, Signedness::Unsigned, "00000000")),
        ]
        .into_iter()
        .collect(),
        [
            ("x".to_string(), vbits(8, Signedness::Unsigned, "10110011")),
            ("s".to_string(), vbits(8, Signedness::Unsigned, "00000100")),
        ]
        .into_iter()
        .collect(),
        [
            ("x".to_string(), vbits(8, Signedness::Unsigned, "10110011")),
            ("s".to_string(), vbits(8, Signedness::Unsigned, "00001000")),
        ]
        .into_iter()
        .collect(),
    ];

    let td = mk_temp_dir();
    let dut_path = td.join("dut.v");
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("reference_sim.vcd");

    {
        let mut f = std::fs::File::create(&dut_path).unwrap();
        f.write_all(dut.as_bytes()).unwrap();
    }

    run_combo_and_write_vcd(&m, &plan, &vectors, &ours_vcd).unwrap();
    run_iverilog_combo_and_collect_vcd(&dut_path, &m, &vectors, &iv_vcd).unwrap();

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    assert_common_var_timelines_match(&ours, &iv);
}

#[test]
fn bit_slice_update_helper_matches_reference_sim() {
    let dut = r#"
module bsu_mod_v(
  input wire [7:0] x,
  input wire [7:0] s,
  input wire [3:0] u,
  output wire [7:0] out
);
  function automatic [7:0] bit_slice_update_w8_8b_4b (input reg [7:0] to_update, input reg [7:0] start, input reg [3:0] update_value);
    begin
      bit_slice_update_w8_8b_4b = start >= 8'h08 ? to_update : {4'h0, update_value} << start | ~(8'h0f << start) & to_update;
    end
  endfunction
  wire [7:0] bsu;
  assign bsu = bit_slice_update_w8_8b_4b(x, s, u);
  assign out = bsu;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let vectors: Vec<BTreeMap<String, Value4>> = vec![
        [
            ("x".to_string(), vbits(8, Signedness::Unsigned, "10110011")),
            ("s".to_string(), vbits(8, Signedness::Unsigned, "00000000")),
            ("u".to_string(), vbits(4, Signedness::Unsigned, "0101")),
        ]
        .into_iter()
        .collect(),
        [
            ("x".to_string(), vbits(8, Signedness::Unsigned, "10110011")),
            ("s".to_string(), vbits(8, Signedness::Unsigned, "00000010")),
            ("u".to_string(), vbits(4, Signedness::Unsigned, "0101")),
        ]
        .into_iter()
        .collect(),
        [
            ("x".to_string(), vbits(8, Signedness::Unsigned, "10110011")),
            ("s".to_string(), vbits(8, Signedness::Unsigned, "00001000")),
            ("u".to_string(), vbits(4, Signedness::Unsigned, "0101")),
        ]
        .into_iter()
        .collect(),
    ];

    let td = mk_temp_dir();
    let dut_path = td.join("dut.v");
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("reference_sim.vcd");

    {
        let mut f = std::fs::File::create(&dut_path).unwrap();
        f.write_all(dut.as_bytes()).unwrap();
    }

    run_combo_and_write_vcd(&m, &plan, &vectors, &ours_vcd).unwrap();
    run_iverilog_combo_and_collect_vcd(&dut_path, &m, &vectors, &iv_vcd).unwrap();

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    assert_common_var_timelines_match(&ours, &iv);
}

#[test]
fn array_index_helper_matches_reference_sim() {
    let dut = r#"
module f(
  input wire [31:0] arr,
  input wire [2:0] start,
  output wire [7:0] out
);
  wire [7:0] arr_unflattened[4];
  assign arr_unflattened[0] = arr[7:0];
  assign arr_unflattened[1] = arr[15:8];
  assign arr_unflattened[2] = arr[23:16];
  assign arr_unflattened[3] = arr[31:24];
  wire [7:0] out__2;
  assign out__2 = arr_unflattened[start > 3'h3 ? 2'h3 : start[1:0]];
  assign out = out__2;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let vectors: Vec<BTreeMap<String, Value4>> = vec![
        [
            (
                "arr".to_string(),
                vbits(32, Signedness::Unsigned, "01000100001100110010001000010001"),
            ),
            ("start".to_string(), vbits(3, Signedness::Unsigned, "000")),
        ]
        .into_iter()
        .collect(),
        [
            (
                "arr".to_string(),
                vbits(32, Signedness::Unsigned, "01000100001100110010001000010001"),
            ),
            ("start".to_string(), vbits(3, Signedness::Unsigned, "010")),
        ]
        .into_iter()
        .collect(),
        [
            (
                "arr".to_string(),
                vbits(32, Signedness::Unsigned, "01000100001100110010001000010001"),
            ),
            ("start".to_string(), vbits(3, Signedness::Unsigned, "111")),
        ]
        .into_iter()
        .collect(),
    ];

    let td = mk_temp_dir();
    let dut_path = td.join("dut.v");
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("reference_sim.vcd");

    {
        let mut f = std::fs::File::create(&dut_path).unwrap();
        f.write_all(dut.as_bytes()).unwrap();
    }

    run_combo_and_write_vcd(&m, &plan, &vectors, &ours_vcd).unwrap();
    run_iverilog_combo_and_collect_vcd(&dut_path, &m, &vectors, &iv_vcd).unwrap();

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    assert_common_var_timelines_match(&ours, &iv);
}

#[test]
fn array_slice_helper_matches_reference_sim() {
    let dut = r#"
module f(
  input wire [31:0] arr,
  input wire [2:0] start,
  output wire [23:0] out
);
  wire [7:0] arr_unflattened[4];
  assign arr_unflattened[0] = arr[7:0];
  assign arr_unflattened[1] = arr[15:8];
  assign arr_unflattened[2] = arr[23:16];
  assign arr_unflattened[3] = arr[31:24];
  wire [7:0] out__2[3];
  assign out__2[0] = arr_unflattened[start > 3'h3 ? 3'h3 : start + 3'h0];
  assign out__2[1] = arr_unflattened[start > 3'h2 ? 3'h3 : start + 3'h1];
  assign out__2[2] = arr_unflattened[start > 3'h1 ? 3'h3 : start + 3'h2];
  assign out = {out__2[2], out__2[1], out__2[0]};
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let vectors: Vec<BTreeMap<String, Value4>> = vec![
        [
            (
                "arr".to_string(),
                vbits(32, Signedness::Unsigned, "01000100001100110010001000010001"),
            ),
            ("start".to_string(), vbits(3, Signedness::Unsigned, "000")),
        ]
        .into_iter()
        .collect(),
        [
            (
                "arr".to_string(),
                vbits(32, Signedness::Unsigned, "01000100001100110010001000010001"),
            ),
            ("start".to_string(), vbits(3, Signedness::Unsigned, "001")),
        ]
        .into_iter()
        .collect(),
        [
            (
                "arr".to_string(),
                vbits(32, Signedness::Unsigned, "01000100001100110010001000010001"),
            ),
            ("start".to_string(), vbits(3, Signedness::Unsigned, "010")),
        ]
        .into_iter()
        .collect(),
    ];

    let td = mk_temp_dir();
    let dut_path = td.join("dut.v");
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("reference_sim.vcd");

    {
        let mut f = std::fs::File::create(&dut_path).unwrap();
        f.write_all(dut.as_bytes()).unwrap();
    }

    run_combo_and_write_vcd(&m, &plan, &vectors, &ours_vcd).unwrap();
    run_iverilog_combo_and_collect_vcd(&dut_path, &m, &vectors, &iv_vcd).unwrap();

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    assert_common_var_timelines_match(&ours, &iv);
}

#[test]
fn array_update_helper_matches_reference_sim() {
    let dut = r#"
module f(
  input wire [31:0] arr,
  input wire [2:0] start,
  input wire [7:0] val,
  output wire [31:0] out
);
  wire [7:0] arr_unflattened[4];
  assign arr_unflattened[0] = arr[7:0];
  assign arr_unflattened[1] = arr[15:8];
  assign arr_unflattened[2] = arr[23:16];
  assign arr_unflattened[3] = arr[31:24];
  wire [7:0] out__2[4];
  assign out = {out__2[3], out__2[2], out__2[1], out__2[0]};
  for (genvar __i0 = 0; __i0 < 4; __i0 = __i0 + 1) begin : gen__out__2_0
    assign out__2[__i0] = start == __i0 ? val : arr_unflattened[__i0];
  end
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let vectors: Vec<BTreeMap<String, Value4>> = vec![
        [
            (
                "arr".to_string(),
                vbits(32, Signedness::Unsigned, "01000100001100110010001000010001"),
            ),
            ("start".to_string(), vbits(3, Signedness::Unsigned, "001")),
            (
                "val".to_string(),
                vbits(8, Signedness::Unsigned, "10101010"),
            ),
        ]
        .into_iter()
        .collect(),
        [
            (
                "arr".to_string(),
                vbits(32, Signedness::Unsigned, "01000100001100110010001000010001"),
            ),
            ("start".to_string(), vbits(3, Signedness::Unsigned, "011")),
            (
                "val".to_string(),
                vbits(8, Signedness::Unsigned, "11110000"),
            ),
        ]
        .into_iter()
        .collect(),
        [
            (
                "arr".to_string(),
                vbits(32, Signedness::Unsigned, "01000100001100110010001000010001"),
            ),
            ("start".to_string(), vbits(3, Signedness::Unsigned, "111")),
            (
                "val".to_string(),
                vbits(8, Signedness::Unsigned, "10101010"),
            ),
        ]
        .into_iter()
        .collect(),
    ];

    let td = mk_temp_dir();
    let dut_path = td.join("dut.v");
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("reference_sim.vcd");

    {
        let mut f = std::fs::File::create(&dut_path).unwrap();
        f.write_all(dut.as_bytes()).unwrap();
    }

    run_combo_and_write_vcd(&m, &plan, &vectors, &ours_vcd).unwrap();
    run_iverilog_combo_and_collect_vcd(&dut_path, &m, &vectors, &iv_vcd).unwrap();

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    assert_common_var_timelines_match(&ours, &iv);
}

#[test]
fn unpacked_array_generated_combo_matches_reference_sim() {
    let dut = r#"
module fuzz_codegen_v(
  input wire [2:0] p0,
  output wire [10:0] out
);
  wire [2:0] tuple_11;
  wire [5:0] tuple_13;
  wire [2:0] array_12[1:0];
  wire [38:0] tuple_15;
  wire [2:0] tuple_index_16;
  wire [10:0] zero_ext_17;
  assign tuple_11 = {p0};
  assign tuple_13 = {tuple_11, tuple_11};
  assign array_12[0] = tuple_11;
  assign array_12[1] = tuple_11;
  assign tuple_15 = {p0, {array_12[1], array_12[0]}, tuple_11, tuple_13, {array_12[1], array_12[0]}, p0, tuple_13, {array_12[1], array_12[0]}};
  assign tuple_index_16 = tuple_15[38:36];
  assign zero_ext_17 = {8'h00, tuple_index_16};
  assign out = zero_ext_17;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let vectors: Vec<BTreeMap<String, Value4>> = vec![
        [("p0".to_string(), vbits(3, Signedness::Unsigned, "000"))]
            .into_iter()
            .collect(),
        [("p0".to_string(), vbits(3, Signedness::Unsigned, "101"))]
            .into_iter()
            .collect(),
        [("p0".to_string(), vbits(3, Signedness::Unsigned, "111"))]
            .into_iter()
            .collect(),
    ];

    let td = mk_temp_dir();
    let dut_path = td.join("dut.v");
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("reference_sim.vcd");

    {
        let mut f = std::fs::File::create(&dut_path).unwrap();
        f.write_all(dut.as_bytes()).unwrap();
    }

    run_combo_and_write_vcd(&m, &plan, &vectors, &ours_vcd).unwrap();
    run_iverilog_combo_and_collect_vcd(&dut_path, &m, &vectors, &iv_vcd).unwrap();

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    assert_common_var_timelines_match(&ours, &iv);
}

#[test]
fn nested_unpacked_array_generated_combo_matches_reference_sim() {
    let dut = r#"
module fuzz_codegen_v(
  input wire [1:0] p0,
  output wire [1:0] out
);
  wire [1:0] array_10[1:0];
  wire [1:0] array_11[1:0][1:0];
  wire [1:0] sel_12[1:0][1:0];
  assign array_10[0] = p0;
  assign array_10[1] = ~p0;
  assign array_11[0][0] = array_10[0];
  assign array_11[0][1] = array_10[1];
  assign array_11[1][0] = array_10[1];
  assign array_11[1][1] = array_10[0];
  assign sel_12[0][0] = p0 == 2'h0 ? array_11[0][0] : array_11[1][1];
  assign sel_12[0][1] = p0 == 2'h0 ? array_11[0][1] : array_11[1][0];
  assign sel_12[1][0] = p0 == 2'h0 ? array_11[1][0] : array_11[0][1];
  assign sel_12[1][1] = p0 == 2'h0 ? array_11[1][1] : array_11[0][0];
  assign out = sel_12[1][0];
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let vectors: Vec<BTreeMap<String, Value4>> = vec![
        [("p0".to_string(), vbits(2, Signedness::Unsigned, "00"))]
            .into_iter()
            .collect(),
        [("p0".to_string(), vbits(2, Signedness::Unsigned, "01"))]
            .into_iter()
            .collect(),
        [("p0".to_string(), vbits(2, Signedness::Unsigned, "10"))]
            .into_iter()
            .collect(),
    ];

    let td = mk_temp_dir();
    let dut_path = td.join("dut.v");
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("reference_sim.vcd");

    {
        let mut f = std::fs::File::create(&dut_path).unwrap();
        f.write_all(dut.as_bytes()).unwrap();
    }

    run_combo_and_write_vcd(&m, &plan, &vectors, &ours_vcd).unwrap();
    run_iverilog_combo_and_collect_vcd(&dut_path, &m, &vectors, &iv_vcd).unwrap();

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    assert_common_var_timelines_match(&ours, &iv);
}

#[test]
fn finding96_codegen_decode_guarded_shift_matches_reference_sim() {
    // Repro from fuzz finding 96 trace.
    let dut = r#"
module fuzz_codegen_v(
  input wire [7:0] p0,
  output wire out
);
  wire [15:0] zero_ext_18;
  wire [23:0] zero_ext_19;
  wire [31:0] zero_ext_20;
  wire [63:0] concat_21;
  wire [70:0] zero_ext_22;
  wire [141:0] concat_23;
  wire [7:0] decode_25;
  wire bit_slice_30;
  assign zero_ext_18 = {8'h00, p0};
  assign zero_ext_19 = {8'h00, zero_ext_18};
  assign zero_ext_20 = {8'h00, zero_ext_19};
  assign concat_21 = {zero_ext_20, zero_ext_20};
  assign zero_ext_22 = {7'h00, concat_21};
  assign concat_23 = {zero_ext_22, zero_ext_22};
  assign decode_25 = concat_23 >= 142'h0000_0000_0000_0000_0000_0000_0000_0000_0008 ? 8'h00 : 8'h01 << concat_23;
  assign bit_slice_30 = decode_25[0];
  assign out = bit_slice_30;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let vectors: Vec<BTreeMap<String, Value4>> = vec![
        [("p0".to_string(), vbits(8, Signedness::Unsigned, "00000000"))]
            .into_iter()
            .collect(),
        [("p0".to_string(), vbits(8, Signedness::Unsigned, "00000001"))]
            .into_iter()
            .collect(),
    ];

    let td = mk_temp_dir();
    let dut_path = td.join("dut.v");
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("reference_sim.vcd");

    {
        let mut f = std::fs::File::create(&dut_path).unwrap();
        f.write_all(dut.as_bytes()).unwrap();
    }

    run_combo_and_write_vcd(&m, &plan, &vectors, &ours_vcd).unwrap();
    run_iverilog_combo_and_collect_vcd(&dut_path, &m, &vectors, &iv_vcd).unwrap();

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    diff_vcd_exact(&ours, &iv, &VcdDiffOptions::default()).unwrap();
}

#[test]
fn finding57_codegen_wide_unary_minus_slice_matches_reference_sim() {
    // Repro class from fuzz finding 57 trace.
    let dut = r#"
module fuzz_codegen_v(
  input wire [5:0] p0,
  output wire [24:0] out
);
  wire [13:0] zero_ext_18;
  wire [21:0] zero_ext_19;
  wire [29:0] zero_ext_20;
  wire [59:0] concat_21;
  wire [67:0] zero_ext_22;
  wire [135:0] concat_23;
  wire [135:0] neg_25;
  wire [24:0] bit_slice_30;
  assign zero_ext_18 = {8'h00, p0};
  assign zero_ext_19 = {8'h00, zero_ext_18};
  assign zero_ext_20 = {8'h00, zero_ext_19};
  assign concat_21 = {zero_ext_20, zero_ext_20};
  assign zero_ext_22 = {8'h00, concat_21};
  assign concat_23 = {zero_ext_22, zero_ext_22};
  assign neg_25 = -concat_23;
  assign bit_slice_30 = neg_25[112:88];
  assign out = bit_slice_30;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let vectors: Vec<BTreeMap<String, Value4>> = vec![
        [("p0".to_string(), vbits(6, Signedness::Unsigned, "000001"))]
            .into_iter()
            .collect(),
        [("p0".to_string(), vbits(6, Signedness::Unsigned, "111111"))]
            .into_iter()
            .collect(),
    ];

    let td = mk_temp_dir();
    let dut_path = td.join("dut.v");
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("reference_sim.vcd");

    {
        let mut f = std::fs::File::create(&dut_path).unwrap();
        f.write_all(dut.as_bytes()).unwrap();
    }

    run_combo_and_write_vcd(&m, &plan, &vectors, &ours_vcd).unwrap();
    run_iverilog_combo_and_collect_vcd(&dut_path, &m, &vectors, &iv_vcd).unwrap();

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    diff_vcd_exact(&ours, &iv, &VcdDiffOptions::default()).unwrap();
}

#[test]
fn finding1_codegen_wide_multiply_slice_matches_reference_sim() {
    // Repro class from fuzz finding 1 trace.
    let dut = r#"
module fuzz_codegen_v(
  input wire [7:0] p0,
  output wire [53:0] out
);
  function automatic [141:0] umul142b_142b_x_142b (input reg [141:0] lhs, input reg [141:0] rhs);
    begin
      umul142b_142b_x_142b = lhs * rhs;
    end
  endfunction
  wire [15:0] zero_ext_18;
  wire [23:0] zero_ext_19;
  wire [31:0] zero_ext_20;
  wire [63:0] concat_21;
  wire [70:0] zero_ext_22;
  wire [141:0] concat_23;
  wire [141:0] umul_25;
  wire [53:0] bit_slice_30;
  assign zero_ext_18 = {8'h00, p0};
  assign zero_ext_19 = {8'h00, zero_ext_18};
  assign zero_ext_20 = {8'h00, zero_ext_19};
  assign concat_21 = {zero_ext_20, zero_ext_20};
  assign zero_ext_22 = {7'h00, concat_21};
  assign concat_23 = {zero_ext_22, zero_ext_22};
  assign umul_25 = umul142b_142b_x_142b(concat_23, concat_23);
  assign bit_slice_30 = umul_25[106:53];
  assign out = bit_slice_30;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let vectors: Vec<BTreeMap<String, Value4>> = vec![
        [("p0".to_string(), vbits(8, Signedness::Unsigned, "00000001"))]
            .into_iter()
            .collect(),
        [("p0".to_string(), vbits(8, Signedness::Unsigned, "11111111"))]
            .into_iter()
            .collect(),
    ];

    let td = mk_temp_dir();
    let dut_path = td.join("dut.v");
    let ours_vcd = td.join("ours.vcd");
    let iv_vcd = td.join("reference_sim.vcd");

    {
        let mut f = std::fs::File::create(&dut_path).unwrap();
        f.write_all(dut.as_bytes()).unwrap();
    }

    run_combo_and_write_vcd(&m, &plan, &vectors, &ours_vcd).unwrap();
    run_iverilog_combo_and_collect_vcd(&dut_path, &m, &vectors, &iv_vcd).unwrap();

    let ours_text = std::fs::read_to_string(&ours_vcd).unwrap();
    let iv_text = std::fs::read_to_string(&iv_vcd).unwrap();
    let ours = Vcd::parse(&ours_text).unwrap();
    let iv = Vcd::parse(&iv_text).unwrap();
    diff_vcd_exact(&ours, &iv, &VcdDiffOptions::default()).unwrap();
}

fn mk_temp_dir() -> std::path::PathBuf {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    for attempt in 0u32..1000u32 {
        let p = base.join(format!("vastly_combo_test_{pid}_{nanos}_{attempt}"));
        match std::fs::create_dir(&p) {
            Ok(()) => return p,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => panic!("create temp dir: {e:?}"),
        }
    }
    panic!("failed to create temp dir");
}
