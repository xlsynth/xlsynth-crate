// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use xlsynth_vastly::CoverageCounters;
use xlsynth_vastly::LogicBit;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::SourceText;
use xlsynth_vastly::Value4;
use xlsynth_vastly::compile_combo_module;
use xlsynth_vastly::eval_combo;
use xlsynth_vastly::eval_combo_seeded_with_coverage;
use xlsynth_vastly::plan_combo_eval;

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
fn parses_combo_ports_wires_assigns_and_casez_function() {
    let dut = r#"
module m(
  input wire [15:0] x,
  input wire [15:0] y,
  output wire [15:0] out
);
  function automatic logic pick (input reg [1:0] sel, input reg case0, input reg case1, input reg default_value);
    begin
      unique casez (sel)
        2'b?1: begin
          pick = case0;
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

  wire [15:0] w;
  assign w = x ^ y;
  assign out = w;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    assert_eq!(m.module_name, "m");
    assert_eq!(m.input_ports.len(), 2);
    assert_eq!(m.input_ports[0].name, "x");
    assert_eq!(m.input_ports[0].width, 16);
    assert_eq!(m.input_ports[1].name, "y");
    assert_eq!(m.input_ports[1].width, 16);
    assert_eq!(m.output_ports.len(), 1);
    assert_eq!(m.output_ports[0].name, "out");
    assert_eq!(m.output_ports[0].width, 16);

    // Wire decls land in decls map:
    assert!(m.decls.contains_key("w"));

    // Assigns and functions should be present.
    assert!(m.assigns.len() >= 2);
    assert!(m.functions.contains_key("pick"));
}

#[test]
fn parses_and_evals_generated_helper_function_with_locals_and_casts() {
    let dut = r#"
module fuzz_codegen_v(
  input wire [6:0] p0,
  output wire [7:0] out
);
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
  wire [6:0] smul_5;
  wire [7:0] one_hot_6;
  assign smul_5 = smul7b_7b_x_7b(p0, p0);
  assign one_hot_6 = {smul_5[6:0] == 7'h00, smul_5[6], smul_5[5] && !smul_5[6], smul_5[4] && smul_5[6:5] == 2'h0, smul_5[3] && smul_5[6:4] == 3'h0, smul_5[2] && smul_5[6:3] == 4'h0, smul_5[1] && smul_5[6:2] == 5'h00, smul_5[0] && smul_5[6:1] == 6'h00};
  assign out = one_hot_6;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let out_one = eval_combo(
        &m,
        &plan,
        &[("p0".to_string(), vbits(7, Signedness::Unsigned, "0000001"))]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_one["out"].to_bit_string_msb_first(), "00000001");

    let out_neg_one = eval_combo(
        &m,
        &plan,
        &[("p0".to_string(), vbits(7, Signedness::Unsigned, "1111111"))]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_neg_one["out"].to_bit_string_msb_first(), "00000001");

    let out_wrap_zero = eval_combo(
        &m,
        &plan,
        &[("p0".to_string(), vbits(7, Signedness::Unsigned, "1000000"))]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_wrap_zero["out"].to_bit_string_msb_first(), "10000000");
}

#[test]
fn indexed_assign_lhs_supports_bitvector_build_up() {
    let dut = r#"
module bitvector_lhs_build_up(
  input wire a,
  input wire b,
  input wire c,
  input wire d,
  output wire [3:0] out
);
  wire [3:0] v;
  assign v[0] = a;
  assign v[1] = b;
  assign v[2] = c;
  assign v[3] = d;
  assign out = v;
endmodule
"#;
    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();
    let out = eval_combo(
        &m,
        &plan,
        &[
            ("a".to_string(), vbits(1, Signedness::Unsigned, "1")),
            ("b".to_string(), vbits(1, Signedness::Unsigned, "0")),
            ("c".to_string(), vbits(1, Signedness::Unsigned, "1")),
            ("d".to_string(), vbits(1, Signedness::Unsigned, "0")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out["out"].to_bit_string_msb_first(), "0101");
}

#[test]
fn indexed_assign_lhs_supports_packed_array_build_up_with_cast() {
    let dut = r#"
module packed_lhs_build_up(
  input wire [1:0] lo,
  input wire [1:0] hi,
  output wire [3:0] out
);
  wire [1:0][1:0] p;
  assign p[0] = lo;
  assign p[1] = hi;
  assign out = $unsigned(p);
endmodule
"#;
    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();
    let out = eval_combo(
        &m,
        &plan,
        &[
            ("lo".to_string(), vbits(2, Signedness::Unsigned, "10")),
            ("hi".to_string(), vbits(2, Signedness::Unsigned, "01")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out["out"].to_bit_string_msb_first(), "0110");
}

#[test]
fn parses_and_evals_assignment_context_sized_ops() {
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
  output wire [7:0] ashl_out,
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
  assign ashl_out = 4'sb1000 <<< 1;
  assign lshr_out = 4'sb1000 >> 1;
  assign sshr_out = 4'sb1000 >>> 1;
  assign bor_ss_out = 4'sb1000 | 4'sb0001;
  assign bor_su_out = 4'sb1111 | 4'b0001;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let out = eval_combo(
        &m,
        &plan,
        &[("a".to_string(), vbits(5, Signedness::Unsigned, "01011"))]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();

    assert_eq!(out["mul_out"].to_bit_string_msb_first(), "0001111001");
    assert_eq!(out["nested_out"].to_bit_string_msb_first(), "0001111010");
    assert_eq!(out["add_out"].to_bit_string_msb_first(), "00010000");
    assert_eq!(out["sub_out"].to_bit_string_msb_first(), "11111111");
    assert_eq!(out["div_out"].to_bit_string_msb_first(), "00000110");
    assert_eq!(out["mod_out"].to_bit_string_msb_first(), "00000001");
    assert_eq!(out["neg_out"].to_bit_string_msb_first(), "11111111");
    assert_eq!(out["not_out"].to_bit_string_msb_first(), "11111110");
    assert_eq!(out["shl_out"].to_bit_string_msb_first(), "11110000");
    assert_eq!(out["ashl_out"].to_bit_string_msb_first(), "11110000");
    assert_eq!(out["lshr_out"].to_bit_string_msb_first(), "01111100");
    assert_eq!(out["sshr_out"].to_bit_string_msb_first(), "11111100");
    assert_eq!(out["bor_ss_out"].to_bit_string_msb_first(), "11111001");
    assert_eq!(out["bor_su_out"].to_bit_string_msb_first(), "00001111");
}

#[test]
fn parses_and_evals_question_mark_digits_as_z() {
    let dut = r#"
module qmark_v(
  output wire [3:0] out_bin,
  output wire [7:0] out_hex
);
  assign out_bin = 4'b10?1;
  assign out_hex = 8'h?f;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();
    let out = eval_combo(&m, &plan, &BTreeMap::new()).unwrap();

    assert_eq!(out["out_bin"].to_bit_string_msb_first(), "10z1");
    assert_eq!(out["out_hex"].to_bit_string_msb_first(), "zzzz1111");
}

#[test]
fn rejects_unsized_numeric_concat_and_dynamic_replication() {
    let bad_concat = r#"
module bad_concat_v(
  output wire [32:0] out
);
  assign out = {1, 1'b0};
endmodule
"#;
    assert!(compile_combo_module(bad_concat).is_err());

    let bad_repl = r#"
module bad_repl_v(
  input wire [1:0] n,
  output wire [1:0] out
);
  assign out = {n{1'b1}};
endmodule
"#;
    assert!(compile_combo_module(bad_repl).is_err());

    let bad_part_select = r#"
module bad_part_select_v(
  input wire [7:0] a,
  input wire [2:0] i,
  output wire [3:0] out
);
  assign out = a[i+3:i];
endmodule
"#;
    assert!(compile_combo_module(bad_part_select).is_err());

    let bad_indexed_width = r#"
module bad_indexed_width_v(
  input wire [7:0] a,
  input wire [2:0] w,
  output wire [3:0] out
);
  assign out = a[1 +: w];
endmodule
"#;
    assert!(compile_combo_module(bad_indexed_width).is_err());
}

#[test]
fn rejects_non_zero_based_decl_ranges() {
    let bad_packed = r#"
module bad_packed_decl_v(
  input wire [7:1] a,
  output wire [6:0] out
);
  assign out = a;
endmodule
"#;
    let err = compile_combo_module(bad_packed).unwrap_err();
    assert!(matches!(
        err,
        xlsynth_vastly::Error::Parse(msg)
            if msg.contains("packed declaration ranges must be zero-based")
    ));

    let bad_unpacked = r#"
module bad_unpacked_decl_v(
  input wire [1:0] in_data,
  output wire [1:0] out
);
  wire [1:0] lanes[0:1];
  assign lanes[0] = in_data;
  assign lanes[1] = in_data;
  assign out = lanes[0];
endmodule
"#;
    let err = compile_combo_module(bad_unpacked).unwrap_err();
    assert!(matches!(
        err,
        xlsynth_vastly::Error::Parse(msg)
            if msg.contains("unpacked declaration ranges must be zero-based")
    ));
}

#[test]
fn signed_casts_are_self_determined_before_assignment_context() {
    let dut = r#"
module signed_cast_ctx_v(
  input wire dummy,
  output wire [3:0] out
);
  assign out = $signed('1 + '1);
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();
    let out = eval_combo(
        &m,
        &plan,
        &[("dummy".to_string(), vbits(1, Signedness::Unsigned, "0"))]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();

    assert_eq!(out["out"].to_bit_string_msb_first(), "0000");
}

#[test]
fn parses_and_evals_function_helper_assignment_context_sized_ops() {
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

    let out = eval_combo(
        &m,
        &plan,
        &[("a".to_string(), vbits(5, Signedness::Unsigned, "01011"))]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();

    assert_eq!(
        out["direct_mul_out"].to_bit_string_msb_first(),
        "0001111001"
    );
    assert_eq!(
        out["nested_mul_out"].to_bit_string_msb_first(),
        "0001111010"
    );
    assert_eq!(out["tmp_mul_out"].to_bit_string_msb_first(), "0001111001");
}

#[test]
fn parses_and_evals_priority_sel_helper_with_decimal_x_default() {
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

    let out_zero = eval_combo(
        &m,
        &plan,
        &[("p0".to_string(), vbits(8, Signedness::Unsigned, "00000000"))]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_zero["out"].to_bit_string_msb_first(), "1");

    let out_lsb = eval_combo(
        &m,
        &plan,
        &[("p0".to_string(), vbits(8, Signedness::Unsigned, "00000001"))]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_lsb["out"].to_bit_string_msb_first(), "0");
}

#[test]
fn parses_and_evals_generated_dynamic_bit_slice_helper() {
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

    let out_lo = eval_combo(
        &m,
        &plan,
        &[
            ("x".to_string(), vbits(8, Signedness::Unsigned, "10110011")),
            ("s".to_string(), vbits(8, Signedness::Unsigned, "00000000")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_lo["out"].to_bit_string_msb_first(), "0011");

    let out_hi = eval_combo(
        &m,
        &plan,
        &[
            ("x".to_string(), vbits(8, Signedness::Unsigned, "10110011")),
            ("s".to_string(), vbits(8, Signedness::Unsigned, "00000100")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_hi["out"].to_bit_string_msb_first(), "1011");

    let out_oob = eval_combo(
        &m,
        &plan,
        &[
            ("x".to_string(), vbits(8, Signedness::Unsigned, "10110011")),
            ("s".to_string(), vbits(8, Signedness::Unsigned, "00001000")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_oob["out"].to_bit_string_msb_first(), "0000");
}

#[test]
fn parses_and_evals_generated_bit_slice_update_helper() {
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

    let out_zero = eval_combo(
        &m,
        &plan,
        &[
            ("x".to_string(), vbits(8, Signedness::Unsigned, "10110011")),
            ("s".to_string(), vbits(8, Signedness::Unsigned, "00000000")),
            ("u".to_string(), vbits(4, Signedness::Unsigned, "0101")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_zero["out"].to_bit_string_msb_first(), "10110101");

    let out_mid = eval_combo(
        &m,
        &plan,
        &[
            ("x".to_string(), vbits(8, Signedness::Unsigned, "10110011")),
            ("s".to_string(), vbits(8, Signedness::Unsigned, "00000010")),
            ("u".to_string(), vbits(4, Signedness::Unsigned, "0101")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_mid["out"].to_bit_string_msb_first(), "10010111");

    let out_oob = eval_combo(
        &m,
        &plan,
        &[
            ("x".to_string(), vbits(8, Signedness::Unsigned, "10110011")),
            ("s".to_string(), vbits(8, Signedness::Unsigned, "00001000")),
            ("u".to_string(), vbits(4, Signedness::Unsigned, "0101")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_oob["out"].to_bit_string_msb_first(), "10110011");
}

#[test]
fn parses_and_evals_generated_array_index_helper() {
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

    let out_mid = eval_combo(
        &m,
        &plan,
        &[
            (
                "arr".to_string(),
                vbits(32, Signedness::Unsigned, "01000100001100110010001000010001"),
            ),
            ("start".to_string(), vbits(3, Signedness::Unsigned, "010")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_mid["out"].to_bit_string_msb_first(), "00110011");

    let out_clamped = eval_combo(
        &m,
        &plan,
        &[
            (
                "arr".to_string(),
                vbits(32, Signedness::Unsigned, "01000100001100110010001000010001"),
            ),
            ("start".to_string(), vbits(3, Signedness::Unsigned, "111")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_clamped["out"].to_bit_string_msb_first(), "01000100");
}

#[test]
fn parses_and_evals_generated_array_slice_helper() {
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

    let out_zero = eval_combo(
        &m,
        &plan,
        &[
            (
                "arr".to_string(),
                vbits(32, Signedness::Unsigned, "01000100001100110010001000010001"),
            ),
            ("start".to_string(), vbits(3, Signedness::Unsigned, "000")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(
        out_zero["out"].to_bit_string_msb_first(),
        "001100110010001000010001"
    );

    let out_tail = eval_combo(
        &m,
        &plan,
        &[
            (
                "arr".to_string(),
                vbits(32, Signedness::Unsigned, "01000100001100110010001000010001"),
            ),
            ("start".to_string(), vbits(3, Signedness::Unsigned, "010")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(
        out_tail["out"].to_bit_string_msb_first(),
        "010001000100010000110011"
    );
}

#[test]
fn parses_and_evals_generated_array_update_helper() {
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

    let out_mid = eval_combo(
        &m,
        &plan,
        &[
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
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(
        out_mid["out"].to_bit_string_msb_first(),
        "01000100001100111010101000010001"
    );

    let out_oob = eval_combo(
        &m,
        &plan,
        &[
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
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(
        out_oob["out"].to_bit_string_msb_first(),
        "01000100001100110010001000010001"
    );
}

#[test]
fn parses_and_evals_signed_divmod_and_div_by_zero() {
    let dut = r#"
module divmod_mod_v(
  input wire signed [7:0] a,
  input wire signed [7:0] b,
  output wire [15:0] out
);
  wire signed [7:0] q;
  wire signed [7:0] r;
  assign q = a / b;
  assign r = a % b;
  assign out = {q, r};
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let out_pos_neg = eval_combo(
        &m,
        &plan,
        &[
            ("a".to_string(), vbits(8, Signedness::Signed, "00001101")),
            ("b".to_string(), vbits(8, Signedness::Signed, "11111011")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(
        out_pos_neg["out"].to_bit_string_msb_first(),
        "1111111000000011"
    );

    let out_neg_pos = eval_combo(
        &m,
        &plan,
        &[
            ("a".to_string(), vbits(8, Signedness::Signed, "11110011")),
            ("b".to_string(), vbits(8, Signedness::Signed, "00000101")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(
        out_neg_pos["out"].to_bit_string_msb_first(),
        "1111111011111101"
    );

    let out_neg_neg = eval_combo(
        &m,
        &plan,
        &[
            ("a".to_string(), vbits(8, Signedness::Signed, "11110011")),
            ("b".to_string(), vbits(8, Signedness::Signed, "11111011")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(
        out_neg_neg["out"].to_bit_string_msb_first(),
        "0000001011111101"
    );

    let out_div0 = eval_combo(
        &m,
        &plan,
        &[
            ("a".to_string(), vbits(8, Signedness::Signed, "00001101")),
            ("b".to_string(), vbits(8, Signedness::Signed, "00000000")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(
        out_div0["out"].to_bit_string_msb_first(),
        "xxxxxxxxxxxxxxxx"
    );
}

#[test]
fn parses_and_evals_mixed_width_and_signedness_multiply() {
    let dut = r#"
module mul_mod_v(
  input wire signed [3:0] a,
  input wire [7:0] b,
  output wire [7:0] p
);
  assign p = a * b;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let out_small_rhs = eval_combo(
        &m,
        &plan,
        &[
            ("a".to_string(), vbits(4, Signedness::Signed, "1101")),
            ("b".to_string(), vbits(8, Signedness::Unsigned, "00000010")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_small_rhs["p"].to_bit_string_msb_first(), "00011010");

    let out_wrap_rhs = eval_combo(
        &m,
        &plan,
        &[
            ("a".to_string(), vbits(4, Signedness::Signed, "1101")),
            ("b".to_string(), vbits(8, Signedness::Unsigned, "11111110")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_wrap_rhs["p"].to_bit_string_msb_first(), "11100110");
}

#[test]
fn parses_and_evals_mixed_signed_unsigned_sized_arithmetic_context() {
    let dut = r#"
module mixed_ctx_arith_v(
  input wire signed [3:0] a,
  input wire [7:0] b,
  output wire [7:0] p_mul
);
  assign p_mul = a * b;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let out = eval_combo(
        &m,
        &plan,
        &[
            ("a".to_string(), vbits(4, Signedness::Signed, "1101")),
            ("b".to_string(), vbits(8, Signedness::Unsigned, "00000010")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out["p_mul"].to_bit_string_msb_first(), "00011010");
}

#[test]
fn dynamic_select_unknown_indices_produce_unknown_results() {
    let dut = r#"
module dyn_sel_unknown_v(
  input wire [3:0] a,
  input wire [1:0] i,
  input wire [2:0] b,
  output wire y,
  output wire [2:0] p_up,
  output wire [2:0] p_down
);
  assign y = a[i];
  assign p_up = a[b +: 3];
  assign p_down = a[b -: 3];
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let out = eval_combo(
        &m,
        &plan,
        &[
            ("a".to_string(), vbits(4, Signedness::Unsigned, "1010")),
            ("i".to_string(), vbits(2, Signedness::Unsigned, "xx")),
            ("b".to_string(), vbits(3, Signedness::Unsigned, "x01")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out["y"].to_bit_string_msb_first(), "x");
    assert_eq!(out["p_up"].to_bit_string_msb_first(), "xxx");
    assert_eq!(out["p_down"].to_bit_string_msb_first(), "xxx");
}

#[test]
fn oob_lhs_indexed_assign_is_noop_for_bitvector() {
    let dut = r#"
module oob_lhs_bitvector(
  output wire [3:0] out
);
  wire [3:0] v;
  assign v = 4'b1010;
  assign v[7] = 1'b1;
  assign out = v;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();
    let out = eval_combo(&m, &plan, &BTreeMap::new()).unwrap();
    assert_eq!(out["out"].to_bit_string_msb_first(), "1010");
}

#[test]
fn oob_lhs_indexed_assign_is_noop_for_packed_array() {
    let dut = r#"
module oob_lhs_packed(
  output wire [7:0] out
);
  wire [1:0][3:0] p;
  assign p = 8'b10100011;
  assign p[2] = 4'b1111;
  assign out = p;
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();
    let out = eval_combo(&m, &plan, &BTreeMap::new()).unwrap();
    assert_eq!(out["out"].to_bit_string_msb_first(), "10100011");
}

#[test]
fn oob_rhs_index_read_returns_x_for_bitvector() {
    let dut = r#"
module oob_rhs_bitvector(
  input wire [3:0] a,
  input wire [2:0] idx,
  output wire y
);
  assign y = a[idx];
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();
    let out = eval_combo(
        &m,
        &plan,
        &[
            ("a".to_string(), vbits(4, Signedness::Unsigned, "1010")),
            ("idx".to_string(), vbits(3, Signedness::Unsigned, "100")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out["y"].to_bit_string_msb_first(), "x");
}

#[test]
fn oob_rhs_index_read_returns_x_for_packed_array() {
    let dut = r#"
module oob_rhs_packed(
  input wire [1:0][3:0] a,
  input wire [1:0] idx,
  output wire [3:0] y
);
  assign y = a[idx];
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();
    let out = eval_combo(
        &m,
        &plan,
        &[
            ("a".to_string(), vbits(8, Signedness::Unsigned, "10100011")),
            ("idx".to_string(), vbits(2, Signedness::Unsigned, "10")),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out["y"].to_bit_string_msb_first(), "xxxx");
}

#[test]
fn coverage_eval_supports_signed_unsigned_cast_builtins() {
    let dut = r#"
module cov_cast_builtin_v(
  input wire signed [3:0] a,
  input wire [3:0] b,
  output wire signed [4:0] y
);
  assign y = $signed(a) + $unsigned(b);
endmodule
"#;
    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();
    let mut cov = CoverageCounters::default();
    let src = SourceText::new(dut.to_string());
    let seed: BTreeMap<String, Value4> = [
        ("a".to_string(), vbits(4, Signedness::Signed, "1101")),
        ("b".to_string(), vbits(4, Signedness::Unsigned, "0011")),
    ]
    .into_iter()
    .collect();
    let mut env = xlsynth_vastly::Env::new();
    for (k, v) in &seed {
        env.insert(k.clone(), v.clone());
    }

    let out_cov =
        eval_combo_seeded_with_coverage(&m, &plan, &env, &src, &mut cov, &BTreeMap::new()).unwrap();
    let out_ref = eval_combo(&m, &plan, &seed).unwrap();
    assert_eq!(
        out_cov["y"].to_bit_string_msb_first(),
        out_ref["y"].to_bit_string_msb_first()
    );
}

#[test]
fn coverage_eval_matches_plain_eval_for_packed_index_in_function_assign() {
    let dut = r#"
module packed_fn_cov(
  input logic [1:0][3:0] a,
  input logic idx,
  output logic [3:0] y
);
  function automatic logic [3:0] pick(input logic i);
    begin
      pick = a[i];
    end
  endfunction
  assign y = pick(idx);
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();
    let seed: BTreeMap<String, Value4> = [
        ("a".to_string(), vbits(8, Signedness::Unsigned, "10100011")),
        ("idx".to_string(), vbits(1, Signedness::Unsigned, "1")),
    ]
    .into_iter()
    .collect();
    let out_ref = eval_combo(&m, &plan, &seed).unwrap();

    let src = SourceText::new(dut.to_string());
    let mut cov = CoverageCounters::default();
    let mut env = xlsynth_vastly::Env::new();
    for (k, v) in &seed {
        env.insert(k.clone(), v.clone());
    }
    let out_cov =
        eval_combo_seeded_with_coverage(&m, &plan, &env, &src, &mut cov, &BTreeMap::new()).unwrap();

    assert_eq!(out_ref["y"].to_bit_string_msb_first(), "1010");
    assert_eq!(
        out_cov["y"].to_bit_string_msb_first(),
        out_ref["y"].to_bit_string_msb_first()
    );
}

#[test]
fn coverage_eval_preserves_unknown_dynamic_selectors() {
    let dut = r#"
module cov_dyn_sel_unknown_v(
  input wire [3:0] a,
  input wire [1:0] i,
  input wire [2:0] b,
  output wire y,
  output wire [2:0] p
);
  assign y = a[i];
  assign p = a[b +: 3];
endmodule
"#;
    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();
    let mut cov = CoverageCounters::default();
    let src = SourceText::new(dut.to_string());
    let mut env = xlsynth_vastly::Env::new();
    env.insert("a", vbits(4, Signedness::Unsigned, "1010"));
    env.insert("i", vbits(2, Signedness::Unsigned, "x1"));
    env.insert("b", vbits(3, Signedness::Unsigned, "x01"));

    let out =
        eval_combo_seeded_with_coverage(&m, &plan, &env, &src, &mut cov, &BTreeMap::new()).unwrap();
    assert_eq!(out["y"].to_bit_string_msb_first(), "x");
    assert_eq!(out["p"].to_bit_string_msb_first(), "xxx");
}

#[test]
fn parses_and_evals_generated_unpacked_array_combo() {
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

    let out = eval_combo(
        &m,
        &plan,
        &[("p0".to_string(), vbits(3, Signedness::Unsigned, "101"))]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out["out"].to_bit_string_msb_first(), "00000000101");
    assert_eq!(out["array_12"].to_bit_string_msb_first(), "101101");
}

#[test]
fn parses_and_evals_generated_sv_unpacked_array_combo() {
    let dut = r#"
module fuzz_codegen_sv(
  input wire [2:0] p0,
  output wire [10:0] out
);
  wire [2:0] tuple_11;
  wire [5:0] tuple_13;
  wire [2:0] array_12[2];
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

    let out = eval_combo(
        &m,
        &plan,
        &[("p0".to_string(), vbits(3, Signedness::Unsigned, "011"))]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out["out"].to_bit_string_msb_first(), "00000000011");
}

#[test]
fn parses_and_evals_generated_nested_unpacked_array_combo() {
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

    let out = eval_combo(
        &m,
        &plan,
        &[("p0".to_string(), vbits(2, Signedness::Unsigned, "01"))]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out["out"].to_bit_string_msb_first(), "10");
    assert_eq!(out["array_11"].to_bit_string_msb_first(), "01101001");
    assert_eq!(out["sel_12"].to_bit_string_msb_first(), "01101001");
}

#[test]
fn parses_and_evals_generated_sv_whole_array_assign_combo() {
    let dut = r#"
module fuzz_codegen_sv(
  input wire [1:0] p0,
  output wire [1:0] out
);
  wire [1:0] array_10[2];
  wire [1:0] array_11[2][2];
  wire [1:0] array_12[2][2];
  wire [1:0] sel_13[2][2];
  assign array_10[0] = p0;
  assign array_10[1] = ~p0;
  assign array_11[0] = array_10;
  assign array_11[1] = array_10;
  assign array_12[0][0] = ~p0;
  assign array_12[0][1] = p0;
  assign array_12[1][0] = p0;
  assign array_12[1][1] = ~p0;
  assign sel_13 = p0 == 2'h0 ? array_11 : array_12;
  assign out = sel_13[1][0];
endmodule
"#;

    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();

    let out_nonzero = eval_combo(
        &m,
        &plan,
        &[("p0".to_string(), vbits(2, Signedness::Unsigned, "01"))]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_nonzero["out"].to_bit_string_msb_first(), "01");
    assert_eq!(out_nonzero["sel_13"].to_bit_string_msb_first(), "10010110");

    let out_zero = eval_combo(
        &m,
        &plan,
        &[("p0".to_string(), vbits(2, Signedness::Unsigned, "00"))]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out_zero["out"].to_bit_string_msb_first(), "00");
    assert_eq!(out_zero["array_11"].to_bit_string_msb_first(), "11001100");
}

#[test]
fn combo_input_is_coerced_to_declared_signedness() {
    let dut = r#"
module signed_passthrough(
  input wire signed [3:0] a,
  output wire signed [3:0] out
);
  assign out = a;
endmodule
"#;
    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();
    // Provide the same raw bits but mark value as unsigned at callsite.
    // The evaluator should coerce to the declared signedness of `a`.
    let out = eval_combo(
        &m,
        &plan,
        &[("a".to_string(), vbits(4, Signedness::Unsigned, "1101"))]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out["a"].signedness, Signedness::Signed);
    assert_eq!(out["out"].signedness, Signedness::Signed);
    assert_eq!(out["out"].to_bit_string_msb_first(), "1101");
}

#[test]
fn casez_treats_selector_z_bits_as_wildcards() {
    let dut = r#"
module casez_selz(
  input wire sel,
  output wire out
);
  function automatic logic pick (input reg sel_i);
    begin
      casez (sel_i)
        1'b0: begin
          pick = 1'b1;
        end
        default: begin
          pick = 1'b0;
        end
      endcase
    end
  endfunction
  assign out = pick(sel);
endmodule
"#;
    let m = compile_combo_module(dut).unwrap();
    let plan = plan_combo_eval(&m).unwrap();
    // `casez` should treat selector Z as wildcard and match the first arm.
    let out = eval_combo(
        &m,
        &plan,
        &[("sel".to_string(), vbits(1, Signedness::Unsigned, "z"))]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    assert_eq!(out["out"].to_bit_string_msb_first(), "1");
}
