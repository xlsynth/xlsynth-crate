// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "iverilog-tests")]

mod iverilog_oracle;

use xlsynth_vastly::Env;
use xlsynth_vastly::LogicBit;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::Value4;

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

fn assert_eval_matches_oracle(expr: &str, env: &Env) {
    let ours = xlsynth_vastly::eval_expr(expr, env).expect("eval_expr");
    let oracle = iverilog_oracle::run_oracle(expr, env);

    assert_eq!(ours.value.width, oracle.width, "width mismatch for {expr}");
    assert_eq!(
        ours.value.to_bit_string_msb_first(),
        oracle.value_bits_msb,
        "value mismatch for {expr}"
    );
}

#[test]
fn ternary_cond_0_1_x_z() {
    let env = Env::new();

    assert_eval_matches_oracle("1 ? 4'b0011 : 4'b1100", &env);
    assert_eval_matches_oracle("0 ? 4'b0011 : 4'b1100", &env);
    assert_eval_matches_oracle("1'bx ? 4'b0011 : 4'b0001", &env);
    assert_eval_matches_oracle("1'bz ? 4'b0011 : 4'b0001", &env);
}

#[test]
fn ternary_mixed_signedness_arithmetic_branch_uses_merged_type_matches_oracle() {
    let env = Env::new();

    assert_eval_matches_oracle(
        "(1'b0 ? 96'b101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010 : (32'sb00000000000000000000000000000100 - 80'b10101010101010101010101010101010101010101010101010101010101010101010101010101010))",
        &env,
    );
    assert_eval_matches_oracle(
        "(1'b1 ? (32'sb00000000000000000000000000000100 - 80'b10101010101010101010101010101010101010101010101010101010101010101010101010101010) : 96'b101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010)",
        &env,
    );
    assert_eval_matches_oracle(
        "(1'bx ? 96'b101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010 : (32'sb00000000000000000000000000000100 - 80'b10101010101010101010101010101010101010101010101010101010101010101010101010101010))",
        &env,
    );
}

#[test]
fn equality_vs_case_equality_with_unknowns() {
    let env = Env::new();

    assert_eval_matches_oracle("4'b10x1 == 4'b10x1", &env);
    assert_eval_matches_oracle("4'b10x1 != 4'b10x1", &env);
    assert_eval_matches_oracle("4'b10x1 === 4'b10x1", &env);
    assert_eval_matches_oracle("4'b10x1 !== 4'b10x1", &env);

    assert_eval_matches_oracle("4'b10x1 === 4'b10z1", &env);
    assert_eval_matches_oracle("4'b10x1 !== 4'b10z1", &env);
}

#[test]
fn unbased_unsized_literals_match_oracle_in_equality_contexts() {
    let env = Env::new();

    assert_eval_matches_oracle("12'b110111011011 == 'x", &env);
    assert_eval_matches_oracle("12'b110111011011 == 'z", &env);
    assert_eval_matches_oracle("12'b110111011011 != 'x", &env);
    assert_eval_matches_oracle("12'b110111011011 != 'z", &env);
    assert_eval_matches_oracle("12'b000000000000 === '0", &env);
    assert_eval_matches_oracle("12'b111111111111 === '1", &env);
    assert_eval_matches_oracle("'x === 12'bxxxxxxxxxxxx", &env);
    assert_eval_matches_oracle("'z === 12'bzzzzzzzzzzzz", &env);
}

#[test]
fn based_literal_sizing_and_unknown_fill_match_oracle() {
    let env = Env::new();

    assert_eval_matches_oracle("'h1", &env);
    assert_eval_matches_oracle("'hx", &env);
    assert_eval_matches_oracle("'bz", &env);
    assert_eval_matches_oracle("'d4294967296", &env);
    assert_eval_matches_oracle("8'hx", &env);
    assert_eval_matches_oracle("4'b10?1", &env);
    assert_eval_matches_oracle("8'h?f", &env);
}

#[test]
fn unsized_decimal_literals_expand_to_minimum_required_width_matches_oracle() {
    let env = Env::new();

    assert_eval_matches_oracle("4294967296 === 34'd4294967296", &env);
    assert_eval_matches_oracle("'d4294967296 === 33'd4294967296", &env);
}

#[test]
fn illegal_unsized_concat_and_dynamic_replication_are_rejected() {
    let env = Env::new();

    assert!(xlsynth_vastly::eval_expr("{1, 1'b0}", &env).is_err());
    assert!(xlsynth_vastly::eval_expr("{foo{1'b1}}", &env).is_err());
}

#[test]
fn mixed_signedness_equality_uses_common_context_matches_oracle() {
    let env = Env::new();

    assert_eval_matches_oracle("4'sb1111 == 8'd255", &env);
    assert_eval_matches_oracle("4'sb1111 === 8'd255", &env);
    assert_eval_matches_oracle("4'sb1111 == 8'd15", &env);
    assert_eval_matches_oracle("4'sb1111 === 8'd15", &env);
    assert_eval_matches_oracle("((4'sb1111 + 4'sd1) == 8'd16)", &env);
    assert_eval_matches_oracle("((4'sb1111 + 4'sd1) === 8'd16)", &env);
}

#[test]
fn logical_ops_4state_truth_tables() {
    let env = Env::new();

    assert_eval_matches_oracle("4'b0000 && 1'bx", &env);
    assert_eval_matches_oracle("4'b0001 && 1'bx", &env);
    assert_eval_matches_oracle("4'b0000 || 1'bx", &env);
    assert_eval_matches_oracle("4'b1000 || 1'bx", &env);
}

#[test]
fn identifiers_flow_through() {
    let mut env = Env::new();
    env.insert("a", vbits(1, Signedness::Unsigned, "x"));
    env.insert("b", vbits(4, Signedness::Unsigned, "0011"));
    env.insert("c", vbits(4, Signedness::Unsigned, "0001"));

    assert_eval_matches_oracle("a ? b : c", &env);
    assert_eval_matches_oracle("!a ? b : c", &env);
}

#[test]
fn signedness_observable_via_extension_for_literals() {
    let env = Env::new();

    // 4'sb1000 should be signed, 4'b1000 should be unsigned.
    let o_signed = iverilog_oracle::run_oracle("4'sb1000", &env);
    let o_unsigned = iverilog_oracle::run_oracle("4'b1000", &env);

    let ours_signed = xlsynth_vastly::eval_expr("4'sb1000", &env)
        .unwrap()
        .value
        .signedness;
    let ours_unsigned = xlsynth_vastly::eval_expr("4'b1000", &env)
        .unwrap()
        .value
        .signedness;

    assert_eq!(ours_signed, Signedness::Signed);
    assert_eq!(ours_unsigned, Signedness::Unsigned);

    assert_eq!(
        iverilog_oracle::infer_signedness_from_ext(&o_signed),
        Some(true)
    );
    assert_eq!(
        iverilog_oracle::infer_signedness_from_ext(&o_unsigned),
        Some(false)
    );
}

#[test]
fn wide_dynamic_index_and_slice_use_known_low_bits() {
    let mut env = Env::new();
    env.insert("x", vbits(8, Signedness::Unsigned, "10100101"));

    assert_eval_matches_oracle("x[200'h5]", &env);
    assert_eval_matches_oracle("x[200'h7:200'h5]", &env);
}

#[test]
fn wide_decimal_literal_parses_beyond_u64() {
    let env = Env::new();

    assert_eval_matches_oracle("200'd18446744073709551616", &env);
}

#[test]
fn signed_and_unsigned_cast_builtins_match_oracle() {
    let env = Env::new();

    assert_eval_matches_oracle("$signed(7'b1111111) >>> 1", &env);
    assert_eval_matches_oracle("$unsigned($signed(7'b1111111)) >>> 1", &env);
}

#[test]
fn decimal_unknown_literals_match_oracle() {
    let env = Env::new();

    assert_eval_matches_oracle("8'dx", &env);
    assert_eval_matches_oracle("8'dz", &env);
}

#[test]
fn divide_and_mod_by_zero_match_oracle() {
    let env = Env::new();

    assert_eval_matches_oracle("8'd13 / 8'd0", &env);
    assert_eval_matches_oracle("8'd13 % 8'd0", &env);
    assert_eval_matches_oracle("8'sd13 / 8'sd0", &env);
    assert_eval_matches_oracle("8'sd13 % 8'sd0", &env);
}

#[test]
fn signed_division_and_modulo_with_negative_divisors_match_oracle() {
    let env = Env::new();

    assert_eval_matches_oracle("8'sd13 / -8'sd5", &env);
    assert_eval_matches_oracle("8'sd13 % -8'sd5", &env);
    assert_eval_matches_oracle("-8'sd13 / 8'sd5", &env);
    assert_eval_matches_oracle("-8'sd13 % 8'sd5", &env);
    assert_eval_matches_oracle("-8'sd13 / -8'sd5", &env);
    assert_eval_matches_oracle("-8'sd13 % -8'sd5", &env);
}

#[test]
fn division_and_modulo_match_oracle_for_mixed_widths() {
    let env = Env::new();

    assert_eval_matches_oracle("(-4'sd3) / 8'sd2", &env);
    assert_eval_matches_oracle("(-4'sd3) % 8'sd2", &env);
    assert_eval_matches_oracle("(-8'sd13) / 4'sd5", &env);
    assert_eval_matches_oracle("(-8'sd13) % 4'sd5", &env);
}

#[test]
fn division_and_modulo_match_oracle_for_mixed_signedness() {
    let env = Env::new();

    assert_eval_matches_oracle("(-4'sd3) / 8'd2", &env);
    assert_eval_matches_oracle("(-4'sd3) % 8'd2", &env);
    assert_eval_matches_oracle("(4'd13) / -8'sd2", &env);
    assert_eval_matches_oracle("(4'd13) % -8'sd2", &env);
}

#[test]
fn multiplication_matches_oracle_for_mixed_widths() {
    let env = Env::new();

    assert_eval_matches_oracle("(-4'sd3) * 8'sd2", &env);
    assert_eval_matches_oracle("(-8'sd13) * 4'sd5", &env);
    assert_eval_matches_oracle("(-4'sd3) * -8'sd2", &env);
}

#[test]
fn multiplication_matches_oracle_for_mixed_signedness() {
    let env = Env::new();

    assert_eval_matches_oracle("(-4'sd3) * 8'd2", &env);
    assert_eval_matches_oracle("(4'd13) * -8'sd2", &env);
    assert_eval_matches_oracle("(-8'sd13) * 4'd5", &env);
}

#[test]
fn subtraction_matches_oracle_for_mixed_signedness_with_wide_unary_rhs() {
    let env = Env::new();

    assert_eval_matches_oracle(
        "(32'sb10001100001111101101000111001110-(-88'b1010101010101010101010101010101010101010101010101010101010101010101010101010101010101010))",
        &env,
    );
}

#[test]
fn chained_mod_and_subtraction_matches_oracle_with_wide_unary_rhs() {
    let env = Env::new();

    assert_eval_matches_oracle(
        "(32'sb11111100101110100000000111000111%32'sb00000000000000000000000000001000)",
        &env,
    );
    assert_eval_matches_oracle(
        "((32'sb11111100101110100000000111000111%32'sb00000000000000000000000000001000)-(-88'b0000000000000000000000000000000000000000000000000000110100010001000100010001000100011000))",
        &env,
    );
    assert_eval_matches_oracle(
        "(((32'sb11111100101110100000000111000111%32'sb00000000000000000000000000001000)-(-88'b0000000000000000000000000000000000000000000000000000110100010001000100010001000100011000))-(-88'b1010101010101010101010101010101010101010101010101010101010101010101010101010101010101010))",
        &env,
    );
}

#[test]
fn relational_with_nested_unary_chain_matches_oracle_in_wide_xor_context() {
    let env = Env::new();

    assert_eval_matches_oracle(
        "((32'sb00000000000000000000000000001001<(~(!(!(!(!(!(!(!32'sb00000000000000000000000000001001)))))))))^44'b10101001101010101010101010101010101010101010)",
        &env,
    );
}

#[test]
fn relational_mixed_signedness_zero_extends_signed_operand_before_compare() {
    let env = Env::new();

    assert_eval_matches_oracle(
        "(32'sb11100011100011100011100011100111<(32'sb00000000000000000000000000001001-44'b10101001101010101010101010101010101010101010))",
        &env,
    );
}

#[test]
fn relational_signed_shift_with_wide_z_xor_context_matches_oracle() {
    let env = Env::new();

    assert_eval_matches_oracle(
        "((((-32'sb00000000000000000000000000010110)>>32'sb00000000000000000000000000100111)>32'sb00000000000000000000000110000001)^448'bzzzz101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010)",
        &env,
    );
}

#[test]
fn relational_bitnot_signed_compare_in_bitwise_or_context_matches_oracle() {
    let env = Env::new();

    assert_eval_matches_oracle(
        "(((32'sb00100111101111001000011010101010^32'sb00000000000000000010001010111000)|32'sb00000000000000000000000001011000)|((~32'sb00000000000000010101101100111000)>32'sb00000000000011011001000000111000))",
        &env,
    );
}

#[test]
fn ternary_known_cond_still_uses_merged_branch_width_matches_oracle() {
    let env = Env::new();

    assert_eval_matches_oracle(
        "((32'sb11110000100110111011100000000110)?(32'sb00000000000000000000000000000000):(((32'sb00000000000000000000000000010110-32'sb00000000000000000000000000001000)&224'b11011011101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010)))",
        &env,
    );
}

#[test]
fn shift_amount_division_ignores_parent_signedness_matches_oracle() {
    let env = Env::new();

    assert_eval_matches_oracle(
        "(((+(^~(+(^~(+32'sb00000000000000000000000000001000)))))-(-32'sb00000000000000000000000000001000))^((5'b11000-(-85'b0000000000000000000000000000000000000000000001101101100010001000100010001000110100111))>>(32'sb00000000000000000000010001011110/(~32'sb00000000000000000000000000001011))))",
        &env,
    );
}
