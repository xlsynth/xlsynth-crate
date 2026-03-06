// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "iverilog-tests")]

// This file provides deterministic, oracle-backed "context sweep" coverage for
// expression semantics that frequently break under width/signedness inference.
//
// Why this exists:
// - The `diff_iverilog` fuzzer is excellent at finding edge cases, but it is
//   stochastic and can regress silently if a previously-found corner case
//   becomes harder to rediscover.
// - These sweeps pin down families of context-sensitive rules as stable,
//   reproducible integration tests against an external oracle (`iverilog`).
//
// How this is structured:
// - Generate bounded cartesian-style combinations across widths, signedness,
//   operators, and parent-context forcing expressions.
// - Keep runtime CI-friendly with deterministic subsampling in large matrices.
// - Assert both width and value bits against oracle results for each
//   expression.

mod iverilog_oracle;

use xlsynth_vastly::Env;
use xlsynth_vastly::LogicBit;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::Value4;

fn bit_pattern(width: u32, seed: u32, force_msb_one: bool) -> String {
    let mut out = String::with_capacity(width as usize);
    for i in 0..width {
        let v = seed
            .wrapping_mul(1_664_525)
            .wrapping_add((i + 1).wrapping_mul(1_013_904_223));
        let bit = if (v & 0x3) == 0 { '1' } else { '0' };
        out.push(bit);
    }
    if force_msb_one && !out.is_empty() {
        out.replace_range(0..1, "1");
    }
    out
}

fn mk_lit(width: u32, signed: bool, seed: u32, force_msb_one: bool) -> String {
    let signed_ch = if signed { "s" } else { "" };
    let bits = bit_pattern(width, seed, force_msb_one);
    format!("{width}'{signed_ch}b{bits}")
}

fn mk_u_dec(width: u32, value: u32) -> String {
    format!("{width}'d{value}")
}

fn assert_eval_matches_oracle(expr: &str, label: &str) {
    let env = Env::new();
    assert_eval_matches_oracle_with_env(expr, &env, label);
}

fn assert_eval_matches_oracle_with_env(expr: &str, env: &Env, label: &str) {
    let ours = xlsynth_vastly::eval_expr(expr, &env).expect("eval_expr");
    let oracle = iverilog_oracle::run_oracle(expr, &env);
    assert_eq!(
        ours.value.width, oracle.width,
        "width mismatch [{label}] expr={expr}"
    );
    assert_eq!(
        ours.value.to_bit_string_msb_first(),
        oracle.value_bits_msb,
        "value mismatch [{label}] expr={expr}"
    );
}

fn mk_unsigned_value_from_msb(bits_msb: &str) -> Value4 {
    let mut bits_lsb = Vec::with_capacity(bits_msb.len());
    for c in bits_msb.chars().rev() {
        let b = match c {
            '0' => LogicBit::Zero,
            '1' => LogicBit::One,
            'x' | 'X' => LogicBit::X,
            'z' | 'Z' => LogicBit::Z,
            _ => panic!("bad bit char in mk_unsigned_value_from_msb: {c}"),
        };
        bits_lsb.push(b);
    }
    Value4::new(bits_msb.len() as u32, Signedness::Unsigned, bits_lsb)
}

#[test]
fn mixed_signedness_equality_context_sweep_matches_oracle() {
    for &lhs_w in &[4_u32, 8, 12] {
        for &rhs_w in &[8_u32, 16, 33] {
            if rhs_w <= lhs_w {
                continue;
            }
            let lhs = format!("{}'sb{}", lhs_w, "1".repeat(lhs_w as usize));
            let rhs_all_ones = format!("{}'b{}", rhs_w, "1".repeat(rhs_w as usize));
            let rhs_low_ones = mk_u_dec(rhs_w, (1_u32 << lhs_w) - 1);
            let rhs_sum = mk_u_dec(rhs_w, 1_u32 << lhs_w);

            let direct_false = format!("(({lhs}) == ({rhs_all_ones}))");
            let direct_case_false = format!("(({lhs}) === ({rhs_all_ones}))");
            let direct_true = format!("(({lhs}) == ({rhs_low_ones}))");
            let direct_case_true = format!("(({lhs}) === ({rhs_low_ones}))");
            let arith_true = format!("(((({lhs}) + 4'sd1)) == ({rhs_sum}))");
            let arith_case_true = format!("(((({lhs}) + 4'sd1)) === ({rhs_sum}))");

            let false_label = format!("mixed-eq-false lw={lhs_w} rw={rhs_w}");
            let true_label = format!("mixed-eq-true lw={lhs_w} rw={rhs_w}");
            let arith_label = format!("mixed-eq-arith lw={lhs_w} rw={rhs_w}");

            assert_eval_matches_oracle(&direct_false, &false_label);
            assert_eval_matches_oracle(&direct_case_false, &false_label);
            assert_eval_matches_oracle(&direct_true, &true_label);
            assert_eval_matches_oracle(&direct_case_true, &true_label);
            assert_eval_matches_oracle(&arith_true, &arith_label);
            assert_eval_matches_oracle(&arith_case_true, &arith_label);
        }
    }
}

#[test]
fn arithmetic_context_sweep_matches_oracle() {
    let widths = [1_u32, 8, 32, 33];
    let parent_widths = [8_u32, 32, 44];
    let ops = ["+", "-", "*", "/", "%"];
    let mut ordinal = 0_u32;

    for (op_i, op) in ops.iter().enumerate() {
        for &lhs_w in &widths {
            for &rhs_w in &widths {
                for &lhs_signed in &[false, true] {
                    for &rhs_signed in &[false, true] {
                        for &parent_w in &parent_widths {
                            ordinal = ordinal.wrapping_add(1);
                            // Bound runtime while still sampling a wide cartesian surface.
                            if ordinal % 5 != 0 {
                                continue;
                            }
                            let lhs =
                                mk_lit(lhs_w, lhs_signed, 100 + ordinal + op_i as u32, lhs_signed);
                            let rhs =
                                mk_lit(rhs_w, rhs_signed, 200 + ordinal + op_i as u32, rhs_signed);
                            let expr = format!("({parent_w}'b0 + (({lhs}) {op} ({rhs})))");
                            let label = format!(
                                "arith op={op} lhs_w={lhs_w} rhs_w={rhs_w} lhs_s={lhs_signed} rhs_s={rhs_signed} pw={parent_w}"
                            );
                            assert_eval_matches_oracle(&expr, &label);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn relational_context_sweep_matches_oracle() {
    let widths = [1_u32, 8, 32, 33, 44];
    let rel_ops = ["<", "<=", ">", ">="];
    let mut ordinal = 0_u32;

    for (op_i, op) in rel_ops.iter().enumerate() {
        for &lhs_w in &widths {
            for &rhs_w in &widths {
                for &lhs_signed in &[false, true] {
                    for &rhs_signed in &[false, true] {
                        ordinal = ordinal.wrapping_add(1);
                        if ordinal % 4 != 0 {
                            continue;
                        }
                        let lhs =
                            mk_lit(lhs_w, lhs_signed, 300 + ordinal + op_i as u32, lhs_signed);
                        let rhs_a =
                            mk_lit(rhs_w, rhs_signed, 400 + ordinal + op_i as u32, rhs_signed);
                        let rhs_b = mk_lit(rhs_w, false, 500 + ordinal + op_i as u32, false);
                        let rhs_expr = format!("(({rhs_a}) - ({rhs_b}))");
                        let expr = format!("(({lhs}) {op} {rhs_expr})");
                        let label = format!(
                            "rel op={op} lhs_w={lhs_w} rhs_w={rhs_w} lhs_s={lhs_signed} rhs_s={rhs_signed}"
                        );
                        assert_eval_matches_oracle(&expr, &label);
                    }
                }
            }
        }
    }
}

#[test]
fn unary_chain_context_sweep_matches_oracle() {
    let widths = [8_u32, 32, 44];
    let mut ordinal = 0_u32;

    for &base_w in &widths {
        for &parent_w in &[32_u32, 44] {
            for &signed in &[false, true] {
                ordinal = ordinal.wrapping_add(1);
                let base = mk_lit(base_w, signed, 600 + ordinal, signed);
                let chain = format!("(~(!(!(!(!({base}))))))");
                let rhs = mk_lit(parent_w, false, 700 + ordinal, false);
                let expr = format!("((({chain}) < ({rhs})) ^ ({rhs}))");
                let label =
                    format!("unary-chain base_w={base_w} parent_w={parent_w} signed={signed}");
                assert_eval_matches_oracle(&expr, &label);
            }
        }
    }
}

#[test]
fn equality_and_case_equality_unknown_bit_sweep_matches_oracle() {
    let values = [
        "8'b00001111",
        "8'b11110000",
        "8'b10x1z0x1",
        "8'b10z1x0z1",
        "8'bxxxxxxxx",
        "8'bzzzzzzzz",
    ];
    let eq_ops = ["==", "!=", "===", "!=="];
    let mut ordinal = 0_u32;

    for &lhs in &values {
        for &rhs in &values {
            for (op_i, op) in eq_ops.iter().enumerate() {
                ordinal = ordinal.wrapping_add(1);
                // Keep this matrix bounded but deterministic.
                if ordinal % 3 != 0 {
                    continue;
                }
                let parent = mk_lit(16, false, 800 + ordinal + op_i as u32, false);
                let expr = format!("((({lhs}) {op} ({rhs})) ^ ({parent}))");
                let label = format!("eq op={op} lhs={lhs} rhs={rhs}");
                assert_eval_matches_oracle(&expr, &label);
            }
        }
    }
}

#[test]
fn unbased_unsized_equality_context_sweep_matches_oracle() {
    let lhs_values = [
        "1'b0",
        "1'b1",
        "8'b00001111",
        "8'b11110000",
        "12'b110111011011",
        "12'bxxxxxxxxxxxx",
        "12'bzzzzzzzzzzzz",
    ];
    let rhs_values = ["'0", "'1", "'x", "'z"];
    let eq_ops = ["==", "!=", "===", "!=="];
    let mut ordinal = 0_u32;

    for &lhs in &lhs_values {
        for &rhs in &rhs_values {
            for (op_i, op) in eq_ops.iter().enumerate() {
                ordinal = ordinal.wrapping_add(1);
                let parent = mk_lit(16, false, 850 + ordinal + op_i as u32, false);
                let expr = format!("((({lhs}) {op} ({rhs})) ^ ({parent}))");
                let label = format!("unbased-unsized lhs={lhs} rhs={rhs} op={op}");
                assert_eval_matches_oracle(&expr, &label);

                let swapped = format!("((({rhs}) {op} ({lhs})) ^ ({parent}))");
                let swapped_label = format!("unbased-unsized lhs={rhs} rhs={lhs} op={op}");
                assert_eval_matches_oracle(&swapped, &swapped_label);
            }
        }
    }
}

#[test]
fn shift_edge_sweep_matches_oracle() {
    let widths = [8_u32, 32, 33, 44];
    let shift_ops = ["<<", ">>", ">>>"];
    let shift_amounts = [0_u32, 1, 7, 8, 31, 32, 33, 43, 44, 45];
    let mut ordinal = 0_u32;

    for (op_i, op) in shift_ops.iter().enumerate() {
        for &lhs_w in &widths {
            for &lhs_signed in &[false, true] {
                for &sh in &shift_amounts {
                    ordinal = ordinal.wrapping_add(1);
                    if ordinal % 2 != 0 {
                        continue;
                    }
                    let lhs = mk_lit(lhs_w, lhs_signed, 900 + ordinal + op_i as u32, lhs_signed);
                    let sh_lit = mk_u_dec(8, sh);
                    let parent = mk_lit(64, false, 1000 + ordinal + op_i as u32, false);
                    let expr = format!("((({lhs}) {op} ({sh_lit})) ^ ({parent}))");
                    let label = format!("shift op={op} lhs_w={lhs_w} lhs_s={lhs_signed} sh={sh}");
                    assert_eval_matches_oracle(&expr, &label);
                }
            }
        }
    }
}

#[test]
fn ternary_unknown_cond_context_sweep_matches_oracle() {
    let conds = ["1'b0", "1'b1", "1'bx", "1'bz"];
    let mut ordinal = 0_u32;

    for &t_signed in &[false, true] {
        for &f_signed in &[false, true] {
            for &t_w in &[8_u32, 32, 44] {
                for &f_w in &[8_u32, 33, 44] {
                    for &cond in &conds {
                        ordinal = ordinal.wrapping_add(1);
                        if ordinal % 3 != 0 {
                            continue;
                        }
                        let t = mk_lit(t_w, t_signed, 1100 + ordinal, t_signed);
                        let f = mk_lit(f_w, f_signed, 1200 + ordinal, f_signed);
                        let parent = mk_lit(48, false, 1300 + ordinal, false);
                        let expr = format!("((({cond}) ? ({t}) : ({f})) ^ ({parent}))");
                        let label = format!(
                            "ternary cond={cond} tw={t_w} fw={f_w} ts={t_signed} fs={f_signed}"
                        );
                        assert_eval_matches_oracle(&expr, &label);
                    }
                }
            }
        }
    }
}

#[test]
fn relational_mixed_signedness_broad_sweep_matches_oracle() {
    let rel_ops = ["<", "<=", ">", ">="];
    let mut ordinal = 0_u32;

    for (op_i, op) in rel_ops.iter().enumerate() {
        for &lhs_w in &[8_u32, 32, 44] {
            for &rhs_w in &[8_u32, 33, 44] {
                for &lhs_signed in &[false, true] {
                    for &rhs_signed in &[false, true] {
                        ordinal = ordinal.wrapping_add(1);
                        if ordinal % 2 != 0 {
                            continue;
                        }
                        let lhs =
                            mk_lit(lhs_w, lhs_signed, 1400 + ordinal + op_i as u32, lhs_signed);
                        let rhs_a =
                            mk_lit(rhs_w, rhs_signed, 1500 + ordinal + op_i as u32, rhs_signed);
                        let rhs_b = mk_lit(rhs_w, false, 1600 + ordinal + op_i as u32, false);
                        let expr = format!("(({lhs}) {op} (({rhs_a}) - ({rhs_b})))");
                        let label = format!(
                            "rel-mixed op={op} lw={lhs_w} rw={rhs_w} ls={lhs_signed} rs={rhs_signed}"
                        );
                        assert_eval_matches_oracle(&expr, &label);
                    }
                }
            }
        }
    }
}

#[test]
fn cast_interaction_sweep_matches_oracle() {
    let mut ordinal = 0_u32;

    for &base_w in &[8_u32, 32, 44] {
        for &rhs_w in &[8_u32, 33, 44] {
            for &signed in &[false, true] {
                ordinal = ordinal.wrapping_add(1);
                let base = mk_lit(base_w, signed, 1700 + ordinal, signed);
                let rhs = mk_lit(rhs_w, false, 1800 + ordinal, false);
                let casted_add = format!("($unsigned($signed({base})) + {rhs})");
                let casted_rel = format!("($signed($unsigned({base})) < {rhs})");
                let parent = mk_lit(64, false, 1900 + ordinal, false);
                let expr1 = format!("(({casted_add}) ^ ({parent}))");
                let expr2 = format!("(({casted_rel}) ^ ({parent}))");
                let label1 = format!("cast-add bw={base_w} rw={rhs_w} s={signed}");
                let label2 = format!("cast-rel bw={base_w} rw={rhs_w} s={signed}");
                assert_eval_matches_oracle(&expr1, &label1);
                assert_eval_matches_oracle(&expr2, &label2);
            }
        }
    }
}

#[test]
fn slice_and_indexed_slice_boundary_sweep_matches_oracle() {
    let mut ordinal = 0_u32;
    let bases = [0_u32, 1, 7, 8, 15, 31, 32, 40, 63];

    for &w in &[8_u32, 32, 44, 64] {
        for &b in &bases {
            ordinal = ordinal.wrapping_add(1);
            if ordinal % 2 != 0 {
                continue;
            }
            let v_bits = bit_pattern(w, 2000 + ordinal, false);
            let msb = if w > 0 { w - 1 } else { 0 };
            let lsb = if w > 4 { w - 4 } else { 0 };
            let expr_slice = format!("(v[{msb}:{lsb}] ^ 16'b1010101010101010)");
            let expr_plus = format!("((v[{b} +: 4]) ^ 16'b1010101010101010)");
            let expr_minus = format!("((v[{b} -: 4]) ^ 16'b1010101010101010)");
            let mut env = Env::new();
            env.insert("v", mk_unsigned_value_from_msb(&v_bits));
            let label_slice = format!("slice w={w} msb={msb} lsb={lsb}");
            let label_plus = format!("idx-slice-plus w={w} b={b}");
            let label_minus = format!("idx-slice-minus w={w} b={b}");
            assert_eval_matches_oracle_with_env(&expr_slice, &env, &label_slice);
            assert_eval_matches_oracle_with_env(&expr_plus, &env, &label_plus);
            assert_eval_matches_oracle_with_env(&expr_minus, &env, &label_minus);
        }
    }
}

#[test]
fn logical_truthiness_sweep_matches_oracle() {
    let vals = [
        "8'b00000000",
        "8'b00000001",
        "8'b10000000",
        "8'b0000x000",
        "8'b0000z000",
        "8'bxxxx0000",
        "8'bzzzz0000",
        "8'bxzxz1010",
    ];
    let mut ordinal = 0_u32;

    for &lhs in &vals {
        for &rhs in &vals {
            ordinal = ordinal.wrapping_add(1);
            if ordinal % 2 != 0 {
                continue;
            }
            let parent = mk_lit(16, false, 2100 + ordinal, false);
            let expr_and = format!("((({lhs}) && ({rhs})) ^ ({parent}))");
            let expr_or = format!("((({lhs}) || ({rhs})) ^ ({parent}))");
            let expr_not = format!("((!({lhs})) ^ ({parent}))");
            let label_and = format!("logical-and lhs={lhs} rhs={rhs}");
            let label_or = format!("logical-or lhs={lhs} rhs={rhs}");
            let label_not = format!("logical-not lhs={lhs}");
            assert_eval_matches_oracle(&expr_and, &label_and);
            assert_eval_matches_oracle(&expr_or, &label_or);
            assert_eval_matches_oracle(&expr_not, &label_not);
        }
    }
}

#[test]
fn ternary_known_cond_still_uses_merged_branch_type_sweep_matches_oracle() {
    let mut ordinal = 0_u32;
    let conds = ["1'b0", "1'b1"];
    for &cond in &conds {
        for &tw in &[8_u32, 32, 44] {
            for &fw in &[33_u32, 64, 96] {
                for &ts in &[false, true] {
                    for &fs in &[false, true] {
                        ordinal = ordinal.wrapping_add(1);
                        if ordinal % 2 != 0 {
                            continue;
                        }
                        let t = mk_lit(tw, ts, 2200 + ordinal, ts);
                        let f_a = mk_lit(fw, fs, 2300 + ordinal, fs);
                        let f_b = mk_lit(fw, false, 2400 + ordinal, false);
                        let f = format!("(({f_a}) & ({f_b}))");
                        let expr = format!("(({cond}) ? ({t}) : ({f}))");
                        let label =
                            format!("ternary-known cond={cond} tw={tw} fw={fw} ts={ts} fs={fs}");
                        assert_eval_matches_oracle(&expr, &label);
                    }
                }
            }
        }
    }
}

#[test]
fn ternary_mixed_signedness_branch_context_sweep_matches_oracle() {
    let conds = ["1'b0", "1'b1", "1'bx", "1'bz"];
    let mut ordinal = 0_u32;
    for &signed_w in &[8_u32, 32, 44] {
        for &arith_unsigned_w in &[33_u32, 64, 80] {
            for &other_w in &[64_u32, 96, 128] {
                if other_w <= arith_unsigned_w {
                    continue;
                }
                ordinal = ordinal.wrapping_add(1);
                let signed_branch_lhs = mk_lit(signed_w, true, 2900 + ordinal, true);
                let arith_branch_rhs = mk_lit(arith_unsigned_w, false, 3000 + ordinal, false);
                let arith_branch = format!("(({signed_branch_lhs}) - ({arith_branch_rhs}))");
                let other_branch = mk_lit(other_w, false, 3100 + ordinal, false);
                for &cond in &conds {
                    let expr_arith_true =
                        format!("(({cond}) ? ({arith_branch}) : ({other_branch}))");
                    let label_arith_true = format!(
                        "ternary-branch-ctx cond={cond} arith_true sw={signed_w} aw={arith_unsigned_w} ow={other_w}"
                    );
                    assert_eval_matches_oracle(&expr_arith_true, &label_arith_true);

                    let expr_arith_false =
                        format!("(({cond}) ? ({other_branch}) : ({arith_branch}))");
                    let label_arith_false = format!(
                        "ternary-branch-ctx cond={cond} arith_false sw={signed_w} aw={arith_unsigned_w} ow={other_w}"
                    );
                    assert_eval_matches_oracle(&expr_arith_false, &label_arith_false);
                }
            }
        }
    }
}

#[test]
fn shift_rhs_context_isolation_sweep_matches_oracle() {
    let mut ordinal = 0_u32;
    for &op in &[">>", ">>>"] {
        for &lw in &[32_u32, 44, 85] {
            for &rw in &[32_u32, 44] {
                ordinal = ordinal.wrapping_add(1);
                if ordinal % 2 != 0 {
                    continue;
                }
                let lhs = mk_lit(lw, true, 2500 + ordinal, true);
                let div_num = mk_lit(rw, true, 2600 + ordinal, true);
                let div_den = mk_lit(rw, true, 2700 + ordinal, false);
                let rhs = format!("(({div_num}) / (~({div_den})))");
                let shifted = format!("(({lhs}) {op} ({rhs}))");
                let parent = mk_lit(96, false, 2800 + ordinal, false);
                let expr = format!("(({shifted}) ^ ({parent}))");
                let label = format!("shift-rhs-isolation op={op} lw={lw} rw={rw}");
                assert_eval_matches_oracle(&expr, &label);
            }
        }
    }
}

#[test]
fn relational_parent_signedness_isolation_sweep_matches_oracle() {
    let mut ordinal = 0_u32;
    let rel_ops = ["<", ">", "<=", ">="];
    for (op_i, rel_op) in rel_ops.iter().enumerate() {
        for &lw in &[8_u32, 32, 44] {
            for &rw in &[8_u32, 32, 44] {
                ordinal = ordinal.wrapping_add(1);
                if ordinal % 2 != 0 {
                    continue;
                }
                let lhs = mk_lit(lw, true, 2900 + ordinal + op_i as u32, true);
                let rhs = mk_lit(rw, true, 3000 + ordinal + op_i as u32, false);
                let rel = format!("((~({lhs})) {rel_op} ({rhs}))");
                let parent = mk_lit(64, false, 3100 + ordinal + op_i as u32, false);
                let expr = format!("(({parent}) | ({rel}))");
                let label = format!("rel-parent-isolation op={rel_op} lw={lw} rw={rw}");
                assert_eval_matches_oracle(&expr, &label);
            }
        }
    }
}

#[test]
fn shift_amount_unknown_rhs_expression_sweep_matches_oracle() {
    let mut ordinal = 0_u32;
    for &op in &[">>", ">>>"] {
        for &lhs_w in &[8_u32, 32, 44] {
            for rhs_pat in [
                "32'b00000000000000000000000000000001",
                "32'b00000000000000000000000000000100",
                "32'b0000000000000000000000000000x001",
                "32'b0000000000000000000000000000z001",
                "32'b000000000000000000000000000xzz1",
            ] {
                ordinal = ordinal.wrapping_add(1);
                if ordinal % 2 != 0 {
                    continue;
                }
                let lhs = mk_lit(lhs_w, true, 3200 + ordinal, true);
                let rhs_expr = format!("(({rhs_pat}) + 32'd1)");
                let expr = format!("(({lhs}) {op} ({rhs_expr}))");
                let label = format!("shift-rhs-unknown op={op} lhs_w={lhs_w} rhs_pat={rhs_pat}");
                assert_eval_matches_oracle(&expr, &label);
            }
        }
    }
}

#[test]
fn nested_ternary_parent_context_sweep_matches_oracle() {
    let mut ordinal = 0_u32;
    for cond1 in ["1'b0", "1'b1", "1'bx", "1'bz"] {
        for cond2 in ["1'b0", "1'b1", "1'bx", "1'bz"] {
            for &aw in &[8_u32, 32] {
                for &bw in &[33_u32, 44] {
                    ordinal = ordinal.wrapping_add(1);
                    if ordinal % 3 != 0 {
                        continue;
                    }
                    let a = mk_lit(aw, true, 3300 + ordinal, true);
                    let b = mk_lit(bw, false, 3400 + ordinal, false);
                    let c = mk_lit(aw, false, 3500 + ordinal, false);
                    let d = mk_lit(bw, true, 3600 + ordinal, true);
                    let nested = format!("(({cond1}) ? (({cond2}) ? ({a}) : ({b})) : ({c}))");
                    let expr_bitwise = format!("(({nested}) ^ ({d}))");
                    let expr_arith = format!("(({nested}) + ({d}))");
                    let label_bitwise =
                        format!("nested-ternary-bitwise c1={cond1} c2={cond2} aw={aw} bw={bw}");
                    let label_arith =
                        format!("nested-ternary-arith c1={cond1} c2={cond2} aw={aw} bw={bw}");
                    assert_eval_matches_oracle(&expr_bitwise, &label_bitwise);
                    assert_eval_matches_oracle(&expr_arith, &label_arith);
                }
            }
        }
    }
}

#[test]
fn relational_under_equality_and_logical_parents_sweep_matches_oracle() {
    let mut ordinal = 0_u32;
    for &rel_op in &["<", "<=", ">", ">="] {
        for &lw in &[8_u32, 32, 44] {
            for &rw in &[8_u32, 33, 44] {
                ordinal = ordinal.wrapping_add(1);
                if ordinal % 2 != 0 {
                    continue;
                }
                let lhs = mk_lit(lw, true, 3700 + ordinal, true);
                let rhs = mk_lit(rw, false, 3800 + ordinal, false);
                let rel = format!("(({lhs}) {rel_op} ({rhs}))");
                let eq_parent = mk_lit(1, false, 3900 + ordinal, false);
                let log_parent = mk_lit(8, false, 4000 + ordinal, false);
                let expr_eq = format!("(({rel}) == ({eq_parent}))");
                let expr_log_and = format!("(({rel}) && ({log_parent}))");
                let expr_log_or = format!("(({rel}) || ({log_parent}))");
                let label_eq = format!("rel-parent-eq op={rel_op} lw={lw} rw={rw}");
                let label_log_and = format!("rel-parent-logand op={rel_op} lw={lw} rw={rw}");
                let label_log_or = format!("rel-parent-logor op={rel_op} lw={lw} rw={rw}");
                assert_eval_matches_oracle(&expr_eq, &label_eq);
                assert_eval_matches_oracle(&expr_log_and, &label_log_and);
                assert_eval_matches_oracle(&expr_log_or, &label_log_or);
            }
        }
    }
}

#[test]
fn concat_cast_rel_shift_context_sweep_matches_oracle() {
    let mut ordinal = 0_u32;
    for &aw in &[8_u32, 16, 32] {
        for &bw in &[8_u32, 17, 33] {
            for &cw in &[8_u32, 32] {
                ordinal = ordinal.wrapping_add(1);
                if ordinal % 2 != 0 {
                    continue;
                }
                let a = mk_lit(aw, true, 4100 + ordinal, true);
                let b = mk_lit(bw, false, 4200 + ordinal, false);
                let c = mk_lit(cw, true, 4300 + ordinal, true);
                let cat = format!("{{({a}), ({b})}}");
                let casted = format!("$signed($unsigned({cat}))");
                let shifted = format!("(({casted}) >>> 8'd3)");
                let rel = format!("(({shifted}) >= ({c}))");
                let parent = mk_lit(64, false, 4400 + ordinal, false);
                let expr = format!("(({rel}) ^ ({parent}))");
                let label = format!("concat-cast-rel-shift aw={aw} bw={bw} cw={cw}");
                assert_eval_matches_oracle(&expr, &label);
            }
        }
    }
}

#[test]
fn dynamic_select_unknown_index_sweep_matches_oracle() {
    let mut ordinal = 0_u32;
    let idx_bits = ["00", "01", "10", "11", "x0", "0x", "xz", "zx", "zz", "xx"];
    for &v_w in &[4_u32, 8, 16] {
        for bits in idx_bits {
            ordinal = ordinal.wrapping_add(1);
            if ordinal % 2 != 0 {
                continue;
            }
            let v_bits = bit_pattern(v_w, 4500 + ordinal, false);
            let mut env = Env::new();
            env.insert("v", mk_unsigned_value_from_msb(&v_bits));
            env.insert("idx", mk_unsigned_value_from_msb(bits));
            env.insert("base", mk_unsigned_value_from_msb(bits));

            let expr_index = "(v[idx])";
            let expr_plus = "(v[base +: 3])";
            let expr_minus = "(v[base -: 3])";
            let label_index = format!("dyn-index vw={v_w} bits={bits}");
            let label_plus = format!("dyn-plus vw={v_w} bits={bits}");
            let label_minus = format!("dyn-minus vw={v_w} bits={bits}");
            assert_eval_matches_oracle_with_env(expr_index, &env, &label_index);
            assert_eval_matches_oracle_with_env(expr_plus, &env, &label_plus);
            assert_eval_matches_oracle_with_env(expr_minus, &env, &label_minus);
        }
    }
}

#[test]
fn sized_mixed_signedness_arithmetic_sweep_matches_oracle() {
    let mut ordinal = 0_u32;
    for &op in &["+", "-", "*", "/", "%"] {
        for &lhs_w in &[4_u32, 8] {
            for &rhs_w in &[8_u32, 16] {
                for &parent_w in &[8_u32, 16, 32] {
                    ordinal = ordinal.wrapping_add(1);
                    if ordinal % 2 != 0 {
                        continue;
                    }
                    let lhs = mk_lit(lhs_w, true, 4600 + ordinal, true);
                    let rhs = mk_lit(rhs_w, false, 4700 + ordinal, false);
                    let expr = format!("({parent_w}'b0 + (({lhs}) {op} ({rhs})))");
                    let label = format!(
                        "sized-mixed-sign op={op} lhs_w={lhs_w} rhs_w={rhs_w} parent_w={parent_w}"
                    );
                    assert_eval_matches_oracle(&expr, &label);
                }
            }
        }
    }
}
