// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::get_summary_stats;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::ir2gates;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExtNaryAddSubRow {
    bit_count: usize,
    live_nodes: usize,
    deepest_path: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExtNaryAddPow2Minus1Row {
    literal_value: u64,
    live_nodes: usize,
    deepest_path: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExtNaryAddSingleOnesRunRow {
    literal_value: u64,
    live_nodes: usize,
    deepest_path: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExtNaryAddCorrectionCountRow {
    correction_count: usize,
    live_nodes: usize,
    deepest_path: usize,
}

fn build_sub_ir_text(bit_count: usize) -> String {
    format!(
        r#"package test

top fn f(lhs: bits[{bit_count}] id=1, rhs: bits[{bit_count}] id=2) -> bits[{bit_count}] {{
  ret sub.3: bits[{bit_count}] = sub(lhs, rhs, id=3)
}}
"#
    )
}

fn build_nary_sub_ir_text(bit_count: usize) -> String {
    format!(
        r#"package test

top fn f(lhs: bits[{bit_count}] id=1, rhs: bits[{bit_count}] id=2) -> bits[{bit_count}] {{
  ret ext_nary_add.3: bits[{bit_count}] = ext_nary_add(lhs, rhs, signed=[false, false], negated=[false, true], arch=ripple_carry, id=3)
}}
"#
    )
}

fn build_add_const_ir_text(bit_count: usize, literal_value: u64) -> String {
    format!(
        r#"package test

top fn f(p0: bits[{bit_count}] id=1) -> bits[{bit_count}] {{
  lit: bits[{bit_count}] = literal(value={literal_value}, id=2)
  ret r: bits[{bit_count}] = add(p0, lit, id=3)
}}
"#
    )
}

fn build_nary_add_const_ir_text(bit_count: usize, literal_value: u64) -> String {
    format!(
        r#"package test

top fn f(p0: bits[{bit_count}] id=1) -> bits[{bit_count}] {{
  lit: bits[{bit_count}] = literal(value={literal_value}, id=2)
  ret r: bits[{bit_count}] = ext_nary_add(p0, lit, signed=[false, false], negated=[false, false], arch=ripple_carry, id=3)
}}
"#
    )
}

fn format_bool_attrs(values: &[bool]) -> String {
    values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn build_nary_add_unit_inc_ir_text(bit_count: usize, correction_count: usize) -> String {
    let mut params = vec![format!("x: bits[{bit_count}] id=1")];
    let mut operands = vec!["x".to_string()];
    let mut signed = vec![false];
    let mut negated = vec![false];
    for i in 0..correction_count {
        let id = 2 + i;
        let name = format!("b{i}");
        params.push(format!("{name}: bits[1] id={id}"));
        operands.push(name);
        signed.push(false);
        negated.push(false);
    }
    let ret_id = 2 + correction_count;
    format!(
        r#"package test

top fn f({params}) -> bits[{bit_count}] {{
  ret ext_nary_add.{ret_id}: bits[{bit_count}] = ext_nary_add({operands}, signed=[{signed}], negated=[{negated}], arch=ripple_carry, id={ret_id})
}}
"#,
        params = params.join(", "),
        operands = operands.join(", "),
        signed = format_bool_attrs(&signed),
        negated = format_bool_attrs(&negated),
    )
}

fn build_nary_add_unit_dec_ir_text(bit_count: usize, correction_count: usize) -> String {
    let mut params = vec![format!("x: bits[{bit_count}] id=1")];
    let mut operands = vec!["x".to_string()];
    let mut signed = vec![false];
    let mut negated = vec![false];
    for i in 0..correction_count {
        let id = 2 + i;
        let name = format!("b{i}");
        params.push(format!("{name}: bits[1] id={id}"));
        operands.push(name);
        signed.push(false);
        negated.push(true);
    }
    let ret_id = 2 + correction_count;
    format!(
        r#"package test

top fn f({params}) -> bits[{bit_count}] {{
  ret ext_nary_add.{ret_id}: bits[{bit_count}] = ext_nary_add({operands}, signed=[{signed}], negated=[{negated}], arch=ripple_carry, id={ret_id})
}}
"#,
        params = params.join(", "),
        operands = operands.join(", "),
        signed = format_bool_attrs(&signed),
        negated = format_bool_attrs(&negated),
    )
}

fn build_nary_sub_plus_signext_ir_text(bit_count: usize, correction_count: usize) -> String {
    let mut params = vec![
        format!("lhs: bits[{bit_count}] id=1"),
        format!("rhs: bits[{bit_count}] id=2"),
    ];
    let mut operands = vec!["lhs".to_string(), "rhs".to_string()];
    let mut signed = vec![false, false];
    let mut negated = vec![false, true];
    for i in 0..correction_count {
        let id = 3 + i;
        let name = format!("b{i}");
        params.push(format!("{name}: bits[1] id={id}"));
        operands.push(name);
        signed.push(true);
        negated.push(false);
    }
    let ret_id = 3 + correction_count;
    format!(
        r#"package test

top fn f({params}) -> bits[{bit_count}] {{
  ret ext_nary_add.{ret_id}: bits[{bit_count}] = ext_nary_add({operands}, signed=[{signed}], negated=[{negated}], arch=ripple_carry, id={ret_id})
}}
"#,
        params = params.join(", "),
        operands = operands.join(", "),
        signed = format_bool_attrs(&signed),
        negated = format_bool_attrs(&negated),
    )
}

fn build_nary_counter_inc_dec_ir_text(bit_count: usize, correction_count: usize) -> String {
    let mut params = vec![format!("counter: bits[{bit_count}] id=1")];
    let mut operands = vec!["counter".to_string()];
    let mut signed = vec![false];
    let mut negated = vec![false];
    for i in 0..correction_count {
        let id = 2 + i;
        let name = format!("inc{i}");
        params.push(format!("{name}: bits[1] id={id}"));
        operands.push(name);
        signed.push(false);
        negated.push(false);
    }
    for i in 0..correction_count {
        let id = 2 + correction_count + i;
        let name = format!("dec{i}");
        params.push(format!("{name}: bits[1] id={id}"));
        operands.push(name);
        signed.push(false);
        negated.push(true);
    }
    let ret_id = 2 + 2 * correction_count;
    format!(
        r#"package test

top fn f({params}) -> bits[{bit_count}] {{
  ret ext_nary_add.{ret_id}: bits[{bit_count}] = ext_nary_add({operands}, signed=[{signed}], negated=[{negated}], arch=ripple_carry, id={ret_id})
}}
"#,
        params = params.join(", "),
        operands = operands.join(", "),
        signed = format_bool_attrs(&signed),
        negated = format_bool_attrs(&negated),
    )
}

fn get_ir_gate_stats(ir_text: &str) -> (usize, usize) {
    let out = ir2gates::ir2gates_from_ir_text(
        ir_text,
        None,
        ir2gates::Ir2GatesOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            enable_rewrite_nary_add: false,
            adder_mapping: AdderMapping::RippleCarry,
            mul_adder_mapping: None,
            aug_opt: Default::default(),
        },
    )
    .expect("ir2gates");
    let stats = get_summary_stats(&out.gatify_output.gate_fn);
    (stats.live_nodes, stats.deepest_path)
}

fn gather_ext_nary_add_sub_rows() -> Vec<ExtNaryAddSubRow> {
    let mut got = Vec::new();
    for bit_count in 1..=16 {
        let sub_stats = get_ir_gate_stats(&build_sub_ir_text(bit_count));
        let nary_sub_stats = get_ir_gate_stats(&build_nary_sub_ir_text(bit_count));
        assert_eq!(
            sub_stats, nary_sub_stats,
            "expected explicit ext_nary_add(lhs, -rhs) to match dedicated sub lowering for {bit_count}-bit operands"
        );
        got.push(ExtNaryAddSubRow {
            bit_count,
            live_nodes: sub_stats.0,
            deepest_path: sub_stats.1,
        });
    }
    got
}

fn gather_ext_nary_add_pow2_minus1_rows() -> Vec<ExtNaryAddPow2Minus1Row> {
    let mut got = Vec::new();
    for k in 1..=8 {
        let literal_value = (1u64 << k) - 1;
        let add_stats = get_ir_gate_stats(&build_add_const_ir_text(8, literal_value));
        let nary_add_stats = get_ir_gate_stats(&build_nary_add_const_ir_text(8, literal_value));
        assert_eq!(
            add_stats, nary_add_stats,
            "expected explicit ext_nary_add(p0, {literal_value}) to match dedicated add lowering"
        );
        got.push(ExtNaryAddPow2Minus1Row {
            literal_value,
            live_nodes: add_stats.0,
            deepest_path: add_stats.1,
        });
    }
    got
}

fn gather_ext_nary_add_single_ones_run_rows() -> Vec<ExtNaryAddSingleOnesRunRow> {
    let mut got = Vec::new();
    for literal_value in [0u64, 0b0001_0000, 0b0001_1100, 0b0011_1100, 0b1111_0000] {
        let nary_add_stats = get_ir_gate_stats(&build_nary_add_const_ir_text(8, literal_value));
        got.push(ExtNaryAddSingleOnesRunRow {
            literal_value,
            live_nodes: nary_add_stats.0,
            deepest_path: nary_add_stats.1,
        });
    }
    got
}

fn gather_ext_nary_add_correction_count_rows(
    build_ir_text: impl Fn(usize) -> String,
) -> Vec<ExtNaryAddCorrectionCountRow> {
    let mut got = Vec::new();
    for correction_count in 0..=4 {
        let stats = get_ir_gate_stats(&build_ir_text(correction_count));
        got.push(ExtNaryAddCorrectionCountRow {
            correction_count,
            live_nodes: stats.0,
            deepest_path: stats.1,
        });
    }
    got
}

#[test]
fn test_ext_nary_add_sub_gate_stats_sweep_1_to_16() {
    let got = gather_ext_nary_add_sub_rows();

    // This is a "microbenchmark sweep" style test: lock in the expected gate
    // count + depth for `sub` / `ext_nary_add(lhs, -rhs)`, so we can notice
    // regressions and improvements.
    // eprintln!("{got:#?}");
    #[rustfmt::skip]
    let want: &[ExtNaryAddSubRow] = &[
        ExtNaryAddSubRow { bit_count: 1, live_nodes: 5, deepest_path: 3 },
        ExtNaryAddSubRow { bit_count: 2, live_nodes: 14, deepest_path: 5 },
        ExtNaryAddSubRow { bit_count: 3, live_nodes: 27, deepest_path: 7 },
        ExtNaryAddSubRow { bit_count: 4, live_nodes: 40, deepest_path: 10 },
        ExtNaryAddSubRow { bit_count: 5, live_nodes: 53, deepest_path: 13 },
        ExtNaryAddSubRow { bit_count: 6, live_nodes: 66, deepest_path: 16 },
        ExtNaryAddSubRow { bit_count: 7, live_nodes: 79, deepest_path: 19 },
        ExtNaryAddSubRow { bit_count: 8, live_nodes: 92, deepest_path: 22 },
        ExtNaryAddSubRow { bit_count: 9, live_nodes: 105, deepest_path: 25 },
        ExtNaryAddSubRow { bit_count: 10, live_nodes: 118, deepest_path: 28 },
        ExtNaryAddSubRow { bit_count: 11, live_nodes: 131, deepest_path: 31 },
        ExtNaryAddSubRow { bit_count: 12, live_nodes: 144, deepest_path: 34 },
        ExtNaryAddSubRow { bit_count: 13, live_nodes: 157, deepest_path: 37 },
        ExtNaryAddSubRow { bit_count: 14, live_nodes: 170, deepest_path: 40 },
        ExtNaryAddSubRow { bit_count: 15, live_nodes: 183, deepest_path: 43 },
        ExtNaryAddSubRow { bit_count: 16, live_nodes: 196, deepest_path: 46 },
    ];

    assert_eq!(got.as_slice(), want);
}

#[test]
fn test_ext_nary_add_pow2_minus1_gate_stats_sweep_w8() {
    let got = gather_ext_nary_add_pow2_minus1_rows();

    // This locks in the shared `add(p0, 2^k-1)` / `ext_nary_add(p0, 2^k-1)`
    // gate stats for width 8.
    // eprintln!("{got:#?}");
    #[rustfmt::skip]
    let want: &[ExtNaryAddPow2Minus1Row] = &[
        ExtNaryAddPow2Minus1Row { literal_value:   1, live_nodes: 35, deepest_path:  9 },
        ExtNaryAddPow2Minus1Row { literal_value:   3, live_nodes: 35, deepest_path:  9 },
        ExtNaryAddPow2Minus1Row { literal_value:   7, live_nodes: 35, deepest_path:  9 },
        ExtNaryAddPow2Minus1Row { literal_value:  15, live_nodes: 35, deepest_path:  9 },
        ExtNaryAddPow2Minus1Row { literal_value:  31, live_nodes: 35, deepest_path:  9 },
        ExtNaryAddPow2Minus1Row { literal_value:  63, live_nodes: 35, deepest_path:  9 },
        ExtNaryAddPow2Minus1Row { literal_value: 127, live_nodes: 35, deepest_path:  9 },
        ExtNaryAddPow2Minus1Row { literal_value: 255, live_nodes: 35, deepest_path:  9 },
    ];

    assert_eq!(got.as_slice(), want);
}

#[test]
fn test_ext_nary_add_single_ones_run_gate_stats_w8() {
    let got = gather_ext_nary_add_single_ones_run_rows();

    // This locks in the one-dense-term shifted-ones-run finalizer for explicit
    // `ext_nary_add(p0, literal)` with no dynamic carry-in.
    // eprintln!("{got:#?}");
    #[rustfmt::skip]
    let want: &[ExtNaryAddSingleOnesRunRow] = &[
        ExtNaryAddSingleOnesRunRow { literal_value:   0, live_nodes:  8, deepest_path:  1 },
        ExtNaryAddSingleOnesRunRow { literal_value:  16, live_nodes: 19, deepest_path:  5 },
        ExtNaryAddSingleOnesRunRow { literal_value:  28, live_nodes: 27, deepest_path:  7 },
        ExtNaryAddSingleOnesRunRow { literal_value:  60, live_nodes: 27, deepest_path:  7 },
        ExtNaryAddSingleOnesRunRow { literal_value: 240, live_nodes: 19, deepest_path:  5 },
    ];

    assert_eq!(got.as_slice(), want);
}

#[test]
fn test_ext_nary_add_unit_inc_gate_stats_sweep_w16_corrections_0_to_4() {
    let got = gather_ext_nary_add_correction_count_rows(|correction_count| {
        build_nary_add_unit_inc_ir_text(16, correction_count)
    });

    // This locks in the non-stacked `+bits[1]` lowering shape, where at most
    // one unit increment becomes a carry-in and the rest join the dense add.
    // eprintln!("{got:#?}");
    #[rustfmt::skip]
    let want: &[ExtNaryAddCorrectionCountRow] = &[
        ExtNaryAddCorrectionCountRow { correction_count: 0, live_nodes:  16, deepest_path:  1 },
        ExtNaryAddCorrectionCountRow { correction_count: 1, live_nodes:  80, deepest_path: 18 },
        ExtNaryAddCorrectionCountRow { correction_count: 2, live_nodes:  88, deepest_path: 20 },
        ExtNaryAddCorrectionCountRow { correction_count: 3, live_nodes: 100, deepest_path: 24 },
        ExtNaryAddCorrectionCountRow { correction_count: 4, live_nodes: 116, deepest_path: 28 },
    ];

    assert_eq!(got.as_slice(), want);
}

#[test]
fn test_ext_nary_add_unit_dec_gate_stats_sweep_w16_corrections_0_to_4() {
    let got = gather_ext_nary_add_correction_count_rows(|correction_count| {
        build_nary_add_unit_dec_ir_text(16, correction_count)
    });

    // This locks in the non-stacked `-bits[1]` lowering shape, where only one
    // decrement can use carry-in fusion and the rest join the dense add.
    // eprintln!("{got:#?}");
    #[rustfmt::skip]
    let want: &[ExtNaryAddCorrectionCountRow] = &[
        ExtNaryAddCorrectionCountRow { correction_count: 0, live_nodes:  16, deepest_path:  1 },
        ExtNaryAddCorrectionCountRow { correction_count: 1, live_nodes:  80, deepest_path: 18 },
        ExtNaryAddCorrectionCountRow { correction_count: 2, live_nodes:  88, deepest_path: 34 },
        ExtNaryAddCorrectionCountRow { correction_count: 3, live_nodes: 198, deepest_path: 50 },
        ExtNaryAddCorrectionCountRow { correction_count: 4, live_nodes: 204, deepest_path: 50 },
    ];

    assert_eq!(got.as_slice(), want);
}

#[test]
fn test_ext_nary_add_sub_plus_signext_gate_stats_sweep_w16_corrections_0_to_4() {
    let got = gather_ext_nary_add_correction_count_rows(|correction_count| {
        build_nary_sub_plus_signext_ir_text(16, correction_count)
    });

    // This captures `lhs - rhs + sign_ext(b*)`, where one sign-extended
    // correction can fuse into the subtract carry-in and the rest fall back to
    // dense terms.
    // eprintln!("{got:#?}");
    #[rustfmt::skip]
    let want: &[ExtNaryAddCorrectionCountRow] = &[
        ExtNaryAddCorrectionCountRow { correction_count: 0, live_nodes: 196, deepest_path: 46 },
        ExtNaryAddCorrectionCountRow { correction_count: 1, live_nodes: 204, deepest_path: 48 },
        ExtNaryAddCorrectionCountRow { correction_count: 2, live_nodes: 327, deepest_path: 50 },
        ExtNaryAddCorrectionCountRow { correction_count: 3, live_nodes: 384, deepest_path: 50 },
        ExtNaryAddCorrectionCountRow { correction_count: 4, live_nodes: 396, deepest_path: 52 },
    ];

    assert_eq!(got.as_slice(), want);
}

#[test]
fn test_ext_nary_add_counter_inc_dec_gate_stats_sweep_w16_corrections_0_to_4() {
    let got = gather_ext_nary_add_correction_count_rows(|correction_count| {
        build_nary_counter_inc_dec_ir_text(16, correction_count)
    });

    // This captures `counter + inc* - dec*`, where one bit-0 correction may be
    // carried in directly and the rest fall back to dense terms.
    // eprintln!("{got:#?}");
    #[rustfmt::skip]
    let want: &[ExtNaryAddCorrectionCountRow] = &[
        ExtNaryAddCorrectionCountRow { correction_count: 0, live_nodes:  16, deepest_path:  1 },
        ExtNaryAddCorrectionCountRow { correction_count: 1, live_nodes: 186, deepest_path: 48 },
        ExtNaryAddCorrectionCountRow { correction_count: 2, live_nodes: 310, deepest_path: 50 },
        ExtNaryAddCorrectionCountRow { correction_count: 3, live_nodes: 337, deepest_path: 54 },
        ExtNaryAddCorrectionCountRow { correction_count: 4, live_nodes: 257, deepest_path: 54 },
    ];

    assert_eq!(got.as_slice(), want);
}
