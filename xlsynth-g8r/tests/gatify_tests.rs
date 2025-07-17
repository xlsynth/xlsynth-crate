// SPDX-License-Identifier: Apache-2.0

#[macro_use]
extern crate xlsynth_g8r;

use std::collections::HashMap;
use std::path::Path;
use test_case::{test_case, test_matrix};
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::get_summary_stats::{SummaryStats, get_summary_stats};
use xlsynth_g8r::ir2gate::{GatifyOptions, gatify, gatify_ule_via_bit_tests};
use xlsynth_g8r::ir2gate_utils::gatify_one_hot;
use xlsynth_g8r::test_utils::Opt;
use xlsynth_g8r::xls_ir::ir_parser;

use maplit::btreemap;

/// Proves that the gate-mapped version fo the top function in `ir_package_text`
/// is equivalent to the IR version.
///
/// Returns the summary stats for the gate function that is created in case the
/// user is also interested in the statistics for it.
fn do_test_ir_conversion_with_top(
    ir_package_text: &str,
    opt: Opt,
    top_name: Option<&str>,
) -> SummaryStats {
    // Now we'll parse the IR and turn it into a gate function.
    let mut parser = ir_parser::Parser::new(&ir_package_text);
    let ir_package = parser.parse_package().unwrap();
    let ir_fn = match top_name {
        Some(name) => ir_package.get_fn(name).unwrap(),
        None => ir_package.get_top().unwrap(),
    };
    let gatify_output = gatify(
        &ir_fn,
        GatifyOptions {
            fold: opt == Opt::Yes,
            check_equivalence: false,
            hash: opt == Opt::Yes,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::RippleCarry,
        },
    )
    .unwrap();

    check_equivalence::validate_same_fn(&ir_fn, &gatify_output.gate_fn)
        .expect("should validate IR to gate function equivalence");

    get_summary_stats(&gatify_output.gate_fn)
}

/// Wrapper around `do_test_ir_conversion_with_top` that assumes the top
/// function should be used.
///
/// `opt` specifies whether to use the optimizing `GateBuilderOptions` or not.
fn do_test_ir_conversion(ir_package_text: &str, opt: Opt) -> SummaryStats {
    do_test_ir_conversion_with_top(ir_package_text, opt, None)
}

/// Similar to `do_test_ir_conversion` but does not attempt to convert the
/// resulting gate function back to IR for equivalence checking. This is useful
/// for testing constructs (like `fail!` or `assert!`) that currently lack a
/// reverse translation path.
fn do_test_ir_conversion_no_equiv(ir_package_text: &str, opt: Opt) {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut parser = ir_parser::Parser::new(&ir_package_text);
    let ir_package = parser.parse_package().unwrap();
    let ir_fn = ir_package.get_top().unwrap();
    let _ = gatify(
        &ir_fn,
        GatifyOptions {
            fold: opt == Opt::Yes,
            check_equivalence: false,
            hash: opt == Opt::Yes,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::RippleCarry,
        },
    )
    .unwrap();
}

#[test_case(1, Opt::No; "bit_count=1, fold=false")]
#[test_case(1, Opt::Yes; "bit_count=1, fold=true")]
#[test_case(2, Opt::No; "bit_count=2, fold=false")]
#[test_case(2, Opt::Yes; "bit_count=2, fold=true")]
fn test_prio_sel_ir_binary(bit_count: u32, opt: Opt) {
    let ir_text_tmpl = "package sample
fn do_prio_sel(sel: bits[2], a: bits[$BIT_COUNT], b: bits[$BIT_COUNT], default: bits[$BIT_COUNT]) -> bits[$BIT_COUNT] {
    ret result: bits[$BIT_COUNT] = priority_sel(sel, cases=[a, b], default=default, id=5)
}
";
    let ir_text = ir_text_tmpl.replace("$BIT_COUNT", &bit_count.to_string());

    do_test_ir_conversion(&ir_text, opt);
}

#[test]
fn test_tuple_index_ir_to_gates() {
    let _ = env_logger::builder().is_test(true).try_init();
    let ir_text = "package sample
fn do_tuple_index(t: (bits[1], bits[1])) -> bits[1] {
    ret result: bits[1] = tuple_index(t, index=0, id=2)
}
";
    do_test_ir_conversion(&ir_text, Opt::No);
}

fn do_test_dslx_conversion(input_bits: u32, opt: Opt, dslx_text: &str) {
    let _ = env_logger::builder().is_test(true).try_init();

    let module = format!("const N: u32 = u32:{};\n{}", input_bits, dslx_text);
    let path = Path::new("sample.x");
    let ir = xlsynth::convert_dslx_to_ir(&module, path, &xlsynth::DslxConvertOptions::default())
        .unwrap();
    let ir_package = &ir.ir;
    let ir_text = ir_package.to_string();
    log::info!("IR: {}", ir_text);

    do_test_ir_conversion(&ir_text, opt);
}

macro_rules! bit_count_test_cases {
    ($test_name:ident, $lambda:expr) => {
        #[test_case(1, Opt::No; "bit_count=1, fold=false")]
        #[test_case(2, Opt::No; "bit_count=2, fold=false")]
        #[test_case(3, Opt::No; "bit_count=3, fold=false")]
        #[test_case(4, Opt::No; "bit_count=4, fold=false")]
        #[test_case(1, Opt::Yes; "bit_count=1, fold=true")]
        #[test_case(2, Opt::Yes; "bit_count=2, fold=true")]
        #[test_case(3, Opt::Yes; "bit_count=3, fold=true")]
        #[test_case(4, Opt::Yes; "bit_count=4, fold=true")]
        fn $test_name(input_bits: u32, opt: Opt) {
            let _ = env_logger::builder().is_test(true).try_init();
            $lambda(input_bits, opt)
        }
    };
}

bit_count_test_cases!(test_or_zero_ir_to_gates, |input_bits: u32,
                                                 opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_or_zero(x: uN[N]) -> uN[N] { x | zero!<uN[N]>() }",
    );
});

bit_count_test_cases!(test_and_ones_ir_to_gates, |input_bits: u32,
                                                  opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_and_ones(x: uN[N]) -> uN[N] { x & all_ones!<uN[N]>() }",
    );
});

// This test has a problem with returning a parameter in the IR formatting.
/*
bit_count_test_cases!(test_identity_ir_to_gates, |input_bits: u32| -> () {
    do_test_dslx_conversion(input_bits, "fn do_identity(x: uN[N]) -> uN[N] { x }");
});
*/

bit_count_test_cases!(test_and_ir_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_and(x: uN[N], y: uN[N]) -> uN[N] { x & y }",
    );
});

bit_count_test_cases!(test_or_ir_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_or(x: uN[N], y: uN[N]) -> uN[N] { x | y }",
    );
});

bit_count_test_cases!(test_xor_ir_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_xor(x: uN[N], y: uN[N]) -> uN[N] { x ^ y }",
    );
});

bit_count_test_cases!(test_ne_all_zeros_all_ones_to_gates, |input_bits: u32,
                                                            opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_ne_all_zeros_all_ones() -> bool { zero!<uN[N]>() != all_ones!<uN[N]>() }",
    );
});

bit_count_test_cases!(test_add_ir_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_add(x: uN[N], y: uN[N]) -> uN[N] { x + y }",
    );
});

bit_count_test_cases!(test_sign_ext_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "const NP1: u32 = N + u32:1;
        fn do_sign_ext(x: uN[N]) -> uN[NP1] { x as sN[NP1] as uN[NP1] }",
    );
});

bit_count_test_cases!(test_or_reduce_ir_to_gates, |input_bits: u32,
                                                   opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_or_reduce(x: uN[N]) -> bool { or_reduce(x) }",
    );
});

bit_count_test_cases!(test_decode_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "const TPN: u32 = u32:1 << N;
        fn do_decode(x: uN[N]) -> uN[TPN] { decode<uN[TPN]>(x) }",
    );
});

bit_count_test_cases!(test_priority_sel_match_ir_to_gates, |input_bits: u32,
                                                            opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        r#"fn do_priority_sel(s: bool, x: uN[N], y: uN[N]) -> uN[N] {
    match s {
        false => x,
        true => y,
    }
}"#,
    );
});

bit_count_test_cases!(test_neg_ir_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(input_bits, opt, "fn do_neg(x: uN[N]) -> uN[N] { -x }");
});

bit_count_test_cases!(test_not_ir_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(input_bits, opt, "fn do_not(x: uN[N]) -> uN[N] { !x }");
});

bit_count_test_cases!(test_concat_ir_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "const NT2: u32 = N * u32:2;
        fn do_concat(x: uN[N], y: uN[N]) -> uN[NT2] { x ++ y }",
    );
});

bit_count_test_cases!(test_sub_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_sub(x: uN[N], y: uN[N]) -> uN[N] { x - y }",
    );
});

bit_count_test_cases!(test_nand_ir_to_gates, |input_bits: u32, opt: Opt| -> () {
    let ir_tmpl = "package sample
fn do_nand(x: bits[$BIT_COUNT], y: bits[$BIT_COUNT]) -> bits[$BIT_COUNT] {
  ret nand.3: bits[$BIT_COUNT] = nand(x, y, id=3)
}";
    let ir_text = ir_tmpl.replace("$BIT_COUNT", &input_bits.to_string());
    do_test_ir_conversion(&ir_text, opt);
});

bit_count_test_cases!(test_encode_dslx_to_gates, |input_bits: u32,
                                                  opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_encode(x: uN[N]) -> u32 { encode(x) as u32 }",
    );
});

bit_count_test_cases!(test_tuple_index_to_gates, |input_bits: u32,
                                                  opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_tuple_index(t: (u2, uN[N], u2)) -> u2 { t.0 + t.2 }",
    );
});

bit_count_test_cases!(test_array_index_to_gates, |input_bits: u32,
                                                  opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_array_index(a: uN[N][N], index: u32) -> uN[N] { a[index] }",
    );
});

bit_count_test_cases!(test_array_index_multidim_to_gates, |input_bits: u32,
                                                           opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_array_index_multidim(a: uN[N][N][N], i: u32, j: u32) -> uN[N] { a[i][j] }",
    );
});

bit_count_test_cases!(test_dslx_array_to_gates, |input_bits: u32,
                                                 opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_array(x: uN[N], y: uN[N], z: uN[N]) -> uN[N][3] { [x, y, z] }",
    );
});

bit_count_test_cases!(test_dslx_array_literal_to_gates, |input_bits: u32,
                                                         opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_array_literal() -> uN[N][2] { uN[N][2]:[0, 1] }",
    );
});

// Emits a priority select via the DSLX builtin.
bit_count_test_cases!(test_priority_sel_builtin_to_gates, |input_bits: u32,
                                                           opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_priority_sel_builtin(s: uN[N], cases: uN[N][N], default_value: uN[N]) -> uN[N] {
            priority_sel(s, cases, default_value)
        }",
    );
});

bit_count_test_cases!(test_one_hot_lsb_prio_dslx_to_gates, |input_bits: u32,
                                                            opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "const NP1: u32 = N + u32:1;
        fn do_one_hot(x: uN[N]) -> uN[NP1] { one_hot(x, true) }",
    );
});

bit_count_test_cases!(test_one_hot_msb_prio_dslx_to_gates, |input_bits: u32,
                                                            opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "const NP1: u32 = N + u32:1;
        fn do_one_hot(x: uN[N]) -> uN[NP1] { one_hot(x, false) }",
    );
});

// Emits a one-hot-select via the DSLX builtin.
bit_count_test_cases!(test_one_hot_select_builtin_to_gates, |input_bits: u32,
                                                             opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_one_hot_select_builtin(s: uN[N], cases: uN[N][N]) -> uN[N] { one_hot_sel(s, cases) }",
    );
});

bit_count_test_cases!(test_shrl_by_u32_to_gates, |input_bits: u32,
                                                  opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_shrl_by_u32(x: uN[N], amount: u32) -> uN[N] { x >> amount }",
    );
});

bit_count_test_cases!(test_dynamic_bit_slice, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_dynamic_bit_slice(x: uN[N], i: uN[N]) -> uN[N] { x[i +: uN[N]] }",
    );
});

bit_count_test_cases!(test_bit_slice_update, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_bit_slice_update(x: uN[N], i: uN[N], u: uN[N]) -> uN[N] { bit_slice_update(x, i, u) }",
    );
});

#[test]
fn test_array_update_ir_to_gates() {
    let ir_text = "package sample
fn f(arr: bits[8][4], value: bits[8], index: bits[1]) -> bits[8][4] {
  ret result: bits[8][4] = array_update(arr, value, indices=[index], id=3)
}";
    do_test_ir_conversion_no_equiv(ir_text, Opt::No);
}

#[test]
fn test_array_update_multidim_ir_to_gates() {
    let ir_text = "package sample
fn f(arr: bits[8][2][2], value: bits[8], i0: bits[1], i1: bits[1]) -> bits[8][2][2] {
  ret result: bits[8][2][2] = array_update(arr, value, indices=[i0, i1], id=3)
}";
    do_test_ir_conversion_no_equiv(ir_text, Opt::No);
}

bit_count_test_cases!(test_shra_dslx_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_shra_by_u32(x: sN[N], amount: uN[N]) -> sN[N] { x >> amount }",
    );
});

bit_count_test_cases!(test_shra_by_u32_dslx_to_gates, |input_bits: u32,
                                                       opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_shra_by_u32(x: sN[N], amount: u32) -> sN[N] { x >> amount }",
    );
});

bit_count_test_cases!(test_shrl_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_shrl(x: uN[N], amount: uN[N]) -> uN[N] { x >> amount }",
    );
});

bit_count_test_cases!(test_shll_dslx_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_shll(x: uN[N], amount: u32) -> uN[N] { x << amount }",
    );
});

bit_count_test_cases!(test_sel_cond_dslx_to_gates, |input_bits: u32,
                                                    opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_sel_cond(s: bool, on_true: uN[N], on_false: uN[N]) -> uN[N] { if s { on_true } else { on_false } }",
    );
});

bit_count_test_cases!(test_width_slice_static_start_to_gates, |input_bits: u32,
                                                               opt: Opt|
 -> () {
    let ir_tmpl = "package sample
fn f(x: bits[32]) -> bits[$BIT_COUNT] {
    ret result.2: bits[$BIT_COUNT] = bit_slice(x, start=1, width=$BIT_COUNT, id=2)
}";
    let ir_text = ir_tmpl.replace("$BIT_COUNT", &input_bits.to_string());
    do_test_ir_conversion(&ir_text, opt);
});

// -- comparisons

bit_count_test_cases!(test_eq_ir_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_eq(x: uN[N], y: uN[N]) -> bool { x == y }",
    );
});

bit_count_test_cases!(test_ne_ir_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_ne_all_ones(x: uN[N]) -> bool { x != all_ones!<uN[N]>() }",
    );
});

bit_count_test_cases!(test_eqz_ir_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_eqz(x: uN[N]) -> bool { x == zero!<uN[N]>() }",
    );
});

bit_count_test_cases!(test_eq_all_zeros_all_ones_to_gates, |input_bits: u32,
                                                            opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_eq_all_zeros_all_ones() -> bool { zero!<uN[N]>() == all_ones!<uN[N]>() }",
    );
});

bit_count_test_cases!(test_ugt_ir_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_ugt(x: uN[N], y: uN[N]) -> bool { x > y }",
    );
});

bit_count_test_cases!(test_ult_ir_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_ult(x: uN[N], y: uN[N]) -> bool { x < y }",
    );
});

bit_count_test_cases!(test_ule_ir_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_ule(x: uN[N], y: uN[N]) -> bool { x <= y }",
    );
});

bit_count_test_cases!(test_uge_ir_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_uge(x: uN[N], y: uN[N]) -> bool { x >= y }",
    );
});

bit_count_test_cases!(test_sgt_dslx_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_sgt(x: sN[N], y: sN[N]) -> bool { x > y }",
    );
});

bit_count_test_cases!(test_slt_dslx_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_slt(x: sN[N], y: sN[N]) -> bool { x < y }",
    );
});

bit_count_test_cases!(test_sge_dslx_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_sge(x: sN[N], y: sN[N]) -> bool { x >= y }",
    );
});

bit_count_test_cases!(test_sle_dslx_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_sle(x: sN[N], y: sN[N]) -> bool { x <= y }",
    );
});

bit_count_test_cases!(test_encode_ir_to_gates, |input_bits: u32, opt: Opt| -> () {
    let output_bits = (input_bits as f32).log2().ceil() as u32;
    if output_bits == 0 {
        return;
    }
    log::info!("input_bits: {}, output_bits: {}", input_bits, output_bits);
    do_test_ir_conversion(
        &format!(
            "package sample
fn do_encode(x: bits[{input_bits}]) -> bits[{output_bits}] {{
    ret result: bits[{output_bits}] = encode(x, id=2)
}}"
        ),
        opt,
    );
});

#[test_matrix(
    4..8,
    1..=4,
    [Opt::Yes, Opt::No]
)]
fn test_decode_ir_to_gates(input_bits: u32, output_bits: u32, opt: Opt) {
    do_test_ir_conversion(
        &format!(
            "package sample
fn do_decode(x: bits[{input_bits}]) -> bits[{output_bits}] {{
    ret result: bits[{output_bits}] = decode(x, width={output_bits}, id=2)
}}"
        ),
        opt,
    );
}

#[test_matrix(
    1..3,
    1..3,
    1..7,
    [Opt::Yes, Opt::No]
)]
fn test_umul_ir_to_gates(lhs_bits: u32, rhs_bits: u32, output_bits: u32, opt: Opt) {
    do_test_ir_conversion(
        &format!(
            "package sample
fn do_umul(lhs: bits[{lhs_bits}], rhs: bits[{rhs_bits}]) -> bits[{output_bits}] {{
    ret result: bits[{output_bits}] = umul(lhs, rhs, id=3)
}}"
        ),
        opt,
    );
}

#[test_matrix(
    1..3,
    1..3,
    1..7,
    [Opt::Yes, Opt::No]
)]
fn test_smul_ir_to_gates(lhs_bits: u32, rhs_bits: u32, output_bits: u32, opt: Opt) {
    do_test_ir_conversion(
        &format!(
            "package sample
fn do_smul(lhs: bits[{lhs_bits}], rhs: bits[{rhs_bits}]) -> bits[{output_bits}] {{
    ret result: bits[{output_bits}] = smul(lhs, rhs, id=3)
}}",
        ),
        opt,
    );
}

#[test_matrix(
    1..3,
    [Opt::Yes, Opt::No]
)]
fn test_udiv_ir_to_gates(bit_count: u32, opt: Opt) {
    do_test_ir_conversion(
        &format!(
            "package sample
fn do_udiv(x: bits[{bit_count}], y: bits[{bit_count}]) -> bits[{bit_count}] {{
    ret result: bits[{bit_count}] = udiv(x, y, id=3)
}}",
        ),
        opt,
    );
}

#[test_matrix(
    1..3,
    [Opt::Yes, Opt::No]
)]
fn test_umod_ir_to_gates(bit_count: u32, opt: Opt) {
    do_test_ir_conversion(
        &format!(
            "package sample
fn do_umod(x: bits[{bit_count}], y: bits[{bit_count}]) -> bits[{bit_count}] {{
    ret result: bits[{bit_count}] = umod(x, y, id=3)
}}",
        ),
        opt,
    );
}

bit_count_test_cases!(test_sdiv_dslx_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_sdiv(x: sN[N], y: sN[N]) -> sN[N] { if y == sN[N]:0 { x } else { x / y } }",
    );
});

bit_count_test_cases!(test_smod_dslx_to_gates, |input_bits: u32, opt: Opt| -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_smod(x: sN[N], y: sN[N]) -> sN[N] { if y == sN[N]:0 { x } else { x % y } }",
    );
});

#[test_case(1, 2, Opt::Yes)]
#[test_case(1, 2, Opt::No)]
fn test_array_index_in_bounds_ir_to_gates(element_bits: u32, input_elements: u32, opt: Opt) {
    do_test_ir_conversion(
        &format!(
            "package sample
fn do_array_index_in_bounds(arr: bits[{element_bits}][{input_elements}], index: bits[1]) -> bits[{element_bits}] {{
    ret result: bits[{element_bits}] = array_index(arr, indices=[index], assumed_in_bounds=true, id=3)
}}",
        ),
        opt,
    );
}

fn gather_stats_for_widths(
    widths: &[usize],
    builder_fn: impl Fn(&mut GateBuilder, usize) -> (),
) -> HashMap<usize, SummaryStats> {
    let mut stats = HashMap::new();
    for width in widths {
        let mut builder = GateBuilder::new(format!("op_{}_bits", width), GateBuilderOptions::opt());
        builder_fn(&mut builder, *width);
        let gate_fn = builder.build();
        log::info!("gate_fn for width {}", width);
        log::info!("{}", gate_fn.to_string());
        let stat = get_summary_stats(&gate_fn);
        stats.insert(*width, stat);
    }
    stats
}

#[test]
fn test_gatify_one_hot() {
    let _ = env_logger::builder().is_test(true).try_init();
    let stats = gather_stats_for_widths(&[1, 2, 3, 4, 5, 6, 7, 8], |builder, bit_count| {
        let arg = builder.add_input("arg".to_string(), bit_count);
        let one_hot = gatify_one_hot(&mut *builder, &arg, true);
        builder.add_output("result".to_string(), one_hot);
    });
    #[rustfmt::skip]
    let want = &[
        (1, SummaryStats { live_nodes: 1, deepest_path: 1, fanout_histogram: btreemap!{} }),
        (2, SummaryStats { live_nodes: 4, deepest_path: 2, fanout_histogram: btreemap!{2 => 1} }),
        (3, SummaryStats { live_nodes: 8, deepest_path: 3, fanout_histogram: btreemap!{2 => 1, 3 => 1, 1 => 2} }),
        (4, SummaryStats { live_nodes: 12, deepest_path: 4, fanout_histogram: btreemap!{2 => 2, 1 => 3, 0 => 1, 4 => 1, 3 => 1} }),
        (5, SummaryStats { live_nodes: 17, deepest_path: 4, fanout_histogram: btreemap!{4 => 1, 5 => 1, 2 => 1, 3 => 2, 1 => 6, 0 => 2} }),
        (6, SummaryStats { live_nodes: 22, deepest_path: 5, fanout_histogram: btreemap!{1 => 7, 6 => 1, 2 => 3, 4 => 1, 3 => 2, 5 => 1, 0 => 4} }),
        (7, SummaryStats { live_nodes: 27, deepest_path: 5, fanout_histogram: btreemap!{4 => 1, 3 => 4, 0 => 7, 7 => 1, 1 => 9, 2 => 2, 5 => 1, 6 => 1} }),
        (8, SummaryStats { live_nodes: 32, deepest_path: 5, fanout_histogram: btreemap!{6 => 1, 3 => 3, 4 => 2, 5 => 1, 7 => 1, 8 => 1, 0 => 11, 1 => 9, 2 => 5} }),
    ];
    for &(bits, ref expected) in want {
        let got = stats.get(&bits).unwrap();
        assert_eq!(got, expected, "for width {}", bits);
    }
    // Validate that "want" has full coverage of the keys in "stats".
    for key in stats.keys() {
        assert!(
            want.iter().any(|&(bits, _)| bits == *key),
            "missing width {}",
            key
        );
    }
}

#[test]
fn test_gatify_ule() {
    let _ = env_logger::builder().is_test(true).try_init();
    let stats = gather_stats_for_widths(&[1, 2, 3, 4, 5, 6, 7, 8], |builder, bit_count| {
        let arg1 = builder.add_input("arg1".to_string(), bit_count);
        let arg2 = builder.add_input("arg2".to_string(), bit_count);
        const TEXT_ID: usize = 3;
        let result = gatify_ule_via_bit_tests(&mut *builder, TEXT_ID, &arg1, &arg2);
        builder.add_output("result".to_string(), result.into());
    });
    let mut sorted_stats = stats.iter().collect::<Vec<_>>();
    sorted_stats.sort_by_key(|(bits, _)| *bits);
    for (bits, stat) in sorted_stats {
        log::info!(
            "({}, SummaryStats {{ live_nodes: {}, deepest_path: {} }})",
            bits,
            stat.live_nodes,
            stat.deepest_path
        );
    }
    #[rustfmt::skip]
    let want = &[
        (1, SummaryStats { live_nodes: 6, deepest_path: 4, fanout_histogram: btreemap!{0 => 1, 1 => 2, 2 => 1, 3 => 2} }),
        (2, SummaryStats { live_nodes: 14, deepest_path: 6, fanout_histogram: btreemap!{0 => 2, 1 => 6, 2 => 3, 3 => 4} }),
        (3, SummaryStats { live_nodes: 22, deepest_path: 8, fanout_histogram: btreemap!{0 => 4, 1 => 9, 2 => 5, 3 => 7} }),
        (4, SummaryStats { live_nodes: 31, deepest_path: 9, fanout_histogram: btreemap!{0 => 6, 1 => 14, 2 => 5, 3 => 10, 4 => 1} }),
        (5, SummaryStats { live_nodes: 40, deepest_path: 10, fanout_histogram: btreemap!{0 => 9, 1 => 18, 2 => 7, 3 => 11, 4 => 2, 5 => 1} }),
        (6, SummaryStats { live_nodes: 49, deepest_path: 11, fanout_histogram: btreemap!{0 => 13, 1 => 22, 2 => 8, 3 => 14, 4 => 1, 5 => 2, 6 => 1} }),
        (7, SummaryStats { live_nodes: 59, deepest_path: 11, fanout_histogram: btreemap!{0 => 17, 1 => 26, 2 => 11, 3 => 16, 4 => 1, 5 => 1, 6 => 2, 7 => 1} }),
        (8, SummaryStats { live_nodes: 69, deepest_path: 11, fanout_histogram: btreemap!{0 => 22, 1 => 31, 2 => 11, 3 => 20, 4 => 1, 5 => 1, 6 => 1, 7 => 2, 8 => 1} }),
    ];
    for &(bits, ref expected) in want {
        let got = stats.get(&bits).unwrap();
        assert_eq!(got, expected, "for width {}", bits);
    }
    // Validate that "want" has full coverage of the keys in "stats".
    for key in stats.keys() {
        assert!(
            want.iter().any(|&(bits, _)| bits == *key),
            "missing width {}",
            key
        );
    }
}

/// Tests that we can convert the bf16 multiplier in the DSLX standard library.
#[test_case(Opt::Yes)]
#[test_case(Opt::No)]
fn test_gatify_bf16_mul(opt: Opt) {
    let _ = env_logger::builder().is_test(true).try_init();
    let dslx = "import bfloat16;
fn bf16_mul(x: bfloat16::BF16, y: bfloat16::BF16) -> bfloat16::BF16 {
    bfloat16::mul(x, y)
}";
    let fake_path = Path::new("test.x");
    let ir =
        xlsynth::convert_dslx_to_ir_text(dslx, fake_path, &xlsynth::DslxConvertOptions::default())
            .unwrap();
    log::info!("Unoptimized IR:\n{}", ir.ir);
    let ir_top = "__test__bf16_mul";
    let xlsynth_ir_package = xlsynth::IrPackage::parse_ir(&ir.ir, Some(ir_top)).unwrap();
    let optimized_ir_package = xlsynth::optimize_ir(&xlsynth_ir_package, ir_top).unwrap();
    let optimized_ir_text = optimized_ir_package.to_string();
    log::info!("Optimized IR:\n{}", optimized_ir_text);

    let stats = do_test_ir_conversion(&optimized_ir_text, opt);
    if opt == Opt::Yes {
        assert_within!(stats.live_nodes as isize, 1153 as isize, 20 as isize);
        assert_within!(stats.deepest_path as isize, 109 as isize, 10 as isize);
    }
}

/// Tests that we can convert the bf16 adder in the DSLX standard library.
#[test_case(Opt::Yes)]
#[test_case(Opt::No)]
fn test_gatify_bf16_add(opt: Opt) {
    let _ = env_logger::builder().is_test(true).try_init();
    let dslx = "import bfloat16;
fn bf16_add(x: bfloat16::BF16, y: bfloat16::BF16) -> bfloat16::BF16 {
    bfloat16::add(x, y)
}";
    let fake_path = Path::new("test.x");
    let ir =
        xlsynth::convert_dslx_to_ir_text(dslx, fake_path, &xlsynth::DslxConvertOptions::default())
            .unwrap();
    log::info!("Unoptimized IR:\n{}", ir.ir);
    let ir_top = "__test__bf16_add";
    let xlsynth_ir_package = xlsynth::IrPackage::parse_ir(&ir.ir, Some(ir_top)).unwrap();
    let optimized_ir_package = xlsynth::optimize_ir(&xlsynth_ir_package, ir_top).unwrap();
    let optimized_ir_text = optimized_ir_package.to_string();
    log::info!("Optimized IR:\n{}", optimized_ir_text);

    let stats = do_test_ir_conversion(&optimized_ir_text, opt);
    if opt == Opt::Yes {
        assert_within!(stats.live_nodes as isize, 1292 as isize, 20 as isize);
        assert_within!(stats.deepest_path as isize, 130 as isize, 10 as isize);
    }
}

#[test_case(Opt::No; "fold=false")]
#[test_case(Opt::Yes; "fold=true")]
fn test_fail_macro_gatify(opt: Opt) {
    let dslx = r#"fn top(x: bits[N]) -> bits[N] {
        fail!("boom", x)
    }"#;
    let _ = env_logger::builder().is_test(true).try_init();
    let module = format!("const N: u32 = u32:{};\n{}", 1, dslx);
    let path = Path::new("sample.x");
    let ir = xlsynth::convert_dslx_to_ir(&module, path, &xlsynth::DslxConvertOptions::default())
        .unwrap();
    let ir_package = &ir.ir;
    let ir_text = ir_package.to_string();
    do_test_ir_conversion_no_equiv(&ir_text, opt);
}

#[test_case(Opt::No; "fold=false")]
#[test_case(Opt::Yes; "fold=true")]
fn test_assert_macro_gatify(opt: Opt) {
    let dslx = r#"fn top(x: bits[N]) -> bits[N] {
        assert!(x == x, "must_equal");
        x
    }"#;
    let _ = env_logger::builder().is_test(true).try_init();
    let module = format!("const N: u32 = u32:{};\n{}", 1, dslx);
    let path = Path::new("sample.x");
    let ir = xlsynth::convert_dslx_to_ir(&module, path, &xlsynth::DslxConvertOptions::default())
        .unwrap();
    let ir_package = &ir.ir;
    let ir_text = ir_package.to_string();
    do_test_ir_conversion_no_equiv(&ir_text, opt);
}

#[test]
fn test_bit_slice_update_width_truncates_update() {
    // Regression IR that previously caused a panic: update value wider than the
    // destination bits. Now we expect successful lowering and semantic
    // equivalence.
    let ir_text = "package fuzz_test
fn fuzz_test(input: bits[7]) -> bits[1] {
  literal.2: bits[1] = literal(value=0, id=2)
  ret bit_slice_update.3: bits[1] = bit_slice_update(literal.2, input, input)
}
";

    let _ = env_logger::builder().is_test(true).try_init();

    // Use existing helper to convert and validate equivalence.
    do_test_ir_conversion(ir_text, Opt::No);
}

#[test_case(65, 4, Opt::No; "in65_out4_fold_no")]
#[test_case(65, 4, Opt::Yes; "in65_out4_fold_yes")]
#[test_case(96, 4, Opt::No; "in96_out4_fold_no")]
#[test_case(96, 4, Opt::Yes; "in96_out4_fold_yes")]
fn test_decode_wide_input_ir_to_gates(input_bits: u32, output_bits: u32, opt: Opt) {
    do_test_ir_conversion(
        &format!(
            "package sample\nfn do_decode_wide(x: bits[{input_bits}]) -> bits[{output_bits}] {{\n    ret result: bits[{output_bits}] = decode(x, width={output_bits}, id=2)\n}}"
        ),
        opt,
    );
}
