// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use test_case::{test_case, test_matrix};
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::ir2gate::{gatify, GatifyOptions};
use xlsynth_g8r::ir_parser;

fn do_test_ir_conversion(ir_package_text: &str, fold: bool) {
    // Now we'll parse the IR and turn it into a gate function.
    let mut parser = ir_parser::Parser::new(&ir_package_text);
    let ir_package = parser.parse_package().unwrap();
    let ir_top = ir_package.get_top().unwrap();
    let gate_fn = gatify(
        &ir_top,
        GatifyOptions {
            fold,
            check_equivalence: false,
        },
    )
    .unwrap();

    /*
    // Push the zero value through the gate function and check it matches.
    let input_zero = todo!();
    let gate_outputs = gate_sim::eval(&gate_fn, input_zero);

    let ir_outputs = ir_top.interpret(&[input_zero]);
    assert_eq!(gate_outputs, ir_outputs);
    */

    check_equivalence::validate_same_fn(&ir_top, &gate_fn)
        .expect("should validate IR to gate function equivalence");
}

#[test_case(1, false; "bit_count=1, fold=false")]
#[test_case(1, true; "bit_count=1, fold=true")]
#[test_case(2, false; "bit_count=2, fold=false")]
#[test_case(2, true; "bit_count=2, fold=true")]
fn test_prio_sel_ir_binary(bit_count: u32, fold: bool) {
    let ir_text_tmpl = "package sample
fn do_prio_sel(sel: bits[2], a: bits[$BIT_COUNT], b: bits[$BIT_COUNT], default: bits[$BIT_COUNT]) -> bits[$BIT_COUNT] {
    ret result: bits[$BIT_COUNT] = priority_sel(sel, cases=[a, b], default=default, id=5)
}
";
    let ir_text = ir_text_tmpl.replace("$BIT_COUNT", &bit_count.to_string());

    do_test_ir_conversion(&ir_text, fold);
}

#[test]
fn test_tuple_index_ir_to_gates() {
    let _ = env_logger::builder().is_test(true).try_init();
    let ir_text = "package sample
fn do_tuple_index(t: (bits[1], bits[1])) -> bits[1] {
    ret result: bits[1] = tuple_index(t, index=0, id=2)
}
";
    do_test_ir_conversion(&ir_text, false);
}

fn do_test_dslx_conversion(input_bits: u32, fold: bool, dslx_text: &str) {
    let _ = env_logger::builder().is_test(true).try_init();

    let module = format!("const N: u32 = u32:{};\n{}", input_bits, dslx_text);
    let path = Path::new("sample.x");
    let ir = xlsynth::convert_dslx_to_ir(&module, path, &xlsynth::DslxConvertOptions::default())
        .unwrap();
    let ir_package = &ir.ir;
    let ir_text = ir_package.to_string();
    log::info!("IR: {}", ir_text);

    do_test_ir_conversion(&ir_text, fold);
}

macro_rules! bit_count_test_cases {
    ($test_name:ident, $lambda:expr) => {
        #[test_case(1, false; "bit_count=1, fold=false")]
        #[test_case(2, false; "bit_count=2, fold=false")]
        #[test_case(3, false; "bit_count=3, fold=false")]
        #[test_case(4, false; "bit_count=4, fold=false")]
        #[test_case(5, false; "bit_count=5, fold=false")]
        #[test_case(6, false; "bit_count=6, fold=false")]
        #[test_case(7, false; "bit_count=7, fold=false")]
        #[test_case(8, false; "bit_count=8, fold=false")]
        #[test_case(1, true; "bit_count=1, fold=true")]
        #[test_case(2, true; "bit_count=2, fold=true")]
        #[test_case(3, true; "bit_count=3, fold=true")]
        #[test_case(4, true; "bit_count=4, fold=true")]
        #[test_case(5, true; "bit_count=5, fold=true")]
        #[test_case(6, true; "bit_count=6, fold=true")]
        #[test_case(7, true; "bit_count=7, fold=true")]
        #[test_case(8, true; "bit_count=8, fold=true")]
        fn $test_name(input_bits: u32, fold: bool) {
            let _ = env_logger::builder().is_test(true).try_init();
            $lambda(input_bits, fold)
        }
    };
}

bit_count_test_cases!(test_or_zero_ir_to_gates, |input_bits: u32,
                                                 fold: bool|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_or_zero(x: uN[N]) -> uN[N] { x | zero!<uN[N]>() }",
    );
});

bit_count_test_cases!(test_and_ones_ir_to_gates, |input_bits: u32,
                                                  fold: bool|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_and_ones(x: uN[N]) -> uN[N] { x & all_ones!<uN[N]>() }",
    );
});

// This test has a problem with returning a parameter in the IR formatting.
/*
bit_count_test_cases!(test_identity_ir_to_gates, |input_bits: u32| -> () {
    do_test_dslx_conversion(input_bits, "fn do_identity(x: uN[N]) -> uN[N] { x }");
});
*/

bit_count_test_cases!(test_and_ir_to_gates, |input_bits: u32, fold: bool| -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_and(x: uN[N], y: uN[N]) -> uN[N] { x & y }",
    );
});

bit_count_test_cases!(test_or_ir_to_gates, |input_bits: u32, fold: bool| -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_or(x: uN[N], y: uN[N]) -> uN[N] { x | y }",
    );
});

bit_count_test_cases!(test_xor_ir_to_gates, |input_bits: u32, fold: bool| -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_xor(x: uN[N], y: uN[N]) -> uN[N] { x ^ y }",
    );
});

bit_count_test_cases!(test_ne_all_zeros_all_ones_to_gates, |input_bits: u32,
                                                            fold: bool|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_ne_all_zeros_all_ones() -> bool { zero!<uN[N]>() != all_ones!<uN[N]>() }",
    );
});

bit_count_test_cases!(test_add_ir_to_gates, |input_bits: u32, fold: bool| -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_add(x: uN[N], y: uN[N]) -> uN[N] { x + y }",
    );
});

bit_count_test_cases!(test_sign_ext_to_gates, |input_bits: u32,
                                               fold: bool|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "const NP1: u32 = N + u32:1;
        fn do_sign_ext(x: uN[N]) -> uN[NP1] { x as sN[NP1] as uN[NP1] }",
    );
});

bit_count_test_cases!(test_or_reduce_ir_to_gates, |input_bits: u32,
                                                   fold: bool|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_or_reduce(x: uN[N]) -> bool { or_reduce(x) }",
    );
});

bit_count_test_cases!(test_decode_to_gates, |input_bits: u32, fold: bool| -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "const TPN: u32 = u32:1 << N;
        fn do_decode(x: uN[N]) -> uN[TPN] { decode<uN[TPN]>(x) }",
    );
});

bit_count_test_cases!(test_priority_sel_match_ir_to_gates, |input_bits: u32,
                                                            fold: bool|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        r#"fn do_priority_sel(s: bool, x: uN[N], y: uN[N]) -> uN[N] {
    match s {
        false => x,
        true => y,
    }
}"#,
    );
});

bit_count_test_cases!(test_neg_ir_to_gates, |input_bits: u32, fold: bool| -> () {
    do_test_dslx_conversion(input_bits, fold, "fn do_neg(x: uN[N]) -> uN[N] { -x }");
});

bit_count_test_cases!(test_not_ir_to_gates, |input_bits: u32, fold: bool| -> () {
    do_test_dslx_conversion(input_bits, fold, "fn do_not(x: uN[N]) -> uN[N] { !x }");
});

bit_count_test_cases!(test_concat_ir_to_gates, |input_bits: u32,
                                                fold: bool|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "const NT2: u32 = N * u32:2;
        fn do_concat(x: uN[N], y: uN[N]) -> uN[NT2] { x ++ y }",
    );
});

bit_count_test_cases!(test_sub_to_gates, |input_bits: u32, fold: bool| -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_sub(x: uN[N], y: uN[N]) -> uN[N] { x - y }",
    );
});

bit_count_test_cases!(test_nand_ir_to_gates, |input_bits: u32, fold: bool| -> () {
    let ir_tmpl = "package sample
fn do_nand(x: bits[$BIT_COUNT], y: bits[$BIT_COUNT]) -> bits[$BIT_COUNT] {
  ret nand.3: bits[$BIT_COUNT] = nand(x, y, id=3)
}";
    let ir_text = ir_tmpl.replace("$BIT_COUNT", &input_bits.to_string());
    do_test_ir_conversion(&ir_text, fold);
});

bit_count_test_cases!(test_encode_dslx_to_gates, |input_bits: u32,
                                                  fold: bool|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_encode(x: uN[N]) -> u32 { encode(x) as u32 }",
    );
});

bit_count_test_cases!(test_tuple_index_to_gates, |input_bits: u32,
                                                  fold: bool|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_tuple_index(t: (u2, uN[N], u2)) -> u2 { t.0 + t.2 }",
    );
});

bit_count_test_cases!(test_array_index_to_gates, |input_bits: u32,
                                                  fold: bool|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_array_index(a: uN[N][N], index: u32) -> uN[N] { a[index] }",
    );
});

// Emits a priority select via the DSLX builtin.
bit_count_test_cases!(test_priority_sel_builtin_to_gates, |input_bits: u32,
                                                           fold: bool|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_priority_sel_builtin(s: uN[N], cases: uN[N][N], default_value: uN[N]) -> uN[N] {
            priority_sel(s, cases, default_value)
        }",
    );
});

// Emits a one-hot-select via the DSLX builtin.
bit_count_test_cases!(test_one_hot_select_builtin_to_gates, |input_bits: u32,
                                                             fold: bool|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_one_hot_select_builtin(s: uN[N], cases: uN[N][N]) -> uN[N] { one_hot_sel(s, cases) }",
    );
});

bit_count_test_cases!(test_shrl_by_u32_to_gates, |input_bits: u32,
                                                  fold: bool|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_shrl_by_u32(x: uN[N], amount: u32) -> uN[N] { x >> amount }",
    );
});

bit_count_test_cases!(test_shrl_to_gates, |input_bits: u32, fold: bool| -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_shrl(x: uN[N], amount: uN[N]) -> uN[N] { x >> amount }",
    );
});

bit_count_test_cases!(test_shll_dslx_to_gates, |input_bits: u32,
                                                fold: bool|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_shll(x: uN[N], amount: u32) -> uN[N] { x << amount }",
    );
});

bit_count_test_cases!(test_sel_cond_dslx_to_gates, |input_bits: u32,
                                                    fold: bool|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_sel_cond(s: bool, on_true: uN[N], on_false: uN[N]) -> uN[N] { if s { on_true } else { on_false } }",
    );
});

bit_count_test_cases!(test_width_slice_static_start_to_gates, |input_bits: u32,
                                                               fold: bool|
 -> () {
    let ir_tmpl = "package sample
fn f(x: bits[32]) -> bits[$BIT_COUNT] {
    ret result.2: bits[$BIT_COUNT] = bit_slice(x, start=1, width=$BIT_COUNT, id=2)
}";
    let ir_text = ir_tmpl.replace("$BIT_COUNT", &input_bits.to_string());
    do_test_ir_conversion(&ir_text, fold);
});

// -- comparisons

bit_count_test_cases!(test_eq_ir_to_gates, |input_bits: u32, fold: bool| -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_eq(x: uN[N], y: uN[N]) -> bool { x == y }",
    );
});

bit_count_test_cases!(test_ne_ir_to_gates, |input_bits: u32, fold: bool| -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_ne_all_ones(x: uN[N]) -> bool { x != all_ones!<uN[N]>() }",
    );
});

bit_count_test_cases!(test_eqz_ir_to_gates, |input_bits: u32, fold: bool| -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_eqz(x: uN[N]) -> bool { x == zero!<uN[N]>() }",
    );
});

bit_count_test_cases!(test_eq_all_zeros_all_ones_to_gates, |input_bits: u32,
                                                            fold: bool|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_eq_all_zeros_all_ones() -> bool { zero!<uN[N]>() == all_ones!<uN[N]>() }",
    );
});

bit_count_test_cases!(test_ugt_ir_to_gates, |input_bits: u32, fold: bool| -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_ugt(x: uN[N], y: uN[N]) -> bool { x > y }",
    );
});

bit_count_test_cases!(test_ult_ir_to_gates, |input_bits: u32, fold: bool| -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_ult(x: uN[N], y: uN[N]) -> bool { x < y }",
    );
});

bit_count_test_cases!(test_ule_ir_to_gates, |input_bits: u32, fold: bool| -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_ule(x: uN[N], y: uN[N]) -> bool { x <= y }",
    );
});

bit_count_test_cases!(test_uge_ir_to_gates, |input_bits: u32, fold: bool| -> () {
    do_test_dslx_conversion(
        input_bits,
        fold,
        "fn do_uge(x: uN[N], y: uN[N]) -> bool { x >= y }",
    );
});

bit_count_test_cases!(test_encode_ir_to_gates, |input_bits: u32,
                                                fold: bool|
 -> () {
    let output_bits = (input_bits as f32).log2().ceil() as u32;
    log::info!("input_bits: {}, output_bits: {}", input_bits, output_bits);
    do_test_ir_conversion(
        &format!(
            "package sample
fn do_encode(x: bits[{input_bits}]) -> bits[{output_bits}] {{
    ret result: bits[{output_bits}] = encode(x, id=2)
}}"
        ),
        fold,
    );
});

#[test_matrix(
    4..8,
    1..=4,
    [false, true]
)]
fn test_decode_ir_to_gates(input_bits: u32, output_bits: u32, fold: bool) {
    do_test_ir_conversion(
        &format!(
            "package sample
fn do_decode(x: bits[{input_bits}]) -> bits[{output_bits}] {{
    ret result: bits[{output_bits}] = decode(x, width={output_bits}, id=2)
}}"
        ),
        fold,
    );
}
