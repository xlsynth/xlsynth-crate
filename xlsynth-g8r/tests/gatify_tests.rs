// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::path::Path;
use test_case::{test_case, test_matrix};
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::get_summary_stats::{get_summary_stats, SummaryStats};
use xlsynth_g8r::ir2gate::{gatify, gatify_ule_via_bit_tests, GatifyOptions};
use xlsynth_g8r::ir2gate_utils::gatify_one_hot;
use xlsynth_g8r::xls_ir::ir_parser;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Opt {
    Yes,
    No,
}

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

bit_count_test_cases!(test_dslx_array_to_gates, |input_bits: u32,
                                                 opt: Opt|
 -> () {
    do_test_dslx_conversion(
        input_bits,
        opt,
        "fn do_array(x: uN[N], y: uN[N], z: uN[N]) -> uN[N][3] { [x, y, z] }",
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
        (1, SummaryStats { live_nodes: 1, deepest_path: 1 }),
        (2, SummaryStats { live_nodes: 4, deepest_path: 2 }),
        (3, SummaryStats { live_nodes: 8, deepest_path: 3 }),
        (4, SummaryStats { live_nodes: 12, deepest_path: 4 }),
        (5, SummaryStats { live_nodes: 17, deepest_path: 4 }),
        (6, SummaryStats { live_nodes: 22, deepest_path: 5 }),
        (7, SummaryStats { live_nodes: 27, deepest_path: 5 }),
        (8, SummaryStats { live_nodes: 32, deepest_path: 5 }),
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
        (1, SummaryStats { live_nodes: 6, deepest_path: 4 }),
        (2, SummaryStats { live_nodes: 14, deepest_path: 6 }),
        (3, SummaryStats { live_nodes: 22, deepest_path: 8 }),
        (4, SummaryStats { live_nodes: 31, deepest_path: 9 }),
        (5, SummaryStats { live_nodes: 40, deepest_path: 10 }),
        (6, SummaryStats { live_nodes: 49, deepest_path: 11 }),
        (7, SummaryStats { live_nodes: 59, deepest_path: 11 }),
        (8, SummaryStats { live_nodes: 69, deepest_path: 11 }),
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
        assert_eq!(stats.live_nodes, 1172);
        assert_eq!(stats.deepest_path, 109);
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
        assert_eq!(stats.live_nodes, 1303);
        assert_eq!(stats.deepest_path, 130);
    }
}
