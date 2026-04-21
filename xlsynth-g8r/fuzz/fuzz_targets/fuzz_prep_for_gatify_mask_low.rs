// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth::IrValue;
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_pir::ir;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::math::ceil_log2;

#[derive(Debug, Clone)]
struct MaskLowPrepSample {
    output_width: u8,
    count_width: u8,
    variant: MaskLowVariant,
}

#[derive(Debug, Clone)]
struct BuiltMaskLowIr {
    text: String,
    expected_ext_mask_low: bool,
    expected_concat_eliminated: Option<bool>,
    count_width: usize,
}

#[derive(Debug, Clone, Copy)]
enum MaskLowVariant {
    ValidSub,
    ValidAddRhs,
    ValidAddLhs,
    ValidShiftedAllOnes,
    ValidSlicedShiftedAllOnes,
    ValidNotShiftedAllOnes,
    ValidShiftedShrlAllOnes,
    ValidShrlAllOnes,
    ValidZeroConcatLowMask,
    WrongShiftLhsLiteral,
    WrongSubRhsLiteral,
    WrongAddAllOnesLiteral,
    WrongShiftDirection,
    SharedShiftedAllOnesLiteral,
    MismatchedShiftedShrlCounts,
    MismatchedConsumerWidth,
    NonzeroConcatPrefix,
    UnboundedConcatCount,
}

impl MaskLowVariant {
    fn from_byte(raw: u8) -> Self {
        match raw % 18 {
            0 => Self::ValidSub,
            1 => Self::ValidAddRhs,
            2 => Self::ValidAddLhs,
            3 => Self::ValidShiftedAllOnes,
            4 => Self::ValidSlicedShiftedAllOnes,
            5 => Self::ValidNotShiftedAllOnes,
            6 => Self::ValidShiftedShrlAllOnes,
            7 => Self::ValidShrlAllOnes,
            8 => Self::ValidZeroConcatLowMask,
            9 => Self::WrongShiftLhsLiteral,
            10 => Self::WrongSubRhsLiteral,
            11 => Self::WrongAddAllOnesLiteral,
            12 => Self::WrongShiftDirection,
            13 => Self::SharedShiftedAllOnesLiteral,
            14 => Self::MismatchedShiftedShrlCounts,
            15 => Self::MismatchedConsumerWidth,
            16 => Self::NonzeroConcatPrefix,
            17 => Self::UnboundedConcatCount,
            _ => unreachable!("raw % 18 must be in 0..18"),
        }
    }
}

fn sample_from_data(data: &[u8]) -> MaskLowPrepSample {
    MaskLowPrepSample {
        output_width: data.first().copied().unwrap_or(0),
        count_width: data.get(1).copied().unwrap_or(0),
        variant: MaskLowVariant::from_byte(data.get(2).copied().unwrap_or(0)),
    }
}

fn sample_output_width(raw: u8) -> usize {
    usize::from(raw % 32) + 1
}

fn sample_count_width(raw: u8) -> usize {
    usize::from(raw % 6)
}

fn count_width_always_le_width(count_width: usize, width: usize) -> bool {
    let max_count = if count_width == 0 {
        0usize
    } else {
        (1usize << count_width) - 1
    };
    max_count <= width
}

fn bounded_count_width_for_width(width: usize) -> usize {
    let mut count_width = ceil_log2(width.saturating_add(1));
    if count_width > 0 && ((1usize << count_width) - 1) > width {
        count_width -= 1;
    }
    count_width
}

fn all_ones_value(width: usize) -> u128 {
    (1u128 << width) - 1
}

fn parse_top_fn(ir_text: &str) -> ir::Fn {
    let mut parser = Parser::new(ir_text);
    let pkg = parser
        .parse_and_validate_package()
        .unwrap_or_else(|e| panic!("constructed mask-low IR should parse:\n{ir_text}\n{e}"));
    pkg.get_top_fn().expect("constructed package has top").clone()
}

fn build_ir_text(sample: &MaskLowPrepSample) -> BuiltMaskLowIr {
    let output_width = sample_output_width(sample.output_width);
    let count_width = sample_count_width(sample.count_width);
    let all_ones = all_ones_value(output_width);
    let (text, expected_ext_mask_low, expected_concat_eliminated, actual_count_width) =
        match sample.variant {
            MaskLowVariant::ValidSub => (
                format!(
                    "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  one: bits[{output_width}] = literal(value=1, id=2)
  sh: bits[{output_width}] = shll(one, count, id=3)
  ret mask: bits[{output_width}] = sub(sh, one, id=4)
}}
"
                ),
                true,
                None,
                count_width,
            ),
            MaskLowVariant::ValidAddRhs => (
                format!(
                    "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  one: bits[{output_width}] = literal(value=1, id=2)
  all_ones: bits[{output_width}] = literal(value={all_ones}, id=3)
  sh: bits[{output_width}] = shll(one, count, id=4)
  ret mask: bits[{output_width}] = add(sh, all_ones, id=5)
}}
"
                ),
                true,
                None,
                count_width,
            ),
            MaskLowVariant::ValidAddLhs => (
                format!(
                    "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  one: bits[{output_width}] = literal(value=1, id=2)
  all_ones: bits[{output_width}] = literal(value={all_ones}, id=3)
  sh: bits[{output_width}] = shll(one, count, id=4)
  ret mask: bits[{output_width}] = add(all_ones, sh, id=5)
}}
"
                ),
                true,
                None,
                count_width,
            ),
            MaskLowVariant::ValidShiftedAllOnes => (
                format!(
                    "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  all_ones: bits[{output_width}] = literal(value={all_ones}, id=2)
  ret mask: bits[{output_width}] = shll(all_ones, count, id=3)
}}
"
                ),
                true,
                None,
                count_width,
            ),
            MaskLowVariant::ValidSlicedShiftedAllOnes => {
                let source_width = output_width.saturating_add(4);
                let source_all_ones = all_ones_value(source_width);
                (
                    format!(
                        "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  all_ones: bits[{source_width}] = literal(value={source_all_ones}, id=2)
  sh: bits[{source_width}] = shll(all_ones, count, id=3)
  ret mask: bits[{output_width}] = bit_slice(sh, start=3, width={output_width}, id=4)
}}
"
                    ),
                    true,
                    None,
                    count_width,
                )
            }
            MaskLowVariant::ValidNotShiftedAllOnes => (
                format!(
                    "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  all_ones: bits[{output_width}] = literal(value={all_ones}, id=2)
  high: bits[{output_width}] = shll(all_ones, count, id=3)
  ret low: bits[{output_width}] = not(high, id=4)
}}
"
                ),
                true,
                None,
                count_width,
            ),
            MaskLowVariant::ValidShiftedShrlAllOnes => (
                format!(
                    "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  all_ones: bits[{output_width}] = literal(value={all_ones}, id=2)
  low_from_top: bits[{output_width}] = shrl(all_ones, count, id=3)
  ret high: bits[{output_width}] = shll(low_from_top, count, id=4)
}}
"
                ),
                true,
                None,
                count_width,
            ),
            MaskLowVariant::ValidShrlAllOnes => (
                format!(
                    "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  all_ones: bits[{output_width}] = literal(value={all_ones}, id=2)
  ret low_from_top: bits[{output_width}] = shrl(all_ones, count, id=3)
}}
"
                ),
                true,
                None,
                count_width,
            ),
            MaskLowVariant::ValidZeroConcatLowMask => {
                let prefix_width = 4usize;
                let bounded_count_width = bounded_count_width_for_width(output_width);
                (
                    format!(
                        "package sample

top fn f(count: bits[{bounded_count_width}] id=1) -> bits[{}] {{
  zero: bits[{prefix_width}] = literal(value=0, id=2)
  all_ones: bits[{output_width}] = literal(value={all_ones}, id=3)
  high: bits[{output_width}] = shll(all_ones, count, id=4)
  low: bits[{output_width}] = not(high, id=5)
  ret out: bits[{}] = concat(zero, low, id=6)
}}
",
                        prefix_width + output_width,
                        prefix_width + output_width
                    ),
                    true,
                    Some(true),
                    bounded_count_width,
                )
            }
            MaskLowVariant::WrongShiftLhsLiteral => {
                let bad_lhs = if output_width == 1 { 0 } else { 2 };
                (
                    format!(
                        "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  bad_lhs: bits[{output_width}] = literal(value={bad_lhs}, id=2)
  one: bits[{output_width}] = literal(value=1, id=3)
  sh: bits[{output_width}] = shll(bad_lhs, count, id=4)
  ret mask: bits[{output_width}] = sub(sh, one, id=5)
}}
"
                    ),
                    false,
                    None,
                    count_width,
                )
            }
            MaskLowVariant::WrongSubRhsLiteral => {
                let bad_rhs = if output_width == 1 { 0 } else { 2 };
                let has_valid_shifted_all_ones = output_width == 1;
                (
                    format!(
                        "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  one: bits[{output_width}] = literal(value=1, id=2)
  bad: bits[{output_width}] = literal(value={bad_rhs}, id=3)
  sh: bits[{output_width}] = shll(one, count, id=4)
  ret mask: bits[{output_width}] = sub(sh, bad, id=5)
}}
"
                    ),
                    has_valid_shifted_all_ones,
                    None,
                    count_width,
                )
            }
            MaskLowVariant::WrongAddAllOnesLiteral => {
                let bad_all_ones = all_ones.saturating_sub(1);
                let has_valid_shifted_all_ones = output_width == 1;
                (
                    format!(
                        "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  one: bits[{output_width}] = literal(value=1, id=2)
  bad: bits[{output_width}] = literal(value={bad_all_ones}, id=3)
  sh: bits[{output_width}] = shll(one, count, id=4)
  ret mask: bits[{output_width}] = add(sh, bad, id=5)
}}
"
                    ),
                    has_valid_shifted_all_ones,
                    None,
                    count_width,
                )
            }
            MaskLowVariant::WrongShiftDirection => (
                format!(
                    "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  one: bits[{output_width}] = literal(value=1, id=2)
  sh: bits[{output_width}] = shrl(one, count, id=3)
  ret mask: bits[{output_width}] = sub(sh, one, id=4)
}}
"
                ),
                false,
                None,
                count_width,
            ),
            MaskLowVariant::SharedShiftedAllOnesLiteral => (
                format!(
                    "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  all_ones: bits[{output_width}] = literal(value={all_ones}, id=2)
  sh: bits[{output_width}] = shll(all_ones, count, id=3)
  ret mask: bits[{output_width}] = and(sh, all_ones, id=4)
}}
"
                ),
                false,
                None,
                count_width,
            ),
            MaskLowVariant::MismatchedShiftedShrlCounts => (
                format!(
                    "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{
  all_ones: bits[{output_width}] = literal(value={all_ones}, id=2)
  count_copy: bits[{count_width}] = identity(count, id=3)
  low_from_top: bits[{output_width}] = shrl(all_ones, count, id=4)
  ret high: bits[{output_width}] = shll(low_from_top, count_copy, id=5)
}}
"
                ),
                false,
                None,
                count_width,
            ),
            MaskLowVariant::MismatchedConsumerWidth => {
                let ret_width = output_width + 1;
                let has_valid_shifted_all_ones = output_width == 1;
                (
                    format!(
                        "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{ret_width}] {{
  one: bits[{output_width}] = literal(value=1, id=2)
  one_ret: bits[{ret_width}] = literal(value=1, id=3)
  sh: bits[{output_width}] = shll(one, count, id=4)
  wide: bits[{ret_width}] = zero_ext(sh, new_bit_count={ret_width}, id=5)
  ret mask: bits[{ret_width}] = sub(wide, one_ret, id=6)
}}
"
                    ),
                    has_valid_shifted_all_ones,
                    None,
                    count_width,
                )
            }
            MaskLowVariant::NonzeroConcatPrefix => {
                let prefix_width = 4usize;
                (
                    format!(
                        "package sample

top fn f(count: bits[{count_width}] id=1) -> bits[{}] {{
  prefix: bits[{prefix_width}] = literal(value=1, id=2)
  all_ones: bits[{output_width}] = literal(value={all_ones}, id=3)
  high: bits[{output_width}] = shll(all_ones, count, id=4)
  low: bits[{output_width}] = not(high, id=5)
  ret out: bits[{}] = concat(prefix, low, id=6)
}}
",
                        prefix_width + output_width,
                        prefix_width + output_width
                    ),
                    true,
                    Some(false),
                    count_width,
                )
            }
            MaskLowVariant::UnboundedConcatCount => {
                let prefix_width = 4usize;
                let unbounded_count_width = ceil_log2(output_width.saturating_add(1)) + 1;
                debug_assert!(!count_width_always_le_width(
                    unbounded_count_width,
                    output_width
                ));
                (
                    format!(
                        "package sample

top fn f(count: bits[{unbounded_count_width}] id=1) -> bits[{}] {{
  zero: bits[{prefix_width}] = literal(value=0, id=2)
  all_ones: bits[{output_width}] = literal(value={all_ones}, id=3)
  high: bits[{output_width}] = shll(all_ones, count, id=4)
  low: bits[{output_width}] = not(high, id=5)
  ret out: bits[{}] = concat(zero, low, id=6)
}}
",
                        prefix_width + output_width,
                        prefix_width + output_width
                    ),
                    true,
                    Some(false),
                    unbounded_count_width,
                )
            }
        };
    BuiltMaskLowIr {
        text,
        expected_ext_mask_low,
        expected_concat_eliminated,
        count_width: actual_count_width,
    }
}

fn eval_success(f: &ir::Fn, count_width: usize, count: u64) -> IrValue {
    let args = [IrValue::make_ubits(count_width, count).expect("count literal")];
    match eval_fn(f, &args) {
        FnEvalResult::Success(success) => success.value,
        FnEvalResult::Failure(failure) => panic!("unexpected eval failure: {failure:?}"),
    }
}

fuzz_target!(|data: &[u8]| {
    let _ = env_logger::builder().is_test(true).try_init();

    let sample = sample_from_data(data);
    let built = build_ir_text(&sample);
    let orig_fn = parse_top_fn(&built.text);
    let prepared = prep_for_gatify(
        &orig_fn,
        None,
        PrepForGatifyOptions {
            enable_rewrite_mask_low: true,
            enable_rewrite_nary_add: false,
            ..PrepForGatifyOptions::default()
        },
    );
    let prepared_text = prepared.to_string();
    let has_rewrite = prepared_text.contains("ext_mask_low(");
    assert_eq!(
        has_rewrite, built.expected_ext_mask_low,
        "unexpected mask-low rewrite decision for sample {sample:?}\ninput:\n{}\nprepared:\n{prepared_text}",
        built.text
    );
    if let Some(expected_eliminated) = built.expected_concat_eliminated {
        let concat_eliminated = !prepared_text.contains("concat(");
        assert_eq!(
            concat_eliminated, expected_eliminated,
            "unexpected zero-concat rewrite decision for sample {sample:?}\ninput:\n{}\nprepared:\n{prepared_text}",
            built.text
        );
    }

    let count_width = built.count_width;
    let max_count = if count_width == 0 {
        0
    } else {
        (1u64 << count_width) - 1
    };
    for count in 0..=max_count {
        let orig_value = eval_success(&orig_fn, count_width, count);
        let prepared_value = eval_success(&prepared, count_width, count);
        assert_eq!(
            prepared_value, orig_value,
            "prep_for_gatify changed semantics for sample {sample:?} count={count}\ninput:\n{}\nprepared:\n{prepared_text}",
            built.text
        );
    }
});
