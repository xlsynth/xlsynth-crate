// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth::IrValue;
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_pir::ir;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;

#[derive(Debug, Clone)]
struct MaskLowPrepSample {
    output_width: u8,
    count_width: u8,
    variant: MaskLowVariant,
}

#[derive(Debug, Clone, Copy)]
enum MaskLowVariant {
    ValidSub,
    ValidAddRhs,
    ValidAddLhs,
    WrongShiftLhsLiteral,
    WrongSubRhsLiteral,
    WrongAddAllOnesLiteral,
    WrongShiftDirection,
    MismatchedConsumerWidth,
}

impl MaskLowVariant {
    fn from_byte(raw: u8) -> Self {
        match raw % 8 {
            0 => MaskLowVariant::ValidSub,
            1 => MaskLowVariant::ValidAddRhs,
            2 => MaskLowVariant::ValidAddLhs,
            3 => MaskLowVariant::WrongShiftLhsLiteral,
            4 => MaskLowVariant::WrongSubRhsLiteral,
            5 => MaskLowVariant::WrongAddAllOnesLiteral,
            6 => MaskLowVariant::WrongShiftDirection,
            _ => MaskLowVariant::MismatchedConsumerWidth,
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

fn build_ir_text(sample: &MaskLowPrepSample) -> (String, bool) {
    let output_width = sample_output_width(sample.output_width);
    let count_width = sample_count_width(sample.count_width);
    let all_ones = all_ones_value(output_width);
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
        ),
        MaskLowVariant::WrongShiftLhsLiteral => {
            // For bits[1], zero is the only representable literal value that is
            // not the required literal one.
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
            )
        }
        MaskLowVariant::WrongSubRhsLiteral => {
            let bad_rhs = if output_width == 1 { 0 } else { 2 };
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
                false,
            )
        }
        MaskLowVariant::WrongAddAllOnesLiteral => {
            let bad_all_ones = all_ones.saturating_sub(1);
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
                false,
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
        ),
        MaskLowVariant::MismatchedConsumerWidth => {
            let ret_width = output_width + 1;
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
                false,
            )
        }
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
    let (ir_text, expected_rewrite) = build_ir_text(&sample);
    let orig_fn = parse_top_fn(&ir_text);
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
        has_rewrite, expected_rewrite,
        "unexpected mask-low rewrite decision for sample {sample:?}\ninput:\n{ir_text}\nprepared:\n{prepared_text}"
    );

    let count_width = sample_count_width(sample.count_width);
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
            "prep_for_gatify changed semantics for sample {sample:?} count={count}\ninput:\n{ir_text}\nprepared:\n{prepared_text}"
        );
    }
});
