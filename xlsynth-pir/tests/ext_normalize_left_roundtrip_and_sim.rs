// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrValue;
use xlsynth_pir::desugar_extensions::{
    ExtensionEmitMode, desugar_extensions_in_fn, emit_package_with_extension_mode,
};
use xlsynth_pir::ir::{Fn, NodePayload};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;

fn build_ext_normalize_left_ir(
    width: usize,
    normalized_bit_count: usize,
    shift_offset: usize,
    clz_bit_count: Option<usize>,
) -> String {
    let ret_ty = match clz_bit_count {
        Some(clz_bit_count) => format!("(bits[{normalized_bit_count}], bits[{clz_bit_count}])"),
        None => format!("bits[{normalized_bit_count}]"),
    };
    let clz_attr = clz_bit_count
        .map(|clz_bit_count| format!(", clz_bit_count={clz_bit_count}"))
        .unwrap_or_default();
    format!(
        "package test\n\nfn f(arg: bits[{width}] id=1) -> {ret_ty} {{\n  ret r: {ret_ty} = ext_normalize_left(arg, shift_offset={shift_offset}, normalized_bit_count={normalized_bit_count}{clz_attr}, id=2)\n}}\n"
    )
}

fn get_ext_normalize_left_count(f: &Fn) -> usize {
    f.nodes
        .iter()
        .filter(|n| matches!(n.payload, NodePayload::ExtNormalizeLeft { .. }))
        .count()
}

fn eval_success_value(f: &Fn, args: &[IrValue]) -> IrValue {
    match eval_fn(f, args) {
        FnEvalResult::Success(s) => s.value,
        FnEvalResult::Failure(failure) => {
            panic!("unexpected eval failure: {:?}", failure.assertion_failures)
        }
    }
}

fn clz_reference(width: usize, input: u64) -> usize {
    for leading_zero_count in 0..width {
        let bit_index = width - 1 - leading_zero_count;
        if ((input >> bit_index) & 1) == 1 {
            return leading_zero_count;
        }
    }
    width
}

fn normalized_reference(
    width: usize,
    normalized_bit_count: usize,
    shift_offset: usize,
    input: u64,
) -> u64 {
    let clz = clz_reference(width, input);
    let shift = clz.saturating_add(shift_offset);
    if shift >= normalized_bit_count {
        0
    } else {
        (input << shift) & ((1u64 << normalized_bit_count) - 1)
    }
}

#[test]
fn ext_normalize_left_round_trips_via_text_and_ffi() {
    let ir = build_ext_normalize_left_ir(4, 8, 1, Some(3));
    let pkg = {
        let mut p = Parser::new(&ir);
        p.parse_and_validate_package().expect("parse/validate")
    };
    let f = pkg.get_fn("f").expect("fn f present");
    assert_eq!(get_ext_normalize_left_count(f), 1);

    let text = pkg.to_string();
    assert!(
        text.contains(
            "ext_normalize_left(arg, shift_offset=1, normalized_bit_count=8, clz_bit_count=3, id=2)"
        ),
        "expected ext_normalize_left to appear in emitted text:\n{}",
        text
    );
    let reparsed_pkg = {
        let mut p = Parser::new(&text);
        p.parse_and_validate_package().expect("re-parse/validate")
    };
    assert_eq!(
        get_ext_normalize_left_count(reparsed_pkg.get_fn("f").unwrap()),
        1
    );

    let wrapped = emit_package_with_extension_mode(&pkg, ExtensionEmitMode::AsFfiFunction)
        .expect("emit ffi-wrapped text");
    assert!(
        wrapped.contains("__pir_ext__ext_normalize_left__inw4__normw8__off1__clzw3"),
        "expected deterministic helper name in wrapped text:\n{}",
        wrapped
    );
    let wrapped_pkg = {
        let mut p = Parser::new(&wrapped);
        p.parse_and_validate_package()
            .expect("parse/validate wrapped text")
    };
    assert_eq!(wrapped_pkg.members.len(), 1);
    assert_eq!(
        get_ext_normalize_left_count(wrapped_pkg.get_fn("f").unwrap()),
        1
    );
}

#[test]
fn ext_normalize_left_eval_matches_reference_and_desugared_implementation() {
    for shift_offset in 0usize..=1usize {
        let ir = build_ext_normalize_left_ir(4, 8, shift_offset, Some(3));
        let pkg = {
            let mut p = Parser::new(&ir);
            p.parse_and_validate_package().expect("parse/validate")
        };
        let f = pkg.get_fn("f").expect("fn f present");
        let mut desugared = f.clone();
        desugar_extensions_in_fn(&mut desugared).expect("desugar ext_normalize_left");

        for input in 0u64..16u64 {
            let args = [IrValue::make_ubits(4, input).unwrap()];
            let got_ext = eval_success_value(f, &args);
            let got_desugared = eval_success_value(&desugared, &args);
            assert_eq!(
                got_ext, got_desugared,
                "desugared mismatch at shift_offset={shift_offset} input={input}"
            );

            let normalized =
                IrValue::make_ubits(8, normalized_reference(4, 8, shift_offset, input)).unwrap();
            let raw_clz = IrValue::make_ubits(3, clz_reference(4, input) as u64).unwrap();
            let expected = IrValue::make_tuple(&[normalized, raw_clz]);
            assert_eq!(
                got_ext, expected,
                "reference mismatch at shift_offset={shift_offset} input={input}"
            );
        }
    }
}
