// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrValue;
use xlsynth_pir::desugar_extensions::{
    ExtensionEmitMode, desugar_extensions_in_fn, emit_package_with_extension_mode,
};
use xlsynth_pir::ir::{Fn, NodePayload};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::math::ceil_log2;

fn build_ext_mask_low_ir(output_width: usize, count_width: usize) -> String {
    format!(
        "package test\n\nfn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{\n  ret r: bits[{output_width}] = ext_mask_low(count, id=2)\n}}\n"
    )
}

fn get_ext_mask_low_count(f: &Fn) -> usize {
    f.nodes
        .iter()
        .filter(|n| matches!(n.payload, NodePayload::ExtMaskLow { .. }))
        .count()
}

fn parse_fn(ir: &str) -> Fn {
    let mut p = Parser::new(ir);
    let pkg = p.parse_and_validate_package().expect("parse/validate");
    pkg.get_fn("f").expect("fn f present").clone()
}

fn eval_success(f: &Fn, count_width: usize, count: u64) -> IrValue {
    let args = [IrValue::make_ubits(count_width, count).expect("count literal")];
    match eval_fn(f, &args) {
        FnEvalResult::Success(s) => s.value,
        FnEvalResult::Failure(f) => panic!("unexpected eval failure: {:?}", f),
    }
}

fn expected_mask(output_width: usize, count: usize) -> IrValue {
    let bits = (0..output_width).map(|i| count > i).collect::<Vec<bool>>();
    IrValue::from_bits(&xlsynth::IrBits::from_lsb_is_0(&bits))
}

#[test]
fn ext_mask_low_round_trips_via_text() {
    let ir = build_ext_mask_low_ir(8, 4);
    let f = parse_fn(&ir);
    assert_eq!(get_ext_mask_low_count(&f), 1);
    let text = f.to_string();
    assert!(
        text.contains("ext_mask_low(count, id=2)"),
        "expected ext_mask_low in emitted text:\n{}",
        text
    );
    let reparsed = parse_fn(&format!("package test\n\n{text}\n"));
    assert_eq!(get_ext_mask_low_count(&reparsed), 1);
}

#[test]
fn ext_mask_low_round_trips_via_ffi_wrapped_text() {
    let ir = build_ext_mask_low_ir(8, 4);
    let pkg = {
        let mut p = Parser::new(&ir);
        p.parse_and_validate_package().expect("parse/validate")
    };
    let wrapped = emit_package_with_extension_mode(&pkg, ExtensionEmitMode::AsFfiFunction)
        .expect("emit ffi-wrapped text");
    assert!(
        wrapped.contains("__pir_ext__ext_mask_low__outw8__countw4"),
        "expected deterministic helper name in wrapped text:\n{}",
        wrapped
    );
    assert!(
        wrapped.contains("invoke(count, to_apply=__pir_ext__ext_mask_low__outw8__countw4, id=2)"),
        "expected invoke of the ffi helper in wrapped text:\n{}",
        wrapped
    );

    let wrapped_pkg = {
        let mut p = Parser::new(&wrapped);
        p.parse_and_validate_package()
            .expect("parse/validate wrapped text")
    };
    let wrapped_f = wrapped_pkg.get_fn("f").expect("fn f present");
    assert_eq!(wrapped_pkg.members.len(), 1);
    assert!(
        wrapped_pkg
            .get_fn("__pir_ext__ext_mask_low__outw8__countw4")
            .is_none()
    );
    assert_eq!(get_ext_mask_low_count(wrapped_f), 1);
}

#[test]
fn ext_mask_low_explicit_eight_bit_cases() {
    let f = parse_fn(&build_ext_mask_low_ir(8, 4));
    for (count, expected) in [
        (0u64, 0x00u64),
        (1, 0x01),
        (2, 0x03),
        (7, 0x7f),
        (8, 0xff),
        (9, 0xff),
    ] {
        let got = eval_success(&f, 4, count);
        assert_eq!(
            got,
            IrValue::make_ubits(8, expected).unwrap(),
            "count={count}"
        );
    }
}

#[test]
fn ext_mask_low_eval_matches_reference_and_desugared_for_small_widths() {
    for output_width in 0usize..=16 {
        let natural_count_width = ceil_log2(output_width.saturating_add(1));
        for count_width in [0usize, natural_count_width, natural_count_width + 2] {
            let ir = build_ext_mask_low_ir(output_width, count_width);
            let f = parse_fn(&ir);
            let mut desugared = f.clone();
            desugar_extensions_in_fn(&mut desugared).expect("desugar ext_mask_low");
            assert_eq!(get_ext_mask_low_count(&desugared), 0);

            let max_count = if count_width >= 6 {
                32
            } else if count_width == 0 {
                0
            } else {
                (1u64 << count_width) - 1
            };
            for count in 0u64..=max_count {
                let got = eval_success(&f, count_width, count);
                let got_desugared = eval_success(&desugared, count_width, count);
                let expected = expected_mask(output_width, count as usize);
                assert_eq!(
                    got, expected,
                    "native mismatch output_width={output_width} count_width={count_width} count={count}"
                );
                assert_eq!(
                    got_desugared, expected,
                    "desugared mismatch output_width={output_width} count_width={count_width} count={count}"
                );
            }
        }
    }
}
