// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrValue;
use xlsynth_pir::desugar_extensions::{
    ExtensionEmitMode, desugar_extensions_in_fn, emit_package_with_extension_mode,
};
use xlsynth_pir::ir::{Fn, NodePayload};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::math::ceil_log2;

fn build_ext_clz_ir(width: u64) -> String {
    let out_w = ceil_log2((width as usize).saturating_add(1));
    format!(
        "package test\n\nfn f(arg: bits[{width}] id=1) -> bits[{out_w}] {{\n  ret r: bits[{out_w}] = ext_clz(arg, id=2)\n}}\n"
    )
}

fn get_ext_clz_count(f: &Fn) -> usize {
    f.nodes
        .iter()
        .filter(|n| matches!(n.payload, NodePayload::ExtClz { .. }))
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

fn clz_reference(width: u64, input: u64) -> usize {
    let width = width as usize;
    for leading_zero_count in 0..width {
        let bit_index = width - 1 - leading_zero_count;
        if ((input >> bit_index) & 1) == 1 {
            return leading_zero_count;
        }
    }
    width
}

#[test]
fn ext_clz_round_trips_via_text() {
    let ir = build_ext_clz_ir(8);

    let pkg = {
        let mut p = Parser::new(&ir);
        p.parse_and_validate_package().expect("parse/validate")
    };
    let f = pkg.get_fn("f").expect("fn f present");
    assert_eq!(get_ext_clz_count(f), 1);

    let text = pkg.to_string();
    assert!(
        text.contains("ext_clz(arg, id=2)"),
        "expected ext_clz to appear in emitted text:\n{}",
        text
    );

    let reparsed_pkg = {
        let mut p = Parser::new(&text);
        p.parse_and_validate_package().expect("re-parse/validate")
    };
    let reparsed_f = reparsed_pkg.get_fn("f").expect("fn f present in reparsed");
    assert_eq!(get_ext_clz_count(reparsed_f), 1);
}

#[test]
fn ext_clz_round_trips_via_ffi_wrapped_text() {
    let ir = build_ext_clz_ir(8);

    let pkg = {
        let mut p = Parser::new(&ir);
        p.parse_and_validate_package().expect("parse/validate")
    };
    let wrapped = emit_package_with_extension_mode(&pkg, ExtensionEmitMode::AsFfiFunction)
        .expect("emit ffi-wrapped text");
    assert!(
        wrapped.contains("#[ffi_proto(\"\"\""),
        "expected ffi_proto attribute in wrapped text:\n{}",
        wrapped
    );
    assert!(
        wrapped.contains("__pir_ext__ext_clz__w8"),
        "expected deterministic helper name in wrapped text:\n{}",
        wrapped
    );
    assert!(
        wrapped.contains("invoke(arg, to_apply=__pir_ext__ext_clz__w8, id=2)"),
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
    assert!(wrapped_pkg.get_fn("__pir_ext__ext_clz__w8").is_none());
    assert_eq!(get_ext_clz_count(wrapped_f), 1);

    let rewrapped =
        emit_package_with_extension_mode(&wrapped_pkg, ExtensionEmitMode::AsFfiFunction).unwrap();
    assert_eq!(rewrapped, wrapped);
}

#[test]
fn ext_clz_simulation_matches_software_reference_for_small_widths() {
    for width in 1u64..=8u64 {
        let ir = build_ext_clz_ir(width);
        let pkg = {
            let mut p = Parser::new(&ir);
            p.parse_and_validate_package().expect("parse/validate")
        };
        let f = pkg.get_fn("f").expect("fn f present");

        let out_w = ceil_log2((width as usize).saturating_add(1));
        let limit = 1u64 << width;
        for input in 0u64..limit {
            let args = [IrValue::make_ubits(width as usize, input).unwrap()];
            let got = eval_success_value(f, &args);
            let expected = IrValue::make_ubits(out_w, clz_reference(width, input) as u64).unwrap();
            assert_eq!(got, expected, "mismatch at width={width} input={input}");
        }
    }
}

#[test]
fn ext_clz_eval_matches_desugared_implementation_for_small_widths() {
    for width in 1u64..=8u64 {
        let ir = build_ext_clz_ir(width);
        let pkg = {
            let mut p = Parser::new(&ir);
            p.parse_and_validate_package().expect("parse/validate")
        };
        let f = pkg.get_fn("f").expect("fn f present");
        let mut desugared = f.clone();
        desugar_extensions_in_fn(&mut desugared).expect("desugar ext_clz");

        let limit = 1u64 << width;
        for input in 0u64..limit {
            let args = [IrValue::make_ubits(width as usize, input).unwrap()];
            let got_ext = eval_success_value(f, &args);
            let got_desugared = eval_success_value(&desugared, &args);
            assert_eq!(
                got_ext, got_desugared,
                "desugared mismatch at width={width} input={input}"
            );
        }
    }
}
