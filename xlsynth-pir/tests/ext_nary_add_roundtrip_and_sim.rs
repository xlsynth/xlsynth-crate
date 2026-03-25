// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrValue;
use xlsynth_pir::desugar_extensions::{ExtensionEmitMode, emit_package_with_extension_mode};
use xlsynth_pir::ir::NodePayload;
use xlsynth_pir::ir_eval::eval_fn;
use xlsynth_pir::ir_parser::Parser;

#[test]
fn ext_nary_add_round_trips_via_text() {
    let ir = r#"package test

fn f(a: bits[3] id=1, b: bits[5] id=2, c: bits[9] id=3) -> bits[6] {
  ret r: bits[6] = ext_nary_add(a, b, c, id=4)
}
"#;

    let pkg = {
        let mut p = Parser::new(ir);
        p.parse_and_validate_package().expect("parse/validate")
    };
    let f = pkg.get_fn("f").expect("fn f present");
    let ext_count: usize = f
        .nodes
        .iter()
        .filter(|n| matches!(n.payload, NodePayload::ExtNaryAdd { .. }))
        .count();
    assert_eq!(ext_count, 1);

    let text = pkg.to_string();
    assert!(
        text.contains("ext_nary_add"),
        "expected ext_nary_add to appear in emitted text:\n{}",
        text
    );

    let pkg2 = {
        let mut p = Parser::new(&text);
        p.parse_and_validate_package().expect("re-parse/validate")
    };
    let f2 = pkg2.get_fn("f").expect("fn f present in reparsed");
    let ext_count2: usize = f2
        .nodes
        .iter()
        .filter(|n| matches!(n.payload, NodePayload::ExtNaryAdd { .. }))
        .count();
    assert_eq!(ext_count2, 1);
}

#[test]
fn ext_nary_add_handles_zero_operands() {
    let ir = r#"package test

fn f() -> bits[7] {
  ret r: bits[7] = ext_nary_add(id=1)
}
"#;
    let pkg = {
        let mut p = Parser::new(ir);
        p.parse_and_validate_package().expect("parse/validate")
    };
    let f = pkg.get_fn("f").expect("fn f present");
    let got = match eval_fn(f, &[]) {
        xlsynth_pir::ir_eval::FnEvalResult::Success(s) => s.value,
        xlsynth_pir::ir_eval::FnEvalResult::Failure(f) => {
            panic!("unexpected eval failure: {:?}", f.assertion_failures)
        }
    };
    assert_eq!(got, IrValue::make_ubits(7, 0).unwrap());
}

#[test]
fn ext_nary_add_round_trips_via_ffi_wrapped_text() {
    let ir = r#"package test

fn f(a: bits[3] id=1, b: bits[5] id=2, c: bits[9] id=3) -> bits[6] {
  ret r: bits[6] = ext_nary_add(a, b, c, id=4)
}
"#;

    let pkg = {
        let mut p = Parser::new(ir);
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
        !wrapped.contains("ext_nary_add(a, b, c, id=4)"),
        "expected ext_nary_add site to be rewritten as invoke:\n{}",
        wrapped
    );
    assert!(
        wrapped.contains("to_apply=__pir_ext__ext_nary_add"),
        "expected invoke of synthesized ext_nary_add helper:\n{}",
        wrapped
    );

    let reparsed = {
        let mut p = Parser::new(&wrapped);
        p.parse_and_validate_package()
            .expect("parse/validate ffi-wrapped text")
    };
    assert_eq!(
        reparsed.members.len(),
        1,
        "expected wrapped helper to be lifted away after parse"
    );
    let reparsed_fn = reparsed
        .get_fn("f")
        .expect("fn f present after wrapped parse");
    let ext_count: usize = reparsed_fn
        .nodes
        .iter()
        .filter(|n| matches!(n.payload, NodePayload::ExtNaryAdd { .. }))
        .count();
    assert_eq!(ext_count, 1, "expected ext_nary_add to be reconstructed");

    let rewrapped = emit_package_with_extension_mode(&reparsed, ExtensionEmitMode::AsFfiFunction)
        .expect("re-emit ffi-wrapped text");
    assert_eq!(rewrapped, wrapped);
}

#[test]
fn ext_nary_add_simulation_matches_resized_sum_reference() {
    let ir = r#"package test

fn f(a: bits[3] id=1, b: bits[5] id=2, c: bits[9] id=3) -> bits[6] {
  ret r: bits[6] = ext_nary_add(a, b, c, id=4)
}
"#;
    let pkg = {
        let mut p = Parser::new(ir);
        p.parse_and_validate_package().expect("parse/validate")
    };
    let f = pkg.get_fn("f").expect("fn f present");

    for a in 0u64..8u64 {
        for b in 0u64..32u64 {
            for c in [0u64, 1, 0x3f, 0x12a, 0x1ff] {
                let args = [
                    IrValue::make_ubits(3, a).unwrap(),
                    IrValue::make_ubits(5, b).unwrap(),
                    IrValue::make_ubits(9, c).unwrap(),
                ];
                let got = match eval_fn(f, &args) {
                    xlsynth_pir::ir_eval::FnEvalResult::Success(s) => s.value,
                    xlsynth_pir::ir_eval::FnEvalResult::Failure(f) => {
                        panic!("unexpected eval failure: {:?}", f.assertion_failures)
                    }
                };
                let expected = (a + b + (c & 0x3f)) & 0x3f;
                assert_eq!(
                    got,
                    IrValue::make_ubits(6, expected).unwrap(),
                    "mismatch for a={} b={} c={}",
                    a,
                    b,
                    c
                );
            }
        }
    }
}
