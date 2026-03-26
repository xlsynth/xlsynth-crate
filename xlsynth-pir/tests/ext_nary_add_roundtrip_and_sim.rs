// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrValue;
use xlsynth_pir::desugar_extensions::{ExtensionEmitMode, emit_package_with_extension_mode};
use xlsynth_pir::ir::{ExtNaryAddArchitecture, ExtNaryAddTerm, Fn, NodePayload};
use xlsynth_pir::ir_eval::eval_fn;
use xlsynth_pir::ir_parser::Parser;

fn get_ext_nary_add_arch(f: &Fn) -> Option<ExtNaryAddArchitecture> {
    f.nodes
        .iter()
        .find_map(|n| match n.payload {
            NodePayload::ExtNaryAdd { arch, .. } => Some(arch),
            _ => None,
        })
        .expect("expected ext_nary_add node")
}

fn get_ext_nary_add_terms(f: &Fn) -> Vec<ExtNaryAddTerm> {
    f.nodes
        .iter()
        .find_map(|n| match &n.payload {
            NodePayload::ExtNaryAdd { terms, .. } => Some(terms.clone()),
            _ => None,
        })
        .expect("expected ext_nary_add node")
}

#[test]
fn ext_nary_add_round_trips_via_text() {
    let ir = r#"package test

fn f(a: bits[3] id=1, b: bits[5] id=2, c: bits[9] id=3) -> bits[6] {
  ret r: bits[6] = ext_nary_add(a, b, c, signed=[false, false, false], negated=[false, false, false], arch=kogge_stone, id=4)
}
"#;

    let pkg = {
        let mut p = Parser::new(ir);
        p.parse_and_validate_package().expect("parse/validate")
    };
    let f = pkg.get_fn("f").expect("fn f present");
    assert_eq!(
        get_ext_nary_add_arch(f),
        Some(ExtNaryAddArchitecture::KoggeStone)
    );
    let ext_count: usize = f
        .nodes
        .iter()
        .filter(|n| matches!(n.payload, NodePayload::ExtNaryAdd { .. }))
        .count();
    assert_eq!(ext_count, 1);

    let text = pkg.to_string();
    assert!(
        text.contains(
            "ext_nary_add(a, b, c, signed=[false, false, false], negated=[false, false, false], arch=kogge_stone, id=4)"
        ),
        "expected ext_nary_add to appear in emitted text:\n{}",
        text
    );

    let pkg2 = {
        let mut p = Parser::new(&text);
        p.parse_and_validate_package().expect("re-parse/validate")
    };
    let f2 = pkg2.get_fn("f").expect("fn f present in reparsed");
    assert_eq!(
        get_ext_nary_add_arch(f2),
        Some(ExtNaryAddArchitecture::KoggeStone)
    );
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
  ret r: bits[7] = ext_nary_add(signed=[], negated=[], arch=brent_kung, id=1)
}
"#;
    let pkg = {
        let mut p = Parser::new(ir);
        p.parse_and_validate_package().expect("parse/validate")
    };
    assert_eq!(
        get_ext_nary_add_arch(pkg.get_fn("f").expect("fn f present")),
        Some(ExtNaryAddArchitecture::BrentKung)
    );
    let text = pkg.to_string();
    assert!(
        text.contains("ext_nary_add(signed=[], negated=[], arch=brent_kung, id=1)"),
        "expected ext_nary_add with explicit arch in emitted text:\n{}",
        text
    );
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
fn ext_nary_add_arch_is_optional_in_text() {
    let ir = r#"package test

fn f(a: bits[3] id=1, b: bits[5] id=2) -> bits[6] {
  ret r: bits[6] = ext_nary_add(a, b, signed=[false, false], negated=[false, false], id=3)
}
"#;
    let pkg = {
        let mut p = Parser::new(ir);
        p.parse_and_validate_package()
            .expect("missing arch should use lowering default")
    };
    assert!(
        pkg.to_string()
            .contains("ext_nary_add(a, b, signed=[false, false], negated=[false, false], id=3)"),
        "expected emitted text to omit arch when not specified:\n{}",
        pkg.to_string()
    );
    assert_eq!(
        get_ext_nary_add_arch(pkg.get_fn("f").expect("fn f present")),
        None
    );
}

#[test]
fn ext_nary_add_requires_signed_and_negated_in_text() {
    let missing_signed = r#"package test

fn f(a: bits[3] id=1, b: bits[5] id=2) -> bits[6] {
  ret r: bits[6] = ext_nary_add(a, b, negated=[false, false], id=3)
}
"#;
    let err = {
        let mut p = Parser::new(missing_signed);
        p.parse_and_validate_package()
            .expect_err("missing signed should be rejected")
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("missing signed attribute for ext_nary_add"),
        "unexpected parse error: {}",
        msg
    );

    let missing_negated = r#"package test

fn f(a: bits[3] id=1, b: bits[5] id=2) -> bits[6] {
  ret r: bits[6] = ext_nary_add(a, b, signed=[false, false], id=3)
}
"#;
    let err = {
        let mut p = Parser::new(missing_negated);
        p.parse_and_validate_package()
            .expect_err("missing negated should be rejected")
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("missing negated attribute for ext_nary_add"),
        "unexpected parse error: {}",
        msg
    );
}

#[test]
fn ext_nary_add_term_attributes_round_trip_via_text() {
    let ir = r#"package test

fn f(a: bits[3] id=1, b: bits[5] id=2, c: bits[4] id=3) -> bits[6] {
  ret r: bits[6] = ext_nary_add(a, b, c, signed=[false, true, false], negated=[false, false, true], arch=kogge_stone, id=4)
}
"#;

    let pkg = {
        let mut p = Parser::new(ir);
        p.parse_and_validate_package().expect("parse/validate")
    };
    let f = pkg.get_fn("f").expect("fn f present");
    let terms = get_ext_nary_add_terms(f);
    assert_eq!(terms.len(), 3);
    assert!(!terms[0].signed && !terms[0].negated);
    assert!(terms[1].signed);
    assert!(!terms[1].negated);
    assert!(!terms[2].signed);
    assert!(terms[2].negated);

    let text = pkg.to_string();
    assert!(
        text.contains(
            "ext_nary_add(a, b, c, signed=[false, true, false], negated=[false, false, true], arch=kogge_stone, id=4)"
        ),
        "expected attributed ext_nary_add to appear in emitted text:\n{}",
        text
    );

    let reparsed = {
        let mut p = Parser::new(&text);
        p.parse_and_validate_package().expect("re-parse/validate")
    };
    let reparsed_terms = get_ext_nary_add_terms(reparsed.get_fn("f").expect("fn f present"));
    assert_eq!(reparsed_terms, terms);
}

#[test]
fn ext_nary_add_round_trips_via_ffi_wrapped_text() {
    let ir = r#"package test

fn f(a: bits[3] id=1, b: bits[5] id=2, c: bits[9] id=3) -> bits[6] {
  ret r: bits[6] = ext_nary_add(a, b, c, signed=[false, false, false], negated=[false, false, false], arch=kogge_stone, id=4)
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
        !wrapped.contains(
            "ext_nary_add(a, b, c, signed=[false, false, false], negated=[false, false, false], arch=kogge_stone, id=4)"
        ),
        "expected ext_nary_add site to be rewritten as invoke:\n{}",
        wrapped
    );
    assert!(
        wrapped.contains("to_apply=__pir_ext__ext_nary_add__outw6__ops3_5_9__archkogge_stone"),
        "expected invoke of synthesized ext_nary_add helper:\n{}",
        wrapped
    );
    assert!(
        wrapped.contains("arch=kogge_stone"),
        "expected wrapped metadata to preserve arch:\n{}",
        wrapped
    );
    assert!(
        wrapped.contains("operand_signed=false,false,false"),
        "expected wrapped metadata to preserve explicit default term signedness:\n{}",
        wrapped
    );
    assert!(
        wrapped.contains("operand_negated=false,false,false"),
        "expected wrapped metadata to preserve explicit default term negation:\n{}",
        wrapped
    );

    let wrapped_pkg = {
        let mut p = Parser::new(&wrapped);
        p.parse_and_validate_package()
            .expect("parse/validate wrapped text")
    };
    assert_eq!(
        wrapped_pkg.members.len(),
        1,
        "expected wrapped helper to be lifted away after parse"
    );
    let wrapped_f = wrapped_pkg.get_fn("f").expect("fn f present after reparse");
    assert_eq!(
        get_ext_nary_add_arch(wrapped_f),
        Some(ExtNaryAddArchitecture::KoggeStone)
    );
    let ext_count: usize = wrapped_f
        .nodes
        .iter()
        .filter(|n| matches!(n.payload, NodePayload::ExtNaryAdd { .. }))
        .count();
    assert_eq!(ext_count, 1, "expected ext_nary_add to be reconstructed");

    let wrapped_again =
        emit_package_with_extension_mode(&wrapped_pkg, ExtensionEmitMode::AsFfiFunction)
            .expect("re-emit ffi-wrapped text");
    assert_eq!(wrapped_again, wrapped);
}

#[test]
fn ext_nary_add_without_arch_round_trips_via_ffi_wrapped_text() {
    let ir = r#"package test

fn f(a: bits[3] id=1, b: bits[5] id=2, c: bits[9] id=3) -> bits[6] {
  ret r: bits[6] = ext_nary_add(a, b, c, signed=[false, false, false], negated=[false, false, false], id=4)
}
"#;

    let pkg = {
        let mut p = Parser::new(ir);
        p.parse_and_validate_package().expect("parse/validate")
    };
    let wrapped = emit_package_with_extension_mode(&pkg, ExtensionEmitMode::AsFfiFunction)
        .expect("emit ffi-wrapped text");
    assert!(
        wrapped.contains("to_apply=__pir_ext__ext_nary_add__outw6__ops3_5_9"),
        "expected invoke of synthesized ext_nary_add helper without arch suffix:\n{}",
        wrapped
    );
    assert!(
        wrapped.contains("operand_signed=false,false,false"),
        "expected wrapped metadata to preserve explicit default term signedness:\n{}",
        wrapped
    );
    assert!(
        wrapped.contains("operand_negated=false,false,false"),
        "expected wrapped metadata to preserve explicit default term negation:\n{}",
        wrapped
    );
    assert!(
        !wrapped.contains("arch="),
        "expected wrapped metadata to omit arch when not specified:\n{}",
        wrapped
    );

    let wrapped_pkg = {
        let mut p = Parser::new(&wrapped);
        p.parse_and_validate_package()
            .expect("parse/validate wrapped text")
    };
    assert_eq!(
        get_ext_nary_add_arch(wrapped_pkg.get_fn("f").expect("fn f present after reparse")),
        None
    );
}

#[test]
fn ext_nary_add_term_attributes_round_trip_via_ffi_wrapped_text() {
    let ir = r#"package test

fn f(a: bits[3] id=1, b: bits[5] id=2, c: bits[4] id=3) -> bits[6] {
  ret r: bits[6] = ext_nary_add(a, b, c, signed=[false, true, false], negated=[false, false, true], arch=kogge_stone, id=4)
}
"#;

    let pkg = {
        let mut p = Parser::new(ir);
        p.parse_and_validate_package().expect("parse/validate")
    };
    let wrapped = emit_package_with_extension_mode(&pkg, ExtensionEmitMode::AsFfiFunction)
        .expect("emit ffi-wrapped text");
    assert!(
        wrapped.contains(
            "__pir_ext__ext_nary_add__outw6__ops3_5_4__sgn0_1_0__neg0_0_1__archkogge_stone"
        ),
        "expected helper name to encode attributed term metadata:\n{}",
        wrapped
    );
    assert!(
        wrapped.contains("operand_signed=false,true,false"),
        "expected wrapped metadata to preserve term signedness:\n{}",
        wrapped
    );
    assert!(
        wrapped.contains("operand_negated=false,false,true"),
        "expected wrapped metadata to preserve term negation:\n{}",
        wrapped
    );

    let wrapped_pkg = {
        let mut p = Parser::new(&wrapped);
        p.parse_and_validate_package()
            .expect("parse/validate wrapped text")
    };
    let wrapped_terms = get_ext_nary_add_terms(wrapped_pkg.get_fn("f").expect("fn f present"));
    let orig_terms = get_ext_nary_add_terms(pkg.get_fn("f").expect("fn f present"));
    assert_eq!(wrapped_terms, orig_terms);
    assert_eq!(
        emit_package_with_extension_mode(&wrapped_pkg, ExtensionEmitMode::AsFfiFunction)
            .expect("re-emit ffi-wrapped text"),
        wrapped
    );
}

#[test]
fn ext_nary_add_simulation_matches_resized_sum_reference() {
    let ir = r#"package test

fn f(a: bits[3] id=1, b: bits[5] id=2, c: bits[9] id=3) -> bits[6] {
  ret r: bits[6] = ext_nary_add(a, b, c, signed=[false, false, false], negated=[false, false, false], arch=brent_kung, id=4)
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

#[test]
fn ext_nary_add_simulation_matches_signed_and_negated_reference() {
    let ir = r#"package test

fn f(a: bits[3] id=1, b: bits[5] id=2, c: bits[4] id=3) -> bits[6] {
  ret r: bits[6] = ext_nary_add(a, b, c, signed=[false, true, false], negated=[false, false, true], id=4)
}
"#;
    let pkg = {
        let mut p = Parser::new(ir);
        p.parse_and_validate_package().expect("parse/validate")
    };
    let f = pkg.get_fn("f").expect("fn f present");

    for a in 0u64..8 {
        for b in 0u64..32 {
            for c in 0u64..16 {
                let args = [
                    IrValue::make_ubits(3, a).unwrap(),
                    IrValue::make_ubits(5, b).unwrap(),
                    IrValue::make_ubits(4, c).unwrap(),
                ];
                let got = match eval_fn(f, &args) {
                    xlsynth_pir::ir_eval::FnEvalResult::Success(s) => s.value,
                    xlsynth_pir::ir_eval::FnEvalResult::Failure(f) => {
                        panic!("unexpected eval failure: {:?}", f.assertion_failures)
                    }
                };
                let b_signed = if (b & 0x10) != 0 {
                    (b as i64) - 32
                } else {
                    b as i64
                };
                let expected = ((a as i64) + b_signed - (c as i64)).rem_euclid(64) as u64;
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
