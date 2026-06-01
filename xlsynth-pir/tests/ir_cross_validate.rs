// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrPackage as XlsIrPackage;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser::Parser as PirParser;
use xlsynth_pir::ir_verify;
use xlsynth_pir::ir_verify_parity::{categorize_pir_error, categorize_xls_error_text};

fn verify_with_pir(ir_text: &str) -> bool {
    let mut p = PirParser::new(ir_text);
    match p.parse_package() {
        Ok(pkg) => ir_verify::verify_package(&pkg).is_ok(),
        Err(_) => false,
    }
}

fn verify_with_xlsynth(ir_text: &str) -> bool {
    match XlsIrPackage::parse_ir(ir_text, None) {
        Ok(pkg) => pkg.verify().is_ok(),
        Err(_) => false,
    }
}

fn ir_text_has_extension_ops(ir_text: &str) -> bool {
    let mut p = PirParser::new(ir_text);
    let Ok(pkg) = p.parse_package() else {
        return false;
    };
    pkg.members.iter().any(|member| match member {
        ir::PackageMember::Function(f) => f.nodes.iter().any(|n| n.payload.is_extension_op()),
        ir::PackageMember::Block { func, .. } => {
            func.nodes.iter().any(|n| n.payload.is_extension_op())
        }
    })
}

fn assert_cross_validates_same(ir_text: &str) {
    if ir_text_has_extension_ops(ir_text) {
        // Early-return rationale: this test checks verification parity between
        // PIR and upstream XLS. Upstream does not understand PIR extension ops
        // (e.g. `ext_carry_out`). Extension behavior is covered by dedicated
        // PIR-level tests.
        return;
    }
    let pir_ok = verify_with_pir(ir_text);
    let xls_ok = verify_with_xlsynth(ir_text);
    assert_eq!(
        pir_ok, xls_ok,
        "verification parity mismatch for IR:\n{}",
        ir_text
    );
}

fn assert_error_category_matches(ir_text: &str) {
    // PIR
    let pir_err = {
        let mut p = PirParser::new(ir_text);
        let pkg = p.parse_package().expect("parse package");
        ir_verify::verify_package(&pkg).expect_err("expect PIR verify to fail")
    };
    let pir_cat = categorize_pir_error(&pir_err);

    // XLS reference
    let xls_err = match XlsIrPackage::parse_ir(ir_text, None) {
        Ok(pkg) => pkg
            .verify()
            .expect_err("expect xls verify to fail")
            .to_string(),
        Err(e) => e.to_string(),
    };
    let xls_cat = categorize_xls_error_text(&xls_err);

    assert_eq!(
        pir_cat, xls_cat,
        "category mismatch: PIR={:?} XLS={}\nIR=\n{}",
        pir_cat, xls_err, ir_text
    );
}

#[test]
fn cross_validate_ok_identity() {
    let ir = r#"package test

fn id(x: bits[8] id=1) -> bits[8] {
  ret x: bits[8] = param(name=x, id=1)
}
"#;
    assert_cross_validates_same(ir);
}

#[test]
fn cross_validate_distinct_empty_block_names() {
    let ir = r#"package test

block i() {
}

block RRRRb() {
}
"#;
    assert_cross_validates_same(ir);
}

#[test]
fn cross_validate_oob_static_bit_slice_fails() {
    let ir = r#"package test

fn f(x: bits[8] id=1) -> bits[2] {
  ret s: bits[2] = bit_slice(x, start=7, width=2, id=2)
}
"#;
    assert_cross_validates_same(ir);
}

#[test]
fn cross_validate_duplicate_text_id_fails() {
    let ir = r#"package test

fn f(x: bits[1] id=1) -> bits[1] {
  a.2: bits[1] = add(x, x, id=2)
  b.2: bits[1] = add(a.2, x, id=2)
  ret b.2: bits[1] = identity(b.2, id=3)
}
"#;
    assert_cross_validates_same(ir);
}

#[test]
fn cross_validate_unknown_callee_fails() {
    let ir = r#"package test

fn f(x: bits[8] id=1) -> bits[8] {
  ret r: bits[8] = invoke(x, to_apply=missing, id=2)
}
"#;
    assert_cross_validates_same(ir);
    assert_error_category_matches(ir);
}

#[test]
fn cross_validate_invoke_type_mismatch_category_matches() {
    let ir = r#"
package test

fn callee(x: bits[1] id=1) -> (bits[1], bits[1]) {
  ret tuple.3: (bits[1], bits[1]) = tuple(x, x, id=3)
}

fn foo(x: bits[1] id=1) -> bits[1] {
  invoke.2: bits[1] = invoke(x, to_apply=callee, id=2)
  ret identity.3: bits[1] = identity(invoke.2, id=3)
}
"#;
    assert_error_category_matches(ir);
}

#[test]
fn cross_validate_cover_has_empty_tuple_result() {
    let ir = r#"package test

fn f(x: bits[1] id=1) -> () {
  ret cover.2: () = cover(x, label="covered", id=2)
}
"#;
    assert!(verify_with_pir(ir));
    assert!(verify_with_xlsynth(ir));
}

#[test]
fn cross_validate_mixed_width_add_fails() {
    let ir = r#"package test

fn f(x: bits[8] id=1, y: bits[7] id=2) -> bits[8] {
  ret add.3: bits[8] = add(x, y, id=3)
}
"#;
    assert_cross_validates_same(ir);
    assert!(!verify_with_pir(ir));
}

#[test]
fn cross_validate_dynamic_bit_slice_wider_than_input_fails() {
    let ir = r#"package test

fn f(x: bits[8] id=1, start: bits[3] id=2) -> bits[9] {
  ret dynamic_bit_slice.3: bits[9] = dynamic_bit_slice(x, start, width=9, id=3)
}
"#;
    assert_cross_validates_same(ir);
    assert!(!verify_with_pir(ir));
}

#[test]
fn cross_validate_sel_default_rules_fail_invalid_forms() {
    let no_default = r#"package test

fn f(s: bits[2] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4) -> bits[8] {
  ret sel.5: bits[8] = sel(s, cases=[a, b, c], id=5)
}
"#;
    assert_cross_validates_same(no_default);
    assert!(!verify_with_pir(no_default));

    let useless_default = r#"package test

fn f(s: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3, d: bits[8] id=4) -> bits[8] {
  ret sel.5: bits[8] = sel(s, cases=[a, b], default=d, id=5)
}
"#;
    assert_cross_validates_same(useless_default);
    assert!(!verify_with_pir(useless_default));
}

#[test]
fn cross_validate_invoke_argument_type_fails() {
    let ir = r#"package test

fn callee(x: bits[8] id=1) -> bits[8] {
  ret identity.2: bits[8] = identity(x, id=2)
}

fn f(y: bits[7] id=3) -> bits[8] {
  ret invoke.4: bits[8] = invoke(y, to_apply=callee, id=4)
}
"#;
    assert_cross_validates_same(ir);
    assert!(!verify_with_pir(ir));
}

#[test]
fn cross_validate_counted_for_induction_width_fails() {
    let ir = r#"package test

fn body(i: bits[1] id=1, carry: bits[8] id=2) -> bits[8] {
  ret identity.3: bits[8] = identity(carry, id=3)
}

fn f(x: bits[8] id=4) -> bits[8] {
  ret counted_for.5: bits[8] = counted_for(x, trip_count=4, stride=1, body=body, id=5)
}
"#;
    assert_cross_validates_same(ir);
    assert!(!verify_with_pir(ir));
}
