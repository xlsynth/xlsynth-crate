// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrPackage as XlsIrPackage;
use xlsynth_pir::ir_parser::Parser as PirParser;
use xlsynth_pir::ir_validate;

fn verify_with_pir(ir_text: &str) -> bool {
    let mut p = PirParser::new(ir_text);
    match p.parse_package() {
        Ok(pkg) => ir_validate::validate_package(&pkg).is_ok(),
        Err(_) => false,
    }
}

fn verify_with_xlsynth(ir_text: &str) -> bool {
    match XlsIrPackage::parse_ir(ir_text, None) {
        Ok(pkg) => pkg.verify().is_ok(),
        Err(_) => false,
    }
}

fn assert_cross_validates_same(ir_text: &str) {
    let pir_ok = verify_with_pir(ir_text);
    let xls_ok = verify_with_xlsynth(ir_text);
    assert_eq!(
        pir_ok, xls_ok,
        "verification parity mismatch for IR:\n{}",
        ir_text
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
}
