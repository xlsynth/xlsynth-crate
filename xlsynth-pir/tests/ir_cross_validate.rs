// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrPackage as XlsIrPackage;
use xlsynth_pir::ir_parser::Parser as PirParser;
use xlsynth_pir::ir_validate::{self, ValidationError};

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

#[derive(Debug, PartialEq, Eq)]
enum ErrorCategory {
    UnknownCallee,
    NodeTypeMismatch,
    OperandOutOfBounds,
    OperandUsesUndefined,
    DuplicateTextId,
    ReturnTypeMismatch,
    DuplicateParamName,
    MissingParamNode,
    ExtraParamNode,
    NodeNameOpMismatch,
    NodeNameIdSuffixMismatch,
    Other,
}

fn categorize_pir_error(err: &ValidationError) -> ErrorCategory {
    use ErrorCategory::*;
    match err {
        ValidationError::UnknownCallee { .. } => UnknownCallee,
        ValidationError::NodeTypeMismatch { .. } => NodeTypeMismatch,
        ValidationError::OperandOutOfBounds { .. } => OperandOutOfBounds,
        ValidationError::OperandUsesUndefined { .. } => OperandUsesUndefined,
        ValidationError::DuplicateTextId { .. } => DuplicateTextId,
        ValidationError::ReturnTypeMismatch { .. } => ReturnTypeMismatch,
        ValidationError::DuplicateParamName { .. } => DuplicateParamName,
        ValidationError::MissingParamNode { .. } => MissingParamNode,
        ValidationError::ExtraParamNode { .. } => ExtraParamNode,
        ValidationError::NodeNameOpMismatch { .. } => NodeNameOpMismatch,
        ValidationError::NodeNameIdSuffixMismatch { .. } => NodeNameIdSuffixMismatch,
        _ => Other,
    }
}

fn categorize_xls_error_text(s: &str) -> ErrorCategory {
    use ErrorCategory::*;
    let lower = s.to_lowercase();
    if lower.contains("unknown callee")
        || lower.contains("unknown function")
        || lower.contains("cannot find function")
        || lower.contains("does not have a function with name")
    {
        return UnknownCallee;
    }
    if lower.contains("type mismatch")
        || (lower.contains("type") && lower.contains("mismatch"))
        || lower.contains("does not match expected type")
    {
        return NodeTypeMismatch; // coarse bucket for our purposes
    }
    if lower.contains("out of bounds") {
        return OperandOutOfBounds;
    }
    if lower.contains("before definition") || lower.contains("uses operand") {
        return OperandUsesUndefined;
    }
    if lower.contains("duplicate") && lower.contains("id") {
        return DuplicateTextId;
    }
    if lower.contains("return type") && lower.contains("mismatch") {
        return ReturnTypeMismatch;
    }
    if lower.contains("duplicate param") || lower.contains("duplicate parameter") {
        return DuplicateParamName;
    }
    if lower.contains("missing getparam") || lower.contains("missing param") {
        return MissingParamNode;
    }
    if lower.contains("not declared in signature") || lower.contains("extra param") {
        return ExtraParamNode;
    }
    Other
}

fn assert_error_category_matches(ir_text: &str) {
    // PIR
    let pir_err = {
        let mut p = PirParser::new(ir_text);
        let pkg = p.parse_package().expect("parse package");
        ir_validate::validate_package(&pkg).expect_err("expect PIR verify to fail")
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
