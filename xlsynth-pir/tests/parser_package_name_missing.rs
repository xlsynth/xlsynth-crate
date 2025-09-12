// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir_parser::Parser;

#[test]
fn package_without_name_is_rejected() {
    let ir = "package ";
    let mut p = Parser::new(ir);
    let err = p
        .parse_package()
        .expect_err("expected parse error for missing package name");
    let msg = format!("{}", err);
    assert!(
        msg.contains("expected identifier") || msg.contains("expected identifier, got EOF"),
        "unexpected error: {}",
        msg
    );
}

#[test]
fn package_keyword_requires_delimiter() {
    // Without whitespace delimiter, this should NOT be accepted as 'package'.
    let ir = "packagette test\n\nfn f() -> bits[1] {\n  ret literal.1: bits[1] = literal(value=0, id=1)\n}\n";
    let mut p = Parser::new(ir);
    let err = p
        .parse_package()
        .expect_err("expected parse error for missing 'package' keyword");
    let msg = format!("{}", err);
    assert!(
        msg.contains("expected keyword \"package\"") || msg.contains("expected \"package\""),
        "unexpected error: {}",
        msg
    );
}

#[test]
fn non_ascii_identifier_is_rejected() {
    // Non-ASCII letter in package name should be rejected.
    let ir =
        "package Қ\n\nfn f() -> bits[1] {\n  ret literal.1: bits[1] = literal(value=0, id=1)\n}\n";
    let mut p = Parser::new(ir);
    let err = p
        .parse_package()
        .expect_err("expected parse error for non-ascii identifier");
    let msg = format!("{}", err);
    assert!(
        msg.contains("expected identifier"),
        "unexpected error: {}",
        msg
    );
}
