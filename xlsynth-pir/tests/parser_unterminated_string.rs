// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir_parser::Parser;

#[test]
fn unterminated_quoted_string_in_file_number_rejected() {
    let ir = "package test\n\nfile_number 0 \"foo";
    let mut p = Parser::new(ir);
    let err = p
        .parse_package()
        .expect_err("expected parse error for unterminated quoted string");
    let msg = format!("{}", err);
    assert!(
        msg.contains("unterminated quoted string"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn newline_in_string_rejected() {
    let ir = "package test\n\nfile_number 0 \"foo\nbar\"\n";
    let mut p = Parser::new(ir);
    let err = p
        .parse_package()
        .expect_err("expected parse error for newline within quoted string");
    let msg = format!("{}", err);
    assert!(
        msg.contains("unterminated quoted string"),
        "unexpected: {}",
        msg
    );
}
