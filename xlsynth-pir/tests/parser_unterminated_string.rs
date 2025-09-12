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

#[test]
fn newline_in_assert_message_rejected() {
    let _ = env_logger::builder().is_test(true).try_init();
    let ir = "package p\n\nfn main(t: token id=1) -> token {\n  literal.2: bits[1] = literal(value=1, id=2)\n  ret assert.3: token = assert(t, literal.2, message=\"foo\nbar\", label=\"L\", id=3)\n}\n";
    let mut p = Parser::new(ir);
    let err = p
        .parse_package()
        .expect_err("expected parse error for newline within quoted string (assert message)");
    let msg = format!("{}", err);
    assert!(
        msg.contains("unterminated quoted string"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn newline_in_assert_label_rejected() {
    let _ = env_logger::builder().is_test(true).try_init();
    let ir = "package p\n\nfn main(t: token id=1) -> token {\n  literal.2: bits[1] = literal(value=1, id=2)\n  ret assert.3: token = assert(t, literal.2, message=\"m\", label=\"L\nX\", id=3)\n}\n";
    let mut p = Parser::new(ir);
    let err = p
        .parse_package()
        .expect_err("expected parse error for newline within quoted string (assert label)");
    let msg = format!("{}", err);
    assert!(
        msg.contains("unterminated quoted string"),
        "unexpected: {}",
        msg
    );
}
