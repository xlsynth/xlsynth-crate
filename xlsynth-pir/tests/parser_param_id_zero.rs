// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir_parser::Parser;

#[test]
fn parse_param_header_id_zero_rejected() {
    let ir = r#"package test

fn f(x: bits[8] id=0) -> bits[8] {
  ret x: bits[8] = param(name=x, id=1)
}
"#;
    let mut p = Parser::new(ir);
    let err = p
        .parse_package()
        .expect_err("expected parse error for id=0 in header");
    let msg = format!("{}", err);
    assert!(
        msg.contains("id must be greater than zero") || msg.contains("greater than zero"),
        "unexpected error: {}",
        msg
    );
}

#[test]
fn parse_param_node_id_zero_rejected() {
    let ir = r#"package test

fn f(x: bits[8] id=1) -> bits[8] {
  ret x: bits[8] = param(name=x, id=0)
}
"#;
    let mut p = Parser::new(ir);
    let err = p
        .parse_package()
        .expect_err("expected parse error for id=0 in param node");
    let msg = format!("{}", err);
    assert!(
        msg.contains("id must be greater than zero") || msg.contains("greater than zero"),
        "unexpected error: {}",
        msg
    );
}
