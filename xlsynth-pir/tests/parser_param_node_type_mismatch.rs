// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir_parser::Parser;

#[test]
fn param_node_type_mismatch_with_header_rejected() {
    let ir = r#"package test

fn id(x: bits[8] id=1) -> bits[8] {
  ret x: bits[4] = param(name=x, id=1)
}
"#;
    let mut p = Parser::new(ir);
    let err = p
        .parse_package()
        .expect_err("expected parse error for param type mismatch");
    let msg = format!("{}", err);
    assert!(
        msg.contains("type mismatch")
            && msg.contains("header bits[8]")
            && msg.contains("node bits[4]"),
        "unexpected error: {}",
        msg
    );
}
