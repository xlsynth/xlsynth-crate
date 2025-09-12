// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir_parser::Parser;

#[test]
fn param_node_with_unknown_name_rejected() {
    let ir = r#"package test

fn id(x: bits[8] id=1) -> bits[8] {
  r: bits[8] = param(name=p, id=1)
}
"#;
    let mut p = Parser::new(ir);
    let err = p
        .parse_package()
        .expect_err("expected parse error for unknown param name");
    assert_eq!(
        format!("{}", err),
        "ParseError: param name/id mismatch: name=p id=1"
    );
}
