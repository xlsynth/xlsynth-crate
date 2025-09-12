// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir_parser::Parser;

#[test]
fn param_name_id_mismatch_rejected() {
    let ir = r#"package test

fn id(x: bits[4] id=1) -> bits[4] {
  ret x: bits[4] = param(name=x, id=2)
}
"#;
    let mut p = Parser::new(ir);
    let err = p
        .parse_package()
        .expect_err("expected parse error for param name/id mismatch");
    let msg = format!("{}", err);
    assert!(
        msg.contains("name/id mismatch"),
        "unexpected error: {}",
        msg
    );
}
