// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir_parser::Parser;

#[test]
fn overly_large_id_is_rejected_in_param() {
    // Use a very large decimal that will not fit in usize on typical platforms.
    let ir = r#"package test

fn f(x: bits[8] id=340282366920938463463374607431768211456) -> bits[8] {
  ret x: bits[8] = param(name=x, id=1)
}
"#;
    let mut p = Parser::new(ir);
    let err = p
        .parse_package()
        .expect_err("expected parse error for oversized id");
    let msg = format!("{}", err);
    assert!(
        msg.contains("expected unsigned integer"),
        "unexpected: {}",
        msg
    );
}
