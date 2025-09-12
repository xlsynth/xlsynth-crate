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
