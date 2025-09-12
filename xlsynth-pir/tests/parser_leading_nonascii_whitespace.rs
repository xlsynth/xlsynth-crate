// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir_parser::Parser;

#[test]
fn non_ascii_whitespace_before_package_is_rejected() {
    // Use a non-breaking space (U+00A0) before the package keyword.
    let ir = "\u{00A0}package test\n\nfn f() -> bits[1] {\n  ret literal.1: bits[1] = literal(value=0, id=1)\n}\n";
    let mut p = Parser::new(ir);
    let err = p
        .parse_package()
        .expect_err("expected parse error for leading non-ascii whitespace");
    let msg = format!("{}", err);
    // We should fail early expecting 'package' at the start, not skip the NBSP.
    assert!(
        msg.contains("expected") || msg.contains("unexpected"),
        "unexpected error: {}",
        msg
    );
}
