// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir::NodePayload;
use xlsynth_pir::ir_parser::Parser;

#[test]
fn ext_prio_encode_accepts_spaced_attributes_and_round_trips() {
    let ir = r#"package test

fn f(arg: bits[8] id=1) -> bits[4] {
  ret r: bits[4] = ext_prio_encode(arg, lsb_prio=false, id=2)
}
"#;
    let package = Parser::new(ir)
        .parse_and_validate_package()
        .expect("parse/validate spaced ext_prio_encode");
    assert!(matches!(
        &package.get_fn("f").unwrap().nodes.last().unwrap().payload,
        NodePayload::ExtPrioEncode {
            lsb_prio: false,
            ..
        }
    ));

    Parser::new(&package.to_string())
        .parse_and_validate_package()
        .expect("reparse emitted ext_prio_encode");
}
