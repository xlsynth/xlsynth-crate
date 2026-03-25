// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrValue;
use xlsynth_pir::desugar_extensions::{ExtensionEmitMode, emit_package_with_extension_mode};
use xlsynth_pir::ir::NodePayload;
use xlsynth_pir::ir_eval::eval_fn;
use xlsynth_pir::ir_parser::Parser;

#[test]
fn ext_carry_out_round_trips_via_text() {
    let ir = r#"package test

fn f(lhs: bits[8] id=1, rhs: bits[8] id=2, c_in: bits[1] id=3) -> bits[1] {
  ret r: bits[1] = ext_carry_out(lhs, rhs, c_in, id=4)
}
"#;

    let pkg = {
        let mut p = Parser::new(ir);
        p.parse_and_validate_package().expect("parse/validate")
    };
    let f = pkg.get_fn("f").expect("fn f present");
    let ext_count: usize = f
        .nodes
        .iter()
        .filter(|n| matches!(n.payload, NodePayload::ExtCarryOut { .. }))
        .count();
    assert_eq!(ext_count, 1);

    let text = pkg.to_string();
    assert!(
        text.contains("ext_carry_out"),
        "expected ext_carry_out to appear in emitted text:\n{}",
        text
    );

    let pkg2 = {
        let mut p = Parser::new(&text);
        p.parse_and_validate_package().expect("re-parse/validate")
    };
    let f2 = pkg2.get_fn("f").expect("fn f present in reparsed");
    let ext_count2: usize = f2
        .nodes
        .iter()
        .filter(|n| matches!(n.payload, NodePayload::ExtCarryOut { .. }))
        .count();
    assert_eq!(ext_count2, 1);
}

#[test]
fn ext_carry_out_round_trips_via_ffi_wrapped_text() {
    let ir = r#"package test

fn f(lhs: bits[8] id=1, rhs: bits[8] id=2, c_in: bits[1] id=3) -> bits[1] {
  ret r: bits[1] = ext_carry_out(lhs, rhs, c_in, id=4)
}
"#;

    let pkg = {
        let mut p = Parser::new(ir);
        p.parse_and_validate_package().expect("parse/validate")
    };
    let wrapped = emit_package_with_extension_mode(&pkg, ExtensionEmitMode::AsFfiFunction).unwrap();
    assert!(
        wrapped.contains("#[ffi_proto(\"\"\""),
        "expected ffi_proto attribute in wrapped text:\n{}",
        wrapped
    );
    assert!(
        wrapped.contains("__pir_ext__ext_carry_out__w8"),
        "expected deterministic helper name in wrapped text:\n{}",
        wrapped
    );
    assert!(
        wrapped.contains("invoke(lhs, rhs, c_in, to_apply=__pir_ext__ext_carry_out__w8, id=4)"),
        "expected invoke of the ffi helper in wrapped text:\n{}",
        wrapped
    );

    let wrapped_pkg = {
        let mut p = Parser::new(&wrapped);
        p.parse_and_validate_package()
            .expect("parse/validate wrapped text")
    };
    let wrapped_f = wrapped_pkg.get_fn("f").expect("fn f present");
    assert_eq!(wrapped_pkg.members.len(), 1);
    assert!(wrapped_pkg.get_fn("__pir_ext__ext_carry_out__w8").is_none());
    assert!(
        wrapped_f
            .nodes
            .iter()
            .any(|n| matches!(n.payload, NodePayload::ExtCarryOut { .. }))
    );
    let rewrapped =
        emit_package_with_extension_mode(&wrapped_pkg, ExtensionEmitMode::AsFfiFunction).unwrap();
    assert_eq!(rewrapped, wrapped);

    let xls_pkg = xlsynth::IrPackage::parse_ir(&wrapped, None).expect("xls parse wrapped");
    let xls_text = xls_pkg.to_string();
    assert!(
        xls_text.contains("ffi_proto"),
        "expected ffi_proto to survive XLS round-trip:\n{}",
        xls_text
    );

    let reparsed = {
        let mut p = Parser::new(&xls_text);
        p.parse_and_validate_package()
            .expect("re-parse/validate xls formatted wrapped text")
    };
    assert_eq!(reparsed.members.len(), 1);
    assert!(reparsed.get_fn("__pir_ext__ext_carry_out__w8").is_none());
    let rewrapped_xls =
        emit_package_with_extension_mode(&reparsed, ExtensionEmitMode::AsFfiFunction).unwrap();
    assert_eq!(rewrapped_xls, xls_text);
}

#[test]
fn ext_carry_out_simulation_matches_software_reference_for_small_widths() {
    for w in 1u64..=8u64 {
        let ir = format!(
            "package test\n\nfn f(lhs: bits[{w}] id=1, rhs: bits[{w}] id=2, c_in: bits[1] id=3) -> bits[1] {{\n  ret r: bits[1] = ext_carry_out(lhs, rhs, c_in, id=4)\n}}\n"
        );
        let pkg = {
            let mut p = Parser::new(&ir);
            p.parse_and_validate_package().expect("parse/validate")
        };
        let f = pkg.get_fn("f").expect("fn f present");

        let mask: u64 = (1u64 << w) - 1;
        for lhs in 0u64..=mask {
            for rhs in 0u64..=mask {
                for c_in in [0u64, 1u64] {
                    let args = [
                        IrValue::make_ubits(w as usize, lhs).unwrap(),
                        IrValue::make_ubits(w as usize, rhs).unwrap(),
                        IrValue::make_ubits(1, c_in).unwrap(),
                    ];
                    let got = match eval_fn(f, &args) {
                        xlsynth_pir::ir_eval::FnEvalResult::Success(s) => {
                            s.value.to_bool().unwrap()
                        }
                        xlsynth_pir::ir_eval::FnEvalResult::Failure(f) => {
                            panic!("unexpected eval failure: {:?}", f.assertion_failures)
                        }
                    };
                    let sum = (lhs as u128) + (rhs as u128) + (c_in as u128);
                    let expected = ((sum >> w) & 1) != 0;
                    assert_eq!(
                        got, expected,
                        "mismatch at w={} lhs={} rhs={} c_in={}",
                        w, lhs, rhs, c_in
                    );
                }
            }
        }
    }
}
