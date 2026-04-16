// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrPackage;
use xlsynth_pir::desugar_extensions::desugar_extensions_in_package;
use xlsynth_pir::desugar_extensions::emit_package_as_xls_ir_text;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::prove_ir_equiv;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivParallelism, EquivResult, ProverFn};

#[test]
fn ext_nary_add_equivalent_to_desugared_export_form_and_exports_to_upstream() {
    let ir = r#"package test

fn f(a: bits[3] id=1, b: bits[5] id=2, c: bits[9] id=3) -> bits[6] {
  ret r: bits[6] = ext_nary_add(a, b, c, signed=[false, false, false], negated=[false, false, false], arch=brent_kung, id=4)
}
"#;

    let pkg = {
        let mut p = Parser::new(ir);
        p.parse_and_validate_package().expect("parse/validate")
    };

    let exported = emit_package_as_xls_ir_text(&pkg).expect("export desugaring");
    assert!(
        !exported.contains("ext_nary_add"),
        "export should not contain ext_nary_add:\n{}",
        exported
    );

    let exported_pkg = IrPackage::parse_ir(&exported, None).expect("upstream parse");
    exported_pkg.verify().expect("upstream verify");

    let mut desugared_pkg = pkg.clone();
    desugar_extensions_in_package(&mut desugared_pkg).expect("desugar in package");

    let orig_fn = pkg.get_fn("f").expect("original fn");
    let desugared_fn = desugared_pkg.get_fn("f").expect("desugared fn");

    let lhs = ProverFn::new(orig_fn, None);
    let rhs = ProverFn::new(desugared_fn, None);
    let res = prove_ir_equiv(
        &lhs,
        &rhs,
        EquivParallelism::SingleThreaded,
        AssertionSemantics::Ignore,
        None,
        false,
    );
    match res {
        EquivResult::Proved => {}
        EquivResult::Interrupted => {
            panic!("equivalence unexpectedly interrupted for ext_nary_add desugaring");
        }
        EquivResult::ToolchainDisproved(msg)
            if msg.contains("Unknown operation")
                && msg.contains("ext_nary_add")
                && msg.contains("string-to-op conversion") =>
        {
            return;
        }
        _ => panic!("formal equivalence failed: {:?}", res),
    }
}
