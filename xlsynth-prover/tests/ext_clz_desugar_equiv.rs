// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrPackage;
use xlsynth_pir::desugar_extensions::desugar_extensions_in_package;
use xlsynth_pir::desugar_extensions::emit_package_as_xls_ir_text;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::prove_ir_equiv;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivParallelism, EquivResult, ProverFn};

#[test]
fn ext_clz_equivalent_to_desugared_export_form_and_exports_to_upstream() {
    for w in 1u64..=16u64 {
        let out_w = xlsynth_pir::math::ceil_log2((w as usize).saturating_add(1));
        let ir = format!(
            "package test\n\nfn f(arg: bits[{w}] id=1) -> bits[{out_w}] {{\n  ret r: bits[{out_w}] = ext_clz(arg, id=2)\n}}\n"
        );
        let pkg = {
            let mut p = Parser::new(&ir);
            p.parse_and_validate_package().expect("parse/validate")
        };

        let exported = emit_package_as_xls_ir_text(&pkg).expect("export desugaring");
        assert!(
            !exported.contains("ext_clz"),
            "export should not contain ext_clz:\n{}",
            exported
        );
        let exported_pkg = IrPackage::parse_ir(&exported, None).expect("upstream parse");
        exported_pkg.verify().expect("upstream verify");

        let mut desugared_pkg = pkg.clone();
        desugar_extensions_in_package(&mut desugared_pkg).expect("desugar in package");

        let f_ext = pkg.get_fn("f").expect("fn f present");
        let f_desugared = desugared_pkg
            .get_fn("f")
            .expect("fn f present in desugared");

        let lhs = ProverFn::new(f_ext, None);
        let rhs = ProverFn::new(f_desugared, None);
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
            EquivResult::ToolchainDisproved(msg)
                if msg.contains("Unknown operation")
                    && msg.contains("ext_clz")
                    && msg.contains("string-to-op conversion") =>
            {
                // Not a sample failure: when no SMT backend is enabled,
                // `xlsynth-prover` falls back to the external XLS toolchain
                // prover, and upstream XLS does not understand PIR extension ops
                // like `ext_clz`.
                return;
            }
            _ => panic!("formal equivalence failed at w={}: {:?}", w, res),
        }
    }
}
