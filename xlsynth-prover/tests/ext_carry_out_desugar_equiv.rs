// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrPackage;
use xlsynth_pir::desugar_extensions::desugar_extensions_in_package;
use xlsynth_pir::desugar_extensions::emit_package_as_xls_ir_text;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::prove_ir_equiv;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivParallelism, EquivResult, ProverFn};

#[test]
fn ext_carry_out_equivalent_to_desugared_export_form_and_exports_to_upstream() {
    for w in 1u64..=16u64 {
        let ir = format!(
            "package test\n\nfn f(lhs: bits[{w}] id=1, rhs: bits[{w}] id=2, c_in: bits[1] id=3) -> bits[1] {{\n  ret r: bits[1] = ext_carry_out(lhs, rhs, c_in, id=4)\n}}\n"
        );
        let pkg = {
            let mut p = Parser::new(&ir);
            p.parse_and_validate_package().expect("parse/validate")
        };

        // Export smoke: desugaring removes extension spelling and is accepted by
        // upstream XLS.
        let exported = emit_package_as_xls_ir_text(&pkg).expect("export desugaring");
        assert!(
            !exported.contains("ext_carry_out"),
            "export should not contain ext_carry_out:\n{}",
            exported
        );
        let exported_pkg = IrPackage::parse_ir(&exported, None).expect("upstream parse");
        exported_pkg.verify().expect("upstream verify");

        // Formal equivalence: original PIR with ext op is equivalent to the
        // desugared PIR in the XLS IR basis ops.
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
            // External provers require ignore today; this is still a full
            // equivalence proof for functions without assertions.
            AssertionSemantics::Ignore,
            None,
            false,
        );
        match res {
            EquivResult::Proved => {}
            EquivResult::ToolchainDisproved(msg)
                if msg.contains("Unknown operation")
                    && msg.contains("ext_carry_out")
                    && msg.contains("string-to-op conversion") =>
            {
                // Not a sample failure: when no SMT backend is enabled,
                // `xlsynth-prover` falls back to the external XLS toolchain
                // prover, and upstream XLS does not understand PIR extension ops
                // like `ext_carry_out`.
                //
                // This test provides strong coverage when an SMT backend is
                // available; in toolchain-only configurations we rely on
                // `xlsynth-pir`'s interpreter/roundtrip tests for extension-op
                // semantics.
                return;
            }
            _ => panic!("formal equivalence failed at w={}: {:?}", w, res),
        }
    }
}
