// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrPackage;
use xlsynth_pir::desugar_extensions::desugar_extensions_in_package;
use xlsynth_pir::desugar_extensions::emit_package_as_xls_ir_text;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::prove_ir_equiv;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivParallelism, EquivResult, ProverFn};

#[test]
fn ext_mask_low_equivalent_to_desugared_export_form_and_exports_to_upstream() {
    for (output_width, count_width) in [(0usize, 0usize), (1, 1), (3, 2), (8, 4), (9, 6), (16, 5)] {
        let ir = format!(
            "package test\n\nfn f(count: bits[{count_width}] id=1) -> bits[{output_width}] {{\n  ret r: bits[{output_width}] = ext_mask_low(count, id=2)\n}}\n"
        );
        let pkg = {
            let mut p = Parser::new(&ir);
            p.parse_and_validate_package().expect("parse/validate")
        };

        let exported = emit_package_as_xls_ir_text(&pkg).expect("export desugaring");
        assert!(
            !exported.contains("ext_mask_low"),
            "export should not contain ext_mask_low:\n{}",
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
                    && msg.contains("ext_mask_low")
                    && msg.contains("string-to-op conversion") =>
            {
                // Not a sample failure: when no SMT backend is enabled,
                // `xlsynth-prover` falls back to the external XLS toolchain
                // prover, and upstream XLS does not understand PIR extension ops
                // like `ext_mask_low`.
                return;
            }
            _ => panic!(
                "formal equivalence failed at output_width={} count_width={}: {:?}",
                output_width, count_width, res
            ),
        }
    }
}
