// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrPackage;
use xlsynth_pir::desugar_extensions::desugar_extensions_in_package;
use xlsynth_pir::desugar_extensions::emit_package_as_xls_ir_text;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::prove_ir_equiv;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivParallelism, EquivResult, ProverFn};

#[test]
fn ext_normalize_left_equivalent_to_desugared_export_form_and_exports_to_upstream() {
    for (input_width, normalized_bit_count, shift_offset, clz_bit_count) in [
        (1usize, 1usize, 0usize, None),
        (4, 4, 0, Some(3)),
        (4, 8, 1, Some(3)),
        (7, 8, 0, Some(8)),
        (8, 16, 1, None),
    ] {
        let ret_ty = match clz_bit_count {
            Some(clz_bit_count) => {
                format!("(bits[{normalized_bit_count}], bits[{clz_bit_count}])")
            }
            None => format!("bits[{normalized_bit_count}]"),
        };
        let clz_attr = clz_bit_count
            .map(|clz_bit_count| format!(", clz_bit_count={clz_bit_count}"))
            .unwrap_or_default();
        let ir = format!(
            "package test\n\nfn f(arg: bits[{input_width}] id=1) -> {ret_ty} {{\n  ret r: {ret_ty} = ext_normalize_left(arg, shift_offset={shift_offset}, normalized_bit_count={normalized_bit_count}{clz_attr}, id=2)\n}}\n"
        );
        let pkg = {
            let mut p = Parser::new(&ir);
            p.parse_and_validate_package().expect("parse/validate")
        };

        let exported = emit_package_as_xls_ir_text(&pkg).expect("export desugaring");
        assert!(
            !exported.contains("ext_normalize_left"),
            "export should not contain ext_normalize_left:\n{}",
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
            _ => panic!(
                "formal equivalence failed at input_width={} normalized_bit_count={} shift_offset={} clz_bit_count={:?}: {:?}",
                input_width, normalized_bit_count, shift_offset, clz_bit_count, res
            ),
        }
    }
}
