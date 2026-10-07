// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_pir::desugar_extensions::desugar_extensions_in_fn;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::math::ceil_log2;

#[test]
fn both_priority_sentinels_match_basis_for_irregular_and_wide_inputs() {
    for n in [
        0usize, 1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 96, 127, 128,
    ] {
        for low in [false, true] {
            let w = ceil_log2(n + 1);
            let text = format!(
                r#"package test
top fn main(x: bits[{n}] id=1) -> bits[{w}] {{
  ret out: bits[{w}] = ext_prio_encode(x, lsb_prio={low}, id=2)
}}
"#
            );
            let package = Parser::new(&text).parse_and_validate_package().unwrap();
            let source = package.get_top_fn().unwrap().clone();
            let mut basis = source.clone();
            desugar_extensions_in_fn(&mut basis).unwrap();
            let options = GatifyOptions::all_opts_disabled();
            let a = gatify(&source, options.clone()).unwrap().gate_fn;
            let b = gatify(&basis, options).unwrap().gate_fn;
            if n == 0 {
                assert_eq!(a.outputs[0].get_bit_count(), 0);
                assert_eq!(b.outputs[0].get_bit_count(), 0);
            } else {
                check_equivalence::prove_same_gate_fn_via_ir_via_toolchain(&a, &b)
                    .unwrap_or_else(|e| panic!("n={n} low={low}: {e}"));
            }
        }
    }
}

#[test]
fn priority_index_reuse_preserves_pir_node_ids() {
    let text = r#"package test
top fn main(x: bits[2] id=1) -> bits[2] {
  ret out: bits[2] = ext_prio_encode(x, lsb_prio=false, id=2)
}
"#;
    let package = Parser::new(text).parse_and_validate_package().unwrap();
    for fold_and_hash in [false, true] {
        let options = GatifyOptions {
            fold: fold_and_hash,
            hash: fold_and_hash,
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_disabled()
        };
        let gates = gatify(package.get_top_fn().unwrap(), options)
            .unwrap()
            .gate_fn;
        let index_bit = gates.outputs[0].bit_vector.get_lsb(0);
        // The high-priority index reuses x[1] but still implements IR node 2.
        assert_eq!(index_bit.node, gates.inputs[0].bit_vector.get_lsb(1).node);
        assert_eq!(
            gates.gates[index_bit.node.id].get_pir_node_ids(),
            &[1, 2],
            "fold_and_hash={fold_and_hash}"
        );
    }
}
