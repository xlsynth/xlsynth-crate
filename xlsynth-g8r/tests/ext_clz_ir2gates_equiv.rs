// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_pir::desugar_extensions::desugar_extensions_in_fn;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;
use xlsynth_pir::math::ceil_log2;

fn parse_top_fn(ir_text: &str) -> ir::Fn {
    let mut parser = ir_parser::Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().expect("parse/validate");
    pkg.get_top_fn().expect("top fn").clone()
}

fn build_ext_clz_ir_text(bit_count: u32) -> String {
    let out_w = ceil_log2((bit_count as usize).saturating_add(1));
    format!(
        "package sample\n\
top fn ext_clz_{bit_count}b(input: bits[{bit_count}] id=1) -> bits[{out_w}] {{\n\
  ret ext_clz.2: bits[{out_w}] = ext_clz(input, id=2)\n\
}}\n"
    )
}

fn gatify_for_test(pir_fn: &ir::Fn) -> xlsynth_g8r::aig::GateFn {
    gatify(
        pir_fn,
        GatifyOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            array_index_lowering_strategy: Default::default(),
        },
    )
    .expect("gatify")
    .gate_fn
}

#[test]
fn direct_ext_clz_matches_desugared_semantics_width_sweep() {
    for bit_count in 0u32..=16 {
        let ir_text = build_ext_clz_ir_text(bit_count);
        let pir_fn = parse_top_fn(&ir_text);
        let mut desugared_fn = pir_fn.clone();
        desugar_extensions_in_fn(&mut desugared_fn).expect("desugar ext_clz");

        let gate_ext = gatify_for_test(&pir_fn);
        let gate_desugared = gatify_for_test(&desugared_fn);
        if bit_count == 0 {
            assert!(
                gate_ext.outputs.len() == 1 && gate_ext.outputs[0].get_bit_count() == 0,
                "expected zero-width direct ext_clz lowering to produce one zero-width output"
            );
            assert!(
                gate_desugared.outputs.len() == 1 && gate_desugared.outputs[0].get_bit_count() == 0,
                "expected zero-width desugared ext_clz lowering to produce one zero-width output"
            );
            continue;
        }

        check_equivalence::prove_same_gate_fn_via_ir(&gate_ext, &gate_desugared).unwrap_or_else(
            |e| {
                panic!(
                    "expected direct ext_clz lowering to match desugared semantics for bit_count={bit_count}: {e}"
                )
            },
        );
    }
}
