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

fn build_ext_mask_low_ir_text(output_width: usize, count_width: usize) -> String {
    format!(
        "package sample\n\
top fn ext_mask_low_{output_width}b_count{count_width}b(count: bits[{count_width}] id=1) -> bits[{output_width}] {{\n\
  ret ext_mask_low.2: bits[{output_width}] = ext_mask_low(count, id=2)\n\
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
            enable_rewrite_nary_add: false,
            enable_rewrite_mask_low: false,
            array_index_lowering_strategy: Default::default(),
            unsafe_gatify_gate_operation: false,
        },
    )
    .expect("gatify")
    .gate_fn
}

#[test]
fn direct_ext_mask_low_matches_desugared_semantics_width_sweep() {
    for output_width in 0usize..=32 {
        let count_width = ceil_log2(output_width.saturating_add(1)).saturating_add(2);
        let ir_text = build_ext_mask_low_ir_text(output_width, count_width);
        let pir_fn = parse_top_fn(&ir_text);
        let mut desugared_fn = pir_fn.clone();
        desugar_extensions_in_fn(&mut desugared_fn).expect("desugar ext_mask_low");

        let gate_ext = gatify_for_test(&pir_fn);
        let gate_desugared = gatify_for_test(&desugared_fn);
        if output_width == 0 {
            assert!(
                gate_ext.outputs.len() == 1 && gate_ext.outputs[0].get_bit_count() == 0,
                "expected zero-width direct ext_mask_low lowering"
            );
            assert!(
                gate_desugared.outputs.len() == 1 && gate_desugared.outputs[0].get_bit_count() == 0,
                "expected zero-width desugared ext_mask_low lowering"
            );
            continue;
        }

        check_equivalence::prove_same_gate_fn_via_ir(&gate_ext, &gate_desugared).unwrap_or_else(
            |e| {
                panic!(
                    "expected direct ext_mask_low lowering to match desugared semantics for output_width={output_width} count_width={count_width}: {e}"
                )
            },
        );
    }
}

#[test]
fn direct_ext_mask_low_matches_desugared_for_non_power_of_two_widths() {
    for output_width in [3usize, 5, 9, 10, 17] {
        let count_width = ceil_log2(output_width.saturating_add(1)).saturating_add(3);
        let pir_fn = parse_top_fn(&build_ext_mask_low_ir_text(output_width, count_width));
        let mut desugared_fn = pir_fn.clone();
        desugar_extensions_in_fn(&mut desugared_fn).expect("desugar ext_mask_low");

        let gate_ext = gatify_for_test(&pir_fn);
        let gate_desugared = gatify_for_test(&desugared_fn);
        check_equivalence::prove_same_gate_fn_via_ir(&gate_ext, &gate_desugared).unwrap_or_else(
            |e| {
                panic!(
                    "expected non-power-of-two ext_mask_low lowering to match desugared semantics for output_width={output_width}: {e}"
                )
            },
        );
    }
}
