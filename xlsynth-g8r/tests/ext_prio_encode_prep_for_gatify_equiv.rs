// SPDX-License-Identifier: Apache-2.0

use xlsynth::FnBuilder;
use xlsynth::IrPackage;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_pir::desugar_extensions::desugar_extensions_in_fn;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;
use xlsynth_pir::math::ceil_log2;

fn parse_top_fn(ir_text: &str) -> ir::Fn {
    let mut parser = ir_parser::Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().expect("parse/validate");
    pkg.get_top_fn().expect("top fn").clone()
}

fn assert_ir_fns_equivalent(orig_fn: &ir::Fn, prepared_fn: &ir::Fn) {
    let mut orig_desugared = orig_fn.clone();
    desugar_extensions_in_fn(&mut orig_desugared).expect("desugar original PIR");
    let mut prepared_desugared = prepared_fn.clone();
    desugar_extensions_in_fn(&mut prepared_desugared).expect("desugar prepared PIR");
    let orig_pkg_text = format!("package orig\n\ntop {}", orig_desugared);
    let prepared_pkg_text = format!("package prepared\n\ntop {}", prepared_desugared);
    check_equivalence::check_equivalence(&orig_pkg_text, &prepared_pkg_text)
        .expect("prepared PIR should be equivalent to original PIR");
}

fn build_encode_one_hot_ir_text(bit_count: u32, lsb_prio: bool) -> String {
    let mut package = IrPackage::new("sample").expect("create package");
    let fn_name = match lsb_prio {
        true => format!("encode_one_hot_lsb_{bit_count}b"),
        false => format!("encode_one_hot_msb_{bit_count}b"),
    };
    let mut fb = FnBuilder::new(&mut package, &fn_name, /* should_verify= */ true);

    let ty_bits = package.get_bits_type(u64::from(bit_count));
    let input = fb.param("input", &ty_bits);
    let one_hot = fb.one_hot(&input, lsb_prio, Some("one_hot"));
    let encoded = fb.encode(&one_hot, Some("encode"));

    let _ = fb
        .build_with_return_value(&encoded)
        .expect("build function");
    package.set_top_by_name(&fn_name).expect("set top");
    package.to_string()
}

fn build_encode_one_hot_with_extra_user_ir_text(bit_count: u32, lsb_prio: bool) -> String {
    let mut package = IrPackage::new("sample").expect("create package");
    let fn_name = match lsb_prio {
        true => format!("encode_one_hot_with_extra_user_lsb_{bit_count}b"),
        false => format!("encode_one_hot_with_extra_user_msb_{bit_count}b"),
    };
    let mut fb = FnBuilder::new(&mut package, &fn_name, /* should_verify= */ true);

    let ty_bits = package.get_bits_type(u64::from(bit_count));
    let input = fb.param("input", &ty_bits);
    let one_hot = fb.one_hot(&input, lsb_prio, Some("one_hot"));
    let encoded = fb.encode(&one_hot, Some("encode"));
    let any = fb.or_reduce(&one_hot, Some("any"));
    let result = fb.tuple(&[&encoded, &any], Some("result"));

    let _ = fb.build_with_return_value(&result).expect("build function");
    package.set_top_by_name(&fn_name).expect("set top");
    package.to_string()
}

fn build_ext_prio_encode_ir_text(bit_count: u32, lsb_prio: bool) -> String {
    let out_w = ceil_log2((bit_count as usize).saturating_add(1));
    format!(
        "package sample\n\
top fn ext_prio_encode_{bit_count}b_lsb{lsb_prio}(input: bits[{bit_count}] id=1) -> bits[{out_w}] {{\n\
  ret ext_prio_encode.2: bits[{out_w}] = ext_prio_encode(input,lsb_prio={lsb_prio},id=2)\n\
}}\n"
    )
}

fn gatify_for_test(pir_fn: &ir::Fn, enable_rewrite_prio_encode: bool) -> xlsynth_g8r::aig::GateFn {
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
            enable_rewrite_prio_encode,
            enable_rewrite_nary_add: false,
            enable_rewrite_mask_low: false,
            array_index_lowering_strategy: Default::default(),
        },
    )
    .expect("gatify")
    .gate_fn
}

#[test]
fn rewrite_triggers_for_all_positive_widths() {
    for bit_count in 1u32..=16 {
        for lsb_prio in [true, false] {
            let ir_text = build_encode_one_hot_ir_text(bit_count, lsb_prio);
            let pir_fn = parse_top_fn(&ir_text);

            let prepared = prep_for_gatify(
                &pir_fn,
                None,
                PrepForGatifyOptions {
                    enable_rewrite_prio_encode: true,
                    enable_rewrite_nary_add: false,
                    enable_rewrite_mask_low: false,
                    ..PrepForGatifyOptions::default()
                },
            );
            let prepared_text = prepared.to_string();
            assert!(
                prepared_text.contains("ext_prio_encode("),
                "expected ext_prio_encode rewrite for bit_count={bit_count} lsb_prio={lsb_prio}; got:\n{prepared_text}"
            );
            assert_ir_fns_equivalent(&pir_fn, &prepared);
        }
    }
}

#[test]
fn rewrite_does_not_trigger_when_one_hot_has_multiple_users() {
    let bit_count = 8u32;
    for lsb_prio in [true, false] {
        let ir_text = build_encode_one_hot_with_extra_user_ir_text(bit_count, lsb_prio);
        let pir_fn = parse_top_fn(&ir_text);

        let prepared = prep_for_gatify(
            &pir_fn,
            None,
            PrepForGatifyOptions {
                enable_rewrite_prio_encode: true,
                enable_rewrite_nary_add: false,
                enable_rewrite_mask_low: false,
                ..PrepForGatifyOptions::default()
            },
        );
        let prepared_text = prepared.to_string();
        assert!(
            !prepared_text.contains("ext_prio_encode("),
            "unexpected ext_prio_encode rewrite when one_hot has multiple users; got:\n{prepared_text}"
        );
        assert_ir_fns_equivalent(&pir_fn, &prepared);
    }
}

#[test]
fn gate_graph_equivalence_old_vs_rewrite_width_sweep() {
    for bit_count in 1u32..=16 {
        for lsb_prio in [true, false] {
            let ir_text = build_encode_one_hot_ir_text(bit_count, lsb_prio);
            let pir_fn = parse_top_fn(&ir_text);

            let gate_old = gatify_for_test(&pir_fn, /* enable_rewrite_prio_encode= */ false);
            let gate_new = gatify_for_test(&pir_fn, /* enable_rewrite_prio_encode= */ true);

            check_equivalence::prove_same_gate_fn_via_ir(&gate_old, &gate_new)
                .expect("expected old vs rewritten prio-encode lowering to be equivalent");
        }
    }
}

#[test]
fn direct_ext_prio_encode_matches_desugared_semantics_width_sweep() {
    for bit_count in 0u32..=16 {
        for lsb_prio in [true, false] {
            let ir_text = build_ext_prio_encode_ir_text(bit_count, lsb_prio);
            let pir_fn = parse_top_fn(&ir_text);
            let mut desugared_fn = pir_fn.clone();
            desugar_extensions_in_fn(&mut desugared_fn).expect("desugar ext_prio_encode");

            let gate_ext = gatify_for_test(&pir_fn, /* enable_rewrite_prio_encode= */ false);
            let gate_desugared =
                gatify_for_test(&desugared_fn, /* enable_rewrite_prio_encode= */ false);
            if bit_count == 0 {
                assert!(
                    gate_ext.outputs.len() == 1 && gate_ext.outputs[0].get_bit_count() == 0,
                    "expected zero-width direct ext_prio_encode lowering to produce one zero-width output"
                );
                assert!(
                    gate_desugared.outputs.len() == 1
                        && gate_desugared.outputs[0].get_bit_count() == 0,
                    "expected zero-width desugared ext_prio_encode lowering to produce one zero-width output"
                );
                continue;
            }

            check_equivalence::prove_same_gate_fn_via_ir(&gate_ext, &gate_desugared)
                .unwrap_or_else(|e| {
                    panic!(
                        "expected direct ext_prio_encode lowering to match desugared semantics for bit_count={bit_count} lsb_prio={lsb_prio}: {e}"
                    )
                });
        }
    }
}
