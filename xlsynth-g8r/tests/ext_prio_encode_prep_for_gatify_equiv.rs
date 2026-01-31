// SPDX-License-Identifier: Apache-2.0

use xlsynth::FnBuilder;
use xlsynth::IrPackage;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;

fn parse_top_fn(ir_text: &str) -> ir::Fn {
    let mut parser = ir_parser::Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().expect("parse/validate");
    pkg.get_top_fn().expect("top fn").clone()
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
        },
    )
    .expect("gatify")
    .gate_fn
}

#[test]
fn rewrite_triggers_only_for_pow2_width() {
    for bit_count in 1u32..=10 {
        for lsb_prio in [true, false] {
            let ir_text = build_encode_one_hot_ir_text(bit_count, lsb_prio);
            let pir_fn = parse_top_fn(&ir_text);

            let prepared = prep_for_gatify(
                &pir_fn,
                None,
                PrepForGatifyOptions {
                    enable_rewrite_prio_encode: true,
                    ..PrepForGatifyOptions::default()
                },
            );
            let prepared_text = prepared.to_string();
            let should_rewrite = (bit_count as usize).is_power_of_two();

            if should_rewrite {
                assert!(
                    prepared_text.contains("ext_prio_encode("),
                    "expected ext_prio_encode rewrite for bit_count={bit_count} lsb_prio={lsb_prio}; got:\n{prepared_text}"
                );
            } else {
                assert!(
                    !prepared_text.contains("ext_prio_encode("),
                    "unexpected ext_prio_encode rewrite for bit_count={bit_count} lsb_prio={lsb_prio}; got:\n{prepared_text}"
                );
            }
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
                ..PrepForGatifyOptions::default()
            },
        );
        let prepared_text = prepared.to_string();
        assert!(
            !prepared_text.contains("ext_prio_encode("),
            "unexpected ext_prio_encode rewrite when one_hot has multiple users; got:\n{prepared_text}"
        );
    }
}

#[test]
fn gate_graph_equivalence_old_vs_rewrite_pow2_sweep() {
    for bit_count in [1u32, 2, 4, 8, 16] {
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
