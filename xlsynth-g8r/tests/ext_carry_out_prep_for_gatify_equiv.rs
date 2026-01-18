// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate;
use xlsynth_g8r::gatify::prep_for_gatify::PrepForGatifyOptions;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_range_info::IrRangeInfo;

fn parse_top_fn(ir_text: &str) -> ir::Fn {
    let mut parser = ir_parser::Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().expect("parse/validate");
    pkg.get_top_fn().expect("top fn").clone()
}

fn build_range_info(ir_text: &str, top: &str, pir_fn: &ir::Fn) -> Arc<IrRangeInfo> {
    let mut xlsynth_pkg = xlsynth::IrPackage::parse_ir(ir_text, None).expect("xlsynth parse");
    xlsynth_pkg.set_top_by_name(top).expect("set top");
    let analysis = xlsynth_pkg.create_ir_analysis().expect("analysis");
    IrRangeInfo::build_from_analysis(&analysis, pir_fn).expect("range info")
}

fn make_add_slice_ir_text(w: usize, start: usize) -> String {
    assert!(w >= 1);
    assert!(start <= w);
    let add_w = w + 1;

    // Shape (a): bit_slice(add(add(zero_ext(x), zero_ext(y)), zero_ext(c_in)),
    // start, width=1)
    //
    // This is the canonical carry-out idiom when start==w (the MSB of the add
    // result).
    format!(
        "package sample

top fn cone(x: bits[{w}] id=1, y: bits[{w}] id=2, c_in: bits[1] id=3) -> bits[1] {{
  zx: bits[{add_w}] = zero_ext(x, new_bit_count={add_w}, id=4)
  zy: bits[{add_w}] = zero_ext(y, new_bit_count={add_w}, id=5)
  zc: bits[{add_w}] = zero_ext(c_in, new_bit_count={add_w}, id=6)
  add.7: bits[{add_w}] = add(zx, zy, id=7)
  add.8: bits[{add_w}] = add(add.7, zc, id=8)
  ret bit_slice.9: bits[1] = bit_slice(add.8, start={start}, width=1, id=9)
}}
"
    )
}

#[test]
fn carry_out_rewrite_safe_when_operands_are_explicitly_zero_extended() {
    // Safe rewrite shape:
    //   add: bits[9] = add(zero_ext(x: bits[8]), zero_ext(y: bits[8]))
    //   ret: bits[1] = bit_slice(add, start=8, width=1)
    //
    // Here the MSBs of the add operands are *provably zero by construction*.
    let ir_text = "package sample

top fn cone(x: bits[8] id=1, y: bits[8] id=2) -> bits[1] {
  zx: bits[9] = zero_ext(x, new_bit_count=9, id=3)
  zy: bits[9] = zero_ext(y, new_bit_count=9, id=4)
  add.5: bits[9] = add(zx, zy, id=5)
  ret bit_slice.6: bits[1] = bit_slice(add.5, start=8, width=1, id=6)
}
";
    let pir_fn = parse_top_fn(ir_text);

    let range_info = build_range_info(ir_text, "cone", &pir_fn);
    let prepared = xlsynth_g8r::gatify::prep_for_gatify::prep_for_gatify(
        &pir_fn,
        Some(range_info.as_ref()),
        PrepForGatifyOptions {
            enable_rewrite_carry_out: true,
        },
    );
    let prepared_text = prepared.to_string();
    assert!(
        prepared_text.contains("ext_carry_out(") && !prepared_text.contains("add("),
        "expected rewrite to ext_carry_out; got:\n{}",
        prepared_text
    );

    let gatify_output = ir2gate::gatify(
        &pir_fn,
        ir2gate::GatifyOptions {
            fold: false,
            hash: false,
            check_equivalence: true,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: Some(range_info),
            enable_rewrite_carry_out: true,
        },
    )
    .expect("gatify");

    // Double-check explicitly (gatify also checks when check_equivalence=true).
    check_equivalence::validate_same_fn(&pir_fn, &gatify_output.gate_fn).expect("equiv");
}

#[test]
fn carry_out_rewrite_sweep_up_to_4_bits_only_triggers_for_msb_slice() {
    for w in 1..=4 {
        for start in 0..=w {
            let ir_text = make_add_slice_ir_text(w, start);
            let pir_fn = parse_top_fn(&ir_text);

            let prepared = xlsynth_g8r::gatify::prep_for_gatify::prep_for_gatify(
                &pir_fn,
                None,
                PrepForGatifyOptions {
                    enable_rewrite_carry_out: true,
                },
            );
            let prepared_text = prepared.to_string();

            if start == w {
                assert!(
                    prepared_text.contains("ext_carry_out("),
                    "expected ext_carry_out rewrite for w={w} start={start}; got:\n{prepared_text}"
                );
            } else {
                assert!(
                    !prepared_text.contains("ext_carry_out("),
                    "unexpected ext_carry_out rewrite for w={w} start={start}; got:\n{prepared_text}"
                );
            }

            // Prove semantics for every case (including non-carry slices).
            let gatify_output = ir2gate::gatify(
                &pir_fn,
                ir2gate::GatifyOptions {
                    fold: false,
                    hash: false,
                    check_equivalence: true,
                    adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::default(),
                    mul_adder_mapping: None,
                    range_info: None,
                    enable_rewrite_carry_out: true,
                },
            )
            .expect("gatify");
            check_equivalence::validate_same_fn(&pir_fn, &gatify_output.gate_fn).expect("equiv");
        }
    }
}

#[test]
fn carry_out_rewrite_does_not_trigger_on_unconstrained_full_width_operands() {
    // This is the previously-incorrect rewrite case:
    //   add: bits[9] = add(a: bits[9], b: bits[9])
    //   ret: bits[1] = bit_slice(add, start=8, width=1)
    //
    // This is *not* generally equal to carry-out of the low 8-bit add, so we
    // must not rewrite it without an MSB=0 proof.
    let ir_text = "package sample

top fn cone(a: bits[9] id=1, b: bits[9] id=2) -> bits[1] {
  add.3: bits[9] = add(a, b, id=3)
  ret bit_slice.4: bits[1] = bit_slice(add.3, start=8, width=1, id=4)
}
";
    let pir_fn = parse_top_fn(ir_text);

    let prepared = xlsynth_g8r::gatify::prep_for_gatify::prep_for_gatify(
        &pir_fn,
        None,
        PrepForGatifyOptions {
            enable_rewrite_carry_out: true,
        },
    );
    let prepared_text = prepared.to_string();
    assert!(
        !prepared_text.contains("ext_carry_out("),
        "unexpected rewrite to ext_carry_out; got:\n{}",
        prepared_text
    );

    let gatify_output = ir2gate::gatify(
        &pir_fn,
        ir2gate::GatifyOptions {
            fold: false,
            hash: false,
            check_equivalence: true,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: true,
        },
    )
    .expect("gatify");

    check_equivalence::validate_same_fn(&pir_fn, &gatify_output.gate_fn).expect("equiv");
}

#[test]
fn carry_out_rewrite_can_trigger_with_range_info_msb_zero_proof() {
    // With a range proof, this becomes safe:
    // `umod(..., 256)` constrains the 9-bit values to <256, so their MSB (bit 8)
    // is provably 0. Then the MSB of the 9-bit sum equals carry-out of the low
    // 8-bit add.
    let ir_text = "package sample

top fn cone(p0: bits[9] id=1, p1: bits[9] id=2) -> bits[1] {
  m: bits[9] = literal(value=256, id=3)
  x: bits[9] = umod(p0, m, id=4)
  y: bits[9] = umod(p1, m, id=5)
  add.6: bits[9] = add(x, y, id=6)
  ret bit_slice.7: bits[1] = bit_slice(add.6, start=8, width=1, id=7)
}
";
    let pir_fn = parse_top_fn(ir_text);
    let range_info = build_range_info(ir_text, "cone", &pir_fn);

    let prepared = xlsynth_g8r::gatify::prep_for_gatify::prep_for_gatify(
        &pir_fn,
        Some(range_info.as_ref()),
        PrepForGatifyOptions {
            enable_rewrite_carry_out: true,
        },
    );
    let prepared_text = prepared.to_string();
    assert!(
        prepared_text.contains("ext_carry_out(") && !prepared_text.contains("add("),
        "expected rewrite to ext_carry_out with range proof; got:\n{}",
        prepared_text
    );

    let gatify_output = ir2gate::gatify(
        &pir_fn,
        ir2gate::GatifyOptions {
            fold: false,
            hash: false,
            check_equivalence: true,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: Some(range_info),
            enable_rewrite_carry_out: true,
        },
    )
    .expect("gatify");

    check_equivalence::validate_same_fn(&pir_fn, &gatify_output.gate_fn).expect("equiv");
}
