// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;
use test_case::test_case;
use xlsynth_g8r::aig::gate::{AigNode, GateFn};
use xlsynth_g8r::aig::get_summary_stats::{get_aig_cone_stats, get_aig_stats};
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify, gatify_prepared_fn};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_pir::IrBits;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_range_info::{IrRangeInfo, NodeRangeInfo};
use xlsynth_pir::known_bits::KnownBits;

#[test_case("ult(x, limit)", "", true; "guard prevents carry")]
#[test_case("uge(x, limit)", "", false; "guard admits carry")]
#[test_case("ult(x, limit)", ", arch=ripple_carry", false; "explicit ripple")]
#[test_case("ult(x, limit)", ", arch=brent_kung", false; "explicit brent kung")]
#[test_case("ult(x, limit)", ", arch=kogge_stone", false; "explicit kogge stone")]
fn guarded_increment_keeps_escaping_enable(comparison: &str, architecture: &str, forwarded: bool) {
    let gates = lower(comparison, architecture, false);
    let x_bits = &gates.inputs[0].bit_vector;
    let result = &gates.outputs[0].bit_vector;
    // The low output bit is the separate status rail. Bits 3..7 of the
    // unsigned result can be forwarded only on the proven, unmapped operation.
    assert_eq!(
        (3..7).all(|i| result.get_lsb(i + 1) == x_bits.get_lsb(i)),
        forwarded
    );
    exhaustive(&gates, comparison.starts_with("ult"), false);
}

#[test]
fn guarded_increment_can_keep_borrow_live_above_the_carry_bound() {
    let gates = lower("ult(x, limit)", "", true);
    exhaustive(&gates, true, true);
}

#[test_case(AdderMapping::BrentKung; "brent kung")]
#[test_case(AdderMapping::KoggeStone; "kogge stone")]
fn wide_guarded_borrow_preserves_prefix_depth(adder_mapping: AdderMapping) {
    let text = r#"fn update(x: bits[255] id=1, up: bits[1] id=2, down: bits[1] id=3) -> bits[256] {
  zero: bits[1] = literal(value=0, id=4)
  base: bits[256] = concat(x, zero, id=5)
  ret sum: bits[256] = ext_nary_add(base, up, down, signed=[false, false, false], negated=[false, false, true], id=6)
}"#;
    let f = Parser::new(text).parse_fn().unwrap();
    let options = GatifyOptions {
        adder_mapping,
        ..GatifyOptions::all_opts_disabled()
    };
    let ordinary = gatify_prepared_fn(&f, options.clone()).unwrap().gate_fn;
    let guarded = gatify_prepared_fn(
        &f,
        GatifyOptions {
            enable_rewrite_nary_add: true,
            ..options
        },
    )
    .unwrap()
    .gate_fn;

    // The zero LSB stops the increment immediately; 0 - 1 must still
    // propagate the borrow all the way to the high output bit.
    let mut sim = PreparedGateSim::new(&guarded);
    assert_eq!(
        sim.eval_outputs(&[IrBits::zero(255), IrBits::bool(false), IrBits::bool(true)]),
        vec![IrBits::all_ones(256)]
    );
    // A set bit in x stops the borrow at the corresponding bit of base.
    for stop in [0, 127, 254] {
        let x = IrBits::from_lsb_fn(255, |i| i == stop);
        let expected = IrBits::from_lsb_fn(256, |i| i <= stop);
        assert_eq!(
            sim.eval_outputs(&[x, IrBits::bool(false), IrBits::bool(true)]),
            vec![expected]
        );
    }
    let ordinary = get_aig_stats(&ordinary);
    let guarded = get_aig_stats(&guarded);
    assert!(
        guarded.max_depth <= ordinary.max_depth,
        "{adder_mapping}: guarded depth {}, ordinary depth {}",
        guarded.max_depth,
        ordinary.max_depth
    );
}

#[test_case("late, early"; "late increment")]
#[test_case("early, late"; "late decrement")]
fn late_unit_control_only_masks_the_finished_prefix(controls: &str) {
    let text = format!(
        r#"fn update(high: bits[4] id=1, low: bits[3] id=2, flags: bits[128] id=3, early: bits[1] id=4) -> bits[9] {{
  zero: bits[1] = literal(value=0, id=5)
  base: bits[8] = concat(high, zero, low, id=6)
  late: bits[1] = and_reduce(flags, id=7)
  sum: bits[8] = ext_nary_add(base, {controls}, signed=[false, false, false], negated=[false, false, true], id=8)
  ret observed: bits[9] = concat(late, sum, id=9)
}}"#
    );
    let f = Parser::new(&text).parse_fn().unwrap();
    let gates = gatify_prepared_fn(
        &f,
        GatifyOptions {
            enable_rewrite_nary_add: true,
            adder_mapping: AdderMapping::BrentKung,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    let outputs = gates.outputs[0]
        .bit_vector
        .iter_lsb_to_msb()
        .copied()
        .collect::<Vec<_>>();
    let depths = get_aig_cone_stats(&gates.gates, &outputs).root_depths;
    let late = depths[8];
    // Above the zero bit only borrow survives: one AND after the late control,
    // then two AND levels for XOR. Below it the two directions can need an OR.
    assert!(depths[4..8].iter().all(|&d| d <= late + 3), "{depths:?}");
    assert!(depths[..4].iter().all(|&d| d <= late + 4), "{depths:?}");
}

#[test]
fn decrement_enable_in_base_stops_borrow() {
    let text = r#"fn update(x: bits[254] id=1, up: bits[1] id=2, down: bits[1] id=3) -> bits[256] {
  not_up: bits[1] = not(up, id=4)
  decrement: bits[1] = and(down, not_up, id=5)
  zero: bits[1] = literal(value=0, id=6)
  base: bits[256] = concat(x, decrement, zero, id=7)
  ret sum: bits[256] = ext_nary_add(base, up, down, signed=[false, false, false], negated=[false, false, true], id=8)
}"#;
    let f = Parser::new(text).parse_fn().unwrap();
    let gates = gatify_prepared_fn(
        &f,
        GatifyOptions {
            enable_rewrite_nary_add: true,
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;

    // Bit 0 stops the carry. Bit 1 is set whenever a borrow reaches it.
    let mut sim = PreparedGateSim::new(&gates);
    for up in [false, true] {
        for down in [false, true] {
            assert_eq!(
                sim.eval_outputs(&[IrBits::zero(254), IrBits::bool(up), IrBits::bool(down)]),
                vec![IrBits::make_ubits(256, u64::from(up ^ down)).unwrap()]
            );
        }
    }
    for i in 2..256 {
        let pin = gates.outputs[0].bit_vector.get_lsb(i);
        assert_eq!(pin, gates.inputs[0].bit_vector.get_lsb(i - 2), "bit {i}");
        assert!(gates.gates[pin.node.id].get_pir_node_ids().contains(&8));
    }
}

#[test]
fn guarded_sum_applies_known_result_bits() {
    let text = r#"fn update(s: bits[1] id=1, up: bits[1] id=2, down: bits[1] id=3) -> bits[64] {
  two: bits[64] = literal(value=2, id=4)
  high: bits[64] = literal(value=128, id=5)
  base: bits[64] = sel(s, cases=[two, high], id=6)
  ret sum: bits[64] = ext_nary_add(base, up, down, signed=[false, false, false], negated=[false, false, true], id=7)
}"#;
    let f = Parser::new(text).parse_fn().unwrap();
    // The sum ranges from 1 to 129. Supply only the result fact, so that
    // folding the upper bits depends on applying it to the guarded sum.
    let mask = IrBits::make_ubits(64, u64::MAX << 8).unwrap();
    let facts = IrRangeInfo::from_node_range_info(BTreeMap::from([(
        7,
        NodeRangeInfo {
            known_bits: Some(KnownBits::from_mask_value(mask, IrBits::zero(64)).unwrap()),
            intervals: None,
            unsigned_min: None,
            unsigned_max: None,
        },
    )]));
    let gates = gatify_prepared_fn(
        &f,
        GatifyOptions {
            enable_rewrite_nary_add: true,
            adder_mapping: AdderMapping::RippleCarry,
            range_info: Some(Arc::new(facts)),
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    let mut sim = PreparedGateSim::new(&gates);
    for s in [false, true] {
        for up in [false, true] {
            for down in [false, true] {
                let result =
                    sim.eval_outputs(&[IrBits::bool(s), IrBits::bool(up), IrBits::bool(down)]);
                let base = if s { 128 } else { 2 };
                assert_eq!(
                    result[0].to_u64().unwrap(),
                    base + u64::from(up) - u64::from(down)
                );
            }
        }
    }
    for i in 8..64 {
        let bit = gates.outputs[0].bit_vector.get_lsb(i);
        assert!(
            matches!(gates.gates[bit.node.id], AigNode::Literal { value, .. } if value == bit.negated),
            "bit {i} should be a literal zero"
        );
    }
}

#[test]
fn forwarded_increment_bits_retain_update_provenance() {
    let text = r#"fn update(x: bits[7] id=1, enable: bits[1] id=2) -> bits[8] {
  zero: bits[1] = literal(value=0, id=3)
  base: bits[8] = concat(x, zero, id=4)
  ret sum: bits[8] = ext_nary_add(base, enable, signed=[false, false], negated=[false, false], id=5)
}"#;
    let f = Parser::new(text).parse_fn().unwrap();
    let lowered = gatify(
        &f,
        GatifyOptions {
            enable_rewrite_nary_add: true,
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    let gates = GateFn::try_from(lowered.to_string().as_str()).unwrap();
    for i in 1..8 {
        let pin = gates.outputs[0].bit_vector.get_lsb(i);
        assert_eq!(pin, gates.inputs[0].bit_vector.get_lsb(i - 1));
        let ids = gates.gates[pin.node.id].get_pir_node_ids();
        assert!(
            ids.contains(&1) && ids.contains(&5),
            "bit {i} has IDs {ids:?}"
        );
    }
}

fn lower(comparison: &str, architecture: &str, subtract: bool) -> GateFn {
    let (terms, signed, negated) = if subtract {
        (
            "x, enable, down",
            "false, false, false",
            "false, false, true",
        )
    } else {
        ("x, enable", "false, false", "false, false")
    };
    let comparison = comparison.trim_end_matches(')');
    let program = format!(
        r#"package update
top fn main(x: bits[7] id=1, request: bits[1] id=2, down: bits[1] id=3, notify: bits[1] id=4) -> (bits[7], bits[1]) {{
  limit: bits[7] = literal(value=6, id=5)
  selected: bits[1] = {comparison}, id=6)
  enable: bits[1] = and(request, selected, id=7)
  sum: bits[7] = ext_nary_add({terms}, signed=[{signed}], negated=[{negated}]{architecture}, id=8)
  status: bits[1] = xor(enable, notify, id=9)
  ret result: (bits[7], bits[1]) = tuple(sum, status, id=10)
}}"#
    );
    let package = Parser::new(&program).parse_and_validate_package().unwrap();
    gatify(
        package.get_top_fn().unwrap(),
        GatifyOptions {
            fold: true,
            hash: true,
            enable_rewrite_nary_add: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn
}

fn exhaustive(gates: &GateFn, below: bool, subtract: bool) {
    let mut sim = PreparedGateSim::new(gates);
    for x in 0u64..128 {
        for controls in 0u64..8 {
            let request = controls & 1 != 0;
            let down = controls & 2 != 0;
            let notify = controls & 4 != 0;
            let enabled = request && ((x < 6) == below);
            let sum = (x + u64::from(enabled)).wrapping_sub(u64::from(subtract && down)) & 127;
            let observed = sim.eval_outputs(&[
                IrBits::make_ubits(7, x).unwrap(),
                IrBits::bool(request),
                IrBits::bool(down),
                IrBits::bool(notify),
            ]);
            assert_eq!(
                observed[0].to_u64().unwrap(),
                (sum << 1) | u64::from(enabled ^ notify)
            );
        }
    }
}
