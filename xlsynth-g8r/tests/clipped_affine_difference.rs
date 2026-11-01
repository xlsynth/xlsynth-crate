// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig::dce::dce;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::check_equivalence::validate_same_fn_via_toolchain;
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::gatify::ir2gate::{
    GatifyOptions, IrRegion, gatify, gatify_node_as_fn, gatify_prepared_fn, gatify_region,
};
use xlsynth_pir::IrBits;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_utils::find_node_by_name;

#[test]
fn region_uses_the_supplied_distance_and_predicates() {
    let shape = Clamp {
        width: 3,
        result_width: 2,
        origin: 0,
        ..SMALL
    };
    let f = Parser::new(&shape.ir(Form::Interval, Use::FixedPosition))
        .parse_fn()
        .unwrap();
    let [distance, above_zero, below, positive, out] =
        ["distance", "above_zero", "below", "positive", "out"]
            .map(|name| find_node_by_name(&f, name).unwrap());
    let nodes = ["position", "mask", "positive", "saturated", "low", "out"]
        .map(|name| find_node_by_name(&f, name).unwrap());
    let region = gatify_region(
        IrRegion {
            function: &f,
            nodes: &nodes,
            inputs: &[f.params[0], distance, above_zero, below],
            outputs: &[distance, positive, out],
        },
        GateBuilderOptions::opt(),
    )
    .unwrap();
    let mut sim = PreparedGateSim::new(&region);
    // With x = position = origin = 0, the enclosing function computes zero.
    // The region gets its distance and both comparisons independently.
    for distance in 0..8 {
        for above in [false, true] {
            for below in [false, true] {
                let inputs = [
                    IrBits::make_ubits(3, 0).unwrap(),
                    IrBits::make_ubits(3, distance).unwrap(),
                    IrBits::bool(above),
                    IrBits::bool(below),
                ];
                let positive = if above { distance } else { 0 };
                let out = if below { positive & 3 } else { 2 };
                let expected = (distance << 5) | (positive << 2) | out;
                assert_eq!(
                    sim.eval_outputs(&inputs)[0].to_u64().unwrap(),
                    expected,
                    "distance={distance}, above={above}, below={below}"
                );
            }
        }
    }
}

#[test]
fn standalone_clamp_select_uses_its_supplied_arms() {
    // Reusing cap in the subtraction makes both of its terms available at
    // the standalone select. Only the select itself is being lowered.
    let f = Parser::new(
        r#"fn main() -> bits[3] {
  cap: bits[3] = literal(value=2, id=1)
  distance: bits[3] = ext_nary_add(cap, cap, signed=[false, false], negated=[false, true], id=2)
  zero: bits[3] = literal(value=0, id=3)
  above_zero: bits[1] = sgt(distance, zero, id=4)
  mask: bits[3] = sign_ext(above_zero, new_bit_count=3, id=5)
  positive: bits[3] = and(mask, distance, id=6)
  below: bits[1] = slt(positive, cap, id=7)
  low: bits[3] = bit_slice(positive, start=0, width=3, id=8)
  ret out: bits[3] = sel(below, cases=[cap, low], id=9)
}"#,
    )
    .parse_fn()
    .unwrap();
    let select = gatify_node_as_fn(
        &f,
        f.ret_node_ref.unwrap(),
        &GatifyOptions::all_opts_disabled(),
    )
    .unwrap();
    let mut sim = PreparedGateSim::new(&select);
    for (below, expected) in [(false, 3), (true, 5)] {
        let inputs = [
            IrBits::bool(below),
            IrBits::make_ubits(3, 3).unwrap(),
            IrBits::make_ubits(3, 5).unwrap(),
        ];
        assert_eq!(sim.eval_outputs(&inputs)[0].to_u64().unwrap(), expected);
    }
}

#[test]
fn folded_clamp_selector_keeps_the_wire_only_arm() {
    let gates = map(r#"fn main(y: bits[4] id=1) -> bits[6] {
  prefix: bits[3] = literal(value=6, id=2)
  data: bits[7] = concat(prefix, y, id=3)
  offset: bits[7] = literal(value=127, id=4)
  distance: bits[7] = ext_nary_add(offset, data, signed=[false, false], negated=[false, true], id=5)
  zero: bits[7] = literal(value=0, id=6)
  above_zero: bits[1] = sgt(distance, zero, id=7)
  mask: bits[7] = sign_ext(above_zero, new_bit_count=7, id=8)
  positive: bits[7] = and(mask, distance, id=9)
  cap: bits[7] = literal(value=32, id=10)
  below: bits[1] = slt(positive, cap, id=11)
  saturated: bits[6] = literal(value=32, id=12)
  low: bits[6] = bit_slice(positive, start=0, width=6, id=13)
  ret out: bits[6] = sel(below, cases=[saturated, low], id=14)
}"#);
    // 127 - (96 + y) = 31 - y, always in the unsaturated interval.
    let mut sim = PreparedGateSim::new(&gates);
    for y in 0..16 {
        let inputs = [IrBits::make_ubits(4, y).unwrap()];
        assert_eq!(sim.eval_outputs(&inputs)[0].to_u64().unwrap(), 31 - y);
    }
    let stats = get_aig_stats(&gates);
    assert_eq!((stats.and_nodes, stats.max_depth), (0, 0));
}

#[test]
fn wire_only_clamp_inputs_keep_the_ordinary_select() {
    let gates = map(r#"fn main(s: bits[1] id=1, y: bits[4] id=2) -> bits[6] {
  prefix: bits[2] = literal(value=3, id=3)
  pad: bits[1] = literal(value=0, id=4)
  data: bits[8] = concat(prefix, s, pad, y, id=5)
  zero: bits[8] = literal(value=0, id=6)
  one: bits[8] = literal(value=1, id=7)
  distance: bits[8] = ext_nary_add(zero, data, one, signed=[false, false, false], negated=[false, true, true], id=8)
  above_zero: bits[1] = sgt(distance, zero, id=9)
  mask: bits[8] = sign_ext(above_zero, new_bit_count=8, id=10)
  positive: bits[8] = and(mask, distance, id=11)
  cap: bits[8] = literal(value=32, id=12)
  below: bits[1] = slt(positive, cap, id=13)
  saturated: bits[6] = literal(value=32, id=14)
  low: bits[6] = bit_slice(positive, start=0, width=6, id=15)
  ret out: bits[6] = sel(below, cases=[saturated, low], id=16)
}"#);
    // The difference is ~data. When s is clear it saturates, and when s is
    // set it is 31 - y. Only the low four output bits require gates.
    let mut sim = PreparedGateSim::new(&gates);
    for s in [false, true] {
        for y in 0..16 {
            let inputs = [IrBits::bool(s), IrBits::make_ubits(4, y).unwrap()];
            let expected = if s { 31 - y } else { 32 };
            assert_eq!(sim.eval_outputs(&inputs)[0].to_u64().unwrap(), expected);
        }
    }
    let stats = get_aig_stats(&gates);
    assert_eq!((stats.and_nodes, stats.max_depth), (4, 1));
}

#[test]
fn normal_preparation_maps_flattened_and_shared_biased_differences() {
    let shape = Clamp {
        width: 64,
        digit_width: 1,
        ..SMALL
    };
    for usage in [Use::Clamp, Use::SignedPosition, Use::ReturnShifted] {
        let arithmetic = Parser::new(&shape.ir(Form::Arithmetic, usage))
            .parse_fn()
            .unwrap();
        let interval = gatify(&arithmetic, GatifyOptions::all_opts_enabled())
            .unwrap()
            .gate_fn;
        validate_same_fn_via_toolchain(&arithmetic, &interval).unwrap();

        // The same preparation runs on the reference. Its reversed mux keeps
        // the generic arithmetic path available for a depth comparison.
        let reversed = Parser::new(&shape.ir(Form::ReversedArithmeticMux, usage))
            .parse_fn()
            .unwrap();
        let generic = gatify(&reversed, GatifyOptions::all_opts_enabled())
            .unwrap()
            .gate_fn;
        let interval = get_aig_stats(&interval);
        let generic = get_aig_stats(&generic);
        assert!(
            interval.max_depth < generic.max_depth,
            "{usage:?}: {interval:?} vs {generic:?}"
        );
    }
}

#[test]
fn every_small_residue_agrees_with_the_signed_clamp() {
    for origin in [0, 11, 29] {
        for limit in [2, 6] {
            let shape = Clamp {
                origin,
                limit,
                ..SMALL
            };
            for usage in [Use::Clamp, Use::SignedPosition] {
                let gates = map(&shape.ir(Form::Interval, usage));
                let mut sim = PreparedGateSim::new(&gates);
                for x in 0..64u64 {
                    for digit in 0..4u64 {
                        let position = if usage == Use::SignedPosition && digit >= 2 {
                            digit as i64 - 4
                        } else {
                            digit as i64
                        };
                        let residue = (position - x as i64 - origin as i64).rem_euclid(64);
                        let signed = if residue >= 32 { residue - 64 } else { residue };
                        let expected = signed.clamp(0, limit as i64) as u64;
                        let inputs = [
                            IrBits::make_ubits(6, x).unwrap(),
                            IrBits::make_ubits(2, digit).unwrap(),
                        ];
                        assert_eq!(
                            sim.eval_outputs(&inputs)[0].to_u64().unwrap(),
                            expected,
                            "x={x}, digit={digit}, origin={origin}, limit={limit}, use={usage:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn interval_mapping_respects_expansion_and_word_width_limits() {
    // The interval table enumerates the digit values for each unsaturated
    // distance.
    for (width, digit_width, limit, maps) in [
        (6, 2, 8, true),
        (6, 2, 9, false),
        (64, 1, 2, true),
        (65, 1, 2, false),
    ] {
        let shape = Clamp {
            width,
            digit_width,
            limit,
            ..SMALL
        };
        check_mapping(shape, Use::Clamp, maps);
    }
}

#[test]
fn position_fanout_and_cap_limits_preserve_modular_semantics() {
    // Sign extension repeats the same digit bit without adding a new
    // independent value.
    check_mapping(Clamp { limit: 8, ..SMALL }, Use::SignedPosition, true);
    // A fixed unsigned position isolates the cap from the digit enumeration
    // limit.
    for (limit, maps) in [(32, true), (33, false)] {
        check_mapping(
            Clamp {
                width: 9,
                result_width: 6,
                limit,
                ..SMALL
            },
            Use::FixedPosition,
            maps,
        );
    }
}

#[test]
fn prescribed_adders_and_other_observers_keep_their_values() {
    for (usage, maps) in [
        (Use::OuterPrefix, false),
        (Use::InnerPrefix, true),
        (Use::ReturnInnerPrefix, true),
        (Use::ReturnDistance, false),
        (Use::ReturnPositive, false),
        (Use::ReturnShifted, true),
        (Use::OtherSign, false),
        (Use::UnsignedUpper, false),
    ] {
        check_mapping(SMALL, usage, maps);
    }
}

#[derive(Clone, Copy)]
struct Clamp {
    width: usize,
    digit_width: usize,
    result_width: usize,
    origin: u64,
    limit: u64,
}

const SMALL: Clamp = Clamp {
    width: 6,
    digit_width: 2,
    result_width: 4,
    origin: 11,
    limit: 2,
};

#[derive(Clone, Copy, PartialEq)]
enum Form {
    Arithmetic,
    ReversedArithmeticMux,
    Interval,
    ReversedMux,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Use {
    Clamp,
    FixedPosition,
    SignedPosition,
    OuterPrefix,
    InnerPrefix,
    ReturnInnerPrefix,
    ReturnDistance,
    ReturnPositive,
    ReturnShifted,
    OtherSign,
    UnsignedUpper,
}

fn check_mapping(shape: Clamp, usage: Use, maps: bool) {
    let reference = Parser::new(&shape.ir(Form::Arithmetic, usage))
        .parse_fn()
        .unwrap();
    let interval = map(&shape.ir(Form::Interval, usage));
    let mux = map(&shape.ir(Form::ReversedMux, usage));
    // The reversed selector and arms express the same clamp through the generic
    // mux.
    for gates in [&interval, &mux] {
        validate_same_fn_via_toolchain(&reference, gates).unwrap();
    }
    let selected = get_aig_stats(&interval);
    let generic = get_aig_stats(&mux);
    if maps {
        assert!(
            selected.max_depth < generic.max_depth,
            "width={}, limit={}, use={usage:?}: {selected:?} vs {generic:?}",
            shape.width,
            shape.limit
        );
    } else {
        assert_eq!(
            selected, generic,
            "width={}, limit={}, use={usage:?}",
            shape.width, shape.limit
        );
    }
}

fn map(text: &str) -> GateFn {
    let f = Parser::new(text).parse_fn().unwrap();
    let opts = GatifyOptions {
        fold: true,
        hash: true,
        ..GatifyOptions::all_opts_disabled()
    };
    dce(&gatify_prepared_fn(&f, opts).unwrap().gate_fn)
}

impl Clamp {
    fn ir(self, form: Form, usage: Use) -> String {
        let Clamp {
            width,
            digit_width,
            result_width: rw,
            origin,
            limit,
        } = self;
        let arithmetic = matches!(form, Form::Arithmetic | Form::ReversedArithmeticMux);
        let reversed_mux = matches!(form, Form::ReversedMux | Form::ReversedArithmeticMux);
        let shifted = if arithmetic {
            "add(x, origin, id=5)".into()
        } else {
            let arch = if matches!(usage, Use::InnerPrefix | Use::ReturnInnerPrefix) {
                ", arch=kogge_stone"
            } else {
                ""
            };
            format!(
                "ext_nary_add(x, origin, signed=[false, false], negated=[false, false]{arch}, id=5)"
            )
        };
        let distance = if arithmetic {
            "sub(position, shifted, id=6)".into()
        } else {
            let arch = if usage == Use::OuterPrefix {
                ", arch=brent_kung"
            } else {
                ""
            };
            format!(
                "ext_nary_add(position, shifted, signed=[false, false], negated=[false, true]{arch}, id=6)"
            )
        };
        let test = if usage == Use::OtherSign {
            "x"
        } else {
            "distance"
        };
        let bound = if usage == Use::UnsignedUpper {
            "ult"
        } else {
            "slt"
        };
        let decision = if reversed_mux {
            format!(
                "below: bits[1] = {bound}(positive, cap, id=12)\n  above: bits[1] = not(below, id=13)"
            )
        } else {
            format!("below: bits[1] = {bound}(positive, cap, id=12)")
        };
        let mux = if reversed_mux {
            "above, cases=[low, saturated]"
        } else {
            "below, cases=[saturated, low]"
        };
        let observe = match usage {
            Use::ReturnDistance => Some("distance"),
            Use::ReturnPositive => Some("positive"),
            Use::ReturnInnerPrefix | Use::ReturnShifted => Some("shifted"),
            _ => None,
        };

        let (result, tail) = if let Some(extra) = observe {
            let total = width + rw;
            (
                total,
                format!(
                    "out: bits[{rw}] = sel({mux}, id=16)\n  ret pair: bits[{total}] = concat({extra}, out, id=17)"
                ),
            )
        } else {
            (rw, format!("ret out: bits[{rw}] = sel({mux}, id=16)"))
        };
        let position = if usage == Use::FixedPosition {
            "literal(value=0, id=3)".into()
        } else if usage == Use::SignedPosition {
            format!("sign_ext(digit, new_bit_count={width}, id=3)")
        } else {
            format!("zero_ext(digit, new_bit_count={width}, id=3)")
        };
        format!(
            r#"fn main(x: bits[{width}] id=1, digit: bits[{digit_width}] id=2) -> bits[{result}] {{
  position: bits[{width}] = {position}
  origin: bits[{width}] = literal(value={origin}, id=4)
  shifted: bits[{width}] = {shifted}
  distance: bits[{width}] = {distance}
  zero: bits[{width}] = literal(value=0, id=7)
  above_zero: bits[1] = sgt({test}, zero, id=8)
  mask: bits[{width}] = sign_ext(above_zero, new_bit_count={width}, id=9)
  positive: bits[{width}] = and(mask, distance, id=10)
  cap: bits[{width}] = literal(value={limit}, id=11)
  {decision}
  saturated: bits[{rw}] = literal(value={limit}, id=14)
  low: bits[{rw}] = bit_slice(positive, start=0, width={rw}, id=15)
  {tail}
}}"#
        )
    }
}
