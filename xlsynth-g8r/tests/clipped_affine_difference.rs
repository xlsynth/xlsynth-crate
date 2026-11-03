// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig::dce::dce;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::check_equivalence::validate_same_fn_via_toolchain;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};
use xlsynth_pir::IrBits;
use xlsynth_pir::ir_parser::Parser;

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
        let shifted = if form == Form::Arithmetic {
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
        let distance = if form == Form::Arithmetic {
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
        let decision = if form == Form::ReversedMux {
            format!(
                "below: bits[1] = {bound}(positive, cap, id=12)\n  above: bits[1] = not(below, id=13)"
            )
        } else {
            format!("below: bits[1] = {bound}(positive, cap, id=12)")
        };
        let mux = if form == Form::ReversedMux {
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
