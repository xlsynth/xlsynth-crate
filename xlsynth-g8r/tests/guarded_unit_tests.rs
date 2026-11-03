// SPDX-License-Identifier: Apache-2.0

use test_case::test_case;
use xlsynth_g8r::aig::gate::GateFn;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_pir::IrBits;
use xlsynth_pir::ir_parser::Parser;

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
