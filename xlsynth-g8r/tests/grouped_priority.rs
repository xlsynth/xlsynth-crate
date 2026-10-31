// SPDX-License-Identifier: Apache-2.0

use test_case::test_case;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_pir::ir_parser::Parser;

#[test_case("x, y, x, z, x, y, z, y", "z", 4; "two payloads with intervening zero arms")]
#[test_case("w, x, y, w, x, z, y, w", "z", 5; "three interleaved payloads")]
#[test_case("m, x, z, m, x, m, x, z", "z", 4; "word mask becomes a scalar enable")]
#[test_case("x, y, w, x, y, x, w, y", "w", 0; "nonzero default")]
#[test_case("w, x, n, y, x, n, z, y", "z", 7; "overlapping data leaves")]
#[test_case("z, y, t, x, t, z, x, x", "z", 7; "nested mux can return zero")]
fn repeated_words_preserve_priority(arms: &str, default: &str, gates_per_bit: usize) {
    let widths = [8, 32];
    let costs = widths.map(|width| {
    let text = format!(
        r#"package priority_words

top fn choose(s: bits[8] id=1, x: bits[{width}] id=2, y: bits[{width}] id=3, w: bits[{width}] id=4, e: bits[1] id=5) -> bits[{width}] {{
  z: bits[{width}] = literal(value=0, id=6)
  mask: bits[{width}] = sign_ext(e, new_bit_count={width}, id=7)
  m: bits[{width}] = and(mask, y, id=8)
  n: bits[{width}] = sel(e, cases=[w, x], id=9)
  t: bits[{width}] = sel(e, cases=[z, x], id=10)
  ret selected: bits[{width}] = priority_sel(s, cases=[{arms}], default={default}, id=11)
}}
"#
    );
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let gates = gatify(
        f,
        GatifyOptions {
            fold: true,
            hash: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    check_equivalence::validate_same_fn_via_toolchain(f, &gates).unwrap();
    get_aig_stats(&gates).and_nodes
    });
    if gates_per_bit > 0 {
        // Word width changes only the data path. Measuring the incremental
        // cost removes the scalar decoder and works across different tag
        // shapes.
        let extra_bits = widths[1] - widths[0];
        assert!(
            costs[1] <= costs[0] + extra_bits * gates_per_bit,
            "{costs:?}"
        );
    }
}

#[test]
fn grouped_priority_preserves_aggregate_layout() {
    let text = r#"package aggregate_priority

top fn choose(s: bits[5] id=1, a: (bits[2], bits[7]) id=2, b: (bits[2], bits[7]) id=3) -> (bits[2], bits[7]) {
  z: (bits[2], bits[7]) = literal(value=(0, 0), id=4)
  ret value: (bits[2], bits[7]) = priority_sel(s, cases=[b, z, a, b, a], default=z, id=5)
}
"#;
    let package = Parser::new(text).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let gate_fn = gatify(
        f,
        GatifyOptions {
            fold: true,
            hash: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn;
    check_equivalence::validate_same_fn_via_toolchain(f, &gate_fn).unwrap();
}
