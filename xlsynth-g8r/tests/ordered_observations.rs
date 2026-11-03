// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::process_ir_path::{
    CanonicalG8rOptions, canonical_ir_text_to_g8r_lowering_artifacts,
};
use xlsynth_pir::IrBits;

#[test]
fn carry_select_preserves_word_and_endpoints_on_both_sides_of_ties() {
    let ir = r#"package test
top fn sample(a: bits[4] id=1, b: bits[4] id=2) -> bits[6] {
  not_a: bits[4] = not(a, id=3)
  aw: bits[5] = zero_ext(not_a, new_bit_count=5, id=4)
  bw: bits[5] = zero_ext(b, new_bit_count=5, id=5)
  sum: bits[5] = add(aw, bw, id=6)
  decision: bits[1] = bit_slice(sum, start=4, width=1, id=7)
  maximum: bits[4] = sel(decision, cases=[a,b], id=8)
  minimum: bits[4] = sel(decision, cases=[b,a], id=9)
  z: bits[4] = literal(value=0, id=10)
  k: bits[4] = literal(value=11, id=11)
  empty: bits[1] = eq(minimum,z,id=12)
  high: bits[1] = uge(maximum,k,id=13)
  ret packed: bits[6] = concat(maximum,empty,high,id=14)
}"#;
    let mut opts = CanonicalG8rOptions {
        enable_formal_array_alias_analysis: false,
        fold: true,
        hash: true,
        fraig: false,
        enable_rewrite_carry_out: false,
        enable_rewrite_prio_encode: false,
        ..Default::default()
    };
    let graph = canonical_ir_text_to_g8r_lowering_artifacts(ir, Some("sample"), &opts)
        .unwrap()
        .gate_fn;
    opts.enable_ordered_observations = false;
    let off = canonical_ir_text_to_g8r_lowering_artifacts(ir, Some("sample"), &opts)
        .unwrap()
        .gate_fn;
    let emit = xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary;
    assert_eq!(emit(&graph, true).unwrap(), emit(&off, true).unwrap());
    for a in 0..16 {
        for b in 0..16 {
            let values = [a, b].map(|x| IrBits::make_ubits(4, x).unwrap());
            let expected = a.max(b) * 4 + u64::from(a.min(b) == 0) * 2 + u64::from(a.max(b) >= 11);
            assert_eq!(
                gate_sim::eval(&graph, &values, Collect::None).outputs,
                vec![IrBits::make_ubits(6, expected).unwrap()]
            );
        }
    }
}

#[test]
fn predicates_of_an_owned_signed_or_unsigned_extremum_follow_its_order() {
    for signed in [false, true] {
        for maximum in [false, true] {
            let cmp = if signed { "sge" } else { "uge" };
            let arms = if maximum { "b,a" } else { "a,b" };
            let bound = if signed { 9 } else { 7 };
            let peak = if signed { 7 } else { 15 };
            let ir = format!(
                "package t
top fn sample(a: bits[4] id=1, b: bits[4] id=2) -> bits[2] {{
  c: bits[1] = {cmp}(a,b,id=3)
  w: bits[4] = sel(c, cases=[{arms}], id=4)
  k: bits[4] = literal(value={bound},id=5)
  m: bits[4] = literal(value={peak},id=6)
  above: bits[1] = {cmp}(w,k,id=7)
  endpoint: bits[1] = eq(w,m,id=8)
  ret result: bits[2] = concat(above,endpoint,id=9)
}}"
            );
            let mut opts = CanonicalG8rOptions {
                enable_formal_array_alias_analysis: false,
                fold: true,
                hash: true,
                fraig: false,
                cut_db_rewrite: false,
                ..Default::default()
            };
            let on = canonical_ir_text_to_g8r_lowering_artifacts(&ir, Some("sample"), &opts)
                .unwrap()
                .gate_fn;
            opts.enable_ordered_observations = false;
            let off = canonical_ir_text_to_g8r_lowering_artifacts(&ir, Some("sample"), &opts)
                .unwrap()
                .gate_fn;
            let emit = xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary;
            assert_ne!(emit(&on, true).unwrap(), emit(&off, true).unwrap());
            let order_key = |x: u64| if signed { x ^ 8 } else { x };
            for a in 0..16 {
                for b in 0..16 {
                    let value = if (order_key(a) >= order_key(b)) == maximum {
                        a
                    } else {
                        b
                    };
                    let expected = 2 * u64::from(order_key(value) >= order_key(bound))
                        + u64::from(value == peak);
                    let inputs = [a, b].map(|v| IrBits::make_ubits(4, v).unwrap());
                    let result = vec![IrBits::make_ubits(2, expected).unwrap()];
                    for graph in [&on, &off] {
                        assert_eq!(
                            gate_sim::eval(graph, &inputs, Collect::None).outputs,
                            result
                        );
                    }
                }
            }
        }
    }
}
