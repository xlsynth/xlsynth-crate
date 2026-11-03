// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::process_ir_path::{
    CanonicalG8rOptions, canonical_ir_text_to_g8r_lowering_artifacts,
};
use xlsynth_pir::IrBits;

#[test]
fn threshold_of_a_folded_maximum_uses_the_surviving_arm() {
    let ir = r#"package test
top fn sample(x: bits[4] id=1) -> bits[1] {
  one: bits[4] = literal(value=1, id=2)
  y: bits[4] = or(x, one, id=3)
  c: bits[1] = uge(x, y, id=4)
  w: bits[4] = sel(c, cases=[y, x], id=5)
  k: bits[4] = literal(value=10, id=6)
  ret high: bits[1] = ugt(w, k, id=7)
}"#;
    for enable_ordered_observations in [false, true] {
        let options = CanonicalG8rOptions {
            enable_ordered_observations,
            enable_formal_array_alias_analysis: false,
            fraig: false,
            reassociation: false,
            cut_db_rewrite: false,
            ..Default::default()
        };
        let mapped =
            canonical_ir_text_to_g8r_lowering_artifacts(ir, Some("sample"), &options).unwrap();
        let stats = get_aig_stats(&mapped.gate_fn);
        assert_eq!((stats.and_nodes, stats.max_depth), (2, 2));
        for x in 0..16 {
            let inputs = [IrBits::make_ubits(4, x).unwrap()];
            let expected = IrBits::make_ubits(1, u64::from((x | 1) > 10)).unwrap();
            assert_eq!(
                gate_sim::eval(&mapped.gate_fn, &inputs, Collect::None).outputs,
                [expected]
            );
        }
    }
}

#[test]
fn threshold_and_reduced_slice_share_an_exclusively_observed_maximum() {
    for expose_slice in [false, true] {
        let (width, outputs) = if expose_slice {
            (4, "above, both, upper")
        } else {
            (2, "above, both")
        };
        let ir = format!(
            r#"package test
top fn sample(a: bits[4] id=1, b: bits[4] id=2) -> bits[{width}] {{
  c: bits[1] = uge(a, b, id=3)
  w: bits[4] = sel(c, cases=[b, a], id=4)
  k: bits[4] = literal(value=5, id=5)
  above: bits[1] = ugt(w, k, id=6)
  upper: bits[2] = bit_slice(w, start=2, width=2, id=7)
  both: bits[1] = and_reduce(upper, id=8)
  ret result: bits[{width}] = concat({outputs}, id=9)
}}"#
        );
        let mut options = CanonicalG8rOptions {
            enable_formal_array_alias_analysis: false,
            fraig: false,
            reassociation: false,
            cut_db_rewrite: false,
            ..Default::default()
        };
        let on =
            canonical_ir_text_to_g8r_lowering_artifacts(&ir, Some("sample"), &options).unwrap();
        options.enable_ordered_observations = false;
        let off =
            canonical_ir_text_to_g8r_lowering_artifacts(&ir, Some("sample"), &options).unwrap();
        if expose_slice {
            let emit = xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary;
            assert_eq!(
                emit(&on.gate_fn, true).unwrap(),
                emit(&off.gate_fn, true).unwrap()
            );
        } else {
            let distributed = get_aig_stats(&on.gate_fn);
            let ordinary = get_aig_stats(&off.gate_fn);
            assert!(
                distributed.and_nodes < ordinary.and_nodes
                    && distributed.max_depth < ordinary.max_depth,
                "distributed {distributed:?}, ordinary {ordinary:?}"
            );
        }
        for a in 0..16 {
            for b in 0..16 {
                let w = a.max(b);
                let predicates = 2 * u64::from(w > 5) + u64::from(w >= 12);
                let expected = if expose_slice {
                    predicates * 4 + (w >> 2)
                } else {
                    predicates
                };
                let inputs = [a, b].map(|v| IrBits::make_ubits(4, v).unwrap());
                assert_eq!(
                    gate_sim::eval(&on.gate_fn, &inputs, Collect::None).outputs,
                    [IrBits::make_ubits(width, expected).unwrap()]
                );
            }
        }
    }
}

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
                    let above = order_key(value) >= order_key(bound);
                    let expected = 2 * u64::from(above) + u64::from(value == peak);
                    let inputs = [a, b].map(|v| IrBits::make_ubits(4, v).unwrap());
                    let result = vec![IrBits::make_ubits(2, expected).unwrap()];
                    for graph in [&on, &off] {
                        let sim = gate_sim::eval(graph, &inputs, Collect::Tagged);
                        assert_eq!(sim.outputs, result);
                        // Tags expose the AND node before the output edge's
                        // inversion.
                        let output = graph.outputs[0].bit_vector.get_lsb(1);
                        assert_eq!(
                            sim.tagged_values.unwrap().get(&format!("{cmp}_7")),
                            Some(&(above ^ output.negated))
                        );
                    }
                }
            }
        }
    }
}
