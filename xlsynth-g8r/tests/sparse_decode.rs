// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig::{AigNode, GateFn};
use xlsynth_g8r::aig_sim::gate_sim::{Collect, eval};
use xlsynth_g8r::process_ir_path::{
    CanonicalG8rOptions, canonical_ir_text_to_g8r_lowering_artifacts,
};
use xlsynth_pir::IrBits;

/// Isolate the decoder pass at the canonical lowering entry point.
fn lower(ir: &str, track_pir_node_ids: bool) -> Result<GateFn, String> {
    let opts = CanonicalG8rOptions {
        fold: true,
        hash: true,
        fraig: false,
        reassociation: false,
        cut_db_rewrite: false,
        enable_formal_array_alias_analysis: false,
        track_pir_node_ids,
        ..Default::default()
    };
    Ok(canonical_ir_text_to_g8r_lowering_artifacts(ir, Some("main"), &opts)?.gate_fn)
}

#[test]
fn decoder_preserves_observable_equality_tags() {
    // Tuple construction leaves only the two equality probes to preserve.
    let ir = r#"package tagged
top fn main(key: bits[2] id=1, spare: bits[1] id=2) -> (bits[1], bits[1]) {
  zero: bits[2] = literal(value=0, id=3)
  e0: bits[1] = eq(key, zero, id=4)
  one: bits[2] = literal(value=1, id=5)
  e1: bits[1] = eq(key, one, id=6)
  hit: bits[1] = or(e0, e1, id=7)
  miss: bits[1] = not(hit, id=8)
  ret out: (bits[1], bits[1]) = tuple(hit, miss, id=9)
}
"#;
    let graph = lower(ir, false).unwrap();
    for key in 0..4 {
        let inputs = [
            IrBits::make_ubits(2, key).unwrap(),
            IrBits::make_ubits(1, 0).unwrap(),
        ];
        let tags = eval(&graph, &inputs, Collect::Tagged)
            .tagged_values
            .unwrap();
        assert_eq!(tags.get("eq_4"), Some(&(key == 0)));
        assert_eq!(tags.get("eq_6"), Some(&(key == 1)));
    }
}

#[test]
fn decoder_tags_require_a_positive_and_for_the_observed_value() {
    // The positive node under the OR is key[1] in the first case and !e0
    // in the second. The bank can supply only an input or an inverted AND.
    for (args, first_true_key) in [("e0, e1", 2), ("t, f", 1)] {
        let ir = format!(
            r#"package phase
top fn main(key: bits[2] id=1, s: bits[1] id=2) -> bits[3] {{
  zero: bits[2] = literal(value=0, id=3)
  e0: bits[1] = eq(key, zero, id=4)
  one: bits[2] = literal(value=1, id=5)
  e1: bits[1] = eq(key, one, id=6)
  ns: bits[1] = not(s, id=7)
  t: bits[1] = and(e0, s, id=8)
  f: bits[1] = and(e0, ns, id=9)
  probe: bits[1] = or({args}, id=10)
  ret out: bits[3] = concat(e0, e1, probe, id=11)
}}
"#
        );
        let graph = lower(&ir, false).unwrap();
        for key in 0..4 {
            let inputs = [
                IrBits::make_ubits(2, key).unwrap(),
                IrBits::make_ubits(1, 0).unwrap(),
            ];
            let tags = eval(&graph, &inputs, Collect::Tagged)
                .tagged_values
                .unwrap();
            assert_eq!(
                tags.get("concat_11_output_bit_0"),
                Some(&(key >= first_true_key))
            );
        }
    }
}

#[test]
fn decoder_attributes_retained_and_newly_shared_ands() {
    let ir = r#"package retained
top fn main(key: bits[2] id=1, a: bits[1] id=2, b: bits[1] id=3, c: bits[1] id=4) -> (bits[1], bits[1], bits[1], bits[1], bits[1], bits[1], bits[1]) {
  zero: bits[2] = literal(value=0, id=5)
  e0: bits[1] = eq(key, zero, id=6)
  one: bits[2] = literal(value=1, id=7)
  e1: bits[1] = eq(key, one, id=8)
  hit: bits[1] = or(e0, e1, id=9)
  p: bits[1] = and(a, b, id=10)
  q: bits[1] = and(p, c, id=11)
  na: bits[1] = not(a, id=12)
  nc: bits[1] = not(c, id=13)
  bc: bits[1] = and(b, c, id=14)
  bnc: bits[1] = and(b, nc, id=15)
  r: bits[1] = and(na, bc, id=16)
  s: bits[1] = and(na, bnc, id=17)
  ret out: (bits[1], bits[1], bits[1], bits[1], bits[1], bits[1], bits[1]) = tuple(e0, e1, hit, r, s, p, q, id=18)
}
"#;
    let graph = lower(ir, true).unwrap();
    let output = &graph.outputs[0].bit_vector;
    // Both tagged equalities survive. Their disjunction becomes a wire.
    assert_eq!(get_aig_stats(&graph).and_nodes, 7);
    for (bit, tag) in [(6, "eq_6"), (5, "eq_8")] {
        let tags = graph.get(output.get_lsb(bit).node).get_tags().unwrap();
        assert!(tags.iter().any(|t| t == tag));
    }
    assert_eq!(
        *output.get_lsb(4),
        graph.inputs[0].bit_vector.get_lsb(1).negate()
    );
    assert_eq!(graph.get(output.get_lsb(1).node).get_pir_node_ids(), &[10]);
    assert_eq!(graph.get(output.get_lsb(0).node).get_pir_node_ids(), &[11]);
    // Both r and s use the newly built !a & b.
    let r = graph.get(output.get_lsb(3).node);
    let s = graph.get(output.get_lsb(2).node);
    let shared = r
        .operands()
        .find(|op| s.operands().any(|other| other == *op))
        .unwrap();
    assert_eq!(graph.get(shared.node).get_pir_node_ids(), &[16, 17]);
}

#[test]
fn decoder_preserves_producers_of_complementary_output_roots() {
    let ir = r#"package parity
top fn main(key: bits[2] id=1, a: bits[1] id=2, b: bits[1] id=3, spare: bits[1] id=4) -> (bits[1], bits[1], bits[1], bits[1], bits[1]) {
  zero: bits[2] = literal(value=0, id=5)
  e0: bits[1] = eq(key, zero, id=6)
  one: bits[2] = literal(value=1, id=7)
  e1: bits[1] = eq(key, one, id=8)
  hit: bits[1] = or(e0, e1, id=9)
  na: bits[1] = not(a, id=10)
  nb: bits[1] = not(b, id=11)
  m01: bits[1] = and(na, b, id=12)
  m10: bits[1] = and(a, nb, id=13)
  p: bits[1] = or(m01, m10, id=14)
  m00: bits[1] = and(na, nb, id=15)
  m11: bits[1] = and(a, b, id=16)
  q: bits[1] = or(m00, m11, id=17)
  ret out: (bits[1], bits[1], bits[1], bits[1], bits[1]) = tuple(e0, e1, hit, p, q, id=18)
}
"#;
    let graph = lower(ir, true).unwrap();
    let output = &graph.outputs[0].bit_vector;
    assert_eq!(graph.get(output.get_lsb(1).node).get_pir_node_ids(), &[14]);
    assert_eq!(graph.get(output.get_lsb(0).node).get_pir_node_ids(), &[17]);
    assert_eq!(
        *output.get_lsb(2),
        graph.inputs[0].bit_vector.get_lsb(1).negate()
    );
}

#[test]
fn decoder_preserves_ids_of_unused_interface_bits_and_constant() {
    let ir = r#"package unused
top fn main(key: bits[2] id=1, data: bits[2] id=2, spare: bits[1] id=3) -> (bits[1], bits[1], bits[1]) {
  zero: bits[2] = literal(value=0, id=4)
  e0: bits[1] = eq(key, zero, id=5)
  one: bits[2] = literal(value=1, id=6)
  e1: bits[1] = eq(key, one, id=7)
  low: bits[1] = bit_slice(data, start=0, width=1, id=8)
  high: bits[1] = bit_slice(data, start=1, width=1, id=9)
  nlow: bits[1] = not(low, id=10)
  t: bits[1] = and(low, high, id=11)
  f: bits[1] = and(nlow, high, id=12)
  y: bits[1] = or(t, f, id=13)
  z: bits[1] = literal(value=0, id=14)
  nz: bits[1] = not(z, id=15)
  masked: bits[1] = and(y, nz, id=16)
  ret out: (bits[1], bits[1], bits[1]) = tuple(e0, e1, masked, id=17)
}
"#;
    let graph = lower(ir, true).unwrap();
    // The NOT is recorded on the permanent constant even after the AND folds.
    assert_eq!(graph.gates[0].get_pir_node_ids(), &[15]);
    // Rebuilding y as data[1] removes the last use of data[0]. The spare
    // input was already unused before the rewrite.
    let data = &graph.inputs[1].bit_vector;
    assert_eq!(graph.outputs[0].bit_vector.get_lsb(0), data.get_lsb(1));
    assert!(
        graph
            .get(data.get_lsb(0).node)
            .get_pir_node_ids()
            .contains(&2)
    );
    let spare = graph.inputs[2].bit_vector.get_lsb(0);
    assert_eq!(graph.get(spare.node).get_pir_node_ids(), &[3]);
}

#[test]
fn dead_aggregate_comparisons_do_not_hide_a_live_decoder() {
    let ir = r#"package dead_comparisons
top fn main(key: bits[2] id=1, aggregate: bits[1][2] id=2) -> (bits[1], bits[1], bits[1]) {
  zero: bits[2] = literal(value=0, id=3)
  e0: bits[1] = eq(key, zero, id=4)
  one: bits[2] = literal(value=1, id=5)
  e1: bits[1] = eq(key, one, id=6)
  hit: bits[1] = or(e0, e1, id=7)
  v0: bits[1][2] = literal(value=[0, 0], id=8)
  a0: bits[1] = eq(aggregate, v0, id=9)
  v1: bits[1][2] = literal(value=[1, 0], id=10)
  a1: bits[1] = eq(v1, aggregate, id=11)
  ret out: (bits[1], bits[1], bits[1]) = tuple(e0, e1, hit, id=12)
}
"#;
    let graph = lower(ir, false).unwrap();
    let output = &graph.outputs[0].bit_vector;
    assert_eq!(get_aig_stats(&graph).and_nodes, 2);
    assert_eq!(
        *output.get_lsb(0),
        graph.inputs[0].bit_vector.get_lsb(1).negate()
    );
    for (bit, tag) in [(2, "eq_4"), (1, "eq_6")] {
        let tags = graph.get(output.get_lsb(bit).node).get_tags().unwrap();
        assert!(tags.iter().any(|t| t == tag));
    }
}

#[test]
fn distinct_aggregate_literals_identify_a_second_decoder_key() {
    for (ty, zero, one) in [
        ("bits[1][2]", "[0, 0]", "[1, 0]"),
        ("(bits[1], bits[1])", "(0, 0)", "(1, 0)"),
    ] {
        for other in [zero, one] {
            let ir = format!(
                r#"package two_keys
top fn main(key: bits[2] id=1, aggregate: {ty} id=2) -> (bits[1], bits[1], bits[1], bits[1], bits[1], bits[1]) {{
  zero: bits[2] = literal(value=0, id=3)
  e0: bits[1] = eq(key, zero, id=4)
  one: bits[2] = literal(value=1, id=5)
  e1: bits[1] = eq(key, one, id=6)
  hit: bits[1] = or(e0, e1, id=7)
  v0: {ty} = literal(value={zero}, id=8)
  a0: bits[1] = eq(aggregate, v0, id=9)
  v1: {ty} = literal(value={other}, id=10)
  a1: bits[1] = eq(v1, aggregate, id=11)
  any: bits[1] = or(a0, a1, id=12)
  ret out: (bits[1], bits[1], bits[1], bits[1], bits[1], bits[1]) = tuple(e0, e1, a0, a1, any, hit, id=13)
}}
"#
            );
            let graph = lower(&ir, false).unwrap();
            // All tagged equality signals are outputs, so they can survive
            // either mapping. Only a second distinct key disallows the bank.
            let hit = *graph.outputs[0].bit_vector.get_lsb(0);
            let wire = graph.inputs[0].bit_vector.get_lsb(1).negate();
            assert_eq!(hit == wire, other == zero, "{ty} with {other}");
        }
    }
}

#[test]
fn decoder_uses_tagged_equalities_in_a_complemented_output_cover() {
    // The two addresses simplify to a two-bit comparison. Keeping them as
    // separate terms must still allow the independent data cone to collapse.
    let ir = r#"package covered
top fn main(key: bits[3] id=1, flag: bits[1] id=2, sample: bits[1] id=3) -> (bits[1], bits[1]) {
  zero: bits[3] = literal(value=0, id=4)
  one: bits[3] = literal(value=1, id=5)
  e0: bits[1] = eq(key, zero, id=6)
  e1: bits[1] = eq(key, one, id=7)
  hit: bits[1] = or(e0, e1, id=8)
  miss: bits[1] = not(hit, id=9)
  ns: bits[1] = not(sample, id=10)
  left: bits[1] = and(flag, sample, id=11)
  right: bits[1] = and(flag, ns, id=12)
  data: bits[1] = or(left, right, id=13)
  ret out: (bits[1], bits[1]) = tuple(miss, data, id=14)
}
"#;
    let graph = lower(ir, true).unwrap();
    assert_eq!(
        graph.outputs[0].bit_vector.get_lsb(0),
        graph.inputs[1].bit_vector.get_lsb(0)
    );
    for id in [6, 7] {
        let name = format!("eq_{id}");
        let node = graph
            .post_order_refs()
            .into_iter()
            .map(|r| graph.get(r))
            .find(|n| n.get_tags().is_some_and(|tags| tags.contains(&name)))
            .unwrap();
        assert_eq!(node.get_pir_node_ids(), &[id]);
    }
    for key in 0..8 {
        for flag in 0..2 {
            for sample in 0..2 {
                let inputs = [
                    IrBits::make_ubits(3, key).unwrap(),
                    IrBits::make_ubits(1, flag).unwrap(),
                    IrBits::make_ubits(1, sample).unwrap(),
                ];
                let result = eval(&graph, &inputs, Collect::Tagged);
                assert_eq!(
                    result.outputs,
                    [IrBits::make_ubits(2, 2 * u64::from(key >= 2) + flag).unwrap()]
                );
                let tags = result.tagged_values.unwrap();
                assert_eq!(tags["eq_6"], key == 0);
                assert_eq!(tags["eq_7"], key == 1);
            }
        }
    }
}

/// Check every ordered output and named comparison across the small input
/// domain.
fn check_decoder_table(
    graph: &GateFn,
    width: usize,
    tag_keys: &[(&str, u64)],
    outputs: impl Fn(u64, bool) -> Vec<bool>,
) {
    for key in 0..1 << width {
        for gate in [false, true] {
            let inputs = [
                IrBits::make_ubits(width, key).unwrap(),
                IrBits::make_ubits(1, u64::from(gate)).unwrap(),
            ];
            let result = eval(graph, &inputs, Collect::Tagged);
            let bits = outputs(key, gate);
            let packed = bits.iter().fold(0, |n, &b| n * 2 + u64::from(b));
            assert_eq!(
                result.outputs,
                [IrBits::make_ubits(bits.len(), packed).unwrap()]
            );
            let tags = result.tagged_values.unwrap();
            for &(tag, address) in tag_keys {
                assert_eq!(tags.get(tag), Some(&(key == address)), "{tag} at key {key}");
            }
        }
    }
}

#[test]
fn decoder_distributes_a_key_bit_to_reuse_a_named_hit() {
    // The guarded pair needs both observable comparisons. The independent
    // strobe can reuse the complete hit by distributing over ready | grant.
    let ir = r#"package lane
top fn main(key: bits[3] id=1, grant: bits[1] id=2) -> (bits[1], bits[1], bits[1]) {
  hot: bits[3] = literal(value=7, id=3)
  hit: bits[1] = eq(key, hot, id=4)
  strobe: bits[1] = bit_slice(key, start=2, width=1, id=5)
  low: bits[1] = bit_slice(key, start=0, width=1, id=6)
  mid: bits[1] = bit_slice(key, start=1, width=1, id=7)
  ready: bits[1] = and(low, mid, id=8)
  ok: bits[1] = or(ready, grant, id=9)
  take: bits[1] = and(strobe, ok, id=10)
  first: bits[3] = literal(value=0, id=11)
  second: bits[3] = literal(value=1, id=12)
  e0: bits[1] = eq(first, key, id=13)
  e1: bits[1] = eq(second, key, id=14)
  pair: bits[1] = or(e0, e1, id=15)
  guarded: bits[1] = and(pair, grant, id=16)
  ret out: (bits[1], bits[1], bits[1]) = tuple(hit, take, guarded, id=17)
}
"#;
    let graph = lower(ir, true).unwrap();
    let live_ands = get_aig_stats(&graph).and_nodes;
    assert_eq!(live_ands, 9);
    // UGV emits every stored AND, including any bypassed by distribution.
    let stored_ands = graph
        .gates
        .iter()
        .filter(|n| matches!(n, AigNode::And2 { .. }))
        .count();
    assert_eq!(stored_ands, live_ands);
    let output = &graph.outputs[0].bit_vector;
    let hit = *output.get_lsb(2);
    let take = *output.get_lsb(1);
    assert!(take.negated);
    assert!(graph.get(take.node).operands().any(|e| e == hit.negate()));
    check_decoder_table(
        &graph,
        3,
        &[("eq_4", 7), ("eq_13", 0), ("eq_14", 1)],
        |key, grant| vec![key == 7, key == 7 || (key >= 4 && grant), key < 2 && grant],
    );
}

#[test]
fn decoder_reuses_both_signed_partial_comparisons() {
    // Both consumers of high & gate can use their own decoded address.
    let ir = r#"package shared
top fn main(key: bits[2] id=1, gate: bits[1] id=2) -> (bits[1], bits[1], bits[1], bits[1], bits[1]) {
  odd: bits[1] = bit_slice(key, start=0, width=1, id=3)
  high: bits[1] = bit_slice(key, start=1, width=1, id=4)
  even: bits[1] = not(odd, id=5)
  last: bits[2] = literal(value=3, id=6)
  prev: bits[2] = literal(value=2, id=7)
  p: bits[1] = eq(key, last, id=8)
  q: bits[1] = eq(key, prev, id=9)
  open: bits[1] = and(high, gate, id=10)
  left: bits[1] = and(open, odd, id=11)
  right: bits[1] = and(even, open, id=12)
  first: bits[2] = literal(value=0, id=13)
  second: bits[2] = literal(value=1, id=14)
  e0: bits[1] = eq(first, key, id=15)
  e1: bits[1] = eq(second, key, id=16)
  pair: bits[1] = or(e0, e1, id=17)
  guarded: bits[1] = and(pair, gate, id=18)
  ret out: (bits[1], bits[1], bits[1], bits[1], bits[1]) = tuple(p, q, left, right, guarded, id=19)
}
"#;
    let graph = lower(ir, true).unwrap();
    assert_eq!(get_aig_stats(&graph).and_nodes, 8);
    let output = &graph.outputs[0].bit_vector;
    // The AND under pair's complemented edge is equivalent to key[1].
    // It still computes pair for guarded and must keep that producer.
    let guarded = graph.get(output.get_lsb(0).node);
    let pair = guarded
        .operands()
        .find(|e| *e != *graph.inputs[1].bit_vector.get_lsb(0))
        .unwrap();
    assert_eq!(graph.get(pair.node).get_pir_node_ids(), &[17]);
    let high = graph.inputs[0].bit_vector.get_lsb(1);
    assert!(!graph.get(high.node).get_pir_node_ids().contains(&17));
    for (gated, decoded) in [(2, 4), (1, 3)] {
        assert!(
            graph
                .get(output.get_lsb(gated).node)
                .operands()
                .any(|e| e == *output.get_lsb(decoded))
        );
    }
    check_decoder_table(
        &graph,
        2,
        &[("eq_8", 3), ("eq_9", 2), ("eq_15", 0), ("eq_16", 1)],
        |key, gate| {
            vec![
                key == 3,
                key == 2,
                key == 3 && gate,
                key == 2 && gate,
                key < 2 && gate,
            ]
        },
    );
}
