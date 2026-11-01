// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::aig::AigBitVector;
use crate::aig::get_summary_stats::get_output_cone_stats;
use crate::aig_sim::gate_sim::PreparedGateSim;
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::gatify::ir2gate::{
    GatifyOptions, TerminalObservationMode, gatify_with_terminal_observations,
};
use xlsynth_pir::IrBits;
use xlsynth_pir::ir_parser::Parser;

#[test]
fn captures_a_scalar_predicate_even_when_the_word_is_also_returned() {
    let pkg = Parser::new(
        "package p
top fn f(x: bits[3] id=1, sel_bit: bits[1] id=2) -> (bits[1], bits[3]) {
    fill: bits[3] = literal(value=2, id=3)
    word: bits[3] = sel(sel_bit, cases=[fill, x], id=4)
    code: bits[3] = literal(value=5, id=5)
    class: bits[1] = eq(word, code, id=6)
    ret result: (bits[1], bits[3]) = tuple(class, word, id=7)
}",
    )
    .parse_and_validate_package()
    .unwrap();
    let f = pkg.get_top_fn().unwrap();
    let opts = GatifyOptions {
        fold: true,
        hash: true,
        ..GatifyOptions::all_opts_disabled()
    };
    let (out, mut plans) =
        gatify_with_terminal_observations(f, opts, TerminalObservationMode::Incumbent).unwrap();
    assert_eq!(plans.words.len(), 1);
    assert_eq!(plans.words[0].bits.len(), 4);
    let candidate = apply(out.gate_fn, &plans.words, &mut plans.rail_budget);
    let mut sim = PreparedGateSim::new(&candidate);
    for x in 0..8 {
        for select in 0..2 {
            let got = sim.eval_outputs(&[
                IrBits::make_ubits(3, x).unwrap(),
                IrBits::make_ubits(1, select).unwrap(),
            ]);
            let word = if select == 0 { 2 } else { x };
            assert_eq!(
                got,
                vec![IrBits::make_ubits(4, word | (u64::from(word == 5) << 3)).unwrap()]
            );
        }
    }
}

#[test]
fn coalesces_complementary_boundaries_at_every_consumer() {
    let mut gb = GateBuilder::new("fixture".into(), GateBuilderOptions::no_opt());
    let data = gb.add_input("data".into(), 2);
    let outside = gb.add_input("outside".into(), 2);
    let (a, b) = (*data.get_lsb(0), *data.get_lsb(1));
    let both = gb.add_and_binary(a, b);
    let impossible = gb.add_and_binary(a, a.negate());
    let not_both = gb.add_and_binary(both.negate(), impossible.negate());
    let p = gb.add_and_binary(both, *outside.get_lsb(0));
    let q = gb.add_and_binary(not_both, *outside.get_lsb(1));
    gb.add_tag(p.node, "observed-output".into());
    gb.add_output(
        "mixed".into(),
        AigBitVector::from_lsb_is_index_0(&[q, p, not_both]),
    );
    gb.add_output("payload".into(), data);
    let stock = gb.build();
    let bits = (0..2)
        .map(|bit| CutBit {
            port: "data".into(),
            width: 2,
            bit,
        })
        .collect::<Vec<_>>();
    assert!(coalesce_cut_boundary(&stock, &bits[..1], &mut RailBudget::default()).is_none());
    let trial = coalesce_cut_boundary(&stock, &bits, &mut RailBudget::default()).unwrap();
    assert!(get_output_cone_stats(&trial).and_nodes < get_output_cone_stats(&stock).and_nodes);
    assert!(trial.gates.iter().any(|g| {
        g.get_tags()
            .is_some_and(|t| t.iter().any(|n| n == "observed-output"))
    }));
    let mut tagged = stock.clone();
    tagged.gates[impossible.node.id].add_tag("observable-ancestor".into());
    assert!(coalesce_cut_boundary(&tagged, &bits, &mut RailBudget::default()).is_none());
    // Provenance belongs to the surviving signed function, not to a dead
    // gate that computed a different signal within the replaced cone.
    let mut tracked = stock.clone();
    tracked.gates[both.node.id].add_pir_node_id(101);
    tracked.gates[not_both.node.id].add_pir_node_id(202);
    tracked.gates[impossible.node.id].add_pir_node_id(303);
    let merged = coalesce_cut_boundary(&tracked, &bits, &mut RailBudget::default()).unwrap();
    let emit = crate::aig_serdes::emit_aiger_binary::emit_aiger_binary;
    assert_eq!(emit(&trial, true).unwrap(), emit(&merged, true).unwrap());
    assert_eq!(
        merged
            .gates
            .iter()
            .filter(|n| n.get_pir_node_ids() == [101, 202])
            .count(),
        1
    );
    assert!(
        merged
            .gates
            .iter()
            .all(|n| !n.get_pir_node_ids().contains(&303))
    );
    let mut parent = PreparedGateSim::new(&stock);
    let mut child = PreparedGateSim::new(&trial);
    for data in 0..4 {
        for out in 0..4 {
            let args = [
                IrBits::make_ubits(2, data).unwrap(),
                IrBits::make_ubits(2, out).unwrap(),
            ];
            assert_eq!(parent.eval_outputs(&args), child.eval_outputs(&args));
        }
    }
}
