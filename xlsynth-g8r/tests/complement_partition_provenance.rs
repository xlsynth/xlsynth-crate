// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use xlsynth_g8r::aig::gate::AigNode;
use xlsynth_g8r::aig::table::build_area_table_report;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify, gatify_prepared_fn};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_g8r::ir_aig_sharing::{
    CandidateProofResult, IrAigCandidateRhs, IrAigEquivalenceCandidate,
    prove_equivalence_candidates_with_backend_streaming,
};
use xlsynth_g8r::prove_gate_fn_equiv_common::GateFormalBackend;
use xlsynth_pir::ir::{Binop, NodePayload, Unop};
use xlsynth_pir::{IrBits, ir_parser::Parser, ir_range_info::IrRangeInfo, ir_utils, known_bits};

const CONSTANT_DISJUNCTION: &str = r#"fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  s: bits[1] = or(a, b, id=3)
  na: bits[1] = not(a, id=4)
  nb: bits[1] = not(b, id=5)
  r: bits[1] = and(na, nb, id=6)
  ret result: bits[1] = or(s, r, id=7)
}"#;

#[test]
fn only_the_disjunction_can_be_proved_constant() {
    let f = Parser::new(CONSTANT_DISJUNCTION).parse_fn().unwrap();
    let opts = GatifyOptions {
        track_pir_node_ids: true,
        ..GatifyOptions::all_opts_disabled()
    };
    let lowered = gatify(&f, opts.clone()).unwrap();
    let original_term = ir_utils::find_node_by_name(&f, "r").unwrap();
    let candidates = [original_term, f.ret_node_ref.unwrap()].map(|r| IrAigEquivalenceCandidate {
        pir_node_ref: r,
        pir_node_text_id: f.get_node(r).text_id,
        bit_index: 0,
        rhs: IrAigCandidateRhs::Const(true),
    });
    let proofs = prove_equivalence_candidates_with_backend_streaming(
        &f,
        &lowered.gate_fn,
        &candidates,
        &opts,
        GateFormalBackend::Cadical,
        |_| {},
    )
    .unwrap();
    assert!(matches!(
        proofs[0].result,
        CandidateProofResult::Skipped { .. }
    ));
    assert!(matches!(proofs[1].result, CandidateProofResult::Proved));
    assert!(!lowered.lowering_map.contains_key(&original_term));
    assert!(
        lowered
            .gate_fn
            .gates
            .iter()
            .all(|g| !g.get_pir_node_ids().contains(&6))
    );
}

#[test]
fn widened_literal_does_not_inherit_a_known_zero() {
    let f = Parser::new(
        r#"fn f(b: bits[1] id=1) -> bits[1] {
  one: bits[1] = literal(value=1, id=2)
  s: bits[1] = or(one, b, id=3)
  na: bits[1] = not(one, id=4)
  nb: bits[1] = not(b, id=5)
  r: bits[1] = and(na, nb, id=6)
  ret result: bits[1] = or(s, r, id=7)
}"#,
    )
    .parse_fn()
    .unwrap();
    let analysis = known_bits::analyze_fn(&f).unwrap();
    let ranges = IrRangeInfo::from_known_bits(&analysis);
    let known = ranges.get(6).unwrap().known_bits.as_ref().unwrap();
    assert!(known.is_fully_known() && known.value().is_zero());

    // The enclosing OR is always true, so inspecting only the output would
    // hide a stale zero applied to the widened term.
    let prepared = prep_for_gatify(&f, Some(&ranges), PrepForGatifyOptions::all_opts_disabled());
    let widened = ir_utils::find_node_by_name(&prepared, "r").unwrap();
    let lowered = gatify_prepared_fn(
        &prepared,
        GatifyOptions {
            range_info: Some(Arc::new(ranges)),
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap();
    let bit = lowered.lowering_map[&widened].get_lsb(0);
    let AigNode::Literal { value, .. } = lowered.gate_fn.gates[bit.node.id] else {
        panic!("expected a constant for the widened term");
    };
    assert!(value ^ bit.negated);
}

#[test]
fn absorption_retires_newly_projected_mask_values() {
    let f = Parser::new(
        r#"fn f(x: bits[2] id=1, b: bits[1] id=2) -> (bits[1], bits[1]) {
  zero: bits[2] = literal(value=0, id=3)
  c: bits[1] = sle(x, zero, id=4)
  nc: bits[1] = not(c, id=5)
  m: bits[1] = sign_ext(nc, new_bit_count=1, id=6)
  lo: bits[1] = bit_slice(x, start=0, width=1, id=7)
  masked: bits[1] = and(lo, m, id=8)
  p: bits[1] = bit_slice(masked, start=0, width=1, id=9)
  np: bits[1] = not(p, id=10)
  s: bits[1] = or(np, b, id=11)
  nb: bits[1] = not(b, id=12)
  r: bits[1] = and(lo, m, nb, id=13)
  o: bits[1] = or(s, r, id=14)
  ret result: (bits[1], bits[1]) = tuple(o, p, id=15)
}"#,
    )
    .parse_fn()
    .unwrap();
    let lowered = gatify(&f, GatifyOptions::all_opts_disabled()).unwrap();
    // Absorbing r leaves m with one use. The sign test can now replace c,
    // but x == 0 distinguishes each changed intermediate from its old value.
    for name in ["c", "nc", "m"] {
        let original = ir_utils::find_node_by_name(&f, name).unwrap();
        assert!(!lowered.lowering_map.contains_key(&original), "{name}");
    }
    let mut sim = PreparedGateSim::new(&lowered.gate_fn);
    for x in 0..4 {
        for b in [false, true] {
            // The tuple (1, x == 1) is flattened with p in the low bit.
            assert_eq!(
                sim.eval_outputs(&[IrBits::make_ubits(2, x).unwrap(), IrBits::bool(b)]),
                [IrBits::make_ubits(2, 2 + u64::from(x == 1)).unwrap()]
            );
        }
    }
}

#[test]
fn retained_and_is_attributed_to_its_enclosing_or() {
    for guards in ["g, q", "q, g"] {
        let text = format!(
            r#"fn f(g: bits[1] id=1, q: bits[1] id=2, a: bits[1] id=3, b: bits[1] id=4) -> bits[1] {{
  s: bits[1] = or(a, b, id=5)
  left: bits[1] = and(g, s, id=6)
  na: bits[1] = not(a, id=7)
  nb: bits[1] = not(b, id=8)
  right: bits[1] = and({guards}, na, nb, id=9)
  ret result: bits[1] = or(left, right, id=10)
}}"#
        );
        let f = Parser::new(&text).parse_fn().unwrap();
        let lowered = gatify(&f, tracked_unfolded()).unwrap();
        let area = build_area_table_report(&lowered.gate_fn, &f).unwrap();
        assert_eq!(
            area.rows
                .iter()
                .map(|row| (row.pir_node_id, row.weighted_aig_node_count))
                .collect::<Vec<_>>(),
            [(5, 1.0), (6, 1.0), (10, 2.0)]
        );
        let right = ir_utils::find_node_by_name(&f, "right").unwrap();
        assert!(!lowered.lowering_map.contains_key(&right));
    }
}

#[test]
fn widened_term_ids_fit_prepared_provenance() {
    for (result_id, term_id) in [(u32::MAX - 1, u32::MAX), (u32::MAX, 6)] {
        let mut f = Parser::new(CONSTANT_DISJUNCTION).parse_fn().unwrap();
        let result = f.ret_node_ref.unwrap();
        f.get_node_mut(result).text_id = result_id as usize;
        let prepared = prep_for_gatify(&f, None, PrepForGatifyOptions::all_opts_disabled());
        gatify_prepared_fn(
            &prepared,
            GatifyOptions {
                track_pir_node_ids: true,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();
        let term = ir_utils::find_node_by_name(&prepared, "r").unwrap();
        assert_eq!(prepared.get_node(term).text_id, term_id as usize);
    }
}

/// The widened right term's midpoint separates (nz, z) from the guards.
fn masked_partition(guards: &str, result_id: u32) -> String {
    format!(
        r#"fn f(g: bits[1] id=1, q: bits[1] id=2, a: bits[1] id=3, b: bits[1] id=4, z: bits[1] id=5) -> bits[1] {{
  s: bits[1] = or(a, b, id=6)
  left: bits[1] = and(g, s, id=7)
  na: bits[1] = not(a, id=8)
  nb: bits[1] = not(b, id=9)
  c: bits[1] = and(na, nb, id=10)
  nz: bits[1] = or_reduce(z, id=11)
  r: bits[1] = and(nz, z, {guards}, c, id=12)
  ret result: bits[1] = or(left, r, id={result_id})
}}"#
    )
}

/// Keep the retained AND gates visible in the per-node area report.
fn tracked_unfolded() -> GatifyOptions {
    GatifyOptions {
        fold: false,
        hash: false,
        track_pir_node_ids: true,
        ..GatifyOptions::all_opts_disabled()
    }
}

#[test]
fn pruned_subgroup_inherits_the_enclosing_or_as_its_owner() {
    for (guards, term_name) in [("g, q", "r"), ("q, g", "r"), ("g, q", "and.12")] {
        let mut f = Parser::new(&masked_partition(guards, 13))
            .parse_fn()
            .unwrap();
        let term = ir_utils::find_node_by_name(&f, "r").unwrap();
        // In-memory PIR can carry an explicit opcode.ID name.
        f.get_node_mut(term).name = Some(term_name.into());
        let lowered = gatify(&f, tracked_unfolded()).unwrap();
        let prepared = prep_for_gatify(&f, None, PrepForGatifyOptions::all_opts_disabled());
        gatify_prepared_fn(&prepared, tracked_unfolded()).unwrap();
        let area = build_area_table_report(&lowered.gate_fn, &f).unwrap();
        // The result owns and(g, q), its AND with z, and the final OR.
        assert_eq!(
            area.rows
                .iter()
                .map(|row| (row.pir_node_id, row.weighted_aig_node_count))
                .collect::<Vec<_>>(),
            [(6, 1.0), (7, 1.0), (13, 3.0)],
            "{guards}, {term_name}"
        );
        assert!(!lowered.lowering_map.contains_key(&term));
    }
}

#[test]
fn pruning_a_widened_term_needs_no_additional_id() {
    for remaining in [1, 2] {
        let result_id = u32::MAX - remaining;
        let f = Parser::new(&masked_partition("g, q", result_id))
            .parse_fn()
            .unwrap();
        let prepared = prep_for_gatify(&f, None, PrepForGatifyOptions::all_opts_disabled());
        let term = ir_utils::find_node_by_name(&prepared, "r").unwrap();
        assert_eq!(prepared.get_node(term).text_id, (result_id + 1) as usize);
        // The surviving halves [z] and [g, q] fit a single three-input AND.
        assert!(ir_utils::find_node_by_name(&prepared, "nz").is_none());
        let lowered = gatify(&f, tracked_unfolded()).unwrap();
        let standalone = gatify_prepared_fn(&prepared, tracked_unfolded()).unwrap();
        for gates in [lowered.gate_fn, standalone.gate_fn] {
            let mut sim = PreparedGateSim::new(&gates);
            for bits in 0..32 {
                let [g, q, a, b, z] = [0, 1, 2, 3, 4].map(|i| (bits >> i) & 1 != 0);
                let inputs = [g, q, a, b, z].map(IrBits::bool);
                assert_eq!(
                    sim.eval_outputs(&inputs),
                    [IrBits::bool(g && (a || b || (z && q)))]
                );
            }
        }
    }
}

#[test]
fn comparison_helpers_after_absorption_fit_the_trackable_id_range() {
    for result_id in [10, u32::MAX - 17, u32::MAX - 16] {
        let text = format!(
            r#"fn f(x: bits[2] id=1, y: bits[2] id=2) -> bits[1] {{
  a: bits[1] = ult(x, y, id=3)
  b: bits[1] = slt(x, y, id=4)
  e: bits[1] = eq(x, y, id=5)
  s: bits[1] = or(a, b, id=6)
  na: bits[1] = not(a, id=7)
  nb: bits[1] = not(b, id=8)
  r: bits[1] = and(na, nb, e, id=9)
  ret result: bits[1] = or(s, r, id={result_id})
}}"#
        );
        let f = Parser::new(&text).parse_fn().unwrap();
        let prepared = prep_for_gatify(&f, None, PrepForGatifyOptions::all_opts_disabled());
        let lowered = gatify(&f, tracked_unfolded()).unwrap();
        let standalone = gatify_prepared_fn(&prepared, tracked_unfolded()).unwrap();

        // Absorption uses one ID. The equality helper is the last of the 16
        // allocated when the entire comparison group fits.
        let term = ir_utils::find_node_by_name(&prepared, "r").unwrap();
        assert_eq!(prepared.get_node(term).text_id, (result_id + 1) as usize);
        let eq = ir_utils::find_node_by_name(&prepared, "e").unwrap();
        let helper_id = match prepared.get_node(eq).payload {
            NodePayload::Unop(Unop::Identity, helper) => Some(prepared.get_node(helper).text_id),
            NodePayload::Binop(Binop::Eq, ..) => None,
            ref payload => panic!("unexpected prepared equality: {payload:?}"),
        };
        assert_eq!(helper_id, result_id.checked_add(17).map(|id| id as usize));
        for gates in [lowered.gate_fn, standalone.gate_fn] {
            let mut sim = PreparedGateSim::new(&gates);
            for x in 0..4 {
                for y in 0..4 {
                    // Flipping the sign bit puts two-bit words in signed order.
                    let expected = x <= y || (x ^ 2) < (y ^ 2);
                    let inputs = [x, y].map(|v| IrBits::make_ubits(2, v).unwrap());
                    assert_eq!(sim.eval_outputs(&inputs), [IrBits::bool(expected)]);
                }
            }
        }
    }
}
