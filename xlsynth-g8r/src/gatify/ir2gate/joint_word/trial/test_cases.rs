// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::mcmc_logic::oracle_equiv_sat;

#[test]
fn word_choice_budget_keeps_completed_candidates_and_discards_the_unpriced_form() {
    let (stock, group) = fixture(false);
    let mut full = Stats::default();
    let complete = generate(&stock, &group, Admission::Diagnostic, &mut full, usize::MAX).unwrap();
    let mut partial = Stats::default();
    let prefix = generate(
        &stock,
        &group,
        Admission::Diagnostic,
        &mut partial,
        full.constructed_ands - 1,
    )
    .unwrap();
    assert!(partial.construction_limited);
    assert!(!prefix.is_empty() && partial.reports.len() < full.reports.len());
    for (offer, original) in prefix.iter().zip(&complete) {
        assert_eq!(offer.report, original.report);
        assert_eq!(offer.graph.to_string(), original.graph.to_string());
        assert!(oracle_equiv_sat(&stock, &offer.graph).unwrap());
    }
    let mut zero = Stats::default();
    assert!(
        generate(&stock, &group, Admission::Diagnostic, &mut zero, 0)
            .unwrap()
            .is_empty()
    );
    assert!(zero.construction_limited && zero.reports.is_empty());
}

#[test]
fn complete_signed_return_and_escape_for_every_distinct_encoding() {
    let (stock, group) = fixture(false);
    let mut stats = Stats::default();
    let options = alternatives(&stock, &group, Admission::Diagnostic, &mut stats);
    assert!(stats.declined.is_none() && !options.is_empty());
    assert!(stats.reports.iter().any(|r| r.verdict == Verdict::Eligible));
    assert!(
        stats
            .reports
            .iter()
            .any(|r| r.form == Form::Active && r.route == Route::Direct)
    );
    for axis in 0..group.coords.len() {
        assert!(
            stats
                .reports
                .iter()
                .any(|r| r.route == Route::Shannon(axis))
        );
    }
    let last = stock.outputs.last().unwrap();
    for candidate in options {
        let g = &candidate.graph;
        assert!(oracle_equiv_sat(&stock, g).unwrap());
        assert!(
            g.outputs
                .last()
                .unwrap()
                .bit_vector
                .iter_lsb_to_msb()
                .eq(last.bit_vector.iter_lsb_to_msb())
        );
        let pins: Vec<_> = g
            .outputs
            .iter()
            .flat_map(|o| o.bit_vector.iter_lsb_to_msb().copied())
            .collect();
        let measured = get_aig_cone_stats(&g.gates, &pins);
        let report = &stats.reports[candidate.report];
        assert_eq!(measured.and_nodes, report.and_nodes);
        assert_eq!(measured.root_depths, report.depths);
    }
    eprintln!(
        "suite=complete stock={} price={} elapsed_ns={} scalar_queries={} cubetests={} appended={} distinct={}",
        stats.stock.unwrap().and_nodes,
        stats.prices,
        stats.elapsed.as_nanos(),
        stats.scalar_queries,
        stats.cube_candidates,
        stats.constructed_ands,
        stats
            .reports
            .iter()
            .filter(|r| !matches!(r.verdict, Verdict::Duplicate(_) | Verdict::Unchanged))
            .count()
    );
}

#[test]
fn eligible_pin_subset_preserves_fast_incumbent_and_other_clients() {
    let (stock, group) = fixture(true);
    let mut stats = Stats::default();
    let diagnostic = alternatives(&stock, &group, Admission::Diagnostic, &mut stats);
    assert!(
        diagnostic
            .iter()
            .any(|c| !stats.reports[c.report].late.is_empty())
    );
    assert!(diagnostic.iter().any(|c| {
        let r = &stats.reports[c.report];
        r.keep_late && r.route == Route::Direct && r.verdict == Verdict::Eligible
    }));
    let safe = alternatives(&stock, &group, Admission::SafeOnly, &mut stats);
    assert!(!safe.is_empty());
    for candidate in safe {
        let r = &stats.reports[candidate.report];
        assert_eq!(r.verdict, Verdict::Eligible);
        assert!(r.late.is_empty() && r.and_nodes < stats.stock.as_ref().unwrap().and_nodes);
        assert!(oracle_equiv_sat(&stock, &candidate.graph).unwrap());
    }
}

#[test]
fn missing_values_repeated_slots_and_unbounded_cuts_decline() {
    for kind in 0..6 {
        let (stock, mut group) = fixture(false);
        match kind {
            0 => {
                group.words[1].table.pop();
            }
            1 => {
                group.words[0].lanes[0].output = group.words[1].lanes[0].output;
            }
            2 => {
                group.coords[1] = group.coords[0].negate();
            }
            3 => {
                group.words[0].lanes[0].x = AigRef {
                    id: stock.gates.len(),
                }
                .into();
            }
            4 => {
                group.coords.resize(7, group.coords[0]);
            }
            _ => {
                group.words[1].lanes[0].bit = 27;
            }
        }
        let before = crate::aig_serdes::emit_aiger_binary::emit_aiger_binary(&stock, true).unwrap();
        let mut stats = Stats::default();
        assert!(alternatives(&stock, &group, Admission::Diagnostic, &mut stats).is_empty());
        assert!(stats.declined.is_some());
        assert_eq!(
            before,
            crate::aig_serdes::emit_aiger_binary::emit_aiger_binary(&stock, true).unwrap()
        );
    }
}

#[test]
fn six_control_cared_functions_are_exact_on_every_cared_row() {
    let mut g = GateBuilder::new("bounded".into(), GateBuilderOptions::no_opt());
    let xs: Vec<_> = (0..6)
        .map(|i| *g.add_input(format!("x{i}"), 1).get_lsb(0))
        .collect();
    let known = g.add_and_binary(xs[1].negate(), xs[2]);
    g.add_output("retained".into(), known.into());
    let mut arena = Arena::new(&g.build()).unwrap();
    let mut solver = Solver::new(&arena, &xs);
    let mut stats = Stats::default();
    let mut state = 0x0a81e5937537fb36_u64;
    let mut roots = Vec::new();
    for _ in 0..80 {
        state = state.wrapping_mul(0x9e3779b97f4a7c15).rotate_left(7);
        let care = state ^ (state >> 23);
        state = state.wrapping_add(0x576ac5b4ee4ca037).rotate_left(17);
        let on = state & care;
        let r = solver.boolean(&mut arena, on, care, 0x3f, &mut stats);
        roots.push((r, on, care));
    }
    let mut tt = vec![0u64; arena.g.gates.len()];
    for (i, p) in xs.iter().enumerate() {
        tt[p.node.id] = axis(6, i);
    }
    let eval = |p: AigOperand, tt: &[u64]| {
        if p.negated {
            !tt[p.node.id]
        } else {
            tt[p.node.id]
        }
    };
    for r in topo_sort_refs(&arena.g.gates) {
        match arena.g.gates[r.id] {
            AigNode::And2 { a, b, .. } => tt[r.id] = eval(a, &tt) & eval(b, &tt),
            AigNode::Literal { value, .. } => tt[r.id] = if value { u64::MAX } else { 0 },
            _ => {}
        }
    }
    for (p, on, care) in roots {
        assert_eq!(eval(p, &tt) & care, on);
    }
    assert!(stats.cube_candidates > 0);
}

#[test]
fn existing_signed_control_grants_are_reused_across_words() {
    let mut g = GateBuilder::new("shared".into(), GateBuilderOptions::no_opt());
    let xs: Vec<_> = (0..3)
        .map(|i| *g.add_input(format!("axis{i}"), 1).get_lsb(0))
        .collect();
    let choose = g.add_and_binary(xs[1].negate(), xs[2]);
    g.add_output("caller".into(), choose.negate().into());
    let mut a = Arena::new(&g.build()).unwrap();
    let mut solver = Solver::new(&a, &xs);
    let valid = axis(3, 0);
    let x = valid & !axis(3, 1) & axis(3, 2);
    let mut stats = Stats::default();
    let xgrant = solver.grants(&mut a, [x, valid ^ x], Form::Active, &mut stats);
    let expected = a.and(xs[0], choose);
    assert_eq!(xgrant[0], expected);
    assert_eq!(xgrant[1], a.and(xs[0], choose.negate()));
    assert!(stats.reusable_hits > 0);
    let other = valid & !axis(3, 1);
    let othergrant = solver.grants(&mut a, [other, valid ^ other], Form::Active, &mut stats);
    assert_eq!(
        othergrant,
        [a.and(xs[0], xs[1].negate()), a.and(xs[0], xs[1])]
    );
}

// The reference renderer independently expands the full table to nested muxes.
// The group has signed controls, mixed constants, complemented and escaped
// outputs.
fn fixture(fast: bool) -> (GateFn, Group) {
    let mut g = GateBuilder::new("two_returns".into(), GateBuilderOptions::no_opt());
    let ns: Vec<_> = (0..4)
        .map(|i| *g.add_input(format!("c{i}"), 1).get_lsb(0))
        .collect();
    let expensive = g.add_and_binary(ns[2], ns[3].negate());
    let coords = vec![ns[0].negate(), ns[1], expensive];
    let zero: AigOperand = AigRef { id: 0 }.into();
    let mut words = Vec::new();
    let mut escaped = zero;
    for w in 0..2 {
        let table: Vec<_> = (0..8)
            .map(|v| {
                if (!fast || w == 0) && v & 1 == 0 {
                    Label::Constant
                } else if v & 2 == 0 && (w == 1 || v & 4 != 0) {
                    Label::X
                } else {
                    Label::Y
                }
            })
            .collect();
        let mut lanes = Vec::new();
        for i in 0..4 {
            let x = *g.add_input(format!("x_{w}_{i}"), 1).get_lsb(0);
            let only_wire = fast && w == 1 && i == 0;
            let y = if only_wire {
                x
            } else {
                *g.add_input(format!("y_{w}_{i}"), 1).get_lsb(0)
            };
            let constant = i % 2 == 1;
            let mut level: Vec<_> = table
                .iter()
                .map(|v| match *v {
                    Label::X => x,
                    Label::Y => y,
                    Label::Constant => phase(zero, constant),
                })
                .collect();
            for &s in &coords {
                level = level
                    .chunks_exact(2)
                    .map(|p| {
                        if p[0] == p[1] {
                            return p[0];
                        }
                        let lo = g.add_and_binary(s.negate(), p[0]);
                        let hi = g.add_and_binary(s, p[1]);
                        g.add_and_binary(lo.negate(), hi.negate()).negate()
                    })
                    .collect();
            }
            let f = level[0];
            if w == 0 && i == 0 {
                escaped = f;
            }
            let output = w * 4 + i;
            g.add_output(format!("o{w}_{i}"), phase(f, i % 3 == 1).into());
            lanes.push(Lane {
                output,
                bit: 0,
                selected: f,
                x,
                y,
                constant,
            });
        }
        words.push(Word { table, lanes });
    }
    g.add_output("escape".into(), escaped.negate().into());
    (g.build(), Group { coords, words })
}

#[test]
fn packed_slots_share_ordinary_data_and_preserve_an_optimal_incumbent() {
    use crate::aig::AigBitVector;
    let mut b = GateBuilder::new("packed".into(), GateBuilderOptions::opt());
    let s = *b.add_input("control".into(), 1).get_lsb(0);
    let xs = b.add_input("left".into(), 3);
    let ys = b.add_input("right".into(), 3);
    let mut words = Vec::new();
    for output in 0..2 {
        let mut bits = AigBitVector::zeros(3);
        let mut lanes = Vec::new();
        for bit in 0..3 {
            let x = *xs.get_lsb(bit);
            let y = *ys.get_lsb(bit);
            let on = if output == 0 { x } else { y };
            let off = if output == 0 { y } else { x };
            let hi = b.add_and_binary(s, on);
            let lo = b.add_and_binary(s.negate(), off);
            let selected = b.add_and_binary(hi.negate(), lo.negate()).negate();
            bits.set_lsb(bit, phase(selected, bit == 1));
            lanes.push(Lane {
                output,
                bit,
                selected,
                x,
                y,
                constant: false,
            });
        }
        b.add_output(format!("out{output}"), bits);
        words.push(Word {
            lanes,
            table: if output == 0 {
                vec![Label::Y, Label::X]
            } else {
                vec![Label::X, Label::Y]
            },
        });
    }
    let stock = b.build();
    let group = Group {
        coords: vec![s],
        words,
    };
    let mut stats = Stats::default();
    let values = alternatives(&stock, &group, Admission::Diagnostic, &mut stats);
    assert!(stats.declined.is_none());
    assert!(stats.reports.iter().all(|r| r.depths.len() == 6));
    assert!(
        stats
            .reports
            .iter()
            .any(|r| r.verdict == Verdict::Unchanged)
    );
    assert!(stats.reports.iter().all(|r| r.verdict != Verdict::Eligible));
    for c in values {
        assert!(oracle_equiv_sat(&stock, &c.graph).unwrap());
    }
    assert!(alternatives(&stock, &group, Admission::SafeOnly, &mut stats).is_empty());
}
