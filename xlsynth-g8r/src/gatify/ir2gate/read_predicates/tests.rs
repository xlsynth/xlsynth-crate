// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::aig::get_summary_stats::{get_aig_cone_stats, get_aig_stats};
use crate::aig::table::build_area_table_report;
use crate::aig::topo::extract_cone;
use crate::aig_sim::gate_sim::{Collect, PreparedGateSim};
use crate::prove_gate_fn_equiv_sat::{
    EquivResult, GateFormalBackend, prove_gate_fn_equiv_with_backend,
};
use std::fmt::Write;
use xlsynth_pir::ir_parser::Parser;

fn function(chained: bool, exact_outer: bool, exact_inner: bool, aliases: bool) -> ir::Fn {
    let read = if chained {
        format!(
            "v: bits[3][2] = array_index(rows, indices=[i], assumed_in_bounds={exact_outer}, id=4)\n \
                 r: bits[3] = array_index(v, indices=[j], assumed_in_bounds={exact_inner}, id=5)"
        )
    } else {
        assert_eq!(exact_outer, exact_inner);
        format!(
            "r: bits[3] = array_index(rows, indices=[i, j], assumed_in_bounds={exact_inner}, id=5)"
        )
    };
    let observers = if aliases {
        "an: bits[1] = not(a, id=10)\n bi: bits[1] = identity(b, id=11)\n ret y: bits[2] = concat(an, bi, id=12)"
    } else {
        "ret y: bits[2] = concat(a, b, id=10)"
    };
    let ir = format!(
        "package projected
top fn f(rows: bits[3][2][3] id=1, i: bits[2] id=2, j: bits[2] id=3) -> bits[2] {{
 {read}
 high: bits[2] = bit_slice(r, start=1, width=2, id=6)
 zero: bits[2] = literal(value=0, id=7)
 a: bits[1] = eq(high, zero, id=8)
 b: bits[1] = or_reduce(r, id=9)
 {observers}
}}"
    );
    Parser::new(&ir)
        .parse_and_validate_package()
        .unwrap()
        .get_top_fn()
        .unwrap()
        .clone()
}

#[test]
fn compact_class_preserves_clamps_exact_zero_and_output_polarity() {
    for (chained, exact_outer, exact_inner, aliases) in [
        (false, false, false, false),
        (false, false, false, true),
        (false, true, true, false),
        (true, false, true, false),
        (true, true, false, false),
    ] {
        let f = function(chained, exact_outer, exact_inner, aliases);
        let r = f
            .node_refs()
            .into_iter()
            .find(|n| f.get_node(*n).text_id == 5)
            .unwrap();
        let region = Region::find(&f, r, &ir_utils::compute_users(&f)).unwrap();
        assert_eq!(region.common, [(1, false), (2, false)]);
        let opts = GatifyOptions {
            hash: true,
            fold: true,
            ..GatifyOptions::all_opts_disabled()
        };
        let result = gatify_prepared_fn(&f, opts).unwrap();
        let mut sim = PreparedGateSim::new(&result.gate_fn);
        // Every class occurs in every matrix position, including the clamped
        // final row. Invalid exact addresses must classify an all-zero record.
        for seed in 0..8u64 {
            let rows = (0..6).fold(0u64, |v, k| v | (((seed + k) % 8) << (3 * k)));
            for i in 0..4u64 {
                for j in 0..4u64 {
                    let value = if (i >= 3 && exact_outer) || (j >= 2 && exact_inner) {
                        0
                    } else {
                        (rows >> (3 * (2 * i.min(2) + j.min(1)))) & 7
                    };
                    let got = sim.eval(
                        &[
                            IrBits::make_ubits(18, rows).unwrap(),
                            IrBits::make_ubits(2, i).unwrap(),
                            IrBits::make_ubits(2, j).unwrap(),
                        ],
                        Collect::Tagged,
                    );
                    let high = (value < 2) ^ aliases; // The alias returns not(a).
                    assert_eq!(
                        got.outputs[0],
                        IrBits::make_ubits(2, 2 * u64::from(high) + u64::from(value != 0)).unwrap()
                    );
                    // Tags report the canonical AND-node value. The output
                    // edge of a factored predicate can have either polarity.
                    let tags = got.tagged_values.unwrap();
                    let output = &result.gate_fn.outputs[0].bit_vector;
                    assert_eq!(tags.get("eq_8"), Some(&(high ^ output.get_lsb(1).negated)));
                    assert_eq!(
                        tags.get("or_reduce_9"),
                        Some(&((value != 0) ^ output.get_lsb(0).negated))
                    );
                }
            }
        }
    }
}

#[test]
fn a_shared_consumer_observes_both_an_indexed_read_and_a_word_select() {
    // The read's scalar pair costs ten ANDs when factored and eleven on
    // the ordinary word. Returning the select word always permits its cut.
    for (payload, read_ands) in [("x", 10), ("r", 11)] {
        let text = format!(
            r#"package mixed
top fn f(rows: bits[3][2] id=1, i: bits[1] id=2, x: bits[3] id=3, s: bits[1] id=4) -> (bits[1], bits[1], bits[3], bits[3]) {{
 r: bits[3] = array_index(rows, indices=[i], assumed_in_bounds=true, id=5)
 hi: bits[2] = bit_slice(r, start=1, width=2, id=6)
 a: bits[1] = and_reduce(hi, id=7)
 b: bits[1] = and_reduce(r, id=8)
 an: bits[1] = not(a, id=9)
 both: bits[1] = and(b, b, id=16)
 bi: bits[1] = identity(both, id=10)
 fill: bits[3] = literal(value=2, id=11)
 word: bits[3] = sel(s, cases=[fill, x], id=12)
 c: bits[1] = or_reduce(word, id=13)
 ci: bits[1] = identity(c, id=17)
 mixed: bits[1] = and(an, ci, id=14)
 ret result: (bits[1], bits[1], bits[3], bits[3]) = tuple(mixed, bi, word, {payload}, id=15)
}}"#
        );
        let pkg = Parser::new(&text).parse_and_validate_package().unwrap();
        let f = pkg.get_top_fn().unwrap();
        let options = GatifyOptions {
            fold: true,
            hash: true,
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_disabled()
        };
        let mut captures = TerminalObservations::default();
        let out = gatify_lower_prepared_fn(f, &options, None, f, Some(&mut captures)).unwrap();
        assert_eq!(captures.words.len(), 1);
        let predicates = ["an", "bi"].map(|name| {
            let n = ir_utils::find_node_by_name(f, name).unwrap();
            *out.lowering_map[&n].get_lsb(0)
        });
        assert_eq!(
            get_aig_cone_stats(&out.gate_fn.gates, &predicates).and_nodes,
            read_ands
        );
        let trial =
            observed_select_rails::apply(out.gate_fn, &captures.words, &mut captures.rail_budget);
        let live = trial.post_order_operands(false);
        for (tag, origins) in [
            ("and_reduce_7", &[7, 9][..]),
            ("and_reduce_8", &[8, 10, 16]),
            ("or_reduce_13", &[13, 17]),
        ] {
            let gate = live
                .iter()
                .map(|r| &trial.gates[r.node.id])
                .find(|g| g.get_tags().unwrap_or_default().iter().any(|t| t == tag))
                .unwrap();
            assert!(
                origins
                    .iter()
                    .all(|id| gate.get_pir_node_ids().contains(id)),
                "{payload}: {tag} has origins {:?}",
                gate.get_pir_node_ids()
            );
        }
        let ordinary = gatify_prepared_fn(
            f,
            GatifyOptions {
                array_index_lowering_strategy: ArrayIndexLoweringStrategy::ForceOobOneHot,
                ..options
            },
        )
        .unwrap();
        assert_eq!(
            prove_gate_fn_equiv_with_backend(&ordinary.gate_fn, &trial, GateFormalBackend::Cadical)
                .unwrap(),
            EquivResult::Proved
        );
    }
}

#[test]
fn disjoint_signatures_single_observers_and_escaping_data_stay_ordinary() {
    for (body, result_type) in [
        ("ret y: bits[1] = xor(a, enable, id=11)", "bits[1]"),
        ("ret y: bits[5] = concat(a, r, id=11)", "bits[5]"),
        (
            "b: bits[1] = and_reduce(high, id=10)\n ret y: bits[2] = concat(a, b, id=11)",
            "bits[2]",
        ),
        (
            "b: bits[1] = or_reduce(low, id=10)\n ret y: bits[2] = concat(a, b, id=11)",
            "bits[2]",
        ),
    ] {
        let input = format!(
            "package negative
top fn f(arr: bits[4][3] id=1, i: bits[2] id=2, enable: bits[1] id=3) -> {result_type} {{
 r: bits[4] = array_index(arr, indices=[i], id=4)
 high: bits[2] = bit_slice(r, start=2, width=2, id=5)
 low: bits[2] = bit_slice(r, start=0, width=2, id=6)
 a: bits[1] = or_reduce(high, id=7)
 {body}
}}"
        );
        let pkg = Parser::new(&input).parse_and_validate_package().unwrap();
        let f = pkg.get_top_fn().unwrap();
        let r = f
            .node_refs()
            .into_iter()
            .find(|n| f.get_node(*n).text_id == 4)
            .unwrap();
        assert!(Region::find(f, r, &ir_utils::compute_users(f)).is_none());
    }
}

#[test]
fn many_residual_predicates_keep_the_shared_test_after_the_read() {
    let mut text = String::from(
        r#"package residuals
top fn f(rows: bits[34][2] id=1, i: bits[1] id=2) -> bits[32] {
 r: bits[34] = array_index(rows, indices=[i], assumed_in_bounds=true, id=3)
 high: bits[2] = bit_slice(r, start=32, width=2, id=4)
 p: bits[1] = and_reduce(high, id=5)
"#,
    );
    let mut results = Vec::new();
    for k in 0..32 {
        writeln!(
            text,
            " b{k}: bits[1] = bit_slice(r, start={k}, width=1, id={})",
            6 + 2 * k
        )
        .unwrap();
        writeln!(text, " t{k}: bits[1] = and(p, b{k}, id={})", 7 + 2 * k).unwrap();
        results.push(format!("t{k}"));
    }
    writeln!(
        text,
        " ret out: bits[32] = concat({}, id=70)\n}}",
        results.join(", ")
    )
    .unwrap();
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let result = gatify_prepared_fn(
        f,
        GatifyOptions {
            fold: true,
            hash: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap();
    let stats = get_aig_stats(&result.gate_fn);
    // 34 two-way selections, one shared AND, and 32 result ANDs.
    assert_eq!((stats.and_nodes, stats.max_depth), (3 * 34 + 1 + 32, 4));
}

#[test]
fn literal_banks_only_build_predicates_for_the_addressed_rows() {
    let mut and_counts = Vec::new();
    for banks in [2, 4096] {
        let text = format!(
            r#"package banks
top fn f(rows: bits[3][2][{banks}] id=1, i: bits[1] id=2) -> bits[4] {{
 first: bits[1] = literal(value=0, id=3)
 past_end: bits[100] = literal(value=0x100000000000000000000, id=4)
 a: bits[3] = array_index(rows, indices=[first, i], id=5)
 b: bits[3] = array_index(rows, indices=[past_end, i], id=6)
 ah: bits[2] = bit_slice(a, start=1, width=2, id=7)
 bh: bits[2] = bit_slice(b, start=1, width=2, id=8)
 p: bits[1] = and_reduce(ah, id=9)
 q: bits[1] = and_reduce(a, id=10)
 s: bits[1] = and_reduce(bh, id=11)
 t: bits[1] = and_reduce(b, id=12)
 ret out: bits[4] = concat(p, q, s, t, id=13)
}}"#
        );
        let pkg = Parser::new(&text).parse_and_validate_package().unwrap();
        let result = gatify_prepared_fn(
            pkg.get_top_fn().unwrap(),
            GatifyOptions {
                fold: true,
                hash: true,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();
        let gates = &result.gate_fn;
        and_counts.push(
            gates
                .gates
                .iter()
                .filter(|g| matches!(g, AigNode::And2 { .. }))
                .count(),
        );
        // Each addressed bank needs four row ANDs and two two-way selections.
        assert_eq!(get_aig_stats(gates).and_nodes, 2 * (4 + 2 * 3));
        let mut sim = PreparedGateSim::new(gates);
        // The last bank is zero. The first contains 6 and 7.
        for (i, expected) in [(0, 0b1000), (1, 0b1100)] {
            let sample = sim.eval_outputs(&[
                IrBits::make_ubits(6 * banks, 0b111110).unwrap(),
                IrBits::make_ubits(1, i).unwrap(),
            ]);
            assert_eq!(sample[0], IrBits::make_ubits(4, expected).unwrap());
        }
    }
    // Include dead gates retained by GateBuilder as well as reachable gates.
    assert_eq!(and_counts[0], and_counts[1]);
}

#[test]
fn literal_between_dynamic_dimensions_matches_ordinary_selection() {
    for (bank, exact) in [(1, true), (3, false), (3, true)] {
        let text = format!(
            r#"package strided
top fn f(rows: bits[3][2][3][2] id=1, i: bits[2] id=2, j: bits[2] id=3) -> bits[2] {{
 bank: bits[2] = literal(value={bank}, id=4)
 r: bits[3] = array_index(rows, indices=[i, bank, j], assumed_in_bounds={exact}, id=5)
 high: bits[2] = bit_slice(r, start=1, width=2, id=6)
 a: bits[1] = or_reduce(high, id=7)
 b: bits[1] = or_reduce(r, id=8)
 ret out: bits[2] = concat(a, b, id=9)
}}"#
        );
        let pkg = Parser::new(&text).parse_and_validate_package().unwrap();
        let f = pkg.get_top_fn().unwrap();
        let opts = GatifyOptions {
            fold: true,
            hash: true,
            ..GatifyOptions::all_opts_disabled()
        };
        let factored = gatify_prepared_fn(f, opts.clone()).unwrap();
        let ordinary = gatify_prepared_fn(
            f,
            GatifyOptions {
                array_index_lowering_strategy: ArrayIndexLoweringStrategy::ForceOobOneHot,
                ..opts
            },
        )
        .unwrap();
        let result = prove_gate_fn_equiv_with_backend(
            &factored.gate_fn,
            &ordinary.gate_fn,
            GateFormalBackend::Cadical,
        )
        .unwrap();
        assert_eq!(result, EquivResult::Proved);
    }
}

#[test]
fn repeated_scalar_operands_expand_a_wide_cube_once() {
    let repeated = vec!["p"; 4096].join(", ");
    let text = format!(
        r#"package repeated
top fn f(arr: bits[256][2] id=1, i: bits[1] id=2) -> bits[4] {{
 r: bits[256] = array_index(arr, indices=[i], id=3)
 p: bits[1] = and_reduce(r, id=4)
 short: bits[1] = and(p, p, id=5)
 long: bits[1] = and({repeated}, id=6)
 en_first: bits[1] = and(i, p, id=7)
 en_last: bits[1] = and(p, i, id=8)
 ret out: bits[4] = concat(short, long, en_first, en_last, id=9)
}}"#
    );
    let pkg = Parser::new(&text).parse_and_validate_package().unwrap();
    let f = pkg.get_top_fn().unwrap();
    let [read, reduce, short, long] =
        ["r", "p", "short", "long"].map(|name| ir_utils::find_node_by_name(f, name).unwrap());
    let mut copied_pins = 0;
    let observations = WordObservations::find(f, read, &ir_utils::compute_users(f), |pins| {
        copied_pins += pins;
        true
    })
    .unwrap();
    let boundary = observations.boundary_scalars().collect::<Vec<_>>();
    let [p, a, b] = boundary.as_slice() else {
        panic!("expected the reduction and both scalar conjunctions")
    };
    assert_eq!([p.value, a.value, b.value], [reduce, short, long]);
    assert_eq!(a.cube, b.cube);
    assert_eq!(a.cube.pins.len(), 256);
    // The independent enables in either position cost no pin copies. Each
    // ready conjunction copies the wide predicate once, regardless of fan-in.
    assert_eq!(copied_pins, 2 * 256);
}

#[test]
fn independent_addresses_share_the_table_of_an_aliased_array() {
    let text = "package aliases
top fn f(x: bits[3] id=1, y: bits[3] id=2, i: bits[2] id=3, j: bits[2] id=4) -> bits[4] {
 rows: bits[3][3] = array(x, y, x, id=5)
 r: bits[3] = array_index(rows, indices=[i], id=6)
 s: bits[3] = array_index(rows, indices=[j], id=7)
 ra: bits[2] = bit_slice(r, start=1, width=2, id=8)
 sa: bits[2] = bit_slice(s, start=1, width=2, id=9)
 p: bits[1] = or_reduce(ra, id=10)
 q: bits[1] = or_reduce(r, id=11)
 t: bits[1] = or_reduce(sa, id=12)
 u: bits[1] = or_reduce(s, id=13)
 ret out: bits[4] = concat(p, q, t, u, id=14)
}";
    let pkg = Parser::new(text).parse_and_validate_package().unwrap();
    let f = pkg.get_top_fn().unwrap();
    let opts = GatifyOptions {
        fold: true,
        hash: true,
        ..GatifyOptions::all_opts_disabled()
    };
    let mut gb = GateBuilder::new("memo".into(), GateBuilderOptions::opt());
    let mut env = GateEnv::new(f);
    let mut planner = ReadPredicates::default();
    for &n in &f.params {
        let node = f.get_node(n);
        env.add(
            n,
            GateOrVec::BitVector(gb.add_input(node.param_name().to_owned(), node.ty.bit_count())),
        );
    }
    for n in ir_utils::get_topological(f) {
        planner.consider(f, n, &mut gb, &env, &opts);
        gatify_node(
            f,
            n,
            f.get_node(n),
            &mut gb,
            &mut env,
            &opts,
            LoweringScope::WholeFunction,
        )
        .unwrap();
    }
    assert_eq!(planner.tables.len(), 1);
    assert_eq!(planner.results.len(), 4);
    let result = gatify_prepared_fn(
        f,
        GatifyOptions {
            track_pir_node_ids: true,
            ..opts
        },
    )
    .unwrap();
    let gates = &result.gate_fn.gates;
    let out = &result.gate_fn.outputs[0].bit_vector;
    // Independent indices can share only the row predicates and their inputs.
    let read_cone = |start| {
        let (cone, _) = extract_cone(
            &[out.get_lsb(start).node, out.get_lsb(start + 1).node],
            gates,
        );
        cone.into_iter()
            .filter(|n| matches!(gates[n.id], AigNode::And2 { .. }))
            .collect::<HashSet<_>>()
    };
    let first = read_cone(2);
    let second = read_cone(0);
    let shared = first.intersection(&second).collect::<Vec<_>>();
    assert_eq!(shared.len(), 4);
    for n in shared {
        let origins = gates[n.id].get_pir_node_ids();
        assert!(
            origins.contains(&6) && origins.contains(&7),
            "row gate {n:?} has origins {origins:?}"
        );
    }
    let mut sim = PreparedGateSim::new(&result.gate_fn);
    for x in 0..8u64 {
        for y in 0..8u64 {
            for i in 0..4u64 {
                for j in 0..4u64 {
                    let class = |k| {
                        let v = if k == 1 { y } else { x };
                        (u64::from(v >= 2) << 1) | u64::from(v != 0)
                    };
                    let bits = sim.eval(
                        &[
                            IrBits::make_ubits(3, x).unwrap(),
                            IrBits::make_ubits(3, y).unwrap(),
                            IrBits::make_ubits(2, i).unwrap(),
                            IrBits::make_ubits(2, j).unwrap(),
                        ],
                        Collect::None,
                    );
                    assert_eq!(
                        bits.outputs[0],
                        IrBits::make_ubits(4, (class(i) << 2) | class(j)).unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn an_observed_outer_array_prevents_partial_motion() {
    let text = "package retained
top fn f(rows: bits[3][2][3] id=1, outer: bits[2] id=2, inner: bits[2] id=3) -> (bits[2], bits[3][2]) {
 a: bits[3][2] = array_index(rows, indices=[outer], id=4)
 r: bits[3] = array_index(a, indices=[inner], id=5)
 high: bits[2] = bit_slice(r, start=1, width=2, id=6)
 p: bits[1] = or_reduce(high, id=7)
 q: bits[1] = or_reduce(r, id=8)
 both: bits[2] = concat(p, q, id=9)
 ret result: (bits[2], bits[3][2]) = tuple(both, a, id=10)
}";
    let pkg = Parser::new(text).parse_and_validate_package().unwrap();
    let f = pkg.get_top_fn().unwrap();
    let r = f
        .node_refs()
        .into_iter()
        .find(|n| f.get_node(*n).text_id == 5)
        .unwrap();
    let mut planner = ReadPredicates::default();
    let env = GateEnv::new(f);
    let mut gb = GateBuilder::new("skip".into(), GateBuilderOptions::opt());
    planner.consider(f, r, &mut gb, &env, &GatifyOptions::all_opts_disabled());
    assert!(planner.results.is_empty());
}

#[test]
fn an_independent_word_selection_keeps_the_read_gates_live() {
    for (outputs, width, ordinary_ands, expected_ands) in
        [("a, b, word", 5, 11, 11), ("a, b, word, c, d", 7, 22, 21)]
    {
        let text = format!(
            r#"package shared
top fn f(x: bits[3] id=1, y: bits[3] id=2, i: bits[1] id=3, j: bits[1] id=10) -> bits[{width}] {{
 rows: bits[3][2] = array(x, y, id=4)
 r: bits[3] = array_index(rows, indices=[i], assumed_in_bounds=true, id=5)
 hi: bits[2] = bit_slice(r, start=1, width=2, id=6)
 a: bits[1] = and_reduce(hi, id=7)
 b: bits[1] = and_reduce(r, id=8)
 word: bits[3] = sel(i, cases=[x, y], id=9)
 s: bits[3] = array_index(rows, indices=[j], assumed_in_bounds=true, id=11)
 sh: bits[2] = bit_slice(s, start=1, width=2, id=12)
 c: bits[1] = and_reduce(sh, id=13)
 d: bits[1] = and_reduce(s, id=14)
 ret out: bits[{width}] = concat({outputs}, id=15)
}}"#
        );
        let pkg = Parser::new(&text).parse_and_validate_package().unwrap();
        let f = pkg.get_top_fn().unwrap();
        for (strategy, ands) in [
            (ArrayIndexLoweringStrategy::ForceOobOneHot, ordinary_ands),
            (ArrayIndexLoweringStrategy::Auto, expected_ands),
        ] {
            let result = gatify_prepared_fn(
                f,
                GatifyOptions {
                    fold: true,
                    hash: true,
                    array_index_lowering_strategy: strategy,
                    ..GatifyOptions::all_opts_disabled()
                },
            )
            .unwrap();
            let stats = get_aig_stats(&result.gate_fn);
            // At i, keep the nine shared selection ANDs and two reductions.
            // At j, factoring still costs ten ANDs rather than eleven.
            assert_eq!(
                (stats.and_nodes, stats.max_depth),
                (ands, 4),
                "{strategy:?}"
            );
        }
    }
}

#[test]
fn narrow_dynamic_indices_only_build_addressable_rows() {
    for width in [0, 1] {
        let mut counts = Vec::new();
        for size in [1 << width, 4096] {
            let text = format!(
                r#"package narrow
top fn f(rows: bits[3][{size}][2] id=1, i: bits[{width}] id=2) -> bits[2] {{
 bank: bits[1] = literal(value=1, id=3)
 r: bits[3] = array_index(rows, indices=[bank, i], assumed_in_bounds=true, id=4)
 hi: bits[2] = bit_slice(r, start=1, width=2, id=5)
 a: bits[1] = and_reduce(hi, id=6)
 b: bits[1] = and_reduce(r, id=7)
 ret out: bits[2] = concat(a, b, id=8)
}}"#
            );
            let pkg = Parser::new(&text).parse_and_validate_package().unwrap();
            let result = gatify_prepared_fn(
                pkg.get_top_fn().unwrap(),
                GatifyOptions {
                    fold: true,
                    hash: true,
                    ..GatifyOptions::all_opts_disabled()
                },
            )
            .unwrap();
            counts.push(
                result
                    .gate_fn
                    .gates
                    .iter()
                    .filter(|g| matches!(g, AigNode::And2 { .. }))
                    .count(),
            );
            // Addressing bank one must still skip the full physical bank zero.
            let mut rows = vec![false; 6 * size];
            rows[3 * size + 1..3 * size + 3].fill(true);
            if width == 1 {
                rows[3 * size + 3..3 * size + 6].fill(true);
            }
            let mut sim = PreparedGateSim::new(&result.gate_fn);
            for i in 0..1 << width {
                let out = sim.eval_outputs(&[
                    IrBits::from_lsb_is_0(&rows),
                    IrBits::make_ubits(width, i).unwrap(),
                ]);
                assert_eq!(out[0], IrBits::make_ubits(2, 2 + i).unwrap());
            }
        }
        assert_eq!(counts[0], counts[1], "stored ANDs with a {width}-bit index");
    }
}

#[test]
fn cached_row_predicates_credit_each_read_including_interior_hash_hits() {
    for (width, params, rows, row_ands) in [
        (3, "rows: bits[3][2] id=1", "", 4),
        (
            5,
            "a: bits[1] id=1, b: bits[1] id=2, c: bits[1] id=3, d: bits[1] id=4, other: bits[5] id=5",
            r#"t: bits[1] = and(a, b, id=8)
 row: bits[5] = concat(d, c, b, a, t, id=9)
 rows: bits[5][2] = array(row, other, id=10)"#,
            8,
        ),
    ] {
        let high = width - 1;
        let text = format!(
            r#"package shared_gate
top fn f({params}, i: bits[1] id=6, j: bits[1] id=7) -> bits[4] {{
 {rows}
 r: bits[{width}] = array_index(rows, indices=[i], id=11)
 s: bits[{width}] = array_index(rows, indices=[j], id=12)
 rh: bits[{high}] = bit_slice(r, start=1, width={high}, id=13)
 sh: bits[{high}] = bit_slice(s, start=1, width={high}, id=14)
 p: bits[1] = and_reduce(rh, id=15)
 q: bits[1] = and_reduce(r, id=16)
 u: bits[1] = and_reduce(sh, id=17)
 v: bits[1] = and_reduce(s, id=18)
 ret out: bits[4] = concat(p, q, u, v, id=19)
}}"#
        );
        let package = Parser::new(&text).parse_and_validate_package().unwrap();
        let f = package.get_top_fn().unwrap();
        for lower in [gatify_prepared_fn, gatify] {
            let result = lower(
                f,
                GatifyOptions {
                    fold: true,
                    hash: true,
                    track_pir_node_ids: true,
                    ..GatifyOptions::all_opts_disabled()
                },
            )
            .unwrap();
            if width == 5 {
                // Bit zero is also the first AND in the high-four reduction.
                let t = ir_utils::find_node_by_name(f, "t").unwrap();
                let t_gate = result.lowering_map[&t].get_lsb(0).node;
                let gates = &result.gate_fn.gates;
                let output = &result.gate_fn.outputs[0].bit_vector;
                for start in [0, 2] {
                    let (cone, _) = extract_cone(
                        &[output.get_lsb(start).node, output.get_lsb(start + 1).node],
                        gates,
                    );
                    assert!(cone.contains(&t_gate));
                }
                assert_eq!(gates[t_gate.id].get_pir_node_ids(), &[8, 11, 12]);
            }
            // Each read uses the shared row ANDs and its own six selection
            // ANDs.
            let area = build_area_table_report(&result.gate_fn, f).unwrap();
            let counts: Vec<_> = area
                .rows
                .iter()
                .filter(|row| matches!(row.pir_node_id, 11 | 12))
                .map(|row| (row.pir_node_id, row.raw_aig_node_count))
                .collect();
            assert_eq!(counts, [(11, row_ands + 6), (12, row_ands + 6)]);
        }
    }
}

#[test]
fn folded_predicate_does_not_cache_a_discarded_subtree() {
    let mut gb = GateBuilder::new("folded".into(), GateBuilderOptions::opt());
    let data = gb.add_input("data".into(), 3);
    let [a, b, c] = std::array::from_fn(|i| *data.get_lsb(i));
    let t = gb.add_and_binary(a, b);
    // The high-four tree builds t again, then folds t & (c & !c) to zero.
    let row = AigBitVector::from_lsb_is_index_0(&[t, a, b, c, c.negate()]);
    let common: Vec<_> = (1..5).map(|i| (i, true)).collect();
    let key = TableKey {
        source: NodeRef { index: 0 },
        first_bit: 0,
        axes: vec![(1, 5)],
        tests: vec![
            Cube {
                pins: common.as_slice().into(),
                inverted: false,
            },
            Cube {
                pins: (0..5).map(|i| (i, true)).collect(),
                inverted: false,
            },
        ],
    };
    let table = PredicateTable::build(&key, &row, &common, &[0, 1, 2, 3, 4], &mut gb, true);
    assert_eq!(table.predicate_gates, HashSet::from([gb.get_false().node]));
}

#[test]
fn independent_reads_of_a_dead_word_column_fold_to_zero() {
    const READS: usize = 1024;
    let mut text = format!(
        r#"package folded
top fn dead_columns(data: bits[4096] id=1, indices: bits[{READS}] id=2) -> bits[{}] {{
 d: bits[1] = and_reduce(data, id=3)
 z: bits[1] = literal(value=0, id=4)
 row: bits[3] = concat(z, d, z, id=5)
 zero: bits[3] = literal(value=0, id=6)
 rows: bits[3][2] = array(row, zero, id=7)"#,
        2 * READS
    );
    let mut outputs = Vec::new();
    for k in 0..READS {
        let id = 8 + 5 * k;
        writeln!(
            text,
            "
 i{k}: bits[1] = bit_slice(indices, start={k}, width=1, id={id})
 r{k}: bits[3] = array_index(rows, indices=[i{k}], id={})
 h{k}: bits[2] = bit_slice(r{k}, start=1, width=2, id={})
 p{k}: bits[1] = and_reduce(h{k}, id={})
 q{k}: bits[1] = and_reduce(r{k}, id={})",
            id + 1,
            id + 2,
            id + 3,
            id + 4
        )
        .unwrap();
        outputs.push(format!("p{k}, q{k}"));
    }
    writeln!(
        text,
        " ret out: bits[{}] = concat({}, id={})\n}}",
        2 * READS,
        outputs.join(", "),
        8 + 5 * READS
    )
    .unwrap();
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    for lower in [gatify_prepared_fn, gatify] {
        let result = lower(
            f,
            GatifyOptions {
                fold: true,
                hash: true,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();
        assert!(
            result.gate_fn.outputs[0]
                .bit_vector
                .iter_lsb_to_msb()
                .all(|b| b.node.id == 0 && !b.negated)
        );
    }
}
