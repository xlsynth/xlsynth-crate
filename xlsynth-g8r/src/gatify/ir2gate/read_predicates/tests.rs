// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::aig_sim::gate_sim::{Collect, PreparedGateSim};
use xlsynth_pir::ir_parser::Parser;

fn function(chained: bool, exact_outer: bool, exact_inner: bool) -> ir::Fn {
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
    let ir = format!(
        "package projected
top fn f(rows: bits[3][2][3] id=1, i: bits[2] id=2, j: bits[2] id=3) -> bits[2] {{
 {read}
 high: bits[2] = bit_slice(r, start=1, width=2, id=6)
 zero: bits[2] = literal(value=0, id=7)
 a: bits[1] = eq(high, zero, id=8)
 b: bits[1] = or_reduce(r, id=9)
 ret y: bits[2] = concat(a, b, id=10)
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
    for (chained, exact_outer, exact_inner) in [
        (false, false, false),
        (false, true, true),
        (true, false, true),
        (true, true, false),
    ] {
        let f = function(chained, exact_outer, exact_inner);
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
                        Collect::None,
                    );
                    assert_eq!(
                        got.outputs[0],
                        IrBits::make_ubits(2, 2 * u64::from(value < 2) + u64::from(value != 0))
                            .unwrap()
                    );
                }
            }
        }
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
        gatify_node(f, n, f.get_node(n), &mut gb, &mut env, &opts).unwrap();
    }
    assert_eq!(planner.tables.len(), 1);
    assert_eq!(planner.results.len(), 4);
    let result = gatify_prepared_fn(f, opts).unwrap();
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
