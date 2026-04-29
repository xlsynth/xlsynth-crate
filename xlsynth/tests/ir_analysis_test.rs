// SPDX-License-Identifier: Apache-2.0

use xlsynth::*;

#[test]
fn test_known_bits_for_and_with_constant_mask() {
    let ir = r#"package kb

fn f(x: bits[8] id=1) -> bits[8] {
  k: bits[8] = literal(value=240, id=2)
  ret r: bits[8] = and(x, k, id=3)
}
"#;

    let mut package = IrPackage::parse_ir(ir, None).unwrap();
    package.set_top_by_name("f").unwrap();

    let analysis = package.create_ir_analysis().unwrap();
    let known = analysis.get_known_bits_for_node_id(3).unwrap();

    assert_eq!(known.mask.get_bit_count(), 8);
    assert_eq!(known.value.get_bit_count(), 8);

    // Since `k` is 0b1111_0000, the low 4 result bits are forced to 0 and thus
    // known; the high 4 result bits depend on `x` and are unknown.
    assert_eq!(known.mask.to_u64().unwrap(), 0x0f);
    assert_eq!(known.value.to_u64().unwrap(), 0x00);
}

#[test]
fn test_sel_range_with_context_is_tighter_than_fast() {
    // Mirrors the upstream XLS C API test demonstrating that context-sensitive
    // range analysis can tighten the bounds for clamp-like `sel` patterns.
    //
    // In particular, `sel(sgt(x, 2), cases=[2, x])` implies the result is
    // always >= 2, but fast analysis is expected to be weaker on the lower
    // bound.
    let ir = r#"package p
top fn f(x: bits[4] id=1) -> bits[4] {
  k: bits[4] = literal(value=2, id=2)
  p: bits[1] = sgt(x, k, id=3)
  ret y: bits[4] = sel(p, cases=[k, x], id=4)
}
"#;

    let mut package = IrPackage::parse_ir(ir, None).unwrap();
    package.set_top_by_name("f").unwrap();

    let fast = package
        .create_ir_analysis_with_level(IrAnalysisLevel::Fast)
        .unwrap();
    let ctx = package
        .create_ir_analysis_with_level(IrAnalysisLevel::RangeWithContext)
        .unwrap();

    let fast_bounds: Vec<(u64, u64)> = fast
        .get_intervals_for_node_id(4)
        .unwrap()
        .intervals()
        .unwrap()
        .iter()
        .map(|it| (it.lo.to_u64().unwrap(), it.hi.to_u64().unwrap()))
        .collect();
    let ctx_bounds: Vec<(u64, u64)> = ctx
        .get_intervals_for_node_id(4)
        .unwrap()
        .intervals()
        .unwrap()
        .iter()
        .map(|it| (it.lo.to_u64().unwrap(), it.hi.to_u64().unwrap()))
        .collect();

    assert_eq!(ctx_bounds.len(), 1, "ctx_bounds={ctx_bounds:?}");
    assert_eq!(fast_bounds.len(), 1, "fast_bounds={fast_bounds:?}");

    let (fast_lo, fast_hi) = fast_bounds[0];
    let (ctx_lo, ctx_hi) = ctx_bounds[0];

    assert_eq!(ctx_lo, 2);
    assert_eq!(ctx_hi, 7);

    // Fast analysis is expected to be weaker (at least on the lower bound).
    assert!(
        fast_lo < 2,
        "fast_bounds={:?} ctx_bounds={:?}",
        fast_bounds,
        ctx_bounds
    );
    assert!(
        fast_hi >= ctx_hi,
        "fast_bounds={:?} ctx_bounds={:?}",
        fast_bounds,
        ctx_bounds
    );
}

#[test]
fn test_bdd_predicate_queries_by_node_id() {
    let ir = r#"package p
top fn f(x: bits[8] id=1) -> bits[1] {
  zero: bits[8] = literal(value=0, id=2)
  one: bits[8] = literal(value=1, id=3)
  two: bits[8] = literal(value=2, id=4)
  x_eq_0: bits[1] = eq(x, zero, id=5)
  x_ne_0: bits[1] = not(x_eq_0, id=6)
  x_eq_1: bits[1] = eq(x, one, id=7)
  x_lt_2: bits[1] = ult(x, two, id=8)
  exclusive_eqs: bits[2] = concat(x_eq_0, x_eq_1, id=9)
  exhaustive_pair: bits[2] = concat(x_eq_0, x_ne_0, id=10)
  ret result: bits[1] = identity(x_lt_2, id=11)
}
"#;

    let mut package = IrPackage::parse_ir(ir, None).unwrap();
    package.set_top_by_name("f").unwrap();

    let analysis = package.create_ir_analysis().unwrap();

    assert!(analysis.at_most_one_bit_true(9).unwrap());
    assert!(!analysis.at_least_one_bit_true(9).unwrap());
    assert!(analysis.at_least_one_bit_true(10).unwrap());
    assert!(analysis.exactly_one_bit_true(10).unwrap());
    assert!(!analysis.exactly_one_bit_true(9).unwrap());

    assert!(analysis.known_not_equals(10, 1, 10, 0).unwrap());
    assert!(!analysis.known_not_equals(9, 1, 9, 0).unwrap());

    assert!(analysis.implies(9, 1, 8, 0).unwrap());
    assert!(!analysis.implies(8, 0, 9, 1).unwrap());
}
