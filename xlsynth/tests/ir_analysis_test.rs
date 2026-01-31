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
