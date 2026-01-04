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
