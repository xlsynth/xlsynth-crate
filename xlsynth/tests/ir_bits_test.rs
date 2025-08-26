// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrBits, XlsynthError};

#[test]
fn arith_and_logic_ops() -> Result<(), XlsynthError> {
    let a = IrBits::make_ubits(8, 0b0001_1010)?; // 26
    let b = IrBits::make_ubits(8, 0b0000_0011)?; // 3

    assert!(a.equals(&a));
    assert!(!a.equals(&b));
    assert_eq!(a.clone(), a.clone());
    assert_ne!(a.clone(), b.clone());
    assert!(a.get_bit(1)?);

    let sum = a.add(&b);
    assert_eq!(sum.to_u64()?, 29);

    let diff = a.sub(&b);
    assert_eq!(diff.to_u64()?, 23);

    let and = a.and(&b);
    assert_eq!(and.to_u64()?, 0b0000_0010);

    let or = a.or(&b);
    assert_eq!(or.to_u64()?, 0b0001_1011);

    let xor = a.xor(&b);
    assert_eq!(xor.to_u64()?, 0b0001_1001);

    let not_a = a.not();
    assert_eq!(not_a.to_u64()?, 0b1110_0101);

    let neg = a.negate();
    let abs = neg.abs();
    assert_eq!(abs, a);

    let umul = a.umul(&b);
    assert_eq!(umul.to_u64()?, 78);

    let smul = a.smul(&b);
    assert_eq!(smul.to_u64()?, 78);

    Ok(())
}

#[test]
fn conversions_shifts_and_slices() -> Result<(), XlsynthError> {
    let bits = IrBits::make_sbits(8, -2)?;
    assert_eq!(bits.to_i64()?, -2);
    assert_eq!(bits.to_u64()?, 0xFE);
    assert!(!bits.get_bit(0)?);
    assert!(bits.get_bit(1)?);

    let bytes = bits.to_bytes()?;
    assert_eq!(bytes, vec![0xFE]);

    let wide = IrBits::make_ubits(16, 0xABCD)?;
    assert_eq!(wide.to_bytes()?, vec![0xCD, 0xAB]);

    let shll = bits.shll(2);
    assert_eq!(shll.to_u64()?, (0xFEu64 << 2) & 0xFF);

    let shrl = bits.shrl(2);
    assert_eq!(shrl.to_u64()?, 0xFE >> 2);

    let shra = bits.shra(2);
    assert_eq!(shra.to_i64()?, -1);

    let slice = wide.width_slice(8, 8);
    assert_eq!(slice.to_u64()?, 0xAB);

    Ok(())
}
