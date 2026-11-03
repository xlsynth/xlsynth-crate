// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrPackage, XlsIrBits, XlsIrValue};
use xlsynth_dslx_fuzzer::Sample;
use xlsynth_dslx_fuzzer::array_fragment::{ArrayCast, CastDirection, MatrixUpdate};
use xlsynth_dslx_fuzzer::ast::{BinaryOp, BitsType, Body, Fragment, ValueRef, Width};
use xlsynth_dslx_fuzzer::compound::CompoundOp;

#[test]
fn both_cast_directions_and_each_rectangular_update_cell_match() -> Result<(), String> {
    for (width, signed) in [(1, false), (7, false), (129, true)] {
        let ty = BitsType {
            signed,
            width: Width::Concrete(width),
        };
        let mut operations = Vec::new();
        for direction in [CastDirection::BitsToArray, CastDirection::ArrayToBits] {
            for length in [2, 3] {
                operations.push(CompoundOp::ArrayCast(ArrayCast::new(direction, length)?));
            }
        }
        for row in 0..2 {
            for col in 0..3 {
                operations.push(CompoundOp::MatrixUpdate(MatrixUpdate::new(2, 3, row, col)?));
            }
        }
        let mut lanes = Vec::new();
        for op in operations {
            let mut body = Body::new(ty)?;
            let z = body.push(Fragment::Binary {
                op: BinaryOp::Xor,
                lhs: ValueRef::X,
                rhs: ValueRef::Y,
            })?;
            body.push(Fragment::Compound {
                op,
                args: [ValueRef::X, ValueRef::Y, z],
            })?;
            lanes.push(body);
        }
        let lane_count = lanes.len();
        let sample = Sample::new(vec![], lanes)?.prepare()?;
        let reference = IrPackage::parse_ir(&sample.expected_ir, None).unwrap();
        let converted = IrPackage::parse_ir(&sample.converted_ir, None).unwrap();
        let reference = reference.get_function("main").unwrap();
        let converted = converted.get_function(&sample.converted_top).unwrap();
        let limit = if width == 1 { 2 } else { 6 };
        for x in 0..limit {
            for y in 0..limit {
                let operand = |low: usize, high_bit: usize| {
                    let bits = (0..width)
                        .map(|i| i == high_bit || (i < 32 && low & (1 << i) != 0))
                        .collect::<Vec<_>>();
                    XlsIrBits::from_lsb_is_0(&bits)
                };
                let xb = operand(x, 128);
                let yb = operand(y, 67);
                let words = [&xb, &yb, &xb.xor(&yb)];
                let mut expected = Vec::new();
                for length in [2, 3, 2, 3] {
                    expected.push(XlsIrValue::from_bits(words[(x ^ y) % length]));
                }
                for row in 0..2 {
                    for col in 0..3 {
                        let slot = if row == x % 2 && col == y % 3 {
                            2
                        } else {
                            (row + col) % 2
                        };
                        expected.push(XlsIrValue::from_bits(words[slot]));
                    }
                }
                let expected = XlsIrValue::make_tuple(&expected);
                let args = [XlsIrValue::from_bits(&xb), XlsIrValue::from_bits(&yb)]
                    .into_iter()
                    .cycle()
                    .take(2 * lane_count)
                    .collect::<Vec<_>>();
                assert_eq!(reference.interpret(&args).unwrap(), expected);
                assert_eq!(converted.interpret(&args).unwrap(), expected);
            }
        }
    }
    Ok(())
}
