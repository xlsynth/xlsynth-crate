// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrPackage, XlsIrBits, XlsIrValue};
use xlsynth_dslx_fuzzer::Sample;
use xlsynth_dslx_fuzzer::ast::{BitsType, Body, Fragment, ValueRef, Width};
use xlsynth_dslx_fuzzer::compound::CompoundOp;
use xlsynth_dslx_fuzzer::sum_pattern::{SumComparison, SumPattern, SumShape, SumVariant};

/// Sets the sign bit and a payload bit, truncating to a field when requested.
fn operand(width: usize, field_width: usize, low_bit: bool, payload_bit: usize) -> XlsIrBits {
    let mut bits = vec![false; width];
    bits[width - 1] = true;
    if payload_bit < width {
        bits[payload_bit] = true;
    }
    bits[0] = low_bit;
    if field_width < width {
        bits[field_width..].fill(false);
    }
    XlsIrBits::from_lsb_is_0(&bits)
}

#[test]
fn both_tag_values_and_nested_arms_observe_payload_fields() {
    for width in [1, 8, 129] {
        for signed in [false, true] {
            let ty = BitsType {
                signed,
                width: Width::Concrete(width),
            };
            let mut lanes = Vec::new();
            let mut patterns = Vec::new();
            for (index, off) in SumVariant::ALL.into_iter().enumerate() {
                for on in &SumVariant::ALL[index + 1..] {
                    for tail_when_marked in [false, true] {
                        let pattern = SumPattern {
                            variants: [off, *on],
                            widths: [3, 128],
                            tail_when_marked,
                        };
                        let mut body = Body::new(ty).unwrap();
                        body.push(Fragment::Compound {
                            op: CompoundOp::Sum(pattern),
                            args: [ValueRef::X, ValueRef::Y, ValueRef::X],
                        })
                        .unwrap();
                        lanes.push(body);
                        patterns.push(pattern);
                    }
                }
            }
            let sample = Sample::new(vec![], lanes).unwrap().prepare().unwrap();
            let expected_ir = IrPackage::parse_ir(&sample.expected_ir, None).unwrap();
            let converted_ir = IrPackage::parse_ir(&sample.converted_ir, None).unwrap();
            let reference = expected_ir.get_function("main").unwrap();
            let converted = converted_ir.get_function(&sample.converted_top).unwrap();
            for selected in [false, true] {
                for marked in [false, true] {
                    let x = operand(width, width, selected, 5);
                    let y = operand(width, width, marked, 2);
                    let low = operand(width, 3, marked, 2);
                    let high = operand(width, 128, selected, 5);
                    let mut args = Vec::new();
                    let mut results = Vec::new();
                    for pattern in &patterns {
                        args.extend([XlsIrValue::from_bits(&x), XlsIrValue::from_bits(&y)]);
                        let bits = match pattern.variants[usize::from(selected)] {
                            SumVariant::Empty => &x,
                            SumVariant::Word => &low,
                            SumVariant::Pair => &high.sub(&low),
                            SumVariant::Nested if marked == pattern.tail_when_marked => &high,
                            SumVariant::Nested => &low,
                        };
                        results.push(XlsIrValue::from_bits(bits));
                    }
                    let expected = XlsIrValue::make_tuple(&results);
                    assert_eq!(reference.interpret(&args).unwrap(), expected);
                    assert_eq!(converted.interpret(&args).unwrap(), expected);
                }
            }
        }
    }
}

#[test]
fn equality_observes_only_fields_of_the_active_variants() {
    let ty = BitsType {
        signed: true,
        width: Width::Concrete(8),
    };
    let mut lanes = Vec::new();
    let mut comparisons = Vec::new();
    for variants in [
        [SumVariant::Empty, SumVariant::Word],
        [SumVariant::Pair, SumVariant::Nested],
    ] {
        for shape in SumShape::ALL {
            for unequal in [false, true] {
                let comparison = SumComparison {
                    variants,
                    widths: [3, 65],
                    shape,
                    unequal,
                };
                let mut body = Body::new(ty).unwrap();
                body.push(Fragment::Compound {
                    op: CompoundOp::SumComparison(comparison),
                    args: [ValueRef::X, ValueRef::Y, ValueRef::X],
                })
                .unwrap();
                lanes.push(body);
                comparisons.push(comparison);
            }
        }
    }
    let sample = Sample::new(vec![], lanes).unwrap().prepare().unwrap();
    let reference = IrPackage::parse_ir(&sample.expected_ir, None).unwrap();
    let converted = IrPackage::parse_ir(&sample.converted_ir, None).unwrap();
    let reference_main = reference.get_function("main").unwrap();
    let converted_main = converted.get_function(&sample.converted_top).unwrap();
    for x in [0u8, 1, 3, 4, 8, 12, 15, 255] {
        for y in [x, x ^ 1, x ^ 128] {
            let x_arg = XlsIrValue::make_ubits(8, u64::from(x)).unwrap();
            let y_arg = XlsIrValue::make_ubits(8, u64::from(y)).unwrap();
            let mut args = Vec::new();
            let mut results = Vec::new();
            for comparison in &comparisons {
                args.extend([x_arg.clone(), y_arg.clone()]);
                let left_variant = comparison.variants[usize::from(x & 1)];
                let right_variant = comparison.variants[usize::from((x >> 1) & 1)];
                let low_equal = (x & 7) == (y & 7);
                let high_equal = y == x;
                let marks_equal = (x == 255) == (y == 255);
                let item_equal = left_variant == right_variant
                    && match left_variant {
                        SumVariant::Empty => true,
                        SumVariant::Word => low_equal,
                        SumVariant::Pair => low_equal && high_equal,
                        SumVariant::Nested => low_equal && high_equal && marks_equal,
                    };
                let equal = item_equal
                    && match comparison.shape {
                        SumShape::Bare => true,
                        SumShape::Tuple => marks_equal,
                        SumShape::Array => low_equal,
                        SumShape::Nested => match (x & 4 != 0, x & 8 != 0) {
                            (false, false) => true,
                            (true, true) => low_equal && high_equal,
                            _ => false,
                        },
                    };
                results.push(
                    XlsIrValue::make_ubits(8, u64::from(equal ^ comparison.unequal)).unwrap(),
                );
            }
            let expected = XlsIrValue::make_tuple(&results);
            assert_eq!(reference_main.interpret(&args).unwrap(), expected);
            assert_eq!(converted_main.interpret(&args).unwrap(), expected);
        }
    }
}
