// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use xlsynth::dslx::{ImportData, parse_and_typecheck};
use xlsynth::{IrPackage, XlsIrValue};
use xlsynth_dslx_fuzzer::Sample;
use xlsynth_dslx_fuzzer::ast::{BinaryOp, BitsType, Body, Fragment, UnaryOp, ValueRef, Width};
use xlsynth_dslx_fuzzer::bit_projection::{BitProjection, ProjectionKind};
use xlsynth_dslx_fuzzer::compound::CompoundOp;
use xlsynth_dslx_fuzzer::constexpr::{ConstExpr, ConstInput};
use xlsynth_dslx_fuzzer::select_fold::ArrayFold;
use xlsynth_pir::IrBits;

#[test]
fn fold_result_can_refer_to_each_preceding_binding() -> Result<(), Box<dyn Error>> {
    let ty = BitsType {
        signed: false,
        width: Width::Concrete(8),
    };
    let mut lanes = vec![];
    for result in 0..5 {
        let mut body = Body::new(ty)?;
        let capture = body.push(Fragment::Binary {
            op: BinaryOp::Xor,
            lhs: ValueRef::X,
            rhs: ValueRef::Y,
        })?;
        body.push(Fragment::Compound {
            op: CompoundOp::ArrayFold(ArrayFold::new(3, 2, result)?),
            args: [ValueRef::X, ValueRef::Y, capture],
        })?;
        lanes.push(body);
    }
    let sample = Sample::new(vec![], lanes)?;
    let args = [XlsIrValue::make_ubits(8, 3)?, XlsIrValue::make_ubits(8, 5)?]
        .into_iter()
        .cycle()
        .take(10)
        .collect::<Vec<_>>();
    let reference = IrPackage::parse_ir(&sample.expected_ir()?.to_string(), None)?;
    assert_eq!(
        reference.get_function("main")?.interpret(&args)?,
        XlsIrValue::parse_typed("(bits[8]:3, bits[8]:5, bits[8]:6, bits[8]:19, bits[8]:17)")?
    );
    // Bytecode independently checks the source, including cases where the
    // converter currently returns an unrelated, later binding from the loop.
    let mut source = sample.sources()?.remove(0).text;
    source.push_str("const_assert!(main(u8:3, u8:5, u8:3, u8:5, u8:3, u8:5, u8:3, u8:5, u8:3, u8:5) == (u8:3, u8:5, u8:6, u8:19, u8:17));");
    parse_and_typecheck(&source, "sample.x", "sample", &mut ImportData::default())?;
    Ok(())
}

#[test]
fn bit_projections_match_scalar_semantics_across_the_sign_boundary() -> Result<(), Box<dyn Error>> {
    let ty = BitsType {
        signed: false,
        width: Width::Concrete(8),
    };
    let mut lanes = vec![];
    for kind in ProjectionKind::ALL {
        for signed in [false, true] {
            for unary in [None, Some(UnaryOp::Not), Some(UnaryOp::Neg)] {
                let mut body = Body::new(ty)?;
                body.push(Fragment::Compound {
                    op: CompoundOp::BitProjection(BitProjection::new(kind, 3, signed, unary)?),
                    args: [ValueRef::X, ValueRef::Y, ValueRef::X],
                })?;
                lanes.push(body);
            }
        }
    }
    let prepared = Sample::new(vec![], lanes)?.prepare()?;
    let reference = IrPackage::parse_ir(&prepared.expected_ir, None)?;
    let converted = IrPackage::parse_ir(&prepared.converted_ir, None)?;
    for x in [0u8, 7, 32, 56, 255] {
        for y in [0u8, 1, 3, 5, 6, 7, 65, 255] {
            let mut expected = vec![];
            for (bits, width) in [
                (((x & 7) << 3) | (y & 7), 6),
                (if y & 1 == 0 { (x >> 3) & 7 } else { x & 7 }, 3),
                ((x & 63).checked_shr(u32::from(y)).unwrap_or(0) & 7, 3),
            ] {
                for signed in [false, true] {
                    for raw in [bits, !bits, bits.wrapping_neg()] {
                        let mask = (1 << width) - 1;
                        let value = if signed && raw & (1 << (width - 1)) != 0 {
                            raw | !mask
                        } else {
                            raw & mask
                        };
                        expected.push(XlsIrValue::make_ubits(8, u64::from(value))?);
                    }
                }
            }
            let args = [
                XlsIrValue::make_ubits(8, u64::from(x))?,
                XlsIrValue::make_ubits(8, u64::from(y))?,
            ]
            .into_iter()
            .cycle()
            .take(2 * expected.len())
            .collect::<Vec<_>>();
            let expected = XlsIrValue::make_tuple(&expected);
            assert_eq!(reference.get_function("main")?.interpret(&args)?, expected);
            assert_eq!(
                converted
                    .get_function(&prepared.converted_top)?
                    .interpret(&args)?,
                expected
            );
        }
    }
    Ok(())
}

#[test]
fn specialization_and_runtime_call_are_independently_observable() -> Result<(), Box<dyn Error>> {
    let ty = BitsType {
        signed: true,
        width: Width::Concrete(7),
    };
    let mut expression = Body::new(ty)?;
    expression.push(Fragment::Binary {
        op: BinaryOp::Xor,
        lhs: ValueRef::X,
        rhs: ValueRef::Y,
    })?;
    let check = ConstExpr::new(
        expression,
        [
            ConstInput::Literal(IrBits::all_ones(7)),
            ConstInput::Literal(IrBits::zero(7)),
        ],
    )?;
    let sample = Sample::new(vec![], vec![Body::new(ty)?])?.with_specialization_check(check)?;
    let prepared = sample.prepare()?;
    let args = [
        XlsIrValue::make_ubits(7, 42)?,
        XlsIrValue::make_ubits(7, 0)?,
    ];
    let expected = XlsIrValue::parse_typed("(bits[7]:42, bits[7]:127, bits[7]:127)")?;
    for (ir, top) in [
        (&prepared.expected_ir, "main"),
        (&prepared.converted_ir, prepared.converted_top.as_str()),
    ] {
        assert_eq!(
            IrPackage::parse_ir(ir, None)?
                .get_function(top)?
                .interpret(&args)?,
            expected
        );
    }
    Ok(())
}
