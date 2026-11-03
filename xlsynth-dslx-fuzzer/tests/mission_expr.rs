// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrPackage, XlsIrBits, XlsIrValue};
use xlsynth_dslx_fuzzer::Sample;
use xlsynth_dslx_fuzzer::ast::{
    BinaryOp, BitsType, Body, Comparison, Fragment, HelperModule, ValueRef, Width,
};
use xlsynth_dslx_fuzzer::compound::CompoundOp;
use xlsynth_dslx_fuzzer::constexpr::{ConstExpr, ConstInput, ConstantScope};
use xlsynth_dslx_fuzzer::mission_expr::{CheckedCast, LogicalPredicates, ReceiverMap};
use xlsynth_pir::IrBits;

/// Supplies three lane values so the receiver, selector and comparisons can
/// vary with the two independent parameters.
fn fragment_body(ty: BitsType, op: CompoundOp) -> Result<Body, String> {
    let mut body = Body::new(ty)?;
    let z = body.push(Fragment::Binary {
        op: BinaryOp::Add,
        lhs: ValueRef::X,
        rhs: ValueRef::Y,
    })?;
    body.push(Fragment::Compound {
        op,
        args: [ValueRef::X, ValueRef::Y, z],
    })?;
    Ok(body)
}

#[test]
fn runtime_receiver_and_predicates_select_expected_elements_and_branches() -> Result<(), String> {
    let ty = BitsType {
        signed: false,
        width: Width::Concrete(4),
    };
    let comparisons = [Comparison::Lt, Comparison::Ne];
    let lanes = [
        CompoundOp::ReceiverMap(ReceiverMap::new(7, 4)?),
        CompoundOp::LogicalPredicates(LogicalPredicates::new(comparisons, false)),
        CompoundOp::LogicalPredicates(LogicalPredicates::new(comparisons, true)),
    ]
    .map(|op| fragment_body(ty, op))
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;
    let sample = Sample::new(vec![], lanes)?.prepare()?;
    let reference = IrPackage::parse_ir(&sample.expected_ir, None).unwrap();
    let converted = IrPackage::parse_ir(&sample.converted_ir, None).unwrap();
    let reference = reference.get_function("main").unwrap();
    let converted = converted.get_function(&sample.converted_top).unwrap();
    for x in 0..16 {
        for y in 0..16 {
            let z = (x + y) & 15;
            let element = [x, z, y, x ^ z][(z % 4) as usize];
            let mapped = ((element ^ y) + y) & 15;
            let either = if x < y || y != z { x } else { z };
            let both = if x < y && y != z { x } else { z };
            let expected = [mapped, either, both].map(|n| XlsIrValue::make_ubits(4, n).unwrap());
            let expected = XlsIrValue::make_tuple(&expected);
            let args = [x, y, x, y, x, y].map(|n| XlsIrValue::make_ubits(4, n).unwrap());
            assert_eq!(reference.interpret(&args).unwrap(), expected);
            assert_eq!(converted.interpret(&args).unwrap(), expected);
        }
    }
    Ok(())
}

#[test]
fn mission_expressions_specialize_at_multiple_widths_and_work_in_constants() -> Result<(), String> {
    for op in [
        CompoundOp::ReceiverMap(ReceiverMap::new(65, 3)?),
        CompoundOp::LogicalPredicates(LogicalPredicates::new(
            [Comparison::Lt, Comparison::Ge],
            true,
        )),
        CompoundOp::CheckedCast(CheckedCast::new(0)?),
        CompoundOp::CheckedCast(CheckedCast::new(4)?),
    ] {
        let helper_ty = |signed| BitsType {
            signed,
            width: Width::Parametric,
        };
        let helper = HelperModule::new(
            fragment_body(helper_ty(false), op)?,
            fragment_body(helper_ty(true), op)?,
        )?;
        let mut lanes = Vec::new();
        let mut args = Vec::new();
        for width in [1, 129] {
            for signed in [false, true] {
                let ty = BitsType {
                    signed,
                    width: Width::Concrete(width),
                };
                let mut body = Body::new(ty)?;
                let runtime = body.push(Fragment::Call {
                    module: 0,
                    args: [ValueRef::X, ValueRef::Y],
                    explicit_parametric: signed,
                })?;
                let expression = ConstExpr::new(
                    fragment_body(ty, op)?,
                    [
                        ConstInput::Literal(IrBits::from_lsb_fn(width, |i| i % 4 == 0)),
                        ConstInput::Literal(IrBits::all_ones(width)),
                    ],
                )?;
                let constant = body.push(Fragment::Constant {
                    expression: Box::new(expression),
                    scope: ConstantScope::Module,
                    assert_value: true,
                })?;
                body.push(Fragment::Binary {
                    op: BinaryOp::Add,
                    lhs: runtime,
                    rhs: constant,
                })?;
                lanes.push(body);
                for stride in [4, 7] {
                    let bits = (0..width).map(|i| i % stride == 0).collect::<Vec<_>>();
                    args.push(XlsIrValue::from_bits(&XlsIrBits::from_lsb_is_0(&bits)));
                }
            }
        }
        let sample = Sample::new(vec![helper], lanes)?.prepare()?;
        let reference = IrPackage::parse_ir(&sample.expected_ir, None).unwrap();
        let converted = IrPackage::parse_ir(&sample.converted_ir, None).unwrap();
        assert_eq!(
            reference
                .get_function("main")
                .unwrap()
                .interpret(&args)
                .unwrap(),
            converted
                .get_function(&sample.converted_top)
                .unwrap()
                .interpret(&args)
                .unwrap(),
            "{op:?}",
        );
    }
    Ok(())
}
