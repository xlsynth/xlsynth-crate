// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrPackage, XlsIrBits, XlsIrValue};
use xlsynth_dslx_fuzzer::Sample;
use xlsynth_dslx_fuzzer::ast::{BinaryOp, BitsType, Body, Fragment, HelperModule, ValueRef, Width};
use xlsynth_dslx_fuzzer::compound::CompoundOp;
use xlsynth_dslx_fuzzer::constexpr::{ConstExpr, ConstInput, ConstantScope};
use xlsynth_dslx_fuzzer::select_fold::{ArraySelect, CaseArray, NestedFold, SelectKind};
use xlsynth_pir::IrBits;

/// Uses both runtime inputs and a distinct third value as fragment operands.
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
fn array_selects_match_enabled_cases_and_low_bit_priority() -> Result<(), String> {
    let ty = BitsType {
        signed: false,
        width: Width::Concrete(5),
    };
    let mut configurations = Vec::new();
    let mut lanes = Vec::new();
    for length in [3, 5] {
        for cases in CaseArray::ALL {
            for kind in SelectKind::ALL {
                configurations.push((length, cases, kind));
                let select = ArraySelect::new(kind, cases, length)?;
                lanes.push(fragment_body(ty, CompoundOp::ArraySelect(select))?);
            }
        }
    }
    let sample = Sample::new(vec![], lanes)?.prepare()?;
    let reference = IrPackage::parse_ir(&sample.expected_ir, None).unwrap();
    let converted = IrPackage::parse_ir(&sample.converted_ir, None).unwrap();
    let reference = reference.get_function("main").unwrap();
    let converted = converted.get_function(&sample.converted_top).unwrap();
    for x in 0..32u64 {
        for y in 0..8u64 {
            let z = (x + y) & 31;
            let mut expected = Vec::new();
            for &(length, cases, kind) in &configurations {
                let mut elements: Vec<u64> = [y, z, x].into_iter().cycle().take(length).collect();
                match cases {
                    CaseArray::Updated => elements[y as usize % length] = x ^ z,
                    CaseArray::Ellipsis => elements[1..].fill(z),
                }
                let mut enabled = elements
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| x & (1 << i) != 0)
                    .map(|(_, case)| *case);
                let result = match kind {
                    SelectKind::Priority => enabled.next().unwrap_or((y + z) & 31),
                    SelectKind::OneHot => enabled.fold(0, |acc, case| acc | case),
                };
                expected.push(XlsIrValue::make_ubits(5, result).unwrap());
            }
            let pair = [
                XlsIrValue::make_ubits(5, x).unwrap(),
                XlsIrValue::make_ubits(5, y).unwrap(),
            ];
            let args = pair
                .into_iter()
                .cycle()
                .take(2 * configurations.len())
                .collect::<Vec<_>>();
            let expected = XlsIrValue::make_tuple(&expected);
            assert_eq!(reference.interpret(&args).unwrap(), expected);
            assert_eq!(converted.interpret(&args).unwrap(), expected);
        }
    }
    Ok(())
}

#[test]
fn tuple_folds_preserve_all_carry_projections() -> Result<(), String> {
    let ty = BitsType {
        signed: true,
        width: Width::Concrete(5),
    };
    let mut lanes = Vec::new();
    for field in 0..3 {
        let fold = NestedFold::new(3, 2, BinaryOp::Xor, field)?;
        lanes.push(fragment_body(ty, CompoundOp::NestedFold(fold))?);
    }
    let sample = Sample::new(vec![], lanes)?.prepare()?;
    let reference = IrPackage::parse_ir(&sample.expected_ir, None).unwrap();
    let converted = IrPackage::parse_ir(&sample.converted_ir, None).unwrap();
    let reference = reference.get_function("main").unwrap();
    let converted = converted.get_function(&sample.converted_top).unwrap();
    // x=3, y=7, z=10. The three outer carry triples, modulo 32, are
    // (15, 9, 20), (9, 19, 13), (23, 7, 25).
    let pair = [
        XlsIrValue::make_ubits(5, 3).unwrap(),
        XlsIrValue::make_ubits(5, 7).unwrap(),
    ];
    let args = pair.into_iter().cycle().take(6).collect::<Vec<_>>();
    let expected = [23, 7, 25].map(|n| XlsIrValue::make_ubits(5, n).unwrap());
    let expected = XlsIrValue::make_tuple(&expected);
    assert_eq!(reference.interpret(&args).unwrap(), expected);
    assert_eq!(converted.interpret(&args).unwrap(), expected);
    Ok(())
}

#[test]
fn array_selects_and_nested_folds_specialize_and_evaluate_in_constant_bindings()
-> Result<(), String> {
    for op in [
        CompoundOp::ArraySelect(ArraySelect::new(SelectKind::OneHot, CaseArray::Updated, 5)?),
        CompoundOp::ArraySelect(ArraySelect::new(
            SelectKind::Priority,
            CaseArray::Ellipsis,
            3,
        )?),
        CompoundOp::NestedFold(NestedFold::new(2, 1, BinaryOp::Sub, 0)?),
        CompoundOp::NestedFold(NestedFold::new(4, 3, BinaryOp::Add, 2)?),
    ] {
        let parametric = |signed| BitsType {
            signed,
            width: Width::Parametric,
        };
        let helper = HelperModule::new(
            fragment_body(parametric(false), op)?,
            fragment_body(parametric(true), op)?,
        )?;
        let mut lanes = Vec::new();
        let mut args = Vec::new();
        for (width, signed, scope) in [
            (1, false, ConstantScope::Local),
            (129, true, ConstantScope::Module),
        ] {
            let ty = BitsType {
                signed,
                width: Width::Concrete(width),
            };
            let mut body = Body::new(ty)?;
            let result = body.push(Fragment::Call {
                module: 0,
                args: [ValueRef::X, ValueRef::Y],
                explicit_parametric: true,
            })?;
            let expression = ConstExpr::new(
                fragment_body(ty, op)?,
                [
                    ConstInput::Literal(IrBits::all_ones(width)),
                    ConstInput::Literal(IrBits::from_lsb_fn(width, |i| i % 3 == 0)),
                ],
            )?;
            let constant = body.push(Fragment::Constant {
                expression: Box::new(expression),
                scope,
                assert_value: true,
            })?;
            body.push(Fragment::Binary {
                op: BinaryOp::Xor,
                lhs: result,
                rhs: constant,
            })?;
            lanes.push(body);
            for stride in [2, 5] {
                let bits = (0..width).map(|i| i % stride == 0).collect::<Vec<_>>();
                args.push(XlsIrValue::from_bits(&XlsIrBits::from_lsb_is_0(&bits)));
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
                .unwrap()
        );
    }
    Ok(())
}
