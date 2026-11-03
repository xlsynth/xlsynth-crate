// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrPackage, XlsIrBits, XlsIrValue};
use xlsynth_dslx_fuzzer::Sample;
use xlsynth_dslx_fuzzer::aggregate_match::AggregateMatch;
use xlsynth_dslx_fuzzer::ast::{BitsType, Body, Fragment, HelperModule, ValueRef, Width};
use xlsynth_dslx_fuzzer::compound::CompoundOp;
use xlsynth_dslx_fuzzer::constexpr::{ConstExpr, ConstInput, ConstantScope};
use xlsynth_pir::IrBits;

/// Builds the same match in a concrete constant or a parametric helper.
fn match_body(ty: BitsType) -> Result<Body, String> {
    let mut body = Body::new(ty)?;
    body.push(Fragment::Compound {
        op: CompoundOp::AggregateMatch(AggregateMatch::new([65, 4], ty.signed)?),
        args: [ValueRef::X, ValueRef::Y, ValueRef::X],
    })?;
    Ok(body)
}

#[test]
fn runtime_tags_indices_and_fields_survive_parametric_unpacking() -> Result<(), String> {
    let helper = HelperModule::new(
        match_body(BitsType {
            signed: false,
            width: Width::Parametric,
        })?,
        match_body(BitsType {
            signed: true,
            width: Width::Parametric,
        })?,
    )?;
    let mut lanes = Vec::new();
    let mut shapes = Vec::new();
    for width in [1, 8, 129] {
        for signed in [false, true] {
            let mut body = Body::new(BitsType {
                signed,
                width: Width::Concrete(width),
            })?;
            body.push(Fragment::Call {
                module: 0,
                args: [ValueRef::X, ValueRef::Y],
                explicit_parametric: signed,
            })?;
            lanes.push(body);
            shapes.push((width, signed));
        }
    }
    let sample = Sample::new(vec![helper], lanes)?.prepare()?;
    let reference = IrPackage::parse_ir(&sample.expected_ir, None).unwrap();
    let converted = IrPackage::parse_ir(&sample.converted_ir, None).unwrap();
    let reference = reference.get_function("main").unwrap();
    let converted = converted.get_function(&sample.converted_top).unwrap();
    for tags in 0..8 {
        let mut args = Vec::new();
        let mut results = Vec::new();
        for &(width, second_header) in &shapes {
            let x_bits = (0..width)
                .map(|i| i == 64 || i == 70 || (i < 3 && tags & (1 << i) != 0))
                .collect::<Vec<_>>();
            let y_bits = (0..width)
                .map(|i| matches!(i, 3 | 15 | 66))
                .collect::<Vec<_>>();
            let x = XlsIrBits::from_lsb_is_0(&x_bits);
            let y = XlsIrBits::from_lsb_is_0(&y_bits);
            let expected = if !x_bits[0] {
                y.clone()
            } else {
                let flip = x_bits.get(1) == Some(&true);
                let index = x_bits.get(2) == Some(&true);
                let word = match (index, flip) {
                    (false, true) => Some(&x_bits),
                    (true, false) => Some(&y_bits),
                    _ => None,
                };
                let value = XlsIrBits::from_lsb_is_0(
                    &(0..width)
                        .map(|i| i < 65 && word.is_some_and(|bits| bits[i]))
                        .collect::<Vec<_>>(),
                );
                let header = if second_header { &x_bits } else { &y_bits };
                let field = XlsIrBits::from_lsb_is_0(
                    &(0..width).map(|i| i < 4 && header[i]).collect::<Vec<_>>(),
                );
                field.add(&value)
            };
            args.extend([XlsIrValue::from_bits(&x), XlsIrValue::from_bits(&y)]);
            results.push(XlsIrValue::from_bits(&expected));
        }
        let expected = XlsIrValue::make_tuple(&results);
        assert_eq!(reference.interpret(&args).unwrap(), expected);
        assert_eq!(converted.interpret(&args).unwrap(), expected);
    }
    Ok(())
}

#[test]
fn bound_aggregate_fields_work_in_module_constants() -> Result<(), String> {
    let ty = BitsType {
        signed: false,
        width: Width::Concrete(8),
    };
    let expression = ConstExpr::new(
        match_body(ty)?,
        [
            ConstInput::Literal(IrBits::from_lsb_fn(8, |i| i == 0 || i == 1)),
            ConstInput::Literal(IrBits::from_lsb_fn(8, |i| i == 3 || i == 7)),
        ],
    )?;
    let mut lane = Body::new(ty)?;
    lane.push(Fragment::Constant {
        expression: Box::new(expression),
        scope: ConstantScope::Module,
        assert_value: true,
    })?;
    let sample = Sample::new(vec![], vec![lane])?.prepare()?;
    let reference = IrPackage::parse_ir(&sample.expected_ir, None).unwrap();
    let converted = IrPackage::parse_ir(&sample.converted_ir, None).unwrap();
    let args = [
        XlsIrValue::make_ubits(8, 0).unwrap(),
        XlsIrValue::make_ubits(8, 0).unwrap(),
    ];
    // At tags=3, slot zero contains Word(3) and the low four header bits are 8.
    let expected = XlsIrValue::make_tuple(&[XlsIrValue::make_ubits(8, 11).unwrap()]);
    assert_eq!(
        reference
            .get_function("main")
            .unwrap()
            .interpret(&args)
            .unwrap(),
        expected
    );
    assert_eq!(
        converted
            .get_function(&sample.converted_top)
            .unwrap()
            .interpret(&args)
            .unwrap(),
        expected
    );
    Ok(())
}
