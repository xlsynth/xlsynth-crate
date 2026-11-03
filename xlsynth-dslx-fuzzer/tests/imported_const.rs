// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrPackage, XlsIrBits, XlsIrValue};
use xlsynth_dslx_fuzzer::Sample;
use xlsynth_dslx_fuzzer::ast::{BinaryOp, BitsType, Body, Fragment, HelperModule, ValueRef, Width};
use xlsynth_dslx_fuzzer::constexpr::{ConstExpr, ConstInput, ConstantScope};
use xlsynth_dslx_fuzzer::imported_const::ExportedConstants;
use xlsynth_pir::IrBits;

/// Builds a body that consumes both exports of a preceding module.
fn imported_body(ty: BitsType, module: usize) -> Result<Body, String> {
    let mut body = Body::new(ty)?;
    body.push(Fragment::ImportedConstant {
        module,
        arg: ValueRef::X,
        combine: [BinaryOp::Xor, BinaryOp::Add],
    })?;
    Ok(body)
}

#[test]
fn qualified_constants_resolve_in_parametric_helpers_and_module_constexprs() -> Result<(), String> {
    let unsigned = BitsType {
        signed: false,
        width: Width::Parametric,
    };
    let signed = BitsType {
        signed: true,
        ..unsigned
    };
    // At 65 bits the base is 2^64+3, the private step is 7, and the derived
    // export is 2^64+10.
    let exports = ExportedConstants::new(
        IrBits::from_lsb_fn(65, |i| matches!(i, 0 | 1 | 64)),
        IrBits::from_lsb_fn(65, |i| matches!(i, 2 | 64)),
        [BinaryOp::Xor, BinaryOp::Add],
    )?;
    let exporter =
        HelperModule::new(Body::new(unsigned)?, Body::new(signed)?)?.with_exports(exports);
    let importer = HelperModule::new(imported_body(unsigned, 0)?, imported_body(signed, 0)?)?;
    let mut lanes = Vec::new();
    let mut args = Vec::new();
    let mut outputs = Vec::new();
    for width in [1, 7, 65, 129] {
        for signed in [false, true] {
            let ty = BitsType {
                signed,
                width: Width::Concrete(width),
            };
            let mut lane = Body::new(ty)?;
            lane.push(Fragment::Call {
                module: 1,
                args: [ValueRef::X, ValueRef::Y],
                explicit_parametric: signed,
            })?;
            lanes.push(lane);

            let mut lane = Body::new(ty)?;
            let expression = ConstExpr::new(
                imported_body(ty, 0)?,
                [
                    ConstInput::Literal(IrBits::from_lsb_fn(width, |i| i == 2)),
                    ConstInput::Literal(IrBits::from_lsb_fn(width, |_| false)),
                ],
            )?;
            lane.push(Fragment::Constant {
                expression: Box::new(expression),
                scope: ConstantScope::Module,
                assert_value: true,
            })?;
            lanes.push(lane);

            // ((2^64+10) ^ 4) + (2^64+3) = 2^65+17, modulo the consuming width.
            let x = XlsIrValue::from_bits(&XlsIrBits::from_lsb_is_0(
                &(0..width).map(|i| i == 2).collect::<Vec<_>>(),
            ));
            let expected = XlsIrValue::from_bits(&XlsIrBits::from_lsb_is_0(
                &(0..width)
                    .map(|i| matches!(i, 0 | 4 | 65))
                    .collect::<Vec<_>>(),
            ));
            for _ in 0..2 {
                args.extend([x.clone(), x.clone()]);
                outputs.push(expected.clone());
            }
        }
    }
    let sample = Sample::new(vec![exporter, importer], lanes)?.prepare()?;
    let expected = XlsIrValue::make_tuple(&outputs);
    for (ir, top) in [
        (&sample.expected_ir, "main"),
        (&sample.converted_ir, sample.converted_top.as_str()),
    ] {
        let package = IrPackage::parse_ir(ir, None).map_err(|err| err.to_string())?;
        let result = package
            .get_function(top)
            .map_err(|err| err.to_string())?
            .interpret(&args)
            .map_err(|err| err.to_string())?;
        assert_eq!(result, expected);
    }
    Ok(())
}

#[test]
fn imports_require_a_preceding_module_with_public_constants() -> Result<(), String> {
    let unsigned = BitsType {
        signed: false,
        width: Width::Parametric,
    };
    let signed = BitsType {
        signed: true,
        ..unsigned
    };
    let lane_ty = BitsType {
        signed: false,
        width: Width::Concrete(9),
    };
    let missing = HelperModule::new(Body::new(unsigned)?, Body::new(signed)?)?;
    assert!(Sample::new(vec![missing], vec![imported_body(lane_ty, 0)?]).is_err());

    let exports = ExportedConstants::new(
        IrBits::all_ones(9),
        IrBits::all_ones(9),
        [BinaryOp::Sub, BinaryOp::Or],
    )?;
    let self_import =
        HelperModule::new(imported_body(unsigned, 0)?, Body::new(signed)?)?.with_exports(exports);
    assert!(Sample::new(vec![self_import], vec![Body::new(lane_ty)?]).is_err());
    Ok(())
}
