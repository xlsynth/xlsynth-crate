// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrPackage, XlsIrBits, XlsIrValue};
use xlsynth_dslx_fuzzer::Sample;
use xlsynth_dslx_fuzzer::ast::{BinaryOp, BitsType, Body, Fragment, HelperModule, ValueRef, Width};
use xlsynth_dslx_fuzzer::imported_struct::StructExport;

/// Builds a body that replaces one field and returns the XOR of both fields.
fn updated_body(signed: bool, update_hi: bool) -> Result<Body, String> {
    let mut body = Body::new(BitsType {
        signed,
        width: Width::Parametric,
    })?;
    body.push(Fragment::ImportedStructUpdate {
        module: 1,
        args: [ValueRef::X, ValueRef::Y, ValueRef::X],
        update_hi,
        combine: BinaryOp::Xor,
    })?;
    Ok(body)
}

fn exporting_helper(export: StructExport) -> Result<HelperModule, String> {
    let body = |signed| {
        Body::new(BitsType {
            signed,
            width: Width::Parametric,
        })
    };
    Ok(HelperModule::new(body(false)?, body(true)?)?.with_struct_export(export))
}

#[test]
fn runtime_splat_update_through_aliases_preserves_the_other_field() -> Result<(), String> {
    let helpers = vec![
        exporting_helper(StructExport::Definition { widths: [5, 65] })?,
        exporting_helper(StructExport::Alias { module: 0 })?,
        HelperModule::new(updated_body(false, false)?, updated_body(true, true)?)?,
    ];
    let mut lanes = Vec::new();
    let mut args = Vec::new();
    let mut outputs = Vec::new();
    for width in [1, 9, 129] {
        for signed in [false, true] {
            let ty = BitsType {
                signed,
                width: Width::Concrete(width),
            };
            let mut body = Body::new(ty)?;
            body.push(Fragment::Call {
                module: 2,
                args: [ValueRef::X, ValueRef::Y],
                explicit_parametric: signed,
            })?;
            lanes.push(body);

            // Include each lane's sign bit and bits on either side of the
            // field boundaries to test truncation and unsigned extension.
            let x = |bit| bit == width - 1 || matches!(bit, 0 | 4 | 5 | 64 | 65);
            let y = |bit| bit == width - 1 || matches!(bit, 1 | 3 | 7 | 60 | 65);
            let bits_value =
                |bits: Vec<bool>| XlsIrValue::from_bits(&XlsIrBits::from_lsb_is_0(&bits));
            args.push(bits_value((0..width).map(x).collect()));
            args.push(bits_value((0..width).map(y).collect()));
            let expected = (0..width).map(|bit| {
                if signed {
                    (bit < 5 && x(bit)) ^ (bit < 65 && (y(bit) ^ x(bit)))
                } else {
                    // Updating lo with x ^ x leaves only the preserved hi.
                    bit < 65 && y(bit)
                }
            });
            outputs.push(bits_value(expected.collect()));
        }
    }
    let sample = Sample::new(helpers, lanes)?.prepare()?;
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
fn struct_exports_require_earlier_definitions_with_positive_field_widths() -> Result<(), String> {
    for export in [
        StructExport::Alias { module: 0 },
        StructExport::Definition { widths: [65, 0] },
    ] {
        let lane = Body::new(BitsType {
            signed: false,
            width: Width::Concrete(9),
        })?;
        assert!(Sample::new(vec![exporting_helper(export)?], vec![lane]).is_err());
    }
    Ok(())
}
