// SPDX-License-Identifier: Apache-2.0

use super::directed_expr_eval::{
    assert_expr_matches_dslx, assert_expr_rejected_by_both, compile_dslx, compile_dslx_module,
};
use super::expr_eval::{ExpressionEvaluator, bx_source, dslx_source};
use xlsynth_bx::check::CheckedStmt;
use xlsynth_bx::{ImportOptions, check_module, parse_module};
use xlsynth_pir::IrValue;

#[test]
fn zero_and_update_preserve_nested_struct_layout_and_wide_indices()
-> Result<(), Box<dyn std::error::Error>> {
    let declaration = "struct Row { words: s8[2], flags: (bool, u2) }";
    let params = "i: uN[96], row: Row";
    let result = "Row[2]";
    let body = "update(zero!<Row[2]>(), i, row)";
    let evaluator = ExpressionEvaluator::compile_block(
        &format!("{declaration}\n{}", bx_source(params, result, body)),
        &ImportOptions::default(),
        compile_dslx_module(
            &format!("{declaration}\n{}", dslx_source(params, result, body)),
            &[],
        ),
    )?;
    let zero = IrValue::parse_typed("([bits[8]:0, bits[8]:0], (bits[1]:0, bits[2]:0))")?;
    let row = IrValue::parse_typed("([bits[8]:-7, bits[8]:31], (bits[1]:1, bits[2]:3))")?;
    for (index, expected) in [
        ("0", [&row, &zero]),
        ("1", [&zero, &row]),
        ("2", [&zero, &zero]),
        ("0x1_0000_0000_0000_0001", [&zero, &zero]),
    ] {
        let index = IrValue::parse_typed(&format!("bits[96]:{index}"))?;
        assert_eq!(
            evaluator.compare(&[index, row.clone()])?,
            IrValue::make_array(&expected.map(Clone::clone))?
        );
    }
    Ok(())
}

#[test]
fn update_infers_its_arguments_independently() {
    assert_expr_matches_dslx(
        "",
        "u2[2]",
        "let i = 1; update([1, 2], i, u2:3)",
        [(
            vec![],
            IrValue::parse_typed("[bits[2]:1, bits[2]:3]").unwrap(),
        )],
    );
    for body in [
        "update(x, 1, u8:2)",
        "update(x, u2:1, 2 + 1)",
        "update(x, s2:1, u8:2)",
        "update(x, u2:1, s8:2)",
        "update(x, u2:1)",
        "update(u8:2, u2:1, u8:2)",
        "let update = u2:1; update(x, u2:1, u8:2)",
    ] {
        assert_expr_rejected_by_both("x: u8[2]", "u8[2]", body);
    }
}

#[test]
fn array_update_in_constant_contexts_requires_an_in_bounds_index() {
    let source = r#"block seeded(clk) -> u8[2] {
        reg q: u8[2] = update(zero!<u8[2]>(), u3:1, u8:23);
        q := q;
        q
    }"#;
    let checked = check_module(&parse_module(source).unwrap()).unwrap();
    let CheckedStmt::Reg { reset, .. } = &checked.blocks()[0].statements[0] else {
        panic!("expected q register");
    };
    assert_eq!(
        reset.as_ref().unwrap(),
        &IrValue::parse_typed("[bits[8]:0, bits[8]:23]").unwrap()
    );
    let bad_reset = source.replace("u3:1", "u3:2");
    assert!(check_module(&parse_module(&bad_reset).unwrap()).is_err());
    assert_expr_rejected_by_both(
        "x: u8",
        "u1",
        "let widths = update([u32:1], u1:1, u32:2); x[0+:uN[widths[0]]]",
    );
    // This constant-input call occurs in an ordinary runtime expression.
    assert_expr_matches_dslx(
        "",
        "u8[2]",
        "update([u8:1, u8:2], u2:2, u8:9)",
        [(
            vec![],
            IrValue::parse_typed("[bits[8]:1, bits[8]:2]").unwrap(),
        )],
    );
}

#[test]
fn declared_update_extern_keeps_its_signature() {
    let source = "extern fn update(x: u8) -> u1; block b(clk, x: u8) -> u1 { update(x) }";
    check_module(&parse_module(source).unwrap()).unwrap();
}

#[test]
fn tuple_indices_update_nested_arrays_without_wrapping_wide_indices()
-> Result<(), Box<dyn std::error::Error>> {
    let original = "[[bits[8]:1, bits[8]:2], [bits[8]:3, bits[8]:4]]";
    let changed = "[[bits[8]:1, bits[8]:2], [bits[8]:9, bits[8]:4]]";
    assert_expr_matches_dslx(
        "x: u8[2][2], row: u1, col: uN[96], value: u8",
        "u8[2][2]",
        "update(x, (row, col), value)",
        [
            ("bits[96]:0", changed),
            ("bits[96]:2", original),
            ("bits[96]:0x1_0000_0000_0000_0000", original),
        ]
        .into_iter()
        .map(|(index, result)| {
            Ok((
                [original, "bits[1]:1", index, "bits[8]:9"]
                    .into_iter()
                    .map(IrValue::parse_typed)
                    .collect::<Result<Vec<_>, _>>()?,
                IrValue::parse_typed(result)?,
            ))
        })
        .collect::<Result<Vec<_>, xlsynth_pir::ValueError>>()?,
    );
    Ok(())
}

#[test]
fn array_rev_zip_and_array_size_preserve_element_types() -> Result<(), Box<dyn std::error::Error>> {
    assert_expr_matches_dslx(
        "x: u8[3], y: s4[3]",
        "(u8[3], (u8, s4)[3], u32)",
        "(array_rev(x), zip(x, y), array_size(y))",
        [(
            vec![
                IrValue::parse_typed("[bits[8]:2, bits[8]:4, bits[8]:8]")?,
                IrValue::parse_typed("[bits[4]:-1, bits[4]:-2, bits[4]:-3]")?,
            ],
            IrValue::parse_typed(
                "([bits[8]:8, bits[8]:4, bits[8]:2], [(bits[8]:2, bits[4]:-1), (bits[8]:4, bits[4]:-2), (bits[8]:8, bits[4]:-3)], bits[32]:3)",
            )?,
        )],
    );
    Ok(())
}

#[test]
fn array_slice_uses_witness_length_and_repeats_the_last_element()
-> Result<(), Box<dyn std::error::Error>> {
    let input = IrValue::parse_typed("[bits[8]:1, bits[8]:2]")?;
    let in_bounds = IrValue::parse_typed("[bits[8]:1, bits[8]:2, bits[8]:2]")?;
    let clamped = IrValue::parse_typed("[bits[8]:2, bits[8]:2, bits[8]:2]")?;
    assert_expr_matches_dslx(
        "x: u8[2], start: uN[96]",
        "u8[3]",
        "array_slice(x, start, [0, 0, 0])",
        [
            (
                vec![input.clone(), IrValue::parse_typed("bits[96]:0")?],
                in_bounds,
            ),
            (
                vec![
                    input,
                    IrValue::parse_typed("bits[96]:0x1_0000_0000_0000_0000")?,
                ],
                clamped,
            ),
        ],
    );
    Ok(())
}

#[test]
fn array_slice_infers_a_shared_literal_element_type() -> Result<(), Box<dyn std::error::Error>> {
    // The bundled compiler typechecks the unsized source literals but lowers
    // them at their minimum widths. Give the reference source explicit types
    // so the comparison uses the common element type.
    for (params, result, body, reference, args, expected) in [
        (
            "a: u8[2]",
            "u8[2]",
            "array_slice([0, 0], u2:1, a)",
            "array_slice([u8:0, u8:0], u2:1, a)",
            vec![IrValue::parse_typed("[bits[8]:19, bits[8]:31]")?],
            "[bits[8]:0, bits[8]:0]",
        ),
        (
            "a: u8[3]",
            "u8[3]",
            "array_slice({ let x = 2; [1, x] }, u2:0, a)",
            "array_slice([u8:1, u8:2], u2:0, a)",
            vec![IrValue::parse_typed(
                "[bits[8]:19, bits[8]:31, bits[8]:43]",
            )?],
            "[bits[8]:1, bits[8]:2, bits[8]:2]",
        ),
        (
            "",
            "u3[3]",
            "array_slice([1, 2], u2:0, [0, 7, 0])",
            "array_slice([u3:1, u3:2], u2:0, [0, 7, 0])",
            vec![],
            "[bits[3]:1, bits[3]:2, bits[3]:2]",
        ),
    ] {
        let evaluator = ExpressionEvaluator::compile_block(
            &bx_source(params, result, body),
            &ImportOptions::default(),
            compile_dslx(params, result, reference),
        )?;
        assert_eq!(evaluator.compare(&args)?, IrValue::parse_typed(expected)?);
    }
    Ok(())
}

#[test]
fn array_slice_rejects_explicit_element_mismatches_and_bare_empty_arrays() {
    assert_expr_rejected_by_both("a: u8[2]", "u8[2]", "array_slice([u4:0, 0], u2:1, a)");
    assert_expr_rejected_by_both("a: u8[2]", "u8[0]", "array_slice(a, u2:1, [])");
}
