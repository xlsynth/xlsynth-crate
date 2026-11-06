// SPDX-License-Identifier: Apache-2.0

use super::directed_expr_eval::{
    assert_expr_matches_dslx, assert_expr_rejected_by_both, compile_dslx_module,
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
    let source = r#"block seeded(clk, rst) -> u8[2] {
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
