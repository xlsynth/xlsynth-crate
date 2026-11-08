// SPDX-License-Identifier: Apache-2.0

use xlsynth::dslx::{self, ImportData, MatchableModuleMember};
use xlsynth_bx::ast::Type;
use xlsynth_bx::{ImportOptions, check_module, parse_module};
use xlsynth_pir::IrValue;

use super::directed_expr_eval::{
    assert_expr_matches_dslx, assert_expr_rejected_by_both, compile_dslx_module,
};
use super::expr_eval::{ExpressionEvaluator, bx_source, dslx_source};

#[test]
fn unroll_for_folds_in_range_order_with_dynamic_seed_and_local_lets() {
    for annotation in [": (u32, u8)", ""] {
        let body = format!(
            r#"
            let first = u32:2;
            let limit = first + u32:3;
            let folded = unroll_for! (i, acc){annotation} in first..limit {{
                let scaled: u8 = acc * u8:3;
                let step = i as u8;
                scaled - step
            }}(seed);
            folded
        "#
        );
        assert_expr_matches_dslx(
            "seed: u8",
            "u8",
            &body,
            [1, 7].into_iter().zip([252, 158]).map(|(seed, result)| {
                (
                    vec![IrValue::make_ubits(8, seed).unwrap()],
                    IrValue::make_ubits(8, result).unwrap(),
                )
            }),
        );
    }
}

#[test]
fn unroll_for_arrays_use_the_previous_iteration_and_preserve_empty_seed() {
    let body = r#"
        let seeded = [x, y];
        let empty = unroll_for! (i, acc): (u32, u8[2]) in u32:2..u32:2 {
            update(acc, i % u32:2, u8:0)
        }(seeded);
        let folded = unroll_for! (i, acc) in u32:1..u32:4 {
            let other = acc[(i + u32:1) % u32:2];
            update(acc, i % u32:2, other + (i as u8))
        }(empty);
        (empty, folded)
    "#;
    assert_expr_matches_dslx(
        "x: u8, y: u8",
        "(u8[2], u8[2])",
        body,
        [(
            vec![
                IrValue::make_ubits(8, 2).unwrap(),
                IrValue::make_ubits(8, 8).unwrap(),
            ],
            IrValue::parse_typed("([bits[8]:2, bits[8]:8], [bits[8]:5, bits[8]:8])").unwrap(),
        )],
    );
}

#[test]
fn unrolled_tuple_patterns_bind_and_shadow_each_iteration() {
    for (count, expected) in [(0, 5), (3, 8)] {
        let body = format!(
            r#"
            unroll_for! (i, a) in u32:0..u32:{count} {{
                let (left, (_, .., tail)): (u8, (u8, u8, u8)) = (a, (x, a, i as u8));
                let (left, _) = (left + tail, x);
                left
            }}(x)
            "#
        );
        assert_expr_matches_dslx(
            "x: u8",
            "u8",
            &body,
            [(
                vec![IrValue::make_ubits(8, 5).unwrap()],
                IrValue::make_ubits(8, expected).unwrap(),
            )],
        );
    }
}

#[test]
fn empty_unroll_resolves_pattern_names_before_skipping_type_checks() {
    let fold = |limit, body| format!("unroll_for! (_, a) in u32:0..u32:{limit} {{ {body} }}(x)");
    // DSLX never elaborates the scalar-to-tuple mismatch in an empty fold.
    let mismatch = "let (kept, _) = a; kept";
    assert_expr_matches_dslx(
        "x: u8",
        "u8",
        &fold(0, mismatch),
        [(
            vec![IrValue::make_ubits(8, 9).unwrap()],
            IrValue::make_ubits(8, 9).unwrap(),
        )],
    );
    assert_expr_rejected_by_both("x: u8", "u8", &fold(1, mismatch));
    assert_expr_rejected_by_both("x: u8", "u8", &fold(0, "let (kept, _) = (kept, a); kept"));
}

#[test]
fn guarded_destructuring_restores_outer_names_and_rejects_lambda_captures() -> Result<(), String> {
    let source = r#"
block sample(clk, en:bool, x:u8)->bool {
    reg state:u8;
    let (selected, _) = (en, x);
    if selected {
        let (selected, saved) = (x, state);
        let folded = unroll_for! (_, a) in u8:0..u8:0 {
            let (offset, _) = (selected, saved);
            let values = map([a], |v| v + v);
            values[u1:0]
        }(selected);
        state := folded + saved;
    }
    selected
}
"#;
    for limit in ["u8:0..u8:0", "u8:0..u8:1"] {
        let source = source.replace("u8:0..u8:0", limit);
        check_module(&parse_module(&source)?)?;
        for name in ["offset", "saved", "selected"] {
            let captures = source.replace("|v| v + v", &format!("|v| v + {name}"));
            let error = check_module(&parse_module(&captures)?).unwrap_err();
            assert!(
                error.contains(&format!("cannot capture enclosing value '{name}'")),
                "{error}"
            );
        }
    }
    Ok(())
}

#[test]
fn unrolled_tuple_projections_and_call_operands_charge_expanded_ir_types() -> Result<(), String> {
    let mut types = String::from("struct Tree0 { bit:u1 }\n");
    for depth in 0..17 {
        types.push_str(&format!(
            "struct Tree{} {{ left:Tree{depth}, right:Tree{depth} }}\n",
            depth + 1
        ));
    }
    let projection = r#"
block sample(clk, pair:(Tree17,), x:u8)->u8 {
    unroll_for! (_, a) in u8:0..u8:2 { let (selected,) = pair; a }(x)
}
"#;
    let call = r#"
fn copy(value:Tree17)->Tree17 { value }
block sample(clk, value:Tree17)->bool {
    let (pair,) = ((1, value),);
    unroll_for! (_, a) in u8:0..u8:2 { pair == (1, copy(value)) }(false)
}
"#;
    for block in [projection, call] {
        let source = format!("{types}{block}");
        check_module(&parse_module(&source)?)?;
        // Each tuple_index or captured invoke retains Tree17's 393,215 IR type
        // nodes.
        let large = source.replace("u8:0..u8:2", "u8:0..u8:3");
        let error = check_module(&parse_module(&large)?).unwrap_err();
        assert!(error.contains("expansion limit"), "{error}");
    }
    Ok(())
}

#[test]
fn unroll_for_preserves_indices_above_u64() -> Result<(), String> {
    let fold = r#"unroll_for! (i, acc) in
        uN[96]:0x10000000000000001..uN[96]:0x10000000000000004 {
            acc * u8:3 - (i as u8) + i[u32:64+:u8]
        }(u8:1)"#;
    // Native constexpr evaluation supports this base. IR conversion of a
    // runtime loop currently narrows its iteration values to u64.
    let native = format!("const FOLDED = {fold}; fn sample() -> u8 {{ FOLDED }}");
    let evaluator = ExpressionEvaluator::compile_block(
        &bx_source("", "u8", fold),
        &ImportOptions::default(),
        compile_dslx_module(&native, &[]),
    )?;
    assert_eq!(evaluator.compare(&[])?, IrValue::make_ubits(8, 22).unwrap());
    Ok(())
}

#[test]
fn unroll_for_requires_constant_ranges_and_compatible_binders() {
    for expr in [
        "unroll_for! (i, a) in u32:0..n { a + (i as u8) }(x)",
        "unroll_for! (i, a) in n..u32:2 { a + (i as u8) }(x)",
        "unroll_for! (i, a) in u32:3..u32:2 { a + (i as u8) }(x)",
        "unroll_for! (i, a): (u16, u8) in u32:0..u32:2 { a + (i as u8) }(x)",
        "unroll_for! (i, a): (u32, u16) in u32:0..u32:2 { a + (i as u16) }(x)",
        "unroll_for! (i, a) in u32:0..u32:2 { a as u16 }(x)",
        "unroll_for! (i, a) in u32:0..u32:3 { [a, x][i] }(x)",
        "unroll_for! (i, a) in u32:0..[u32:1, u32:2][u32:2] { a }(x)",
        "unroll_for! (_, a) in u8:0..u8:0 { _ }(x)",
        "unroll_for! (i, a) in u8:0..u8:0 { unroll_for! (j, j) in u8:0..u8:1 { a }(x) }(x)",
    ] {
        assert_expr_rejected_by_both("x: u8, n: u32", "u8", expr);
    }
}

#[test]
fn unroll_for_implicit_seed_infers_from_concrete_siblings_in_either_order() {
    let fold = "(unroll_for! (i, a) in u8:0..u8:2 { a }(3))";
    let params = "x: u8, b: bool";
    for sibling in ["x", "1"] {
        for (initializer, result, expected) in [
            (format!("{fold} + {sibling}"), "v", 8),
            (format!("{sibling} + {fold}"), "v", 8),
            (format!("[{fold}, {sibling}]"), "v[u32:0]", 3),
            (format!("[{sibling}, {fold}]"), "v[u32:0]", 5),
            (format!("if b {{ {fold} }} else {{ {sibling} }}"), "v", 3),
            (format!("if b {{ {sibling} }} else {{ {fold} }}"), "v", 5),
        ] {
            if sibling == "1" {
                assert_expr_rejected_by_both(params, "u8", &format!("let v = {initializer}; x"));
            } else {
                assert_expr_matches_dslx(
                    params,
                    "u8",
                    &format!("let v = {initializer}; {result}"),
                    [(
                        vec![IrValue::make_ubits(8, 5).unwrap(), IrValue::bool(true)],
                        IrValue::make_ubits(8, expected).unwrap(),
                    )],
                );
            }
        }
    }
}

#[test]
fn unroll_for_widens_an_implicit_index_to_its_arithmetic_operand() -> Result<(), String> {
    for (sum, cast_sum) in [
        ("a + i", "a + (i as u8)"),
        ("i + a", "(i as u8) + a"),
        (
            "let pair = (i, i); a + pair.0",
            "let pair = (i, i); a + (pair.0 as u8)",
        ),
        (
            "let (.., step) = (a, i); a + step",
            "let (.., step) = (a, i); a + (step as u8)",
        ),
        (
            "let xs = [i, i]; a + xs[u8:0]",
            "let xs = [i, i]; a + (xs[u8:0] as u8)",
        ),
        (
            "let j = if a == u8:255 { 0 } else { i }; a + j",
            "let j = if a == u8:255 { 0 } else { i }; a + (j as u8)",
        ),
    ] {
        let fold =
            |sum: &str, seed: &str| format!("unroll_for! (i, a) in 0..4 {{ {sum} }}({seed})");
        let body = fold(sum, "x");
        let block = bx_source("x: u8", "u8", &body);
        let bx = check_module(&parse_module(&block)?)?;
        // Typecheck the runtime function and inspect the type of the same
        // expression in an unannotated DSLX constant. Native IR conversion
        // currently needs an explicit cast for this implicit index use.
        let source = format!(
            "const FOLDED = {};\n{}",
            fold(sum, "u8:0"),
            dslx_source("x: u8", "u8", &body)
        );
        let mut imports = ImportData::new(None, &[]);
        let native = dslx::parse_and_typecheck(&source, "oracle.x", "oracle", &mut imports)
            .map_err(|e| format!("{e}\n{source}"))?;
        let Some(MatchableModuleMember::ConstantDef(folded)) =
            native.get_module().get_member(0).to_matchable()
        else {
            return Err("missing native folded constant".into());
        };
        let native_type = native.get_type_info().get_type_for_constant_def(&folded);
        let (signed, width) = native_type
            .is_bits_like()
            .ok_or("expected a native bits type")?;
        assert_eq!(bx.blocks()[0].output.ty, Type::Bits { signed, width });
        let explicit = dslx_source("x: u8", "u8", &fold(cast_sum, "x"));
        let evaluator = ExpressionEvaluator::compile_block(
            &block,
            &ImportOptions::default(),
            compile_dslx_module(&explicit, &[]),
        )?;
        for x in [11, 253] {
            assert_eq!(
                evaluator.compare(&[IrValue::make_ubits(8, x).unwrap()])?,
                IrValue::make_ubits(8, (x + 6) % 256).unwrap()
            );
        }
    }
    Ok(())
}

#[test]
fn unroll_for_uses_iteration_values_for_shifts_and_width_dimensions() {
    for (expr, expected) in [
        ("unroll_for! (i, a) in u8:9..u8:10 { a << i }(x)", 0),
        (
            "unroll_for! (i, a) in 1..3 { let v = x[u32:0+:uN[i]]; a + (v as u8) }(x)",
            11,
        ),
    ] {
        assert_expr_matches_dslx(
            "x: u8",
            "u8",
            expr,
            [(
                vec![IrValue::make_ubits(8, 7).unwrap()],
                IrValue::make_ubits(8, expected).unwrap(),
            )],
        );
    }
}
