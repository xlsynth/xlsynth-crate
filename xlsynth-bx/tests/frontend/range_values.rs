// SPDX-License-Identifier: Apache-2.0

use super::expr_eval::{BlockEvaluator, bx_source};
use xlsynth::dslx::{ImportData, parse_and_typecheck};
use xlsynth_bx::{check_module, parse_module};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::Type as IrType;

#[test]
fn array_range_bounds_and_values_survive_specialization() -> Result<(), String> {
    let source = r#"
        fn end(x: u32) -> u32 { x + u32:1 }
        fn ids<N: u32>() -> u32[N] { u32:0..N }
        const LANES = ids<3>();
        const_assert!(LANES == [u32:0, u32:1, u32:2]);
        const_assert!(array_size(ids<0>()) == u32:0);
        #[allow(combinational_block)]
        block sample(index: u32) -> u32 {
            let xs = LANES ++ (u32:3..=end(u32:3));
            xs[index]
        }
    "#;
    let module = check_module(&parse_module(source)?)?;
    let block = BlockEvaluator::lower(&module, source)?;
    for i in 0..=4 {
        let input = IrValue::make_ubits(32, i).map_err(|e| e.to_string())?;
        assert_eq!(block.evaluate(std::slice::from_ref(&input))?, input);
    }
    Ok(())
}

#[test]
fn array_range_infers_unsized_bounds_in_array_contexts() -> Result<(), String> {
    let prefix = "fn take(x: u8[3]) -> u8[3] { x }";
    for body in [
        "2..5",
        "let lanes: u8[3] = 2..5; lanes",
        "let lanes = 2..5; take(lanes)",
        "let lanes = 2..({ let limit = 5; limit }); lanes",
        "let lanes = 2..5; let alias = lanes; take(alias)",
        "let lanes = if index == u32:0 { 2..5 } else { u8:2..u8:5 }; lanes",
        "take(2..5)",
        "if index == u32:0 { 2..5 } else { u8:2..u8:5 }",
    ] {
        let src = format!("{prefix}\n{}", bx_source("index: u32", "u8[3]", body));
        let module = check_module(&parse_module(&src)?).map_err(|e| format!("{body}: {e}"))?;
        let block = BlockEvaluator::lower(&module, &src)?;
        let expected =
            IrValue::parse_typed("[bits[8]:2, bits[8]:3, bits[8]:4]").map_err(|e| e.to_string())?;
        for index in 0..2 {
            assert_eq!(
                block.evaluate(&[IrValue::make_ubits(32, index).map_err(|e| e.to_string())?])?,
                expected,
                "{body}"
            );
        }
    }
    Ok(())
}

#[test]
fn array_range_inferred_when_indexed_and_empty_when_bounds_equal() -> Result<(), String> {
    for body in [
        "(2..5)[index]",
        "let lanes = 2..5; lanes[index]",
        "(if index == u32:2 { 2..5 } else { 3..6 })[index]",
    ] {
        let src = bx_source("index: u32", "u8", body);
        let block = BlockEvaluator::lower(&check_module(&parse_module(&src)?)?, &src)?;
        let actual = block.evaluate(&[IrValue::make_ubits(32, 2).map_err(|e| e.to_string())?])?;
        assert_eq!(
            actual,
            IrValue::make_ubits(8, 4).map_err(|e| e.to_string())?,
            "{body}"
        );
    }
    for body in ["s8:-4..s8:-4", "let xs = -4..-4; xs"] {
        let src = bx_source("", "s8[0]", body);
        let block = BlockEvaluator::lower(&check_module(&parse_module(&src)?)?, &src)?;
        assert_eq!(
            block.evaluate(&[])?,
            IrValue::make_array_typed(IrType::Bits(8), &[]).map_err(|e| e.to_string())?
        );
    }
    Ok(())
}

#[test]
fn array_range_requires_static_compatible_bounds_and_bounded_expansion() -> Result<(), String> {
    for (body, result, diagnostic) in [
        ("u8:0..n", "u8[2]", "compile-time constants"),
        ("u8:4..=u8:3", "u8[0]", "limit precedes"),
        ("s8:1..u8:4", "s8[3]", "expected"),
        ("u8:0..u8:1", "u8[2]", "array length"),
        (
            "array_size(u32:0..=u32:0xffffffff)",
            "u32",
            "maximum concrete type size",
        ),
        (
            "array_size(uN[129]:0..uN[129]:0x10000000000000000)",
            "u32",
            "too large",
        ),
    ] {
        let src = format!("fn bad(n: u8) -> {result} {{ {body} }}");
        let err = check_module(&parse_module(&src)?).unwrap_err();
        assert!(err.contains(diagnostic), "{body}: {err}");
    }
    Ok(())
}

#[test]
fn array_range_siblings_share_element_width() -> Result<(), String> {
    for body in [
        "let x = if b { 0..2 } else { 4..6 }; x",
        "let x = [0..2, 4..6]; x[!b]",
        "let x = if b { [0, 1] } else { 4..6 }; x",
    ] {
        let src = bx_source("b: bool", "u3[2]", body);
        let module = check_module(&parse_module(&src)?).map_err(|e| format!("{body}: {e}"))?;
        let block = BlockEvaluator::lower(&module, &src)?;
        for (b, expected) in [
            (false, "[bits[3]:4, bits[3]:5]"),
            (true, "[bits[3]:0, bits[3]:1]"),
        ] {
            let expected = IrValue::parse_typed(expected).map_err(|e| e.to_string())?;
            assert_eq!(block.evaluate(&[IrValue::bool(b)])?, expected, "{body}");
        }
    }
    Ok(())
}

#[test]
fn nested_range_projections_keep_their_inferred_element_width() -> Result<(), String> {
    let expected =
        IrValue::parse_typed("[bits[8]:2, bits[8]:3, bits[8]:4]").map_err(|e| e.to_string())?;
    for body in [
        "(2..5,).0",
        "let group = (2..5,); group.0",
        "let group = (2..5,); let alias = group.0; alias",
        "let group = [(2..5,), (2..5,)]; group[index].0",
        "let group = (2..5, u3:1); take(group).0",
    ] {
        let source = format!(
            "fn take(x: (u8[3], u3)) -> (u8[3], u3) {{ x }}\n{}",
            bx_source("index: u32", "u8[3]", body)
        );
        let module = check_module(&parse_module(&source)?).map_err(|e| format!("{body}: {e}"))?;
        let block = BlockEvaluator::lower(&module, &source)?;
        assert_eq!(
            block.evaluate(&[IrValue::make_ubits(32, 1).map_err(|e| e.to_string())?])?,
            expected,
            "{body}"
        );
    }
    for source in [
        "fn f() -> (u8[3], u8[3]) { let group = (2..5, [2, 3, 4]); group }",
        "fn f(x: u8[1]) -> u8[3][1] { let group = (2..5,); map(x, |_| -> u8[3] { group.0 }) }",
    ] {
        let error = check_module(&parse_module(source)?).unwrap_err();
        assert!(error.contains("expected"), "{source}: {error}");
    }
    Ok(())
}

#[test]
fn array_range_supplies_generic_shape_and_supports_boolean_endpoints() -> Result<(), String> {
    let source = r#"
        fn offsets() -> s8[2] { 0xfe..=0xff }
        const_assert!(offsets() == [s8:-2, s8:-1]);
        fn no_lanes() -> s8[0] { let xs = -4..-4; xs }
        const_assert!(array_size(no_lanes()) == u32:0);
        fn shape<W: u32, N: u32>(x: uN[W][N]) -> (u32, u32) { (W, N) }
        const_assert!(shape(2..=5) == (u32:3, u32:4));
        const_assert!((false..=true) == [false, true]);
    "#;
    check_module(&parse_module(source)?)?;
    Ok(())
}

#[test]
fn constexpr_range_bounds_only_require_callbacks_when_specialized() -> Result<(), String> {
    let src = r#"
        fn only_zero<N: u32>(x: u8) -> u8 { const_assert!(N == u32:0); x }
        fn unused<F: fn(u8) -> u8>(x: u8) -> u8 {
            let lanes = u8:1..u8:2;
            only_zero<u32:1>(x)
        }
        fn two() -> u32 { u32:2 }
        fn lanes<F: fn() -> u32>() -> u32[2] { u32:0..F() }
        fn width<N: u32>() -> u32 { N }
        fn by_width() -> u32[2] { u32:0..width<2>() }
        const_assert!(lanes<two>() == [u32:0, u32:1]);
    "#;
    let module = check_module(&parse_module(src)?)?;
    assert!(
        !module
            .functions()
            .iter()
            .any(|f| f.name.starts_with("width__"))
    );
    Ok(())
}

#[test]
fn range_length_requires_constexpr_bounds_even_when_the_array_is_unused() -> Result<(), String> {
    for bound in ["u32:3", "{ x; u32:3 }"] {
        for expr in ["u32:0..k", "k..=u32:4"] {
            let source = format!("fn f(x: bool) {{ let k = {bound}; let _ = {expr}; }}");
            let bx = check_module(&parse_module(&source)?);
            let dslx =
                parse_and_typecheck(&source, "bounds.x", "bounds", &mut ImportData::default());
            assert_eq!(bx.is_ok(), dslx.is_ok(), "{source}");
            assert_eq!(bx.is_ok(), bound == "u32:3", "{source}");
        }
    }
    Ok(())
}

#[test]
fn signed_range_arguments_unify_bounds_before_the_formal_type() -> Result<(), String> {
    let decl = "fn take(a: s1[1]) -> s1[1] { a }";
    for (body, accepted) in [
        ("0x1..=0x1", true),
        ("take(0x1..=0x1)", false),
        ("let h = 0x1; take(h..=h)", true),
    ] {
        let source = format!("{decl} fn sample() -> s1[1] {{ {body} }}");
        let dslx = parse_and_typecheck(&source, "bounds.x", "bounds", &mut ImportData::default());
        assert_eq!(dslx.is_ok(), accepted, "{source}: {:?}", dslx.err());
        let bx = check_module(&parse_module(&source)?);
        assert_eq!(bx.is_ok(), accepted, "{source}: {:?}", bx.err());
    }
    Ok(())
}

#[test]
fn bound_arithmetic_uses_the_subject_width_before_a_projection_or_call() -> Result<(), String> {
    for (ty, body, valid) in [
        ("u8[3]", "2..(2+3)", true),
        ("u8", "(2..(2+3))[u32:0]", false),
        ("u8[3]", "take(2..(2+3))", false),
        (
            "u32",
            "array_size(if true { 2..(2+3) } else { 2..5 })",
            false,
        ),
        (
            "u32",
            "array_size(if true { u8:2..u8:5 } else { 2..(2+3) })",
            false,
        ),
        ("u32", "array_size([2..5, 2..(2+3)])", false),
        (
            "u32",
            "array_size(if true { 2..(1+2) } else { 2..3 })",
            true,
        ),
        (
            "u8[3]",
            "take(if true { 2..(2+3) } else { u8:2..u8:5 })",
            false,
        ),
        (
            "u8[3]",
            "take(for (i, acc) in u32:0..u32:1 { acc }(2..(2+3)))",
            false,
        ),
        (
            "u8[3]",
            "take(unroll_for! (i, acc) in u32:0..u32:1 { acc }(2..(2+3)))",
            false,
        ),
        (
            "u8[3]",
            "take(for (i, acc) in u32:0..u32:1 { acc }(2..5))",
            true,
        ),
    ] {
        let source = format!("fn take(a:u8[3])->u8[3] {{ a }} fn f()->{ty} {{ {body} }}");
        let native = parse_and_typecheck(&source, "bounds.x", "bounds", &mut ImportData::default());
        assert_eq!(native.is_ok(), valid, "{source}: {:?}", native.err());
        let bx = check_module(&parse_module(&source)?);
        assert_eq!(bx.is_ok(), valid, "{source}: {:?}", bx.err());
    }
    let source = r#"
        const LANES = if true { (2+2)..=(3+2) } else { u8:4..=u8:5 };
        const_assert!((LANES[u32:0] as u8) == u8:0);
        const_assert!((LANES[u32:1] as u8) == u8:1);
    "#;
    parse_and_typecheck(source, "bounds.x", "bounds", &mut ImportData::default())
        .map_err(|e| e.to_string())?;
    check_module(&parse_module(source)?)?;
    Ok(())
}

#[test]
fn a_range_keeps_existing_unary_precedence_for_other_block_expressions() -> Result<(), String> {
    let body = "let ids = u8:0..u8:2; -if flag { u8:3 } else { u8:1 } + ids[u32:1]";
    let source = bx_source("flag: bool", "u8", body);
    let block = BlockEvaluator::lower(&check_module(&parse_module(&source)?)?, &source)?;
    for (flag, expected) in [(true, 254), (false, 0)] {
        assert_eq!(
            block.evaluate(&[IrValue::bool(flag)])?,
            IrValue::make_ubits(8, expected).map_err(|e| e.to_string())?
        );
    }
    Ok(())
}

#[test]
fn indexed_range_takes_the_common_operand_width_in_either_order() -> Result<(), String> {
    for predicate in ["(2..5)[index] == u8:3", "u8:3 == (2..5)[index]"] {
        let source = bx_source("index: u32", "bool", predicate);
        let block = BlockEvaluator::lower(&check_module(&parse_module(&source)?)?, &source)?;
        for index in 0..3 {
            let value = IrValue::make_ubits(32, index).map_err(|e| e.to_string())?;
            assert_eq!(block.evaluate(&[value])?, IrValue::bool(index == 1));
        }
    }
    Ok(())
}
