// SPDX-License-Identifier: Apache-2.0

use super::directed_expr_eval::{
    assert_expr_matches_dslx, assert_expr_rejected_by_both, compile_dslx_module,
};
use super::expr_eval::{BlockEvaluator, ExpressionEvaluator, bx_source, dslx_source};
use std::error::Error;
use xlsynth::dslx::{self, ImportData};
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, compile, compile_with_options, empty_package,
    parse_module,
};
use xlsynth_pir::IrValue;
use xlsynth_pir::block_inline::inline_all_blocks_in_package;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir::NodePayload;
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_pir::libxls_bridge::value_to_libxls;

#[test]
fn specialized_fold_captures_a_child_output() -> Result<(), Box<dyn Error>> {
    let source = r#"
block offset(clk, bias: u8) -> u8 { bias }
block accumulate<T>(clk, data: T[2], bias: u8) -> T {
    inst adjust: offset;
    adjust.bias := bias;
    for (word, total): (T, T) in data {
        total + word + (adjust.out as T)
    }(zero!<T>())
}
pub block wrapper(clk, data: u16[2], bias: u8) -> u16 {
    let sum = accumulate(data, bias);
    sum
}
"#;
    let mut package = compile(source, empty_package("capture"), ResetBehavior::default())?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    inline_all_blocks_in_package(&mut package)?;
    let function = combinational_block_to_fn(package.get_top_block().unwrap())?;
    let args = [
        IrValue::parse_typed("[bits[16]:100, bits[16]:200]")?,
        IrValue::make_ubits(8, 3)?,
    ];
    assert_eq!(
        eval_fn_in_package_value(&package, &function, &args),
        IrValue::make_ubits(16, 306)?
    );
    Ok(())
}

#[test]
fn counted_loop_consumes_imported_enumerate_results() -> Result<(), Box<dyn Error>> {
    let params = "xs: u8[3], seed: u8";
    let body = "for ((i, item), acc) in std::enumerate(xs) { acc + (i as u8) * item }(seed)";
    let bx = format!("use std;\n{}", bx_source(params, "u8", body));
    let package = compile_with_options(
        &bx,
        empty_package("enumerated"),
        ResetBehavior::default(),
        &ImportOptions::default(),
    )?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    let function = combinational_block_to_fn(package.get_top_block().unwrap())?;
    let args = [
        IrValue::parse_typed("[bits[8]:1,bits[8]:3,bits[8]:4]")?,
        IrValue::make_ubits(8, 2)?,
    ];
    let actual = eval_fn_in_package_value(&package, &function, &args);
    assert_eq!(actual, IrValue::make_ubits(8, 13)?);
    let dslx = format!("import std;\n{}", dslx_source(params, "u8", body));
    let native = compile_dslx_module(&dslx, &[])?;
    let native_args = args
        .iter()
        .map(value_to_libxls)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(native.interpret(&native_args)?, value_to_libxls(&actual)?);
    Ok(())
}

#[test]
fn counted_ranges_capture_inputs_and_preserve_iteration_order() -> Result<(), Box<dyn Error>> {
    let body = r#"
        let bias = x ^ u8:1;
        let first = u32:2;
        for (i, acc) in first..(first + u32:3) {
            let term = (i as u8) + bias;
            acc * u8:3 - term
        }(x)
    "#;
    // x=2, bias=3 gives accumulator values 1, 253, 240.
    assert_expr_matches_dslx(
        "x: u8",
        "u8",
        body,
        [(
            vec![IrValue::make_ubits(8, 2)?],
            IrValue::make_ubits(8, 240)?,
        )],
    );
    Ok(())
}

#[test]
fn annotated_element_supplies_signedness_to_an_inferred_array_binding() -> Result<(), Box<dyn Error>>
{
    for body in [
        "let xs = [0xff]; for (e, a): (s8, s8) in xs { a + e }(x)",
        "for (e, a): (s8, s8) in { let xs = [0xff]; xs } { a + e }(x)",
    ] {
        assert_expr_matches_dslx(
            "x: s8",
            "s8",
            body,
            [(vec![IrValue::make_ubits(8, 3)?], IrValue::make_ubits(8, 2)?)],
        );
    }
    assert_expr_rejected_by_both(
        "x: s8",
        "s8",
        "let xs = [u8:255]; for (e, a): (s8, s8) in xs { a + e }(x)",
    );
    Ok(())
}

#[test]
fn annotated_element_types_projected_arrays_and_nested_loop_carries() -> Result<(), Box<dyn Error>>
{
    for iterable in [
        "([0xff], ()).0",
        "[[0xff]][u32:0]",
        "(for (_, carry) in u32:0..u32:1 { carry }([0xff]))",
        "(for (_, carry) in u32:0..u32:1 { carry }({ let seed = [0xff]; seed }))",
    ] {
        let fold = format!("for (e, a): (s8, s8) in {iterable} {{ a + e }}(x)");
        assert_expr_matches_dslx(
            "x: s8",
            "s8",
            &fold,
            [(vec![IrValue::make_ubits(8, 3)?], IrValue::make_ubits(8, 2)?)],
        );
        assert_expr_rejected_by_both("x: s8", "s8", &fold.replace("0xff", "u8:0xff"));
    }
    assert_expr_rejected_by_both(
        "x: s8",
        "s8",
        "for (e, a): (s8, s8) in (for (_, carry) in u32:0..u32:1 { [u8:1] }([0xff])) { a + e }(x)",
    );
    Ok(())
}

#[test]
fn projection_and_conditional_type_scoped_counted_seeds() -> Result<(), Box<dyn Error>> {
    let projected =
        "(for (_, c) in u8:0..u8:1 { ([c.0[u8:0] >> u8:1], ()) }({ let p = ([0xff], ()); p })).0";
    let direct = "for (_, c) in u8:0..u8:1 { [c[u8:0] >> u8:1] }({ let p = [0xff]; p })";
    let conditional = format!(
        "if x == s8:0 {{ {direct} }} else {{ {} }}",
        direct.replace("0xff", "0x81")
    );
    let bound = format!("let xs = {projected}; for (e, a): (s8, s8) in xs {{ a + e }}(x)");
    assert_expr_matches_dslx(
        "x: s8",
        "s8",
        &bound,
        [(
            vec![IrValue::make_ubits(8, 0)?],
            IrValue::make_ubits(8, 127)?,
        )],
    );
    for (iterable, at_zero, at_one) in [
        (projected, 127, 128),
        (direct, 255, 0),
        (conditional.as_str(), 255, 193),
    ] {
        let fold = format!("for (e, a): (s8, s8) in {iterable} {{ a + e }}(x)");
        assert_expr_matches_dslx(
            "x: s8",
            "s8",
            &fold,
            [
                (
                    vec![IrValue::make_ubits(8, 0)?],
                    IrValue::make_ubits(8, at_zero)?,
                ),
                (
                    vec![IrValue::make_ubits(8, 1)?],
                    IrValue::make_ubits(8, at_one)?,
                ),
            ],
        );
        assert_expr_rejected_by_both("x: s8", "s8", &fold.replace("0xff", "u8:0xff"));
    }
    Ok(())
}

#[test]
fn counted_loop_accepts_a_contextual_seed_and_widens_an_implicit_index()
-> Result<(), Box<dyn Error>> {
    let body = "for (i, a) in 0..3 { a + i + x }(0)";
    let bx = bx_source("x: u8", "u8", body);
    let source = dslx_source("x: u8", "u8", body);
    dslx::parse_and_typecheck(&source, "check.x", "check", &mut ImportData::default())?;
    // The pinned native converter retains the range's minimum index width in
    // add; an explicit cast realizes the type accepted by the DSLX checker.
    let reference = dslx_source("x: u8", "u8", "for (i, a) in 0..3 { a + (i as u8) + x }(0)");
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&reference, &[]),
    )?;
    assert_eq!(
        evaluator.compare(&[IrValue::make_ubits(8, 2)?])?,
        IrValue::make_ubits(8, 9)?
    );
    assert_expr_rejected_by_both(
        "x: u8",
        "u8",
        "let unknown = for (i, a) in u32:0..3 { a + (i as u8) + x }(0); x",
    );
    Ok(())
}

#[test]
fn array_iteration_destructures_elements_and_carry_in_local_functions() -> Result<(), Box<dyn Error>>
{
    let helpers = r#"
fn fold(rows: (u8, u8)[3], bias: u8) -> (u8, u8) {
    for ((x, _), (sum, prev)): ((u8, u8), (u8, u8)) in rows {
        let shifted = sum * u8:2 + x + prev;
        (shifted, x + bias)
    }((u8:0, bias))
}
"#;
    let params = "left: (u8, u8)[3], right: (u8, u8)[3], pick: bool, bias: u8";
    let call = "fold(if pick { left } else { right }, bias)";
    let bx = format!("{helpers}\n{}", bx_source(params, "(u8, u8)", call));
    let dslx = format!("{helpers}\n{}", dslx_source(params, "(u8, u8)", call));
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    let left = IrValue::parse_typed(
        "[(bits[8]:1,bits[8]:9),(bits[8]:3,bits[8]:0),(bits[8]:5,bits[8]:1)]",
    )?;
    let right = IrValue::parse_typed(
        "[(bits[8]:2,bits[8]:1),(bits[8]:4,bits[8]:2),(bits[8]:6,bits[8]:3)]",
    )?;
    for (pick, expected) in [(1, "(bits[8]:34,bits[8]:7)"), (0, "(bits[8]:44,bits[8]:8)")] {
        let args = [
            left.clone(),
            right.clone(),
            IrValue::make_ubits(1, pick)?,
            IrValue::make_ubits(8, 2)?,
        ];
        assert_eq!(evaluator.compare(&args)?, IrValue::parse_typed(expected)?);
    }
    Ok(())
}

#[test]
fn nested_loops_capture_the_outer_element_and_restore_shadowed_names() -> Result<(), Box<dyn Error>>
{
    let body = r#"
        let value = bias + u8:1;
        for (value, acc) in rows {
            let folded = for (i, acc) in u8:0..u8:2 {
                acc + value + i + bias
            }(acc);
            folded
        }(value)
    "#;
    assert_expr_matches_dslx(
        "rows: u8[2], bias: u8",
        "u8",
        body,
        [(
            vec![
                IrValue::parse_typed("[bits[8]:3,bits[8]:6]")?,
                IrValue::make_ubits(8, 2)?,
            ],
            IrValue::make_ubits(8, 31)?,
        )],
    );
    Ok(())
}

#[test]
fn zero_trip_loops_check_the_body_and_return_the_seed() -> Result<(), Box<dyn Error>> {
    assert_expr_matches_dslx(
        "x: u8",
        "u8",
        "for (i, acc) in u16:3..u16:3 { acc + (i as u8) }(x)",
        [(
            vec![IrValue::make_ubits(8, 43)?],
            IrValue::make_ubits(8, 43)?,
        )],
    );
    for body in [
        "for (i, acc) in u16:3..u16:3 { u32:1 }(x)",
        "for (i, acc) in u16:3..u16:3 { missing }(x)",
        "for (i, acc) in u16:3..u16:2 { acc }(x)",
        "for (i, acc) in u8:0..x { acc }(x)",
        "for (i, acc) in x { acc }(x)",
        "for (i, acc): (u8, u8) in [] { acc + i }(x)",
        "for (i, acc): (u8, u8) in [u16:4] { acc }(x)",
        "for ((i, i), acc) in [(u16:4, u16:5)] { acc }(x)",
    ] {
        assert_expr_rejected_by_both("x: u8", "u8", body);
    }
    Ok(())
}

#[test]
fn signed_aggregate_carry_has_the_same_shift_semantics_at_every_trip_count()
-> Result<(), Box<dyn Error>> {
    for (count, expected) in [(0, 255), (1, 7), (2, 251)] {
        for (binders, body, seed, projection) in [
            (
                "(_, c): (u8, s8[1])",
                "[(c[u32:0] >> u8:1) ^ x]",
                "[0xff]",
                "q[u32:0]",
            ),
            ("(_, (v, _))", "((v >> u8:1) ^ x, ())", "(s8:-1, ())", "q.0"),
        ] {
            let fold = format!(
                "let q = for {binders} in u8:0..u8:{count} {{ {body} }}({seed}); {projection}"
            );
            assert_expr_matches_dslx(
                "x: s8",
                "s8",
                &fold,
                [(
                    vec![IrValue::make_ubits(8, 248)?],
                    IrValue::make_ubits(8, expected)?,
                )],
            );
        }
    }
    // With body-only signedness inference, the pinned DSLX bytecode shifts
    // the zero-trip seed logically (127) while the converted IR shifts it
    // arithmetically (255). BX requires the signedness before checking the
    // body.
    let implicit = "let q = for (_, c) in u8:0..u8:0 { [x] }([0xff]); q[u32:0] >> u8:1";
    assert!(check_module(&parse_module(&bx_source("x: s8", "s8", implicit))?).is_err());
    Ok(())
}

#[test]
fn inclusive_ranges_visit_the_maximum_representable_endpoint() -> Result<(), Box<dyn Error>> {
    for (range, expected) in [("u2:0..=u2:3", 6), ("u8:255..=u8:255", 255)] {
        let fold = format!("for (i, a) in {range} {{ a + (i as u8) }}(x)");
        assert_expr_matches_dslx(
            "x: u8",
            "u8",
            &fold,
            [(
                vec![IrValue::make_ubits(8, 0)?],
                IrValue::make_ubits(8, expected)?,
            )],
        );
    }
    Ok(())
}

#[test]
fn wide_and_negative_starts_preserve_source_induction_values() -> Result<(), Box<dyn Error>> {
    // These results are also exercised by the DSLX concrete-call fuzz oracle.
    // The pinned converter narrows the base when generating native counted_for
    // IR.
    for (body, expected) in [
        ("for (i, a) in s8:-3..s8:2 { a + (i as u8) }(x)", 251),
        (
            "for (i, a) in uN[96]:0x10000000000000001..uN[96]:0x10000000000000004 { a + (i as u8) + i[u32:64+:u8] }(x)",
            9,
        ),
    ] {
        let source = bx_source("x: u8", "u8", body);
        let checked = check_module(&parse_module(&source)?)?;
        let value =
            BlockEvaluator::lower(&checked, &source)?.evaluate(&[IrValue::make_ubits(8, 0)?])?;
        assert_eq!(value, IrValue::make_ubits(8, expected)?);
    }
    Ok(())
}

#[test]
fn large_counted_ranges_emit_a_single_loop_node() -> Result<(), Box<dyn Error>> {
    let source = bx_source(
        "x: u16",
        "u16",
        "for (i, acc) in u32:7..u32:100007 { acc ^ (i as u16) ^ x }(x)",
    );
    let package = compile(
        &source,
        empty_package("large_fold"),
        ResetBehavior::default(),
    )?;
    let trips: Vec<_> = package
        .members
        .iter()
        .flat_map(|m| &m.graph().nodes)
        .filter_map(|node| {
            if let NodePayload::CountedFor {
                trip_count, stride, ..
            } = &node.payload
            {
                Some((*trip_count, *stride))
            } else {
                None
            }
        })
        .collect();
    assert_eq!(trips, [(100_000, 1)]);
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn register_reset_evaluates_a_constant_counted_loop() -> Result<(), Box<dyn Error>> {
    let source = r#"
block reset_fold(clk, rst, x: u8) -> u8 {
    reg q: u8 = for (v, a) in [u8:2, u8:5, u8:7] { a * u8:3 + v }(u8:1);
    q := x;
    q
}
"#;
    let package = compile(
        source,
        empty_package("fold_reset"),
        ResetBehavior::default(),
    )?;
    assert_eq!(
        package.get_block("reset_fold").unwrap().registers[0].reset_value,
        Some(IrValue::make_ubits(8, 67)?)
    );
    Ok(())
}

#[test]
fn counted_seed_unifies_complementary_tuple_members_with_a_sibling() -> Result<(), Box<dyn Error>> {
    let body = r#"
        let pair = if take_loop {
            for (i, (a, b)) in u8:0..u8:2 { (a + i, b + y) }((0, y))
        } else { (x, 0) };
        pair
    "#;
    assert_expr_matches_dslx(
        "take_loop: bool, x: u8, y: u12",
        "(u8, u12)",
        body,
        [
            (
                vec![
                    IrValue::bool(true),
                    IrValue::make_ubits(8, 5)?,
                    IrValue::make_ubits(12, 9)?,
                ],
                IrValue::parse_typed("(bits[8]:1, bits[12]:27)")?,
            ),
            (
                vec![
                    IrValue::bool(false),
                    IrValue::make_ubits(8, 5)?,
                    IrValue::make_ubits(12, 9)?,
                ],
                IrValue::parse_typed("(bits[8]:5, bits[12]:0)")?,
            ),
        ],
    );
    Ok(())
}

#[test]
fn nested_counted_bodies_compile_on_a_small_stack() -> Result<(), String> {
    let mut body = "x".to_owned();
    for _ in 0..48 {
        body = format!("for (_, a) in u8:0..u8:1 {{ {body} }}(x)");
    }
    let runtime = format!("block folded(clk, x: u8) -> u8 {{ {body} }}");
    let reset = format!(
        "block folded(clk, rst, x: u8) -> u8 {{ reg q: u8 = {}; q := x; q }}",
        body.replace('x', "u8:7"),
    );
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || {
            for source in [runtime, reset] {
                compile(&source, empty_package("nested"), ResetBehavior::default())?;
            }
            Ok(())
        })
        .map_err(|e| e.to_string())?
        .join()
        .expect("compiler thread completed")
}

#[test]
fn counted_loop_scoped_iterable_and_seed_use_enclosing_values() -> Result<(), Box<dyn Error>> {
    let body = r#"
        for (x, acc) in { let pair = [x, x + u8:1]; pair } { acc + x }(
            { let start = x + u8:4; start }
        )
    "#;
    assert_expr_matches_dslx(
        "x: u8",
        "u8",
        body,
        [(
            vec![IrValue::make_ubits(8, 5)?],
            IrValue::make_ubits(8, 20)?,
        )],
    );
    Ok(())
}

#[test]
fn lexical_lets_in_iterables_and_seeds_hide_instances_and_externs() -> Result<(), String> {
    let source = r#"
block child(clk, z: u8) -> u8 { z }
pub block wrapper(clk, x: u8) -> u8 {
    inst pair: child;
    pair.z := x;
    for (e, a) in { let pair = (x, x); let v = pair.0; [v] } {
        a + e
    }({ let pair = (x, x); let v = pair.1; v })
}
"#;
    compile(source, empty_package("shadow"), ResetBehavior::default())?;
    for field in ["pair.0", "pair.1"] {
        let module = parse_module(&source.replace(field, "pair.out"))?;
        assert!(
            check_module(&module)
                .unwrap_err()
                .contains("invalid tuple field 'out'")
        );
    }
    let module = parse_module(
        r#"
extern fn f(x: u8) -> u8;
block wrapper(clk, x: u8) -> u8 {
    for (_, a) in [x] { a }({ let f = x; let v = f(x); v })
}
"#,
    )?;
    assert!(
        check_module(&module)
            .unwrap_err()
            .contains("'f' names a local value")
    );
    Ok(())
}

#[test]
fn guarded_let_cannot_mark_a_later_runtime_range_bound_as_constexpr() -> Result<(), String> {
    let guard = r#"
    if en {
        let y = for (_, a) in {
            let one = u32:0 + u32:1;
            let two = u32:0 + u32:2;
            [one, two]
        } { a }(u32:0);
    }
"#;
    for prefix in ["", guard] {
        let source = format!(
            r#"
block folded(clk, x: u32, en: bool) -> u32 {{
    {prefix}
    for (i, a) in u32:0..({{ let k = if true {{ u32:2 }} else {{ x }}; k }}) {{
        a + i
    }}(u32:0)
}}
"#
        );
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert!(
            error.contains("range bounds must be compile-time constants"),
            "{error}"
        );
    }
    Ok(())
}
