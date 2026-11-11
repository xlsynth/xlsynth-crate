// SPDX-License-Identifier: Apache-2.0

use super::support::search_paths;
use std::error::Error;
use std::fs;
use tempfile::tempdir;

use super::directed_expr_eval::{
    assert_expr_matches_dslx, assert_expr_rejected_by_both, compile_dslx_module,
};
use super::expr_eval::{BlockEvaluator, ExpressionEvaluator, bx_source, dslx_source};
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, compile, empty_package, parse_module,
};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::{NodePayload, Type};

#[test]
fn typed_prefix_and_runtime_tail_match_dslx() -> Result<(), Box<dyn Error>> {
    let x = IrValue::make_ubits(8, 137)?;
    assert_expr_matches_dslx(
        "x: u8",
        "(u8[4], u8[2], u8[1])",
        "let filled = u8[4]:[1, x, ...]; (filled, u8[2]:[3, 5], u8[1]:[x, ...])",
        [(
            vec![x],
            IrValue::parse_typed(
                "([bits[8]:1, bits[8]:137, bits[8]:137, bits[8]:137], [bits[8]:3, bits[8]:5], [bits[8]:137])",
            )?,
        )],
    );
    Ok(())
}

#[test]
fn filled_constants_and_parametric_runtime_rows_match_dslx() -> Result<(), Box<dyn Error>> {
    let declarations = r#"
const ROWS = u32:2;
const GRID = u8[3][ROWS]:[[1, 4, ...], ...];
fn row<N: u32>(x: u8) -> u8[N] { u8[N]:[x + u8:1, ...] }
"#;
    let body = "(GRID, row<3>(x))";
    let result = "(u8[3][2], u8[3])";
    let source = format!("{declarations}\n{}", bx_source("x: u8", result, body));
    let reference = format!("{declarations}\n{}", dslx_source("x: u8", result, body));
    let evaluator = ExpressionEvaluator::compile_block(
        &source,
        &ImportOptions::default(),
        compile_dslx_module(&reference, &[]),
    )?;
    let row = IrValue::parse_typed("[bits[8]:1, bits[8]:4, bits[8]:4]")?;
    let grid = IrValue::make_array(&[row.clone(), row])?;
    let runtime_row = IrValue::parse_typed("[bits[8]:7, bits[8]:7, bits[8]:7]")?;
    assert_eq!(
        evaluator.compare(&[IrValue::make_ubits(8, 6)?])?,
        IrValue::make_tuple(&[grid, runtime_row])
    );
    Ok(())
}

#[test]
fn imported_calls_get_array_dimensions_from_typed_arguments_and_siblings()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("rows.x"),
        "pub fn last<N: u32>(a: u8[N]) -> u8 { a[N - u32:1] }",
    )?;
    let options = search_paths(&[directory.path()]);
    for (body, expected) in [
        ("rows::last(u8[4]:[1, x, ...])", [37, 37]),
        ("rows::last<4>(u8[4]:[1, x, ...])", [37, 37]),
        (
            "rows::last(if sel { [1, x, ...] } else { u8[4]:[5, ...] })",
            [5, 37],
        ),
        (
            "rows::last<4>(if sel { u8[4]:[5, ...] } else { [1, x, ...] })",
            [37, 5],
        ),
    ] {
        let params = "x: u8, sel: bool";
        let bx = format!("use rows;\n{}", bx_source(params, "u8", body));
        let native = format!("import rows;\n{}", dslx_source(params, "u8", body));
        let evaluator = ExpressionEvaluator::compile_block(
            &bx,
            &options,
            compile_dslx_module(&native, &[directory.path()]),
        )?;
        let x = IrValue::make_ubits(8, 37)?;
        for (sel, expected) in [false, true].into_iter().zip(expected) {
            assert_eq!(
                evaluator.compare(&[x.clone(), IrValue::bool(sel)])?,
                IrValue::make_ubits(8, expected)?,
                "{body}"
            );
        }
    }
    Ok(())
}

#[test]
fn imported_array_parametrics_infer_unrolled_seeds_from_branch_siblings()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("rows.x"),
        "pub fn head<N: u32>(a: u8[N]) -> u8 { a[0] }",
    )?;
    let options = search_paths(&[directory.path()]);
    let params = "x: u8, sel: bool";
    let fold = "unroll_for! (_, a) in u32:0..u32:1 { u8[3]:[a[0] + u8:1, ...] }([x, ...])";
    for head in ["rows::head", "rows::head<3>"] {
        for sibling in ["u8[3]:[5, ...]", "[5, 4, 3]"] {
            let body = format!("{head}(if sel {{ {fold} }} else {{ {sibling} }})");
            let bx = format!("use rows;\n{}", bx_source(params, "u8", &body));
            let native = format!("import rows;\n{}", dslx_source(params, "u8", &body));
            let evaluator = ExpressionEvaluator::compile_block(
                &bx,
                &options,
                compile_dslx_module(&native, &[directory.path()]),
            )?;
            for (sel, expected) in [(false, 5), (true, 38)] {
                assert_eq!(
                    evaluator.compare(&[IrValue::make_ubits(8, 37)?, IrValue::bool(sel)])?,
                    IrValue::make_ubits(8, expected)?,
                    "{body}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn empty_typed_arrays_retain_their_element_type_in_constants_and_ir() -> Result<(), Box<dyn Error>>
{
    // DSLX typechecks the empty literal, but the pinned converter fails to
    // lower it. Check its element type and the BX result with PIR.
    let source = r#"
const EMPTY = u8[0]:[];
#[allow(combinational_block)]
block sample(clk) -> (u8[0], u8[0][2]) { (EMPTY, u8[0][2]:[u8[0]:[], ...]) }
"#;
    let checked = check_module(&parse_module(source)?)?;
    let empty = IrValue::make_array_typed(Type::Bits(8), &[])?;
    assert_eq!(checked.constants()[0].value, empty);
    let rows = IrValue::make_array(&[empty.clone(), empty.clone()])?;
    let evaluator = BlockEvaluator::lower(&checked, source)?;
    assert_eq!(
        evaluator.evaluate(&[])?,
        IrValue::make_tuple(&[empty, rows])
    );
    Ok(())
}

#[test]
fn generic_blocks_infer_types_from_a_fill_prefix() -> Result<(), Box<dyn Error>> {
    let source = r#"
fn last_value<T>(a: T[3]) -> T { a[2] }
#[allow(combinational_block)]
block last<T>(clk, a: T[3]) -> T { a[2] }
#[allow(combinational_block)]
block first<T>(clk, a: T, b: T) -> T { a }
#[allow(combinational_block)]
pub block sample(clk, x: u8) -> (u8, u8, u8) {
    let a = last([1, x, ...]);
    let b = first([x, ...], [1, 2, 3]);
    (a, b[2], last_value(if x == u8:0 { [1, 2, 3] } else { [x, ...] }))
}
"#;
    let checked = check_module(&parse_module(source)?)?;
    let evaluator = BlockEvaluator::lower(&checked, source)?;
    for (x, expected_last) in [(0, 3), (37, 37), (255, 255)] {
        let x = IrValue::make_ubits(8, x)?;
        assert_eq!(
            evaluator.evaluate(std::slice::from_ref(&x))?,
            IrValue::make_tuple(&[x.clone(), x, IrValue::make_ubits(8, expected_last)?])
        );
    }
    let overfull = source.replace("[1, x, ...]", "[1, x, x, x, ...]");
    assert!(check_module(&parse_module(&overfull)?).is_err());
    Ok(())
}

#[test]
fn inferred_generic_branches_evaluate_each_written_fill_member_once() -> Result<(), Box<dyn Error>>
{
    for row in [
        "[bump(x), ...]",
        "for (_, a) in u32:0..u32:1 { a }([{ let b = bump(x); b }, ...])",
    ] {
        for select in [
            format!("id(if c {{ {row} }} else {{ y }})[0]"),
            format!("first(if c {{ {row} }} else {{ y }})"),
        ] {
            let source = format!(
                r#"
fn bump(x: bool) -> bool {{ !x }}
fn id<T>(a: T) -> T {{ a }}
fn first<T>(a: T[3]) -> T {{ a[0] }}
#[allow(combinational_block)]
pub block sample(clk, x: bool, c: bool, y: bool[3]) -> bool {{ {select} }}
"#
            );
            let package = compile(&source, empty_package("design"), ResetBehavior::default())?;
            let sample = package.get_top_block().unwrap();
            let bump_calls = sample.nodes.iter().filter(|node| {
                matches!(&node.payload, NodePayload::Invoke { to_apply, .. } if to_apply == "bump")
            }).count();
            assert_eq!(bump_calls, 1, "{select}");
        }
    }
    Ok(())
}

#[test]
fn trailing_fill_invokes_a_helper_once_on_a_clocked_child_output() -> Result<(), Box<dyn Error>> {
    let source = r#"
fn bump(x: u8) -> u8 { x + u8:1 }
block delay(clk, rst, x: u8) -> u8 {
    reg q: u8 = 0;
    q := x;
    q
}
pub block sample(clk, rst, x: u8) -> u8[4] {
    let q = delay(x);
    u8[4]:[1, bump(q), ...]
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    let sample = package.get_top_block().unwrap();
    assert_eq!(sample.instantiations.len(), 1);
    assert_eq!(sample.instantiations[0].block, "delay");
    assert_eq!(
        sample
            .nodes
            .iter()
            .filter(|node| matches!(node.payload, NodePayload::Invoke { .. }))
            .count(),
        1
    );
    let arrays = sample
        .nodes
        .iter()
        .filter_map(|node| match &node.payload {
            NodePayload::Array(inputs) => Some(inputs.as_slice()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [[head, x, y, z]] = arrays.as_slice() else {
        panic!("expected one four-element array, got {arrays:?}");
    };
    assert_ne!(head, x);
    assert_eq!(x, y);
    assert_eq!(x, z);
    Ok(())
}

#[test]
fn annotated_lengths_and_element_types_are_checked() {
    for body in [
        "u8[2]:[x]",
        "u8[2]:[x, x, x, ...]",
        "u8[2]:[...]",
        "u8[2]:[u7:1, ...]",
        "u8[2]:[256, ...]",
        "u8[3]:[x, ...]",
    ] {
        assert_expr_rejected_by_both("x: u8", "u8[2]", body);
    }
}

#[test]
fn a_concrete_function_result_supplies_a_sibling_fill_length() -> Result<(), Box<dyn Error>> {
    // Pinned DSLX requires more context here when make is nonparametric.
    // BX can use its declared u8[3] result just like a named u8[3] value.
    let source = r#"
fn make(x: u8) -> u8[3] { u8[3]:[0, !x, ...] }
fn last(row: u8[3]) -> u8 { row[2] }
fn from_let(x: u8, c: bool) -> u8 {
    let row = if c { [x, ...] } else { make(x) };
    row[2]
}
#[allow(combinational_block)]
pub block sample(clk, x: u8, c: bool) -> (u8, u8) {
    (from_let(x, c), last(if c { [x, ...] } else { make(x) }))
}
"#;
    let evaluator = BlockEvaluator::lower(&check_module(&parse_module(source)?)?, source)?;
    for (c, expected) in [(true, 37), (false, 218)] {
        let expected = IrValue::make_ubits(8, expected)?;
        assert_eq!(
            evaluator.evaluate(&[IrValue::make_ubits(8, 37)?, IrValue::bool(c)])?,
            IrValue::make_tuple(&[expected.clone(), expected])
        );
    }
    Ok(())
}
