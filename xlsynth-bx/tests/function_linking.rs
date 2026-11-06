// SPDX-License-Identifier: Apache-2.0

use std::{error::Error, fs};

use tempfile::tempdir;
use xlsynth_bx::{ImportOptions, ResetBehavior, compile, compile_with_options, empty_package};
use xlsynth_pir::IrValue;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_pir::ir_parser::Parser;

#[test]
fn local_functions_resolve_qualified_types_in_source_order() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(dir.path().join("wire.x"), "pub struct T { x: u8 }")?;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..ImportOptions::default()
    };
    for function in [
        "fn probe(x: (wire::T,)) -> u8 { x.0.x }",
        "fn probe(x: u8) -> u8 { let t = wire::T { x }; t.x }",
    ] {
        let early = format!("use wire;\n{function}");
        compile_with_options(
            &early,
            xlsynth_bx::empty_package("probe"),
            ResetBehavior::default(),
            &options,
        )?;
        let late = format!("{function}\nuse wire;");
        let error = compile_with_options(
            &late,
            xlsynth_bx::empty_package("probe"),
            ResetBehavior::default(),
            &options,
        )
        .unwrap_err();
        assert!(
            error.contains("type 'wire::T' is declared later"),
            "{error}"
        );
    }
    Ok(())
}

#[test]
fn local_function_calls_remain_distinct_from_imports_and_externs() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("math.x"),
        r#"
fn offset(x: u8) -> u8 { x + u8:2 }
pub fn transform(x: u8) -> u8 { offset(x) * u8:3 }
"#,
    )?;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..ImportOptions::default()
    };
    let linked = Parser::new(
        r#"package design
fn invert(x: bits[8] id=1) -> bits[8] { ret not.2: bits[8] = not(x, id=2) }
"#,
    )
    .parse_and_verify_package()?;
    let source = r#"
use math;
extern fn flip(x: u8) -> u8 = "invert";

fn __math__offset(x: u8) -> u8 { x + u8:9 }
fn collect(x: u8) -> (u8, u8, u8) { (__math__offset(x), math::transform(x), flip(x)) }
block results(clk, x: u8) -> (u8, u8, u8) { collect(x) }
"#;
    let package = compile_with_options(source, linked, ResetBehavior::default(), &options)?;
    let top = package.get_top_block().ok_or("expected a top block")?;
    let function = combinational_block_to_fn(top)?;
    assert_eq!(
        eval_fn_in_package_value(&package, &function, &[IrValue::make_ubits(8, 5)?]),
        IrValue::parse_typed("(bits[8]:14, bits[8]:21, bits[8]:250)")?
    );
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn local_function_rejects_an_occupied_destination_symbol() -> Result<(), Box<dyn Error>> {
    for member in [
        "fn occupied() -> bits[8] { ret literal.1: bits[8] = literal(value=1, id=1) }",
        "block occupied() {}",
    ] {
        let linked =
            Parser::new(&format!("package design\n{member}")).parse_and_verify_package()?;
        let error = compile(
            "fn occupied() -> u8 { u8:2 }",
            linked,
            ResetBehavior::default(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            r#"fn 'occupied': duplicate IR identifier: "occupied""#
        );
    }
    Ok(())
}

#[test]
fn local_function_and_parameter_names_can_be_ir_keywords() -> Result<(), Box<dyn Error>> {
    let linked = Parser::new(
        r#"package design
fn stage__bx_2() -> bits[8] { ret literal.1: bits[8] = literal(value=42, id=1) }
"#,
    )
    .parse_and_verify_package()?;
    let source = r#"
fn stage(next: u8, next__bx_1: u8) -> u8 { next - next__bx_1 }
fn stage__bx_1(x: u8) -> u8 { x + u8:7 }
block results(clk, x: u8) -> u8 { stage(stage__bx_1(x), x) }
"#;
    let package = compile(source, linked, ResetBehavior::default())?;
    let top = package.get_top_block().ok_or("expected a top block")?;
    let function = combinational_block_to_fn(top)?;
    assert_eq!(
        eval_fn_in_package_value(&package, &function, &[IrValue::make_ubits(8, 12)?]),
        IrValue::make_ubits(8, 7)?
    );
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn block_typed_zero_preserves_the_intrinsic_with_a_same_named_import() -> Result<(), Box<dyn Error>>
{
    let dir = tempdir()?;
    fs::write(dir.path().join("wire.x"), "pub fn zero!() -> u8 { u8:7 }")?;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..ImportOptions::default()
    };
    for (source, expected) in [
        ("use wire::zero!; block probe(clk) -> u8 { zero!<u8>() }", 0),
        (
            "block probe(clk) -> u8 { zero!<u8>() } fn zero!() -> u8 { u8:7 }",
            7,
        ),
        (
            "block probe(clk) -> u8 { zero!() } fn zero!() -> u8 { u8:7 }",
            7,
        ),
    ] {
        let package = compile_with_options(
            source,
            empty_package("design"),
            ResetBehavior::default(),
            &options,
        )?;
        let top = package.get_top_block().ok_or("expected a top block")?;
        let function = combinational_block_to_fn(top)?;
        assert_eq!(
            eval_fn_in_package_value(&package, &function, &[]),
            IrValue::make_ubits(8, expected)?,
            "{source}"
        );
    }
    Ok(())
}
