// SPDX-License-Identifier: Apache-2.0

use super::support::search_paths;
use std::error::Error;
use std::fs;

use tempfile::tempdir;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_source_with_options, compile_with_options, empty_package,
};
use xlsynth_pir::IrValue;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir_eval::eval_fn_in_package_value;

use super::directed_expr_eval::{compile_dslx, compile_dslx_module};
use super::expr_eval::{ExpressionEvaluator, bx_source};

#[test]
fn imported_bx_builtin_calls_keep_their_definition_scope() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    fs::write(
        root.path().join("lanes.bx"),
        r#"
pub const FIRST = rev(u8:1);
pub fn reversed(x: u8) -> u8 { rev(x) }
"#,
    )?;
    let source = r#"
use lanes;
fn rev(x: u8) -> u8 { x + u8:1 }

#[allow(combinational_block)]
block sample(clk, x: u8) -> (u8, u8, u8) {
    (lanes::FIRST, lanes::reversed(x), rev(x))
}
"#;
    let options = search_paths(&[root.path()]);
    let package = compile_with_options(
        source,
        empty_package("test"),
        ResetBehavior::default(),
        &options,
    )?;
    let sample = combinational_block_to_fn(package.get_block("sample").ok_or("missing sample")?)?;
    assert_eq!(
        eval_fn_in_package_value(&package, &sample, &[IrValue::make_ubits(8, 3)?]),
        IrValue::parse_typed("(bits[8]:128, bits[8]:192, bits[8]:4)")?
    );
    Ok(())
}

#[test]
fn update_builtin_yields_to_a_struct_only_after_it_is_visible() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    let declaration = "struct update { x: u8 }";
    fs::write(
        root.path().join("bx_types.bx"),
        format!("pub {declaration}"),
    )?;
    fs::write(
        root.path().join("dslx_types.x"),
        format!("pub {declaration}"),
    )?;
    let options = search_paths(&[root.path()]);
    let block = r#"
#[allow(combinational_block)]
pub block sample(clk, xs: u8[2]) -> u8[2] { update(xs, u1:1, u8:3) }
"#;
    let function = "fn sample(xs: u8[2]) -> u8[2] { update(xs, u1:1, u8:3) }";
    let args = [IrValue::parse_typed("[bits[8]:1, bits[8]:2]")?];
    let expected = IrValue::parse_typed("[bits[8]:1, bits[8]:3]")?;
    for (bx_type, dslx_type) in [
        (declaration, declaration),
        ("use bx_types::update;", "use dslx_types::{update};"),
        ("use dslx_types::update;", "use dslx_types::{update};"),
    ] {
        let source = format!("{block}\n{bx_type}");
        let native = format!("#![feature(use_syntax)]\n{function}\n{dslx_type}");
        let evaluator = ExpressionEvaluator::compile_block(
            &source,
            &options,
            compile_dslx_module(&native, &[root.path()]),
        )?;
        assert_eq!(evaluator.compare(&args)?, expected);

        fs::write(root.path().join("library.bx"), source)?;
        let caller = r#"
use library;
block sample(clk, xs: u8[2]) -> u8[2] {
    let result = library::sample(xs);
    result
}
"#;
        let evaluator = ExpressionEvaluator::compile_block(
            caller,
            &options,
            compile_dslx_module(&native, &[root.path()]),
        )?;
        assert_eq!(evaluator.compare(&args)?, expected);

        let source = format!("{bx_type}\n{block}");
        assert!(check_source_with_options(&source, &options).is_err());
        let native = format!("#![feature(use_syntax)]\n{dslx_type}\n{function}");
        assert!(compile_dslx_module(&native, &[root.path()]).is_err());
    }
    Ok(())
}

#[test]
fn update_accepts_a_tuple_valued_index() -> Result<(), Box<dyn Error>> {
    let params = "xs: u8[2][2], index: (u1, u1), value: u8";
    let result = "u8[2][2]";
    let evaluator = ExpressionEvaluator::compile_block(
        &bx_source(params, result, "update(xs, index, value)"),
        &ImportOptions::default(),
        // The pinned converter needs inline tuple syntax for this operation.
        compile_dslx(params, result, "update(xs, (index.0, index.1), value)"),
    )?;
    assert_eq!(
        evaluator.compare(&[
            IrValue::parse_typed("[[bits[8]:1, bits[8]:2], [bits[8]:3, bits[8]:4]]")?,
            IrValue::parse_typed("(bits[1]:1, bits[1]:0)")?,
            IrValue::parse_typed("bits[8]:9")?,
        ])?,
        IrValue::parse_typed("[[bits[8]:1, bits[8]:2], [bits[8]:9, bits[8]:4]]")?
    );
    Ok(())
}
