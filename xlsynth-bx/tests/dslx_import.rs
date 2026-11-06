// SPDX-License-Identifier: Apache-2.0

use std::{error::Error, fs, io, path::Path};

use tempfile::tempdir;
use xlsynth_bx::ast::Type;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module_with_options, compile_with_options, empty_package,
    lower_module, parse_module,
};
use xlsynth_pir::IrValue;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir::Package;
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_pir::ir_parser::Parser;

/// Writes a DSLX module under one of the test's import roots.
fn write_module(root: &Path, relative_path: &str, source: &str) -> io::Result<()> {
    let path = root.join(relative_path);
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, source)
}

/// Evaluates the entire block, including its linked calls, for one u8 input.
fn run_block(package: &Package, x: u8) -> Result<IrValue, Box<dyn Error>> {
    let block = package.get_top_block().ok_or("expected a top block")?;
    let function = combinational_block_to_fn(block)?;
    let input = IrValue::make_ubits(8, u64::from(x))?;
    Ok(eval_fn_in_package_value(package, &function, &[input]))
}

#[test]
fn imported_declarations_bind_to_their_linked_specializations() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    write_module(
        root.path(),
        "ops.x",
        "pub fn id<N: u32>(operand: uN[N]) -> uN[N] { operand }",
    )?;
    let options = ImportOptions {
        additional_search_paths: vec![root.path()],
        ..Default::default()
    };
    let compiled = xlsynth_bx::compile_with_source(
        "use ops; block widths(clk, a: u8, b: u4) -> (u8, u4) { (ops::id(a), ops::id(b)) }",
        empty_package("top"),
        ResetBehavior::default(),
        &options,
    )?;
    let [batch] = &*compiled.imports else {
        panic!("expected one DSLX compilation")
    };
    assert_eq!(batch.symbols.len(), 2);
    let widths: std::collections::BTreeSet<_> = batch
        .symbols
        .values()
        .map(|symbol| {
            let function = compiled.package.get_fn(symbol).unwrap();
            assert_eq!(function.get_param(0).param_name(), "operand");
            let xlsynth_pir::ir::Type::Bits(width) = function.ret_ty else {
                panic!("expected bits")
            };
            width
        })
        .collect();
    assert_eq!(widths, [4, 8].into());
    Ok(())
}

#[test]
fn ordered_roots_select_the_body_and_qualified_and_direct_calls_share_one_specialization()
-> Result<(), Box<dyn Error>> {
    let first = tempdir()?;
    let second = tempdir()?;
    write_module(
        first.path(),
        "device/logic.x",
        "pub fn tag(x: u8) -> u8 { x + u8:1 }",
    )?;
    write_module(
        second.path(),
        "device/logic.x",
        "pub fn tag(x: u8) -> u8 { x + u8:9 }",
    )?;
    let module = parse_module(
        r#"
use device::logic;
use device::logic::tag;
block tags(clk, x: u8) -> (u8, u8) { (logic::tag(x), tag(x)) }
"#,
    )?;
    for (paths, expected) in [
        ([first.path(), second.path()], "(bits[8]:6, bits[8]:6)"),
        ([second.path(), first.path()], "(bits[8]:14, bits[8]:14)"),
    ] {
        let options = ImportOptions {
            dslx_stdlib_path: None,
            additional_search_paths: paths.to_vec(),
        };
        let checked = check_module_with_options(&module, &options)?;
        let [call] = checked.imported_calls() else {
            panic!("expected a single specialization of device.logic::tag");
        };
        assert_eq!(call.module, "device.logic");
        assert_eq!(call.function, "tag");
        let byte = Type::Bits {
            signed: false,
            width: 8,
        };
        assert_eq!(call.args.as_slice(), std::slice::from_ref(&byte));
        assert_eq!(call.result, byte);
        let package = lower_module(&checked, empty_package("design"), ResetBehavior::default())?;
        assert_eq!(run_block(&package, 5)?, IrValue::parse_typed(expected)?);
    }
    Ok(())
}

#[test]
fn dotted_and_underscored_subjects_keep_distinct_callees_alongside_externs_and_stdlib()
-> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    write_module(
        root.path(),
        "a/b.x",
        "pub fn score(x: u8) -> u8 { x + u8:2 }",
    )?;
    write_module(
        root.path(),
        "a_b.x",
        "pub fn score(x: u8) -> u8 { x + u8:7 }",
    )?;
    let supplied = Parser::new(
        r#"package circuit
fn score(x: bits[8] id=1) -> bits[8] { ret n: bits[8] = not(x, id=2) }"#,
    )
    .parse_and_verify_package()?;
    let source = r#"
use a::b;
use a_b;
use acm_random::rng_deterministic_seed;
extern fn score(x: u8) -> u8;
block results(clk, x: u8) -> (u8, u8, u8, u32) {
    (b::score(x), a_b::score(x), score(x), rng_deterministic_seed())
}
"#;
    let options = ImportOptions {
        dslx_stdlib_path: None,
        additional_search_paths: vec![root.path()],
    };
    let package = compile_with_options(source, supplied, ResetBehavior::default(), &options)?;
    assert_eq!(
        run_block(&package, 5)?,
        IrValue::parse_typed("(bits[8]:7, bits[8]:12, bits[8]:250, bits[32]:301)")?
    );
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn checked_module_keeps_its_transitive_dslx_bodies_after_sources_change()
-> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    write_module(
        root.path(),
        "fabric/adjust.x",
        "pub fn twice(x: u8) -> u8 { x + x }",
    )?;
    write_module(
        root.path(),
        "fabric/stage.x",
        r#"
import fabric.adjust;
pub fn score(x: u8) -> u8 { adjust::twice(x) + u8:3 }
pub fn checked_score(x: u8) -> u8 { if x == u8:0 { fail!("zero", x) } else { score(x) } }
"#,
    )?;
    let options = ImportOptions {
        dslx_stdlib_path: None,
        additional_search_paths: vec![root.path()],
    };
    let source = "use fabric::stage; block b(clk, x: u8) -> u8 { stage::score(x) }";
    let checked = check_module_with_options(&parse_module(source)?, &options)?;
    let effectful = "use fabric::stage; block b(clk, x: u8) -> u8 { stage::checked_score(x) }";
    let error = compile_with_options(
        effectful,
        empty_package("effects"),
        ResetBehavior::default(),
        &options,
    )
    .unwrap_err();
    assert!(
        error.contains("checked_score")
            && (error.contains("implicit token") || error.contains("effectful")),
        "{error}"
    );

    write_module(
        root.path(),
        "fabric/stage.x",
        "pub fn score(x: u8) -> u8 { x - u8:1 }",
    )?;
    fs::remove_file(root.path().join("fabric/adjust.x"))?;
    let saved = lower_module(&checked, empty_package("saved"), ResetBehavior::default())?;
    assert_eq!(run_block(&saved, 5)?, IrValue::make_ubits(8, 13)?);
    let fresh = compile_with_options(
        source,
        empty_package("fresh"),
        ResetBehavior::default(),
        &options,
    )?;
    assert_eq!(run_block(&fresh, 5)?, IrValue::make_ubits(8, 4)?);
    Ok(())
}

#[test]
fn an_unused_missing_import_fails_during_checking() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    let options = ImportOptions {
        dslx_stdlib_path: None,
        additional_search_paths: vec![root.path()],
    };
    let module = parse_module("use test::missing_probe; block b(clk, x: u8) -> u8 { x }")?;
    let error = check_module_with_options(&module, &options).unwrap_err();
    assert!(error.contains("missing_probe"), "{error}");
    Ok(())
}

#[test]
fn stdlib_override_supplies_the_body_and_its_portable_source_path() -> Result<(), Box<dyn Error>> {
    let stdlib = tempdir()?;
    write_module(
        stdlib.path(),
        "acm_random.x",
        "pub fn rng_deterministic_seed() -> u32 { u32:42 }",
    )?;
    let options = ImportOptions {
        dslx_stdlib_path: Some(stdlib.path()),
        additional_search_paths: Vec::new(),
    };
    let source = r#"
use acm_random;
block seed(clk, x: u8) -> u32 { acm_random::rng_deterministic_seed() + (x as u32) }
"#;
    let package = compile_with_options(
        source,
        empty_package("design"),
        ResetBehavior::default(),
        &options,
    )?;
    assert_eq!(run_block(&package, 5)?, IrValue::make_ubits(32, 47)?);
    let function = package
        .get_fn("__acm_random__rng_deterministic_seed")
        .unwrap();
    let literal = function.get_node(function.ret_node_ref.unwrap());
    let paths: Vec<_> = literal
        .pos
        .iter()
        .flatten()
        .map(|pos| package.file_table.id_to_path[&pos.fileno].as_str())
        .collect();
    assert_eq!(paths, ["xls/dslx/stdlib/acm_random.x"]);
    Ok(())
}
