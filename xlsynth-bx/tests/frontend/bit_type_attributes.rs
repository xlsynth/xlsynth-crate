// SPDX-License-Identifier: Apache-2.0

use std::{error::Error, fs};

use tempfile::tempdir;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, check_source_with_options, compile, empty_package,
    parse_module,
};
use xlsynth_pir::IrValue;

use super::directed_expr_eval::compile_dslx_module;
use super::expr_eval::{ExpressionEvaluator, bx_source, dslx_source};
use super::support::search_paths;

#[test]
fn bounds_and_zero_keep_the_bit_type_and_its_full_width() -> Result<(), Box<dyn Error>> {
    for (declaration, subject, width, minimum, maximum) in [
        ("", "u1", 1, "0", "1"),
        ("", "s1", 1, "1", "0"),
        ("", "s8", 8, "128", "127"),
        ("type Flag = bool;", "Flag", 1, "0", "1"),
        (
            "type Size = u32; type U = uN[Size::ZERO];",
            "U",
            0,
            "0",
            "0",
        ),
        ("type S = sN[0];", "S", 0, "0", "0"),
        (
            "type B = bits[81];",
            "B",
            81,
            "0",
            "0x1_ffff_ffff_ffff_ffff_ffff",
        ),
        (
            "type U = uN[129];",
            "U",
            129,
            "0",
            "0x1_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff",
        ),
        (
            "type Signed = sN[129]; type W = Signed;",
            "W",
            129,
            "0x1_0000_0000_0000_0000_0000_0000_0000_0000",
            "0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff",
        ),
        (
            "type Flag = bool; type S = xN[Flag::MAX][13];",
            "S",
            13,
            "4096",
            "4095",
        ),
    ] {
        let result = format!("({subject}, {subject}, {subject})");
        let body = format!("({subject}::MIN, {subject}::MAX, {subject}::ZERO)");
        let bx = format!("{declaration}\n{}", bx_source("", &result, &body));
        let native = format!("{declaration}\n{}", dslx_source("", &result, &body));
        let evaluator = ExpressionEvaluator::compile_block(
            &bx,
            &ImportOptions::default(),
            compile_dslx_module(&native, &[]),
        )?;
        assert_eq!(
            evaluator.compare(&[])?,
            IrValue::parse_typed(&format!(
                "(bits[{width}]:{minimum}, bits[{width}]:{maximum}, bits[{width}]:0)"
            ))?,
            "{declaration} {body}"
        );
    }
    Ok(())
}

#[test]
fn local_alias_bounds_specialize_and_work_in_constexprs_and_patterns() -> Result<(), Box<dyn Error>>
{
    let declarations = r#"
type W = u8;
fn clamp<N: u32>(x: sN[N]) -> sN[N] {
    type W = sN[N];
    const BOTTOM = W::MIN;
    match x { W::MAX => BOTTOM, _ => W::ZERO }
}
const N = u3::MAX as u32;
type Signed = sN[N];
const FLOOR = clamp(Signed::MAX);
const_assert!(FLOOR == Signed::MIN);
"#;
    let result = "(s7, s4, s8, u8)";
    let body = "(FLOOR, clamp(x as s4), clamp(x), W::MAX)";
    let bx = format!("{declarations}\n{}", bx_source("x:s8", result, body));
    let native = format!("{declarations}\n{}", dslx_source("x:s8", result, body));
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&native, &[]),
    )?;
    for (x, narrow, full) in [(7, 8, 0), (127, 0, 128), (128, 0, 0)] {
        assert_eq!(
            evaluator.compare(&[IrValue::make_ubits(8, x)?])?,
            IrValue::parse_typed(&format!(
                "(bits[7]:64, bits[4]:{narrow}, bits[8]:{full}, bits[8]:255)"
            ))?
        );
    }
    Ok(())
}

#[test]
fn attribute_names_keep_their_module_and_enum_meanings_across_imports() -> Result<(), Box<dyn Error>>
{
    let dir = tempdir()?;
    let reference_dir = tempdir()?;
    let native_types = r#"
pub type Remote = s6;
pub const MAX = u8:43;
pub enum Marker: s4 { MAX = -3 }
pub type Tag = Marker;
"#;
    let shared = r#"
pub type Byte = u8;
pub enum Mode: u4 { MIN = 9, MAX = 2, ZERO = 5 }
type State = Mode;
pub const MAX = Byte:27;
pub fn values() -> (Byte, Mode, Mode, Mode) {
    (u8::MAX, State::MIN, Mode::MAX, State::ZERO)
}
"#;
    fs::write(dir.path().join("wire.x"), native_types)?;
    fs::write(dir.path().join("library.bx"), shared)?;
    fs::write(reference_dir.path().join("library.x"), shared)?;
    let result = "((u8, library::Mode, library::Mode, library::Mode), u8, u8, s6, s6, u8, s8)";
    let body = r#"(
        library::values(), library::MAX, library::Byte::MAX,
        Remote::MIN, wire::Remote::MAX, wire::MAX, wire::Tag::MAX as s8
    )"#;
    let bx = format!(
        "use library; use wire; use wire::Remote;\n{}",
        bx_source("", result, body)
    );
    let native = format!(
        "import library; import wire; type Remote = wire::Remote;\n{}",
        dslx_source("", result, body)
    );
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &search_paths(&[dir.path()]),
        compile_dslx_module(&native, &[reference_dir.path(), dir.path()]),
    )?;
    assert_eq!(
        evaluator.compare(&[])?,
        IrValue::parse_typed(
            "((bits[8]:255,bits[4]:9,bits[4]:2,bits[4]:5),bits[8]:27,bits[8]:255,bits[6]:32,bits[6]:31,bits[8]:43,bits[8]:253)"
        )?
    );
    Ok(())
}

#[test]
fn register_reset_uses_an_alias_bound_in_block_declaration_order() -> Result<(), Box<dyn Error>> {
    let capture = r#"
block capture(clk, input: s7) -> s7 {
    reg previous: s7 = Word::MIN;
    previous := input;
    previous
}
"#;
    let alias = "type Word = s7;";
    let source = format!("{alias}\n{capture}");
    let package = compile(&source, empty_package("capture"), ResetBehavior::default())?;
    let block = package
        .get_block("capture")
        .ok_or("missing capture block")?;
    assert_eq!(
        block.registers[0].reset_value,
        Some(IrValue::make_ubits(7, 64)?)
    );
    let forward = format!("{capture}\n{alias}");
    let expected = "block 'capture': reset of register 'previous': type 'Word' is declared later in the module";
    assert_eq!(
        compile(&forward, empty_package("capture"), ResetBehavior::default()).unwrap_err(),
        expected
    );
    assert_eq!(
        check_module(&parse_module(&forward)?).unwrap_err(),
        expected
    );
    Ok(())
}

#[test]
fn non_bits_subjects_unknown_attributes_and_shadowed_names_are_rejected() {
    for source in [
        "fn sample()->u8 { u8::ONES }",
        "type Word=u5; fn sample()->Word { Word::max }",
        "type Pair=(u8,u8); fn sample()->Pair { Pair::ZERO }",
        "type Row=u8[2]; fn sample()->Row { Row::MAX }",
        "struct Packet { x:u8 } fn sample()->Packet { Packet::MIN }",
        "enum Mode:u4 { Idle=0 } type Alias=Mode; fn sample()->Alias { Alias::ZERO }",
        "type Word=u8; fn sample(Word:u8)->u8 { Word::MAX }",
        "type Word=u8; fn sample()->u8 { let Word=u8:1; Word::MIN }",
        "const VALUE=u8:2; fn sample()->u8 { VALUE::ZERO }",
        "fn helper()->u8 { u8:1 } fn sample()->u8 { helper::MAX }",
        "fn sample()->u8 { Missing::MAX }",
        "fn sample()->u8 { Later::MAX } type Later=u8;",
        "fn sample()->s8 { u8::MAX }",
    ] {
        assert!(compile_dslx_module(source, &[]).is_err(), "{source}");
        assert!(
            parse_module(source).and_then(|m| check_module(&m)).is_err(),
            "{source}"
        );
    }
}

#[test]
fn qualified_type_attributes_require_a_visible_type() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let library = "type Hidden=u8; pub type Word=u8; pub fn id(x:u8)->u8 { x }";
    fs::write(dir.path().join("wire.x"), library)?;
    fs::write(dir.path().join("bus.bx"), library)?;
    for suffix in ["Hidden::MAX", "Missing::MAX", "id::ZERO", "Word::UNKNOWN"] {
        let native = format!("import wire; fn sample()->u8 {{ wire::{suffix} }}");
        assert!(
            compile_dslx_module(&native, &[dir.path()]).is_err(),
            "{native}"
        );
        for module in ["wire", "bus"] {
            let bx = format!("use {module}; fn sample()->u8 {{ {module}::{suffix} }}");
            assert!(
                check_source_with_options(&bx, &search_paths(&[dir.path()])).is_err(),
                "{bx}"
            );
        }
    }
    for module in ["wire", "bus"] {
        let bx = format!("use {module}; fn sample({module}:u8)->u8 {{ {module}::Word::MAX }}");
        assert!(
            check_source_with_options(&bx, &search_paths(&[dir.path()])).is_err(),
            "{bx}"
        );
    }
    Ok(())
}
