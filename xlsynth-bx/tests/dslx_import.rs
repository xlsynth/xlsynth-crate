// SPDX-License-Identifier: Apache-2.0

mod support;

use std::{collections::BTreeSet, error::Error, fs, io, path::Path};
use support::search_paths;

use tempfile::tempdir;
use xlsynth::dslx::{ImportData, parse_and_typecheck};
use xlsynth_bx::ast::Type;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module_with_options, check_source_with_options,
    compile_with_options, empty_package, lower_module, parse_module,
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

/// Compare rejection at the public BX compiler and native DSLX typechecker.
fn assert_import_rejected(module: &str, body: &str, result: &str, options: &ImportOptions) {
    let bx = format!(
        "use {module}; #[allow(combinational_block)] block b(clk,x:u8)->{result} {{ {body} }}"
    );
    let native = format!("import {module}; fn sample(x:u8)->{result} {{ {body} }}");
    assert!(
        compile_with_options(
            &bx,
            empty_package("test"),
            ResetBehavior::default(),
            options
        )
        .is_err(),
        "{bx}"
    );
    assert!(
        parse_and_typecheck(
            &native,
            "sample.x",
            "sample",
            &mut ImportData::new(None, &options.additional_search_paths)
        )
        .is_err(),
        "{native}"
    );
}

#[test]
fn native_constants_supply_dimensions_initializers_and_match_patterns() -> Result<(), Box<dyn Error>>
{
    let root = tempdir()?;
    write_module(
        root.path(),
        "device/config.x",
        r#"
const BASE = u32:4;
fn increment(x: u32) -> u32 { x + u32:1 }
pub const WIDTH = increment(BASE);
pub const SIGNED = true;
pub const LANES = u32:2;
pub const NEG = s5:-3;
pub const HIGH = uN[81]:0x1_0000_0000_0000_0000;
"#,
    )?;
    let source = r#"
use device::config;
use device::config::{LANES, NEG};
const TAG = config::HIGH + uN[81]:7;
fn classify(x: xN[config::SIGNED][config::WIDTH]) -> bool {
    match x { NEG => true, _ => false }
}
#[allow(combinational_block)]
block sample(clk, x: u8) -> (bool, s8, uN[81]) {
    let words: sN[config::WIDTH][LANES] = [NEG, x as s5];
    (classify(words[u32:1]), words[u32:0] as s8, TAG)
}
"#;
    let options = search_paths(&[root.path()]);
    let package = compile_with_options(
        source,
        empty_package("test"),
        ResetBehavior::default(),
        &options,
    )?;
    for (x, matched) in [(29, true), (30, false)] {
        assert_eq!(
            run_block(&package, x)?,
            IrValue::make_tuple(&[
                IrValue::bool(matched),
                IrValue::make_sbits(8, -3)?,
                IrValue::parse_typed("bits[81]:0x1_0000_0000_0000_0007")?,
            ])
        );
    }
    Ok(())
}

#[test]
fn native_constant_types_and_values_keep_their_defining_module() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    for (subject, offset) in [("device", 13), ("other", 41)] {
        write_module(
            root.path(),
            &format!("{subject}.x"),
            &format!(
                r#"
pub enum Mode : u2 {{ Ready = 1, Done = 2 }}
pub struct Record {{ offset: u8, mode: Mode }}
pub const BOOT = Record {{ offset: u8:{offset}, mode: Mode::Ready }};
pub const MODES = [Mode::Done, Mode::Ready];
"#
            ),
        )?;
    }
    write_module(
        root.path(),
        "stage.bx",
        r#"
use device;
use device::BOOT;
pub fn score(x: u8) -> u8 {
    let boot: device::Record = BOOT;
    if boot.mode == device::MODES[u32:1] { x + boot.offset } else { x }
}
"#,
    )?;
    let source = r#"
use stage::score;
use other;
use other::BOOT;
#[allow(combinational_block)]
block sample(clk, x: u8) -> (u8, other::Mode, u8) {
    let first = score(x);
    let BOOT = x;
    (first, other::MODES[u32:0], BOOT + other::BOOT.offset)
}
"#;
    let options = search_paths(&[root.path()]);
    let package = compile_with_options(
        source,
        empty_package("test"),
        ResetBehavior::default(),
        &options,
    )?;
    assert_eq!(
        run_block(&package, 4)?,
        IrValue::parse_typed("(bits[8]:17,bits[2]:2,bits[8]:45)")?,
    );
    for source in [
        "use device; use other; fn bad() -> device::Record { other::BOOT }",
        "use device; use other; fn bad() -> device::Mode { other::MODES[u32:0] }",
        "use stage; fn bad() -> u8 { device::BOOT.offset }",
    ] {
        assert!(
            check_source_with_options(source, &options).is_err(),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn native_constant_parametric_structs_keep_bindings_absent_from_their_fields()
-> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    let declarations = r#"
pub struct Tag<K:u32> { data:u8 }
pub fn label<K:u32>(_tag:Tag<K>) -> u32 { K }
"#;
    for subject in ["first", "second"] {
        write_module(root.path(), &format!("{subject}.x"), declarations)?;
    }
    write_module(
        root.path(),
        "presets.x",
        r#"
import first;
import second;
pub const TAGS = (second::Tag<u32:7> { data:u8:3 }, [first::Tag<u32:2> { data:u8:3 }]);
"#,
    )?;
    let options = search_paths(&[root.path()]);
    let source = r#"
use first;
use second;
use presets::TAGS;
#[allow(combinational_block)]
block sample(clk, x:u8) -> (u32, u32, u8) {
    (second::label(TAGS.0), first::label(TAGS.1[u32:0]), x + TAGS.0.data)
}
"#;
    let package = compile_with_options(
        source,
        empty_package("test"),
        ResetBehavior::default(),
        &options,
    )?;
    assert_eq!(
        run_block(&package, 1)?,
        IrValue::parse_typed("(bits[32]:7,bits[32]:2,bits[8]:4)")?
    );
    for wrong in ["first::Tag<7>", "second::Tag<2>"] {
        let source =
            format!("use first; use second; use presets::TAGS; fn bad()->{wrong} {{ TAGS.0 }}");
        assert!(
            check_source_with_options(&source, &options).is_err(),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn native_constant_imports_obey_visibility_and_source_order() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    write_module(
        root.path(),
        "config.x",
        "const PRIVATE = u32:5; pub const WIDTH = PRIVATE;",
    )?;
    write_module(
        root.path(),
        "middle.x",
        "#![feature(use_syntax)]\nuse config::WIDTH;",
    )?;
    let options = search_paths(&[root.path()]);
    let empty_loop = r#"block fold(clk)->u32 {
    unroll_for! (_, acc) in u32:0..u32:0 {
        match acc { config::WIDTH => config::WIDTH, _ => acc }
    }(u32:0)
}"#;
    check_source_with_options(&format!("use config; {empty_loop}"), &options)?;
    assert!(check_source_with_options(&format!("{empty_loop} use config;"), &options).is_err());
    // As for imported types and functions, DSLX resolves an unused member
    // binding. BX checks that the declaration is public at the read site.
    check_source_with_options("use config::PRIVATE; use middle::WIDTH;", &options)?;
    for source in [
        "use config::PRIVATE; const BAD = PRIVATE;",
        "use middle::WIDTH; const BAD = WIDTH;",
        "use middle; const BAD = middle::WIDTH;",
        "use config::MISSING; fn id(x:u32)->u32 { x }",
        "use config; const BAD = config::PRIVATE;",
        "use config; const BAD = config::MISSING;",
        "use config::WIDTH; fn bad()->u32 { config::WIDTH }",
        "use config; fn bad(config:u32)->u32 { config::WIDTH }",
        "const BAD = config::WIDTH; use config;",
        "fn bad(x:uN[WIDTH])->u5 { x } use config::WIDTH;",
    ] {
        assert!(
            check_source_with_options(source, &options).is_err(),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn native_constant_register_reset_keeps_the_checked_source_value() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    write_module(
        root.path(),
        "config.x",
        "pub const INITIAL = [s8:-4, s8:7];",
    )?;
    let source = r#"
use config::INITIAL;
block delay(clk, x:s8[2])->s8[2] {
    reg q:s8[2] = INITIAL;
    q := x;
    q
}
"#;
    let options = search_paths(&[root.path()]);
    let checked = check_module_with_options(&parse_module(source)?, &options)?;
    write_module(
        root.path(),
        "config.x",
        "pub const INITIAL = [s8:5, s8:-1];",
    )?;
    let saved = lower_module(&checked, empty_package("saved"), ResetBehavior::default())?;
    let fresh = compile_with_options(
        source,
        empty_package("fresh"),
        ResetBehavior::default(),
        &options,
    )?;
    for (package, expected) in [
        (saved, "[bits[8]:252,bits[8]:7]"),
        (fresh, "[bits[8]:5,bits[8]:255]"),
    ] {
        assert_eq!(
            package
                .get_top_block()
                .ok_or("expected delay block")?
                .registers[0]
                .reset_value,
            Some(IrValue::parse_typed(expected)?)
        );
    }
    Ok(())
}

#[test]
fn direct_and_mapped_imports_bind_values_in_the_callers_scope() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    write_module(
        root.path(),
        "math.x",
        "pub fn mix<K:uN[81], N:u32>(x:uN[N])->uN[81] { (x as uN[81]) ^ K }",
    )?;
    write_module(
        root.path(),
        "lanes.bx",
        r#"
use math::mix;
const HIGH = uN[81]:0x1_0000_0000_0000_0000;
pub fn mapped<K:uN[81]>(x:u8)->uN[81][2] {
    map([x, x], mix<{HIGH + K}>)
}"#,
    )?;
    let source = r#"
use lanes;
use math;
use std::{lsbs, msbs};
const HIGH = uN[81]:1;
#[allow(combinational_block)]
block sample(clk, x:u8)->(u3, u4[2], uN[81], uN[81][2], uN[81]) {
    let n = u32:2 + u32:1;
    (lsbs<n>(x), map([x, x], msbs<{n+u32:1}>), math::mix<HIGH>(x), lanes::mapped<HIGH>(x), math::mix<1>(x))
}
"#;
    let options = search_paths(&[root.path()]);
    let package = compile_with_options(
        source,
        empty_package("test"),
        ResetBehavior::default(),
        &options,
    )?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    assert_eq!(
        run_block(&package, 0xc6)?,
        IrValue::parse_typed(
            "(bits[3]:6,[bits[4]:12,bits[4]:12],bits[81]:199,[bits[81]:0x100000000000000c7,bits[81]:0x100000000000000c7],bits[81]:199)"
        )?
    );
    Ok(())
}

#[test]
fn invalid_explicit_import_arguments_are_rejected_for_direct_and_mapped_calls()
-> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    write_module(
        root.path(),
        "logic.x",
        "pub fn flip<K:u8, N:u32>(x:uN[N])->uN[N] { x ^ (K as uN[N]) }",
    )?;
    let options = search_paths(&[root.path()]);
    for actuals in ["<u16:3>", "<x>", "<3,7>", "<3,8,true>", "<256>"] {
        for (call, result) in [
            (format!("logic::flip{actuals}(x)"), "u8"),
            (format!("map([x],logic::flip{actuals})"), "u8[1]"),
        ] {
            assert_import_rejected("logic", &call, result, &options);
        }
    }
    Ok(())
}

#[test]
fn native_value_arguments_supply_context_to_scalar_and_aggregate_data_literals()
-> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    write_module(
        root.path(),
        "data.x",
        r#"
pub fn scalar<N:u32>(x:uN[N])->uN[N] { x + uN[N]:1 }
pub fn pair<N:u32>(p:(uN[N],uN[N]))->uN[N] { p.0 - p.1 }
pub fn lanes<N:u32>(xs:uN[N][2])->uN[N] { xs[u32:0] ^ xs[u32:1] }
pub fn width<K:u32,N:u32>(x:uN[N])->u32 { N + K }
"#,
    )?;
    let options = search_paths(&[root.path()]);
    let helper = "fn increment(v:u32)->u32 { v + u32:1 }";
    let body = r#"
    let width = increment(u32:7);
    let literal = 3;
    (data::scalar<width>(3), data::pair<8>((x, 7)), data::lanes<8>([9, 2]),
     data::pair<8>(if x == u8:7 { (10, x) } else { (x, 5) }), data::width<2>(literal),
     data::pair<8>(for (_, acc) in u32:0..u32:3 { (acc.0+2, acc.1+1) }((0, 0))))
"#;
    let result = "(u8,u8,u8,u8,u32,u8)";
    let source = format!(
        "use data; {helper} #[allow(combinational_block)] block b(clk,x:u8)->{result} {{ {body} }}"
    );
    let package = compile_with_options(
        &source,
        empty_package("test"),
        ResetBehavior::default(),
        &options,
    )?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    let reference = format!("import data; {helper} fn sample(x:u8)->{result} {{ {body} }}");
    let reference = xlsynth::convert_dslx_to_ir(
        &reference,
        Path::new("sample.x"),
        &xlsynth::DslxConvertOptions {
            additional_search_paths: vec![root.path()],
            ..Default::default()
        },
    )?
    .ir
    .get_function("__sample__sample")?;
    for (x, expected) in [
        (
            7,
            "(bits[8]:4,bits[8]:0,bits[8]:11,bits[8]:3,bits[32]:4,bits[8]:3)",
        ),
        (
            11,
            "(bits[8]:4,bits[8]:4,bits[8]:11,bits[8]:6,bits[32]:4,bits[8]:3)",
        ),
    ] {
        let actual = run_block(&package, x)?;
        assert_eq!(actual, IrValue::parse_typed(expected)?);
        assert_eq!(
            xlsynth_pir::libxls_bridge::value_to_libxls(&actual)?,
            reference.interpret(&[xlsynth::XlsIrValue::make_ubits(8, u64::from(x))?])?
        );
    }
    for body in [
        "data::scalar<8>(256)",
        "data::width<2>(3) as u8",
        "data::pair<8>((x, u7:3))",
        "data::lanes<8>([1, u9:2])",
        "data::pair<8>(if x == u8:0 { (x, 1) } else { (u7:2, x) })",
        "let n=match true { true=>u32:8, _=>x as u32 }; let alias=n; data::scalar<alias>(3)",
    ] {
        assert_import_rejected("data", body, "u8", &options);
    }
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
#[allow(combinational_block)]
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
    let source = "use fabric::stage; #[allow(combinational_block)] block b(clk, x: u8) -> u8 { stage::score(x) }";
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
fn imported_proc_spawning_tests_are_checked_and_preserve_function_positions()
-> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    let dslx = r#"// Keep blank lines so pretty-printing changes the source locations.

proc P {
    config() { () }
    init { () }
    next(state: ()) { () }
}

#[test]
fn spawn_test() { spawn P(); () }

// Deliberately indent the expression independently of the function.
pub fn tag(x: u8) -> u8 {
        x + u8:3
}"#;
    write_module(root.path(), "wire.x", dslx)?;
    let options = ImportOptions {
        dslx_stdlib_path: None,
        additional_search_paths: vec![root.path()],
    };
    let bx = "use wire; #[allow(combinational_block)] block run(clk, x: u8) -> u8 { wire::tag(x) }";
    let package = compile_with_options(
        bx,
        empty_package("design"),
        ResetBehavior::default(),
        &options,
    )?;
    assert_eq!(run_block(&package, 5)?, IrValue::make_ubits(8, 8)?);
    let function = package.get_fn("__wire__tag").ok_or("missing wire::tag")?;
    let add = function.get_node(function.ret_node_ref.ok_or("missing tag result")?);
    let [pos] = add.pos.as_deref().ok_or("missing tag source location")? else {
        panic!("expected one source location for the imported addition");
    };
    assert_eq!(
        Path::new(&package.file_table.id_to_path[&pos.fileno]),
        root.path().join("wire.x")
    );
    assert_eq!((pos.lineno, pos.colno), (13, 8));

    let invalid = dslx.replace("spawn P(); ()", "spawn P(); let bad: u8 = u32:1; ()");
    write_module(root.path(), "wire.x", &invalid)?;
    let error = check_module_with_options(&parse_module(bx)?, &options).unwrap_err();
    assert!(error.contains("size mismatch: u32 vs. u8"), "{error}");
    Ok(())
}

#[test]
fn imported_module_errors_report_the_original_path_across_checks() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    write_module(root.path(), "wire.x", "pub fn broken() -> u8 { missing }")?;
    let options = ImportOptions {
        dslx_stdlib_path: None,
        additional_search_paths: vec![root.path()],
    };
    let bx = parse_module("use wire; block run(clk, x: u8) -> u8 { x }")?;
    let first = check_module_with_options(&bx, &options).unwrap_err();
    let second = check_module_with_options(&bx, &options).unwrap_err();
    assert_eq!(first, second);
    let location = format!("{}:1:25-1:32", root.path().join("wire.x").display());
    assert!(first.contains(&location), "{first}");
    Ok(())
}

#[test]
fn imported_lambda_symbols_match_native_source_spans_across_checks() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    let modules = root.path().join("lambda search-path");
    write_module(
        &modules,
        "wire.x",
        "pub fn adjust(x: u8) -> u8 { x + u8:1 }",
    )?;
    // A native lambda symbol includes its filename and both endpoint columns.
    // Putting the import on this line makes relocation affect both columns.
    write_module(
        &modules,
        "wire_x.x",
        "import wire; pub fn score(x: u8, arr: u8[2]) -> u8[2] { map(arr, |i| wire::adjust(i + x)) }",
    )?;
    let bx = "use wire_x; #[allow(combinational_block)] block run(clk, x: u8, arr: u8[2]) -> u8[2] { wire_x::score(x, arr) }";
    let options = search_paths(&[&modules]);
    let compile = || {
        compile_with_options(
            bx,
            empty_package("design"),
            ResetBehavior::default(),
            &options,
        )
    };
    let first = compile()?;
    assert_eq!(first.to_string(), compile()?.to_string());

    let mut data = ImportData::new(None, &[&modules]);
    let caller = parse_and_typecheck(
        "import wire_x; fn main(x: u8, arr: u8[2]) -> u8[2] { wire_x::score(x, arr) }",
        "caller.x",
        "caller",
        &mut data,
    )?;
    let native = Parser::new(&caller.to_ir_text()?).parse_and_verify_package()?;
    let imported_names = |package: &Package, entry: &str| -> BTreeSet<String> {
        package
            .members
            .iter()
            .map(|m| &m.graph().name)
            .filter(|name| name.as_str() != entry)
            .cloned()
            .collect()
    };
    let expected = imported_names(&native, "__caller__main");
    assert!(
        expected
            .iter()
            .any(|name| name.starts_with("__wire_x__lambda_capture_struct_at_"))
    );
    assert_eq!(imported_names(&first, "run"), expected);
    Ok(())
}

#[test]
fn a_generated_public_fuzz_domain_can_be_passed_directly_between_calls()
-> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    write_module(
        root.path(),
        "wire.x",
        r#"#[fuzz_domain("SDomain")]
pub struct S { x: u8 }
pub fn make() -> SDomain { SDomain { x: () } }
pub fn read(_value: SDomain, x: u8) -> u8 { x + u8:5 }"#,
    )?;
    let options = ImportOptions {
        dslx_stdlib_path: None,
        additional_search_paths: vec![root.path()],
    };
    let package = compile_with_options(
        "use wire; #[allow(combinational_block)] block run(clk, x: u8) -> u8 { wire::read(wire::make(), x) }",
        empty_package("design"),
        ResetBehavior::default(),
        &options,
    )?;
    assert_eq!(run_block(&package, 7)?, IrValue::make_ubits(8, 12)?);
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
#[allow(combinational_block)]
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

#[test]
fn native_callbacks_can_compute_explicit_values_inside_imported_functions()
-> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    write_module(
        root.path(),
        "bias.x",
        r#"
struct Config { offset: u8 }
fn add<K: u8>(x: u8) -> u8 { x + K }
pub fn shifted<K: u8>(x: u8) -> u8 {
    let xs = map([x], add<{zero!<u8>()}>);
    map(xs, add<{Config { offset: K }.offset}>)[u32:0]
}
"#,
    )?;
    let options = search_paths(&[root.path()]);
    let package = compile_with_options(
        "use bias; #[allow(combinational_block)] block b(clk, x:u8)->u8 { bias::shifted<2>(x) }",
        empty_package("test"),
        ResetBehavior::default(),
        &options,
    )?;
    assert_eq!(run_block(&package, 4)?, IrValue::make_ubits(8, 6)?);
    Ok(())
}
