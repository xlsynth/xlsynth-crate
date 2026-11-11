// SPDX-License-Identifier: Apache-2.0

use std::{error::Error, fs, path::Path};

use tempfile::tempdir;
use xlsynth_bx::ast::SourcePosition;
use xlsynth_bx::{
    BxWarningKind, ImportOptions, ResetBehavior, check_module, check_source_with_options, compile,
    compile_with_options, empty_package, lower_module, parse_module,
};

#[test]
fn required_block_timing_applies_to_every_block_defined_in_the_module() -> Result<(), String> {
    let source = r#"
block before(clk, x: u8) -> u8 { reg q: u8; q := x; q }

#![require_block_timing]
pub block after(clk, x: u8) -> u8 { reg q: u8; q := x; q }
block timed(clk, x: u8) -> u8 @1 { reg q: u8; q := x; q }

#[allow(missing_block_timing)]
block exempt(clk, x: u8) -> u8 { reg q: u8; q := x; q }
"#;
    let checked = check_module(&parse_module(source)?)?;
    let warnings: Vec<_> = checked
        .warnings()
        .iter()
        .map(|warning| (warning.declaration.as_str(), warning.kind))
        .collect();
    assert_eq!(
        warnings,
        [
            ("before", BxWarningKind::MissingBlockTiming),
            ("after", BxWarningKind::MissingBlockTiming),
        ]
    );
    Ok(())
}

#[test]
fn required_block_timing_stays_in_its_source_file() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let lib = dir.path().join("device.bx");
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    let required = "#![require_block_timing]";
    let library = "pub block stage(clk, x: u8) -> u8 { reg q: u8; q := x; q }";
    let caller = r#"use device::stage;
pub block top(clk, x: u8) -> u8 { let y = stage(x); y }"#;
    for (caller_attr, lib_attr, name, file) in [
        (required, "", "top", None),
        ("", required, "device::stage", Some(lib.as_path())),
    ] {
        fs::write(&lib, format!("{lib_attr}\n{library}"))?;
        let checked = check_source_with_options(&format!("{caller_attr}\n{caller}"), &options)?;
        let [warning] = checked.warnings() else {
            panic!(
                "expected one warning for {name}, got {:?}",
                checked.warnings()
            );
        };
        assert_eq!(warning.kind, BxWarningKind::MissingBlockTiming);
        assert_eq!(warning.declaration, name);
        assert_eq!(warning.file.as_deref(), file);
    }
    Ok(())
}

#[test]
fn rust_compile_helpers_deny_collected_warnings_by_default() -> Result<(), Box<dyn Error>> {
    let source = "block increment(clk, x: u8) -> u8 { x + u8:1 }\n\
                  pub block invert(clk, x: u8) -> u8 { !x }";
    let options = ImportOptions::default();
    let checked = check_source_with_options(source, &options)?;
    assert_eq!(
        checked.warnings(),
        check_module(&parse_module(source)?)?.warnings()
    );
    let warnings = checked.warnings();
    assert_eq!(warnings.len(), 2);
    assert_eq!(warnings[0].kind, BxWarningKind::CombinationalBlock);
    assert_eq!(warnings[0].declaration, "increment");
    assert_eq!(
        warnings[0].position,
        Some(SourcePosition { line: 1, column: 7 })
    );
    assert_eq!(warnings[1].declaration, "invert");
    assert_eq!(
        warnings[1].position,
        Some(SourcePosition {
            line: 2,
            column: 11
        })
    );
    let expected = "<input>:1:7: warning[combinational_block]: block 'increment' has no registers or child instances. Consider defining it as a function.\n\
                    <input>:2:11: warning[combinational_block]: block 'invert' has no registers or child instances. Consider defining it as a function.\n\
                    BX warnings treated as errors";
    let reset = ResetBehavior::default();
    assert_eq!(
        compile(source, empty_package("test"), reset).unwrap_err(),
        expected
    );
    assert_eq!(
        compile_with_options(source, empty_package("test"), reset, &options).unwrap_err(),
        expected
    );
    // A caller that consumes diagnostics can still lower the checked module.
    let package = lower_module(&checked, empty_package("test"), reset)?;
    assert_eq!(
        package.get_top_block().map(|b| b.name.as_str()),
        Some("invert")
    );
    Ok(())
}

#[test]
fn using_an_implicit_reset_as_data_keeps_a_leaf_block_combinational() -> Result<(), String> {
    let source = "block reset_asserted(clk, rst) -> bool { rst }";
    let checked = check_source_with_options(source, &ImportOptions::default())?;
    let [warning] = checked.warnings() else {
        panic!("expected one combinational block warning");
    };
    assert_eq!(warning.kind, BxWarningKind::CombinationalBlock);
    assert_eq!(warning.declaration, "reset_asserted");
    Ok(())
}

#[test]
fn redundant_struct_splats_warn_at_each_splat_and_compile_denies_them() -> Result<(), String> {
    let source = r#"struct S { a: u8, b: bool }
fn replace(x: S, a: u8, b: bool) -> S {
    let copy = S { ..x };
    let partial = S { a, ..copy };
    let first = S { a, b, ..partial };
    S { b, a, ..first }
}
"#;
    let checked = check_module(&parse_module(source)?)?;
    let sites: Vec<_> = checked
        .warnings()
        .iter()
        .map(|w| (w.kind, w.declaration.as_str(), w.position))
        .collect();
    assert_eq!(
        sites,
        [
            (
                BxWarningKind::UselessStructSplat,
                "replace",
                Some(SourcePosition {
                    line: 5,
                    column: 27
                })
            ),
            (
                BxWarningKind::UselessStructSplat,
                "replace",
                Some(SourcePosition {
                    line: 6,
                    column: 15
                })
            ),
        ]
    );
    let expected = "<input>:5:27: warning[useless_struct_splat]: all struct fields are supplied explicitly. Remove the trailing struct splat.\n\
                    <input>:6:15: warning[useless_struct_splat]: all struct fields are supplied explicitly. Remove the trailing struct splat.\n\
                    BX warnings treated as errors";
    let reset = ResetBehavior::default();
    assert_eq!(
        compile(source, empty_package("test"), reset).unwrap_err(),
        expected
    );
    let package = lower_module(&checked, empty_package("test"), reset)?;
    assert!(package.get_fn("replace").is_some());
    Ok(())
}

#[test]
fn a_base_only_copy_of_an_empty_struct_has_no_fields_to_carry() -> Result<(), String> {
    let checked = check_module(&parse_module(
        "struct Empty {} fn copy(x: Empty) -> Empty { Empty { ..x } }",
    )?)?;
    let [warning] = checked.warnings() else {
        panic!("unexpected warnings: {:?}", checked.warnings());
    };
    assert_eq!(warning.kind, BxWarningKind::UselessStructSplat);
    Ok(())
}

#[test]
fn useless_struct_splat_allowances_are_local_to_each_declaration() -> Result<(), String> {
    let source = r#"struct S { a: u8 }
const BASE = S { a: u8:0 };
#[allow(useless_struct_splat)]
const REPLACED = S { a: u8:1, ..BASE };
#[allow(useless_struct_splat)]
fn replace(x: S, a: u8) -> S { S { a, ..x } }
#[allow(combinational_block, useless_struct_splat)]
block quiet(clk, x: S, data: u8) -> S { S { a: data, ..x } }
#[allow(combinational_block)]
block noisy(clk, x: S, data: u8) -> S { S { a: data, ..x } }
const NOISY_CONST = S { a: u8:2, ..REPLACED };
"#;
    let checked = check_module(&parse_module(source)?)?;
    let mut warnings: Vec<_> = checked
        .warnings()
        .iter()
        .map(|warning| (warning.declaration.as_str(), warning.kind))
        .collect();
    warnings.sort();
    assert_eq!(
        warnings,
        [
            ("NOISY_CONST", BxWarningKind::UselessStructSplat),
            ("noisy", BxWarningKind::UselessStructSplat),
        ]
    );
    Ok(())
}

#[test]
fn imported_type_declarations_and_extern_signatures_own_their_splat_allowances()
-> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let declarations = r#"struct Size { n: u32 }
const BASE = Size { n: u32:1 };
#[allow(useless_struct_splat)]
enum Code : u32 { Read = Size { n: u32:2, ..BASE }.n }
#[allow(useless_struct_splat)]
struct Word<N: u32 = {Size { n: u32:3, ..BASE }.n}> { data: uN[N] }
fn read(w: Word<>) -> u3 { w.data }
#[allow(useless_struct_splat)]
extern fn adjust(x: Word<{Size { n: u32:4, ..BASE }.n}>) -> u4;
#[allow(useless_struct_splat)]
extern block stage(clk, x: Word<{Size { n: u32:5, ..BASE }.n}>) -> u5;
"#;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    let path = dir.path().join("decls.bx");
    fs::write(
        &path,
        declarations.replace("#[allow(useless_struct_splat)]", ""),
    )?;
    let checked = check_source_with_options("use decls;", &options)?;
    let owners: Vec<_> = checked
        .warnings()
        .iter()
        .map(|w| w.declaration.as_str())
        .collect();
    assert_eq!(
        owners,
        [
            "decls::Code",
            "decls::Word",
            "decls::adjust",
            "decls::stage"
        ]
    );
    fs::write(&path, declarations)?;
    let checked = check_source_with_options("use decls;", &options)?;
    assert!(checked.warnings().is_empty(), "{:?}", checked.warnings());
    Ok(())
}

#[test]
fn imported_struct_splat_warns_once_across_type_specializations() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let library = dir.path().join("records.bx");
    fs::write(
        &library,
        r#"pub struct Record<T> { data: T }
pub fn replace<T>(x: Record<T>, data: T) -> Record<T> {
    Record { data, ..x }
}"#,
    )?;
    let source = r#"use records;
fn byte(x: records::Record<u8>) -> records::Record<u8> { records::replace(x, u8:1) }
fn word(x: records::Record<u32>) -> records::Record<u32> { records::replace(x, u32:2) }
"#;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    let checked = check_source_with_options(source, &options)?;
    let [warning] = checked.warnings() else {
        panic!("unexpected warnings: {:?}", checked.warnings());
    };
    assert_eq!(warning.kind, BxWarningKind::UselessStructSplat);
    assert_eq!(warning.declaration, "records::replace");
    assert_eq!(warning.file.as_deref(), Some(library.as_path()));
    assert_eq!(
        warning.position,
        Some(SourcePosition {
            line: 3,
            column: 20
        })
    );
    Ok(())
}

#[test]
fn a_wire_shadowing_a_callback_keeps_its_template_combinational() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    let declarations = r#"
block child(clk, x: u8) -> u8 { reg q: u8; q := x; q }
pub block stage<child: fn(u8) -> u8>(clk, x: u8) -> u8 {
    let child = x;
    unroll_for! (_, a) in u8:0..u8:0 { let y = child(a); y }(x)
}
"#;
    fs::write(dir.path().join("device.bx"), declarations)?;
    let caller = r#"
fn id(x: u8) -> u8 { x }
block circuit(clk, x: u8) -> u8 { let y = stage<id>(x); y }
"#;
    for (source, name) in [
        (declarations, "stage"),
        ("use device::stage;", "device::stage"),
    ] {
        for caller in ["", caller] {
            let checked = check_source_with_options(&format!("{source}\n{caller}"), &options)?;
            let [warning] = checked.warnings() else {
                panic!(
                    "expected one warning for {name}, got {:?}",
                    checked.warnings()
                );
            };
            assert_eq!(warning.declaration, name);
            assert_eq!(warning.kind, BxWarningKind::CombinationalBlock);
        }
    }
    Ok(())
}

#[test]
fn imported_warnings_use_the_declaration_file_and_its_suppression() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let lib = dir.path().join("ip").join("stages.bx");
    fs::create_dir(lib.parent().unwrap())?;
    let stages = r#"#[allow(combinational_block)]
pub block wire(clk, x: u8) -> u8 { x }

pub block leaf(clk, x: u8) -> u8 { x }
pub block hold<T>(clk, x: T) -> T { x }
"#;
    fs::write(&lib, stages)?;
    let root = r#"
use ip::stages;
use ip::stages::leaf;
pub block circuit(clk, x: u8) -> u8 {
    let a = stages::leaf(x);
    let b = leaf(a);
    b
}
"#;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    let checked = check_source_with_options(root, &options)?;
    let [warning] = checked.warnings() else {
        panic!("unexpected warnings: {:?}", checked.warnings());
    };
    assert_eq!(warning.declaration, "ip::stages::leaf");
    assert_eq!(warning.file.as_deref(), Some(lib.as_path()));
    assert_eq!(
        warning.position,
        Some(SourcePosition {
            line: 4,
            column: 11
        })
    );
    assert_eq!(
        warning
            .display_with_root(Path::new("caller.bx"))
            .to_string(),
        format!(
            "{}:4:11: warning[combinational_block]: block 'ip::stages::leaf' has no registers or child instances. Consider defining it as a function.",
            lib.display()
        )
    );

    fs::write(
        &lib,
        stages.replace(
            "pub block leaf",
            "#[allow(combinational_block)]\npub block leaf",
        ),
    )?;
    assert!(
        check_source_with_options(root, &options)?
            .warnings()
            .is_empty()
    );
    let package = compile_with_options(
        root,
        empty_package("test"),
        ResetBehavior::default(),
        &options,
    )?;
    assert_eq!(
        package.get_top_block().map(|b| b.name.as_str()),
        Some("circuit")
    );
    Ok(())
}
