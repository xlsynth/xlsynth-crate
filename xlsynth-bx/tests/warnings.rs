// SPDX-License-Identifier: Apache-2.0

use std::{error::Error, fs, path::Path};

use tempfile::tempdir;
use xlsynth_bx::ast::SourcePosition;
use xlsynth_bx::{
    BxWarningKind, ImportOptions, ResetBehavior, check_module, check_source_with_options, compile,
    compile_with_options, empty_package, lower_module, parse_module,
};

#[test]
fn required_block_timing_warns_only_when_the_output_annotation_is_missing()
-> Result<(), Box<dyn Error>> {
    let source = r#"
#[require_block_timing]
block missing(clk, x: u8) -> u8 { reg q: u8; q := x; q }

#[require_block_timing]
block timed(clk, x: u8) -> u8 @1 { reg q: u8; q := x; q }

block ordinary(clk, x: u8) -> u8 { reg q: u8; q := x; q }
"#;
    let checked = check_module(&parse_module(source)?)?;
    let [warning] = checked.warnings() else {
        panic!("expected one warning, got {:?}", checked.warnings());
    };
    assert_eq!(warning.kind, BxWarningKind::MissingBlockTiming);
    assert_eq!(warning.declaration, "missing");
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
