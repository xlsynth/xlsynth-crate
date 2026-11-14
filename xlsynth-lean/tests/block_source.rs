// SPDX-License-Identifier: Apache-2.0

//! Public BX source, saved IR and block selection contracts.

use xlsynth_lean::block::{CompiledBx, compile_bx, emit, select_block};
use xlsynth_pir::ir_parser::Parser;

fn compile(source: &str) -> CompiledBx {
    compile_bx(
        source,
        xlsynth_bx::empty_package("example"),
        Default::default(),
        &Default::default(),
    )
    .unwrap()
}

#[test]
fn checked_bx_and_saved_ir_emit_identical_raw_equations() {
    let compiled = compile("block counter(clk) -> u8 { reg n: u8 = 1; n := n + u8:1; n }");
    let bundle = compiled.emit_readable(None).unwrap();
    let saved = Parser::new(&compiled.package().to_string())
        .parse_and_validate_package()
        .unwrap();
    let raw = emit(&saved, select_block(&saved, None).unwrap()).unwrap();
    assert_eq!(bundle.files[0], ("Generated.lean".into(), raw));
}

#[test]
fn only_interface_structs_are_emitted_in_dependency_order() {
    let compiled = compile(
        r#"
struct Packet { data: Word }
struct Unused { recOn: u8 }
struct Word { bit: bool }
block show(clk, value: Packet) -> (Packet,) { (value,) }
"#,
    );
    let bundle = compiled.emit_readable(None).unwrap();
    let types = &bundle.files[1].1;
    assert!(types.find("structure «Word»").unwrap() < types.find("structure «Packet»").unwrap());
    assert!(!types.contains("Unused"));
}

#[test]
fn signed_public_output_reports_the_bx_port() {
    let compiled = compile("block machine(clk, data: u8) -> s8 { data as s8 }");
    let diagnostic = format!("{:#}", compiled.emit_readable(None).err().unwrap());
    assert!(
        diagnostic.contains("output out") && diagnostic.contains("signed"),
        "{diagnostic}"
    );
}

#[test]
fn unsupported_lean_field_reports_the_bx_declaration() {
    let compiled =
        compile("struct Packet { recOn: u8 } block machine(clk, x: Packet) -> u8 { x.recOn }");
    let diagnostic = format!("{:#}", compiled.emit_readable(None).err().unwrap());
    assert!(
        diagnostic.contains("Packet")
            && diagnostic.contains("recOn")
            && diagnostic.contains("Lean generates"),
        "{diagnostic}"
    );
}

#[test]
fn readable_emission_requires_a_selected_bx_declaration() {
    let package = Parser::new("package example\nblock preexisting() {}")
        .parse_and_validate_package()
        .unwrap();
    let compiled = compile_bx(
        "block a(clk) -> u8 { u8:1 } block b(clk) -> u8 { u8:2 }",
        package,
        Default::default(),
        &Default::default(),
    )
    .unwrap();
    assert!(compiled.emit_readable(None).is_err());
    assert!(compiled.emit_readable(Some("a")).is_ok());
    let diagnostic = format!(
        "{:#}",
        compiled.emit_readable(Some("preexisting")).err().unwrap()
    );
    assert!(
        diagnostic.contains("no checked BX declaration"),
        "{diagnostic}"
    );
}
