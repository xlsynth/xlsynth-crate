// SPDX-License-Identifier: Apache-2.0

//! Public BX source, saved IR and block selection contracts.

use xlsynth_lean::block::{CompiledBx, compile_bx, emit, select_block};
use xlsynth_pir::ir_parser::Parser;

#[cfg(feature = "lean-tests")]
#[path = "common/lean_bundle.rs"]
mod lean_bundle;

#[test]
fn readable_export_preserves_imported_block_identity() {
    let library = tempfile::tempdir().unwrap();
    std::fs::write(
        library.path().join("lib.bx"),
        "pub block counter(clk) -> u8 { reg n: u8 = 1; n := n + u8:1; n }",
    )
    .unwrap();
    let parent = "use lib; pub block parent(clk) -> u8 { let y = lib::counter(); y }";
    let local = "block _bx_lib_counter(clk, value: bool) -> bool { value }";
    let compile = |source: &str| {
        compile_bx(
            "design",
            source,
            xlsynth_bx::empty_package("example"),
            Default::default(),
            &xlsynth_bx::ImportOptions {
                additional_search_paths: vec![library.path()],
                ..Default::default()
            },
        )
        .unwrap()
    };
    let export = |compiled: &CompiledBx, symbol: &str, callable: &str| {
        let bundle = compiled.emit_readable(Some(symbol)).unwrap();
        let declaration = format!("abbrev {callable} := _root_.XlsReadable.eval");
        assert!(
            bundle
                .files
                .iter()
                .any(|(_, text)| text.contains(&declaration)),
            "{declaration}"
        );
        bundle
    };
    let public_name = "XLS.«lib».«counter»";
    export(&compile(parent), "_bx_lib_counter", public_name);
    let collision = compile(&format!("{parent}\n{local}"));
    export(
        &collision,
        "_bx_lib_counter",
        "XLS.«design».«_bx_lib_counter»",
    );
    let symbol = "_bx_lib_counter__bx_1";
    let bundle = export(&collision, symbol, public_name);
    let package = collision.package();
    let graph = select_block(package, Some(symbol)).unwrap();
    assert_eq!(bundle.files[0].1, emit(package, graph).unwrap());
    #[cfg(feature = "lean-tests")]
    lean_bundle::check_bundle(&bundle).unwrap();
}

/// Compile one self-contained BX fixture in an empty package.
fn compile(source: &str) -> CompiledBx {
    compile_bx(
        "design",
        source,
        xlsynth_bx::empty_package("example"),
        Default::default(),
        &Default::default(),
    )
    .unwrap()
}

// Verifies: The BX bundle and reparsed counter IR emit identical raw Lean.
// Catches: Raw equations that depend on BX metadata absent from saved IR.
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

// Verifies: Only used structs appear, in dependency order.
// Catches: Copying BX declaration order or emitting unreachable types.
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

// Verifies: Readable BX interfaces preserve public and private library names.
// Catches: Leaking linker names or accepting a duplicate library declaration.
#[test]
fn readable_interface_uses_linked_bx_library_structs() {
    let library = tempfile::tempdir().unwrap();
    std::fs::write(
        library.path().join("lib.bx"),
        "struct Word { value: u8 } pub struct Packet { payload: Word }",
    )
    .unwrap();
    let emit = |module_name| {
        compile_bx(
            module_name,
            "use lib; struct Packet { tag: bool }
         block demo(clk, p: lib::Packet, q: Packet) -> lib::Packet { p }",
            xlsynth_bx::empty_package("example"),
            Default::default(),
            &xlsynth_bx::ImportOptions {
                additional_search_paths: vec![library.path()],
                ..Default::default()
            },
        )
        .unwrap()
        .emit_readable(Some("demo"))
    };
    let bundle = emit("design").unwrap();
    let types = &bundle.files[1].1;
    assert!(
        types.contains("-- Source struct lib::Packet { payload }"),
        "{types}"
    );
    assert!(
        types.contains("-- Source struct design::Packet { tag }"),
        "{types}"
    );
    assert!(
        types.contains("-- Source struct lib::Word { value }"),
        "{types}"
    );
    assert!(!types.contains("$bx$"), "{types}");
    assert!(types.contains("«payload» :"), "{types}");
    assert_eq!(
        emit("lib").err().unwrap().to_string(),
        "duplicate BX source declaration 'lib::Packet' in the root and an imported library"
    );
}

// Negative test: A signed BX output must report its port and unsupported type.
#[test]
fn signed_public_output_reports_the_bx_port() {
    let compiled = compile("block machine(clk, data: u8) -> s8 { data as s8 }");
    let diagnostic = format!("{:#}", compiled.emit_readable(None).err().unwrap());
    assert!(
        diagnostic.contains("output out") && diagnostic.contains("signed"),
        "{diagnostic}"
    );
}

// Verifies: Fields with the listed Lean-generated names can be emitted.
// Catches: Rejecting a legal record that needs an explicit field accessor.
#[test]
fn source_struct_fields_can_use_lean_generated_member_names() {
    for field in [
        "rec",
        "recOn",
        "casesOn",
        "noConfusion",
        "noConfusionType",
        "ctorIdx",
        "_sizeOf_1",
        "_sizeOf_inst",
    ] {
        let compiled = compile(&format!(
            "struct Packet {{ {field}: u8 }} block machine(clk, x: Packet) -> u8 {{ x.{field} }}"
        ));
        compiled.emit_readable(None).unwrap();
    }
}

// Negative test: Reject a block, type or explicit accessor named `_root_`.
#[test]
fn root_qualifier_is_rejected_for_explicit_lean_declarations() {
    for (source, owner) in [
        ("block _root_(clk) -> u8 { u8:0 }", "block 'design::_root_'"),
        (
            "struct _root_ { x: u8 } block run(clk, p: _root_) -> u8 { p.x }",
            "type 'design::_root_'",
        ),
        (
            "struct Packet { _root_: u8, recOn: u8 } block run(clk, p: Packet) -> u8 { p.recOn }",
            "«Packet» field '_root_'",
        ),
    ] {
        let err = compile(source).emit_readable(None).err().unwrap();
        assert!(err.to_string().contains(owner), "{err:#}");
    }
}

// Negative test: Reject ambiguous tops and linked blocks lacking a BX source.
#[test]
fn readable_emission_requires_a_selected_bx_declaration() {
    let package = Parser::new("package example\nblock preexisting() {}")
        .parse_and_validate_package()
        .unwrap();
    let compiled = compile_bx(
        "design",
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
