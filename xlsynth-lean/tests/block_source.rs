// SPDX-License-Identifier: Apache-2.0

//! Public BX source, saved IR and block selection contracts.

use xlsynth_lean::block::{CompiledBx, compile_bx, emit, select_block};
use xlsynth_pir::ir_parser::Parser;

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

#[test]
fn checked_bx_and_saved_ir_emit_identical_raw_equations() {
    let compiled = compile("block counter(clk, rst) -> u8 { reg n: u8 = 1; n := n + u8:1; n }");
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

#[test]
fn checked_helpers_accept_defaulted_struct_arguments() {
    let source = r#"
struct Packet<N: u32, M: u32 = {N * u32:2}> { x: bits[N], y: bits[M] }
struct Tagged<N: u32 = {u32:9}> { value: u8 }
fn copy(packet: Packet<3>, tag: Tagged) -> (Packet<3>, Tagged<9>) { (packet, tag) }
fn advance(current: u6, value: u8) -> u6 {
    let pair = copy(Packet<3> { x: u3:0, y: current }, Tagged<9> { value: value });
    pair.0.y + (pair.1.value as u6)
}
"#;
    let bx = r#"
extern fn machine::advance(current: u6, value: u8) -> u6;
block accumulate(clk, rst, value: u8) -> u6 {
    reg current: u6 = 0;
    current := machine::advance(current, value);
    current
}
"#;
    let mut imports = xlsynth::dslx::ImportData::new(None, &[]);
    let module =
        xlsynth::dslx::parse_and_typecheck(source, "machine.x", "machine", &mut imports).unwrap();
    let compiled = xlsynth_lean::block::compile_bx_with_dslx(
        "design",
        bx,
        module,
        Default::default(),
        &Default::default(),
    )
    .unwrap();
    compiled.emit_readable(None).unwrap();
}

#[test]
fn indexed_arrays_use_record_fields_when_the_element_type_is_known() {
    let constructed = "[Slot { write: Write { data: a } }, Slot { write: Write { data: b } }]";
    for (inputs, array, next, named) in [
        ("a: Slot, b: Slot", "[a, b]", "last", true),
        ("a: Alias, b: Slot", "[a, b]", "last", true),
        (
            "c: bool, a: Slot[2], b: Slot[2]",
            "if c { a } else { b }",
            "last",
            true,
        ),
        (
            "c: bool, a: Alias[2], b: Slot[2]",
            "if c { a } else { b }",
            "last",
            true,
        ),
        // The indexed value's destination supplies the type of both constructors.
        ("a: u8, b: u8", constructed, "slot", true),
        // The block retains only the tuple shapes of these struct literals.
        ("a: u8, b: u8", constructed, "last", false),
        // One known operand does not establish the source type of the other.
        (
            "a: Slot, b: u8",
            "[a, Slot { write: Write { data: b } }]",
            "last",
            false,
        ),
    ] {
        let source = format!(
            r#"
struct Write {{ data: u8 }}
struct Slot {{ write: Write }}
type Alias = Slot;
fn machine({inputs}, index: u8, last: Slot) -> (u8, Slot) {{
    let slots = {array};
    let slot = slots[index];
    (slot.write.data, {next})
}}
"#
        );
        // Aliases and nominal record projections belong to the pure DSLX
        // helper. BX supplies its concrete port representation and
        // register feedback.
        let bx_inputs = inputs.replace("Alias", "Slot");
        let arguments = inputs
            .split(", ")
            .map(|param| param.split_once(':').unwrap().0)
            .collect::<Vec<_>>()
            .join(", ");
        let bx = format!(
            r#"
struct Write {{ data: u8 }}
struct Slot {{ write: Write }}
extern fn machine::machine({bx_inputs}, index: u8, last: Slot) -> (u8, Slot);
block storage(clk, {bx_inputs}, index: u8) -> u8 {{
    reg last: Slot;
    let result = machine::machine({arguments}, index, last);
    last := result.1;
    result.0
}}
"#
        );
        let mut imports = xlsynth::dslx::ImportData::new(None, &[]);
        let module =
            xlsynth::dslx::parse_and_typecheck(&source, "machine.x", "machine", &mut imports)
                .unwrap();
        let model = xlsynth_lean::block::compile_bx_with_dslx(
            "design",
            &bx,
            module,
            Default::default(),
            &Default::default(),
        )
        .unwrap()
        .emit_readable(None)
        .unwrap();
        let (_, readable) = model
            .files
            .iter()
            .find(|(file, _)| file == "ReadableGenerated.lean")
            .unwrap();
        let projections: Vec<_> = readable
            .lines()
            .map(str::trim)
            .filter(|line| {
                line.starts_with("let «slot» :")
                    || line.starts_with("let «slot_write» :")
                    || line.starts_with("let «slot_write_data» :")
            })
            .collect();
        let expected = if named {
            [
                "let «slot» : «machine».«Slot» := (let «indexed array» := «slots»; (_root_.Vector.getD «indexed array» («index»).toNat (_root_.Vector.back «indexed array»)))",
                "let «slot_write» : «machine».«Write» := («slot»).«write»",
                "let «slot_write_data» : _root_.BitVec 8 := («slot_write»).«data»",
            ]
        } else {
            [
                "let «slot» : Prod (Prod (BitVec 8) (Unit)) (Unit) := (let «indexed array» := «slots»; (_root_.Vector.getD «indexed array» («index»).toNat (_root_.Vector.back «indexed array»)))",
                "let «slot_write» : Prod (BitVec 8) (Unit) := («slot»).1",
                "let «slot_write_data» : _root_.BitVec 8 := («slot_write»).1",
            ]
        };
        assert_eq!(projections, expected, "{source}\n{readable}");
    }
}
