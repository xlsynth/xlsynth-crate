// SPDX-License-Identifier: Apache-2.0

//! Explicit native checks for generated readable block correspondence proofs.

use std::{fmt::Write as _, fs, path::Path, process::Command};

use anyhow::{Context, Result};
use xlsynth_lean::block::{LeanBlockBundle, compile_bx, emit, select_block};
use xlsynth_pir::ir_parser::Parser;

fn bx_bundle(source: &str) -> LeanBlockBundle {
    compile_bx(
        "design",
        source,
        xlsynth_bx::empty_package("example"),
        Default::default(),
        &Default::default(),
    )
    .unwrap()
    .emit_readable(None)
    .unwrap()
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn bx_module_and_struct_names_can_coincide_with_a_lean_declaration() -> Result<()> {
    for module in ["Circuit", "Nat", "hw.Nat"] {
        let mut bundle = compile_bx(
            module,
            "struct succ { value: u8 } block demo(clk, XLS: u8, p: succ) -> u8 { p.value + XLS }",
            xlsynth_bx::empty_package("example"),
            Default::default(),
            &Default::default(),
        )?
        .emit_readable(None)?;
        bundle.files.push((
            "SourceName.lean".into(),
            format!(
                "import ReadableGenerated\nexample (x : XlsReadable.Input) : XLS.{module}.succ := x.p\nexample : _root_.Nat.succ 0 = 1 := rfl\n"
            ),
        ));
        check_bundle(bundle, module);
    }
    Ok(())
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn bx_preserves_each_imported_struct_spelling_in_either_port_order() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fs::create_dir(dir.path().join("left"))?;
    fs::write(
        dir.path().join("left/ops.x"),
        "pub struct Packet { data: u8 } pub type Message = Packet; pub type Data = Packet;
         pub struct Other { data: u8 }",
    )?;
    // The checked BX type is identical for direct, message and data. Their
    // Lean annotations must still reflect the type named at each source site.
    for params in [
        "direct: ops::Packet, message: Message, data: ops::Data",
        "data: ops::Data, message: Message, direct: ops::Packet",
    ] {
        let source = format!(
            "use left::ops; use left::ops::Message;
            block inspect(clk, {params}, other: ops::Other) -> (u8, u8, u8, u8) {{
                (direct.data, message.data, data.data, other.data)
            }}"
        );
        let mut bundle = compile_bx(
            "design",
            &source,
            xlsynth_bx::empty_package("example"),
            Default::default(),
            &xlsynth_bx::ImportOptions {
                additional_search_paths: vec![dir.path()],
                ..Default::default()
            },
        )?
        .emit_readable(None)?;
        let generated = &bundle
            .files
            .iter()
            .find(|(p, _)| p == "ReadableGenerated.lean")
            .unwrap()
            .1;
        for (field, ty) in [
            ("direct", "Packet"),
            ("message", "Message"),
            ("data", "Data"),
            ("other", "Other"),
        ] {
            assert!(
                generated.contains(&format!("«{field}» : XLS.«left».«ops».«{ty}»")),
                "{generated}"
            );
        }
        bundle.files.push((
            "AliasPorts.lean".into(),
            r#"import ReadableCorrespondence
open XlsReadable
example : XLS.left.ops.Data = XLS.left.ops.Packet := rfl
example : XLS.left.ops.Message = XLS.left.ops.Packet := rfl
#check_failure fun (x : Input) => (show XLS.left.ops.Other from x.direct)
example (q : State) (x : Input) :
  observe q x = (x.direct.data, x.message.data, x.data.data, x.other.data) := rfl
"#
            .into(),
        ));
        check_bundle(
            bundle,
            "BX ports retaining direct and aliased DSLX type names",
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn imported_struct_alias_survives_bx_register_and_nested_ports() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fs::write(
        dir.path().join("ops.x"),
        r#"
pub struct Packet { zebra: u8, apple: u8 }
pub type Inner = Packet;
pub type Alias = Inner;
pub fn id(x: Alias) -> Alias { x }
"#,
    )?;
    let mut bundle = compile_bx(
        "design",
        r#"
use ops;
struct Envelope { item: ops::Alias }
block hold(clk, src: Envelope) -> (ops::Alias, u8) {
    reg stored: ops::Alias;
    stored := ops::id(src.item);
    (stored, src.item.zebra)
}
"#,
        xlsynth_bx::empty_package("example"),
        Default::default(),
        &xlsynth_bx::ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        },
    )?
    .emit_readable(None)?;
    let types = &bundle.files[1].1;
    assert!(types.contains("«item» : XLS.«ops».«Alias»"), "{types}");
    bundle.files.push((
        "PublicInterface.lean".into(),
        r#"
import ReadableCorrespondence
open XlsReadable
example : XLS.ops.Alias = XLS.ops.Inner := rfl
example : XLS.ops.Inner = XLS.ops.Packet := rfl
example (q : State) (x : Input) : (step q x).stored = x.src.item := rfl
example (q : State) (x : Input) : observe q x = (q.stored, x.src.item.zebra) := rfl
"#
        .into(),
    ));
    check_bundle(
        bundle,
        "BX fields, registers and outputs with a linked DSLX alias",
    );
    Ok(())
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn bx_and_imported_dslx_fields_keep_their_source_names() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fs::write(
        dir.path().join("field0.x"),
        r#"
pub struct Packet { recOn: u8, valid: bool }
pub fn read(p: Packet) -> u8 { p.recOn }
"#,
    )?;
    let compiled = compile_bx(
        "design",
        r#"
use field0;
struct Outputs { seen: bool }
struct Report { result: Outputs, packet: field0::Packet }
struct Packet { Outputs: u8, design: u8, result: Outputs }
struct Result { recOn: u8, other_seen: bool, report_seen: bool }
block monitor(clk, p: field0::Packet, other: Packet, report: Report) -> Result {
    Result { recOn: field0::read(p), other_seen: other.result.seen, report_seen: report.result.seen }
}
"#,
        xlsynth_bx::empty_package("linked_ir"),
        Default::default(),
        &xlsynth_bx::ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        },
    )?;
    let mut bundle = compiled.emit_readable(None)?;
    let types = &bundle.files[1].1;
    assert!(types.contains("«result» : «Outputs»"), "{types}");
    bundle.files.push((
        "SourceFields.lean".into(),
        r#"
import ReadableCorrespondence
open XlsReadable
def packet : XLS.field0.Packet := ⟨41, true⟩
#guard packet.recOn == 41 && packet.valid
example (p : XLS.field0.Packet) : (match p with | ⟨n, _⟩ => n) = p.recOn := by cases p; rfl
example (q : State) (x : Input) :
    observe q x = (x.p.recOn, x.other.result.seen, x.report.result.seen) := rfl
example (q : XlsBlock.State) (x : XlsBlock.Input) :
    (XlsBlock.observe q x).recOn = x.p.1 := rfl
"#
        .into(),
    ));
    check_bundle(bundle, "BX and imported DSLX field names");
    Ok(())
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn selected_bx_block_and_dslx_function_share_a_module() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fs::write(
        dir.path().join("common.x"),
        "pub fn inc(x: u8) -> u8 { x + u8:1 }",
    )?;
    let mut bundle = compile_bx(
        "common",
        "use common; block controller(clk, x: u8) -> u8 { common::inc(x) }",
        xlsynth_bx::empty_package("pkg"),
        Default::default(),
        &xlsynth_bx::ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        },
    )?
    .emit_readable(None)?;
    bundle.files.push((
        "PublicBlock.lean".into(),
        r#"import ReadableCorrespondence
example (q : XlsReadable.State) (x : XlsReadable.Input) :
    XLS.common.controller q x = XlsReadable.eval q x := rfl
#guard XLS.common.inc 41 == 42
"#
        .into(),
    ));
    check_bundle(bundle, "public BX and DSLX declarations");
    Ok(())
}

#[test]
fn module_owner_can_be_shared_by_disjoint_bx_and_dslx_declarations() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fs::write(
        dir.path().join("bus.x"),
        "pub struct Packet { data: u8 } pub type Message = Packet;",
    )?;
    let emit = |source| {
        compile_bx(
            "bus",
            source,
            xlsynth_bx::empty_package("linked_ir"),
            Default::default(),
            &xlsynth_bx::ImportOptions {
                additional_search_paths: vec![dir.path()],
                ..Default::default()
            },
        )?
        .emit_readable(None)
    };
    emit(
        "use bus; struct Local { x: u8 }
          block copy(clk, a: Local, b: bus::Packet) -> (u8, u8) { (a.x, b.data) }",
    )?;
    emit(
        "use bus::Message; struct Packet { x: u8 }
          block copy(clk, p: Packet) -> u8 { p.x }",
    )?;
    let collision = emit(
        "use bus; struct Packet { value: u8 }
          block copy(clk, a: Packet, b: bus::Message) -> (u8, u8) { (a.value, b.data) }",
    );
    assert_eq!(
        collision.err().unwrap().to_string(),
        "duplicate source declaration 'bus::Packet' in BX 'bus.bx' and DSLX 'bus.x'; rename one declaration"
    );
    Ok(())
}

#[test]
fn module_owner_cannot_supply_declarations_under_a_source_type() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fs::create_dir(dir.path().join("design"))?;
    fs::write(
        dir.path().join("design/Packet.x"),
        "pub fn get() -> u8 { u8:7 }",
    )?;
    fs::write(dir.path().join("design.x"), "pub struct Packet { get: u8 }")?;
    fs::write(
        dir.path().join("ops.x"),
        "import design.Packet as packet; pub fn call() -> u8 { packet::get() }",
    )?;
    let emit = |owner, source| {
        compile_bx(
            owner,
            source,
            xlsynth_bx::empty_package("pkg"),
            Default::default(),
            &xlsynth_bx::ImportOptions {
                additional_search_paths: vec![dir.path()],
                ..Default::default()
            },
        )?
        .emit_readable(None)
    };
    // Only emitted declarations claim a public namespace.
    emit(
        "design",
        "use ops; struct Packet { get: u8 }
         block demo(clk, p: Packet) -> u8 { p.get }",
    )?;
    for (owner, source) in [
        (
            "design",
            "use ops; struct Packet { get: u8 }
         block demo(clk, p: Packet) -> u8 { p.get + ops::call() }",
        ),
        (
            "design.Packet",
            "use design; block get(clk, p: design::Packet) -> u8 { p.get }",
        ),
    ] {
        let err = emit(owner, source).err().unwrap().to_string();
        assert_eq!(
            err,
            "source module 'design.Packet' would emit declarations under the Lean namespace of source type 'design::Packet' (XLS.design.Packet); rename the type or module"
        );
    }
    Ok(())
}

#[test]
fn module_owner_is_shared_by_transitive_imports() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fs::write(dir.path().join("bus.x"), "pub fn get() -> u8 { u8:3 }")?;
    for name in ["left", "right"] {
        fs::write(
            dir.path().join(format!("{name}.x")),
            "import bus; pub fn get() -> u8 { bus::get() }",
        )?;
    }
    let bx = "use left; use right; block both(clk) -> u8 { left::get() + right::get() }";
    let options = xlsynth_bx::ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    for module in ["design", "bus"] {
        let compiled = compile_bx(
            module,
            bx,
            xlsynth_bx::empty_package("pkg"),
            Default::default(),
            &options,
        )?;
        compiled.emit_readable(None)?;
    }
    Ok(())
}

#[test]
fn module_owner_cannot_come_from_independent_dslx_inputs() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fs::write(dir.path().join("common.x"), "pub fn get() -> u8 { u8:3 }")?;
    fs::write(
        dir.path().join("wrapper.x"),
        "import common; pub fn get() -> u8 { common::get() }",
    )?;
    let options = xlsynth_bx::ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    for (name, source, import) in [
        ("common", "pub fn unrelated() -> u8 { u8:1 }", "use common;"),
        (
            "external",
            "import common; pub fn unrelated() -> u8 { u8:1 }",
            "use wrapper::get;",
        ),
    ] {
        let mut imports = xlsynth::dslx::ImportData::new(None, &[dir.path()]);
        let module =
            xlsynth::dslx::parse_and_typecheck(source, &format!("{name}.x"), name, &mut imports)?;
        let compiled = xlsynth_lean::block::compile_bx_with_dslx(
            "design",
            &format!("{import} block demo(clk, x: u8) -> u8 {{ x }}"),
            module,
            Default::default(),
            &options,
        )?;
        let err = compiled.emit_readable(None).err().unwrap().to_string();
        assert!(
            err.contains("DSLX module 'common' was checked independently"),
            "{err}"
        );
        assert!(err.contains("remove the redundant BX use"), "{err}");
    }
    Ok(())
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn module_owner_uses_the_full_dotted_subject() -> Result<()> {
    let dir = tempfile::tempdir()?;
    for package in ["left", "right"] {
        let root = dir.path().join(package);
        fs::create_dir(&root)?;
        fs::write(
            root.join("myarbiter.x"),
            format!(
                "pub fn compute() -> u8 {{ u8:7 }} pub fn {package}_value() -> u8 {{ compute() }}"
            ),
        )?;
    }
    let options = xlsynth_bx::ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    let emit = |name, uses: &str| -> Result<_> {
        let source = format!("{uses} block demo(clk, x: u8) -> u8 {{ x }}");
        compile_bx(
            name,
            &source,
            xlsynth_bx::empty_package("pkg"),
            Default::default(),
            &options,
        )?
        .emit_readable(None)
    };
    let repeated = "use left::myarbiter; use left::myarbiter::left_value;";
    emit("design", repeated)?;
    emit("myarbiter", repeated)?;
    let source = "use right::myarbiter::right_value; use left::myarbiter::left_value;
                  block demo(clk) -> u8 { right_value() + left_value() }";
    let mut bundle = compile_bx(
        "myarbiter",
        source,
        xlsynth_bx::empty_package("pkg"),
        Default::default(),
        &options,
    )?
    .emit_readable(None)?;
    bundle.files.push(("DottedOwners.lean".into(),
        "import ReadableGenerated\n#check XLS.left.myarbiter.compute\n#check XLS.right.myarbiter.compute\n".into()));
    check_bundle(bundle, "separate dotted DSLX owners with the same basename");
    Ok(())
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn bx_imports_keep_source_fields_through_linker_renames() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("picker.x");
    fs::write(
        &path,
        r#"
pub type Byte = u8;
pub struct Packet { zebra: Byte, apple: u8 }
fn first(original: Packet) -> u8 {
    let result = for (_i, carry): (u32, Packet) in u32:0..u32:2 {
        Packet { zebra: carry.zebra, apple: carry.apple + u8:1 }
    }(original);
    result.zebra
}
pub fn pick(actual: Packet) -> u8 { first(actual) }
pub fn id<N: u32>(argument: uN[N]) -> uN[N] { argument }
"#,
    )?;
    // Preoccupy the names of both the entry and its private callee. Matching
    // either declaration to its original symbol would type the wrong graph.
    let linked = Parser::new(
        r#"package chips
fn __picker__pick(other: bits[8] id=1) -> bits[8] { ret c: bits[8] = not(other, id=2) }
fn __picker__first(other: bits[8] id=3) -> bits[8] { ret c: bits[8] = not(other, id=4) }
"#,
    )
    .parse_and_validate_package()?;
    let compiled = compile_bx(
        "design",
        r#"
use picker;
block probe(clk, packet: picker::Packet, flag: bool) -> (u8, bool) {
    (picker::id(picker::pick(packet)), picker::id(flag))
}
"#,
        linked,
        Default::default(),
        &xlsynth_bx::ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        },
    )?;
    fs::write(path, "the original source is no longer available")?;
    let mut bundle = compiled.emit_readable(None)?;
    let executable = &bundle
        .files
        .iter()
        .find(|(file, _)| file == "ReadableGenerated.lean")
        .unwrap()
        .1;
    for (name, parameter) in [
        ("pick", "actual"),
        ("first", "original"),
        ("id<8>", "argument"),
        ("id<1>", "argument"),
    ] {
        assert!(
            executable.contains(&format!("def _root_.XLS.«picker».«{name}» («{parameter}»")),
            "{executable}"
        );
    }
    bundle.files.push((
        "PublicInterface.lean".into(),
        r#"
import ReadableCorrespondence
open XlsReadable
example (q : State) (x : Input) : observe q x = (x.packet.zebra, x.flag) := by rfl
"#
        .into(),
    ));
    check_bundle(bundle, "ordinary BX imports and renamed linked functions");
    Ok(())
}

// Inline the external DSLX function with XLS so the straight-line raw
// exporter sees its field projection in the same graph as the BX constructor.
#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn bx_field_order_across_linked_dslx() {
    let ir = xlsynth::convert_dslx_to_ir_text(
        include_str!("fixtures/bx_field_order/picker.x"),
        Path::new("picker.x"),
        &Default::default(),
    )
    .unwrap()
    .ir;
    let package = Parser::new(&ir).parse_and_validate_package().unwrap();
    let compiled = compile_bx(
        "field_order",
        include_str!("fixtures/bx_field_order/field_order.bx"),
        package,
        Default::default(),
        &Default::default(),
    )
    .unwrap();
    let inlined =
        xlsynth::optimize_ir_text(&compiled.package().to_string(), "field_order").unwrap();
    let package = Parser::new(&inlined).parse_and_validate_package().unwrap();
    let generated = emit(
        &package,
        select_block(&package, Some("field_order")).unwrap(),
    )
    .unwrap();
    check_bundle(
        LeanBlockBundle {
            files: vec![
                ("Generated.lean".into(), generated),
                (
                    "FieldOrder.lean".into(),
                    include_str!("fixtures/bx_field_order/FieldOrder.lean").into(),
                ),
            ],
        },
        "BX field order through linked DSLX",
    );
}

// Declaration order differs from both constructor and alphabetical order.
// The two fields have identical widths, so a shape-only check misses a swap.
#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn bx_fields_keep_declaration_order_after_local_rename() {
    const BX: &str = r#"
struct Pair { zebra: bool, apple: bool }
struct Ports { pair: Pair, selected: bool }
pub block machine(clk, left: bool, right: bool) -> Ports {
    let assembled = Pair { apple: right, zebra: left };
    Ports { selected: assembled.zebra, pair: assembled }
}
"#;
    const PROOF: &str = r#"
import ReadableCorrespondence
open XlsReadable
example (q : State) (x : Input) :
    (observe q x).1.zebra = x.left ∧ (observe q x).1.apple = x.right ∧
    (observe q x).2 = x.left := by
  rcases x with ⟨left, right⟩
  cases left <;> cases right <;> exact ⟨rfl, rfl, rfl⟩
example (q : XlsBlock.State) (x : XlsBlock.Input) :
    (XlsBlock.observe q x).pair = (x.left, x.right, ()) ∧
    (XlsBlock.observe q x).selected = x.left := ⟨rfl, rfl⟩
"#;
    for name in ["assembled", "renamed"] {
        let mut bundle = bx_bundle(&BX.replace("assembled", name));
        bundle
            .files
            .push(("PublicInterface.lean".into(), PROOF.into()));
        check_bundle(bundle, name);
    }
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn bx_positional_outputs_and_simultaneous_registers() {
    let source = r#"
struct InputPacket { sample: (bool, u8) }
block swap(clk, rst, x: InputPacket) -> (bool, u8) {
    reg left: u8 = 0;
    reg right: u8 = 3;
    left := right;
    right := left;
    (x.sample.0, left)
}
"#;
    let mut bundle = bx_bundle(source);
    bundle.files.push((
        "PublicInterface.lean".into(),
        r#"
import ReadableCorrespondence
open XlsReadable
example (q : State) (x : Input) : observe q x = (x.x.sample.1, q.left) := by rfl
example (q : State) (x : Input) :
    step q { x with rst := false } = ⟨q.right, q.left⟩ := by rfl
example (q : State) (x : Input) : (reset q x).next = ⟨0, 3⟩ := by rfl
"#
        .into(),
    ));
    check_bundle(bundle, "swap registers");
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn bx_unit_output_and_singleton_tuple_register() {
    let mut bundle = bx_bundle(
        r#"
block machine(clk, rst, x: (bool,)) -> () {
    reg one: (bool,) = (false,);
    one := x;
    ()
}
"#,
    );
    bundle.files.push((
        "PublicInterface.lean".into(),
        r#"
import ReadableCorrespondence
open XlsReadable
example (q : State) (x : Input) : observe q x = () := by rfl
example (q : State) (x : Input) : (step q { x with rst := false }).one.1 = x.x.1 := by rfl
"#
        .into(),
    ));
    check_bundle(bundle, "singleton register and unit output");
    check_bundle(bx_bundle("block empty(clk) -> () { () }"), "empty block");
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn bx_fields_keep_lean_names() {
    let mut bundle = bx_bundle(
        r#"
struct Word { mk: u8 }
struct Packet { design: u8, _root_: u8, word: Word }
block instReprPacket(clk, rst, BitVec: u8, packet: Packet) -> (u8, Packet) {
    reg mk: Packet;
    mk := packet;
    (BitVec, mk)
}
"#,
    );
    bundle.files.push((
        "PublicInterface.lean".into(),
        r#"
import ReadableCorrespondence
open XlsReadable
#check XLS.design.Word.Constructors.mk
#synth Repr XLS.design.Packet
example : XLS.design.instReprPacket = eval := rfl
example (q : State) (x : Input) : (observe q x).2.word.mk = q.mk.word.mk := by rfl
example (q : State) (x : Input) : (step q x).mk.«_root_» = x.packet.«_root_» := by rfl
"#
        .into(),
    ));
    check_bundle(bundle, "qualified constructors and fields");
}

// Assertions are named from IR nodes and always have Lean type Bool.
#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn raw_assertion_field_named_bool_compiles() {
    let block = r#"
package flags
top block flags(en: bits[1], BitVec: bits[1], other: bits[1]) {
  en: bits[1] = input_port(name=en, id=1)
  tok: token = literal(value=token, id=2)
  Bool: token = assert(tok, en, message="first", label="first", id=3)
  second: token = assert(Bool, en, message="second", label="second", id=4)
  BitVec: () = output_port(en, name=BitVec, id=5)
  other: () = output_port(en, name=other, id=6)
}
"#;
    let package = Parser::new(block).parse_and_validate_package().unwrap();
    let source = emit(&package, select_block(&package, None).unwrap()).unwrap();
    check_bundle(
        LeanBlockBundle {
            files: vec![("Generated.lean".into(), source)],
        },
        "raw output and assertion field types",
    );
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn prelinked_helpers_drive_a_bx_register() {
    let source = r#"
fn sum_fields(left: u8, right: u8) -> u8 { left + right }
fn advance(left: u8, right: u8, previous: u8) -> u8 { sum_fields(left, right) + previous }
"#;
    let ir =
        xlsynth::convert_dslx_to_ir(source, Path::new("machine.x"), &Default::default()).unwrap();
    let package = Parser::new(&ir.ir.to_string())
        .parse_and_validate_package()
        .unwrap();
    let bx = r#"
extern fn machine::advance(left: u8, right: u8, previous: u8) -> u8;
block pipeline(clk, rst, left: u8, right: u8) -> u8 {
    reg count: u8 = 0;
    count := machine::advance(left, right, count);
    count
}
"#;
    let mut bundle = xlsynth_lean::block::compile_bx(
        "design",
        bx,
        package,
        Default::default(),
        &Default::default(),
    )
    .unwrap()
    .emit_readable(None)
    .unwrap();
    let model = &bundle
        .files
        .iter()
        .find(|(name, _)| name == "Generated.lean")
        .unwrap()
        .1;
    for helper in ["__machine__sum_fields", "__machine__advance"] {
        assert!(
            model.contains(&format!("_root_.XlsBlock.Helpers.«{helper}»")),
            "{model}"
        );
    }
    bundle.files.push((
        "CheckCalls.lean".into(),
        r#"import ReadableCorrespondence
open XlsReadable
private def stimulus : Input := { rst := false, left := 13, right := 5 }
private def initial : State := { count := 7 }
#guard observe initial stimulus == 7
#guard (step initial stimulus).count == 25
"#
        .into(),
    ));
    check_bundle(bundle, "nested prelinked calls across a register update");
}

fn checked_helper_bundle(bx: &str, source: &str, paths: &[&Path]) -> Result<LeanBlockBundle> {
    let mut imports = xlsynth::dslx::ImportData::new(None, paths);
    let module = xlsynth::dslx::parse_and_typecheck(source, "machine.x", "machine", &mut imports)?;
    xlsynth_lean::block::compile_bx_with_dslx(
        "design",
        bx,
        module,
        Default::default(),
        &Default::default(),
    )?
    .emit_readable(None)
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn linked_field_order_uses_the_checked_helper() {
    let mut imports = xlsynth::dslx::ImportData::new(None, &[]);
    let module = xlsynth::dslx::parse_and_typecheck(
        include_str!("fixtures/bx_field_order/picker.x"),
        "picker.x",
        "picker",
        &mut imports,
    )
    .unwrap();
    let compiled = xlsynth_lean::block::compile_bx_with_dslx(
        "design",
        include_str!("fixtures/bx_field_order/field_order.bx"),
        module,
        Default::default(),
        &Default::default(),
    )
    .unwrap();
    let mut bundle = compiled.emit_readable(None).unwrap();
    bundle.files.push((
        "FieldOrder.lean".into(),
        include_str!("fixtures/bx_field_order/FieldOrder.lean").into(),
    ));
    bundle.files.push((
        "NamedFields.lean".into(),
        r#"import ReadableCorrespondence
open XlsReadable
example (q : State) (x : Input) :
    (observe q x).1.zebra = x.left ∧ (observe q x).1.apple = x.right ∧
    (observe q x).2 = x.left := by
  rcases x with ⟨left, right⟩
  cases left <;> cases right <;> exact ⟨rfl, rfl, rfl⟩
"#
        .into(),
    ));
    check_bundle(
        bundle,
        "BX fields and an unoptimized linked DSLX projection",
    );
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn nested_helper_calls() {
    let modules = tempfile::tempdir().unwrap();
    fs::write(
        modules.path().join("left_side.x"),
        r#"
pub struct Packet { base: u8, delta: u4 }
pub fn advance(command: Packet) -> Packet {
    Packet { base: command.base + signex(command.delta, command.base), ..command }
}
"#,
    )
    .unwrap();
    let source = r#"
import left_side as foreign;
struct Packet { left: u8, right: u4 }
fn advance(request: foreign::Packet, previous: u8) -> u8 {
    foreign::advance(request).base + previous
}
fn machine(packet: Packet, count: u8) -> (u8, u8) {
    let request = foreign::Packet { base: packet.left, delta: packet.right };
    (count, advance(request, count))
}
"#;
    let bx = r#"
struct Packet { left: u8, right: u4 }
extern fn machine::machine(packet: Packet, count: u8) -> (u8, u8);
block pipeline(clk, rst, packet: Packet) -> u8 {
    reg count: u8 = 0;
    let result = machine::machine(packet, count);
    count := result.1;
    result.0
}
"#;
    let mut bundle = checked_helper_bundle(bx, source, &[modules.path()]).unwrap();
    let (_, model) = bundle
        .files
        .iter()
        .find(|(name, _)| name == "ReadableGenerated.lean")
        .unwrap();
    for owner in ["left_side", "machine"] {
        assert!(model.contains(&format!("_root_.XLS.«{owner}».«advance»")));
    }
    bundle.files.push((
        "CheckCalls.lean".into(),
        r#"
import ReadableCorrespondence
open XlsReadable
private def stimulus : Input := { rst := false, packet := { left := 13, right := 5 } }
private def initial : State := { count := 7 }
#guard (XLS.left_side.advance (command := { base := 13, delta := 5 })).base == 18
#guard observe initial stimulus == 7
#guard (step initial stimulus).count == 25
"#
        .into(),
    ));
    check_bundle(bundle, "nested calls across a register update");
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn computed_enum_dispatch_in_checked_helper() {
    for (control, choose, subject) in [
        (
            "use_new: bool",
            "if use_new { incoming } else { sender }",
            "chosen",
        ),
        (
            "mode: u2",
            "match mode { u2:0 => incoming, u2:1 => sender, _ => Sender::C }",
            "chosen",
        ),
        (
            "use_new: bool",
            "if use_new { incoming } else { sender }",
            "sender",
        ),
    ] {
        let control_name = control.split_once(':').unwrap().0;
        let source = format!(
            r#"
enum Sender : u2 {{ A = 0, B = 1, C = 2 }}
fn route({control}, incoming: Sender, a: bool, b: bool, c: bool,
         sender: Sender) -> (bool, Sender) {{
    let chosen = {choose};
    let payload = match {subject} {{ Sender::A => a, Sender::B => b, _ => c }};
    (payload, chosen)
}}
"#
        );
        let bx = format!(
            r#"
extern fn machine::route({control}, incoming: u2, a: bool, b: bool, c: bool, sender: u2) -> (bool, u2);
block router(clk, rst, {control}, incoming: u2, a: bool, b: bool, c: bool) -> bool {{
    reg sender: u2 = 0;
    let pair = machine::route({control_name}, incoming, a, b, c, sender);
    sender := pair.1;
    pair.0
}}
"#
        );
        let bundle = checked_helper_bundle(&bx, &source, &[]).unwrap();
        let readable = &bundle
            .files
            .iter()
            .find(|(name, _)| name == "ReadableGenerated.lean")
            .unwrap()
            .1;
        assert!(readable.contains("| .«A» =>") && readable.contains("| .«B» =>"));
        check_bundle(bundle, choose);
    }
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn independent_enum_inputs_pass_through_symbolically() {
    let source = r#"
enum Tag : u2 { A = 0, B = 1, C = 2 }
fn pass(a: Tag, b: Tag, c: Tag, d: Tag, e: Tag,
        f: Tag, g: Tag, h: Tag, i: Tag, j: Tag)
        -> (Tag, Tag, Tag, Tag, Tag, Tag, Tag, Tag, Tag, Tag) {
    (a, b, c, d, e, f, g, h, i, j)
}
"#;
    let bx = r#"
extern fn machine::pass(a: u2, b: u2, c: u2, d: u2, e: u2, f: u2, g: u2, h: u2, i: u2, j: u2)
    -> (u2, u2, u2, u2, u2, u2, u2, u2, u2, u2);
block tags(clk, a: u2, b: u2, c: u2, d: u2, e: u2, f: u2, g: u2, h: u2, i: u2, j: u2)
    -> (u2, u2, u2, u2, u2, u2, u2, u2, u2, u2) {
    machine::pass(a, b, c, d, e, f, g, h, i, j)
}
"#;
    let bundle = checked_helper_bundle(bx, source, &[]).unwrap();
    let (_, correspondence) = bundle
        .files
        .iter()
        .find(|(name, _)| name == "ReadableCorrespondence.lean")
        .unwrap();
    // Fail before asking Lean to elaborate 4^10 combinations of the inputs.
    assert!(
        !correspondence.contains("all_goals cases"),
        "pass-through correspondence splits independent enum inputs"
    );
    check_bundle(bundle, "ten independent enum inputs");
}

fn compile_lean_and_xls_machine(
    bx: &str,
    source: &str,
) -> Result<(LeanBlockBundle, xlsynth::IrFunction)> {
    let path = Path::new("machine.x");
    let bundle = checked_helper_bundle(bx, source, &[]).context("compile BX and emit Lean")?;
    let package = xlsynth::convert_dslx_to_ir(source, path, &Default::default())
        .context("convert DSLX to XLS IR")?;
    let machine = package
        .ir
        .get_function("__machine__machine")
        .context("look up __machine__machine in XLS IR")?;
    Ok((bundle, machine))
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn counted_loop_matches_xls() {
    let source = r#"
fn fold(seed: u8, mask: u8, offset: u8) -> u8 {
    for (i, carry): (u8, u8) in u8:1..u8:5 {
        (carry & mask) + offset + i
    }(seed)
}
fn machine(mask: u8, offset: u8, current_count: u8) -> (u8, u8) {
    (current_count, fold(current_count, mask, offset))
}
"#;
    let bx = r#"
extern fn machine::fold(seed: u8, mask: u8, offset: u8) -> u8;
block counter(clk, rst, mask: u8, offset: u8) -> u8 {
    reg count: u8 = 0;
    count := machine::fold(count, mask, offset);
    count
}
"#;
    let (mut bundle, machine) =
        compile_lean_and_xls_machine(bx, source).expect("counted loop fixture (machine.x)");
    let mut checks = String::from("import ReadableCorrespondence\nopen XlsReadable\n");
    for (seed, mask, offset) in [(1, 3, 2), (255, 1, 0), (128, 255, 5)] {
        let args = [mask, offset, seed].map(|n| xlsynth::XlsIrValue::make_ubits(8, n).unwrap());
        let expected = machine
            .interpret(&args)
            .unwrap()
            .get_element(1)
            .unwrap()
            .to_u64()
            .unwrap();
        writeln!(checks, "#guard (step {{ count := {seed} }} {{ rst := false, mask := {mask}, offset := {offset} }}).count == {expected}").unwrap();
    }
    bundle.files.push(("CheckLoop.lean".into(), checks));
    check_bundle(
        bundle,
        "counted loop over a register and two captured inputs",
    );
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn counted_loop_proves_computed_enum_dispatch_in_linked_helpers() {
    let choose = r#"
        let chosen = if fresh { incoming } else { current };
        let routed = match chosen { Sender::A => a, Sender::B => b, _ => c };
        (routed, chosen)"#;
    let helper = format!(
        "fn choose(fresh: bool, incoming: Sender, current: Sender,
        a: bool, b: bool, c: bool) -> (bool, Sender) {{ {choose} }}"
    );
    for (name, definition, result) in [
        ("in the clocked helper", "", choose),
        (
            "helper",
            helper.as_str(),
            "choose(fresh, incoming, current, a, b, c)",
        ),
    ] {
        let source = format!(
            r#"
enum Sender : u2 {{ A = 0, B = 1, C = 2 }}
{definition}
fn route(fresh: bool, incoming: Sender, current: Sender, a: bool, b: bool, c: bool)
         -> (bool, Sender) {{
    let c = for (i, acc) in u32:0..u32:2 {{ if i == u32:0 {{ acc }} else {{ !acc }} }}(c);
    {result}
}}
"#
        );
        let bx = r#"
extern fn machine::route(fresh: bool, incoming: u2, current: u2, a: bool, b: bool, c: bool) -> (bool, u2);
block router(clk, rst, fresh: bool, incoming: u2, a: bool, b: bool, c: bool) -> bool {
    reg sender: u2 = 0;
    let result = machine::route(fresh, incoming, sender, a, b, c);
    sender := result.1;
    result.0
}
"#;
        check_bundle(checked_helper_bundle(bx, &source, &[]).unwrap(), name);
    }
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn imported_parametric_helpers_preserve_instances() {
    let modules = tempfile::tempdir().unwrap();
    let packets = r#"
#![feature(generics)]
pub struct Packet<N: u32> { value: bits[N], enabled: bool }
pub struct Fallback<N: u32> { value: bits[N], enabled: bool }
pub fn choose<N: u32>(packet: Packet<N>, fallback: bits[N]) -> Packet<N> {
    Packet<N> { value: if packet.enabled { packet.value } else { fallback }, ..packet }
}
pub fn bias<K: u32>(x: u6) -> u6 { x + (K as u6) }
pub fn passthrough<T: type>(x: T) -> T { x }
"#;
    std::fs::write(modules.path().join("packets.x"), packets).unwrap();
    let source = r#"
import packets;
struct Packet { current: u6 }
fn choose_values(small: packets::Packet<u32:5>, large: packets::Packet<u32:6>,
           fallback: packets::Fallback<u32:5>, last: u6) -> (u6, u6) {
    let pair = packets::passthrough<(u8, bool)>((u3:0 ++ fallback.value, small.enabled));
    let a = packets::choose<u32:5>(small, pair.0 as u5);
    let b = packets::choose<u32:6>(large, u6:41);
    let output = packets::passthrough<Packet>(Packet { current: last });
    (output.current, packets::bias<u32:1>(u1:0 ++ a.value) + packets::bias<u32:2>(b.value))
}
fn machine(small: u5, large: u6, fallback: u5, last: u6) -> (u6, u6) {
    choose_values(packets::Packet<5> { value: small, enabled: false },
                  packets::Packet<6> { value: large, enabled: true },
                  packets::Fallback<5> { value: fallback, enabled: true }, last)
}
"#;
    let bx = r#"
extern fn machine::machine(small: u5, large: u6, fallback: u5, last: u6) -> (u6, u6);
block select_values(clk, rst, small: u5, large: u6, fallback: u5) -> u6 {
    reg last: u6 = 0;
    let result = machine::machine(small, large, fallback, last);
    last := result.1;
    result.0
}
"#;
    let mut bundle = checked_helper_bundle(bx, source, &[modules.path()]).unwrap();
    bundle.files.push((
        "Check.lean".into(),
        r#"import ReadableCorrespondence
open XlsReadable
private def stimulus : Input := {rst := false, small := 3, large := 37, fallback := 17}
private def small : XLS.packets.«Packet<5>» := {value := stimulus.small, enabled := false}
private def large : XLS.packets.«Packet<6>» := {value := stimulus.large, enabled := true}
private def fallback : XLS.packets.«Fallback<5>» := {value := 17, enabled := true}
#guard (XLS.packets.«choose<5>» (packet := small) (fallback := 17)).value == 17
#check_failure XLS.packets.«choose<5>» (packet := fallback) (fallback := 17)
#guard (XLS.packets.«choose<6>» (packet := large) (fallback := 41)).value == 37
#guard (XLS.packets.«bias<1>» (x := 0), XLS.packets.«bias<2>» (x := 0)) == (1, 2)
#guard XLS.packets.«passthrough<(u8,bool)>» (x := (13, true)) == ((13 : BitVec 8), true)
#guard (XLS.packets.«passthrough<Packet>» (x := { current := 7 })).current == 7
#guard (step { last := 7 } stimulus).last == 57
"#
        .into(),
    ));
    check_bundle(bundle, "parametric");
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn helpers_distinguish_equal_layout_struct_instances() {
    let source = r#"
const TWO = u32:2;
struct Tagged<N: u32> { value: u8 }
fn take_two(x: Tagged<TWO>) -> u8 { x.value }
fn combine(one: Tagged<1>, two: Tagged<2>) -> u8 { one.value + take_two(two) }
fn machine(one: u8, two: u8) -> u8 {
    combine(Tagged<1> { value: one }, Tagged<2> { value: two })
}
"#;
    let bx = r#"
extern fn machine::machine(one: u8, two: u8) -> u8;
block distinguish(clk, rst, one: u8, two: u8) -> u8 {
    reg last: u8 = 0;
    last := machine::machine(one, two);
    last
}
"#;
    let mut bundle = checked_helper_bundle(bx, source, &[]).unwrap();
    bundle.files.push((
        "Check.lean".into(),
        r#"import ReadableCorrespondence
open XlsReadable
variable (one : XLS.machine.«Tagged<1>») (two : XLS.machine.«Tagged<2>»)
example : XLS.machine.take_two two = two.value := rfl
#check_failure XLS.machine.take_two one
"#
        .into(),
    ));
    check_bundle(bundle, "equal-layout struct instances");
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn compose_helpers_across_dependent_struct_defaults() {
    let source = r#"
struct Packet<N: u32, M: u32 = {N * u32:2}, K: u32 = {M + u32:1}> { data: bits[M] }
fn pack(x: u6) -> Packet<3> { Packet<3> { data: x } }
fn relay(p: Packet<3, 6>) -> Packet<3, 6> { p }
fn unpack(p: Packet<3, 6, 7>) -> u6 { p.data }
fn other(p: Packet<3, 6, 8>) -> u6 { p.data }
fn advance(x: u6) -> u6 {
    unpack(relay(pack(x))) + other(Packet<3, 6, 8> { data: x })
}
"#;
    let bx = r#"
extern fn machine::advance(x: u6) -> u6;
block pass(clk, x: u6) -> u6 { machine::advance(x) }
"#;
    let mut bundle = checked_helper_bundle(bx, source, &[]).unwrap();
    bundle.files.push((
        "Compose.lean".into(),
        r#"import ReadableGenerated
open XlsReadable
variable (x : BitVec 6)
example : XLS.machine.«Packet<3,6,7>» := XLS.machine.pack x
example : XLS.machine.unpack (XLS.machine.relay (XLS.machine.pack x)) = x := rfl
example (p : XLS.machine.«Packet<3,6,8>») : XLS.machine.other p = p.data := rfl
#check_failure XLS.machine.other (XLS.machine.pack x)
"#
        .into(),
    ));
    check_bundle(
        bundle,
        "omitted, partial and fully explicit struct bindings",
    );
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn neg_and_left_shift_match_xls() {
    let source = r#"
fn shift_value(min: u64, value: u5) -> u5 { (-value) << min }
fn machine(amount: u64, current_value: u5) -> (u5, u5) {
    (current_value, shift_value(amount, current_value))
}
"#;
    let bx = r#"
extern fn machine::shift_value(min: u64, value: u5) -> u5;
block shift(clk, rst, amount: u64) -> u5 {
    reg value: u5 = 0;
    value := machine::shift_value(amount, value);
    value
}
"#;
    let (mut bundle, machine) = compile_lean_and_xls_machine(bx, source)
        .expect("negation and left shift fixture (machine.x)");
    let mut checks = String::from("import ReadableCorrespondence\nopen XlsReadable\n");
    for amount in [0, 1, 5, 1 << 60] {
        let args = [
            xlsynth::XlsIrValue::make_ubits(64, amount).unwrap(),
            xlsynth::XlsIrValue::make_ubits(5, 3).unwrap(),
        ];
        let expected = machine
            .interpret(&args)
            .unwrap()
            .get_element(1)
            .unwrap()
            .to_u64()
            .unwrap();
        writeln!(
            checks,
            "#guard (step {{ value := 3 }} {{ rst := false, amount := {amount} }}).value == {expected}"
        )
        .unwrap();
    }
    bundle.files.push(("CheckShift.lean".into(), checks));
    check_bundle(bundle, "negation and variable left shift");
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn source_binding_named_vector_preserves_physical_array_types() {
    let bx = r#"
block pick(clk, Vector: u8, b: u8, index: u8) -> u8 {
    let slots = [(Vector, u8:0), (b, u8:0)];
    slots[index].0
}
"#;
    let bundle = bx_bundle(bx);
    let (_, model) = bundle
        .files
        .iter()
        .find(|(name, _)| name == "ReadableGenerated.lean")
        .unwrap();
    assert!(model.contains("let «Vector» :"), "{model}");
    check_bundle(bundle, "local Vector and a physical array of tuples");
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn arrays_keep_record_elements_and_xls_index_bounds() {
    let source = r#"
struct Write { data: u8 }
struct Slot { write: Write }
fn replace(slots: Slot[2], index: u8, replacement: Slot) -> Slot[2] {
    update(slots, index, replacement)
}
"#;
    let bx = r#"
struct Write { data: u8 }
struct Slot { write: Write }
extern fn machine::replace(slots: Slot[2], index: u8, replacement: Slot) -> Slot[2];
block storage(clk, rst, index: u8, replacement: Slot) -> u8 {
    reg slots: Slot[2];
    slots := machine::replace(slots, index, replacement);
    slots[index].write.data
}
"#;
    let mut bundle = checked_helper_bundle(bx, source, &[]).unwrap();
    let (_, readable) = bundle
        .files
        .iter()
        .find(|(file, _)| file == "ReadableGenerated.lean")
        .unwrap();
    for projection in [").«write»", ").«data»"] {
        assert!(readable.contains(projection), "{readable}");
    }
    bundle.files.push((
        "CheckArray.lean".into(),
        r#"import ReadableCorrespondence
open XlsReadable
private def clock (index : BitVec 8) :=
  let q : State := { slots := #v[⟨⟨7⟩⟩, ⟨⟨13⟩⟩] }
  let x : Input := { rst := false, index, replacement := ⟨⟨19⟩⟩ }
  let slots := (step q x).slots
  (observe q x, slots[0].write.data, slots[1].write.data)
#guard clock 0 == (7, 19, 13)
#guard clock 255 == (13, 7, 13)
"#
        .into(),
    ));
    check_bundle(bundle, "array of nested records");
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn constructed_and_selected_record_arrays_prove_correspondence() {
    let bx = r#"
struct Write { data: u8 }
struct Slot { write: Write }
block pick(clk, rst, choose: bool, a: Slot, b: Slot, supplied: Slot[2], index: u8) -> u8 {
    reg last: Slot;
    let slots = if choose { [a, b] } else { supplied };
    let slot = slots[index];
    last := slot;
    slot.write.data
}
"#;
    check_bundle(bx_bundle(bx), "indexed field of a selected record array");
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn dynamic_slices_match_xls_at_and_beyond_the_boundary() {
    let source = r#"
fn replace(data: u8, index: u64) -> u8 { bit_slice_update(data, index, u3:2) }
fn machine(index: u64, data: u8) -> (u8, u8) {
    (data[index+:u3] as u8, replace(data, index))
}
"#;
    let bx = r#"
extern fn machine::replace(data: u8, index: u64) -> u8;
block slices(clk, rst, index: u64) -> u8 {
    reg data: u8 = 0;
    data := machine::replace(data, index);
    data[index+:u3] as u8
}
"#;
    let (mut bundle, machine) =
        compile_lean_and_xls_machine(bx, source).expect("dynamic bit slices fixture (machine.x)");
    let mut checks = String::from(
        r#"import ReadableCorrespondence
open XlsReadable
private def clock (index : BitVec 64) :=
  let q : State := { data := 165 }
  let x : Input := { rst := false, index }
  (observe q x, (step q x).data)
"#,
    );
    for index in [0, 2, 7, 1 << 60] {
        let result = machine
            .interpret(&[
                xlsynth::XlsIrValue::make_ubits(64, index).unwrap(),
                xlsynth::XlsIrValue::make_ubits(8, 165).unwrap(),
            ])
            .unwrap();
        let read = result.get_element(0).unwrap().to_u64().unwrap();
        let updated = result.get_element(1).unwrap().to_u64().unwrap();
        writeln!(checks, "#guard clock {index} == ({read}, {updated})").unwrap();
    }
    bundle.files.push(("CheckSlices.lean".into(), checks));
    check_bundle(bundle, "dynamic bit slice and update");
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn encode_changed_bits_matches_xls_for_multihot_masks() {
    let source = "fn encode_mask(List: u8) -> u3 { encode(List) }
    fn machine(mask: u8, current_previous: u8) -> (u3, u8) {
        (encode_mask(mask ^ current_previous), mask)
    }";
    let bx = r#"
extern fn machine::encode_mask(List: u8) -> u3;
block changed(clk, rst, mask: u8) -> u3 {
    reg previous: u8 = 0;
    previous := mask;
    machine::encode_mask(mask ^ previous)
}
"#;
    let (mut bundle, machine) =
        compile_lean_and_xls_machine(bx, source).expect("changed-bit mask fixture (machine.x)");
    assert!(
        bundle.files[2]
            .1
            .contains("def _root_.XLS.«machine».«encode_mask» («List»"),
        "{}",
        bundle.files[2].1
    );
    let mut checks = String::from("import ReadableCorrespondence\nopen XlsReadable\n");
    let previous = 0b0000_0100;
    // No change, clear bit 2, set bit 4, then flip both bits relative to
    // previous.
    for mask in [0b0000_0100, 0, 0b0001_0100, 0b0001_0000] {
        let args = [mask, previous].map(|v| xlsynth::XlsIrValue::make_ubits(8, v).unwrap());
        let index = machine
            .interpret(&args)
            .unwrap()
            .get_element(0)
            .unwrap()
            .to_u64()
            .unwrap();
        writeln!(
            checks,
            "#guard observe {{ previous := {previous} }} {{ rst := false, mask := {mask} }} == {index}"
        )
        .unwrap();
    }
    bundle.files.push(("CheckEncode.lean".into(), checks));
    check_bundle(bundle, "encode changed mask bits");
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn unsigned_arithmetic_matches_xls_at_width_boundaries() {
    let source = "fn machine(a: u5, b: u5, current_value: u5) -> (bool, u5, u5, u5) {
        (a < b, a - b, a / b, current_value * a)
    }";
    let bx = r#"
block arithmetic(clk, rst, a: u5, b: u5) -> (bool, u5, u5) {
    reg value: u5 = 0;
    value := value * a;
    (a < b, a - b, a / b)
}
"#;
    let (mut bundle, machine) =
        compile_lean_and_xls_machine(bx, source).expect("unsigned arithmetic fixture (machine.x)");
    let mut checks = String::from(
        r#"import ReadableCorrespondence
open XlsReadable
private def clock (a b : BitVec 5) :=
  let q : State := { value := 7 }
  let x : Input := { rst := false, a, b }
  (observe q x, (step q x).value)
"#,
    );
    for (a, b) in [(3, 8), (31, 0), (19, 3)] {
        let args = [a, b, 7].map(|v| xlsynth::XlsIrValue::make_ubits(5, v).unwrap());
        let result = machine.interpret(&args).unwrap();
        let [lt, diff, quot, prod] =
            [0, 1, 2, 3].map(|i| result.get_element(i).unwrap().to_u64().unwrap());
        writeln!(
            checks,
            "#guard clock {a} {b} == (({}, {diff}, {quot}), {prod})",
            lt != 0
        )
        .unwrap();
    }
    bundle.files.push(("CheckArith.lean".into(), checks));
    check_bundle(bundle, "unsigned fixed-width arithmetic");
}

fn checked_nonzero_range() -> xlsynth::dslx::TypecheckedModule {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bx_loop_bindings");
    let mut imports = xlsynth::dslx::ImportData::new(None, &[&root]);
    xlsynth::dslx::parse_and_typecheck(
        include_str!("fixtures/bx_loop_bindings/machine.x"),
        "machine.x",
        "machine",
        &mut imports,
    )
    .unwrap()
}

#[cfg(feature = "dslx-loop-bindings")]
#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn bx_import_loop_names_follow_rebased_nodes_and_function_symbols() -> Result<()> {
    let root = tempfile::tempdir()?;
    fs::write(
        root.path().join("sum.x"),
        r#"
pub fn lanes(xs: u8[3]) -> u8 {
    for (lane, total) in u32:1..u32:3 { total + xs[lane] }(u8:0)
}
"#,
    )?;
    let source = "use sum; block rows(clk, xs: u8[3]) -> u8 { sum::lanes(xs) }";
    let imports = xlsynth_bx::ImportOptions {
        additional_search_paths: vec![root.path()],
        ..Default::default()
    };
    let plain = compile_bx(
        "rows",
        source,
        xlsynth_bx::empty_package("plain"),
        Default::default(),
        &imports,
    )?;
    fn loop_body(compiled: &xlsynth_lean::block::CompiledBx) -> &str {
        compiled
            .package()
            .members
            .iter()
            .flat_map(|m| &m.graph().nodes)
            .find_map(|n| match &n.payload {
                xlsynth_pir::ir::NodePayload::CountedFor { body, .. } => Some(body.as_str()),
                _ => None,
            })
            .unwrap()
    }
    let body = loop_body(&plain);
    let reserved = format!(
        "package occupied\nfn {body}(a: bits[8] id=500) -> bits[8] {{ ret n: bits[8] = not(a, id=501) }}"
    );
    let compiled = xlsynth_lean::block::compile_bx_with_loop_bindings(
        "rows",
        source,
        Parser::new(&reserved).parse_and_validate_package()?,
        Default::default(),
        &imports,
    )?;
    let linked = loop_body(&compiled);
    assert_ne!(linked, body);
    assert!(
        compiled
            .package()
            .get_fn(linked)
            .unwrap()
            .param_nodes()
            .all(|p| p.text_id > 501)
    );
    let mut bundle = compiled.emit_readable(None)?;
    let model = &bundle
        .files
        .iter()
        .find(|(name, _)| name == "ReadableGenerated.lean")
        .unwrap()
        .1;
    let (_, body) = model
        .split_once(&format!("def Helpers.«{linked}»"))
        .unwrap();
    let body = body.split("\n\n").next().unwrap();
    assert!(
        body.lines()
            .any(|l| l.trim_start().starts_with("let «lane» :")
                && l.contains("BitVec.add «xls_iteration»")),
        "{body}"
    );
    bundle.files.push((
        "CheckImport.lean".into(),
        r#"
import ReadableCorrespondence
open XlsReadable
#guard observe {} { xs := #v[100, 3, 5] } == (8 : BitVec 8)
"#
        .into(),
    ));
    check_bundle(
        bundle,
        "imported nonzero loop with colliding IDs and symbol",
    );
    Ok(())
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn nonzero_range_without_loop_records() {
    let mut bundle = xlsynth_lean::block::compile_bx_with_dslx(
        "counter",
        include_str!("fixtures/bx_loop_bindings/counter.bx"),
        checked_nonzero_range(),
        Default::default(),
        &Default::default(),
    )
    .unwrap()
    .emit_readable(None)
    .unwrap();
    bundle.files.push((
        "CheckLoop.lean".into(),
        include_str!("fixtures/bx_loop_bindings/CheckLoop.lean").into(),
    ));
    check_bundle(bundle, "nonzero range without loop records");
}

#[cfg(feature = "dslx-loop-bindings")]
#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn loop_keeps_the_source_index_and_captured_record_fields() {
    let mut bundle = xlsynth_lean::block::compile_bx_with_dslx_loop_bindings(
        "counter",
        include_str!("fixtures/bx_loop_bindings/counter.bx"),
        checked_nonzero_range(),
        Default::default(),
        &Default::default(),
    )
    .unwrap()
    .emit_readable(None)
    .unwrap();
    let (_, model) = bundle
        .files
        .iter()
        .find(|(name, _)| name == "ReadableGenerated.lean")
        .unwrap();
    // The helper still computes the same result if the wrong local is named
    // bank.
    let (_, body) = model
        .split_once("def Helpers.«____machine__fold_counted_for_0_body»")
        .unwrap();
    let body = body.split("\n\n").next().unwrap();
    let offset = body
        .lines()
        .find_map(|line| {
            let (binding, _) = line.trim().strip_prefix("let ")?.split_once(" : ")?;
            line.ends_with(" := (BitVec.ofNat 32 0x1)")
                .then_some(binding)
        })
        .expect("the loop offset should be the literal one");
    let offset_iteration = format!(" := BitVec.add «xls_iteration» {offset}");
    assert!(
        body.lines()
            .any(|line| line.trim().starts_with("let «bank» : ")
                && line.ends_with(&offset_iteration)),
        "bank should name the offset iteration"
    );
    assert!(body.lines().any(|line| {
        line.split_once("Vector.getD ")
            .is_some_and(|(_, args)| args.contains(" («bank»).toNat "))
    }));
    assert!(body.contains(".«valid»"));
    let checks = r#"import ReadableCorrespondence
open XlsReadable
private def pool : packets.«Pool<5>» := { status := (3, 4), slots := #v[
  {write := {valid := true, data := 11}},
  {write := {valid := false, data := 19}},
  {write := {valid := true, data := 23}}] }
#guard Helpers.«____machine__fold_counted_for_0_body»
  (xls_iteration := 1) (acc := 7) (pool := pool) (mode := packets.Mode.Run) == 30
"#;
    bundle
        .files
        .push(("CheckLoopNames.lean".into(), checks.into()));
    bundle.files.push((
        "CheckLoop.lean".into(),
        include_str!("fixtures/bx_loop_bindings/CheckLoop.lean").into(),
    ));
    check_bundle(bundle, "nonzero range with loop records");
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
#[cfg(feature = "dslx-loop-bindings")]
fn loop_captures_keep_their_declared_struct_arguments() {
    let source = r#"
struct Tagged<N: u32> { value: u8 }
fn fold(first: Tagged<1>, other: Tagged<2>, q: u8) -> u8 {
    let second = first;
    for (_, acc) in u32:0..u32:2 {
        acc + second.value + other.value
    }(q)
}
fn machine(first: u8, other: u8, q: u8) -> u8 {
    fold(Tagged<1> {value: first}, Tagged<2> {value: other}, q)
}
"#;
    let bx = r#"
extern fn machine::machine(first: u8, other: u8, q: u8) -> u8;
block sum(clk, rst, first: u8, other: u8) -> u8 {
    reg q: u8 = 0;
    q := machine::machine(first, other, q);
    q
}
"#;
    let mut imports = xlsynth::dslx::ImportData::new(None, &[]);
    let module =
        xlsynth::dslx::parse_and_typecheck(source, "machine.x", "machine", &mut imports).unwrap();
    let mut bundle = xlsynth_lean::block::compile_bx_with_dslx_loop_bindings(
        "sum",
        bx,
        module,
        Default::default(),
        &Default::default(),
    )
    .unwrap()
    .emit_readable(None)
    .unwrap();
    bundle.files.push((
        "Captures.lean".into(),
        r#"import ReadableCorrespondence
open XlsReadable
variable (first : machine.«Tagged<1>») (other : machine.«Tagged<2>»)
#check Helpers.«____machine__fold_counted_for_0_body» 0 0 (second := first) (other := other)
#check_failure Helpers.«____machine__fold_counted_for_0_body» 0 0 (second := other) (other := first)
"#
        .into(),
    ));
    check_bundle(bundle, "distinct equal-layout captures");
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn counted_loop_proves_indexed_record_fields_in_block_and_body() {
    let source = r#"
struct Write { data: u8 }
struct Slot { write: Write }
fn fold(index: u8, a: Slot, b: Slot, last: u8) -> u8 {
    for (i, acc) in u32:0..u32:3 {
        let inner_slots = [b, a];
        let inner_slot = inner_slots[index + (i as u8)];
        acc + inner_slot.write.data
    }(last)
}
"#;
    let bx = r#"
struct Write { data: u8 }
struct Slot { write: Write }
extern fn machine::fold(index: u8, a: Slot, b: Slot, last: u8) -> u8;
block indexed(clk, rst, index: u8, a: Slot, b: Slot) -> u8 {
    reg last: u8 = 0;
    let slots = [a, b];
    last := machine::fold(index, a, b, last);
    slots[index].write.data
}
"#;
    let bundle = checked_helper_bundle(bx, source, &[]).unwrap();
    let readable = &bundle
        .files
        .iter()
        .find(|(name, _)| name == "ReadableGenerated.lean")
        .unwrap()
        .1;
    for projection in [").«write»", ").«data»"] {
        assert!(readable.matches(projection).count() >= 2, "{readable}");
    }
    check_bundle(bundle, "indexed records in a block and its loop body");
}

/// Elaborate every generated file in dependency order with the requested Lean.
fn check_bundle(bundle: LeanBlockBundle, fixture: &str) {
    let lean = std::env::var_os("XLSYNTH_LEAN")
        .expect("set XLSYNTH_LEAN to the actual Lean executable path");
    let directory = tempfile::tempdir().unwrap();
    for (name, source) in bundle.files {
        let path = directory.path().join(name);
        fs::write(&path, source).unwrap();
        let output = Command::new(&lean)
            .arg("-o")
            .arg(path.with_extension("olean"))
            .arg(&path)
            .current_dir(directory.path())
            .env("LEAN_PATH", directory.path())
            .env_remove("LEAN_SYSROOT")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "Lean failed for {fixture}, {}:\n{}\n{}",
            path.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}
