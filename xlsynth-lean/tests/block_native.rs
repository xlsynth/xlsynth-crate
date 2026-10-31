// SPDX-License-Identifier: Apache-2.0

//! Explicit native checks for generated readable block correspondence proofs.

use std::{fmt::Write as _, fs, path::Path, process::Command};

use anyhow::{Context, Result};
use xlsynth_lean::block::{LeanBlockBundle, compile_bx, emit, select_block};
use xlsynth_pir::ir_parser::Parser;

fn bx_bundle(source: &str) -> LeanBlockBundle {
    compile_bx(
        source,
        xlsynth_bx::empty_package("example"),
        Default::default(),
        &Default::default(),
    )
    .unwrap()
    .emit_readable(None)
    .unwrap()
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
struct Packet { XlsSource: u8, _root_: u8, word: Word }
block machine(clk, rst, BitVec: u8, packet: Packet) -> (u8, Packet) {
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
#check XlsSource.Word.Constructors.mk
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
    let mut bundle =
        xlsynth_lean::block::compile_bx(bx, package, Default::default(), &Default::default())
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
    xlsynth_lean::block::compile_bx_with_dslx(bx, module, Default::default(), &Default::default())?
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
    for helper in ["left_side::advance", "machine::advance"] {
        assert!(model.contains(&format!("_root_.XlsReadable.Helpers.«{helper}»")));
    }
    bundle.files.push((
        "CheckCalls.lean".into(),
        r#"
import ReadableCorrespondence
open XlsReadable
private def stimulus : Input := { rst := false, packet := { left := 13, right := 5 } }
private def initial : State := { count := 7 }
#guard (Helpers.«left_side::advance» (command := { base := 13, delta := 5 })).base == 18
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
