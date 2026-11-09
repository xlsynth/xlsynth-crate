// SPDX-License-Identifier: Apache-2.0

//! Explicit native checks for generated readable block correspondence proofs.

#[path = "common/block_project.rs"]
mod block_project;

use std::{fs, path::Path};

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
    (observe q x).2 = x.left := ⟨rfl, rfl, rfl⟩
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
fn bx_counter_reset_polarities() {
    let source = r#"
block counter(clk) -> u8 {
    reg count: u8 = 3;
    count := count + u8:1;
    count
}
"#;
    for active_low in [true, false] {
        let reset = xlsynth_bx::ResetBehavior {
            asynchronous: false,
            active_low,
        };
        let mut bundle = compile_bx(
            source,
            xlsynth_bx::empty_package("example"),
            reset,
            &Default::default(),
        )
        .unwrap()
        .emit_readable(None)
        .unwrap();
        let asserted = !active_low;
        let released = active_low;
        let client = format!(
            r#"
import ReadableCorrespondence
open XlsReadable
#guard (step {{count := 7}} {{rst := {asserted}}}).count == 3
#guard (step {{count := 7}} {{rst := {released}}}).count == 8
#guard observe {{count := 7}} {{rst := {asserted}}} == 7
#guard observe {{count := 7}} {{rst := {released}}} == 7
"#
        );
        bundle.files.push(("PublicInterface.lean".into(), client));
        check_bundle(
            bundle,
            &format!("counter with reset active_low={active_low}"),
        );
    }
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn bx_positional_outputs_and_simultaneous_registers() {
    let source = r#"
struct InputPacket { sample: (bool, u8) }
block swap(clk, x: InputPacket) -> (bool, u8) {
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
block machine(clk, x: (bool,)) -> () {
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
block machine(clk, BitVec: u8, packet: Packet) -> (u8, Packet) {
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

/// Elaborate every generated file in dependency order with the requested Lean.
fn check_bundle(bundle: LeanBlockBundle, _fixture: &str) {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let sources = bundle.files.into_iter().map(|(name, source)| {
        let path = root.join(name);
        fs::write(&path, source).unwrap();
        path
    });
    block_project::compile_project(root, sources);
}
