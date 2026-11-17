// SPDX-License-Identifier: Apache-2.0

//! Explicit native checks for generated readable block correspondence proofs.

#[cfg(feature = "lean-tests")]
use std::fmt::Write as _;
use std::{fs, path::Path};

use anyhow::Result;
use xlsynth_lean::block::{LeanBlockBundle, compile_bx, emit, select_block};
use xlsynth_pir::ir_parser::Parser;

#[path = "common/lean_bundle.rs"]
mod lean_bundle;

/// Compile a self-contained BX fixture under the source module `design`.
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

// Verifies: Each port retains its record or alias name in either order.
// Catches: reusing the first type spelling for every port with the same layout.
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

// Verifies: BX state and ports retain both 4-bit and 8-bit Packet types.
// Catches: assigning a nested field or register the wrong specialization.
#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn imported_parametric_structs_preserve_bx_interface_types() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fs::write(
        dir.path().join("ops.x"),
        "pub struct Packet<N: u32> { zebra: uN[N], apple: uN[N] }",
    )?;
    let compiled = compile_bx(
        "design",
        r#"
use ops;
struct Envelope { narrow: ops::Packet<4> }
block direct(clk, p: ops::Packet<8>) -> ops::Packet<8> { p }
block pair(clk, p: ops::Packet<8>, src: Envelope) -> (ops::Packet<8>, ops::Packet<4>) {
    reg stored: ops::Packet<8>;
    stored := p;
    (stored, src.narrow)
}
"#,
        xlsynth_bx::empty_package("example"),
        Default::default(),
        &xlsynth_bx::ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        },
    )?;
    for (block, proof) in [
        (
            "direct",
            "example (q : State) (x : Input) : observe q x = (x.p.zebra, x.p.apple) := rfl",
        ),
        (
            "pair",
            r#"
example (q : State) (x : Input) : (step q x).stored = x.p := rfl
example (q : State) (x : Input) : (XLS.ops.«Packet<8>» × XLS.ops.«Packet<4>») := observe q x
example (q : State) (x : Input) : observe q x = (q.stored, x.src.narrow) := rfl
"#,
        ),
    ] {
        let mut bundle = compiled.emit_readable(Some(block))?;
        bundle.files.push((
            "Check.lean".into(),
            format!("import ReadableCorrespondence\nopen XlsReadable\n{proof}\n"),
        ));
        check_bundle(bundle, block);
    }
    Ok(())
}

// Verifies: Forwarded Boolean fields reduce by reflexivity. The step and
// empty assertions proofs both project the public evaluation theorem.
// Catches: A codec blocking reduction or simplification erasing that
// dependency.
#[cfg(feature = "lean-tests")]
#[test]
fn bx_segmented_boolean_passthrough_reduces_symbolically() {
    let mut bx = String::from(
        "struct Packet { status: (bool, u8) }
         block passthrough(clk, flag: bool, p: (bool, u8), packet: Packet, x: u8)
             -> (bool, bool, bool, u8) {
             reg saved: Packet;
             saved := packet;
             let v0 = x;\n",
    );
    for i in 1..=161 {
        writeln!(bx, "let v{i} = v{} + x;", i - 1).unwrap();
    }
    bx.push_str("(flag, p.0, saved.status.0, v161)\n}");
    let mut bundle = bx_bundle(&bx);
    bundle.files.push((
        "PublicInterface.lean".into(),
        r#"import ReadableCorrespondence
import Lean
open XlsReadable
example (q : State) (x : Input) : (observe q x).1 = x.flag := by rfl
example (q : State) (x : Input) : (observe q x).2.1 = x.p.1 := by rfl
example (q : State) (x : Input) : (observe q x).2.2.1 = q.saved.status.1 := by rfl

open Lean in
run_cmd do
  for projection in [``encoded_step, ``encoded_assertions] do
    let .thmInfo info ← getConstInfo projection | throwError "expected a theorem"
    unless info.value.getUsedConstantsAsSet.contains ``encoded_eval do
      throwError "{projection} lost its dependency on encoded_eval"
"#
        .into(),
    ));
    check_bundle(
        bundle,
        "direct Boolean input and forwarded input and register fields",
    );
}

// Verifies: An inlined DSLX projection reads the intended BX-built field.
// Catches: Swapping equal-width fields in construction or projection.
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

// Verifies: Both reset polarities reset the next count and output old count.
// Catches: Inverting reset or exposing the reset value one clock too early.
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
            "design",
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

// Verifies: Outputs follow port order and both registers swap old values.
// Catches: Reordered outputs or sequential evaluation of register writes.
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
example (q : State) (x : Input) :
    step (step q { x with rst := false }) { x with rst := false } = q := by cases q; rfl
example (q : State) (x : Input) : (reset q x).next = ⟨0, 3⟩ := by rfl
"#
        .into(),
    ));
    check_bundle(bundle, "swap registers");
}

// Verifies: Empty output is unit and a singleton register stays a tuple.
// Catches: Flattening the singleton or emitting an invalid empty interface.
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
example (q : State) (x : Input) : (step q { x with rst := false }).one.1 = x.x.1 :=
  XlsCodec.decode_encodeBool _
"#
        .into(),
    ));
    check_bundle(bundle, "singleton register and unit output");
    check_bundle(bx_bundle("block empty(clk) -> () { () }"), "empty block");
}

// Verifies: Nested prelinked helpers compute the next register from old state.
// Catches: Dropping a transitive helper or feeding next state into the call.
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
block pipeline(clk, left: u8, right: u8) -> u8 {
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

/// Checks the complete bundle and names the fixture in any compilation failure.
fn check_bundle(bundle: LeanBlockBundle, fixture: &str) {
    lean_bundle::check_bundle(&bundle).unwrap_or_else(|error| panic!("{fixture}: {error:#}"));
}
