// SPDX-License-Identifier: Apache-2.0

//! Explicit native checks for generated readable block correspondence proofs.

use std::{fs, path::Path, process::Command};

use xlsynth_lean::block::{LeanBlockBundle, compile_stateful_function, emit, select_block};
use xlsynth_pir::ir_parser::Parser;

// The same theorem uses the named output fields after a private binding
// rename. Inputs and outputs deliberately have identical physical layouts.
#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn stateful_struct_fields_and_private_binding_rename() {
    const SOURCE: &str = r#"
struct Request { offered: bool, enabled: bool }
struct Outputs { seen: bool, valid: bool }
type Reply = Outputs;
fn machine(x: Request, __reg_count: u8) -> (Reply, u8) {
    let local_seen = x.offered;
    (Outputs { seen: local_seen, valid: x.enabled }, __reg_count + u8:1)
}
"#;
    const PROOF: &str = r#"
import ReadableCorrespondence
open XlsReadable
example (q : State) (x : Input) :
    let out0 : XlsSource.Reply := observe q x
    out0.seen = x.x.offered ∧ out0.valid = x.x.enabled := by
  exact ⟨rfl, rfl⟩
example (q : State) (x : Input) : (reset q x).next.count = 0 := by rfl
"#;
    for name in ["local_seen", "renamed_seen"] {
        let source = SOURCE.replace("local_seen", name);
        let compiled =
            compile_stateful_function(&source, Path::new("machine.x"), "machine", None, &[])
                .unwrap();
        let mut bundle = compiled.emit_readable().unwrap();
        bundle
            .files
            .push(("PublicInterface.lean".into(), PROOF.into()));
        check_bundle(bundle, name);
    }
}

// Check a source tuple alias nested in the input and one positional pair of
// outputs. The two registers swap simultaneously unless reset is asserted.
#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn stateful_positional_outputs_and_simultaneous_registers() {
    let source = r#"
type Sample = (bool, u8);
struct InputPacket { sample: Sample }
fn machine(x: InputPacket, __reg_left: u8, __reg_right: u8) -> (bool, u8, u8, u8) {
    let (flag, _data) = x.sample;
    (flag, __reg_left, __reg_right, __reg_left)
}
"#;
    let proof = r#"
import ReadableCorrespondence
open XlsReadable
example (q : State) (x : Input) :
    observe q x = (x.x.sample.1, q.left) := by rfl
example (q : State) (x : Input) :
    step q { x with rst := false } = ⟨q.right, q.left⟩ := by rfl
example (q : State) (x : Input) : (reset q x).next = ⟨0, 0⟩ := by rfl
"#;
    let compiled =
        compile_stateful_function(source, Path::new("machine.x"), "machine", None, &[]).unwrap();
    let mut bundle = compiled.emit_readable().unwrap();
    bundle
        .files
        .push(("PublicInterface.lean".into(), proof.into()));
    check_bundle(bundle, "positional outputs");
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn stateful_nested_helper_calls() {
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
fn machine(packet: Packet, __reg_count: u8) -> (u8, u8) {
    let request = foreign::Packet { base: packet.left, delta: packet.right };
    (__reg_count, advance(request, __reg_count))
}
"#;
    let compiled = compile_stateful_function(
        source,
        Path::new("machine.x"),
        "machine",
        None,
        &[modules.path()],
    )
    .unwrap();
    let mut bundle = compiled.emit_readable().unwrap();
    let (_, model) = bundle
        .files
        .iter()
        .find(|(name, _)| name == "ReadableGenerated.lean")
        .unwrap();
    for helper in ["left_side::advance", "advance"] {
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
fn imported_parametric_helpers_preserve_instances() {
    let modules = tempfile::tempdir().unwrap();
    let packets = r#"
pub struct Packet<N: u32> { value: bits[N], enabled: bool }
pub fn choose<N: u32>(packet: Packet<N>, fallback: bits[N]) -> Packet<N> {
    Packet<N> { value: if packet.enabled { packet.value } else { fallback }, ..packet }
}
pub fn bias<K: u32>(x: u6) -> u6 { x + (K as u6) }
"#;
    std::fs::write(modules.path().join("packets.x"), packets).unwrap();
    let source = r#"
import packets;
fn machine(small: packets::Packet<u32:5>, large: packets::Packet<u32:6>, __reg_last: u6) -> (u6, u6) {
    let a = packets::choose<u32:5>(small, u5:17);
    let b = packets::choose<u32:6>(large, u6:41);
    (__reg_last, packets::bias<u32:1>(u1:0 ++ a.value) + packets::bias<u32:2>(b.value))
}
"#;
    let model = compile_stateful_function(
        source,
        Path::new("machine.x"),
        "machine",
        None,
        &[modules.path()],
    )
    .unwrap();
    let mut bundle = model.emit_readable().unwrap();
    bundle.files.push((
        "Check.lean".into(),
        r#"import ReadableCorrespondence
open XlsReadable
private def stimulus : Input := {
  rst := false, small := { value := 3, enabled := false }, large := { value := 37, enabled := true } }
#guard (Helpers.«packets::choose<5>» (packet := stimulus.small) (fallback := 17)).value == 17
#guard (Helpers.«packets::choose<6>» (packet := stimulus.large) (fallback := 41)).value == 37
#guard (Helpers.«packets::bias<1>» (x := 0), Helpers.«packets::bias<2>» (x := 0)) == (1, 2)
#guard (step { last := 7 } stimulus).last == 57
"#
        .into(),
    ));
    check_bundle(bundle, "parametric");
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn stateful_counted_loop_matches_xls() {
    use std::fmt::Write as _;

    let source = r#"
fn fold(seed: u8, mask: u8, offset: u8) -> u8 {
    for (i, carry): (u8, u8) in u8:0..u8:4 {
        (carry & mask) + offset + i
    }(seed)
}
fn machine(mask: u8, offset: u8, __reg_count: u8) -> (u8, u8) {
    (__reg_count, fold(__reg_count, mask, offset))
}
"#;
    let path = Path::new("machine.x");
    let compiled = compile_stateful_function(source, path, "machine", None, &[]).unwrap();
    let mut bundle = compiled.emit_readable().unwrap();
    let package = xlsynth::convert_dslx_to_ir(source, path, &Default::default()).unwrap();
    let machine = package.ir.get_function("__machine__machine").unwrap();
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
fn stateful_neg_and_left_shift_match_xls() {
    use std::fmt::Write as _;

    let source = "fn machine(amount: u64, __reg_value: u5) -> (u5, u5) { (__reg_value, (-__reg_value) << amount) }";
    let path = Path::new("machine.x");
    let compiled = compile_stateful_function(source, path, "machine", None, &[]).unwrap();
    let mut bundle = compiled.emit_readable().unwrap();
    let package = xlsynth::convert_dslx_to_ir(source, path, &Default::default()).unwrap();
    let machine = package.ir.get_function("__machine__machine").unwrap();
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

// The outer singleton is a state update; the inner singleton is a tuple.
#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn stateful_unit_output_and_singleton_tuple_register() {
    let source = "type One = (bool,); fn machine(x: One, __reg_one: One) -> (One,) { (x,) }";
    let compiled =
        compile_stateful_function(source, Path::new("machine.x"), "machine", None, &[]).unwrap();
    let mut bundle = compiled.emit_readable().unwrap();
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
    check_bundle(bundle, "singleton register and no outputs");

    // With no registers the trace proof still must advance by one clock.
    for signature in ["fn machine() -> () { () }", "fn machine() { () }"] {
        let empty =
            compile_stateful_function(signature, Path::new("machine.x"), "machine", None, &[])
                .unwrap();
        check_bundle(empty.emit_readable().unwrap(), signature);
    }
}

// Alias transparency permits the update while the public field keeps Current.
#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn stateful_enum_and_transparent_next_state_alias() {
    let source = r#"
enum Phase: u2 { Start = 0, Run = 2 }
type Current = Phase;
fn machine(x: Phase, __reg_phase: Current) -> (Current, Phase) {
    (__reg_phase, x)
}
"#;
    let compiled =
        compile_stateful_function(source, Path::new("machine.x"), "machine", None, &[]).unwrap();
    let mut bundle = compiled.emit_readable().unwrap();
    bundle.files.push((
        "PublicInterface.lean".into(),
        r#"
import ReadableCorrespondence
open XlsReadable
example : XlsCodec.Phase.encode XlsSource.Phase.Run = BitVec.ofNat 2 2 := by rfl
example (q : State) (x : Input) : (observe q x : XlsSource.Current) = q.phase := by rfl
example (q : State) (x : Input) : (step q { x with rst := false }).phase = x.x := by
  cases x with | mk r p => cases p <;> rfl
example (q : State) (x : Input) : (reset q x).next.phase = XlsSource.Phase.Start := by rfl
"#
        .into(),
    ));
    check_bundle(bundle, "enum with aliased register");
}

// The source fields occupy both the default and alternative constructor names.
#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn stateful_mk_fields_keep_their_source_names() {
    let source = r#"
struct Packet { mk: u8, mk_: u8, mk__: u8 }
fn machine(mk: Packet, __reg_mk: Packet) -> (Packet, Packet) { (__reg_mk, mk) }
"#;
    let compiled =
        compile_stateful_function(source, Path::new("machine.x"), "machine", None, &[]).unwrap();
    let mut bundle = compiled.emit_readable().unwrap();
    bundle.files.push((
        "PublicInterface.lean".into(),
        r#"
import ReadableCorrespondence
open XlsReadable
example (q : State) (x : Input) : (observe q x).mk = q.mk.mk := by rfl
example (q : State) (x : Input) :
    let next := step q { x with rst := false }
    (next.mk.mk_, next.mk.mk__) = (x.mk.mk_, x.mk.mk__) := by rfl
"#
        .into(),
    ));
    check_bundle(bundle, "mk field on source struct, input and register");
}

// Field names remain projections even when they match a primitive type or
// the first component of the qualified name for a source type.
#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn stateful_fields_do_not_capture_later_types() {
    let source = r#"
struct Word { value: u8 }
struct Packet { XlsSource: u8, _root_: u8, word: Word }
fn machine(BitVec: u8, y: u8, _root_: u8, Prod: u8, Unit: u8,
           XlsSource: u8, packet: Packet, pair: (u8,),
           __reg_BitVec: u8, __reg__root_: u8, __reg_data: Packet) -> (u8, Packet, u8, u8, Packet) {
    (BitVec + y, __reg_data, _root_, pair.0, packet)
}
"#;
    let compiled =
        compile_stateful_function(source, Path::new("machine.x"), "machine", None, &[]).unwrap();
    let mut bundle = compiled.emit_readable().unwrap();
    bundle.files.push((
        "PublicInterface.lean".into(),
        r#"
import ReadableCorrespondence
open XlsReadable
example (q : State) (x : Input) : (observe q x).1 = x.BitVec + x.y := by rfl
example (q : State) (x : Input) : (observe q x).2.word.value = q.data.word.value := by rfl
example (q : State) (x : Input) :
    ((step q {x with rst := false}).BitVec, (step q {x with rst := false}).«_root_») =
    (x.«_root_», x.pair.1) := by rfl
"#
        .into(),
    ));
    check_bundle(bundle, "field names that can capture Lean types");
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
