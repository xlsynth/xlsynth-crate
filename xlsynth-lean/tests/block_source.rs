// SPDX-License-Identifier: Apache-2.0

//! Source-input contracts exercised through the actual native compiler.

use std::{fs, path::Path};
use xlsynth_lean::block::{compile_stateful_function, stateful_ir_to_block};
use xlsynth_lean::block::{emit, select_block};
use xlsynth_pir::ir_parser::Parser;

#[test]
fn registered_lowering_only_forwards_identities_and_removes_dead_nodes() {
    let ir = r#"package example

fn grant(x: bits[8] id=1) -> bits[8] {
  zero: bits[8] = literal(value=0, id=2)
  sum: bits[8] = add(x, zero, id=3)
  ret copy: bits[8] = identity(sum, id=4)
}

top fn machine(x: bits[8] id=5, __reg_q: bits[8] id=6) -> (bits[8], bits[8]) {
  winner: bits[8] = invoke(x, to_apply=grant, id=7)
  dead: bits[8] = xor(x, __reg_q, id=8)
  first: bits[8] = identity(winner, id=9)
  second: bits[8] = identity(first, id=10)
  ret results: (bits[8], bits[8]) = tuple(second, __reg_q, id=11)
}"#;
    let block = stateful_ir_to_block(ir, "machine").unwrap();
    assert!(!block.contains("identity("), "{block}");
    assert!(!block.contains("xor("), "{block}");
    assert!(block.contains("sum: bits[8] = add(x, zero,"), "{block}");
    assert!(block.contains("winner: bits[8] = invoke(x,"), "{block}");
    assert!(block.contains("to_apply=grant"), "{block}");
}

#[test]
fn dslx_registered_lowering_preserves_helper_calls_and_arithmetic() {
    let source = r#"
fn grant(requests: u8) -> u8 { requests + u8:0 }
fn machine(requests: u8, __reg_q: u8) -> (u8, u8) {
    let winner = grant(requests);
    (winner, __reg_q)
}
"#;
    let compiled =
        compile_stateful_function(source, Path::new("machine.x"), "machine", None, &[]).unwrap();
    let block = compiled.block_ir();
    assert!(block.contains(" = invoke("), "{block}");
    assert!(block.contains("to_apply=__machine__grant"), "{block}");
    assert!(block.contains(" = add("), "{block}");
}

#[test]
fn private_declarations_do_not_enter_the_public_type_module() {
    let source = r#"
struct Private { recOn: u1 }
enum Phase: u2 { Start = 0, Run = 2 }
type Pair = (Phase, u1);
struct Packet { pair: Pair }
type ResultTuple = (Packet,);
type ResultAlias = ResultTuple;
fn machine(x: Packet) -> ResultAlias { (x,) }
"#;
    let compiled =
        compile_stateful_function(source, Path::new("machine.x"), "machine", None, &[]).unwrap();
    let bundle = compiled.emit_readable().unwrap();
    let (_, types) = bundle
        .files
        .iter()
        .find(|(name, _)| name == "ReadableTypes.lean")
        .unwrap();
    for name in ["Phase", "Pair", "Packet"] {
        assert!(types.contains(name), "{name} missing from {types}");
    }
    for name in ["Private", "ResultTuple", "ResultAlias"] {
        assert!(!types.contains(name), "{name} unexpectedly in {types}");
    }
}

// Reject the signed public output even though its physical block is valid.
#[test]
fn unsupported_stateful_output_reports_its_position() {
    let source = "fn f(x: u8, __reg_n: u8) -> (s8, u8) { (x as s8, __reg_n) }";
    let compiled =
        compile_stateful_function(source, Path::new("counter.x"), "f", None, &[]).unwrap();
    let error = compiled.emit_readable().err().unwrap();
    let diagnostic = format!("{error:#}");
    assert!(
        diagnostic.contains("output out0") && diagnostic.contains("signed"),
        "{diagnostic}"
    );
}

// An identical physical layout cannot change an S register into a T.
#[test]
fn stateful_register_rejects_distinct_nominal_next_value_inside_tuple() {
    let source = r#"
struct S { a: u8 }
struct T { b: u8 }
fn machine(x: T, __reg_q: (S, bool)) -> ((T, bool),) { ((x, false),) }
"#;
    let compiled =
        compile_stateful_function(source, Path::new("machine.x"), "machine", None, &[]).unwrap();
    let diagnostic = format!("{:#}", compiled.emit_readable().err().unwrap());
    for expected in ["next state of register q", "S", "T"] {
        assert!(diagnostic.contains(expected), "{diagnostic}");
    }
}

// A generated kernel declaration cannot double as a projection or enum case.
#[test]
fn stateful_interface_rejects_fields_and_enum_members_reserved_by_lean() {
    for (declaration, member) in [
        ("struct Packet { recOn: u8 }", "recOn"),
        ("enum Packet: u2 { rec = 0, A = 1 }", "rec"),
        ("enum Packet: u2 { ofNat = 0, A = 1 }", "ofNat"),
    ] {
        let source = format!("{declaration} fn f(x: Packet) -> (Packet,) {{ (x,) }}");
        let compiled =
            compile_stateful_function(&source, Path::new("packet.x"), "f", None, &[]).unwrap();
        let diagnostic = format!("{:#}", compiled.emit_readable().err().unwrap());
        assert!(
            diagnostic.contains("Packet")
                && diagnostic.contains(member)
                && diagnostic.contains("Lean generates"),
            "{diagnostic}"
        );
    }
}

#[test]
fn stateful_interface_preserves_alias_from_source_with_imports() {
    let directory = tempfile::tempdir().unwrap();
    let stdlib = directory.path().join("stdlib");
    let imports = directory.path().join("imports");
    fs::create_dir(&stdlib).unwrap();
    fs::create_dir(&imports).unwrap();
    fs::write(stdlib.join("std.x"), "pub const ONE = u8:1;").unwrap();
    fs::write(imports.join("widths.x"), "pub const WIDTH = u32:8;").unwrap();
    let source = r#"
import std;
import widths;
const WIDTH = widths::WIDTH;
type Count = uN[WIDTH];
fn machine(__reg_count: Count) -> (Count, Count) {
    (__reg_count, __reg_count + std::ONE)
}
"#;
    let compiled = compile_stateful_function(
        source,
        Path::new("machine.x"),
        "machine",
        Some(&stdlib),
        &[&imports],
    )
    .unwrap();
    let bundle = compiled.emit_readable().unwrap();
    assert!(bundle.files.iter().any(|(_, text)| text.contains("Count")));
    // Emitting from the retained package must match emitting from its saved IR.
    let saved = Parser::new(compiled.block_ir())
        .parse_and_validate_package()
        .unwrap();
    let raw = emit(&saved, select_block(&saved, None).unwrap()).unwrap();
    let (_, generated) = bundle
        .files
        .iter()
        .find(|(name, _)| name == "Generated.lean")
        .unwrap();
    assert_eq!(generated, &raw);
}
