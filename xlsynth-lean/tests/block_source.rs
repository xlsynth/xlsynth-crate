// SPDX-License-Identifier: Apache-2.0

//! Source-input contracts exercised through the actual native compiler.

use std::{fs, path::Path};
use xlsynth_lean::block::compile_stateful_function;

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
}
