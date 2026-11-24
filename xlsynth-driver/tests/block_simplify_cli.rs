// SPDX-License-Identifier: Apache-2.0

//! The standalone simplification command composes with ordinary block2sv.

use std::path::Path;
use std::process::{Command, Output};

use xlsynth_pir::ir::{Binop, NodePayload};
use xlsynth_pir::ir_parser::Parser;

#[allow(dead_code)]
#[path = "../../xlsynth-codegen/tests/support/golden_fixture.rs"]
mod golden_fixture;
use golden_fixture::GoldenFixture;

const ALTERNATE: &str = r#"
block alternate(d: bits[8], q: bits[8]) {
  d: bits[8] = input_port(name=d, id=20)
  amount: bits[2] = literal(value=1, id=21)
  value: bits[8] = shll(d, amount, id=22)
  q: () = output_port(value, name=q, id=23)
}
"#;

/// Runs one driver subcommand without inheriting noisy logging settings.
fn invoke(directory: &Path, command: &str, arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(directory)
        .env_remove("RUST_LOG")
        .arg(command)
        .args(arguments)
        .output()
        .unwrap()
}

/// Returns stdout from a successful CLI command.
fn stdout(output: Output) -> String {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    String::from_utf8(output.stdout).unwrap()
}

/// Reads the independently authored expected SV for the fixed-shift design.
fn fixture() -> GoldenFixture {
    GoldenFixture::parse(include_str!("block2sv_simplify.golden.ir")).unwrap()
}

/// Checks whether a block still has the original shift operation.
fn has_shift(package: &xlsynth_pir::ir::Package, name: &str) -> bool {
    package
        .get_block(name)
        .unwrap()
        .nodes
        .iter()
        .any(|node| matches!(node.payload, NodePayload::Binop(Binop::Shll, ..)))
}

#[test]
fn simplifies_block_ir_before_independent_sv_emission() {
    let fixture = fixture();
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("input.ir"), &fixture.source).unwrap();
    let raw_sv = stdout(invoke(dir.path(), "block2sv", &["input.ir"]));
    let simplified_ir = stdout(invoke(dir.path(), "block-simplify", &["input.ir"]));
    let original = Parser::new(&fixture.source)
        .parse_and_verify_package()
        .unwrap();
    let expected =
        xlsynth_pir::block_simplify::simplify_block_hierarchy(&original, "shift").unwrap();
    assert_eq!(simplified_ir, expected.to_string());
    let parsed = Parser::new(&simplified_ir)
        .parse_and_verify_package()
        .unwrap();
    assert!(!has_shift(&parsed, "shift"));
    std::fs::write(dir.path().join("simplified.ir"), simplified_ir).unwrap();
    let simplified_sv = stdout(invoke(dir.path(), "block2sv", &["simplified.ir"]));
    assert_eq!(simplified_sv, fixture.expected_output());
    assert_ne!(raw_sv, simplified_sv);
}

#[test]
fn explicit_top_scopes_simplification_and_marks_output_top() {
    let fixture = fixture();
    let mut source = fixture.source;
    source.push_str(ALTERNATE);
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("input.ir"), source).unwrap();
    let output = stdout(invoke(
        dir.path(),
        "block-simplify",
        &["input.ir", "--top", "alternate"],
    ));
    let package = Parser::new(&output).parse_and_verify_package().unwrap();
    assert_eq!(package.get_top_block().unwrap().name, "alternate");
    assert!(has_shift(&package, "shift"));
    assert!(!has_shift(&package, "alternate"));
}

#[test]
fn no_selected_top_simplifies_every_block() {
    let fixture = fixture();
    let mut source = fixture.source.replace("top block shift", "block shift");
    source.push_str(ALTERNATE);
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("input.ir"), source).unwrap();
    let output = stdout(invoke(dir.path(), "block-simplify", &["input.ir"]));
    let package = Parser::new(&output).parse_and_verify_package().unwrap();
    assert!(package.top.is_none());
    assert!(!has_shift(&package, "shift"));
    assert!(!has_shift(&package, "alternate"));
}

#[test]
fn marked_top_avoids_expanding_unreachable_function_dag() {
    use std::fmt::Write as _;
    let fixture = fixture();
    let mut source = fixture.source.clone();
    source.push_str(
        "fn f0(x: bits[8] id=100) -> bits[8] { ret value: bits[8] = identity(x, id=101) }\n",
    );
    for n in 1..=18 {
        let id = 100 + 4 * n;
        writeln!(source, "fn f{n}(x: bits[8] id={id}) -> bits[8] {{ a: bits[8] = invoke(x, to_apply=f{}, id={}) b: bits[8] = invoke(a, to_apply=f{}, id={}) ret value: bits[8] = identity(b, id={}) }}", n - 1, id + 1, n - 1, id + 2, id + 3).unwrap();
    }
    source.push_str("block unrelated(d: bits[8], q: bits[8]) { d: bits[8] = input_port(name=d, id=300) v: bits[8] = invoke(d, to_apply=f18, id=301) q: () = output_port(v, name=q, id=302) }\n");
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("input.ir"), source).unwrap();
    let simplified = stdout(invoke(dir.path(), "block-simplify", &["input.ir"]));
    let package = Parser::new(&simplified).parse_and_verify_package().unwrap();
    assert!(
        package
            .get_block("unrelated")
            .unwrap()
            .nodes
            .iter()
            .any(|node| matches!(node.payload, NodePayload::Invoke { .. }))
    );
    std::fs::write(dir.path().join("simplified.ir"), simplified).unwrap();
    assert_eq!(
        stdout(invoke(dir.path(), "block2sv", &["simplified.ir"])),
        fixture.expected_output()
    );
}

#[test]
fn invalid_top_fails_without_output() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("input.ir"), fixture().source).unwrap();
    let output = invoke(
        dir.path(),
        "block-simplify",
        &["input.ir", "--top", "missing"],
    );
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("missing"));
}
