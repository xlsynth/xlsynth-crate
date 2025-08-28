// SPDX-License-Identifier: Apache-2.0

//! Tests for invoking the xlsynth-driver CLI and its subcommands as a
//! subprocess.

use std::collections::HashMap;
use std::io::Write;
use std::process::Command;
use xlsynth::IrBits;
use xlsynth_g8r::gate::AigBitVector;
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};

use test_case::test_case;

use pretty_assertions::assert_eq;
use xlsynth_test_helpers::{compare_golden_sv, compare_golden_text};

fn add_tool_path_value(toolchain_toml_contents: &str) -> String {
    let tool_path =
        std::env::var("XLSYNTH_TOOLS").expect("XLSYNTH_TOOLS environment variable must be set");
    format!(
        "{}
tool_path = \"{}\"",
        toolchain_toml_contents, tool_path
    )
}

// dslx-show tests

#[test]
fn test_dslx_show_function() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = r#"fn f(a: u32, b: u8) -> u32 { a + (b as u32) }"#;
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("f.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(driver)
        .arg("dslx-show")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("f")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let got = stdout.trim();
    let expected = "fn f(a: u32, b: u8) -> u32 {\n    a + (b as u32)\n}\n".trim();
    assert_eq!(got, expected, "stdout: {}", stdout);
}

#[test]
fn test_dslx_show_enum() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = r#"enum E : u8 { A = 0, B = 10 }"#;
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("e.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(driver)
        .arg("dslx-show")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("E")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let got = stdout.trim();
    let expected = "enum E : u8 {\n    A = 0,\n    B = 10,\n}\n".trim();
    assert_eq!(got, expected, "stdout: {}", stdout);
}

#[test]
fn test_dslx_show_struct() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = r#"struct S {
    a: u32,
    b: u8,
}"#;
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("s.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(driver)
        .arg("dslx-show")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("S")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let got = stdout.trim();
    let expected = "struct S {\n    a: u32,\n    b: u8,\n}\n".trim();
    assert_eq!(got, expected, "stdout: {}", stdout);
}

#[test]
fn test_dslx_show_std_max() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Create a local std.x so resolution works without external stdlib.
    let temp_dir = tempfile::tempdir().unwrap();
    let std_path = temp_dir.path().join("std.x");
    let std_text = r#"pub fn max<S: bool, N: u32>(x: xN[S][N], y: xN[S][N]) -> xN[S][N] { if x > y { x } else { y } }"#;
    std::fs::write(&std_path, std_text).unwrap();
    let main_path = temp_dir.path().join("main.x");
    std::fs::write(&main_path, "").unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(driver)
        .arg("dslx-show")
        .arg("--dslx_input_file")
        .arg(main_path.to_str().unwrap())
        .arg("std::max")
        .arg("--dslx_stdlib_path")
        .arg(temp_dir.path().to_str().unwrap())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let got = stdout.trim();
    let expected = "pub fn max<S: bool, N: u32>(x: xN[S][N], y: xN[S][N]) -> xN[S][N] {\n    if x > y { x } else { y }\n}\n".trim();
    assert_eq!(got, expected, "stdout: {}", stdout);
}

#[test]
fn test_dslx_show_imported_function_from_dslx_path() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Create separate lib and client dirs to ensure we exercise --dslx_path
    // resolution.
    let temp_dir = tempfile::tempdir().unwrap();
    let lib_dir = temp_dir.path().join("lib");
    let client_dir = temp_dir.path().join("client");
    std::fs::create_dir(&lib_dir).unwrap();
    std::fs::create_dir(&client_dir).unwrap();

    // Library module providing a public function.
    let lib_path = lib_dir.join("mylib.x");
    let lib_text = r#"pub fn inc(x: u8) -> u8 {
    x + u8:1
}
"#;
    std::fs::write(&lib_path, lib_text).unwrap();

    // Client module that imports the library.
    let main_path = client_dir.join("main.x");
    std::fs::write(&main_path, "import mylib;").unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(driver)
        .arg("dslx-show")
        .arg("--dslx_input_file")
        .arg(main_path.to_str().unwrap())
        .arg("mylib::inc")
        .arg("--dslx_path")
        .arg(lib_dir.to_str().unwrap())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let got = stdout.trim();
    let expected = "pub fn inc(x: u8) -> u8 {\n    x + u8:1\n}\n".trim();
    assert_eq!(got, expected, "stdout: {}", stdout);
}

#[test]
fn test_dslx_show_imported_function_without_input_file() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Library-only query: no input file provided; resolve via --dslx_path.
    let temp_dir = tempfile::tempdir().unwrap();
    let lib_dir = temp_dir.path().join("lib");
    std::fs::create_dir(&lib_dir).unwrap();
    let lib_path = lib_dir.join("mylib.x");
    let lib_text = r#"pub fn inc(x: u8) -> u8 {
    x + u8:1
}
"#;
    std::fs::write(&lib_path, lib_text).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(driver)
        .arg("dslx-show")
        .arg("mylib::inc")
        .arg("--dslx_path")
        .arg(lib_dir.to_str().unwrap())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let got = stdout.trim();
    let expected = "pub fn inc(x: u8) -> u8 {\n    x + u8:1\n}\n".trim();
    assert_eq!(got, expected, "stdout: {}", stdout);
}

#[test]
fn test_dslx_show_dotted_module_path_nested_type() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Library root with nested module path: foo/bar/baz.x
    let temp_dir = tempfile::tempdir().unwrap();
    let lib_root = temp_dir.path().join("lib");
    std::fs::create_dir_all(lib_root.join("foo/bar")).unwrap();

    let baz_path = lib_root.join("foo/bar/baz.x");
    let baz_text = r#"pub type T = u8;
pub fn inc(x: u8) -> u8 { x + u8:1 }
"#;
    std::fs::write(&baz_path, baz_text).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    // Use dotted module path with :: for the member name.
    let output = std::process::Command::new(driver)
        .arg("dslx-show")
        .arg("foo.bar.baz::T")
        .arg("--dslx_path")
        .arg(lib_root.to_str().unwrap())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert_eq!(stdout.trim(), "pub type T = u8;".trim());
}

#[test]
fn test_dslx_show_type_alias() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = r#"type Alias = u8;"#;
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("a.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out = std::process::Command::new(driver)
        .arg("dslx-show")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("Alias")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert_eq!(
        stdout.trim(),
        "type Alias = u8;".trim(),
        "stdout: {}",
        stdout
    );
}

#[test]
fn test_dslx_show_constant() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = r#"const TEN = u8:10;"#;
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("c.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out = std::process::Command::new(driver)
        .arg("dslx-show")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("TEN")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert_eq!(
        stdout.trim(),
        "const TEN = u8:10;".trim(),
        "stdout: {}",
        stdout
    );
}

#[test]
fn test_dslx_show_quickcheck() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = r#"#[quickcheck] fn qc(x: u8) -> bool { x == x }"#;
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("q.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(driver)
        .arg("dslx-show")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("qc")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let got = stdout.trim();
    let expected = "#[quickcheck]\nfn qc(x: u8) -> bool {\n    x == x\n}\n".trim();
    assert_eq!(got, expected, "stdout: {}", stdout);
}

#[test]
fn test_dslx_show_quickcheck_exhaustive() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = r#"#[quickcheck(exhaustive)] fn qc_exh(x: u1) -> bool { x == x }"#;
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("q_exh.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(driver)
        .arg("dslx-show")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("qc_exh")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let got = stdout.trim();
    let expected = "#[quickcheck(exhaustive)]\nfn qc_exh(x: u1) -> bool {\n    x == x\n}\n".trim();
    assert_eq!(got, expected, "stdout: {}", stdout);
}

#[test]
fn test_dslx_show_quickcheck_count() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = r#"#[quickcheck(test_count=7)] fn qc_cnt(x: u8) -> bool { x == x }"#;
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("q_cnt.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(driver)
        .arg("dslx-show")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("qc_cnt")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let got = stdout.trim();
    let expected =
        "#[quickcheck(test_count=7)]\nfn qc_cnt(x: u8) -> bool {\n    x == x\n}\n".trim();
    assert_eq!(got, expected, "stdout: {}", stdout);
}

#[test]
fn test_ir2gates_adder_mapping_flag() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = "fn main(a: u8, b: u8) -> u8 { a + b }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("main.x");
    let ir_path = temp_dir.path().join("main.ir");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");

    // dslx2ir
    let dslx2ir_output = std::process::Command::new(command_path)
        .arg("dslx2ir")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .output()
        .unwrap();
    assert!(dslx2ir_output.status.success());
    std::fs::write(&ir_path, &dslx2ir_output.stdout).unwrap();

    fn run(driver: &str, ir: &std::path::Path, mapping: &str) -> serde_json::Value {
        let out = std::process::Command::new(driver)
            .arg("ir2gates")
            .arg(ir.to_str().unwrap())
            .arg("--quiet=true")
            .arg(format!("--adder-mapping={}", mapping))
            .output()
            .unwrap();
        assert!(out.status.success());
        serde_json::from_slice(&out.stdout).unwrap()
    }

    let rc = run(command_path, &ir_path, "ripple-carry");
    let bk = run(command_path, &ir_path, "brent-kung");
    let ks = run(command_path, &ir_path, "kogge-stone");

    assert_eq!(rc["deepest_path"], 22);
    assert_eq!(bk["deepest_path"], 13);
    assert_eq!(ks["deepest_path"], 17);
}

#[test]
fn test_dslx2ir_no_top_required_when_no_opt() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = "\nfn foo(x: u32) -> u32 { x + u32:1 }\nfn bar(y: u32) -> u32 { y + u32:2 }\n";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("foo.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("dslx2ir")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "dslx2ir should not require --dslx_top when --opt is not set;\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("package foo"),
        "Expected package name 'foo' in IR, got: {}",
        stdout
    );
    assert!(
        stdout.contains("fn __foo__foo("),
        "Expected function __foo__foo in IR, got: {}",
        stdout
    );
    assert!(
        stdout.contains("fn __foo__bar("),
        "Expected function __foo__bar in IR, got: {}",
        stdout
    );
}

#[test]
fn test_ir2gates_prints_source_positions() {
    let _ = env_logger::builder().is_test(true).try_init();

    let ir = r#"package prio_pkg
file_number 0 "foo.x"
file_number 1 "bar.x"

top fn main(sel: bits[1] id=1, a: bits[1] id=2, b: bits[1] id=3) -> bits[1] {
  p: bits[1] = priority_sel(sel, cases=[a], default=b, id=4, pos=[(0,1,0), (1,2,0)])
  ret result: bits[1] = identity(p, id=5)
}"#;

    let mut temp_file = tempfile::Builder::new().suffix(".ir").tempfile().unwrap();
    write!(temp_file, "{}", ir).unwrap();
    let ir_path = temp_file.into_temp_path();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("ir2gates")
        .arg(ir_path.to_str().unwrap())
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();

    xlsynth_test_helpers::compare_golden_text(
        &stdout,
        "tests/test_ir2gates_show_source.golden.txt",
    );
}

/// Simple test that converts a DSLX module with an enum into a SV definition.
#[test_case(true; "with_tool_path")]
#[test_case(false; "without_tool_path")]
fn test_dslx2sv_types_subcommand(use_tool_path: bool) {
    let dslx = "enum OpType : u2 { READ = 0, WRITE = 1 }";
    // Make a temporary file to hold the DSLX code.
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("my_module.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    // Run the dslx2sv subcommand.
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");

    let toolchain_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml = "[toolchain]\n";
    let toolchain_toml_path = if use_tool_path {
        add_tool_path_value(&toolchain_toml)
    } else {
        toolchain_toml.to_string()
    };
    std::fs::write(&toolchain_path, toolchain_toml_path).unwrap();

    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_path.to_str().unwrap())
        .arg("dslx2sv-types")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    xlsynth_test_helpers::assert_valid_sv(&stdout);
    assert_eq!(
        stdout.trim(),
        r"typedef enum logic [1:0] {
    Read = 2'd0,
    Write = 2'd1
} op_type_t;"
    );
}

#[test]
fn test_dslx2sv_types_with_std_clog2() {
    let dslx = "import std;

const COUNT = u32:24;
const WIDTH = std::clog2(COUNT);

struct MyStruct {
    data: bits[WIDTH],
}
";
    // Make a temporary file to hold the DSLX code.
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("my_module.x");
    std::fs::write(&dslx_path, dslx).unwrap();
    // Run the dslx2sv subcommand.
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("dslx2sv-types")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    xlsynth_test_helpers::assert_valid_sv(&stdout);
    assert_eq!(
        stdout.trim(),
        "localparam bit unsigned [31:0] Count = 32'h00000018;

localparam bit unsigned [31:0] Width = 32'h00000005;

typedef struct packed {
    logic [4:0] data;
} my_struct_t;"
    );
}

/// Tests that we can point at a xlsynth-toolchain.toml file to get
/// an alternative DSLX stdlib implementation.
#[test_case(true; "with_tool_path")]
#[test_case(false; "without_tool_path")]
fn test_dslx2ir_with_toolchain_toml(use_tool_path: bool) {
    let dslx = "import std; fn main(x: u32) -> u32 { std::popcount(x) }";
    let fake_std = "pub fn popcount(x: u32) -> u32 { x }";
    // Get a temporary directory to use as a root.
    let temp_dir = tempfile::tempdir().unwrap();

    // Make a subdirectory for our fake stdlib and one for our client code.
    let stdlib_dir = temp_dir.path().join("stdlib");
    std::fs::create_dir(&stdlib_dir).unwrap();
    let client_dir = temp_dir.path().join("client");
    std::fs::create_dir(&client_dir).unwrap();

    // Write the fake std.x file to the stdlib dir.
    let stdlib_path = stdlib_dir.join("std.x");
    std::fs::write(&stdlib_path, fake_std).unwrap();

    // Write the client code to the client subdir.
    let client_path = client_dir.join("client.x");
    std::fs::write(&client_path, dslx).unwrap();

    // Write a xlsynth-toolchain.toml file to the root dir.
    let toolchain_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml = format!(
        r#"[toolchain]

[toolchain.dslx]
dslx_stdlib_path = "{}"
"#,
        stdlib_dir.to_str().unwrap()
    );
    let toolchain_toml_path = if use_tool_path {
        add_tool_path_value(&toolchain_toml)
    } else {
        toolchain_toml
    };
    std::fs::write(&toolchain_path, toolchain_toml_path).unwrap();

    // Run the dslx2ir subcommand.
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_path.to_str().unwrap())
        .arg("dslx2ir")
        .arg("--dslx_input_file")
        .arg(client_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(stdout.contains(
        "fn __std__popcount(x: bits[32] id=1) -> bits[32] {
  ret x: bits[32] = param(name=x, id=1)
}"
    ))
}

#[test]
fn test_dslx2pipeline_with_update_of_1d_array() {
    let dslx = "import std;

struct MyStruct {
    some_bool: bool,
    data: u32,
}

fn main(x: MyStruct[4]) -> MyStruct[4] {
    update(x, u32:1, MyStruct{some_bool: false, data: u32:42})
}
";

    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("my_module.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("dslx2pipeline")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--delay_model")
        .arg("asap7")
        .arg("--flop_inputs=true")
        .arg("--flop_outputs=false")
        .arg("--add_idle_output=true")
        .arg("--separate_lines=false")
        .arg("--use_system_verilog=true")
        .arg("--array_index_bounds_checking=true")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    xlsynth_test_helpers::assert_valid_sv(&stdout);

    compare_golden_sv(
        &stdout,
        "tests/test_dslx2pipeline_with_update_of_1d_array.golden.sv",
    );
}

// To get the redundant match arm to flag a warning we have to enable a
// non-default warning as of DSO v0.0.134.
#[test_case(true; "with_tool_path")]
#[test_case(false; "without_tool_path")]
fn test_dslx2pipeline_with_redundant_match_arm(use_tool_path: bool) {
    let _ = env_logger::try_init();
    log::info!("test_dslx2pipeline_with_redundant_match_arm");
    let dslx = "fn main(x: bool) -> u32 {
        match x {
            true => u32:42,
            false => u32:64,
            _ => u32:128,
        }
    }";

    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("my_module.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Enable the warning explicitly in the configuration toml.
    let toolchain_toml = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = format!(
        r#"[toolchain]

[toolchain.dslx]
enable_warnings = ["already_exhaustive_match"]
"#
    );
    let toolchain_toml_path = if use_tool_path {
        add_tool_path_value(&toolchain_toml_contents)
    } else {
        toolchain_toml_contents
    };
    std::fs::write(&toolchain_toml, toolchain_toml_path).unwrap();

    let rust_log = std::env::var("RUST_LOG").unwrap_or_default();
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("dslx2pipeline")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--delay_model")
        .arg("asap7")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .env("RUST_LOG", rust_log)
        .output()
        .unwrap();

    // Check that the output shows the warning and that the return code is
    // non-success because we have warnings-as-errors on.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "stdout: {}\nstderr: {}",
        stdout,
        stderr
    );
    assert!(
        stderr.contains("Match is already exhaustive"),
        "stdout: {}\nstderr: {}",
        stdout,
        stderr
    );
}

#[test_case(true; "with_tool_path")]
#[test_case(false; "without_tool_path")]
fn test_dslx2pipeline_with_unused_binding(use_tool_path: bool) {
    let _ = env_logger::try_init();
    log::info!("test_dslx2pipeline_with_unused_binding");

    let dslx = "fn main() -> u32 {
    let x = u32:42;  // unused_definition
    for (i, accum) in u32:0..u32:0 {  // empty_range_literal
        accum
    }(u32:64)
}";

    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("my_module.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let toolchain_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml = "[toolchain]\n";
    let toolchain_toml_contents = if use_tool_path {
        add_tool_path_value(&toolchain_toml)
    } else {
        toolchain_toml.to_string()
    };
    std::fs::write(&toolchain_path, toolchain_toml_contents).unwrap();
    // Initial run should fail since we don't disable the warning.
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let rust_log = std::env::var("RUST_LOG").unwrap_or_default();
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_path.to_str().unwrap())
        .arg("dslx2pipeline")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--delay_model")
        .arg("asap7")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .env("RUST_LOG", &rust_log)
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    log::info!("stdout: {}", stdout);
    log::info!("stderr: {}", stderr);
    assert!(
        stderr.contains("is not used in function"),
        "stdout: {}\nstderr: {}",
        stdout,
        stderr
    );

    // Now run again with the warning disabled via toml config.
    // Also use the tool path instead of the runtime APIs.
    let toolchain_toml = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = r#"[toolchain]

[toolchain.dslx]
disable_warnings = ["unused_definition", "empty_range_literal"]
"#;
    let toolchain_toml_path = if use_tool_path {
        add_tool_path_value(&toolchain_toml_contents)
    } else {
        toolchain_toml_contents.to_string()
    };
    std::fs::write(&toolchain_toml, toolchain_toml_path).unwrap();

    log::info!("running again with warnings disabled...");
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("dslx2pipeline")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--delay_model")
        .arg("asap7")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .env("RUST_LOG", &rust_log)
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    log::info!("stdout: {}", stdout);
    log::info!("stderr: {}", stderr);

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        stdout,
        stderr
    );
    assert!(!stdout.contains("is not used in function"));
}

#[test]
fn test_dslx2pipeline_with_reset_signal() {
    let _ = env_logger::try_init();
    log::info!("test_dslx2pipeline_with_reset_signal");

    let dslx = "fn main(x: u32) -> u32 {
        x + u32:1
    }";

    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("my_module.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");

    let output = Command::new(command_path)
        .arg("dslx2pipeline")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--delay_model")
        .arg("asap7")
        .arg("--flop_inputs=true")
        .arg("--flop_outputs=true")
        .arg("--use_system_verilog=false")
        .arg("--input_valid_signal=input_valid")
        .arg("--output_valid_signal=output_valid")
        .arg("--reset=rst_n")
        .arg("--reset_active_low=true")
        .arg("--reset_asynchronous=false")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr);
    log::info!("stdout: {}", stdout);
    log::info!("stderr: {}", stderr);
    xlsynth_test_helpers::assert_valid_sv(&stdout);

    compare_golden_sv(
        &stdout,
        "tests/test_dslx2pipeline_with_reset_signal.golden.sv",
    );
}

#[test_case(true; "reset_datapath")]
#[test_case(false; "no_reset_datapath")]
fn test_dslx2pipeline_reset_data_path(reset_dp: bool) {
    let _ = env_logger::try_init();
    let dslx = "fn main(x: u32) -> u32 { x + u32:1 }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("my_module.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("dslx2pipeline")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--delay_model")
        .arg("asap7")
        .arg("--flop_inputs=true")
        .arg("--flop_outputs=true")
        .arg("--use_system_verilog=false")
        .arg("--input_valid_signal=input_valid")
        .arg("--output_valid_signal=output_valid")
        .arg("--reset=rst_n")
        .arg("--reset_active_low=true")
        .arg("--reset_asynchronous=false")
        .arg(format!("--reset_data_path={reset_dp}"))
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    xlsynth_test_helpers::assert_valid_sv(&stdout);

    let golden_name = if reset_dp {
        "tests/test_dslx2pipeline_reset_data_path_true.golden.sv"
    } else {
        "tests/test_dslx2pipeline_reset_data_path_false.golden.sv"
    };
    compare_golden_sv(&stdout, golden_name);
}

#[test_case(true; "with_tool_path")]
#[test_case(false; "without_tool_path")]
fn test_ir2opt_subcommand(use_tool_path: bool) {
    let _ = env_logger::try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let ir_path = temp_dir.path().join("ir.ir");
    std::fs::write(
        &ir_path,
        "package sample
fn my_main(x: bits[32]) -> bits[32] {
    literal.3: bits[32] = literal(value=0, id=3)
    ret add.4: bits[32] = add(literal.3, x, id=4)
}",
    )
    .unwrap();

    // Write out toolchain configuration.
    let toolchain_toml = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = "[toolchain]\n".to_string();
    let toolchain_toml_contents = if use_tool_path {
        add_tool_path_value(&toolchain_toml_contents)
    } else {
        toolchain_toml_contents
    };
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("ir2opt")
        .arg(ir_path.to_str().unwrap())
        .arg("--top")
        .arg("my_main")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "ir2opt should succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout,
        "package sample

top fn my_main(x: bits[32] id=5) -> bits[32] {
  ret x: bits[32] = param(name=x, id=5)
}\n\n"
    );
}

#[test_case(true; "with_tool_path")]
#[test_case(false; "without_tool_path")]
fn test_dslx2pipeline_with_dslx_path_two_entries(use_tool_path: bool) {
    let temp_dir = tempfile::tempdir().unwrap();

    // Make dir `a` and `b` in the temp dir.
    let a_dir = temp_dir.path().join("a");
    std::fs::create_dir(&a_dir).unwrap();
    let b_dir = temp_dir.path().join("b");
    std::fs::create_dir(&b_dir).unwrap();

    // Make a file `a/a.x` and `b/b.x` in the temp dir with constants `A` and `B`
    // respectively.
    let a_path = a_dir.join("a.x");
    std::fs::write(&a_path, "pub const A: u32 = u32:42;").unwrap();
    let b_path = b_dir.join("b.x");
    std::fs::write(&b_path, "pub const B: u32 = u32:64;").unwrap();

    // Write out the main file that just imports `a` and `b` and adds a::A + b::B.
    let main_path = temp_dir.path().join("main.x");
    std::fs::write(
        &main_path,
        "import a;
import b;
fn main() -> u32 { a::A + b::B }",
    )
    .unwrap();

    // Run the dslx2pipeline command with the DSLX_PATH set to `a:b` via the
    // toolchain config.
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Write out toolchain configuration.
    let toolchain_toml = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = format!(
        r#"[toolchain]

[toolchain.dslx]
dslx_path = ["{}", "{}"]
"#,
        a_dir.to_str().unwrap(),
        b_dir.to_str().unwrap()
    );
    let toolchain_toml_contents = if use_tool_path {
        add_tool_path_value(&toolchain_toml_contents)
    } else {
        toolchain_toml_contents
    };
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    let rust_log = std::env::var("RUST_LOG").unwrap_or_default();
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("dslx2pipeline")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--delay_model")
        .arg("asap7")
        .arg("--dslx_input_file")
        .arg(main_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .env("RUST_LOG", rust_log)
        .output()
        .expect("xlsynth-driver should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    log::info!("stdout: {}", stdout);
    log::info!("stderr: {}", stderr);
    assert!(
        stdout.contains("06a"),
        "stdout: {} stderr: {}",
        stdout,
        stderr
    );
}

#[test]
fn test_irequiv_subcommand_equivalent() {
    let _ = env_logger::try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let lhs_ir = "package add_then_sub
fn my_main(x: bits[32]) -> bits[32] {
    add.2: bits[32] = add(x, x)
    ret sub.3: bits[32] = sub(add.2, x)
}";
    let rhs_ir = "package identity
fn my_main(x: bits[32]) -> bits[32] {
    ret identity.2: bits[32] = identity(x)
}";
    // Write the IR files to the temp directory.
    let lhs_path = temp_dir.path().join("lhs.ir");
    std::fs::write(&lhs_path, lhs_ir).unwrap();
    let rhs_path = temp_dir.path().join("rhs.ir");
    std::fs::write(&rhs_path, rhs_ir).unwrap();

    // Write out toolchain configuration.
    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = "[toolchain]\n".to_string();
    let toolchain_toml_contents_with_path = add_tool_path_value(&toolchain_toml_contents);
    std::fs::write(&toolchain_toml_path, toolchain_toml_contents_with_path).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("ir-equiv")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--top")
        .arg("my_main")
        .output()
        .expect("xlsynth-driver should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    log::info!("stdout: {}", stdout);
    log::info!("stderr: {}", stderr);
    assert!(stdout.contains("[ir-equiv] success: Solver proved equivalence"));
}

#[test]
fn test_irequiv_subcommand_non_equivalent() {
    let _ = env_logger::try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let lhs_ir = "package add_then_sub
fn my_main(x: bits[32]) -> bits[32] {
    umul.2: bits[32] = umul(x, x)
    ret udiv.3: bits[32] = udiv(umul.2, x)
}";
    let rhs_ir = "package identity
fn my_main(x: bits[32]) -> bits[32] {
    ret identity.2: bits[32] = identity(x)
}";
    // Write the IR files to the temp directory.
    let lhs_path = temp_dir.path().join("lhs.ir");
    std::fs::write(&lhs_path, lhs_ir).unwrap();
    let rhs_path = temp_dir.path().join("rhs.ir");
    std::fs::write(&rhs_path, rhs_ir).unwrap();

    // Write out toolchain configuration.
    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = "[toolchain]\n".to_string();
    let toolchain_toml_contents_with_path = add_tool_path_value(&toolchain_toml_contents);
    std::fs::write(&toolchain_toml_path, toolchain_toml_contents_with_path).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("ir-equiv")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--top")
        .arg("my_main")
        .output()
        .expect("xlsynth-driver should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    log::info!("retcode: {}", output.status);
    log::info!("stdout: {}", stdout);
    log::info!("stderr: {}", stderr);

    // Check that the error code is non-zero.
    assert!(!output.status.success());
    assert!(stdout.is_empty());
    assert!(
        stderr.contains("[ir-equiv] failure: Verified NOT equivalent; results differ for input"),
        "stderr: {:?}",
        stderr
    );
}

#[test]
fn test_irequiv_blocks_subcommand_equivalent() {
    let _ = env_logger::try_init();
    let temp_dir = tempfile::tempdir().unwrap();

    let block = r#"package test_pkg

block my_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=1)
  b: bits[32] = input_port(name=b, id=2)
  add.3: bits[32] = add(a, b, id=3)
  out: () = output_port(add.3, name=out, id=4)
}
"#;
    let lhs_path = temp_dir.path().join("lhs.ir");
    std::fs::write(&lhs_path, block).unwrap();
    let rhs_path = temp_dir.path().join("rhs.ir");
    std::fs::write(&rhs_path, block).unwrap();

    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = "[toolchain]\n".to_string();
    let toolchain_toml_contents_with_path = add_tool_path_value(&toolchain_toml_contents);
    std::fs::write(&toolchain_toml_path, toolchain_toml_contents_with_path).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("ir-equiv-blocks")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--lhs_top")
        .arg("my_block")
        .arg("--rhs_top")
        .arg("my_block")
        .output()
        .expect("xlsynth-driver should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "status not success\nstdout: {}\nstderr: {}",
        stdout,
        stderr
    );
    assert!(
        stdout.contains("[ir-equiv-blocks] success: Solver proved equivalence"),
        "stdout: {}\nstderr: {}",
        stdout,
        stderr
    );
}

#[test]
fn test_irequiv_blocks_subcommand_non_equivalent() {
    let _ = env_logger::try_init();
    let temp_dir = tempfile::tempdir().unwrap();

    let lhs_block = r#"package lhs_pkg

block add_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=1)
  b: bits[32] = input_port(name=b, id=2)
  add.3: bits[32] = add(a, b, id=3)
  out: () = output_port(add.3, name=out, id=4)
}
"#;
    let rhs_block = r#"package rhs_pkg

block sub_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=1)
  b: bits[32] = input_port(name=b, id=2)
  sub.3: bits[32] = sub(a, b, id=3)
  out: () = output_port(sub.3, name=out, id=4)
}
"#;
    let lhs_path = temp_dir.path().join("lhs.ir");
    std::fs::write(&lhs_path, lhs_block).unwrap();
    let rhs_path = temp_dir.path().join("rhs.ir");
    std::fs::write(&rhs_path, rhs_block).unwrap();

    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = "[toolchain]\n".to_string();
    let toolchain_toml_contents_with_path = add_tool_path_value(&toolchain_toml_contents);
    std::fs::write(&toolchain_toml_path, toolchain_toml_contents_with_path).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("ir-equiv-blocks")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--lhs_top")
        .arg("add_block")
        .arg("--rhs_top")
        .arg("sub_block")
        .output()
        .expect("xlsynth-driver should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "expected failure\nstdout: {}\nstderr: {}",
        stdout,
        stderr
    );
    assert!(
        stderr.contains("[ir-equiv-blocks] failure"),
        "stdout: {}\nstderr: {}",
        stdout,
        stderr
    );
}

#[test]
fn test_ir2gates_determinism() {
    let _ = env_logger::try_init();
    log::info!("test_ir2gates_determinism");

    // Use 4-bit operands instead of full 32-bit values so that the
    // gate-level representation is significantly smaller. This keeps the
    // FRAIG optimization step in `ir2gates` from dominating the test runtime.
    let dslx = "fn main(a: u4, b: u4) -> u4 { a + b * (a ^ b) }";

    // Make a temporary directory and files.
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("determinism_test.x");
    let ir_path = temp_dir.path().join("determinism_test.ir");

    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Run dslx2ir first, capture stdout, and write to the IR file.
    let dslx2ir_output = Command::new(command_path)
        .arg("dslx2ir")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        // No output file flag, prints to stdout
        .output()
        .unwrap();

    assert!(
        dslx2ir_output.status.success(),
        "dslx2ir failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&dslx2ir_output.stdout),
        String::from_utf8_lossy(&dslx2ir_output.stderr)
    );
    // Manually write the captured IR to the file
    std::fs::write(&ir_path, &dslx2ir_output.stdout).unwrap();

    // Run ir2gates the first time and capture stdout.
    let mut command = std::process::Command::new(command_path);
    command
        .arg("ir2gates")
        .arg(ir_path.to_str().unwrap())
        .arg("--quiet=true")
        .arg("--toggle-sample-count=2")
        .arg("--toggle-seed=42");
    // Pass through RUST_LOG if present
    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        command.env("RUST_LOG", rust_log);
    }
    let ir2gates_output1 = command.output().unwrap();

    log::debug!(
        "ir2gates stdout:\n{}",
        String::from_utf8_lossy(&ir2gates_output1.stdout)
    );
    log::debug!(
        "ir2gates stderr:\n{}",
        String::from_utf8_lossy(&ir2gates_output1.stderr)
    );

    assert!(
        ir2gates_output1.status.success(),
        "ir2gates run 1 failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&ir2gates_output1.stdout),
        String::from_utf8_lossy(&ir2gates_output1.stderr)
    );
    let output_str_1 = String::from_utf8(ir2gates_output1.stdout).unwrap();

    // Run ir2gates the second time and capture stdout.
    let mut command = std::process::Command::new(command_path);
    command
        .arg("ir2gates")
        .arg(ir_path.to_str().unwrap())
        .arg("--quiet=true")
        .arg("--toggle-sample-count=2")
        .arg("--toggle-seed=42");
    // Pass through RUST_LOG if present
    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        command.env("RUST_LOG", rust_log);
    }
    let ir2gates_output2 = command.output().unwrap();

    log::debug!(
        "ir2gates stdout:\n{}",
        String::from_utf8_lossy(&ir2gates_output2.stdout)
    );
    log::debug!(
        "ir2gates stderr:\n{}",
        String::from_utf8_lossy(&ir2gates_output2.stderr)
    );

    assert!(
        ir2gates_output2.status.success(),
        "ir2gates run 2 failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&ir2gates_output2.stdout),
        String::from_utf8_lossy(&ir2gates_output2.stderr)
    );
    let output_str_2 = String::from_utf8(ir2gates_output2.stdout).unwrap();

    // Compare the captured stdout strings.
    assert_eq!(
        output_str_1, output_str_2,
        "ir2gates output is non-deterministic!"
    );
}

#[test_case(true; "with_tool_path")]
fn test_toolchain_picked_up_when_in_cwd_even_if_no_cmdline_flag(use_tool_path: bool) {
    let temp_dir = tempfile::tempdir().unwrap();
    let toolchain_toml = temp_dir.path().join("xlsynth-toolchain.toml");
    // Create a DSLX stdlib path that points at our temp dir so we can see things
    // worked.
    let dslx_stdlib_path = temp_dir.path().join("dslx_stdlib");
    std::fs::create_dir(&dslx_stdlib_path).unwrap();
    let dslx_stdlib_path_str = dslx_stdlib_path.to_str().unwrap();
    let toolchain_toml_contents = format!(
        r#"[toolchain]

[toolchain.dslx]
dslx_stdlib_path = "{}"
"#,
        dslx_stdlib_path_str
    );
    let toolchain_toml_contents = if use_tool_path {
        add_tool_path_value(&toolchain_toml_contents)
    } else {
        toolchain_toml_contents
    };
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    // Create a std.x file that just has a popcount that always returns 1. This
    // distinguishes it from the real standard library.
    let std_path = dslx_stdlib_path.join("std.x");
    std::fs::write(&std_path, "pub fn popcount(x: u32) -> u32 { u32:1 }").unwrap();

    // Create a main.x file that just imports std and calls popcount.
    let main_path = temp_dir.path().join("main.x");
    std::fs::write(
        &main_path,
        "import std; fn main() -> u32 { std::popcount(u32:42) }",
    )
    .unwrap();

    // We run the command without an explicit --toolchain flag but in the directory
    // where the toolchain.toml is located.
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("dslx2ir")
        .arg("--dslx_input_file")
        .arg(main_path.to_str().unwrap())
        .arg("--opt=true")
        .arg("--dslx_top")
        .arg("main")
        .current_dir(temp_dir.path())
        .output()
        .expect("xlsynth-driver should succeed");

    // Command should succeed.
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.is_empty(), "stderr should be empty; got: {}", stderr);

    // The IR should just be the literal value 1.
    assert!(
        stdout.contains(
            "top fn __main__main() -> bits[32] {
  ret literal.5: bits[32] = literal(value=1, id=5, pos=[(1,0,44)])
}"
        ),
        "stdout: {}",
        stdout
    );
}

#[test]
fn test_dslx_add_sub_opt_ir2gates_pipeline() {
    let _ = env_logger::try_init();
    let dslx = "
import std;
const UNUSED = std::popcount(u3:0b111);
fn f(x: u8) -> u8 { x + x - x }
#[test] fn test_my_add() { assert_eq(f(u8::MAX), u8::MAX); }
#[quickcheck] fn quickcheck_my_add(x: u8) -> bool { f(x) == x }
";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("f.x");
    let ir_path = temp_dir.path().join("f.opt.ir");
    std::fs::write(&dslx_path, dslx).unwrap();
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Step 1: dslx2ir
    let dslx2ir_output = std::process::Command::new(command_path)
        .arg("dslx2ir")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("f")
        .output()
        .unwrap();
    assert!(
        dslx2ir_output.status.success(),
        "dslx2ir failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&dslx2ir_output.stdout),
        String::from_utf8_lossy(&dslx2ir_output.stderr)
    );
    std::fs::write(&ir_path, &dslx2ir_output.stdout).unwrap();

    log::info!(
        "unoptimized IR:\n{}",
        String::from_utf8_lossy(&dslx2ir_output.stdout)
    );

    // Compute the mangled IR top function name as the driver does
    let module_name = dslx_path.file_stem().unwrap().to_str().unwrap();
    let dslx_top = "f";
    let ir_top = format!("__{}__{}", module_name, dslx_top);

    // Step 2: ir2opt
    let ir2opt_output = std::process::Command::new(command_path)
        .arg("ir2opt")
        .arg(ir_path.to_str().unwrap())
        .arg("--top")
        .arg(&ir_top)
        .output()
        .unwrap();
    assert!(
        ir2opt_output.status.success(),
        "ir2opt failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&ir2opt_output.stdout),
        String::from_utf8_lossy(&ir2opt_output.stderr)
    );

    // Capture optimized IR and strip position data for deterministic golden.
    let optimized_ir = String::from_utf8_lossy(&ir2opt_output.stdout).to_string();
    let opt_ir_path = temp_dir.path().join("f.after_ir2opt.ir");
    std::fs::write(&opt_ir_path, &optimized_ir).unwrap();

    let ir_round_trip = std::process::Command::new(command_path)
        .arg("ir-round-trip")
        .arg(opt_ir_path.to_str().unwrap())
        .arg("--strip-pos-attrs")
        .arg("true")
        .output()
        .unwrap();
    assert!(
        ir_round_trip.status.success(),
        "ir-round-trip --strip-pos-attrs failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&ir_round_trip.stdout),
        String::from_utf8_lossy(&ir_round_trip.stderr)
    );
    let stripped_ir = String::from_utf8_lossy(&ir_round_trip.stdout).to_string();
    eprintln!(
        "[ir2opt optimized IR for __f__f (stripped)]:\n{}",
        stripped_ir
    );

    // Ensure golden directory exists, then compare/update via helper.
    let golden_rel = "tests/test_dslx_add_sub_opt_ir2gates_pipeline.golden.ir";
    let golden_dir = std::path::Path::new(golden_rel).parent().unwrap();
    let _ = std::fs::create_dir_all(golden_dir);
    compare_golden_text(&stripped_ir, golden_rel);
    // Overwrite IR with optimized IR for next step
    std::fs::write(&ir_path, &ir2opt_output.stdout).unwrap();

    log::info!(
        "optimized IR:\n{}",
        String::from_utf8_lossy(&ir2opt_output.stdout)
    );

    // Step 3: ir2gates (no --top argument)
    let ir2gates_output = std::process::Command::new(command_path)
        .arg("ir2gates")
        .arg(ir_path.to_str().unwrap())
        .output()
        .unwrap();
    assert!(
        ir2gates_output.status.success(),
        "ir2gates failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&ir2gates_output.stdout),
        String::from_utf8_lossy(&ir2gates_output.stderr)
    );
}

#[test]
fn test_ir2gates_quiet_json_output() {
    let _ = env_logger::builder().is_test(true).try_init();
    let dslx = "fn main(a: u32, b: u32) -> u32 { a & b }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("main.x");
    let ir_path = temp_dir.path().join("main.ir");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");

    // dslx2ir
    let dslx2ir_output = std::process::Command::new(command_path)
        .arg("dslx2ir")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .output()
        .unwrap();
    assert!(dslx2ir_output.status.success());
    std::fs::write(&ir_path, &dslx2ir_output.stdout).unwrap();

    // ir2gates --quiet
    let mut command = std::process::Command::new(command_path);
    command
        .arg("ir2gates")
        .arg(ir_path.to_str().unwrap())
        .arg("--quiet=true")
        .arg("--toggle-sample-count=32")
        .arg("--toggle-seed=42");
    // Pass through RUST_LOG if present
    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        command.env("RUST_LOG", rust_log);
    }
    let ir2gates_output = command.output().unwrap();

    log::debug!(
        "ir2gates stdout:\n{}",
        String::from_utf8_lossy(&ir2gates_output.stdout)
    );
    log::debug!(
        "ir2gates stderr:\n{}",
        String::from_utf8_lossy(&ir2gates_output.stderr)
    );

    assert!(ir2gates_output.status.success());

    let stdout = String::from_utf8_lossy(&ir2gates_output.stdout).to_string();
    // Try to parse as JSON
    let json: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("Output is not valid JSON");
    log::info!("json: {}", json);
    // Check standard stats
    assert_eq!(json["deepest_path"], 2);
    assert_eq!(json["fanout_histogram"].to_string(), "{\"1\":64}");
    assert_eq!(json["live_nodes"], 96);

    // Check expected values for this simple AND circuit
    let expected_toggle_stats: HashMap<&str, i32> = [
        ("gate_output_toggles", 363),
        ("gate_input_toggles", 980),
        ("primary_input_toggles", 980),
        ("primary_output_toggles", 363),
    ]
    .iter()
    .cloned()
    .collect();
    // Check that we covered all the keys in the toggle object.
    assert_eq!(
        json["toggle_stats"].as_object().unwrap().len(),
        expected_toggle_stats.len()
    );
    for (key, value) in expected_toggle_stats.iter() {
        assert_eq!(json["toggle_stats"][key].as_i64().unwrap(), *value as i64);
    }
}

#[test]
fn test_ir2gates_quiet_json_output_no_toggle() {
    let _ = env_logger::builder().is_test(true).try_init();
    let dslx = "fn main(a: u32, b: u32) -> u32 { a & b }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("main.x");
    let ir_path = temp_dir.path().join("main.ir");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");

    // dslx2ir
    let dslx2ir_output = std::process::Command::new(command_path)
        .arg("dslx2ir")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .output()
        .unwrap();
    assert!(dslx2ir_output.status.success());
    std::fs::write(&ir_path, &dslx2ir_output.stdout).unwrap();

    // ir2gates --quiet (no toggle flag)
    let mut command = std::process::Command::new(command_path);
    command
        .arg("ir2gates")
        .arg(ir_path.to_str().unwrap())
        .arg("--quiet=true");
    // Pass through RUST_LOG if present
    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        command.env("RUST_LOG", rust_log);
    }
    let ir2gates_output = command.output().unwrap();

    log::debug!(
        "ir2gates stdout:\n{}",
        String::from_utf8_lossy(&ir2gates_output.stdout)
    );
    log::debug!(
        "ir2gates stderr:\n{}",
        String::from_utf8_lossy(&ir2gates_output.stderr)
    );

    assert!(ir2gates_output.status.success());

    let stdout = String::from_utf8_lossy(&ir2gates_output.stdout);
    // Try to parse as JSON
    let json: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("Output is not valid JSON");
    log::info!("json: {}", json);
    // Check standard stats
    assert_eq!(json["deepest_path"], 2);
    assert_eq!(json["fanout_histogram"].to_string(), "{\"1\":64}");
    assert_eq!(json["live_nodes"], 96);
    assert_eq!(
        json["graph_logical_effort_worst_case_delay"].to_string(),
        "3.333333333333333"
    );
    // Check toggle stats fields exist and are null
    assert!(
        json["toggle_stats"].is_null(),
        "toggle_stats should be null"
    );
}

#[test]
fn test_ir2gates_output_json_file() {
    let _ = env_logger::builder().is_test(true).try_init();
    let dslx = "fn main(a: u32, b: u32) -> u32 { a & b }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("main.x");
    let ir_path = temp_dir.path().join("main.ir");
    let json_path = temp_dir.path().join("stats.json");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");

    // dslx2ir
    let dslx2ir_output = std::process::Command::new(command_path)
        .arg("dslx2ir")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .output()
        .unwrap();
    assert!(dslx2ir_output.status.success());
    std::fs::write(&ir_path, &dslx2ir_output.stdout).unwrap();

    // ir2gates with output_json
    let mut command = std::process::Command::new(command_path);
    command
        .arg("ir2gates")
        .arg(ir_path.to_str().unwrap())
        .arg(format!("--output_json={}", json_path.display()));
    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        command.env("RUST_LOG", rust_log);
    }
    let output = command.output().unwrap();
    assert!(output.status.success());
    // stdout should contain text, not JSON
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Deepest path"));
    // parse JSON file
    let json_content = std::fs::read_to_string(&json_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&json_content).unwrap();
    assert_eq!(json["deepest_path"], 2);
    assert_eq!(json["live_nodes"], 96);
}

// Test for ir-equiv subcommand using Solver
fn test_irequiv_subcommand_solver_equivalent(solver: &str) {
    let _ = env_logger::try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let lhs_ir = "package add_then_sub\nfn my_main(x: bits[32]) -> bits[32] {\n    add.2: bits[32] = add(x, x)\n    ret sub.3: bits[32] = sub(add.2, x)\n}";
    let rhs_ir = "package identity\nfn my_main(x: bits[32]) -> bits[32] {\n    ret identity.2: bits[32] = identity(x)\n}";
    // Write the IR files to the temp directory.
    let lhs_path = temp_dir.path().join("lhs.ir");
    std::fs::write(&lhs_path, lhs_ir).unwrap();
    let rhs_path = temp_dir.path().join("rhs.ir");
    std::fs::write(&rhs_path, rhs_ir).unwrap();

    // Write out toolchain configuration.
    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = "[toolchain]\n".to_string();
    // Unconditionally add tool_path for this test for now.
    let toolchain_toml_contents_with_path = add_tool_path_value(&toolchain_toml_contents);
    std::fs::write(&toolchain_toml_path, toolchain_toml_contents_with_path).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("ir-equiv")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--top")
        .arg("my_main")
        .arg(format!("--solver={}", solver))
        .output()
        .expect("xlsynth-driver should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    log::info!("stdout: {}", stdout);
    log::info!("stderr: {}", stderr);
    assert!(
        output.status.success(),
        "Solver ir-equiv should succeed; stderr: {}",
        stderr
    );
    assert!(
        stdout.contains("Solver proved equivalence"),
        "stdout: {}",
        stdout
    );
}

// Invoke IR node equivalence across solvers (native SMT backends and binaries).
fn test_irequiv_subcommand_solver_invoke_equivalent(solver: &str) {
    let _ = env_logger::try_init();
    let temp_dir = tempfile::tempdir().unwrap();

    // LHS uses invoke to call callee `g`; RHS inlines the add.
    let lhs_ir = r#"package p_invoke

fn g(x: bits[8]) -> bits[8] {
  ret add.1: bits[8] = add(x, x, id=1)
}

fn my_main(x: bits[8]) -> bits[8] {
  ret invoke.2: bits[8] = invoke(x, to_apply=g, id=2)
}
"#;

    let rhs_ir = r#"package p_inline

fn my_main(x: bits[8]) -> bits[8] {
  ret add.1: bits[8] = add(x, x, id=1)
}
"#;

    // Write the IR files to the temp directory.
    let lhs_path = temp_dir.path().join("lhs.ir");
    std::fs::write(&lhs_path, lhs_ir).unwrap();
    let rhs_path = temp_dir.path().join("rhs.ir");
    std::fs::write(&rhs_path, rhs_ir).unwrap();

    // Toolchain file so driver can run external tools when needed (not used here
    // but consistent with other tests).
    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml_path, toolchain_toml_contents).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("ir-equiv")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--top")
        .arg("my_main")
        .arg(format!("--solver={}", solver))
        .output()
        .expect("xlsynth-driver should run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    log::info!("stdout: {}", stdout);
    log::info!("stderr: {}", stderr);
    assert!(
        output.status.success(),
        "Invoke ir-equiv with solver {} should succeed; stderr: {}",
        solver,
        stderr
    );
    assert!(
        stdout.contains("Solver proved equivalence"),
        "stdout: {}",
        stdout
    );
}

// Parameterized wrappers using cfg_attr + test_case to avoid macros.
#[cfg_attr(feature="has-boolector", test_case::test_case("boolector"; "invoke_boolector"))]
#[cfg_attr(feature="has-bitwuzla", test_case::test_case("bitwuzla"; "invoke_bitwuzla"))]
#[cfg_attr(feature="with-z3-binary-test", test_case::test_case("z3-binary"; "invoke_z3_binary"))]
#[cfg_attr(feature="with-bitwuzla-binary-test", test_case::test_case("bitwuzla-binary"; "invoke_bitwuzla_binary"))]
#[cfg_attr(feature="with-boolector-binary-test", test_case::test_case("boolector-binary"; "invoke_boolector_binary"))]
#[test_case::test_case("toolchain"; "invoke_toolchain")]
fn test_irequiv_subcommand_solver_invoke_equivalent_param(solver: &str) {
    test_irequiv_subcommand_solver_invoke_equivalent(solver);
}

// Test for ir-equiv subcommand using Solver
fn test_irequiv_subcommand_solver_equivalent_with_fixed_implicit_activation(solver: &str) {
    let _ = env_logger::try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let lhs_ir = r#"
        package add_then_sub
        fn my_main(__token: token, __activation: bits[1], x: bits[32]) -> (token, bits[32]) {
            assert.2: token = assert(__token, __activation, message="activation should be false", label="activation_should_be_false")
            add.3: bits[32] = add(x, x)
            sub.4: bits[32] = sub(add.3, x)
            ret tuple.5: (token, bits[32]) = tuple(assert.2, sub.4)
        }
    "#;
    let rhs_ir = r#"
        package identity
        fn my_main(x: bits[32]) -> bits[32] {
            ret identity.2: bits[32] = identity(x)
        }
    "#;
    // Write the IR files to the temp directory.
    let lhs_path = temp_dir.path().join("lhs.ir");
    std::fs::write(&lhs_path, lhs_ir).unwrap();
    let rhs_path = temp_dir.path().join("rhs.ir");
    std::fs::write(&rhs_path, rhs_ir).unwrap();

    // Write out toolchain configuration.
    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = "[toolchain]\n".to_string();
    // Unconditionally add tool_path for this test for now.
    let toolchain_toml_contents_with_path = add_tool_path_value(&toolchain_toml_contents);
    std::fs::write(&toolchain_toml_path, toolchain_toml_contents_with_path).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("ir-equiv")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--top")
        .arg("my_main")
        .arg(format!("--solver={}", solver))
        .arg("--lhs_fixed_implicit_activation=true")
        .output()
        .expect("xlsynth-driver should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    log::info!("stdout: {}", stdout);
    log::info!("stderr: {}", stderr);
    assert!(
        output.status.success(),
        "Solver ir-equiv should succeed; \nstdout:\n {}\nstderr:\n {}",
        stdout,
        stderr
    );
    assert!(
        stdout.contains("Solver proved equivalence"),
        "stdout:\n {}\nstderr:\n {}",
        stdout,
        stderr
    );
}

/// Test for ir-equiv command with different IR top entry
/// points.
fn test_irequiv_subcommand_solver_different_top_entry_points(solver: &str) {
    let _ = env_logger::try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let lhs_ir = "package add_then_sub\nfn lhs_main(x: bits[32]) -> bits[32] {\n    add.2: bits[32] = add(x, x)\n    ret sub.3: bits[32] = sub(add.2, x)\n}";
    let rhs_ir = "package identity\nfn rhs_main(x: bits[32]) -> bits[32] {\n    ret identity.2: bits[32] = identity(x)\n}";
    // Write the IR files to the temp directory.
    let lhs_path = temp_dir.path().join("lhs.ir");
    std::fs::write(&lhs_path, lhs_ir).unwrap();
    let rhs_path = temp_dir.path().join("rhs.ir");
    std::fs::write(&rhs_path, rhs_ir).unwrap();

    // Write out toolchain configuration.
    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = "[toolchain]\n".to_string();
    std::fs::write(&toolchain_toml_path, toolchain_toml_contents).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("ir-equiv")
        .arg(format!("--solver={}", solver))
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--lhs_ir_top")
        .arg("lhs_main")
        .arg("--rhs_ir_top")
        .arg("rhs_main")
        .output()
        .expect("xlsynth-driver should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    log::info!("stdout: {}", stdout);
    log::info!("stderr: {}", stderr);
    assert!(output.status.success());
    assert!(stdout.contains("Solver proved equivalence"));
}

fn test_irequiv_subcommand_solver_output_bits_strategy(solver: &str) {
    let _ = env_logger::try_init();
    let temp_dir = tempfile::tempdir().unwrap();

    let lhs_dslx = r#"import std;
pub fn main(x: u8, y: u8) -> (u8, u8) { (x / y, x % y) }"#;
    let rhs_dslx = r#"import std;
pub fn main(x: u8, y: u8) -> (u8, u8) { if y == u8:0 { (all_ones!<u8>(), zero!<u8>()) } else { std::iterative_div_mod<u32:8, u32:8>(x, y) } }"#;

    let lhs_x = temp_dir.path().join("lhs.x");
    let rhs_x = temp_dir.path().join("rhs.x");
    std::fs::write(&lhs_x, lhs_dslx).unwrap();
    std::fs::write(&rhs_x, rhs_dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");

    let lhs_ir = temp_dir.path().join("lhs.ir");
    let out = Command::new(command_path)
        .arg("dslx2ir")
        .arg("--dslx_input_file")
        .arg(lhs_x.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .arg("--opt=true")
        .output()
        .unwrap();
    assert!(out.status.success());
    std::fs::write(&lhs_ir, &out.stdout).unwrap();

    let rhs_ir = temp_dir.path().join("rhs.ir");
    let out = Command::new(command_path)
        .arg("dslx2ir")
        .arg("--dslx_input_file")
        .arg(rhs_x.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .arg("--opt=true")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "dslx2ir failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::write(&rhs_ir, &out.stdout).unwrap();

    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml_path, toolchain_toml_contents).unwrap();

    // First, check that single-threaded solver proves equivalence for these IRs.
    let output_single = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("ir-equiv")
        .arg(format!("--solver={}", solver))
        .arg(lhs_ir.to_str().unwrap())
        .arg(rhs_ir.to_str().unwrap())
        .output()
        .expect("xlsynth-driver should succeed (single-threaded)");
    let stdout_single = String::from_utf8_lossy(&output_single.stdout);
    let stderr_single = String::from_utf8_lossy(&output_single.stderr);
    log::info!("stdout (single-threaded): {}", stdout_single);
    log::info!("stderr (single-threaded): {}", stderr_single);
    assert!(output_single.status.success());
    assert!(stdout_single.contains("Solver proved equivalence"));

    // Now check the parallelism strategy (output-bits).
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("ir-equiv")
        .arg(format!("--solver={}", solver))
        .arg(lhs_ir.to_str().unwrap())
        .arg(rhs_ir.to_str().unwrap())
        .arg("--parallelism-strategy=output-bits")
        .output()
        .expect("xlsynth-driver should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    log::info!("stdout: {}", stdout);
    log::info!("stderr: {}", stderr);
    println!("stdout: {}", stdout);
    println!("stderr: {}", stderr);
    println!("status: {}", output.status);
    assert!(output.status.success());
    assert!(stdout.contains("Solver proved equivalence"));
}

#[test]
fn test_ir_strip_pos_data_subcommand() {
    let _ = env_logger::try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let ir_path = temp_dir.path().join("with_pos.ir");

    let ir = "package pos_pkg\nfile_number 0 \"foo.x\"\n\n\
top fn main() -> bits[32] {\n  ret literal.1: bits[32] = literal(value=1, id=1, pos=[(0,0,0)])\n}\n";
    std::fs::write(&ir_path, ir).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(command_path)
        .arg("ir-strip-pos-data")
        .arg(ir_path.to_str().unwrap())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "ir-strip-pos-data failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let got = String::from_utf8_lossy(&output.stdout);
    let want = "package pos_pkg\n\n\
top fn main() -> bits[32] {\n  ret literal.1: bits[32] = literal(value=1, id=1)\n}\n";
    assert_eq!(got, want);
}

macro_rules! test_irequiv_subcommand_solver_base {
    ($solver:ident, $feature:expr, $choice:expr) => {
        paste::paste! {
            #[cfg(feature = $feature)]
            #[test]
            fn [<test_irequiv_subcommand_ $solver _different_top_entry_points>]() {
                test_irequiv_subcommand_solver_different_top_entry_points($choice);
            }
            #[cfg(feature = $feature)]
            #[test]
            fn [<test_irequiv_subcommand_ $solver _equivalent>]() {
                test_irequiv_subcommand_solver_equivalent($choice);
            }
            #[cfg(feature = $feature)]
            #[test]
            fn [<test_irequiv_subcommand_ $solver _output_bits_strategy>]() {
                test_irequiv_subcommand_solver_output_bits_strategy($choice);
            }
        }
    };
}

macro_rules! test_irequiv_subcommand_solver {
    ($solver:ident, $feature:expr, $choice:expr, true) => {
        paste::paste! {
            test_irequiv_subcommand_solver_base!($solver, $feature, $choice);
            #[cfg(feature = $feature)]
            #[test]
            fn [<test_irequiv_subcommand_ $solver _equivalent_with_fixed_implicit_activation>]() {
                test_irequiv_subcommand_solver_equivalent_with_fixed_implicit_activation($choice);
            }
        }
    };
    ($solver:ident, $feature:expr, $choice:expr, false) => {
        test_irequiv_subcommand_solver_base!($solver, $feature, $choice);
    };
}

test_irequiv_subcommand_solver!(boolector, "has-boolector", "boolector", true);
test_irequiv_subcommand_solver!(boolector_legacy, "has-boolector", "boolector-legacy", false);
test_irequiv_subcommand_solver!(bitwuzla, "has-bitwuzla", "bitwuzla", true);
test_irequiv_subcommand_solver!(
    boolector_binary,
    "with-boolector-binary-test",
    "boolector-binary",
    true
);
test_irequiv_subcommand_solver!(z3_binary, "with-z3-binary-test", "z3-binary", true);
test_irequiv_subcommand_solver!(
    bitwuzla_binary,
    "with-bitwuzla-binary-test",
    "bitwuzla-binary",
    true
);

#[test_case(true; "with_tool_path")]
#[test_case(false; "without_tool_path")]
fn test_toolchain_common_codegen_flags_resolve(use_tool_path: bool) {
    let _ = env_logger::builder().is_test(true).try_init();
    log::info!("test_toolchain_common_codegen_flags_resolve");
    let temp_dir = tempfile::tempdir().unwrap();

    // Write out toolchain configuration.
    let toolchain_toml = temp_dir.path().join("xlsynth-toolchain.toml");
    let mut toolchain_toml_contents = "[toolchain]\n\n[toolchain.codegen]\n".to_string();
    toolchain_toml_contents +=
        "gate_format = \"br_gate_buf gated_{output}(.in({input}), .out({output}))\"\n";
    toolchain_toml_contents += "assert_format = \"`BR_ASSERT({label}, {condition})\"\n";
    toolchain_toml_contents += "use_system_verilog = true\n";
    let toolchain_toml_contents = if use_tool_path {
        add_tool_path_value(&toolchain_toml_contents)
    } else {
        toolchain_toml_contents
    };
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    let dslx = r#"
fn main(pred: bool, x: u1) -> u1 {
    assert!(x == u1:1, "should_be_one");
    let gated = gate!(pred, x);
    gated
}
"#;
    let dslx_path = temp_dir.path().join("test.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let mut command = std::process::Command::new(command_path);
    command
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("dslx2pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--flop_inputs=true")
        .arg("--flop_outputs=true")
        .arg("--delay_model")
        .arg("unit");
    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        command.env("RUST_LOG", rust_log);
    }
    let output = command.output().unwrap();

    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();

    if !use_tool_path {
        // Compare against golden for runtime API path.
        compare_golden_sv(
            &stdout,
            "tests/test_toolchain_common_codegen_flags_resolve.golden.sv",
        );
    }
}

#[test_case(true, true; "with_tool_path_opt")]
#[test_case(true, false; "with_tool_path_noopt")]
#[test_case(false, true; "without_tool_path_opt")]
#[test_case(false, false; "without_tool_path_noopt")]
fn test_ir2pipeline_subcommand(use_tool_path: bool, optimize: bool) {
    let _ = env_logger::builder().is_test(true).try_init();

    // Simple identity IR package with a declared top function.
    let ir_text = "package sample\n\ntop fn my_main(x: bits[32] id=1) -> bits[32] {\n  ret x: bits[32] = param(name=x, id=1)\n}\n";

    let temp_dir = tempfile::tempdir().unwrap();
    let ir_path = temp_dir.path().join("sample.ir");
    std::fs::write(&ir_path, ir_text).unwrap();

    // Prepare (optional) toolchain.toml.
    let toolchain_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let mut toolchain_toml_contents = "[toolchain]\n".to_string();
    if use_tool_path {
        toolchain_toml_contents = add_tool_path_value(&toolchain_toml_contents);
    }
    std::fs::write(&toolchain_path, toolchain_toml_contents).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");

    let mut cmd = Command::new(command_path);

    // Supply --toolchain flag even for runtime path, mirroring other tests.
    cmd.arg("--toolchain").arg(toolchain_path.to_str().unwrap());

    cmd.arg("ir2pipeline")
        .arg(ir_path.to_str().unwrap())
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--top")
        .arg("my_main")
        .arg("--delay_model")
        .arg("unit")
        .arg("--flop_inputs=false")
        .arg("--flop_outputs=false");

    if optimize {
        cmd.arg("--opt=true");
    }

    // Pass through RUST_LOG if present.
    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        cmd.env("RUST_LOG", rust_log);
    }

    let output = cmd.output().unwrap();

    assert!(
        output.status.success(),
        "ir2pipeline failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    compare_golden_sv(
        &stdout,
        "tests/test_ir2pipeline_identity_pipeline.golden.sv",
    );
}

#[test]
fn test_ir2g8r_emits_all_outputs() {
    // This test checks that ir2g8r emits the pretty GateFn to stdout,
    // and writes both the .g8rbin and stats JSON files when requested.
    let _ = env_logger::try_init();
    // Use a simple DSLX function to generate IR
    let dslx = "fn main(a: u4, b: u4) -> u4 { a & b }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("main.x");
    let ir_path = temp_dir.path().join("main.ir");
    let g8rbin_path = temp_dir.path().join("main.g8rbin");
    let stats_path = temp_dir.path().join("main.stats.json");
    let ugv_path = temp_dir.path().join("main.ugv");
    std::fs::write(&dslx_path, dslx).unwrap();
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    // Step 1: dslx2ir
    let dslx2ir_output = std::process::Command::new(command_path)
        .arg("dslx2ir")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .output()
        .unwrap();
    assert!(
        dslx2ir_output.status.success(),
        "dslx2ir failed: {}",
        String::from_utf8_lossy(&dslx2ir_output.stderr)
    );
    std::fs::write(&ir_path, &dslx2ir_output.stdout).unwrap();
    // Step 2: ir2g8r
    let ir2g8r_output = std::process::Command::new(command_path)
        .arg("ir2g8r")
        .arg(ir_path.to_str().unwrap())
        .arg("--fold=true")
        .arg("--hash=true")
        .arg("--fraig=true")
        .arg("--bin-out")
        .arg(g8rbin_path.to_str().unwrap())
        .arg("--stats-out")
        .arg(stats_path.to_str().unwrap())
        .arg("--netlist-out")
        .arg(ugv_path.to_str().unwrap())
        .output()
        .unwrap();
    assert!(
        ir2g8r_output.status.success(),
        "ir2g8r failed: {}",
        String::from_utf8_lossy(&ir2g8r_output.stderr)
    );
    // Check stdout contains the pretty GateFn (should have 'fn __main__main(')
    let stdout = String::from_utf8_lossy(&ir2g8r_output.stdout);
    assert!(
        stdout.contains("fn __main__main("),
        "stdout did not contain pretty GateFn: {}",
        stdout
    );
    // Check .g8rbin file exists and is non-empty
    let g8rbin_data = std::fs::read(&g8rbin_path).expect(".g8rbin file not found");
    assert!(!g8rbin_data.is_empty(), ".g8rbin file is empty");
    // Check stats JSON file exists and contains expected keys
    let stats_json = std::fs::read_to_string(&stats_path).expect("stats JSON file not found");
    let stats: serde_json::Value =
        serde_json::from_str(&stats_json).expect("stats JSON not valid JSON");
    // Check for a few expected keys
    assert!(
        stats.get("live_nodes").is_some(),
        "stats missing live_nodes"
    );
    assert!(
        stats.get("deepest_path").is_some(),
        "stats missing deepest_path"
    );
    assert!(
        stats.get("fanout_histogram").is_some(),
        "stats missing fanout_histogram"
    );
    // Check .ugv file exists and is non-empty
    let ugv_data = std::fs::read(&ugv_path).expect(".ugv file not found");
    assert!(!ugv_data.is_empty(), ".ugv file is empty");
}

#[test]
fn test_g8r2v_add_clk_port_behavior() {
    let mut g8_builder = GateBuilder::new("testmod".to_string(), GateBuilderOptions::no_opt());
    let a_val = g8_builder.add_input("a".to_string(), 1);
    // Create a simple AND gate y = a & a, which is effectively y = a
    let y_val = g8_builder.add_and_binary(*a_val.get_lsb(0), *a_val.get_lsb(0));
    g8_builder.add_output("y".to_string(), AigBitVector::from_bit(y_val));
    let gate_fn = g8_builder.build();

    let temp_dir = tempfile::tempdir().unwrap();
    let g8r_path = temp_dir.path().join("testmod.g8r");
    std::fs::write(&g8r_path, gate_fn.to_string()).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("g8r2v")
        .arg(g8r_path.to_str().unwrap())
        .arg("--add-clk-port")
        .arg("clk")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let netlist = String::from_utf8_lossy(&output.stdout);

    let expected_netlist = "module testmod(\n  input wire clk,\n  input wire a,\n  output wire y\n);\n  wire G0;\n  wire G2;\n  assign G0 = 1'b0;\n  assign G2 = a & a;\n  assign y = G2;\nendmodule\n\n";
    assert_eq!(netlist, expected_netlist);
}

#[test]
fn test_g8r2v_module_name() {
    let mut g8_builder = GateBuilder::new("testmod".to_string(), GateBuilderOptions::no_opt());
    let a_val = g8_builder.add_input("a".to_string(), 1);
    let y_val = g8_builder.add_and_binary(*a_val.get_lsb(0), *a_val.get_lsb(0));
    g8_builder.add_output("y".to_string(), AigBitVector::from_bit(y_val));
    let gate_fn = g8_builder.build();

    let temp_dir = tempfile::tempdir().unwrap();
    let g8r_path = temp_dir.path().join("testmod.g8r");
    std::fs::write(&g8r_path, gate_fn.to_string()).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("g8r2v")
        .arg(g8r_path.to_str().unwrap())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "g8r2v failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("module testmod("),
        "netlist should use default module name: {}",
        stdout
    );

    // Override module name
    let output = Command::new(command_path)
        .arg("g8r2v")
        .arg(g8r_path.to_str().unwrap())
        .arg("--module-name=newmod")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "g8r2v failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("module newmod("),
        "netlist should use overridden module name: {}",
        stdout
    );
    assert!(
        !stdout.contains("module testmod("),
        "original module name should not appear when overridden: {}",
        stdout
    );
}

#[test]
fn test_g8r2v_flop_inputs_outputs() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut g8_builder = GateBuilder::new("my_flop_inv".to_string(), GateBuilderOptions::no_opt());
    let i_val = g8_builder.add_input("i".to_string(), 1);
    let o_val = g8_builder.add_not(*i_val.get_lsb(0));
    g8_builder.add_output("o".to_string(), AigBitVector::from_bit(o_val));
    let gate_fn = g8_builder.build();

    let temp_dir = tempfile::tempdir().unwrap();
    let g8r_path = temp_dir.path().join("my_flop_inv.g8r");
    std::fs::write(&g8r_path, gate_fn.to_string()).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("g8r2v")
        .arg(g8r_path.to_str().unwrap())
        .arg("--flop-inputs")
        .arg("--flop-outputs")
        .arg("--add-clk-port")
        .arg("clk")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let netlist = String::from_utf8_lossy(&output.stdout);

    let expected_output = "module my_flop_inv(\n  input wire clk,\n  input wire i,\n  output wire o\n);\n  reg p0_i;\n  wire o_comb;\n  reg p0_o;\n  wire G0;\n  assign G0 = 1'b0;\n  assign o_comb = ~p0_i;\n  always_ff @ (posedge clk) begin\n    p0_i <= i;\n    p0_o <= o_comb;\n  end\n  assign o = p0_o;\nendmodule\n\n";
    assert_eq!(netlist, expected_output);
}

#[test]
fn test_g8r2v_flop_inputs_only() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut g8_builder =
        GateBuilder::new("my_flop_inv_fi".to_string(), GateBuilderOptions::no_opt());
    let i_val = g8_builder.add_input("i".to_string(), 1);
    let o_val = g8_builder.add_not(*i_val.get_lsb(0));
    g8_builder.add_output("o".to_string(), AigBitVector::from_bit(o_val));
    let gate_fn = g8_builder.build();

    let temp_dir = tempfile::tempdir().unwrap();
    let g8r_path = temp_dir.path().join("my_flop_inv_fi.g8r");
    std::fs::write(&g8r_path, gate_fn.to_string()).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("g8r2v")
        .arg(g8r_path.to_str().unwrap())
        .arg("--flop-inputs")
        .arg("--add-clk-port")
        .arg("clk")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let netlist = String::from_utf8_lossy(&output.stdout);
    let expected_netlist = r#"module my_flop_inv_fi(
  input wire clk,
  input wire i,
  output wire o
);
  reg p0_i;
  wire G0;
  assign G0 = 1'b0;
  assign o = ~p0_i;
  always_ff @ (posedge clk) begin
    p0_i <= i;
  end
endmodule

"#;
    assert_eq!(netlist, expected_netlist);
}

#[test]
fn test_g8r2v_flop_outputs_only() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut g8_builder =
        GateBuilder::new("my_flop_inv_fo".to_string(), GateBuilderOptions::no_opt());
    let i_val = g8_builder.add_input("i".to_string(), 1);
    let o_val = g8_builder.add_not(*i_val.get_lsb(0));
    g8_builder.add_output("o".to_string(), AigBitVector::from_bit(o_val));
    let gate_fn = g8_builder.build();

    let temp_dir = tempfile::tempdir().unwrap();
    let g8r_path = temp_dir.path().join("my_flop_inv_fo.g8r");
    std::fs::write(&g8r_path, gate_fn.to_string()).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("g8r2v")
        .arg(g8r_path.to_str().unwrap())
        .arg("--flop-outputs")
        .arg("--add-clk-port")
        .arg("clk")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let netlist = String::from_utf8_lossy(&output.stdout);

    let expected_output = "module my_flop_inv_fo(\n  input wire clk,\n  input wire i,\n  output wire o\n);\n  wire o_comb;\n  reg p0_o;\n  wire G0;\n  assign G0 = 1'b0;\n  assign o_comb = ~i;\n  always_ff @ (posedge clk) begin\n    p0_o <= o_comb;\n  end\n  assign o = p0_o;\nendmodule\n\n";
    assert_eq!(netlist, expected_output);
}

#[test]
fn test_g8r2v_flop_requires_clk_port_error() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut g8_builder = GateBuilder::new("dummy".to_string(), GateBuilderOptions::no_opt());
    // Add a dummy input and output to make it a valid, though minimal, GateFn
    let i_val = g8_builder.add_input("i".to_string(), 1);
    g8_builder.add_output("o".to_string(), AigBitVector::from_bit(*i_val.get_lsb(0)));
    let gate_fn = g8_builder.build();

    let temp_dir = tempfile::tempdir().unwrap();
    let g8r_path = temp_dir.path().join("dummy.g8r");
    std::fs::write(&g8r_path, gate_fn.to_string()).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("g8r2v")
        .arg(g8r_path.to_str().unwrap())
        .arg("--flop-inputs") // Enable flopping
        // Missing --add-clk-port
        .output()
        .unwrap();

    assert!(!output.status.success(), "Command should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(
            "--add-clk-port <NAME> is required when --flop-inputs or --flop-outputs is used."
        ),
        "Stderr should contain the specific error message. Stderr: {}",
        stderr
    );
}

#[test]
fn test_g8r2v_flop_with_custom_clk_name() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut g8_builder = GateBuilder::new(
        "my_custom_clk_inv".to_string(),
        GateBuilderOptions::no_opt(),
    );
    let i_val = g8_builder.add_input("i".to_string(), 1);
    let o_val = g8_builder.add_not(*i_val.get_lsb(0));
    g8_builder.add_output("o".to_string(), AigBitVector::from_bit(o_val));
    let gate_fn = g8_builder.build();

    let temp_dir = tempfile::tempdir().unwrap();
    let g8r_path = temp_dir.path().join("my_custom_clk_inv.g8r");
    std::fs::write(&g8r_path, gate_fn.to_string()).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("g8r2v")
        .arg(g8r_path.to_str().unwrap())
        .arg("--flop-inputs")
        .arg("--add-clk-port")
        .arg("my_clk") // Custom clock name
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let netlist = String::from_utf8_lossy(&output.stdout);

    // Check for the custom clock name in the input port list
    let expected_module_def = "module my_custom_clk_inv(\n  input wire my_clk,\n  input wire i,
  output wire o
);";
    assert!(
        netlist.contains(expected_module_def),
        "Netlist module definition not as expected. Netlist:\n{}",
        netlist
    );

    // Check that the always_ff block also uses the custom clock name
    let expected_ff_block_sensitivity = "always_ff @ (posedge my_clk)";
    assert!(
        netlist.contains(expected_ff_block_sensitivity),
        "Netlist should use custom clock in always_ff sensitivity list. Netlist:\n{}",
        netlist
    );
}

#[test]
fn test_g8r2v_use_system_verilog() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut g8_builder = GateBuilder::new("my_sv_inv".to_string(), GateBuilderOptions::no_opt());
    let i_val = g8_builder.add_input("i".to_string(), 1);
    let o_val = g8_builder.add_not(*i_val.get_lsb(0));
    g8_builder.add_output("o".to_string(), AigBitVector::from_bit(o_val));
    let gate_fn = g8_builder.build();

    let temp_dir = tempfile::tempdir().unwrap();
    let g8r_path = temp_dir.path().join("my_sv_inv.g8r");
    std::fs::write(&g8r_path, gate_fn.to_string()).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("g8r2v")
        .arg(g8r_path.to_str().unwrap())
        .arg("--use-system-verilog")
        // No flopping, so clock is not strictly needed by emit_netlist for file type choice,
        // but if we were flopping, we would add --add-clk-port here.
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let netlist = String::from_utf8_lossy(&output.stdout);
    // At this stage, with VAST, the output might be identical for simple Verilog vs
    // SV unless specific SV features are used by emit_netlist.
    // We are mainly testing that the flag is accepted and the command runs.
    let expected_netlist = r#"module my_sv_inv(
  input wire i,
  output wire o
);
  wire G0;
  assign G0 = 1'b0;
  assign o = ~i;
endmodule

"#;
    // For now, we expect identical output. If emit_netlist starts emitting SV
    // specific syntax, this expectation will need to change.
    assert_eq!(netlist, expected_netlist);
}

// Checks that invariant assertions are only emitted when the
// --add_invariant_assertions flag is enabled for combinational codegen.
#[test_case(true; "with_invariant_assertions")]
#[test_case(false; "without_invariant_assertions")]
fn test_ir2combo_priority_sel_invariant(add_inv: bool) {
    let _ = env_logger::builder().is_test(true).try_init();
    log::info!("test_ir2combo_priority_sel_invariant (add_inv={})", add_inv);

    // A simple IR package that contains a priority_sel. The selector is the
    // constant one-hot value 0b01 so the optimizer can prove related
    // invariants.
    const PRIO_IR: &str = r#"package priority_sel_test

top fn my_main() -> bits[32] {
  literal.1: bits[2] = literal(value=1, id=1)
  literal.2: bits[32] = literal(value=11, id=2)
  literal.3: bits[32] = literal(value=22, id=3)
  literal.4: bits[32] = literal(value=0, id=4)
  ret priority_sel.5: bits[32] = priority_sel(literal.1, cases=[literal.2, literal.3], default=literal.4, id=5)
}
"#;

    // Write the IR to a temporary file.
    let temp_dir = tempfile::tempdir().unwrap();
    let ir_path = temp_dir.path().join("priority_sel.ir");
    std::fs::write(&ir_path, PRIO_IR).unwrap();

    // Write out toolchain.toml so we get the external tool path.
    let toolchain_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_path, toolchain_toml_contents).unwrap();

    // Prepare and run the ir2combo command.
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let mut cmd = std::process::Command::new(command_path);
    cmd.arg("--toolchain").arg(toolchain_path.to_str().unwrap());

    cmd.arg("ir2combo")
        .arg(ir_path.to_str().unwrap())
        .arg("--top")
        .arg("my_main")
        .arg("--delay_model")
        .arg("unit")
        .arg(format!("--add_invariant_assertions={}", add_inv))
        .arg("--use_system_verilog=true");

    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        cmd.env("RUST_LOG", rust_log);
    }

    let output = cmd.output().unwrap();
    assert!(
        output.status.success(),
        "ir2combo failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    log::debug!("ir2combo stdout:\n{}", stdout);
    log::debug!(
        "ir2combo stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let has_asserts = stdout.to_lowercase().contains("assert");
    if add_inv {
        assert!(has_asserts, "Expected invariant assertions to be present when --add_invariant_assertions=true, but none were found. stdout: {}", stdout);
    } else {
        assert!(!has_asserts, "Did not expect invariant assertions when --add_invariant_assertions=false, but some were found. stdout: {}", stdout);
    }
}

#[test]
fn test_ir_fn_to_block_smoke() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Minimal IR package with a trivial top.
    const SIMPLE_IR: &str = r#"package simple_pkg

top fn my_main(x: bits[8]) -> bits[8] {
  one: bits[8] = literal(value=1, id=1)
  ret add.2: bits[8] = add(x, one, id=2)
}

"#;
    let temp_dir = tempfile::tempdir().unwrap();
    let ir_path = temp_dir.path().join("simple.ir");
    std::fs::write(&ir_path, SIMPLE_IR).unwrap();

    let toolchain_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_path, toolchain_toml_contents).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_path.to_str().unwrap())
        .arg("ir-fn-to-block")
        .arg(ir_path.to_str().unwrap())
        .arg("--top")
        .arg("my_main")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "ir-fn-to-block failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    xlsynth_test_helpers::compare_golden_text(&stdout, "tests/test_ir_fn_to_block_smoke.golden.ir");
}

#[test]
fn test_ir_fn_to_block_golden() {
    let _ = env_logger::builder().is_test(true).try_init();

    const SIMPLE_IR: &str = r#"package simple_pkg

top fn my_main(x: bits[8]) -> bits[8] {
  one: bits[8] = literal(value=1, id=1)
  ret add.2: bits[8] = add(x, one, id=2)
}
"#;

    let temp_dir = tempfile::tempdir().unwrap();
    let ir_path = temp_dir.path().join("simple.ir");
    std::fs::write(&ir_path, SIMPLE_IR).unwrap();

    let toolchain_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_path, toolchain_toml_contents).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_path.to_str().unwrap())
        .arg("ir-fn-to-block")
        .arg(ir_path.to_str().unwrap())
        .arg("--top")
        .arg("my_main")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "ir-fn-to-block failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    xlsynth_test_helpers::compare_golden_text(&stdout, "tests/test_ir_fn_to_block_smoke.golden.ir");
}

#[test_case(true; "with_tool_path")]
#[test_case(false; "without_tool_path")]
fn test_dslx_g8r_stats_subcommand(use_tool_path: bool) {
    let _ = env_logger::builder().is_test(true).try_init();
    let dslx = "fn main(a: u1, b: u1) -> u1 { a & b }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("main.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let toolchain_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml = "[toolchain]\n";
    let toolchain_toml_contents = if use_tool_path {
        add_tool_path_value(&toolchain_toml)
    } else {
        toolchain_toml.to_string()
    };
    std::fs::write(&toolchain_path, toolchain_toml_contents).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let mut cmd = Command::new(command_path);
    cmd.arg("--toolchain")
        .arg(toolchain_path.to_str().unwrap())
        .arg("dslx-g8r-stats")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main");
    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        cmd.env("RUST_LOG", rust_log);
    }
    let output = cmd.output().unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("Output is not valid JSON");
    assert!(json.get("live_nodes").is_some(), "stats missing live_nodes");
    assert!(
        json.get("deepest_path").is_some(),
        "stats missing deepest_path"
    );
    assert!(
        json.get("fanout_histogram").is_some(),
        "stats missing fanout_histogram"
    );
}

#[test]
fn test_ir_fn_eval() {
    let ir = "package test\n\nfn add(a: bits[32], b: bits[32]) -> bits[32] {\n  ret add.1: bits[32] = add(a, b)\n}\n";
    let dir = tempfile::tempdir().unwrap();
    let ir_path = dir.path().join("add.ir");
    std::fs::write(&ir_path, ir).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("ir-fn-eval")
        .arg(ir_path.to_str().unwrap())
        .arg("add")
        .arg("(bits[32]:1, bits[32]:2)")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "bits[32]:3\n");
}

// Add tests for type inference v2 slice bounds behavior
#[test_case(true; "with_tool_path")]
#[test_case(false; "without_tool_path")]
fn test_tiv1_slice_oob_allows_compilation(use_tool_path: bool) {
    let _ = env_logger::builder().is_test(true).try_init();
    // DSLX attempting to slice starting at bit 32 of a 32-bit value.
    let dslx = "fn f(x: u32) -> u32 { x[32 +: u32] }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("f.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    // Prepare toolchain.toml
    let toolchain_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml = "[toolchain]\n";
    let toolchain_contents = if use_tool_path {
        add_tool_path_value(toolchain_toml)
    } else {
        toolchain_toml.to_string()
    };
    std::fs::write(&toolchain_path, toolchain_contents).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Run without --type_inference_v2 flag (tiv1).
    let output = std::process::Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_path.to_str().unwrap())
        .arg("dslx2ir")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("f")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "tiv1 compile should succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// TODO(cdleary): 2025-06-10 This only works when there is a tool path
/// available because the runtime APIs don't support specifying TIv2.
#[test_case(true; "with_tool_path")]
fn test_tiv2_slice_oob_is_error(use_tool_path: bool) {
    let _ = env_logger::builder().is_test(true).try_init();
    // Same DSLX code as above.
    let dslx = "fn f(x: u32) -> u32 { x[32 +: u32] }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("f.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    // Prepare toolchain.toml
    let toolchain_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml = "[toolchain]\n";
    let toolchain_contents = if use_tool_path {
        add_tool_path_value(toolchain_toml)
    } else {
        toolchain_toml.to_string()
    };
    std::fs::write(&toolchain_path, toolchain_contents).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Pass --type_inference_v2=true
    let output = std::process::Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_path.to_str().unwrap())
        .arg("dslx2ir")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("f")
        .arg("--type_inference_v2=true")
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "tiv2 compile should fail; stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // Optionally, ensure the stderr mentions slice or bounds.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.to_lowercase().contains("slice") || stderr.to_lowercase().contains("bound"),
        "expected slice/bound error message, got: {}",
        stderr
    );
}

#[test]
fn test_simulate_simple_pipeline() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Binary function so we can exercise multiple input ports.
    let dslx = "fn main(a: u32, b: u32) -> u32 { a + b }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("my_module.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("dslx2pipeline")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--delay_model")
        .arg("asap7")
        .arg("--flop_inputs=true")
        .arg("--flop_outputs=true")
        .arg("--input_valid_signal=input_valid")
        .arg("--output_valid_signal=output_valid")
        .arg("--reset=rst_n")
        .arg("--reset_active_low=true")
        .arg("--reset_asynchronous=false")
        .arg("--reset_data_path=true")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let pipeline_sv = String::from_utf8_lossy(&output.stdout).to_string();
    xlsynth_test_helpers::assert_valid_sv(&pipeline_sv);
    log::info!(
        "PIPELINE:\n{}",
        pipeline_sv.lines().take(8).collect::<Vec<_>>().join("\n")
    );

    let inputs = vec![("a", IrBits::u32(5)), ("b", IrBits::u32(6))];
    let expected = IrBits::u32(11);
    let vcd = xlsynth_test_helpers::simulate_pipeline_single_pulse(
        &pipeline_sv,
        "__my_module__main",
        &inputs,
        &expected,
        1,
    )
    .expect("simulation succeeds");
    assert!(vcd.contains("$var"));
}

#[test]
fn test_dslx_stitch_pipeline_signature_mismatch() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = r#"fn foo_cycle0(x: u32, y: u64) -> (u32, u64) { (x, y) }
fn foo_cycle1(a: u64, b: u32) -> u64 { a + b as u64 }"#;
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("foo.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("foo")
        .output()
        .unwrap();

    assert!(!output.status.success(), "command should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("does not match"),
        "unexpected stderr: {}",
        stderr
    );
}

#[test]
fn test_dslx_stitch_pipeline_parametric_stage_error() {
    let _ = env_logger::builder().is_test(true).try_init();

    // DSLX with a parametric pipeline stage function (generic parameter `N`).
    let dslx = "fn foo_cycle0(x: u32) -> u32 { x }\nfn foo_cycle1<N: u32>(x: u32) -> u32 { x }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("foo.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(command_path)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("foo")
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "command should fail, stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.to_lowercase().contains("parametric"),
        "stderr should mention parametric stage error, got: {}",
        stderr
    );
}

#[test]
fn test_dslx_stitch_pipeline_add_mul() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Two-stage pipeline: cycle0 computes x + y, cycle1 multiplies by z.
    let dslx = "fn add_mul_cycle0(x: u32, y: u32, z: u32) -> (u32, u32) { (x + y, z) }\nfn add_mul_cycle1(sum: u32, z: u32) -> u32 { sum * z }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("add_mul.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(command_path)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("add_mul")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let sv = String::from_utf8_lossy(&output.stdout).to_string();
    compare_golden_sv(&sv, "tests/test_dslx_stitch_pipeline_add_mul.golden.v");
}

#[test_case(true; "with_tool_path")]
#[test_case(false; "without_tool_path")]
fn test_dslx_stitch_pipeline_with_dslx_path_two_entries(use_tool_path: bool) {
    let _ = env_logger::try_init();
    log::info!("test_dslx_stitch_pipeline_with_dslx_path_two_entries");

    // Create a temporary directory hierarchy with two separate DSLX library paths.
    let temp_dir = tempfile::tempdir().unwrap();
    let a_dir = temp_dir.path().join("a");
    std::fs::create_dir(&a_dir).unwrap();
    let b_dir = temp_dir.path().join("b");
    std::fs::create_dir(&b_dir).unwrap();

    // Populate the libraries with simple constants so we can check that they are
    // resolved through the dslx_path setting.
    let a_path = a_dir.join("a.x");
    std::fs::write(&a_path, "pub const A: u32 = u32:42;").unwrap();
    let b_path = b_dir.join("b.x");
    std::fs::write(&b_path, "pub const B: u32 = u32:64;").unwrap();

    // Top-level DSLX file that stitches two pipeline stages together and uses the
    // imported constants from the two separate library paths. The final stage
    // computes A + B which should constant-fold to 0x6a (decimal 106) in the
    // generated SystemVerilog.
    let top_path = temp_dir.path().join("foo.x");
    std::fs::write(
        &top_path,
        "import a;\nimport b;\n\nfn foo_cycle0() -> (u32, u32) { (a::A, b::B) }\nfn foo_cycle1(x: u32, y: u32) -> u32 { x + y }",
    )
    .unwrap();

    // Build a toolchain.toml that points at the two library directories via
    // `dslx_path` so that the driver should make them visible during parsing and
    // type-checking.
    let toolchain_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let mut toolchain_toml = format!(
        r#"[toolchain]

[toolchain.dslx]
dslx_path = ["{}", "{}"]
"#,
        a_dir.to_str().unwrap(),
        b_dir.to_str().unwrap()
    );
    if use_tool_path {
        toolchain_toml = add_tool_path_value(&toolchain_toml);
    }
    std::fs::write(&toolchain_path, toolchain_toml).unwrap();

    // Invoke the driver.
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_path.to_str().unwrap())
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(top_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("foo")
        .output()
        .expect("xlsynth-driver should run");

    // Command should succeed.
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let sv = String::from_utf8_lossy(&output.stdout).to_string();
    xlsynth_test_helpers::assert_valid_sv(&sv);

    // Compare against golden in all modes.
    compare_golden_sv(
        &sv,
        "tests/test_dslx_stitch_pipeline_with_dslx_path_two_entries.golden.v",
    );
}

#[test]
fn test_dslx_stitch_pipeline_with_custom_stdlib_path() {
    let _ = env_logger::try_init();
    let temp_dir = tempfile::tempdir().unwrap();

    // Fake stdlib directory with a popcount that always returns 7.
    let stdlib_dir = temp_dir.path().join("fake_stdlib");
    std::fs::create_dir(&stdlib_dir).unwrap();
    let std_path = stdlib_dir.join("std.x");
    std::fs::write(&std_path, "pub fn popcount(x: u32) -> u32 { u32:7 }").unwrap();

    // DSLX file that relies on std::popcount and a two-stage pipeline.
    let dslx = "import std;\nfn foo_cycle0() -> u32 { std::popcount(u32:123) }\nfn foo_cycle1(x: u32) -> u32 { x }";
    let dslx_path = temp_dir.path().join("foo.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    // Toolchain.toml pointing at our fake stdlib.
    let toolchain_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml = format!(
        r#"[toolchain]

[toolchain.dslx]
dslx_stdlib_path = "{}""#,
        stdlib_dir.to_str().unwrap()
    );
    std::fs::write(&toolchain_path, toolchain_toml).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_path.to_str().unwrap())
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("foo")
        .output()
        .expect("driver run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let sv = String::from_utf8_lossy(&output.stdout).to_string();

    compare_golden_sv(
        &sv,
        "tests/test_dslx_stitch_pipeline_custom_stdlib.golden.v",
    );
}

#[test]
fn test_dslx_stitch_pipeline_with_valid() {
    let dslx =
        "fn foo_cycle0(x: u32) -> u32 { x + u32:1 }\nfn foo_cycle1(y: u32) -> u32 { y + u32:2 }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("foo.x");
    std::fs::write(&dslx_path, dslx).unwrap();
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("foo")
        .arg("--input_valid_signal=input_valid")
        .arg("--reset=rst")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();

    compare_golden_sv(
        &stdout,
        "tests/test_dslx_stitch_pipeline_with_valid.golden.v",
    );

    // Simulation (input_valid only, active-low rst)
    simulate_basic_valid_pipeline(&stdout, "foo", "input_valid", "rst", false);
}

#[test]
fn test_stitch_with_valid_custom_in_valid_reset() {
    let dslx =
        "fn foo_cycle0(x: u32) -> u32 { x + u32:1 }\nfn foo_cycle1(y: u32) -> u32 { y + u32:2 }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("foo.x");
    std::fs::write(&dslx_path, dslx).unwrap();
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("foo")
        .arg("--input_valid_signal=in_valid")
        .arg("--reset=rst")
        .arg("--reset_active_low=false")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    compare_golden_sv(
        &stdout,
        "tests/test_dslx_stitch_pipeline_with_valid_in_valid_rst.golden.v",
    );

    // Simulation active-high reset without output_valid
    simulate_basic_valid_pipeline(&stdout, "foo", "in_valid", "rst", false);
}

#[test]
fn test_stitch_with_valid_custom_in_valid_rst_n_active_low() {
    let dslx =
        "fn foo_cycle0(x: u32) -> u32 { x + u32:1 }\nfn foo_cycle1(y: u32) -> u32 { y + u32:2 }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("foo.x");
    std::fs::write(&dslx_path, dslx).unwrap();
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("foo")
        .arg("--input_valid_signal=in_valid")
        .arg("--reset=rst_n")
        .arg("--reset_active_low=true")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    compare_golden_sv(
        &stdout,
        "tests/test_dslx_stitch_pipeline_with_valid_in_valid_rst_n_active_low.golden.v",
    );

    // Simulation active-low reset with rst_n name
    simulate_basic_valid_pipeline(&stdout, "foo", "in_valid", "rst_n", true);
}

#[test]
fn test_stitch_with_valid_custom_in_and_out_valid() {
    let _ = env_logger::try_init();
    let dslx =
        "fn foo_cycle0(x: u32) -> u32 { x + u32:1 }\nfn foo_cycle1(y: u32) -> u32 { y + u32:2 }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("foo.x");
    std::fs::write(&dslx_path, dslx).unwrap();
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("foo")
        .arg("--input_valid_signal=in_valid")
        .arg("--output_valid_signal=out_valid")
        .arg("--reset=rst")
        .arg("--reset_active_low=false")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    compare_golden_sv(
        &stdout,
        "tests/test_dslx_stitch_pipeline_with_valid_in_and_out_valid.golden.v",
    );

    // Simulation check for custom valid signal names.
    let inputs = vec![("x", IrBits::u32(5))];
    let expected = IrBits::u32(8);
    let vcd = xlsynth_test_helpers::simulate_pipeline_single_pulse_custom(
        &stdout,
        "foo",
        &inputs,
        &expected,
        2,
        "in_valid",
        Some("out_valid"),
        "rst",
        false,
    )
    .expect("simulation succeeds");
    assert!(vcd.contains("$var"));
}

#[test]
fn test_stitch_with_valid_custom_in_and_out_valid_no_reset() {
    let _ = env_logger::try_init();
    let dslx =
        "fn foo_cycle0(x: u32) -> u32 { x + u32:1 }\nfn foo_cycle1(y: u32) -> u32 { y + u32:2 }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("foo.x");
    std::fs::write(&dslx_path, dslx).unwrap();
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("foo")
        .arg("--input_valid_signal=in_valid")
        .arg("--output_valid_signal=out_valid")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "driver failed; stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    compare_golden_sv(
        &stdout,
        "tests/test_dslx_stitch_pipeline_with_valid_in_and_out_valid_no_reset.golden.v",
    );
}

// Helper to simulate two-stage valid pipelines without output_valid.
fn simulate_basic_valid_pipeline(
    sv: &str,
    module_name: &str,
    input_valid: &str,
    reset: &str,
    reset_active_low: bool,
) {
    let inputs = vec![("x", IrBits::u32(5))];
    let expected = IrBits::u32(8);
    let vcd = xlsynth_test_helpers::simulate_pipeline_single_pulse_custom(
        sv,
        module_name,
        &inputs,
        &expected,
        2,
        input_valid,
        None,
        reset,
        reset_active_low,
    )
    .expect("simulation succeeds");
    assert!(vcd.contains("$var"));
}

const QUICKCHECK_DSLX: &'static str = r#"
fn f(x: u8) -> bool { x == x }
#[quickcheck] fn qc_success(x: u8) -> bool { f(x) }
#[quickcheck] fn qc_failure(x: u8) -> bool { x == u8:0 }
"#;

#[cfg_attr(feature="has-boolector", test_case("boolector", true; "boolector_success"))]
#[cfg_attr(feature="has-boolector", test_case("boolector", false; "boolector_failure"))]
#[cfg_attr(feature="has-bitwuzla", test_case("bitwuzla", true; "bitwuzla_success"))]
#[cfg_attr(feature="has-bitwuzla", test_case("bitwuzla", false; "bitwuzla_failure"))]
#[cfg_attr(feature="with-z3-binary-test", test_case("z3-binary", true; "z3_binary_success"))]
#[cfg_attr(feature="with-z3-binary-test", test_case("z3-binary", false; "z3_binary_failure"))]
#[cfg_attr(feature="with-bitwuzla-binary-test", test_case("bitwuzla-binary", true; "bitwuzla_bin_success"))]
#[cfg_attr(feature="with-bitwuzla-binary-test", test_case("bitwuzla-binary", false; "bitwuzla_bin_failure"))]
#[cfg_attr(feature="with-boolector-binary-test", test_case("boolector-binary", true; "boolector_bin_success"))]
#[cfg_attr(feature="with-boolector-binary-test", test_case("boolector-binary", false; "boolector_bin_failure"))]
#[test_case("toolchain", true; "toolchain_success")]
#[test_case("toolchain", false; "toolchain_failure")]
fn test_prove_quickcheck_solver_param(solver: &str, should_succeed: bool) {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let file_name = "qc.x";
    let dslx_path = temp_dir.path().join(file_name);
    std::fs::write(&dslx_path, QUICKCHECK_DSLX).unwrap();
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let mut cmd = Command::new(driver);
    let toolchain_toml = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_contents).unwrap();

    cmd.arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("prove-quickcheck")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--solver")
        .arg(solver)
        .arg("--test_filter")
        .arg(if should_succeed {
            ".*success"
        } else {
            ".*failure"
        });
    let output = cmd.output().unwrap();
    if should_succeed {
        assert!(
            output.status.success(),
            "Prove QC with solver {} should succeed. stderr: {}",
            solver,
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Success: All QuickChecks proved"));
    } else {
        assert!(
            !output.status.success(),
            "Prove QC with solver {} should fail (property false).",
            solver
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Failure: Some QuickChecks disproved"));
    }
}

// Parameterized version of the multi-quickcheck test that explicitly invokes
// each solver (external or SMT backend) to ensure the fallback to
// implicit-token mangled names works with each configured solver backend. Only
// success cases are tested.
#[test_case("toolchain"; "multi_qc_toolchain")]
#[cfg_attr(feature="has-boolector", test_case("boolector"; "multi_qc_boolector"))]
#[cfg_attr(feature="has-boolector", test_case("boolector-legacy"; "multi_qc_boolector_legacy"))]
#[cfg_attr(feature="has-bitwuzla", test_case("bitwuzla"; "multi_qc_bitwuzla"))]
#[cfg_attr(feature="with-z3-binary-test", test_case("z3-binary"; "multi_qc_z3_binary"))]
#[cfg_attr(feature="with-bitwuzla-binary-test", test_case("bitwuzla-binary"; "multi_qc_bitwuzla_binary"))]
#[cfg_attr(feature="with-boolector-binary-test", test_case("boolector-binary"; "multi_qc_boolector_binary"))]
fn test_prove_quickcheck_multiple_functions_with_assertions_solver(solver: &str) {
    let _ = env_logger::builder().is_test(true).try_init();
    const MULTI_QC_DSLX: &str = r#"
fn add1(x: u8) -> u8 { x + u8:1 }

#[quickcheck] fn qc_reflexive(x: u8) -> bool {
  assert!(x == x, "reflexive");
  true
}

#[quickcheck] fn qc_commutative(x: u8, y: u8) -> bool {
  assert!(x + y == y + x, "commutative");
  x + y == y + x
}

#[quickcheck] fn qc_increment_changes(x: u8) -> bool {
  let y = add1(x);
  assert!(y != x, "increment differs");
  (y > x) || (x == u8:255)
}
"#;

    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("multi_qc.x");
    std::fs::write(&dslx_path, MULTI_QC_DSLX).unwrap();

    let toolchain_toml = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_contents).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(driver)
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("prove-quickcheck")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--solver")
        .arg(solver)
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "prove-quickcheck should succeed for solver {}.\nstdout: {}\nstderr: {}",
        solver,
        stdout,
        stderr
    );
    assert!(
        stdout.contains("Success: All QuickChecks proved"),
        "stdout: {}",
        stdout
    );
}

/// Helper: skip test when `slang` executable is not available.
/// Returns true if slang is available and tests should proceed, false if tests
/// should be skipped.
fn should_skip_if_no_slang() -> bool {
    // Check if slang exists in PATH
    let slang_path = match which::which("slang") {
        Ok(path) => path,
        Err(_) => {
            println!("SKIPPED: `slang` executable not found in PATH");
            return true;
        }
    };

    // Also verify it's actually executable (same check as in
    // run_verilog_pipeline.rs)
    if let Ok(metadata) = std::fs::metadata(&slang_path) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = metadata.permissions();
            if perms.mode() & 0o111 == 0 {
                println!(
                    "SKIPPED: `slang` found at '{}' but not executable (permissions: {:o})",
                    slang_path.display(),
                    perms.mode()
                );
                return true;
            }
        }
    }

    // Quick test to see if slang can actually execute
    match std::process::Command::new(&slang_path)
        .arg("--version")
        .output()
    {
        Ok(output) => {
            if !output.status.success() {
                println!(
                    "SKIPPED: `slang` exists but failed to run: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                return true;
            }
        }
        Err(e) => {
            println!("SKIPPED: `slang` exists but failed to execute: {}", e);
            return true;
        }
    }

    false // Don't skip - slang is working
}

#[test]
fn test_run_verilog_pipeline_basic_add1() {
    let _ = env_logger::builder().is_test(true).try_init();

    if should_skip_if_no_slang() {
        return;
    }

    // Simple DSLX that increments by one.
    let dslx = "fn main(x: u32) -> u32 { x + u32:1 }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("add1.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    // First, generate the pipeline SV via dslx2pipeline.
    let pipeline_output = Command::new(driver)
        .arg("dslx2pipeline")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--flop_inputs=false")
        .arg("--flop_outputs=false")
        .arg("--delay_model")
        .arg("asap7")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .output()
        .expect("dslx2pipeline run");
    assert!(pipeline_output.status.success());
    let pipeline_sv = String::from_utf8(pipeline_output.stdout).unwrap();
    xlsynth_test_helpers::assert_valid_sv(&pipeline_sv);

    // Now run the pipeline simulation.
    let mut cmd = Command::new(driver);
    cmd.arg("run-verilog-pipeline")
        .arg("--latency")
        .arg("1")
        .arg("-") // Read SystemVerilog from stdin
        .arg("bits[32]:5")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn run-verilog-pipeline");
    {
        let stdin = child.stdin.as_mut().expect("get stdin");
        stdin.write_all(pipeline_sv.as_bytes()).unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "run-verilog-pipeline failed; retcode: {:?}\n  stdout: {:?}\n  stderr: {:?}",
        output.status.code().unwrap(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.trim().contains("out: bits[32]:6"),
        "unexpected stdout: {}",
        stdout
    );
}

#[test]
fn test_run_verilog_pipeline_wave_dump() {
    let _ = env_logger::builder().is_test(true).try_init();

    if should_skip_if_no_slang() {
        return;
    }

    // Simple DSLX that increments by one.
    let dslx = "fn main(x: u32) -> u32 { x + u32:1 }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("add1.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Generate pipeline SV via dslx2pipeline.
    let pipeline_output = Command::new(driver)
        .arg("dslx2pipeline")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--flop_inputs=false")
        .arg("--flop_outputs=false")
        .arg("--delay_model")
        .arg("asap7")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .output()
        .expect("dslx2pipeline run");
    assert!(pipeline_output.status.success());
    let pipeline_sv = String::from_utf8(pipeline_output.stdout).unwrap();
    xlsynth_test_helpers::assert_valid_sv(&pipeline_sv);

    // Path where waves should be dumped.
    let wave_path = temp_dir.path().join("dump.vcd");

    // Run the pipeline simulation with wave dumping.
    let mut cmd = Command::new(driver);
    cmd.arg("run-verilog-pipeline")
        .arg("--latency")
        .arg("1")
        .arg("--waves")
        .arg(wave_path.to_str().unwrap())
        .arg("-") // Read SystemVerilog from stdin
        .arg("bits[32]:5")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn run-verilog-pipeline");
    {
        let stdin = child.stdin.as_mut().expect("get stdin");
        stdin.write_all(pipeline_sv.as_bytes()).unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "run-verilog-pipeline failed; retcode: {:?}\n  stdout: {:?}\n  stderr: {:?}",
        output.status.code().unwrap(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    // Stdout should contain the expected mapping line.
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.trim().contains("out: bits[32]:6"));

    // Waves file should exist and contain VCD markers.
    assert!(wave_path.exists(), "waves file not created");
    let vcd_contents = std::fs::read_to_string(&wave_path).expect("read vcd");
    assert!(
        vcd_contents.contains("$var"),
        "wave file missing VCD var declarations"
    );
}

#[test]
fn test_run_verilog_pipeline_with_valid_signals() {
    let _ = env_logger::builder().is_test(true).try_init();

    if should_skip_if_no_slang() {
        return;
    }

    // DSLX with simple increment.
    let dslx = "fn main(x: u32) -> u32 { x + u32:1 }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("inc.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Generate pipeline with custom valid+reset signals.
    let pipeline_output = Command::new(driver)
        .arg("dslx2pipeline")
        .arg("--pipeline_stages")
        .arg("2")
        .arg("--delay_model")
        .arg("asap7")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .arg("--input_valid_signal=in_valid")
        .arg("--output_valid_signal=out_valid")
        .arg("--reset=rst")
        .arg("--reset_active_low=false")
        .output()
        .expect("dslx2pipeline run");
    assert!(pipeline_output.status.success(), "dslx2pipeline failed");
    let pipeline_sv = String::from_utf8(pipeline_output.stdout).unwrap();
    xlsynth_test_helpers::assert_valid_sv(&pipeline_sv);

    // Run pipeline via run-verilog-pipeline using the handshake signals.
    let mut cmd = Command::new(driver);
    cmd.arg("run-verilog-pipeline")
        .arg("--input_valid_signal=in_valid")
        .arg("--output_valid_signal=out_valid")
        .arg("--reset=rst")
        .arg("--reset_active_low=false")
        .arg("-") // Read SystemVerilog from stdin
        .arg("bits[32]:5") // input value
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn run-verilog-pipeline");
    {
        let stdin = child.stdin.as_mut().expect("stdin");
        stdin.write_all(pipeline_sv.as_bytes()).unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "run-verilog-pipeline failed; retcode: {:?}\n  stdout: {:?}\n  stderr: {:?}",
        output.status.code().unwrap(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.trim().contains("out: bits[32]:6"),
        "unexpected stdout: {}",
        stdout
    );
}

#[test]
fn test_run_verilog_pipeline_with_valid_signals_and_flops() {
    let _ = env_logger::builder().is_test(true).try_init();

    if should_skip_if_no_slang() {
        return;
    }

    // DSLX with simple increment.
    let dslx = "fn main(x: u32) -> u32 { x + u32:1 }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("inc.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Generate pipeline with handshake signals and explicit IO flops.
    let pipeline_output = Command::new(driver)
        .arg("dslx2pipeline")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--delay_model")
        .arg("asap7")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .arg("--input_valid_signal=in_valid")
        .arg("--output_valid_signal=out_valid")
        .arg("--reset=rst")
        .arg("--reset_active_low=false")
        .arg("--flop_inputs=true")
        .arg("--flop_outputs=true")
        .output()
        .expect("dslx2pipeline run");
    assert!(pipeline_output.status.success(), "dslx2pipeline failed");
    let pipeline_sv = String::from_utf8(pipeline_output.stdout).unwrap();
    xlsynth_test_helpers::assert_valid_sv(&pipeline_sv);

    // Run pipeline via run-verilog-pipeline using the handshake signals.
    let mut cmd = Command::new(driver);
    cmd.arg("run-verilog-pipeline")
        .arg("--input_valid_signal=in_valid")
        .arg("--output_valid_signal=out_valid")
        .arg("--reset=rst")
        .arg("--reset_active_low=false")
        .arg("-") // Read SystemVerilog from stdin
        .arg("bits[32]:5") // input value
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn run-verilog-pipeline");
    {
        let stdin = child.stdin.as_mut().expect("stdin");
        stdin.write_all(pipeline_sv.as_bytes()).unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "run-verilog-pipeline failed; retcode: {:?}\n  stdout: {:?}\n  stderr: {:?}",
        output.status.code().unwrap(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.trim().contains("out: bits[32]:6"),
        "unexpected stdout: {}",
        stdout
    );
}

#[cfg_attr(feature="has-boolector", test_case("boolector", true; "dslx_equiv_boolector_equivalent"))]
#[cfg_attr(feature="has-boolector", test_case("boolector", false; "dslx_equiv_boolector_nonequivalent"))]
#[cfg_attr(feature="has-bitwuzla", test_case("bitwuzla", true; "dslx_equiv_bitwuzla_equivalent"))]
#[cfg_attr(feature="has-bitwuzla", test_case("bitwuzla", false; "dslx_equiv_bitwuzla_nonequivalent"))]
#[cfg_attr(feature="with-z3-binary-test", test_case("z3-binary", true; "dslx_equiv_z3_binary_equivalent"))]
#[cfg_attr(feature="with-z3-binary-test", test_case("z3-binary", false; "dslx_equiv_z3_binary_nonequivalent"))]
#[cfg_attr(feature="with-bitwuzla-binary-test", test_case("bitwuzla-binary", true; "dslx_equiv_bitwuzla_binary_equivalent"))]
#[cfg_attr(feature="with-bitwuzla-binary-test", test_case("bitwuzla-binary", false; "dslx_equiv_bitwuzla_binary_nonequivalent"))]
#[cfg_attr(feature="with-boolector-binary-test", test_case("boolector-binary", true; "dslx_equiv_boolector_binary_equivalent"))]
#[cfg_attr(feature="with-boolector-binary-test", test_case("boolector-binary", false; "dslx_equiv_boolector_binary_nonequivalent"))]
#[test_case("toolchain", true; "dslx_equiv_toolchain_equivalent")]
#[test_case("toolchain", false; "dslx_equiv_toolchain_nonequivalent")]
fn test_dslx_equiv_solver_param(solver: &str, should_succeed: bool) {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();

    // LHS DSLX (same in both success/failure cases): algebraically (x + x) - x == x
    // modulo 2^32.
    let lhs_dslx = "fn main(x: u32) -> u32 { (x + x) - x }";
    // RHS varies depending on whether we want equivalence.
    let rhs_dslx_equiv = "fn main(x: u32) -> u32 { x }";
    let rhs_dslx_nonequiv = "fn main(x: u32) -> u32 { x + u32:1 }"; // differs by +1.

    let lhs_path = temp_dir.path().join("lhs.x");
    let rhs_path = temp_dir.path().join("rhs.x");
    std::fs::write(&lhs_path, lhs_dslx).unwrap();
    std::fs::write(
        &rhs_path,
        if should_succeed {
            rhs_dslx_equiv
        } else {
            rhs_dslx_nonequiv
        },
    )
    .unwrap();

    // Toolchain config so the toolchain solver (and default path when unspecified)
    // works.
    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let mut toolchain_toml_contents = "[toolchain]\n".to_string();
    // Always add tool_path so the default / toolchain solver path is available.
    toolchain_toml_contents = add_tool_path_value(&toolchain_toml_contents);
    std::fs::write(&toolchain_toml_path, toolchain_toml_contents).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("dslx-equiv")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .arg("--solver")
        .arg(solver)
        .output()
        .expect("dslx-equiv invocation should run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if should_succeed {
        assert!(
            output.status.success(),
            "dslx-equiv should report equivalence. stdout: {}\nstderr: {}",
            stdout,
            stderr
        );
        assert!(
            stdout.contains("success"),
            "expected success marker in stdout: {}",
            stdout
        );
    } else {
        assert!(
            !output.status.success(),
            "dslx-equiv should report non-equivalence. stdout: {}\nstderr: {}",
            stdout,
            stderr
        );
        assert!(
            stderr.contains("failure") || stdout.contains("failure"),
            "expected failure marker. stdout: {} stderr: {}",
            stdout,
            stderr
        );
    }
}

#[cfg_attr(feature="has-boolector", test_case("boolector", true; "dslx_equiv_diff_tops_boolector_equiv"))]
#[cfg_attr(feature="has-boolector", test_case("boolector", false; "dslx_equiv_diff_tops_boolector_nonequiv"))]
#[cfg_attr(feature="has-bitwuzla", test_case("bitwuzla", true; "dslx_equiv_diff_tops_bitwuzla_equiv"))]
#[cfg_attr(feature="has-bitwuzla", test_case("bitwuzla", false; "dslx_equiv_diff_tops_bitwuzla_nonequiv"))]
#[cfg_attr(feature="with-z3-binary-test", test_case("z3-binary", true; "dslx_equiv_diff_tops_z3_equiv"))]
#[cfg_attr(feature="with-z3-binary-test", test_case("z3-binary", false; "dslx_equiv_diff_tops_z3_nonequiv"))]
#[cfg_attr(feature="with-bitwuzla-binary-test", test_case("bitwuzla-binary", true; "dslx_equiv_diff_tops_bitwuzla_bin_equiv"))]
#[cfg_attr(feature="with-bitwuzla-binary-test", test_case("bitwuzla-binary", false; "dslx_equiv_diff_tops_bitwuzla_bin_nonequiv"))]
#[cfg_attr(feature="with-boolector-binary-test", test_case("boolector-binary", true; "dslx_equiv_diff_tops_boolector_bin_equiv"))]
#[cfg_attr(feature="with-boolector-binary-test", test_case("boolector-binary", false; "dslx_equiv_diff_tops_boolector_bin_nonequiv"))]
#[test_case("toolchain", true; "dslx_equiv_diff_tops_toolchain_equiv")]
#[test_case("toolchain", false; "dslx_equiv_diff_tops_toolchain_nonequiv")]
fn test_dslx_equiv_solver_param_different_tops(solver: &str, should_succeed: bool) {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();

    let lhs_dslx = "fn alpha(x: u32) -> u32 { (x + x) - x }"; // simplifies to x
    let rhs_dslx_equiv = "fn beta(x: u32) -> u32 { x }";
    let rhs_dslx_nonequiv = "fn beta(x: u32) -> u32 { x + u32:1 }";

    let lhs_path = temp_dir.path().join("lhs.x");
    let rhs_path = temp_dir.path().join("rhs.x");
    std::fs::write(&lhs_path, lhs_dslx).unwrap();
    std::fs::write(
        &rhs_path,
        if should_succeed {
            rhs_dslx_equiv
        } else {
            rhs_dslx_nonequiv
        },
    )
    .unwrap();

    // Toolchain TOML with tool_path so external solver binaries (if configured)
    // still ok.
    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let mut toolchain_toml_contents = "[toolchain]\n".to_string();
    toolchain_toml_contents = add_tool_path_value(&toolchain_toml_contents);
    std::fs::write(&toolchain_toml_path, toolchain_toml_contents).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("dslx-equiv")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--lhs_dslx_top")
        .arg("alpha")
        .arg("--rhs_dslx_top")
        .arg("beta")
        .arg("--solver")
        .arg(solver)
        .output()
        .expect("dslx-equiv invocation should run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if should_succeed {
        assert!(
            output.status.success(),
            "dslx-equiv (different tops) should report equivalence. stdout: {}\nstderr: {}",
            stdout,
            stderr
        );
        assert!(
            stdout.contains("success"),
            "expected success marker in stdout: {}",
            stdout
        );
    } else {
        assert!(
            !output.status.success(),
            "dslx-equiv (different tops) should report non-equivalence. stdout: {}\nstderr: {}",
            stdout,
            stderr
        );
        assert!(
            stderr.contains("failure") || stdout.contains("failure"),
            "expected failure marker. stdout: {} stderr: {}",
            stdout,
            stderr
        );
    }
}

#[test]
fn test_run_verilog_pipeline_with_valid_signals_and_output_flops_only() {
    let _ = env_logger::builder().is_test(true).try_init();

    if should_skip_if_no_slang() {
        return;
    }

    let dslx = "fn main(x: u32) -> u32 { x + u32:1 }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("inc_outflop.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    let pipeline_output = Command::new(driver)
        .arg("dslx2pipeline")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--delay_model")
        .arg("asap7")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .arg("--input_valid_signal=in_valid")
        .arg("--output_valid_signal=out_valid")
        .arg("--reset=rst")
        .arg("--reset_active_low=false")
        .arg("--flop_outputs=true")
        .output()
        .expect("dslx2pipeline run");
    assert!(pipeline_output.status.success(), "dslx2pipeline failed");
    let pipeline_sv = String::from_utf8(pipeline_output.stdout).unwrap();
    xlsynth_test_helpers::assert_valid_sv(&pipeline_sv);

    let mut cmd = Command::new(driver);
    cmd.arg("run-verilog-pipeline")
        .arg("--input_valid_signal=in_valid")
        .arg("--output_valid_signal=out_valid")
        .arg("--reset=rst")
        .arg("--reset_active_low=false")
        .arg("-")
        .arg("bits[32]:5")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn run-verilog-pipeline");
    {
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(pipeline_sv.as_bytes())
            .unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "run-verilog-pipeline failed; stdout: {:?} stderr: {:?}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.trim().contains("out: bits[32]:6"),
        "unexpected stdout: {}",
        stdout
    );
}

#[test]
fn test_run_verilog_pipeline_with_valid_signals_and_input_flops_only() {
    let _ = env_logger::builder().is_test(true).try_init();

    if should_skip_if_no_slang() {
        return;
    }

    let dslx = "fn main(x: u32) -> u32 { x + u32:1 }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("inc_inflop.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Generate pipeline with input flops only.
    let pipeline_output = Command::new(driver)
        .arg("dslx2pipeline")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--delay_model")
        .arg("asap7")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .arg("--input_valid_signal=in_valid")
        .arg("--output_valid_signal=out_valid")
        .arg("--reset=rst")
        .arg("--reset_active_low=false")
        .arg("--flop_inputs=true")
        .output()
        .expect("dslx2pipeline run");
    assert!(pipeline_output.status.success(), "dslx2pipeline failed");
    let pipeline_sv = String::from_utf8(pipeline_output.stdout).unwrap();
    xlsynth_test_helpers::assert_valid_sv(&pipeline_sv);

    // Run simulation.
    let mut cmd = Command::new(driver);
    cmd.arg("run-verilog-pipeline")
        .arg("--input_valid_signal=in_valid")
        .arg("--output_valid_signal=out_valid")
        .arg("--reset=rst")
        .arg("--reset_active_low=false")
        .arg("-")
        .arg("bits[32]:5")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn run-verilog-pipeline");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(pipeline_sv.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "run-verilog-pipeline failed; stdout: {:?} stderr: {:?}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.trim().contains("out: bits[32]:6"),
        "unexpected stdout: {}",
        stdout
    );
}

#[test]
fn test_dslx_stitch_pipeline_no_flops() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = "fn noflop_cycle0(x: u32) -> u32 { x + u32:1 }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("noflop.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(command_path)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("noflop")
        .arg("--flop_inputs=false")
        .arg("--flop_outputs=false")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let sv = String::from_utf8_lossy(&output.stdout);
    xlsynth_test_helpers::assert_valid_sv(&sv);
    assert!(
        !sv.contains("reg p0_"),
        "No pipeline flop registers expected, but found some in generated SV"
    );
}

#[test]
fn test_dslx_stitch_pipeline_flop_inputs_only() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = "fn inputflop_cycle0(x: u32) -> u32 { x + u32:1 }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("inputflop.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(command_path)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("inputflop")
        .arg("--flop_inputs=true")
        .arg("--flop_outputs=false")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let sv = String::from_utf8_lossy(&output.stdout);
    xlsynth_test_helpers::assert_valid_sv(&sv);
    assert!(
        sv.contains("p0_x"),
        "Expected input flop register p0_x not found"
    );
    assert!(
        !sv.contains("reg p1_"),
        "Unexpected second stage flop registers found when flop_outputs disabled"
    );
}

#[test]
fn test_dslx_stitch_pipeline_flop_outputs_only() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = "fn outputflop_cycle0(x: u32) -> u32 { x + u32:1 }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("outputflop.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(command_path)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("outputflop")
        .arg("--flop_inputs=false")
        .arg("--flop_outputs=true")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let sv = String::from_utf8_lossy(&output.stdout);
    xlsynth_test_helpers::assert_valid_sv(&sv);
    assert!(
        sv.contains("p0_out"),
        "Expected output flop register p0_out not found"
    );
    assert!(
        !sv.contains("reg p0_x"),
        "Input flop p0_x register should not be present when flop_inputs disabled"
    );
}

#[cfg(feature = "with-z3-binary-test")]
#[test]
fn test_dslx_equiv_assume_enum_in_bound_swapped_default() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Enum has wider underlying width; only A/B are valid, others are out of
    // domain.
    let lhs = r#"
        enum E : u3 { A = 0, B = 1 }
        fn f(e: E) -> u32 {
          match e {
            E::A => u32:0,
            _    => u32:1,
          }
        }
    "#;
    let rhs = r#"
        enum E : u3 { A = 0, B = 1 }
        fn f(e: E) -> u32 {
          match e {
            E::B => u32:1,
            _    => u32:0,
          }
        }
    "#;

    let tmp = tempfile::tempdir().unwrap();
    let lhs_path = tmp.path().join("lhs.x");
    let rhs_path = tmp.path().join("rhs.x");
    std::fs::write(&lhs_path, lhs).unwrap();
    std::fs::write(&rhs_path, rhs).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Default behavior (assume-enum-in-bound enabled) → should be equivalent.
    let out_default = std::process::Command::new(driver)
        .args([
            "dslx-equiv",
            lhs_path.to_str().unwrap(),
            rhs_path.to_str().unwrap(),
            "--dslx_top",
            "f",
            "--solver",
            "z3-binary",
        ])
        .output()
        .unwrap();
    assert!(
        out_default.status.success(),
        "Expected equivalence with default enum-bound assumption. stdout: {} stderr: {}",
        String::from_utf8_lossy(&out_default.stdout),
        String::from_utf8_lossy(&out_default.stderr)
    );

    // With enum-in-bound assumption explicitly set to true → should be equivalent.
    let out_with_flag = std::process::Command::new(driver)
        .args([
            "dslx-equiv",
            lhs_path.to_str().unwrap(),
            rhs_path.to_str().unwrap(),
            "--dslx_top",
            "f",
            "--solver",
            "z3-binary",
            "--assume-enum-in-bound",
            "true",
        ])
        .output()
        .unwrap();
    assert!(
        out_with_flag.status.success(),
        "Expected equivalence with enum-bound assumption. stdout: {} stderr: {}",
        String::from_utf8_lossy(&out_with_flag.stdout),
        String::from_utf8_lossy(&out_with_flag.stderr)
    );

    // With enum-in-bound assumption explicitly disabled → must be inequivalent.
    let out_without_flag = std::process::Command::new(driver)
        .args([
            "dslx-equiv",
            lhs_path.to_str().unwrap(),
            rhs_path.to_str().unwrap(),
            "--dslx_top",
            "f",
            "--solver",
            "z3-binary",
            "--assume-enum-in-bound",
            "false",
        ])
        .output()
        .unwrap();
    assert!(
        !out_without_flag.status.success(),
        "Expected inequivalence without enum-bound assumption. stdout: {} stderr: {}",
        String::from_utf8_lossy(&out_without_flag.stdout),
        String::from_utf8_lossy(&out_without_flag.stderr)
    );
}

// Helper for enum-in-bound assumption matrix test
fn run_dslx_equiv_enum_in_bound_for_solver(solver: &str, expect_supported: bool) {
    let _ = env_logger::builder().is_test(true).try_init();

    let lhs = r#"
        enum E : u3 { A = 0, B = 1 }
        fn f(e: E) -> u32 {
          match e {
            E::A => u32:0,
            _    => u32:1,
          }
        }
    "#;
    let rhs = r#"
        enum E : u3 { A = 0, B = 1 }
        fn f(e: E) -> u32 {
          match e {
            E::B => u32:1,
            _    => u32:0,
          }
        }
    "#;

    let tmp = tempfile::tempdir().unwrap();
    let lhs_path = tmp.path().join("lhs.x");
    let rhs_path = tmp.path().join("rhs.x");
    std::fs::write(&lhs_path, lhs).unwrap();
    std::fs::write(&rhs_path, rhs).unwrap();

    // Provide a toolchain.toml so toolchain-mode is available (and we can catch
    // unsupported behavior).
    let toolchain_toml = tmp.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    // 1) Without assumption → inequivalent for all solvers.
    let out_no_flag = std::process::Command::new(driver)
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("dslx-equiv")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("f")
        .arg("--solver")
        .arg(solver)
        .arg("--assume-enum-in-bound")
        .arg("false")
        .output()
        .unwrap();
    assert!(
        !out_no_flag.status.success(),
        "Expected inequivalence without enum-bound assumption (solver={}). stdout: {} stderr: {}",
        solver,
        String::from_utf8_lossy(&out_no_flag.stdout),
        String::from_utf8_lossy(&out_no_flag.stderr)
    );

    // 2) With assumption → native SMT solvers succeed; toolchain should fail
    //    (unsupported).
    let out_with_flag = std::process::Command::new(driver)
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("dslx-equiv")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("f")
        .arg("--solver")
        .arg(solver)
        .arg("--assume-enum-in-bound")
        .arg("true")
        .output()
        .unwrap();

    if expect_supported {
        assert!(
            out_with_flag.status.success(),
            "Expected equivalence with enum-bound assumption (solver={}). stdout: {} stderr: {}",
            solver,
            String::from_utf8_lossy(&out_with_flag.stdout),
            String::from_utf8_lossy(&out_with_flag.stderr)
        );
    } else {
        assert!(
            !out_with_flag.status.success(),
            "Expected unsupported/failure for toolchain solver with enum-bound assumption (solver={}). stdout: {} stderr: {}",
            solver,
            String::from_utf8_lossy(&out_with_flag.stdout),
            String::from_utf8_lossy(&out_with_flag.stderr)
        );
    }
}

#[cfg_attr(feature="has-boolector", test_case::test_case("boolector", true; "enum_in_bound_boolector"))]
#[cfg_attr(feature="has-boolector", test_case::test_case("boolector-legacy", false; "enum_in_bound_boolector_legacy_unsupported"))]
#[cfg_attr(feature="has-bitwuzla", test_case::test_case("bitwuzla", true; "enum_in_bound_bitwuzla"))]
#[cfg_attr(feature="with-z3-binary-test", test_case::test_case("z3-binary", true; "enum_in_bound_z3_binary"))]
#[cfg_attr(feature="with-bitwuzla-binary-test", test_case::test_case("bitwuzla-binary", true; "enum_in_bound_bitwuzla_binary"))]
#[cfg_attr(feature="with-boolector-binary-test", test_case::test_case("boolector-binary", true; "enum_in_bound_boolector_binary"))]
#[test_case::test_case("toolchain", false; "enum_in_bound_toolchain_unsupported")]
fn test_dslx_equiv_enum_in_bound_solver_matrix(solver: &str, expect_supported: bool) {
    run_dslx_equiv_enum_in_bound_for_solver(solver, expect_supported);
}

// Checks that invariant assertions are emitted (or not) for the
// dslx-stitch-pipeline subcommand when the `add_invariant_assertions` option is
// toggled via the toolchain configuration. We use a simple two-stage pipeline
// where the first stage contains a `priority_sel` so that the optimiser can
// prove the associated invariants and gemit the assertion.
#[test_case(true; "with_invariant_assertions")]
#[test_case(false; "without_invariant_assertions")]
fn test_dslx_stitch_pipeline_priority_sel_invariant(add_inv: bool) {
    let _ = env_logger::builder().is_test(true).try_init();

    // Two-stage pipeline: cycle0 performs a priority-select using a *dynamic*
    // selector provided as an input to the pipeline. This prevents
    // constant-propagation from eliminating the priority_sel node (and
    // therefore its assertion).
    let dslx = r#"fn f_cycle0(x0: u3, x4: u6) -> u6 {
    {
        let x6: u6 = match x0 {
            u3:0x3 | u3:0x4 => x4,
            u3:0x2 => u6:0x2a,
            _ => u6:0x8,
        };
        x6
    }
}

fn f_cycle1(x: u6) -> u6 {
    x
}"#;

    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("prio.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    // Write out toolchain.toml with (1) external tool path and (2) the invariant
    // flag.
    let toolchain_toml_base = if add_inv {
        "[toolchain]\n[toolchain.codegen]\nadd_invariant_assertions = true\n"
    } else {
        "[toolchain]\n[toolchain.codegen]\nadd_invariant_assertions = false\n"
    };
    let toolchain_toml_contents = add_tool_path_value(toolchain_toml_base);
    let toolchain_path = temp_dir.path().join("xlsynth-toolchain.toml");
    std::fs::write(&toolchain_path, toolchain_toml_contents).unwrap();

    // Invoke the driver.
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_path.to_str().unwrap())
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("f")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "command failed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let has_asserts = stdout.to_lowercase().contains("assert");
    if add_inv {
        assert!(has_asserts, "Expected invariant assertions in generated Verilog when add_invariant_assertions=true, but none were found. stdout: {}", stdout);
    } else {
        assert!(!has_asserts, "Did not expect invariant assertions when add_invariant_assertions=false, but some were found. stdout: {}", stdout);
    }
}

// Checks that "array_index_bounds_checking" knob toggles Verilog emission when
// an enum value (cast to bits) is used to index an array. When bounds checking
// is on the generated Verilog should differ (e.g. contain assertion logic)
// compared to when it is off.
#[test]
fn test_dslx2pipeline_array_index_bounds_checking_enum_cast() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Enum with four values (0–3). We index a 4-element array using a cast of
    // the enum to `u32`.
    let dslx = r#"enum E : u2 {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
}

fn main(sel: E, arr: u32[4]) -> u32 {
    arr[(sel as u32) + 1]
}
"#;

    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("enum_index.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Helper closure to invoke the driver with the given flag value.
    let run = |bounds_checking: bool| -> std::process::Output {
        std::process::Command::new(command_path)
            .arg("dslx2pipeline")
            .arg("--pipeline_stages")
            .arg("1")
            .arg("--delay_model")
            .arg("unit")
            .arg(format!("--array_index_bounds_checking={}", bounds_checking))
            .arg("--use_system_verilog=true")
            .arg("--flop_inputs=false")
            .arg("--flop_outputs=false")
            .arg("--dslx_input_file")
            .arg(dslx_path.to_str().unwrap())
            .arg("--dslx_top")
            .arg("main")
            .output()
            .expect("driver execution")
    };

    let out_false = run(false);
    assert!(
        out_false.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out_false.stderr)
    );
    let sv_false = String::from_utf8_lossy(&out_false.stdout).to_string();

    let out_true = run(true);
    assert!(
        out_true.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out_true.stderr)
    );
    let sv_true = String::from_utf8_lossy(&out_true.stdout).to_string();

    compare_golden_sv(
        &sv_true,
        "tests/test_dslx2pipeline_array_index_bounds_checking_enum_cast_true.golden.sv",
    );
    compare_golden_sv(
        &sv_false,
        "tests/test_dslx2pipeline_array_index_bounds_checking_enum_cast_false.golden.sv",
    );

    // The two versions should not be identical – bounds-checking inserts extra
    // logic (e.g. assertions) into the Verilog.
    assert_ne!(
        sv_false, sv_true,
        "Expected different Verilog when bounds checking is toggled"
    );
}

// Similar to the previous test but exercises the bound-checking flag on the
// `dslx-stitch-pipeline` subcommand.
#[test]
fn test_dslx_stitch_pipeline_array_index_bounds_checking_enum_cast() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Two-stage pipeline; first stage indexes an array using an enum-based index.
    let dslx = r#"enum E : u2 {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
}

fn foo_cycle0(sel: E, arr: u32[4]) -> u32 {
    arr[(sel as u3) + u3:1]
}

fn foo_cycle1(x: u32) -> u32 { x }
"#;

    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("enum_index_pipeline.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");

    let run = |bounds_checking: bool| -> std::process::Output {
        std::process::Command::new(command_path)
            .arg("dslx-stitch-pipeline")
            .arg("--dslx_input_file")
            .arg(dslx_path.to_str().unwrap())
            .arg("--dslx_top")
            .arg("foo")
            .arg(format!("--array_index_bounds_checking={}", bounds_checking))
            .arg("--use_system_verilog=true")
            .output()
            .expect("driver execution")
    };

    let out_false = run(false);
    assert!(
        out_false.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out_false.stderr)
    );
    let sv_false = String::from_utf8_lossy(&out_false.stdout).to_string();

    let out_true = run(true);
    assert!(
        out_true.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out_true.stderr)
    );
    let sv_true = String::from_utf8_lossy(&out_true.stdout).to_string();

    compare_golden_sv(
        &sv_true,
        "tests/test_dslx_stitch_pipeline_array_index_bounds_checking_enum_cast_true.golden.sv",
    );
    compare_golden_sv(
        &sv_false,
        "tests/test_dslx_stitch_pipeline_array_index_bounds_checking_enum_cast_false.golden.sv",
    );

    // Ensure the bound-checking knob changes the emitted Verilog.
    assert_ne!(
        sv_false, sv_true,
        "Expected different Verilog with bounds checking toggled"
    );
}

#[test]
fn test_irequiv_subcommand_json_equivalent() {
    let _ = env_logger::try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let lhs_ir = "package add_then_sub\nfn my_main(x: bits[32]) -> bits[32] {\n    add.2: bits[32] = add(x, x)\n    ret sub.3: bits[32] = sub(add.2, x)\n}";
    let rhs_ir = "package identity\nfn my_main(x: bits[32]) -> bits[32] {\n    ret identity.2: bits[32] = identity(x)\n}";
    let lhs_path = temp_dir.path().join("lhs.ir");
    let rhs_path = temp_dir.path().join("rhs.ir");
    std::fs::write(&lhs_path, lhs_ir).unwrap();
    std::fs::write(&rhs_path, rhs_ir).unwrap();

    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml_path, toolchain_toml_contents).unwrap();

    let json_path = temp_dir.path().join("irequiv.json");

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("ir-equiv")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--top")
        .arg("my_main")
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value =
        serde_json::from_str(json_str.trim()).expect("json file must be JSON");
    assert!(
        v.get("time_micros").is_some() && v["time_micros"].is_number(),
        "missing/invalid time in JSON: {}",
        v
    );
    assert_eq!(
        v["success"].as_bool(),
        Some(true),
        "unexpected success in JSON: {}",
        v
    );
}

#[test]
fn test_irequiv_subcommand_json_non_equivalent() {
    let _ = env_logger::try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let lhs_ir = "package add_then_sub\nfn my_main(x: bits[32]) -> bits[32] {\n    umul.2: bits[32] = umul(x, x)\n    ret udiv.3: bits[32] = udiv(umul.2, x)\n}";
    let rhs_ir = "package identity\nfn my_main(x: bits[32]) -> bits[32] {\n    ret identity.2: bits[32] = identity(x)\n}";
    let lhs_path = temp_dir.path().join("lhs.ir");
    let rhs_path = temp_dir.path().join("rhs.ir");
    std::fs::write(&lhs_path, lhs_ir).unwrap();
    std::fs::write(&rhs_path, rhs_ir).unwrap();

    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml_path, toolchain_toml_contents).unwrap();

    let json_path = temp_dir.path().join("irequiv.json");

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("ir-equiv")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--top")
        .arg("my_main")
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "expected failure status; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value =
        serde_json::from_str(json_str.trim()).expect("json file must be JSON");
    assert!(
        v.get("time_micros").is_some() && v["time_micros"].is_number(),
        "missing/invalid time in JSON: {}",
        v
    );
    assert_eq!(
        v["success"].as_bool(),
        Some(false),
        "unexpected success in JSON: {}",
        v
    );
    assert!(
        v.get("counterexample").is_some(),
        "missing counterexample in JSON: {}",
        v
    );
}

#[test]
fn test_dslx_equiv_subcommand_json_equivalent() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let lhs_dslx = "fn main(x: u32) -> u32 { (x + x) - x }";
    let rhs_dslx = "fn main(x: u32) -> u32 { x }";
    let lhs_path = temp_dir.path().join("lhs.x");
    let rhs_path = temp_dir.path().join("rhs.x");
    std::fs::write(&lhs_path, lhs_dslx).unwrap();
    std::fs::write(&rhs_path, rhs_dslx).unwrap();

    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml_path, toolchain_toml_contents).unwrap();

    let json_path = temp_dir.path().join("irequiv.json");

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("dslx-equiv")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value =
        serde_json::from_str(json_str.trim()).expect("json file must be JSON");
    assert!(
        v.get("time_micros").is_some() && v["time_micros"].is_number(),
        "missing/invalid time in JSON: {}",
        v
    );
    assert_eq!(
        v["success"].as_bool(),
        Some(true),
        "unexpected success in JSON: {}",
        v
    );
}

#[test]
fn test_dslx_equiv_subcommand_json_non_equivalent() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let lhs_dslx = "fn main(x: u32) -> u32 { (x + x) - x }";
    let rhs_dslx = "fn main(x: u32) -> u32 { x + u32:1 }";
    let lhs_path = temp_dir.path().join("lhs.x");
    let rhs_path = temp_dir.path().join("rhs.x");
    std::fs::write(&lhs_path, lhs_dslx).unwrap();
    std::fs::write(&rhs_path, rhs_dslx).unwrap();

    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml_path, toolchain_toml_contents).unwrap();

    let json_path = temp_dir.path().join("irequiv.json");

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("dslx-equiv")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "expected failure; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value =
        serde_json::from_str(json_str.trim()).expect("json file must be JSON");
    assert!(
        v.get("time_micros").is_some() && v["time_micros"].is_number(),
        "missing/invalid time in JSON: {}",
        v
    );
    assert_eq!(
        v["success"].as_bool(),
        Some(false),
        "unexpected success in JSON: {}",
        v
    );
    assert!(
        v.get("counterexample").is_some(),
        "missing counterexample in JSON: {}",
        v
    );
}

#[test]
fn test_prove_quickcheck_json_array_mixed() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("qc.x");
    std::fs::write(&dslx_path, QUICKCHECK_DSLX).unwrap();

    let toolchain_toml_path = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml_path, toolchain_toml_contents).unwrap();

    let json_path = temp_dir.path().join("irequiv.json");

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml_path.to_str().unwrap())
        .arg("prove-quickcheck")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--solver")
        .arg("toolchain")
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .output()
        .unwrap();

    // Expect non-zero because one quickcheck is false; results are written to JSON
    // file.
    assert!(
        !output.status.success(),
        "expected failure due to one disproved QC; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value =
        serde_json::from_str(json_str.trim()).expect("json file must be JSON");

    // New schema: object { success: bool, tests: [ ... ] }
    assert_eq!(v["success"].as_bool(), Some(false));
    let arr = v["tests"].as_array().expect("expected tests array in JSON");
    assert_eq!(arr.len(), 2, "expected two quickchecks");
    // Map name->success for stable checks; ensure name is included.
    let mut name_success: std::collections::HashMap<String, bool> =
        std::collections::HashMap::new();
    for item in arr.iter() {
        let name = item["name"].as_str().expect("missing name").to_string();
        assert!(item.get("time_micros").is_some() && item["time_micros"].is_number());
        let ok = item["success"].as_bool().expect("missing success bool");
        if !ok {
            assert!(item.get("counterexample").is_some());
            assert!(item["counterexample"].is_string());
        }
        name_success.insert(name, ok);
    }
    assert_eq!(name_success.get("qc_success"), Some(&true));
    assert_eq!(name_success.get("qc_failure"), Some(&false));
}

#[test]
fn test_prover_all_two_equiv_tasks_succeeds() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path();

    // Prepare IR texts with explicit top my_main.
    let lhs_equiv = "package add_then_sub\nfn my_main(x: bits[32]) -> bits[32] {\n    add.2: bits[32] = add(x, x)\n    ret sub.3: bits[32] = sub(add.2, x)\n}";
    let rhs_equiv = "package identity\nfn my_main(x: bits[32]) -> bits[32] {\n    ret identity.2: bits[32] = identity(x)\n}";

    let lhs1 = dir.join("lhs1.ir");
    let rhs1 = dir.join("rhs1.ir");
    std::fs::write(&lhs1, lhs_equiv).unwrap();
    std::fs::write(&rhs1, rhs_equiv).unwrap();
    // Second pair identical to ensure success.
    let lhs2 = dir.join("lhs2.ir");
    let rhs2 = dir.join("rhs2.ir");
    std::fs::write(&lhs2, lhs_equiv).unwrap();
    std::fs::write(&rhs2, rhs_equiv).unwrap();

    // Toolchain config so child subcommands can run in toolchain mode.
    let toolchain_toml = dir.join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    // Build plan JSON.
    let plan = format!(
        r#"{{
  "kind": "all",
  "tasks": [
    {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }},
    {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }}
  ]
}}"#,
        lhs1.display(),
        rhs1.display(),
        lhs2.display(),
        rhs2.display()
    );
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let json_path = dir.join("prover.json");
    let output = Command::new(driver)
        .arg("prover")
        .arg("--cores")
        .arg("2")
        .arg("--plan_json_file")
        .arg(plan_path.to_str().unwrap())
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .current_dir(dir)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "prover all(two equiv) should succeed; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap();
    assert_eq!(v["success"].as_bool(), Some(true));
}

#[test]
fn test_prover_single_task_reports_task_id() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path();

    // Prepare an equivalent IR pair with explicit top.
    let lhs_equiv = "package add_then_sub\nfn my_main(x: bits[32]) -> bits[32] {\n    add.2: bits[32] = add(x, x)\n    ret sub.3: bits[32] = sub(add.2, x)\n}";
    let rhs_equiv = "package identity\nfn my_main(x: bits[32]) -> bits[32] {\n    ret identity.2: bits[32] = identity(x)\n}";
    let lhs = dir.join("lhs.ir");
    let rhs = dir.join("rhs.ir");
    std::fs::write(&lhs, lhs_equiv).unwrap();
    std::fs::write(&rhs, rhs_equiv).unwrap();

    // Toolchain config so child subcommands can run.
    let toolchain_toml = dir.join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    // Build a single-task plan with task_id.
    let plan = format!(
        r#"{{
  "kind": "ir-equiv",
  "lhs_ir_file": "{}",
  "rhs_ir_file": "{}",
  "top": "my_main",
  "solver": "toolchain",
  "task_id": "tid-xyz"
}}"#,
        lhs.display(),
        rhs.display()
    );
    let plan_path = dir.join("plan_single.json");
    std::fs::write(&plan_path, plan).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let json_path = dir.join("prover_single.json");
    let output = Command::new(driver)
        .arg("prover")
        .arg("--cores")
        .arg("1")
        .arg("--plan_json_file")
        .arg(plan_path.to_str().unwrap())
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .current_dir(dir)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "prover single task should succeed; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap();
    assert_eq!(v["success"].as_bool(), Some(true));
    assert_eq!(v["plan"]["task_id"].as_str(), Some("tid-xyz"));
}

#[test]
fn test_prover_all_mixed_tasks_fails() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path();

    // Equiv pair.
    let lhs_equiv = "package add_then_sub\nfn my_main(x: bits[32]) -> bits[32] {\n    add.2: bits[32] = add(x, x)\n    ret sub.3: bits[32] = sub(add.2, x)\n}";
    let rhs_equiv = "package identity\nfn my_main(x: bits[32]) -> bits[32] {\n    ret identity.2: bits[32] = identity(x)\n}";
    let lhs_eq = dir.join("lhs_eq.ir");
    let rhs_eq = dir.join("rhs_eq.ir");
    std::fs::write(&lhs_eq, lhs_equiv).unwrap();
    std::fs::write(&rhs_eq, rhs_equiv).unwrap();

    // Add a QuickCheck DSLX file with one passing and one failing test.
    let qc_path = dir.join("qc.x");
    std::fs::write(&qc_path, QUICKCHECK_DSLX).unwrap();

    // DSLX files for an equivalent pair: (x + x) - x == x
    let lhs_dslx = dir.join("lhs.x");
    let rhs_dslx = dir.join("rhs.x");
    std::fs::write(&lhs_dslx, "fn main(x: u32) -> u32 { (x + x) - x }").unwrap();
    std::fs::write(&rhs_dslx, "fn main(x: u32) -> u32 { x }").unwrap();

    let toolchain_toml = dir.join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    let plan = format!(
        r#"{{
  "kind": "all",
  "tasks": [
    {{ "kind": "dslx-equiv", "lhs_dslx_file": "{}", "rhs_dslx_file": "{}", "dslx_top": "main", "solver": "toolchain" }},
    {{ "kind": "prove-quickcheck", "dslx_input_file": "{}", "solver": "toolchain" }}
  ]
}}"#,
        lhs_dslx.display(),
        rhs_dslx.display(),
        qc_path.display()
    );
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let json_path = dir.join("prover.json");
    let output = Command::new(driver)
        .arg("prover")
        .arg("--cores")
        .arg("2")
        .arg("--plan_json_file")
        .arg(plan_path.to_str().unwrap())
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .current_dir(dir)
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "prover all(mixed) should fail; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap();
    assert_eq!(v["success"].as_bool(), Some(false));
}

#[test]
fn test_prover_first_both_equiv_succeeds() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path();

    let lhs_equiv = "package add_then_sub\nfn my_main(x: bits[32]) -> bits[32] {\n    add.2: bits[32] = add(x, x)\n    ret sub.3: bits[32] = sub(add.2, x)\n}";
    let rhs_equiv = "package identity\nfn my_main(x: bits[32]) -> bits[32] {\n    ret identity.2: bits[32] = identity(x)\n}";

    let lhs1 = dir.join("lhs1.ir");
    let rhs1 = dir.join("rhs1.ir");
    let lhs2 = dir.join("lhs2.ir");
    let rhs2 = dir.join("rhs2.ir");
    for (p, s) in [
        (&lhs1, lhs_equiv),
        (&rhs1, rhs_equiv),
        (&lhs2, lhs_equiv),
        (&rhs2, rhs_equiv),
    ] {
        std::fs::write(p, s).unwrap();
    }

    let toolchain_toml = dir.join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    let plan = format!(
        r#"{{
  "kind": "first",
  "tasks": [
    {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }},
    {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }}
  ]
}}"#,
        lhs1.display(),
        rhs1.display(),
        lhs2.display(),
        rhs2.display()
    );
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let json_path = dir.join("prover.json");
    let output = Command::new(driver)
        .arg("prover")
        .arg("--cores")
        .arg("2")
        .arg("--plan_json_file")
        .arg(plan_path.to_str().unwrap())
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .current_dir(dir)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "prover first(two equiv) should succeed; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap();
    assert_eq!(v["success"].as_bool(), Some(true));
}

#[test]
fn test_prover_first_both_nonequiv_fails() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path();

    let lhs_ne = "package add_then_sub\nfn my_main(x: bits[32]) -> bits[32] {\n    umul.2: bits[32] = umul(x, x)\n    ret udiv.3: bits[32] = udiv(umul.2, x)\n}";
    let rhs_equiv = "package identity\nfn my_main(x: bits[32]) -> bits[32] {\n    ret identity.2: bits[32] = identity(x)\n}";

    let lhs1 = dir.join("lhs1.ir");
    let rhs1 = dir.join("rhs1.ir");
    let lhs2 = dir.join("lhs2.ir");
    let rhs2 = dir.join("rhs2.ir");
    for (p, s) in [
        (&lhs1, lhs_ne),
        (&rhs1, rhs_equiv),
        (&lhs2, lhs_ne),
        (&rhs2, rhs_equiv),
    ] {
        std::fs::write(p, s).unwrap();
    }

    let toolchain_toml = dir.join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    let plan = format!(
        r#"{{
  "kind": "first",
  "tasks": [
    {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }},
    {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }}
  ]
}}"#,
        lhs1.display(),
        rhs1.display(),
        lhs2.display(),
        rhs2.display()
    );
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let json_path = dir.join("prover.json");
    let output = Command::new(driver)
        .arg("prover")
        .arg("--cores")
        .arg("2")
        .arg("--plan_json_file")
        .arg(plan_path.to_str().unwrap())
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .current_dir(dir)
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "prover first(two nonequiv) should fail; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap();
    assert_eq!(v["success"].as_bool(), Some(false));
}

#[test]
fn test_prover_any_both_equiv_succeeds() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path();

    let lhs_equiv = "package add_then_sub\nfn my_main(x: bits[32]) -> bits[32] {\n    add.2: bits[32] = add(x, x)\n    ret sub.3: bits[32] = sub(add.2, x)\n}";
    let rhs_equiv = "package identity\nfn my_main(x: bits[32]) -> bits[32] {\n    ret identity.2: bits[32] = identity(x)\n}";

    let lhs1 = dir.join("lhs1.ir");
    let rhs1 = dir.join("rhs1.ir");
    let lhs2 = dir.join("lhs2.ir");
    let rhs2 = dir.join("rhs2.ir");
    for (p, s) in [
        (&lhs1, lhs_equiv),
        (&rhs1, rhs_equiv),
        (&lhs2, lhs_equiv),
        (&rhs2, rhs_equiv),
    ] {
        std::fs::write(p, s).unwrap();
    }

    let toolchain_toml = dir.join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    let plan = format!(
        r#"{{
  "kind": "any",
  "tasks": [
    {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }},
    {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }}
  ]
}}"#,
        lhs1.display(),
        rhs1.display(),
        lhs2.display(),
        rhs2.display()
    );
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let json_path = dir.join("prover.json");
    let output = Command::new(driver)
        .arg("prover")
        .arg("--cores")
        .arg("2")
        .arg("--plan_json_file")
        .arg(plan_path.to_str().unwrap())
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .current_dir(dir)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "prover any(two equiv) should succeed; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap();
    assert_eq!(v["success"].as_bool(), Some(true));
}

#[test]
fn test_prover_any_both_nonequiv_fails() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path();

    let lhs_ne = "package add_then_sub\nfn my_main(x: bits[32]) -> bits[32] {\n    umul.2: bits[32] = umul(x, x)\n    ret udiv.3: bits[32] = udiv(umul.2, x)\n}";
    let rhs_equiv = "package identity\nfn my_main(x: bits[32]) -> bits[32] {\n    ret identity.2: bits[32] = identity(x)\n}";

    let lhs1 = dir.join("lhs1.ir");
    let rhs1 = dir.join("rhs1.ir");
    let lhs2 = dir.join("lhs2.ir");
    let rhs2 = dir.join("rhs2.ir");
    for (p, s) in [
        (&lhs1, lhs_ne),
        (&rhs1, rhs_equiv),
        (&lhs2, lhs_ne),
        (&rhs2, rhs_equiv),
    ] {
        std::fs::write(p, s).unwrap();
    }

    let toolchain_toml = dir.join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    let plan = format!(
        r#"{{
  "kind": "any",
  "tasks": [
    {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }},
    {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }}
  ]
}}"#,
        lhs1.display(),
        rhs1.display(),
        lhs2.display(),
        rhs2.display()
    );
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let json_path = dir.join("prover.json");
    let output = Command::new(driver)
        .arg("prover")
        .arg("--cores")
        .arg("2")
        .arg("--plan_json_file")
        .arg(plan_path.to_str().unwrap())
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .current_dir(dir)
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "prover any(two nonequiv) should fail; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap();
    assert_eq!(v["success"].as_bool(), Some(false));
}

#[test]
fn test_prover_any_mixed_succeeds() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path();

    // QuickCheck file: use only the passing test via filter.
    let qc_path = dir.join("qc.x");
    std::fs::write(&qc_path, QUICKCHECK_DSLX).unwrap();

    // DSLX non-equivalent pair.
    let lhs_dslx = dir.join("lhs.x");
    let rhs_dslx = dir.join("rhs.x");
    std::fs::write(&lhs_dslx, "fn main(x: u32) -> u32 { x }").unwrap();
    std::fs::write(&rhs_dslx, "fn main(x: u32) -> u32 { x + u32:1 }").unwrap();

    // Toolchain config in CWD for children.
    let toolchain_toml = dir.join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    // Any: success if any task succeeds; run with cores=2 to avoid order
    // dependence.
    let plan = format!(
        r#"{{
  "kind": "any",
  "tasks": [
    {{ "kind": "prove-quickcheck", "dslx_input_file": "{}", "test_filter": ".*success", "solver": "toolchain" }},
    {{ "kind": "dslx-equiv", "lhs_dslx_file": "{}", "rhs_dslx_file": "{}", "dslx_top": "main", "solver": "toolchain" }}
  ]
}}"#,
        qc_path.display(),
        lhs_dslx.display(),
        rhs_dslx.display()
    );
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let json_path = dir.join("prover.json");
    let output = Command::new(driver)
        .arg("prover")
        .arg("--cores")
        .arg("2")
        .arg("--plan_json_file")
        .arg(plan_path.to_str().unwrap())
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .current_dir(dir)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "prover any(mixed) should succeed; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap();
    assert_eq!(v["success"].as_bool(), Some(true));
}

#[test]
fn test_prover_any_keep_running_mixed_children_finish() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path();

    // Prepare IR texts with explicit top my_main.
    let lhs_equiv = "package add_then_sub\nfn my_main(x: bits[32]) -> bits[32] {\n    add.2: bits[32] = add(x, x)\n    ret sub.3: bits[32] = sub(add.2, x)\n}";
    let rhs_equiv = "package identity\nfn my_main(x: bits[32]) -> bits[32] {\n    ret identity.2: bits[32] = identity(x)\n}";
    let lhs_ne = "package add_then_sub\nfn my_main(x: bits[32]) -> bits[32] {\n    umul.2: bits[32] = umul(x, x)\n    ret udiv.3: bits[32] = udiv(umul.2, x)\n}";

    // Files: one success pair, one failure pair.
    let lhs1 = dir.join("lhs1.ir");
    let rhs1 = dir.join("rhs1.ir");
    let lhs2 = dir.join("lhs2.ir");
    let rhs2 = dir.join("rhs2.ir");
    std::fs::write(&lhs1, lhs_equiv).unwrap();
    std::fs::write(&rhs1, rhs_equiv).unwrap();
    std::fs::write(&lhs2, lhs_ne).unwrap();
    std::fs::write(&rhs2, rhs_equiv).unwrap();

    let toolchain_toml = dir.join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    // Any with keep_running_till_finish=true: expect overall success, and both
    // children completed (no Canceled).
    let plan = format!(
        r#"{{
  "kind": "any",
  "keep_running_till_finish": true,
  "tasks": [
    {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }},
    {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }}
  ]
}}"#,
        lhs1.display(),
        rhs1.display(),
        lhs2.display(),
        rhs2.display()
    );
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let json_path = dir.join("prover_any_keep_running.json");
    let output = Command::new(driver)
        .arg("prover")
        .arg("--cores")
        .arg("2")
        .arg("--plan_json_file")
        .arg(plan_path.to_str().unwrap())
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .current_dir(dir)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "prover any(keep_running, mixed) should succeed; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap();
    assert_eq!(v["success"].as_bool(), Some(true));
    // Verify both children finished with concrete outcomes, not Canceled.
    let tasks = &v["plan"]["tasks"];
    assert!(tasks.is_array());
    let t0 = &tasks[0];
    let t1 = &tasks[1];
    assert_eq!(t0["outcome"].as_str(), Some("Success"));
    assert_eq!(t1["outcome"].as_str(), Some("Failed"));
}

#[test]
fn test_prover_first_keep_running_both_nonequiv_finish() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path();

    let lhs_ne = "package add_then_sub\nfn my_main(x: bits[32]) -> bits[32] {\n    umul.2: bits[32] = umul(x, x)\n    ret udiv.3: bits[32] = udiv(umul.2, x)\n}";
    let rhs_equiv = "package identity\nfn my_main(x: bits[32]) -> bits[32] {\n    ret identity.2: bits[32] = identity(x)\n}";
    let lhs1 = dir.join("lhs1.ir");
    let rhs1 = dir.join("rhs1.ir");
    let lhs2 = dir.join("lhs2.ir");
    let rhs2 = dir.join("rhs2.ir");
    for (p, s) in [
        (&lhs1, lhs_ne),
        (&rhs1, rhs_equiv),
        (&lhs2, lhs_ne),
        (&rhs2, rhs_equiv),
    ] {
        std::fs::write(p, s).unwrap();
    }

    let toolchain_toml = dir.join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    let plan = format!(
        r#"{{
  "kind": "first",
  "keep_running_till_finish": true,
  "tasks": [
    {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }},
    {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }}
  ]
}}"#,
        lhs1.display(),
        rhs1.display(),
        lhs2.display(),
        rhs2.display()
    );
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let json_path = dir.join("prover_first_keep_running.json");
    let output = Command::new(driver)
        .arg("prover")
        .arg("--cores")
        .arg("2")
        .arg("--plan_json_file")
        .arg(plan_path.to_str().unwrap())
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .current_dir(dir)
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "prover first(keep_running, both nonequiv) should fail; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap();
    assert_eq!(v["success"].as_bool(), Some(false));
    let tasks = &v["plan"]["tasks"];
    assert!(tasks.is_array());
    let t0 = &tasks[0];
    let t1 = &tasks[1];
    assert_eq!(t0["outcome"].as_str(), Some("Failed"));
    assert_eq!(t1["outcome"].as_str(), Some("Failed"));
}

#[test]
fn test_prover_nested_all_any_succeeds() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path();

    // Prepare IR texts with explicit top my_main.
    let lhs_equiv = "package add_then_sub\nfn my_main(x: bits[32]) -> bits[32] {\n    add.2: bits[32] = add(x, x)\n    ret sub.3: bits[32] = sub(add.2, x)\n}";
    let rhs_equiv = "package identity\nfn my_main(x: bits[32]) -> bits[32] {\n    ret identity.2: bits[32] = identity(x)\n}";
    let lhs_ne = "package add_then_sub\nfn my_main(x: bits[32]) -> bits[32] {\n    umul.2: bits[32] = umul(x, x)\n    ret udiv.3: bits[32] = udiv(umul.2, x)\n}";

    // Files for ir-equiv success pair.
    let lhs1 = dir.join("lhs1.ir");
    let rhs1 = dir.join("rhs1.ir");
    std::fs::write(&lhs1, lhs_equiv).unwrap();
    std::fs::write(&rhs1, rhs_equiv).unwrap();

    // Files for ir-equiv failure pair.
    let lhs2 = dir.join("lhs2.ir");
    let rhs2 = dir.join("rhs2.ir");
    std::fs::write(&lhs2, lhs_ne).unwrap();
    std::fs::write(&rhs2, rhs_equiv).unwrap();

    // DSLX files for an equivalent pair: (x + x) - x == x
    let lhs_dslx = dir.join("lhs.x");
    let rhs_dslx = dir.join("rhs.x");
    std::fs::write(&lhs_dslx, "fn main(x: u32) -> u32 { (x + x) - x }").unwrap();
    std::fs::write(&rhs_dslx, "fn main(x: u32) -> u32 { x }").unwrap();

    // QuickCheck file with success/failure tests; we'll filter to the success one.
    let qc_path = dir.join("qc.x");
    std::fs::write(&qc_path, QUICKCHECK_DSLX).unwrap();

    // Toolchain config in CWD for child subcommands.
    let toolchain_toml = dir.join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    // Build a nested plan:
    // all(
    //   any(ir-equiv success, ir-equiv failure),
    //   all(dslx-equiv success, prove-quickcheck filtered success)
    // )
    let plan = format!(
        r#"{{
  "kind": "all",
  "tasks": [
    {{
      "kind": "any",
      "tasks": [
        {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }},
        {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }}
      ]
    }},
    {{
      "kind": "all",
      "tasks": [
        {{ "kind": "dslx-equiv", "lhs_dslx_file": "{}", "rhs_dslx_file": "{}", "dslx_top": "main", "solver": "toolchain" }},
        {{ "kind": "prove-quickcheck", "dslx_input_file": "{}", "test_filter": ".*success", "solver": "toolchain" }}
      ]
    }}
  ]
}}"#,
        lhs1.display(),
        rhs1.display(),
        lhs2.display(),
        rhs2.display(),
        lhs_dslx.display(),
        rhs_dslx.display(),
        qc_path.display()
    );
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let json_path = dir.join("prover_nested_success.json");
    let output = Command::new(driver)
        .arg("prover")
        .arg("--cores")
        .arg("3")
        .arg("--plan_json_file")
        .arg(plan_path.to_str().unwrap())
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .current_dir(dir)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "nested prover(all(any, all)) should succeed; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // Optionally, sanity-check the JSON success flag.
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap();
    assert_eq!(v["success"].as_bool(), Some(true));
}

#[test]
fn test_prover_nested_any_all_fails() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path();

    // Prepare IR texts.
    let lhs_ne = "package add_then_sub\nfn my_main(x: bits[32]) -> bits[32] {\n    umul.2: bits[32] = umul(x, x)\n    ret udiv.3: bits[32] = udiv(umul.2, x)\n}";
    let rhs_equiv = "package identity\nfn my_main(x: bits[32]) -> bits[32] {\n    ret identity.2: bits[32] = identity(x)\n}";

    // Failing ir-equiv pair used in both subgroups.
    let lhs1 = dir.join("lhs1.ir");
    let rhs1 = dir.join("rhs1.ir");
    let lhs2 = dir.join("lhs2.ir");
    let rhs2 = dir.join("rhs2.ir");
    for (p, s) in [
        (&lhs1, lhs_ne),
        (&rhs1, rhs_equiv),
        (&lhs2, lhs_ne),
        (&rhs2, rhs_equiv),
    ] {
        std::fs::write(p, s).unwrap();
    }

    // DSLX equivalent pair (won't save the all-group that has a failing ir-equiv).
    let lhs_dslx = dir.join("lhs.x");
    let rhs_dslx = dir.join("rhs.x");
    std::fs::write(&lhs_dslx, "fn main(x: u32) -> u32 { (x + x) - x }").unwrap();
    std::fs::write(&rhs_dslx, "fn main(x: u32) -> u32 { x }").unwrap();

    // QuickCheck file with failure filter to ensure subgroup failure.
    let qc_path = dir.join("qc.x");
    std::fs::write(&qc_path, QUICKCHECK_DSLX).unwrap();

    let toolchain_toml = dir.join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    // Build a nested plan that should fail overall:
    // any(
    //   all(ir-equiv fail, dslx-equiv success),
    //   all(ir-equiv fail, prove-quickcheck filtered failure)
    // )
    let plan = format!(
        r#"{{
  "kind": "any",
  "tasks": [
    {{
      "kind": "all",
      "tasks": [
        {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }},
        {{ "kind": "dslx-equiv", "lhs_dslx_file": "{}", "rhs_dslx_file": "{}", "dslx_top": "main", "solver": "toolchain" }}
      ]
    }},
    {{
      "kind": "all",
      "tasks": [
        {{ "kind": "ir-equiv", "lhs_ir_file": "{}", "rhs_ir_file": "{}", "top": "my_main", "solver": "toolchain" }},
        {{ "kind": "prove-quickcheck", "dslx_input_file": "{}", "test_filter": ".*failure", "solver": "toolchain" }}
      ]
    }}
  ]
}}"#,
        lhs1.display(),
        rhs1.display(),
        lhs_dslx.display(),
        rhs_dslx.display(),
        lhs2.display(),
        rhs2.display(),
        qc_path.display()
    );
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let json_path = dir.join("prover_nested_failure.json");
    let output = Command::new(driver)
        .arg("prover")
        .arg("--cores")
        .arg("3")
        .arg("--plan_json_file")
        .arg(plan_path.to_str().unwrap())
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .current_dir(dir)
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "nested prover(any(all(...), all(...))) should fail; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap();
    println!("{}", json_str);
    assert_eq!(v["success"].as_bool(), Some(false));
}

// -----------------------------------------------------------------------------
// Uninterpreted function (UF) tests for dslx-equiv
// -----------------------------------------------------------------------------

#[cfg_attr(feature="has-boolector", test_case::test_case("boolector"; "dslx_equiv_uf_boolector"))]
#[cfg_attr(feature="has-bitwuzla", test_case::test_case("bitwuzla"; "dslx_equiv_uf_bitwuzla"))]
#[cfg_attr(feature="with-z3-binary-test", test_case::test_case("z3-binary"; "dslx_equiv_uf_z3_binary"))]
#[cfg_attr(feature="with-bitwuzla-binary-test", test_case::test_case("bitwuzla-binary"; "dslx_equiv_uf_bitwuzla_binary"))]
#[cfg_attr(feature="with-boolector-binary-test", test_case::test_case("boolector-binary"; "dslx_equiv_uf_boolector_binary"))]
fn test_dslx_equiv_uninterpreted_functions_solver_param(solver: &str) {
    let _ = env_logger::builder().is_test(true).try_init();

    // LHS calls g(x) = x + x; RHS calls h(x) = x - x. Not equivalent unless both
    // g and h are mapped to the same uninterpreted function symbol.
    let lhs_dslx = "fn g(x: u8) -> u8 { x + x }\nfn main(x: u8) -> u8 { g(x) }";
    let rhs_dslx = "fn h(x: u8) -> u8 { x - x }\nfn main(x: u8) -> u8 { h(x) }";

    let temp_dir = tempfile::tempdir().unwrap();
    let lhs_path = temp_dir.path().join("lhs.x");
    let rhs_path = temp_dir.path().join("rhs.x");
    std::fs::write(&lhs_path, lhs_dslx).unwrap();
    std::fs::write(&rhs_path, rhs_dslx).unwrap();

    // Provide a toolchain.toml so any external binary backends (if selected)
    // can run; native SMT backends will ignore it.
    let toolchain_toml = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_contents).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    // 1) Without UF mapping: should fail (functions compute different results).
    let output_no_uf = std::process::Command::new(driver)
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("dslx-equiv")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .arg("--solver")
        .arg(solver)
        .output()
        .unwrap();
    assert!(
        !output_no_uf.status.success(),
        "dslx-equiv without UF mapping should fail; stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output_no_uf.stdout),
        String::from_utf8_lossy(&output_no_uf.stderr)
    );

    // 2) With UF mapping: map LHS g and RHS h to the same UF symbol "F".
    let output_with_uf = std::process::Command::new(driver)
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("dslx-equiv")
        .arg(lhs_path.to_str().unwrap())
        .arg(rhs_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .arg("--solver")
        .arg(solver)
        .arg("--lhs_uf")
        .arg("g:F")
        .arg("--rhs_uf")
        .arg("h:F")
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output_with_uf.stdout);
    let stderr = String::from_utf8_lossy(&output_with_uf.stderr);
    assert!(
        output_with_uf.status.success(),
        "dslx-equiv with UF mapping should succeed for solver {}.\nstdout: {}\nstderr: {}",
        solver,
        stdout,
        stderr
    );
    assert!(stdout.contains("success"), "stdout: {}", stdout);
}

#[test]
fn test_ir_localized_eco_midlevel_insert_and_substitute() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Build the "old" IR function per description.
    let old_ir = r#"package eco_pkg

top fn eco_top(x: bits[1] id=1, y: bits[1] id=2) -> bits[1] {
  and.3: bits[1] = and(x, y, id=3)
  not.4: bits[1] = not(and.3, id=4)
  not.5: bits[1] = not(and.3, id=5)
  not.6: bits[1] = not(not.4, id=6)
  and.7: bits[1] = and(not.4, not.5, id=7)
  not.8: bits[1] = not(not.5, id=8)
  and.9: bits[1] = and(not.6, and.7, id=9)
  and.10: bits[1] = and(and.7, not.8, id=10)
  ret and.11: bits[1] = and(and.9, and.10, id=11)
}
"#;

    // Build the "new" IR where we insert n1p5_0 = and(n1_0, n1_1) and
    // substitute one operand of n2_1 (and.7) to be that new node.
    // Concretely: change and.7 from and(not.4, not.5) to and(not.4, and.12).
    let new_ir = r#"package eco_pkg

top fn eco_top(x: bits[1] id=1, y: bits[1] id=2) -> bits[1] {
  and.3: bits[1] = and(x, y, id=3)
  not.4: bits[1] = not(and.3, id=4)
  not.5: bits[1] = not(and.3, id=5)
  and.12: bits[1] = and(not.4, not.5, id=12)
  not.6: bits[1] = not(not.4, id=6)
  and.7: bits[1] = and(not.4, and.12, id=7)
  not.8: bits[1] = not(not.5, id=8)
  and.9: bits[1] = and(not.6, and.7, id=9)
  and.10: bits[1] = and(and.7, not.8, id=10)
  ret and.11: bits[1] = and(and.9, and.10, id=11)
}
"#;

    let temp_dir = tempfile::tempdir().unwrap();
    let old_path = temp_dir.path().join("old.ir");
    let new_path = temp_dir.path().join("new.ir");
    std::fs::write(&old_path, old_ir).unwrap();
    std::fs::write(&new_path, new_ir).unwrap();

    let json_out = temp_dir.path().join("eco_report.json");
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = std::process::Command::new(command_path)
        .arg("ir-localized-eco")
        .arg(old_path.to_str().unwrap())
        .arg(new_path.to_str().unwrap())
        .arg("--json_out")
        .arg(json_out.to_str().unwrap())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // Load JSON and check that we have a substitution at the and.7 node for one
    // operand where the new operand signature is an and(...) and the old
    // operand signature is a not(...).
    let json_bytes = std::fs::read(&json_out).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();
    let edits = v
        .get("edits")
        .and_then(|e| e.get("edits"))
        .and_then(|e| e.as_array())
        .expect("edits array present");
    let mut saw_change = false;
    for e in edits.iter() {
        if e.get("RewriteOperands").is_some() {
            saw_change = true;
            break;
        }
    }
    assert!(
        saw_change,
        "Expected a RewriteOperands edit at root. JSON: {}",
        String::from_utf8_lossy(&json_bytes)
    );

    // As a final step, create a "patched old" IR (for now identical to new_ir) and
    // prove equivalence to new_ir via the ir-equiv subcommand to validate the
    // proving pipeline/flags.
    let patched_old_path = temp_dir.path().join("patched_old.ir");
    std::fs::write(&patched_old_path, new_ir).unwrap();
    // Provide a toolchain toml for ir-equiv's toolchain solver.
    let toolchain_toml = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_contents).unwrap();

    let eq_out = std::process::Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("ir-equiv")
        .arg(patched_old_path.to_str().unwrap())
        .arg(new_path.to_str().unwrap())
        .arg("--top")
        .arg("eco_top")
        .output()
        .unwrap();
    assert!(
        eq_out.status.success(),
        "ir-equiv should succeed for patched_old vs new; stdout: {} stderr: {}",
        String::from_utf8_lossy(&eq_out.stdout),
        String::from_utf8_lossy(&eq_out.stderr)
    );
}

#[test]
fn test_ir_round_trip_strip_pos_attrs_flag() {
    let ir_with_file = r#"package p

file_number 0 "foo.x"

top fn main(x: bits[8]) -> bits[8] {
  arr: bits[8][4] = literal(value=[0, 1, 2, 3], id=1)
  ai: bits[8] = array_index(arr, indices=[x], assumed_in_bounds=false, id=2)
  ret identity.3: bits[8] = identity(ai)
}
"#;
    let tmp = tempfile::tempdir().unwrap();
    let ir_path = tmp.path().join("in.ir");
    std::fs::write(&ir_path, ir_with_file).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    // Default: should preserve file_number lines.
    let out_preserve = std::process::Command::new(driver)
        .arg("ir-round-trip")
        .arg(ir_path.to_str().unwrap())
        .output()
        .unwrap();
    assert!(out_preserve.status.success());
    let s_preserve = String::from_utf8_lossy(&out_preserve.stdout);
    assert!(
        s_preserve.contains("file_number"),
        "should preserve file_number by default: {}",
        s_preserve
    );
    // array_index assumed_in_bounds=false should not be printed explicitly
    assert!(
        !s_preserve.contains("assumed_in_bounds=false"),
        "should omit default false assumed_in_bounds"
    );

    // With --strip-pos-attrs=true: should not include file_number lines.
    let out_strip = std::process::Command::new(driver)
        .arg("ir-round-trip")
        .arg(ir_path.to_str().unwrap())
        .arg("--strip-pos-attrs")
        .arg("true")
        .output()
        .unwrap();
    assert!(out_strip.status.success());
    let s_strip = String::from_utf8_lossy(&out_strip.stdout);
    assert!(
        !s_strip.contains("file_number"),
        "should strip file_number with flag: {}",
        s_strip
    );
}

// -----------------------------------------------------------------------------
// Uninterpreted function (UF) tests for prove-quickcheck
// -----------------------------------------------------------------------------

#[cfg_attr(feature="has-boolector", test_case::test_case("boolector"; "qc_uf_boolector"))]
#[cfg_attr(feature="has-bitwuzla", test_case::test_case("bitwuzla"; "qc_uf_bitwuzla"))]
#[cfg_attr(feature="with-z3-binary-test", test_case::test_case("z3-binary"; "qc_uf_z3_binary"))]
#[cfg_attr(feature="with-bitwuzla-binary-test", test_case::test_case("bitwuzla-binary"; "qc_uf_bitwuzla_binary"))]
#[cfg_attr(feature="with-boolector-binary-test", test_case::test_case("boolector-binary"; "qc_uf_boolector_binary"))]
fn test_prove_quickcheck_uninterpreted_functions_solver_param(solver: &str) {
    let _ = env_logger::builder().is_test(true).try_init();

    // Property compares two different helper functions; false without UF mapping,
    // true when both helpers are mapped to the same uninterpreted function symbol.
    let dslx = "fn g(x: u8) -> u8 { x + x }\nfn h(x: u8) -> u8 { x - x }\n#[quickcheck] fn qc_equiv(x: u8) -> bool { g(x) == h(x) }\n";

    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("qc.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    // Toolchain file for parity with other tests (native solvers don't require it).
    let toolchain_toml = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_contents).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    // 1) Without UF mapping: expect failure (property is false).
    let out_no_uf = std::process::Command::new(driver)
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("prove-quickcheck")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--test_filter")
        .arg("qc_equiv")
        .arg("--solver")
        .arg(solver)
        .output()
        .unwrap();
    assert!(
        !out_no_uf.status.success(),
        "prove-quickcheck without UF mapping should fail. stdout: {} stderr: {}",
        String::from_utf8_lossy(&out_no_uf.stdout),
        String::from_utf8_lossy(&out_no_uf.stderr)
    );

    // 2) With UF mapping: compute mangled helper names and map both to symbol F.
    let out_with_uf = std::process::Command::new(driver)
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("prove-quickcheck")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--test_filter")
        .arg("qc_equiv")
        .arg("--solver")
        .arg(solver)
        .arg("--uf")
        .arg(&"g:F")
        .arg("--uf")
        .arg(&"h:F")
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&out_with_uf.stdout);
    let stderr = String::from_utf8_lossy(&out_with_uf.stderr);
    assert!(
        out_with_uf.status.success(),
        "prove-quickcheck with UF mapping should succeed for solver {}.\nstdout: {}\nstderr: {}",
        solver,
        stdout,
        stderr
    );
    assert!(
        stdout.contains("Success: All QuickChecks proved"),
        "stdout: {}",
        stdout
    );
}

// -----------------------------------------------------------------------------
// Timeout behavior tests using fake tasks
// -----------------------------------------------------------------------------

#[cfg(feature = "enable-fake-task")]
#[test]
fn test_prover_single_fake_timeout() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path();

    // Single fake task that will exceed its timeout.
    let plan = r#"{
      "kind": "fake",
      "success": true,
      "stdout_len": 0,
      "stderr_len": 0,
      "timeout_ms": 1
    }"#;
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let json_path = dir.join("prover_timeout_single.json");
    let output = std::process::Command::new(driver)
        .arg("prover")
        .arg("--cores")
        .arg("1")
        .arg("--plan_json_file")
        .arg(plan_path.to_str().unwrap())
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .current_dir(dir)
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "single fake with timeout should fail; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap();
    assert_eq!(v["success"].as_bool(), Some(false));
    assert_eq!(v["plan"]["outcome"].as_str(), Some("Timeout"));
}

#[cfg(feature = "enable-fake-task")]
#[test]
fn test_prover_all_with_timeout_indefinite() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path();

    // all(success within timeout, timeout)
    let plan = r#"{
      "kind": "all",
      "tasks": [
        { "kind": "fake", "delay_ms": 10, "success": true,  "stdout_len": 0, "stderr_len": 0},
        { "kind": "fake", "success": true, "stdout_len": 0, "stderr_len": 0, "timeout_ms": 1 }
      ]
    }"#;
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let json_path = dir.join("prover_all_timeout.json");
    let output = std::process::Command::new(driver)
        .arg("prover")
        .arg("--cores")
        .arg("2")
        .arg("--plan_json_file")
        .arg(plan_path.to_str().unwrap())
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .current_dir(dir)
        .output()
        .unwrap();

    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    assert!(
        !output.status.success(),
        "all(success, timeout) should be overall non-success; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap();
    assert_eq!(v["success"].as_bool(), Some(false));
    // Group outcome becomes IndefiniteChildren because one child timed out.
    assert_eq!(v["plan"]["outcome"].as_str(), Some("IndefiniteChildren"));
    let tasks = &v["plan"]["tasks"];
    assert_eq!(tasks[0]["outcome"].as_str(), Some("Success"));
    assert_eq!(tasks[1]["outcome"].as_str(), Some("Timeout"));
}

#[test]
fn test_gv_read_stats_basic() {
    let netlist = r#"module top(a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire w;
  INV i0 (.A(a), .Y(w));
  INV i1 (.A(w), .Y(y));
endmodule
"#;
    let temp_dir = tempfile::tempdir().unwrap();
    let gv_path = temp_dir.path().join("test.gv");
    std::fs::write(&gv_path, netlist).unwrap();

    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("gv-read-stats")
        .arg(gv_path.to_str().unwrap())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Instances: 2"), "{}", stdout);
    assert!(stdout.contains("INV"), "{}", stdout);
}

#[cfg(feature = "enable-fake-task")]
#[test]
fn test_prover_any_with_timeout_indefinite() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path();

    // any(fail within timeout, timeout)
    let plan = r#"{
      "kind": "any",
      "tasks": [
        { "kind": "fake", "delay_ms": 10, "success": false,  "stdout_len": 0, "stderr_len": 0},
        { "kind": "fake", "success": true, "stdout_len": 0, "stderr_len": 0, "timeout_ms": 1 }
      ]
    }"#;
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let json_path = dir.join("prover_all_timeout.json");
    let output = std::process::Command::new(driver)
        .arg("prover")
        .arg("--cores")
        .arg("2")
        .arg("--plan_json_file")
        .arg(plan_path.to_str().unwrap())
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .current_dir(dir)
        .output()
        .unwrap();

    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    assert!(
        !output.status.success(),
        "any(fail, timeout) should be overall non-success; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap();
    assert_eq!(v["success"].as_bool(), Some(false));
    // Group outcome becomes IndefiniteChildren because one child timed out.
    assert_eq!(v["plan"]["outcome"].as_str(), Some("IndefiniteChildren"));
    let tasks = &v["plan"]["tasks"];
    assert_eq!(tasks[0]["outcome"].as_str(), Some("Failed"));
    assert_eq!(tasks[1]["outcome"].as_str(), Some("Timeout"));
}

#[cfg(feature = "enable-fake-task")]
#[test]
fn test_prover_first_with_timeout_and_success() {
    let _ = env_logger::builder().is_test(true).try_init();
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path();

    // first(success after delay, timeout fast)
    // With 2 cores, the timeout will be observed first but should not resolve
    // the group (indefinite). The later success should resolve the group to
    // Success.
    let plan = r#"{
      "kind": "first",
      "tasks": [
        { "kind": "fake", "delay_ms": 100, "success": true,  "stdout_len": 0, "stderr_len": 0},
        { "kind": "fake", "success": true, "stdout_len": 0, "stderr_len": 0, "timeout_ms": 1 }
      ]
    }"#;
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let json_path = dir.join("prover_first_timeout_success.json");
    let output = std::process::Command::new(driver)
        .arg("prover")
        .arg("--cores")
        .arg("2")
        .arg("--plan_json_file")
        .arg(plan_path.to_str().unwrap())
        .arg("--output_json")
        .arg(json_path.to_str().unwrap())
        .current_dir(dir)
        .output()
        .unwrap();

    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    assert!(
        output.status.success(),
        "first(success, timeout) should be overall success; stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let json_str = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap();
    assert_eq!(v["success"].as_bool(), Some(true));
    assert_eq!(v["plan"]["outcome"].as_str(), Some("Success"));
    let tasks = &v["plan"]["tasks"];
    assert_eq!(tasks[0]["outcome"].as_str(), Some("Success"));
    assert_eq!(tasks[1]["outcome"].as_str(), Some("Timeout"));
}
