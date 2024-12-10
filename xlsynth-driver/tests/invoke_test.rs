// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

/// Simple test that converts a DSLX module with an enum into a SV definition.
#[test]
fn test_dslx2sv_types_subcommand() {
    let dslx = "enum OpType : u2 { READ = 0, WRITE = 1 }";
    // Make a temporary file to hold the DSLX code.
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("my_module.x");
    std::fs::write(&dslx_path, dslx).unwrap();
    // Run the dslx2sv subcommand.
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("dslx2sv-types")
        .arg(dslx_path.to_str().unwrap())
        .output()
        .expect("Failed to run xlsynth-driver");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    test_helpers::assert_valid_sv(&stdout);
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
        .arg(dslx_path.to_str().unwrap())
        .output()
        .expect("Failed to run xlsynth-driver");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    test_helpers::assert_valid_sv(&stdout);
    assert_eq!(
        stdout.trim(),
        "localparam bit unsigned [31:0] Count = 32'h00000018;

localparam bit unsigned [31:0] Width = 32'h00000005;

typedef struct packed {
    logic [4:0] data;
} my_struct_t;"
    );
}

/// Tests that we can point at a xlsynth-toolchain.toml file to get a DSLX_PATH
/// value.
#[test]
fn test_dslx2ir_with_toolchain_toml() {
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
dslx_stdlib_path = "{}"
dslx_path = []
"#,
        stdlib_dir.to_str().unwrap()
    );
    std::fs::write(&toolchain_path, toolchain_toml).unwrap();

    // Run the dslx2ir subcommand.
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("--toolchain")
        .arg(toolchain_path.to_str().unwrap())
        .arg("dslx2ir")
        .arg(client_path.to_str().unwrap())
        .output()
        .expect("Failed to run xlsynth-driver");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains(
        "fn __std__popcount(x: bits[32] id=1) -> bits[32] {
  ret x: bits[32] = param(name=x, id=1)
}"
    ))
}
