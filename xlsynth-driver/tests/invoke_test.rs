// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use test_case::test_case;

fn add_tool_path_value(toolchain_toml_contents: &str) -> String {
    let tool_path = std::env::var("XLSYNTH_TOOL_PATH").unwrap();
    format!(
        "{}
tool_path = \"{}\"",
        toolchain_toml_contents, tool_path
    )
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
        .arg("--dslx_input_file")
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
        .expect("Failed to run xlsynth-driver");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{}", stdout);
    test_helpers::assert_valid_sv(&stdout);

    let golden_path =
        std::path::Path::new("tests/test_dslx2pipeline_with_update_of_1d_array.golden.sv");
    let golden_sv = std::fs::read_to_string(golden_path).unwrap();
    assert_eq!(stdout, golden_sv);
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
        .expect("Failed to run xlsynth-driver");

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
        .expect("Failed to run xlsynth-driver");

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
        .expect("Failed to run xlsynth-driver");

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
