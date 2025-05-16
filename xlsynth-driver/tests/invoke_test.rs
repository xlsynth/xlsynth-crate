// SPDX-License-Identifier: Apache-2.0

//! Tests for invoking the xlsynth-driver CLI and its subcommands as a
//! subprocess.

use std::process::Command;

use test_case::test_case;

use pretty_assertions::assert_eq;

fn add_tool_path_value(toolchain_toml_contents: &str) -> String {
    let tool_path =
        std::env::var("XLSYNTH_TOOLS").expect("XLSYNTH_TOOLS environment variable must be set");
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
        .unwrap();

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
        .unwrap();

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
        .unwrap();

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
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    test_helpers::assert_valid_sv(&stdout);

    let golden_path =
        std::path::Path::new("tests/test_dslx2pipeline_with_update_of_1d_array.golden.sv");

    if std::env::var("XLSYNTH_UPDATE_GOLDEN").is_ok() {
        println!("INFO: Updating golden file: {}", golden_path.display());
        std::fs::write(golden_path, &stdout).expect("Failed to write golden file");
    } else {
        let golden_sv = std::fs::read_to_string(golden_path).expect("Failed to read golden file");
        assert_eq!(
            stdout, golden_sv,
            "Golden file mismatch. Run with XLSYNTH_UPDATE_GOLDEN=1 to update."
        );
    }
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
        .arg("--input_valid_signal=input_valid")
        .arg("--output_valid_signal=output_valid")
        .arg("--reset=rst")
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
    test_helpers::assert_valid_sv(&stdout);

    // Define the path for the new golden file
    let golden_path = std::path::Path::new("tests/test_dslx2pipeline_with_reset_signal.golden.sv");

    if std::env::var("XLSYNTH_UPDATE_GOLDEN").is_ok() {
        println!("INFO: Updating golden file: {}", golden_path.display());
        std::fs::write(golden_path, &stdout).expect("Failed to write golden file");
    } else {
        let golden_sv = std::fs::read_to_string(golden_path).expect("Failed to read golden file");
        assert_eq!(
            stdout, golden_sv,
            "Golden file mismatch. Run with XLSYNTH_UPDATE_GOLDEN=1 to update."
        );
    }
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
    assert!(stdout.contains("success: Verified equivalent"));
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
        stderr.contains(
            "xlsynth-driver: ir-equiv: failure: Verified NOT equivalent; results differ for input"
        ),
        "stderr: {:?}",
        stderr
    );
}

#[test]
fn test_ir2gates_determinism() {
    let _ = env_logger::try_init();
    log::info!("test_ir2gates_determinism");

    let dslx = "fn main(a: u32, b: u32) -> u32 { a + b * (a ^ b) }";

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

    let stdout = String::from_utf8_lossy(&ir2gates_output.stdout);
    // Try to parse as JSON
    let json: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("Output is not valid JSON");
    log::info!("json: {}", json);
    // Check standard stats
    assert_eq!(json["deepest_path"], 2);
    assert_eq!(json["fanout_histogram"].to_string(), "{\"1\":64}");
    assert_eq!(json["live_nodes"], 96);

    // Check expected values for this simple AND circuit
    use std::collections::HashMap;
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

// Test for ir-equiv subcommand using Boolector
#[cfg(feature = "has-boolector")]
#[test]
fn test_irequiv_subcommand_boolector_equivalent() {
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
        .arg("--boolector=true")
        .output()
        .expect("xlsynth-driver should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    log::info!("stdout: {}", stdout);
    log::info!("stderr: {}", stderr);
    assert!(
        output.status.success(),
        "Boolector ir-equiv should succeed; stderr: {}",
        stderr
    );
    assert!(
        stdout.contains("Boolector proved equivalence"),
        "stdout: {}",
        stdout
    );
}

/// Test for ir-equiv command using Boolector with different IR top entry
/// points.
#[cfg(feature = "has-boolector")]
#[test]
fn test_irequiv_subcommand_boolector_different_top_entry_points() {
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
        .arg("--boolector=true")
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
    assert!(stdout.contains("Boolector proved equivalence"));
}

#[test_case(true; "with_tool_path")]
#[test_case(false; "without_tool_path")]
fn test_toolchain_common_codegen_flags_resolve(use_tool_path: bool) {
    let _ = env_logger::builder().is_test(true).try_init();
    log::info!("test_toolchain_common_codegen_flags_resolve");
    let temp_dir = tempfile::tempdir().unwrap();

    // Write out toolchain configuration.
    let toolchain_toml = temp_dir.path().join("xlsynth-toolchain.toml");
    let mut toolchain_toml_contents = "[toolchain]\n".to_string();
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

    // Path to the golden file
    let golden_path =
        std::path::Path::new("tests/test_toolchain_common_codegen_flags_resolve.golden.sv");

    if std::env::var("XLSYNTH_UPDATE_GOLDEN").is_ok() {
        println!("INFO: Updating golden file: {}", golden_path.display());
        std::fs::write(golden_path, &stdout).expect("Failed to write golden file");
    } else {
        let golden_sv = std::fs::read_to_string(golden_path).expect("Failed to read golden file");
        assert_eq!(
            stdout, golden_sv,
            "Golden file mismatch. Run with XLSYNTH_UPDATE_GOLDEN=1 to update."
        );
    }
}
