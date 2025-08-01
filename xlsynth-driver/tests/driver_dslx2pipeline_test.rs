// SPDX-License-Identifier: Apache-2.0

use std::process::Command;
use test_case::test_case;

fn add_tool_path_value(toolchain_toml_contents: &str) -> String {
    let tool_path =
        std::env::var("XLSYNTH_TOOLS").expect("XLSYNTH_TOOLS environment variable must be set");
    format!(
        "{}
tool_path = \"{}\"",
        toolchain_toml_contents, tool_path
    )
}

/// Ensures that the `--output_unopt_ir` and `--output_opt_ir` flags on the
/// `dslx2pipeline` subcommand correctly write the requested artifacts and that
/// the captured optimized IR is indeed an optimization of the unoptimized IR.
#[test_case(true; "with_tool_path")]
#[test_case(false; "without_tool_path")]
fn test_dslx2pipeline_output_ir_files(use_tool_path: bool) {
    let _ = env_logger::builder().is_test(true).try_init();

    let temp_dir = tempfile::tempdir().unwrap();

    // Write out toolchain configuration.
    let toolchain_toml = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = "[toolchain]\n".to_string();
    let toolchain_toml_contents = if use_tool_path {
        add_tool_path_value(&toolchain_toml_contents)
    } else {
        toolchain_toml_contents
    };
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    // Simple DSLX function: add1.
    let dslx_source = "fn main(x: u32) -> u32 { x + u32:1 }";
    let tmp_dir = tempfile::tempdir().unwrap();
    let dslx_path = tmp_dir.path().join("add1.x");
    std::fs::write(&dslx_path, dslx_source).unwrap();

    // Paths for the IR artifacts.
    let unopt_path = tmp_dir.path().join("unopt.ir");
    let opt_path = tmp_dir.path().join("opt.ir");

    // Run the driver with the new flags.
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(driver)
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("dslx2pipeline")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--delay_model")
        .arg("asap7")
        .arg("--flop_inputs=false")
        .arg("--flop_outputs=false")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .arg("--output_unopt_ir")
        .arg(unopt_path.to_str().unwrap())
        .arg("--output_opt_ir")
        .arg(opt_path.to_str().unwrap())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "dslx2pipeline failed (status={}):\nstdout:{}\nstderr:{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // Validate that files exist and are non-empty.
    let unopt_ir_text =
        std::fs::read_to_string(&unopt_path).expect("unoptimized IR file should have been written");
    let opt_ir_text =
        std::fs::read_to_string(&opt_path).expect("optimized IR file should have been written");
    assert!(!unopt_ir_text.trim().is_empty());
    assert!(!opt_ir_text.trim().is_empty());

    // Use `xlsynth-driver ir-equiv` to prove the unoptimized and optimized IRs are
    // equivalent.
    let module_name = xlsynth::dslx_path_to_module_name(&dslx_path).unwrap();
    let ir_top = xlsynth::mangle_dslx_name(module_name, "main").unwrap();

    if !cfg!(feature = "has-bitwuzla") && !use_tool_path {
        println!("Skipping ir-equiv test because bitwuzla is not available and toolchain path is not used");
        return;
    }

    let equiv_output = Command::new(driver)
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("ir-equiv")
        .arg("--solver")
        .arg(if use_tool_path {
            "toolchain"
        } else {
            "bitwuzla"
        })
        .arg(unopt_path.to_str().unwrap())
        .arg(opt_path.to_str().unwrap())
        .arg("--top")
        .arg(&ir_top)
        .output()
        .unwrap();

    assert!(
        equiv_output.status.success(),
        "ir-equiv failed (status={}):\nstdout:{}\nstderr:{}",
        equiv_output.status,
        String::from_utf8_lossy(&equiv_output.stdout),
        String::from_utf8_lossy(&equiv_output.stderr)
    );
}
