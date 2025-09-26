// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

fn add_tool_path_value(toolchain_toml_contents: &str) -> String {
    let tool_path =
        std::env::var("XLSYNTH_TOOLS").expect("XLSYNTH_TOOLS environment variable must be set");
    format!("{}\ntool_path = \"{}\"", toolchain_toml_contents, tool_path)
}

#[test]
fn test_dslx2pipeline_eco_basic() {
    let _ = env_logger::builder().is_test(true).try_init();

    let keep_temps = std::env::var("KEEP_TEMPS").as_deref() == Ok("1");
    // Temp directory for this test.
    let mut temp_dir = tempfile::Builder::new()
        .prefix("dslx2pipeline_eco_test")
        .tempdir()
        .unwrap();
    if keep_temps {
        temp_dir.disable_cleanup(true);
        eprintln!("Working directory: {}", temp_dir.path().display());
    }

    // Write out toolchain configuration with a tool path.
    let toolchain_toml = temp_dir.path().join("xlsynth-toolchain.toml");
    let toolchain_toml_contents = add_tool_path_value("[toolchain]\n");
    std::fs::write(&toolchain_toml, toolchain_toml_contents).unwrap();

    // Baseline DSLX: add1 (written to temp_dir/baseline/source.x)
    let baseline_dslx = "fn main(x: u32) -> u32 { x + u32:1 }\n";
    let baseline_subdir = temp_dir.path().join("baseline_source");
    std::fs::create_dir_all(&baseline_subdir).unwrap();
    let baseline_path = baseline_subdir.join("source.x");
    std::fs::write(&baseline_path, baseline_dslx).unwrap();

    // Modified DSLX: add2 (slightly different), written to temp_dir/source.x
    let modified_dslx = "fn main(x: u32) -> u32 { x + u32:2 }\n";
    let modified_path = temp_dir.path().join("source.x");
    std::fs::write(&modified_path, modified_dslx).unwrap();

    // 1) Run dslx2pipeline on the baseline to capture unoptimized IR.
    let baseline_unopt_ir = temp_dir.path().join("baseline.unopt.ir");
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out_baseline = Command::new(driver)
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("dslx2pipeline")
        .arg("--pipeline_stages=1")
        .arg("--delay_model=asap7")
        .arg("--flop_inputs=false")
        .arg("--flop_outputs=false")
        .arg("--dslx_input_file")
        .arg(baseline_path.to_str().unwrap())
        .arg("--dslx_top=main")
        .arg("--output_unopt_ir")
        .arg(baseline_unopt_ir.to_str().unwrap())
        .arg(format!(
            "--keep_temps={}",
            if keep_temps { "true" } else { "false" }
        ))
        .output()
        .unwrap();
    assert!(
        out_baseline.status.success(),
        "dslx2pipeline (baseline) failed (status={}):\nstdout:{}\nstderr:{}",
        out_baseline.status,
        String::from_utf8_lossy(&out_baseline.stdout),
        String::from_utf8_lossy(&out_baseline.stderr)
    );

    // 2) Run dslx2pipeline-eco using the baseline unoptimized IR and the modified
    //    DSLX.
    let out_eco = Command::new(driver)
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("dslx2pipeline-eco")
        .arg("--pipeline_stages")
        .arg("1")
        .arg("--delay_model")
        .arg("asap7")
        .arg("--flop_inputs=false")
        .arg("--flop_outputs=false")
        .arg("--dslx_input_file")
        .arg(modified_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("main")
        .arg("--baseline_unopt_ir")
        .arg(baseline_unopt_ir.to_str().unwrap())
        .arg(format!(
            "--keep_temps={}",
            if keep_temps { "true" } else { "false" }
        ))
        .output()
        .unwrap();

    assert!(
        out_eco.status.success(),
        "dslx2pipeline-eco failed (status={}):\nstdout:{}\nstderr:{}",
        out_eco.status,
        String::from_utf8_lossy(&out_eco.stdout),
        String::from_utf8_lossy(&out_eco.stderr)
    );
}
