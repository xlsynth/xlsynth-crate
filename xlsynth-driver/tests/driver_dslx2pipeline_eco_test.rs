// SPDX-License-Identifier: Apache-2.0

use pretty_assertions::assert_eq;
use std::process::Command;

fn add_tool_path_value(toolchain_toml_contents: &str) -> String {
    let tool_path =
        std::env::var("XLSYNTH_TOOLS").expect("XLSYNTH_TOOLS environment variable must be set");
    format!("{}\ntool_path = \"{}\"", toolchain_toml_contents, tool_path)
}

fn differ_in_one_line(a: &str, b: &str) -> bool {
    let a_lines: Vec<_> = a.lines().collect();
    let b_lines: Vec<_> = b.lines().collect();

    if a_lines.len() != b_lines.len() {
        return false;
    }

    let mut diffs = 0;
    for (la, lb) in a_lines.iter().zip(b_lines.iter()) {
        if la != lb {
            diffs += 1;
            if diffs > 1 {
                return false;
            }
        }
    }
    diffs == 1
}

#[derive(Debug)]
struct EcoDriverOutputs {
    baseline_verilog: String,
    baseline_from_eco_verilog: String,
    eco_verilog: String,
    edits: String,
}

/// Runs the baseline `dslx2pipeline` and the `dslx2pipeline-eco` flow for
/// tests.
///
/// - `baseline_dslx`: DSLX program for the baseline build.
/// - `changed_dslx`: DSLX program with changes for the ECO flow.
/// - `driver_args`: additional CLI args to pass to both driver invocations
///   (e.g. pipeline/delay flags, `--dslx_top=...`, `--module_name ...`).
///
/// Returns the baseline Verilog (stdout from baseline), the baseline Verilog
/// written by the ECO run, and the ECO Verilog (stdout from ECO run).
fn run_dslx2pipeline_and_eco(
    baseline_dslx: &str,
    changed_dslx: &str,
    driver_args: &[&str],
) -> Result<EcoDriverOutputs, String> {
    let _ = env_logger::builder().is_test(true).try_init();

    let keep_temps = std::env::var("KEEP_TEMPS").as_deref() == Ok("1");
    // Temp directory for this test.
    let mut temp_dir = tempfile::Builder::new()
        .prefix("dslx2pipeline_eco_test.")
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

    // Baseline DSLX: written to temp_dir/baseline_source/source.x
    let baseline_subdir = temp_dir.path().join("baseline_source");
    std::fs::create_dir_all(&baseline_subdir).unwrap();
    let baseline_path = baseline_subdir.join("source.x");
    std::fs::write(&baseline_path, baseline_dslx).unwrap();

    // Modified DSLX: written to temp_dir/source.x
    let modified_path = temp_dir.path().join("source.x");
    std::fs::write(&modified_path, changed_dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let baseline_unopt_ir = temp_dir.path().join("baseline.unopt.ir");

    // Run baseline: `dslx2pipeline` to capture unoptimized IR and baseline Verilog
    // on stdout.
    let mut baseline_cmd = Command::new(driver);
    baseline_cmd
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("dslx2pipeline")
        .arg("--dslx_input_file")
        .arg(baseline_path.to_str().unwrap())
        .arg("--output_unopt_ir")
        .arg(baseline_unopt_ir.to_str().unwrap())
        .arg(format!(
            "--keep_temps={}",
            if keep_temps { "true" } else { "false" }
        ));
    for a in driver_args {
        baseline_cmd.arg(a);
    }
    let out_baseline = baseline_cmd.output().unwrap();
    if !out_baseline.status.success() {
        return Err(String::from_utf8_lossy(&out_baseline.stderr).into_owned());
    }
    let baseline_verilog = String::from_utf8_lossy(&out_baseline.stdout).into_owned();

    // Run ECO: `dslx2pipeline-eco` using the baseline unoptimized IR and the
    // modified DSLX.
    let baseline_sv_from_eco_path = temp_dir.path().join("baseline_from_eco.sv");
    let edits_debug_out_path = temp_dir.path().join("edits.txt");

    let mut eco_cmd = Command::new(driver);
    eco_cmd
        .arg("--toolchain")
        .arg(toolchain_toml.to_str().unwrap())
        .arg("dslx2pipeline-eco")
        .arg("--dslx_input_file")
        .arg(modified_path.to_str().unwrap())
        .arg("--baseline_unopt_ir")
        .arg(baseline_unopt_ir.to_str().unwrap())
        .arg("--output_baseline_verilog_path")
        .arg(baseline_sv_from_eco_path.to_str().unwrap())
        .arg("--edits_debug_out")
        .arg(edits_debug_out_path.to_str().unwrap())
        .arg(format!(
            "--keep_temps={}",
            if keep_temps { "true" } else { "false" }
        ));
    for a in driver_args {
        eco_cmd.arg(a);
    }
    let out_eco = eco_cmd.output().unwrap();
    if !out_eco.status.success() {
        return Err(String::from_utf8_lossy(&out_eco.stderr).into_owned());
    }
    let eco_verilog = String::from_utf8_lossy(&out_eco.stdout).into_owned();

    let baseline_from_eco_verilog =
        std::fs::read_to_string(&baseline_sv_from_eco_path).expect("read baseline_from_eco.sv");

    let edits = std::fs::read_to_string(&edits_debug_out_path).expect("read edits.txt");

    Ok(EcoDriverOutputs {
        baseline_verilog,
        baseline_from_eco_verilog,
        eco_verilog,
        edits,
    })
}

#[test]
fn test_dslx2pipeline_eco_basic() {
    let baseline_dslx = "fn main(x: u32) -> u32 { x + u32:1 }\n";
    let modified_dslx = "fn main(x: u32) -> u32 { -x + u32:1 }\n";
    let args = vec![
        "--pipeline_stages=1",
        "--delay_model=asap7",
        "--flop_inputs=false",
        "--flop_outputs=false",
        "--dslx_top=main",
    ];
    let outputs = run_dslx2pipeline_and_eco(baseline_dslx, modified_dslx, &args)
        .expect("dslx2pipeline and eco should succeed");

    // Expect exactly one-line difference between ECO stdout and baseline stdout.
    let eco_stdout = outputs.eco_verilog.clone();
    let baseline_stdout = outputs.baseline_verilog.clone();
    if !differ_in_one_line(&eco_stdout, &baseline_stdout) {
        // Fall back to a pretty diff to aid debugging.
        assert_eq!(
            eco_stdout, baseline_stdout,
            "expected exactly one-line difference"
        );
    }
    assert_eq!(
        outputs.baseline_from_eco_verilog, baseline_stdout,
        "baseline Verilog from eco does not match non-ECO baseline Verilog"
    );
    assert!(
        outputs.edits.contains("AddNode: neg"),
        "edits should contain AddNode: {}",
        outputs.edits
    );
    assert!(
        outputs.edits.contains("SubstituteOperand: add"),
        "edits should contain SubstituteOperand: {}",
        outputs.edits
    );
}

#[test]
fn test_dslx2pipeline_eco_module_name() {
    let baseline_dslx = "fn main(x: u32) -> u32 { x + u32:1 }\n";
    let modified_dslx = "fn main(x: u32) -> u32 { -x + u32:1 }\n";
    let args = vec![
        "--pipeline_stages=1",
        "--delay_model=asap7",
        "--flop_inputs=false",
        "--flop_outputs=false",
        "--dslx_top=main",
        "--module_name",
        "my_module",
    ];
    let outputs = run_dslx2pipeline_and_eco(baseline_dslx, modified_dslx, &args)
        .expect("dslx2pipeline and eco should succeed");

    // Expect exactly one-line difference between ECO stdout and baseline stdout.
    let eco_stdout = outputs.eco_verilog.clone();
    let baseline_stdout = outputs.baseline_verilog.clone();
    if !differ_in_one_line(&eco_stdout, &baseline_stdout) {
        // Fall back to a pretty diff to aid debugging.
        assert_eq!(
            eco_stdout, baseline_stdout,
            "expected exactly one-line difference"
        );
    }
    assert_eq!(
        outputs.baseline_from_eco_verilog, baseline_stdout,
        "baseline Verilog from eco does not match non-ECO baseline Verilog"
    );
    assert!(
        outputs
            .baseline_from_eco_verilog
            .contains("module my_module("),
        "baseline Verilog from eco should use module name: {}",
        outputs.baseline_from_eco_verilog
    );
}

#[test]
fn test_dslx2pipeline_eco_with_registers() {
    let baseline_dslx = "fn main(x: u32) -> u32 { x + u32:1 }\n";
    let modified_dslx = "fn main(x: u32) -> u32 { !x + u32:1 }\n";
    let args = vec![
        "--pipeline_stages=1",
        "--delay_model=asap7",
        "--flop_inputs=true",
        "--flop_outputs=false",
        "--dslx_top=main",
    ];
    match run_dslx2pipeline_and_eco(baseline_dslx, modified_dslx, &args) {
        Ok(_) => panic!("expected ECO run to fail due to registers"),
        Err(e) => assert!(
            e.contains("ECOs not supported on designs with registers"),
            "unexpected error: {}",
            e
        ),
    }
}
