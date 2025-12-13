// SPDX-License-Identifier: Apache-2.0

use xlsynth_test_helpers::compare_golden_text;

#[test]
fn test_dslx_stitch_pipeline_emits_unopt_and_opt_ir_packages() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = r#"fn foo_cycle0(x: u32, y: u32) -> (u32, u32) { (x + y, x) }
fn foo_cycle1(a: u32, b: u32) -> u32 { a - b }
"#;

    let temp_dir = tempfile::tempdir().unwrap();
    std::fs::write(temp_dir.path().join("foo.x"), dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out = std::process::Command::new(driver)
        .current_dir(temp_dir.path())
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg("foo.x")
        .arg("--dslx_top")
        .arg("foo")
        .arg("--output_unopt_ir")
        .arg("unopt.ir")
        .arg("--output_opt_ir")
        .arg("opt.ir")
        .output()
        .unwrap();

    assert!(
        out.status.success(),
        "expected success; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let unopt_text = std::fs::read_to_string(temp_dir.path().join("unopt.ir")).unwrap();
    compare_golden_text(
        &unopt_text,
        "tests/test_dslx_stitch_pipeline_output_unopt_ir.golden.ir",
    );

    let opt_text = std::fs::read_to_string(temp_dir.path().join("opt.ir")).unwrap();
    compare_golden_text(
        &opt_text,
        "tests/test_dslx_stitch_pipeline_output_opt_ir.golden.ir",
    );
}





