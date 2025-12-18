// SPDX-License-Identifier: Apache-2.0

//! CLI validation tests for `dslx-stitch-pipeline` flags.

use flate2::read::GzDecoder;
use std::io::Read;
use tar::Archive;
use xlsynth::IrPackage;

#[test]
fn test_mutual_exclusion_dslx_top_and_stages() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Minimal DSLX with one stage function.
    let dslx = "fn foo_cycle0(x: u32) -> u32 { x }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("foo.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out = std::process::Command::new(driver)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("foo")
        .arg("--stages")
        .arg("foo_cycle0")
        .output()
        .unwrap();

    assert!(
        !out.status.success(),
        "expected failure for mutually-exclusive flags"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    let expected = "the argument '--dslx_top <DSLX_TOP>' cannot be used with '--stages <CSV>'";
    assert!(
        stderr.contains(expected),
        "expected phrase not found. stderr: {}",
        stderr
    );
}

#[test]
fn test_stages_requires_output_module_name() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = "fn foo_cycle0(x: u32) -> u32 { x }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("foo.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out = std::process::Command::new(driver)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--stages")
        .arg("foo_cycle0")
        .output()
        .unwrap();

    assert!(
        !out.status.success(),
        "expected failure when --stages lacks --output_module_name"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--output_module_name is required"),
        "unexpected stderr: {}",
        stderr
    );
}

#[test]
fn test_output_module_name_controls_wrapper_name() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = "fn foo_cycle0(x: u32) -> u32 { x }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("foo.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out = std::process::Command::new(driver)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--stages")
        .arg("foo_cycle0")
        .arg("--output_module_name")
        .arg("my_wrap")
        .output()
        .unwrap();

    assert!(
        out.status.success(),
        "expected success with explicit wrapper name; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Wrapper module name should be the provided value.
    assert!(
        stdout.contains("module my_wrap"),
        "wrapper module name not found in output: {}",
        stdout
    );
}

#[test]
fn test_output_ir_tgz_emits_per_stage_ir_packages() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dslx = "fn foo_cycle0(x: u32) -> u32 { x }\nfn foo_cycle1(x: u32) -> u32 { x }";
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("foo.x");
    std::fs::write(&dslx_path, dslx).unwrap();

    let unopt_tgz = temp_dir.path().join("unopt_ir.tar.gz");
    let opt_tgz = temp_dir.path().join("opt_ir.tar.gz");

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let out = std::process::Command::new(driver)
        .arg("dslx-stitch-pipeline")
        .arg("--dslx_input_file")
        .arg(dslx_path.to_str().unwrap())
        .arg("--dslx_top")
        .arg("foo")
        .arg("--output_unopt_ir_tgz")
        .arg(unopt_tgz.to_str().unwrap())
        .arg("--output_opt_ir_tgz")
        .arg(opt_tgz.to_str().unwrap())
        .output()
        .unwrap();

    assert!(
        out.status.success(),
        "expected success; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    fn read_tgz_entries(path: &std::path::Path) -> Vec<(String, String)> {
        let bytes = std::fs::read(path).expect("read tar.gz");
        let decoder = GzDecoder::new(&bytes[..]);
        let mut archive = Archive::new(decoder);

        let mut out: Vec<(String, String)> = Vec::new();
        for e in archive.entries().expect("tar entries") {
            let mut entry = e.expect("tar entry");
            let p = entry
                .path()
                .expect("tar path")
                .to_string_lossy()
                .to_string();
            let mut s = String::new();
            entry.read_to_string(&mut s).expect("read entry text");
            out.push((p, s));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    let unopt_entries = read_tgz_entries(&unopt_tgz);
    let opt_entries = read_tgz_entries(&opt_tgz);

    assert_eq!(
        unopt_entries
            .iter()
            .map(|(p, _)| p.as_str())
            .collect::<Vec<_>>(),
        vec!["unopt.ir"]
    );
    assert_eq!(
        opt_entries
            .iter()
            .map(|(p, _)| p.as_str())
            .collect::<Vec<_>>(),
        vec!["foo_cycle0.ir", "foo_cycle1.ir"]
    );

    for (_p, ir_text) in unopt_entries {
        let pkg = IrPackage::parse_ir(&ir_text, None).expect("IR parses as a package");
        let funcs = pkg.get_functions().expect("functions");
        assert_eq!(
            funcs.len(),
            2,
            "expected unoptimized IR package to contain all stage functions"
        );
    }
    for (_p, ir_text) in opt_entries {
        let pkg = IrPackage::parse_ir(&ir_text, None).expect("IR parses as a package");
        let funcs = pkg.get_functions().expect("functions");
        assert_eq!(
            funcs.len(),
            1,
            "expected one function per optimized IR package"
        );
    }
}
