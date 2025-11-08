// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use pretty_assertions::assert_eq;

use xlsynth_pir::ir_fuzz::{FuzzSample, generate_ir_fn};
use xlsynth_pir::ir_parser::Parser;

fn run_codegen_block_ir_to_string(
    input_file: &std::path::Path,
    top: &str,
    tool_path: &str,
    output_block_ir_path: &std::path::Path,
    pipeline_generator: bool,
) -> String {
    let codegen_main_path = format!("{}/{}", tool_path, "codegen_main");
    assert!(
        std::path::Path::new(&codegen_main_path).exists(),
        "codegen_main not found at {} (set XLS_TOOLCHAIN_PATH correctly)",
        codegen_main_path
    );

    let mut command = std::process::Command::new(codegen_main_path);
    command
        .arg(input_file)
        .arg("--delay_model")
        .arg("unit")
        .arg("--top")
        .arg(top)
        .arg("--output_block_ir_path")
        .arg(output_block_ir_path);
    if pipeline_generator {
        command
            .arg("--generator=pipeline")
            .arg("--pipeline_stages=1")
            .arg("--flop_inputs=false")
            .arg("--flop_outputs=false");
    } else {
        command.arg("--generator=combinational");
    }

    let output = command.output().expect("failed to execute codegen_main");
    assert!(
        output.status.success(),
        "codegen_main failed: status={} stderr=\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(output_block_ir_path)
        .expect("reading output_block_ir_path should succeed")
}

fuzz_target!(|sample_input: (FuzzSample, bool)| {
    let (sample, pipeline_generator) = sample_input;
    // Toolchain path must be provided via environment for this fuzz target.
    let tool_path =
        std::env::var("XLSYNTH_TOOLS").expect("XLSYNTH_TOOLS must be set for fuzz_block_roundtrip");

    // 1) Build IR package with a single function from the sample.
    let mut pkg = xlsynth::IrPackage::new("fuzz_pkg").expect("IrPackage::new should succeed");
    let fn_name = "fuzz_test";
    if generate_ir_fn(sample.ops.clone(), &mut pkg, None).is_err() {
        return;
    }

    // 2) Write IR to a temporary directory with a recognizable prefix.
    let tmpdir = tempfile::Builder::new()
        .prefix("fuzz_block_roundtrip")
        .tempdir()
        .expect("TempDir::new should succeed");

    let ir_path = tmpdir.path().join("input.ir");
    std::fs::write(&ir_path, format!("{}", pkg)).expect("write IR file should succeed");

    // 3) Run codegen_main to emit combinational block IR for the function top.
    let block_path = tmpdir.path().join("block.ir");
    let generated_ir_text = run_codegen_block_ir_to_string(
        &ir_path,
        fn_name,
        &tool_path,
        &block_path,
        pipeline_generator,
    );

    // 4) Roundtrip the IR through the parser and emitter.
    let pkg1 = Parser::new(&generated_ir_text)
        .parse_and_validate_package()
        .expect("parse and validate package should succeed");
    let roundtrip_ir_text = pkg1.to_string();
    let roundtrip_path = tmpdir.path().join("roundtrip.ir");
    std::fs::write(&roundtrip_path, &roundtrip_ir_text).expect("write roundtrip.ir should succeed");

    assert_eq!(generated_ir_text, roundtrip_ir_text);
});
