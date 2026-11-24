// SPDX-License-Identifier: Apache-2.0

//! The optional legalization command produces Block IR consumable by block2fn.

use std::path::Path;
use std::process::{Command, Output};

use xlsynth_pir::IrValue;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;

const FEEDBACK: &str = r#"package feedback
block leaf(d: bits[4], q: bits[4]) {
  d: bits[4] = input_port(name=d, id=1)
  q: () = output_port(d, name=q, id=2)
}
top block root(seed: bits[1], result: bits[4]) {
  instantiation u(block=leaf, kind=block)
  seed: bits[1] = input_port(name=seed, id=3)
  q: bits[4] = instantiation_output(instantiation=u, port_name=q, id=4)
  low: bits[3] = bit_slice(q, start=0, width=3, id=5)
  driver: bits[4] = concat(low, seed, id=6)
  drive: () = instantiation_input(driver, instantiation=u, port_name=d, id=7)
  result: () = output_port(q, name=result, id=8)
}
"#;

/// Runs one driver command without inheriting noisy logging settings.
fn invoke(directory: &Path, command: &str, arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(directory)
        .env_remove("RUST_LOG")
        .arg(command)
        .args(arguments)
        .output()
        .unwrap()
}

/// Returns stdout from a successful command.
fn stdout(output: Output) -> String {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn automatic_and_explicit_legalization_produce_equivalent_functions() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("input.ir"), FEEDBACK).unwrap();
    let direct = stdout(invoke(dir.path(), "block2fn", &["--block_ir", "input.ir"]));
    let direct = Parser::new(&direct).parse_and_verify_package().unwrap();
    let output = stdout(invoke(dir.path(), "block-legalize", &["input.ir"]));
    let legalized = Parser::new(&output).parse_and_verify_package().unwrap();
    assert!(legalized.get_top_block().unwrap().instantiations.is_empty());
    std::fs::write(dir.path().join("legalized.ir"), output).unwrap();
    let output = stdout(invoke(
        dir.path(),
        "block2fn",
        &["--block_ir", "legalized.ir"],
    ));
    let package = Parser::new(&output).parse_and_verify_package().unwrap();
    let function = package.get_top_fn().unwrap();
    for seed in 0..=1 {
        let input = IrValue::make_ubits(1, seed).unwrap();
        let expected = IrValue::make_ubits(4, seed * 15).unwrap();
        for function in [direct.get_top_fn().unwrap(), function] {
            match eval_fn(function, &[input.clone()]) {
                FnEvalResult::Success(result) => assert_eq!(result.value, expected),
                FnEvalResult::Failure(error) => panic!("function evaluation failed: {error:?}"),
            }
        }
    }
}

#[test]
fn unmarked_multiblock_package_requires_and_honors_explicit_top() {
    let dir = tempfile::tempdir().unwrap();
    let unmarked = FEEDBACK.replace("top block root", "block root");
    std::fs::write(dir.path().join("input.ir"), unmarked).unwrap();
    let missing = invoke(dir.path(), "block-legalize", &["input.ir"]);
    assert_eq!(missing.status.code(), Some(1));
    assert!(missing.stdout.is_empty());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("--top"));

    let leaf = stdout(invoke(
        dir.path(),
        "block-legalize",
        &["input.ir", "--top", "leaf"],
    ));
    let package = Parser::new(&leaf).parse_and_verify_package().unwrap();
    assert_eq!(package.get_top_block().unwrap().name, "leaf");
}

#[test]
fn a_single_unmarked_block_is_selected_automatically() {
    let dir = tempfile::tempdir().unwrap();
    let source = FEEDBACK.split("top block root").next().unwrap();
    std::fs::write(dir.path().join("input.ir"), source).unwrap();
    let output = stdout(invoke(dir.path(), "block-legalize", &["input.ir"]));
    let package = Parser::new(&output).parse_and_verify_package().unwrap();
    assert_eq!(package.get_top_block().unwrap().name, "leaf");
}

#[test]
fn invalid_top_fails_without_output() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("input.ir"), FEEDBACK).unwrap();
    let result = invoke(
        dir.path(),
        "block-legalize",
        &["input.ir", "--top", "missing"],
    );
    assert_eq!(result.status.code(), Some(1));
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("missing"));
}
