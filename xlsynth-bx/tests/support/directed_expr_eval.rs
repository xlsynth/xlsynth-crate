// SPDX-License-Identifier: Apache-2.0

//! Directed BX expression tests can select an external DSLX compiler. The
//! fuzz target always uses the pinned in-process compiler.

use std::{env, ffi::OsStr, fs, path::Path, process::Command};
use tempfile::tempdir;
use xlsynth::{IrFunction, IrPackage};
use xlsynth_bx::{check_module, parse_module};
use xlsynth_pir::IrValue;

use super::expr_eval::{
    ExpressionEvaluator, bx_source, compile_bundled_dslx, compile_bundled_module, dslx_source,
};

/// Compile with the selected reference. Only ordinary source rejection by
/// the external compiler returns Err; a broken tool or invalid IR fails the
/// test.
pub fn compile_dslx(params: &str, result: &str, body: &str) -> Result<IrFunction, String> {
    let Some(converter) = env::var_os("BX_DSLX_IR_CONVERTER") else {
        return compile_bundled_dslx(params, result, body);
    };
    compile_external(&converter, &dslx_source(params, result, body), &[])
}

/// Compile a complete reference module, allowing its imports to come from a
/// fixture.
pub fn compile_dslx_module(source: &str, paths: &[&Path]) -> Result<IrFunction, String> {
    match env::var_os("BX_DSLX_IR_CONVERTER") {
        Some(converter) => compile_external(&converter, source, paths),
        None => compile_bundled_module(source, paths),
    }
}

fn compile_external(
    converter: &OsStr,
    source: &str,
    paths: &[&Path],
) -> Result<IrFunction, String> {
    let converter = Path::new(converter);
    let dir = tempdir().expect("create DSLX reference directory");
    let path = dir.path().join("expr_eval.x");
    fs::write(&path, source).expect("write DSLX reference source");
    let mut command = Command::new(converter);
    command.arg("--warnings_as_errors=false");
    if !paths.is_empty() {
        command
            .arg("--dslx_path")
            .arg(env::join_paths(paths).expect("valid DSLX import paths"));
    }
    if let Some(stdlib) = env::var_os("DSLX_STDLIB_PATH") {
        assert!(
            Path::new(&stdlib).is_dir(),
            "invalid DSLX_STDLIB_PATH: {stdlib:?}"
        );
        command.arg("--dslx_stdlib_path").arg(stdlib);
    }
    let output = command
        .arg(path)
        .output()
        .unwrap_or_else(|e| panic!("could not launch {}: {e}", converter.display()));
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        assert_eq!(
            output.status.code(),
            Some(1),
            "{} failed with {}\n{stderr}\n{source}",
            converter.display(),
            output.status
        );
        return Err(stderr.into_owned());
    }
    let ir = String::from_utf8(output.stdout).expect("DSLX converter output must be UTF-8");
    let package = IrPackage::parse_ir(&ir, None).unwrap_or_else(|e| {
        panic!(
            "{} emitted invalid IR: {e}\n{source}\n{ir}",
            converter.display()
        )
    });
    Ok(package
        .get_function("__expr_eval__sample")
        .expect("DSLX output must define sample"))
}

/// Compare the frontends on each concrete input and check the hand-computed
/// value.
pub fn assert_expr_matches_dslx(
    params: &str,
    result: &str,
    body: &str,
    cases: impl IntoIterator<Item = (Vec<IrValue>, IrValue)>,
) {
    let evaluator = ExpressionEvaluator::compile(params, result, body, compile_dslx).unwrap();
    for (args, expected) in cases {
        let actual = evaluator
            .compare(&args)
            .unwrap_or_else(|error| panic!("{error}\n{body}"));
        assert_eq!(
            actual,
            expected,
            "{body} with {}",
            IrValue::make_tuple(&args)
        );
    }
}

/// Require a BX frontend error and a clean DSLX compilation error.
pub fn assert_expr_rejected_by_both(params: &str, result: &str, body: &str) {
    let block = bx_source(params, result, body);
    let bx = parse_module(&block).and_then(|module| check_module(&module));
    let dslx = compile_dslx(params, result, body);
    assert!(bx.is_err(), "BX accepted {block}");
    assert!(
        dslx.is_err(),
        "DSLX accepted {}",
        dslx_source(params, result, body)
    );
}
