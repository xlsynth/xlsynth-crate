// SPDX-License-Identifier: Apache-2.0

use multithread::validate_all_threads_compute_add1;
use xlsynth::DslxConvertOptions;
use xlsynth_pir::ir::Type;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::{FnBuilder, IrValue};

// Show that the API works even if the Rust (caller) program is using an
// alternative allocator.
extern crate mimalloc;
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod multithread;

/// Loads DSLX code from a relative path in the crate and invokes the
/// (no-parameter) function named `func` inside of it.
fn load_and_invoke(
    dslx_relpath: &str,
    func: &str,
) -> Result<xlsynth::XlsIrValue, Box<dyn std::error::Error>> {
    let dslx_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(dslx_relpath);
    log::info!("reading DSLX program from: {:?}", dslx_path);
    let dslx = std::fs::read_to_string(&dslx_path)?;

    // Convert the DSLX to IR.
    let result =
        xlsynth::convert_dslx_to_ir(&dslx, dslx_path.as_path(), &DslxConvertOptions::default())?;
    for warning in result.warnings {
        log::warn!(
            "DSLX warning for {}: {}",
            dslx_path.to_str().unwrap(),
            warning
        );
    }
    let package = result.ir;
    let dslx_module_name = dslx_path.file_stem().unwrap().to_str().unwrap();

    // Determine what the mangled name is in the converted IR package.
    let mangled = xlsynth::mangle_dslx_name(dslx_module_name, func)?;

    // Extract the IR function from the package.
    let f = package.get_function(&mangled)?;
    log::info!("function {} type: {}", f.get_name(), f.get_type()?);
    // Invoke it with no arguments.
    let result = f.interpret(&[])?;
    // Return the result.
    Ok(result)
}

/// Validates the "meaning of life" value that comes from the `mol` function in
/// `sample.x`.
fn validate_mol() -> Result<(), Box<dyn std::error::Error>> {
    let file = "src/sample.x";
    let result = load_and_invoke(file, "mol")?;
    let want = xlsynth::XlsIrValue::parse_typed("bits[32]:42")?;
    assert_eq!(result, want);
    Ok(())
}

fn validate_use_popcount() -> Result<(), Box<dyn std::error::Error>> {
    let file = "src/sample_with_stdlib.x";
    let result = load_and_invoke(file, "use_popcount")?;
    let want = xlsynth::XlsIrValue::parse_typed("bits[32]:11")?;
    assert_eq!(result, want);
    Ok(())
}

fn validate_fail() -> Result<(), Box<dyn std::error::Error>> {
    let file = "src/sample_with_fail.x";
    let result = load_and_invoke(file, "always_fail");
    match result {
        Ok(_) => Err(Box::new(xlsynth::XlsynthError(
            "expected function call to fail".to_string(),
        ))),
        Err(e) => {
            if e.to_string()
                .contains("ABORTED: Assertion failure via fail!")
            {
                Ok(())
            } else {
                Err(Box::new(xlsynth::XlsynthError(format!(
                    "got unexpected error: {e}"
                ))))
            }
        }
    }
}

fn validate_use() -> Result<(), Box<dyn std::error::Error>> {
    let file = "src/sample_with_use.x";
    let result = load_and_invoke(file, "main")?;
    let want = xlsynth::XlsIrValue::parse_typed("bits[32]:4294967295")?;
    assert_eq!(result, want);
    Ok(())
}

/// Builds and interprets a PIR function.
fn validate_add_invert_via_ir_builder() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = FnBuilder::new("add_invert");
    let a = builder.param("a", Type::Bits(1))?;
    let invert = builder.not(a)?;
    builder.set_name(invert, "invert")?;
    let function = builder.build(invert)?;

    let false_value = IrValue::make_ubits(1, 0)?;
    let true_value = IrValue::make_ubits(1, 1)?;
    for (input, expected) in [(&false_value, &true_value), (&true_value, &false_value)] {
        let FnEvalResult::Success(result) = eval_fn(&function, std::slice::from_ref(input)) else {
            return Err("PIR evaluation unexpectedly failed".into());
        };
        assert_eq!(&result.value, expected);
    }

    Ok(())
}

fn main() {
    let _ = env_logger::try_init();
    let mol_result = validate_mol();
    let popcount_result = validate_use_popcount();
    println!(
        "meaning-of-life validation result: {:?} {:?}",
        mol_result, popcount_result
    );

    validate_fail().expect("failing function invocation should work as expected");
    println!("fail-function validation complete");

    validate_all_threads_compute_add1();
    println!("multi-threaded validation complete");

    validate_use().expect("use validation should work");
    println!("use validation complete");

    validate_add_invert_via_ir_builder().expect("add_invert validation should work");
    println!("add_invert validation complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_mol() {
        let _ = env_logger::try_init();
        validate_mol().expect("validation should succeed");
    }

    #[test]
    fn test_validate_use_popcount() {
        let _ = env_logger::try_init();
        validate_use_popcount().expect("validation should succeed");
    }

    #[test]
    fn test_validate_fail() {
        let _ = env_logger::try_init();
        validate_fail().expect("validation should succeed");
    }

    #[test]
    // TODO(meheff): Re-enable when https://github.com/google/xls/issues/2876 is fixed.
    #[ignore]
    fn test_validate_use() {
        let _ = env_logger::try_init();
        validate_use().expect("validation should succeed");
    }

    #[test]
    fn test_validate_add_invert_via_ir_builder() {
        let _ = env_logger::try_init();
        validate_add_invert_via_ir_builder().expect("validation should succeed");
    }
}
