// SPDX-License-Identifier: Apache-2.0

use multithread::validate_all_threads_compute_add1;
use xlsynth::DslxConvertOptions;

// Show that the API works even if the Rust (caller) program is using an
// alternative allocator.
extern crate mimalloc;
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod multithread;

include! {env!("DSLX_SAMPLE_WITH_ENUM_DEF_PATH")} // generated by build.rs

// TODO(cdleary): 2024-12-02: Broken for the moment due to the type that has an
// extern-type member.
// include! {env!("DSLX_STRUCTURE_ZOO")} // generated by build.rs

/// Loads DSLX code from a relative path in the crate and invokes the
/// (no-parameter) function named `func` inside of it.
fn load_and_invoke(
    dslx_relpath: &str,
    func: &str,
) -> Result<xlsynth::IrValue, Box<dyn std::error::Error>> {
    let dslx_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(dslx_relpath);
    log::info!("reading DSLX program from: {:?}", dslx_path);
    let dslx = std::fs::read_to_string(&dslx_path)?;

    // Convert the DSLX to IR.
    let package =
        xlsynth::convert_dslx_to_ir(&dslx, dslx_path.as_path(), &DslxConvertOptions::default())?;
    let dslx_module_name = dslx_path.file_stem().unwrap().to_str().unwrap();

    // Determine what the mangled name is in the converted IR package.
    let mangled = xlsynth::mangle_dslx_name(dslx_module_name, func)?;

    // Extract the IR function from the package.
    let f = package.get_function(&mangled)?;
    log::info!(
        "function {} type: {}",
        f.get_name(),
        f.get_type()?.to_string()
    );
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
    let want = xlsynth::IrValue::parse_typed("bits[32]:42")?;
    assert_eq!(result, want);
    Ok(())
}

fn validate_use_popcount() -> Result<(), Box<dyn std::error::Error>> {
    let file = "src/sample_with_stdlib.x";
    let result = load_and_invoke(file, "use_popcount")?;
    let want = xlsynth::IrValue::parse_typed("bits[32]:11")?;
    assert_eq!(result, want);
    Ok(())
}

fn validate_fail() -> Result<(), Box<dyn std::error::Error>> {
    let file = "src/sample_with_fail.x";
    let result = load_and_invoke(file, "always_fail");
    match result {
        Ok(_) => Err(Box::new(xlsynth::XlsynthError(
            "expected function to fail".to_string(),
        ))),
        Err(e) => {
            if e.to_string()
                .contains("ABORTED: Assertion failure via fail!")
            {
                Ok(())
            } else {
                Err(Box::new(xlsynth::XlsynthError(format!(
                    "got unexpected error: {}",
                    e.to_string()
                ))))
            }
        }
    }
}

/// Validates that we can convert an enum value that we generated using the
/// bridge into an IR value.
fn validate_passing_generated_enum() -> Result<(), Box<dyn std::error::Error>> {
    let enum_c_value = sample_with_enum_def::MyEnum::C;
    let enum_c_value_into_ir: xlsynth::IrValue = enum_c_value.into();
    assert_eq!(enum_c_value_into_ir.to_string(), "bits[8]:255");
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

    validate_passing_generated_enum().expect("enum conversion should work");
    println!("enum conversion validation complete");

    validate_all_threads_compute_add1();
    println!("multi-threaded validation complete");
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
}
