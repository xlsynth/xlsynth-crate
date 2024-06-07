// SPDX-License-Identifier: Apache-2.0

use xlsynth;

fn load_and_invoke(file: &str, func: &str) -> Result<xlsynth::IrValue, Box<dyn std::error::Error>> {
    let filename = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(file);
    log::info!("reading DSLX program from: {:?}", filename);
    let dslx = std::fs::read_to_string(filename)?;
    let package = xlsynth::convert_dslx_to_ir(&dslx)?;
    let mangled = xlsynth::mangle_dslx_name("test_mod", func)?;
    let f = package.get_function(&mangled)?;
    let result = f.interpret(&[])?;
    Ok(result)
}

/// Validates the "meaning of life" value that comes from the `mol` function in `sample.x`.
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

fn main() {
    let _ = env_logger::try_init();
    let mol_result = validate_mol();
    let popcount_result = validate_use_popcount();
    println!(
        "meaning-of-life validation result: {:?} {:?}",
        mol_result, popcount_result
    );
}

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
