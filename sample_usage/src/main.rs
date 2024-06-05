use xlsynth;

/// Validates the "meaning of life" value that comes from the `mol` function in `sample.x`.
fn validate_mol() -> Result<(), Box<dyn std::error::Error>> {
    let file = "src/sample.x";
    // Get the file relative to the cargo manifest.
    let filename = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(file);
    log::info!("reading DSLX program from: {:?}", filename);
    // Read the DSLX code from sample.x in this same directory.
    let dslx = std::fs::read_to_string(filename)?;
    let package = xlsynth::convert_dslx_to_ir(&dslx)?;
    let mangled = xlsynth::mangle_dslx_name("test_mod", "mol")?;
    let f = package.get_function(&mangled)?;
    let result = f.interpret(&[])?;
    let want = xlsynth::IrValue::parse_typed("bits[32]:42")?;
    assert_eq!(result, want);
    Ok(())
}

fn main() {
    let _ = env_logger::try_init();
    let result = validate_mol();
    println!("meaning-of-life validation result: {:?}", result);
}

#[test]
fn test_validate_mol() {
    let _ = env_logger::try_init();
    validate_mol().expect("validation should succeed");
}
