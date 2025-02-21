// SPDX-License-Identifier: Apache-2.0

use xlsynth::*;

/// Demonstrates running functions that produce/accept array IR values.
#[test]
fn test_ir_interpret_array_values() {
    let path = std::path::Path::new("tests/function_zoo.x");
    let dslx = match std::fs::read_to_string(path) {
        Ok(dslx) => dslx,
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                panic!(
                    "Could not find file: {}; cwd: {}",
                    path.display(),
                    std::env::current_dir().unwrap().display()
                );
            } else {
                panic!("Error reading file: {}", e);
            }
        }
    };
    let result = convert_dslx_to_ir(&dslx, &path, &DslxConvertOptions::default()).unwrap();
    assert!(result.warnings.is_empty());
    let ir = result.ir;

    let make_u32x2_mangled = mangle_dslx_name("function_zoo", "make_u32x2").unwrap();
    let make_u32x2 = ir.get_function(&make_u32x2_mangled).unwrap();

    let array = make_u32x2
        .interpret(&[IrValue::u32(1), IrValue::u32(2)])
        .unwrap();
    assert_eq!(array.to_string(), "[bits[32]:1, bits[32]:2]");

    let sum_elements_2_mangled = mangle_dslx_name("function_zoo", "sum_elements_2").unwrap();
    let sum_elements_2 = ir.get_function(&sum_elements_2_mangled).unwrap();
    let result = sum_elements_2.interpret(&[array.clone()]).unwrap();
    assert_eq!(result.to_string(), "bits[32]:3");

    let make_u32_3x2_mangled = mangle_dslx_name("function_zoo", "make_u32_3x2").unwrap();
    let make_u32_3x2 = ir.get_function(&make_u32_3x2_mangled).unwrap();
    let result = make_u32_3x2
        .interpret(&[array.clone(), array.clone(), array.clone()])
        .unwrap();
    assert_eq!(
        result.to_string(),
        "[[bits[32]:1, bits[32]:2], [bits[32]:1, bits[32]:2], [bits[32]:1, bits[32]:2]]"
    );
}
