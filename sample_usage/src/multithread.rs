// SPDX-License-Identifier: Apache-2.0

use lazy_static::lazy_static;
///! Example of multi-threaded invocation of XLS functions.
use rayon::prelude::*;
use xlsynth::IrPackage;

fn load_package(cargo_relpath: &str) -> IrPackage {
    let filename = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(cargo_relpath);
    let dslx = std::fs::read_to_string(filename).expect("read_to_string failed");
    let package = xlsynth::convert_dslx_to_ir(&dslx).expect("convert_dslx_to_ir failed");
    package
}

lazy_static! {
    static ref SAMPLE_PACKAGE: xlsynth::IrPackage = load_package("src/sample.x");
}

// Do a lazy_static initialization of the function.
lazy_static! {
    // Load sample.x and expose the "add1" function so that threads can invoke it via the XLS
    // interpreter.
    static ref ADD1_FUNCTION: xlsynth::IrFunction = {
        let mangled = xlsynth::mangle_dslx_name("test_mod", "add1").expect("mangle_dslx_name failed");
        let function = SAMPLE_PACKAGE.get_function(&mangled).expect("get_function failed");
        assert_eq!(function.get_name(), mangled);
        function
    };
}

pub fn validate_all_threads_compute_add1() {
    // Use rayon to compute the "add1" function in parallel on every available core.
    let results: Vec<xlsynth::IrValue> = (0..num_cpus::get())
        .into_par_iter()
        .map(|i| {
            let i_ir = xlsynth::IrValue::parse_typed(&format!("bits[32]:{}", i)).unwrap();
            ADD1_FUNCTION.interpret(&[i_ir]).expect("interpret failed")
        })
        .collect();

    // Check that all the results are index+1.
    for (i, result) in results.iter().enumerate() {
        let want = xlsynth::IrValue::parse_typed(&format!("bits[32]:{}", i + 1)).unwrap();
        assert_eq!(*result, want);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_all_threads_compute_mol() {
        let _ = env_logger::try_init();
        validate_all_threads_compute_add1();
    }
}
