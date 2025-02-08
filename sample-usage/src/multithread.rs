// SPDX-License-Identifier: Apache-2.0

use lazy_static::lazy_static;
///! Example of multi-threaded invocation of XLS functions.
use rayon::prelude::*;
use xlsynth::DslxConvertOptions;
use xlsynth::IrPackage;
use xlsynth::IrValue;

fn load_package(cargo_relpath: &str) -> IrPackage {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(cargo_relpath);
    let dslx = std::fs::read_to_string(&path).expect("read_to_string failed");
    let result =
        xlsynth::convert_dslx_to_ir(&dslx, path.as_path(), &DslxConvertOptions::default())
            .expect("convert_dslx_to_ir failed");
    // Note: we're discarding warnings here.
    result.ir
}

// Do a lazy_static initialization of the function.
lazy_static! {
    // Load sample.x and expose the "add1" function so that threads can invoke it via the XLS
    // interpreter.
    static ref ADD1_FUNCTION: xlsynth::IrFunction = {
        let package = load_package("src/sample.x");
        let mangled = xlsynth::mangle_dslx_name("sample", "add1").expect("mangle_dslx_name failed");
        let function = package.get_function(&mangled).expect("get_function failed");
        assert_eq!(function.get_name(), mangled);
        function
    };
}

fn run_dslx_add1(x: u32) -> u32 {
    let x_ir = IrValue::u32(x);
    let result = ADD1_FUNCTION
        .interpret(&[x_ir])
        .expect("interpretation success");
    result.to_u32().unwrap()
}

pub fn validate_all_threads_compute_add1() {
    // Use rayon to compute the "add1" function in parallel on every available core.
    let results: Vec<u32> = (0..num_cpus::get() as u32)
        .into_par_iter()
        .map(|i| run_dslx_add1(i))
        .collect();

    // Check that all the results are index+1.
    for (i, result) in results.iter().enumerate() {
        let want = i as u32 + 1;
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
